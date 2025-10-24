/**
 * @file savgol_simd_ops.h
 * @brief SIMD operation wrappers for Savitzky-Golay filter
 * 
 * @details
 * Provides unified SIMD operation macros with automatic FMA fallback.
 * Similar to the radix-4 FFT architecture but for float operations.
 * 
 * Architecture support:
 * - AVX-512: 16-wide float vectors (FMA always available)
 * - AVX2: 8-wide float vectors (FMA-aware fallback)
 * - SSE2: 4-wide float vectors (no FMA)
 * - Scalar: 1-wide operations
 * 
 * @author Tugbars Heptaskin
 * @date 2025-10-24
 */

#ifndef SAVGOL_SIMD_OPS_H
#define SAVGOL_SIMD_OPS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Include appropriate SIMD headers
#ifdef __AVX512F__
    #include <immintrin.h>
    #define HAS_AVX512
#endif

#ifdef __AVX2__
    #include <immintrin.h>
    #define HAS_AVX2
#endif

#ifdef __SSE2__
    #include <emmintrin.h>
    #define HAS_SSE2
#endif

//==============================================================================
// CONFIGURATION: Architecture-Specific Parameters
//==============================================================================

#ifdef __AVX512F__
    #define SAVGOL_ALIGNMENT 64          // 512-bit = 64 bytes
    #define SAVGOL_VECTOR_WIDTH 16       // 16 floats per vector
    #define SAVGOL_STREAM_THRESHOLD 4096 // Enable streaming stores for large arrays
    #define SAVGOL_PREFETCH_DISTANCE 64  // Prefetch 64 elements ahead
#elif defined(__AVX2__)
    #define SAVGOL_ALIGNMENT 32          // 256-bit = 32 bytes
    #define SAVGOL_VECTOR_WIDTH 8        // 8 floats per vector
    #define SAVGOL_STREAM_THRESHOLD 4096
    #define SAVGOL_PREFETCH_DISTANCE 32
#elif defined(__SSE2__)
    #define SAVGOL_ALIGNMENT 16          // 128-bit = 16 bytes
    #define SAVGOL_VECTOR_WIDTH 4        // 4 floats per vector
    #define SAVGOL_STREAM_THRESHOLD 8192 // Higher threshold for SSE2
    #define SAVGOL_PREFETCH_DISTANCE 16
#else
    #define SAVGOL_ALIGNMENT 8           // Scalar alignment
    #define SAVGOL_VECTOR_WIDTH 1
    #define SAVGOL_STREAM_THRESHOLD SIZE_MAX
    #define SAVGOL_PREFETCH_DISTANCE 0
#endif

// Window size threshold: use scalar for small windows
#define SAVGOL_SCALAR_THRESHOLD 12

//==============================================================================
// AVX-512 Operations (16-wide float)
//==============================================================================

#ifdef HAS_AVX512

// FMA operations (always available on AVX-512)
#define VFMADD_PS_512(a, b, c)  _mm512_fmadd_ps(a, b, c)   // a*b + c
#define VFMSUB_PS_512(a, b, c)  _mm512_fmsub_ps(a, b, c)   // a*b - c
#define VFNMADD_PS_512(a, b, c) _mm512_fnmadd_ps(a, b, c)  // -(a*b) + c

// Basic arithmetic
#define VADD_PS_512(a, b)       _mm512_add_ps(a, b)
#define VSUB_PS_512(a, b)       _mm512_sub_ps(a, b)
#define VMUL_PS_512(a, b)       _mm512_mul_ps(a, b)
#define VSETZERO_PS_512()       _mm512_setzero_ps()

// Memory operations - aligned
#define VLOAD_PS_512(ptr)       _mm512_load_ps(ptr)
#define VSTORE_PS_512(ptr, v)   _mm512_store_ps(ptr, v)

// Memory operations - unaligned
#define VLOADU_PS_512(ptr)      _mm512_loadu_ps(ptr)
#define VSTOREU_PS_512(ptr, v)  _mm512_storeu_ps(ptr, v)

// Streaming stores (non-temporal)
#define VSTREAM_PS_512(ptr, v)  _mm512_stream_ps(ptr, v)

// Masked operations (AVX-512 exclusive feature)
#define VMASK_LOAD_PS_512(mask, ptr)       _mm512_maskz_loadu_ps(mask, ptr)
#define VMASK_STORE_PS_512(ptr, mask, v)   _mm512_mask_storeu_ps(ptr, mask, v)

// Horizontal sum (reduce 16 floats to scalar)
static inline float hsum_ps_512(__m512 v) {
    // Reduce 512 -> 256
    __m256 lo256 = _mm512_castps512_ps256(v);
    __m256 hi256 = _mm512_extractf32x8_ps(v, 1);
    __m256 sum256 = _mm256_add_ps(lo256, hi256);
    
    // Reduce 256 -> 128
    __m128 lo128 = _mm256_castps256_ps128(sum256);
    __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    
    // Reduce 128 -> 64: [3,2,1,0] + [1,0,3,2]
    __m128 shuf = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(sum128, shuf);
    
    // Reduce 64 -> 32: [1,0,x,x] + [x,x,1,0]
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    
    return _mm_cvtss_f32(sums);
}

// Prefetch hints
#define VPREFETCH_512(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T0)

#endif // HAS_AVX512

//==============================================================================
// AVX2 Operations (8-wide float)
//==============================================================================

#ifdef HAS_AVX2

// FMA operations (with fallback for non-FMA AVX2)
#if defined(__FMA__)
    #define VFMADD_PS_256(a, b, c)  _mm256_fmadd_ps(a, b, c)
    #define VFMSUB_PS_256(a, b, c)  _mm256_fmsub_ps(a, b, c)
    #define VFNMADD_PS_256(a, b, c) _mm256_fnmadd_ps(a, b, c)
#else
    // Non-FMA fallback: separate multiply and add/sub
    #define VFMADD_PS_256(a, b, c)  _mm256_add_ps(_mm256_mul_ps(a, b), c)
    #define VFMSUB_PS_256(a, b, c)  _mm256_sub_ps(_mm256_mul_ps(a, b), c)
    #define VFNMADD_PS_256(a, b, c) _mm256_sub_ps(c, _mm256_mul_ps(a, b))
#endif

// Basic arithmetic
#define VADD_PS_256(a, b)       _mm256_add_ps(a, b)
#define VSUB_PS_256(a, b)       _mm256_sub_ps(a, b)
#define VMUL_PS_256(a, b)       _mm256_mul_ps(a, b)
#define VSETZERO_PS_256()       _mm256_setzero_ps()

// Memory operations - aligned
#define VLOAD_PS_256(ptr)       _mm256_load_ps(ptr)
#define VSTORE_PS_256(ptr, v)   _mm256_store_ps(ptr, v)

// Memory operations - unaligned
#define VLOADU_PS_256(ptr)      _mm256_loadu_ps(ptr)
#define VSTOREU_PS_256(ptr, v)  _mm256_storeu_ps(ptr, v)

// Streaming stores
#define VSTREAM_PS_256(ptr, v)  _mm256_stream_ps(ptr, v)

// Horizontal sum (reduce 8 floats to scalar)
static inline float hsum_ps_256(__m256 v) {
    // Reduce 256 -> 128
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    
    // Reduce 128 -> 64
    __m128 shuf = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(sum128, shuf);
    
    // Reduce 64 -> 32
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    
    return _mm_cvtss_f32(sums);
}

// Prefetch hints
#define VPREFETCH_256(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T0)

#endif // HAS_AVX2

//==============================================================================
// SSE2 Operations (4-wide float)
//==============================================================================

#ifdef HAS_SSE2

// FMA not available on SSE2 - always use separate multiply and add/sub
#define VFMADD_PS_128(a, b, c)  _mm_add_ps(_mm_mul_ps(a, b), c)
#define VFMSUB_PS_128(a, b, c)  _mm_sub_ps(_mm_mul_ps(a, b), c)
#define VFNMADD_PS_128(a, b, c) _mm_sub_ps(c, _mm_mul_ps(a, b))

// Basic arithmetic
#define VADD_PS_128(a, b)       _mm_add_ps(a, b)
#define VSUB_PS_128(a, b)       _mm_sub_ps(a, b)
#define VMUL_PS_128(a, b)       _mm_mul_ps(a, b)
#define VSETZERO_PS_128()       _mm_setzero_ps()

// Memory operations - aligned
#define VLOAD_PS_128(ptr)       _mm_load_ps(ptr)
#define VSTORE_PS_128(ptr, v)   _mm_store_ps(ptr, v)

// Memory operations - unaligned
#define VLOADU_PS_128(ptr)      _mm_loadu_ps(ptr)
#define VSTOREU_PS_128(ptr, v)  _mm_storeu_ps(ptr, v)

// SSE2 doesn't have streaming store for single precision
// Use regular store or use _mm_stream_si128 with casting
#define VSTREAM_PS_128(ptr, v)  _mm_store_ps(ptr, v)

// Horizontal sum (reduce 4 floats to scalar)
static inline float hsum_ps_128(__m128 v) {
    // Reduce 128 -> 64: [3,2,1,0] + [1,0,3,2]
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    
    // Reduce 64 -> 32: [1,0,x,x] + [x,x,1,0]
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    
    return _mm_cvtss_f32(sums);
}

// Prefetch hints
#define VPREFETCH_128(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T0)

#endif // HAS_SSE2

//==============================================================================
// Memory Management: Aligned Allocation
//==============================================================================

#include <stdlib.h>

/**
 * @brief Allocate aligned memory for SIMD operations
 * 
 * Similar to fftw_malloc - uses architecture-appropriate alignment.
 */
static inline float* savgol_alloc_aligned(size_t count) {
    void *ptr = NULL;
    
#if defined(_MSC_VER)
    // MSVC
    ptr = _aligned_malloc(count * sizeof(float), SAVGOL_ALIGNMENT);
#elif defined(__MINGW32__) || defined(__MINGW64__)
    // MinGW
    ptr = __mingw_aligned_malloc(count * sizeof(float), SAVGOL_ALIGNMENT);
#else
    // POSIX (Linux, macOS)
    if (posix_memalign(&ptr, SAVGOL_ALIGNMENT, count * sizeof(float)) != 0) {
        ptr = NULL;
    }
#endif
    
    return (float*)ptr;
}

/**
 * @brief Free aligned memory
 */
static inline void savgol_free_aligned(float *ptr) {
    if (ptr) {
#if defined(_MSC_VER)
        _aligned_free(ptr);
#elif defined(__MINGW32__) || defined(__MINGW64__)
        __mingw_aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

#endif // SAVGOL_SIMD_OPS_H