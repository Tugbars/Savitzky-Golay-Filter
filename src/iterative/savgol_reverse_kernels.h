/**
 * @file savgol_reverse_kernels.h
 * @brief Optimized SIMD kernels for reverse-order dot products
 *
 * @details
 * Specialized kernels for computing dot products with reversed data access:
 *   result = weights[0]*data[N-1] + weights[1]*data[N-2] + ... + weights[N-1]*data[0]
 *
 * Used for leading edge processing in Savitzky-Golay filter.
 *
 * Strategy:
 * 1. Load forward-ordered weights
 * 2. Load data and reverse in-register using permute instructions
 * 3. FMA as normal
 * 4. Much faster than scalar or gather for stride=-1
 *
 * @author Tugbars Heptaskin
 * @date 2025-10-24
 */

#ifndef SAVGOL_REVERSE_KERNELS_H
#define SAVGOL_REVERSE_KERNELS_H

#include "savgol_simd_ops.h"

//==============================================================================
// AVX-512: Reverse Dot Product (16-wide)
//==============================================================================

#ifdef HAS_AVX512
/**
 * @brief AVX-512 reverse dot product kernel
 *
 * Computes: sum = Σ(weights[i] * data[N-1-i]) for i = 0..windowSize-1
 *
 * @param weights Forward-ordered weight array
 * @param data_end Pointer to LAST element of data (data[N-1])
 * @param windowSize Number of elements
 * @return Dot product result
 */
static inline float savgol_reverse_dot_avx512(
    const float *weights,
    const float *data_end,
    size_t windowSize)
{
    __m512 sum0 = VSETZERO_PS_512();
    __m512 sum1 = VSETZERO_PS_512();
    __m512 sum2 = VSETZERO_PS_512();
    __m512 sum3 = VSETZERO_PS_512();
    
    // Index vector for reversing: [15, 14, 13, ..., 2, 1, 0]
    const __m512i reverse_idx = _mm512_set_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    
    size_t i = 0;
    
    // Process 64 elements (4x16) at a time
    for (; i + 64 <= windowSize; i += 64) {
        // First 16 elements
        __m512 w0 = VLOADU_PS_512(weights + i);
        __m512 d0 = VLOADU_PS_512(data_end - i - 15);  // Load from end backwards
        d0 = _mm512_permutexvar_ps(reverse_idx, d0);   // Reverse in-register
        sum0 = VFMADD_PS_512(w0, d0, sum0);
        
        // Second 16 elements
        __m512 w1 = VLOADU_PS_512(weights + i + 16);
        __m512 d1 = VLOADU_PS_512(data_end - i - 31);
        d1 = _mm512_permutexvar_ps(reverse_idx, d1);
        sum1 = VFMADD_PS_512(w1, d1, sum1);
        
        // Third 16 elements
        __m512 w2 = VLOADU_PS_512(weights + i + 32);
        __m512 d2 = VLOADU_PS_512(data_end - i - 47);
        d2 = _mm512_permutexvar_ps(reverse_idx, d2);
        sum2 = VFMADD_PS_512(w2, d2, sum2);
        
        // Fourth 16 elements
        __m512 w3 = VLOADU_PS_512(weights + i + 48);
        __m512 d3 = VLOADU_PS_512(data_end - i - 63);
        d3 = _mm512_permutexvar_ps(reverse_idx, d3);
        sum3 = VFMADD_PS_512(w3, d3, sum3);
    }
    
    // Process remaining 16-element chunks
    for (; i + 16 <= windowSize; i += 16) {
        __m512 w = VLOADU_PS_512(weights + i);
        __m512 d = VLOADU_PS_512(data_end - i - 15);
        d = _mm512_permutexvar_ps(reverse_idx, d);
        sum0 = VFMADD_PS_512(w, d, sum0);
    }
    
    // Combine accumulators
    sum0 = VADD_PS_512(sum0, sum1);
    sum2 = VADD_PS_512(sum2, sum3);
    sum0 = VADD_PS_512(sum0, sum2);
    
    // Horizontal sum
    float result = hsum_ps_512(sum0);
    
    // Scalar remainder
    for (; i < windowSize; i++) {
        result += weights[i] * data_end[-i];
    }
    
    return result;
}
#endif // HAS_AVX512

//==============================================================================
// AVX2: Reverse Dot Product (8-wide)
//==============================================================================

#ifdef HAS_AVX2
/**
 * @brief AVX2 reverse dot product kernel
 *
 * @param weights Forward-ordered weight array
 * @param data_end Pointer to LAST element of data
 * @param windowSize Number of elements
 * @return Dot product result
 */
static inline float savgol_reverse_dot_avx2(
    const float *weights,
    const float *data_end,
    size_t windowSize)
{
    __m256 sum0 = VSETZERO_PS_256();
    __m256 sum1 = VSETZERO_PS_256();
    __m256 sum2 = VSETZERO_PS_256();
    __m256 sum3 = VSETZERO_PS_256();
    
    size_t i = 0;
    
    // Process 32 elements (4x8) at a time
    for (; i + 32 <= windowSize; i += 32) {
        // First 8 elements
        __m256 w0 = VLOADU_PS_256(weights + i);
        __m256 d0 = VLOADU_PS_256(data_end - i - 7);
        // Reverse 8-element vector: swap 128-bit lanes, then reverse within lanes
        d0 = _mm256_permute2f128_ps(d0, d0, 0x01);  // Swap high/low 128 bits
        d0 = _mm256_shuffle_ps(d0, d0, _MM_SHUFFLE(0, 1, 2, 3));  // Reverse within lanes
        sum0 = VFMADD_PS_256(w0, d0, sum0);
        
        // Second 8 elements
        __m256 w1 = VLOADU_PS_256(weights + i + 8);
        __m256 d1 = VLOADU_PS_256(data_end - i - 15);
        d1 = _mm256_permute2f128_ps(d1, d1, 0x01);
        d1 = _mm256_shuffle_ps(d1, d1, _MM_SHUFFLE(0, 1, 2, 3));
        sum1 = VFMADD_PS_256(w1, d1, sum1);
        
        // Third 8 elements
        __m256 w2 = VLOADU_PS_256(weights + i + 16);
        __m256 d2 = VLOADU_PS_256(data_end - i - 23);
        d2 = _mm256_permute2f128_ps(d2, d2, 0x01);
        d2 = _mm256_shuffle_ps(d2, d2, _MM_SHUFFLE(0, 1, 2, 3));
        sum2 = VFMADD_PS_256(w2, d2, sum2);
        
        // Fourth 8 elements
        __m256 w3 = VLOADU_PS_256(weights + i + 24);
        __m256 d3 = VLOADU_PS_256(data_end - i - 31);
        d3 = _mm256_permute2f128_ps(d3, d3, 0x01);
        d3 = _mm256_shuffle_ps(d3, d3, _MM_SHUFFLE(0, 1, 2, 3));
        sum3 = VFMADD_PS_256(w3, d3, sum3);
    }
    
    // Process remaining 8-element chunks
    for (; i + 8 <= windowSize; i += 8) {
        __m256 w = VLOADU_PS_256(weights + i);
        __m256 d = VLOADU_PS_256(data_end - i - 7);
        d = _mm256_permute2f128_ps(d, d, 0x01);
        d = _mm256_shuffle_ps(d, d, _MM_SHUFFLE(0, 1, 2, 3));
        sum0 = VFMADD_PS_256(w, d, sum0);
    }
    
    // Combine accumulators
    sum0 = VADD_PS_256(sum0, sum1);
    sum2 = VADD_PS_256(sum2, sum3);
    sum0 = VADD_PS_256(sum0, sum2);
    
    // Horizontal sum
    float result = hsum_ps_256(sum0);
    
    // Scalar remainder
    for (; i < windowSize; i++) {
        result += weights[i] * data_end[-i];
    }
    
    return result;
}
#endif // HAS_AVX2

//==============================================================================
// SSE2: Reverse Dot Product (4-wide)
//==============================================================================

#ifdef HAS_SSE2
/**
 * @brief SSE2 reverse dot product kernel
 *
 * @param weights Forward-ordered weight array
 * @param data_end Pointer to LAST element of data
 * @param windowSize Number of elements
 * @return Dot product result
 */
static inline float savgol_reverse_dot_sse2(
    const float *weights,
    const float *data_end,
    size_t windowSize)
{
    __m128 sum_vec = VSETZERO_PS_128();
    size_t i = 0;
    
    // Process 4 elements at a time
    for (; i + 4 <= windowSize; i += 4) {
        __m128 w = VLOADU_PS_128(weights + i);
        __m128 d = VLOADU_PS_128(data_end - i - 3);
        
        // Reverse 4-element vector: [3,2,1,0] -> [0,1,2,3]
        d = _mm_shuffle_ps(d, d, _MM_SHUFFLE(0, 1, 2, 3));
        
        sum_vec = VFMADD_PS_128(w, d, sum_vec);
    }
    
    // Horizontal sum
    float result = hsum_ps_128(sum_vec);
    
    // Scalar remainder
    for (; i < windowSize; i++) {
        result += weights[i] * data_end[-i];
    }
    
    return result;
}
#endif // HAS_SSE2

//==============================================================================
// Dispatch Macro: Select Best Reverse Kernel
//==============================================================================

/**
 * @brief Dispatch to best available reverse dot product implementation
 *
 * Usage:
 *   float result = SAVGOL_REVERSE_DOT_PRODUCT(weights, &data[N-1], N);
 *
 * This computes: weights[0]*data[N-1] + weights[1]*data[N-2] + ... + weights[N-1]*data[0]
 */
#define SAVGOL_REVERSE_DOT_PRODUCT(weights, data_end, windowSize) \
    ((windowSize) < SAVGOL_SCALAR_THRESHOLD ? \
        _savgol_reverse_dot_scalar(weights, data_end, windowSize) : \
        _SAVGOL_REVERSE_DOT_DISPATCH(weights, data_end, windowSize))

// Compile-time dispatch
#ifdef __AVX512F__
    #define _SAVGOL_REVERSE_DOT_DISPATCH(w, d, s) savgol_reverse_dot_avx512(w, d, s)
#elif defined(__AVX2__)
    #define _SAVGOL_REVERSE_DOT_DISPATCH(w, d, s) savgol_reverse_dot_avx2(w, d, s)
#elif defined(__SSE2__)
    #define _SAVGOL_REVERSE_DOT_DISPATCH(w, d, s) savgol_reverse_dot_sse2(w, d, s)
#else
    #define _SAVGOL_REVERSE_DOT_DISPATCH(w, d, s) _savgol_reverse_dot_scalar(w, d, s)
#endif

//==============================================================================
// Scalar Fallback (for small windows or non-SIMD builds)
//==============================================================================

/**
 * @brief Scalar reverse dot product with 4-way parallel accumulation
 */
static inline float _savgol_reverse_dot_scalar(
    const float *weights,
    const float *data_end,
    size_t windowSize)
{
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    size_t i = 0;
    
    // 4-way unrolled loop
    for (; i + 4 <= windowSize; i += 4) {
        sum0 += weights[i]     * data_end[-(i)];
        sum1 += weights[i + 1] * data_end[-(i + 1)];
        sum2 += weights[i + 2] * data_end[-(i + 2)];
        sum3 += weights[i + 3] * data_end[-(i + 3)];
    }
    
    // Remainder
    for (; i < windowSize; i++) {
        sum0 += weights[i] * data_end[-i];
    }
    
    return (sum0 + sum1) + (sum2 + sum3);
}

#endif // SAVGOL_REVERSE_KERNELS_H