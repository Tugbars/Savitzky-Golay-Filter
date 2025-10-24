/**
 * @file savgol_kernels.h
 * @brief Vectorized convolution kernels for Savitzky-Golay filter
 *
 * @details
 * Core computational kernels using SIMD instructions.
 * Follows the radix-4 FFT macro-based design pattern.
 *
 * Three kernel types:
 * 1. DOT_PRODUCT: Single dot product (weights · data)
 * 2. CENTER_REGION: Vectorized convolution for center points
 * 3. EDGE_REGION: Specialized handling for leading/trailing edges
 *
 * @author Tugbars Heptaskin
 * @date 2025-10-24
 */

#ifndef SAVGOL_KERNELS_H
#define SAVGOL_KERNELS_H

#include "savgol_simd_ops.h"
#include <string.h>

//==============================================================================
// DOT PRODUCT KERNELS: weights · data
//==============================================================================

#ifdef HAS_AVX512
/**
 * @brief AVX-512 dot product kernel (16-wide FMA)
 *
 * Computes: sum = Σ(weights[i] * data[i]) for i = 0..windowSize-1
 *
 * Strategy:
 * - Process 16 elements per iteration
 * - Use FMA for efficiency (2 ops per cycle)
 * - Software prefetch for next iteration
 * - Single horizontal sum at the end
 *
 * @param weights Pointer to weight array (aligned)
 * @param data Pointer to data array (may be unaligned)
 * @param windowSize Number of elements in window
 * @return Dot product result (scalar)
 */
static inline float savgol_dot_product_avx512(
    const float *weights,
    const float *data,
    size_t windowSize)
{
    // Four accumulators to reduce dependency chains
    __m512 sum0 = VSETZERO_PS_512();
    __m512 sum1 = VSETZERO_PS_512();
    __m512 sum2 = VSETZERO_PS_512();
    __m512 sum3 = VSETZERO_PS_512();
    
    size_t i = 0;
    
    // Unrolled loop: process 64 elements (4x16) at a time
    for (; i + 64 <= windowSize; i += 64) {
        // Prefetch ahead
        VPREFETCH_512(data + i + SAVGOL_PREFETCH_DISTANCE);
        
        // Load and FMA - unrolled 4x
        __m512 w0 = VLOADU_PS_512(weights + i);
        __m512 d0 = VLOADU_PS_512(data + i);
        sum0 = VFMADD_PS_512(w0, d0, sum0);
        
        __m512 w1 = VLOADU_PS_512(weights + i + 16);
        __m512 d1 = VLOADU_PS_512(data + i + 16);
        sum1 = VFMADD_PS_512(w1, d1, sum1);
        
        __m512 w2 = VLOADU_PS_512(weights + i + 32);
        __m512 d2 = VLOADU_PS_512(data + i + 32);
        sum2 = VFMADD_PS_512(w2, d2, sum2);
        
        __m512 w3 = VLOADU_PS_512(weights + i + 48);
        __m512 d3 = VLOADU_PS_512(data + i + 48);
        sum3 = VFMADD_PS_512(w3, d3, sum3);
    }
    
    // Process remaining 16-element chunks
    for (; i + 16 <= windowSize; i += 16) {
        __m512 w = VLOADU_PS_512(weights + i);
        __m512 d = VLOADU_PS_512(data + i);
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
        result += weights[i] * data[i];
    }
    
    return result;
}
#endif

/**
 * @brief AVX-512 dot product with strided data access (for reverse traversal)
 *
 * Used for leading edge where we access data[0], data[-1], data[-2], ...
 *
 * @param weights Pointer to weight array
 * @param data_end Pointer to END of data array
 * @param windowSize Number of elements
 * @param stride Stride between elements (typically -1 for reverse)
 */
#define SAVGOL_DOT_PRODUCT_AVX512_STRIDED(weights, data_end, windowSize, stride) \
    ({                                                                           \
        __m512 acc = VSETZERO_PS_512();                                          \
        int i = 0;                                                               \
        const int vec_end = (windowSize) & ~15;                                  \
                                                                                 \
        /* For strided access, we need to gather elements manually */            \
        /* AVX-512 has gather instructions, but they're often slower */          \
        /* than scalar for small strides due to high latency */                  \
        float scalar_sum = 0.0f;                                                 \
        for (i = 0; i < (windowSize); i++)                                       \
        {                                                                        \
            scalar_sum += weights[i] * data_end[i * (stride)];                   \
        }                                                                        \
        scalar_sum;                                                              \
    })

#endif // HAS_AVX512

//==============================================================================
// AVX2 Dot Product (8-wide)
//==============================================================================

#ifdef HAS_AVX2
static inline float savgol_dot_product_avx2(
    const float *weights,
    const float *data,
    size_t windowSize)
{
    // Four accumulators to reduce dependency chains
    __m256 sum0 = VSETZERO_PS_256();
    __m256 sum1 = VSETZERO_PS_256();
    __m256 sum2 = VSETZERO_PS_256();
    __m256 sum3 = VSETZERO_PS_256();
    
    size_t i = 0;
    
    // Unrolled loop: process 32 elements (4x8) at a time
    for (; i + 32 <= windowSize; i += 32) {
        // Prefetch ahead
        VPREFETCH_256(data + i + SAVGOL_PREFETCH_DISTANCE);
        
        // Load and FMA - unrolled 4x
        __m256 w0 = VLOADU_PS_256(weights + i);
        __m256 d0 = VLOADU_PS_256(data + i);
        sum0 = VFMADD_PS_256(w0, d0, sum0);
        
        __m256 w1 = VLOADU_PS_256(weights + i + 8);
        __m256 d1 = VLOADU_PS_256(data + i + 8);
        sum1 = VFMADD_PS_256(w1, d1, sum1);
        
        __m256 w2 = VLOADU_PS_256(weights + i + 16);
        __m256 d2 = VLOADU_PS_256(data + i + 16);
        sum2 = VFMADD_PS_256(w2, d2, sum2);
        
        __m256 w3 = VLOADU_PS_256(weights + i + 24);
        __m256 d3 = VLOADU_PS_256(data + i + 24);
        sum3 = VFMADD_PS_256(w3, d3, sum3);
    }
    
    // Process remaining 8-element chunks
    for (; i + 8 <= windowSize; i += 8) {
        __m256 w = VLOADU_PS_256(weights + i);
        __m256 d = VLOADU_PS_256(data + i);
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
        result += weights[i] * data[i];
    }
    
    return result;
}


#define SAVGOL_DOT_PRODUCT_AVX2_STRIDED(weights, data_end, windowSize, stride) \
    ({                                                                         \
        float scalar_sum = 0.0f;                                               \
        for (int i = 0; i < (windowSize); i++)                                 \
        {                                                                      \
            scalar_sum += weights[i] * data_end[i * (stride)];                 \
        }                                                                      \
        scalar_sum;                                                            \
    })

#endif // HAS_AVX2

#ifdef HAS_SSE2
/**
 * @brief SSE2 dot product kernel (4-wide mul+add)
 */
#ifdef HAS_SSE2
static inline float savgol_dot_product_sse2(
    const float *weights,
    const float *data,
    size_t windowSize)
{
    __m128 sum_vec = VSETZERO_PS_128();
    size_t i = 0;
    
    // Vectorized loop: process 4 elements at a time
    for (; i + 4 <= windowSize; i += 4) {
        // Prefetch ahead
        if (i + SAVGOL_PREFETCH_DISTANCE + 4 <= windowSize) {
            VPREFETCH_128(data + i + SAVGOL_PREFETCH_DISTANCE);
        }
        
        __m128 w = VLOADU_PS_128(weights + i);
        __m128 d = VLOADU_PS_128(data + i);
        sum_vec = VFMADD_PS_128(w, d, sum_vec);
    }
    
    // Horizontal sum
    float result = hsum_ps_128(sum_vec);
    
    // Scalar remainder
    for (; i < windowSize; i++) {
        result += weights[i] * data[i];
    }
    
    return result;
}
#endif

#define SAVGOL_DOT_PRODUCT_SSE2_STRIDED(weights, data_end, windowSize, stride) \
    ({                                                                         \
        float scalar_sum = 0.0f;                                               \
        for (int i = 0; i < (windowSize); i++)                                 \
        {                                                                      \
            scalar_sum += weights[i] * data_end[i * (stride)];                 \
        }                                                                      \
        scalar_sum;                                                            \
    })

#endif // HAS_SSE2

/**
 * @brief Scalar dot product with 4-way parallel accumulation chains
 *
 * Optimized scalar implementation with Duff's device remainder handling
 * and 4 independent accumulation chains for instruction-level parallelism.
 */
static inline float savgol_dot_product_scalar(
    const float *weights,
    const float *data,
    size_t windowSize)
{
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    const float *w_ptr = weights;
    const float *d_ptr = data;
    
    // Handle remainder (windowSize % 4) using Duff's device
    int remainder = windowSize & 3;
    switch (remainder)
    {
    case 3:
        sum0 += w_ptr[0] * d_ptr[0];
        sum1 += w_ptr[1] * d_ptr[1];
        sum2 += w_ptr[2] * d_ptr[2];
        w_ptr += 3;
        d_ptr += 3;
        break;
    case 2:
        sum0 += w_ptr[0] * d_ptr[0];
        sum1 += w_ptr[1] * d_ptr[1];
        w_ptr += 2;
        d_ptr += 2;
        break;
    case 1:
        sum0 += w_ptr[0] * d_ptr[0];
        w_ptr += 1;
        d_ptr += 1;
        break;
    case 0:
        break;
    }
    
    // Main 4-way unrolled loop
    int main_iters = (windowSize - remainder) / 4;
    for (int k = 0; k < main_iters; ++k)
    {
        sum0 += w_ptr[0] * d_ptr[0]; // Chain 0
        sum1 += w_ptr[1] * d_ptr[1]; // Chain 1 (independent!)
        sum2 += w_ptr[2] * d_ptr[2]; // Chain 2 (independent!)
        sum3 += w_ptr[3] * d_ptr[3]; // Chain 3 (independent!)
        w_ptr += 4;
        d_ptr += 4;
    }
    
    // Pairwise reduction for numerical precision
    return (sum0 + sum1) + (sum2 + sum3);
}

#define SAVGOL_DOT_PRODUCT_SCALAR_STRIDED(weights, data_end, windowSize, stride) \
    ({                                                                           \
        float scalar_sum = 0.0f;                                                 \
        for (int i = 0; i < (windowSize); i++)                                   \
        {                                                                        \
            scalar_sum += weights[i] * data_end[i * (stride)];                   \
        }                                                                        \
        scalar_sum;                                                              \
    })

//==============================================================================
// DISPATCH MACRO: Select Optimal Kernel
//==============================================================================

/**
 * @brief Automatic dispatch to best available SIMD implementation
 *
 * Dispatch hierarchy:
 * 1. If windowSize < 12: Use scalar 4-chain (lowest overhead)
 * 2. Else: Dispatch to best available SIMD at compile-time
 *    - AVX-512: 16-wide vectorization
 *    - AVX2:     8-wide vectorization
 *    - SSE2:     4-wide vectorization
 *    - Scalar:   4-chain fallback
 */
#define SAVGOL_DOT_PRODUCT(weights, data, windowSize) \
    ((windowSize) < SAVGOL_SCALAR_THRESHOLD ? \
        savgol_dot_product_scalar(weights, data, windowSize) : \
        _SAVGOL_DOT_PRODUCT_DISPATCH(weights, data, windowSize))

// Compile-time architecture dispatch (resolved at compile-time, zero overhead)
#ifdef __AVX512F__
    #define _SAVGOL_DOT_PRODUCT_DISPATCH(w, d, s) savgol_dot_product_avx512(w, d, s)
#elif defined(__AVX2__)
    #define _SAVGOL_DOT_PRODUCT_DISPATCH(w, d, s) savgol_dot_product_avx2(w, d, s)
#elif defined(__SSE2__)
    #define _SAVGOL_DOT_PRODUCT_DISPATCH(w, d, s) savgol_dot_product_sse2(w, d, s)
#else
    #define _SAVGOL_DOT_PRODUCT_DISPATCH(w, d, s) savgol_dot_product_scalar(w, d, s)
#endif