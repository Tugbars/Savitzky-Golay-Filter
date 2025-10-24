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
#define SAVGOL_DOT_PRODUCT_AVX512(weights, data, windowSize)                    \
    ({                                                                           \
        __m512 acc = VSETZERO_PS_512();                                          \
        int i = 0;                                                               \
        const int vec_end = (windowSize) & ~15;  /* Round down to multiple of 16 */ \
                                                                                 \
        /* Main vectorized loop: 16 elements per iteration */                   \
        for (; i < vec_end; i += 16) {                                           \
            /* Prefetch next iteration */                                        \
            if (i + SAVGOL_PREFETCH_DISTANCE < (windowSize)) {                  \
                VPREFETCH_512(&weights[i + SAVGOL_PREFETCH_DISTANCE]);          \
                VPREFETCH_512(&data[i + SAVGOL_PREFETCH_DISTANCE]);             \
            }                                                                    \
                                                                                 \
            __m512 w = VLOADU_PS_512(&weights[i]);                               \
            __m512 d = VLOADU_PS_512(&data[i]);                                  \
            acc = VFMADD_PS_512(w, d, acc);  /* acc += w * d */                  \
        }                                                                        \
                                                                                 \
        /* Scalar cleanup for remaining elements */                             \
        float scalar_sum = hsum_ps_512(acc);                                     \
        for (; i < (windowSize); i++) {                                          \
            scalar_sum += weights[i] * data[i];                                  \
        }                                                                        \
        scalar_sum;                                                              \
    })

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
    ({                                                                            \
        __m512 acc = VSETZERO_PS_512();                                           \
        int i = 0;                                                                \
        const int vec_end = (windowSize) & ~15;                                   \
                                                                                  \
        /* For strided access, we need to gather elements manually */            \
        /* AVX-512 has gather instructions, but they're often slower */          \
        /* than scalar for small strides due to high latency */                  \
        float scalar_sum = 0.0f;                                                  \
        for (i = 0; i < (windowSize); i++) {                                      \
            scalar_sum += weights[i] * data_end[i * (stride)];                    \
        }                                                                         \
        scalar_sum;                                                               \
    })

#endif // HAS_AVX512

#ifdef HAS_AVX2
/**
 * @brief AVX2 dot product kernel (8-wide FMA or mul+add)
 */
#define SAVGOL_DOT_PRODUCT_AVX2(weights, data, windowSize)                      \
    ({                                                                           \
        __m256 acc = VSETZERO_PS_256();                                          \
        int i = 0;                                                               \
        const int vec_end = (windowSize) & ~7;  /* Multiple of 8 */             \
                                                                                 \
        /* Main vectorized loop: 8 elements per iteration */                    \
        for (; i < vec_end; i += 8) {                                            \
            if (i + SAVGOL_PREFETCH_DISTANCE < (windowSize)) {                  \
                VPREFETCH_256(&weights[i + SAVGOL_PREFETCH_DISTANCE]);          \
                VPREFETCH_256(&data[i + SAVGOL_PREFETCH_DISTANCE]);             \
            }                                                                    \
                                                                                 \
            __m256 w = VLOADU_PS_256(&weights[i]);                               \
            __m256 d = VLOADU_PS_256(&data[i]);                                  \
            acc = VFMADD_PS_256(w, d, acc);  /* FMA or mul+add fallback */      \
        }                                                                        \
                                                                                 \
        /* Scalar cleanup */                                                    \
        float scalar_sum = hsum_ps_256(acc);                                     \
        for (; i < (windowSize); i++) {                                          \
            scalar_sum += weights[i] * data[i];                                  \
        }                                                                        \
        scalar_sum;                                                              \
    })

#define SAVGOL_DOT_PRODUCT_AVX2_STRIDED(weights, data_end, windowSize, stride)  \
    ({                                                                           \
        float scalar_sum = 0.0f;                                                 \
        for (int i = 0; i < (windowSize); i++) {                                 \
            scalar_sum += weights[i] * data_end[i * (stride)];                   \
        }                                                                        \
        scalar_sum;                                                              \
    })

#endif // HAS_AVX2

#ifdef HAS_SSE2
/**
 * @brief SSE2 dot product kernel (4-wide mul+add)
 */
#define SAVGOL_DOT_PRODUCT_SSE2(weights, data, windowSize)                      \
    ({                                                                           \
        __m128 acc = VSETZERO_PS_128();                                          \
        int i = 0;                                                               \
        const int vec_end = (windowSize) & ~3;  /* Multiple of 4 */             \
                                                                                 \
        /* Main vectorized loop: 4 elements per iteration */                    \
        for (; i < vec_end; i += 4) {                                            \
            if (i + SAVGOL_PREFETCH_DISTANCE < (windowSize)) {                  \
                VPREFETCH_128(&weights[i + SAVGOL_PREFETCH_DISTANCE]);          \
                VPREFETCH_128(&data[i + SAVGOL_PREFETCH_DISTANCE]);             \
            }                                                                    \
                                                                                 \
            __m128 w = VLOADU_PS_128(&weights[i]);                               \
            __m128 d = VLOADU_PS_128(&data[i]);                                  \
            acc = VFMADD_PS_128(w, d, acc);  /* mul+add */                       \
        }                                                                        \
                                                                                 \
        /* Scalar cleanup */                                                    \
        float scalar_sum = hsum_ps_128(acc);                                     \
        for (; i < (windowSize); i++) {                                          \
            scalar_sum += weights[i] * data[i];                                  \
        }                                                                        \
        scalar_sum;                                                              \
    })

#define SAVGOL_DOT_PRODUCT_SSE2_STRIDED(weights, data_end, windowSize, stride)  \
    ({                                                                           \
        float scalar_sum = 0.0f;                                                 \
        for (int i = 0; i < (windowSize); i++) {                                 \
            scalar_sum += weights[i] * data_end[i * (stride)];                   \
        }                                                                        \
        scalar_sum;                                                              \
    })

#endif // HAS_SSE2

/**
 * @brief Scalar dot product with 4-way parallel accumulation chains
 * 
 * This is your existing optimized scalar code - keep the 4-chain pattern!
 */
#define SAVGOL_DOT_PRODUCT_SCALAR(weights, data, windowSize)                    \
    ({                                                                           \
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;               \
        const float *w_ptr = weights;                                            \
        const float *d_ptr = data;                                               \
                                                                                 \
        /* Handle remainder (windowSize % 4) */                                 \
        int remainder = (windowSize) & 3;                                        \
        switch (remainder) {                                                     \
            case 3:                                                              \
                sum0 += w_ptr[0] * d_ptr[0];                                     \
                sum1 += w_ptr[1] * d_ptr[1];                                     \
                sum2 += w_ptr[2] * d_ptr[2];                                     \
                w_ptr += 3;                                                      \
                d_ptr += 3;                                                      \
                break;                                                           \
            case 2:                                                              \
                sum0 += w_ptr[0] * d_ptr[0];                                     \
                sum1 += w_ptr[1] * d_ptr[1];                                     \
                w_ptr += 2;                                                      \
                d_ptr += 2;                                                      \
                break;                                                           \
            case 1:                                                              \
                sum0 += w_ptr[0] * d_ptr[0];                                     \
                w_ptr += 1;                                                      \
                d_ptr += 1;                                                      \
                break;                                                           \
            case 0:                                                              \
                break;                                                           \
        }                                                                        \
                                                                                 \
        /* Main 4-way unrolled loop */                                          \
        int main_iters = ((windowSize) - remainder) / 4;                         \
        for (int k = 0; k < main_iters; ++k) {                                   \
            sum0 += w_ptr[0] * d_ptr[0];  /* Chain 0 */                          \
            sum1 += w_ptr[1] * d_ptr[1];  /* Chain 1 (independent!) */           \
            sum2 += w_ptr[2] * d_ptr[2];  /* Chain 2 (independent!) */           \
            sum3 += w_ptr[3] * d_ptr[3];  /* Chain 3 (independent!) */           \
            w_ptr += 4;                                                          \
            d_ptr += 4;                                                          \
        }                                                                        \
                                                                                 \
        /* Pairwise reduction for numerical precision */                        \
        (sum0 + sum1) + (sum2 + sum3);                                           \
    })

#define SAVGOL_DOT_PRODUCT_SCALAR_STRIDED(weights, data_end, windowSize, stride) \
    ({                                                                            \
        float scalar_sum = 0.0f;                                                  \
        for (int i = 0; i < (windowSize); i++) {                                  \
            scalar_sum += weights[i] * data_end[i * (stride)];                    \
        }                                                                         \
        scalar_sum;                                                               \
    })

//==============================================================================
// DISPATCH MACRO: Select Optimal Kernel
//==============================================================================

/**
 * @brief Automatic dispatch to best available SIMD implementation
 * 
 * Dispatch hierarchy:
 * 1. If windowSize < 12: Use scalar 4-chain (lowest overhead)
 * 2. Else if AVX-512: Use 16-wide vectorization
 * 3. Else if AVX2: Use 8-wide vectorization
 * 4. Else if SSE2: Use 4-wide vectorization
 * 5. Else: Scalar fallback
 */
#define SAVGOL_DOT_PRODUCT(weights, data, windowSize)                           \
    ({                                                                           \
        float result;                                                            \
        if ((windowSize) < SAVGOL_SCALAR_THRESHOLD) {                           \
            result = SAVGOL_DOT_PRODUCT_SCALAR(weights, data, windowSize);      \
        }                                                                        \
        else {                                                                   \
            _Pragma("GCC diagnostic push")                                       \
            _Pragma("GCC diagnostic ignored \"-Wunknown-pragmas\"")             \
            IF_AVX512(result = SAVGOL_DOT_PRODUCT_AVX512(weights, data, windowSize);) \
            ELIF_AVX2(result = SAVGOL_DOT_PRODUCT_AVX2(weights, data, windowSize);)   \
            ELIF_SSE2(result = SAVGOL_DOT_PRODUCT_SSE2(weights, data, windowSize);)   \
            ELSE(result = SAVGOL_DOT_PRODUCT_SCALAR(weights, data, windowSize);)      \
            _Pragma("GCC diagnostic pop")                                        \
        }                                                                        \
        result;                                                                  \
    })

// Compile-time dispatch helpers
#ifdef HAS_AVX512
    #define IF_AVX512(...) __VA_ARGS__
    #define ELIF_AVX2(...)
    #define ELIF_SSE2(...)
    #define ELSE(...)
#elif defined(HAS_AVX2)
    #define IF_AVX512(...)
    #define ELIF_AVX2(...) __VA_ARGS__
    #define ELIF_SSE2(...)
    #define ELSE(...)
#elif defined(HAS_SSE2)
    #define IF_AVX512(...)
    #define ELIF_AVX2(...)
    #define ELIF_SSE2(...) __VA_ARGS__
    #define ELSE(...)
#else
    #define IF_AVX512(...)
    #define ELIF_AVX2(...)
    #define ELIF_SSE2(...)
    #define ELSE(...) __VA_ARGS__
#endif

#endif // SAVGOL_KERNELS_H