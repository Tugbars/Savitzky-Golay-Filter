/**
 * @file savgol_filter.c
 * @brief Optimized Savitzky–Golay filter with AVX-512 and versioning.
 *
 * This file provides an optimized Savitzky-Golay filter implementation, including
 * Gram polynomial evaluation, weight calculation, and filter application.
 * Key features:
 * - AVX-512 vectorization for central and edge regions, with AVX/SSE fallbacks.
 * - Fused multiply-add (FMA) in scalar, SSE, AVX, and AVX-512 paths.
 * - Per-instance memoization with versioning for Gram polynomials (thread-safe).
 * - Efficient cache invalidation via versioning.
 * - Deeper interleaved prefetching and unrolled loops for improved ILP.
 * - Optional non-temporal stores to reduce cache pollution.
 * - Specialized GramPolyVectorized4 for small polynomial orders (<4).
 *
 * Author: Tugbars Heptaskin
 * Date: 2025-06-14
 * Updated: 2025-06-27
 */

#include "savgolFilter.h"
#include <immintrin.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Ensure alignment for SIMD (64-byte for AVX-512)
#define ALIGNED __attribute__((aligned(64)))
#define MAX_WINDOW 255
#define MAX_HALF_WINDOW_FOR_MEMO 127
#define MAX_POLY_ORDER_FOR_MEMO 10
#define MAX_DERIVATIVE_FOR_MEMO 5

// Logging macro for error reporting
#define LOG_ERROR(fmt, ...) fprintf(stderr, "ERROR: " fmt "\n", ##__VA_ARGS__)

// Compiler attributes for optimization (GCC/Clang-specific)
#ifdef __GNUC__
#define HOT __attribute__((hot, flatten))
#define ALWAYS_INLINE __attribute__((always_inline))
#else
#define HOT
#define ALWAYS_INLINE
#endif

// Enable memoization for Gram polynomial caching
#define ENABLE_MEMOIZATION

// Helper function for horizontal sum of __m512 vector
// - Reduces 16 floats to a single sum using AVX-512
static inline float ALWAYS_INLINE hsum512(__m512 v) {
    // Perform horizontal reduction to sum 16 floats in the vector
    return _mm512_reduce_add_ps(v);
}

// Helper function to create a 16-bit mask for n elements (0 <= n <= 16)
// - Returns a mask with the first n bits set (e.g., n=8 -> 0x00FF)
static inline __mmask16 ALWAYS_INLINE make_mask(int n) {
    // Create mask with n bits set, handling edge cases (n outside 0–16)
    return (n >= 0 && n <= 16) ? (1u << n) - 1 : 0;
}

/**
 * @brief Computes the natural logarithm of the generalized factorial.
 *
 * Calculates the sum of logarithms for the product of consecutive integers
 * from (upperLimit - termCount + 1) to upperLimit, ensuring numerical stability
 * by avoiding direct multiplication to prevent overflow.
 *
 * @param upperLimit The upper limit of the product.
 * @param termCount The number of terms in the product.
 * @return The natural logarithm of the generalized factorial as a double.
 */
static inline double logGenFact(uint8_t upperLimit, uint8_t termCount) {
    // Initialize accumulator for logarithmic sum
    double log_product = 0.0;
    // Sum logarithms of integers from (upperLimit - termCount + 1) to upperLimit
    for (uint8_t j = upperLimit - termCount + 1; j <= upperLimit; j++) {
        log_product += log((double)j);
    }
    // Return the logarithmic factorial
    return log_product;
}

/**
 * @brief Iteratively computes the Gram polynomial.
 *
 * Computes the Gram polynomial F(k, d) using dynamic programming for efficiency.
 * Uses a recurrence relation to build polynomial values iteratively for orders
 * k=0 to polynomialOrder and derivative orders d=0 to derivativeOrder.
 *
 * @param polynomialOrder The polynomial order (k) to compute.
 * @param dataIndex The data index (can be negative, shifted relative to window center).
 * @param ctx Pointer to GramPolyContext with filter parameters (halfWindowSize, derivativeOrder).
 * @return The computed Gram polynomial value.
 */
static float GramPolyIterative(uint8_t polynomialOrder, int dataIndex, const GramPolyContext* ctx) {
    // Extract filter parameters from context
    uint8_t halfWindowSize = ctx->halfWindowSize;    // Half window size (m)
    uint8_t derivativeOrder = ctx->derivativeOrder;  // Derivative order (d)

    // Create 2D array to store intermediate Gram polynomial values
    // - dp[k][d] stores F(k, d) for k=0 to polynomialOrder, d=0 to derivativeOrder
    float dp[polynomialOrder + 1][derivativeOrder + 1];

    // Base case: k=0
    // - F(0, d) = 1 if d=0 (no derivative), 0 otherwise
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        dp[0][d] = (d == 0) ? 1.0f : 0.0f;
    }

    // Early return for polynomialOrder=0
    // - If k=0, result is F(0, derivativeOrder)
    if (polynomialOrder == 0) {
        return dp[0][derivativeOrder];
    }

    // Compute k=1: F(1, d) = (1/m) * (x * F(0, d) + d * F(0, d-1))
    // - x is dataIndex, m is halfWindowSize
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        dp[1][d] = (1.0f / halfWindowSize) * (dataIndex * dp[0][d] + (d > 0 ? d * dp[0][d - 1] : 0));
    }

    // Compute k>=2 using recurrence: F(k, d) = a * (x * F(k-1, d) + d * F(k-1, d-1)) - c * F(k-2, d)
    for (uint8_t k = 2; k <= polynomialOrder; k++) {
        // Compute coefficients a and c for the recurrence
        // - a = (4k - 2) / [k * (2m - k + 1)]
        // - c = [(k - 1) * (2m + k)] / [k * (2m - k + 1)]
        float a = (4.0f * k - 2.0f) / (k * (2.0f * halfWindowSize - k + 1.0f));
        float c = ((k - 1.0f) * (2.0f * halfWindowSize + k)) / (k * (2.0f * halfWindowSize - k + 1.0f));

        // Compute F(k, d) for each derivative order d
        for (uint8_t d = 0; d <= derivativeOrder; d++) {
            // Start with term = x * F(k-1, d)
            float term = dataIndex * dp[k - 1][d];
            // Add derivative term if d > 0: d * F(k-1, d-1)
            if (d > 0) {
                term += d * dp[k - 1][d - 1];
            }
            // Apply recurrence: F(k, d) = a * term - c * F(k-2, d)
            dp[k][d] = a * term - c * dp[k - 2][d];
        }
    }

    // Return the final Gram polynomial value F(polynomialOrder, derivativeOrder)
    return dp[polynomialOrder][derivativeOrder];
}

/**
 * @brief Computes Gram polynomials for 8 data indices simultaneously using AVX.
 *
 * Vectorizes the Gram polynomial computation F(k, d) across 8 data indices using
 * 256-bit AVX instructions for performance. Mirrors the scalar GramPolyIterative
 * function but processes 8 indices in parallel, producing a __m256 vector containing
 * the polynomial values. Uses dynamic programming for iterative computation.
 *
 * @param polynomialOrder The polynomial order (k) to compute.
 * @param dataIndices Array of 8 data indices (shifted relative to window center).
 * @param ctx Pointer to GramPolyContext with filter parameters (halfWindowSize, derivativeOrder).
 * @return A __m256 vector containing F(k, d) for the 8 indices.
 */
static __m256 HOT GramPolyVectorized(uint8_t polynomialOrder, const int dataIndices[8], const GramPolyContext* ctx) {
    // Extract filter parameters from the context for use in polynomial computation
    uint8_t halfWindowSize = ctx->halfWindowSize;    // Half the window size (m), used for scaling terms
    uint8_t derivativeOrder = ctx->derivativeOrder;  // Derivative order (d), determines derivative terms

    // Calculate the size of the dynamic programming array in bytes
    // - dp_vec[k][d] stores F(k, d) for k=0 to polynomialOrder, d=0 to derivativeOrder
    // - Each element is a 256-bit vector (__m256, 32 bytes) holding 8 float values
    size_t dp_size = (polynomialOrder + 1) * (derivativeOrder + 1) * sizeof(__m256);

    // Allocate dp_vec with 32-byte alignment for AVX operations using _mm_malloc
    // - Prevents stack overflow for large polynomialOrder or derivativeOrder
    // - Ensures efficient vector loads/stores with 32-byte alignment
    __m256* dp_vec = _mm_malloc(dp_size, 32);
    if (!dp_vec) {
        // Log allocation failure and return zero vector to handle gracefully
        LOG_ERROR("Failed to allocate memory for dp_vec in GramPolyVectorized.");
        return _mm256_setzero_ps();
    }

    // Initialize vector constants for efficiency
    // - zero: All 8 lanes set to 0.0f, used for base cases with d > 0
    // - one: All 8 lanes set to 1.0f, used for base case with k=0, d=0
    __m256 zero = _mm256_setzero_ps();
    __m256 one = _mm256_set1_ps(1.0f);

    // Base case: k=0
    // - F(0, d) = 1 if d=0 (no derivative), 0 otherwise
    // - Set dp_vec[d] for all 8 indices simultaneously
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        dp_vec[d] = (d == 0) ? one : zero;
    }

    // Early return for polynomialOrder=0
    // - If k=0, result is F(0, derivativeOrder)
    if (polynomialOrder == 0) {
        __m256 result = dp_vec[derivativeOrder];
        _mm_free(dp_vec);
        return result;
    }

    // Load 8 data indices into a 256-bit vector for parallel processing
    // - Convert dataIndices[0..7] to floats, packed high-to-low (7 to 0)
    __m256 dataIndexVec = _mm256_set_ps(
        (float)dataIndices[7], (float)dataIndices[6], (float)dataIndices[5], (float)dataIndices[4],
        (float)dataIndices[3], (float)dataIndices[2], (float)dataIndices[1], (float)dataIndices[0]
    );

    // Precompute inverse of halfWindowSize for k=1 scaling
    // - Broadcast 1.0f / halfWindowSize to all 8 lanes
    __m256 hwsInv = _mm256_set1_ps(1.0f / halfWindowSize);

    // Compute k=1: F(1, d) = (1/m) * (x * F(0, d) + d * F(0, d-1))
    // - Process all 8 indices in parallel for each derivative order
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        // Compute term = x * F(0, d), where x is dataIndexVec
        __m256 term = _mm256_mul_ps(dataIndexVec, dp_vec[d]);
        // Add derivative term if d > 0: d * F(0, d-1)
        if (d > 0) {
            term = _mm256_fmadd_ps(_mm256_set1_ps((float)d), dp_vec[d - 1], term);
        }
        // Scale term by 1/m and store for k=1
        dp_vec[(derivativeOrder + 1) + d] = _mm256_mul_ps(hwsInv, term);
    }

    // Compute k>=2 using recurrence: F(k, d) = a * (x * F(k-1, d) + d * F(k-1, d-1)) - c * F(k-2, d)
    for (uint8_t k = 2; k <= polynomialOrder; k++) {
        // Compute coefficients a and c for the recurrence
        // - a = (4k - 2) / [k * (2m - k + 1)]
        // - c = [(k - 1) * (2m + k)] / [k * (2m - k + 1)]
        float a = (4.0f * k - 2.0f) / (k * (2.0f * halfWindowSize - k + 1.0f));
        float c = ((k - 1.0f) * (2.0f * halfWindowSize + k)) / (k * (2.0f * halfWindowSize - k + 1.0f));

        // Broadcast coefficients to 256-bit vectors
        __m256 aVec = _mm256_set1_ps(a);
        __m256 cVec = _mm256_set1_ps(c);

        // Compute F(k, d) for each derivative order across all 8 indices
        for (uint8_t d = 0; d <= derivativeOrder; d++) {
            // Compute term = x * F(k-1, d)
            __m256 term = _mm256_mul_ps(dataIndexVec, dp_vec[(k - 1) * (derivativeOrder + 1) + d]);
            // Add derivative term if d > 0: d * F(k-1, d-1)
            if (d > 0) {
                term = _mm256_fmadd_ps(_mm256_set1_ps((float)d), dp_vec[(k - 1) * (derivativeOrder + 1) + d - 1], term);
            }
            // Apply recurrence: F(k, d) = a * term - c * F(k-2, d)
            dp_vec[k * (derivativeOrder + 1) + d] = _mm256_sub_ps(
                _mm256_mul_ps(aVec, term),
                _mm256_mul_ps(cVec, dp_vec[(k - 2) * (derivativeOrder + 1) + d])
            );
        }
    }

    // Extract final result: F(polynomialOrder, derivativeOrder)
    __m256 result = dp_vec[polynomialOrder * (derivativeOrder + 1) + derivativeOrder];

    // Free allocated memory to prevent leaks
    _mm_free(dp_vec);

    // Return the vector of Gram polynomial values
    return result;
}

/**
 * @brief Computes Gram polynomials for 8 data indices using AVX, specialized for polynomialOrder < 4.
 *
 * Vectorizes the Gram polynomial computation F(k, d) across 8 data indices using
 * 256-bit AVX instructions, unrolling the k loop for polynomialOrder=0..3 to avoid
 * dynamic allocation. Optimized for common small polynomial orders.
 *
 * @param polynomialOrder The polynomial order (k) to compute (must be < 4).
 * @param dataIndices Array of 8 data indices (shifted relative to window center).
 * @param ctx Pointer to GramPolyContext with filter parameters (halfWindowSize, derivativeOrder).
 * @return A __m256 vector containing F(k, d) for the 8 indices.
 */
static __m256 HOT GramPolyVectorized4(uint8_t polynomialOrder, const int dataIndices[8], const GramPolyContext* ctx) {
    // Extract filter parameters
    uint8_t halfWindowSize = ctx->halfWindowSize;    // Half the window size (m)
    uint8_t derivativeOrder = ctx->derivativeOrder;  // Derivative order (d)

    // Validate polynomialOrder
    if (polynomialOrder >= 4) {
        LOG_ERROR("GramPolyVectorized4 called with polynomialOrder=%d >= 4", polynomialOrder);
        return _mm256_setzero_ps();
    }

    // Initialize vector constants
    __m256 zero = _mm256_setzero_ps();
    __m256 one = _mm256_set1_ps(1.0f);

    // Create array to store intermediate Gram polynomial values
    // - dp_vec[k][d] for k=0..3, d=0..derivativeOrder
    __m256 dp_vec[4][MAX_DERIVATIVE_FOR_MEMO];

    // Base case: k=0
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        dp_vec[0][d] = (d == 0) ? one : zero;
    }

    // Early return for polynomialOrder=0
    if (polynomialOrder == 0) {
        return dp_vec[0][derivativeOrder];
    }

    // Load 8 data indices into a 256-bit vector
    __m256 dataIndexVec = _mm256_set_ps(
        (float)dataIndices[7], (float)dataIndices[6], (float)dataIndices[5], (float)dataIndices[4],
        (float)dataIndices[3], (float)dataIndices[2], (float)dataIndices[1], (float)dataIndices[0]
    );

    // Precompute inverse of halfWindowSize for k=1 scaling
    __m256 hwsInv = _mm256_set1_ps(1.0f / halfWindowSize);

    // Compute k=1: F(1, d) = (1/m) * (x * F(0, d) + d * F(0, d-1))
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        __m256 term = _mm256_mul_ps(dataIndexVec, dp_vec[0][d]);
        if (d > 0) {
            term = _mm256_fmadd_ps(_mm256_set1_ps((float)d), dp_vec[0][d - 1], term);
        }
        dp_vec[1][d] = _mm256_mul_ps(hwsInv, term);
    }

    // Early return for polynomialOrder=1
    if (polynomialOrder == 1) {
        return dp_vec[1][derivativeOrder];
    }

    // Compute k=2
    float a2 = (4.0f * 2 - 2.0f) / (2 * (2.0f * halfWindowSize - 2 + 1.0f));
    float c2 = ((2 - 1.0f) * (2.0f * halfWindowSize + 2)) / (2 * (2.0f * halfWindowSize - 2 + 1.0f));
    __m256 a2Vec = _mm256_set1_ps(a2);
    __m256 c2Vec = _mm256_set1_ps(c2);
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        __m256 term = _mm256_mul_ps(dataIndexVec, dp_vec[1][d]);
        if (d > 0) {
            term = _mm256_fmadd_ps(_mm256_set1_ps((float)d), dp_vec[1][d - 1], term);
        }
        dp_vec[2][d] = _mm256_sub_ps(
            _mm256_mul_ps(a2Vec, term),
            _mm256_mul_ps(c2Vec, dp_vec[0][d])
        );
    }

    // Early return for polynomialOrder=2
    if (polynomialOrder == 2) {
        return dp_vec[2][derivativeOrder];
    }

    // Compute k=3
    float a3 = (4.0f * 3 - 2.0f) / (3 * (2.0f * halfWindowSize - 3 + 1.0f));
    float c3 = ((3 - 1.0f) * (2.0f * halfWindowSize + 3)) / (3 * (2.0f * halfWindowSize - 3 + 1.0f));
    __m256 a3Vec = _mm256_set1_ps(a3);
    __m256 c3Vec = _mm256_set1_ps(c3);
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        __m256 term = _mm256_mul_ps(dataIndexVec, dp_vec[2][d]);
        if (d > 0) {
            term = _mm256_fmadd_ps(_mm256_set1_ps((float)d), dp_vec[2][d - 1], term);
        }
        dp_vec[3][d] = _mm256_sub_ps(
            _mm256_mul_ps(a3Vec, term),
            _mm256_mul_ps(c3Vec, dp_vec[1][d])
        );
    }

    // Return F(3, derivativeOrder)
    return dp_vec[3][derivativeOrder];
}

#ifdef ENABLE_MEMOIZATION
/**
 * @brief Invalidates the Gram polynomial cache.
 *
 * Increments the cache version to mark all cached Gram polynomial values as invalid,
 * ensuring recomputation when parameters change. Uses versioning for O(1) invalidation
 * in multi-threaded contexts, avoiding costly cache clearing.
 *
 * @param filter Pointer to SavitzkyGolayFilter containing the cache.
 */
void HOT ClearGramPolyCache(SavitzkyGolayFilter* filter) {
    // Increment cache version to invalidate all cache entries
    filter->state.cacheVersion++;
}

/**
 * @brief Computes a Gram polynomial with memoization for efficiency.
 *
 * Checks the cache for a precomputed Gram polynomial value based on polynomial order,
 * data index, and derivative order. If cached and valid, returns the cached value;
 * otherwise, computes the value using GramPolyIterative, stores it in the cache, and
 * returns it. Uses versioning to ensure cache validity in multi-threaded contexts.
 *
 * Note: The ALWAYS_INLINE attribute may trigger a warning with low optimization levels
 * (-O0 or -Og). Use -O2 or -Wno-attributes to suppress.
 *
 * @param polynomialOrder The polynomial order (k) to compute.
 * @param dataIndex The data index (shifted relative to window center).
 * @param ctx Pointer to GramPolyContext with filter parameters (halfWindowSize, derivativeOrder).
 * @param filter Pointer to SavitzkyGolayFilter for cache access.
 * @return The computed or cached Gram polynomial value.
 */
float ALWAYS_INLINE MemoizedGramPoly(uint8_t polynomialOrder, int dataIndex, const GramPolyContext* ctx, SavitzkyGolayFilter* filter) {
    // Shift dataIndex to a nonnegative index for cache lookup
    // - Adjusts index to range [0, 2*MAX_HALF_WINDOW_FOR_MEMO]
    int shiftedIndex = dataIndex + ctx->halfWindowSize;

    // Check if parameters are out of cache bounds
    // - Fall back to GramPolyIterative for out-of-range indices
    if (shiftedIndex < 0 || shiftedIndex >= (2 * MAX_HALF_WINDOW_FOR_MEMO + 1) ||
        polynomialOrder >= MAX_POLY_ORDER_FOR_MEMO ||
        ctx->derivativeOrder >= MAX_DERIVATIVE_FOR_MEMO) {
        return GramPolyIterative(polynomialOrder, dataIndex, ctx);
    }

    // Access cache entry for the given parameters
    GramPolyCacheEntry* entry = &filter->state.gramPolyCache[shiftedIndex][polynomialOrder][ctx->derivativeOrder];

    // Return cached value if valid and matches current cache version
    if (entry->isComputed && entry->version == filter->state.cacheVersion) {
        return entry->value;
    }

    // Compute the Gram polynomial using the iterative method
    float value = GramPolyIterative(polynomialOrder, dataIndex, ctx);

    // Store computed value in cache and mark as valid
    entry->value = value;
    entry->isComputed = true;
    entry->version = filter->state.cacheVersion;

    // Return the computed value
    return value;
}
#endif

/**
 * @brief Calculates the weight for a single data index in the Savitzky-Golay filter.
 *
 * Computes the weight by summing contributions from Gram polynomials over all
 * polynomial orders k (0 to polynomialOrder). Uses logarithmic generalized factorials
 * for numerical stability and memoization (if enabled) for performance.
 *
 * @param dataIndex The shifted data index (relative to window center).
 * @param targetPoint The target point within the window for evaluation.
 * @param polynomialOrder The maximum polynomial order used in the filter.
 * @param ctx Pointer to GramPolyContext with filter parameters.
 * @param filter Pointer to SavitzkyGolayFilter for cache access (if memoization enabled).
 * @return The computed weight for the data index.
 */
static float Weight(int dataIndex, int targetPoint, uint8_t polynomialOrder, const GramPolyContext* ctx, SavitzkyGolayFilter* filter) {
    // Initialize weight accumulator
    float w = 0.0f;

    // Iterate over polynomial orders to compute weight contributions
    for (uint8_t k = 0; k <= polynomialOrder; ++k) {
        // Compute part1: Gram polynomial at dataIndex with derivative order 0
        #ifdef ENABLE_MEMOIZATION
        float part1 = MemoizedGramPoly(k, dataIndex, ctx, filter);
        #else
        float part1 = GramPolyIterative(k, dataIndex, ctx);
        #endif

        // Compute part2: Gram polynomial at targetPoint with derivative order
        #ifdef ENABLE_MEMOIZATION
        float part2 = MemoizedGramPoly(k, targetPoint, ctx, filter);
        #else
        float part2 = GramPolyIterative(k, targetPoint, ctx);
        #endif

        // Compute scaling factor using logarithmic generalized factorials
        // - log_num: log of product from (2m-k+1) to 2m
        // - log_den: log of product from (2m+2) to (2m+k+1)
        double log_num = logGenFact(2 * ctx->halfWindowSize, k);
        double log_den = logGenFact(2 * ctx->halfWindowSize + k + 1, k + 1);
        double log_factor = log(2.0 * k + 1.0) + log_num - log_den;

        // Exponentiate to get the scaling factor
        double factor = exp(log_factor);

        // Accumulate: w += factor * part1 * part2
        w += (float)(factor * (double)part1 * (double)part2);
    }

    // Return the final weight
    return w;
}

/**
 * @brief Computes Savitzky-Golay weights for eight data indices using AVX.
 *
 * Calculates weights for eight data points in parallel using 256-bit AVX instructions
 * for performance. Uses memoization (if enabled) and logarithmic factorials for
 * stability. Stores results in an aligned output array. Uses specialized
 * GramPolyVectorized4 for polynomialOrder < 4 to avoid allocation overhead.
 *
 * @param dataIndices Array of 8 data indices (shifted relative to window center).
 * @param targetPoint The target point within the window.
 * @param polynomialOrder The maximum polynomial order used.
 * @param ctx Pointer to GramPolyContext with filter parameters.
 * @param filter Pointer to SavitzkyGolayFilter for cache access (if memoization enabled).
 * @param weightsOut Pointer to a __m256 vector to store the 8 weights.
 */
static void HOT WeightVectorized(int dataIndices[8], int targetPoint, uint8_t polynomialOrder, const GramPolyContext* ctx, SavitzkyGolayFilter* filter, __m256* weightsOut) {
    // Initialize 256-bit weight accumulator
    __m256 w = _mm256_setzero_ps();

    // Iterate over polynomial orders to compute contributions
    for (uint8_t k = 0; k <= polynomialOrder; ++k) {
        // Compute part1: Vectorized Gram polynomial for 8 indices
        __m256 part1 = (polynomialOrder < 4) ? GramPolyVectorized4(k, dataIndices, ctx) : GramPolyVectorized(k, dataIndices, ctx);

        // Compute part2: Scalar Gram polynomial at targetPoint
        #ifdef ENABLE_MEMOIZATION
        float part2 = MemoizedGramPoly(k, targetPoint, ctx, filter);
        #else
        float part2 = GramPolyIterative(k, targetPoint, ctx);
        #endif

        // Compute scaling factor using logarithmic generalized factorials
        // - log_num: log of product from (2m-k+1) to 2m
        // - log_den: log of product from (2m+2) to (2m+k+1)
        double log_num = logGenFact(2 * ctx->halfWindowSize, k);
        double log_den = logGenFact(2 * ctx->halfWindowSize + k + 1, k + 1);
        double log_factor = log(2.0 * k + 1.0) + log_num - log_den;
        float factor = (float)exp(log_factor);

        // Fuse factor and part2 into a single broadcast
        __m256 facp2 = _mm256_set1_ps(factor * part2);

        // Accumulate weights using FMA
        w = _mm256_fmadd_ps(facp2, part1, w);
    }

    // Store weights in aligned output array (use non-temporal stores if enabled)
    #ifdef USE_NONTEMPORAL_STORES
    _mm256_stream_ps((float*)weightsOut, w);
    #else
    _mm256_store_ps((float*)weightsOut, w);
    #endif
}

/**
 * @brief Computes the Savitzky–Golay weights for the entire filter window.
 *
 * Calculates convolution weights for the filter window (size 2*halfWindowSize+1)
 * using vectorized (AVX2) and scalar paths. Caches weights to avoid recomputation
 * and uses memoization (if enabled) for Gram polynomials. Supports non-temporal
 * stores for centralWeights if USE_NONTEMPORAL_STORES is defined.
 *
 * @param filter Pointer to SavitzkyGolayFilter with configuration and state.
 */
static void HOT ComputeWeights(SavitzkyGolayFilter* filter) {
    // Extract configuration parameters
    uint8_t halfWindowSize = filter->conf.halfWindowSize;    // Half the window size (m in N=2m+1)
    uint16_t targetPoint = filter->conf.targetPoint;         // Target point for evaluating the polynomial fit
    uint8_t polynomialOrder = filter->conf.polynomialOrder;  // Maximum polynomial order for the filter
    uint8_t derivativeOrder = filter->conf.derivativeOrder;  // Derivative order (0 for smoothing)
    SavGolState* state = &filter->state;                    // Pointer to filter state for weights and cache
    GramPolyContext ctx = {halfWindowSize, targetPoint, derivativeOrder}; // Context for Gram polynomial calculations
    uint16_t fullWindowSize = 2 * halfWindowSize + 1;       // Full window size (N=2m+1) for convolution

    // Check if cached weights are valid to avoid recomputation
    if (state->weightsValid &&
        halfWindowSize == state->lastHalfWindowSize &&
        polynomialOrder == state->lastPolyOrder &&
        derivativeOrder == state->lastDerivOrder &&
        targetPoint == state->lastTargetPoint) {
        return;
    }

    #ifdef ENABLE_MEMOIZATION
    // Invalidate Gram polynomial cache if parameters have changed
    if (!state->weightsValid ||
        halfWindowSize != state->lastHalfWindowSize ||
        polynomialOrder != state->lastPolyOrder ||
        derivativeOrder != state->lastDerivOrder ||
        targetPoint != state->lastTargetPoint) {
        ClearGramPolyCache(filter);
        state->lastHalfWindowSize = halfWindowSize;
        state->lastPolyOrder = polynomialOrder;
        state->lastDerivOrder = derivativeOrder;
        state->lastTargetPoint = targetPoint;
    }
    #endif

    // AVX2 vectorized weight computation: process 8 weights at a time
    int i;
    for (i = 0; i <= fullWindowSize - 8; i += 8) {
        int dataIndices[8] = {
            i - halfWindowSize, i + 1 - halfWindowSize, i + 2 - halfWindowSize, i + 3 - halfWindowSize,
            i + 4 - halfWindowSize, i + 5 - halfWindowSize, i + 6 - halfWindowSize, i + 7 - halfWindowSize
        };
        __m256 w;
        WeightVectorized(dataIndices, targetPoint, polynomialOrder, &ctx, filter, &w);
        // Use non-temporal stores if enabled to avoid cache pollution
        #ifdef USE_NONTEMPORAL_STORES
        _mm256_stream_ps(&state->centralWeights[i], w);
        #else
        _mm256_store_ps(&state->centralWeights[i], w);
        #endif
    }

    // Handle remaining elements (1–7) with masked store
    int remaining = fullWindowSize - i;
    if (remaining) {
        int dataIndices[8] = {0};
        for (int j = 0; j < remaining; j++) {
            dataIndices[j] = (i + j) - halfWindowSize;
        }
        __m256 w;
        WeightVectorized(dataIndices, targetPoint, polynomialOrder, &ctx, filter, &w);
        __m256i mask = _mm256_setr_epi32(
            remaining > 0 ? -1 : 0, remaining > 1 ? -1 : 0, remaining > 2 ? -1 : 0, remaining > 3 ? -1 : 0,
            remaining > 4 ? -1 : 0, remaining > 5 ? -1 : 0, remaining > 6 ? -1 : 0, remaining > 7 ? -1 : 0
        );
        // Use non-temporal masked store if enabled
        #ifdef USE_NONTEMPORAL_STORES
        // Note: AVX-512 required for _mm256_mask_stream_ps; fallback to standard store
        _mm256_maskstore_ps(&state->centralWeights[i], mask, w);
        #else
        _mm256_maskstore_ps(&state->centralWeights[i], mask, w);
        #endif
    }

    // Mark weights as valid to enable caching
    state->weightsValid = true;
}

/**
 * @brief Initializes the Savitzky–Golay filter structure.
 *
 * Configures a SavitzkyGolayFilter instance with the specified parameters, using
 * fixed-size arrays for weights and temporary storage. Allocates the structure with
 * 64-byte alignment to ensure centralWeights and tempWindow are aligned.
 *
 * @param halfWindowSize Half-window size (m in N=2m+1).
 * @param polynomialOrder Order of the polynomial for fitting.
 * @param targetPoint Target point within the window for evaluation.
 * @param derivativeOrder Order of the derivative (0 for smoothing).
 * @param time_step Time step value for scaling derivatives.
 * @return Pointer to initialized SavitzkyGolayFilter, or NULL on failure.
 */
SavitzkyGolayFilter* initFilter(uint8_t halfWindowSize, uint8_t polynomialOrder, uint8_t targetPoint, uint8_t derivativeOrder, float time_step) {
    // Allocate filter structure with 64-byte alignment for AVX-512
    SavitzkyGolayFilter* filter = (SavitzkyGolayFilter*)_mm_malloc(sizeof(SavitzkyGolayFilter), 64);
    if (filter == NULL) {
        // Log error and return NULL if allocation fails
        LOG_ERROR("Failed to allocate memory for SavitzkyGolayFilter.");
        return NULL;
    }

    // Zero-initialize filter to ensure clean state
    // - Ensures weightsValid=false and cache is initialized
    memset(filter, 0, sizeof(SavitzkyGolayFilter));

    // Initialize configuration parameters
    filter->conf.halfWindowSize = halfWindowSize;    // Half-window size for filter
    filter->conf.polynomialOrder = polynomialOrder;  // Polynomial order for fitting
    filter->conf.targetPoint = targetPoint;          // Target point for evaluation
    filter->conf.derivativeOrder = derivativeOrder;  // Derivative order (0 for smoothing)
    filter->conf.time_step = time_step;              // Time step for scaling derivatives

    // Compute dt as time_step^derivativeOrder for derivative scaling
    filter->dt = pow(time_step, derivativeOrder);

    // Initialize cache version for memoization
    filter->state.cacheVersion = 0;

    // Return initialized filter pointer
    return filter;
}

/**
 * @brief Frees the Savitzky–Golay filter structure.
 *
 * Deallocates the filter structure using _mm_free to match the aligned allocation
 * in initFilter.
 *
 * @param filter Pointer to the SavitzkyGolayFilter to free.
 */
void freeFilter(SavitzkyGolayFilter* filter) {
    if (filter != NULL) {
        _mm_free(filter);
    }
}

/**
 * @brief Applies the Savitzky–Golay filter to an array of raw data points.
 *
 * Performs smoothing or differentiation via weighted convolution of the input data
 * with Gram polynomial-derived weights. Handles central, leading, and trailing edge
 * regions with vectorized (AVX-512/AVX/SSE) and scalar paths. Uses mirror padding
 * for edge handling and reuses weights where possible. Includes deeper interleaved
 * prefetching and unrolled loops for large windows to reduce memory latency and
 * increase ILP.
 *
 * @param data Input array of measurements (phase angles).
 * @param dataSize Number of elements in data (must be >= window size).
 * @param halfWindowSize Half the smoothing window size (m in N=2m+1).
 * @param targetPoint Index within [0..2*m] for polynomial evaluation.
 * @param filter Pointer to configured SavitzkyGolayFilter instance.
 * @param filteredData Output array for filtered results.
 */
static void HOT ApplyFilter(
    MqsRawDataPoint_t data[],
    size_t dataSize,
    uint8_t halfWindowSize,
    uint16_t targetPoint,
    SavitzkyGolayFilter* filter,
    MqsRawDataPoint_t filteredData[])
{
    // Validate inputs to prevent invalid operations
    if (dataSize < 1 || halfWindowSize == 0) {
        LOG_ERROR("Invalid input: dataSize=%zu, halfWindowSize=%d", dataSize, halfWindowSize);
        return;
    }

    // Cap halfWindowSize to prevent buffer overflows
    uint8_t maxHalfWindowSize = (MAX_WINDOW - 1) / 2;
    if (halfWindowSize > maxHalfWindowSize) {
        LOG_ERROR("halfWindowSize (%d) exceeds maximum (%d). Adjusting.", halfWindowSize, maxHalfWindowSize);
        halfWindowSize = maxHalfWindowSize;
    }

    // Compute window parameters: full window size (N = 2m + 1), last index, and center offset
    int windowSize = 2 * halfWindowSize + 1; // Full window size for convolution
    int lastIndex = dataSize - 1;            // Index of the last data point
    uint8_t width = halfWindowSize;          // Offset to center of the window

    // Ensure dataSize is sufficient to apply the filter
    if (dataSize < windowSize) {
        LOG_ERROR("dataSize (%zu) is smaller than windowSize (%d)", dataSize, windowSize);
        return;
    }

    // Verify 64-byte alignment for filter arrays to ensure efficient SIMD loads
    assert(((uintptr_t)filter->state.centralWeights & 63) == 0 && "centralWeights must be 64-byte aligned");
    assert(((uintptr_t)filter->state.tempWindow & 63) == 0 && "tempWindow must be 64-byte aligned");
    // Note: data[].phaseAngle may not be aligned, so use unaligned loads (_mm512_loadu_ps)

    // Step 1: Precompute weights for the central region
    filter->conf.targetPoint = targetPoint;
    ComputeWeights(filter); // Fills centralWeights with Savitzky-Golay coefficients

    // Step 2: Apply filter to central data points (where a full window fits)
    #ifdef __AVX512F__
    // Optimized path using AVX-512 for Skylake-AVX512 CPUs
    for (int i = 0; i <= dataSize - windowSize; ++i) {
        // Restrict pointers to inform compiler of no aliasing
        float * __restrict__ w = filter->state.centralWeights; // Aligned weights
        float * __restrict__ x = &data[i].phaseAngle;         // Input data (may not be aligned)

        float acc; // Accumulator for the filtered value
        if (windowSize <= 16) {
            // Handle small windows (≤16 elements) with a single masked load
            __mmask16 mask = make_mask(windowSize);
            __m512 w0 = _mm512_maskz_load_ps(mask, w);         // Load weights (aligned)
            __m512 x0 = _mm512_maskz_loadu_ps(mask, x);        // Load data (unaligned)
            acc = hsum512(_mm512_fmadd_ps(w0, x0, _mm512_setzero_ps()));
        } else {
            // Handle larger windows (>16 elements) with unrolled vectorized loops
            __m512 sum0 = _mm512_setzero_ps(); // First 16-lane accumulator
            __m512 sum1 = _mm512_setzero_ps(); // Second 16-lane accumulator

            // Unroll loop to process 32 floats (2x16 lanes) per iteration
            int j = 0;
            for (; j + 31 < windowSize; j += 32) {
                // Deeper interleaved prefetching for large windows
                if (windowSize > 128) {
                    _mm_prefetch((char*)(x + j + 128), _MM_HINT_T0); // Prefetch 2 cache lines ahead
                    _mm_prefetch((char*)(x + j + 256), _MM_HINT_T0); // Prefetch 4 cache lines ahead
                    _mm_prefetch((char*)(w + j + 128), _MM_HINT_T0);
                    _mm_prefetch((char*)(w + j + 256), _MM_HINT_T0);
                }
                // Process lanes 0–15
                __m512 w0 = _mm512_load_ps(w + j);      // Load weights (aligned)
                __m512 x0 = _mm512_loadu_ps(x + j);     // Load data (unaligned)
                sum0 = _mm512_fmadd_ps(w0, x0, sum0);   // Fused multiply-add
                // Process lanes 16–31
                __m512 w1 = _mm512_load_ps(w + j + 16); // Load weights (aligned)
                __m512 x1 = _mm512_loadu_ps(x + j + 16); // Load data (unaligned)
                sum1 = _mm512_fmadd_ps(w1, x1, sum1);    // Fused multiply-add
            }
            // Handle remaining elements (<32) with masked store
            if (j < windowSize) {
                __mmask16 mask = make_mask(windowSize - j);
                __m512 w0 = _mm512_maskz_load_ps(mask, w + j);
                __m512 x0 = _mm512_maskz_loadu_ps(mask, x + j);
                sum0 = _mm512_fmadd_ps(w0, x0, sum0);
            }
            // Combine accumulators and reduce to a single float
            __m512 totsum = _mm512_add_ps(sum0, sum1);
            acc = hsum512(totsum);
        }
        // Store the filtered value at the window's center
        filteredData[i + width].phaseAngle = acc;
    }
    #else
    // Fallback for non-AVX-512 CPUs using AVX/SSE or scalar operations
    for (int i = 0; i <= dataSize - windowSize; ++i) {
        // Accumulator for the weighted sum
        float sum = 0.0f;
        int j = 0;
        #if defined(__AVX__)
        // AVX path: process 8 floats at a time
        for (; j <= windowSize - 8; j += 8) {
            __m256 w = _mm256_load_ps(&filter->state.centralWeights[j]); // Load weights
            __m256 d = _mm256_loadu_ps(&data[i + j].phaseAngle);        // Load data
            #ifdef __FMA__
            __m256 prod = _mm256_fmadd_ps(w, d, _mm256_setzero_ps());   // FMA
            #else
            __m256 prod = _mm256_mul_ps(w, d);                          // Multiply
            #endif
            // Reduce 8 floats to 1 via horizontal adds
            __m128 hi = _mm256_extractf128_ps(prod, 1);
            __m128 lo = _mm256_castps256_ps128(prod);
            __m128 sum128 = _mm_add_ps(hi, lo);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum += _mm_cvtss_f32(sum128);
        }
        #endif
        // SSE path: process 4 floats at a time
        for (; j <= windowSize - 4; j += 4) {
            __m128 w = _mm_load_ps(&filter->state.centralWeights[j]);    // Load weights
            __m128 d = _mm_loadu_ps(&data[i + j].phaseAngle);           // Load data
            #ifdef __FMA__
            __m128 prod = _mm_fmadd_ps(w, d, _mm_setzero_ps());         // FMA
            #else
            __m128 prod = _mm_mul_ps(w, d);                             // Multiply
            #endif
            // Reduce 4 floats to 1 via horizontal adds
            prod = _mm_hadd_ps(prod, prod);
            prod = _mm_hadd_ps(prod, prod);
            sum += _mm_cvtss_f32(prod);
        }
        // Scalar path: handle remaining elements
        for (; j < windowSize; ++j) {
            sum = fmaf(filter->state.centralWeights[j], data[i + j].phaseAngle, sum);
        }
        // Store the filtered value
        filteredData[i + width].phaseAngle = sum;
    }
    #endif

    // Step 3: Handle edge cases (leading and trailing edges) using mirror padding
    for (int i = 0; i < width; ++i) {
        int j;
        // --- Leading Edge ---
        // Adjust target point to fit polynomial at the edge and compute weights
        filter->conf.targetPoint = width - i;
        ComputeWeights(filter); // Fills centralWeights with edge-specific coefficients
        float leadingSum = 0.0f;

        // Fill tempWindow with mirrored data for boundary effects
        // - Reflect points around index 0 (e.g., for N=5: 4, 3, 2, 1, 0)
        for (j = 0; j < windowSize; ++j) {
            int dataIdx = windowSize - j - 1;
            filter->state.tempWindow[j] = data[dataIdx].phaseAngle;
        }

        #ifdef __AVX512F__
        if (windowSize <= 16) {
            // Small windows: use masked load for efficiency
            __mmask16 mask = make_mask(windowSize);
            __m512 w0 = _mm512_maskz_load_ps(mask, filter->state.centralWeights); // Load weights
            __m512 d0 = _mm512_maskz_load_ps(mask, filter->state.tempWindow);     // Load mirrored data
            leadingSum = hsum512(_mm512_fmadd_ps(w0, d0, _mm512_setzero_ps()));
        } else {
            // Larger windows: vectorized loop with deeper interleaved prefetching
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            // Unroll loop by 2x to process 32 floats per iteration
            for (j = 0; j <= windowSize - 32; j += 32) {
                // Deeper interleaved prefetching two iterations ahead
                if (windowSize > 128) {
                    _mm_prefetch((char*)(filter->state.centralWeights + j + 128), _MM_HINT_T0);
                    _mm_prefetch((char*)(filter->state.tempWindow + j + 128), _MM_HINT_T0);
                    _mm_prefetch((char*)(filter->state.centralWeights + j + 256), _MM_HINT_T0);
                    _mm_prefetch((char*)(filter->state.tempWindow + j + 256), _MM_HINT_T0);
                }
                // Process lanes 0–15
                __m512 w0 = _mm512_load_ps(filter->state.centralWeights + j); // Load weights
                __m512 d0 = _mm512_load_ps(filter->state.tempWindow + j);     // Load mirrored data
                sum0 = _mm512_fmadd_ps(w0, d0, sum0);                        // FMA
                // Process lanes 16–31
                __m512 w1 = _mm512_load_ps(filter->state.centralWeights + j + 16);
                __m512 d1 = _mm512_load_ps(filter->state.tempWindow + j + 16);
                sum1 = _mm512_fmadd_ps(w1, d1, sum1);                        // FMA
            }
            // Handle remaining elements (<32) with masked store
            if (j < windowSize) {
                __mmask16 mask = make_mask(windowSize - j);
                __m512 w0 = _mm512_maskz_load_ps(mask, filter->state.centralWeights + j);
                __m512 d0 = _mm512_maskz_load_ps(mask, filter->state.tempWindow + j);
                sum0 = _mm512_fmadd_ps(w0, d0, sum0);
            }
            leadingSum = hsum512(_mm512_add_ps(sum0, sum1));
        }
        #else
        // Fallback: scalar loop for non-AVX-512 CPUs
        for (j = 0; j < windowSize; ++j) {
            leadingSum = fmaf(filter->state.centralWeights[j], filter->state.tempWindow[j], leadingSum);
        }
        #endif
        filteredData[i].phaseAngle = leadingSum; // Store leading edge result

        // --- Trailing Edge (reuse mirror-edge weights) ---
        float trailingSum = 0.0f;

        // Fill tempWindow with mirrored data for the trailing edge
        for (j = 0; j < windowSize; ++j) {
            int dataIdx = lastIndex - windowSize + j + 1;
            filter->state.tempWindow[j] = data[dataIdx].phaseAngle;
        }

        #ifdef __AVX512F__
        if (windowSize <= 16) {
            // Small windows: masked load
            __mmask16 mask = make_mask(windowSize);
            __m512 w0 = _mm512_maskz_load_ps(mask, filter->state.centralWeights); // Reuse weights
            __m512 d0 = _mm512_maskz_load_ps(mask, filter->state.tempWindow);     // Load mirrored data
            trailingSum = hsum512(_mm512_fmadd_ps(w0, d0, _mm512_setzero_ps()));
        } else {
            // Larger windows: vectorized loop with deeper interleaved prefetching
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            // Unroll loop by 2x to process 32 floats per iteration
            for (j = 0; j <= windowSize - 32; j += 32) {
                // Deeper interleaved prefetching two iterations ahead
                if (windowSize > 128) {
                    _mm_prefetch((char*)(filter->state.centralWeights + j + 128), _MM_HINT_T0);
                    _mm_prefetch((char*)(filter->state.tempWindow + j + 128), _MM_HINT_T0);
                    _mm_prefetch((char*)(filter->state.centralWeights + j + 256), _MM_HINT_T0);
                    _mm_prefetch((char*)(filter->state.tempWindow + j + 256), _MM_HINT_T0);
                }
                // Process lanes 0–15
                __m512 w0 = _mm512_load_ps(filter->state.centralWeights + j); // Reuse weights
                __m512 d0 = _mm512_load_ps(filter->state.tempWindow + j);     // Load mirrored data
                sum0 = _mm512_fmadd_ps(w0, d0, sum0);                        // FMA
                // Process lanes 16–31
                __m512 w1 = _mm512_load_ps(filter->state.centralWeights + j + 16);
                __m512 d1 = _mm512_load_ps(filter->state.tempWindow + j + 16);
                sum1 = _mm512_fmadd_ps(w1, d1, sum1);                        // FMA
            }
            // Handle remaining elements (<32) with masked store
            if (j < windowSize) {
                __mmask16 mask = make_mask(windowSize - j);
                __m512 w0 = _mm512_maskz_load_ps(mask, filter->state.centralWeights + j);
                __m512 d0 = _mm512_maskz_load_ps(mask, filter->state.tempWindow + j);
                sum0 = _mm512_fmadd_ps(w0, d0, sum0);
            }
            trailingSum = hsum512(_mm512_add_ps(sum0, sum1));
        }
        #else
        // Fallback: scalar loop
        for (j = 0; j < windowSize; ++j) {
            trailingSum = fmaf(filter->state.centralWeights[j], filter->state.tempWindow[j], trailingSum);
        }
        #endif
        filteredData[lastIndex - i].phaseAngle = trailingSum; // Store trailing edge result
    }
}

/**
 * @brief Main function to apply the Savitzky–Golay filter.
 *
 * Initializes a filter instance and applies it to the input data, performing
 * input validation and error handling. Returns the filter instance for caller
 * to free using freeFilter.
 *
 * @param data Input array of measurements.
 * @param dataSize Number of elements in data.
 * @param halfWindowSize Half the smoothing window size.
 * @param filteredData Output array for filtered results.
 * @param polynomialOrder Polynomial order for fitting.
 * @param targetPoint Target point within the window.
 * @param derivativeOrder Derivative order (0 for smoothing).
 * @return Pointer to initialized SavitzkyGolayFilter, or NULL on failure.
 */
SavitzkyGolayFilter* mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                                      MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder,
                                      uint8_t targetPoint, uint8_t derivativeOrder) {
    // Validate input parameters with assertions for development
    assert(data != NULL && "Input data pointer must not be NULL");
    assert(filteredData != NULL && "Filtered data pointer must not be NULL");
    assert(dataSize > 0 && "Data size must be greater than 0");
    assert(halfWindowSize > 0 && "Half-window size must be greater than 0");
    assert((2 * halfWindowSize + 1) <= dataSize && "Filter window size must not exceed data size");
    assert(polynomialOrder < (2 * halfWindowSize + 1) && "Polynomial order must be less than the filter window size");
    assert(targetPoint <= (2 * halfWindowSize) && "Target point must be within the filter window");

    // Runtime validation with error logging
    if (data == NULL || filteredData == NULL) {
        LOG_ERROR("NULL pointer passed to mes_savgolFilter.");
        return NULL;
    }
    if (dataSize == 0 || halfWindowSize == 0 ||
        polynomialOrder >= 2 * halfWindowSize + 1 ||
        targetPoint > 2 * halfWindowSize ||
        (2 * halfWindowSize + 1) > dataSize) {
        LOG_ERROR("Invalid filter parameters: dataSize=%zu, halfWindowSize=%d, polynomialOrder=%d, targetPoint=%d.",
                  dataSize, halfWindowSize, polynomialOrder, targetPoint);
        return NULL;
    }

    // Initialize filter with provided parameters
    SavitzkyGolayFilter* filter = initFilter(halfWindowSize, polynomialOrder, targetPoint, derivativeOrder, 1.0f);
    if (filter == NULL) {
        LOG_ERROR("Failed to initialize filter.");
        return NULL;
    }

    // Apply the filter to the input data
    ApplyFilter(data, dataSize, halfWindowSize, targetPoint, filter, filteredData);

    // Return the filter instance for caller to free using freeFilter
    return filter;
}