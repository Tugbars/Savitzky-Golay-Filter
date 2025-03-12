/**
 * @file savgol_filter.c
 * @brief Implementation of the Savitzky–Golay filter.
 *
 * This file provides the implementation for the Savitzky–Golay filter functions,
 * including Gram polynomial evaluation, weight calculation, and application of the filter.
 * The filter uses an iterative (dynamic programming) method to compute Gram polynomials,
 * and (optionally) an optimized precomputation for generalized factorial (GenFact) values.
 *
 * Author: Tugbars Heptaskin
 * Date: 2025-03-12
 */

#include "savgolFilter.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h> // For SSE/AVX intrinsics

// Ensure alignment for SIMD (32-byte for AVX, 16-byte for SSE)
#define ALIGNED __attribute__((aligned(32)))

/*-------------------------
  Logging Macro
-------------------------*/
#define LOG_ERROR(fmt, ...) fprintf(stderr, "ERROR: " fmt "\n", ##__VA_ARGS__)

//-------------------------
// Preprocessor Definitions for Memoization
//-------------------------

// Uncomment the following line to enable memoization of Gram polynomial calculations.
#define ENABLE_MEMOIZATION

/**
 * @brief Computes the natural logarithm of the generalized factorial.
 *
 * This function calculates the sum of logarithms for the product of consecutive integers
 * from (upperLimit - termCount + 1) to upperLimit, avoiding direct multiplication to
 * prevent overflow and improve numerical stability.
 *
 * @param upperLimit The upper limit of the product.
 * @param termCount The number of terms in the product.
 * @return The natural logarithm of the generalized factorial as a double.
 */
static inline double logGenFact(uint8_t upperLimit, uint8_t termCount) {
    double log_product = 0.0;
    for (uint8_t j = upperLimit - termCount + 1; j <= upperLimit; j++) {
        log_product += log((double)j);
    }
    return log_product;
}

//-------------------------
// Iterative Gram Polynomial Calculation
//-------------------------
/**
 * @brief Iteratively computes the Gram polynomial.
 *
 * This function computes the Gram polynomial F(k, d) using dynamic programming.
 *
 * @param polynomialOrder The current polynomial order k.
 * @param dataIndex The data index (can be negative if shifted so that the center is 0).
 * @param ctx Pointer to a GramPolyContext containing filter parameters.
 * @return The computed Gram polynomial value.
 */
static float GramPolyIterative(uint8_t polynomialOrder, int dataIndex, const GramPolyContext* ctx) {
    // Retrieve necessary parameters from the context.
    uint8_t halfWindowSize = ctx->halfWindowSize;    // Half window size used in the filter.
    uint8_t derivativeOrder = ctx->derivativeOrder;    // Order of the derivative to compute.

    // Create a 2D array 'dp' to store intermediate Gram polynomial values.
    // dp[k][d] will store F(k, d): the Gram polynomial of order k and derivative order d.
    float dp[polynomialOrder + 1][derivativeOrder + 1];

    // Base case: k = 0.
    // For the zeroth order, the polynomial is 1 when derivative order is 0, and 0 for d > 0.
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        dp[0][d] = (d == 0) ? 1.0f : 0.0f;
    }
    // If the requested polynomial order is 0, return the base case directly.
    if (polynomialOrder == 0) {
        return dp[0][derivativeOrder];
    }

    // k = 1: Compute first order polynomial values using the base case.
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        // The formula for F(1, d) uses the base value F(0, d) and, if needed, the derivative of F(0, d-1).
        dp[1][d] = (1.0f / halfWindowSize) * (dataIndex * dp[0][d] + (d > 0 ? d * dp[0][d - 1] : 0));
    }

    // Iteratively compute F(k, d) for k >= 2.
    // The recurrence relation uses previously computed values for orders k-1 and k-2.
    for (uint8_t k = 2; k <= polynomialOrder; k++) {
        // Compute constants 'a' and 'c' for the recurrence:
        // a = (4k - 2) / [k * (2*halfWindowSize - k + 1)]
        // c = [(k - 1) * (2*halfWindowSize + k)] / [k * (2*halfWindowSize - k + 1)]
        float a = (4.0f * k - 2.0f) / (k * (2.0f * halfWindowSize - k + 1.0f));
        float c = ((k - 1.0f) * (2.0f * halfWindowSize + k)) / (k * (2.0f * halfWindowSize - k + 1.0f));

        // For each derivative order from 0 up to derivativeOrder:
        for (uint8_t d = 0; d <= derivativeOrder; d++) {
            // Start with term = dataIndex * F(k-1, d)
            float term = dataIndex * dp[k - 1][d];
            // If computing a derivative (d > 0), add the derivative term: d * F(k-1, d-1)
            if (d > 0) {
                term += d * dp[k - 1][d - 1];
            }
            // The recurrence: F(k, d) = a * (term) - c * F(k-2, d)
            dp[k][d] = a * term - c * dp[k - 2][d];
        }
    }

    // Return the computed Gram polynomial for the requested polynomial order and derivative order.
    return dp[polynomialOrder][derivativeOrder];
}

/**
 * @brief Computes Gram polynomials for 8 data indices simultaneously using AVX.
 *
 * This function vectorizes the Gram polynomial computation F(k, d) across 8 data indices,
 * leveraging 256-bit AVX instructions to improve performance. It mirrors the scalar GramPolyIterative
 * function but processes 8 indices in parallel, producing a 256-bit vector (__m256) containing
 * the polynomial values for each index. The computation follows a dynamic programming approach,
 * building the polynomial values iteratively for orders k = 0 to polynomialOrder and derivative
 * orders d = 0 to derivativeOrder.
 *
 * @param polynomialOrder The polynomial order (k) to compute, ranging from 0 to a maximum value.
 * @param dataIndices Array of 8 data indices (shifted relative to the window center, e.g., -m to m).
 * @param ctx Pointer to a GramPolyContext containing filter parameters (halfWindowSize, derivativeOrder).
 * @return A 256-bit vector (__m256) containing the computed Gram polynomial values F(k, d) for the 8 indices.
 */
static __m256 GramPolyVectorized(uint8_t polynomialOrder, const int dataIndices[8], const GramPolyContext* ctx) {
    // Extract filter parameters from the context structure.
    uint8_t halfWindowSize = ctx->halfWindowSize;    // Half the window size (m), used in scaling and normalization.
    uint8_t derivativeOrder = ctx->derivativeOrder;  // Derivative order (d), determines the number of derivative terms.

    // Calculate the size of the dynamic programming array dp_vec in bytes.
    // - dp_vec[k][d] stores F(k, d) for k = 0 to polynomialOrder and d = 0 to derivativeOrder.
    // - Each element is a 256-bit vector (__m256, 32 bytes) holding 8 float values.
    // - Total size: (polynomialOrder + 1) * (derivativeOrder + 1) * sizeof(__m256).
    size_t dp_size = (polynomialOrder + 1) * (derivativeOrder + 1) * sizeof(__m256);

    // Dynamically allocate dp_vec using _mm_malloc to ensure 32-byte alignment for AVX operations.
    // - This avoids stack overflow issues that would occur with large polynomialOrder or derivativeOrder.
    // - _mm_malloc aligns memory to 32 bytes, matching AVX requirements for efficient vector loads/stores.
    __m256* dp_vec = _mm_malloc(dp_size, 32);
    if (!dp_vec) {
        // If allocation fails, log an error and return a zero vector to indicate failure gracefully.
        LOG_ERROR("Failed to allocate memory for dp_vec in GramPolyVectorized.");
        return _mm256_setzero_ps();
    }

    // Predefine common vector constants for efficiency.
    // - zero: A 256-bit vector of all 0.0f, used for initialization and base cases where d > 0.
    // - one: A 256-bit vector of all 1.0f, used for the base case where k = 0 and d = 0.
    __m256 zero = _mm256_setzero_ps();
    __m256 one = _mm256_set1_ps(1.0f);

    // Base case: k = 0
    // - For polynomial order k = 0, F(0, d) = 1 if d = 0 (no derivative), 0 otherwise.
    // - Set dp_vec[d] for all 8 indices simultaneously: all lanes are 1.0f when d = 0, 0.0f when d > 0.
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        dp_vec[d] = (d == 0) ? one : zero;
    }

    // Early return if polynomialOrder is 0.
    // - If k = 0, the result is simply F(0, derivativeOrder), already computed above.
    // - Free memory and return the result to avoid unnecessary computation.
    if (polynomialOrder == 0) {
        __m256 result = dp_vec[derivativeOrder];
        _mm_free(dp_vec);
        return result;
    }

    // Load the 8 data indices into a 256-bit vector for vectorized operations.
    // - dataIndices[0] to dataIndices[7] are cast to floats and packed into dataIndexVec.
    // - Order is high-to-low (7 to 0) to match AVX register layout for subsequent stores.
    __m256 dataIndexVec = _mm256_set_ps(
        (float)dataIndices[7], (float)dataIndices[6], (float)dataIndices[5], (float)dataIndices[4],
        (float)dataIndices[3], (float)dataIndices[2], (float)dataIndices[1], (float)dataIndices[0]
    );

    // Precompute the inverse of halfWindowSize as a vector for scaling in k = 1 case.
    // - 1.0f / halfWindowSize is broadcast to all 8 lanes, used to normalize terms.
    __m256 hwsInv = _mm256_set1_ps(1.0f / halfWindowSize);

    // Compute k = 1 case: F(1, d) = (1/m) * (x * F(0, d) + d * F(0, d-1)).
    // - For each derivative order d, compute the term for all 8 indices in parallel.
    // - dp_vec is a 1D array, so k=1 values start at index derivativeOrder + 1.
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        // term = x * F(0, d), where x is dataIndexVec (8 indices) and F(0, d) is dp_vec[d].
        __m256 term = _mm256_mul_ps(dataIndexVec, dp_vec[d]);
        // If d > 0, add the derivative term: d * F(0, d-1), using fused multiply-add for efficiency.
        if (d > 0) {
            term = _mm256_fmadd_ps(_mm256_set1_ps((float)d), dp_vec[d - 1], term);
        }
        // Scale by 1/m and store in dp_vec for k = 1.
        dp_vec[(derivativeOrder + 1) + d] = _mm256_mul_ps(hwsInv, term);
    }

    // Compute k >= 2 cases using the recurrence relation: F(k, d) = a * (x * F(k-1, d) + d * F(k-1, d-1)) - c * F(k-2, d).
    // - Iterate over polynomial orders k = 2 to polynomialOrder.
    for (uint8_t k = 2; k <= polynomialOrder; k++) {
        // Compute scalar coefficients a and c for the recurrence, same for all 8 indices.
        // - a = (4k - 2) / [k * (2m - k + 1)], scales the current term.
        // - c = [(k - 1) * (2m + k)] / [k * (2m - k + 1)], weights the previous term.
        float a = (4.0f * k - 2.0f) / (k * (2.0f * halfWindowSize - k + 1.0f));
        float c = ((k - 1.0f) * (2.0f * halfWindowSize + k)) / (k * (2.0f * halfWindowSize - k + 1.0f));
        
        // Broadcast a and c to 256-bit vectors for vectorized operations across all 8 indices.
        __m256 aVec = _mm256_set1_ps(a);
        __m256 cVec = _mm256_set1_ps(c);

        // For each derivative order d, compute F(k, d) for all 8 indices.
        for (uint8_t d = 0; d <= derivativeOrder; d++) {
            // term = x * F(k-1, d), where F(k-1, d) is accessed from dp_vec at the previous k index.
            __m256 term = _mm256_mul_ps(dataIndexVec, dp_vec[(k - 1) * (derivativeOrder + 1) + d]);
            // If d > 0, add the derivative term: d * F(k-1, d-1).
            if (d > 0) {
                term = _mm256_fmadd_ps(_mm256_set1_ps((float)d), dp_vec[(k - 1) * (derivativeOrder + 1) + d - 1], term);
            }
            // Compute F(k, d) = a * term - c * F(k-2, d), storing in dp_vec at the current k index.
            dp_vec[k * (derivativeOrder + 1) + d] = _mm256_sub_ps(
                _mm256_mul_ps(aVec, term),
                _mm256_mul_ps(cVec, dp_vec[(k - 2) * (derivativeOrder + 1) + d])
            );
        }
    }

    // Extract the final result: F(polynomialOrder, derivativeOrder) for all 8 indices.
    // - Located at dp_vec[polynomialOrder * (derivativeOrder + 1) + derivativeOrder] in the 1D array.
    __m256 result = dp_vec[polynomialOrder * (derivativeOrder + 1) + derivativeOrder];

    // Free the dynamically allocated memory to prevent leaks.
    _mm_free(dp_vec);

    // Return the 256-bit vector containing the Gram polynomial values for the 8 indices.
    return result;
}

//-------------------------
// Optional Memoization for Gram Polynomial Calculation
//-------------------------
#ifdef ENABLE_MEMOIZATION

/**
 * @brief Structure for caching Gram polynomial results.
 */
typedef struct {
    bool isComputed;
    float value;
} GramPolyCacheEntry;

// Define maximum cache dimensions (adjust as needed).
#define MAX_HALF_WINDOW_FOR_MEMO 32
#define MAX_POLY_ORDER_FOR_MEMO 5       // Supports polynomial orders 0..4.
#define MAX_DERIVATIVE_FOR_MEMO 5       // Supports derivative orders 0..4.

static GramPolyCacheEntry gramPolyCache[2 * MAX_HALF_WINDOW_FOR_MEMO + 1][MAX_POLY_ORDER_FOR_MEMO][MAX_DERIVATIVE_FOR_MEMO];

/**
 * @brief Clears the memoization cache for the current domain.
 *
 * @param halfWindowSize Half-window size.
 * @param polynomialOrder Polynomial order.
 * @param derivativeOrder Derivative order.
 */
static void ClearGramPolyCache(uint8_t halfWindowSize, uint8_t polynomialOrder, uint8_t derivativeOrder) {
    int maxIndex = 2 * halfWindowSize + 1;
    for (int i = 0; i < maxIndex; i++) {
        for (int k = 0; k <= polynomialOrder; k++) {
            for (int d = 0; d <= derivativeOrder; d++) {
                gramPolyCache[i][k][d].isComputed = false;
            }
        }
    }
}

/**
 * @brief Wrapper for GramPolyIterative with memoization.
 *
 * This function first checks if the Gram polynomial for a given set of parameters has
 * already been computed and stored in the cache. The cache is indexed by:
 * - dataIndex (shifted by halfWindowSize to ensure a nonnegative index),
 * - polynomial order,
 * - derivative order.
 * 
 * If a cached value is found, it is returned directly. Otherwise, the function computes
 * the value using GramPolyIterative, stores it in the cache, and then returns the result.
 *
 * @param polynomialOrder The polynomial order (k).
 * @param dataIndex The (shifted) data index (expected range: [-halfWindowSize, halfWindowSize]).
 * @param ctx Pointer to a GramPolyContext containing filter parameters.
 * @return The computed Gram polynomial value.
 */
static float MemoizedGramPoly(uint8_t polynomialOrder, int dataIndex, const GramPolyContext* ctx) {
    // Shift dataIndex to a nonnegative index for cache lookup.
    int shiftedIndex = dataIndex + ctx->halfWindowSize;
    
    // Check if the shifted index falls outside the range supported by the cache.
    if (shiftedIndex < 0 || shiftedIndex >= (2 * MAX_HALF_WINDOW_FOR_MEMO + 1)) {
        // If it's out of range, compute the value directly without memoization.
        return GramPolyIterative(polynomialOrder, dataIndex, ctx);
    }
    
    // If the polynomial order or derivative order exceeds our cache capacity,
    // fall back to the iterative computation.
    if (polynomialOrder >= MAX_POLY_ORDER_FOR_MEMO || ctx->derivativeOrder >= MAX_DERIVATIVE_FOR_MEMO) {
        return GramPolyIterative(polynomialOrder, dataIndex, ctx);
    }
    
    // Check if the value for these parameters is already computed.
    if (gramPolyCache[shiftedIndex][polynomialOrder][ctx->derivativeOrder].isComputed) {
        // Return the cached value.
        return gramPolyCache[shiftedIndex][polynomialOrder][ctx->derivativeOrder].value;
    }
    
    // Compute the Gram polynomial using the iterative method.
    float value = GramPolyIterative(polynomialOrder, dataIndex, ctx);
    
    // Store the computed value in the cache and mark it as computed.
    gramPolyCache[shiftedIndex][polynomialOrder][ctx->derivativeOrder].value = value;
    gramPolyCache[shiftedIndex][polynomialOrder][ctx->derivativeOrder].isComputed = true;
    
    // Return the newly computed value.
    return value;
}

#endif // ENABLE_MEMOIZATION

//-------------------------
// Weight Calculation Using Gram Polynomials
//-------------------------

/**
 * @brief Calculates the weight for a single data index in the Savitzky-Golay filter window.
 *
 * This function computes the weight for a specific data point by summing contributions from
 * Gram polynomials over all polynomial orders k (from 0 to polynomialOrder). For each order k:
 * - Evaluates 'part1': Gram polynomial at the data index with derivative order 0.
 * - Evaluates 'part2': Gram polynomial at the target point with the specified derivative order.
 * - Computes a scaling factor using the logarithmic generalized factorial method and accumulates the result.
 *
 * The function uses memoization for Gram polynomials if enabled. The scaling factor is calculated
 * on-the-fly using logarithms to ensure numerical stability, eliminating the need for a precomputed
 * lookup table.
 *
 * @param dataIndex The shifted data index (relative to the window center).
 * @param targetPoint The target point within the window where the fit is evaluated.
 * @param polynomialOrder The maximum order of the polynomial used in the filter.
 * @param ctx Pointer to a GramPolyContext containing filter parameters (e.g., halfWindowSize, derivativeOrder).
 * @return The computed weight for the data index.
 */
static float Weight(int dataIndex, int targetPoint, uint8_t polynomialOrder, const GramPolyContext* ctx) {
    float w = 0.0f;  // Initialize weight accumulator to zero.

    // Iterate over polynomial orders from 0 to polynomialOrder.
    for (uint8_t k = 0; k <= polynomialOrder; ++k) {
        // Compute part1: Gram polynomial at dataIndex with derivative order 0.
#ifdef ENABLE_MEMOIZATION
        float part1 = MemoizedGramPoly(k, dataIndex, ctx);
#else
        float part1 = GramPolyIterative(k, dataIndex, ctx);
#endif

        // Compute part2: Gram polynomial at targetPoint with the derivative order from ctx.
#ifdef ENABLE_MEMOIZATION
        float part2 = MemoizedGramPoly(k, targetPoint, ctx);
#else
        float part2 = GramPolyIterative(k, targetPoint, ctx);
#endif

        // Compute the logarithmic terms for numerator and denominator of the scaling factor.
        double log_num = logGenFact(2 * ctx->halfWindowSize, k);
        double log_den = logGenFact(2 * ctx->halfWindowSize + k + 1, k + 1);

        // Compute log_factor = log(2k + 1) + log_num - log_den
        double log_factor = log(2.0 * k + 1.0) + log_num - log_den;

        // Exponentiate to get the scaling factor
        double factor = exp(log_factor);

        // Accumulate the contribution to the weight using double precision for stability, then cast to float.
        w += (float)(factor * (double)part1 * (double)part2);
    }

    return w;  // Return the final computed weight.
}
/**
 * @brief Computes Savitzky-Golay weights for eight data indices simultaneously using AVX.
 *
 * This function calculates weights for eight data points in parallel using 256-bit AVX instructions,
 * improving performance over the scalar Weight() function for large window sizes. For each
 * polynomial order k (from 0 to polynomialOrder):
 * - Computes 'part1': Vectorized Gram polynomial for eight data indices (derivative order 0).
 * - Computes 'part2': Scalar Gram polynomial at the target point, broadcasted to all eight lanes.
 * - Scales by a factor computed on-the-fly using logarithmic generalized factorials, broadcasted to all lanes.
 * - Accumulates results using fused multiply-add (FMA) operations.
 *
 * The function uses memoization for Gram polynomials if enabled. The scaling factor is calculated
 * using logarithms to ensure numerical stability, eliminating the need for a precomputed lookup table.
 *
 * @param dataIndices Array of 8 data indices (shifted relative to the window center).
 * @param targetPoint The target point within the window where the fit is evaluated.
 * @param polynomialOrder The maximum order of the polynomial used in the filter.
 * @param ctx Pointer to a GramPolyContext containing filter parameters (e.g., halfWindowSize, derivativeOrder).
 * @param weightsOut Pointer to a 256-bit vector (__m256) to store the 8 computed weights.
 */
static void WeightVectorized(int dataIndices[8], int targetPoint, uint8_t polynomialOrder, const GramPolyContext* ctx, __m256* weightsOut) {
    __m256 w = _mm256_setzero_ps();  // Initialize 256-bit weight accumulator to zero for all eight indices.

    // Iterate over polynomial orders from 0 to polynomialOrder.
    for (uint8_t k = 0; k <= polynomialOrder; ++k) {
        // Compute part1: Vectorized Gram polynomial for eight data indices.
        // Returns a __m256 vector with F(k, dataIndices[0..7], 0).
        __m256 part1 = GramPolyVectorized(k, dataIndices, ctx);

        // Compute part2: Scalar Gram polynomial at targetPoint (reused across all eight indices).
#ifdef ENABLE_MEMOIZATION
        float part2 = MemoizedGramPoly(k, targetPoint, ctx);
#else
        float part2 = GramPolyIterative(k, targetPoint, ctx);
#endif

        // Compute logarithmic terms for the scaling factor.
        double log_num = logGenFact(2 * ctx->halfWindowSize, k);
        double log_den = logGenFact(2 * ctx->halfWindowSize + k + 1, k + 1);
        double log_factor = log(2.0 * k + 1.0) + log_num - log_den;
        float factor = (float)exp(log_factor);

        // Broadcast scalar values to 256-bit vectors for vectorized operations.
        __m256 factorVec = _mm256_set1_ps(factor);  // All eight lanes set to factor.
        __m256 part2Vec = _mm256_set1_ps(part2);    // All eight lanes set to part2.

        // Accumulate weights using FMA: w = w + (factorVec * part1 * part2Vec).
        // Computes w[i] += factor * part1[i] * part2 for i=0..7 in parallel.
        w = _mm256_fmadd_ps(factorVec, _mm256_mul_ps(part1, part2Vec), w);
    }

    // Store the final weights in the output vector.
    *weightsOut = w;
}

/**
 * @brief Computes the Savitzky–Golay weights for the entire filter window.
 *
 * This function calculates the convolution weights used in the Savitzky–Golay filter.
 * It loops through each index in the filter window (of size 2*halfWindowSize+1) and
 * computes the corresponding weight by evaluating the Gram polynomial-based weight function.
 *
 * @param halfWindowSize Half-window size.
 * @param targetPoint The target point in the window (the point where the fit is evaluated).
 * @param polynomialOrder Polynomial order for fitting.
 * @param derivativeOrder Derivative order for the filter.
 * @param weights Array (size: 2*halfWindowSize+1) to store computed weights.
 */
static void ComputeWeights(uint8_t halfWindowSize, uint16_t targetPoint, uint8_t polynomialOrder, uint8_t derivativeOrder, float* weights) {
    // Create a GramPolyContext with the current filter parameters.
    GramPolyContext ctx = { halfWindowSize, targetPoint, derivativeOrder };

    // Calculate the full window size (total number of data points in the filter window).
    uint16_t fullWindowSize = 2 * halfWindowSize + 1;

#ifdef ENABLE_MEMOIZATION
    // Clear the memoization cache to ensure that previous values do not interfere
    // with the current computation. This is necessary when filter parameters change.
    ClearGramPolyCache(halfWindowSize, polynomialOrder, derivativeOrder);
#endif

    // Loop over each index in the filter window using AVX for batches of 8 elements.
    int i;
    for (i = 0; i <= fullWindowSize - 8; i += 8) {
        // Prepare an array of 8 shifted data indices for vectorized weight computation.
        // Shift each index so that the center of the window corresponds to 0, making
        // the weight calculation symmetric around the center.
        int dataIndices[8] = {
            i - halfWindowSize, i + 1 - halfWindowSize, i + 2 - halfWindowSize, i + 3 - halfWindowSize,
            i + 4 - halfWindowSize, i + 5 - halfWindowSize, i + 6 - halfWindowSize, i + 7 - halfWindowSize
        };
        __m256 w; // 256-bit vector to store 8 weights.
        // Compute 8 weights simultaneously using a vectorized version of the Weight function.
        WeightVectorized(dataIndices, targetPoint, polynomialOrder, &ctx, &w);
        // Store the 8 computed weights into the weights array.
        _mm256_store_ps(&weights[i], w);
    }

    // Scalar remainder: Handle any leftover indices not processed by AVX.
    for (; i < fullWindowSize; ++i) {
        // Shift the dataIndex so that the center of the window corresponds to 0.
        // This makes the weight calculation symmetric around the center.
        weights[i] = Weight(i - halfWindowSize, targetPoint, polynomialOrder, &ctx);
    }
}

//-------------------------
// Filter Initialization
//-------------------------
/**
 * @brief Initializes the Savitzky–Golay filter structure.
 *
 * @param halfWindowSize Half-window size.
 * @param polynomialOrder Order of the polynomial.
 * @param targetPoint Target point within the window.
 * @param derivativeOrder Order of the derivative (0 for smoothing).
 * @param time_step Time step value.
 * @return An initialized SavitzkyGolayFilter structure.
 */
SavitzkyGolayFilter initFilter(uint8_t halfWindowSize, uint8_t polynomialOrder, uint8_t targetPoint, uint8_t derivativeOrder, float time_step) {
    SavitzkyGolayFilter filter;
    filter.conf.halfWindowSize = halfWindowSize;
    filter.conf.polynomialOrder = polynomialOrder;
    filter.conf.targetPoint = targetPoint;
    filter.conf.derivativeOrder = derivativeOrder;
    filter.conf.time_step = time_step;
    filter.dt = pow(time_step, derivativeOrder);
    return filter;
}

//-------------------------
// Filter Application
//-------------------------

/**
 * @brief Applies the Savitzky–Golay filter to the input data with SIMD optimization.
 *
 * The Savitzky–Golay filter performs smoothing (or differentiation) by computing a
 * weighted convolution of the input data. The weights are derived from Gram polynomials,
 * ensuring that the filter performs a least-squares fit over a moving window.
 *
 * **Mathematical Background:**
 * Given a window of data points and corresponding weights \(w_j\) (computed from
 * Gram polynomials), the filtered value at a central point is given by:
 *
 * \[
 * y_{\text{filtered}} = \sum_{j=0}^{N-1} w_j \cdot x_{i+j}
 * \]
 *
 * where \(N = 2 \times \text{halfWindowSize} + 1\) is the window size.
 *
 * For the border cases (leading and trailing edges), mirror padding is applied. This
 * means that the data is reflected at the edges to compensate for missing values, ensuring
 * that the convolution can still be applied.
 *
 * @param data Array of input data points.
 * @param dataSize Number of data points in the input array.
 * @param halfWindowSize Half-window size (thus, filter window size = \(2 \times \text{halfWindowSize} + 1\)).
 * @param targetPoint The target point within the window where the fit is evaluated.
 * @param filter The SavitzkyGolayFilter structure containing configuration parameters.
 * @param filteredData Array to store the filtered data points.
 */
static void ApplyFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize, uint16_t targetPoint, SavitzkyGolayFilter filter, MqsRawDataPoint_t filteredData[]) {
    // Validate and adjust halfWindowSize to ensure it doesn't exceed the maximum allowed window size.
    // Decision: Cap halfWindowSize to prevent buffer overflows in weights arrays.
    uint8_t maxHalfWindowSize = (MAX_WINDOW - 1) / 2;
    if (halfWindowSize > maxHalfWindowSize) {
        printf("Warning: halfWindowSize (%d) exceeds maximum allowed (%d). Adjusting.\n", halfWindowSize, maxHalfWindowSize);
        halfWindowSize = maxHalfWindowSize;
    }

    // Compute window parameters: full window size (N = 2m + 1), last data index, and center offset (m).
    int windowSize = 2 * halfWindowSize + 1;
    int lastIndex = dataSize - 1;
    uint8_t width = halfWindowSize;

    // Step 1: Precompute weights for the central region once, stored in an aligned array for SIMD.
    // Decision: Use a static array to avoid recomputing weights for each central position, improving efficiency.
    ALIGNED static float centralWeights[MAX_WINDOW];
    ComputeWeights(halfWindowSize, targetPoint, filter.conf.polynomialOrder, filter.conf.derivativeOrder, centralWeights);

    // Step 2: Apply the filter to central data points (where a full window is available).
    // Flow: Iterate over all positions where the window fits within dataSize, computing the convolution.
    for (int i = 0; i <= (int)dataSize - windowSize; ++i) {
        float sum = 0.0f; // Accumulator for the weighted sum.
        int j;

        // Vectorization (AVX): Process 8 elements at a time using 256-bit registers for performance.
        // - Load 8 weights and 8 data points, multiply element-wise, and sum horizontally.
        // - Alignment: centralWeights is aligned (load_ps), data may not be (loadu_ps).
        // Decision: Use AVX when available to exploit parallelism, falling back to SSE or scalar if needed.
#if defined(__AVX__)
        for (j = 0; j <= windowSize - 8; j += 8) {
            __m256 w = _mm256_load_ps(&centralWeights[j]); // Load 8 weights into a 256-bit vector.
            __m256 d = _mm256_loadu_ps(&data[i + j].phaseAngle); // Load 8 unaligned data points.
            __m256 prod = _mm256_mul_ps(w, d); // Element-wise multiplication: w[j+k] * d[i+j+k], k=0..7.
            // Horizontal sum: Reduce 8 products to 1 scalar value.
            __m128 hi = _mm256_extractf128_ps(prod, 1); // Upper 4 elements.
            __m128 lo = _mm256_castps256_ps128(prod);   // Lower 4 elements.
            __m128 sum128 = _mm_add_ps(hi, lo);         // Sum pairs: 4 elements.
            sum128 = _mm_hadd_ps(sum128, sum128);       // Sum pairs again: 2 elements.
            sum128 = _mm_hadd_ps(sum128, sum128);       // Final sum: 1 element.
            sum += _mm_cvtss_f32(sum128);               // Add to accumulator.
        }
#endif

        // Vectorization (SSE): Process 4 elements at a time using 128-bit registers for remaining elements.
        // - Similar to AVX but with fewer elements per iteration.
        // Decision: Use SSE for smaller chunks or when AVX isn’t available, ensuring broad compatibility.
        for (; j <= windowSize - 4; j += 4) {
            __m128 w = _mm_load_ps(&centralWeights[j]); // Load 4 aligned weights.
            __m128 d = _mm_loadu_ps(&data[i + j].phaseAngle); // Load 4 unaligned data points.
            __m128 prod = _mm_mul_ps(w, d); // Element-wise multiplication: w[j+k] * d[i+j+k], k=0..3.
            // Horizontal sum: Reduce 4 products to 1 scalar.
            prod = _mm_hadd_ps(prod, prod); // Sum pairs: 2 elements.
            prod = _mm_hadd_ps(prod, prod); // Final sum: 1 element.
            sum += _mm_cvtss_f32(prod);     // Add to accumulator.
        }

        // Scalar remainder: Handle any leftover elements not divisible by 4 or 8.
        // Decision: Ensure correctness by processing all elements, even if vectorization doesn’t cover them.
        for (; j < windowSize; ++j) {
            sum += centralWeights[j] * data[i + j].phaseAngle;
        }

        // Store the result at the center of the window (i + width).
        // Decision: Omit filter.dt to match the original scalar implementation’s behavior.
        filteredData[i + width].phaseAngle = sum;
    }

    // Step 3: Handle edge cases (leading and trailing edges) using mirror padding.
    // Flow: For positions where a full window isn’t available, mirror data and apply weights.
    ALIGNED static float weights[MAX_WINDOW]; // Reusable weights array for edge computations.
    ALIGNED float tempWindow[MAX_WINDOW];     // Temporary buffer for mirrored data, aligned for SIMD.

    for (int i = 0; i < width; ++i) {
        int j;

        // --- Leading Edge ---
        // Compute weights with targetPoint shifting toward the start (width - i).
        // Decision: Adjust targetPoint per position to fit the polynomial at the edge, mirroring scalar logic.
        ComputeWeights(halfWindowSize, width - i, filter.conf.polynomialOrder, filter.conf.derivativeOrder, weights);
        float leadingSum = 0.0f;

        // Fill tempWindow with mirrored data: reflect points around index 0.
        // - Indices go from windowSize-1 down to 0 (e.g., for N=5: 4, 3, 2, 1, 0).
        for (j = 0; j < windowSize; ++j) {
            int dataIdx = windowSize - j - 1;
            tempWindow[j] = data[dataIdx].phaseAngle;
        }

        // Vectorization (AVX): Process 8 mirrored elements at a time.
        // - tempWindow is aligned, allowing load_ps instead of loadu_ps for better performance.
#if defined(__AVX__)
        for (j = 0; j <= windowSize - 8; j += 8) {
            __m256 w = _mm256_load_ps(&weights[j]); // Load 8 weights.
            __m256 d = _mm256_load_ps(&tempWindow[j]); // Load 8 mirrored data points.
            __m256 prod = _mm256_mul_ps(w, d); // Element-wise multiplication.
            __m128 hi = _mm256_extractf128_ps(prod, 1);
            __m128 lo = _mm256_castps256_ps128(prod);
            __m128 sum128 = _mm_add_ps(hi, lo);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            leadingSum += _mm_cvtss_f32(sum128);
        }
#endif

        // Vectorization (SSE): Process 4 mirrored elements at a time.
        for (; j <= windowSize - 4; j += 4) {
            __m128 w = _mm_load_ps(&weights[j]); // Load 4 weights.
            __m128 d = _mm_load_ps(&tempWindow[j]); // Load 4 mirrored data points.
            __m128 prod = _mm_mul_ps(w, d); // Element-wise multiplication.
            prod = _mm_hadd_ps(prod, prod);
            prod = _mm_hadd_ps(prod, prod);
            leadingSum += _mm_cvtss_f32(prod);
        }

        // Scalar remainder for leading edge.
        for (; j < windowSize; ++j) {
            leadingSum += weights[j] * tempWindow[j];
        }
        filteredData[i].phaseAngle = leadingSum; // No filter.dt to match scalar.

        // --- Trailing Edge ---
        // Reuse weights from the last leading edge iteration (targetPoint = 1 when i = width - 1).
        float trailingSum = 0.0f;

        // Fill tempWindow with mirrored data: use points from lastIndex - windowSize + 1 to lastIndex.
        // - Indices go from lastIndex - N + 1 to lastIndex (e.g., for N=5, lastIndex=9: 5, 6, 7, 8, 9).
        for (j = 0; j < windowSize; ++j) {
            int dataIdx = lastIndex - windowSize + j + 1;
            tempWindow[j] = data[dataIdx].phaseAngle;
        }

        // Vectorization (AVX) for trailing edge.
#if defined(__AVX__)
        for (j = 0; j <= windowSize - 8; j += 8) {
            __m256 w = _mm256_load_ps(&weights[j]); // Reuse last leading edge weights.
            __m256 d = _mm256_load_ps(&tempWindow[j]); // Load 8 mirrored data points.
            __m256 prod = _mm256_mul_ps(w, d);
            __m128 hi = _mm256_extractf128_ps(prod, 1);
            __m128 lo = _mm256_castps256_ps128(prod);
            __m128 sum128 = _mm_add_ps(hi, lo);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            trailingSum += _mm_cvtss_f32(sum128);
        }
#endif

        // Vectorization (SSE) for trailing edge.
        for (; j <= windowSize - 4; j += 4) {
            __m128 w = _mm_load_ps(&weights[j]);
            __m128 d = _mm_load_ps(&tempWindow[j]);
            __m128 prod = _mm_mul_ps(w, d);
            prod = _mm_hadd_ps(prod, prod);
            prod = _mm_hadd_ps(prod, prod);
            trailingSum += _mm_cvtss_f32(prod);
        }

        // Scalar remainder for trailing edge.
        for (; j < windowSize; ++j) {
            trailingSum += weights[j] * tempWindow[j];
        }
        filteredData[lastIndex - i].phaseAngle = trailingSum; // No filter.dt
    }
}


//-------------------------
// Main Filter Function with Error Handling
//-------------------------
/**
 * @brief Applies the Savitzky–Golay filter to a data sequence.
 *
 * Performs error checking on the parameters, initializes the filter, and calls ApplyFilter().
 *
 * @param data Array of raw data points (input).
 * @param dataSize Number of data points.
 * @param halfWindowSize Half-window size for the filter.
 * @param filteredData Array to store the filtered data points (output).
 * @param polynomialOrder Polynomial order used for the filter.
 * @param targetPoint The target point within the window.
 * @param derivativeOrder Derivative order (0 for smoothing).
 */
int mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                     MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder,
                     uint8_t targetPoint, uint8_t derivativeOrder) {
    // Assertions for development to catch invalid parameters early.
    assert(data != NULL && "Input data pointer must not be NULL");
    assert(filteredData != NULL && "Filtered data pointer must not be NULL");
    assert(dataSize > 0 && "Data size must be greater than 0");
    assert(halfWindowSize > 0 && "Half-window size must be greater than 0");
    assert((2 * halfWindowSize + 1) <= dataSize && "Filter window size must not exceed data size");
    assert(polynomialOrder < (2 * halfWindowSize + 1) && "Polynomial order must be less than the filter window size");
    assert(targetPoint <= (2 * halfWindowSize) && "Target point must be within the filter window");
    
    // Runtime checks with error logging.
    if (data == NULL || filteredData == NULL) {
        LOG_ERROR("NULL pointer passed to mes_savgolFilter.");
        return -1;
    }
    if (dataSize == 0 || halfWindowSize == 0 ||
        polynomialOrder >= 2 * halfWindowSize + 1 ||
        targetPoint > 2 * halfWindowSize ||
        (2 * halfWindowSize + 1) > dataSize) {
        LOG_ERROR("Invalid filter parameters provided: dataSize=%zu, halfWindowSize=%d, polynomialOrder=%d, targetPoint=%d.",
                  dataSize, halfWindowSize, polynomialOrder, targetPoint);
        return -2;
    }
    
    SavitzkyGolayFilter filter = initFilter(halfWindowSize, polynomialOrder, targetPoint, derivativeOrder, 1.0f);
    ApplyFilter(data, dataSize, halfWindowSize, targetPoint, filter, filteredData);
    
    return 0;
}

