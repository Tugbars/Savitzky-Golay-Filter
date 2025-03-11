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
 * Date: 2025-02-01
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
// Preprocessor Definitions for Optimized GenFact and Memoization
//-------------------------

#ifdef OPTIMIZE_GENFACT
/// Maximum supported polynomial order for optimized GenFact precomputation.
#define MAX_POLY_ORDER 4
/// Precomputed numerator factors for GenFact.
static float precomputedGenFactNum[MAX_POLY_ORDER + 1];
/// Precomputed denominator factors for GenFact.
static float precomputedGenFactDen[MAX_POLY_ORDER + 1];
#endif

// Uncomment the following line to enable memoization of Gram polynomial calculations.
#define ENABLE_MEMOIZATION

//-------------------------
// Optimized GenFact Precomputation
//-------------------------
#ifdef OPTIMIZE_GENFACT
/**
 * @brief Precompute generalized factorial numerators and denominators.
 *
 * This function precomputes the numerator and denominator factors for the generalized factorial
 * used in weight calculations.
 *
 * @param halfWindowSize Half-window size used in the filter.
 * @param polynomialOrder Order of the polynomial.
 */
static void PrecomputeGenFacts(uint8_t halfWindowSize, uint8_t polynomialOrder) {
    uint32_t upperLimitNum = 2 * halfWindowSize;
    for (uint8_t k = 0; k <= polynomialOrder; ++k) {
        float numProduct = 1.0f;
        for (uint8_t j = (upperLimitNum - k) + 1; j <= upperLimitNum; j++) {
            numProduct *= j;
        }
        precomputedGenFactNum[k] = numProduct;
        uint32_t upperLimitDen = 2 * halfWindowSize + k + 1;
        float denProduct = 1.0f;
        for (uint8_t j = (upperLimitDen - (k + 1)) + 1; j <= upperLimitDen; j++) {
            denProduct *= j;
        }
        precomputedGenFactDen[k] = denProduct;
    }
}
#else
/**
 * @brief Compute the generalized factorial (GenFact) for a given term.
 *
 * @param upperLimit The upper limit of the product.
 * @param termCount The number of terms in the product.
 * @return The computed generalized factorial as a float.
 */
static inline float GenFact(uint8_t upperLimit, uint8_t termCount) {
    float product = 1.0f;
    for (uint8_t j = (upperLimit - termCount) + 1; j <= upperLimit; j++) {
        product *= j;
    }
    return product;
}
#endif

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
 * @brief Calculates the weight for a single data index in the filter window.
 *
 * This function computes the weight for a given data point by summing over Gram polynomials.
 * For each polynomial order k from 0 to polynomialOrder, it computes two parts:
 * - part1: The Gram polynomial evaluated at the data index (with derivative order 0).
 * - part2: The Gram polynomial evaluated at the target point (with the derivative order from the context).
 *
 * Each term is multiplied by a factor that involves a generalized factorial ratio.
 * Depending on preprocessor settings, the function uses either the memoized version or the
 * iterative computation directly.
 *
 * @param dataIndex The shifted data index (relative to the window center).
 * @param targetPoint The target point within the window.
 * @param polynomialOrder The order of the polynomial.
 * @param ctx Pointer to a GramPolyContext containing filter parameters.
 * @return The computed weight for the data index.
 */
static float Weight(int dataIndex, int targetPoint, uint8_t polynomialOrder, const GramPolyContext* ctx) {
    float w = 0.0f;  // Initialize weight accumulator.
    
    // Loop over polynomial orders from 0 to polynomialOrder.
    for (uint8_t k = 0; k <= polynomialOrder; ++k) {
#ifdef ENABLE_MEMOIZATION
        // If memoization is enabled, use the cached version.
        float part1 = MemoizedGramPoly(k, dataIndex, ctx);   // Evaluate at data point (derivative order = 0)
        float part2 = MemoizedGramPoly(k, targetPoint, ctx);   // Evaluate at target point (with derivative order from ctx)
#else
        // Otherwise, compute the Gram polynomial iteratively without caching.
        float part1 = GramPolyIterative(k, dataIndex, ctx);
        float part2 = GramPolyIterative(k, targetPoint, ctx);
#endif

#ifdef OPTIMIZE_GENFACT
        // If optimized GenFact is enabled, use precomputed numerator/denominator.
        float factor = (2 * k + 1) * (precomputedGenFactNum[k] / precomputedGenFactDen[k]);
#else
        // Otherwise, compute the generalized factorial ratio on the fly.
        float factor = (2 * k + 1) * (GenFact(2 * ctx->halfWindowSize, k) /
                                      GenFact(2 * ctx->halfWindowSize + k + 1, k + 1));
#endif

        // Accumulate the weighted contribution.
        w += factor * part1 * part2;
    }
    
    return w;
}

/**
 * @brief Computes Savitzky-Golay weights for eight data indices simultaneously using AVX.
 *
 * This function calculates the weights for eight data points in parallel, leveraging
 * 256-bit AVX instructions to improve performance. It computes the weight for each
 * data index by summing contributions over polynomial orders, similar to the scalar
 * Weight() function, but processes eight indices at once. The weights are derived from
 * Gram polynomials evaluated at the data indices and the target point, scaled by a
 * factor involving generalized factorials.
 *
 * The result is stored in a 256-bit vector, which can be written directly to memory
 * by the caller (e.g., in ComputeWeights()). This vectorized approach is particularly
 * efficient when the filter window size is large, allowing multiple weights to be
 * computed in a single pass.
 *
 * @param dataIndices Array of 8 data indices (shifted relative to the window center).
 * @param targetPoint The target point within the window where the fit is evaluated.
 * @param polynomialOrder The order of the polynomial used for fitting.
 * @param ctx Pointer to a GramPolyContext containing filter parameters (halfWindowSize, derivativeOrder).
 * @param weightsOut Pointer to a 256-bit vector (__m256) to store the 8 computed weights.
 */
static void WeightVectorized(int dataIndices[8], int targetPoint, uint8_t polynomialOrder, const GramPolyContext* ctx, __m256* weightsOut) {
    // Initialize a 256-bit vector to accumulate 8 weights, starting at zero.
    __m256 w = _mm256_setzero_ps();

    // Loop over polynomial orders from 0 to polynomialOrder to compute contributions.
    for (uint8_t k = 0; k <= polynomialOrder; ++k) {
        // Compute the scalar Gram polynomial value at the target point (shared across all 8 weights).
        float part2 = GramPolyIterative(k, targetPoint, ctx);

        // Compute Gram polynomial values for all 8 data indices in parallel and load into a 256-bit vector.
        // The order in _mm256_set_ps is reversed (high to low) to match AVX register layout.
        __m256 part1 = _mm256_set_ps(
            GramPolyIterative(k, dataIndices[7], ctx),
            GramPolyIterative(k, dataIndices[6], ctx),
            GramPolyIterative(k, dataIndices[5], ctx),
            GramPolyIterative(k, dataIndices[4], ctx),
            GramPolyIterative(k, dataIndices[3], ctx),
            GramPolyIterative(k, dataIndices[2], ctx),
            GramPolyIterative(k, dataIndices[1], ctx),
            GramPolyIterative(k, dataIndices[0], ctx)
        );

#ifdef OPTIMIZE_GENFACT
        // Use precomputed generalized factorial ratio for efficiency, scaled by (2k + 1).
        float factor = (2 * k + 1) * (precomputedGenFactNum[k] / precomputedGenFactDen[k]);
#else
        // Compute the generalized factorial ratio on the fly, scaled by (2k + 1).
        float factor = (2 * k + 1) * (GenFact(2 * ctx->halfWindowSize, k) / GenFact(2 * ctx->halfWindowSize + k + 1, k + 1));
#endif

        // Broadcast the scalar factor and part2 to 256-bit vectors for element-wise operations.
        __m256 factorVec = _mm256_set1_ps(factor);
        __m256 part2Vec = _mm256_set1_ps(part2);

        // Compute w += factor * part1 * part2 for all 8 weights in parallel using fused multiply-add.
        // This updates the accumulator with the contribution from the current polynomial order.
        w = _mm256_fmadd_ps(factorVec, _mm256_mul_ps(part1, part2Vec), w);
    }

    // Store the final 8 weights in the output vector for the caller to use.
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

#ifdef OPTIMIZE_GENFACT
    // Precompute the GenFact numerator and denominator factors for the current parameters.
    // This step avoids recomputation of these factors during each weight calculation.
    PrecomputeGenFacts(halfWindowSize, polynomialOrder);
#endif

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
    // Ensure halfWindowSize does not exceed the maximum allowed value, adjusting if necessary.
    uint8_t maxHalfWindowSize = (MAX_WINDOW - 1) / 2;
    if (halfWindowSize > maxHalfWindowSize) {
        printf("Warning: halfWindowSize (%d) exceeds maximum allowed (%d). Adjusting.\n", halfWindowSize, maxHalfWindowSize);
        halfWindowSize = maxHalfWindowSize;
    }

    // Calculate window size and edge indices.
    int windowSize = 2 * halfWindowSize + 1;  // Total number of points in the filter window.
    int lastIndex = dataSize - 1;             // Index of the last data point.
    uint8_t width = halfWindowSize;           // Number of points on either side of the center.

    // Step 1: Compute weights for the central window.
    ALIGNED static float weights[MAX_WINDOW]; // Aligned array to hold weights for SIMD access.
    ComputeWeights(halfWindowSize, targetPoint, filter.conf.polynomialOrder, filter.conf.derivativeOrder, weights);

    // Step 2: Apply filter to central data points using AVX.
    int i;
    for (i = 0; i <= (int)dataSize - windowSize - 7; i += 8) { // Process 8 points at a time, ensuring full window fits.
        __m256 sums = _mm256_setzero_ps(); // Initialize 256-bit vector to accumulate sums for 8 points.
        for (int j = 0; j < windowSize; j += 8) { // Process window in chunks of 8 for vectorization.
            // Load 8 weights into a 256-bit register (assumes weights are 32-byte aligned).
            __m256 w = _mm256_load_ps(&weights[j]);
            // Load 8 data points (unaligned load for safety, as data may not be aligned).
            __m256 d = _mm256_loadu_ps(&data[i + j].phaseAngle);
            // Compute weighted sum for 8 points using fused multiply-add.
            sums = _mm256_fmadd_ps(w, d, sums);
        }
        // Store the 8 filtered values at the center of each window.
        _mm256_store_ps(&filteredData[i + width].phaseAngle, sums);
    }

    // Scalar remainder for central region (handles leftover points not fitting in AVX batches).
    for (; i <= (int)dataSize - windowSize; ++i) {
        float sum = 0.0f; // Scalar accumulator for convolution.
        for (int j = 0; j < windowSize; ++j) {
            sum += weights[j] * data[i + j].phaseAngle; // Compute weighted sum.
        }
        filteredData[i + width].phaseAngle = sum; // Store result at window center.
    }

    // Step 3: Precompute weights for edge cases.
    // 2D array to store weights for each target point in edge regions (assumes MAX_WINDOW_SIZE defined).
    ALIGNED float edgeWeights[MAX_WINDOW][MAX_WINDOW];
    for (int k = 0; k < width; ++k) {
        // Compute weights for each shifted target point in the leading/trailing edges.
        ComputeWeights(halfWindowSize, width - k, filter.conf.polynomialOrder, filter.conf.derivativeOrder, edgeWeights[k]);
    }

    // --- Leading Edge ---
    // Process leading edge points with AVX, handling 8 points simultaneously.
    for (i = 0; i <= width - 8; i += 8) {
        __m256 sums = _mm256_setzero_ps(); // Accumulate sums for 8 leading edge points.
        for (int j = 0; j < windowSize; j += 8) { // Process window in chunks of 8.
            // Load weights for 8 consecutive target points into 256-bit vectors.
            __m256 w[8];
            for (int k = 0; k < 8; ++k) { // Load weights for 8 target points (i to i+7).
                w[k] = _mm256_load_ps(&edgeWeights[i + k][j]);
            }

            // Compute mirrored indices and load 8 data points into a 256-bit vector.
            int indices[8];
            for (int k = 0; k < 8 && (j + k) < windowSize; ++k) {
                indices[k] = windowSize - (j + k) - 1; // Reflect data around the start.
            }
            __m256 d = _mm256_set_ps(
                data[indices[7]].phaseAngle, data[indices[6]].phaseAngle, data[indices[5]].phaseAngle, data[indices[4]].phaseAngle,
                data[indices[3]].phaseAngle, data[indices[2]].phaseAngle, data[indices[1]].phaseAngle, data[indices[0]].phaseAngle
            );

            // Accumulate weighted sums for 8 target points.
            for (int k = 0; k < 8; ++k) {
                sums = _mm256_fmadd_ps(w[k], d, sums); // Fused multiply-add for efficiency.
            }
        }
        // Store the 8 filtered values for the leading edge.
        _mm256_store_ps(&filteredData[i].phaseAngle, sums);
    }

    // Scalar remainder for leading edge (handles remaining points < 8).
    for (; i < width; ++i) {
        float leadingSum = 0.0f; // Scalar accumulator for convolution.
        for (int j = 0; j < windowSize; ++j) {
            leadingSum += edgeWeights[i][j] * data[windowSize - j - 1].phaseAngle; // Mirror padding.
        }
        filteredData[i].phaseAngle = leadingSum; // Store result.
    }

    // --- Trailing Edge ---
    // Process trailing edge points with AVX, handling 8 points simultaneously.
    for (i = 0; i <= width - 8; i += 8) {
        __m256 sums = _mm256_setzero_ps(); // Accumulate sums for 8 trailing edge points.
        for (int j = 0; j < windowSize; j += 8) { // Process window in chunks of 8.
            // Load 8 weights from the central weights (same as central region).
            __m256 w = _mm256_load_ps(&weights[j]);

            // Compute mirrored indices and load 8 data points into a 256-bit vector.
            int indices[8];
            for (int k = 0; k < 8 && (j + k) < windowSize; ++k) {
                indices[k] = lastIndex - windowSize + (j + k) + 1; // Reflect data around the end.
            }
            __m256 d = _mm256_set_ps(
                data[indices[7]].phaseAngle, data[indices[6]].phaseAngle, data[indices[5]].phaseAngle, data[indices[4]].phaseAngle,
                data[indices[3]].phaseAngle, data[indices[2]].phaseAngle, data[indices[1]].phaseAngle, data[indices[0]].phaseAngle
            );

            // Compute weighted sum for 8 points using fused multiply-add.
            sums = _mm256_fmadd_ps(w, d, sums);
        }
        // Store the 8 filtered values for the trailing edge in reverse order.
        _mm256_store_ps(&filteredData[lastIndex - i - 7].phaseAngle, sums);
    }

    // Scalar remainder for trailing edge (handles remaining points < 8).
    for (; i < width; ++i) {
        float trailingSum = 0.0f; // Scalar accumulator for convolution.
        for (int j = 0; j < windowSize; ++j) {
            trailingSum += weights[j] * data[lastIndex - windowSize + j + 1].phaseAngle; // Mirror padding.
        }
        filteredData[lastIndex - i].phaseAngle = trailingSum; // Store result.
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

