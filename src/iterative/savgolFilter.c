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
static FLOAT precomputedGenFactNum[MAX_POLY_ORDER + 1];
/// Precomputed denominator factors for GenFact.
static FLOAT precomputedGenFactDen[MAX_POLY_ORDER + 1];
#endif

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
        FLOAT numProduct = ONE;
        for (uint8_t j = (upperLimitNum - k) + 1; j <= upperLimitNum; j++) {
            numProduct *= j;
        }
        precomputedGenFactNum[k] = numProduct;
        uint32_t upperLimitDen = 2 * halfWindowSize + k + 1;
        FLOAT denProduct = ONE;
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
 * @return The computed generalized factorial as a FLOAT.
 */
static inline FLOAT GenFact(uint8_t upperLimit, uint8_t termCount) {
    FLOAT product = ONE;
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
static FLOAT GramPolyIterative(uint8_t polynomialOrder, int dataIndex, const GramPolyContext* ctx) {
    // Retrieve necessary parameters from the context.
    uint8_t halfWindowSize = ctx->halfWindowSize;    // Half window size used in the filter.
    uint8_t derivativeOrder = ctx->derivativeOrder;    // Order of the derivative to compute.

    // Create a 2D array 'dp' to store intermediate Gram polynomial values.
    // dp[k][d] will store F(k, d): the Gram polynomial of order k and derivative order d.
#ifdef _MSC_VER
    // Prevent MSVC error C2057
    FLOAT** dp = (FLOAT**)malloc((polynomialOrder + 1)*sizeof(FLOAT*));
    for (int i=0; i < polynomialOrder+1; ++i) {
      dp[i] = (FLOAT*)malloc((derivativeOrder + 1)*sizeof(FLOAT));
    }
#else
    FLOAT dp[polynomialOrder + 1][derivativeOrder + 1];
#endif

    // Base case: k = 0.
    // For the zeroth order, the polynomial is 1 when derivative order is 0, and 0 for d > 0.
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        dp[0][d] = (d == 0) ? ONE : ZERO;
    }
    // If the requested polynomial order is 0, return the base case directly.
    if (polynomialOrder == 0) {
        return dp[0][derivativeOrder];
    }

    // k = 1: Compute first order polynomial values using the base case.
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        // The formula for F(1, d) uses the base value F(0, d) and, if needed, the derivative of F(0, d-1).
        dp[1][d] = (ONE / halfWindowSize) * (dataIndex * dp[0][d] + (d > 0 ? d * dp[0][d - 1] : 0));
    }

    // Iteratively compute F(k, d) for k >= 2.
    // The recurrence relation uses previously computed values for orders k-1 and k-2.
    for (uint8_t k = 2; k <= polynomialOrder; k++) {
        // Compute constants 'a' and 'c' for the recurrence:
        // a = (4k - 2) / [k * (2*halfWindowSize - k + 1)]
        // c = [(k - 1) * (2*halfWindowSize + k)] / [k * (2*halfWindowSize - k + 1)]
        FLOAT a = (FOUR * k - TWO) / (k * (TWO * halfWindowSize - k + ONE));
        FLOAT c = ((k - ONE) * (TWO * halfWindowSize + k)) / (k * (TWO * halfWindowSize - k + ONE));

        // For each derivative order from 0 up to derivativeOrder:
        for (uint8_t d = 0; d <= derivativeOrder; d++) {
            // Start with term = dataIndex * F(k-1, d)
            FLOAT term = dataIndex * dp[k - 1][d];
            // If computing a derivative (d > 0), add the derivative term: d * F(k-1, d-1)
            if (d > 0) {
                term += d * dp[k - 1][d - 1];
            }
            // The recurrence: F(k, d) = a * (term) - c * F(k-2, d)
            dp[k][d] = a * term - c * dp[k - 2][d];
        }
    }

    // Return the computed Gram polynomial for the requested polynomial order and derivative order.
#ifdef _MSC_VER
    FLOAT gramPolynomial = dp[polynomialOrder][derivativeOrder];
    for (int i=0; i < polynomialOrder+1; ++i) {
      free(dp[i]);
    }
    free(dp);
    return gramPolynomial;
#else
    return dp[polynomialOrder][derivativeOrder];
#endif
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
    FLOAT value;
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
static FLOAT MemoizedGramPoly(uint8_t polynomialOrder, int dataIndex, const GramPolyContext* ctx) {
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
    FLOAT value = GramPolyIterative(polynomialOrder, dataIndex, ctx);
    
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
static FLOAT Weight(int dataIndex, int targetPoint, uint8_t polynomialOrder, const GramPolyContext* ctx) {
    FLOAT w = ZERO;  // Initialize weight accumulator.
    
    // Loop over polynomial orders from 0 to polynomialOrder.
    for (uint8_t k = 0; k <= polynomialOrder; ++k) {
#ifdef ENABLE_MEMOIZATION
        // If memoization is enabled, use the cached version.
        FLOAT part1 = MemoizedGramPoly(k, dataIndex, ctx);   // Evaluate at data point (derivative order = 0)
        FLOAT part2 = MemoizedGramPoly(k, targetPoint, ctx);   // Evaluate at target point (with derivative order from ctx)
#else
        // Otherwise, compute the Gram polynomial iteratively without caching.
        FLOAT part1 = GramPolyIterative(k, dataIndex, ctx);
        FLOAT part2 = GramPolyIterative(k, targetPoint, ctx);
#endif

#ifdef OPTIMIZE_GENFACT
        // If optimized GenFact is enabled, use precomputed numerator/denominator.
        FLOAT factor = (2 * k + 1) * (precomputedGenFactNum[k] / precomputedGenFactDen[k]);
#else
        // Otherwise, compute the generalized factorial ratio on the fly.
        FLOAT factor = (2 * k + 1) * (GenFact(2 * ctx->halfWindowSize, k) /
                                      GenFact(2 * ctx->halfWindowSize + k + 1, k + 1));
#endif

        // Accumulate the weighted contribution.
        w += factor * part1 * part2;
    }
    
    return w;
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
static void ComputeWeights(uint8_t halfWindowSize, uint16_t targetPoint, uint8_t polynomialOrder, uint8_t derivativeOrder, FLOAT* weights) {
    // Create a GramPolyContext with the current filter parameters.
  GramPolyContext ctx = { halfWindowSize, (uint8_t)targetPoint, derivativeOrder };

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

    // Loop over each index in the filter window.
    for (int dataIndex = 0; dataIndex < fullWindowSize; ++dataIndex) {
        // Shift the dataIndex so that the center of the window corresponds to 0.
        // This makes the weight calculation symmetric around the center.
        weights[dataIndex] = Weight(dataIndex - halfWindowSize, targetPoint, polynomialOrder, &ctx);
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
SavitzkyGolayFilter initFilter(uint8_t halfWindowSize, uint8_t polynomialOrder, uint8_t targetPoint, uint8_t derivativeOrder, FLOAT time_step) {
    SavitzkyGolayFilter filter;
    filter.conf.halfWindowSize = halfWindowSize;
    filter.conf.polynomialOrder = polynomialOrder;
    filter.conf.targetPoint = targetPoint;
    filter.conf.derivativeOrder = derivativeOrder;
    filter.conf.time_step = time_step;
    filter.dt = (FLOAT)pow(time_step, derivativeOrder);
    return filter;
}

//-------------------------
// Filter Application
//-------------------------

/**
 * @brief Applies the Savitzky–Golay filter to the input data.
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
    // Ensure that the halfWindowSize does not exceed the maximum allowed value.
    uint8_t maxHalfWindowSize = (MAX_WINDOW - 1) / 2;
    if (halfWindowSize > maxHalfWindowSize) {
        printf("Warning: halfWindowSize (%d) exceeds maximum allowed (%d). Adjusting.\n", halfWindowSize, maxHalfWindowSize);
        halfWindowSize = maxHalfWindowSize;
    }
    
    // Calculate the total number of points in the filter window.
    int windowSize = 2 * halfWindowSize + 1;
    int lastIndex = (int)dataSize - 1;
    uint8_t width = halfWindowSize;  // Number of points on either side of the center.
    
    // Declare an array to hold the computed weights for the window.
    static FLOAT weights[MAX_WINDOW];

    // Step 1: Compute weights for the central window.
    // The weights are computed based on the filter's polynomial and derivative orders.
    ComputeWeights(halfWindowSize, targetPoint, filter.conf.polynomialOrder, filter.conf.derivativeOrder, weights);

    // Step 2: Apply the filter to the central data points using convolution.
    // For each valid window position, multiply each data point by its corresponding weight
    // and sum the results.
    for (int i = 0; i <= (int)dataSize - windowSize; ++i) {
        FLOAT sum = ZERO;
        for (int j = 0; j < windowSize; ++j) {
            sum += weights[j] * data[i + j].phaseAngle;
        }
        // The filtered value is placed at the center of the current window.
        filteredData[i + width].phaseAngle = sum;
    }

    // Step 3: Handle edge cases using mirror padding.
    // At the beginning and end of the data array, a full window is not available.
    // Mirror padding reflects the data about the edge, creating a virtual window.
    for (int i = 0; i < width; ++i) {
        // --- Leading Edge ---
        // For the leading edge, re-compute the weights for a window with a target
        // shifted towards the beginning (i.e., mirror the data).
        ComputeWeights(halfWindowSize, width - i, filter.conf.polynomialOrder, filter.conf.derivativeOrder, weights);
        FLOAT leadingSum = ZERO;
        // Apply the weights to the mirrored segment of the data.
        for (int j = 0; j < windowSize; ++j) {
            leadingSum += weights[j] * data[windowSize - j - 1].phaseAngle;
        }
        filteredData[i].phaseAngle = leadingSum;

        // --- Trailing Edge ---
        // For the trailing edge, mirror padding is similarly applied.
        FLOAT trailingSum = ZERO;
        for (int j = 0; j < windowSize; ++j) {
            trailingSum += weights[j] * data[lastIndex - windowSize + j + 1].phaseAngle;
        }
        filteredData[lastIndex - i].phaseAngle = trailingSum;
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
    
    SavitzkyGolayFilter filter = initFilter(halfWindowSize, polynomialOrder, targetPoint, derivativeOrder, ONE);
    ApplyFilter(data, dataSize, halfWindowSize, targetPoint, filter, filteredData);
    
    return 0;
}

