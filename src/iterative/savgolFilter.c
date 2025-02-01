/**
 * @file savgol_filter.c
 * @brief Implementation of the Savitzky–Golay filter.
 *
 * This file provides the implementation for the Savitzky–Golay filter functions,
 * including Gram polynomial evaluation, weight calculation, and application of the filter.
 *
 * The filter uses an iterative (dynamic programming) method to compute Gram polynomials,
 * and (optionally) an optimized precomputation for generalized factorial (GenFact) values.
 *
 * @author 
 * Tugbars Heptaskin
 * @date 
 * 2025-02-01
 */

#include "savgolFilter.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//-------------------------
// Preprocessor Definitions for Optimized GenFact
//-------------------------

#ifdef OPTIMIZE_GENFACT
/// Maximum supported polynomial order for optimized GenFact precomputation.
#define MAX_POLY_ORDER 4

/// Precomputed numerator factors for GenFact.
static float precomputedGenFactNum[MAX_POLY_ORDER + 1];
/// Precomputed denominator factors for GenFact.
static float precomputedGenFactDen[MAX_POLY_ORDER + 1];
#endif

//-------------------------
// Optimized GenFact Precomputation
//-------------------------

#ifdef OPTIMIZE_GENFACT
/**
 * @brief Precompute generalized factorial numerators and denominators.
 *
 * This function precomputes the numerator and denominator factors for the generalized factorial
 * used in weight calculations. This avoids repeated multiplications in the inner loops.
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
 * Computes the product:
 * \f[
 *   \prod_{j=(upperLimit-termCount+1)}^{upperLimit} j,
 * \f]
 * which is used in the weight calculation.
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
 * This function computes the Gram polynomial \f$ F(k, d) \f$ using dynamic programming.
 * The recurrence relations incorporate the current data index and the derivative order.
 *
 * @param polynomialOrder The current polynomial order.
 * @param dataIndex The data index (can be negative if shifted so that the center is 0).
 * @param ctx Pointer to a GramPolyContext containing filter parameters.
 * @return The computed Gram polynomial value.
 */
static float GramPolyIterative(uint8_t polynomialOrder, int dataIndex, const GramPolyContext* ctx) {
    uint8_t halfWindowSize = ctx->halfWindowSize;
    uint8_t derivativeOrder = ctx->derivativeOrder;
    float dp[polynomialOrder + 1][derivativeOrder + 1];

    // Base case: k = 0. For derivative d = 0, F(0,0)=1; for d>0, F(0,d)=0.
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        dp[0][d] = (d == 0) ? 1.0f : 0.0f;
    }
    if (polynomialOrder == 0) {
        return dp[0][derivativeOrder];
    }

    // k = 1 case. Compute F(1,d) using the base case.
    for (uint8_t d = 0; d <= derivativeOrder; d++) {
        dp[1][d] = (1.0f / halfWindowSize) * (dataIndex * dp[0][d] + (d > 0 ? d * dp[0][d - 1] : 0));
    }

    // Iteratively compute F(k,d) for k >= 2.
    for (uint8_t k = 2; k <= polynomialOrder; k++) {
        float a = (4.0f * k - 2.0f) / (k * (2.0f * halfWindowSize - k + 1.0f));
        float c = ((k - 1.0f) * (2.0f * halfWindowSize + k)) / (k * (2.0f * halfWindowSize - k + 1.0f));
        for (uint8_t d = 0; d <= derivativeOrder; d++) {
            float term = dataIndex * dp[k - 1][d];
            if (d > 0) {
                term += d * dp[k - 1][d - 1];
            }
            dp[k][d] = a * term - c * dp[k - 2][d];
        }
    }
    return dp[polynomialOrder][derivativeOrder];
}

//-------------------------
// Weight Calculation Using Gram Polynomials
//-------------------------

/**
 * @brief Calculates the weight for a single data index in the filter window.
 *
 * The weight is computed by summing over Gram polynomials multiplied by a generalized factorial factor.
 *
 * @param dataIndex The shifted data index (relative to the window center).
 * @param targetPoint The target point within the window.
 * @param polynomialOrder The order of the polynomial.
 * @param ctx Pointer to a GramPolyContext with filter parameters.
 * @return The computed weight for the data index.
 */
static float Weight(int dataIndex, int targetPoint, uint8_t polynomialOrder, const GramPolyContext* ctx) {
    float w = 0.0f;
    // Sum over k = 0 to polynomialOrder.
    for (uint8_t k = 0; k <= polynomialOrder; ++k) {
        float part1 = GramPolyIterative(k, dataIndex, ctx);   // Evaluate at data point (with d = 0)
        float part2 = GramPolyIterative(k, targetPoint, ctx);   // Evaluate at target point (with derivative order in ctx)
#ifdef OPTIMIZE_GENFACT
        float factor = (2 * k + 1) * (precomputedGenFactNum[k] / precomputedGenFactDen[k]);
#else
        float factor = (2 * k + 1) * (GenFact(2 * ctx->halfWindowSize, k) /
                                      GenFact(2 * ctx->halfWindowSize + k + 1, k + 1));
#endif
        w += factor * part1 * part2;
    }
    return w;
}

//-------------------------
// Compute Filter Weights for the Window
//-------------------------

/**
 * @brief Computes the Savitzky–Golay weights for the entire filter window.
 *
 * This function computes the weight for every data point in the window by calling the
 * Weight() function. The data index is shifted so that the center of the window is 0.
 *
 * @param halfWindowSize Half-window size.
 * @param targetPoint The target point in the window.
 * @param polynomialOrder Polynomial order for fitting.
 * @param derivativeOrder Derivative order for the filter.
 * @param weights Array of size (2*halfWindowSize+1) to store computed weights.
 */
static void ComputeWeights(uint8_t halfWindowSize, uint16_t targetPoint, uint8_t polynomialOrder, uint8_t derivativeOrder, float* weights) {
    GramPolyContext ctx = { halfWindowSize, targetPoint, derivativeOrder };
    uint16_t fullWindowSize = 2 * halfWindowSize + 1;

#ifdef OPTIMIZE_GENFACT
    PrecomputeGenFacts(halfWindowSize, polynomialOrder);
#endif

    for (int dataIndex = 0; dataIndex < fullWindowSize; ++dataIndex) {
        // Shift dataIndex so that the center of the window is 0.
        weights[dataIndex] = Weight(dataIndex - halfWindowSize, targetPoint, polynomialOrder, &ctx);
    }
}

//-------------------------
// Filter Initialization
//-------------------------

/**
 * @brief Initializes the Savitzky–Golay filter structure.
 *
 * This function sets up the filter configuration parameters and computes a scaling factor
 * based on the derivative order (i.e., dt = time_step^derivativeOrder).
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
 * @brief Applies the Savitzky–Golay filter to the input data.
 *
 * The filter is applied to the central data points using precomputed weights. For the
 * leading and trailing edges, mirror padding is used and the weights are recomputed.
 *
 * @param data Array of input data points.
 * @param dataSize Number of data points in the input array.
 * @param halfWindowSize Half-window size (filter window size = 2*halfWindowSize+1).
 * @param targetPoint The target point within the window.
 * @param filter The SavitzkyGolayFilter structure containing configuration parameters.
 * @param filteredData Array to store the filtered data points.
 */
static void ApplyFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize, uint16_t targetPoint, SavitzkyGolayFilter filter, MqsRawDataPoint_t filteredData[]) {
    uint8_t maxHalfWindowSize = (MAX_WINDOW - 1) / 2;

    if (halfWindowSize > maxHalfWindowSize) {
        printf("Warning: halfWindowSize (%d) exceeds maximum allowed (%d). Adjusting.\n", halfWindowSize, maxHalfWindowSize);
        halfWindowSize = maxHalfWindowSize;
    }

    int windowSize = 2 * halfWindowSize + 1;
    int lastIndex = dataSize - 1;
    uint8_t width = halfWindowSize;
    static float weights[MAX_WINDOW];

    // Compute weights for the central window.
    ComputeWeights(halfWindowSize, targetPoint, filter.conf.polynomialOrder, filter.conf.derivativeOrder, weights);

    // Apply filter to central data points.
    for (int i = 0; i <= (int)dataSize - windowSize; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < windowSize; ++j) {
            sum += weights[j] * data[i + j].phaseAngle;
        }
        filteredData[i + width].phaseAngle = sum;
    }

    // Handle edge cases: leading and trailing edges.
    for (int i = 0; i < width; ++i) {
        // Leading edge: mirror padding and recompute weights.
        ComputeWeights(halfWindowSize, width - i, filter.conf.polynomialOrder, filter.conf.derivativeOrder, weights);
        float leadingSum = 0.0f;
        for (int j = 0; j < windowSize; ++j) {
            leadingSum += weights[j] * data[windowSize - j - 1].phaseAngle;
        }
        filteredData[i].phaseAngle = leadingSum;

        // Trailing edge: mirror padding for the trailing segment.
        float trailingSum = 0.0f;
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
 * @brief Entry point to apply the Savitzky–Golay filter to a data sequence.
 *
 * This function performs error checking on the parameters, initializes the filter, and calls the
 * ApplyFilter() function to produce filtered data.
 *
 * @param data Array of raw data points (input).
 * @param dataSize Number of data points in the array.
 * @param halfWindowSize Half-window size for the filter.
 * @param filteredData Array to store the filtered data points (output).
 * @param polynomialOrder Polynomial order used for the filter.
 * @param targetPoint The target point within the window.
 * @param derivativeOrder Derivative order (0 for smoothing).
 */
void mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                      MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder,
                      uint8_t targetPoint, uint8_t derivativeOrder) {
    if (data == NULL || filteredData == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to mes_savgolFilter.\n");
        return;
    }

    if (dataSize == 0 || halfWindowSize == 0 ||
        polynomialOrder >= 2 * halfWindowSize + 1 ||
        targetPoint > 2 * halfWindowSize ||
        (2 * halfWindowSize + 1) > dataSize) {
        fprintf(stderr, "Error: Invalid filter parameters provided.\n");
        return;
    }

    SavitzkyGolayFilter filter = initFilter(halfWindowSize, polynomialOrder, targetPoint, derivativeOrder, 1.0f);
    ApplyFilter(data, dataSize, halfWindowSize, targetPoint, filter, filteredData);
}



