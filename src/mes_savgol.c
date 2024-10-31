#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "mes_savgol.h"

/**
 * @file mes_savgol.c
 * @brief Optimized Savitzky-Golay Filter Implementation with Recursive Calculation and Memoization.
 *
 * Author: Tugbars Heptaskin
 * Date: 31/10/2024
 * Company: Aminic Aps
 *
 * This implementation provides an efficient and optimized version of the Savitzky-Golay filter,
 * commonly used for smoothing and differentiating data. The key enhancements in this implementation
 * include the use of global variables to minimize stack footprint and memoization to reduce
 * redundant recursive computations.
 */

#define MAX_WINDOW 33

// Global counter for GramPoly calls
int gramPolyCallCount = 0;

// Global values to minimize the stack footprint of GramPoly
static int g_dataIndex;
static uint8_t g_halfWindowSize;
static uint8_t g_targetPoint = 0;
static uint8_t g_derivativeOrder;

// Global MemoizationContext
static MemoizationContext context;

//#define OPTIMIZE_GENFACT

// Precomputed GenFact values
#ifdef OPTIMIZE_GENFACT
#define MAX_POLY_ORDER 4
static float precomputedGenFactNum[MAX_POLY_ORDER + 1];
static float precomputedGenFactDen[MAX_POLY_ORDER + 1];
#endif

/**
 * @brief Combines dataIndex and polynomialOrder into a single key for memoization.
 *
 * This function merges a 16-bit data index and an 8-bit polynomial order into a single
 * 32-bit key. This key is then used in the hash function for memoization purposes.
 *
 * @param dataIndex The data index.
 * @param polynomialOrder The polynomial order.
 * @return The combined 32-bit key.
 */
static inline uint32_t combineKey(uint16_t dataIndex, uint8_t polynomialOrder) {
    return ((uint32_t)dataIndex << 8) | polynomialOrder;
}

/**
 * @brief Generates a hash value for a given key using a multiplicative hash function or a simple modulo.
 *
 * This function generates a hash value based on the defined hashing strategy.
 * By default, it uses a simple modulo-based hash. If `USE_ADVANCED_HASH` is defined,
 * it employs a multiplicative hash function to achieve a more uniform distribution.
 *
 * Users can select the desired hash function by defining or undefining the `USE_ADVANCED_HASH` macro.
 *
 * @param key The combined 32-bit key.
 * @param tableSize The size of the hash table.
 * @return The generated hash value.
 */
static inline unsigned int hashGramPolyKey(uint32_t key, int tableSize) {
#ifdef USE_ADVANCED_HASH
    /* Advanced Multiplicative Hash Function */
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = (key >> 16) ^ key;
    return key % tableSize;
#else
    /* Lightweight Modulo-Based Hash Function */
    return key % tableSize;
#endif
}

/**
 * @brief Initialize the memoization table.
 *
 * This function initializes the memoization table used for caching the results of
 * Gram Polynomial calculations. Each entry in the table is marked as unoccupied,
 * and the total number of entries is set to zero.
 */
static void initializeMemoizationTable() {
    for (int i = 0; i < MAX_ENTRIES; i++) {
        context.memoizationTable[i].isOccupied = false;
    }
    context.totalHashMapEntries = 0;
}

/**
 * @brief Calculates the value of a Gram Polynomial or its derivative at a specific index within a data window.
 *
 * The Gram polynomials are orthogonal polynomials used in the Savitzky-Golay filter to ensure
 * that the integral (or sum in discrete cases) of the product of any two polynomials of different
 * orders over the window is zero. This property is essential for minimizing distortion during
 * data smoothing and differentiation.
 *
 * The function uses recursion to compute the Gram polynomial values, with memoization to
 * avoid redundant calculations.
 *
 * @param polynomialOrder The order of the polynomial (k) to compute.
 * @return The value of the Gram Polynomial or its specified derivative at the given index.
 */
static float GramPoly(uint8_t polynomialOrder) {
    gramPolyCallCount++;

    uint8_t halfWindowSize = g_halfWindowSize;
    uint8_t derivativeOrder = g_derivativeOrder;
    int dataIndex = g_dataIndex;

    if (polynomialOrder == 0) {
        return (derivativeOrder == 0) ? 1.0f : 0.0f;
    }

    float a = (4.0f * polynomialOrder - 2.0f) / (polynomialOrder * (2.0f * halfWindowSize - polynomialOrder + 1.0f));
    float b = 0.0f;
    float c = ((polynomialOrder - 1.0f) * (2.0f * halfWindowSize + polynomialOrder)) / (polynomialOrder * (2.0f * halfWindowSize - polynomialOrder + 1.0f));

    if (polynomialOrder >= 2) {
        // Recursive calculation for higher-order polynomials
        b += dataIndex * GramPoly(polynomialOrder - 1);

        if (derivativeOrder > 0) {
            // Adjust derivative order for recursion
            g_derivativeOrder = derivativeOrder - 1;
            b += derivativeOrder * GramPoly(polynomialOrder - 1);
            g_derivativeOrder = derivativeOrder;
        }

        return a * b - c * GramPoly(polynomialOrder - 2);
    } else if (polynomialOrder == 1) {
        // Base case for first-order polynomial
        a = (2.0f) / (2.0f * halfWindowSize);
        b += dataIndex * GramPoly(0);
        if (derivativeOrder > 0) {
            // Adjust derivative order for recursion
            g_derivativeOrder = derivativeOrder - 1;
            b += derivativeOrder * GramPoly(0);
            g_derivativeOrder = derivativeOrder;
        }
        return a * b;
    }

    return 0.0f;
}

/**
 * @brief Memoization wrapper for the GramPoly function.
 *
 * This function serves as a memoization wrapper for the GramPoly function,
 * aiming to optimize the computational efficiency by storing and reusing
 * previously calculated values. It uses a hash table for memoization with
 * linear probing to handle collisions.
 *
 * The function calculates a hash index for the given Gram Polynomial parameters
 * (data index and polynomial order). If the calculated polynomial value for these
 * parameters is already stored in the memoization table, it returns the stored value.
 * Otherwise, it calculates the value using the GramPoly function, stores it in the table,
 * and then returns the value. This significantly reduces the number of recursive calls
 * to GramPoly, especially when the same polynomial values are needed multiple times.
 *
 * @param polynomialOrder The order of the polynomial.
 * @param ctx Pointer to the GramPolyContext containing necessary parameters.
 * @param memoCtx Pointer to the MemoizationContext for caching.
 * @return The memoized value of the Gram Polynomial or its derivative.
 */
static float memoizedGramPoly(uint8_t polynomialOrder) {
    if (context.totalHashMapEntries >= MAX_ENTRIES) {
        // Memoization table is full; compute directly.
        return GramPoly(polynomialOrder);
    }

    uint32_t key = combineKey(g_dataIndex, polynomialOrder);
    unsigned int hashIndex = hashGramPolyKey(key, MAX_ENTRIES);

    while (context.memoizationTable[hashIndex].isOccupied) {
        if (context.memoizationTable[hashIndex].key == key) {
            // Found a cached value.
            return context.memoizationTable[hashIndex].value;
        }
        // Linear probing in case of collision.
        hashIndex = (hashIndex + 1) % MAX_ENTRIES;
    }

    // Compute and store the new value.
    float value = GramPoly(polynomialOrder);

    context.memoizationTable[hashIndex].key = key;
    context.memoizationTable[hashIndex].value = value;
    context.memoizationTable[hashIndex].isOccupied = true;
    context.totalHashMapEntries++;

    return value;
}

/**
 * @brief Calculate a generalized factorial product for use in polynomial coefficient normalization.
 *
 * This function computes the product of a sequence of integers, representing a generalized factorial.
 * It is used in the calculation of normalization factors for the Gram polynomials.
 *
 * @param upperLimit The upper boundary of the range (inclusive) for the product calculation.
 * @param termCount The number of consecutive integers from `upperLimit` to include in the product.
 * @return The product of the sequence as a float.
 */
static float GenFact(uint8_t upperLimit, uint8_t termCount) {
#ifdef OPTIMIZE_GENFACT
    // Use precomputed values
    if (termCount == 0) return 1.0f;
    return precomputedGenFactNum[termCount];
#else
    float product = 1.0f;
    for (uint8_t j = (upperLimit - termCount) + 1; j <= upperLimit; j++) {
        product *= j;
    }
    return product;
#endif
}

/**
 * @brief Computes the weight of a specific data point within a window for least-squares polynomial fitting.
 *
 * This function calculates the weight of a data point within the window of the Savitzky-Golay filter.
 * The weight is computed based on the Gram Polynomial values and their derivatives, which are memoized
 * to enhance computational efficiency.
 *
 * @param dataIndex Index of the data point within the window for which the weight is being calculated.
 * @param targetPoint The index within the dataset at which the least-squares fit is evaluated.
 * @param polynomialOrder The order of the polynomial used in the least-squares fitting process.
 * @return The calculated weight for the data point at `dataIndex`.
 */
static float Weight(int dataIndex, int targetPoint, uint8_t polynomialOrder) {
    float w = 0.0f;
    uint8_t derivativeOrder = g_derivativeOrder;

    for (uint8_t k = 0; k <= polynomialOrder; ++k) {
        // Compute Gram Polynomial at dataIndex
        g_dataIndex = dataIndex;
        g_derivativeOrder = 0;
        float part1 = memoizedGramPoly(k);

        // Compute Gram Polynomial (or its derivative) at targetPoint
        g_derivativeOrder = derivativeOrder;
        g_dataIndex = targetPoint;
        float part2 = memoizedGramPoly(k);

        // Accumulate the weighted contribution
#ifdef OPTIMIZE_GENFACT
        float factor = (2 * k + 1) * (precomputedGenFactNum[k] / precomputedGenFactDen[k]);
#else
        float factor = (2 * k + 1) * (GenFact(2 * g_halfWindowSize, k) / GenFact(2 * g_halfWindowSize + k + 1, k + 1));
#endif
        w += factor * part1 * part2;
    }
    return w;
}

/**
 * @brief Precompute generalized factorials for optimization.
 *
 * This function precomputes the numerator and denominator generalized factorials used in the weight calculation.
 *
 * @param halfWindowSize The half window size of the filter.
 * @param polynomialOrder The polynomial order used in the filter.
 */
#ifdef OPTIMIZE_GENFACT
static void PrecomputeGenFacts(uint8_t halfWindowSize, uint8_t polynomialOrder) {
    uint32_t upperLimitNum = 2 * halfWindowSize;

    for (uint8_t k = 0; k <= polynomialOrder; ++k) {
        // Numerator: GenFact(2 * halfWindowSize, k)
        float numProduct = 1.0f;
        for (uint8_t j = (upperLimitNum - k) + 1; j <= upperLimitNum; j++) {
            numProduct *= j;
        }
        precomputedGenFactNum[k] = numProduct;

        // Denominator: GenFact(2 * halfWindowSize + k + 1, k + 1)
        uint32_t upperLimitDen = 2 * halfWindowSize + k + 1;
        float denProduct = 1.0f;
        for (uint8_t j = (upperLimitDen - (k + 1)) + 1; j <= upperLimitDen; j++) {
            denProduct *= j;
        }
        precomputedGenFactDen[k] = denProduct;
    }
}
#endif

/**
 * @brief Computes the weights for each data point in a specified window for Savitzky-Golay filtering.
 *
 * This function calculates the weights for all data points within the filter window based on the
 * specified half-window size, target point, polynomial order, and derivative order. These weights
 * are used in the convolution step of the Savitzky-Golay filter.
 *
 * @param halfWindowSize The half window size of the Savitzky-Golay filter.
 * @param targetPoint The point at which the least-squares fit is evaluated.
 * @param polynomialOrder The order of the polynomial used in the least-squares fit.
 * @param derivativeOrder The order of the derivative for which the weights are being calculated.
 * @param weights An array to store the calculated weights.
 */
static void ComputeWeights(uint8_t halfWindowSize, uint16_t targetPoint, uint8_t polynomialOrder, uint8_t derivativeOrder, float* weights) {
    g_halfWindowSize = halfWindowSize;
    g_derivativeOrder = derivativeOrder;
    g_targetPoint = targetPoint;
    uint16_t fullWindowSize = 2 * halfWindowSize + 1;

#ifdef OPTIMIZE_GENFACT
    // Precompute GenFact values
    PrecomputeGenFacts(halfWindowSize, polynomialOrder);
#endif

    for (int dataIndex = 0; dataIndex < fullWindowSize; ++dataIndex) {
        weights[dataIndex] = Weight(dataIndex - g_halfWindowSize, g_targetPoint, polynomialOrder);
    }
}

/**
 * @brief Applies the Savitzky-Golay filter to the given data set.
 *
 * This function applies the Savitzky-Golay smoothing filter to a data set based on
 * the provided filter configuration. It handles both central and edge cases within the data set.
 *
 * For central cases, the filter is applied using a symmetric window centered on each data point.
 * The filter weights are applied across this window to compute the smoothed value for each central data point.
 *
 * For edge cases (leading and trailing edges of the data set), the function computes
 * specific weights for each border case. These weights account for the asymmetry at the data
 * set edges. The filter is then applied to these edge cases using the respective calculated weights.
 *
 * @param data         The array of data points to which the filter is to be applied.
 * @param dataSize     The size of the data array.
 * @param filteredData The array where the filtered data points will be stored.
 * @param filter       Pointer to the SavitzkyGolayFilter structure containing filter configuration and precomputed weights.
 */
static void ApplyFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize, uint16_t targetPoint, SavitzkyGolayFilter filter, MqsRawDataPoint_t filteredData[]) {
    uint8_t maxHalfWindowSize = (MAX_WINDOW - 1) / 2;

    if (halfWindowSize > maxHalfWindowSize) {
        printf("Warning: halfWindowSize (%d) exceeds the maximum allowed value (%d). Adjusting halfWindowSize to the maximum allowed value.\n", halfWindowSize, maxHalfWindowSize);
        halfWindowSize = maxHalfWindowSize;
    }

    const int window = 2 * halfWindowSize + 1;
    const int endidx = dataSize - 1;
    uint8_t width = halfWindowSize;
    static float weights[MAX_WINDOW];

    // Compute the central weights
    ComputeWeights(halfWindowSize, targetPoint, filter.conf.polynomialOrder, filter.conf.derivativeOrder, weights);

    // Apply filter to central data points
    for (int i = 0; i <= dataSize - window; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < window; ++j) {
            sum += weights[j] * data[i + j].phaseAngle;
        }
        filteredData[i + width].phaseAngle = sum;
    }

    // Handle edge cases
    for (int i = 0; i < width; ++i) {
        // Leading edge
        ComputeWeights(halfWindowSize, width - i, filter.conf.polynomialOrder, filter.conf.derivativeOrder, weights);
        float leadingSum = 0.0f;
        for (int j = 0; j < window; ++j) {
            leadingSum += weights[j] * data[window - j - 1].phaseAngle;
        }
        filteredData[i].phaseAngle = leadingSum;

        // Trailing edge
        ComputeWeights(halfWindowSize, width + i, filter.conf.polynomialOrder, filter.conf.derivativeOrder, weights);
        float trailingSum = 0.0f;
        for (int j = 0; j < window; ++j) {
            trailingSum += weights[j] * data[endidx - window + j + 1].phaseAngle;
        }
        filteredData[endidx - i].phaseAngle = trailingSum;
    }
}

/**
 * @brief Initializes and configures the Savitzky-Golay filter.
 *
 * This function initializes a Savitzky-Golay filter with specified configuration parameters.
 * The filter is used for smoothing data points in a dataset and can be configured to operate
 * as either a causal filter or a non-causal filter.
 *
 * @param halfWindowSize The half window size of the filter.
 * @param polynomialOrder The polynomial order used in the filter.
 * @param targetPoint The point at which the least-squares fit is evaluated.
 * @param derivativeOrder The order of the derivative to compute.
 * @param time_step The time step between data points.
 *
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

/**
 * @brief Applies the Savitzky-Golay filter to the input data with basic error handling.
 *
 * This function applies the Savitzky-Golay smoothing filter to a data set based on
 * the provided filter configuration. It handles both central and edge cases within the data set.
 * Basic error checks ensure that input pointers are valid and that the data size is sufficient
 * for the specified filter configuration.
 *
 * @param data           The input array of data points to which the filter is to be applied.
 * @param dataSize       The number of elements in the input array.
 * @param filteredData   The output array where the filtered data points will be stored.
 * @param filter         Pointer to the SavitzkyGolayFilter structure containing filter configuration and precomputed weights.
 *
 * @note
 *   - The function assumes that `filteredData` has been allocated with at least `dataSize` elements.
 *   - If any input pointer is `NULL` or if `dataSize` is insufficient, the function will log an error message
 *     and terminate the filtering process early.
 */
void mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                      MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder,
                      uint8_t targetPoint, uint8_t derivativeOrder) {
    // Check for NULL problems
    if (data == NULL || filteredData == NULL) {
        fprintf(stderr, "Error: NULL problem detected. One or more input arrays are NULL.\n");
        return;
    }

    // Check for wrong passed argument problems
    if (dataSize == 0 || halfWindowSize == 0 ||
        polynomialOrder >= 2 * halfWindowSize + 1 ||
        targetPoint > 2 * halfWindowSize ||
        (2 * halfWindowSize + 1) > dataSize) {
        fprintf(stderr, "Error: Invalid filter parameters provided.\n");
        return;
    }

    // Proceed with filtering if inputs are valid
    gramPolyCallCount = 0;
    initializeMemoizationTable();

    // Initialize the filter configuration
    SavitzkyGolayFilter filter = initFilter(halfWindowSize, polynomialOrder, targetPoint, derivativeOrder, 1.0f);

    // Apply the filter
    ApplyFilter(data, dataSize, halfWindowSize, targetPoint, filter, filteredData);

    printf("GramPoly call count after applying filter: %d\n", gramPolyCallCount);
}
