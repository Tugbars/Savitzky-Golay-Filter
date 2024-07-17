/*!
 * Optimized Savitzky-Golay Filter Implementation with Recursive Calculation and Memoization.
 *
 * Author: Tugbars Heptaskin
 * Date: 07/17/2024
 * Company: Aminic Aps
 *
 * This implementation provides an efficient and optimized version of the Savitzky-Golay filter,
 * commonly used for smoothing and differentiating data. The key enhancements in this implementation
 * include the use of global variables to minimize stack footprint and memoization to reduce
 * redundant recursive computations.
 *
 */
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "mes_savgol.h"

// Global counter for GramPoly calls
int gramPolyCallCount = 0;

// Global values to minimize the stack footprint of GramPoly
int g_dataIndex;
uint8_t g_halfWindowSize;
uint8_t g_targetPoint = 0;
uint8_t g_derivativeOrder;

// Global MemoizationContext
static MemoizationContext context;

/*!
 * @brief Combines dataIndex and polynomialOrder into a single key for memoization.
 *
 * This function combines a 16-bit data index and an 8-bit polynomial order into a single
 * 32-bit key. This combined key is used in the hash function for memoization.
 *
 * @param dataIndex The data index.
 * @param polynomialOrder The polynomial order.
 * @return The combined 32-bit key.
 */
static inline uint32_t combineKey(uint16_t dataIndex, uint8_t polynomialOrder) {
    return ((uint32_t)dataIndex << 8) | polynomialOrder;
}

/*!
 * @brief Generates a hash value for a given key using a simple modulo operation.
 *
 * This function generates a hash value by taking the modulo of the key with the table size.
 * This ensures that the hash value fits within the bounds of the memoization table.
 *
 * @param key The combined 32-bit key.
 * @param tableSize The size of the hash table.
 * @return The generated hash value.
 */
static unsigned int hashGramPolyKey(uint32_t key, int tableSize) {
    return key % tableSize;
}

/*!
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

/*!
 * @brief Calculates the value of a Gram Polynomial or its derivative at a specific index within a data window.
 *
 * The Gram polynomials are utilized due to their orthogonality property, which ensures that the
 * integral (or in discrete cases, the sum) of the product of any two polynomials of different orders
 * over the window is zero. This orthogonality is crucial for isolating the contributions of each polynomial
 * order to the filtering process, thus minimizing distortion during data smoothing and differentiation.
 *
 * @param polynomialOrder The order of the polynomial (k) to compute.
 * @return The value of the Gram Polynomial or its specified derivative at the given index.
 */
static float GramPoly(uint8_t polynomialOrder) {
    //gramPolyCallCount++;

    uint8_t halfWindowSize = g_halfWindowSize;
    uint8_t derivativeOrder = g_derivativeOrder;
    int dataIndex = g_dataIndex;

    if (polynomialOrder == 0) {
        return (g_derivativeOrder == 0) ? 1.0 : 0.0;
    }

    float a = (4.0 * polynomialOrder - 2.0) / (polynomialOrder * (2.0 * halfWindowSize - polynomialOrder + 1.0));
    float b = 0.0;
    float c = ((polynomialOrder - 1.0) * (2.0 * halfWindowSize + polynomialOrder)) / (polynomialOrder * (2.0 * halfWindowSize - polynomialOrder + 1.0));

    if (polynomialOrder >= 2) {
        b += dataIndex * GramPoly(polynomialOrder - 1);

        if (derivativeOrder > 0) {
            g_derivativeOrder = derivativeOrder - 1;
            b += derivativeOrder * GramPoly(polynomialOrder - 1);
            g_derivativeOrder = derivativeOrder;
        }

        return a * b - c * GramPoly(polynomialOrder - 2);
    } else if (polynomialOrder == 1) {
        a = (2.0) / (2.0 * halfWindowSize);
        b += dataIndex * GramPoly(0);
        if (derivativeOrder > 0) {
            g_derivativeOrder = derivativeOrder - 1;
            b += derivativeOrder * GramPoly(0);
            g_derivativeOrder = derivativeOrder;
        }
        return a * b;
    }

    return 0.0;
}

/*!
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
 * @return The memoized value of the Gram Polynomial or its derivative.
 */
static float memoizedGramPoly(uint8_t polynomialOrder) {
    if (context.totalHashMapEntries >= MAX_ENTRIES) {
        return GramPoly(polynomialOrder);
    }

    uint32_t key = combineKey(g_dataIndex, polynomialOrder);
    unsigned int hashIndex = hashGramPolyKey(key, MAX_ENTRIES);

    while (context.memoizationTable[hashIndex].isOccupied) {
        if (context.memoizationTable[hashIndex].key == key) {
            return context.memoizationTable[hashIndex].value;
        }
        hashIndex = (hashIndex + 1) % MAX_ENTRIES;
    }

    float value = GramPoly(polynomialOrder);

    context.memoizationTable[hashIndex].key = key;
    context.memoizationTable[hashIndex].value = value;
    context.memoizationTable[hashIndex].isOccupied = true;
    context.totalHashMapEntries++;

    return value;
}

/*!
 * @brief Calculate a generalized factorial product for use in polynomial coefficient normalization.
 *
 * @param upperLimit The upper boundary of the range (inclusive) for the product calculation.
 * @param termCount The number of consecutive integers from `upperLimit` to include in the product.
 * @return The product of the sequence, representing a generalized factorial, as a float.
 */
static float GenFact(uint8_t upperLimit, uint8_t termCount) {
    float product = 1.0f;
    for (uint8_t j = (upperLimit - termCount) + 1; j <= upperLimit; j++) {
        product *= j;
    }
    return product;
}

/*!
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
    float w = 0.0;
    uint8_t derivativeOrder = g_derivativeOrder;

    for (uint8_t k = 0; k <= polynomialOrder; ++k) {
        g_dataIndex = dataIndex;
        g_derivativeOrder = 0;
        float part1 = memoizedGramPoly(k);
        g_derivativeOrder = derivativeOrder;
        g_dataIndex = targetPoint;

        float part2 = memoizedGramPoly(k);

        w += (2 * k + 1) * (GenFact(2 * g_halfWindowSize, k) / GenFact(2 * g_halfWindowSize + k + 1, k + 1)) * part1 * part2;
    }
    return w;
}

/*!
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
    for (int dataIndex = 0; dataIndex < fullWindowSize; ++dataIndex) {
        weights[dataIndex] = Weight(dataIndex - g_halfWindowSize, g_targetPoint, polynomialOrder);
    }
}

#define MAX_WINDOW_SIZE 33

/*!
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
 * @param data The array of data points to which the filter is to be applied.
 * @param dataSize The size of the data array.
 * @param halfWindowSize The half window size (m) of the Savitzky-Golay filter. The full window
 *                       size is '2m + 1'.
 * @param targetPoint The point at which the least-squares fit is evaluated. This is typically the
 *                    center of the window but can vary based on the filter application.
 * @param filter A pointer to the Savitzky-GolayFilter structure containing filter configuration
 *               and precomputed weights.
 * @param filteredData The array where the filtered data points will be stored.
 */
static void ApplyFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize, uint16_t targetPoint, SavitzkyGolayFilter filter, MqsRawDataPoint_t filteredData[]) {
    uint8_t maxHalfWindowSize = (MAX_WINDOW_SIZE - 1) / 2;

    if (halfWindowSize > maxHalfWindowSize) {
        printf("Warning: halfWindowSize (%d) exceeds the maximum allowed value (%d). Adjusting halfWindowSize to the maximum allowed value.\n", halfWindowSize, maxHalfWindowSize);
        halfWindowSize = maxHalfWindowSize;
    }

    const int window = 2 * halfWindowSize + 1;
    const int endidx = dataSize - 1;
    uint8_t width = halfWindowSize;
    static float weights[MAX_WINDOW_SIZE];

    ComputeWeights(halfWindowSize, targetPoint, filter.conf.polynomialOrder, filter.conf.derivativeOrder, weights);

    for (int i = 0; i <= dataSize - window; ++i) {
        float sum = 0.0;
        for (int j = 0; j < window; ++j) {
            sum += weights[j] * data[i + j].phaseAngle;
        }
        filteredData[i + width].phaseAngle = sum;
    }

    for (int i = 0; i < width; ++i) {
        ComputeWeights(halfWindowSize, width - i, filter.conf.polynomialOrder, filter.conf.derivativeOrder, weights);
        float leadingSum = 0.0;
        for (int j = 0; j < window; ++j) {
            leadingSum += weights[j] * data[window - j - 1].phaseAngle;
        }
        filteredData[i].phaseAngle = leadingSum;

        float trailingSum = 0.0;
        for (int j = 0; j < window; ++j) {
            trailingSum += weights[j] * data[endidx - window + j + 1].phaseAngle;
        }
        filteredData[endidx - i].phaseAngle = trailingSum;
    }
}

/*!
 * @brief Initializes and configures the Savitzky-Golay filter.
 *
 * This function initializes a Savitzky-Golay filter with specified configuration parameters.
 * The filter is used for smoothing data points in a dataset and can be configured to operate
 * as either a causal filter or a non-causal filter.
 *
 * @param conf The configuration structure containing filter parameters.
 *
 * @return An initialized Savitzky-GolayFilter structure.
 */
SavitzkyGolayFilter SavitzkyGolayFilter_init(SavitzkyGolayFilterConfig conf) {
    SavitzkyGolayFilter filter;
    filter.conf = conf;
    filter.dt = pow(conf.time_step, conf.derivation_order);
    return filter;
}

/*!
 * @brief Initializes a Savitzky-Golay filter with specified parameters.
 *
 * @param halfWindowSize The half window size.
 * @param polynomialOrder The polynomial order.
 * @param targetPoint The target point.
 * @param derivativeOrder The derivative order.
 * @param time_step The time step.
 * @return An initialized Savitzky-GolayFilter structure.
 */
SavitzkyGolayFilter initFilter(uint8_t halfWindowSize, uint8_t polynomialOrder, uint8_t targetPoint, uint8_t derivativeOrder, float time_step) {
    SavitzkyGolayFilterConfig conf = { halfWindowSize, targetPoint, polynomialOrder, derivativeOrder, time_step, derivativeOrder };
    return SavitzkyGolayFilter_init(conf);
}

/*!
 * @brief Manages a singleton instance of the Savitzky-Golay Filter.
 *
 * Implements a singleton pattern for the Savitzky-Golay Filter. It initializes the filter on
 * the first call and returns the same instance on subsequent calls, ensuring a single shared
 * instance is used.
 *
 * @param halfWindowSize The half window size.
 * @param polynomialOrder The polynomial order.
 * @param targetPoint The target point.
 * @param derivativeOrder The derivative order.
 * @param reset Boolean flag to indicate whether to reset the filter instance.
 * @return Pointer to the filter instance, or NULL if reset.
 */
SavitzkyGolayFilter* getFilterInstance(uint8_t halfWindowSize, uint8_t polynomialOrder, uint8_t targetPoint, uint8_t derivativeOrder, bool reset) {
    static SavitzkyGolayFilter filterInstance;
    filterInstance = initFilter(halfWindowSize, polynomialOrder, targetPoint, derivativeOrder, 1.0);
    return &filterInstance;
}

/*!
 * @brief Applies the Savitzky-Golay filter to the input data.
 *
 * @param data The input array of data points.
 * @param dataSize The number of elements in the input array.
 * @param halfWindowSize The half window size of the filter.
 * @param filteredData The output array to store the filtered data points.
 * @param polynomialOrder The polynomial order.
 * @param targetPoint The target point.
 * @param derivativeOrder The derivative order.
 */
void mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize, MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder, uint8_t targetPoint, uint8_t derivativeOrder) {
    initializeMemoizationTable();
    //gramPolyCallCount = 0;
    SavitzkyGolayFilter* filter = getFilterInstance(halfWindowSize, polynomialOrder, targetPoint, derivativeOrder, false);
    
    ApplyFilter(data, dataSize, halfWindowSize, targetPoint, *filter, filteredData);

    //printf("GramPoly call count after applying filter: %d\n", gramPolyCallCount);
}

