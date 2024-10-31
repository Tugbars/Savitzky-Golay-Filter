/**
 * @file mes_savgol.h
 * @brief Header file for the optimized Savitzky-Golay filter implementation.
 *
 * This header file contains the declarations and data structures for the
 * Savitzky-Golay filter, which is used for smoothing and differentiating data.
 * The implementation includes optimizations such as memoization to improve
 * computational efficiency.
 */

#ifndef MES_SAVGOL_H
#define MES_SAVGOL_H

#include "mqs_def.h"
#include <stdlib.h>
#include <stdbool.h>  /**< Include for boolean type */

//#define HALF_WINDOW_SIZE 13
//#define POLYNOMIAL_ORDER 4
//#define CALC_MEMO_ENTRIES(halfWindowSize, polynomialOrder) (((2 * (halfWindowSize) + 1) * ((polynomialOrder) + 1))) //not accurate, ignores edge Cases

#define MAX_ENTRIES 138  /**< Maximum number of entries in the memoization table */

/**
 * @struct SavitzkyGolayFilterConfig
 * @brief Configuration parameters for the Savitzky-Golay filter.
 *
 * This structure holds the configuration settings for initializing
 * the Savitzky-Golay filter, including window size, polynomial order,
 * derivative order, and time step.
 */
typedef struct {
    uint16_t halfWindowSize;     /**< Half of the window size used in the filter */
    uint16_t targetPoint;        /**< Target point index for the least-squares fit */
    uint8_t polynomialOrder;     /**< Order of the polynomial used in the filter */
    uint8_t derivativeOrder;     /**< Order of the derivative to compute */
    float time_step;             /**< Time step between data points */
    uint8_t derivation_order;    /**< Order of the derivation (typically same as derivativeOrder) */
} SavitzkyGolayFilterConfig;

/**
 * @struct HashMapEntry
 * @brief Entry in the memoization table for caching Gram Polynomial calculations.
 *
 * This structure represents an entry in the hash table used for memoizing
 * Gram Polynomial calculations to optimize performance.
 */
typedef struct {
    uint32_t key;        /**< Combined key representing data index and polynomial order */
    float value;         /**< Computed value of the Gram Polynomial */
    bool isOccupied;     /**< Flag indicating if the hash table entry is occupied */
} HashMapEntry;

/**
 * @struct MemoizationContext
 * @brief Context for memoization, including the memoization table.
 *
 * This structure holds the memoization table and tracks the total number
 * of entries in the table.
 */
typedef struct {
    HashMapEntry memoizationTable[MAX_ENTRIES]; /**< Array of hash map entries for memoization */
    uint16_t totalHashMapEntries;               /**< Total number of entries in the memoization table */
} MemoizationContext;

/**
 * @struct SavitzkyGolayFilter
 * @brief The Savitzky-Golay filter structure.
 *
 * This structure encapsulates the filter configuration and computed parameters
 * necessary for applying the Savitzky-Golay filter to data.
 */
typedef struct {
    SavitzkyGolayFilterConfig conf; /**< Configuration parameters for the filter */
    float dt;                       /**< Time step raised to the power of derivative order */
} SavitzkyGolayFilter;

/**
 * @brief Initializes and configures the Savitzky-Golay filter.
 *
 * This function initializes a Savitzky-Golay filter with specified configuration parameters.
 * The filter is used for smoothing data points in a dataset and can be configured to operate
 * as either a causal filter or a non-causal filter.
 *
 * @param conf The configuration structure containing filter parameters.
 * @return An initialized SavitzkyGolayFilter structure.
 */
SavitzkyGolayFilter SavitzkyGolayFilter_init(SavitzkyGolayFilterConfig conf);

/**
 * @brief Applies the Savitzky-Golay filter to the input data.
 *
 * This function applies the Savitzky-Golay smoothing filter to a data set based on
 * the provided filter configuration. It handles both central and edge cases within the data set.
 *
 * @param data           The input array of data points to which the filter is to be applied.
 * @param dataSize       The number of elements in the input array.
 * @param halfWindowSize The half window size of the filter.
 * @param filteredData   The output array where the filtered data points will be stored.
 * @param polynomialOrder The order of the polynomial used in the least-squares fit.
 * @param targetPoint    The point at which the least-squares fit is evaluated.
 * @param derivativeOrder The order of the derivative to compute.
 */
void mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                      MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder,
                      uint8_t targetPoint, uint8_t derivativeOrder);

#endif /**< MES_SAVGOL_H */
