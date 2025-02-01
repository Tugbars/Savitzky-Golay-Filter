#ifndef MES_SAVGOL_H
#define MES_SAVGOL_H

/**
 * @file mes_savgol.h
 * @brief Header file for the optimized Savitzky-Golay filter implementation with recursive calculation and memoization.
 *
 * This header file contains the declarations and data structures used in the Savitzky-Golay filter implementation.
 * The filter is designed for efficient data smoothing and differentiation, utilizing memoization to optimize
 * recursive Gram polynomial calculations.
 */

#include "mqs_def.h"
#include <stdlib.h>
#include <stdbool.h>

/** Maximum number of entries in the memoization table */
#define MAX_ENTRIES 188

/**
 * @struct SavitzkyGolayFilterConfig
 * @brief Configuration parameters for the Savitzky-Golay filter.
 *
 * This structure holds the configuration settings for the filter, including window size,
 * polynomial order, derivative order, and time step between data points.
 */
typedef struct {
    uint16_t halfWindowSize;    /**< Half of the filter window size */
    uint16_t targetPoint;       /**< The target point in the window */
    uint8_t polynomialOrder;    /**< The polynomial order used in the filter */
    uint8_t derivativeOrder;    /**< The derivative order to compute */
    float time_step;            /**< Time step between data points */
    uint8_t derivation_order;   /**< Redundant field for derivative order (consider removing if not used) */
} SavitzkyGolayFilterConfig;

/**
 * @struct HashMapEntry
 * @brief Entry in the memoization hash table.
 *
 * This structure represents an entry in the memoization table used for caching Gram polynomial values.
 */
typedef struct {
    uint32_t key;       /**< Unique key combining data index and polynomial order */
    float value;        /**< Cached value of the Gram polynomial */
    bool isOccupied;    /**< Flag indicating if the entry is occupied */
} HashMapEntry;

/**
 * @struct MemoizationContext
 * @brief Context for memoization, including the hash table and entry count.
 *
 * This structure holds the memoization table and keeps track of the total number of entries.
 */
typedef struct {
    HashMapEntry memoizationTable[MAX_ENTRIES];   /**< Memoization hash table */
    uint16_t totalHashMapEntries;                 /**< Total number of entries in the hash table */
} MemoizationContext;

/**
 * @struct SavitzkyGolayFilter
 * @brief Represents a Savitzky-Golay filter with its configuration and precomputed values.
 *
 * This structure encapsulates the filter configuration and any precomputed values needed for filtering.
 */
typedef struct {
    SavitzkyGolayFilterConfig conf;   /**< Filter configuration parameters */
    float dt;                         /**< Time step raised to the power of derivative order */
} SavitzkyGolayFilter;

/**
 * @brief Initializes a Savitzky-Golay filter with the given configuration.
 *
 * This function sets up a Savitzky-Golay filter by initializing its configuration parameters
 * and precomputing any necessary constants. It prepares the filter for application to data.
 *
 * @param conf The configuration parameters for the filter.
 * @return An initialized SavitzkyGolayFilter structure.
 */
SavitzkyGolayFilter SavitzkyGolayFilter_init(SavitzkyGolayFilterConfig conf);

/**
 * @brief Applies the Savitzky-Golay filter to the input data array.
 *
 * This function filters the input data using the specified filter parameters and stores the result in the output array.
 * It handles the computation of weights, application of the filter to central and edge data points, and manages
 * memoization to optimize performance.
 *
 * @param data            The input array of data points to be filtered.
 * @param dataSize        The number of elements in the input data array.
 * @param halfWindowSize  The half window size of the filter.
 * @param filteredData    The output array where filtered data points will be stored.
 * @param polynomialOrder The polynomial order used in the filter.
 * @param targetPoint     The target point in the window for evaluation.
 * @param derivativeOrder The derivative order to compute.
 */
void mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize, MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder, uint8_t targetPoint, uint8_t derivativeOrder);

#endif // MES_SAVGOL_H
