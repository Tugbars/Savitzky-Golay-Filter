/**
 * @file savgol_filter.h
 * @brief Header file for the Savitzky–Golay filter implementation.
 *
 * This header declares the data structures, constants, and functions used in
 * a Savitzky–Golay filtering algorithm for smoothing and derivative estimation.
 *
 * @author Tugbars Heptaskin
 * @date 2025-11-10
 */

#ifndef SAVGOL_FILTER_H
#define SAVGOL_FILTER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

//-------------------------
// Constant Definitions
//-------------------------

/// Maximum window size for the filter (must be an odd number)
#define MAX_WINDOW 33

#define MAX_DERIVATIVE_FOR_MEMO 5

/// Define to use the optimized version of GenFact (precomputed values)
#define OPTIMIZE_GENFACT

#define MAX_ORDER 5

//-------------------------
// Data Type Definitions
//-------------------------

/**
 * @brief Minimal raw data structure.
 *
 * Represents a single measurement with a phase angle.
 */
typedef struct
{
    float phaseAngle;
} MqsRawDataPoint_t;

/**
 * @brief Savitzky–Golay filter configuration.
 *
 * Contains filter parameters such as the half-window size, the polynomial order,
 * the target point in the window, the derivative order, and the time step.
 */
typedef struct
{
    uint8_t halfWindowSize;  /**< Half window size (window size = 2*halfWindowSize+1) */
    uint8_t polynomialOrder; /**< Order of the polynomial used for fitting */
    uint8_t targetPoint;     /**< Target point within the window (usually center) */
    uint8_t derivativeOrder; /**< Order of the derivative to compute (0 for smoothing) */
    float time_step;         /**< Time step used in the filter */
} SavitzkyGolayFilterConfig;

/**
 * @brief Savitzky–Golay filter context.
 *
 * Contains the filter configuration and a scaling factor (dt) computed based on the derivative order.
 */
typedef struct
{
    SavitzkyGolayFilterConfig conf; /**< Filter configuration parameters */
    float dt;                       /**< Scaling factor computed as (time_step)^derivativeOrder */
} SavitzkyGolayFilter;

/**
 * @brief Context for iterative Gram polynomial evaluation.
 *
 * This context is passed to functions that compute Gram polynomials for use in
 * calculating filter weights.
 */
typedef struct
{
    uint8_t halfWindowSize;  /**< Half window size for the filter */
    uint8_t targetPoint;     /**< Target point (index offset) in the window */
    uint8_t derivativeOrder; /**< Order of the derivative */
} GramPolyContext;

typedef struct
{
    bool isComputed;
    float value;
} GramPolyCacheEntry;

//-------------------------
// Function Declarations
//-------------------------

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Applies the Savitzky–Golay filter to a data array.
     *
     * This is the main entry point for filtering a data sequence. It performs error checking,
     * initializes the filter, and applies the filter to the raw data.
     *
     * @param data         Array of raw data points (input).
     * @param dataSize     Number of data points in the array.
     * @param halfWindowSize Half-window size (filter window size = 2*halfWindowSize+1).
     * @param filteredData Array to store the filtered data points (output).
     * @param polynomialOrder Order of the polynomial used for fitting.
     * @param targetPoint  The target point in the filter window.
     * @param derivativeOrder Order of the derivative (0 for smoothing).
     * @return 0 on success, negative error code on failure.
     */
    int mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                         MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder,
                         uint8_t targetPoint, uint8_t derivativeOrder);

#ifdef SAVGOL_PARALLEL_BUILD
    /**
     * @brief Applies the Savitzky–Golay filter with explicit thread count control.
     *
     * This function is only available when building with OpenMP support (parallel version).
     * It provides fine-grained control over the number of threads used for filtering.
     *
     * @param data         Array of raw data points (input).
     * @param dataSize     Number of data points in the array.
     * @param halfWindowSize Half-window size (filter window size = 2*halfWindowSize+1).
     * @param filteredData Array to store the filtered data points (output).
     * @param polynomialOrder Order of the polynomial used for fitting.
     * @param targetPoint  The target point in the filter window.
     * @param derivativeOrder Order of the derivative (0 for smoothing).
     * @param numThreads   Number of threads to use:
     *                     - 0: Auto-detect (use all available cores)
     *                     - -1: Force sequential execution (single-threaded)
     *                     - >0: Use exactly this many threads
     * @return 0 on success, negative error code on failure.
     *
     * @note This function is only available in the parallel build (USE_PARALLEL_SAVGOL=ON).
     * @note If OpenMP is not available at runtime, this function will execute sequentially
     *       regardless of the numThreads parameter.
     */
    int mes_savgolFilter_threaded(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                                  MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder,
                                  uint8_t targetPoint, uint8_t derivativeOrder, int numThreads);
#endif // SAVGOL_PARALLEL_BUILD

    /**
     * @brief Initializes a Savitzky–Golay filter instance.
     *
     * Sets the configuration parameters and computes the scaling factor based on the derivative order.
     *
     * @param halfWindowSize Half-window size.
     * @param polynomialOrder Polynomial order.
     * @param targetPoint Target point within the window.
     * @param derivativeOrder Derivative order.
     * @param time_step Time step value.
     * @return A SavitzkyGolayFilter structure with initialized values.
     */
    SavitzkyGolayFilter initFilter(uint8_t halfWindowSize, uint8_t polynomialOrder, uint8_t targetPoint,
                                   uint8_t derivativeOrder, float time_step);

    /**
     * @brief Retrieves a Gram polynomial cache entry (for testing/debugging).
     *
     * @param shiftedIndex Data index shifted to non-negative range [0, 2n].
     * @param polyOrder Polynomial order k.
     * @param derivOrder Derivative order d.
     * @return Pointer to cache entry, or NULL if indices out of bounds.
     */
    const GramPolyCacheEntry *GetGramPolyCacheEntry(int shiftedIndex, uint8_t polyOrder, uint8_t derivOrder);

#ifdef __cplusplus
}
#endif

#endif // SAVGOL_FILTER_H