/**
 * @file savgol_filter.h
 * @brief Header file for the Savitzky–Golay filter implementation.
 *
 * This header declares the data structures, constants, and functions used in
 * a Savitzky–Golay filtering algorithm for smoothing and derivative estimation.
 *
 * The implementation includes:
 * - Gram polynomial evaluation (iterative method).
 * - Weight calculation based on Gram polynomials.
 * - An optional optimized precomputation for generalized factorial (GenFact) calculations.
 * - A complete filter application function with edge‐handling.
 *
 * @author 
 * Tugbars Heptaskin
 * @date 
 * 2025-02-01
 */

#ifndef SAVGOL_FILTER_H
#define SAVGOL_FILTER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Define ALIGNED macro for alignment (32 bytes for AVX)
#if defined(_MSC_VER) // Microsoft Visual C++
#include <stdalign.h>
#define ALIGNED alignas(32)
#elif defined(__GNUC__) || defined(__clang__) // GCC or Clang
#define ALIGNED __attribute__((aligned(32)))
#else
#warning "ALIGNED macro not defined for this compiler. Alignment may not be guaranteed."
#define ALIGNED
#endif

//-------------------------
// Constant Definitions
//-------------------------

#define MAX_HALF_WINDOW_FOR_MEMO 32
#define MAX_POLY_ORDER_FOR_MEMO   5
#define MAX_DERIVATIVE_FOR_MEMO   5

/// Maximum window size for the filter (must be an odd number)
#define MAX_WINDOW 33

/// Define to use the optimized version of GenFact (precomputed values)
#define OPTIMIZE_GENFACT

/// Uncomment the following line to build with a test main
//#define TEST_MAIN

//-------------------------
// Data Type Definitions
//-------------------------

/**
 * @brief Minimal raw data structure.
 *
 * Represents a single measurement with a phase angle.
 */
typedef struct {
    float phaseAngle;
} MqsRawDataPoint_t;

/**
 * @brief Savitzky–Golay filter configuration.
 *
 * Contains filter parameters such as the half-window size, the polynomial order,
 * the target point in the window, the derivative order, and the time step.
 */
typedef struct {
    uint8_t halfWindowSize;   /**< Half window size (window size = 2*halfWindowSize+1) */
    uint8_t polynomialOrder;  /**< Order of the polynomial used for fitting */
    uint8_t targetPoint;      /**< Target point within the window (usually center) */
    uint8_t derivativeOrder;  /**< Order of the derivative to compute (0 for smoothing) */
    float time_step;          /**< Time step used in the filter */
} SavitzkyGolayFilterConfig;


/**
 * @brief Per-instance state for the Savitzky–Golay filter.
 *
 * Encapsulates all mutable state to ensure thread-safety and reentrancy.
 */
typedef struct {
    float centralWeights[MAX_WINDOW] __attribute__((aligned(32)));
    float weights[MAX_WINDOW]        __attribute__((aligned(32)));
    float tempWindow[MAX_WINDOW]     __attribute__((aligned(32)));
    bool weightsValid;                             // Flag for weights validity
    uint8_t lastHalfWindowSize;                    // Last half-window size for change detection
    uint8_t lastPolyOrder;                         // Last polynomial order
    uint8_t lastDerivOrder;                        // Last derivative order
    uint16_t lastTargetPoint;                      // Last target point
#ifdef ENABLE_MEMOIZATION
   /// Per-instance Gram‑poly cache: [2*m+1][k+1][d+1]
    struct {
        bool   isComputed;
        float  value;
    } gramPolyCache[
        2*MAX_HALF_WINDOW_FOR_MEMO + 1
    ][
        MAX_POLY_ORDER_FOR_MEMO
    ][
        MAX_DERIVATIVE_FOR_MEMO
    ];
#endif
} SavGolState;


/**
 * @brief Savitzky–Golay filter context.
 *
 * Contains the filter configuration, a scaling factor (dt), and per-instance state.
 */
typedef struct {
    SavitzkyGolayFilterConfig conf; /**< Filter configuration parameters */
    float dt;                       /**< Scaling factor computed as (time_step)^derivativeOrder */
    SavGolState state;              // Per-instance mutable state
} SavitzkyGolayFilter;
/**
 * @brief Context for iterative Gram polynomial evaluation.
 *
 * This context is passed to functions that compute Gram polynomials for use in
 * calculating filter weights.
 */
typedef struct {
    uint8_t halfWindowSize;  /**< Half window size for the filter */
    uint8_t targetPoint;     /**< Target point (index offset) in the window */
    uint8_t derivativeOrder; /**< Order of the derivative */
} GramPolyContext;


//-------------------------
// Function Declarations
//-------------------------

#ifdef __cplusplus
extern "C" {
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
 * @return Pointer to the initialized filter instance (must be freed by the caller).
 */
SavitzkyGolayFilter* mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
    MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder,
    uint8_t targetPoint, uint8_t derivativeOrder);

/**
 * @brief Initializes a Savitzky–Golay filter instance.
 *
 * Allocates and sets the configuration parameters and computes the scaling factor based on the derivative order.
 *
 * @param halfWindowSize Half-window size.
 * @param polynomialOrder Polynomial order.
 * @param targetPoint Target point within the window.
 * @param derivativeOrder Derivative order.
 * @param time_step Time step value.
 * @return Pointer to the initialized SavitzkyGolayFilter instance (must be freed by the caller).
 */
SavitzkyGolayFilter* initFilter(uint8_t halfWindowSize, uint8_t polynomialOrder, uint8_t targetPoint,
                              uint8_t derivativeOrder, float time_step);

#ifdef __cplusplus
}
#endif

#endif // SAVGOL_FILTER_H