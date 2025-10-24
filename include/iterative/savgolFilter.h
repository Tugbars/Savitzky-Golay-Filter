/**
 * @file savgol_filter.h
 * @brief Header file for the Savitzky–Golay filter implementation.
 *
 * This header declares the data structures, constants, and functions used in
 * a Savitzky–Golay filtering algorithm for smoothing and derivative estimation.
 * The implementation includes:
 * - Gram polynomial evaluation using iterative dynamic programming.
 * - Weight calculation with vectorized (AVX-512/AVX) and scalar paths.
 * - Per-instance memoization with versioning for thread-safety.
 * - Efficient filter application with edge handling via mirror padding.
 * - SIMD-accelerated SoA (Structure of Arrays) implementation.
 *
 * Author: Tugbars Heptaskin
 * Date: 2025-02-01
 * Updated: 2025-10-24 (added vectorized implementation)
 */

#ifndef SAVGOL_FILTER_H
#define SAVGOL_FILTER_H

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdbool.h>

// Define ALIGNED macro for alignment (64 bytes for AVX-512)
#if defined(_MSC_VER)
#include <stdalign.h>
#define ALIGNED alignas(64)
#elif defined(__GNUC__) || defined(__clang__)
#define ALIGNED __attribute__((aligned(64)))
#else
#warning "ALIGNED macro not defined for this compiler. Alignment may not be guaranteed."
#define ALIGNED
#endif


// Constant Definitions
#define MAX_WINDOW 255                  // Maximum window size (must be odd)
#define MAX_HALF_WINDOW_FOR_MEMO 127    // Maximum half-window size for memoization
#define MAX_POLY_ORDER_FOR_MEMO 10      // Maximum polynomial order for memoization
#define MAX_DERIVATIVE_FOR_MEMO 5       // Maximum derivative order for memoization
#define MAX_ORDER 11                    // Maximum polynomial order + 1

// Data Type Definitions
typedef struct {
    float phaseAngle; // Input data point (phase angle)
} MqsRawDataPoint_t;

typedef struct {
    uint8_t halfWindowSize;   // Half the filter window size (m in N=2m+1)
    uint16_t targetPoint;     // Target point for polynomial evaluation
    uint8_t polynomialOrder;  // Maximum polynomial order for fitting
    uint8_t derivativeOrder;  // Derivative order (0 for smoothing)
    float time_step;          // Time step for derivative scaling
} SavGolConf;

typedef struct {
    bool isComputed; // Flag indicating if cache entry is valid
    float value;     // Cached Gram polynomial value
    uint64_t version; // Cache version for invalidation
} GramPolyCacheEntry;

typedef struct {
    ALIGNED float centralWeights[MAX_WINDOW];  // Aligned array for filter weights
    ALIGNED float tempWindow[MAX_WINDOW];      // Aligned array for mirrored edge data
    GramPolyCacheEntry gramPolyCache[2 * MAX_HALF_WINDOW_FOR_MEMO + 1][MAX_POLY_ORDER_FOR_MEMO][MAX_DERIVATIVE_FOR_MEMO]; // Memoization cache
    bool weightsValid;             // Flag for weights validity
    uint8_t lastHalfWindowSize;    // Last used halfWindowSize for caching
    uint8_t lastPolyOrder;         // Last used polynomialOrder for caching
    uint8_t lastDerivOrder;        // Last used derivativeOrder for caching
    uint16_t lastTargetPoint;      // Last used targetPoint for caching
    uint64_t cacheVersion;         // Cache version for memoization
} SavGolState;

typedef struct {
    SavGolConf conf;  // Filter configuration
    SavGolState state; // Filter state (weights, cache)
    float dt;         // Time step scaling factor for derivatives
    float weights[MAX_WINDOW]; // Precomputed weights for center region
} SavitzkyGolayFilter;

typedef struct {
    uint8_t halfWindowSize;  // Half window size for polynomial calculations
    uint16_t targetPoint;    // Target point for polynomial evaluation
    uint8_t derivativeOrder; // Derivative order for polynomial
} GramPolyContext;

// Compile-time alignment checks for SavGolState fields
#ifdef __cplusplus
static_assert(offsetof(SavGolState, centralWeights) % 64 == 0, "centralWeights must be 64-byte aligned");
static_assert(offsetof(SavGolState, tempWindow) % 64 == 0, "tempWindow must be 64-byte aligned");
#else
_Static_assert(offsetof(SavGolState, centralWeights) % 64 == 0, "centralWeights must be 64-byte aligned");
_Static_assert(offsetof(SavGolState, tempWindow) % 64 == 0, "tempWindow must be 64-byte aligned");
#endif

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief Vectorized Savitzky-Golay filter (AVX-512/AVX2/SSE2/Scalar dispatch)
 * 
 * Uses native SoA architecture for optimal SIMD performance:
 * - Converts AoS → SoA once at input boundary
 * - Computes entirely in SoA format (zero struct access in hot path)
 * - Writes back SoA → AoS once at output boundary
 * 
 * Expected speedup over scalar: 2-8x depending on data size and architecture
 * 
 * @param data Input data array
 * @param dataSize Number of data points
 * @param halfWindowSize Half-window size (window = 2*halfWindowSize + 1)
 * @param filteredData Output filtered data array
 * @param polynomialOrder Polynomial order for fitting
 * @param targetPoint Target point within window
 * @param derivativeOrder Derivative order (0 = smoothing)
 * @return 0 on success, negative on error
 */
int mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                     MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder,
                     uint8_t targetPoint, uint8_t derivativeOrder);

/**
 * @brief Initialize filter structure with configuration
 * 
 * @param halfWindowSize Half-window size
 * @param polynomialOrder Polynomial order
 * @param targetPoint Target point within window
 * @param derivativeOrder Derivative order
 * @param time_step Time step for derivative scaling
 * @return Initialized filter structure
 */
SavitzkyGolayFilter initFilter(uint8_t halfWindowSize, uint8_t polynomialOrder, uint8_t targetPoint,
                                uint8_t derivativeOrder, float time_step);

/**
 * @brief Compute filter weights for a given configuration
 * 
 * Used internally by both scalar and vectorized implementations.
 * 
 * @param halfWindowSize Half-window size
 * @param targetPoint Target point within window
 * @param polynomialOrder Polynomial order
 * @param derivativeOrder Derivative order
 * @param weights Output array for computed weights (must be at least windowSize elements)
 */
void ComputeWeights(uint8_t halfWindowSize, uint8_t targetPoint,
                   uint8_t polynomialOrder, uint8_t derivativeOrder, float *weights);

#ifdef __cplusplus
}
#endif

#endif // SAVGOL_FILTER_H