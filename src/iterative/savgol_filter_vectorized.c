/**
 * @file savgol_filter_vectorized.c
 * @brief Vectorized Savitzky-Golay filter with SoA architecture
 * 
 * @details
 * Complete vectorized implementation following the radix-4 FFT design pattern:
 * 
 * Architecture:
 * 1. Convert AoS → SoA ONCE at input boundary (extract .phaseAngle)
 * 2. Compute entirely in SoA format (ZERO struct access in hot path)
 * 3. Write back SoA → AoS ONCE at output boundary
 * 
 * Performance benefits:
 * - Sequential memory access (perfect for SIMD)
 * - Aligned loads (full vector efficiency)
 * - No struct field indirection
 * - Automatic SIMD dispatch (AVX-512 → AVX2 → SSE2 → Scalar)
 * 
 * @author Tugbars Heptaskin
 * @date 2025-10-24
 */

#include "savgolFilter.h"
#include "savgol_simd_ops.h"
#include "savgol_soa_convert.h"
#include "savgol_kernels.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

// External declarations (from original savgol implementation)
extern SavitzkyGolayFilter initFilter(uint8_t halfWindowSize, uint8_t polynomialOrder,
                                      uint8_t targetPoint, uint8_t derivativeOrder, float samplingRate);
extern void ComputeWeights(uint8_t halfWindowSize, uint8_t targetPoint,
                           uint8_t polynomialOrder, uint8_t derivativeOrder, float *weights);

// Edge cache structures (from original implementation)
#define MAX_HALF_WINDOW_FOR_MEMO 16
#define MAX_WINDOW (2 * MAX_HALF_WINDOW_FOR_MEMO + 1)

typedef struct {
    float weights[MAX_WINDOW];
    uint8_t halfWindowSize;
    uint8_t polynomialOrder;
    uint8_t derivativeOrder;
    uint8_t targetPoint;
    bool valid;
} EdgeWeightCache;

static EdgeWeightCache leadingEdgeCache[MAX_HALF_WINDOW_FOR_MEMO];
static EdgeWeightCache trailingEdgeCache[MAX_HALF_WINDOW_FOR_MEMO];
static bool edgeCacheInitialized = false;

static void InitEdgeCacheIfNeeded(void) {
    if (!edgeCacheInitialized) {
        memset(leadingEdgeCache, 0, sizeof(leadingEdgeCache));
        memset(trailingEdgeCache, 0, sizeof(trailingEdgeCache));
        edgeCacheInitialized = true;
    }
}

//==============================================================================
// VECTORIZED FILTER APPLICATION (SoA Format)
//==============================================================================

/**
 * @brief Apply Savitzky-Golay filter to SoA data
 * 
 * This function operates entirely on float arrays - ZERO struct access!
 * 
 * @param input_soa Input data (float array, already extracted from structs)
 * @param dataSize Number of data points
 * @param halfWindowSize Half-window size
 * @param targetPoint Target point within window
 * @param filter Filter parameters
 * @param output_soa Output data (float array, will be written back to structs)
 */
static void ApplyFilterSoA(
    const float *input_soa,
    size_t dataSize,
    uint8_t halfWindowSize,
    uint8_t targetPoint,
    const SavitzkyGolayFilter *filter,
    float *output_soa)
{
    uint8_t windowSize = 2 * halfWindowSize + 1;
    uint8_t width = halfWindowSize;
    int lastIndex = dataSize - 1;
    
    //==========================================================================
    // STEP 1: Center Region - Vectorized Dot Products
    //==========================================================================
    // Use the vectorized dot product kernel for each output point
    
    for (int i = 0; i < (int)dataSize - windowSize + 1; ++i) {
        // Compute dot product: weights · data[i..i+windowSize-1]
        // This uses AVX-512/AVX2/SSE2/Scalar automatically
        float result = SAVGOL_DOT_PRODUCT(
            filter->weights, 
            &input_soa[i], 
            windowSize
        );
        output_soa[i + width] = result;
    }
    
    //==========================================================================
    // STEP 2: Edge Handling (Leading and Trailing)
    //==========================================================================
    InitEdgeCacheIfNeeded();
    
    // Temporary buffer for edge weights (stack allocation)
    float tempWeights[MAX_WINDOW];
    
    for (int i = 0; i < width; ++i) {
        uint8_t target = width - i;
        
        //----------------------------------------------------------------------
        // Check cache for edge weights
        //----------------------------------------------------------------------
        bool useCache = false;
        const float *edgeWeights = NULL;
        
        if (i < MAX_HALF_WINDOW_FOR_MEMO &&
            leadingEdgeCache[i].valid &&
            leadingEdgeCache[i].halfWindowSize == halfWindowSize &&
            leadingEdgeCache[i].polynomialOrder == filter->conf.polynomialOrder &&
            leadingEdgeCache[i].derivativeOrder == filter->conf.derivativeOrder &&
            leadingEdgeCache[i].targetPoint == target)
        {
            useCache = true;
            edgeWeights = leadingEdgeCache[i].weights;
        }
        else {
            // Compute fresh weights
            ComputeWeights(halfWindowSize, target, 
                          filter->conf.polynomialOrder,
                          filter->conf.derivativeOrder, 
                          tempWeights);
            
            // Cache if possible
            if (i < MAX_HALF_WINDOW_FOR_MEMO) {
                memcpy(leadingEdgeCache[i].weights, tempWeights, 
                       windowSize * sizeof(float));
                leadingEdgeCache[i].halfWindowSize = halfWindowSize;
                leadingEdgeCache[i].polynomialOrder = filter->conf.polynomialOrder;
                leadingEdgeCache[i].derivativeOrder = filter->conf.derivativeOrder;
                leadingEdgeCache[i].targetPoint = target;
                leadingEdgeCache[i].valid = true;
            }
            
            edgeWeights = tempWeights;
        }
        
        //----------------------------------------------------------------------
        // Leading Edge: Reverse traversal (data[windowSize-1], ..., data[0])
        //----------------------------------------------------------------------
        // For reverse access, we use strided kernel or scalar
        // Note: Gather instructions for reverse access are often slower than scalar
        
        float leadingSum = 0.0f;
        
        // Simple scalar loop for reverse access (often faster than gather)
        for (int k = 0; k < windowSize; ++k) {
            leadingSum += edgeWeights[k] * input_soa[windowSize - 1 - k];
        }
        
        output_soa[i] = leadingSum;
        
        //----------------------------------------------------------------------
        // Trailing Edge: Forward traversal from offset
        //----------------------------------------------------------------------
        int offset = lastIndex - windowSize + 1;
        
        // Use vectorized dot product for forward access
        float trailingSum = SAVGOL_DOT_PRODUCT(
            edgeWeights,
            &input_soa[offset],
            windowSize
        );
        
        output_soa[lastIndex - i] = trailingSum;
    }
}

//==============================================================================
// MAIN FILTER FUNCTION: Public API with SoA Architecture
//==============================================================================

/**
 * @brief Applies the Savitzky–Golay filter to a data sequence (VECTORIZED)
 * 
 * This is the main entry point - wraps the vectorized SoA implementation.
 * 
 * Architecture:
 * 1. Extract .phaseAngle field (AoS → SoA) - ONCE
 * 2. Apply vectorized filter in pure SoA format - HOT PATH
 * 3. Write results back (SoA → AoS) - ONCE
 * 
 * @param data Array of raw data points (input)
 * @param dataSize Number of data points
 * @param halfWindowSize Half-window size for the filter
 * @param filteredData Array to store the filtered data points (output)
 * @param polynomialOrder Polynomial order used for the filter
 * @param targetPoint The target point within the window
 * @param derivativeOrder Derivative order (0 for smoothing)
 * @return 0 on success, negative on error
 */
int mes_savgolFilter_vectorized(
    MqsRawDataPoint_t data[], 
    size_t dataSize, 
    uint8_t halfWindowSize,
    MqsRawDataPoint_t filteredData[], 
    uint8_t polynomialOrder,
    uint8_t targetPoint, 
    uint8_t derivativeOrder)
{
    //==========================================================================
    // Validation (same as original)
    //==========================================================================
    assert(data != NULL && "Input data pointer must not be NULL");
    assert(filteredData != NULL && "Filtered data pointer must not be NULL");
    assert(dataSize > 0 && "Data size must be greater than 0");
    assert(halfWindowSize > 0 && "Half-window size must be greater than 0");
    assert((2 * halfWindowSize + 1) <= dataSize && "Filter window size must not exceed data size");
    assert(polynomialOrder < (2 * halfWindowSize + 1) && "Polynomial order must be less than the filter window size");
    assert(targetPoint <= (2 * halfWindowSize) && "Target point must be within the filter window");

    if (data == NULL || filteredData == NULL) {
        fprintf(stderr, "ERROR: NULL pointer passed to mes_savgolFilter_vectorized.\n");
        return -1;
    }
    
    if (dataSize == 0 || halfWindowSize == 0 ||
        polynomialOrder >= 2 * halfWindowSize + 1 ||
        targetPoint > 2 * halfWindowSize ||
        (2 * halfWindowSize + 1) > dataSize)
    {
        fprintf(stderr, "ERROR: Invalid filter parameters.\n");
        return -2;
    }

    //==========================================================================
    // STEP 1: Allocate aligned SoA buffers
    //==========================================================================
    float *input_soa = savgol_alloc_aligned(dataSize);
    float *output_soa = savgol_alloc_aligned(dataSize);
    
    if (!input_soa || !output_soa) {
        fprintf(stderr, "ERROR: Failed to allocate aligned memory for SoA buffers.\n");
        savgol_free_aligned(input_soa);
        savgol_free_aligned(output_soa);
        return -3;
    }

    //==========================================================================
    // STEP 2: Convert AoS → SoA (extract .phaseAngle field) - ONCE
    //==========================================================================
    savgol_aos_to_soa(data, input_soa, dataSize);

    //==========================================================================
    // STEP 3: Initialize filter and compute weights
    //==========================================================================
    SavitzkyGolayFilter filter = initFilter(
        halfWindowSize, 
        polynomialOrder, 
        targetPoint, 
        derivativeOrder, 
        1.0f
    );
    
    // Ensure weights are computed for center region
    if (!filter.state.weightsValid ||
        filter.state.lastHalfWindowSize != halfWindowSize ||
        filter.state.lastPolyOrder != polynomialOrder ||
        filter.state.lastDerivOrder != derivativeOrder ||
        filter.state.lastTargetPoint != targetPoint)
    {
        ComputeWeights(halfWindowSize, targetPoint, polynomialOrder, 
                      derivativeOrder, filter.weights);
    }

    //==========================================================================
    // STEP 4: Apply vectorized filter (ZERO struct access in hot path!)
    //==========================================================================
    ApplyFilterSoA(
        input_soa, 
        dataSize, 
        halfWindowSize, 
        targetPoint, 
        &filter, 
        output_soa
    );

    //==========================================================================
    // STEP 5: Write back SoA → AoS (.phaseAngle field) - ONCE
    //==========================================================================
    savgol_soa_to_aos(output_soa, filteredData, dataSize);

    //==========================================================================
    // STEP 6: Cleanup
    //==========================================================================
    savgol_free_aligned(input_soa);
    savgol_free_aligned(output_soa);

    return 0;
}