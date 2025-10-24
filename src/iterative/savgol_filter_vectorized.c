/**
 * @file savgol_filter_simd_ready.c
 * @brief SIMD-ready implementation of the Savitzky–Golay filter.
 *
 * This version is structured to seamlessly accommodate SIMD operations:
 * - Uses SoA (Structure of Arrays) layout for vectorization
 * - Aligned memory allocation for weights
 * - Clean separation between weight computation and convolution
 * - Drop-in replacement for SIMD dot product kernels
 *
 * Architecture:
 * 1. Boundary operations: AoS ↔ SoA conversion (done ONCE)
 * 2. Weight computation: Scalar (not the bottleneck)
 * 3. Convolution: SIMD-optimized dot products (the hot loop)
 * 4. Edge handling: Reuses convolution kernels
 *
 * Author: Tugbars Heptaskin
 * Date: 2025-10-24
 * SIMD-Ready Version: 2025-10-24
 */

#include "savgolFilter.h"
#include "savgol_simd_ops.h"
#include "savgol_kernels.h"
#include "savgol_soa_convert.h"
#include "savgol_reverse_kernels.h"  // <-- NEW: Reverse dot product kernels

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
static float precomputedGenFactNum[MAX_POLY_ORDER + 1];
/// Precomputed denominator factors for GenFact.
static float precomputedGenFactDen[MAX_POLY_ORDER + 1];
#endif

// Uncomment the following line to enable memoization of Gram polynomial calculations.
#define ENABLE_MEMOIZATION

//-------------------------
// NEW: GenFact Lookup Table (Static Allocation)
//-------------------------
#ifndef OPTIMIZE_GENFACT
#define GENFACT_TABLE_SIZE 65  // Supports up to halfWindow=32
static float genFactTable[GENFACT_TABLE_SIZE][GENFACT_TABLE_SIZE];
static bool genFactTableInit = false;

/**
 * @brief Initialize the GenFact lookup table (called once at startup or first use).
 */
static void InitGenFactTable(void)
{
    if (genFactTableInit) return;
    
    for (uint8_t upperLimit = 0; upperLimit < GENFACT_TABLE_SIZE; upperLimit++) {
        genFactTable[upperLimit][0] = 1.0f;
        
        for (uint8_t termCount = 1; termCount < GENFACT_TABLE_SIZE; termCount++) {
            if (upperLimit < termCount) {
                genFactTable[upperLimit][termCount] = 0.0f;
            } else {
                float product = 1.0f;
                uint8_t start = (upperLimit - termCount) + 1;
                for (uint8_t j = start; j <= upperLimit; j++) {
                    product *= (float)j;
                }
                genFactTable[upperLimit][termCount] = product;
            }
        }
    }
    genFactTableInit = true;
}
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
static void PrecomputeGenFacts(uint8_t halfWindowSize, uint8_t polynomialOrder)
{
    uint32_t upperLimitNum = 2 * halfWindowSize;
    for (uint8_t k = 0; k <= polynomialOrder; ++k)
    {
        float numProduct = 1.0f;
        for (uint8_t j = (upperLimitNum - k) + 1; j <= upperLimitNum; j++)
        {
            numProduct *= j;
        }
        precomputedGenFactNum[k] = numProduct;
        uint32_t upperLimitDen = 2 * halfWindowSize + k + 1;
        float denProduct = 1.0f;
        for (uint8_t j = (upperLimitDen - (k + 1)) + 1; j <= upperLimitDen; j++)
        {
            denProduct *= j;
        }
        precomputedGenFactDen[k] = denProduct;
    }
}
#else
/**
 * @brief Compute the generalized factorial (GenFact) using lookup table.
 *
 * @param upperLimit The upper limit of the product.
 * @param termCount The number of terms in the product.
 * @return The computed generalized factorial as a float.
 */
static inline float GenFact(uint8_t upperLimit, uint8_t termCount)
{
    if (!genFactTableInit) {
        InitGenFactTable();
    }
    
    if (upperLimit < GENFACT_TABLE_SIZE && termCount < GENFACT_TABLE_SIZE) {
        return genFactTable[upperLimit][termCount];
    }
    
    // Fallback for out-of-range values (shouldn't happen with proper constraints)
    float product = 1.0f;
    for (uint8_t j = (upperLimit - termCount) + 1; j <= upperLimit; j++)
    {
        product *= (float)j;
    }
    return product;
}
#endif

/**
 * @brief Iteratively computes the Gram polynomial (OPTIMIZED: strength reduction applied).
 *
 * This function computes the Gram polynomial F(k, d) using dynamic programming
 * with a rolling buffer approach and branch elimination.
 *
 * Optimizations:
 * - No VLA (uses fixed-size buffers)
 * - Eliminated branches in inner loops
 * - Strength reduction (moved divisions out of loops)
 * - Separate handling of d=0 case to eliminate conditionals
 *
 * @param polynomialOrder The current polynomial order k.
 * @param dataIndex The data index (can be negative if shifted so that the center is 0).
 * @param ctx Pointer to a GramPolyContext containing filter parameters.
 * @return The computed Gram polynomial value.
 */
static float GramPolyIterative(uint8_t polynomialOrder, int dataIndex, const GramPolyContext *ctx)
{
    // Retrieve necessary parameters from the context.
    uint8_t halfWindowSize = ctx->halfWindowSize;
    uint8_t derivativeOrder = ctx->derivativeOrder;

    // Fixed-size buffers on stack
    float buf0[MAX_ORDER];
    float buf1[MAX_ORDER];
    float buf2[MAX_ORDER];
    
    float *prev2 = buf0;
    float *prev = buf1;
    float *curr = buf2;

    // Base case: k = 0.
    for (uint8_t d = 0; d <= derivativeOrder; d++)
    {
        prev2[d] = (d == 0) ? 1.0f : 0.0f;
    }
    if (polynomialOrder == 0)
    {
        return prev2[derivativeOrder];
    }

    // k = 1: Compute first order polynomial values using the base case.
    // OPTIMIZATION: Hoist division out of loop (compute once)
    float inv_half = 1.0f / halfWindowSize;
    
    // CRITICAL: Maintain exact operation order from original for numerical stability
    // Handle d=0 case separately
    prev[0] = inv_half * (dataIndex * prev2[0]);
    
    // Handle d > 0 cases (eliminates branch from loop body)
    for (uint8_t d = 1; d <= derivativeOrder; d++)
    {
        // CRITICAL: Keep same operation order as original: inv_half * (dataIndex * prev2[d] + d * prev2[d-1])
        float inner_term = dataIndex * prev2[d] + d * prev2[d - 1];
        prev[d] = inv_half * inner_term;
    }
    
    if (polynomialOrder == 1)
    {
        return prev[derivativeOrder];
    }

    // Precompute constants outside the k loop
    float two_halfWinSize = 2.0f * halfWindowSize;
    
    // Iteratively compute F(k, d) for k >= 2.
    for (uint8_t k = 2; k <= polynomialOrder; k++)
    {
        // OPTIMIZATION: Compute reciprocal once to avoid repeated division
        float k_f = (float)k;
        float denom_recip = 1.0f / (k_f * (two_halfWinSize - k_f + 1.0f));
        
        // Precompute coefficients (same as original, but computed once per k)
        float a = (4.0f * k_f - 2.0f) * denom_recip;
        float c = ((k_f - 1.0f) * (two_halfWinSize + k_f)) * denom_recip;

        // OPTIMIZATION: Handle d=0 separately (eliminates conditional from loop)
        curr[0] = a * (dataIndex * prev[0]) - c * prev2[0];
        
        // OPTIMIZATION: d > 0 loop has no branches now
        for (uint8_t d = 1; d <= derivativeOrder; d++)
        {
            // CRITICAL: Maintain exact operation order
            float term = dataIndex * prev[d] + d * prev[d - 1];
            curr[d] = a * term - c * prev2[d];
        }

        // Rotate pointers (zero-copy)
        float *temp = prev2;
        prev2 = prev;
        prev = curr;
        curr = temp;
    }

    return prev[derivativeOrder];
}

//-------------------------
// Optional Memoization for Gram Polynomial Calculation
//-------------------------
#ifdef ENABLE_MEMOIZATION
// Define maximum cache dimensions (adjust as needed).
#define MAX_DERIVATIVE_FOR_MEMO 5 // Supports derivative orders 0..4.

static GramPolyCacheEntry gramPolyCache[2 * MAX_HALF_WINDOW_FOR_MEMO + 1][MAX_POLY_ORDER_FOR_MEMO][MAX_DERIVATIVE_FOR_MEMO];

// Helper function to access gramPolyCache for testing
const GramPolyCacheEntry *GetGramPolyCacheEntry(int shiftedIndex, uint8_t polyOrder, uint8_t derivOrder)
{
    if (shiftedIndex < 0 || shiftedIndex >= (2 * MAX_HALF_WINDOW_FOR_MEMO + 1) ||
        polyOrder >= MAX_POLY_ORDER_FOR_MEMO || derivOrder >= MAX_DERIVATIVE_FOR_MEMO)
    {
        return NULL;
    }
    return &gramPolyCache[shiftedIndex][polyOrder][derivOrder];
}

/**
 * @brief Clears the memoization cache for the current domain.
 *
 * @param halfWindowSize Half-window size.
 * @param polynomialOrder Polynomial order.
 * @param derivativeOrder Derivative order.
 */
static void ClearGramPolyCache(uint8_t halfWindowSize, uint8_t polynomialOrder, uint8_t derivativeOrder)
{
    int maxIndex = 2 * halfWindowSize + 1;
    for (int i = 0; i < maxIndex; i++)
    {
        for (int k = 0; k <= polynomialOrder; k++)
        {
            for (int d = 0; d <= derivativeOrder; d++)
            {
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
static float MemoizedGramPoly(uint8_t polynomialOrder, int dataIndex, const GramPolyContext *ctx)
{
    // Shift dataIndex to a nonnegative index for cache lookup.
    int shiftedIndex = dataIndex + ctx->halfWindowSize;

    // Check if the shifted index falls outside the range supported by the cache.
    if (shiftedIndex < 0 || shiftedIndex >= (2 * MAX_HALF_WINDOW_FOR_MEMO + 1))
    {
        // If it's out of range, compute the value directly without memoization.
        return GramPolyIterative(polynomialOrder, dataIndex, ctx);
    }

    // If the polynomial order or derivative order exceeds our cache capacity,
    // fall back to the iterative computation.
    if (polynomialOrder >= MAX_POLY_ORDER_FOR_MEMO || ctx->derivativeOrder >= MAX_DERIVATIVE_FOR_MEMO)
    {
        return GramPolyIterative(polynomialOrder, dataIndex, ctx);
    }

    // Check if the value for these parameters is already computed.
    if (gramPolyCache[shiftedIndex][polynomialOrder][ctx->derivativeOrder].isComputed)
    {
        // Return the cached value.
        return gramPolyCache[shiftedIndex][polynomialOrder][ctx->derivativeOrder].value;
    }

    // Compute the Gram polynomial using the iterative method.
    float value = GramPolyIterative(polynomialOrder, dataIndex, ctx);

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
 * OPTIMIZATION: Uses GenFact lookup table instead of computing on the fly.
 *
 * @param dataIndex The shifted data index (relative to the window center).
 * @param targetPoint The target point within the window.
 * @param polynomialOrder The order of the polynomial.
 * @param ctx Pointer to a GramPolyContext containing filter parameters.
 * @return The computed weight for the data index.
 */
static float Weight(int dataIndex, int targetPoint, uint8_t polynomialOrder, const GramPolyContext *ctx)
{
    float w = 0.0f; // Initialize weight accumulator.
    
    //Precompute common value
    uint8_t twoM = 2 * ctx->halfWindowSize;

    // Loop over polynomial orders from 0 to polynomialOrder.
    for (uint8_t k = 0; k <= polynomialOrder; ++k)
    {
#ifdef ENABLE_MEMOIZATION
        // If memoization is enabled, use the cached version.
        float part1 = MemoizedGramPoly(k, dataIndex, ctx);   // Evaluate at data point (derivative order = 0)
        float part2 = MemoizedGramPoly(k, targetPoint, ctx); // Evaluate at target point (with derivative order from ctx)
#else
        // Otherwise, compute the Gram polynomial iteratively without caching.
        float part1 = GramPolyIterative(k, dataIndex, ctx);
        float part2 = GramPolyIterative(k, targetPoint, ctx);
#endif

#ifdef OPTIMIZE_GENFACT
        // If optimized GenFact is enabled, use precomputed numerator/denominator.
        float factor = (2 * k + 1) * (precomputedGenFactNum[k] / precomputedGenFactDen[k]);
#else
        // Use lookup table instead of computing
        float num = GenFact(twoM, k);
        float den = GenFact(twoM + k + 1, k + 1);
        float factor = (2 * k + 1) * (num / den);
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
 * Weights are stored in ALIGNED memory for optimal SIMD performance.
 *
 * @param halfWindowSize Half-window size.
 * @param targetPoint The target point in the window (the point where the fit is evaluated).
 * @param polynomialOrder Polynomial order for fitting.
 * @param derivativeOrder Derivative order for the filter.
 * @param weights Array (size: 2*halfWindowSize+1, ALIGNED) to store computed weights.
 */
void ComputeWeights(uint8_t halfWindowSize, uint8_t targetPoint, uint8_t polynomialOrder, uint8_t derivativeOrder, float *weights)
{
    // Create a GramPolyContext with the current filter parameters.
    GramPolyContext ctx = {halfWindowSize, targetPoint, derivativeOrder};

    // Calculate the full window size (total number of data points in the filter window).
    uint16_t fullWindowSize = 2 * halfWindowSize + 1;

#ifdef OPTIMIZE_GENFACT
    // Precompute the GenFact numerator and denominator factors for the current parameters.
    PrecomputeGenFacts(halfWindowSize, polynomialOrder);
#endif

#ifdef ENABLE_MEMOIZATION
    // Clear the memoization cache to ensure that previous values do not interfere.
    ClearGramPolyCache(halfWindowSize, polynomialOrder, derivativeOrder);
#endif

    // Loop over each index in the filter window.
    // NOTE: This loop is NOT the bottleneck - it's called once per filter configuration.
    // The real hot loop is the convolution in ApplyFilter.
    for (int dataIndex = 0; dataIndex < fullWindowSize; ++dataIndex)
    {
        // Shift the dataIndex so that the center of the window corresponds to 0.
        weights[dataIndex] = Weight(dataIndex - halfWindowSize, targetPoint, polynomialOrder, &ctx);
    }
}

//-------------------------
// NEW: Edge Weight Caching (Static Allocation)
//-------------------------
typedef struct {
    float weights[MAX_WINDOW];
    uint8_t halfWindowSize;
    uint8_t polynomialOrder;
    uint8_t derivativeOrder;
    uint8_t targetPoint;
    bool valid;
} EdgeWeightCache;

// Pre-allocate caches for edge weights (static allocation only)
static EdgeWeightCache leadingEdgeCache[MAX_HALF_WINDOW_FOR_MEMO];
static bool edgeCacheInitialized = false;

/**
 * @brief Initialize edge cache (called once at first use).
 */
static void InitEdgeCacheIfNeeded(void)
{
    if (!edgeCacheInitialized) {
        for (int i = 0; i < MAX_HALF_WINDOW_FOR_MEMO; i++) {
            leadingEdgeCache[i].valid = false;
        }
        edgeCacheInitialized = true;
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
SavitzkyGolayFilter initFilter(uint8_t halfWindowSize, uint8_t polynomialOrder, uint8_t targetPoint, uint8_t derivativeOrder, float time_step)
{
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
// SIMD-READY: Filter Application
//-------------------------

//=============================================================================
// MAIN FILTER APPLICATION (FULLY SIMD-OPTIMIZED)
//=============================================================================

/**
 * @brief Apply Savitzky-Golay filter with complete SIMD optimization
 *
 * @details
 * ALL three regions now use SIMD:
 * - Center region: SAVGOL_DOT_PRODUCT (forward access)
 * - Trailing edge: SAVGOL_DOT_PRODUCT (forward access)
 * - Leading edge: SAVGOL_REVERSE_DOT_PRODUCT (reverse access) <-- NEW!
 *
 * Mathematical logic unchanged - only performance improved.
 */
static void ApplyFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                       uint16_t targetPoint, SavitzkyGolayFilter filter,
                       MqsRawDataPoint_t filteredData[])
{
    // Ensure that the halfWindowSize does not exceed the maximum allowed value
    uint8_t maxHalfWindowSize = (MAX_WINDOW - 1) / 2;
    if (halfWindowSize > maxHalfWindowSize)
    {
        printf("Warning: halfWindowSize (%d) exceeds maximum allowed (%d). Adjusting.\n",
               halfWindowSize, maxHalfWindowSize);
        halfWindowSize = maxHalfWindowSize;
    }

    // Calculate the total number of points in the filter window
    int windowSize = 2 * halfWindowSize + 1;
    int lastIndex = dataSize - 1;
    uint8_t width = halfWindowSize;

    //==========================================================================
    // STEP 1: BOUNDARY OPERATION - Convert AoS to SoA
    //==========================================================================
    // This is done ONCE at the input boundary
    float *soa_data = savgol_alloc_aligned(dataSize);
    float *soa_output = savgol_alloc_aligned(dataSize);
    
    if (!soa_data || !soa_output) {
        LOG_ERROR("Failed to allocate aligned memory for SoA conversion");
        if (soa_data) savgol_free_aligned(soa_data);
        if (soa_output) savgol_free_aligned(soa_output);
        return;
    }
    
    // Extract phaseAngle field into flat array
    savgol_aos_to_soa(data, soa_data, dataSize);

    //==========================================================================
    // STEP 2: WEIGHT COMPUTATION
    //==========================================================================
    float *weights = savgol_alloc_aligned(windowSize);
    if (!weights) {
        LOG_ERROR("Failed to allocate aligned memory for weights");
        savgol_free_aligned(soa_data);
        savgol_free_aligned(soa_output);
        return;
    }

    // Compute weights for the central window (called ONCE per filter configuration)
    ComputeWeights(halfWindowSize, targetPoint, filter.conf.polynomialOrder,
                   filter.conf.derivativeOrder, weights);

    //==========================================================================
    // STEP 3: CENTER REGION CONVOLUTION (THE HOT LOOP)
    //==========================================================================
    // This is where 90%+ of compute time is spent
    // Uses SIMD dot product kernel which auto-dispatches to AVX-512/AVX2/SSE2/scalar
    
    for (int i = 0; i <= (int)dataSize - windowSize; ++i)
    {
        // SIMD dot product: weights · data
        float result = SAVGOL_DOT_PRODUCT(weights, &soa_data[i], windowSize);
        soa_output[i + width] = result;
    }

    //==========================================================================
    // STEP 4: EDGE HANDLING (NOW FULLY SIMD!)
    //==========================================================================
    InitEdgeCacheIfNeeded();
    
    float *edgeWeights = savgol_alloc_aligned(windowSize);
    if (!edgeWeights) {
        LOG_ERROR("Failed to allocate memory for edge weights");
        savgol_free_aligned(weights);
        savgol_free_aligned(soa_data);
        savgol_free_aligned(soa_output);
        return;
    }
    
    for (int i = 0; i < width; ++i)
    {
        uint8_t target = width - i;
        
        // Check if we can use cached weights
        bool useCache = false;
        const float *current_edge_weights = NULL;
        
        if (i < MAX_HALF_WINDOW_FOR_MEMO && leadingEdgeCache[i].valid &&
            leadingEdgeCache[i].halfWindowSize == halfWindowSize &&
            leadingEdgeCache[i].polynomialOrder == filter.conf.polynomialOrder &&
            leadingEdgeCache[i].derivativeOrder == filter.conf.derivativeOrder &&
            leadingEdgeCache[i].targetPoint == target)
        {
            useCache = true;
            current_edge_weights = leadingEdgeCache[i].weights;
        }
        else
        {
            // Compute fresh weights
            ComputeWeights(halfWindowSize, target, filter.conf.polynomialOrder,
                          filter.conf.derivativeOrder, edgeWeights);
            
            // Cache if possible
            if (i < MAX_HALF_WINDOW_FOR_MEMO)
            {
                memcpy(leadingEdgeCache[i].weights, edgeWeights, windowSize * sizeof(float));
                leadingEdgeCache[i].halfWindowSize = halfWindowSize;
                leadingEdgeCache[i].polynomialOrder = filter.conf.polynomialOrder;
                leadingEdgeCache[i].derivativeOrder = filter.conf.derivativeOrder;
                leadingEdgeCache[i].targetPoint = target;
                leadingEdgeCache[i].valid = true;
            }
            
            current_edge_weights = edgeWeights;
        }
        
        //----------------------------------------------------------------------
        // Leading Edge: REVERSED access pattern (NOW USING SIMD!)
        //----------------------------------------------------------------------
        // Mathematical operation: weights[0]*data[N-1] + weights[1]*data[N-2] + ...
        // 
        // BEFORE (scalar loop):
        //   for (int j = 0; j < windowSize; j++)
        //       sum += weights[j] * soa_data[windowSize - 1 - j];
        //
        // AFTER (SIMD):
        //   Use optimized reverse dot product kernel
        //
        // Mathematical equivalence:
        //   soa_data[windowSize-1-j] == (&soa_data[windowSize-1])[-j]
        //
        float leadingSum = SAVGOL_REVERSE_DOT_PRODUCT(
            current_edge_weights,
            &soa_data[windowSize - 1],  // Point to LAST element of window
            windowSize
        );
        soa_output[i] = leadingSum;
        
        //----------------------------------------------------------------------
        // Trailing Edge: FORWARD access pattern (already SIMD-optimized)
        //----------------------------------------------------------------------
        float trailingSum = SAVGOL_DOT_PRODUCT(
            current_edge_weights, 
            &soa_data[lastIndex - windowSize + 1], 
            windowSize
        );
        soa_output[lastIndex - i] = trailingSum;
    }

    //==========================================================================
    // STEP 5: BOUNDARY OPERATION - Convert SoA back to AoS
    //==========================================================================
    // This is done ONCE at the output boundary
    savgol_soa_to_aos(soa_output, filteredData, dataSize);

    //==========================================================================
    // CLEANUP
    //==========================================================================
    savgol_free_aligned(weights);
    savgol_free_aligned(edgeWeights);
    savgol_free_aligned(soa_data);
    savgol_free_aligned(soa_output);
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
                     uint8_t targetPoint, uint8_t derivativeOrder)
{
    // Assertions for development to catch invalid parameters early.
    assert(data != NULL && "Input data pointer must not be NULL");
    assert(filteredData != NULL && "Filtered data pointer must not be NULL");
    assert(dataSize > 0 && "Data size must be greater than 0");
    assert(halfWindowSize > 0 && "Half-window size must be greater than 0");
    assert((2 * halfWindowSize + 1) <= dataSize && "Filter window size must not exceed data size");
    assert(polynomialOrder < (2 * halfWindowSize + 1) && "Polynomial order must be less than the filter window size");
    assert(targetPoint <= (2 * halfWindowSize) && "Target point must be within the filter window");

    // Runtime checks with error logging.
    if (data == NULL || filteredData == NULL)
    {
        LOG_ERROR("NULL pointer passed to mes_savgolFilter.");
        return -1;
    }
    if (dataSize == 0 || halfWindowSize == 0 ||
        polynomialOrder >= 2 * halfWindowSize + 1 ||
        targetPoint > 2 * halfWindowSize ||
        (2 * halfWindowSize + 1) > dataSize)
    {
        LOG_ERROR("Invalid filter parameters provided: dataSize=%zu, halfWindowSize=%d, polynomialOrder=%d, targetPoint=%d.",
                  dataSize, halfWindowSize, polynomialOrder, targetPoint);
        return -2;
    }

    SavitzkyGolayFilter filter = initFilter(halfWindowSize, polynomialOrder, targetPoint, derivativeOrder, 1.0f);
    ApplyFilter(data, dataSize, halfWindowSize, targetPoint, filter, filteredData);

    return 0;
}