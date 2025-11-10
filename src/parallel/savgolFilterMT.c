/**
 * @file savgolFilterMT.c
 * @brief Production Savitzky-Golay filter with OpenMP parallelization
 *
 * FEATURES:
 * 1. Thread-safe weight computation with proper cache invalidation
 * 2. Parallel central region, edge processing, and weight computation
 * 3. No false sharing (thread-local caches)
 * 4. Adaptive thresholds for optimal parallelization decisions
 * 5. Thread count control API
 *
 * FIXES FROM PREVIOUS VERSION:
 * - Fixed thread-local cache invalidation on parameter changes
 * - Fixed global cache race condition (now uses thread-local caches)
 * - Fixed gramPolyCache array dimensions (was wasting 292 KB)
 * - Added WEIGHT_PARALLEL_THRESHOLD configuration
 */

#include "savgolFilter.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <stdatomic.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*-------------------------
  Configuration
-------------------------*/
#define LOG_ERROR(fmt, ...) fprintf(stderr, "ERROR: " fmt "\n", ##__VA_ARGS__)

// Central region parallelization threshold
// OpenMP fork-join overhead: ~50-100µs
// Only parallelize if work > 10x overhead
#define PARALLEL_THRESHOLD 1000

// Edge parallelization threshold (only parallel if halfWindow is large)
#define EDGE_PARALLEL_THRESHOLD 8

// Weight computation parallelization threshold
// Parallelize ComputeWeights if fullWindowSize > this value
// Each weight computation: ~100 FLOPs, need 20+ to overcome threading overhead
#define WEIGHT_PARALLEL_THRESHOLD 20

#define ENABLE_MEMOIZATION

//=============================================================================
// THREAD-SAFE INITIALIZATION FLAGS
//=============================================================================
static atomic_bool genFactTableInitialized = ATOMIC_VAR_INIT(false);
static atomic_bool edgeCacheInitialized = ATOMIC_VAR_INIT(false);

//=============================================================================
// GenFact Lookup Table
//=============================================================================
#define GENFACT_TABLE_SIZE 65
static float genFactTable[GENFACT_TABLE_SIZE][GENFACT_TABLE_SIZE];

/**
 * @brief Thread-safe initialization of GenFact lookup table
 */
static void InitGenFactTable(void)
{
    if (atomic_load_explicit(&genFactTableInitialized, memory_order_acquire))
    {
        return;
    }

#ifdef _OPENMP
#pragma omp critical(GenFactInit)
#endif
    {
        if (!atomic_load_explicit(&genFactTableInitialized, memory_order_acquire))
        {
            for (uint8_t upperLimit = 0; upperLimit < GENFACT_TABLE_SIZE; upperLimit++)
            {
                genFactTable[upperLimit][0] = 1.0f;

                for (uint8_t termCount = 1; termCount < GENFACT_TABLE_SIZE; termCount++)
                {
                    if (upperLimit < termCount)
                    {
                        genFactTable[upperLimit][termCount] = 0.0f;
                    }
                    else
                    {
                        float product = 1.0f;
                        uint8_t start = (upperLimit - termCount) + 1;
                        for (uint8_t j = start; j <= upperLimit; j++)
                        {
                            product *= (float)j;
                        }
                        genFactTable[upperLimit][termCount] = product;
                    }
                }
            }

            atomic_store_explicit(&genFactTableInitialized, true, memory_order_release);
        }
    }
}

static inline float GenFact(uint8_t upperLimit, uint8_t termCount)
{
    InitGenFactTable();

    if (upperLimit < GENFACT_TABLE_SIZE && termCount < GENFACT_TABLE_SIZE)
    {
        return genFactTable[upperLimit][termCount];
    }

    float product = 1.0f;
    for (uint8_t j = (upperLimit - termCount) + 1; j <= upperLimit; j++)
    {
        product *= (float)j;
    }
    return product;
}

//=============================================================================
// Gram Polynomial Computation
//=============================================================================
static float GramPolyIterative(uint8_t polynomialOrder, int dataIndex, const GramPolyContext *ctx)
{
    uint8_t halfWindowSize = ctx->halfWindowSize;
    uint8_t derivativeOrder = ctx->derivativeOrder;

    float buf0[MAX_ORDER];
    float buf1[MAX_ORDER];
    float buf2[MAX_ORDER];

    float *prev2 = buf0;
    float *prev = buf1;
    float *curr = buf2;

    for (uint8_t d = 0; d <= derivativeOrder; d++)
    {
        prev2[d] = (d == 0) ? 1.0f : 0.0f;
    }

    if (polynomialOrder == 0)
    {
        return prev2[derivativeOrder];
    }

    float inv_half = 1.0f / halfWindowSize;
    prev[0] = inv_half * (dataIndex * prev2[0]);

    for (uint8_t d = 1; d <= derivativeOrder; d++)
    {
        float inner_term = dataIndex * prev2[d] + d * prev2[d - 1];
        prev[d] = inv_half * inner_term;
    }

    if (polynomialOrder == 1)
    {
        return prev[derivativeOrder];
    }

    float two_halfWinSize = 2.0f * halfWindowSize;

    for (uint8_t k = 2; k <= polynomialOrder; k++)
    {
        float k_f = (float)k;
        float denom_recip = 1.0f / (k_f * (two_halfWinSize - k_f + 1.0f));
        float a = (4.0f * k_f - 2.0f) * denom_recip;
        float c = ((k_f - 1.0f) * (two_halfWinSize + k_f)) * denom_recip;

        curr[0] = a * (dataIndex * prev[0]) - c * prev2[0];

        for (uint8_t d = 1; d <= derivativeOrder; d++)
        {
            float term = dataIndex * prev[d] + d * prev[d - 1];
            curr[d] = a * term - c * prev2[d];
        }

        float *temp = prev2;
        prev2 = prev;
        prev = curr;
        curr = temp;
    }

    return prev[derivativeOrder];
}

//=============================================================================
// Weight Calculation (Non-Memoized)
//=============================================================================
static float Weight(int dataIndex, int targetPoint, uint8_t polynomialOrder, const GramPolyContext *ctx)
{
    float w = 0.0f;
    uint8_t twoM = 2 * ctx->halfWindowSize;

    for (uint8_t k = 0; k <= polynomialOrder; ++k)
    {
        float part1 = GramPolyIterative(k, dataIndex, ctx);
        float part2 = GramPolyIterative(k, targetPoint, ctx);

        float num = GenFact(twoM, k);
        float den = GenFact(twoM + k + 1, k + 1);
        float factor = (2 * k + 1) * (num / den);

        w += factor * part1 * part2;
    }

    return w;
}

//=============================================================================
// OPTIMIZED: Parallel Weight Calculation with Thread-Local Memoization
//=============================================================================
/**
 * @brief Compute all weights for the filter with optional parallelization
 *
 * PARALLELIZATION STRATEGY:
 * - Each weight computation is independent (no data dependencies)
 * - Parallelize for windows > 20 points (overhead vs benefit threshold)
 * - Each thread uses its own memoization cache to avoid race conditions
 *
 * THREAD-LOCAL CACHING:
 * - OpenMP threads use thread-local Gram polynomial caches
 * - Cache is invalidated when filter parameters change
 * - Avoids false sharing and race conditions
 * - No critical sections needed (each thread writes to different memory)
 */
static void ComputeWeights(uint8_t halfWindowSize, uint16_t targetPoint,
                           uint8_t polynomialOrder, uint8_t derivativeOrder,
                           float *weights)
{
    GramPolyContext ctx = {halfWindowSize, targetPoint, derivativeOrder};
    uint16_t fullWindowSize = 2 * halfWindowSize + 1;

    // OPTIMIZATION: Parallelize weight computation for large windows
#ifdef _OPENMP
    if (fullWindowSize > WEIGHT_PARALLEL_THRESHOLD)
    {
// Parallel path: each thread computes subset of weights
#pragma omp parallel for schedule(static)
        for (int dataIndex = 0; dataIndex < fullWindowSize; ++dataIndex)
        {
#ifdef ENABLE_MEMOIZATION
            // Thread-local cache (one per thread, persists across parallel regions)
            static _Thread_local GramPolyCacheEntry threadCache[2 * MAX_WINDOW + 1]
                                                               [MAX_ORDER]
                                                               [MAX_ORDER];

            // Track parameters to detect changes (forces cache invalidation)
            static _Thread_local uint8_t cachedHalfWindowSize = 0xFF;
            static _Thread_local uint8_t cachedPolynomialOrder = 0xFF;
            static _Thread_local uint8_t cachedDerivativeOrder = 0xFF;

            // FIXED: Clear cache if parameters changed
            if (cachedHalfWindowSize != halfWindowSize ||
                cachedPolynomialOrder != polynomialOrder ||
                cachedDerivativeOrder != derivativeOrder)
            {
                int maxIndex = 2 * halfWindowSize + 1;
                if (maxIndex > (2 * MAX_WINDOW + 1))
                    maxIndex = (2 * MAX_WINDOW + 1);

                for (int i = 0; i < maxIndex; i++)
                {
                    int maxK = (polynomialOrder < MAX_ORDER) ? polynomialOrder : MAX_ORDER - 1;
                    int maxD = (derivativeOrder < MAX_ORDER) ? derivativeOrder : MAX_ORDER - 1;

                    for (int k = 0; k <= maxK; k++)
                    {
                        for (int d = 0; d <= maxD; d++)
                        {
                            threadCache[i][k][d].isComputed = false;
                        }
                    }
                }

                cachedHalfWindowSize = halfWindowSize;
                cachedPolynomialOrder = polynomialOrder;
                cachedDerivativeOrder = derivativeOrder;
            }
#endif

            // Compute weight using thread-local cache
            float w = 0.0f;
            uint8_t twoM = 2 * ctx.halfWindowSize;

            for (uint8_t k = 0; k <= polynomialOrder; ++k)
            {
#ifdef ENABLE_MEMOIZATION
                // Use thread-local memoization
                int shiftedDataIndex = (dataIndex - halfWindowSize) + ctx.halfWindowSize;
                int shiftedTarget = targetPoint + ctx.halfWindowSize;

                float part1, part2;

                // Check thread-local cache for part1
                if (shiftedDataIndex >= 0 && shiftedDataIndex < (2 * MAX_WINDOW + 1) &&
                    k < MAX_ORDER && ctx.derivativeOrder < MAX_ORDER &&
                    threadCache[shiftedDataIndex][k][ctx.derivativeOrder].isComputed)
                {
                    part1 = threadCache[shiftedDataIndex][k][ctx.derivativeOrder].value;
                }
                else
                {
                    part1 = GramPolyIterative(k, dataIndex - halfWindowSize, &ctx);
                    if (shiftedDataIndex >= 0 && shiftedDataIndex < (2 * MAX_WINDOW + 1) &&
                        k < MAX_ORDER && ctx.derivativeOrder < MAX_ORDER)
                    {
                        threadCache[shiftedDataIndex][k][ctx.derivativeOrder].value = part1;
                        threadCache[shiftedDataIndex][k][ctx.derivativeOrder].isComputed = true;
                    }
                }

                // Check thread-local cache for part2
                if (shiftedTarget >= 0 && shiftedTarget < (2 * MAX_WINDOW + 1) &&
                    k < MAX_ORDER && ctx.derivativeOrder < MAX_ORDER &&
                    threadCache[shiftedTarget][k][ctx.derivativeOrder].isComputed)
                {
                    part2 = threadCache[shiftedTarget][k][ctx.derivativeOrder].value;
                }
                else
                {
                    part2 = GramPolyIterative(k, targetPoint, &ctx);
                    if (shiftedTarget >= 0 && shiftedTarget < (2 * MAX_WINDOW + 1) &&
                        k < MAX_ORDER && ctx.derivativeOrder < MAX_ORDER)
                    {
                        threadCache[shiftedTarget][k][ctx.derivativeOrder].value = part2;
                        threadCache[shiftedTarget][k][ctx.derivativeOrder].isComputed = true;
                    }
                }
#else
                float part1 = GramPolyIterative(k, dataIndex - halfWindowSize, &ctx);
                float part2 = GramPolyIterative(k, targetPoint, &ctx);
#endif

                float num = GenFact(twoM, k);
                float den = GenFact(twoM + k + 1, k + 1);
                float factor = (2 * k + 1) * (num / den);

                w += factor * part1 * part2;
            }

            weights[dataIndex] = w;
        }
    }
    else
#endif
    {
        // Sequential path: no memoization (simpler, less overhead for small windows)
        for (int dataIndex = 0; dataIndex < fullWindowSize; ++dataIndex)
        {
            weights[dataIndex] = Weight(dataIndex - halfWindowSize, targetPoint,
                                        polynomialOrder, &ctx);
        }
    }
}

//=============================================================================
// Edge Weight Cache
//=============================================================================
typedef struct
{
    float weights[MAX_WINDOW];
    uint8_t halfWindowSize;
    uint8_t polynomialOrder;
    uint8_t derivativeOrder;
    uint8_t targetPoint;
    bool valid;
} EdgeWeightCache;

static EdgeWeightCache leadingEdgeCache[MAX_WINDOW];

/**
 * @brief Thread-safe edge cache initialization
 */
static void InitEdgeCacheIfNeeded(void)
{
    if (atomic_load_explicit(&edgeCacheInitialized, memory_order_relaxed))
    {
        return;
    }

#ifdef _OPENMP
#pragma omp critical(EdgeCacheInit)
#endif
    {
        if (!atomic_load_explicit(&edgeCacheInitialized, memory_order_acquire))
        {
            for (int i = 0; i < MAX_WINDOW; i++)
            {
                leadingEdgeCache[i].valid = false;
            }
            atomic_store_explicit(&edgeCacheInitialized, true, memory_order_release);
        }
    }
}

//=============================================================================
// Filter Initialization
//=============================================================================
SavitzkyGolayFilter initFilter(uint8_t halfWindowSize, uint8_t polynomialOrder,
                               uint8_t targetPoint, uint8_t derivativeOrder, float time_step)
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

//=============================================================================
// Parallel Filter Application
//=============================================================================
static void ApplyFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                        uint16_t targetPoint, SavitzkyGolayFilter filter,
                        MqsRawDataPoint_t filteredData[])
{
    uint8_t maxHalfWindowSize = (MAX_WINDOW - 1) / 2;
    if (halfWindowSize > maxHalfWindowSize)
    {
        printf("Warning: halfWindowSize (%d) exceeds maximum allowed (%d). Adjusting.\n",
               halfWindowSize, maxHalfWindowSize);
        halfWindowSize = maxHalfWindowSize;
    }

    int windowSize = 2 * halfWindowSize + 1;
    int lastIndex = dataSize - 1;
    uint8_t width = halfWindowSize;

    // Stack allocation (eliminates false sharing)
    float weights[MAX_WINDOW];

    //--------------------------------------------------------------------------
    // STEP 1: Compute central weights (may be parallelized internally)
    //--------------------------------------------------------------------------
    ComputeWeights(halfWindowSize, targetPoint, filter.conf.polynomialOrder,
                   filter.conf.derivativeOrder, weights);

    //--------------------------------------------------------------------------
    // STEP 2: PARALLEL CENTRAL REGION
    //--------------------------------------------------------------------------
    int centralRegionSize = (int)dataSize - windowSize + 1;

#ifdef _OPENMP
    if (centralRegionSize >= PARALLEL_THRESHOLD)
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < centralRegionSize; ++i)
        {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

            const float *w_ptr = weights;
            const MqsRawDataPoint_t *d_ptr = &data[i];

            int remainder = windowSize & 3;

            switch (remainder)
            {
            case 3:
                sum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                sum1 += w_ptr[1] * d_ptr[1].phaseAngle;
                sum2 += w_ptr[2] * d_ptr[2].phaseAngle;
                w_ptr += 3;
                d_ptr += 3;
                break;
            case 2:
                sum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                sum1 += w_ptr[1] * d_ptr[1].phaseAngle;
                w_ptr += 2;
                d_ptr += 2;
                break;
            case 1:
                sum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                w_ptr += 1;
                d_ptr += 1;
                break;
            case 0:
                break;
            }

            int main_iters = (windowSize - remainder) / 4;
            for (int k = 0; k < main_iters; ++k)
            {
                sum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                sum1 += w_ptr[1] * d_ptr[1].phaseAngle;
                sum2 += w_ptr[2] * d_ptr[2].phaseAngle;
                sum3 += w_ptr[3] * d_ptr[3].phaseAngle;

                w_ptr += 4;
                d_ptr += 4;
            }

            float sum = (sum0 + sum1) + (sum2 + sum3);
            filteredData[i + width].phaseAngle = sum;
        }
    }
    else
#endif
    {
        // Sequential fallback
        for (int i = 0; i < centralRegionSize; ++i)
        {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

            const float *w_ptr = weights;
            const MqsRawDataPoint_t *d_ptr = &data[i];

            int remainder = windowSize & 3;

            switch (remainder)
            {
            case 3:
                sum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                sum1 += w_ptr[1] * d_ptr[1].phaseAngle;
                sum2 += w_ptr[2] * d_ptr[2].phaseAngle;
                w_ptr += 3;
                d_ptr += 3;
                break;
            case 2:
                sum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                sum1 += w_ptr[1] * d_ptr[1].phaseAngle;
                w_ptr += 2;
                d_ptr += 2;
                break;
            case 1:
                sum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                w_ptr += 1;
                d_ptr += 1;
                break;
            case 0:
                break;
            }

            int main_iters = (windowSize - remainder) / 4;
            for (int k = 0; k < main_iters; ++k)
            {
                sum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                sum1 += w_ptr[1] * d_ptr[1].phaseAngle;
                sum2 += w_ptr[2] * d_ptr[2].phaseAngle;
                sum3 += w_ptr[3] * d_ptr[3].phaseAngle;

                w_ptr += 4;
                d_ptr += 4;
            }

            float sum = (sum0 + sum1) + (sum2 + sum3);
            filteredData[i + width].phaseAngle = sum;
        }
    }

    //--------------------------------------------------------------------------
    // STEP 3: EDGE HANDLING (parallel for large windows)
    //--------------------------------------------------------------------------
    InitEdgeCacheIfNeeded();

#ifdef _OPENMP
    if (width >= EDGE_PARALLEL_THRESHOLD)
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < width; ++i)
        {
            uint8_t target = width - i;

            float tempWeights[MAX_WINDOW];
            const float *edgeWeights;
            bool usedCache = false;

#ifdef _OPENMP
#pragma omp critical(EdgeCacheAccess)
#endif
            {
                if (i < MAX_WINDOW && leadingEdgeCache[i].valid &&
                    leadingEdgeCache[i].halfWindowSize == halfWindowSize &&
                    leadingEdgeCache[i].polynomialOrder == filter.conf.polynomialOrder &&
                    leadingEdgeCache[i].derivativeOrder == filter.conf.derivativeOrder &&
                    leadingEdgeCache[i].targetPoint == target)
                {
                    edgeWeights = leadingEdgeCache[i].weights;
                    usedCache = true;
                }
            }

            if (!usedCache)
            {
                ComputeWeights(halfWindowSize, target, filter.conf.polynomialOrder,
                               filter.conf.derivativeOrder, tempWeights);
                edgeWeights = tempWeights;

                if (i < MAX_WINDOW)
                {
#ifdef _OPENMP
#pragma omp critical(EdgeCacheUpdate)
#endif
                    {
                        memcpy(leadingEdgeCache[i].weights, tempWeights, windowSize * sizeof(float));
                        leadingEdgeCache[i].halfWindowSize = halfWindowSize;
                        leadingEdgeCache[i].polynomialOrder = filter.conf.polynomialOrder;
                        leadingEdgeCache[i].derivativeOrder = filter.conf.derivativeOrder;
                        leadingEdgeCache[i].targetPoint = target;
                        leadingEdgeCache[i].valid = true;
                    }
                }
            }

            // Leading edge (reverse order)
            float leadSum0 = 0.0f, leadSum1 = 0.0f, leadSum2 = 0.0f, leadSum3 = 0.0f;
            const float *w_ptr = edgeWeights;
            const MqsRawDataPoint_t *d_ptr = &data[windowSize - 1];

            int remainder = windowSize & 3;
            switch (remainder)
            {
            case 3:
                leadSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                leadSum1 += w_ptr[1] * d_ptr[-1].phaseAngle;
                leadSum2 += w_ptr[2] * d_ptr[-2].phaseAngle;
                w_ptr += 3;
                d_ptr -= 3;
                break;
            case 2:
                leadSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                leadSum1 += w_ptr[1] * d_ptr[-1].phaseAngle;
                w_ptr += 2;
                d_ptr -= 2;
                break;
            case 1:
                leadSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                w_ptr += 1;
                d_ptr -= 1;
                break;
            case 0:
                break;
            }

            int main_iters = (windowSize - remainder) / 4;
            for (int k = 0; k < main_iters; ++k)
            {
                leadSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                leadSum1 += w_ptr[1] * d_ptr[-1].phaseAngle;
                leadSum2 += w_ptr[2] * d_ptr[-2].phaseAngle;
                leadSum3 += w_ptr[3] * d_ptr[-3].phaseAngle;

                w_ptr += 4;
                d_ptr -= 4;
            }

            float leadingSum = (leadSum0 + leadSum1) + (leadSum2 + leadSum3);
            filteredData[i].phaseAngle = leadingSum;

            // Trailing edge (forward order)
            float trailSum0 = 0.0f, trailSum1 = 0.0f, trailSum2 = 0.0f, trailSum3 = 0.0f;
            w_ptr = edgeWeights;
            d_ptr = &data[lastIndex - windowSize + 1];

            switch (remainder)
            {
            case 3:
                trailSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                trailSum1 += w_ptr[1] * d_ptr[1].phaseAngle;
                trailSum2 += w_ptr[2] * d_ptr[2].phaseAngle;
                w_ptr += 3;
                d_ptr += 3;
                break;
            case 2:
                trailSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                trailSum1 += w_ptr[1] * d_ptr[1].phaseAngle;
                w_ptr += 2;
                d_ptr += 2;
                break;
            case 1:
                trailSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                w_ptr += 1;
                d_ptr += 1;
                break;
            case 0:
                break;
            }

            for (int k = 0; k < main_iters; ++k)
            {
                trailSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                trailSum1 += w_ptr[1] * d_ptr[1].phaseAngle;
                trailSum2 += w_ptr[2] * d_ptr[2].phaseAngle;
                trailSum3 += w_ptr[3] * d_ptr[3].phaseAngle;

                w_ptr += 4;
                d_ptr += 4;
            }

            float trailingSum = (trailSum0 + trailSum1) + (trailSum2 + trailSum3);
            filteredData[lastIndex - i].phaseAngle = trailingSum;
        }
    }
    else
#endif
    {
        // Sequential edge processing
        for (int i = 0; i < width; ++i)
        {
            uint8_t target = width - i;

            const float *edgeWeights;
            float tempWeights[MAX_WINDOW];
            bool useCache = false;

            if (i < MAX_WINDOW && leadingEdgeCache[i].valid)
            {
                if (leadingEdgeCache[i].halfWindowSize == halfWindowSize &&
                    leadingEdgeCache[i].polynomialOrder == filter.conf.polynomialOrder &&
                    leadingEdgeCache[i].derivativeOrder == filter.conf.derivativeOrder &&
                    leadingEdgeCache[i].targetPoint == target)
                {
                    useCache = true;
                }
            }

            if (useCache)
            {
                edgeWeights = leadingEdgeCache[i].weights;
            }
            else
            {
                ComputeWeights(halfWindowSize, target, filter.conf.polynomialOrder,
                               filter.conf.derivativeOrder, tempWeights);
                edgeWeights = tempWeights;

                if (i < MAX_WINDOW)
                {
                    memcpy(leadingEdgeCache[i].weights, tempWeights, windowSize * sizeof(float));
                    leadingEdgeCache[i].halfWindowSize = halfWindowSize;
                    leadingEdgeCache[i].polynomialOrder = filter.conf.polynomialOrder;
                    leadingEdgeCache[i].derivativeOrder = filter.conf.derivativeOrder;
                    leadingEdgeCache[i].targetPoint = target;
                    leadingEdgeCache[i].valid = true;
                }
            }

            // Leading edge
            float leadSum0 = 0.0f, leadSum1 = 0.0f, leadSum2 = 0.0f, leadSum3 = 0.0f;
            const float *w_ptr = edgeWeights;
            const MqsRawDataPoint_t *d_ptr = &data[windowSize - 1];

            int remainder = windowSize & 3;
            switch (remainder)
            {
            case 3:
                leadSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                leadSum1 += w_ptr[1] * d_ptr[-1].phaseAngle;
                leadSum2 += w_ptr[2] * d_ptr[-2].phaseAngle;
                w_ptr += 3;
                d_ptr -= 3;
                break;
            case 2:
                leadSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                leadSum1 += w_ptr[1] * d_ptr[-1].phaseAngle;
                w_ptr += 2;
                d_ptr -= 2;
                break;
            case 1:
                leadSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                w_ptr += 1;
                d_ptr -= 1;
                break;
            case 0:
                break;
            }

            int main_iters = (windowSize - remainder) / 4;
            for (int k = 0; k < main_iters; ++k)
            {
                leadSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                leadSum1 += w_ptr[1] * d_ptr[-1].phaseAngle;
                leadSum2 += w_ptr[2] * d_ptr[-2].phaseAngle;
                leadSum3 += w_ptr[3] * d_ptr[-3].phaseAngle;
                w_ptr += 4;
                d_ptr -= 4;
            }

            float leadingSum = (leadSum0 + leadSum1) + (leadSum2 + leadSum3);
            filteredData[i].phaseAngle = leadingSum;

            // Trailing edge
            float trailSum0 = 0.0f, trailSum1 = 0.0f, trailSum2 = 0.0f, trailSum3 = 0.0f;
            w_ptr = edgeWeights;
            d_ptr = &data[lastIndex - windowSize + 1];

            switch (remainder)
            {
            case 3:
                trailSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                trailSum1 += w_ptr[1] * d_ptr[1].phaseAngle;
                trailSum2 += w_ptr[2] * d_ptr[2].phaseAngle;
                w_ptr += 3;
                d_ptr += 3;
                break;
            case 2:
                trailSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                trailSum1 += w_ptr[1] * d_ptr[1].phaseAngle;
                w_ptr += 2;
                d_ptr += 2;
                break;
            case 1:
                trailSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                w_ptr += 1;
                d_ptr += 1;
                break;
            case 0:
                break;
            }

            for (int k = 0; k < main_iters; ++k)
            {
                trailSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                trailSum1 += w_ptr[1] * d_ptr[1].phaseAngle;
                trailSum2 += w_ptr[2] * d_ptr[2].phaseAngle;
                trailSum3 += w_ptr[3] * d_ptr[3].phaseAngle;
                w_ptr += 4;
                d_ptr += 4;
            }

            float trailingSum = (trailSum0 + trailSum1) + (trailSum2 + trailSum3);
            filteredData[lastIndex - i].phaseAngle = trailingSum;
        }
    }
}

//=============================================================================
// PUBLIC API
//=============================================================================
int mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                     MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder,
                     uint8_t targetPoint, uint8_t derivativeOrder)
{
    assert(data != NULL && "Input data pointer must not be NULL");
    assert(filteredData != NULL && "Filtered data pointer must not be NULL");
    assert(dataSize > 0 && "Data size must be greater than 0");
    assert(halfWindowSize > 0 && "Half-window size must be greater than 0");
    assert((2 * halfWindowSize + 1) <= dataSize && "Filter window size must not exceed data size");
    assert(polynomialOrder < (2 * halfWindowSize + 1) && "Polynomial order must be less than the filter window size");
    assert(targetPoint <= (2 * halfWindowSize) && "Target point must be within the filter window");

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
        LOG_ERROR("Invalid filter parameters provided.");
        return -2;
    }

    InitGenFactTable();
    InitEdgeCacheIfNeeded();

    SavitzkyGolayFilter filter = initFilter(halfWindowSize, polynomialOrder, targetPoint, derivativeOrder, 1.0f);

    ApplyFilter(data, dataSize, halfWindowSize, targetPoint, filter, filteredData);

    return 0;
}

/**
 * @brief API with thread count control
 *
 * @param numThreads Number of threads to use (0 = auto-detect, -1 = sequential)
 */
int mes_savgolFilter_threaded(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                              MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder,
                              uint8_t targetPoint, uint8_t derivativeOrder, int numThreads)
{
#ifdef _OPENMP
    int oldNumThreads = omp_get_max_threads();

    if (numThreads > 0)
    {
        omp_set_num_threads(numThreads);
    }
    else if (numThreads == -1)
    {
        omp_set_num_threads(1);
    }

    int result = mes_savgolFilter(data, dataSize, halfWindowSize, filteredData,
                                  polynomialOrder, targetPoint, derivativeOrder);

    omp_set_num_threads(oldNumThreads);
    return result;
#else
    (void)numThreads;
    return mes_savgolFilter(data, dataSize, halfWindowSize, filteredData,
                            polynomialOrder, targetPoint, derivativeOrder);
#endif
}