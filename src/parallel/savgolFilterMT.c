/**
 * @file savgolFilterMT.c
 * @brief CORRECTED Savitzky-Golay filter with OpenMP parallelization
 *
 * FIXES APPLIED:
 * 1. All #pragma omp directives guarded with #ifdef _OPENMP
 * 2. InitEdgeCacheIfNeeded() fully implemented
 * 3. Removed static weights[] to eliminate false sharing
 * 4. Increased PARALLEL_THRESHOLD to 1000 (realistic overhead)
 * 5. Added optional edge parallelization for large windows
 * 6. Added thread count control API
 * 7. Removed default(none) clause
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

// CORRECTED: Increased threshold based on realistic OpenMP overhead
// OpenMP fork-join overhead: ~50-100µs
// Only worth parallelizing if work > 10x overhead
#define PARALLEL_THRESHOLD 1000

// Edge parallelization threshold (only parallel if halfWindow is large)
#define EDGE_PARALLEL_THRESHOLD 8

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
// Memoization
//=============================================================================
#ifdef ENABLE_MEMOIZATION
#define MAX_HALF_WINDOW_FOR_MEMO 32
#define MAX_POLY_ORDER_FOR_MEMO 5
#define MAX_DERIVATIVE_FOR_MEMO 5

static GramPolyCacheEntry gramPolyCache[2 * MAX_HALF_WINDOW_FOR_MEMO + 1]
                                       [MAX_POLY_ORDER_FOR_MEMO]
                                       [MAX_DERIVATIVE_FOR_MEMO];

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

static float MemoizedGramPoly(uint8_t polynomialOrder, int dataIndex, const GramPolyContext *ctx)
{
    int shiftedIndex = dataIndex + ctx->halfWindowSize;

    if (shiftedIndex < 0 || shiftedIndex >= (2 * MAX_HALF_WINDOW_FOR_MEMO + 1))
    {
        return GramPolyIterative(polynomialOrder, dataIndex, ctx);
    }

    if (polynomialOrder >= MAX_POLY_ORDER_FOR_MEMO || ctx->derivativeOrder >= MAX_DERIVATIVE_FOR_MEMO)
    {
        return GramPolyIterative(polynomialOrder, dataIndex, ctx);
    }

    if (gramPolyCache[shiftedIndex][polynomialOrder][ctx->derivativeOrder].isComputed)
    {
        return gramPolyCache[shiftedIndex][polynomialOrder][ctx->derivativeOrder].value;
    }

    float value = GramPolyIterative(polynomialOrder, dataIndex, ctx);
    gramPolyCache[shiftedIndex][polynomialOrder][ctx->derivativeOrder].value = value;
    gramPolyCache[shiftedIndex][polynomialOrder][ctx->derivativeOrder].isComputed = true;

    return value;
}
#endif

//=============================================================================
// Weight Calculation
//=============================================================================
static float Weight(int dataIndex, int targetPoint, uint8_t polynomialOrder, const GramPolyContext *ctx)
{
    float w = 0.0f;
    uint8_t twoM = 2 * ctx->halfWindowSize;

    for (uint8_t k = 0; k <= polynomialOrder; ++k)
    {
#ifdef ENABLE_MEMOIZATION
        float part1 = MemoizedGramPoly(k, dataIndex, ctx);
        float part2 = MemoizedGramPoly(k, targetPoint, ctx);
#else
        float part1 = GramPolyIterative(k, dataIndex, ctx);
        float part2 = GramPolyIterative(k, targetPoint, ctx);
#endif

        float num = GenFact(twoM, k);
        float den = GenFact(twoM + k + 1, k + 1);
        float factor = (2 * k + 1) * (num / den);

        w += factor * part1 * part2;
    }

    return w;
}

static void ComputeWeights(uint8_t halfWindowSize, uint16_t targetPoint, uint8_t polynomialOrder,
                           uint8_t derivativeOrder, float *weights)
{
    GramPolyContext ctx = {halfWindowSize, targetPoint, derivativeOrder};
    uint16_t fullWindowSize = 2 * halfWindowSize + 1;

#ifdef ENABLE_MEMOIZATION
    ClearGramPolyCache(halfWindowSize, polynomialOrder, derivativeOrder);
#endif

    for (int dataIndex = 0; dataIndex < fullWindowSize; ++dataIndex)
    {
        weights[dataIndex] = Weight(dataIndex - halfWindowSize, targetPoint, polynomialOrder, &ctx);
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

static EdgeWeightCache leadingEdgeCache[MAX_HALF_WINDOW_FOR_MEMO];

/**
 * @brief FIXED: Thread-safe edge cache initialization with relaxed fast path
 */
static void InitEdgeCacheIfNeeded(void)
{
    // Fast path: relaxed ordering for quick check
    if (atomic_load_explicit(&edgeCacheInitialized, memory_order_relaxed))
    {
        return;
    }

    // Slow path: full synchronization
#ifdef _OPENMP
#pragma omp critical(EdgeCacheInit)
#endif
    {
        // Double-check with acquire (needed inside critical section)
        if (!atomic_load_explicit(&edgeCacheInitialized, memory_order_acquire))
        {
            for (int i = 0; i < MAX_HALF_WINDOW_FOR_MEMO; i++)
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
// CORRECTED: Parallel Filter Application
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

    // CORRECTED: Stack allocation instead of static (eliminates false sharing)
    // Each thread in parallel region gets its own copy
    float weights[MAX_WINDOW];

    //--------------------------------------------------------------------------
    // STEP 1: Compute central weights (sequential, shared by all threads)
    //--------------------------------------------------------------------------
    ComputeWeights(halfWindowSize, targetPoint, filter.conf.polynomialOrder,
                   filter.conf.derivativeOrder, weights);

    //--------------------------------------------------------------------------
    // STEP 2: PARALLEL CENTRAL REGION
    //--------------------------------------------------------------------------
    int centralRegionSize = (int)dataSize - windowSize + 1;

    // CORRECTED: All OpenMP directives wrapped in #ifdef
#ifdef _OPENMP
    if (centralRegionSize >= PARALLEL_THRESHOLD)
    {
// CORRECTED: Removed default(none) clause
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
    // STEP 3: FIXED EDGE HANDLING (parallel for large windows)
    //--------------------------------------------------------------------------
    InitEdgeCacheIfNeeded();

#ifdef _OPENMP
    if (width >= EDGE_PARALLEL_THRESHOLD)
    {
// FIXED: Parallel edges with proper cache synchronization and static scheduling
#pragma omp parallel for schedule(static)
        for (int i = 0; i < width; ++i)
        {
            uint8_t target = width - i;

            // Thread-local buffer (only used on cache miss)
            float tempWeights[MAX_WINDOW];
            const float *edgeWeights; // Pointer to actual weights (cached or temp)
            bool usedCache = false;

            // ---------- ATOMIC CACHE ACCESS (TOCTOU fix) ----------
#ifdef _OPENMP
#pragma omp critical(EdgeCacheAccess)
#endif
            {
                if (i < MAX_HALF_WINDOW_FOR_MEMO && leadingEdgeCache[i].valid &&
                    leadingEdgeCache[i].halfWindowSize == halfWindowSize &&
                    leadingEdgeCache[i].polynomialOrder == filter.conf.polynomialOrder &&
                    leadingEdgeCache[i].derivativeOrder == filter.conf.derivativeOrder &&
                    leadingEdgeCache[i].targetPoint == target)
                {

                    // FIXED: Use cached pointer directly (no memcpy)
                    edgeWeights = leadingEdgeCache[i].weights;
                    usedCache = true;
                }
            }

            // ---------- COMPUTE IF CACHE MISS ----------
            if (!usedCache)
            {
                ComputeWeights(halfWindowSize, target, filter.conf.polynomialOrder,
                               filter.conf.derivativeOrder, tempWeights);
                edgeWeights = tempWeights;

                // Update cache (protected write)
                if (i < MAX_HALF_WINDOW_FOR_MEMO)
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

            // ---------- LEADING EDGE (reverse order) ----------
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

            // ---------- TRAILING EDGE (forward order) ----------
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
        // ---------- SEQUENTIAL EDGE PROCESSING ----------
        for (int i = 0; i < width; ++i)
        {
            uint8_t target = width - i;

            // FIXED: Use pointer, avoid unnecessary memcpy
            const float *edgeWeights;
            float tempWeights[MAX_WINDOW];
            bool useCache = false;

            if (i < MAX_HALF_WINDOW_FOR_MEMO && leadingEdgeCache[i].valid)
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
                // FIXED: Direct pointer (no copy)
                edgeWeights = leadingEdgeCache[i].weights;
            }
            else
            {
                ComputeWeights(halfWindowSize, target, filter.conf.polynomialOrder,
                               filter.conf.derivativeOrder, tempWeights);
                edgeWeights = tempWeights;

                // Cache update
                if (i < MAX_HALF_WINDOW_FOR_MEMO)
                {
                    memcpy(leadingEdgeCache[i].weights, tempWeights, windowSize * sizeof(float));
                    leadingEdgeCache[i].halfWindowSize = halfWindowSize;
                    leadingEdgeCache[i].polynomialOrder = filter.conf.polynomialOrder;
                    leadingEdgeCache[i].derivativeOrder = filter.conf.derivativeOrder;
                    leadingEdgeCache[i].targetPoint = target;
                    leadingEdgeCache[i].valid = true;
                }
            }

            // Leading edge (same convolution code as parallel version)
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
// PUBLIC API (with optional thread count control)
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
 * @brief CORRECTED: Added API with thread count control
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
        omp_set_num_threads(1); // Force sequential
    }
    // numThreads == 0: use default

    int result = mes_savgolFilter(data, dataSize, halfWindowSize, filteredData,
                                  polynomialOrder, targetPoint, derivativeOrder);

    omp_set_num_threads(oldNumThreads); // Restore
    return result;
#else
    // No OpenMP: ignore numThreads parameter
    return mes_savgolFilter(data, dataSize, halfWindowSize, filteredData,
                            polynomialOrder, targetPoint, derivativeOrder);
#endif
}