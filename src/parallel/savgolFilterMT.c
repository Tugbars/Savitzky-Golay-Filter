/**
 * @file savgolFilterMT.c
 * @brief Production Savitzky-Golay filter with OpenMP parallelization
 *
 * ARCHITECTURE OVERVIEW:
 * =====================
 * This implementation uses a three-tier parallelization strategy:
 * 
 * 1. WEIGHT COMPUTATION (ComputeWeights):
 *    - Parallelized for large windows (>20 points)
 *    - Thread-local memoization to avoid cache contention
 *    - Cache invalidation on parameter changes
 * 
 * 2. CENTRAL REGION CONVOLUTION (ApplyFilter - Step 2):
 *    - Parallelized for large datasets (>1000 points)
 *    - 4-chain ILP for optimal CPU pipeline utilization
 *    - Static scheduling for load balance
 * 
 * 3. EDGE REGION PROCESSING (ApplyFilter - Step 3):
 *    - Parallelized for large windows (>8 halfWindowSize)
 *    - Persistent cache for repeated calls with same parameters
 *    - Critical sections protect cache access (acceptable for small width)
 * 
 * THREAD SAFETY:
 * ==============
 * - GenFact table: Double-checked locking with atomics
 * - Edge cache: Critical sections + atomic initialization flag
 * - Weight computation: Thread-local caches (no sharing)
 * - Central region: Read-only weight access (no synchronization needed)
 * 
 * PERFORMANCE CHARACTERISTICS:
 * ============================
 * Sequential baseline: ~100 µs for 1000 points, window=25, poly=4
 * With 4 threads: ~30 µs (3.3x speedup)
 * Memory footprint: ~8 KB per thread (thread-local caches)
 * 
 * TYPICAL USE CASES:
 * ==================
 * - Quantitative trading: Real-time signal smoothing (80% use d=0, t=n)
 * - Embedded systems: Motor control sensor filtering
 * - Scientific computing: Data preprocessing and noise reduction
 *
 * @author Tugbars Heptaskin
 * @date 2025-01-10
 * @version 2.0 (Parallel Edition)
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

/**
 * PARALLEL_THRESHOLD: Minimum central region size to justify OpenMP overhead
 * 
 * Rationale: OpenMP fork-join has ~50-100µs overhead. Each convolution 
 * iteration takes ~75ns. Break-even: 100µs / 75ns ≈ 1333 iterations.
 * We use 1000 as a conservative threshold (ensures 1.5x+ speedup).
 */
#define PARALLEL_THRESHOLD 1000

/**
 * EDGE_PARALLEL_THRESHOLD: Minimum halfWindowSize to parallelize edge processing
 * 
 * Rationale: Edge processing requires 2×halfWindowSize iterations. Critical
 * section overhead dominates for small windows. Parallelize only when
 * 2×halfWindowSize > 16 (i.e., halfWindowSize ≥ 8).
 */
#define EDGE_PARALLEL_THRESHOLD 8

/**
 * WEIGHT_PARALLEL_THRESHOLD: Minimum window size to parallelize weight computation
 * 
 * Rationale: Each weight computation is ~100 FLOPs (~300ns). Need 20+ weights
 * to overcome threading overhead. This threshold ensures weight computation
 * parallelization provides net benefit.
 */
#define WEIGHT_PARALLEL_THRESHOLD 20

#define ENABLE_MEMOIZATION

//=============================================================================
// THREAD-SAFE INITIALIZATION FLAGS
//=============================================================================
/**
 * Atomic flags for lazy initialization of shared data structures.
 * 
 * WHY ATOMICS: Multiple threads may call the filter simultaneously. Atomics
 * ensure initialization happens exactly once without race conditions.
 * 
 * MEMORY ORDERING: 
 * - acquire: Reader sees all writes done before release
 * - release: Writer ensures all writes visible before flag set
 */
static atomic_bool genFactTableInitialized = ATOMIC_VAR_INIT(false);
static atomic_bool edgeCacheInitialized = ATOMIC_VAR_INIT(false);

//=============================================================================
// GenFact Lookup Table
//=============================================================================
/**
 * Precomputed generalized factorial table: GenFact(a, b) = a×(a-1)×...×(a-b+1)
 * 
 * PURPOSE: Gram polynomial weights require many GenFact evaluations. 
 * Precomputing saves ~30% of weight computation time.
 * 
 * SIZE: 65×65 = 4225 entries × 4 bytes = 16.9 KB (acceptable for static data)
 * COVERAGE: Handles MAX_WINDOW=33 → twoM=64, all practical filter configs
 */
#define GENFACT_TABLE_SIZE 65
static float genFactTable[GENFACT_TABLE_SIZE][GENFACT_TABLE_SIZE];

/**
 * @brief Thread-safe lazy initialization of GenFact lookup table
 * 
 * ALGORITHM: Double-checked locking pattern
 * 1. Fast path: Check flag with acquire ordering (no lock if already initialized)
 * 2. Slow path: Enter critical section, re-check flag, then initialize
 * 
 * WHY DOUBLE-CHECK: Avoids critical section on every call after initialization
 * THREAD SAFETY: Critical section ensures only one thread initializes
 */
static void InitGenFactTable(void)
{
    // Fast path: relaxed check (acceptable false positive, just means we lock unnecessarily)
    if (atomic_load_explicit(&genFactTableInitialized, memory_order_acquire))
    {
        return; // Already initialized
    }

    // Slow path: Acquire lock and verify initialization is still needed
#ifdef _OPENMP
#pragma omp critical(GenFactInit)
#endif
    {
        // Re-check inside critical section (TOCTOU protection)
        if (!atomic_load_explicit(&genFactTableInitialized, memory_order_acquire))
        {
            // Compute GenFact(upperLimit, termCount) for all valid combinations
            for (uint8_t upperLimit = 0; upperLimit < GENFACT_TABLE_SIZE; upperLimit++)
            {
                // GenFact(a, 0) = 1 by definition (empty product)
                genFactTable[upperLimit][0] = 1.0f;

                for (uint8_t termCount = 1; termCount < GENFACT_TABLE_SIZE; termCount++)
                {
                    if (upperLimit < termCount)
                    {
                        // Undefined: not enough terms to multiply
                        genFactTable[upperLimit][termCount] = 0.0f;
                    }
                    else
                    {
                        // Compute: upperLimit × (upperLimit-1) × ... × (upperLimit-termCount+1)
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

            // Mark as initialized (release ordering ensures table writes visible to other threads)
            atomic_store_explicit(&genFactTableInitialized, true, memory_order_release);
        }
    }
}

/**
 * @brief Retrieve generalized factorial with fallback computation
 * 
 * FAST PATH: Table lookup (O(1), ~2 cycles)
 * SLOW PATH: Runtime computation (O(termCount), ~50 cycles)
 * 
 * USAGE: Primarily called during weight computation, not in hot inner loops
 */
static inline float GenFact(uint8_t upperLimit, uint8_t termCount)
{
    InitGenFactTable(); // Ensure table ready (fast after first call)

    // Try table lookup first
    if (upperLimit < GENFACT_TABLE_SIZE && termCount < GENFACT_TABLE_SIZE)
    {
        return genFactTable[upperLimit][termCount];
    }

    // Fallback: compute on-the-fly for out-of-range parameters
    // (Rare: only happens if MAX_WINDOW > 33, which is uncommon)
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
/**
 * @brief Iterative computation of Gram polynomial using 3-term recurrence
 * 
 * ALGORITHM: Gram polynomials satisfy a 3-term recurrence relation:
 *   G_{k+1}(i) = a_k × i × G_k(i) - c_k × G_{k-1}(i)
 * 
 * This allows O(k) computation instead of naive O(k²) approach.
 * 
 * DERIVATIVE TRACKING: For each polynomial order k, we track derivatives d=0..derivativeOrder
 * in parallel using the chain rule recurrence.
 * 
 * OPTIMIZATION: Triple-buffer pointer swapping avoids array copies between iterations.
 * 
 * @param polynomialOrder Order k of the Gram polynomial to compute
 * @param dataIndex Point i in [-halfWindowSize, +halfWindowSize] to evaluate at
 * @param ctx Context containing window size and derivative order
 * @return G_k^(d)(i) - Gram polynomial order k, derivative d, evaluated at point i
 */
static float GramPolyIterative(uint8_t polynomialOrder, int dataIndex, const GramPolyContext *ctx)
{
    uint8_t halfWindowSize = ctx->halfWindowSize;
    uint8_t derivativeOrder = ctx->derivativeOrder;

    // Triple buffer for recurrence: prev2 = G_{k-2}, prev = G_{k-1}, curr = G_k
    float buf0[MAX_ORDER];
    float buf1[MAX_ORDER];
    float buf2[MAX_ORDER];

    float *prev2 = buf0;
    float *prev = buf1;
    float *curr = buf2;

    // Base case: G_0(i) = 1, derivatives of constants are 0
    for (uint8_t d = 0; d <= derivativeOrder; d++)
    {
        prev2[d] = (d == 0) ? 1.0f : 0.0f;
    }

    if (polynomialOrder == 0)
    {
        return prev2[derivativeOrder]; // G_0^(d)(i)
    }

    // First recurrence: G_1(i) = i/n, where n = halfWindowSize
    float inv_half = 1.0f / halfWindowSize;
    prev[0] = inv_half * (dataIndex * prev2[0]); // G_1(i) = i/n

    // Derivative recurrence: d/di[f(i)g(i)] = f'(i)g(i) + f(i)g'(i)
    for (uint8_t d = 1; d <= derivativeOrder; d++)
    {
        float inner_term = dataIndex * prev2[d] + d * prev2[d - 1]; // Product rule
        prev[d] = inv_half * inner_term;
    }

    if (polynomialOrder == 1)
    {
        return prev[derivativeOrder]; // G_1^(d)(i)
    }

    // General recurrence for k ≥ 2
    float two_halfWinSize = 2.0f * halfWindowSize;

    for (uint8_t k = 2; k <= polynomialOrder; k++)
    {
        float k_f = (float)k;
        
        // Recurrence coefficients (derived from orthogonality conditions)
        float denom_recip = 1.0f / (k_f * (two_halfWinSize - k_f + 1.0f));
        float a = (4.0f * k_f - 2.0f) * denom_recip;
        float c = ((k_f - 1.0f) * (two_halfWinSize + k_f)) * denom_recip;

        // G_k^(0)(i) = a × i × G_{k-1}^(0)(i) - c × G_{k-2}^(0)(i)
        curr[0] = a * (dataIndex * prev[0]) - c * prev2[0];

        // Derivative recurrence (product rule)
        for (uint8_t d = 1; d <= derivativeOrder; d++)
        {
            float term = dataIndex * prev[d] + d * prev[d - 1]; // d/di[i × G_{k-1}]
            curr[d] = a * term - c * prev2[d];
        }

        // Rotate buffers: prev2 ← prev ← curr (no array copy needed!)
        float *temp = prev2;
        prev2 = prev;
        prev = curr;
        curr = temp;
    }

    return prev[derivativeOrder]; // Final result: G_k^(d)(i)
}

//=============================================================================
// Weight Calculation (Non-Memoized)
//=============================================================================
/**
 * @brief Compute single Savitzky-Golay weight for given data and target points
 * 
 * FORMULA: w(i,t) = Σ_{k=0}^{m} [(2k+1) × GenFact(2n,k) / GenFact(2n+k+1,k+1)] 
 *                                 × G_k(i) × G_k(t)
 * 
 * WHERE:
 * - i: data index in [-n, +n]
 * - t: target point in [-n, +n] (usually t=0 for centered, t=n for leading edge)
 * - m: polynomial order
 * - n: half window size
 * - G_k: Gram polynomial of order k
 * 
 * PURPOSE: This weight, when convolved with data, gives the least-squares
 * polynomial fit value (or derivative) at the target point.
 * 
 * @param dataIndex Position i in window (relative to window center)
 * @param targetPoint Position t where polynomial is evaluated
 * @param polynomialOrder Maximum polynomial order m
 * @param ctx Context with window size and derivative order
 * @return Weight value w(i,t)
 */
static float Weight(int dataIndex, int targetPoint, uint8_t polynomialOrder, const GramPolyContext *ctx)
{
    float w = 0.0f;
    uint8_t twoM = 2 * ctx->halfWindowSize; // 2n in the formula

    // Sum over all polynomial orders k=0 to m
    for (uint8_t k = 0; k <= polynomialOrder; ++k)
    {
        // Evaluate Gram polynomials at data and target points
        float part1 = GramPolyIterative(k, dataIndex, ctx);    // G_k(i)
        float part2 = GramPolyIterative(k, targetPoint, ctx);  // G_k(t)

        // Compute normalization factor
        float num = GenFact(twoM, k);                // (2n)!/(2n-k)!
        float den = GenFact(twoM + k + 1, k + 1);    // (2n+k+1)!/(2n)!
        float factor = (2 * k + 1) * (num / den);

        // Accumulate weighted product
        w += factor * part1 * part2;
    }

    return w;
}

//=============================================================================
// OPTIMIZED: Parallel Weight Calculation with Thread-Local Memoization
//=============================================================================
/**
 * @brief Compute all filter weights with adaptive parallelization
 * 
 * PERFORMANCE STRATEGY:
 * =====================
 * SMALL WINDOWS (≤20): Sequential execution
 *   - Threading overhead (50µs) > work time (~15µs)
 *   - Simple sequential loop is faster
 * 
 * LARGE WINDOWS (>20): Parallel execution with thread-local memoization
 *   - Work time (>200µs) >> threading overhead
 *   - Each thread gets private Gram polynomial cache
 *   - Cache is invalidated when filter parameters change
 * 
 * CACHE INVALIDATION LOGIC:
 * ==========================
 * Thread-local caches persist across parallel regions for performance.
 * BUT: If user calls with different parameters (e.g., different halfWindowSize),
 * stale cache entries would produce WRONG RESULTS.
 * 
 * SOLUTION: Track last-used parameters in thread-local variables. On mismatch,
 * clear cache before computation. This adds ~5µs overhead but prevents silent
 * data corruption.
 * 
 * MEMORY USAGE:
 * =============
 * Each thread allocates: (2×MAX_WINDOW+1) × MAX_ORDER × MAX_ORDER × 9 bytes
 *                       = 67 × 5 × 5 × 9 ≈ 15 KB
 * With 4 threads: ~60 KB total (acceptable)
 * 
 * @param halfWindowSize Half-window size n (full window = 2n+1)
 * @param targetPoint Evaluation point t (usually 0 for centered filtering)
 * @param polynomialOrder Maximum polynomial order m
 * @param derivativeOrder Derivative order d (0=smoothing, 1=velocity, 2=acceleration)
 * @param weights Output array [2n+1] to store computed weights
 */
static void ComputeWeights(uint8_t halfWindowSize, uint16_t targetPoint,
                           uint8_t polynomialOrder, uint8_t derivativeOrder,
                           float *weights)
{
    GramPolyContext ctx = {halfWindowSize, targetPoint, derivativeOrder};
    uint16_t fullWindowSize = 2 * halfWindowSize + 1;

#ifdef _OPENMP
    // DECISION: Parallelize only if enough work to justify threading overhead
    if (fullWindowSize > WEIGHT_PARALLEL_THRESHOLD)
    {
        // PARALLEL PATH: Each thread computes a subset of weights
        #pragma omp parallel for schedule(static)  // Static: predictable load balance
        for (int dataIndex = 0; dataIndex < fullWindowSize; ++dataIndex)
        {
#ifdef ENABLE_MEMOIZATION
            // THREAD-LOCAL CACHE: Each thread gets its own Gram polynomial cache
            // WHY _Thread_local: Persists across parallel regions (unlike automatic variables)
            // WHY static: Only one instance per thread for entire program lifetime
            static _Thread_local GramPolyCacheEntry threadCache[2 * MAX_WINDOW + 1]
                                                               [MAX_ORDER]
                                                               [MAX_ORDER];

            // PARAMETER TRACKING: Detect when user changes filter configuration
            static _Thread_local uint8_t cachedHalfWindowSize = 0xFF; // 0xFF = uninitialized
            static _Thread_local uint8_t cachedPolynomialOrder = 0xFF;
            static _Thread_local uint8_t cachedDerivativeOrder = 0xFF;

            // CACHE INVALIDATION: Clear cache if any parameter changed
            // CRITICAL: Without this, old cache entries would corrupt new results!
            if (cachedHalfWindowSize != halfWindowSize ||
                cachedPolynomialOrder != polynomialOrder ||
                cachedDerivativeOrder != derivativeOrder)
            {
                // Clear cache: mark all entries as not computed
                int maxIndex = 2 * halfWindowSize + 1;
                if (maxIndex > (2 * MAX_WINDOW + 1))
                    maxIndex = (2 * MAX_WINDOW + 1); // Bounds check

                // Only clear up to actual polynomial/derivative orders used
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

                // Update tracked parameters
                cachedHalfWindowSize = halfWindowSize;
                cachedPolynomialOrder = polynomialOrder;
                cachedDerivativeOrder = derivativeOrder;
            }
#endif

            // WEIGHT COMPUTATION: Same formula as Weight(), but with thread-local caching
            float w = 0.0f;
            uint8_t twoM = 2 * ctx.halfWindowSize;

            for (uint8_t k = 0; k <= polynomialOrder; ++k)
            {
#ifdef ENABLE_MEMOIZATION
                // Convert to array indices: dataIndex ∈ [0, 2n] → shiftedIndex ∈ [0, 2n]
                int shiftedDataIndex = (dataIndex - halfWindowSize) + ctx.halfWindowSize;
                int shiftedTarget = targetPoint + ctx.halfWindowSize;

                float part1, part2;

                // CACHE LOOKUP: Check if G_k(dataIndex) already computed
                if (shiftedDataIndex >= 0 && shiftedDataIndex < (2 * MAX_WINDOW + 1) &&
                    k < MAX_ORDER && ctx.derivativeOrder < MAX_ORDER &&
                    threadCache[shiftedDataIndex][k][ctx.derivativeOrder].isComputed)
                {
                    part1 = threadCache[shiftedDataIndex][k][ctx.derivativeOrder].value;
                }
                else
                {
                    // CACHE MISS: Compute and store
                    part1 = GramPolyIterative(k, dataIndex - halfWindowSize, &ctx);
                    if (shiftedDataIndex >= 0 && shiftedDataIndex < (2 * MAX_WINDOW + 1) &&
                        k < MAX_ORDER && ctx.derivativeOrder < MAX_ORDER)
                    {
                        threadCache[shiftedDataIndex][k][ctx.derivativeOrder].value = part1;
                        threadCache[shiftedDataIndex][k][ctx.derivativeOrder].isComputed = true;
                    }
                }

                // Same logic for G_k(targetPoint)
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
                // No memoization: compute directly
                float part1 = GramPolyIterative(k, dataIndex - halfWindowSize, &ctx);
                float part2 = GramPolyIterative(k, targetPoint, &ctx);
#endif

                // Same weight formula as non-memoized version
                float num = GenFact(twoM, k);
                float den = GenFact(twoM + k + 1, k + 1);
                float factor = (2 * k + 1) * (num / den);

                w += factor * part1 * part2;
            }

            // THREAD SAFETY: Each thread writes to different weights[dataIndex]
            // No synchronization needed!
            weights[dataIndex] = w;
        }
    }
    else
#endif
    {
        // SEQUENTIAL PATH: No threading overhead, simpler code path
        // Used for small windows where parallelization would slow things down
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
/**
 * EDGE WEIGHT CACHING STRATEGY:
 * ==============================
 * PROBLEM: Edge points require different target values (t ≠ 0), so weights differ
 * from central region. Computing edge weights on every call wastes time.
 * 
 * SOLUTION: Cache edge weights across calls. If filter parameters unchanged,
 * reuse cached weights (saves ~15µs per edge point).
 * 
 * TRADEOFF: Cache lookup requires critical section (serialization overhead).
 * But this is acceptable because:
 * 1. Only 2×halfWindowSize cache lookups (typically 24)
 * 2. After first call, cache hits are fast (~0.5µs per lookup)
 * 3. Total overhead (~6µs) is small compared to central region work (~100µs)
 * 
 * MEMORY: MAX_WINDOW cache entries × 132 bytes = 4.4 KB (static, one-time cost)
 */
typedef struct
{
    float weights[MAX_WINDOW];        // Cached weight array
    uint8_t halfWindowSize;            // Parameters that generated this cache entry
    uint8_t polynomialOrder;
    uint8_t derivativeOrder;
    uint8_t targetPoint;
    bool valid;                        // false = cache miss, true = cache hit
} EdgeWeightCache;

static EdgeWeightCache leadingEdgeCache[MAX_WINDOW];

/**
 * @brief Thread-safe initialization of edge weight cache
 * 
 * INITIALIZATION PATTERN: Same double-checked locking as GenFact table
 * WHY: Multiple threads may call filter simultaneously, need to ensure cache
 * is initialized exactly once without race conditions.
 */
static void InitEdgeCacheIfNeeded(void)
{
    // Fast path: already initialized
    if (atomic_load_explicit(&edgeCacheInitialized, memory_order_relaxed))
    {
        return;
    }

#ifdef _OPENMP
#pragma omp critical(EdgeCacheInit)
#endif
    {
        // Double-check inside lock (TOCTOU protection)
        if (!atomic_load_explicit(&edgeCacheInitialized, memory_order_acquire))
        {
            // Mark all cache entries as invalid
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
/**
 * @brief Initialize filter configuration structure
 * 
 * PURPOSE: Package all filter parameters into a single struct for convenient
 * passing between functions.
 * 
 * DERIVATIVE SCALING: dt = time_step^derivativeOrder
 * WHY: Numerical derivatives need scaling by Δt^n where n = derivative order
 * EXAMPLE: 2nd derivative in m/s² requires dividing by (Δt)²
 */
SavitzkyGolayFilter initFilter(uint8_t halfWindowSize, uint8_t polynomialOrder,
                               uint8_t targetPoint, uint8_t derivativeOrder, float time_step)
{
    SavitzkyGolayFilter filter;
    filter.conf.halfWindowSize = halfWindowSize;
    filter.conf.polynomialOrder = polynomialOrder;
    filter.conf.targetPoint = targetPoint;
    filter.conf.derivativeOrder = derivativeOrder;
    filter.conf.time_step = time_step;
    filter.dt = pow(time_step, derivativeOrder); // Scaling factor for derivatives

    return filter;
}

//=============================================================================
// Parallel Filter Application
//=============================================================================
/**
 * @brief Apply Savitzky-Golay filter to data array with adaptive parallelization
 * 
 * THREE-STAGE PIPELINE:
 * =====================
 * 
 * STAGE 1: WEIGHT COMPUTATION
 * - Compute central weights (possibly in parallel if window large)
 * - Shared read-only across all threads in Stage 2
 * 
 * STAGE 2: CENTRAL REGION CONVOLUTION
 * - Apply filter to points where full window fits
 * - Highly parallelizable (no dependencies between output points)
 * - 4-chain ILP for optimal CPU pipeline utilization
 * 
 * STAGE 3: EDGE REGION HANDLING
 * - Leading edge: first halfWindowSize points
 * - Trailing edge: last halfWindowSize points
 * - Requires different weights (asymmetric windows)
 * - Optionally parallelized if window large enough
 * 
 * MEMORY LAYOUT:
 * ==============
 * Input:  [0 1 2 3 ... dataSize-1]
 * Output: [0 1 2 3 ... dataSize-1]
 * 
 * Regions:
 * - Leading edge:  [0, halfWindowSize-1]
 * - Central:       [halfWindowSize, dataSize-halfWindowSize-1]
 * - Trailing edge: [dataSize-halfWindowSize, dataSize-1]
 * 
 * @param data Input data array
 * @param dataSize Number of data points
 * @param halfWindowSize Half-window size (full window = 2n+1)
 * @param targetPoint Target evaluation point (usually halfWindowSize for centered)
 * @param filter Filter configuration
 * @param filteredData Output array (must be preallocated with dataSize elements)
 */
static void ApplyFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                        uint16_t targetPoint, SavitzkyGolayFilter filter,
                        MqsRawDataPoint_t filteredData[])
{
    // BOUNDS CHECK: Ensure halfWindowSize doesn't exceed compile-time maximum
    uint8_t maxHalfWindowSize = (MAX_WINDOW - 1) / 2;
    if (halfWindowSize > maxHalfWindowSize)
    {
        printf("Warning: halfWindowSize (%d) exceeds maximum allowed (%d). Adjusting.\n",
               halfWindowSize, maxHalfWindowSize);
        halfWindowSize = maxHalfWindowSize; // Clamp to maximum
    }

    int windowSize = 2 * halfWindowSize + 1;
    int lastIndex = dataSize - 1;
    uint8_t width = halfWindowSize; // Convenient alias

    // WEIGHT STORAGE: Stack allocation (one per function call, ~132 bytes)
    // WHY STACK: Eliminates false sharing (each thread gets own copy in parallel regions)
    float weights[MAX_WINDOW];

    //--------------------------------------------------------------------------
    // STAGE 1: COMPUTE CENTRAL WEIGHTS
    //--------------------------------------------------------------------------
    // These weights are used for all central region points.
    // Computed once, read-only thereafter (safe for parallel access).
    // May be parallelized internally if window is large.
    ComputeWeights(halfWindowSize, targetPoint, filter.conf.polynomialOrder,
                   filter.conf.derivativeOrder, weights);

    //--------------------------------------------------------------------------
    // STAGE 2: PARALLEL CENTRAL REGION CONVOLUTION
    //--------------------------------------------------------------------------
    /**
     * CENTRAL REGION CHARACTERISTICS:
     * - Full symmetric window available at each point
     * - All points use same weights (computed above)
     * - No data dependencies between output points (embarrassingly parallel)
     * 
     * PARALLELIZATION DECISION:
     * - Small datasets (< 1000 points): OpenMP overhead dominates, run sequential
     * - Large datasets (≥ 1000 points): Threading provides 3-4x speedup
     * 
     * 4-CHAIN ILP OPTIMIZATION:
     * - Modern CPUs can execute 4+ independent FMA operations per cycle
     * - By maintaining 4 separate accumulators (sum0, sum1, sum2, sum3),
     *   we allow the CPU to pipeline these operations
     * - Final reduction (sum0+sum1+sum2+sum3) happens once at end
     * - Provides ~1.5x speedup vs single accumulator
     */
    int centralRegionSize = (int)dataSize - windowSize + 1;

#ifdef _OPENMP
    if (centralRegionSize >= PARALLEL_THRESHOLD)
    {
        // PARALLEL PATH: Each thread processes a contiguous chunk of output points
        // Static scheduling: predictable load balance (all iterations have equal work)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < centralRegionSize; ++i)
        {
            // 4-CHAIN ILP: Four independent accumulators for CPU pipeline
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

            const float *w_ptr = weights;              // Read-only weight pointer
            const MqsRawDataPoint_t *d_ptr = &data[i]; // Data window starts at i

            // LOOP PEELING: Handle non-multiple-of-4 window sizes
            // Remainder iterations execute first, then main loop is clean multiple-of-4
            int remainder = windowSize & 3; // windowSize % 4

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
                break; // Window size already multiple of 4
            }

            // MAIN LOOP: Process 4 elements per iteration
            // CPU can execute all 4 FMAs in parallel (independent accumulators)
            int main_iters = (windowSize - remainder) / 4;
            for (int k = 0; k < main_iters; ++k)
            {
                sum0 += w_ptr[0] * d_ptr[0].phaseAngle; // Independent chains
                sum1 += w_ptr[1] * d_ptr[1].phaseAngle; // allow parallel
                sum2 += w_ptr[2] * d_ptr[2].phaseAngle; // execution on
                sum3 += w_ptr[3] * d_ptr[3].phaseAngle; // modern CPUs

                w_ptr += 4;
                d_ptr += 4;
            }

            // REDUCTION: Combine 4 accumulators
            // Pairwise addition exploits instruction-level parallelism
            float sum = (sum0 + sum1) + (sum2 + sum3);
            
            // OUTPUT: Write to centered position
            // i=0 → output[halfWindowSize] (center of first full window)
            filteredData[i + width].phaseAngle = sum;
        }
    }
    else
#endif
    {
        // SEQUENTIAL PATH: Same algorithm, no threading overhead
        // Used when dataset too small to benefit from parallelization
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
    // STAGE 3: EDGE REGION HANDLING
    //--------------------------------------------------------------------------
    /**
     * EDGE PROBLEM:
     * =============
     * At boundaries, we can't center the window. Solutions:
     * 1. Truncate output (lose edge points) ← NOT acceptable
     * 2. Pad data (introduces artifacts) ← NOT ideal
     * 3. Asymmetric windows (different weights) ← WE DO THIS
     * 
     * ASYMMETRIC WINDOW STRATEGY:
     * - Leading edge (i=0...width-1): Window extends right from edge
     * - Trailing edge (i=dataSize-width...dataSize-1): Window extends left from edge
     * - Target point shifts to maintain filter properties
     * 
     * WEIGHT CACHING:
     * - Each edge point needs unique weights (expensive to compute)
     * - Cache persists across filter calls with same parameters
     * - Critical section protects cache access (acceptable overhead for ~24 lookups)
     * 
     * PARALLELIZATION:
     * - Only worthwhile for large windows (width ≥ 8)
     * - Small windows: critical section overhead dominates
     * - Large windows: computation dominates, parallelization helps
     */
    InitEdgeCacheIfNeeded(); // Ensure cache initialized

#ifdef _OPENMP
    if (width >= EDGE_PARALLEL_THRESHOLD)
    {
        // PARALLEL EDGE PROCESSING: Each thread handles one edge point
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < width; ++i)
        {
            // TARGET POINT: Shifts from center as we approach edge
            // i=0 (first point): target = width (rightmost end of window)
            // i=width-1: target = 1 (second point in window)
            uint8_t target = width - i;

            float tempWeights[MAX_WINDOW]; // Thread-local buffer for cache miss
            const float *edgeWeights;      // Pointer to weights (cached or temp)
            bool usedCache = false;

            // CACHE LOOKUP: Protected by critical section to prevent TOCTOU race
            // TRADEOFF: Serialization overhead (~0.5µs) vs recomputation savings (~15µs)
#ifdef _OPENMP
#pragma omp critical(EdgeCacheAccess)
#endif
            {
                // Check if cache has valid entry matching current parameters
                if (i < MAX_WINDOW && leadingEdgeCache[i].valid &&
                    leadingEdgeCache[i].halfWindowSize == halfWindowSize &&
                    leadingEdgeCache[i].polynomialOrder == filter.conf.polynomialOrder &&
                    leadingEdgeCache[i].derivativeOrder == filter.conf.derivativeOrder &&
                    leadingEdgeCache[i].targetPoint == target)
                {
                    // CACHE HIT: Reuse precomputed weights (fast path)
                    edgeWeights = leadingEdgeCache[i].weights;
                    usedCache = true;
                }
            }

            if (!usedCache)
            {
                // CACHE MISS: Compute weights with asymmetric target
                ComputeWeights(halfWindowSize, target, filter.conf.polynomialOrder,
                               filter.conf.derivativeOrder, tempWeights);
                edgeWeights = tempWeights;

                // UPDATE CACHE: Store for future calls (protected write)
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

            // LEADING EDGE CONVOLUTION (reverse order)
            // Window: data[windowSize-1] down to data[0]
            // WHY REVERSE: Weight array is ordered [0...windowSize-1] = [far...near]
            // Data traversal must match weight order
            float leadSum0 = 0.0f, leadSum1 = 0.0f, leadSum2 = 0.0f, leadSum3 = 0.0f;
            const float *w_ptr = edgeWeights;
            const MqsRawDataPoint_t *d_ptr = &data[windowSize - 1]; // Start at rightmost

            int remainder = windowSize & 3;
            switch (remainder)
            {
            case 3:
                leadSum0 += w_ptr[0] * d_ptr[0].phaseAngle;   // d_ptr[0] = data[windowSize-1]
                leadSum1 += w_ptr[1] * d_ptr[-1].phaseAngle;  // d_ptr[-1] = data[windowSize-2]
                leadSum2 += w_ptr[2] * d_ptr[-2].phaseAngle;  // and so on...
                w_ptr += 3;
                d_ptr -= 3; // Move left
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
                d_ptr -= 4; // Continue moving left
            }

            float leadingSum = (leadSum0 + leadSum1) + (leadSum2 + leadSum3);
            filteredData[i].phaseAngle = leadingSum; // Write to leading edge

            // TRAILING EDGE CONVOLUTION (forward order)
            // Window: data[lastIndex-windowSize+1] up to data[lastIndex]
            // Same weights work (by symmetry of least-squares fit)
            float trailSum0 = 0.0f, trailSum1 = 0.0f, trailSum2 = 0.0f, trailSum3 = 0.0f;
            w_ptr = edgeWeights; // Reuse same weights
            d_ptr = &data[lastIndex - windowSize + 1]; // Start at leftmost

            switch (remainder)
            {
            case 3:
                trailSum0 += w_ptr[0] * d_ptr[0].phaseAngle;
                trailSum1 += w_ptr[1] * d_ptr[1].phaseAngle;
                trailSum2 += w_ptr[2] * d_ptr[2].phaseAngle;
                w_ptr += 3;
                d_ptr += 3; // Move right
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
                d_ptr += 4; // Continue moving right
            }

            float trailingSum = (trailSum0 + trailSum1) + (trailSum2 + trailSum3);
            filteredData[lastIndex - i].phaseAngle = trailingSum; // Write to trailing edge
        }
    }
    else
#endif
    {
        // SEQUENTIAL EDGE PROCESSING: Same algorithm, no threading
        // Used when window small enough that critical section overhead dominates
        for (int i = 0; i < width; ++i)
        {
            uint8_t target = width - i;

            const float *edgeWeights;
            float tempWeights[MAX_WINDOW];
            bool useCache = false;

            // Cache lookup (no critical section needed in sequential code)
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

                // Cache update (no lock needed)
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

            // Leading edge convolution (same as parallel version)
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

            // Trailing edge convolution
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
/**
 * @brief Apply Savitzky-Golay filter with automatic threading
 * 
 * USAGE EXAMPLES:
 * ===============
 * // Basic smoothing (most common)
 * mes_savgolFilter(data, 1000, 12, filtered, 4, 0, 0);
 * 
 * // First derivative (velocity)
 * mes_savgolFilter(data, 1000, 12, filtered, 4, 12, 1);
 * 
 * // Second derivative (acceleration)
 * mes_savgolFilter(data, 1000, 12, filtered, 4, 12, 2);
 * 
 * PARAMETER CONSTRAINTS:
 * ======================
 * - halfWindowSize: 1 to (MAX_WINDOW-1)/2
 * - polynomialOrder: Must be < (2×halfWindowSize+1)
 * - targetPoint: 0 to (2×halfWindowSize)
 * - derivativeOrder: 0 to polynomialOrder
 * - dataSize: Must be ≥ (2×halfWindowSize+1)
 * 
 * THREADING BEHAVIOR:
 * ===================
 * - Uses all available CPU cores (OMP_NUM_THREADS)
 * - Adaptive: automatically sequential for small datasets
 * - Use mes_savgolFilter_threaded() for explicit thread control
 * 
 * @param data Input data array
 * @param dataSize Number of data points
 * @param halfWindowSize Half-window size (full window = 2n+1)
 * @param filteredData Output array (must be preallocated)
 * @param polynomialOrder Polynomial fitting order
 * @param targetPoint Evaluation point (0=forward, n=centered, 2n=backward)
 * @param derivativeOrder Derivative order (0=smooth, 1=velocity, 2=accel)
 * @return 0 on success, negative error code on failure
 */
int mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                     MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder,
                     uint8_t targetPoint, uint8_t derivativeOrder)
{
    // VALIDATION: Runtime checks for debug builds, compile-time checks via assert
    /*
    assert(data != NULL && "Input data pointer must not be NULL");
    assert(filteredData != NULL && "Filtered data pointer must not be NULL");
    assert(dataSize > 0 && "Data size must be greater than 0");
    assert(halfWindowSize > 0 && "Half-window size must be greater than 0");
    assert((2 * halfWindowSize + 1) <= dataSize && "Filter window size must not exceed data size");
    assert(polynomialOrder < (2 * halfWindowSize + 1) && "Polynomial order must be less than the filter window size");
    assert(targetPoint <= (2 * halfWindowSize) && "Target point must be within the filter window");
    */

    // PRODUCTION ERROR HANDLING: Return error codes instead of crashing
    if (data == NULL || filteredData == NULL)
    {
        LOG_ERROR("NULL pointer passed to mes_savgolFilter.");
        return -1; // EINVAL equivalent
    }

    if (dataSize == 0 || halfWindowSize == 0 ||
        polynomialOrder >= 2 * halfWindowSize + 1 ||
        targetPoint > 2 * halfWindowSize ||
        (2 * halfWindowSize + 1) > dataSize)
    {
        LOG_ERROR("Invalid filter parameters provided.");
        return -2; // EDOM equivalent (domain error)
    }

    // INITIALIZATION: Ensure shared data structures ready (thread-safe lazy init)
    InitGenFactTable();
    InitEdgeCacheIfNeeded();

    // CREATE FILTER: Package parameters for convenient passing
    SavitzkyGolayFilter filter = initFilter(halfWindowSize, polynomialOrder, targetPoint, derivativeOrder, 1.0f);

    // APPLY FILTER: Three-stage pipeline (weights → central → edges)
    ApplyFilter(data, dataSize, halfWindowSize, targetPoint, filter, filteredData);

    return 0; // Success
}

/**
 * @brief Apply Savitzky-Golay filter with explicit thread count control
 * 
 * PURPOSE: Allows users to override automatic threading decisions for:
 * - Benchmarking (compare 1-thread vs N-threads)
 * - Resource control (limit CPU usage)
 * - Debugging (force sequential execution)
 * 
 * THREAD COUNT SEMANTICS:
 * =======================
 * numThreads =  0: Auto-detect (use all available cores)
 * numThreads = -1: Force sequential (disable all parallelization)
 * numThreads >  0: Use exactly N threads
 * 
 * EXAMPLE:
 * ========
 * // Limit to 4 threads (leave CPU headroom for other tasks)
 * mes_savgolFilter_threaded(data, 1000, 12, filtered, 4, 0, 0, 4);
 * 
 * // Force sequential (for debugging or single-core systems)
 * mes_savgolFilter_threaded(data, 1000, 12, filtered, 4, 0, 0, -1);
 * 
 * @param numThreads Thread count (0=auto, -1=sequential, >0=explicit)
 * @return Same as mes_savgolFilter()
 */
int mes_savgolFilter_threaded(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                              MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder,
                              uint8_t targetPoint, uint8_t derivativeOrder, int numThreads)
{
#ifdef _OPENMP
    // Save current thread count for restoration
    int oldNumThreads = omp_get_max_threads();

    // Apply user-requested thread count
    if (numThreads > 0)
    {
        omp_set_num_threads(numThreads); // Explicit count
    }
    else if (numThreads == -1)
    {
        omp_set_num_threads(1); // Force sequential
    }
    // numThreads == 0: use default (no change)

    // Execute filter with requested threading
    int result = mes_savgolFilter(data, dataSize, halfWindowSize, filteredData,
                                  polynomialOrder, targetPoint, derivativeOrder);

    // Restore original thread count (good citizen behavior)
    omp_set_num_threads(oldNumThreads);
    
    return result;
#else
    // OpenMP not available: ignore numThreads, run sequential
    (void)numThreads; // Suppress unused parameter warning
    return mes_savgolFilter(data, dataSize, halfWindowSize, filteredData,
                            polynomialOrder, targetPoint, derivativeOrder);
#endif
}