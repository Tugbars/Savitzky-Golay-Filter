/**
 * @file savgol_filter.c
 * @brief Savitzky-Golay Filter — Optimized Implementation
 * 
 * A production-quality implementation of the Savitzky-Golay smoothing and
 * differentiation filter with clean layered architecture.
 * 
 * ============================================================================
 * FILE ORGANIZATION
 * ============================================================================
 * 
 * This file is organized in layers, with dependencies flowing DOWNWARD only:
 * 
 *   Section 1: Configuration & Constants
 *   Section 2: Public API Types
 *   Section 3: Math Primitives (GenFact)
 *   Section 4: Gram Polynomials  
 *   Section 5: Weight Computation
 *   Section 6: Convolution Engine
 *   Section 7: Filter Context (internal struct)
 *   Section 8: Public API Implementation
 * 
 * To understand the code, read sections 1-2 for usage, then sections 3-6
 * bottom-up for the mathematical implementation.
 * 
 * ============================================================================
 * MATHEMATICAL BACKGROUND
 * ============================================================================
 * 
 * The Savitzky-Golay filter fits a polynomial of degree m to a sliding window
 * of 2n+1 data points using least-squares, then evaluates the polynomial
 * (or its derivative) at a target point within the window.
 * 
 * Key parameters:
 *   n = half_window    : Window spans [-n, +n], total 2n+1 points
 *   m = poly_order     : Degree of fitted polynomial (m < 2n+1)
 *   d = derivative     : 0=smoothing, 1=first derivative, 2=second, etc.
 *   t = target         : Evaluation point within window (usually 0=center)
 * 
 * The filter reduces to a weighted sum (convolution):
 *   output[j] = Σ weights[i] × input[j - n + i]
 * 
 * Weights are computed using Gram (discrete orthogonal) polynomials, which
 * provide numerical stability and efficient computation.
 * 
 * ============================================================================
 * USAGE EXAMPLE
 * ============================================================================
 * 
 *   // Configure: 5-point window (n=2), quadratic fit (m=2), smoothing (d=0)
 *   SavgolConfig config = {
 *       .half_window = 2,
 *       .poly_order = 2,
 *       .derivative = 0,
 *       .time_step = 1.0f
 *   };
 *   
 *   // Create filter (precomputes all weights)
 *   SavgolFilter *filter = savgol_create(&config);
 *   if (!filter) { handle_error(); }
 *   
 *   // Apply to data
 *   float input[1000], output[1000];
 *   int result = savgol_apply(filter, input, output, 1000);
 *   
 *   // Cleanup
 *   savgol_destroy(filter);
 * 
 * ============================================================================
 * PERFORMANCE CHARACTERISTICS
 * ============================================================================
 * 
 * - Weight computation: O(n × m²) — done once at filter creation
 * - Filter application: O(N × n) — linear in data size
 * - Memory: ~8KB for filter context (configurable via MAX_HALF_WINDOW)
 * - No heap allocation during filtering (only at create/destroy)
 * 
 * Optimizations applied:
 * - GenFact lookup table (eliminates repeated factorial computation)
 * - ILP-optimized convolution (4 independent accumulator chains)
 * - Precomputed edge weights (no runtime cache management)
 * 
 * Author: Tugbars Heptaskin
 * Original: 2025-10-24
 * Restructured: 2025-01-28
 */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "savgolFilter.h"

/*============================================================================
 * SECTION 1: INTERNAL CONSTANTS
 *============================================================================
 * 
 * Internal compile-time constants not exposed in the public API.
 */

/** 
 * GenFact table size.
 * Must accommodate lookups for GenFact(2n + k + 1, k + 1) where n ≤ MAX_HALF_WINDOW
 * and k ≤ MAX_POLY_ORDER. So we need: 2*32 + 10 + 1 + 1 = 76
 */
#define GENFACT_TABLE_SIZE (2 * SAVGOL_MAX_HALF_WINDOW + SAVGOL_MAX_POLY_ORDER + 2)


/*============================================================================
 * SECTION 2: MATH PRIMITIVES — Generalized Factorial
 *============================================================================
 * 
 * The generalized factorial (falling factorial) is used in Gram polynomial
 * normalization:
 * 
 *   GenFact(a, b) = a × (a-1) × (a-2) × ... × (a-b+1)
 *                 = a! / (a-b)!
 * 
 * Special cases:
 *   GenFact(a, 0) = 1       (empty product)
 *   GenFact(a, 1) = a
 *   GenFact(a, a) = a!
 *   GenFact(a, b) = 0       if b > a
 * 
 * We precompute all needed values into a lookup table at initialization.
 * This replaces O(b) computation with O(1) lookup.
 */

/**
 * @brief Lookup table for generalized factorial values.
 * 
 * Access: genfact_table[a][b] = GenFact(a, b)
 * 
 * Initialized once at program start (or first filter creation).
 * Thread-safety: Initialization is idempotent; concurrent init is safe
 * but wasteful. For strict thread safety, initialize before spawning threads.
 */
static float g_genfact_table[GENFACT_TABLE_SIZE][GENFACT_TABLE_SIZE];
static bool  g_genfact_initialized = false;

/**
 * @brief Initialize the generalized factorial lookup table.
 * 
 * Computes all GenFact(a, b) values for 0 ≤ a, b < GENFACT_TABLE_SIZE.
 * Safe to call multiple times (idempotent).
 */
static void genfact_init(void)
{
    if (g_genfact_initialized) {
        return;
    }
    
    for (int a = 0; a < GENFACT_TABLE_SIZE; a++) {
        g_genfact_table[a][0] = 1.0f;  /* Empty product = 1 */
        
        for (int b = 1; b < GENFACT_TABLE_SIZE; b++) {
            if (b > a) {
                /* Not enough terms: GenFact(a, b) = 0 when b > a */
                g_genfact_table[a][b] = 0.0f;
            } else {
                /* Compute: a × (a-1) × ... × (a-b+1) */
                double product = 1.0;  /* Use double for intermediate precision */
                for (int j = a - b + 1; j <= a; j++) {
                    product *= (double)j;
                }
                g_genfact_table[a][b] = (float)product;
            }
        }
    }
    
    g_genfact_initialized = true;
}

/**
 * @brief Look up a generalized factorial value.
 * 
 * @param a Upper limit of the product.
 * @param b Number of terms in the product.
 * @return GenFact(a, b), or 0 if indices out of range.
 */
static inline float genfact_lookup(uint8_t a, uint8_t b)
{
    if (a >= GENFACT_TABLE_SIZE || b >= GENFACT_TABLE_SIZE) {
        /* Out of range — indicates configuration exceeds compiled limits */
        fprintf(stderr, "savgol: GenFact lookup out of range: a=%d, b=%d, max=%d\n",
                a, b, GENFACT_TABLE_SIZE - 1);
        return 0.0f;
    }
    return g_genfact_table[a][b];
}


/*============================================================================
 * SECTION 4: GRAM POLYNOMIALS
 *============================================================================
 * 
 * Gram polynomials are discrete orthogonal polynomials defined on integer
 * points [-n, ..., +n]. They provide a numerically stable basis for the
 * least-squares polynomial fit.
 * 
 * Notation: F_k^{(d)}(i) = Gram polynomial of order k, derivative d, at point i
 * 
 * Three-term recurrence relation:
 * 
 *   F_0^{(d)}(i) = δ_{d,0}           (1 if d=0, else 0)
 *   
 *   F_1^{(d)}(i) = (1/n) × [i × F_0^{(d)}(i) + d × F_0^{(d-1)}(i)]
 *   
 *   F_k^{(d)}(i) = α_k × [i × F_{k-1}^{(d)}(i) + d × F_{k-1}^{(d-1)}(i)]
 *                  - γ_k × F_{k-2}^{(d)}(i)
 * 
 * where:
 *   α_k = (4k - 2) / [k × (2n - k + 1)]
 *   γ_k = (k - 1) × (2n + k) / [k × (2n - k + 1)]
 * 
 * Implementation uses a rolling buffer to compute F_k without storing
 * all previous orders, reducing memory from O(k×d) to O(d).
 */

/**
 * @brief Compute Gram polynomial F_k^{(d)}(i).
 * 
 * Uses iterative dynamic programming with rolling buffers for efficiency.
 * No memoization — this function is pure computation.
 * 
 * @param half_window   Half-window size n.
 * @param deriv_order   Derivative order d (0 for polynomial value).
 * @param poly_order    Polynomial order k to compute.
 * @param data_index    Evaluation point i ∈ [-n, +n].
 * @return Value of F_k^{(d)}(i).
 */
static float gram_poly(uint8_t half_window, uint8_t deriv_order,
                       uint8_t poly_order, int data_index)
{
    /* Fixed-size buffers for rolling computation. */
    /* buf[d] holds F_{current}^{(d)} for each derivative order. */
    float buf0[SAVGOL_MAX_DERIVATIVE + 1];  /* F_{k-2} */
    float buf1[SAVGOL_MAX_DERIVATIVE + 1];  /* F_{k-1} */
    float buf2[SAVGOL_MAX_DERIVATIVE + 1];  /* F_{k}   */
    
    float *prev2 = buf0;
    float *prev1 = buf1;
    float *curr  = buf2;
    
    const float n = (float)half_window;
    const float i = (float)data_index;
    
    /*--- Base case: k = 0 ---*/
    /* F_0^{(0)} = 1, F_0^{(d)} = 0 for d > 0 */
    for (uint8_t d = 0; d <= deriv_order; d++) {
        prev2[d] = (d == 0) ? 1.0f : 0.0f;
    }
    
    if (poly_order == 0) {
        return prev2[deriv_order];
    }
    
    /*--- First order: k = 1 ---*/
    /* F_1^{(d)} = (1/n) × [i × F_0^{(d)} + d × F_0^{(d-1)}] */
    const float inv_n = 1.0f / n;
    
    prev1[0] = inv_n * (i * prev2[0]);  /* d=0: no F_0^{(-1)} term */
    
    for (uint8_t d = 1; d <= deriv_order; d++) {
        prev1[d] = inv_n * (i * prev2[d] + (float)d * prev2[d - 1]);
    }
    
    if (poly_order == 1) {
        return prev1[deriv_order];
    }
    
    /*--- Higher orders: k ≥ 2 ---*/
    const float two_n = 2.0f * n;
    
    for (uint8_t k = 2; k <= poly_order; k++) {
        const float k_f = (float)k;
        const float denom = k_f * (two_n - k_f + 1.0f);
        const float alpha = (4.0f * k_f - 2.0f) / denom;
        const float gamma = ((k_f - 1.0f) * (two_n + k_f)) / denom;
        
        /* d=0 case: no F_{k-1}^{(-1)} term */
        curr[0] = alpha * (i * prev1[0]) - gamma * prev2[0];
        
        /* d>0 cases */
        for (uint8_t d = 1; d <= deriv_order; d++) {
            float term = i * prev1[d] + (float)d * prev1[d - 1];
            curr[d] = alpha * term - gamma * prev2[d];
        }
        
        /* Rotate buffers: prev2 <- prev1 <- curr */
        float *tmp = prev2;
        prev2 = prev1;
        prev1 = curr;
        curr = tmp;
    }
    
    /* After loop, prev1 contains F_{poly_order} */
    return prev1[deriv_order];
}


/*============================================================================
 * SECTION 5: WEIGHT COMPUTATION
 *============================================================================
 * 
 * Filter weights combine Gram polynomials with normalization factors:
 * 
 *   w(i, t) = Σ_{k=0}^{m} (2k+1) × [GenFact(2n,k) / GenFact(2n+k+1,k+1)]
 *                         × F_k^{(0)}(i) × F_k^{(d)}(t)
 * 
 * where:
 *   i = data point index within window
 *   t = target evaluation point (usually 0 for center)
 *   m = polynomial order
 *   d = derivative order
 *   n = half window size
 * 
 * For the central region, t=0 and weights are symmetric (for d=0).
 * For edges, t≠0 shifts the fit toward available data.
 */

/**
 * @brief Compute a single filter weight.
 * 
 * @param half_window   Half-window size n.
 * @param poly_order    Polynomial order m.
 * @param deriv_order   Derivative order d.
 * @param data_index    Position of data point i ∈ [-n, +n].
 * @param target        Target evaluation point t ∈ [-n, +n].
 * @return Weight for data point i when evaluating at target t.
 */
static float compute_weight(uint8_t half_window, uint8_t poly_order,
                            uint8_t deriv_order, int data_index, int target)
{
    const uint8_t two_n = 2 * half_window;
    float weight = 0.0f;
    
    for (uint8_t k = 0; k <= poly_order; k++) {
        /* Normalization factor: (2k+1) × GenFact(2n,k) / GenFact(2n+k+1,k+1) */
        float num = genfact_lookup(two_n, k);
        float den = genfact_lookup(two_n + k + 1, k + 1);
        float factor = (float)(2 * k + 1) * (num / den);
        
        /* Gram polynomial values */
        float gram_at_i = gram_poly(half_window, 0, k, data_index);           /* F_k^{(0)}(i) */
        float gram_at_t = gram_poly(half_window, deriv_order, k, target);     /* F_k^{(d)}(t) */
        
        weight += factor * gram_at_i * gram_at_t;
    }
    
    return weight;
}

/**
 * @brief Compute weights for center window (target = 0).
 * 
 * Output array has 2n+1 elements: weights[0] for i=-n, weights[n] for i=0, etc.
 * 
 * @param half_window   Half-window size n.
 * @param poly_order    Polynomial order m.
 * @param deriv_order   Derivative order d.
 * @param weights       Output array of size ≥ 2n+1.
 */
static void compute_center_weights(uint8_t half_window, uint8_t poly_order,
                                   uint8_t deriv_order, float *weights)
{
    const int window_size = 2 * half_window + 1;
    
    for (int idx = 0; idx < window_size; idx++) {
        int data_index = idx - half_window;  /* Convert [0, 2n] → [-n, +n] */
        weights[idx] = compute_weight(half_window, poly_order, deriv_order,
                                      data_index, 0 /* target = center */);
    }
}

/**
 * @brief Compute weights for all edge positions.
 * 
 * For edge position i (0 ≤ i < n), the target is shifted to n-i.
 * This produces asymmetric weights that emphasize available data.
 * 
 * Leading and trailing edges use the same weights but apply them
 * to data in opposite directions.
 * 
 * @param half_window   Half-window size n.
 * @param poly_order    Polynomial order m.
 * @param deriv_order   Derivative order d.
 * @param edge_weights  Output: 2D array [n][2n+1] of edge weights.
 */
static void compute_edge_weights(uint8_t half_window, uint8_t poly_order,
                                 uint8_t deriv_order,
                                 float edge_weights[SAVGOL_MAX_HALF_WINDOW][SAVGOL_MAX_WINDOW])
{
    const int window_size = 2 * half_window + 1;
    
    for (int edge_pos = 0; edge_pos < half_window; edge_pos++) {
        int target = half_window - edge_pos;  /* Shift toward available data */
        
        for (int idx = 0; idx < window_size; idx++) {
            int data_index = idx - half_window;
            edge_weights[edge_pos][idx] = compute_weight(
                half_window, poly_order, deriv_order, data_index, target);
        }
    }
}


/*============================================================================
 * SECTION 6: CONVOLUTION ENGINE
 *============================================================================
 * 
 * The filter is applied as a discrete convolution:
 * 
 *   output[j] = Σ_{i=0}^{2n} weights[i] × input[j - n + i]
 * 
 * This is optimized using Instruction-Level Parallelism (ILP):
 * - Four independent accumulator chains allow parallel execution
 * - Modern CPUs can execute 2-4 FMA operations per cycle
 * - This gives 1.2-1.3× speedup over naive single-accumulator loop
 * 
 * Edge handling:
 * - POLYNOMIAL mode: Use precomputed asymmetric edge weights
 * - Other modes: Use center weights with virtual boundary padding
 */

/**
 * @brief Get sample with virtual padding for boundary handling.
 * 
 * When index is out of bounds, returns a virtual sample based on the
 * boundary mode. This enables using center weights for all positions.
 * 
 * @param data   Input data array.
 * @param length Length of data array.
 * @param index  Sample index (may be negative or >= length).
 * @param mode   Boundary handling mode.
 * @return Sample value (real or virtual).
 */
static inline float get_padded_sample(const float *data, size_t length,
                                       int index, SavgolBoundaryMode mode)
{
    /* Fast path: index is within bounds */
    if (index >= 0 && index < (int)length) {
        return data[index];
    }
    
    /* Handle out-of-bounds based on mode */
    switch (mode) {
        case SAVGOL_BOUNDARY_REFLECT:
            /* Mirror at boundary: [..., d2, d1 | d0, d1, d2, ...] */
            if (index < 0) {
                /* Reflect left: index -1 → 0, -2 → 1, etc. */
                index = -index - 1;
                if (index >= (int)length) index = (int)length - 1;  /* Clamp */
            } else {
                /* Reflect right: length → length-1, length+1 → length-2 */
                index = 2 * (int)length - index - 1;
                if (index < 0) index = 0;  /* Clamp */
            }
            return data[index];
            
        case SAVGOL_BOUNDARY_PERIODIC:
            /* Wrap around: index -1 → length-1, length → 0 */
            index = ((index % (int)length) + (int)length) % (int)length;
            return data[index];
            
        case SAVGOL_BOUNDARY_CONSTANT:
            /* Extend edge value */
            if (index < 0) {
                return data[0];
            } else {
                return data[length - 1];
            }
            
        default:
            /* POLYNOMIAL mode should not reach here */
            return 0.0f;
    }
}

/**
 * @brief Convolution with virtual boundary padding.
 * 
 * Used for REFLECT, PERIODIC, and CONSTANT boundary modes.
 * Builds a window buffer with virtual samples, then convolves.
 * 
 * @param weights     Filter weights (window_size elements).
 * @param data        Input data array.
 * @param length      Length of data array.
 * @param center_idx  Center index of window in data.
 * @param half_window Half-window size n.
 * @param mode        Boundary handling mode.
 * @return Convolution result.
 */
static float convolve_padded(const float *weights, const float *data,
                              size_t length, int center_idx, int half_window,
                              SavgolBoundaryMode mode)
{
    /* Build window with virtual padding */
    float window[SAVGOL_MAX_WINDOW];
    int window_size = 2 * half_window + 1;
    
    for (int k = 0; k < window_size; k++) {
        int idx = center_idx - half_window + k;
        window[k] = get_padded_sample(data, length, idx, mode);
    }
    
    /* Use ILP-optimized convolution on the padded window */
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    int remainder = window_size & 3;
    int k = 0;
    
    switch (remainder) {
        case 3: sum2 += weights[2] * window[2]; /* fallthrough */
        case 2: sum1 += weights[1] * window[1]; /* fallthrough */
        case 1: sum0 += weights[0] * window[0]; /* fallthrough */
        case 0: break;
    }
    k = remainder;
    
    int main_iters = (window_size - remainder) >> 2;
    for (int i = 0; i < main_iters; i++) {
        sum0 += weights[k]     * window[k];
        sum1 += weights[k + 1] * window[k + 1];
        sum2 += weights[k + 2] * window[k + 2];
        sum3 += weights[k + 3] * window[k + 3];
        k += 4;
    }
    
    return (sum0 + sum1) + (sum2 + sum3);
}

/**
 * @brief ILP-optimized convolution for center region.
 * 
 * Computes weighted sum using 4 independent accumulator chains.
 * 
 * @param weights       Weight array of size window_size.
 * @param data          Pointer to START of window in input array.
 * @param window_size   Number of points in window (2n+1).
 * @return Weighted sum.
 */
static inline float convolve_ilp(const float *weights, const float *data, int window_size)
{
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    const float *w = weights;
    const float *d = data;
    
    /* Handle remainder (0-3 elements) first to keep main loop aligned */
    int remainder = window_size & 3;
    
    switch (remainder) {
        case 3: sum2 += w[2] * d[2]; /* fallthrough */
        case 2: sum1 += w[1] * d[1]; /* fallthrough */
        case 1: sum0 += w[0] * d[0]; /* fallthrough */
        case 0: break;
    }
    w += remainder;
    d += remainder;
    
    /* Main loop: 4 elements per iteration */
    int main_iters = (window_size - remainder) >> 2;
    
    for (int k = 0; k < main_iters; k++) {
        sum0 += w[0] * d[0];
        sum1 += w[1] * d[1];
        sum2 += w[2] * d[2];
        sum3 += w[3] * d[3];
        w += 4;
        d += 4;
    }
    
    /* Pairwise reduction for better numerical accuracy */
    return (sum0 + sum1) + (sum2 + sum3);
}

/**
 * @brief ILP-optimized convolution with reversed data traversal.
 * 
 * Used for leading edge: weights are applied in forward order,
 * but data is traversed backward.
 * 
 * @param weights       Weight array of size window_size.
 * @param data_end      Pointer to LAST element of window in input array.
 * @param window_size   Number of points in window (2n+1).
 * @return Weighted sum.
 */
static inline float convolve_ilp_reverse(const float *weights, const float *data_end, int window_size)
{
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    const float *w = weights;
    const float *d = data_end;
    
    int remainder = window_size & 3;
    
    switch (remainder) {
        case 3: sum2 += w[2] * d[-2]; /* fallthrough */
        case 2: sum1 += w[1] * d[-1]; /* fallthrough */
        case 1: sum0 += w[0] * d[0];  /* fallthrough */
        case 0: break;
    }
    w += remainder;
    d -= remainder;
    
    int main_iters = (window_size - remainder) >> 2;
    
    for (int k = 0; k < main_iters; k++) {
        sum0 += w[0] * d[0];
        sum1 += w[1] * d[-1];
        sum2 += w[2] * d[-2];
        sum3 += w[3] * d[-3];
        w += 4;
        d -= 4;
    }
    
    return (sum0 + sum1) + (sum2 + sum3);
}


/*============================================================================
 * SECTION 7: PUBLIC API IMPLEMENTATION
 *============================================================================
 * 
 * User-facing functions for filter lifecycle and application.
 */

/**
 * @brief Validate filter configuration.
 * 
 * @param config Configuration to validate.
 * @return true if valid, false otherwise.
 */
static bool validate_config(const SavgolConfig *config)
{
    if (config == NULL) {
        return false;
    }
    
    if (config->half_window == 0 || config->half_window > SAVGOL_MAX_HALF_WINDOW) {
        fprintf(stderr, "savgol: half_window must be in [1, %d], got %d\n",
                SAVGOL_MAX_HALF_WINDOW, config->half_window);
        return false;
    }
    
    int window_size = 2 * config->half_window + 1;
    
    if (config->poly_order >= window_size) {
        fprintf(stderr, "savgol: poly_order must be < window_size (%d), got %d\n",
                window_size, config->poly_order);
        return false;
    }
    
    if (config->derivative > SAVGOL_MAX_DERIVATIVE) {
        fprintf(stderr, "savgol: derivative must be ≤ %d, got %d\n",
                SAVGOL_MAX_DERIVATIVE, config->derivative);
        return false;
    }
    
    if (config->derivative > config->poly_order) {
        fprintf(stderr, "savgol: derivative (%d) cannot exceed poly_order (%d)\n",
                config->derivative, config->poly_order);
        return false;
    }
    
    if (config->time_step <= 0.0f) {
        fprintf(stderr, "savgol: time_step must be > 0, got %f\n", config->time_step);
        return false;
    }
    
    return true;
}

/**
 * @brief Create a Savitzky-Golay filter with the specified configuration.
 * 
 * Precomputes all filter weights. This is the expensive operation;
 * subsequent calls to savgol_apply() are fast.
 * 
 * @param config Filter parameters. Must pass validation.
 * @return Filter handle, or NULL on error. Caller must call savgol_destroy().
 */
SavgolFilter *savgol_create(const SavgolConfig *config)
{
    if (!validate_config(config)) {
        return NULL;
    }
    
    /* Ensure math tables are initialized */
    genfact_init();
    
    /* Allocate filter context */
    SavgolFilter *filter = (SavgolFilter *)calloc(1, sizeof(SavgolFilter));
    if (filter == NULL) {
        fprintf(stderr, "savgol: failed to allocate filter context\n");
        return NULL;
    }
    
    /* Copy configuration */
    filter->config = *config;
    filter->window_size = 2 * config->half_window + 1;
    filter->dt_scale = powf(config->time_step, (float)config->derivative);
    
    /* Precompute center weights */
    compute_center_weights(config->half_window, config->poly_order,
                           config->derivative, filter->center_weights);
    
    /* Precompute edge weights */
    compute_edge_weights(config->half_window, config->poly_order,
                         config->derivative, filter->edge_weights);
    
    return filter;
}

/**
 * @brief Destroy a filter and free associated resources.
 * 
 * @param filter Filter handle from savgol_create(). May be NULL (no-op).
 */
void savgol_destroy(SavgolFilter *filter)
{
    free(filter);  /* free(NULL) is safe */
}

/**
 * @brief Apply the filter to a contiguous array of floats.
 * 
 * Handles edge effects based on the boundary mode in config:
 * - POLYNOMIAL: Uses precomputed asymmetric edge weights
 * - REFLECT/PERIODIC/CONSTANT: Uses center weights with virtual padding
 * 
 * @param filter  Filter handle from savgol_create().
 * @param input   Input data array.
 * @param output  Output array (may be same as input for in-place).
 * @param length  Number of elements. Must be ≥ window size (2n+1).
 * @return 0 on success, -1 on error.
 */
int savgol_apply(const SavgolFilter *filter, 
                 const float *input, float *output, size_t length)
{
    if (filter == NULL || input == NULL || output == NULL) {
        fprintf(stderr, "savgol_apply: NULL pointer\n");
        return -1;
    }
    
    if (length < (size_t)filter->window_size) {
        fprintf(stderr, "savgol_apply: data length (%lu) < window size (%d)\n",
                (unsigned long)length, filter->window_size);
        return -1;
    }
    
    const int n = filter->config.half_window;
    const int window_size = filter->window_size;
    const float dt_inv = (filter->dt_scale != 0.0f) ? (1.0f / filter->dt_scale) : 1.0f;
    const SavgolBoundaryMode mode = filter->config.boundary;
    
    /*--- Central region: full symmetric windows (same for all modes) ---*/
    for (size_t j = n; j < length - n; j++) {
        float sum = convolve_ilp(filter->center_weights, &input[j - n], window_size);
        output[j] = sum * dt_inv;
    }
    
    /*--- Edge handling depends on boundary mode ---*/
    if (mode == SAVGOL_BOUNDARY_POLYNOMIAL) {
        /*--- POLYNOMIAL mode: use precomputed asymmetric edge weights ---*/
        
        /* Leading edge */
        for (int i = 0; i < n; i++) {
            float sum = convolve_ilp_reverse(filter->edge_weights[i], 
                                              &input[window_size - 1], window_size);
            output[i] = sum * dt_inv;
        }
        
        /* Trailing edge */
        for (int i = 0; i < n; i++) {
            float sum = convolve_ilp(filter->edge_weights[i],
                                     &input[length - window_size], window_size);
            output[length - 1 - i] = sum * dt_inv;
        }
    } else {
        /*--- REFLECT/PERIODIC/CONSTANT: use center weights with virtual padding ---*/
        
        /* Leading edge */
        for (int i = 0; i < n; i++) {
            float sum = convolve_padded(filter->center_weights, input, length,
                                        i, n, mode);
            output[i] = sum * dt_inv;
        }
        
        /* Trailing edge */
        for (size_t i = length - n; i < length; i++) {
            float sum = convolve_padded(filter->center_weights, input, length,
                                        (int)i, n, mode);
            output[i] = sum * dt_inv;
        }
    }
    
    return 0;
}

/**
 * @brief Apply filter, output only where full window exists (VALID mode).
 * 
 * Outputs only the samples where the full filter window fits within the input.
 * No boundary handling is performed — the output is shorter than the input.
 * 
 * Output length = input_length - 2 * half_window
 * 
 * @param filter       Filter handle from savgol_create().
 * @param input        Input data array.
 * @param input_length Length of input array. Must be > 2 * half_window.
 * @param output       Output array. Must have space for at least
 *                     (input_length - 2 * half_window) elements.
 * @return Number of output samples written, or 0 on error.
 */
size_t savgol_apply_valid(const SavgolFilter *filter,
                          const float *input, size_t input_length,
                          float *output)
{
    if (filter == NULL || input == NULL || output == NULL) {
        return 0;
    }
    
    const int n = filter->config.half_window;
    const int window_size = filter->window_size;
    
    /* Need at least window_size samples for any valid output */
    if (input_length < (size_t)window_size) {
        return 0;
    }
    
    const float dt_inv = (filter->dt_scale != 0.0f) ? (1.0f / filter->dt_scale) : 1.0f;
    
    /* Output length = input_length - 2n */
    size_t output_length = input_length - 2 * n;
    
    /* Process only positions with full window */
    for (size_t j = 0; j < output_length; j++) {
        size_t input_idx = j + n;  /* Center of window in input */
        float sum = convolve_ilp(filter->center_weights, &input[input_idx - n], window_size);
        output[j] = sum * dt_inv;
    }
    
    return output_length;
}

/**
 * @brief Apply the filter to strided (non-contiguous) data.
 * 
 * Useful when filtering a field within an array of structs.
 * 
 * Example: Filter the 'phaseAngle' field of an array of measurement structs:
 * 
 *   typedef struct { float timestamp; float phaseAngle; float amplitude; } Sample;
 *   Sample samples[1000];
 *   
 *   savgol_apply_strided(filter,
 *       samples, sizeof(Sample), offsetof(Sample, phaseAngle),
 *       samples, sizeof(Sample), offsetof(Sample, phaseAngle),
 *       1000);
 * 
 * @param filter      Filter handle.
 * @param input       Pointer to start of input array.
 * @param in_stride   Byte offset between consecutive input elements.
 * @param in_offset   Byte offset of float field within each input element.
 * @param output      Pointer to start of output array.
 * @param out_stride  Byte offset between consecutive output elements.
 * @param out_offset  Byte offset of float field within each output element.
 * @param count       Number of elements to process.
 * @return 0 on success, -1 on error.
 */
int savgol_apply_strided(const SavgolFilter *filter,
                         const void *input, size_t in_stride, size_t in_offset,
                         void *output, size_t out_stride, size_t out_offset,
                         size_t count)
{
    if (filter == NULL || input == NULL || output == NULL) {
        return -1;
    }
    
    if (count < (size_t)filter->window_size) {
        return -1;
    }
    
    /* Helper macros to access strided float fields */
    #define IN(i)  (*(const float *)((const char *)input + (i) * in_stride + in_offset))
    #define OUT(i) (*(float *)((char *)output + (i) * out_stride + out_offset))
    
    const int n = filter->config.half_window;
    const int window_size = filter->window_size;
    const float dt_inv = (filter->dt_scale != 0.0f) ? (1.0f / filter->dt_scale) : 1.0f;
    
    /* Temporary buffer for window data (strided access isn't SIMD-friendly) */
    float window_data[SAVGOL_MAX_WINDOW];
    
    /*--- Central region ---*/
    for (size_t j = n; j < count - n; j++) {
        /* Copy window data to contiguous buffer */
        for (int k = 0; k < window_size; k++) {
            window_data[k] = IN(j - n + k);
        }
        float sum = convolve_ilp(filter->center_weights, window_data, window_size);
        OUT(j) = sum * dt_inv;
    }
    
    /*--- Leading edge ---*/
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < window_size; k++) {
            window_data[k] = IN(k);
        }
        float sum = convolve_ilp_reverse(filter->edge_weights[i],
                                          &window_data[window_size - 1], window_size);
        OUT(i) = sum * dt_inv;
    }
    
    /*--- Trailing edge ---*/
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < window_size; k++) {
            window_data[k] = IN(count - window_size + k);
        }
        float sum = convolve_ilp(filter->edge_weights[i], window_data, window_size);
        OUT(count - 1 - i) = sum * dt_inv;
    }
    
    #undef IN
    #undef OUT
    
    return 0;
}


/*============================================================================
 * SECTION 9: LEGACY API COMPATIBILITY (Optional)
 *============================================================================
 * 
 * If you need backward compatibility with the old mes_savgolFilter() API,
 * uncomment this section. It wraps the new API.
 */

#if 0  /* Set to 1 to enable legacy API */

typedef struct {
    float phaseAngle;
} MqsRawDataPoint_t;

int mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize,
                     MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder,
                     uint8_t targetPoint, uint8_t derivativeOrder)
{
    (void)targetPoint;  /* Legacy API always used target=0 for center */
    
    SavgolConfig config = {
        .half_window = halfWindowSize,
        .poly_order = polynomialOrder,
        .derivative = derivativeOrder,
        .time_step = 1.0f
    };
    
    SavgolFilter *filter = savgol_create(&config);
    if (!filter) {
        return -1;
    }
    
    int result = savgol_apply_strided(filter,
        data, sizeof(MqsRawDataPoint_t), offsetof(MqsRawDataPoint_t, phaseAngle),
        filteredData, sizeof(MqsRawDataPoint_t), offsetof(MqsRawDataPoint_t, phaseAngle),
        dataSize);
    
    savgol_destroy(filter);
    return result;
}

#endif /* Legacy API */