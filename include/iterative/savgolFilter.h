/**
 * @file savgolFilter.h
 * @brief Savitzky-Golay Filter — Public API
 * 
 * A digital smoothing filter that fits successive subsets of data points
 * with a low-degree polynomial using least-squares. Can compute smoothed
 * values and derivatives.
 * 
 * Features:
 * - Configurable window size and polynomial order
 * - Multiple boundary handling modes
 * - Derivative computation (velocity, acceleration, etc.)
 * - Strided access for struct arrays
 * - VALID mode for truncated output without boundary artifacts
 * 
 * Thread Safety:
 * - Filter creation/destruction: NOT thread-safe
 * - Filter application: Thread-safe (filter is read-only after creation)
 * - Multiple threads can share one filter for concurrent filtering
 * 
 * @author Tugbars Heptaskin
 */

#ifndef SAVGOL_FILTER_H
#define SAVGOL_FILTER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * CONFIGURATION CONSTANTS
 *============================================================================*/

/** Maximum supported half-window size. Window spans [-n, +n] for n ≤ this. */
#define SAVGOL_MAX_HALF_WINDOW 32

/** Maximum window size (derived). */
#define SAVGOL_MAX_WINDOW (2 * SAVGOL_MAX_HALF_WINDOW + 1)

/** Maximum polynomial order. */
#define SAVGOL_MAX_POLY_ORDER 10

/** Maximum derivative order. */
#define SAVGOL_MAX_DERIVATIVE 4

/*============================================================================
 * TYPES
 *============================================================================*/

/**
 * @brief Boundary handling modes for edge samples.
 * 
 * When the filter window extends beyond data boundaries:
 * - POLYNOMIAL: Fit asymmetric polynomials (preserves signal features best)
 * - REFLECT: Mirror data at boundary
 * - PERIODIC: Wrap data around (for periodic signals)
 * - CONSTANT: Extend edge value
 */
typedef enum {
    SAVGOL_BOUNDARY_POLYNOMIAL = 0,  /**< Asymmetric polynomial fit (default) */
    SAVGOL_BOUNDARY_REFLECT,         /**< Mirror: [d1,d0 | d0,d1,d2...] */
    SAVGOL_BOUNDARY_PERIODIC,        /**< Wrap: [...dn-1 | d0,d1...] */
    SAVGOL_BOUNDARY_CONSTANT         /**< Extend edge: [d0,d0 | d0,d1,d2...] */
} SavgolBoundaryMode;

/**
 * @brief Filter configuration.
 * 
 * @param half_window  Half-window size n. Window spans 2n+1 points.
 *                     Valid range: [1, SAVGOL_MAX_HALF_WINDOW]
 * 
 * @param poly_order   Polynomial order m for least-squares fit.
 *                     Must be < 2n+1. Higher = more features preserved.
 *                     Typical values: 2 (quadratic), 3 (cubic), 4 (quartic).
 * 
 * @param derivative   Derivative order d.
 *                     0 = smoothing (default)
 *                     1 = first derivative (velocity)
 *                     2 = second derivative (acceleration)
 *                     Must be ≤ poly_order.
 * 
 * @param time_step    Time interval between samples (Δt).
 *                     Used to scale derivative outputs correctly.
 *                     Set to 1.0 for smoothing or if output scaling not needed.
 * 
 * @param boundary     Boundary handling mode. Default (0) = POLYNOMIAL.
 */
typedef struct {
    uint8_t half_window;
    uint8_t poly_order;
    uint8_t derivative;
    float   time_step;
    SavgolBoundaryMode boundary;
} SavgolConfig;

/**
 * @brief Filter state containing precomputed weights.
 * 
 * Created by savgol_create(), destroyed by savgol_destroy().
 * After creation, all fields are read-only and the filter can be
 * safely shared across threads.
 */
typedef struct SavgolFilter {
    SavgolConfig config;                              /**< Original configuration */
    int window_size;                                  /**< 2 * half_window + 1 */
    float dt_scale;                                   /**< time_step^derivative */
    float center_weights[SAVGOL_MAX_WINDOW];          /**< Weights for interior */
    float edge_weights[SAVGOL_MAX_HALF_WINDOW][SAVGOL_MAX_WINDOW]; /**< Edge weights */
} SavgolFilter;

/*============================================================================
 * LIFECYCLE
 *============================================================================*/

/**
 * @brief Create a filter with the given configuration.
 * 
 * Allocates memory and precomputes all filter weights. The returned
 * filter can be used for multiple calls to savgol_apply().
 * 
 * @param config  Filter parameters. Copied internally.
 * @return Filter handle, or NULL on error (invalid config or allocation failure).
 * 
 * @note Call savgol_destroy() when done to free resources.
 */
SavgolFilter *savgol_create(const SavgolConfig *config);

/**
 * @brief Destroy a filter and free resources.
 * 
 * @param filter  Filter to destroy. NULL is safe (no-op).
 */
void savgol_destroy(SavgolFilter *filter);

/*============================================================================
 * FILTERING
 *============================================================================*/

/**
 * @brief Apply the filter to a contiguous array.
 * 
 * @param filter  Filter handle from savgol_create().
 * @param input   Input data array.
 * @param output  Output array (may equal input for in-place).
 * @param length  Number of elements. Must be ≥ window_size.
 * @return 0 on success, -1 on error.
 */
int savgol_apply(const SavgolFilter *filter,
                 const float *input, float *output, size_t length);

/**
 * @brief Apply the filter to strided (non-contiguous) data.
 * 
 * Useful for filtering a field within an array of structs.
 * 
 * Example: Filter the 'value' field of a struct array:
 * @code
 *   typedef struct { float time; float value; } Sample;
 *   Sample data[1000];
 *   
 *   savgol_apply_strided(filter,
 *       data, sizeof(Sample), offsetof(Sample, value),
 *       data, sizeof(Sample), offsetof(Sample, value),
 *       1000);
 * @endcode
 * 
 * @param filter     Filter handle.
 * @param input      Base pointer to input data.
 * @param in_stride  Byte stride between input elements.
 * @param in_offset  Byte offset to float field within each element.
 * @param output     Base pointer to output data.
 * @param out_stride Byte stride between output elements.
 * @param out_offset Byte offset to float field within each element.
 * @param count      Number of elements.
 * @return 0 on success, -1 on error.
 */
int savgol_apply_strided(const SavgolFilter *filter,
                         const void *input, size_t in_stride, size_t in_offset,
                         void *output, size_t out_stride, size_t out_offset,
                         size_t count);

/**
 * @brief Apply filter with VALID output only (no boundary handling).
 * 
 * Outputs only samples where the full window fits within input data.
 * Output is shorter than input: output_length = input_length - 2*half_window
 * 
 * Useful when boundary artifacts are unacceptable and you prefer
 * shorter output over extrapolated edge values.
 * 
 * @param filter       Filter handle.
 * @param input        Input data array.
 * @param input_length Length of input array.
 * @param output       Output array (needs input_length - 2*half_window space).
 * @return Number of samples written to output, or 0 on error.
 */
size_t savgol_apply_valid(const SavgolFilter *filter,
                          const float *input, size_t input_length,
                          float *output);

/*============================================================================
 * CONVENIENCE MACROS
 *============================================================================*/

/** Create a smoothing filter config (no derivative). */
#define SAVGOL_SMOOTH(half_win, order) \
    (SavgolConfig){ .half_window = (half_win), .poly_order = (order), \
                    .derivative = 0, .time_step = 1.0f, .boundary = SAVGOL_BOUNDARY_POLYNOMIAL }

/** Create a first-derivative filter config. */
#define SAVGOL_DERIV1(half_win, order, dt) \
    (SavgolConfig){ .half_window = (half_win), .poly_order = (order), \
                    .derivative = 1, .time_step = (dt), .boundary = SAVGOL_BOUNDARY_POLYNOMIAL }

/** Create a second-derivative filter config. */
#define SAVGOL_DERIV2(half_win, order, dt) \
    (SavgolConfig){ .half_window = (half_win), .poly_order = (order), \
                    .derivative = 2, .time_step = (dt), .boundary = SAVGOL_BOUNDARY_POLYNOMIAL }

#ifdef __cplusplus
}
#endif

#endif /* SAVGOL_FILTER_H */