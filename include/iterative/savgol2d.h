/**
 * @file savgol2d.h
 * @brief 2D Savitzky-Golay Filter — Full polynomial fit with Hessian support
 * 
 * Performs true 2D polynomial least-squares fitting over rectangular windows.
 * Supports all partial derivatives including mixed derivatives (∂²/∂x∂y).
 * 
 * Mathematical basis:
 *   Fits polynomial p(x,y) = Σ a_ij * x^i * y^j  where i+j ≤ poly_order
 *   to each window, then evaluates the polynomial or its derivatives at center.
 * 
 * Example polynomial terms for order 2:
 *   1, x, y, x², xy, y²  (6 terms)
 * 
 * Example polynomial terms for order 3:
 *   1, x, y, x², xy, y², x³, x²y, xy², y³  (10 terms)
 * 
 * Number of terms: (poly_order + 1) * (poly_order + 2) / 2
 * 
 * Constraint: Window must have at least as many points as polynomial terms
 *   (2*half_window_x + 1) * (2*half_window_y + 1) >= num_terms
 * 
 * Usage:
 * @code
 *   // Gaussian smoothing alternative
 *   Savgol2DConfig cfg = {
 *       .half_window_x = 5, .half_window_y = 5,
 *       .poly_order = 2,
 *       .deriv_x = 0, .deriv_y = 0,
 *       .delta_x = 1.0f, .delta_y = 1.0f
 *   };
 *   Savgol2DFilter *f = savgol2d_create(&cfg);
 *   savgol2d_apply(f, image, rows, cols, cols, output);
 *   savgol2d_destroy(f);
 *   
 *   // Compute gradient magnitude
 *   Savgol2DFilter *fx = savgol2d_create(&(Savgol2DConfig){5,5,2, 1,0, 1,1});
 *   Savgol2DFilter *fy = savgol2d_create(&(Savgol2DConfig){5,5,2, 0,1, 1,1});
 *   savgol2d_apply(fx, image, rows, cols, cols, grad_x);
 *   savgol2d_apply(fy, image, rows, cols, cols, grad_y);
 *   for (int i = 0; i < rows*cols; i++)
 *       grad_mag[i] = sqrtf(grad_x[i]*grad_x[i] + grad_y[i]*grad_y[i]);
 * @endcode
 * 
 * @author Tugbars Heptaskin
 */

#ifndef SAVGOL2D_H
#define SAVGOL2D_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

/** Maximum half-window size in each dimension */
#define SAVGOL2D_MAX_HALF_WINDOW 16

/** Maximum polynomial order */
#define SAVGOL2D_MAX_POLY_ORDER 6

/** Maximum number of polynomial terms: (6+1)*(6+2)/2 = 28 */
#define SAVGOL2D_MAX_TERMS 28

/** Maximum window area: (2*16+1)^2 = 1089 */
#define SAVGOL2D_MAX_WINDOW_AREA ((2*SAVGOL2D_MAX_HALF_WINDOW+1)*(2*SAVGOL2D_MAX_HALF_WINDOW+1))

/*============================================================================
 * TYPES
 *============================================================================*/

/**
 * @brief 2D filter configuration.
 */
typedef struct {
    uint8_t half_window_x;  /**< Half-window in x (columns). Window = 2n+1 */
    uint8_t half_window_y;  /**< Half-window in y (rows). Window = 2m+1 */
    uint8_t poly_order;     /**< Polynomial order. Terms: x^i * y^j where i+j ≤ order */
    uint8_t deriv_x;        /**< Derivative order in x: ∂^dx/∂x^dx */
    uint8_t deriv_y;        /**< Derivative order in y: ∂^dy/∂y^dy */
    float   delta_x;        /**< Sample spacing in x direction */
    float   delta_y;        /**< Sample spacing in y direction */
} Savgol2DConfig;

/**
 * @brief 2D filter with precomputed weights.
 */
typedef struct Savgol2DFilter {
    Savgol2DConfig config;
    int window_width;       /**< 2 * half_window_x + 1 */
    int window_height;      /**< 2 * half_window_y + 1 */
    int window_area;        /**< window_width * window_height */
    int num_terms;          /**< Number of polynomial terms */
    float scale;            /**< Derivative scaling: 1 / (delta_x^dx * delta_y^dy) */
    float *weights;         /**< Convolution weights [window_height][window_width] */
} Savgol2DFilter;

/**
 * @brief Boundary handling for 2D.
 */
typedef enum {
    SAVGOL2D_BOUNDARY_VALID = 0,    /**< Output only valid region (smaller output) */
    SAVGOL2D_BOUNDARY_CONSTANT,     /**< Extend edge pixels */
    SAVGOL2D_BOUNDARY_REFLECT       /**< Mirror at boundaries */
} Savgol2DBoundary;

/*============================================================================
 * LIFECYCLE
 *============================================================================*/

/**
 * @brief Create a 2D filter.
 * 
 * Allocates filter and precomputes convolution weights using least-squares.
 * 
 * @param config Filter configuration.
 * @return Filter handle, or NULL on error.
 */
Savgol2DFilter *savgol2d_create(const Savgol2DConfig *config);

/**
 * @brief Destroy a 2D filter.
 * @param filter Filter to destroy (NULL safe).
 */
void savgol2d_destroy(Savgol2DFilter *filter);

/*============================================================================
 * FILTERING
 *============================================================================*/

/**
 * @brief Apply 2D filter to image (VALID mode).
 * 
 * Output is smaller than input:
 *   out_rows = rows - 2 * half_window_y
 *   out_cols = cols - 2 * half_window_x
 * 
 * @param filter     Filter handle.
 * @param input      Input image (row-major).
 * @param rows       Number of rows in input.
 * @param cols       Number of columns in input.
 * @param in_stride  Row stride of input (typically = cols).
 * @param output     Output image (row-major).
 * @param out_stride Row stride of output.
 * @return 0 on success, -1 on error.
 */
int savgol2d_apply_valid(const Savgol2DFilter *filter,
                         const float *input, int rows, int cols, int in_stride,
                         float *output, int out_stride);

/**
 * @brief Apply 2D filter with boundary handling (same-size output).
 * 
 * @param filter     Filter handle.
 * @param input      Input image.
 * @param rows       Number of rows.
 * @param cols       Number of columns.
 * @param in_stride  Input row stride.
 * @param output     Output image (same size as input).
 * @param out_stride Output row stride.
 * @param boundary   Boundary handling mode.
 * @return 0 on success, -1 on error.
 */
int savgol2d_apply(const Savgol2DFilter *filter,
                   const float *input, int rows, int cols, int in_stride,
                   float *output, int out_stride,
                   Savgol2DBoundary boundary);

/*============================================================================
 * CONVENIENCE: COMMON OPERATIONS
 *============================================================================*/

/**
 * @brief Compute image gradient (∂/∂x, ∂/∂y).
 * 
 * @param half_win_x  Half-window in x.
 * @param half_win_y  Half-window in y.
 * @param poly_order  Polynomial order (typically 2 or 3).
 * @param input       Input image.
 * @param rows, cols  Image dimensions.
 * @param stride      Row stride.
 * @param grad_x      Output: ∂I/∂x (can be NULL to skip).
 * @param grad_y      Output: ∂I/∂y (can be NULL to skip).
 * @param delta_x, delta_y  Sample spacing.
 * @param boundary    Boundary mode.
 * @return 0 on success.
 */
int savgol2d_gradient(int half_win_x, int half_win_y, int poly_order,
                      const float *input, int rows, int cols, int stride,
                      float *grad_x, float *grad_y,
                      float delta_x, float delta_y,
                      Savgol2DBoundary boundary);

/**
 * @brief Compute Hessian matrix (∂²/∂x², ∂²/∂x∂y, ∂²/∂y²).
 * 
 * @param half_win_x  Half-window in x.
 * @param half_win_y  Half-window in y.
 * @param poly_order  Polynomial order (must be ≥ 2).
 * @param input       Input image.
 * @param rows, cols  Image dimensions.
 * @param stride      Row stride.
 * @param hess_xx     Output: ∂²I/∂x² (can be NULL).
 * @param hess_xy     Output: ∂²I/∂x∂y (can be NULL).
 * @param hess_yy     Output: ∂²I/∂y² (can be NULL).
 * @param delta_x, delta_y  Sample spacing.
 * @param boundary    Boundary mode.
 * @return 0 on success.
 */
int savgol2d_hessian(int half_win_x, int half_win_y, int poly_order,
                     const float *input, int rows, int cols, int stride,
                     float *hess_xx, float *hess_xy, float *hess_yy,
                     float delta_x, float delta_y,
                     Savgol2DBoundary boundary);

/**
 * @brief Compute Laplacian (∇²I = ∂²I/∂x² + ∂²I/∂y²).
 * 
 * @param half_win_x  Half-window in x.
 * @param half_win_y  Half-window in y.
 * @param poly_order  Polynomial order (must be ≥ 2).
 * @param input       Input image.
 * @param rows, cols  Image dimensions.
 * @param stride      Row stride.
 * @param output      Output: Laplacian.
 * @param delta_x, delta_y  Sample spacing.
 * @param boundary    Boundary mode.
 * @return 0 on success.
 */
int savgol2d_laplacian(int half_win_x, int half_win_y, int poly_order,
                       const float *input, int rows, int cols, int stride,
                       float *output,
                       float delta_x, float delta_y,
                       Savgol2DBoundary boundary);

/*============================================================================
 * UTILITIES
 *============================================================================*/

/**
 * @brief Get output dimensions for VALID mode.
 */
static inline void savgol2d_valid_size(const Savgol2DFilter *filter,
                                       int in_rows, int in_cols,
                                       int *out_rows, int *out_cols)
{
    *out_rows = in_rows - 2 * filter->config.half_window_y;
    *out_cols = in_cols - 2 * filter->config.half_window_x;
}

/**
 * @brief Get number of polynomial terms for given order.
 */
static inline int savgol2d_num_terms(int poly_order)
{
    return (poly_order + 1) * (poly_order + 2) / 2;
}

/**
 * @brief Check if configuration is valid.
 */
bool savgol2d_config_valid(const Savgol2DConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* SAVGOL2D_H */
