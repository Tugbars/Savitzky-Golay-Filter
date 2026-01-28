/**
 * @file savgol2d.c
 * @brief 2D Savitzky-Golay Filter — Implementation
 * 
 * Full 2D polynomial least-squares fitting with support for all partial
 * derivatives including mixed derivatives (Hessian).
 * 
 * Weight Computation:
 *   For window centered at origin, spanning x ∈ [-nx, nx], y ∈ [-ny, ny]:
 *   
 *   1. Build design matrix A of size [window_area × num_terms]
 *      Each row = [1, x, y, x², xy, y², x³, x²y, xy², y³, ...]
 *      for point (x, y) in window
 *   
 *   2. Compute pseudo-inverse: pinv(A) = (A^T A)^(-1) A^T
 *   
 *   3. Extract weights: Row of pinv(A) for monomial x^dx * y^dy
 *      gives convolution weights for derivative ∂^(dx+dy)/∂x^dx∂y^dy
 *   
 *   4. Scale by dx! * dy! for correct derivative normalization
 * 
 * @author Tugbars Heptaskin
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "savgol2d.h"

/*============================================================================
 * INTERNAL: MATRIX OPERATIONS
 *============================================================================*/

/**
 * @brief Compute factorial.
 */
static double factorial(int n)
{
    double result = 1.0;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

/**
 * @brief Get index of monomial x^i * y^j in the polynomial term list.
 * 
 * Terms are ordered by total degree, then by x power descending:
 *   order 0: 1                    (index 0)
 *   order 1: x, y                 (indices 1, 2)
 *   order 2: x², xy, y²           (indices 3, 4, 5)
 *   order 3: x³, x²y, xy², y³     (indices 6, 7, 8, 9)
 */
static int monomial_index(int i, int j)
{
    int total = i + j;
    /* Number of terms before this total degree */
    int base = total * (total + 1) / 2;
    /* Position within this degree (x^total, x^(total-1)*y, ..., y^total) */
    int offset = j;
    return base + offset;
}

/**
 * @brief Build design matrix A for polynomial fitting.
 * 
 * @param nx       Half-window in x.
 * @param ny       Half-window in y.
 * @param order    Polynomial order.
 * @param A        Output matrix [window_area × num_terms], row-major.
 * @param num_rows Output: number of rows (window_area).
 * @param num_cols Output: number of columns (num_terms).
 */
static void build_design_matrix(int nx, int ny, int order,
                                double *A, int *num_rows, int *num_cols)
{
    int width = 2 * nx + 1;
    int height = 2 * ny + 1;
    int area = width * height;
    int nterms = (order + 1) * (order + 2) / 2;
    
    *num_rows = area;
    *num_cols = nterms;
    
    int row = 0;
    for (int yi = -ny; yi <= ny; yi++) {
        for (int xi = -nx; xi <= nx; xi++) {
            double x = (double)xi;
            double y = (double)yi;
            
            /* Compute all monomials x^i * y^j where i+j <= order */
            for (int tot = 0; tot <= order; tot++) {
                for (int j = 0; j <= tot; j++) {
                    int i = tot - j;
                    int col = monomial_index(i, j);
                    A[row * nterms + col] = pow(x, i) * pow(y, j);
                }
            }
            row++;
        }
    }
}

/**
 * @brief Compute A^T * A.
 */
static void matrix_ata(const double *A, int m, int n, double *ATA)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += A[k * n + i] * A[k * n + j];
            }
            ATA[i * n + j] = sum;
        }
    }
}

/**
 * @brief Solve linear system ATA * x = b using Cholesky decomposition.
 * 
 * ATA must be symmetric positive definite.
 * 
 * @param ATA  n×n matrix (will be modified to store L).
 * @param b    Right-hand side vector (length n).
 * @param x    Solution vector (length n).
 * @param n    Dimension.
 * @return 0 on success, -1 if not positive definite.
 */
static int solve_cholesky(double *ATA, const double *b, double *x, int n)
{
    /* Cholesky decomposition: ATA = L * L^T */
    /* Store L in lower triangle of ATA */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = ATA[i * n + j];
            for (int k = 0; k < j; k++) {
                sum -= ATA[i * n + k] * ATA[j * n + k];
            }
            if (i == j) {
                if (sum <= 0.0) {
                    return -1;  /* Not positive definite */
                }
                ATA[i * n + i] = sqrt(sum);
            } else {
                ATA[i * n + j] = sum / ATA[j * n + j];
            }
        }
    }
    
    /* Forward substitution: L * y = b */
    double y[SAVGOL2D_MAX_TERMS];
    for (int i = 0; i < n; i++) {
        double sum = b[i];
        for (int j = 0; j < i; j++) {
            sum -= ATA[i * n + j] * y[j];
        }
        y[i] = sum / ATA[i * n + i];
    }
    
    /* Back substitution: L^T * x = y */
    for (int i = n - 1; i >= 0; i--) {
        double sum = y[i];
        for (int j = i + 1; j < n; j++) {
            sum -= ATA[j * n + i] * x[j];
        }
        x[i] = sum / ATA[i * n + i];
    }
    
    return 0;
}

/**
 * @brief Compute convolution weights for a specific derivative.
 * 
 * @param nx      Half-window x.
 * @param ny      Half-window y.
 * @param order   Polynomial order.
 * @param dx      Derivative order in x.
 * @param dy      Derivative order in y.
 * @param weights Output weights [height][width], row-major.
 * @return 0 on success.
 */
static int compute_weights(int nx, int ny, int order, int dx, int dy, float *weights)
{
    int width = 2 * nx + 1;
    int height = 2 * ny + 1;
    int area = width * height;
    int nterms = (order + 1) * (order + 2) / 2;
    
    /* Allocate design matrix */
    double *A = (double *)malloc(area * nterms * sizeof(double));
    double *ATA = (double *)malloc(nterms * nterms * sizeof(double));
    double *b = (double *)malloc(nterms * sizeof(double));
    double *x = (double *)malloc(nterms * sizeof(double));
    
    if (!A || !ATA || !b || !x) {
        free(A); free(ATA); free(b); free(x);
        return -1;
    }
    
    /* Build design matrix */
    int m, n;
    build_design_matrix(nx, ny, order, A, &m, &n);
    
    /* Derivative scaling factor: dx! * dy! */
    double deriv_scale = factorial(dx) * factorial(dy);
    
    /* Index of monomial x^dx * y^dy */
    int target_idx = monomial_index(dx, dy);
    
    /* 
     * We need row 'target_idx' of (A^T A)^(-1) A^T
     * 
     * Let e_k be unit vector with 1 at position k.
     * Row k of pinv(A) = e_k^T (A^T A)^(-1) A^T
     *                  = [(A^T A)^(-1) e_k]^T A^T
     * 
     * So we solve (A^T A) c = e_k, then compute A^T c = weights
     * 
     * Actually, we want weights[j] = [row k of pinv(A)][j]
     *                              = sum_i (A^T A)^(-1)[k,i] * A[j,i]
     * 
     * So: solve (A^T A) c = e_k to get column k of (A^T A)^(-1)
     *     Then weights = A * c
     */
    
    /* Compute A^T A */
    matrix_ata(A, m, n, ATA);
    
    /* Setup e_k (unit vector) */
    memset(b, 0, n * sizeof(double));
    b[target_idx] = 1.0;
    
    /* Solve (A^T A) c = e_k */
    /* Make a copy of ATA since Cholesky modifies it */
    double *ATA_copy = (double *)malloc(nterms * nterms * sizeof(double));
    memcpy(ATA_copy, ATA, nterms * nterms * sizeof(double));
    
    if (solve_cholesky(ATA_copy, b, x, n) != 0) {
        free(A); free(ATA); free(ATA_copy); free(b); free(x);
        return -1;
    }
    
    /* Compute weights = A * c, scaled by derivative factor */
    for (int j = 0; j < m; j++) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += A[j * n + i] * x[i];
        }
        weights[j] = (float)(sum * deriv_scale);
    }
    
    free(A);
    free(ATA);
    free(ATA_copy);
    free(b);
    free(x);
    
    return 0;
}

/*============================================================================
 * PUBLIC API: LIFECYCLE
 *============================================================================*/

bool savgol2d_config_valid(const Savgol2DConfig *config)
{
    if (config == NULL) return false;
    
    if (config->half_window_x == 0 || config->half_window_x > SAVGOL2D_MAX_HALF_WINDOW) {
        return false;
    }
    if (config->half_window_y == 0 || config->half_window_y > SAVGOL2D_MAX_HALF_WINDOW) {
        return false;
    }
    if (config->poly_order > SAVGOL2D_MAX_POLY_ORDER) {
        return false;
    }
    if (config->deriv_x + config->deriv_y > config->poly_order) {
        return false;
    }
    if (config->delta_x <= 0.0f || config->delta_y <= 0.0f) {
        return false;
    }
    
    /* Check window has enough points for polynomial */
    int width = 2 * config->half_window_x + 1;
    int height = 2 * config->half_window_y + 1;
    int area = width * height;
    int nterms = savgol2d_num_terms(config->poly_order);
    
    if (area < nterms) {
        return false;
    }
    
    return true;
}

Savgol2DFilter *savgol2d_create(const Savgol2DConfig *config)
{
    if (!savgol2d_config_valid(config)) {
        fprintf(stderr, "savgol2d_create: invalid configuration\n");
        return NULL;
    }
    
    Savgol2DFilter *filter = (Savgol2DFilter *)malloc(sizeof(Savgol2DFilter));
    if (!filter) return NULL;
    
    filter->config = *config;
    filter->window_width = 2 * config->half_window_x + 1;
    filter->window_height = 2 * config->half_window_y + 1;
    filter->window_area = filter->window_width * filter->window_height;
    filter->num_terms = savgol2d_num_terms(config->poly_order);
    
    /* Derivative scaling: 1 / (dx^deriv_x * dy^deriv_y) */
    filter->scale = 1.0f / (powf(config->delta_x, config->deriv_x) *
                            powf(config->delta_y, config->deriv_y));
    
    /* Allocate weights */
    filter->weights = (float *)malloc(filter->window_area * sizeof(float));
    if (!filter->weights) {
        free(filter);
        return NULL;
    }
    
    /* Compute weights */
    if (compute_weights(config->half_window_x, config->half_window_y,
                        config->poly_order, config->deriv_x, config->deriv_y,
                        filter->weights) != 0) {
        fprintf(stderr, "savgol2d_create: weight computation failed\n");
        free(filter->weights);
        free(filter);
        return NULL;
    }
    
    return filter;
}

void savgol2d_destroy(Savgol2DFilter *filter)
{
    if (filter) {
        free(filter->weights);
        free(filter);
    }
}

/*============================================================================
 * PUBLIC API: FILTERING
 *============================================================================*/

int savgol2d_apply_valid(const Savgol2DFilter *filter,
                         const float *input, int rows, int cols, int in_stride,
                         float *output, int out_stride)
{
    if (!filter || !input || !output) return -1;
    
    const int nx = filter->config.half_window_x;
    const int ny = filter->config.half_window_y;
    const int ww = filter->window_width;
    const float scale = filter->scale;
    const float *W = filter->weights;
    
    const int out_rows = rows - 2 * ny;
    const int out_cols = cols - 2 * nx;
    
    if (out_rows <= 0 || out_cols <= 0) return -1;
    
    /* Convolve */
    for (int oy = 0; oy < out_rows; oy++) {
        const int iy = oy + ny;  /* Input center y */
        
        for (int ox = 0; ox < out_cols; ox++) {
            const int ix = ox + nx;  /* Input center x */
            
            float sum = 0.0f;
            int w_idx = 0;
            
            for (int wy = -ny; wy <= ny; wy++) {
                const float *in_row = input + (iy + wy) * in_stride + (ix - nx);
                
                for (int wx = 0; wx < ww; wx++) {
                    sum += W[w_idx++] * in_row[wx];
                }
            }
            
            output[oy * out_stride + ox] = sum * scale;
        }
    }
    
    return 0;
}

int savgol2d_apply(const Savgol2DFilter *filter,
                   const float *input, int rows, int cols, int in_stride,
                   float *output, int out_stride,
                   Savgol2DBoundary boundary)
{
    if (!filter || !input || !output) return -1;
    
    const int nx = filter->config.half_window_x;
    const int ny = filter->config.half_window_y;
    const float scale = filter->scale;
    const float *W = filter->weights;
    
    if (boundary == SAVGOL2D_BOUNDARY_VALID) {
        /* Delegate to valid-only version, output smaller */
        return savgol2d_apply_valid(filter, input, rows, cols, in_stride,
                                    output + ny * out_stride + nx, out_stride);
    }
    
    /* Full output with boundary handling */
    for (int oy = 0; oy < rows; oy++) {
        for (int ox = 0; ox < cols; ox++) {
            float sum = 0.0f;
            int w_idx = 0;
            
            for (int wy = -ny; wy <= ny; wy++) {
                for (int wx = -nx; wx <= nx; wx++) {
                    int iy = oy + wy;
                    int ix = ox + wx;
                    
                    /* Handle boundary */
                    if (boundary == SAVGOL2D_BOUNDARY_REFLECT) {
                        if (iy < 0) iy = -iy - 1;
                        else if (iy >= rows) iy = 2 * rows - iy - 1;
                        if (ix < 0) ix = -ix - 1;
                        else if (ix >= cols) ix = 2 * cols - ix - 1;
                        
                        /* Clamp in case of double reflection */
                        if (iy < 0) iy = 0;
                        else if (iy >= rows) iy = rows - 1;
                        if (ix < 0) ix = 0;
                        else if (ix >= cols) ix = cols - 1;
                    } else {
                        /* CONSTANT: clamp to edge */
                        if (iy < 0) iy = 0;
                        else if (iy >= rows) iy = rows - 1;
                        if (ix < 0) ix = 0;
                        else if (ix >= cols) ix = cols - 1;
                    }
                    
                    sum += W[w_idx++] * input[iy * in_stride + ix];
                }
            }
            
            output[oy * out_stride + ox] = sum * scale;
        }
    }
    
    return 0;
}

/*============================================================================
 * PUBLIC API: CONVENIENCE FUNCTIONS
 *============================================================================*/

int savgol2d_gradient(int half_win_x, int half_win_y, int poly_order,
                      const float *input, int rows, int cols, int stride,
                      float *grad_x, float *grad_y,
                      float delta_x, float delta_y,
                      Savgol2DBoundary boundary)
{
    if (grad_x) {
        Savgol2DConfig cfg = {
            .half_window_x = (uint8_t)half_win_x,
            .half_window_y = (uint8_t)half_win_y,
            .poly_order = (uint8_t)poly_order,
            .deriv_x = 1, .deriv_y = 0,
            .delta_x = delta_x, .delta_y = delta_y
        };
        Savgol2DFilter *f = savgol2d_create(&cfg);
        if (!f) return -1;
        int ret = savgol2d_apply(f, input, rows, cols, stride, grad_x, stride, boundary);
        savgol2d_destroy(f);
        if (ret != 0) return ret;
    }
    
    if (grad_y) {
        Savgol2DConfig cfg = {
            .half_window_x = (uint8_t)half_win_x,
            .half_window_y = (uint8_t)half_win_y,
            .poly_order = (uint8_t)poly_order,
            .deriv_x = 0, .deriv_y = 1,
            .delta_x = delta_x, .delta_y = delta_y
        };
        Savgol2DFilter *f = savgol2d_create(&cfg);
        if (!f) return -1;
        int ret = savgol2d_apply(f, input, rows, cols, stride, grad_y, stride, boundary);
        savgol2d_destroy(f);
        if (ret != 0) return ret;
    }
    
    return 0;
}

int savgol2d_hessian(int half_win_x, int half_win_y, int poly_order,
                     const float *input, int rows, int cols, int stride,
                     float *hess_xx, float *hess_xy, float *hess_yy,
                     float delta_x, float delta_y,
                     Savgol2DBoundary boundary)
{
    if (poly_order < 2) {
        fprintf(stderr, "savgol2d_hessian: poly_order must be >= 2\n");
        return -1;
    }
    
    if (hess_xx) {
        Savgol2DConfig cfg = {
            .half_window_x = (uint8_t)half_win_x,
            .half_window_y = (uint8_t)half_win_y,
            .poly_order = (uint8_t)poly_order,
            .deriv_x = 2, .deriv_y = 0,
            .delta_x = delta_x, .delta_y = delta_y
        };
        Savgol2DFilter *f = savgol2d_create(&cfg);
        if (!f) return -1;
        int ret = savgol2d_apply(f, input, rows, cols, stride, hess_xx, stride, boundary);
        savgol2d_destroy(f);
        if (ret != 0) return ret;
    }
    
    if (hess_xy) {
        Savgol2DConfig cfg = {
            .half_window_x = (uint8_t)half_win_x,
            .half_window_y = (uint8_t)half_win_y,
            .poly_order = (uint8_t)poly_order,
            .deriv_x = 1, .deriv_y = 1,
            .delta_x = delta_x, .delta_y = delta_y
        };
        Savgol2DFilter *f = savgol2d_create(&cfg);
        if (!f) return -1;
        int ret = savgol2d_apply(f, input, rows, cols, stride, hess_xy, stride, boundary);
        savgol2d_destroy(f);
        if (ret != 0) return ret;
    }
    
    if (hess_yy) {
        Savgol2DConfig cfg = {
            .half_window_x = (uint8_t)half_win_x,
            .half_window_y = (uint8_t)half_win_y,
            .poly_order = (uint8_t)poly_order,
            .deriv_x = 0, .deriv_y = 2,
            .delta_x = delta_x, .delta_y = delta_y
        };
        Savgol2DFilter *f = savgol2d_create(&cfg);
        if (!f) return -1;
        int ret = savgol2d_apply(f, input, rows, cols, stride, hess_yy, stride, boundary);
        savgol2d_destroy(f);
        if (ret != 0) return ret;
    }
    
    return 0;
}

int savgol2d_laplacian(int half_win_x, int half_win_y, int poly_order,
                       const float *input, int rows, int cols, int stride,
                       float *output,
                       float delta_x, float delta_y,
                       Savgol2DBoundary boundary)
{
    if (poly_order < 2) {
        fprintf(stderr, "savgol2d_laplacian: poly_order must be >= 2\n");
        return -1;
    }
    
    /* Compute ∂²/∂x² */
    Savgol2DConfig cfg_xx = {
        .half_window_x = (uint8_t)half_win_x,
        .half_window_y = (uint8_t)half_win_y,
        .poly_order = (uint8_t)poly_order,
        .deriv_x = 2, .deriv_y = 0,
        .delta_x = delta_x, .delta_y = delta_y
    };
    Savgol2DFilter *fxx = savgol2d_create(&cfg_xx);
    if (!fxx) return -1;
    
    int ret = savgol2d_apply(fxx, input, rows, cols, stride, output, stride, boundary);
    savgol2d_destroy(fxx);
    if (ret != 0) return ret;
    
    /* Add ∂²/∂y² */
    Savgol2DConfig cfg_yy = {
        .half_window_x = (uint8_t)half_win_x,
        .half_window_y = (uint8_t)half_win_y,
        .poly_order = (uint8_t)poly_order,
        .deriv_x = 0, .deriv_y = 2,
        .delta_x = delta_x, .delta_y = delta_y
    };
    Savgol2DFilter *fyy = savgol2d_create(&cfg_yy);
    if (!fyy) return -1;
    
    /* Allocate temp buffer for yy */
    float *temp = (float *)malloc(rows * stride * sizeof(float));
    if (!temp) {
        savgol2d_destroy(fyy);
        return -1;
    }
    
    ret = savgol2d_apply(fyy, input, rows, cols, stride, temp, stride, boundary);
    savgol2d_destroy(fyy);
    
    if (ret == 0) {
        /* output += temp (Laplacian = ∂²/∂x² + ∂²/∂y²) */
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                output[y * stride + x] += temp[y * stride + x];
            }
        }
    }
    
    free(temp);
    return ret;
}
