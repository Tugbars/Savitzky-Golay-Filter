/**
 * @file test_savgol2d.c
 * @brief Tests for 2D Savitzky-Golay filter.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "savgol2d.h"

/*============================================================================
 * TEST FRAMEWORK
 *============================================================================*/

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) printf("  %-55s ", name)
#define PASS() do { printf("[PASS]\n"); g_tests_passed++; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); g_tests_failed++; } while(0)

/*============================================================================
 * LIFECYCLE TESTS
 *============================================================================*/

static void test_create_destroy(void)
{
    Savgol2DConfig cfg = {
        .half_window_x = 3, .half_window_y = 3,
        .poly_order = 2,
        .deriv_x = 0, .deriv_y = 0,
        .delta_x = 1.0f, .delta_y = 1.0f
    };
    
    TEST("Create 2D filter");
    Savgol2DFilter *f = savgol2d_create(&cfg);
    if (f != NULL) PASS();
    else { FAIL("returned NULL"); return; }
    
    TEST("Destroy 2D filter");
    savgol2d_destroy(f);
    PASS();
    
    TEST("Destroy NULL (no crash)");
    savgol2d_destroy(NULL);
    PASS();
}

static void test_config_validation(void)
{
    TEST("Reject zero half_window_x");
    Savgol2DConfig cfg = {0, 3, 2, 0, 0, 1.0f, 1.0f};
    if (!savgol2d_config_valid(&cfg)) PASS();
    else FAIL("should reject");
    
    TEST("Reject derivative > poly_order");
    cfg = (Savgol2DConfig){3, 3, 2, 2, 1, 1.0f, 1.0f};  /* dx+dy=3 > order=2 */
    if (!savgol2d_config_valid(&cfg)) PASS();
    else FAIL("should reject");
    
    TEST("Reject window too small for polynomial");
    cfg = (Savgol2DConfig){1, 1, 4, 0, 0, 1.0f, 1.0f};  /* 3x3=9, but order 4 needs 15 terms */
    if (!savgol2d_config_valid(&cfg)) PASS();
    else FAIL("should reject");
    
    TEST("Accept valid config");
    cfg = (Savgol2DConfig){3, 3, 2, 0, 0, 1.0f, 1.0f};
    if (savgol2d_config_valid(&cfg)) PASS();
    else FAIL("should accept");
}

/*============================================================================
 * WEIGHT PROPERTY TESTS
 *============================================================================*/

static void test_smoothing_weights_sum(void)
{
    Savgol2DConfig cfg = {
        .half_window_x = 2, .half_window_y = 2,
        .poly_order = 2,
        .deriv_x = 0, .deriv_y = 0,
        .delta_x = 1.0f, .delta_y = 1.0f
    };
    
    Savgol2DFilter *f = savgol2d_create(&cfg);
    
    TEST("Smoothing weights sum to 1");
    float sum = 0.0f;
    for (int i = 0; i < f->window_area; i++) {
        sum += f->weights[i];
    }
    if (fabsf(sum - 1.0f) < 1e-5f) PASS();
    else { char buf[64]; snprintf(buf, 64, "sum = %.6f", sum); FAIL(buf); }
    
    savgol2d_destroy(f);
}

static void test_derivative_weights_sum(void)
{
    /* First derivative weights should sum to ~0 */
    Savgol2DConfig cfg = {
        .half_window_x = 3, .half_window_y = 3,
        .poly_order = 2,
        .deriv_x = 1, .deriv_y = 0,
        .delta_x = 1.0f, .delta_y = 1.0f
    };
    
    Savgol2DFilter *f = savgol2d_create(&cfg);
    
    TEST("Derivative weights sum to ~0");
    float sum = 0.0f;
    for (int i = 0; i < f->window_area; i++) {
        sum += f->weights[i];
    }
    if (fabsf(sum) < 1e-5f) PASS();
    else { char buf[64]; snprintf(buf, 64, "sum = %.6f", sum); FAIL(buf); }
    
    savgol2d_destroy(f);
}

/*============================================================================
 * FILTER APPLICATION TESTS
 *============================================================================*/

static void test_constant_image(void)
{
    Savgol2DConfig cfg = {3, 3, 2, 0, 0, 1.0f, 1.0f};
    Savgol2DFilter *f = savgol2d_create(&cfg);
    
    /* Create constant image */
    int rows = 20, cols = 20;
    float *input = (float *)malloc(rows * cols * sizeof(float));
    float *output = (float *)malloc(rows * cols * sizeof(float));
    
    for (int i = 0; i < rows * cols; i++) input[i] = 42.0f;
    
    savgol2d_apply(f, input, rows, cols, cols, output, cols, SAVGOL2D_BOUNDARY_CONSTANT);
    
    TEST("Constant image: output unchanged");
    int errors = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (fabsf(output[i] - 42.0f) > 0.01f) errors++;
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    free(input);
    free(output);
    savgol2d_destroy(f);
}

static void test_linear_image(void)
{
    /* Linear image f(x,y) = 2x + 3y should be preserved */
    Savgol2DConfig cfg = {3, 3, 2, 0, 0, 1.0f, 1.0f};
    Savgol2DFilter *f = savgol2d_create(&cfg);
    
    int rows = 30, cols = 30;
    float *input = (float *)malloc(rows * cols * sizeof(float));
    float *output = (float *)malloc(rows * cols * sizeof(float));
    
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            input[y * cols + x] = 2.0f * x + 3.0f * y;
        }
    }
    
    savgol2d_apply_valid(f, input, rows, cols, cols, output, cols - 6);
    
    TEST("Linear image: interior preserved");
    int errors = 0;
    int out_rows = rows - 6, out_cols = cols - 6;
    for (int y = 0; y < out_rows; y++) {
        for (int x = 0; x < out_cols; x++) {
            float expected = 2.0f * (x + 3) + 3.0f * (y + 3);
            if (fabsf(output[y * out_cols + x] - expected) > 0.01f) errors++;
        }
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    free(input);
    free(output);
    savgol2d_destroy(f);
}

static void test_gradient_x(void)
{
    /* For f(x,y) = 5x, ∂f/∂x = 5 */
    Savgol2DConfig cfg = {3, 3, 2, 1, 0, 1.0f, 1.0f};
    Savgol2DFilter *f = savgol2d_create(&cfg);
    
    int rows = 20, cols = 20;
    float *input = (float *)malloc(rows * cols * sizeof(float));
    float *output = (float *)malloc(rows * cols * sizeof(float));
    
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            input[y * cols + x] = 5.0f * x;
        }
    }
    
    savgol2d_apply_valid(f, input, rows, cols, cols, output, cols - 6);
    
    TEST("Gradient ∂(5x)/∂x = 5");
    int errors = 0;
    int out_rows = rows - 6, out_cols = cols - 6;
    for (int y = 0; y < out_rows; y++) {
        for (int x = 0; x < out_cols; x++) {
            if (fabsf(output[y * out_cols + x] - 5.0f) > 0.01f) errors++;
        }
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    free(input);
    free(output);
    savgol2d_destroy(f);
}

static void test_gradient_y(void)
{
    /* For f(x,y) = 7y, ∂f/∂y = 7 */
    Savgol2DConfig cfg = {3, 3, 2, 0, 1, 1.0f, 1.0f};
    Savgol2DFilter *f = savgol2d_create(&cfg);
    
    int rows = 20, cols = 20;
    float *input = (float *)malloc(rows * cols * sizeof(float));
    float *output = (float *)malloc(rows * cols * sizeof(float));
    
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            input[y * cols + x] = 7.0f * y;
        }
    }
    
    savgol2d_apply_valid(f, input, rows, cols, cols, output, cols - 6);
    
    TEST("Gradient ∂(7y)/∂y = 7");
    int errors = 0;
    int out_rows = rows - 6, out_cols = cols - 6;
    for (int y = 0; y < out_rows; y++) {
        for (int x = 0; x < out_cols; x++) {
            if (fabsf(output[y * out_cols + x] - 7.0f) > 0.01f) errors++;
        }
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    free(input);
    free(output);
    savgol2d_destroy(f);
}

static void test_hessian_xx(void)
{
    /* For f(x,y) = x², ∂²f/∂x² = 2 */
    Savgol2DConfig cfg = {3, 3, 2, 2, 0, 1.0f, 1.0f};
    Savgol2DFilter *f = savgol2d_create(&cfg);
    
    int rows = 20, cols = 20;
    float *input = (float *)malloc(rows * cols * sizeof(float));
    float *output = (float *)malloc(rows * cols * sizeof(float));
    
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            input[y * cols + x] = (float)(x * x);
        }
    }
    
    savgol2d_apply_valid(f, input, rows, cols, cols, output, cols - 6);
    
    TEST("Hessian ∂²(x²)/∂x² = 2");
    int errors = 0;
    int out_rows = rows - 6, out_cols = cols - 6;
    for (int y = 0; y < out_rows; y++) {
        for (int x = 0; x < out_cols; x++) {
            if (fabsf(output[y * out_cols + x] - 2.0f) > 0.01f) errors++;
        }
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    free(input);
    free(output);
    savgol2d_destroy(f);
}

static void test_hessian_yy(void)
{
    /* For f(x,y) = 3y², ∂²f/∂y² = 6 */
    Savgol2DConfig cfg = {3, 3, 2, 0, 2, 1.0f, 1.0f};
    Savgol2DFilter *f = savgol2d_create(&cfg);
    
    int rows = 20, cols = 20;
    float *input = (float *)malloc(rows * cols * sizeof(float));
    float *output = (float *)malloc(rows * cols * sizeof(float));
    
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            input[y * cols + x] = 3.0f * y * y;
        }
    }
    
    savgol2d_apply_valid(f, input, rows, cols, cols, output, cols - 6);
    
    TEST("Hessian ∂²(3y²)/∂y² = 6");
    int errors = 0;
    int out_rows = rows - 6, out_cols = cols - 6;
    for (int y = 0; y < out_rows; y++) {
        for (int x = 0; x < out_cols; x++) {
            if (fabsf(output[y * out_cols + x] - 6.0f) > 0.01f) errors++;
        }
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    free(input);
    free(output);
    savgol2d_destroy(f);
}

static void test_mixed_derivative(void)
{
    /* For f(x,y) = 4xy, ∂²f/∂x∂y = 4 */
    Savgol2DConfig cfg = {3, 3, 2, 1, 1, 1.0f, 1.0f};
    Savgol2DFilter *f = savgol2d_create(&cfg);
    
    int rows = 20, cols = 20;
    float *input = (float *)malloc(rows * cols * sizeof(float));
    float *output = (float *)malloc(rows * cols * sizeof(float));
    
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            input[y * cols + x] = 4.0f * x * y;
        }
    }
    
    savgol2d_apply_valid(f, input, rows, cols, cols, output, cols - 6);
    
    TEST("Mixed ∂²(4xy)/∂x∂y = 4");
    int errors = 0;
    int out_rows = rows - 6, out_cols = cols - 6;
    for (int y = 0; y < out_rows; y++) {
        for (int x = 0; x < out_cols; x++) {
            if (fabsf(output[y * out_cols + x] - 4.0f) > 0.01f) errors++;
        }
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    free(input);
    free(output);
    savgol2d_destroy(f);
}

/*============================================================================
 * CONVENIENCE FUNCTION TESTS
 *============================================================================*/

static void test_gradient_convenience(void)
{
    int rows = 20, cols = 20;
    float *input = (float *)malloc(rows * cols * sizeof(float));
    float *grad_x = (float *)malloc(rows * cols * sizeof(float));
    float *grad_y = (float *)malloc(rows * cols * sizeof(float));
    
    /* f(x,y) = 2x + 3y */
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            input[y * cols + x] = 2.0f * x + 3.0f * y;
        }
    }
    
    int ret = savgol2d_gradient(3, 3, 2, input, rows, cols, cols,
                                grad_x, grad_y, 1.0f, 1.0f,
                                SAVGOL2D_BOUNDARY_CONSTANT);
    
    TEST("savgol2d_gradient returns 0");
    if (ret == 0) PASS();
    else FAIL("returned error");
    
    TEST("savgol2d_gradient: ∂f/∂x = 2 (interior)");
    int errors = 0;
    for (int y = 5; y < rows - 5; y++) {
        for (int x = 5; x < cols - 5; x++) {
            if (fabsf(grad_x[y * cols + x] - 2.0f) > 0.05f) errors++;
        }
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    TEST("savgol2d_gradient: ∂f/∂y = 3 (interior)");
    errors = 0;
    for (int y = 5; y < rows - 5; y++) {
        for (int x = 5; x < cols - 5; x++) {
            if (fabsf(grad_y[y * cols + x] - 3.0f) > 0.05f) errors++;
        }
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    free(input);
    free(grad_x);
    free(grad_y);
}

static void test_hessian_convenience(void)
{
    int rows = 20, cols = 20;
    float *input = (float *)malloc(rows * cols * sizeof(float));
    float *hess_xx = (float *)malloc(rows * cols * sizeof(float));
    float *hess_xy = (float *)malloc(rows * cols * sizeof(float));
    float *hess_yy = (float *)malloc(rows * cols * sizeof(float));
    
    /* f(x,y) = x² + 2xy + 3y² */
    /* ∂²f/∂x² = 2, ∂²f/∂x∂y = 2, ∂²f/∂y² = 6 */
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            input[y * cols + x] = (float)(x*x + 2*x*y + 3*y*y);
        }
    }
    
    int ret = savgol2d_hessian(3, 3, 2, input, rows, cols, cols,
                               hess_xx, hess_xy, hess_yy,
                               1.0f, 1.0f, SAVGOL2D_BOUNDARY_CONSTANT);
    
    TEST("savgol2d_hessian returns 0");
    if (ret == 0) PASS();
    else FAIL("returned error");
    
    TEST("savgol2d_hessian: ∂²f/∂x² = 2");
    int errors = 0;
    for (int y = 5; y < rows - 5; y++) {
        for (int x = 5; x < cols - 5; x++) {
            if (fabsf(hess_xx[y * cols + x] - 2.0f) > 0.05f) errors++;
        }
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    TEST("savgol2d_hessian: ∂²f/∂x∂y = 2");
    errors = 0;
    for (int y = 5; y < rows - 5; y++) {
        for (int x = 5; x < cols - 5; x++) {
            if (fabsf(hess_xy[y * cols + x] - 2.0f) > 0.05f) errors++;
        }
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    TEST("savgol2d_hessian: ∂²f/∂y² = 6");
    errors = 0;
    for (int y = 5; y < rows - 5; y++) {
        for (int x = 5; x < cols - 5; x++) {
            if (fabsf(hess_yy[y * cols + x] - 6.0f) > 0.05f) errors++;
        }
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    free(input);
    free(hess_xx);
    free(hess_xy);
    free(hess_yy);
}

static void test_laplacian_convenience(void)
{
    int rows = 20, cols = 20;
    float *input = (float *)malloc(rows * cols * sizeof(float));
    float *output = (float *)malloc(rows * cols * sizeof(float));
    
    /* f(x,y) = x² + y² */
    /* ∇²f = ∂²f/∂x² + ∂²f/∂y² = 2 + 2 = 4 */
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            input[y * cols + x] = (float)(x*x + y*y);
        }
    }
    
    int ret = savgol2d_laplacian(3, 3, 2, input, rows, cols, cols,
                                  output, 1.0f, 1.0f, SAVGOL2D_BOUNDARY_CONSTANT);
    
    TEST("savgol2d_laplacian returns 0");
    if (ret == 0) PASS();
    else FAIL("returned error");
    
    TEST("savgol2d_laplacian: ∇²(x²+y²) = 4");
    int errors = 0;
    for (int y = 5; y < rows - 5; y++) {
        for (int x = 5; x < cols - 5; x++) {
            if (fabsf(output[y * cols + x] - 4.0f) > 0.1f) errors++;
        }
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    free(input);
    free(output);
}

/*============================================================================
 * RECTANGULAR WINDOW TESTS
 *============================================================================*/

static void test_rectangular_window(void)
{
    /* Non-square window: 5 wide, 3 tall */
    Savgol2DConfig cfg = {
        .half_window_x = 2, .half_window_y = 1,  /* 5x3 window */
        .poly_order = 2,
        .deriv_x = 0, .deriv_y = 0,
        .delta_x = 1.0f, .delta_y = 1.0f
    };
    
    TEST("Create filter with rectangular window (5x3)");
    Savgol2DFilter *f = savgol2d_create(&cfg);
    if (f && f->window_width == 5 && f->window_height == 3) PASS();
    else { FAIL("wrong dimensions"); if (f) savgol2d_destroy(f); return; }
    
    /* Test on constant image */
    int rows = 20, cols = 20;
    float *input = (float *)malloc(rows * cols * sizeof(float));
    float *output = (float *)malloc(rows * cols * sizeof(float));
    
    for (int i = 0; i < rows * cols; i++) input[i] = 100.0f;
    
    savgol2d_apply(f, input, rows, cols, cols, output, cols, SAVGOL2D_BOUNDARY_CONSTANT);
    
    TEST("Rectangular window: constant image preserved");
    int errors = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (fabsf(output[i] - 100.0f) > 0.01f) errors++;
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    free(input);
    free(output);
    savgol2d_destroy(f);
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(void)
{
    printf("\n=== 2D Savitzky-Golay Filter Tests ===\n\n");
    
    printf("Lifecycle:\n");
    test_create_destroy();
    test_config_validation();
    
    printf("\nWeight Properties:\n");
    test_smoothing_weights_sum();
    test_derivative_weights_sum();
    
    printf("\nFilter Application:\n");
    test_constant_image();
    test_linear_image();
    
    printf("\nGradient (First Derivatives):\n");
    test_gradient_x();
    test_gradient_y();
    
    printf("\nHessian (Second Derivatives):\n");
    test_hessian_xx();
    test_hessian_yy();
    test_mixed_derivative();
    
    printf("\nConvenience Functions:\n");
    test_gradient_convenience();
    test_hessian_convenience();
    test_laplacian_convenience();
    
    printf("\nRectangular Windows:\n");
    test_rectangular_window();
    
    printf("\n=== Results: %d passed, %d failed ===\n\n",
           g_tests_passed, g_tests_failed);
    
    return (g_tests_failed > 0) ? 1 : 0;
}
