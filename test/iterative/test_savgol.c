/**
 * @file test_savgol.c
 * @brief Unit tests for Savitzky-Golay filter (public API only).
 * 
 * Tests link against the compiled library. Internal functions are
 * implicitly tested through the public API.
 * 
 * Build:
 *   gcc -o test_savgol test_savgol.c -lsavgolFilter -lm
 *   
 * Or via CMake (preferred).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>

#include "savgolFilter.h"

/*============================================================================
 * TEST FRAMEWORK
 *============================================================================*/

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) printf("  %-50s ", name)
#define PASS() do { printf("[PASS]\n"); g_tests_passed++; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); g_tests_failed++; } while(0)

/*============================================================================
 * FILTER LIFECYCLE TESTS
 *============================================================================*/

static void test_create_valid(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(5, 2);
    
    TEST("Create filter with valid config");
    SavgolFilter *f = savgol_create(&cfg);
    if (f != NULL) PASS();
    else { FAIL("returned NULL"); return; }
    
    savgol_destroy(f);
}

static void test_destroy_null(void)
{
    TEST("Destroy NULL filter (should not crash)");
    savgol_destroy(NULL);
    PASS();
}

static void test_create_invalid_half_window(void)
{
    SavgolConfig cfg = { .half_window = 0, .poly_order = 2, .derivative = 0, .time_step = 1.0f };
    
    TEST("Reject invalid config: half_window = 0");
    SavgolFilter *f = savgol_create(&cfg);
    if (f == NULL) PASS();
    else { FAIL("should have returned NULL"); savgol_destroy(f); }
}

static void test_create_invalid_poly_order(void)
{
    /* half_window=2 -> window_size=5, so poly_order must be < 5 */
    SavgolConfig cfg = { .half_window = 2, .poly_order = 10, .derivative = 0, .time_step = 1.0f };
    
    TEST("Reject invalid config: poly_order >= window_size");
    SavgolFilter *f = savgol_create(&cfg);
    if (f == NULL) PASS();
    else { FAIL("should have returned NULL"); savgol_destroy(f); }
}

static void test_create_derivative_exceeds_order(void)
{
    SavgolConfig cfg = { .half_window = 5, .poly_order = 2, .derivative = 3, .time_step = 1.0f };
    
    TEST("Reject invalid config: derivative > poly_order");
    SavgolFilter *f = savgol_create(&cfg);
    if (f == NULL) PASS();
    else { FAIL("should have returned NULL"); savgol_destroy(f); }
}

/*============================================================================
 * WEIGHT PROPERTY TESTS
 *============================================================================*/

static void test_weights_sum_to_one(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(5, 3);
    SavgolFilter *f = savgol_create(&cfg);
    
    TEST("Smoothing weights sum to 1");
    float sum = 0.0f;
    for (int i = 0; i < f->window_size; i++) {
        sum += f->center_weights[i];
    }
    if (fabsf(sum - 1.0f) < 1e-5f) PASS();
    else { char buf[64]; snprintf(buf, 64, "sum = %.6f", sum); FAIL(buf); }
    
    savgol_destroy(f);
}

static void test_weights_symmetric(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(5, 3);
    SavgolFilter *f = savgol_create(&cfg);
    int ws = f->window_size;
    
    TEST("Smoothing weights symmetric: w[0] == w[2n]");
    if (fabsf(f->center_weights[0] - f->center_weights[ws-1]) < 1e-6f) PASS();
    else FAIL("asymmetric");
    
    TEST("Smoothing weights symmetric: w[1] == w[2n-1]");
    if (fabsf(f->center_weights[1] - f->center_weights[ws-2]) < 1e-6f) PASS();
    else FAIL("asymmetric");
    
    savgol_destroy(f);
}

static void test_derivative_weights_antisymmetric(void)
{
    SavgolConfig cfg = SAVGOL_DERIV1(5, 3, 1.0f);
    SavgolFilter *f = savgol_create(&cfg);
    int ws = f->window_size;
    
    TEST("Derivative weights antisymmetric: w[0] == -w[2n]");
    if (fabsf(f->center_weights[0] + f->center_weights[ws-1]) < 1e-6f) PASS();
    else FAIL("not antisymmetric");
    
    TEST("Derivative weights: center weight == 0");
    int center = ws / 2;
    if (fabsf(f->center_weights[center]) < 1e-6f) PASS();
    else { char buf[64]; snprintf(buf, 64, "center = %.6f", f->center_weights[center]); FAIL(buf); }
    
    savgol_destroy(f);
}

/*============================================================================
 * FILTER APPLICATION TESTS
 *============================================================================*/

static void test_apply_constant(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(5, 2);
    SavgolFilter *f = savgol_create(&cfg);
    
    /* Constant signal should be unchanged */
    float input[50], output[50];
    for (int i = 0; i < 50; i++) input[i] = 42.0f;
    
    TEST("Apply to constant signal: returns 0");
    int result = savgol_apply(f, input, output, 50);
    if (result == 0) PASS();
    else FAIL("returned non-zero");
    
    TEST("Apply to constant signal: output unchanged");
    int errors = 0;
    for (int i = 0; i < 50; i++) {
        if (fabsf(output[i] - 42.0f) > 0.01f) errors++;
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    savgol_destroy(f);
}

static void test_apply_linear(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(5, 2);
    SavgolFilter *f = savgol_create(&cfg);
    
    /* Linear signal: y = 3x + 7 */
    float input[50], output[50];
    for (int i = 0; i < 50; i++) input[i] = 3.0f * i + 7.0f;
    
    savgol_apply(f, input, output, 50);
    
    /* Interior should be preserved exactly */
    TEST("Apply to linear signal: interior preserved");
    int errors = 0;
    for (int i = 10; i < 40; i++) {
        float expected = 3.0f * i + 7.0f;
        if (fabsf(output[i] - expected) > 0.01f) errors++;
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    savgol_destroy(f);
}

static void test_derivative_linear(void)
{
    /* For y = 3x, first derivative should be 3 */
    SavgolConfig cfg = SAVGOL_DERIV1(5, 2, 1.0f);
    SavgolFilter *f = savgol_create(&cfg);
    
    float input[50], output[50];
    for (int i = 0; i < 50; i++) input[i] = 3.0f * i;
    
    savgol_apply(f, input, output, 50);
    
    TEST("First derivative of y=3x equals 3");
    int errors = 0;
    for (int i = 10; i < 40; i++) {
        if (fabsf(output[i] - 3.0f) > 0.01f) errors++;
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    savgol_destroy(f);
}

static void test_inplace(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(5, 2);
    SavgolFilter *f = savgol_create(&cfg);
    
    float data[50];
    for (int i = 0; i < 50; i++) data[i] = 42.0f;
    
    TEST("In-place filtering: returns 0");
    int result = savgol_apply(f, data, data, 50);
    if (result == 0) PASS();
    else FAIL("returned non-zero");
    
    TEST("In-place filtering: constant preserved");
    int errors = 0;
    for (int i = 0; i < 50; i++) {
        if (fabsf(data[i] - 42.0f) > 0.01f) errors++;
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    savgol_destroy(f);
}

/*============================================================================
 * STRIDED ACCESS TESTS
 *============================================================================*/

typedef struct {
    float timestamp;
    float value;
    float other;
} TestSample;

static void test_strided(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(3, 2);
    SavgolFilter *f = savgol_create(&cfg);
    
    TestSample input[30], output[30];
    for (int i = 0; i < 30; i++) {
        input[i].timestamp = (float)i;
        input[i].value = 100.0f;  /* Constant */
        input[i].other = 999.0f;
        output[i].timestamp = 0.0f;
        output[i].value = 0.0f;
        output[i].other = 0.0f;
    }
    
    TEST("Strided access: returns 0");
    int result = savgol_apply_strided(f,
        input,  sizeof(TestSample), offsetof(TestSample, value),
        output, sizeof(TestSample), offsetof(TestSample, value),
        30);
    if (result == 0) PASS();
    else FAIL("returned non-zero");
    
    TEST("Strided access: filtered field correct");
    int errors = 0;
    for (int i = 0; i < 30; i++) {
        if (fabsf(output[i].value - 100.0f) > 0.01f) errors++;
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    TEST("Strided access: other fields unchanged");
    int unchanged = 1;
    for (int i = 0; i < 30; i++) {
        if (output[i].timestamp != 0.0f || output[i].other != 0.0f) {
            unchanged = 0;
            break;
        }
    }
    if (unchanged) PASS();
    else FAIL("other fields modified");
    
    savgol_destroy(f);
}

/*============================================================================
 * BOUNDARY MODE TESTS
 *============================================================================*/

static void test_boundary_reflect(void)
{
    SavgolConfig cfg = { .half_window = 5, .poly_order = 2, .derivative = 0,
                         .time_step = 1.0f, .boundary = SAVGOL_BOUNDARY_REFLECT };
    SavgolFilter *f = savgol_create(&cfg);
    
    float input[50], output[50];
    for (int i = 0; i < 50; i++) input[i] = 42.0f;
    
    savgol_apply(f, input, output, 50);
    
    TEST("REFLECT mode: constant signal unchanged");
    int errors = 0;
    for (int i = 0; i < 50; i++) {
        if (fabsf(output[i] - 42.0f) > 0.01f) errors++;
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    savgol_destroy(f);
}

static void test_boundary_periodic(void)
{
    SavgolConfig cfg = { .half_window = 5, .poly_order = 2, .derivative = 0,
                         .time_step = 1.0f, .boundary = SAVGOL_BOUNDARY_PERIODIC };
    SavgolFilter *f = savgol_create(&cfg);
    
    float input[50], output[50];
    for (int i = 0; i < 50; i++) input[i] = 42.0f;
    
    savgol_apply(f, input, output, 50);
    
    TEST("PERIODIC mode: constant signal unchanged");
    int errors = 0;
    for (int i = 0; i < 50; i++) {
        if (fabsf(output[i] - 42.0f) > 0.01f) errors++;
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    savgol_destroy(f);
}

static void test_boundary_constant(void)
{
    SavgolConfig cfg = { .half_window = 5, .poly_order = 2, .derivative = 0,
                         .time_step = 1.0f, .boundary = SAVGOL_BOUNDARY_CONSTANT };
    SavgolFilter *f = savgol_create(&cfg);
    
    float input[50], output[50];
    for (int i = 0; i < 50; i++) input[i] = 42.0f;
    
    savgol_apply(f, input, output, 50);
    
    TEST("CONSTANT mode: constant signal unchanged");
    int errors = 0;
    for (int i = 0; i < 50; i++) {
        if (fabsf(output[i] - 42.0f) > 0.01f) errors++;
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    savgol_destroy(f);
}

/*============================================================================
 * VALID MODE TESTS
 *============================================================================*/

static void test_valid_mode_length(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(5, 2);
    SavgolFilter *f = savgol_create(&cfg);
    
    float input[100], output[100];
    for (int i = 0; i < 100; i++) input[i] = (float)i;
    
    size_t out_len = savgol_apply_valid(f, input, 100, output);
    
    TEST("savgol_apply_valid: correct output length");
    size_t expected = 100 - 2 * 5;  /* 90 */
    if (out_len == expected) PASS();
    else { char buf[64]; snprintf(buf, 64, "got %lu, expected %lu", (unsigned long)out_len, (unsigned long)expected); FAIL(buf); }
    
    savgol_destroy(f);
}

static void test_valid_mode_linear(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(5, 2);
    SavgolFilter *f = savgol_create(&cfg);
    
    float input[100], output[100];
    for (int i = 0; i < 100; i++) input[i] = (float)i;
    
    size_t out_len = savgol_apply_valid(f, input, 100, output);
    
    TEST("savgol_apply_valid: linear signal preserved");
    int errors = 0;
    for (size_t i = 0; i < out_len; i++) {
        float expected = (float)(i + 5);  /* Offset by half_window */
        if (fabsf(output[i] - expected) > 0.1f) errors++;
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    savgol_destroy(f);
}

/*============================================================================
 * NOISE REDUCTION TEST
 *============================================================================*/

static void test_noise_reduction(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(10, 3);
    SavgolFilter *f = savgol_create(&cfg);
    
    /* Create noisy sine wave */
    float input[200], output[200];
    srand(12345);
    for (int i = 0; i < 200; i++) {
        float signal = sinf(2.0f * 3.14159f * i / 50.0f);
        float noise = 0.2f * ((float)rand() / RAND_MAX - 0.5f);
        input[i] = signal + noise;
    }
    
    savgol_apply(f, input, output, 200);
    
    /* Compute RMS error vs true signal */
    float input_err = 0.0f, output_err = 0.0f;
    for (int i = 20; i < 180; i++) {
        float true_val = sinf(2.0f * 3.14159f * i / 50.0f);
        input_err += (input[i] - true_val) * (input[i] - true_val);
        output_err += (output[i] - true_val) * (output[i] - true_val);
    }
    input_err = sqrtf(input_err / 160);
    output_err = sqrtf(output_err / 160);
    
    TEST("Noise reduction: output error < input error");
    if (output_err < input_err) PASS();
    else { char buf[64]; snprintf(buf, 64, "in=%.3f, out=%.3f", input_err, output_err); FAIL(buf); }
    
    savgol_destroy(f);
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(void)
{
    printf("\n=== Savitzky-Golay Filter Tests ===\n\n");
    
    printf("Filter Lifecycle:\n");
    test_create_valid();
    test_destroy_null();
    test_create_invalid_half_window();
    test_create_invalid_poly_order();
    test_create_derivative_exceeds_order();
    
    printf("\nWeight Properties:\n");
    test_weights_sum_to_one();
    test_weights_symmetric();
    test_derivative_weights_antisymmetric();
    
    printf("\nFilter Application:\n");
    test_apply_constant();
    test_apply_linear();
    test_derivative_linear();
    test_inplace();
    
    printf("\nStrided Access:\n");
    test_strided();
    
    printf("\nBoundary Modes:\n");
    test_boundary_reflect();
    test_boundary_periodic();
    test_boundary_constant();
    
    printf("\nVALID Mode:\n");
    test_valid_mode_length();
    test_valid_mode_linear();
    
    printf("\nNoise Reduction:\n");
    test_noise_reduction();
    
    printf("\n=== Results: %d passed, %d failed ===\n\n",
           g_tests_passed, g_tests_failed);
    
    return (g_tests_failed > 0) ? 1 : 0;
}