/**
 * @file test_savgol_stream.c
 * @brief Tests for streaming Savitzky-Golay filter.
 * 
 * Build:
 *   gcc -o test_savgol_stream test_savgol_stream.c -lsavgolStream -lsavgolFilter -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "savgolFilter.h"
#include "savgol_stream.h"

/*============================================================================
 * TEST FRAMEWORK
 *============================================================================*/

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) printf("  %-50s ", name)
#define PASS() do { printf("[PASS]\n"); g_tests_passed++; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); g_tests_failed++; } while(0)

/*============================================================================
 * TESTS
 *============================================================================*/

static void test_create_destroy(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(5, 2);
    
    TEST("Create streaming filter");
    SavgolStream *stream = savgol_stream_create(&cfg);
    if (stream != NULL) PASS();
    else { FAIL("returned NULL"); return; }
    
    TEST("Destroy streaming filter");
    savgol_stream_destroy(stream);
    PASS();
    
    TEST("Destroy NULL stream (no crash)");
    savgol_stream_destroy(NULL);
    PASS();
}

static void test_static_init(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(5, 2);
    SavgolFilter *filter = savgol_create(&cfg);
    SavgolStream stream;
    
    TEST("Static init with user-provided struct");
    int result = savgol_stream_init(&stream, filter);
    if (result == 0) PASS();
    else FAIL("returned error");
    
    TEST("Static init: stream ready after filling");
    for (int i = 0; i < 11; i++) {
        savgol_stream_push(&stream, 1.0f, NULL);
    }
    if (savgol_stream_ready(&stream)) PASS();
    else FAIL("not ready after window_size samples");
    
    savgol_destroy(filter);
}

static void test_latency(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(5, 2);  /* half_window = 5 */
    SavgolStream *stream = savgol_stream_create(&cfg);
    
    TEST("Latency equals half_window");
    if (savgol_stream_latency(stream) == 5) PASS();
    else FAIL("wrong latency");
    
    TEST("Not ready before buffer full");
    for (int i = 0; i < 10; i++) {  /* Push 10 samples (need 11) */
        bool valid;
        savgol_stream_push(stream, (float)i, &valid);
        if (valid) { FAIL("premature output"); goto cleanup; }
    }
    if (!savgol_stream_ready(stream)) PASS();
    else FAIL("ready too early");
    
    TEST("Ready after buffer full");
    bool valid;
    savgol_stream_push(stream, 10.0f, &valid);  /* 11th sample */
    if (savgol_stream_ready(stream) && valid) PASS();
    else FAIL("not ready or no output");
    
cleanup:
    savgol_stream_destroy(stream);
}

static void test_constant_signal(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(5, 2);
    SavgolStream *stream = savgol_stream_create(&cfg);
    
    const float constant = 42.0f;
    const int num_samples = 50;
    float outputs[100];
    int out_idx = 0;
    
    /* Push all samples using push_full */
    for (int i = 0; i < num_samples; i++) {
        float buf[10];
        int count = savgol_stream_push_full(stream, constant, buf, 10);
        for (int j = 0; j < count; j++) {
            outputs[out_idx++] = buf[j];
        }
    }
    
    /* Flush remaining */
    float tail[10];
    int tail_count = savgol_stream_flush(stream, tail, 10);
    for (int i = 0; i < tail_count; i++) {
        outputs[out_idx++] = tail[i];
    }
    
    TEST("Constant signal: output count matches input");
    if (out_idx == num_samples) PASS();
    else { char buf[64]; snprintf(buf, 64, "got %d, expected %d", out_idx, num_samples); FAIL(buf); }
    
    TEST("Constant signal: values unchanged");
    int errors = 0;
    for (int i = 0; i < out_idx; i++) {
        if (fabsf(outputs[i] - constant) > 0.01f) errors++;
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    savgol_stream_destroy(stream);
}

static void test_matches_batch(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(5, 3);
    
    SavgolStream *stream = savgol_stream_create(&cfg);
    SavgolFilter *batch_filter = savgol_create(&cfg);
    
    /* Test signal: noisy sine */
    const int N = 100;
    float input[100], batch_output[100], stream_output[200];
    
    srand(12345);
    for (int i = 0; i < N; i++) {
        input[i] = sinf(0.1f * i) + 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    }
    
    /* Batch filter */
    savgol_apply(batch_filter, input, batch_output, N);
    
    /* Streaming filter */
    int out_idx = 0;
    for (int i = 0; i < N; i++) {
        float buf[10];
        int count = savgol_stream_push_full(stream, input[i], buf, 10);
        for (int j = 0; j < count; j++) {
            stream_output[out_idx++] = buf[j];
        }
    }
    float tail[10];
    int tail_count = savgol_stream_flush(stream, tail, 10);
    for (int i = 0; i < tail_count; i++) {
        stream_output[out_idx++] = tail[i];
    }
    
    TEST("Streaming matches batch: output count");
    if (out_idx == N) PASS();
    else { char buf[64]; snprintf(buf, 64, "got %d, expected %d", out_idx, N); FAIL(buf); }
    
    TEST("Streaming matches batch: values match");
    float max_diff = 0.0f;
    for (int i = 0; i < N && i < out_idx; i++) {
        float diff = fabsf(stream_output[i] - batch_output[i]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff < 1e-5f) PASS();
    else { char buf[64]; snprintf(buf, 64, "max diff = %.2e", max_diff); FAIL(buf); }
    
    savgol_stream_destroy(stream);
    savgol_destroy(batch_filter);
}

static void test_derivative(void)
{
    SavgolConfig cfg = SAVGOL_DERIV1(5, 2, 1.0f);
    SavgolStream *stream = savgol_stream_create(&cfg);
    
    const int N = 50;
    float outputs[100];
    int out_idx = 0;
    
    for (int i = 0; i < N; i++) {
        float buf[10];
        int count = savgol_stream_push_full(stream, 2.0f * i, buf, 10);
        for (int j = 0; j < count; j++) {
            outputs[out_idx++] = buf[j];
        }
    }
    
    float tail[10];
    int tail_count = savgol_stream_flush(stream, tail, 10);
    for (int i = 0; i < tail_count; i++) {
        outputs[out_idx++] = tail[i];
    }
    
    TEST("Derivative of y=2x equals 2 (center)");
    int n = cfg.half_window;
    int errors = 0;
    for (int i = n; i < out_idx - n; i++) {
        if (fabsf(outputs[i] - 2.0f) > 0.01f) errors++;
    }
    if (errors == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d errors", errors); FAIL(buf); }
    
    savgol_stream_destroy(stream);
}

static void test_reset(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(3, 2);
    SavgolStream *stream = savgol_stream_create(&cfg);
    
    for (int i = 0; i < 10; i++) {
        savgol_stream_push(stream, (float)i, NULL);
    }
    
    TEST("Reset clears buffer");
    savgol_stream_reset(stream);
    if (!savgol_stream_ready(stream) && savgol_stream_buffered(stream) == 0) PASS();
    else FAIL("buffer not cleared");
    
    TEST("Can reuse after reset");
    for (int i = 0; i < 7; i++) {
        savgol_stream_push(stream, 1.0f, NULL);
    }
    if (savgol_stream_ready(stream)) PASS();
    else FAIL("not ready after refilling");
    
    savgol_stream_destroy(stream);
}

static void test_flush_count(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(8, 3);  /* half_window = 8 */
    SavgolStream *stream = savgol_stream_create(&cfg);
    
    for (int i = 0; i < 20; i++) {
        savgol_stream_push(stream, (float)i, NULL);
    }
    
    TEST("Flush returns half_window samples");
    float tail[20];
    int count = savgol_stream_flush(stream, tail, 20);
    if (count == 8) PASS();
    else { char buf[64]; snprintf(buf, 64, "got %d, expected 8", count); FAIL(buf); }
    
    TEST("Flush respects max_count");
    savgol_stream_reset(stream);
    for (int i = 0; i < 20; i++) {
        savgol_stream_push(stream, (float)i, NULL);
    }
    count = savgol_stream_flush(stream, tail, 3);
    if (count == 3) PASS();
    else { char buf[64]; snprintf(buf, 64, "got %d, expected 3", count); FAIL(buf); }
    
    savgol_stream_destroy(stream);
}

static void test_total_output_count(void)
{
    SavgolConfig cfg = SAVGOL_SMOOTH(5, 2);
    SavgolStream *stream = savgol_stream_create(&cfg);
    
    const int N = 100;
    int push_outputs = 0;
    
    for (int i = 0; i < N; i++) {
        float buf[10];
        int count = savgol_stream_push_full(stream, (float)i, buf, 10);
        push_outputs += count;
    }
    
    float tail[10];
    int flush_outputs = savgol_stream_flush(stream, tail, 10);
    
    TEST("Total outputs = input count");
    int total = push_outputs + flush_outputs;
    if (total == N) PASS();
    else { char buf[64]; snprintf(buf, 64, "got %d, expected %d", total, N); FAIL(buf); }
    
    TEST("samples_output counter matches");
    if (savgol_stream_samples_output(stream) == (size_t)total) PASS();
    else FAIL("counter mismatch");
    
    savgol_stream_destroy(stream);
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(void)
{
    printf("\n=== Savitzky-Golay Streaming Filter Tests ===\n\n");
    
    printf("Lifecycle:\n");
    test_create_destroy();
    test_static_init();
    
    printf("\nTiming & Latency:\n");
    test_latency();
    
    printf("\nFiltering Correctness:\n");
    test_constant_signal();
    test_matches_batch();
    test_derivative();
    
    printf("\nState Management:\n");
    test_reset();
    test_flush_count();
    test_total_output_count();
    
    printf("\n=== Results: %d passed, %d failed ===\n\n",
           g_tests_passed, g_tests_failed);
    
    return (g_tests_failed > 0) ? 1 : 0;
}
