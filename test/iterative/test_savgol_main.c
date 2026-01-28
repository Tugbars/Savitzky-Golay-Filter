/**
 * @file test_savgol_main.c
 * @brief Benchmark and demonstration for Savitzky-Golay filter.
 * 
 * Build:
 *   gcc -O2 -o test_savgol_main test_savgol_main.c -lsavgolFilter -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>

#include "savgolFilter.h"

/*============================================================================
 * UTILITY
 *============================================================================*/

static void print_data(const char *label, const float *data, size_t size)
{
    printf("%s[%lu] = [", label, (unsigned long)size);
    for (size_t i = 0; i < size; ++i) {
        if (i < 5) {
            printf("%.3f, ", data[i]);
        } else if (i == 5) {
            printf("..., ");
        } else if (i >= size - 5) {
            printf("%.3f%s", data[i], (i < size - 1) ? ", " : "");
        }
    }
    printf("]\n");
}

static void print_build_info(void)
{
    printf("=== Savitzky-Golay Filter Benchmark ===\n");
    printf("Build: Embedded-optimized (pure C)\n");
    printf("Max half-window: %d\n", SAVGOL_MAX_HALF_WINDOW);
    printf("Max window size: %d\n", SAVGOL_MAX_WINDOW);
    printf("\n");
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(void)
{
    print_build_info();

    /* Test dataset */
    float dataset[] = {
        11.272f, 11.254f, 11.465f, 11.269f, 11.31f,  11.388f, 11.385f, 11.431f, 11.333f, 11.437f,
        11.431f, 11.527f, 11.483f, 11.449f, 11.544f, 11.39f,  11.469f, 11.526f, 11.498f, 11.522f,
        11.709f, 11.503f, 11.564f, 11.428f, 11.714f, 11.707f, 11.619f, 11.751f, 11.626f, 11.681f,
        11.838f, 11.658f, 11.859f, 11.916f, 11.814f, 11.833f, 12.046f, 11.966f, 12.031f, 12.079f,
        11.958f, 12.114f, 12.041f, 12.186f, 12.048f, 12.258f, 12.312f, 12.126f, 12.159f, 12.393f,
        12.221f, 12.45f,  12.439f, 12.282f, 12.373f, 12.573f, 12.647f, 12.545f, 12.467f, 12.629f,
        12.686f, 12.668f, 12.748f, 12.71f,  12.852f, 13.02f,  12.848f, 13.144f, 13.225f, 13.211f,
        13.496f, 13.311f, 13.33f,  13.634f, 13.189f, 13.623f, 13.671f, 13.618f, 13.645f, 13.779f,
        14.006f, 14.13f,  14.071f, 14.277f, 14.223f, 14.457f, 14.378f, 14.698f, 14.599f, 14.84f,
        15.143f, 15.106f, 15.343f, 15.506f, 15.665f, 15.889f, 15.878f, 16.055f, 16.153f, 15.966f,
        16.637f, 16.783f, 16.746f, 17.193f, 16.877f, 17.656f, 17.522f, 17.842f, 18.086f, 18.336f,
        18.863f, 18.977f, 19.534f, 19.308f, 19.626f, 19.956f, 20.221f, 20.673f, 20.59f,  21.229f,
        21.767f, 22.225f, 22.477f, 22.695f, 22.828f, 23.586f, 23.776f, 24.39f,  25.316f, 24.639f,
        25.767f, 26.469f, 26.976f, 27.651f, 27.807f, 28.089f, 28.869f, 29.964f, 30.367f, 30.159f,
        31.133f, 32.034f, 33.131f, 32.775f, 34.372f, 34.516f, 35.603f, 36.214f, 37.742f, 38.868f,
        38.702f, 39.811f, 40.818f, 41.422f, 41.521f, 42.57f,  42.819f, 42.871f, 42.944f, 43.851f,
        44.086f, 44.272f, 44.466f, 44.274f, 44.473f, 44.348f, 43.932f, 43.817f, 43.48f,  42.943f,
        42.491f, 41.793f, 41.071f, 39.491f, 39.231f, 38.365f, 37.833f, 36.583f, 35.787f, 34.949f,
        33.006f, 32.827f, 32.266f, 31.012f, 30.436f, 29.737f, 28.097f, 28.76f,  27.068f, 26.195f,
        25.262f, 24.677f, 24.211f, 23.574f, 22.868f, 22.781f, 22.258f, 21.475f, 21.247f, 21.982f,
        20.771f, 20.383f, 20.349f, 19.866f, 19.433f, 18.573f, 18.723f, 18.325f, 18.084f, 18.226f,
        17.492f, 17.505f, 16.762f, 16.907f, 16.606f, 16.265f, 16.234f, 15.983f, 16.147f, 15.811f,
        15.667f, 15.509f, 15.325f, 15.031f, 14.884f, 14.881f, 14.836f, 14.814f, 14.706f, 14.158f,
        14.399f, 14.123f, 14.084f, 14.173f, 13.963f, 13.981f, 14.218f, 13.898f, 13.869f, 13.701f,
        13.397f, 13.528f, 13.321f, 13.071f, 13.393f, 13.164f, 12.876f, 13.021f, 12.989f, 12.869f,
        13.004f, 12.833f, 12.795f, 12.661f, 12.761f, 12.547f, 12.775f, 12.388f, 12.425f, 12.564f,
        12.408f, 12.301f, 12.469f, 12.173f, 12.323f, 12.248f, 12.281f, 12.208f, 11.887f, 12.149f,
        12.073f, 12.053f, 11.88f,  12.066f, 11.958f, 12.007f, 11.868f, 11.921f, 11.898f, 11.804f,
        11.7f,   11.81f,  11.758f, 11.717f, 11.715f, 11.611f, 11.719f, 11.679f, 11.619f, 11.58f,
        11.576f, 11.589f, 11.491f, 11.659f, 11.506f, 11.431f, 11.535f, 11.349f, 11.464f, 11.343f,
        11.492f, 11.407f, 11.479f, 11.269f, 11.355f, 11.323f, 11.341f, 11.238f, 11.32f,  11.333f,
        11.262f, 11.31f,  11.221f, 11.302f, 11.135f, 11.139f, 11.217f, 11.343f, 11.225f, 11.089f,
        11.079f, 11.127f, 11.082f, 11.141f, 11.186f, 11.184f, 11.231f, 11.025f, 11.058f, 11.076f,
        11.087f, 11.047f, 11.02f,  10.996f, 10.906f, 11.144f, 11.005f, 10.911f, 10.993f, 10.858f,
        11.086f, 10.954f, 10.906f, 11.026f, 11.005f, 10.934f, 10.922f, 10.914f, 10.955f, 11.057f,
        10.967f, 10.811f, 10.833f, 10.747f, 10.821f, 10.946f, 10.844f, 10.838f, 10.848f, 10.847f
    };
    
    const size_t data_size = sizeof(dataset) / sizeof(dataset[0]);
    float filtered[360];
    
    /* Filter configuration */
    SavgolConfig config = {
        .half_window = 6,
        .poly_order  = 3,
        .derivative  = 0,
        .time_step   = 1.0f,
        .boundary    = SAVGOL_BOUNDARY_POLYNOMIAL
    };
    
    printf("Filter parameters:\n");
    printf("  Half-window: %d\n", config.half_window);
    printf("  Window size: %d points\n", 2 * config.half_window + 1);
    printf("  Polynomial order: %d\n", config.poly_order);
    printf("  Derivative order: %d\n", config.derivative);
    printf("  Data points: %lu\n", (unsigned long)data_size);
    printf("\n");
    
    /* Create filter */
    SavgolFilter *filter = savgol_create(&config);
    if (filter == NULL) {
        fprintf(stderr, "Failed to create filter\n");
        return 1;
    }
    
    printf("Filter created successfully.\n\n");
    
    /* Single run for verification */
    int result = savgol_apply(filter, dataset, filtered, data_size);
    if (result != 0) {
        fprintf(stderr, "Filter application failed\n");
        savgol_destroy(filter);
        return 1;
    }
    
    print_data("Input ", dataset, data_size);
    print_data("Output", filtered, data_size);
    printf("\n");
    
    /* Benchmark */
    const int iterations = 10000;
    
    printf("Running benchmark (%d iterations)...\n", iterations);
    
    clock_t tic = clock();
    for (int iter = 0; iter < iterations; ++iter) {
        savgol_apply(filter, dataset, filtered, data_size);
    }
    clock_t toc = clock();
    
    double total_time = (double)(toc - tic) / CLOCKS_PER_SEC;
    double avg_time = total_time / iterations;
    double throughput = (data_size * iterations) / total_time;
    
    printf("\nBenchmark Results:\n");
    printf("  Total time: %.6f seconds\n", total_time);
    printf("  Average time per iteration: %.6f seconds (%.3f ms)\n",
           avg_time, avg_time * 1000.0);
    printf("  Throughput: %.2f samples/sec (%.2f Msamples/sec)\n",
           throughput, throughput / 1e6);
    
    /* Strided access test */
    printf("\n--- Strided Access Test ---\n");
    
    typedef struct { float phaseAngle; } MqsRawDataPoint_t;
    
    MqsRawDataPoint_t raw_points[360];
    MqsRawDataPoint_t filtered_points[360];
    
    for (size_t i = 0; i < data_size; i++) {
        raw_points[i].phaseAngle = dataset[i];
        filtered_points[i].phaseAngle = 0.0f;
    }
    
    result = savgol_apply_strided(filter,
        raw_points,      sizeof(MqsRawDataPoint_t), offsetof(MqsRawDataPoint_t, phaseAngle),
        filtered_points, sizeof(MqsRawDataPoint_t), offsetof(MqsRawDataPoint_t, phaseAngle),
        data_size);
    
    if (result == 0) {
        printf("Strided access: OK\n");
        
        int mismatches = 0;
        for (size_t i = 0; i < data_size; i++) {
            if (fabsf(filtered_points[i].phaseAngle - filtered[i]) > 1e-5f) {
                mismatches++;
            }
        }
        printf("Verification: %s (%d mismatches)\n", 
               mismatches == 0 ? "PASS" : "FAIL", mismatches);
    } else {
        printf("Strided access: FAILED\n");
    }
    
    /* First derivative test */
    printf("\n--- First Derivative Test ---\n");
    
    SavgolConfig deriv_config = SAVGOL_DERIV1(10, 3, 1.0f);
    SavgolFilter *deriv_filter = savgol_create(&deriv_config);
    
    if (deriv_filter) {
        float derivative[360];
        savgol_apply(deriv_filter, dataset, derivative, data_size);
        
        printf("First derivative at peak region (indices 155-165):\n");
        for (int i = 155; i <= 165; i++) {
            printf("  d[%d] = %+.4f  (value=%.2f)\n", i, derivative[i], dataset[i]);
        }
        printf("\nNote: Derivative crosses zero near the peak (~index 162)\n");
        
        savgol_destroy(deriv_filter);
    }
    
    /* Cleanup */
    savgol_destroy(filter);
    
    printf("\nBenchmark complete.\n");
    return 0;
}