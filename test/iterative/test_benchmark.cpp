#include <stdio.h>
#include <time.h>
#include "savgolFilter.h"

#ifdef _WIN32
#include <windows.h>
double get_time(void) {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}
#else
#include <time.h>
double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
#endif

void benchmark_threshold(void) {
    printf("=== Finding Optimal PARALLEL_THRESHOLD ===\n\n");
    
    // Test data sizes
    int test_sizes[] = {100, 200, 500, 1000, 2000, 5000, 10000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    uint8_t halfWindowSize = 12;
    uint8_t polynomialOrder = 4;
    
    printf("Window size: %d, Polynomial order: %d\n", 
           2 * halfWindowSize + 1, polynomialOrder);
    printf("Iterations per test: 100\n\n");
    
    printf("%-10s | %-12s | %-12s | %-10s\n", 
           "Data Size", "Sequential", "Parallel", "Speedup");
    printf("-----------|--------------|--------------|------------\n");
    
    for (int t = 0; t < num_tests; t++) {
        size_t dataSize = test_sizes[t];
        
        // Allocate data
        MqsRawDataPoint_t *data = malloc(dataSize * sizeof(MqsRawDataPoint_t));
        MqsRawDataPoint_t *filtered = malloc(dataSize * sizeof(MqsRawDataPoint_t));
        
        for (size_t i = 0; i < dataSize; i++) {
            data[i].phaseAngle = (float)i;
        }
        
        // Warmup
        mes_savgolFilter(data, dataSize, halfWindowSize, filtered, 
                        polynomialOrder, 0, 0);
        
        const int iterations = 100;
        
        // Sequential timing
        double seq_start = get_time();
        for (int i = 0; i < iterations; i++) {
            mes_savgolFilter(data, dataSize, halfWindowSize, filtered, 
                            polynomialOrder, 0, 0);
        }
        double seq_end = get_time();
        double seq_time = (seq_end - seq_start) / iterations * 1e6; // microseconds
        
#ifdef SAVGOL_PARALLEL_BUILD
        // Parallel timing with 4 threads
        double par_start = get_time();
        for (int i = 0; i < iterations; i++) {
            mes_savgolFilter_threaded(data, dataSize, halfWindowSize, filtered, 
                                     polynomialOrder, 0, 0, 4);
        }
        double par_end = get_time();
        double par_time = (par_end - par_start) / iterations * 1e6; // microseconds
        
        double speedup = seq_time / par_time;
        
        printf("%-10d | %8.2f µs | %8.2f µs | %6.2fx %s\n", 
               (int)dataSize, seq_time, par_time, speedup,
               speedup > 1.5 ? "✓" : speedup > 1.0 ? "~" : "✗");
#else
        printf("%-10d | %8.2f µs | %-12s | %-10s\n", 
               (int)dataSize, seq_time, "N/A", "N/A");
#endif
        
        free(data);
        free(filtered);
    }
    
    printf("\n");
    printf("Legend:\n");
    printf("  ✓ = Good speedup (>1.5x) - worth parallelizing\n");
    printf("  ~ = Marginal speedup (1.0-1.5x) - borderline\n");
    printf("  ✗ = Slowdown (<1.0x) - overhead dominates\n");
    printf("\n");
    printf("Recommendation: Set PARALLEL_THRESHOLD to the first ✓ value\n");
}

int main(void) {
    benchmark_threshold();
    return 0;
}
