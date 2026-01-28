/**
 * @file savgol_export.c
 * @brief Standalone tool to export Savitzky-Golay filter coefficients as C headers.
 * 
 * Generates precomputed weight tables for embedded systems, FPGAs, or any
 * environment where runtime weight computation is too expensive.
 * 
 * Build:
 *   gcc -O2 -o savgol_export savgol_export.c -lm
 * 
 * Usage:
 *   ./savgol_export -n <half_window> -m <poly_order> [-d <derivative>] [-o <output.h>]
 * 
 * Examples:
 *   ./savgol_export -n 5 -m 2                    # Print to stdout
 *   ./savgol_export -n 5 -m 2 -o coeffs.h       # Write to file
 *   ./savgol_export -n 10 -m 3 -d 1 -o deriv.h  # First derivative
 * 
 * The generated header can be used with a lightweight apply function that
 * only performs convolution (no weight computation at runtime).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

#define MAX_HALF_WINDOW 32
#define MAX_WINDOW (2 * MAX_HALF_WINDOW + 1)
#define MAX_POLY_ORDER 10
#define MAX_DERIVATIVE 4
#define GENFACT_TABLE_SIZE (2 * MAX_HALF_WINDOW + MAX_POLY_ORDER + 2)

/*============================================================================
 * MATH: GenFact Lookup Table
 *============================================================================*/

static float g_genfact_table[GENFACT_TABLE_SIZE][GENFACT_TABLE_SIZE];
static bool g_genfact_initialized = false;

static void genfact_init(void)
{
    if (g_genfact_initialized) return;
    
    for (int a = 0; a < GENFACT_TABLE_SIZE; a++) {
        g_genfact_table[a][0] = 1.0f;
        for (int b = 1; b < GENFACT_TABLE_SIZE; b++) {
            if (b > a) {
                g_genfact_table[a][b] = 0.0f;
            } else {
                double product = 1.0;
                for (int j = a - b + 1; j <= a; j++) {
                    product *= (double)j;
                }
                g_genfact_table[a][b] = (float)product;
            }
        }
    }
    g_genfact_initialized = true;
}

static inline float genfact_lookup(uint8_t a, uint8_t b)
{
    if (a >= GENFACT_TABLE_SIZE || b >= GENFACT_TABLE_SIZE) {
        return 0.0f;
    }
    return g_genfact_table[a][b];
}

/*============================================================================
 * MATH: Gram Polynomials
 *============================================================================*/

static float gram_poly(uint8_t half_window, uint8_t deriv_order,
                       uint8_t poly_order, int data_index)
{
    float buf0[MAX_DERIVATIVE + 1];
    float buf1[MAX_DERIVATIVE + 1];
    float buf2[MAX_DERIVATIVE + 1];
    
    float *prev2 = buf0;
    float *prev1 = buf1;
    float *curr  = buf2;
    
    const float n = (float)half_window;
    const float i = (float)data_index;
    
    /* Base case: k = 0 */
    for (uint8_t d = 0; d <= deriv_order; d++) {
        prev2[d] = (d == 0) ? 1.0f : 0.0f;
    }
    
    if (poly_order == 0) {
        return prev2[deriv_order];
    }
    
    /* k = 1 */
    const float inv_n = 1.0f / n;
    prev1[0] = inv_n * (i * prev2[0]);
    for (uint8_t d = 1; d <= deriv_order; d++) {
        prev1[d] = inv_n * (i * prev2[d] + (float)d * prev2[d - 1]);
    }
    
    if (poly_order == 1) {
        return prev1[deriv_order];
    }
    
    /* k >= 2 */
    const float two_n = 2.0f * n;
    for (uint8_t k = 2; k <= poly_order; k++) {
        const float k_f = (float)k;
        const float denom = k_f * (two_n - k_f + 1.0f);
        const float alpha = (4.0f * k_f - 2.0f) / denom;
        const float gamma = ((k_f - 1.0f) * (two_n + k_f)) / denom;
        
        curr[0] = alpha * (i * prev1[0]) - gamma * prev2[0];
        for (uint8_t d = 1; d <= deriv_order; d++) {
            float term = i * prev1[d] + (float)d * prev1[d - 1];
            curr[d] = alpha * term - gamma * prev2[d];
        }
        
        float *tmp = prev2;
        prev2 = prev1;
        prev1 = curr;
        curr = tmp;
    }
    
    return prev1[deriv_order];
}

/*============================================================================
 * WEIGHT COMPUTATION
 *============================================================================*/

static float compute_weight(uint8_t half_window, uint8_t poly_order,
                            uint8_t deriv_order, int data_index, int target)
{
    const uint8_t two_n = 2 * half_window;
    float weight = 0.0f;
    
    for (uint8_t k = 0; k <= poly_order; k++) {
        float num = genfact_lookup(two_n, k);
        float den = genfact_lookup(two_n + k + 1, k + 1);
        float factor = (float)(2 * k + 1) * (num / den);
        
        float gram_at_i = gram_poly(half_window, 0, k, data_index);
        float gram_at_t = gram_poly(half_window, deriv_order, k, target);
        
        weight += factor * gram_at_i * gram_at_t;
    }
    
    return weight;
}

static void compute_center_weights(uint8_t half_window, uint8_t poly_order,
                                   uint8_t deriv_order, float *weights)
{
    const int window_size = 2 * half_window + 1;
    for (int idx = 0; idx < window_size; idx++) {
        int data_index = idx - half_window;
        weights[idx] = compute_weight(half_window, poly_order, deriv_order,
                                      data_index, 0);
    }
}

static void compute_edge_weights(uint8_t half_window, uint8_t poly_order,
                                 uint8_t deriv_order,
                                 float edge_weights[MAX_HALF_WINDOW][MAX_WINDOW])
{
    const int window_size = 2 * half_window + 1;
    
    for (int edge_pos = 0; edge_pos < half_window; edge_pos++) {
        int target = half_window - edge_pos;
        for (int idx = 0; idx < window_size; idx++) {
            int data_index = idx - half_window;
            edge_weights[edge_pos][idx] = compute_weight(
                half_window, poly_order, deriv_order, data_index, target);
        }
    }
}

/*============================================================================
 * HEADER GENERATION
 *============================================================================*/

static void print_usage(const char *prog)
{
    fprintf(stderr, "Usage: %s -n <half_window> -m <poly_order> [options]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -n, --half-window=N   Half-window size (1-%d)\n", MAX_HALF_WINDOW);
    fprintf(stderr, "  -m, --poly-order=M    Polynomial order (< 2N+1)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Optional:\n");
    fprintf(stderr, "  -d, --derivative=D    Derivative order (default: 0)\n");
    fprintf(stderr, "  -o, --output=FILE     Output file (default: stdout)\n");
    fprintf(stderr, "  -p, --prefix=STR      Symbol prefix (default: SAVGOL)\n");
    fprintf(stderr, "  -h, --help            Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s -n 5 -m 2                      # 11-point quadratic smoother\n", prog);
    fprintf(stderr, "  %s -n 10 -m 3 -d 1 -o deriv.h    # First derivative filter\n", prog);
}

static void generate_header(FILE *out, uint8_t half_window, uint8_t poly_order,
                            uint8_t deriv_order, const char *prefix)
{
    const int window_size = 2 * half_window + 1;
    
    /* Compute weights */
    float center_weights[MAX_WINDOW];
    float edge_weights[MAX_HALF_WINDOW][MAX_WINDOW];
    
    genfact_init();
    compute_center_weights(half_window, poly_order, deriv_order, center_weights);
    compute_edge_weights(half_window, poly_order, deriv_order, edge_weights);
    
    /* Generate timestamp */
    time_t now = time(NULL);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    
    /* Generate guard macro */
    char guard[128];
    snprintf(guard, sizeof(guard), "%s_COEFFS_N%d_M%d_D%d_H",
             prefix, half_window, poly_order, deriv_order);
    
    /* Header comment */
    fprintf(out, "/**\n");
    fprintf(out, " * @file Auto-generated Savitzky-Golay filter coefficients\n");
    fprintf(out, " * \n");
    fprintf(out, " * DO NOT EDIT — regenerate with savgol_export\n");
    fprintf(out, " * Generated: %s\n", timestamp);
    fprintf(out, " * \n");
    fprintf(out, " * Configuration:\n");
    fprintf(out, " *   half_window  = %d\n", half_window);
    fprintf(out, " *   poly_order   = %d\n", poly_order);
    fprintf(out, " *   derivative   = %d\n", deriv_order);
    fprintf(out, " *   window_size  = %d\n", window_size);
    fprintf(out, " * \n");
    fprintf(out, " * Usage:\n");
    fprintf(out, " *   #include \"%s_coeffs_n%d_m%d_d%d.h\"\n",
            prefix, half_window, poly_order, deriv_order);
    fprintf(out, " *   \n");
    fprintf(out, " *   // Central region (full window available):\n");
    fprintf(out, " *   for (int j = %d; j < length - %d; j++) {\n", half_window, half_window);
    fprintf(out, " *       float sum = 0;\n");
    fprintf(out, " *       for (int k = 0; k < %d; k++) {\n", window_size);
    fprintf(out, " *           sum += %s_CENTER_WEIGHTS[k] * input[j - %d + k];\n",
            prefix, half_window);
    fprintf(out, " *       }\n");
    fprintf(out, " *       output[j] = sum;\n");
    fprintf(out, " *   }\n");
    fprintf(out, " */\n\n");
    
    /* Include guard */
    fprintf(out, "#ifndef %s\n", guard);
    fprintf(out, "#define %s\n\n", guard);
    
    /* Metadata macros */
    fprintf(out, "/* Filter configuration */\n");
    fprintf(out, "#define %s_HALF_WINDOW   %d\n", prefix, half_window);
    fprintf(out, "#define %s_POLY_ORDER    %d\n", prefix, poly_order);
    fprintf(out, "#define %s_DERIVATIVE    %d\n", prefix, deriv_order);
    fprintf(out, "#define %s_WINDOW_SIZE   %d\n\n", prefix, window_size);
    
    /* Center weights */
    fprintf(out, "/**\n");
    fprintf(out, " * Center weights for interior samples.\n");
    fprintf(out, " * Index 0 = leftmost point (i = -%d)\n", half_window);
    fprintf(out, " * Index %d = center point (i = 0)\n", half_window);
    fprintf(out, " * Index %d = rightmost point (i = +%d)\n", window_size - 1, half_window);
    fprintf(out, " */\n");
    fprintf(out, "static const float %s_CENTER_WEIGHTS[%d] = {\n", prefix, window_size);
    
    for (int i = 0; i < window_size; i++) {
        if (i % 5 == 0) fprintf(out, "    ");
        fprintf(out, "%14.10ef", center_weights[i]);
        if (i < window_size - 1) fprintf(out, ",");
        if ((i + 1) % 5 == 0 || i == window_size - 1) fprintf(out, "\n");
    }
    fprintf(out, "};\n\n");
    
    /* Edge weights */
    fprintf(out, "/**\n");
    fprintf(out, " * Edge weights for boundary samples.\n");
    fprintf(out, " * \n");
    fprintf(out, " * edge_weights[i] is used for:\n");
    fprintf(out, " *   - Leading edge: sample index i (0 to %d)\n", half_window - 1);
    fprintf(out, " *   - Trailing edge: sample index (length - 1 - i)\n");
    fprintf(out, " * \n");
    fprintf(out, " * Leading edge: convolve with data[0..%d] in REVERSE order\n", window_size - 1);
    fprintf(out, " * Trailing edge: convolve with data[length-%d..length-1] in FORWARD order\n", window_size);
    fprintf(out, " */\n");
    fprintf(out, "static const float %s_EDGE_WEIGHTS[%d][%d] = {\n",
            prefix, half_window, window_size);
    
    for (int edge = 0; edge < half_window; edge++) {
        fprintf(out, "    { /* edge %d */\n", edge);
        for (int i = 0; i < window_size; i++) {
            if (i % 5 == 0) fprintf(out, "        ");
            fprintf(out, "%14.10ef", edge_weights[edge][i]);
            if (i < window_size - 1) fprintf(out, ",");
            if ((i + 1) % 5 == 0 || i == window_size - 1) fprintf(out, "\n");
        }
        fprintf(out, "    }%s\n", edge < half_window - 1 ? "," : "");
    }
    fprintf(out, "};\n\n");
    
    /* Convenience apply function (optional) */
    fprintf(out, "/**\n");
    fprintf(out, " * Apply filter using precomputed weights (inline for performance).\n");
    fprintf(out, " * \n");
    fprintf(out, " * @param input   Input array of at least `length` elements.\n");
    fprintf(out, " * @param output  Output array of at least `length` elements.\n");
    fprintf(out, " * @param length  Number of samples (must be >= %d).\n", window_size);
    fprintf(out, " */\n");
    fprintf(out, "static inline void %s_apply(const float *input, float *output, int length)\n", prefix);
    fprintf(out, "{\n");
    fprintf(out, "    const int n = %s_HALF_WINDOW;\n", prefix);
    fprintf(out, "    const int ws = %s_WINDOW_SIZE;\n", prefix);
    fprintf(out, "    \n");
    fprintf(out, "    /* Central region */\n");
    fprintf(out, "    for (int j = n; j < length - n; j++) {\n");
    fprintf(out, "        float sum = 0.0f;\n");
    fprintf(out, "        for (int k = 0; k < ws; k++) {\n");
    fprintf(out, "            sum += %s_CENTER_WEIGHTS[k] * input[j - n + k];\n", prefix);
    fprintf(out, "        }\n");
    fprintf(out, "        output[j] = sum;\n");
    fprintf(out, "    }\n");
    fprintf(out, "    \n");
    fprintf(out, "    /* Leading edge (reversed data traversal) */\n");
    fprintf(out, "    for (int i = 0; i < n; i++) {\n");
    fprintf(out, "        float sum = 0.0f;\n");
    fprintf(out, "        for (int k = 0; k < ws; k++) {\n");
    fprintf(out, "            sum += %s_EDGE_WEIGHTS[i][k] * input[ws - 1 - k];\n", prefix);
    fprintf(out, "        }\n");
    fprintf(out, "        output[i] = sum;\n");
    fprintf(out, "    }\n");
    fprintf(out, "    \n");
    fprintf(out, "    /* Trailing edge (forward data traversal) */\n");
    fprintf(out, "    for (int i = 0; i < n; i++) {\n");
    fprintf(out, "        float sum = 0.0f;\n");
    fprintf(out, "        for (int k = 0; k < ws; k++) {\n");
    fprintf(out, "            sum += %s_EDGE_WEIGHTS[i][k] * input[length - ws + k];\n", prefix);
    fprintf(out, "        }\n");
    fprintf(out, "        output[length - 1 - i] = sum;\n");
    fprintf(out, "    }\n");
    fprintf(out, "}\n\n");
    
    /* Close guard */
    fprintf(out, "#endif /* %s */\n", guard);
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(int argc, char *argv[])
{
    int half_window = -1;
    int poly_order = -1;
    int deriv_order = 0;
    const char *output_file = NULL;
    const char *prefix = "SAVGOL";
    
    static struct option long_options[] = {
        {"half-window", required_argument, 0, 'n'},
        {"poly-order",  required_argument, 0, 'm'},
        {"derivative",  required_argument, 0, 'd'},
        {"output",      required_argument, 0, 'o'},
        {"prefix",      required_argument, 0, 'p'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "n:m:d:o:p:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'n':
                half_window = atoi(optarg);
                break;
            case 'm':
                poly_order = atoi(optarg);
                break;
            case 'd':
                deriv_order = atoi(optarg);
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'p':
                prefix = optarg;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    /* Validate required arguments */
    if (half_window < 0 || poly_order < 0) {
        fprintf(stderr, "Error: -n (half_window) and -m (poly_order) are required.\n\n");
        print_usage(argv[0]);
        return 1;
    }
    
    /* Validate ranges */
    if (half_window < 1 || half_window > MAX_HALF_WINDOW) {
        fprintf(stderr, "Error: half_window must be in [1, %d], got %d\n",
                MAX_HALF_WINDOW, half_window);
        return 1;
    }
    
    int window_size = 2 * half_window + 1;
    if (poly_order >= window_size) {
        fprintf(stderr, "Error: poly_order must be < window_size (%d), got %d\n",
                window_size, poly_order);
        return 1;
    }
    
    if (deriv_order > poly_order) {
        fprintf(stderr, "Error: derivative (%d) cannot exceed poly_order (%d)\n",
                deriv_order, poly_order);
        return 1;
    }
    
    if (deriv_order > MAX_DERIVATIVE) {
        fprintf(stderr, "Error: derivative must be <= %d, got %d\n",
                MAX_DERIVATIVE, deriv_order);
        return 1;
    }
    
    /* Open output */
    FILE *out = stdout;
    if (output_file != NULL) {
        out = fopen(output_file, "w");
        if (out == NULL) {
            fprintf(stderr, "Error: cannot open '%s' for writing\n", output_file);
            return 1;
        }
    }
    
    /* Generate header */
    generate_header(out, (uint8_t)half_window, (uint8_t)poly_order,
                    (uint8_t)deriv_order, prefix);
    
    /* Cleanup */
    if (output_file != NULL) {
        fclose(out);
        fprintf(stderr, "Generated: %s\n", output_file);
        fprintf(stderr, "  half_window = %d\n", half_window);
        fprintf(stderr, "  poly_order  = %d\n", poly_order);
        fprintf(stderr, "  derivative  = %d\n", deriv_order);
        fprintf(stderr, "  window_size = %d\n", window_size);
    }
    
    return 0;
}
