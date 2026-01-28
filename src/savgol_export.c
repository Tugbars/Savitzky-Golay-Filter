/**
 * @file savgol_export.c
 * @brief Tool to export Savitzky-Golay filter coefficients as C headers.
 * 
 * Generates precomputed weight tables for embedded systems, FPGAs, or any
 * environment where runtime weight computation is too expensive.
 * 
 * Build (via CMake):
 *   Links against savgolFilter library
 * 
 * Build (standalone):
 *   gcc -O2 -o savgol_export savgol_export.c savgolFilter.c -lm
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
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <getopt.h>

#include "savgolFilter.h"

/*============================================================================
 * COMMAND LINE PARSING
 *============================================================================*/

typedef struct {
    int half_window;
    int poly_order;
    int derivative;
    const char *output_file;
    const char *prefix;
    int show_help;
} ExportOptions;

static void print_usage(const char *prog)
{
    fprintf(stderr, "Usage: %s -n <half_window> -m <poly_order> [options]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -n, --half-window=N   Half-window size (1-%d)\n", SAVGOL_MAX_HALF_WINDOW);
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

static int parse_args(int argc, char *argv[], ExportOptions *opts)
{
    static struct option long_options[] = {
        {"half-window", required_argument, 0, 'n'},
        {"poly-order",  required_argument, 0, 'm'},
        {"derivative",  required_argument, 0, 'd'},
        {"output",      required_argument, 0, 'o'},
        {"prefix",      required_argument, 0, 'p'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    /* Defaults */
    opts->half_window = -1;
    opts->poly_order = -1;
    opts->derivative = 0;
    opts->output_file = NULL;
    opts->prefix = "SAVGOL";
    opts->show_help = 0;
    
    int opt;
    while ((opt = getopt_long(argc, argv, "n:m:d:o:p:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'n':
                opts->half_window = atoi(optarg);
                break;
            case 'm':
                opts->poly_order = atoi(optarg);
                break;
            case 'd':
                opts->derivative = atoi(optarg);
                break;
            case 'o':
                opts->output_file = optarg;
                break;
            case 'p':
                opts->prefix = optarg;
                break;
            case 'h':
                opts->show_help = 1;
                return 0;
            default:
                return -1;
        }
    }
    
    /* Validate required arguments */
    if (opts->half_window < 0) {
        fprintf(stderr, "Error: --half-window (-n) is required\n");
        return -1;
    }
    if (opts->poly_order < 0) {
        fprintf(stderr, "Error: --poly-order (-m) is required\n");
        return -1;
    }
    
    return 0;
}

/*============================================================================
 * HEADER GENERATION
 *============================================================================*/

/**
 * @brief Convert string to uppercase for header guard.
 */
static void str_to_upper(char *dst, const char *src, size_t max_len)
{
    size_t i;
    for (i = 0; i < max_len - 1 && src[i]; i++) {
        dst[i] = (char)toupper((unsigned char)src[i]);
    }
    dst[i] = '\0';
}

/**
 * @brief Generate the C header file with precomputed coefficients.
 */
static int generate_header(FILE *out, const SavgolFilter *filter, const ExportOptions *opts)
{
    const int n = filter->config.half_window;
    const int m = filter->config.poly_order;
    const int d = filter->config.derivative;
    const int window_size = filter->window_size;
    
    /* Get current time for header comment */
    time_t now = time(NULL);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));
    
    /* Create uppercase prefix for macros */
    char prefix_upper[64];
    str_to_upper(prefix_upper, opts->prefix, sizeof(prefix_upper));
    
    /* Generate header guard name */
    char guard_name[128];
    snprintf(guard_name, sizeof(guard_name), "%s_COEFFS_N%d_M%d_D%d_H", 
             prefix_upper, n, m, d);
    
    /*--- File header ---*/
    fprintf(out, "/**\n");
    fprintf(out, " * @file %s_coeffs_n%d_m%d_d%d.h\n", opts->prefix, n, m, d);
    fprintf(out, " * @brief Precomputed Savitzky-Golay filter coefficients.\n");
    fprintf(out, " * \n");
    fprintf(out, " * AUTO-GENERATED FILE - DO NOT EDIT\n");
    fprintf(out, " * Generated by savgol_export on %s\n", time_str);
    fprintf(out, " * \n");
    fprintf(out, " * Configuration:\n");
    fprintf(out, " *   half_window  = %d\n", n);
    fprintf(out, " *   poly_order   = %d\n", m);
    fprintf(out, " *   derivative   = %d\n", d);
    fprintf(out, " *   window_size  = %d\n", window_size);
    fprintf(out, " * \n");
    fprintf(out, " * Usage:\n");
    fprintf(out, " *   #include \"%s_coeffs_n%d_m%d_d%d.h\"\n", opts->prefix, n, m, d);
    fprintf(out, " *   \n");
    fprintf(out, " *   // Apply to center samples (index n to length-n-1)\n");
    fprintf(out, " *   for (int i = %d; i < length - %d; i++) {\n", n, n);
    fprintf(out, " *       float sum = 0;\n");
    fprintf(out, " *       for (int k = 0; k < %d; k++) {\n", window_size);
    fprintf(out, " *           sum += %s_CENTER_WEIGHTS[k] * data[i - %d + k];\n", prefix_upper, n);
    fprintf(out, " *       }\n");
    fprintf(out, " *       output[i] = sum;\n");
    fprintf(out, " *   }\n");
    fprintf(out, " */\n\n");
    
    /*--- Header guard ---*/
    fprintf(out, "#ifndef %s\n", guard_name);
    fprintf(out, "#define %s\n\n", guard_name);
    
    /*--- Configuration macros ---*/
    fprintf(out, "/* Filter Configuration */\n");
    fprintf(out, "#define %s_HALF_WINDOW   %d\n", prefix_upper, n);
    fprintf(out, "#define %s_POLY_ORDER    %d\n", prefix_upper, m);
    fprintf(out, "#define %s_DERIVATIVE    %d\n", prefix_upper, d);
    fprintf(out, "#define %s_WINDOW_SIZE   %d\n\n", prefix_upper, window_size);
    
    /*--- Center weights ---*/
    fprintf(out, "/* Center Weights (for interior samples) */\n");
    fprintf(out, "static const float %s_CENTER_WEIGHTS[%d] = {\n", prefix_upper, window_size);
    
    for (int i = 0; i < window_size; i++) {
        if (i % 4 == 0) fprintf(out, "    ");
        fprintf(out, "%+.10ef", filter->center_weights[i]);
        if (i < window_size - 1) fprintf(out, ",");
        if (i % 4 == 3 || i == window_size - 1) fprintf(out, "\n");
        else fprintf(out, " ");
    }
    fprintf(out, "};\n\n");
    
    /*--- Edge weights ---*/
    fprintf(out, "/* Edge Weights [edge_index][window_index] */\n");
    fprintf(out, "/*\n");
    fprintf(out, " * edge_index 0: for first and last sample\n");
    fprintf(out, " * edge_index %d: for sample at index %d and length-%d-1\n", n-1, n-1, n);
    fprintf(out, " * \n");
    fprintf(out, " * Leading edge (i < %d):  convolve with data[0..%d] in REVERSE\n", n, window_size-1);
    fprintf(out, " * Trailing edge (i >= length-%d): convolve with data[length-%d..length-1] FORWARD\n", n, window_size);
    fprintf(out, " */\n");
    fprintf(out, "static const float %s_EDGE_WEIGHTS[%d][%d] = {\n", prefix_upper, n, window_size);
    
    for (int edge = 0; edge < n; edge++) {
        fprintf(out, "    { /* edge %d */\n", edge);
        for (int i = 0; i < window_size; i++) {
            if (i % 4 == 0) fprintf(out, "        ");
            fprintf(out, "%+.10ef", filter->edge_weights[edge][i]);
            if (i < window_size - 1) fprintf(out, ",");
            if (i % 4 == 3 || i == window_size - 1) fprintf(out, "\n");
            else fprintf(out, " ");
        }
        fprintf(out, "    }%s\n", edge < n - 1 ? "," : "");
    }
    fprintf(out, "};\n\n");
    
    /*--- Inline apply function ---*/
    fprintf(out, "/* Inline apply function for convenience */\n");
    fprintf(out, "static inline void %s_apply(const float *input, float *output, int length)\n", prefix_upper);
    fprintf(out, "{\n");
    fprintf(out, "    const int n = %s_HALF_WINDOW;\n", prefix_upper);
    fprintf(out, "    const int ws = %s_WINDOW_SIZE;\n", prefix_upper);
    fprintf(out, "    int i, k;\n");
    fprintf(out, "    float sum;\n");
    fprintf(out, "    \n");
    fprintf(out, "    /* Leading edge */\n");
    fprintf(out, "    for (i = 0; i < n; i++) {\n");
    fprintf(out, "        sum = 0.0f;\n");
    fprintf(out, "        for (k = 0; k < ws; k++) {\n");
    fprintf(out, "            sum += %s_EDGE_WEIGHTS[i][k] * input[ws - 1 - k];\n", prefix_upper);
    fprintf(out, "        }\n");
    fprintf(out, "        output[i] = sum;\n");
    fprintf(out, "    }\n");
    fprintf(out, "    \n");
    fprintf(out, "    /* Center region */\n");
    fprintf(out, "    for (i = n; i < length - n; i++) {\n");
    fprintf(out, "        sum = 0.0f;\n");
    fprintf(out, "        for (k = 0; k < ws; k++) {\n");
    fprintf(out, "            sum += %s_CENTER_WEIGHTS[k] * input[i - n + k];\n", prefix_upper);
    fprintf(out, "        }\n");
    fprintf(out, "        output[i] = sum;\n");
    fprintf(out, "    }\n");
    fprintf(out, "    \n");
    fprintf(out, "    /* Trailing edge */\n");
    fprintf(out, "    for (i = 0; i < n; i++) {\n");
    fprintf(out, "        sum = 0.0f;\n");
    fprintf(out, "        for (k = 0; k < ws; k++) {\n");
    fprintf(out, "            sum += %s_EDGE_WEIGHTS[i][k] * input[length - ws + k];\n", prefix_upper);
    fprintf(out, "        }\n");
    fprintf(out, "        output[length - 1 - i] = sum;\n");
    fprintf(out, "    }\n");
    fprintf(out, "}\n\n");
    
    /*--- Footer ---*/
    fprintf(out, "#endif /* %s */\n", guard_name);
    
    return 0;
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(int argc, char *argv[])
{
    ExportOptions opts;
    
    if (parse_args(argc, argv, &opts) != 0) {
        print_usage(argv[0]);
        return 1;
    }
    
    if (opts.show_help) {
        print_usage(argv[0]);
        return 0;
    }
    
    /* Create filter configuration */
    SavgolConfig config = {
        .half_window = (uint8_t)opts.half_window,
        .poly_order = (uint8_t)opts.poly_order,
        .derivative = (uint8_t)opts.derivative,
        .time_step = 1.0f,
        .boundary = SAVGOL_BOUNDARY_POLYNOMIAL
    };
    
    /* Create filter (computes all weights) */
    SavgolFilter *filter = savgol_create(&config);
    if (filter == NULL) {
        fprintf(stderr, "Error: Failed to create filter with given parameters\n");
        return 1;
    }
    
    /* Open output file or use stdout */
    FILE *out = stdout;
    if (opts.output_file != NULL) {
        out = fopen(opts.output_file, "w");
        if (out == NULL) {
            fprintf(stderr, "Error: Cannot open output file '%s'\n", opts.output_file);
            savgol_destroy(filter);
            return 1;
        }
    }
    
    /* Generate header */
    int result = generate_header(out, filter, &opts);
    
    /* Cleanup */
    if (opts.output_file != NULL) {
        fclose(out);
        if (result == 0) {
            fprintf(stderr, "Generated: %s\n", opts.output_file);
            fprintf(stderr, "  half_window = %d\n", opts.half_window);
            fprintf(stderr, "  poly_order  = %d\n", opts.poly_order);
            fprintf(stderr, "  derivative  = %d\n", opts.derivative);
            fprintf(stderr, "  window_size = %d\n", filter->window_size);
        }
    }
    
    savgol_destroy(filter);
    
    return result;
}