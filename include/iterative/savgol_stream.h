/**
 * @file savgol_stream.h
 * @brief Streaming (real-time) Savitzky-Golay filter API.
 * 
 * Extends the batch Savitzky-Golay filter with real-time capability.
 * Processes samples one at a time with fixed latency.
 */

#ifndef SAVGOL_STREAM_H
#define SAVGOL_STREAM_H

#include <stdbool.h>
#include <stddef.h>
#include "savgolFilter.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * TYPES
 *============================================================================*/

/**
 * @brief Streaming filter state.
 * 
 * Contains circular buffer and position tracking for real-time filtering.
 */
typedef struct SavgolStream {
    const SavgolFilter *filter;          /**< Reference to filter weights */
    float buffer[SAVGOL_MAX_WINDOW];     /**< Circular buffer for samples */
    int write_pos;                       /**< Next write position */
    size_t samples_received;             /**< Total samples pushed */
    size_t samples_output;               /**< Total samples output */
    bool owns_filter;                    /**< True if we allocated the filter */
    float dt_inv;                        /**< Cached 1/dt_scale */
} SavgolStream;

/*============================================================================
 * LIFECYCLE
 *============================================================================*/

/**
 * @brief Create a streaming filter (heap allocation).
 * 
 * @param config  Filter configuration.
 * @return Stream handle, or NULL on error.
 */
SavgolStream *savgol_stream_create(const SavgolConfig *config);

/**
 * @brief Initialize a streaming filter with user-provided storage.
 * 
 * @param stream  User-allocated SavgolStream.
 * @param filter  Pre-created filter (must outlive stream).
 * @return 0 on success, -1 on error.
 */
int savgol_stream_init(SavgolStream *stream, const SavgolFilter *filter);

/**
 * @brief Destroy a streaming filter.
 * @param stream  Stream to destroy (NULL is safe).
 */
void savgol_stream_destroy(SavgolStream *stream);

/**
 * @brief Reset stream to initial state.
 * @param stream  Stream to reset.
 */
void savgol_stream_reset(SavgolStream *stream);

/*============================================================================
 * STREAMING OPERATION
 *============================================================================*/

/**
 * @brief Push one sample (simple interface, no leading edge handling).
 * 
 * @param stream        Stream handle.
 * @param sample        Input sample.
 * @param output_valid  Set to true when return value is valid.
 * @return Filtered value (valid only when *output_valid is true).
 */
float savgol_stream_push(SavgolStream *stream, float sample, bool *output_valid);

/**
 * @brief Push sample with full edge handling.
 * 
 * @param stream       Stream handle.
 * @param sample       Input sample.
 * @param output       Output array (needs half_window+1 space).
 * @param max_outputs  Maximum outputs to write.
 * @return Number of outputs written.
 */
int savgol_stream_push_full(SavgolStream *stream, float sample,
                            float *output, int max_outputs);

/**
 * @brief Flush trailing edge samples at end of stream.
 * 
 * @param stream     Stream handle.
 * @param output     Output array.
 * @param max_count  Maximum samples to write.
 * @return Number of samples written (up to half_window).
 */
int savgol_stream_flush(SavgolStream *stream, float *output, int max_count);

/**
 * @brief Flush leading edge samples.
 * 
 * @param stream     Stream handle.
 * @param output     Output array.
 * @param max_count  Maximum samples to write.
 * @return Number of samples written (up to half_window).
 */
int savgol_stream_flush_leading(SavgolStream *stream, float *output, int max_count);

/*============================================================================
 * STATE QUERIES
 *============================================================================*/

bool   savgol_stream_ready(const SavgolStream *stream);
size_t savgol_stream_latency(const SavgolStream *stream);
size_t savgol_stream_buffered(const SavgolStream *stream);
size_t savgol_stream_samples_received(const SavgolStream *stream);
size_t savgol_stream_samples_output(const SavgolStream *stream);

#ifdef __cplusplus
}
#endif

#endif /* SAVGOL_STREAM_H */
