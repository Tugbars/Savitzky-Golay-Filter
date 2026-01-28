/**
 * @file savgol_stream.c
 * @brief Streaming (real-time) Savitzky-Golay filter implementation.
 * 
 * Processes samples one at a time using a circular buffer. Outputs filtered
 * values with a fixed latency of `half_window` samples.
 * 
 * This module links against the savgolFilter library.
 * 
 * @author Tugbars Heptaskin
 */

#include <stdlib.h>
#include <string.h>

#include "savgol_stream.h"

/*============================================================================
 * INTERNAL: CIRCULAR BUFFER CONVOLUTION
 *============================================================================*/

/**
 * @brief Apply center weights to circular buffer.
 */
static inline float convolve_center_circular(const SavgolStream *stream)
{
    const SavgolFilter *f = stream->filter;
    const int ws = f->window_size;
    const int start = stream->write_pos;  /* Oldest sample position */
    
    float sum = 0.0f;
    for (int i = 0; i < ws; i++) {
        int buf_idx = (start + i) % ws;
        sum += f->center_weights[i] * stream->buffer[buf_idx];
    }
    
    return sum;
}

/**
 * @brief Apply edge weights for trailing edge (forward traversal).
 */
static inline float convolve_edge_trailing(const SavgolStream *stream, int edge_index)
{
    const SavgolFilter *f = stream->filter;
    const int ws = f->window_size;
    const int start = stream->write_pos;
    
    float sum = 0.0f;
    for (int i = 0; i < ws; i++) {
        int buf_idx = (start + i) % ws;
        sum += f->edge_weights[edge_index][i] * stream->buffer[buf_idx];
    }
    
    return sum;
}

/**
 * @brief Apply edge weights for leading edge (reversed traversal).
 */
static inline float convolve_edge_leading(const SavgolStream *stream, int edge_index)
{
    const SavgolFilter *f = stream->filter;
    const int ws = f->window_size;
    const int start = stream->write_pos;
    
    float sum = 0.0f;
    for (int i = 0; i < ws; i++) {
        int buf_idx = (start + ws - 1 - i) % ws;  /* Reverse */
        sum += f->edge_weights[edge_index][i] * stream->buffer[buf_idx];
    }
    
    return sum;
}

/*============================================================================
 * PUBLIC API: LIFECYCLE
 *============================================================================*/

SavgolStream *savgol_stream_create(const SavgolConfig *config)
{
    if (config == NULL) {
        return NULL;
    }
    
    /* Create underlying filter */
    SavgolFilter *filter = savgol_create(config);
    if (filter == NULL) {
        return NULL;
    }
    
    /* Allocate stream struct */
    SavgolStream *stream = (SavgolStream *)malloc(sizeof(SavgolStream));
    if (stream == NULL) {
        savgol_destroy(filter);
        return NULL;
    }
    
    /* Initialize */
    stream->filter = filter;
    stream->owns_filter = true;
    stream->dt_inv = (filter->dt_scale != 0.0f) ? (1.0f / filter->dt_scale) : 1.0f;
    savgol_stream_reset(stream);
    
    return stream;
}

int savgol_stream_init(SavgolStream *stream, const SavgolFilter *filter)
{
    if (stream == NULL || filter == NULL) {
        return -1;
    }
    
    stream->filter = filter;
    stream->owns_filter = false;
    stream->dt_inv = (filter->dt_scale != 0.0f) ? (1.0f / filter->dt_scale) : 1.0f;
    savgol_stream_reset(stream);
    
    return 0;
}

void savgol_stream_destroy(SavgolStream *stream)
{
    if (stream == NULL) {
        return;
    }
    
    if (stream->owns_filter && stream->filter != NULL) {
        savgol_destroy((SavgolFilter *)stream->filter);
    }
    
    free(stream);
}

void savgol_stream_reset(SavgolStream *stream)
{
    if (stream == NULL) {
        return;
    }
    
    stream->write_pos = 0;
    stream->samples_received = 0;
    stream->samples_output = 0;
    
    memset(stream->buffer, 0, sizeof(stream->buffer));
}

/*============================================================================
 * PUBLIC API: STREAMING OPERATION
 *============================================================================*/

float savgol_stream_push(SavgolStream *stream, float sample, bool *output_valid)
{
    if (stream == NULL || stream->filter == NULL) {
        if (output_valid) *output_valid = false;
        return 0.0f;
    }
    
    const int ws = stream->filter->window_size;
    
    /* Write sample to circular buffer */
    stream->buffer[stream->write_pos] = sample;
    stream->write_pos = (stream->write_pos + 1) % ws;
    stream->samples_received++;
    
    /* Check if buffer is full */
    if (stream->samples_received < (size_t)ws) {
        if (output_valid) *output_valid = false;
        return 0.0f;
    }
    
    /* Buffer full — compute filtered output */
    float result = convolve_center_circular(stream) * stream->dt_inv;
    stream->samples_output++;
    
    if (output_valid) *output_valid = true;
    return result;
}

int savgol_stream_push_full(SavgolStream *stream, float sample,
                            float *output, int max_outputs)
{
    if (stream == NULL || stream->filter == NULL || output == NULL || max_outputs <= 0) {
        return 0;
    }
    
    const SavgolFilter *f = stream->filter;
    const int ws = f->window_size;
    const int n = f->config.half_window;
    
    /* Track if this push will fill the buffer */
    bool was_filling = (stream->samples_received < (size_t)ws);
    
    /* Write sample to circular buffer */
    stream->buffer[stream->write_pos] = sample;
    stream->write_pos = (stream->write_pos + 1) % ws;
    stream->samples_received++;
    
    /* Still filling? */
    if (stream->samples_received < (size_t)ws) {
        return 0;
    }
    
    /* Buffer just became full? Output leading edge + first center */
    if (was_filling) {
        int count = 0;
        
        /* Leading edge samples */
        for (int i = 0; i < n && count < max_outputs; i++) {
            output[count++] = convolve_edge_leading(stream, i) * stream->dt_inv;
            stream->samples_output++;
        }
        
        /* First center sample */
        if (count < max_outputs) {
            output[count++] = convolve_center_circular(stream) * stream->dt_inv;
            stream->samples_output++;
        }
        
        return count;
    }
    
    /* Normal: one center output */
    output[0] = convolve_center_circular(stream) * stream->dt_inv;
    stream->samples_output++;
    return 1;
}

int savgol_stream_flush(SavgolStream *stream, float *output, int max_count)
{
    if (stream == NULL || output == NULL || max_count <= 0) {
        return -1;
    }
    
    const SavgolFilter *f = stream->filter;
    const int n = f->config.half_window;
    
    /* Can only flush if buffer was filled */
    if (stream->samples_received < (size_t)f->window_size) {
        return 0;
    }
    
    int count = (max_count < n) ? max_count : n;
    
    for (int i = 0; i < count; i++) {
        int edge_index = n - 1 - i;
        output[i] = convolve_edge_trailing(stream, edge_index) * stream->dt_inv;
        stream->samples_output++;
    }
    
    return count;
}

int savgol_stream_flush_leading(SavgolStream *stream, float *output, int max_count)
{
    if (stream == NULL || output == NULL || max_count <= 0) {
        return 0;
    }
    
    const SavgolFilter *f = stream->filter;
    const int n = f->config.half_window;
    
    if (stream->samples_received < (size_t)f->window_size) {
        return 0;
    }
    
    int count = (max_count < n) ? max_count : n;
    
    for (int i = 0; i < count; i++) {
        output[i] = convolve_edge_leading(stream, i) * stream->dt_inv;
        stream->samples_output++;
    }
    
    return count;
}

/*============================================================================
 * PUBLIC API: STATE QUERIES
 *============================================================================*/

bool savgol_stream_ready(const SavgolStream *stream)
{
    if (stream == NULL || stream->filter == NULL) {
        return false;
    }
    return stream->samples_received >= (size_t)stream->filter->window_size;
}

size_t savgol_stream_latency(const SavgolStream *stream)
{
    if (stream == NULL || stream->filter == NULL) {
        return 0;
    }
    return stream->filter->config.half_window;
}

size_t savgol_stream_buffered(const SavgolStream *stream)
{
    if (stream == NULL || stream->filter == NULL) {
        return 0;
    }
    
    size_t ws = (size_t)stream->filter->window_size;
    return (stream->samples_received < ws) ? stream->samples_received : ws;
}

size_t savgol_stream_samples_received(const SavgolStream *stream)
{
    return (stream != NULL) ? stream->samples_received : 0;
}

size_t savgol_stream_samples_output(const SavgolStream *stream)
{
    return (stream != NULL) ? stream->samples_output : 0;
}
