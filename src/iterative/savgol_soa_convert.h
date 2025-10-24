/**
 * @file savgol_soa_convert.h
 * @brief SoA conversion functions for Savitzky-Golay filter
 * 
 * @details
 * Boundary operations to convert between AoS (struct with phaseAngle field)
 * and SoA (pure float array) format.
 * 
 * These operations are done ONCE per filter execution:
 * - Extract: MqsRawDataPoint_t[] → float[]  (input boundary)
 * - Write back: float[] → MqsRawDataPoint_t[]  (output boundary)
 * 
 * Similar to FFT split/join operations in the radix-4 architecture.
 * 
 * @author Tugbars Heptaskin
 * @date 2025-10-24
 */

#ifndef SAVGOL_SOA_CONVERT_H
#define SAVGOL_SOA_CONVERT_H

#include "savgolFilter.h"
#include "savgol_simd_ops.h"
#include <string.h>

//==============================================================================
// AoS to SoA Conversion: Extract phaseAngle Field
//==============================================================================

/**
 * @brief Extract phaseAngle field from struct array (AoS → SoA)
 * 
 * This is the "split" operation - done ONCE at input boundary.
 * For large arrays, uses streaming stores to avoid cache pollution.
 * 
 * @param aos_data Input array of structs
 * @param soa_data Output float array (aligned)
 * @param count Number of elements
 */
static inline void savgol_aos_to_soa(
    const MqsRawDataPoint_t *aos_data,
    float *soa_data,
    size_t count)
{
    // For very large arrays, use streaming stores
    if (count >= SAVGOL_STREAM_THRESHOLD) {
#ifdef HAS_AVX512
        size_t i = 0;
        // Process 16 elements at a time
        for (; i + 16 <= count; i += 16) {
            // Gather from struct array (strided load)
            __m512 v = _mm512_set_ps(
                aos_data[i+15].phaseAngle, aos_data[i+14].phaseAngle,
                aos_data[i+13].phaseAngle, aos_data[i+12].phaseAngle,
                aos_data[i+11].phaseAngle, aos_data[i+10].phaseAngle,
                aos_data[i+9].phaseAngle,  aos_data[i+8].phaseAngle,
                aos_data[i+7].phaseAngle,  aos_data[i+6].phaseAngle,
                aos_data[i+5].phaseAngle,  aos_data[i+4].phaseAngle,
                aos_data[i+3].phaseAngle,  aos_data[i+2].phaseAngle,
                aos_data[i+1].phaseAngle,  aos_data[i].phaseAngle
            );
            // Streaming store (bypass cache)
            VSTREAM_PS_512(&soa_data[i], v);
        }
        // Handle remainder
        for (; i < count; i++) {
            soa_data[i] = aos_data[i].phaseAngle;
        }
        
#elif defined(HAS_AVX2)
        size_t i = 0;
        // Process 8 elements at a time
        for (; i + 8 <= count; i += 8) {
            __m256 v = _mm256_set_ps(
                aos_data[i+7].phaseAngle, aos_data[i+6].phaseAngle,
                aos_data[i+5].phaseAngle, aos_data[i+4].phaseAngle,
                aos_data[i+3].phaseAngle, aos_data[i+2].phaseAngle,
                aos_data[i+1].phaseAngle, aos_data[i].phaseAngle
            );
            VSTREAM_PS_256(&soa_data[i], v);
        }
        for (; i < count; i++) {
            soa_data[i] = aos_data[i].phaseAngle;
        }
        
#else
        // Fallback: simple loop
        for (size_t i = 0; i < count; i++) {
            soa_data[i] = aos_data[i].phaseAngle;
        }
#endif
    } else {
        // Small arrays: simple loop (compiler will auto-vectorize)
        for (size_t i = 0; i < count; i++) {
            soa_data[i] = aos_data[i].phaseAngle;
        }
    }
}

//==============================================================================
// SoA to AoS Conversion: Write Back Results
//==============================================================================

/**
 * @brief Write filtered results back to struct array (SoA → AoS)
 * 
 * This is the "join" operation - done ONCE at output boundary.
 * Usually not performance-critical since it's a simple write.
 * 
 * @param soa_data Input float array (filtered results)
 * @param aos_data Output array of structs
 * @param count Number of elements
 */
static inline void savgol_soa_to_aos(
    const float *soa_data,
    MqsRawDataPoint_t *aos_data,
    size_t count)
{
    // Simple write-back
    // Note: We could vectorize this with scatter operations in AVX-512,
    // but it's typically not the bottleneck
    for (size_t i = 0; i < count; i++) {
        aos_data[i].phaseAngle = soa_data[i];
    }
}

#endif // SAVGOL_SOA_CONVERT_H