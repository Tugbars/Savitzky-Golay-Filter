#ifndef MES_SAVGOL_H
#define MES_SAVGOL_H

#include <stdint.h>
#include "mqs_def.h"
#include <stdlib.h> 
/*!
 * @brief Performance Trade-off in Savitzky-Golay Filter with Selective Memoization.
 *
 * Author: Tugbars Heptaskin
 * Date: 12/01/2024
 *
 * This implementation of the Savitzky-Golay filter demonstrates a significant trade-off 
 * between memory usage and CPU speed, achieved through selective memoization.
 *
 * The effect of memoization is evident when comparing the total number of function calls with 
 * and without it. For a filter with a window size of 51 and a polynomial order of 4, the total 
 * number of calls to the GramPoly function reaches 68,927 when memoization is not active 
 * (MAX_ENTRIES set to 1). However, enabling memoization with MAX_ENTRIES set to 255 drastically 
 * reduces this number to just 1,326 calls.
 *
 * This reduction in function calls translates to a significant decrease in CPU load and 
 * computational time, showcasing the efficiency gains from memoization. However, it's important 
 * to note that memoization requires additional memory for storing precomputed values. This 
 * presents a trade-off: users can optimize the code based on their specific constraints and 
 * requirements, balancing between memory availability and CPU speed.
 *
 * The flexibility to adjust MAX_ENTRIES allows users to tailor the filter's performance to 
 * their system's capabilities, making this implementation versatile for a wide range of 
 * applications, from resource-constrained embedded systems to more robust computing environments.
 *
 */

typedef struct {
    uint8_t halfWindowSize;      // was m
    uint16_t targetPoint;        // was t
    uint8_t polynomialOrder;     // was n
    uint8_t derivativeOrder;     // was s
    double time_step;         
    uint8_t derivation_order;    // Adjusting to match the derivativeOrder type
} SavitzkyGolayFilterConfig;

typedef struct GramPolyKey {
    int dataIndex;
    uint8_t polynomialOrder;
    uint8_t derivativeOrder;
    uint8_t caseType; // 0 for central, 1 for border
} GramPolyKey;

typedef struct HashMapEntry {
    GramPolyKey key;
    double value;
    int isOccupied;
} HashMapEntry;

typedef struct {
    SavitzkyGolayFilterConfig conf;
    double* weights;
    double dt;
} SavitzkyGolayFilter;

SavitzkyGolayFilter* SavitzkyGolayFilter_init(SavitzkyGolayFilterConfig conf);
void mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize, MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder, uint8_t targetPoint, uint8_t derivativeOrder);

#endif // MES_SAVGOL_H
