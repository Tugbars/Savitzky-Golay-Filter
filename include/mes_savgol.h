#ifndef MES_SAVGOL_H
#define MES_SAVGOL_H

#include "mqs_def.h"
#include <stdlib.h>
#include <stdbool.h>  // Include for boolean type

//#define HALF_WINDOW_SIZE 13
//#define POLYNOMIAL_ORDER 4
//#define CALC_MEMO_ENTRIES(halfWindowSize, polynomialOrder) (((2 * (halfWindowSize) + 1) * ((polynomialOrder) + 1))) //not accurate, ignores edge Cases

#define MAX_ENTRIES 138

typedef struct {
    uint16_t halfWindowSize;
    uint16_t targetPoint;
    uint8_t polynomialOrder;
    uint8_t derivativeOrder;
    float time_step;
    uint8_t derivation_order;
} SavitzkyGolayFilterConfig;

typedef struct {
    uint32_t key;
    float value;
    bool isOccupied;
} HashMapEntry;

typedef struct {
    HashMapEntry memoizationTable[MAX_ENTRIES];
    uint16_t totalHashMapEntries;
} MemoizationContext;

typedef struct {
    SavitzkyGolayFilterConfig conf;
    float dt;
} SavitzkyGolayFilter;

SavitzkyGolayFilter SavitzkyGolayFilter_init(SavitzkyGolayFilterConfig conf);
void mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize, MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder, uint8_t targetPoint, uint8_t derivativeOrder);

#endif // MES_SAVGOL_H
