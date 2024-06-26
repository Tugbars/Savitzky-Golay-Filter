# C implementation of Least-Squares Smoothing and Differentiation by the Convolution (Savitzky-Golay) Method

**Author:** Tugbars Heptaskin  
**Date:** 15/01/2024  
**Source:** The GramPoly function and the general concept were adapted from arntanguy's implementation: [arntanguy/gram_savitzky_golay](https://github.com/arntanguy/gram_savitzky_golay/tree/master)

## Overview
This implementation optimizes the traditional Savitzky-Golay filter, utilized for smoothing and differentiating data. Key improvements include global variables to reduce stack footprint and memoization for computational efficiency. The implementation to handle both central and border cases, optimization for stack footprint, memoization strategy, and overall documentation were contributed by the author.

## Core Functionality

- **GramPoly Function:** 
  - The GramPoly function in the code calculates Gram polynomials or their derivatives, which are crucial for determining the coefficients of the least squares fitting polynomial in the Savitzky-Golay filter. The function ensures the polynomial basis functions are orthogonal, meaning each order of the polynomial is independent of the others. This orthogonality 
  leads to more stable and meaningful results, especially when dealing with noisy data.
  - The function employs a recursive approach, where higher-order polynomials are generated from base cases using a defined recursive formula. To optimize efficiency, the function incorporates memoization, storing and reusing previously computed values to minimize redundant calculations and enhance performance. This combination of recursion and memoization ensures 
  the calculations are both efficient and robust, handling repeated computations effectively.

- **Filter Application:**
  - Applies the Savitzky-Golay filter to data arrays, smoothly handling both central data points and border cases.
  - In border cases, specifically computes weights for each scenario, ensuring accurate processing at the boundaries of the dataset.
  - Validated to match the output of Matlab's Savitzky-Golay filter, ensuring reliability and accuracy in various applications.
  - Adaptable to function as a causal filter for real-time filtering applications. In this mode, the filter uses only past and present data, making it suitable for on-the-fly data processing.

## Optimizations
- **Minimized Stack Footprint:** Uses global variables for critical parameters, reducing the risk of stack overflows in large datasets or high polynomial orders. I also added the version with no optimization for stack footprint minimization and no memoization.
- **Memoization for Computational Efficiency:**
  - Demonstrates a significant balance between memory usage and CPU speed.
  - With memoization, function calls for a filter with a window size of 51 and a polynomial order of 4 are reduced from 68,927 to just 1,326.
  - This reduction lowers CPU load and computational time, making the filter more efficient.
  - The `MAX_ENTRIES` parameter allows flexibility in tuning the balance between memory usage and CPU speed, making this implementation adaptable for diverse system capabilities.

## Suitability
Ideal for data analysis, signal processing, and similar fields where effective data smoothing and differentiation are crucial, especially in resource-constrained embedded environments.

This new section provides clear guidance on configuring: 

### Configuring Filter for Past Values
To make the filter work for past values, you can adjust the `targetPoint` parameter in the `initFilter` function:

- **targetPoint = 0:** The filter smoothes data based on both future and past values. This setting is more suited for non-real-time applications where all data points are available.
- **targetPoint = halfWindowSize:** The filter smoothes data based on only the present and past data, making it suitable for real-time applications or causal filtering.

- **Non-Real-Time Filtering (`ApplyFilter`):**
  - The `ApplyFilter` function is designed to smoothen data by considering both past and future data points. This configuration is akin to an Infinite Impulse Response (IIR) filter, making it suitable for scenarios where the future values are available for analysis.
  
- **Real-Time Filtering (`ApplyFilterAtAPoint`):**
  - For real-time applications, an alternative function, `ApplyFilterAtAPoint`, was conceptualized. This function demonstrates filtering using only past and present data, aligning with the requirements of real-time data processing.
  - Please note that `ApplyFilterAtAPoint` is a preliminary implementation, intended to illustrate the approach for real-time filtering. It is not fully developed or tested. Developers are encouraged to refine and adapt this function according to their specific real-time processing needs.

## Testing the Code

```c
#include "mes_savgol.h"

int main() {
    double dataset[] = { /*... your data ...*/ };
    size_t dataSize = sizeof(dataset)/sizeof(dataset[0]);
    
    MqsRawDataPoint_t rawData[dataSize]; //insert your own dataSize here. 
    for (int i = 0; i < dataSize; ++i) {
        rawData[i].phaseAngle = dataset[i];
        rawData[i].impedance = 0.0;  // You can set the impedance to a default value
    }
    MqsRawDataPoint_t filteredData[dataSize] = {0.0};

    mes_SavgolFilter(rawData, dataSize, filteredData);
    
    // Output or use the filtered data
    printf("yourSavgolData = [");
    for(int i = 0; i < dataSize; ++i) {
        if(i == dataSize - 1){
            printf("%f", filteredData[i].phaseAngle);
        }else{
            printf("%f, ", filteredData[i].phaseAngle);
        }
    }
    printf("];\n");
    
    cleanupFilterInstance();

    // optional: you may like to link these variables from mes_savgol.c for efficiency min maxing purposes. 
    // printf("GramPoly was called %d times.\n", gramPolyCallCount);
    // printf("total map entries %d times.\n", totalHashMapEntries);
    return 0;
}

```
Increasing MAX_ENTRIES to a higher value and monitoring the output of totalHashMapEntries can help determine the optimal number of entries needed for memoization, thereby minimizing CPU load. 

