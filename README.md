# C implementation of Least-Squares Smoothing and Differentiation by the Convolution (Savitzky-Golay) Method

**Author:** Tugbars Heptaskin  
**Date:** 02/01/2025

## Overview
This implementation optimizes the traditional Savitzky-Golay filter, utilized for smoothing and differentiating data. Key improvements include global variables to reduce stack footprint and memoization for computational efficiency. The implementation to handle both central and border cases, optimization for stack footprint, memoization strategy, and overall documentation were contributed by the author.

## Core Functionality

- **Gram Polynomial Evaluation:** 
  - Iterative vs. Recursive Approach:
  Recursive Approach:
  The original implementation computed Gram polynomials using recursion. Although the recursive method directly mirrored the mathematical definition, it incurred significant overhead due to multiple function calls and a larger stack footprint.
  Iterative Approach:
  - The current implementation uses dynamic programming to compute Gram polynomials iteratively. This approach reduces the function call overhead and improves speed—benchmarks indicate the iterative method is roughly four times faster than the recursive version.
  - Memoization:
  An optional memoization layer (enabled with the ENABLE_MEMOIZATION preprocessor guard) caches computed values of the Gram polynomials for given indices, polynomial orders, and derivative orders. With memoization, the number of repeated calculations is drastically reduced. For example, with a filter window of size 51 and a polynomial order of 4, the number of Gram polynomial evaluations can drop from 
  68,927 to just 1,326.

- **GramPoly Function:** 
  - The GramPoly function in the code calculates Gram polynomials or their derivatives, which are crucial for determining the coefficients of the least squares fitting polynomial in the Savitzky-Golay filter. The function ensures the polynomial basis functions are orthogonal, meaning each order of the polynomial is independent of the others. This orthogonality 
  leads to more stable and meaningful results, especially when dealing with noisy data.
  - The function employs a recursive approach, where higher-order polynomials are generated from base cases using a defined recursive formula. To optimize efficiency, the function incorporates memoization, storing and reusing previously computed values to minimize redundant calculations and enhance performance. 

- **Filter Application:**
  - Applies the Savitzky-Golay filter to data arrays, smoothly handling both central data points and border cases.
  - In border cases, specifically computes weights for each scenario, ensuring accurate processing at the boundaries of the dataset.
  - Validated to **match the output of Matlab's Savitzky-Golay filter**.
  - Adaptable to function as a causal filter for real-time filtering applications. In this mode, the filter uses only past and present data, making it suitable for on-the-fly data processing.

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
     size_t dataSize = sizeof(dataset) / sizeof(dataset[0]);
    
    // Allocate arrays for the raw and filtered data.
    MqsRawDataPoint_t rawData[dataSize];
    MqsRawDataPoint_t filteredData[dataSize];
    for (size_t i = 0; i < dataSize; ++i) {
        rawData[i].phaseAngle = dataset[i];
        filteredData[i].phaseAngle = 0.0f;
    }

    // Set filter parameters.
    uint8_t halfWindowSize = 12;
    uint8_t polynomialOrder = 4;
    uint8_t targetPoint = 0;
    uint8_t derivativeOrder = 0;
  
    clock_t tic = clock();
    mes_savgolFilter(rawData, dataSize, halfWindowSize, filteredData, polynomialOrder, targetPoint, derivativeOrder);
    clock_t toc = clock();

    printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
    printData(filteredData, dataSize);
    return 0;
}

```
Increasing MAX_ENTRIES to a higher value and monitoring the output of totalHashMapEntries can help determine the optimal number of entries needed for memoization, thereby minimizing CPU load. 

