/*!
 * Optimized Savitzky-Golay Filter Implementation with Recursive Calculation and Memoization.
 *
 * Author: Tugbars Heptaskin
 * Date: 06/18/2024
 * Company: Aminic Aps
 *
 * This implementation provides an efficient and optimized version of the Savitzky-Golay filter,
 * commonly used for smoothing and differentiating data. The key enhancements in this implementation
 * include the use of global variables to minimize stack footprint and memoization to reduce
 * redundant recursive computations.
 *
 */

#include <stdio.h>
#include <math.h>
#include "mes_savgol.h"
#include <stdbool.h>

// global values to minimize the stack footprint of GramPoly
int g_dataIndex;
uint8_t g_halfWindowSize;
uint8_t g_targetPoint = 0;
uint8_t g_derivativeOrder;
// notifies how many hash map entries the function has registered. 
int totalHashMapEntries = 0;

#define MAX_POLYNOMIAL_ORDER 3
#define MAX_ENTRIES 100 // for computing weights for each border case. 
HashMapEntry memoizationTable[MAX_ENTRIES];

/*!
 * @brief Generates a hash value for a given GramPolyKey structure.
 *
 * This function creates a hash value for a GramPolyKey, which is used in the memoization 
 * process of the Gram Polynomial calculations. The hash is generated using a combination of 
 * the fields within the GramPolyKey structure: dataIndex, polynomialOrder, derivativeOrder, 
 * and caseType. 
 *
 * The hashing algorithm combines these fields using a series of multiplications and additions, 
 * along with a chosen prime number (37 in this case), to generate a relatively unique hash 
 * for each distinct GramPolyKey. The resulting hash value is then modulated by the `tableSize` 
 * to ensure it fits within the bounds of the memoization table.
 *
 * This hash function is designed to distribute keys uniformly across the hash table, thereby 
 * reducing collisions and improving the efficiency of the memoization process.
 *
 * @param key The GramPolyKey structure containing the parameters for which the hash is to be generated.
 * @param tableSize The size of the hash table, used to ensure the hash value fits within the table.
 * @return The generated hash value as an unsigned integer.
 */
unsigned int hashGramPolyKey(GramPolyKey key, int tableSize) {
    unsigned int hash = 0;
    hash = key.dataIndex;
    hash = (hash * 37 + key.polynomialOrder) % tableSize;
    hash = (hash * 37 + key.derivativeOrder) % tableSize;
    hash = (hash * 37 + key.caseType) % tableSize;  // Include caseType in the hash
    return hash % tableSize;
}

/*!
 * @brief Initialize the memoization table.
 */
static void initializeMemoizationTable() {
    for (int i = 0; i < MAX_ENTRIES; i++) {
        memoizationTable[i].isOccupied = 0;
    }
    totalHashMapEntries = 0;  // Reset the counter
}

/*!
 * @brief Compare two GramPolyKey structures.
 *
 * @param key1 The first GramPolyKey structure.
 * @param key2 The second GramPolyKey structure.
 * @return 1 if keys are equal, otherwise 0.
 */
static int compareGramPolyKeys(GramPolyKey key1, GramPolyKey key2) {
    return key1.dataIndex == key2.dataIndex &&
           key1.polynomialOrder == key2.polynomialOrder &&
           key1.derivativeOrder == key2.derivativeOrder &&
           key1.caseType == key2.caseType;  // Include caseType in the comparison
}

/*!
 * @brief Calculate a generalized factorial product for use in polynomial coefficient normalization.
 *
 * @param upperLimit The upper boundary of the range (inclusive) for the product calculation.
 * @param termCount The number of consecutive integers from `upperLimit` to include in the product.
 * @return The product of the sequence, representing a generalized factorial, as a float.
 */
static inline float GenFact(uint8_t upperLimit, uint8_t termCount) {
    float product = 1.0f;  // Changed from double to float
    for (uint8_t j = (upperLimit - termCount) + 1; j <= upperLimit; j++) {
        product *= j;
    }
    return product;
}


/*!
 * @brief Calculates the value of a Gram Polynomial or its derivative at a specific index within a data window.
 *
 * The Gram polynomials are utilized due to their orthogonality property, which ensures that the 
 * integral (or in discrete cases, the sum) of the product of any two polynomials of different orders 
 * over the window is zero. This orthogonality is crucial for isolating the contributions of each polynomial 
 * order to the filtering process, thus minimizing distortion during data smoothing and differentiation.
 *
 * @param dataIndex The index within the data window at which to evaluate the polynomial.
 * @param halfWindowSize The half size of the window ('m'), defining the symmetric window as '2m+1'.
 * @param polynomialOrder The order of the polynomial (k) to compute.
 * @param derivativeOrder The order of the derivative to compute (0 for the polynomial itself).
 * @return The value of the Gram Polynomial or its specified derivative at the given index.
 */
static float_t GramPoly(uint8_t polynomialOrder) {
    //gramPolyCallCount++;

    // Use global variables within the function
    uint8_t halfWindowSize = g_halfWindowSize;
    uint8_t derivativeOrder = g_derivativeOrder;
    int dataIndex = g_dataIndex;

    if (polynomialOrder == 0) { //base case. memoization mostly aims to skip this part if it was executed in the previous iterations of Gram Poly. 
        return (g_derivativeOrder == 0) ? 1.0 : 0.0;
    }

    float a = (4.0 * polynomialOrder - 2.0) / (polynomialOrder * (2.0 * halfWindowSize - polynomialOrder + 1.0));
    float b = 0.0;
    float c = ((polynomialOrder - 1.0) * (2.0 * halfWindowSize + polynomialOrder)) / (polynomialOrder * (2.0 * halfWindowSize - polynomialOrder + 1.0));

    if (polynomialOrder >= 2) {
        // Calculate the first part of b
        b += dataIndex * GramPoly(polynomialOrder - 1); // Recursion with updated polynomialOrder
        
        // Calculate the second part of b, taking into account derivativeOrder
        if (derivativeOrder > 0) {
            // Temporarily decrement the global derivativeOrder for the recursive call
            g_derivativeOrder = derivativeOrder - 1;
           
            b += derivativeOrder * GramPoly(polynomialOrder - 1); // Recursion with updated polynomialOrder and derivativeOrder
            // Restore the global derivativeOrder after the recursive call
            g_derivativeOrder = derivativeOrder;
        }
        
        // Recursion for the second term of GramPoly
        return a * b - c * GramPoly(polynomialOrder - 2);
    } else if (polynomialOrder == 1) {
     
        a = (2.0) / (2.0 * halfWindowSize);
        // Calculate b for polynomialOrder == 1
        b += dataIndex * GramPoly(0);
        if (derivativeOrder > 0) {
            // Temporarily decrement the global derivativeOrder for the recursive call
            g_derivativeOrder = derivativeOrder - 1;
            b += derivativeOrder * GramPoly(0);
            // Restore the global derivativeOrder after the recursive call
            g_derivativeOrder = derivativeOrder;
        }
        return a * b;
    }

    return 0.0;
}

/*!
 * @brief Memoization wrapper for the GramPoly function.
 *
 * This function serves as a memoization wrapper for the GramPoly function, 
 * aiming to optimize the computational efficiency by storing and reusing 
 * previously calculated values. It uses a hash table for memoization.
 *
 * It first calculates a hash index for the given Gram Polynomial parameters 
 * (data index, polynomial order, derivative order, and case type). If the 
 * calculated polynomial value for these parameters is already stored in the 
 * memoization table, it returns the stored value. Otherwise, it calculates 
 * the value using the GramPoly function, stores it in the table, and then 
 * returns the value.
 *
 * This approach significantly reduces the number of recursive calls to GramPoly, 
 * especially in cases where the same polynomial values are needed multiple times.
 *
 * @param polynomialOrder The order of the polynomial.
 * @param caseType The type of case (central or border) for which the polynomial is evaluated.
 * @return The memoized value of the Gram Polynomial or its derivative.
 */
static float memoizedGramPoly(uint8_t polynomialOrder, uint8_t caseType) {
    GramPolyKey key = {g_dataIndex, polynomialOrder, g_derivativeOrder, caseType};
    unsigned int hashIndex = hashGramPolyKey(key, MAX_ENTRIES);

    // Linear probing for collision resolution
    int startIndex = hashIndex;  // Remember where we started
    while (memoizationTable[hashIndex].isOccupied) {
        if (compareGramPolyKeys(memoizationTable[hashIndex].key, key)) {
            // Key found, return the stored value
            return memoizationTable[hashIndex].value;
        }
        hashIndex = (hashIndex + 1) % MAX_ENTRIES;  // Move to next index
        if (hashIndex == startIndex) {
            // We've looped all the way around; the table is full
            break;
        }
    }

    // If we're here, we didn't find the key, so calculate the value
    float value = GramPoly(polynomialOrder);

    // Check if we can add a new entry
    if (totalHashMapEntries < MAX_ENTRIES) {
        memoizationTable[hashIndex].key = key;
        memoizationTable[hashIndex].value = value;
        memoizationTable[hashIndex].isOccupied = 1;
        //totalHashMapEntries++;
    }  // Otherwise, the table is full; we can't memoize this value

    return value;
}

/*!
 * @brief Computes the weight of a specific data point within a window for least-squares polynomial fitting.
 *
 * @param dataIndex Index of the data point within the window for which the weight is being calculated.
 * @param targetPoint The index within the dataset at which the least-squares fit is evaluated.
 * @param polynomialOrder The order of the polynomial used in the least-squares fitting process.
 * @param caseType Specifies the handling method for calculating weights at the edges of the data window.
 * @return The calculated weight for the data point at `dataIndex`.
 */
static float Weight(int dataIndex, int targetPoint, uint8_t polynomialOrder, uint8_t caseType) {
    float w = 0.0;
    uint8_t derivativeOrder = g_derivativeOrder;
    // calculating binomial-like coefficients
    for (uint8_t k = 0; k <= polynomialOrder; ++k) {
        g_dataIndex = dataIndex;
        g_derivativeOrder = 0;
        float part1 = memoizedGramPoly(k, caseType);  // Uses g_dataIndex implicitly
        g_derivativeOrder = derivativeOrder;
        g_dataIndex = targetPoint; 
        
        float part2 = memoizedGramPoly(k, caseType);  // Uses g_dataIndex (now targetPoint) implicitly

        w += (2 * k + 1) * (GenFact(2 * g_halfWindowSize, k) / GenFact(2 * g_halfWindowSize + k + 1, k + 1)) * part1 * part2;
    }
    return w;
}

/*!
 * @brief Computes the weights for each data point in a specified window for Savitzky-Golay filtering.
 *
 * @param halfWindowSize The half window size of the Savitzky-Golay filter.
 * @param targetPoint The point at which the least-squares fit is evaluated.
 * @param polynomialOrder The order of the polynomial used in the least-squares fit.
 * @param derivativeOrder The order of the derivative for which the weights are being calculated.
 * @param weights An array to store the calculated weights.
 * @param caseType Indicates the type of case for weight calculation.
 */
static void ComputeWeights(uint8_t halfWindowSize, uint16_t targetPoint, uint8_t polynomialOrder, uint8_t derivativeOrder, float* weights, int caseType) {
    g_halfWindowSize = halfWindowSize;
    g_derivativeOrder = derivativeOrder;
    g_targetPoint = targetPoint;
    //printf("g_targetPoint %d\n", g_targetPoint);
    uint16_t fullWindowSize = 2 * halfWindowSize + 1;
    for(int dataIndex = 0; dataIndex < fullWindowSize; ++dataIndex) {
        // Now Weight function is provided all necessary arguments including table for memoization
        weights[dataIndex] = Weight(dataIndex - g_halfWindowSize, g_targetPoint, polynomialOrder, caseType);
    }
}

/*!
 * @brief Applies the Savitzky-Golay filter to the given data set.
 *
 * This function applies the Savitzky-Golay smoothing filter to a data set based on 
 * the provided filter configuration. It handles both central and edge cases within the data set.
 *
 * For central cases, the filter is applied using a symmetric window centered on each data point. 
 * The filter weights are applied across this window to compute the smoothed value for each central data point.
 *
 * For edge cases (leading and trailing edges of the data set), the function computes 
 * specific weights for each border case. These weights account for the asymmetry at the data 
 * set edges. The filter is then applied to these edge cases using the respective calculated weights.
 *
 * @param data The array of data points to which the filter is to be applied.
 * @param dataSize The size of the data array.
 * @param halfWindowSize The half window size (m) of the Savitzky-Golay filter. The full window 
 *                       size is '2m + 1'.
 * @param targetPoint The point at which the least-squares fit is evaluated. This is typically the 
 *                    center of the window but can vary based on the filter application.
 * @param filter A pointer to the Savitzky-GolayFilter structure containing filter configuration 
 *               and precomputed weights.
 * @param filteredData The array where the filtered data points will be stored.
 */
void ApplyFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize, uint16_t targetPoint, SavitzkyGolayFilter* filter, MqsRawDataPoint_t filteredData[]) {
    const int window = 2 * halfWindowSize + 1;  // Full window size
    const int endidx = dataSize - 1;
    uint8_t width = halfWindowSize;

    // Handle Central Cases
    for(int i = 0; i <= dataSize - window; ++i) {  // Adjusted indices to account for halfWindowSize
        float sum = 0.0;
        for(int j = 0; j < window; ++j) {  // Loop from -halfWindowSize to halfWindowSize
            // Apply weights centered at 'i', spanning from 'i - halfWindowSize' to 'i + halfWindowSize'
            sum += filter->weights[j] * data[i + j].phaseAngle;  // Adjusted indices for weights and data
        } 
        filteredData[i + width].phaseAngle = sum;
    }
    
    // Handle both Leading and Trailing Edge Cases in a single loop
    for (int i = 0; i < width; ++i) {
        // Leading edge case
        ComputeWeights(halfWindowSize, width - i, filter->conf.polynomialOrder, filter->conf.derivativeOrder, filter->weights, 1);
        float leadingSum = 0.0;
        for (int j = 0; j < window; ++j) {
            //printf("dataIndex %d\n", dataIndex);
            leadingSum += filter->weights[j] * data[window - j - 1].phaseAngle;
        }
        filteredData[i].phaseAngle = leadingSum;

        // Trailing edge case
       
        float trailingSum = 0.0;
        for (int j = 0; j < window; ++j) {
           
            trailingSum += filter->weights[j] * data[endidx - window + j + 1].phaseAngle;
        }
        filteredData[endidx - i].phaseAngle = trailingSum;
    }
}

/**
 * @brief Applies a causal Savitzky-Golay filter to a specific point in a dataset.
 *
 * This function is designed for data smoothing using the Savitzky-Golay filter method. 
 * It is a causal filter, meaning it uses only past values up to the specified window size 
 * for smoothing. The function handles edge cases with either mirror padding or by utilizing 
 * a previous dataset. This code below just exists to give an idea. 
 *
 * @param data An array of data points where each point is of type MqsRawDataPoint_t.
 *             This array represents the current dataset to which the filter is applied.
 * @param index The index in the 'data' array at which the filter is to be applied. The filter
 *              uses values from 'data' at and before this index, up to the size of the window.
 * @param dataSize The total number of elements in the 'data' array.
 * @param halfWindowSize The half-size of the filter window. The actual window size used
 *                       is computed as (2 * halfWindowSize + 1). The filter uses this number
 *                       of past values from 'data' for calculating the smoothed value.
 * @param filter A pointer to a SavitzkyGolayFilter object, which contains the filter weights.
 * @param filteredData An array where the filtered data points will be stored. It should be
 *                     pre-allocated with a size at least equal to 'dataSize'.
 * @param previous An optional array of previous data points, used when the filter needs to access
 *                 data points before the start of the 'data' array. This is relevant for indices
 *                 smaller than the window size.
 * @param usePrevious A boolean flag to determine the behavior for indices smaller than the window size.
 *                    If true, values from the 'previous' array are used. If false, mirror padding
 *                    (using data[0]) is applied.
 */
/*
---DEACTIVATED---
void ApplyFilterAtPoint(MqsRawDataPoint_t data[], int index, size_t dataSize, uint8_t halfWindowSize, SavitzkyGolayFilter* filter, MqsRawDataPoint_t filteredData[], MqsRawDataPoint_t previous[], bool usePrevious) {
    const int window = 2 * halfWindowSize + 1;
    double sum = 0.0;
    if (index < dataSize) {
        for (int j = 0; j < window; ++j) {
            int dataIndex = index - window + j;
            double phaseAngle;
            if (dataIndex < 0) {
                if (usePrevious) {
                    // Use corresponding value from previous[] array
                    int previousIndex = dataSize + dataIndex; // Equivalent to dataSize - abs(dataIndex)
                    phaseAngle = previous[previousIndex].phaseAngle;
                } else {
                    // Mirror padding
                    phaseAngle = data[0].phaseAngle;
                }
            } else {
                phaseAngle = data[dataIndex].phaseAngle;
            }
            sum += filter->weights[j] * phaseAngle;
        }
        filteredData[index].phaseAngle = sum;
    }
}
*/

/*!
 * @brief Initializes and configures the Savitzky-Golay filter.
 *
 * This function initializes a Savitzky-Golay filter with specified configuration parameters. 
 * The filter is used for smoothing data points in a dataset and can be configured to operate 
 * as either a causal filter or a non-causal filter.
 *
 * @param halfWindowSize The half window size of the filter.
 * @param polynomialOrder The order of the polynomial used in the least-squares fit.
 * @param targetPoint The target point for the filter.
 * @param derivativeOrder The order of the derivative for which the weights are being calculated.
 * @param time_step The time step used in the filter.
 * 
 * @return A pointer to the initialized Savitzky-GolayFilter structure.
 */
SavitzkyGolayFilter* SavitzkyGolayFilter_init(SavitzkyGolayFilterConfig conf) {
    SavitzkyGolayFilter* filter = (SavitzkyGolayFilter*)malloc(sizeof(SavitzkyGolayFilter));
    if (!filter) return NULL; 

    filter->conf = conf;
    filter->weights = (float*)malloc((2 * conf.halfWindowSize + 1) * sizeof(double));

    if (!filter->weights) {
        free(filter);
        return NULL;
    }

    // Now passing the memoization table and size to ComputeWeights
    ComputeWeights(conf.halfWindowSize, conf.targetPoint, conf.polynomialOrder, conf.derivativeOrder, filter->weights, 1);
    //printf()
    filter->dt = pow(conf.time_step, conf.derivation_order); //unused feature. Would be useful for real time applications. 

    return filter;
}

/*!
 * @brief Initializes a Savitzky-Golay filter with specified parameters.
 *
 * @param halfWindowSize The half window size.
 * @param polynomialOrder The polynomial order.
 * @param targetPoint The target point.
 * @param derivativeOrder The derivative order.
 * @param time_step The time step.
 * @return A pointer to the initialized Savitzky-GolayFilter structure.
 */
SavitzkyGolayFilter* initFilter(uint8_t halfWindowSize, uint8_t polynomialOrder, uint8_t targetPoint, uint8_t derivativeOrder, float time_step) {
    // Initialize configuration for the Savitzky-Golay filter
    SavitzkyGolayFilterConfig conf = {halfWindowSize, targetPoint, polynomialOrder, derivativeOrder, time_step, derivativeOrder};

    // Initialize filter with the given configuration and return it
    return SavitzkyGolayFilter_init(conf);
}

/*!
 * @brief Frees the memory allocated for a Savitzky-Golay filter.
 *
 * @param filter The pointer to the Savitzky-Golay filter to free.
 */
void SavitzkyGolayFilter_free(SavitzkyGolayFilter** filter) {
    if (*filter != NULL) {
        free((*filter)->weights);
        free(*filter);
        *filter = NULL;
    }
}  

/*!
 * @brief Manages a singleton instance of the Savitzky-Golay Filter.
 *
 * Implements a singleton pattern for the Savitzky-Golay Filter. It initializes the filter on 
 * the first call and returns the same instance on subsequent calls, ensuring a single shared 
 * instance is used. 
 *
 * @param halfWindowSize The half window size.
 * @param polynomialOrder The polynomial order.
 * @param targetPoint The target point.
 * @param derivativeOrder The derivative order.
 * @param reset Boolean flag to indicate whether to reset the filter instance.
 * @return Pointer to the filter instance, or NULL if reset.
 */
SavitzkyGolayFilter* getFilterInstance(uint8_t halfWindowSize, uint8_t polynomialOrder, uint8_t targetPoint, uint8_t derivativeOrder, bool reset) {
    static SavitzkyGolayFilter* filterInstance = NULL;
    if (filterInstance != NULL) {
        SavitzkyGolayFilter_free(&filterInstance);  // Free and reset the instance
        //return NULL; // Return immediately after resetting
    }
    
    float_t time_step = 1.0;
    if (filterInstance == NULL) {
        filterInstance = initFilter(halfWindowSize, polynomialOrder, targetPoint, derivativeOrder, time_step);  // Initialize the filter if it hasn't been already
    }
    return filterInstance;
}

/*!
 * @brief Resets the singleton instance of the Savitzky-Golay filter.
 */
void resetFilterInstance() {
    static SavitzkyGolayFilter* filterInstance = NULL;
    SavitzkyGolayFilter_free(&filterInstance);
}

/*!
 * @brief Applies the Savitzky-Golay filter to the input data.
 *
 * @param data The input array of data points.
 * @param dataSize The number of elements in the input array.
 * @param halfWindowSize The half window size of the filter.
 * @param filteredData The output array to store the filtered data points.
 * @param polynomialOrder The polynomial order.
 * @param targetPoint The target point.
 * @param derivativeOrder The derivative order.
 */
void mes_savgolFilter(MqsRawDataPoint_t data[], size_t dataSize, uint8_t halfWindowSize, MqsRawDataPoint_t filteredData[], uint8_t polynomialOrder, uint8_t targetPoint, uint8_t derivativeOrder) {
    initializeMemoizationTable();
   printf("halfWindowSize %u:\n", halfWindowSize);
    SavitzkyGolayFilter *filter = getFilterInstance(halfWindowSize, polynomialOrder, targetPoint, derivativeOrder, false); //halfWindowSize, 3, 0, 0 
   
   // Print the weights
   /*
    int windowSize = 2 * halfWindowSize + 1;
    printf("Filter Weights:\n");
    for (int i = 0; i < windowSize; ++i) {
        printf("Weight[%d] = %f\n", i, filter->weights[i]);
    }
    */
       
   
    ApplyFilter(data, dataSize, halfWindowSize, targetPoint, filter, filteredData);
    resetFilterInstance(); // Reset the filter instance after use
}

