/*!
 * Optimized Savitzky-Golay Filter Implementation with Recursive Calculation and Memoization.
 *
 * Author: Tugbars Heptaskin
 * Date: 12/01/2024
 * Company: Aminic Aps
 *
 * This implementation provides an efficient and optimized version of the Savitzky-Golay filter,
 * commonly used for smoothing and differentiating data. The key enhancements in this implementation
 * include the use of global variables to minimize stack footprint and memoization to reduce
 * redundant recursive computations.
 *
 * The core of this implementation is the GramPoly function, which computes the Gram Polynomial
 * or its derivative. These polynomials form the foundation for calculating the coefficients of
 * the least squares fitting polynomial in the Savitzky-Golay filter. The filter applies these 
 * coefficients to smooth or differentiate data with minimal distortion.
 *
 * Optimizations:
 * 1. Minimizing Stack Footprint:
 *    Critical parameters like half window size, derivative order, and data index are stored in
 *    global variables, significantly reducing the function's stack size requirement. This approach
 *    is particularly effective in preventing stack overflows in scenarios involving large datasets
 *    or high polynomial orders.
 *
 * 2. Memoization for Computational Efficiency:
 *    The GramPoly function uses a memoization technique where computed values are stored in a hash
 *    table. This strategy drastically lowers the number of recursive calls, enhancing the filter's
 *    efficiency, especially when the same polynomial values are needed repeatedly.
 *
 * The implementation also includes functions for initializing and applying the filter, handling
 * both central and edge cases in data sets. The output of this code has been validated to match
 * exactly with Matlab's Savitzky-Golay filter, ensuring its accuracy and reliability for 
 * professional applications.
 *
 * This optimized Savitzky-Golay filter implementation is designed for robust performance in 
 * resource-constrained environments and is particularly well-suited for applications in data
 * analysis, signal processing, and similar fields where efficient data smoothing and differentiation
 * are critical.
 */

#include <stdio.h>
#include <stdint.h>  // for uint16_t, uint8_t
#include <stdbool.h>
#include <math.h>  
#include "mes_savgol.h"

int gramPolyCallCount = 0;

int g_dataIndex;
uint8_t g_halfWindowSize;
uint8_t g_targetPoint = 0;
uint8_t g_derivativeOrder;
int totalHashMapEntries = 0;

/*!
 * @brief Performance Trade-off in Savitzky-Golay Filter with Selective Memoization.
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
#define MAX_POLYNOMIAL_ORDER 3
#define MAX_ENTRIES 255 // for computing weights for each border case. 
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

void initializeMemoizationTable() {
    for (int i = 0; i < MAX_ENTRIES; i++) {
        memoizationTable[i].isOccupied = 0;
    }
    totalHashMapEntries = 0;  // Reset the counter
}

int compareGramPolyKeys(GramPolyKey key1, GramPolyKey key2) {
    return key1.dataIndex == key2.dataIndex &&
           key1.polynomialOrder == key2.polynomialOrder &&
           key1.derivativeOrder == key2.derivativeOrder &&
           key1.caseType == key2.caseType;  // Include caseType in the comparison
}

/*!
 * @brief Calculates a generalized factorial product.
 *
 * This function computes the product of a sequence of integers, which is a generalized 
 * form of factorial. Unlike the standard factorial which is the product of all positive 
 * integers up to a certain number, this generalized factorial computes the product of 
 * integers in a specified range. 
 *
 * The range is determined by the `upperLimit` and `termCount`. It starts from 
 * `(upperLimit - termCount + 1)` and goes up to `upperLimit`. For example, if `upperLimit` 
 * is 5 and `termCount` is 3, the function will calculate the product of 3, 4, and 5.
 *
 * @param upperLimit The upper limit of the product range.
 * @param termCount The number of terms to include in the product.
 * @return The generalized factorial product as a float.
 */
double GenFact(uint8_t upperLimit, uint8_t termCount) {
    //printf("Calculating GenFact for upperLimit = %d, termCount = %d\n", upperLimit, termCount);
    double product = 1.0;
    for (uint8_t j = (upperLimit - termCount) + 1; j <= upperLimit; j++) {
        product *= j;
    }
    return product;
}

/*!
 * @brief Calculates the Gram Polynomial or its derivative.
 *
 * This function computes the Gram Polynomial (or its derivative) which is used 
 * in the calculation of Savitzky-Golay filter coefficients. The polynomial is 
 * evaluated at a specific data point index (i) with a given order (k) over 
 * a symmetric window of '2m+1' points.
 *
 * Gram polynomials are a series of orthogonal polynomials. Orthogonality means that the 
 * integral of the product of any two different Gram polynomials over a certain interval is zero. 
 * This property is crucial for minimizing distortion when smoothing and differentiating data 
 * using Savitzky-Golay filters. In these filters, Gram polynomials form the basis for calculating 
 * the coefficients of the least squares fitting polynomial, which are then used to convolve 
 * with the data for smoothing or differentiating it.
 *
 * The function is recursive, utilizing the properties of Gram polynomials to build each 
 * polynomial upon the previous ones. The recursive formula used is as follows:
 * 
 * Base Cases:
 *  - For k = 0: G_0(x) = 1
 *  - For k = 1: G_1(x) = x
 * 
 * Recursive Step:
 *  - For k >= 2: G_k(x) = ((4k - 2) * x * G_(k-1)(x) - (k - 1) * (2m + k) * G_(k-2)(x)) / (k * (2m - k + 1))
 
 * 4k - 2, k - 1, 2m + k, and 2m - k + 1 are coefficients that ensure each subsequent polynomial is orthogonal to the others.
 * x * G_(k-1)(x) and G_(k-2)(x) are the recursive elements, meaning each polynomial is built upon the previous ones.
 * The division by k * (2m - k + 1) is part of the normalization process, ensuring that the polynomials maintain the correct
 * scaling to be orthogonal over the specified interval.
 *
 * @param polynomialOrder The order of the polynomial (k).
 * @return The value of the Gram Polynomial or its derivative at a specific data point.
 */
double GramPoly(uint8_t polynomialOrder) {
    gramPolyCallCount++;

    // Use global variables within the function
    uint8_t halfWindowSize = g_halfWindowSize;
    uint8_t derivativeOrder = g_derivativeOrder;
    int dataIndex = g_dataIndex;

    if (polynomialOrder == 0) { //base case. memoization mostly aims to skip this part if it was executed in the previous iterations of Gram Poly. 
        return (g_derivativeOrder == 0) ? 1.0 : 0.0;
    }

    double a = (4.0 * polynomialOrder - 2.0) / (polynomialOrder * (2.0 * halfWindowSize - polynomialOrder + 1.0));
    double b = 0.0;
    double c = ((polynomialOrder - 1.0) * (2.0 * halfWindowSize + polynomialOrder)) / (polynomialOrder * (2.0 * halfWindowSize - polynomialOrder + 1.0));

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

    // Catch-all return, function should never reach this point
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
double memoizedGramPoly(uint8_t polynomialOrder, uint8_t caseType) {
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
    double value = GramPoly(polynomialOrder);

    // Check if we can add a new entry
    if (totalHashMapEntries < MAX_ENTRIES) {
        memoizationTable[hashIndex].key = key;
        memoizationTable[hashIndex].value = value;
        memoizationTable[hashIndex].isOccupied = 1;
        totalHashMapEntries++;
    }  // Otherwise, the table is full; we can't memoize this value

    return value;
}

/*!
 * @brief Calculates the weight for a specific data point in least-squares fitting.
 *
 * This function computes the weight of the ith data point (specified by dataIndex) 
 * for the least-squares fit at a specific point (targetPoint). The calculation is 
 * performed over a window of size '2 * g_halfWindowSize + 1'. This function is a part 
 * of the Savitzky-Golay filter implementation, which uses the Gram-Schmidt process 
 * in orthogonalization of vectors for numerical stabilization.
 * 
 * The weight is calculated for a specified polynomial order (polynomialOrder) and 
 * takes into account the derivative order (g_derivativeOrder) of the smoothing 
 * function. The caseType parameter determines the type of edge handling for 
 * weight calculation (central or border case).
 * 
 * The weight calculation involves several components:
 * - (2 * k + 1): This term, from the integration of the squared Gram polynomials, 
 *   helps normalize the weight and ensures that the filter coefficients lead to an 
 *   orthogonal polynomial fit.
 * - GenFact(2 * g_halfWindowSize, k) / GenFact(2 * g_halfWindowSize + k + 1, k + 1): 
 *   A ratio of generalized factorial functions, playing a role in scaling the coefficients 
 *   appropriately. This is used for calculating binomial coefficients in the context of 
 *   polynomial coefficients.
 * - part1 * part2: Evaluations of the Gram polynomial at the data index (dataIndex) and 
 *   the target point (targetPoint). This multiplication represents the convolution of the 
 *   Gram polynomial evaluated at different points, integrating the polynomial across the window.
 *
 * @param dataIndex Index of the data point for which the weight is to be calculated.
 * @param targetPoint The point at which the least-squares fit is evaluated.
 * @param polynomialOrder The order of the polynomial used in the least-squares fit.
 * @param caseType The type of case for weight calculation (0 for central, 1 for border).
 * 
 * @return Calculated weight for the specified data point.
 */
double Weight(int dataIndex, int targetPoint, uint8_t polynomialOrder, uint8_t caseType) {
    double w = 0.0;
    uint8_t derivativeOrder = g_derivativeOrder;
    // calculating binomial-like coefficients
    for (uint8_t k = 0; k <= polynomialOrder; ++k) {
        g_dataIndex = dataIndex;
        g_derivativeOrder = 0;
        double part1 = memoizedGramPoly(k, caseType);  // Uses g_dataIndex implicitly
        g_derivativeOrder = derivativeOrder;
        g_dataIndex = targetPoint; 
        
        double part2 = memoizedGramPoly(k, caseType);  // Uses g_dataIndex (now targetPoint) implicitly

        w += (2 * k + 1) * (GenFact(2 * g_halfWindowSize, k) / GenFact(2 * g_halfWindowSize + k + 1, k + 1)) * part1 * part2;
    }
    return w;
}

/*!
 * @brief Computes the weights for each data point in a specified window for Savitzky-Golay filtering.
 *
 * This function calculates the weights for a moving window of data points, used in the 
 * Savitzky-Golay smoothing filter. The weights are computed for each point within the window
 * defined by '2 * halfWindowSize + 1'. The calculation considers the polynomial order and the
 * derivative order of the smoothing function, along with the specific target point for the 
 * least-squares fitting.
 *
 * @param halfWindowSize The half window size (m) of the Savitzky-Golay filter. The full window 
 *                       size is '2m + 1'.
 * @param targetPoint The point at which the least-squares fit is evaluated. This is typically the 
 *                    center of the window but can vary based on the filter application.
 * @param polynomialOrder The order of the polynomial used in the least-squares fit.
 * @param derivativeOrder The order of the derivative for which the weights are being calculated.
 * @param weights An array to store the calculated weights. The array size should be at least 
 *                '2 * halfWindowSize + 1'.
 * @param caseType Indicates the type of case for weight calculation: 0 for central, 1 for border 
 *                 cases. This affects how weights are computed near the edges of the data set.
 */
void ComputeWeights(uint8_t halfWindowSize, uint16_t targetPoint, uint8_t polynomialOrder, uint8_t derivativeOrder, double* weights, int caseType) {
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
        double sum = 0.0;
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
        double leadingSum = 0.0;
        for (int j = 0; j < window; ++j) {
            //printf("dataIndex %d\n", dataIndex);
            leadingSum += filter->weights[j] * data[window - j - 1].phaseAngle;
        }
        filteredData[i].phaseAngle = leadingSum;

        // Trailing edge case
       
        double trailingSum = 0.0;
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

// Compute weights for the time window 2*m+1, for the polyOrder'th least-square
// point of the derivativeOrder'th derivative
SavitzkyGolayFilter* SavitzkyGolayFilter_init(SavitzkyGolayFilterConfig conf) {
    SavitzkyGolayFilter* filter = (SavitzkyGolayFilter*)malloc(sizeof(SavitzkyGolayFilter));
    if (!filter) return NULL; 

    filter->conf = conf;
    filter->weights = (double*)malloc((2 * conf.halfWindowSize + 1) * sizeof(double));

    if (!filter->weights) {
        free(filter);
        return NULL;
    }

    // Now passing the memoization table and size to ComputeWeights
    ComputeWeights(conf.halfWindowSize, conf.targetPoint, conf.polynomialOrder, conf.derivativeOrder, filter->weights, 1);
    filter->dt = pow(conf.time_step, conf.derivation_order);

    return filter;
}

/*!
 * @brief Initializes and configures the Savitzky-Golay filter.
 *
 * This function initializes a Savitzky-Golay filter with specified configuration parameters. 
 * The filter is used for smoothing data points in a dataset and can be configured to operate 
 * as either a causal filter or a non-causal filter.
 *
 * Setting the 'targetPoint' parameter to 'halfWindowSize' configures the filter to act as a 
 * causal filter, using only past data points. This allows the filter to be used in real-time 
 * applications but may result in reduced accuracy as it does not consider future data points.
 *
 * Alternatively, setting 'targetPoint' to 0 configures the filter as a non-causal filter, 
 * which uses both past and future data points. This approach typically offers higher accuracy 
 * in smoothing but introduces a delay, making it unsuitable for real-time applications.
 *
 * @param halfWindowSize The half window size (m) for the filter. The full window size is '2m + 1'.
 * @param polynomialOrder The order of the polynomial (n) used in the least-squares fit.
 * @param targetPoint The target point for the filter. Set to 'halfWindowSize' for a causal filter 
 *                    or 0 for a non-causal filter.
 * @param derivativeOrder The order of the derivative (d) for which the weights are being calculated.
 * @param time_step The time step used in the filter.
 * 
 * @return A pointer to the initialized Savitzky-GolayFilter structure.
 */
SavitzkyGolayFilter* initFilter() {
    uint8_t halfWindowSize = 6; // m value
    uint8_t polynomialOrder = 3; // n value
    uint16_t targetPoint = 0; // t value
    uint8_t derivativeOrder = 0; // d value
    double time_step = 1.0; // time step. I kept the time step here so that you can use arntanguy's way of doing it. 

    // Initialize configuration for the Savitzky-Golay filter
    SavitzkyGolayFilterConfig conf = {halfWindowSize, targetPoint, polynomialOrder, derivativeOrder, time_step, derivativeOrder};

    // Initialize filter with the given configuration and return it
    return SavitzkyGolayFilter_init(conf);
}

/*!
 * @brief Retrieves a singleton instance of the Savitzky-Golay Filter.
 *
 * This function implements the singleton design pattern to manage a single, shared instance 
 * of the Savitzky-Golay Filter. 
 *
 * If the filter instance has not been created yet (i.e., it is NULL), the function initializes 
 * it using `initFilter`. Subsequent calls to this function will return the existing initialized 
 * instance, thereby avoiding re-initialization and maintaining a single shared instance.
 *
 * @return A pointer to the singleton instance of the Savitzky-Golay Filter.
 */
SavitzkyGolayFilter* getFilterInstance() {
    static SavitzkyGolayFilter* filterInstance = NULL;
    if (filterInstance == NULL) {
        filterInstance = initFilter();  // Initialize the filter if it hasn't been already
    }
    return filterInstance;
}

void SavitzkyGolayFilter_free(SavitzkyGolayFilter* filter) {
    free(filter->weights);
    free(filter);
}

//currentFilterSweep dogru filterData yerlerini gostersin diye updateler yapiliyor mu calibrationda?
//code for Mes_sweep.c side. 
void mes_SavgolFilter(MqsRawDataPoint_t data[], size_t dataSize, MqsRawDataPoint_t filteredData[]){
    initializeMemoizationTable(); 
    SavitzkyGolayFilter *filter = getFilterInstance();
    
    ApplyFilter(data, dataSize, g_halfWindowSize, g_targetPoint, filter, filteredData);
    //SavitzkyGolayFilter_free(filter);
}

void cleanupFilterInstance() {
    SavitzkyGolayFilter* filter = getFilterInstance();
    if (filter != NULL) {
        SavitzkyGolayFilter_free(filter);
    }
}


