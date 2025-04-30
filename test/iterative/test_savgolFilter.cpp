#include <gtest/gtest.h>
#include <fff.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "savgolFilter.h"

// Define fff mocks
DEFINE_FFF_GLOBALS;

// Mock for ComputeWeights
FAKE_VOID_FUNC(ComputeWeights, uint8_t, uint16_t, uint8_t, uint8_t, float *);

// Mock for GramPolyIterative (to test memoization and cache boundary conditions)
FAKE_VALUE_FUNC(float, GramPolyIterative, uint8_t, int, const GramPolyContext *);

// Test Fixture for Savitzky-Golay Filter Tests
class SavitzkyGolayFilterTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Reset fff mocks before each test to ensure a clean state
        RESET_FAKE(ComputeWeights);
        RESET_FAKE(GramPolyIterative);

        // Initialize dataset with the provided data
        dataSize = sizeof(dataset) / sizeof(dataset[0]);
        rawData = new MqsRawDataPoint_t[dataSize];
        filteredData = new MqsRawDataPoint_t[dataSize];

        // Populate rawData with the dataset values and initialize filteredData to 0
        for (size_t i = 0; i < dataSize; ++i)
        {
            rawData[i].phaseAngle = dataset[i];
            filteredData[i].phaseAngle = 0.0f;
        }

        // Set default filter parameters
        halfWindowSize = 12;
        polynomialOrder = 4;
        targetPoint = 0;
        derivativeOrder = 0;

        // Calculate the window size based on halfWindowSize
        windowSize = 2 * halfWindowSize + 1;
    }

    void TearDown() override
    {
        // Clean up dynamically allocated memory to prevent leaks
        delete[] rawData;
        delete[] filteredData;
    }

    // Utility function to compare floating-point values with tolerance
    bool isClose(float a, float b, float tol = 1e-5)
    {
        return fabs(a - b) <= tol;
    }

    // Dataset (same as provided)
    static double dataset[449];
    static double expectedOutput[449];

    size_t dataSize;
    MqsRawDataPoint_t *rawData;
    MqsRawDataPoint_t *filteredData;

    uint8_t halfWindowSize;
    uint8_t polynomialOrder;
    uint8_t targetPoint;
    uint8_t derivativeOrder;
    int windowSize;
};

// Define the dataset and expected output (same as before)
double SavitzkyGolayFilterTest::dataset[] = {11.272, 11.254, 11.465, 11.269, 11.31, 11.388, 11.385, 11.431, 11.333, 11.437,
                                             11.431, 11.527, 11.483, 11.449, 11.544, 11.39, 11.469, 11.526, 11.498, 11.522,
                                             11.709, 11.503, 11.564, 11.428, 11.714, 11.707, 11.619, 11.751, 11.626, 11.681,
                                             11.838, 11.658, 11.859, 11.916, 11.814, 11.833, 12.046, 11.966, 12.031, 12.079,
                                             11.958, 12.114, 12.041, 12.186, 12.048, 12.258, 12.312, 12.126, 12.159, 12.393,
                                             12.221, 12.45, 12.439, 12.282, 12.373, 12.573, 12.647, 12.545, 12.467, 12.629,
                                             12.686, 12.668, 12.748, 12.71, 12.852, 13.02, 12.848, 13.144, 13.225, 13.211,
                                             13.496, 13.311, 13.33, 13.634, 13.189, 13.623, 13.671, 13.618, 13.645, 13.779,
                                             14.006, 14.13, 14.071, 14.277, 14.223, 14.457, 14.378, 14.698, 14.599, 14.84,
                                             15.143, 15.106, 15.343, 15.506, 15.665, 15.889, 15.878, 16.055, 16.153, 15.966,
                                             16.637, 16.783, 16.746, 17.193, 16.877, 17.656, 17.522, 17.842, 18.086, 18.336,
                                             18.863, 18.977, 19.534, 19.308, 19.626, 19.956, 20.221, 20.673, 20.59, 21.229,
                                             21.767, 22.225, 22.477, 22.695, 22.828, 23.586, 23.776, 24.39, 25.316, 24.639,
                                             25.767, 26.469, 26.976, 27.651, 27.807, 28.089, 28.869, 29.964, 30.367, 30.159,
                                             31.133, 32.034, 33.131, 32.775, 34.372, 34.516, 35.603, 36.214, 37.742, 38.868,
                                             38.702, 39.811, 40.818, 41.422, 41.521, 42.57, 42.819, 42.871, 42.944, 43.851,
                                             44.086, 44.272, 44.466, 44.274, 44.473, 44.348, 43.932, 43.817, 43.48, 42.943,
                                             42.491, 41.793, 41.071, 39.491, 39.231, 38.365, 37.833, 36.583, 35.787, 34.949,
                                             33.006, 32.827, 32.266, 31.012, 30.436, 29.737, 28.097, 28.76, 27.068, 26.195,
                                             25.262, 24.677, 24.211, 23.574, 22.868, 22.781, 22.258, 21.475, 21.247, 21.982,
                                             20.771, 20.383, 20.349, 19.866, 19.433, 18.573, 18.723, 18.325, 18.084, 18.226,
                                             17.492, 17.505, 16.762, 16.907, 16.606, 16.265, 16.234, 15.983, 16.147, 15.811,
                                             15.667, 15.509, 15.325, 15.031, 14.884, 14.881, 14.836, 14.814, 14.706, 14.158,
                                             14.399, 14.123, 14.084, 14.173, 13.963, 13.981, 14.218, 13.898, 13.869, 13.701,
                                             13.397, 13.528, 13.321, 13.071, 13.393, 13.164, 12.876, 13.021, 12.989, 12.869,
                                             13.004, 12.833, 12.795, 12.661, 12.761, 12.547, 12.775, 12.388, 12.425, 12.564,
                                             12.408, 12.301, 12.469, 12.173, 12.323, 12.248, 12.281, 12.208, 11.887, 12.149,
                                             12.073, 12.053, 11.88, 12.066, 11.958, 12.007, 11.868, 11.921, 11.898, 11.804,
                                             11.7, 11.81, 11.758, 11.717, 11.715, 11.611, 11.719, 11.679, 11.619, 11.58,
                                             11.576, 11.589, 11.491, 11.659, 11.506, 11.431, 11.535, 11.349, 11.464, 11.343,
                                             11.492, 11.407, 11.479, 11.269, 11.355, 11.323, 11.341, 11.238, 11.32, 11.333,
                                             11.262, 11.31, 11.221, 11.302, 11.135, 11.139, 11.217, 11.343, 11.225, 11.089,
                                             11.079, 11.127, 11.082, 11.141, 11.186, 11.184, 11.231, 11.025, 11.058, 11.076,
                                             11.087, 11.047, 11.02, 10.996, 10.906, 11.144, 11.005, 10.911, 10.993, 10.858,
                                             11.086, 10.954, 10.906, 11.026, 11.005, 10.934, 10.922, 10.914, 10.955, 11.057,
                                             10.967, 10.811, 10.833, 10.747, 10.821, 10.946, 10.844, 10.838, 10.848, 10.847};

double SavitzkyGolayFilterTest::expectedOutput[] = {11.286263, 11.300699, 11.316540, 11.333239, 11.350313, 11.367350, 11.383996, 11.399975, 11.415064, 11.429111,
                                                    11.442038, 11.453818, 11.464497, 11.478695, 11.489540, 11.489944, 11.492422, 11.504765, 11.521872, 11.525830,
                                                    11.541789, 11.565474, 11.577266, 11.597882, 11.623141, 11.637012, 11.657558, 11.669603, 11.687865, 11.720676,
                                                    11.746226, 11.786491, 11.818863, 11.846914, 11.880419, 11.896534, 11.940505, 11.981898, 12.001115, 12.039563,
                                                    12.056039, 12.071033, 12.099255, 12.124499, 12.160623, 12.175566, 12.184244, 12.217986, 12.260163, 12.286033,
                                                    12.321880, 12.350359, 12.381068, 12.425369, 12.443513, 12.488060, 12.516522, 12.552017, 12.588384, 12.596244,
                                                    12.628742, 12.692409, 12.717324, 12.802947, 12.871245, 12.929922, 13.008872, 13.100385, 13.184816, 13.246750,
                                                    13.293182, 13.355146, 13.397689, 13.446552, 13.487468, 13.544438, 13.599324, 13.671164, 13.744465, 13.829150,
                                                    13.918334, 14.009408, 14.111757, 14.197618, 14.287376, 14.395479, 14.482785, 14.606400, 14.758994, 14.894108,
                                                    15.022083, 15.180305, 15.313669, 15.486173, 15.616129, 15.765237, 15.901664, 16.046532, 16.191078, 16.332191,
                                                    16.476105, 16.612799, 16.785231, 16.971855, 17.189964, 17.429535, 17.657516, 17.923758, 18.183990, 18.441175,
                                                    18.674620, 18.901684, 19.171598, 19.454920, 19.719179, 20.019333, 20.315153, 20.586527, 20.903183, 21.254852,
                                                    21.603359, 21.969740, 22.333229, 22.696753, 23.115231, 23.532743, 23.951433, 24.344376, 24.807137, 25.304558,
                                                    25.806808, 26.286869, 26.852825, 27.368206, 27.904888, 28.426615, 28.993700, 29.574608, 30.071323, 30.658245,
                                                    31.282366, 31.865133, 32.553013, 33.315155, 34.085800, 34.877800, 35.679558, 36.564789, 37.464283, 38.340271,
                                                    39.130142, 39.835079, 40.565998, 41.227657, 41.800159, 42.266232, 42.744019, 43.095047, 43.443588, 43.730831,
                                                    44.012276, 44.172188, 44.290283, 44.401081, 44.439568, 44.327469, 44.114922, 43.839390, 43.369324, 42.876610,
                                                    42.315254, 41.673763, 40.949219, 40.155350, 39.261417, 38.418736, 37.475685, 36.519966, 35.569218, 34.674091,
                                                    33.800404, 32.927299, 32.060524, 31.213284, 30.342306, 29.437513, 28.640974, 27.893448, 27.081963, 26.276363,
                                                    25.544647, 24.822084, 24.116240, 23.543085, 23.062775, 22.600355, 22.217718, 21.872690, 21.486902, 21.222120,
                                                    20.816534, 20.476971, 20.121323, 19.773655, 19.439875, 19.087105, 18.754610, 18.428917, 18.091599, 17.771034,
                                                    17.516739, 17.309605, 17.041180, 16.841337, 16.665405, 16.468464, 16.263027, 16.032488, 15.896288, 15.736573,
                                                    15.605758, 15.495446, 15.331453, 15.197728, 15.045272, 14.904403, 14.752657, 14.608253, 14.488420, 14.403268,
                                                    14.337738, 14.254896, 14.212329, 14.145469, 14.062309, 13.992709, 13.926799, 13.858660, 13.792374, 13.701324,
                                                    13.592616, 13.461172, 13.374629, 13.259208, 13.183605, 13.098215, 13.031052, 13.001228, 12.971628, 12.918434,
                                                    12.889190, 12.829733, 12.782700, 12.744785, 12.691183, 12.635571, 12.589600, 12.543287, 12.489879, 12.457473,
                                                    12.411427, 12.381772, 12.351021, 12.309167, 12.263808, 12.224651, 12.193840, 12.149176, 12.119839, 12.078433,
                                                    12.056309, 12.037029, 12.006192, 11.976708, 11.966196, 11.932602, 11.915422, 11.888301, 11.864414, 11.831606,
                                                    11.787500, 11.781461, 11.746796, 11.714657, 11.691741, 11.675840, 11.661769, 11.646091, 11.633719, 11.618774,
                                                    11.600723, 11.564595, 11.542892, 11.529607, 11.510800, 11.487458, 11.473676, 11.456483, 11.443320, 11.426746,
                                                    11.403642, 11.389740, 11.369459, 11.358132, 11.351963, 11.357075, 11.331485, 11.305193, 11.292272, 11.275845,
                                                    11.270910, 11.263124, 11.265541, 11.251977, 11.239865, 11.203058, 11.190477, 11.178247, 11.169724, 11.157026,
                                                    11.154992, 11.150828, 11.141689, 11.151517, 11.138066, 11.125181, 11.110279, 11.098274, 11.106550, 11.096605,
                                                    11.069006, 11.047166, 11.026806, 11.006600, 10.989899, 10.985172, 10.986661, 10.990328, 10.982807, 10.959945,
                                                    10.960291, 10.961726, 10.971353, 10.980612, 10.975060, 10.967223, 10.958820, 10.959219, 10.949122, 10.935029,
                                                    10.917485, 10.897425, 10.876154, 10.855362, 10.837117, 10.823871, 10.818449, 10.824062, 10.844298, 10.883123};

// Define static weights arrays for use in mock functions
static float weightsCenter[25] = {
    -0.0219f, -0.0156f, -0.0099f, -0.0047f, 0.0f,
    0.0035f, 0.0065f, 0.0090f, 0.0111f, 0.0127f,
    0.0139f, 0.0146f, 0.0149f, 0.0146f, 0.0139f,
    0.0127f, 0.0111f, 0.0090f, 0.0065f, 0.0035f,
    0.0f, -0.0047f, -0.0099f, -0.0156f, -0.0219f};

static float weightsLeading12[25] = {
    0.2143f, 0.1429f, 0.0781f, 0.0195f, -0.0336f,
    -0.0812f, -0.1240f, -0.1623f, -0.1964f, -0.2266f,
    -0.2532f, -0.2766f, -0.2970f, -0.3147f, -0.3300f,
    -0.3430f, -0.3540f, -0.3632f, -0.3708f, -0.3769f,
    -0.3817f, -0.3853f, -0.3879f, -0.3895f, -0.3903f};

static float weightsTrailing12[25] = {
    -0.3903f, -0.3895f, -0.3879f, -0.3853f, -0.3817f,
    -0.3769f, -0.3708f, -0.3632f, -0.3540f, -0.3430f,
    -0.3300f, -0.3147f, -0.2970f, -0.2766f, -0.2532f,
    -0.2266f, -0.1964f, -0.1623f, -0.1240f, -0.0812f,
    -0.0336f, 0.0195f, 0.0781f, 0.1429f, 0.2143f};

// Static helper functions for mocking ComputeWeights
static void MockComputeWeightsLeadingEdge(uint8_t hws, uint16_t tp, uint8_t po, uint8_t d, float *weights)
{
    if (tp == 12)
    {
        for (int i = 0; i < 25; ++i)
        {
            weights[i] = weightsLeading12[i];
        }
    }
    else
    {
        for (int i = 0; i < 25; ++i)
        {
            weights[i] = weightsCenter[i];
        }
    }
}

static void MockComputeWeightsTrailingEdge(uint8_t hws, uint16_t tp, uint8_t po, uint8_t d, float *weights)
{
    if (tp == 12)
    {
        for (int i = 0; i < 25; ++i)
        {
            weights[i] = weightsTrailing12[i];
        }
    }
    else
    {
        for (int i = 0; i < 25; ++i)
        {
            weights[i] = weightsCenter[i];
        }
    }
}

static void MockComputeWeightsCenterCase(uint8_t hws, uint16_t tp, uint8_t po, uint8_t d, float *weights)
{
    for (int i = 0; i < 25; ++i)
    {
        weights[i] = weightsCenter[i];
    }
}

// Test: Leading Edge (Compare with Expected Output)
TEST_F(SavitzkyGolayFilterTest, LeadingEdge)
{
    // Step 1: Run the filter with halfWindowSize = 12 and polynomialOrder = 4
    int result = mes_savgolFilter(rawData, dataSize, halfWindowSize, filteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for leading edge";

    // Step 2: Compare the leading edge points (indices 0 to halfWindowSize-1) with expectedOutput
    for (int i = 0; i < halfWindowSize; ++i)
    {
        EXPECT_TRUE(isClose(filteredData[i].phaseAngle, expectedOutput[i], 1e-5))
            << "Leading edge mismatch at index " << i << ": got " << filteredData[i].phaseAngle
            << ", expected " << expectedOutput[i];
    }
}

// Test: Trailing Edge (Compare with Expected Output)
TEST_F(SavitzkyGolayFilterTest, TrailingEdge)
{
    // Step 1: Run the filter with halfWindowSize = 12 and polynomialOrder = 4
    int result = mes_savgolFilter(rawData, dataSize, halfWindowSize, filteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for trailing edge";

    // Step 2: Compare the trailing edge points (indices dataSize-halfWindowSize to dataSize-1) with expectedOutput
    for (int i = dataSize - halfWindowSize; i < static_cast<int>(dataSize); ++i)
    {
        EXPECT_TRUE(isClose(filteredData[i].phaseAngle, expectedOutput[i], 1e-5))
            << "Trailing edge mismatch at index " << i << ": got " << filteredData[i].phaseAngle
            << ", expected " << expectedOutput[i];
    }
}

// Test: Center Case (Compare with Expected Output)
TEST_F(SavitzkyGolayFilterTest, CenterCase)
{
    // Step 1: Run the filter with halfWindowSize = 12 and polynomialOrder = 4
    int result = mes_savgolFilter(rawData, dataSize, halfWindowSize, filteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for center case";

    // Step 2: Compare the center points (around dataSize/2) with expectedOutput
    int centerStart = dataSize / 2 - 5;
    int centerEnd = dataSize / 2 + 5;
    for (int i = centerStart; i <= centerEnd; ++i)
    {
        EXPECT_TRUE(isClose(filteredData[i].phaseAngle, expectedOutput[i], 1e-5))
            << "Center case mismatch at index " << i << ": got " << filteredData[i].phaseAngle
            << ", expected " << expectedOutput[i];
    }
}

// Test: Leading Edge with Mocked Weights
TEST_F(SavitzkyGolayFilterTest, LeadingEdgeWithMockedWeights)
{
    // Step 1: Set the mock function using the static helper
    ComputeWeights_fake.custom_fake = MockComputeWeightsLeadingEdge;

    // Step 2: Run the filter
    int result = mes_savgolFilter(rawData, dataSize, halfWindowSize, filteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for leading edge with mocked weights";

    // Step 3: Compare the leading edge points with expectedOutput
    for (int i = 0; i < halfWindowSize; ++i)
    {
        EXPECT_TRUE(isClose(filteredData[i].phaseAngle, expectedOutput[i], 1e-5))
            << "Leading edge (mocked weights) mismatch at index " << i << ": got " << filteredData[i].phaseAngle
            << ", expected " << expectedOutput[i];
    }
}

// Test: Trailing Edge with Mocked Weights
TEST_F(SavitzkyGolayFilterTest, TrailingEdgeWithMockedWeights)
{
    // Step 1: Set the mock function using the static helper
    ComputeWeights_fake.custom_fake = MockComputeWeightsTrailingEdge;

    // Step 2: Run the filter
    int result = mes_savgolFilter(rawData, dataSize, halfWindowSize, filteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for trailing edge with mocked weights";

    // Step 3: Compare the trailing edge points with expectedOutput
    for (int i = dataSize - halfWindowSize; i < static_cast<int>(dataSize); ++i)
    {
        EXPECT_TRUE(isClose(filteredData[i].phaseAngle, expectedOutput[i], 1e-5))
            << "Trailing edge (mocked weights) mismatch at index " << i << ": got " << filteredData[i].phaseAngle
            << ", expected " << expectedOutput[i];
    }
}

// Test: Center Case with Mocked Weights
TEST_F(SavitzkyGolayFilterTest, CenterCaseWithMockedWeights)
{
    // Step 1: Set the mock function using the static helper
    ComputeWeights_fake.custom_fake = MockComputeWeightsCenterCase;

    // Step 2: Run the filter
    int result = mes_savgolFilter(rawData, dataSize, halfWindowSize, filteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for center case with mocked weights";

    // Step 3: Compare the center points with expectedOutput
    int centerStart = dataSize / 2 - 5;
    int centerEnd = dataSize / 2 + 5;
    for (int i = centerStart; i <= centerEnd; ++i)
    {
        EXPECT_TRUE(isClose(filteredData[i].phaseAngle, expectedOutput[i], 1e-5))
            << "Center case (mocked weights) mismatch at index " << i << ": got " << filteredData[i].phaseAngle
            << ", expected " << expectedOutput[i];
    }
}

// Test: Invalid Parameters (same as before)
TEST_F(SavitzkyGolayFilterTest, InvalidParameters)
{
    // Step 1: Test with NULL input data pointer
    int result = mes_savgolFilter(nullptr, dataSize, halfWindowSize, filteredData, polynomialOrder, targetPoint, derivativeOrder);
    EXPECT_EQ(result, -1) << "Expected failure for NULL input data";

    // Step 2: Test with NULL filtered data pointer
    result = mes_savgolFilter(rawData, dataSize, halfWindowSize, nullptr, polynomialOrder, targetPoint, derivativeOrder);
    EXPECT_EQ(result, -1) << "Expected failure for NULL filtered data";

    // Step 3: Test with halfWindowSize = 0
    result = mes_savgolFilter(rawData, dataSize, 0, filteredData, polynomialOrder, targetPoint, derivativeOrder);
    EXPECT_EQ(result, -2) << "Expected failure for halfWindowSize = 0";

    // Step 4: Test with invalid polynomial order (too large)
    result = mes_savgolFilter(rawData, dataSize, halfWindowSize, filteredData, 25, targetPoint, derivativeOrder);
    EXPECT_EQ(result, -2) << "Expected failure for invalid polynomial order";
}

// Test: Enhanced Memoization Functionality
TEST_F(SavitzkyGolayFilterTest, MemoizationWorks)
{
    // Step 1: Set up a small dataset and parameters to test memoization
    halfWindowSize = 5; // Window size = 11
    polynomialOrder = 2;
    targetPoint = 0;
    derivativeOrder = 0;
    windowSize = 2 * halfWindowSize + 1;

    size_t smallDataSize = windowSize;
    MqsRawDataPoint_t smallRawData[windowSize];
    MqsRawDataPoint_t smallFilteredData[windowSize];
    for (size_t i = 0; i < smallDataSize; ++i)
    {
        smallRawData[i].phaseAngle = 1.0f;
        smallFilteredData[i].phaseAngle = 0.0f;
    }

    // Step 2: Mock GramPolyIterative to return a predictable value based on parameters
    // This allows us to verify the cached value matches the computed value
    GramPolyIterative_fake.custom_fake = [](uint8_t polyOrder, int dataIndex, const GramPolyContext *ctx) -> float
    {
        // Return a unique value based on the parameters to make verification straightforward
        return static_cast<float>(polyOrder * 100 + dataIndex * 10 + ctx->derivativeOrder);
    };

    // Step 3: Run the filter for the first time
    int result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed on first run";

    // Step 4: Verify that GramPolyIterative was called as expected
    int expectedCallsFirstRun = windowSize * (polynomialOrder + 1) * 2;
    EXPECT_EQ(GramPolyIterative_fake.call_count, expectedCallsFirstRun)
        << "Expected GramPolyIterative to be called for each unique computation on first run";

    // Step 5: Verify the cache contents match the computed values
    // Test a few specific parameter combinations
    for (int k = 0; k <= polynomialOrder; ++k)
    {
        for (int dataIndex = -halfWindowSize; dataIndex <= halfWindowSize; ++dataIndex)
        {
            int shiftedIndex = dataIndex + halfWindowSize;
            const GramPolyCacheEntry *entry = GetGramPolyCacheEntry(shiftedIndex, k, derivativeOrder);
            ASSERT_NE(entry, nullptr) << "Cache entry should exist for shiftedIndex=" << shiftedIndex << ", k=" << k;
            EXPECT_TRUE(entry->isComputed) << "Cache entry should be computed for shiftedIndex=" << shiftedIndex << ", k=" << k;

            // Compute the expected value from GramPolyIterative
            float expectedValue = static_cast<float>(k * 100 + dataIndex * 10 + derivativeOrder);
            EXPECT_TRUE(isClose(entry->value, expectedValue, 1e-5))
                << "Cache value mismatch at shiftedIndex=" << shiftedIndex << ", k=" << k
                << ": got " << entry->value << ", expected " << expectedValue;
        }
    }

    // Step 6: Run the filter again with the same parameters to test cache reuse
    GramPolyIterative_fake.call_count = 0;
    result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed on second run";

    // Step 7: Verify that GramPolyIterative was not called (cache hit)
    EXPECT_EQ(GramPolyIterative_fake.call_count, 0)
        << "Expected no calls to GramPolyIterative due to memoization on second run";

    // Step 8: Change parameters to trigger cache clear and verify new values
    targetPoint = 1; // This should trigger ClearGramPolyCache
    GramPolyIterative_fake.call_count = 0;
    result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed after changing parameters";

    // Step 9: Verify that GramPolyIterative was called again after cache clear
    EXPECT_EQ(GramPolyIterative_fake.call_count, expectedCallsFirstRun)
        << "Expected GramPolyIterative to be called again after cache clear";

    // Step 10: Verify the new cache contents
    for (int k = 0; k <= polynomialOrder; ++k)
    {
        for (int dataIndex = -halfWindowSize; dataIndex <= halfWindowSize; ++dataIndex)
        {
            int shiftedIndex = dataIndex + halfWindowSize;
            const GramPolyCacheEntry *entry = GetGramPolyCacheEntry(shiftedIndex, k, derivativeOrder);
            ASSERT_NE(entry, nullptr) << "Cache entry should exist for shiftedIndex=" << shiftedIndex << ", k=" << k;
            EXPECT_TRUE(entry->isComputed) << "Cache entry should be computed for shiftedIndex=" << shiftedIndex << ", k=" << k;

            float expectedValue = static_cast<float>(k * 100 + dataIndex * 10 + derivativeOrder);
            EXPECT_TRUE(isClose(entry->value, expectedValue, 1e-5))
                << "Cache value mismatch after parameter change at shiftedIndex=" << shiftedIndex << ", k=" << k
                << ": got " << entry->value << ", expected " << expectedValue;
        }
    }
}

// Test: Memoization Cache Boundaries and Consistency
TEST_F(SavitzkyGolayFilterTest, MemoizationCacheBoundariesAndConsistency)
{
    // Step 1: Set parameters to the maximum cache limits
    halfWindowSize = 32; // MAX_HALF_WINDOW_FOR_MEMO = 32
    polynomialOrder = 4; // MAX_POLY_ORDER_FOR_MEMO = 5 (supports 0 to 4)
    derivativeOrder = 4; // MAX_DERIVATIVE_FOR_MEMO = 5 (supports 0 to 4)
    targetPoint = 0;
    windowSize = 2 * halfWindowSize + 1;

    size_t smallDataSize = windowSize;
    MqsRawDataPoint_t smallRawData[windowSize];
    MqsRawDataPoint_t smallFilteredData[windowSize];
    for (size_t i = 0; i < smallDataSize; ++i)
    {
        smallRawData[i].phaseAngle = 1.0f;
        smallFilteredData[i].phaseAngle = 0.0f;
    }

    // Step 2: Mock GramPolyIterative to return a predictable value
    GramPolyIterative_fake.custom_fake = [](uint8_t polyOrder, int dataIndex, const GramPolyContext *ctx) -> float
    {
        return static_cast<float>(polyOrder * 100 + dataIndex * 10 + ctx->derivativeOrder);
    };

    // Step 3: Run the filter for the first time at cache boundary
    int result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed at cache boundary";

    // Step 4: Verify that GramPolyIterative was called as expected
    int expectedCallsFirstRun = windowSize * (polynomialOrder + 1) * 2;
    EXPECT_EQ(GramPolyIterative_fake.call_count, expectedCallsFirstRun)
        << "Expected GramPolyIterative to be called for each unique computation at cache boundary";

    // Step 5: Verify the cache contents at the boundaries
    // Test the edge cases: dataIndex = -32 and 32, polyOrder = 4, derivativeOrder = 4
    int testIndices[] = {-32, 0, 32};
    for (int dataIndex : testIndices)
    {
        int shiftedIndex = dataIndex + halfWindowSize;
        for (int k = 0; k <= polynomialOrder; ++k)
        {
            const GramPolyCacheEntry *entry = GetGramPolyCacheEntry(shiftedIndex, k, derivativeOrder);
            ASSERT_NE(entry, nullptr) << "Cache entry should exist for shiftedIndex=" << shiftedIndex << ", k=" << k;
            EXPECT_TRUE(entry->isComputed) << "Cache entry should be computed for shiftedIndex=" << shiftedIndex << ", k=" << k;

            float expectedValue = static_cast<float>(k * 100 + dataIndex * 10 + derivativeOrder);
            EXPECT_TRUE(isClose(entry->value, expectedValue, 1e-5))
                << "Cache value mismatch at boundary shiftedIndex=" << shiftedIndex << ", k=" << k
                << ": got " << entry->value << ", expected " << expectedValue;
        }
    }

    // Step 6: Test cache consistency by running the filter again with a different dataset
    for (size_t i = 0; i < smallDataSize; ++i)
    {
        smallRawData[i].phaseAngle = 2.0f; // Different dataset
        smallFilteredData[i].phaseAngle = 0.0f;
    }
    GramPolyIterative_fake.call_count = 0;
    result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed on second run with different dataset";

    // Step 7: Verify that GramPolyIterative was not called (cache hit)
    EXPECT_EQ(GramPolyIterative_fake.call_count, 0)
        << "Expected no calls to GramPolyIterative due to memoization on second run with different dataset";

    // Step 8: Verify the cache contents remain consistent
    for (int dataIndex : testIndices)
    {
        int shiftedIndex = dataIndex + halfWindowSize;
        for (int k = 0; k <= polynomialOrder; ++k)
        {
            const GramPolyCacheEntry *entry = GetGramPolyCacheEntry(shiftedIndex, k, derivativeOrder);
            ASSERT_NE(entry, nullptr) << "Cache entry should exist for shiftedIndex=" << shiftedIndex << ", k=" << k;
            EXPECT_TRUE(entry->isComputed) << "Cache entry should be computed for shiftedIndex=" << shiftedIndex << ", k=" << k;

            float expectedValue = static_cast<float>(k * 100 + dataIndex * 10 + derivativeOrder);
            EXPECT_TRUE(isClose(entry->value, expectedValue, 1e-5))
                << "Cache value mismatch after second run at shiftedIndex=" << shiftedIndex << ", k=" << k
                << ": got " << entry->value << ", expected " << expectedValue;
        }
    }
}

// Test: Different Derivative Orders (same as before)
TEST_F(SavitzkyGolayFilterTest, DerivativeOrders)
{
    const size_t smallDataSize = 7;
    MqsRawDataPoint_t smallRawData[smallDataSize];
    MqsRawDataPoint_t smallFilteredData[smallDataSize];

    for (size_t i = 0; i < smallDataSize; ++i)
    {
        smallRawData[i].phaseAngle = i * i;
        smallFilteredData[i].phaseAngle = 0.0f;
    }

    halfWindowSize = 2;
    polynomialOrder = 2;
    targetPoint = 0;

    derivativeOrder = 1;
    int result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for first derivative";

    float expectedFirstDerivative[] = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
    for (size_t i = halfWindowSize; i < smallDataSize - halfWindowSize; ++i)
    {
        EXPECT_TRUE(isClose(smallFilteredData[i].phaseAngle, expectedFirstDerivative[i], 1e-3))
            << "First derivative mismatch at index " << i << ": got " << smallFilteredData[i].phaseAngle
            << ", expected " << expectedFirstDerivative[i];
    }

    for (size_t i = 0; i < smallDataSize; ++i)
    {
        smallFilteredData[i].phaseAngle = 0.0f;
    }
    derivativeOrder = 2;
    result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for second derivative";

    float expectedSecondDerivative[] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    for (size_t i = halfWindowSize; i < smallDataSize - halfWindowSize; ++i)
    {
        EXPECT_TRUE(isClose(smallFilteredData[i].phaseAngle, expectedSecondDerivative[i], 1e-3))
            << "Second derivative mismatch at index " << i << ": got " << smallFilteredData[i].phaseAngle
            << ", expected " << expectedSecondDerivative[i];
    }
}

// Test: Different Window Sizes and Polynomial Orders (same as before)
TEST_F(SavitzkyGolayFilterTest, DifferentWindowSizesAndOrders)
{
    halfWindowSize = 5;
    polynomialOrder = 2;
    targetPoint = 0;
    derivativeOrder = 0;

    const size_t smallDataSize = 50;
    MqsRawDataPoint_t smallRawData[smallDataSize];
    MqsRawDataPoint_t smallFilteredData[smallDataSize];
    for (size_t i = 0; i < smallDataSize; ++i)
    {
        smallRawData[i].phaseAngle = dataset[i];
        smallFilteredData[i].phaseAngle = 0.0f;
    }

    int result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for smaller window size";

    float mockWeights[11];
    for (int i = 0; i < 11; ++i)
    {
        mockWeights[i] = 1.0f / 11.0f;
    }
    ComputeWeights_fake.custom_fake = [](uint8_t hws, uint16_t tp, uint8_t po, uint8_t d, float *weights)
    {
        for (int i = 0; i < 11; ++i)
        {
            weights[i] = 1.0f / 11.0f;
        }
    };
    for (int i = halfWindowSize; i < smallDataSize - halfWindowSize; i += 10)
    {
        float expectedSum = 0.0f;
        for (int j = 0; j < 11; ++j)
        {
            expectedSum += mockWeights[j] * smallRawData[i - halfWindowSize + j].phaseAngle;
        }
        EXPECT_TRUE(isClose(smallFilteredData[i].phaseAngle, expectedSum, 1e-3))
            << "Mismatch at index " << i << " for smaller window: got " << smallFilteredData[i].phaseAngle
            << ", expected " << expectedSum;
    }

    halfWindowSize = 20;
    polynomialOrder = 4;
    windowSize = 2 * halfWindowSize + 1;

    for (size_t i = 0; i < smallDataSize; ++i)
    {
        smallFilteredData[i].phaseAngle = 0.0f;
    }

    result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for larger window size";

    float mockWeightsLarge[41];
    for (int i = 0; i < 41; ++i)
    {
        mockWeightsLarge[i] = 1.0f / 41.0f;
    }
    ComputeWeights_fake.custom_fake = [](uint8_t hws, uint16_t tp, uint8_t po, uint8_t d, float *weights)
    {
        for (int i = 0; i < 41; ++i)
        {
            weights[i] = 1.0f / 41.0f;
        }
    };
    for (int i = halfWindowSize; i < smallDataSize - halfWindowSize; i += 10)
    {
        float expectedSum = 0.0f;
        for (int j = 0; j < 41; ++j)
        {
            expectedSum += mockWeightsLarge[j] * smallRawData[i - halfWindowSize + j].phaseAngle;
        }
        EXPECT_TRUE(isClose(smallFilteredData[i].phaseAngle, expectedSum, 1e-3))
            << "Mismatch at index " << i << " for larger window: got " << smallFilteredData[i].phaseAngle
            << ", expected " << expectedSum;
    }
}

// Test: Edge Case with Small Dataset (same as before)
TEST_F(SavitzkyGolayFilterTest, SmallDataset)
{
    halfWindowSize = 3;
    polynomialOrder = 2;
    targetPoint = 0;
    derivativeOrder = 0;

    const size_t smallDataSize = 7;
    MqsRawDataPoint_t smallRawData[smallDataSize];
    MqsRawDataPoint_t smallFilteredData[smallDataSize];

    for (size_t i = 0; i < smallDataSize; ++i)
    {
        smallRawData[i].phaseAngle = static_cast<float>(i + 1);
        smallFilteredData[i].phaseAngle = 0.0f;
    }

    int result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for small dataset";

    for (size_t i = 0; i < smallDataSize; ++i)
    {
        EXPECT_TRUE(isClose(smallFilteredData[i].phaseAngle, smallRawData[i].phaseAngle, 1e-3))
            << "Mismatch at index " << i << ": got " << smallFilteredData[i].phaseAngle
            << ", expected " << smallRawData[i].phaseAngle;
    }
}

// Test: Numerical Stability (same as before)
TEST_F(SavitzkyGolayFilterTest, NumericalStability)
{
    for (size_t i = 0; i < dataSize; ++i)
    {
        rawData[i].phaseAngle *= 1e6;
        filteredData[i].phaseAngle = 0.0f;
    }

    int result = mes_savgolFilter(rawData, dataSize, halfWindowSize, filteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for large values";

    for (size_t i = 0; i < dataSize; ++i)
    {
        EXPECT_FALSE(std::isnan(filteredData[i].phaseAngle)) << "NaN detected at index " << i;
        EXPECT_FALSE(std::isinf(filteredData[i].phaseAngle)) << "Inf detected at index " << i;
    }

    const size_t noiseDataSize = 50;
    MqsRawDataPoint_t noiseRawData[noiseDataSize];
    MqsRawDataPoint_t noiseFilteredData[noiseDataSize];

    for (size_t i = 0; i < noiseDataSize; ++i)
    {
        noiseRawData[i].phaseAngle = (i % 2 == 0) ? 1.0f : -1.0f;
        noiseFilteredData[i].phaseAngle = 0.0f;
    }

    halfWindowSize = 5;
    polynomialOrder = 2;
    targetPoint = 0;
    derivativeOrder = 0;

    result = mes_savgolFilter(noiseRawData, noiseDataSize, halfWindowSize, noiseFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for high-frequency noise";

    for (size_t i = halfWindowSize; i < noiseDataSize - halfWindowSize; ++i)
    {
        EXPECT_TRUE(fabs(noiseFilteredData[i].phaseAngle) < 0.5f)
            << "High-frequency noise not smoothed at index " << i << ": got " << noiseFilteredData[i].phaseAngle;
    }

    const size_t flatDataSize = 50;
    MqsRawDataPoint_t flatRawData[flatDataSize];
    MqsRawDataPoint_t flatFilteredData[flatDataSize];

    for (size_t i = 0; i < flatDataSize; ++i)
    {
        flatRawData[i].phaseAngle = 5.0f;
        flatFilteredData[i].phaseAngle = 0.0f;
    }

    result = mes_savgolFilter(flatRawData, flatDataSize, halfWindowSize, flatFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for flat dataset";

    for (size_t i = 0; i < flatDataSize; ++i)
    {
        EXPECT_TRUE(isClose(flatFilteredData[i].phaseAngle, 5.0f, 1e-5))
            << "Flat dataset not preserved at index " << i << ": got " << flatFilteredData[i].phaseAngle
            << ", expected 5.0";
    }
}

// Test: Target Point Variations (same as before)
TEST_F(SavitzkyGolayFilterTest, TargetPointVariations)
{
    halfWindowSize = 5;
    polynomialOrder = 2;
    derivativeOrder = 0;

    const size_t smallDataSize = 50;
    MqsRawDataPoint_t smallRawData[smallDataSize];
    MqsRawDataPoint_t smallFilteredData[smallDataSize];
    for (size_t i = 0; i < smallDataSize; ++i)
    {
        smallRawData[i].phaseAngle = dataset[i];
        smallFilteredData[i].phaseAngle = 0.0f;
    }

    targetPoint = 0;
    int result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for targetPoint = 0";

    float mockWeights[11];
    for (int i = 0; i < 11; ++i)
    {
        mockWeights[i] = 1.0f / 11.0f;
    }
    ComputeWeights_fake.custom_fake = [](uint8_t hws, uint16_t tp, uint8_t po, uint8_t d, float *weights)
    {
        for (int i = 0; i < 11; ++i)
        {
            weights[i] = 1.0f / 11.0f;
        }
    };

    for (int i = halfWindowSize; i < smallDataSize - halfWindowSize; i += 10)
    {
        float expectedSum = 0.0f;
        for (int j = 0; j < 11; ++j)
        {
            expectedSum += mockWeights[j] * smallRawData[i - halfWindowSize + j].phaseAngle;
        }
        EXPECT_TRUE(isClose(smallFilteredData[i].phaseAngle, expectedSum, 1e-3))
            << "Mismatch at index " << i << " for targetPoint = 0: got " << smallFilteredData[i].phaseAngle
            << ", expected " << expectedSum;
    }

    for (size_t i = 0; i < smallDataSize; ++i)
    {
        smallFilteredData[i].phaseAngle = 0.0f;
    }
    targetPoint = halfWindowSize;
    result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for targetPoint = halfWindowSize";

    for (int i = halfWindowSize; i < smallDataSize - halfWindowSize; i += 10)
    {
        float expectedSum = 0.0f;
        for (int j = 0; j < 11; ++j)
        {
            expectedSum += mockWeights[j] * smallRawData[i - halfWindowSize + j].phaseAngle;
        }
        EXPECT_TRUE(isClose(smallFilteredData[i].phaseAngle, expectedSum, 1e-3))
            << "Mismatch at index " << i << " for targetPoint = halfWindowSize: got " << smallFilteredData[i].phaseAngle
            << ", expected " << expectedSum;
    }
}

// Test: Cache Boundary Conditions
TEST_F(SavitzkyGolayFilterTest, CacheBoundaryConditions)
{
    // Use a small dataset for simplicity
    const size_t smallDataSize = 50;
    MqsRawDataPoint_t smallRawData[smallDataSize];
    MqsRawDataPoint_t smallFilteredData[smallDataSize];
    for (size_t i = 0; i < smallDataSize; ++i)
    {
        smallRawData[i].phaseAngle = dataset[i];
        smallFilteredData[i].phaseAngle = 0.0f;
    }

    // Mock GramPolyIterative to return a fixed value and count calls
    GramPolyIterative_fake.custom_fake = [](uint8_t polynomialOrder, int dataIndex, const GramPolyContext *ctx) -> float
    {
        return 1.0f;
    };

    // Test 1: Exceed MAX_HALF_WINDOW_FOR_MEMO (32)
    halfWindowSize = 33; // Exceeds MAX_HALF_WINDOW_FOR_MEMO = 32
    polynomialOrder = 4;
    targetPoint = 0;
    derivativeOrder = 0;
    windowSize = 2 * halfWindowSize + 1;

    int result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for exceeding MAX_HALF_WINDOW_FOR_MEMO";

    // Since halfWindowSize exceeds the cache limit, GramPolyIterative should be called directly
    int expectedCalls = windowSize * (polynomialOrder + 1) * 2; // Same as memoization test logic
    EXPECT_EQ(GramPolyIterative_fake.call_count, expectedCalls)
        << "Expected GramPolyIterative to be called directly for exceeding MAX_HALF_WINDOW_FOR_MEMO";

    // Test 2: Exceed MAX_POLY_ORDER_FOR_MEMO (5, supports 0 to 4)
    halfWindowSize = 5;
    polynomialOrder = 5; // Exceeds MAX_POLY_ORDER_FOR_MEMO = 5 (cache supports 0 to 4)
    windowSize = 2 * halfWindowSize + 1;

    for (size_t i = 0; i < smallDataSize; ++i)
    {
        smallFilteredData[i].phaseAngle = 0.0f;
    }
    GramPolyIterative_fake.call_count = 0;

    result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for exceeding MAX_POLY_ORDER_FOR_MEMO";

    expectedCalls = windowSize * (polynomialOrder + 1) * 2; // k from 0 to 5
    EXPECT_EQ(GramPolyIterative_fake.call_count, expectedCalls)
        << "Expected GramPolyIterative to be called directly for exceeding MAX_POLY_ORDER_FOR_MEMO";

    // Test 3: Exceed MAX_DERIVATIVE_FOR_MEMO (5, supports 0 to 4)
    polynomialOrder = 2;
    derivativeOrder = 5; // Exceeds MAX_DERIVATIVE_FOR_MEMO = 5 (cache supports 0 to 4)

    for (size_t i = 0; i < smallDataSize; ++i)
    {
        smallFilteredData[i].phaseAngle = 0.0f;
    }
    GramPolyIterative_fake.call_count = 0;

    result = mes_savgolFilter(smallRawData, smallDataSize, halfWindowSize, smallFilteredData, polynomialOrder, targetPoint, derivativeOrder);
    ASSERT_EQ(result, 0) << "Filter application failed for exceeding MAX_DERIVATIVE_FOR_MEMO";

    expectedCalls = windowSize * (polynomialOrder + 1) * 2; // k from 0 to 2
    EXPECT_EQ(GramPolyIterative_fake.call_count, expectedCalls)
        << "Expected GramPolyIterative to be called directly for exceeding MAX_DERIVATIVE_FOR_MEMO";

    // Verify the output is valid (no NaNs or infinities)
    for (size_t i = 0; i < smallDataSize; ++i)
    {
        EXPECT_FALSE(std::isnan(smallFilteredData[i].phaseAngle)) << "NaN detected at index " << i;
        EXPECT_FALSE(std::isinf(smallFilteredData[i].phaseAngle)) << "Inf detected at index " << i;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}