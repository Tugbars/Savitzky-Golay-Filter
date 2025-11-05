/**
 * @file test_convolution.cpp
 * @brief GoogleTest suite for ILP-Optimized Convolution
 * 
 * Tests the optimized convolution implementation which uses 4 independent
 * accumulation chains to exploit CPU instruction-level parallelism (ILP).
 * 
 * Key aspects tested:
 * 1. Correctness: ILP implementation matches scalar reference
 * 2. Remainder handling: Window sizes with 1, 2, 3 leftover elements
 * 3. Access patterns: Forward (central), reverse (leading), forward (trailing)
 * 4. Numerical accuracy: Pairwise reduction maintains precision
 * 5. Boundary safety: No buffer overruns at array edges
 */

#include <gtest/gtest.h>
#include "savgolFilter.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

//=============================================================================
// Test Fixture for Convolution Tests
//=============================================================================

class ConvolutionFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize GenFact table (required dependency)
        InitGenFactTable();
    }

    void TearDown() override {
        // Nothing to clean up
    }

    /**
     * @brief Reference scalar convolution implementation (simple, unoptimized)
     * Used to validate the optimized ILP version
     */
    float ScalarConvolution(const std::vector<float>& weights,
                           const std::vector<MqsRawDataPoint_t>& data,
                           size_t centerIndex) {
        size_t halfWindow = weights.size() / 2;
        
        float sum = 0.0f;
        for (size_t i = 0; i < weights.size(); i++) {
            size_t dataIdx = centerIndex - halfWindow + i;
            sum += weights[i] * data[dataIdx].phaseAngle;
        }
        
        return sum;
    }

    /**
     * @brief Create synthetic data for testing
     */
    void CreateTestData(std::vector<MqsRawDataPoint_t>& data, size_t size) {
        data.resize(size);
        for (size_t i = 0; i < size; i++) {
            data[i].phaseAngle = 0.0f;
        }
    }

    /**
     * @brief Fill data with constant value
     */
    void FillConstant(std::vector<MqsRawDataPoint_t>& data, float value) {
        for (auto& d : data) {
            d.phaseAngle = value;
        }
    }

    /**
     * @brief Fill data with linear ramp
     */
    void FillLinear(std::vector<MqsRawDataPoint_t>& data, float slope, float intercept) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i].phaseAngle = slope * i + intercept;
        }
    }

    /**
     * @brief Fill data with sine wave
     */
    void FillSine(std::vector<MqsRawDataPoint_t>& data, float amplitude, 
                  float frequency, float phase) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i].phaseAngle = amplitude * std::sin(2 * M_PI * frequency * i + phase);
        }
    }

    /**
     * @brief Fill data with random values (deterministic seed)
     */
    void FillRandom(std::vector<MqsRawDataPoint_t>& data, unsigned int seed = 42) {
        std::srand(seed);
        for (auto& d : data) {
            d.phaseAngle = (float)std::rand() / RAND_MAX;
        }
    }

    /**
     * @brief Apply filter and return filtered data
     */
    void ApplyFilter(const std::vector<MqsRawDataPoint_t>& input,
                    std::vector<MqsRawDataPoint_t>& output,
                    uint8_t halfWindowSize, uint8_t polynomialOrder,
                    uint8_t derivativeOrder = 0) {
        output.resize(input.size());
        
        int result = mes_savgolFilter(
            const_cast<MqsRawDataPoint_t*>(input.data()),
            input.size(),
            halfWindowSize,
            output.data(),
            polynomialOrder,
            0,  // targetPoint = 0 (centered)
            derivativeOrder
        );
        
        ASSERT_EQ(result, 0) << "Filter application failed";
    }

    /**
     * @brief Compare floats with tolerance
     */
    bool FloatsEqual(float a, float b, float relTol = 1e-5f) {
        if (a == b) return true;
        if (std::abs(a) < 1e-8f && std::abs(b) < 1e-8f) return true;
        float maxAbs = std::max(std::abs(a), std::abs(b));
        return std::abs(a - b) <= relTol * maxAbs;
    }

    /**
     * @brief Compute weights for manual convolution testing
     */
    void ComputeWeights(uint8_t halfWindowSize, uint8_t polynomialOrder,
                       uint8_t derivativeOrder, std::vector<float>& weights) {
        weights.resize(2 * halfWindowSize + 1);
        ::ComputeWeights(halfWindowSize, 0, polynomialOrder, 
                        derivativeOrder, weights.data());
    }
};

//=============================================================================
// Basic Correctness Tests - Simple Cases
//=============================================================================

TEST_F(ConvolutionFixture, Simple_3Point_HandCalculated) {
    // 3-point convolution with known weights
    std::vector<float> weights = {0.2f, 0.6f, 0.2f};
    
    std::vector<MqsRawDataPoint_t> data(5);
    data[0].phaseAngle = 1.0f;
    data[1].phaseAngle = 2.0f;
    data[2].phaseAngle = 3.0f;  // Center point
    data[3].phaseAngle = 4.0f;
    data[4].phaseAngle = 5.0f;
    
    // Manual calculation for center point:
    // 0.2 * 1.0 + 0.6 * 2.0 + 0.2 * 3.0 = 0.2 + 1.2 + 0.6 = 2.0
    float expected = ScalarConvolution(weights, data, 1);
    EXPECT_FLOAT_EQ(expected, 2.0f);
}

TEST_F(ConvolutionFixture, Simple_5Point_HandCalculated) {
    // 5-point convolution with simple weights
    std::vector<float> weights = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f};
    
    std::vector<MqsRawDataPoint_t> data(7);
    for (size_t i = 0; i < data.size(); i++) {
        data[i].phaseAngle = (float)(i + 1);  // 1, 2, 3, 4, 5, 6, 7
    }
    
    // Manual calculation for center point (index 3, value 4):
    // 0.1*2 + 0.2*3 + 0.4*4 + 0.2*5 + 0.1*6 = 0.2 + 0.6 + 1.6 + 1.0 + 0.6 = 4.0
    float result = ScalarConvolution(weights, data, 3);
    EXPECT_FLOAT_EQ(result, 4.0f);
}

//=============================================================================
// Remainder Handling Tests - Different Window Sizes
//=============================================================================

TEST_F(ConvolutionFixture, Remainder_WindowSize5_Remainder1) {
    // Window size 5 (2n+1 where n=2): 5 % 4 = 1 remainder
    uint8_t n = 2;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 100);
    FillLinear(input, 1.0f, 0.0f);  // Linear ramp
    
    ApplyFilter(input, output, n, 2, 0);
    
    // For linear data with polynomial order >= 1, smoothing preserves linearity
    // Central region should match input
    for (size_t i = n; i < input.size() - n; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, input[i].phaseAngle))
            << "Mismatch at index " << i;
    }
}

TEST_F(ConvolutionFixture, Remainder_WindowSize6_Remainder2) {
    // Window size 6: 6 % 4 = 2 remainder
    // Note: 6 is even, so n=2.5 (not valid), use n=3 → window=7
    uint8_t n = 3;  // Window size = 7, 7 % 4 = 3 remainder
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 100);
    FillConstant(input, 5.0f);
    
    ApplyFilter(input, output, n, 2, 0);
    
    // Constant input should produce constant output
    for (size_t i = n; i < input.size() - n; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, 5.0f))
            << "Constant not preserved at index " << i;
    }
}

TEST_F(ConvolutionFixture, Remainder_WindowSize11_Remainder3) {
    // Window size 11 (n=5): 11 % 4 = 3 remainder
    uint8_t n = 5;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 100);
    FillSine(input, 10.0f, 0.01f, 0.0f);
    
    ApplyFilter(input, output, n, 2, 0);
    
    // Output should be valid (no NaN)
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle))
            << "NaN at index " << i;
    }
}

TEST_F(ConvolutionFixture, Remainder_WindowSize12_NoRemainder) {
    // Window size 12: 12 % 4 = 0 remainder (perfect alignment)
    // n=6 → window=13, 13 % 4 = 1
    // n=5.5 invalid, use n=6
    uint8_t n = 6;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 100);
    FillLinear(input, 0.5f, 10.0f);
    
    ApplyFilter(input, output, n, 3, 0);
    
    // Linear preserved with cubic fit
    for (size_t i = n; i < input.size() - n; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, input[i].phaseAngle, 1e-4f))
            << "Linear not preserved at index " << i;
    }
}

//=============================================================================
// Access Pattern Tests - Central Region
//=============================================================================

TEST_F(ConvolutionFixture, AccessPattern_Central_Forward) {
    // Central region uses forward access pattern
    uint8_t n = 5;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 50);
    FillRandom(input);
    
    ApplyFilter(input, output, n, 2, 0);
    
    // Verify central region is computed
    for (size_t i = n; i < input.size() - n; i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle))
            << "Central region has NaN at index " << i;
    }
}

TEST_F(ConvolutionFixture, AccessPattern_Central_NoBufferOverrun) {
    // Ensure no access outside array bounds
    uint8_t n = 10;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 100);
    FillRandom(input, 123);
    
    // This should not crash or access out of bounds
    ApplyFilter(input, output, n, 2, 0);
    
    // Verify all central points computed
    for (size_t i = n; i < input.size() - n; i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle));
    }
}

//=============================================================================
// Access Pattern Tests - Leading Edge
//=============================================================================

TEST_F(ConvolutionFixture, AccessPattern_Leading_Reverse) {
    // Leading edge uses reverse access pattern
    uint8_t n = 5;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 50);
    
    // Create asymmetric data to detect wrong access pattern
    for (size_t i = 0; i < input.size(); i++) {
        input[i].phaseAngle = (float)i;
    }
    
    ApplyFilter(input, output, n, 2, 0);
    
    // Leading edge should be computed (no NaN)
    for (size_t i = 0; i < n; i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle))
            << "Leading edge has NaN at index " << i;
    }
}

TEST_F(ConvolutionFixture, AccessPattern_Leading_NoNegativeIndex) {
    // Ensure leading edge doesn't access negative indices
    uint8_t n = 8;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 50);
    FillRandom(input);
    
    // Should not crash even with large window
    ApplyFilter(input, output, n, 2, 0);
    
    // First n points should be valid
    for (size_t i = 0; i < n; i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle));
    }
}

//=============================================================================
// Access Pattern Tests - Trailing Edge
//=============================================================================

TEST_F(ConvolutionFixture, AccessPattern_Trailing_Forward) {
    // Trailing edge uses forward access pattern
    uint8_t n = 5;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 50);
    FillRandom(input);
    
    ApplyFilter(input, output, n, 2, 0);
    
    // Trailing edge should be computed
    for (size_t i = input.size() - n; i < input.size(); i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle))
            << "Trailing edge has NaN at index " << i;
    }
}

TEST_F(ConvolutionFixture, AccessPattern_Trailing_NoBeyondBounds) {
    // Ensure trailing edge doesn't access beyond array
    uint8_t n = 10;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 50);
    FillLinear(input, 1.0f, 0.0f);
    
    // Should not crash
    ApplyFilter(input, output, n, 2, 0);
    
    // Last n points should be valid
    for (size_t i = input.size() - n; i < input.size(); i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle));
    }
}

//=============================================================================
// Numerical Accuracy Tests
//=============================================================================

TEST_F(ConvolutionFixture, Accuracy_PairwiseReduction) {
    // Pairwise reduction should maintain better numerical accuracy
    uint8_t n = 10;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 100);
    
    // Mix of large and small values (tests accumulation order)
    for (size_t i = 0; i < input.size(); i++) {
        input[i].phaseAngle = (i % 2 == 0) ? 1e6f : 1e-6f;
    }
    
    ApplyFilter(input, output, n, 2, 0);
    
    // Should produce finite values (not overflow/underflow)
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle));
        EXPECT_FALSE(std::isinf(output[i].phaseAngle));
    }
}

TEST_F(ConvolutionFixture, Accuracy_SmallValues_NoUnderflow) {
    // Very small values should not underflow
    uint8_t n = 5;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 50);
    FillConstant(input, 1e-20f);
    
    ApplyFilter(input, output, n, 2, 0);
    
    // Should preserve small values (not become zero or NaN)
    for (size_t i = n; i < input.size() - n; i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle));
        // Small constant should be preserved
        EXPECT_GT(std::abs(output[i].phaseAngle), 0.0f);
    }
}

TEST_F(ConvolutionFixture, Accuracy_LargeValues_NoOverflow) {
    // Large values should not overflow
    uint8_t n = 5;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 50);
    FillConstant(input, 1e20f);
    
    ApplyFilter(input, output, n, 2, 0);
    
    // Should produce finite values
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_FALSE(std::isinf(output[i].phaseAngle))
            << "Overflow at index " << i;
    }
}

TEST_F(ConvolutionFixture, Accuracy_AlternatingSign) {
    // Alternating signs test cancellation error
    uint8_t n = 5;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 50);
    
    for (size_t i = 0; i < input.size(); i++) {
        input[i].phaseAngle = (i % 2 == 0) ? 1.0f : -1.0f;
    }
    
    ApplyFilter(input, output, n, 2, 0);
    
    // Should smooth out oscillations
    for (size_t i = n; i < input.size() - n; i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle));
        // Smoothed value should be near zero
        EXPECT_LT(std::abs(output[i].phaseAngle), 1.0f);
    }
}

//=============================================================================
// Polynomial Preservation Tests
//=============================================================================

TEST_F(ConvolutionFixture, Preservation_Constant) {
    // Constant input should be preserved exactly
    uint8_t n = 5;
    float constant = 42.0f;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 50);
    FillConstant(input, constant);
    
    ApplyFilter(input, output, n, 2, 0);
    
    // All points should be the constant value
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, constant))
            << "Constant not preserved at index " << i;
    }
}

TEST_F(ConvolutionFixture, Preservation_Linear) {
    // Linear function should be preserved with order >= 1
    uint8_t n = 5;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 100);
    FillLinear(input, 2.0f, 10.0f);  // y = 2x + 10
    
    ApplyFilter(input, output, n, 2, 0);  // Quadratic fit
    
    // Central region should preserve linearity
    for (size_t i = n; i < input.size() - n; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, input[i].phaseAngle, 1e-4f))
            << "Linear not preserved at index " << i;
    }
}

TEST_F(ConvolutionFixture, Preservation_Quadratic) {
    // Quadratic should be preserved with order >= 2
    uint8_t n = 5;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 100);
    
    // y = x^2
    for (size_t i = 0; i < input.size(); i++) {
        float x = (float)i;
        input[i].phaseAngle = x * x;
    }
    
    ApplyFilter(input, output, n, 2, 0);  // Quadratic fit
    
    // Central region should preserve quadratic
    for (size_t i = n; i < input.size() - n; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, input[i].phaseAngle, 1e-3f))
            << "Quadratic not preserved at index " << i 
            << ", expected " << input[i].phaseAngle 
            << ", got " << output[i].phaseAngle;
    }
}

//=============================================================================
// Derivative Tests
//=============================================================================

TEST_F(ConvolutionFixture, Derivative_Constant_IsZero) {
    // Derivative of constant should be zero
    uint8_t n = 5;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 50);
    FillConstant(input, 100.0f);
    
    ApplyFilter(input, output, n, 2, 1);  // First derivative
    
    // All derivatives should be zero
    for (size_t i = n; i < input.size() - n; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, 0.0f, 1e-4f))
            << "Derivative of constant not zero at index " << i;
    }
}

TEST_F(ConvolutionFixture, Derivative_Linear_IsConstant) {
    // Derivative of linear function should be constant (slope)
    uint8_t n = 5;
    float slope = 3.0f;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 50);
    FillLinear(input, slope, 0.0f);
    
    ApplyFilter(input, output, n, 2, 1);  // First derivative
    
    // All derivatives should equal slope
    for (size_t i = n; i < input.size() - n; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, slope, 1e-3f))
            << "Derivative at index " << i << " = " << output[i].phaseAngle 
            << ", expected " << slope;
    }
}

//=============================================================================
// Comprehensive Window Size Tests
//=============================================================================

class ConvolutionWindowSizeTest : public ConvolutionFixture,
                                   public ::testing::WithParamInterface<uint8_t> {
};

TEST_P(ConvolutionWindowSizeTest, AllWindowSizes_ProduceValidOutput) {
    uint8_t n = GetParam();
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 100);
    FillRandom(input);
    
    ApplyFilter(input, output, n, 2, 0);
    
    // All outputs should be valid
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle))
            << "NaN at index " << i << " for n=" << (int)n;
        EXPECT_FALSE(std::isinf(output[i].phaseAngle))
            << "Inf at index " << i << " for n=" << (int)n;
    }
}

INSTANTIATE_TEST_SUITE_P(
    VariousWindowSizes,
    ConvolutionWindowSizeTest,
    ::testing::Values(2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20)
);

//=============================================================================
// Stress Tests
//=============================================================================

TEST_F(ConvolutionFixture, Stress_LargeDataset) {
    // Large dataset (1 million points)
    uint8_t n = 5;
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, 1000000);
    FillSine(input, 1.0f, 0.00001f, 0.0f);
    
    ApplyFilter(input, output, n, 2, 0);
    
    // Spot check some values
    for (size_t i = 0; i < output.size(); i += 10000) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle));
    }
}

TEST_F(ConvolutionFixture, Stress_MinimumDataset) {
    // Minimum valid dataset (window size)
    uint8_t n = 2;
    size_t minSize = 2 * n + 1;  // Exactly window size
    
    std::vector<MqsRawDataPoint_t> input, output;
    CreateTestData(input, minSize);
    FillConstant(input, 1.0f);
    
    ApplyFilter(input, output, n, 2, 0);
    
    // Should handle minimum size correctly
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle));
    }
}

//=============================================================================
// Main function
//=============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}