/**
 * @file test_filter_application.cpp
 * @brief GoogleTest suite for End-to-End Savitzky-Golay Filter Application
 * 
 * Tests the complete filter pipeline from input to output, including:
 * - Signal preservation properties (constant, linear, polynomial)
 * - Derivative computation accuracy
 * - Noise smoothing effectiveness
 * - Edge handling (leading, trailing, transitions)
 * - Various filter configurations
 * - Real-world use cases
 * 
 * This is integration testing - we're validating that all components
 * (GenFact, GramPoly, Weights, Cache, Convolution) work together correctly.
 */

#include <gtest/gtest.h>
#include "savgolFilter.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

//=============================================================================
// Test Fixture for Filter Application Tests
//=============================================================================

class FilterApplicationFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize required components
        InitGenFactTable();
    }

    void TearDown() override {
        // Nothing to clean up
    }

    /**
     * @brief Apply Savitzky-Golay filter
     */
    int ApplyFilter(const std::vector<MqsRawDataPoint_t>& input,
                    std::vector<MqsRawDataPoint_t>& output,
                    uint8_t halfWindowSize,
                    uint8_t polynomialOrder,
                    uint8_t targetPoint,
                    uint8_t derivativeOrder) {
        output.resize(input.size());
        
        return mes_savgolFilter(
            const_cast<MqsRawDataPoint_t*>(input.data()),
            input.size(),
            halfWindowSize,
            output.data(),
            polynomialOrder,
            targetPoint,
            derivativeOrder
        );
    }

    /**
     * @brief Create test data array
     */
    void CreateData(std::vector<MqsRawDataPoint_t>& data, size_t size) {
        data.resize(size);
        for (size_t i = 0; i < size; i++) {
            data[i].phaseAngle = 0.0f;
        }
    }

    /**
     * @brief Fill with constant value
     */
    void FillConstant(std::vector<MqsRawDataPoint_t>& data, float value) {
        for (auto& d : data) {
            d.phaseAngle = value;
        }
    }

    /**
     * @brief Fill with linear function: y = slope * x + intercept
     */
    void FillLinear(std::vector<MqsRawDataPoint_t>& data, float slope, float intercept) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i].phaseAngle = slope * i + intercept;
        }
    }

    /**
     * @brief Fill with quadratic: y = a*x^2 + b*x + c
     */
    void FillQuadratic(std::vector<MqsRawDataPoint_t>& data, float a, float b, float c) {
        for (size_t i = 0; i < data.size(); i++) {
            float x = (float)i;
            data[i].phaseAngle = a * x * x + b * x + c;
        }
    }

    /**
     * @brief Fill with cubic: y = a*x^3 + b*x^2 + c*x + d
     */
    void FillCubic(std::vector<MqsRawDataPoint_t>& data, float a, float b, float c, float d) {
        for (size_t i = 0; i < data.size(); i++) {
            float x = (float)i;
            data[i].phaseAngle = a * x * x * x + b * x * x + c * x + d;
        }
    }

    /**
     * @brief Fill with sine wave
     */
    void FillSine(std::vector<MqsRawDataPoint_t>& data, float amplitude, 
                  float frequency, float phase) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i].phaseAngle = amplitude * std::sin(2 * M_PI * frequency * i + phase);
        }
    }

    /**
     * @brief Add Gaussian noise
     */
    void AddNoise(std::vector<MqsRawDataPoint_t>& data, float stddev, unsigned int seed = 42) {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, stddev);
        
        for (auto& d : data) {
            d.phaseAngle += dist(gen);
        }
    }

    /**
     * @brief Add uniform noise
     */
    void AddUniformNoise(std::vector<MqsRawDataPoint_t>& data, float amplitude, 
                        unsigned int seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-amplitude, amplitude);
        
        for (auto& d : data) {
            d.phaseAngle += dist(gen);
        }
    }

    /**
     * @brief Compute impulse response
     */
    void ComputeImpulseResponse(std::vector<float>& response, 
                               uint8_t halfWindowSize,
                               uint8_t polynomialOrder,
                               uint8_t derivativeOrder) {
        size_t size = 2 * halfWindowSize + 1;
        std::vector<MqsRawDataPoint_t> input, output;
        CreateData(input, size * 3);  // Large enough for edges
        
        // Place impulse at center
        size_t centerIdx = size + halfWindowSize;
        FillConstant(input, 0.0f);
        input[centerIdx].phaseAngle = 1.0f;
        
        ApplyFilter(input, output, halfWindowSize, polynomialOrder, 0, derivativeOrder);
        
        // Extract response around impulse
        response.resize(size);
        for (size_t i = 0; i < size; i++) {
            response[i] = output[centerIdx - halfWindowSize + i].phaseAngle;
        }
    }

    /**
     * @brief Compute RMS difference between two signals
     */
    float ComputeRMS(const std::vector<MqsRawDataPoint_t>& a,
                    const std::vector<MqsRawDataPoint_t>& b,
                    size_t startIdx = 0,
                    size_t endIdx = 0) {
        if (endIdx == 0) endIdx = a.size();
        
        float sumSq = 0.0f;
        size_t count = 0;
        
        for (size_t i = startIdx; i < endIdx && i < a.size() && i < b.size(); i++) {
            float diff = a[i].phaseAngle - b[i].phaseAngle;
            sumSq += diff * diff;
            count++;
        }
        
        return std::sqrt(sumSq / count);
    }

    /**
     * @brief Compute Signal-to-Noise Ratio
     */
    float ComputeSNR(const std::vector<MqsRawDataPoint_t>& signal,
                    const std::vector<MqsRawDataPoint_t>& noisy) {
        float signalPower = 0.0f;
        float noisePower = 0.0f;
        
        for (size_t i = 0; i < signal.size() && i < noisy.size(); i++) {
            signalPower += signal[i].phaseAngle * signal[i].phaseAngle;
            float noise = noisy[i].phaseAngle - signal[i].phaseAngle;
            noisePower += noise * noise;
        }
        
        return 10.0f * std::log10(signalPower / noisePower);
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
};

//=============================================================================
// Basic Functionality Tests - Signal Preservation
//=============================================================================

TEST_F(FilterApplicationFixture, Constant_PreservedExactly) {
    // Constant signal should pass through unchanged
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    FillConstant(input, 42.5f);
    
    int result = ApplyFilter(input, output, 5, 2, 0, 0);
    ASSERT_EQ(result, 0);
    
    // All points should equal the constant
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_FLOAT_EQ(output[i].phaseAngle, 42.5f)
            << "Constant not preserved at index " << i;
    }
}

TEST_F(FilterApplicationFixture, Linear_PreservedWithOrder1) {
    // Linear function preserved with polynomial order >= 1
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    FillLinear(input, 2.5f, 10.0f);  // y = 2.5x + 10
    
    int result = ApplyFilter(input, output, 5, 2, 0, 0);
    ASSERT_EQ(result, 0);
    
    // Check central region (edges may have small errors)
    for (size_t i = 5; i < 95; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, input[i].phaseAngle, 1e-4f))
            << "Linear not preserved at index " << i
            << ", expected " << input[i].phaseAngle
            << ", got " << output[i].phaseAngle;
    }
}

TEST_F(FilterApplicationFixture, Quadratic_PreservedWithOrder2) {
    // Quadratic preserved with polynomial order >= 2
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    FillQuadratic(input, 0.1f, 2.0f, 5.0f);  // y = 0.1x^2 + 2x + 5
    
    int result = ApplyFilter(input, output, 5, 2, 0, 0);
    ASSERT_EQ(result, 0);
    
    // Central region should preserve quadratic
    for (size_t i = 10; i < 90; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, input[i].phaseAngle, 1e-3f))
            << "Quadratic not preserved at index " << i;
    }
}

TEST_F(FilterApplicationFixture, Cubic_PreservedWithOrder3) {
    // Cubic preserved with polynomial order >= 3
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    FillCubic(input, 0.001f, 0.1f, 1.0f, 5.0f);  // y = 0.001x^3 + 0.1x^2 + x + 5
    
    int result = ApplyFilter(input, output, 5, 3, 0, 0);
    ASSERT_EQ(result, 0);
    
    // Central region should preserve cubic
    for (size_t i = 10; i < 90; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, input[i].phaseAngle, 1e-2f))
            << "Cubic not preserved at index " << i;
    }
}

TEST_F(FilterApplicationFixture, Sine_PreservedWithMinimalDistortion) {
    // Sine wave should be smoothed with minimal distortion
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 200);
    FillSine(input, 10.0f, 0.05f, 0.0f);  // Low frequency sine
    
    int result = ApplyFilter(input, output, 5, 2, 0, 0);
    ASSERT_EQ(result, 0);
    
    // Low frequency should pass through with minimal attenuation
    float rms = ComputeRMS(input, output, 10, 190);
    EXPECT_LT(rms, 0.5f) << "Sine wave distorted too much, RMS = " << rms;
}

//=============================================================================
// Derivative Tests - First Derivative
//=============================================================================

TEST_F(FilterApplicationFixture, FirstDerivative_Constant_IsZero) {
    // d/dx(constant) = 0
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    FillConstant(input, 100.0f);
    
    int result = ApplyFilter(input, output, 5, 2, 0, 1);
    ASSERT_EQ(result, 0);
    
    // Central region should be zero
    for (size_t i = 10; i < 90; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, 0.0f, 1e-4f))
            << "Derivative of constant not zero at index " << i
            << ", got " << output[i].phaseAngle;
    }
}

TEST_F(FilterApplicationFixture, FirstDerivative_Linear_IsSlope) {
    // d/dx(mx + b) = m
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    float slope = 3.5f;
    FillLinear(input, slope, 10.0f);
    
    int result = ApplyFilter(input, output, 5, 2, 0, 1);
    ASSERT_EQ(result, 0);
    
    // Central region should equal slope
    for (size_t i = 10; i < 90; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, slope, 1e-3f))
            << "First derivative at index " << i
            << " = " << output[i].phaseAngle
            << ", expected " << slope;
    }
}

TEST_F(FilterApplicationFixture, FirstDerivative_Quadratic_IsLinear) {
    // d/dx(ax^2 + bx + c) = 2ax + b
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    float a = 0.2f, b = 1.0f, c = 5.0f;
    FillQuadratic(input, a, b, c);
    
    int result = ApplyFilter(input, output, 5, 2, 0, 1);
    ASSERT_EQ(result, 0);
    
    // Central region: derivative should be 2ax + b
    for (size_t i = 10; i < 90; i++) {
        float expected = 2 * a * i + b;
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, expected, 1e-2f))
            << "First derivative at index " << i
            << " = " << output[i].phaseAngle
            << ", expected " << expected;
    }
}

TEST_F(FilterApplicationFixture, FirstDerivative_Sine_IsCosine) {
    // d/dx(A*sin(ωx)) ≈ A*ω*cos(ωx)
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 200);
    float amplitude = 10.0f;
    float omega = 2 * M_PI * 0.02f;  // Low frequency
    FillSine(input, amplitude, 0.02f, 0.0f);
    
    int result = ApplyFilter(input, output, 10, 4, 0, 1);
    ASSERT_EQ(result, 0);
    
    // Check a few points in central region
    for (size_t i = 20; i < 180; i += 20) {
        float expected = amplitude * omega * std::cos(omega * i);
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, expected, 0.1f))
            << "Derivative of sine at index " << i;
    }
}

//=============================================================================
// Derivative Tests - Second Derivative
//=============================================================================

TEST_F(FilterApplicationFixture, SecondDerivative_Constant_IsZero) {
    // d²/dx²(constant) = 0
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    FillConstant(input, 50.0f);
    
    int result = ApplyFilter(input, output, 5, 3, 0, 2);
    ASSERT_EQ(result, 0);
    
    for (size_t i = 10; i < 90; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, 0.0f, 1e-4f))
            << "Second derivative of constant not zero at " << i;
    }
}

TEST_F(FilterApplicationFixture, SecondDerivative_Linear_IsZero) {
    // d²/dx²(mx + b) = 0
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    FillLinear(input, 2.0f, 5.0f);
    
    int result = ApplyFilter(input, output, 5, 3, 0, 2);
    ASSERT_EQ(result, 0);
    
    for (size_t i = 10; i < 90; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, 0.0f, 1e-3f))
            << "Second derivative of linear not zero at " << i;
    }
}

TEST_F(FilterApplicationFixture, SecondDerivative_Quadratic_IsConstant) {
    // d²/dx²(ax^2 + bx + c) = 2a
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    float a = 0.5f;
    FillQuadratic(input, a, 1.0f, 0.0f);
    
    int result = ApplyFilter(input, output, 5, 3, 0, 2);
    ASSERT_EQ(result, 0);
    
    float expected = 2 * a;
    for (size_t i = 10; i < 90; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, expected, 1e-2f))
            << "Second derivative at " << i
            << " = " << output[i].phaseAngle
            << ", expected " << expected;
    }
}

//=============================================================================
// Noise Handling Tests
//=============================================================================

TEST_F(FilterApplicationFixture, NoiseSmoothing_GaussianNoise) {
    // Filter should reduce Gaussian noise
    std::vector<MqsRawDataPoint_t> clean, noisy, filtered;
    CreateData(clean, 200);
    FillSine(clean, 10.0f, 0.02f, 0.0f);
    
    noisy = clean;
    AddNoise(noisy, 2.0f);  // Add noise with σ = 2.0
    
    int result = ApplyFilter(noisy, filtered, 5, 2, 0, 0);
    ASSERT_EQ(result, 0);
    
    // Filtered should be closer to clean than noisy
    float rmsNoisy = ComputeRMS(clean, noisy, 10, 190);
    float rmsFiltered = ComputeRMS(clean, filtered, 10, 190);
    
    EXPECT_LT(rmsFiltered, rmsNoisy)
        << "Filter did not reduce noise: noisy RMS = " << rmsNoisy
        << ", filtered RMS = " << rmsFiltered;
    
    // Should reduce noise by at least 30%
    EXPECT_LT(rmsFiltered, 0.7f * rmsNoisy)
        << "Noise reduction insufficient";
}

TEST_F(FilterApplicationFixture, NoiseSmoothing_UniformNoise) {
    // Filter should reduce uniform noise
    std::vector<MqsRawDataPoint_t> clean, noisy, filtered;
    CreateData(clean, 200);
    FillLinear(clean, 1.0f, 0.0f);
    
    noisy = clean;
    AddUniformNoise(noisy, 5.0f);
    
    int result = ApplyFilter(noisy, filtered, 5, 2, 0, 0);
    ASSERT_EQ(result, 0);
    
    float rmsNoisy = ComputeRMS(clean, noisy, 10, 190);
    float rmsFiltered = ComputeRMS(clean, filtered, 10, 190);
    
    EXPECT_LT(rmsFiltered, rmsNoisy);
}

TEST_F(FilterApplicationFixture, NoiseSmoothing_SNRImprovement) {
    // SNR should improve after filtering
    std::vector<MqsRawDataPoint_t> clean, noisy, filtered;
    CreateData(clean, 300);
    FillSine(clean, 5.0f, 0.03f, 0.0f);
    
    noisy = clean;
    AddNoise(noisy, 1.0f);
    
    int result = ApplyFilter(noisy, filtered, 8, 3, 0, 0);
    ASSERT_EQ(result, 0);
    
    float snrNoisy = ComputeSNR(clean, noisy);
    float snrFiltered = ComputeSNR(clean, filtered);
    
    EXPECT_GT(snrFiltered, snrNoisy)
        << "SNR did not improve: noisy = " << snrNoisy << " dB, "
        << "filtered = " << snrFiltered << " dB";
}

TEST_F(FilterApplicationFixture, NoiseSmoothing_HighFreqAttenuation) {
    // High frequency noise should be attenuated more than low frequency
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 200);
    
    // Low frequency signal + high frequency noise
    FillSine(input, 5.0f, 0.02f, 0.0f);
    
    // Add high frequency component
    for (size_t i = 0; i < input.size(); i++) {
        input[i].phaseAngle += 2.0f * std::sin(2 * M_PI * 0.3f * i);
    }
    
    int result = ApplyFilter(input, output, 5, 2, 0, 0);
    ASSERT_EQ(result, 0);
    
    // High frequency should be attenuated
    // Check by looking at variation - filtered should be smoother
    float inputVar = 0.0f, outputVar = 0.0f;
    for (size_t i = 11; i < 189; i++) {
        float inputDiff = input[i + 1].phaseAngle - input[i].phaseAngle;
        float outputDiff = output[i + 1].phaseAngle - output[i].phaseAngle;
        inputVar += inputDiff * inputDiff;
        outputVar += outputDiff * outputDiff;
    }
    
    EXPECT_LT(outputVar, inputVar)
        << "High frequency not attenuated";
}

//=============================================================================
// Edge Handling Tests
//=============================================================================

TEST_F(FilterApplicationFixture, EdgeHandling_Leading_NoDiscontinuity) {
    // No discontinuity at transition from leading edge to central
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    FillSine(input, 5.0f, 0.03f, 0.0f);
    
    int result = ApplyFilter(input, output, 5, 2, 0, 0);
    ASSERT_EQ(result, 0);
    
    // Check smoothness at transition (index 5)
    float diff1 = std::abs(output[4].phaseAngle - output[5].phaseAngle);
    float diff2 = std::abs(output[5].phaseAngle - output[6].phaseAngle);
    
    EXPECT_LT(diff1, 1.0f) << "Discontinuity at leading edge transition";
    EXPECT_TRUE(FloatsEqual(diff1, diff2, 0.5f))
        << "Abrupt change at transition";
}

TEST_F(FilterApplicationFixture, EdgeHandling_Trailing_NoDiscontinuity) {
    // No discontinuity at transition from central to trailing edge
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    FillSine(input, 5.0f, 0.03f, 0.0f);
    
    int result = ApplyFilter(input, output, 5, 2, 0, 0);
    ASSERT_EQ(result, 0);
    
    // Check smoothness at transition (index 94)
    float diff1 = std::abs(output[93].phaseAngle - output[94].phaseAngle);
    float diff2 = std::abs(output[94].phaseAngle - output[95].phaseAngle);
    
    EXPECT_LT(diff1, 1.0f) << "Discontinuity at trailing edge transition";
    EXPECT_TRUE(FloatsEqual(diff1, diff2, 0.5f))
        << "Abrupt change at transition";
}

TEST_F(FilterApplicationFixture, EdgeHandling_LeadingEdge_AllValid) {
    // All leading edge points should be valid
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    FillLinear(input, 1.0f, 0.0f);
    
    int result = ApplyFilter(input, output, 10, 2, 0, 0);
    ASSERT_EQ(result, 0);
    
    // First n points should be valid
    for (size_t i = 0; i < 10; i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle))
            << "Leading edge NaN at " << i;
        EXPECT_FALSE(std::isinf(output[i].phaseAngle))
            << "Leading edge Inf at " << i;
    }
}

TEST_F(FilterApplicationFixture, EdgeHandling_TrailingEdge_AllValid) {
    // All trailing edge points should be valid
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    FillLinear(input, 1.0f, 0.0f);
    
    int result = ApplyFilter(input, output, 10, 2, 0, 0);
    ASSERT_EQ(result, 0);
    
    // Last n points should be valid
    for (size_t i = 90; i < 100; i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle))
            << "Trailing edge NaN at " << i;
        EXPECT_FALSE(std::isinf(output[i].phaseAngle))
            << "Trailing edge Inf at " << i;
    }
}

TEST_F(FilterApplicationFixture, EdgeHandling_ConstantSignal_UniformOutput) {
    // Constant signal should produce constant output even at edges
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 50);
    FillConstant(input, 7.5f);
    
    int result = ApplyFilter(input, output, 5, 2, 0, 0);
    ASSERT_EQ(result, 0);
    
    // All points including edges should equal constant
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, 7.5f, 1e-4f))
            << "Constant not preserved at edge index " << i;
    }
}

//=============================================================================
// Impulse Response Tests
//=============================================================================

TEST_F(FilterApplicationFixture, ImpulseResponse_Smoothing_RecoverWeights) {
    // Impulse response should recover filter weights
    std::vector<float> response;
    ComputeImpulseResponse(response, 5, 2, 0);
    
    // Response should sum to 1 (weights normalize)
    float sum = std::accumulate(response.begin(), response.end(), 0.0f);
    EXPECT_TRUE(FloatsEqual(sum, 1.0f, 1e-4f))
        << "Impulse response sum = " << sum;
    
    // Should be symmetric for centered filter
    for (size_t i = 0; i < response.size() / 2; i++) {
        EXPECT_TRUE(FloatsEqual(response[i], response[response.size() - 1 - i], 1e-5f))
            << "Impulse response not symmetric at " << i;
    }
}

TEST_F(FilterApplicationFixture, ImpulseResponse_Derivative_SumZero) {
    // Derivative impulse response should sum to zero
    std::vector<float> response;
    ComputeImpulseResponse(response, 5, 2, 1);
    
    float sum = std::accumulate(response.begin(), response.end(), 0.0f);
    EXPECT_TRUE(FloatsEqual(sum, 0.0f, 1e-4f))
        << "Derivative impulse response sum = " << sum;
    
    // Should be antisymmetric
    for (size_t i = 0; i < response.size() / 2; i++) {
        EXPECT_TRUE(FloatsEqual(response[i], -response[response.size() - 1 - i], 1e-5f))
            << "Derivative impulse not antisymmetric at " << i;
    }
}

//=============================================================================
// Configuration Tests
//=============================================================================

TEST_F(FilterApplicationFixture, Configuration_YourMainCase_25Point) {
    // Your documented use case: halfWindow=12, order=4, 25 points
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 200);
    FillSine(input, 10.0f, 0.02f, 0.0f);
    AddNoise(input, 1.0f);
    
    int result = ApplyFilter(input, output, 12, 4, 0, 0);
    ASSERT_EQ(result, 0);
    
    // Should smooth effectively
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle));
    }
}

TEST_F(FilterApplicationFixture, Configuration_Causal_PastOnly) {
    // Causal filter: targetPoint = halfWindow (uses only past data)
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    FillLinear(input, 1.0f, 0.0f);
    
    uint8_t n = 5;
    int result = ApplyFilter(input, output, n, 2, n, 0);  // targetPoint = n
    ASSERT_EQ(result, 0);
    
    // Should produce valid output
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle));
    }
}

TEST_F(FilterApplicationFixture, Configuration_SmallWindow_5Point) {
    // Small window for embedded applications
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    FillQuadratic(input, 0.1f, 1.0f, 0.0f);
    
    int result = ApplyFilter(input, output, 2, 2, 0, 0);
    ASSERT_EQ(result, 0);
    
    // Quadratic should be preserved
    for (size_t i = 5; i < 95; i++) {
        EXPECT_TRUE(FloatsEqual(output[i].phaseAngle, input[i].phaseAngle, 1e-3f));
    }
}

TEST_F(FilterApplicationFixture, Configuration_LargeWindow_HighSmoothing) {
    // Large window for aggressive smoothing
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 200);
    FillSine(input, 5.0f, 0.05f, 0.0f);
    AddNoise(input, 2.0f);
    
    int result = ApplyFilter(input, output, 20, 4, 0, 0);
    ASSERT_EQ(result, 0);
    
    // Should heavily smooth
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle));
    }
}

//=============================================================================
// Parameter Validation Tests
//=============================================================================

TEST_F(FilterApplicationFixture, Validation_NullInput_ReturnsError) {
    std::vector<MqsRawDataPoint_t> output(100);
    
    int result = mes_savgolFilter(nullptr, 100, 5, output.data(), 2, 0, 0);
    EXPECT_EQ(result, -1) << "Should reject NULL input";
}

TEST_F(FilterApplicationFixture, Validation_NullOutput_ReturnsError) {
    std::vector<MqsRawDataPoint_t> input(100);
    
    int result = mes_savgolFilter(input.data(), 100, 5, nullptr, 2, 0, 0);
    EXPECT_EQ(result, -1) << "Should reject NULL output";
}

TEST_F(FilterApplicationFixture, Validation_WindowTooLarge_ReturnsError) {
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 20);
    output.resize(20);
    
    // Window size 21 but only 20 data points
    int result = mes_savgolFilter(input.data(), 20, 10, output.data(), 2, 0, 0);
    EXPECT_EQ(result, -2) << "Should reject window larger than data";
}

TEST_F(FilterApplicationFixture, Validation_PolyOrderTooHigh_ReturnsError) {
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    output.resize(100);
    
    // Polynomial order must be < window size
    int result = mes_savgolFilter(input.data(), 100, 5, output.data(), 11, 0, 0);
    EXPECT_EQ(result, -2) << "Should reject poly order >= window size";
}

TEST_F(FilterApplicationFixture, Validation_TargetOutOfBounds_ReturnsError) {
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 100);
    output.resize(100);
    
    // Target point must be <= 2*halfWindow
    int result = mes_savgolFilter(input.data(), 100, 5, output.data(), 2, 11, 0);
    EXPECT_EQ(result, -2) << "Should reject target point out of bounds";
}

//=============================================================================
// Stress Tests
//=============================================================================

TEST_F(FilterApplicationFixture, Stress_LargeDataset_1Million) {
    // 1 million points
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 1000000);
    FillSine(input, 1.0f, 0.00001f, 0.0f);
    
    int result = ApplyFilter(input, output, 5, 2, 0, 0);
    EXPECT_EQ(result, 0);
    
    // Spot check
    for (size_t i = 0; i < output.size(); i += 100000) {
        EXPECT_FALSE(std::isnan(output[i].phaseAngle));
    }
}

TEST_F(FilterApplicationFixture, Stress_MinimumDataset) {
    // Minimum size = window size
    uint8_t n = 5;
    std::vector<MqsRawDataPoint_t> input, output;
    CreateData(input, 2 * n + 1);
    FillConstant(input, 1.0f);
    
    int result = ApplyFilter(input, output, n, 2, 0, 0);
    EXPECT_EQ(result, 0);
    
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_FLOAT_EQ(output[i].phaseAngle, 1.0f);
    }
}

//=============================================================================
// Main function
//=============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}