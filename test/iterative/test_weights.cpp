/**
 * @file test_weights.cpp
 * @brief GoogleTest suite for Savitzky-Golay Filter Weight Calculation
 * 
 * Tests the computation of filter weights which are the convolution coefficients
 * applied to the data. Weights are computed using Gram polynomials and GenFact
 * normalization factors.
 * 
 * Mathematical properties tested:
 * - Weight normalization: sum(weights) = 1 for smoothing (d=0)
 * - Weight sum = 0 for derivatives (d>0)
 * - Symmetry for centered filters (targetPoint=0, d=0)
 * - Antisymmetry for centered derivatives (targetPoint=0, d=1)
 * - Known published weight values
 * - Edge case weight calculations
 */

#include <gtest/gtest.h>
#include "savgolFilter.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

//=============================================================================
// Test Fixture for Weight Calculation Tests
//=============================================================================

class WeightFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize GenFact table (required dependency)
        InitGenFactTable();
    }

    void TearDown() override {
        // Nothing to clean up
    }

    /**
     * @brief Compute weights using the implementation under test
     */
    void ComputeWeights(uint8_t halfWindowSize, uint16_t targetPoint, 
                       uint8_t polynomialOrder, uint8_t derivativeOrder,
                       std::vector<float>& weights) {
        uint16_t windowSize = 2 * halfWindowSize + 1;
        weights.resize(windowSize);
        
        ::ComputeWeights(halfWindowSize, targetPoint, polynomialOrder, 
                        derivativeOrder, weights.data());
    }

    /**
     * @brief Sum all weights in a vector
     */
    float SumWeights(const std::vector<float>& weights) {
        return std::accumulate(weights.begin(), weights.end(), 0.0f);
    }

    /**
     * @brief Check if weights are symmetric about center
     */
    bool AreSymmetric(const std::vector<float>& weights, float tolerance = 1e-6f) {
        size_t n = weights.size();
        for (size_t i = 0; i < n / 2; i++) {
            if (std::abs(weights[i] - weights[n - 1 - i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Check if weights are antisymmetric about center
     */
    bool AreAntisymmetric(const std::vector<float>& weights, float tolerance = 1e-6f) {
        size_t n = weights.size();
        for (size_t i = 0; i < n / 2; i++) {
            if (std::abs(weights[i] + weights[n - 1 - i]) > tolerance) {
                return false;
            }
        }
        // Center should be zero for odd-sized antisymmetric
        if (n % 2 == 1) {
            if (std::abs(weights[n / 2]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Find the index of maximum absolute value
     */
    size_t FindMaxAbsIndex(const std::vector<float>& weights) {
        return std::distance(weights.begin(),
                           std::max_element(weights.begin(), weights.end(),
                                          [](float a, float b) {
                                              return std::abs(a) < std::abs(b);
                                          }));
    }

    /**
     * @brief Compare two weight vectors element-wise
     */
    bool WeightsEqual(const std::vector<float>& a, const std::vector<float>& b, 
                     float tolerance = 1e-5f) {
        if (a.size() != b.size()) return false;
        
        for (size_t i = 0; i < a.size(); i++) {
            if (std::abs(a[i] - b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Compare floats with relative tolerance
     */
    bool FloatsEqual(float a, float b, float relTol = 1e-5f) {
        if (a == b) return true;
        if (std::abs(a) < 1e-8f && std::abs(b) < 1e-8f) return true;
        float maxAbs = std::max(std::abs(a), std::abs(b));
        return std::abs(a - b) <= relTol * maxAbs;
    }
};

//=============================================================================
// Normalization Tests - Fundamental Property
//=============================================================================

TEST_F(WeightFixture, Normalization_Smoothing_SumToOne) {
    // For smoothing (d=0), weights should sum to 1.0
    std::vector<uint8_t> halfWindows = {2, 5, 10, 15};
    std::vector<uint8_t> polyOrders = {2, 3, 4};
    
    for (uint8_t n : halfWindows) {
        for (uint8_t m : polyOrders) {
            if (m >= 2 * n + 1) continue;  // Skip invalid combinations
            
            std::vector<float> weights;
            ComputeWeights(n, 0, m, 0, weights);
            
            float sum = SumWeights(weights);
            EXPECT_TRUE(FloatsEqual(sum, 1.0f))
                << "Weights sum to " << sum << " for n=" << (int)n 
                << ", m=" << (int)m << " (should be 1.0)";
        }
    }
}

TEST_F(WeightFixture, Normalization_FirstDerivative_SumToZero) {
    // For first derivative (d=1), weights should sum to 0.0
    std::vector<uint8_t> halfWindows = {2, 5, 10};
    std::vector<uint8_t> polyOrders = {2, 3, 4};
    
    for (uint8_t n : halfWindows) {
        for (uint8_t m : polyOrders) {
            if (m >= 2 * n + 1) continue;
            
            std::vector<float> weights;
            ComputeWeights(n, 0, m, 1, weights);
            
            float sum = SumWeights(weights);
            EXPECT_TRUE(FloatsEqual(sum, 0.0f, 1e-4f))
                << "Derivative weights sum to " << sum << " for n=" << (int)n 
                << ", m=" << (int)m << " (should be 0.0)";
        }
    }
}

TEST_F(WeightFixture, Normalization_SecondDerivative_SumToZero) {
    // For second derivative (d=2), weights should sum to 0.0
    std::vector<uint8_t> halfWindows = {3, 5, 8};
    std::vector<uint8_t> polyOrders = {2, 3, 4};
    
    for (uint8_t n : halfWindows) {
        for (uint8_t m : polyOrders) {
            if (m >= 2 * n + 1) continue;
            
            std::vector<float> weights;
            ComputeWeights(n, 0, m, 2, weights);
            
            float sum = SumWeights(weights);
            EXPECT_TRUE(FloatsEqual(sum, 0.0f, 1e-4f))
                << "2nd derivative weights sum to " << sum << " for n=" << (int)n 
                << ", m=" << (int)m << " (should be 0.0)";
        }
    }
}

//=============================================================================
// Symmetry Tests - Centered Filters
//=============================================================================

TEST_F(WeightFixture, Symmetry_Centered_Smoothing) {
    // Centered smoothing (targetPoint=0, d=0) should have symmetric weights
    std::vector<uint8_t> halfWindows = {2, 5, 10};
    std::vector<uint8_t> polyOrders = {2, 3, 4};
    
    for (uint8_t n : halfWindows) {
        for (uint8_t m : polyOrders) {
            if (m >= 2 * n + 1) continue;
            
            std::vector<float> weights;
            ComputeWeights(n, 0, m, 0, weights);
            
            EXPECT_TRUE(AreSymmetric(weights))
                << "Centered smoothing weights should be symmetric for n=" 
                << (int)n << ", m=" << (int)m;
        }
    }
}

TEST_F(WeightFixture, Symmetry_Centered_FirstDerivative) {
    // Centered first derivative (targetPoint=0, d=1) should be antisymmetric
    std::vector<uint8_t> halfWindows = {2, 5, 10};
    std::vector<uint8_t> polyOrders = {2, 3, 4};
    
    for (uint8_t n : halfWindows) {
        for (uint8_t m : polyOrders) {
            if (m >= 2 * n + 1) continue;
            
            std::vector<float> weights;
            ComputeWeights(n, 0, m, 1, weights);
            
            EXPECT_TRUE(AreAntisymmetric(weights))
                << "Centered derivative weights should be antisymmetric for n=" 
                << (int)n << ", m=" << (int)m;
        }
    }
}

TEST_F(WeightFixture, Symmetry_Centered_MaxAtCenter) {
    // For centered smoothing, max weight should be at center
    std::vector<uint8_t> halfWindows = {2, 5, 10};
    
    for (uint8_t n : halfWindows) {
        std::vector<float> weights;
        ComputeWeights(n, 0, 2, 0, weights);
        
        size_t maxIdx = FindMaxAbsIndex(weights);
        size_t centerIdx = n;  // Center is at index n
        
        EXPECT_EQ(maxIdx, centerIdx)
            << "Max weight should be at center for n=" << (int)n;
    }
}

TEST_F(WeightFixture, Symmetry_Asymmetric_EdgeTarget) {
    // For edge-targeted filters (targetPoint != 0), weights should be asymmetric
    uint8_t n = 5;
    uint8_t targetPoint = n;  // Target at right edge
    
    std::vector<float> weights;
    ComputeWeights(n, targetPoint, 2, 0, weights);
    
    EXPECT_FALSE(AreSymmetric(weights))
        << "Edge-targeted weights should be asymmetric";
}

//=============================================================================
// Known Values Tests - Published Filter Coefficients
//=============================================================================

TEST_F(WeightFixture, KnownValues_5Point_Quadratic_Smoothing) {
    // 5-point quadratic smoothing (n=2, m=2, t=0, d=0)
    // Published values: [-3, 12, 17, 12, -3] / 35
    
    std::vector<float> weights;
    ComputeWeights(2, 0, 2, 0, weights);
    
    std::vector<float> expected = {
        -3.0f / 35.0f,
        12.0f / 35.0f,
        17.0f / 35.0f,
        12.0f / 35.0f,
        -3.0f / 35.0f
    };
    
    ASSERT_EQ(weights.size(), expected.size());
    
    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_TRUE(FloatsEqual(weights[i], expected[i]))
            << "Weight[" << i << "] = " << weights[i] 
            << ", expected " << expected[i];
    }
}

TEST_F(WeightFixture, KnownValues_5Point_Quadratic_FirstDerivative) {
    // 5-point quadratic first derivative (n=2, m=2, t=0, d=1)
    // Published values: [-2, -1, 0, 1, 2] / 10
    
    std::vector<float> weights;
    ComputeWeights(2, 0, 2, 1, weights);
    
    std::vector<float> expected = {
        -2.0f / 10.0f,
        -1.0f / 10.0f,
         0.0f / 10.0f,
         1.0f / 10.0f,
         2.0f / 10.0f
    };
    
    ASSERT_EQ(weights.size(), expected.size());
    
    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_TRUE(FloatsEqual(weights[i], expected[i]))
            << "Weight[" << i << "] = " << weights[i] 
            << ", expected " << expected[i];
    }
}

TEST_F(WeightFixture, KnownValues_7Point_Cubic_Smoothing) {
    // 7-point cubic smoothing (n=3, m=3, t=0, d=0)
    // Published values: [-2, 3, 6, 7, 6, 3, -2] / 21
    
    std::vector<float> weights;
    ComputeWeights(3, 0, 3, 0, weights);
    
    std::vector<float> expected = {
        -2.0f / 21.0f,
         3.0f / 21.0f,
         6.0f / 21.0f,
         7.0f / 21.0f,
         6.0f / 21.0f,
         3.0f / 21.0f,
        -2.0f / 21.0f
    };
    
    ASSERT_EQ(weights.size(), expected.size());
    
    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_TRUE(FloatsEqual(weights[i], expected[i], 1e-4f))
            << "Weight[" << i << "] = " << weights[i] 
            << ", expected " << expected[i];
    }
}

TEST_F(WeightFixture, KnownValues_5Point_Quartic_Smoothing) {
    // 5-point quartic/quadratic smoothing should match quadratic
    // (quartic and quadratic give same results for 5 points)
    
    std::vector<float> quadratic, quartic;
    ComputeWeights(2, 0, 2, 0, quadratic);
    ComputeWeights(2, 0, 4, 0, quartic);
    
    EXPECT_TRUE(WeightsEqual(quadratic, quartic, 1e-5f))
        << "5-point quartic should match quadratic";
}

//=============================================================================
// Edge Weight Tests
//=============================================================================

TEST_F(WeightFixture, EdgeWeights_LeadingEdge_Position0) {
    // At position 0, target should be at right edge (targetPoint = n)
    uint8_t n = 5;
    
    std::vector<float> weights;
    ComputeWeights(n, n, 2, 0, weights);
    
    // Weights should be valid (sum to 1)
    float sum = SumWeights(weights);
    EXPECT_TRUE(FloatsEqual(sum, 1.0f))
        << "Leading edge weights should sum to 1.0";
    
    // Should be asymmetric (favoring right side)
    EXPECT_FALSE(AreSymmetric(weights));
}

TEST_F(WeightFixture, EdgeWeights_TrailingEdge_LastPosition) {
    // At last position, same weights as leading edge (but reversed data access)
    uint8_t n = 5;
    
    std::vector<float> weights;
    ComputeWeights(n, n, 2, 0, weights);
    
    // Weights should be valid
    float sum = SumWeights(weights);
    EXPECT_TRUE(FloatsEqual(sum, 1.0f))
        << "Trailing edge weights should sum to 1.0";
}

TEST_F(WeightFixture, EdgeWeights_TransitionToCentral) {
    // Weights should transition smoothly from edge to central
    uint8_t n = 5;
    
    // Edge weight (targetPoint = n)
    std::vector<float> edgeWeights;
    ComputeWeights(n, n, 2, 0, edgeWeights);
    
    // Near-edge weight (targetPoint = n-1)
    std::vector<float> nearEdgeWeights;
    ComputeWeights(n, n - 1, 2, 0, nearEdgeWeights);
    
    // Central weight (targetPoint = 0)
    std::vector<float> centralWeights;
    ComputeWeights(n, 0, 2, 0, centralWeights);
    
    // All should be valid
    EXPECT_TRUE(FloatsEqual(SumWeights(edgeWeights), 1.0f));
    EXPECT_TRUE(FloatsEqual(SumWeights(nearEdgeWeights), 1.0f));
    EXPECT_TRUE(FloatsEqual(SumWeights(centralWeights), 1.0f));
    
    // Should be progressively different
    EXPECT_FALSE(WeightsEqual(edgeWeights, centralWeights));
}

TEST_F(WeightFixture, EdgeWeights_AllPositions_SumToOne) {
    // For all target positions, smoothing weights should sum to 1
    uint8_t n = 5;
    uint16_t windowSize = 2 * n + 1;
    
    for (uint16_t t = 0; t <= windowSize - 1; t++) {
        std::vector<float> weights;
        ComputeWeights(n, t, 2, 0, weights);
        
        float sum = SumWeights(weights);
        EXPECT_TRUE(FloatsEqual(sum, 1.0f))
            << "Weights for targetPoint=" << t << " sum to " << sum;
    }
}

//=============================================================================
// Monotonicity and Smoothness Tests
//=============================================================================

TEST_F(WeightFixture, Monotonicity_CenteredSmoothing_DecreaseFromCenter) {
    // For centered smoothing, weights should decrease moving away from center
    uint8_t n = 10;
    
    std::vector<float> weights;
    ComputeWeights(n, 0, 2, 0, weights);
    
    size_t center = n;
    
    // Check left side
    for (size_t i = 0; i < center - 1; i++) {
        EXPECT_LE(weights[i], weights[i + 1])
            << "Weights should increase toward center on left side";
    }
    
    // Check right side
    for (size_t i = center + 1; i < weights.size() - 1; i++) {
        EXPECT_GE(weights[i], weights[i + 1])
            << "Weights should decrease from center on right side";
    }
}

TEST_F(WeightFixture, Smoothness_NoWildOscillations) {
    // Weights should vary smoothly, no wild oscillations
    uint8_t n = 10;
    
    std::vector<float> weights;
    ComputeWeights(n, 0, 4, 0, weights);
    
    // Check that adjacent weight differences are reasonable
    for (size_t i = 0; i < weights.size() - 1; i++) {
        float diff = std::abs(weights[i + 1] - weights[i]);
        EXPECT_LT(diff, 0.5f)
            << "Weights oscillate wildly between indices " << i << " and " << i + 1;
    }
}

//=============================================================================
// Parameter Variation Tests
//=============================================================================

TEST_F(WeightFixture, ParameterVariation_IncreasingPolyOrder) {
    // Increasing polynomial order should change weights
    uint8_t n = 5;
    
    std::vector<float> order2, order3, order4;
    ComputeWeights(n, 0, 2, 0, order2);
    ComputeWeights(n, 0, 3, 0, order3);
    ComputeWeights(n, 0, 4, 0, order4);
    
    // All valid
    EXPECT_TRUE(FloatsEqual(SumWeights(order2), 1.0f));
    EXPECT_TRUE(FloatsEqual(SumWeights(order3), 1.0f));
    EXPECT_TRUE(FloatsEqual(SumWeights(order4), 1.0f));
    
    // Should be different
    EXPECT_FALSE(WeightsEqual(order2, order3));
    EXPECT_FALSE(WeightsEqual(order3, order4));
}

TEST_F(WeightFixture, ParameterVariation_IncreasingWindowSize) {
    // Increasing window size should change weights
    std::vector<float> window5, window11, window21;
    ComputeWeights(2, 0, 2, 0, window5);   // 5 points
    ComputeWeights(5, 0, 2, 0, window11);  // 11 points
    ComputeWeights(10, 0, 2, 0, window21); // 21 points
    
    // All valid
    EXPECT_TRUE(FloatsEqual(SumWeights(window5), 1.0f));
    EXPECT_TRUE(FloatsEqual(SumWeights(window11), 1.0f));
    EXPECT_TRUE(FloatsEqual(SumWeights(window21), 1.0f));
    
    // Sizes should be different
    EXPECT_EQ(window5.size(), 5);
    EXPECT_EQ(window11.size(), 11);
    EXPECT_EQ(window21.size(), 21);
}

TEST_F(WeightFixture, ParameterVariation_DerivativeOrder) {
    // Different derivative orders produce different weights
    uint8_t n = 5;
    
    std::vector<float> smooth, deriv1, deriv2;
    ComputeWeights(n, 0, 3, 0, smooth);
    ComputeWeights(n, 0, 3, 1, deriv1);
    ComputeWeights(n, 0, 3, 2, deriv2);
    
    // Different sums
    EXPECT_TRUE(FloatsEqual(SumWeights(smooth), 1.0f));
    EXPECT_TRUE(FloatsEqual(SumWeights(deriv1), 0.0f, 1e-4f));
    EXPECT_TRUE(FloatsEqual(SumWeights(deriv2), 0.0f, 1e-4f));
    
    // Different weights
    EXPECT_FALSE(WeightsEqual(smooth, deriv1));
    EXPECT_FALSE(WeightsEqual(deriv1, deriv2));
}

//=============================================================================
// Numerical Stability Tests
//=============================================================================

TEST_F(WeightFixture, Stability_NoNaNorInf) {
    // No NaN or Inf values in any reasonable configuration
    std::vector<uint8_t> halfWindows = {2, 5, 10, 15, 20};
    std::vector<uint8_t> polyOrders = {2, 3, 4};
    std::vector<uint8_t> derivOrders = {0, 1, 2};
    
    for (uint8_t n : halfWindows) {
        for (uint8_t m : polyOrders) {
            if (m >= 2 * n + 1) continue;
            
            for (uint8_t d : derivOrders) {
                std::vector<float> weights;
                ComputeWeights(n, 0, m, d, weights);
                
                for (size_t i = 0; i < weights.size(); i++) {
                    EXPECT_FALSE(std::isnan(weights[i]))
                        << "NaN at index " << i << " for n=" << (int)n 
                        << ", m=" << (int)m << ", d=" << (int)d;
                    
                    EXPECT_FALSE(std::isinf(weights[i]))
                        << "Inf at index " << i << " for n=" << (int)n 
                        << ", m=" << (int)m << ", d=" << (int)d;
                }
            }
        }
    }
}

TEST_F(WeightFixture, Stability_ReasonableMagnitude) {
    // Weights should be reasonable magnitude (not too large or small)
    std::vector<uint8_t> halfWindows = {2, 5, 10, 20};
    
    for (uint8_t n : halfWindows) {
        std::vector<float> weights;
        ComputeWeights(n, 0, 2, 0, weights);
        
        for (float w : weights) {
            EXPECT_LT(std::abs(w), 10.0f)
                << "Weight unexpectedly large: " << w << " for n=" << (int)n;
        }
    }
}

TEST_F(WeightFixture, Stability_RepeatedComputation) {
    // Repeated computation should give identical results
    uint8_t n = 5;
    
    std::vector<float> first;
    ComputeWeights(n, 0, 2, 0, first);
    
    for (int rep = 0; rep < 10; rep++) {
        std::vector<float> current;
        ComputeWeights(n, 0, 2, 0, current);
        
        EXPECT_TRUE(WeightsEqual(first, current, 1e-8f))
            << "Weights changed on repetition " << rep;
    }
}

//=============================================================================
// Typical Usage Pattern Tests
//=============================================================================

TEST_F(WeightFixture, TypicalUsage_YourMainCase_25Point_Order4) {
    // Your documented main use case: halfWindow=12, order=4
    uint8_t n = 12;
    uint8_t m = 4;
    
    // Smoothing
    std::vector<float> smooth;
    ComputeWeights(n, 0, m, 0, smooth);
    
    EXPECT_EQ(smooth.size(), 25);
    EXPECT_TRUE(FloatsEqual(SumWeights(smooth), 1.0f));
    EXPECT_TRUE(AreSymmetric(smooth));
    
    // First derivative
    std::vector<float> deriv1;
    ComputeWeights(n, 0, m, 1, deriv1);
    
    EXPECT_EQ(deriv1.size(), 25);
    EXPECT_TRUE(FloatsEqual(SumWeights(deriv1), 0.0f, 1e-4f));
    EXPECT_TRUE(AreAntisymmetric(deriv1));
}

TEST_F(WeightFixture, TypicalUsage_EmbeddedCommon_5Point) {
    // Common embedded use case: 5-point quadratic
    std::vector<float> weights;
    ComputeWeights(2, 0, 2, 0, weights);
    
    EXPECT_EQ(weights.size(), 5);
    EXPECT_TRUE(FloatsEqual(SumWeights(weights), 1.0f));
}

TEST_F(WeightFixture, TypicalUsage_RealTimeCausal) {
    // Real-time causal filter: targetPoint = n
    uint8_t n = 5;
    
    std::vector<float> weights;
    ComputeWeights(n, n, 2, 0, weights);
    
    EXPECT_TRUE(FloatsEqual(SumWeights(weights), 1.0f));
    EXPECT_FALSE(AreSymmetric(weights)); // Should be asymmetric
}

//=============================================================================
// Comprehensive Parameterized Tests
//=============================================================================

class WeightParameterizedTest : public WeightFixture,
                                 public ::testing::WithParamInterface<
                                     std::tuple<uint8_t, uint8_t, uint8_t>> {
};

TEST_P(WeightParameterizedTest, AllConfigurations_ValidWeights) {
    auto [halfWindow, polyOrder, derivOrder] = GetParam();
    
    std::vector<float> weights;
    ComputeWeights(halfWindow, 0, polyOrder, derivOrder, weights);
    
    // Check size
    EXPECT_EQ(weights.size(), 2 * halfWindow + 1);
    
    // Check for NaN/Inf
    for (float w : weights) {
        EXPECT_FALSE(std::isnan(w));
        EXPECT_FALSE(std::isinf(w));
    }
    
    // Check normalization
    float sum = SumWeights(weights);
    if (derivOrder == 0) {
        EXPECT_TRUE(FloatsEqual(sum, 1.0f))
            << "Smoothing sum=" << sum << " for n=" << (int)halfWindow 
            << ", m=" << (int)polyOrder;
    } else {
        EXPECT_TRUE(FloatsEqual(sum, 0.0f, 1e-4f))
            << "Derivative sum=" << sum << " for n=" << (int)halfWindow 
            << ", m=" << (int)polyOrder << ", d=" << (int)derivOrder;
    }
}

// Generate test cases
std::vector<std::tuple<uint8_t, uint8_t, uint8_t>> GenerateWeightTestCases() {
    std::vector<std::tuple<uint8_t, uint8_t, uint8_t>> cases;
    
    std::vector<uint8_t> halfWindows = {2, 3, 5, 10, 15, 20};
    std::vector<uint8_t> polyOrders = {2, 3, 4};
    std::vector<uint8_t> derivOrders = {0, 1, 2};
    
    for (uint8_t n : halfWindows) {
        for (uint8_t m : polyOrders) {
            if (m < 2 * n + 1) {  // Valid combination
                for (uint8_t d : derivOrders) {
                    cases.push_back({n, m, d});
                }
            }
        }
    }
    
    return cases;
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    WeightParameterizedTest,
    ::testing::ValuesIn(GenerateWeightTestCases())
);

//=============================================================================
// Main function
//=============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}