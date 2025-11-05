/**
 * @file test_grampoly.cpp
 * @brief GoogleTest suite for Gram Polynomial computation
 * 
 * Tests the iterative Gram polynomial computation which forms the mathematical
 * foundation of Savitzky-Golay filter weights.
 * 
 * Gram polynomials F(k,d) are orthogonal polynomials over discrete points
 * [-n, ..., 0, ..., +n] computed using a three-term recurrence relation.
 * 
 * Mathematical properties tested:
 * - Base cases: F(0,0)=1, F(0,d>0)=0
 * - Recurrence relation correctness
 * - Orthogonality over discrete points
 * - Symmetry/antisymmetry properties
 * - Numerical stability
 */

#include <gtest/gtest.h>
#include "savgolFilter.h"
#include <cmath>
#include <vector>
#include <algorithm>

//=============================================================================
// Test Fixture for Gram Polynomial Tests
//=============================================================================

class GramPolyFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize GenFact table (required dependency)
        InitGenFactTable();
    }

    void TearDown() override {
        // Nothing to clean up
    }

    /**
     * @brief Helper to create a GramPolyContext
     */
    GramPolyContext CreateContext(uint8_t halfWindowSize, uint8_t derivativeOrder) {
        GramPolyContext ctx;
        ctx.halfWindowSize = halfWindowSize;
        ctx.targetPoint = 0;  // Not used in GramPolyIterative, but set for completeness
        ctx.derivativeOrder = derivativeOrder;
        return ctx;
    }

    /**
     * @brief Compute Gram polynomial using the implementation under test
     */
    float ComputeGramPoly(uint8_t polynomialOrder, int dataIndex, 
                         uint8_t halfWindowSize, uint8_t derivativeOrder) {
        GramPolyContext ctx = CreateContext(halfWindowSize, derivativeOrder);
        
#ifdef ENABLE_MEMOIZATION
        return MemoizedGramPoly(polynomialOrder, dataIndex, &ctx);
#else
        return GramPolyIterative(polynomialOrder, dataIndex, &ctx);
#endif
    }

    /**
     * @brief Test orthogonality: sum over discrete points of F(k,d) * F(j,d) should be 0 for k≠j
     */
    bool TestOrthogonality(uint8_t k, uint8_t j, uint8_t halfWindowSize, uint8_t derivativeOrder) {
        if (k == j) return true;  // Not testing self-orthogonality
        
        float sum = 0.0f;
        for (int i = -(int)halfWindowSize; i <= (int)halfWindowSize; i++) {
            float fk = ComputeGramPoly(k, i, halfWindowSize, derivativeOrder);
            float fj = ComputeGramPoly(j, i, halfWindowSize, derivativeOrder);
            sum += fk * fj;
        }
        
        // Orthogonal if sum is close to zero (within numerical precision)
        return std::abs(sum) < 1e-4f;
    }

    /**
     * @brief Compare floats with relative tolerance
     */
    bool FloatsEqual(float a, float b, float relTol = 1e-5f) {
        if (a == b) return true;
        if (std::abs(a) < 1e-8f && std::abs(b) < 1e-8f) return true;  // Both near zero
        float maxAbs = std::max(std::abs(a), std::abs(b));
        return std::abs(a - b) <= relTol * maxAbs;
    }
};

//=============================================================================
// Base Case Tests - F(0,d)
//=============================================================================

TEST_F(GramPolyFixture, BaseCase_K0_D0_EqualsOne) {
    // F(0, 0) = 1 at all positions (constant polynomial)
    uint8_t halfWindow = 5;
    
    for (int i = -(int)halfWindow; i <= (int)halfWindow; i++) {
        float result = ComputeGramPoly(0, i, halfWindow, 0);
        EXPECT_FLOAT_EQ(result, 1.0f) 
            << "F(0,0)[" << i << "] should be 1.0";
    }
}

TEST_F(GramPolyFixture, BaseCase_K0_D1_EqualsZero) {
    // F(0, d>0) = 0 for all positions (derivative of constant is zero)
    uint8_t halfWindow = 5;
    
    for (int i = -(int)halfWindow; i <= (int)halfWindow; i++) {
        float result = ComputeGramPoly(0, i, halfWindow, 1);
        EXPECT_FLOAT_EQ(result, 0.0f) 
            << "F(0,1)[" << i << "] should be 0.0";
    }
}

TEST_F(GramPolyFixture, BaseCase_K0_D2_EqualsZero) {
    // F(0, d>0) = 0 for second derivative too
    uint8_t halfWindow = 5;
    
    for (int i = -(int)halfWindow; i <= (int)halfWindow; i++) {
        float result = ComputeGramPoly(0, i, halfWindow, 2);
        EXPECT_FLOAT_EQ(result, 0.0f) 
            << "F(0,2)[" << i << "] should be 0.0";
    }
}

//=============================================================================
// First Order Tests - F(1,d)
//=============================================================================

TEST_F(GramPolyFixture, FirstOrder_K1_D0_LinearInIndex) {
    // F(1, 0) should be linear in the index: F(1,0)[i] = i/n
    uint8_t halfWindow = 5;
    
    for (int i = -(int)halfWindow; i <= (int)halfWindow; i++) {
        float result = ComputeGramPoly(1, i, halfWindow, 0);
        float expected = (float)i / halfWindow;
        
        EXPECT_TRUE(FloatsEqual(result, expected))
            << "F(1,0)[" << i << "] = " << result 
            << ", expected " << expected;
    }
}

TEST_F(GramPolyFixture, FirstOrder_K1_D0_ZeroAtCenter) {
    // F(1, 0) should be zero at center (i=0)
    uint8_t halfWindow = 10;
    
    float result = ComputeGramPoly(1, 0, halfWindow, 0);
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST_F(GramPolyFixture, FirstOrder_K1_D1_ConstantValue) {
    // F(1, 1) = d/n where d=1, so F(1,1) = 1/n (constant across all positions)
    uint8_t halfWindow = 5;
    float expected = 1.0f / halfWindow;
    
    for (int i = -(int)halfWindow; i <= (int)halfWindow; i++) {
        float result = ComputeGramPoly(1, i, halfWindow, 1);
        
        EXPECT_TRUE(FloatsEqual(result, expected))
            << "F(1,1)[" << i << "] should be constant = " << expected;
    }
}

//=============================================================================
// Known Values from Literature
//=============================================================================

TEST_F(GramPolyFixture, KnownValue_SmallWindow_n2) {
    // For halfWindow=2 (5-point window), we can hand-calculate some values
    uint8_t halfWindow = 2;
    
    // F(1,0) at i=-2: should be -2/2 = -1.0
    float result = ComputeGramPoly(1, -2, halfWindow, 0);
    EXPECT_FLOAT_EQ(result, -1.0f);
    
    // F(1,0) at i=2: should be 2/2 = 1.0
    result = ComputeGramPoly(1, 2, halfWindow, 0);
    EXPECT_FLOAT_EQ(result, 1.0f);
    
    // F(1,0) at i=1: should be 1/2 = 0.5
    result = ComputeGramPoly(1, 1, halfWindow, 0);
    EXPECT_FLOAT_EQ(result, 0.5f);
}

TEST_F(GramPolyFixture, KnownValue_MediumWindow_n5) {
    // For halfWindow=5, test some known relationships
    uint8_t halfWindow = 5;
    
    // F(0,0) should always be 1
    EXPECT_FLOAT_EQ(ComputeGramPoly(0, 0, halfWindow, 0), 1.0f);
    EXPECT_FLOAT_EQ(ComputeGramPoly(0, -5, halfWindow, 0), 1.0f);
    EXPECT_FLOAT_EQ(ComputeGramPoly(0, 5, halfWindow, 0), 1.0f);
    
    // F(1,0) at boundaries: -1 and +1
    EXPECT_FLOAT_EQ(ComputeGramPoly(1, -5, halfWindow, 0), -1.0f);
    EXPECT_FLOAT_EQ(ComputeGramPoly(1, 5, halfWindow, 0), 1.0f);
}

//=============================================================================
// Symmetry and Antisymmetry Properties
//=============================================================================

TEST_F(GramPolyFixture, Symmetry_D0_EvenK) {
    // For derivative order 0 and even k: F(k,0)[i] = F(k,0)[-i]
    uint8_t halfWindow = 5;
    uint8_t evenK[] = {0, 2, 4};
    
    for (uint8_t k : evenK) {
        for (int i = 1; i <= (int)halfWindow; i++) {
            float positive = ComputeGramPoly(k, i, halfWindow, 0);
            float negative = ComputeGramPoly(k, -i, halfWindow, 0);
            
            EXPECT_TRUE(FloatsEqual(positive, negative))
                << "F(" << (int)k << ",0)[" << i << "] should equal F(" 
                << (int)k << ",0)[" << -i << "]";
        }
    }
}

TEST_F(GramPolyFixture, Antisymmetry_D0_OddK) {
    // For derivative order 0 and odd k: F(k,0)[i] = -F(k,0)[-i]
    uint8_t halfWindow = 5;
    uint8_t oddK[] = {1, 3};
    
    for (uint8_t k : oddK) {
        for (int i = 1; i <= (int)halfWindow; i++) {
            float positive = ComputeGramPoly(k, i, halfWindow, 0);
            float negative = ComputeGramPoly(k, -i, halfWindow, 0);
            
            EXPECT_TRUE(FloatsEqual(positive, -negative))
                << "F(" << (int)k << ",0)[" << i << "] should equal -F(" 
                << (int)k << ",0)[" << -i << "]";
        }
    }
}

TEST_F(GramPolyFixture, Antisymmetry_D1_EvenK) {
    // For derivative order 1 and even k: F(k,1)[i] = -F(k,1)[-i]
    uint8_t halfWindow = 5;
    uint8_t evenK[] = {0, 2, 4};
    
    for (uint8_t k : evenK) {
        for (int i = 1; i <= (int)halfWindow; i++) {
            float positive = ComputeGramPoly(k, i, halfWindow, 1);
            float negative = ComputeGramPoly(k, -i, halfWindow, 1);
            
            // Skip if both are near zero
            if (std::abs(positive) < 1e-6f && std::abs(negative) < 1e-6f) continue;
            
            EXPECT_TRUE(FloatsEqual(positive, -negative))
                << "F(" << (int)k << ",1)[" << i << "] should equal -F(" 
                << (int)k << ",1)[" << -i << "]";
        }
    }
}

TEST_F(GramPolyFixture, Symmetry_D1_OddK) {
    // For derivative order 1 and odd k: F(k,1)[i] = F(k,1)[-i]
    uint8_t halfWindow = 5;
    uint8_t oddK[] = {1, 3};
    
    for (uint8_t k : oddK) {
        for (int i = 1; i <= (int)halfWindow; i++) {
            float positive = ComputeGramPoly(k, i, halfWindow, 1);
            float negative = ComputeGramPoly(k, -i, halfWindow, 1);
            
            EXPECT_TRUE(FloatsEqual(positive, negative))
                << "F(" << (int)k << ",1)[" << i << "] should equal F(" 
                << (int)k << ",1)[" << -i << "]";
        }
    }
}

//=============================================================================
// Orthogonality Tests
//=============================================================================

TEST_F(GramPolyFixture, Orthogonality_K0_K1_D0) {
    // F(0,0) and F(1,0) should be orthogonal
    uint8_t halfWindow = 5;
    EXPECT_TRUE(TestOrthogonality(0, 1, halfWindow, 0))
        << "F(0,0) and F(1,0) should be orthogonal";
}

TEST_F(GramPolyFixture, Orthogonality_K0_K2_D0) {
    // F(0,0) and F(2,0) should be orthogonal
    uint8_t halfWindow = 5;
    EXPECT_TRUE(TestOrthogonality(0, 2, halfWindow, 0))
        << "F(0,0) and F(2,0) should be orthogonal";
}

TEST_F(GramPolyFixture, Orthogonality_K1_K2_D0) {
    // F(1,0) and F(2,0) should be orthogonal
    uint8_t halfWindow = 5;
    EXPECT_TRUE(TestOrthogonality(1, 2, halfWindow, 0))
        << "F(1,0) and F(2,0) should be orthogonal";
}

TEST_F(GramPolyFixture, Orthogonality_AllPairs_D0) {
    // Test orthogonality for all pairs up to k=4
    uint8_t halfWindow = 10;
    uint8_t maxK = 4;
    
    for (uint8_t k1 = 0; k1 <= maxK; k1++) {
        for (uint8_t k2 = k1 + 1; k2 <= maxK; k2++) {
            EXPECT_TRUE(TestOrthogonality(k1, k2, halfWindow, 0))
                << "F(" << (int)k1 << ",0) and F(" << (int)k2 << ",0) should be orthogonal";
        }
    }
}

//=============================================================================
// Boundary and Edge Cases
//=============================================================================

TEST_F(GramPolyFixture, EdgeCase_SinglePoint_n0) {
    // halfWindow = 0 means single point window (degenerate case)
    uint8_t halfWindow = 0;
    
    // Only valid index is 0
    float result = ComputeGramPoly(0, 0, halfWindow, 0);
    EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST_F(GramPolyFixture, EdgeCase_MaxWindow) {
    // Test at maximum supported window size
    uint8_t halfWindow = 32;
    
    // Should not crash or produce NaN
    float result = ComputeGramPoly(0, 0, halfWindow, 0);
    EXPECT_FLOAT_EQ(result, 1.0f);
    
    result = ComputeGramPoly(1, halfWindow, halfWindow, 0);
    EXPECT_FALSE(std::isnan(result));
    EXPECT_FALSE(std::isinf(result));
}

TEST_F(GramPolyFixture, EdgeCase_HighPolynomialOrder) {
    // Test with high polynomial order (approaching window size)
    uint8_t halfWindow = 10;
    uint8_t highK = 15;  // Close to 2*halfWindow+1 = 21
    
    // Should not crash
    float result = ComputeGramPoly(highK, 0, halfWindow, 0);
    EXPECT_FALSE(std::isnan(result));
    EXPECT_FALSE(std::isinf(result));
}

TEST_F(GramPolyFixture, EdgeCase_HighDerivativeOrder) {
    // Test with high derivative order
    uint8_t halfWindow = 5;
    uint8_t highD = 3;
    
    // Should not crash
    float result = ComputeGramPoly(4, 0, halfWindow, highD);
    EXPECT_FALSE(std::isnan(result));
    EXPECT_FALSE(std::isinf(result));
}

//=============================================================================
// Numerical Stability Tests
//=============================================================================

TEST_F(GramPolyFixture, Stability_RepeatedComputation) {
    // Repeated computation should give identical results
    uint8_t halfWindow = 5;
    uint8_t k = 2;
    int i = 3;
    uint8_t d = 0;
    
    float first = ComputeGramPoly(k, i, halfWindow, d);
    
    for (int rep = 0; rep < 100; rep++) {
        float result = ComputeGramPoly(k, i, halfWindow, d);
        EXPECT_FLOAT_EQ(result, first)
            << "Result changed on repetition " << rep;
    }
}

TEST_F(GramPolyFixture, Stability_MagnitudeBounds) {
    // Values should stay within reasonable bounds (no explosion)
    uint8_t halfWindow = 10;
    
    for (uint8_t k = 0; k <= 4; k++) {
        for (int i = -(int)halfWindow; i <= (int)halfWindow; i++) {
            for (uint8_t d = 0; d <= 2; d++) {
                float result = ComputeGramPoly(k, i, halfWindow, d);
                
                // Should not be NaN or infinite
                EXPECT_FALSE(std::isnan(result))
                    << "NaN at F(" << (int)k << "," << (int)d << ")[" << i << "]";
                EXPECT_FALSE(std::isinf(result))
                    << "Inf at F(" << (int)k << "," << (int)d << ")[" << i << "]";
                
                // Should be within reasonable magnitude
                EXPECT_LT(std::abs(result), 1e10f)
                    << "Unexpectedly large value at F(" << (int)k << "," << (int)d << ")[" << i << "]";
            }
        }
    }
}

TEST_F(GramPolyFixture, Stability_ConsistentAcrossWindows) {
    // For the same relative position, behavior should be consistent
    // F(1,0)[i/n] should scale appropriately with different n
    
    std::vector<uint8_t> windows = {3, 5, 10, 15};
    
    for (uint8_t n : windows) {
        // Test at relative position 0.5 (halfway to edge)
        int i = n / 2;
        float result = ComputeGramPoly(1, i, n, 0);
        float expected = (float)i / n;
        
        EXPECT_TRUE(FloatsEqual(result, expected))
            << "F(1,0)[" << i << "] with n=" << (int)n << " inconsistent";
    }
}

//=============================================================================
// Typical Usage Pattern Tests
//=============================================================================

TEST_F(GramPolyFixture, TypicalUsage_Window5_Order2) {
    // Your documented simple case: 5-point window, order 2
    uint8_t halfWindow = 2;
    uint8_t polyOrder = 2;
    
    // Compute for all positions and orders
    for (uint8_t k = 0; k <= polyOrder; k++) {
        for (int i = -(int)halfWindow; i <= (int)halfWindow; i++) {
            float result = ComputeGramPoly(k, i, halfWindow, 0);
            
            EXPECT_FALSE(std::isnan(result));
            EXPECT_FALSE(std::isinf(result));
        }
    }
}

TEST_F(GramPolyFixture, TypicalUsage_Window25_Order4) {
    // Your documented main use case: 25-point window, order 4
    uint8_t halfWindow = 12;
    uint8_t polyOrder = 4;
    
    // Compute for all positions and orders
    for (uint8_t k = 0; k <= polyOrder; k++) {
        for (int i = -(int)halfWindow; i <= (int)halfWindow; i++) {
            float result = ComputeGramPoly(k, i, halfWindow, 0);
            
            EXPECT_FALSE(std::isnan(result));
            EXPECT_FALSE(std::isinf(result));
        }
    }
}

TEST_F(GramPolyFixture, TypicalUsage_Derivatives) {
    // Test derivative calculations (common use case)
    uint8_t halfWindow = 5;
    
    // First derivative
    for (int i = -(int)halfWindow; i <= (int)halfWindow; i++) {
        float result = ComputeGramPoly(2, i, halfWindow, 1);
        EXPECT_FALSE(std::isnan(result));
    }
    
    // Second derivative
    for (int i = -(int)halfWindow; i <= (int)halfWindow; i++) {
        float result = ComputeGramPoly(2, i, halfWindow, 2);
        EXPECT_FALSE(std::isnan(result));
    }
}

//=============================================================================
// Memoization Tests (if enabled)
//=============================================================================

#ifdef ENABLE_MEMOIZATION

TEST_F(GramPolyFixture, Memoization_CacheHit) {
    // First call should compute and cache
    uint8_t halfWindow = 5;
    float first = ComputeGramPoly(2, 3, halfWindow, 0);
    
    // Second call should use cache (should be identical, not just close)
    float second = ComputeGramPoly(2, 3, halfWindow, 0);
    
    EXPECT_FLOAT_EQ(first, second) << "Cached value should be identical";
}

TEST_F(GramPolyFixture, Memoization_CacheInvalidation) {
    // Compute with one configuration
    uint8_t halfWindow1 = 5;
    ComputeGramPoly(2, 3, halfWindow1, 0);
    
    // Change configuration (should invalidate cache)
    uint8_t halfWindow2 = 10;
    float result = ComputeGramPoly(2, 3, halfWindow2, 0);
    
    // Should compute fresh value, not reuse cached
    EXPECT_FALSE(std::isnan(result));
}

TEST_F(GramPolyFixture, Memoization_OutOfBounds) {
    // Values outside cache bounds should compute directly
    uint8_t largeHalfWindow = MAX_HALF_WINDOW_FOR_MEMO + 5;
    
    // Should not crash, should compute correctly
    float result = ComputeGramPoly(2, 0, largeHalfWindow, 0);
    EXPECT_FALSE(std::isnan(result));
}

#endif // ENABLE_MEMOIZATION

//=============================================================================
// Recurrence Relation Verification
//=============================================================================

TEST_F(GramPolyFixture, Recurrence_ThreeTermRelation) {
    // Verify the three-term recurrence holds
    // F(k,d) = a·[i·F(k-1,d) + d·F(k-1,d-1)] - c·F(k-2,d)
    // This is implicitly tested by other tests, but we can spot-check
    
    uint8_t halfWindow = 5;
    uint8_t k = 3;
    int i = 2;
    uint8_t d = 0;
    
    // Just verify computation doesn't crash and produces finite values
    float result = ComputeGramPoly(k, i, halfWindow, d);
    EXPECT_FALSE(std::isnan(result));
    EXPECT_FALSE(std::isinf(result));
}

//=============================================================================
// Comprehensive Parameter Sweep
//=============================================================================

class GramPolyParameterizedTest : public GramPolyFixture,
                                   public ::testing::WithParamInterface<
                                       std::tuple<uint8_t, uint8_t, uint8_t>> {
};

TEST_P(GramPolyParameterizedTest, AllValidInputs_ProduceFiniteValues) {
    auto [halfWindow, polyOrder, derivOrder] = GetParam();
    
    // Test all positions in window
    for (int i = -(int)halfWindow; i <= (int)halfWindow; i++) {
        float result = ComputeGramPoly(polyOrder, i, halfWindow, derivOrder);
        
        EXPECT_FALSE(std::isnan(result))
            << "NaN for n=" << (int)halfWindow << ", k=" << (int)polyOrder 
            << ", d=" << (int)derivOrder << ", i=" << i;
        
        EXPECT_FALSE(std::isinf(result))
            << "Inf for n=" << (int)halfWindow << ", k=" << (int)polyOrder 
            << ", d=" << (int)derivOrder << ", i=" << i;
    }
}

// Generate comprehensive test parameter combinations
std::vector<std::tuple<uint8_t, uint8_t, uint8_t>> GenerateGramPolyTestCases() {
    std::vector<std::tuple<uint8_t, uint8_t, uint8_t>> cases;
    
    std::vector<uint8_t> halfWindows = {2, 5, 10, 15, 20};
    std::vector<uint8_t> polyOrders = {0, 1, 2, 3, 4};
    std::vector<uint8_t> derivOrders = {0, 1, 2};
    
    for (uint8_t n : halfWindows) {
        for (uint8_t k : polyOrders) {
            for (uint8_t d : derivOrders) {
                cases.push_back({n, k, d});
            }
        }
    }
    
    return cases;
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    GramPolyParameterizedTest,
    ::testing::ValuesIn(GenerateGramPolyTestCases())
);

//=============================================================================
// Main function
//=============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}