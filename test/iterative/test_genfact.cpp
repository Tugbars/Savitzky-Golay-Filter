/**
 * @file test_genfact.cpp
 * @brief GoogleTest suite for Generalized Factorial (GenFact) computation and lookup table
 * 
 * Tests the GenFact lookup table initialization and computation accuracy.
 * GenFact(a,b) computes the falling factorial: a × (a-1) × ... × (a-b+1)
 * 
 * This is a critical foundation component - all filter weights depend on correct GenFact values.
 */

#include <gtest/gtest.h>
#include "savgolFilter.h"
#include <cmath>
#include <vector>

//=============================================================================
// Test Fixture for GenFact Tests
//=============================================================================

class GenFactFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Force initialization of the GenFact table
        // The implementation initializes lazily on first use, so we trigger it here
        InitGenFactTable();
    }

    void TearDown() override {
        // Nothing to clean up - static table persists
    }

    /**
     * @brief Reference implementation: direct computation of GenFact
     * Used to validate the lookup table values
     */
    float ComputeGenFactDirect(uint8_t upperLimit, uint8_t termCount) {
        if (termCount == 0) {
            return 1.0f;
        }
        if (upperLimit < termCount) {
            return 0.0f;
        }
        
        float product = 1.0f;
        uint8_t start = (upperLimit - termCount) + 1;
        for (uint8_t j = start; j <= upperLimit; j++) {
            product *= (float)j;
        }
        return product;
    }

    /**
     * @brief Helper to access the GenFact function under test
     */
    float CallGenFact(uint8_t upperLimit, uint8_t termCount) {
        return GenFact(upperLimit, termCount);
    }

    /**
     * @brief Compare two floats with relative tolerance
     */
    bool FloatsEqual(float a, float b, float relTol = 1e-5f) {
        if (a == b) return true;
        float maxAbs = std::max(std::abs(a), std::abs(b));
        return std::abs(a - b) <= relTol * maxAbs;
    }
};

//=============================================================================
// Basic Initialization Tests
//=============================================================================

TEST_F(GenFactFixture, Initialization_TableIsInitialized) {
    // After SetUp(), table should be initialized
    // Test by calling GenFact - should not crash and return valid values
    float result = CallGenFact(5, 2);
    EXPECT_GT(result, 0.0f);
}

TEST_F(GenFactFixture, Initialization_Idempotent) {
    // Multiple initializations should be safe
    InitGenFactTable();
    InitGenFactTable();
    InitGenFactTable();
    
    // Values should still be correct
    float result = CallGenFact(5, 3);
    EXPECT_FLOAT_EQ(result, 60.0f);
}

//=============================================================================
// Base Case Tests - Mathematical Properties
//=============================================================================

TEST_F(GenFactFixture, BaseCase_TermCountZero_ReturnsOne) {
    // GenFact(n, 0) = 1 for all n (empty product)
    EXPECT_FLOAT_EQ(CallGenFact(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(CallGenFact(1, 0), 1.0f);
    EXPECT_FLOAT_EQ(CallGenFact(5, 0), 1.0f);
    EXPECT_FLOAT_EQ(CallGenFact(10, 0), 1.0f);
    EXPECT_FLOAT_EQ(CallGenFact(32, 0), 1.0f);
    EXPECT_FLOAT_EQ(CallGenFact(64, 0), 1.0f);
}

TEST_F(GenFactFixture, BaseCase_TermCountOne_ReturnsUpperLimit) {
    // GenFact(n, 1) = n
    EXPECT_FLOAT_EQ(CallGenFact(0, 1), 0.0f);  // 0 < 1, should be 0
    EXPECT_FLOAT_EQ(CallGenFact(1, 1), 1.0f);
    EXPECT_FLOAT_EQ(CallGenFact(5, 1), 5.0f);
    EXPECT_FLOAT_EQ(CallGenFact(10, 1), 10.0f);
    EXPECT_FLOAT_EQ(CallGenFact(25, 1), 25.0f);
    EXPECT_FLOAT_EQ(CallGenFact(64, 1), 64.0f);
}

TEST_F(GenFactFixture, BaseCase_InsufficientTerms_ReturnsZero) {
    // GenFact(a, b) = 0 when a < b (not enough terms)
    EXPECT_FLOAT_EQ(CallGenFact(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(CallGenFact(2, 5), 0.0f);
    EXPECT_FLOAT_EQ(CallGenFact(3, 5), 0.0f);
    EXPECT_FLOAT_EQ(CallGenFact(5, 10), 0.0f);
    EXPECT_FLOAT_EQ(CallGenFact(10, 15), 0.0f);
}

//=============================================================================
// Known Value Tests - Hand-Calculated Results
//=============================================================================

TEST_F(GenFactFixture, KnownValue_5_3_Equals60) {
    // GenFact(5, 3) = 5 × 4 × 3 = 60
    EXPECT_FLOAT_EQ(CallGenFact(5, 3), 60.0f);
}

TEST_F(GenFactFixture, KnownValue_8_2_Equals56) {
    // GenFact(8, 2) = 8 × 7 = 56
    EXPECT_FLOAT_EQ(CallGenFact(8, 2), 56.0f);
}

TEST_F(GenFactFixture, KnownValue_10_4_Equals5040) {
    // GenFact(10, 4) = 10 × 9 × 8 × 7 = 5040
    EXPECT_FLOAT_EQ(CallGenFact(10, 4), 5040.0f);
}

TEST_F(GenFactFixture, KnownValue_6_6_Equals720) {
    // GenFact(6, 6) = 6! = 720
    EXPECT_FLOAT_EQ(CallGenFact(6, 6), 720.0f);
}

TEST_F(GenFactFixture, KnownValue_12_3_Equals1320) {
    // GenFact(12, 3) = 12 × 11 × 10 = 1320
    EXPECT_FLOAT_EQ(CallGenFact(12, 3), 1320.0f);
}

TEST_F(GenFactFixture, KnownValue_20_2_Equals380) {
    // GenFact(20, 2) = 20 × 19 = 380
    EXPECT_FLOAT_EQ(CallGenFact(20, 2), 380.0f);
}

TEST_F(GenFactFixture, KnownValue_15_5_Equals360360) {
    // GenFact(15, 5) = 15 × 14 × 13 × 12 × 11 = 360360
    EXPECT_FLOAT_EQ(CallGenFact(15, 5), 360360.0f);
}

//=============================================================================
// Boundary Value Tests
//=============================================================================

TEST_F(GenFactFixture, Boundary_MaxTableSize_DoesNotCrash) {
    // Test at the upper limit of the lookup table
    uint8_t maxIndex = GENFACT_TABLE_SIZE - 1;
    
    // Should not crash
    float result = CallGenFact(maxIndex, 0);
    EXPECT_FLOAT_EQ(result, 1.0f);
    
    // Should not crash for various term counts
    result = CallGenFact(maxIndex, 1);
    EXPECT_FLOAT_EQ(result, (float)maxIndex);
    
    result = CallGenFact(maxIndex, 2);
    EXPECT_GT(result, 0.0f);
}

TEST_F(GenFactFixture, Boundary_ZeroZero) {
    // GenFact(0, 0) is defined as 1 (empty product)
    EXPECT_FLOAT_EQ(CallGenFact(0, 0), 1.0f);
}

TEST_F(GenFactFixture, Boundary_EqualTerms) {
    // GenFact(n, n) = n!
    EXPECT_FLOAT_EQ(CallGenFact(1, 1), 1.0f);       // 1! = 1
    EXPECT_FLOAT_EQ(CallGenFact(2, 2), 2.0f);       // 2! = 2
    EXPECT_FLOAT_EQ(CallGenFact(3, 3), 6.0f);       // 3! = 6
    EXPECT_FLOAT_EQ(CallGenFact(4, 4), 24.0f);      // 4! = 24
    EXPECT_FLOAT_EQ(CallGenFact(5, 5), 120.0f);     // 5! = 120
}

//=============================================================================
// Accuracy Tests - Compare Lookup Table vs Direct Computation
//=============================================================================

TEST_F(GenFactFixture, Accuracy_LookupMatchesDirect_SmallValues) {
    // Test that lookup table matches direct computation for small values
    for (uint8_t upperLimit = 0; upperLimit <= 10; upperLimit++) {
        for (uint8_t termCount = 0; termCount <= 10; termCount++) {
            float lookupResult = CallGenFact(upperLimit, termCount);
            float directResult = ComputeGenFactDirect(upperLimit, termCount);
            
            EXPECT_TRUE(FloatsEqual(lookupResult, directResult))
                << "Mismatch at GenFact(" << (int)upperLimit << ", " << (int)termCount << "): "
                << "lookup=" << lookupResult << ", direct=" << directResult;
        }
    }
}

TEST_F(GenFactFixture, Accuracy_LookupMatchesDirect_MediumValues) {
    // Test for values typical in filter usage (halfWindow up to 20)
    for (uint8_t upperLimit = 10; upperLimit <= 40; upperLimit += 2) {
        for (uint8_t termCount = 0; termCount <= 8; termCount++) {
            float lookupResult = CallGenFact(upperLimit, termCount);
            float directResult = ComputeGenFactDirect(upperLimit, termCount);
            
            EXPECT_TRUE(FloatsEqual(lookupResult, directResult))
                << "Mismatch at GenFact(" << (int)upperLimit << ", " << (int)termCount << ")";
        }
    }
}

TEST_F(GenFactFixture, Accuracy_LookupMatchesDirect_LargeValues) {
    // Test for larger values (up to max supported window size)
    for (uint8_t upperLimit = 50; upperLimit < GENFACT_TABLE_SIZE; upperLimit += 5) {
        for (uint8_t termCount = 0; termCount <= 5; termCount++) {
            float lookupResult = CallGenFact(upperLimit, termCount);
            float directResult = ComputeGenFactDirect(upperLimit, termCount);
            
            EXPECT_TRUE(FloatsEqual(lookupResult, directResult, 1e-4f))  // Slightly relaxed for large values
                << "Mismatch at GenFact(" << (int)upperLimit << ", " << (int)termCount << ")";
        }
    }
}

//=============================================================================
// Mathematical Property Tests
//=============================================================================

TEST_F(GenFactFixture, Property_Symmetry_PascalTriangle) {
    // GenFact has relationships similar to Pascal's triangle
    // GenFact(n, k) relates to binomial coefficients
    // We test: GenFact(n, k) / k! = C(n, k) × (n-k)!
    // Which simplifies to specific identities we can verify
    
    // For example: GenFact(n, 2) = n × (n-1)
    for (uint8_t n = 2; n <= 20; n++) {
        float result = CallGenFact(n, 2);
        float expected = (float)n * (n - 1);
        EXPECT_FLOAT_EQ(result, expected);
    }
}

TEST_F(GenFactFixture, Property_Multiplicative) {
    // GenFact(n, a+b) = GenFact(n, a) × GenFact(n-a, b)
    // Test with small values to avoid overflow
    
    uint8_t n = 10;
    uint8_t a = 2;
    uint8_t b = 3;
    
    float lhs = CallGenFact(n, a + b);
    float rhs = CallGenFact(n, a) * CallGenFact(n - a, b);
    
    EXPECT_TRUE(FloatsEqual(lhs, rhs))
        << "GenFact(" << (int)n << ", " << (int)(a+b) << ") should equal "
        << "GenFact(" << (int)n << ", " << (int)a << ") × GenFact(" << (int)(n-a) << ", " << (int)b << ")";
}

TEST_F(GenFactFixture, Property_Monotonicity) {
    // For fixed termCount, GenFact should increase with upperLimit
    uint8_t termCount = 3;
    
    for (uint8_t upperLimit = termCount; upperLimit < 30; upperLimit++) {
        float current = CallGenFact(upperLimit, termCount);
        float next = CallGenFact(upperLimit + 1, termCount);
        
        EXPECT_LT(current, next)
            << "GenFact should increase: GenFact(" << (int)upperLimit << ", " << (int)termCount 
            << ") >= GenFact(" << (int)(upperLimit+1) << ", " << (int)termCount << ")";
    }
}

//=============================================================================
// Typical Usage Pattern Tests
//=============================================================================

TEST_F(GenFactFixture, TypicalUsage_FilterWindow5_PolyOrder2) {
    // halfWindow = 2 (window size = 5), polyOrder = 2
    // These are the GenFact values needed for weight calculation
    
    uint8_t halfWindow = 2;
    uint8_t twoM = 2 * halfWindow;  // 4
    
    for (uint8_t k = 0; k <= 2; k++) {
        float num = CallGenFact(twoM, k);          // GenFact(4, k)
        float den = CallGenFact(twoM + k + 1, k + 1);  // GenFact(4+k+1, k+1)
        
        // Should not be zero or NaN
        EXPECT_GT(num, 0.0f) << "Numerator should be positive for k=" << (int)k;
        EXPECT_GT(den, 0.0f) << "Denominator should be positive for k=" << (int)k;
        EXPECT_FALSE(std::isnan(num));
        EXPECT_FALSE(std::isnan(den));
        
        // Ratio should be well-defined
        float ratio = num / den;
        EXPECT_FALSE(std::isnan(ratio));
        EXPECT_FALSE(std::isinf(ratio));
    }
}

TEST_F(GenFactFixture, TypicalUsage_FilterWindow25_PolyOrder4) {
    // halfWindow = 12 (window size = 25), polyOrder = 4
    // Your documented use case
    
    uint8_t halfWindow = 12;
    uint8_t twoM = 2 * halfWindow;  // 24
    
    for (uint8_t k = 0; k <= 4; k++) {
        float num = CallGenFact(twoM, k);          // GenFact(24, k)
        float den = CallGenFact(twoM + k + 1, k + 1);  // GenFact(24+k+1, k+1)
        
        EXPECT_GT(num, 0.0f) << "Numerator should be positive for k=" << (int)k;
        EXPECT_GT(den, 0.0f) << "Denominator should be positive for k=" << (int)k;
        
        // Check that values are reasonable (not overflow)
        EXPECT_LT(num, 1e20f) << "Value suspiciously large, possible overflow";
        EXPECT_LT(den, 1e20f) << "Value suspiciously large, possible overflow";
    }
}

//=============================================================================
// Parameterized Tests - Comprehensive Coverage
//=============================================================================

class GenFactParameterizedTest : public GenFactFixture,
                                  public ::testing::WithParamInterface<std::pair<uint8_t, uint8_t>> {
};

TEST_P(GenFactParameterizedTest, LookupMatchesDirect) {
    auto [upperLimit, termCount] = GetParam();
    
    float lookupResult = CallGenFact(upperLimit, termCount);
    float directResult = ComputeGenFactDirect(upperLimit, termCount);
    
    EXPECT_TRUE(FloatsEqual(lookupResult, directResult, 1e-4f))
        << "GenFact(" << (int)upperLimit << ", " << (int)termCount << "): "
        << "lookup=" << lookupResult << ", direct=" << directResult;
}

// Generate test cases: all combinations up to reasonable sizes
std::vector<std::pair<uint8_t, uint8_t>> GenerateGenFactTestCases() {
    std::vector<std::pair<uint8_t, uint8_t>> cases;
    
    // Comprehensive coverage for small values (0-15)
    for (uint8_t upper = 0; upper <= 15; upper++) {
        for (uint8_t term = 0; term <= 15; term++) {
            cases.push_back({upper, term});
        }
    }
    
    // Sparse coverage for larger values (to keep test count reasonable)
    for (uint8_t upper = 16; upper < 64; upper += 4) {
        for (uint8_t term = 0; term <= 10; term += 2) {
            cases.push_back({upper, term});
        }
    }
    
    return cases;
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    GenFactParameterizedTest,
    ::testing::ValuesIn(GenerateGenFactTestCases())
);

//=============================================================================
// Edge Case and Special Value Tests
//=============================================================================

TEST_F(GenFactFixture, EdgeCase_LargeFactorial) {
    // Test larger factorials that might approach float limits
    // GenFact(20, 10) should be large but representable
    float result = CallGenFact(20, 10);
    EXPECT_GT(result, 0.0f);
    EXPECT_FALSE(std::isinf(result));
    EXPECT_FALSE(std::isnan(result));
}

TEST_F(GenFactFixture, EdgeCase_ConsecutiveValues) {
    // Verify that consecutive calls maintain consistency
    float prev = CallGenFact(10, 2);
    for (int i = 0; i < 100; i++) {
        float current = CallGenFact(10, 2);
        EXPECT_FLOAT_EQ(prev, current) << "Value changed between calls";
        prev = current;
    }
}

//=============================================================================
// Documentation Examples
//=============================================================================

TEST_F(GenFactFixture, DocumentedExample_60) {
    // From documentation: "GenFact(5,3) = 5 × 4 × 3 = 60"
    EXPECT_FLOAT_EQ(CallGenFact(5, 3), 60.0f);
}

TEST_F(GenFactFixture, DocumentedExample_56) {
    // From documentation: "GenFact(8,2) = 8 × 7 = 56"
    EXPECT_FLOAT_EQ(CallGenFact(8, 2), 56.0f);
}

TEST_F(GenFactFixture, DocumentedExample_Zero) {
    // From documentation: "GenFact(3,5) = 0 (not enough terms)"
    EXPECT_FLOAT_EQ(CallGenFact(3, 5), 0.0f);
}

//=============================================================================
// Main function
//=============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}