/**
 * @file test_cache.cpp
 * @brief GoogleTest suite for Cache Mechanisms (Edge Weights + Memoization)
 * 
 * Tests two cache layers in the Savitzky-Golay implementation:
 * 1. Edge Weight Cache: Caches computed weights for leading/trailing edges
 * 2. Gram Polynomial Memoization: Caches Gram polynomial computations
 * 
 * These optimizations provide significant speedup but must maintain
 * correctness - cached values must exactly match freshly computed values.
 * 
 * Cache invalidation is critical: when parameters change, old cached
 * values must not be reused.
 */

#include <gtest/gtest.h>
#include "savgolFilter.h"
#include <cmath>
#include <vector>
#include <cstring>

//=============================================================================
// Test Fixture for Cache Tests
//=============================================================================

class CacheFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize GenFact table (required dependency)
        InitGenFactTable();
        
        // Initialize edge cache
        InitEdgeCacheIfNeeded();
    }

    void TearDown() override {
        // Clear edge cache for clean state between tests
        ClearEdgeCache();
    }

    /**
     * @brief Manually clear edge cache (reset all valid flags)
     */
    void ClearEdgeCache() {
        for (int i = 0; i < MAX_HALF_WINDOW_FOR_MEMO; i++) {
            leadingEdgeCache[i].valid = false;
        }
    }

    /**
     * @brief Compute weights and return them
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
     * @brief Check if two weight arrays are identical
     */
    bool WeightsIdentical(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) return false;
        
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) {  // Exact binary equality for cache test
                return false;
            }
        }
        return true;
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

#ifdef ENABLE_MEMOIZATION
    /**
     * @brief Clear memoization cache
     */
    void ClearMemoizationCache(uint8_t halfWindowSize, uint8_t polynomialOrder, 
                               uint8_t derivativeOrder) {
        ::ClearGramPolyCache(halfWindowSize, polynomialOrder, derivativeOrder);
    }

    /**
     * @brief Access memoization cache entry
     */
    const GramPolyCacheEntry* GetMemoEntry(int shiftedIndex, uint8_t polyOrder, 
                                          uint8_t derivOrder) {
        return ::GetGramPolyCacheEntry(shiftedIndex, polyOrder, derivOrder);
    }
#endif
};

//=============================================================================
// Edge Weight Cache Tests - Initialization
//=============================================================================

TEST_F(CacheFixture, EdgeCache_Initialization_Idempotent) {
    // Multiple initializations should be safe
    InitEdgeCacheIfNeeded();
    InitEdgeCacheIfNeeded();
    InitEdgeCacheIfNeeded();
    
    // Should still function correctly
    std::vector<float> weights;
    ComputeWeights(5, 5, 2, 0, weights);
    EXPECT_GT(weights.size(), 0);
}

TEST_F(CacheFixture, EdgeCache_InitialState_AllInvalid) {
    // After clear, all entries should be invalid
    ClearEdgeCache();
    
    // Check that cache entries start as invalid
    // We can't directly access the cache, but we can verify behavior
    // by computing weights twice and checking they're computed fresh
    std::vector<float> first, second;
    ComputeWeights(5, 5, 2, 0, first);
    ComputeWeights(5, 5, 2, 0, second);
    
    EXPECT_TRUE(WeightsIdentical(first, second))
        << "Fresh computation should be consistent";
}

//=============================================================================
// Edge Weight Cache Tests - Basic Functionality
//=============================================================================

TEST_F(CacheFixture, EdgeCache_FirstAccess_Computes) {
    // First access should compute and cache
    ClearEdgeCache();
    
    std::vector<float> weights;
    ComputeWeights(5, 5, 2, 0, weights);
    
    // Should produce valid weights
    EXPECT_EQ(weights.size(), 11);
    for (float w : weights) {
        EXPECT_FALSE(std::isnan(w));
    }
}

TEST_F(CacheFixture, EdgeCache_SecondAccess_UsesCache) {
    // Second access with same parameters should use cache
    ClearEdgeCache();
    
    uint8_t n = 5;
    uint8_t m = 2;
    uint8_t d = 0;
    uint8_t t = n;
    
    std::vector<float> first, second;
    ComputeWeights(n, t, m, d, first);
    ComputeWeights(n, t, m, d, second);
    
    // Should be identical (not just close - cached values are binary identical)
    EXPECT_TRUE(WeightsIdentical(first, second))
        << "Cached weights should be binary identical";
}

TEST_F(CacheFixture, EdgeCache_DifferentTargetPoints_DifferentWeights) {
    // Different target points should produce different weights
    uint8_t n = 5;
    
    std::vector<float> target3, target4, target5;
    ComputeWeights(n, 3, 2, 0, target3);
    ComputeWeights(n, 4, 2, 0, target4);
    ComputeWeights(n, 5, 2, 0, target5);
    
    // Should all be different
    EXPECT_FALSE(WeightsIdentical(target3, target4));
    EXPECT_FALSE(WeightsIdentical(target4, target5));
    EXPECT_FALSE(WeightsIdentical(target3, target5));
}

//=============================================================================
// Edge Weight Cache Tests - Parameter Validation
//=============================================================================

TEST_F(CacheFixture, EdgeCache_Invalidation_HalfWindowChange) {
    // Changing halfWindowSize should invalidate cache
    ClearEdgeCache();
    
    std::vector<float> weights1, weights2;
    
    // Compute with n=5
    ComputeWeights(5, 5, 2, 0, weights1);
    
    // Compute with n=6 (should not use cached weights from n=5)
    ComputeWeights(6, 6, 2, 0, weights2);
    
    // Should produce valid but different weights
    EXPECT_EQ(weights1.size(), 11);
    EXPECT_EQ(weights2.size(), 13);
    EXPECT_FALSE(std::isnan(weights2[0]));
}

TEST_F(CacheFixture, EdgeCache_Invalidation_PolyOrderChange) {
    // Changing polynomial order should invalidate cache
    ClearEdgeCache();
    
    uint8_t n = 5;
    std::vector<float> order2, order3;
    
    ComputeWeights(n, n, 2, 0, order2);
    ComputeWeights(n, n, 3, 0, order3);
    
    // Should produce different weights
    EXPECT_FALSE(WeightsIdentical(order2, order3));
}

TEST_F(CacheFixture, EdgeCache_Invalidation_DerivOrderChange) {
    // Changing derivative order should invalidate cache
    ClearEdgeCache();
    
    uint8_t n = 5;
    std::vector<float> smooth, deriv;
    
    ComputeWeights(n, n, 2, 0, smooth);
    ComputeWeights(n, n, 2, 1, deriv);
    
    // Should produce different weights
    EXPECT_FALSE(WeightsIdentical(smooth, deriv));
}

TEST_F(CacheFixture, EdgeCache_Invalidation_TargetPointChange) {
    // Changing target point should invalidate cache
    ClearEdgeCache();
    
    uint8_t n = 5;
    std::vector<float> target4, target5;
    
    ComputeWeights(n, 4, 2, 0, target4);
    ComputeWeights(n, 5, 2, 0, target5);
    
    // Should produce different weights
    EXPECT_FALSE(WeightsIdentical(target4, target5));
}

TEST_F(CacheFixture, EdgeCache_Validation_AllParamsMustMatch) {
    // Cache is only valid if ALL parameters match
    ClearEdgeCache();
    
    uint8_t n = 5;
    std::vector<float> original, recomputed;
    
    // Compute and cache with specific parameters
    ComputeWeights(n, n, 2, 0, original);
    
    // Compute with ANY different parameter
    ComputeWeights(n + 1, n + 1, 2, 0, recomputed);  // Different n
    
    // Now recompute with original parameters - should get original weights
    std::vector<float> checkOriginal;
    ComputeWeights(n, n, 2, 0, checkOriginal);
    
    EXPECT_TRUE(WeightsIdentical(original, checkOriginal))
        << "Same parameters should produce same weights";
}

//=============================================================================
// Edge Weight Cache Tests - Correctness
//=============================================================================

TEST_F(CacheFixture, EdgeCache_Correctness_MultiplePositions) {
    // Test caching for all edge positions
    ClearEdgeCache();
    
    uint8_t n = 10;
    
    // Compute weights for several edge positions
    for (int i = 0; i < std::min((int)n, 5); i++) {
        uint8_t target = n - i;
        
        std::vector<float> first, second;
        
        // First access - computes and caches
        ComputeWeights(n, target, 2, 0, first);
        
        // Second access - should use cache
        ComputeWeights(n, target, 2, 0, second);
        
        EXPECT_TRUE(WeightsIdentical(first, second))
            << "Cached weights incorrect for target=" << (int)target;
    }
}

TEST_F(CacheFixture, EdgeCache_Correctness_RepeatedAccess) {
    // Repeated access should always return identical weights
    ClearEdgeCache();
    
    uint8_t n = 5;
    std::vector<float> first;
    ComputeWeights(n, n, 2, 0, first);
    
    // Access many times
    for (int rep = 0; rep < 20; rep++) {
        std::vector<float> current;
        ComputeWeights(n, n, 2, 0, current);
        
        EXPECT_TRUE(WeightsIdentical(first, current))
            << "Weights changed on repetition " << rep;
    }
}

TEST_F(CacheFixture, EdgeCache_Correctness_OutOfBounds) {
    // Positions beyond cache capacity should still work (computed directly)
    
    // If edge position >= MAX_HALF_WINDOW_FOR_MEMO, no caching
    if (MAX_HALF_WINDOW_FOR_MEMO < 30) {
        uint8_t largeN = MAX_HALF_WINDOW_FOR_MEMO + 5;
        
        std::vector<float> weights;
        ComputeWeights(largeN, largeN, 2, 0, weights);
        
        // Should still produce valid weights (just not cached)
        EXPECT_EQ(weights.size(), 2 * largeN + 1);
        for (float w : weights) {
            EXPECT_FALSE(std::isnan(w));
        }
    }
}

//=============================================================================
// Memoization Tests (Gram Polynomial Cache)
//=============================================================================

#ifdef ENABLE_MEMOIZATION

TEST_F(CacheFixture, Memo_CacheEntry_InitiallyNotComputed) {
    // Clear cache
    ClearMemoizationCache(5, 4, 0);
    
    // Check that entries are marked as not computed
    const GramPolyCacheEntry* entry = GetMemoEntry(5, 2, 0);
    if (entry != nullptr) {
        EXPECT_FALSE(entry->isComputed)
            << "Cache entry should initially be uncomputed";
    }
}

TEST_F(CacheFixture, Memo_FirstComputation_PopulatesCache) {
    // Clear and compute
    uint8_t n = 5;
    ClearMemoizationCache(n, 4, 0);
    
    // Trigger computation
    std::vector<float> weights;
    ComputeWeights(n, 0, 2, 0, weights);
    
    // Check that some cache entries are now populated
    int shiftedIndex = n;  // Center point
    const GramPolyCacheEntry* entry = GetMemoEntry(shiftedIndex, 0, 0);
    if (entry != nullptr) {
        EXPECT_TRUE(entry->isComputed)
            << "Cache entry should be computed after weight calculation";
    }
}

TEST_F(CacheFixture, Memo_CachedValue_MatchesFresh) {
    uint8_t n = 5;
    uint8_t k = 2;
    int i = 3;
    uint8_t d = 0;
    
    // Clear cache and compute
    ClearMemoizationCache(n, 4, d);
    
    GramPolyContext ctx;
    ctx.halfWindowSize = n;
    ctx.targetPoint = 0;
    ctx.derivativeOrder = d;
    
    // First computation - computes and caches
    float first = MemoizedGramPoly(k, i, &ctx);
    
    // Second computation - uses cache
    float second = MemoizedGramPoly(k, i, &ctx);
    
    // Should be exactly equal (binary identical from cache)
    EXPECT_EQ(first, second)
        << "Cached value should be binary identical";
}

TEST_F(CacheFixture, Memo_DifferentParams_DifferentValues) {
    uint8_t n = 5;
    
    ClearMemoizationCache(n, 4, 0);
    
    GramPolyContext ctx;
    ctx.halfWindowSize = n;
    ctx.targetPoint = 0;
    ctx.derivativeOrder = 0;
    
    // Compute for different k values
    float k0 = MemoizedGramPoly(0, 3, &ctx);
    float k1 = MemoizedGramPoly(1, 3, &ctx);
    float k2 = MemoizedGramPoly(2, 3, &ctx);
    
    // Should all be different
    EXPECT_NE(k0, k1);
    EXPECT_NE(k1, k2);
    EXPECT_NE(k0, k2);
}

TEST_F(CacheFixture, Memo_CacheInvalidation_OnParamChange) {
    // Compute with one configuration
    uint8_t n1 = 5;
    ClearMemoizationCache(n1, 4, 0);
    
    std::vector<float> weights1;
    ComputeWeights(n1, 0, 2, 0, weights1);
    
    // Change configuration (invalidates cache)
    uint8_t n2 = 10;
    ClearMemoizationCache(n2, 4, 0);
    
    std::vector<float> weights2;
    ComputeWeights(n2, 0, 2, 0, weights2);
    
    // Should produce valid but different weights
    EXPECT_EQ(weights1.size(), 2 * n1 + 1);
    EXPECT_EQ(weights2.size(), 2 * n2 + 1);
}

TEST_F(CacheFixture, Memo_OutOfBounds_ComputesDirectly) {
    // Values outside cache bounds should compute directly (not crash)
    uint8_t n = MAX_HALF_WINDOW_FOR_MEMO + 5;
    
    GramPolyContext ctx;
    ctx.halfWindowSize = n;
    ctx.targetPoint = 0;
    ctx.derivativeOrder = 0;
    
    // Should not crash, should compute correctly
    float result = MemoizedGramPoly(2, 0, &ctx);
    EXPECT_FALSE(std::isnan(result));
    EXPECT_FALSE(std::isinf(result));
}

TEST_F(CacheFixture, Memo_PolyOrderOutOfBounds_ComputesDirectly) {
    // Polynomial order beyond cache capacity
    uint8_t k = MAX_POLY_ORDER_FOR_MEMO + 2;
    
    GramPolyContext ctx;
    ctx.halfWindowSize = 5;
    ctx.targetPoint = 0;
    ctx.derivativeOrder = 0;
    
    float result = MemoizedGramPoly(k, 0, &ctx);
    EXPECT_FALSE(std::isnan(result));
}

TEST_F(CacheFixture, Memo_DerivOrderOutOfBounds_ComputesDirectly) {
    // Derivative order beyond cache capacity
    uint8_t d = MAX_DERIVATIVE_FOR_MEMO + 1;
    
    GramPolyContext ctx;
    ctx.halfWindowSize = 5;
    ctx.targetPoint = 0;
    ctx.derivativeOrder = d;
    
    float result = MemoizedGramPoly(2, 0, &ctx);
    EXPECT_FALSE(std::isnan(result));
}

TEST_F(CacheFixture, Memo_CacheEfficiency_ReducesComputations) {
    // Memoization should drastically reduce number of computations
    // We can't directly count computations, but we can verify consistency
    
    uint8_t n = 12;
    uint8_t m = 4;
    
    ClearMemoizationCache(n, m, 0);
    
    // Compute weights multiple times with same parameters
    std::vector<float> first, second, third;
    ComputeWeights(n, 0, m, 0, first);
    ComputeWeights(n, 0, m, 0, second);
    ComputeWeights(n, 0, m, 0, third);
    
    // All should be identical (proving cache is working)
    EXPECT_TRUE(WeightsIdentical(first, second));
    EXPECT_TRUE(WeightsIdentical(second, third));
}

TEST_F(CacheFixture, Memo_AllPositions_Cacheable) {
    // All positions in window should be cacheable
    uint8_t n = 10;
    
    ClearMemoizationCache(n, 4, 0);
    
    GramPolyContext ctx;
    ctx.halfWindowSize = n;
    ctx.targetPoint = 0;
    ctx.derivativeOrder = 0;
    
    std::vector<float> firstPass, secondPass;
    
    // First pass - populate cache
    for (int i = -(int)n; i <= (int)n; i++) {
        float value = MemoizedGramPoly(2, i, &ctx);
        firstPass.push_back(value);
    }
    
    // Second pass - use cache
    for (int i = -(int)n; i <= (int)n; i++) {
        float value = MemoizedGramPoly(2, i, &ctx);
        secondPass.push_back(value);
    }
    
    // Should be identical
    EXPECT_EQ(firstPass.size(), secondPass.size());
    for (size_t i = 0; i < firstPass.size(); i++) {
        EXPECT_EQ(firstPass[i], secondPass[i])
            << "Mismatch at position " << i;
    }
}

#endif // ENABLE_MEMOIZATION

//=============================================================================
// Combined Cache Tests - Both Layers Working Together
//=============================================================================

TEST_F(CacheFixture, Combined_EdgeAndMemo_WorkTogether) {
    // Both cache layers should work correctly together
    ClearEdgeCache();
    
    uint8_t n = 5;
    
#ifdef ENABLE_MEMOIZATION
    ClearMemoizationCache(n, 4, 0);
#endif
    
    // Compute edge weights (uses both caches)
    std::vector<float> first, second, third;
    ComputeWeights(n, n, 2, 0, first);
    ComputeWeights(n, n, 2, 0, second);
    ComputeWeights(n, n, 2, 0, third);
    
    // All should be identical
    EXPECT_TRUE(WeightsIdentical(first, second));
    EXPECT_TRUE(WeightsIdentical(second, third));
}

TEST_F(CacheFixture, Combined_MultipleConfigurations_Sequential) {
    // Using different configurations in sequence should work correctly
    ClearEdgeCache();
    
    std::vector<uint8_t> configs = {3, 5, 8, 10};
    
    for (uint8_t n : configs) {
#ifdef ENABLE_MEMOIZATION
        ClearMemoizationCache(n, 4, 0);
#endif
        
        std::vector<float> smooth, deriv;
        ComputeWeights(n, 0, 2, 0, smooth);
        ComputeWeights(n, 0, 2, 1, deriv);
        
        // Both should be valid
        EXPECT_EQ(smooth.size(), 2 * n + 1);
        EXPECT_EQ(deriv.size(), 2 * n + 1);
        
        for (float w : smooth) {
            EXPECT_FALSE(std::isnan(w));
        }
        for (float w : deriv) {
            EXPECT_FALSE(std::isnan(w));
        }
    }
}

TEST_F(CacheFixture, Combined_Interleaved_Configurations) {
    // Interleaving different configurations should maintain correctness
    ClearEdgeCache();
    
    uint8_t n1 = 5;
    uint8_t n2 = 8;
    
#ifdef ENABLE_MEMOIZATION
    ClearMemoizationCache(std::max(n1, n2), 4, 0);
#endif
    
    std::vector<float> config1_first, config2_first;
    std::vector<float> config1_second, config2_second;
    
    // Compute config 1
    ComputeWeights(n1, 0, 2, 0, config1_first);
    
    // Compute config 2
    ComputeWeights(n2, 0, 2, 0, config2_first);
    
    // Compute config 1 again
    ComputeWeights(n1, 0, 2, 0, config1_second);
    
    // Compute config 2 again
    ComputeWeights(n2, 0, 2, 0, config2_second);
    
    // Each configuration should be self-consistent
    EXPECT_TRUE(WeightsIdentical(config1_first, config1_second))
        << "Config 1 inconsistent";
    EXPECT_TRUE(WeightsIdentical(config2_first, config2_second))
        << "Config 2 inconsistent";
    
    // Configurations should be different
    EXPECT_FALSE(WeightsIdentical(config1_first, config2_first))
        << "Different configs should produce different weights";
}

//=============================================================================
// Performance/Stress Tests
//=============================================================================

TEST_F(CacheFixture, Stress_ManyRepeatedCalls) {
    // Many repeated calls should consistently use cache
    ClearEdgeCache();
    
    uint8_t n = 5;
    
#ifdef ENABLE_MEMOIZATION
    ClearMemoizationCache(n, 4, 0);
#endif
    
    std::vector<float> first;
    ComputeWeights(n, n, 2, 0, first);
    
    // Repeat many times
    for (int rep = 0; rep < 100; rep++) {
        std::vector<float> current;
        ComputeWeights(n, n, 2, 0, current);
        
        EXPECT_TRUE(WeightsIdentical(first, current))
            << "Weights changed on repetition " << rep;
    }
}

TEST_F(CacheFixture, Stress_AllEdgePositions) {
    // Test caching for all possible edge positions
    uint8_t n = 15;
    
    ClearEdgeCache();
    
#ifdef ENABLE_MEMOIZATION
    ClearMemoizationCache(n, 4, 0);
#endif
    
    // Compute for all positions that would be used in leading edge
    for (int i = 0; i < std::min((int)n, (int)MAX_HALF_WINDOW_FOR_MEMO); i++) {
        uint8_t target = n - i;
        
        std::vector<float> first, second;
        ComputeWeights(n, target, 2, 0, first);
        ComputeWeights(n, target, 2, 0, second);
        
        EXPECT_TRUE(WeightsIdentical(first, second))
            << "Cache failure at position " << i << ", target=" << (int)target;
    }
}

//=============================================================================
// Main function
//=============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}