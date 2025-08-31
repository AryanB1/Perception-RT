#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include "metrics.hpp"

class RollingHistTest : public ::testing::Test {
protected:
    void SetUp() override {
        hist = std::make_unique<RollingHist>(5);  // Small capacity for testing
    }

    std::unique_ptr<RollingHist> hist;
};

TEST_F(RollingHistTest, EmptyHistogram) {
    EXPECT_EQ(hist->size(), 0);
    EXPECT_DOUBLE_EQ(hist->perc(50.0), 0.0);
    EXPECT_DOUBLE_EQ(hist->perc(95.0), 0.0);
}

TEST_F(RollingHistTest, SingleValue) {
    hist->add(42.0);
    EXPECT_EQ(hist->size(), 1);
    EXPECT_DOUBLE_EQ(hist->perc(0.0), 42.0);
    EXPECT_DOUBLE_EQ(hist->perc(50.0), 42.0);
    EXPECT_DOUBLE_EQ(hist->perc(100.0), 42.0);
}

TEST_F(RollingHistTest, MultipleValues) {
    // Add values: 1, 2, 3, 4, 5
    for (int i = 1; i <= 5; ++i) {
        hist->add(static_cast<double>(i));
    }
    
    EXPECT_EQ(hist->size(), 5);
    EXPECT_DOUBLE_EQ(hist->perc(0.0), 1.0);   // Min value
    EXPECT_DOUBLE_EQ(hist->perc(50.0), 3.0);  // Median
    EXPECT_DOUBLE_EQ(hist->perc(100.0), 5.0); // Max value
}

TEST_F(RollingHistTest, CapacityOverflow) {
    // Add 7 values to a capacity-5 histogram
    for (int i = 1; i <= 7; ++i) {
        hist->add(static_cast<double>(i));
    }
    
    EXPECT_EQ(hist->size(), 5);  // Should cap at 5
    // Should contain values 3, 4, 5, 6, 7 (oldest dropped)
    EXPECT_DOUBLE_EQ(hist->perc(0.0), 3.0);
    EXPECT_DOUBLE_EQ(hist->perc(100.0), 7.0);
}

TEST_F(RollingHistTest, PercentileCalculation) {
    // Add values: 10, 20, 30, 40, 50
    for (int i = 1; i <= 5; ++i) {
        hist->add(static_cast<double>(i * 10));
    }
    
    EXPECT_DOUBLE_EQ(hist->perc(25.0), 20.0);  // 25th percentile
    EXPECT_DOUBLE_EQ(hist->perc(75.0), 40.0);  // 75th percentile
}

TEST_F(RollingHistTest, ThreadSafety) {
    const int num_threads = 4;
    const int values_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    // Launch multiple threads to add values concurrently
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, values_per_thread]() {
            for (int i = 0; i < values_per_thread; ++i) {
                hist->add(static_cast<double>(t * values_per_thread + i));
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Should have exactly 5 values due to capacity limit
    EXPECT_EQ(hist->size(), 5);
    
    // All values should be valid (no corruption)
    double p50 = hist->perc(50.0);
    double p95 = hist->perc(95.0);
    EXPECT_GE(p50, 0.0);
    EXPECT_GE(p95, p50);
}

class MetricsRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry = std::make_unique<MetricsRegistry>();
    }

    std::unique_ptr<MetricsRegistry> registry;
};

TEST_F(MetricsRegistryTest, InitialState) {
    EXPECT_EQ(registry->frames_total(), 0);
    EXPECT_EQ(registry->deadline_miss_total(), 0);
}

TEST_F(MetricsRegistryTest, FrameCounters) {
    registry->inc_frame();
    registry->inc_frame();
    registry->inc_miss();
    
    EXPECT_EQ(registry->frames_total(), 2);
    EXPECT_EQ(registry->deadline_miss_total(), 1);
}

TEST_F(MetricsRegistryTest, TimingMetrics) {
    registry->add_pre(1.5);
    registry->add_inf(2.0);
    registry->add_post(0.8);
    registry->add_e2e(4.3);
    
    registry->inc_frame();
    
    auto snapshot = registry->snapshot(1.0);
    EXPECT_DOUBLE_EQ(snapshot.pre_p50, 1.5);
    EXPECT_DOUBLE_EQ(snapshot.inf_p50, 2.0);
    EXPECT_DOUBLE_EQ(snapshot.post_p50, 0.8);
    EXPECT_DOUBLE_EQ(snapshot.e2e_p50, 4.3);
}

TEST_F(MetricsRegistryTest, MissRateCalculation) {
    // Add 10 frames, 2 misses
    for (int i = 0; i < 10; ++i) {
        registry->inc_frame();
    }
    registry->inc_miss();
    registry->inc_miss();
    
    auto snapshot = registry->snapshot(1.0);
    EXPECT_DOUBLE_EQ(snapshot.miss_rate, 0.2);  // 2/10 = 0.2
}

TEST_F(MetricsRegistryTest, PrometheusOutput) {
    registry->add_e2e(5.0);
    registry->inc_frame();
    
    auto snapshot = registry->snapshot(1.0);
    std::string prometheus_text = registry->prometheus_text(snapshot);
    
    // Should contain required Prometheus metrics
    EXPECT_NE(prometheus_text.find("framekeeper_frames_processed_total"), std::string::npos);
    EXPECT_NE(prometheus_text.find("framekeeper_deadline_misses_total"), std::string::npos);
    EXPECT_NE(prometheus_text.find("framekeeper_latency_seconds"), std::string::npos);
}

TEST_F(MetricsRegistryTest, ConcurrentAccess) {
    const int num_threads = 4;
    const int operations_per_thread = 1000;
    
    std::vector<std::thread> threads;
    
    // Launch threads to perform concurrent operations
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, operations_per_thread]() {
            for (int i = 0; i < operations_per_thread; ++i) {
                registry->add_e2e(1.0);
                registry->inc_frame();
                if (i % 10 == 0) {
                    registry->inc_miss();
                }
            }
        });
    }
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify final counts
    EXPECT_EQ(registry->frames_total(), num_threads * operations_per_thread);
    EXPECT_EQ(registry->deadline_miss_total(), num_threads * (operations_per_thread / 10));
}
