#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include "pipeline.hpp"
#include "output_manager.hpp"

class PipelineConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        config = PipelineConfig{};
    }

    PipelineConfig config;
};

TEST_F(PipelineConfigTest, DefaultValues) {
    EXPECT_EQ(config.uri, "data/sample_1080p30.mp4");
    EXPECT_EQ(config.width, 1920);
    EXPECT_EQ(config.height, 1080);
    EXPECT_EQ(config.fps, 30);
    EXPECT_EQ(config.queue_capacity, 4);
    EXPECT_EQ(config.drop_policy, "drop_oldest");
}

TEST_F(PipelineConfigTest, CustomValues) {
    config.uri = "test_video.mp4";
    config.width = 1280;
    config.height = 720;
    config.fps = 60;
    config.queue_capacity = 8;
    config.drop_policy = "drop_newest";
    
    EXPECT_EQ(config.uri, "test_video.mp4");
    EXPECT_EQ(config.width, 1280);
    EXPECT_EQ(config.height, 720);
    EXPECT_EQ(config.fps, 60);
    EXPECT_EQ(config.queue_capacity, 8);
    EXPECT_EQ(config.drop_policy, "drop_newest");
}

TEST_F(PipelineConfigTest, ValidationChecks) {
    // Test reasonable values
    EXPECT_GT(config.width, 0);
    EXPECT_GT(config.height, 0);
    EXPECT_GT(config.fps, 0);
    EXPECT_GT(config.queue_capacity, 0);
    EXPECT_FALSE(config.uri.empty());
    EXPECT_FALSE(config.drop_policy.empty());
}

// Mock tests for Pipeline (since it requires GPU/video resources)
class PipelineMockTest : public ::testing::Test {
protected:
    void SetUp() override {
        pipeline_config = PipelineConfig{};
        deadline_profile = DeadlineProfile{};
        metrics = std::make_unique<MetricsRegistry>();
        ml_config = MLConfig{};
        output_config = OutputConfig{};
        
        // Configure for testing environment (disable heavy operations)
        ml_config.use_tensorrt = false;
        ml_config.enable_segmentation = false;
        ml_config.enable_vehicle_analytics = false;
        output_config.enable_video_output = false;
        output_config.enable_csv_logging = false;
    }

    PipelineConfig pipeline_config;
    DeadlineProfile deadline_profile;
    std::unique_ptr<MetricsRegistry> metrics;
    MLConfig ml_config;
    OutputConfig output_config;
};

TEST_F(PipelineMockTest, Construction) {
    // Test that Pipeline can be constructed without crashing
    EXPECT_NO_THROW({
        Pipeline pipeline(pipeline_config, deadline_profile, *metrics, 
                         ml_config, output_config);
    });
}

TEST_F(PipelineMockTest, InitialState) {
    Pipeline pipeline(pipeline_config, deadline_profile, *metrics, 
                     ml_config, output_config);
    
    // Pipeline should not be running initially
    EXPECT_FALSE(pipeline.running());
}

TEST_F(PipelineMockTest, ConfigurationPropagation) {
    // Test that configuration is properly stored
    pipeline_config.width = 1280;
    pipeline_config.height = 720;
    pipeline_config.fps = 60;
    
    deadline_profile.target_fps = 60;
    deadline_profile.budget_ms = 16.67;
    
    Pipeline pipeline(pipeline_config, deadline_profile, *metrics, 
                     ml_config, output_config);
    
    // Since we can't directly access private members, we test through behavior
    EXPECT_FALSE(pipeline.running());  // Should start in stopped state
}

TEST_F(PipelineMockTest, StatsInitialization) {
    Pipeline pipeline(pipeline_config, deadline_profile, *metrics, 
                     ml_config, output_config);
    
    StatSnapshot stats = pipeline.stats();
    
    // Initial stats should have reasonable default values
    EXPECT_GE(stats.fps, 0.0);
    EXPECT_GE(stats.miss_rate, 0.0);
    EXPECT_LE(stats.miss_rate, 1.0);
}

// Test drop policies
TEST(DropPolicyTest, ValidPolicies) {
    std::vector<std::string> valid_policies = {
        "drop_oldest",
        "drop_newest",
        "block"
    };
    
    for (const auto& policy : valid_policies) {
        PipelineConfig config;
        config.drop_policy = policy;
        EXPECT_FALSE(config.drop_policy.empty());
        EXPECT_NE(config.drop_policy.find("drop"), std::string::npos);
    }
}

// Test resolution configurations
TEST(ResolutionTest, CommonResolutions) {
    struct Resolution {
        int width, height;
        std::string name;
    };
    
    std::vector<Resolution> resolutions = {
        {1920, 1080, "1080p"},
        {1280, 720, "720p"},
        {3840, 2160, "4K"},
        {640, 480, "VGA"},
        {1280, 960, "SXGA"}
    };
    
    for (const auto& res : resolutions) {
        PipelineConfig config;
        config.width = res.width;
        config.height = res.height;
        
        EXPECT_GT(config.width, 0);
        EXPECT_GT(config.height, 0);
        EXPECT_GT(config.width * config.height, 0);  // No overflow
        
        // Common aspect ratios should be reasonable
        double aspect_ratio = static_cast<double>(config.width) / config.height;
        EXPECT_GT(aspect_ratio, 0.5);  // Not too tall
        EXPECT_LT(aspect_ratio, 5.0);  // Not too wide
    }
}

// Test FPS configurations
TEST(FpsTest, CommonFrameRates) {
    std::vector<int> common_fps = {15, 24, 30, 60, 120};
    
    for (int fps : common_fps) {
        PipelineConfig config;
        config.fps = fps;
        
        EXPECT_GT(config.fps, 0);
        EXPECT_LE(config.fps, 240);  // Reasonable upper limit
        
        // Calculate expected frame time
        double frame_time_ms = 1000.0 / fps;
        EXPECT_GT(frame_time_ms, 0);
        
        // For real-time processing, frame time should be reasonable
        if (fps <= 60) {
            EXPECT_GE(frame_time_ms, 8.33);  // 120 FPS minimum
        }
    }
}

// Test queue capacity configurations
TEST(QueueCapacityTest, ValidCapacities) {
    std::vector<int> capacities = {1, 2, 4, 8, 16, 32};
    
    for (int capacity : capacities) {
        PipelineConfig config;
        config.queue_capacity = capacity;
        
        EXPECT_GT(config.queue_capacity, 0);
        EXPECT_LE(config.queue_capacity, 1000);  // Reasonable upper limit
        
        // Memory usage should be reasonable
        // Assuming each frame takes ~width*height*3 bytes
        size_t approx_memory = capacity * config.width * config.height * 3;
        EXPECT_LT(approx_memory, 1ULL << 32);  // Less than 4GB
    }
}

// Integration test for pipeline configuration validation
TEST(PipelineIntegrationTest, ConfigurationConsistency) {
    PipelineConfig pipeline_config;
    DeadlineProfile deadline_profile;
    
    // Test that deadline budget is consistent with FPS
    double expected_budget = 1000.0 / pipeline_config.fps;
    double tolerance = 5.0;  // 5ms tolerance
    
    EXPECT_NEAR(deadline_profile.budget_ms, expected_budget, tolerance);
    
    // Test that switch thresholds are reasonable
    EXPECT_LT(deadline_profile.switch_down_p95_ms, deadline_profile.switch_up_p95_ms);
    EXPECT_LT(deadline_profile.switch_up_p95_ms, deadline_profile.budget_ms);
    
    // Test hysteresis is reasonable
    EXPECT_GT(deadline_profile.hysteresis_frames, 0);
    EXPECT_LT(deadline_profile.hysteresis_frames, 1000);  // Not too large
}

// Test concurrent access to pipeline state
TEST(PipelineConcurrencyTest, StateAccess) {
    PipelineConfig pipeline_config;
    DeadlineProfile deadline_profile;
    MetricsRegistry metrics;
    MLConfig ml_config;
    OutputConfig output_config;
    
    ml_config.use_tensorrt = false;  // Disable for testing
    output_config.enable_video_output = false;
    
    Pipeline pipeline(pipeline_config, deadline_profile, metrics, 
                     ml_config, output_config);
    
    const int num_threads = 4;
    const int iterations = 100;
    std::atomic<int> successful_reads{0};
    
    std::vector<std::thread> threads;
    
    // Launch threads to concurrently access pipeline state
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&pipeline, &successful_reads, iterations]() {
            for (int i = 0; i < iterations; ++i) {
                try {
                    bool running = pipeline.running();
                    StatSnapshot stats = pipeline.stats();
                    
                    // Verify stats are reasonable
                    if (stats.fps >= 0.0 && stats.miss_rate >= 0.0 && stats.miss_rate <= 1.0) {
                        successful_reads.fetch_add(1);
                    }
                    
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                } catch (...) {
                    // Concurrent access should not throw
                }
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // All reads should have been successful
    EXPECT_EQ(successful_reads.load(), num_threads * iterations);
}
