#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include "output_manager.hpp"

class OutputConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        config = OutputConfig{};
    }

    OutputConfig config;
};

TEST_F(OutputConfigTest, DefaultValues) {
    EXPECT_FALSE(config.verbose_logging);
    EXPECT_EQ(config.log_level, "info");
    EXPECT_EQ(config.performance_summary_interval, 30);
    EXPECT_TRUE(config.enable_csv_logging);
    EXPECT_EQ(config.csv_output_path, "output/frame_log.csv");
    EXPECT_TRUE(config.csv_comprehensive_mode);
    EXPECT_FALSE(config.enable_video_output);
    EXPECT_EQ(config.output_video_path, "output/dashcam_analytics.mp4");
    EXPECT_EQ(config.video_codec, "mp4v");
    EXPECT_EQ(config.output_fps, 30);
    EXPECT_TRUE(config.enable_overlay);
    EXPECT_TRUE(config.show_vehicle_boxes);
    EXPECT_TRUE(config.show_tracking_ids);
    EXPECT_TRUE(config.show_analytics_panel);
    EXPECT_TRUE(config.show_collision_warnings);
    EXPECT_FALSE(config.show_lane_detection);
    EXPECT_FLOAT_EQ(config.overlay_opacity, 0.8f);
    EXPECT_TRUE(config.use_memory_buffering);
    EXPECT_EQ(config.max_buffered_frames, 1000);
    EXPECT_EQ(config.buffer_flush_threshold, 500);
}

TEST_F(OutputConfigTest, CustomConfiguration) {
    config.verbose_logging = true;
    config.log_level = "debug";
    config.performance_summary_interval = 60;
    config.enable_video_output = true;
    config.output_fps = 60;
    config.overlay_opacity = 0.5f;
    config.max_buffered_frames = 2000;
    
    EXPECT_TRUE(config.verbose_logging);
    EXPECT_EQ(config.log_level, "debug");
    EXPECT_EQ(config.performance_summary_interval, 60);
    EXPECT_TRUE(config.enable_video_output);
    EXPECT_EQ(config.output_fps, 60);
    EXPECT_FLOAT_EQ(config.overlay_opacity, 0.5f);
    EXPECT_EQ(config.max_buffered_frames, 2000);
}

TEST_F(OutputConfigTest, OverlayConfiguration) {
    config.enable_overlay = false;
    config.show_vehicle_boxes = false;
    config.show_tracking_ids = false;
    config.show_analytics_panel = false;
    config.show_collision_warnings = false;
    config.show_lane_detection = true;
    
    EXPECT_FALSE(config.enable_overlay);
    EXPECT_FALSE(config.show_vehicle_boxes);
    EXPECT_FALSE(config.show_tracking_ids);
    EXPECT_FALSE(config.show_analytics_panel);
    EXPECT_FALSE(config.show_collision_warnings);
    EXPECT_TRUE(config.show_lane_detection);
}

TEST_F(OutputConfigTest, ValidationChecks) {
    // Test reasonable values
    EXPECT_GT(config.performance_summary_interval, 0);
    EXPECT_GT(config.output_fps, 0);
    EXPECT_LE(config.output_fps, 240);  // Reasonable upper limit
    EXPECT_GE(config.overlay_opacity, 0.0f);
    EXPECT_LE(config.overlay_opacity, 1.0f);
    EXPECT_GT(config.max_buffered_frames, 0);
    EXPECT_LE(config.buffer_flush_threshold, config.max_buffered_frames);
}

class PerformanceStatsTest : public ::testing::Test {
protected:
    void SetUp() override {
        stats = PerformanceStats{};
        stats.reset();  // Initialize timestamps
    }

    PerformanceStats stats;
};

TEST_F(PerformanceStatsTest, DefaultValues) {
    EXPECT_EQ(stats.total_frames, 0);
    EXPECT_DOUBLE_EQ(stats.total_inference_time, 0.0);
    EXPECT_DOUBLE_EQ(stats.total_e2e_time, 0.0);
    EXPECT_EQ(stats.missed_frames, 0);
    EXPECT_EQ(stats.motion_detected_frames, 0);
    EXPECT_EQ(stats.vehicles_detected, 0);
    EXPECT_EQ(stats.collision_warnings, 0);
}

TEST_F(PerformanceStatsTest, AverageCalculations) {
    stats.total_frames = 10;
    stats.total_inference_time = 25.0;  // 25ms total
    stats.total_e2e_time = 50.0;        // 50ms total
    
    EXPECT_DOUBLE_EQ(stats.getAvgInferenceTime(), 2.5);  // 25/10
    EXPECT_DOUBLE_EQ(stats.getAvgE2ETime(), 5.0);        // 50/10
}

TEST_F(PerformanceStatsTest, MissRateCalculation) {
    stats.total_frames = 100;
    stats.missed_frames = 5;
    
    EXPECT_DOUBLE_EQ(stats.getMissRate(), 5.0);  // 5/100 * 100
}

TEST_F(PerformanceStatsTest, ZeroFrameEdgeCase) {
    // Test with zero frames to avoid division by zero
    EXPECT_DOUBLE_EQ(stats.getAvgInferenceTime(), 0.0);
    EXPECT_DOUBLE_EQ(stats.getAvgE2ETime(), 0.0);
    EXPECT_DOUBLE_EQ(stats.getMissRate(), 0.0);
    EXPECT_DOUBLE_EQ(stats.getFPS(), 0.0);
}

TEST_F(PerformanceStatsTest, FPSCalculation) {
    stats.total_frames = 30;
    
    // Simulate 1 second delay
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    double fps = stats.getFPS();
    EXPECT_GT(fps, 0.0);
    EXPECT_LT(fps, 1000.0);  // Reasonable upper bound
}

TEST_F(PerformanceStatsTest, StatsReset) {
    // Set some values
    stats.total_frames = 100;
    stats.total_inference_time = 50.0;
    stats.missed_frames = 5;
    
    // Reset
    stats.reset();
    
    // Should be back to defaults
    EXPECT_EQ(stats.total_frames, 0);
    EXPECT_DOUBLE_EQ(stats.total_inference_time, 0.0);
    EXPECT_EQ(stats.missed_frames, 0);
}

TEST_F(PerformanceStatsTest, StatsAccumulation) {
    // Simulate processing multiple frames
    for (int i = 0; i < 5; ++i) {
        stats.total_frames++;
        stats.total_inference_time += 2.0;  // 2ms per frame
        stats.total_e2e_time += 5.0;        // 5ms per frame
        
        if (i % 2 == 0) {
            stats.motion_detected_frames++;
        }
        
        if (i % 3 == 0) {
            stats.vehicles_detected += 2;  // 2 vehicles detected
        }
    }
    
    EXPECT_EQ(stats.total_frames, 5);
    EXPECT_DOUBLE_EQ(stats.total_inference_time, 10.0);
    EXPECT_DOUBLE_EQ(stats.total_e2e_time, 25.0);
    EXPECT_EQ(stats.motion_detected_frames, 3);  // frames 0, 2, 4
    EXPECT_EQ(stats.vehicles_detected, 4);       // frames 0, 3 (2 vehicles each)
    
    EXPECT_DOUBLE_EQ(stats.getAvgInferenceTime(), 2.0);
    EXPECT_DOUBLE_EQ(stats.getAvgE2ETime(), 5.0);
}

// Mock tests for OutputManager
class OutputManagerMockTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary directory for test outputs
        test_dir = std::filesystem::temp_directory_path() / "framekeeper_output_tests";
        std::filesystem::create_directories(test_dir);
        
        config = OutputConfig{};
        config.enable_video_output = false;  // Disable for testing
        config.enable_csv_logging = true;
        config.csv_output_path = (test_dir / "test_log.csv").string();
        config.use_memory_buffering = true;
        config.max_buffered_frames = 10;  // Small buffer for testing
    }

    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists(test_dir)) {
            std::filesystem::remove_all(test_dir);
        }
    }

    OutputConfig config;
    std::filesystem::path test_dir;
};

TEST_F(OutputManagerMockTest, Construction) {
    EXPECT_NO_THROW({
        OutputManager manager(config);
    });
}

TEST_F(OutputManagerMockTest, ConfigurationAccess) {
    OutputManager manager(config);
    
    // Test that construction doesn't modify the config
    EXPECT_TRUE(config.enable_csv_logging);
    EXPECT_FALSE(config.enable_video_output);
    EXPECT_EQ(config.max_buffered_frames, 10);
}

TEST_F(OutputManagerMockTest, Initialization) {
    OutputManager manager(config);
    
    // Test initialization with common video resolutions
    EXPECT_TRUE(manager.initialize(1920, 1080));  // 1080p
    manager.cleanup();
    
    EXPECT_TRUE(manager.initialize(1280, 720));   // 720p
    manager.cleanup();
    
    EXPECT_TRUE(manager.initialize(640, 480));    // VGA
    manager.cleanup();
}

TEST_F(OutputManagerMockTest, InvalidResolution) {
    OutputManager manager(config);
    
    // Test with invalid resolutions
    EXPECT_FALSE(manager.initialize(0, 480));     // Zero width
    EXPECT_FALSE(manager.initialize(1920, 0));    // Zero height
    EXPECT_FALSE(manager.initialize(-1920, 1080)); // Negative dimensions
}

// Integration tests for output functionality
TEST(OutputIntegrationTest, PathValidation) {
    OutputConfig config;
    
    // Test various path formats
    std::vector<std::string> test_paths = {
        "output/test.csv",
        "/tmp/framekeeper_test.csv",
        "relative/path/test.csv",
        "./current_dir_test.csv"
    };
    
    for (const auto& path : test_paths) {
        config.csv_output_path = path;
        EXPECT_NO_THROW({
            OutputManager manager(config);
        });
    }
}

TEST(OutputIntegrationTest, VideoCodecValidation) {
    OutputConfig config;
    config.enable_video_output = true;
    
    // Test common video codecs
    std::vector<std::string> codecs = {
        "mp4v",
        "MJPG",
        "XVID",
        "H264"
    };
    
    for (const auto& codec : codecs) {
        config.video_codec = codec;
        EXPECT_NO_THROW({
            OutputManager manager(config);
        });
    }
}

TEST(OutputIntegrationTest, OverlayOpacityRange) {
    OutputConfig config;
    
    // Test opacity range validation
    std::vector<float> opacities = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    
    for (float opacity : opacities) {
        config.overlay_opacity = opacity;
        EXPECT_GE(config.overlay_opacity, 0.0f);
        EXPECT_LE(config.overlay_opacity, 1.0f);
        
        EXPECT_NO_THROW({
            OutputManager manager(config);
        });
    }
}

TEST(OutputIntegrationTest, BufferConfiguration) {
    OutputConfig config;
    config.use_memory_buffering = true;
    
    // Test different buffer sizes
    std::vector<size_t> buffer_sizes = {1, 10, 100, 1000};
    
    for (size_t size : buffer_sizes) {
        config.max_buffered_frames = size;
        config.buffer_flush_threshold = size / 2;  // Half of max
        
        EXPECT_LT(config.buffer_flush_threshold, config.max_buffered_frames);
        
        EXPECT_NO_THROW({
            OutputManager manager(config);
        });
    }
}

// Test performance stats calculations under realistic conditions
TEST(PerformanceStatsRealisticTest, TypicalScenario) {
    PerformanceStats stats;
    stats.reset();
    
    // Simulate 30 FPS processing for 1 second (30 frames)
    const int target_frames = 30;
    const double inference_time_ms = 2.5;  // Typical inference time
    const double e2e_time_ms = 15.0;       // Typical end-to-end time
    
    for (int i = 0; i < target_frames; ++i) {
        stats.total_frames++;
        stats.total_inference_time += inference_time_ms;
        stats.total_e2e_time += e2e_time_ms;
        
        // Simulate occasional missed frames (5% miss rate)
        if (i % 20 == 0) {
            stats.missed_frames++;
        }
        
        // Simulate motion detection (70% of frames)
        if (i % 10 != 0 && i % 10 != 1 && i % 10 != 2) {
            stats.motion_detected_frames++;
        }
        
        // Simulate vehicle detection (average 1.5 vehicles per frame)
        if (i % 2 == 0) {
            stats.vehicles_detected += 1;
        } else {
            stats.vehicles_detected += 2;
        }
    }
    
    // Verify calculations
    EXPECT_EQ(stats.total_frames, target_frames);
    EXPECT_DOUBLE_EQ(stats.getAvgInferenceTime(), inference_time_ms);
    EXPECT_DOUBLE_EQ(stats.getAvgE2ETime(), e2e_time_ms);
    EXPECT_DOUBLE_EQ(stats.getMissRate(), 10.0);  // 3 misses out of 30 = 10%
    EXPECT_EQ(stats.motion_detected_frames, 21);  // 70% of 30
    EXPECT_EQ(stats.vehicles_detected, 45);       // 1.5 * 30
}

TEST(PerformanceStatsRealisticTest, HighLoadScenario) {
    PerformanceStats stats;
    stats.reset();
    
    // Simulate high-load scenario with missed deadlines
    const int total_frames = 100;
    
    for (int i = 0; i < total_frames; ++i) {
        stats.total_frames++;
        
        // Simulate variable processing times
        double inference_time = 3.0 + (i % 5);  // 3-7ms
        double e2e_time = 20.0 + (i % 10);      // 20-29ms
        
        stats.total_inference_time += inference_time;
        stats.total_e2e_time += e2e_time;
        
        // Higher miss rate under load (15%)
        if (i % 7 == 0) {
            stats.missed_frames++;
        }
        
        // More vehicles detected under high load
        stats.vehicles_detected += 2;
        
        // Collision warnings occasionally
        if (i % 20 == 0) {
            stats.collision_warnings++;
        }
    }
    
    EXPECT_GT(stats.getAvgInferenceTime(), 3.0);
    EXPECT_LT(stats.getAvgInferenceTime(), 7.0);
    EXPECT_GT(stats.getAvgE2ETime(), 20.0);
    EXPECT_LT(stats.getAvgE2ETime(), 29.0);
    EXPECT_GT(stats.getMissRate(), 10.0);  // Higher than normal
    EXPECT_EQ(stats.vehicles_detected, 200);  // 2 per frame
    EXPECT_EQ(stats.collision_warnings, 5);   // Every 20 frames
}
