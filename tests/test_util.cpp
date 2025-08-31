#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include "util.hpp"

class ConfigLoadTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for test files
        test_dir = std::filesystem::temp_directory_path() / "framekeeper_tests";
        std::filesystem::create_directories(test_dir);
    }

    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists(test_dir)) {
            std::filesystem::remove_all(test_dir);
        }
    }

    void createTestConfig(const std::string& filename, const std::string& content) {
        std::ofstream file(test_dir / filename);
        file << content;
        file.close();
    }

    std::filesystem::path test_dir;
};

TEST_F(ConfigLoadTest, BasicConfigLoad) {
    const std::string config_content = R"(
input:
  uri: "test_video.mp4"
  width: 1920
  height: 1080
  fps: 30

pipeline:
  budget_ms: 33.0
  target_fps: 30
  queue_capacity: 4
  drop_policy: "drop_oldest"

controller:
  switch_up_p95_ms: 28.0
  switch_down_p95_ms: 22.0
  hysteresis_frames: 90

telemetry:
  metrics_port: 9090
)";

    createTestConfig("basic_config.yaml", config_content);
    
    AppConfig config = load_config((test_dir / "basic_config.yaml").string());
    
    // Test input configuration
    EXPECT_EQ(config.pipeline.uri, "test_video.mp4");
    EXPECT_EQ(config.pipeline.width, 1920);
    EXPECT_EQ(config.pipeline.height, 1080);
    EXPECT_EQ(config.pipeline.fps, 30);
    
    // Test pipeline configuration
    EXPECT_DOUBLE_EQ(config.deadline.budget_ms, 33.0);
    EXPECT_EQ(config.deadline.target_fps, 30);
    EXPECT_EQ(config.pipeline.queue_capacity, 4);
    EXPECT_EQ(config.pipeline.drop_policy, "drop_oldest");
    
    // Test controller configuration
    EXPECT_DOUBLE_EQ(config.deadline.switch_up_p95_ms, 28.0);
    EXPECT_DOUBLE_EQ(config.deadline.switch_down_p95_ms, 22.0);
    EXPECT_EQ(config.deadline.hysteresis_frames, 90);
    
    // Test telemetry
    EXPECT_EQ(config.metrics_port, 9090);
}

TEST_F(ConfigLoadTest, MLEngineConfig) {
    const std::string config_content = R"(
ml_engine:
  yolo_model_path: "models/yolo11s.onnx"
  yolo_engine_path: "models/yolo11s.engine"
  segmentation_model_path: "models/deeplabv3.onnx"
  use_tensorrt: true
  use_fp16: true
  max_batch_size: 2
  max_workspace_size: 2147483648
  detection_threshold: 0.6
  nms_threshold: 0.3
  max_detections: 150
  max_corners: 120
  optical_flow_threshold: 1.5
  segmentation_threshold: 0.6
  input_width: 416
  input_height: 416
  enable_detection: true
  enable_optical_flow: false
  enable_segmentation: true
  enable_vehicle_analytics: true
  enable_tracking: true
  enable_proximity_detection: true
  focus_vehicle_detection: true
  vehicle_classes: [1, 2, 3, 5, 7, 8]
)";

    createTestConfig("ml_config.yaml", config_content);
    
    AppConfig config = load_config((test_dir / "ml_config.yaml").string());
    
    // Test ML Engine configuration
    EXPECT_EQ(config.ml_config.yolo_model_path, "models/yolo11s.onnx");
    EXPECT_EQ(config.ml_config.yolo_engine_path, "models/yolo11s.engine");
    EXPECT_EQ(config.ml_config.segmentation_model_path, "models/deeplabv3.onnx");
    EXPECT_TRUE(config.ml_config.use_tensorrt);
    EXPECT_TRUE(config.ml_config.use_fp16);
    EXPECT_EQ(config.ml_config.max_batch_size, 2);
    EXPECT_EQ(config.ml_config.max_workspace_size, 2147483648ULL);
    EXPECT_FLOAT_EQ(config.ml_config.detection_threshold, 0.6f);
    EXPECT_FLOAT_EQ(config.ml_config.nms_threshold, 0.3f);
    EXPECT_EQ(config.ml_config.max_detections, 150);
    EXPECT_EQ(config.ml_config.max_corners, 120);
    EXPECT_FLOAT_EQ(config.ml_config.optical_flow_threshold, 1.5f);
    EXPECT_FLOAT_EQ(config.ml_config.segmentation_threshold, 0.6f);
    EXPECT_EQ(config.ml_config.input_size.width, 416);
    EXPECT_EQ(config.ml_config.input_size.height, 416);
    EXPECT_TRUE(config.ml_config.enable_detection);
    EXPECT_FALSE(config.ml_config.enable_optical_flow);
    EXPECT_TRUE(config.ml_config.enable_segmentation);
    EXPECT_TRUE(config.ml_config.enable_vehicle_analytics);
    EXPECT_TRUE(config.ml_config.enable_tracking);
    EXPECT_TRUE(config.ml_config.enable_proximity_detection);
    EXPECT_TRUE(config.ml_config.focus_vehicle_detection);
    
    std::vector<int> expected_classes = {1, 2, 3, 5, 7, 8};
    EXPECT_EQ(config.ml_config.vehicle_classes, expected_classes);
}

TEST_F(ConfigLoadTest, OutputConfig) {
    const std::string config_content = R"(
output:
  logging:
    verbose_logging: true
    performance_summary_interval: 60
  csv:
    enable_csv_logging: true
    csv_output_path: "output/test_log.csv"
    csv_comprehensive_mode: true
  enable_video_output: true
  output_video_path: "output/test_output.mp4"
  video_codec: "MJPG"
  output_fps: 30
  use_memory_buffering: true
  max_buffered_frames: 1000
  buffer_flush_threshold: 100
  enable_overlay: true
  show_vehicle_boxes: true
  show_tracking_ids: true
  show_analytics_panel: true
  show_collision_warnings: true
  show_lane_detection: true
  overlay_opacity: 0.8
)";

    createTestConfig("output_config.yaml", config_content);
    
    AppConfig config = load_config((test_dir / "output_config.yaml").string());
    
    // Test output configuration
    EXPECT_TRUE(config.output_config.verbose_logging);
    EXPECT_EQ(config.output_config.performance_summary_interval, 60);
    EXPECT_TRUE(config.output_config.enable_csv_logging);
    EXPECT_EQ(config.output_config.csv_output_path, "output/test_log.csv");
    EXPECT_TRUE(config.output_config.csv_comprehensive_mode);
    EXPECT_TRUE(config.output_config.enable_video_output);
    EXPECT_EQ(config.output_config.output_video_path, "output/test_output.mp4");
    EXPECT_EQ(config.output_config.video_codec, "MJPG");
    EXPECT_EQ(config.output_config.output_fps, 30);
    EXPECT_TRUE(config.output_config.use_memory_buffering);
    EXPECT_EQ(config.output_config.max_buffered_frames, 1000);
    EXPECT_EQ(config.output_config.buffer_flush_threshold, 100);
    EXPECT_TRUE(config.output_config.enable_overlay);
    EXPECT_TRUE(config.output_config.show_vehicle_boxes);
    EXPECT_TRUE(config.output_config.show_tracking_ids);
    EXPECT_TRUE(config.output_config.show_analytics_panel);
    EXPECT_TRUE(config.output_config.show_collision_warnings);
    EXPECT_TRUE(config.output_config.show_lane_detection);
    EXPECT_FLOAT_EQ(config.output_config.overlay_opacity, 0.8f);
}

TEST_F(ConfigLoadTest, EmptyConfig) {
    const std::string config_content = "{}";
    
    createTestConfig("empty_config.yaml", config_content);
    
    AppConfig config = load_config((test_dir / "empty_config.yaml").string());
    
    // Should use default values for all settings
    EXPECT_EQ(config.pipeline.uri, "data/sample_1080p30.mp4");  // Default from types
    EXPECT_EQ(config.pipeline.width, 1920);
    EXPECT_EQ(config.deadline.target_fps, 30);
    EXPECT_DOUBLE_EQ(config.deadline.budget_ms, 33.0);
    EXPECT_EQ(config.metrics_port, 9090);
}

TEST_F(ConfigLoadTest, PartialConfig) {
    const std::string config_content = R"(
input:
  uri: "custom_video.mp4"
  width: 1280

pipeline:
  budget_ms: 25.0

ml_engine:
  detection_threshold: 0.8
  enable_segmentation: true
)";

    createTestConfig("partial_config.yaml", config_content);
    
    AppConfig config = load_config((test_dir / "partial_config.yaml").string());
    
    // Should override specified values
    EXPECT_EQ(config.pipeline.uri, "custom_video.mp4");
    EXPECT_EQ(config.pipeline.width, 1280);
    EXPECT_DOUBLE_EQ(config.deadline.budget_ms, 25.0);
    EXPECT_FLOAT_EQ(config.ml_config.detection_threshold, 0.8f);
    EXPECT_TRUE(config.ml_config.enable_segmentation);
    
    // Should keep defaults for unspecified values
    EXPECT_EQ(config.pipeline.height, 1080);  // Default
    EXPECT_EQ(config.deadline.target_fps, 30);  // Default
    EXPECT_FLOAT_EQ(config.ml_config.nms_threshold, 0.4f);  // Default
}

TEST_F(ConfigLoadTest, InvalidFile) {
    // Test loading non-existent file
    EXPECT_THROW(load_config("/nonexistent/path/config.yaml"), std::exception);
}

TEST_F(ConfigLoadTest, MalformedYAML) {
    const std::string malformed_content = R"(
input:
  uri: "test.mp4
  width: [invalid
)";

    createTestConfig("malformed.yaml", malformed_content);
    
    // Should throw an exception for malformed YAML
    EXPECT_THROW(load_config((test_dir / "malformed.yaml").string()), std::exception);
}

TEST_F(ConfigLoadTest, CompleteConfig) {
    const std::string complete_config = R"(
input:
  uri: "data/complete_test.mp4"
  width: 1920
  height: 1080
  fps: 60

pipeline:
  budget_ms: 16.67
  target_fps: 60
  queue_capacity: 8
  drop_policy: "drop_newest"

controller:
  switch_up_p95_ms: 14.0
  switch_down_p95_ms: 11.0
  hysteresis_frames: 180

telemetry:
  metrics_port: 8080

ml_engine:
  yolo_model_path: "models/yolo11m.onnx"
  yolo_engine_path: "engines/yolo11m.engine"
  use_tensorrt: true
  use_fp16: false
  detection_threshold: 0.7
  enable_vehicle_analytics: true
  vehicle_classes: [1, 2, 3]

output:
  enable_video_output: true
  output_video_path: "output/complete_test.mp4"
  enable_overlay: true
  overlay_opacity: 0.9
)";

    createTestConfig("complete_config.yaml", complete_config);
    
    AppConfig config = load_config((test_dir / "complete_config.yaml").string());
    
    // Verify all major sections are loaded correctly
    EXPECT_EQ(config.pipeline.uri, "data/complete_test.mp4");
    EXPECT_EQ(config.pipeline.fps, 60);
    EXPECT_DOUBLE_EQ(config.deadline.budget_ms, 16.67);
    EXPECT_EQ(config.deadline.target_fps, 60);
    EXPECT_EQ(config.metrics_port, 8080);
    EXPECT_EQ(config.ml_config.yolo_model_path, "models/yolo11m.onnx");
    EXPECT_FALSE(config.ml_config.use_fp16);
    EXPECT_FLOAT_EQ(config.ml_config.detection_threshold, 0.7f);
    EXPECT_TRUE(config.output_config.enable_video_output);
    EXPECT_FLOAT_EQ(config.output_config.overlay_opacity, 0.9f);
}

// Test AppConfig structure
TEST(AppConfigTest, DefaultConstruction) {
    AppConfig config;
    
    // Should have sensible defaults
    EXPECT_EQ(config.metrics_port, 9090);
    
    // Pipeline defaults
    EXPECT_EQ(config.pipeline.uri, "data/sample_1080p30.mp4");
    EXPECT_EQ(config.pipeline.width, 1920);
    EXPECT_EQ(config.pipeline.height, 1080);
    EXPECT_EQ(config.pipeline.fps, 30);
    
    // Deadline defaults
    EXPECT_EQ(config.deadline.target_fps, 30);
    EXPECT_DOUBLE_EQ(config.deadline.budget_ms, 33.0);
}

TEST(AppConfigTest, StructureCopyability) {
    AppConfig config1;
    config1.metrics_port = 8080;
    config1.pipeline.uri = "test.mp4";
    
    AppConfig config2 = config1;  // Copy constructor
    
    EXPECT_EQ(config2.metrics_port, 8080);
    EXPECT_EQ(config2.pipeline.uri, "test.mp4");
    
    AppConfig config3;
    config3 = config1;  // Assignment operator
    
    EXPECT_EQ(config3.metrics_port, 8080);
    EXPECT_EQ(config3.pipeline.uri, "test.mp4");
}
