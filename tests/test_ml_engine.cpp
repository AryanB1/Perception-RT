#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "ml_engine.hpp"

class DetectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        detection = Detection{};
    }

    Detection detection;
};

TEST_F(DetectionTest, DefaultConstruction) {
    EXPECT_EQ(detection.class_id, -1);
    EXPECT_FLOAT_EQ(detection.confidence, 0.0f);
    EXPECT_EQ(detection.bbox.x, 0);
    EXPECT_EQ(detection.bbox.y, 0);
    EXPECT_EQ(detection.bbox.width, 0);
    EXPECT_EQ(detection.bbox.height, 0);
    EXPECT_TRUE(detection.label.empty());
}

TEST_F(DetectionTest, CenterCalculation) {
    detection.bbox = cv::Rect(10, 20, 100, 200);
    
    cv::Point2f center = detection.center();
    EXPECT_FLOAT_EQ(center.x, 60.0f);  // 10 + 100/2
    EXPECT_FLOAT_EQ(center.y, 120.0f); // 20 + 200/2
}

TEST_F(DetectionTest, ValidDetection) {
    detection.class_id = 1;
    detection.confidence = 0.85f;
    detection.bbox = cv::Rect(50, 50, 200, 150);
    detection.label = "car";
    
    EXPECT_EQ(detection.class_id, 1);
    EXPECT_FLOAT_EQ(detection.confidence, 0.85f);
    EXPECT_EQ(detection.bbox.width, 200);
    EXPECT_EQ(detection.bbox.height, 150);
    EXPECT_EQ(detection.label, "car");
}

class OpticalFlowResultTest : public ::testing::Test {
protected:
    void SetUp() override {
        flow_result = OpticalFlowResult{};
    }

    OpticalFlowResult flow_result;
};

TEST_F(OpticalFlowResultTest, DefaultConstruction) {
    EXPECT_TRUE(flow_result.points.empty());
    EXPECT_TRUE(flow_result.flow_vectors.empty());
    EXPECT_FLOAT_EQ(flow_result.magnitude_mean, 0.0f);
    EXPECT_FLOAT_EQ(flow_result.magnitude_max, 0.0f);
    EXPECT_EQ(flow_result.moving_points, 0);
}

TEST_F(OpticalFlowResultTest, FlowData) {
    flow_result.points = {{10.0f, 20.0f}, {30.0f, 40.0f}};
    flow_result.flow_vectors = {{1.0f, 2.0f}, {-1.0f, -2.0f}};
    flow_result.magnitude_mean = 2.5f;
    flow_result.magnitude_max = 5.0f;
    flow_result.moving_points = 2;
    
    EXPECT_EQ(flow_result.points.size(), 2);
    EXPECT_EQ(flow_result.flow_vectors.size(), 2);
    EXPECT_FLOAT_EQ(flow_result.magnitude_mean, 2.5f);
    EXPECT_FLOAT_EQ(flow_result.magnitude_max, 5.0f);
    EXPECT_EQ(flow_result.moving_points, 2);
}

class SegmentationResultTest : public ::testing::Test {
protected:
    void SetUp() override {
        seg_result = SegmentationResult{};
    }

    SegmentationResult seg_result;
};

TEST_F(SegmentationResultTest, DefaultConstruction) {
    EXPECT_TRUE(seg_result.class_mask.empty());
    EXPECT_TRUE(seg_result.confidence_mask.empty());
    EXPECT_TRUE(seg_result.detected_classes.empty());
    EXPECT_TRUE(seg_result.class_areas.empty());
}

TEST_F(SegmentationResultTest, WithMasks) {
    // Create dummy masks
    seg_result.class_mask = cv::Mat::ones(100, 100, CV_8UC1);
    seg_result.confidence_mask = cv::Mat::ones(100, 100, CV_32FC1) * 0.8f;
    seg_result.detected_classes = {1, 2, 3};
    seg_result.class_areas[1] = 0.3f;
    seg_result.class_areas[2] = 0.5f;
    
    EXPECT_FALSE(seg_result.class_mask.empty());
    EXPECT_FALSE(seg_result.confidence_mask.empty());
    EXPECT_EQ(seg_result.detected_classes.size(), 3);
    EXPECT_EQ(seg_result.class_areas.size(), 2);
    EXPECT_FLOAT_EQ(seg_result.class_areas[1], 0.3f);
}

class MLResultTest : public ::testing::Test {
protected:
    void SetUp() override {
        ml_result = MLResult{};
    }

    MLResult ml_result;
};

TEST_F(MLResultTest, DefaultConstruction) {
    EXPECT_TRUE(ml_result.detections.empty());
    EXPECT_EQ(ml_result.total_objects, 0);
    EXPECT_FLOAT_EQ(ml_result.max_confidence, 0.0f);
    EXPECT_FALSE(ml_result.significant_motion);
    EXPECT_FALSE(ml_result.motion_detected);
    EXPECT_FLOAT_EQ(ml_result.motion_intensity, 0.0f);
    EXPECT_EQ(ml_result.motion_pixels, 0);
    EXPECT_FALSE(ml_result.vehicle_analytics_enabled);
    EXPECT_EQ(ml_result.vehicle_analytics.get(), nullptr);
}

TEST_F(MLResultTest, WithDetections) {
    Detection det1;
    det1.class_id = 1;
    det1.confidence = 0.9f;
    det1.bbox = cv::Rect(10, 10, 50, 50);
    
    Detection det2;
    det2.class_id = 2;
    det2.confidence = 0.7f;
    det2.bbox = cv::Rect(100, 100, 80, 60);
    
    ml_result.detections = {det1, det2};
    ml_result.total_objects = 2;
    ml_result.max_confidence = 0.9f;
    
    EXPECT_EQ(ml_result.detections.size(), 2);
    EXPECT_EQ(ml_result.total_objects, 2);
    EXPECT_FLOAT_EQ(ml_result.max_confidence, 0.9f);
    EXPECT_EQ(ml_result.detections[0].class_id, 1);
    EXPECT_EQ(ml_result.detections[1].class_id, 2);
}

TEST_F(MLResultTest, PerformanceMetrics) {
    ml_result.inference_time_ms = 2.5f;
    ml_result.preprocessing_time_ms = 0.8f;
    ml_result.postprocessing_time_ms = 0.3f;
    
    EXPECT_FLOAT_EQ(ml_result.inference_time_ms, 2.5f);
    EXPECT_FLOAT_EQ(ml_result.preprocessing_time_ms, 0.8f);
    EXPECT_FLOAT_EQ(ml_result.postprocessing_time_ms, 0.3f);
    
    // Calculate total processing time
    float total_time = ml_result.preprocessing_time_ms + 
                      ml_result.inference_time_ms + 
                      ml_result.postprocessing_time_ms;
    EXPECT_FLOAT_EQ(total_time, 3.6f);
}

class MLConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        config = MLConfig{};
    }

    MLConfig config;
};

TEST_F(MLConfigTest, DefaultValues) {
    EXPECT_EQ(config.yolo_model_path, "model/yolo/yolo11s.onnx");
    EXPECT_EQ(config.yolo_engine_path, "model/yolo/yolo11s.engine");
    EXPECT_TRUE(config.use_tensorrt);
    EXPECT_TRUE(config.use_fp16);
    EXPECT_EQ(config.max_batch_size, 1);
    EXPECT_FLOAT_EQ(config.detection_threshold, 0.5f);
    EXPECT_FLOAT_EQ(config.nms_threshold, 0.4f);
    EXPECT_EQ(config.max_detections, 100);
    EXPECT_EQ(config.max_corners, 100);
    EXPECT_TRUE(config.enable_detection);
    EXPECT_TRUE(config.enable_optical_flow);
    EXPECT_FALSE(config.enable_segmentation);
    EXPECT_FALSE(config.enable_vehicle_analytics);
}

TEST_F(MLConfigTest, CustomConfiguration) {
    config.detection_threshold = 0.7f;
    config.nms_threshold = 0.3f;
    config.enable_segmentation = true;
    config.enable_vehicle_analytics = true;
    config.input_size = cv::Size(416, 416);
    
    EXPECT_FLOAT_EQ(config.detection_threshold, 0.7f);
    EXPECT_FLOAT_EQ(config.nms_threshold, 0.3f);
    EXPECT_TRUE(config.enable_segmentation);
    EXPECT_TRUE(config.enable_vehicle_analytics);
    EXPECT_EQ(config.input_size.width, 416);
    EXPECT_EQ(config.input_size.height, 416);
}

TEST_F(MLConfigTest, VehicleClassConfiguration) {
    std::vector<int> expected_vehicle_classes = {1, 2, 3, 5, 7};
    EXPECT_EQ(config.vehicle_classes, expected_vehicle_classes);
    
    config.vehicle_classes = {1, 2};
    EXPECT_EQ(config.vehicle_classes.size(), 2);
    EXPECT_EQ(config.vehicle_classes[0], 1);
    EXPECT_EQ(config.vehicle_classes[1], 2);
}

class PerformanceStatsTest : public ::testing::Test {
protected:
    void SetUp() override {
        stats = MLEngine::PerformanceStats{};
    }

    MLEngine::PerformanceStats stats;
};

TEST_F(PerformanceStatsTest, DefaultValues) {
    EXPECT_FLOAT_EQ(stats.avg_inference_time, 0.0f);
    EXPECT_FLOAT_EQ(stats.avg_preprocessing_time, 0.0f);
    EXPECT_FLOAT_EQ(stats.avg_postprocessing_time, 0.0f);
    EXPECT_EQ(stats.frames_processed, 0);
}

TEST_F(PerformanceStatsTest, UpdateStats) {
    stats.update(2.0f, 0.5f, 0.3f);
    
    EXPECT_FLOAT_EQ(stats.avg_inference_time, 2.0f);
    EXPECT_FLOAT_EQ(stats.avg_preprocessing_time, 0.5f);
    EXPECT_FLOAT_EQ(stats.avg_postprocessing_time, 0.3f);
    EXPECT_EQ(stats.frames_processed, 1);
}

TEST_F(PerformanceStatsTest, MultipleUpdates) {
    stats.update(2.0f, 0.5f, 0.3f);  // First frame
    stats.update(3.0f, 0.7f, 0.4f);  // Second frame
    
    // Should calculate running average
    EXPECT_FLOAT_EQ(stats.avg_inference_time, 2.5f);     // (2.0 + 3.0) / 2
    EXPECT_FLOAT_EQ(stats.avg_preprocessing_time, 0.6f); // (0.5 + 0.7) / 2
    EXPECT_FLOAT_EQ(stats.avg_postprocessing_time, 0.35f); // (0.3 + 0.4) / 2
    EXPECT_EQ(stats.frames_processed, 2);
}

TEST_F(PerformanceStatsTest, ResetStats) {
    stats.update(2.0f, 0.5f, 0.3f);
    EXPECT_GT(stats.frames_processed, 0);
    
    stats.reset();
    
    EXPECT_FLOAT_EQ(stats.avg_inference_time, 0.0f);
    EXPECT_FLOAT_EQ(stats.avg_preprocessing_time, 0.0f);
    EXPECT_FLOAT_EQ(stats.avg_postprocessing_time, 0.0f);
    EXPECT_EQ(stats.frames_processed, 0);
}

// Mock tests for MLEngine (since it requires GPU/TensorRT)
class MLEngineMockTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a minimal config for testing
        config = MLConfig{};
        config.use_tensorrt = false;  // Disable TensorRT for testing
        config.enable_segmentation = false;  // Disable heavy operations
        config.enable_vehicle_analytics = false;
    }

    MLConfig config;
};

TEST_F(MLEngineMockTest, ConfigurationTest) {
    auto engine = createMLEngine(config);
    EXPECT_NE(engine.get(), nullptr);
    
    const auto& engine_config = engine->getConfig();
    EXPECT_FALSE(engine_config.use_tensorrt);
    EXPECT_FALSE(engine_config.enable_segmentation);
    EXPECT_FALSE(engine_config.enable_vehicle_analytics);
}

TEST_F(MLEngineMockTest, StatsInitialization) {
    auto engine = createMLEngine(config);
    const auto& stats = engine->getStats();
    
    EXPECT_FLOAT_EQ(stats.avg_inference_time, 0.0f);
    EXPECT_FLOAT_EQ(stats.avg_preprocessing_time, 0.0f);
    EXPECT_FLOAT_EQ(stats.avg_postprocessing_time, 0.0f);
    EXPECT_EQ(stats.frames_processed, 0);
}
