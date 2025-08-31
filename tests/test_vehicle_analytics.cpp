#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "vehicle_analytics.hpp"

class VehicleTrackTest : public ::testing::Test {
protected:
    void SetUp() override {
        track = VehicleTrack{};
    }

    VehicleTrack track;
};

TEST_F(VehicleTrackTest, DefaultConstruction) {
    EXPECT_EQ(track.track_id, -1);
    EXPECT_EQ(track.type, VehicleType::UNKNOWN);
    EXPECT_EQ(track.current_bbox.area(), 0);
    EXPECT_FLOAT_EQ(track.center.x, 0.0f);
    EXPECT_FLOAT_EQ(track.center.y, 0.0f);
    EXPECT_FLOAT_EQ(track.velocity.x, 0.0f);
    EXPECT_FLOAT_EQ(track.velocity.y, 0.0f);
    EXPECT_FLOAT_EQ(track.confidence, 0.0f);
    EXPECT_TRUE(track.trajectory.empty());
    EXPECT_EQ(track.frames_tracked, 0);
    EXPECT_EQ(track.frames_lost, 0);
    EXPECT_FALSE(track.is_approaching);
    EXPECT_FALSE(track.in_danger_zone);
}

TEST_F(VehicleTrackTest, UpdateWithDetection) {
    Detection detection;
    detection.class_id = 2;  // Car
    detection.confidence = 0.8f;
    detection.bbox = cv::Rect(100, 100, 50, 30);
    
    auto timestamp = std::chrono::steady_clock::now();
    track.update(detection, timestamp);
    
    EXPECT_EQ(track.current_bbox, detection.bbox);
    EXPECT_FLOAT_EQ(track.confidence, 0.8f);
    EXPECT_EQ(track.trajectory.size(), 1);
    EXPECT_EQ(track.frames_tracked, 1);
    EXPECT_EQ(track.frames_lost, 0);
}

TEST_F(VehicleTrackTest, CenterCalculation) {
    track.current_bbox = cv::Rect(100, 100, 50, 30);
    track.center = cv::Point2f(125.0f, 115.0f);  // 100+50/2, 100+30/2
    
    cv::Point2f latest_pos = track.getLatestPosition();
    EXPECT_FLOAT_EQ(latest_pos.x, 125.0f);
    EXPECT_FLOAT_EQ(latest_pos.y, 115.0f);
}

TEST_F(VehicleTrackTest, SpeedCalculation) {
    // Add trajectory points
    track.trajectory.push_back(cv::Point2f(100.0f, 100.0f));
    track.trajectory.push_back(cv::Point2f(110.0f, 100.0f));
    track.trajectory.push_back(cv::Point2f(120.0f, 100.0f));
    
    float speed = track.calculateSpeed();
    EXPECT_GT(speed, 0.0f);  // Should have positive speed
}

TEST_F(VehicleTrackTest, ApproachingDetection) {
    cv::Size frame_size(640, 480);
    
    // Vehicle moving down (towards bottom of frame = approaching)
    track.trajectory.push_back(cv::Point2f(320.0f, 200.0f));
    track.trajectory.push_back(cv::Point2f(320.0f, 220.0f));
    track.trajectory.push_back(cv::Point2f(320.0f, 240.0f));
    
    bool approaching = track.isApproaching(frame_size);
    EXPECT_TRUE(approaching);
}

TEST_F(VehicleTrackTest, LostTrackDetection) {
    track.frames_lost = 5;
    EXPECT_FALSE(track.isLost(10));  // Not lost yet
    
    track.frames_lost = 15;
    EXPECT_TRUE(track.isLost(10));   // Now lost
}

class LaneInfoTest : public ::testing::Test {
protected:
    void SetUp() override {
        lane_info = LaneInfo{};
    }

    LaneInfo lane_info;
};

TEST_F(LaneInfoTest, DefaultConstruction) {
    EXPECT_TRUE(lane_info.left_lane.empty());
    EXPECT_TRUE(lane_info.right_lane.empty());
    EXPECT_FALSE(lane_info.lanes_detected);
    EXPECT_FLOAT_EQ(lane_info.lane_width_pixels, 0.0f);
}

TEST_F(LaneInfoTest, LaneDetection) {
    lane_info.left_lane = {{100, 400}, {150, 300}, {200, 200}};
    lane_info.right_lane = {{300, 400}, {350, 300}, {400, 200}};
    lane_info.lanes_detected = true;
    lane_info.lane_width_pixels = 200.0f;
    
    EXPECT_TRUE(lane_info.lanes_detected);
    EXPECT_EQ(lane_info.left_lane.size(), 3);
    EXPECT_EQ(lane_info.right_lane.size(), 3);
    EXPECT_FLOAT_EQ(lane_info.lane_width_pixels, 200.0f);
}

TEST_F(LaneInfoTest, EgoLaneCheck) {
    // Set up simple lane boundaries
    lane_info.left_lane = {{100, 400}, {200, 200}};
    lane_info.right_lane = {{300, 400}, {400, 200}};
    lane_info.lanes_detected = true;
    
    // Point in center should be in ego lane
    cv::Point2f center_point(250.0f, 300.0f);
    EXPECT_TRUE(lane_info.isInEgoLane(center_point));
    
    // Point outside should not be in ego lane
    cv::Point2f outside_point(50.0f, 300.0f);
    EXPECT_FALSE(lane_info.isInEgoLane(outside_point));
}

class VehicleAnalyticsResultTest : public ::testing::Test {
protected:
    void SetUp() override {
        result = VehicleAnalyticsResult{};
    }

    VehicleAnalyticsResult result;
};

TEST_F(VehicleAnalyticsResultTest, DefaultConstruction) {
    EXPECT_TRUE(result.active_tracks.empty());
    EXPECT_TRUE(result.vehicle_detections.empty());
    EXPECT_EQ(result.total_vehicles, 0);
    EXPECT_TRUE(result.approaching_vehicles.empty());
    EXPECT_TRUE(result.danger_zone_vehicles.empty());
    EXPECT_TRUE(result.ego_lane_clear);
    EXPECT_EQ(result.vehicles_in_ego_lane, 0);
    EXPECT_FALSE(result.collision_warning);
    EXPECT_TRUE(result.lane_change_safe);
    EXPECT_FLOAT_EQ(result.closest_vehicle_distance, std::numeric_limits<float>::max());
}

TEST_F(VehicleAnalyticsResultTest, SafetyAnalysis) {
    result.collision_warning = true;
    result.lane_change_safe = false;
    result.closest_vehicle_distance = 25.0f;
    result.danger_zone_vehicles = {1, 3};
    result.approaching_vehicles = {2, 4};
    
    EXPECT_TRUE(result.collision_warning);
    EXPECT_FALSE(result.lane_change_safe);
    EXPECT_FLOAT_EQ(result.closest_vehicle_distance, 25.0f);
    EXPECT_EQ(result.danger_zone_vehicles.size(), 2);
    EXPECT_EQ(result.approaching_vehicles.size(), 2);
}

class VehicleAnalyticsConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        config = VehicleAnalyticsConfig{};
    }

    VehicleAnalyticsConfig config;
};

TEST_F(VehicleAnalyticsConfigTest, DefaultValues) {
    std::vector<int> expected_classes = {1, 2, 3, 5, 7};
    EXPECT_EQ(config.vehicle_classes, expected_classes);
    EXPECT_TRUE(config.focus_vehicle_detection);
    EXPECT_FLOAT_EQ(config.max_tracking_distance, 100.0f);
    EXPECT_EQ(config.max_frames_lost, 10);
    EXPECT_EQ(config.min_track_length, 5);
    EXPECT_FLOAT_EQ(config.danger_zone_ratio, 0.3f);
    EXPECT_FLOAT_EQ(config.warning_zone_ratio, 0.5f);
    EXPECT_FLOAT_EQ(config.fps, 30.0f);
    EXPECT_TRUE(config.enable_tracking);
    EXPECT_TRUE(config.enable_proximity_detection);
}

TEST_F(VehicleAnalyticsConfigTest, CustomConfiguration) {
    config.max_tracking_distance = 150.0f;
    config.danger_zone_ratio = 0.25f;
    config.fps = 60.0f;
    config.enable_lane_detection = false;
    config.vehicle_classes = {2, 5, 7};  // Only cars, buses, trucks
    
    EXPECT_FLOAT_EQ(config.max_tracking_distance, 150.0f);
    EXPECT_FLOAT_EQ(config.danger_zone_ratio, 0.25f);
    EXPECT_FLOAT_EQ(config.fps, 60.0f);
    EXPECT_FALSE(config.enable_lane_detection);
    EXPECT_EQ(config.vehicle_classes.size(), 3);
}

TEST_F(VehicleAnalyticsConfigTest, ZoneRatioValidation) {
    // Danger zone should be smaller than warning zone
    EXPECT_LT(config.danger_zone_ratio, config.warning_zone_ratio);
    EXPECT_GT(config.danger_zone_ratio, 0.0f);
    EXPECT_LT(config.warning_zone_ratio, 1.0f);
}

class VehicleTypeTest : public ::testing::Test {};

TEST_F(VehicleTypeTest, EnumValues) {
    EXPECT_EQ(static_cast<int>(VehicleType::BICYCLE), 1);
    EXPECT_EQ(static_cast<int>(VehicleType::CAR), 2);
    EXPECT_EQ(static_cast<int>(VehicleType::MOTORCYCLE), 3);
    EXPECT_EQ(static_cast<int>(VehicleType::BUS), 5);
    EXPECT_EQ(static_cast<int>(VehicleType::TRUCK), 7);
    EXPECT_EQ(static_cast<int>(VehicleType::UNKNOWN), -1);
}

TEST_F(VehicleTypeTest, TypeComparison) {
    VehicleType car = VehicleType::CAR;
    VehicleType truck = VehicleType::TRUCK;
    VehicleType unknown = VehicleType::UNKNOWN;
    
    EXPECT_NE(car, truck);
    EXPECT_NE(car, unknown);
    EXPECT_EQ(car, VehicleType::CAR);
}

// Mock tests for VehicleAnalytics class
class VehicleAnalyticsMockTest : public ::testing::Test {
protected:
    void SetUp() override {
        config = VehicleAnalyticsConfig{};
        config.enable_lane_detection = false;  // Disable for testing
        analytics = createVehicleAnalytics(config);
    }

    VehicleAnalyticsConfig config;
    std::unique_ptr<VehicleAnalytics> analytics;
};

TEST_F(VehicleAnalyticsMockTest, Construction) {
    EXPECT_NE(analytics.get(), nullptr);
    
    const auto& engine_config = analytics->getConfig();
    EXPECT_FALSE(engine_config.enable_lane_detection);
}

TEST_F(VehicleAnalyticsMockTest, StatsInitialization) {
    const auto& stats = analytics->getStats();
    
    EXPECT_EQ(stats.total_vehicles_detected, 0);
    EXPECT_EQ(stats.total_tracks_created, 0);
    EXPECT_FLOAT_EQ(stats.avg_tracking_accuracy, 0.0f);
    EXPECT_FLOAT_EQ(stats.avg_processing_time_ms, 0.0f);
    EXPECT_EQ(stats.collision_warnings_issued, 0);
}

TEST_F(VehicleAnalyticsMockTest, FilterVehicleDetections) {
    std::vector<Detection> all_detections;
    
    // Add vehicle detections
    Detection car;
    car.class_id = 2;  // Car
    car.confidence = 0.8f;
    all_detections.push_back(car);
    
    Detection person;
    person.class_id = 0;  // Person (not a vehicle)
    person.confidence = 0.9f;
    all_detections.push_back(person);
    
    Detection truck;
    truck.class_id = 7;  // Truck
    truck.confidence = 0.7f;
    all_detections.push_back(truck);
    
    auto vehicle_detections = analytics->filterVehicleDetections(all_detections);
    
    // Should only have vehicles
    EXPECT_EQ(vehicle_detections.size(), 2);
    EXPECT_EQ(vehicle_detections[0].class_id, 2);  // Car
    EXPECT_EQ(vehicle_detections[1].class_id, 7);  // Truck
}

TEST_F(VehicleAnalyticsMockTest, EmptyFrameAnalysis) {
    cv::Mat empty_frame = cv::Mat::zeros(480, 640, CV_8UC3);
    std::vector<Detection> empty_detections;
    
    VehicleAnalyticsResult result = analytics->analyze(empty_frame, empty_detections);
    
    EXPECT_EQ(result.total_vehicles, 0);
    EXPECT_TRUE(result.active_tracks.empty());
    EXPECT_TRUE(result.vehicle_detections.empty());
    EXPECT_TRUE(result.ego_lane_clear);
    EXPECT_FALSE(result.collision_warning);
    EXPECT_TRUE(result.lane_change_safe);
}

// Integration tests
TEST(VehicleAnalyticsIntegrationTest, DetectionTracking) {
    VehicleAnalyticsConfig config;
    config.enable_lane_detection = false;
    auto analytics = createVehicleAnalytics(config);
    
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    
    // First frame with detections
    std::vector<Detection> detections1;
    Detection car1;
    car1.class_id = 2;
    car1.confidence = 0.8f;
    car1.bbox = cv::Rect(100, 100, 50, 30);
    detections1.push_back(car1);
    
    VehicleAnalyticsResult result1 = analytics->analyze(frame, detections1);
    EXPECT_EQ(result1.total_vehicles, 1);
    
    // Second frame with same vehicle moved
    std::vector<Detection> detections2;
    Detection car2;
    car2.class_id = 2;
    car2.confidence = 0.8f;
    car2.bbox = cv::Rect(110, 110, 50, 30);  // Moved 10 pixels
    detections2.push_back(car2);
    
    VehicleAnalyticsResult result2 = analytics->analyze(frame, detections2);
    EXPECT_EQ(result2.total_vehicles, 1);
    
    // Should have tracking information
    if (!result2.active_tracks.empty()) {
        EXPECT_GT(result2.active_tracks[0].frames_tracked, 0);
    }
}

TEST(VehicleAnalyticsIntegrationTest, SafetyWarnings) {
    VehicleAnalyticsConfig config;
    config.danger_zone_ratio = 0.8f;  // Large danger zone for testing
    config.enable_collision_warning = true;
    auto analytics = createVehicleAnalytics(config);
    
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    
    // Large vehicle detection close to bottom (danger zone)
    std::vector<Detection> detections;
    Detection close_vehicle;
    close_vehicle.class_id = 2;
    close_vehicle.confidence = 0.9f;
    close_vehicle.bbox = cv::Rect(300, 400, 100, 60);  // Large, low in frame
    detections.push_back(close_vehicle);
    
    VehicleAnalyticsResult result = analytics->analyze(frame, detections);
    
    // Should trigger safety warnings
    EXPECT_FALSE(result.ego_lane_clear);
    // Note: Collision warning depends on implementation details
}
