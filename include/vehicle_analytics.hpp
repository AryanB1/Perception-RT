#pragma once

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "ml_engine.hpp"

// Vehicle-specific detection classes (COCO dataset)
enum class VehicleType { BICYCLE = 1, CAR = 2, MOTORCYCLE = 3, BUS = 5, TRUCK = 7, UNKNOWN = -1 };

// Vehicle tracking information
struct VehicleTrack {
  int track_id{-1};
  VehicleType type{VehicleType::UNKNOWN};
  cv::Rect current_bbox;
  cv::Point2f center;
  cv::Point2f velocity{0.0f, 0.0f};  // pixels per frame
  float confidence{0.0f};

  // Tracking history
  std::vector<cv::Point2f> trajectory;  // Position history
  std::vector<cv::Rect> bbox_history;   // Bounding box history
  std::vector<float> confidence_history;

  // Analytics data
  float speed_estimate{0.0f};     // estimated speed in pixels/frame
  float distance_estimate{0.0f};  // relative distance (larger bbox = closer)
  bool is_approaching{false};
  bool is_departing{false};
  bool in_danger_zone{false};  // too close for comfort

  // Temporal information
  std::chrono::steady_clock::time_point first_seen;
  std::chrono::steady_clock::time_point last_updated;
  int frames_tracked{0};
  int frames_lost{0};  // consecutive frames without detection

  // Calculate current speed based on trajectory
  float calculateSpeed() const;

  // Check if vehicle is moving towards ego vehicle
  bool isApproaching(const cv::Size& frame_size) const;

  // Get latest position
  cv::Point2f getLatestPosition() const { return center; }

  // Update track with new detection
  void update(const Detection& detection, std::chrono::steady_clock::time_point timestamp);

  // Check if track should be considered lost
  bool isLost(int max_frames_lost = 10) const { return frames_lost > max_frames_lost; }
};

// Lane detection and analysis
struct LaneInfo {
  std::vector<cv::Point> left_lane;
  std::vector<cv::Point> right_lane;
  cv::Point vanishing_point;
  bool lanes_detected{false};
  float lane_width_pixels{0.0f};

  // Check if a point is in the ego lane
  bool isInEgoLane(const cv::Point2f& point) const;

  // Get relative position in lane (-1 = left, 0 = center, 1 = right)
  float getLanePosition(const cv::Point2f& point) const;
};

// Comprehensive vehicle analytics result
struct VehicleAnalyticsResult {
  // Vehicle detections and tracking
  std::vector<VehicleTrack> active_tracks;
  std::vector<Detection> vehicle_detections;  // All vehicle detections this frame
  int total_vehicles{0};

  // Safety analytics
  std::vector<int> approaching_vehicles;  // track IDs of approaching vehicles
  std::vector<int> danger_zone_vehicles;  // track IDs of vehicles too close
  std::vector<int> overtaking_vehicles;   // track IDs of vehicles overtaking

  // Lane analysis
  LaneInfo lane_info;
  bool ego_lane_clear{true};
  int vehicles_in_ego_lane{0};

  // Traffic flow analysis
  float traffic_density{0.0f};                     // vehicles per unit area
  cv::Point2f traffic_flow_direction{0.0f, 0.0f};  // average motion vector

  // Proximity warnings
  bool collision_warning{false};
  bool lane_change_safe{true};
  float closest_vehicle_distance{std::numeric_limits<float>::max()};

  // Performance metrics
  float analytics_time_ms{0.0f};
  int tracks_updated{0};
  int new_tracks{0};
  int lost_tracks{0};
};

// Configuration for vehicle analytics
struct VehicleAnalyticsConfig {
  // Vehicle type filtering
  std::vector<int> vehicle_classes{1, 2, 3, 5, 7};  // COCO class IDs for vehicles
  bool focus_vehicle_detection{true};

  // Tracking parameters
  float max_tracking_distance{100.0f};  // max pixels between detections to link
  int max_frames_lost{10};              // max frames before dropping a track
  int min_track_length{5};              // minimum frames to consider a valid track

  // Safety zone parameters
  float danger_zone_ratio{0.3f};     // fraction of frame height for danger zone
  float warning_zone_ratio{0.5f};    // fraction of frame height for warning zone
  float collision_threshold{50.0f};  // pixels - very close proximity

  // Speed estimation parameters
  float pixels_per_meter{20.0f};  // calibration for speed estimation
  float fps{30.0f};               // frames per second for speed calculation

  // Lane detection parameters
  bool enable_lane_detection{true};
  float lane_detection_roi_top{0.4f};  // start ROI at 40% of frame height

  // Analytics features
  bool enable_tracking{true};
  bool enable_proximity_detection{true};
  bool enable_speed_estimation{true};
  bool enable_collision_warning{true};
};

// Main vehicle analytics engine
class VehicleAnalytics {
public:
  explicit VehicleAnalytics(const VehicleAnalyticsConfig& config = VehicleAnalyticsConfig{});
  ~VehicleAnalytics() = default;

  // Main processing function
  VehicleAnalyticsResult analyze(const cv::Mat& frame,
                                 const std::vector<Detection>& all_detections);

  // Component functions
  std::vector<Detection> filterVehicleDetections(const std::vector<Detection>& detections);
  void updateTracks(const std::vector<Detection>& vehicle_detections,
                    std::chrono::steady_clock::time_point timestamp);
  LaneInfo detectLanes(const cv::Mat& frame);
  void analyzeSafety(VehicleAnalyticsResult& result, const cv::Size& frame_size);
  void analyzeTrafficFlow(VehicleAnalyticsResult& result);

  // Visualization
  cv::Mat visualizeAnalytics(const cv::Mat& frame, const VehicleAnalyticsResult& result);

  // Configuration
  void setConfig(const VehicleAnalyticsConfig& config) { config_ = config; }
  const VehicleAnalyticsConfig& getConfig() const { return config_; }

  // Statistics
  struct AnalyticsStats {
    int total_vehicles_detected{0};
    int total_tracks_created{0};
    float avg_tracking_accuracy{0.0f};
    float avg_processing_time_ms{0.0f};
    int collision_warnings_issued{0};

    void reset();
  };

  const AnalyticsStats& getStats() const { return stats_; }
  void resetStats() { stats_.reset(); }

private:
  VehicleAnalyticsConfig config_;
  AnalyticsStats stats_;

  // Tracking state
  std::unordered_map<int, VehicleTrack> active_tracks_;
  int next_track_id_{1};

  // Lane detection state
  cv::Ptr<cv::LineSegmentDetector> line_detector_;

  // Timing
  std::chrono::steady_clock::time_point timer_start_;

  // Helper methods
  VehicleType classIdToVehicleType(int class_id);
  std::string vehicleTypeToString(VehicleType type);

  // Tracking helpers
  float calculateTrackingDistance(const cv::Rect& bbox1, const cv::Rect& bbox2);
  void associateDetectionsToTracks(const std::vector<Detection>& detections,
                                   std::chrono::steady_clock::time_point timestamp);
  void pruneExpiredTracks();

  // Safety analysis helpers
  bool isInDangerZone(const cv::Rect& bbox, const cv::Size& frame_size);
  bool isInWarningZone(const cv::Rect& bbox, const cv::Size& frame_size);
  float estimateDistance(const cv::Rect& bbox, const cv::Size& frame_size);

  // Lane detection helpers
  std::vector<cv::Vec4f> detectLaneLines(const cv::Mat& frame);
  cv::Point findVanishingPoint(const std::vector<cv::Vec4f>& lines);

  // Utility
  void startTimer();
  float getElapsedMs();
};

// Factory function
std::unique_ptr<VehicleAnalytics> createVehicleAnalytics(
    const VehicleAnalyticsConfig& config = VehicleAnalyticsConfig{});

// Utility functions for vehicle analytics visualization
namespace VehicleViz {
cv::Mat drawVehicleTracks(const cv::Mat& frame, const std::vector<VehicleTrack>& tracks);
cv::Mat drawSafetyZones(const cv::Mat& frame, const VehicleAnalyticsConfig& config);
cv::Mat drawLaneDetection(const cv::Mat& frame, const LaneInfo& lane_info);
cv::Mat drawProximityWarnings(const cv::Mat& frame, const VehicleAnalyticsResult& result);
cv::Mat drawAnalyticsOverlay(const cv::Mat& frame, const VehicleAnalyticsResult& result);
}  // namespace VehicleViz
