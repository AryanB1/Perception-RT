#include "vehicle_analytics.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace std::chrono;

// VehicleTrack Implementation
float VehicleTrack::calculateSpeed() const {
  if (trajectory.size() < 2) return 0.0f;

  // Calculate speed based on last few positions
  int samples = std::min(5, static_cast<int>(trajectory.size()));
  float total_distance = 0.0f;

  for (int i = static_cast<int>(trajectory.size()) - samples;
       i < static_cast<int>(trajectory.size()) - 1; ++i) {
    cv::Point2f diff = trajectory[static_cast<size_t>(i) + 1] - trajectory[static_cast<size_t>(i)];
    total_distance += std::sqrt(diff.x * diff.x + diff.y * diff.y);
  }

  return total_distance / static_cast<float>(samples - 1);
}

bool VehicleTrack::isApproaching(const cv::Size& /*frame_size*/) const {
  if (trajectory.size() < 3) return false;

  // Check if vehicle is moving towards the bottom of the frame (closer to ego vehicle)
  cv::Point2f recent_motion = trajectory.back() - trajectory[trajectory.size() - 3];
  return recent_motion.y > 0;  // Moving down = approaching in typical dashcam view
}

void VehicleTrack::update(const Detection& detection, steady_clock::time_point timestamp) {
  current_bbox = detection.bbox;
  center = detection.center();
  confidence = detection.confidence;
  last_updated = timestamp;
  frames_tracked++;
  frames_lost = 0;

  // Update trajectory
  trajectory.push_back(center);
  bbox_history.push_back(current_bbox);
  confidence_history.push_back(confidence);

  // Limit history size
  const size_t max_history = 50;
  if (trajectory.size() > max_history) {
    trajectory.erase(trajectory.begin());
    bbox_history.erase(bbox_history.begin());
    confidence_history.erase(confidence_history.begin());
  }

  // Update velocity
  if (trajectory.size() >= 2) {
    cv::Point2f prev_center = trajectory[trajectory.size() - 2];
    velocity = center - prev_center;
  }

  // Update analytics
  speed_estimate = calculateSpeed();
  is_approaching = isApproaching(cv::Size(1000, 1000));  // dummy size, will be updated

  // Update distance estimate (larger bbox = closer)
  distance_estimate = 1.0f / (static_cast<float>(current_bbox.area()) / 10000.0f + 0.01f);
}

// LaneInfo Implementation
bool LaneInfo::isInEgoLane(const cv::Point2f& /*point*/) const {
  if (!lanes_detected || left_lane.empty() || right_lane.empty()) return false;

  // Simple check: is point between left and right lane at the point's y-coordinate
  // This is a simplified implementation
  return true;  // Placeholder - would need more sophisticated implementation
}

float LaneInfo::getLanePosition(const cv::Point2f& /*point*/) const {
  if (!lanes_detected) return 0.0f;

  // Return relative position in lane (-1 = left, 0 = center, 1 = right)
  // Placeholder implementation
  return 0.0f;
}

// VehicleAnalytics Implementation
VehicleAnalytics::VehicleAnalytics(const VehicleAnalyticsConfig& config) : config_(config) {
  line_detector_ = cv::createLineSegmentDetector();
  spdlog::info("Vehicle Analytics initialized");
  spdlog::info("  - Vehicle classes: {}", config_.vehicle_classes.size());
  spdlog::info("  - Tracking enabled: {}", config_.enable_tracking);
  spdlog::info("  - Proximity detection: {}", config_.enable_proximity_detection);
  spdlog::info("  - Collision warning: {}", config_.enable_collision_warning);
}

VehicleAnalyticsResult VehicleAnalytics::analyze(const cv::Mat& frame,
                                                 const std::vector<Detection>& all_detections) {
  startTimer();
  VehicleAnalyticsResult result;
  auto timestamp = steady_clock::now();

  // Filter vehicle detections
  result.vehicle_detections = filterVehicleDetections(all_detections);
  result.total_vehicles = static_cast<int>(result.vehicle_detections.size());

  if (config_.enable_tracking) {
    // Update vehicle tracking
    updateTracks(result.vehicle_detections, timestamp);

    // Copy active tracks to result
    result.active_tracks.clear();
    for (const auto& [track_id, track] : active_tracks_) {
      if (!track.isLost()) {
        result.active_tracks.push_back(track);
      }
    }

    result.tracks_updated = static_cast<int>(result.active_tracks.size());
  }

  // Lane detection
  if (config_.enable_lane_detection) {
    result.lane_info = detectLanes(frame);
  }

  // Safety analysis
  if (config_.enable_proximity_detection || config_.enable_collision_warning) {
    analyzeSafety(result, frame.size());
  }

  // Traffic flow analysis
  analyzeTrafficFlow(result);

  result.analytics_time_ms = getElapsedMs();

  // Update statistics
  stats_.total_vehicles_detected += result.total_vehicles;
  stats_.avg_processing_time_ms = (stats_.avg_processing_time_ms + result.analytics_time_ms) / 2.0f;
  if (result.collision_warning) {
    stats_.collision_warnings_issued++;
  }

  return result;
}

std::vector<Detection> VehicleAnalytics::filterVehicleDetections(
    const std::vector<Detection>& detections) {
  std::vector<Detection> vehicle_detections;

  for (const auto& detection : detections) {
    // Check if detection is a vehicle class
    auto it = std::find(config_.vehicle_classes.begin(), config_.vehicle_classes.end(),
                        detection.class_id);
    if (it != config_.vehicle_classes.end()) {
      vehicle_detections.push_back(detection);
    }
  }

  return vehicle_detections;
}

void VehicleAnalytics::updateTracks(const std::vector<Detection>& vehicle_detections,
                                    steady_clock::time_point timestamp) {
  // Association step: match detections to existing tracks
  associateDetectionsToTracks(vehicle_detections, timestamp);

  // Cleanup step: remove expired tracks
  pruneExpiredTracks();
}

void VehicleAnalytics::associateDetectionsToTracks(const std::vector<Detection>& detections,
                                                   steady_clock::time_point timestamp) {
  std::vector<bool> detection_matched(detections.size(), false);
  std::vector<bool> track_updated(active_tracks_.size(), false);

  // For each existing track, find the best matching detection
  for (auto& [track_id, track] : active_tracks_) {
    float best_distance = std::numeric_limits<float>::max();
    int best_detection_idx = -1;

    for (size_t i = 0; i < detections.size(); ++i) {
      if (detection_matched[i]) continue;

      // Check if detection is same vehicle type
      VehicleType det_type = classIdToVehicleType(detections[i].class_id);
      if (track.type != VehicleType::UNKNOWN && track.type != det_type) continue;

      float distance = calculateTrackingDistance(track.current_bbox, detections[i].bbox);
      if (distance < best_distance && distance < config_.max_tracking_distance) {
        best_distance = distance;
        best_detection_idx = static_cast<int>(i);
      }
    }

    if (best_detection_idx >= 0) {
      // Update track with matched detection
      track.update(detections[best_detection_idx], timestamp);
      if (track.type == VehicleType::UNKNOWN) {
        track.type = classIdToVehicleType(detections[best_detection_idx].class_id);
      }
      detection_matched[best_detection_idx] = true;
    } else {
      // Track not matched - increment lost frames
      track.frames_lost++;
    }
  }

  // Create new tracks for unmatched detections
  for (size_t i = 0; i < detections.size(); ++i) {
    if (!detection_matched[i]) {
      VehicleTrack new_track;
      new_track.track_id = next_track_id_++;
      new_track.type = classIdToVehicleType(detections[i].class_id);
      new_track.first_seen = timestamp;
      new_track.update(detections[i], timestamp);

      active_tracks_[new_track.track_id] = new_track;
      stats_.total_tracks_created++;
    }
  }
}

void VehicleAnalytics::pruneExpiredTracks() {
  auto it = active_tracks_.begin();
  while (it != active_tracks_.end()) {
    if (it->second.isLost(config_.max_frames_lost)) {
      it = active_tracks_.erase(it);
    } else {
      ++it;
    }
  }
}

LaneInfo VehicleAnalytics::detectLanes(const cv::Mat& frame) {
  LaneInfo lane_info;

  if (!config_.enable_lane_detection) {
    return lane_info;
  }

  // Simple lane detection using edge detection and line detection
  cv::Mat gray, edges;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  // Focus on lower part of the image where lanes are typically visible
  int roi_top = static_cast<int>(static_cast<float>(frame.rows) * config_.lane_detection_roi_top);
  cv::Rect roi(0, roi_top, frame.cols, frame.rows - roi_top);
  cv::Mat roi_gray = gray(roi);

  // Edge detection
  cv::Canny(roi_gray, edges, 50, 150);

  // Detect lines
  std::vector<cv::Vec4f> lines = detectLaneLines(edges);

  // Process lines to extract lane boundaries
  // This is a simplified implementation - real lane detection would be more sophisticated
  if (lines.size() >= 2) {
    lane_info.lanes_detected = true;
    // Placeholder: would implement proper lane line clustering and fitting here
  }

  return lane_info;
}

std::vector<cv::Vec4f> VehicleAnalytics::detectLaneLines(const cv::Mat& edges) {
  std::vector<cv::Vec4f> lines;

  if (line_detector_) {
    line_detector_->detect(edges, lines);
  }

  // Filter lines by angle (lane lines should be roughly vertical in image)
  std::vector<cv::Vec4f> filtered_lines;
  for (const auto& line : lines) {
    float dx = line[2] - line[0];
    float dy = line[3] - line[1];
    float angle = std::atan2(dy, dx) * 180.0f / static_cast<float>(M_PI);

    // Keep lines that are roughly vertical (lanes) or diagonal
    if (std::abs(angle) > 30 && std::abs(angle) < 150) {
      filtered_lines.push_back(line);
    }
  }

  return filtered_lines;
}

void VehicleAnalytics::analyzeSafety(VehicleAnalyticsResult& result, const cv::Size& frame_size) {
  result.closest_vehicle_distance = std::numeric_limits<float>::max();
  result.collision_warning = false;
  result.lane_change_safe = true;

  for (const auto& track : result.active_tracks) {
    // Check danger zones
    if (isInDangerZone(track.current_bbox, frame_size)) {
      result.danger_zone_vehicles.push_back(track.track_id);
      result.lane_change_safe = false;

      float distance = estimateDistance(track.current_bbox, frame_size);
      if (distance < result.closest_vehicle_distance) {
        result.closest_vehicle_distance = distance;
      }

      if (distance < config_.collision_threshold) {
        result.collision_warning = true;
      }
    }

    // Check for approaching vehicles
    if (track.is_approaching && track.speed_estimate > 2.0f) {
      result.approaching_vehicles.push_back(track.track_id);
    }

    // Check for overtaking (vehicles moving faster than ego vehicle in adjacent lanes)
    if (track.velocity.x > 5.0f && !isInDangerZone(track.current_bbox, frame_size)) {
      result.overtaking_vehicles.push_back(track.track_id);
    }
  }

  // Determine if ego lane is clear
  result.ego_lane_clear = result.danger_zone_vehicles.empty();
  result.vehicles_in_ego_lane = static_cast<int>(result.danger_zone_vehicles.size());
}

void VehicleAnalytics::analyzeTrafficFlow(VehicleAnalyticsResult& result) {
  if (result.active_tracks.empty()) {
    result.traffic_density = 0.0f;
    result.traffic_flow_direction = cv::Point2f(0.0f, 0.0f);
    return;
  }

  // Calculate traffic density (vehicles per unit area)
  // Simplified: just count vehicles per frame area
  result.traffic_density = static_cast<float>(result.active_tracks.size()) / 100.0f;  // normalized

  // Calculate average flow direction
  cv::Point2f total_velocity(0.0f, 0.0f);
  int moving_vehicles = 0;

  for (const auto& track : result.active_tracks) {
    if (track.speed_estimate > 1.0f) {  // Only consider moving vehicles
      total_velocity += track.velocity;
      moving_vehicles++;
    }
  }

  if (moving_vehicles > 0) {
    result.traffic_flow_direction = total_velocity / static_cast<float>(moving_vehicles);
  }
}

cv::Mat VehicleAnalytics::visualizeAnalytics(const cv::Mat& frame,
                                             const VehicleAnalyticsResult& result) {
  cv::Mat output = frame.clone();

  // Draw vehicle tracks
  output = VehicleViz::drawVehicleTracks(output, result.active_tracks);

  // Draw safety zones
  output = VehicleViz::drawSafetyZones(output, config_);

  // Draw lane detection
  output = VehicleViz::drawLaneDetection(output, result.lane_info);

  // Draw proximity warnings
  output = VehicleViz::drawProximityWarnings(output, result);

  // Draw analytics overlay
  output = VehicleViz::drawAnalyticsOverlay(output, result);

  return output;
}

// Helper methods
VehicleType VehicleAnalytics::classIdToVehicleType(int class_id) {
  switch (class_id) {
    case 1:
      return VehicleType::BICYCLE;
    case 2:
      return VehicleType::CAR;
    case 3:
      return VehicleType::MOTORCYCLE;
    case 5:
      return VehicleType::BUS;
    case 7:
      return VehicleType::TRUCK;
    default:
      return VehicleType::UNKNOWN;
  }
}

std::string VehicleAnalytics::vehicleTypeToString(VehicleType type) {
  switch (type) {
    case VehicleType::BICYCLE:
      return "Bicycle";
    case VehicleType::CAR:
      return "Car";
    case VehicleType::MOTORCYCLE:
      return "Motorcycle";
    case VehicleType::BUS:
      return "Bus";
    case VehicleType::TRUCK:
      return "Truck";
    default:
      return "Unknown";
  }
}

float VehicleAnalytics::calculateTrackingDistance(const cv::Rect& bbox1, const cv::Rect& bbox2) {
  cv::Point2f center1(static_cast<float>(bbox1.x) + static_cast<float>(bbox1.width) / 2.0f,
                      static_cast<float>(bbox1.y) + static_cast<float>(bbox1.height) / 2.0f);
  cv::Point2f center2(static_cast<float>(bbox2.x) + static_cast<float>(bbox2.width) / 2.0f,
                      static_cast<float>(bbox2.y) + static_cast<float>(bbox2.height) / 2.0f);

  cv::Point2f diff = center1 - center2;
  return std::sqrt(diff.x * diff.x + diff.y * diff.y);
}

bool VehicleAnalytics::isInDangerZone(const cv::Rect& bbox, const cv::Size& frame_size) {
  // Danger zone is the lower portion of the frame (close to ego vehicle)
  float danger_zone_top =
      static_cast<float>(frame_size.height) * (1.0f - config_.danger_zone_ratio);
  return static_cast<float>(bbox.y + bbox.height) > danger_zone_top;
}

bool VehicleAnalytics::isInWarningZone(const cv::Rect& bbox, const cv::Size& frame_size) {
  float warning_zone_top =
      static_cast<float>(frame_size.height) * (1.0f - config_.warning_zone_ratio);
  return static_cast<float>(bbox.y + bbox.height) > warning_zone_top;
}

float VehicleAnalytics::estimateDistance(const cv::Rect& bbox, const cv::Size& frame_size) {
  // Simple distance estimation based on bbox size and position
  // Larger objects closer to bottom of frame are closer
  float normalized_y =
      static_cast<float>(bbox.y + bbox.height) / static_cast<float>(frame_size.height);
  float normalized_area = static_cast<float>(bbox.area()) / static_cast<float>(frame_size.area());

  // Combine position and size for distance estimate (lower values = closer)
  return (1.0f - normalized_y) + (1.0f - normalized_area * 100.0f);
}

void VehicleAnalytics::startTimer() { timer_start_ = steady_clock::now(); }

float VehicleAnalytics::getElapsedMs() {
  auto end = steady_clock::now();
  return static_cast<float>(duration_cast<microseconds>(end - timer_start_).count()) / 1000.0f;
}

void VehicleAnalytics::AnalyticsStats::reset() {
  total_vehicles_detected = 0;
  total_tracks_created = 0;
  avg_tracking_accuracy = 0.0f;
  avg_processing_time_ms = 0.0f;
  collision_warnings_issued = 0;
}

// Factory function
std::unique_ptr<VehicleAnalytics> createVehicleAnalytics(const VehicleAnalyticsConfig& config) {
  return std::make_unique<VehicleAnalytics>(config);
}

// Visualization utilities
namespace VehicleViz {

cv::Mat drawVehicleTracks(const cv::Mat& frame, const std::vector<VehicleTrack>& tracks) {
  cv::Mat result = frame.clone();

  for (const auto& track : tracks) {
    // Draw bounding box with different colors for different vehicle types
    cv::Scalar color;
    switch (track.type) {
      case VehicleType::CAR:
        color = cv::Scalar(0, 255, 0);
        break;  // Green
      case VehicleType::TRUCK:
        color = cv::Scalar(0, 0, 255);
        break;  // Red
      case VehicleType::BUS:
        color = cv::Scalar(255, 0, 0);
        break;  // Blue
      case VehicleType::MOTORCYCLE:
        color = cv::Scalar(0, 255, 255);
        break;  // Yellow
      case VehicleType::BICYCLE:
        color = cv::Scalar(255, 255, 0);
        break;  // Cyan
      default:
        color = cv::Scalar(128, 128, 128);
        break;  // Gray
    }

    // Draw bounding box
    cv::rectangle(result, track.current_bbox, color, 2);

    // Draw track ID and vehicle type
    std::string label = "ID:" + std::to_string(track.track_id);
    cv::putText(result, label, cv::Point(track.current_bbox.x, track.current_bbox.y - 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

    // Draw speed if available
    if (track.speed_estimate > 0.1f) {
      std::string speed_label =
          "Speed: " + std::to_string(static_cast<int>(track.speed_estimate)) + "px/f";
      cv::putText(result, speed_label, cv::Point(track.current_bbox.x, track.current_bbox.y - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
    }

    // Draw trajectory
    if (track.trajectory.size() > 1) {
      for (size_t i = 1; i < track.trajectory.size(); ++i) {
        cv::line(result, track.trajectory[i - 1], track.trajectory[i], color, 1);
      }
    }

    // Draw velocity vector
    if (track.speed_estimate > 1.0f) {
      cv::Point2f end_point = track.center + track.velocity * 5.0f;  // Scale for visibility
      cv::arrowedLine(result, track.center, end_point, color, 2);
    }
  }

  return result;
}

cv::Mat drawSafetyZones(const cv::Mat& frame, const VehicleAnalyticsConfig& config) {
  cv::Mat result = frame.clone();

  // Draw danger zone (red tint)
  int danger_zone_top =
      static_cast<int>(static_cast<float>(frame.rows) * (1.0f - config.danger_zone_ratio));
  cv::Rect danger_zone(0, danger_zone_top, frame.cols, frame.rows - danger_zone_top);
  cv::Mat danger_overlay = result(danger_zone);
  danger_overlay += cv::Scalar(0, 0, 30);  // Red tint

  // Draw warning zone (yellow tint)
  int warning_zone_top =
      static_cast<int>(static_cast<float>(frame.rows) * (1.0f - config.warning_zone_ratio));
  cv::Rect warning_zone(0, warning_zone_top, frame.cols, danger_zone_top - warning_zone_top);
  if (warning_zone.height > 0) {
    cv::Mat warning_overlay = result(warning_zone);
    warning_overlay += cv::Scalar(0, 15, 15);  // Yellow tint
  }

  // Draw zone labels
  cv::putText(result, "DANGER ZONE", cv::Point(10, danger_zone_top + 30), cv::FONT_HERSHEY_SIMPLEX,
              0.7, cv::Scalar(0, 0, 255), 2);

  if (warning_zone.height > 0) {
    cv::putText(result, "WARNING ZONE", cv::Point(10, warning_zone_top + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
  }

  return result;
}

cv::Mat drawLaneDetection(const cv::Mat& frame, const LaneInfo& lane_info) {
  cv::Mat result = frame.clone();

  if (!lane_info.lanes_detected) return result;

  // Draw left lane
  if (!lane_info.left_lane.empty()) {
    for (size_t i = 1; i < lane_info.left_lane.size(); ++i) {
      cv::line(result, lane_info.left_lane[i - 1], lane_info.left_lane[i], cv::Scalar(255, 255, 0),
               3);  // Cyan
    }
  }

  // Draw right lane
  if (!lane_info.right_lane.empty()) {
    for (size_t i = 1; i < lane_info.right_lane.size(); ++i) {
      cv::line(result, lane_info.right_lane[i - 1], lane_info.right_lane[i],
               cv::Scalar(255, 255, 0), 3);  // Cyan
    }
  }

  // Draw vanishing point
  if (lane_info.vanishing_point.x > 0 && lane_info.vanishing_point.y > 0) {
    cv::circle(result, lane_info.vanishing_point, 5, cv::Scalar(0, 255, 255), -1);
  }

  return result;
}

cv::Mat drawProximityWarnings(const cv::Mat& frame, const VehicleAnalyticsResult& result) {
  cv::Mat output = frame.clone();

  // Draw collision warning
  if (result.collision_warning) {
    cv::putText(output, "COLLISION WARNING!", cv::Point(frame.cols / 2 - 150, 50),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 3);

    // Flash effect
    cv::Mat warning_overlay;
    output.copyTo(warning_overlay);
    warning_overlay += cv::Scalar(0, 0, 50);  // Red flash
    cv::addWeighted(output, 0.7, warning_overlay, 0.3, 0, output);
  }

  // Highlight vehicles in danger zone
  for (const auto& track : result.active_tracks) {
    auto it = std::find(result.danger_zone_vehicles.begin(), result.danger_zone_vehicles.end(),
                        track.track_id);
    if (it != result.danger_zone_vehicles.end()) {
      cv::rectangle(output, track.current_bbox, cv::Scalar(0, 0, 255), 4);  // Red highlight
      cv::putText(output, "DANGER", cv::Point(track.current_bbox.x, track.current_bbox.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }
  }

  return output;
}

cv::Mat drawAnalyticsOverlay(const cv::Mat& frame, const VehicleAnalyticsResult& result) {
  cv::Mat output = frame.clone();

  // Draw analytics information panel
  int panel_width = 300;
  int panel_height = 200;
  cv::Rect panel_rect(frame.cols - panel_width - 10, 10, panel_width, panel_height);

  // Semi-transparent background
  cv::Mat panel_overlay = output(panel_rect);
  panel_overlay += cv::Scalar(50, 50, 50);
  cv::addWeighted(output(panel_rect), 0.7, panel_overlay, 0.3, 0, output(panel_rect));

  // Add text information
  int y_offset = 30;
  int line_height = 25;

  std::string total_vehicles = "Total Vehicles: " + std::to_string(result.total_vehicles);
  cv::putText(output, total_vehicles, cv::Point(panel_rect.x + 10, panel_rect.y + y_offset),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  y_offset += line_height;

  std::string active_tracks = "Active Tracks: " + std::to_string(result.active_tracks.size());
  cv::putText(output, active_tracks, cv::Point(panel_rect.x + 10, panel_rect.y + y_offset),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  y_offset += line_height;

  std::string approaching = "Approaching: " + std::to_string(result.approaching_vehicles.size());
  cv::putText(output, approaching, cv::Point(panel_rect.x + 10, panel_rect.y + y_offset),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
  y_offset += line_height;

  std::string danger_zone = "In Danger Zone: " + std::to_string(result.danger_zone_vehicles.size());
  cv::putText(output, danger_zone, cv::Point(panel_rect.x + 10, panel_rect.y + y_offset),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
  y_offset += line_height;

  std::string traffic_density =
      "Traffic Density: " + std::to_string(static_cast<int>(result.traffic_density * 100)) + "%";
  cv::putText(output, traffic_density, cv::Point(panel_rect.x + 10, panel_rect.y + y_offset),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  y_offset += line_height;

  std::string processing_time =
      "Analytics: " + std::to_string(static_cast<int>(result.analytics_time_ms)) + "ms";
  cv::putText(output, processing_time, cv::Point(panel_rect.x + 10, panel_rect.y + y_offset),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

  return output;
}

}  // namespace VehicleViz
