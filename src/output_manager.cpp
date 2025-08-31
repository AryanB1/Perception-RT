#include "output_manager.hpp"

#include <spdlog/fmt/fmt.h>

#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>

#include "vehicle_analytics.hpp"

OutputManager::OutputManager(const OutputConfig& config)
    : config_(config), frame_width_(0), frame_height_(0), csv_header_written_(false),
      video_writer_initialized_(false) {
  // Set logging level
  if (config_.log_level == "debug") {
    spdlog::set_level(spdlog::level::debug);
  } else if (config_.log_level == "info") {
    spdlog::set_level(spdlog::level::info);
  } else if (config_.log_level == "warn") {
    spdlog::set_level(spdlog::level::warn);
  } else if (config_.log_level == "error") {
    spdlog::set_level(spdlog::level::err);
  }

  stats_.reset();

  // Initialize CSV logging
  if (config_.enable_csv_logging) {
    initializeCSV();
  }

  // Reserve space for frame buffer if memory buffering is enabled
  if (config_.use_memory_buffering && config_.enable_video_output) {
    buffered_frames_.reserve(config_.max_buffered_frames);
    spdlog::info("Memory buffering enabled - max {} frames, flush threshold: {}",
                 config_.max_buffered_frames, config_.buffer_flush_threshold);
  }
}

OutputManager::~OutputManager() { 
  cleanup(); 
}

bool OutputManager::initialize(int frame_width, int frame_height) {
  frame_width_ = frame_width;
  frame_height_ = frame_height;

  if (config_.enable_video_output) {
    ensureOutputDirectory();

    // Pre-allocate buffer space for better performance
    buffered_frames_.reserve(10000);  // Reserve space for ~5 minutes at 30fps

    spdlog::info("Video output will be buffered in memory: {} ({}x{} @ {}fps)",
                 config_.output_video_path, frame_width, frame_height, config_.output_fps);
    spdlog::info("Frames will be written to file when processing completes");
  }

  if (!config_.verbose_logging) {
    spdlog::info("Verbose logging disabled - performance summaries every {}s",
                 config_.performance_summary_interval);
  }

  return true;
}

void OutputManager::cleanup() {
  // Write all buffered frames to video file
  finalizeVideo();

  // Close CSV file
  closeCSV();

  // Final performance summary
  logPerformanceSummary(true);
}

void OutputManager::processFrame(const cv::Mat& frame, const MLResult& ml_result,
                                 const GpuMotionResult& motion_result, uint64_t frame_id,
                                 float pre_ms, float inf_ms, float post_ms, float e2e_ms,
                                 bool missed) {
  // Update statistics
  stats_.total_frames++;
  stats_.total_inference_time += inf_ms;
  stats_.total_e2e_time += e2e_ms;

  if (missed) stats_.missed_frames++;
  if (motion_result.motion_detected) stats_.motion_detected_frames++;

  if (ml_result.vehicle_analytics_enabled && ml_result.vehicle_analytics) {
    stats_.vehicles_detected += ml_result.vehicle_analytics->total_vehicles;
    if (ml_result.vehicle_analytics->collision_warning) {
      stats_.collision_warnings++;
    }
  }

  // Calculate video timestamp (frame_id / fps)
  float video_timestamp_sec = static_cast<float>(frame_id) / 30.0f;  // Assuming 30fps

  // Write comprehensive CSV log entry
  if (config_.enable_csv_logging) {
    writeCSVRow(frame_id, video_timestamp_sec, pre_ms, inf_ms, post_ms, e2e_ms, missed, 
                motion_result, ml_result);
  }

  // Handle logging
  if (config_.verbose_logging) {
    logFrameInfo(frame_id, pre_ms, inf_ms, post_ms, e2e_ms, missed, motion_result, ml_result);
  } else {
    // Only log important events
    if (ml_result.vehicle_analytics_enabled && ml_result.vehicle_analytics) {
      if (ml_result.vehicle_analytics->collision_warning) {
        spdlog::warn("COLLISION WARNING - Frame {}: {} vehicles, {} in danger zone", frame_id,
                     ml_result.vehicle_analytics->total_vehicles,
                     ml_result.vehicle_analytics->danger_zone_vehicles.size());
      }
    }

    // Log performance summary periodically
    logPerformanceSummary();
  }

  // Handle video output with overlays - buffer frames instead of writing immediately
  if (config_.enable_video_output) {
    cv::Mat output_frame = frame.clone();

    if (config_.enable_overlay) {
      output_frame = generateOverlay(output_frame, ml_result);
    }

    // Buffer the frame in memory instead of writing to file
    buffered_frames_.push_back(output_frame.clone());

    // Periodic progress update
    if (buffered_frames_.size() % 300 == 0) {  // Every 10 seconds at 30fps
      spdlog::debug("Buffered {} frames in memory", buffered_frames_.size());
    }
  }
}

void OutputManager::logFrameInfo(uint64_t frame_id, float pre_ms, float inf_ms, float post_ms,
                                 float e2e_ms, bool missed, const GpuMotionResult& motion_result,
                                 const MLResult& ml_result) {
  // This is the original verbose logging format, but cleaner
  std::string vehicle_info = "";
  if (ml_result.vehicle_analytics_enabled && ml_result.vehicle_analytics) {
    vehicle_info =
        fmt::format(" | vehicles={} tracks={} danger={} approaching={}{}",
                    ml_result.vehicle_analytics->total_vehicles,
                    ml_result.vehicle_analytics->active_tracks.size(),
                    ml_result.vehicle_analytics->danger_zone_vehicles.size(),
                    ml_result.vehicle_analytics->approaching_vehicles.size(),
                    ml_result.vehicle_analytics->collision_warning ? " [COLLISION_WARNING]" : "");
  }

  spdlog::info(
      "frame_id={} [CUDA] pre_ms={:.3f} inf_ms={:.3f} post_ms={:.3f} e2e_ms={:.3f} "
      "missed={} motion={} motion_intensity={:.4f} ml_objects={}{}",
      frame_id, pre_ms, inf_ms, post_ms, e2e_ms, missed, motion_result.motion_detected,
      motion_result.motion_intensity, ml_result.total_objects, vehicle_info);
}

void OutputManager::logPerformanceSummary(bool force) {
  auto now = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - stats_.last_summary);

  if (!force && duration.count() < config_.performance_summary_interval) {
    return;
  }

  spdlog::info("=== PERFORMANCE SUMMARY ===");
  spdlog::info("Frames processed: {}", stats_.total_frames);
  spdlog::info("Average FPS: {:.1f}", stats_.getFPS());
  spdlog::info("Average inference time: {:.3f}ms", stats_.getAvgInferenceTime());
  spdlog::info("Average E2E time: {:.3f}ms", stats_.getAvgE2ETime());
  spdlog::info("Frame miss rate: {:.2f}%", stats_.getMissRate());
  spdlog::info("Motion detected: {}/{} frames ({:.1f}%)", stats_.motion_detected_frames,
               stats_.total_frames,
               stats_.total_frames > 0 ? static_cast<double>(stats_.motion_detected_frames) /
                                             static_cast<double>(stats_.total_frames) * 100.0
                                       : 0.0);
  spdlog::info("Total vehicles detected: {}", stats_.vehicles_detected);
  spdlog::info("Collision warnings issued: {}", stats_.collision_warnings);

  if (config_.enable_video_output) {
    spdlog::info("Buffered frames: {} (will write to {})", buffered_frames_.size(),
                 config_.output_video_path);
  }

  stats_.last_summary = now;
}

void OutputManager::finalizeVideo() {
  if (!config_.enable_video_output || buffered_frames_.empty()) {
    return;
  }

  spdlog::info("Writing {} buffered frames to video file...", buffered_frames_.size());

  // Initialize video writer now
  int fourcc = cv::VideoWriter::fourcc(config_.video_codec[0], config_.video_codec[1],
                                       config_.video_codec[2], config_.video_codec[3]);

  video_writer_.open(config_.output_video_path, fourcc, config_.output_fps,
                     cv::Size(frame_width_, frame_height_));

  if (!video_writer_.isOpened()) {
    spdlog::error("Failed to initialize video writer for: {}", config_.output_video_path);
    return;
  }

  // Write all buffered frames
  auto start_time = std::chrono::steady_clock::now();
  size_t frames_written = 0;

  for (const auto& frame : buffered_frames_) {
    video_writer_.write(frame);
    frames_written++;

    // Progress update every 1000 frames
    if (frames_written % 1000 == 0) {
      spdlog::info("Written {}/{} frames to video file", frames_written, buffered_frames_.size());
    }
  }

  video_writer_.release();

  auto end_time = std::chrono::steady_clock::now();
  auto write_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  spdlog::info("Video output completed: {} ({} frames in {:.2f}s)", config_.output_video_path,
               frames_written, static_cast<double>(write_duration.count()) / 1000.0);

  // Clear buffered frames to free memory
  buffered_frames_.clear();
  buffered_frames_.shrink_to_fit();
}

cv::Mat OutputManager::generateOverlay(const cv::Mat& frame, const MLResult& ml_result) {
  cv::Mat result = frame.clone();

  if (config_.show_vehicle_boxes) {
    result = drawVehicleBoxes(result, ml_result);
  }

  if (config_.show_collision_warnings) {
    result = drawCollisionWarnings(result, ml_result);
  }

  if (config_.show_analytics_panel) {
    result = drawAnalyticsPanel(result, ml_result);
  }

  return result;
}

cv::Mat OutputManager::drawVehicleBoxes(const cv::Mat& frame, const MLResult& ml_result) {
  cv::Mat result = frame.clone();

  if (!ml_result.vehicle_analytics_enabled || !ml_result.vehicle_analytics) {
    return result;
  }

  const auto& analytics = *ml_result.vehicle_analytics;

  // Draw vehicle bounding boxes with tracking IDs
  for (const auto& track : analytics.active_tracks) {
    cv::Scalar color = getVehicleTypeColor(track.type);

    // Check if this vehicle is in danger zone
    bool in_danger =
        std::find(analytics.danger_zone_vehicles.begin(), analytics.danger_zone_vehicles.end(),
                  track.track_id) != analytics.danger_zone_vehicles.end();

    bool approaching =
        std::find(analytics.approaching_vehicles.begin(), analytics.approaching_vehicles.end(),
                  track.track_id) != analytics.approaching_vehicles.end();

    // Adjust color and thickness based on status
    int thickness = in_danger ? 4 : (approaching ? 3 : 2);
    if (in_danger) {
      color = cv::Scalar(0, 0, 255);  // Red for danger
    } else if (approaching) {
      color = cv::Scalar(0, 165, 255);  // Orange for approaching
    }

    // Draw bounding box
    cv::rectangle(result, track.current_bbox, color, thickness);

    // Draw vehicle type and tracking ID
    if (config_.show_tracking_ids) {
      std::string label = fmt::format("ID:{} {}", track.track_id, getVehicleTypeName(track.type));
      cv::putText(result, label, cv::Point(track.current_bbox.x, track.current_bbox.y - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }

    // Draw speed if available
    if (track.speed_estimate > 0) {
      // Convert pixels/frame to km/h using approximation
      // Assuming: 1 pixel ≈ 0.1 meters, 30 fps
      float pixels_per_frame = track.speed_estimate;
      float meters_per_second = pixels_per_frame * 0.1f * 30.0f;  // 0.1m per pixel, 30 fps
      float speed_kmh = meters_per_second * 3.6f;                 // convert m/s to km/h

      if (speed_kmh > 1.0f) {  // Only show if meaningful speed
        std::string speed_label = fmt::format("{:.0f} km/h", speed_kmh);
        cv::putText(
            result, speed_label,
            cv::Point(track.current_bbox.x, track.current_bbox.y + track.current_bbox.height + 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
      }
    }
  }

  return result;
}

cv::Mat OutputManager::drawCollisionWarnings(const cv::Mat& frame, const MLResult& ml_result) {
  cv::Mat result = frame.clone();

  if (!ml_result.vehicle_analytics_enabled || !ml_result.vehicle_analytics) {
    return result;
  }

  const auto& analytics = *ml_result.vehicle_analytics;

  if (analytics.collision_warning) {
    // Draw prominent collision warning
    cv::putText(result, "COLLISION WARNING!", cv::Point(frame.cols / 2 - 200, 80),
                cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 4);

    // Add flashing effect
    cv::Mat warning_overlay = result.clone();
    warning_overlay += cv::Scalar(0, 0, 30);  // Red tint
    cv::addWeighted(result, 0.8, warning_overlay, 0.2, 0, result);

    // Draw danger zone visualization
    cv::circle(result, cv::Point(frame.cols / 2, frame.rows - 100), 150, cv::Scalar(0, 0, 255), 3);
    cv::putText(result, "DANGER ZONE", cv::Point(frame.cols / 2 - 80, frame.rows - 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
  }

  return result;
}

cv::Mat OutputManager::drawAnalyticsPanel(const cv::Mat& frame, const MLResult& ml_result) {
  cv::Mat result = frame.clone();

  if (!ml_result.vehicle_analytics_enabled || !ml_result.vehicle_analytics) {
    return result;
  }

  const auto& analytics = *ml_result.vehicle_analytics;

  // Panel configuration
  int panel_width = 320;
  int panel_height = 180;
  cv::Rect panel_rect(frame.cols - panel_width - 20, 20, panel_width, panel_height);

  // Semi-transparent background
  cv::Mat panel_overlay = result(panel_rect);
  panel_overlay = cv::Scalar(0, 0, 0);  // Black background
  cv::addWeighted(result(panel_rect), 0.3, panel_overlay, 0.7, 0, result(panel_rect));

  // Add border
  cv::rectangle(result, panel_rect, cv::Scalar(255, 255, 255), 2);

  // Add text information
  int y_offset = 35;
  int line_height = 22;
  cv::Scalar text_color(255, 255, 255);
  double font_scale = 0.6;
  int thickness = 1;

  // Title
  cv::putText(result, "VEHICLE ANALYTICS", cv::Point(panel_rect.x + 10, panel_rect.y + 20),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

  // Vehicle count
  std::string vehicles_text = fmt::format("Vehicles: {}", analytics.total_vehicles);
  cv::putText(result, vehicles_text, cv::Point(panel_rect.x + 10, panel_rect.y + y_offset),
              cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness);
  y_offset += line_height;

  // Active tracks
  std::string tracks_text = fmt::format("Active Tracks: {}", analytics.active_tracks.size());
  cv::putText(result, tracks_text, cv::Point(panel_rect.x + 10, panel_rect.y + y_offset),
              cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness);
  y_offset += line_height;

  // Approaching vehicles
  std::string approaching_text =
      fmt::format("Approaching: {}", analytics.approaching_vehicles.size());
  cv::Scalar approaching_color =
      analytics.approaching_vehicles.size() > 0 ? cv::Scalar(0, 255, 255) : text_color;
  cv::putText(result, approaching_text, cv::Point(panel_rect.x + 10, panel_rect.y + y_offset),
              cv::FONT_HERSHEY_SIMPLEX, font_scale, approaching_color, thickness);
  y_offset += line_height;

  // Danger zone vehicles
  std::string danger_text =
      fmt::format("In Danger Zone: {}", analytics.danger_zone_vehicles.size());
  cv::Scalar danger_color =
      analytics.danger_zone_vehicles.size() > 0 ? cv::Scalar(0, 0, 255) : text_color;
  cv::putText(result, danger_text, cv::Point(panel_rect.x + 10, panel_rect.y + y_offset),
              cv::FONT_HERSHEY_SIMPLEX, font_scale, danger_color, thickness);
  y_offset += line_height;

  // Traffic density
  std::string density_text =
      fmt::format("Traffic Density: {:.0f}%", analytics.traffic_density * 100);
  cv::putText(result, density_text, cv::Point(panel_rect.x + 10, panel_rect.y + y_offset),
              cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness);
  y_offset += line_height;

  // Processing time
  std::string time_text = fmt::format("Analytics: {:.1f}ms", analytics.analytics_time_ms);
  cv::putText(result, time_text, cv::Point(panel_rect.x + 10, panel_rect.y + y_offset),
              cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness);

  return result;
}

void OutputManager::ensureOutputDirectory() {
  std::filesystem::path output_path(config_.output_video_path);
  std::filesystem::path directory = output_path.parent_path();

  if (!directory.empty() && !std::filesystem::exists(directory)) {
    std::filesystem::create_directories(directory);
    spdlog::info("Created output directory: {}", directory.string());
  }
}

cv::Scalar OutputManager::getVehicleTypeColor(VehicleType type) {
  switch (type) {
    case VehicleType::CAR:
      return cv::Scalar(0, 255, 0);  // Green
    case VehicleType::TRUCK:
      return cv::Scalar(255, 0, 0);  // Blue
    case VehicleType::BUS:
      return cv::Scalar(0, 255, 255);  // Yellow
    case VehicleType::MOTORCYCLE:
      return cv::Scalar(255, 0, 255);  // Magenta
    case VehicleType::BICYCLE:
      return cv::Scalar(128, 255, 0);  // Light green
    default:
      return cv::Scalar(255, 255, 255);  // White
  }
}

std::string OutputManager::getVehicleTypeName(VehicleType type) {
  switch (type) {
    case VehicleType::CAR:
      return "Car";
    case VehicleType::TRUCK:
      return "Truck";
    case VehicleType::BUS:
      return "Bus";
    case VehicleType::MOTORCYCLE:
      return "Motorcycle";
    case VehicleType::BICYCLE:
      return "Bicycle";
    default:
      return "Vehicle";
  }
}

void OutputManager::initializeCSV() {
  if (!config_.enable_csv_logging) return;

  // Ensure output directory exists
  std::filesystem::path csv_path(config_.csv_output_path);
  std::filesystem::path directory = csv_path.parent_path();
  
  if (!directory.empty() && !std::filesystem::exists(directory)) {
    std::filesystem::create_directories(directory);
    spdlog::info("Created CSV output directory: {}", directory.string());
  }

  // Open CSV file
  csv_file_.open(config_.csv_output_path, std::ios::out | std::ios::trunc);
  if (!csv_file_.is_open()) {
    spdlog::error("Failed to open CSV file for writing: {}", config_.csv_output_path);
    return;
  }

  // Write CSV header
  writeCSVHeader();
  csv_header_written_ = true;
  
  spdlog::info("CSV logging initialized: {}", config_.csv_output_path);
}

void OutputManager::writeCSVHeader() {
  if (!csv_file_.is_open()) return;

  if (config_.csv_comprehensive_mode) {
    // Comprehensive CSV header with all possible fields
    csv_file_ << "frame_id,video_timestamp_sec,pre_process_ms,inference_ms,post_process_ms,"
              << "end_to_end_ms,deadline_missed,motion_detected,motion_intensity,motion_pixels,"
              << "motion_bbox_x,motion_bbox_y,motion_bbox_w,motion_bbox_h,"
              << "total_objects,max_confidence,significant_motion,"
              << "vehicles_detected,active_tracks,approaching_vehicles,danger_zone_vehicles,"
              << "overtaking_vehicles,vehicles_in_ego_lane,ego_lane_clear,"
              << "traffic_density,collision_warning,lane_change_safe,closest_vehicle_distance,"
              << "analytics_time_ms,tracks_updated,new_tracks,lost_tracks,"
              << "current_fps,avg_inference_time,total_frames_processed,"
              << "cumulative_vehicles_detected,cumulative_collision_warnings,"
              << "track_details,detection_details\n";
  } else {
    // Compact CSV header (similar to existing format)
    csv_file_ << "frame_id,video_timestamp_sec,pre_process_ms,inference_ms,post_process_ms,"
              << "end_to_end_ms,deadline_missed,motion_detected,motion_intensity,"
              << "total_objects,vehicles_detected,active_tracks,approaching_vehicles,"
              << "danger_zone_vehicles,traffic_density,collision_warning,"
              << "current_fps,avg_inference_time,total_frames_processed\n";
  }
  
  csv_file_.flush();
}

void OutputManager::writeCSVRow(uint64_t frame_id, float video_timestamp_sec, float pre_ms, 
                                float inf_ms, float post_ms, float e2e_ms, bool missed,
                                const GpuMotionResult& motion_result, const MLResult& ml_result) {
  if (!csv_file_.is_open()) return;

  // Basic timing and frame info
  csv_file_ << frame_id << ","
            << std::fixed << std::setprecision(3) << video_timestamp_sec << ","
            << std::fixed << std::setprecision(3) << pre_ms << ","
            << std::fixed << std::setprecision(3) << inf_ms << ","
            << std::fixed << std::setprecision(3) << post_ms << ","
            << std::fixed << std::setprecision(3) << e2e_ms << ","
            << (missed ? 1 : 0) << ","
            << (motion_result.motion_detected ? 1 : 0) << ","
            << std::fixed << std::setprecision(3) << motion_result.motion_intensity;

  if (config_.csv_comprehensive_mode) {
    // Extended motion information
    csv_file_ << "," << motion_result.motion_pixels
              << "," << ml_result.motion_bbox.x
              << "," << ml_result.motion_bbox.y  
              << "," << ml_result.motion_bbox.width
              << "," << ml_result.motion_bbox.height;
  }

  // Object detection info
  csv_file_ << "," << ml_result.total_objects
            << "," << std::fixed << std::setprecision(3) << ml_result.max_confidence;

  if (config_.csv_comprehensive_mode) {
    csv_file_ << "," << (ml_result.significant_motion ? 1 : 0);
  }

  // Vehicle analytics
  int vehicles_detected = 0;
  int active_tracks = 0;
  int approaching_vehicles = 0;
  int danger_zone_vehicles = 0;
  int overtaking_vehicles = 0;
  int vehicles_in_ego_lane = 0;
  bool ego_lane_clear = true;
  float traffic_density = 0.0f;
  bool collision_warning = false;
  bool lane_change_safe = true;
  float closest_vehicle_distance = std::numeric_limits<float>::max();
  float analytics_time_ms = 0.0f;
  int tracks_updated = 0;
  int new_tracks = 0;
  int lost_tracks = 0;

  if (ml_result.vehicle_analytics_enabled && ml_result.vehicle_analytics) {
    const auto& analytics = *ml_result.vehicle_analytics;
    vehicles_detected = analytics.total_vehicles;
    active_tracks = static_cast<int>(analytics.active_tracks.size());
    approaching_vehicles = static_cast<int>(analytics.approaching_vehicles.size());
    danger_zone_vehicles = static_cast<int>(analytics.danger_zone_vehicles.size());
    overtaking_vehicles = static_cast<int>(analytics.overtaking_vehicles.size());
    vehicles_in_ego_lane = analytics.vehicles_in_ego_lane;
    ego_lane_clear = analytics.ego_lane_clear;
    traffic_density = analytics.traffic_density;
    collision_warning = analytics.collision_warning;
    lane_change_safe = analytics.lane_change_safe;
    closest_vehicle_distance = analytics.closest_vehicle_distance;
    analytics_time_ms = analytics.analytics_time_ms;
    tracks_updated = analytics.tracks_updated;
    new_tracks = analytics.new_tracks;
    lost_tracks = analytics.lost_tracks;
  }

  csv_file_ << "," << vehicles_detected
            << "," << active_tracks
            << "," << approaching_vehicles
            << "," << danger_zone_vehicles;

  if (config_.csv_comprehensive_mode) {
    csv_file_ << "," << overtaking_vehicles
              << "," << vehicles_in_ego_lane
              << "," << (ego_lane_clear ? 1 : 0);
  }

  csv_file_ << "," << std::fixed << std::setprecision(3) << traffic_density
            << "," << (collision_warning ? 1 : 0);

  if (config_.csv_comprehensive_mode) {
    csv_file_ << "," << (lane_change_safe ? 1 : 0)
              << "," << std::fixed << std::setprecision(3) << closest_vehicle_distance
              << "," << std::fixed << std::setprecision(3) << analytics_time_ms
              << "," << tracks_updated
              << "," << new_tracks
              << "," << lost_tracks;
  }

  // Performance metrics
  csv_file_ << "," << std::fixed << std::setprecision(3) << stats_.getFPS()
            << "," << std::fixed << std::setprecision(3) << stats_.getAvgInferenceTime()
            << "," << stats_.total_frames;

  if (config_.csv_comprehensive_mode) {
    csv_file_ << "," << stats_.vehicles_detected
              << "," << stats_.collision_warnings;

    // Detailed track information (JSON-like format)
    csv_file_ << ",\"[";
    if (ml_result.vehicle_analytics_enabled && ml_result.vehicle_analytics) {
      bool first_track = true;
      for (const auto& track : ml_result.vehicle_analytics->active_tracks) {
        if (!first_track) csv_file_ << ",";
        csv_file_ << "{id:" << track.track_id
                  << ",type:" << static_cast<int>(track.type)
                  << ",conf:" << std::fixed << std::setprecision(2) << track.confidence
                  << ",speed:" << std::fixed << std::setprecision(1) << track.speed_estimate
                  << ",x:" << track.current_bbox.x
                  << ",y:" << track.current_bbox.y
                  << ",w:" << track.current_bbox.width
                  << ",h:" << track.current_bbox.height << "}";
        first_track = false;
      }
    }
    csv_file_ << "]\"";

    // Detailed detection information (JSON-like format)
    csv_file_ << ",\"[";
    bool first_det = true;
    for (const auto& detection : ml_result.detections) {
      if (!first_det) csv_file_ << ",";
      csv_file_ << "{cls:" << detection.class_id
                << ",conf:" << std::fixed << std::setprecision(2) << detection.confidence
                << ",x:" << detection.bbox.x
                << ",y:" << detection.bbox.y
                << ",w:" << detection.bbox.width
                << ",h:" << detection.bbox.height << "}";
      first_det = false;
    }
    csv_file_ << "]\"";
  }

  csv_file_ << "\n";
  csv_file_.flush();  // Ensure data is written immediately
}

void OutputManager::closeCSV() {
  if (csv_file_.is_open()) {
    csv_file_.close();
    if (config_.enable_csv_logging) {
      spdlog::info("CSV logging completed: {}", config_.csv_output_path);
    }
  }
}
