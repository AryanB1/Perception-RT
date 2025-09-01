#pragma once

#include <spdlog/spdlog.h>

#include <chrono>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "gpu.hpp"
#include "types.hpp"
#include "vehicle_analytics.hpp"

struct OutputConfig {
  // Logging settings
  bool verbose_logging = false;
  std::string log_level = "info";
  int performance_summary_interval = 30;  // seconds

  // CSV logging settings
  bool enable_csv_logging = true;
  std::string csv_output_path = "output/frame_log.csv";
  bool csv_comprehensive_mode = true;  // Log everything vs. compact mode

  // Video output settings
  bool enable_video_output = false;
  std::string output_video_path = "output/dashcam_analytics.mp4";
  std::string video_codec = "mp4v";
  int output_fps = 30;

  // Overlay settings
  bool enable_overlay = true;
  bool show_vehicle_boxes = true;
  bool show_tracking_ids = true;
  bool show_analytics_panel = true;
  bool show_collision_warnings = true;
  float overlay_opacity = 0.8f;

  // Memory buffering settings
  bool use_memory_buffering = true;
  size_t max_buffered_frames = 1000;    // Maximum frames to buffer in memory
  size_t buffer_flush_threshold = 500;  // Flush when this many frames are buffered
};

struct PerformanceStats {
  uint64_t total_frames = 0;
  double total_inference_time = 0.0;
  double total_e2e_time = 0.0;
  uint64_t missed_frames = 0;
  uint64_t motion_detected_frames = 0;
  uint64_t vehicles_detected = 0;
  uint64_t collision_warnings = 0;

  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point last_summary;

  void reset() {
    total_frames = 0;
    total_inference_time = 0.0;
    total_e2e_time = 0.0;
    missed_frames = 0;
    motion_detected_frames = 0;
    vehicles_detected = 0;
    collision_warnings = 0;
    start_time = std::chrono::steady_clock::now();
    last_summary = start_time;
  }

  double getAvgInferenceTime() const {
    return total_frames > 0 ? total_inference_time / static_cast<double>(total_frames) : 0.0;
  }

  double getAvgE2ETime() const {
    return total_frames > 0 ? total_e2e_time / static_cast<double>(total_frames) : 0.0;
  }

  double getFPS() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
    double seconds = static_cast<double>(duration.count()) / 1000.0;
    return seconds > 0 ? static_cast<double>(total_frames) / seconds : 0.0;
  }

  double getMissRate() const {
    return total_frames > 0
               ? static_cast<double>(missed_frames) / static_cast<double>(total_frames) * 100.0
               : 0.0;
  }
};

class OutputManager {
public:
  explicit OutputManager(const OutputConfig& config);
  ~OutputManager();

  bool initialize(int frame_width, int frame_height);
  void cleanup();

  // Process frame and generate output
  void processFrame(const cv::Mat& frame, const MLResult& ml_result,
                    const GpuMotionResult& motion_result, uint64_t frame_id, float pre_ms,
                    float inf_ms, float post_ms, float e2e_ms, bool missed);

  // Finalize video output (write all buffered frames)
  void finalizeVideo();

  // Logging control
  void logFrameInfo(uint64_t frame_id, float pre_ms, float inf_ms, float post_ms, float e2e_ms,
                    bool missed, const GpuMotionResult& motion_result, const MLResult& ml_result);

  void logPerformanceSummary(bool force = false);

  // CSV logging methods
  void initializeCSV();
  void writeCSVHeader();
  void writeCSVRow(uint64_t frame_id, float video_timestamp_sec, float pre_ms, float inf_ms,
                   float post_ms, float e2e_ms, bool missed, const GpuMotionResult& motion_result,
                   const MLResult& ml_result);
  void closeCSV();

  // Video output
  bool isVideoOutputEnabled() const { return config_.enable_video_output; }
  size_t getBufferedFrameCount() const { return buffered_frames_.size(); }

private:
  OutputConfig config_;
  cv::VideoWriter video_writer_;
  PerformanceStats stats_;

  // CSV logging
  std::ofstream csv_file_;
  bool csv_header_written_;

  // Frame buffering for video output
  std::vector<cv::Mat> buffered_frames_;
  int frame_width_;
  int frame_height_;
  bool video_writer_initialized_;

  // Overlay generation
  cv::Mat generateOverlay(const cv::Mat& frame, const MLResult& ml_result);
  cv::Mat drawVehicleBoxes(const cv::Mat& frame, const MLResult& ml_result);
  cv::Mat drawAnalyticsPanel(const cv::Mat& frame, const MLResult& ml_result);
  cv::Mat drawCollisionWarnings(const cv::Mat& frame, const MLResult& ml_result);

  // Buffer management
  void bufferFrame(const cv::Mat& frame);
  void flushFrameBuffer();
  void initializeVideoWriter();

  // Utility
  void ensureOutputDirectory();
  cv::Scalar getVehicleTypeColor(VehicleType type);
  std::string getVehicleTypeName(VehicleType type);
};
