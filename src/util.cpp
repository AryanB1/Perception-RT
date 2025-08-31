#include "util.hpp"

#include <yaml-cpp/yaml.h>

AppConfig load_config(const std::string& path) {
  YAML::Node y = YAML::LoadFile(path);
  AppConfig c{};

  if (y["input"]) {
    if (y["input"]["uri"]) c.pipeline.uri = y["input"]["uri"].as<std::string>();
    if (y["input"]["width"]) c.pipeline.width = y["input"]["width"].as<int>();
    if (y["input"]["height"]) c.pipeline.height = y["input"]["height"].as<int>();
    if (y["input"]["fps"]) c.pipeline.fps = y["input"]["fps"].as<int>();
  }
  if (y["pipeline"]) {
    if (y["pipeline"]["budget_ms"]) c.deadline.budget_ms = y["pipeline"]["budget_ms"].as<double>();
    if (y["pipeline"]["target_fps"]) c.deadline.target_fps = y["pipeline"]["target_fps"].as<int>();
    if (y["pipeline"]["queue_capacity"])
      c.pipeline.queue_capacity = y["pipeline"]["queue_capacity"].as<int>();
    if (y["pipeline"]["drop_policy"])
      c.pipeline.drop_policy = y["pipeline"]["drop_policy"].as<std::string>();
  }
  if (y["controller"]) {
    auto n = y["controller"];
    if (n["switch_up_p95_ms"]) c.deadline.switch_up_p95_ms = n["switch_up_p95_ms"].as<double>();
    if (n["switch_down_p95_ms"])
      c.deadline.switch_down_p95_ms = n["switch_down_p95_ms"].as<double>();
    if (n["hysteresis_frames"]) c.deadline.hysteresis_frames = n["hysteresis_frames"].as<int>();
  }
  if (y["telemetry"] && y["telemetry"]["metrics_port"])
    c.metrics_port = y["telemetry"]["metrics_port"].as<int>();

  // Load ML Engine configuration
  if (y["ml_engine"]) {
    auto ml = y["ml_engine"];
    if (ml["yolo_model_path"])
      c.ml_config.yolo_model_path = ml["yolo_model_path"].as<std::string>();
    if (ml["yolo_engine_path"])
      c.ml_config.yolo_engine_path = ml["yolo_engine_path"].as<std::string>();
    if (ml["segmentation_model_path"])
      c.ml_config.segmentation_model_path = ml["segmentation_model_path"].as<std::string>();

    if (ml["use_tensorrt"]) c.ml_config.use_tensorrt = ml["use_tensorrt"].as<bool>();
    if (ml["use_fp16"]) c.ml_config.use_fp16 = ml["use_fp16"].as<bool>();
    if (ml["max_batch_size"]) c.ml_config.max_batch_size = ml["max_batch_size"].as<int>();
    if (ml["max_workspace_size"])
      c.ml_config.max_workspace_size = ml["max_workspace_size"].as<size_t>();

    if (ml["detection_threshold"])
      c.ml_config.detection_threshold = ml["detection_threshold"].as<float>();
    if (ml["nms_threshold"]) c.ml_config.nms_threshold = ml["nms_threshold"].as<float>();
    if (ml["max_detections"]) c.ml_config.max_detections = ml["max_detections"].as<int>();

    if (ml["max_corners"]) c.ml_config.max_corners = ml["max_corners"].as<int>();
    if (ml["optical_flow_threshold"])
      c.ml_config.optical_flow_threshold = ml["optical_flow_threshold"].as<float>();

    if (ml["segmentation_threshold"])
      c.ml_config.segmentation_threshold = ml["segmentation_threshold"].as<float>();

    if (ml["input_width"] && ml["input_height"]) {
      c.ml_config.input_size = cv::Size(ml["input_width"].as<int>(), ml["input_height"].as<int>());
    }

    if (ml["enable_detection"]) c.ml_config.enable_detection = ml["enable_detection"].as<bool>();
    if (ml["enable_optical_flow"])
      c.ml_config.enable_optical_flow = ml["enable_optical_flow"].as<bool>();
    if (ml["enable_segmentation"])
      c.ml_config.enable_segmentation = ml["enable_segmentation"].as<bool>();

    // Vehicle analytics configuration
    if (ml["enable_vehicle_analytics"])
      c.ml_config.enable_vehicle_analytics = ml["enable_vehicle_analytics"].as<bool>();
    if (ml["enable_tracking"]) c.ml_config.enable_tracking = ml["enable_tracking"].as<bool>();
    if (ml["enable_proximity_detection"])
      c.ml_config.enable_proximity_detection = ml["enable_proximity_detection"].as<bool>();
    if (ml["focus_vehicle_detection"])
      c.ml_config.focus_vehicle_detection = ml["focus_vehicle_detection"].as<bool>();

    if (ml["vehicle_classes"]) {
      c.ml_config.vehicle_classes.clear();
      for (const auto& vehicle_class : ml["vehicle_classes"]) {
        c.ml_config.vehicle_classes.push_back(vehicle_class.as<int>());
      }
    }
  }

  // Load Output configuration
  if (y["output"]) {
    auto output = y["output"];

    // Logging settings
    if (output["logging"] && output["logging"]["verbose_logging"])
      c.output_config.verbose_logging = output["logging"]["verbose_logging"].as<bool>();
    if (output["logging"] && output["logging"]["performance_summary_interval"])
      c.output_config.performance_summary_interval =
          output["logging"]["performance_summary_interval"].as<int>();

    // CSV logging settings
    if (output["csv"]) {
      auto csv = output["csv"];
      if (csv["enable_csv_logging"])
        c.output_config.enable_csv_logging = csv["enable_csv_logging"].as<bool>();
      if (csv["csv_output_path"])
        c.output_config.csv_output_path = csv["csv_output_path"].as<std::string>();
      if (csv["csv_comprehensive_mode"])
        c.output_config.csv_comprehensive_mode = csv["csv_comprehensive_mode"].as<bool>();
    }

    // Video output settings
    if (output["enable_video_output"])
      c.output_config.enable_video_output = output["enable_video_output"].as<bool>();
    if (output["output_video_path"])
      c.output_config.output_video_path = output["output_video_path"].as<std::string>();
    if (output["video_codec"])
      c.output_config.video_codec = output["video_codec"].as<std::string>();
    if (output["output_fps"]) c.output_config.output_fps = output["output_fps"].as<int>();

    // Memory buffering settings
    if (output["use_memory_buffering"])
      c.output_config.use_memory_buffering = output["use_memory_buffering"].as<bool>();
    if (output["max_buffered_frames"])
      c.output_config.max_buffered_frames = output["max_buffered_frames"].as<size_t>();
    if (output["buffer_flush_threshold"])
      c.output_config.buffer_flush_threshold = output["buffer_flush_threshold"].as<size_t>();

    // Overlay settings
    if (output["enable_overlay"])
      c.output_config.enable_overlay = output["enable_overlay"].as<bool>();
    if (output["show_vehicle_boxes"])
      c.output_config.show_vehicle_boxes = output["show_vehicle_boxes"].as<bool>();
    if (output["show_tracking_ids"])
      c.output_config.show_tracking_ids = output["show_tracking_ids"].as<bool>();
    if (output["show_analytics_panel"])
      c.output_config.show_analytics_panel = output["show_analytics_panel"].as<bool>();
    if (output["show_collision_warnings"])
      c.output_config.show_collision_warnings = output["show_collision_warnings"].as<bool>();
    if (output["show_lane_detection"])
      c.output_config.show_lane_detection = output["show_lane_detection"].as<bool>();
    if (output["overlay_opacity"])
      c.output_config.overlay_opacity = output["overlay_opacity"].as<float>();
  }

  return c;
}
