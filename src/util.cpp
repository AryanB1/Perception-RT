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
    if (ml["yolo_model_path"]) c.ml_config.yolo_model_path = ml["yolo_model_path"].as<std::string>();
    if (ml["yolo_engine_path"]) c.ml_config.yolo_engine_path = ml["yolo_engine_path"].as<std::string>();
    if (ml["segmentation_model_path"]) c.ml_config.segmentation_model_path = ml["segmentation_model_path"].as<std::string>();
    
    if (ml["use_tensorrt"]) c.ml_config.use_tensorrt = ml["use_tensorrt"].as<bool>();
    if (ml["use_fp16"]) c.ml_config.use_fp16 = ml["use_fp16"].as<bool>();
    if (ml["max_batch_size"]) c.ml_config.max_batch_size = ml["max_batch_size"].as<int>();
    if (ml["max_workspace_size"]) c.ml_config.max_workspace_size = ml["max_workspace_size"].as<size_t>();
    
    if (ml["detection_threshold"]) c.ml_config.detection_threshold = ml["detection_threshold"].as<float>();
    if (ml["nms_threshold"]) c.ml_config.nms_threshold = ml["nms_threshold"].as<float>();
    if (ml["max_detections"]) c.ml_config.max_detections = ml["max_detections"].as<int>();
    
    if (ml["max_corners"]) c.ml_config.max_corners = ml["max_corners"].as<int>();
    if (ml["optical_flow_threshold"]) c.ml_config.optical_flow_threshold = ml["optical_flow_threshold"].as<float>();
    
    if (ml["segmentation_threshold"]) c.ml_config.segmentation_threshold = ml["segmentation_threshold"].as<float>();
    
    if (ml["input_width"] && ml["input_height"]) {
      c.ml_config.input_size = cv::Size(ml["input_width"].as<int>(), ml["input_height"].as<int>());
    }
    
    if (ml["enable_detection"]) c.ml_config.enable_detection = ml["enable_detection"].as<bool>();
    if (ml["enable_optical_flow"]) c.ml_config.enable_optical_flow = ml["enable_optical_flow"].as<bool>();
    if (ml["enable_segmentation"]) c.ml_config.enable_segmentation = ml["enable_segmentation"].as<bool>();
  }

  return c;
}
