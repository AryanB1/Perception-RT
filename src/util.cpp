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

  return c;
}
