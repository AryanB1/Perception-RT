#pragma once
#include <string>

#include "ml_engine.hpp"
#include "output_manager.hpp"
#include "pipeline.hpp"
#include "types.hpp"

struct AppConfig {
  PipelineConfig pipeline;
  DeadlineProfile deadline;
  MLConfig ml_config;
  OutputConfig output_config;
  int metrics_port{9090};
};

AppConfig load_config(const std::string& path);
