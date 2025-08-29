#pragma once
#include <string>

#include "pipeline.hpp"
#include "types.hpp"

struct AppConfig {
  PipelineConfig pipeline;
  DeadlineProfile deadline;
  int metrics_port{9090};
};

AppConfig load_config(const std::string& path);
