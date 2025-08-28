#pragma once
#include <cstdint>
#include <chrono>
#include <string>

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

struct Frame {
  uint64_t id{};
  TimePoint t_capture{};
};

struct StageTimings {
  double pre_ms{0}, infer_ms{0}, post_ms{0}, e2e_ms{0};
  bool missed{false};
};

struct DeadlineProfile {
  int    target_fps{30};
  double budget_ms{33.0};
  double switch_up_p95_ms{28.0};
  double switch_down_p95_ms{22.0};
  int    hysteresis_frames{90};
};

enum class ModelPrecision { FP16, INT8 };

struct PrecisionPlan {
  ModelPrecision precision{ModelPrecision::FP16};
  std::string reason;
};

struct StreamPlan {
  int preprocess_priority{-1};
  int infer_priority{-2};
};
