#include "controller.hpp"

PrecisionPlan Controller::decide_precision(const StatSnapshot& s) {
  if (s.e2e_p95 > dl_.switch_up_p95_ms) {
    over_count_++;
    under_count_ = 0;
  } else if (s.e2e_p95 < dl_.switch_down_p95_ms) {
    under_count_++;
    over_count_ = 0;
  } else {
    if (over_count_ > 0)  over_count_--;
    if (under_count_ > 0) under_count_--;
  }

  if (over_count_ >= dl_.hysteresis_frames) {
    return {ModelPrecision::INT8, "p95 above threshold"};
  }
  if (under_count_ >= dl_.hysteresis_frames) {
    return {ModelPrecision::FP16, "p95 below threshold"};
  }
  return {ModelPrecision::FP16, "no-change"};
}

StreamPlan Controller::decide_streams(const StatSnapshot& /*s*/) {
  return {-1, -2};
}
