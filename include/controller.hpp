#pragma once
#include "metrics.hpp"
#include "types.hpp"

class Controller {
public:
  explicit Controller(DeadlineProfile dl) : dl_(dl) {}
  PrecisionPlan decide_precision(const StatSnapshot& s);
  StreamPlan decide_streams(const StatSnapshot& s);

private:
  DeadlineProfile dl_;
  int under_count_{0};
  int over_count_{0};
};
