#include "metrics.hpp"
#include <sstream>

StatSnapshot MetricsRegistry::snapshot(double /*window_secs*/) const {
  StatSnapshot s{};
  s.pre_p50 = pre_.perc(50);  s.pre_p95 = pre_.perc(95);  s.pre_p99 = pre_.perc(99);
  s.inf_p50 = inf_.perc(50);  s.inf_p95 = inf_.perc(95);  s.inf_p99 = inf_.perc(99);
  s.post_p50 = post_.perc(50); s.post_p95 = post_.perc(95); s.post_p99 = post_.perc(99);
  s.e2e_p50 = e2e_.perc(50);  s.e2e_p95 = e2e_.perc(95);  s.e2e_p99 = e2e_.perc(99);
  const auto frames = frames_total_.load();
  const auto misses = deadline_miss_total_.load();
  s.miss_rate = frames ? (static_cast<double>(misses) / static_cast<double>(frames)) : 0.0;
  return s;
}

std::string MetricsRegistry::prometheus_text(const StatSnapshot& s) const {
  std::ostringstream os;
  os << "frame_e2e_ms{quantile=\"0.5\"} "  << s.e2e_p50 << "\n";
  os << "frame_e2e_ms{quantile=\"0.95\"} " << s.e2e_p95 << "\n";
  os << "frame_e2e_ms{quantile=\"0.99\"} " << s.e2e_p99 << "\n";

  os << "deadline_miss_total " << deadline_miss_total_.load() << "\n";

  os << "deadline_miss_rate " << s.miss_rate << "\n";
  return os.str();
}
