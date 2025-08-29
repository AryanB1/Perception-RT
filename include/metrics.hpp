#pragma once
#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

class RollingHist {
public:
  explicit RollingHist(size_t cap = 512) : cap_(cap) {}
  void add(double x) {
    std::lock_guard<std::mutex> g(mu_);
    if (vals_.size() == cap_) vals_.pop_front();
    vals_.push_back(x);
  }
  // Percentile p in [0,100]
  double perc(double p) const {
    std::lock_guard<std::mutex> g(mu_);
    if (vals_.empty()) return 0.0;
    std::vector<double> v(vals_.begin(), vals_.end());
    std::sort(v.begin(), v.end());
    double rank = (p / 100.0) * static_cast<double>(v.size() - 1);
    size_t lo = static_cast<size_t>(rank);
    size_t hi = std::min(v.size() - 1, lo + 1);
    double frac = rank - static_cast<double>(lo);
    return v[lo] + (v[hi] - v[lo]) * frac;
  }
  size_t size() const {
    std::lock_guard<std::mutex> g(mu_);
    return vals_.size();
  }

private:
  size_t cap_;
  mutable std::mutex mu_;
  std::deque<double> vals_;
};

struct StatSnapshot {
  double pre_p50{0}, pre_p95{0}, pre_p99{0};
  double inf_p50{0}, inf_p95{0}, inf_p99{0};
  double post_p50{0}, post_p95{0}, post_p99{0};
  double e2e_p50{0}, e2e_p95{0}, e2e_p99{0};
  double miss_rate{0};
  double fps{0};
};

class MetricsRegistry {
public:
  void add_pre(double ms) { pre_.add(ms); }
  void add_inf(double ms) { inf_.add(ms); }
  void add_post(double ms) { post_.add(ms); }
  void add_e2e(double ms) { e2e_.add(ms); }

  void inc_frame() { frames_total_.fetch_add(1, std::memory_order_relaxed); }
  void inc_miss() { deadline_miss_total_.fetch_add(1, std::memory_order_relaxed); }

  uint64_t frames_total() const { return frames_total_.load(std::memory_order_relaxed); }
  uint64_t deadline_miss_total() const {
    return deadline_miss_total_.load(std::memory_order_relaxed);
  }

  StatSnapshot snapshot(double /*window_secs*/) const;
  std::string prometheus_text(const StatSnapshot& s) const;

private:
  RollingHist pre_, inf_, post_, e2e_;
  std::atomic<uint64_t> frames_total_{0};
  std::atomic<uint64_t> deadline_miss_total_{0};
};
