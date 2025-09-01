#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "metrics.hpp"
#include "ml_engine.hpp"
#include "output_manager.hpp"
#include "types.hpp"

struct PipelineConfig {
  std::string uri{"data/sample_1080p30.mp4"};
  int width{1920};
  int height{1080};
  int fps{30};
  int queue_capacity{4};
  std::string drop_policy{"drop_oldest"};
};

class Pipeline {
public:
  Pipeline(PipelineConfig cfg, DeadlineProfile dl, MetricsRegistry& m, const MLConfig& ml_cfg,
           const OutputConfig& output_cfg);
  bool open();   // Validate input availability
  void start();  // Start processing loop in a background thread
  void stop();   // Stop and join thread
  bool running() const { return running_.load(); }
  StatSnapshot stats() const;

private:
  PipelineConfig cfg_;
  DeadlineProfile dl_;
  MetricsRegistry& metrics_;
  std::unique_ptr<MLEngine> ml_engine_;
  std::unique_ptr<OutputManager> output_manager_;

  mutable std::mutex stat_mu_;
  StatSnapshot last_stats_{};

  std::atomic<bool> running_{false};
  std::thread loop_thread_;
};
