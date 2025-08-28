#include "pipeline.hpp"
#include "types.hpp"
#include <spdlog/spdlog.h>
#include <chrono>
#include <thread>

using namespace std::chrono;

#if defined(HAVE_OPENCV)
  #include <opencv2/opencv.hpp>
  static bool read_frame_cv(cv::VideoCapture& cap, cv::Mat& frame) {
    cap >> frame;
    return !frame.empty();
  }
#endif

Pipeline::Pipeline(PipelineConfig cfg, DeadlineProfile dl, MetricsRegistry& m)
  : cfg_(std::move(cfg)), dl_(dl), metrics_(m) {}

bool Pipeline::open() {
#if defined(HAVE_OPENCV)
  if (cfg_.uri.empty()) return false;
  cv::VideoCapture cap;
  if (cfg_.uri == "0") {
    if (!cap.open(0)) {
      spdlog::error("Failed to open webcam (0).");
      return false;
    }
  } else if (!cap.open(cfg_.uri)) {
    spdlog::error("Failed to open input: {}", cfg_.uri);
    return false;
  }
  spdlog::info("Probed input '{}' ({}x{} @ ~{} fps)", cfg_.uri, cfg_.width, cfg_.height, cfg_.fps);
  return true;
#else
  spdlog::warn("OpenCV not available; Pipeline::open() cannot validate inputs.");
  return false;
#endif
}

void Pipeline::start() {
  if (running_.exchange(true)) return; // already running
  loop_thread_ = std::thread([this]{
    uint64_t frame_id = 0;
    const double budget = dl_.budget_ms;
    const double period_ms = 1000.0 / static_cast<double>(std::max(1, dl_.target_fps));

    int frames_in_window = 0;
    auto window_start = Clock::now();

#if defined(HAVE_OPENCV)
    cv::VideoCapture cap;
    if (cfg_.uri == "0") {
      if (!cap.open(0)) { spdlog::error("Cannot open webcam in start()."); running_ = false; return; }
    } else {
      if (!cap.open(cfg_.uri)) { spdlog::error("Cannot open '{}' in start().", cfg_.uri); running_ = false; return; }
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg_.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg_.height);
#endif

    while (running_) {
      auto t0 = Clock::now();

#if defined(HAVE_OPENCV)
      cv::Mat img;
      if (!read_frame_cv(cap, img)) {
        spdlog::warn("End of stream or empty frame.");
        std::this_thread::sleep_for(50ms);
        continue;
      }
      Frame f{.id = frame_id++, .t_capture = t0};

      auto t_pre0 = Clock::now();
      // Baseline "preprocess": resize + BGR->RGB (simulate ~2 ms)
      cv::Mat pre;
      cv::resize(img, pre, cv::Size(640, 640));
      cv::cvtColor(pre, pre, cv::COLOR_BGR2RGB);
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      auto t_pre1 = Clock::now();

      auto t_inf0 = Clock::now();
      // Placeholder "inference": copy + sleep (~8 ms)
      cv::Mat inf = pre.clone();
      (void)inf;
      std::this_thread::sleep_for(std::chrono::milliseconds(8));
      auto t_inf1 = Clock::now();

      auto t_post0 = Clock::now();
      // Placeholder "post": draw overlay (~2 ms)
      cv::Mat post = pre.clone();
      cv::rectangle(post, {10,10}, {100,100}, {255,0,0}, 2);
      cv::putText(post, "FrameKeeper-RT", {12, 35}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,255,255}, 2);
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      auto t_post1 = Clock::now();

      double pre_ms  = duration<double, std::milli>(t_pre1 - t_pre0).count();
      double inf_ms  = duration<double, std::milli>(t_inf1 - t_inf0).count();
      double post_ms = duration<double, std::milli>(t_post1 - t_post0).count();
      double e2e_ms  = duration<double, std::milli>(t_post1 - t0).count();

      metrics_.add_pre(pre_ms);
      metrics_.add_inf(inf_ms);
      metrics_.add_post(post_ms);
      metrics_.add_e2e(e2e_ms);
      metrics_.inc_frame();

      bool missed = (e2e_ms > budget);
      if (missed) metrics_.inc_miss();

      // rolling FPS
      frames_in_window++;
      auto now = Clock::now();
      double win_secs = duration<double>(now - window_start).count();
      if (win_secs >= 1.0) {
        StatSnapshot snap = metrics_.snapshot(1.0);
        snap.fps = frames_in_window / win_secs;
        frames_in_window = 0; window_start = now;
        std::lock_guard<std::mutex> g(stat_mu_);
        last_stats_ = snap;
      }

      // Try to maintain target period (best effort)
      double elapsed = duration<double, std::milli>(Clock::now() - t0).count();
      double to_sleep = period_ms - elapsed;
      if (to_sleep > 0) std::this_thread::sleep_for(milliseconds((int)to_sleep));

      spdlog::info("frame_id={} pre_ms={:.3f} inf_ms={:.3f} post_ms={:.3f} e2e_ms={:.3f} missed={}",
                   f.id, pre_ms, inf_ms, post_ms, e2e_ms, missed);
#else
      // No OpenCV build: idle loop so service stays alive if you ever want to add other inputs.
      std::this_thread::sleep_for(200ms);
#endif
    }
  });
}

void Pipeline::stop() {
  if (!running_.exchange(false)) return;
  if (loop_thread_.joinable()) loop_thread_.join();
}

StatSnapshot Pipeline::stats() const {
  std::lock_guard<std::mutex> g(stat_mu_);
  return last_stats_;
}

bool Pipeline::preprocess_any(void*, void*) { return true; }
bool Pipeline::inference_any (void*, void*) { return true; }
bool Pipeline::postprocess_any(void*, void*) { return true; }
