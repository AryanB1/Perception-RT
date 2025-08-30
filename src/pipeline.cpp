#include "pipeline.hpp"

#include <spdlog/spdlog.h>

#include <chrono>
#include <thread>

#include "ml_engine.hpp"
#include "motion_detector.hpp"
#include "types.hpp"
#include "vehicle_analytics.hpp"

using namespace std::chrono;

#include <opencv2/opencv.hpp>

#include "gpu.hpp"

Pipeline::Pipeline(PipelineConfig cfg, DeadlineProfile dl, MetricsRegistry& m,
                   const MLConfig& ml_cfg, const OutputConfig& output_cfg)
    : cfg_(std::move(cfg)), dl_(dl), metrics_(m) {
  // Initialize ML Engine
  ml_engine_ = createMLEngine(ml_cfg);
  if (!ml_engine_->initialize()) {
    spdlog::error("Failed to initialize ML Engine");
    throw std::runtime_error("ML Engine initialization failed");
  }
  spdlog::info("ML Engine initialized successfully");

  // Initialize Output Manager
  output_manager_ = std::make_unique<OutputManager>(output_cfg);
}

bool Pipeline::open() {
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
}

void Pipeline::start() {
  if (running_.exchange(true)) return;
  loop_thread_ = std::thread([this] {
    const double budget = dl_.budget_ms;
    const double period_ms = 1000.0 / static_cast<double>(std::max(1, dl_.target_fps));

    int frames_in_window = 0;
    auto window_start = Clock::now();

    cv::VideoCapture cap;
    if (cfg_.uri == "0") {
      if (!cap.open(0)) {
        spdlog::error("Cannot open webcam in start().");
        running_ = false;
        return;
      }
    } else {
      if (!cap.open(cfg_.uri)) {
        spdlog::error("Cannot open '{}' in start().", cfg_.uri);
        running_ = false;
        return;
      }
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg_.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg_.height);

    spdlog::info("Starting pipeline with CUDA/GPU processing path");
    GpuContext g;
    gpu_init(g, /*W*/ 640, /*H*/ 640, /*C*/ 3, /*iters*/ 32);

    // Initialize output manager with frame dimensions
    if (!output_manager_->initialize(cfg_.width, cfg_.height)) {
      spdlog::error("Failed to initialize output manager");
      running_ = false;
      return;
    }

    std::array<TimePoint, 2> t_cap{};
    bool first_launched = false;

    uint64_t frame_id = 0;
    while (running_) {
      auto host_loop_t0 = Clock::now();

      const unsigned char* rgb_ptr = nullptr;
      cv::Mat raw;
      cap >> raw;
      if (raw.empty()) {
        spdlog::info("End of video stream reached after {} frames", frame_id);
        break;  // Exit loop instead of continuing
      }
      cv::Mat resized, rgb;
      cv::resize(raw, resized, cv::Size(g.W, g.H));
      cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
      rgb_ptr = rgb.ptr<unsigned char>(0);
      const int cur = static_cast<int>(frame_id & 1);
      const int prev = static_cast<int>((frame_id + 1) & 1);
      t_cap[cur] = host_loop_t0;

      gpu_stage_h2d_async(g, cur, rgb_ptr);

      gpu_launch_infer_async(g, cur);

      if (first_launched) {
        gpu_wait_infer_done(g, prev);

        float pre_ms = gpu_elapsed_ms(g.pre_start[prev], g.pre_end[prev]);
        float inf_ms = gpu_elapsed_ms(g.inf_start[prev], g.inf_end[prev]);
        float d2h_ms = gpu_elapsed_ms(g.d2h_start[prev], g.d2h_end[prev]);

        // Calculate e2e using GPU event timing from start to completion
        float gpu_total_ms = gpu_elapsed_ms(g.pre_start[prev], g.d2h_end[prev]);
        double e2e_ms = static_cast<double>(gpu_total_ms);

        metrics_.add_pre(pre_ms);
        metrics_.add_inf(inf_ms);
        metrics_.add_post(d2h_ms);
        metrics_.add_e2e(e2e_ms);
        metrics_.inc_frame();

        bool missed = (e2e_ms > budget);
        if (missed) metrics_.inc_miss();

        frames_in_window++;
        auto now = Clock::now();
        double win_secs = duration<double>(now - window_start).count();
        if (win_secs >= 1.0) {
          StatSnapshot snap = metrics_.snapshot(1.0);
          snap.fps = frames_in_window / win_secs;
          frames_in_window = 0;
          window_start = now;
          std::lock_guard<std::mutex> gmu(stat_mu_);
          last_stats_ = snap;
        }

        // Get motion detection results from GPU
        GpuMotionResult motion_result = gpu_get_motion_result(g, prev);

        // Run ML inference (YOLO) on the current frame
        MLResult ml_result;
        if (ml_engine_) {
          // Convert GPU frame back to OpenCV Mat for ML processing
          cv::Mat cv_frame(g.H, g.W, CV_8UC3);
          // Note: In a real implementation, you'd need to copy the frame data from GPU
          // For now, we'll use the original resized frame
          cv::Mat ml_resized, bgr_frame;
          cv::resize(raw, ml_resized, cv::Size(g.W, g.H));
          cv::cvtColor(ml_resized, bgr_frame, cv::COLOR_BGR2RGB);
          cv::cvtColor(bgr_frame, cv_frame, cv::COLOR_RGB2BGR);

          ml_result = ml_engine_->process(cv_frame);
        }

        // Use OutputManager instead of direct logging
        output_manager_->processFrame(raw,  // Use original frame for output
                                      ml_result, motion_result, frame_id - 1, pre_ms, inf_ms,
                                      d2h_ms, e2e_ms, missed);
      } else {
        first_launched = true;
      }

      frame_id++;

      double elapsed = duration<double, std::milli>(Clock::now() - host_loop_t0).count();
      double to_sleep = period_ms - elapsed;
      if (to_sleep > 0) std::this_thread::sleep_for(std::chrono::milliseconds((int)to_sleep));
    }

    if (first_launched) {
      const int last = static_cast<int>((frame_id + 1) & 1);
      gpu_wait_infer_done(g, last);
      float pre_ms = gpu_elapsed_ms(g.pre_start[last], g.pre_end[last]);
      float inf_ms = gpu_elapsed_ms(g.inf_start[last], g.inf_end[last]);
      float d2h_ms = gpu_elapsed_ms(g.d2h_start[last], g.d2h_end[last]);

      metrics_.add_pre(pre_ms);
      metrics_.add_inf(inf_ms);
      metrics_.add_post(d2h_ms);
    }

    // Ensure proper cleanup of output manager
    spdlog::info("Pipeline processing completed. Finalizing outputs...");
    output_manager_->cleanup();

    gpu_destroy(g);
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

bool Pipeline::next_frame_step() { return true; }
bool Pipeline::preprocess_any(void*, void*) { return true; }
bool Pipeline::inference_any(void*, void*) { return true; }
bool Pipeline::postprocess_any(void*, void*) { return true; }
