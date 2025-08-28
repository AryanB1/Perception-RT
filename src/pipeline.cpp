#include "pipeline.hpp"
#include "types.hpp"
#include "motion_detector.hpp"
#include "ml_engine.hpp"
#include <spdlog/spdlog.h>
#include <chrono>
#include <thread>

using namespace std::chrono;

#if defined(HAVE_OPENCV)
  #include <opencv2/opencv.hpp>
#endif

#if !defined(NO_CUDA)
  #include "gpu.hpp"
#endif

Pipeline::Pipeline(PipelineConfig cfg, DeadlineProfile dl, MetricsRegistry& m)
  : cfg_(std::move(cfg)), dl_(dl), metrics_(m) {}

bool Pipeline::open() {
#if defined(HAVE_OPENCV)
  if (cfg_.uri.empty()) return false;
  cv::VideoCapture cap;
  if (cfg_.uri == "0") {
    if (!cap.open(0)) { spdlog::error("Failed to open webcam (0)."); return false; }
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
  if (running_.exchange(true)) return;
  loop_thread_ = std::thread([this]{
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

#if !defined(NO_CUDA)
    spdlog::info("Starting pipeline with CUDA/GPU processing path");
    GpuContext g;
    gpu_init(g, /*W*/640, /*H*/640, /*C*/3, /*iters*/32);
    std::array<TimePoint,2> t_cap{};
    bool first_launched = false;

    uint64_t frame_id = 0;
    while (running_) {
      auto host_loop_t0 = Clock::now();

      const unsigned char* rgb_ptr = nullptr;
#if defined(HAVE_OPENCV)
      cv::Mat raw;
      cap >> raw;
      if (raw.empty()) {
        spdlog::warn("End of stream or empty frame.");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }
      cv::Mat resized, rgb;
      cv::resize(raw, resized, cv::Size(g.W, g.H));
      cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
      rgb_ptr = rgb.ptr<unsigned char>(0);
#else
      // If no OpenCV, just idle
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
#endif

      const int cur = static_cast<int>(frame_id & 1);
      const int prev = static_cast<int>((frame_id + 1) & 1);
      t_cap[cur] = host_loop_t0;

      gpu_stage_h2d_async(g, cur, rgb_ptr);

      gpu_launch_infer_async(g, cur);

      if (first_launched) {
        gpu_wait_infer_done(g, prev);

        float pre_ms  = gpu_elapsed_ms(g.pre_start[prev], g.pre_end[prev]);
        float inf_ms  = gpu_elapsed_ms(g.inf_start[prev], g.inf_end[prev]);
        float d2h_ms  = gpu_elapsed_ms(g.d2h_start[prev], g.d2h_end[prev]);

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
          frames_in_window = 0; window_start = now;
          std::lock_guard<std::mutex> gmu(stat_mu_);
          last_stats_ = snap;
        }

        // Get motion detection results from GPU
        GpuMotionResult motion_result = gpu_get_motion_result(g, prev);

        spdlog::info("frame_id={} [CUDA] pre_ms={:.3f} inf_ms={:.3f} post(D2H)_ms={:.3f} e2e_ms={:.3f} missed={} motion_detected={} motion_intensity={:.4f} motion_pixels={}",
                     frame_id - 1, pre_ms, inf_ms, d2h_ms, e2e_ms, missed, motion_result.motion_detected, motion_result.motion_intensity, motion_result.motion_pixels);
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
      float pre_ms  = gpu_elapsed_ms(g.pre_start[last], g.pre_end[last]);
      float inf_ms  = gpu_elapsed_ms(g.inf_start[last], g.inf_end[last]);
      float d2h_ms  = gpu_elapsed_ms(g.d2h_start[last], g.d2h_end[last]);
      double e2e_ms = 0.0;

      metrics_.add_pre(pre_ms);
      metrics_.add_inf(inf_ms);
      metrics_.add_post(d2h_ms);
    }

    gpu_destroy(g);

#else
    spdlog::info("Starting pipeline with CPU processing path (NO_CUDA defined)");
    uint64_t frame_id = 0;

#if defined(HAVE_OPENCV)
    // Initialize motion detector for CPU path
    MotionDetector motion_detector(MotionDetector::Algorithm::BACKGROUND_SUB, 640, 640);
    motion_detector.set_threshold(25.0);
    motion_detector.set_min_contour_area(500.0);
    
    // Initialize ML Engine for advanced inference
    MLConfig ml_config;
    ml_config.enable_detection = true;
    ml_config.enable_optical_flow = true;
    ml_config.enable_segmentation = false; // Disable for performance
    ml_config.use_tensorrt = true;
    ml_config.input_size = cv::Size(640, 640);
    ml_config.detection_threshold = 0.5f;
    ml_config.nms_threshold = 0.4f;
    
    auto ml_engine = createMLEngine(ml_config);
    if (!ml_engine->initialize()) {
        spdlog::warn("Failed to initialize ML Engine, falling back to simple motion detection");
        ml_engine.reset();
    }
#endif

    while (running_) {
      auto t0 = Clock::now();
#if defined(HAVE_OPENCV)
      cv::Mat img; cap >> img;
      if (img.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }
      
      // Preprocessing: resize and convert to RGB
      auto t_pre0 = Clock::now();
      cv::Mat pre;  cv::resize(img, pre, cv::Size(640,640)); cv::cvtColor(pre, pre, cv::COLOR_BGR2RGB);
      auto t_pre1 = Clock::now();

      // Motion detection inference
      auto t_inf0 = Clock::now();
      MLResult ml_result;
      MotionResult motion_result;
      
      if (ml_engine) {
        // Use advanced ML inference
        ml_result = ml_engine->process(pre);
        
        // Convert ML result to motion result for compatibility
        motion_result.motion_detected = ml_result.motion_detected;
        motion_result.motion_intensity = ml_result.motion_intensity;
        motion_result.motion_pixels = ml_result.motion_pixels;
        motion_result.bounding_box = ml_result.motion_bbox;
      } else {
        // Fallback to simple motion detection
        motion_result = motion_detector.process_frame(pre);
      }
      auto t_inf1 = Clock::now();

      // Postprocessing: visualize detection results
      auto t_post0 = Clock::now();
      cv::Mat post = pre.clone();
      
      if (ml_engine && ml_result.total_objects > 0) {
        // Draw advanced ML results
        post = MLViz::drawComprehensiveResults(post, ml_result);
        
        // Add ML-specific information
        std::string obj_text = "Objects: " + std::to_string(ml_result.total_objects);
        cv::putText(post, obj_text, {10, 90}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        std::string conf_text = "Max Conf: " + std::to_string(ml_result.max_confidence).substr(0, 4);
        cv::putText(post, conf_text, {10, 110}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        std::string flow_text = "Flow Pts: " + std::to_string(ml_result.optical_flow.moving_points);
        cv::putText(post, flow_text, {10, 130}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
      } else {
        // Draw simple motion detection results
        if (motion_result.motion_detected) {
          cv::rectangle(post, motion_result.bounding_box, cv::Scalar(0, 255, 0), 2);
          cv::putText(post, "MOTION DETECTED", {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        } else {
          cv::putText(post, "NO MOTION", {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        }
        
        // Display motion intensity and pixel count
        std::string intensity_text = "Intensity: " + std::to_string(motion_result.motion_intensity).substr(0, 5);
        std::string pixels_text = "Pixels: " + std::to_string(motion_result.motion_pixels);
        cv::putText(post, intensity_text, {10, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::putText(post, pixels_text, {10, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // Draw optical flow vectors if available
        for (const auto& vec : motion_result.motion_vectors) {
          cv::circle(post, cv::Point2f(vec.x, vec.y), 2, cv::Scalar(255, 255, 0), -1);
        }
      }
      
      auto t_post1 = Clock::now();

      double pre_ms  = duration<double, std::milli>(t_pre1 - t_pre0).count();
      double inf_ms  = duration<double, std::milli>(t_inf1 - t_inf0).count();
      double post_ms = duration<double, std::milli>(t_post1 - t_post0).count();
      double e2e_ms  = duration<double, std::milli>(t_post1 - t0).count();
#else
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      double pre_ms=0, inf_ms=0, post_ms=0, e2e_ms=0;
      MotionResult motion_result{}; // Empty result for non-OpenCV builds
#endif
      metrics_.add_pre(pre_ms);
      metrics_.add_inf(inf_ms);
      metrics_.add_post(post_ms);
      metrics_.add_e2e(e2e_ms);
      metrics_.inc_frame();
      bool missed = (e2e_ms > budget);
      if (missed) metrics_.inc_miss();

#if defined(HAVE_OPENCV)
      spdlog::info("frame_id={} [CPU] pre_ms={:.3f} inf_ms={:.3f} post_ms={:.3f} e2e_ms={:.3f} missed={} motion_detected={} motion_intensity={:.4f} motion_pixels={} objects={}",
                   frame_id, pre_ms, inf_ms, post_ms, e2e_ms, missed, motion_result.motion_detected, motion_result.motion_intensity, motion_result.motion_pixels, 
                   ml_engine ? ml_result.total_objects : 0);
#else
      spdlog::info("frame_id={} [CPU] pre_ms={:.3f} inf_ms={:.3f} post_ms={:.3f} e2e_ms={:.3f} missed={}",
                   frame_id, pre_ms, inf_ms, post_ms, e2e_ms, missed);
#endif

      frames_in_window++;
      auto now = Clock::now();
      double win_secs = duration<double>(now - window_start).count();
      if (win_secs >= 1.0) {
        StatSnapshot snap = metrics_.snapshot(1.0);
        snap.fps = frames_in_window / win_secs;
        frames_in_window = 0; window_start = now;
        std::lock_guard<std::mutex> gmu(stat_mu_);
        last_stats_ = snap;
      }

      double elapsed = duration<double, std::milli>(Clock::now() - t0).count();
      double to_sleep = period_ms - elapsed;
      if (to_sleep > 0) std::this_thread::sleep_for(std::chrono::milliseconds((int)to_sleep));
      frame_id++;
    }
#endif // NO_CUDA
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
bool Pipeline::inference_any (void*, void*) { return true; }
bool Pipeline::postprocess_any(void*, void*) { return true; }
