#include <httplib.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <atomic>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <thread>

#include "controller.hpp"
#include "metrics.hpp"
#include "pipeline.hpp"
#include "util.hpp"

int main(int argc, char** argv) {
  CLI::App cli_app{"FrameKeeper-RT: Real-time video motion detection and ML inference system"};

  std::string cfg_path = "configs/config.yaml";
  cli_app.add_option("-c,--config", cfg_path, "Configuration file path")->check(CLI::ExistingFile);

  bool show_version = false;
  cli_app.add_flag("-v,--version", show_version, "Show version information");

  try {
    cli_app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return cli_app.exit(e);
  }

  if (show_version) {
    std::cout << "FrameKeeper-RT v1.0.0" << std::endl;
    std::cout << "Real-time video processing with CUDA acceleration" << std::endl;
    std::cout << "Features: YOLOv11, Optical Flow, Semantic Segmentation, TensorRT" << std::endl;
    return 0;
  }

  spdlog::set_pattern("[%H:%M:%S.%e] %^[%l]%$ %v");
  spdlog::info("FrameKeeper-RT starting (config: {})", cfg_path);

  AppConfig app = load_config(cfg_path);
  MetricsRegistry metrics;
  Pipeline pipe(app.pipeline, app.deadline, metrics);

  if (!pipe.open()) {
    spdlog::error("Failed to open input. Exiting.");
    return 1;
  }

  std::atomic<bool> ready{false};
  std::atomic<bool> running{false};

  httplib::Server svr;

  svr.Get("/healthz", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content("{\"status\":\"ok\"}", "application/json");
  });

  svr.Get("/readyz", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(std::string("{\"ready\":") + (ready ? "true" : "false") + "}",
                    "application/json");
  });

  svr.Post("/pipeline/start", [&](const httplib::Request&, httplib::Response& res) {
    if (!running.exchange(true)) {
      pipe.start();
      ready = true;
    }
    res.set_content("{\"started\":true}", "application/json");
  });

  svr.Post("/pipeline/stop", [&](const httplib::Request&, httplib::Response& res) {
    if (running.exchange(false)) {
      pipe.stop();
      ready = false;
    }
    res.set_content("{\"stopped\":true}", "application/json");
  });

  svr.Get("/pipeline/stats", [&](const httplib::Request&, httplib::Response& res) {
    auto s = pipe.stats();
    nlohmann::json j{{"fps", s.fps},
                     {"e2e_p50", s.e2e_p50},
                     {"e2e_p95", s.e2e_p95},
                     {"e2e_p99", s.e2e_p99},
                     {"miss_rate", s.miss_rate}};
    res.set_content(j.dump(2), "application/json");
  });

  svr.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
    auto s = pipe.stats();
    res.set_content(metrics.prometheus_text(s), "text/plain; version=0.0.4");
  });

  pipe.start();
  running = true;
  ready = true;

  spdlog::info("HTTP server listening on 0.0.0.0:{}", app.metrics_port);
  svr.listen("0.0.0.0", app.metrics_port);

  // Cleanup
  if (running) pipe.stop();
  spdlog::info("Shutdown complete.");
  return 0;
}
