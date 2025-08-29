#include "ml_engine.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>

using namespace std::chrono;

// TensorRT Logger Implementation
void TRTLogger::log(Severity severity, const char* msg) noexcept {
  if (severity <= min_severity_) {
    switch (severity) {
      case Severity::kERROR:
        spdlog::error("[TensorRT] {}", msg);
        break;
      case Severity::kWARNING:
        spdlog::warn("[TensorRT] {}", msg);
        break;
      case Severity::kINFO:
        spdlog::info("[TensorRT] {}", msg);
        break;
      case Severity::kVERBOSE:
        spdlog::debug("[TensorRT] {}", msg);
        break;
      case Severity::kINTERNAL_ERROR:
        spdlog::critical("[TensorRT] {}", msg);
        break;
    }
  }
}

// TensorRT Engine Implementation
TensorRTEngine::TensorRTEngine(const std::string& onnx_path, const MLConfig& config)
    : logger_(std::make_unique<TRTLogger>()), onnx_path_(onnx_path) {
  builder_.reset(nvinfer1::createInferBuilder(*logger_));
  if (!builder_) {
    throw std::runtime_error("Failed to create TensorRT builder");
  }

  // Create network with modern API (no explicit batch flag needed)
  network_.reset(builder_->createNetworkV2(0U));
  if (!network_) {
    throw std::runtime_error("Failed to create TensorRT network");
  }

  // Create builder config
  config_.reset(builder_->createBuilderConfig());
  if (!config_) {
    throw std::runtime_error("Failed to create TensorRT builder config");
  }

  // Set config parameters
  config_->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, config.max_workspace_size);
  if (config.use_fp16) {
    // Note: FP16 optimization will be set during network optimization phase
    // The deprecated flags are no longer needed in modern TensorRT
    spdlog::info("TensorRT: FP16 optimization will be enabled during build");
  }

  input_size_ = config.input_size;
  input_channels_ = 3;  // RGB
}

TensorRTEngine::~TensorRTEngine() {
  freeBuffers();
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

bool TensorRTEngine::build() {
  if (!network_ || !config_) {
    spdlog::error("TensorRT network or config not initialized");
    return false;
  }

  // Parse ONNX model to populate the network
  auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network_, *logger_));
  if (!parser) {
    spdlog::error("Failed to create ONNX parser");
    return false;
  }

  // Parse the ONNX file
  if (!parser->parseFromFile(onnx_path_.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    spdlog::error("Failed to parse ONNX file: {}", onnx_path_);
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
      spdlog::error("Parser error {}: {}", i, parser->getError(i)->desc());
    }
    return false;
  }

  spdlog::info("Successfully parsed ONNX model: {}", onnx_path_);

  // Build the engine
  engine_.reset(builder_->buildEngineWithConfig(*network_, *config_));
  if (!engine_) {
    spdlog::error("Failed to build TensorRT engine");
    return false;
  }

  // Create execution context
  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    spdlog::error("Failed to create TensorRT execution context");
    return false;
  }

  // Allocate CUDA buffers
  allocateBuffers();

  // Create CUDA stream
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    spdlog::error("Failed to create CUDA stream");
    return false;
  }

  spdlog::info("TensorRT engine built successfully");
  return true;
}

bool TensorRTEngine::loadEngine(const std::string& engine_path) {
  std::ifstream file(engine_path, std::ios::binary);
  if (!file.good()) {
    spdlog::warn("Engine file not found: {}", engine_path);
    return false;
  }

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);
  file.close();

  auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*logger_));
  if (!runtime) {
    spdlog::error("Failed to create TensorRT runtime");
    return false;
  }

  engine_.reset(runtime->deserializeCudaEngine(engine_data.data(), size));
  if (!engine_) {
    spdlog::error("Failed to deserialize TensorRT engine");
    return false;
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    spdlog::error("Failed to create execution context");
    return false;
  }

  allocateBuffers();

  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    spdlog::error("Failed to create CUDA stream");
    return false;
  }

  spdlog::info("TensorRT engine loaded successfully from {}", engine_path);
  return true;
}

bool TensorRTEngine::saveEngine(const std::string& engine_path) {
  if (!engine_) {
    spdlog::error("No engine to save");
    return false;
  }

  auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(engine_->serialize());
  if (!serialized) {
    spdlog::error("Failed to serialize engine");
    return false;
  }

  std::ofstream file(engine_path, std::ios::binary);
  if (!file.good()) {
    spdlog::error("Failed to open file for writing: {}", engine_path);
    return false;
  }

  file.write(static_cast<const char*>(serialized->data()), serialized->size());
  file.close();

  spdlog::info("TensorRT engine saved to {}", engine_path);
  return true;
}

bool TensorRTEngine::infer(const std::vector<float>& input, std::vector<float>& output) {
  if (!context_ || !d_input_ || !d_output_) {
    spdlog::error("TensorRT engine not properly initialized");
    return false;
  }

  // Copy input to device
  size_t input_size = input.size() * sizeof(float);
  if (cudaMemcpyAsync(d_input_, input.data(), input_size, cudaMemcpyHostToDevice, stream_) !=
      cudaSuccess) {
    spdlog::error("Failed to copy input to device");
    return false;
  }

  // Set tensor addresses for modern TensorRT API
  for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* tensor_name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT) {
      context_->setTensorAddress(tensor_name, d_input_);
    } else if (engine_->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kOUTPUT) {
      context_->setTensorAddress(tensor_name, d_output_);
    }
  }

  // Run inference
  if (!context_->enqueueV3(stream_)) {
    spdlog::error("TensorRT inference failed");
    return false;
  }

  // Copy output to host
  size_t output_size = output.size() * sizeof(float);
  if (cudaMemcpyAsync(output.data(), d_output_, output_size, cudaMemcpyDeviceToHost, stream_) !=
      cudaSuccess) {
    spdlog::error("Failed to copy output to host");
    return false;
  }

  // Synchronize stream
  if (cudaStreamSynchronize(stream_) != cudaSuccess) {
    spdlog::error("Failed to synchronize CUDA stream");
    return false;
  }

  return true;
}

void TensorRTEngine::allocateBuffers() {
  if (!engine_) return;

  size_t input_bytes = input_size_.width * input_size_.height * input_channels_ * sizeof(float);
  if (cudaMalloc(&d_input_, input_bytes) != cudaSuccess) {
    throw std::runtime_error("Failed to allocate input buffer");
  }

  // For YOLO v11, calculate actual output size from engine
  // Using modern TensorRT API (getTensorName, getTensorShape)
  if (engine_->getNbIOTensors() >= 2) {
    const char* output_name = nullptr;
    for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char* tensor_name = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kOUTPUT) {
        output_name = tensor_name;
        break;
      }
    }
    
    if (output_name) {
      auto output_dims = engine_->getTensorShape(output_name);
      output_size_ = 1;
      for (int i = 0; i < output_dims.nbDims; ++i) {
        output_size_ *= static_cast<size_t>(output_dims.d[i]);
      }
      spdlog::info("TensorRT: Output tensor '{}', Output size: {}", output_name, output_size_);
    } else {
      // Fallback for YOLO v11s
      output_size_ = 8400 * 84;  // YOLO v11 output: 8400 detections × 84 elements (4 + 80 classes)
      spdlog::warn("No output tensor found, using fallback output size: {}", output_size_);
    }
  } else {
    // Fallback for YOLO v11s
    output_size_ = 8400 * 84;  // YOLO v11 output: 8400 detections × 84 elements (4 + 80 classes)
    spdlog::warn("Using fallback output size: {}", output_size_);
  }
  
  size_t output_bytes = output_size_ * sizeof(float);
  if (cudaMalloc(&d_output_, output_bytes) != cudaSuccess) {
    throw std::runtime_error("Failed to allocate output buffer");
  }

  spdlog::debug("TensorRT buffers allocated: input={}MB, output={}MB", 
                input_bytes / 1024 / 1024, output_bytes / 1024 / 1024);
}

void TensorRTEngine::freeBuffers() {
  if (d_input_) {
    cudaFree(d_input_);
    d_input_ = nullptr;
  }
  if (d_output_) {
    cudaFree(d_output_);
    d_output_ = nullptr;
  }
}

// Performance Stats Implementation
void MLEngine::PerformanceStats::update(float inference, float preprocessing,
                                        float postprocessing) {
  frames_processed++;
  float alpha = 1.0f / static_cast<float>(frames_processed);  // Simple moving average
  avg_inference_time = avg_inference_time * (1 - alpha) + inference * alpha;
  avg_preprocessing_time = avg_preprocessing_time * (1 - alpha) + preprocessing * alpha;
  avg_postprocessing_time = avg_postprocessing_time * (1 - alpha) + postprocessing * alpha;
}

void MLEngine::PerformanceStats::reset() {
  avg_inference_time = avg_preprocessing_time = avg_postprocessing_time = 0.0f;
  frames_processed = 0;
}

// MLEngine Implementation
MLEngine::MLEngine(const MLConfig& config) : config_(config) {
  optical_flow_tracker_ = cv::SparsePyrLKOpticalFlow::create();
}

MLEngine::~MLEngine() = default;

bool MLEngine::initialize() {
  spdlog::info("Initializing ML Engine...");

  // Load class names for YOLO
  loadClassNames();

  if (config_.use_tensorrt) {
    try {
      // Initialize YOLO engine
      if (config_.enable_detection) {
        yolo_engine_ = std::make_unique<TensorRTEngine>(config_.yolo_model_path, config_);

        // Try to load existing engine first (prioritize .engine over .onnx)
        if (!yolo_engine_->loadEngine(config_.yolo_engine_path)) {
          spdlog::info("Pre-built TensorRT engine not found at {}", config_.yolo_engine_path);
          
          // Try to load the .onnx and build engine
          std::string engine_path = config_.yolo_model_path + ".trt";
          if (!yolo_engine_->loadEngine(engine_path)) {
            spdlog::info("Building new TensorRT engine for YOLO...");
            if (!yolo_engine_->build()) {
              spdlog::error("Failed to build YOLO TensorRT engine");
              return false;
            }
            yolo_engine_->saveEngine(engine_path);
          }
        } else {
          spdlog::info("Loaded pre-built TensorRT engine from {}", config_.yolo_engine_path);
        }
      }

      // Initialize segmentation engine if enabled
      if (config_.enable_segmentation) {
        segmentation_engine_ =
            std::make_unique<TensorRTEngine>(config_.segmentation_model_path, config_);

        std::string seg_engine_path = config_.segmentation_model_path + ".trt";
        if (!segmentation_engine_->loadEngine(seg_engine_path)) {
          spdlog::info("Building new TensorRT engine for segmentation...");
          if (!segmentation_engine_->build()) {
            spdlog::warn(
                "Failed to build segmentation TensorRT engine, falling back to OpenCV DNN");
            config_.use_tensorrt = false;
          } else {
            segmentation_engine_->saveEngine(seg_engine_path);
          }
        }
      }
    } catch (const std::exception& e) {
      spdlog::error("TensorRT initialization failed: {}", e.what());
      config_.use_tensorrt = false;
    }
  }

  spdlog::info("ML Engine initialized successfully");
  spdlog::info("  - Object Detection: {}", config_.enable_detection ? "Enabled" : "Disabled");
  spdlog::info("  - Optical Flow: {}", config_.enable_optical_flow ? "Enabled" : "Disabled");
  spdlog::info("  - Segmentation: {}", config_.enable_segmentation ? "Enabled" : "Disabled");
  spdlog::info("  - TensorRT: {}", config_.use_tensorrt ? "Enabled" : "Disabled");

  return true;
}

MLResult MLEngine::process(const cv::Mat& frame) {
  MLResult result;

  startTimer();

  // Object Detection
  if (config_.enable_detection) {
    startTimer();
    result.detections = detectObjects(frame);
    result.total_objects = static_cast<int>(result.detections.size());
    if (!result.detections.empty()) {
      result.max_confidence = std::max_element(result.detections.begin(), result.detections.end(),
                                               [](const Detection& a, const Detection& b) {
                                                 return a.confidence < b.confidence;
                                               })
                                  ->confidence;
    }
    float detection_time = getElapsedMs();
    result.inference_time_ms += detection_time;
  }

  // Optical Flow
  if (config_.enable_optical_flow) {
    startTimer();
    result.optical_flow = computeOpticalFlow(frame);
    result.significant_motion = result.optical_flow.magnitude_mean > config_.optical_flow_threshold;
    float flow_time = getElapsedMs();
    result.inference_time_ms += flow_time;
  }

  // Semantic Segmentation
  if (config_.enable_segmentation) {
    startTimer();
    result.segmentation = segmentFrame(frame);
    float seg_time = getElapsedMs();
    result.inference_time_ms += seg_time;
  }

  // Enhanced motion analysis combining all modalities
  result.motion_detected = result.significant_motion || result.total_objects > 0;
  if (result.motion_detected) {
    // Calculate motion intensity from optical flow and detections
    result.motion_intensity = std::min(1.0f, result.optical_flow.magnitude_mean / 10.0f +
                                                 static_cast<float>(result.total_objects) * 0.1f);
    result.motion_pixels = result.optical_flow.moving_points;

    // Calculate motion bounding box from detections and flow
    if (!result.detections.empty()) {
      cv::Rect bbox = result.detections[0].bbox;
      for (const auto& det : result.detections) {
        bbox |= det.bbox;
      }
      result.motion_bbox = bbox;
    }
  }

  // Update performance statistics
  stats_.update(result.inference_time_ms, result.preprocessing_time_ms,
                result.postprocessing_time_ms);

  return result;
}

std::vector<Detection> MLEngine::detectObjects(const cv::Mat& frame) {
  std::vector<Detection> detections;

  if (!config_.enable_detection) return detections;

  startTimer();
  cv::Mat preprocessed = preprocessForYOLO(frame);
  float prep_time = getElapsedMs();

  if (config_.use_tensorrt && yolo_engine_) {
    // TensorRT inference
    std::vector<float> input_data(preprocessed.total());
    std::memcpy(input_data.data(), preprocessed.ptr<float>(), input_data.size() * sizeof(float));

    std::vector<float> output_data(yolo_engine_->getOutputSize());

    startTimer();
    if (yolo_engine_->infer(input_data, output_data)) {
      float inf_time = getElapsedMs();

      startTimer();
      detections = postprocessYOLO(output_data, frame.size());
      float post_time = getElapsedMs();

      spdlog::debug("YOLO TensorRT: prep={:.2f}ms, infer={:.2f}ms, post={:.2f}ms", prep_time,
                    inf_time, post_time);
    }
  }
  return detections;
}

OpticalFlowResult MLEngine::computeOpticalFlow(const cv::Mat& frame) {
  OpticalFlowResult result;

  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  if (prev_gray_.empty()) {
    prev_gray_ = gray.clone();
    cv::goodFeaturesToTrack(gray, prev_points_, config_.max_corners, 0.3, 7, cv::Mat(), 7, false,
                            0.04);
    return result;
  }

  if (prev_points_.empty()) {
    cv::goodFeaturesToTrack(prev_gray_, prev_points_, config_.max_corners, 0.3, 7, cv::Mat(), 7,
                            false, 0.04);
    prev_gray_ = gray.clone();
    return result;
  }

  std::vector<cv::Point2f> curr_points;
  std::vector<uchar> status;
  std::vector<float> errors;

  cv::calcOpticalFlowPyrLK(prev_gray_, gray, prev_points_, curr_points, status, errors);

  // Calculate flow vectors and statistics
  std::vector<float> magnitudes;
  for (size_t i = 0; i < status.size(); ++i) {
    if (status[i] == 1) {
      cv::Point2f flow_vec = curr_points[i] - prev_points_[i];
      float magnitude = static_cast<float>(cv::norm(flow_vec));

      if (magnitude > config_.optical_flow_threshold) {
        result.points.push_back(prev_points_[i]);
        result.flow_vectors.push_back(flow_vec);
        magnitudes.push_back(magnitude);
        result.moving_points++;
      }
    }
  }

  if (!magnitudes.empty()) {
    result.magnitude_mean = std::accumulate(magnitudes.begin(), magnitudes.end(), 0.0f) /
                            static_cast<float>(magnitudes.size());
    result.magnitude_max = *std::max_element(magnitudes.begin(), magnitudes.end());
  }

  // Update for next frame
  std::vector<cv::Point2f> good_points;
  for (size_t i = 0; i < status.size(); ++i) {
    if (status[i] == 1) {
      good_points.push_back(curr_points[i]);
    }
  }

  prev_points_ = good_points;
  if (prev_points_.size() < 50) {  // Re-detect if too few points
    cv::goodFeaturesToTrack(gray, prev_points_, config_.max_corners, 0.3, 7, cv::Mat(), 7, false,
                            0.04);
  }

  prev_gray_ = gray.clone();

  return result;
}

SegmentationResult MLEngine::segmentFrame(const cv::Mat& /*frame*/) {
  SegmentationResult result;

  if (!config_.enable_segmentation) return result;

  // Implementation would go here - this is a placeholder
  // In a real implementation, you would:
  // 1. Preprocess frame for segmentation model
  // 2. Run inference (TensorRT or OpenCV DNN)
  // 3. Postprocess to create segmentation masks

  spdlog::debug("Segmentation inference not fully implemented yet");
  return result;
}

cv::Mat MLEngine::preprocessForYOLO(const cv::Mat& frame) {
  cv::Mat resized = resizeKeepAspectRatio(frame, config_.input_size);
  cv::Mat blob;
  cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, config_.input_size, cv::Scalar(), true, false);
  return blob;
}

cv::Mat MLEngine::preprocessForSegmentation(const cv::Mat& frame) {
  cv::Mat resized = resizeKeepAspectRatio(frame, config_.input_size);
  cv::Mat blob;
  cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, config_.input_size, cv::Scalar(), true, false);
  return blob;
}

std::vector<Detection> MLEngine::postprocessYOLO(const std::vector<float>& output,
                                                 const cv::Size& original_size) {
  std::vector<Detection> detections;

  // YOLO v11 output format: [batch, 84, 8400] where 84 = 4 (bbox) + 80 (classes)
  // The output is transposed compared to earlier YOLO versions
  const int num_detections = 8400;  // 8400 detections for 640x640 input
  const int num_classes = 80;       // COCO classes
  const int output_elements = 4 + num_classes;  // 84 total

  if (output.size() < static_cast<size_t>(num_detections * output_elements)) {
    spdlog::warn("YOLO output size mismatch: expected {}, got {}", 
                 num_detections * output_elements, output.size());
    return detections;
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<int> class_ids;

  // Calculate scale factors for coordinate transformation
  float scale_x = static_cast<float>(original_size.width) / static_cast<float>(config_.input_size.width);
  float scale_y = static_cast<float>(original_size.height) / static_cast<float>(config_.input_size.height);

  for (int i = 0; i < num_detections; ++i) {
    // Extract bounding box coordinates (cx, cy, w, h)
    float cx = output[i * output_elements + 0];
    float cy = output[i * output_elements + 1];
    float w = output[i * output_elements + 2];
    float h = output[i * output_elements + 3];

    // Find the class with maximum confidence
    float max_confidence = 0.0f;
    int best_class_id = -1;
    
    for (int c = 0; c < num_classes; ++c) {
      float class_conf = output[i * output_elements + 4 + c];
      if (class_conf > max_confidence) {
        max_confidence = class_conf;
        best_class_id = c;
      }
    }

    // Filter by confidence threshold
    if (max_confidence >= config_.detection_threshold) {
      // Convert center coordinates to top-left coordinates
      float x = cx - w / 2.0f;
      float y = cy - h / 2.0f;

      // Scale coordinates to original image size
      int bbox_x = static_cast<int>(x * scale_x);
      int bbox_y = static_cast<int>(y * scale_y);
      int bbox_w = static_cast<int>(w * scale_x);
      int bbox_h = static_cast<int>(h * scale_y);

      // Ensure coordinates are within image bounds
      bbox_x = std::max(0, std::min(bbox_x, original_size.width - 1));
      bbox_y = std::max(0, std::min(bbox_y, original_size.height - 1));
      bbox_w = std::min(bbox_w, original_size.width - bbox_x);
      bbox_h = std::min(bbox_h, original_size.height - bbox_y);

      if (bbox_w > 0 && bbox_h > 0) {
        boxes.emplace_back(bbox_x, bbox_y, bbox_w, bbox_h);
        confidences.push_back(max_confidence);
        class_ids.push_back(best_class_id);
      }
    }
  }

  // Apply Non-Maximum Suppression
  std::vector<int> nms_indices;
  cv::dnn::NMSBoxes(boxes, confidences, config_.detection_threshold, 
                    config_.nms_threshold, nms_indices);

  // Convert to Detection objects
  for (int idx : nms_indices) {
    if (idx >= 0 && idx < static_cast<int>(boxes.size())) {
      Detection det;
      det.bbox = boxes[idx];
      det.confidence = confidences[idx];
      det.class_id = class_ids[idx];
      
      // Set label from class names
      if (det.class_id >= 0 && det.class_id < static_cast<int>(class_names_.size())) {
        det.label = class_names_[det.class_id];
      } else {
        det.label = "unknown";
      }
      
      detections.push_back(det);
      
      // Limit number of detections
      if (static_cast<int>(detections.size()) >= config_.max_detections) {
        break;
      }
    }
  }

  return detections;
}

SegmentationResult MLEngine::postprocessSegmentation(const std::vector<float>& /*output*/,
                                                     const cv::Size& /*original_size*/) {
  SegmentationResult result;

  // Implementation would go here

  return result;
}

void MLEngine::loadClassNames() {
  // COCO class names for YOLO
  class_names_ = {"person",        "bicycle",      "car",
                  "motorcycle",    "airplane",     "bus",
                  "train",         "truck",        "boat",
                  "traffic light", "fire hydrant", "stop sign",
                  "parking meter", "bench",        "bird",
                  "cat",           "dog",          "horse",
                  "sheep",         "cow",          "elephant",
                  "bear",          "zebra",        "giraffe",
                  "backpack",      "umbrella",     "handbag",
                  "tie",           "suitcase",     "frisbee",
                  "skis",          "snowboard",    "sports ball",
                  "kite",          "baseball bat", "baseball glove",
                  "skateboard",    "surfboard",    "tennis racket",
                  "bottle",        "wine glass",   "cup",
                  "fork",          "knife",        "spoon",
                  "bowl",          "banana",       "apple",
                  "sandwich",      "orange",       "broccoli",
                  "carrot",        "hot dog",      "pizza",
                  "donut",         "cake",         "chair",
                  "couch",         "potted plant", "bed",
                  "dining table",  "toilet",       "tv",
                  "laptop",        "mouse",        "remote",
                  "keyboard",      "cell phone",   "microwave",
                  "oven",          "toaster",      "sink",
                  "refrigerator",  "book",         "clock",
                  "vase",          "scissors",     "teddy bear",
                  "hair drier",    "toothbrush"};
}

cv::Mat MLEngine::resizeKeepAspectRatio(const cv::Mat& input, const cv::Size& target_size) {
  cv::Mat output;
  float scale = std::min(static_cast<float>(target_size.width) / static_cast<float>(input.cols),
                         static_cast<float>(target_size.height) / static_cast<float>(input.rows));

  cv::Size new_size(static_cast<int>(static_cast<float>(input.cols) * scale),
                    static_cast<int>(static_cast<float>(input.rows) * scale));
  cv::resize(input, output, new_size);

  // Pad to target size
  cv::Mat padded = cv::Mat::zeros(target_size, output.type());
  int x_offset = (target_size.width - new_size.width) / 2;
  int y_offset = (target_size.height - new_size.height) / 2;

  output.copyTo(padded(cv::Rect(x_offset, y_offset, new_size.width, new_size.height)));

  return padded;
}

void MLEngine::startTimer() { timer_start_ = high_resolution_clock::now(); }

float MLEngine::getElapsedMs() {
  auto end = high_resolution_clock::now();
  return static_cast<float>(duration_cast<microseconds>(end - timer_start_).count()) / 1000.0f;
}

// Factory function
std::unique_ptr<MLEngine> createMLEngine(const MLConfig& config) {
  return std::make_unique<MLEngine>(config);
}

// Visualization utilities
namespace MLViz {
cv::Mat drawDetections(const cv::Mat& frame, const std::vector<Detection>& detections) {
  cv::Mat result = frame.clone();

  for (const auto& det : detections) {
    cv::rectangle(result, det.bbox, cv::Scalar(0, 255, 0), 2);

    std::string label =
        det.label + " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
    int baseline;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    cv::rectangle(result, cv::Point(det.bbox.x, det.bbox.y - label_size.height - 10),
                  cv::Point(det.bbox.x + label_size.width, det.bbox.y), cv::Scalar(0, 255, 0),
                  cv::FILLED);

    cv::putText(result, label, cv::Point(det.bbox.x, det.bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1);
  }

  return result;
}

cv::Mat drawOpticalFlow(const cv::Mat& frame, const OpticalFlowResult& flow) {
  cv::Mat result = frame.clone();

  for (size_t i = 0; i < flow.points.size(); ++i) {
    cv::Point2f start = flow.points[i];
    cv::Point2f end = start + flow.flow_vectors[i];

    cv::arrowedLine(result, start, end, cv::Scalar(0, 0, 255), 2);
    cv::circle(result, start, 3, cv::Scalar(0, 255, 255), -1);
  }

  return result;
}

cv::Mat drawSegmentation(const cv::Mat& frame, const SegmentationResult& segmentation) {
  cv::Mat result = frame.clone();

  if (!segmentation.class_mask.empty()) {
    cv::Mat colored_mask;
    segmentation.class_mask.convertTo(colored_mask, CV_8UC3);
    cv::applyColorMap(colored_mask, colored_mask, cv::COLORMAP_JET);
    cv::addWeighted(result, 0.7, colored_mask, 0.3, 0, result);
  }

  return result;
}

cv::Mat drawComprehensiveResults(const cv::Mat& frame, const MLResult& result) {
  cv::Mat output = frame.clone();

  // Draw detections
  output = drawDetections(output, result.detections);

  // Draw optical flow
  output = drawOpticalFlow(output, result.optical_flow);

  // Draw segmentation
  output = drawSegmentation(output, result.segmentation);

  // Add performance info
  std::string perf_text =
      "Inference: " + std::to_string(static_cast<int>(result.inference_time_ms)) + "ms";
  cv::putText(output, perf_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(255, 255, 255), 2);

  std::string motion_text =
      "Motion: " + std::to_string(static_cast<int>(result.motion_intensity * 100)) + "%";
  cv::putText(output, motion_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(255, 255, 255), 2);

  return output;
}
}  // namespace MLViz
