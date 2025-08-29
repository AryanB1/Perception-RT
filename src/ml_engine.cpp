#include "ml_engine.hpp"
#include <spdlog/spdlog.h>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <cmath>
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
        }
    }
}

// TensorRT Engine Implementation
TensorRTEngine::TensorRTEngine(const std::string& onnx_path, const MLConfig& config)
    : logger_(std::make_unique<TRTLogger>()) {
    
    builder_.reset(nvinfer1::createInferBuilder(*logger_));
    if (!builder_) {
        throw std::runtime_error("Failed to create TensorRT builder");
    }
    
    // Create network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network_.reset(builder_->createNetworkV2(explicitBatch));
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
    if (config.use_fp16 && builder_->platformHasFastFp16()) {
        config_->setFlag(nvinfer1::BuilderFlag::kFP16);
        spdlog::info("TensorRT: FP16 optimization enabled");
    }
    
    input_size_ = config.input_size;
    input_channels_ = 3; // RGB
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

bool TensorRTEngine::infer(const std::vector<float>& input, std::vector<float>& output) {
    if (!context_ || !d_input_ || !d_output_) {
        spdlog::error("TensorRT engine not properly initialized");
        return false;
    }
    
    // Copy input to device
    size_t input_size = input.size() * sizeof(float);
    if (cudaMemcpyAsync(d_input_, input.data(), input_size, cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
        spdlog::error("Failed to copy input to device");
        return false;
    }
    
    // Run inference
    void* bindings[] = {d_input_, d_output_};
    if (!context_->executeV2(bindings)) {
        spdlog::error("TensorRT inference failed");
        return false;
    }
    
    // Copy output to host
    size_t output_size = output.size() * sizeof(float);
    if (cudaMemcpyAsync(output.data(), d_output_, output_size, cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
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
    
    // For YOLO, output size varies based on model architecture
    // This is a simplified calculation - should be determined from actual model
    output_size_ = 25200 * 85; // Typical YOLOv11 output size
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
void MLEngine::PerformanceStats::update(float inference, float preprocessing, float postprocessing) {
    frames_processed++;
    float alpha = 1.0f / frames_processed; // Simple moving average
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
                
                // Try to load existing engine, otherwise build new one
                std::string engine_path = config_.yolo_model_path + ".trt";
                if (!yolo_engine_->loadEngine(engine_path)) {
                    spdlog::info("Building new TensorRT engine for YOLO...");
                    if (!yolo_engine_->build()) {
                        spdlog::error("Failed to build YOLO TensorRT engine");
                        return false;
                    }
                    yolo_engine_->saveEngine(engine_path);
                }
            }
            
            // Initialize segmentation engine if enabled
            if (config_.enable_segmentation) {
                segmentation_engine_ = std::make_unique<TensorRTEngine>(config_.segmentation_model_path, config_);
                
                std::string seg_engine_path = config_.segmentation_model_path + ".trt";
                if (!segmentation_engine_->loadEngine(seg_engine_path)) {
                    spdlog::info("Building new TensorRT engine for segmentation...");
                    if (!segmentation_engine_->build()) {
                        spdlog::warn("Failed to build segmentation TensorRT engine, falling back to OpenCV DNN");
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
        result.total_objects = result.detections.size();
        if (!result.detections.empty()) {
            result.max_confidence = std::max_element(result.detections.begin(), result.detections.end(),
                [](const Detection& a, const Detection& b) { return a.confidence < b.confidence; })->confidence;
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
                                          result.total_objects * 0.1f);
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
    stats_.update(result.inference_time_ms, result.preprocessing_time_ms, result.postprocessing_time_ms);
    
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
            
            spdlog::debug("YOLO TensorRT: prep={:.2f}ms, infer={:.2f}ms, post={:.2f}ms", 
                         prep_time, inf_time, post_time);
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
        cv::goodFeaturesToTrack(gray, prev_points_, config_.max_corners, 0.3, 7, cv::Mat(), 7, false, 0.04);
        return result;
    }
    
    if (prev_points_.empty()) {
        cv::goodFeaturesToTrack(prev_gray_, prev_points_, config_.max_corners, 0.3, 7, cv::Mat(), 7, false, 0.04);
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
            float magnitude = cv::norm(flow_vec);
            
            if (magnitude > config_.optical_flow_threshold) {
                result.points.push_back(prev_points_[i]);
                result.flow_vectors.push_back(flow_vec);
                magnitudes.push_back(magnitude);
                result.moving_points++;
            }
        }
    }
    
    if (!magnitudes.empty()) {
        result.magnitude_mean = std::accumulate(magnitudes.begin(), magnitudes.end(), 0.0f) / magnitudes.size();
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
        cv::goodFeaturesToTrack(gray, prev_points_, config_.max_corners, 0.3, 7, cv::Mat(), 7, false, 0.04);
    }
    
    prev_gray_ = gray.clone();
    
    return result;
}

SegmentationResult MLEngine::segmentFrame(const cv::Mat& frame) {
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
    cv::dnn::blobFromImage(resized, blob, 1.0/255.0, config_.input_size, cv::Scalar(), true, false);
    return blob;
}

cv::Mat MLEngine::preprocessForSegmentation(const cv::Mat& frame) {
    cv::Mat resized = resizeKeepAspectRatio(frame, config_.input_size);
    cv::Mat blob;
    cv::dnn::blobFromImage(resized, blob, 1.0/255.0, config_.input_size, cv::Scalar(), true, false);
    return blob;
}

std::vector<Detection> MLEngine::postprocessYOLO(const std::vector<float>& output, const cv::Size& original_size) {
    std::vector<Detection> detections;
    
    // This is a simplified YOLO postprocessing
    // Real implementation would depend on specific YOLOv11 output format
    
    return detections;
}

SegmentationResult MLEngine::postprocessSegmentation(const std::vector<float>& output, const cv::Size& original_size) {
    SegmentationResult result;
    
    // Implementation would go here
    
    return result;
}

void MLEngine::loadClassNames() {
    // COCO class names for YOLO
    class_names_ = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
}

cv::Mat MLEngine::resizeKeepAspectRatio(const cv::Mat& input, const cv::Size& target_size) {
    cv::Mat output;
    float scale = std::min(static_cast<float>(target_size.width) / input.cols,
                          static_cast<float>(target_size.height) / input.rows);
    
    cv::Size new_size(static_cast<int>(input.cols * scale), static_cast<int>(input.rows * scale));
    cv::resize(input, output, new_size);
    
    // Pad to target size
    cv::Mat padded = cv::Mat::zeros(target_size, output.type());
    int x_offset = (target_size.width - new_size.width) / 2;
    int y_offset = (target_size.height - new_size.height) / 2;
    
    output.copyTo(padded(cv::Rect(x_offset, y_offset, new_size.width, new_size.height)));
    
    return padded;
}

void MLEngine::startTimer() {
    timer_start_ = high_resolution_clock::now();
}

float MLEngine::getElapsedMs() {
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - timer_start_).count() / 1000.0f;
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
            
            std::string label = det.label + " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
            int baseline;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            
            cv::rectangle(result, 
                         cv::Point(det.bbox.x, det.bbox.y - label_size.height - 10),
                         cv::Point(det.bbox.x + label_size.width, det.bbox.y),
                         cv::Scalar(0, 255, 0), cv::FILLED);
            
            cv::putText(result, label, cv::Point(det.bbox.x, det.bbox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
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
        std::string perf_text = "Inference: " + std::to_string(static_cast<int>(result.inference_time_ms)) + "ms";
        cv::putText(output, perf_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        std::string motion_text = "Motion: " + std::to_string(static_cast<int>(result.motion_intensity * 100)) + "%";
        cv::putText(output, motion_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        return output;
    }
}
