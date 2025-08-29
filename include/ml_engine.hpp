#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <memory>

// Forward declarations
struct MLResult;
struct Detection;
struct OpticalFlowResult;
struct SegmentationResult;

// Detection result for object detection
struct Detection {
    int class_id{-1};
    float confidence{0.0f};
    cv::Rect bbox;  // Use integer Rect instead of Rect2f
    std::string label;
    cv::Point2f center() const { 
        return cv::Point2f(static_cast<float>(bbox.x) + static_cast<float>(bbox.width)/2.0f, 
                          static_cast<float>(bbox.y) + static_cast<float>(bbox.height)/2.0f); 
    }
};

// Optical flow result
struct OpticalFlowResult {
    std::vector<cv::Point2f> points;
    std::vector<cv::Point2f> flow_vectors;
    float magnitude_mean{0.0f};
    float magnitude_max{0.0f};
    int moving_points{0};
};

// Semantic segmentation result
struct SegmentationResult {
    cv::Mat class_mask;        // HxW mask with class IDs
    cv::Mat confidence_mask;   // HxW confidence scores
    std::vector<int> detected_classes;
    std::unordered_map<int, float> class_areas; // class_id -> normalized area
};

// Comprehensive ML result combining all inference types
struct MLResult {
    // Object Detection (YOLOv11)
    std::vector<Detection> detections;
    int total_objects{0};
    float max_confidence{0.0f};
    
    // Optical Flow
    OpticalFlowResult optical_flow;
    bool significant_motion{false};
    
    // Semantic Segmentation
    SegmentationResult segmentation;
    
    // Motion Analysis (enhanced)
    bool motion_detected{false};
    float motion_intensity{0.0f};
    int motion_pixels{0};
    cv::Rect motion_bbox;
    
    // Performance metrics
    float inference_time_ms{0.0f};
    float preprocessing_time_ms{0.0f};
    float postprocessing_time_ms{0.0f};
};

// ML Engine configuration
struct MLConfig {
    // Model paths
    std::string yolo_model_path = "models/yolov11n.onnx";
    std::string segmentation_model_path = "models/deeplabv3.onnx";
    
    // TensorRT settings
    bool use_tensorrt = true;
    bool use_fp16 = true;
    int max_batch_size = 1;
    size_t max_workspace_size = 1ULL << 30; // 1GB
    
    // Detection settings
    float detection_threshold = 0.5f;
    float nms_threshold = 0.4f;
    int max_detections = 100;
    
    // Optical flow settings
    int max_corners = 100;
    float optical_flow_threshold = 1.0f;
    
    // Segmentation settings
    float segmentation_threshold = 0.5f;
    
    // Processing settings
    cv::Size input_size{640, 640};
    bool enable_detection = true;
    bool enable_optical_flow = true;
    bool enable_segmentation = false; // Disabled by default due to computational cost
};

// TensorRT Logger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
private:
    Severity min_severity_ = Severity::kWARNING;
};

// TensorRT Engine wrapper
class TensorRTEngine {
public:
    TensorRTEngine(const std::string& onnx_path, const MLConfig& config);
    ~TensorRTEngine();
    
    bool build();
    bool loadEngine(const std::string& engine_path);
    bool saveEngine(const std::string& engine_path);
    
    bool infer(const std::vector<float>& input, std::vector<float>& output);
    
    cv::Size getInputSize() const { return input_size_; }
    size_t getInputChannels() const { return input_channels_; }
    size_t getOutputSize() const { return output_size_; }

private:
    std::unique_ptr<nvinfer1::IBuilder> builder_;
    std::unique_ptr<nvinfer1::INetworkDefinition> network_;
    std::unique_ptr<nvinfer1::IBuilderConfig> config_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    std::unique_ptr<TRTLogger> logger_;
    
    cv::Size input_size_;
    size_t input_channels_;
    size_t output_size_;
    
    // CUDA resources
    void* d_input_{nullptr};
    void* d_output_{nullptr};
    cudaStream_t stream_{nullptr};
    
    void allocateBuffers();
    void freeBuffers();
};

// Main ML Engine class
class MLEngine {
public:
    MLEngine(const MLConfig& config = MLConfig{});
    ~MLEngine();
    
    bool initialize();
    MLResult process(const cv::Mat& frame);
    
    // Individual processing methods
    std::vector<Detection> detectObjects(const cv::Mat& frame);
    OpticalFlowResult computeOpticalFlow(const cv::Mat& frame);
    SegmentationResult segmentFrame(const cv::Mat& frame);
    
    // Model management
    bool loadModel(const std::string& model_path, const std::string& model_type);
    void setConfig(const MLConfig& config) { config_ = config; }
    const MLConfig& getConfig() const { return config_; }
    
    // Performance monitoring
    struct PerformanceStats {
        float avg_inference_time{0.0f};
        float avg_preprocessing_time{0.0f};
        float avg_postprocessing_time{0.0f};
        int frames_processed{0};
        
        void update(float inference, float preprocessing, float postprocessing);
        void reset();
    };
    
    const PerformanceStats& getStats() const { return stats_; }

private:
    MLConfig config_;
    PerformanceStats stats_;
    
    std::unique_ptr<TensorRTEngine> yolo_engine_;
    std::unique_ptr<TensorRTEngine> segmentation_engine_;

    // OpenCV DNN fallback
    cv::dnn::Net yolo_net_;
    cv::dnn::Net segmentation_net_;
    
    // Optical flow tracker
    cv::Ptr<cv::SparsePyrLKOpticalFlow> optical_flow_tracker_;
    std::vector<cv::Point2f> prev_points_;
    cv::Mat prev_gray_;
    
    // YOLO class names
    std::vector<std::string> class_names_;
    
    // Preprocessing
    cv::Mat preprocessForYOLO(const cv::Mat& frame);
    cv::Mat preprocessForSegmentation(const cv::Mat& frame);
    
    // Postprocessing
    std::vector<Detection> postprocessYOLO(const std::vector<float>& output, 
                                           const cv::Size& original_size);
    SegmentationResult postprocessSegmentation(const std::vector<float>& output,
                                               const cv::Size& original_size);
    
    // Utility methods
    void loadClassNames();
    cv::Mat resizeKeepAspectRatio(const cv::Mat& input, const cv::Size& target_size);
    
    // Performance timing
    std::chrono::high_resolution_clock::time_point timer_start_;
    void startTimer();
    float getElapsedMs();
};

// Factory function for easy creation
std::unique_ptr<MLEngine> createMLEngine(const MLConfig& config = MLConfig{});

// Utility functions for visualization
namespace MLViz {
    cv::Mat drawDetections(const cv::Mat& frame, const std::vector<Detection>& detections);
    cv::Mat drawOpticalFlow(const cv::Mat& frame, const OpticalFlowResult& flow);
    cv::Mat drawSegmentation(const cv::Mat& frame, const SegmentationResult& segmentation);
    cv::Mat drawComprehensiveResults(const cv::Mat& frame, const MLResult& result);
}
