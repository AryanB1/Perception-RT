#pragma once

#include <memory>
#include <vector>

#if defined(HAVE_OPENCV)
#include <opencv2/opencv.hpp>

struct MotionResult {
    bool motion_detected{false};
    double motion_intensity{0.0};  // 0.0 to 1.0
    int motion_pixels{0};
    std::vector<cv::Point2f> motion_vectors;  // optical flow vectors
    cv::Rect bounding_box;  // bounding box of motion region
};

class MotionDetector {
public:
    enum class Algorithm {
        FRAME_DIFF,      // Simple frame differencing
        BACKGROUND_SUB,  // Background subtraction
        OPTICAL_FLOW     // Lucas-Kanade optical flow
    };

    MotionDetector(Algorithm algo = Algorithm::BACKGROUND_SUB, 
                   int width = 640, int height = 640);
    ~MotionDetector();

    // Process a frame and return motion detection results
    MotionResult process_frame(const cv::Mat& frame);
    
    // Reset detector state (useful for scene changes)
    void reset();

    // Configuration
    void set_threshold(double threshold) { threshold_ = threshold; }
    void set_min_contour_area(double area) { min_contour_area_ = area; }

private:
    Algorithm algorithm_;
    int width_, height_;
    double threshold_{25.0};
    double min_contour_area_{500.0};
    
    // Previous frame for frame differencing
    cv::Mat prev_frame_;
    
    // Background subtractor
    cv::Ptr<cv::BackgroundSubtractor> bg_subtractor_;
    
    // For optical flow
    std::vector<cv::Point2f> prev_corners_;
    cv::Mat prev_gray_;

    MotionResult detect_frame_diff(const cv::Mat& frame);
    MotionResult detect_background_sub(const cv::Mat& frame);
    MotionResult detect_optical_flow(const cv::Mat& frame);
};

#else

struct MotionResult {
    bool motion_detected{false};
    double motion_intensity{0.0};  // 0.0 to 1.0
    int motion_pixels{0};
};

class MotionDetector {
public:
    enum class Algorithm {
        FRAME_DIFF,      // Simple frame differencing
        BACKGROUND_SUB,  // Background subtraction
        OPTICAL_FLOW     // Lucas-Kanade optical flow
    };

    MotionDetector(Algorithm algo = Algorithm::BACKGROUND_SUB, 
                   int width = 640, int height = 640);
    ~MotionDetector();

    // Process a frame and return motion detection results
    MotionResult process_frame(const unsigned char* frame_data);
    
    // Reset detector state
    void reset();

    // Configuration
    void set_threshold(double threshold) { threshold_ = threshold; }
    void set_min_contour_area(double area) { min_contour_area_ = area; }

private:
    Algorithm algorithm_;
    int width_, height_;
    double threshold_{25.0};
    double min_contour_area_{500.0};
    
    // Simple motion detection without OpenCV
    std::vector<unsigned char> prev_frame_;
    
    MotionResult detect_simple_diff(const unsigned char* frame_data);
};

#endif
