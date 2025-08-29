#include "motion_detector.hpp"

#include <algorithm>
#include <cmath>

MotionDetector::MotionDetector(Algorithm algo, int width, int height)
    : algorithm_(algo), width_(width), height_(height) {
  switch (algorithm_) {
    case Algorithm::BACKGROUND_SUB:
      // Use MOG2 background subtractor
      bg_subtractor_ = cv::createBackgroundSubtractorMOG2(500, 16, true);
      break;
    case Algorithm::FRAME_DIFF:
    case Algorithm::OPTICAL_FLOW:
    default:
      break;
  }
}

MotionDetector::~MotionDetector() = default;

MotionResult MotionDetector::process_frame(const cv::Mat& frame) {
  switch (algorithm_) {
    case Algorithm::FRAME_DIFF:
      return detect_frame_diff(frame);
    case Algorithm::BACKGROUND_SUB:
      return detect_background_sub(frame);
    case Algorithm::OPTICAL_FLOW:
      return detect_optical_flow(frame);
    default:
      return detect_frame_diff(frame);
  }
}

void MotionDetector::reset() {
  prev_frame_ = cv::Mat();
  prev_gray_ = cv::Mat();
  prev_corners_.clear();

  if (bg_subtractor_) {
    bg_subtractor_ = cv::createBackgroundSubtractorMOG2(500, 16, true);
  }
}

MotionResult MotionDetector::detect_frame_diff(const cv::Mat& frame) {
  MotionResult result;

  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);

  if (prev_frame_.empty()) {
    prev_frame_ = gray.clone();
    return result;  // No motion on first frame
  }

  // Compute absolute difference
  cv::Mat diff;
  cv::absdiff(prev_frame_, gray, diff);

  // Apply threshold
  cv::Mat thresh;
  cv::threshold(diff, thresh, threshold_, 255, cv::THRESH_BINARY);

  // Apply morphological operations to reduce noise
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel);

  // Find contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  // Analyze contours
  double total_area = 0;
  std::vector<cv::Rect> motion_rects;

  for (const auto& contour : contours) {
    double area = cv::contourArea(contour);
    if (area > min_contour_area_) {
      total_area += area;
      motion_rects.push_back(cv::boundingRect(contour));
    }
  }

  result.motion_pixels = cv::countNonZero(thresh);
  result.motion_intensity = std::min(1.0, total_area / (width_ * height_));
  result.motion_detected = result.motion_intensity > 0.01;  // 1% threshold

  // Combine all motion rectangles into one bounding box
  if (!motion_rects.empty()) {
    cv::Rect combined = motion_rects[0];
    for (size_t i = 1; i < motion_rects.size(); ++i) {
      combined |= motion_rects[i];
    }
    result.bounding_box = combined;
  }

  prev_frame_ = gray.clone();
  return result;
}

MotionResult MotionDetector::detect_background_sub(const cv::Mat& frame) {
  MotionResult result;

  cv::Mat fg_mask;
  bg_subtractor_->apply(frame, fg_mask);

  // Apply morphological operations to clean up the mask
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_CLOSE, kernel);

  // Find contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(fg_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  // Analyze contours
  double total_area = 0;
  std::vector<cv::Rect> motion_rects;

  for (const auto& contour : contours) {
    double area = cv::contourArea(contour);
    if (area > min_contour_area_) {
      total_area += area;
      motion_rects.push_back(cv::boundingRect(contour));
    }
  }

  result.motion_pixels = cv::countNonZero(fg_mask);
  result.motion_intensity = std::min(1.0, total_area / (width_ * height_));
  result.motion_detected =
      result.motion_intensity > 0.005;  // 0.5% threshold for background subtraction

  // Combine all motion rectangles into one bounding box
  if (!motion_rects.empty()) {
    cv::Rect combined = motion_rects[0];
    for (size_t i = 1; i < motion_rects.size(); ++i) {
      combined |= motion_rects[i];
    }
    result.bounding_box = combined;
  }

  return result;
}

MotionResult MotionDetector::detect_optical_flow(const cv::Mat& frame) {
  MotionResult result;

  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);

  if (prev_gray_.empty()) {
    prev_gray_ = gray.clone();
    // Detect initial corner features
    cv::goodFeaturesToTrack(gray, prev_corners_, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
    return result;  // No motion on first frame
  }

  if (prev_corners_.empty()) {
    // Re-detect corners if we lost them
    cv::goodFeaturesToTrack(prev_gray_, prev_corners_, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
    if (prev_corners_.empty()) {
      prev_gray_ = gray.clone();
      return result;
    }
  }

  // Calculate optical flow
  std::vector<cv::Point2f> curr_corners;
  std::vector<uchar> status;
  std::vector<float> errors;

  cv::calcOpticalFlowPyrLK(prev_gray_, gray, prev_corners_, curr_corners, status, errors);

  // Filter good tracks and calculate motion vectors
  std::vector<cv::Point2f> good_prev, good_curr;
  double total_motion = 0;

  for (size_t i = 0; i < status.size(); ++i) {
    if (status[i] == 1) {
      good_prev.push_back(prev_corners_[i]);
      good_curr.push_back(curr_corners[i]);

      cv::Point2f motion_vec = curr_corners[i] - prev_corners_[i];
      double magnitude = cv::norm(motion_vec);
      if (magnitude > 1.0) {  // Minimum motion threshold
        result.motion_vectors.push_back(motion_vec);
        total_motion += magnitude;
      }
    }
  }

  result.motion_intensity = std::min(1.0, total_motion / (width_ + height_));
  result.motion_detected = result.motion_intensity > 0.02;  // Motion threshold
  result.motion_pixels = static_cast<int>(result.motion_vectors.size());

  // Update for next iteration
  prev_gray_ = gray.clone();
  if (good_curr.size() < 50) {  // Re-detect if we have too few points
    cv::goodFeaturesToTrack(gray, prev_corners_, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
  } else {
    prev_corners_ = good_curr;
  }

  return result;
}
