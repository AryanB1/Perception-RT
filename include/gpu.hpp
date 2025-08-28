#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

inline void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    fprintf(stderr, "[CUDA] %s failed: %s\n", what, cudaGetErrorString(e));
    std::abort();
  }
}

struct GpuContext {
  int W{640}, H{640}, C{3};
  size_t bytes_per_frame{0};

  cudaStream_t s0{nullptr}, s1{nullptr};

  unsigned char* d_in[2]{nullptr, nullptr};    // device RGB input
  unsigned char* d_prev[2]{nullptr, nullptr};  // device previous frame for motion detection
  float*         d_background[2]{nullptr, nullptr}; // device background model
  float*         d_out[2]{nullptr, nullptr};   // device motion output
  float*         d_stats[2]{nullptr, nullptr}; // device motion statistics
  
  unsigned char* h_in[2]{nullptr, nullptr};    // pinned host RGB input
  float*         h_out[2]{nullptr, nullptr};   // pinned host motion output
  float*         h_stats[2]{nullptr, nullptr}; // pinned host motion statistics

  cudaEvent_t pre_start[2]{}, pre_end[2]{};
  cudaEvent_t inf_start[2]{}, inf_end[2]{};
  cudaEvent_t d2h_start[2]{}, d2h_end[2]{};
  cudaEvent_t pre_done[2]{}, inf_done[2]{};

  // Motion detection parameters
  float motion_threshold{25.0f};
  float learning_rate{0.01f};
  bool has_previous_frame[2]{false, false};
  
  // ML inference parameters
  bool enable_ml_inference{false};
  float* d_ml_output[2]{nullptr, nullptr};  // ML inference output buffer
  float* h_ml_output[2]{nullptr, nullptr};  // Host ML output buffer
  size_t ml_output_size{0};
  
  int infer_iters{2048}; // Kept for compatibility
};

inline void gpu_init(GpuContext& ctx, int W, int H, int C = 3, int infer_iters = 2048) {
  ctx.W = W; ctx.H = H; ctx.C = C; ctx.infer_iters = infer_iters;
  ctx.bytes_per_frame = static_cast<size_t>(W) * H * C;
  const size_t pixels = static_cast<size_t>(W) * H;

  int leastPri = 0, greatestPri = 0;
  cuda_check(cudaDeviceGetStreamPriorityRange(&leastPri, &greatestPri), "GetStreamPriorityRange");
  cuda_check(cudaStreamCreateWithPriority(&ctx.s1, cudaStreamNonBlocking, greatestPri), "Create stream s1");
  cuda_check(cudaStreamCreateWithPriority(&ctx.s0, cudaStreamNonBlocking, leastPri),    "Create stream s0");

  for (int i = 0; i < 2; ++i) {
    // Allocate device memory
    cuda_check(cudaMalloc(&ctx.d_in[i],  ctx.bytes_per_frame), "Malloc d_in");
    cuda_check(cudaMalloc(&ctx.d_prev[i], ctx.bytes_per_frame), "Malloc d_prev");
    cuda_check(cudaMalloc(&ctx.d_background[i], pixels * sizeof(float)), "Malloc d_background");
    cuda_check(cudaMalloc(&ctx.d_out[i], pixels * sizeof(float)), "Malloc d_out");
    
    // Allocate stats memory (for reduction results)
    int stats_blocks = (static_cast<int>(pixels) + 255) / 256;
    cuda_check(cudaMalloc(&ctx.d_stats[i], stats_blocks * 2 * sizeof(float)), "Malloc d_stats");
    
    // Allocate ML inference buffers if enabled
    if (ctx.enable_ml_inference) {
      ctx.ml_output_size = 25200 * 85; // YOLOv11 output size (approximate)
      cuda_check(cudaMalloc(&ctx.d_ml_output[i], ctx.ml_output_size * sizeof(float)), "Malloc d_ml_output");
      cuda_check(cudaHostAlloc(&ctx.h_ml_output[i], ctx.ml_output_size * sizeof(float), cudaHostAllocDefault), "HostAlloc h_ml_output");
    }
    
    // Allocate pinned host memory
    cuda_check(cudaHostAlloc(&ctx.h_in[i],  ctx.bytes_per_frame, cudaHostAllocDefault), "HostAlloc h_in");
    cuda_check(cudaHostAlloc(&ctx.h_out[i], pixels * sizeof(float), cudaHostAllocDefault), "HostAlloc h_out");
    cuda_check(cudaHostAlloc(&ctx.h_stats[i], stats_blocks * 2 * sizeof(float), cudaHostAllocDefault), "HostAlloc h_stats");

    // Initialize background model to zero
    cuda_check(cudaMemset(ctx.d_background[i], 0, pixels * sizeof(float)), "Memset d_background");
    cuda_check(cudaMemset(ctx.d_prev[i], 0, ctx.bytes_per_frame), "Memset d_prev");
    
    // Create events
    cuda_check(cudaEventCreateWithFlags(&ctx.pre_start[i], cudaEventDefault), "Event pre_start");
    cuda_check(cudaEventCreateWithFlags(&ctx.pre_end[i],   cudaEventDefault), "Event pre_end");
    cuda_check(cudaEventCreateWithFlags(&ctx.inf_start[i], cudaEventDefault), "Event inf_start");
    cuda_check(cudaEventCreateWithFlags(&ctx.inf_end[i],   cudaEventDefault), "Event inf_end");
    cuda_check(cudaEventCreateWithFlags(&ctx.d2h_start[i], cudaEventDefault), "Event d2h_start");
    cuda_check(cudaEventCreateWithFlags(&ctx.d2h_end[i],   cudaEventDefault), "Event d2h_end");
    cuda_check(cudaEventCreateWithFlags(&ctx.pre_done[i],  cudaEventDisableTiming), "Event pre_done");
    cuda_check(cudaEventCreateWithFlags(&ctx.inf_done[i],  cudaEventDisableTiming), "Event inf_done");
  }
}

inline void gpu_destroy(GpuContext& ctx) {
  for (int i = 0; i < 2; ++i) {
    if (ctx.d_in[i])  cudaFree(ctx.d_in[i]);
    if (ctx.d_prev[i]) cudaFree(ctx.d_prev[i]);
    if (ctx.d_background[i]) cudaFree(ctx.d_background[i]);
    if (ctx.d_out[i]) cudaFree(ctx.d_out[i]);
    if (ctx.d_stats[i]) cudaFree(ctx.d_stats[i]);
    if (ctx.d_ml_output[i]) cudaFree(ctx.d_ml_output[i]);
    
    if (ctx.h_in[i])  cudaFreeHost(ctx.h_in[i]);
    if (ctx.h_out[i]) cudaFreeHost(ctx.h_out[i]);
    if (ctx.h_stats[i]) cudaFreeHost(ctx.h_stats[i]);
    if (ctx.h_ml_output[i]) cudaFreeHost(ctx.h_ml_output[i]);

    if (ctx.pre_start[i]) cudaEventDestroy(ctx.pre_start[i]);
    if (ctx.pre_end[i])   cudaEventDestroy(ctx.pre_end[i]);
    if (ctx.inf_start[i]) cudaEventDestroy(ctx.inf_start[i]);
    if (ctx.inf_end[i])   cudaEventDestroy(ctx.inf_end[i]);
    if (ctx.d2h_start[i]) cudaEventDestroy(ctx.d2h_start[i]);
    if (ctx.d2h_end[i])   cudaEventDestroy(ctx.d2h_end[i]);
    if (ctx.pre_done[i])  cudaEventDestroy(ctx.pre_done[i]);
    if (ctx.inf_done[i])  cudaEventDestroy(ctx.inf_done[i]);
  }
  if (ctx.s0) cudaStreamDestroy(ctx.s0);
  if (ctx.s1) cudaStreamDestroy(ctx.s1);
}

void gpu_stage_h2d_async(GpuContext& ctx, int idx, const unsigned char* host_rgb);

void gpu_launch_infer_async(GpuContext& ctx, int idx);

inline void gpu_wait_infer_done(GpuContext& ctx, int idx) {
  cuda_check(cudaEventSynchronize(ctx.inf_done[idx]), "EventSync inf_done");
}

inline float gpu_elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.f;
  cuda_check(cudaEventElapsedTime(&ms, a, b), "EventElapsedTime");
  return ms;
}

// Helper function to extract motion detection results
struct GpuMotionResult {
  bool motion_detected;
  double motion_intensity;
  int motion_pixels;
};

inline GpuMotionResult gpu_get_motion_result(GpuContext& ctx, int idx) {
  GpuMotionResult result{};
  
  // Calculate motion statistics from reduction results
  int stats_blocks = (static_cast<int>(ctx.W * ctx.H) + 255) / 256;
  
  float total_motion_pixels = 0.0f;
  float total_motion_intensity = 0.0f;
  
  for (int i = 0; i < stats_blocks; ++i) {
    total_motion_pixels += ctx.h_stats[idx][i];
    total_motion_intensity += ctx.h_stats[idx][i + stats_blocks];
  }
  
  result.motion_pixels = static_cast<int>(total_motion_pixels);
  result.motion_intensity = static_cast<double>(total_motion_intensity) / (ctx.W * ctx.H);
  result.motion_detected = result.motion_intensity > 0.01; // 1% threshold
  
  return result;
}
