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

  unsigned char* d_in[2]{nullptr, nullptr};   // device RGB input
  float*         d_out[2]{nullptr, nullptr};  // device output
  unsigned char* h_in[2]{nullptr, nullptr};   // pinned host RGB input
  float*         h_out[2]{nullptr, nullptr};  // pinned host output

  cudaEvent_t pre_start[2]{}, pre_end[2]{};
  cudaEvent_t inf_start[2]{}, inf_end[2]{};
  cudaEvent_t d2h_start[2]{}, d2h_end[2]{};
  cudaEvent_t pre_done[2]{}, inf_done[2]{};

  int infer_iters{2048};
};

inline void gpu_init(GpuContext& ctx, int W, int H, int C = 3, int infer_iters = 2048) {
  ctx.W = W; ctx.H = H; ctx.C = C; ctx.infer_iters = infer_iters;
  ctx.bytes_per_frame = static_cast<size_t>(W) * H * C;

  int leastPri = 0, greatestPri = 0;
  cuda_check(cudaDeviceGetStreamPriorityRange(&leastPri, &greatestPri), "GetStreamPriorityRange");
  cuda_check(cudaStreamCreateWithPriority(&ctx.s1, cudaStreamNonBlocking, greatestPri), "Create stream s1");
  cuda_check(cudaStreamCreateWithPriority(&ctx.s0, cudaStreamNonBlocking, leastPri),    "Create stream s0");

  for (int i = 0; i < 2; ++i) {
    cuda_check(cudaMalloc(&ctx.d_in[i],  ctx.bytes_per_frame),      "Malloc d_in");
    cuda_check(cudaMalloc(&ctx.d_out[i], ctx.bytes_per_frame * sizeof(float)), "Malloc d_out");
    cuda_check(cudaHostAlloc(&ctx.h_in[i],  ctx.bytes_per_frame, cudaHostAllocDefault), "HostAlloc h_in");
    cuda_check(cudaHostAlloc(&ctx.h_out[i], ctx.bytes_per_frame * sizeof(float), cudaHostAllocDefault), "HostAlloc h_out");

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
    if (ctx.d_out[i]) cudaFree(ctx.d_out[i]);
    if (ctx.h_in[i])  cudaFreeHost(ctx.h_in[i]);
    if (ctx.h_out[i]) cudaFreeHost(ctx.h_out[i]);

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
