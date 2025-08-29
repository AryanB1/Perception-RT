#include <cmath>

#include "gpu.hpp"

__global__ void identity_rgb(const unsigned char* __restrict__ in, unsigned char* __restrict__ out,
                             int n_bytes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_bytes) out[i] = in[i];
}

// Motion detection kernel using frame differencing
__global__ void motion_detection_kernel(const unsigned char* __restrict__ curr_frame,
                                        const unsigned char* __restrict__ prev_frame,
                                        float* __restrict__ motion_out, int width, int height,
                                        float threshold) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int idx = y * width + x;
  int rgb_idx = idx * 3;

  // Convert to grayscale and compute difference
  float curr_gray =
      (curr_frame[rgb_idx] + curr_frame[rgb_idx + 1] + curr_frame[rgb_idx + 2]) / 3.0f;
  float prev_gray =
      (prev_frame[rgb_idx] + prev_frame[rgb_idx + 1] + prev_frame[rgb_idx + 2]) / 3.0f;

  float diff = fabsf(curr_gray - prev_gray);

  // Apply threshold and normalize
  motion_out[idx] = (diff > threshold) ? diff / 255.0f : 0.0f;
}

// Background subtraction kernel (simplified MOG)
__global__ void background_subtraction_kernel(const unsigned char* __restrict__ curr_frame,
                                              float* __restrict__ background_model,
                                              float* __restrict__ motion_out, int width, int height,
                                              float learning_rate, float threshold) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int idx = y * width + x;
  int rgb_idx = idx * 3;

  // Convert current frame to grayscale
  float curr_gray =
      (curr_frame[rgb_idx] + curr_frame[rgb_idx + 1] + curr_frame[rgb_idx + 2]) / 3.0f;

  // Update background model
  float bg_pixel = background_model[idx];
  float diff = fabsf(curr_gray - bg_pixel);

  // Adaptive background update
  if (diff < threshold * 2.0f) {
    background_model[idx] = bg_pixel * (1.0f - learning_rate) + curr_gray * learning_rate;
  }

  // Output motion
  motion_out[idx] = (diff > threshold) ? diff / 255.0f : 0.0f;
}

// Reduction kernel to compute motion statistics
__global__ void motion_stats_kernel(const float* __restrict__ motion_data,
                                    float* __restrict__ stats_out, int width, int height) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pixels = width * height;

  // Load data into shared memory
  float motion_val = (idx < total_pixels) ? motion_data[idx] : 0.0f;
  sdata[tid] = (motion_val > 0.0f) ? 1.0f : 0.0f;  // Count motion pixels
  sdata[tid + blockDim.x] = motion_val;            // Sum motion intensity

  __syncthreads();

  // Reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];                            // Motion pixel count
      sdata[tid + blockDim.x] += sdata[tid + s + blockDim.x];  // Motion intensity sum
    }
    __syncthreads();
  }

  // Write result for this block
  if (tid == 0) {
    stats_out[blockIdx.x] = sdata[0];                       // Motion pixel count
    stats_out[blockIdx.x + gridDim.x] = sdata[blockDim.x];  // Motion intensity sum
  }
}

void gpu_stage_h2d_async(GpuContext& ctx, int idx, const unsigned char* host_rgb) {
  memcpy(ctx.h_in[idx], host_rgb, ctx.bytes_per_frame);
  // H2D on S0 with timing
  cuda_check(cudaEventRecord(ctx.pre_start[idx], ctx.s0), "EventRecord pre_start");
  cuda_check(cudaMemcpyAsync(ctx.d_in[idx], ctx.h_in[idx], ctx.bytes_per_frame,
                             cudaMemcpyHostToDevice, ctx.s0),
             "MemcpyAsync H2D");

  // Simple preprocessing - just copy for now
  const int n = static_cast<int>(ctx.bytes_per_frame);
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  identity_rgb<<<blocks, threads, 0, ctx.s0>>>(ctx.d_in[idx], ctx.d_in[idx], n);
  cuda_check(cudaGetLastError(), "identity_rgb launch");
  cuda_check(cudaEventRecord(ctx.pre_end[idx], ctx.s0), "EventRecord pre_end");
  cuda_check(cudaEventRecord(ctx.pre_done[idx], ctx.s0), "EventRecord pre_done");
}

void gpu_launch_infer_async(GpuContext& ctx, int idx) {
  cuda_check(cudaStreamWaitEvent(ctx.s1, ctx.pre_done[idx], 0), "StreamWaitEvent pre_done");

  const int pixels = ctx.W * ctx.H;

  // Configure 2D grid for motion detection
  dim3 blockSize(16, 16);
  dim3 gridSize((ctx.W + blockSize.x - 1) / blockSize.x, (ctx.H + blockSize.y - 1) / blockSize.y);

  cuda_check(cudaEventRecord(ctx.inf_start[idx], ctx.s1), "EventRecord inf_start");

  // Choose motion detection algorithm based on context
  if (ctx.has_previous_frame[idx]) {
    // Use frame differencing motion detection
    motion_detection_kernel<<<gridSize, blockSize, 0, ctx.s1>>>(
        ctx.d_in[idx], ctx.d_prev[idx], ctx.d_out[idx], ctx.W, ctx.H, ctx.motion_threshold);
  } else {
    // Use background subtraction for first frame or when previous is not available
    background_subtraction_kernel<<<gridSize, blockSize, 0, ctx.s1>>>(
        ctx.d_in[idx], ctx.d_background[idx], ctx.d_out[idx], ctx.W, ctx.H, ctx.learning_rate,
        ctx.motion_threshold);
    ctx.has_previous_frame[idx] = true;
  }

  cuda_check(cudaGetLastError(), "motion detection kernel launch");

  // Compute motion statistics
  int stats_threads = 256;
  int stats_blocks = (pixels + stats_threads - 1) / stats_threads;
  size_t shared_mem_size = stats_threads * 2 * sizeof(float);

  motion_stats_kernel<<<stats_blocks, stats_threads, shared_mem_size, ctx.s1>>>(
      ctx.d_out[idx], ctx.d_stats[idx], ctx.W, ctx.H);
  cuda_check(cudaGetLastError(), "motion stats kernel launch");

  cuda_check(cudaEventRecord(ctx.inf_end[idx], ctx.s1), "EventRecord inf_end");

  // Copy motion data and stats back to host
  cuda_check(cudaEventRecord(ctx.d2h_start[idx], ctx.s1), "EventRecord d2h_start");
  cuda_check(cudaMemcpyAsync(ctx.h_out[idx], ctx.d_out[idx], pixels * sizeof(float),
                             cudaMemcpyDeviceToHost, ctx.s1),
             "MemcpyAsync D2H motion");
  cuda_check(cudaMemcpyAsync(ctx.h_stats[idx], ctx.d_stats[idx], stats_blocks * 2 * sizeof(float),
                             cudaMemcpyDeviceToHost, ctx.s1),
             "MemcpyAsync D2H stats");
  cuda_check(cudaEventRecord(ctx.d2h_end[idx], ctx.s1), "EventRecord d2h_end");

  cuda_check(cudaEventRecord(ctx.inf_done[idx], ctx.s1), "EventRecord inf_done");

  // Copy current frame to previous for next iteration
  cuda_check(cudaMemcpyAsync(ctx.d_prev[idx], ctx.d_in[idx], ctx.bytes_per_frame,
                             cudaMemcpyDeviceToDevice, ctx.s1),
             "MemcpyAsync curr to prev");
}
