#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>
#include <cmath>

template <typename scalar_t>
__global__ void maxpool_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<int16_t,4,torch::RestrictPtrTraits> provenance,
    const int THREADS, const int STRIDE, const int K, const int W_OUT, const int PLANE) {
  // batch and c_out indicex.
  const int b = blockIdx.x;
  const int c = blockIdx.y;
  // height and width indices.
  const int cell_idx = blockIdx.z * THREADS + threadIdx.x;
  if (cell_idx >= PLANE) return;
  const int h_in = (cell_idx / W_OUT) * STRIDE;
  const int w_in = fmod(cell_idx, W_OUT) * STRIDE;
  const int h_out = (cell_idx / W_OUT);
  const int w_out = fmod(cell_idx, W_OUT);
  // Keep temporary storage for output and provenance.
  scalar_t out_ = -100.0;
  int max_p_ = -1;
  int idx = 0;
  // Loop over a kernel for a single pixel.
  for (int i=0; i < K; i++) {
    for (int j=0; j < K; j++) {
      if (input[b][c][h_in + i][w_in + j] > out_) {
        out_ = input[b][c][h_in + i][w_in + j];
        max_p_ = idx;
      }
      idx++;
    }
  }
  // Now assign to the correct output position and register provenance.
  output[b][c][h_out][w_out] = out_;
  provenance[b][c][h_out][w_out] = max_p_;
}

std::vector<torch::Tensor> maxpool_cuda_forward(
  torch::Tensor input, const int kernel_size, const int stride, const int device) {
  // Get the dimensions of the operation.
  const int B = input.size(0);
  const int C = input.size(1);
  const int H_IN = input.size(2);
  const int W_IN = input.size(3);
  // We need the output sizes of the operation. PyTorch has a way of cutting
  // off edges of the image if the sizes are odd. This is to replicate that behaviour.
  const int H_OUT = (H_IN + (stride > 1 ? 1 : 0)) / stride;
  const int W_OUT = (W_IN + (stride > 1 ? 1 : 0)) / stride;
  const int PLANE_SIZE = H_OUT * W_OUT;
  // Pad the input with a large minus value if the kernel has odd size.
  const int pad = kernel_size / 2;
  if (kernel_size % 2 == 0) {
    input = torch::constant_pad_nd(input, torch::IntList{0, pad, 0, pad}, -10);
  } else {
    input = torch::constant_pad_nd(input, torch::IntList{pad, pad, pad, pad}, -10);
  }
  // Initialize the output volume, compensated for the amount of stride.
  torch::Tensor output = torch::empty(torch::IntList{B, C, H_OUT, W_OUT}, torch::dtype(torch::kF32).device(torch::kCUDA, device));
  // Initialize a volume to track the provenance of maximum values.
  torch::Tensor provenance = torch::empty(torch::IntList{B, C, H_OUT, W_OUT}, torch::dtype(torch::kI16).device(torch::kCUDA, device));
  const int threads = 192;
  const int Z = (H_OUT * W_OUT + threads - 1) / threads;
  const dim3 blocks(B, C, Z);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool_cuda_forward", ([&] {
    maxpool_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        provenance.packed_accessor32<int16_t,4,torch::RestrictPtrTraits>(),
        threads, stride, kernel_size, W_OUT, PLANE_SIZE);
  }));

  cudaDeviceSynchronize();

  return {output, provenance};
}

template <typename scalar_t>
__global__ void parameterized_maxpool_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<int16_t,4,torch::RestrictPtrTraits> provenance,
    const int THREADS, const int STRIDE, const int K, const int W_OUT, const int PLANE) {
  // // batch and c_out indicex.
  const int b = blockIdx.x;
  const int c = blockIdx.y;
  // height and width indices.
  const int cell_idx = blockIdx.z * THREADS + threadIdx.x;
  if (cell_idx >= PLANE) return;
  const int h_in = (cell_idx / W_OUT) * STRIDE;
  const int w_in = fmod(cell_idx, W_OUT) * STRIDE;
  const int h_out = (cell_idx / W_OUT);
  const int w_out = fmod(cell_idx, W_OUT);

  extern __shared__ unsigned char kdata_uchar[];
  scalar_t *kdata = reinterpret_cast<scalar_t *>(kdata_uchar);
  // Fill the shared memory.
  int idx = 0;
  for (int i=0; i < K; i++) {
    for (int j=0; j < K; j++) {
      kdata[idx] = weights[c][i][j];
      idx++;
    }
  }
  __syncthreads();
  // Keep temporary storage for output and provenance.
  scalar_t out_ = -100.0;
  int max_p_ = -1;
  idx = 0;
  // Loop over a kernel for a single pixel.
  for (int i=0; i < K; i++) {
    for (int j=0; j < K; j++) {
      scalar_t val = input[b][c][h_in + i][w_in + j] + kdata[idx];
      if (val > out_) {
        out_ = val;
        max_p_ = idx;
      }
      idx++;
    }
  }
  // Now assign to the correct output position and register provenance.
  output[b][c][h_out][w_out] = out_;
  provenance[b][c][h_out][w_out] = max_p_;
}

std::vector<torch::Tensor> parameterized_maxpool_cuda_forward(
  torch::Tensor input, torch::Tensor weights, const int stride, const int device) {
  // Get the dimensions of the operation.
  const int B = input.size(0);
  const int C = input.size(1);
  const int H_IN = input.size(2);
  const int W_IN = input.size(3);
  const int K = weights.size(2);
  // We need the output sizes of the operation. PyTorch has a way of cutting
  // off edges of the image if the sizes are odd. This is to replicate that behaviour.
  const int H_OUT = (H_IN + (stride > 1 ? 1 : 0)) / stride;
  const int W_OUT = (W_IN + (stride > 1 ? 1 : 0)) / stride;
  const int PLANE_SIZE = H_OUT * W_OUT;
  const int pad = K / 2;
  if (K % 2 == 0) {
    input = torch::constant_pad_nd(input, torch::IntList{0, pad, 0, pad}, -10);
  } else {
    input = torch::constant_pad_nd(input, torch::IntList{pad, pad, pad, pad}, -10);
  }

  // Initialize the output volume, compensated for the amount of stride.
  torch::Tensor output = torch::empty(torch::IntList{B, C, H_OUT, W_OUT}, torch::dtype(torch::kF32).device(torch::kCUDA, device));
  // Initialize a volume to track the provenance of maximum values.
  torch::Tensor provenance = torch::empty(torch::IntList{B, C, H_OUT, W_OUT}, torch::dtype(torch::kI16).device(torch::kCUDA, device));

  const int threads = 192;
  const int Z = (H_OUT * W_OUT + threads - 1) / threads;
  const dim3 blocks(B, C, Z);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "parameterized_maxpool_cuda_forward", ([&] {
    parameterized_maxpool_forward_kernel<scalar_t><<<blocks, threads, K*K*sizeof(scalar_t)>>>(
        input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        provenance.packed_accessor32<int16_t,4,torch::RestrictPtrTraits>(),
        threads, stride, K, W_OUT, PLANE_SIZE);
  }));

  cudaDeviceSynchronize();

  return {output, provenance};
}
