#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>
#include <cmath>


template <typename scalar_t>
__global__ void parameterized_pool_backward_h_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> delta_up,
    torch::PackedTensorAccessor32<int16_t,4,torch::RestrictPtrTraits> provenance,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> delta_weights,
    const int THREADS, const int K) {
  // kernel dimensions in x and y.
  const int kx = blockIdx.x;
  const int ky = blockIdx.y;
  const int c = blockIdx.z * THREADS + threadIdx.x;
  if (c >= provenance.size(1)) return;
  // Get dimensions provenance.
  const int B = provenance.size(0);
  const int H = provenance.size(2);
  const int W = provenance.size(3);
  // Loop over batches, height and weight to sum all deltas for ky, kx.
  double_t out_ = 0.0;
  int16_t idx = kx + (ky * K);
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        if (provenance[b][c][h][w] == idx) {
          out_ += static_cast<double_t>(delta_up[b][c][h][w]);
        }
      }
    }
  }
  delta_weights[c][ky][kx] = static_cast<scalar_t>(out_);
}

torch::Tensor parameterized_pool_cuda_backward_h(
    torch::Tensor delta_up, torch::Tensor provenance, const int K, const int device) {
  // Get the dimensions of the operation.
  const int C = delta_up.size(1);
  // Initialize as zeros the delta_kernel volume (which is df+/dh).
  torch::Tensor delta_weights = torch::zeros(torch::IntList{C, K, K}, torch::dtype(torch::kF32).device(torch::kCUDA, device));

  const int THREADS = 32;
  const int Z = (C + THREADS - 1) / THREADS;
  const dim3 blocks(K, K, Z);
  AT_DISPATCH_FLOATING_TYPES(delta_up.scalar_type(), "parameterized_pool_cuda_backward_h", ([&] {
    parameterized_pool_backward_h_kernel<scalar_t><<<blocks, THREADS>>>(
        delta_up.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        provenance.packed_accessor32<int16_t,4,torch::RestrictPtrTraits>(),
        delta_weights.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        THREADS, K);
  }));

  cudaDeviceSynchronize();

  return delta_weights;
}
