#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>
#include <cmath>


template <typename scalar_t>
__global__ void unpool_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> delta_up,
    torch::PackedTensorAccessor32<int16_t,4,torch::RestrictPtrTraits> provenance,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> delta_down,
    const int THREADS, const int STRIDE, const int K, const int PLANE) {
  // batch and c_out indicex.
  const int b = blockIdx.x;
  const int c = blockIdx.y;
  // height and width indices.
  const int W_OUT = provenance.size(3);
  const int cell_idx = blockIdx.z * THREADS + threadIdx.x;
  if (cell_idx >= PLANE) return;
  const int h_out = (cell_idx / W_OUT);
  const int w_out = fmod(cell_idx, W_OUT);
  // Obtain the provenance for this element.
  const int k = (K - 1) / 2;
  const int prov = provenance[b][c][h_out][w_out];
  // Determine the location of the error in the delta_up, given the provenance.
  const int loc_y = (h_out * STRIDE) + (prov / K - k);
  const int loc_x = (w_out * STRIDE) + (prov % K - k);
  // If the index goes out of bounds, return. This can theoretically never happen.
  if (loc_y < 0 || loc_y >= delta_up.size(2) || loc_x < 0 || loc_x >= delta_up.size(3)) return;
  // Set the derivative w.r.t. the input to the element we found through provenance.
  delta_down[b][c][h_out][w_out] = delta_up[b][c][loc_y][loc_x];
}

torch::Tensor unpool_cuda_backward(
    torch::Tensor delta_up, torch::Tensor provenance, const int K, const int stride, const int device) {
  // Get the dimensions of the operation.
  const int B = delta_up.size(0);
  const int C = delta_up.size(1);
  const int H_OUT = provenance.size(2);
  const int W_OUT = provenance.size(3);
  const int PLANE_SIZE = H_OUT * W_OUT;

  // Initialize as zeros the delta_down volume (which is df+/df-).
  torch::Tensor delta_down = torch::zeros(torch::IntList{B, C, H_OUT, W_OUT}, torch::dtype(torch::kF32).device(torch::kCUDA, device));

  const int THREADS = 192;
  const int Z = (H_OUT * W_OUT + THREADS - 1) / THREADS;
  const dim3 blocks(B, C, Z);
  AT_DISPATCH_FLOATING_TYPES(delta_up.scalar_type(), "unpool_cuda_backward", ([&] {
    unpool_backward_kernel<scalar_t><<<blocks, THREADS>>>(
        delta_up.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        provenance.packed_accessor32<int16_t,4,torch::RestrictPtrTraits>(),
        delta_down.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        THREADS, stride, K, PLANE_SIZE);
  }));

  cudaDeviceSynchronize();

  return delta_down;
}
