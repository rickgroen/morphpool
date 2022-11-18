#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>
#include <cmath>


template <typename scalar_t>
__global__ void pool_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> delta_up,
    torch::PackedTensorAccessor32<int16_t,4,torch::RestrictPtrTraits> provenance,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> delta_down,
    const int THREADS, const int STRIDE, const int K, const int W_OUT, const int PLANE) {
  // batch and c_out indicex.
  const int b = blockIdx.x;
  const int c = blockIdx.y;
  // height and width indices.
  const int cell_idx = blockIdx.z * THREADS + threadIdx.x;
  if (cell_idx >= PLANE) return;
  const int h_out = (cell_idx / W_OUT);
  const int w_out = fmod(cell_idx, W_OUT);
  // In even-sized kernels the centre pixel is shifted compared to odd-sized kernels.
  const int k = K / 2;
  // Keep a record for the size of provenance, to never go outside bounds.
  const int PY = provenance.size(2);
  const int PX = provenance.size(3);
  // This is the (h_out, w_out) pixel location which we match to the provenance.
  int original_location = cell_idx;
  // Record the summed back-propped error as a double to deal with precision problems.
  double_t out_ = 0.0;
  // These variables are here to obtain the (h_out, w_out) pixel location
  // from the location of the provenance map and the value in the provenance map.
  int prov_i, prov_j, prov_ij, prov_ki, prov_kj, matched_location;
  for (int i=0; i < K; i += STRIDE) {
    for (int j=0; j < K; j += STRIDE) {
      // Determine which provenance location we should examine.
      // First, determine the h_in, w_in, then center around the kernel with (k / STRIDE).
      prov_i = (h_out + i) / STRIDE - (k / STRIDE);
      prov_j = (w_out + j) / STRIDE - (k / STRIDE);
      // If that goes out of bounds, continue.
      if (prov_i < 0 || prov_i >= PY || prov_j < 0 || prov_j >= PX) {
        continue;
      }
      // Read the provenance in the provenance map at that location.
      prov_ij = provenance[b][c][prov_i][prov_j];
      // This provenance value is kernel-centric, we go back to full spatial coordinates. Compute the offset.
      prov_ki = prov_ij / K - ((K % 2 == 1) ? k : 0);
      prov_kj = prov_ij % K - ((K % 2 == 1) ? k : 0);
      // Using the offset compute the actual spatial coordinates in (h_out, w_out).
      matched_location = ((prov_i * STRIDE) + prov_ki) * W_OUT + ((prov_j * STRIDE) + prov_kj);
      // And if these match our pixel in this thread, we save the delta_up value.
      if (matched_location == original_location) {
        out_ += static_cast<double_t>(delta_up[b][c][prov_i][prov_j]);
      }
    }
  }
  // Only write if necessary.
  if (out_ != 0.0) {
    delta_down[b][c][h_out][w_out] = static_cast<scalar_t>(out_);
  }
}

torch::Tensor pool_cuda_backward(
    torch::Tensor delta_up, torch::Tensor provenance,
    const int K, const int stride, const int H_OUT, const int W_OUT, const int device) {
  // Get the dimensions of the operation.
  const int B = delta_up.size(0);
  const int C = delta_up.size(1);
  const int H_IN = delta_up.size(2);
  const int W_IN = delta_up.size(3);
  const int PLANE_SIZE = H_OUT * W_OUT;

  // Initialize as zeros the delta_down volume (which is df+/df-).
  torch::Tensor delta_down = torch::zeros(torch::IntList{B, C, H_OUT, W_OUT}, torch::dtype(torch::kF32).device(torch::kCUDA, device));

  const int THREADS = 192;
  const int Z = (H_OUT * W_OUT + THREADS - 1) / THREADS;
  const dim3 blocks(B, C, Z);
  AT_DISPATCH_FLOATING_TYPES(delta_up.scalar_type(), "pool_cuda_backward", ([&] {
    pool_backward_kernel<scalar_t><<<blocks, THREADS>>>(
        delta_up.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        provenance.packed_accessor32<int16_t,4,torch::RestrictPtrTraits>(),
        delta_down.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        THREADS, stride, K, W_OUT, PLANE_SIZE);
  }));

  cudaDeviceSynchronize();

  return delta_down;
}
