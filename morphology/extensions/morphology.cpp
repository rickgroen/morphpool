#include "morphology.h"
#include "utils.h"
#include <c10/cuda/CUDAGuard.h>


// -- Pooling operations --
std::vector<torch::Tensor> maxpool_forward(
    torch::Tensor input, const int kernel_size, const int stride, const int device) {
  CHECK_INPUT(input);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return maxpool_cuda_forward(input, kernel_size, stride, device);
}

torch::Tensor maxpool_backward(
    torch::Tensor delta_up, torch::Tensor provenance,
    const int kernel_size, const int stride, const int H_OUT,
    const int W_OUT, const int device) {
  CHECK_INPUT(delta_up);
  CHECK_INPUT(provenance);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(delta_up));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(delta_up));
  return pool_cuda_backward(delta_up, provenance, kernel_size, stride, H_OUT, W_OUT, device);
}

std::vector<torch::Tensor> parameterized_maxpool_forward(
    torch::Tensor input, torch::Tensor weights, const int stride, const int device) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return parameterized_maxpool_cuda_forward(input, weights, stride, device);
}

torch::Tensor parameterized_maxpool_backward_f(
    torch::Tensor delta_up, torch::Tensor provenance,
    const int kernel_size, const int stride, const int H_OUT,
    const int W_OUT, const int device) {
  CHECK_INPUT(delta_up);
  CHECK_INPUT(provenance);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(delta_up));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(provenance));
  return pool_cuda_backward(delta_up, provenance, kernel_size, stride, H_OUT, W_OUT, device);
}

torch::Tensor parameterized_maxpool_backward_h(
    torch::Tensor delta_up, torch::Tensor provenance, const int K, const int device) {
  CHECK_INPUT(delta_up);
  CHECK_INPUT(provenance);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(delta_up));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(provenance));
  return parameterized_pool_cuda_backward_h(delta_up, provenance, K, device);
}

// -- Unpooling operations --
torch::Tensor unpool_forward(
  torch::Tensor input, torch::Tensor provenance,
  const int kernel_size, const int stride, const int H_OUT,
  const int W_OUT, const int device) {
  CHECK_INPUT(input);
  CHECK_INPUT(provenance);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(provenance));
  return unpool_cuda_forward(input, provenance, kernel_size, stride, H_OUT, W_OUT, device);
}

torch::Tensor unpool_backward(
  torch::Tensor delta_up, torch::Tensor provenance, const int kernel_size,
  const int stride, const int device) {
  CHECK_INPUT(delta_up);
  CHECK_INPUT(provenance);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(delta_up));
  const at::cuda::OptionalCUDAGuard device_guard2(device_of(provenance));
  return unpool_cuda_backward(delta_up, provenance, kernel_size, stride, device);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("maxpool_forward", &maxpool_forward, "Standard (unparameterized) forward maxpool (CUDA)");
  m.def("maxpool_backward", &maxpool_backward, "Standard (unparameterized) backward maxpool (CUDA)");
  m.def("parameterized_maxpool_forward", &parameterized_maxpool_forward, "Parameterized maxpool forward (CUDA)");
  m.def("parameterized_maxpool_backward_f", &parameterized_maxpool_backward_f, "Parameterized maxpool backward for signal (CUDA)");
  m.def("parameterized_maxpool_backward_h", &parameterized_maxpool_backward_h, "Parameterized maxpool backward for weights (CUDA)");
  m.def("unpool_forward", &unpool_forward, "(unparameterized) forward unpool operation (CUDA)");
  m.def("unpool_backward", &unpool_backward, "backward unpool operation (CUDA)");
}
