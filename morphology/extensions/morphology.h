#include <torch/extension.h>
#include <vector>
#include <iostream>


// -- Pooling operations --
std::vector<torch::Tensor> maxpool_cuda_forward(
    torch::Tensor input, const int kernel_size, const int stride, const int device
);

torch::Tensor pool_cuda_backward(
    torch::Tensor delta_up, torch::Tensor provenance, const int K, const int stride, const int H_OUT, const int W_OUT, const int device
);

// -- Parameterized pooling operations --
std::vector<torch::Tensor> parameterized_maxpool_cuda_forward(
    torch::Tensor input, torch::Tensor weights, const int stride, const int device
);

torch::Tensor parameterized_pool_cuda_backward_h(
    torch::Tensor delta_up, torch::Tensor provenance, const int K, const int device
);

// -- Unpooling operations --
torch::Tensor unpool_cuda_forward(
    torch::Tensor input, torch::Tensor provenance, const int K, const int stride, const int H_OUT, const int W_OUT, const int device
);

torch::Tensor unpool_cuda_backward(
    torch::Tensor delta_up, torch::Tensor provenance, const int K, const int stride, const int device
);
