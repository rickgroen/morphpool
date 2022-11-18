#include <torch/extension.h>


// C++ interface
// == To get a working cuda compiler tools (v 11+) ==
// https://linuxhint.com/install-cuda-ubuntu/
// Use this to get working keys: https://askubuntu.com/questions/1408016/the-following-signatures-couldnt-be-verified-because-the-public-key-is-not-avai
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
