from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="morphpool_cuda",
    ext_modules=[
        CUDAExtension(name="morphpool_cuda",
                      sources=["morphology.cpp",
                               # Pooling operations.
                               "operations/pool_forward.cu",
                               "operations/pool_backward_f.cu",
                               "operations/pool_backward_h.cu",
                               # Unpooling operations.
                               "operations/unpool_forward.cu",
                               "operations/unpool_backward_f.cu"],
                      extra_compile_args=['-g'])
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
