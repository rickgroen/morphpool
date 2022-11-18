#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void check_cuda_error() {
    cudaError err = cudaGetLastError();
    if(err != cudaSuccess) {
	AT_ERROR("Cuda error=", err, " : ", cudaGetErrorString(err));
    }
}
