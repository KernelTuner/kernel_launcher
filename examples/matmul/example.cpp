#define NVRTC_GET_TYPE_NAME 1
#include <exception>
#include <iostream>
#include <cstdio>
#include <cuda.h>

#include "json.hpp"
#include "kernel_launcher.h"

using namespace kernel_launcher;

#define CUDA_CHECK(expr) cuda_check(expr,  __FILE__, __LINE__, __func__);

void cuda_check(cudaError_t code, const char* file, int line, const char* func) {
    if (code != cudaSuccess) {
        const char* name = cudaGetErrorName(code);
        const char* message = cudaGetErrorString(code);

        std::cerr << "ERROR:" << file << ":" << line << ":" << func << ": "
            << name << ": " << message << std::endl;
        exit(1);
    }
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    int n = 4096*4096;
    std::vector<float> A(n);
    std::vector<float> B(n);
    std::vector<float> C(n);

    for (int i = 0; i < n; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
        C[i] = 0;
    }

    float *dev_A;
    float *dev_B;
    float *dev_C;
    CUDA_CHECK(cudaMalloc((void**) &dev_A, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &dev_B, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &dev_C, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy((void*) dev_A, (void*) A.data(), n * sizeof(float), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy((void*) dev_B, (void*) B.data(), n * sizeof(float), cudaMemcpyDefault));

    auto matmul = CudaKernel<float*, float*, float*>::compile_best_for_current_device(
            "matmul_results.json", "4096x4096", "matmul.cu", {"-std=c++11"});

    dim3 grid(4096 / matmul.get_block_dim().x, 4096 / matmul.get_block_dim().y, 1);

    matmul(grid)(dev_C, dev_A, dev_B);

    // Or launch on a stream:
    //   matmul(grid_size, 0, stream)(dev_C, dev_A, dev_B, n);
    //
    // Alternative way to call kernel is:
    //   matmul.configure(grid_size).launch(dev_C, dev_A, dev_B, n);
    //
    // Or using a stream:
    //   matmul.configure_async(grid_size, 0, stream).launch(dev_C, dev_A, dev_B, n);

    CUDA_CHECK(cudaMemcpy((void*) C.data(), (void*) dev_C, n * sizeof(float), cudaMemcpyDefault));

    //for (int i = 0; i < n; i++) {
    //    printf("%d] %f + %f = %f\n", i, A[i], B[i], C[i]);
    //}

    printf("keep on trucking\n");
    return 0;
}
