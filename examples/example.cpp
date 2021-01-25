#define NVRTC_GET_TYPE_NAME 1
#include <exception>
#include <iostream>
#include <cstdio>
#include <cuda.h>

#include "json.hpp"
#include "kernel_launcher.h"

using namespace kernel_launcher;

#define CU_CHECK(expr) cu_check(expr,  __FILE__, __LINE__, __func__);

void cu_check(CUresult code, const char* file, int line, const char* func) {
    if (code != CUDA_SUCCESS) {
        const char* name = "";
        const char* message = "";

        cuGetErrorName(code, &name);
        cuGetErrorString(code, &message);

        std::cerr << "ERROR:" << file << ":" << line << ":" << func << ": " 
            << name << ": " << message << std::endl;
        exit(1);
    }
}

int main() {
    CU_CHECK(cuInit(0));

    CUdevice device;
    CU_CHECK(cuDeviceGet(&device, 0));

    CUcontext ctx;
    CU_CHECK(cuCtxCreate(&ctx, 0, device));

    int n = 100;
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
    CU_CHECK(cuMemAlloc((CUdeviceptr*) &dev_A, n * sizeof(float)));
    CU_CHECK(cuMemAlloc((CUdeviceptr*) &dev_B, n * sizeof(float)));
    CU_CHECK(cuMemAlloc((CUdeviceptr*) &dev_C, n * sizeof(float)));

    CU_CHECK(cuMemcpy((CUdeviceptr) dev_A, (CUdeviceptr) A.data(), n * sizeof(float)));
    CU_CHECK(cuMemcpy((CUdeviceptr) dev_B, (CUdeviceptr) B.data(), n * sizeof(float)));


    typedef Kernel<float*, float*, float*, int> VectorAddKernel;

    auto config = Config::load_best_for_current_device("vector_add_results.json", "800000000", "GFLOP/s");
    auto vector_add = VectorAddKernel("vector_add", "vector_add.cu", config, {"-std=c++11"});

    vector_add(
            4,
            128, 
            dev_C,
            dev_A,
            dev_B,
            n
    );


    CU_CHECK(cuMemcpy((CUdeviceptr) C.data(), (CUdeviceptr) dev_C, n * sizeof(float)));

    for (int i = 0; i < n; i++) {
        printf("%d] %f + %f = %f\n", i, A[i], B[i], C[i]);
    }

    printf("keep on trucking\n");
    return 0;
}
