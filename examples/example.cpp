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
    int *old_values;
    float *new_values;
    CU_CHECK(cuMemAlloc((CUdeviceptr*) &old_values, n * sizeof(int)));
    CU_CHECK(cuMemAlloc((CUdeviceptr*) &new_values, n * sizeof(int)));

    typedef Kernel<int, int*, float*> StencilKernel;

    auto config = Config::load_best_for_current_device("vector_add_results.json", "800000000", "GFLOP/s");
    auto stencil = StencilKernel("stencil", "kernel.cu", config, {"-std=c++11"});

    stencil(
            1024, 
            16, 
            n,
            old_values,
            new_values
    );

    std::vector<int> old_vals(n);
    std::vector<int> new_vals(n);

    CU_CHECK(cuMemcpy((CUdeviceptr) old_vals.data(), (CUdeviceptr) old_values, n * sizeof(int)));
    CU_CHECK(cuMemcpy((CUdeviceptr) new_vals.data(), (CUdeviceptr) new_values, n * sizeof(int)));

    for (int i = 0; i < n; i++) {
        printf("%d] %d %d\n", i, new_vals[i], old_vals[i]);
    }

    printf("keep on trucking\n");
}
