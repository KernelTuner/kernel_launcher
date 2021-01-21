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

    CUdevice device = 0;

    CUcontext ctx;
    CU_CHECK(cuCtxCreate(&ctx, 0, device));

    try {
        KernelLauncher launcher("kernel.cu");
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}
