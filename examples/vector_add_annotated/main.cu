#include <unistd.h>

#include <vector>

// This is just to check that `kernel_annotated.cu` is still valid C++/CUDA code
#include "kernel_annotated.cu"
#include "kernel_launcher.h"
#include "kernel_launcher/pragma.h"

namespace kl = kernel_launcher;

void cuda_check(cudaError_t code) {
    if (code != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") + cudaGetErrorString(code));
    }
}

std::string kernel_directory() {
    // Find kernel file
    std::string this_file = __FILE__;
    std::string this_directory = this_file.substr(0, this_file.rfind('/'));
    return this_directory + "/";
}

int main(int argc, char* argv[]) {
    chdir(kernel_directory().c_str());

    // Parse the number of elements N
    int n = 1'000'000;

    if (argc > 1) {
        char* end = nullptr;
        n = strtol(argv[1], &end, 10);

        if (strlen(end)) {
            std::cerr << "usage: " << argv[0] << " n\n";
            return 1;
        }
    }

    // Initialize inputs
    std::vector<float> A(n), B(n), C_answer(n), C_result(n);
    for (int i = 0; i < n; i++) {
        A[i] = static_cast<float>(i);
        B[i] = 1.0f;
        C_answer[i] = A[i] + B[i];
    }

    // Allocate GPU memory
    float *A_dev, *B_dev, *C_dev;
    cuda_check(cudaSetDevice(0));
    cuda_check(cudaMalloc(&A_dev, sizeof(float) * n));
    cuda_check(cudaMalloc(&B_dev, sizeof(float) * n));
    cuda_check(cudaMalloc(&C_dev, sizeof(float) * n));
    cuda_check(
        cudaMemcpy(A_dev, A.data(), sizeof(float) * n, cudaMemcpyDefault));
    cuda_check(
        cudaMemcpy(B_dev, B.data(), sizeof(float) * n, cudaMemcpyDefault));

    // Call kernel
    kl::launch(
        kl::PragmaKernel("vector_add", "kernel_annotated.cu", {"float"}),
        n,
        C_dev,
        (const float*)A_dev,
        (const float*)B_dev);

    // Copy results back
    cuda_check(cudaMemcpy(
        C_result.data(),
        C_dev,
        sizeof(float) * n,
        cudaMemcpyDefault));

    // Check results
    for (int i = 0; i < n; i++) {
        float result = C_result[i];
        float answer = C_answer[i];

        if (result != answer) {
            std::cout << "error: index " << i << " is incorrect: " << result
                      << " != " << answer << "\n";
            return 1;
        }
    }

    std::cout << "result correct\n";
    return 0;
}
