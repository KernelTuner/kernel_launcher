#include <vector>

#include "kernel_launcher.h"

namespace kl = kernel_launcher;

void cuda_check(cudaError_t code) {
    if (code != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") + cudaGetErrorString(code));
    }
}

kl::KernelBuilder build_vector_add() {
    // Find kernel file
    std::string this_file = __FILE__;
    std::string this_directory = this_file.substr(0, this_file.rfind('/'));
    std::string kernel_file = this_directory + "/kernel.cu";

    // Tunable parameters
    kl::KernelBuilder builder("vector_add", kernel_file);
    auto block_size =
        builder.tune("threads_per_block", {32, 64, 128, 256, 512, 1024});
    auto granularity =
        builder.tune("elements_per_thread", {1, 2, 3, 4, 5, 6, 7, 8});
    auto strategy = builder.tune("tiling_strategy", {0, 1, 2});

    // Set options
    builder.template_args(block_size, granularity, strategy)
        .block_size(block_size)
        .grid_divisors(block_size * granularity);

    return builder;
}

int main(int argc, char* argv[]) {
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

    // Create wisdom kernel
    kl::WisdomKernel vector_add("vector_add", build_vector_add());

    // Call kernel
    vector_add(n)(
        n,
        kl::CudaSpan<float>(C_dev, n),
        kl::CudaSpan<const float>(A_dev, n),
        kl::CudaSpan<const float>(B_dev, n));

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