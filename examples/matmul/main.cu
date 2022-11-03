#include "kernel_launcher.h"

namespace kl = kernel_launcher;

void cuda_check(cudaError_t code) {
    if (code != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") + cudaGetErrorString(code));
    }
}

int main() {
    using TF = float;
    int N = 4096;

    std::vector<TF> A(N * N);
    std::vector<TF> B(N * N);

    for (size_t i = 0; i < N * N; i++) {
        A[i] = TF(i % 7);
        B[i] = TF(i % 13);
    }

    std::string this_file = __FILE__;
    std::string this_directory = this_file.substr(0, this_file.rfind('/'));

    kl::KernelBuilder builder("matmul_kernel", this_directory + "/matmul.cu");

    auto bx = builder.tune("block_size_x", {16, 32, 48}, 32);
    auto by = builder.tune("block_size_y", {1, 2, 4, 8, 16, 32}, 32);
    auto tx = builder.tune("tile_size_x", {1, 2, 4, 8});
    auto ty = builder.tune("tile_size_y", {1, 2, 4, 8});
    auto sm = builder.tune("blocks_per_sm", {0, 1, 2, 3, 4, 5, 6, 7, 8});

    builder.restriction(bx == by * ty);
    builder.restriction(bx * by <= kl::DEVICE_MAX_THREADS_PER_BLOCK);
    builder.restriction(
        sm * bx * by <= kl::DEVICE_MAX_THREADS_PER_MULTIPROCESSOR);

    builder.define(bx)
        .define(by)
        .define(tx)
        .define(ty)
        .problem_size(N, N)
        .block_size(bx, by)
        .grid_divisors(bx * tx, by * ty);

    kl::WisdomKernel matmul(builder);

    // Allocate GPU memory
    float *A_dev, *B_dev, *C_dev;
    cuda_check(cudaSetDevice(0));
    cuda_check(cudaMalloc(&A_dev, sizeof(TF) * N * N));
    cuda_check(cudaMalloc(&B_dev, sizeof(TF) * N * N));
    cuda_check(cudaMalloc(&C_dev, sizeof(TF) * N * N));
    cuda_check(
        cudaMemcpy(A_dev, A.data(), sizeof(TF) * N * N, cudaMemcpyDefault));
    cuda_check(
        cudaMemcpy(B_dev, B.data(), sizeof(TF) * N * N, cudaMemcpyDefault));

    // Call kernel
    matmul(
        kl::cuda_span(C_dev, N * N),
        kl::cuda_span(A_dev, N * N),
        kl::cuda_span(B_dev, N * N));

    return 0;
}
