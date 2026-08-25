#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernel_launcher.h"

namespace kl = kernel_launcher;

static void cuda_check(cudaError_t code) {
    if (code != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") + cudaGetErrorString(code));
    }
}

static std::string kernel_directory() {
    std::string self = __FILE__;
    return self.substr(0, self.rfind('/') + 1);
}

static const int MAX_BLOCKS_PER_SM = 8;

// One builder, two uses. `single_block` produces the final pass, which must
// land on exactly one block so that it writes exactly one float.
static kl::KernelBuilder build_reduce(bool single_block) {
    kl::KernelBuilder builder(
        "reduce",
        kernel_directory() + "reduce_kernel.cu");

    // The two passes see wildly different input sizes and grid shapes, so they
    // deserve separate wisdom.
    builder.tuning_key(single_block ? "reduce_final" : "reduce_partial");

    auto bs = builder.tune("block_size", {32, 64, 128, 256, 512, 1024}, 256);
    auto bpsm = builder.tune("blocks_per_sm", {1, 2, 3, 4, 5, 6, 7, 8}, 2);
    auto ipt = builder.tune("items_per_thread", {1, 2, 4, 8}, 4);
    auto vec = builder.tune("vector_size", {1, 2, 4}, 4);
    auto shuffle = builder.tune("use_shuffle", {0, 1}, 1);

    builder.restriction(bs <= kl::DEVICE_MAX_THREADS_PER_BLOCK);
    builder.restriction(bs * bpsm <= kl::DEVICE_MAX_THREADS_PER_MULTIPROCESSOR);

    auto [n, out, in] = kl::args<3>();

    builder.define(bs)
        .define(bpsm)
        .define(ipt)
        .define(vec)
        .define(shuffle)
        .problem_size(
            n)  // not used for the grid; this is the wisdom lookup key
        .block_size(bs)
        .buffer_size(in, n);

    if (single_block) {
        builder.grid_size(1u);
        builder.buffer_size(out, 1);
    } else {
        // The grid is sized to the machine, not to the problem. The kernel
        // strides over whatever input it is given.
        builder.grid_size(kl::DEVICE_MULTIPROCESSOR_COUNT * bpsm);
        builder.buffer_size(out, kl::DEVICE_MULTIPROCESSOR_COUNT * bpsm);
    }

    return builder;
}

int main() {
    const int n = 50'000'003;  // deliberately not a multiple of 4

    std::vector<float> in(n);
    double ref = 0;
    for (int i = 0; i < n; i++) {
        in[i] = float((i * 5) % 11);
        ref += in[i];
    }

    int num_sms = 0;
    cuda_check(cudaSetDevice(0));
    cuda_check(
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));

    // The tuner picks blocks_per_sm, so the host does not know how many
    // partials pass 1 will actually write. Allocate for the worst case and
    // zero the buffer: pass 2 then sums the whole thing and the unwritten
    // entries contribute nothing.
    const int max_blocks = num_sms * MAX_BLOCKS_PER_SM;

    float *in_dev, *partial_dev, *result_dev;
    cuda_check(cudaMalloc(&in_dev, sizeof(float) * size_t(n)));
    cuda_check(cudaMalloc(&partial_dev, sizeof(float) * size_t(max_blocks)));
    cuda_check(cudaMalloc(&result_dev, sizeof(float)));
    cuda_check(cudaMemcpy(
        in_dev,
        in.data(),
        sizeof(float) * size_t(n),
        cudaMemcpyDefault));
    cuda_check(cudaMemset(partial_dev, 0, sizeof(float) * size_t(max_blocks)));

    kl::WisdomKernel reduce_partial(build_reduce(false));
    kl::WisdomKernel reduce_final(build_reduce(true));

    reduce_partial(
        n,
        kl::cuda_span(partial_dev, size_t(max_blocks)),
        kl::cuda_span<const float>(in_dev, size_t(n)));

    reduce_final(
        max_blocks,
        kl::cuda_span(result_dev, size_t(1)),
        kl::cuda_span<const float>(partial_dev, size_t(max_blocks)));

    float result = 0.0f;
    cuda_check(cudaDeviceSynchronize());
    cuda_check(
        cudaMemcpy(&result, result_dev, sizeof(float), cudaMemcpyDefault));

    double rel = std::fabs(ref - result) / std::fabs(ref);
    std::printf("got %.1f, want %.1f (relative error %g)\n", result, ref, rel);

    cudaFree(in_dev);
    cudaFree(partial_dev);
    cudaFree(result_dev);
    return rel < 1e-5 ? 0 : 1;
}
