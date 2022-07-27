#include "kernel_launcher/kernel.h"

#include <cuda.h>

#include "catch.hpp"

using namespace kernel_launcher;

static constexpr const char* kernel_source = R"(
template <typename T>
__global__
void vector_add(int n, int *c, const int* a, const int* b) {
    for (int k = 0; k < ELEMENTS_PER_THREAD; k++) {
        int index = (blockIdx.x * ELEMENTS_PER_THREAD + k) * blockDim.x + threadIdx.x;

        if (index < n) {
            c[index] = a[index] + b[index];
        }
    }
}
)";

TEST_CASE("kernel builder", "[CUDA]") {
    CUcontext ctx;
    KERNEL_LAUNCHER_CUDA_CHECK(cuInit(0));
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxCreate(&ctx, 0, 0));

    std::string assets_dir = __FILE__;
    assets_dir = assets_dir.substr(0, assets_dir.rfind('/'));
    assets_dir += "/assets";

    KernelBuilder builder(
        "vector_add",
        KernelSource("vector_add.cu", kernel_source));
    auto tb = builder.tune("threads_per_block", {32, 128, 256});
    auto et = builder.tune("elements_per_thread", {1, 2, 4});
    auto eb = et * tb;

    builder.define("ELEMENTS_PER_THREAD", et)
        .template_args(type_of<int>())
        .block_size(tb)
        .grid_divisors(eb);

    std::vector<Config> configs;
    for (auto x : tb.parameter().values()) {
        for (auto y : et.parameter().values()) {
            Config config;
            config.insert(tb, x);
            config.insert(et, y);
            configs.push_back(config);
        }
    }

    uint n = 1000;
    std::vector<int> a(n);
    std::vector<int> b(n);
    std::vector<int> c(n);
    std::vector<int> c_ref(n);

    for (uint i = 0; i < n; i++) {
        a[i] = int(i);
        b[i] = 2 * int(i);
        c_ref[i] = 3 * int(i);
        c[i] = 0;
    }

    CUdeviceptr dev_a, dev_b, dev_c;
    KERNEL_LAUNCHER_CUDA_CHECK(cuMemAlloc(&dev_a, n * sizeof(int)));
    KERNEL_LAUNCHER_CUDA_CHECK(cuMemAlloc(&dev_b, n * sizeof(int)));
    KERNEL_LAUNCHER_CUDA_CHECK(cuMemAlloc(&dev_c, n * sizeof(int)));

    cuda_copy(a.data(), (int*)dev_a, n);
    cuda_copy(b.data(), (int*)dev_b, n);

    for (auto& config : configs) {
        // Overwrite C
        cuda_copy(a.data(), (int*)dev_c, n);

        Kernel<int, int*, const int*, const int*> kernel;
        CHECK_NOTHROW(kernel.compile(builder, config));
        CHECK_NOTHROW(kernel(n)(int(n), (int*)dev_c, (int*)dev_a, (int*)dev_b));

        // Copy C out
        cuda_copy((int*)dev_c, c.data(), n);

        REQUIRE(c == c_ref);
    }

    KERNEL_LAUNCHER_CUDA_CHECK(cuMemFree(dev_a));
    KERNEL_LAUNCHER_CUDA_CHECK(cuMemFree(dev_b));
    KERNEL_LAUNCHER_CUDA_CHECK(cuMemFree(dev_c));
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxDestroy(ctx));
}
