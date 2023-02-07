#include <cuda.h>

#include "catch.hpp"
#include "kernel_launcher/arg.h"
#include "test_utils.h"

using namespace kernel_launcher;

TEST_CASE("KernelBuilder compilation", "[CUDA]") {
    CUcontext ctx;
    KERNEL_LAUNCHER_CUDA_CHECK(cuInit(0));
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxCreate(&ctx, 0, 0));

    KernelBuilder builder = build_vector_add_kernel();
    auto tb = builder["threads_per_block"];
    auto et = builder["elements_per_thread"];

    std::vector<Config> configs;
    for (auto x : tb.parameter().values()) {
        for (auto y : et.parameter().values()) {
            Config config;
            config.insert(tb, x);
            config.insert(et, y);

            if (builder.is_valid(config)) {
                configs.push_back(config);
            }
        }
    }

    // There should be 9 configurations
    CHECK(configs.size() == 9);

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

    CudaVector<int> dev_a(n);
    CudaVector<int> dev_b(n);
    CudaVector<int> dev_c(n);

    cuda_copy(a.data(), (int*)dev_a, n);
    cuda_copy(b.data(), (int*)dev_b, n);

    for (auto& config : configs) {
        // Overwrite C
        cuda_copy(a.data(), (int*)dev_c, n);

        Kernel<int, int*, const int*, const int*> kernel;
        REQUIRE_NOTHROW(kernel.compile(builder, config));
        REQUIRE_NOTHROW(kernel(int(n), (int*)dev_c, (int*)dev_a, (int*)dev_b));

        // Copy C out
        cuda_copy((int*)dev_c, c.data(), n);

        REQUIRE(c == c_ref);
    }

    //    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxDestroy(ctx));
}
