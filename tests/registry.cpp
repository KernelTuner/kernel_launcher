#include "kernel_launcher/registry.h"

#include "catch.hpp"
#include "test_utils.h"

using namespace kernel_launcher;

struct VectorAddDescriptor: IKernelDescriptor {
    KernelBuilder build() const override {
        return build_vector_add_kernel();
    }

    bool equals(const IKernelDescriptor& that) const override {
        return dynamic_cast<const VectorAddDescriptor*>(&that) != nullptr;
    }
};

TEST_CASE("KernelRegistry", "[CUDA]") {
    CUcontext ctx;
    KERNEL_LAUNCHER_CUDA_CHECK(cuInit(0));
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxCreate(&ctx, 0, 0));

    std::string assets_dir = assets_directory();
    WisdomSettings wisdom_settings(assets_dir, assets_dir);
    KernelRegistry registry(default_compiler(), wisdom_settings);

    uint n = 10;
    std::vector<int> a(n), b(n), c(n), c_ref(n);

    for (uint i = 0; i < n; i++) {
        a[i] = int(i);
        b[i] = 2 * int(i);
        c_ref[i] = 3 * int(i);
        c[i] = 0;
    }

    CudaVector<int> dev_a(n), dev_b(n), dev_c(n);
    cuda_copy(a.data(), (int*)dev_a, n);
    cuda_copy(b.data(), (int*)dev_b, n);
    cuda_copy(c.data(), (int*)dev_c, n);

    REQUIRE_NOTHROW(
        registry.lookup(VectorAddDescriptor {})
            .launch(int(n), (int*)dev_c, (const int*)dev_a, (const int*)dev_b));

    cudaDeviceSynchronize();

    // Copy C out
    cuda_copy((int*)dev_c, c.data(), n);

    REQUIRE(c == c_ref);

    //    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxDestroy(ctx));
}
