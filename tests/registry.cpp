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

struct MatrixMulDescriptor: IKernelDescriptor {
    MatrixMulDescriptor(int size) : size_(size) {}

    KernelBuilder build() const override {
        return KernelBuilder("matrix_mul", "TODO");
    }

    bool equals(const IKernelDescriptor& that) const override {
        if (auto ptr = dynamic_cast<const MatrixMulDescriptor*>(&that)) {
            return ptr->size_ == size_;
        } else {
            return false;
        }
    }

    hash_t hash() const override {
        return size_;
    }

    int size_;
};

TEST_CASE("KernelDescriptor", "[CUDA]") {
    CUcontext ctx, ctx2;
    KERNEL_LAUNCHER_CUDA_CHECK(cuInit(0));

    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxCreate(&ctx, 0, 0));
    auto a = KernelDescriptor(VectorAddDescriptor());
    auto b = KernelDescriptor(std::make_shared<MatrixMulDescriptor>(1));
    auto c = KernelDescriptor(std::shared_ptr<IKernelDescriptor>(
        std::make_shared<MatrixMulDescriptor>(1)));
    auto d = KernelDescriptor(MatrixMulDescriptor(2));

    // A KernelDescriptor is based on the current CUDA context.
    // Creating a new CUDA context here will mean that new descriptors will be
    // based on a different CUDA context than before.
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxCreate(&ctx2, 0, 0));
    auto e = KernelDescriptor(VectorAddDescriptor());

    CHECK(a == a);
    CHECK_FALSE(a == b);
    CHECK_FALSE(a == c);
    CHECK_FALSE(a == d);
    CHECK_FALSE(a == e);

    CHECK_FALSE(b == a);
    CHECK(b == b);
    CHECK(b == c);
    CHECK_FALSE(b == d);
    CHECK_FALSE(b == e);

    CHECK_FALSE(c == a);
    CHECK(c == b);
    CHECK(c == c);
    CHECK_FALSE(c == d);
    CHECK_FALSE(c == e);

    CHECK_FALSE(d == a);
    CHECK_FALSE(d == b);
    CHECK_FALSE(d == c);
    CHECK(d == d);
    CHECK_FALSE(d == e);

    CHECK_FALSE(e == a);
    CHECK_FALSE(e == b);
    CHECK_FALSE(e == c);
    CHECK_FALSE(e == d);
    CHECK(e == e);

    // These match the ones above
    CHECK(a.hash() == a.hash());
    CHECK_FALSE(a.hash() == b.hash());
    CHECK_FALSE(a.hash() == c.hash());
    CHECK_FALSE(a.hash() == d.hash());
    CHECK_FALSE(a.hash() == e.hash());

    CHECK_FALSE(b.hash() == a.hash());
    CHECK(b.hash() == b.hash());
    CHECK(b.hash() == c.hash());
    CHECK_FALSE(b.hash() == d.hash());
    CHECK_FALSE(b.hash() == e.hash());

    CHECK_FALSE(c.hash() == a.hash());
    CHECK(c.hash() == b.hash());
    CHECK(c.hash() == c.hash());
    CHECK_FALSE(c.hash() == d.hash());
    CHECK_FALSE(c.hash() == e.hash());

    CHECK_FALSE(d.hash() == a.hash());
    CHECK_FALSE(d.hash() == b.hash());
    CHECK_FALSE(d.hash() == c.hash());
    CHECK(d.hash() == d.hash());
    CHECK_FALSE(d.hash() == e.hash());

    CHECK_FALSE(e.hash() == a.hash());
    CHECK_FALSE(e.hash() == b.hash());
    CHECK_FALSE(e.hash() == c.hash());
    CHECK_FALSE(e.hash() == d.hash());
    CHECK(e.hash() == e.hash());

    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxDestroy(ctx));
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxDestroy(ctx2));
}

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
