#include "catch.hpp"
#include "kernel_launcher/registry.hpp"

using namespace kernel_launcher;

static int build_count = 0;

static KernelBuilder make_kernel() {
    build_count++;
    return KernelBuilder("kernel", FileSource("kernel.cu", ""));
}

static KernelBuilder make_kernel_with_args(int size, bool flag) {
    build_count++;
    return KernelBuilder(concat("kernel_", size, "_", flag), FileSource("kernel.cu", ""));
}

// Same hash as every other instance, so equality must use `typeid` and `equals`.
struct ConstantDescriptor: KernelDescriptor {
    KernelBuilder build() const override {
        return make_kernel();
    }

    bool equals(const KernelDescriptor&) const override {
        return true;
    }

    size_t hash() const override {
        return 0;
    }
};

struct OtherConstantDescriptor: ConstantDescriptor {};

TEST_CASE("FnKernelDescriptor") {
    FnKernelDescriptor a(make_kernel_with_args, 1, true);
    FnKernelDescriptor b(make_kernel_with_args, 1, true);
    FnKernelDescriptor c(make_kernel_with_args, 2, true);

    CHECK(a.equals(b));
    CHECK_FALSE(a.equals(c));
    CHECK(a.hash() == b.hash());

    build_count = 0;
    KernelBuilder builder = a.build();
    CHECK(build_count == 1);
}

TEST_CASE("fn_kernel") {
    DescriptorKey a(fn_kernel(make_kernel_with_args, 1, true));
    DescriptorKey b(FnKernelDescriptor(make_kernel_with_args, 1, true));
    DescriptorKey c(fn_kernel(make_kernel_with_args, 2, true));

    CHECK(a == b);
    CHECK(a != c);
    CHECK(DescriptorKey(fn_kernel(make_kernel)) == DescriptorKey(make_kernel));
}

TEST_CASE("DescriptorKey") {
    DescriptorKey a(make_kernel);
    DescriptorKey b {FnKernelDescriptor(make_kernel)};
    DescriptorKey c(FnKernelDescriptor(make_kernel_with_args, 1, true));
    DescriptorKey d(FnKernelDescriptor(make_kernel_with_args, 1, true));

    CHECK(a == b);
    CHECK(a.hash() == b.hash());
    CHECK(a != c);
    CHECK(c == d);

    // Equal hashes, but different types.
    DescriptorKey e(ConstantDescriptor {});
    DescriptorKey f(OtherConstantDescriptor {});
    CHECK(e == DescriptorKey(ConstantDescriptor {}));
    CHECK(e != f);

    CHECK(dynamic_cast<const ConstantDescriptor*>(&e.get()) != nullptr);
}

TEST_CASE("Registry") {
    Registry registry;
    build_count = 0;

    // Each descriptor is only built once.
    registry.lookup(make_kernel);
    registry.lookup(make_kernel);
    CHECK(build_count == 1);

    registry.lookup(FnKernelDescriptor(make_kernel_with_args, 1, true));
    registry.lookup(FnKernelDescriptor(make_kernel_with_args, 1, true));
    CHECK(build_count == 2);

    registry.lookup(FnKernelDescriptor(make_kernel_with_args, 2, true));
    CHECK(build_count == 3);

    // Copies share the same kernels.
    Registry copy = registry;
    copy.lookup(make_kernel);
    CHECK(build_count == 3);

    // A new registry does not.
    Registry other;
    other.lookup(make_kernel);
    CHECK(build_count == 4);
}

TEST_CASE("default_registry") {
    CHECK(&default_registry() == &default_registry());
}
