#include "catch.hpp"
#include "kernel_launcher/gpu_utils.hpp"
#include "test_helpers.hpp"

using namespace kernel_launcher;

// `set_value(uint32_t* ptr, uint32_t value)` writes `value` to `*ptr`.
static const char* SET_VALUE_PTX = R"(
.version 7.0
.target sm_50
.address_size 64

.visible .entry set_value(
    .param .u64 ptr,
    .param .u32 value
)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [ptr];
    ld.param.u32 %r1, [value];
    cvta.to.global.u64 %rd2, %rd1;
    st.global.u32 [%rd2], %r1;
    ret;
}
)";

TEST_CASE("GPUException") {
    GPUException e(CUDA_ERROR_INVALID_VALUE);
    CHECK(e.error_code() == CUDA_ERROR_INVALID_VALUE);
    CHECK_THAT(e.what(), Catch::Contains("[1]"));

    GPUException with_msg(CUDA_ERROR_INVALID_VALUE, "something failed");
    CHECK(with_msg.error_code() == CUDA_ERROR_INVALID_VALUE);
    CHECK_THAT(with_msg.what(), Catch::StartsWith("something failed ("));
}

TEST_CASE("LaunchConfig defaults") {
    LaunchConfig config {};
    CHECK(config.block_size[0] == 1);
    CHECK(config.block_size[1] == 1);
    CHECK(config.block_size[2] == 1);
    ;
    CHECK(config.grid_size[0] == 0);
    CHECK(config.grid_size[1] == 1);
    CHECK(config.grid_size[2] == 1);
    CHECK(config.shared_mem == 0);
    CHECK_FALSE(config.cooperative);
}

TEST_CASE("GPUDevice without device") {
    CHECK(GPUDevice().get() == -1);
    CHECK(GPUDevice(3).get() == 3);
}

TEST_CASE("GPUContextRef with null context") {
    CHECK_THROWS_AS(GPUContextRef(nullptr), std::invalid_argument);
}

TEST_CASE("gpu_runtime_version") {
    CHECK(gpu_runtime_version() > 0);
}

TEST_CASE("GPUDevice") {
    REQUIRE_GPU_OR_SKIP();

    CHECK(gpu_driver_version() > 0);
    CHECK(GPUDevice::count() > 0);

    GPUDevice device = GPUDevice::current();
    CHECK(device.get() >= 0);
    CHECK_FALSE(device.name().empty());
    CHECK(device.attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) > 0);
    CHECK_THROWS_AS(device.attribute(CUdevice_attribute(-1)), GPUException);

    CHECK_THROWS_AS(GPUDevice(-1).name(), GPUException);
}

TEST_CASE("GPUArch") {
    GPUArch fixed("sm_80", "my device");
    CHECK(fixed.get() == "sm_80");
    CHECK(fixed.name() == "my device");
    CHECK(GPUArch().get().empty());

    REQUIRE_GPU_OR_SKIP();

    GPUArch arch = GPUArch::current();
    GPUArch same = GPUArch::of(GPUDevice::current());

    CHECK_FALSE(arch.get().empty());
    CHECK(arch.get() == same.get());
    CHECK(arch.name() == GPUDevice::current().name());
}

TEST_CASE("GPUContextRef") {
    REQUIRE_GPU_OR_SKIP();

    CUcontext raw;
    REQUIRE(cuCtxGetCurrent(&raw) == CUDA_SUCCESS);

    GPUContextRef context = GPUContextRef::current();
    CHECK(context.get() == raw);
    CHECK(context == GPUContextRef(raw));
    CHECK(std::hash<GPUContextRef> {}(context) == std::hash<GPUContextRef> {}(GPUContextRef(raw)));
    CHECK(context.device().get() == GPUDevice::current().get());

    SECTION("other context") {
        CUcontext other;
        CUcontext popped;
        REQUIRE(cuCtxCreate(&other, 0, GPUDevice::current().get()) == CUDA_SUCCESS);
        REQUIRE(cuCtxPopCurrent(&popped) == CUDA_SUCCESS);
        REQUIRE(popped == other);

        GPUContextRef other_ref(other);
        CHECK(other_ref != context);
        CHECK(other_ref.id() != context.id());

        // Querying the device does not change the current context.
        CHECK(other_ref.device().get() == GPUDevice::current().get());
        CHECK(GPUContextRef::current() == context);

        {
            GPUContextGuard guard(other_ref);
            CHECK(GPUContextRef::current() == other_ref);
        }

        CHECK(GPUContextRef::current() == context);
        cuCtxDestroy(other);
    }
}

TEST_CASE("GPUKernel") {
    REQUIRE_GPU_OR_SKIP();

    std::shared_ptr<GPUKernel> kernel_ptr = GPUKernel::load_image(SET_VALUE_PTX, "set_value");
    const GPUKernel& kernel = *kernel_ptr;
    CHECK(kernel.module() != nullptr);
    CHECK(kernel.function() != nullptr);
    CHECK(kernel.name() == "set_value");

    SECTION("display name") {
        auto named = GPUKernel::load_image(SET_VALUE_PTX, "set_value", "my_kernel");
        CHECK(named->name() == "my_kernel");
    }

    SECTION("launch") {
        CUdeviceptr ptr;
        REQUIRE(cuMemAlloc(&ptr, sizeof(uint32_t)) == CUDA_SUCCESS);

        uint32_t value = 42;
        const void* args[] = {&ptr, &value};
        kernel.launch(nullptr, LaunchConfig {{1, 1, 1}, {1, 1, 1}}, args);
        gpu_synchronize();

        uint32_t result = 0;
        gpu_copy(&result, reinterpret_cast<const void*>(ptr), sizeof(uint32_t));
        gpu_synchronize();
        CHECK(result == 42);

        // Block size is too large.
        CHECK_THROWS_AS(
            kernel.launch(nullptr, LaunchConfig {{100000, 1, 1}, {1, 1, 1}}, args),
            GPUException);

        cuMemFree(ptr);
    }

    SECTION("attributes") {
        int max_threads = kernel.max_threads_per_block();
        CHECK(max_threads > 0);
        CHECK(kernel.attribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK) == max_threads);
        CHECK(kernel.num_regs() >= 0);

        int per_sm = kernel.max_active_blocks_per_sm(128);
        CHECK(per_sm > 0);
        CHECK(kernel.max_active_blocks(128) >= per_sm);

        GPUKernel::BlockSizeHint hint = kernel.optimal_threads_per_block();
        CHECK(hint.block_size > 0);
        CHECK(hint.min_grid_size > 0);
    }
}

TEST_CASE("GPUKernel::load_image errors") {
    REQUIRE_GPU_OR_SKIP();

    CHECK_THROWS_AS(GPUKernel::load_image("this is not ptx", "set_value"), GPUException);
    CHECK_THROWS_AS(GPUKernel::load_image(SET_VALUE_PTX, "missing"), GPUException);
}
