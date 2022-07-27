#include "kernel_launcher/wisdom.h"

#include "catch.hpp"

using namespace kernel_launcher;

TEST_CASE("test load_best_config") {
    std::string wisdom_dir = __FILE__;
    wisdom_dir = wisdom_dir.substr(0, wisdom_dir.rfind('/'));
    wisdom_dir += "/assets";

    ConfigSpace space;
    space.tune("block_size_x", {1, 2, 3, 4, 5, 6, 7, 8});
    space.tune("block_size_y", {1, 2});

    std::string device_name = "GTX fictional";
    CudaArch device_arch = 80;
    ProblemSize problem_size = {100, 100};
    WisdomResult result;
    std::string tuning_key = "example_kernel";

    SECTION("test not found") {
        Config config = load_best_config(
            wisdom_dir,
            "does_not_exist",
            space,
            device_name,
            device_arch,
            problem_size,
            &result);

        CHECK(result == WisdomResult::NotFound);
        CHECK(config["block_size_x"] == 1);
        CHECK(config["block_size_y"] == 1);
    }

    SECTION("test wrong device") {
        Config config = load_best_config(
            wisdom_dir,
            tuning_key,
            space,
            "unknown device",
            device_arch,
            problem_size,
            &result);

        CHECK(result == WisdomResult::DeviceMismatch);
        CHECK(config["block_size_x"] == 2);
        CHECK(config["block_size_y"] == 1);
    }

    SECTION("test valid configuration") {
        Config config = load_best_config(
            wisdom_dir,
            tuning_key,
            space,
            device_name,
            device_arch,
            problem_size,
            &result);

        CHECK(result == WisdomResult::Ok);
        CHECK(config["block_size_x"] == 3);
        CHECK(config["block_size_y"] == 1);
    }

    SECTION("test valid configuration") {
        Config config = load_best_config(
            wisdom_dir,
            tuning_key,
            space,
            "unknown device",
            device_arch,
            {200, 199},
            &result);

        CHECK(result == WisdomResult::DeviceMismatch);
        CHECK(config["block_size_x"] == 4);
        CHECK(config["block_size_y"] == 1);
    }

    SECTION("test wrong problem size") {
        Config config = load_best_config(
            wisdom_dir,
            tuning_key,
            space,
            device_name,
            device_arch,
            {200, 200},
            &result);

        CHECK(result == WisdomResult::ProblemSizeMismatch);
        CHECK(config["block_size_x"] == 6);
        CHECK(config["block_size_y"] == 1);
    }
}

TEST_CASE("test KernelArg") {
    SECTION("scalar int") {
        KernelArg v = into_kernel_arg(5);
        CHECK(v.type() == type_of<int>());
        CHECK(
            v.to_bytes() == std::vector<uint8_t> {5, 0, 0, 0});  // Assuming LE?
        CHECK(v.is_array() == false);
        CHECK(v.is_scalar() == true);

        int result;
        ::memcpy(&result, v.as_void_ptr(), sizeof(int));
        CHECK(result == 5);

        CHECK(v.to<int>() == 5);
        CHECK_THROWS(v.to<double>());
    }

    SECTION("scalar point3") {
        struct point3 {
            unsigned long long x;
            unsigned long long y;
            unsigned long long z;

            bool operator==(point3 v) const {
                return x == v.x && y == v.y && z == v.z;
            }
        };

        point3 input = {1, 2, 3};

        KernelArg v = into_kernel_arg(input);
        CHECK(v.type() == type_of<point3>());
        CHECK(
            v.to_bytes()
            == std::vector<uint8_t> {
                1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
                0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0});  // Assuming LE?
        CHECK(v.is_array() == false);
        CHECK(v.is_scalar() == true);

        point3 result;
        ::memcpy(&result, v.as_void_ptr(), sizeof(point3));
        CHECK(result == input);

        CHECK(v.to<point3>() == input);
        CHECK_THROWS(v.to<double>());
    }

    SECTION("array int*") {
        std::vector<int> input = {1, 2, 3};

        KernelArg v = KernelArg::for_array(input.data(), input.size());
        CHECK(v.type() == type_of<int*>());
        //        // This only works if cuda is enabled :-(
        //        CHECK(
        //            v.to_bytes()
        //            == std::vector<
        //                uint8_t> {1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0});  // Assuming LE?
        CHECK(v.is_array() == true);
        CHECK(v.is_scalar() == false);
        CHECK(*(int**)(v.as_void_ptr()) == input.data());

        CHECK(v.to<int*>() == input.data());
        CHECK(v.to<const int*>() == input.data());
        CHECK_THROWS(v.to<double>());
    }

    SECTION("array const int*") {
        std::vector<int> input = {1, 2, 3};

        KernelArg v =
            KernelArg::for_array<const int>(input.data(), input.size());
        CHECK(v.type() == type_of<const int*>());
        CHECK(v.is_array() == true);
        CHECK(v.is_scalar() == false);
        CHECK(*(const int**)(v.as_void_ptr()) == input.data());

        CHECK_THROWS(v.to<int*>() == input.data());
        CHECK(v.to<const int*>() == input.data());
        CHECK_THROWS(v.to<double>());
    }
}

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

TEST_CASE("WisdomKernel", "[CUDA]") {
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

    WisdomSettings wisdom_settings(assets_dir, assets_dir);
    WisdomKernel kernel(
        "vector_add",
        builder,
        default_compiler(),
        wisdom_settings);

    uint n = 10;
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
    cuda_copy(c.data(), (int*)dev_c, n);

    REQUIRE_NOTHROW(
        kernel(n)(int(n), (int*)dev_c, (const int*)dev_a, (const int*)dev_b));

    // Copy C out
    cuda_copy((int*)dev_c, c.data(), n);

    REQUIRE(c == c_ref);

    KERNEL_LAUNCHER_CUDA_CHECK(cuMemFree(dev_a));
    KERNEL_LAUNCHER_CUDA_CHECK(cuMemFree(dev_b));
    KERNEL_LAUNCHER_CUDA_CHECK(cuMemFree(dev_c));
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxDestroy(ctx));
}
