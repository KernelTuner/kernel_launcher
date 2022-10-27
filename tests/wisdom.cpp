#include "catch.hpp"
#include "kernel_launcher/wisdom_kernel.h"
#include "test_utils.h"

using namespace kernel_launcher;

struct point3 {
    unsigned long long x;
    unsigned long long y;
    unsigned long long z;

    bool operator==(point3 v) const {
        return x == v.x && y == v.y && z == v.z;
    }
};

namespace kernel_launcher {
template<>
struct IntoKernelArg<point3> {
    static KernelArg convert(point3 value) {
        return KernelArg::for_scalar<point3>(value);
    }
};
}  // namespace kernel_launcher

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

    SECTION("test invalid key") {
        // The file name should still be `example_kernel.wisdom` since
        // the `$` is replace by `_`. But key in the file will be incorrect.
        Config config = load_best_config(
            wisdom_dir,
            "example$kernel",
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

    SECTION("test wisdom file missing parameters") {
        // Expand parameter space. This will not be in the wisdom file.
        space.tune("block_size_z", {1024});

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
        CHECK(config["block_size_z"] == 1024);
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
        CHECK(v.to_value() == TunableValue(5));
        CHECK_THROWS(v.to<double>());
    }

    SECTION("scalar point3") {
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

    SECTION("to_value") {
        CHECK(KernelArg::for_scalar<signed char>(13).to_value() == 13);
        CHECK(KernelArg::for_scalar<unsigned char>(14).to_value() == 14);
        CHECK(KernelArg::for_scalar<char>(15).to_value() == 15);
        CHECK(KernelArg::for_scalar<int>(16).to_value() == 16);
        CHECK(KernelArg::for_scalar<double>(17).to_value() == 17.0);

        // `const` should not matter
        CHECK(KernelArg::for_scalar<const int>(18).to_value() == 18);

        // These should fail
        CHECK_THROWS(KernelArg::for_scalar<point3>({1, 2, 3}).to_value());
        CHECK_THROWS(KernelArg::for_array<int>(nullptr, 2).to_value());
    }
}

TEST_CASE("WisdomKernel", "[CUDA]") {
    CUcontext ctx;
    KERNEL_LAUNCHER_CUDA_CHECK(cuInit(0));
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxCreate(&ctx, 0, 0));

    std::string assets_dir = assets_directory();
    KernelBuilder builder = build_vector_add_kernel();

    WisdomSettings wisdom_settings(assets_dir, assets_dir);
    WisdomKernel kernel(
        "vector_add",
        builder,
        default_compiler(),
        wisdom_settings);

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
        kernel(n)(int(n), (int*)dev_c, (const int*)dev_a, (const int*)dev_b));

    // Copy C out
    cuda_copy((int*)dev_c, c.data(), n);

    REQUIRE(c == c_ref);

    //    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxDestroy(ctx));
}
