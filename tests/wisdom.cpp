#include "catch.hpp"
#include "kernel_launcher/kernel.h"
#include "test_utils.h"

using namespace kernel_launcher;

TEST_CASE("test load_best_config") {
    std::string wisdom_dir = assets_directory();
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

    SECTION("test wrong device name") {
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
}

TEST_CASE("test process_wisdom_file") {
    std::string wisdom_dir = assets_directory();
    ConfigSpace space;
    auto block_size_x = space.tune("block_size_x", {1, 2, 3, 4, 5, 6, 7, 8});
    auto block_size_y = space.tune("block_size_y", {1, 2});

    SECTION("test unknown wisdom file") {
        std::string tuning_key = "does_not_exist";

        bool success =
            process_wisdom_file(wisdom_dir, tuning_key, space, [&](auto& r) {
                FAIL();  // Calling the callback should fail the test
            });

        CHECK_FALSE(success);
    }

    SECTION("test existing wisdom file") {
        std::string tuning_key = "example_kernel";

        int lineno = 0;
        bool success =
            process_wisdom_file(wisdom_dir, tuning_key, space, [&](auto& r) {
                if (lineno == 0) {
                    CHECK(r.device_name() == "K20");
                    CHECK(r.environment("device_name") == "K20");
                    CHECK(r.problem_size() == ProblemSize {100, 100});
                    CHECK(r.objective() == 0.5);
                }

                auto config = r.config();
                CHECK(config[block_size_x] == lineno + 2);
                CHECK(config[block_size_y] == 1);

                lineno++;
            });

        CHECK(lineno == 5);
        CHECK(success);
    }
}

TEST_CASE("WisdomKernel", "[CUDA]") {
    CUcontext ctx;
    KERNEL_LAUNCHER_CUDA_CHECK(cuInit(0));
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxCreate(&ctx, 0, 0));

    std::string assets_dir = assets_directory();
    KernelBuilder builder = build_vector_add_kernel();

    WisdomSettings wisdom_settings(assets_dir, assets_dir);
    WisdomKernel kernel(builder, default_compiler(), wisdom_settings);

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
        kernel(int(n), (int*)dev_c, (const int*)dev_a, (const int*)dev_b));

    // Copy C out
    cuda_copy((int*)dev_c, c.data(), n);

    REQUIRE(c == c_ref);

    //    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxDestroy(ctx));
}
