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