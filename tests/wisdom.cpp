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
    }

    SECTION("scalar point3") {
        struct point3 {
            unsigned long long x;
            unsigned long long y;
            unsigned long long z;
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
        CHECK((result.x == 1 && result.y == 2 && result.z == 3));
    }

    SECTION("scalar int*") {
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
    }
}