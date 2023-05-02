#include <sstream>

#include "catch.hpp"
#include "kernel_launcher/kernel.h"
#include "test_utils.h"

using namespace kernel_launcher;

struct MyObject {};

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
        CHECK(v.to_value() == Value(5));
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

        KernelArg v = KernelArg::from_array(input.data(), input.size());
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
            KernelArg::from_array<const int>(input.data(), input.size());
        CHECK(v.type() == type_of<const int*>());
        CHECK(v.is_array() == true);
        CHECK(v.is_scalar() == false);
        CHECK(*(const int**)(v.as_void_ptr()) == input.data());

        CHECK_THROWS(v.to<int*>() == input.data());
        CHECK(v.to<const int*>() == input.data());
        CHECK_THROWS(v.to<double>());
    }

    SECTION("to_value") {
        CHECK(KernelArg::from_scalar<signed char>(13).to_value() == 13);
        CHECK(KernelArg::from_scalar<unsigned char>(14).to_value() == 14);
        CHECK(KernelArg::from_scalar<char>(15).to_value() == 15);
        CHECK(KernelArg::from_scalar<int>(16).to_value() == 16);
        CHECK(KernelArg::from_scalar<double>(17).to_value() == 17.0);

        // `const` should not matter
        CHECK(KernelArg::from_scalar<const int>(18).to_value() == 18);

        // These should fail
        CHECK_THROWS(KernelArg::from_scalar<point3>({1, 2, 3}).to_value());
        CHECK_THROWS(KernelArg::from_array<int>(nullptr, 2).to_value());
    }

    SECTION("to_array") {
        std::vector<char> input = {1, 2, 3};

        // non-pointer types can never be converted to arrays
        CHECK_THROWS(KernelArg::from_scalar(123).to_array(0));
        CHECK_THROWS(KernelArg::from_scalar(123).to_array(1));

        // pointer types can be converted to arrays
        CHECK_NOTHROW(KernelArg::from_scalar(input.data()).to_array(0));
        CHECK_NOTHROW(KernelArg::from_scalar(input.data()).to_array(3));

        // arrays can only be converted to arrays which are smaller
        CHECK_NOTHROW(
            KernelArg::from_array(input.data(), input.size()).to_array(2));
        CHECK_THROWS(
            KernelArg::from_array(input.data(), input.size()).to_array(5));
    }

    SECTION("operator<<") {
        std::stringstream stream;

        SECTION("scalar primitive") {
            stream << KernelArg::from_scalar(int(5));
            CHECK(stream.str() == "scalar 5 (type: int)");
        }

        SECTION("scalar arbitrary") {
            stream << KernelArg::from_scalar(MyObject {});
            CHECK(stream.str() == "scalar <...> (type: MyObject)");
        }

        SECTION("scalar pointer") {
            int* ptr = reinterpret_cast<int*>(0x123);
            stream << KernelArg::from_scalar(ptr);
            CHECK(stream.str() == "array 0x123 (type: int*)");
        }

        SECTION("array") {
            int* ptr = reinterpret_cast<int*>(0x123);
            stream << KernelArg::from_array(ptr, 5);
            CHECK(stream.str() == "array 0x123 of length 5 (type: int*)");
        }
    }
}

// These tests are seperate since they require CUDA
TEST_CASE("test KernelArg::copy_array", "[CUDA]") {
    CUcontext ctx;
    KERNEL_LAUNCHER_CUDA_CHECK(cuInit(0));
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxCreate(&ctx, 0, 0));

    SECTION("scalar int") {
        KernelArg v = KernelArg::from_scalar((int)123);
        CHECK(v.to_bytes() == std::vector<uint8_t> {123, 0, 0, 0});
        CHECK_THROWS(v.copy_array());
    }

    SECTION("array int*") {
        std::vector<int> array {1, 2, 3};

        std::vector<uint8_t> ptr_bytes(sizeof(int*));
        int* array_ptr = array.data();
        ::memcpy(ptr_bytes.data(), (uint8_t*)&array_ptr, sizeof(int*));

        KernelArg v = KernelArg::from_array(array.data(), array.size());
        CHECK_THROWS(v.to_bytes());  // pointers cannot be exported
        CHECK(
            v.copy_array()
            == std::vector<uint8_t> {1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0});
    }

    SECTION("array NULL") {
        KernelArg v = into_kernel_arg((int*)nullptr);
        CHECK(v.to_bytes() == std::vector<uint8_t> {0, 0, 0, 0, 0, 0, 0, 0});
        CHECK_THROWS(v.copy_array());
    }

    SECTION("array nullptr") {
        KernelArg v = into_kernel_arg(nullptr);
        CHECK(v.to_bytes() == std::vector<uint8_t> {0, 0, 0, 0, 0, 0, 0, 0});
        CHECK_THROWS(v.copy_array());
    }
}
