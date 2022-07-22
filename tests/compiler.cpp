#include "kernel_launcher/compiler.h"

#include "catch.hpp"

using namespace kernel_launcher;

static std::string base_dir() {
    std::string full_path = __FILE__;
    return full_path.substr(0, full_path.rfind('/'));
}

TEST_CASE("test NvrtcCompiler") {
    std::vector<std::string> options = {"-I" + base_dir()};
    NvrtcCompiler c = NvrtcCompiler(options);
    std::string ptx;
    std::string symbol;

    SECTION("basic compile") {
        std::string src = R"(
            extern "C" __global__ void example_kernel() {}
        )";

        KernelSource kernel_source {"example.cu", src};
        KernelDef def {"example_kernel", kernel_source};
        CHECK_NOTHROW(c.compile_ptx(def, 80, ptx, symbol));

        CHECK(ptx.find(".visible .entry example_kernel(") != std::string::npos);
        CHECK(symbol == "example_kernel");
    }

    SECTION("namespace compile") {
        std::string src = R"(
            namespace example_ns {
                __global__ void example_kernel() {}
            }
        )";

        KernelSource kernel_source {"example.cu", src};
        KernelDef def {"example_ns::example_kernel", kernel_source};
        CHECK_NOTHROW(c.compile_ptx(def, 80, ptx, symbol));
    }

    SECTION("arguments compile") {
        std::string src = R"(
            extern "C" __global__ void example_kernel(int x, float* y) {}
        )";

        KernelSource kernel_source {"example.cu", src};
        KernelDef def {"example_kernel", kernel_source};

        // This should fail since there are no arguments
        CHECK_THROWS(c.compile_ptx(def, 80, ptx, symbol));

        // Add arguments and retry
        def.add_parameter(TypeInfo::of<int>());
        def.add_parameter(TypeInfo::of<float*>());
        CHECK_NOTHROW(c.compile_ptx(def, 80, ptx, symbol));
    }

    SECTION("template arguments compile") {
        std::string src = R"(
            template <typename T>
            __global__ void example_kernel(T x, T* y) {}
        )";

        KernelSource kernel_source {"example.cu", src};
        KernelDef def {"example_kernel", kernel_source};

        // This should fail since there are no arguments
        CHECK_THROWS(c.compile_ptx(def, 80, ptx, symbol));

        // Add arguments and retry
        def.add_template_arg(TypeInfo::of<double>());
        def.add_parameter(TypeInfo::of<double>());
        def.add_parameter(TypeInfo::of<double*>());
        CHECK_NOTHROW(c.compile_ptx(def, 80, ptx, symbol));
    }

    SECTION("std headers compile") {
        std::string src = R"(
            #include <assert.h>
            #include <float.h>
            #include <limits.h>
            #include <math.h>
            #include <stddef.h>
            #include <stdint.h>
            #include <stdio.h>
            #include <stdlib.h>

            #include <algorithm>
            #include <cassert>
            #include <cfloat>
            #include <climits>
            #include <cmath>
            #include <complex>
            #include <cstddef>
            #include <cstdint>
            #include <cstdio>
            #include <cstdlib>
            #include <iterator>
            #include <limits>
            #include <type_traits>
            #include <utility>

            extern "C" __global__ void example_kernel() {}
        )";

        KernelSource kernel_source {"example.cu", src};
        KernelDef def {"example_kernel", kernel_source};
        CHECK_NOTHROW(c.compile_ptx(def, 80, ptx, symbol));
    }

    SECTION("custom headers compile") {
        std::string src = R"(
            #include "assets/example_header.cuh"

            extern "C" __global__ void example_kernel() {
                function_from_header();
            }
        )";

        KernelSource kernel_source {"example.cu", src};
        KernelDef def {"example_kernel", kernel_source};
        CHECK_NOTHROW(c.compile_ptx(def, 80, ptx, symbol));
    }

    SECTION("compiler options") {
        NvrtcCompiler c2 = NvrtcCompiler({"-DDEFINE_FROM_COMPILER=42"});

        std::string src = R"(
            extern "C" __global__ void example_kernel() {
                static_assert(DEFINE_FROM_COMPILER == 42, "failed");
                static_assert(DEFINE_FROM_KERNEL_DEF == 42, "failed");
            }
        )";

        KernelSource kernel_source {"example.cu", src};
        KernelDef def {"example_kernel", kernel_source};
        CHECK_THROWS(c.compile_ptx(def, 80, ptx, symbol));
        CHECK_THROWS(c2.compile_ptx(def, 80, ptx, symbol));

        def.add_compiler_option("-DDEFINE_FROM_KERNEL_DEF=42");
        CHECK_THROWS(c.compile_ptx(def, 80, ptx, symbol));
        CHECK_NOTHROW(c2.compile_ptx(def, 80, ptx, symbol));
    }
}