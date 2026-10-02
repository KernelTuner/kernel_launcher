#include <string>
#include <unordered_map>

#include "catch.hpp"
#include "kernel_launcher/compiler.hpp"
#include "test_helpers.hpp"

using namespace kernel_launcher;

namespace kernel_launcher {
// Defined in jitsafe_headers.cpp; not part of the public headers.
const std::unordered_map<std::string, std::string>& jitsafe_headers();
}  // namespace kernel_launcher

TEST_CASE("jitsafe_headers") {
    const auto& headers = jitsafe_headers();
    CHECK(&headers == &jitsafe_headers());

    for (const char* name :
         {"float.h",
          "limits.h",
          "stdint.h",
          "stddef.h",
          "stdio.h",
          "iterator",
          "limits",
          "type_traits",
          "utility",
          "math.h",
          "complex",
          "algorithm",
          "stdlib.h",
          "assert.h"}) {
        INFO("header: " << name);
        REQUIRE(headers.count(name) == 1);
        CHECK_FALSE(headers.at(name).empty());
    }

    // The C++ variants of the C headers have the same content.
    std::pair<const char*, const char*> aliases[] = {
        {"cfloat", "float.h"},
        {"cassert", "assert.h"},
        {"cstdlib", "stdlib.h"},
        {"cmath", "math.h"},
        {"cstdio", "stdio.h"},
        {"cstddef", "stddef.h"},
        {"cstdint", "stdint.h"},
        {"climits", "limits.h"}};

    for (const auto& [alias, name] : aliases) {
        INFO("header: " << alias);
        REQUIRE(headers.count(alias) == 1);
        CHECK(headers.at(alias) == headers.at(name));
    }
}

TEST_CASE("jitsafe_headers compile with NVRTC") {
    std::string source = R"(
        #include <stdint.h>
        #include <climits>
        #include <cstddef>
        #include <limits>
        #include <type_traits>

        __global__ void kernel(int64_t* out) {
            *out = int64_t(INT_MAX) + std::numeric_limits<int>::max() + int64_t(sizeof(size_t));
            static_assert(std::is_same<int, int>::value, "");
        }
    )";

    KernelDef def("kernel", FileSource("kernel.cu", source));
    def.param_types = {TypeInfo::of<int64_t*>()};

    std::string image;
    std::string symbol;
    RuntimeCompiler compiler;
    CHECK_NOTHROW(compiler.compile_image(test::fixed_arch(), def, image, symbol));
}
