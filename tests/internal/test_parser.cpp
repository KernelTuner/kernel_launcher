#include "catch.hpp"
#include "kernel_launcher/internal/parser.hpp"

using namespace kernel_launcher;
using namespace kernel_launcher::internal;

static const char* SOURCE = R"(#include <stdio.h>
using namespace std;
namespace alias = std;

namespace a { namespace b::c {
#pragma kernel tune(x=1,2)
#pragma kernel block_size(x)
template <typename T, unsigned int N = 4, bool B = true, int M = -3, int K = sizeof(T), typename U = foo<int, 2>>
__global__ void my_kernel(const T* __restrict__ input, foo<int, 3> bar, int n) {
    #pragma unroll
    for (int i = 0; i < n; i++) {}
}
}}

__global__ void not_annotated(int x) {}

namespace {
#pragma kernel_tuner problem_size(10)
__global__ void second(void) {}
}
)";

// Replaces every occurrence of `from` in `input` by `to`.
static std::string replace_all(std::string input, const std::string& from, const std::string& to) {
    for (size_t i = input.find(from); i != std::string::npos; i = input.find(from, i + to.size())) {
        input.replace(i, from.size(), to);
    }

    return input;
}

TEST_CASE("extract_annotated_kernels") {
    TokenStream stream("test.cu", SOURCE);
    AnnotatedDocument doc = extract_annotated_kernels(stream);

    REQUIRE(doc.kernels.size() == 2);

    SECTION("namespaced template kernel") {
        const AnnotatedKernelSpec& kernel = doc.kernels[0];

        CHECK(kernel.qualified_name == "a::b::c::my_kernel");
        CHECK(stream.span(kernel.name) == "my_kernel");

        // One entry per pragma, pointing to the first token after the name.
        REQUIRE(kernel.directives.size() == 2);
        CHECK(stream.span(kernel.directives[0]) == "tune");
        CHECK(stream.span(kernel.directives[1]) == "block_size");

        const auto& params = kernel.template_params;
        REQUIRE(params.size() == 6);

        CHECK(stream.span(params[0].name) == "T");
        CHECK_FALSE(params[0].is_integral);
        CHECK(params[0].default_value.is_empty());

        CHECK(stream.span(params[1].name) == "N");
        CHECK(params[1].is_integral);
        CHECK(params[1].integral_type == "unsigned int");
        CHECK(params[1].default_value == 4);

        CHECK(stream.span(params[2].name) == "B");
        CHECK(params[2].integral_type == "bool");
        CHECK(params[2].default_value == true);
        CHECK(params[2].default_value.kind() == Primitive::Kind::Bool);

        CHECK(stream.span(params[3].name) == "M");
        CHECK(params[3].default_value == -3);

        // Only literals are supported as default values.
        CHECK(stream.span(params[4].name) == "K");
        CHECK(params[4].is_integral);
        CHECK(params[4].default_value.is_empty());

        CHECK(stream.span(params[5].name) == "U");
        CHECK_FALSE(params[5].is_integral);

        const auto& fun_params = kernel.fun_params;
        REQUIRE(fun_params.size() == 3);
        CHECK(fun_params[0].type == "const T* __restrict__");
        CHECK(stream.span(fun_params[0].name) == "input");
        CHECK(fun_params[1].type == "foo<int, 3>");
        CHECK(stream.span(fun_params[1].name) == "bar");
        CHECK(fun_params[2].type == "int");
        CHECK(stream.span(fun_params[2].name) == "n");
    }

    SECTION("anonymous namespace") {
        const AnnotatedKernelSpec& kernel = doc.kernels[1];

        CHECK(kernel.qualified_name == "second");
        REQUIRE(kernel.directives.size() == 1);
        CHECK(stream.span(kernel.directives[0]) == "problem_size");
        CHECK(kernel.template_params.empty());
        CHECK(kernel.fun_params.empty());
    }

    SECTION("processed source") {
        // Only the kernel pragmas are commented out; everything else is unchanged.
        CHECK(doc.processed_source == replace_all(SOURCE, "#pragma kernel", "//#pragma kernel"));
    }
}

TEST_CASE("extract_annotated_kernels without annotations") {
    TokenStream stream("test.cu", "__global__ void kernel() {}");
    AnnotatedDocument doc = extract_annotated_kernels(stream);

    CHECK(doc.kernels.empty());
    CHECK(doc.processed_source == "__global__ void kernel() {}");
}

TEST_CASE("extract_annotated_kernels errors") {
    auto parse = [](const std::string& source) {
        TokenStream stream("test.cu", source);
        return extract_annotated_kernels(stream);
    };

    CHECK_THROWS_WITH(parse("void f() { }}"), Catch::Contains("no matching '{' found"));
    CHECK_THROWS_WITH(parse("void f(]"), Catch::Contains("no matching '[' found"));
    CHECK_THROWS_WITH(parse("void f[)"), Catch::Contains("no matching '(' found"));

    // An annotated function must be a kernel.
    CHECK_THROWS(parse("#pragma kernel tune(x=1)\nvoid f() {}"));
    CHECK_THROWS(parse("#pragma kernel tune(x=1)\n__global__ int f() {}"));
    CHECK_THROWS(parse("#pragma kernel tune(x=1)\n__global__ void f(int) {}"));
    CHECK_THROWS(parse("#pragma kernel tune(x=1)\ntemplate <int> __global__ void f() {}"));
    CHECK_THROWS(parse("#pragma kernel tune(x=1)\n"));
}
