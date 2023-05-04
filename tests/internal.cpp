#include <cuda.h>

#include "catch.hpp"
#include "kernel_launcher/internal/directives.h"
#include "kernel_launcher/internal/parser.h"
#include "kernel_launcher/internal/tokens.h"
#include "test_utils.h"

using namespace kernel_launcher;

TEST_CASE("tokenizer real") {
    using internal::TokenKind;

    std::string input = R"(
        #include <stdio>

        #pragma kernel_tuner test
        void foo(int x) {
            /* Test
            multi-line * /
            comment */

            "a" == 1.2 // just a silly comment
        }
    )";

    std::vector<std::pair<std::string, TokenKind>> expected = {
        {"#", TokenKind::DirectiveBegin},   {"include", TokenKind::Ident},
        {"<", TokenKind::AngleL},           {"stdio", TokenKind::Ident},
        {">", TokenKind::AngleR},           {"\n", TokenKind::DirectiveEnd},
        {"#", TokenKind::DirectiveBegin},   {"pragma", TokenKind::Ident},
        {"kernel_tuner", TokenKind::Ident}, {"test", TokenKind::Ident},
        {"\n", TokenKind::DirectiveEnd},    {"void", TokenKind::Ident},
        {"foo", TokenKind::Ident},          {"(", TokenKind::ParenL},
        {"int", TokenKind::Ident},          {"x", TokenKind::Ident},
        {")", TokenKind::ParenR},           {"{", TokenKind::BraceL},
        {"\"a\"", TokenKind::String},       {"==", TokenKind::Punct},
        {"1", TokenKind::Number},           {".", TokenKind::Unknown},
        {"2", TokenKind::Number},           {"}", TokenKind::BraceR},
        {"", TokenKind::EndOfFile}};

    auto stream = internal::TokenStream("<stdin>", input);

    for (size_t i = 0; i < expected.size(); i++) {
        auto token = stream.next();

        INFO("index=" << i);
        CHECK(stream.span(token) == expected[i].first);
        CHECK(token.kind == expected[i].second);
    }
}

TEST_CASE("tokenizer exhaustive") {
    using internal::TokenKind;

    std::string input = R"(
        == != <= >= && || << >> ::
        { } [ ] ( ) , < > + = - * / ! ~ & | ^ %
        123 4.5
        "test" "a\"b" 'c'
        foo $foo foo_bar _bar _ bar1
    )";

    std::vector<std::pair<std::string, TokenKind>> expected = {
        {"==", TokenKind::Punct},         {"!=", TokenKind::Punct},
        {"<=", TokenKind::Punct},         {">=", TokenKind::Punct},
        {"&&", TokenKind::Punct},         {"||", TokenKind::Punct},
        {"<<", TokenKind::Punct},         {">>", TokenKind::Punct},
        {"::", TokenKind::Punct},         {"{", TokenKind::BraceL},
        {"}", TokenKind::BraceR},         {"[", TokenKind::BracketL},
        {"]", TokenKind::BracketR},       {"(", TokenKind::ParenL},
        {")", TokenKind::ParenR},         {",", TokenKind::Comma},
        {"<", TokenKind::AngleL},         {">", TokenKind::AngleR},
        {"+", TokenKind::Punct},          {"=", TokenKind::Punct},
        {"-", TokenKind::Punct},          {"*", TokenKind::Punct},
        {"/", TokenKind::Punct},          {"!", TokenKind::Punct},
        {"~", TokenKind::Punct},          {"&", TokenKind::Punct},
        {"|", TokenKind::Punct},          {"^", TokenKind::Punct},
        {"%", TokenKind::Punct},          {"123", TokenKind::Number},
        {"4", TokenKind::Number},         {".", TokenKind::Unknown},
        {"5", TokenKind::Number},         {R"("test")", TokenKind::String},
        {R"("a\"b")", TokenKind::String}, {"'c'", TokenKind::String},
        {"foo", TokenKind::Ident},        {"$foo", TokenKind::Ident},
        {"foo_bar", TokenKind::Ident},    {"_bar", TokenKind::Ident},
        {"_", TokenKind::Ident},          {"bar1", TokenKind::Ident},
        {"", TokenKind::EndOfFile}};

    auto stream = internal::TokenStream("<stdin>", input);

    for (size_t i = 0; i < expected.size(); i++) {
        auto token = stream.next();

        INFO("index=" << i);
        CHECK(stream.span(token) == expected[i].first);
        CHECK(token.kind == expected[i].second);
    }
}

TEST_CASE("parser") {
    std::string input = R"(
// This is a comment
namespace foo {
namespace bar {

#ifdef SOMECONSTANT
#endif

#pragma kernel_tuner tune(block_size=32, 64, 128, 256) default(128)
#pragma kernel_tuner problem_size(n)
__global__ void baz(int n, const float* a) {
    if (threadIdx.x < 10) {
        return a[threadIdx.x];
    }
}
} // namespace bar

#pragma kernel_tuner tune(block_size=32, 64, 128, 256)
#pragma kernel_tuner tune(tiling_factor=1,2,3,4)
#pragma kernel_tuner problem_size(n) \
                     grid_divisor(tiling_factor * block_size)
template <int tiling_factor>
__global__ void spaz(int n, const float* input, float* output) {
    if (threadIdx.x < 10) {
        return a[threadIdx.x];
    }
}
} // namespace foo
    )";

    std::string expected = R"(
// This is a comment
namespace foo {
namespace bar {

#ifdef SOMECONSTANT
#endif

/*#pragma kernel_tuner tune(block_size=32, 64, 128, 256) default(128)
#pragma kernel_tuner problem_size(n)
*/__global__ void baz(int n, const float* a) {
    if (threadIdx.x < 10) {
        return a[threadIdx.x];
    }
}
} // namespace bar

/*#pragma kernel_tuner tune(block_size=32, 64, 128, 256)
#pragma kernel_tuner tune(tiling_factor=1,2,3,4)
#pragma kernel_tuner problem_size(n) \
                     grid_divisor(tiling_factor * block_size)
*/template <int tiling_factor>
__global__ void spaz(int n, const float* input, float* output) {
    if (threadIdx.x < 10) {
        return a[threadIdx.x];
    }
}
} // namespace foo
    )";

    auto stream = internal::TokenStream("<stdin>", input);
    auto result = extract_annotated_kernels(stream);
    CHECK(result.processed_source == expected);

    const auto& kernels = result.kernels;
    REQUIRE(kernels.size() == 2);

    const auto& baz = kernels[0];
    CHECK(baz.qualified_name == "foo::bar::baz");
    REQUIRE(baz.fun_params.size() == 2);
    CHECK(stream.span(baz.fun_params[0].name) == "n");
    CHECK(stream.span(baz.fun_params[1].name) == "a");
    REQUIRE(baz.template_params.size() == 0);

    const auto& spaz = kernels[1];
    CHECK(spaz.qualified_name == "foo::spaz");
    REQUIRE(spaz.fun_params.size() == 3);
    CHECK(stream.span(spaz.fun_params[0].name) == "n");
    CHECK(stream.span(spaz.fun_params[1].name) == "input");
    CHECK(stream.span(spaz.fun_params[2].name) == "output");
    REQUIRE(spaz.template_params.size() == 1);
    CHECK(stream.span(spaz.template_params[0].name) == "tiling_factor");
}

TEST_CASE("directives") {
    std::string input = R"(
    namespace bar {
    namespace foo {

    #pragma kernel_tuner tune(block_size=32, 64, 128, 256)
    #pragma kernel_tuner tune(tile_factor=1, 2, 3, 4)
    #pragma kernel_tuner problem_size(n)
    #pragma kernel_tuner grid_divisor(block_size * tile_factor)
    template <typename T, int tile_factor>
    __global__ void baz(int n, const T* a) {
        if (threadIdx.x < 10) {
            return a[threadIdx.x];
        }
    }
    } // namespace foo
    } // namespace bar
    )";

    auto stream = internal::TokenStream("<stdin>", input);
    auto result = extract_annotated_kernels(stream);
    const auto& kernels = result.kernels;

    REQUIRE(kernels.size() == 1);
    const auto& kernel = kernels[0];

    KernelSource source("<stdin>", result.processed_source);
    KernelBuilder builder =
        builder_from_annotated_kernel(stream, source, kernels[0], {"float"});
}