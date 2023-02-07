#include "kernel_launcher/builder.h"

#include "catch.hpp"
#include "test_utils.h"

using namespace kernel_launcher;

TEST_CASE("KernelBuilder") {
    std::array<int, 5> data = {1, 2, 3, 4, 5};
    KernelBuilder builder("nothing", "stdin");
    std::vector<KernelArg> args = {
        KernelArg::from_scalar(100),
        KernelArg::from_scalar(200),
        KernelArg::from_scalar(&data[0])};

    SECTION("problem_size static") {
        builder.problem_size(ProblemSize(100, 200));
        auto f = builder.problem_processor();
        CHECK(f(args) == ProblemSize {100, 200});
    }

    SECTION("problem_size expression") {
        builder.problem_size(arg0, arg1);
        auto f = builder.problem_processor();
        CHECK(f(args) == ProblemSize {100, 200});
    }

    SECTION("problem_size function") {
        builder.problem_size(ProblemProcessor([](auto& args) {
            return ProblemSize {
                args[0].to_value().template to<uint32_t>(),
                args[1].to_value().template to<uint32_t>()};
        }));

        auto f = builder.problem_processor();
        CHECK(f(args) == ProblemSize {100, 200});
    }

    SECTION("buffer_size") {
        builder.problem_size(arg0, arg1);
        builder.buffer_size(arg2, arg0);
        auto f = builder.problem_processor();
        CHECK(f(args) == ProblemSize {100, 200});
        CHECK(args[2].is_array());
    }

    SECTION("buffers") {
        builder.problem_size(arg0, arg1);
        builder.buffers(arg2[arg0]);
        auto f = builder.problem_processor();
        CHECK(f(args) == ProblemSize {100, 200});
        CHECK(args[2].is_array());
    }
}