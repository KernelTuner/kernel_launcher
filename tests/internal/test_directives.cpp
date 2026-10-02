#include "catch.hpp"
#include "kernel_launcher/internal/directives.hpp"
#include "kernel_launcher/kernel_arg.hpp"

using namespace kernel_launcher;
using namespace kernel_launcher::internal;

// Builds the single annotated kernel in `source`. `pragmas` are placed on
// separate lines in front of the kernel.
static KernelBuilder build(
    const std::string& pragmas,
    const std::string& kernel = "__global__ void kernel(int n, float* C, const float* A)",
    const std::vector<Primitive>& template_args = {}) {
    std::string source = pragmas + "\n" + kernel + " {}\n";

    TokenStream stream("test.cu", source);
    AnnotatedDocument doc = extract_annotated_kernels(stream);
    REQUIRE(doc.kernels.size() == 1);

    FileSource file("test.cu", doc.processed_source);
    return builder_from_annotated_kernel(stream, file, doc.kernels[0], template_args);
}

// Evaluates the values of a single tunable parameter `x`.
static std::vector<Primitive> tune_values(const std::string& values) {
    KernelBuilder builder = build("#pragma kernel tune(x = " + values + ")");
    REQUIRE(builder.parameters().size() == 1);
    return builder["x"].values();
}

static GPUKernel dummy_kernel() {
    return GPUKernel(nullptr, GPUFunction(nullptr), "dummy");
}

TEST_CASE("directive tune") {
    SECTION("default value") {
        KernelBuilder builder = build(
            "#pragma kernel tune(x=1, 2, 3)\n"
            "#pragma kernel tune(y=1, 2, 3) default(2)");

        REQUIRE(builder.parameters().size() == 2);
        CHECK(builder["x"].values() == std::vector<Primitive> {1, 2, 3});
        CHECK(builder["x"].default_value() == 1);
        CHECK(builder["y"].default_value() == 2);
    }

    SECTION("multiple directives on one line") {
        KernelBuilder builder = build("#pragma kernel tune(x=1) tune(y=2)");
        CHECK(builder.parameters().size() == 2);
    }

    SECTION("pragma aliases") {
        CHECK(build("#pragma kernel_tuner tune(x=1)").parameters().size() == 1);
        CHECK(build("#pragma kernel_launcher tune(x=1)").parameters().size() == 1);
    }

    SECTION("literals") {
        CHECK(
            tune_values("1, -4, +5, true, false")
            == std::vector<Primitive> {1, -4, 5, true, false});
        CHECK(
            tune_values("\"abc\", 'x', \"a\\tb\\\"\"")
            == std::vector<Primitive> {"abc", "x", "a\tb\""});
        CHECK(tune_values("null") == std::vector<Primitive> {Primitive()});
    }

    SECTION("operators") {
        CHECK(
            tune_values("2 * 3, 10 / 3, 10 % 3, 1 + 2, 1 - 2")
            == std::vector<Primitive> {6, 3, 1, 3, -1});
        CHECK(
            tune_values("1 < 2, 1 > 2, 2 <= 2, 1 >= 2, 1 == 1, 1 != 1")
            == std::vector<Primitive> {true, false, true, false, true, false});
        CHECK(
            tune_values("true && false, true || false, !true")
            == std::vector<Primitive> {false, true, false});
    }

    SECTION("precedence") {
        CHECK(
            tune_values("1 + 2 * 3, (1 + 2) * 3, 8 - 2 - 1, 16 / 4 / 2")
            == std::vector<Primitive> {7, 9, 5, 2});
        CHECK(tune_values("1 + 1 == 2 && 3 > 2 || false") == std::vector<Primitive> {true});
        CHECK(tune_values("-2 * 3, !0 + 1") == std::vector<Primitive> {-6, 2});
        CHECK(tune_values("1 < 2 ? 10 : 20, 0 ? 10 : 20") == std::vector<Primitive> {10, 20});
    }

    SECTION("functions") {
        CHECK(
            tune_values(
                "round(float(5) / 2), floor(float(5) / 2), ceil(float(5) / 2), div_ceil(10, 3)")
            == std::vector<Primitive> {3, 2, 3, 4});
        CHECK(
            tune_values("min(2, 5), min(5, 2), max(2, 5), max(5, 2), max(1 + 1, 3 * 1)")
            == std::vector<Primitive> {2, 2, 5, 5, 3});

        CHECK(
            tune_values("min(7), min(4, 1, 3), max(4, 9, 3), min(5, 4, 3, 2, 6), max(1, 2, 8, 3)")
            == std::vector<Primitive> {7, 1, 9, 2, 8});

        CHECK_THROWS_WITH(
            tune_values("min()"),
            Catch::Contains("function expects at least 1 argument"));

        CHECK_THROWS_WITH(
            tune_values("round(1, 2)"),
            Catch::Contains("function expects 1 arguments"));
        CHECK_THROWS_WITH(tune_values("unknown(1)"), Catch::Contains("unknown function name"));
    }

    SECTION("errors") {
        CHECK_THROWS_WITH(tune_values("y"), Catch::Contains("unknown variable name"));
        CHECK_THROWS_WITH(tune_values("n"), Catch::Contains("error while evaluating expression"));
        CHECK_THROWS_WITH(
            tune_values("1 / 0"),
            Catch::Contains("error while evaluating expression"));
        CHECK_THROWS_WITH(tune_values("*"), Catch::Contains("expecting expression"));
        CHECK_THROWS(tune_values("99999999999999999999999"));

        CHECK_THROWS_WITH(
            build("#pragma kernel tune(x=1)\n#pragma kernel tune(x=2)"),
            Catch::Contains("variable redefined"));
        CHECK_THROWS_WITH(build("#pragma kernel unknown(x)"), Catch::Contains("not supported"));
    }
}

TEST_CASE("directive set") {
    KernelBuilder builder = build(
        "#pragma kernel tune(x=1, 2, 4)\n"
        "#pragma kernel set(a = 2 * x, b = a + 1)\n"
        "#pragma kernel let(c = 3)\n"
        "#pragma kernel tune(y = c, c + 1)\n"
        "#pragma kernel restriction(b < 9)");

    CHECK(builder["y"].values() == std::vector<Primitive> {3, 4});
    CHECK(builder.valid_configs().size() == 4);  // x=4 is not allowed

    CHECK_THROWS_WITH(
        build("#pragma kernel tune(x=1)\n#pragma kernel set(x = 1)"),
        Catch::Contains("variable redefined"));
    CHECK_THROWS_WITH(
        build("#pragma kernel set(a = 1, a = 2)"),
        Catch::Contains("variable redefined"));
    CHECK_THROWS_WITH(
        build("#pragma kernel set(a = 1)\n#pragma kernel tune(a = 2)"),
        Catch::Contains("variable redefined"));
}

TEST_CASE("directive restriction") {
    for (const char* name : {"restriction", "restrictions", "restrict", "if"}) {
        KernelBuilder builder = build(
            "#pragma kernel tune(x=1, 2, 3, 4)\n"
            "#pragma kernel "
            + std::string(name) + "(x > 1, x != 3)");

        CHECK(builder.valid_configs().size() == 2);
    }
}

TEST_CASE("directive tuning_key") {
    KernelBuilder builder = build(
        "#pragma kernel tuning_key(\"kernel_\", 1 + 1, \"_\", true)\n"
        "#pragma kernel problem_size(n)");

    std::vector<KernelArg> args = {
        into_kernel_arg(10),
        into_kernel_arg((float*)nullptr),
        into_kernel_arg((const float*)nullptr)};
    ProblemKey problem = builder.build_problem_key(args);
    CHECK(problem == ProblemKey("kernel_2_true", 10));
}

TEST_CASE("directive launch configuration") {
    KernelBuilder builder = build(
        "#pragma kernel tune(block_size=32, 64)\n"
        "#pragma kernel tune(items_per_thread=1, 2) default(2)\n"
        "#pragma kernel problem_size(n, 2)\n"
        "#pragma kernel block_size(block_size)\n"
        "#pragma kernel grid_divisor(block_size * items_per_thread)\n"
        "#pragma kernel buffers(C[n], A[n * 2])\n"
        "#pragma kernel assert(block_size * items_per_thread < 128)");

    GPUDevice device;
    GPUKernel kernel = dummy_kernel();

    std::vector<KernelArg> args = {
        into_kernel_arg(1000),
        into_kernel_arg((float*)nullptr),
        into_kernel_arg((const float*)nullptr)};
    ProblemKey problem = builder.build_problem_key(args);
    CHECK(problem == ProblemKey("kernel", 1000, 2));
    CHECK(args[1].buffer_size() == 1000 * sizeof(float));
    CHECK(args[2].buffer_size() == 2000 * sizeof(float));

    LaunchConfig launch =
        builder.build_launch_config(device, kernel, builder.default_config(), problem);
    CHECK(launch.block_size[0] == 32);
    CHECK(launch.block_size[1] == 1);
    CHECK(launch.block_size[2] == 1);
    CHECK(launch.grid_size[0] == 16);
    CHECK(launch.grid_size[1] == 2);
    CHECK(launch.grid_size[2] == 1);

    SECTION("assertion") {
        Config config = builder.default_config();
        config.insert("block_size", 64);
        CHECK_THROWS_WITH(
            builder.build_launch_config(device, kernel, config, problem),
            Catch::Contains("assertion failed"));
    }

    SECTION("aliases") {
        KernelBuilder alias = build(
            "#pragma kernel problem_dim(100)\n"
            "#pragma kernel block_dim(32, 2)\n"
            "#pragma kernel grid_dim(7, 8, 9)");

        std::vector<KernelArg> no_args;
        ProblemKey alias_problem = alias.build_problem_key(no_args);
        CHECK(alias_problem == ProblemKey("kernel", 100));

        LaunchConfig alias_launch =
            alias.build_launch_config(device, kernel, Config(), alias_problem);
        CHECK(alias_launch.block_size[1] == 2);
        CHECK(alias_launch.grid_size[0] == 7);
        CHECK(alias_launch.grid_size[1] == 8);
        CHECK(alias_launch.grid_size[2] == 9);
    }

    SECTION("buffer of unknown argument") {
        CHECK_THROWS_WITH(
            build("#pragma kernel buffers(D[n])"),
            Catch::Contains("not the name of a kernel argument"));
    }

    SECTION("device attributes") {
        CHECK_NOTHROW(build(
            "#pragma kernel tune(x=1)\n#pragma kernel restriction(x <= DEVICE_MAX_THREADS_PER_BLOCK)"));
        CHECK_THROWS(build("#pragma kernel restriction(DEVICE_UNKNOWN > 0)"));
    }
}

TEST_CASE("directive template arguments") {
    const std::string kernel =
        "template <int N, bool flag = true, int M = 3> __global__ void kernel(int n)";

    SECTION("from tunable parameter") {
        KernelBuilder builder = build("#pragma kernel tune(N=16, 32) default(32)", kernel);

        std::vector<KernelArg> args = {into_kernel_arg(10)};
        ProblemKey problem = builder.build_problem_key(args);
        KernelDef def =
            builder.build_kernel_def(GPUDevice(), builder.default_config(), problem, {});

        CHECK(def.kernel_name == "kernel");
        CHECK(
            def.template_args
            == std::vector<TemplateArg> {
                TemplateArg(Primitive(32)),
                TemplateArg(true),
                TemplateArg(Primitive(3))});
    }

    SECTION("from user") {
        KernelBuilder builder = build("#pragma kernel set(M = N * 2)", kernel, {5, false});

        std::vector<KernelArg> args = {into_kernel_arg(10)};
        ProblemKey problem = builder.build_problem_key(args);
        KernelDef def = builder.build_kernel_def(GPUDevice(), Config(), problem, {});

        CHECK(
            def.template_args
            == std::vector<TemplateArg> {
                TemplateArg(Primitive(5)),
                TemplateArg(false),
                TemplateArg(Primitive(10))});
    }

    SECTION("errors") {
        CHECK_THROWS_WITH(
            build("#pragma kernel tune(x=1)", kernel),
            Catch::Contains("this template parameter is not defined"));
        CHECK_THROWS_WITH(
            build("#pragma kernel tune(x=1)", kernel, {1, true, 3, 4}),
            Catch::Contains("since it takes at most 3 arguments"));
        CHECK_THROWS_WITH(
            build(
                "#pragma kernel tune(x=1)",
                "template <typename T> __global__ void kernel(T* x)",
                {}),
            Catch::Contains("type template parameters are not supported"));
    }
}
