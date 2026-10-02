#include "catch.hpp"
#include "kernel_launcher/kernel_arg.hpp"
#include "kernel_launcher/kernel_builder.hpp"

using namespace kernel_launcher;

// A kernel that is never loaded; only valid for expressions that do not query
// kernel attributes.
static GPUKernel dummy_kernel() {
    return GPUKernel(nullptr, GPUFunction(nullptr), "dummy");
}

static std::vector<KernelArg> vector_add_args(int n) {
    return {
        into_kernel_arg(n),
        into_kernel_arg((float*)nullptr),
        into_kernel_arg((const float*)nullptr)};
}

TEST_CASE("KernelBuilder") {
    KernelBuilder builder("vector_add", FileSource("vector_add.cu", "source"));
    TunableParam bs = builder.tune("block_size", {32, 64, 128});
    TunableParam ept = builder.tune("elements_per_thread", {1, 2, 4}, 2);
    builder.restriction(bs * ept <= 256);

    ArgumentExpr n(0);
    ArgumentExpr c(1);
    ArgumentExpr a(2);

    builder.tuning_key("vector_add")
        .problem_size(n, 1, 1)
        .feature("n_is_even", n % 2 == 0)
        .buffers(c[n], a[n])
        .grid_divisor(bs * ept)
        .shared_memory(bs * 4)
        .define("ELEMENTS", ept)
        .define(bs)
        .template_args(bs, true)
        .compiler_flag("-lineinfo")
        .include_header(FileSource("header.cuh", "header"))
        .assertion(problem_size_x() > 0);

    GPUDevice device;
    std::vector<KernelArg> args = vector_add_args(1000);

    ProblemKey problem = builder.build_problem_key(args);
    ProblemKey expected_problem("vector_add", 1000);
    expected_problem.features.add("n_is_even", true);
    CHECK(problem == expected_problem);

    CHECK(args[0].buffer_size() == 0);
    CHECK(args[1].buffer_size() == 1000 * sizeof(float));
    CHECK(args[2].buffer_size() == 1000 * sizeof(float));

    ConfigSpace space = builder.build_config_space(device, problem);
    CHECK(space.valid_configs().size() == 8);

    SECTION("build_kernel_def") {
        Config config = space.default_config();
        std::vector<TypeInfo> param_types = {
            TypeInfo::of<int>(),
            TypeInfo::of<float*>(),
            TypeInfo::of<const float*>()};

        KernelDef def = builder.build_kernel_def(device, config, problem, param_types);

        CHECK(def.kernel_name == "vector_add");
        CHECK(def.source == FileSource("vector_add.cu", "source"));
        CHECK(def.headers == std::vector<FileSource> {FileSource("header.cuh", "header")});
        CHECK(def.param_types == param_types);
        CHECK(
            def.template_args
            == std::vector<TemplateArg> {TemplateArg(Primitive(32)), TemplateArg(true)});
        CHECK(
            def.compiler_flags
            == std::vector<std::string> {"-lineinfo", "-DELEMENTS=2", "-Dblock_size=32"});
    }

    SECTION("build_launch_config") {
        GPUKernel kernel = dummy_kernel();

        Config config = space.default_config();
        LaunchConfig launch = builder.build_launch_config(device, kernel, config, problem);
        CHECK(launch.block_size[0] == 32);
        CHECK(launch.block_size[1] == 1);
        CHECK(launch.block_size[2] == 1);
        CHECK(launch.grid_size[0] == 16);  // ceil(1000 / (32 * 2))
        CHECK(launch.grid_size[1] == 1);
        CHECK(launch.grid_size[2] == 1);
        CHECK(launch.shared_mem == 32 * 4);

        config.insert("block_size", 128);
        launch = builder.build_launch_config(device, kernel, config, problem);
        CHECK(launch.block_size[0] == 128);
        CHECK(launch.grid_size[0] == 4);  // ceil(1000 / (128 * 2))
        CHECK(launch.shared_mem == 128 * 4);
    }

    SECTION("assertion") {
        GPUKernel kernel = dummy_kernel();
        std::vector<KernelArg> empty_args = vector_add_args(0);
        ProblemKey empty_problem = builder.build_problem_key(empty_args);

        CHECK_THROWS(
            builder.build_launch_config(device, kernel, space.default_config(), empty_problem));
    }
}

TEST_CASE("KernelBuilder::build_problem_key") {
    KernelBuilder builder("kernel", FileSource("kernel.cu", ""));
    std::vector<KernelArg> args = vector_add_args(10);

    SECTION("defaults") {
        CHECK(builder.build_problem_key(args) == ProblemKey("kernel", 1, 1, 1));
    }

    SECTION("buffer index out of bounds") {
        builder.buffer_size(ArgumentExpr(3), 10);
        CHECK_THROWS(builder.build_problem_key(args));
    }

    SECTION("buffer of non-pointer argument") {
        builder.buffer_size(ArgumentExpr(0), 10);
        CHECK_THROWS(builder.build_problem_key(args));
    }

    SECTION("buffer of void pointer argument") {
        void* ptr = nullptr;
        std::vector<KernelArg> void_args = {KernelArg::of(ptr)};

        builder.buffer_size(ArgumentExpr(0), 10);
        CHECK_THROWS(builder.build_problem_key(void_args));
    }

    SECTION("argument that is not a primitive") {
        builder.problem_size(ArgumentExpr(1), 1, 1);
        CHECK_THROWS(builder.build_problem_key(args));
    }

    SECTION("buffer size from problem size") {
        builder.problem_size(ArgumentExpr(0), 3, 1);
        builder.buffer_size(ArgumentExpr(1), problem_size_x() * problem_size_y());
        builder.build_problem_key(args);
        CHECK(args[1].buffer_size() == 30 * sizeof(float));
    }

    SECTION("buffer size from feature") {
        builder.feature("padded", ArgumentExpr(0) + 6);
        builder.buffer_size(ArgumentExpr(1), FeatureExpr("padded"));
        builder.build_problem_key(args);
        CHECK(args[1].buffer_size() == 16 * sizeof(float));
    }

    SECTION("problem size from problem size") {
        builder.problem_size(ArgumentExpr(0), problem_size_x(), 1);
        CHECK_THROWS(builder.build_problem_key(args));
    }
}

TEST_CASE("KernelBuilder::resolve_block_size") {
    KernelBuilder builder("kernel", FileSource("kernel.cu", ""));
    CHECK_THROWS(builder.resolve_block_size());

    // Parameter names are matched case-insensitively.
    builder.tune("BLOCK_SIZE_X", {64});
    builder.tune("block_dim_y", {2});
    Config config = builder.default_config();

    std::array<Expr<uint32_t>, 3> block_size = builder.resolve_block_size();
    CHECK(block_size[0].eval(config) == 64);
    CHECK(block_size[1].eval(config) == 2);
    CHECK(block_size[2].eval(config) == 1);

    SECTION("ambiguous") {
        builder.tune("block_x", {32});
        CHECK_THROWS(builder.resolve_block_size());
    }

    SECTION("explicit") {
        builder.tune("block_x", {32});
        builder.block_size(128, 2);

        block_size = builder.resolve_block_size();
        CHECK(block_size[0].eval(config) == 128);
        CHECK(block_size[1].eval(config) == 2);
        CHECK(block_size[2].eval(config) == 1);
    }
}

TEST_CASE("KernelBuilder::resolve_grid_size") {
    KernelBuilder builder("kernel", FileSource("kernel.cu", ""));
    builder.problem_size(100, 20, 3).block_size(32, 4);

    GPUDevice device;
    GPUKernel kernel = dummy_kernel();
    std::vector<KernelArg> args;
    ProblemKey problem = builder.build_problem_key(args);
    Config config;

    SECTION("divided by block size") {
        LaunchConfig launch = builder.build_launch_config(device, kernel, config, problem);
        CHECK(launch.grid_size[0] == 4);
        CHECK(launch.grid_size[1] == 5);
        CHECK(launch.grid_size[2] == 3);
    }

    SECTION("grid divisor") {
        builder.grid_divisor(10, 20, 2);
        LaunchConfig launch = builder.build_launch_config(device, kernel, config, problem);
        CHECK(launch.grid_size[0] == 10);
        CHECK(launch.grid_size[1] == 1);
        CHECK(launch.grid_size[2] == 2);
    }

    SECTION("explicit grid size") {
        builder.grid_size(problem_size_x() / 10, 7);
        LaunchConfig launch = builder.build_launch_config(device, kernel, config, problem);
        CHECK(launch.grid_size[0] == 10);
        CHECK(launch.grid_size[1] == 7);
        CHECK(launch.grid_size[2] == 1);

        builder.grid_divisor(10);
        CHECK_THROWS(builder.resolve_grid_size());
    }
}

TEST_CASE("KernelBuilder features") {
    KernelBuilder builder("kernel", FileSource("kernel.cu", ""));
    builder.problem_size(10, 1, 1)
        .feature("blocks", ArgumentExpr(0) / 100)
        .block_size(32)
        .grid_size(FeatureExpr("blocks"));

    GPUDevice device;
    GPUKernel kernel = dummy_kernel();
    std::vector<KernelArg> args = {into_kernel_arg(500)};
    ProblemKey problem = builder.build_problem_key(args);
    CHECK(problem.features.get("blocks") == 5);

    LaunchConfig launch = builder.build_launch_config(device, kernel, Config(), problem);
    CHECK(launch.grid_size[0] == 5);
}

TEST_CASE("KernelBuilder kernel attributes") {
    KernelBuilder builder("kernel", FileSource("kernel.cu", ""));
    builder.template_arg(kernel_attribute(CU_FUNC_ATTRIBUTE_NUM_REGS));

    GPUDevice device;
    std::vector<KernelArg> args;
    ProblemKey problem = builder.build_problem_key(args);

    // Kernel attributes are unknown before the kernel is compiled.
    CHECK_THROWS_WITH(
        builder.build_kernel_def(device, Config(), problem, {}),
        Catch::Equals(
            "failed to evaluate expression '$kernel_attribute_NUM_REGS': "
            "unknown variable '$kernel_attribute_NUM_REGS'"));
}
