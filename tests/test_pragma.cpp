#include "catch.hpp"
#include "kernel_launcher/pragma.hpp"
#include "test_helpers.hpp"

using namespace kernel_launcher;

static const char* VECTOR_ADD_SOURCE = R"(
namespace demo {
#pragma kernel tune(block_size=32, 64, 128)
#pragma kernel tune(elements_per_thread=1, 2) default(2)
#pragma kernel problem_size(n)
#pragma kernel grid_divisor(block_size * elements_per_thread)
#pragma kernel buffers(C[n], A[n], B[n])
#pragma kernel tuning_key("vector_add_", N)
template <int N, int block_size, int elements_per_thread>
__global__ void vector_add(int n, float* C, const float* A, const float* B) {}
}

__global__ void not_annotated(int n) {}
)";

TEST_CASE("build_pragma_kernel") {
    FileSource source("vector_add.cu", VECTOR_ADD_SOURCE);
    KernelBuilder builder = build_pragma_kernel("vector_add", source, {7});

    REQUIRE(builder.parameters().size() == 2);
    CHECK(builder["block_size"].values() == std::vector<Primitive> {32, 64, 128});
    CHECK(builder["block_size"].default_value() == 32);
    CHECK(builder["elements_per_thread"].default_value() == 2);

    std::vector<KernelArg> args = {
        into_kernel_arg(1000),
        into_kernel_arg((float*)nullptr),
        into_kernel_arg((const float*)nullptr),
        into_kernel_arg((const float*)nullptr)};

    GPUDevice device;
    ProblemKey problem = builder.build_problem_key(args);
    CHECK(problem == ProblemKey("vector_add_7", 1000));
    CHECK(args[1].buffer_size() == 1000 * sizeof(float));

    Config config = builder.default_config();
    KernelDef def = builder.build_kernel_def(device, config, problem, {});
    CHECK(def.kernel_name == "demo::vector_add");
    CHECK(
        def.template_args
        == std::vector<TemplateArg> {
            TemplateArg(Primitive(7)),
            TemplateArg(Primitive(32)),
            TemplateArg(Primitive(2))});

    // The pragmas are commented out in the source that is compiled.
    REQUIRE(def.source.content().has_value());
    std::string_view content = *def.source.content();
    CHECK(def.source.file_name() == "vector_add.cu");
    CHECK(content.find("//#pragma kernel tune(block_size=32, 64, 128)") != std::string::npos);
    CHECK(content.find("\n#pragma") == std::string::npos);
}

TEST_CASE("build_pragma_kernel name lookup") {
    FileSource source("vector_add.cu", VECTOR_ADD_SOURCE);

    CHECK_NOTHROW(build_pragma_kernel("demo::vector_add", source, {7}));
    CHECK_THROWS(build_pragma_kernel("other::vector_add", source, {7}));
    CHECK_THROWS(build_pragma_kernel("not_annotated", source, {7}));
    CHECK_THROWS(build_pragma_kernel("missing", source, {7}));
}

TEST_CASE("build_pragma_kernel template arguments") {
    FileSource source("vector_add.cu", VECTOR_ADD_SOURCE);

    // `N` is not defined.
    CHECK_THROWS(build_pragma_kernel("vector_add", source));

    // Too many arguments.
    CHECK_THROWS(build_pragma_kernel("vector_add", source, {1, 2, 3, 4}));
}

TEST_CASE("build_pragma_kernel from file") {
    std::string dir = test::make_temp_dir();
    write_file(path_join(dir, "vector_add.cu"), VECTOR_ADD_SOURCE);

    FileSource source("vector_add.cu");
    KernelBuilder builder = build_pragma_kernel("vector_add", source, {7}, DefaultLoader({dir}));
    CHECK(builder.parameters().size() == 2);

    CHECK_THROWS_AS(build_pragma_kernel("vector_add", source, {7}), FileNotFoundException);
}

TEST_CASE("PragmaKernel") {
    PragmaKernel a("vector_add", "vector_add.cu", {7});
    PragmaKernel b("vector_add", "vector_add.cu", {7});
    PragmaKernel c("vector_add", "vector_add.cu", {8});
    PragmaKernel d("vector_add", "other.cu", {7});

    CHECK(a.equals(b));
    CHECK(a.hash() == b.hash());
    CHECK_FALSE(a.equals(c));
    CHECK_FALSE(a.equals(d));
    CHECK(DescriptorKey(a) == DescriptorKey(b));

    std::string dir = test::make_temp_dir();
    std::string path = path_join(dir, "vector_add.cu");
    write_file(path, VECTOR_ADD_SOURCE);

    KernelBuilder builder = PragmaKernel("vector_add", path, {7}).build();
    CHECK(builder.parameters().size() == 2);
}
