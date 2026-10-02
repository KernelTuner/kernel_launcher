#include "catch.hpp"
#include "kernel_launcher/compiler.hpp"
#include "test_helpers.hpp"

using namespace kernel_launcher;
using test::MapLoader;

static const char* VECTOR_ADD_SOURCE = R"(
template <typename T, int factor>
__global__ void vector_add(int n, T* C, const T* A, const T* B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = factor * (A[i] + B[i]);
}
)";

static KernelDef vector_add_def() {
    KernelDef def("vector_add", FileSource("vector_add.cu", VECTOR_ADD_SOURCE));
    def.template_args = {TemplateArg(TypeInfo::of<float>()), TemplateArg(2)};
    def.param_types = {
        TypeInfo::of<int>(),
        TypeInfo::of<float*>(),
        TypeInfo::of<const float*>(),
        TypeInfo::of<const float*>()};
    return def;
}

TEST_CASE("FileSource") {
    FileSource file("assets/example_header.cuh");
    FileSource inline_source("kernel.cu", "content");

    CHECK(file.file_name() == "assets/example_header.cuh");
    CHECK_FALSE(file.content().has_value());
    CHECK(inline_source.file_name() == "kernel.cu");
    CHECK(inline_source.content() == std::string_view("content"));

    SECTION("read") {
        DefaultLoader loader;
        CHECK(file.read(loader).find("function_from_header") != std::string::npos);
        CHECK(inline_source.read(loader) == "content");
        CHECK_THROWS_AS(FileSource("missing.cu").read(loader), FileNotFoundException);
    }

    SECTION("equality") {
        CHECK(file == FileSource("assets/example_header.cuh"));
        CHECK(file != FileSource("assets/other.cuh"));
        CHECK(inline_source == FileSource("kernel.cu", "content"));
        CHECK(inline_source != FileSource("kernel.cu", "other"));
        CHECK(inline_source != FileSource("kernel.cu"));

        CHECK(file.hash() == FileSource("assets/example_header.cuh").hash());
        CHECK(std::hash<FileSource> {}(inline_source) == inline_source.hash());
    }
}

TEST_CASE("KernelDef") {
    KernelDef a = vector_add_def();
    KernelDef b = vector_add_def();

    CHECK(a == b);
    CHECK(a.calculate_hash() == b.calculate_hash());
    CHECK(std::hash<KernelDef> {}(a) == a.calculate_hash());

    SECTION("name") {
        b.kernel_name = "other";
        CHECK(a != b);
    }

    SECTION("source") {
        b.source = FileSource("vector_add.cu");
        CHECK(a != b);
    }

    SECTION("headers") {
        b.headers.push_back(FileSource("header.cuh"));
        CHECK(a != b);
    }

    SECTION("parameter types") {
        b.param_types.pop_back();
        CHECK(a != b);
    }

    SECTION("template arguments") {
        b.template_args[1] = TemplateArg(3);
        CHECK(a != b);
    }

    SECTION("compiler flags") {
        b.compiler_flags.push_back("-lineinfo");
        CHECK(a != b);
    }
}

TEST_CASE("RuntimeCompiler::version") {
    int version = RuntimeCompiler::version();
    CHECK(version >= 11000);
    CHECK(version % 10 == 0);
}

TEST_CASE("RuntimeCompiler::compile_image") {
    GPUArch arch = test::fixed_arch();
    std::string image;
    std::string symbol;

    SECTION("inline source") {
        RuntimeCompiler compiler;
        compiler.compile_image(arch, vector_add_def(), image, symbol);

        CHECK(
            image.compare(
                0,
                4,
                "\x7f"
                "ELF")
            == 0);
        CHECK(image.find(symbol) != std::string::npos);
        CHECK(symbol.find("vector_add") != std::string::npos);
    }

    SECTION("headers and flags") {
        auto loader = std::make_shared<MapLoader>();
        loader->files["kernel.cu"] = R"(
            #include "my_header.cuh"
            #include "unknown_header.cuh"  // Not found, replaced by an empty header
            #include <stdint.h>             // From the jitsafe headers
            __global__ void kernel(int* out) { *out = header_value() + VALUE + PRE_A + PRE_B; }
        )";
        loader->files["my_header.cuh"] = "__device__ int header_value() { return 1; }";

        KernelDef def("kernel", FileSource("kernel.cu"));
        def.headers = {
            FileSource("a.cuh", "#define PRE_A 3"),
            FileSource("b.cuh", "#define PRE_B 4")};
        def.param_types = {TypeInfo::of<int*>()};
        def.compiler_flags = {"-DVALUE=2"};

        RuntimeCompiler compiler(loader);
        compiler.compile_image(arch, def, image, symbol);
        CHECK(
            image.compare(
                0,
                4,
                "\x7f"
                "ELF")
            == 0);

        // Without the define, compilation fails.
        def.compiler_flags.clear();
        CHECK_THROWS(compiler.compile_image(arch, def, image, symbol));

        // The define can also be given as a default option of the compiler.
        RuntimeCompiler compiler_with_defaults(loader, {"."}, {"-DVALUE=2"});
        compiler_with_defaults.compile_image(arch, def, image, symbol);
        CHECK(
            image.compare(
                0,
                4,
                "\x7f"
                "ELF")
            == 0);
    }

    SECTION("search directories") {
        KernelDef def("matmul_kernel", FileSource("matmul_kernel.cu"));
        def.template_args = {TemplateArg(16), TemplateArg(16), TemplateArg(1), TemplateArg(1)};
        def.param_types = {
            TypeInfo::of<int>(),
            TypeInfo::of<float*>(),
            TypeInfo::of<float*>(),
            TypeInfo::of<float*>()};

        RuntimeCompiler compiler(nullptr, {"assets"});
        CHECK_NOTHROW(compiler.compile_image(arch, def, image, symbol));

        RuntimeCompiler no_dirs(nullptr, {});
        CHECK_THROWS_AS(no_dirs.compile_image(arch, def, image, symbol), FileNotFoundException);
    }

    SECTION("compilation error") {
        KernelDef def("kernel", FileSource("kernel.cu", "__global__ void kernel() { error }"));

        RuntimeCompiler compiler;
        CHECK_THROWS_AS(compiler.compile_image(arch, def, image, symbol), CompilerException);
    }
}

TEST_CASE("RuntimeCompiler::compile") {
    REQUIRE_GPU_OR_SKIP();

    RuntimeCompiler compiler;
    std::shared_ptr<GPUKernel> kernel =
        compiler.compile(GPUContextRef::current(), vector_add_def());

    REQUIRE(kernel != nullptr);
    CHECK(kernel->function() != nullptr);
    CHECK(kernel->max_threads_per_block() > 0);
}
