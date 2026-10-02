#include <cstring>

#include "catch.hpp"
#include "kernel_launcher/kernel.hpp"
#include "test_helpers.hpp"

using namespace kernel_launcher;

static const char* VECTOR_ADD_SOURCE = R"(
__global__ void vector_add(int n, float* C, const float* A, const float* B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}
)";

static KernelBuilder vector_add_builder() {
    KernelBuilder builder("vector_add", FileSource("vector_add.cu", VECTOR_ADD_SOURCE));
    builder.tune("block_size", {32, 64, 128});

    ArgumentExpr n(0);
    builder.tuning_key("vector_add")
        .problem_size(n, 1, 1)
        .buffers(ArgumentExpr(1)[n], ArgumentExpr(2)[n], ArgumentExpr(3)[n]);

    return builder;
}

struct CountingCompiler: CompilerBase {
    std::unique_ptr<GPUKernel> compile(GPUContextRef ctx, KernelDef def) override {
        count++;
        return inner.compile(ctx, std::move(def));
    }

    int count = 0;
    RuntimeCompiler inner;
};

// Always selects `block_size=64` and reports `action` as the tuning action.
struct MockWisdom: WisdomProvider {
    Config load_config(
        const GPUDevice& device,
        const ConfigSpace& space,
        const ProblemKey& problem,
        TuningAction* action_out) override {
        load_count++;
        *action_out = action;

        Config config = space.default_config();
        config.insert("block_size", 64);
        return config;
    }

    Config autotune(
        const GPUDevice& device,
        const KernelBuilder& builder,
        const ProblemKey& problem,
        const BenchmarkFn& benchmark) override {
        throw std::logic_error("autotune is not used by `Kernel`");
    }

    void capture_kernel(
        const GPUDevice& device,
        const KernelBuilder& builder,
        const ProblemKey& problem,
        const std::vector<KernelArg>& arguments,
        const std::vector<std::vector<std::byte>>& input_arrays,
        const std::vector<std::vector<std::byte>>& output_arrays) override {
        capture_count++;
        captured_problem = problem;
        captured_args = arguments;
        inputs = input_arrays;
        outputs = output_arrays;
    }

    TuningAction action = TuningAction::None;
    int load_count = 0;
    int capture_count = 0;
    ProblemKey captured_problem = ProblemKey("");
    std::vector<KernelArg> captured_args;
    std::vector<std::vector<std::byte>> inputs;
    std::vector<std::vector<std::byte>> outputs;
};

// Device buffers for the arguments of `vector_add`. Requires a GPU.
struct VectorAddData {
    explicit VectorAddData(int n) : n(n), a(size_t(n)), b(size_t(n)) {
        for (int i = 0; i < n; i++) {
            a[size_t(i)] = float(i);
            b[size_t(i)] = float(2 * i);
        }

        size_t nbytes = size_t(n) * sizeof(float);
        REQUIRE(cuMemAlloc(&dev_c, nbytes) == CUDA_SUCCESS);
        REQUIRE(cuMemAlloc(&dev_a, nbytes) == CUDA_SUCCESS);
        REQUIRE(cuMemAlloc(&dev_b, nbytes) == CUDA_SUCCESS);
        REQUIRE(cuMemsetD8(dev_c, 0, nbytes) == CUDA_SUCCESS);
        gpu_copy(reinterpret_cast<void*>(dev_a), a.data(), nbytes);
        gpu_copy(reinterpret_cast<void*>(dev_b), b.data(), nbytes);
        gpu_synchronize();
    }

    ~VectorAddData() {
        cuMemFree(dev_c);
        cuMemFree(dev_a);
        cuMemFree(dev_b);
    }

    float* c() const {
        return reinterpret_cast<float*>(dev_c);
    }

    const float* a_ptr() const {
        return reinterpret_cast<const float*>(dev_a);
    }

    const float* b_ptr() const {
        return reinterpret_cast<const float*>(dev_b);
    }

    std::vector<float> result() const {
        std::vector<float> output(static_cast<size_t>(n));
        gpu_synchronize();
        gpu_copy(output.data(), c(), size_t(n) * sizeof(float));
        gpu_synchronize();
        return output;
    }

    void check_result() const {
        std::vector<float> output = result();

        for (int i = 0; i < n; i++) {
            REQUIRE(output[size_t(i)] == float(3 * i));
        }
    }

    int n;
    std::vector<float> a;
    std::vector<float> b;
    CUdeviceptr dev_c = 0;
    CUdeviceptr dev_a = 0;
    CUdeviceptr dev_b = 0;
};

TEST_CASE("Kernel without builder") {
    // `compile_args` needs the current context before it can report the missing builder.
    REQUIRE_GPU_OR_SKIP();

    Kernel kernel;
    std::vector<KernelArg> args;
    CHECK_THROWS_WITH(kernel.compile_args(args), Catch::Contains("not initialized"));
}

TEST_CASE("RawKernelInstance::capture_next_launch without wisdom") {
    RawKernelInstance instance(
        nullptr,
        ProblemKey("vector_add"),
        Config(),
        LaunchConfig {{1, 1, 1}, {1, 1, 1}});

    CHECK_THROWS_AS(
        instance.capture_next_launch(nullptr, vector_add_builder()),
        std::invalid_argument);
}

TEST_CASE("compile_kernel") {
    REQUIRE_GPU_OR_SKIP();

    KernelBuilder builder = vector_add_builder();
    VectorAddData data(1000);
    ProblemKey problem("vector_add", 1000);

    Config config = builder.default_config();
    config.insert("block_size", 128);

    auto instance =
        compile_kernel<int, float*, const float*, const float*>(builder, problem, config);
    instance.launch(data.n, data.c(), data.a_ptr(), data.b_ptr());
    data.check_result();

    SECTION("invalid config") {
        config.insert("block_size", 100);

        CHECK_THROWS(
            compile_kernel<int, float*, const float*, const float*>(builder, problem, config));
    }
}

TEST_CASE("Kernel") {
    REQUIRE_GPU_OR_SKIP();

    auto compiler = std::make_shared<CountingCompiler>();
    auto wisdom = std::make_shared<MockWisdom>();
    Kernel kernel(vector_add_builder(), KernelSettings {compiler, wisdom});

    VectorAddData data(1000);
    kernel.launch(data.n, data.c(), data.a_ptr(), data.b_ptr());
    data.check_result();
    CHECK(compiler->count == 1);
    CHECK(wisdom->load_count == 1);
    CHECK(wisdom->capture_count == 0);

    SECTION("instances are cached per problem") {
        kernel.launch(data.n, data.c(), data.a_ptr(), data.b_ptr());
        CHECK(compiler->count == 1);

        VectorAddData smaller(10);
        kernel.launch(smaller.n, smaller.c(), smaller.a_ptr(), smaller.b_ptr());
        smaller.check_result();
        CHECK(compiler->count == 2);
        CHECK(wisdom->load_count == 2);
    }

    SECTION("compile") {
        auto instance = kernel.compile(data.n, data.c(), data.a_ptr(), data.b_ptr());
        CHECK(compiler->count == 1);

        GPUStream stream;
        REQUIRE(cuStreamCreate(&stream, CU_STREAM_DEFAULT) == CUDA_SUCCESS);
        instance.launch(stream, data.n, data.c(), data.a_ptr(), data.b_ptr());
        gpu_synchronize(stream);
        data.check_result();
        cuStreamDestroy(stream);
    }

    SECTION("copies share the cache") {
        Kernel copy = kernel;
        copy.launch(data.n, data.c(), data.a_ptr(), data.b_ptr());
        CHECK(compiler->count == 1);
    }
}

TEST_CASE("Kernel without wisdom uses default config") {
    REQUIRE_GPU_OR_SKIP();

    Kernel kernel(vector_add_builder());
    VectorAddData data(100);
    kernel.launch(data.n, data.c(), data.a_ptr(), data.b_ptr());
    data.check_result();
}

TEST_CASE("Kernel capture") {
    REQUIRE_GPU_OR_SKIP();

    auto wisdom = std::make_shared<MockWisdom>();
    wisdom->action = TuningAction::Capture;
    Kernel kernel(vector_add_builder(), KernelSettings {nullptr, wisdom});

    VectorAddData data(100);
    kernel.launch(data.n, data.c(), data.a_ptr(), data.b_ptr());
    data.check_result();

    REQUIRE(wisdom->capture_count == 1);
    CHECK(wisdom->captured_problem == ProblemKey("vector_add", 100));

    size_t nbytes = 100 * sizeof(float);
    REQUIRE(wisdom->captured_args.size() == 4);
    CHECK(wisdom->captured_args[0].get<int>() == 100);
    CHECK(wisdom->captured_args[1].buffer_size() == nbytes);

    REQUIRE(wisdom->inputs.size() == 4);
    REQUIRE(wisdom->outputs.size() == 4);
    CHECK(wisdom->inputs[0].empty());
    CHECK(wisdom->outputs[0].empty());

    // C is zero before the launch and A+B after it; A is not modified.
    std::vector<std::byte> zeros(nbytes);
    CHECK(wisdom->inputs[1] == zeros);
    std::vector<float> expected = data.result();
    REQUIRE(wisdom->outputs[1].size() == nbytes);
    CHECK(std::memcmp(wisdom->outputs[1].data(), expected.data(), nbytes) == 0);
    CHECK(std::memcmp(wisdom->inputs[2].data(), data.a.data(), nbytes) == 0);
    CHECK(wisdom->inputs[2] == wisdom->outputs[2]);

    // Only the first launch is captured.
    kernel.launch(data.n, data.c(), data.a_ptr(), data.b_ptr());
    CHECK(wisdom->capture_count == 1);
}
