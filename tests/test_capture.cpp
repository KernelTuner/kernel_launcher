#include <cstring>
#include <fstream>

#include "catch.hpp"
#include "kernel_launcher/capture.hpp"
#include "nlohmann/json.hpp"
#include "test_helpers.hpp"

using namespace kernel_launcher;
using json = nlohmann::json;

static KernelBuilder vector_add_builder() {
    KernelBuilder builder("vector_add", FileSource("vector_add.cu", "source"));
    TunableParam bs = builder.tune("block_size", {32, 64});
    builder.restriction(bs <= problem_size_x());

    ArgumentExpr n(0);
    builder.tuning_key("vector/add")
        .problem_size(n, 1, 1)
        .buffers(ArgumentExpr(1)[n], ArgumentExpr(2)[n])
        .define("BLOCK_SIZE", bs)
        .template_arg(bs)
        .compiler_flag("-lineinfo")
        .include_header(FileSource("header.cuh", "header"))
        .assertion(n > 0);

    return builder;
}

static std::vector<std::byte> float_bytes(const std::vector<float>& values) {
    std::vector<std::byte> result(values.size() * sizeof(float));
    std::memcpy(result.data(), values.data(), result.size());
    return result;
}

static json read_json(const std::string& path) {
    std::ifstream stream(path);
    REQUIRE(stream.good());
    return json::parse(stream);
}

TEST_CASE("capture_file_exists") {
    std::string dir = test::make_temp_dir();

    CHECK_FALSE(capture_file_exists(dir, ProblemKey("kernel", 10)));

    write_file(path_join(dir, "kernel_10.json"), "{}");
    CHECK(capture_file_exists(dir, ProblemKey("kernel", 10)));
    CHECK_FALSE(capture_file_exists(dir, ProblemKey("kernel", 20)));

    write_file(path_join(dir, "kernel_10x20.json"), "{}");
    CHECK(capture_file_exists(dir, ProblemKey("kernel", 10, 20)));

    write_file(path_join(dir, "kernel_10x1x5.json"), "{}");
    CHECK(capture_file_exists(dir, ProblemKey("kernel", 10, 1, 5)));

    // Invalid characters in the tuning key are replaced.
    write_file(path_join(dir, "my_kernel_10.json"), "{}");
    CHECK(capture_file_exists(dir, ProblemKey("my/kernel", 10)));
}

TEST_CASE("export_capture_file") {
    std::string dir = test::make_temp_dir();
    KernelBuilder builder = vector_add_builder();

    std::vector<KernelArg> args = {
        into_kernel_arg(4),
        KernelArg::of((float*)nullptr, 4 * sizeof(float)),
        KernelArg::of((const float*)nullptr, 4 * sizeof(float))};
    ProblemKey problem("vector/add", 4);

    std::vector<std::vector<std::byte>> inputs = {
        {},
        float_bytes({0, 0, 0, 0}),
        float_bytes({1, 2, 3, 4})};

    SECTION("with outputs") {
        std::vector<std::vector<std::byte>> outputs = {
            {},
            float_bytes({2, 4, 6, 8}),
            float_bytes({1, 2, 3, 4})};

        export_capture_file(dir, builder, problem, args, inputs, outputs);
        REQUIRE(capture_file_exists(dir, problem));

        json capture = read_json(path_join(dir, "vector_add_4.json"));
        CHECK(capture["key"] == "vector/add");
        CHECK(capture["problem_size"] == json::array({4, 1, 1}));
        CHECK(capture["environment"].contains("nvrtc_version"));

        const json& kernel = capture["kernel"];
        CHECK(kernel["name"] == "vector_add");
        CHECK(kernel["source"] == "source");
        CHECK(kernel["compile_flags"] == json::array({"-lineinfo"}));
        CHECK(kernel["defines"]["BLOCK_SIZE"]["operator"] == "parameter");
        CHECK(kernel["template_args"].size() == 1);
        CHECK(kernel["headers"][0]["file"] == "header.cuh");
        CHECK(kernel["headers"][0]["content"] == "header");
        CHECK(kernel["block_size"].size() == 3);
        CHECK(kernel["grid_size"].size() == 3);

        const json& space = capture["config_space"];
        REQUIRE(space["parameters"].size() == 1);
        CHECK(space["parameters"][0]["name"] == "block_size");
        CHECK(space["parameters"][0]["values"] == json::array({32, 64}));
        CHECK(space["parameters"][0]["default"] == 32);
        // One restriction and one assertion.
        CHECK(space["restrictions"].size() == 2);

        const json& arguments = capture["arguments"];
        REQUIRE(arguments.size() == 3);

        CHECK(arguments[0]["kind"] == "scalar");
        CHECK(arguments[0]["type"] == "int");
        CHECK(arguments[0]["data"] == json::array({4, 0, 0, 0}));

        // C changed, so it has a separate reference file.
        CHECK(arguments[1]["kind"] == "array");
        CHECK(arguments[1]["type"] == "float*");
        CHECK(arguments[1].contains("reference_file"));
        CHECK(arguments[1]["reference_hash"] != arguments[1]["hash"]);

        // A did not change, so it has no reference file.
        CHECK(arguments[2]["kind"] == "array");
        CHECK_FALSE(arguments[2].contains("reference_file"));

        for (size_t i = 1; i < 3; i++) {
            std::ifstream data_file(path_join(dir, arguments[i]["file"].get<std::string>()));
            CHECK(data_file.good());
        }
    }

    SECTION("without outputs") {
        export_capture_file(dir, builder, problem, args, inputs);
        REQUIRE(capture_file_exists(dir, problem));

        json capture = read_json(path_join(dir, "vector_add_4.json"));
        CHECK_FALSE(capture["arguments"][1].contains("reference_file"));
    }

    SECTION("identical buffers share a data file") {
        inputs[1] = inputs[2];

        export_capture_file(dir, builder, problem, args, inputs);
        json capture = read_json(path_join(dir, "vector_add_4.json"));
        CHECK(capture["arguments"][1]["file"] == capture["arguments"][2]["file"]);
    }

    SECTION("invalid inputs are not written") {
        // Errors are logged, not thrown.
        inputs.pop_back();
        CHECK_NOTHROW(export_capture_file(dir, builder, problem, args, inputs));
        CHECK_FALSE(capture_file_exists(dir, problem));
    }

    SECTION("input size is not a multiple of the element size") {
        inputs[1].pop_back();
        export_capture_file(dir, builder, problem, args, inputs);
        CHECK_FALSE(capture_file_exists(dir, problem));
    }
}
