#include "catch.hpp"
#include "kernel_launcher/wisdom.hpp"
#include "test_helpers.hpp"

using namespace kernel_launcher;

static ConfigSpace vector_add_space(std::vector<Primitive> threads_per_block = {1, 32, 128, 256}) {
    ConfigSpace space;
    space.tune("threads_per_block", std::move(threads_per_block));
    space.tune("elements_per_thread", {1, 2, 4});
    return space;
}

TEST_CASE("load_wisdom_config") {
    // The default `GPUArch` matches none of the devices in the file.
    GPUArch arch;
    WisdomResult result;

    SECTION("best config") {
        ConfigSpace space = vector_add_space();
        Config config = load_wisdom_config(
            "assets",
            arch,
            space,
            ProblemKey("vector_add", 1000000000),
            &result);

        CHECK(result == WisdomResult::DeviceMismatch);
        CHECK(config["threads_per_block"] == 128);
        CHECK(config["elements_per_thread"] == 1);
    }

    SECTION("configs that are not in the space are skipped") {
        ConfigSpace space = vector_add_space({32, 256});
        Config config = load_wisdom_config(
            "assets",
            arch,
            space,
            ProblemKey("vector_add", 1000000000),
            &result);

        CHECK(result == WisdomResult::DeviceMismatch);
        CHECK(config["threads_per_block"] == 256);
        CHECK(config["elements_per_thread"] == 1);
    }

    SECTION("no valid configs") {
        ConfigSpace space = vector_add_space({64});
        Config config = load_wisdom_config(
            "assets",
            arch,
            space,
            ProblemKey("vector_add", 1000000000),
            &result);

        CHECK(result == WisdomResult::NotFound);
        CHECK(config == space.default_config());
    }

    SECTION("missing file") {
        ConfigSpace space = vector_add_space();
        Config config =
            load_wisdom_config("assets", arch, space, ProblemKey("missing", 1000), &result);

        CHECK(result == WisdomResult::NotFound);
        CHECK(config == space.default_config());
    }
}

TEST_CASE("load_wisdom_config matching") {
    std::string dir = test::make_temp_dir();
    GPUArch arch;
    WisdomResult result;

    ConfigSpace space;
    space.tune("x", {1, 2, 3, 4});
    space.tune("y", {"a", "b"});

    // `GPUArch()` has an empty name, so records with an empty `device_name`
    // are for the same device.
    std::string content = R"(
{"version": "1.0", "objective": "time", "key": "my/kernel", "tunable_parameters": ["x", "y"]}
{"config": [1, "a"], "problem_size": [100, 100], "time": 3.0, "environment": {"device_name": ""}}
{"config": [2, "b"], "problem_size": [100, 100], "time": 2.0, "environment": {"device_name": ""}}
{"config": [3, "a"], "problem_size": [1000, 1000], "time": 1.0, "environment": {"device_name": ""}}
{"config": [4, "b"], "problem_size": [100, 100], "time": 0.5, "environment": {"device_name": "other"}}
// comments and invalid lines are skipped
# this is also a comment
{"config": [1], "problem_size": [100, 100], "time": 0.1, "environment": {"device_name": ""}}
not json
)";
    content.erase(0, 1);  // Leading newline

    // Invalid characters in the tuning key are replaced in the file name.
    write_file(path_join(dir, "my_kernel.wisdom"), content);

    SECTION("exact match") {
        Config config =
            load_wisdom_config(dir, arch, space, ProblemKey("my/kernel", 100, 100), &result);
        CHECK(result == WisdomResult::Ok);
        CHECK(config["x"] == 2);
        CHECK(config["y"] == "b");
    }

    SECTION("nearest problem size") {
        Config config =
            load_wisdom_config(dir, arch, space, ProblemKey("my/kernel", 800, 800), &result);
        CHECK(result == WisdomResult::ProblemMismatch);
        CHECK(config["x"] == 3);
        CHECK(config["y"] == "a");
    }

    SECTION("tuning key in file does not match") {
        Config config =
            load_wisdom_config(dir, arch, space, ProblemKey("my_kernel", 100, 100), &result);
        CHECK(result == WisdomResult::NotFound);
        CHECK(config == space.default_config());
    }
}

TEST_CASE("load_wisdom_config invalid header") {
    std::string dir = test::make_temp_dir();
    GPUArch arch;
    WisdomResult result;

    ConfigSpace space;
    space.tune("x", {1, 2});

    std::string content =
        R"({"version": "0.1", "objective": "time", "key": "kernel", "tunable_parameters": ["x"]})"
        "\n"
        R"({"config": [2], "problem_size": [100], "time": 1.0, "environment": {"device_name": ""}})"
        "\n";
    write_file(path_join(dir, "kernel.wisdom"), content);

    Config config = load_wisdom_config(dir, arch, space, ProblemKey("kernel", 100), &result);
    CHECK(result == WisdomResult::NotFound);
    CHECK(config == space.default_config());
}

TEST_CASE("load_wisdom_config value types") {
    std::string dir = test::make_temp_dir();
    GPUArch arch;
    WisdomResult result;

    ConfigSpace space;
    space.tune("x", {1.0, 2.0});

    // The Python writer stores floats without a fractional part as integers.
    std::string content =
        R"({"version": "1.0", "objective": "time", "key": "kernel", "tunable_parameters": ["x"]})"
        "\n"
        R"({"config": [2], "problem_size": [100], "time": 1.0, "environment": {"device_name": ""}})"
        "\n";
    write_file(path_join(dir, "kernel.wisdom"), content);

    Config config = load_wisdom_config(dir, arch, space, ProblemKey("kernel", 100), &result);
    CHECK(result == WisdomResult::Ok);
    CHECK(config["x"] == 2.0);
    CHECK(config["x"].kind() == Primitive::Kind::Float64);
}
