#include "catch.hpp"
#include "kernel_launcher/file_loader.hpp"
#include "test_helpers.hpp"

using namespace kernel_launcher;
using test::MapLoader;

TEST_CASE("path_join") {
    CHECK(path_join("a", "b") == "a/b");
    CHECK(path_join("a/", "b") == "a/b");
    CHECK(path_join("a/b", "c/d") == "a/b/c/d");
    CHECK(path_join("", "b") == "b");
    CHECK(path_join("a", "/b") == "/b");
    CHECK(path_join("a", "") == "a/");
}

TEST_CASE("write_file") {
    std::string path = path_join(test::make_temp_dir(), "file.txt");
    DefaultLoader loader;

    CHECK(write_file(path, "hello"));
    CHECK(loader.load(path) == "hello");

    // Existing files are not overwritten by default.
    CHECK_FALSE(write_file(path, "world"));
    CHECK(loader.load(path) == "hello");

    CHECK(write_file(path, std::string_view("world"), true));
    CHECK(loader.load(path) == "world");

    CHECK(write_file(path, std::string_view("abc", 2), true));
    CHECK(loader.load(path) == "ab");

    CHECK_THROWS(write_file("/nonexistent-directory/file.txt", "hello"));
}

TEST_CASE("include_dirs_from_env") {
    const std::vector<std::string>& dirs = include_dirs_from_env();
    CHECK(&dirs == &include_dirs_from_env());

    for (const auto& dir : dirs) {
        CHECK_FALSE(dir.empty());
    }
}

TEST_CASE("FileLoader::load") {
    MapLoader loader;
    loader.files["a.cu"] = "A";

    CHECK(loader.load("a.cu") == "A");
    CHECK_THROWS_AS(loader.load("b.cu"), FileNotFoundException);
}

TEST_CASE("DefaultLoader") {
    SECTION("search directories") {
        DefaultLoader loader({"nonexistent", "assets"});

        auto content = loader.try_load("example_header.cuh");
        REQUIRE(content.has_value());
        CHECK(content->find("function_from_header") != std::string::npos);
        CHECK(loader.load("example_header.cuh") == *content);

        CHECK_FALSE(loader.try_load("missing.cuh").has_value());
        CHECK_THROWS_AS(loader.load("missing.cuh"), FileNotFoundException);
        CHECK_THROWS_WITH(
            loader.load("missing.cuh"),
            "could not find file 'missing.cuh' in directories: nonexistent, assets");
    }

    SECTION("directories are skipped") {
        // `./assets` is a directory, not a file.
        DefaultLoader loader({".", "assets"});
        CHECK_FALSE(loader.try_load("assets").has_value());
        CHECK(loader.try_load("example_header.cuh").has_value());
    }

    SECTION("no search directories") {
        DefaultLoader loader(std::vector<std::string> {});
        CHECK_FALSE(loader.try_load("assets/example_header.cuh").has_value());
        CHECK_THROWS_AS(loader.load("assets/example_header.cuh"), FileNotFoundException);
    }

    SECTION("default is the working directory") {
        DefaultLoader loader;
        CHECK(loader.try_load("assets/example_header.cuh").has_value());
    }

    SECTION("inner loader takes priority") {
        auto inner = std::make_shared<MapLoader>();
        inner->files["example_header.cuh"] = "override";
        inner->files["only_in_memory.cuh"] = "memory";

        DefaultLoader loader({"assets"}, inner);
        CHECK(loader.load("example_header.cuh") == "override");
        CHECK(loader.load("only_in_memory.cuh") == "memory");
        CHECK(loader.load("matmul_kernel.cu").find("matmul_kernel") != std::string::npos);
    }
}
