#include "kernel_launcher/fs.h"

#include <filesystem>

#include "catch.hpp"
#include "test_utils.h"

using namespace kernel_launcher;

TEST_CASE("test fs") {
    SECTION("path_join") {
        CHECK(path_join("foo/bar", "baz/klazz") == "foo/bar/baz/klazz");

        // All kinds of edge cases
        CHECK(path_join("a", "b") == "a/b");
        CHECK(path_join("a", "") == "a/");
        CHECK(path_join("", "b") == "b");
        CHECK(path_join("a/", "b") == "a/b");
        CHECK(path_join("a/", "") == "a/");

        CHECK(path_join("/", "b") == "/b");
        CHECK(path_join("/a", "b") == "/a/b");
        CHECK(path_join("/a", "") == "/a/");
        CHECK(path_join("/", "b") == "/b");
        CHECK(path_join("/a/", "b") == "/a/b");
        CHECK(path_join("/a/", "") == "/a/");
        CHECK(path_join("//", "b") == "//b");

        CHECK(path_join("a", "/b") == "/b");
        CHECK(path_join("a", "/") == "/");
        CHECK(path_join("", "/b") == "/b");
        CHECK(path_join("a/", "/b") == "/b");
        CHECK(path_join("a/", "/") == "/");

        CHECK(path_join("/", "/b") == "/b");
        CHECK(path_join("/a", "/b") == "/b");
        CHECK(path_join("/a", "/") == "/");
        CHECK(path_join("/", "/b") == "/b");
        CHECK(path_join("/a/", "/b") == "/b");
        CHECK(path_join("/a/", "/") == "/");
        CHECK(path_join("//", "/b") == "/b");
    }
}

TEST_CASE("test read_file/write_file") {
    std::string filename = assets_directory() + "/temporary_file.txt";

    // delete file if, for example, previous test crashed
    std::filesystem::remove(filename);

    SECTION("write/read simple") {
        // write 3 bytes
        std::vector<char> expected = {1, 2, 3};
        CHECK(write_file(filename, expected));

        // check if file exists
        CHECK(std::filesystem::exists(filename));

        // read data into buffer
        std::vector<char> data;
        CHECK(read_file(filename, data));
        CHECK(data == expected);
    }

    SECTION("write/read empty") {
        // write 0 bytes
        std::vector<char> expected = {};
        CHECK(write_file(filename, expected));

        // check if file exists
        CHECK(std::filesystem::exists(filename));

        // read empty data into buffer
        std::vector<char> data;
        CHECK(read_file(filename, data) == true);
        CHECK(data == expected);
    }

    SECTION("read non-existing file") {
        std::vector<char> data;
        CHECK(read_file(filename, data) == false);
        CHECK(data.empty());
    }

    SECTION("write existing file") {
        // write 3 bytes
        std::vector<char> expected = {1, 2, 3};
        CHECK(write_file(filename, expected));

        // check if file exists
        CHECK(std::filesystem::exists(filename));

        expected = {4, 5};

        SECTION("overwrite=false") {
            // this should fail, we cannot overwrite it
            CHECK(write_file(filename, expected, false) == false);
        }

        SECTION("overwrite=true") {
            // this should succeed since we set the `overwrite` flag
            CHECK(write_file(filename, expected, true));

            // check if the data was overwritten
            std::vector<char> data;
            CHECK(read_file(filename, data));
            CHECK(expected == data);
        }
    }

    // cleanup
    std::filesystem::remove(filename);
}