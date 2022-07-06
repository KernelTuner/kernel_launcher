#include "kernel_launcher/fs.h"

#include "catch.hpp"

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