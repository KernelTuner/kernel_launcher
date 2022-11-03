#include "kernel_launcher/config.h"

#include "catch.hpp"

using namespace kernel_launcher;

TEST_CASE("Config") {
    Config c;
    TunableParam param_foo("foo", {1, 2, 3}, 1);
    TunableParam param_bar("bar", {true, false}, true);
    TunableParam param_foo2("foo", {1, 2, 3}, 1);

    CHECK(c.size() == 0);
    CHECK(c == c);
    CHECK(c == Config {});
    CHECK_THROWS(c["foo"]);
    CHECK_THROWS(c[param_foo]);

    CHECK_NOTHROW(c.insert(param_foo, 2));
    CHECK(c.size() == 1);
    CHECK(c["foo"] == 2);
    CHECK(c[param_foo] == 2);
    CHECK(c[ParamExpr(param_foo)] == 2);
    CHECK_THROWS(c[param_bar]);
    CHECK_THROWS(c[param_foo2]);

    // Even though param_foo and param_foo2 have the same name, they are
    // not identical since configs match on identity.
    Config c2;
    c2.insert(param_foo2, 2);
    CHECK(param_foo != param_foo2);
    CHECK(c != c2);

    // Can only insert key with same name if it is the same parameter.
    CHECK_NOTHROW(c.insert(param_foo, 3));
    CHECK_THROWS(c.insert(param_foo2, 3));

    CHECK_NOTHROW(c.insert(param_bar, true));
    CHECK(c.size() == 2);
    CHECK(c["bar"] == true);
    CHECK(c[param_bar] == true);
}

TEST_CASE("ConfigSpace") {
    ConfigSpace c;

    SECTION("add") {
        // empty name
        CHECK_THROWS(c.tune("", {1, 2, 3}, 2));

        // empty parameters
        CHECK_THROWS(c.tune("a", {}, 2));

        // invalid default value
        CHECK_THROWS(c.tune("a", {1, 2, 3}, 4));

        // valid
        auto a = c.tune("a", {1, 2, 3}, 1).parameter();

        CHECK(a.name() == "a");
        CHECK(a.default_value() == 1);
        CHECK(a.values() == std::vector<Value> {1, 2, 3});
        CHECK(c.at("a").parameter() == a);

        // cannot add same parameter twice
        CHECK_THROWS(c.tune("a", {5, 6, 7}, 8));
    }

    SECTION("different tune methods") {
        // initializer list
        CHECK_THROWS(c.tune("a", {}));
        CHECK_NOTHROW(c.tune("a", {1, 2, 3}));
        CHECK_NOTHROW(c.tune("b", {1, 2, 3}, 2));

        // collections
        CHECK_THROWS(c.tune("c", std::vector<int> {}));
        CHECK_NOTHROW(c.tune("c", std::vector<int> {1, 2, 3}));
        CHECK_NOTHROW(c.tune("d", std::vector<int> {1, 2, 3}, 2));

        // long version
        std::vector<int> values {1, 2, 3};
        CHECK_NOTHROW(c.tune("e", values, 2));
        CHECK_NOTHROW(c.add("f", {1, 2, 3}, {1, 1, 1}, 2));
    }

    auto x = c.tune("x", {1, 2, 3}, 1);
    auto y = c.tune("y", {4, 5, 6}, 5);
    auto z = c.tune("z", {7, 8, 9}, 9);

    SECTION("at") {
        CHECK(c.at("x").parameter() == x.parameter());
        CHECK(c.at("y").parameter() == y.parameter());
        CHECK(c.at("z").parameter() == z.parameter());
        CHECK_THROWS(c.at("unknown").parameter());
    }

    SECTION("default_config") {
        auto config = c.default_config();
        CHECK(config.size() == 3);
        CHECK(config[x] == 1);
        CHECK(config[y] == 5);
        CHECK(config[z] == 9);
    }

    SECTION("is_valid") {
        auto config = Config();
        CHECK(c.is_valid(config) == false);

        config.insert(x, 2);
        CHECK(c.is_valid(config) == false);

        config.insert(y, 4);
        CHECK(c.is_valid(config) == false);

        config.insert(z, 7);
        CHECK(c.is_valid(config) == true);

        // `is_valid` just ignores any additional config variables
        TunableParam w("w", {true, false}, true);
        config.insert(w, true);
        CHECK(c.is_valid(config) == true);
    }

    SECTION("restrictions") {
        c.restriction(x * 2 < y);

        auto valid = Config();
        valid.insert(x, 1);
        valid.insert(y, 4);
        valid.insert(z, 7);
        CHECK(c.is_valid(valid) == true);

        // Dpes not satisify "x * 2 < y"
        auto invalid = Config();
        invalid.insert(x, 3);
        invalid.insert(y, 4);
        invalid.insert(z, 7);
        CHECK(c.is_valid(invalid) == false);
    }
}