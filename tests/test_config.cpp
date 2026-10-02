#include <sstream>

#include "catch.hpp"
#include "kernel_launcher/config.hpp"

using namespace kernel_launcher;

// Resolves `ProblemSizeExpr` to a fixed value; everything else is unknown.
struct ProblemSizeEval: EvalContext {
    explicit ProblemSizeEval(int64_t n) : n(n) {}

    bool try_lookup(const Variable& v, Primitive& out) const override {
        if (dynamic_cast<const ProblemSizeExpr*>(&v) != nullptr) {
            out = n;
            return true;
        }

        return false;
    }

    int64_t n;
};

TEST_CASE("TunableParam") {
    ConfigSpace space;
    TunableParam x = space.tune("x", {1, 2, 3});
    TunableParam y = space.tune("y", {"a", "b"}, "b");

    CHECK(x.name() == "x");
    CHECK(x.to_string() == "x");
    CHECK(x.values() == std::vector<Primitive> {1, 2, 3});
    CHECK(x.default_value() == 1);
    CHECK(x.has_value(2));
    CHECK_FALSE(x.has_value(4));
    CHECK(x.has_value(2.0));

    CHECK(y.default_value() == "b");
    CHECK(y.has_value("a"));
    CHECK_FALSE(y.has_value("c"));

    CHECK(space.parameters().size() == 2);
    CHECK(space["x"].name() == "x");
    CHECK(space.at("y").default_value() == "b");
    CHECK_THROWS(space.at("z"));

    CHECK_THROWS_AS(space.tune("", {1}), std::invalid_argument);
    CHECK_THROWS_AS(space.tune("z", {}), std::invalid_argument);
    CHECK_THROWS_AS(space.tune("x", {1}), std::invalid_argument);  //duplicate
    CHECK_THROWS_AS(space.tune("z", {1, 2}, 3), std::invalid_argument);  // invalid default
    CHECK(space.parameters().size() == 2);
}

TEST_CASE("Config") {
    ConfigSpace space;
    TunableParam x = space.tune("x", {1, 2});
    TunableParam y = space.tune("y", {true, false});

    Config config;
    config.insert("x", 2);

    CHECK(config.size() == 1);
    CHECK(config["x"] == 2);
    CHECK(config.get("x") == 2);
    CHECK_THROWS(config.get("y"));

    Primitive out;
    CHECK(config.try_lookup(x, out));
    CHECK(out == 2);
    CHECK_FALSE(config.try_lookup(y, out));
    CHECK_FALSE(config.try_lookup(ProblemSizeExpr(0), out));
    CHECK(config.lookup(x) == 2);
    CHECK_THROWS(config.lookup(y));

    // Inserting an existing name replaces its value.
    config.insert("x", 1);
    CHECK(config.size() == 1);
    CHECK(config["x"] == 1);

    Config other;
    other.insert("x", 1);
    CHECK(config == other);
    CHECK(config.hash() == other.hash());
    CHECK(std::hash<Config> {}(config) == config.hash());

    other.insert("y", true);
    CHECK(config != other);

    std::ostringstream stream;
    stream << config;
    CHECK(stream.str() == "{x=1}");

    std::ostringstream empty_stream;
    empty_stream << Config();
    CHECK(empty_stream.str() == "{}");
}

TEST_CASE("Config equality/hash does not depend on insertion order") {
    Config a;
    a.insert("p", 1);
    a.insert("q", "foo");

    Config b;
    b.insert("q", "foo");
    b.insert("p", 1);

    CHECK(a == b);
    CHECK(a.hash() == b.hash());
}

TEST_CASE("ConfigSpace") {
    ConfigSpace space;
    TunableParam x = space.tune("x", {1, 2, 3, 4});
    TunableParam y = space.tune("y", {1, 2}, 2);
    space.restriction(x * y <= 4);

    Config def = space.default_config();
    CHECK(def.size() == 2);
    CHECK(def["x"] == 1);
    CHECK(def["y"] == 2);
    CHECK(space.contains(def));
    CHECK(space.is_valid(def));

    Config too_large;
    too_large.insert("x", 4);
    too_large.insert("y", 2);
    CHECK(space.contains(too_large));
    CHECK_FALSE(space.is_valid(too_large));

    Config invalid_value;
    invalid_value.insert("x", 5);
    invalid_value.insert("y", 1);
    CHECK_FALSE(space.contains(invalid_value));
    CHECK_FALSE(space.is_valid(invalid_value));

    Config missing;
    missing.insert("x", 1);
    CHECK_FALSE(space.contains(missing));

    Config extra = def;
    extra.insert("z", 1);
    CHECK_FALSE(space.contains(extra));

    std::vector<Config> configs = space.valid_configs();
    REQUIRE(configs.size() == 6);

    for (const auto& config : configs) {
        CHECK(space.is_valid(config));
    }

    // all configs are different.
    for (size_t i = 0; i < configs.size(); i++) {
        for (size_t j = 0; j < configs.size(); j++) {
            if (i == j) {
                CHECK(configs[i] == configs[j]);
            } else {
                CHECK(configs[i] != configs[j]);
            }
        }
    }

    // The last parameter changes fastest.
    CHECK(configs[0]["x"] == 1);
    CHECK(configs[0]["y"] == 1);
    CHECK(configs[1]["x"] == 1);
    CHECK(configs[1]["y"] == 2);
    CHECK(configs[5]["x"] == 4);
    CHECK(configs[5]["y"] == 1);
}

TEST_CASE("ConfigSpace::valid_configs of empty space") {
    ConfigSpace space;
    std::vector<Config> configs = space.valid_configs();

    REQUIRE(configs.size() == 1);
    CHECK(configs[0].size() == 0);
}

TEST_CASE("ConfigSpace::resolve") {
    ConfigSpace space;
    TunableParam x = space.tune("x", {1, 2, 4, 8});
    space.restriction(x <= problem_size_x());
    space.restriction(problem_size_x() > 0);

    // The problem size is unknown, so the restriction cannot be evaluated.
    Config config;
    config.insert("x", 1);
    CHECK_THROWS(space.is_valid(config));

    ConfigSpace resolved = space.resolve(ProblemSizeEval(4));
    CHECK(resolved.parameters().size() == 1);
    CHECK(resolved.is_valid(config));
    CHECK(resolved.valid_configs().size() == 3);

    // A restriction that resolves to `false` can never be satisfied.
    CHECK_THROWS(space.resolve(ProblemSizeEval(0)));
}
