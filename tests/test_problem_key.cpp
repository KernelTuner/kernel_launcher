#include "catch.hpp"
#include "kernel_launcher/problem_key.hpp"

using namespace kernel_launcher;

TEST_CASE("FeatureVector") {
    FeatureVector features;
    features.add("a", 1);
    features.add("b", "foo");

    CHECK(features.get("a") == 1);
    CHECK(features.get("b") == "foo");
    CHECK_THROWS(features.get("c"));

    // Adding an existing key replaces its value.
    features.add("a", 2);
    CHECK(features.get("a") == 2);

    SECTION("equality does not depend on order") {
        FeatureVector other;
        other.add("b", "foo");
        other.add("a", 2);

        CHECK(features == other);
        CHECK(features.hash() == other.hash());
        CHECK(std::hash<FeatureVector> {}(features) == features.hash());
    }

    SECTION("different values") {
        FeatureVector other;
        other.add("a", 2);
        other.add("b", "bar");
        CHECK(features != other);
    }

    SECTION("different keys") {
        FeatureVector other;
        other.add("a", 2);
        CHECK(features != other);

        other.add("c", "foo");
        CHECK(features != other);
    }
}

TEST_CASE("ProblemKey") {
    ProblemKey key("vector_add", 100);
    CHECK(key.tuning_key == "vector_add");
    CHECK(key.problem_size == std::array<int64_t, 3> {100, 1, 1});

    ProblemKey same("vector_add", 100, 1, 1);
    CHECK(key == same);
    CHECK(key.hash() == same.hash());
    CHECK(std::hash<ProblemKey> {}(key) == key.hash());

    CHECK(key != ProblemKey("vector_sub", 100));
    CHECK(key != ProblemKey("vector_add", 200));
    CHECK(key != ProblemKey("vector_add", 100, 2));
    CHECK(key != ProblemKey("vector_add", 100, 1, 2));

    same.features.add("dtype", "float");
    CHECK(key != same);
    CHECK(same.features.get("dtype") == "float");

    key.features.add("dtype", "float");
    CHECK(key == same);
    CHECK(key.hash() == same.hash());
}
