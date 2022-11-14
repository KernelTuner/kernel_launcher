#include "kernel_launcher/value.h"

#include "catch.hpp"

using namespace kernel_launcher;

#define CHECK_INT(expression, v)      \
    CHECK((expression).is_integer()); \
    CHECK((expression) == Value(Value::integer_type {(v)}));

#define CHECK_LT(a, b)       \
    CHECK((a) < (b));        \
    CHECK_FALSE((b) < (a));  \
    CHECK_FALSE((a) > (b));  \
    CHECK((b) > (a));        \
    CHECK((a) <= (b));       \
    CHECK_FALSE((b) <= (a)); \
    CHECK_FALSE((a) >= (b)); \
    CHECK((b) >= (a));       \
    CHECK_FALSE((a) == (b)); \
    CHECK_FALSE((b) == (a)); \
    CHECK((a) != (b));       \
    CHECK((b) != (a));

#define CHECK_EQ(a, b)       \
    CHECK_FALSE((a) < (b));  \
    CHECK_FALSE((b) < (a));  \
    CHECK_FALSE((a) > (b));  \
    CHECK_FALSE((b) > (a));  \
    CHECK((a) <= (b));       \
    CHECK((b) <= (a));       \
    CHECK((a) >= (b));       \
    CHECK((b) >= (a));       \
    CHECK((a) == (b));       \
    CHECK((b) == (a));       \
    CHECK_FALSE((a) != (b)); \
    CHECK_FALSE((b) != (a));

#define CHECK_GT(a, b) CHECK_LT(b, a);

TEST_CASE("test Value") {
    using T = Value;

    T empty;
    T intval(300);
    T floatval(8.0);
    T boolval(true);
    T strval("abc");
    T trueval(true);
    T falseval(false);

    SECTION("basics conversion") {
        CHECK(empty.is_empty());
        CHECK(empty.is_bool() == false);
        CHECK(empty.is_string() == false);
        CHECK(empty.is_integer() == false);
        CHECK(empty.is_double() == false);
        CHECK(empty.to_bool() == false);  // ???
        CHECK(empty.to_string() == "");
        CHECK_THROWS(empty.to_integer());
        CHECK_THROWS(empty.to_double());

        CHECK(intval.is_empty() == false);
        CHECK(intval.is_bool() == false);
        CHECK(intval.is_string() == false);
        CHECK(intval.is_integer());
        CHECK(intval.is_double() == false);
        CHECK(intval.to_bool() == true);
        CHECK(intval.to_string() == "300");
        CHECK(intval.to_integer() == 300);
        CHECK(intval.to_double() == 300.0);

        CHECK(floatval.is_empty() == false);
        CHECK(floatval.is_bool() == false);
        CHECK(floatval.is_string() == false);
        CHECK(floatval.is_integer() == false);
        CHECK(floatval.is_double());
        CHECK(floatval.to_bool() == true);
        CHECK(floatval.to_string() == "8.000000");
        CHECK(floatval.to_integer() == 8);
        CHECK(floatval.to_double() == 8.0);

        CHECK(boolval.is_empty() == false);
        CHECK(boolval.is_bool());
        CHECK(boolval.is_string() == false);
        CHECK(boolval.is_integer() == false);
        CHECK(boolval.is_double() == false);
        CHECK(boolval.to_bool() == true);
        CHECK(boolval.to_string() == "true");
        CHECK(boolval.to_integer() == 1);
        CHECK(boolval.to_double() == 1.0);

        CHECK(strval.is_empty() == false);
        CHECK(strval.is_bool() == false);
        CHECK(strval.is_string());
        CHECK(strval.is_integer() == false);
        CHECK(strval.is_double() == false);
        CHECK(strval.to_bool() == true);
        CHECK(strval.to_string() == "abc");
        CHECK_THROWS(strval.to_integer());
        CHECK_THROWS(strval.to_double());
    }

    SECTION("generic conversion") {
        CHECK(intval.is<std::string>() == false);
        CHECK(intval.is<bool>() == false);
        CHECK(intval.is<double>() == false);
        CHECK(intval.is_int16() == true);
        CHECK(intval.is_int8() == false);

        CHECK(intval.to<std::string>() == "300");
        CHECK(intval.to<bool>() == true);
        CHECK(intval.to<double>() == 300.0);
        CHECK(intval.to_int16() == 300);
        CHECK(intval.to_int16() == 300);
    }

    SECTION("operator+") {
        CHECK_THROWS(empty + empty);
        CHECK_THROWS(empty + intval);
        CHECK_THROWS(empty + floatval);
        CHECK_THROWS(empty + boolval);
        CHECK_THROWS(empty + strval);

        CHECK_THROWS(intval + empty);
        CHECK(intval + intval == T(600));
        CHECK(intval + floatval == T(300.0 + 8.0));
        CHECK(intval + boolval == T(300 + 1));
        CHECK_THROWS(intval + strval);

        CHECK_THROWS(floatval + empty);
        CHECK(floatval + intval == T(300.0 + 8.0));
        CHECK(floatval + floatval == T(8.0 + 8.0));
        CHECK(floatval + boolval == T(8.0 + 1.0));
        CHECK_THROWS(floatval + strval);

        CHECK_THROWS(boolval + empty);
        CHECK(boolval + intval == T(300 + 1));
        CHECK(boolval + floatval == T(8.0 + 1.0));
        CHECK_THROWS(boolval + strval);
        CHECK(boolval * boolval == trueval);

        CHECK_THROWS(strval + empty);
        CHECK_THROWS(strval + intval);
        CHECK_THROWS(strval + floatval);
        CHECK_THROWS(strval + boolval);
        CHECK(strval + strval == "abcabc");

        CHECK(trueval + trueval == trueval);
        CHECK(trueval + falseval == trueval);
        CHECK(falseval + trueval == trueval);
        CHECK(falseval + falseval == falseval);
    }

    SECTION("operator*") {
        CHECK_THROWS(empty * empty);
        CHECK_THROWS(empty * intval);
        CHECK_THROWS(empty * floatval);
        CHECK_THROWS(empty * boolval);
        CHECK_THROWS(empty * strval);

        CHECK_THROWS(intval * empty);
        CHECK(intval * intval == T(300 * 300));
        CHECK(intval * floatval == T(300.0 * 8.0));
        CHECK(intval * boolval == T(300));
        CHECK_THROWS(intval * strval);

        CHECK_THROWS(floatval * empty);
        CHECK(floatval * intval == T(300.0 * 8.0));
        CHECK(floatval * floatval == T(8.0 * 8.0));
        CHECK(floatval * boolval == T(8.0));
        CHECK_THROWS(floatval * strval);

        CHECK_THROWS(boolval * empty);
        CHECK(boolval * intval == T(300));
        CHECK(boolval * floatval == T(8.0));
        CHECK_THROWS(boolval * strval);
        CHECK(boolval * boolval == trueval);

        CHECK_THROWS(strval * empty);
        CHECK_THROWS(strval * intval);
        CHECK_THROWS(strval * floatval);
        CHECK_THROWS(strval * boolval);
        CHECK_THROWS(strval * strval);

        CHECK(trueval * trueval == trueval);
        CHECK(trueval * falseval == falseval);
        CHECK(falseval * trueval == falseval);
        CHECK(falseval * falseval == falseval);
    }

    SECTION("comparison operator") {
        CHECK_EQ(empty, empty);
        CHECK_LT(empty, intval);
        CHECK_LT(empty, floatval);
        CHECK_LT(empty, boolval);
        CHECK_LT(empty, strval);

        CHECK_GT(intval, empty);
        CHECK_EQ(intval, intval);
        CHECK_GT(intval, floatval);
        CHECK_GT(intval, boolval);
        CHECK_LT(intval, strval);

        CHECK_GT(floatval, empty);
        CHECK_LT(floatval, intval);
        CHECK_EQ(floatval, floatval);
        CHECK_GT(floatval, boolval);
        CHECK_LT(floatval, strval);

        CHECK_GT(boolval, empty);
        CHECK_LT(boolval, intval);
        CHECK_LT(boolval, floatval);
        CHECK_LT(boolval, strval);
        CHECK_EQ(boolval, boolval);

        CHECK_GT(strval, empty);
        CHECK_GT(strval, intval);
        CHECK_GT(strval, floatval);
        CHECK_GT(strval, boolval);
        CHECK_EQ(strval, strval);

        CHECK_EQ(trueval, trueval);
        CHECK_GT(trueval, falseval);
        CHECK_LT(falseval, trueval);
        CHECK_EQ(falseval, falseval);

        // int vs double vs bool comparisons
        CHECK_EQ(T(0), T(0.0));
        CHECK_EQ(T(0), T(false));
        CHECK_EQ(T(0), T(0));
        CHECK_EQ(T(false), T(0.0));
        CHECK_EQ(T(false), T(false));
        CHECK_EQ(T(false), T(0));
        CHECK_EQ(T(0.0), T(0.0));
        CHECK_EQ(T(0.0), T(false));
        CHECK_EQ(T(0.0), T(0));

        CHECK_EQ(T(1), T(1.0));
        CHECK_EQ(T(1), T(true));
        CHECK_EQ(T(1), T(1));
        CHECK_EQ(T(true), T(1.0));
        CHECK_EQ(T(true), T(true));
        CHECK_EQ(T(true), T(1));
        CHECK_EQ(T(1.0), T(1.0));
        CHECK_EQ(T(1.0), T(true));
        CHECK_EQ(T(1.0), T(1));

        CHECK_LT(T(2), T(3.0));
        CHECK_GT(T(2), T(true));
        CHECK_LT(T(2), T(3));
        CHECK_LT(T(true), T(3.0));
        CHECK_LT(T(false), T(true));
        CHECK_LT(T(true), T(3));
        CHECK_LT(T(2.0), T(3.0));
        CHECK_GT(T(2.0), T(false));
        CHECK_LT(T(2.0), T(3));

        // Some tricky int vs double corner cases
        int64_t x = std::numeric_limits<int64_t>::max();
        CHECK(T(x) != T(double(x)));

        double v = 1.5;
        CHECK_THROWS(T(v).to_integer());
        CHECK_EQ(T(v), T(v));

        v = double(std::numeric_limits<uint64_t>::max());
        CHECK_THROWS(T(v).to_integer());
        CHECK_EQ(T(v), T(v));

        v = std::numeric_limits<double>::max();
        CHECK_THROWS(T(v).to_integer());
        CHECK_EQ(T(v), T(v));

        v = std::numeric_limits<double>::infinity();
        CHECK_THROWS(T(v).to_integer());
        CHECK_EQ(T(v), T(v));

        v = std::numeric_limits<double>::quiet_NaN();
        CHECK_THROWS(T(v).to_integer());
        CHECK(T(v) != T(v));
    }

    SECTION("test total order") {
        // Check that there is a total order on `Value`s
        std::vector<Value> sorted = {
            empty,
            -std::numeric_limits<double>::infinity(),
            std::numeric_limits<int64_t>::min(),
            -100,
            -50.0,
            0.0,
            0,
            false,
            1,
            true,
            100,
            std::numeric_limits<int64_t>::max(),
            std::numeric_limits<double>::infinity(),
            "",
            " ",
            "0",
            "false",
            "string",
            "true"};

        for (size_t i = 0; i < sorted.size(); i++) {
            for (size_t j = 0; j < sorted.size(); j++) {
                if (i < j) {
                    CHECK(sorted[i] <= sorted[j]);
                } else if (i > j) {
                    CHECK(sorted[i] >= sorted[j]);
                }
            }
        }
    }

    SECTION("test TunableParam") {
        TunableParam x {"foo", {1, 2, 3}, 3};
        CHECK(x.name() == "foo");
        CHECK(x.default_value() == 3);
        CHECK(x[2] == 3);
        CHECK_THROWS(x[3]);
        CHECK(x.size() == 3);

        TunableParam y {"foo", {1, 2, 3}, 3};
        TunableParam z = x;
        CHECK(x != y);
        CHECK(x == z);
    }

    // Check if overflow is detected for +, -, *, /
    SECTION("test overflow") {
        static int64_t MIN = std::numeric_limits<int64_t>::min();
        static int64_t MAX = std::numeric_limits<int64_t>::max();

        CHECK_THROWS(T(MAX) + T(1));
        CHECK(T(MAX) + T(-1) == T(MAX - 1));
        CHECK(T(MAX) + T(0) == T(MAX));
        CHECK(T(MAX - 1) + T(1) == T(MAX));
        CHECK_THROWS(T(MIN) + T(-1));
        CHECK(T(MIN) + T(1) == T(MIN + 1));
        CHECK(T(MIN) + T(MAX) == T(-1));
        CHECK(T(MAX / 2) + T(MAX / 2) == T(MAX - 1));
        CHECK(T(MAX / 2 + 1) + T(MAX / 2) == T(MAX));
        CHECK_THROWS(T(MAX / 2 + 1) + T(MAX / 2 + 1));

        CHECK_THROWS(T(MIN) - T(1));
        CHECK(T(MIN) - T(-1));
        CHECK(T(MIN) - T(0));
        CHECK(T(MIN + 1) - T(1));
        CHECK(T(MIN) - T(-1));
        CHECK_THROWS(T(MIN) - T(1));
        CHECK_THROWS(T(MIN) - T(MAX));
        CHECK(T(MIN / 2) - T(-(MIN / 2)) == T(MIN));
        CHECK_THROWS(T(MIN / 2 - 1) - T(-(MIN / 2) + 1));
        CHECK_THROWS(T(MIN / 2 - 1) - T(-(MIN / 2)));

        CHECK(T(MAX) * T(1) == T(MAX));
        CHECK(T(MAX) * T(0) == T(0));
        CHECK_THROWS(T(MAX) * T(MAX));
        CHECK_THROWS(T(MIN) * T(-1));
        CHECK(T(MAX) * T(-1) == T(-MAX));
        CHECK(T(1L << 31) * T(1L << 31));
        int64_t half = 3037000499;  // sqrt of MAX
        CHECK(T(half) * T(half) == T(half * half));
        CHECK(T(half) * T(half + 1) == T(half * (half + 1)));
        CHECK_THROWS(T(half + 1) * T(half + 1));

        CHECK(T(1) / T(1) == T(1));
        CHECK(T(0) / T(1) == T(0));
        CHECK(T(-1) / T(1) == T(-1));
        CHECK_THROWS(T(1) / T(0));
        CHECK_THROWS(T(0) / T(0));
        CHECK_THROWS(T(-1) / T(0));
        CHECK(T(1) / T(-1) == T(-1));
        CHECK(T(0) / T(-1) == T(0));
        CHECK(T(-1) / T(-1) == T(1));
        CHECK(T(MAX) / T(-1) == T(-MAX));
        CHECK_THROWS(T(MIN) / T(-1));  // corner case
    }

    SECTION("round ceil floor") {
        // boolean
        CHECK_INT(Value(true).round(), 1);
        CHECK_INT(Value(true).floor(), 1);
        CHECK_INT(Value(true).ceil(), 1);

        CHECK_INT(Value(false).round(), 0);
        CHECK_INT(Value(false).floor(), 0);
        CHECK_INT(Value(false).ceil(), 0);

        // string
        CHECK_THROWS(Value("test").round());
        CHECK_THROWS(Value("test").floor());
        CHECK_THROWS(Value("test").ceil());

        // empty
        CHECK_THROWS(Value().round());
        CHECK_THROWS(Value().floor());
        CHECK_THROWS(Value().ceil());

        // integer
        CHECK_INT(Value(0).round(), 0);
        CHECK_INT(Value(0).floor(), 0);
        CHECK_INT(Value(0).ceil(), 0);

        CHECK_INT(Value(123).round(), 123);
        CHECK_INT(Value(123).floor(), 123);
        CHECK_INT(Value(123).ceil(), 123);

        // double
        CHECK_INT(Value(0.0).round(), 0);
        CHECK_INT(Value(0.0).floor(), 0);
        CHECK_INT(Value(0.0).ceil(), 0);

        CHECK_INT(Value(0.5).round(), 1);
        CHECK_INT(Value(0.5).floor(), 0);
        CHECK_INT(Value(0.5).ceil(), 1);

        CHECK_INT(Value(-0.5).round(), -1);
        CHECK_INT(Value(-0.5).floor(), -1);
        CHECK_INT(Value(-0.5).ceil(), 0);

        CHECK_THROWS(Value(std::numeric_limits<double>::quiet_NaN()).round());
        CHECK_THROWS(Value(std::numeric_limits<double>::quiet_NaN()).floor());
        CHECK_THROWS(Value(std::numeric_limits<double>::quiet_NaN()).ceil());

        CHECK_THROWS(Value(std::numeric_limits<double>::infinity()).round());
        CHECK_THROWS(Value(std::numeric_limits<double>::infinity()).floor());
        CHECK_THROWS(Value(std::numeric_limits<double>::infinity()).ceil());
    }
}
