#include "kernel_launcher/value.h"

#include "catch.hpp"

using namespace kernel_launcher;

TEST_CASE("test T") {
    using T = TunableValue;

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

    SECTION("operator== and operator!=") {
        CHECK(empty == empty);
        CHECK(empty != intval);
        CHECK(empty != floatval);
        CHECK(empty != boolval);
        CHECK(empty != strval);

        CHECK(intval != empty);
        CHECK(intval == intval);
        CHECK(intval != floatval);
        CHECK(intval != boolval);
        CHECK(intval != strval);

        CHECK(floatval != empty);
        CHECK(floatval != intval);
        CHECK(floatval == floatval);
        CHECK(floatval != boolval);
        CHECK(floatval != strval);

        CHECK(boolval != empty);
        CHECK(boolval != intval);
        CHECK(boolval != floatval);
        CHECK(boolval != strval);
        CHECK(boolval == boolval);

        CHECK(strval != empty);
        CHECK(strval != intval);
        CHECK(strval != floatval);
        CHECK(strval != boolval);
        CHECK(strval == strval);

        CHECK(trueval == trueval);
        CHECK(trueval != falseval);
        CHECK(falseval != trueval);
        CHECK(falseval == falseval);

        // int vs double vs bool comparisons
        CHECK(T(0) == T(0.0));
        CHECK(T(0) == T(false));
        CHECK(T(0) == T(0));
        CHECK(T(false) == T(0.0));
        CHECK(T(false) == T(false));
        CHECK(T(false) == T(0));
        CHECK(T(0.0) == T(0.0));
        CHECK(T(0.0) == T(false));
        CHECK(T(0.0) == T(0));

        CHECK(T(1) == T(1.0));
        CHECK(T(1) == T(true));
        CHECK(T(1) == T(1));
        CHECK(T(true) == T(1.0));
        CHECK(T(true) == T(true));
        CHECK(T(true) == T(1));
        CHECK(T(1.0) == T(1.0));
        CHECK(T(1.0) == T(true));
        CHECK(T(1.0) == T(1));

        CHECK(T(2) != T(3.0));
        CHECK(T(2) != T(true));
        CHECK(T(2) != T(3));
        CHECK(T(true) != T(3.0));
        CHECK(T(false) != T(true));
        CHECK(T(true) != T(3));
        CHECK(T(2.0) != T(3.0));
        CHECK(T(2.0) != T(false));
        CHECK(T(2.0) != T(3));

        // Some tricky int vs double corner cases
        int64_t x = std::numeric_limits<int64_t>::max();
        CHECK(T(x) != T(double(x)));

        double v = 1.5;
        CHECK_THROWS(T(v).to_integer());
        CHECK(T(v) == T(v));

        v = double(std::numeric_limits<uint64_t>::max());
        CHECK_THROWS(T(v).to_integer());
        CHECK(T(v) == T(v));

        v = std::numeric_limits<double>::max();
        CHECK_THROWS(T(v).to_integer());
        CHECK(T(v) == T(v));

        v = std::numeric_limits<double>::infinity();
        CHECK_THROWS(T(v).to_integer());
        CHECK(T(v) == T(v));

        v = std::numeric_limits<double>::quiet_NaN();
        CHECK_THROWS(T(v).to_integer());
        CHECK(T(v) != T(v));
    }

    // TODO: repeat for remaining operators

    SECTION("test TunableParam") {
        TunableParam x {"foo", {1, 2, 3}, 4};
        CHECK(x.name() == "foo");
        CHECK(x.default_value() == 4);
        CHECK(x[2] == 3);
        CHECK_THROWS(x[3]);
        CHECK(x.size() == 3);

        TunableParam y {"foo", {1, 2, 3}, 4};
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
}
