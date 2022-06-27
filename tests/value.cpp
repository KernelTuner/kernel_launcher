#include "kernel_launcher/value.h"

#include "catch.hpp"

using namespace kernel_launcher;

TEST_CASE("test TunableValue") {
    TunableValue empty;
    TunableValue intval(300);
    TunableValue floatval(8.0);
    TunableValue boolval(true);
    TunableValue strval("abc");
    TunableValue trueval(true);
    TunableValue falseval(false);

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
        CHECK_THROWS(floatval.to_integer());
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
        CHECK(intval + intval == TunableValue(600));
        CHECK(intval + floatval == TunableValue(300.0 + 8.0));
        CHECK(intval + boolval == TunableValue(300 + 1));
        CHECK_THROWS(intval + strval);

        CHECK_THROWS(floatval + empty);
        CHECK(floatval + intval == TunableValue(300.0 + 8.0));
        CHECK(floatval + floatval == TunableValue(8.0 + 8.0));
        CHECK(floatval + boolval == TunableValue(8.0 + 1.0));
        CHECK_THROWS(floatval + strval);

        CHECK_THROWS(boolval + empty);
        CHECK(boolval + intval == TunableValue(300 + 1));
        CHECK(boolval + floatval == TunableValue(8.0 + 1.0));
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
        CHECK(intval * intval == TunableValue(300 * 300));
        CHECK(intval * floatval == TunableValue(300.0 * 8.0));
        CHECK(intval * boolval == TunableValue(300));
        CHECK_THROWS(intval * strval);

        CHECK_THROWS(floatval * empty);
        CHECK(floatval * intval == TunableValue(300.0 * 8.0));
        CHECK(floatval * floatval == TunableValue(8.0 * 8.0));
        CHECK(floatval * boolval == TunableValue(8.0));
        CHECK_THROWS(floatval * strval);

        CHECK_THROWS(boolval * empty);
        CHECK(boolval * intval == TunableValue(300));
        CHECK(boolval * floatval == TunableValue(8.0));
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

    SECTION("operator==") {}

    // TODO: repeat for remaining operators
}
