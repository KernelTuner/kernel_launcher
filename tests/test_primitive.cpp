#include <cmath>
#include <limits>

#include "catch.hpp"
#include "kernel_launcher/primitive.hpp"
#include "kernel_launcher/type_info.hpp"

using namespace kernel_launcher;
using Kind = Primitive::Kind;

namespace test_primitive {
enum struct Color : int { Red = 1, Green = 2 };
struct NotPrimitive {};
}  // namespace test_primitive

static constexpr std::intmax_t INT_MAX_VALUE = std::numeric_limits<std::intmax_t>::max();
static constexpr std::intmax_t INT_MIN_VALUE = std::numeric_limits<std::intmax_t>::min();

TEST_CASE("Primitive construction") {
    CHECK(Primitive().kind() == Kind::Empty);
    CHECK(Primitive().is_empty());

    CHECK(Primitive(1).kind() == Kind::Int64);
    CHECK(Primitive(1u).kind() == Kind::Int64);
    CHECK(Primitive(int8_t(-1)).kind() == Kind::Int64);
    CHECK(Primitive(uint64_t(INT_MAX_VALUE)).kind() == Kind::Int64);
    CHECK_THROWS_AS(Primitive(std::numeric_limits<uint64_t>::max()), std::overflow_error);

    CHECK(Primitive(1.5).kind() == Kind::Float64);
    CHECK(Primitive(1.5f).kind() == Kind::Float64);

    CHECK(Primitive(true).kind() == Kind::Bool);

    CHECK(Primitive("abc").kind() == Kind::String);
    CHECK(Primitive(std::string("abc")).kind() == Kind::String);
    CHECK(Primitive(std::string_view("abc")).kind() == Kind::String);
    CHECK(Primitive(Symbol("abc")).kind() == Kind::String);

    // A null pointer is not a string.
    CHECK(Primitive(nullptr).is_empty());
    CHECK(Primitive(static_cast<const char*>(nullptr)).is_empty());

    CHECK(Primitive::from_or_empty(5) == Primitive(5));
    CHECK(Primitive::from_or_empty(test_primitive::NotPrimitive {}).is_empty());
}

TEST_CASE("Primitive kinds") {
    CHECK(Primitive().kind() == Primitive::Kind::Empty);
    CHECK(Primitive(1).kind() == Primitive::Kind::Int64);
    CHECK(Primitive(1.0).kind() == Primitive::Kind::Float64);
    CHECK(Primitive(true).kind() == Primitive::Kind::Bool);
    CHECK(Primitive("x").kind() == Primitive::Kind::String);

    CHECK(Primitive().kind_name() == std::string("empty"));
    CHECK(Primitive(1).kind_name() == std::string("int"));
    CHECK(Primitive(1.0).kind_name() == std::string("float"));
    CHECK(Primitive(true).kind_name() == std::string("bool"));
    CHECK(Primitive("x").kind_name() == std::string("string"));

    CHECK_FALSE(Primitive().is_numeric());
    CHECK(Primitive(1).is_numeric());
    CHECK(Primitive(1.0).is_numeric());
    CHECK(Primitive(true).is_numeric());
    CHECK_FALSE(Primitive("1").is_numeric());

    std::ostringstream stream;
    stream << Kind::Float64;
    CHECK(stream.str() == "float");
}

TEST_CASE("Primitive conversions") {
    SECTION("to_integer") {
        CHECK(Primitive(42).to_integer() == 42);
        CHECK(Primitive(3.0).to_integer() == 3);
        CHECK(Primitive(-3.0).to_integer() == -3);
        // Largest double below 2^63; `double(INT_MAX_VALUE)` itself rounds up to 2^63.
        CHECK(Primitive(double(INT_MAX_VALUE - 1023)).to_integer() == INT_MAX_VALUE - 1023);
        CHECK(Primitive(true).to_integer() == 1);
        CHECK(Primitive(false).to_integer() == 0);

        CHECK_THROWS(Primitive(3.5).to_integer());
        CHECK_THROWS(Primitive(1e30).to_integer());
        CHECK_THROWS(Primitive(std::nan("")).to_integer());
        CHECK_THROWS(Primitive(INFINITY).to_integer());
        CHECK_THROWS(Primitive().to_integer());
        CHECK_THROWS(Primitive("1").to_integer());
        CHECK(Primitive(-double(INT_MAX_VALUE)).to_integer());
    }

    SECTION("to_float") {
        CHECK(Primitive(42).to_float() == 42.0);
        CHECK(Primitive(1.5).to_float() == 1.5);
        CHECK(Primitive(true).to_float() == 1.0);

        CHECK_THROWS(Primitive().to_float());
        CHECK_THROWS(Primitive("1.0").to_float());
    }

    SECTION("to_bool") {
        CHECK_FALSE(Primitive().to_bool());
        CHECK_FALSE(Primitive(0).to_bool());
        CHECK(Primitive(2).to_bool());
        CHECK_FALSE(Primitive(0.0).to_bool());
        CHECK(Primitive(0.5).to_bool());
        CHECK(Primitive(true).to_bool());
        CHECK_FALSE(Primitive(false).to_bool());
        CHECK_FALSE(Primitive("").to_bool());
        CHECK(Primitive("x").to_bool());
    }

    SECTION("to_string") {
        CHECK(Primitive().to_string() == "");
        CHECK(Primitive(-42).to_string() == "-42");
        CHECK(Primitive(1.5).to_string() == "1.5");
        CHECK(Primitive(2.0).to_string() == "2.0");
        CHECK(Primitive(0.1).to_string() == "0.1");
        CHECK(Primitive(1e-7).to_string() == "1e-07");
        CHECK(Primitive(1e30).to_string() == "1e+30");
        CHECK(Primitive(1.0 / 3.0).to_string() == "0.3333333333333333");
        CHECK(Primitive(std::nan("")).to_string() == "nan");
        CHECK(Primitive(-double(INFINITY)).to_string() == "-inf");
        CHECK(Primitive(1e-7).to_symbol() != Primitive(2e-7).to_symbol());
        CHECK(Primitive(true).to_string() == "true");
        CHECK(Primitive("abc").to_string() == "abc");

        std::ostringstream stream;
        stream << Primitive(7);
        CHECK(stream.str() == "7");
    }

    SECTION("to<T>") {
        CHECK(Primitive(300).to<int>() == 300);
        CHECK(Primitive(300).to<uint16_t>() == 300);
        CHECK(Primitive(2.0).to<long>() == 2);
        CHECK_THROWS_AS(Primitive(300).to<int8_t>(), std::overflow_error);
        CHECK_THROWS_AS(Primitive(-1).to<unsigned int>(), std::overflow_error);

        CHECK(Primitive(2).to<double>() == 2.0);
        CHECK(Primitive(0.5).to<float>() == 0.5f);
        CHECK(Primitive(1).to<bool>());
        CHECK(Primitive(5).to<std::string>() == "5");
        CHECK(Primitive("abc").to<Symbol>() == Symbol("abc"));
        CHECK(Primitive(5).to<Primitive>() == Primitive(5));
    }
}

TEST_CASE("Primitive rounding") {
    CHECK(Primitive(2.5).round() == 3);
    CHECK(Primitive(-2.5).round() == -3);
    CHECK(Primitive(2.4).round() == 2);
    CHECK(Primitive(2.7).floor() == 2);
    CHECK(Primitive(-2.3).floor() == -3);
    CHECK(Primitive(2.1).ceil() == 3);
    CHECK(Primitive(-2.7).ceil() == -2);

    CHECK(Primitive(7).round() == 7);
    CHECK(Primitive(7).floor() == 7);
    CHECK(Primitive(true).ceil() == 1);

    CHECK_THROWS(Primitive(1e300).round());
    CHECK_THROWS(Primitive("1").floor());
}

TEST_CASE("Primitive comparison") {
    SECTION("equality") {
        CHECK(Primitive(1) == Primitive(1));
        CHECK(Primitive(1) == Primitive(1.0));
        CHECK(Primitive(1.0) == Primitive(1));
        CHECK(Primitive(1) == Primitive(true));
        CHECK(Primitive(0.0) == Primitive(false));
        CHECK(Primitive(1) != Primitive(1.5));
        CHECK(Primitive(1) != Primitive(2));

        CHECK(Primitive("a") == Primitive("a"));
        CHECK(Primitive("a") != Primitive("b"));
        CHECK(Primitive() == Primitive());
        CHECK(Primitive() != Primitive(0));
        CHECK(Primitive("1") != Primitive(1));
    }

    SECTION("large integers are compared exactly") {
        double two_pow_63 = 9223372036854775808.0;
        CHECK(Primitive(INT_MAX_VALUE) != Primitive(two_pow_63));
        CHECK(Primitive(INT_MAX_VALUE) < Primitive(two_pow_63));
        CHECK(Primitive(INT_MIN_VALUE) == Primitive(-two_pow_63));

        std::intmax_t two_pow_53 = std::intmax_t(1) << 53;
        CHECK(Primitive(two_pow_53 + 1) != Primitive(double(two_pow_53)));
        CHECK(Primitive(double(two_pow_53)) < Primitive(two_pow_53 + 1));
    }

    SECTION("NaN") {
        Primitive nan = Primitive(std::nan(""));
        CHECK(nan != nan);
        CHECK(Primitive(1) != nan);
        CHECK_FALSE(Primitive(1) < nan);
        CHECK_FALSE(nan < Primitive(1));
    }

    SECTION("ordering") {
        CHECK(Primitive(1) < Primitive(2));
        CHECK(Primitive(1) < Primitive(1.5));
        CHECK(Primitive(1.5) < Primitive(2));
        CHECK(Primitive(-1.5) < Primitive(-1));
        CHECK(Primitive(false) < Primitive(true));
        CHECK(Primitive(2) <= Primitive(2.0));
        CHECK(Primitive(3) > Primitive(2.5));
        CHECK(Primitive(3) >= Primitive(3));
        CHECK_FALSE(Primitive(3) < Primitive(3));

        CHECK(Primitive("a") < Primitive("b"));
        CHECK_FALSE(Primitive("b") < Primitive("a"));

        // Different kinds are ordered by kind.
        CHECK(Primitive() < Primitive(0));
        CHECK(Primitive(5) < Primitive("a"));
        CHECK_FALSE(Primitive() < Primitive());
    }

    SECTION("hash") {
        CHECK(Primitive(1).hash() == Primitive(1.0).hash());
        CHECK(Primitive(1).hash() == Primitive(true).hash());
        CHECK(Primitive("abc").hash() == Primitive(std::string("abc")).hash());
        CHECK(Primitive().hash() == Primitive().hash());
        CHECK(std::hash<Primitive> {}(Primitive(3)) == Primitive(3).hash());
    }
}

TEST_CASE("Primitive arithmetic") {
    SECTION("addition") {
        CHECK((Primitive(2) + Primitive(3)) == 5);
        CHECK((Primitive(2) + Primitive(3)).kind() == Kind::Int64);
        CHECK((Primitive(2) + Primitive(0.5)) == 2.5);
        CHECK((Primitive(2) + Primitive(0.5)).kind() == Kind::Float64);
        CHECK((Primitive(true) + Primitive(true)) == 2);
        CHECK((Primitive(true) + Primitive(true)).kind() == Kind::Int64);
        CHECK((Primitive("ab") + Primitive("cd")) == "abcd");

        CHECK_THROWS(Primitive("a") + Primitive(1));
        CHECK_THROWS(Primitive() + Primitive(1));
        CHECK_THROWS_AS(Primitive(INT_MAX_VALUE) + Primitive(1), std::overflow_error);
    }

    SECTION("subtraction") {
        CHECK((Primitive(5) - Primitive(7)) == -2);
        CHECK((Primitive(5) - Primitive(0.5)) == 4.5);
        CHECK_THROWS(Primitive("a") - Primitive("a"));
        CHECK_THROWS_AS(Primitive(INT_MIN_VALUE) - Primitive(1), std::overflow_error);
    }

    SECTION("multiplication") {
        CHECK((Primitive(6) * Primitive(7)) == 42);
        CHECK((Primitive(2) * Primitive(1.5)) == 3.0);
        CHECK((Primitive(2) * Primitive(1.5)).kind() == Kind::Float64);
        CHECK_THROWS(Primitive("a") * Primitive(2));
        CHECK_THROWS_AS(Primitive(INT_MAX_VALUE) * Primitive(2), std::overflow_error);
    }

    SECTION("division") {
        CHECK((Primitive(7) / Primitive(2)) == 3);
        CHECK((Primitive(-7) / Primitive(2)) == -3);
        CHECK((Primitive(7) / Primitive(2.0)) == 3.5);
        CHECK(std::isinf((Primitive(1.0) / Primitive(0)).to_float()));
        CHECK_THROWS(Primitive(7) / Primitive(0));
        CHECK_THROWS_AS(Primitive(INT_MIN_VALUE) / Primitive(-1), std::overflow_error);
        CHECK_THROWS(Primitive() / Primitive(1));
    }

    SECTION("modulo") {
        CHECK((Primitive(7) % Primitive(3)) == 1);
        CHECK((Primitive(-7) % Primitive(3)) == -1);
        CHECK((Primitive(7.5) % Primitive(2)) == 1.5);
        CHECK((Primitive(7.5) % Primitive(2)).kind() == Kind::Float64);
        CHECK((Primitive(INT_MIN_VALUE) % Primitive(-1)) == 0);
        CHECK_THROWS(Primitive(7) % Primitive(0));
        CHECK_THROWS(Primitive("a") % Primitive(1));
    }

    SECTION("negation") {
        CHECK(-Primitive(5) == -5);
        CHECK(-Primitive(1.5) == -1.5);
        CHECK(-Primitive(true) == -1);
        CHECK((-Primitive(true)).kind() == Kind::Int64);
        CHECK_THROWS_AS(-Primitive(INT_MIN_VALUE), std::overflow_error);
        CHECK_THROWS(-Primitive("a"));
        CHECK_THROWS(-Primitive());
    }

    SECTION("logical") {
        CHECK(!Primitive(0));
        CHECK_FALSE(!Primitive("a"));

        // `&&` and `||` return one of their operands, like in Python.
        CHECK((Primitive(0) && Primitive(5)) == 0);
        CHECK((Primitive(1) && Primitive(5)) == 5);
        CHECK((Primitive(0) || Primitive(5)) == 5);
        CHECK((Primitive(2) || Primitive(5)) == 2);
    }
}

TEST_CASE("TemplateArg") {
    CHECK(TemplateArg(TypeInfo::of<float>()).get() == "float");
    CHECK(TemplateArg(true).get() == "true");
    CHECK(TemplateArg(false).get() == "false");
    CHECK(TemplateArg(5).get() == "(int)5ll");
    CHECK(TemplateArg(-5).get() == "(int)-5ll");
    CHECK(TemplateArg(5u).get() == "(unsigned int)5ull");
    CHECK(TemplateArg(char(65)).get() == "(char)65ll");
    CHECK(
        TemplateArg(std::numeric_limits<long>::min()).get()
        == "(long)(-9223372036854775807ll - 1)");
    CHECK(TemplateArg(test_primitive::Color::Green).get() == "(test_primitive::Color)2ll");

    CHECK(TemplateArg(Primitive(7)) == TemplateArg(std::intmax_t(7)));
    CHECK(TemplateArg(Primitive(true)).get() == "true");
    CHECK_THROWS(TemplateArg(Primitive(1.5)));
    CHECK_THROWS(TemplateArg(Primitive("x")));
    CHECK_THROWS(TemplateArg(Primitive()));

    CHECK(TemplateArg::unsafe_from_string("foo").get() == "foo");

    CHECK(TemplateArg(1) == TemplateArg(1));
    CHECK(TemplateArg(1) != TemplateArg(1u));
    CHECK(TemplateArg(1).hash() == TemplateArg(1).hash());
    CHECK(std::hash<TemplateArg> {}(TemplateArg(1)) == TemplateArg(1).hash());
}
