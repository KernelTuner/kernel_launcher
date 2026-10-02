#include "catch.hpp"
#include "kernel_launcher/type_info.hpp"
#include "kernel_launcher/utils.hpp"

using namespace kernel_launcher;

namespace test_utils {
struct MyStruct {};
}  // namespace test_utils

TEST_CASE("intern_string") {
    const std::string& a = intern_string("hello");
    const std::string& b = intern_string(std::string("hel") + "lo");
    const std::string& c = intern_string("world");

    CHECK(a == "hello");
    CHECK(&a == &b);  // same address
    CHECK(&a != &c);
}

TEST_CASE("Symbol") {
    Symbol a("foo");
    Symbol b(std::string("foo"));
    Symbol c(std::string_view("bar"));
    Symbol empty;

    CHECK(a == b);
    CHECK(a != c);
    CHECK(empty.empty());
    CHECK(empty == Symbol(""));
    CHECK(empty.get() == "");
    CHECK(std::string(empty.c_str()) == "");
    CHECK(empty.size() == 0);
    CHECK(empty < a);
    CHECK_FALSE(a < empty);
    CHECK(empty.hash() == Symbol("").hash());
    CHECK_FALSE(a.empty());

    static constexpr Symbol constexpr_empty;
    CHECK(constexpr_empty == empty);

    CHECK(a.get() == "foo");
    CHECK(std::string(a.c_str()) == "foo");
    CHECK(a.size() == 3);
    CHECK(std::string_view(a) == "foo");

    CHECK(c < a);
    CHECK_FALSE(a < c);
    CHECK_FALSE(a < b);

    CHECK(a.hash() == b.hash());
    CHECK(std::hash<Symbol> {}(a) == a.hash());

    std::ostringstream stream;
    stream << a;
    CHECK(stream.str() == "foo");
}

TEST_CASE("demangle_type_info") {
    CHECK(demangle_type_info(typeid(int)) == "int");
    CHECK(demangle_type_info(typeid(const float*)) == "float const*");
    CHECK(demangle_type_info(typeid(test_utils::MyStruct)) == "test_utils::MyStruct");
}

TEST_CASE("TypeInfo") {
    SECTION("int") {
        TypeInfo t = TypeInfo::of<int>();

        CHECK(t.name() == "int");
        CHECK(t.size_in_bytes() == sizeof(int));
        CHECK(t.alignment_in_bytes() == alignof(int));
        CHECK_FALSE(t.is_const());
        CHECK_FALSE(t.is_pointer());
        CHECK(t == TypeInfo::of(42));
        CHECK(t != TypeInfo::of<long>());
        CHECK(t.hash() == TypeInfo::of<int>().hash());
        CHECK(std::hash<TypeInfo> {}(t) == t.hash());
    }

    SECTION("pointer") {
        TypeInfo t = TypeInfo::of<const double*>();

        CHECK(t.name() == "double const*");
        CHECK(t.size_in_bytes() == sizeof(void*));
        CHECK(t.is_pointer());
        CHECK_FALSE(t.is_const());

        TypeInfo pointee = t.remove_pointer();
        CHECK(pointee.is_const());
        CHECK_FALSE(pointee.is_pointer());
        CHECK(pointee.size_in_bytes() == sizeof(double));
        CHECK(pointee.remove_const() == TypeInfo::of<double>());
        CHECK_FALSE(pointee.remove_const().is_const());
    }

    SECTION("void") {
        CHECK(TypeInfo::of<void>().is_void());
        CHECK(TypeInfo::of<const void>().is_void());
        CHECK(TypeInfo::of<void*>().remove_pointer().is_void());
        CHECK(TypeInfo::of<const void*>().remove_pointer().is_void());

        CHECK_FALSE(TypeInfo::of<void*>().is_void());
        CHECK_FALSE(TypeInfo::of<int>().is_void());
        CHECK_FALSE(TypeInfo::opaque("ns::DeviceStruct", 24, 8).is_void());
    }

    SECTION("struct") {
        TypeInfo t = TypeInfo::of<test_utils::MyStruct>();
        CHECK(t.name() == "test_utils::MyStruct");
        CHECK(t.size_in_bytes() == sizeof(test_utils::MyStruct));
    }

    SECTION("opaque") {
        TypeInfo t = TypeInfo::opaque("ns::DeviceStruct", 24, 8);
        CHECK(t.name() == "ns::DeviceStruct");
        CHECK(t.size_in_bytes() == 24);
        CHECK(t.alignment_in_bytes() == 8);
        CHECK_FALSE(t.is_pointer());
        CHECK_FALSE(t.is_const());
        CHECK(t.remove_const() == t);

        CHECK(t == TypeInfo::opaque("ns::DeviceStruct", 24, 8));
        CHECK(t.hash() == TypeInfo::opaque("ns::DeviceStruct", 24, 8).hash());
        CHECK(t != TypeInfo::opaque("ns::DeviceStruct", 32, 8));
        CHECK(t != TypeInfo::opaque("ns::DeviceStruct", 24, 4));
        CHECK(t != TypeInfo::opaque("ns::OtherStruct", 24, 8));
        CHECK(TypeInfo::opaque("int", sizeof(int), alignof(int)) != TypeInfo::of<int>());
        CHECK(TypeInfo::of<int>() != TypeInfo::opaque("int", sizeof(int), alignof(int)));

        CHECK_THROWS(TypeInfo::opaque("", 8, 8));
        CHECK_THROWS(TypeInfo::opaque("Bad", 8, 3));
        CHECK_THROWS(TypeInfo::opaque("Bad", 12, 8));
    }

    SECTION("void") {
        TypeInfo t = TypeInfo::of<void>();
        CHECK(t.name() == "void");
        CHECK(t.size_in_bytes() == 0);
        CHECK(TypeInfo::of<void*>().remove_pointer() == t);
    }
}

TEST_CASE("hash_combine") {
    size_t seed = 0;
    hash_combine(seed, 1);
    hash_combine(seed, 2);
    hash_combine(seed, 3);
    CHECK(seed == hash_fields(1, 2, 3));

    CHECK(hash_fields(1, 2) == hash_fields(1, 2));
    CHECK(hash_fields(1, 2) != hash_fields(2, 1));
    CHECK(hash_fields(std::vector<int> {1}) != hash_fields(std::vector<int> {1, 0}));
    CHECK(hash_fields(std::string("a"), 1) == hash_fields(std::string("a"), 1));
}

TEST_CASE("concat") {
    CHECK(concat() == "");
    CHECK(concat("a", 1, '-', 2.5) == "a1-2.5");
    CHECK(concat(Symbol("x"), "=", std::string("y")) == "x=y");
}

TEST_CASE("logging") {
    // Warnings are always enabled, and the levels are ordered.
    CHECK(log_warning_enabled());

    if (log_debug_enabled()) {
        CHECK(log_info_enabled());
    }

    CHECK_NOTHROW(log_debug() << "debug message from test" << std::endl);
}
