#include "kernel_launcher/utils.h"

#include "catch.hpp"

using namespace kernel_launcher;

TEST_CASE("test TypeInfo") {
    // typeid ignore constness, but TypeInfo should not ignore it!
    CHECK(typeid(int) == typeid(const int));
    CHECK(TypeInfo::of<int>() != TypeInfo::of<const int>());

    TypeInfo x = TypeInfo::of<int>();
    CHECK(x.size() == sizeof(int));
    CHECK(x.alignment() == alignof(int));
    CHECK(x.name() == "int");
    CHECK(x.is_pointer() == false);
    CHECK(x.is_const() == false);
    CHECK(x.add_const() == TypeInfo::of<const int>());
    CHECK(x.remove_const() == x);

    TypeInfo y = TypeInfo::of<int*>();
    CHECK(y.size() == sizeof(int*));
    CHECK(y.alignment() == alignof(int*));
    CHECK(y.name() == "int*");
    CHECK(y.is_pointer() == true);
    CHECK(y.is_const() == false);
    CHECK(y.remove_pointer() == x);
    CHECK(y.add_const() == TypeInfo::of<int* const>());
    CHECK(y.remove_const() == y);

    TypeInfo z = TypeInfo::of<const int>();
    CHECK(z.size() == sizeof(int));
    CHECK(z.alignment() == alignof(int));
    CHECK(z.name() == "int");
    CHECK(z.is_pointer() == false);
    CHECK(z.is_const() == true);
    CHECK(z.add_const() == z);
    CHECK(z.remove_const() == TypeInfo::of<int>());

    TypeInfo w = TypeInfo::of<const int*>();
    CHECK(w.size() == sizeof(int*));
    CHECK(w.alignment() == alignof(int*));
    CHECK(w.name() == "int const*");
    CHECK(w.is_pointer() == true);
    CHECK(w.is_const() == false);
    CHECK(w.remove_pointer() == z);
    CHECK(w.add_const() == TypeInfo::of<const int* const>());
    CHECK(w.remove_const() == w);

    TypeInfo v = TypeInfo::of<int* const>();
    CHECK(v.size() == sizeof(int*));
    CHECK(v.alignment() == alignof(int*));
    CHECK(v.name() == "int*");
    CHECK(v.is_pointer() == true);
    CHECK(v.is_const() == true);
    CHECK(v.remove_pointer() == x);
    CHECK(v.add_const() == v);
    CHECK(v.remove_const() == TypeInfo::of<int*>());

    CHECK(type_of<int>() == x);
    CHECK(type_of((int)5) == x);
    CHECK(type_name<int>() == x.name());
    CHECK(type_name((int)5) == x.name());
}

TEST_CASE("test cmp_less") {
    CHECK(in_range(5, 0, std::numeric_limits<uint8_t>::max()));
    CHECK(in_range(300, 0, std::numeric_limits<uint8_t>::max()) == false);
    CHECK(in_range<int>(300));
    CHECK(in_range<unsigned int>(-300) == false);

    CHECK(in_range<int>(std::numeric_limits<unsigned int>::max()) == false);
    CHECK(in_range<int>(std::numeric_limits<unsigned int>::min()));
    CHECK(in_range<unsigned int>(std::numeric_limits<int>::max()));
    CHECK(in_range<unsigned int>(std::numeric_limits<int>::min()) == false);
}

TEST_CASE("test hash") {
    CHECK(hash_string("") == 0xcbf29ce484222325);
    CHECK(hash_string("foo") == 0xdcb27518fed9d577);

    // hash combine should not be commutative not
    CHECK(hash_combine(0, 1) != hash_combine(1, 0));

    // hash combine should not use xor or or
    CHECK(hash_combine(1, 1) != 0);
    CHECK(hash_combine(1, 1) != 1);
    CHECK(hash_combine(1, 0) != 1);
    CHECK(hash_combine(0, 1) != 1);
}

TEST_CASE("test string_match") {
    CHECK(string_match("foo", "foo"));
    CHECK(string_match("", ""));
    CHECK(string_match("*", ""));
    CHECK(string_match("**", ""));
    CHECK(string_match("*", "foo"));
    CHECK(string_match("*foo", "foo"));
    CHECK(string_match("foo*", "foo"));
    CHECK(string_match("fo*o", "foo"));
    CHECK(string_match("*fo*o", "foo"));
    CHECK(string_match("*fo*o*", "foo"));
    CHECK(string_match("*foo*", "foo"));
    CHECK(string_match("*fo*", "foo"));
    CHECK(string_match("**", "foo"));

    CHECK(string_match("bar", "foo") == false);
    CHECK(string_match("foe", "foo") == false);
    CHECK(string_match("foobar", "foo") == false);
    CHECK(string_match("barfoo", "foo") == false);
    CHECK(string_match("bar*", "foo") == false);
    CHECK(string_match("*bar", "foo") == false);
    CHECK(string_match("*r*", "foo") == false);
    CHECK(string_match("*r*", "") == false);
    CHECK(string_match("", "foo") == false);
}

TEST_CASE("test string_split") {
    using v = std::vector<std::string>;
    CHECK(string_split("a,b,c", ',') == v {"a", "b", "c"});
    CHECK(string_split(",b,c", ',') == v {"", "b", "c"});
    CHECK(string_split("a,,c", ',') == v {"a", "", "c"});
    CHECK(string_split("a,b,", ',') == v {"a", "b", ""});
    CHECK(string_split("a,,", ',') == v {"a", "", ""});
    CHECK(string_split(",b,", ',') == v {"", "b", ""});
    CHECK(string_split(",,c", ',') == v {"", "", "c"});
    CHECK(string_split(",,", ',') == v {"", "", ""});
    CHECK(string_split(",", ',') == v {"", ""});
    CHECK(string_split("", ',') == v {""});
    CHECK(string_split("aaaaa|bbbbb", '|') == v {"aaaaa", "bbbbb"});

    CHECK(string_split("a|b,c:d", {'|', ',', ':'}) == v {"a", "b", "c", "d"});
}

TEST_CASE("test ProblemSize") {
    ProblemSize p {1, 2, 3};

    CHECK(p.x == 1);
    CHECK(p.y == 2);
    CHECK(p.z == 3);

    CHECK(p[0] == 1);
    CHECK(p[1] == 2);
    CHECK(p[2] == 3);

    auto [x, y, z] = p;
    CHECK(x == 1);
    CHECK(y == 2);
    CHECK(z == 3);

    CHECK(p.size() == 3);
    CHECK(p.data() == &p.x);

    CHECK(p == p);
    CHECK_FALSE(p != p);

    ProblemSize q {2, 5, 1};
    CHECK(p != q);
    CHECK_FALSE(p == q);
}