#include "kernel_launcher/expr.h"

#include "catch.hpp"

using namespace kernel_launcher;

using TunableMap = std::unordered_map<TunableParam, Value>;

struct MyEval: Eval {
    MyEval(TunableMap m) : map(m) {}

    bool lookup(const Variable& v, Value& value) const override {
        if (auto that = dynamic_cast<const TunableParam*>(&v)) {
            auto it = map.find(*that);

            if (it != map.end()) {
                value = it->second;
                return true;
            }
        }

        return false;
    }

    TunableMap map;
};

TEST_CASE("test Expr") {
    TunableParam x {"x", {1, 2, 3}, 1};
    TunableParam y {"y", {100, 200, 300}, 200};
    TunableParam z {"z", {false, true}, true};
    TunableParam w {"w", {-1, 1}, -1};
    TunableMap c = {{x, x[2]}, {y, y[1]}, {z, z[0]}};
    MyEval eval {c};

    auto xe = ParamExpr {x};
    auto ye = ParamExpr {y};
    auto ze = ParamExpr {z};

    SECTION("test Eval") {
        CHECK(ParamExpr {x}.eval(eval) == x[2]);
        CHECK_THROWS(ParamExpr {w}.eval(eval));
    }

    SECTION("test ScalarExpr") {
        auto v = scalar(5);
        CHECK(v.eval(eval) == 5);
        CHECK(v.to_string() == "5");
    }

    SECTION("test ParamExpr") {
        auto v = ParamExpr {x};
        CHECK(v.eval(eval) == 3);
        CHECK(v.to_string() == "$x");
        CHECK(v.parameter() == x);

        v = ParamExpr {w};
        CHECK_THROWS(v.eval(eval));
        CHECK(v.to_string() == "$w");
        CHECK(v.parameter() == w);
    }

    SECTION("test UnaryExpr") {
        CHECK((+xe).eval(eval) == 3);
        CHECK((-xe).eval(eval) == -3);
        CHECK((!xe).eval(eval) == false);

        CHECK((+ye).eval(eval) == 200);
        CHECK((-ye).eval(eval) == -200);
        CHECK((!ye).eval(eval) == false);

        CHECK((+ze).eval(eval) == false);
        CHECK_THROWS((-ze).eval(eval));
        CHECK((!ze).eval(eval) == true);
    }

    SECTION("test BinaryExpr") {
        CHECK((xe + ye).eval(eval) == 203);
        CHECK((xe - ye).eval(eval) == -197);
        CHECK((xe * ye).eval(eval) == 600);
        CHECK((ye / xe).eval(eval) == 66);
        CHECK((xe == ye).eval(eval) == false);
        CHECK((xe != ye).eval(eval) == true);
        CHECK((xe < ye).eval(eval) == true);
        CHECK((xe > ye).eval(eval) == false);
        CHECK((xe <= ye).eval(eval) == true);
        CHECK((xe >= ye).eval(eval) == false);
        CHECK((xe || ye).eval(eval) == true);
        CHECK((xe && ye).eval(eval) == true);

        CHECK((xe + ze).eval(eval) == 3);
        CHECK((xe - ze).eval(eval) == 3);
        CHECK((xe * ze).eval(eval) == 0);
        CHECK_THROWS((ye / ze).eval(eval));
        CHECK((xe == ze).eval(eval) == false);
        CHECK((xe != ze).eval(eval) == true);
        CHECK((xe < ze).eval(eval) == false);
        CHECK((xe > ze).eval(eval) == true);
        CHECK((xe <= ze).eval(eval) == false);
        CHECK((xe >= ze).eval(eval) == true);
        CHECK((xe || ze).eval(eval) == true);
        CHECK((xe && ze).eval(eval) == false);
    }

    SECTION("test SelectExpr") {
        auto e = ifelse(xe < ye, xe, ye);
        CHECK(e.eval(eval) == 3);

        auto e1 = ifelse(xe >= ye, xe, ye);
        CHECK(e1.eval(eval) == 200);

        auto e2 = select(xe, "a", "b", "c", "d", "e");
        CHECK(e2.eval(eval) == "d");

        auto e3 = select(100, "a", "b", "c", "d", "e");
        CHECK_THROWS(e3.eval(eval));

        std::vector<std::string> options = {"a", "b", "c", "d", "e"};
        auto e4 = index(xe, options);
        CHECK(e4.eval(eval) == "d");

        auto e5 = index(100, options);
        CHECK_THROWS(e5.eval(eval));
    }

    SECTION("test cast") {
        cast<int, int>(4);
        cast<int, const int&>(4);
        cast<int, double>(4);

        cast<Value, int>(4);
        cast<Value, double>(4);
        cast<Value, bool>(4);
        cast<Value, std::string>("abc");

        cast<int>(xe);
        cast<double>(xe);
        cast<bool>(xe);
        cast<std::string>(xe);
        cast<Value>(xe);

        cast<int>(xe.parameter());
        cast<double>(xe.parameter());
        cast<bool>(xe.parameter());
        cast<std::string>(xe.parameter());
        cast<Value>(xe.parameter());
    }
}