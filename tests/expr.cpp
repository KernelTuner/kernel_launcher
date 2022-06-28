#include "kernel_launcher/expr.h"

#include "catch.hpp"

using namespace kernel_launcher;

TEST_CASE("test Expr") {
    TunableParam x {"x", {1, 2, 3}, 1};
    TunableParam y {"y", {100, 200, 300}, 200};
    TunableParam z {"z", {false, true}, true};
    TunableParam w {"w", {-1, 1}, -1};
    TunableMap c = {{x, x[2]}, {y, y[1]}, {z, z[0]}};
    Eval eval {c};

    auto xe = ParamExpr {x};
    auto ye = ParamExpr {y};
    auto ze = ParamExpr {z};

    SECTION("test Eval") {
        CHECK(eval.lookup(x) == x[2]);
        CHECK_THROWS(eval.lookup(w));
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
        auto e = select(xe < ye, xe, ye);
        CHECK(e.eval(eval) == 200);

        auto e2 = select(xe, "a", "b", "c", "d", "e");
        CHECK(e2.eval(eval) == "d");
    }
}