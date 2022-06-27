#include "kernel_launcher/expr.h"

#include "catch.hpp"

using namespace kernel_launcher;

TEST_CASE("test Expr") {
    auto e = into_expr(4);
    auto e2 = into_expr("abc");

    CHECK(e.eval({}) == 0);
    CHECK(e2.eval({}) == 1);
}