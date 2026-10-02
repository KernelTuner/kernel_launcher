#include "catch.hpp"
#include "kernel_launcher/config.hpp"
#include "kernel_launcher/expr.hpp"

using namespace kernel_launcher;

static Config config_with_x(Primitive value) {
    Config config;
    config.insert("x", value);
    return config;
}

// Resolves variables by their `to_string()` representation.
struct MapEval: EvalContext {
    MapEval(std::initializer_list<std::pair<const std::string, Primitive>> values) :
        values(values) {}

    bool try_lookup(const Variable& v, Primitive& out) const override {
        auto it = values.find(v.to_string());

        if (it == values.end()) {
            return false;
        }

        out = it->second;
        return true;
    }

    std::unordered_map<std::string, Primitive> values;
};

static const MapEval empty_eval = {};

TEST_CASE("ScalarExpr") {
    ScalarExpr e(42);
    CHECK(e.eval(empty_eval) == 42);
    CHECK(e.value() == 42);
    CHECK(e.to_string() == "42");
    CHECK(e.is_constant());
    CHECK(e.resolve(empty_eval).is_constant());
    CHECK(e.resolve(empty_eval).eval(empty_eval) == 42);
}

TEST_CASE("into_expr and is_expr") {
    CHECK(is_expr<ScalarExpr>::value);
    CHECK(is_expr<const ArgumentExpr&>::value);
    CHECK(is_expr<ExprPtr>::value);
    CHECK_FALSE(is_expr<int>::value);
    CHECK_FALSE(is_expr<Primitive>::value);

    SECTION("from value") {
        ExprPtr e = into_expr(7);
        CHECK(e.is_constant());
        CHECK(e.eval(empty_eval) == 7);
    }

    SECTION("from expression") {
        ExprPtr e = into_expr(arg(1));
        CHECK_FALSE(e.is_constant());
        CHECK(dynamic_cast<const ArgumentExpr*>(&e.inner()) != nullptr);
    }

    SECTION("from ExprPtr") {
        ExprPtr a = into_expr(3);
        ExprPtr b = into_expr(a);
        CHECK(&a.inner() == &b.inner());
    }
}

TEST_CASE("ExprPtr") {
    ExprPtr e = into_expr(arg(0, "n"));
    CHECK(e.to_string() == "$arg_0:n");
    CHECK_FALSE(e.is_constant());
    CHECK(e.eval(MapEval {{"$arg_0:n", 5}}) == 5);
    CHECK(e.resolve(MapEval {{"$arg_0:n", 5}}).is_constant());
    CHECK(dynamic_cast<const ArgumentExpr*>(&e.inner()) != nullptr);

    CHECK_THROWS(ExprPtr(nullptr));
}

TEST_CASE("Variable lookup") {
    CHECK_THROWS_WITH(arg(0, "n").eval(empty_eval), Catch::Contains("unknown variable '$arg_0:n'"));
    CHECK_THROWS(empty_eval.lookup(arg(0, "n")));
    CHECK(MapEval {{"$arg_0:n", 1}}.lookup(arg(0, "n")) == 1);
}

TEST_CASE("ArgumentExpr") {
    SECTION("arg") {
        ArgumentExpr a = arg(2);
        CHECK(a.index() == 2);
        CHECK(a.to_string() == "$arg_2");
        CHECK(arg(2, "n").to_string() == "$arg_2:n");
    }

    SECTION("args<N>") {
        auto [a, b, c] = args<3>();
        CHECK(a.index() == 0);
        CHECK(b.index() == 1);
        CHECK(c.index() == 2);
        CHECK(c.to_string() == "$arg_2");
    }

    SECTION("args(names)") {
        auto [n, C] = args("n", "C");
        CHECK(n.index() == 0);
        CHECK(C.index() == 1);
        CHECK(n.to_string() == "$arg_0:n");
        CHECK(C.to_string() == "$arg_1:C");
    }

    SECTION("resolve") {
        ArgumentExpr a = arg(0, "n");
        CHECK(a.resolve(MapEval {{"$arg_0:n", 10}}).is_constant());
        CHECK(a.resolve(MapEval {{"$arg_0:n", 10}}).eval(empty_eval) == 10);
        CHECK_FALSE(a.resolve(empty_eval).is_constant());
        CHECK(a.resolve(empty_eval).to_string() == "$arg_0:n");
    }

    SECTION("buffer size") {
        auto [n, C] = args("n", "C");
        ArgumentBufferSize buffer = C[n * 2];
        CHECK(buffer.index == 1);
        CHECK(buffer.length.eval(MapEval {{"$arg_0:n", 10}}) == 20);
    }

    SECTION("map") {
        ArgumentExpr a = arg(0, "n");
        CHECK(a.get(KernelArg::of(5)) == 5);

        ArgumentExpr b = a.map([](const int& x) { return x * 2; });
        CHECK(b.index() == 0);
        CHECK(a.to_string() == "$arg_0:n");
        CHECK(b.to_string() == "map($arg_0:n)");
        CHECK(b.get(KernelArg::of(5)) == 10);
        CHECK(a.get(KernelArg::of(5)) == 5);
        CHECK_THROWS(b.get(KernelArg::of(5.0f)));
    }
}

TEST_CASE("ProblemSizeExpr") {
    CHECK(problem_size_x().axis() == 0);
    CHECK(problem_size_y().axis() == 1);
    CHECK(problem_size_z().axis() == 2);

    CHECK(problem_size_x().to_string() == "$problem_size_x");
    CHECK(problem_size_y().to_string() == "$problem_size_y");
    CHECK(problem_size_z().to_string() == "$problem_size_z");
    CHECK(ProblemSizeExpr(3).to_string() == "$problem_size_?");

    MapEval eval = {{"$problem_size_y", 100}};
    CHECK(problem_size_y().eval(eval) == 100);
    CHECK(problem_size_y().resolve(eval).is_constant());
    CHECK_FALSE(problem_size_x().resolve(eval).is_constant());
}

TEST_CASE("FeatureExpr") {
    FeatureExpr f("dtype");
    CHECK(f.name() == "dtype");
    CHECK(f.to_string() == "$feature_dtype");

    MapEval eval = {{"$feature_dtype", 4}};
    CHECK(f.eval(eval) == 4);
    CHECK(f.resolve(eval).is_constant());
    CHECK_FALSE(f.resolve(empty_eval).is_constant());
}

TEST_CASE("DeviceAttributeExpr") {
    DeviceAttributeExpr d = device_attribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE);
    CHECK(d.attribute() == CU_DEVICE_ATTRIBUTE_WARP_SIZE);
    CHECK(d.to_string() == "$device_attribute_WARP_SIZE");

    DeviceAttributeExpr other = device_attribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
    CHECK(
        other.to_string()
        == "$device_attribute_" + std::to_string(int(CU_DEVICE_ATTRIBUTE_CLOCK_RATE)));

    MapEval eval = {{"$device_attribute_WARP_SIZE", 32}};
    CHECK(d.eval(eval) == 32);
    CHECK(d.resolve(eval).is_constant());
    CHECK_FALSE(d.resolve(empty_eval).is_constant());
}

TEST_CASE("KernelAttributeExpr") {
    KernelAttributeExpr k = kernel_attribute(CU_FUNC_ATTRIBUTE_NUM_REGS);
    CHECK(k.attribute() == CU_FUNC_ATTRIBUTE_NUM_REGS);
    CHECK(k.to_string() == "$kernel_attribute_NUM_REGS");

    KernelAttributeExpr other = kernel_attribute(CU_FUNC_ATTRIBUTE_PTX_VERSION);
    CHECK(
        other.to_string()
        == "$kernel_attribute_" + std::to_string(int(CU_FUNC_ATTRIBUTE_PTX_VERSION)));

    MapEval eval = {{"$kernel_attribute_NUM_REGS", 64}};
    CHECK(k.eval(eval) == 64);
    CHECK(k.resolve(eval).is_constant());
    CHECK_FALSE(k.resolve(empty_eval).is_constant());
}

TEST_CASE("unary operators") {
    ArgumentExpr a = arg(0);
    MapEval int_eval = {{"$arg_0", 3}};
    MapEval float_eval = {{"$arg_0", 2.5}};

    SECTION("eval") {
        CHECK((-a).eval(int_eval) == -3);
        CHECK((-a).eval(float_eval) == -2.5);
        CHECK((!a).eval(int_eval) == false);
        CHECK((!a).eval(MapEval {{"$arg_0", 0}}) == true);
        CHECK(round(a).eval(float_eval) == 3);
        CHECK(floor(a).eval(float_eval) == 2);
        CHECK(ceil(a).eval(float_eval) == 3);
        CHECK(to_float(a).eval(int_eval) == 3.0);
        CHECK(to_float(a).eval(int_eval).kind() == Primitive::Kind::Float64);
        CHECK(to_bool(a).eval(int_eval) == true);
        CHECK(to_bool(a).eval(int_eval).kind() == Primitive::Kind::Bool);
        CHECK(to_int(a).eval(MapEval {{"$arg_0", 3.0}}) == 3);
        CHECK(to_int(a).eval(MapEval {{"$arg_0", 3.0}}).kind() == Primitive::Kind::Int64);
        CHECK_THROWS(to_int(a).eval(float_eval));
        CHECK(to_string(a).eval(int_eval) == "3");
        CHECK(to_string(a).eval(int_eval).kind() == Primitive::Kind::String);
    }

    SECTION("to_string") {
        CHECK((-a).to_string() == "(-$arg_0)");
        CHECK((!a).to_string() == "(!$arg_0)");
        CHECK(round(a).to_string() == "round($arg_0)");
        CHECK(floor(a).to_string() == "floor($arg_0)");
        CHECK(ceil(a).to_string() == "ceil($arg_0)");
        CHECK(to_float(a).to_string() == "float($arg_0)");
        CHECK(to_bool(a).to_string() == "bool($arg_0)");
        CHECK(to_int(a).to_string() == "int($arg_0)");
        CHECK(to_string(a).to_string() == "str($arg_0)");
    }

    SECTION("resolve") {
        CHECK_FALSE((-a).is_constant());
        CHECK_FALSE((-a).resolve(empty_eval).is_constant());

        ExprPtr folded = (-a).resolve(int_eval);
        CHECK(folded.is_constant());
        CHECK(dynamic_cast<const ScalarExpr*>(&folded.inner()) != nullptr);
        CHECK(folded.eval(empty_eval) == -3);
    }
}

TEST_CASE("binary operators") {
    ArgumentExpr a = arg(0);
    ArgumentExpr b = arg(1);
    MapEval eval = {{"$arg_0", 7}, {"$arg_1", 2}};

    SECTION("eval") {
        CHECK((a + b).eval(eval) == 9);
        CHECK((a - b).eval(eval) == 5);
        CHECK((a * b).eval(eval) == 14);
        CHECK((a / b).eval(eval) == 3);
        CHECK((a % b).eval(eval) == 1);
        CHECK((a && b).eval(eval) == true);
        CHECK((a && 0).eval(eval) == false);
        CHECK((a || 0).eval(eval) == true);
        CHECK((0 || into_expr(0)).eval(eval) == false);
        CHECK((a == b).eval(eval) == false);
        CHECK((a != b).eval(eval) == true);
        CHECK((a < b).eval(eval) == false);
        CHECK((a <= 7).eval(eval) == true);
        CHECK((a > b).eval(eval) == true);
        CHECK((a >= 8).eval(eval) == false);
    }

    SECTION("short circuit") {
        // The right operand divides by zero, but it is not evaluated.
        CHECK(((b == 0) && (a / 0 > 1)).eval(eval) == false);
        CHECK(((b != 0) || (a / 0 > 1)).eval(eval) == true);
        CHECK_THROWS(((b != 0) && (a / 0 > 1)).eval(eval));

        // The left operand decides the result, even if the right operand is unknown.
        ExprPtr folded = ((a > 10) && arg(2)).resolve(eval);
        CHECK(dynamic_cast<const ScalarExpr*>(&folded.inner()) != nullptr);
        CHECK(folded.eval(empty_eval) == false);

        CHECK_FALSE(((a < 10) && arg(2)).resolve(eval).is_constant());
    }

    SECTION("scalar operands") {
        CHECK((a + 1).eval(eval) == 8);
        CHECK((1 + a).eval(eval) == 8);
        CHECK((10 - a).eval(eval) == 3);
        CHECK((a - 10).eval(eval) == -3);
    }

    SECTION("to_string") {
        CHECK((a + b).to_string() == "($arg_0 + $arg_1)");
        CHECK((a - b).to_string() == "($arg_0 - $arg_1)");
        CHECK((a * b).to_string() == "($arg_0 * $arg_1)");
        CHECK((a / b).to_string() == "($arg_0 / $arg_1)");
        CHECK((a % b).to_string() == "($arg_0 % $arg_1)");
        CHECK((a && b).to_string() == "($arg_0 && $arg_1)");
        CHECK((a || b).to_string() == "($arg_0 || $arg_1)");
        CHECK((a == b).to_string() == "($arg_0 == $arg_1)");
        CHECK((a != b).to_string() == "($arg_0 != $arg_1)");
        CHECK((a < b).to_string() == "($arg_0 < $arg_1)");
        CHECK((a <= b).to_string() == "($arg_0 <= $arg_1)");
        CHECK((a > b).to_string() == "($arg_0 > $arg_1)");
        CHECK((a >= b).to_string() == "($arg_0 >= $arg_1)");
        CHECK((a + 1).to_string() == "($arg_0 + 1)");
    }

    SECTION("operands") {
        ExprPtr e = a + b;
        const auto& bin = dynamic_cast<const BinaryExpr&>(e.inner());
        CHECK(bin.left_operand().to_string() == "$arg_0");
        CHECK(bin.right_operand().to_string() == "$arg_1");
    }

    SECTION("resolve") {
        ExprPtr e = a * b + 1;
        CHECK_FALSE(e.is_constant());

        ExprPtr folded = e.resolve(eval);
        CHECK(folded.is_constant());
        CHECK(dynamic_cast<const ScalarExpr*>(&folded.inner()) != nullptr);
        CHECK(folded.eval(empty_eval) == 15);

        // Only `a` is known, so the expression cannot be folded.
        ExprPtr partial = e.resolve(MapEval {{"$arg_0", 7}});
        CHECK_FALSE(partial.is_constant());
        CHECK(partial.to_string() == "((7 * $arg_1) + 1)");
        CHECK(partial.eval(MapEval {{"$arg_1", 3}}) == 22);
    }
}

TEST_CASE("div_ceil") {
    ArgumentExpr n = arg(0);
    CHECK(div_ceil(n, 4).eval(MapEval {{"$arg_0", 8}}) == 2);
    CHECK(div_ceil(n, 4).eval(MapEval {{"$arg_0", 9}}) == 3);
    CHECK(div_ceil(n, 4).eval(MapEval {{"$arg_0", 1}}) == 1);
    CHECK(div_ceil(100, n).eval(MapEval {{"$arg_0", 32}}) == 4);

    // Negative and floating-point operands.
    CHECK(div_ceil(n, 4).eval(MapEval {{"$arg_0", -9}}) == -2);
    CHECK(div_ceil(n, -4).eval(MapEval {{"$arg_0", 9}}) == -2);
    CHECK(div_ceil(n, 2).eval(MapEval {{"$arg_0", 4.5}}) == 3);
    CHECK(div_ceil(n, 0.5).eval(MapEval {{"$arg_0", 3}}) == 6);
    CHECK(div_ceil(n, 4).eval(MapEval {{"$arg_0", 8}}).kind() == Primitive::Kind::Int64);
}

TEST_CASE("select") {
    ConfigSpace space;
    TunableParam x = space.tune("x", {0, 1, 2});

    SelectExpr e = select(x, 10, 20, 30);
    CHECK(e.eval(config_with_x(0)) == 10);
    CHECK(e.eval(config_with_x(1)) == 20);
    CHECK(e.eval(config_with_x(2)) == 30);

    CHECK_THROWS(e.eval(config_with_x(3)));
    CHECK_THROWS(e.eval(config_with_x(-1)));

    CHECK(e.to_string() == "select(x, 10, 20, 30)");
    CHECK(e.index().to_string() == "x");
    CHECK(e.options().size() == 3);

    SECTION("is_constant") {
        CHECK_FALSE(e.is_constant());
        CHECK(select(into_expr(1), 10, 20).is_constant());
        CHECK_FALSE(select(1, 10, x).is_constant());
    }

    SECTION("resolve") {
        CHECK(e.resolve(config_with_x(1)).is_constant());
        CHECK(e.resolve(config_with_x(1)).eval(Config()) == 20);
        CHECK_FALSE(e.resolve(Config()).is_constant());

        // An out of bounds index is not folded. The result is still constant (all operands
        // are), but evaluating it throws.
        ExprPtr out_of_bounds = e.resolve(config_with_x(5));
        CHECK(out_of_bounds.is_constant());
        CHECK_THROWS(out_of_bounds.eval(Config()));
    }
}

TEST_CASE("index") {
    ConfigSpace space;
    TunableParam x = space.tune("x", {0, 1, 2});

    std::vector<int> options = {5, 6, 7};
    SelectExpr e = index(x, options);
    CHECK(e.eval(config_with_x(0)) == 5);
    CHECK(e.eval(config_with_x(2)) == 7);
    CHECK(e.options().size() == 3);
}

TEST_CASE("ifelse") {
    ConfigSpace space;
    TunableParam x = space.tune("x", {1, 5});

    SelectExpr e = ifelse(x < 3, 100, 200);
    CHECK(e.eval(config_with_x(1)) == 100);
    CHECK(e.eval(config_with_x(5)) == 200);
    CHECK(e.to_string() == "select((x < 3), 200, 100)");

    // The condition must be a boolean or 0/1.
    CHECK(ifelse(x, 100, 200).eval(config_with_x(1)) == 100);
    CHECK(ifelse(x - 1, 100, 200).eval(config_with_x(1)) == 200);
    CHECK_THROWS(ifelse(x, 100, 200).eval(config_with_x(5)));
}

TEST_CASE("min and max") {
    ConfigSpace space;
    TunableParam x = space.tune("x", {1, 5, 9});

    SECTION("eval") {
        CHECK(min(x, 7).eval(config_with_x(1)) == 1);
        CHECK(min(x, 7).eval(config_with_x(9)) == 7);
        CHECK(max(x, 7).eval(config_with_x(1)) == 7);
        CHECK(max(x, 7).eval(config_with_x(9)) == 9);

        // Constant as the first operand.
        CHECK(min(7, x).eval(config_with_x(9)) == 7);
        CHECK(max(7, x).eval(config_with_x(9)) == 9);

        // Equal operands.
        CHECK(min(x, 5).eval(config_with_x(5)) == 5);
        CHECK(max(x, 5).eval(config_with_x(5)) == 5);
    }

    SECTION("nested") {
        auto clamp = min(max(x, 2), 8);
        CHECK(clamp.eval(config_with_x(1)) == 2);
        CHECK(clamp.eval(config_with_x(5)) == 5);
        CHECK(clamp.eval(config_with_x(9)) == 8);
    }

    SECTION("mixed types") {
        CHECK(max(x, 2.5).eval(config_with_x(1)) == 2.5);
        CHECK(max(x, 2.5).eval(config_with_x(5)) == 5);
    }

    SECTION("resolve") {
        CHECK(min(x, 7).resolve(config_with_x(9)).is_constant());
        CHECK(min(x, 7).resolve(config_with_x(9)).eval(Config()) == 7);
        CHECK_FALSE(min(x, 7).resolve(Config()).is_constant());
    }
}

TEST_CASE("Expr<T>") {
    SECTION("default") {
        Expr<int> e;
        CHECK(e.is_constant());
        CHECK(e.eval(empty_eval) == 0);
    }

    SECTION("from value") {
        Expr<uint32_t> e = 5;
        CHECK(e.is_constant());
        CHECK(e.eval(empty_eval) == 5u);
        CHECK(e.to_string() == "5");
    }

    SECTION("from expression") {
        ArgumentExpr n = arg(0, "n");
        Expr<int64_t> e = n * 2;
        CHECK_FALSE(e.is_constant());
        CHECK(e.to_string() == "($arg_0:n * 2)");
        CHECK(e.eval(MapEval {{"$arg_0:n", 21}}) == 42);
        CHECK(e.inner().to_string() == "($arg_0:n * 2)");

        Expr<int64_t> resolved = e.resolve(MapEval {{"$arg_0:n", 21}});
        CHECK(resolved.is_constant());
        CHECK(resolved.eval(empty_eval) == 42);
    }

    SECTION("conversion") {
        Expr<double> e = arg(0, "n");
        CHECK(e.eval(MapEval {{"$arg_0:n", 3}}) == 3.0);

        Expr<uint32_t> negative = arg(0, "n");
        CHECK_THROWS(negative.eval(MapEval {{"$arg_0:n", -1}}));
    }
}
