#include "kernel_launcher/expr.h"

namespace kernel_launcher {

std::string ScalarExpr::to_string() const {
    return value_.to_string();
}

TunableValue ScalarExpr::eval(const Eval& ctx) const {
    return value_;
}

Expr ScalarExpr::resolve(const Eval& eval) const {
    return *this;
}

std::string ParamExpr::to_string() const {
    return "$" + param_.name();
}

TunableValue ParamExpr::eval(const Eval& ctx) const {
    return ctx.lookup(param_);
}

Expr ParamExpr::resolve(const Eval& eval) const {
    if (eval.has(param_)) {
        return ScalarExpr(eval.lookup(param_));
    } else {
        return ParamExpr(param_);
    }
}

std::string ProblemExpr::to_string() const {
    std::stringstream ss;
    ss << "$problem_size_" << axis_;
    return ss.str();
}

TunableValue ProblemExpr::eval(const Eval& eval) const {
    return eval.problem_size(axis_);
}

Expr ProblemExpr::resolve(const Eval& eval) const {
    if (eval.has_problem_size()) {
        return ScalarExpr(eval.problem_size(axis_));
    } else {
        return ProblemExpr(axis_);
    }
}

TunableValue SelectExpr::eval(const Eval& ctx) const {
    auto index = cond_.eval(ctx).to<int64_t>();
    if (index < 0 || size_t(index) >= options_.size()) {
        throw std::invalid_argument("index out of bounds");
    }

    return options_[size_t(index)].eval(ctx);
}

std::string SelectExpr::to_string() const {
    std::stringstream ss;
    ss << "select(" << cond_.to_string();
    for (const auto& v : options_) {
        ss << ", " << v.to_string();
    }
    ss << ")";

    return ss.str();
}

Expr SelectExpr::resolve(const Eval& eval) const {
    Expr cond = cond_.resolve(eval);

    std::vector<Expr> options;
    for (const auto& v : options_) {
        options.push_back(v.resolve(eval));
    }

    return SelectExpr(cond_.resolve(eval), options);
}

TunableValue UnaryExpr::eval(const Eval& ctx) const {
    TunableValue operand = operand_.eval(ctx);

    switch (operator_) {
        case Op::Plus:
            return +operand;
        case Op::Minus:
            return -operand;
        case Op::LogicNot:
            return !operand;
        default:
            throw std::runtime_error("invalid operator");
    }
}

std::string UnaryExpr::to_string() const {
    std::stringstream ss;
    ss << "(" << op_name() << " " << operand_.to_string() << ")";
    return ss.str();
}

Expr UnaryExpr::resolve(const Eval& eval) const {
    UnaryExpr result(operator_, operand_.resolve(eval));

    if (result.operand_.is_constant()) {
        return ScalarExpr(result.eval({}));
    } else {
        return result;
    }
}

std::string UnaryExpr::op_name() const {
    switch (operator_) {
        case Op::Plus:
            return "-";
        case Op::Minus:
            return "-";
        case Op::LogicNot:
            return "!";
        default:
            return "???";
    }
}

TunableValue BinaryExpr::eval(const Eval& ctx) const {
    TunableValue lhs = lhs_.eval(ctx);
    TunableValue rhs = rhs_.eval(ctx);

    switch (operator_) {
        case Op::Add:
            return lhs + rhs;
        case Op::Sub:
            return lhs - rhs;
        case Op::Mul:
            return lhs * rhs;
        case Op::Div:
            return lhs / rhs;
        case Op::Mod:
            return lhs % rhs;
        case Op::Eq:
            return lhs == rhs;
        case Op::Neq:
            return lhs != rhs;
        case Op::Lt:
            return lhs < rhs;
        case Op::Gt:
            return lhs > rhs;
        case Op::Lte:
            return lhs <= rhs;
        case Op::Gte:
            return lhs >= rhs;
        case Op::LogicAnd:
            return lhs.to_bool() && rhs.to_bool();
        case Op::LogicOr:
            return lhs.to_bool() || rhs.to_bool();
        default:
            throw std::runtime_error("invalid operator");
    }
}

std::string BinaryExpr::op_name() const {
    switch (operator_) {
        case Op::Add:
            return "+";
        case Op::Sub:
            return "-";
        case Op::Mul:
            return "*";
        case Op::Div:
            return "/";
        case Op::Mod:
            return "%";
        case Op::Eq:
            return "==";
        case Op::Neq:
            return "!=";
        case Op::Lt:
            return "<";
        case Op::Gt:
            return ">";
        case Op::Lte:
            return "<=";
        case Op::Gte:
            return ">=";
        case Op::LogicAnd:
            return "&&";
        case Op::LogicOr:
            return "||";
        default:
            return "???";
    }
}

std::string BinaryExpr::to_string() const {
    std::stringstream ss;
    ss << "(" + lhs_.to_string() << " " << op_name() << " " << rhs_.to_string()
       << ")";
    return ss.str();
}

Expr BinaryExpr::resolve(const Eval& eval) const {
    BinaryExpr result(operator_, lhs_.resolve(eval), rhs_.resolve(eval));

    if (result.lhs_.is_constant() && result.lhs_.is_constant()) {
        return ScalarExpr(result.eval({}));
    } else {
        return result;
    }
}
}  // namespace kernel_launcher