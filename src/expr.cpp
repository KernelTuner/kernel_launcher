#include "kernel_launcher/expr.h"

#include "kernel_launcher/cuda.h"

namespace kernel_launcher {

std::string ScalarExpr::to_string() const {
    return value_.to_string();
}

Value ScalarExpr::eval(const Eval& ctx) const {
    return value_;
}

Expr ScalarExpr::resolve(const Eval& eval) const {
    return *this;
}

std::string ParamExpr::to_string() const {
    return "$" + param_.name();
}

Value ParamExpr::eval(const Eval& ctx) const {
    Value value;
    if (!ctx.lookup(param_, value)) {
        throw std::runtime_error(
            "parameter $" + param_.name() + "is undefined");
    }

    return value;
}

Expr ParamExpr::resolve(const Eval& eval) const {
    Value value;

    if (eval.lookup(param_, value)) {
        return ScalarExpr(value);
    }

    return ParamExpr(param_);
}

std::string ProblemExpr::to_string() const {
    std::stringstream ss;
    ss << "$problem_size_" << axis_;
    return ss.str();
}

Value ProblemExpr::eval(const Eval& eval) const {
    Value value;

    if (!eval.lookup(*this, value)) {
        throw std::runtime_error(
            "attempted to evaluate expression that "
            "cannot be problem dependent");
    }

    return value;
}

Expr ProblemExpr::resolve(const Eval& eval) const {
    Value value;

    if (eval.lookup(*this, value)) {
        return ScalarExpr(value);
    }

    return ProblemExpr(axis_);
}

ArgExpr::ArgExpr(uint8_t i, const char* name) noexcept : index_(i) {
    if (name != nullptr && name[0] != '\0') {
        name_ = &intern_string(name);
    }
}

std::string ArgExpr::to_string() const {
    if (name_ != nullptr) {
        return "$" + *name_;
    } else {
        return "$argument_" + std::to_string(index_);
    }
}

Value ArgExpr::eval(const Eval& eval) const {
    Value out;

    if (!eval.lookup(*this, out)) {
        std::string msg;

        if (name_ != nullptr) {
            msg = "cannot find argument " + *name_ + " (argument at index "
                + std::to_string(index_) + ")";
        } else {
            msg = "cannot find argument at index" + std::to_string(index_);
        }

        throw std::runtime_error(msg);
    }

    return out;
}

Expr ArgExpr::resolve(const Eval& eval) const {
    Value out;

    if (eval.lookup(*this, out)) {
        return out;
    }

    return *this;
}

std::string DeviceAttributeExpr::to_string() const {
    const char* name = "";

#define IMPL_CASE(key)              \
    case CU_DEVICE_ATTRIBUTE_##key: \
        name = #key;                \
        break;

    switch (attribute_) {
        KERNEL_LAUNCHER_DEVICE_ATTRIBUTES_FORALL(IMPL_CASE)
        default:
            throw std::runtime_error(
                "unknown device attribute: " + std::to_string(attribute_));
    }
#undef IMPL_CASE

    return name;
}

Value DeviceAttributeExpr::eval(const Eval& eval) const {
    Value out;
    if (!eval.lookup(*this, out)) {
        throw std::runtime_error(
            "error while reading device attribute: " + to_string());
    }

    return out;
}

Expr DeviceAttributeExpr::resolve(const Eval& eval) const {
    return *this;
}

Value SelectExpr::eval(const Eval& ctx) const {
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

Value UnaryExpr::eval(const Eval& ctx) const {
    Value operand = operand_.eval(ctx);

    switch (operator_) {
        case Op::Plus:
            return +operand;
        case Op::Minus:
            return -operand;
        case Op::LogicNot:
            return !operand;
        case Op::RoundNearest:
            return operand.round();
        case Op::RoundDown:
            return operand.floor();
        case Op::RoundUp:
            return operand.ceil();
        default:
            throw std::runtime_error("invalid operator");
    }
}

std::string UnaryExpr::to_string() const {
    std::stringstream ss;
    ss << op_name() << "(" << operand_.to_string() << ")";
    return ss.str();
}

Expr UnaryExpr::resolve(const Eval& eval) const {
    UnaryExpr result(operator_, operand_.resolve(eval));

    if (result.operand_.is_constant()) {
        return ScalarExpr(result.eval(eval));
    }

    return result;
}

std::string UnaryExpr::op_name() const {
    switch (operator_) {
        case Op::Plus:
            return "+";
        case Op::Minus:
            return "-";
        case Op::LogicNot:
            return "!";
        case Op::RoundNearest:
            return "round";
        case Op::RoundUp:
            return "ceil";
        case Op::RoundDown:
            return "floor";
        default:
            return "???";
    }
}

Value BinaryExpr::eval(const Eval& ctx) const {
    Value lhs = lhs_.eval(ctx);
    Value rhs = rhs_.eval(ctx);

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
        return ScalarExpr(result.eval(eval));
    }

    return result;
}
}  // namespace kernel_launcher
