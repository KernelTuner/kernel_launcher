#ifndef KERNEL_LAUNCHER_EXPR_H
#define KERNEL_LAUNCHER_EXPR_H

#include <sstream>
#include <unordered_map>

#include "kernel_launcher/value.h"

namespace kernel_launcher {

struct Eval;

template<typename T>
struct Expr;
using AnyExpr = Expr<TunableValue>;
struct ScalarExpr;

struct BaseExpr {
    virtual ~BaseExpr() {};
    virtual std::string to_string() const = 0;
    virtual TunableValue eval(const Eval& eval) const = 0;
    virtual AnyExpr resolve(const Eval& eval) const = 0;
};

using TunableMap = std::unordered_map<TunableParam, TunableValue>;
static const TunableMap EMPTY_CONFIG = {};

struct Eval {
    Eval(const TunableMap& mapping = EMPTY_CONFIG) : inner_(mapping) {
        //
    }

    TunableValue lookup(const TunableParam& param) const {
        return inner_.at(param);
    }

    TunableValue eval(const BaseExpr& expr) const {
        return expr.eval(*this);
    }

    template<typename T>
    T operator()(const Expr<T>& expr) const {
        return expr.get(*this);
    }

  private:
    const TunableMap& inner_;
};

namespace detail {
    std::true_type is_expr_helper(const BaseExpr*);
    std::false_type is_expr_helper(...);
}  // namespace detail

template<typename T>
constexpr bool is_expr = decltype(detail::is_expr_helper(
    std::declval<typename std::decay<T>::type*>()))::value;

struct SharedExpr: BaseExpr {
    SharedExpr(std::shared_ptr<BaseExpr> inner) : inner_(std::move(inner)) {}

    const BaseExpr& inner() const {
        return *inner_.get();
    }

  private:
    std::shared_ptr<BaseExpr> inner_ {};
};

template<typename T>
struct Expr: SharedExpr {
    Expr(T value = {}) : SharedExpr(std::make_shared<ScalarExpr>(value)) {}

    template<typename E, typename = typename std::enable_if<is_expr<E>>::type>
    Expr(E&& expr) :
        SharedExpr(std::make_shared<typename std::decay<E>::type>(
            std::forward<E>(expr))) {}

    Expr(Expr&) = default;
    Expr(const Expr&) = default;
    Expr(Expr&&) noexcept = default;
    Expr& operator=(const Expr&) = default;
    Expr& operator=(Expr&&) noexcept = default;

    std::string to_string() const override {
        return inner().to_string();
    }

    T get(const Eval& ctx) const {
        return inner().eval(ctx).template to<T>();
    }

    TunableValue eval(const Eval& ctx) const override {
        return get(ctx);
    }

    AnyExpr resolve(const Eval& eval) const override {
        AnyExpr result = this->resolve(eval);

        while (const Expr* v = dynamic_cast<const Expr*>(&result.inner())) {
            result = *v;
        }

        return result;
    }
};

struct ScalarExpr: BaseExpr {
    ScalarExpr(TunableValue v) : value_(std::move(v)) {}

    std::string to_string() const override {
        return value_.to_string();
    }

    TunableValue eval(const Eval& ctx) const override {
        return value_;
    }

    AnyExpr resolve(const Eval& eval) const override {
        return *this;
    }

    const TunableValue& value() const {
        return value_;
    }

  private:
    TunableValue value_;
};

template<typename T>
ScalarExpr scalar(T value = {}) {
    return ScalarExpr(std::move(value));
}

struct ParamExpr: BaseExpr {
    ParamExpr(TunableParam p) : param_(std::move(p)) {
        //
    }

    std::string to_string() const override {
        return "$" + param_.name();
    }

    TunableValue eval(const Eval& ctx) const override {
        return ctx.lookup(param_);
    }

    const TunableParam& parameter() const {
        return param_;
    }

    AnyExpr resolve(const Eval& eval) const override {
        return *this;
    }

  private:
    TunableParam param_;
};

template<typename T, typename E>
Expr<T> cast(E&& value) {
    return {std::forward<E>(value)};
}

template<typename E>
AnyExpr into_expr(E&& value) {
    return {std::forward<E>(value)};
}

struct SelectExpr: BaseExpr {
    SelectExpr(AnyExpr index, std::vector<AnyExpr> options) :
        cond_(std::move(index)),
        options_(std::move(options)) {}

    TunableValue eval(const Eval& ctx) const override {
        size_t index = cond_.eval(ctx).to<size_t>();
        if (index >= options_.size()) {
            throw std::invalid_argument("index out of bounds");
        }

        return options_[index].eval(ctx);
    }

    std::string to_string() const override {
        std::stringstream ss;
        ss << "select(" << cond_.to_string();
        for (const auto& v : options_) {
            ss << ", " << v.to_string();
        }
        ss << ")";

        return ss.str();
    }

    AnyExpr resolve(const Eval& eval) const override {
        std::vector<AnyExpr> options;
        for (const auto& v : options_) {
            options.push_back(v.resolve(eval));
        }

        return SelectExpr(cond_.resolve(eval), options);
    }

    const BaseExpr& condition() const {
        return cond_.inner();
    }

    const std::vector<AnyExpr>& operands() const {
        return options_;
    }

  private:
    AnyExpr cond_;
    std::vector<AnyExpr> options_;
};

template<typename C, typename... Es>
SelectExpr select(C&& cond, Es&&... operands) {
    return {
        into_expr(std::forward<C>(cond)),
        {into_expr(std::forward<Es>(operands))...}};
}

struct UnaryExpr: BaseExpr {
    enum struct Op {
        Plus,
        Minus,
        LogicNot,
        //BitNot,
    };

    UnaryExpr(Op op, AnyExpr operand) :
        operator_(op),
        operand_(std::move(operand)) {}

    TunableValue eval(const Eval& ctx) const override {
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

    std::string op_name() const {
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

    std::string to_string() const override {
        std::stringstream ss;
        ss << "(" << op_name() << " " << operand_.to_string() << ")";
        return ss.str();
    }

    Op op() const {
        return operator_;
    }

    AnyExpr resolve(const Eval& eval) const override {
        return UnaryExpr(operator_, operand_.resolve(eval));
    }

    const BaseExpr& operand() const {
        return operand_.inner();
    }

  private:
    Op operator_;
    AnyExpr operand_;
};

struct BinaryExpr: BaseExpr {
    enum struct Op {
        Add,
        Sub,
        Mul,
        Div,
        Mod,
        LogicAnd,
        LogicOr,
        //LogicXor,
        //BitAnd,
        //BitOr,
        //BItXor,
        Eq,
        Neq,
        Lt,
        Lte,
        Gt,
        Gte,
        //Shl, // These just cause confusion with iostreams
        //Shr,
    };

    BinaryExpr(Op op, AnyExpr left, AnyExpr right) :
        operator_(op),
        lhs_(std::move(left)),
        rhs_(std::move(right)) {}

    TunableValue eval(const Eval& ctx) const override {
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

    std::string op_name() const {
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

    std::string to_string() const override {
        std::stringstream ss;
        ss << "(" + lhs_.to_string() << " " << op_name() << " "
           << rhs_.to_string() << ")";
        return ss.str();
    }

    AnyExpr resolve(const Eval& eval) const override {
        return BinaryExpr(operator_, lhs_.resolve(eval), rhs_.resolve(eval));
    }

    Op op() const {
        return operator_;
    }

    const BaseExpr& left_operand() const {
        return lhs_.inner();
    }

    const BaseExpr& right_operand() const {
        return rhs_.inner();
    }

  private:
    Op operator_;
    AnyExpr lhs_;
    AnyExpr rhs_;
};

#define KERNEL_LAUNCHER_EXPR_UN_OP_IMPL(op, name)                              \
    template<typename E, typename = typename std::enable_if<is_expr<E>>::type> \
    UnaryExpr operator op(E expr) {                                            \
        return {UnaryExpr::Op::name, into_expr(expr)};                         \
    }

#define KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(op, name)                          \
    template<                                                               \
        typename L,                                                         \
        typename R,                                                         \
        typename = typename std::enable_if<is_expr<L> || is_expr<R>>::type> \
    BinaryExpr operator op(L left, R right) {                               \
        return {BinaryExpr::Op::name, into_expr(left), into_expr(right)};   \
    }

KERNEL_LAUNCHER_EXPR_UN_OP_IMPL(+, Plus)
KERNEL_LAUNCHER_EXPR_UN_OP_IMPL(-, Minus)
KERNEL_LAUNCHER_EXPR_UN_OP_IMPL(!, LogicNot)

KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(+, Add)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(-, Sub)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(*, Mul)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(/, Div)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(%, Mod)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(==, Eq)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(!=, Neq)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(<, Lt)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(>, Gt)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(<=, Lte)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(>=, Gte)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(&&, LogicAnd)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(||, LogicOr)
#undef KERNEL_LAUNCHER_EXPR_UN_OP_IMPL
#undef KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_EXPR_H
