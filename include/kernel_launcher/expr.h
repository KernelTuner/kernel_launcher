#ifndef KERNEL_LAUNCHER_EXPR_H
#define KERNEL_LAUNCHER_EXPR_H

#include <sstream>
#include <unordered_map>

#include "value.h"

namespace kernel_launcher {

struct Eval;

struct BaseExpr {
    virtual ~BaseExpr() {};
    virtual std::string to_string() const = 0;
    virtual TunableValue eval(const Eval& eval) const = 0;
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

    TunableValue operator()(const BaseExpr& expr) const {
        return eval(expr);
    }

  private:
    const TunableMap& inner_;
};

struct ScalarExpr: BaseExpr {
    ScalarExpr(TunableValue v) : value_(std::move(v)) {}

    std::string to_string() const override {
        return value_.to_string();
    }

    TunableValue eval(const Eval& ctx) const override {
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

  private:
    TunableParam param_;
};

namespace detail {
    std::true_type is_expr_helper(const BaseExpr*);
    std::false_type is_expr_helper(...);
}  // namespace detail

template<typename T>
constexpr bool is_expr = decltype(detail::is_expr_helper(
    std::declval<typename std::decay<T>::type*>()))::value;

struct ErasedExpr: BaseExpr {
    ErasedExpr(std::shared_ptr<BaseExpr> inner) : inner_(std::move(inner)) {}

    const BaseExpr& inner() const {
        return *inner_.get();
    }

  private:
    std::shared_ptr<BaseExpr> inner_ {};
};

template<typename T>
struct TypedExpr: ErasedExpr {
    TypedExpr(T value = {}) : ErasedExpr(std::make_shared<ScalarExpr>(value)) {}

    template<typename E, typename = typename std::enable_if<is_expr<E>>::type>
    TypedExpr(E&& expr) :
        ErasedExpr(std::make_shared<typename std::decay<E>::type>(
            std::forward<E>(expr))) {}

    TypedExpr(TypedExpr&) = default;
    TypedExpr(const TypedExpr&) = default;
    TypedExpr(TypedExpr&&) = default;
    TypedExpr& operator=(const TypedExpr&) = default;
    TypedExpr& operator=(TypedExpr&&) = default;

    std::string to_string() const override {
        return inner().to_string();
    }

    T get(const Eval& ctx) const {
        return inner().eval(ctx).template to<T>();
    }

    TunableValue eval(const Eval& ctx) const override {
        return get(ctx);
    }
};

template<typename T, typename E>
TypedExpr<T> cast(E&& value) {
    return {std::forward<E>(value)};
}

using Expr = TypedExpr<TunableValue>;

template<typename E>
Expr into_expr(E&& value) {
    return Expr {std::forward<E>(value)};
}

struct SelectExpr: BaseExpr {
    SelectExpr(Expr index, std::vector<Expr> options) :
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

    const BaseExpr& condition() const {
        return cond_.inner();
    }

    const std::vector<Expr>& operands() const {
        return options_;
    }

  private:
    Expr cond_;
    std::vector<Expr> options_;
};

template<typename C, typename... Es>
SelectExpr select(C&& cond, Es&&... operands) {
    return {
        into_expr(std::forward<C>(cond)),
        {into_expr(std::forward<Es>(operands)...)}};
}

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

    BinaryExpr(Op op, Expr left, Expr right) :
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
    Expr lhs_;
    Expr rhs_;
};

#define KERNEL_LAUNCHER_EXPR_OP_IMPL(op, name)                              \
    template<                                                               \
        typename L,                                                         \
        typename R,                                                         \
        typename = typename std::enable_if<is_expr<L> || is_expr<R>>::type> \
    BinaryExpr operator op(L left, R right) {                               \
        return {BinaryExpr::Op::name, into_expr(left), into_expr(right)};   \
    }

KERNEL_LAUNCHER_EXPR_OP_IMPL(+, Add)
KERNEL_LAUNCHER_EXPR_OP_IMPL(-, Sub)
KERNEL_LAUNCHER_EXPR_OP_IMPL(*, Mul)
KERNEL_LAUNCHER_EXPR_OP_IMPL(/, Div)
KERNEL_LAUNCHER_EXPR_OP_IMPL(%, Mod)
KERNEL_LAUNCHER_EXPR_OP_IMPL(==, Eq)
KERNEL_LAUNCHER_EXPR_OP_IMPL(!=, Neq)
KERNEL_LAUNCHER_EXPR_OP_IMPL(<, Lt)
KERNEL_LAUNCHER_EXPR_OP_IMPL(>, Gt)
KERNEL_LAUNCHER_EXPR_OP_IMPL(<=, Lte)
KERNEL_LAUNCHER_EXPR_OP_IMPL(>=, Gte)
KERNEL_LAUNCHER_EXPR_OP_IMPL(&&, LogicAnd)
KERNEL_LAUNCHER_EXPR_OP_IMPL(||, LogicOr)
#undef KERNEL_LAUNCHER_EXPR_OP_IMPL

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_EXPR_H
