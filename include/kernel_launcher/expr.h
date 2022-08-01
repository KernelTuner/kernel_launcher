#ifndef KERNEL_LAUNCHER_EXPR_H
#define KERNEL_LAUNCHER_EXPR_H

#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "kernel_launcher/value.h"

namespace kernel_launcher {

struct Eval;

template<typename T>
struct TypedExpr;
using Expr = TypedExpr<TunableValue>;
struct ScalarExpr;

struct ParamExpr;

struct BaseExpr {
    virtual ~BaseExpr() = default;
    virtual std::string to_string() const = 0;
    virtual TunableValue eval(const Eval& eval) const = 0;
    virtual Expr resolve(const Eval& eval) const = 0;
};

using TunableMap = std::unordered_map<TunableParam, TunableValue>;
static const TunableMap EMPTY_CONFIG = {};

struct Eval {
    Eval() {}

    Eval(const TunableMap& mapping) : inner_(mapping) {
        //
    }

    Eval(const TunableMap& mapping, ProblemSize problem_size) :
        inner_(mapping),
        problem_size_(problem_size),
        has_problem_size_(true) {
        //
    }

    Eval(ProblemSize problem_size) :
        problem_size_(problem_size),
        has_problem_size_(true) {
        //
    }

    bool has_problem_size() const {
        return has_problem_size_;
    }

    uint32_t problem_size(size_t axis) const {
        if (!has_problem_size_) {
            throw std::runtime_error("expression cannot be problem-dependent");
        }

        return problem_size_[axis];
    }

    bool has(const TunableParam& param) const {
        return inner_.find(param) != inner_.end();
    }

    TunableValue lookup(const TunableParam& param) const {
        return inner_.at(param);
    }

    TunableValue eval(const BaseExpr& expr) const {
        return expr.eval(*this);
    }

    template<typename T>
    T operator()(const TypedExpr<T>& expr) const {
        try {
            return expr.eval(*this).template to<T>();
        } catch (const std::exception& e) {
            log_warning() << "error while evaluating expression \""
                          << expr.to_string() << "\": " << e.what() << "\n";
            throw;
        }
    }

  private:
    const TunableMap& inner_ = EMPTY_CONFIG;
    ProblemSize problem_size_ = {};
    bool has_problem_size_ = false;
};

struct SharedExpr: BaseExpr {
    SharedExpr() noexcept = default;

    template<typename E>
    SharedExpr(std::shared_ptr<E> inner) : inner_(std::move(inner)) {}

    const BaseExpr& inner() const {
        if (!inner_) {
            throw std::runtime_error("SharedExpr is not initialized");
        }

        return *inner_;
    }

    std::string to_string() const override {
        return inner().to_string();
    }

    TunableValue eval(const Eval& ctx) const override {
        return inner().eval(ctx);
    }

  private:
    std::shared_ptr<BaseExpr> inner_ {};
};

struct ScalarExpr: BaseExpr {
    ScalarExpr(TunableValue v) : value_(std::move(v)) {}

    std::string to_string() const override {
        return value_.to_string();
    }

    TunableValue eval(const Eval& ctx) const override {
        return value_;
    }

    Expr resolve(const Eval& eval) const override;

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

    Expr resolve(const Eval& eval) const override;

  private:
    TunableParam param_;
};

namespace detail {
    std::true_type is_expr_helper(const BaseExpr*);
    std::false_type is_expr_helper(...);

    template<typename T>
    constexpr bool is_expr = decltype(detail::is_expr_helper(
        std::declval<typename std::decay<T>::type*>()))::value;

    template<typename I, typename T, typename Enabled = void>
    struct into_expr_helper;

    // TunableParam -> ParamExpr
    template<typename I, typename T>
    struct into_expr_helper<
        I,
        T,
        typename std::enable_if<
            std::is_same<typename std::decay<I>::type, TunableParam>::value>::
            type> {
        using type = ParamExpr;

        static type call(TunableParam p) {
            return ParamExpr(std::move(p));
        }
    };

    // TypedExpr -> TypedExpr
    template<typename E, typename T>
    struct into_expr_helper<
        E,
        T,
        typename std::enable_if<detail::is_expr<E>>::type> {
        using type = typename std::decay<E>::type;

        static type call(E&& expr) {
            return expr;
        }
    };

    // R -> ScalarExpr (Where R is convertible to T)
    template<typename R, typename T>
    struct into_expr_helper<
        R,
        T,
        typename std::enable_if<std::is_convertible<R, T>::value>::type> {
        using type = ScalarExpr;

        static ScalarExpr call(R&& value) {
            return ScalarExpr(T(std::forward<R>(value)));
        }
    };
}  // namespace detail

template<typename T>
struct TypedExpr: SharedExpr {
    template<typename R>
    TypedExpr(R&& value) :
        SharedExpr(
            std::make_shared<typename detail::into_expr_helper<R, T>::type>(
                detail::into_expr_helper<R, T>::call(std::forward<R>(value)))) {

    }

    TypedExpr() = delete;
    TypedExpr(TypedExpr&) = default;
    TypedExpr(const TypedExpr&) = default;
    TypedExpr(TypedExpr&&) noexcept = default;
    TypedExpr& operator=(const TypedExpr&) = default;
    TypedExpr& operator=(TypedExpr&&) noexcept = default;

    Expr resolve(const Eval& eval) const override {
        Expr result = inner().resolve(eval);

        while (const TypedExpr* v =
                   dynamic_cast<const TypedExpr*>(&result.inner())) {
            result = *v;
        }

        return result;
    }

    bool is_constant() const {
        return dynamic_cast<const ScalarExpr*>(&inner()) != nullptr;
    }
};

template<typename T, typename E>
TypedExpr<T> cast(E&& value) {
    return {std::forward<E>(value)};
}

template<typename E>
Expr into_expr(E&& value) {
    return {std::forward<E>(value)};
}

inline Expr ScalarExpr::resolve(const Eval& eval) const {
    return *this;
}

inline Expr ParamExpr::resolve(const Eval& eval) const {
    if (eval.has(param_)) {
        return ScalarExpr(eval.lookup(param_));
    } else {
        return ParamExpr(param_);
    }
}

struct ProblemExpr: BaseExpr {
    ProblemExpr(size_t axis) : axis_(axis) {
        if (axis_ >= 3) {
            throw std::runtime_error("axis out of bounds");
        }
    }

    std::string to_string() const override {
        std::stringstream ss;
        ss << "$problem_size_" << axis_;
        return ss.str();
    }

    TunableValue eval(const Eval& eval) const override {
        return eval.problem_size(axis_);
    }

    Expr resolve(const Eval& eval) const override {
        if (eval.has_problem_size()) {
            return ScalarExpr(eval.problem_size(axis_));
        } else {
            return ProblemExpr(axis_);
        }
    }

    size_t axis() const {
        return axis_;
    }

  private:
    size_t axis_;
};

inline ProblemExpr problem_size(size_t axis = 0) {
    return axis;
}

inline ProblemExpr problem_size_x() {
    return problem_size(0);
}

inline ProblemExpr problem_size_y() {
    return problem_size(1);
}

inline ProblemExpr problem_size_z() {
    return problem_size(2);
}

struct SelectExpr: BaseExpr {
    SelectExpr(Expr index, std::vector<Expr> options) :
        cond_(std::move(index)),
        options_(std::move(options)) {}

    TunableValue eval(const Eval& ctx) const override {
        auto index = cond_.eval(ctx).to<int64_t>();
        if (index < 0 || size_t(index) >= options_.size()) {
            throw std::invalid_argument("index out of bounds");
        }

        return options_[size_t(index)].eval(ctx);
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

    Expr resolve(const Eval& eval) const override {
        Expr cond = cond_.resolve(eval);

        std::vector<Expr> options;
        for (const auto& v : options_) {
            options.push_back(v.resolve(eval));
        }

        return SelectExpr(cond_.resolve(eval), options);
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
        {into_expr(std::forward<Es>(operands))...}};
}

struct UnaryExpr: BaseExpr {
    enum struct Op {
        Plus,
        Minus,
        LogicNot,
        //BitNot,
    };

    UnaryExpr(Op op, Expr operand) :
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

    const BaseExpr& operand() const {
        return operand_.inner();
    }

    Expr resolve(const Eval& eval) const override {
        UnaryExpr result(operator_, operand_.resolve(eval));

        if (result.operand_.is_constant()) {
            return ScalarExpr(result.eval({}));
        } else {
            return result;
        }
    }

  private:
    Op operator_;
    Expr operand_;
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

    Expr resolve(const Eval& eval) const override {
        BinaryExpr result(operator_, lhs_.resolve(eval), rhs_.resolve(eval));

        if (result.lhs_.is_constant() && result.lhs_.is_constant()) {
            return ScalarExpr(result.eval({}));
        } else {
            return result;
        }
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

#define KERNEL_LAUNCHER_EXPR_UN_OP_IMPL(op, name)                     \
    template<                                                         \
        typename E,                                                   \
        typename = typename std::enable_if<detail::is_expr<E>>::type> \
    UnaryExpr operator op(E expr) {                                   \
        return {UnaryExpr::Op::name, into_expr(expr)};                \
    }

#define KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(op, name)                        \
    template<                                                             \
        typename L,                                                       \
        typename R,                                                       \
        typename = typename std::enable_if<                               \
            detail::is_expr<L> || detail::is_expr<R>>::type>              \
    BinaryExpr operator op(L left, R right) {                             \
        return {BinaryExpr::Op::name, into_expr(left), into_expr(right)}; \
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
