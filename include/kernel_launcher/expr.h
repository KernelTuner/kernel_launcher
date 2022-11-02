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

struct Eval {
    Eval() = default;
    virtual ~Eval() = default;

    virtual bool lookup(const Variable& v, TunableValue& out) const = 0;

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
};

struct ScalarExpr: BaseExpr {
    ScalarExpr(TunableValue v) : value_(std::move(v)) {}

    std::string to_string() const override;
    TunableValue eval(const Eval& ctx) const override;
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

    std::string to_string() const override;
    TunableValue eval(const Eval& ctx) const override;
    Expr resolve(const Eval& eval) const override;

    const TunableParam& parameter() const {
        return param_;
    }

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
            return std::move(expr);
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

    bool is_constant() const {
        return dynamic_cast<const ScalarExpr*>(inner_.get()) != nullptr;
    }

  private:
    std::shared_ptr<BaseExpr> inner_ {};
};

template<typename T>
struct TypedExpr: SharedExpr {
    template<
        typename R,
        typename = typename detail::into_expr_helper<R, T>::type>
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
};

template<typename T, typename E>
TypedExpr<T> cast(E&& value) {
    return {std::forward<E>(value)};
}

template<typename E>
Expr into_expr(E&& value) {
    return {std::forward<E>(value)};
}

struct ProblemExpr: BaseExpr, Variable {
    ProblemExpr(size_t axis) : axis_(axis) {
        if (axis_ >= 3) {
            throw std::runtime_error("axis out of bounds");
        }
    }

    std::string to_string() const override;
    TunableValue eval(const Eval& eval) const override;
    Expr resolve(const Eval& eval) const override;

    size_t axis() const {
        return axis_;
    }

    bool equals(const Variable& v) const override {
        if (auto that = dynamic_cast<const ProblemExpr*>(&v)) {
            return this->axis_ == that->axis_;
        } else {
            return false;
        }
    }

    size_t hash() const override {
        return axis_;
    }

  private:
    size_t axis_;
};

extern ProblemExpr problem_size_x, problem_size_y, problem_size_z;

inline ProblemExpr problem_size(size_t axis = 0) {
    return axis;
}

struct ArgExpr: BaseExpr, Variable {
    constexpr ArgExpr(uint8_t i) : index_(i) {};
    std::string to_string() const override;
    TunableValue eval(const Eval& eval) const override;
    Expr resolve(const Eval& eval) const override;

    bool equals(const Variable& v) const override {
        if (auto that = dynamic_cast<const ArgExpr*>(&v)) {
            return this->index_ == that->index_;
        } else {
            return false;
        }
    }

    size_t hash() const override {
        return index_;
    }

    size_t get() const {
        return index_;
    }

  private:
    uint8_t index_;
};

extern ArgExpr arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8;

inline ArgExpr arg(uint8_t i) {
    return i;
}

struct SelectExpr: BaseExpr {
    SelectExpr(Expr index, std::vector<Expr> options) :
        cond_(std::move(index)),
        options_(std::move(options)) {}

    TunableValue eval(const Eval& ctx) const override;
    std::string to_string() const override;
    Expr resolve(const Eval& eval) const override;

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

/**
 * Returns a new expression that first evaluates ``index``` as an integer `n`
 * and then evaluates the `n`-th operand in the list of ``operand`` expressions.
 * For example, if ``index`` returns 0, then the first operand is evaluated.
 * Note that ``index`` can also be a boolean expression, in which case ``false``
 * is interpreted as ``0`` and ``true`` is interpreted as ``1``.
 *
 * @param index The index expression
 * @param operands The operand expressions.
 */
template<typename C, typename... Es>
SelectExpr select(C&& index, Es&&... operands) {
    return {
        into_expr(std::forward<C>(index)),
        {into_expr(std::forward<Es>(operands))...}};
}

/**
 * Returns a new expression that first evaluates ``index``` as an integer `n`
 * and then evaluates the `n`-th operand in the list of ``operand`` expressions.
 * For example, if ``index`` returns 0, then the first operand is evaluated.
 *
 * This function is equivalent to ``select``, expect it takes an iterable
 * (e..g, vector, array) instead of a variadic list of arguments.
 *
 * @param index The index expression
 * @param operands The operand expressions.
 */
template<typename C, typename Es>
SelectExpr index(C&& cond, Es&& operands) {
    std::vector<Expr> options(std::begin(operands), std::end(operands));
    return {into_expr(std::forward<C>(cond)), options};
}

/**
 * Returns a new expression that first evaluates ``cond`` and then evaluates
 * either ``true_expr`` if the condition is true and ``false_expr`` otherwise.
 *
 * @param cond Condition expression.
 * @param true_expr The expression if the condition is true.
 * @param false_expr The expression if the condition is false.
 */
template<typename C, typename ET, typename EF>
SelectExpr ifelse(C&& cond, ET&& true_expr, EF&& false_expr) {
    return {
        cast<bool>(std::forward<C>(cond)),
        {into_expr(std::forward<EF>(false_expr)),
         into_expr(std::forward<ET>(true_expr))}};
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

    TunableValue eval(const Eval& ctx) const override;
    std::string to_string() const override;
    Expr resolve(const Eval& eval) const override;
    std::string op_name() const;

    Op op() const {
        return operator_;
    }

    const BaseExpr& operand() const {
        return operand_.inner();
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

    TunableValue eval(const Eval& ctx) const override;
    std::string to_string() const override;
    Expr resolve(const Eval& eval) const override;

    std::string op_name() const;

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
