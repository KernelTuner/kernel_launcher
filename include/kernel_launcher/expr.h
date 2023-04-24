#ifndef KERNEL_LAUNCHER_EXPR_H
#define KERNEL_LAUNCHER_EXPR_H

#include <cuda.h>

#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "kernel_launcher/value.h"

namespace kernel_launcher {

struct Eval;

template<typename T>
struct TypedExpr;
using Expr = TypedExpr<Value>;
struct ScalarExpr;

struct ParamExpr;

struct BaseExpr {
    virtual ~BaseExpr() = default;
    virtual std::string to_string() const = 0;
    virtual Value eval(const Eval& eval) const = 0;
    virtual Expr resolve(const Eval& eval) const = 0;
};

struct Eval {
    Eval() = default;
    virtual ~Eval() = default;

    virtual bool lookup(const Variable& v, Value& out) const = 0;

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
    ScalarExpr(Value v) : value_(std::move(v)) {}

    std::string to_string() const override;
    Value eval(const Eval& ctx) const override;
    Expr resolve(const Eval& eval) const override;

    const Value& value() const {
        return value_;
    }

  private:
    Value value_;
};

template<typename T>
ScalarExpr scalar(T value = {}) {
    return ScalarExpr(std::move(value));
}

struct ParamExpr: BaseExpr {
    ParamExpr(const TunableParam& p) : param_(p) {
        //
    }

    std::string to_string() const override;
    Value eval(const Eval& ctx) const override;
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

    static type call(const TunableParam& p) {
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
        return std::forward<E>(expr);
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

    Value eval(const Eval& ctx) const override {
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
    constexpr ProblemExpr(size_t axis) noexcept : axis_(axis) {}

    std::string to_string() const override;
    Value eval(const Eval& eval) const override;
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

static ProblemExpr problem_size_x = 0, problem_size_y = 1, problem_size_z = 2;

inline ProblemExpr problem_size(size_t axis = 0) {
    return axis;
}

struct ArgBuffer {
    uint8_t index;
    TypedExpr<size_t> length;
};

struct ArgExpr: BaseExpr, Variable {
    constexpr ArgExpr() noexcept = default;
    constexpr ArgExpr(uint8_t i) noexcept : index_(i) {};

    ArgExpr(uint8_t i, const char* name) noexcept;
    ArgExpr(uint8_t i, const std::string& name) noexcept :
        ArgExpr(i, name.c_str()) {};

    std::string to_string() const override;
    Value eval(const Eval& eval) const override;
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

    ArgBuffer operator[](TypedExpr<size_t> len) const {
        return {index_, std::move(len)};
    }

  private:
    uint8_t index_ = std::numeric_limits<uint8_t>::max();
    const std::string* name_ = nullptr;
};

static ArgExpr arg0 = 0, arg1 = 1, arg2 = 2, arg3 = 3, arg4 = 4, arg5 = 5,
               arg6 = 6, arg7 = 7, arg8 = 8;

inline ArgExpr arg(uint8_t i) {
    return i;
}

namespace detail {
template<typename T>
struct ArgsHelper;

template<size_t... Is>
struct ArgsHelper<std::index_sequence<Is...>> {
    using type = std::tuple<typename std::enable_if<
        Is <= std::numeric_limits<uint8_t>::max(),
        ArgExpr>::type...>;

    static type call() {
        return {Is...};
    }

    static type call(std::string* names) {
        return {{Is, std::move(names[Is])}...};
    }
};
}  // namespace detail

template<size_t N>
using args_type =
    typename detail::ArgsHelper<std::make_index_sequence<N>>::type;

/**
 * Given a template parameter `N`, returns a tuple of size `N` that contains
 * `ArgExpr` for each element. This function can be used in combination with
 * `std::tie` or structured binding to quickly assign names to argument
 * expression. For example, to bind four arguments to the variables `n`, `A`,
 * `B`, and `C`:
 *
 * ```
 * auto [n, A, B, C] = kernel_launcher::args<4>();
 * ```
 *
 * The above example only works in C++17 or higher. For, C++11 or higher, one
 * can use `std::tie` as follows:
 *
 * ```
 * ArgExpr n, A, B, C;
 * std::tie(n, A, B, C) = kernel_launcher::args<4>();
 * ```
 */
template<size_t N>
inline args_type<N> args() {
    return detail::ArgsHelper<std::make_index_sequence<N>>::call();
}

/**
 * Given `N` argument names, returns a tuple of size `N` that contains
 * `ArgExpr` for each element. This function can be used in combination with
 * `std::tie` or structured binding to quickly assign names to argument
 * expression. For example, to bind four arguments to the variables `n`, `A`,
 * `B`, and `C` with the corresponding names:
 *
 * ```
 * auto [n, A, B, C] = kernel_launcher::args("n", "A", "B", "C");
 * ```
 *
 * The above example only works in C++17 or higher. For, C++11 or higher, one
 * can use `std::tie` as follows:
 *
 * ```
 * ArgExpr n, A, B, C;
 * std::tie(n, A, B, C) = kernel_launcher::args("n", "A", "B", "C");
 * ```
 */
template<typename... Ts>
inline args_type<sizeof...(Ts)> args(Ts&&... names) {
    static constexpr size_t N = sizeof...(Ts);
    std::string strings[N] = {names...};

    return detail::ArgsHelper<std::make_index_sequence<N>>::call(strings);
}

struct DeviceAttributeExpr: BaseExpr, Variable {
    constexpr DeviceAttributeExpr(CUdevice_attribute attribute) noexcept :
        attribute_(attribute) {}

    CUdevice_attribute get() const {
        return attribute_;
    }

    std::string to_string() const override;
    Value eval(const Eval& eval) const override;
    Expr resolve(const Eval& eval) const override;

    bool equals(const Variable& v) const override {
        if (auto that = dynamic_cast<const DeviceAttributeExpr*>(&v)) {
            return this->attribute_ == that->attribute_;
        }

        return false;
    }

    size_t hash() const override {
        return attribute_;
    }

  private:
    CUdevice_attribute attribute_;
};

#define KERNEL_LAUNCHER_DEVICE_ATTRIBUTES_FORALL(F) \
    F(COMPUTE_CAPABILITY_MAJOR)                     \
    F(COMPUTE_CAPABILITY_MINOR)                     \
    F(MAX_BLOCKS_PER_MULTIPROCESSOR)                \
    F(MAX_BLOCK_DIM_X)                              \
    F(MAX_BLOCK_DIM_Y)                              \
    F(MAX_BLOCK_DIM_Z)                              \
    F(MAX_GRID_DIM_X)                               \
    F(MAX_GRID_DIM_Y)                               \
    F(MAX_GRID_DIM_Z)                               \
    F(MAX_REGISTERS_PER_BLOCK)                      \
    F(MAX_REGISTERS_PER_MULTIPROCESSOR)             \
    F(MAX_SHARED_MEMORY_PER_BLOCK)                  \
    F(MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)         \
    F(MAX_THREADS_PER_BLOCK)                        \
    F(MAX_THREADS_PER_MULTIPROCESSOR)               \
    F(MULTIPROCESSOR_COUNT)                         \
    F(WARP_SIZE)

#define KERNEL_LAUNCHER_DEFINE_DEVICE_ATTRIBUTE(name) \
    static DeviceAttributeExpr DEVICE_##name = CU_DEVICE_ATTRIBUTE_##name;

KERNEL_LAUNCHER_DEVICE_ATTRIBUTES_FORALL(
    KERNEL_LAUNCHER_DEFINE_DEVICE_ATTRIBUTE)

struct SelectExpr: BaseExpr {
    SelectExpr(Expr index, std::vector<Expr> options) :
        cond_(std::move(index)),
        options_(std::move(options)) {}

    Value eval(const Eval& ctx) const override;
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
 * Returns a new expression that first evaluates ``index`` as an integer `n`
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
 * Returns a new expression that first evaluates ``index`` as an integer `n`
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
        RoundNearest,
        RoundUp,
        RoundDown
        //BitNot,
    };

    UnaryExpr(Op op, Expr operand) :
        operator_(op),
        operand_(std::move(operand)) {}

    Value eval(const Eval& ctx) const override;
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

inline UnaryExpr round(Expr e) {
    return {UnaryExpr::Op::RoundNearest, e};
}

inline UnaryExpr floor(Expr e) {
    return {UnaryExpr::Op::RoundDown, e};
}

inline UnaryExpr ceil(Expr e) {
    return {UnaryExpr::Op::RoundUp, e};
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

    Value eval(const Eval& ctx) const override;
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
