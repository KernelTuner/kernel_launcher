#ifndef KERNEL_LAUNCHER_EXPR_H
#define KERNEL_LAUNCHER_EXPR_H

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "kernel_launcher/gpu_utils.hpp"
#include "kernel_launcher/kernel_arg.hpp"
#include "kernel_launcher/primitive.hpp"

namespace kernel_launcher {

/**
 * @addtogroup expr
 * @{
 */

class Variable;
class ExprPtr;

class EvalContext {
  public:
    virtual ~EvalContext() = default;
    virtual bool try_lookup(const Variable& v, Primitive& out) const = 0;
    Primitive lookup(const Variable& v) const;
};

struct DummyEval: EvalContext {
    bool try_lookup(const Variable& /*v*/, Primitive& /*out*/) const override {
        return false;
    }
};

template<typename T = Primitive>
class ExprBase {
  public:
    virtual ~ExprBase() = default;
    virtual std::string to_string() const = 0;
    virtual T eval(const EvalContext& eval) const = 0;
    virtual ExprPtr resolve(const EvalContext& eval) const = 0;

    virtual bool is_constant() const {
        return false;
    }
};

class Variable: public ExprBase<Primitive> {
  public:
    Primitive eval(const EvalContext& context) const override final {
        return context.lookup(*this);
    }
};

inline Primitive EvalContext::lookup(const Variable& v) const {
    Primitive out;

    if (!try_lookup(v, out)) {
        throw std::runtime_error("unknown variable '" + v.to_string() + "'");
    }

    return out;
}

class ExprPtr: public ExprBase<Primitive> {
  public:
    explicit ExprPtr(std::shared_ptr<ExprBase<Primitive>> ptr) : ptr_(std::move(ptr)) {
        if (ptr_ == nullptr) {
            throw std::invalid_argument("ExprPtr cannot be constructed from a null pointer");
        }
    }

    template<typename Derived, typename... Args>
    static ExprPtr make(Args&&... args) {
        return ExprPtr(std::make_shared<Derived>(std::forward<Args>(args)...));
    }

    std::string to_string() const override {
        return ptr_->to_string();
    }

    Primitive eval(const EvalContext& eval) const override {
        return ptr_->eval(eval);
    }

    ExprPtr resolve(const EvalContext& eval) const override {
        return ptr_->resolve(eval);
    }

    bool is_constant() const override {
        return ptr_->is_constant();
    }

    const ExprBase<Primitive>& inner() const {
        return *ptr_;
    }

  private:
    std::shared_ptr<ExprBase<Primitive>> ptr_;
};

struct ScalarExpr: ExprBase<Primitive> {
    ScalarExpr(Primitive v = {}) : value_(std::move(v)) {}

    std::string to_string() const override {
        return value_.to_string();
    }

    Primitive eval(const EvalContext&) const override {
        return value_;
    }

    ExprPtr resolve(const EvalContext&) const override {
        return ExprPtr::make<ScalarExpr>(value_);
    }

    bool is_constant() const override {
        return true;
    }

    const Primitive& value() const {
        return value_;
    }

  private:
    Primitive value_;
};

template<typename T>
struct is_expr: std::is_base_of<ExprBase<Primitive>, typename std::decay<T>::type> {};

template<typename T, typename Enable = void>
struct IntoExprTraits {
    static ExprPtr convert(T value) {
        return ExprPtr::make<ScalarExpr>(Primitive(std::move(value)));
    }
};

template<typename T>
struct IntoExprTraits<T, typename std::enable_if<is_expr<T>::value>::type> {
    static ExprPtr convert(T value) {
        return ExprPtr::make<T>(std::move(value));
    }
};

template<>
struct IntoExprTraits<ExprPtr> {
    static ExprPtr convert(ExprPtr value) {
        return value;
    }
};

template<typename T>
ExprPtr into_expr(T value) {
    return IntoExprTraits<T>::convert(std::move(value));
}

struct ArgumentBufferSize {
    size_t index;
    ExprPtr length;
};

/** @} */

namespace detail {
class ArgumentMapper {
  public:
    virtual ~ArgumentMapper() = default;
    virtual Primitive apply(const KernelArg& arg) const = 0;
};

// Only defined for lambdas `F` that take exactly one parameter.
template<typename F, typename M = decltype(&F::operator())>
class ArgumentMapperImpl;

template<typename F, typename C, typename R, typename A>
class ArgumentMapperImpl<F, R (C::*)(A) const>: public ArgumentMapper {
  public:
    ArgumentMapperImpl(F fun) : fun_(std::move(fun)) {}

    Primitive apply(const KernelArg& arg) const override {
        return Primitive(fun_(arg.get<typename std::decay<A>::type>()));
    }

  private:
    F fun_;
};

// Indicates whether `ArgumentMapperImpl<F>` is defined for type `F`.
template<typename F, typename = void>
struct HasArgumentMapper: std::false_type {};

template<typename F>
struct HasArgumentMapper<F, std::void_t<decltype(sizeof(ArgumentMapperImpl<F>))>>:
    std::true_type {};
}  // namespace detail

/**
 * @addtogroup expr
 * @{
 */

class ArgumentExpr: public Variable {
  public:
    ArgumentExpr(size_t index = 0, std::string_view name = {}) : index_(index), name_(name) {}

    ExprPtr resolve(const EvalContext& eval) const override final {
        Primitive out;
        return eval.try_lookup(*this, out) ? ExprPtr::make<ScalarExpr>(out)
                                           : ExprPtr::make<ArgumentExpr>(*this);
    }

    std::string to_string() const override {
        std::string result = "$arg_" + std::to_string(index_);

        if (!name_.empty()) {
            result += ":" + name_.get();
        }

        if (mapper_ != nullptr) {
            result = "map(" + result + ")";
        }

        return result;
    }

    size_t index() const noexcept {
        return index_;
    }

    /**
     * Returns a copy of this expression that converts the kernel argument into a value by
     * calling `fun`, instead of using `KernelArg::to_primitive`. The argument is passed to `fun`
     * as the type of its first parameter and the result is converted to a `Primitive`.
     *
     * Usage: `arg(0).map([](const Matrix& m) { return m.rows; })`
     */
    template<typename F>
    ArgumentExpr map(F fun) const {
        static_assert(
            detail::HasArgumentMapper<F>::value,
            "ArgumentExpr::map requires a non-generic, non-mutable lambda that takes exactly "
            "one parameter");

        ArgumentExpr result = *this;
        result.mapper_ = std::make_shared<detail::ArgumentMapperImpl<F>>(std::move(fun));
        return result;
    }

    /**
     * Converts the given kernel argument into a value. Returns an empty `Primitive` if the
     * conversion is not possible.
     */
    Primitive get(const KernelArg& arg) const {
        return mapper_ != nullptr ? mapper_->apply(arg) : arg.to_primitive_or_empty();
    }

    template<typename E>
    ArgumentBufferSize operator[](const E& length) const {
        return {index_, into_expr(length)};
    }

  private:
    size_t index_;
    Symbol name_;
    std::shared_ptr<const detail::ArgumentMapper> mapper_;
};

/**
 * Returns the expression for the kernel argument at position `index`. The optional `name`
 * is only used when printing the expression.
 */
inline ArgumentExpr arg(size_t index, std::string_view name = {}) {
    return ArgumentExpr(index, name);
}

/**
 * Returns the expressions for the first `N` kernel arguments.
 *
 * Usage: `auto [n, C, A, B] = args<4>();`
 */
template<size_t N>
std::array<ArgumentExpr, N> args() {
    std::array<ArgumentExpr, N> result;

    for (size_t i = 0; i < N; i++) {
        result[i] = ArgumentExpr(i);
    }

    return result;
}

/**
 * Returns the expressions for the first `N` kernel arguments. The names are used when printing the
 * expression and are useful for debugging, but do not serve a function.
 *
 * Usage: `auto [n, C, A, B] = args("n", "C", "A", "B");`
 */
template<typename... Ts>
std::array<ArgumentExpr, sizeof...(Ts)> args(const Ts&... names) {
    size_t i = 0;
    return {ArgumentExpr(i++, names)...};
}

class ProblemSizeExpr: public Variable {
  public:
    ProblemSizeExpr(size_t axis = 0) : axis_(axis) {}

    ExprPtr resolve(const EvalContext& eval) const override final {
        Primitive out;
        return eval.try_lookup(*this, out) ? ExprPtr::make<ScalarExpr>(out)
                                           : ExprPtr::make<ProblemSizeExpr>(*this);
    }

    std::string to_string() const override {
        static const char names[3] = {'x', 'y', 'z'};
        return std::string("$problem_size_") + (axis_ < 3 ? names[axis_] : '?');
    }

    size_t axis() const noexcept {
        return axis_;
    }

  private:
    size_t axis_;
};

class FeatureExpr: public Variable {
  public:
    FeatureExpr(std::string_view name) : name_(name) {}

    ExprPtr resolve(const EvalContext& eval) const override final {
        Primitive out;
        return eval.try_lookup(*this, out) ? ExprPtr::make<ScalarExpr>(out)
                                           : ExprPtr::make<FeatureExpr>(*this);
    }

    std::string to_string() const override {
        return concat("$feature_", name_);
    }

    Symbol name() const noexcept {
        return name_;
    }

  private:
    Symbol name_;
};

#define KERNEL_LAUNCHER_FOR_EACH_DEVICE_ATTRIBUTE(F) \
    F(MAX_BLOCKS_PER_MULTIPROCESSOR)                 \
    F(MAX_BLOCK_DIM_X)                               \
    F(MAX_BLOCK_DIM_Y)                               \
    F(MAX_BLOCK_DIM_Z)                               \
    F(MAX_GRID_DIM_X)                                \
    F(MAX_GRID_DIM_Y)                                \
    F(MAX_GRID_DIM_Z)                                \
    F(MAX_REGISTERS_PER_BLOCK)                       \
    F(MAX_REGISTERS_PER_MULTIPROCESSOR)              \
    F(MAX_SHARED_MEMORY_PER_BLOCK)                   \
    F(MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)          \
    F(MAX_THREADS_PER_BLOCK)                         \
    F(MAX_THREADS_PER_MULTIPROCESSOR)                \
    F(MULTIPROCESSOR_COUNT)                          \
    F(WARP_SIZE)

class DeviceAttributeExpr: public Variable {
  public:
    DeviceAttributeExpr(CUdevice_attribute attribute) : attribute_(attribute) {}

    ExprPtr resolve(const EvalContext& eval) const override final {
        Primitive out;
        return eval.try_lookup(*this, out) ? ExprPtr::make<ScalarExpr>(out)
                                           : ExprPtr::make<DeviceAttributeExpr>(*this);
    }

    std::string to_string() const override {
        switch (attribute_) {
#define KERNEL_LAUNCHER_IMPL(name)   \
    case CU_DEVICE_ATTRIBUTE_##name: \
        return "$device_attribute_" #name;

            KERNEL_LAUNCHER_FOR_EACH_DEVICE_ATTRIBUTE(KERNEL_LAUNCHER_IMPL)
#undef KERNEL_LAUNCHER_IMPL

            default:
                return "$device_attribute_" + std::to_string(static_cast<int>(attribute_));
        }
    }

    CUdevice_attribute attribute() const noexcept {
        return attribute_;
    }

  private:
    CUdevice_attribute attribute_;
};

inline DeviceAttributeExpr device_attribute(CUdevice_attribute attribute) {
    return DeviceAttributeExpr(attribute);
}

#define KERNEL_LAUNCHER_FOR_EACH_KERNEL_ATTRIBUTE(F) \
    F(MAX_THREADS_PER_BLOCK)                         \
    F(SHARED_SIZE_BYTES)                             \
    F(CONST_SIZE_BYTES)                              \
    F(LOCAL_SIZE_BYTES)                              \
    F(NUM_REGS)                                      \
    F(MAX_DYNAMIC_SHARED_SIZE_BYTES)                 \
    F(PREFERRED_SHARED_MEMORY_CARVEOUT)

// Attribute of a compiled kernel function (register usage, static shared
// memory size, ...). Unlike DeviceAttributeExpr, this is only resolvable
// once a kernel has actually been compiled, i.e. while building the launch
// configuration, not while building the kernel definition itself.
class KernelAttributeExpr: public Variable {
  public:
    KernelAttributeExpr(CUfunction_attribute attribute) : attribute_(attribute) {}

    ExprPtr resolve(const EvalContext& eval) const override final {
        Primitive out;
        return eval.try_lookup(*this, out) ? ExprPtr::make<ScalarExpr>(out)
                                           : ExprPtr::make<KernelAttributeExpr>(*this);
    }

    std::string to_string() const override {
        switch (attribute_) {
#define KERNEL_LAUNCHER_IMPL(name) \
    case CU_FUNC_ATTRIBUTE_##name: \
        return "$kernel_attribute_" #name;

            KERNEL_LAUNCHER_FOR_EACH_KERNEL_ATTRIBUTE(KERNEL_LAUNCHER_IMPL)
#undef KERNEL_LAUNCHER_IMPL

            default:
                return "$kernel_attribute_" + std::to_string(static_cast<int>(attribute_));
        }
    }

    CUfunction_attribute attribute() const noexcept {
        return attribute_;
    }

  private:
    CUfunction_attribute attribute_;
};

inline KernelAttributeExpr kernel_attribute(CUfunction_attribute attribute) {
    return KernelAttributeExpr(attribute);
}

inline ProblemSizeExpr problem_size_x() {
    return ProblemSizeExpr(0);
}

inline ProblemSizeExpr problem_size_y() {
    return ProblemSizeExpr(1);
}

inline ProblemSizeExpr problem_size_z() {
    return ProblemSizeExpr(2);
}

enum struct UnaryOp { Neg, Not, Round, Floor, Ceil, ToBool, ToInt, ToFloat, ToString };

class UnaryExpr: public ExprBase<Primitive> {
  public:
    UnaryExpr(UnaryOp op, ExprPtr operand) : op_(op), operand_(std::move(operand)) {}

    std::string to_string() const override {
        if (op_ == UnaryOp::Neg || op_ == UnaryOp::Not) {
            return "(" + std::string(op_name()) + operand_.to_string() + ")";
        }

        return std::string(op_name()) + "(" + operand_.to_string() + ")";
    }

    const char* op_name() const {
        switch (op_) {
            case UnaryOp::Neg:
                return "-";
            case UnaryOp::Not:
                return "!";
            case UnaryOp::Round:
                return "round";
            case UnaryOp::Floor:
                return "floor";
            case UnaryOp::Ceil:
                return "ceil";
            case UnaryOp::ToBool:
                return "bool";
            case UnaryOp::ToInt:
                return "int";
            case UnaryOp::ToFloat:
                return "float";
            case UnaryOp::ToString:
                return "str";
            default:
                return "???";
        }
    }

    const ExprPtr& operand() const {
        return operand_;
    }

    Primitive eval(const EvalContext& context) const override {
        Primitive v = operand_.eval(context);

        switch (op_) {
            case UnaryOp::Neg:
                return -v;
            case UnaryOp::Not:
                return Primitive(!v.to_bool());
            case UnaryOp::Round:
                return Primitive(v.round());
            case UnaryOp::Floor:
                return Primitive(v.floor());
            case UnaryOp::Ceil:
                return Primitive(v.ceil());
            case UnaryOp::ToBool:
                return Primitive(v.to_bool());
            case UnaryOp::ToInt:
                return Primitive(v.to_integer());
            case UnaryOp::ToFloat:
                return Primitive(v.to_float());
            case UnaryOp::ToString:
                return Primitive(v.to_string());
            default:
                throw std::runtime_error("invalid unary operator");
        }
    }

    ExprPtr resolve(const EvalContext& context) const override {
        ExprPtr operand = operand_.resolve(context);
        ExprPtr result = ExprPtr::make<UnaryExpr>(op_, operand);

        if (operand.is_constant()) {
            return ExprPtr::make<ScalarExpr>(result.eval(context));
        }

        return result;
    }

    bool is_constant() const override {
        return operand_.is_constant();
    }

  private:
    UnaryOp op_;
    ExprPtr operand_;
};

#define KERNEL_LAUNCHER_EXPR_UNARY_OP_IMPL(op, name)                                         \
    template<typename A, typename = typename std::enable_if<is_expr<A>::value>::type>        \
    ExprPtr operator op(A&& operand) {                                                       \
        return ExprPtr::make<UnaryExpr>(UnaryOp::name, into_expr(std::forward<A>(operand))); \
    }

KERNEL_LAUNCHER_EXPR_UNARY_OP_IMPL(-, Neg)
KERNEL_LAUNCHER_EXPR_UNARY_OP_IMPL(!, Not)

#undef KERNEL_LAUNCHER_EXPR_UNARY_OP_IMPL

#define KERNEL_LAUNCHER_EXPR_UNARY_FUN_IMPL(fun, name)                                       \
    template<typename A, typename = typename std::enable_if<is_expr<A>::value>::type>        \
    ExprPtr fun(A&& operand) {                                                               \
        return ExprPtr::make<UnaryExpr>(UnaryOp::name, into_expr(std::forward<A>(operand))); \
    }

KERNEL_LAUNCHER_EXPR_UNARY_FUN_IMPL(round, Round)
KERNEL_LAUNCHER_EXPR_UNARY_FUN_IMPL(floor, Floor)
KERNEL_LAUNCHER_EXPR_UNARY_FUN_IMPL(ceil, Ceil)
KERNEL_LAUNCHER_EXPR_UNARY_FUN_IMPL(to_bool, ToBool)
KERNEL_LAUNCHER_EXPR_UNARY_FUN_IMPL(to_int, ToInt)
KERNEL_LAUNCHER_EXPR_UNARY_FUN_IMPL(to_float, ToFloat)
KERNEL_LAUNCHER_EXPR_UNARY_FUN_IMPL(to_string, ToString)

#undef KERNEL_LAUNCHER_EXPR_UNARY_FUN_IMPL

enum struct BinaryOp { Add, Sub, Mul, Div, Mod, LogicalAnd, LogicalOr, Eq, Neq, Lt, Lte, Gt, Gte };

class BinaryExpr: public ExprBase<Primitive> {
  public:
    BinaryExpr(BinaryOp op, ExprPtr left, ExprPtr right) :
        op_(op),
        left_(std::move(left)),
        right_(std::move(right)) {}

    std::string to_string() const override {
        return "(" + left_.to_string() + " " + op_name() + " " + right_.to_string() + ")";
    }

    const char* op_name() const {
        switch (op_) {
            case BinaryOp::Add:
                return "+";
            case BinaryOp::Sub:
                return "-";
            case BinaryOp::Mul:
                return "*";
            case BinaryOp::Div:
                return "/";
            case BinaryOp::Mod:
                return "%";
            case BinaryOp::LogicalAnd:
                return "&&";
            case BinaryOp::LogicalOr:
                return "||";
            case BinaryOp::Eq:
                return "==";
            case BinaryOp::Neq:
                return "!=";
            case BinaryOp::Lt:
                return "<";
            case BinaryOp::Lte:
                return "<=";
            case BinaryOp::Gt:
                return ">";
            case BinaryOp::Gte:
                return ">=";
            default:
                return "???";
        }
    }

    const ExprPtr& left_operand() const {
        return left_;
    }

    const ExprPtr& right_operand() const {
        return right_;
    }

    Primitive eval(const EvalContext& context) const override {
        Primitive l = left_.eval(context);

        // The right operand is only evaluated if the left operand does not decide the result.
        if (op_ == BinaryOp::LogicalAnd) {
            return Primitive(l.to_bool() && right_.eval(context).to_bool());
        }

        if (op_ == BinaryOp::LogicalOr) {
            return Primitive(l.to_bool() || right_.eval(context).to_bool());
        }

        Primitive r = right_.eval(context);

        switch (op_) {
            case BinaryOp::Add:
                return l + r;
            case BinaryOp::Sub:
                return l - r;
            case BinaryOp::Mul:
                return l * r;
            case BinaryOp::Div:
                return l / r;
            case BinaryOp::Mod:
                return l % r;
            case BinaryOp::Eq:
                return Primitive(l == r);
            case BinaryOp::Neq:
                return Primitive(l != r);
            case BinaryOp::Lt:
                return Primitive(l < r);
            case BinaryOp::Lte:
                return Primitive(l <= r);
            case BinaryOp::Gt:
                return Primitive(l > r);
            case BinaryOp::Gte:
                return Primitive(l >= r);
            default:
                throw std::runtime_error("invalid binary operator");
        }
    }

    ExprPtr resolve(const EvalContext& context) const override {
        ExprPtr left = left_.resolve(context);

        // If the left operand decides the result, the right operand is not needed.
        if (left.is_constant() && (op_ == BinaryOp::LogicalAnd || op_ == BinaryOp::LogicalOr)) {
            bool l = left.eval(context).to_bool();

            if (l == (op_ == BinaryOp::LogicalOr)) {
                return ExprPtr::make<ScalarExpr>(Primitive(l));
            }
        }

        ExprPtr right = right_.resolve(context);
        ExprPtr result = ExprPtr::make<BinaryExpr>(op_, left, right);

        if (left.is_constant() && right.is_constant()) {
            return ExprPtr::make<ScalarExpr>(result.eval(context));
        }

        return result;
    }

    bool is_constant() const override {
        return left_.is_constant() && right_.is_constant();
    }

  private:
    BinaryOp op_;
    ExprPtr left_;
    ExprPtr right_;
};

#define KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(op, name)                                        \
    template<                                                                             \
        typename A,                                                                       \
        typename B,                                                                       \
        typename = typename std::enable_if<is_expr<A>::value || is_expr<B>::value>::type> \
    ExprPtr operator op(A&& left, B&& right) {                                            \
        return ExprPtr::make<BinaryExpr>(                                                 \
            BinaryOp::name,                                                               \
            into_expr(std::forward<A>(left)),                                             \
            into_expr(std::forward<B>(right)));                                           \
    }

KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(+, Add)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(-, Sub)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(*, Mul)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(/, Div)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(%, Mod)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(&&, LogicalAnd)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(||, LogicalOr)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(==, Eq)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(!=, Neq)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(<, Lt)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(<=, Lte)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(>, Gt)
KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL(>=, Gte)

#undef KERNEL_LAUNCHER_EXPR_BIN_OP_IMPL

template<
    typename A,
    typename B,
    typename = typename std::enable_if<is_expr<A>::value || is_expr<B>::value>::type>
ExprPtr div_ceil(A&& numerator, B&& denominator) {
    ExprPtr n = into_expr(std::forward<A>(numerator));
    ExprPtr d = into_expr(std::forward<B>(denominator));
    ExprPtr r = n % d;

    // `(n - r) / d` is the quotient rounded towards zero. This is one below the ceiling if
    // there is a remainder and the exact quotient is positive (`r` and `d` have the same sign).
    return (n - r) / d + (r != 0 && (r < 0) == (d < 0));
}

class SelectExpr: public ExprBase<Primitive> {
  public:
    SelectExpr(ExprPtr index, std::vector<ExprPtr> options) :
        index_(std::move(index)),
        options_(std::move(options)) {}

    std::string to_string() const override {
        std::string result = "select(" + index_.to_string();

        for (const auto& option : options_) {
            result += ", " + option.to_string();
        }

        return result + ")";
    }

    const ExprPtr& index() const {
        return index_;
    }

    const std::vector<ExprPtr>& options() const {
        return options_;
    }

    Primitive eval(const EvalContext& context) const override {
        std::intmax_t index = index_.eval(context).to_integer();

        if (index < 0 || size_t(index) >= options_.size()) {
            throw std::runtime_error("SelectExpr: index out of bounds");
        }

        return options_[size_t(index)].eval(context);
    }

    ExprPtr resolve(const EvalContext& context) const override {
        ExprPtr index = index_.resolve(context);

        // If the index is known, only the selected option remains.
        if (index.is_constant()) {
            std::intmax_t i = index.eval(context).to_integer();

            if (i >= 0 && size_t(i) < options_.size()) {
                return options_[size_t(i)].resolve(context);
            }
        }

        std::vector<ExprPtr> options;
        options.reserve(options_.size());

        for (const auto& option : options_) {
            options.push_back(option.resolve(context));
        }

        return ExprPtr::make<SelectExpr>(std::move(index), std::move(options));
    }

    bool is_constant() const override {
        if (!index_.is_constant()) {
            return false;
        }

        for (const auto& option : options_) {
            if (!option.is_constant()) {
                return false;
            }
        }

        return true;
    }

  private:
    ExprPtr index_;
    std::vector<ExprPtr> options_;
};

template<
    typename C,
    typename... Es,
    typename = typename std::enable_if<std::disjunction<is_expr<C>, is_expr<Es>...>::value>::type>
SelectExpr select(C&& index, Es&&... options) {
    return SelectExpr(
        into_expr(std::forward<C>(index)),
        std::vector<ExprPtr> {into_expr(std::forward<Es>(options))...});
}

template<typename C, typename Es, typename = typename std::enable_if<is_expr<C>::value>::type>
SelectExpr index(C&& index, Es&& options) {
    std::vector<ExprPtr> result;

    for (auto&& option : options) {
        result.push_back(into_expr(option));
    }

    return SelectExpr(into_expr(std::forward<C>(index)), std::move(result));
}

template<
    typename C,
    typename ET,
    typename EF,
    typename = typename std::enable_if<
        std::disjunction<is_expr<C>, is_expr<ET>, is_expr<EF>>::value>::type>
SelectExpr ifelse(C&& cond, ET&& true_expr, EF&& false_expr) {
    // The condition is used as an index, so it must evaluate to a boolean or to 0 or 1.
    return SelectExpr(
        into_expr(std::forward<C>(cond)),
        std::vector<ExprPtr> {
            into_expr(std::forward<EF>(false_expr)),
            into_expr(std::forward<ET>(true_expr))});
}

template<
    typename A,
    typename B,
    typename... Rest,
    typename = typename std::enable_if<
        std::disjunction<is_expr<A>, is_expr<B>, is_expr<Rest>...>::value>::type>
SelectExpr min(A&& left, B&& right, Rest&&... rest) {
    ExprPtr l = into_expr(std::forward<A>(left));
    ExprPtr r = into_expr(std::forward<B>(right));

    if constexpr (sizeof...(Rest) == 0) {
        return ifelse(r < l, r, l);
    } else {
        return min(min(l, r), std::forward<Rest>(rest)...);
    }
}

template<
    typename A,
    typename B,
    typename... Rest,
    typename = typename std::enable_if<
        std::disjunction<is_expr<A>, is_expr<B>, is_expr<Rest>...>::value>::type>
SelectExpr max(A&& left, B&& right, Rest&&... rest) {
    ExprPtr l = into_expr(std::forward<A>(left));
    ExprPtr r = into_expr(std::forward<B>(right));

    if constexpr (sizeof...(Rest) == 0) {
        return ifelse(l < r, r, l);
    } else {
        return max(max(l, r), std::forward<Rest>(rest)...);
    }
}

template<typename T>
class Expr {
  public:
    Expr(ExprPtr ptr) : ptr_(std::move(ptr)) {}

    Expr() : ptr_(ExprPtr::make<ScalarExpr>(T {})) {}

    template<typename U>
    Expr(U value) : ptr_(into_expr(std::move(value))) {}

    std::string to_string() const {
        return ptr_.to_string();
    }

    T eval(const EvalContext& eval) const {
        try {
            return ptr_.eval(eval).template to<T>();
        } catch (const std::exception& e) {
            throw std::runtime_error(
                concat("failed to evaluate expression '", to_string(), "': ", e.what()));
        }
    }

    Expr<T> resolve(const EvalContext& eval) const {
        return Expr<T>(ptr_.resolve(eval));
    }

    bool is_constant() const {
        return ptr_.is_constant();
    }

    const ExprPtr& inner() const {
        return ptr_;
    }

  private:
    ExprPtr ptr_;
};

/** @} */

}  // namespace kernel_launcher

#endif
