#ifndef KERNEL_LAUNCHER_CONFIG_H
#define KERNEL_LAUNCHER_CONFIG_H

#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "kernel_launcher/expr.hpp"

namespace kernel_launcher {

/**
 * @addtogroup builder
 * @{
 */

/**
 * Represents a single tunable parameter. A parameter has a name, a set of possible values,
 * and a default value.
 */
class TunableParam: public Variable {
  public:
    // Can only be created via `ConfigSpace`.
    TunableParam() = delete;

    Symbol name() const {
        return impl_->name;
    }

    const std::vector<Primitive>& values() const {
        return impl_->values;
    }

    const Primitive& default_value() const {
        return impl_->default_value;
    }

    /**
     * Returns `true` if `value==x` for any `x` in `values()`.
     */
    bool has_value(const Primitive& value) const;

    ExprPtr resolve(const EvalContext& eval) const override final {
        Primitive out;
        return eval.try_lookup(*this, out) ? ExprPtr::make<ScalarExpr>(out)
                                           : ExprPtr::make<TunableParam>(*this);
    }

    std::string to_string() const override final {
        return std::string(name());
    }

  private:
    friend class ConfigSpace;

    struct Impl {
        Symbol name;
        std::vector<Primitive> values;
        Primitive default_value;
    };

    explicit TunableParam(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

    // Invariant: never null.
    std::shared_ptr<const Impl> impl_;
};

/**
 * A particular configuration from a `ConfigSpace`. This class basically maps
 * each `TunableParam` to a corresponding `Primitive`.
 */
class Config: public EvalContext {
  public:
    bool try_lookup(const Variable& v, Primitive& out) const override;

    /**
     * Returns the value of the parameter with the given name. Throws `std::runtime_error`
     * if this configuration has no such parameter.
     */
    const Primitive& get(std::string_view name) const;

    const Primitive& operator[](std::string_view name) const {
        return get(name);
    }

    /**
     * Sets the value of the parameter with the given name. An existing value for
     * that name is replaced.
     */
    void insert(Symbol name, Primitive value) {
        values_.insert_or_assign(name, value);
    }

    void insert(std::string_view name, Primitive value) {
        insert(Symbol(name), value);
    }

    size_t size() const {
        return values_.size();
    }

    bool operator==(const Config& that) const {
        return values_ == that.values_;
    }

    bool operator!=(const Config& that) const {
        return !(*this == that);
    }

    size_t hash() const {
        // Combine using XOR ^ to not be dependent on order of fields.
        size_t result = 0;
        for (const auto& entry : values_) {
            result ^= hash_fields(entry.first, entry.second);
        }
        return result;
    }

  private:
    friend std::ostream& operator<<(std::ostream& stream, const Config& config);

    std::unordered_map<Symbol, Primitive> values_;
};

/**
 * Writes the config as `{name=value, ...}`. The order of the parameters is unspecified.
 */
std::ostream& operator<<(std::ostream& stream, const Config& config);

/**
 * A set of `TunableParam`s that together define a search space.
 */
class ConfigSpace {
  public:
    /**
     * Add a new tunable parameter to this configuration space and return it.
     *
     * Throws `std::invalid_argument` if `name` is empty, `values` is empty, a parameter
     * named `name` already exists, or `default_value` is not one of `values`.
     */
    TunableParam
    tune(std::string_view name, std::vector<Primitive> values, Primitive default_value);

    /**
     * Same as `tune(name, values, default_value)`, where the default value is the
     * first entry of `values`.
     */
    TunableParam tune(std::string_view name, std::vector<Primitive> values) {
        Primitive default_value = values.empty() ? Primitive {} : values[0];
        return tune(name, std::move(values), default_value);
    }

    /**
     * Add a restriction to this configuration space. A configuration is
     * only considered valid if this expression yields `true`.
     */
    void restriction(Expr<bool> expr) {
        restrictions_.push_back(std::move(expr));
    }

    /**
     * Returns the parameter with the given name. Throws `std::runtime_error` if this
     * configuration space has no such parameter.
     */
    const TunableParam& at(std::string_view name) const;

    const TunableParam& operator[](std::string_view name) const {
        return at(name);
    }

    /**
     * Return the parameters of this configuration space.
     */
    const std::vector<TunableParam>& parameters() const {
        return params_;
    }

    /**
     * Returns a copy of this search space with the same parameters. However, all restrictions
     * are 'resolved' using the given evaluation context, meaning that free-standing variables
     * are replaced.
     */
    ConfigSpace resolve(const EvalContext& context) const;

    /**
     * Returns the default configuration for this configuration space.
     */
    Config default_config() const;

    /**
     * Check if the given configuration is a member of this configuration space. This essentially
     * checks two things: 1) does the configuration contain the correct parameters, and 2)
     * do these parameters contain values that are part of that parameter.
     *
     * However, it does _not_ check if the configuration satisfies the restrictions.
     */
    bool contains(const Config& config) const;

    /**
     *  Checks `contains(config)`, as well as if the configuration meets all restrictions.
     */
    bool is_valid(const Config& config) const;

    /**
     * Returns the set of all configurations that meet the restrictions.
     */
    std::vector<Config> valid_configs() const;

  private:
    friend struct KernelBuilderSerializerHack;

    std::vector<TunableParam> params_;
    std::vector<Expr<bool>> restrictions_;
};

/** @} */

}  // namespace kernel_launcher

namespace std {
template<>
struct hash<kernel_launcher::Config> {
    size_t operator()(const kernel_launcher::Config& v) const noexcept {
        return v.hash();
    }
};
}  // namespace std

#endif
