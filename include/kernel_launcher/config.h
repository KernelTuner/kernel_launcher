#ifndef KERNEL_LAUNCHER_CONFIG_H
#define KERNEL_LAUNCHER_CONFIG_H

#include <unordered_map>

#include "kernel_launcher/expr.h"
#include "kernel_launcher/value.h"

namespace kernel_launcher {

using TunableMap = std::unordered_map<TunableParam, Value>;

struct ConfigSpace;

/**
 * A particular configuration from a `ConfigSpace`. This class is essentially
 * a lookup table that maps `TunableParam`s to `Value`s.
 */
struct Config: Eval {
    using const_iterator = typename TunableMap::const_iterator;

    Config() = default;
    Config(Config&&) = default;
    explicit Config(const Config&) = default;
    Config& operator=(Config&&) = default;
    Config& operator=(const Config&) = delete;

    bool lookup(const Variable& v, Value& out) const override;

    const Value& at(const std::string& param) const;
    const Value& at(const TunableParam& param) const;

    /**
     * Insert a `TunableParam` with the associated `Value`.
     */
    void insert(TunableParam k, Value v);

    void insert(const ParamExpr& k, Value v) {
        insert(k.parameter(), std::move(v));
    }

    /**
     * Returns the value that corresponds to the parameter with the given name.
     */
    const Value& operator[](const std::string& name) const {
        return at(name);
    }

    /**
     * Returns the value that corresponds to the given parameter.
     */
    const Value& operator[](const TunableParam& param) const {
        return at(param);
    }

    /**
     * Returns the value that corresponds to the given parameter.
     */
    const Value& operator[](const ParamExpr& param) const {
        return at(param.parameter());
    }

    /**
     * Returns the number of parameters in this configuration.
     */
    size_t size() const {
        return inner_.size();
    }

    const_iterator begin() const {
        return inner_.begin();
    }

    const_iterator end() const {
        return inner_.end();
    }

    bool operator==(const Config& that) const {
        return inner_ == that.inner_;
    }

    bool operator!=(const Config& that) const {
        return !operator==(that);
    }

    friend std::ostream& operator<<(std::ostream&, const Config& c);

  private:
    std::unordered_map<TunableParam, Value> inner_;
};

struct KernelBuilderSerializerHack;

struct ConfigSpace {
    friend ::kernel_launcher::KernelBuilderSerializerHack;

    ConfigSpace() = default;
    ConfigSpace(ConfigSpace&&) = default;
    explicit ConfigSpace(const ConfigSpace&) = default;
    ConfigSpace& operator=(ConfigSpace&&) = default;
    ConfigSpace& operator=(const ConfigSpace&) = delete;

    /**
     * Add a new parameter to the configuration space.
     *
     * @param name The name of the parameter.
     * @param values The values of this parameter.
     * @param priors The prior probabilities of the given values.
     * @param default_value The default value that will be return by
     *                      `ConfigSpace::default_config()`.
     */
    template<typename T = Value, typename P = double>
    ParamExpr tune(
        std::string name,
        std::vector<T> values,
        std::vector<P> priors,
        Value default_value) {
        std::vector<Value> tvalues {values.begin(), values.end()};
        std::vector<double> tpriors {priors.begin(), priors.end()};

        return add(
            std::move(name),
            std::move(tvalues),
            std::move(tpriors),
            std::move(default_value));
    }

    /**
     * Add a new parameter to the configuration space. The default value is
     * assume to be the first value in the list.
     *
     * @param name The name of the parameter.
     * @param values The values of this parameter.
     * @param priors The prior probabilities of the given values.
     */
    template<typename T = Value, typename P = double>
    ParamExpr
    tune(std::string name, std::vector<T> values, std::vector<P> priors) {
        T default_value = values.at(0);
        return tune(
            std::move(name),
            std::move(values),
            std::move(priors),
            default_value);
    }

    /**
     * Add a new parameter to the configuration space.
     *
     * @param name The name of the parameter.
     * @param values The values of this parameter.
     * @param default_value The default value that will be return by
     *                      `ConfigSpace::default_config()`.
     */
    template<typename T = Value>
    ParamExpr
    tune(std::string name, std::vector<T> values, Value default_value) {
        std::vector<double> priors(values.size(), 1.0);
        return tune(
            std::move(name),
            std::move(values),
            std::move(priors),
            std::move(default_value));
    }

    /**
     * Add a new parameter to the configuration space. The default value is
     * assume to be the first value in the list.
     *
     * @param name The name of the parameter.
     * @param values The values of this parameter.
     */
    template<typename T = Value>
    ParamExpr tune(std::string name, std::vector<T> values) {
        std::vector<double> priors(values.size(), 1.0);
        return tune(std::move(name), std::move(values), std::move(priors));
    }

    /**
     * Returns the parameter expression with the given name.
     */
    ParamExpr operator[](const std::string& name) const {
        return at(name);
    }

    /**
     * Returns the parameters of this configuration space.
     */
    const std::vector<TunableParam>& parameters() const {
        return params_;
    }

    TunableParam
    add(std::string name,
        std::vector<Value> values,
        std::vector<double> priors,
        Value default_value);
    ParamExpr at(const std::string& name) const;

    /**
     * Add a restriction to this configuration space. A configuration is
     * only considered valid if this expression yields `true`.
     */
    void restriction(TypedExpr<bool> e);

    /**
     * Returns the default configuration for this configuration space.
     */
    Config default_config() const;

    /**
     *  Check if the given configuration is a valid member of this configuration
     *  space. This method essentially checks three things:
     *
     *  * Does the configuration contain the correct parameters.
     *  * Do these parameter contain valid values.
     *  * Does the configuration meet the restrictions.
     */
    bool is_valid(const Eval& config) const;

  private:
    std::vector<TunableParam> params_;
    std::vector<TypedExpr<bool>> restrictions_;
};

}  // namespace kernel_launcher

namespace std {
template<>
struct hash<kernel_launcher::Config> {
    size_t operator()(const kernel_launcher::Config& config) const {
        size_t result = 0;

        for (const auto& it : config) {
            size_t k = std::hash<kernel_launcher::Variable> {}(it.first);
            result = kernel_launcher::hash_combine(result, k);

            size_t v = std::hash<kernel_launcher::Value> {}(it.second);
            result = kernel_launcher::hash_combine(result, v);
        }

        return result;
    }
};
}  // namespace std

#endif  //KERNEL_LAUNCHER_CONFIG_H
