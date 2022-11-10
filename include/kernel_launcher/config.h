#ifndef KERNEL_LAUNCHER_CONFIG_H
#define KERNEL_LAUNCHER_CONFIG_H

#include <unordered_map>

#include "kernel_launcher/expr.h"
#include "kernel_launcher/value.h"

namespace kernel_launcher {

using TunableMap = std::unordered_map<TunableParam, Value>;

struct ConfigSpace;
struct Config: Eval {
    Config() = default;
    Config(Config&&) = default;
    explicit Config(const Config&) = default;
    Config& operator=(Config&&) = default;
    Config& operator=(const Config&) = delete;

    bool lookup(const Variable& v, Value& out) const override;

    const Value& at(const std::string& param) const;
    const Value& at(const TunableParam& param) const;
    void insert(TunableParam k, Value v);

    const Value& operator[](const std::string& name) const {
        return at(name);
    }

    const Value& operator[](const TunableParam& param) const {
        return at(param);
    }

    const Value& operator[](const ParamExpr& param) const {
        return at(param.parameter());
    }

    void insert(const ParamExpr& k, Value v) {
        insert(k.parameter(), std::move(v));
    }

    size_t size() const {
        return inner_.size();
    }
    bool operator==(const Config& that) const {
        return inner_ == that.inner_;
    }

    bool operator!=(const Config& that) const {
        return !operator==(that);
    }

    const TunableMap& get() const {
        return inner_;
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

    template<typename T = Value>
    ParamExpr tune(std::string name, std::vector<T> values) {
        std::vector<double> priors(values.size(), 1.0);
        return tune(std::move(name), std::move(values), std::move(priors));
    }

    ParamExpr operator[](const std::string& name) const {
        return at(name);
    }

    const std::vector<TunableParam>& parameters() const {
        return params_;
    }

    TunableParam
    add(std::string name,
        std::vector<Value> values,
        std::vector<double> priors,
        Value default_value);
    ParamExpr at(const std::string& name) const;
    void restriction(TypedExpr<bool> e);
    Config default_config() const;
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

        for (const auto& it : config.get()) {
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
