#ifndef KERNEL_LAUNCHER_CONFIG_H
#define KERNEL_LAUNCHER_CONFIG_H

#include <unordered_map>

#include "kernel_launcher/expr.h"
#include "kernel_launcher/value.h"

namespace kernel_launcher {

struct ConfigSpace;
struct Config {
    Config() = default;
    Config(Config&&) = default;
    explicit Config(const Config&) = default;
    Config& operator=(Config&&) = default;
    Config& operator=(const Config&) = delete;

    const TunableValue& at(const std::string& param) const;
    const TunableValue& at(const TunableParam& param) const;
    void insert(TunableParam k, TunableValue v);

    const TunableValue& operator[](const std::string& name) const {
        return at(name);
    }

    const TunableValue& operator[](const TunableParam& param) const {
        return at(param);
    }

    const TunableValue& operator[](const ParamExpr& param) const {
        return at(param.parameter());
    }

    void insert(const ParamExpr& k, TunableValue v) {
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

  private:
    std::unordered_map<TunableParam, TunableValue> inner_;
};

struct KernelBuilderSerializerHack;

struct ConfigSpace {
    friend ::kernel_launcher::KernelBuilderSerializerHack;

    ConfigSpace() = default;
    ConfigSpace(ConfigSpace&&) = default;
    explicit ConfigSpace(const ConfigSpace&) = default;
    ConfigSpace& operator=(ConfigSpace&&) = default;
    ConfigSpace& operator=(const ConfigSpace&) = delete;

    template<typename T, typename P>
    ParamExpr tune(
        std::string name,
        std::vector<T> values,
        std::vector<P> priors,
        T default_value) {
        std::vector<TunableValue> tvalues;
        for (const T& v : values) {
            tvalues.push_back(v);
        }

        std::vector<double> tpriors;
        for (const P& v : priors) {
            tpriors.push_back(v);
        }

        return add(
            std::move(name),
            std::move(tvalues),
            std::move(tpriors),
            std::move(default_value));
    }

    ParamExpr tune(
        std::string name,
        std::initializer_list<TunableValue> values,
        std::initializer_list<double> priors,
        TunableValue default_value) {
        return tune(
            std::move(name),
            std::vector<TunableValue>(values),
            std::vector<double>(priors),
            std::move(default_value));
    }

    template<typename T, typename P>
    ParamExpr
    tune(std::string name, std::vector<T> values, std::vector<P> priors) {
        T default_value = values.at(0);
        return tune(
            std::move(name),
            std::move(values),
            std::move(priors),
            default_value);
    }

    ParamExpr tune(
        std::string name,
        std::initializer_list<TunableValue> values,
        std::initializer_list<double> priors) {
        return tune(
            std::move(name),
            std::vector<TunableValue>(values),
            std::vector<double>(priors));
    }

    template<typename T>
    ParamExpr tune(std::string name, std::vector<T> values, T default_value) {
        std::vector<double> priors(values.size(), 1.0);
        return tune(
            std::move(name),
            std::move(values),
            std::move(priors),
            std::move(default_value));
    }

    template<typename T>
    ParamExpr tune(
        std::string name,
        std::initializer_list<TunableValue> values,
        TunableValue default_value) {
        return tune(
            std::move(name),
            std::vector<T> {values},
            std::move(default_value));
    }

    template<typename T>
    ParamExpr tune(std::string name, std::vector<T> values) {
        std::vector<double> priors(values.size(), 1.0);
        return tune(std::move(name), std::move(values), std::move(priors));
    }

    ParamExpr
    tune(std::string name, std::initializer_list<TunableValue> values) {
        return tune(std::move(name), std::vector<TunableValue> {values});
    }

    ParamExpr operator[](const std::string& name) const {
        return at(name);
    }

    const std::vector<TunableParam>& parameters() const {
        return params_;
    }

    TunableParam
    add(std::string name,
        std::vector<TunableValue> values,
        std::vector<double> priors,
        TunableValue default_value);
    ParamExpr at(const std::string& name) const;
    void restriction(TypedExpr<bool> e);
    Config default_config() const;
    bool is_valid(const Config& config) const;

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
            size_t k = std::hash<kernel_launcher::TunableParam> {}(it.first);
            result = kernel_launcher::hash_combine(result, k);

            size_t v = std::hash<kernel_launcher::TunableValue> {}(it.second);
            result = kernel_launcher::hash_combine(result, v);
        }

        return result;
    }
};
}  // namespace std

#endif  //KERNEL_LAUNCHER_CONFIG_H
