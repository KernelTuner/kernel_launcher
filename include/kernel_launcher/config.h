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

    template<typename T, typename It>
    ParamExpr tune(std::string name, It begin, It end, T default_value) {
        std::vector<TunableValue> values;
        for (It current = begin; current != end; ++current) {
            T value = *current;
            values.push_back(std::move(value));
        }

        return add(std::move(name), std::move(values), default_value);
    }

    template<typename Collection, typename T>
    ParamExpr
    tune(std::string name, const Collection& values, T default_value) {
        return tune(std::move(name), begin(values), end(values), default_value);
    }

    template<typename Collection, typename T = typename Collection::value_type>
    ParamExpr tune(std::string name, const Collection& values) {
        if (values.size() == 0) {
            throw std::invalid_argument("empty list of values");
        }

        return tune(
            std::move(name),
            begin(values),
            end(values),
            *values.begin());
    }

    template<typename T>
    ParamExpr
    tune(std::string name, std::initializer_list<T> values, T default_value) {
        return tune(
            std::move(name),
            values.begin(),
            values.end(),
            std::move(default_value));
    }

    template<typename T>
    ParamExpr tune(std::string name, std::initializer_list<T> values) {
        return tune(std::move(name), std::vector<T> {values});
    }

    ParamExpr operator[](const std::string& name) const {
        return at(name);
    }

    TunableParam
    add(std::string name,
        std::vector<TunableValue> values,
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
