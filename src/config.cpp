#include "kernel_launcher/config.h"

namespace kernel_launcher {

const TunableValue& Config::at(const std::string& param) const {
    for (const auto& it : inner_) {
        if (it.first.name() == param) {
            return it.second;
        }
    }

    throw std::invalid_argument("unknown parameter");
}

const TunableValue& Config::at(const TunableParam& param) const {
    auto it = inner_.find(param);

    if (it == inner_.end()) {
        throw std::invalid_argument("unknown parameter");
    }

    return it->second;
}

TunableParam ConfigSpace::add(
    std::string name,
    std::vector<TunableValue> values,
    TunableValue default_value) {
    TunableParam p {
        std::move(name),
        std::move(values),
        std::move(default_value)};
    params_.push_back(p);
    return p;
}

void ConfigSpace::restriction(Expr<bool> e) {
    restrictions_.push_back(e);
}

Config ConfigSpace::default_config() const {
    Config config;

    for (const auto& param : params_) {
        config.insert(param, param.default_value());
    }

    if (!is_valid(config)) {
        throw std::runtime_error("default config does not pass restrictions");
    }

    return config;
}

bool ConfigSpace::is_valid(const Config& config) const {
    TunableMap m = config.get();

    if (m.size() != params_.size()) {
        return false;
    }

    for (const auto& p : params_) {
        auto it = m.find(p);
        if (it == m.end()) {
            return false;
        }

        const TunableParam& key = it->first;
        const TunableValue& value = it->second;
        bool in_list = false;

        for (const auto& allowed_value : p.values()) {
            in_list |= value == allowed_value;
        }

        if (!in_list) {
            return false;
        }
    }

    for (const auto& r : restrictions_) {
        Eval eval = {config.get()};

        if (!r.get(eval)) {
            return false;
        }
    }

    return true;
}

}  // namespace kernel_launcher