#include "kernel_launcher/config.h"

namespace kernel_launcher {

void Config::insert(TunableParam k, TunableValue v) {
    const std::string& name = k.name();

    for (const auto& it : inner_) {
        if (it.first.name() == name) {
            throw std::runtime_error(
                "duplicate parameter: key " + name + " already exists");
        }
    }

    inner_.insert({std::move(k), std::move(v)});
}

const TunableValue& Config::at(const std::string& param) const {
    for (const auto& it : inner_) {
        if (it.first.name() == param) {
            return it.second;
        }
    }

    throw std::invalid_argument("unknown parameter: " + param);
}

const TunableValue& Config::at(const TunableParam& param) const {
    auto it = inner_.find(param);
    if (it == inner_.end()) {
        throw std::invalid_argument("unknown parameter: " + param.name());
    }

    return it->second;
}

std::ostream& operator<<(std::ostream& s, const Config& c) {
    s << "{";
    bool is_first = true;
    for (const auto& p : c.inner_) {
        if (is_first) {
            is_first = false;
        } else {
            s << ", ";
        }

        s << "\"" << p.first.name() << "\": " << p.second;
    }
    return s << "}";
}

TunableParam ConfigSpace::add(
    std::string name,
    std::vector<TunableValue> values,
    std::vector<double> priors,
    TunableValue default_value) {
    for (const auto& p : params_) {
        if (p.name() == name) {
            throw std::runtime_error(
                "duplicate parameter: key " + name + " already exists");
        }
    }

    TunableParam p {
        std::move(name),
        std::move(values),
        std::move(priors),
        std::move(default_value)};
    params_.push_back(p);
    return p;
}

ParamExpr ConfigSpace::at(const std::string& name) const {
    for (const auto& p : params_) {
        if (p.name() == name) {
            return p;
        }
    }

    throw std::runtime_error("could not find key: " + name);
}

void ConfigSpace::restriction(TypedExpr<bool> e) {
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

        if (!p.has_value(it->second)) {
            return false;
        }
    }

    Eval eval = {config.get()};

    for (const auto& r : restrictions_) {
        if (!eval(r)) {
            return false;
        }
    }

    return true;
}
}  // namespace kernel_launcher