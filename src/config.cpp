#include "kernel_launcher/config.hpp"

#include <ostream>
#include <stdexcept>

namespace kernel_launcher {

bool TunableParam::has_value(const Primitive& value) const {
    for (const auto& v : impl_->values) {
        if (v == value) {
            return true;
        }
    }

    return false;
}

const Primitive& Config::get(std::string_view name) const {
    // Compare by content, since interning `name` would keep it alive forever.
    for (const auto& entry : values_) {
        if (entry.first.get() == name) {
            return entry.second;
        }
    }

    throw std::runtime_error("unknown parameter: " + std::string(name));
}

bool Config::try_lookup(const Variable& v, Primitive& out) const {
    if (const auto* param = dynamic_cast<const TunableParam*>(&v)) {
        auto it = values_.find(param->name());
        if (it != values_.end()) {
            out = it->second;
            return true;
        }
    }

    return false;
}

std::ostream& operator<<(std::ostream& stream, const Config& config) {
    bool first = true;
    stream << "{";

    for (const auto& entry : config.values_) {
        if (!first) {
            stream << ", ";
        }

        first = false;
        stream << entry.first << "=" << entry.second;
    }

    return stream << "}";
}

TunableParam
ConfigSpace::tune(std::string_view name, std::vector<Primitive> values, Primitive default_value) {
    if (name.empty()) {
        throw std::invalid_argument("parameter name cannot be empty");
    }

    if (values.empty()) {
        throw std::invalid_argument(
            concat("parameter \"", name, "\" must have at least one value"));
    }

    for (const auto& p : params_) {
        if (p.name().get() == name) {
            throw std::invalid_argument(
                concat("parameter \"", name, "\" has already been defined"));
        }
    }

    auto impl = std::make_shared<TunableParam::Impl>();
    impl->name = Symbol(name);
    impl->values = std::move(values);
    impl->default_value = std::move(default_value);

    TunableParam param(std::move(impl));

    if (!param.has_value(param.default_value())) {
        throw std::invalid_argument(
            concat("default value of parameter \"", name, "\" is not one of its values"));
    }

    params_.push_back(param);
    return param;
}

const TunableParam& ConfigSpace::at(std::string_view name) const {
    for (const auto& p : params_) {
        if (p.name().get() == name) {
            return p;
        }
    }

    throw std::runtime_error("unknown parameter: " + std::string(name));
}

ConfigSpace ConfigSpace::resolve(const EvalContext& context) const {
    ConfigSpace result;
    result.params_ = params_;

    for (const auto& r : restrictions_) {
        Expr<bool> resolved = r.resolve(context);

        if (!resolved.is_constant()) {
            result.restrictions_.push_back(std::move(resolved));
        } else if (!resolved.eval(context)) {
            throw std::runtime_error("restriction is never satisfied: " + r.to_string());
        }
    }

    return result;
}

Config ConfigSpace::default_config() const {
    Config config;

    for (const auto& p : params_) {
        config.insert(p.name(), p.default_value());
    }

    return config;
}

bool ConfigSpace::contains(const Config& config) const {
    // Config must not have parameters that are not in this space.
    if (config.size() != params_.size()) {
        return false;
    }

    for (const auto& p : params_) {
        Primitive value;

        // Parameter must be present in the config with a valid value.
        if (!config.try_lookup(p, value) || !p.has_value(value)) {
            return false;
        }
    }

    return true;
}

bool ConfigSpace::is_valid(const Config& config) const {
    if (!contains(config)) {
        return false;
    }

    for (const auto& r : restrictions_) {
        if (!r.eval(config)) {
            return false;
        }
    }

    return true;
}

std::vector<Config> ConfigSpace::valid_configs() const {
    std::vector<Config> result;

    // Indices of cartesian product of all parameter values.
    std::vector<size_t> index(params_.size(), 0);

    while (true) {
        Config config;
        for (size_t i = 0; i < params_.size(); i++) {
            config.insert(params_[i].name(), params_[i].values()[index[i]]);
        }

        if (is_valid(config)) {
            result.push_back(std::move(config));
        }

        // Advance to the next combination; stop once every digit wrapped.
        size_t i = params_.size();
        while (i > 0) {
            i--;

            if (++index[i] < params_[i].values().size()) {
                break;
            }

            index[i] = 0;

            if (i == 0) {
                return result;
            }
        }

        if (params_.empty()) {
            return result;
        }
    }
}

}  // namespace kernel_launcher
