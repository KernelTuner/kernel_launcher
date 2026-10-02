#include "kernel_launcher/kernel_builder.hpp"

#include <strings.h>

#include <string_view>

#include "kernel_launcher/kernel.hpp"

namespace kernel_launcher {

static bool iequals(std::string_view name, const char* option) {
    return strncasecmp(name.data(), option, name.size()) == 0 && option[name.size()] == '\0';
}

// Resolves `DeviceAttributeExpr` by querying `device`. Base for `BuildEval`,
// which resolves everything else itself and falls back to this for device
// attributes.
class DeviceEval: public EvalContext {
  public:
    DeviceEval(const GPUDevice& device) : device_(device) {}

    bool try_lookup(const Variable& v, Primitive& out) const override {
        if (const auto* p = dynamic_cast<const DeviceAttributeExpr*>(&v)) {
            out = Primitive(device_.attribute(p->attribute()));
            return true;
        }

        return false;
    }

  private:
    const GPUDevice& device_;
};

// Resolves `ArgumentExpr` from `args`. If `problem_key` is given, also
// resolves `ProblemSizeExpr` and `FeatureExpr`, which is used for the buffer
// sizes.
class ArgEval: public EvalContext {
  public:
    ArgEval(const std::vector<KernelArg>* args, const ProblemKey* problem_key = nullptr) :
        args_(args),
        problem_key_(problem_key) {}

    bool try_lookup(const Variable& v, Primitive& out) const final {
        if (const auto* p = dynamic_cast<const ProblemSizeExpr*>(&v)) {
            if (problem_key_ == nullptr) {
                return false;
            }

            out = Primitive(problem_key_->problem_size[p->axis()]);
            return true;
        }

        if (const auto* p = dynamic_cast<const FeatureExpr*>(&v)) {
            if (problem_key_ == nullptr) {
                return false;
            }

            out = problem_key_->features.get(p->name());
            return true;
        }

        if (const auto* ptr = dynamic_cast<const ArgumentExpr*>(&v)) {
            if (ptr->index() < args_->size()) {
                const KernelArg& arg = (*args_)[ptr->index()];
                out = ptr->get(arg);

                if (out.is_empty()) {
                    throw std::runtime_error(concat(
                        "argument ",
                        ptr->to_string(),
                        " of type '",
                        arg.type().name(),
                        "' cannot be converted to a primitive value"));
                }

                return true;
            }
        }

        return false;
    }

  private:
    const std::vector<KernelArg>* args_;
    const ProblemKey* problem_key_;
};

class BuildEval: public DeviceEval {
  public:
    BuildEval(
        const GPUDevice& device,
        const Config& config,
        const ProblemKey& problem_key,
        const GPUKernel* kernel = nullptr) :
        DeviceEval(device),
        config_(config),
        problem_key_(problem_key),
        kernel_(kernel) {}

    bool try_lookup(const Variable& v, Primitive& out) const final {
        if (const auto* p = dynamic_cast<const ProblemSizeExpr*>(&v)) {
            out = Primitive(problem_key_.problem_size[p->axis()]);
            return true;
        }

        if (const auto* p = dynamic_cast<const FeatureExpr*>(&v)) {
            out = problem_key_.features.get(p->name());
            return true;
        }

        if (const auto* p = dynamic_cast<const KernelAttributeExpr*>(&v)) {
            if (kernel_ == nullptr) {
                return false;
            }

            out = Primitive(kernel_->attribute(p->attribute()));
            return true;
        }

        if (config_.try_lookup(v, out)) {
            return true;
        }

        return DeviceEval::try_lookup(v, out);
    }

  private:
    const Config& config_;
    const ProblemKey& problem_key_;
    const GPUKernel* kernel_;
};

ProblemKey KernelBuilder::build_problem_key(std::vector<KernelArg>& args) const {
    ArgEval context(&args);

    ProblemKey key(
        tuning_key_,
        problem_size_[0].eval(context),
        problem_size_[1].eval(context),
        problem_size_[2].eval(context));

    for (const auto& p : features_) {
        key.features.add(p.first, p.second.eval(context));
    }

    // The buffer sizes are evaluated last, so they can depend on the problem
    // size and features.
    ArgEval buffer_context(&args, &key);

    for (const auto& buf : buffer_sizes_) {
        if (buf.index >= args.size()) {
            throw std::runtime_error(
                "buffer_size: argument index " + std::to_string(buf.index) + " is out of bounds");
        }

        KernelArg& arg = args[buf.index];
        TypeInfo pointee = arg.type().remove_pointer();

        // The length is in elements, and `void` has no element size.
        if (pointee.is_void()) {
            throw std::runtime_error(
                "buffer_size: argument " + std::to_string(buf.index)
                + " is a void pointer, which has no element size");
        }

        size_t length = Expr<size_t>(buf.length).eval(buffer_context);
        arg = arg.with_buffer_size(length * pointee.size_in_bytes());
    }

    return key;
}

ConfigSpace
KernelBuilder::build_config_space(const GPUDevice& device, const ProblemKey& problem_key) const {
    // The empty config leaves the tunable parameters unresolved, so only the
    // problem and device variables are substituted.
    Config config;
    BuildEval context(device, config, problem_key);
    return resolve(context);
}

KernelDef KernelBuilder::build_kernel_def(
    const GPUDevice& device,
    const Config& config,
    const ProblemKey& problem_key,
    std::vector<TypeInfo> param_types) const {
    BuildEval context(device, config, problem_key);

    std::vector<TemplateArg> template_args;
    for (const auto& arg : template_args_) {
        if (const auto* expr = std::get_if<Expr<Primitive>>(&arg)) {
            template_args.push_back(TemplateArg(expr->eval(context)));
        } else {
            template_args.push_back(std::get<TemplateArg>(arg));
        }
    }

    std::vector<std::string> compiler_flags;
    for (const auto& flag : compiler_flags_) {
        compiler_flags.push_back(flag.eval(context).get());
    }

    for (const auto& p : defines_) {
        compiler_flags.push_back(concat("-D", p.first, "=", p.second.eval(context)));
    }

    KernelDef def(kernel_name_, kernel_source_);
    def.headers = headers_;
    def.template_args = std::move(template_args);
    def.compiler_flags = std::move(compiler_flags);
    def.param_types = std::move(param_types);
    return def;
}

LaunchConfig KernelBuilder::build_launch_config(
    const GPUDevice& device,
    const GPUKernel& kernel,
    const Config& config,
    const ProblemKey& problem_key) const {
    BuildEval context(device, config, problem_key, &kernel);

    for (const auto& assertion : assertions_) {
        if (!assertion.eval(context)) {
            throw std::runtime_error("assertion failed: " + assertion.to_string());
        }
    }

    std::array<Expr<uint32_t>, 3> block = resolve_block_size();
    std::array<Expr<uint32_t>, 3> grid = resolve_grid_size();

    return LaunchConfig {
        {block[0].eval(context), block[1].eval(context), block[2].eval(context)},
        {grid[0].eval(context), grid[1].eval(context), grid[2].eval(context)},
        shared_memory_.eval(context)};
}

const TunableParam*
find_param(const ConfigSpace& space, std::initializer_list<const char*> options) {
    const TunableParam* result = nullptr;

    for (const char* option : options) {
        for (const auto& param : space.parameters()) {
            if (iequals(param.name(), option)) {
                if (result != nullptr) {
                    throw std::runtime_error(concat(
                        "could not resolve parameter: ",
                        result->name(),
                        " and ",
                        param.name(),
                        " are both valid options"));
                }

                result = &param;
            }
        }
    }

    return result;
}

std::array<Expr<uint32_t>, 3> KernelBuilder::resolve_block_size() const {
    if (block_size_) {
        return *block_size_;
    }

    std::array<Expr<uint32_t>, 3> block_size = {1, 1, 1};

    if (const auto* block_size_x = find_param(
            *this,
            {"block_size", "block_dim", "block_size_x", "block_dim_x", "block_x"})) {
        block_size[0] = *block_size_x;
    } else {
        throw std::runtime_error(
            "no parameter found that defines the block size, please call `builder.block_size(...)`");
    }

    if (const auto* block_size_y = find_param(*this, {"block_size_y", "block_dim_y", "block_y"})) {
        block_size[1] = *block_size_y;
    }

    if (const auto* block_size_z = find_param(*this, {"block_size_z", "block_dim_z", "block_z"})) {
        block_size[2] = *block_size_z;
    }

    return block_size;
}

std::array<Expr<uint32_t>, 3> KernelBuilder::resolve_grid_size() const {
    if (grid_size_ && grid_divisors_) {
        throw std::runtime_error(
            "it is not allowed to set both `builder.grid_size(...)` and `builder.grid_divisor(...)`");
    }

    if (grid_size_) {
        return *grid_size_;
    }

    std::array<Expr<uint32_t>, 3> grid_divisors =
        grid_divisors_ ? *grid_divisors_ : resolve_block_size();

    return {
        div_ceil(problem_size_x(), grid_divisors[0].inner()),
        div_ceil(problem_size_y(), grid_divisors[1].inner()),
        div_ceil(problem_size_z(), grid_divisors[2].inner())};
}

}  // namespace kernel_launcher
