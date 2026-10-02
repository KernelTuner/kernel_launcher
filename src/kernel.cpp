#include "kernel_launcher/kernel.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace kernel_launcher {

struct RawKernelInstance::PendingCapture {
    PendingCapture(
        std::shared_ptr<WisdomProvider> wisdom,
        KernelBuilder builder,
        size_t skip_launches) :
        wisdom(std::move(wisdom)),
        builder(std::move(builder)),
        skip_launches(skip_launches) {}

    // Called before a launch. Returns false if this launch is not captured,
    // either because it is skipped or because the arguments do not match
    // `problem`.
    bool
    pre_launch(const ProblemKey& problem, GPUStream stream, const std::vector<KernelArg>& args);

    // Called after a launch for which `pre_launch` returned true.
    void post_launch(const ProblemKey& problem, GPUStream stream);

    std::shared_ptr<WisdomProvider> wisdom;
    KernelBuilder builder;

    // The members below are only accessed by the thread that has taken this
    // capture out of `State::pending_capture`.

    // Number of launches left to skip before capturing.
    size_t skip_launches;
    // Set by `pre_launch` for use by `post_launch`.
    std::vector<KernelArg> sized_args;
    std::vector<std::vector<std::byte>> inputs;
};

RawKernelInstance::State::State(
    std::shared_ptr<GPUKernel> kernel,
    ProblemKey problem,
    Config config,
    LaunchConfig launch_config) :
    kernel(std::move(kernel)),
    problem(std::move(problem)),
    config(std::move(config)),
    launch_config(launch_config) {}

RawKernelInstance::State::~State() {
    delete pending_capture.load();
}

RawKernelInstance::RawKernelInstance(
    std::shared_ptr<GPUKernel> kernel,
    ProblemKey problem,
    Config config,
    LaunchConfig launch_config) :
    state_(
        std::make_shared<const State>(
            std::move(kernel),
            std::move(problem),
            std::move(config),
            launch_config)) {}

void RawKernelInstance::capture_next_launch(
    std::shared_ptr<WisdomProvider> wisdom,
    KernelBuilder builder,
    size_t skip_launches) const {
    if (!wisdom) {
        throw std::invalid_argument("wisdom provider cannot be null");
    }

    auto next =
        std::make_unique<PendingCapture>(std::move(wisdom), std::move(builder), skip_launches);
    std::unique_ptr<PendingCapture> previous {state_->pending_capture.exchange(next.release())};
}

static void launch_kernel(
    const GPUKernel& kernel,
    const LaunchConfig& config,
    GPUStream stream,
    const std::vector<KernelArg>& args) {
    auto raw_args = std::make_unique<const void*[]>(args.size());

    for (size_t i = 0; i < args.size(); i++) {
        raw_args[i] = args[i].data();
    }

    kernel.launch(stream, config, raw_args.get());
}

void RawKernelInstance::launch_args(GPUStream stream, const std::vector<KernelArg>& args) const {
    std::unique_ptr<PendingCapture> capture;

    if (state_->pending_capture.load() != nullptr) {
        // Taking the pointer gives this thread sole ownership of the capture.
        // If the capture throws, it is dropped and not retried.
        capture.reset(state_->pending_capture.exchange(nullptr));

        if (capture && !capture->pre_launch(state_->problem, stream, args)) {
            // Not captured: put the capture back, unless a new one was set in the meantime.
            PendingCapture* expected = nullptr;
            PendingCapture* raw = capture.release();
            if (!state_->pending_capture.compare_exchange_strong(expected, raw)) {
                delete raw;
            }
        }
    }

    launch_kernel(*state_->kernel, state_->launch_config, stream, args);

    if (capture) {
        capture->post_launch(state_->problem, stream);
    }
}

// Copies the contents of every buffer argument to the host. Non-buffer
// arguments get an empty vector.
static std::vector<std::vector<std::byte>>
copy_buffers_to_host(GPUStream stream, const std::vector<KernelArg>& args) {
    // Buffers may be written by work on other streams, so wait for the whole
    // device before copying.
    gpu_synchronize();

    std::vector<std::vector<std::byte>> result(args.size());

    for (size_t i = 0; i < args.size(); i++) {
        const KernelArg& arg = args[i];

        if (!arg.type().is_pointer() || arg.buffer_size() == 0) {
            continue;
        }

        void* ptr;
        std::memcpy(&ptr, arg.data(), sizeof(void*));

        result[i].resize(arg.buffer_size());
        gpu_copy(result[i].data(), ptr, arg.buffer_size(), stream);
    }

    gpu_synchronize(stream);

    return result;
}

bool RawKernelInstance::PendingCapture::pre_launch(
    const ProblemKey& problem,
    GPUStream stream,
    const std::vector<KernelArg>& args) {
    // The arguments of a launch carry no buffer sizes. Building the problem
    // key sets them, and also checks that the arguments match the problem
    // this instance was compiled for.
    std::vector<KernelArg> new_args = args;

    try {
        if (builder.build_problem_key(new_args) != problem) {
            return false;
        }
    } catch (const std::exception&) {
        return false;
    }

    if (skip_launches > 0) {
        skip_launches--;
        return false;
    }

    sized_args = std::move(new_args);
    inputs = copy_buffers_to_host(stream, sized_args);
    return true;
}

void RawKernelInstance::PendingCapture::post_launch(const ProblemKey& problem, GPUStream stream) {
    auto outputs = copy_buffers_to_host(stream, sized_args);
    wisdom->capture_kernel(GPUDevice::current(), builder, problem, sized_args, inputs, outputs);
}

RawKernelInstance compile_kernel_raw(
    const KernelBuilder& builder,
    const ProblemKey& problem,
    const Config& config,
    std::vector<TypeInfo> param_types,
    CompilerBase& compiler,
    GPUContextRef context) {
    GPUDevice device = context.device();

    if (!builder.build_config_space(device, problem).is_valid(config)) {
        throw std::runtime_error("config is not valid for this problem and device");
    }

    KernelDef def = builder.build_kernel_def(device, config, problem, std::move(param_types));

    auto kernel = compiler.compile(context, std::move(def));

    // Querying the kernel's attributes requires its context to be current.
    GPUContextGuard guard(context);
    auto launch = builder.build_launch_config(device, *kernel, config, problem);

    return RawKernelInstance(std::move(kernel), problem, config, launch);
}

struct KernelKey {
    GPUContextRef context;
    ProblemKey problem;
    std::vector<TypeInfo> param_types;
    size_t hash = 0;

    KernelKey(GPUContextRef context, ProblemKey problem, std::vector<TypeInfo> param_types) :
        context(context),
        problem(std::move(problem)),
        param_types(std::move(param_types)) {
        // precompute the hash since it expensive to compute on each lookup
        hash_combine(hash, this->context);
        hash_combine(hash, this->problem);
        hash_combine(hash, this->param_types);
    }

    bool operator==(const KernelKey& that) const {
        // Comparing the precomputed hash first is a cheap early-out.
        return hash == that.hash  //
            && problem == that.problem  //
            && param_types == that.param_types  //
            && context == that.context;
    }
};

struct KernelKeyHash {
    size_t operator()(const KernelKey& k) const noexcept {
        return k.hash;
    }
};

struct Kernel::Impl {
    Impl(KernelBuilder&& builder, KernelSettings&& settings) :
        builder(std::move(builder)),
        compiler(std::move(settings.compiler)),
        wisdom(std::move(settings.wisdom)) {
        if (!compiler) {
            compiler = std::make_shared<RuntimeCompiler>();
        }
    }

    std::mutex mutex;
    KernelBuilder builder;
    std::shared_ptr<CompilerBase> compiler;
    // May be null, in which case the default config is always used.
    std::shared_ptr<WisdomProvider> wisdom;
    std::unordered_map<KernelKey, RawKernelInstance, KernelKeyHash> instance_cache;
};

Kernel::Kernel(KernelBuilder builder, KernelSettings settings) :
    impl_(std::make_shared<Impl>(std::move(builder), std::move(settings))) {}

RawKernelInstance Kernel::compile_args(std::vector<KernelArg>& args, GPUContextRef context) {
    if (!impl_) {
        throw std::runtime_error("Kernel is not initialized");
    }

    // The builder is never modified, so this does not need the lock.
    ProblemKey problem = impl_->builder.build_problem_key(args);

    std::vector<TypeInfo> param_types;
    param_types.reserve(args.size());

    for (const auto& arg : args) {
        param_types.push_back(arg.type());
    }

    return compile_for(std::move(problem), std::move(param_types), context);
}

RawKernelInstance
Kernel::compile_for(ProblemKey problem, std::vector<TypeInfo> param_types, GPUContextRef context) {
    if (!impl_) {
        throw std::runtime_error("Kernel is not initialized");
    }

    auto device = context.device();

    std::lock_guard guard {impl_->mutex};
    KernelKey kernel_key {context, std::move(problem), std::move(param_types)};

    if (auto p = impl_->instance_cache.find(kernel_key); p != impl_->instance_cache.end()) {
        return p->second;
    }

    ConfigSpace space = impl_->builder.build_config_space(device, kernel_key.problem);

    TuningAction action = TuningAction::None;
    Config config = impl_->wisdom
        ? impl_->wisdom->load_config(device, space, kernel_key.problem, &action)
        : space.default_config();

    RawKernelInstance instance = compile_kernel_raw(
        impl_->builder,
        kernel_key.problem,
        config,
        kernel_key.param_types,
        *impl_->compiler,
        context);

    // The pending capture is shared with the cached copy, so only the first
    // launch of this instance is captured.
    if (action == TuningAction::Capture) {
        instance.capture_next_launch(impl_->wisdom, impl_->builder);
    }

    auto it = impl_->instance_cache.emplace(std::move(kernel_key), std::move(instance));

    return it.first->second;
}

}  // namespace kernel_launcher
