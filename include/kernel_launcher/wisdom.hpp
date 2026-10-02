#ifndef KERNEL_LAUNCHER_WISDOM_H
#define KERNEL_LAUNCHER_WISDOM_H

#include <cstddef>
#include <functional>
#include <memory>

#include "kernel_launcher/compiler.hpp"
#include "kernel_launcher/config.hpp"
#include "kernel_launcher/gpu_utils.hpp"
#include "kernel_launcher/kernel_arg.hpp"
#include "kernel_launcher/kernel_builder.hpp"
#include "kernel_launcher/problem_key.hpp"

namespace kernel_launcher {

/**
 * @addtogroup wisdom
 * @{
 */

/**
 * A single record (one line) of a wisdom file, as passed to the callback of
 * `process_wisdom_files`. It is only valid for the duration of that callback.
 */
class WisdomRecord {
  public:
    struct Impl;
    WisdomRecord(Impl* impl) : impl_(impl) {}

    /**
     * The configuration that was measured.
     */
    const Config& config() const;

    /**
     * The measured objective of the configuration (lower is better).
     */
    double objective() const;

    /**
     * The problem size for which the configuration was measured.
     */
    std::array<int64_t, MAX_PROBLEM_DIMS> problem_size() const;

    /**
     * The value of the given environment entry (such as `device_name`), or an
     * empty string if it is not present.
     */
    std::string_view env(std::string_view key) const;

    /**
     * The value of the given problem feature, or an empty `Primitive` if it is
     * not present.
     */
    Primitive feature(std::string_view key) const;

  private:
    Impl* impl_;
};

/**
 * Reads the wisdom file for the given tuning key from each of the given wisdom
 * directories and invokes `callback` for every valid record. This can be used
 * to write a custom wisdom selection algorithm.
 */
void process_wisdom_files(
    const std::vector<std::string_view>& wisdom_dirs,
    const ConfigSpace& space,
    std::string_view tuning_key,
    std::function<void(WisdomRecord)> callback);

enum struct WisdomResult { Ok = 0, ProblemMismatch = 1, DeviceMismatch = 2, NotFound = 3 };

/**
 * Selects the best configuration from the wisdom files for `problem` in the
 * given wisdom directories. Records are ranked by device match, then problem
 * size, then features, and finally by objective. If no record is found, the
 * default configuration of `space` is returned. `result_out` is set to how
 * well the selected record matches.
 */
Config load_wisdom_config(
    const std::vector<std::string_view>& wisdom_dirs,
    const GPUArch& arch,
    const ConfigSpace& space,
    const ProblemKey& problem,
    WisdomResult* result_out);

/**
 * See `load_wisdom_config` above.
 */
inline Config load_wisdom_config(
    std::string_view wisdom_dir,
    const GPUArch& arch,
    const ConfigSpace& space,
    const ProblemKey& problem,
    WisdomResult* result_out) {
    return load_wisdom_config(
        std::vector<std::string_view> {wisdom_dir},
        arch,
        space,
        problem,
        result_out);
}

/**
 * Appends a record for `config` with the given `objective` (lower is better)
 * to the wisdom file for `problem.tuning_key` in `wisdom_dir`. If the file does
 * not exist, it is created with a header derived from `space`. Otherwise, the
 * existing header is checked against `space` and an exception is thrown if
 * they do not match.
 */
void write_wisdom_record(
    std::string_view wisdom_dir,
    const GPUArch& arch,
    const ConfigSpace& space,
    const ProblemKey& problem,
    const Config& config,
    double objective);

enum struct TuningAction { None, Autotune, Capture };

// Compiles and runs the kernel with the given config, and returns the
// execution time in seconds. Throws if the config cannot be compiled or
// launched, in which case the provider should skip it.
using BenchmarkFn = std::function<double(const Config&)>;

class WisdomProvider {
  public:
    virtual ~WisdomProvider() = default;

    virtual Config load_config(
        const GPUDevice& device,
        const ConfigSpace& space,
        const ProblemKey& problem,
        TuningAction* action_out) = 0;

    virtual Config autotune(
        const GPUDevice& device,
        const KernelBuilder& builder,
        const ProblemKey& problem,
        const BenchmarkFn& benchmark) = 0;

    virtual void capture_kernel(
        const GPUDevice& device,
        const KernelBuilder& builder,
        const ProblemKey& problem,
        const std::vector<KernelArg>& arguments,
        const std::vector<std::vector<std::byte>>& input_arrays,
        const std::vector<std::vector<std::byte>>& output_arrays) = 0;
};

struct KernelSettings {
    std::shared_ptr<CompilerBase> compiler;
    std::shared_ptr<WisdomProvider> wisdom;
};

/** @} */

}  // namespace kernel_launcher

#endif
