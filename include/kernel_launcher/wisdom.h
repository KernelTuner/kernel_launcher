#ifndef KERNEL_LAUNCHER_WISDOM_H
#define KERNEL_LAUNCHER_WISDOM_H

#include <cuda_runtime_api.h>

#include <fstream>
#include <memory>
#include <string>

#include "kernel_launcher/arg.h"
#include "kernel_launcher/export.h"

namespace kernel_launcher {

struct WisdomRecordImpl;

/**
 * Represents a record read from a wisdom file. Use methods such as
 * ``problem_size()`` and ``device_name()`` to retrieve fields of this record.
 */
struct WisdomRecord {
    WisdomRecord(const WisdomRecordImpl& impl) : impl_(impl) {}
    WisdomRecord(const WisdomRecord&) = delete;

    ProblemSize problem_size() const;
    double objective() const;
    const std::string& environment(const char* key) const;
    const std::string& device_name() const;
    Config config() const;

  private:
    const WisdomRecordImpl& impl_;
};

/**
 * Processes a wisdom file and calls the provided callback for each record of
 * the file.
 *
 * @param wisdom_dir The directory where files are located.
 * @param tuning_key The tuning key of the wisdom file.
 * @param space Configuration space that corresponds to the parameters in the
 *              wisdom file.
 * @param callback User-provided function that is called for each record.
 * @return ``true`` if the file was processed successfully, ``false`` if an
 * error ocurred (for example, IO error or invalid file format).
 */
bool process_wisdom_file(
    const std::string& wisdom_dir,
    const std::string& tuning_key,
    const ConfigSpace& space,
    std::function<void(const WisdomRecord&)> callback);

/**
 * Return by `load_best_config` to indicate the result:
 * * Ok: Device and problem size was found exactly.
 * * ProblemSizeMismatch: File was found, but did not contain results for
 *   the provided problem size. Results for another size was selected instead.
 * * DeviceMismatch: File was found, but did not contain results for the
 *   current device. Results for another device was chosen.
 * * NotFound: Wisdom file was not found. Default configuration was returned.
 */
enum struct WisdomResult {
    Ok = 0,
    ProblemSizeMismatch = 1,
    DeviceMismatch = 2,
    NotFound = 3
};

/**
 * Load the optimal configuration for a kernel from a wisdom file based on
 * the provided parameters.
 *
 * @param wisdom_dir Directory where to find wisdom files.
 * @param tuning_key The tuning key of the kernel.
 * @param space The `ConfigSpace` of the kernel.
 * @param device_name The name of the CUDA device.
 * @param device_arch The architecture of the CUDA device.
 * @param problem_size The current problem size.
 * @param result_out Optional, returns one of `WisdomResult`.
 */
Config load_best_config(
    const std::string& wisdom_dir,
    const std::string& tuning_key,
    const ConfigSpace& space,
    const std::string& device_name,
    CudaArch device_arch,
    ProblemSize problem_size,
    WisdomResult* result_out = nullptr);

inline Config load_best_config(
    const std::string& wisdom_dir,
    const std::string& tuning_key,
    const ConfigSpace& space,
    ProblemSize problem_size,
    WisdomResult* result = nullptr) {
    CudaDevice device = CudaDevice::current();

    return load_best_config(
        wisdom_dir,
        tuning_key,
        space,
        device.name(),
        device.arch(),
        problem_size,
        result);
}

struct IWisdomSettings {
    virtual ~IWisdomSettings() = default;

    virtual Config load_config(
        const std::string& tuning_key,
        const ConfigSpace& space,
        ProblemSize problem_size,
        CudaDevice device,
        bool* should_capture_out) const = 0;

    virtual void capture_kernel(
        const std::string& tuning_key,
        const KernelBuilder& builder,
        ProblemSize problem_size,
        const std::vector<KernelArg>& arguments,
        const std::vector<std::vector<uint8_t>>& input_arrays,
        const std::vector<std::vector<uint8_t>>& output_arrays) const = 0;
};

struct CaptureRule {
    CaptureRule(std::string pattern, bool force = false) :
        pattern(std::move(pattern)),
        force(force) {}
    CaptureRule(const char* pattern) : CaptureRule(std::string(pattern)) {}

    std::string pattern;
    bool force = false;
};

struct DefaultWisdomSettings: IWisdomSettings {
    static DefaultWisdomSettings from_env();

    DefaultWisdomSettings();
    DefaultWisdomSettings(
        std::vector<std::string> wisdom_dirs,
        std::string capture_dir,
        std::vector<CaptureRule> capture_rules = {});

    ~DefaultWisdomSettings() override = default;

    Config load_config(
        const std::string& tuning_key,
        const ConfigSpace& space,
        ProblemSize problem_size,
        CudaDevice device,
        bool* should_capture_out) const override;

    void capture_kernel(
        const std::string& tuning_key,
        const KernelBuilder& builder,
        ProblemSize problem_size,
        const std::vector<KernelArg>& arguments,
        const std::vector<std::vector<uint8_t>>& input_arrays,
        const std::vector<std::vector<uint8_t>>& output_arrays) const override;

    virtual bool should_capture_kernel(
        const std::string& tuning_key,
        ProblemSize problem_size,
        WisdomResult result) const;

    bool should_capture_kernel(
        const std::string& tuning_key,
        ProblemSize problem_size) const {
        return should_capture_kernel(
            tuning_key,
            problem_size,
            WisdomResult::NotFound);
    }

    const std::vector<std::string>& wisdom_directories() const {
        return wisdom_dirs_;
    }

    const std::string& capture_directory() const {
        return capture_dir_;
    }

    const std::vector<CaptureRule>& capture_rules() const {
        return capture_rules_;
    }

  private:
    std::vector<std::string> wisdom_dirs_;
    std::string capture_dir_;
    std::vector<CaptureRule> capture_rules_;
};

/**
 * Describes how to load the configuration for a `WisdomKernel`. The most
 * important method is `load_config` which loads a configuration for a kernel.
 */
struct WisdomSettings {
    WisdomSettings();
    WisdomSettings(
        std::string wisdom_dir,
        std::string capture_dir,
        std::vector<CaptureRule> capture_rules = {});
    WisdomSettings(std::shared_ptr<IWisdomSettings> oracle);

    template<typename T>
    WisdomSettings(std::shared_ptr<T> ptr) :
        WisdomSettings(std::shared_ptr<IWisdomSettings> {std::move(ptr)}) {}

    WisdomSettings(const WisdomSettings&) = default;

    /**
     * Load the configuration for the given parameters.
     *
     * @param tuning_key The tuning key of the kernel.
     * @param space The configuration space of the kernel.
     * @param problem_size The current problem size.
     * @param device The current device.
     * @param should_capture_out Optional. Indicates if kernel must be captured.
     */
    Config load_config(
        const std::string& tuning_key,
        const ConfigSpace& space,
        ProblemSize problem_size,
        CudaDevice device,
        bool* should_capture_out = nullptr) const {
        return impl_->load_config(
            tuning_key,
            space,
            problem_size,
            device,
            should_capture_out);
    }

    /**
     *
     *
     * @param tuning_key
     * @param builder
     * @param problem_size
     * @param param_types
     * @param inputs
     * @param outputs
     */
    void capture_kernel(
        const std::string& tuning_key,
        const KernelBuilder& builder,
        ProblemSize problem_size,
        const std::vector<KernelArg>& arguments,
        const std::vector<std::vector<uint8_t>>& input_arrays,
        const std::vector<std::vector<uint8_t>>& output_arrays) const {
        return impl_->capture_kernel(
            tuning_key,
            builder,
            problem_size,
            arguments,
            input_arrays,
            output_arrays);
    }

  private:
    std::shared_ptr<IWisdomSettings> impl_;
};

/**
 * Returns the default global `WisdomSettings`.
 */
WisdomSettings default_wisdom_settings();

/**
 * Append directory where to search for wisdom files by the `WisdomSettings`
 * returned by `default_wisdom_settings`.
 */
void append_global_wisdom_directory(std::string);

// Deprecated
void set_global_wisdom_directory(std::string);

/**
 * Set directory where captures will be stored for the `WisdomSettings`
 * returned by `default_wisdom_settings`.
 */
void set_global_capture_directory(std::string);

/**
 * Add capture pattern to the `WisdomSettings` returned by
 * `default_wisdom_settings`.
 */
void add_global_capture_pattern(CaptureRule rule);

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_WISDOM_H
