#ifndef KERNEL_LAUNCHER_WISDOM_H
#define KERNEL_LAUNCHER_WISDOM_H

#include <cuda_runtime_api.h>

#include <fstream>
#include <memory>
#include <string>

#include "kernel_launcher/export.h"
#include "kernel_launcher/kernel.h"

namespace kernel_launcher {

/// Returned by `load_best_config` to indicate the result:
/// - NotFound: Wisdom file was not found. Default config is returned.
/// - DeviceMismatch: File was found, but did not contain results for the
///                   the current device. Results for another device was chosen.
/// - ProblemSizeMismatch: File was found, but did not contain results for
///                        the current problem size. Results for another size
///                        was selected instead.
/// - Ok: Device and problem size was found exactly.
enum struct WisdomResult {
    NotFound = 0,
    DeviceMismatch = 1,
    ProblemSizeMismatch = 2,
    Ok = 3
};

Config load_best_config(
    const std::string& wisdom_dir,
    const std::string& tuning_key,
    const ConfigSpace& space,
    const std::string& device_name,
    CudaArch device_arch,
    ProblemSize problem_size,
    WisdomResult* result_out);

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

struct Oracle {
    virtual ~Oracle() = default;

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
        const std::vector<TypeInfo>& param_types,
        const std::vector<std::vector<uint8_t>>& inputs,
        const std::vector<std::vector<uint8_t>>& outputs) const = 0;
};

struct DefaultOracle: Oracle {
    static DefaultOracle from_env();

    DefaultOracle();
    DefaultOracle(
        std::vector<std::string> wisdom_dirs,
        std::string capture_dir,
        std::vector<std::string> capture_patterns = {},
        bool force_capture = false);

    virtual ~DefaultOracle() = default;

    virtual Config load_config(
        const std::string& tuning_key,
        const ConfigSpace& space,
        ProblemSize problem_size,
        CudaDevice device,
        bool* should_capture_out) const override;

    virtual void capture_kernel(
        const std::string& tuning_key,
        const KernelBuilder& builder,
        ProblemSize problem_size,
        const std::vector<TypeInfo>& param_types,
        const std::vector<std::vector<uint8_t>>& inputs,
        const std::vector<std::vector<uint8_t>>& outputs) const override;

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

    const std::vector<std::string>& capture_patterns() const {
        return capture_patterns_;
    }

    bool is_capture_forced() const {
        return force_capture_;
    }

  private:
    std::vector<std::string> wisdom_dirs_;
    std::string capture_dir_;
    std::vector<std::string> capture_patterns_;
    bool force_capture_;
};

struct WisdomSettings {
    WisdomSettings();
    WisdomSettings(
        std::string wisdom_dir,
        std::string capture_dir,
        std::vector<std::string> capture_patterns = {},
        bool force_capture = false);
    WisdomSettings(std::shared_ptr<Oracle> oracle);

    template<typename T>
    WisdomSettings(std::shared_ptr<T> ptr) :
        WisdomSettings(std::shared_ptr<Oracle> {std::move(ptr)}) {}

    WisdomSettings(const WisdomSettings&) = default;

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

    void capture_kernel(
        const std::string& tuning_key,
        const KernelBuilder& builder,
        ProblemSize problem_size,
        const std::vector<TypeInfo>& param_types,
        const std::vector<std::vector<uint8_t>>& inputs,
        const std::vector<std::vector<uint8_t>>& outputs) const {
        return impl_->capture_kernel(
            tuning_key,
            builder,
            problem_size,
            param_types,
            inputs,
            outputs);
    }

  private:
    std::shared_ptr<Oracle> impl_;
};

void append_global_wisdom_directory(std::string);

void set_global_wisdom_directory(std::string);
void set_global_tuning_directory(std::string);
void add_global_capture_pattern(std::string);
WisdomSettings default_wisdom_settings();

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_WISDOM_H
