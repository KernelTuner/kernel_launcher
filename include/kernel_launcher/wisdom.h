#ifndef KERNEL_LAUNCHER_WISDOME_H
#define KERNEL_LAUNCHER_WISDOME_H

#include <string.h>

#include "kernel_launcher/cuda.h"
#include "kernel_launcher/kernel.h"
#include "kernel_launcher/utils.h"

namespace kernel_launcher {

struct KernelArg {
    virtual ~KernelArg() {}
    virtual bool is_scalar() const = 0;
    virtual TypeInfo type_info() const = 0;
    virtual std::vector<char> to_bytes() const = 0;
    virtual void* as_ptr() const = 0;
};

template<typename T>
struct is_valid_kernel_arg {
    static constexpr bool value = std::is_trivially_copyable<T>::value
        && !std::is_pointer<T>::value && !std::is_reference<T>::value
        && !std::is_void<T>::value;
};

template<typename T>
struct KernelArgScalar: KernelArg {
    static_assert(is_valid_kernel_arg<T>::value, "type must be trivial");

    bool is_scalar() const override {
        return true;
    }

    TypeInfo type_info() const override {
        return TypeInfo::of<T>();
    }

    std::vector<char> to_bytes() const override {
        std::vector<char> result(sizeof(T));
        ::memcpy(result.data(), &data_, sizeof(T));
        return result;
    }

    void* as_ptr() const {
        return static_cast<void*>(&data_);
    }

  private:
    T data_;
};

template<typename T>
struct KernelArgArray: KernelArg {
    static_assert(is_valid_kernel_arg<T>::value, "type must be trivial");

    KernelArgArray(T* ptr, size_t num_elements) :
        ptr_(ptr),
        num_elements_(num_elements_) {}

    bool is_scalar() const override {
        return false;
    }

    TypeInfo type_info() const override {
        return TypeInfo::of<T*>();
    }

    std::vector<char> to_bytes() const override {
        std::vector<char> result(sizeof(T) * num_elements_);
        cuda_raw_copy(ptr_, result.data(), result.size());
        return result;
    }

    void* as_ptr() const {
        return static_cast<void*>(&ptr_);
    }

  private:
    T* ptr_;
    size_t num_elements_;
};

enum struct WisdomResult {
    Success,  // Wisdom file was found with valid configuration
    Invalid,  // Wisdom file was found, but without a valid configuration
    NotFound,  // Wisdom file was not found, it should be written
    IoError,  // An error occurred while reading or parsing files
};

WisdomResult read_wisdom_file(
    const std::string& tuning_key,
    const KernelBuilder& builder,
    const std::string& path,
    CudaDevice device,
    Config& config_out);

WisdomResult read_wisdom_file(
    const std::string& tuning_key,
    const KernelBuilder& builder,
    const std::string& path,
    Config& config_out) {
    return read_wisdom_file(
        tuning_key,
        builder,
        path,
        CudaDevice::current(),
        config_out);
}

void write_wisdom_file(
    const std::string& tuning_key,
    const KernelBuilder& builder,
    const std::string& path,
    const std::string& data_dir,
    dim3 problem_size,
    const std::vector<const KernelArg*>& inputs,
    const std::vector<const KernelArg*>& outputs = {},
    CudaDevice device = CudaDevice::current());

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_WISDOME_H
