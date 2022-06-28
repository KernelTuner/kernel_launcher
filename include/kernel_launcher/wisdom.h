#ifndef KERNEL_LAUNCHER_WISDOME_H
#define KERNEL_LAUNCHER_WISDOME_H

#include "kernel_launcher/kernel.h"
#include "kernel_launcher/utils.h"

namespace kernel_launcher {

struct KernelArg {
    virtual ~KernelArg() {}
    virtual TypeInfo type_info() const = 0;
    virtual std::vector<char> to_bytes() const = 0;
    virtual void* as_ptr() const = 0;
};

template<typename T, typename Enabled = void>
struct KernelArgImpl;

template<typename T>
struct KernelArgImpl<T>: KernelArg {};

enum struct WisdomResult {
    Success,
    NotFound,
    IoError,
    Untuned,
};

WisdomResult read_wisdom_file(
    const std::string& path,
    const std::string& tuning_key,
    const KernelBuilder& builder,
    Config& config);

void write_wisdom_file(
    const std::string& path,
    const std::string& tuning_key,
    const KernelBuilder& builder,
    const std::string& data_dir,
    const std::vector<KernelArg>& inputs,
    const std::vector<KernelArg>& outputs = {});

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_WISDOME_H
