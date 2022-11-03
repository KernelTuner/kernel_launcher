#ifndef KERNEL_LAUNCHER_EXPORT_H
#define KERNEL_LAUNCHER_EXPORT_H

#include <cuda_runtime_api.h>

#include <cstring>
#include <string>

#include "kernel_launcher/builder.h"

namespace kernel_launcher {

bool tuning_file_exists(
    const std::string& directory,
    const std::string& tuning_key,
    ProblemSize problem_size);

void export_tuning_file(
    const std::string& directory,
    const std::string& tuning_key,
    const KernelBuilder& builder,
    ProblemSize problem_size,
    const std::vector<TypeInfo>& param_types,
    const std::vector<std::vector<uint8_t>>& inputs,
    const std::vector<std::vector<uint8_t>>& outputs = {});

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_EXPORT_H
