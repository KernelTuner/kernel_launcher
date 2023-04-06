#ifndef KERNEL_LAUNCHER_EXPORT_H
#define KERNEL_LAUNCHER_EXPORT_H

#include <cuda_runtime_api.h>

#include <cstring>
#include <string>

#include "kernel_launcher/builder.h"

namespace kernel_launcher {

/**
 * Returns `true` if there is already a capture available for the given
 * `tuning_key` and `problem_size` in the given `directory`.
 */
bool capture_file_exists(
    const std::string& directory,
    const std::string& tuning_key,
    ProblemSize problem_size);

/**
 * Export a capture for the given `tuning_key` and `problem_size` to the
 * given `directory`.
 */
void export_capture_file(
    const std::string& directory,
    const std::string& tuning_key,
    const KernelBuilder& builder,
    ProblemSize problem_size,
    const std::vector<KernelArg>& arguments,
    const std::vector<std::vector<uint8_t>>& input_arrays,
    const std::vector<std::vector<uint8_t>>& output_arrays = {});

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_EXPORT_H
