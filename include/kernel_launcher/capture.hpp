#ifndef KERNEL_LAUNCHER_CAPTURE_H
#define KERNEL_LAUNCHER_CAPTURE_H

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "kernel_launcher/kernel_arg.hpp"
#include "kernel_launcher/kernel_builder.hpp"
#include "kernel_launcher/problem_key.hpp"

namespace kernel_launcher {

/**
 * @addtogroup wisdom
 * @{
 */

/**
 * Returns `true` if there is already a capture available for the given
 * `problem_key` in the given `directory`.
 */
bool capture_file_exists(std::string_view directory, const ProblemKey& problem_key);

/**
 * Export a capture of the kernel defined by `builder` for the given
 * `problem_key` to the given `directory`. If a capture already exists for
 * `problem_key`, it is kept and nothing is written.
 *
 * `input_arrays` has one entry per argument: the buffer contents for
 * pointer arguments and an empty vector otherwise. `output_arrays` follows
 * the same layout, or is empty if there are no outputs.
 */
void export_capture_file(
    std::string_view directory,
    const KernelBuilder& builder,
    const ProblemKey& problem_key,
    const std::vector<KernelArg>& arguments,
    const std::vector<std::vector<std::byte>>& input_arrays,
    const std::vector<std::vector<std::byte>>& output_arrays = {});

/** @} */

}  // namespace kernel_launcher

#endif  //KERNEL_LAUNCHER_CAPTURE_H
