#ifndef KERNEL_LAUNCHER_H
#define KERNEL_LAUNCHER_H

/**
 * @defgroup kernel Kernels
 * Compiling and launching kernels, and passing arguments to them.
 */

/**
 * @defgroup builder Kernel definition
 * Describing a kernel and its tunable parameters.
 */

/**
 * @defgroup expr Expressions
 * Lazily evaluated expressions over arguments, tunable parameters, and
 * device attributes.
 */

/**
 * @defgroup registry Registry
 * Caching of kernels by descriptor.
 */

/**
 * @defgroup wisdom Wisdom and tuning
 * Wisdom files, problem keys, captures, and autotuning.
 */

/**
 * @defgroup compiler Compilation
 * Compiling kernel sources and locating source files.
 */

/**
 * @defgroup gpu GPU utilities
 * Thin wrappers around the GPU driver API (CUDA or HIP).
 */

/**
 * @defgroup types Values and types
 * Dynamically typed values and runtime type information.
 */

/**
 * @defgroup utils Utilities
 * Hashing, string, logging, and file helpers.
 */

#include "kernel_launcher/capture.hpp"
#include "kernel_launcher/compiler.hpp"
#include "kernel_launcher/config.hpp"
#include "kernel_launcher/expr.hpp"
#include "kernel_launcher/file_loader.hpp"
#include "kernel_launcher/gpu_utils.hpp"
#include "kernel_launcher/kernel.hpp"
#include "kernel_launcher/kernel_arg.hpp"
#include "kernel_launcher/kernel_builder.hpp"
#include "kernel_launcher/primitive.hpp"
#include "kernel_launcher/problem_key.hpp"
#include "kernel_launcher/registry.hpp"
#include "kernel_launcher/tuner.hpp"
#include "kernel_launcher/type_info.hpp"
#include "kernel_launcher/utils.hpp"
#include "kernel_launcher/wisdom.hpp"

#endif
