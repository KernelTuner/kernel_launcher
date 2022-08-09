# Kernel Launcher

![Kernel Launcher logo](https://kerneltuner.github.io/kernel_launcher/_images/logo.png)


[![github](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/KernelTuner/kernel_launcher/)
![GitHub branch checks state](https://img.shields.io/github/checks-status/KernelTuner/kernel_launcher/master)
![GitHub](https://img.shields.io/github/license/KernelTuner/kernel_launcher)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/KernelTuner/kernel_launcher)
![GitHub Repo stars](https://img.shields.io/github/stars/KernelTuner/kernel_launcher?style=social)




_Kernel Launcher_ is a C++ library that makes it easy to dynamically compile _CUDA_ kernels at run time (using [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)) and call them in an easy type-safe way using C++ magic.
Additionally, _Kernel Launcher_ supports exporting kernel specifications, such that they can be tuned by [Kernel Tuner](https://github.com/benvanwerkhoven/kernel_tuner), and importing the tuning results, known as _wisdom_ files.



## Installation

Recommended installation is using CMake. See the [installation guide](https://kerneltuner.github.io/kernel_launcher/install.html).

## Example

See the documentation for [examples](https://kerneltuner.github.io/kernel_launcher/example.html) or check out the [examples](https://github.com/KernelTuner/kernel_launcher/tree/master/examples) directory.

```cpp
#include "kernel_launcher.h"

int main() {
    // Namespace alias.
    namespace kl = kernel_launcher;

    // Create a kernel builder
    kl::KernelBuilder builder("vector_add", "vector_add_kernel.cu");

    // Define the variables that can be tuned for this kernel.
    kl::ParamExpr threads_per_block = builder.tune("block_size", {32, 64, 128, 256, 512, 1024});
    kl::ParamExpr elements_per_thread = builder.tune("elements_per_thread", {1, 2, 4, 8});

    // Set kernel properties such as block size, grid divisor, template arguments, etc.
    builder
        .block_size(threads_per_block)
        .grid_divisors(threads_per_block * elements_per_thread)
        .template_args(kl::type_of<float>())
        .define("ELEMENTS_PER_THREAD", elements_per_thread);

    // Define the kernel
    kl::WisdomKernel vector_add_kernel("vector_add", builder);

    // Initialize CUDA memory. This is outside the scope of kernel_launcher.
    unsigned int n = 1000000;
    float *dev_A, *dev_B, *dev_C;
    /* cudaMalloc, cudaMemcpy, ... */

    // Launch the kernel! Note that kernel is compiled on the first call.
    // The grid size and block size do not need to be specified, they are
    // derived from the kernel specifications and problem size.
    unsigned int problem_size = n;
    vector_add_kernel(problem_size)(n, dev_C, dev_A, dev_B);
}

```

## License

Licensed under Apache 2.0. See [LICENSE](https://github.com/KernelTuner/kernel_launcher/blob/master/LICENSE).


## Related Work

* [Kernel Tuner](https://github.com/benvanwerkhoven/kernel_tuner)

