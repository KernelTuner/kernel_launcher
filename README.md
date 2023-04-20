# Kernel Launcher

![Kernel Launcher logo](https://kerneltuner.github.io/kernel_launcher/_images/logo.png)


[![github](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/KernelTuner/kernel_launcher/)
![GitHub branch checks state](https://img.shields.io/github/actions/workflow/status/KernelTuner/kernel_launcher/docs.yml)
![GitHub](https://img.shields.io/github/license/KernelTuner/kernel_launcher)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/KernelTuner/kernel_launcher)
![GitHub Repo stars](https://img.shields.io/github/stars/KernelTuner/kernel_launcher?style=social)




_Kernel Launcher_ is a C++ library that makes it easy to dynamically compile _CUDA_ kernels at run time (using [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)) and call them in an easy type-safe way using C++ magic.
Additionally, _Kernel Launcher_ supports exporting kernel specifications, to enable tuning by [Kernel Tuner](https://github.com/KernelTuner/kernel_tuner), and importing the tuning results, known as _wisdom_ files, back into the application.



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
    auto threads_per_block = builder.tune("block_size", {32, 64, 128, 256, 512, 1024});
    auto elements_per_thread = builder.tune("elements_per_thread", {1, 2, 4, 8});

    // Set kernel properties such as block size, grid divisor, template arguments, etc.
    builder
        .problem_size(kl::arg0)
        .block_size(threads_per_block)
        .grid_divisors(threads_per_block * elements_per_thread)
        .template_args(kl::type_of<float>())
        .define("ELEMENTS_PER_THREAD", elements_per_thread);

    // Define the kernel
    kl::WisdomKernel vector_add_kernel(builder);

    // Initialize CUDA memory. This is outside the scope of kernel_launcher.
    unsigned int n = 1000000;
    float *dev_A, *dev_B, *dev_C;
    /* cudaMalloc, cudaMemcpy, ... */

    // Launch the kernel! Note that kernel is compiled on the first call.
    // The grid size and block size do not need to be specified, they are
    // derived from the kernel specifications and run-time arguments.
    vector_add_kernel(n, dev_C, dev_A, dev_B);
}

```

## License

Licensed under Apache 2.0. See [LICENSE](https://github.com/KernelTuner/kernel_launcher/blob/master/LICENSE).

## Citation

```
@article{heldens2023kernellauncher,
  title={Kernel Launcher: C++ Library for Optimal-Performance Portable CUDA Applications},
  author={Heldens, Stijn and van Werkhoven, Ben},
  journal={The Eighteenth International Workshop on Automatic Performance Tuning (iWAPT2023) co-located with IPDPS 2023},
  year={2023}
}
```

## Related Work

* [Kernel Tuner](https://github.com/KernelTuner/kernel_tuner)

