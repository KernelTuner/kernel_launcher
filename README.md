# Kernel Launcher

![Kernel Launcher logo](https://kerneltuner.github.io/kernel_launcher/_images/logo.png)


[![github](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/KernelTuner/kernel_launcher/)
![GitHub branch checks state](https://img.shields.io/github/actions/workflow/status/KernelTuner/kernel_launcher/docs.yml)
![GitHub](https://img.shields.io/github/license/KernelTuner/kernel_launcher)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/KernelTuner/kernel_launcher)
![GitHub Repo stars](https://img.shields.io/github/stars/KernelTuner/kernel_launcher?style=social)




_Kernel Launcher_ is a C++ library that enables dynamic compilation _CUDA_ kernels at run time (using [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)) and launching them in an easy type-safe way using C++ magic.
On top of that, Kernel Launcher supports _capturing_ kernel launches, to enable tuning by [Kernel Tuner](https://github.com/KernelTuner/kernel_tuner), and importing the tuning results, known as _wisdom_ files, back into the application.
The result: highly efficient GPU applications with maximum portability.




## Installation

Recommended installation is using CMake. See the [installation guide](https://kerneltuner.github.io/kernel_launcher/install.html).

## Example

There are many ways of using Kernel Launcher. See the documentation for [examples](https://kerneltuner.github.io/kernel_launcher/example.html) or check out the [examples](https://github.com/KernelTuner/kernel_launcher/tree/master/examples) directory.


### Pragma-based API
Below shows an example of using the pragma-based API, which allows existing CUDA kernels to be annotated with Kernel-Launcher-specific directives.

**kernel.cu**
```cpp
#pragma kernel tune(threads_per_block=32, 64, 128, 256, 512, 1024)
#pragma kernel block_size(threads_per_block)
#pragma kernel problem_size(n)
#pragma kernel buffers(A[n], B[n], C[n])
template <typename T>
__global__ void vector_add(int n, T *C, const T *A, const T *B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

**main.cpp**
```cpp
#include "kernel_launcher.h"

int main() {
    // Initialize CUDA memory. This is outside the scope of kernel_launcher.
    unsigned int n = 1000000;
    float *dev_A, *dev_B, *dev_C;
    /* cudaMalloc, cudaMemcpy, ... */

    // Namespace alias.
    namespace kl = kernel_launcher;

    // Launch the kernel! Again, the grid size and block size do not need to
    // be specified, they are calculated from the kernel specifications and
    // run-time arguments.
    kl::launch(
        kl::PragmaKernel("vector_add", "kernel.cu", {"float"}),
        n, dev_C, dev_A, dev_B
    );
}

```


### Builder-based API
Below shows an example of the `KernelBuilder`-based API.
This offers more flexiblity than the pragma-based API, but is also more verbose:

**kernel.cu**
```cpp
template <typename T>
__global__ void vector_add(int n, T *C, const T *A, const T *B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

**main.cpp**
```cpp
#include "kernel_launcher.h"

int main() {
    // Namespace alias.
    namespace kl = kernel_launcher;

    // Define the variables that can be tuned for this kernel.
    auto space = kl::ConfigSpace();
    auto threads_per_block = space.tune("block_size", {32, 64, 128, 256, 512, 1024});

    // Create a kernel builder and set kernel properties such as block size,
    // grid divisor, template arguments, etc.
    auto builder = kl::KernelBuilder("vector_add", "kernel.cu", space);
    builder
        .template_args(kl::type_of<float>())
        .problem_size(kl::arg0)
        .block_size(threads_per_block);

    // Define the kernel
    auto vector_add_kernel = kl::WisdomKernel(builder);

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

If you use Kernel Launcher in your work, please cite the following publication:

> S. Heldens, B. van Werkhoven (2023), "Kernel Launcher: C++ Library for Optimal-Performance Portable CUDA Applications", The Eighteenth International Workshop on Automatic Performance Tuning (iWAPT2023) co-located with IPDPS 2023

As BibTeX:

```Latex
@article{heldens2023kernellauncher,
  title={Kernel Launcher: C++ Library for Optimal-Performance Portable CUDA Applications},
  author={Heldens, Stijn and van Werkhoven, Ben},
  journal={The Eighteenth International Workshop on Automatic Performance Tuning (iWAPT2023) co-located with IPDPS 2023},
  year={2023}
}
```

## Related Work

* [Kernel Tuner](https://github.com/KernelTuner/kernel_tuner)

