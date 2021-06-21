# Kernel Launcher

_Kernel Launcher_ is a header-only C++11 library that can load the results for a CUDA kernel tuned by [Kernel Tuner](https://github.com/benvanwerkhoven/kernel_tuner), dynamically compile the optimal kernel configuration for the current CUDA device (using [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)), and call the kernel in type-safe way using C++ magic.


## Installation
Move the `include` and `thirdparty` directories into you project and include these directories. Link the final binary with `nvrtc`, `cuda`, and `cudart`.

```
gcc source.cpp -Iinclude/ -Ithirdparty/nlohmann/json/single_include -lnvrtc -lcuda -lcudart -std=c++11
```

## Example

See `examples` for example on how to use this library.


First, load a kernel configuration:

```
using namespace kernel_launcher;

// Load optimal configuration for given device and problem size:
auto config = Config::load_best("tuning_results.json", "1000x1000", "Titan_X");

// Load optimal configuration for current device (set using cudaSetDevice).
auto config = Config::load_best_for_current_device("tuning_results.json", "1000x1000");
```


Next, define the kernel in C++ and compile it a run-time.

```
// Define the argument types for the given kernel. It is convenient to do this using a typedef.
using VectorAddKernel = CudaKernel<int, float*, float*, float*>;

// Compile the kernel for the given configuration.
auto kernel = VectorAddKernel::compile(config, "vector_add.cu");

// Or load + compile in one single line.
auto kernel = VectorAddKernel::compile_best_for_current_device("tuning_results.json", "1000x1000", "vector_add.cu");
```

Finaly, call the kernel:

```
// Get tuned thread block size
dim3 block_dim = kernel.get_config().get_block_dim(); // Or: kernel.get_block_dim()
dim3 grid_dim = n / block_dim.x;


// Launch kernel synchronously.
kernel.configure(grid_dim).launch(n, a, b, c);

// Or use syntactic suger:
kernel(grid_dim)(n, a, b, c);


// Launch kernel asynchronously.
kernel.configure_async(grid_dim, smem_size, stream).launch(n, a, b, c);

// Or use syntactic suger:
kernel(grid_dim, smem_size, stream)(n, a, b, c);
```


## Related Work

* [Kernel Tuner](https://github.com/benvanwerkhoven/kernel_tuner)

