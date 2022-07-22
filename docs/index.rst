Kernel Launcher
===========================================

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contents

   Kernel Launcher <self>
   install
   example
   api/index
   license


*Kernel Launcher* is a C++ library that makes it easy to dynamically compile *CUDA* kernels at run time (using `NVRTC <https://docs.nvidia.com/cuda/nvrtc/index.html>`_) and call them in easy type-safe way using C++ magic.
There are two reasons for using run-time compilation:

* Kernels that have tunable parameters (block size, elements per thread, loop unroll factors, etc.) where the optimal configuration can only be determined at runtime since they depend factors such as the type of GPU and the problem size.

* Increase performance by injecting runtime values as compile-time constant values into the kernel code (dimensions, array strides, weights, etc.).




Basic Example
=============

This sections hows a basic code example. See :ref:`example` for a more advance example.

Consider the following CUDA kernel for vector addition.
This kernel has a template parameter ``T`` and a tunable parameter ``ELEMENTS_PER_THREAD``.

::

    template <typename T>
    __global__
    void vector_add(int n, T* C, const T* A, const T* B) {
        for (int k = 0; k < ELEMENTS_PER_THREAD; k++) {
            int i = (blockIdx.x * ELEMENTS_PER_THREAD + k) * blockDim.x + threadIdx.x;

            if (i < n) {
                C[i] = A[i] + B[i];
            }
        }
    }


The following snippet shows how to use *Kernel Launcher* in host code::

    #include "kernel_launcher.h"

    int main() {
        // Namespace alias.
        namespace kl = kernel_launcher;

        // Create a kernel builder
        kl::KernelBuilder builder("vector_add", "vector_add_kernel.cu");

        // Define the variables that can be tuned for this kernel.
        kl::ParamExpr threads_per_block = builder.tune("block_size", {32, 64, 128, 256, 512, 1024});
        kl::ParamExpr elements_per_thread = builder.tune("elements_per_thread", {1, 2, 4, 8});

        // `threads_per_block` and `elements_per_thread` are expression placeholders.
        // Expressions can be combined using operators to create new expressions.
        auto elements_per_block = elements_per_thread * threads_per_block;

        // Set kernel properties such as block size, grid divisor, template arguments, etc.
        builder
            .block_size(threads_per_block)
            .grid_divisors(elements_per_block)
            .template_args(kl::type_of<float>())
            .define("ELEMENTS_PER_THREAD", elements_per_thread);

        // Create a configuration. Here we use hard-coded values, but these values
        // can also be loaded from a file or database with tuning results.
        kl::Config config;
        config.insert(threads_per_block, 128);
        config.insert(elements_per_thread, 4);

        // Compile the kernel for the given configuration. This evaluates all
        // expressions and calls NVRTC to perform the actual compilation.
        kl::Kernel<unsigned int, float*, const float*, const float*> vector_add_kernel;
        vector_add_kernel.compile(builder, config);

        // Initialize device memory. This is outside the scope of kernel_launcher.
        // Here we just allocate memory using cudaMalloc.
        unsigned int n = 1000000;
        float *dev_A, *dev_B, *dev_C;
        cudaMalloc((void**)(&dev_A), sizeof(float) * n);
        cudaMalloc((void**)(&dev_B), sizeof(float) * n);
        cudaMalloc((void**)(&dev_C), sizeof(float) * n);
        cudaSetDevice(0);

        // Launch the kernel! Note that the block size and grid size are
        // derived from the problem size, thus they do not need to be specified
        unsigned int problem_size = n;
        vector_add_kernel(problem_size)(n, dev_C, dev_A, dev_B);
    }




Indices and tables
============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

