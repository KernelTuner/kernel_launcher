.. highlight:: c++
   :linenothreshold: 1

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contents

   Kernel Launcher <self>
   install
   example
   api
   license

.. raw:: html

   <!--

Kernel Launcher
===============

.. raw:: html

   --><h1>

.. image:: /logo.png
   :width: 670
   :alt: kernel launcher

.. raw:: html

   </h1>

*Kernel Launcher* is a C++ library that makes it easy to dynamically compile *CUDA* kernels at run time (using `NVRTC <https://docs.nvidia.com/cuda/nvrtc/index.html>`_) and call them in easy type-safe way using C++ magic.
There are two reasons for using run-time compilation:

* Kernels that have tunable parameters (block size, elements per thread, loop unroll factors, etc.) where the optimal configuration can only be determined at runtime since it depends dynamic factors such as the type of GPU and the problem size.

* Improve performance by injecting runtime values as compile-time constant values into kernel code (dimensions, array strides, weights, etc.).




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


The following C++ snippet shows how to use *Kernel Launcher* in host code::

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




Indices and tables
============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

