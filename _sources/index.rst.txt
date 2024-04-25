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
   env_vars
   license
   Github repository <https://github.com/KernelTuner/kernel_launcher>

Kernel Launcher
===============

.. image:: /logo.png
   :width: 670
   :alt: Kernel Launcher logo

**Kernel Launcher** is a C++ library designed to dynamically compile *CUDA* kernels at runtime (using `NVRTC <https://docs.nvidia.com/cuda/nvrtc/index.html>`_) and to launch them in a type-safe manner using C++ magic. Runtime compilation offers two significant advantages:

* Kernels that have tunable parameters (block size, elements per thread, loop unroll factors, etc.) where the optimal configuration  depends on dynamic factors such as the GPU type and problem size.

* Improve performance by injecting runtime values as compile-time constant values into kernel code (dimensions, array strides, weights, etc.).


Kernel Tuner Integration
========================

.. image:: /kernel_tuner_integration.png
   :width: 670
   :alt: Kernel Launcher and Kernel Tuner integration


The tight integration of **Kernel Launcher** with `Kernel Tuner <https://kerneltuner.github.io/>`_ ensures that kernels are highly optimized, as illustrated in the image above.
Kernel Launcher can **capture** kernel launches within your application at runtime.
These captured kernels can then be **tuned** by Kernel Tuner and the tuning results are saved as **wisdom** files. 
These wisdom files are used by Kernel Launcher during execution to **compile** the tuned kernel at runtime.


See :doc:`examples/wisdom` for an example of how this works in practise.




Basic Example
=============

This section presents a simple code example illustrating how to use the Kernel Launcher. 
For a more detailed example, refer to :ref:`example`.

Consider the following CUDA kernel for vector addition.
This kernel has a template parameter ``T`` and a tunable parameter ``ELEMENTS_PER_THREAD``.

.. literalinclude:: examples/vector_add.cu


The following C++ snippet demonstrates how to use the Kernel Launcher in the host code:

.. literalinclude:: examples/index.cpp



Indices and Tables
============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

