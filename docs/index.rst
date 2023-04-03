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
   :alt: kernel launcher

**Kernel Launcher** is a C++ library that makes it easy to dynamically compile *CUDA* kernels at runtime (using `NVRTC <https://docs.nvidia.com/cuda/nvrtc/index.html>`_) and launching them in a type-safe manner using C++ magic. There are two main reasons for using runtime compilation:

* Kernels that have tunable parameters (block size, elements per thread, loop unroll factors, etc.) where the optimal configuration  depends on dynamic factors such as the GPU type and problem size.

* Improve performance by injecting runtime values as compile-time constant values into kernel code (dimensions, array strides, weights, etc.).


Kernel Tuner Integration
========================

.. image:: /kernel_tuner_integration.png
   :width: 670
   :alt: kernel launcher integration


Kernel Launcher's tight integration with `Kernel Tuner <https://kerneltuner.github.io/>`_ results in highly-tuned kernels, as visualized above. 
Kernel Launcher **captures** kernel launches within your application, which are then **tuned** by Kernel Tuner and saved as **wisdom** files. 
These files are processed by Kernel Launcher during execution to **compile** the tuned kernel at runtime.

See :doc:`examples/wisdom` for an example of how this works in practise.




Basic Example
=============

This sections hows a basic code example. See :ref:`example` for a more advance example.

Consider the following CUDA kernel for vector addition.
This kernel has a template parameter ``T`` and a tunable parameter ``ELEMENTS_PER_THREAD``.

.. literalinclude:: examples/vector_add.cu


The following C++ snippet shows how to use *Kernel Launcher* in host code:

.. literalinclude:: examples/index.cpp



Indices and tables
============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

