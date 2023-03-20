Pragma Kernels
===========================

In the previous examples, we demonstrated how a tunable kernel can be specified by defining a ``KernelBuilder`` instance in the host-side code.
While this API offers flexiblity, it can be cumbersome and requires keeping the kernel code in CUDA in sync with the host-side code in C++.

Kernel Launcher also provides a way to define kernel specifications directly in the CUDA code by using pragma directives to annotate the kernel code.
Although this method is less flexible than the ``KernelBuilder`` API, it is much more convenient and suitable for most CUDA kernels.


Source Code
-----------

The following code example shows valid CUDA kernel code containing pragma directives.
The ``#pragma`` annotations will be ignored by the ``nvcc`` compiler (but they may produce compiler warnings).


.. literalinclude:: vector_add_annotated.cu
   :lines: 1-20
   :lineno-start: 1


Code Explanation
----------------

The kernel contains the following ``pragma`` directives:

.. literalinclude:: vector_add_annotated.cu
   :lines: 1-2
   :lineno-start: 1

The tune directives specify the tunable parameters: ``threads_per_block`` and ``items_per_thread``.
Since ``items_per_thread`` is also the name of the template parameter, so it is passed to the kernel as a compile-time constant via this parameter.
The value of ``threads_per_block`` is not passed to the kernel but is used by subsequent pragmas.

.. literalinclude:: vector_add_annotated.cu
   :lines: 3-3
   :lineno-start: 3

The ``set`` directives defines a constant.
In this case, the constant ``items_per_block`` is defined as the product of ``threads_per_block`` and ``items_per_thread``.

.. literalinclude:: vector_add_annotated.cu
   :lines: 4-6
   :lineno-start: 4

The ``problem_size`` directive defines the problem size (as discussed in as discussed in :doc:`basic`), ``block_size`` specifies the thread block size, and ``grid_divisor`` specifies how the problem size should be divided to obtain the thread grid size.
Alternatively, ``grid_size`` can be used to specify the grid size directly.


.. literalinclude:: vector_add_annotated.cu
   :lines: 7-7
   :lineno-start: 7

The ``buffers`` directive specifies the size of each buffer (``A``, ``B``, and ``C``) as ``n`` elements to be known by Kernel Launcher.
This is necessary since raw pointers can be used for buffer arguments, for which size information may not be available.
If the ``buffers`` pragma is not specified, Kernel Launcher can still be used but it is not possible to capture kernel launches.

.. literalinclude:: vector_add_annotated.cu
   :lines: 8-8
   :lineno-start: 8

The ``tuning_key`` directive specifies the tuning key, which can be a concatenation of strings or variables.
In this example, the tuning key is ``"vector_add_" + T``, where ``T`` is the name of the type.


Host Code
---------

The below code shows how to call the kernel from the host in C++::

    #include "kernel_launcher/pragma.h"
    using namespace kl = kernel_launcher;

    void launch_vector_add(float* C, const float* A, const float* B) {
        kl::launch(
            kl::PragmaKernel("vector_add_annotated.cu", "vector_add", {"float"}),
            n, C, A, B
        );
    );


The ``PragmaKernel`` class implements the ``IKernelDescriptor`` interface, as described in :doc:`registry`.
This class reads the specified file, extracts the Kernel Launcher pragmas from the source code, and compiles the kernel.

The ``launch`` function launches the kernel and, as discussed in :doc:`registry`, it uses the default registry to cache kernel compilations.
This means that the kernel is only compiled once, even if the same kernel is called from different locations in the program.


List of pragmas
---------------

The table below lists the valid directives.

.. list-table::

   * - Directive
     - Description

   * - ``tune``
     - Add a new tunable variable.

   * - ``set``
     - Add a new variable.

   * - ``buffers``
     - Specify the size of buffer arguments. This directive may occur multiple times.

   * - ``tuning_key``
     - Specify the tuning key used to search for the corresponding wisdom file.

   * - ``problem_size``
     - An N-dimensional vector that indicates workload size.

   * - ``grid_size``
     - An N-dimensional vector that indicates the CUDA grid size.

   * - ``block_size``
     - An N-dimensional vector that indicates the CUDA thread block size.

   * - ``grid_divisor``
     - Alternative way of specifying the grid size. The problem size is divided by the grid divisors to obtain the grid dimensions.

   * - ``restriction``
     - Boolean expression that must evaluate to ``true`` for a kernel configuration to be valid.
