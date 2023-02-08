Pragma Kernels
===========================

In the previous examples, we saw how it was possible to specify a tunable kernel by defining a ``KernelBuilder`` instance in the host-side code.
While this API offers flexibility, it is also somewhat cumbersome and it requires keeping the actual kernel code in CUDA in sync with the host-side code in C++.

Kernel Launcher also offers a way to define the kernel specifications inside the actual CUDA code by annotating the kernel code with directives.
While this method is less flexible than the ``KernelBuilder`` API, it is a lot more convenient and should be usable for the majority of CUDA kernels.


Source Code
-----------

Below shows the CUDA kernel code.
This is valid regular CUDA code since the ``#pragma`` will be ignored by the ``nvcc`` compiler (although they might cause compiler warnings).

.. literalinclude:: vector_add_annotated.cu
   :lines: 1-20
   :lineno-start: 1


Code Explanation
----------------

The kernel contains the following ``pragma`` directives:

.. literalinclude:: vector_add_annotated.cu
   :lines: 1-2
   :lineno-start: 1

The ``tune`` directives defines the tunable parameters.
In this case, there are two parameters: ``threads_per_block`` and ``items_per_thread``.
Since ``items_per_thread`` is also the name of template parameter (line 9), it is passed to the kernel as compile-time constant to the kernel via this parameter.
The value of ``threads_per_block`` is not passed to the kernel but is used by subsequent pragmas.

.. literalinclude:: vector_add_annotated.cu
   :lines: 3-3
   :lineno-start: 3

The ``set`` directives defines a constant.
In this case, the constant ``items_per_block`` is defined as the product of ``threads_per_block`` and ``items_per_thread``.

.. literalinclude:: vector_add_annotated.cu
   :lines: 4-6
   :lineno-start: 4

The above lines specify information required to launch the kernel.
The ``problem_size`` defines the problem size as discussed in :doc:`basic`.
The ``block_size`` specifies the thread block size and ``grid_divisors`` specifies how the problem size should be divided to obtain the thread grid size.
Alternatively, it is possible to specify the grid size directly using the ``grid_size`` directive.

.. literalinclude:: vector_add_annotated.cu
   :lines: 7-7
   :lineno-start: 7

The above line specifies that the kernel arguments ``A``, ``B``, and ``C`` are buffers each having ``n`` elements.
This is required since Kernel Launcher requires the size of each buffer to be known, but the kernel could be called with raw pointers for which no size information is available.
If the ``buffers`` pragma is not specified, Kernel Launcher can still be used but it is not possible to capture kernel launches.

.. literalinclude:: vector_add_annotated.cu
   :lines: 8-8
   :lineno-start: 8

The ``tuning_key`` pragma specifies the tuning key.
All arguments given to this pragma will be concatenated and these arguments can be either strings or variables.
In this example, the tuning key is ``"vector_add_" + T`` where ``T`` is the name of the type.


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
This class will read the specified file, extract the Kernel Launcher pragmas from the source code, and compile the kernel.

The ``launch`` function launches the kernel and, as discussed in :doc:`registry`, it uses the default registry to cache kernel compilations.
This means that the kernel is only compiled once, even if the same kernel is called from different locations in the program.

