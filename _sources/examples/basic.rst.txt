.. highlight:: c++
   :linenothreshold: 1

Basic Example
=============

On this page, we show a basic example of how to use `Kernel Launcher`.
We first show the full source code and then go over the example line by line.


Source Code
-----------


vector_add_kernel.cu
++++++++++++++++++++
.. literalinclude:: vector_add.cu

main.cpp
++++++++
.. literalinclude:: basic.cpp



Code Explanation
----------------

.. literalinclude:: basic.cpp
   :lines: 8-9
   :lineno-start: 8

First, we need to define a ``KernelBuilder`` instance.
A ``KernelBuilder`` is essentially a `blueprint` that describes the information required to compile the CUDA kernel.
The constructor takes the name of the kernel function and the `.cu` file where the code is located.
Optionally, we can also provide the kernel source as the third parameter.


.. literalinclude:: basic.cpp
   :lines: 11-13
   :lineno-start: 11

CUDA kernels often have tunable parameters that can impact their performance, such as block size, thread granularity, register usage, and the use of shared memory. 
Here, we define two tunable parameters: the number of threads per blocks and the number of elements processed per thread.



.. literalinclude:: basic.cpp
   :lines: 15-16
   :lineno-start: 15

The values returned by ``tune`` are placeholder objecs.
These objects can be combined using C++ operators to create new expressions objects.
Note that ``elements_per_block`` does not actually contain a specific value;
instead, it is an abstract expression that, upon kernel instantiation, is evaluated as the product of ``threads_per_block`` and ``elements_per_thread``.

.. literalinclude:: basic.cpp
   :lines: 18-24
   :lineno-start: 18

Next, we define properties of the kernel such as block size and template arguments. 
These properties can take on expressions, as demonstrated above. 
The full list of properties is documented as :doc:`api/KernelBuilder`
The following properties are supported:

* ``problem_size``: This is an N-dimensional vector that represents the size of the problem. In this case, is one-dimensional and ``kl::arg0`` means that the size is specified as the first kernel argument (`argument 0`).
* ``block_size``: A triplet ``(x, y, z)`` representing the block dimensions.
* ``grid_divsor``: This property is used to calculate the size of the grid (i.e., the number of blocks along each axis). For each kernel launch, the problem size is divided by the divisors to calculate the grid size. In other words, this property expresses the number of elements processed per thread block.
* ``template_args``: This property specifies template arguments, which can be type names and integral values.
* ``define``: Define preprocessor constants.
* ``shared_memory``: Specify the amount of shared memory required, in bytes.
* ``compiler_flags``: Additional flags passed to the compiler.


.. literalinclude:: basic.cpp
   :lines: 26-29
   :lineno-start: 26

The configuration defines the values of the tunable parameters to be used for compilation.
Here, the ``Config`` instance is constructed manually, but it could also be loaded from file or a tuning database.

.. literalinclude:: basic.cpp
   :lines: 31-33
   :lineno-start: 31

Compiling a ``Kernel`` requires a ``KernelBuilder`` together with a ``Config``.
The ``Kernel`` instance should be stored, for example, in a class and only compiled once during initialization.

.. literalinclude:: basic.cpp
   :lines: 39-43
   :lineno-start: 39

To launch the kernel, we simply call ``launch``.

Alternatively, it is also possible to use the short-hand form::

        // Launch the kernel!
        vector_add_kernel(n, dev_C, dev_A, dev_B);

To pass a CUDA stream use::

        // Launch the kernel!
        vector_add_kernel(stream, n, dev_C, dev_A, dev_B);

For 2D or 3D problems, we must configure the ``KernelBuilder`` with additional dimensions::

        // Define kernel properties
        builder.problem_size(kl::arg0, kl::arg1, 100);
