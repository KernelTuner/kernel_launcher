.. highlight:: c++
   :linenothreshold: 1

Basic Example
=============

On this page, we show a basic example of how to use `kernel_launcher`.
We first show the full source code and then go over the example line by line.


Code
----

.. literalinclude:: basic.cpp



Explanation
-----------

.. literalinclude:: basic.cpp
   :lines: 8-9
   :lineno-start: 8

First, we need to define a ``KernelBuilder`` instance.
A ``KernelBuilder`` is essentially a `blueprint` that describes the information required to compile the CUDA kernel.
The constructor takes the function name of the kernel and the `cu` file where the code is located.
Optionally, we can also give the kernel source as a third parameter.


.. literalinclude:: basic.cpp
   :lines: 11-13
   :lineno-start: 11

CUDA kernels often have tunable parameters that affect performance, such as block size, thread granularity, how registers to use, whether to use shared memory.
Here we define two tunable parameters: the number of threads per blocks and the number of elements processed per thread.


.. literalinclude:: basic.cpp
   :lines: 15-16
   :lineno-start: 15

The values returned by ``tune`` are placeholder expressions.
Expressions can be combined using C++ operators to create new expressions.
Note that ``elements_per_block`` here does not contain an actual value.
Instead, it is an abstract expression that, when the kernel is instantiated, evaluates to the product of ``threads_per_block`` and ``elements_per_thread``.

.. literalinclude:: basic.cpp
   :lines: 18-23
   :lineno-start: 18

Next, we define properties of the kernel such as the block size and template arguments
The value for these properties can be expressions as shown above.
The following properties are supported:

* ``block_size``: The block size as an `(x, y, z)` triplet.
* ``grid_divsor``: Used to calculate the size of the grid (i.e., number of blocks along each axis).
  For each kernel launch, the `problem size` is divided by the `divisors` to calculate the grid size.
* ``template_args``: Template arguments which can be either type names or integral values.
* ``define``: Define preprocessor macro.
* ``shared_memory``: The amount of shared memory used in bytes.
* ``compiler_flags``: Additional flags passed to the compiler.


.. literalinclude:: basic.cpp
   :lines: 25-28
   :lineno-start: 25

The configuration defines the values of the tunable parameters to be used for compilation.
Here, the `Config` instance is constructed manually, but it could also be loaded from file or a tuning database.

.. literalinclude:: basic.cpp
   :lines: 30-32
   :lineno-start: 30

Compiling a `Kernel` requires a `KernelBuilder` together with a `Config`.
The `Kernel` instance should be stored, for example, in a class and only compiled once during initialization.

.. literalinclude:: basic.cpp
   :lines: 39-43
   :lineno-start: 39

To launch the kernel, we first prepare a kernel launch for a specific problem size using `instantiate` and then call the kernel using `launch`.

Alternatively, it is also possible to use the short-hand form::

        // Launch the kernel!
        vector_add_kernel(problem_size)(n, dev_C, dev_A, dev_B);

To pass a CUDA stream use::

        // Launch the kernel!
        vector_add_kernel(stream, problem_size)(n, dev_C, dev_A, dev_B);

For 2D or 3D problems, pass additional problem sizes::

        // Launch the kernel!
        vector_add_kernel(stream, problem_size_x, problem_size_y)(n, dev_C, dev_A, dev_B);
