.. highlight:: c++
   :linenothreshold: 2

Wisdom Files
============

In the previous example, we saw how it was possible to compile a kernel by providing both a ``KernelBuilder`` instance (describing the `blueprint` for the kernel) and a ``Config`` instance (describing the configuration of the tunable parameters).

However, determining the optimal configuration is often difficult since it highly depends on both the `problem size` and the type of `GPU` being used.
`Kernel Launcher` offers a solution this problem in form of `wisdom` files (terminology borrowed from `FFTW <http://www.fftw.org/>`_).

Let's see this in action.


Source code
---------------

vector_add_kernel.cu
++++++++++++++++++++
.. literalinclude:: vector_add.cu

main.cpp
++++++++
.. literalinclude:: wisdom.cpp


Code Explanation
----------------

Notice how this example is similar to the previous example, with some minor differences such that ``kl::Kernel`` has been replaced by ``kl::WisdomKernel``.
We now highlight the important lines of this code example.

.. literalinclude:: wisdom.cpp
   :lines: 6-22
   :lineno-start: 6

This function creates a ``KernelBuilder`` object.

.. literalinclude:: wisdom.cpp
   :lines: 13-14
   :lineno-start: 13

Using a ``WisdomKernel`` requires the `tuning key` to be set.
If no tuning key is specified, then the kernel name is taken by default (which, in this example is ``vector_add``).
The `tuning key` is a string that uniquely identifies this kernel and is used to search for the kernel's wisdom file.
If no wisdom file can be been found, the default configuration is chosen (in this case, that will be ``block_size=32,elements_per_thread=1``).

.. literalinclude:: wisdom.cpp
   :lines: 25-26
   :lineno-start: 25

These two lines set global settings for the application.

The function ``set_global_wisdom_directory`` sets the directory containing the wisdom files.
When a kernel is compiled, this is where Kernel Launcher will search for the associated wisdom file.
In this example, Kernel Launcher will search for the file ``wisdom/vector_add_float.wisdom`` since ``wisdim/`` is the
wisdom directory and ``vector_add_float`` is the tuning key.

The function ``set_global_capture_directory`` sets the directory for capture files.
When capturing a kernel launch, this is where Kernel Launcher`` will store the resulting files.

.. literalinclude:: wisdom.cpp
   :lines: 28-30
   :lineno-start: 28

These lines construct the ``KernelBuilder`` and pass it on to the ``WisdomKernel``.

Export the kernel
-----------------
.. highlight:: bash
   :linenothreshold: 1000

To tune the kernel, we first need to capture the kernel launch.
To do this, we run the program with the environment variable ``KERNEL_LAUNCHER_CAPTURE``::

    $ KERNEL_LAUNCHER_CAPTURE=vector_add ./main

This generates a file ``vector_add_1000000.json`` in the directory set by ``set_global_capture_directory``.

Alternatively, it is possible to capture several kernels at once by using the wildcard ``*``.
For example, the following command export all kernels that are start with ``vector_``::

    $ KERNEL_LAUNCHER_CAPTURE=vector_* ./main

See :doc:`../env_vars` for an overview and description of additional environment variables.

Tune the kernel
---------------
To tune the kernel, run the Python script ``tune.py`` in the directory ``python/`` which internally uses ``kernel_tuner`` to tune the kernel.
Use ``--help`` to get an overview of all available options.
For example, to tune spend 10 minutes (600 seconds) tuning the kernel for the current kernel, use the following command::

    $ python tune.py tuning/vector_add_1000000.json --output wisdom/ --time 600

To tune multiple kernels all at once, use a wildcard::

    $ python tune.py tuning/*.json --output wisdom/

If everything goes well, the script should run for ten minutes and eventually create a file ``wisdom/vector_add_float.wisdom`` containing the tuning results.
Note that it is possible to tune the same kernel for different devices and problem sizes, for which all results will be stored in the same wisdom file.
After tuning, the files in the ``tuning/`` directory can safely be deleted.



Import the wisdom
-----------------
To use the wisdom file, make sure that the file ``wisdom/vector_add_float.wisdom`` is available and simply rerun the program.
Now, when running the program, on the first call to ``vector_add_kernel``, the kernel finds the wisdom file and compiles the kernel given the optimal configuration.
To confirm that wisdom file has indeed been found, check the debugging output by define the environment variable ``KERNEL_LAUNCHER_LOG=debug``::


    $ KERNEL_LAUNCHER_LOG=debug ./main

    KERNEL_LAUNCHER [DEBUG] reading wisdom file wisdom/vector_add_float.wisdom
    KERNEL_LAUNCHER [DEBUG] found configuration for kernel vector_add, device NVIDIA A100-PCIE-40GB, problem size (1000000, 1, 1): {"block_size": 128, "elements_per_thread": 4}
    KERNEL_LAUNCHER [DEBUG] compiling kernel (vector_add.cu)
