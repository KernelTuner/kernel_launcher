.. highlight:: c++
   :linenothreshold: 2

Wisdom Files
============

In the previous example, we demonstrated how to compile a kernel by providing both a  ``KernelBuilder`` instance (describing the `blueprint` for the kernel) and a ``Config`` instance (describing the configuration of the tunable parameters).

However, determining the optimal configuration can often be challenging, as it depends on both the problem size and the specific type of GPU being used. 
To address this problem, Kernel Launcher provides a solution in the form of **wisdom files** (terminology borrowed from `FFTW <http://www.fftw.org/>`_).

To use the Kernel Launcher's wisdom files, we need to run the application twice. 
First, we **capture** the kernels that we want to tune, and then we use Kernel Tuner to tune those kernels. 
Second, when we run the application again, but this time the kernel configuration is **selected** from the wisdom file that was generated during the tuning process.

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


When using a ``WisdomKernel``, we need to set the **tuning key**. 
If no tuning key is specified, the default kernel name is used (in this case, ``vector_add``)
The tuning key is a string that uniquely identifies the kernel and is used to locate the corresponding wisdom file. 
If no wisdom file can be found, the default configuration is used (in this example, ``block_size=32`` and ``elements_per_thread=1``). 



.. literalinclude:: wisdom.cpp
   :lines: 25-26
   :lineno-start: 25
   
The following two lines of code set global configuration for the application.

The function ``set_global_wisdom_directory`` sets the directory where Kernel Launcher will search for wisdom files associated with a compiled kernel. 
In this example, the directory ``wisdom/`` is set as the wisdom directory, and Kernel Launcher will search for the file ``wisdom/vector_add_float.wisdom`` since ``vector_add_float`` is the tuning key.

The function ``set_global_capture_directory`` sets the directory where Kernel Launcher will store resulting files when capturing a kernel launch.

.. literalinclude:: wisdom.cpp
   :lines: 28-30
   :lineno-start: 28

These lines construct the ``KernelBuilder`` and pass it on to the ``WisdomKernel``.



Export the kernel
-----------------
.. highlight:: bash
   :linenothreshold: 1000

In order to tune the kernel, the first step is to capture the kernel launch.
To do so, we need to run the program with the environment variable ``KERNEL_LAUNCHER_CAPTURE`` set to the name of the kernel we want to capture::

    $ KERNEL_LAUNCHER_CAPTURE=vector_add ./main

This generates a file called ``vector_add_1000000.json`` in the directory set by ``set_global_capture_directory``.

Alternatively, it is possible to capture several kernels at once by using the wildcard ``*``.
For example, the following command export all kernels that are start with ``vector_``::

    $ KERNEL_LAUNCHER_CAPTURE=vector_* ./main

See :doc:`../env_vars` for an overview and description of additional environment variables.



Tune the kernel
---------------
To tune the kernel, run the Python script ``tune.py`` in the directory ``python/`` which uses `Kernel Tuner <https://kerneltuner.github.io/>`_ to tune the kernel.
To view all available options, use ``--help``.
For example, to spend 10 minutes tuning the kernel for the current GPU, use the following command::

    $ python tune.py captures/vector_add_1000000.json --output wisdom/ --time 10:00

To tune multiple kernels at once, use a wildcard::

    $ python tune.py captures/*.json --output wisdom/

If everything goes well, the script should run for ten minutes and eventually generate a file ``wisdom/vector_add_float.wisdom`` containing the tuning results.
Note that it is possible to tune the same kernel for different GPUs and problem sizes, and all results will be saved in the same wisdom file.
After tuning, the files in the ``captures/`` directory can be removed safely.



Import the wisdom
-----------------
To use the wisdom file, make sure that the file ``wisdom/vector_add_float.wisdom`` is available and simply rerun the application.
Now, when the program calls the ``vector_add_kernel`` function, Kernel Launcher finds the wisdom file and compiles the kernel given the optimal configuration.
You can check the debugging output to verify that the wisdom file has been found by defining the environment variable ``KERNEL_LAUNCHER_LOG=debug``::


    $ KERNEL_LAUNCHER_LOG=debug ./main

    KERNEL_LAUNCHER [DEBUG] reading wisdom file wisdom/vector_add_float.wisdom
    KERNEL_LAUNCHER [DEBUG] found configuration for kernel vector_add, device NVIDIA A100-PCIE-40GB, problem size (1000000, 1, 1): {"block_size": 128, "elements_per_thread": 4}
    KERNEL_LAUNCHER [DEBUG] compiling kernel (vector_add.cu)
