Wisdom Files
============

In the previous example, we saw how it is possible to compile a kernel by providing both a ``KernelBuilder`` instance (describing `blueprint` for the kernel) and a ``Config`` instance (describing the configuration of the tunable parameters).

However, determining the optimal configuration is often difficult since it highly depends both on the `problem size` and the type of `GPU` being used.
`Kernel Launcher` offers a solution this problem in form of `wisdom` files (terminology borrowed from `FFTW <http://www.fftw.org/>`_).

Let's see this in action.


C++ source code
---------------

The following snippet show an example:

.. literalinclude:: wisdom.cpp


Notice how this example is similar to the previous example, except ``kl::Kernel`` has been replaced by ``kl::WisdomKernel``.
On the first call this kernel, the kernel searches for the wisdom file for the key ``vector_add`` and compiles the kernel for the given ``problem_size`` and the current GPU.
If no wisdom file has been found, the default configuration is chosen (in this case, that will be ``block_size=32,elements_per_thread=1``).



Export the kernel
-----------------
To tune the kernel, we first need to export the tuning specifications. To do this, we run the program with the environment variable ``KERNEL_LAUNCHER_TUNE=vector_add``::

    KERNEL_LAUNCHER_TUNE=vector_add ./main

This generates a file ``vector_add_1000000.json`` in the directory set by ``set_global_tuning_directory``.


Tune the kernel
---------------
TODO: Using kernel tuner


Import the wisdom
-----------------
After tuning the kernel and obtaining the wisdom file, we place this wisdom file in the directory specified by ``set_global_wisdom_directory``.
Now, when running the program, on the first call to ``vector_add_kernel``, the kernel finds the wisdom file and compiles the kernel given the optimal configuration.

To confirm that wisdom file has indeed been found, check the debugging output by define the environment variable ``KERNEL_LAUNCHER_LOG=debug``.


