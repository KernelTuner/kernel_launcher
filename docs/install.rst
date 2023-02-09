Installation
============

There are three ways of using *Kernel Launcher*:

* Using CMake (recommended)
* Compile a static library
* Header-only library (discouraged).


CMake dependency (Recommended)
------------------------------

If your project already uses CMake, integrating *Kernel Launcher* should be straightforward.

First, check out the repostitory.

.. code-block:: bash

    git clone https://github.com/KernelTuner/kernel_launcher/


Second, add the following lines to your ``CMakeLists.txt``::

    add_subdirectory(kernel_launcher)
    target_link_libraries(<your-application> PRIVATE kernel_launcher)



Static library
--------------

An alternative is to build a static library that can be linked to your project.

.. code-block:: bash

   git clone https://github.com/KernelTuner/kernel_launcher/
   cd kernel_launcher
   cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
   cmake --build build

You can now link your applications to `build/libkernel_launcher.a`.
Further steps depend on your particular build system.



Header-only library
-------------------

.. warning::
    Using *Kernel Launcher* as a header-only library, while possible, is discouraged: it signficantly increases build times and not all functionality is supported.
    However, if there is no other way, it is supported.


TODO
