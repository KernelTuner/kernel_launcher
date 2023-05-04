Environment Variables
=====================

`Kernel Launcher` recognizes the following environment variables:

.. list-table::

   * - Key
     - Default
     - Description

   * - **KERNEL_LAUNCHER_CAPTURE**
     - ``_``

     - Kernels for which a tuning specification will be captured.
       The value should a comma-seperated list of kernel names.
       Additionally, an ``*`` can be used as a wild card.

       Examples:

       * ``foo,bar``: matches kernels ``foo`` and ``bar``.
       * ``vector_*``: matches kernels that start with ``vector``.
       * ``*_matrix_*``: matches kernels that contains ``matrix``.
       * ``*``: matches all kernels.

   * - **KERNEL_LAUNCHER_CAPTURE _FORCE**
     - ``_``
     - Same as the previous variable.
       However, while ``KERNEL_LAUNCHER_CAPTURE`` skips kernels that have already been tuned
       (i.e., a wisdom file was found), the ``KERNEL_LAUNCHER_CAPTURE_FORCE`` will force to always
       capture kernels regardless of whether wisdom files are available.

   * - **KERNEL_LAUNCHER_CAPTURE _SKIP**
     - ``0``
     - Set the number of kernel launches to skip before capturing a particular kernel.
       For example, if you set the value to ``3``, only the fourth launch will be captured since the
       first three launches will be skipped.

       Note that this option is applied on a `per-kernel basis`, which means that each individual kernel keeps its own skip counter.

   * - **KERNEL_LAUNCHER_LOG**
     - ``info``
     - Controls how much logging information is printed to stderr. There are three possible options:

       * ``debug``: Everything is logged.
       * ``info``: Only warnings and high-level information is logged.
       * ``warn``: Only warnings are logged.

   * -  **KERNEL_LAUNCHER_DIR**
     - ``.``
     - The directory were the tuning specifications will be stored. Defaults to the current working directory.

   * - **KERNEL_LAUNCHER_WISDOM**
     - ``.``
     - The default directory where wisdom files are located. Defaults to the current working directory.

   * - **KERNEL_LAUNCHER_INCLUDE**
     - ``.``
     - List of comma-seperate directories that are considered while compiling kernels when searching for header files.
