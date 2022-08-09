Environment Variables
=====================

`Kernel Launcher` recognizes the following environment variables:

* **KERNEL_LAUNCHER_TUNE** (default: ``0``):
  Kernels for which a tuning specification will be exported on the first call to the kernel.
  The value should a comma-seperated list of kernel names.
  Additionally, an ``*`` can be used as a wild card.

  Examples:

  * ``foo,bar``: matches kernels ``foo`` and ``bar``.
  * ``vector_*``: matches kernels that start with ``vector``.
  * ``*_matrix_*``: matches kernels that contains ``matrix``.
  * ``*``: matches all kernels.


* **KERNEL_LAUNCHER_WISDOM** (default: ``.``):
  The default directory where wisdom files are located. Defaults to the current working directory.

* **KERNEL_LAUNCHER_LOG** (default: ``info``):
  Controls how much logging information is printed to stderr. There are three possible options:

  * ``debug``: Everything is logged.
  * ``info``: Only warnings and high-level information is logged.
  * ``warn``: Only warnings are logged.

* **KERNEL_LAUNCHER_INCLUDE** (default: ``.``):
  List of comma-seperate directories that are considered while compiling kernels when searching for header files.
