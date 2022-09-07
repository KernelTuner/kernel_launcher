.. highlight:: c++
   :linenothreshold: 2

Kernel Registry
===============

.. The kernel registry essentially acts like a global cache of compiled kernels.

In the previous example, we saw how to use wisdom files by creating ``WisdomKernel`` object.
This object will compile the kernel code on the first call and the keep the kernel loaded as long as the object exists.
Typically, one would define the ``WisdomKernel`` object as part of a class or as a global variable.

However, in certain scenarios, it is inconvenient or impractical to store ``WisdomKernel`` objects.
In these cases, it is possible to use the ``KernelRegistry``, that essentially acts like a global table of compiled kernel instances.


C++ source code
---------------

Consider the following code snippet:

.. literalinclude:: registry.cpp


Code Explanation
----------------

The code example consists of two parts.
In the first part, a class ``VectorAddDescriptor`` is defined.
In the second part, this class is searched in the global kernel registry.


Defining KernelDescriptor
^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: registry.cpp
   :lines: 6-45
   :lineno-start: 6

This part of the code defines a ``KernelDescriptor``:
a class that encapsulate the information required to compile a kernel.
This class should override three methods:

- ``tuning_key`` to obtain the tuning key
- ``build`` to instantiate a ``KernelBuilder``,
-  ``equals`` to check for equality with another ``KernelDescriptor``.

The last method is required since a kernel registry is essentially a hash table that maps ``KernelDescriptor`` objects to ``Kernel`` objects.
The ``equals`` method is used to check if two keys in the hash table are equivalent.


Using KernelRegistry
^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: registry.cpp
   :lines: 59-62
   :lineno-start: 59

Here, the vector-add kernel is searched in the registry and launched with the given arguments.
It is important to note that this code can be called multiple times from different parts of the program, but the kernel is only compiled once.


.. literalinclude:: registry.cpp
   :lines: 64-65
   :lineno-start: 64

Alternatively, it is possible to use the above short-hand syntax.
This syntax also make it is easy to replace the element type ``float`` to some other type such as ``int``::

    kl::launch(VectorAddDescriptor::for_type<int>(), problem_size)(n, dev_C, dev_A, dev_B);

It is even possible to define a templated function that passes type ``T`` on to ``VectorAddDescriptor``, for some extra template magic::

    template <typename T>
    void launch_vector_add(T* C, const T* A, const T* B) {
        kl::launch(VectorAddDescriptor::for_type<T>(), n)(n, C, A, B);
    }
