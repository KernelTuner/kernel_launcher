#include "kernel_launcher.h"

int main() {
    // Namespace alias.
    namespace kl = kernel_launcher;

    // Create a kernel builder
    kl::KernelBuilder builder("vector_add", "vector_add_kernel.cu");

    // Define the variables that can be tuned for this kernel.
    kl::ParamExpr threads_per_block = builder.tune("block_size", {32, 64, 128, 256, 512, 1024});
    kl::ParamExpr elements_per_thread = builder.tune("elements_per_thread", {1, 2, 4, 8});

    // Set kernel properties such as block size, grid divisor, template arguments, etc.
    builder
        .problem_size(kl::arg0)
        .block_size(threads_per_block)
        .grid_divisors(threads_per_block * elements_per_thread)
        .template_args(kl::type_of<float>())
        .define("ELEMENTS_PER_THREAD", elements_per_thread);

    // Define the kernel
    kl::WisdomKernel vector_add_kernel(builder);

    // Initialize CUDA memory. This is outside the scope of kernel_launcher.
    unsigned int n = 1000000;
    float *dev_A, *dev_B, *dev_C;
    /* cudaMalloc, cudaMemcpy, ... */

    // Launch the kernel! Note that kernel is compiled on the first call.
    // The grid size and block size do not need to be specified, they are
    // derived from the kernel specifications.
    vector_add_kernel(n, dev_C, dev_A, dev_B);
}
