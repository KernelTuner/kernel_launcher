#include "kernel_launcher.h"

// Namespace alias.
namespace kl = kernel_launcher;

kl::KernelBuilder build_kernel() {
    kl::KernelBuilder builder("vector_add", "vector_add.cu");

    auto threads_per_block = builder.tune("block_size", {32, 64, 128, 256, 512, 1024});
    auto elements_per_thread = builder.tune("elements_per_thread", {1, 2, 4, 8});
    auto elements_per_block = threads_per_block * elements_per_thread;

    builder
        .tuning_key("vector_add_float")
        .problem_size(kl::arg0)
        .block_size(threads_per_block)
        .grid_divisors(threads_per_block * elements_per_thread)
        .template_args(kl::type_of<float>())
        .define("ELEMENTS_PER_THREAD", elements_per_thread);

    return builder;
}

int main() {
    kl::set_global_wisdom_directory("wisdom/");
    kl::set_global_capture_directory("captures/");

    // Define the kernel.
    kl::KernelBuilder builder = build_kernel();
    kl::WisdomKernel vector_add_kernel(builder);

    // Initialize CUDA memory. This is outside the scope of kernel_launcher.
    unsigned int n = 1000000;
    float *dev_A, *dev_B, *dev_C;
    /* cudaMalloc, cudaMemcpy, ... */

    // Launch the kernel!
    unsigned int problem_size = n;
    vector_add_kernel(n, dev_C, dev_A, dev_B);
    return 0;
}
