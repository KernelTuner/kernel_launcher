#include "kernel_launcher.h"

int main() {
    // Namespace alias.
    namespace kl = kernel_launcher;

    // Create a kernel builder
    kl::KernelBuilder builder("vector_add", "vector_add_kernel.cu");

    // Define the variables that can be tuned for this kernel.
    kl::ParamExpr threads_per_block = builder.tune("block_size", {32, 64, 128, 256, 512, 1024});
    kl::ParamExpr elements_per_thread = builder.tune("elements_per_thread", {1, 2, 4, 8});

    // `threads_per_block` and `elements_per_thread` are expression placeholders.
    // Expressions can be combined using operators to create new expressions.
    auto elements_per_block = elements_per_thread * threads_per_block;

    // Set kernel properties such as block size, grid divisor, template arguments, etc.
    builder
        .block_size(threads_per_block)
        .grid_divisors(elements_per_block)
        .template_args(kl::type_of<float>())
        .define("ELEMENTS_PER_THREAD", elements_per_thread);

    // Create a configuration. Here we use hard-coded values, but these values
    // can also be loaded from a file or database with tuning results.
    kl::Config config;
    config.insert(threads_per_block, 128);
    config.insert(elements_per_thread, 4);

    // Compile the kernel for the given configuration. This evaluates all
    // expressions and calls NVRTC to perform the actual compilation.
    kl::Kernel<unsigned int, float*, const float*, const float*> vector_add_kernel;
    vector_add_kernel.compile(builder, config);

    // Initialize device memory. This is outside the scope of kernel_launcher.
    // Here we just allocate memory using cudaMalloc.
    unsigned int n = 1000000;
    float *dev_A, *dev_B, *dev_C;
    cudaMalloc((void**)(&dev_A), sizeof(float) * n);
    cudaMalloc((void**)(&dev_B), sizeof(float) * n);
    cudaMalloc((void**)(&dev_C), sizeof(float) * n);
    cudaSetDevice(0);

    // Launch the kernel! Note that the block size and grid size are
    // derived from the problem size, thus they do not need to be specified
    unsigned int problem_size = n;
    vector_add_kernel(problem_size)(n, dev_C, dev_A, dev_B);
}