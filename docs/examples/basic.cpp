#include "kernel_launcher.h"


int main() {
    // Namespace alias.
    namespace kl = kernel_launcher;
    
    // Create a kernel builder
    kl::KernelBuilder builder("vector_add", "vector_add_kernel.cu");
    
    // Define tunable parameters 
    auto threads_per_block = builder.tune("block_size", {32, 64, 128, 256, 512, 1024});
    auto elements_per_thread = builder.tune("elements_per_thread", {1, 2, 4, 8});
    
    // Define expressions
    auto elements_per_block = threads_per_block * elements_per_thread;
    
    // Define kernel properties
    builder
        .problem_size(kl::arg0)
        .block_size(threads_per_block)
        .grid_divisors(threads_per_block * elements_per_thread)
        .template_args(kl::type_of<float>())
        .define("ELEMENTS_PER_THREAD", elements_per_thread);

    // Define configuration
    kl::Config config;
    config.insert(threads_per_block, 32);
    config.insert(elements_per_thread, 2);

    // Compile kernel
    kl::Kernel<int, float*, const float*, const float*> vector_add_kernel;
    vector_add_kernel.compile(builder, config);
    
    // Initialize CUDA memory. This is outside the scope of kernel_launcher.
    int n = 1000000;
    float *dev_A, *dev_B, *dev_C;
    /* cudaMalloc, cudaMemcpy, ... */
        
    // Launch the kernel!
    vector_add_kernel.launch(n, dev_C, dev_A, dev_B);
}
