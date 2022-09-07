#include "kernel_launcher.h"

// Namespace alias.
namespace kl = kernel_launcher;

class VectorAddDescriptor: KernelDescriptor {
public:
    template <typename T>
    static VectorAddKernel for_type() {
        return VectorAddKernel(kl::type_of<T>());
    }

    VectorAddKernel(kl::TypeInfo t): element_type(t) {}

    std::string tuning_key() const override {
        return "vector_add_" + this->element_type.name();
    }

    kl::KernelBuilder build() const override {
        kl::KernelBuilder builder("vector_add", "vector_add.cu");

        auto threads_per_block = builder.tune("block_size", {32, 64, 128, 256, 512, 1024});
        auto elements_per_thread = builder.tune("elements_per_thread", {1, 2, 4, 8});
        auto elements_per_block = threads_per_block * elements_per_thread;

        builder
            .block_size(threads_per_block)
            .grid_divisors(threads_per_block * elements_per_thread)
            .template_args(element_type)
            .define("ELEMENTS_PER_THREAD", elements_per_thread);

        return builder;
    }

    bool equals(const KernelDescriptor& other) const override {
        if (auto p = dynamic_cast<const VectorAddKernel*>(&other)) {
            return this->element_type == p->element_type;
        }

        return false;
    }

    private:
        kl::TypeInfo element_type;
}

void main() {
    kl::set_global_wisdom_directory("wisdom/");
    kl::set_global_tuning_directory("tuning/");

    // Initialize CUDA memory. This is outside the scope of kernel_launcher.
    unsigned int n = 1000000;
    float *dev_A, *dev_B, *dev_C;
    /* cudaMalloc, cudaMemcpy, ... */

    // Launch the kernel!
    unsigned int problem_size = n;

    kl::default_registry()
        .lookup(VectorAddDescriptor::for_type<float>())
        .instantiate(problem_size)
        .launch(n, dev_C, dev_A, dev_B);

    // Or use the short equivalent syntax:
    kl::launch(VectorAddDescriptor::for_type<float>(), problem_size)(n, dev_C, dev_A, dev_B);
}
