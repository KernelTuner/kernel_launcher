#include "kernel_launcher/export.h"

#include "catch.hpp"
#include "kernel_launcher/wisdom.h"

using namespace kernel_launcher;

TEST_CASE("test export_tuning_file") {
    std::string assets_dir = __FILE__;
    assets_dir = assets_dir.substr(0, assets_dir.rfind('/'));
    assets_dir += "/assets";

    std::string source = R"(
    template <typename T>
    __global__
    void vector_add(int n, T* C, const T* A, const T* B) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) C[i] = A[i] + B[i];
    }
    )";

    KernelBuilder builder("vector_add", KernelSource("vector_add.cu", source));
    auto b = builder.tune("block_size_x", {32, 64, 128, 256});

    builder.template_arg(type_of<float>());
    builder.grid_divisors(b);

    export_tuning_file(
        assets_dir,
        "vector_add_key",
        builder,
        {1024},
        {type_of<int>(),
         type_of<float*>(),
         type_of<const float*>(),
         type_of<const float*>()},
        {KernelArg::for_scalar<int>(4).to_bytes(), {}, {}, {}});
}