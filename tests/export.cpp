#include "kernel_launcher/export.h"

#include "catch.hpp"
#include "kernel_launcher/wisdom.h"
#include "test_utils.h"

using namespace kernel_launcher;

TEST_CASE("test export_tuning_file") {
    CUcontext ctx;
    KERNEL_LAUNCHER_CUDA_CHECK(cuInit(0));
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxCreate(&ctx, 0, 0));

    std::string assets_dir = assets_directory();

    SECTION("vector add") {
        KernelBuilder builder = build_vector_add_kernel();
        size_t n = 1024;
        std::vector<float> a(n);
        std::vector<float> b(n);
        std::vector<float> c_ref(n);
        std::vector<float> c(n);

        for (size_t i = 0; i < n; i++) {
            a[i] = 1;
            b[i] = float(i);
            c[i] = 0;
            c_ref[i] = a[i] + b[i];
        }

        export_tuning_file(
            assets_dir,
            "vector_add_key",
            builder,
            {uint32_t(n)},
            {type_of<int>(),
             type_of<float*>(),
             type_of<const float*>(),
             type_of<const float*>()},
            {KernelArg::for_scalar(int(n)).to_bytes(),
             KernelArg::for_array(c.data(), c.size()).to_bytes(),
             KernelArg::for_array(a.data(), a.size()).to_bytes(),
             KernelArg::for_array(b.data(), b.size()).to_bytes()},
            {KernelArg::for_scalar(int(n)).to_bytes(),
             KernelArg::for_array(c_ref.data(), c_ref.size()).to_bytes(),
             KernelArg::for_array(a.data(), a.size()).to_bytes(),
             KernelArg::for_array(b.data(), b.size()).to_bytes()});
    }

    SECTION("matmul") {
        KernelBuilder builder = build_matmul_kernel();
        size_t n = 1024;
        std::vector<float> a(n * n);
        std::vector<float> b(n * n);
        std::vector<float> c_ref(n * n);
        std::vector<float> c(n * n);

        auto a_fun = [](size_t i) { return (i % 7); };
        auto b_fun = [](size_t i) { return (i % 11); };

        for (size_t i = 0; i < n * n; i++) {
            a[i] = float(a_fun(i));
            b[i] = float(b_fun(i));
            c[i] = 0.0f;
        }

        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                int result = 0.0f;

                for (size_t k = 0; k < n; k++) {
                    result += a_fun(i * n + k) * b_fun(k * n + j);
                }

                c_ref[i * n + j] = float(result);
            }
        }

        export_tuning_file(
            assets_dir,
            "matmul_key",
            builder,
            {uint32_t(n), uint32_t(n)},
            {type_of<int>(),
             type_of<float*>(),
             type_of<const float*>(),
             type_of<const float*>()},
            {KernelArg::for_scalar(int(n)).to_bytes(),
             KernelArg::for_array(c.data(), c.size()).to_bytes(),
             KernelArg::for_array(a.data(), a.size()).to_bytes(),
             KernelArg::for_array(b.data(), b.size()).to_bytes()},
            {KernelArg::for_scalar(int(n)).to_bytes(),
             KernelArg::for_array(c_ref.data(), c_ref.size()).to_bytes(),
             KernelArg::for_array(a.data(), a.size()).to_bytes(),
             KernelArg::for_array(b.data(), b.size()).to_bytes()});
    }
}