#include "kernel_launcher/export.h"

#include <filesystem>

#include "catch.hpp"
#include "kernel_launcher/wisdom.h"
#include "nlohmann/json.hpp"
#include "test_utils.h"

using namespace kernel_launcher;

void compare_exports(
    const std::string& key,
    const std::string& output_dir,
    const std::string& ref_dir) {
    std::ifstream output_stream(path_join(output_dir, key + ".json"));
    auto output = nlohmann::ordered_json::parse(output_stream).flatten();

    std::ifstream ref_stream(path_join(ref_dir, key + ".json"));
    auto ref = nlohmann::ordered_json::parse(ref_stream).flatten();

    for (auto entry : ref.items()) {
        const std::string& k = entry.key();

        // environment is platform-dependent. skip it.
        if (k.find("/environment") == 0) {
            continue;
        }

        // File names contain random generate symbols. skip them
        if (k.find("/file") != std::string::npos
            || k.find("/reference_file") != std::string::npos) {
            continue;
        }

        INFO("check key=" << k);
        CHECK(output[k] == entry.value());
    }
}

TEST_CASE("test export_tuning_file", "[CUDA]") {
    CUcontext ctx;
    KERNEL_LAUNCHER_CUDA_CHECK(cuInit(0));
    KERNEL_LAUNCHER_CUDA_CHECK(cuCtxCreate(&ctx, 0, 0));

    std::string assets_dir = assets_directory();
    std::string tmp_dir = path_join(assets_dir, "temporary");

    // Create temporary directory and clear its contents
    std::filesystem::create_directory(tmp_dir);
    for (const auto& entry : std::filesystem::directory_iterator(tmp_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::filesystem::remove(entry);
    }

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
            tmp_dir,
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

        compare_exports("vector_add_key_1024", tmp_dir, assets_dir);
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
                int result = 0;

                // We use integer arithmetic here, otherwise just generating
                // the reference data would take a long time on the CPU.
                for (size_t k = 0; k < n; k++) {
                    result += a_fun(i * n + k) * b_fun(k * n + j);
                }

                c_ref[i * n + j] = float(result);
            }
        }

        export_tuning_file(
            tmp_dir,
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

        compare_exports("matmul_key_1024x1024", tmp_dir, assets_dir);
    }
}