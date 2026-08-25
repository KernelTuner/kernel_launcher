#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernel_launcher.h"

namespace kl = kernel_launcher;

static void cuda_check(cudaError_t code) {
    if (code != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") + cudaGetErrorString(code));
    }
}

static std::string kernel_directory() {
    std::string self = __FILE__;
    return self.substr(0, self.rfind('/') + 1);
}

// The filter dimensions are host values, not kernel arguments. Passing them in
// here bakes them into the compiled kernel, which is what lets the filter loops
// unroll completely. The cost is a separate compilation per filter shape.
static kl::KernelBuilder build_convolution(int fh, int fw) {
    kl::KernelBuilder builder(
        "convolution",
        kernel_directory() + "convolution_kernel.cu");

    // A kernel compiled for a 3x3 filter has nothing to learn from a 9x9 one,
    // so give each shape its own tuning key and therefore its own wisdom.
    builder.tuning_key(
        "convolution_" + std::to_string(fh) + "x" + std::to_string(fw));

    auto bx = builder.tune("block_size_x", {16, 32, 64}, 32);
    auto by = builder.tune("block_size_y", {1, 2, 4, 8, 16}, 4);
    auto tx = builder.tune("tile_size_x", {1, 2, 4}, 2);
    auto ty = builder.tune("tile_size_y", {1, 2, 4}, 2);
    auto shmem = builder.tune("use_shmem", {0, 1}, 1);

    builder.restriction(bx * by <= kl::DEVICE_MAX_THREADS_PER_BLOCK);
    // Conservative: applied to use_shmem=0 configurations too, which do not
    // actually allocate the input tile.
    builder.restriction(
        4 * ((by * ty + fh - 1) * (bx * tx + fw - 1) + fh * fw) <= 48 * 1024);

    auto [h, w, dst, src, filt] = kl::args<5>();

    // define parameters
    builder.define(bx)
        .define(by)
        .define(tx)
        .define(ty)
        .define(shmem)
        .define("filter_height", std::to_string(fh))
        .define("filter_width", std::to_string(fw))
        .problem_size(w, h)  // x covers columns, y covers rows
        .block_size(bx, by)
        .grid_divisors(bx * tx, by * ty);

    // define the buffers for captures
    builder.buffers(dst[h * w], src[h * w], filt[fh * fw]);

    return builder;
}

int main() {
    const int h = 2160, w = 3840;
    const int fh = 5, fw = 5;

    std::vector<float> src(size_t(h) * w), dst(size_t(h) * w);
    std::vector<float> filt(size_t(fh) * fw);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : src)
        v = dist(rng);
    for (auto& v : filt)
        v = dist(rng);

    float *src_dev, *dst_dev, *filt_dev;
    cuda_check(cudaSetDevice(0));
    cuda_check(cudaMalloc(&src_dev, sizeof(float) * src.size()));
    cuda_check(cudaMalloc(&dst_dev, sizeof(float) * dst.size()));
    cuda_check(cudaMalloc(&filt_dev, sizeof(float) * filt.size()));
    cuda_check(cudaMemcpy(
        src_dev,
        src.data(),
        sizeof(float) * src.size(),
        cudaMemcpyDefault));
    cuda_check(cudaMemcpy(
        filt_dev,
        filt.data(),
        sizeof(float) * filt.size(),
        cudaMemcpyDefault));

    // build wisdom kernel.
    kl::WisdomKernel convolution(build_convolution(fh, fw));

    // run convolution
    convolution(
        h,
        w,
        kl::cuda_span(dst_dev, dst.size()),
        kl::cuda_span<const float>(src_dev, src.size()),
        kl::cuda_span<const float>(filt_dev, filt.size()));

    // wait until completion
    cuda_check(cudaDeviceSynchronize());
    cuda_check(cudaMemcpy(
        dst.data(),
        dst_dev,
        sizeof(float) * dst.size(),
        cudaMemcpyDefault));

    // check a handful of pixels, including ones on the border.
    double worst = 0;
    for (int t = 0; t < 256; t++) {
        int y = (t < 32) ? int(rng() % 2) * (h - 1) : int(rng() % h);
        int x = (t < 32) ? int(rng() % 2) * (w - 1) : int(rng() % w);
        double ref = 0;
        for (int i = 0; i < fh; i++) {
            for (int j = 0; j < fw; j++) {
                int gy = y + i - fh / 2, gx = x + j - fw / 2;
                if (gy >= 0 && gy < h && gx >= 0 && gx < w) {
                    ref += double(src[size_t(gy) * w + gx])
                        * double(filt[size_t(i) * fw + j]);
                }
            }
        }
        worst = std::max(worst, std::abs(ref - dst[size_t(y) * w + x]));
    }
    std::printf("max error over 256 samples: %g\n", worst);

    cudaFree(src_dev);
    cudaFree(dst_dev);
    cudaFree(filt_dev);
    return worst < 1e-3 ? 0 : 1;
}
