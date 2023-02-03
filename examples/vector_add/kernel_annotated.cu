#pragma kernel_tuner tune(block_size = 32, 64, 128, 256, 512, 1024)
#pragma kernel_tuner tune(elements_per_thread = 1, 2, 3, 4)
#pragma kernel_tuner tune(tiling_strategy = 0, 1, 2)
#pragma kernel_tuner block_size(block_size)
#pragma kernel_tuner problem_size(n)
#pragma kernel_tuner grid_divisor(block_size* elements_per_thread)
#pragma kernel_tuner restriction(block_size* elements_per_thread >= 32)
template<
    typename T,
    int block_size = 32,
    int elements_per_thread = 2,
    int tiling_strategy = 2>
__global__ void vector_add(int n, T* C, const T* A, const T* B) {
    static_assert(
        tiling_strategy >= 0 && tiling_strategy <= 2,
        "invalid tiling strategy");

    for (int k = 0; k < elements_per_thread; k++) {
        int i;

        // contiguous. thread processes element i, i+1, i+2, ...
        if (tiling_strategy == 0) {
            i = (blockIdx.x * block_size + threadIdx.x) * elements_per_thread
                + k;
        }

        // block-strided. thread processes elements i, i + block_size, i + 2*block_size
        else if (tiling_strategy == 1) {
            i = blockIdx.x * elements_per_thread * block_size + threadIdx.x
                + k * block_size;
        }

        // grid-strided. thread processes elements i, i + grid_size, i + 2 * grid_size
        else if (tiling_strategy == 2) {
            i = blockIdx.x * block_size + threadIdx.x
                + k * (gridDim.x * block_size);
        }

        if (i < n) {
            C[i] = A[i] + B[i];
        }
    }
}
