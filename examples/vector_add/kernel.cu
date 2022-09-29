
template<int block_size, int elements_per_thread, int tiling_strategy>
__global__ void vector_add(int n, float* C, const float* A, const float* B) {
    static_assert(
        tiling_strategy >= 0 && tiling_strategy < 3,
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