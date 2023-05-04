#pragma kernel tune(threads_per_block = 32, 64, 128, 256, 512, 1024)
#pragma kernel tune(items_per_thread = 1, 2, 3, 4)
#pragma kernel tune(tiling_strategy = 0, 1, 2)
#pragma kernel set(items_per_block = items_per_thread * threads_per_block)
#pragma kernel set(block_size = items_per_block)
#pragma kernel restriction(items_per_block <= 1024)
#pragma kernel problem_size(n)
#pragma kernel block_size(threads_per_block)
#pragma kernel grid_divisor(items_per_block)
#pragma kernel buffers(C[n], A[n], B[n])
#pragma kernel tuning_key("vector_add_" + T)
template<
    typename T,
    int block_size = 32,
    int items_per_thread = 2,
    int tiling_strategy = 2>
__global__ void vector_add(int n, T* C, const T* A, const T* B) {
    static_assert(
        tiling_strategy >= 0 && tiling_strategy <= 2,
        "invalid tiling strategy");

    for (int k = 0; k < items_per_thread; k++) {
        int i;

        // contiguous. thread processes items i, i+1, i+2, ...
        if (tiling_strategy == 0) {
            i = (blockIdx.x * block_size + threadIdx.x) * items_per_thread + k;
        }

        // block-strided. thread processes items i, i + block_size, i + 2*block_size
        else if (tiling_strategy == 1) {
            i = blockIdx.x * items_per_thread * block_size + threadIdx.x
                + k * block_size;
        }

        // grid-strided. thread processes items i, i + grid_size, i + 2 * grid_size
        else if (tiling_strategy == 2) {
            i = blockIdx.x * block_size + threadIdx.x
                + k * (gridDim.x * block_size);
        }

        if (i < n) {
            C[i] = A[i] + B[i];
        }
    }
}
