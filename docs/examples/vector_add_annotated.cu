#pragma kernel_tuner tune(threads_per_block=32, 64, 128, 256, 512, 1024)
#pragma kernel_tuner tune(items_per_thread=1, 2, 4, 8)
#pragma kernel_tuner set(items_per_block=threads_per_block * items_per_thread)
#pragma kernel_tuner problem_size(n)
#pragma kernel_tuner block_size(threads_per_block)
#pragma kernel_tuner grid_divisor(items_per_block)
#pragma kernel_tuner buffers(C[n], A[n], B[n])
#pragma kernel_tuner tuning_key("vector_add_", T)
template <typename T, int items_per_thread=1>
__global__
void vector_add(int n, T* C, const T* A, const T* B) {
    for (int k = 0; k < items_per_thread; k++) {
        int i = blockIdx.x * items_per_thread * blockDim.x + k * blockDim.x + threadIdx.x;

        if (i < n) {
            C[i] = A[i] + B[i];
        }
    }
}

