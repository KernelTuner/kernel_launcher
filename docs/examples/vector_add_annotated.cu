#pragma kernel tune(threads_per_block=32, 64, 128, 256, 512, 1024)
#pragma kernel tune(items_per_thread=1, 2, 4, 8)
#pragma kernel set(items_per_block=threads_per_block * items_per_thread)
#pragma kernel problem_size(n)
#pragma kernel block_size(threads_per_block)
#pragma kernel grid_divisor(items_per_block)
#pragma kernel buffers(C[n], A[n], B[n])
#pragma kernel tuning_key("vector_add_" + T)
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

