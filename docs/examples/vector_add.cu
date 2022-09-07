template <typename T>
__global__
void vector_add(int n, T* C, const T* A, const T* B) {
    for (int k = 0; k < ELEMENTS_PER_THREAD; k++) {
        int i = blockIdx.x * ELEMENTS_PER_THREAD * blockDim.x + k * blockDim.x + threadIdx.x;

        if (i < n) {
            C[i] = A[i] + B[i];
        }
    }
}
