__global__ void stencil(int n, int *a, float *b) {
   int i = threadIdx.x + blockDim.x * blockIdx.x;
   if (i < n) {
       a[i] = 2 * i;
       b[i] = i + 100;
   }
}
