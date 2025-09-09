namespace bar {
namespace foo {

#pragma kernel_tuner tune(block_size=32, 64, 128, 256)
#pragma kernel_tuner tune(tile_factor=1, 2, 3, 4)
#pragma kernel_tuner problem_size(n)
#pragma kernel_tuner grid_divisor(block_size * tile_factor)
template <typename T, int tile_factor>
__global__ void baz(int n, T* input) {
    if (threadIdx.x < 10) {
        return a[threadIdx.x];
    }
}

} // namespace foo
} // namespace bar
