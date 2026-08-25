/*
 * Sum of a float array. The same kernel is used for both passes of a two-pass
 * reduction: pass 1 launches one block per resident slot and writes one
 * partial per block, pass 2 launches a single block over those partials.
 *
 * Identifiers not defined here are supplied by Kernel Launcher:
 *
 *   block_size        threads per block (multiple of 32, power of two)
 *   blocks_per_sm     resident blocks per SM; also sets the grid size
 *   items_per_thread  loads issued per thread per loop iteration
 *   vector_size       1, 2 or 4 - width of each load
 *   use_shuffle       0 = shared-memory tree, 1 = warp shuffles
 *
 * `in` must be 16-byte aligned when vector_size is 4. cudaMalloc guarantees
 * this, as does the partials buffer written by pass 1.
 */

#if vector_size == 1
typedef float vec_t;
__device__ __forceinline__ float vec_sum(vec_t v) {
    return v;
}
#elif vector_size == 2
typedef float2 vec_t;
__device__ __forceinline__ float vec_sum(vec_t v) {
    return v.x + v.y;
}
#elif vector_size == 4
typedef float4 vec_t;
__device__ __forceinline__ float vec_sum(vec_t v) {
    return (v.x + v.y) + (v.z + v.w);
}
#else
    #error "vector_size must be 1, 2 or 4"
#endif

__global__ __launch_bounds__(block_size, blocks_per_sm) void reduce(
    int n,
    float* __restrict__ out,
    const float* __restrict__ in) {
    const unsigned tid = threadIdx.x;
    const size_t gid = (size_t)blockIdx.x * block_size + tid;
    const size_t stride = (size_t)block_size * gridDim.x;
    const size_t n_vec = (size_t)n / vector_size;
    const vec_t* in_vec = reinterpret_cast<const vec_t*>(in);

    float sum = 0.0f;
    size_t i = gid;

    // Bulk: items_per_thread independent loads in flight at once.
    for (; i + (size_t)(items_per_thread - 1) * stride < n_vec;
         i += (size_t)items_per_thread * stride) {
#pragma unroll
        for (int k = 0; k < items_per_thread; k++) {
            sum += vec_sum(in_vec[i + (size_t)k * stride]);
        }
    }

    // Whatever is left of the vectorised range.
    for (; i < n_vec; i += stride) {
        sum += vec_sum(in_vec[i]);
    }

    // Fewer than vector_size elements at the very end.
    for (size_t j = n_vec * vector_size + gid; j < (size_t)n; j += stride) {
        sum += in[j];
    }

#if use_shuffle
    // Uniform across the block: every thread takes part in every shuffle.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    }

    __shared__ float warp_sums[block_size / 32];
    if (tid % 32 == 0) {
        warp_sums[tid / 32] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
    #pragma unroll
        for (int k = 0; k < block_size / 32; k++) {
            total += warp_sums[k];
        }
        out[blockIdx.x] = total;
    }
#else
    __shared__ float sdata[block_size];
    sdata[tid] = sum;
    __syncthreads();

    #pragma unroll
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
#endif
}
