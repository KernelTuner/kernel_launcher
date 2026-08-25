/*
 * 2D convolution with zero padding: out and in are both h x w, the filter is
 * filter_height x filter_width and is anchored at its centre.
 *
 *   out[y][x] = sum_i sum_j in[y + i - PAD_Y][x + j - PAD_X] * filter[i][j]
 *
 * Identifiers not defined here are supplied by Kernel Launcher as
 * preprocessor definitions:
 *
 *   block_size_x, block_size_y    thread block dimensions
 *   tile_size_x,  tile_size_y     output pixels computed per thread
 *   use_shmem                     0 = read the input straight from global
 *                                 memory, 1 = stage a tile in shared memory
 *   filter_height, filter_width   filter dimensions, baked in by the host so
 *                                 the filter loops unroll completely
 */

#define TILE_W (block_size_x * tile_size_x)  // output columns per block
#define TILE_H (block_size_y * tile_size_y)  // output rows per block
#define HALO_W (filter_width - 1)
#define HALO_H (filter_height - 1)
#define PAD_X  (filter_width / 2)
#define PAD_Y  (filter_height / 2)

__global__ __launch_bounds__(block_size_x* block_size_y) void convolution(
    int h,
    int w,
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ filter) {
    // The filter is tiny and every thread reads all of it, so it is always
    // worth staging. The input tile is the part that is up for tuning.
    __shared__ float f[filter_height][filter_width];
#if use_shmem
    __shared__ float tile[TILE_H + HALO_H][TILE_W + HALO_W];
#endif

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x0 = blockIdx.x * TILE_W;  // first output column of this block
    const int y0 = blockIdx.y * TILE_H;  // first output row of this block

    for (int i = ty; i < filter_height; i += block_size_y) {
        for (int j = tx; j < filter_width; j += block_size_x) {
            f[i][j] = filter[i * filter_width + j];
        }
    }

#if use_shmem
    // tile[i][j] holds the input pixel at (y0 + i - PAD_Y, x0 + j - PAD_X).
    for (int i = ty; i < TILE_H + HALO_H; i += block_size_y) {
        const int gy = y0 + i - PAD_Y;
        for (int j = tx; j < TILE_W + HALO_W; j += block_size_x) {
            const int gx = x0 + j - PAD_X;
            tile[i][j] = (gy >= 0 && gy < h && gx >= 0 && gx < w)
                ? in[(size_t)gy * w + gx]
                : 0.0f;
        }
    }
#endif

    __syncthreads();

    float acc[tile_size_y][tile_size_x];
#pragma unroll
    for (int a = 0; a < tile_size_y; a++) {
#pragma unroll
        for (int b = 0; b < tile_size_x; b++) {
            acc[a][b] = 0.0f;
        }
    }

#pragma unroll
    for (int i = 0; i < filter_height; i++) {
#pragma unroll
        for (int j = 0; j < filter_width; j++) {
            const float coef = f[i][j];

#pragma unroll
            for (int a = 0; a < tile_size_y; a++) {
#pragma unroll
                for (int b = 0; b < tile_size_x; b++) {
                    const int ly = ty + a * block_size_y;
                    const int lx = tx + b * block_size_x;
#if use_shmem
                    acc[a][b] += coef * tile[ly + i][lx + j];
#else
                    const int gy = y0 + ly + i - PAD_Y;
                    const int gx = x0 + lx + j - PAD_X;
                    const float v = (gy >= 0 && gy < h && gx >= 0 && gx < w)
                        ? in[(size_t)gy * w + gx]
                        : 0.0f;
                    acc[a][b] += coef * v;
#endif
                }
            }
        }
    }

#pragma unroll
    for (int a = 0; a < tile_size_y; a++) {
        const int y = y0 + ty + a * block_size_y;
#pragma unroll
        for (int b = 0; b < tile_size_x; b++) {
            const int x = x0 + tx + b * block_size_x;
            if (y < h && x < w) {
                out[(size_t)y * w + x] = acc[a][b];
            }
        }
    }
}
