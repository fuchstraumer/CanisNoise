#include "generators/ridged_multi.cuh"
#include "noise_generators.cuh"

__device__ float Ridged2D_Simplex(float2 point, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
    float result = 0.0f;
    float amplitude = 1.0f;
    // Scale starting point by frequency.
    point.x = point.x * freq;
    point.y = point.y * freq;
    // Use loop for fractal octave bit
    for (size_t i = 0; i < octaves; ++i) {
        int seed = (init_seed + i) & 0xffffffff;
        result += (1.0f - fabsf(simplex2d(point.x, point.y, seed, nullptr))) * amplitude;
        point.x *= lacun;
        point.y *= lacun;
        amplitude *= persist;
    }
    return result;
}

__device__ float Ridged2D(float2 point, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
    // Will be incremented upon.
    float result = 0.0f;
    float amplitude = 1.0f;
    // Scale point by freq
    point.x = point.x * freq;
    point.y = point.y * freq;
    // TODO: Seeding the function is currently pointless and doesn't actually do anything.
    // Use loop for octav-ing
    for (size_t i = 0; i < octaves; ++i) {
        int seed = (init_seed + i) & 0xffffffff;
        result += (1.0f - fabsf(perlin2d(point.x, point.y, seed, nullptr)))* amplitude;
        // Modify vars for next octave.
        point.x *= lacun;
        point.y *= lacun;
        amplitude *= persist;
    }
    return result;
}

__global__ void Ridged2DKernel(float* out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < width && j < height) {
        // Get offset pos.
        float2 p = make_float2(i + origin.x, j + origin.y);
        // Call ridged function
        float val;
        switch (noise_type) {
            case(noise_t::PERLIN): {
                val = Ridged2D(p, freq, lacun, persist, seed, octaves);
                break;
            }
            case(noise_t::SIMPLEX): {
                val = Ridged2D_Simplex(p, freq, lacun, persist, seed, octaves);
                break;
            }
        }
        // Write val to the surface
        out[(j * width) + i] = val;
    }
    
}

void cudaRidgedMultiLauncher(float* out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {

    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Ridged2DKernel, 0, 0); 
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    Ridged2DKernel<<<grid, block>>>(out, width, height, noise_type, origin, freq, lacun, persist, seed, octaves);
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);

}