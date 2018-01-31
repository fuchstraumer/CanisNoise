#include "FBM.cuh"

__device__ float FBM2d_Simplex(float2 point, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
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
        result += simplex2d(point.x, point.y, seed, nullptr) * amplitude;
        // Modify vars for next octave.
        point.x *= lacun;
        point.y *= lacun;
        amplitude *= persist;
    }

    return result;
}

__device__ float FBM2d(float2 point, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
    float amplitude = 1.0f;
    // Scale point by freq
    point.x = point.x * freq;
    point.y = point.y * freq;
    // TODO: Seeding the function is currently pointless and doesn't actually do anything.
    // Use loop for octav-ing
    float result = 0.0f;
    for (size_t i = 0; i < octaves; ++i) {
        int seed = (init_seed + i) & 0xffffffff;
        result += perlin2d(point.x, point.y, seed, nullptr) * amplitude;
        // Modify vars for next octave.
        point.x *= lacun;
        point.y *= lacun;
        amplitude *= persist;
    }

    return result;
}

__global__ void FBM2DKernel(float* out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width && j >= height) {
        return;
    }

    float2 p = make_float2(origin.x + i, origin.y + j);
    // Call billow function
    float val;
    switch (noise_type) {
    case(noise_t::PERLIN): {
        val = FBM2d(p, freq, lacun, persist, seed, octaves);
        break;
    }
    case(noise_t::SIMPLEX): {
        val = FBM2d_Simplex(p, freq, lacun, persist, seed, octaves);
        break;
    }
    }

    // Write val to the surface
    out[(j * width) + i] = val;
}

void cudaFBM_Launcher(float* out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves){

    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, FBM2DKernel, 0, 0); 
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    FBM2DKernel<<<grid, block>>>(out, width, height, noise_type, origin, freq, lacun, persist, seed, octaves);
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);

}




