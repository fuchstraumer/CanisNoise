#include "billow.cuh"

__device__ float billow2D_Simplex(float2 point, float freq, float lacun, float persist, int init_seed, int octaves) {
    float result = 0.0f;
    float amplitude = 1.0f;
    // Scale starting point by frequency.
    point.x = point.x * freq;
    point.y = point.y * freq;
    // Use loop for fractal octave bit
    for (size_t i = 0; i < octaves; ++i) {
        int seed = (init_seed + i) & 0xffffffff;
        result += fabsf(simplex2d(point.x, point.y, seed, nullptr)) * amplitude;
        point.x *= lacun;
        point.y *= lacun;
        amplitude *= persist;
    }
    //result /= 100.0f;
    return result;
}

__device__ float billow2D(float2 point, float freq, float lacun, float persist, int init_seed, int octaves) {
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
        result += fabsf(perlin2d(point.x, point.y, seed, nullptr)) * amplitude;
        // Modify vars for next octave.
        point.x *= lacun;
        point.y *= lacun;
        amplitude *= persist;
    }
    // float tmp = result / 100.0f;
    // * // 
    return result;
}

__device__ float billow3D(float3 point, const float freq, const float lacun, const float persist, const int init_seed, const int octaves) {
    float result = 0.0f;
    float amplitude = 1.0f;
    point *= freq;
    for (short i = 0; i < octaves; ++i) {
        int seed = (init_seed + i) & 0xffffffff;
        result += fabsf(simplex3d(point.x, point.y, point.z, seed, nullptr)) * amplitude;
        point *= lacun;
        amplitude *= persist;
    }
    return result;
}



__global__ void Billow2DKernel(float* output, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        float x, y;
        x = i + origin.x;
        y = j + origin.y;
        float2 p = make_float2(x, y);
        // Call billow function
        float val;
        switch (noise_type) {
            case(noise_t::PERLIN): {
                val = billow2D(p, freq, lacun, persist, seed, octaves);
                break;
            }
            case(noise_t::SIMPLEX): {
                val = billow2D_Simplex(p, freq, lacun, persist, seed, octaves);
                break;
            }
        }
        // Write val to the surface
        output[(j * width) + i] = val;
    }

    
}

__global__ void Billow3DKernel(float* output, const int width, const int height, const int depth, const float3 origin, const float freq, const float lacun, const float persist, const int seed, const int octaves) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= width || j >= height || k >= depth) {
        return;
    }

    float val;
    val = billow3D(make_float3(origin.x + i, origin.y + j, origin.z + k), freq, lacun, persist, seed, octaves);
    output[i + (j * width) + (k * width * height)] = val;
}

void cudaBillowLauncher2D(float* out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves) {

    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Billow2DKernel, 0, 0); 
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    Billow2DKernel<<<grid,block>>>(out, width, height, noise_type, origin, freq, lacun, persist, seed, octaves);
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);

}

void BillowLauncher3D(float* out, const int width, const int height, const int depth, const float3 origin, const float freq, const float lacun, const float persist, const int seed, const int octaves) {

    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Billow2DKernel, 0, 0); 
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    Billow3DKernel<<<grid, block >>>(out, width, height, depth, origin, freq, lacun, persist, seed, octaves);
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);

}