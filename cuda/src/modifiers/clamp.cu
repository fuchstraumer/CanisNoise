#include "modifiers/clamp.cuh"

__global__ void ClampKernel(float* output, float* input, const int width, const int height, const float lower_value, const float upper_value) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= width || j >= height) {
        return;
    }
    
    // Get previous value.
    float prev = input[(j * width) + i];

    // Compare and clamp to range appropriately.
    if (prev < lower_value) {
        output[(j * width) + i] = lower_value;
    }
    else if (prev > upper_value) {
        output[(j * width) + i] = upper_value;
    }
    else {
        output[(j * width) + i] = prev;
    }

}

void cudaClampLauncher(float* output, float* input, const int width, const int height, const float lower_value, const float upper_value) {
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ClampKernel, 0, 0); 
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    ClampKernel<<<grid, block>>>(output, input, width, height, lower_value, upper_value);
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);
}