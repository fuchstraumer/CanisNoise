#include "modifiers/abs.cuh"

__global__ void absKernel(float* output, float* input, const int width, const int height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) {
        return;
    }

    float prev = input[(width * j) + i];
    output[(j * width) + i] = (prev <= 0.0f) ? -prev : prev;
}

void cudaAbsLauncher(float* output, float* input, const int width, const int height) {
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, absKernel, 0, 0);
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    absKernel<<<grid,block>>>(output, input, width, height);
    cudaAssert(cudaGetLastError());
    cudaError_t err = cudaDeviceSynchronize();
    cudaAssert(err);
}
