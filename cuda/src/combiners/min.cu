#include "min.cuh"

__global__ void MinKernel(float* output, const float* in0, const float* in1, const int width, const int height) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= width || j >= height) {
        return;
    }
    float out_val = in0[(j * width) + i] < in1[(j * width) + i] ? in0[(j * width) + i] : in1[(j * width) + i];
    output[(j * width) + i] = out_val;
}

void cudaMinLauncher(float* output, const float* in0, const float* in1, const int width, const int height) {
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MinKernel, 0, 0); 
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    MinKernel<<<grid, block>>>(output, in0, in1, width, height);
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);
}