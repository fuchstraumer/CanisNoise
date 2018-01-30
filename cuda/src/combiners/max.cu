#include "combiners/max.cuh"

__global__ void MaxKernel(float *output, const float *in0, const float *in1, const int width, const int height) {
    const int i = blockDim.x * blockIdx.x * threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= width || j >= height) {
        return;
    }

    float out_val = in0[(j * width) + i] > in1[(j * width) + i] ? in0[(j * width) + i] : in1[(j * width) + i];
    output[(j * width) + i] = out_val;
}

void cudaMaxLauncher(float *output, const float *in0, const float *in1, const int width, const int height) {
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MaxKernel, 0, 0);
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    MaxKernel<<<grid, block>>>(output, in0, in1, width, height);
    // Check for succesfull kernel launch
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    // Synchronize device
    err = cudaDeviceSynchronize();
    cudaAssert(err);
}