#include "combiners/power.cuh"


__global__ void powerKernel(float* output, float* input0, float* input1, const int width, const int height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) {
        return;
    }
    float prev0, prev1;
    prev0 = input0[(j * width) + i];
    prev1 = input1[(j * width) + i];

    // Raise prev0 to the power of prev1 and write to the output.
    output[(j * width) + i] = powf(prev0, prev1);
}

void cudaPowerLauncher(float* output, float* input0, float* input1, const int width, const int height) {
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, powerKernel, 0, 0); 
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    powerKernel<<<grid, block >>>(output, input0, input1, width, height); //Call Kernel
    cudaAssert(cudaGetLastError());
    cudaError_t err = cudaDeviceSynchronize();
    cudaAssert(err);
}


