#include "combiners/multiply.cuh"


__global__ void multiplyKernelF(float* output, float* input, const int width, const int height, const float factor) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) {
        return;
    }
    output[(j * width) + i] = input[(j * width) + i] * factor;
}

__global__ void multiplyKernel(float* output, const float* in0, const float* in1, const int& width, const int& height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= width || j >= height) {
        return;
    }

    output[i + (j * width)] = in0[i + (j * width)] * in1[i + (j * width)];
}

void cudaMultiplyLauncherF(float* output, float* input, const int width, const int height, float factor) {
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, multiplyKernelF, 0, 0); //???
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    multiplyKernelF<<<grid, block>>>(output, input, width, height, factor);
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);
}

void cudaMultiplyLauncher(float* out, const float* in0, const float* in1, const int& width, const int& height) {
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, multiplyKernel, 0, 0); //???
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    multiplyKernel<<<grid, block>>>(out, in0, in1, width, height);
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);
}