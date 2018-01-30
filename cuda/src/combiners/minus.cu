#include "combiners/minus.cuh"

__global__ void minusKernelF(float* output, const float* input, const float& amt, const int& width, const int& height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) {
        return;
    }
    output[(j * width) + i] = input[(j * width) + i] - amt;
}

__global__ void minusKernel(float* output, const float* in0, const float* in1, const int& width, const int& height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= width || j >= height) {
        return;
    }

    output[i + (j * width)] = in0[i + (j * width)] - in1[i + (j * width)];
}

void cudaMinusLauncher(float* out, const float* in0, const float* in1, const int& width, const int& height) {
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, minusKernel, 0, 0); 
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    minusKernel<<<grid, block>>>(out, in0, in1, width, height);
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);
}

void cudaMinusLauncherF(float* out, const float* in0, const float& amt, const int& width, const int& height) {
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, minusKernelF, 0, 0); 
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    minusKernelF<<<grid, block>>>(out, in0, amt, width, height);
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);
}