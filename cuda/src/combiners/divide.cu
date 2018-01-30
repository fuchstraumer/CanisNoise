#include "combiners/divide.cuh"

__global__ void divideKernel(float* out, const float* in0, const float* in1, const int& width, const int& height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) {
        return;
    }
    out[i + (j * width)] = in0[i + (j * width)] / in1[i + (j * width)];
}

__global__ void divideKernelF(float* out, const float* in0, const float& factor, const int& width, const int& height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) {
        return;
    }
    out[i + (j * width)] = in0[i + (j * width)] / factor;
}

void cudaDivideLauncher(float* out, const float* in0, const float* in1, const int& width, const int& height) {

    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, divideKernel, 0, 0);
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    divideKernel<<<grid, block>>>(out, in0, in1, width, height);
    // Check for succesfull kernel launch
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    // Synchronize device
    err = cudaDeviceSynchronize();
    cudaAssert(err);

}

void cudaDivideLauncherF(float* out, const float* in0, const float& factor, const int& width, const int& height) { 
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, divideKernelF, 0, 0); 
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    divideKernelF<<<grid, block>>>(out, in0, factor, width, height);
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);  
}