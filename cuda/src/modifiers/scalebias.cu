#include "modifiers/scalebias.cuh"

__global__ void scalebiasKernel(float* output, float* input, const int width, const int height, float scale, float bias) {
    // Get current pixel.
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= width || j >= height) {
        return;
    }

    output[(j * width) + i] = (input[(j * width) + i] * scale) + bias; // for default value for scale is 1 and bias is 0;
}

void cudaScaleBiasLauncher(float* output, float* input, const int width, const int height, float scale, float bias){

    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scalebiasKernel, 0, 0);
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    scalebiasKernel<<<grid, block>>>(output, input, width, height, scale, bias);
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);

}