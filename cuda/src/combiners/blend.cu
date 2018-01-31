#include "blend.cuh"
#include "../cutil_math.cuh"

__global__ void BlendKernel(float *output, const float* in0, const float* in1, const float* control, const int width, const int height) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= width || j >= height) {
        return;
    }
    output[(j * width) + i] = lerp(in0[(j * width) + i], in1[(j * width) + i], (control[(j * width) + i] + 1.0f) / 2.0f);
}

void cudaBlendLauncher(float * output, const float * in0, const float * in1, const float * weight, const int width, const int height){
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, BlendKernel, 0, 0); 
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    BlendKernel<<<grid,block>>>(output, in0, in1, weight, width, height);
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);
}