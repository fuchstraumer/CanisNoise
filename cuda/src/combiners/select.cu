#include "combiners/select.cuh"

__device__ float SCurve3(float a){
    return (a * a * (3.0f - 2.0f * a));
}

__device__ float lerp_s(float n0, float n1, float a) {
    return ((1.0f - a) * n0) + (a * n1);
}

__global__ void SelectKernel(float* out, float* select_item, float* subject0, float* subject1, int width, int height, float upper_bound, float lower_bound, float falloff){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= width || j >= height) {
        return;
    }
    // Get previous values to select between.
    float prev0, prev1;
    prev0 = subject0[(j * width) + i];
    prev1 = subject1[(j * width) + i];

    // Get value used for selection.
    float select = select_item[(j * width) + i];

    // Get result vlaue by checking select for bounds (and apply falloff)
    float result, alpha;

    // If we have a falloff value / are using falloff
    if (falloff > 0.0f) {
        
        if (select < (lower_bound - falloff)) {
            result = prev0;
        }
        else if (select < (lower_bound + falloff)) {
            // Apply falloff now.
            float lCurve = lower_bound - falloff;
            float uCurve = lower_bound + falloff;
            alpha = SCurve3((select - lCurve) / (uCurve - lCurve));
            result = lerp_s(prev0, prev1, alpha);
        }
        else if (select < (upper_bound - falloff)) {
            result = prev1;
        }
        else if (select < (upper_bound + falloff)) {
            float lCurve = upper_bound - falloff;
            float uCurve = upper_bound + falloff;
            alpha = SCurve3((select - lCurve) / (uCurve - lCurve));
            result = lerp_s(prev1, prev0, alpha);
        }
        else {
            result = prev0;
        }
    }
    // No falloff = simpler method of getting result.
    else {
        if (select < lower_bound || select > upper_bound) {
            result = prev0;
        }
        else {
            result = prev1;
        }
    }

    // Write value to surface object.
    out[(j * width) + i] = result;
}

void cudaSelectLauncher(float* out, float* select_item, float* subject0, float* subject1, int width, int height, float upper_bound, float lower_bound, float falloff){

    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, SelectKernel, 0, 0); 
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    SelectKernel<<<grid, block>>>(out, select_item, subject0, subject1, width, height, upper_bound, lower_bound, falloff);
    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);

}