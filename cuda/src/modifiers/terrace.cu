#include "terrace.cuh"
#include "../cutil_math.cuh"

__global__ void TerraceKernel(float* output, const float* input, const int width, const int height, const float* pts, const int num_pts, const bool invert) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= width || j >= height) {
        return;
    }

    // Terrace calculation.
    float prev = input[(j * width) + i];

    // Get index to terrace near
    int idx;
    for (idx = 0; idx < num_pts; ++idx) {
        if (prev < pts[idx]) {
            break;
        }
    }

    // Get indices into pts container defining our curving range.
    int idx0, idx1;
    idx0 = clamp(idx - 1, 0, num_pts - 1);
    idx1 = clamp(idx, 0, num_pts - 1);

    // Bounds check
    if (idx0 == idx1) {
        output[(j * width) + i] = pts[idx1];
        return;
    }

    
    float val0, val1;
    val0 = pts[idx0];
    val1 = pts[idx1];
    float alpha = (prev - val0) / (val1 - val0);

    // Invert check
    if (invert) {
        alpha = 1.0f - alpha;
        // Swap val0,val1
        float tmp = val0;
        val0 = val1;
        val1 = tmp;
    }

    alpha *= alpha;

    // Write output value
    output[(j * width) + i] = lerp(val0, val1, alpha);
    return;
}

void cudaTerraceLauncher(float * output, const float * input, const int width, const int height, const float* pts, const int& num_Pts, bool invert){

    float* device_pts;
    err = cudaMalloc(&device_pts, sizeof(float) * num_Pts);
    cudaAssert(err);
    err = cudaMemcpy(device_pts, &pts[0], sizeof(float) * num_Pts, cudaMemcpyHostToDevice);
    cudaAssert(err);

    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, TerraceKernel, 0, 0);
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    TerraceKernel<<<grid, block>>>(output, input, width, height, device_pts, num_Pts, invert);
    err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);

    // Free device_pts
    cudaFree(device_pts);

}