#include "curve.cuh"

__device__ int clamp(int val, int lower_bound, int upper_bound) {
    if (val < lower_bound) {
        return lower_bound;
    }
    else if (val > upper_bound) {
        return upper_bound;
    }
    else {
        return val;
    }
}

__device__ float cubicInterp(float n0, float n1, float n2, float n3, float a){
    float p = (n3 - n2) - (n0 - n1);
    float q = (n0 - n1) - p;
    float r = n2 - n0;
    float s = n1;
    return p * a * a * a + q * a * a + r * a + s;
}

__global__ void CurveKernel(float* output, float* input, const int width, const int height, ControlPoint* control_points, size_t num_pts) {
    // Get current pos and return if out of bounds.
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= width || j >= width) {
        return;
    }

    // Get previous value.
    float prev = input[(j * width) + i];

    // Get appropriate control point.
    size_t idx;
    for (idx = 0; idx < num_pts; ++idx) {
        if (prev < control_points[idx].InputVal) {
            // Found appropriate index.
            break;
        }
    }

    // Get next four nearest control points so we can interpolate.
    size_t i0, i1, i2, i3;
    i0 = clamp(idx - 2, 0, num_pts - 1);
    i1 = clamp(idx - 1, 0, num_pts - 1);
    i2 = clamp(idx, 0, num_pts - 1);
    i3 = clamp(idx + 1, 0, num_pts - 1);

    // If we don't have enough control points, just write control point value to output
    if (i1 = i2) {
        output[(j * width) + i] = control_points[i1].OutputVal;
        return;
    }

    // Compute alpha value used for the cubic interpolation
    float input0 = control_points[i1].InputVal;
    float input1 = control_points[i2].InputVal;
    float alpha = (prev - input0) / (input1 - input0);

    // Perform the interpolation.
    output[(j * width) + i] = cubicInterp(control_points[i0].OutputVal, control_points[i1].OutputVal, control_points[i2].OutputVal, control_points[i3].OutputVal, alpha);
}

void cudaCurveLauncher(float* output, float* input, const int width, const int height, const ControlPoint* control_points, const int& num_pts) {

    // Setup structs on GPU
    ControlPoint *device_point_array;
    cudaMalloc(&device_point_array, num_pts * sizeof(ControlPoint));

    // Copy structs to GPU
    cudaMemcpy(device_point_array, &control_points[0], num_pts * sizeof(ControlPoint), cudaMemcpyHostToDevice);

    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, CurveKernel, 0, 0); 
    dim3 block(blockSize, blockSize, 1);
    dim3 grid((width - 1) / blockSize + 1, (height - 1) / blockSize + 1, 1);
    CurveKernel<<<grid, block>>>(output, input, width, height, device_point_array, num_pts);

    cudaError_t err = cudaGetLastError();
    cudaAssert(err);
    err = cudaDeviceSynchronize();
    cudaAssert(err);

    // Free control points array
    cudaFree(device_point_array);

}