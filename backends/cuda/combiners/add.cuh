#pragma once
#ifndef ADD_CUH
#define ADD_CUH
#include "../CUDA_Include.cuh"
extern "C" {
API_CALL void AddLauncher(float* output, float* input0, float* input1, const int width, const int height);

API_CALL void AddLauncher3D(float* output, float* input0, float* input1, const int width, const int height, const int depth);
}
#endif // !ADD_CUH
