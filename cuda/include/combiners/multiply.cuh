#pragma once
#ifndef MULTIPLY_CUH
#define MULTIPLY_CUH
#include "../CUDA_Include.cuh"

API_CALL void cudaMultiplyLauncherF(float* out, float* in, const int width, const int height, float factor);
API_CALL void cudaMultiplyLauncher(float* out, const float* in0, const float* in1, const int& width, const int& height);

#endif 
