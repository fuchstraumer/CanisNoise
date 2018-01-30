#pragma once
#ifndef MULTIPLY_CUH
#define MULTIPLY_CUH
#include "CUDA_Include.cuh"

CN_API void cudaMultiplyLauncherF(float* out, float* in, const int width, const int height, float factor);
CN_API void cudaMultiplyLauncher(float* out, const float* in0, const float* in1, const int& width, const int& height);

#endif 
