#pragma once
#ifndef POWER_CUH
#define POWER_CUH
#include "../CUDA_Include.cuh"

CN_API void cudaPowerLauncher(float* output, float* input0, float* input1, const int width, const int height);

#endif 
