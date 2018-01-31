#pragma once
#ifndef POWER_CUH
#define POWER_CUH
#include "../CUDA_Include.cuh"

API_CALL void cudaPowerLauncher(float* output, float* input0, float* input1, const int width, const int height);

#endif 
