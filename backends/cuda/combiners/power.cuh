#pragma once
#ifndef POWER_CUH
#define POWER_CUH
#include "../CUDA_Include.cuh"
extern "C" {
API_CALL void powerLauncher(float* output, float* input0, float* input1, const int width, const int height);
}
#endif 
