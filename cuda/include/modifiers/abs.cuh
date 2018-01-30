#pragma once
#ifndef ABS_CUH
#define ABS_CUH
#include "CUDA_Include.cuh"

CN_API void cudaAbsLauncher(float* out, float* in, const int width, const int height);

#endif 
