#pragma once
#ifndef MIN_CUH
#define MIN_CUH
#include "../CUDA_Include.cuh"

CN_API void cudaMinLauncher(float *output, const float* in0, const float* in1, const int width, const int height);

#endif // !MIN_CUH