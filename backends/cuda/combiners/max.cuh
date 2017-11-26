#pragma once
#ifndef MAX_CUH
#define MAX_CUH
#include "CUDA_Include.cuh"

API_CALL void cudaMaxLauncher(float* output, const float* in0, const float* in1, const int width, const int height);

#endif // !MAX_CUH
