#pragma once
#ifndef SELECT_CUH
#define SELECT_CUH
#include "CUDA_Include.cuh"

CN_API void cudaSelectLauncher(float* out, float* select_item, float* subject0, float* subject1, int width, int height, float upper_bound, float lower_bound, float falloff);

#endif // !SELECT_CUH