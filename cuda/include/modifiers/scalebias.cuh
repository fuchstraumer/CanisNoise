#pragma once
#ifndef SCALEBIAS_CUH
#define SCALEBIAS_CUH
#include "../CUDA_Include.cuh"

API_CALL void cudaScaleBiasLauncher(float* output, float* input, const int width, const int height, float scale, float bias);

#endif // 
