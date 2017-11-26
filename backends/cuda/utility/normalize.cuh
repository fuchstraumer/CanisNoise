#pragma once
#ifndef NORMALIZE_CUH
#define NORMALIZE_CUH
#include "../CUDA_Include.cuh"

API_CALL void NormalizeLauncher(float* input, float* output, const int width, const int height);

#endif // !NORMALIZE_CUH
