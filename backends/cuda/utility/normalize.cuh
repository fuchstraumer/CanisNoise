#pragma once
#ifndef NORMALIZE_CUH
#define NORMALIZE_CUH
#include "../CUDA_Include.cuh"

API_CALL void cudaNormalizeLauncher(float* input, float* output, const int width, const int height);

#endif // !NORMALIZE_CUH
