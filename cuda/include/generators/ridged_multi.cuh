#pragma once
#ifndef RIDGED_MULTI_CUH
#define RIDGED_MULTI_CUH
#include "../CUDA_Include.cuh"
#include "../noise_generators.cuh"

API_CALL void cudaRidgedMultiLauncher(float* out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

#endif // !RIDGED_MULTI_CUH
