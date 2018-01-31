#pragma once
#ifndef DECARPIENTIER_SWISS_CUH
#define DECARPIENTIER_SWISS_CUH
#include "../CUDA_Include.cuh"
#include "../noise_generators.cuh"

API_CALL void cudaDecarpientierSwissLauncher(float* out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

#endif // !DECARPIENTIER_SWISS_CUH
