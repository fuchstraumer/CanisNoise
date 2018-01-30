#pragma once
#ifndef BILLOW_CUH
#define BILLOW_CUH
#include "CUDA_Include.cuh"

CN_API void cudaBillowLauncher2D(float* out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves);
CN_API void BillowLauncher3D(float* out, const int width, const int height, const int depth, const float3 origin, const float freq, const float lacun, const float persist, const int seed, const int octaves);

#endif // !BILLOW_CUH
