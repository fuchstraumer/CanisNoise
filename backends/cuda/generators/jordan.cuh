#pragma once
#ifndef JORDAN_CUH
#define JORDAN_CUH
#include "../CUDA_Include.cuh"
#include "../noise_generators.cuh"

API_CALL void JordanLauncher(cudaSurfaceObject_t out, int width, int height, noise_t noise_type, float2 origin, float freq, float lacun, float persist, int seed, int octaves);

#endif // !JORDAN_CUH
