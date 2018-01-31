#pragma once
#ifndef CUDA_BACKEND_DIVIDE_CUH
#define CUDA_BACKEND_DIVIDE_CUH

#include "CUDA_Include.cuh"

CN_API void cudaDivideLauncher(float* out, const float* in0, const float* in1, const int& width, const int& height);
CN_API void cudaDivideLauncherF(float* out, const float* in0, const float& factor, const int& width, const int& height);

#endif //!CUDA_BACKEND_DIVIDE_CUH