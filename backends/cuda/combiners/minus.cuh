#pragma once
#ifndef CUDA_BACKEND_MINUS_CUH
#define CUDA_BACKEND_MINUS_CUH

#include "CUDA_Include.cuh"

API_CALL void cudaMinusLauncher(float* out, const float* in0, const float* in1, const int& width, const int& height);
API_CALL void cudaMinusLauncherF(float* out, const float* in0, const float& amt, const int& width, const int& height);

#endif //!CUDA_BACKEND_MINUS_CUH