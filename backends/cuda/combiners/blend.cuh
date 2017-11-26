#pragma once
#ifndef BLEND_CUH
#define BLEND_CUH
#include "CUDA_Include.cuh"
extern "C" {
API_CALL void BlendLauncher(float * output, const float* in0, const float* in1, const float* weight, const int width, const int height);
}
#endif // !BLEND_CUH
