#pragma once
#ifndef CURVE_CUH
#define CURVE_CUH
#include "../CUDA_Include.cuh"

CN_API void cudaCurveLauncher(float* output, float* input, const int width, const int height, const ControlPoint* control_points, const int& num_pts);

#endif // !CURVE_CUH
