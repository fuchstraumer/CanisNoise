#pragma once
#ifndef CURVE_CUH
#define CURVE_CUH
#include "../CUDA_Include.h"
extern "C" {
API_CALL void CurveLauncher(float* output, float* input, const int width, const int height, std::vector<ControlPoint>& control_points);
}
#endif // !CURVE_CUH
