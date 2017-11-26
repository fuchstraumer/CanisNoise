#pragma once
#ifndef MULTIPLY_CUH
#define MULTIPLY_CUH
#include "../CUDA_Include.h"
extern "C" {
API_CALL void multiplyLauncher(float* out, float* in, const int width, const int height, float factor);
}
#endif 
