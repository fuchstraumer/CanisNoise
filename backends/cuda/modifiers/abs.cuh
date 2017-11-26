#pragma once
#ifndef ABS_CUH
#define ABS_CUH
#include "../CUDA_Include.h"
extern "C" {
API_CALL void absLauncher(float* out, float* in, const int width, const int height);
}
#endif 
