#ifndef MULTIPLY_CUH
#define MULTIPLY_CUH
#include "../common/CUDA_Include.h"

API_CALL void multiplyLauncher(float* out, float* in, const int width, const int height, float factor);

#endif 
