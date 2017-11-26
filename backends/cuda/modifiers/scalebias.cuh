#ifndef SCALEBIAS_CUH
#define SCALEBIAS_CUH
#include "../common/CUDA_Include.h"

API_CALL void scalebiasLauncher(float* output, float* input, const int width, const int height, float scale, float bias);

#endif // 
