#ifndef NORMALIZE_CUH
#define NORMALIZE_CUH
#include "../common/CUDA_Include.h"

API_CALL void NormalizeLauncher(float* input, float* output, const int width, const int height);

#endif // !NORMALIZE_CUH
