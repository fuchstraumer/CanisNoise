#ifndef TERRACE_CUH
#define TERRACE_CUH
#include "CUDA_Include.cuh"

void TerraceLauncher(float* output, const float* input, const int width, const int height, const std::vector<float>& pts, bool invert);

#endif // !TERRACE_CUH
