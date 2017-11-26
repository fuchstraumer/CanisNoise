#pragma once
#ifndef CHECKERBOARD_CUH
#define CHECKERBOARD_CUH
#include "../CUDA_Include.h"

API_CALL void cudaCheckerboardLauncher(float* output, const int width, const int height);

#endif //!CHECKERBOARD_CUH