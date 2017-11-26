#pragma once
#ifndef CHECKERBOARD_CUH
#define CHECKERBOARD_CUH
#include "../CUDA_Include.h"
extern "C" {
API_CALL void CheckerboardLauncher(float* output, const int width, const int height);
}
#endif //!CHECKERBOARD_CUH