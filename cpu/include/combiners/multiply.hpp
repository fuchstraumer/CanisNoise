#pragma once
#ifndef CPU_MULTIPLY_HPP
#define CPU_MULTIPLY_HPP
#include "cpu_include.hpp"

CN_API void cpuMultiplyLauncher(float* output, const float* input0, const float* input1, const int& width, const int& height);
CN_API void cpuMultiplyLauncherF(float* output, const float* input, const float& factor, const int& width, const int& height);


#endif 