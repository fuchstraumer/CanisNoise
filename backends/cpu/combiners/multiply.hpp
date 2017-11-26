#pragma once
#ifndef CPU_MULTIPLY_HPP
#define CPU_MULTIPLY_HPP

#include "../cpu_include.hpp"
extern "C" {
API_CALL void MultiplyLauncher(float* output, const float* input0, const float* input1, const int& width, const int& height);
API_CALL void MultiplyLauncherF(float* output, const float* input, const float& factor, const int& width, const int& height);
}

#endif 