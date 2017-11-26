#pragma once
#ifndef CPU_DIVIDE_HPP
#define CPU_DIVIDE_HPP
#include "../cpu_include.hpp"

API_CALL void cpuDivideLauncher(float* output, const float* input0, const float* input1, const int& width, const int& height);
API_CALL void cpuDivideLauncherF(float* output, const float* input, const float& factor, const int& width, const int& height);

#endif //!CPU_DIVIDE_HPP