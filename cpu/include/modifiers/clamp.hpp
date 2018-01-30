#pragma once
#ifndef CPU_CLAMP_HPP
#define CPU_CLAMP_HPP
#include "cpu_include.hpp"

CN_API void cpuClampLauncher(float* output, const float* input, const float& min, const float& max, const int& height, const int& width);

#endif //!CPU_CLAMP_HPP