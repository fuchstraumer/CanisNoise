#pragma once
#ifndef CPU_SCALE_BIAS_HPP
#define CPU_SCALE_BIAS_HPP
#include "../cpu_include.hpp"

CN_API void cpuScaleBiasLauncher(float* output, const float* input, const int& width, 
                                const int& height, const float& scale, const float& bias);

#endif //!CPU_SCALE_BIAS_HPP