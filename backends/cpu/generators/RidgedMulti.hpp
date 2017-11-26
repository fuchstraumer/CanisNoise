#pragma once
#ifndef CPU_RIDGED_MULTI_HPP
#define CPU_RIDGED_MULTI_HPP

#include "cpu_include.hpp"

void cpuRidgedMultiLauncher(float* out, const int& width, const int& height, const noise_t& noise_type, const float& origin_x,
    const float& origin_y, const float& freq, const float& lacun, const float& persist, const int& octaves, const int& seed);

#endif //!CPU_RIDGED_MULTI_HPP