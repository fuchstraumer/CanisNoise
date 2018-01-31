#pragma once
#ifndef CPU_BACKEND_FBM_HPP
#define CPU_BACKEND_FBM_HPP

#include "cpu_include.hpp"

CN_API void cpuFBM_Launcher(float* out, const int& width, const int& height, const noise_t& noise_type, const float& origin_x,
    const float& origin_y, const float& freq, const float& lacun, const float& persist, const int& octaves, const int& seed);


#endif //!CPU_INCLUDE_HPP