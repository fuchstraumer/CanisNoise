#pragma once
#ifndef CPU_DECARP_SWISS_HPP
#define CPU_DECARP_SWISS_HPP

#include "cpu_include.hpp"

CN_API void cpuDecarpientierSwissLauncher(float* out, const int& width, const int& height, const noise_t& noise_type, const float& warp_amt, 
    const float& origin_x, const float& origin_y, const float& freq, const float& lacun, const float& persist, const int& octaves, const int& seed);


#endif //!CPU_DECARP_SWISS_HPP