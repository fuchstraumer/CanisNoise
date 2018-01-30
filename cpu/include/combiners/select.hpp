#pragma once
#ifndef CPU_SELECT_HPP
#define CPU_SELECT_HPP
#include "cpu_include.hpp"

CN_API void cpuSelectLauncher(float* out, const float* select_item, const float* subject0, 
                             const float* subject1, const int& width, const int& height, 
                             const float& upper_bound, const float& lower_bound, const float& falloff);

#endif //!CPU_SELECT_HPP