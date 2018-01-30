#pragma once
#ifndef CPU_TERRACE_HPP
#define CPU_TERRACE_HPP
#include "cpu_include.hpp"

CN_API void cpuTerraceLauncher(float* output, const float* input, const int& width, const int& height, const float* pts, const int& num_pts, const bool& invert);

#endif //!CPU_TERRACE_HPP