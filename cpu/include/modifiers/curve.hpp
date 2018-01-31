#pragma once
#ifndef CPU_CURVE_HPP
#define CPU_CURVE_HPP
#include "../cpu_include.hpp"

struct ControlPoint;

CN_API void cpuCurveLauncher(float* output, const float* input0, 
                            const int& width, const int& height, ControlPoint* pts,
                            const int& num_pts);

#endif //!CPU_CURVE_HPP