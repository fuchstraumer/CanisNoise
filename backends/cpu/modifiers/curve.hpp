#pragma once
#ifndef CPU_CURVE_HPP
#define CPU_CURVE_HPP

#include "../cpu_include.hpp"

struct control_point_t;

API_CALL void CurveLauncher(float* output, const float* input0, 
                            const int& width, const int& height, control_point_t* pts,
                            const int& num_pts);

#endif //!CPU_CURVE_HPP