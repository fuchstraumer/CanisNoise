#pragma once
#ifndef CPU_BLEND_HPP
#define CPU_BLEND_HPP

#include "../cpu_include.hpp"

API_CALL void BlendLauncher(float* output, const float* input0, const float* input1, 
                            const float* control, const int& width, const int& height);

#endif //!CPU_BLEND_HPP