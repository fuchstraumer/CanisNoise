#pragma once
#ifndef CPU_ABS_HPP
#define CPU_ABS_HPP

#include "../cpu_include.hpp"
extern "C" {
API_CALL void AbsLauncher(float* out, const float* input, const int& width, const int& height);
}
#endif //!CPU_ABS_HPP