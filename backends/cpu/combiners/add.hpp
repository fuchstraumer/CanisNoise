#pragma once
#ifndef ADD_HPP
#define ADD_HPP

#include "../cpu_include.hpp"
extern "C" {
API_CALL void AddLauncher(float* output, const float* input0, const float* input1, const int width, const int height);
}
#endif //!ADD_HPP