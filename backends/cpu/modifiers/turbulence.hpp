#pragma once
#ifndef CPU_TURBULENCE_HPP
#define CPU_TURBULENCE_HPP

#include "../cpu_include.hpp"

API_CALL void TurbulenceLauncher(float* output, const float* input, const int& width,
                                 const int& height, const int& roughness, const int& seed,
                                 const float& strength, const float& freq);

#endif //!CPU_TURBULENCE_HPP