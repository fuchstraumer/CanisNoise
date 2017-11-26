#pragma once
#ifndef CPU_BACKEND_BILLOW_HPP
#define CPU_BACKEND_BILLOW_HPP

#include "../cpu_include.hpp"
extern "C" {
API_CALL void BillowLauncher(float* out, const int& width, const int& height, const float& origin_x, const float& origin_y, 
                             const float& freq, const float& lacunarity, const float& persistence, const int& octaves, const int& seed);
}
#endif //!CPU_BACKEND_BILLOW_HPP