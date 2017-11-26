#pragma once
#ifndef CPU_NOISE_GEN_HPP
#define CPU_NOISE_GEN_HPP

#include "cpu_include.hpp"

float perlin2d(const float& px, const float& py, const int& seed, 
               float* dx = nullptr, float* dy = nullptr);

float perlin3d(const float& px, const float& py, const float& pz, const int& seed,
               const float* dx = nullptr, const float* dy = nullptr, const float* dz = nullptr);

float simplex2d(const float& px, const float& py, const int& seed, 
                const float* dx = nullptr, const float* dy = nullptr);

float simplex3d(const float& px, const float& py, const float& pz, const int& seed,
                const float* dx = nullptr, const float* dy = nullptr, const float* dz = nullptr);

#endif //!CPU_NOISE_GEN_HPP