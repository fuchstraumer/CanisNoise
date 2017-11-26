#include "RidgedMulti.hpp"
#include "noisegen.hpp"

inline float ridgedVal(const float& _x, const float& _y, const noise_t& noise_type, const float& freq, const float& lacun, const float& persist,
    const int& octaves, const int& _seed) {
    float result = 0.0f;
    float ampl = 1.0f;
    float x = _x * freq;
    float y = _y * freq;

    for(int i = 0; i < octaves; ++i) {
        int seed = (_seed + i) & 0xffffffff;
        if (noise_type == noise_t::SIMPLEX) {
            result += (1.0f - simplex2d(x, y, seed)) * ampl;
        }
        else {
            result += (1.0f - perlin2d(x, y, seed)) * ampl;
        }
        x *= lacun;
        y *= lacun;
        ampl *= persist;
    }

    return result;
}

void cpuRidgedMultiLauncher(float* out, const int& width, const int& height, const noise_t& noise_type, const float& origin_x,
    const float& origin_y, const float& freq, const float& lacun, const float& persist, const int& octaves, const int& seed) {
    for(int j = 0; j < height; ++j) {
        for(int i = 0; i < width; ++i) {
            const float x = static_cast<float>(i) + origin_x;
            const float y = static_cast<float>(j) + origin_y;
            out[i + (j * width)] = ridgedVal(x, y, noise_type, freq, lacun, persist, octaves, seed);
        }
    }
}