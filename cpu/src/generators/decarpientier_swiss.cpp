#include "generators/decarpientier_swiss.hpp"
#include "noisegen.hpp"
#include <algorithm>

inline float dcSwissVal(const float& _x, const float& _y, const float& warp_amt, const noise_t& noise_type, 
    const float& freq, const float& lacun, const float& persist, const int& octaves, const int& _seed) {
    
    float result = 0.0f;
    float ampl = 1.0f;
    float x = _x * freq;
    float y = _y * freq;
    float warp = warp_amt;
    float dx_sum = 0.0f;
    float dy_sum = 0.0f;

    for(int i = 0; i < octaves; ++i) {
        int seed = (_seed + i) & 0xffffffff;
        float dx, dy;
        float n;

        if(noise_type == noise_t::SIMPLEX) {
            n = simplex2d(x, y, seed, &dx, &dy);
        }
        else {
            n = perlin2d(x, y, seed, &dx, &dy);
        }

        dx_sum += ampl * dx * -n;
        dy_sum += ampl * dy * -n;

        x *= lacun;
        y *= lacun;
        x += (warp * dx_sum);
        y += (warp * dy_sum);
        ampl *= persist * (std::clamp(result, 0.0f, 1.0f));
    }

    return result;
}


void cpuDecarpientierSwissLauncher(float* out, const int& width, const int& height, const noise_t& noise_type, const float& warp_amt, 
    const float& origin_x, const float& origin_y, const float& freq, const float& lacun, const float& persist, const int& octaves, const int& seed) {
    for(int j = 0; j < height; ++j) {
        for(int i = 0; i < width; ++i) {
            const float x = static_cast<float>(i) + origin_x;
            const float y = static_cast<float>(j) + origin_y;
            out[i + (j * width)] = dcSwissVal(x, y, warp_amt, noise_type, freq, lacun, persist, octaves, seed);
        }
    }
}