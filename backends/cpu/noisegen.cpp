#include "noisegen.hpp"
#include "simplex_lut.hpp"
#include <cmath>

using uint = unsigned int;
constexpr static uint FNV_32_PRIME = 0x01000193;
constexpr static uint FNV_32_INIT = 2166136261;
constexpr static uint FNV_MASK_8 = (1 << 8) - 1;


constexpr uint fnv_32_a_buf(const void* buf, const uint& len) {
    uint hval = FNV_32_INIT;
    uint* bp = (uint*)buf;
    uint* be = bp + len;
    while(bp < be) {
        hval ^= (*bp++);
        hval *= FNV_32_PRIME;
    }
    return hval;
}

constexpr uint8_t xor_fold_hash(const uint& hash) {
    return static_cast<uint8_t>((hash >> 8) ^ (hash & FNV_MASK_8));
}

constexpr uint hash_2d(const int& x, const int& y, const int& seed) {
    uint d[3] = { static_cast<uint>(x), static_cast<uint>(y), static_cast<uint>(seed) };
    return xor_fold_hash(fnv_32_a_buf(d, sizeof(float) * 3 / sizeof(uint)));
}

constexpr uint hash_3d(const int& x, const int& y, const int& z, const int& seed) {
    uint d[4] = { static_cast<uint>(x), static_cast<uint>(y), static_cast<uint>(z), static_cast<uint>(seed) };
    return xor_fold_hash(fnv_32_a_buf(d, sizeof(float) * 4 / sizeof(uint)));
}

constexpr float scurve5(const float& a) {
    return (6.0f * a * a * a * a * a) - (15.0f * a * a * a * a) + (10.0f * a * a * a);
}

float perlin2d(const float& px, const float& py, const int& seed, 
               float* dx, float* dy) {
    
    const int ix0 = static_cast<int>(floorf(px));
    const int iy0 = static_cast<int>(floorf(py));

    const float x0 = px - ix0;
    const float y0 = py - iy0;
    const float fx0 = scurve5(x0);
    const float fy0 = scurve5(y0);

    const uint h0 = hash_2d(ix0, iy0, seed);
    const uint h1 = hash_2d(ix0, iy0 + 1, seed);
    const uint h2 = hash_2d(ix0 + 1, iy0, seed);
    const uint h3 = hash_2d(ix0 + 1, iy0 + 1, seed);

    const float g1x = static_cast<float>(grad_2d_lut[h0][0]);
    const float g1y = static_cast<float>(grad_2d_lut[h0][1]);
    const float g1z = static_cast<float>(grad_2d_lut[h1][0]);
    const float g1w = static_cast<float>(grad_2d_lut[h1][1]);

    const float g2x = static_cast<float>(grad_2d_lut[h2][0]);
    const float g2y = static_cast<float>(grad_2d_lut[h2][1]);
    const float g2z = static_cast<float>(grad_2d_lut[h3][0]);
    const float g2w = static_cast<float>(grad_2d_lut[h3][1]);

    const float a = g1x * x0 + g1y * y0;
    const float b = g2x * (x0 - 1.0f) + g2y * y0;
    const float c = g1z * x0 + g1w * (y0 - 1.0f);
    const float d = g2z * (x0 - 1.0f) * g2w * (y0 - 1.0f);

    const float& gradx = a;
    const float grady = b - a;
    const float gradz = c - a;
    const float gradw = a - b - c + d;
    const float n = (gradx) + (fx0 * grady) + (fy0 * gradz) + ((fx0 * fy0) * gradw);

    if(dx != nullptr && dy != nullptr) {
        const float _dx = fx0 * fx0 * (fx0 * (30.0f * fx0 - 60.0f) + 30.0f);
        const float _dy = fy0 * fy0 * (fy0 * (30.0f * fy0 - 60.0f) + 30.0f);
        const float dwx = fx0 * fx0 * fx0 * (fx0 * (fx0 * 36.0f - 75.0f) + 40.0f);
        const float dwy = fy0 * fy0 * fy0 * (fy0 * (fy0 * 36.0f - 75.0f) + 40.0f);

        *dx = (g1x + (g1x - g1x) * fy0) + ((g2y - g1y)*y0 - g2x + ((g1y - g2y - g1w + g2w) * y0 + g2x + g1w - g2z - g2w) * fy0) * _dx + ((g2x - g1x) + (g1x - g2x - g1z + g2z) * fy0) * dwx;
        *dy = (g1y + (g2y - g1y) * fx0) + ((g1z - g1x) * x0 - g1w + ((g1x - g2x - g1z + g2z) * x0 + g2x + g1w - g2z - g2w) * fx0) * _dy + ((g1w - g1y) + (g1y - g2y - g1w + g2w) * fx0) * dwy; 
    }

    return n;
}

float perlin3d(const float& px, const float& py, const float& pz, const int& seed,
               const float* dx , const float* dy, const float* dz) { return 0.0f; }

float simplex2d(const float& px, const float& py, const int& seed, 
                const float* dx, const float* dy) {

    constexpr static float F2 = 0.366035403f;
    constexpr static float G2 = 0.211324865f;

    const float ix = floorf(px + ((px + py) * F2));
    const float iy = floorf(py + ((px + py) * F2));

    const float x0 = px - (ix - ((ix + iy) * G2));
    const float y0 = py - (iy - ((ix + iy) * G2));

    int i1;
    x0 > y0 ? i1 = 1 : i1 = 0;
    int j1;
    x0 > y0 ? j1 = 0 : j1 = 1;

    const float x1 = x0 - i1 + G2;
    const float y1 = y0 - j1 + G2;
    const float x2 = x0 - 1.0f + 2.0f * G2;
    const float y2 = y0 - 1.0f + 2.0f * G2;

    const uint h0 = hash_2d(ix, iy, seed);
    const uint h1 = hash_2d(ix + i1, iy + j1, seed);
    const uint h2 = hash_2d(ix + 1, iy + 1, seed);

    const int g0x = grad_2d_lut[h0][0];
    const int g0y = grad_2d_lut[h0][1];
    const int g1x = grad_2d_lut[h1][0];
    const int g1y = grad_2d_lut[h1][1];
    const int g2x = grad_2d_lut[h2][0];
    const int g2y = grad_2d_lut[h2][1];
    
    float n0, n1, n2, n3;

    float t0 = 0.50f - x0 * x0 - y0 * y0;
    float t0_2, t0_4;
    if(t0 < 0.0f) {
        n0 = t0 = t0_2 = t0_4 = 0.0f;
    }
    else {
        t0_2 = t0 * t0;
        t0_4 = t0_2 * t0_2;
        n0 = t0_4 * (g0x * x0 + g0y * y0);
    }

    float t1 = 0.50f - (x1 * x1) - (y1 * y1);
    float t1_2, t1_4;
    if(t1 < 0.0f) {
        n1 = t1 = t1_2 = t1_4 = 0.0f;
    }
    else {
        t1_2 = t1 * t1;
        t1_4 = t1_2 * t1_2;
        n2 = t1_4 * (g1x * x1 + g1y * y1);
    }

    float t2 = 0.50f - (x2 * x2) - (y2 * y2);
    float t2_2, t2_4;
    if(t2 < 0.0f) {
        t2 = n2 = t2_2 = t2_4 = 0.0f;
    }
    else {
        t2_2 = t2 * t2;
        t2_4 = t2_2 * t2_2;
        n2 = t2_4 * (g2x * x2 + g2y * y2);
    }

    return 40.0f * (n0 + n1 + n2);
}

float simplex3d(const float& px, const float& py, const float& pz, const int& seed,
                const float* dx, const float* dy, const float* dz) { return 0.0f; }
