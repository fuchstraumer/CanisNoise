#include "power.hpp"
#include <cmath>

void cpuPowerLauncher(float* output, const float* input0, const float* input1, const int width, const int height) {
    for(int i = 0; i < width; ++i) {
        for(int j= 0; j < height; ++j) {
             const int idx = i + (j * width);
             output[idx] = powf(input0[idx], input1[idx]);
        }
    }
}