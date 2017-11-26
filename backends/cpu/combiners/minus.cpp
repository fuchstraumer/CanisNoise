#include "minus.hpp"

void cpuMinusLauncher(float* output, const float* input0, const float* input1, const int width, const int height) {
    for(int i = 0; i < width; ++i) {
        for(int j = 0; j < height; ++j) {
            output[i + (j * width)] = input0[i + (j * width)] - input1[i + (j * width)];
        }
    }
}