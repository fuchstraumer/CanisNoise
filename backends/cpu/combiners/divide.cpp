#include "divide.hpp"

void DivideLauncher(float* output, const float* input0, const float* input1, const int& width, const int& height) {
    for(int i = 0; i < width; ++i) {
        for(int j = 0; j < height; ++j) {
            const int idx = i + (j * height);
            output[idx] = input0[idx] / input1[idx];
        }
    }
}

void DivideLauncherF(float* output, const float* input0, const float& factor, const int& width, const int& height) {
    for(int i = 0; i < width; ++i) {
        for(int j = 0; j < height; ++j) {
            const int idx = i + (j * height);
            output[idx] = input0[idx] / factor;
        }
    }
}