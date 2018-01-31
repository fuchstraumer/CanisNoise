#include "divide.hpp"

void cpuDivideLauncher(float* output, const float* input0, const float* input1, const int& width, const int& height) {
    for(int j = 0; j < height; ++j) {
        for(int i = 0; i < width; ++i) {  
            const int idx = i + (j * height);
            output[idx] = input0[idx] / input1[idx];
        }
    }
}

void cpuDivideLauncherF(float* output, const float* input0, const float& factor, const int& width, const int& height) {
    for(int j = 0; j < height; ++j) {
        for(int i = 0; i < width; ++i) {  
            const int idx = i + (j * height);
            output[idx] = input0[idx] / factor;
        }
    }
}