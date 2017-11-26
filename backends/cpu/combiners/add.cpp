#include "add.hpp"

void cpuAddLauncher(float* output, const float* input0, const float* input1, const int width, const int height) {
    for(int j = 0; j < height; ++j) {
        for(int i = 0; i < width; ++i) {
            output[i + (j * width)] = input0[i + (j * width)] + input1[i + (j * width)];
        }
    }
}