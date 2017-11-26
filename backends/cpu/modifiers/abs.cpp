#include "abs.hpp"

void cpuAbsLauncher(float* output, const float* input, const int& width, const int& height) {
    for(int i = 0; i < width; ++i) {
        for(int j = 0; j < height; ++j) {
            output[i + (j * width)] = input[i + (j * width)] > 0.0f ? input[i + (j * width)] : -input[i + (j * width)];
        }
    }
}