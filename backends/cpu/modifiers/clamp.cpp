#include "clamp.hpp"

void cpuClampLauncher(float* output, const float* input, const float& min, const float& max, const int& width, const int& height) {
    for(int i = 0; i < width; ++i) {
        for(int j = 0; j < height; ++j) {
            const int idx = i + (j * width);
            output[idx] = input[idx] < min ? min : input[idx] > max ? max : input[idx];
        }
    }
}