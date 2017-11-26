#include "scalebias.hpp"

void cpuScaleBiasLauncher(float* output, const float* input, const int& width, const int& height, const float& scale, const float& bias) {
    for(int j = 0; j < height; ++j) {
        for(int i = 0; i < width; ++i) {  
            output[i + (j * width)] = (input[i + (j * width)] * scale) + bias;
        }
    }
}