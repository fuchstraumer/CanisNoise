#include "blend.hpp"

void cpuBlendLauncher(float* output, const float* input0, const float* input1, const float* control, 
                   const int& width, const int& height) {
    for(int j = 0; j < height; ++j) {
        for(int i = 0; i < width; ++i) {    
            const int idx = i + (j * width);
            const float& in0 = input0[idx];
            const float& in1 = input1[idx];
            const float ctrl = (control[idx] + 1.0f) / 2.0f;
            // lerp the value
            output[idx] = in0 + ctrl * (in1 - in0);
        }
    }
}