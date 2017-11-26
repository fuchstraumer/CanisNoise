#include "terrace.hpp"
#include <algorithm> 

void cpuTerraceLauncher(float* output, const float* input, const int& width, const int& height, const float* pts, const int& num_pts, const bool& invert) {
    for(int i = 0; i < width; ++i) {
        for(int j = 0; j < height; ++j) {

            int idx;
            for(idx = 0; idx < num_pts; ++idx) {
                if (input[i + (j * width)] < pts[idx]) {
                    break;
                }
            }

            const int idx0 = std::clamp(idx - 1, 0, num_pts - 1);
            const int idx1 = std::clamp(idx, 0, num_pts - 1);

            if (idx0 == idx1) {
                output[i + (j * width)] = pts[idx1];
                return;
            }

            float val0 = pts[idx0];
            float val1 = pts[idx1];
            float alpha = (input[i + (j * width)] - val0) / (val1 - val0);

            if(invert) {
                alpha = 1.0f - alpha;
                std::swap(val0, val1);
            }
            
            alpha *= alpha;

            output[i + (j * width)] = val0 + alpha * (val1 - val0);
        }
    }
    
}