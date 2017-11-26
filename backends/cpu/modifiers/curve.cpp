#include "curve.hpp"
#include <algorithm>
#include "CommonStructs.hpp"

struct control_point_t {
    float input_val;
    float output_val;
};

constexpr float cubic_interp(const float& n0, const float& n1, const float& n2, const float& n3, const float& a) {
    const float p = (n3 - n2) - (n0 - n1);
    const float q = (n0 - n1) - p;
    const float r = n2 - n0;
    return p * a * a * a + p * a * a + r * a + n1;
}

void CurveLauncher(float* output, const float* input0, const int& width, const int& height, ControlPoint* pts, const int& num_pts) {
    for(int i = 0; i < width; ++i) {
        for(int j = 0; j < height; ++j) {
            
            int idx;
            for(idx = 0; idx < num_pts; ++idx) {
                if(input0[i + (j * width)] < pts[idx].InputVal) {
                    break;
                }
            }

            const int i0 = std::clamp(idx - 2, 0, num_pts - 1);
            const int i1 = std::clamp(idx - 1, 0, num_pts - 1);
            const int i2 = std::clamp(idx, 0, num_pts - 1);
            const int i3 = std::clamp(idx + 1, 0, num_pts - 1);

            if (i1 == i2) {
                output[i + (j * width)] = pts[i1].OutputVal;
                continue;
            }

            const float& in0 = pts[i1].InputVal;
            const float& in1 = pts[i2].InputVal;
            const float alpha = (input0[i + (j * width)] - in0) / (in1 - in0);

            output[i + (j * width)] = cubic_interp(pts[i0].OutputVal, pts[i1].OutputVal, pts[i2].OutputVal, pts[i3].OutputVal, alpha);
        }
    }
}