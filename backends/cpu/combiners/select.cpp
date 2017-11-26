#include "select.hpp"


float SCurve3(const float& a) {
	return (a * a * (3.0f - 2.0f * a));
}

float lerp_s(const float& n0, const float& n1, const float& a) {
	return ((1.0f - a) * n0) + (a * n1);
}

void SelectLauncher(float* out, const float* select_item, const float* input0, const float* input1, 
                    const int& width, const int& height, const float& lower_bound, const float& upper_bound,
                    const float& falloff) {

    for(int i = 0; i < width; ++i) {
        for(int j = 0; j < height; ++j) {
            const int idx = i + (j * width);

            const float& select = select_item[idx];
            const float& prev0 = input0[idx];
            const float& prev1 = input1[idx];

            if (falloff > 0.0f) {
                if (select < lower_bound - falloff) {
                    const float l_curve = lower_bound - falloff;
                    const float u_curve = lower_bound + falloff;
                    const float alpha = SCurve3((select - l_curve) / (u_curve - l_curve));
                    out[idx] = lerp_s(prev0, prev1, alpha);
                    continue;
                }
                else if (select < (upper_bound - falloff)) {
                    out[idx] = prev1;
                    continue;
                }
                else if (select < (upper_bound + falloff)) {
                    const float l_curve = upper_bound - falloff;
                    const float u_curve = upper_bound + falloff;
                    const float alpha = SCurve3((select - l_curve) / (u_curve - l_curve));
                    out[idx] = lerp_s(prev1, prev0, alpha);
                    continue;
                }
                else {
                    out[idx] = prev0;
                    continue;
                }
            }
            else {
                if(select < lower_bound || select > upper_bound) {
                    out[idx] = prev0;
                    continue;
                }
                else {
                    out[idx] = prev1;
                    continue;
                }
            }
        }
    }
}