#ifndef CANIS_NOISE_COMMON_STRUCTS_HPP
#define CANIS_NOISE_COMMON_STRUCTS_HPP
#include <utility>

typedef struct alignas(sizeof(float)) ControlPoint {
	float InputVal, OutputVal;
	ControlPoint(float in, float out) : InputVal(std::move(in)), OutputVal(std::move(out)) {}
    ControlPoint() = default;
} ControlPoint;

struct alignas(sizeof(float)) cnoise_coord_t {
	float x,y,z;
	float value;
};

#endif //!CANIS_NOISE_COMMON_STRUCTS_HPP