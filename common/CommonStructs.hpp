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

namespace cpu {

	template<size_t sz>
	struct cpu_coord_pack_t {
		std::array<__m128, sz / 4> x;
		std::array<__m128, sz / 4> y;
		std::array<__m128, sz / 4> z;
		std::array<__m128, sz / 4> value;
	};
};

#endif //!CANIS_NOISE_COMMON_STRUCTS_HPP