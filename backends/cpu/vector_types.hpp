#ifndef CPU_BACKEND_VECTOR_TYPES_HPP
#define CPU_BACKEND_VECTOR_TYPES_HPP
#include <utility>

struct vec4 {
    float x, y, z, w;
    vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    vec4(const float& val) : x(val), y(val), z(val), w(val) {}
    vec4(const float& _x, const float& _y, const float& _z, const float& _w) :
        x(_x), y(_y), z(_z), w(_w) {}
    vec4(const vec4& other) noexcept : x(other.x), y(other.y), z(other.z), w(other.w) {}
    vec4& operator=(const vec4& other) noexcept { x = other.x; y = other.y; z = other.z; w = other.w; return *this; }
    vec4(vec4&& other) noexcept : x(std::move(other.x)), y(std::move(other.y)), z(std::move(other.z)), w(std::move(other.w)) {}

};

#endif //!CPU_BACKEND_VECTOR_TYPES_HPP