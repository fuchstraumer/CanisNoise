#pragma once
#ifndef CANIS_NOISE_NODE_TYPES_HPP
#define CANIS_NOISE_NODE_TYPES_HPP
#include <cstdint>

enum class NodeType : uint8_t
{
    Invalid = 0,
    Combiner = 1,
    Modifier = 2,
    Generator = 3
};

// Currently, needs to match defines in combiners.comp shader source
enum class CombinerNodes : uint8_t
{
    Add = 0,
    Subtract,
    Multiply,
    Divide,
    Min,
    Max,
    Power,
    Blend,
    Select,
    Count
};

enum class ModifierNodes : uint8_t
{
    Abs = 0,
    Clamp = 1,
    ScaleBias = 2,
    Downsample = 3,
    Terrace = 4,
    Curve = 5,
    Count
};

struct ControlPoint
{
    float x;
    float y;
};

struct ModifierPushConstants
{
    float ClampLowerBound;
    float ClampUpperBound;
    float sbScale;
    float sbBias;
};

#endif //!CANIS_NOISE_NODE_TYPES