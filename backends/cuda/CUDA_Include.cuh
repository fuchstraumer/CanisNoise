#pragma once
#ifndef CUDA_INCLUDE_H
#define CUDA_INCLUDE_H

/*

	CUDA_INCLUDE_H

	Used for including the required CUDA components in C++.
	
*/
#define CUDA_KERNEL_TIMING
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <device_functions.h>
#include "cuda_assert.h"
#include "CommonStructs.hpp"

enum class noise_t {
    PERLIN,
    SIMPLEX
};

// Type of distance function to use in voronoi generation
enum voronoi_distance_t {
    MANHATTAN,
    EUCLIDEAN,
    CELLULAR,
};

// Type of value to get from a voronoi function, and then store in the output texture.
enum voronoi_return_t {
    CELL_VALUE, // Get cell coord/val. Analagous to value noise.
    NOISE_LOOKUP, // Use coords to get a noise value
    DISTANCE, // Get distance to node.
};


#ifdef BUILDING_DLL
#define API_CALL __declspec(dllexport)
#else
#define API_CALL __declspec(dllimport)
#endif

#endif // !CUDA_INCLUDE_H
