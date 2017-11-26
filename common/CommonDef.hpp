#pragma once
#ifndef CANIS_NOISE_COMMON_DEF_HPP
#define CANIS_NOISE_COMMON_DEF_HPP

enum noise_t {
	PERLIN,
	SIMPLEX,
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

#endif //!CANIS_NOISE_COMMON_DEF_HPP