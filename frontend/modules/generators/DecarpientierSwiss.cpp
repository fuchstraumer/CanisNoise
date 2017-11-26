#include "DecarpientierSwiss.hpp"
#include "generators/decarpientier_swiss.cuh"

cnoise::generators::DecarpientierSwiss::DecarpientierSwiss(const size_t& width, const size_t& height, const noise_t& noise_type, const float& x, const float& y, const int& seed, const float& freq, const float& lacun, const int& octaves, const float& persist) : Module(width, height), Attributes(seed, freq, lacun, octaves, persist), Origin(x, y), NoiseType(noise_type){}

size_t cnoise::generators::DecarpientierSwiss::GetSourceModuleCount() const{
	return 0;
}

void cnoise::generators::DecarpientierSwiss::Generate(){
	DecarpientierSwissLauncher(Output, dims.first, dims.second, NoiseType, make_float2(Origin.first, Origin.second), Attributes.Frequency, Attributes.Lacunarity, Attributes.Persistence, Attributes.Seed, Attributes.Octaves);
	Generated = true;
}
