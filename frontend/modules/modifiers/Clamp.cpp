#include "Clamp.hpp"
#include "modifiers/clamp.cuh"

cnoise::modifiers::Clamp::Clamp(const size_t& width, const size_t& height, const float& lower_bound = 0.0f, const float& upper_bound = 1.0f, const std::shared_ptr<Module>& source = nullptr) : Module(width, height), lowerBound(lower_bound), upperBound(upper_bound) {
	ConnectModule(source);
}

size_t cnoise::modifiers::Clamp::GetSourceModuleCount() const{
	return 1;
}

void cnoise::modifiers::Clamp::Generate(){
	if (sourceModules.front() == nullptr) {
		throw;
	}
	if (!sourceModules.front()->Generated) {
		sourceModules.front()->Generate();
	}
	ClampLauncher(Output, sourceModules.front()->Output, dims.first, dims.second, lowerBound, upperBound);
	Generated = true;
}

float cnoise::modifiers::Clamp::GetLowerBound() const{
	return lowerBound;
}

float cnoise::modifiers::Clamp::GetUpperBound() const{
	return upperBound;
}

void cnoise::modifiers::Clamp::SetLowerBound(const float& lower){
	lowerBound = lower;
}

void cnoise::modifiers::Clamp::SetUpperBound(const float& upper){
	upperBound = upper;
}
