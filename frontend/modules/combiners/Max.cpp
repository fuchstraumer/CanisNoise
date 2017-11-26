#include "Max.hpp"
#include "combiners/max.cuh"

cnoise::combiners::Max::Max(const size_t& width, const size_t& height, const std::shared_ptr<Module>& in0 = nullptr, const std::shared_ptr<Module>& in1 = nullptr) : Module(width, height) {
	sourceModules.push_back(in0);
	sourceModules.push_back(in1);
}

void cnoise::combiners::Max::Generate() {
	for (const auto m : sourceModules) {
		if (m == nullptr) {
			throw;
		}
		if (!m->Generated) {
			m->Generate();
		}
	}
	MaxLauncher(Output, sourceModules[0]->Output, sourceModules[1]->Output, dims.first, dims.second);
	Generated = true;
}

size_t cnoise::combiners::Max::GetSourceModuleCount() const{
	return 2;
}
