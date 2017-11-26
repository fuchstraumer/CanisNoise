#include "Power.hpp"
#include "combiners/power.cuh"

cnoise::combiners::Power::Power(const size_t& width, const size_t& height, const std::shared_ptr<Module>& in0, const std::shared_ptr<Module>& in1) : Module(width, height){
    sourceModules.push_back(in0);
    sourceModules.push_back(in1);
}

void cnoise::combiners::Power::Generate(){
    for (const auto& m : sourceModules) {
        if (m == nullptr) {
            throw;
        }
        if (!m->Generated) {
            m->Generate();
        }
    }
    powerLauncher(Output, sourceModules[0]->Output, sourceModules[1]->Output, dims.first, dims.second);
    Generated = true;
}

size_t cnoise::combiners::Power::GetSourceModuleCount() const{
    return 2;
}
