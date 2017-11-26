#include "Cache.hpp"
#include "cuda_assert.h"

cnoise::utility::Cache::Cache(const size_t& width, const size_t& height, const std::shared_ptr<Module>& source) : Module(width, height) {
    sourceModules.push_back(source);
}

void cnoise::utility::Cache::Generate(){
    if (sourceModules.front() == nullptr) {
        throw;
    }
    if (!sourceModules.front()->Generated) {
        sourceModules.front()->Generate();
    }

    auto err = cudaDeviceSynchronize();
    cudaAssert(err);

    err = cudaMemcpy(Output, sourceModules.front()->Output, sizeof(sourceModules.front()->Output), cudaMemcpyDefault);
    cudaAssert(err);

    err = cudaDeviceSynchronize();
    cudaAssert(err);

    // Data copied, remove our reference to the shared pointer
    sourceModules.front().reset();
    sourceModules.clear();
    sourceModules.shrink_to_fit();
}

size_t cnoise::utility::Cache::GetSourceModuleCount() const{
    return 1;
}
