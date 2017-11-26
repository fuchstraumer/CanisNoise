// Include modules
#define USING_CNOISE_NAMESPACES
#include "modules/Modules.hpp"
#include "cuda_assert.h"
#include <cuda_runtime.h>
#include <iostream>

// Check for CUDA support
int cuda_supported() {
    int result = 0;
    auto err = cudaGetDeviceCount(&result);
    return result;
}

int main() {
    int cuda_check = cuda_supported();
    if (cuda_check != 0) {
        cnoise::Module::CUDA_LOADED = true;
    }
    else {
        cnoise::Module::CUDA_LOADED = false;
    }

    
}