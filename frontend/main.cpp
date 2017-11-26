// Include modules
#define USING_CNOISE_NAMESPACES
#include "modules/Modules.hpp"
#include <iostream>

// Check for CUDA support
int cuda_supported() {
    try {
        
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Couldn't verify CUDA support: " << e.what() << "\n";
        return 0;
    }
}

int main() {

    int img_size_x = 2048;
    int img_size_y = 1024;
    
}