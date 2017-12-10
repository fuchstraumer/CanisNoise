#pragma once
#ifndef NOISE_SURFACE_HPP
#define NOISE_SURFACE_HPP

#include "VulpesRender/Resource/Image.hpp"

// Used to draw previews and final images onto.
class NoiseSurface {
public:

    NoiseSurface(const size_t& width, const size_t& height);
    void AddData(const std::vector<float>& raw_img_data);
    
private:
    
    std::unique_ptr<vulpes::Image> texture;
};

#endif //!NOISE_SURFACE_HPP
