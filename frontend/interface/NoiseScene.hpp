#pragma once
#ifndef CANIS_NOISE_FRONTEND_SCENE_HPP
#define CANIS_NOISE_FRONTEND_SCENE_HPP

#include "VulpesRender/BaseScene.hpp"

class NoiseScene : public BaseScene {
public:

    NoiseScene();
    ~NoiseScene();

    virtual void WindowResized() override;
    virtual void RecreateObjects() override;
    virtual void RecordCommands() override;
    
private:

    void create();
    void destroy();

    VkViewport viewport;
    VkRect2D scissor;
};

#endif //!CANIS_NOISE_FRONTEND_SCENE_HPP