#pragma once
#ifndef CANIS_NOISE_EDITOR_SCENE_HPP
#define CANIS_NOISE_EDITOR_SCENE_HPP
#include <memory>
#include <chrono>
#include <unordered_map>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace vpr
{
    class Instance;
    class PhysicalDevice;
    class Device;
    class Swapchain;
    class Semaphore;
    class Renderpass;
    class CommandPool;
    class Framebuffer;
    class Fence;
}

namespace
{
    struct DepthStencil;
}

class EditorScene
{
    EditorScene() = default;
    ~EditorScene() = default;
    EditorScene(const EditorScene&) = delete;
    EditorScene& operator=(const EditorScene&) = delete;
public:
    static EditorScene& Get() noexcept;

    void Initialize(vpr::Instance* instancePtr, vpr::PhysicalDevice* physDevicePtr, vpr::Device* devicePtr, vpr::Swapchain* swapchainPtr);
    void Destroy();
    // run any internal per-frame update logic
    void Update();
    // render an actual image of the editor to the screen
    void Render();

private:

    void createSemaphores();
    void limitFrame();
    void update();
    void acquireImage();
    void recordCommands();
    void draw();
    void present();
    void endFrame();

    void setupCommandPool();
    void setupDepthStencil();
    void setupRenderpass();
    void setupFramebuffers();
    void setupSyncPrimitives();

    std::chrono::system_clock::time_point limiterA;
    std::chrono::system_clock::time_point limiterB;
    uint32_t currentBuffer;
    std::unique_ptr<vpr::Semaphore> imageAcquireSemaphore;
    std::unique_ptr<vpr::Semaphore> renderCompleteSemaphore;
    vpr::Instance* instance;
    vpr::PhysicalDevice* physicalDevice;
    vpr::Device* device;
    vpr::Swapchain* swapchain;
    VkRenderPass renderPass;
    std::unique_ptr<DepthStencil> depthStencil;
    std::unique_ptr<vpr::CommandPool> commandPool;

    std::vector<VkFramebuffer> framebuffers;
    std::vector<VkFence> fences;
    bool initialized = false;

};

#endif //!CANIS_NOISE_EDITOR_SCENE_HPP
