#include "RenderingContext.hpp"
#include "ResourceContext.hpp"
#include "ImGuiWrapper.hpp"
#include "Instance.hpp"
#include "PipelineCache.hpp"
#include "nodegraph/psoCache.hpp"
#include "EditorScene.hpp"
#include <filesystem>
#include <string>
#ifdef APIENTRY
#undef APIENTRY
#endif
#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

void SetupPipelineCacheDir()
{
    const std::filesystem::path tmpDir = std::filesystem::temp_directory_path() / "CanisNoiseCache";
    if (!std::filesystem::exists(tmpDir))
    {
        if (!std::filesystem::create_directories(tmpDir))
        {
            std::cerr << "Couldn't create a directory to store pipeline caches in.";
        }
    }
    const std::string dirString = tmpDir.string();
    vpr::PipelineCache::SetCacheDirectory(dirString.c_str());
}

void MuxEasyloggingStorage()
{
//    void* storage_ptr = reinterpret_cast<void*>(&el::Helpers::storage());
//    vpr::SetLoggingRepository_VprCore(storage_ptr);

}

static void BeginRecreationCallback(VkSwapchainKHR handle, uint32_t width, uint32_t height)
{
    auto& scene = EditorScene::Get();
    scene.Destroy();
}

static void CompleteRecreationCallback(VkSwapchainKHR handle, uint32_t width, uint32_t height)
{
    auto& context = RenderingContext::Get();
    auto& scene = EditorScene::Get();
    scene.Initialize(context.Instance(), context.PhysicalDevice(), context.Device(), context.Swapchain());
}

int main(int argc, char* argv[])
{
    SetupPipelineCacheDir();
    auto& renderingCtxt = RenderingContext::Get();
    renderingCtxt.Construct();
    auto& resourceContext = ResourceContext::Get();
    resourceContext.Initialize(renderingCtxt.Device(), renderingCtxt.PhysicalDevice(), VTF_VALIDATION_ENABLED);

    SwapchainCallbacks resizeCallbacks;
    resizeCallbacks.BeginResize = &BeginRecreationCallback;
    resizeCallbacks.CompleteResize = &CompleteRecreationCallback;
    renderingCtxt.AddSwapchainCallbacks(resizeCallbacks);

    auto& scene = EditorScene::Get();
    scene.Initialize(renderingCtxt.Instance(), renderingCtxt.PhysicalDevice(), renderingCtxt.Device(), renderingCtxt.Swapchain());

    while (!RenderingContext::ShouldWindowClose())
    {
        renderingCtxt.Update();
        resourceContext.Update();
        scene.Update();
        ImGuiWrapper::GetImGuiWrapper().NewFrame();
        ImGui::BeginMainMenuBar();
        ImGui::EndMainMenuBar();
        ImGuiWrapper::GetImGuiWrapper().EndImGuiFrame();
        scene.Render();
    }

    return 0;
}