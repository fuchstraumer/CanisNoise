#include "RenderingContext.hpp"
#include "PlatformWindow.hpp"
#include "Instance.hpp"
#include "PhysicalDevice.hpp"
#include "LogicalDevice.hpp"
#include "Swapchain.hpp"
#include "SurfaceKHR.hpp"
#include "VkDebugUtils.hpp"
#include "vkAssert.hpp"
#include <thread>
#include <sstream>
#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include <atomic>
#include <forward_list>
#include "GLFW/glfw3.h"
#ifdef APIENTRY
// re-defined by glfw on windows, then seen again by easylogging
#undef APIENTRY
#endif

static post_physical_device_pre_logical_device_function_t postPhysicalPreLogicalSetupFunction = nullptr;
static post_logical_device_function_t postLogicalDeviceFunction = nullptr;
static void* usedNextPtr = nullptr;
static VkPhysicalDeviceFeatures* enabledDeviceFeatures = nullptr;
static std::vector<std::string> extensionsBuffer;
static std::string windowingModeBuffer;
static bool validationEnabled{ false };
struct swapchain_callbacks_storage_t {
    std::forward_list<decltype(SwapchainCallbacks::SwapchainCreated)> CreationFns;
    std::forward_list<decltype(SwapchainCallbacks::BeginResize)> BeginFns;
    std::forward_list<decltype(SwapchainCallbacks::CompleteResize)> CompleteFns;
    std::forward_list<decltype(SwapchainCallbacks::SwapchainDestroyed)> DestroyedFns;
};
static swapchain_callbacks_storage_t SwapchainCallbacksStorage;
inline void RecreateSwapchain();

std::string objectTypeToString(const VkObjectType type)
{
    switch (type)
    {
    case VK_OBJECT_TYPE_INSTANCE:
        return "VkInstance";
    case VK_OBJECT_TYPE_PHYSICAL_DEVICE:
        return "VkPhysicalDevice";
    case VK_OBJECT_TYPE_DEVICE:
        return "VkDevice";
    case VK_OBJECT_TYPE_QUEUE:
        return "VkQueue";
    case VK_OBJECT_TYPE_SEMAPHORE:
        return "VkSemaphore";
    case VK_OBJECT_TYPE_COMMAND_BUFFER:
        return "VkCommandBuffer";
    case VK_OBJECT_TYPE_FENCE:
        return "VkFence";
    case VK_OBJECT_TYPE_DEVICE_MEMORY:
        return "VkDeviceMemory";
    case VK_OBJECT_TYPE_BUFFER:
        return "VkBuffer";
    case VK_OBJECT_TYPE_IMAGE:
        return "VkImage";
    case VK_OBJECT_TYPE_EVENT:
        return "VkEvent";
    case VK_OBJECT_TYPE_QUERY_POOL:
        return "VkQueryPool";
    case VK_OBJECT_TYPE_BUFFER_VIEW:
        return "VkBufferView";
    case VK_OBJECT_TYPE_IMAGE_VIEW:
        return "VkImageView";
    case VK_OBJECT_TYPE_SHADER_MODULE:
        return "VkShaderModule";
    case VK_OBJECT_TYPE_PIPELINE_CACHE:
        return "VkPipelineCache";
    case VK_OBJECT_TYPE_PIPELINE_LAYOUT:
        return "VkPipelineLayout";
    case VK_OBJECT_TYPE_RENDER_PASS:
        return "VkRenderPass";
    case VK_OBJECT_TYPE_PIPELINE:
        return "VkPipeline";
    case VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT:
        return "VkDescriptorSetLayout";
    case VK_OBJECT_TYPE_SAMPLER:
        return "VkSampler";
    case VK_OBJECT_TYPE_DESCRIPTOR_POOL:
        return "VkDescriptorPool";
    case VK_OBJECT_TYPE_DESCRIPTOR_SET:
        return "VkDescriptorSet";
    case VK_OBJECT_TYPE_FRAMEBUFFER:
        return "VkFramebuffer";
    case VK_OBJECT_TYPE_COMMAND_POOL:
        return "VkCommandPool";
    case VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION:
        return "VkSamplerYcbcrConversion";
    case VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE:
        return "VkDescriptorUpdateTemplate";
    case VK_OBJECT_TYPE_SURFACE_KHR:
        return "VkSurfaceKHR";
    case VK_OBJECT_TYPE_SWAPCHAIN_KHR:
        return "VkSwapchainKHR";
    case VK_OBJECT_TYPE_DISPLAY_KHR:
        return "VkDisplayKHR";
    case VK_OBJECT_TYPE_DISPLAY_MODE_KHR:
        return "VkDisplayModeKHR";
    case VK_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT:
        return "VkDebugReportCallbackEXT";
    case VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NV:
        return "VkIndirectCommandsLayoutNVX";
    case VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT:
        return "VkDebugUtilsMessengerEXT";
    case VK_OBJECT_TYPE_VALIDATION_CACHE_EXT:
        return "VkValidationCacheEXT";
    case VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV:
        return "VkAccelerationStructureNV";
    default:
        return std::string("TYPE_UNKNOWN:" + std::to_string(size_t(type)));
    };
}

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity, VkDebugUtilsMessageTypeFlagBitsEXT message_type, const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data)
{

    std::stringstream output_string_stream;
    if (callback_data->messageIdNumber != 0u)
    {
        output_string_stream << "VUID:" << callback_data->messageIdNumber << ":VUID_NAME:" << callback_data->pMessageIdName << "\n";
    }

    const static std::string SKIP_STR{ "CREATE" };
    const std::string message_str{ callback_data->pMessage };
    size_t found_skippable = message_str.find(SKIP_STR);

    if (found_skippable != std::string::npos)
    {
        return VK_FALSE;
    }

    output_string_stream << "    Message: " << message_str.c_str() << "\n";
    if (callback_data->queueLabelCount != 0u)
    {
        output_string_stream << "    Error occured in queue: " << callback_data->pQueueLabels[0].pLabelName << "\n";
    }

    if (callback_data->cmdBufLabelCount != 0u)
    {
        output_string_stream << "    Error occured executing command buffer(s): \n";
        for (uint32_t i = 0; i < callback_data->cmdBufLabelCount; ++i)
        {
            output_string_stream << "    " << callback_data->pCmdBufLabels[i].pLabelName << "\n";
        }
    }
    if (callback_data->objectCount != 0u)
    {
        auto& p_objects = callback_data->pObjects;
        output_string_stream << "    Object(s) involved: \n";
        for (uint32_t i = 0; i < callback_data->objectCount; ++i)
        {
            if (p_objects[i].pObjectName)
            {
                output_string_stream << "        ObjectName: " << p_objects[i].pObjectName << "\n";
            }
            else
            {
                output_string_stream << "        UNNAMED_OBJECT\n";
            }
            output_string_stream << "            ObjectType: " << objectTypeToString(p_objects[i].objectType) << "\n";
            output_string_stream << "            ObjectHandle: " << std::hex << std::to_string(p_objects[i].objectHandle) << "\n";
        }
    }

    if (message_severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        std::cerr << output_string_stream.str();
    }
    else if (message_severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        std::cerr << output_string_stream.str();
    }
    else if (message_severity <= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
    {
        std::cerr << output_string_stream.str();
    }

    return VK_FALSE;
}

static void SplitVersionString(std::string version_string, uint32_t& major_version, uint32_t& minor_version, uint32_t& patch_version) {
    const size_t minor_dot_pos = version_string.find('.');
    const size_t patch_dot_pos = version_string.rfind('.');
    if (patch_dot_pos == std::string::npos) {
        patch_version = 0;
        if (minor_dot_pos == std::string::npos) {
            minor_version = 0;
            major_version = static_cast<uint32_t>(strtod(version_string.c_str(), nullptr));
        }
        else {
            minor_version = static_cast<uint32_t>(strtod(version_string.substr(minor_dot_pos).c_str(), nullptr));
            major_version = static_cast<uint32_t>(strtod(version_string.substr(0, minor_dot_pos).c_str(), nullptr));
        }
    }
    else {
        if (minor_dot_pos == std::string::npos) {
            major_version = static_cast<uint32_t>(strtod(version_string.c_str(), nullptr));
            minor_version = 0;
            patch_version = 0;
            return;
        }
        else {
            major_version = static_cast<uint32_t>(strtod(version_string.substr(0, minor_dot_pos + 1).c_str(), nullptr));
            minor_version = static_cast<uint32_t>(strtod(version_string.substr(minor_dot_pos + 1, patch_dot_pos - minor_dot_pos - 1).c_str(), nullptr));
            patch_version = static_cast<uint32_t>(strtod(version_string.substr(patch_dot_pos).c_str(), nullptr));
        }
    }
}

static void GetVersions(uint32_t& app_version, uint32_t& engine_version, uint32_t& api_version)
{ 
    app_version = VK_MAKE_VERSION(0, 0, 1);
    engine_version = VK_MAKE_VERSION(0, 0, 1);
    api_version = VK_API_VERSION_1_2;
}

static const std::unordered_map<std::string, windowing_mode> windowing_mode_str_to_flag{
    { "Windowed", windowing_mode::Windowed },
    { "BorderlessWindowed", windowing_mode::BorderlessWindowed },
    { "Fullscreen", windowing_mode::Fullscreen }
};

void createInstanceAndWindow(std::unique_ptr<vpr::Instance>* instance, std::unique_ptr<PlatformWindow>* window, std::string& _window_mode) {

    int window_width = 1440;
    int window_height = 720;
    const std::string app_name = "NoiseNodegraphApp";
    const std::string windowing_mode_str = "Bordered";
    windowingModeBuffer = windowing_mode_str;
    _window_mode = windowingModeBuffer;
    auto iter = windowing_mode_str_to_flag.find(windowing_mode_str);
    windowing_mode window_mode = windowing_mode::Windowed;
    if (iter != std::cend(windowing_mode_str_to_flag)) {
        window_mode = iter->second;
    }

    *window = std::make_unique<PlatformWindow>(window_width, window_height, app_name.c_str(), window_mode);

    const std::string engine_name = "CanisNoise";
    const bool using_validation = true;
    validationEnabled = using_validation;

    uint32_t app_version = 0;
    uint32_t engine_version = 0;
    uint32_t api_version = 0;
    GetVersions(app_version, engine_version, api_version);

    vpr::VprExtensionPack pack;
    pack.RequiredExtensionCount = 0u;
    pack.RequiredExtensionNames = nullptr;
    pack.OptionalExtensionCount = 0u;
    pack.OptionalExtensionNames = nullptr;

    const VkApplicationInfo application_info{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        app_name.c_str(),
        app_version,
        engine_name.c_str(),
        engine_version,
        api_version
    };

    auto layers = using_validation ? vpr::Instance::instance_layers::Full : vpr::Instance::instance_layers::Disabled;
    *instance = std::make_unique<vpr::Instance>(layers, &application_info, &pack);

}

void createLogicalDevice(VkSurfaceKHR surface, std::unique_ptr<vpr::Device>* device, vpr::Instance* instance, vpr::PhysicalDevice* physical_device) {

    std::vector<const char*> required_extensions;
    required_extensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    std::vector<const char*> requested_extensions;
    requested_extensions.emplace_back(VK_EXT_PIPELINE_CREATION_FEEDBACK_EXTENSION_NAME);
    requested_extensions.emplace_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    requested_extensions.emplace_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);

    vpr::VprExtensionPack pack;
    pack.RequiredExtensionCount = static_cast<uint32_t>(required_extensions.size());
    pack.RequiredExtensionNames = required_extensions.data();
    pack.OptionalExtensionCount = static_cast<uint32_t>(requested_extensions.size());
    pack.OptionalExtensionNames = requested_extensions.data();

    if (usedNextPtr != nullptr)
    {
        pack.pNextChainStart = usedNextPtr;
    }

    if (enabledDeviceFeatures != nullptr)
    {
        pack.featuresToEnable = enabledDeviceFeatures;
    }

    *device = std::make_unique<vpr::Device>(instance, physical_device, surface, &pack, nullptr, 0);

    if (postLogicalDeviceFunction != nullptr)
    {
        postLogicalDeviceFunction(usedNextPtr);
    }
}

static std::atomic<bool>& GetShouldResizeFlag() {
    static std::atomic<bool> should_resize{ false };
    return should_resize;
}

DescriptorLimits::DescriptorLimits(const vpr::PhysicalDevice* hostDevice)
{
    const VkPhysicalDeviceLimits limits = hostDevice->GetProperties().limits;
    MaxSamplers = limits.maxDescriptorSetSamplers;
    MaxUniformBuffers = limits.maxDescriptorSetUniformBuffers;
    MaxDynamicUniformBuffers = limits.maxDescriptorSetUniformBuffersDynamic;
    MaxStorageBuffers = limits.maxDescriptorSetStorageBuffers;
    MaxDynamicStorageBuffers = limits.maxDescriptorSetStorageBuffersDynamic;
    MaxSampledImages = limits.maxDescriptorSetSampledImages;
    MaxStorageImages = limits.maxDescriptorSetStorageImages;
    MaxInputAttachments = limits.maxDescriptorSetInputAttachments;
}

RenderingContext& RenderingContext::Get() noexcept {
    static RenderingContext ctxt;
    return ctxt;
}

void RenderingContext::SetShouldResize(bool resize) {
    auto& flag = GetShouldResizeFlag();
    flag = resize;
}

bool RenderingContext::ShouldResizeExchange(bool value) {
    return GetShouldResizeFlag().exchange(value);
}

void RenderingContext::Construct() {

    createInstanceAndWindow(&vulkanInstance, &window, windowMode);
    window->SetWindowUserPointer(this);
    // Physical devices to be redone for multi-GPU support if device group extension is supported.
    physicalDevices.emplace_back(std::make_unique<vpr::PhysicalDevice>(vulkanInstance->vkHandle()));

    if (postPhysicalPreLogicalSetupFunction != nullptr)
    {
        postPhysicalPreLogicalSetupFunction(physicalDevices.back()->vkHandle(), &enabledDeviceFeatures, &usedNextPtr);
    }

    {
        size_t num_instance_extensions = 0;
        vulkanInstance->GetEnabledExtensions(&num_instance_extensions, nullptr);
        if (num_instance_extensions != 0) {
            std::vector<char*> extensions_buffer(num_instance_extensions);
            vulkanInstance->GetEnabledExtensions(&num_instance_extensions, extensions_buffer.data());
            for (auto& str : extensions_buffer) {
                instanceExtensions.emplace_back(str);
                free(str);
            }
        }
    }

    windowSurface = std::make_unique<vpr::SurfaceKHR>(vulkanInstance.get(), physicalDevices[0]->vkHandle(), (void*)window->glfwWindow());

    createLogicalDevice(windowSurface->vkHandle(), &logicalDevice, vulkanInstance.get(), physicalDevices[0].get());

    if constexpr (VTF_VALIDATION_ENABLED)
    {
        SetObjectNameFn = logicalDevice->DebugUtilsHandler().vkSetDebugUtilsObjectName;

        const VkDebugUtilsMessengerCreateInfoEXT messenger_info{
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            nullptr,
            0,
            // capture warnings and info that the current one does not
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            (PFN_vkDebugUtilsMessengerCallbackEXT)DebugUtilsMessengerCallback,
            nullptr
        };

        const auto& debugUtilsFnPtrs = logicalDevice->DebugUtilsHandler();

        if (!debugUtilsFnPtrs.vkCreateDebugUtilsMessenger)
        {
            std::cerr << "Debug utils function pointers struct doesn't have function pointer for debug utils messenger creation!";
            throw std::runtime_error("Failed to create debug utils messenger: function pointer not loaded!");
        }

        VkResult result = debugUtilsFnPtrs.vkCreateDebugUtilsMessenger(vulkanInstance->vkHandle(), &messenger_info, nullptr, &DebugUtilsMessenger);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create debug utils messenger.");
        }
    }

    {
        size_t num_device_extensions = 0;
        logicalDevice->GetEnabledExtensions(&num_device_extensions, nullptr);
        if (num_device_extensions != 0) {
            std::vector<char*> extensions_buffer(num_device_extensions);
            logicalDevice->GetEnabledExtensions(&num_device_extensions, extensions_buffer.data());
            for (auto& str : extensions_buffer) {
                deviceExtensions.emplace_back(str);
                free(str);
            }
        }
    }

    static const std::unordered_map<std::string, vpr::vertical_sync_mode> present_mode_from_str_map{
        { "None", vpr::vertical_sync_mode::None },
        { "VerticalSync", vpr::vertical_sync_mode::VerticalSync },
        { "VerticalSyncRelaxed", vpr::vertical_sync_mode::VerticalSyncRelaxed },
        { "VerticalSyncMailbox", vpr::vertical_sync_mode::VerticalSyncMailbox }
    };

    // We want to go for this, as it's the ideal mode usually (on desktop, at least). Especially because we don't care to build a fancier system for a desktop app.
    vpr::vertical_sync_mode desired_mode = vpr::vertical_sync_mode::VerticalSyncMailbox;
    swapchain = std::make_unique<vpr::Swapchain>(logicalDevice.get(), window->glfwWindow(), windowSurface->vkHandle(), desired_mode);

    if constexpr (VTF_VALIDATION_ENABLED && VTF_USE_DEBUG_INFO)
    {
        SetObjectName(VK_OBJECT_TYPE_SWAPCHAIN_KHR, swapchain->vkHandle(), "RenderingContextSwapchain");

        for (size_t i = 0u; i < swapchain->ImageCount(); ++i)
        {
            const std::string view_name = std::string("RenderingContextSwapchain_ImageView") + std::to_string(i);
            SetObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, swapchain->ImageView(i), view_name.c_str());
            const std::string img_name = std::string("RenderingContextSwapchain_Image") + std::to_string(i);
            SetObjectName(VK_OBJECT_TYPE_IMAGE, swapchain->Image(i), img_name.c_str());
        }
    }

}

void RenderingContext::Update() {
    window->Update();
    if (ShouldResizeExchange(false)) {
        RecreateSwapchain();
    }
}

void RenderingContext::Destroy() {
    swapchain.reset();
    if constexpr (VTF_VALIDATION_ENABLED)
    {
        logicalDevice->DebugUtilsHandler().vkDestroyDebugUtilsMessenger(vulkanInstance->vkHandle(), DebugUtilsMessenger, nullptr);
    }
    logicalDevice.reset();
    windowSurface.reset();
    physicalDevices.clear(); physicalDevices.shrink_to_fit();
    vulkanInstance.reset();
    window.reset();
}

vpr::Instance * RenderingContext::Instance() noexcept {
    return vulkanInstance.get();
}

vpr::PhysicalDevice * RenderingContext::PhysicalDevice(const size_t idx) noexcept {
    return physicalDevices[idx].get();
}

vpr::Device* RenderingContext::Device() noexcept {
    return logicalDevice.get();
}

vpr::Swapchain* RenderingContext::Swapchain() noexcept {
    return swapchain.get();
}

vpr::SurfaceKHR* RenderingContext::Surface() noexcept {
    return windowSurface.get();
}

PlatformWindow* RenderingContext::Window() noexcept {
    return window.get();
}

GLFWwindow* RenderingContext::glfwWindow() noexcept {
    return window->glfwWindow();
}

inline GLFWwindow* getWindow() {
    auto& ctxt = RenderingContext::Get();
    return ctxt.glfwWindow();
}

#pragma warning(push)
#pragma warning(disable: 4302)
#pragma warning(disable: 4311)
inline void RecreateSwapchain() {

    auto& Context = RenderingContext::Get();

    int width = 0;
    int height = 0;
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(Context.glfwWindow(), &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(Context.Device()->vkHandle());

    Context.Window()->GetWindowSize(width, height);

    for (auto& fn : SwapchainCallbacksStorage.BeginFns) {
        fn(Context.Swapchain()->vkHandle(), width, height);
    }

    vpr::RecreateSwapchainAndSurface(Context.Swapchain(), Context.Surface());
    Context.Device()->UpdateSurface(Context.Surface()->vkHandle());

    Context.Window()->GetWindowSize(width, height);
    for (auto& fn : SwapchainCallbacksStorage.CompleteFns) {
        fn(Context.Swapchain()->vkHandle(), width, height);
    }

    vkDeviceWaitIdle(Context.Device()->vkHandle());
}
#pragma warning(pop)


void AddSwapchainCallbacks(SwapchainCallbacks callbacks) {
    if (callbacks.SwapchainCreated) {
        SwapchainCallbacksStorage.CreationFns.emplace_front(callbacks.SwapchainCreated);
    }
    if (callbacks.BeginResize) {
        SwapchainCallbacksStorage.BeginFns.emplace_front(callbacks.BeginResize);
    }
    if (callbacks.CompleteResize) {
        SwapchainCallbacksStorage.CompleteFns.emplace_front(callbacks.CompleteResize);
    }
    if (callbacks.SwapchainDestroyed) {
        SwapchainCallbacksStorage.DestroyedFns.emplace_front(callbacks.SwapchainDestroyed);
    }
}

void RenderingContext::AddSetupFunctions(post_physical_device_pre_logical_device_function_t fn0, post_logical_device_function_t fn1)
{
    postPhysicalPreLogicalSetupFunction = fn0;
    postLogicalDeviceFunction = fn1;
}

void RenderingContext::AddSwapchainCallbacks(SwapchainCallbacks callbacks) {
    SwapchainCallbacksStorage.BeginFns.emplace_front(callbacks.BeginResize);
    SwapchainCallbacksStorage.CompleteFns.emplace_front(callbacks.CompleteResize);
}

void RenderingContext::GetWindowSize(int& w, int& h) {
    glfwGetWindowSize(getWindow(), &w, &h);
}

void RenderingContext::GetFramebufferSize(int& w, int& h) {
    glfwGetFramebufferSize(getWindow(), &w, &h);
}

void RenderingContext::RegisterCursorPosCallback(cursor_pos_callback_t callback_fn) {
    auto& ctxt = Get();
    ctxt.Window()->AddCursorPosCallbackFn(callback_fn);
}

void RenderingContext::RegisterCursorEnterCallback(cursor_enter_callback_t callback_fn) {
    auto& ctxt = Get();
    ctxt.Window()->AddCursorEnterCallbackFn(callback_fn);
}

void RenderingContext::RegisterScrollCallback(scroll_callback_t callback_fn) {
    auto& ctxt = Get();
    ctxt.Window()->AddScrollCallbackFn(callback_fn);
}

void RenderingContext::RegisterCharCallback(char_callback_t callback_fn) {
    auto& ctxt = Get();
    ctxt.Window()->AddCharCallbackFn(callback_fn);
}

void RenderingContext::RegisterPathDropCallback(path_drop_callback_t callback_fn) {
    auto& ctxt = Get();
    ctxt.Window()->AddPathDropCallbackFn(callback_fn);
}

void RenderingContext::RegisterMouseButtonCallback(mouse_button_callback_t callback_fn) {
    auto& ctxt = Get();
    ctxt.Window()->AddMouseButtonCallbackFn(callback_fn);
}

void RenderingContext::RegisterKeyboardKeyCallback(keyboard_key_callback_t callback_fn) {
    auto& ctxt = Get();
    ctxt.Window()->AddKeyboardKeyCallbackFn(callback_fn);
}

int RenderingContext::GetMouseButton(int button) {
    return glfwGetMouseButton(getWindow(), button);
}

void RenderingContext::GetCursorPosition(double& x, double& y) {
    glfwGetCursorPos(getWindow(), &x, &y);
}

void RenderingContext::SetCursorPosition(double x, double y) {
    glfwSetCursorPos(getWindow(), x, y);
}

void RenderingContext::SetCursor(GLFWcursor* cursor) {
    glfwSetCursor(getWindow(), cursor);
}

GLFWcursor* RenderingContext::CreateCursor(GLFWimage* image, int w, int h) {
    return glfwCreateCursor(image, w, h);
}

GLFWcursor* RenderingContext::CreateStandardCursor(int type) {
    return glfwCreateStandardCursor(type);
}

void RenderingContext::DestroyCursor(GLFWcursor* cursor) {
    glfwDestroyCursor(cursor);
}

bool RenderingContext::ShouldWindowClose() {
    return glfwWindowShouldClose(getWindow());
}

int RenderingContext::GetWindowAttribute(int attrib) {
    return glfwGetWindowAttrib(getWindow(), attrib);
}

void RenderingContext::SetInputMode(int mode, int val) {
    glfwSetInputMode(getWindow(), mode, val);
}

int RenderingContext::GetInputMode(int mode) {
    return glfwGetInputMode(getWindow(), mode);
}

const char* RenderingContext::GetShaderCacheDir()
{
    auto& ctxt = Get();
    return ctxt.shaderCacheDir.c_str();
}

void RenderingContext::SetShaderCacheDir(const char* dir)
{
    auto& ctxt = Get();
    ctxt.shaderCacheDir = dir;
}

VkResult RenderingContext::SetObjectName(VkObjectType object_type, void* handle, const char* name)
{
    if constexpr (VTF_VALIDATION_ENABLED && VTF_USE_DEBUG_INFO)
    {
        auto& ctxt = Get();

        if constexpr (VTF_DEBUG_INFO_THREADING || VTF_DEBUG_INFO_TIMESTAMPS)
        {
            std::string object_name_str{ name };
            std::stringstream extra_info_stream;
            if constexpr (VTF_DEBUG_INFO_THREADING)
            {
                extra_info_stream << std::string("_ThreadID:") << std::this_thread::get_id();
            }
            if constexpr (VTF_DEBUG_INFO_TIMESTAMPS)
            {

            }

            object_name_str += extra_info_stream.str();

            const VkDebugUtilsObjectNameInfoEXT name_info{
                VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                nullptr,
                object_type,
                reinterpret_cast<uint64_t>(handle),
                object_name_str.c_str()
            };

            return ctxt.SetObjectNameFn(ctxt.logicalDevice->vkHandle(), &name_info);
        }
        else
        {
            const VkDebugUtilsObjectNameInfoEXT name_info{
                VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                nullptr,
                object_type,
                reinterpret_cast<uint64_t>(handle),
                name
            };
            return ctxt.SetObjectNameFn(ctxt.logicalDevice->vkHandle(), &name_info);
        }
    }
    else
    {
        return VK_SUCCESS;
    }
}
