#include "EditorScene.hpp"
#include "Instance.hpp"
#include "PhysicalDevice.hpp"
#include "LogicalDevice.hpp"
#include "Swapchain.hpp"
#include "ImGuiWrapper.hpp"
#include "vkAssert.hpp"
#include "Semaphore.hpp"
#include "CommandPool.hpp"
#include <thread>

namespace
{
    struct DepthStencil
    {
        DepthStencil() = default;
        DepthStencil(const vpr::Device* device, const vpr::PhysicalDevice* p_device, const vpr::Swapchain* swap);
        ~DepthStencil();
        VkImage Image{ VK_NULL_HANDLE };
        VkDeviceMemory Memory{ VK_NULL_HANDLE };
        VkImageView View{ VK_NULL_HANDLE };
        VkFormat Format;
        VkDevice Parent{ VK_NULL_HANDLE };
    };

    struct BasicPipelineCreateInfo
    {
        const vpr::Device* device{ nullptr };
        VkPipelineCreateFlags pipelineFlags{ 0 };
        uint32_t numStages{ 0u };
        const VkPipelineShaderStageCreateInfo* stages{ nullptr };
        const VkPipelineVertexInputStateCreateInfo* vertexState{ nullptr };
        VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
        VkRenderPass renderPass{ VK_NULL_HANDLE };
        VkCompareOp depthCompareOp;
        VkPipelineCache pipelineCache{ VK_NULL_HANDLE };
        VkPipeline derivedPipeline{ VK_NULL_HANDLE };
        VkCullModeFlags cullMode{ VK_CULL_MODE_BACK_BIT };
        VkPrimitiveTopology topology{ VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    };

    uint32_t GetMemoryTypeIndex(uint32_t type_bits, VkMemoryPropertyFlags properties, VkPhysicalDeviceMemoryProperties memory_properties);
    DepthStencil CreateDepthStencil(const vpr::Device* device, const vpr::PhysicalDevice* physical_device, const vpr::Swapchain* swapchain);
    VkRenderPass CreateBasicRenderpass(const vpr::Device* device, const vpr::Swapchain* swapchain, VkFormat depth_format);
    VkPipeline CreateBasicPipeline(const BasicPipelineCreateInfo& createInfo);

}

EditorScene& EditorScene::Get() noexcept
{
    static EditorScene scene;
    return scene;
}

void EditorScene::Initialize(
    vpr::Instance* instancePtr,
    vpr::PhysicalDevice* physDevicePtr,
    vpr::Device* devicePtr,
    vpr::Swapchain* swapchainPtr)
{
    instance = instancePtr;
    physicalDevice = physDevicePtr;
    device = devicePtr;
    swapchain = swapchainPtr;
    ImGui::CreateContext();
    setupCommandPool();
    setupDepthStencil();
    setupSyncPrimitives();
    createSemaphores();
    setupRenderpass();
    setupFramebuffers();

    ImGuiWrapper::GetImGuiWrapper().Construct(renderPass);

    limiterA = std::chrono::system_clock::now();
    limiterB = std::chrono::system_clock::now();
}

void EditorScene::Destroy()
{
    vkDeviceWaitIdle(device->vkHandle());

    ImGuiWrapper::GetImGuiWrapper().Destroy();
    imageAcquireSemaphore.reset();
    renderCompleteSemaphore.reset();

    for (auto& fence : fences)
    {
        vkDestroyFence(device->vkHandle(), fence, nullptr);
    }

    for (auto& fbuff : framebuffers)
    {
        vkDestroyFramebuffer(device->vkHandle(), fbuff, nullptr);
    }

    vkDestroyRenderPass(device->vkHandle(), renderPass, nullptr);
    depthStencil.reset();
    commandPool.reset();

}

void EditorScene::Update()
{
    update();
}

void EditorScene::Render()
{
    acquireImage();
    recordCommands();
    draw();
    present();
    limitFrame();
    endFrame();
}

void EditorScene::createSemaphores()
{
    imageAcquireSemaphore = std::make_unique<vpr::Semaphore>(device->vkHandle());
    renderCompleteSemaphore = std::make_unique<vpr::Semaphore>(device->vkHandle());
}

void EditorScene::limitFrame()
{
    limiterA = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> work_time = limiterA - limiterB;
    if (work_time.count() < 16.0)
    {
        std::chrono::duration<double, std::milli> delta_ms(16.0 - work_time.count());
        auto delta_ms_dur = std::chrono::duration_cast<std::chrono::milliseconds>(delta_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(delta_ms_dur.count()));
    }
    limiterB = std::chrono::system_clock::now();
}

void EditorScene::update()
{
}

void EditorScene::acquireImage()
{
    VkResult result = vkAcquireNextImageKHR(device->vkHandle(), swapchain->vkHandle(), UINT64_MAX, imageAcquireSemaphore->vkHandle(), VK_NULL_HANDLE, &currentBuffer);
    VkAssert(result);
}

void EditorScene::recordCommands()
{
    auto& imguiWrapper = ImGuiWrapper::GetImGuiWrapper();

    constexpr static VkCommandBufferBeginInfo beginInfo
    {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
        nullptr
    };

    constexpr static std::array<VkClearValue, 2> clearValues
    {
        VkClearValue{ VkClearColorValue{ 94.0f / 255.0f, 156.0f / 255.0f, 1.0f, 1.0f } },
        VkClearValue{ 1.0f, 0 }
    };

    const VkRect2D renderArea
    {
        VkOffset2D{ 0, 0 },
        VkExtent2D{ swapchain->Extent() }
    };

    const VkViewport viewport
    {
        0.0f,
        0.0f,
        static_cast<float>(swapchain->Extent().width),
        static_cast<float>(swapchain->Extent().height),
        0.0f,
        1.0f
    };

    const VkRect2D scissor
    {
        renderArea
    };

    VkRenderPassBeginInfo rpBegin
    {
        VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        nullptr,
        renderPass,
        VK_NULL_HANDLE,
        renderArea,
        static_cast<uint32_t>(clearValues.size()),
        clearValues.data()
    };

    {
        rpBegin.framebuffer = framebuffers[currentBuffer];
        VkResult result = VK_SUCCESS;
        VkCommandBuffer currentCmdBuffer = commandPool->GetCmdBuffer(currentBuffer);
        result = vkBeginCommandBuffer(currentCmdBuffer, &beginInfo);
        vkCmdBeginRenderPass(currentCmdBuffer, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
        imguiWrapper.DrawFrame(currentBuffer, commandPool->GetCmdBuffer(currentBuffer));
        vkCmdEndRenderPass(currentCmdBuffer);
        result = vkEndCommandBuffer(currentCmdBuffer);
        VkAssert(result);
    }
}

void EditorScene::draw()
{

    constexpr static VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const VkSubmitInfo submission
    {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        1,
        &imageAcquireSemaphore->vkHandle(),
        &waitStageMask,
        1,
        &commandPool->GetCmdBuffer(currentBuffer),
        1,
        &renderCompleteSemaphore->vkHandle()
    };

    VkResult result = vkQueueSubmit(device->GraphicsQueue(), 1, &submission, fences[currentBuffer]);

}

void EditorScene::present()
{

    VkResult present_results[1]{ VK_SUCCESS };

    const VkPresentInfoKHR present_info
    {
        VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        nullptr,
        1,
        &renderCompleteSemaphore->vkHandle(),
        1,
        &swapchain->vkHandle(),
        &currentBuffer,
        present_results
    };

    VkResult result = vkQueuePresentKHR(device->GraphicsQueue(), &present_info);
    VkAssert(result);

}

void EditorScene::endFrame()
{

    VkResult result = vkWaitForFences(device->vkHandle(), 1, &fences[currentBuffer], VK_TRUE, UINT64_MAX);
    VkAssert(result);
    result = vkResetFences(device->vkHandle(), 1, &fences[currentBuffer]);
    VkAssert(result);

    commandPool->ResetCmdBuffer(currentBuffer);
    currentBuffer = (currentBuffer + 1u) % static_cast<uint32_t>(framebuffers.size());
}

void EditorScene::setupCommandPool()
{
    const VkCommandPoolCreateInfo poolInfo
    {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        device->QueueFamilyIndices().Graphics
    };

    commandPool = std::make_unique<vpr::CommandPool>(device->vkHandle(), poolInfo);

    const VkCommandBufferAllocateInfo cmdInfo
    {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,

    };

    commandPool->AllocateCmdBuffers(swapchain->ImageCount(), VK_COMMAND_BUFFER_LEVEL_PRIMARY);

}

void EditorScene::setupDepthStencil()
{
    depthStencil = std::make_unique<DepthStencil>(device, physicalDevice, swapchain);
}

void EditorScene::setupRenderpass()
{
    renderPass = CreateBasicRenderpass(device, swapchain, depthStencil->Format);
}

void EditorScene::setupFramebuffers()
{
    framebuffers.resize(swapchain->ImageCount());
    for (size_t i = 0; i < framebuffers.size(); ++i)
    {
        std::array<VkImageView, 2> imageViews
        {
            swapchain->ImageView(i),
            depthStencil->View
        };

        const VkFramebufferCreateInfo fbufInfo
        {
            VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            nullptr,
            0,
            renderPass,
            static_cast<uint32_t>(imageViews.size()),
            imageViews.data(),
            swapchain->Extent().width,
            swapchain->Extent().height,
            1
        };

        VkResult result = vkCreateFramebuffer(device->vkHandle(), &fbufInfo, nullptr, &framebuffers[i]);
        VkAssert(result);
    }
}

void EditorScene::setupSyncPrimitives()
{
    constexpr static VkFenceCreateInfo fenceCreateInfo
    {
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        nullptr,
        0
    };

    fences.resize(commandPool->size());

    for (auto& fence : fences)
    {
        VkResult result = vkCreateFence(device->vkHandle(), &fenceCreateInfo, nullptr, &fence);
        VkAssert(result);
    }

}

namespace
{

    uint32_t GetMemoryTypeIndex(uint32_t type_bits, VkMemoryPropertyFlags properties, VkPhysicalDeviceMemoryProperties memory_properties)
    {
        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
        {
            if ((type_bits & 1) == 1)
            {
                if ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
                {
                    return i;
                }
            }
            type_bits >>= 1;
        }

        throw std::domain_error("Could not find matching memory type for given bits and property flags");
    }

    DepthStencil CreateDepthStencil(
        const vpr::Device* device,
        const vpr::PhysicalDevice* physical_device,
        const vpr::Swapchain* swapchain)
    {
        DepthStencil depth_stencil;
        depth_stencil.Format = device->FindDepthFormat();

        const VkImageCreateInfo image_info
        {
            VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            nullptr,
            0,
            VK_IMAGE_TYPE_2D,
            depth_stencil.Format,
            VkExtent3D{ swapchain->Extent().width, swapchain->Extent().height, 1 },
            1,
            1,
            VK_SAMPLE_COUNT_1_BIT,
            device->GetFormatTiling(depth_stencil.Format, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT),
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            0,
            nullptr,
            VK_IMAGE_LAYOUT_UNDEFINED
        };

        VkResult result = VK_SUCCESS;
        result = vkCreateImage(device->vkHandle(), &image_info, nullptr, &depth_stencil.Image);
        VkAssert(result);

        VkMemoryAllocateInfo alloc_info{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr };
        VkMemoryRequirements memreqs{};
        vkGetImageMemoryRequirements(device->vkHandle(), depth_stencil.Image, &memreqs);
        alloc_info.allocationSize = memreqs.size;
        alloc_info.memoryTypeIndex =
            GetMemoryTypeIndex(memreqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, physical_device->GetMemoryProperties());
        result = vkAllocateMemory(device->vkHandle(), &alloc_info, nullptr, &depth_stencil.Memory);
        VkAssert(result);
        result = vkBindImageMemory(device->vkHandle(), depth_stencil.Image, depth_stencil.Memory, 0);
        VkAssert(result);

        const VkImageViewCreateInfo view_info
        {
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            nullptr,
            0,
            depth_stencil.Image,
            VK_IMAGE_VIEW_TYPE_2D,
            depth_stencil.Format,
            {},
            VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 }
        };

        result = vkCreateImageView(device->vkHandle(), &view_info, nullptr, &depth_stencil.View);
        VkAssert(result);

        return depth_stencil;
    }

    VkRenderPass CreateBasicRenderpass(const vpr::Device* device, const vpr::Swapchain* swapchain, VkFormat depth_format)
    {

        VkRenderPass renderpass = VK_NULL_HANDLE;

        const std::array<const VkAttachmentDescription, 2> attachmentDescriptions
        {
            VkAttachmentDescription{
            0, swapchain->ColorFormat(), VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        },
            VkAttachmentDescription{
            0, depth_format, VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        }
        };

        const std::array<VkAttachmentReference, 2> attachmentReferences
        {
            VkAttachmentReference{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
            VkAttachmentReference{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL }
        };

        const VkSubpassDescription subpassDescription
        {
            0,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            0,
            nullptr,
            1,
            &attachmentReferences[0],
            nullptr,
            &attachmentReferences[1],
            0,
            nullptr
        };

        const std::array<VkSubpassDependency, 2> subpassDependencies
        {
                VkSubpassDependency
                {
                    VK_SUBPASS_EXTERNAL,
                    0,
                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_MEMORY_READ_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_DEPENDENCY_BY_REGION_BIT
                },
                VkSubpassDependency
                {
                    0,
                    VK_SUBPASS_EXTERNAL,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_ACCESS_MEMORY_READ_BIT,
                    VK_DEPENDENCY_BY_REGION_BIT
                }
        };

        const VkRenderPassCreateInfo create_info
        {
            VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(attachmentDescriptions.size()),
            attachmentDescriptions.data(),
            1,
            &subpassDescription,
            static_cast<uint32_t>(subpassDependencies.size()),
            subpassDependencies.data()
        };

        VkResult result = vkCreateRenderPass(device->vkHandle(), &create_info, nullptr, &renderpass);
        VkAssert(result);

        return renderpass;
    }

    VkPipeline CreateBasicPipeline(const BasicPipelineCreateInfo& createInfo)
    {
        VkPipeline pipeline = VK_NULL_HANDLE;

        const VkPipelineInputAssemblyStateCreateInfo assembly_info
        {
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            nullptr,
            0,
            createInfo.topology,
            VK_FALSE
        };

        constexpr static VkPipelineViewportStateCreateInfo viewport_info
        {
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            nullptr,
            0,
            1,
            nullptr,
            1,
            nullptr
        };

        const VkPipelineRasterizationStateCreateInfo rasterization_info
        {
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            nullptr,
            0,
            VK_FALSE,
            VK_FALSE,
            VK_POLYGON_MODE_FILL,
            createInfo.cullMode,
            VK_FRONT_FACE_COUNTER_CLOCKWISE,
            VK_FALSE,
            0.0f,
            0.0f,
            0.0f,
            1.0f
        };

        constexpr static VkPipelineMultisampleStateCreateInfo multisample_info
        {
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            nullptr,
            0,
            VK_SAMPLE_COUNT_1_BIT,
            VK_FALSE,
            0.0f,
            nullptr,
            VK_FALSE,
            VK_FALSE
        };

        const VkPipelineDepthStencilStateCreateInfo depth_info
        {
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            nullptr,
            0,
            VK_TRUE,
            VK_TRUE,
            createInfo.depthCompareOp,
            VK_FALSE,
            VK_FALSE,
            VK_STENCIL_OP_ZERO,
            VK_STENCIL_OP_ZERO,
        };

        constexpr static VkPipelineColorBlendAttachmentState colorBlendAttachment
        {
            VK_TRUE,
            VK_BLEND_FACTOR_SRC_ALPHA,
            VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            VK_BLEND_OP_ADD,
            VK_BLEND_FACTOR_SRC_ALPHA,
            VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            VK_BLEND_OP_ADD,
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
        };

        constexpr static VkPipelineColorBlendStateCreateInfo color_blend_info
        {
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            nullptr,
            0,
            VK_FALSE,
            VK_LOGIC_OP_COPY,
            1,
            &colorBlendAttachment,
        { 1.0f, 1.0f, 1.0f, 1.0f }
        };

        constexpr static VkDynamicState dynamic_states[2]
        {
            VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_VIEWPORT
        };

        constexpr static VkPipelineDynamicStateCreateInfo dynamic_state_info
        {
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            nullptr,
            0,
            2,
            dynamic_states
        };

        VkPipelineCreateFlags createFlags = createInfo.pipelineFlags;

        if (createInfo.derivedPipeline != VK_NULL_HANDLE)
        {
            createFlags |= VK_PIPELINE_CREATE_DERIVATIVE_BIT;
        }
        else
        {
            createFlags |= VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT;
        }

        VkGraphicsPipelineCreateInfo pipeline_create_info
        {
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            nullptr,
            VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT,
            createInfo.numStages,
            createInfo.stages,
            createInfo.vertexState,
            &assembly_info,
            nullptr,
            &viewport_info,
            &rasterization_info,
            &multisample_info,
            &depth_info,
            &color_blend_info,
            &dynamic_state_info,
            createInfo.pipelineLayout,
            createInfo.renderPass,
            0,
            createInfo.derivedPipeline,
            -1
        };

        VkResult result =
            vkCreateGraphicsPipelines(createInfo.device->vkHandle(), createInfo.pipelineCache, 1, &pipeline_create_info, nullptr, &pipeline);
        VkAssert(result);

        return pipeline;
    }

    DepthStencil::DepthStencil(
        const vpr::Device* device,
        const vpr::PhysicalDevice* p_device,
        const vpr::Swapchain* swap) : Parent(device->vkHandle())
    {
        *this = CreateDepthStencil(device, p_device, swap);
        Parent = device->vkHandle();
    }

    DepthStencil::~DepthStencil()
    {
        if (Parent == VK_NULL_HANDLE)
        {
            return;
        }

        if (Memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(Parent, Memory, nullptr);
        }

        if (View != VK_NULL_HANDLE)
        {
            vkDestroyImageView(Parent, View, nullptr);
        }

        if (Image != VK_NULL_HANDLE)
        {
            vkDestroyImage(Parent, Image, nullptr);
        }
    }

}