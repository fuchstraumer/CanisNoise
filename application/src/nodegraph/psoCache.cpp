#include "nodegraph/psoCache.hpp"
#include "nodegraph/CompiledShaders.hpp"
#include "RenderingContext.hpp"
#include "LogicalDevice.hpp"
#include "PhysicalDevice.hpp"
#include "PipelineCache.hpp"
#include "PipelineLayout.hpp"
#include "DescriptorSetLayout.hpp"
#include "ShaderModule.hpp"
#include "vkAssert.hpp"
#include <array>
#include <unordered_map>

constexpr static std::array<VkDescriptorSetLayoutBinding, 5> combinerLayoutBindings
{
    VkDescriptorSetLayoutBinding{ 0, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    VkDescriptorSetLayoutBinding{ 1, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    VkDescriptorSetLayoutBinding{ 2, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    VkDescriptorSetLayoutBinding{ 3, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    VkDescriptorSetLayoutBinding{ 4, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
};

constexpr static VkPushConstantRange combinerPushConstantRange
{
    VK_SHADER_STAGE_COMPUTE_BIT,
    0u,
    sizeof(float) * 3u
};

constexpr static std::array<VkDescriptorSetLayoutBinding, 4> modifierLayoutBindings
{
    VkDescriptorSetLayoutBinding{ 0, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    VkDescriptorSetLayoutBinding{ 1, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    VkDescriptorSetLayoutBinding{ 2, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    VkDescriptorSetLayoutBinding{ 3, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
};

constexpr static VkPushConstantRange modifierPushConstantRange
{
    VK_SHADER_STAGE_COMPUTE_BIT,
    0u,
    sizeof(float) * 4u
};

constexpr static VkComputePipelineCreateInfo baseComputePipelineInfo
{
    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    nullptr,
    0,
    VkPipelineShaderStageCreateInfo(),
    VK_NULL_HANDLE,
    VK_NULL_HANDLE,
    -1
};

constexpr static std::array<VkSpecializationMapEntry, 4> combinerSpecializationMapEntries
{
    VkSpecializationMapEntry{ 0, 0 * sizeof(uint32_t), sizeof(uint32_t) }, // Operation type
    VkSpecializationMapEntry{ 1, 1 * sizeof(uint32_t), sizeof(uint32_t) }, // MaxX: max size of input image data
    VkSpecializationMapEntry{ 2, 2 * sizeof(uint32_t), sizeof(uint32_t) }, // MaxY: ^
    VkSpecializationMapEntry{ 3, 3 * sizeof(uint32_t), sizeof(uint32_t) }  // MaxZ: ^
};

constexpr static std::array<uint32_t, 4> combinerSpecializationDefaultVals
{
    0u, // base operation is just an add
    256u,
    256u,
    1u // z dimension is almost never used
};

static VkSpecializationInfo combinerSpecializationInfo
{
    static_cast<uint32_t>(combinerSpecializationMapEntries.size()),
    combinerSpecializationMapEntries.data(),
    combinerSpecializationMapEntries.size() * sizeof(uint32_t),
    reinterpret_cast<const void*>(combinerSpecializationDefaultVals.data())
};

constexpr static std::array<VkSpecializationMapEntry, 9> modifierSpecializationMapEntries
{
    VkSpecializationMapEntry{ 0, 0 * sizeof(uint32_t), sizeof(uint32_t) }, // Operation type
    VkSpecializationMapEntry{ 1, 1 * sizeof(uint32_t), sizeof(uint32_t) }, // MaxX: max size of input image data
    VkSpecializationMapEntry{ 2, 2 * sizeof(uint32_t), sizeof(uint32_t) }, // MaxY: ^
    VkSpecializationMapEntry{ 3, 3 * sizeof(uint32_t), sizeof(uint32_t) }, // MaxZ: ^
    VkSpecializationMapEntry{ 4, 4 * sizeof(uint32_t), sizeof(uint32_t) }, // NumControlPoints: size of control points array for terrace and curve
    VkSpecializationMapEntry{ 5, 5 * sizeof(uint32_t), sizeof(uint32_t) }, // TerraceInvert: does what it says on the tin
    VkSpecializationMapEntry{ 6, 6 * sizeof(uint32_t), sizeof(uint32_t) }, // MinX: Max size of output image data for downsample node
    VkSpecializationMapEntry{ 7, 7 * sizeof(uint32_t), sizeof(uint32_t) }, // MinY: ^
    VkSpecializationMapEntry{ 8, 8 * sizeof(uint32_t), sizeof(uint32_t) }, // MinZ: ^
};

constexpr static std::array<uint32_t, modifierSpecializationMapEntries.size()> modifierSpecializationDefaultVals
{
    0u, // base op is just abs()
    256u,
    256u,
    1u,
    16u, // more than enough control points
    static_cast<uint32_t>(VK_FALSE),
    256u / 4u,
    256u / 4u,
    1u
};

static VkSpecializationInfo modifierSpecializationInfo
{
    static_cast<uint32_t>(modifierSpecializationMapEntries.size()),
    modifierSpecializationMapEntries.data(),
    modifierSpecializationMapEntries.size() * sizeof(uint32_t),
    reinterpret_cast<const void*>(modifierSpecializationDefaultVals.data())
};

struct PsoCacheImpl
{
    PsoCacheImpl() = default;
    ~PsoCacheImpl();
    
    void Initialize(const VkDevice device, const VkPhysicalDevice host_physical_device);
    void createCombinerPipelines(const VkDevice device);
    void createModifierPipelines(const VkDevice device);

    using node_pso_map = std::unordered_map<uint8_t, VkPipeline>;
    std::array<node_pso_map, 3> nodePSOs;
    std::array<std::unique_ptr<vpr::DescriptorSetLayout>, 3> descriptorSetLayouts;
    std::array<std::unique_ptr<vpr::PipelineLayout>, 3> pipelineLayouts;
    std::unique_ptr<vpr::PipelineCache> pipelineCache;
    
    // Since we change how these work with spec constants, we only need these two
    std::unique_ptr<vpr::ShaderModule> combinerShaderModule;
    std::unique_ptr<vpr::ShaderModule> modifierShaderModule;
    // The generators, unfortunately, are all too unique and need their own shader modules :(
    std::unordered_map<uint8_t, std::unique_ptr<vpr::ShaderModule>> generatorShaderModules;
};

PsoCacheImpl::~PsoCacheImpl()
{
    auto& renderingContext = RenderingContext::Get();
    const VkDevice device = renderingContext.Device()->vkHandle();

    // nodePSOs = std::array<std::unordered_map<uint8_t, VkPipeline>, 3>
    // one map per major node type: combiners, modifiers, generators
    // each distinct category represents a bunch of shared shader code
    for (auto& pipelineMap : nodePSOs) // take as reference since it's a whole map
    {
        // cleanup all used PSOs
        // taken by value to structured binding since it's a (uint8_t, void*) effectively: reference saves 8 bits lol
        for (auto [type, pso] : pipelineMap)
        {
            vkDestroyPipeline(device, pso, nullptr);
        }
    }
}

void PsoCacheImpl::Initialize(const VkDevice device, const VkPhysicalDevice host_physical_device)
{
    pipelineCache = std::make_unique<vpr::PipelineCache>(device, host_physical_device, typeid(PsoCacheImpl).hash_code());
    createCombinerPipelines(device);
    createModifierPipelines(device);
}

void PsoCacheImpl::createCombinerPipelines(const VkDevice device)
{
    const size_t typeIdx = static_cast<size_t>(NodeType::Combiner);
    descriptorSetLayouts[typeIdx] = std::make_unique<vpr::DescriptorSetLayout>(device);
    descriptorSetLayouts[typeIdx]->AddDescriptorBindings(static_cast<uint32_t>(combinerLayoutBindings.size()), combinerLayoutBindings.data());

    pipelineLayouts[typeIdx] = std::make_unique<vpr::PipelineLayout>(device);
    pipelineLayouts[typeIdx]->Create(1u, &combinerPushConstantRange, 1u, &descriptorSetLayouts[typeIdx]->vkHandle());

    combinerShaderModule = std::make_unique<vpr::ShaderModule>(
        device,
        VK_SHADER_STAGE_COMPUTE_BIT,
        combinersCompiledSpvSource,
        static_cast<uint32_t>(sizeof(combinersCompiledSpvSource)));

    auto& combinerPSOs = nodePSOs[typeIdx];

    auto combinerSpecValues = combinerSpecializationDefaultVals;
    VkSpecializationInfo combinerSpecInfo = combinerSpecializationInfo;

    VkPipelineShaderStageCreateInfo combinerShaderInfo
    {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_COMPUTE_BIT,
        combinerShaderModule->vkHandle(),
        "main",
        &combinerSpecInfo
    };

    VkComputePipelineCreateInfo pipelineInfo = baseComputePipelineInfo;
    pipelineInfo.stage = combinerShaderInfo;
    pipelineInfo.layout = pipelineLayouts[typeIdx]->vkHandle();

    const uint32_t numCombinerNodeTypes = static_cast<uint32_t>(CombinerNodes::Count);
    for (uint32_t i = 0; i < numCombinerNodeTypes; ++i)
    {
        // lol this sucks to just cast right back to the uint8 but... alright
        const uint8_t nodeType = static_cast<uint8_t>(i);
        combinerSpecValues[0] = i;
        combinerSpecInfo.pData = combinerSpecValues.data();
        VkPipeline newPipeline = VK_NULL_HANDLE;
        VkResult result = vkCreateComputePipelines(device, pipelineCache->vkHandle(), 1u, &pipelineInfo, nullptr, &newPipeline);
        VkAssert(result);
        combinerPSOs.emplace(nodeType, newPipeline);
    }
}

void PsoCacheImpl::createModifierPipelines(const VkDevice device)
{
    const size_t typeIdx = static_cast<size_t>(NodeType::Modifier);
    descriptorSetLayouts[typeIdx] = std::make_unique<vpr::DescriptorSetLayout>(device);
    descriptorSetLayouts[typeIdx]->AddDescriptorBindings(static_cast<uint32_t>(modifierLayoutBindings.size()), modifierLayoutBindings.data());

    pipelineLayouts[typeIdx] = std::make_unique<vpr::PipelineLayout>(device);
    pipelineLayouts[typeIdx]->Create(1u, &modifierPushConstantRange, 1u, &descriptorSetLayouts[typeIdx]->vkHandle());

    modifierShaderModule = std::make_unique<vpr::ShaderModule>(
        device,
        VK_SHADER_STAGE_COMPUTE_BIT,
        modifiersCompiledSpvSource,
        static_cast<uint32_t>(sizeof(modifiersCompiledSpvSource)));

    auto& modifierPSOs = nodePSOs[typeIdx];

    auto modifierSpecValues = modifierSpecializationDefaultVals;
    VkSpecializationInfo modifierSpecInfo = modifierSpecializationInfo;
    modifierSpecInfo.pData = modifierSpecValues.data();

    VkPipelineShaderStageCreateInfo modifierShaderInfo
    {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_COMPUTE_BIT,
        modifierShaderModule->vkHandle(),
        "main",
        &modifierSpecInfo
    };

    VkComputePipelineCreateInfo pipelineInfo = baseComputePipelineInfo;
    pipelineInfo.stage = modifierShaderInfo;
    pipelineInfo.layout = pipelineLayouts[typeIdx]->vkHandle();

    const uint32_t numModifierNodeTypes = static_cast<uint32_t>(ModifierNodes::Count);
    for (uint32_t i = 0; i < numModifierNodeTypes; ++i)
    {
        const uint8_t nodeType = static_cast<uint8_t>(i);
        modifierSpecValues[0] = i;
        VkPipeline newPipeline = VK_NULL_HANDLE;
        VkResult result = vkCreateComputePipelines(device, pipelineCache->vkHandle(), 1u, &pipelineInfo, nullptr, &newPipeline);
        VkAssert(result);
        modifierPSOs.emplace(nodeType, newPipeline);
    }

}

PsoCache::PsoCache() : impl(std::make_unique<PsoCacheImpl>())
{
}

PsoCache::~PsoCache()
{
}

PsoCache& PsoCache::Get() noexcept
{
    static PsoCache cache;
    return cache;
}

void PsoCache::PreheatCache()
{
    auto& renderingContext = RenderingContext::Get();
    impl->Initialize(renderingContext.Device()->vkHandle(), renderingContext.PhysicalDevice()->vkHandle());
}

VkPipeline PsoCache::GetPipeline(const NodeType node_type, const uint8_t node_subtype)
{
    auto& currPSOs = impl->nodePSOs[static_cast<size_t>(node_type)];
    auto foundIter = currPSOs.find(node_subtype);
    if (foundIter != currPSOs.end())
    {
        return foundIter->second;
    }
    else
    {
        return VK_NULL_HANDLE;
    }
}
