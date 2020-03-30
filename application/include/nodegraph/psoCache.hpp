#pragma once
#ifndef CANIS_NOISE_PSO_CACHE_HPP
#define CANIS_NOISE_PSO_CACHE_HPP
#include "NodeTypes.hpp"
#include <memory>
#include "vulkan/vulkan_core.h"

struct PsoCacheImpl;

class PsoCache
{
public:
    ~PsoCache();
    static PsoCache& Get() noexcept;
    void PreheatCache();
    VkPipeline GetPipeline(const NodeType node_type, const uint8_t node_subtype);
private:
    PsoCache();
    std::unique_ptr<PsoCacheImpl> impl = nullptr;
};

#endif //!CANISE_NOISE_PSO_CACHE_HPP
