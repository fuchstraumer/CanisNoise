
set(vpr_include_directories
    "../submodules/vulpesrender/command/include"
    "../submodules/vulpesrender/core/include"
    "../submodules/vulpesrender/render/include"
    "../submodules/vulpesrender/resource/include"
    "../submodules/vulpesrender/sync/include")

set(canis_noise_headers
    "${CMAKE_CURRENT_SOURCE_DIR}/include/EditorScene.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Image.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ImGuiWrapper.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/PipelineExecutableInfo.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/PlatformWindow.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/RenderingContext.hpp")

set(canis_noise_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/EditorScene.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Image.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ImGuiWrapper.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PipelineExecutableInfo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/PlatformWindow.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/RenderingContext.cpp")

set(canis_noise_resource_context
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ResourceContext.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ResourceTypes.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/impl/ResourceContextImpl.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/impl/ResourceContextImpl.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/impl/TransferSystem.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/impl/TransferSystem.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/impl/UploadBuffer.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/impl/UploadBuffer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ResourceContext.cpp")

set(canis_noise_nodegraph_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/include/nodegraph/psoCache.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/nodegraph/NodeTypes.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/nodegraph/NodeCreation.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/nodegraph/psoCache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/nodegraph/NodeCreation.cpp")

set(canis_noise_imgui_sources
    "../submodules/imgui/imgui.h"
    "../submodules/imgui/imgui.cpp"
    "../submodules/imgui/imgui_draw.cpp"
    "../submodules/imgui/imconfig.h"
    "../submodules/imgui/imgui_demo.cpp"
    "../submodules/imgui/imgui_internal.h"
    "../submodules/imgui/imgui_widgets.cpp")

set(canis_noise_imgui_nodegraph_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/include/imgui_node_editor.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_nodegraph/crude_json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_nodegraph/crude_json.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_nodegraph/imgui_bezier_math.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_nodegraph/imgui_bezier_math.inl"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_nodegraph/imgui_canvas.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_nodegraph/imgui_canvas.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_nodegraph/imgui_extra_math.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_nodegraph/imgui_extra_math.inl"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_nodegraph/imgui_node_editor_api.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_nodegraph/imgui_node_editor_internal.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_nodegraph/imgui_node_editor_internal.inl"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_nodegraph/imgui_node_editor.cpp")

source_group("resource_context" FILES ${canis_noise_resource_context})
source_group("nodegraph" FILES ${canis_noise_nodegraph_sources})
source_group("imgui" FILES ${canis_noise_imgui_sources})
source_group("imgui_nodegraph" FILES ${canis_noise_imgui_nodegraph_sources})

add_executable(canis_noise
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    ${canis_noise_headers}
    ${canis_noise_sources}
    ${canis_noise_resource_context}
    ${canis_noise_nodegraph_sources}
    ${canis_noise_imgui_sources}
    ${canis_noise_imgui_nodegraph_sources})

if(MSVC)
    target_compile_options(canis_noise PRIVATE "/std:c++latest")
else()
    set_target_properties(canis_noise PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
endif()

target_include_directories(canis_noise PRIVATE
    ${Vulkan_INCLUDE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "../submodules/vma/src"
    "../submodules/svs"
    "../submodules/imgui")

target_link_libraries(canis_noise PRIVATE
    vpr_core vpr_command vpr_sync vpr_resource vpr_render glfw ${Vulkan_LIBRARY})