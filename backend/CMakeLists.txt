
SET(COMBINER_SHADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/combiners/add.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/combiners/blend.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/combiners/divide.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/combiners/max.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/combiners/min.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/combiners/multiply.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/combiners/power.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/combiners/select.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/combiners/subtract.comp"
)

SET(GENERATOR_SHADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/generators/Billow3D.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/generators/Decarpientier2D.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/generators/Decarpientier3D.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/generators/FBM3D.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/generators/Jordan2D.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/generators/perlin2d.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/generators/Ridged3D.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/generators/simplex3d_deriv.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/generators/simplex3d.comp"
)

SET(MODIFIER_SHADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/modifiers/abs.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/modifiers/clamp.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/modifiers/curve.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/modifiers/downsample.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/modifiers/scalebias.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/modifiers/terrace.comp"
)

SOURCE_GROUP("combiners" FILES ${COMBINER_SHADERS})
SOURCE_GROUP("generators" FILES ${GENERATOR_SHADERS})
SOURCE_GROUP("modifiers" FILES ${MODIFIER_SHADERS})

FIND_PACKAGE(Vulkan REQUIRED)

ADD_LIBRARY(CANIS_VK_BACKEND SHARED ${COMBINER_SHADERS} ${GENERATOR_SHADERS} ${MODIFIER_SHADERS} 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Dispatcher.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/Dispatcher.cpp")
TARGET_INCLUDE_DIRECTORIES(CANIS_VK_BACKEND PRIVATE ${Vulkan_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/include" "../common/"
    "${CMAKE_CURRENT_SOURCE_DIR}/ext/include")
TARGET_LINK_LIBRARIES(CANIS_VK_BACKEND PRIVATE ${Vulkan_LIBRARIES})

TARGET_COMPILE_DEFINITIONS(CANIS_VK_BACKEND PRIVATE -DBUILDING_DLL)

IF(MSVC) # MSVC refuses to use C++17 unless we set it manually
    TARGET_COMPILE_OPTIONS(CANIS_VK_BACKEND PRIVATE "/std:c++17")
ELSE()
    TARGET_COMPILE_OPTIONS(CANIS_VK_BACKEND PRIVATE "-std=c++17")
ENDIF()
