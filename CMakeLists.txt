cmake_minimum_required(VERSION 3.6)
project(CanisNoise)

find_package(Vulkan REQUIRED)
add_subdirectory(submodules/vulpesrender)
add_subdirectory(application)
