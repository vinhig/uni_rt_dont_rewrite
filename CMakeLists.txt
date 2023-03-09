cmake_minimum_required(VERSION 3.24)
project(uni_rt_dont_rewrite)

set(CMAKE_CXX_STANDARD 17)

find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(embree REQUIRED)
find_package(glm REQUIRED)

add_library(imgui imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp imgui/imgui.cpp imgui/backends/imgui_impl_sdl2.cpp imgui/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui PUBLIC imgui/)
target_link_libraries(imgui SDL2::SDL2)

add_executable(uni_rt_dont_rewrite
    source/buffer_view.cpp
    source/file_mapping.cpp
    source/flatten_gltf.cpp
    source/gltf_types.cpp
    source/main.cpp
    source/render.cpp
    source/glad.c
    source/mesh.cpp
    source/scene.cpp
    source/util.cpp)

target_link_libraries(uni_rt_dont_rewrite embree)
target_link_libraries(uni_rt_dont_rewrite SDL2::SDL2)
target_link_libraries(uni_rt_dont_rewrite OpenGL)
target_link_libraries(uni_rt_dont_rewrite imgui)
target_link_libraries(uni_rt_dont_rewrite glm)
target_include_directories(uni_rt_dont_rewrite PUBLIC imgui/)
target_include_directories(uni_rt_dont_rewrite PUBLIC source/parallel_hashmap)
