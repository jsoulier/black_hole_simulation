project(savepoint_imgui)
add_library(imgui STATIC
    imgui.cpp
    imgui_demo.cpp
    imgui_draw.cpp
    imgui_impl_sdl3.cpp
    imgui_impl_sdlrenderer3.cpp
    imgui_tables.cpp
    imgui_widgets.cpp
)
target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(imgui PROPERTIES CXX_STANDARD 17)
target_link_libraries(imgui PUBLIC SDL3::SDL3)
