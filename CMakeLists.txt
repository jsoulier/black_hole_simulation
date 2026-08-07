cmake_minimum_required(VERSION 3.24)
project(black_hole_simulation C)

set(BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${BINARY_DIR})
make_directory(${BINARY_DIR})

add_subdirectory(SDL)
add_executable(black_hole_simulation WIN32 main.c)
set_target_properties(black_hole_simulation PROPERTIES C_STANDARD 11)
target_link_libraries(black_hole_simulation PRIVATE SDL3::SDL3)

find_program(SHADERCROSS shadercross)
foreach(FORMAT dxil msl spv)
    if(SHADERCROSS)
        add_custom_command(
            TARGET black_hole_simulation POST_BUILD
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMAND ${SHADERCROSS} shader.comp -s hlsl -o bin/shader.comp.${FORMAT})
    endif()
    add_custom_command(
        TARGET black_hole_simulation POST_BUILD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy bin/shader.comp.${FORMAT} ${BINARY_DIR})
endforeach()
