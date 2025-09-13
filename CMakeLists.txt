cmake_minimum_required(VERSION 3.24)
project(black_hole_simulation)

set(BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${BINARY_DIR})
make_directory(${BINARY_DIR})

set(GLM_BUILD_LIBRARY OFF)
add_subdirectory(SDL)
add_subdirectory(glm)
add_executable(black_hole_simulation WIN32 main.cpp shader.cpp)
set_target_properties(black_hole_simulation PROPERTIES CXX_STANDARD 23)
target_link_libraries(black_hole_simulation PRIVATE SDL3::SDL3 glm)

function(add_shader FILE)
    set(DEPENDS ${ARGN})
    set(HLSL ${CMAKE_SOURCE_DIR}/${FILE})
    set(SPV ${CMAKE_SOURCE_DIR}/bin/${FILE}.spv)
    set(DXIL ${CMAKE_SOURCE_DIR}/bin/${FILE}.dxil)
    set(MSL ${CMAKE_SOURCE_DIR}/bin/${FILE}.msl)
    set(JSON ${CMAKE_SOURCE_DIR}/bin/${FILE}.json)
    if(MSVC)
        set(SHADERCROSS SDL_shadercross/msvc/shadercross.exe)
    endif()
    function(compile OUTPUT)
        add_custom_command(
            OUTPUT ${OUTPUT}
            COMMAND ${SHADERCROSS} ${HLSL} -s hlsl -o ${OUTPUT} -I src
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            DEPENDS ${HLSL} ${DEPENDS}
            COMMENT ${OUTPUT}
        )
        get_filename_component(NAME ${OUTPUT} NAME)
        string(REPLACE . _ NAME ${NAME})
        set(NAME compile_${NAME})
        add_custom_target(${NAME} DEPENDS ${OUTPUT})
        add_dependencies(black_hole_simulation ${NAME})
    endfunction()
    if(MSVC)
        compile(${SPV})
        compile(${DXIL})
        compile(${MSL})
        compile(${JSON})
    endif()
    function(package OUTPUT)
        get_filename_component(NAME ${OUTPUT} NAME)
        set(BINARY ${BINARY_DIR}/${NAME})
        add_custom_command(
            OUTPUT ${BINARY}
            COMMAND ${CMAKE_COMMAND} -E copy ${OUTPUT} ${BINARY}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            DEPENDS ${OUTPUT}
            COMMENT ${BINARY}
        )
        string(REPLACE . _ NAME ${NAME})
        set(NAME package_${NAME})
        add_custom_target(${NAME} DEPENDS ${BINARY})
        add_dependencies(black_hole_simulation ${NAME})
    endfunction()
    package(${SPV})
    if(WIN32)
        package(${DXIL})
    elseif(APPLE)
        package(${MSL})
    endif()
    package(${JSON})
endfunction()
add_shader(geodesic.comp config.h)

configure_file(LICENSE.txt ${BINARY_DIR} COPYONLY)
configure_file(README.md ${BINARY_DIR} COPYONLY)