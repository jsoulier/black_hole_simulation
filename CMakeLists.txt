cmake_minimum_required(VERSION 3.24)
project(savepoint)

add_library(savepoint_sqlite3 src/drivers/sqlite3.c)
add_library(savepoint
    src/base.cpp
    src/drivers/null.cpp
    src/drivers/sqlite3.cpp
    src/log.cpp
    src/savepoint.cpp
)
add_library(savepoint::savepoint ALIAS savepoint)
target_link_libraries(savepoint PRIVATE savepoint_sqlite3)
target_include_directories(savepoint PUBLIC include)
set_target_properties(savepoint PROPERTIES CXX_STANDARD 23)
set_target_properties(savepoint PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
set_target_properties(savepoint_sqlite3 PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_precompile_headers(savepoint PUBLIC
    <savepoint/base.hpp>
    <savepoint/driver.hpp>
    <savepoint/fwd.hpp>
    <savepoint/id.hpp>
    <savepoint/log.hpp>
    <savepoint/savepoint.hpp>
    <savepoint/traits.hpp>
    <savepoint/version.hpp>
    <savepoint/visitor.hpp>
)

if(PROJECT_IS_TOP_LEVEL)
    set(BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BINARY_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${BINARY_DIR})
    make_directory(${BINARY_DIR})
    function(savepoint_example NAME)
        set(TARGET savepoint_${NAME}_example)
        add_executable(${TARGET} examples/${NAME}.cpp)
        set_target_properties(${TARGET} PROPERTIES CXX_STANDARD 23)
        target_link_libraries(${TARGET} savepoint)
        add_test(NAME ${TARGET} COMMAND ${TARGET})
    endfunction()
    savepoint_example(basic_upgraded)
    savepoint_example(basic_usage)
    savepoint_example(external_types)
    savepoint_example(nested_types)
    savepoint_example(polymorphic_types)
    savepoint_example(spatial_types)
    function(savepoint_test NAME)
        set(TARGET savepoint_${NAME}_test)
        add_executable(${TARGET} tests/${NAME}.cpp)
        set_target_properties(${TARGET} PROPERTIES CXX_STANDARD 23)
        target_link_libraries(${TARGET} savepoint)
        add_test(NAME ${TARGET} COMMAND ${TARGET})
    endfunction()
    savepoint_test(coverage)
    savepoint_test(traits)
    enable_testing()
endif()