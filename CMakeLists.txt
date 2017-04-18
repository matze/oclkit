cmake_minimum_required(VERSION 2.6)
project(oclkit C)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(OpenCL REQUIRED)

include_directories(
    ${OPENCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

add_definitions("-std=c99 -Wall -Wextra -pedantic")
add_definitions("-Wno-unused-parameter")
add_definitions("-DCL_USE_DEPRECATED_OPENCL_2_0_APIS")

add_subdirectory(src)
add_subdirectory(examples)
