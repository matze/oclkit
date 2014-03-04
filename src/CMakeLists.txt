cmake_minimum_required(VERSION 2.6)

add_library(oclkit ocl.c)

target_link_libraries(oclkit ${OPENCL_LIBRARIES})
