find_package(PkgConfig)
pkg_check_modules(GLIB2 glib-2.0>=2.26)


set(KERNELS "test.cl" "lut.cl")
set(BINARIES "info" "leak" "maxlocal")
set(DEPS m oclkit ${OPENCL_LIBRARIES})


if (GLIB2_FOUND)
    include_directories(${GLIB2_INCLUDE_DIRS})
    list(APPEND DEPS ${GLIB2_LIBRARIES})
    list(APPEND BINARIES "lut" "ptransfer" "regr")
endif ()


foreach (BINARY ${BINARIES})
    add_executable(${BINARY} ${BINARY}.c)

    target_link_libraries(${BINARY} ${DEPS})
endforeach ()


foreach (KERNEL ${KERNELS})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}
                   ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL})
endforeach ()
