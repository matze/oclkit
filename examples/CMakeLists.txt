find_package(PkgConfig)
pkg_check_modules(GLIB2 glib-2.0>=2.26)
pkg_check_modules(GOBJECT2 gobject-2.0>=2.26)


set(KERNELS "check.cl" "callback.cl" "test.cl")
set(BINARIES
    "check-leak"
    "check-opencl-workgroup-allocation"
    "dump-opencl-binary"
    "test-double-flags"
    "test-profile-timer-resolution"
)
set(DEPS m oclkit ${OPENCL_LIBRARIES})


if (GLIB2_FOUND)
    include_directories(${GLIB2_INCLUDE_DIRS})
    list(APPEND DEPS ${GLIB2_LIBRARIES})
    list(APPEND BINARIES
         "check-allocation-times"
         "check-concurrent-queues"
         "check-infrastructure-times"
         "check-launch-latencies"
         "check-launch-latencies-chained"
         "check-max-allocation"
         "check-pci-bandwidth"
         "check-queue-impact"
         "test-regressions"
    )

    if (GOBJECT2_FOUND)
        list(APPEND BINARIES
             "test-callback"
        )

        include_directories(${GOBJECT2_INCLUDE_DIRS})
        list(APPEND DEPS ${GOBJECT2_LIBRARIES})
    endif ()
endif ()


foreach (BINARY ${BINARIES})
    add_executable(${BINARY} ${BINARY}.c)

    target_link_libraries(${BINARY} ${DEPS})
endforeach ()


foreach (KERNEL ${KERNELS})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}
                   ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL})
endforeach ()
