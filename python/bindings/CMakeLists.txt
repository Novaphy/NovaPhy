# Python bindings via pybind11

set(BIND_SOURCES
    bind_main.cpp
    bind_math.cpp
    bind_core.cpp
    bind_collision.cpp
    bind_performance.cpp
    bind_sim.cpp
    bind_dynamics.cpp
    bind_fluid.cpp
    bind_io.cpp
)

if(NOVAPHY_WITH_IPC)
    list(APPEND BIND_SOURCES bind_ipc.cpp)
endif()

pybind11_add_module(_core ${BIND_SOURCES})

target_link_libraries(_core PRIVATE
    novaphy_core
)

if(NOVAPHY_WITH_IPC)
    target_link_libraries(_core PRIVATE novaphy_ipc)
    # C++20 needed for bind_ipc.cpp which includes libuipc headers
    set_source_files_properties(bind_ipc.cpp PROPERTIES
        COMPILE_FLAGS "$<$<CXX_COMPILER_ID:MSVC>:/std:c++20>$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-std=c++20>"
    )

    set_target_properties(_core PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin/"
    )

    if (NOVAPHY_BUNDLE_UIPC)
        set_target_properties(_core PROPERTIES
            INSTALL_RPATH "$ORIGIN/lib/novaphy-bundled-libs/"
        )
    endif()
endif()

if (NOVAPHY_ENABLE_INSTALL)
    if (SKBUILD)
        # With scikit-build, the Python extension module is built in-place and should not be installed by CMake.
        install(TARGETS _core 
            COMPONENT pybind
            DESTINATION novaphy
        )
    else()
        # TODO Without scikit-build, we need to generate a simple setup.py and related files.
    endif()
endif()