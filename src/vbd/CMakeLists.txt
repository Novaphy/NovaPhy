add_library(novaphy_vbd SHARED
    vbd_collide.cpp
    vbd_world.cpp
    vbd_solver.cpp
)

if(CMAKE_CUDA_COMPILER)
    target_sources(novaphy_vbd PRIVATE
        vbd_solver_cuda.cu
    )
    target_compile_definitions(novaphy_vbd PRIVATE NOVAPHY_VBD_CUDA)
    # Enable separable compilation for CUDA and select architectures.
    set_target_properties(novaphy_vbd PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
    )
endif()

target_include_directories(novaphy_vbd PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(novaphy_vbd PUBLIC
    novaphy_core
)

