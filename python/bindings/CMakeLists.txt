# Python bindings via pybind11

pybind11_add_module(_core
    bind_main.cpp
    bind_math.cpp
    bind_core.cpp
    bind_collision.cpp
    bind_sim.cpp
)

target_link_libraries(_core PRIVATE
    novaphy_core
)

target_compile_definitions(_core PRIVATE
    VERSION_INFO=${PROJECT_VERSION}
)

install(TARGETS _core DESTINATION novaphy)
