#include <pybind11/pybind11.h>

#include "novaphy/novaphy.h"

namespace py = pybind11;

// Forward declarations for sub-module binders
void bind_math(py::module_& m);
void bind_core(py::module_& m);
void bind_collision(py::module_& m);
void bind_sim(py::module_& m);

PYBIND11_MODULE(_core, m) {
    m.doc() = "NovaPhy: A 3D physics engine for embodied intelligence";

    m.def("version", &novaphy::version, "Returns the NovaPhy version string");

    bind_math(m);
    bind_core(m);
    bind_collision(m);
    bind_sim(m);
}
