#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "novaphy/core/model.h"
#include "novaphy/core/model_builder.h"
#include "novaphy/sim/state.h"
#include "novaphy/sim/world.h"

namespace py = pybind11;
using namespace novaphy;

void bind_sim(py::module_& m) {
    // --- ModelBuilder ---
    py::class_<ModelBuilder>(m, "ModelBuilder")
        .def(py::init<>())
        .def("add_body", &ModelBuilder::add_body,
             py::arg("body"),
             py::arg("transform") = Transform::identity(),
             "Add a dynamic rigid body. Returns body index.")
        .def("add_shape", &ModelBuilder::add_shape,
             py::arg("shape"),
             "Add a collision shape. Returns shape index.")
        .def("add_ground_plane", &ModelBuilder::add_ground_plane,
             py::arg("y") = 0.0f,
             py::arg("friction") = 0.5f,
             py::arg("restitution") = 0.0f,
             "Add an infinite ground plane. Returns shape index.")
        .def("build", &ModelBuilder::build, "Build the immutable Model.")
        .def_property_readonly("num_bodies", &ModelBuilder::num_bodies)
        .def_property_readonly("num_shapes", &ModelBuilder::num_shapes);

    // --- Model ---
    py::class_<Model>(m, "Model")
        .def_property_readonly("num_bodies", &Model::num_bodies)
        .def_property_readonly("num_shapes", &Model::num_shapes)
        .def_readonly("bodies", &Model::bodies)
        .def_readonly("shapes", &Model::shapes);

    // --- SolverSettings ---
    py::class_<SolverSettings>(m, "SolverSettings")
        .def(py::init<>())
        .def_readwrite("velocity_iterations", &SolverSettings::velocity_iterations)
        .def_readwrite("baumgarte", &SolverSettings::baumgarte)
        .def_readwrite("slop", &SolverSettings::slop)
        .def_readwrite("warm_starting", &SolverSettings::warm_starting);

    // --- SimState ---
    py::class_<SimState>(m, "SimState")
        .def(py::init<>())
        .def_readonly("transforms", &SimState::transforms)
        .def_readonly("linear_velocities", &SimState::linear_velocities)
        .def_readonly("angular_velocities", &SimState::angular_velocities)
        .def("apply_force", &SimState::apply_force)
        .def("apply_torque", &SimState::apply_torque);

    // --- World ---
    py::class_<World>(m, "World")
        .def(py::init<const Model&, SolverSettings>(),
             py::arg("model"),
             py::arg("solver_settings") = SolverSettings{})
        .def("step", &World::step, py::arg("dt"),
             "Advance simulation by dt seconds.")
        .def("set_gravity", &World::set_gravity, py::arg("gravity"),
             "Set gravity vector.")
        .def_property_readonly("gravity", &World::gravity)
        .def_property_readonly("state", py::overload_cast<>(&World::state),
             py::return_value_policy::reference_internal)
        .def_property_readonly("model", &World::model,
             py::return_value_policy::reference_internal)
        .def_property_readonly("contacts", &World::contacts,
             py::return_value_policy::reference_internal)
        .def("apply_force", &World::apply_force,
             py::arg("body_index"), py::arg("force"))
        .def("apply_torque", &World::apply_torque,
             py::arg("body_index"), py::arg("torque"));
}
