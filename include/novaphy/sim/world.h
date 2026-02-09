#pragma once

#include <vector>

#include "novaphy/collision/broadphase.h"
#include "novaphy/collision/narrowphase.h"
#include "novaphy/core/contact.h"
#include "novaphy/core/model.h"
#include "novaphy/dynamics/free_body_solver.h"
#include "novaphy/dynamics/integrator.h"
#include "novaphy/sim/state.h"

namespace novaphy {

/// Top-level simulation container.
/// Owns the Model, SimState, broadphase, and solver.
class World {
public:
    explicit World(const Model& model, SolverSettings solver_settings = {});

    /// Advance simulation by dt seconds
    void step(float dt);

    /// Set gravity vector (default: (0, -9.81, 0))
    void set_gravity(const Vec3f& g) { gravity_ = g; }
    const Vec3f& gravity() const { return gravity_; }

    /// Access simulation state
    SimState& state() { return state_; }
    const SimState& state() const { return state_; }

    /// Access the model
    const Model& model() const { return model_; }

    /// Access solver settings
    SolverSettings& solver_settings() { return solver_.settings(); }

    /// Get contact points from last step (for debugging/visualization)
    const std::vector<ContactPoint>& contacts() const { return contacts_; }

    /// Apply an external force to a body (accumulated for next step)
    void apply_force(int body_index, const Vec3f& force);
    void apply_torque(int body_index, const Vec3f& torque);

private:
    Model model_;
    SimState state_;
    SweepAndPrune broadphase_;
    FreeBodySolver solver_;
    Vec3f gravity_ = Vec3f(0.0f, -9.81f, 0.0f);
    std::vector<ContactPoint> contacts_;
};

}  // namespace novaphy
