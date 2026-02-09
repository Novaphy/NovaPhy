#pragma once

#include <vector>

#include "novaphy/math/math_types.h"

namespace novaphy {

/// Per-body simulation state: transforms, velocities, forces.
struct SimState {
    std::vector<Transform> transforms;
    std::vector<Vec3f> linear_velocities;
    std::vector<Vec3f> angular_velocities;
    std::vector<Vec3f> forces;
    std::vector<Vec3f> torques;

    /// Initialize state for n bodies from initial transforms
    void init(int n, const std::vector<Transform>& initial_transforms);

    /// Clear all accumulated forces/torques (called each step)
    void clear_forces();

    /// Apply an external force at the body's center of mass
    void apply_force(int body_index, const Vec3f& force);

    /// Apply a torque to a body
    void apply_torque(int body_index, const Vec3f& torque);
};

}  // namespace novaphy
