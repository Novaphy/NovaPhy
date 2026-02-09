#pragma once

#include "novaphy/math/math_types.h"

namespace novaphy {

/// Symplectic Euler integration for free rigid bodies.
/// Updates velocity first, then position (semi-implicit).
struct SymplecticEuler {
    /// Integrate linear and angular velocity with forces
    static void integrate_velocity(Vec3f& linear_vel, Vec3f& angular_vel,
                                   const Vec3f& force, const Vec3f& torque,
                                   float inv_mass, const Mat3f& inv_inertia,
                                   const Vec3f& gravity, float dt);

    /// Integrate position and orientation with current velocity
    static void integrate_position(Transform& transform,
                                   const Vec3f& linear_vel,
                                   const Vec3f& angular_vel, float dt);
};

}  // namespace novaphy
