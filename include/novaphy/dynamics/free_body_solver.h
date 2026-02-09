#pragma once

#include <vector>

#include "novaphy/core/body.h"
#include "novaphy/core/contact.h"
#include "novaphy/math/math_types.h"

namespace novaphy {

/// Configuration for the Sequential Impulse solver
struct SolverSettings {
    int velocity_iterations = 10;
    float baumgarte = 0.2f;     // position correction factor
    float slop = 0.005f;        // penetration slop (allowed before correction)
    bool warm_starting = true;
};

/// Sequential Impulse constraint solver for free rigid bodies.
/// Resolves contact constraints using Projected Gauss-Seidel with
/// accumulated impulse clamping.
class FreeBodySolver {
public:
    explicit FreeBodySolver(SolverSettings settings = {});

    /// Resolve all contacts by applying impulses to velocities.
    void solve(std::vector<ContactPoint>& contacts,
               const std::vector<RigidBody>& bodies,
               const std::vector<Transform>& transforms,
               std::vector<Vec3f>& linear_velocities,
               std::vector<Vec3f>& angular_velocities,
               float dt);

    SolverSettings& settings() { return settings_; }

private:
    SolverSettings settings_;

    // Pre-computed per-constraint data
    struct ConstraintData {
        Vec3f r_a, r_b;          // contact point relative to body COM
        float effective_mass_n;   // effective mass along normal
        float effective_mass_t1;  // effective mass along tangent1
        float effective_mass_t2;  // effective mass along tangent2
        Vec3f tangent1, tangent2; // friction directions
        float bias;               // position correction bias
    };

    std::vector<ConstraintData> constraint_data_;

    void pre_step(std::vector<ContactPoint>& contacts,
                  const std::vector<RigidBody>& bodies,
                  const std::vector<Transform>& transforms,
                  const std::vector<Vec3f>& linear_velocities,
                  const std::vector<Vec3f>& angular_velocities,
                  float dt);

    void solve_velocity(std::vector<ContactPoint>& contacts,
                        const std::vector<RigidBody>& bodies,
                        std::vector<Vec3f>& linear_velocities,
                        std::vector<Vec3f>& angular_velocities);
};

}  // namespace novaphy
