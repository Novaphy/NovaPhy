#include "novaphy/dynamics/free_body_solver.h"

#include <algorithm>
#include <cmath>

namespace novaphy {

FreeBodySolver::FreeBodySolver(SolverSettings settings) : settings_(settings) {}

void FreeBodySolver::solve(std::vector<ContactPoint>& contacts,
                            const std::vector<RigidBody>& bodies,
                            const std::vector<Transform>& transforms,
                            std::vector<Vec3f>& linear_velocities,
                            std::vector<Vec3f>& angular_velocities,
                            float dt) {
    if (contacts.empty()) return;

    pre_step(contacts, bodies, transforms, linear_velocities, angular_velocities, dt);

    // Warm start: apply accumulated impulses from previous frame
    if (settings_.warm_starting) {
        for (size_t i = 0; i < contacts.size(); ++i) {
            auto& cp = contacts[i];
            auto& cd = constraint_data_[i];

            Vec3f impulse = cp.normal * cp.accumulated_normal_impulse +
                            cd.tangent1 * cp.accumulated_tangent_impulse_1 +
                            cd.tangent2 * cp.accumulated_tangent_impulse_2;

            if (cp.body_a >= 0 && !bodies[cp.body_a].is_static()) {
                linear_velocities[cp.body_a] -= impulse * bodies[cp.body_a].inv_mass();
                angular_velocities[cp.body_a] -= bodies[cp.body_a].inv_inertia() *
                    cd.r_a.cross(impulse);
            }
            if (cp.body_b >= 0 && !bodies[cp.body_b].is_static()) {
                linear_velocities[cp.body_b] += impulse * bodies[cp.body_b].inv_mass();
                angular_velocities[cp.body_b] += bodies[cp.body_b].inv_inertia() *
                    cd.r_b.cross(impulse);
            }
        }
    }

    // Iterative velocity solving
    for (int iter = 0; iter < settings_.velocity_iterations; ++iter) {
        solve_velocity(contacts, bodies, linear_velocities, angular_velocities);
    }
}

void FreeBodySolver::pre_step(std::vector<ContactPoint>& contacts,
                               const std::vector<RigidBody>& bodies,
                               const std::vector<Transform>& transforms,
                               const std::vector<Vec3f>& linear_velocities,
                               const std::vector<Vec3f>& angular_velocities,
                               float dt) {
    constraint_data_.resize(contacts.size());

    for (size_t i = 0; i < contacts.size(); ++i) {
        auto& cp = contacts[i];
        auto& cd = constraint_data_[i];

        // Compute contact point relative to body COMs
        if (cp.body_a >= 0) {
            cd.r_a = cp.position - transforms[cp.body_a].position;
        } else {
            cd.r_a = Vec3f::Zero();
        }
        if (cp.body_b >= 0) {
            cd.r_b = cp.position - transforms[cp.body_b].position;
        } else {
            cd.r_b = Vec3f::Zero();
        }

        // Compute effective mass along normal
        float inv_mass_a = (cp.body_a >= 0) ? bodies[cp.body_a].inv_mass() : 0.0f;
        float inv_mass_b = (cp.body_b >= 0) ? bodies[cp.body_b].inv_mass() : 0.0f;
        Mat3f inv_I_a = (cp.body_a >= 0) ? bodies[cp.body_a].inv_inertia() : Mat3f::Zero();
        Mat3f inv_I_b = (cp.body_b >= 0) ? bodies[cp.body_b].inv_inertia() : Mat3f::Zero();

        Vec3f rn_a = cd.r_a.cross(cp.normal);
        Vec3f rn_b = cd.r_b.cross(cp.normal);

        cd.effective_mass_n = inv_mass_a + inv_mass_b +
            rn_a.dot(inv_I_a * rn_a) + rn_b.dot(inv_I_b * rn_b);
        if (cd.effective_mass_n > 0) cd.effective_mass_n = 1.0f / cd.effective_mass_n;

        // Compute tangent basis for friction
        if (std::abs(cp.normal.x()) < 0.9f) {
            cd.tangent1 = cp.normal.cross(Vec3f(1, 0, 0)).normalized();
        } else {
            cd.tangent1 = cp.normal.cross(Vec3f(0, 1, 0)).normalized();
        }
        cd.tangent2 = cp.normal.cross(cd.tangent1);

        // Effective mass along tangents
        Vec3f rt1_a = cd.r_a.cross(cd.tangent1);
        Vec3f rt1_b = cd.r_b.cross(cd.tangent1);
        cd.effective_mass_t1 = inv_mass_a + inv_mass_b +
            rt1_a.dot(inv_I_a * rt1_a) + rt1_b.dot(inv_I_b * rt1_b);
        if (cd.effective_mass_t1 > 0) cd.effective_mass_t1 = 1.0f / cd.effective_mass_t1;

        Vec3f rt2_a = cd.r_a.cross(cd.tangent2);
        Vec3f rt2_b = cd.r_b.cross(cd.tangent2);
        cd.effective_mass_t2 = inv_mass_a + inv_mass_b +
            rt2_a.dot(inv_I_a * rt2_a) + rt2_b.dot(inv_I_b * rt2_b);
        if (cd.effective_mass_t2 > 0) cd.effective_mass_t2 = 1.0f / cd.effective_mass_t2;

        // Baumgarte position correction bias
        float pen = std::max(0.0f, cp.penetration - settings_.slop);
        cd.bias = settings_.baumgarte * pen / dt;

        // Restitution bias
        Vec3f vel_a = Vec3f::Zero(), vel_b = Vec3f::Zero();
        if (cp.body_a >= 0) {
            vel_a = linear_velocities[cp.body_a] +
                    angular_velocities[cp.body_a].cross(cd.r_a);
        }
        if (cp.body_b >= 0) {
            vel_b = linear_velocities[cp.body_b] +
                    angular_velocities[cp.body_b].cross(cd.r_b);
        }
        float rel_vel_n = cp.normal.dot(vel_b - vel_a);
        if (rel_vel_n < -1.0f) {
            cd.bias += cp.restitution * rel_vel_n;
        }

        // Reset accumulated impulses if not warm starting
        if (!settings_.warm_starting) {
            cp.accumulated_normal_impulse = 0.0f;
            cp.accumulated_tangent_impulse_1 = 0.0f;
            cp.accumulated_tangent_impulse_2 = 0.0f;
        }
    }
}

void FreeBodySolver::solve_velocity(std::vector<ContactPoint>& contacts,
                                     const std::vector<RigidBody>& bodies,
                                     std::vector<Vec3f>& linear_velocities,
                                     std::vector<Vec3f>& angular_velocities) {
    for (size_t i = 0; i < contacts.size(); ++i) {
        auto& cp = contacts[i];
        auto& cd = constraint_data_[i];

        float inv_mass_a = (cp.body_a >= 0) ? bodies[cp.body_a].inv_mass() : 0.0f;
        float inv_mass_b = (cp.body_b >= 0) ? bodies[cp.body_b].inv_mass() : 0.0f;
        Mat3f inv_I_a = (cp.body_a >= 0) ? bodies[cp.body_a].inv_inertia() : Mat3f::Zero();
        Mat3f inv_I_b = (cp.body_b >= 0) ? bodies[cp.body_b].inv_inertia() : Mat3f::Zero();

        // Relative velocity at contact point
        Vec3f vel_a = Vec3f::Zero(), vel_b = Vec3f::Zero();
        if (cp.body_a >= 0) {
            vel_a = linear_velocities[cp.body_a] +
                    angular_velocities[cp.body_a].cross(cd.r_a);
        }
        if (cp.body_b >= 0) {
            vel_b = linear_velocities[cp.body_b] +
                    angular_velocities[cp.body_b].cross(cd.r_b);
        }
        Vec3f rel_vel = vel_b - vel_a;

        // --- Normal impulse ---
        float vn = rel_vel.dot(cp.normal);
        float lambda_n = cd.effective_mass_n * (-vn + cd.bias);

        // Accumulated clamping: total impulse must be non-negative
        float old_impulse = cp.accumulated_normal_impulse;
        cp.accumulated_normal_impulse = std::max(0.0f, old_impulse + lambda_n);
        lambda_n = cp.accumulated_normal_impulse - old_impulse;

        Vec3f impulse_n = cp.normal * lambda_n;

        if (cp.body_a >= 0 && !bodies[cp.body_a].is_static()) {
            linear_velocities[cp.body_a] -= impulse_n * inv_mass_a;
            angular_velocities[cp.body_a] -= inv_I_a * cd.r_a.cross(impulse_n);
        }
        if (cp.body_b >= 0 && !bodies[cp.body_b].is_static()) {
            linear_velocities[cp.body_b] += impulse_n * inv_mass_b;
            angular_velocities[cp.body_b] += inv_I_b * cd.r_b.cross(impulse_n);
        }

        // --- Friction impulse (tangent 1) ---
        // Recompute relative velocity after normal impulse
        vel_a = Vec3f::Zero();
        vel_b = Vec3f::Zero();
        if (cp.body_a >= 0) {
            vel_a = linear_velocities[cp.body_a] +
                    angular_velocities[cp.body_a].cross(cd.r_a);
        }
        if (cp.body_b >= 0) {
            vel_b = linear_velocities[cp.body_b] +
                    angular_velocities[cp.body_b].cross(cd.r_b);
        }
        rel_vel = vel_b - vel_a;

        float vt1 = rel_vel.dot(cd.tangent1);
        float lambda_t1 = cd.effective_mass_t1 * (-vt1);

        float max_friction = cp.friction * cp.accumulated_normal_impulse;
        float old_t1 = cp.accumulated_tangent_impulse_1;
        cp.accumulated_tangent_impulse_1 = std::max(-max_friction,
            std::min(max_friction, old_t1 + lambda_t1));
        lambda_t1 = cp.accumulated_tangent_impulse_1 - old_t1;

        Vec3f impulse_t1 = cd.tangent1 * lambda_t1;

        if (cp.body_a >= 0 && !bodies[cp.body_a].is_static()) {
            linear_velocities[cp.body_a] -= impulse_t1 * inv_mass_a;
            angular_velocities[cp.body_a] -= inv_I_a * cd.r_a.cross(impulse_t1);
        }
        if (cp.body_b >= 0 && !bodies[cp.body_b].is_static()) {
            linear_velocities[cp.body_b] += impulse_t1 * inv_mass_b;
            angular_velocities[cp.body_b] += inv_I_b * cd.r_b.cross(impulse_t1);
        }

        // --- Friction impulse (tangent 2) ---
        vel_a = Vec3f::Zero();
        vel_b = Vec3f::Zero();
        if (cp.body_a >= 0) {
            vel_a = linear_velocities[cp.body_a] +
                    angular_velocities[cp.body_a].cross(cd.r_a);
        }
        if (cp.body_b >= 0) {
            vel_b = linear_velocities[cp.body_b] +
                    angular_velocities[cp.body_b].cross(cd.r_b);
        }
        rel_vel = vel_b - vel_a;

        float vt2 = rel_vel.dot(cd.tangent2);
        float lambda_t2 = cd.effective_mass_t2 * (-vt2);

        float old_t2 = cp.accumulated_tangent_impulse_2;
        cp.accumulated_tangent_impulse_2 = std::max(-max_friction,
            std::min(max_friction, old_t2 + lambda_t2));
        lambda_t2 = cp.accumulated_tangent_impulse_2 - old_t2;

        Vec3f impulse_t2 = cd.tangent2 * lambda_t2;

        if (cp.body_a >= 0 && !bodies[cp.body_a].is_static()) {
            linear_velocities[cp.body_a] -= impulse_t2 * inv_mass_a;
            angular_velocities[cp.body_a] -= inv_I_a * cd.r_a.cross(impulse_t2);
        }
        if (cp.body_b >= 0 && !bodies[cp.body_b].is_static()) {
            linear_velocities[cp.body_b] += impulse_t2 * inv_mass_b;
            angular_velocities[cp.body_b] += inv_I_b * cd.r_b.cross(impulse_t2);
        }
    }
}

}  // namespace novaphy
