#pragma once

#include "novaphy/core/model.h"
#include "novaphy/sim/state.h"
#include "novaphy/vbd/vbd_config.h"

#include "novaphy/collision/broadphase.h"
#include "novaphy/core/contact.h"
#include "novaphy/math/math_types.h"
#include "novaphy/math/spatial.h"
#include "novaphy/vbd/vbd_collide.h"

#include <vector>

namespace novaphy {

/**
 * @brief Single-point contact constraint (aligned with avbd-demo3d Manifold::Contact).
 *
 * Uses a 3D constraint: normal + two tangents.
 * basis row0 = normal (B→A), row1/2 = tangents.
 * C0 = basis * (xA - xB) + {COLLISION_MARGIN, 0, 0}, and F = K*C + lambda.
 */
struct AvbdContact {
    int body_a = -1;
    int body_b = -1;
    Vec3f rA = Vec3f::Zero();  // Contact point in A local coordinates.
    Vec3f rB = Vec3f::Zero();  // Contact point in B local coordinates.
    Mat3f basis = Mat3f::Identity();  // row0=normal (B→A), row1/2=tangents.

    Vec3f C0 = Vec3f::Zero();   // Constraint value at step start.
    Vec3f penalty = Vec3f::Zero();
    Vec3f lambda = Vec3f::Zero();
    float friction = 0.5f;
    bool stick = false;

    // Optional: feature id forwarded from narrowphase for contact persistence (demo3d FeaturePair::key).
    int feature_id = -1;
};

/**
 * @brief 3D AVBD solver, following avbd-demo3d's step flow and equations.
 *
 * Flow: broadphase → build contacts & initialize (C0, warmstart) → body initialize (inertial, initial) →
 * main loop (primal per-body 6x6 + dual update) → BDF1 velocities.
 */
class VbdSolver {
public:
    explicit VbdSolver(const VBDConfig& cfg);

    void set_config(const VBDConfig& cfg);
    const VBDConfig& config() const { return config_; }

    void set_model(const Model& model);

    /**
     * @brief One AVBD step (matches demo3d Solver::step()).
     */
    void step(const Model& model, SimState& state);

private:
    /** Build contacts at step start and initialize C0 + warmstart lambda/penalty. */
    void build_contact_constraints(const Model& model, const SimState& state);

    /** Main loop primal: assemble per-body 6x6 LHS/RHS, solve and apply dq. */
    void avbd_primal(const Model& model, SimState& state);
    /** Main loop dual: update lambda and penalty. */
    void avbd_dual(const Model& model, const SimState& state);

    VBDConfig config_;
    SweepAndPrune broadphase_;
    std::vector<AvbdContact> avbd_contacts_;
    std::vector<Vec3f> inertial_positions_;
    std::vector<Quatf> inertial_rotations_;
    std::vector<Vec3f> initial_positions_;
    std::vector<Quatf> initial_rotations_;
    std::vector<Vec3f> prev_linear_velocities_;
};

}  // namespace novaphy
