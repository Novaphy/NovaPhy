#pragma once

#include "novaphy/fluid/particle_state.h"
#include "novaphy/fluid/pbf_solver.h"
#include "novaphy/sim/world.h"

namespace novaphy {

/**
 * @brief Extended simulation world with fluid (PBF) support.
 *
 * @details Inherits rigid-body simulation from World and adds
 * a PBF fluid solver operating on ParticleState. The step()
 * method runs both rigid-body and fluid solvers.
 */
class FluidWorld : public World {
public:
    /**
     * @brief Construct a fluid world from a model and fluid definitions.
     *
     * @param[in] model Rigid-body model.
     * @param[in] fluid_blocks Fluid block definitions for particle emission.
     * @param[in] solver_settings Rigid-body contact solver settings.
     * @param[in] pbf_settings PBF fluid solver settings.
     */
    FluidWorld(const Model& model,
               const std::vector<FluidBlockDef>& fluid_blocks = {},
               SolverSettings solver_settings = {},
               PBFSettings pbf_settings = {});

    /**
     * @brief Advance both rigid-body and fluid simulation by one time step.
     *
     * @param[in] dt Time step (s).
     */
    void step(float dt);

    /**
     * @brief Read-only access to fluid particle state.
     *
     * @return Const reference to ParticleState.
     */
    const ParticleState& fluid_state() const { return fluid_state_; }

    /**
     * @brief Mutable access to fluid particle state.
     *
     * @return Reference to ParticleState.
     */
    ParticleState& fluid_state() { return fluid_state_; }

    /**
     * @brief Mutable access to PBF solver settings.
     *
     * @return Reference to PBFSettings.
     */
    PBFSettings& pbf_settings() { return pbf_solver_.settings(); }

    /**
     * @brief Read-only access to PBF solver settings.
     *
     * @return Const reference to PBFSettings.
     */
    const PBFSettings& pbf_settings() const { return pbf_solver_.settings(); }

    /**
     * @brief Get number of fluid particles.
     *
     * @return Particle count.
     */
    int num_particles() const { return fluid_state_.num_particles(); }

private:
    ParticleState fluid_state_;
    PBFSolver pbf_solver_;
    float particle_mass_ = 0.0f;
    float particle_spacing_ = 0.02f;
};

}  // namespace novaphy
