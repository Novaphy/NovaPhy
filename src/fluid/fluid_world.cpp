/**
 * @file fluid_world.cpp
 * @brief Extended simulation world with PBF fluid support.
 */
#include "novaphy/fluid/fluid_world.h"

namespace novaphy {

FluidWorld::FluidWorld(const Model& model,
                       const std::vector<FluidBlockDef>& fluid_blocks,
                       SolverSettings solver_settings,
                       PBFSettings pbf_settings)
    : World(model, solver_settings), pbf_solver_(pbf_settings) {
    // Generate particles from all fluid blocks
    std::vector<Vec3f> all_positions;
    Vec3f initial_vel = Vec3f::Zero();

    for (const auto& block : fluid_blocks) {
        auto block_positions = generate_fluid_block(block);
        all_positions.insert(all_positions.end(),
                             block_positions.begin(), block_positions.end());
        // Use last block's velocity (simplified; could per-block later)
        initial_vel = block.initial_velocity;
        particle_spacing_ = block.particle_spacing;
    }

    if (!all_positions.empty()) {
        fluid_state_.init(all_positions, initial_vel);
        particle_mass_ = pbf_settings.particle_mass(particle_spacing_);
    }
}

void FluidWorld::step(float dt) {
    // Step rigid bodies
    World::step(dt);

    // Step fluid particles
    if (fluid_state_.num_particles() > 0) {
        pbf_solver_.step(fluid_state_, dt, gravity(), particle_mass_);
    }
}

}  // namespace novaphy
