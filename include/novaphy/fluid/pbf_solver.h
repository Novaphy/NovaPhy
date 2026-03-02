#pragma once

#include <vector>

#include "novaphy/fluid/neighbor_search.h"
#include "novaphy/fluid/particle_state.h"

namespace novaphy {

/**
 * @brief Configuration for the PBF fluid solver.
 */
struct PBFSettings {
    float rest_density = 1000.0f;      /**< Rest density rho_0 (kg/m^3). */
    float kernel_radius = 0.1f;        /**< SPH kernel support radius h (m). */
    float particle_radius = 0.025f;    /**< Particle visual/collision radius (m). */
    int solver_iterations = 4;         /**< Number of PBF constraint iterations. */
    float epsilon = 100.0f;            /**< Relaxation parameter for constraint (CFM). */
    float xsph_viscosity = 0.01f;      /**< XSPH artificial viscosity coefficient. */
    float vorticity_epsilon = 0.0f;    /**< Vorticity confinement strength. */
    float corr_delta_q = 0.3f;         /**< Tensile instability correction: delta_q as fraction of h. */
    float corr_k = 0.001f;             /**< Tensile instability correction: k coefficient. */
    int corr_n = 4;                    /**< Tensile instability correction: exponent n. */

    bool use_domain_bounds = false;     /**< Whether to clamp particles to domain AABB. */
    Vec3f domain_lower = Vec3f(-10.0f, 0.0f, -10.0f);   /**< Domain lower bound (m). */
    Vec3f domain_upper = Vec3f(10.0f, 10.0f, 10.0f);     /**< Domain upper bound (m). */

    /**
     * @brief Compute particle mass from rest density and spacing.
     *
     * @param[in] spacing Inter-particle spacing (m).
     * @return Particle mass (kg).
     */
    float particle_mass(float spacing) const {
        return rest_density * spacing * spacing * spacing;
    }
};

/**
 * @brief Position Based Fluids solver (Macklin & Muller, SIGGRAPH 2013).
 *
 * @details Enforces incompressibility via iterative density constraint
 * projection in position space. Works on ParticleState arrays.
 */
class PBFSolver {
public:
    /**
     * @brief Construct PBF solver with given settings.
     *
     * @param[in] settings Solver configuration.
     */
    explicit PBFSolver(const PBFSettings& settings = {});

    /**
     * @brief Run one complete PBF step.
     *
     * @details Steps:
     *   1. Apply external forces to velocities
     *   2. Predict positions: x* = x + v*dt
     *   3. Build neighbor grid
     *   4. Iterative density constraint solving (solver_iterations times)
     *   5. Update velocities: v = (x_new - x_old) / dt
     *   6. Apply XSPH viscosity
     *   7. Update positions
     *
     * @param[in,out] state Particle state to update.
     * @param[in] dt Time step (s).
     * @param[in] gravity Gravity vector (m/s^2).
     * @param[in] particle_mass Mass per particle (kg).
     */
    void step(ParticleState& state, float dt, const Vec3f& gravity, float particle_mass);

    /**
     * @brief Mutable access to solver settings.
     *
     * @return Reference to PBFSettings.
     */
    PBFSettings& settings() { return settings_; }

    /**
     * @brief Read-only access to solver settings.
     *
     * @return Const reference to PBFSettings.
     */
    const PBFSettings& settings() const { return settings_; }

private:
    /**
     * @brief Compute density for all particles.
     */
    void compute_density(ParticleState& state, float particle_mass);

    /**
     * @brief Compute lambda (constraint multiplier) for all particles.
     */
    void compute_lambda(ParticleState& state, float particle_mass);

    /**
     * @brief Compute position corrections (delta_p) for all particles.
     */
    void compute_delta_position(ParticleState& state, float particle_mass);

    /**
     * @brief Apply XSPH viscosity smoothing to velocities.
     */
    void apply_xsph_viscosity(ParticleState& state, float particle_mass);

    PBFSettings settings_;
    SpatialHashGrid grid_;

    // Per-particle neighbor cache (rebuilt each step)
    std::vector<std::vector<int>> neighbors_;
};

}  // namespace novaphy
