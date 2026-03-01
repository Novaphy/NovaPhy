#pragma once

#include "novaphy/math/math_types.h"

namespace novaphy {

/**
 * @brief Configuration for IPCWorld simulation.
 *
 * Controls libuipc backend parameters including timestep, contact tolerances,
 * friction, and Newton solver settings.
 */
struct IPCConfig {
    float dt = 0.01f;                       /**< Timestep in seconds. */
    Vec3f gravity = Vec3f(0.0f, -9.81f, 0.0f); /**< Gravity acceleration (m/s^2). */

    // IPC contact parameters
    float d_hat = 0.01f;      /**< Contact distance threshold (m). */
    float kappa = 1e8f;       /**< Barrier stiffness (Pa). Default 100 MPa. */
    float friction = 0.5f;    /**< Default friction coefficient. */
    float contact_resistance = 1e9f;  /**< Contact resistance (Pa). Default 1 GPa. */

    // Affine body material
    float body_kappa = 1e8f;      /**< Affine body stiffness (Pa). */
    float mass_density = 1e3f;    /**< Default mass density (kg/m^3). */

    // Newton solver
    int newton_max_iter = 100;    /**< Max Newton iterations per step. */
    float newton_tol = 1e-2f;     /**< Newton convergence tolerance. */

    // Backend
    const char* backend = "cuda"; /**< libuipc backend name ("cuda" or "none"). */
    const char* workspace = "./ipc_workspace"; /**< Output workspace directory. */
};

}  // namespace novaphy
