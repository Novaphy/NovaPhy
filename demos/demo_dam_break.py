"""Dam-break fluid simulation demo using PBF.

A rectangular block of fluid collapses under gravity inside a box-shaped
domain. Demonstrates the PBF solver with Polyscope particle visualization.
"""

import sys
import numpy as np

try:
    import polyscope as ps
    HAS_POLYSCOPE = True
except ImportError:
    HAS_POLYSCOPE = False

import novaphy


def build_scene():
    """Build a dam-break fluid scene."""
    # Fluid block: left side of a tank
    block = novaphy.FluidBlockDef()
    block.lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    block.upper = np.array([0.4, 0.6, 0.4], dtype=np.float32)
    block.particle_spacing = 0.025
    block.rest_density = 1000.0

    # PBF settings tuned for this scene
    pbf = novaphy.PBFSettings()
    pbf.rest_density = 1000.0
    pbf.kernel_radius = block.particle_spacing * 4.0
    pbf.solver_iterations = 4
    pbf.xsph_viscosity = 0.01
    pbf.epsilon = 100.0
    pbf.particle_radius = block.particle_spacing * 0.5

    # Empty rigid-body model (no rigid bodies in this demo)
    builder = novaphy.ModelBuilder()
    model = builder.build()

    world = novaphy.FluidWorld(model, [block], novaphy.SolverSettings(), pbf)
    print(f"Created dam-break with {world.num_particles} particles")
    return world, block.particle_spacing


def run_headless(world, n_steps=300, dt=1.0/120.0):
    """Run simulation without visualization."""
    print(f"Running {n_steps} steps headless...")
    for i in range(n_steps):
        world.step(dt)
        if i % 50 == 0:
            positions = world.fluid_state.positions
            ys = [p[1] for p in positions]
            print(f"  step {i}: mean_y={np.mean(ys):.3f}, "
                  f"min_y={np.min(ys):.3f}, max_y={np.max(ys):.3f}")
    print("Done.")


def run_polyscope(world, spacing, dt=1.0/120.0):
    """Run with Polyscope visualization."""
    ps.init()
    ps.set_program_name("NovaPhy Dam Break")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    # Register particles
    positions = np.array([np.array(p) for p in world.fluid_state.positions])
    cloud = ps.register_point_cloud("fluid", positions,
                                     radius=spacing * 0.5,
                                     color=(0.2, 0.5, 0.9))

    def callback():
        world.step(dt)
        positions = np.array([np.array(p) for p in world.fluid_state.positions])
        cloud.update_point_positions(positions)

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    world, spacing = build_scene()

    if "--headless" in sys.argv or not HAS_POLYSCOPE:
        run_headless(world)
    else:
        run_polyscope(world, spacing)
