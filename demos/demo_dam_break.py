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
    pbf.solver_iterations = 6
    pbf.xsph_viscosity = 0.1
    pbf.epsilon = 100.0
    pbf.particle_radius = block.particle_spacing * 0.5

    # Domain bounds (container walls)
    pbf.use_domain_bounds = True
    pbf.domain_lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pbf.domain_upper = np.array([1.0, 1.0, 0.4], dtype=np.float32)

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


def make_box_lines(lo, hi):
    """Create 12 edges of an axis-aligned box."""
    v = np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ], dtype=np.float32)
    edges = np.array([
        [0,1],[1,2],[2,3],[3,0],  # front face
        [4,5],[5,6],[6,7],[7,4],  # back face
        [0,4],[1,5],[2,6],[3,7],  # connecting edges
    ], dtype=np.int32)
    return v, edges


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

    # Register wireframe box for domain bounds
    lo = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    hi = np.array([1.0, 1.0, 0.4], dtype=np.float32)
    box_v, box_e = make_box_lines(lo, hi)
    box_net = ps.register_curve_network("box", box_v, box_e)
    box_net.set_color((0.8, 0.8, 0.8))
    box_net.set_radius(0.003, relative=False)

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
