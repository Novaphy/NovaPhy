"""Demo: Rigid-fluid coupling — boxes and spheres splashing into water.

Demonstrates:
- PBF fluid simulation with thousands of particles
- Akinci boundary particle coupling (fluid pushes rigid bodies)
- Rigid body collision (bodies land on ground plane)
- Full Polyscope visualization with mesh + particle rendering

Scene: A ground plane holds a pool of water. A box and a sphere drop
into the pool from above, creating splashes and interacting with the fluid.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import polyscope as ps
    HAS_POLYSCOPE = True
except ImportError:
    HAS_POLYSCOPE = False

import novaphy
from novaphy.viz import (
    make_box_mesh, make_sphere_mesh, make_ground_plane_mesh,
    transform_vertices, quat_to_rotation_matrix,
)


def build_scene():
    """Build the coupled rigid-fluid scene.

    Returns:
        tuple: (FluidWorld, particle_spacing, list of rigid mesh info)
    """
    builder = novaphy.ModelBuilder()

    # --- Ground plane ---
    builder.add_ground_plane(y=0.0, friction=0.5, restitution=0.0)

    # --- Rigid body 0: A box that drops into the pool ---
    box_half = np.array([0.08, 0.08, 0.08], dtype=np.float32)
    box_body = novaphy.RigidBody.from_box(0.3, box_half)
    box_t = novaphy.Transform.from_translation(
        np.array([0.2, 0.7, 0.2], dtype=np.float32))
    box_idx = builder.add_body(box_body, box_t)
    box_shape = novaphy.CollisionShape.make_box(box_half, box_idx)
    builder.add_shape(box_shape)

    # --- Rigid body 1: A sphere that drops in from the side ---
    sphere_radius = 0.06
    sphere_body = novaphy.RigidBody.from_sphere(0.2, sphere_radius)
    sphere_t = novaphy.Transform.from_translation(
        np.array([0.35, 0.9, 0.35], dtype=np.float32))
    sphere_idx = builder.add_body(sphere_body, sphere_t)
    sphere_shape = novaphy.CollisionShape.make_sphere(sphere_radius, sphere_idx)
    builder.add_shape(sphere_shape)

    model = builder.build()

    # --- Fluid pool ---
    spacing = 0.025
    block = novaphy.FluidBlockDef()
    block.lower = np.array([0.0, 0.01, 0.0], dtype=np.float32)
    block.upper = np.array([0.5, 0.25, 0.5], dtype=np.float32)
    block.particle_spacing = spacing
    block.rest_density = 1000.0

    # --- PBF solver settings ---
    pbf = novaphy.PBFSettings()
    pbf.rest_density = 1000.0
    pbf.kernel_radius = spacing * 4.0
    pbf.solver_iterations = 4
    pbf.xsph_viscosity = 0.01
    pbf.epsilon = 100.0
    pbf.particle_radius = spacing * 0.5

    # Domain bounds to contain the fluid (acts as invisible walls)
    pbf.use_domain_bounds = True
    pbf.domain_lower = np.array([-0.05, 0.0, -0.05], dtype=np.float32)
    pbf.domain_upper = np.array([0.55, 1.5, 0.55], dtype=np.float32)

    # --- Contact solver ---
    solver_settings = novaphy.SolverSettings()
    solver_settings.velocity_iterations = 20

    world = novaphy.FluidWorld(model, [block], solver_settings, pbf,
                                boundary_extent=0.8)

    # Mesh info for visualization: (name, local_verts, faces, body_index)
    rigid_meshes = []
    box_verts, box_faces = make_box_mesh(box_half)
    rigid_meshes.append(("box", box_verts, box_faces, box_idx))

    sphere_verts, sphere_faces = make_sphere_mesh(sphere_radius, n_lat=12, n_lon=24)
    rigid_meshes.append(("sphere", sphere_verts, sphere_faces, sphere_idx))

    print(f"Scene created:")
    print(f"  Fluid particles: {world.num_particles}")
    print(f"  Boundary particles: {world.num_boundary_particles}")
    print(f"  Rigid bodies: {model.num_bodies}")

    return world, spacing, rigid_meshes


def run_headless(world, n_steps=300, dt=1.0/120.0):
    """Run simulation without visualization."""
    print(f"Running {n_steps} steps headless (dt={dt:.4f}s)...")
    for i in range(n_steps):
        world.step(dt)
        if i % 100 == 0:
            box_pos = world.state.transforms[0].position
            sphere_pos = world.state.transforms[1].position
            fluid_ys = [p[1] for p in world.fluid_state.positions]
            print(f"  step {i:4d}: "
                  f"box=({box_pos[0]:.2f}, {box_pos[1]:.2f}, {box_pos[2]:.2f})  "
                  f"sphere=({sphere_pos[0]:.2f}, {sphere_pos[1]:.2f}, {sphere_pos[2]:.2f})  "
                  f"fluid_y=[{np.min(fluid_ys):.2f}, {np.mean(fluid_ys):.2f}, {np.max(fluid_ys):.2f}]")
    print("Done.")


def run_polyscope(world, spacing, rigid_meshes, dt=1.0/120.0):
    """Run with full Polyscope visualization."""
    ps.init()
    ps.set_program_name("NovaPhy - Rigid-Fluid Coupling")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    # Camera setup for a good viewing angle
    ps.look_at((1.2, 0.8, 1.2), (0.25, 0.15, 0.25))

    # --- Ground plane ---
    ground_v, ground_f = make_ground_plane_mesh(size=2.0, y=0.0)
    ground_mesh = ps.register_surface_mesh("ground", ground_v, ground_f)
    ground_mesh.set_color((0.75, 0.75, 0.72))
    ground_mesh.set_transparency(0.5)

    # --- Fluid point cloud ---
    fluid_pos = np.array([np.array(p) for p in world.fluid_state.positions],
                         dtype=np.float32)
    fluid_cloud = ps.register_point_cloud("fluid", fluid_pos)
    fluid_cloud.set_radius(spacing * 0.45, relative=False)
    fluid_cloud.set_color((0.15, 0.45, 0.85))

    # --- Rigid body meshes ---
    ps_meshes = []
    body_colors = [
        (0.85, 0.35, 0.15),  # orange-red for box
        (0.2, 0.75, 0.3),    # green for sphere
    ]
    for i, (name, local_verts, faces, body_idx) in enumerate(rigid_meshes):
        t = world.state.transforms[body_idx]
        world_verts = transform_vertices(local_verts, t)
        mesh = ps.register_surface_mesh(name, world_verts, faces)
        mesh.set_color(body_colors[i % len(body_colors)])
        mesh.set_smooth_shade(True)
        ps_meshes.append((mesh, local_verts, faces, body_idx))

    # --- Simulation callback ---
    step_count = [0]

    def callback():
        # Step physics
        world.step(dt)
        step_count[0] += 1

        # Update fluid particles
        fluid_pos = np.array([np.array(p) for p in world.fluid_state.positions],
                             dtype=np.float32)
        fluid_cloud.update_point_positions(fluid_pos)

        # Color fluid by speed for visual appeal
        if step_count[0] % 3 == 0:  # update color every few frames for performance
            vels = np.array([np.array(v) for v in world.fluid_state.velocities],
                            dtype=np.float32)
            speeds = np.linalg.norm(vels, axis=1)
            max_speed = max(speeds.max(), 0.01)
            # Blue (slow) -> Cyan (medium) -> White (fast)
            t_val = np.clip(speeds / max_speed, 0.0, 1.0)
            colors = np.column_stack([
                0.15 + 0.85 * t_val,      # R: 0.15 -> 1.0
                0.45 + 0.55 * t_val,      # G: 0.45 -> 1.0
                0.85 + 0.15 * t_val,      # B: 0.85 -> 1.0
            ])
            fluid_cloud.add_color_quantity("speed", colors, enabled=True)

        # Update rigid body meshes
        for mesh, local_verts, faces, body_idx in ps_meshes:
            t = world.state.transforms[body_idx]
            world_verts = transform_vertices(local_verts, t)
            mesh.update_vertex_positions(world_verts)

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    world, spacing, rigid_meshes = build_scene()

    if "--headless" in sys.argv or not HAS_POLYSCOPE:
        if not HAS_POLYSCOPE:
            print("Polyscope not available, running headless.")
        run_headless(world)
    else:
        run_polyscope(world, spacing, rigid_meshes)
