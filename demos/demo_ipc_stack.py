"""Demo: Box stacking with IPC (Incremental Potential Contact).

Demonstrates libuipc-backed simulation with guaranteed
penetration-free contact. Requires building with -DNOVAPHY_WITH_IPC=ON.

Usage:
    python demos/demo_ipc_stack.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import novaphy
from novaphy.viz import (
    make_box_mesh, make_ground_plane_mesh,
    transform_vertices, quat_to_rotation_matrix,
)

if not novaphy.has_ipc():
    print("IPC support not available. Build with -DNOVAPHY_WITH_IPC=ON")
    sys.exit(1)

try:
    import polyscope as ps
    import polyscope.imgui as psim
    HAS_POLYSCOPE = True
except ImportError:
    HAS_POLYSCOPE = False


def build_scene():
    builder = novaphy.ModelBuilder()

    # Ground plane
    builder.add_ground_plane(y=0.0, friction=0.5, restitution=0.0)

    # Stack of 5 boxes
    half = np.array([0.4, 0.4, 0.4], dtype=np.float32)
    num_boxes = 5

    for i in range(num_boxes):
        body = novaphy.RigidBody.from_box(1.0, half)
        y = 1.0 + i * 1.0
        t = novaphy.Transform.from_translation(
            np.array([0.0, y, 0.0], dtype=np.float32))
        body_idx = builder.add_body(body, t)
        shape = novaphy.CollisionShape.make_box(
            half, body_idx, novaphy.Transform.identity(), 0.5, 0.0)
        builder.add_shape(shape)

    model = builder.build()

    config = novaphy.IPCConfig()
    config.dt = 0.01
    config.friction = 0.5
    config.kappa = 1e8
    config.body_kappa = 1e8

    world = novaphy.IPCWorld(model, config)
    return world, model, half, num_boxes


def run_headless(world, model, num_steps=300):
    """Fallback: run without GUI."""
    print(f"IPC Stack Demo: {model.num_bodies} boxes (headless)")
    for step in range(num_steps):
        world.step()
        if step % 50 == 0 or step == num_steps - 1:
            state = world.state()
            positions = [state.transforms[j].position[1]
                         for j in range(model.num_bodies)]
            print(f"Step {step:4d}: y = {['%.3f' % p for p in positions]}")
    print("Done.")


def run_gui(world, model, half, num_boxes):
    """Run with Polyscope GUI."""
    ps.init()
    ps.set_program_name("NovaPhy - IPC Stack Demo")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    # Register ground plane mesh
    gnd_v, gnd_f = make_ground_plane_mesh(size=10.0, y=0.0)
    gnd_mesh = ps.register_surface_mesh("ground", gnd_v, gnd_f)
    gnd_mesh.set_color((0.6, 0.6, 0.6))
    gnd_mesh.set_transparency(0.5)

    # Register box meshes at initial positions
    box_local_v, box_f = make_box_mesh(half)
    colors = [
        (0.90, 0.30, 0.25),  # red
        (0.25, 0.75, 0.35),  # green
        (0.30, 0.50, 0.90),  # blue
        (0.95, 0.70, 0.20),  # orange
        (0.70, 0.30, 0.85),  # purple
    ]

    state = world.state()
    for i in range(num_boxes):
        world_v = transform_vertices(box_local_v, state.transforms[i])
        m = ps.register_surface_mesh(f"box_{i}", world_v, box_f)
        m.set_color(colors[i % len(colors)])
        m.set_smooth_shade(True)

    paused = [False]
    frame = [0]

    def callback():
        nonlocal paused, frame

        psim.TextUnformatted(f"Frame: {frame[0]}")
        psim.TextUnformatted(f"Backend: CUDA (libuipc)")

        _, paused[0] = psim.Checkbox("Paused", paused[0])

        if psim.Button("Step Once"):
            world.step()
            frame[0] += 1

        if psim.Button("Reset"):
            # Can't easily reset libuipc, just inform user
            psim.TextUnformatted("(Reset not supported for IPC)")

        psim.Separator()

        if not paused[0]:
            world.step()
            frame[0] += 1

        # Update mesh positions
        st = world.state()
        for i in range(num_boxes):
            world_v = transform_vertices(box_local_v, st.transforms[i])
            ps.get_surface_mesh(f"box_{i}").update_vertex_positions(world_v)

        # Show positions
        for i in range(num_boxes):
            pos = st.transforms[i].position
            psim.TextUnformatted(
                f"box {i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    ps.set_user_callback(callback)
    ps.show()


def main():
    world, model, half, num_boxes = build_scene()

    if HAS_POLYSCOPE:
        run_gui(world, model, half, num_boxes)
    else:
        print("Polyscope not installed. Running headless...")
        run_headless(world, model)


if __name__ == "__main__":
    main()
