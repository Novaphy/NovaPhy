"""Shared demo utilities for NovaPhy demos."""

import sys
import numpy as np

try:
    import polyscope as ps
    HAS_POLYSCOPE = True
except ImportError:
    HAS_POLYSCOPE = False

import novaphy
from novaphy.viz import SceneVisualizer


class DemoApp:
    """Base class for NovaPhy demos with Polyscope visualization."""

    def __init__(self, title="NovaPhy Demo", dt=1.0/120.0, ground_size=20.0):
        self.title = title
        self.dt = dt
        self.ground_size = ground_size
        self.world = None
        self.viz = None

    def build_scene(self):
        """Override to build the scene. Must set self.world."""
        raise NotImplementedError

    def run(self, headless_steps=0):
        """Run the demo.

        Args:
            headless_steps: if > 0, run this many steps without visualization
                           (useful for testing)
        """
        self.build_scene()
        assert self.world is not None, "build_scene() must set self.world"

        if headless_steps > 0:
            for _ in range(headless_steps):
                self.world.step(self.dt)
            return

        if not HAS_POLYSCOPE:
            print("Polyscope not available. Running 500 steps headless...")
            for i in range(500):
                self.world.step(self.dt)
                if i % 100 == 0:
                    state = self.world.state
                    for j in range(self.world.model.num_bodies):
                        pos = state.transforms[j].position
                        print(f"  step {i}, body {j}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            print("Done.")
            return

        # Initialize Polyscope
        ps.init()
        ps.set_program_name(self.title)
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("shadow_only")

        # Create visualizer
        self.viz = SceneVisualizer(self.world, self.ground_size)

        # Simulation callback
        def callback():
            # Step physics
            self.world.step(self.dt)
            # Update meshes
            self.viz.update()

        ps.set_user_callback(callback)
        ps.show()
