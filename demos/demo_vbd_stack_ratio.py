"""Demo: stack with doubling box sizes (avbd-demo3d sceneStackRatio).

Layout matches scenes.h sceneStackRatio(): ground box then 4 boxes (sizes 1, 2, 4, 8) already stacked at rest.

Usage:
    python demos/demo_vbd_stack_ratio.py
    python demos/demo_vbd_stack_ratio.py --headless
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import novaphy

try:
    import polyscope as ps
    HAS_POLYSCOPE = True
except ImportError:
    HAS_POLYSCOPE = False


def build_vbd_world():
    """Build scene identical to avbd-demo3d sceneStackRatio()."""
    builder = novaphy.ModelBuilder()

    # Ground: demo3d {100, 100, groundThickness=1} at {0,0,0} → Y-up center (0,0,0), half (50, 0.5, 50), top y=0.5
    ground_thickness = 1.0
    half_ground = np.array([50.0, ground_thickness * 0.5, 50.0], dtype=np.float32)
    ground_body = novaphy.RigidBody.make_static()
    ground_t = novaphy.Transform.from_translation(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    ground_idx = builder.add_body(ground_body, ground_t)
    ground_shape = novaphy.CollisionShape.make_box(
        half_ground, ground_idx, novaphy.Transform.identity(), 0.5, 0.0
    )
    builder.add_shape(ground_shape)

    # 4 boxes: sizes 1, 2, 4, 8, placed at rest like demo3d (topY = 0.5, centerY = topY+half, topY = centerY+half, s*=2)
    top_y = 0.5
    s = 1.0
    for i in range(4):
        half = s * 0.5
        center_y = top_y + half
        half_arr = np.array([half, half, half], dtype=np.float32)
        body = novaphy.RigidBody.from_box(1.0, half_arr)
        t = novaphy.Transform.from_translation(np.array([0.0, center_y, 0.0], dtype=np.float32))
        body_idx = builder.add_body(body, t)
        shape = novaphy.CollisionShape.make_box(
            half_arr, body_idx, novaphy.Transform.identity(), 0.5, 0.0
        )
        builder.add_shape(shape)
        top_y = center_y + half
        s *= 2.0

    model = builder.build()

    cfg = novaphy.VBDConfig()
    cfg.dt = 1.0 / 60.0
    cfg.iterations = 10
    cfg.gravity = np.array([0.0, -10.0, 0.0], dtype=np.float32)
    cfg.alpha = 0.995
    cfg.gamma = 0.999
    cfg.beta_linear = 10000.0
    cfg.beta_angular = 100.0
    cfg.primal_relaxation = 1.0
    cfg.lhs_regularization = 0.0

    world = novaphy.VBDWorld(model, cfg)
    return world


class _VBDWorldAdapter:
    def __init__(self, vbd_world):
        self._w = vbd_world

    @property
    def model(self):
        return self._w.model

    @property
    def state(self):
        return self._w.state

    def step(self, dt):
        del dt
        self._w.step()


class VBDStackRatioDemo:
    """VBD stack with size ratio 1:2:4:8 (avbd-demo3d sceneStackRatio)."""
    def __init__(self):
        self.title = "NovaPhy - VBD Stack Ratio (demo3d sceneStackRatio)"
        self.dt = 1.0 / 120.0
        self.ground_size = 20.0
        self.world = None
        self.viz = None

    def build_scene(self):
        self.world = _VBDWorldAdapter(build_vbd_world())

    def run(self, headless=False):
        self.build_scene()
        assert self.world is not None
        if headless:
            self._run_headless()
            return
        if not HAS_POLYSCOPE:
            print("Polyscope not available. Running headless...")
            self._run_headless()
            return
        ps.init()
        ps.set_program_name(self.title)
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("shadow_only")
        # Closer default camera for 4-box stack ratio.
        ps.look_at((8.0, 5.0, 8.0), (0.0, 4.0, 0.0))
        from novaphy.viz import SceneVisualizer
        self.viz = SceneVisualizer(self.world, self.ground_size)
        def callback():
            self.world.step(self.dt)
            self.viz.update()
        ps.set_user_callback(callback)
        ps.show()

    def _run_headless(self, steps=300):
        print("Running VBD stack ratio (demo3d sceneStackRatio) headless...")
        n = self.world.model.num_bodies
        for step in range(steps):
            self.world.step(self.dt)
            if step % 60 == 0 or step == steps - 1:
                state = self.world.state
                ys = [state.transforms[i].position[1] for i in range(n)]
                print(f"step {step:4d}: n={n} y = {[f'{y:.3f}' for y in ys]}")
        print("Done.")


if __name__ == "__main__":
    headless = "--headless" in sys.argv
    demo = VBDStackRatioDemo()
    demo.run(headless=headless)
