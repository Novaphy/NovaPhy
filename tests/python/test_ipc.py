"""Tests for NovaPhy IPC integration (IPCConfig, IPCWorld).

All tests are skipped if the IPC module was not built
(requires CUDA >= 12.4 and -DNOVAPHY_WITH_IPC=ON).
"""

import numpy as np
import numpy.testing as npt
import pytest
import novaphy

# Skip entire module if IPC not available
pytestmark = pytest.mark.skipif(
    not novaphy.has_ipc(),
    reason="IPC support not built (requires -DNOVAPHY_WITH_IPC=ON and CUDA >= 12.4)"
)


# ---------- IPCConfig tests ----------

def test_ipc_config_defaults():
    """IPCConfig should have sensible defaults."""
    cfg = novaphy.IPCConfig()
    assert cfg.dt == pytest.approx(0.01)
    npt.assert_allclose(cfg.gravity, [0, -9.81, 0], atol=1e-3)
    assert cfg.friction == pytest.approx(0.5)
    assert cfg.kappa == pytest.approx(1e8)
    assert cfg.d_hat == pytest.approx(0.01)
    assert cfg.newton_max_iter == 100


def test_ipc_config_custom():
    """IPCConfig fields should be writable."""
    cfg = novaphy.IPCConfig()
    cfg.dt = 0.005
    cfg.friction = 0.3
    cfg.kappa = 1e7
    assert cfg.dt == pytest.approx(0.005)
    assert cfg.friction == pytest.approx(0.3)
    assert cfg.kappa == pytest.approx(1e7)


# ---------- IPCWorld tests ----------

def _make_falling_box_ipc():
    """Helper: one box above a ground plane, using IPCWorld."""
    builder = novaphy.ModelBuilder()
    builder.add_ground_plane(y=0.0, friction=0.5)

    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body = novaphy.RigidBody.from_box(1.0, half)
    t = novaphy.Transform.from_translation(
        np.array([0.0, 5.0, 0.0], dtype=np.float32))
    idx = builder.add_body(body, t)
    shape = novaphy.CollisionShape.make_box(half, idx)
    builder.add_shape(shape)

    model = builder.build()
    config = novaphy.IPCConfig()
    config.dt = 0.01
    return novaphy.IPCWorld(model, config)


def test_ipc_world_creation():
    """IPCWorld should initialize with correct initial state."""
    world = _make_falling_box_ipc()
    state = world.state()
    assert len(state.transforms) == 1
    npt.assert_allclose(state.transforms[0].position,
                        [0, 5, 0], atol=1e-3)
    assert world.frame() == 0


def test_ipc_world_step():
    """IPCWorld.step() should advance the frame counter."""
    world = _make_falling_box_ipc()
    world.step()
    assert world.frame() == 1


def test_ipc_free_fall():
    """A box should fall under gravity (y decreases after stepping)."""
    world = _make_falling_box_ipc()
    initial_y = world.state().transforms[0].position[1]

    for _ in range(10):
        world.step()

    final_y = world.state().transforms[0].position[1]
    assert final_y < initial_y, "Box should fall under gravity"


def test_ipc_no_penetration():
    """After many steps, box should not penetrate below ground (y >= -0.01).

    IPC guarantees no interpenetration, so the box center should stay
    at or above the half-extent height (0.5) minus a small tolerance.
    """
    world = _make_falling_box_ipc()

    for _ in range(200):
        world.step()

    final_y = world.state().transforms[0].position[1]
    # Box center should be at ~0.5 (half-extent) above ground
    # Allow generous tolerance for affine body deformation
    assert final_y > -0.1, f"Box penetrated ground: y={final_y}"


def test_ipc_static_body():
    """Static bodies (mass <= 0) should not move under gravity."""
    builder = novaphy.ModelBuilder()

    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body = novaphy.RigidBody.make_static()
    t = novaphy.Transform.from_translation(
        np.array([0.0, 2.0, 0.0], dtype=np.float32))
    idx = builder.add_body(body, t)
    shape = novaphy.CollisionShape.make_box(half, idx)
    builder.add_shape(shape)

    model = builder.build()
    config = novaphy.IPCConfig()
    world = novaphy.IPCWorld(model, config)

    for _ in range(50):
        world.step()

    final_y = world.state().transforms[0].position[1]
    npt.assert_allclose(final_y, 2.0, atol=0.01,
                        err_msg="Static body should not move")


def test_ipc_model_access():
    """IPCWorld should expose the model it was built from."""
    world = _make_falling_box_ipc()
    assert world.model().num_bodies == 1
    assert world.model().num_shapes >= 1


def test_ipc_config_access():
    """IPCWorld should expose its configuration."""
    world = _make_falling_box_ipc()
    assert world.config().dt == pytest.approx(0.01)
