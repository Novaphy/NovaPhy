"""Tests for NovaPhy free body simulation (World, ModelBuilder, solver)."""

import numpy as np
import numpy.testing as npt
import novaphy


def _make_single_falling_box():
    """Helper: one box above ground plane."""
    builder = novaphy.ModelBuilder()
    builder.add_ground_plane(y=0.0)

    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body = novaphy.RigidBody.from_box(1.0, half)
    t = novaphy.Transform.from_translation(
        np.array([0.0, 5.0, 0.0], dtype=np.float32))
    idx = builder.add_body(body, t)
    shape = novaphy.CollisionShape.make_box(half, idx)
    builder.add_shape(shape)

    model = builder.build()
    return novaphy.World(model)


def test_model_builder_counts():
    builder = novaphy.ModelBuilder()
    assert builder.num_bodies == 0
    assert builder.num_shapes == 0

    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body = novaphy.RigidBody.from_box(1.0, half)
    idx = builder.add_body(body)
    assert idx == 0
    assert builder.num_bodies == 1

    builder.add_ground_plane()
    assert builder.num_shapes == 1

    model = builder.build()
    assert model.num_bodies == 1
    assert model.num_shapes == 1


def test_world_creation():
    world = _make_single_falling_box()
    state = world.state
    assert len(state.transforms) == 1
    npt.assert_allclose(state.transforms[0].position,
                        [0, 5, 0], atol=1e-5)


def test_gravity_default():
    world = _make_single_falling_box()
    npt.assert_allclose(world.gravity, [0, -9.81, 0], atol=1e-5)


def test_box_falls_under_gravity():
    world = _make_single_falling_box()
    initial_y = world.state.transforms[0].position[1]

    # Step a few times
    for _ in range(10):
        world.step(1.0 / 120.0)

    new_y = world.state.transforms[0].position[1]
    assert new_y < initial_y, "Box should fall under gravity"


def test_box_rests_on_ground():
    world = _make_single_falling_box()

    # Run for ~2 seconds of sim time
    for _ in range(240):
        world.step(1.0 / 120.0)

    y = world.state.transforms[0].position[1]
    # Box half-extent is 0.5, so center should rest at ~0.5
    assert y > 0.3, f"Box center should be above ground, got y={y}"
    assert y < 1.0, f"Box should have settled, got y={y}"


def test_two_boxes_stack():
    builder = novaphy.ModelBuilder()
    builder.add_ground_plane(y=0.0)

    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    for i in range(2):
        body = novaphy.RigidBody.from_box(1.0, half)
        y = 1.0 + i * 1.5
        t = novaphy.Transform.from_translation(
            np.array([0.0, y, 0.0], dtype=np.float32))
        idx = builder.add_body(body, t)
        shape = novaphy.CollisionShape.make_box(half, idx)
        builder.add_shape(shape)

    model = builder.build()
    settings = novaphy.SolverSettings()
    settings.velocity_iterations = 15
    world = novaphy.World(model, settings)

    # Run for ~3 seconds
    for _ in range(360):
        world.step(1.0 / 120.0)

    y0 = world.state.transforms[0].position[1]
    y1 = world.state.transforms[1].position[1]

    # Bottom box around 0.5, top box around 1.5
    assert y0 > 0.2, f"Bottom box too low: y={y0}"
    assert y0 < 1.0, f"Bottom box too high: y={y0}"
    assert y1 > y0, f"Top box should be above bottom box: y1={y1}, y0={y0}"


def test_sphere_falls_and_rests():
    builder = novaphy.ModelBuilder()
    builder.add_ground_plane(y=0.0)

    body = novaphy.RigidBody.from_sphere(1.0, 0.5)
    t = novaphy.Transform.from_translation(
        np.array([0.0, 3.0, 0.0], dtype=np.float32))
    idx = builder.add_body(body, t)

    shape = novaphy.CollisionShape.make_sphere(0.5, idx)
    builder.add_shape(shape)

    model = builder.build()
    world = novaphy.World(model)

    for _ in range(360):
        world.step(1.0 / 120.0)

    y = world.state.transforms[0].position[1]
    # Sphere radius is 0.5, so center at ~0.5
    assert y > 0.3, f"Sphere too low: y={y}"
    assert y < 1.0, f"Sphere should have settled: y={y}"


def test_solver_settings():
    settings = novaphy.SolverSettings()
    assert settings.velocity_iterations == 10
    assert settings.warm_starting is True

    settings.velocity_iterations = 20
    settings.baumgarte = 0.3
    assert settings.velocity_iterations == 20
    assert abs(settings.baumgarte - 0.3) < 1e-6


def test_apply_force():
    builder = novaphy.ModelBuilder()

    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body = novaphy.RigidBody.from_box(1.0, half)
    t = novaphy.Transform.from_translation(
        np.array([0.0, 5.0, 0.0], dtype=np.float32))
    idx = builder.add_body(body, t)

    model = builder.build()
    world = novaphy.World(model)
    world.set_gravity(np.array([0, 0, 0], dtype=np.float32))

    # Apply force in +X
    world.apply_force(0, np.array([10.0, 0, 0], dtype=np.float32))
    world.step(1.0 / 60.0)

    x = world.state.transforms[0].position[0]
    assert x > 0, f"Box should move in +X with applied force, got x={x}"
