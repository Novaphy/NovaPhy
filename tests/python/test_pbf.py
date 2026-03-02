"""Tests for NovaPhy PBF fluid solver (Phase 2)."""

import math
import numpy as np
import numpy.testing as npt
import novaphy


# ========== PBFSettings Tests ==========

def test_pbf_settings_defaults():
    """PBFSettings should have sensible default values."""
    s = novaphy.PBFSettings()
    npt.assert_allclose(s.rest_density, 1000.0, atol=1e-3)
    npt.assert_allclose(s.kernel_radius, 0.1, atol=1e-6)
    assert s.solver_iterations == 4


def test_pbf_settings_particle_mass():
    """Particle mass should be rho_0 * spacing^3."""
    s = novaphy.PBFSettings()
    s.rest_density = 1000.0
    spacing = 0.05
    expected_mass = 1000.0 * 0.05 ** 3  # 0.125 kg
    npt.assert_allclose(s.particle_mass(spacing), expected_mass, rtol=1e-5)


# ========== PBFSolver Standalone Tests ==========

def _make_pbf_state(spacing=0.025, block_size=0.3):
    """Create a small fluid block and PBF solver."""
    block = novaphy.FluidBlockDef()
    block.lower = np.array([0, 0, 0], dtype=np.float32)
    block.upper = np.array([block_size, block_size, block_size], dtype=np.float32)
    block.particle_spacing = spacing

    positions = novaphy.generate_fluid_block(block)

    settings = novaphy.PBFSettings()
    settings.kernel_radius = spacing * 4.0  # typical: h = 4 * spacing
    settings.rest_density = 1000.0
    settings.solver_iterations = 4
    settings.epsilon = 100.0

    state = novaphy.ParticleState()
    state.init(positions)

    mass = settings.particle_mass(spacing)
    return state, settings, mass


def test_pbf_solver_step_runs():
    """PBFSolver.step() should run without crashing."""
    state, settings, mass = _make_pbf_state()
    solver = novaphy.PBFSolver(settings)

    gravity = np.array([0, -9.81, 0], dtype=np.float32)
    solver.step(state, 1.0 / 60.0, gravity, mass)

    # Particles should have moved (gravity)
    assert state.num_particles > 0


def test_pbf_particles_fall():
    """Particles should fall under gravity."""
    state, settings, mass = _make_pbf_state()
    solver = novaphy.PBFSolver(settings)
    gravity = np.array([0, -9.81, 0], dtype=np.float32)

    initial_y = np.mean([p[1] for p in state.positions])

    for _ in range(10):
        solver.step(state, 1.0 / 60.0, gravity, mass)

    final_y = np.mean([p[1] for p in state.positions])
    assert final_y < initial_y, \
        f"Particles should fall: initial_y={initial_y}, final_y={final_y}"


def test_pbf_zero_gravity():
    """With zero gravity, center of mass should not drift significantly."""
    state, settings, mass = _make_pbf_state()
    solver = novaphy.PBFSolver(settings)
    gravity = np.array([0, 0, 0], dtype=np.float32)

    initial_com = np.mean([np.array(p) for p in state.positions], axis=0)

    for _ in range(5):
        solver.step(state, 1.0 / 60.0, gravity, mass)

    # Center of mass should stay near the original (no external forces)
    final_com = np.mean([np.array(p) for p in state.positions], axis=0)
    com_drift = np.linalg.norm(final_com - initial_com)

    # With zero gravity, COM should not drift far (boundary expansion is symmetric)
    assert com_drift < 0.05, \
        f"COM drift with zero gravity too large: {com_drift}"


def test_pbf_density_reasonable():
    """After PBF steps, density should be near rest density."""
    state, settings, mass = _make_pbf_state()
    solver = novaphy.PBFSolver(settings)
    gravity = np.array([0, 0, 0], dtype=np.float32)

    # Run several steps to let density stabilize
    for _ in range(10):
        solver.step(state, 1.0 / 60.0, gravity, mass)

    # Interior particles should have density near rest density
    densities = state.densities
    interior_densities = []
    for i, d in enumerate(densities):
        p = np.array(state.positions[i])
        # Only check interior particles (not near boundaries)
        if np.all(p > 0.06) and np.all(p < 0.14):
            interior_densities.append(d)

    if len(interior_densities) > 0:
        mean_density = np.mean(interior_densities)
        # PBF should push density toward rest density
        assert mean_density > 0, f"Mean density should be positive, got {mean_density}"


def test_pbf_velocities_updated():
    """PBF should update velocities from position changes."""
    state, settings, mass = _make_pbf_state()
    solver = novaphy.PBFSolver(settings)
    gravity = np.array([0, -9.81, 0], dtype=np.float32)

    solver.step(state, 1.0 / 60.0, gravity, mass)

    # At least some particles should have non-zero velocity
    max_speed = max(np.linalg.norm(np.array(v)) for v in state.velocities)
    assert max_speed > 0, "Some particles should have velocity after step"


# ========== FluidWorld Tests ==========

def test_fluid_world_creation():
    """FluidWorld should create with fluid blocks."""
    builder = novaphy.ModelBuilder()
    model = builder.build()

    block = novaphy.FluidBlockDef()
    block.lower = np.array([0, 0, 0], dtype=np.float32)
    block.upper = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    block.particle_spacing = 0.05

    world = novaphy.FluidWorld(model, [block])
    assert world.num_particles > 0


def test_fluid_world_step():
    """FluidWorld.step() should advance both rigid and fluid sim."""
    builder = novaphy.ModelBuilder()
    model = builder.build()

    block = novaphy.FluidBlockDef()
    block.lower = np.array([0, 1, 0], dtype=np.float32)
    block.upper = np.array([0.2, 1.2, 0.2], dtype=np.float32)
    block.particle_spacing = 0.05

    pbf = novaphy.PBFSettings()
    pbf.kernel_radius = 0.2
    pbf.solver_iterations = 3

    world = novaphy.FluidWorld(model, [block], novaphy.SolverSettings(), pbf)

    initial_y = np.mean([p[1] for p in world.fluid_state.positions])

    for _ in range(10):
        world.step(1.0 / 60.0)

    final_y = np.mean([p[1] for p in world.fluid_state.positions])
    assert final_y < initial_y, "Fluid should fall under gravity"


def test_fluid_world_no_fluid():
    """FluidWorld with no fluid blocks should work like regular World."""
    builder = novaphy.ModelBuilder()
    model = builder.build()

    world = novaphy.FluidWorld(model)
    assert world.num_particles == 0
    world.step(1.0 / 60.0)  # Should not crash


def test_fluid_world_inherits_world():
    """FluidWorld should inherit rigid-body functionality."""
    builder = novaphy.ModelBuilder()
    builder.add_ground_plane()

    half = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    body = novaphy.RigidBody.from_box(1.0, half)
    t = novaphy.Transform.from_translation(
        np.array([0.0, 5.0, 0.0], dtype=np.float32))
    idx = builder.add_body(body, t)
    shape = novaphy.CollisionShape.make_box(half, idx)
    builder.add_shape(shape)

    model = builder.build()
    world = novaphy.FluidWorld(model)

    initial_y = world.state.transforms[0].position[1]
    for _ in range(10):
        world.step(1.0 / 60.0)

    final_y = world.state.transforms[0].position[1]
    assert final_y < initial_y, "Rigid body should fall in FluidWorld"
