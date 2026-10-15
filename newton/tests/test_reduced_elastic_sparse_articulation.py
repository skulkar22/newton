# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reduced elastic bodies in the VBD block-sparse articulation solve."""

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton._src.solvers.vbd import reduced_elastic_kernels, reduced_elastic_sparse_kernels
from newton.tests import _reduced_elastic_fixtures as elastic_fixtures
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

devices = get_test_devices()
cuda_devices = get_cuda_test_devices()

_SPARSE = "block_sparse_joints"


def _solver(model, mode: str, iterations: int = 24, relaxation: float = 1.0, **kwargs):
    return newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_articulation_solve=mode,
        rigid_articulation_relaxation=relaxation,
        **kwargs,
    )


def _run_standalone_rod(device, mode: str, steps: int = 192, dt: float = 1.0 / 960.0):
    """Step the spinning, boosted one-mode rod and record its joint state and momentum metrics."""
    model, state_0, data = elastic_fixtures.rod_model(
        device, mode="cos_offset", spin=(0.0, 0.0, 5.0), velocity=(0.3, -0.2, 0.1)
    )
    state_1 = model.state()
    control = model.control()
    solver = _solver(model, mode)
    com_0, momentum_0, angular_0 = elastic_fixtures.rod_physical(state_0, data)
    total_mass = float(np.sum(data["mass"]))
    ballistic = 0.0
    angular_drift = 0.0
    series = []
    for step in range(steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        com, _momentum, angular = elastic_fixtures.rod_physical(state_0, data)
        expected = com_0 + (step + 1) * dt * momentum_0 / total_mass
        ballistic = max(ballistic, float(np.linalg.norm(com - expected)))
        angular_drift = max(angular_drift, float(np.linalg.norm(angular - angular_0)))
        joint_q, _joint_qd = elastic_fixtures.rod_joint_state(state_0, data)
        series.append(joint_q)
    return np.asarray(series), ballistic, angular_drift, solver


def _make_leaf_link_model(device, *, gravity: float = -9.81):
    """Build a rigid base hinged to the world carrying a reduced elastic leaf link.

    The link's owner joint is created before, and lies outside, the articulation of the two
    revolute joints. Axial and two pinned-pinned bending modes give the hinge attachment both a
    translational (axial) and an angular (out-of-plane bending slope) modal motion.
    """
    length, half_y, half_z = 0.6, 0.03, 0.02
    beam = newton.ModalGeneratorBeam
    basis = beam(
        length=length,
        half_width_y=half_y,
        half_width_z=half_z,
        young_modulus=2.0e6,
        density=1000.0,
        mode_specs=[
            {"type": beam.Mode.AXIAL},
            {"type": beam.Mode.BENDING_Y, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
            {"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
        ],
    ).build()
    mass = 1000.0 * 4.0 * half_y * half_z * length
    inertia = np.diag(
        [
            mass * (half_y**2 + half_z**2) / 3.0,
            mass * (length**2 / 12.0 + half_z**2 / 3.0),
            mass * (length**2 / 12.0 + half_y**2 / 3.0),
        ]
    )
    builder = newton.ModelBuilder(gravity=(0.0, gravity, 0.0))
    base = builder.add_link(
        xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(*np.diag([0.002, 0.02, 0.02]).ravel()),
    )
    link = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(0.4 + 0.5 * length, 0.0, 0.0), wp.quat_identity()),
        mass=mass,
        inertia=wp.mat33(*inertia.ravel()),
        modal_basis=basis,
        mode_q=[0.0, 0.0, 0.01],
    )
    root = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=(0.0, 0.0, 1.0),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    hinge = builder.add_joint_revolute(
        parent=base,
        child=link,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([root, hinge])
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[model.joint_qd_start.numpy()[root]] = 2.0
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state, hinge


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.asarray(q[:3], dtype=np.float64)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + float(q[3]) * v)


def _joint_anchor(model, state, joint: int, parent_side: bool, *, deformed: bool) -> np.ndarray:
    """Return a joint attachment point [m] in world, optionally moved by its elastic endpoint modes."""
    body = int((model.joint_parent if parent_side else model.joint_child).numpy()[joint])
    xform = (model.joint_X_p if parent_side else model.joint_X_c).numpy()[joint]
    point = np.asarray(xform[:3], dtype=np.float64)
    endpoint = int(
        (model.joint_parent_elastic_endpoint if parent_side else model.joint_child_elastic_endpoint).numpy()[joint]
    )
    if deformed and body >= 0 and endpoint >= 0:
        elastic = int(model.body_elastic_index.numpy()[body])
        start = int(model.joint_q_start.numpy()[model.elastic_joint.numpy()[elastic]]) + 7
        max_modes = model.elastic_max_mode_count
        phi = model.elastic_endpoint_phi.numpy()
        joint_q = state.joint_q.numpy()
        for mode in range(int(model.elastic_mode_count.numpy()[elastic])):
            point = point + phi[endpoint * max_modes + mode] * float(joint_q[start + mode])
    if body < 0:
        return point
    pose = state.body_q.numpy()[body]
    return pose[:3] + _quat_rotate(pose[3:7], point)


def _joint_gap(model, state, joint: int, *, deformed: bool = True) -> float:
    child = _joint_anchor(model, state, joint, False, deformed=deformed)
    parent = _joint_anchor(model, state, joint, True, deformed=deformed)
    return float(np.linalg.norm(child - parent))


def _max_joint_gap(model, state) -> float:
    joint_type = model.joint_type.numpy()
    return max(
        _joint_gap(model, state, joint)
        for joint in range(model.joint_count)
        if joint_type[joint] != int(newton.JointType.ELASTIC)
    )


def _run_leaf_link(
    device, mode: str, seconds: float = 0.5, dt: float = 1.0 / 480.0, relaxation: float = 1.0, **solver_kwargs
):
    model, state_0, hinge = _make_leaf_link_model(device)
    state_1 = model.state()
    control = model.control()
    solver = _solver(model, mode, relaxation=relaxation, **solver_kwargs)
    series = []
    for _ in range(round(seconds / dt)):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        series.append(np.concatenate((state_0.body_q.numpy().ravel(), state_0.joint_q.numpy())))
    return np.asarray(series), model, state_0, hinge, solver


def _make_elastic_chain_model(device):
    """Build two reduced elastic beams hinged to the world and to each other."""
    length, half_y, half_z = 0.6, 0.03, 0.02
    beam = newton.ModalGeneratorBeam
    basis = beam(
        length=length,
        half_width_y=half_y,
        half_width_z=half_z,
        young_modulus=2.0e6,
        density=1000.0,
        mode_specs=[
            {"type": beam.Mode.AXIAL},
            {"type": beam.Mode.BENDING_Y, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
            {"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
        ],
    ).build()
    mass = 1000.0 * 4.0 * half_y * half_z * length
    inertia = wp.mat33(
        *np.diag([mass * (half_y**2 + half_z**2) / 3.0, mass * length**2 / 12.0, mass * length**2 / 12.0]).ravel()
    )
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    links = [
        builder.add_link_elastic(
            xform=wp.transform(wp.vec3((i + 0.5) * length, 0.0, 0.0), wp.quat_identity()),
            mass=mass,
            inertia=inertia,
            modal_basis=basis,
            mode_q=[0.0, 0.0, 0.01 * (1 - 2 * i)],
        )
        for i in range(2)
    ]
    half = wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity())
    minus_half = wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity())
    root = builder.add_joint_revolute(parent=-1, child=links[0], axis=(0.0, 0.0, 1.0), child_xform=minus_half)
    builder.add_joint_revolute(
        parent=links[0], child=links[1], axis=(0.0, 0.0, 1.0), parent_xform=half, child_xform=minus_half
    )
    builder.add_articulation([root, root + 1])
    builder.color()
    return builder.finalize(device=device)


def _run_elastic_chain(device, mode: str, iterations: int = 24, steps: int = 240, dt: float = 1.0 / 480.0):
    model = _make_elastic_chain_model(device)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    solver = _solver(model, mode, iterations=iterations)
    gaps = []
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        gaps.append(_max_joint_gap(model, state_0))
    return np.asarray(gaps), state_0, solver


def _make_elastic_fourbar_model(device):
    """Build the elastic-coupler four-bar with its ground closure declared inside the articulation."""
    a, b_rest, c, d, z = 0.35, 0.65, 0.55, 0.75, 0.12
    theta2 = 0.75
    mode_q0 = np.array([0.08, 0.08, 0.0], dtype=np.float32)
    theta3, theta4 = elastic_fixtures.solve_fourbar(theta2, a, b_rest + float(mode_q0[0]), c, d)
    ground_a = np.array([0.0, 0.0, z])
    ground_d = np.array([d, 0.0, z])
    e2 = np.array([math.cos(theta2), math.sin(theta2), 0.0])
    e3 = np.array([math.cos(theta3), math.sin(theta3), 0.0])
    e4 = np.array([math.cos(theta4), math.sin(theta4), 0.0])
    crank_tip = ground_a + a * e2

    beam = newton.ModalGeneratorBeam
    basis = beam(
        length=b_rest,
        half_width_y=0.05,
        half_width_z=0.03,
        density=1.0 / (4.0 * 0.05 * 0.03 * b_rest),
        mode_specs=[
            {"type": beam.Mode.AXIAL},
            {"type": beam.Mode.BENDING_Y, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
            {"type": beam.Mode.BENDING_Y, "boundary": beam.Boundary.PINNED_PINNED, "order": 2},
        ],
    ).build()
    # As in the four-bar example, the 1 kg coupler takes its frame and modal inertia from the basis.
    stiffness = [10.0, 18.0, 288.0]
    damping = [
        2.0 * r * math.sqrt(k * float(m))
        for r, k, m in zip((0.24, 0.19, 0.10), stiffness, basis.mode_mass, strict=True)
    ]

    def rotation_z(angle):
        return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)

    def offset(x):
        return wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity())

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(*np.diag([0.01, 0.01, 0.01]).ravel())
    crank = builder.add_link(
        xform=wp.transform(wp.vec3(*(ground_a + 0.5 * a * e2)), rotation_z(theta2)), mass=1.0, inertia=inertia
    )
    coupler = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(*(crank_tip + 0.5 * (b_rest + float(mode_q0[0])) * e3)), rotation_z(theta3)),
        mass=basis.rigid_mass,
        com=wp.vec3(*basis.rigid_com),
        inertia=wp.mat33(*basis.rigid_inertia.ravel()),
        mode_count=3,
        mode_stiffness=stiffness,
        mode_damping=damping,
        mode_q=mode_q0,
        modal_basis=basis,
    )
    rocker = builder.add_link(
        xform=wp.transform(wp.vec3(*(ground_d + 0.5 * c * e4)), rotation_z(theta4)), mass=1.0, inertia=inertia
    )
    axis = (0.0, 0.0, 1.0)
    drive = builder.add_joint_revolute(
        parent=-1,
        child=crank,
        axis=axis,
        parent_xform=wp.transform(wp.vec3(*ground_a), wp.quat_identity()),
        child_xform=offset(-0.5 * a),
        target_pos=theta2 + 0.3,
        target_ke=40.0,
        target_kd=0.08,
    )
    crank_coupler = builder.add_joint_revolute(
        parent=crank, child=coupler, axis=axis, parent_xform=offset(0.5 * a), child_xform=offset(-0.5 * b_rest)
    )
    builder.add_joint_revolute(
        parent=coupler, child=rocker, axis=axis, parent_xform=offset(0.5 * b_rest), child_xform=offset(0.5 * c)
    )
    closure = builder.add_joint_revolute(
        parent=-1,
        child=rocker,
        axis=axis,
        parent_xform=wp.transform(wp.vec3(*ground_d), wp.quat_identity()),
        child_xform=offset(-0.5 * c),
    )
    builder.joint_q[builder.joint_q_start[drive]] = theta2
    builder.add_articulation(list(range(drive, closure + 1)), allow_closed_loops=True)
    builder.color()
    return builder.finalize(device=device), crank_coupler


def _run_fourbar(device, mode: str, iterations: int, steps: int = 200, dt: float = 1.0 / 120.0):
    model, _crank_coupler = _make_elastic_fourbar_model(device)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    solver = _solver(model, mode, iterations=iterations, relaxation=0.65)
    gaps = []
    series = []
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        gaps.append(_max_joint_gap(model, state_0))
        series.append(state_0.joint_q.numpy().copy())
    return np.asarray(gaps), np.asarray(series), solver


def _beam_basis(mode_specs, length: float = 0.6, half_y: float = 0.03, half_z: float = 0.02):
    """Return a soft beam basis with the given mode specifications."""
    return newton.ModalGeneratorBeam(
        length=length,
        half_width_y=half_y,
        half_width_z=half_z,
        young_modulus=2.0e6,
        density=1000.0,
        mode_specs=mode_specs,
    ).build()


def _make_kinematic_elastic_parent_model(device):
    """Build a kinematic elastic beam, spinning about its root hinge, carrying a hinged rigid child.

    The beam's frame motion is prescribed while its modes respond to the child's hinge load.
    """
    length = 0.6
    beam = newton.ModalGeneratorBeam
    basis = _beam_basis(
        [
            {"type": beam.Mode.AXIAL},
            {"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
        ],
        length=length,
    )
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    link = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity()),
        modal_basis=basis,
        mode_q=[0.0, 0.01],
        is_kinematic=True,
    )
    child = builder.add_link(
        xform=wp.transform(wp.vec3(length + 0.2, 0.0, 0.0), wp.quat_identity()),
        mass=0.5,
        inertia=wp.mat33(*np.diag([0.001, 0.01, 0.01]).ravel()),
    )
    root = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=(0.0, 0.0, 1.0),
        child_xform=wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity()),
    )
    hinge = builder.add_joint_revolute(
        parent=link,
        child=child,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([root, hinge])
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[model.joint_qd_start.numpy()[root]] = 1.5
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


def _make_mixed_mode_count_model(device):
    """Build a chain of a one-mode elastic link, a rigid link, and a three-mode elastic link.

    Both elastic links have fewer modes than the model's widest basis would allow if padded, so the
    sparse layout must handle per-body mode counts below ``elastic_max_mode_count``.
    """
    length = 0.5
    beam = newton.ModalGeneratorBeam
    one_mode = _beam_basis(
        [{"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 1}], length=length
    )
    three_modes = _beam_basis(
        [
            {"type": beam.Mode.AXIAL},
            {"type": beam.Mode.BENDING_Y, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
            {"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
        ],
        length=length,
    )
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    first = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity()),
        modal_basis=one_mode,
        mode_q=[0.01],
    )
    middle = builder.add_link(
        xform=wp.transform(wp.vec3(length + 0.1, 0.0, 0.0), wp.quat_identity()),
        mass=0.4,
        inertia=wp.mat33(*np.diag([0.001, 0.004, 0.004]).ravel()),
    )
    last = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(1.5 * length + 0.2, 0.0, 0.0), wp.quat_identity()),
        modal_basis=three_modes,
        mode_q=[0.0, 0.0, -0.01],
    )
    half = wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity())
    minus_half = wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity())
    axis = (0.0, 0.0, 1.0)
    joints = [
        builder.add_joint_revolute(parent=-1, child=first, axis=axis, child_xform=minus_half),
        builder.add_joint_revolute(
            parent=first,
            child=middle,
            axis=axis,
            parent_xform=half,
            child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
        ),
        builder.add_joint_revolute(
            parent=middle,
            child=last,
            axis=axis,
            parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
            child_xform=minus_half,
        ),
    ]
    builder.add_articulation(joints)
    builder.color()
    model = builder.finalize(device=device)
    return model, model.state()


def _run_model(model, state_0, mode: str, iterations: int, steps: int, dt: float = 1.0 / 480.0, **kwargs):
    """Step ``model`` and return the ``joint_q`` series and the solver."""
    state_1 = model.state()
    control = model.control()
    solver = _solver(model, mode, iterations=iterations, **kwargs)
    series = []
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        series.append(state_0.joint_q.numpy().copy())
    return np.asarray(series), solver


def test_sparse_standalone_elastic_body_matches_local(test, device):
    """Match a free spinning, boosted elastic body between the sparse and local solves.

    The body's one-joint articulation puts it in the sparse solve, where its modes are eliminated
    onto the frame block. Per step, both solves converge to the same state to float32 precision;
    over 0.2 s the trajectories differ only by the accumulated float32 rounding of each update
    form (about 1e-5 here, the same as local CPU versus local CUDA), so the sparse solve must keep
    the ballistic COM and the angular momentum at least as well as the local solve.
    """
    local, local_ballistic, local_angular, _ = _run_standalone_rod(device, "local")
    sparse, sparse_ballistic, sparse_angular, solver = _run_standalone_rod(device, _SPARSE)
    layout = solver.rigid_articulation_sparse_layout
    test.assertEqual(layout.elastic_body_count, 1)
    test.assertEqual(layout.local_body_count, 0)
    np.testing.assert_allclose(sparse, local, atol=1.0e-4, rtol=0.0)
    test.assertGreater(float(np.max(np.abs(sparse[-1, 7] - sparse[0, 7]))), 1.0e-3)
    test.assertLess(sparse_ballistic, 1.1 * local_ballistic + 1.0e-6)
    test.assertLess(sparse_angular, 1.1 * local_angular + 1.0e-6)


def test_sparse_elastic_leaf_link_matches_local(test, device):
    """Match a rigid base carrying an elastic leaf link between the sparse and local solves.

    Also check that the sparse solve holds the hinge together at the deformed attachment: with
    stiff joints, the hinge gap measured with the elastic endpoint displaced by ``Phi q`` is far
    smaller than the gap at the undeformed attachment.
    """
    local, *_ = _run_leaf_link(device, "local")
    sparse, _model, _state, _hinge, solver = _run_leaf_link(device, _SPARSE)
    test.assertEqual(solver.rigid_articulation_sparse_layout.elastic_body_count, 1)
    test.assertTrue(np.isfinite(sparse).all())
    np.testing.assert_allclose(sparse, local, atol=2.0e-4, rtol=0.0)
    test.assertGreater(float(np.max(np.abs(sparse[-1] - sparse[0]))), 0.5)

    _stiff, model, state, hinge, _solver_stiff = _run_leaf_link(
        device, _SPARSE, rigid_joint_linear_ke=1.0e6, rigid_joint_angular_ke=1.0e6
    )
    deformed_gap = _joint_gap(model, state, hinge, deformed=True)
    rigid_gap = _joint_gap(model, state, hinge, deformed=False)
    test.assertGreater(rigid_gap, 5.0e-4)
    test.assertLess(deformed_gap, 0.1 * rigid_gap)


def test_sparse_elastic_coupler_closed_loop(test, device):
    """Close a four-bar with an elastic coupler in one sparse factorization.

    The coupler's modes are eliminated onto the crank and rocker frames, which the layout couples
    through a fill block. At 24 iterations the sparse trajectory matches the converged local solve
    (384 iterations) within 5e-4 in every joint coordinate (about 2e-4 measured), while the local
    solve at 24 iterations is still about 0.12 rad away, and its loop residual stays within 5 % of
    the converged one.
    """
    reference_gaps, reference_series, _ = _run_fourbar(device, "local", iterations=384)
    sparse_gaps, sparse_series, solver = _run_fourbar(device, _SPARSE, iterations=24)
    layout = solver.rigid_articulation_sparse_layout
    test.assertEqual(layout.elastic_body_count, 1)
    test.assertEqual(layout.elastic_group_count, 3)
    test.assertTrue(np.isfinite(sparse_series).all())
    np.testing.assert_allclose(sparse_series, reference_series, rtol=0.0, atol=5.0e-4)
    test.assertLess(float(sparse_gaps.max()), 1.05 * float(reference_gaps.max()))
    test.assertLess(float(sparse_gaps[-1]), 1.0e-4)
    test.assertGreater(float(np.max(np.abs(sparse_series[-1] - sparse_series[0]))), 0.05)


def test_sparse_elastic_elastic_joint_stays_stable(test, device):
    """Keep a hinge between two elastic links stable in the sparse solve.

    The per-body elimination cannot represent the mode-mode coupling of the shared hinge; it adds
    the hinge's modal self term once more instead. Dropping the coupling alone makes the reduced
    system indefinite and diverges. With the conservative term, compared with the converged local
    solve (384 iterations), the peak hinge gap stays within 30 % (about 20 % measured), the final gap
    within 5 % (0.2 % measured), and the joint coordinates within 1e-2 (4e-3 measured, the effect of
    the extra modal term).
    """
    reference_gaps, reference_state, _ = _run_elastic_chain(device, "local", iterations=384)
    sparse_gaps, state, solver = _run_elastic_chain(device, _SPARSE)
    test.assertEqual(solver.rigid_articulation_sparse_layout.elastic_body_count, 2)
    test.assertTrue(np.isfinite(state.joint_q.numpy()).all())
    test.assertTrue(np.isfinite(sparse_gaps).all())
    test.assertLess(float(sparse_gaps.max()), 1.3 * float(reference_gaps.max()))
    test.assertLess(abs(float(sparse_gaps[-1]) - float(reference_gaps[-1])), 0.05 * float(reference_gaps[-1]))
    np.testing.assert_allclose(state.joint_q.numpy(), reference_state.joint_q.numpy(), rtol=0.0, atol=1.0e-2)


def _elastic_mode_series(model, series: np.ndarray, elastic_index: int) -> np.ndarray:
    """Return the modal coordinate columns of one elastic body from a ``joint_q`` series."""
    owner = int(model.elastic_joint.numpy()[elastic_index])
    start = int(model.joint_q_start.numpy()[owner]) + 7
    return series[:, start : start + int(model.elastic_mode_count.numpy()[elastic_index])]


def test_sparse_kinematic_elastic_parent_matches_local(test, device):
    """Match a kinematic elastic beam carrying a hinged rigid child between the sparse and local solves.

    The beam's frame motion is prescribed, so the sparse solve eliminates only its modes. At 128
    iterations both solves agree within 2e-4 in every joint coordinate (about 8e-5 measured, the
    local solve's own distance to 384 iterations is 2e-5), and the child's hinge load moves the modes.
    """
    model, state = _make_kinematic_elastic_parent_model(device)
    local, _ = _run_model(model, state, "local", iterations=128, steps=240)
    model, state = _make_kinematic_elastic_parent_model(device)
    sparse, solver = _run_model(model, state, _SPARSE, iterations=128, steps=240)
    test.assertEqual(solver.rigid_articulation_sparse_layout.elastic_body_count, 1)
    test.assertTrue(np.isfinite(sparse).all())
    np.testing.assert_allclose(sparse, local, rtol=0.0, atol=2.0e-4)
    modes = _elastic_mode_series(model, sparse, 0)
    test.assertGreater(float(np.ptp(modes[:, 0])), 1.0e-4)


def test_sparse_mixed_mode_counts_match_local(test, device):
    """Match a chain of one-mode and three-mode elastic links around a rigid link to the local solve.

    At 128 iterations both solves agree within 2e-4 in every joint coordinate (about 3e-5 measured).
    """
    model, state = _make_mixed_mode_count_model(device)
    test.assertEqual(model.elastic_mode_count.numpy().tolist(), [1, 3])
    local, _ = _run_model(model, state, "local", iterations=128, steps=240)
    model, state = _make_mixed_mode_count_model(device)
    sparse, solver = _run_model(model, state, _SPARSE, iterations=128, steps=240)
    test.assertEqual(solver.rigid_articulation_sparse_layout.elastic_body_count, 2)
    test.assertTrue(np.isfinite(sparse).all())
    np.testing.assert_allclose(sparse, local, rtol=0.0, atol=2.0e-4)
    for elastic_index in range(2):
        modes = _elastic_mode_series(model, sparse, elastic_index)
        test.assertGreater(float(np.max(np.ptp(modes, axis=0))), 1.0e-4)


def _assert_cpu_cuda_parity(test, device, run, label: str):
    cpu = run("cpu")
    cuda = run(device)
    np.testing.assert_allclose(cuda, cpu, rtol=2.0e-4, atol=2.0e-4, err_msg=label)


def test_sparse_elastic_cpu_cuda_parity(test, device):
    """Match the CUDA sparse elastic solve to the CPU serial solve."""
    _assert_cpu_cuda_parity(test, device, lambda d: _run_standalone_rod(d, _SPARSE)[0], "standalone rod")
    _assert_cpu_cuda_parity(test, device, lambda d: _run_leaf_link(d, _SPARSE)[0], "leaf link")


def test_sparse_elastic_leaf_link_graph_capture(test, device):
    """Replay the sparse elastic leaf-link step from a captured CUDA graph."""
    dt = 1.0 / 480.0

    def setup():
        model, state_0, _hinge = _make_leaf_link_model(device)
        return model, _solver(model, _SPARSE, relaxation=0.65), state_0, model.state(), model.control()

    def advance(system):
        _model, solver, state_0, state_1, control = system
        solver.step(state_0, state_1, control, None, dt)
        solver.step(state_1, state_0, control, None, dt)

    captured = setup()
    reference = setup()
    advance(captured)
    advance(reference)
    with wp.ScopedCapture(device=device) as capture:
        advance(captured)
    for _ in range(4):
        wp.capture_launch(capture.graph)
        advance(reference)
    for attr in ("body_q", "body_qd", "joint_q", "joint_qd"):
        np.testing.assert_allclose(
            getattr(captured[2], attr).numpy(), getattr(reference[2], attr).numpy(), atol=1.0e-5, rtol=1.0e-5
        )


def test_sparse_elastic_rejects_unsupported_joint_terms(test, device):
    """Reject elastic-attached joint armature, damping, and friction in the sparse solve.

    The local solve keeps each of them.
    """
    for attribute in ("joint_armature", "joint_damping", "joint_friction"):
        with test.subTest(attribute=attribute):
            model, _state, hinge = _make_leaf_link_model(device)
            values = getattr(model, attribute).numpy()
            values[model.joint_qd_start.numpy()[hinge]] = 0.01
            getattr(model, attribute).assign(values)
            with test.assertRaisesRegex(ValueError, "joint_armature, joint_damping, or joint_friction"):
                _solver(model, _SPARSE)
            _solver(model, "local")


def test_sparse_elastic_rejects_joint_mimics(test, device):
    """Reject joint mimics together with elastic articulation bodies in the sparse solve only."""
    basis = _beam_basis([{"type": newton.ModalGeneratorBeam.Mode.AXIAL}])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    link = builder.add_link_elastic(modal_basis=basis)
    followers = [builder.add_link(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel())) for _ in range(2)]
    joints = [builder.add_joint_revolute(parent=-1, child=link, axis=(0.0, 0.0, 1.0))]
    joints += [builder.add_joint_revolute(parent=link, child=body, axis=(0.0, 0.0, 1.0)) for body in followers]
    builder.set_joint_mimic(joints[2], joints[1], coeffs=(0.0, 1.0))
    builder.add_articulation(joints)
    builder.color()
    model = builder.finalize(device=device)
    with test.assertRaisesRegex(NotImplementedError, "mimic"):
        _solver(model, _SPARSE)
    _solver(model, "local")


def test_elastic_articulation_honors_deterministic_option(test, device):
    """Forward the requested deterministic mode to the reduced elastic kernel modules.

    The elastic block assembly, the frame coupling and the sparse Schur scatter accumulate with
    atomics, so their modules must receive the solver's deterministic mode in both the local and the
    block-sparse articulation solve.
    """
    modules = (reduced_elastic_kernels, reduced_elastic_sparse_kernels)
    original = [wp.get_module_options(module=module) for module in modules]
    try:
        for mode in ("local", _SPARSE):
            with test.subTest(mode=mode):
                for module, options in zip(modules, original, strict=True):
                    wp.set_module_options(options, module=module)
                _solver(_make_elastic_chain_model(device), mode, deterministic=wp.DeterministicMode.RUN_TO_RUN)
                for module in modules:
                    options = wp.get_module_options(module=module)
                    test.assertEqual(options["deterministic"], wp.DeterministicMode.RUN_TO_RUN)
    finally:
        for module, options in zip(modules, original, strict=True):
            wp.set_module_options(options, module=module)


class TestReducedElasticSparseArticulation(unittest.TestCase):
    pass


for _name, _test in (
    ("test_sparse_standalone_elastic_body_matches_local", test_sparse_standalone_elastic_body_matches_local),
    ("test_sparse_elastic_leaf_link_matches_local", test_sparse_elastic_leaf_link_matches_local),
    ("test_sparse_elastic_coupler_closed_loop", test_sparse_elastic_coupler_closed_loop),
    ("test_sparse_elastic_elastic_joint_stays_stable", test_sparse_elastic_elastic_joint_stays_stable),
    ("test_sparse_kinematic_elastic_parent_matches_local", test_sparse_kinematic_elastic_parent_matches_local),
    ("test_sparse_mixed_mode_counts_match_local", test_sparse_mixed_mode_counts_match_local),
    ("test_sparse_elastic_rejects_unsupported_joint_terms", test_sparse_elastic_rejects_unsupported_joint_terms),
    ("test_sparse_elastic_rejects_joint_mimics", test_sparse_elastic_rejects_joint_mimics),
    ("test_elastic_articulation_honors_deterministic_option", test_elastic_articulation_honors_deterministic_option),
):
    add_function_test(TestReducedElasticSparseArticulation, _name, _test, devices=devices)

for _name, _test in (
    ("test_sparse_elastic_cpu_cuda_parity", test_sparse_elastic_cpu_cuda_parity),
    ("test_sparse_elastic_leaf_link_graph_capture", test_sparse_elastic_leaf_link_graph_capture),
):
    add_function_test(TestReducedElasticSparseArticulation, _name, _test, devices=cuda_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
