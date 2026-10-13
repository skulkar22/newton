# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reduced elastic bodies in the VBD block-sparse articulation solve."""

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton.tests import test_reduced_elastic_body as elastic_tests
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
    model, state_0, data = elastic_tests._rod_model(
        device, mode="cos_offset", spin=(0.0, 0.0, 5.0), velocity=(0.3, -0.2, 0.1)
    )
    state_1 = model.state()
    control = model.control()
    solver = _solver(model, mode)
    com_0, momentum_0, angular_0 = elastic_tests._rod_physical(state_0, data)
    total_mass = float(np.sum(data["mass"]))
    ballistic = 0.0
    angular_drift = 0.0
    series = []
    for step in range(steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        com, _momentum, angular = elastic_tests._rod_physical(state_0, data)
        expected = com_0 + (step + 1) * dt * momentum_0 / total_mass
        ballistic = max(ballistic, float(np.linalg.norm(com - expected)))
        angular_drift = max(angular_drift, float(np.linalg.norm(angular - angular_0)))
        joint_q, _joint_qd = elastic_tests._rod_joint_state(state_0, data)
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
    theta3, theta4 = elastic_tests._solve_fourbar(theta2, a, b_rest + float(mode_q0[0]), c, d)
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
        mode_specs=[
            {"type": beam.Mode.AXIAL},
            {"type": beam.Mode.BENDING_Y, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
            {"type": beam.Mode.BENDING_Y, "boundary": beam.Boundary.PINNED_PINNED, "order": 2},
        ],
    ).build()

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
        mass=1.0,
        inertia=inertia,
        mode_count=3,
        mode_mass=[0.04, 0.035, 0.035],
        mode_stiffness=[10.0, 18.0, 288.0],
        mode_damping=[0.3, 0.3, 0.65],
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
    through a fill block. The loop residual stays within 10 % of the local solve at the same
    iteration count. (The local solve is far from converged at 24 iterations here; the sparse
    trajectory instead matches a 384-iteration local reference to about 1e-4.)
    """
    local_gaps, _local_series, _ = _run_fourbar(device, "local", iterations=24)
    sparse_gaps, sparse_series, solver = _run_fourbar(device, _SPARSE, iterations=24)
    layout = solver.rigid_articulation_sparse_layout
    test.assertEqual(layout.elastic_body_count, 1)
    test.assertEqual(layout.elastic_group_count, 3)
    test.assertTrue(np.isfinite(sparse_series).all())
    test.assertLess(float(sparse_gaps.max()), 1.1 * float(local_gaps.max()))
    test.assertLess(float(sparse_gaps[-1]), 1.0e-4)
    test.assertGreater(float(np.max(np.abs(sparse_series[-1] - sparse_series[0]))), 0.05)


def test_sparse_elastic_elastic_joint_stays_stable(test, device):
    """Keep a hinge between two elastic links stable in the sparse solve.

    The per-body elimination cannot represent the mode-mode coupling of the shared hinge; it adds
    the hinge's modal self term once more instead. Dropping the coupling alone makes the reduced
    system indefinite and diverges; with the conservative term the loop gap stays close to the
    local solve's at the same iteration count.
    """
    local_gaps, _local_state, _ = _run_elastic_chain(device, "local")
    sparse_gaps, state, solver = _run_elastic_chain(device, _SPARSE)
    test.assertEqual(solver.rigid_articulation_sparse_layout.elastic_body_count, 2)
    test.assertTrue(np.isfinite(state.joint_q.numpy()).all())
    test.assertTrue(np.isfinite(sparse_gaps).all())
    test.assertLess(float(sparse_gaps.max()), 1.3 * float(local_gaps.max()))
    test.assertLess(abs(float(sparse_gaps[-1]) - float(local_gaps[-1])), 0.05 * float(local_gaps[-1]))


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
    """Reject elastic-attached joint armature in the sparse solve and keep it in the local solve."""
    model, _state, hinge = _make_leaf_link_model(device)
    armature = model.joint_armature.numpy()
    armature[model.joint_qd_start.numpy()[hinge]] = 0.01
    model.joint_armature.assign(armature)
    with test.assertRaisesRegex(ValueError, "joint_armature"):
        _solver(model, _SPARSE)
    _solver(model, "local")


class TestReducedElasticSparseArticulation(unittest.TestCase):
    pass


for _name, _test in (
    ("test_sparse_standalone_elastic_body_matches_local", test_sparse_standalone_elastic_body_matches_local),
    ("test_sparse_elastic_leaf_link_matches_local", test_sparse_elastic_leaf_link_matches_local),
    ("test_sparse_elastic_coupler_closed_loop", test_sparse_elastic_coupler_closed_loop),
    ("test_sparse_elastic_elastic_joint_stays_stable", test_sparse_elastic_elastic_joint_stays_stable),
    ("test_sparse_elastic_rejects_unsupported_joint_terms", test_sparse_elastic_rejects_unsupported_joint_terms),
):
    add_function_test(TestReducedElasticSparseArticulation, _name, _test, devices=devices)

for _name, _test in (
    ("test_sparse_elastic_cpu_cuda_parity", test_sparse_elastic_cpu_cuda_parity),
    ("test_sparse_elastic_leaf_link_graph_capture", test_sparse_elastic_leaf_link_graph_capture),
):
    add_function_test(TestReducedElasticSparseArticulation, _name, _test, devices=cuda_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
