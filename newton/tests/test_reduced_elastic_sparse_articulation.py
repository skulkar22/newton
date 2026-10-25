# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reduced elastic bodies in the VBD block-sparse articulation solve."""

import math
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton._src.solvers.vbd import (
    reduced_elastic_kernels,
    reduced_elastic_sparse_kernels,
    rigid_sparse_articulation_variable_assembly,
    rigid_sparse_articulation_variable_kernels,
)
from newton.tests import _reduced_elastic_fixtures as elastic_fixtures
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

devices = get_test_devices()
cuda_devices = get_cuda_test_devices()

_SPARSE = "block_sparse_joints"


_BACKENDS = ("schur", "coupled")


def _solver(model, mode: str, iterations: int = 24, relaxation: float = 1.0, backend: str = "schur", **kwargs):
    """Build a VBD solver; ``backend`` is ``rigid_articulation_elastic_solve``."""
    return newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_articulation_solve=mode,
        rigid_articulation_relaxation=relaxation,
        rigid_articulation_elastic_solve=backend,
        **kwargs,
    )


def _run_standalone_rod(device, mode: str, steps: int = 192, dt: float = 1.0 / 960.0, backend: str = "schur"):
    """Step the spinning, boosted one-mode rod and record its joint state and momentum metrics."""
    model, state_0, data = elastic_fixtures.rod_model(
        device, mode="cos_offset", spin=(0.0, 0.0, 5.0), velocity=(0.3, -0.2, 0.1)
    )
    state_1 = model.state()
    control = model.control()
    solver = _solver(model, mode, backend=backend)
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


def _run_elastic_chain(
    device, mode: str, iterations: int = 24, steps: int = 240, dt: float = 1.0 / 480.0, backend: str = "schur"
):
    model = _make_elastic_chain_model(device)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    solver = _solver(model, mode, iterations=iterations, backend=backend)
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


def _run_fourbar(device, mode: str, iterations: int, steps: int = 200, dt: float = 1.0 / 120.0, backend: str = "schur"):
    model, _crank_coupler = _make_elastic_fourbar_model(device)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    solver = _solver(model, mode, iterations=iterations, relaxation=0.65, backend=backend)
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


def _make_mixed_mode_count_model(device, copies: int = 1):
    """Build a chain of a one-mode elastic link, a rigid link, and a three-mode elastic link.

    Both elastic links have fewer modes than the model's widest basis would allow if padded, so the
    sparse layout must handle per-body mode counts below ``elastic_max_mode_count``. With
    ``copies > 1`` the chain is replicated as independent articulations, offset along ``z``.
    """
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    for copy in range(copies):
        _add_mixed_mode_count_chain(builder, 0.4 * copy)
    builder.color()
    model = builder.finalize(device=device)
    return model, model.state()


def _add_mixed_mode_count_chain(builder, z: float):
    """Add the one-mode elastic, rigid, three-mode elastic chain of :func:`_make_mixed_mode_count_model`."""
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
    first = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(0.5 * length, 0.0, z), wp.quat_identity()),
        modal_basis=one_mode,
        mode_q=[0.01],
    )
    middle = builder.add_link(
        xform=wp.transform(wp.vec3(length + 0.1, 0.0, z), wp.quat_identity()),
        mass=0.4,
        inertia=wp.mat33(*np.diag([0.001, 0.004, 0.004]).ravel()),
    )
    last = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(1.5 * length + 0.2, 0.0, z), wp.quat_identity()),
        modal_basis=three_modes,
        mode_q=[0.0, 0.0, -0.01],
    )
    half = wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity())
    minus_half = wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity())
    axis = (0.0, 0.0, 1.0)
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=first,
            axis=axis,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()),
            child_xform=minus_half,
        ),
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


def _beam_modes(count: int):
    """Return ``count`` beam mode specifications: axial, then pinned-pinned bending of rising order."""
    beam = newton.ModalGeneratorBeam
    specs = [{"type": beam.Mode.AXIAL}]
    order = 1
    while len(specs) < count:
        for mode_type in (beam.Mode.BENDING_Z, beam.Mode.BENDING_Y):
            if len(specs) < count:
                specs.append({"type": mode_type, "boundary": beam.Boundary.PINNED_PINNED, "order": order})
        order += 1
    return specs


def _make_branched_model(device):
    """Build a rigid base hinged to the world carrying two elastic leaf links with 2 and 5 modes.

    The base row couples two elastic rows of different sizes (8 and 11) in one articulation.
    """
    length = 0.5
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    base = builder.add_link(
        xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(*np.diag([0.002, 0.02, 0.02]).ravel()),
    )
    leaves = [
        builder.add_link_elastic(
            xform=wp.transform(wp.vec3(0.4 + 0.5 * length, 0.0, 0.3 * k), wp.quat_identity()),
            modal_basis=_beam_basis(_beam_modes(count), length=length),
            mode_q=[0.0] * (count - 1) + [0.01],
        )
        for k, count in enumerate((2, 5))
    ]
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=base,
            axis=(0.0, 0.0, 1.0),
            child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
        )
    ]
    for k, leaf in enumerate(leaves):
        joints.append(
            builder.add_joint_revolute(
                parent=base,
                child=leaf,
                axis=(0.0, 0.0, 1.0),
                parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.3 * k), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[model.joint_qd_start.numpy()[joints[0]]] = 2.0
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


def _make_elastic_triple_chain_model(device):
    """Build three directly hinged elastic links with 2, 4 and 3 modes, hinged to the world."""
    length = 0.4
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    links = [
        builder.add_link_elastic(
            xform=wp.transform(wp.vec3((i + 0.5) * length, 0.0, 0.0), wp.quat_identity()),
            modal_basis=_beam_basis(_beam_modes(count), length=length),
            mode_q=[0.0] * (count - 1) + [0.006 * (1 - i)],
        )
        for i, count in enumerate((2, 4, 3))
    ]
    half = wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity())
    minus_half = wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity())
    joints = [builder.add_joint_revolute(parent=-1, child=links[0], axis=(0.0, 0.0, 1.0), child_xform=minus_half)]
    for a, b in ((0, 1), (1, 2)):
        joints.append(
            builder.add_joint_revolute(
                parent=links[a], child=links[b], axis=(0.0, 0.0, 1.0), parent_xform=half, child_xform=minus_half
            )
        )
    builder.add_articulation(joints)
    builder.color()
    model = builder.finalize(device=device)
    return model, model.state()


def _make_elastic_joint_type_model(device, joint_kind: str):
    """Build a rigid base hinged to the world carrying an elastic link through a ``joint_kind`` joint.

    ``joint_kind`` is ``"revolute"``, ``"prismatic"``, ``"d6"`` (one linear and one angular axis),
    ``"ball"`` or ``"fixed"``. Gravity and an initial base spin load every projected constraint row.
    """
    length = 0.5
    basis = _beam_basis(_beam_modes(3), length=length)
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    base = builder.add_link(
        xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(*np.diag([0.002, 0.02, 0.02]).ravel()),
    )
    link = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(0.4 + 0.5 * length, 0.0, 0.0), wp.quat_identity()),
        modal_basis=basis,
        mode_q=[0.0, 0.01, 0.0],
    )
    root = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=(0.0, 0.0, 1.0),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    attachment = {
        "parent": base,
        "child": link,
        "parent_xform": wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        "child_xform": wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity()),
    }
    if joint_kind == "revolute":
        joint = builder.add_joint_revolute(axis=(0.0, 0.0, 1.0), **attachment)
    elif joint_kind == "prismatic":
        joint = builder.add_joint_prismatic(axis=(1.0, 0.0, 0.0), **attachment)
    elif joint_kind == "d6":
        cfg = newton.ModelBuilder.JointDofConfig
        joint = builder.add_joint_d6(
            linear_axes=[cfg(axis=newton.Axis.X)], angular_axes=[cfg(axis=newton.Axis.Z)], **attachment
        )
    elif joint_kind == "ball":
        joint = builder.add_joint_ball(**attachment)
    elif joint_kind == "fixed":
        joint = builder.add_joint_fixed(**attachment)
    else:
        raise ValueError(joint_kind)
    builder.add_articulation([root, joint])
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[model.joint_qd_start.numpy()[root]] = 1.5
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


def _make_standalone_and_mixed_model(device):
    """Build a free spinning one-mode rod (``add_body_elastic``) next to the elastic leaf-link articulation."""
    points, mass, phi = elastic_fixtures.elastic_reference.rod_fixture("cos_offset")
    inertia = elastic_fixtures.elastic_reference.lumped_inertia(points, mass, np.zeros(3))
    basis = newton.ModalBasis(
        sample_points=points.astype(np.float32),
        sample_phi=phi,
        sample_mass=mass,
        mode_stiffness=[30.0],
        mode_damping=[0.0],
    )
    length = 0.5
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    rod = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
        mass=float(mass.sum()),
        inertia=wp.mat33(*inertia.ravel()),
        modal_basis=basis,
        mode_q=[0.02],
        mode_qd=[0.1],
    )
    base = builder.add_link(
        xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(*np.diag([0.002, 0.02, 0.02]).ravel()),
    )
    link = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(0.4 + 0.5 * length, 0.0, 0.0), wp.quat_identity()),
        modal_basis=_beam_basis(_beam_modes(3), length=length),
        mode_q=[0.0, 0.01, 0.0],
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
    q_start, qd_start = elastic_fixtures.elastic_free_joint_starts(model, rod)
    joint_qd = state.joint_qd.numpy()
    joint_qd[qd_start : qd_start + 6] = [0.3, -0.2, 0.1, 0.0, 0.0, 5.0]
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state, rod, q_start


def _with_state(model):
    return model, model.state()


_TOPOLOGIES = {
    # name: function of the device returning (model, initial state)
    "leaf_link": lambda device: _make_leaf_link_model(device)[:2],
    "elastic_chain": lambda device: _with_state(_make_elastic_chain_model(device)),
    "mixed_mode_counts": _make_mixed_mode_count_model,
    "coupler_rocker_loop": lambda device: _with_state(_make_elastic_fourbar_model(device)[0]),
    "kinematic_elastic_parent": _make_kinematic_elastic_parent_model,
    "branched_heterogeneous": _make_branched_model,
    "elastic_triple_chain": _make_elastic_triple_chain_model,
}
"""Seven mixed articulation topologies: leaf, adjacent elastic links, heterogeneous mode counts around a
rigid link, an elastic coupler + rocker loop, a kinematic elastic parent, a branch with 8- and 11-wide
elastic rows, and three directly hinged elastic links with 2, 4 and 3 modes."""


def _series(model, state_0, solver, steps: int, dt: float = 1.0 / 480.0, control=None) -> np.ndarray:
    """Step ``solver`` and return the per-step ``[body_q, joint_q]`` series."""
    state_1 = model.state()
    control = control if control is not None else model.control()
    series = []
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        series.append(np.concatenate((state_0.body_q.numpy().ravel(), state_0.joint_q.numpy())))
    return np.asarray(series)


def _topology_series(device, name: str, mode: str, iterations: int, steps: int, **kwargs) -> tuple:
    model, state = _TOPOLOGIES[name](device)
    relaxation = 0.65 if name == "coupler_rocker_loop" else 1.0
    solver = _solver(model, mode, iterations=iterations, relaxation=relaxation, **kwargs)
    dt = 1.0 / 120.0 if name == "coupler_rocker_loop" else 1.0 / 480.0
    return _series(model, state, solver, steps, dt), solver


def test_sparse_standalone_elastic_body_matches_local(test, device, backend):
    """Match a free spinning, boosted elastic body between the sparse and local solves.

    Under ``"schur"`` the body's one-joint articulation puts it in the sparse solve, where its modes
    are eliminated onto the frame block. Per step, both solves converge to the same state to float32
    precision; over 0.2 s the trajectories differ only by the accumulated float32 rounding of each
    update form (about 1e-5 here, the same as local CPU versus local CUDA), so the sparse solve must
    keep the ballistic COM and the angular momentum at least as well as the local solve. Under
    ``"coupled"`` an articulation whose only body is elastic is left to the local solve, so the model
    has no sparse layout and the trajectory equals the local one.
    """
    local, local_ballistic, local_angular, _ = _run_standalone_rod(device, "local")
    sparse, sparse_ballistic, sparse_angular, solver = _run_standalone_rod(device, _SPARSE, backend=backend)
    layout = solver.rigid_articulation_sparse_layout
    if backend == "schur":
        test.assertEqual(layout.elastic_body_count, 1)
        test.assertEqual(layout.local_body_count, 0)
        np.testing.assert_allclose(sparse, local, atol=1.0e-4, rtol=0.0)
    else:
        test.assertIsNone(layout)
        np.testing.assert_array_equal(sparse, local)
    test.assertGreater(float(np.max(np.abs(sparse[-1, 7] - sparse[0, 7]))), 1.0e-3)
    test.assertLess(sparse_ballistic, 1.1 * local_ballistic + 1.0e-6)
    test.assertLess(sparse_angular, 1.1 * local_angular + 1.0e-6)


def _assert_backend_layout(test, layout, backend: str, elastic_count: int):
    """Check that ``elastic_count`` elastic bodies are solved in the sparse layout by ``backend``."""
    if backend == "schur":
        test.assertEqual(layout.elastic_body_count, elastic_count)
        test.assertEqual(layout.mixed_articulation_count, 0)
    else:
        test.assertEqual(layout.elastic_body_count, 0)
        test.assertEqual(layout.mixed_elastic_body_count, elastic_count)
        test.assertGreater(layout.mixed_articulation_count, 0)


def test_sparse_elastic_leaf_link_matches_local(test, device, backend):
    """Match a rigid base carrying an elastic leaf link between the sparse and local solves.

    Also check that the sparse solve holds the hinge together at the deformed attachment: with
    stiff joints, the hinge gap measured with the elastic endpoint displaced by ``Phi q`` is far
    smaller than the gap at the undeformed attachment.
    """
    local, *_ = _run_leaf_link(device, "local")
    sparse, _model, _state, _hinge, solver = _run_leaf_link(device, _SPARSE, backend=backend)
    _assert_backend_layout(test, solver.rigid_articulation_sparse_layout, backend, 1)
    test.assertTrue(np.isfinite(sparse).all())
    np.testing.assert_allclose(sparse, local, atol=2.0e-4, rtol=0.0)
    test.assertGreater(float(np.max(np.abs(sparse[-1] - sparse[0]))), 0.5)

    _stiff, model, state, hinge, _solver_stiff = _run_leaf_link(
        device, _SPARSE, rigid_joint_linear_ke=1.0e6, rigid_joint_angular_ke=1.0e6, backend=backend
    )
    deformed_gap = _joint_gap(model, state, hinge, deformed=True)
    rigid_gap = _joint_gap(model, state, hinge, deformed=False)
    test.assertGreater(rigid_gap, 5.0e-4)
    test.assertLess(deformed_gap, 0.1 * rigid_gap)


def test_sparse_elastic_coupler_closed_loop(test, device, backend):
    """Close a four-bar with an elastic coupler in one sparse factorization.

    Under ``"schur"`` the coupler's modes are eliminated onto the crank and rocker frames, which the
    layout couples through a fill block; under ``"coupled"`` the coupler is one 9-wide row of the
    loop's factorization. At 24 iterations the sparse trajectory matches the converged local solve
    (384 iterations) within 5e-4 in every joint coordinate (about 2e-4 measured for both backends),
    while the local solve at 24 iterations is still about 0.12 rad away, and its loop residual stays
    within 5 % of the converged one.
    """
    reference_gaps, reference_series, _ = _run_fourbar(device, "local", iterations=384)
    sparse_gaps, sparse_series, solver = _run_fourbar(device, _SPARSE, iterations=24, backend=backend)
    layout = solver.rigid_articulation_sparse_layout
    _assert_backend_layout(test, layout, backend, 1)
    if backend == "schur":
        test.assertEqual(layout.elastic_group_count, 3)
    test.assertTrue(np.isfinite(sparse_series).all())
    np.testing.assert_allclose(sparse_series, reference_series, rtol=0.0, atol=5.0e-4)
    test.assertLess(float(sparse_gaps.max()), 1.05 * float(reference_gaps.max()))
    test.assertLess(float(sparse_gaps[-1]), 1.0e-4)
    test.assertGreater(float(np.max(np.abs(sparse_series[-1] - sparse_series[0]))), 0.05)


def test_sparse_elastic_elastic_joint_stays_stable(test, device):
    """Keep a hinge between two elastic links stable in the Schur sparse solve.

    The per-body elimination cannot represent the mode-mode coupling of the shared hinge; it adds
    the hinge's modal self term once more instead. Dropping the coupling alone makes the reduced
    system indefinite and diverges. With the conservative term, compared with the converged local
    solve (384 iterations), the peak hinge gap stays within 30 % (about 20 % measured), the final gap
    within 5 % (0.2 % measured), and the joint coordinates within 1e-2 (4e-3 measured, the effect of
    the extra modal term). The coupled backend's exact coupling is checked by
    :func:`test_coupled_adjacent_elastic_accuracy`.
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


def test_sparse_kinematic_elastic_parent_matches_local(test, device, backend):
    """Match a kinematic elastic beam carrying a hinged rigid child between the sparse and local solves.

    The beam's frame motion is prescribed, so the sparse solve updates only its modes. At 128
    iterations both solves agree within 2e-4 in every joint coordinate (about 8e-5 measured, the
    local solve's own distance to 384 iterations is 2e-5), and the child's hinge load moves the modes.
    """
    model, state = _make_kinematic_elastic_parent_model(device)
    local, _ = _run_model(model, state, "local", iterations=128, steps=240)
    model, state = _make_kinematic_elastic_parent_model(device)
    sparse, solver = _run_model(model, state, _SPARSE, iterations=128, steps=240, backend=backend)
    _assert_backend_layout(test, solver.rigid_articulation_sparse_layout, backend, 1)
    test.assertTrue(np.isfinite(sparse).all())
    np.testing.assert_allclose(sparse, local, rtol=0.0, atol=2.0e-4)
    modes = _elastic_mode_series(model, sparse, 0)
    test.assertGreater(float(np.ptp(modes[:, 0])), 1.0e-4)


def test_sparse_mixed_mode_counts_match_local(test, device, backend):
    """Match a chain of one-mode and three-mode elastic links around a rigid link to the local solve.

    At 128 iterations both solves agree within 2e-4 in every joint coordinate (about 3e-5 measured).
    """
    model, state = _make_mixed_mode_count_model(device)
    test.assertEqual(model.elastic_mode_count.numpy().tolist(), [1, 3])
    local, _ = _run_model(model, state, "local", iterations=128, steps=240)
    model, state = _make_mixed_mode_count_model(device)
    sparse, solver = _run_model(model, state, _SPARSE, iterations=128, steps=240, backend=backend)
    layout = solver.rigid_articulation_sparse_layout
    _assert_backend_layout(test, layout, backend, 2)
    if backend == "coupled":
        test.assertEqual(sorted(layout.articulation_row_dims.numpy().tolist()), [6, 7, 9])
    test.assertTrue(np.isfinite(sparse).all())
    np.testing.assert_allclose(sparse, local, rtol=0.0, atol=2.0e-4)
    for elastic_index in range(2):
        modes = _elastic_mode_series(model, sparse, elastic_index)
        test.assertGreater(float(np.max(np.ptp(modes, axis=0))), 1.0e-4)


def _assert_cpu_cuda_parity(test, device, run, label: str):
    cpu = run("cpu")
    cuda = run(device)
    np.testing.assert_allclose(cuda, cpu, rtol=2.0e-4, atol=2.0e-4, err_msg=label)


def test_sparse_elastic_cpu_cuda_parity(test, device, backend):
    """Match the CUDA sparse elastic solve to the CPU serial solve."""
    _assert_cpu_cuda_parity(
        test, device, lambda d: _run_standalone_rod(d, _SPARSE, backend=backend)[0], "standalone rod"
    )
    _assert_cpu_cuda_parity(test, device, lambda d: _run_leaf_link(d, _SPARSE, backend=backend)[0], "leaf link")


def _assert_graph_replay_matches(test, device, setup, steps_per_advance: int = 2, replays: int = 4):
    """Replay ``steps_per_advance`` captured steps and compare with the same steps run uncaptured."""
    dt = 1.0 / 480.0

    def advance(system):
        _model, solver, state_0, state_1, control = system
        for _ in range(steps_per_advance // 2):
            solver.step(state_0, state_1, control, None, dt)
            solver.step(state_1, state_0, control, None, dt)

    captured = setup()
    reference = setup()
    advance(captured)
    advance(reference)
    with wp.ScopedCapture(device=device) as capture:
        advance(captured)
    for _ in range(replays):
        wp.capture_launch(capture.graph)
        advance(reference)
    for attr in ("body_q", "body_qd", "joint_q", "joint_qd"):
        np.testing.assert_allclose(
            getattr(captured[2], attr).numpy(), getattr(reference[2], attr).numpy(), atol=1.0e-5, rtol=1.0e-5
        )
    return captured


def test_sparse_elastic_leaf_link_graph_capture(test, device, backend):
    """Replay the sparse elastic leaf-link step from a captured CUDA graph."""

    def setup():
        model, state_0, _hinge = _make_leaf_link_model(device)
        solver = _solver(model, _SPARSE, relaxation=0.65, backend=backend)
        return model, solver, state_0, model.state(), model.control()

    _assert_graph_replay_matches(test, device, setup)


def test_sparse_elastic_rejects_unsupported_joint_terms(test, device, backend):
    """Reject elastic-attached joint armature, damping, and friction in the sparse solve.

    The message names the joint, the attribute and the backend. The local solve keeps each of them.
    """
    for attribute in ("joint_armature", "joint_damping", "joint_friction"):
        with test.subTest(attribute=attribute):
            model, _state, hinge = _make_leaf_link_model(device)
            values = getattr(model, attribute).numpy()
            values[model.joint_qd_start.numpy()[hinge]] = 0.01
            getattr(model, attribute).assign(values)
            with test.assertRaisesRegex(
                ValueError,
                rf"Joint {hinge} .* nonzero {attribute}\. rigid_articulation_elastic_solve='{backend}' does not "
                r"support joint_armature, joint_damping, or joint_friction",
            ):
                _solver(model, _SPARSE, backend=backend)
            _solver(model, "local", backend=backend)


def test_sparse_elastic_rejects_joint_mimics(test, device, backend):
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
    with test.assertRaisesRegex(NotImplementedError, rf"rigid_articulation_elastic_solve='{backend}' .*joint mimics"):
        _solver(model, _SPARSE, backend=backend)
    _solver(model, "local", backend=backend)


def test_elastic_articulation_honors_deterministic_option(test, device, backend):
    """Forward the requested deterministic mode to the reduced elastic kernel modules.

    The elastic block assembly, the frame coupling, the sparse Schur scatter and the coupled
    variable-size assembly accumulate with atomics, so their modules must receive the solver's
    deterministic mode in both the local and the block-sparse articulation solve.
    """
    modules = (
        reduced_elastic_kernels,
        reduced_elastic_sparse_kernels,
        rigid_sparse_articulation_variable_assembly,
        rigid_sparse_articulation_variable_kernels,
    )
    original = [wp.get_module_options(module=module) for module in modules]
    try:
        for mode in ("local", _SPARSE):
            with test.subTest(mode=mode):
                for module, options in zip(modules, original, strict=True):
                    wp.set_module_options(options, module=module)
                _solver(
                    _make_elastic_chain_model(device),
                    mode,
                    backend=backend,
                    deterministic=wp.DeterministicMode.RUN_TO_RUN,
                )
                for module in modules:
                    options = wp.get_module_options(module=module)
                    test.assertEqual(options["deterministic"], wp.DeterministicMode.RUN_TO_RUN)
    finally:
        for module, options in zip(modules, original, strict=True):
            wp.set_module_options(options, module=module)


# ---------------------------------------------------------------------------------------------
# Coupled variable-block backend
# ---------------------------------------------------------------------------------------------

_COUPLED = {"backend": "coupled"}


def test_coupled_topologies_match_converged_local(test, device):
    """Match seven mixed articulation topologies to the converged local solve at 24 iterations.

    Over 40 steps every topology stays within 7e-5 of the local solve at 384 iterations in every body
    pose and joint coordinate (at most 2.8e-5 measured over 60 steps, the three-link elastic chain,
    where the local reference itself is not fully converged; 2.6e-5 for the four-bar and 7e-6 or less
    for the others),
    every coupled factorization succeeds, and the real solve metrics of every mixed elastic body
    report a linear residual below 1e-4 of the initial one.
    """
    for name in _TOPOLOGIES:
        with test.subTest(topology=name):
            reference, _ = _topology_series(device, name, "local", 384, 40)
            series, solver = _topology_series(device, name, _SPARSE, 24, 40, elastic_solver_metrics=True, **_COUPLED)
            layout = solver.rigid_articulation_sparse_layout
            test.assertGreater(layout.mixed_articulation_count, 0)
            test.assertTrue(np.isfinite(series).all())
            status = solver.rigid_articulation_elastic_solve_status()
            test.assertEqual(status[status >= 0].tolist(), [0] * layout.mixed_articulation_count)
            np.testing.assert_allclose(series, reference, rtol=0.0, atol=7.0e-5)
            test.assertGreater(float(np.max(np.abs(reference[-1] - reference[0]))), 1.0e-2)
            metrics = solver.elastic_mode_solve_metrics()
            elastic = solver.model.body_elastic_index.numpy()[layout.mixed_elastic_bodies.numpy()]
            initial = metrics["initial_residual_norm"][elastic]
            test.assertTrue(np.all(initial > 0.0))
            test.assertTrue(np.all(metrics["solve_residual_norm"][elastic] < 1.0e-4 * initial))


def _adjacent_chain_error(device, iterations: int, backend: str) -> float:
    """Return the max ``[body_q, joint_q]`` error of the two-link elastic chain against local at 384 iterations."""
    reference = _ADJACENT_REFERENCE.get(str(device))
    if reference is None:
        model = _make_elastic_chain_model(device)
        reference = _series(model, model.state(), _solver(model, "local", iterations=384), 120)
        _ADJACENT_REFERENCE[str(device)] = reference
    model = _make_elastic_chain_model(device)
    series = _series(model, model.state(), _solver(model, _SPARSE, iterations=iterations, backend=backend), 120)
    if not np.isfinite(series).all():
        return math.inf
    return float(np.max(np.abs(series - reference)))


_ADJACENT_REFERENCE: dict[str, np.ndarray] = {}
# About 2.5x the largest measured error (CPU 7.0e-6 / 8.5e-6, CUDA 4.6e-6 / 5.3e-6 at 24 / 96 iterations),
# which is the float32 floor of the comparison: the local reference at 96 iterations is 4.8e-6 from 384.
_ADJACENT_BOUND_24 = 1.8e-5
_ADJACENT_BOUND_96 = 2.2e-5


def test_coupled_adjacent_elastic_accuracy(test, device):
    """Converge two directly hinged elastic links to the local fixed point within a few iterations.

    The coupled backend factors the hinge's mode-mode cross block exactly. Over 0.25 s it is within
    the float32 floor of the converged local reference (384 iterations) already at 24 iterations:
    at most 7.0e-6 measured at 24 and 8.5e-6 at 96 iterations (bounds 1.8e-5 and 2.2e-5), while the
    Schur backend, which replaces that block by a conservative per-body term, is 1.0e-2 away at 24
    and 8.5e-4 at 96 iterations. At 24 iterations the coupled error must be below a tenth of the
    Schur error.
    """
    coupled_24 = _adjacent_chain_error(device, 24, "coupled")
    coupled_96 = _adjacent_chain_error(device, 96, "coupled")
    schur_24 = _adjacent_chain_error(device, 24, "schur")
    test.assertLess(coupled_24, _ADJACENT_BOUND_24)
    test.assertLess(coupled_96, _ADJACENT_BOUND_96)
    test.assertLess(coupled_24, 0.1 * schur_24)


@wp.kernel
def _zero_mode_mode_entries(
    slots: wp.array[wp.int32],
    block_value_offsets: wp.array[wp.int32],
    block_dims: wp.array2d[wp.int32],
    values: wp.array[float],
):
    entry, r, c = wp.tid()
    slot = slots[entry]
    rows = block_dims[slot, 0]
    cols = block_dims[slot, 1]
    if r >= 6 and c >= 6 and r < rows and c < cols:
        values[block_value_offsets[slot] + r * cols + c] = 0.0


def test_coupled_mode_mode_cross_block_is_required(test, device):
    """Fail the adjacent-elastic accuracy bound when the mode-mode cross block is dropped.

    A mutation of the assembly zeroes the modal-modal part of every off-diagonal block after
    assembly (before factorization the off-diagonal blocks hold only joint cross blocks, so this
    removes exactly ``G_p^T H_pc G_c``). The two-link elastic chain then misses the bound of
    :func:`test_coupled_adjacent_elastic_accuracy` at 24 iterations: without the block the system is
    indefinite, every factorization fails, and the diagonal fallback steps diverge.
    """
    assembly = rigid_sparse_articulation_variable_assembly
    original = assembly.assemble_articulation_variable_system

    def mutated(layout, model, **kwargs):
        result = original(layout, model, **kwargs)
        rows = layout.articulation_block_row_offsets.numpy()
        diag = set(layout.articulation_diag_slots.numpy().tolist())
        dims = layout.articulation_block_dims.numpy()
        slots = [
            s
            for s in range(rows[layout.rigid_body_row_count], layout.block_count)
            if s not in diag and dims[s, 0] > 6 and dims[s, 1] > 6
        ]
        test.assertGreater(len(slots), 0)
        wp.launch(
            _zero_mode_mode_entries,
            dim=(len(slots), layout.articulation_max_row_dim, layout.articulation_max_row_dim),
            inputs=[
                wp.array(slots, dtype=wp.int32, device=device),
                layout.articulation_block_value_offsets,
                layout.articulation_block_dims,
            ],
            outputs=[kwargs["values"]],
            device=device,
        )
        return result

    with mock.patch.object(assembly, "assemble_articulation_variable_system", mutated), warnings.catch_warnings():
        # The mutated system can be indefinite; its failure warnings are expected here.
        warnings.simplefilter("ignore")
        error = _adjacent_chain_error(device, 24, "coupled")
    test.assertGreater(error, _ADJACENT_BOUND_24)


def _capture_variable_delta_application(solver, model, state_0, dt: float) -> dict[str, np.ndarray]:
    """Step once and return the inputs and outputs of the coupled update kernel's last launch."""
    kernels = rigid_sparse_articulation_variable_kernels
    snapshots = {}
    launch = wp.launch

    def spy(*args, **kwargs):
        kernel = kwargs.get("kernel", args[0] if args else None)
        if kernel is not kernels.apply_articulation_variable_delta:
            return launch(*args, **kwargs)
        inputs, outputs = kwargs["inputs"], kwargs["outputs"]
        snapshots["delta"] = inputs[6].numpy().astype(np.float64)
        snapshots["body_q"] = inputs[7].numpy().astype(np.float64)
        snapshots["joint_q_prev"] = inputs[16].numpy().astype(np.float64)
        snapshots["joint_q"] = outputs[3].numpy().astype(np.float64)
        result = launch(*args, **kwargs)
        snapshots["body_q_new"] = outputs[2].numpy().astype(np.float64)
        snapshots["joint_q_new"] = outputs[3].numpy().astype(np.float64)
        snapshots["joint_qd_new"] = outputs[4].numpy().astype(np.float64)
        return result

    with mock.patch.object(wp, "launch", spy):
        solver.step(state_0, model.state(), model.control(), None, dt)
    return snapshots


def test_coupled_relaxation_applies_update_once(test, device):
    """Apply the coupled frame and modal update scaled once by the relaxation.

    For relaxations 0.65 and 1, one iteration's update of every row of the mixed articulation
    (rigid and elastic frames, modal coordinates) is compared with NumPy: the COM moves by
    ``relax * dx``, the rotation by ``exp(relax * dtheta)``, the modes by ``relax * dq`` and the
    modal velocity is ``(q - q_prev) / dt``.
    """
    dt = 1.0 / 480.0
    for relaxation in (0.65, 1.0):
        with test.subTest(relaxation=relaxation):
            model, state_0 = _make_mixed_mode_count_model(device)
            solver = _solver(model, _SPARSE, iterations=1, relaxation=relaxation, **_COUPLED)
            layout = solver.rigid_articulation_sparse_layout
            snapshots = _capture_variable_delta_application(solver, model, state_0, dt)
            com = model.body_com.numpy().astype(np.float64)
            offsets = layout.articulation_row_scalar_offsets.numpy()
            dims = layout.articulation_row_dims.numpy()
            bodies = layout.articulation_bodies.numpy()
            elastic_index = model.body_elastic_index.numpy()
            q_start = model.joint_q_start.numpy()
            qd_start = model.joint_qd_start.numpy()
            owners = model.elastic_joint.numpy()
            moved_modes = 0.0
            for row in range(layout.rigid_body_row_count, layout.articulation_body_count):
                body = int(bodies[row])
                d = snapshots["delta"][offsets[row] : offsets[row] + dims[row]]
                old, new = snapshots["body_q"][body], snapshots["body_q_new"][body]
                c_old = old[:3] + _quat_rotate(old[3:7], com[body])
                c_new = new[:3] + _quat_rotate(new[3:7], com[body])
                np.testing.assert_allclose(c_new - c_old, relaxation * d[:3], rtol=0.0, atol=2.0e-6)
                expected = _quat_multiply(_quat_from_rotation_vector(relaxation * d[3:6]), old[3:7])
                test.assertGreater(abs(float(np.dot(expected, new[3:7]))), 1.0 - 1.0e-6)
                if dims[row] > 6:
                    start = int(q_start[owners[elastic_index[body]]]) + 7
                    vstart = int(qd_start[owners[elastic_index[body]]]) + 6
                    m = dims[row] - 6
                    q_old = snapshots["joint_q"][start : start + m]
                    q_new = snapshots["joint_q_new"][start : start + m]
                    np.testing.assert_allclose(q_new - q_old, relaxation * d[6:], rtol=1.0e-5, atol=1.0e-9)
                    q_prev = snapshots["joint_q_prev"][start : start + m]
                    np.testing.assert_allclose(
                        snapshots["joint_qd_new"][vstart : vstart + m], (q_new - q_prev) / dt, rtol=1.0e-4, atol=1.0e-6
                    )
                    moved_modes = max(moved_modes, float(np.max(np.abs(d[6:]))))
            test.assertGreater(moved_modes, 1.0e-6)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions."""
    av, aw, bv, bw = a[:3], a[3], b[:3], b[3]
    return np.concatenate((aw * bv + bw * av + np.cross(av, bv), [aw * bw - float(np.dot(av, bv))]))


def _quat_from_rotation_vector(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1.0e-12:
        return np.array([0.5 * v[0], 0.5 * v[1], 0.5 * v[2], 1.0]) / np.sqrt(1.0 + 0.25 * angle * angle)
    axis = v / angle
    return np.concatenate((np.sin(0.5 * angle) * axis, [np.cos(0.5 * angle)]))


def test_coupled_cpu_cuda_parity(test, device):
    """Match the coupled CUDA solve to the CPU solve after one step and over 100 steps.

    One step agrees within 1e-5 and the 100-step ``[body_q, joint_q]`` trajectories within 2e-4 for
    adjacent elastic links, heterogeneous mode counts and the elastic coupler loop.
    """
    for name in ("elastic_chain", "mixed_mode_counts", "coupler_rocker_loop", "elastic_triple_chain"):
        with test.subTest(topology=name):
            cpu_one, cuda_one = (_topology_series(d, name, _SPARSE, 24, 1, **_COUPLED)[0] for d in ("cpu", device))
            np.testing.assert_allclose(cuda_one, cpu_one, rtol=0.0, atol=1.0e-5)
            cpu_many, cuda_many = (_topology_series(d, name, _SPARSE, 24, 100, **_COUPLED)[0] for d in ("cpu", device))
            np.testing.assert_allclose(cuda_many, cpu_many, rtol=0.0, atol=2.0e-4)


def test_coupled_graph_capture_replicated_articulations(test, device):
    """Replay coupled steps of three replicated mixed articulations from a captured CUDA graph."""

    def setup():
        model, state_0 = _make_mixed_mode_count_model(device, copies=3)
        solver = _solver(model, _SPARSE, relaxation=0.65, elastic_solver_metrics=True, **_COUPLED)
        return model, solver, state_0, model.state(), model.control()

    captured = _assert_graph_replay_matches(test, device, setup)
    solver = captured[1]
    test.assertEqual(solver.rigid_articulation_sparse_layout.mixed_articulation_count, 3)
    test.assertEqual(solver.rigid_articulation_elastic_solve_status().tolist(), [0, 0, 0])
    test.assertTrue(all(np.isfinite(v).all() for v in solver.elastic_mode_solve_metrics().values()))


def test_coupled_run_to_run_deterministic(test, device):
    """Reproduce coupled trajectories bit for bit under ``DeterministicMode.RUN_TO_RUN``.

    The middle link of the three-link elastic chain receives projected terms from two joints, which
    the assembly accumulates with atomics.
    """
    runs = []
    for _ in range(2):
        series, _solver_run = _topology_series(
            device, "elastic_triple_chain", _SPARSE, 12, 20, deterministic=wp.DeterministicMode.RUN_TO_RUN, **_COUPLED
        )
        runs.append(series)
    np.testing.assert_array_equal(runs[0], runs[1])


def _make_indefinite_leaf_link_model(device):
    """Return the leaf-link model with a strongly negative stiffness on the axial mode, and its state.

    The axial mode's diagonal ``m / dt^2 + k`` is then negative, so the coupled articulation system
    has no Cholesky factor.
    """
    model, state, _hinge = _make_leaf_link_model(device)
    stiffness = model.elastic_mode_stiffness.numpy()
    stiffness[0] = -1.0e7
    model.elastic_mode_stiffness.assign(stiffness)
    return model, state


def test_coupled_failure_reports_status_and_warns(test, device):
    """Report an indefinite coupled articulation system once, without dropping it silently.

    The articulation's status is set, the solver warns once (naming the backend, the status and
    count accessors and ``rigid_articulation_diagonal_regularization``) although every step fails,
    the failure count keeps growing, the state stays finite, and the metrics of the failed elastic
    body are NaN, as they are before the first step.
    """
    model, state_0 = _make_indefinite_leaf_link_model(device)
    solver = _solver(model, _SPARSE, iterations=4, elastic_solver_metrics=True, **_COUPLED)
    test.assertEqual(solver.rigid_articulation_elastic_solve_failure_count(), 0)
    # Before the first step no body has been evaluated.
    for name, values in solver.elastic_mode_solve_metrics().items():
        test.assertTrue(np.isnan(values).all(), name)
    state_1 = model.state()
    control = model.control()
    counts = []
    reported = []

    def record(call, *args):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call(*args)
        messages = [str(w.message) for w in caught if "coupled articulation solve" in str(w.message)]
        counts.append(len(messages))
        reported.extend(messages)

    for _ in range(3):
        record(solver.step, state_0, state_1, control, None, 1.0 / 480.0)
        state_0, state_1 = state_1, state_0
    # Failures are mirrored to the host at the end of a step without synchronizing and reported by
    # the next call; on the CPU the mirror is immediate, so the second step warns.
    if device.is_cpu:
        test.assertEqual(counts, [0, 1, 0])
    wp.synchronize_device(device)
    record(solver.notify_model_changed, newton.ModelFlags.SHAPE_PROPERTIES)
    for _ in range(2):
        record(solver.step, state_0, state_1, control, None, 1.0 / 480.0)
        state_0, state_1 = state_1, state_0
    wp.synchronize_device(device)
    record(solver.notify_model_changed, newton.ModelFlags.SHAPE_PROPERTIES)
    test.assertEqual(sum(counts), 1)
    message = reported[0]
    for name in (
        "rigid_articulation_elastic_solve='coupled'",
        "rigid_articulation_elastic_solve_status()",
        "rigid_articulation_elastic_solve_failure_count()",
        "rigid_articulation_diagonal_regularization",
    ):
        test.assertIn(name, message)
    test.assertNotRegex(message, r"\d")
    # One failure per iteration: 5 steps of 4 iterations.
    test.assertEqual(solver.rigid_articulation_elastic_solve_failure_count(), 20)
    test.assertEqual(solver.rigid_articulation_elastic_solve_status().tolist(), [1])
    test.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
    test.assertTrue(np.isfinite(state_0.joint_q.numpy()).all())
    for name, values in solver.elastic_mode_solve_metrics().items():
        test.assertTrue(np.isnan(values).all(), name)


def _modal_energy_series(model, series: np.ndarray, qd_series: np.ndarray) -> np.ndarray:
    """Return the modal strain plus kinetic energy ``sum(k q^2 + m qd^2) / 2`` per step [J]."""
    energy = np.zeros(series.shape[0])
    stiffness = model.elastic_mode_stiffness.numpy().astype(np.float64)
    mass = model.elastic_mode_mass.numpy().astype(np.float64)
    starts = model.elastic_mode_start.numpy()
    for elastic in range(model.elastic_body_count):
        owner = int(model.elastic_joint.numpy()[elastic])
        count = int(model.elastic_mode_count.numpy()[elastic])
        q0 = int(model.joint_q_start.numpy()[owner]) + 7
        qd0 = int(model.joint_qd_start.numpy()[owner]) + 6
        k = stiffness[starts[elastic] : starts[elastic] + count]
        m = mass[starts[elastic] : starts[elastic] + count]
        q = series[:, q0 : q0 + count]
        qd = qd_series[:, qd0 : qd0 + count]
        energy += 0.5 * (q * q) @ k + 0.5 * (qd * qd) @ m
    return energy


def _ringing_series(device, mode: str, iterations: int, backend: str = "schur"):
    model = _make_elastic_chain_model(device)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    solver = _solver(model, mode, iterations=iterations, backend=backend)
    q_series, qd_series = [], []
    for _ in range(240):
        solver.step(state_0, state_1, control, None, 1.0 / 480.0)
        state_0, state_1 = state_1, state_0
        q_series.append(state_0.joint_q.numpy().astype(np.float64))
        qd_series.append(state_0.joint_qd.numpy().astype(np.float64))
    q, qd = np.asarray(q_series), np.asarray(qd_series)
    modes = np.concatenate([_elastic_mode_series(model, q, e) for e in range(model.elastic_body_count)], axis=1)
    return modes, _modal_energy_series(model, q, qd)


def test_coupled_ringing_matches_converged_local(test, device):
    """Keep the modal ringing of adjacent elastic links equal to the converged local solve.

    Over 0.5 s at 24 iterations the coupled backend's largest modal second difference
    ``|q[n+1] - 2 q[n] + q[n-1]|`` is within 1 % of the local solve at 384 iterations, and the
    modal energy envelope (maximum over 24-step windows) within 1 % in every window, so the coupled
    update neither adds nor removes modal ringing.
    """
    reference_modes, reference_energy = _ringing_series(device, "local", 384)
    modes, energy = _ringing_series(device, _SPARSE, 24, "coupled")
    second = np.abs(np.diff(modes, n=2, axis=0)).max(axis=0)
    reference_second = np.abs(np.diff(reference_modes, n=2, axis=0)).max(axis=0)
    test.assertGreater(float(reference_second.max()), 0.0)
    np.testing.assert_allclose(second, reference_second, rtol=1.0e-2, atol=1.0e-2 * float(reference_second.max()))
    windows = len(energy) // 24
    envelope = energy[: 24 * windows].reshape(windows, 24).max(axis=1)
    reference_envelope = reference_energy[: 24 * windows].reshape(windows, 24).max(axis=1)
    test.assertGreater(float(reference_envelope.min()), 0.0)
    np.testing.assert_allclose(envelope, reference_envelope, rtol=1.0e-2)


def test_coupled_projected_joint_types_on_elastic_link(test, device):
    """Solve revolute, prismatic, D6, ball and fixed joints on an elastic link with the coupled backend.

    Each joint's rows are evaluated once and projected onto the elastic row through its modal
    attachment twists. At 48 iterations the 40-step ``[body_q, joint_q]`` trajectory is within 1e-4
    of the converged local solve (384 iterations), the factorization succeeds and the modes move.
    """
    for kind in ("revolute", "prismatic", "d6", "ball", "fixed"):
        with test.subTest(joint=kind):
            model, state = _make_elastic_joint_type_model(device, kind)
            reference = _series(model, state, _solver(model, "local", iterations=384), 40)
            model, state = _make_elastic_joint_type_model(device, kind)
            solver = _solver(model, _SPARSE, iterations=48, **_COUPLED)
            series = _series(model, state, solver, 40)
            test.assertTrue(np.isfinite(series).all())
            test.assertEqual(solver.rigid_articulation_elastic_solve_status().tolist(), [0])
            np.testing.assert_allclose(series, reference, rtol=0.0, atol=1.0e-4)
            modes = _elastic_mode_series(model, series[:, -model.joint_coord_count :], 0)
            test.assertGreater(float(np.max(np.ptp(modes, axis=0))), 1.0e-5)


def test_coupled_standalone_elastic_body_uses_local_solve(test, device):
    """Leave an articulation whose only body is elastic to the local block solve under ``"coupled"``.

    The free rod is listed in ``standalone_elastic_bodies``, is not a sparse row, joins the local
    color groups, follows ``elastic_body_relaxation`` (not ``rigid_articulation_relaxation``), and its
    trajectory equals that of the all-local solve, while the leaf-link articulation is solved by the
    coupled backend.
    """
    runs = {}
    for label, mode, elastic_relaxation in (
        ("coupled", _SPARSE, 0.5),
        ("local", "local", 0.5),
        ("coupled_unrelaxed", _SPARSE, 1.0),
    ):
        model, state, rod, q_start = _make_standalone_and_mixed_model(device)
        solver = _solver(model, mode, relaxation=1.0, elastic_body_relaxation=elastic_relaxation, **_COUPLED)
        series = _series(model, state, solver, 60)
        runs[label] = series[:, model.body_count * 7 + q_start : model.body_count * 7 + q_start + 8]
        if label == "coupled":
            layout = solver.rigid_articulation_sparse_layout
            test.assertEqual(layout.standalone_elastic_bodies.numpy().tolist(), [rod])
            test.assertNotIn(rod, layout.articulation_bodies.numpy().tolist())
            local_bodies = np.concatenate([group.numpy() for group in layout.local_body_color_groups])
            test.assertIn(rod, local_bodies.tolist())
            test.assertEqual(layout.mixed_articulation_count, 1)
            status = solver.rigid_articulation_elastic_solve_status()
            test.assertEqual(sorted(status.tolist()), [-1, 0])
    np.testing.assert_array_equal(runs["coupled"], runs["local"])
    test.assertGreater(float(np.max(np.abs(runs["coupled_unrelaxed"] - runs["coupled"]))), 1.0e-6)


def test_coupled_metrics_report_real_block_residuals(test, device):
    """Report real per-body solve metrics for coupled elastic bodies and NaN for Schur bodies.

    With relaxation 0.65 the coupled solve's linear residual is far below the initial residual,
    while the applied residual is about ``(1 - 0.65)`` of it.
    """
    for backend in _BACKENDS:
        with test.subTest(backend=backend):
            model, state = _make_mixed_mode_count_model(device)
            solver = _solver(
                model, _SPARSE, iterations=4, relaxation=0.65, backend=backend, elastic_solver_metrics=True
            )
            _series(model, state, solver, 3)
            metrics = solver.elastic_mode_solve_metrics()
            if backend == "schur":
                for name, values in metrics.items():
                    test.assertTrue(np.isnan(values).all(), name)
                continue
            initial = metrics["initial_residual_norm"]
            test.assertTrue(np.all(initial > 0.0))
            test.assertTrue(np.all(metrics["solve_residual_norm"] < 1.0e-4 * initial))
            np.testing.assert_allclose(metrics["applied_residual_norm"] / initial, 0.35, rtol=1.0e-3)
            test.assertTrue(np.all(metrics["update_norm"] > 0.0))
            test.assertTrue(np.all(metrics["update_max"] <= metrics["update_norm"]))


def test_coupled_regularization_applied_once_by_backend(test, device):
    """Apply ``rigid_articulation_diagonal_regularization`` of mixed articulations once, in the coupled backend.

    The rigid-prefix kernel must not regularize the bodies of mixed articulations, so the assembled
    mixed system is identical with and without regularization. The backend adds the value exactly
    once to the six frame diagonal entries of every mixed row (the regularized diagonal it keeps
    for the Jacobi fallback differs by one ``float32`` addition) and leaves the modal diagonal
    entries unchanged, which changes the solved update.
    """
    regularization = 50.0
    assembled, deltas, backups = {}, {}, {}
    for value in (0.0, regularization):
        model, state = _make_mixed_mode_count_model(device)
        solver = _solver(
            model,
            _SPARSE,
            iterations=1,
            elastic_solver_metrics=True,
            rigid_articulation_diagonal_regularization=value,
            **_COUPLED,
        )
        _series(model, state, solver, 1)
        layout = solver.rigid_articulation_sparse_layout
        assembled[value] = solver._coupled_metric_values.numpy()[36 * layout.rigid_block_count :]
        deltas[value] = solver._rigid_articulation_sparse_delta_scalar.numpy().copy()
        backups[value] = solver._coupled_diag_backup.numpy().copy()
    np.testing.assert_array_equal(assembled[regularization], assembled[0.0])
    test.assertGreater(float(np.max(np.abs(deltas[regularization] - deltas[0.0]))), 0.0)

    offsets = layout.articulation_row_scalar_offsets.numpy()
    frame = np.zeros(backups[0.0].shape[0], dtype=bool)
    modal = np.zeros_like(frame)
    for row in range(layout.rigid_body_row_count, layout.articulation_body_count):
        frame[offsets[row] : offsets[row] + 6] = True
        modal[offsets[row] + 6 : offsets[row + 1]] = True
    test.assertTrue(modal.any())
    base = backups[0.0].astype(np.float32)
    once = base + np.float32(regularization)
    np.testing.assert_array_equal(backups[regularization][frame], once[frame])
    # Twice would differ: every frame diagonal entry here is small enough to resolve the addition.
    test.assertTrue(np.all((once + np.float32(regularization))[frame] != once[frame]))
    np.testing.assert_array_equal(backups[regularization][modal], base[modal])


def test_coupled_validation(test, device):
    """Validate the backend option and reject joints of elastic bodies that the coupled assembly would drop."""
    model, _state, hinge = _make_leaf_link_model(device)
    with test.assertRaisesRegex(ValueError, "rigid_articulation_elastic_solve"):
        _solver(model, _SPARSE, backend="dense")
    solver = _solver(model, _SPARSE, **_COUPLED)
    layout = solver.rigid_articulation_sparse_layout
    joints = layout.mixed_articulation_joints.numpy()
    layout.mixed_articulation_joints = wp.array(joints[joints != hinge], dtype=wp.int32, device=device)
    with test.assertRaisesRegex(ValueError, rf"Joint {hinge} attaches reduced elastic body .* outside"):
        solver._validate_sparse_elastic_support()


class TestReducedElasticSparseArticulation(unittest.TestCase):
    pass


class TestReducedElasticCoupledArticulation(unittest.TestCase):
    pass


for _name, _test in (
    ("test_sparse_standalone_elastic_body_matches_local", test_sparse_standalone_elastic_body_matches_local),
    ("test_sparse_elastic_leaf_link_matches_local", test_sparse_elastic_leaf_link_matches_local),
    ("test_sparse_elastic_coupler_closed_loop", test_sparse_elastic_coupler_closed_loop),
    ("test_sparse_kinematic_elastic_parent_matches_local", test_sparse_kinematic_elastic_parent_matches_local),
    ("test_sparse_mixed_mode_counts_match_local", test_sparse_mixed_mode_counts_match_local),
    ("test_sparse_elastic_rejects_unsupported_joint_terms", test_sparse_elastic_rejects_unsupported_joint_terms),
    ("test_sparse_elastic_rejects_joint_mimics", test_sparse_elastic_rejects_joint_mimics),
    ("test_elastic_articulation_honors_deterministic_option", test_elastic_articulation_honors_deterministic_option),
):
    for _backend in _BACKENDS:
        add_function_test(
            TestReducedElasticSparseArticulation, f"{_name}_{_backend}", _test, devices=devices, backend=_backend
        )

add_function_test(
    TestReducedElasticSparseArticulation,
    "test_sparse_elastic_elastic_joint_stays_stable_schur",
    test_sparse_elastic_elastic_joint_stays_stable,
    devices=devices,
)

for _name, _test in (
    ("test_sparse_elastic_cpu_cuda_parity", test_sparse_elastic_cpu_cuda_parity),
    ("test_sparse_elastic_leaf_link_graph_capture", test_sparse_elastic_leaf_link_graph_capture),
):
    for _backend in _BACKENDS:
        add_function_test(
            TestReducedElasticSparseArticulation, f"{_name}_{_backend}", _test, devices=cuda_devices, backend=_backend
        )

for _name, _test in (
    ("test_coupled_topologies_match_converged_local", test_coupled_topologies_match_converged_local),
    ("test_coupled_adjacent_elastic_accuracy", test_coupled_adjacent_elastic_accuracy),
    ("test_coupled_mode_mode_cross_block_is_required", test_coupled_mode_mode_cross_block_is_required),
    ("test_coupled_relaxation_applies_update_once", test_coupled_relaxation_applies_update_once),
    ("test_coupled_run_to_run_deterministic", test_coupled_run_to_run_deterministic),
    ("test_coupled_failure_reports_status_and_warns", test_coupled_failure_reports_status_and_warns),
    ("test_coupled_ringing_matches_converged_local", test_coupled_ringing_matches_converged_local),
    ("test_coupled_projected_joint_types_on_elastic_link", test_coupled_projected_joint_types_on_elastic_link),
    ("test_coupled_standalone_elastic_body_uses_local_solve", test_coupled_standalone_elastic_body_uses_local_solve),
    ("test_coupled_metrics_report_real_block_residuals", test_coupled_metrics_report_real_block_residuals),
    ("test_coupled_regularization_applied_once_by_backend", test_coupled_regularization_applied_once_by_backend),
    ("test_coupled_validation", test_coupled_validation),
):
    add_function_test(TestReducedElasticCoupledArticulation, _name, _test, devices=devices)

for _name, _test in (
    ("test_coupled_cpu_cuda_parity", test_coupled_cpu_cuda_parity),
    ("test_coupled_graph_capture_replicated_articulations", test_coupled_graph_capture_replicated_articulations),
):
    add_function_test(TestReducedElasticCoupledArticulation, _name, _test, devices=cuda_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
