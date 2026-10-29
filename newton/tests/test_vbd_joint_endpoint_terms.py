# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared joint endpoint evaluation of the VBD block-sparse articulation solve.

``evaluate_articulation_joint_terms`` evaluates each joint once and returns its wrench, self and
cross Hessian blocks per endpoint. These tests check it against the three existing joint
evaluation sites (the sparse scatter kernel, the reduced elastic endpoint evaluation and the
local VBD joint evaluation), against finite differences of the joint wrench, and check the
elastic projection helpers against a NumPy ``J^T K J``.
"""

import math
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton._src.solvers.vbd import reduced_elastic_kernels
from newton._src.solvers.vbd import rigid_sparse_articulation_kernels as sparse_kernels
from newton._src.solvers.vbd.joint_coordinates import JointCoordinateData
from newton._src.solvers.vbd.rigid_sparse_articulation_kernels import (
    JointEndpointTerms,
    SparseElasticJointData,
    evaluate_articulation_joint_terms,
    joint_endpoint_mode_twist,
    mat66f,
    vec6f,
)
from newton._src.solvers.vbd.rigid_vbd_kernels import RigidForceElementAdjacencyInfo, evaluate_joint_force_hessian
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()

_SPARSE = "block_sparse_joints"
_JOINT_KERNEL = sparse_kernels.assemble_articulation_joints_scalar
_SERIAL_KERNEL = sparse_kernels.solve_articulation_sparse_serial_scalar
_JOINT_ARGS = [arg.label for arg in _JOINT_KERNEL.adj.args][:-2]
_STRUCTS = (SparseElasticJointData, JointCoordinateData, RigidForceElementAdjacencyInfo)


# ---------------------------------------------------------------------------------------------
# Capturing the sparse joint inputs of one solver iteration
# ---------------------------------------------------------------------------------------------


def _clone(value):
    if isinstance(value, wp.array):
        return wp.clone(value)
    for struct in _STRUCTS:
        if isinstance(value, struct.cls):
            copy = struct()
            for name in struct.vars:
                setattr(copy, name, _clone(getattr(value, name)))
            return copy
    return value


def _kernel_args(kernel, args, kwargs):
    values = list(kwargs.get("inputs", [])) + list(kwargs.get("outputs", []))
    if not values and len(args) > 2:
        values = list(args[2])
    names = [arg.label for arg in kernel.adj.args]
    return dict(zip(names, values, strict=True))


def _capture_joint_inputs(solver, state_0, state_1, control, dt):
    """Step once and return copies of the first sparse joint assembly's inputs.

    Also returns the outputs of the first :func:`evaluate_elastic_endpoint_joints` launch of the
    same iteration (``None`` without reduced elastic bodies).
    """
    captured = {}
    launch = wp.launch

    def spy(*args, **kwargs):
        kernel = kwargs.get("kernel", args[0] if args else None)
        if "joint" not in captured and kernel in (_JOINT_KERNEL, _SERIAL_KERNEL):
            values = _kernel_args(kernel, args, kwargs)
            if kernel is _JOINT_KERNEL or values["assemble"]:
                captured["joint"] = {name: _clone(values[name]) for name in _JOINT_ARGS}
                captured["values_size"] = values["values_scalar"].shape[0]
                captured["rhs_size"] = values["rhs_scalar"].shape[0]
        result = launch(*args, **kwargs)
        if (
            "joint" in captured
            and "elastic" not in captured
            and kernel is reduced_elastic_kernels.evaluate_elastic_endpoint_joints
        ):
            values = _kernel_args(kernel, args, kwargs)
            terms = values["endpoint_terms"]
            captured["elastic"] = {
                "body_ids": values["body_ids"].numpy().copy(),
                "offsets": values["adjacency"].body_adj_joints_offsets.numpy().copy(),
                "joints": values["adjacency"].body_adj_joints.numpy().copy(),
                **{
                    name: getattr(terms, name).numpy().copy()
                    for name in reduced_elastic_kernels.ElasticEndpointJointTerms.vars
                },
            }
        return result

    with mock.patch.object(wp, "launch", spy):
        solver.step(state_0, state_1, control, None, dt)
    return captured


def _launch_legacy(capture, device, cursor=None):
    """Run the sparse scatter kernel on the captured inputs; optionally only for one joint cursor."""
    args = dict(capture["joint"])
    if cursor is not None:
        args["articulation_joints"] = wp.array(
            args["articulation_joints"].numpy()[cursor : cursor + 1], dtype=wp.int32, device=device
        )
        args["articulation_joint_body_start"] = wp.array(
            args["articulation_joint_body_start"].numpy()[cursor : cursor + 1], dtype=wp.int32, device=device
        )
    values = wp.zeros(capture["values_size"], dtype=float, device=device)
    rhs = wp.zeros(capture["rhs_size"], dtype=float, device=device)
    wp.launch(
        _JOINT_KERNEL,
        dim=args["articulation_joints"].shape[0],
        inputs=[args[name] for name in _JOINT_ARGS],
        outputs=[values, rhs],
        device=device,
    )
    return values.numpy(), rhs.numpy()


def _launch_evaluator(capture, device, body_q=None):
    """Run :func:`evaluate_articulation_joint_terms` on the captured inputs (optionally other poses)."""
    args = dict(capture["joint"])
    args["joints"] = args["articulation_joints"]
    if body_q is not None:
        args["body_q"] = wp.array(body_q, dtype=wp.transform, device=device)
    names = [arg.label for arg in evaluate_articulation_joint_terms.adj.args][:-1]
    terms = wp.zeros(args["joints"].shape[0], dtype=JointEndpointTerms, device=device)
    wp.launch(
        evaluate_articulation_joint_terms,
        dim=args["joints"].shape[0],
        inputs=[args[name] for name in names],
        outputs=[terms],
        device=device,
    )
    return terms.numpy()


# ---------------------------------------------------------------------------------------------
# Block-sparse layout helpers (NumPy)
# ---------------------------------------------------------------------------------------------


def _slot(capture, body_start, row_local, col_local):
    offsets = capture["joint"]["articulation_block_row_offsets"].numpy()
    cols = capture["joint"]["articulation_block_cols"].numpy()
    row = body_start + row_local
    for s in range(offsets[row], offsets[row + 1]):
        if cols[s] == col_local:
            return s
    return -1


def _block(values, slot):
    return values[slot * 36 : slot * 36 + 36].reshape(6, 6)


def _cross_slot(capture, joint):
    elastic = capture["joint"]["elastic"]
    if elastic.enabled == 0:
        return -1
    return int(elastic.joint_cross_slot.numpy()[joint])


def _scatter_terms(capture, terms):
    """Scatter evaluator terms into the sparse layout like the legacy kernel (float64)."""
    values = np.zeros(capture["values_size"])
    rhs = np.zeros(capture["rhs_size"])
    local = capture["joint"]["body_articulation_local"].numpy()
    joints = capture["joint"]["articulation_joints"].numpy()
    body_start = capture["joint"]["articulation_joint_body_start"].numpy()
    for cursor, t in enumerate(terms):
        if not t["evaluated"]:
            continue
        start = int(body_start[cursor])
        child_local = int(local[t["child"]])
        rhs[(start + child_local) * 6 : (start + child_local) * 6 + 6] += t["wrench_child"]
        _block(values, _slot(capture, start, child_local, child_local))[:] += t["hessian_child"]
        if t["parent"] < 0:
            continue
        parent_local = int(local[t["parent"]])
        rhs[(start + parent_local) * 6 : (start + parent_local) * 6 + 6] += t["wrench_parent"]
        _block(values, _slot(capture, start, parent_local, parent_local))[:] += t["hessian_parent"]
        if parent_local >= child_local:
            _block(values, _slot(capture, start, parent_local, child_local))[:] += t["hessian_cross"]
        else:
            _block(values, _slot(capture, start, child_local, parent_local))[:] += t["hessian_cross"].T
        cross_slot = _cross_slot(capture, int(joints[cursor]))
        if cross_slot >= 0:
            _block(values, cross_slot)[:] += t["hessian_cross"]
    return values, rhs


# ---------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _quat_from_rotvec(v):
    angle = float(np.linalg.norm(v))
    if angle < 1.0e-12:
        return np.array([0.5 * v[0], 0.5 * v[1], 0.5 * v[2], 1.0]) / np.sqrt(1.0 + 0.25 * angle * angle)
    axis = v / angle
    return np.array([*(axis * math.sin(0.5 * angle)), math.cos(0.5 * angle)])


def _quat_to_matrix(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _perturb(state, translation, rotation):
    body_q = state.body_q.numpy().astype(np.float64)
    for b in range(body_q.shape[0]):
        phase = float(b + 1)
        body_q[b, 0] += translation * math.sin(1.7 * phase)
        body_q[b, 1] += 0.5 * translation * math.cos(2.3 * phase)
        body_q[b, 2] += 0.3 * translation * math.sin(0.7 * phase)
        axis = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])
        q = _quat_mul(_quat_from_rotvec(axis * rotation * math.sin(1.1 * phase)), body_q[b, 3:7])
        body_q[b, 3:7] = q / np.linalg.norm(q)
    state.body_q.assign(body_q.astype(np.float32))


def _chain_model(device, kind: str, *, drives: bool = False, extras: bool = False, gravity=(0.0, 0.0, 0.0)):
    """Build a world-rooted chain of three offset-COM links joined by ``kind`` joints.

    ``drives`` adds drives and (active) limits; ``extras`` adds friction, damping and armature,
    which only the block-sparse path evaluates as joint terms.
    """
    builder = newton.ModelBuilder(gravity=gravity)
    inertia = wp.mat33(np.diag([0.02, 0.03, 0.04]).astype(np.float32))
    links = [
        builder.add_link(
            xform=wp.transform(wp.vec3(0.3 * i + 0.15, 0.0, 1.0), wp.quat_identity()),
            mass=1.0 + 0.2 * i,
            inertia=inertia,
            com=wp.vec3(0.01 * (i + 1), -0.02, 0.005),
        )
        for i in range(3)
    ]
    half = wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity())
    minus_half = wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity())
    root_xform = wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())
    Dof = newton.ModelBuilder.JointDofConfig
    joints = []
    for i, link in enumerate(links):
        parent = -1 if i == 0 else links[i - 1]
        parent_xform = root_xform if i == 0 else half
        common = {"parent": parent, "child": link, "parent_xform": parent_xform, "child_xform": minus_half}
        drive = {}
        if drives:
            drive = {
                "target_ke": 40.0 + 10.0 * i,
                "target_kd": 0.5,
                "limit_lower": -0.02,
                "limit_upper": 0.02,
                "limit_ke": 500.0,
                "limit_kd": 0.5,
            }
        if extras:
            drive["friction"] = 0.2
        if kind == "revolute":
            if extras:
                drive["armature"] = 0.05
            joints.append(
                builder.add_joint_revolute(axis=(0.0, 0.3, 1.0), target_pos=0.2 if drives else 0.0, **drive, **common)
            )
        elif kind == "prismatic":
            joints.append(
                builder.add_joint_prismatic(axis=(1.0, 0.2, 0.0), target_pos=0.01 if drives else 0.0, **drive, **common)
            )
        elif kind == "d6":
            joints.append(
                builder.add_joint_d6(
                    linear_axes=[Dof(axis=newton.Axis.X, **drive), Dof(axis=newton.Axis.Y, **drive)],
                    angular_axes=[Dof(axis=newton.Axis.Z, **drive), Dof(axis=newton.Axis.Y, **drive)],
                    **common,
                )
            )
        elif kind == "ball":
            joints.append(builder.add_joint_ball(**common))
        elif kind == "fixed":
            joints.append(builder.add_joint_fixed(**common))
        elif kind == "rod":
            joints.append(
                builder.add_joint_rod(stretch_stiffness=1.0e5, shear_stiffness=2.0e4, bend_stiffness=1.0e3, **common)
            )
        else:
            raise ValueError(kind)
    builder.add_articulation(joints)
    builder.color()
    model = builder.finalize(device=device)
    if extras:
        damping = model.joint_damping.numpy()
        damping[:] = 0.05
        model.joint_damping.assign(damping)
    return model


def _elastic_chain_model(device):
    """Build world - rigid base - reduced elastic link - rigid tip with revolute joints and drives.

    The elastic link is the child of one joint and the parent of the next, so both endpoint sides
    carry modal attachment motion.
    """
    length = 0.6
    beam = newton.ModalGeneratorBeam
    basis = beam(
        length=length,
        half_width_y=0.03,
        half_width_z=0.02,
        young_modulus=2.0e6,
        density=1000.0,
        mode_specs=[
            {"type": beam.Mode.AXIAL},
            {"type": beam.Mode.BENDING_Y, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
            {"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
        ],
    ).build()
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    base = builder.add_link(
        xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(*np.diag([0.002, 0.02, 0.02]).ravel()),
    )
    link = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(0.4 + 0.5 * length, 0.0, 0.0), wp.quat_identity()),
        modal_basis=basis,
        mode_q=[0.004, -0.01, 0.02],
    )
    tip = builder.add_link(
        xform=wp.transform(wp.vec3(0.4 + length + 0.15, 0.0, 0.0), wp.quat_identity()),
        mass=0.3,
        inertia=wp.mat33(*np.diag([0.001, 0.003, 0.003]).ravel()),
        com=wp.vec3(0.02, 0.01, 0.0),
    )
    axis = (0.0, 0.2, 1.0)
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=base,
            axis=axis,
            child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
            target_ke=30.0,
            target_kd=0.2,
            target_pos=0.3,
        ),
        builder.add_joint_revolute(
            parent=base,
            child=link,
            axis=axis,
            parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity()),
            limit_lower=-0.01,
            limit_upper=0.01,
            limit_ke=200.0,
        ),
        builder.add_joint_revolute(
            parent=link,
            child=tip,
            axis=axis,
            parent_xform=wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity()),
            target_ke=5.0,
            target_pos=-0.2,
        ),
    ]
    builder.add_articulation(joints)
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[model.joint_qd_start.numpy()[joints[0]]] = 1.5
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


def _capture_model(model, device, *, perturb=(0.0, 0.0), state=None, dt=1.0 / 240.0, **solver_kwargs):
    if state is None:
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    if perturb[0] > 0.0 or perturb[1] > 0.0:
        _perturb(state, *perturb)
    # The captured 6 x 6 scatter kernel assembles elastic endpoints only under the Schur backend.
    solver_kwargs.setdefault("rigid_articulation_elastic_solve", "schur")
    solver = newton.solvers.SolverVBD(model, iterations=1, rigid_articulation_solve=_SPARSE, **solver_kwargs)
    return _capture_joint_inputs(solver, state, model.state(), model.control(), dt)


def _rigid_captures(device):
    """Captured joint inputs of perturbed rigid fixtures covering every sparse joint term."""
    return {
        "revolute_drive_limit_extras": _capture_model(
            _chain_model(device, "revolute", drives=True, extras=True), device, perturb=(0.01, 0.06)
        ),
        "prismatic_drive_limit_extras": _capture_model(
            _chain_model(device, "prismatic", drives=True, extras=True), device, perturb=(0.01, 0.06)
        ),
        "d6_drive_limit_extras": _capture_model(
            _chain_model(device, "d6", drives=True, extras=True), device, perturb=(0.01, 0.06)
        ),
        "ball": _capture_model(_chain_model(device, "ball"), device, perturb=(0.01, 0.06)),
        "fixed_penalty_damped": _capture_model(
            _chain_model(device, "fixed"),
            device,
            perturb=(0.01, 0.06),
            rigid_joint_linear_kd=0.02,
            rigid_joint_angular_kd=0.02,
        ),
        "rod": _capture_model(_chain_model(device, "rod"), device, perturb=(0.01, 0.06)),
    }


# ---------------------------------------------------------------------------------------------
# Evaluator vs the legacy sparse scatter kernel
# ---------------------------------------------------------------------------------------------


def _assert_matches_legacy(test, capture, device, label):
    terms = _launch_evaluator(capture, device)
    test.assertTrue(np.all(terms["evaluated"] == 1), label)
    legacy_values, legacy_rhs = _launch_legacy(capture, device)
    values, rhs = _scatter_terms(capture, terms)
    scale = max(float(np.abs(legacy_values).max()), 1.0)
    np.testing.assert_allclose(values, legacy_values, rtol=1.0e-5, atol=1.0e-6 * scale, err_msg=label)
    scale = max(float(np.abs(legacy_rhs).max()), 1.0)
    np.testing.assert_allclose(rhs, legacy_rhs, rtol=1.0e-5, atol=1.0e-6 * scale, err_msg=label)
    test.assertGreater(float(np.abs(legacy_rhs).max()), 0.0, label)

    # Per joint, both walkers add the same blocks in the same order; they differ only by the
    # floating-point contraction of the accumulation (a few ulp).
    joints = capture["joint"]["articulation_joints"].numpy()
    local = capture["joint"]["body_articulation_local"].numpy()
    body_start = capture["joint"]["articulation_joint_body_start"].numpy()
    for cursor, t in enumerate(terms):
        values_j, rhs_j = _launch_legacy(capture, device, cursor)
        start = int(body_start[cursor])
        child_local = int(local[t["child"]])
        expected_rhs = np.zeros_like(rhs_j)
        expected_values = np.zeros_like(values_j)
        expected_rhs[(start + child_local) * 6 : (start + child_local) * 6 + 6] = t["wrench_child"]
        _block(expected_values, _slot(capture, start, child_local, child_local))[:] = t["hessian_child"]
        if t["parent"] >= 0:
            parent_local = int(local[t["parent"]])
            expected_rhs[(start + parent_local) * 6 : (start + parent_local) * 6 + 6] = t["wrench_parent"]
            _block(expected_values, _slot(capture, start, parent_local, parent_local))[:] = t["hessian_parent"]
            if parent_local >= child_local:
                _block(expected_values, _slot(capture, start, parent_local, child_local))[:] = t["hessian_cross"]
            else:
                _block(expected_values, _slot(capture, start, child_local, parent_local))[:] = t["hessian_cross"].T
            cross_slot = _cross_slot(capture, int(joints[cursor]))
            if cross_slot >= 0:
                _block(expected_values, cross_slot)[:] = t["hessian_cross"]
        for actual, expected in ((rhs_j, expected_rhs), (values_j, expected_values)):
            scale = max(float(np.abs(expected).max()), 1.0e-6)
            np.testing.assert_allclose(
                actual, expected, rtol=1.0e-6, atol=1.0e-7 * scale, err_msg=f"{label} joint {cursor}"
            )


def test_evaluator_matches_sparse_scatter_rigid(test, device):
    """Evaluator terms scattered into the layout reproduce the sparse joint assembly."""
    for label, capture in _rigid_captures(device).items():
        _assert_matches_legacy(test, capture, device, label)


def test_evaluator_matches_sparse_scatter_cable_rod(test, device):
    """A ROD cable chain (world-rooted, anisotropic, damped) matches the sparse assembly."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    rod = newton.Rod.create_straight(
        start=wp.vec3(0.0, 0.0, 1.0), direction=wp.vec3(1.0, 0.0, 0.0), length=0.4, segment_count=4, radius=0.02
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="add_rod: wrap_in_articulation=False", category=UserWarning)
        bodies, joints = builder.add_rod(
            rod=rod,
            stretch_stiffness=1.0e4,
            shear_stiffness=1.0e3,
            bend_stiffness=1.0e2,
            twist_stiffness=5.0e2,
            stretch_damping=1.0e-2,
            bend_damping=1.0e-3,
            wrap_in_articulation=False,
            body_frame_origin="start",
        )
    joints.append(
        builder.add_joint_fixed(
            parent=-1,
            child=bodies[0],
            parent_xform=wp.transform(wp.vec3(rod.points[0]), wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
    )
    builder.add_articulation(joints, allow_closed_loops=True)
    builder.color()
    model = builder.finalize(device=device)
    capture = _capture_model(model, device, perturb=(0.005, 0.03))
    _assert_matches_legacy(test, capture, device, "cable")


def test_evaluator_matches_sparse_scatter_elastic(test, device):
    """Joints at deformed elastic attachments, incl. the Schur cross capture, match the sparse assembly."""
    model, state = _elastic_chain_model(device)
    capture = _capture_model(model, device, state=state)
    test.assertEqual(capture["joint"]["elastic"].enabled, 1)
    _assert_matches_legacy(test, capture, device, "elastic chain")


# ---------------------------------------------------------------------------------------------
# Evaluator vs the reduced elastic endpoint evaluation and the local joint evaluation
# ---------------------------------------------------------------------------------------------


@wp.kernel
def _local_joint_terms_kernel(
    joints: wp.array[wp.int32],
    bodies: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    elastic: SparseElasticJointData,
    joint_axis: wp.array[wp.vec3],
    joint_rod_rest_kb_local: wp.array[wp.vec3],
    joint_rod_rest_twist: wp.array[float],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
    joint_rho: wp.array[float],
    joint_material_k: wp.array[float],
    joint_penalty_kd: wp.array[float],
    joint_sigma_start: wp.array[wp.vec3],
    joint_C_fric: wp.array[wp.vec3],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    joint_drive_limit_support: wp.array[float],
    joint_drive_lambda: wp.array[float],
    joint_limit_lambda: wp.array[float],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    avbd_alpha: float,
    joint_compliant_alm: int,
    joint_dof_dim: wp.array2d[int],
    joint_coordinates: JointCoordinateData,
    dt: float,
    wrench: wp.array[vec6f],
    hessian: wp.array[mat66f],
):
    tid = wp.tid()
    force, torque, H_ll, H_al, H_aa = evaluate_joint_force_hessian(
        bodies[tid],
        joints[tid],
        body_q,
        body_q_prev,
        body_q_rest,
        body_com,
        joint_type,
        joint_enabled,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        elastic.body_elastic_index,
        elastic.elastic_joint,
        elastic.elastic_mode_count,
        elastic.joint_q,
        elastic.joint_q_prev,
        elastic.joint_q_start,
        elastic.parent_endpoint,
        elastic.child_endpoint,
        elastic.endpoint_phi,
        elastic.endpoint_psi,
        elastic.max_mode_count,
        joint_axis,
        joint_rod_rest_kb_local,
        joint_rod_rest_twist,
        joint_qd_start,
        joint_target_q_start,
        joint_constraint_start,
        joint_penalty_k,
        joint_rho,
        joint_material_k,
        joint_penalty_kd,
        joint_sigma_start,
        joint_C_fric,
        joint_target_ke,
        joint_target_kd,
        joint_target_q,
        joint_target_vel,
        joint_limit_lower,
        joint_limit_upper,
        joint_limit_ke,
        joint_limit_kd,
        joint_drive_limit_support,
        joint_drive_lambda,
        joint_limit_lambda,
        joint_lambda_lin,
        joint_lambda_ang,
        joint_C0_lin,
        joint_C0_ang,
        joint_is_hard,
        avbd_alpha,
        joint_compliant_alm,
        joint_dof_dim,
        joint_coordinates,
        dt,
    )
    wrench[tid] = vec6f(force[0], force[1], force[2], torque[0], torque[1], torque[2])
    H = mat66f(0.0)
    for i in range(3):
        for j in range(3):
            H[i, j] = H_ll[i, j]
            H[i + 3, j] = H_al[i, j]
            H[i, j + 3] = H_al[j, i]
            H[i + 3, j + 3] = H_aa[i, j]
    hessian[tid] = H


def _local_joint_terms(capture, device, terms):
    """Evaluate every evaluated endpoint with the local VBD joint evaluation."""
    joints = capture["joint"]["articulation_joints"].numpy()
    entries = []
    for cursor, t in enumerate(terms):
        if t["evaluated"]:
            entries.append((cursor, int(joints[cursor]), int(t["child"]), False))
            if t["parent"] >= 0:
                entries.append((cursor, int(joints[cursor]), int(t["parent"]), True))
    args = capture["joint"]
    names = [arg.label for arg in _local_joint_terms_kernel.adj.args][2:-2]
    wrench = wp.zeros(len(entries), dtype=vec6f, device=device)
    hessian = wp.zeros(len(entries), dtype=mat66f, device=device)
    wp.launch(
        _local_joint_terms_kernel,
        dim=len(entries),
        inputs=[
            wp.array([e[1] for e in entries], dtype=wp.int32, device=device),
            wp.array([e[2] for e in entries], dtype=wp.int32, device=device),
            *[args[name] for name in names],
        ],
        outputs=[wrench, hessian],
        device=device,
    )
    return entries, wrench.numpy(), hessian.numpy()


def _assert_close_blocks(test, actual, expected, label, rtol=2.0e-4):
    scale = max(float(np.abs(expected).max()), 1.0e-6)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol * scale, err_msg=label)


def test_evaluator_matches_local_joint_evaluation(test, device):
    """Wrench and self Hessian agree with the local VBD joint evaluation for every structural/drive/limit term.

    Friction, damping and armature rows are absent from the fixtures: the local path evaluates them
    outside the joint force evaluation.
    """
    captures = {
        "revolute_drive_limit": _capture_model(
            _chain_model(device, "revolute", drives=True), device, perturb=(0.01, 0.06)
        ),
        "prismatic_drive_limit": _capture_model(
            _chain_model(device, "prismatic", drives=True), device, perturb=(0.01, 0.06)
        ),
        "d6_drive_limit": _capture_model(_chain_model(device, "d6", drives=True), device, perturb=(0.01, 0.06)),
        "ball": _capture_model(_chain_model(device, "ball"), device, perturb=(0.01, 0.06)),
        "fixed_damped": _capture_model(
            _chain_model(device, "fixed"),
            device,
            perturb=(0.01, 0.06),
            rigid_joint_linear_kd=0.02,
            rigid_joint_angular_kd=0.02,
        ),
        "rod": _capture_model(_chain_model(device, "rod"), device, perturb=(0.01, 0.06)),
    }
    for label, capture in captures.items():
        terms = _launch_evaluator(capture, device)
        entries, wrench, hessian = _local_joint_terms(capture, device, terms)
        for k, (cursor, _joint, _body, is_parent) in enumerate(entries):
            side = "parent" if is_parent else "child"
            _assert_close_blocks(test, terms[cursor][f"wrench_{side}"], wrench[k], f"{label} {k} wrench")
            _assert_close_blocks(test, terms[cursor][f"hessian_{side}"], hessian[k], f"{label} {k} hessian")


def test_evaluator_matches_elastic_endpoint_evaluation(test, device):
    """All three sites agree on wrench, self Hessian, attachment offset and rotation Jacobian at elastic links."""
    model, state = _elastic_chain_model(device)
    capture = _capture_model(model, device, state=state)
    elastic = capture["elastic"]
    test.assertIsNotNone(elastic)
    terms = _launch_evaluator(capture, device)
    entries, local_wrench, local_hessian = _local_joint_terms(capture, device, terms)

    joints = capture["joint"]["articulation_joints"].numpy()
    cursor_of = {int(j): c for c, j in enumerate(joints)}
    body_q = capture["joint"]["body_q"].numpy().astype(np.float64)
    parent_endpoint = model.joint_parent_elastic_endpoint.numpy()
    child_endpoint = model.joint_child_elastic_endpoint.numpy()
    joint_parent = model.joint_parent.numpy()
    body_elastic_index = model.body_elastic_index.numpy()
    compared = 0
    for body in elastic["body_ids"]:
        if body_elastic_index[body] < 0:
            continue
        R = _quat_to_matrix(body_q[body, 3:7])
        for entry in range(elastic["offsets"][body], elastic["offsets"][body + 1]):
            joint = int(elastic["joints"][entry])
            is_parent = int(joint_parent[joint]) == int(body)
            endpoint = parent_endpoint[joint] if is_parent else child_endpoint[joint]
            if endpoint < 0 or joint not in cursor_of:
                continue
            side = "parent" if is_parent else "child"
            t = terms[cursor_of[joint]]
            force = R @ elastic["force"][entry]
            torque = R @ elastic["torque"][entry]
            H = np.zeros((6, 6))
            H[:3, :3] = R @ elastic["hessian_ll"][entry] @ R.T
            H[3:, :3] = R @ elastic["hessian_al"][entry] @ R.T
            H[:3, 3:] = H[3:, :3].T
            H[3:, 3:] = R @ elastic["hessian_aa"][entry] @ R.T
            label = f"joint {joint} {side}"
            _assert_close_blocks(test, t[f"wrench_{side}"], np.concatenate((force, torque)), label + " wrench")
            _assert_close_blocks(test, t[f"hessian_{side}"], H, label + " hessian")
            # The elastic site refers the torque to the deformed attachment: tau - r x f.
            offset = t[f"attachment_offset_{side}"].astype(np.float64)
            _assert_close_blocks(
                test,
                torque - np.cross(offset, force),
                R @ elastic["endpoint_torque"][entry],
                label + " attachment offset",
            )
            np.testing.assert_allclose(
                t[f"rotation_jacobian_{side}"], elastic["rotation_jacobian"][entry], rtol=1.0e-6, atol=1.0e-7
            )
            test.assertGreater(float(np.abs(t[f"rotation_jacobian_{side}"] - np.eye(3)).max()), 1.0e-4)
            compared += 1
    test.assertEqual(compared, 2)

    for k, (cursor, _joint, _body, is_parent) in enumerate(entries):
        side = "parent" if is_parent else "child"
        _assert_close_blocks(test, terms[cursor][f"wrench_{side}"], local_wrench[k], f"local {k} wrench")
        _assert_close_blocks(test, terms[cursor][f"hessian_{side}"], local_hessian[k], f"local {k} hessian")


# ---------------------------------------------------------------------------------------------
# Cross block vs finite differences of the joint wrench
# ---------------------------------------------------------------------------------------------


def _displaced(body_q, body_com, body, direction, eps):
    """Return poses with ``body`` moved by the twist ``eps e_direction`` about its center of mass."""
    poses = body_q.astype(np.float64).copy()
    pose = poses[body]
    R = _quat_to_matrix(pose[3:7])
    com = pose[:3] + R @ body_com[body]
    delta = np.zeros(6)
    delta[direction] = eps
    rot = _quat_mul(_quat_from_rotvec(delta[3:]), pose[3:7])
    rot = rot / np.linalg.norm(rot)
    pose[3:7] = rot
    pose[:3] = com + delta[:3] - _quat_to_matrix(rot) @ body_com[body]
    return poses


def _finite_difference_blocks(capture, device, joint_cursor, eps):
    """Return ``(H_pc, H_cp, H_cc)`` from central differences of the evaluator wrenches."""
    body_q = capture["joint"]["body_q"].numpy()
    body_com = capture["joint"]["body_com"].numpy().astype(np.float64)
    base = _launch_evaluator(capture, device)[joint_cursor]
    parent, child = int(base["parent"]), int(base["child"])
    H_pc = np.zeros((6, 6))
    H_cp = np.zeros((6, 6))
    H_cc = np.zeros((6, 6))
    for d in range(6):
        plus = _launch_evaluator(capture, device, _displaced(body_q, body_com, child, d, eps).astype(np.float32))
        minus = _launch_evaluator(capture, device, _displaced(body_q, body_com, child, d, -eps).astype(np.float32))
        H_pc[:, d] = -(plus[joint_cursor]["wrench_parent"] - minus[joint_cursor]["wrench_parent"]) / (2.0 * eps)
        H_cc[:, d] = -(plus[joint_cursor]["wrench_child"] - minus[joint_cursor]["wrench_child"]) / (2.0 * eps)
        plus = _launch_evaluator(capture, device, _displaced(body_q, body_com, parent, d, eps).astype(np.float32))
        minus = _launch_evaluator(capture, device, _displaced(body_q, body_com, parent, d, -eps).astype(np.float32))
        H_cp[:, d] = -(plus[joint_cursor]["wrench_child"] - minus[joint_cursor]["wrench_child"]) / (2.0 * eps)
    return base, H_pc, H_cp, H_cc


def _assert_fd(test, analytic, fd, label, tol):
    """Compare each nonzero 3 x 3 block (linear/angular rows and columns) relative to its own size."""
    for i in range(2):
        for j in range(2):
            a = analytic[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
            b = fd[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
            scale = max(float(np.abs(a).max()), 1.0e-6 * float(np.abs(analytic).max()), 1.0e-12)
            error = float(np.abs(a - b).max()) / scale
            test.assertLess(error, tol, f"{label} block ({i}, {j}): relative error {error:.3e}\n{a}\n{b}")


def test_cross_block_matches_finite_differences(test, device):
    """``hessian_cross`` (and the child self block) equal finite differences of the wrench where exact.

    The sparse Hessian policy is Gauss-Newton: residual curvature (geometric stiffness) is dropped.
    It is exact at zero residual for every joint type, so all joint types are compared at the
    joint-satisfied configuration (no drives, no damping). Away from it only rows whose Jacobian
    does not depend on the configuration are exact; the parent-child block of a BALL joint
    (constant projector, separable anchor gap) is such a row and is also compared at a perturbed
    configuration, together with a check that the child self block there is *not* exact, which
    documents the policy.
    """
    common = {"rigid_joint_linear_kd": 0.0, "rigid_joint_angular_kd": 0.0}
    for kind in ("ball", "fixed", "revolute", "prismatic", "d6", "rod"):
        capture = _capture_model(_chain_model(device, kind), device, **common)
        for cursor in (1, 2):
            base, H_pc, H_cp, H_cc = _finite_difference_blocks(capture, device, cursor, 2.0e-3)
            test.assertLess(float(np.abs(base["wrench_child"]).max()), 1.0e-2, f"{kind}: residual not zero")
            _assert_fd(test, base["hessian_cross"], H_pc, f"{kind} H_pc", 2.0e-3)
            _assert_fd(test, base["hessian_cross"].T, H_cp, f"{kind} H_cp", 2.0e-3)
            _assert_fd(test, base["hessian_child"], H_cc, f"{kind} H_cc", 2.0e-3)

    capture = _capture_model(_chain_model(device, "ball"), device, perturb=(0.02, 0.15), **common)
    for cursor in (1, 2):
        base, H_pc, H_cp, H_cc = _finite_difference_blocks(capture, device, cursor, 1.0e-3)
        test.assertGreater(float(np.abs(base["wrench_child"][:3]).max()), 1.0, "ball: residual is zero")
        _assert_fd(test, base["hessian_cross"], H_pc, "perturbed ball H_pc", 2.0e-3)
        _assert_fd(test, base["hessian_cross"].T, H_cp, "perturbed ball H_cp", 2.0e-3)
        # The self block drops the lever-arm curvature f x dr: not exact away from zero residual.
        scale = float(np.abs(base["hessian_child"]).max())
        test.assertGreater(float(np.abs(base["hessian_child"] - H_cc).max()) / scale, 1.0e-3)


# ---------------------------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------------------------


@wp.kernel
def _mode_twist_kernel(
    rotation: wp.array[wp.mat33],
    rotation_jacobian: wp.array[wp.mat33],
    offset: wp.array[wp.vec3],
    phi: wp.array2d[wp.vec3],
    psi: wp.array2d[wp.vec3],
    twists: wp.array2d[vec6f],
):
    """For endpoint s and mode k: the projection column G_s,k of :func:`joint_endpoint_mode_twist`."""
    s, k = wp.tid()
    R = rotation[s]
    twists[s, k] = joint_endpoint_mode_twist(R, R * rotation_jacobian[s], offset[s], phi[s, k], psi[s, k])


def _left_jacobian(theta):
    angle = float(np.linalg.norm(theta))
    S = _skew(theta)
    if angle < 1.0e-8:
        return np.eye(3) + 0.5 * S
    return np.eye(3) + (1.0 - math.cos(angle)) / angle**2 * S + (angle - math.sin(angle)) / angle**3 * (S @ S)


def _numpy_G(R, J, r, phi, psi):
    """``G = [R phi + r x (R J psi); R J psi]`` as a 6 x modes matrix."""
    psi_w = R @ J @ psi.T
    return np.vstack((R @ phi.T + np.cross(r, psi_w.T).T, psi_w))


def test_mode_twist_matches_numpy(test, device):
    """Check that ``joint_endpoint_mode_twist`` equals ``G = [R phi + r x psi_w ; psi_w]`` in float64.

    The projected terms ``J^T w`` and ``J^T K J`` built from these columns are checked against
    NumPy on the production gather by ``test_projection_matches_numpy`` in
    ``test_vbd_sparse_variable_assembly``.
    """
    rng = np.random.default_rng(7)
    modes = 3
    rotations, jacobians, offsets = [], [], []
    for _ in range(2):
        rotations.append(_quat_to_matrix(_quat_from_rotvec(rng.normal(size=3))))
        jacobians.append(_left_jacobian(0.2 * rng.normal(size=3)))
        offsets.append(0.3 * rng.normal(size=3))
    phi = 0.1 * rng.normal(size=(2, modes, 3))
    psi = rng.normal(size=(2, modes, 3))

    twists = wp.zeros((2, modes), dtype=vec6f, device=device)
    wp.launch(
        _mode_twist_kernel,
        dim=(2, modes),
        inputs=[
            wp.array(np.array(rotations), dtype=wp.mat33, device=device),
            wp.array(np.array(jacobians), dtype=wp.mat33, device=device),
            wp.array(np.array(offsets), dtype=wp.vec3, device=device),
            wp.array(phi, dtype=wp.vec3, device=device),
            wp.array(psi, dtype=wp.vec3, device=device),
        ],
        outputs=[twists],
        device=device,
    )
    G = [_numpy_G(rotations[s], jacobians[s], offsets[s], phi[s], psi[s]) for s in range(2)]
    for s in range(2):
        np.testing.assert_allclose(twists.numpy()[s].T, G[s], rtol=1.0e-5, atol=1.0e-6)

    # G moves the attachment like the mode: a rigid twist [v, w] moves the point c + r by v + w x r.
    for s in range(2):
        point_motion = G[s][:3] + np.cross(G[s][3:].T, offsets[s]).T
        np.testing.assert_allclose(point_motion, rotations[s] @ phi[s].T, atol=1.0e-12)


def _attachment_referred_projection(H, r, phi_w, psi_w):
    """Today's local elastic modal rows for one self block, in world axes.

    Returns ``(frame_mode, mode_mode)`` of ``assemble_elastic_blocks``: frame columns
    ``[K_ll phi ; K_al phi + (K_aa - S^T K_ll S) psi]`` and modal entries
    ``phi^T K_ll phi + psi^T (K_aa - S^T K_ll S) psi``.
    """
    S = _skew(r)
    K_ll, K_al, K_aa = H[:3, :3], H[3:, :3], H[3:, 3:]
    K_aa_attachment = K_aa - S.T @ K_ll @ S
    frame_mode = np.vstack((K_ll @ phi_w, K_al @ phi_w + K_aa_attachment @ psi_w))
    mode_mode = phi_w.T @ K_ll @ phi_w + psi_w.T @ K_aa_attachment @ psi_w
    return frame_mode, mode_mode


def test_projection_differs_from_attachment_form_only_for_rod_parent(test, device):
    """Exact ``J^T K J`` equals the attachment-referred modal rows unless ``K_al - [r]x K_ll != 0``.

    With ``Khat_al = K_al - S K_ll`` (``S = [r]x``) the differences are, for the frame columns,
    ``[Khat_al^T psi ; Khat_al S psi]`` and, for the modal block,
    ``phi^T Khat_al^T psi + psi^T Khat_al phi + psi^T (Khat_al S - S Khat_al^T) psi``. Of the joint
    rows only the ROD parent (stretch/shear force with the child-anchor lever arm) has
    ``Khat_al != 0``.
    """
    rng = np.random.default_rng(11)
    modes = 4
    phi_w = 0.1 * rng.normal(size=(3, modes))
    psi_w = rng.normal(size=(3, modes))
    nonzero = []
    for kind, perturb in (
        ("revolute", (0.01, 0.06)),
        ("d6", (0.01, 0.06)),
        ("fixed", (0.01, 0.06)),
        ("rod", (0.02, 0.1)),
    ):
        capture = _capture_model(_chain_model(device, kind, drives=kind != "rod"), device, perturb=perturb)
        terms = _launch_evaluator(capture, device)
        for t in terms:
            for side in ("parent", "child"):
                if side == "parent" and t["parent"] < 0:
                    continue
                # Symmetrize the float32 block: the projection identities assume K_la = K_al^T.
                H = t[f"hessian_{side}"].astype(np.float64)
                H = 0.5 * (H + H.T)
                r = t[f"attachment_offset_{side}"].astype(np.float64)
                S = _skew(r)
                K_hat = H[3:, :3] - S @ H[:3, :3]
                G = np.vstack((phi_w + S @ psi_w, psi_w))
                frame_mode = H @ G
                mode_mode = G.T @ H @ G
                frame_today, mode_today = _attachment_referred_projection(H, r, phi_w, psi_w)
                expected_frame = np.vstack((K_hat.T @ psi_w, K_hat @ S @ psi_w))
                expected_mode = (
                    phi_w.T @ K_hat.T @ psi_w + psi_w.T @ K_hat @ phi_w + psi_w.T @ (K_hat @ S - S @ K_hat.T) @ psi_w
                )
                scale = float(np.abs(H).max())
                np.testing.assert_allclose(frame_mode - frame_today, expected_frame, atol=1.0e-9 * scale)
                np.testing.assert_allclose(mode_mode - mode_today, expected_mode, atol=1.0e-9 * scale)
                relative = float(np.abs(K_hat).max()) / scale
                if relative > 1.0e-5:
                    nonzero.append((kind, side))
                    test.assertGreater(float(np.abs(mode_mode - mode_today).max()) / scale, 1.0e-5)
    test.assertEqual(sorted(set(nonzero)), [("rod", "parent")])


class TestVBDJointEndpointTerms(unittest.TestCase):
    pass


add_function_test(
    TestVBDJointEndpointTerms,
    "test_evaluator_matches_sparse_scatter_rigid",
    test_evaluator_matches_sparse_scatter_rigid,
    devices=devices,
)
add_function_test(
    TestVBDJointEndpointTerms,
    "test_evaluator_matches_sparse_scatter_cable_rod",
    test_evaluator_matches_sparse_scatter_cable_rod,
    devices=devices,
)
add_function_test(
    TestVBDJointEndpointTerms,
    "test_evaluator_matches_sparse_scatter_elastic",
    test_evaluator_matches_sparse_scatter_elastic,
    devices=devices,
)
add_function_test(
    TestVBDJointEndpointTerms,
    "test_evaluator_matches_local_joint_evaluation",
    test_evaluator_matches_local_joint_evaluation,
    devices=devices,
)
add_function_test(
    TestVBDJointEndpointTerms,
    "test_evaluator_matches_elastic_endpoint_evaluation",
    test_evaluator_matches_elastic_endpoint_evaluation,
    devices=devices,
)
add_function_test(
    TestVBDJointEndpointTerms,
    "test_cross_block_matches_finite_differences",
    test_cross_block_matches_finite_differences,
    devices=devices,
)
add_function_test(
    TestVBDJointEndpointTerms,
    "test_mode_twist_matches_numpy",
    test_mode_twist_matches_numpy,
    devices=devices,
)
add_function_test(
    TestVBDJointEndpointTerms,
    "test_projection_differs_from_attachment_form_only_for_rod_parent",
    test_projection_differs_from_attachment_form_only_for_rod_parent,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
