# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Variable-size block assembly of mixed rigid/elastic articulations in the VBD block-sparse solve.

The coupled backend assembles one system per mixed articulation with a ``6 + m_i`` block per body:
the body-axes elastic block (without joint terms) rotated to world axes, the rigid body diagonals
of the 6 x 6 kernels, and every joint projected onto both endpoints with ``J_e = [I_6 | G_e]``.
These tests check the cross block against finite-difference Jacobians of the joint constraint
residual (gate 3), the projection kernel against a NumPy ``blockdiag + sum J^T K J``, the elastic
frame rows against the Schur backend's frame overwrite, the rigid rows against the legacy body
diagonal, and the switch that removes joint terms from the elastic block.
"""

import math
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton._src.solvers.vbd import reduced_elastic_kernels, reduced_elastic_sparse_kernels
from newton._src.solvers.vbd import rigid_sparse_articulation_kernels as sparse_kernels
from newton._src.solvers.vbd import rigid_sparse_articulation_variable_assembly as assembly
from newton._src.solvers.vbd.joint_coordinates import JointCoordinateData
from newton._src.solvers.vbd.reduced_elastic_kernels import ElasticEndpointJointTerms
from newton._src.solvers.vbd.rigid_sparse_articulation import build_rigid_articulation_sparse_layout
from newton._src.solvers.vbd.rigid_sparse_articulation_kernels import JointEndpointTerms, SparseElasticJointData
from newton._src.solvers.vbd.rigid_vbd_kernels import RigidForceElementAdjacencyInfo
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()

_SPARSE = "block_sparse_joints"
_STRUCTS = (SparseElasticJointData, JointCoordinateData, RigidForceElementAdjacencyInfo, ElasticEndpointJointTerms)
_CAPTURE_KERNELS = {
    "diag": sparse_kernels.assemble_articulation_body_diagonal_scalar,
    "joint": sparse_kernels.assemble_articulation_joints_scalar,
    "endpoint": reduced_elastic_kernels.evaluate_elastic_endpoint_joints,
    "blocks": reduced_elastic_kernels.assemble_elastic_blocks,
    "frame": reduced_elastic_sparse_kernels.assemble_elastic_sparse_frame_blocks,
}


# ---------------------------------------------------------------------------------------------
# Capturing the assembly inputs of one Schur-backend iteration
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
    names = [arg.label for arg in kernel.adj.args]
    return dict(zip(names, values, strict=True))


def _capture(model, state, device, dt=1.0 / 240.0):
    """Step a Schur-backend solver once and copy the first launch of each assembly kernel.

    The kernels of one iteration read the same iterate, so the copies are mutually consistent.
    ``evaluate_elastic_endpoint_joints`` is copied after its launch, so its output is included.
    """
    solver = newton.solvers.SolverVBD(model, iterations=1, rigid_articulation_solve=_SPARSE)
    captured = {}
    launch = wp.launch

    def spy(*args, **kwargs):
        kernel = kwargs.get("kernel", args[0] if args else None)
        name = next((key for key, value in _CAPTURE_KERNELS.items() if value is kernel), None)
        if name is not None and name not in captured and name != "endpoint":
            captured[name] = {k: _clone(v) for k, v in _kernel_args(kernel, args, kwargs).items()}
            captured[name + "_dim"] = kwargs.get("dim")
        result = launch(*args, **kwargs)
        if name == "endpoint" and name not in captured:
            captured[name] = {k: _clone(v) for k, v in _kernel_args(kernel, args, kwargs).items()}
        return result

    with mock.patch.object(wp, "launch", spy):
        solver.step(state, model.state(), model.control(), None, dt)
    missing = [name for name in _CAPTURE_KERNELS if name not in captured]
    if missing:
        raise RuntimeError(f"assembly kernels not launched: {missing}")
    return captured


def _elastic_blocks(capture, device, joint_terms):
    """Re-run ``assemble_elastic_blocks`` on the captured inputs; return ``(grad, matrix)`` arrays."""
    kernel = reduced_elastic_kernels.assemble_elastic_blocks
    args = dict(capture["blocks"])
    args["joint_terms"] = joint_terms
    args["endpoint_terms"] = capture["endpoint"]["endpoint_terms"]
    for name in ("elastic_body_block_grad", "elastic_body_block_delta", "elastic_body_block_matrix"):
        args[name] = wp.zeros_like(args[name])
    wp.launch(kernel, dim=capture["blocks_dim"], inputs=[args[a.label] for a in kernel.adj.args], device=device)
    return args["elastic_body_block_grad"], args["elastic_body_block_matrix"]


def _coupled_storage(layout, device):
    values = wp.zeros(layout.articulation_value_count, dtype=float, device=device)
    rhs = wp.zeros(layout.articulation_scalar_count, dtype=float, device=device)
    return values, rhs


def _assemble(capture, layout, model, device, grad, matrix, values=None, rhs=None):
    """Run :func:`assemble_articulation_variable_system` on captured inputs; return values, rhs, terms."""
    diag = capture["diag"]
    joint = capture["joint"]
    if values is None:
        values, rhs = _coupled_storage(layout, device)
    terms = wp.zeros(layout.mixed_articulation_joints.shape[0], dtype=JointEndpointTerms, device=device)
    assembly.assemble_articulation_variable_system(
        layout,
        model,
        dt=joint["dt"],
        body_q=diag["body_q"],
        body_inv_mass=diag["body_inv_mass"],
        body_inertia_q=diag["body_inertia_q"],
        body_forces=diag["body_forces"],
        body_torques=diag["body_torques"],
        body_hessian_ll=diag["body_hessian_ll"],
        body_hessian_al=diag["body_hessian_al"],
        body_hessian_aa=diag["body_hessian_aa"],
        body_disp=diag["body_disp"],
        body_disp_ang=diag["body_disp_ang"],
        use_displacement=diag["use_displacement"],
        elastic_body_block_grad=grad,
        elastic_body_block_matrix=matrix,
        elastic=joint["elastic"],
        joint_inputs={name: joint[name] for name in assembly.JOINT_TERM_INPUT_NAMES},
        terms=terms,
        values=values,
        rhs=rhs,
        device=device,
    )
    return values.numpy().astype(np.float64), rhs.numpy().astype(np.float64), terms.numpy()


# ---------------------------------------------------------------------------------------------
# NumPy view of the variable-size storage
# ---------------------------------------------------------------------------------------------


class _Layout:
    """Host copies of the coupled layout fields used by the NumPy references."""

    def __init__(self, layout):
        self.layout = layout
        for name in (
            "articulation_body_offsets",
            "articulation_joint_offsets",
            "articulation_bodies",
            "articulation_joint_body_start",
            "articulation_block_row_offsets",
            "articulation_block_cols",
            "articulation_diag_slots",
            "articulation_row_dims",
            "articulation_row_scalar_offsets",
            "articulation_block_value_offsets",
            "articulation_block_dims",
            "body_articulation_local",
            "mixed_articulation_joints",
            "mixed_elastic_bodies",
            "mixed_elastic_body_rows",
        ):
            setattr(self, name, getattr(layout, name).numpy())

    def row_of_body(self, body):
        local = int(self.body_articulation_local[body])
        for a in range(self.layout.articulation_count):
            start, end = self.articulation_body_offsets[a], self.articulation_body_offsets[a + 1]
            if local >= 0 and start + local < end and self.articulation_bodies[start + local] == body:
                return int(start + local)
        return -1

    def block(self, values, slot):
        rows, cols = self.articulation_block_dims[slot]
        offset = self.articulation_block_value_offsets[slot]
        return values[offset : offset + rows * cols].reshape(rows, cols)

    def find_slot(self, row, col_row):
        start = self.articulation_body_offsets[self.articulation_of_row(row)]
        for slot in range(self.articulation_block_row_offsets[row], self.articulation_block_row_offsets[row + 1]):
            if start + self.articulation_block_cols[slot] == col_row:
                return slot
        return -1

    def articulation_of_row(self, row):
        return int(np.searchsorted(self.articulation_body_offsets, row, side="right") - 1)

    def logical_block(self, values, row_i, row_j):
        """Return the logical block ``H[row_i, row_j]`` (stored or transposed), or ``None`` without a slot."""
        if row_i >= row_j:
            slot = self.find_slot(row_i, row_j)
            return None if slot < 0 else self.block(values, slot)
        slot = self.find_slot(row_j, row_i)
        return None if slot < 0 else self.block(values, slot).T

    def vector(self, rhs, row):
        offset = self.articulation_row_scalar_offsets[row]
        return rhs[offset : offset + self.articulation_row_dims[row]]

    def dense(self, values, rhs, articulation):
        """Return the full symmetric matrix and right-hand side of one articulation."""
        b0, b1 = self.articulation_body_offsets[articulation], self.articulation_body_offsets[articulation + 1]
        s0 = self.articulation_row_scalar_offsets[b0]
        n = self.articulation_row_scalar_offsets[b1] - s0
        H = np.zeros((n, n))
        for row in range(b0, b1):
            r0 = self.articulation_row_scalar_offsets[row] - s0
            for slot in range(self.articulation_block_row_offsets[row], self.articulation_block_row_offsets[row + 1]):
                col_row = b0 + self.articulation_block_cols[slot]
                c0 = self.articulation_row_scalar_offsets[col_row] - s0
                blk = self.block(values, slot)
                H[r0 : r0 + blk.shape[0], c0 : c0 + blk.shape[1]] = blk
                if col_row != row:
                    H[c0 : c0 + blk.shape[1], r0 : r0 + blk.shape[0]] = blk.T
        return H, rhs[s0 : s0 + n].copy()


# ---------------------------------------------------------------------------------------------
# Rotation helpers (NumPy, float64)
# ---------------------------------------------------------------------------------------------


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rotvec_to_matrix(v):
    angle = float(np.linalg.norm(v))
    K = _skew(v)
    if angle < 1.0e-12:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + math.sin(angle) / angle * K + (1.0 - math.cos(angle)) / angle**2 * K @ K


def _matrix_to_rotvec(R):
    cos_angle = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    angle = math.acos(cos_angle)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1.0e-8:
        return 0.5 * w
    return angle / (2.0 * math.sin(angle)) * w


def _quat_to_matrix(q):
    x, y, z, w = (float(c) for c in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _left_jacobian(theta):
    angle = float(np.linalg.norm(theta))
    K = _skew(theta)
    if angle < 1.0e-6:
        return np.eye(3) + 0.5 * K + K @ K / 6.0
    return np.eye(3) + (1.0 - math.cos(angle)) / angle**2 * K + (angle - math.sin(angle)) / angle**3 * K @ K


# ---------------------------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------------------------


def _beam(mode_specs, length):
    return newton.ModalGeneratorBeam(
        length=length,
        half_width_y=0.03,
        half_width_z=0.02,
        young_modulus=2.0e6,
        density=1000.0,
        mode_specs=mode_specs,
    ).build()


def _mixed_model(device, *, rigid_articulation: bool = True):
    """Build a rigid pendulum articulation and a mixed chain world - E1 - E2 - R - E3.

    E1, E2 and E3 are reduced elastic links with 3, 2 and 1 modes; R is rigid. The joints are
    revolute (world-E1 with a drive, E1-E2 elastic-elastic, E2-R elastic-rigid) and a ball
    (R-E3, elastic child). The elastic links start deformed and the state is perturbed, so the
    attachments carry nonzero modal rotations and the joints nonzero residuals.
    """
    beam = newton.ModalGeneratorBeam
    length = 0.6
    three = _beam(
        [
            {"type": beam.Mode.AXIAL},
            {"type": beam.Mode.BENDING_Y, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
            {"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
        ],
        length,
    )
    two = _beam(
        [
            {"type": beam.Mode.BENDING_Y, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
            {"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
        ],
        length,
    )
    one = _beam([{"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 1}], length)

    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    inertia = wp.mat33(*np.diag([0.002, 0.01, 0.01]).ravel())
    half = wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity())
    minus_half = wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity())
    if rigid_articulation:
        pendulum = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, -1.0), wp.quat_identity()), mass=0.5, inertia=inertia
        )
        builder.add_articulation(
            [
                builder.add_joint_revolute(
                    parent=-1,
                    child=pendulum,
                    axis=(0.0, 0.0, 1.0),
                    parent_xform=wp.transform(wp.vec3(-0.2, 0.0, -1.0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
                )
            ]
        )
    e1 = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(0.5 * length, 0.0, 0.0), wp.quat_identity()),
        modal_basis=three,
        mode_q=[0.004, -0.01, 0.02],
    )
    e2 = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(1.5 * length, 0.0, 0.0), wp.quat_identity()),
        modal_basis=two,
        mode_q=[0.015, -0.01],
    )
    rigid = builder.add_link(
        xform=wp.transform(wp.vec3(2.0 * length + 0.15, 0.0, 0.0), wp.quat_identity()),
        mass=0.4,
        inertia=inertia,
        com=wp.vec3(0.02, 0.01, -0.005),
    )
    e3 = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(2.5 * length + 0.3, 0.0, 0.0), wp.quat_identity()),
        modal_basis=one,
        mode_q=[0.02],
    )
    axis = (0.0, 0.2, 1.0)
    joints = [
        builder.add_joint_revolute(
            parent=-1, child=e1, axis=axis, child_xform=minus_half, target_ke=20.0, target_kd=0.1, target_pos=0.2
        ),
        builder.add_joint_revolute(parent=e1, child=e2, axis=axis, parent_xform=half, child_xform=minus_half),
        builder.add_joint_revolute(
            parent=e2,
            child=rigid,
            axis=(0.1, 0.0, 1.0),
            parent_xform=half,
            child_xform=wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity()),
        ),
        builder.add_joint_ball(
            parent=rigid,
            child=e3,
            parent_xform=wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity()),
            child_xform=minus_half,
        ),
    ]
    builder.add_articulation(joints)
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[model.joint_qd_start.numpy()[joints[0]]] = 1.2
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Perturb poses (rotation about each COM plus a small shift) so every joint has a residual.
    body_q = state.body_q.numpy().astype(np.float64)
    com = model.body_com.numpy().astype(np.float64)
    for b in range(model.body_count):
        phase = float(b + 1)
        R = _quat_to_matrix(body_q[b, 3:7])
        dR = _rotvec_to_matrix(0.1 * np.array([math.sin(1.1 * phase), math.cos(0.7 * phase), 0.5]))
        c = body_q[b, :3] + R @ com[b] + 0.004 * np.array([math.sin(1.7 * phase), math.cos(2.3 * phase), 0.3])
        R_new = dR @ R
        body_q[b, :3] = c - R_new @ com[b]
        body_q[b, 3:7] = _matrix_to_quat(R_new)
    state.body_q.assign(body_q.astype(np.float32))
    # The solver reads an elastic frame from its owner joint's coordinates (position, quaternion).
    joint_q = state.joint_q.numpy()
    q_start = model.joint_q_start.numpy()
    for body, owner in zip(model.elastic_body.numpy(), model.elastic_joint.numpy(), strict=True):
        joint_q[q_start[owner] : q_start[owner] + 7] = body_q[body]
    state.joint_q.assign(joint_q)
    return model, state, joints


def _matrix_to_quat(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) * 0.5
    x = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 0.5, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) * 0.5, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) * 0.5, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


_FIXTURE_CACHE = {}


def _fixture(device):
    """Return ``(model, joints, capture, layout)`` for ``device`` (built once per device)."""
    key = str(device)
    if key not in _FIXTURE_CACHE:
        model, state, joints = _mixed_model(device)
        capture = _capture(model, state, device)
        layout = build_rigid_articulation_sparse_layout(model, device, elastic_solve="coupled")
        _FIXTURE_CACHE[key] = (model, joints, capture, layout)
    return _FIXTURE_CACHE[key]


# ---------------------------------------------------------------------------------------------
# Elastic endpoint data in NumPy
# ---------------------------------------------------------------------------------------------


class _Elastic:
    """Host copies of the reduced elastic model data and the captured modal coordinates."""

    def __init__(self, model, elastic: SparseElasticJointData):
        self.body_elastic_index = model.body_elastic_index.numpy()
        self.elastic_joint = model.elastic_joint.numpy()
        self.mode_count = model.elastic_mode_count.numpy()
        self.joint_q_start = model.joint_q_start.numpy()
        self.parent_endpoint = model.joint_parent_elastic_endpoint.numpy()
        self.child_endpoint = model.joint_child_elastic_endpoint.numpy()
        self.phi = model.elastic_endpoint_phi.numpy().astype(np.float64)
        self.psi = model.elastic_endpoint_psi.numpy().astype(np.float64)
        self.max_modes = int(model.elastic_max_mode_count)
        self.joint_q = elastic.joint_q.numpy().astype(np.float64)

    def endpoint(self, joint, body, is_parent):
        if body < 0 or self.body_elastic_index[body] < 0:
            return -1, 0
        endpoint = int(self.parent_endpoint[joint] if is_parent else self.child_endpoint[joint])
        if endpoint < 0:
            return -1, 0
        return endpoint, int(self.mode_count[self.body_elastic_index[body]])

    def modes(self, body):
        """Return the modal coordinates of an elastic body (empty for rigid bodies)."""
        if body < 0 or self.body_elastic_index[body] < 0:
            return np.zeros(0)
        index = self.body_elastic_index[body]
        start = self.joint_q_start[self.elastic_joint[index]] + 7
        return self.joint_q[start : start + self.mode_count[index]].copy()

    def shapes(self, endpoint, count):
        base = endpoint * self.max_modes
        return self.phi[base : base + count], self.psi[base : base + count]


def _projection(elastic: _Elastic, joint, body, is_parent, R, rotation_jacobian, offset, row_dim):
    """Return ``J_e = [I_6 | G_e]`` (6 x row_dim) of one endpoint, from the endpoint mode shapes."""
    J = np.zeros((6, row_dim))
    J[:, :6] = np.eye(6)
    endpoint, count = elastic.endpoint(joint, body, is_parent)
    if endpoint < 0:
        return J
    phi, psi = elastic.shapes(endpoint, count)
    RJ = R @ rotation_jacobian
    for k in range(count):
        psi_w = RJ @ psi[k]
        J[:3, 6 + k] = R @ phi[k] + np.cross(offset, psi_w)
        J[3:, 6 + k] = psi_w
    return J


# ---------------------------------------------------------------------------------------------
# Cross block from finite-difference Jacobians of the joint constraint residual
# ---------------------------------------------------------------------------------------------


def _k_eff(material_k, rho):
    """Compliant-ALM row stiffness ``K rho / (K + rho)`` (0 for a retired row)."""
    if material_k <= 0.0 or rho <= 0.0:
        return 0.0
    if math.isinf(material_k):
        return rho
    return material_k * rho / (material_k + rho)


class _JointResidual:
    """Structural residual of a revolute or ball joint as a function of both endpoints' coordinates.

    The coordinates of a body are ``x = [dc, dtheta, dq]``: the world displacement of its center
    of mass, the world rotation vector of ``R_new = exp(dtheta) R`` and the modal increments. The
    attachment frame follows the model's definition: translation ``p + sum phi_k q_k`` and rotation
    ``exp(sum psi_k q_k) q_X`` in body axes. The residual rows are the anchor gap
    ``a_c - a_p`` (world) and, for a revolute joint, the parent-frame relative rotation vector
    ``kappa = log(q_rel q_rel_rest^-1)`` with ``q_rel = q_ap^-1 q_ac``. The row weights are the
    structural stiffness of the selected formulation plus ``kd / dt``, with the free axis projected
    out of ``kappa``. Nothing here reads a solver Hessian or the projection twists.
    """

    def __init__(self, model, capture, elastic: _Elastic, joint):
        args = capture["joint"]
        self.elastic = elastic
        self.joint = joint
        self.type = int(model.joint_type.numpy()[joint])
        self.parent = int(model.joint_parent.numpy()[joint])
        self.child = int(model.joint_child.numpy()[joint])
        self.X_p = model.joint_X_p.numpy()[joint].astype(np.float64)
        self.X_c = model.joint_X_c.numpy()[joint].astype(np.float64)
        self.body_q = args["body_q"].numpy().astype(np.float64)
        self.body_q_rest = args["body_q_rest"].numpy().astype(np.float64)
        self.body_com = model.body_com.numpy().astype(np.float64)
        dt = float(args["dt"])
        c_start = int(args["joint_constraint_start"].numpy()[joint])
        alm = int(args["joint_compliant_alm"])
        penalty = args["joint_penalty_k"].numpy()
        rho = args["joint_rho"].numpy()
        material = args["joint_material_k"].numpy()
        kd = args["joint_penalty_kd"].numpy()

        def weight(row):
            k = _k_eff(float(material[row]), float(rho[row])) if alm == 1 else float(penalty[row])
            if kd[row] > 0.0:
                k += float(kd[row]) / dt
            return k

        self.weights = [weight(c_start)] * 3
        self.P_ang = None
        if self.type == int(newton.JointType.REVOLUTE):
            axis = model.joint_axis.numpy()[model.joint_qd_start.numpy()[joint]].astype(np.float64)
            axis /= np.linalg.norm(axis)
            self.P_ang = np.eye(3) - np.outer(axis, axis)
            self.weights += [weight(c_start + 1)] * 3
        elif self.type != int(newton.JointType.BALL):
            raise ValueError("only revolute and ball joints are modeled")

    def body_dim(self, body):
        return 6 + self.elastic.modes(body).shape[0]

    def _anchor(self, body, is_parent, x):
        X = self.X_p if is_parent else self.X_c
        p_local = X[:3].copy()
        R_X = _quat_to_matrix(X[3:7])
        if body < 0:
            return p_local, R_X
        endpoint, count = self.elastic.endpoint(self.joint, body, is_parent)
        q = self.elastic.modes(body) + x[6:]
        if endpoint >= 0:
            phi, psi = self.elastic.shapes(endpoint, count)
            p_local = p_local + phi.T @ q[:count]
            R_X = _rotvec_to_matrix(psi.T @ q[:count]) @ R_X
        R = _quat_to_matrix(self.body_q[body, 3:7])
        com_world = self.body_q[body, :3] + R @ self.body_com[body]
        R_new = _rotvec_to_matrix(x[3:6]) @ R
        origin = com_world + x[:3] - R_new @ self.body_com[body]
        return origin + R_new @ p_local, R_new @ R_X

    def _rest_rotation(self, body, is_parent):
        X = self.X_p if is_parent else self.X_c
        if body < 0:
            return _quat_to_matrix(X[3:7])
        return _quat_to_matrix(self.body_q_rest[body, 3:7]) @ _quat_to_matrix(X[3:7])

    def residual(self, x_parent, x_child):
        a_p, R_p = self._anchor(self.parent, True, x_parent)
        a_c, R_c = self._anchor(self.child, False, x_child)
        rows = [a_c - a_p]
        if self.P_ang is not None:
            rel = R_p.T @ R_c
            rel_rest = self._rest_rotation(self.parent, True).T @ self._rest_rotation(self.child, False)
            rows.append(self.P_ang @ _matrix_to_rotvec(rel @ rel_rest.T))
        return np.concatenate(rows)

    def jacobians(self, eps=1.0e-6):
        """Central-difference Jacobians of the residual with respect to the parent and child coordinates."""
        n_p = self.body_dim(self.parent) if self.parent >= 0 else 0
        n_c = self.body_dim(self.child)
        x_p0 = np.zeros(n_p)
        x_c0 = np.zeros(n_c)
        J_p = np.zeros((len(self.weights), n_p))
        J_c = np.zeros((len(self.weights), n_c))
        for i in range(n_p):
            d = np.zeros(n_p)
            d[i] = eps
            J_p[:, i] = (self.residual(x_p0 + d, x_c0) - self.residual(x_p0 - d, x_c0)) / (2.0 * eps)
        for i in range(n_c):
            d = np.zeros(n_c)
            d[i] = eps
            J_c[:, i] = (self.residual(x_p0, x_c0 + d) - self.residual(x_p0, x_c0 - d)) / (2.0 * eps)
        return J_p, J_c

    def cross_block(self):
        """Return the Gauss-Newton parent-child block ``J_p^T W J_c``."""
        J_p, J_c = self.jacobians()
        return J_p.T @ np.diag(self.weights) @ J_c


_BLOCK_PARTS = {
    "frame-frame": (slice(0, 6), slice(0, 6)),
    "frame-mode": (slice(0, 6), slice(6, None)),
    "mode-frame": (slice(6, None), slice(0, 6)),
    "mode-mode": (slice(6, None), slice(6, None)),
}


def _assert_block_parts(test, actual, expected, label, rtol=2.0e-5):
    """Compare the frame-frame, frame-mode, mode-frame and mode-mode parts, each against its own scale.

    A part whose reference is exactly zero (for example the modal translation of a pinned-pinned
    bending mode at a ball joint) is compared against the scale of the whole block.
    """
    whole = float(np.max(np.abs(expected)))
    test.assertGreater(whole, 0.0, label)
    for name, (rows, cols) in _BLOCK_PARTS.items():
        a = actual[rows, cols]
        e = expected[rows, cols]
        if e.size == 0:
            continue
        scale = float(np.max(np.abs(e)))
        if scale < 1.0e-9 * whole:
            scale = whole
        error = float(np.max(np.abs(a - e)))
        test.assertLessEqual(error, rtol * scale, f"{label} {name}: max error {error:.3e} vs scale {scale:.3e}")


def test_cross_block_matches_constraint_jacobians(test, device):
    """The assembled cross block equals ``J_p^T W J_c`` from finite differences of the residual.

    The reference uses only the joint residual (no projection twists, no solver Hessian). The
    angular rows' ``J_r^-1(kappa)`` is the small-angle series of the existing joint evaluation,
    whose relative error ``O(|kappa|^4 / 720)`` stays below the tolerance for the fixture's
    rotations of about 0.1 rad.
    """
    model, joints, capture, layout = _fixture(device)
    grad, matrix = _elastic_blocks(capture, device, joint_terms=False)
    values, _rhs, _terms = _assemble(capture, layout, model, device, grad, matrix)
    L = _Layout(layout)
    elastic = _Elastic(model, capture["joint"]["elastic"])
    checked = []
    for joint, label in ((joints[1], "elastic-elastic"), (joints[2], "elastic-rigid"), (joints[3], "rigid-elastic")):
        residual = _JointResidual(model, capture, elastic, joint)
        expected = residual.cross_block()
        row_p = L.row_of_body(residual.parent)
        row_c = L.row_of_body(residual.child)
        actual = L.logical_block(values, row_p, row_c)
        test.assertIsNotNone(actual, label)
        test.assertEqual(actual.shape, expected.shape, label)
        _assert_block_parts(test, actual, expected, label)
        checked.append((label, row_p > row_c))
        if label == "elastic-elastic":
            # The mode-mode cross block is the coupling the Schur backend cannot represent.
            test.assertGreater(float(np.max(np.abs(expected[6:, 6:]))), 1.0e-3 * float(np.max(np.abs(expected))))
    # Both storage orientations (direct and transposed) are exercised by the fixture.
    test.assertEqual({orientation for _label, orientation in checked}, {True, False}, checked)


# ---------------------------------------------------------------------------------------------
# Projection kernel vs NumPy blockdiag + sum J^T K J (synthetic terms)
# ---------------------------------------------------------------------------------------------


def _random_terms(rng, model, layout, L, inactive):
    """Return random endpoint terms for the mixed joints; joints in ``inactive`` are not evaluated."""
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    terms = np.zeros(layout.mixed_articulation_joints.shape[0], dtype=JointEndpointTerms.numpy_dtype())
    for j, joint in enumerate(L.mixed_articulation_joints):
        t = terms[j]
        for name in ("wrench_parent", "wrench_child"):
            t[name] = rng.normal(size=6)
        for name in ("hessian_parent", "hessian_child"):
            A = rng.normal(size=(6, 6))
            t[name] = A @ A.T
        t["hessian_cross"] = rng.normal(size=(6, 6))
        t["attachment_offset_parent"] = rng.normal(scale=0.3, size=3)
        t["attachment_offset_child"] = rng.normal(scale=0.3, size=3)
        t["rotation_jacobian_parent"] = _left_jacobian(rng.normal(scale=0.2, size=3))
        t["rotation_jacobian_child"] = _left_jacobian(rng.normal(scale=0.2, size=3))
        t["parent"] = joint_parent[joint]
        t["child"] = joint_child[joint]
        t["evaluated"] = 0 if j in inactive else 1
    return terms


def _numpy_projection(model, L, elastic, body_q, terms, H, g, articulation):
    """Add ``sum_j J^T K J`` and ``J^T w`` of the evaluated joints to one articulation's dense system."""
    b0 = L.articulation_body_offsets[articulation]
    s0 = L.articulation_row_scalar_offsets[b0]

    def rows(row):
        start = L.articulation_row_scalar_offsets[row] - s0
        return slice(start, start + L.articulation_row_dims[row])

    for j, joint in enumerate(L.mixed_articulation_joints):
        t = terms[j]
        if not t["evaluated"] or L.articulation_of_row(L.row_of_body(t["child"])) != articulation:
            continue
        endpoints = []
        for is_parent, prefix in ((True, "parent"), (False, "child")):
            body = int(t[prefix])
            if body < 0:
                endpoints.append(None)
                continue
            row = L.row_of_body(body)
            R = _quat_to_matrix(body_q[body, 3:7])
            J = _projection(
                elastic,
                joint,
                body,
                is_parent,
                R,
                t["rotation_jacobian_" + prefix].astype(np.float64),
                t["attachment_offset_" + prefix].astype(np.float64),
                L.articulation_row_dims[row],
            )
            K = t["hessian_" + prefix].astype(np.float64)
            g[rows(row)] += J.T @ t["wrench_" + prefix].astype(np.float64)
            H[rows(row), rows(row)] += J.T @ K @ J
            endpoints.append((row, J))
        if endpoints[0] is not None:
            (row_p, J_p), (row_c, J_c) = endpoints
            cross = J_p.T @ t["hessian_cross"].astype(np.float64) @ J_c
            H[rows(row_p), rows(row_c)] += cross
            H[rows(row_c), rows(row_p)] += cross.T


def test_projection_matches_numpy(test, device):
    """Projected joint terms over a random block diagonal equal NumPy ``blockdiag + sum J^T K J``."""
    model, _joints, capture, layout = _fixture(device)
    L = _Layout(layout)
    rng = np.random.default_rng(7)
    elastic = capture["joint"]["elastic"]
    elastic_np = _Elastic(model, elastic)

    # Random rotations and a random symmetric block diagonal; off-diagonal slots start at zero.
    body_q = capture["joint"]["body_q"].numpy().astype(np.float64)
    for b in range(body_q.shape[0]):
        body_q[b, 3:7] = _matrix_to_quat(_rotvec_to_matrix(rng.normal(scale=0.8, size=3)))
    values = np.zeros(layout.articulation_value_count)
    rhs = rng.normal(size=layout.articulation_scalar_count)
    for row in range(layout.rigid_body_row_count, layout.articulation_body_count):
        n = L.articulation_row_dims[row]
        A = rng.normal(size=(n, n))
        L.block(values, L.articulation_diag_slots[row])[:] = A @ A.T + n * np.eye(n)
    inactive = {2}
    terms = _random_terms(rng, model, layout, L, inactive)

    values_wp = wp.array(values.astype(np.float32), dtype=float, device=device)
    rhs_wp = wp.array(rhs.astype(np.float32), dtype=float, device=device)
    assembly.launch_articulation_joint_terms_variable(
        layout,
        wp.array(body_q.astype(np.float32), dtype=wp.transform, device=device),
        elastic,
        wp.array(terms, dtype=JointEndpointTerms, device=device),
        values_wp,
        rhs_wp,
        device,
    )
    assembled = values_wp.numpy().astype(np.float64)
    assembled_rhs = rhs_wp.numpy().astype(np.float64)
    # The rows of the rigid prefix are never touched.
    prefix_rhs = 6 * layout.rigid_body_row_count
    np.testing.assert_array_equal(assembled_rhs[:prefix_rhs], rhs.astype(np.float32)[:prefix_rhs])
    np.testing.assert_array_equal(assembled[: 36 * layout.rigid_block_count], 0.0)

    body_q_f32 = body_q.astype(np.float32).astype(np.float64)
    terms_used = terms.copy()
    for articulation in range(layout.rigid_articulation_count, layout.articulation_count):
        H_ref, g_ref = L.dense(
            values.astype(np.float32).astype(np.float64), rhs.astype(np.float32).astype(np.float64), articulation
        )
        _numpy_projection(model, L, elastic_np, body_q_f32, terms_used, H_ref, g_ref, articulation)
        H, g = L.dense(assembled, assembled_rhs, articulation)
        scale = float(np.max(np.abs(H_ref)))
        np.testing.assert_allclose(H, H_ref, rtol=0.0, atol=2.0e-6 * scale)
        np.testing.assert_allclose(g, g_ref, rtol=0.0, atol=2.0e-6 * float(np.max(np.abs(g_ref))))
    # The active joints exercise both cross-slot orientations and an elastic-elastic mode-mode block.
    orientations = set()
    elastic_pairs = 0
    for j, joint in enumerate(L.mixed_articulation_joints):
        t = terms[j]
        if t["parent"] < 0 or j in inactive:
            continue
        row_p, row_c = L.row_of_body(int(t["parent"])), L.row_of_body(int(t["child"]))
        orientations.add(bool(row_p > row_c))
        if (
            elastic_np.endpoint(joint, int(t["parent"]), True)[1]
            and elastic_np.endpoint(joint, int(t["child"]), False)[1]
        ):
            elastic_pairs += 1
            block = L.logical_block(assembled, row_p, row_c)
            test.assertGreater(float(np.max(np.abs(block[6:, 6:]))), 0.0)
    test.assertEqual(orientations, {True, False})
    test.assertGreater(elastic_pairs, 0)

    # An unevaluated joint adds nothing: switching it on changes the system.
    terms_on = terms.copy()
    terms_on["evaluated"][list(inactive)] = 1
    values_on = wp.array(values.astype(np.float32), dtype=float, device=device)
    rhs_on = wp.array(rhs.astype(np.float32), dtype=float, device=device)
    assembly.launch_articulation_joint_terms_variable(
        layout,
        wp.array(body_q.astype(np.float32), dtype=wp.transform, device=device),
        elastic,
        wp.array(terms_on, dtype=JointEndpointTerms, device=device),
        values_on,
        rhs_on,
        device,
    )
    test.assertFalse(np.allclose(values_on.numpy(), values_wp.numpy()))


# ---------------------------------------------------------------------------------------------
# Elastic body blocks: Schur frame overwrite, local block, joint switch, static frames
# ---------------------------------------------------------------------------------------------


def _dense_elastic(matrix, elastic_index, width):
    upper = np.triu(matrix[elastic_index, :width, :width].astype(np.float64))
    return upper + np.triu(upper, 1).T


def _launch_scatter(layout, model, capture, grad, matrix, values, rhs, device, body_inv_mass=None):
    wp.launch(
        assembly.scatter_elastic_body_block_variable,
        dim=(layout.mixed_elastic_body_count, layout.articulation_max_row_dim),
        inputs=[
            layout.mixed_elastic_bodies,
            layout.mixed_elastic_body_rows,
            layout.articulation_row_dims,
            layout.articulation_row_scalar_offsets,
            layout.articulation_block_value_offsets,
            layout.articulation_block_dims,
            layout.articulation_diag_slots,
            model.body_elastic_index,
            capture["diag"]["body_q"],
            body_inv_mass if body_inv_mass is not None else capture["diag"]["body_inv_mass"],
            grad,
            matrix,
        ],
        outputs=[values, rhs],
        device=device,
    )


def test_elastic_frame_rows_match_schur_frame_overwrite(test, device):
    """The world frame rows of an elastic block equal the Schur backend's frame overwrite; modal rows are ``T A T^T``."""
    model, _joints, capture, layout = _fixture(device)
    grad, matrix = _elastic_blocks(capture, device, joint_terms=False)
    values, rhs = _coupled_storage(layout, device)
    _launch_scatter(layout, model, capture, grad, matrix, values, rhs, device)

    frame = dict(capture["frame"])
    frame["elastic_body_block_grad"] = grad
    frame["elastic_body_block_matrix"] = matrix
    frame["values_scalar"] = wp.zeros_like(frame["values_scalar"])
    frame["rhs_scalar"] = wp.zeros_like(frame["rhs_scalar"])
    kernel = reduced_elastic_sparse_kernels.assemble_elastic_sparse_frame_blocks
    wp.launch(kernel, dim=capture["frame_dim"], inputs=[frame[a.label] for a in kernel.adj.args], device=device)

    L = _Layout(layout)
    values_np = values.numpy().astype(np.float64)
    rhs_np = rhs.numpy().astype(np.float64)
    schur_values = frame["values_scalar"].numpy().astype(np.float64)
    schur_rhs = frame["rhs_scalar"].numpy().astype(np.float64)
    schur_bodies = list(frame["elastic_bodies"].numpy())
    schur_row_start = frame["elastic_body_row_start"].numpy()
    schur_local = frame["body_articulation_local"].numpy()
    schur_diag = frame["articulation_diag_slots"].numpy()
    body_q = capture["diag"]["body_q"].numpy().astype(np.float64)
    body_elastic_index = model.body_elastic_index.numpy()
    grad_np = grad.numpy().astype(np.float64)
    matrix_np = matrix.numpy()
    test.assertEqual(sorted(schur_bodies), sorted(L.mixed_elastic_bodies.tolist()))
    for entry, body in enumerate(L.mixed_elastic_bodies):
        row = L.mixed_elastic_body_rows[entry]
        n = L.articulation_row_dims[row]
        block = L.block(values_np, L.articulation_diag_slots[row])
        vec = L.vector(rhs_np, row)
        tid = schur_bodies.index(body)
        schur_row = schur_row_start[tid] + schur_local[body]
        schur_block = schur_values[36 * schur_diag[schur_row] : 36 * schur_diag[schur_row] + 36].reshape(6, 6)
        scale = float(np.max(np.abs(schur_block)))
        np.testing.assert_allclose(block[:6, :6], schur_block, rtol=0.0, atol=2.0e-6 * scale)
        np.testing.assert_allclose(
            vec[:6], schur_rhs[6 * schur_row : 6 * schur_row + 6], rtol=0.0, atol=2.0e-6 * np.max(np.abs(vec[:6]))
        )

        # Modal rows: T A T^T with T = diag(R, R, I) and -T g, from the body-axes block.
        index = body_elastic_index[body]
        A = _dense_elastic(matrix_np, index, n)
        R = _quat_to_matrix(body_q[body, 3:7])
        T = np.eye(n)
        T[:3, :3] = R
        T[3:6, 3:6] = R
        expected = T @ A @ T.T
        np.testing.assert_array_equal(block[6:, 6:], A[6:, 6:])
        np.testing.assert_allclose(block[:6, 6:], expected[:6, 6:], rtol=0.0, atol=2.0e-6 * np.max(np.abs(A[:6, 6:])))
        np.testing.assert_allclose(block[6:, :6], expected[6:, :6], rtol=0.0, atol=2.0e-6 * np.max(np.abs(A[:6, 6:])))
        np.testing.assert_array_equal(vec[6:], -grad_np[index, 6:n])
        # Symmetric to rounding.
        np.testing.assert_allclose(block, block.T, rtol=0.0, atol=2.0e-6 * np.max(np.abs(block)))


def test_coupled_elastic_rows_match_local_block_with_joints(test, device):
    """Elastic block without joints plus the projected joint self terms equals the local block with joints.

    The local block adds its joints in attachment-referred form, which equals ``J^T K J`` for every
    joint row whose moment is taken about the attachment (all joints here), so the coupled diagonal
    block and right-hand side of each elastic body reproduce ``T A_local T^T`` and ``-T g_local``.
    """
    model, _joints, capture, layout = _fixture(device)
    grad_off, matrix_off = _elastic_blocks(capture, device, joint_terms=False)
    grad_on, matrix_on = _elastic_blocks(capture, device, joint_terms=True)
    values, rhs, _terms = _assemble(capture, layout, model, device, grad_off, matrix_off)
    L = _Layout(layout)
    body_q = capture["diag"]["body_q"].numpy().astype(np.float64)
    body_elastic_index = model.body_elastic_index.numpy()
    matrix_on_np = matrix_on.numpy()
    matrix_off_np = matrix_off.numpy()
    grad_on_np = grad_on.numpy().astype(np.float64)
    for entry, body in enumerate(L.mixed_elastic_bodies):
        row = L.mixed_elastic_body_rows[entry]
        n = L.articulation_row_dims[row]
        index = body_elastic_index[body]
        A_on = _dense_elastic(matrix_on_np, index, n)
        A_off = _dense_elastic(matrix_off_np, index, n)
        # The switch only removes joint terms: the blocks differ.
        test.assertGreater(float(np.max(np.abs(A_on - A_off))), 0.0)
        R = _quat_to_matrix(body_q[body, 3:7])
        T = np.eye(n)
        T[:3, :3] = R
        T[3:6, 3:6] = R
        expected = T @ A_on @ T.T
        block = L.block(values, L.articulation_diag_slots[row])
        # Each part is compared against the size of the joint terms it holds (or of all joint terms).
        joint_part = expected - T @ A_off @ T.T
        whole = float(np.max(np.abs(joint_part)))
        test.assertGreater(whole, 0.0)
        for name, (rows, cols) in _BLOCK_PARTS.items():
            scale = max(float(np.max(np.abs(joint_part[rows, cols]))), 1.0e-3 * whole)
            # float32 rounding of the summed entries adds a few ulp of the total entry size.
            rounding = 2.0e-6 * float(np.max(np.abs(expected[rows, cols])))
            error = float(np.max(np.abs(block[rows, cols] - expected[rows, cols])))
            test.assertLessEqual(
                error, 2.0e-4 * scale + rounding, f"body {body} {name}: {error:.3e} vs joint scale {scale:.3e}"
            )
        g_expected = -(T @ grad_on_np[index, :n])
        g = L.vector(rhs, row)
        g_scale = float(np.max(np.abs(g_expected - (-(T @ grad_off.numpy().astype(np.float64)[index, :n])))))
        test.assertLessEqual(float(np.max(np.abs(g - g_expected))), 2.0e-4 * g_scale + 1.0e-6)


def test_rigid_rows_match_legacy_body_diagonal(test, device):
    """A rigid row of a mixed articulation gets exactly the legacy 6 x 6 body diagonal and right-hand side."""
    model, _joints, capture, layout = _fixture(device)
    diag = dict(capture["diag"])
    diag["values_scalar"] = wp.zeros_like(diag["values_scalar"])
    diag["rhs_scalar"] = wp.zeros_like(diag["rhs_scalar"])
    kernel = sparse_kernels.assemble_articulation_body_diagonal_scalar
    wp.launch(kernel, dim=capture["diag_dim"], inputs=[diag[a.label] for a in kernel.adj.args], device=device)
    legacy_values = diag["values_scalar"].numpy()
    legacy_rhs = diag["rhs_scalar"].numpy()
    legacy_bodies = list(diag["articulation_bodies"].numpy())
    legacy_diag = diag["articulation_diag_slots"].numpy()

    values, rhs = _coupled_storage(layout, device)
    _launch_body_diagonal(layout, model, capture, values, rhs, device)
    L = _Layout(layout)
    values_np = values.numpy()
    rhs_np = rhs.numpy()
    body_elastic_index = model.body_elastic_index.numpy()
    rigid_rows = 0
    for row in range(layout.rigid_body_row_count, layout.articulation_body_count):
        body = L.articulation_bodies[row]
        block = L.block(values_np, L.articulation_diag_slots[row])
        if body_elastic_index[body] >= 0:
            np.testing.assert_array_equal(block, 0.0)
            np.testing.assert_array_equal(L.vector(rhs_np, row), 0.0)
            continue
        rigid_rows += 1
        legacy_row = legacy_bodies.index(body)
        slot = legacy_diag[legacy_row]
        np.testing.assert_array_equal(block, legacy_values[36 * slot : 36 * slot + 36].reshape(6, 6))
        np.testing.assert_array_equal(L.vector(rhs_np, row), legacy_rhs[6 * legacy_row : 6 * legacy_row + 6])
    test.assertEqual(rigid_rows, 1)


def _launch_body_diagonal(layout, model, capture, values, rhs, device, body_inv_mass=None):
    diag = capture["diag"]
    wp.launch(
        assembly.assemble_articulation_body_diagonal_variable,
        dim=layout.articulation_body_count - layout.rigid_body_row_count,
        inputs=[
            diag["dt"],
            layout.rigid_body_row_count,
            layout.articulation_bodies,
            layout.articulation_diag_slots,
            layout.articulation_row_scalar_offsets,
            layout.articulation_block_value_offsets,
            layout.articulation_block_dims,
            model.body_elastic_index,
            diag["body_q"],
            model.body_mass,
            body_inv_mass if body_inv_mass is not None else diag["body_inv_mass"],
            model.body_com,
            model.body_inertia,
            diag["body_inertia_q"],
            diag["body_forces"],
            diag["body_torques"],
            diag["body_hessian_ll"],
            diag["body_hessian_al"],
            diag["body_hessian_aa"],
            diag["body_disp"],
            diag["body_disp_ang"],
            diag["use_displacement"],
        ],
        outputs=[values, rhs],
        device=device,
    )


def test_static_frames(test, device):
    """Static frames get ``1e30 I_6``, a zero frame right-hand side and no elastic frame-mode coupling."""
    model, _joints, capture, layout = _fixture(device)
    grad, matrix = _elastic_blocks(capture, device, joint_terms=False)
    static = wp.zeros_like(capture["diag"]["body_inv_mass"])
    values, rhs = _coupled_storage(layout, device)
    _launch_scatter(layout, model, capture, grad, matrix, values, rhs, device, body_inv_mass=static)
    _launch_body_diagonal(layout, model, capture, values, rhs, device, body_inv_mass=static)
    L = _Layout(layout)
    values_np = values.numpy().astype(np.float64)
    rhs_np = rhs.numpy().astype(np.float64)
    matrix_np = matrix.numpy()
    body_elastic_index = model.body_elastic_index.numpy()
    for row in range(layout.rigid_body_row_count, layout.articulation_body_count):
        body = L.articulation_bodies[row]
        block = L.block(values_np, L.articulation_diag_slots[row])
        np.testing.assert_array_equal(block[:6, :6], 1.0e30 * np.eye(6, dtype=np.float32))
        np.testing.assert_array_equal(L.vector(rhs_np, row)[:6], 0.0)
        np.testing.assert_array_equal(block[:6, 6:], 0.0)
        np.testing.assert_array_equal(block[6:, :6], 0.0)
        if body_elastic_index[body] >= 0:
            n = L.articulation_row_dims[row]
            np.testing.assert_array_equal(block[6:, 6:], _dense_elastic(matrix_np, body_elastic_index[body], n)[6:, 6:])


def test_assembly_leaves_rigid_prefix_and_validates_inputs(test, device):
    """The helper writes only mixed rows, assembles symmetric diagonal blocks and checks its inputs."""
    model, _joints, capture, layout = _fixture(device)
    test.assertGreater(layout.rigid_block_count, 0)
    grad, matrix = _elastic_blocks(capture, device, joint_terms=False)
    values = wp.full(layout.articulation_value_count, 7.0, dtype=float, device=device)
    rhs = wp.full(layout.articulation_scalar_count, 7.0, dtype=float, device=device)
    values_np, rhs_np, terms = _assemble(capture, layout, model, device, grad, matrix, values, rhs)
    np.testing.assert_array_equal(values_np[: 36 * layout.rigid_block_count], 7.0)
    np.testing.assert_array_equal(rhs_np[: 6 * layout.rigid_body_row_count], 7.0)
    test.assertTrue(np.all(terms["evaluated"] == 1))
    L = _Layout(layout)
    for row in range(layout.rigid_body_row_count, layout.articulation_body_count):
        block = L.block(values_np, L.articulation_diag_slots[row])
        np.testing.assert_allclose(block, block.T, rtol=0.0, atol=2.0e-6 * np.max(np.abs(block)))
        test.assertTrue(np.all(np.isfinite(block)))
    # The assembled mixed system is symmetric positive definite (before the frame regularization).
    for articulation in range(layout.rigid_articulation_count, layout.articulation_count):
        H, _g = L.dense(values_np, rhs_np, articulation)
        test.assertGreater(float(np.min(np.linalg.eigvalsh(0.5 * (H + H.T)))), 0.0)

    joint_inputs = {name: capture["joint"][name] for name in assembly.JOINT_TERM_INPUT_NAMES}
    common = {
        "dt": 0.01,
        "body_q": None,
        "body_inv_mass": None,
        "body_inertia_q": None,
        "body_forces": None,
        "body_torques": None,
        "body_hessian_ll": None,
        "body_hessian_al": None,
        "body_hessian_aa": None,
        "body_disp": None,
        "body_disp_ang": None,
        "use_displacement": 0,
        "elastic_body_block_grad": grad,
        "elastic_body_block_matrix": matrix,
        "elastic": capture["joint"]["elastic"],
        "terms": None,
        "values": values,
        "rhs": rhs,
    }
    schur_layout = build_rigid_articulation_sparse_layout(model, device, elastic_solve="schur")
    with test.assertRaisesRegex(ValueError, "coupled"):
        assembly.assemble_articulation_variable_system(schur_layout, model, joint_inputs=joint_inputs, **common)
    partial = dict(joint_inputs)
    partial.pop("joint_type")
    with test.assertRaisesRegex(ValueError, "joint_type"):
        assembly.assemble_articulation_variable_system(layout, model, joint_inputs=partial, **common)


class TestVBDSparseVariableAssembly(unittest.TestCase):
    pass


add_function_test(
    TestVBDSparseVariableAssembly,
    "test_cross_block_matches_constraint_jacobians",
    test_cross_block_matches_constraint_jacobians,
    devices=devices,
)
add_function_test(
    TestVBDSparseVariableAssembly, "test_projection_matches_numpy", test_projection_matches_numpy, devices=devices
)
add_function_test(
    TestVBDSparseVariableAssembly,
    "test_elastic_frame_rows_match_schur_frame_overwrite",
    test_elastic_frame_rows_match_schur_frame_overwrite,
    devices=devices,
)
add_function_test(
    TestVBDSparseVariableAssembly,
    "test_coupled_elastic_rows_match_local_block_with_joints",
    test_coupled_elastic_rows_match_local_block_with_joints,
    devices=devices,
)
add_function_test(
    TestVBDSparseVariableAssembly,
    "test_rigid_rows_match_legacy_body_diagonal",
    test_rigid_rows_match_legacy_body_diagonal,
    devices=devices,
)
add_function_test(TestVBDSparseVariableAssembly, "test_static_frames", test_static_frames, devices=devices)
add_function_test(
    TestVBDSparseVariableAssembly,
    "test_assembly_leaves_rigid_prefix_and_validates_inputs",
    test_assembly_leaves_rigid_prefix_and_validates_inputs,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
