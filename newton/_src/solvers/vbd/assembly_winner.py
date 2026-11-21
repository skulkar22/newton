# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import IntEnum

from typing import Any

import json

import torch

import warp as wp

wp.init()

wp.set_module_options({"enable_backward": False, "block_dim": 32})

@wp.struct
class JointCoordinateData:
    body_com: wp.array[wp.vec3]
    joint_type: wp.array[int]
    parent: wp.array[int]
    child: wp.array[int]
    X_p: wp.array[wp.transform]
    X_c: wp.array[wp.transform]
    q_start: wp.array[int]
    qd_start: wp.array[int]
    dof_dim: wp.array2d[int]
    axis: wp.array[wp.vec3]
    world: wp.array[int]
    history: wp.array[float]
    body_elastic_index: wp.array[wp.int32]
    elastic_joint: wp.array[wp.int32]
    elastic_mode_count: wp.array[wp.int32]
    parent_elastic_endpoint: wp.array[wp.int32]
    child_elastic_endpoint: wp.array[wp.int32]
    elastic_endpoint_phi: wp.array[wp.vec3]
    elastic_endpoint_psi: wp.array[wp.vec3]
    elastic_max_mode_count: int
    elastic_q: wp.array[float]
    elastic_q_prev: wp.array[float]

@wp.struct
class SparseElasticJointData:
    """Reduced elastic endpoint data for joints in the block-sparse articulation solve.

    Joints attached to a reduced elastic body use the deformed attachment frame, and joints
    whose cross block feeds a modal Schur elimination also write their parent-child Hessian
    block to ``joint_cross_slot``. With ``enabled == 0`` every joint uses its rigid frames.
    """
    enabled: int
    body_elastic_index: wp.array[wp.int32]
    elastic_joint: wp.array[wp.int32]
    elastic_mode_count: wp.array[wp.int32]
    joint_q_start: wp.array[int]
    parent_endpoint: wp.array[wp.int32]
    child_endpoint: wp.array[wp.int32]
    endpoint_phi: wp.array[wp.vec3]
    endpoint_psi: wp.array[wp.vec3]
    max_mode_count: int
    joint_q: wp.array[float]
    'Modal coordinates that pair with the current body poses.'
    joint_q_prev: wp.array[float]
    'Modal coordinates that pair with the previous-step body poses.'
    joint_cross_slot: wp.array[wp.int32]

class JointType(IntEnum):
    PRISMATIC = 0
    REVOLUTE = 1
    BALL = 2
    FIXED = 3
    FREE = 4
    DISTANCE = 5
    D6 = 6
    ROD = 7
    ELASTIC = 8

_SPARSE_BLOCK_DIM = wp.constant(6)

_SPARSE_BLOCK_SIZE = wp.constant(36)

@wp.func
def _scalar_block_index(slot: int, row: int, col: int) -> int:
    return slot * _SPARSE_BLOCK_SIZE + row * _SPARSE_BLOCK_DIM + col

class mat66f(wp.types.matrix(shape=(6, 6), dtype=wp.float32)):
    pass

@wp.func
def _add_mat66_scalar_atomic(values_scalar: wp.array[float], slot: int, block: mat66f, stripe: int, lane_mask: int):
    if slot >= 0:
        for i in range(6):
            for j in range(6):
                if i * 6 + j & lane_mask == stripe and block[i, j] != 0.0:
                    wp.atomic_add(values_scalar, _scalar_block_index(slot, i, j), block[i, j])

@wp.func
def _scalar_vec_index(row: int, comp: int) -> int:
    return row * _SPARSE_BLOCK_DIM + comp

class vec6f(wp.types.vector(length=6, dtype=wp.float32)):
    pass

@wp.func
def _add_rhs_scalar_atomic(rhs_scalar: wp.array[float], index: int, value: vec6f, stripe: int, lane_mask: int):
    for i in range(6):
        if i & lane_mask == stripe:
            wp.atomic_add(rhs_scalar, _scalar_vec_index(index, i), value[i])

@wp.func
def _find_block_slot(articulation_block_row_offsets: wp.array[wp.int32], articulation_block_cols: wp.array[wp.int32], body_start: int, row_local: int, col_local: int) -> int:
    row = body_start + row_local
    begin = articulation_block_row_offsets[row]
    hi = articulation_block_row_offsets[row + 1]
    if hi > begin:
        last = hi - 1
        if articulation_block_cols[last] == col_local:
            return last
        if last > begin and articulation_block_cols[last - 1] == col_local:
            return last - 1
        if articulation_block_cols[begin] == col_local:
            return begin
    lo = begin
    end = hi
    while lo < hi:
        mid = (lo + hi) // 2
        if articulation_block_cols[mid] < col_local:
            lo = mid + 1
        else:
            hi = mid
    if lo < end and articulation_block_cols[lo] == col_local:
        return lo
    return -1

@wp.func
def _mat66_from_blocks(ll: wp.mat33, al: wp.mat33, aa: wp.mat33) -> mat66f:
    A = mat66f(0.0)
    for i in range(3):
        for j in range(3):
            A[i, j] = ll[i, j]
            A[i + 3, j] = al[i, j]
            A[i, j + 3] = al[j, i]
            A[i + 3, j + 3] = aa[i, j]
    return A

@wp.func
def _vec6_from_parts(linear: wp.vec3, angular: wp.vec3) -> vec6f:
    return vec6f(linear[0], linear[1], linear[2], angular[0], angular[1], angular[2])

@wp.func
def _add_rod_body_contribution_scalar(values_scalar: wp.array[float], rhs_scalar: wp.array[float], articulation_block_row_offsets: wp.array[wp.int32], articulation_block_cols: wp.array[wp.int32], body_start: int, local_body: int, force: wp.vec3, torque: wp.vec3, H_ll: wp.mat33, H_al: wp.mat33, H_aa: wp.mat33, stripe: int, lane_mask: int):
    row = body_start + local_body
    for component in range(3):
        if (component & lane_mask) == stripe:
            wp.atomic_add(rhs_scalar, _scalar_vec_index(row, component), force[component])
        if ((component + 3) & lane_mask) == stripe:
            wp.atomic_add(rhs_scalar, _scalar_vec_index(row, component + 3), torque[component])
    slot = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, local_body, local_body)
    _rigid_add_tile(values_scalar, slot, 0, 0, H_ll, False, stripe, lane_mask)
    _rigid_add_tile(values_scalar, slot, 3, 0, H_al, False, stripe, lane_mask)
    _rigid_add_tile(values_scalar, slot, 3, 0, H_al, True, stripe, lane_mask)
    _rigid_add_tile(values_scalar, slot, 3, 3, H_aa, False, stripe, lane_mask)

@wp.func
def _add_rod_cross_block_scalar(values_scalar: wp.array[float], articulation_block_row_offsets: wp.array[wp.int32], articulation_block_cols: wp.array[wp.int32], body_start: int, parent_local: int, child_local: int, cross: mat66f, cross_slot: int, stripe: int, lane_mask: int):
    slot = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, wp.max(parent_local, child_local), wp.min(parent_local, child_local))
    for rr in range(6):
        for cc in range(6):
            offset = rr * 6 + cc
            if offset & lane_mask == stripe:
                value = cross[rr, cc]
                if value != 0.0:
                    if slot >= 0:
                        dst = cc * 6 + rr if parent_local < child_local else offset
                        wp.atomic_add(values_scalar, slot * 36 + dst, value)
                    if cross_slot >= 0:
                        wp.atomic_add(values_scalar, cross_slot * 36 + offset, value)


@wp.func
def _mat66_from_angular_block(aa: wp.mat33) -> mat66f:
    A = mat66f(0.0)
    for i in range(3):
        for j in range(3):
            A[i + 3, j + 3] = aa[i, j]
    return A

@wp.func
def _angular_direct_pair_blocks(torque_parent: wp.vec3, H_aa: wp.mat33):
    """Return ``(rhs_parent, rhs_child, H_self, H_cross)`` of an angular term with parent torque ``torque_parent``.

    The child receives ``-torque_parent``; both self blocks are ``H_aa`` and the parent-child block
    is ``-H_aa`` (angular rows and columns only).
    """
    rhs_parent = _vec6_from_parts(wp.vec3(0.0), torque_parent)
    rhs_child = _vec6_from_parts(wp.vec3(0.0), -torque_parent)
    H_block = _mat66_from_angular_block(H_aa)
    H_cross = _mat66_from_angular_block(-H_aa)
    return (rhs_parent, rhs_child, H_block, H_cross)

@wp.func
def _rigid_add_rhs3(rhs: wp.array[float], row: int, offset: int, value: wp.vec3, stripe: int, lane_mask: int):
    base = row * 6 + offset
    for component in range(3):
        if offset + component & lane_mask == stripe and value[component] != 0.0:
            wp.atomic_add(rhs, base + component, value[component])

@wp.func
def _rigid_add_tile(values: wp.array[float], slot: int, row0: int, col0: int, tile: wp.mat33, flip: bool, stripe: int, lane_mask: int):
    if slot >= 0:
        base = slot * 36 + row0 * 6 + col0
        row_stride = int(6)
        col_stride = int(1)
        if flip:
            base = slot * 36 + col0 * 6 + row0
            row_stride = 1
            col_stride = 6
        for row in range(3):
            for col in range(3):
                value = tile[row, col]
                if ((col0 + col) * 6 + row0 + row if flip else (row0 + row) * 6 + col0 + col) & lane_mask == stripe and value != 0.0:
                    wp.atomic_add(values, base + row * row_stride + col * col_stride, value)

@wp.func
def _assemble_angular_direct_pair_scalar(values_scalar: wp.array[float], rhs_scalar: wp.array[float], articulation_block_row_offsets: wp.array[wp.int32], articulation_block_cols: wp.array[wp.int32], body_start: int, parent_local: int, child_local: int, parent_body: int, torque_parent: wp.vec3, H_aa: wp.mat33, cross_slot: int, stripe: int, lane_mask: int):
    has_parent = parent_body >= 0 and parent_local >= 0
    if has_parent:
        _rigid_add_rhs3(rhs_scalar, body_start + parent_local, 3, torque_parent, stripe, lane_mask)
        slot_pp = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, parent_local)
        _rigid_add_tile(values_scalar, slot_pp, 3, 3, H_aa, False, stripe, lane_mask)
    _rigid_add_rhs3(rhs_scalar, body_start + child_local, 3, -torque_parent, stripe, lane_mask)
    slot_cc = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, child_local, child_local)
    _rigid_add_tile(values_scalar, slot_cc, 3, 3, H_aa, False, stripe, lane_mask)
    if has_parent:
        slot_pc = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, wp.max(parent_local, child_local), wp.min(parent_local, child_local))
        _rigid_add_tile(values_scalar, slot_pc, 3, 3, -H_aa, False, stripe, lane_mask)
        _rigid_add_tile(values_scalar, cross_slot, 3, 3, -H_aa, False, stripe, lane_mask)

@wp.func
def _assemble_angular_axis_row_scalar(values_scalar: wp.array[float], rhs_scalar: wp.array[float], articulation_block_row_offsets: wp.array[wp.int32], articulation_block_cols: wp.array[wp.int32], body_start: int, parent_local: int, child_local: int, parent_body: int, angular_jacobian_world: wp.vec3, force_scalar: float, hessian_scalar: float, cross_slot: int, stripe: int, lane_mask: int):
    torque_parent = force_scalar * angular_jacobian_world
    H_aa = hessian_scalar * wp.outer(angular_jacobian_world, angular_jacobian_world)
    _assemble_angular_direct_pair_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, parent_body, torque_parent, H_aa, cross_slot, stripe, lane_mask)

_SMALL_ANGLE_EPS = wp.constant(1e-07)

_USE_SMALL_ANGLE_APPROX = wp.constant(True)

@wp.func
def compute_right_jacobian_inverse(kappa: wp.vec3) -> wp.mat33:
    """Inverse right Jacobian Jr^{-1}(kappa) for SO(3) rotation vectors.

    Args:
        kappa: Rotation vector theta*axis (any frame).

    Returns:
        wp.mat33: Jr^{-1}(kappa) in the same frame as kappa.
    """
    theta = wp.length(kappa)
    kappa_skew = wp.skew(kappa)
    if theta < _SMALL_ANGLE_EPS or _USE_SMALL_ANGLE_APPROX:
        return wp.identity(3, float) + 0.5 * kappa_skew + 1.0 / 12.0 * (kappa_skew * kappa_skew)
    sin_theta = wp.sin(theta)
    cos_theta = wp.cos(theta)
    b = 1.0 / (theta * theta) - (1.0 + cos_theta) / (2.0 * theta * sin_theta)
    return wp.identity(3, float) + 0.5 * kappa_skew + b * (kappa_skew * kappa_skew)

@wp.func
def compute_kappa_and_jacobian(q_wp: wp.quat, q_wc: wp.quat, q_wp_rest: wp.quat, q_wc_rest: wp.quat):
    """Compute rotation-vector residual and world-frame Jacobian from quaternion poses.

    Returns:
        (kappa, J_world) -- rotation vector and world-frame force Jacobian.
    """
    q_rel = wp.quat_inverse(q_wp) * q_wc
    q_rel_rest = wp.quat_inverse(q_wp_rest) * q_wc_rest
    q_align = q_rel * wp.quat_inverse(q_rel_rest)
    if q_align[3] < 0.0:
        q_align = wp.quat(-q_align[0], -q_align[1], -q_align[2], -q_align[3])
    axis, angle = wp.quat_to_axis_angle(q_align)
    kappa = axis * angle
    Jr_inv = compute_right_jacobian_inverse(kappa)
    R_wp = wp.quat_to_matrix(q_wp)
    R_align = wp.quat_to_matrix(q_align)
    J_world = R_wp * (R_align * wp.transpose(Jr_inv))
    return (kappa, J_world)

@wp.func
def _wrap_angle_pm_pi(theta: float) -> float:
    """Wrap an angle to the principal interval :math:`[-\\pi, \\pi)`.

    Args:
        theta: Input angle [rad].

    Returns:
        float: Wrapped angle [rad] in :math:`[-\\pi, \\pi)`.
    """
    two_pi = 2.0 * wp.pi
    wrapped = wp.mod(theta + wp.pi, two_pi)
    if wrapped < 0.0:
        wrapped += two_pi
    return wrapped - wp.pi

@wp.func
def quat_decompose(q: wp.quat) -> wp.vec3:
    """Decompose a quaternion into wrapped XYZ Euler coordinates.

    This wrapper calls :func:`warp.quat_to_euler() <warp._src.lang.quat_to_euler>` with the Newton convention
    :math:`(i, j, k) = (2, 1, 0)`, then wraps each angle to the principal
    branch. Wrapping avoids equivalent representations that differ by
    :math:`2\\pi`,
    which improves stability when reconstructing joint coordinates.

    .. math::
       e = \\operatorname{quat\\_to\\_euler}(q, 2, 1, 0),\\quad
       e_i \\leftarrow \\operatorname{wrap}_{[-\\pi,\\pi)}(e_i)

    Args:
        q: Input quaternion in Warp layout ``(x, y, z, w)``.

    Returns:
        wp.vec3: Wrapped Euler coordinates ``(x, y, z)`` [rad].
    """
    angles = wp.quat_to_euler(q, 2, 1, 0)
    return wp.vec3(_wrap_angle_pm_pi(angles[0]), _wrap_angle_pm_pi(angles[1]), _wrap_angle_pm_pi(angles[2]))

@wp.func
def invert_3d_rotational_dofs(axis_0: wp.vec3, axis_1: wp.vec3, axis_2: wp.vec3, q_p: wp.quat, q_c: wp.quat, w_err: wp.vec3):
    """
    Computes generalized joint position and velocity coordinates for a 3D rotational joint given the joint axes, relative orientations and angular velocity differences between the two bodies the joint connects.

    Builds ``q_off`` from a canonical right-handed basis ``(axis_0, axis_1, cross(axis_0, axis_1))``
    so that the XYZ Euler decomposition is always well-defined. If the user's ``axis_2`` points opposite
    that canonical third column (left-handed triple), the third angle and velocity component are
    negated to express the result in the user's axes — the inverse of :func:`compute_3d_rotational_dofs`.
    """
    axis_2_rh = wp.cross(axis_0, axis_1)
    s = float(1.0)
    if wp.dot(axis_2_rh, axis_2) < 0.0:
        s = float(-1.0)
    q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, axis_2_rh))
    q_pc = wp.quat_inverse(q_off) * wp.quat_inverse(q_p) * q_c * q_off
    angles = quat_decompose(q_pc)
    local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
    local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
    local_2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))
    a0 = local_0
    q_0 = wp.quat_from_axis_angle(a0, angles[0])
    a1 = wp.quat_rotate(q_0, local_1)
    q_1 = wp.quat_from_axis_angle(a1, angles[1])
    a2 = wp.quat_rotate(q_1 * q_0, local_2)
    w_err_p = wp.quat_rotate_inv(q_p, w_err)
    c12 = wp.cross(a1, a2)
    c02 = wp.cross(a0, a2)
    c01 = wp.cross(a0, a1)
    velocities = wp.vec3(wp.dot(w_err_p, c12) / wp.dot(a0, c12), wp.dot(w_err_p, c02) / wp.dot(a1, c02), wp.dot(w_err_p, c01) / wp.dot(a2, c01))
    return (wp.vec3(angles[0], angles[1], s * angles[2]), wp.vec3(velocities[0], velocities[1], s * velocities[2]))

@wp.func
def transform_3d_rotational_axes(axis_0: wp.vec3, axis_1: wp.vec3, axis_2: wp.vec3, q0: float, q1: float):
    """Returns the three world-space angular axes of a 3D rotational joint at coordinates ``(q0, q1)``.

    These are the per-DOF angular Jacobian columns of the intrinsic-Euler chain: ``axis_0`` and each
    later axis transported through the rotations applied before it. Matches the axis transport in
    :func:`compute_3d_rotational_dofs` exactly.
    """
    q_0 = wp.quat_from_axis_angle(axis_0, q0)
    axis_1_w = wp.quat_rotate(q_0, axis_1)
    q_1 = wp.quat_from_axis_angle(axis_1_w, q1)
    axis_2_w = wp.quat_rotate(q_1 * q_0, axis_2)
    return (axis_0, axis_1_w, axis_2_w)

@wp.func
def eval_angular_coordinates(joint_axis: wp.array[wp.vec3], axis_start: int, axis_count: int, q_p: wp.quat, q_c: wp.quat):
    """Returns the coordinates of a 2- or 3-axis angular D6 and the world-frame covectors as matrix rows.

    Covector ``k`` maps the relative angular velocity to ``dq_k/dt``, so it is also the torque direction of a
    generalized force on coordinate ``k``. For two axes, coordinate 2 is the locked rotation about their cross product.
    """
    axis_0 = joint_axis[axis_start]
    axis_1 = joint_axis[axis_start + 1]
    axis_2 = wp.cross(axis_0, axis_1)
    if axis_count == 3:
        axis_2 = joint_axis[axis_start + 2]
    else:
        axis_2 = axis_1
        axis_1 = wp.cross(axis_0, axis_2)
    coords, _vels = invert_3d_rotational_dofs(axis_0, axis_1, axis_2, q_p, q_c, wp.vec3(0.0))
    a0, a1, a2 = transform_3d_rotational_axes(axis_0, axis_1, axis_2, coords[0], coords[1])
    a0 = wp.quat_rotate(q_p, a0)
    a1 = wp.quat_rotate(q_p, a1)
    a2 = wp.quat_rotate(q_p, a2)
    if axis_count == 2:
        coords = wp.vec3(coords[0], coords[2], coords[1])
        a_locked = a1
        a1 = a2
        a2 = a_locked
    det = wp.dot(a0, wp.cross(a1, a2))
    inv_det = 1.0 / (wp.sign(det) * wp.max(wp.abs(det), 0.0001))
    return (coords, wp.matrix_from_rows(wp.cross(a1, a2), wp.cross(a2, a0), wp.cross(a0, a1)) * inv_det)

@wp.func
def compute_joint_kappa_and_jacobian(joint_axis: wp.array[wp.vec3], ang_start: int, ang_count: int, q_wp: wp.quat, q_wc: wp.quat, q_wp_rest: wp.quat, q_wc_rest: wp.quat):
    """:func:`compute_kappa_and_jacobian` with D6 coordinates for joints with 2-3 angular axes.

    Such joints use the absolute coordinates of ``eval_fk`` expressed in their joint-axis basis, so
    ``dot(kappa, a_k)`` is coordinate ``k`` and ``J_world * a_k`` its world-frame covector.
    """
    if ang_count < 2:
        return compute_kappa_and_jacobian(q_wp, q_wc, q_wp_rest, q_wc_rest)
    coords, covectors = eval_angular_coordinates(joint_axis, ang_start, ang_count, q_wp, q_wc)
    a0 = wp.normalize(joint_axis[ang_start])
    a1 = wp.normalize(joint_axis[ang_start + 1])
    a2 = wp.cross(a0, a1)
    if ang_count == 3:
        a2 = wp.normalize(joint_axis[ang_start + 2])
    axes = wp.matrix_from_cols(a0, a1, a2)
    return (axes * coords, wp.transpose(covectors) * wp.transpose(axes))

@wp.func
def compute_kappa_dot(J_world: wp.mat33, omega_p_world: wp.vec3, omega_c_world: wp.vec3) -> wp.vec3:
    """Time derivative of the rotation-vector residual d(kappa)/dt in parent frame.

    Exploits J_world^T = Jr_inv * R_align^T * R_wp^T, so
    kappa_dot = J_world^T * (omega_c - omega_p).

    Args:
        J_world: World-frame force Jacobian from compute_kappa_and_jacobian.
        omega_p_world: Parent angular velocity (world) [rad/s].
        omega_c_world: Child angular velocity (world) [rad/s].

    Returns:
        wp.vec3: Rotation-vector rate kappa_dot in parent frame [rad/s].
    """
    return wp.transpose(J_world) * (omega_c_world - omega_p_world)

@wp.func
def quat_velocity(q_now: wp.quat, q_prev: wp.quat, dt: float) -> wp.vec3:
    """Approximate angular velocity from successive world quaternions (world frame).

    Uses right-trivialized mapping via
    :math:`\\Delta q = q_{\\text{now}} q_{\\text{prev}}^{-1}`.

    .. math::
       \\Delta q = q_{now} q_{prev}^{-1},\\quad
       \\omega \\approx \\hat{u}(\\Delta q)\\,\\frac{\\theta(\\Delta q)}{\\Delta t}

    Args:
        q_now: Current orientation in world frame.
        q_prev: Previous orientation in world frame.
        dt: Time step [s].

    Returns:
        Angular velocity :math:`\\omega` in world frame [rad/s].
    """
    q1 = wp.normalize(q_now)
    q0 = wp.normalize(q_prev)
    if wp.dot(q1, q0) < 0.0:
        q0 = wp.quat(-q0[0], -q0[1], -q0[2], -q0[3])
    dq = wp.normalize(wp.mul(q1, wp.quat_inverse(q0)))
    axis, angle = wp.quat_to_axis_angle(dq)
    return axis * (angle / dt)

@wp.func
def _angular_constraint_force_hessian(parent_anchor_q: wp.quat, child_anchor_q: wp.quat, parent_anchor_q_prev: wp.quat, child_anchor_q_prev: wp.quat, parent_rest_q: wp.quat, child_rest_q: wp.quat, stiffness: float, P: wp.mat33, sigma0: wp.vec3, C_fric: wp.vec3, lambda_ang: wp.vec3, C0_ang: wp.vec3, alpha: float, damping: float, dt: float, joint_axis: wp.array[wp.vec3], ang_start: int, ang_count: int):
    kappa, J_world = compute_joint_kappa_and_jacobian(joint_axis, ang_start, ang_count, parent_anchor_q, child_anchor_q, parent_rest_q, child_rest_q)
    kappa_stab = kappa - alpha * C0_ang
    f_local = stiffness * (P * kappa_stab) + sigma0 + P * lambda_ang
    H_local = stiffness * P + wp.mat33(C_fric[0], 0.0, 0.0, 0.0, C_fric[1], 0.0, 0.0, 0.0, C_fric[2])
    if damping > 0.0:
        omega_parent = quat_velocity(parent_anchor_q, parent_anchor_q_prev, dt)
        omega_child = quat_velocity(child_anchor_q, child_anchor_q_prev, dt)
        dkappa_dt = compute_kappa_dot(J_world, omega_parent, omega_child)
        f_local = f_local + damping * (P * dkappa_dt)
        H_local = H_local + damping / dt * P
    torque_parent = J_world * f_local
    H_aa = J_world * (H_local * wp.transpose(J_world))
    return (torque_parent, H_aa, kappa, J_world)

@wp.func
def _assemble_angular_joint_scalar(values_scalar: wp.array[float], rhs_scalar: wp.array[float], articulation_block_row_offsets: wp.array[wp.int32], articulation_block_cols: wp.array[wp.int32], body_start: int, parent_local: int, child_local: int, parent_body: int, parent_anchor_q: wp.quat, child_anchor_q: wp.quat, parent_anchor_q_prev: wp.quat, child_anchor_q_prev: wp.quat, parent_rest_q: wp.quat, child_rest_q: wp.quat, stiffness: float, damping: float, dt: float, P: wp.mat33, sigma0: wp.vec3, C_fric: wp.vec3, lambda_ang: wp.vec3, C0_ang: wp.vec3, alpha: float, cross_slot: int, joint_axis: wp.array[wp.vec3], ang_start: int, ang_count: int, stripe: int, lane_mask: int):
    torque_parent, H_aa, kappa, J_world = _angular_constraint_force_hessian(parent_anchor_q, child_anchor_q, parent_anchor_q_prev, child_anchor_q_prev, parent_rest_q, child_rest_q, stiffness, P, sigma0, C_fric, lambda_ang, C0_ang, alpha, damping, dt, joint_axis, ang_start, ang_count)
    _assemble_angular_direct_pair_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, parent_body, torque_parent, H_aa, cross_slot, stripe, lane_mask)
    return (kappa, J_world)

@wp.func
def _add_coordinate_outer_scalar(values: wp.array[float], slot: int, left: vec6f, right: vec6f, hessian: float, transposed: bool, stripe: int, lane_mask: int):
    if slot >= 0:
        for entry in range(stripe, 36, lane_mask + 1):
            row = entry // 6
            col = entry % 6
            if transposed:
                value = hessian * (left[col] * right[row])
            else:
                value = hessian * (left[row] * right[col])
            if value != 0.0:
                wp.atomic_add(values, _scalar_block_index(slot, row, col), value)

@wp.func
def _assemble_coordinate_row_scalar(values_scalar: wp.array[float], rhs_scalar: wp.array[float], row_offsets: wp.array[wp.int32], cols: wp.array[wp.int32], body_start: int, parent_local: int, child_local: int, gradient_parent: wp.spatial_vector, gradient_child: wp.spatial_vector, force: float, hessian: float, cross_slot: int, stripe: int, lane_mask: int):
    """Assemble a joint-coordinate row, including the parent-child coupling."""
    gp = _vec6_from_parts(wp.spatial_top(gradient_parent), wp.spatial_bottom(gradient_parent))
    gc = _vec6_from_parts(wp.spatial_top(gradient_child), wp.spatial_bottom(gradient_child))
    slot_cc = _find_block_slot(row_offsets, cols, body_start, child_local, child_local)
    _add_coordinate_outer_scalar(values_scalar, slot_cc, gc, gc, hessian, False, stripe, lane_mask)
    _add_rhs_scalar_atomic(rhs_scalar, body_start + child_local, -force * gc, stripe, lane_mask)
    if parent_local >= 0:
        slot_pp = _find_block_slot(row_offsets, cols, body_start, parent_local, parent_local)
        _add_coordinate_outer_scalar(values_scalar, slot_pp, gp, gp, hessian, False, stripe, lane_mask)
        _add_rhs_scalar_atomic(rhs_scalar, body_start + parent_local, -force * gp, stripe, lane_mask)
        if parent_local >= child_local:
            slot_pc = _find_block_slot(row_offsets, cols, body_start, parent_local, child_local)
            _add_coordinate_outer_scalar(values_scalar, slot_pc, gp, gc, hessian, False, stripe, lane_mask)
        else:
            slot_cp = _find_block_slot(row_offsets, cols, body_start, child_local, parent_local)
            _add_coordinate_outer_scalar(values_scalar, slot_cp, gp, gc, hessian, True, stripe, lane_mask)
        _add_coordinate_outer_scalar(values_scalar, cross_slot, gp, gc, hessian, False, stripe, lane_mask)

@wp.func
def _anchor_offset(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], body: int, anchor: wp.vec3) -> wp.vec3:
    """Return the world vector from ``body``'s center of mass to the world point ``anchor``."""
    return anchor - wp.transform_point(body_q[body], body_com[body])

@wp.func
def _joint_angular_jacobian_value(P: wp.mat33, is_parent: bool, constraint_row: int, dof: int) -> float:
    if dof < 3:
        return 0.0
    sign = float(1.0)
    if is_parent:
        sign = -1.0
    return sign * P[constraint_row, dof - 3]

@wp.func
def _joint_linear_jacobian_value(P: wp.mat33, r: wp.vec3, is_parent: bool, constraint_row: int, dof: int) -> float:
    value = float(0.0)
    sign = float(1.0)
    if is_parent:
        sign = -1.0
    if dof < 3:
        value = sign * P[constraint_row, dof]
    else:
        S = wp.skew(r)
        local_dof = dof - 3
        if is_parent:
            for k in range(3):
                value = value + P[constraint_row, k] * S[k, local_dof]
        else:
            for k in range(3):
                value = value - P[constraint_row, k] * S[k, local_dof]
    return value


@wp.func
def jac_column(projector: wp.mat33, arm: wp.vec3, parent_side: bool, angular: bool, dof: int) -> wp.vec3:
    column = wp.vec3(0.0)
    for row in range(3):
        if angular:
            column[row] = _joint_angular_jacobian_value(projector,parent_side,row,dof)
        else:
            column[row] = _joint_linear_jacobian_value(projector,arm,parent_side,row,dof)
    return column

@wp.func
def column_gram(left:wp.vec3, scale:float, right:wp.vec3) -> float:
    result = float(0.0)
    for component in range(3):
        result = result + scale * left[component] * right[component]
    return result

@wp.func
def column_rhs(gradient:wp.vec3, residual:wp.vec3, scale:float) -> float:
    result = float(0.0)
    for component in range(3):
        result = result + gradient[component]*residual[component]
    return -scale*result

@wp.func
def write_nonzero(dest: wp.array[float], index:int, value:float):
    if value != 0.0:
        wp.atomic_add(dest,index,value)

@wp.func
def _assemble_constraint_pair_scalar(values_scalar: wp.array[float], rhs_scalar: wp.array[float], articulation_block_row_offsets: wp.array[wp.int32], articulation_block_cols: wp.array[wp.int32], body_start: int, local_a: int, local_b: int, residual: wp.vec3, force_scale: float, hessian_scale: float, P: wp.mat33, r_a: wp.vec3, r_b: wp.vec3, is_parent_a: bool, is_parent_b: bool, angular_only: bool, cross_slot: int, stripe: int, lane_mask: int):
    if force_scale == 0.0 and hessian_scale <= 0.0:
        return
    width = lane_mask + 1
    if force_scale != 0.0:
        for dof in range(stripe,6,width):
            grad_a = jac_column(P,r_a,is_parent_a,angular_only,dof)
            grad_b = jac_column(P,r_b,is_parent_b,angular_only,dof)
            write_nonzero(rhs_scalar,(body_start+local_a)*6+dof,column_rhs(grad_a,residual,force_scale))
            write_nonzero(rhs_scalar,(body_start+local_b)*6+dof,column_rhs(grad_b,residual,force_scale))
    if hessian_scale == 0.0:
        return
    slot_a = _find_block_slot(articulation_block_row_offsets,articulation_block_cols,body_start,local_a,local_a)
    slot_b = _find_block_slot(articulation_block_row_offsets,articulation_block_cols,body_start,local_b,local_b)
    slot_cross = _find_block_slot(articulation_block_row_offsets,articulation_block_cols,body_start,wp.max(local_a,local_b),wp.min(local_a,local_b))
    for entry in range(stripe,36,width):
        row = entry // 6
        col = entry % 6
        left_a = jac_column(P,r_a,is_parent_a,angular_only,row)
        right_b = jac_column(P,r_b,is_parent_b,angular_only,col)
        if slot_a >= 0:
            right_a = jac_column(P,r_a,is_parent_a,angular_only,col)
            write_nonzero(values_scalar,slot_a*36+entry,column_gram(left_a,hessian_scale,right_a))
        if slot_b >= 0:
            left_b = jac_column(P,r_b,is_parent_b,angular_only,row)
            write_nonzero(values_scalar,slot_b*36+entry,column_gram(left_b,hessian_scale,right_b))
        if slot_cross >= 0 or cross_slot >= 0:
            cross = column_gram(left_a,hessian_scale,right_b)
            if slot_cross >= 0:
                dest = col*6+row if local_a < local_b else entry
                write_nonzero(values_scalar,slot_cross*36+dest,cross)
            if cross_slot >= 0:
                write_nonzero(values_scalar,cross_slot*36+entry,cross)


@wp.func
def _assemble_constraint_single_scalar(values_scalar: wp.array[float], rhs_scalar: wp.array[float], articulation_block_row_offsets: wp.array[wp.int32], articulation_block_cols: wp.array[wp.int32], body_start: int, local_body: int, residual: wp.vec3, force_scale: float, hessian_scale: float, P: wp.mat33, r: wp.vec3, is_parent: bool, angular_only: bool, stripe: int, lane_mask: int):
    if force_scale == 0.0 and hessian_scale <= 0.0:
        return
    width = lane_mask + 1
    if force_scale != 0.0:
        for dof in range(stripe,6,width):
            grad = jac_column(P,r,is_parent,angular_only,dof)
            write_nonzero(rhs_scalar,(body_start+local_body)*6+dof,column_rhs(grad,residual,force_scale))
    if hessian_scale == 0.0:
        return
    slot = _find_block_slot(articulation_block_row_offsets,articulation_block_cols,body_start,local_body,local_body)
    if slot >= 0:
        for entry in range(stripe,36,width):
            row = entry // 6
            col = entry % 6
            left = jac_column(P,r,is_parent,angular_only,row)
            right = jac_column(P,r,is_parent,angular_only,col)
            write_nonzero(values_scalar,slot*36+entry,column_gram(left,hessian_scale,right))

@wp.func
def _assemble_linear_axis_row_scalar(values_scalar: wp.array[float], rhs_scalar: wp.array[float], articulation_block_row_offsets: wp.array[wp.int32], articulation_block_cols: wp.array[wp.int32], body_start: int, parent_local: int, child_local: int, parent_body: int, child_body: int, parent_anchor: wp.vec3, child_anchor: wp.vec3, body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], axis_world: wp.vec3, force_scalar: float, hessian_scalar: float, cross_slot: int, stripe: int, lane_mask: int):
    P = wp.outer(axis_world, axis_world)
    residual = axis_world
    r_child = _anchor_offset(body_q, body_com, child_body, child_anchor)
    if parent_body >= 0 and parent_local >= 0:
        r_parent = _anchor_offset(body_q, body_com, parent_body, parent_anchor)
        _assemble_constraint_pair_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, residual, force_scalar, hessian_scalar, P, r_parent, r_child, True, False, False, cross_slot, stripe, lane_mask)
    else:
        _assemble_constraint_single_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, child_local, residual, force_scalar, hessian_scalar, P, r_child, False, False, stripe, lane_mask)

@wp.func
def _linear_joint_residual(parent_anchor: wp.vec3, child_anchor: wp.vec3, parent_anchor_prev: wp.vec3, child_anchor_prev: wp.vec3, stiffness: float, damping: float, dt: float, P: wp.mat33, lambda_lin: wp.vec3, C0_lin: wp.vec3, alpha: float):
    """Return the projected linear force residual and its Gauss-Newton scale of a joint anchor pair."""
    C_vec = child_anchor - parent_anchor
    C_stab = C_vec - alpha * C0_lin
    force_residual = stiffness * (P * C_stab) + P * lambda_lin
    hessian_scale = stiffness
    if damping > 0.0:
        residual_prev = child_anchor_prev - parent_anchor_prev
        dC_dt = (C_vec - residual_prev) / dt
        force_residual = force_residual + damping * (P * dC_dt)
        hessian_scale = stiffness + damping / dt
    return (force_residual, hessian_scale)

@wp.func
def _assemble_linear_joint_scalar(values_scalar: wp.array[float], rhs_scalar: wp.array[float], articulation_block_row_offsets: wp.array[wp.int32], articulation_block_cols: wp.array[wp.int32], body_start: int, parent_local: int, child_local: int, parent_body: int, child_body: int, parent_anchor: wp.vec3, child_anchor: wp.vec3, parent_anchor_prev: wp.vec3, child_anchor_prev: wp.vec3, body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], stiffness: float, damping: float, dt: float, P: wp.mat33, lambda_lin: wp.vec3, C0_lin: wp.vec3, alpha: float, cross_slot: int, stripe: int, lane_mask: int):
    force_residual, hessian_scale = _linear_joint_residual(parent_anchor, child_anchor, parent_anchor_prev, child_anchor_prev, stiffness, damping, dt, P, lambda_lin, C0_lin, alpha)
    r_child = _anchor_offset(body_q, body_com, child_body, child_anchor)
    if parent_body >= 0 and parent_local >= 0:
        r_parent = _anchor_offset(body_q, body_com, parent_body, parent_anchor)
        _assemble_constraint_pair_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, force_residual, 1.0, hessian_scale, P, r_parent, r_child, True, False, False, cross_slot, stripe, lane_mask)
    else:
        _assemble_constraint_single_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, child_local, force_residual, 1.0, hessian_scale, P, r_child, False, False, stripe, lane_mask)

@wp.func
def _load_solve_weight(legacy_penalty_k: wp.array[float], compliant_rho: wp.array[float], idx: int, use_compliant_alm: int):
    """Load the active solve weight without reading the inactive array.

    Compliant ALM uses ``rho``, constant across the sweeps; legacy AVBD uses the
    mutable ``legacy_penalty_k`` (material stiffness in soft mode, a ramping
    penalty in hard mode).
    """
    if use_compliant_alm == 1:
        return compliant_rho[idx]
    return legacy_penalty_k[idx]

@wp.func
def _compliant_alm_coefficients(material_k: float, rho: float):
    """Return stable ``(s, k_eff, a)`` coefficients for compliant ALM.

    ``s=K/(K+rho)``, ``a=rho/(K+rho)``, ``k_eff=K*a``, branched to preserve
    whichever of ``s`` and ``a`` is small. A nonpositive input retires the row:
    ``a=1``, not the ``rho -> 0`` limit, lets the ascent clear its dual too.
    """
    if material_k <= 0.0 or rho <= 0.0:
        return (0.0, 0.0, 1.0)
    if material_k >= rho:
        r = rho / material_k
        s = 1.0 / (1.0 + r)
        return (s, rho * s, r * s)
    r = material_k / rho
    a = 1.0 / (1.0 + r)
    return (r * a, material_k * a, a)

@wp.func
def _material_force_terms(rho: float, material_k: float, dual: Any, use_compliant_alm: int):
    """Return the ``(k, lam_eff)`` pair of a row force ``k*C + lam_eff``.

    Compliant ALM gives ``(k_eff, s*dual)``, legacy ``(rho, dual)``. Coulomb
    tangents bypass this: their dual is not scaled by ``s``.
    """
    if use_compliant_alm == 1:
        s, k_eff, _a = _compliant_alm_coefficients(material_k, rho)
        return (k_eff, dual * s)
    return (rho, dual)

@wp.func
def _structural_row_has_stiffness(solve_weight: float, material_k: float, use_compliant_alm: int):
    """Return whether a structural row has stiffness in the selected formulation."""
    if use_compliant_alm == 1:
        return material_k > 0.0
    return solve_weight > 0.0

@wp.struct
class RodBendTwistMeasure:
    """Live bend/twist geometry for one rod joint.

    Measured once per force/Hessian evaluation and reused by the residual and
    analytic Jacobian columns.
    """
    t0: wp.vec3
    t1: wp.vec3
    m0: wp.vec3
    m1: wp.vec3
    kb_world: wp.vec3
    twist: float

_ROD_KB_CURVATURE_CAP = wp.constant(20.0)

_ROD_KB_FOLD_EPS = wp.constant(1e-12)

_SMALL_LENGTH_EPS = wp.constant(1e-09)

@wp.func
def _normalize_with_fallback(v: wp.vec3, fallback: wp.vec3) -> wp.vec3:
    """Normalize v, then fallback, with a final fixed axis if both are tiny."""
    v_len = wp.length(v)
    if v_len > _SMALL_LENGTH_EPS:
        return v / v_len
    fb_len = wp.length(fallback)
    if fb_len > _SMALL_LENGTH_EPS:
        return fallback / fb_len
    return wp.vec3(1.0, 0.0, 0.0)

@wp.func
def _project_perp(v: wp.vec3, axis: wp.vec3) -> wp.vec3:
    """Project v onto the plane orthogonal to an already-unit axis."""
    return v - axis * wp.dot(v, axis)

@wp.func
def _perpendicular_axis_with_fallback(axis: wp.vec3, preferred: wp.vec3) -> wp.vec3:
    """Return a unit vector perpendicular to axis, preferring preferred when possible."""
    axis = _normalize_with_fallback(axis, wp.vec3(0.0, 0.0, 1.0))
    perp = _project_perp(preferred, axis)
    perp_len = wp.length(perp)
    if perp_len > _SMALL_LENGTH_EPS:
        return perp / perp_len
    alt = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(axis[0]) > 0.9:
        alt = wp.vec3(0.0, 1.0, 0.0)
    return _normalize_with_fallback(_project_perp(alt, axis), wp.vec3(0.0, 0.0, 1.0))

@wp.func
def _finite_curvature_binormal(t0: wp.vec3, t1: wp.vec3, fallback_axis: wp.vec3) -> wp.vec3:
    """Capped DER finite curvature binormal with a true-fold fallback."""
    tangent_dot = wp.clamp(wp.dot(t0, t1), -1.0, 1.0)
    tangent_cross = wp.cross(t0, t1)
    denom = 1.0 + tangent_dot
    cross_sq = wp.dot(tangent_cross, tangent_cross)
    if denom <= _ROD_KB_FOLD_EPS and cross_sq <= _ROD_KB_FOLD_EPS:
        axis = _perpendicular_axis_with_fallback(t0, fallback_axis)
        return _ROD_KB_CURVATURE_CAP * axis
    kb = 2.0 / wp.max(_ROD_KB_FOLD_EPS, denom) * tangent_cross
    kb_len = wp.length(kb)
    if kb_len > _ROD_KB_CURVATURE_CAP:
        kb = _ROD_KB_CURVATURE_CAP / kb_len * kb
    return kb

@wp.func
def _quat_rotate_local_x(q: wp.quat) -> wp.vec3:
    """Rotate local +X by a unit quaternion; the first rotation-matrix column."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return wp.vec3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w))

_ROD_TRANSPORT_DENOM_EPS = wp.constant(1e-08)

@wp.func
def _bishop_transport_quat(t0: wp.vec3, t1: wp.vec3, fallback_axis: wp.vec3) -> wp.quat:
    """Minimal no-twist rotation that transports unit tangent t0 to unit tangent t1."""
    c = wp.clamp(wp.dot(t0, t1), -1.0, 1.0)
    v = wp.cross(t0, t1)
    s = wp.length(v)
    if s > _SMALL_ANGLE_EPS:
        return wp.quat_from_axis_angle(v / s, wp.atan2(s, c))
    if c > 0.0:
        return wp.quat_identity()
    axis = _perpendicular_axis_with_fallback(t0, fallback_axis)
    return wp.quat_from_axis_angle(axis, wp.pi)

@wp.func
def _transport_material_axis(t0: wp.vec3, t1: wp.vec3, m0: wp.vec3, fallback_axis: wp.vec3) -> wp.vec3:
    """Parallel-transport a material normal from tangent t0 to tangent t1."""
    c = wp.clamp(wp.dot(t0, t1), -1.0, 1.0)
    denom = 1.0 + c
    if denom > _ROD_TRANSPORT_DENOM_EPS:
        w = t0 + t1
        transported = m0 - wp.dot(m0, t1) / denom * w
        return _normalize_with_fallback(transported, fallback_axis)
    B = _bishop_transport_quat(t0, t1, fallback_axis)
    return _normalize_with_fallback(wp.quat_rotate(B, m0), fallback_axis)

@wp.func
def _transported_twist_angle_from_material_axes(t0: wp.vec3, t1: wp.vec3, m0: wp.vec3, m1: wp.vec3, fallback_axis: wp.vec3) -> float:
    """Signed twist from transported parent material normal to child material normal."""
    m0_transport = _transport_material_axis(t0, t1, m0, fallback_axis)
    sin_theta = wp.dot(t1, wp.cross(m0_transport, m1))
    cos_theta = wp.dot(m0_transport, m1)
    return wp.atan2(sin_theta, cos_theta)

@wp.func
def quat_rotate_local_z(q: wp.quat) -> wp.vec3:
    """Rotate local +Z by a unit quaternion; the third rotation-matrix column."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return wp.vec3(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y))

@wp.func
def measure_rod_bend_twist_z(q_wp: wp.quat, q_wc: wp.quat) -> RodBendTwistMeasure:
    """Measure bend/twist for rod joints whose material tangent is local +Z.

    The fixed rod material basis is local ``+X, +Y, +Z``.
    Both input quaternions must be normalized so the rotated basis axes are
    orthonormal.
    """
    t0 = quat_rotate_local_z(q_wp)
    t1 = quat_rotate_local_z(q_wc)
    m0 = _quat_rotate_local_x(q_wp)
    m1 = _quat_rotate_local_x(q_wc)
    measure = RodBendTwistMeasure()
    measure.t0 = t0
    measure.t1 = t1
    measure.m0 = m0
    measure.m1 = m1
    measure.twist = _transported_twist_angle_from_material_axes(t0, t1, m0, m1, m0)
    measure.kb_world = _finite_curvature_binormal(t0, t1, m0)
    return measure

@wp.func
def _finite_curvature_binormal_jacobian(t0: wp.vec3, t1: wp.vec3, is_parent: bool) -> wp.mat33:
    """Jacobian of the finite curvature binormal for one endpoint rotation.

    This is the matrix form of ``_finite_curvature_binormal_derivative`` and
    evaluates geometry shared by the three world-axis columns only once.
    """
    zero = wp.mat33(0.0)
    raw_tangent_dot = wp.dot(t0, t1)
    tangent_dot = wp.clamp(raw_tangent_dot, -1.0, 1.0)
    tangent_cross = wp.cross(t0, t1)
    denom = 1.0 + tangent_dot
    cross_sq = wp.dot(tangent_cross, tangent_cross)
    if denom <= _ROD_KB_FOLD_EPS and cross_sq <= _ROD_KB_FOLD_EPS:
        return zero
    identity = wp.identity(3, float)
    if is_parent:
        ddenom_domega = tangent_cross
        dcross_domega = wp.outer(t0, t1) - raw_tangent_dot * identity
    else:
        ddenom_domega = -tangent_cross
        dcross_domega = raw_tangent_dot * identity - wp.outer(t1, t0)
    denom_safe = wp.max(_ROD_KB_FOLD_EPS, denom)
    inv_denom = 1.0 / denom_safe
    kb_raw = 2.0 * inv_denom * tangent_cross
    dkb_domega = 2.0 * inv_denom * dcross_domega
    if denom > _ROD_KB_FOLD_EPS:
        dkb_domega = dkb_domega - 2.0 * inv_denom * inv_denom * wp.outer(tangent_cross, ddenom_domega)
    kb_len = wp.length(kb_raw)
    if kb_len > _ROD_KB_CURVATURE_CAP:
        inv_len = 1.0 / kb_len
        projection = wp.identity(3, float) - inv_len * inv_len * wp.outer(kb_raw, kb_raw)
        dkb_domega = _ROD_KB_CURVATURE_CAP * inv_len * projection * dkb_domega
    return dkb_domega

_ROD_TWIST_JACOBIAN_DIRECTIONAL_DENOM = wp.constant(0.02)

_ROD_TWIST_ATAN2_DENOM_EPS = wp.constant(1e-12)

@wp.func
def _transported_twist_angle_derivative_from_measure(measure: RodBendTwistMeasure, omega_world: wp.vec3, is_parent: bool) -> float:
    """Directional derivative of transported twist for one endpoint rotation.

    Linear in the infinitesimal world-space rotation ``omega_world``; frames
    are read from ``measure``.
    """
    t0 = measure.t0
    t1 = measure.t1
    m0 = measure.m0
    m1 = measure.m1
    dt0 = wp.vec3(0.0)
    dt1 = wp.vec3(0.0)
    dm0 = wp.vec3(0.0)
    dm1 = wp.vec3(0.0)
    if is_parent:
        dt0 = wp.cross(omega_world, t0)
        dm0 = wp.cross(omega_world, m0)
    else:
        dt1 = wp.cross(omega_world, t1)
        dm1 = wp.cross(omega_world, m1)
    c = wp.clamp(wp.dot(t0, t1), -1.0, 1.0)
    denom = 1.0 + c
    if denom <= _ROD_TRANSPORT_DENOM_EPS:
        if is_parent:
            return -wp.dot(omega_world, t0)
        return wp.dot(omega_world, t1)
    w = t0 + t1
    n = wp.dot(m0, t1)
    scale = n / denom
    m0_transport_raw = m0 - scale * w
    m0_transport = _normalize_with_fallback(m0_transport_raw, m0)
    ddenom = wp.dot(dt0, t1) + wp.dot(t0, dt1)
    dn = wp.dot(dm0, t1) + wp.dot(m0, dt1)
    dscale = (dn * denom - n * ddenom) / (denom * denom)
    dm0_transport_raw = dm0 - dscale * w - scale * (dt0 + dt1)
    raw_len = wp.length(m0_transport_raw)
    dm0_transport = wp.vec3(0.0)
    if raw_len > _SMALL_LENGTH_EPS:
        inv_len = 1.0 / raw_len
        dm0_transport = inv_len * (dm0_transport_raw - m0_transport * wp.dot(m0_transport, dm0_transport_raw))
    sin_theta = wp.dot(t1, wp.cross(m0_transport, m1))
    cos_theta = wp.dot(m0_transport, m1)
    dsin = wp.dot(dt1, wp.cross(m0_transport, m1)) + wp.dot(t1, wp.cross(dm0_transport, m1) + wp.cross(m0_transport, dm1))
    dcos = wp.dot(dm0_transport, m1) + wp.dot(m0_transport, dm1)
    denom_angle = wp.max(_ROD_TWIST_ATAN2_DENOM_EPS, sin_theta * sin_theta + cos_theta * cos_theta)
    return (cos_theta * dsin - sin_theta * dcos) / denom_angle

@wp.func
def _transported_twist_angle_jacobian_from_measure(measure: RodBendTwistMeasure, is_parent: bool) -> wp.vec3:
    """Jacobian row of transported twist for one endpoint rotation.

    Use the tangent-bisector closed form for well-conditioned tangents. Near a
    fold, evaluate the full directional derivative along each world axis so the
    Jacobian remains consistent with the float32 transport residual.
    """
    t0 = measure.t0
    t1 = measure.t1
    denom = 1.0 + wp.clamp(wp.dot(t0, t1), -1.0, 1.0)
    if denom <= _ROD_TWIST_JACOBIAN_DIRECTIONAL_DENOM:
        e0 = wp.vec3(1.0, 0.0, 0.0)
        e1 = wp.vec3(0.0, 1.0, 0.0)
        e2 = wp.vec3(0.0, 0.0, 1.0)
        return wp.vec3(_transported_twist_angle_derivative_from_measure(measure, e0, is_parent), _transported_twist_angle_derivative_from_measure(measure, e1, is_parent), _transported_twist_angle_derivative_from_measure(measure, e2, is_parent))
    jacobian = (t0 + t1) / denom
    return -jacobian if is_parent else jacobian

@wp.func
def rod_bend_twist_jacobian_z_from_measure(measure: RodBendTwistMeasure, is_parent: bool) -> wp.mat33:
    """Jacobian of [bend_x, bend_y, twist_z] for rod joints with fixed local +Z.

    The local residual is exactly ``[bend_x, bend_y, twist_z]``, so no bend
    projector or twist-axis vector is needed in this hot path.
    """
    dkb_domega = _finite_curvature_binormal_jacobian(measure.t0, measure.t1, is_parent)
    if is_parent:
        dkb_domega = dkb_domega + wp.skew(measure.kb_world)
    parent_x_world = measure.m0
    parent_y_world = wp.cross(measure.t0, measure.m0)
    bend_jacobian_x = wp.transpose(dkb_domega) * parent_x_world
    bend_jacobian_y = wp.transpose(dkb_domega) * parent_y_world
    twist_jacobian = _transported_twist_angle_jacobian_from_measure(measure, is_parent)
    return wp.matrix_from_rows(bend_jacobian_x, bend_jacobian_y, twist_jacobian)

@wp.func
def _sparse_cross_element(row: int, col: int, material: wp.mat33, iso: float, child: wp.vec3, anchor: wp.vec3, parent: wp.vec3) -> float:
    if row < 3 and col < 3:
        return (-iso if row == col else 0.0) - material[row, col]
    skew_child = wp.skew(child)
    if row < 3:
        ang_col = col - 3
        entry = iso * skew_child[row, ang_col]
        for component in range(3):
            if component != ang_col:
                entry = entry + material[row, component] * skew_child[component, ang_col]
        return entry
    ang_row = row - 3
    skew_parent = wp.skew(parent)
    skew_anchor = wp.skew(anchor)
    result = float(0.0)
    for axis in range(3):
        if axis != ang_row:
            iso_arm = iso * skew_anchor[axis, ang_row]
            if col < 3:
                if axis == col:
                    result = result + iso_arm
                result = result + skew_parent[axis, ang_row] * material[axis, col]
            else:
                ang_col = col - 3
                if axis != ang_col:
                    result = result + iso_arm * -skew_child[axis, ang_col]
                for component in range(3):
                    if component != ang_col:
                        left = skew_parent[axis, ang_row] * material[axis, component]
                        result = result + left * -skew_child[component, ang_col]
    return result

@wp.func
def _rod_cross_hessian(dt: float, joint: int, parent: int, child: int, body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], X_pj: wp.transform, X_cj: wp.transform, joint_constraint_start: wp.array[int], joint_penalty_k: wp.array[float], joint_rho: wp.array[float], joint_material_k: wp.array[float], joint_penalty_kd: wp.array[float], joint_C_fric: wp.array[wp.vec3], joint_is_hard: wp.array[wp.int32], joint_compliant_alm: int, stripe: int, lane_mask: int):
    """Return the parent-child Gauss-Newton block for one ROD joint."""
    cross = mat66f(0.0)
    c_start = joint_constraint_start[joint]
    stretch_idx = c_start
    shear_idx = c_start + 1
    bend_idx = c_start + 2
    twist_idx = c_start + 3
    stretch_weight = _load_solve_weight(joint_penalty_k, joint_rho, stretch_idx, joint_compliant_alm)
    shear_weight = _load_solve_weight(joint_penalty_k, joint_rho, shear_idx, joint_compliant_alm)
    bend_weight = _load_solve_weight(joint_penalty_k, joint_rho, bend_idx, joint_compliant_alm)
    twist_weight = _load_solve_weight(joint_penalty_k, joint_rho, twist_idx, joint_compliant_alm)
    k_stretch, _ = _material_force_terms(stretch_weight, joint_material_k[stretch_idx], wp.vec3(0.0), joint_compliant_alm)
    k_shear, _ = _material_force_terms(shear_weight, joint_material_k[shear_idx], wp.vec3(0.0), joint_compliant_alm)
    k_bend, _ = _material_force_terms(bend_weight, joint_material_k[bend_idx], wp.vec3(0.0), joint_compliant_alm)
    k_twist, _ = _material_force_terms(twist_weight, joint_material_k[twist_idx], wp.vec3(0.0), joint_compliant_alm)
    X_wp = body_q[parent] * X_pj
    X_wc = body_q[child] * X_cj
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    h_shear = k_shear + joint_penalty_kd[shear_idx] / dt
    h_stretch = k_stretch + joint_penalty_kd[stretch_idx] / dt
    if h_shear > 0.0 or h_stretch > 0.0:
        tangent = wp.quat_rotate(q_wp, wp.vec3(0.0, 0.0, 1.0))
        K_world = h_shear * wp.identity(3, float) + (h_stretch - h_shear) * wp.outer(tangent, tangent)
        k_iso = wp.min(k_shear, k_stretch)
        K_material = K_world - k_iso * wp.identity(3, float)
        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        parent_com_world = wp.transform_point(body_q[parent], body_com[parent])
        child_com_world = wp.transform_point(body_q[child], body_com[child])
        r_parent_anchor = x_p - parent_com_world
        r_parent_material = x_c - parent_com_world
        r_child = x_c - child_com_world
        identity = wp.identity(3, float)
        for i in range(6):
            for j in range(6):
                if i * 6 + j & lane_mask == stripe:
                    cross[i, j] = _sparse_cross_element(i, j, K_material, k_iso, r_child, r_parent_anchor, r_parent_material)
    bend_stiff = _structural_row_has_stiffness(bend_weight, joint_material_k[bend_idx], joint_compliant_alm)
    twist_stiff = _structural_row_has_stiffness(twist_weight, joint_material_k[twist_idx], joint_compliant_alm)
    bend_hard = bend_stiff and joint_compliant_alm == 0 and (joint_is_hard[bend_idx] == 1)
    twist_hard = twist_stiff and joint_compliant_alm == 0 and (joint_is_hard[twist_idx] == 1)
    friction = joint_C_fric[joint]
    h_bend_x = k_bend + joint_penalty_kd[bend_idx] / dt
    h_bend_y = h_bend_x
    h_twist = k_twist + joint_penalty_kd[twist_idx] / dt
    if bend_stiff and (not bend_hard):
        h_bend_x = h_bend_x + friction[0]
        h_bend_y = h_bend_y + friction[1]
    if twist_stiff and (not twist_hard):
        h_twist = h_twist + friction[2]
    if h_bend_x > 0.0 or h_bend_y > 0.0 or h_twist > 0.0:
        measure = measure_rod_bend_twist_z(q_wp, q_wc)
        J_parent_angular = rod_bend_twist_jacobian_z_from_measure(measure, True)
        J_child_angular = rod_bend_twist_jacobian_z_from_measure(measure, False)
        H_local_diag = wp.vec3(h_bend_x, h_bend_y, h_twist)
        for i in range(3):
            for j in range(3):
                if (i + 3) * 6 + j + 3 & lane_mask == stripe:
                    value = float(0.0)
                    for row in range(3):
                        value = value + J_parent_angular[row, i] * H_local_diag[row] * J_child_angular[row, j]
                    cross[i + 3, j + 3] = cross[i + 3, j + 3] + value
    return cross

@wp.func
def _drive_row_applies_force(drive_ke: float, drive_kd: float):
    """Whether the drive row contributes force on either path.

    A pure damper (zero stiffness, nonzero damping) is a valid authored drive,
    so damping alone activates the row.
    """
    return drive_ke > 0.0 or drive_kd > 0.0

@wp.func
def _quat_from_rotation_vector(theta: wp.vec3, angle: float):
    """Return exp(theta) for ``angle = |theta| > 0``, accurate for small angles under fast math.

    Fast-math sin/cos carry an absolute error near 4e-7, a large relative error for the small
    rotations of elastic endpoints; below a half angle of 0.1 rad a Taylor series replaces them.
    """
    half = 0.5 * angle
    if half < 0.1:
        h2 = half * half
        sin_half_over_angle = 0.5 * (1.0 - h2 / 6.0 * (1.0 - h2 / 20.0 * (1.0 - h2 / 42.0)))
        cos_half = 1.0 - 0.5 * h2 * (1.0 - h2 / 12.0 * (1.0 - h2 / 30.0))
        v = theta * sin_half_over_angle
        return wp.quat(v[0], v[1], v[2], cos_half)
    return wp.quat_from_axis_angle(theta / angle, angle)

@wp.func
def eval_elastic_endpoint_xform(joint_index: int, body_index: int, is_parent_side: bool, xform_rest: wp.transform, body_elastic_index: wp.array[wp.int32], elastic_joint: wp.array[wp.int32], elastic_mode_count: wp.array[wp.int32], joint_q: wp.array[float], joint_q_start: wp.array[int], joint_parent_elastic_endpoint: wp.array[wp.int32], joint_child_elastic_endpoint: wp.array[wp.int32], elastic_endpoint_phi: wp.array[wp.vec3], elastic_endpoint_psi: wp.array[wp.vec3], elastic_max_mode_count: int):
    """Return a joint's body-local attachment frame, deformed by its reduced elastic endpoint.

    The translation moves by ``sum_i phi_i q_i`` [m] and the rotation is pre-multiplied by
    ``exp(sum_i psi_i q_i)``, both in body-local coordinates. Rigid bodies and joints without an
    elastic endpoint on this side return ``xform_rest`` unchanged.
    """
    if body_index < 0:
        return xform_rest
    elastic_index = body_elastic_index[body_index]
    if elastic_index < 0:
        return xform_rest
    endpoint = joint_child_elastic_endpoint[joint_index]
    if is_parent_side:
        endpoint = joint_parent_elastic_endpoint[joint_index]
    if endpoint < 0:
        return xform_rest
    p = wp.transform_get_translation(xform_rest)
    q = wp.transform_get_rotation(xform_rest)
    owner_joint = elastic_joint[elastic_index]
    q_start = joint_q_start[owner_joint] + 7
    mode_count = elastic_mode_count[elastic_index]
    theta = wp.vec3(0.0, 0.0, 0.0)
    for i in range(elastic_max_mode_count):
        if i < mode_count:
            idx = endpoint * elastic_max_mode_count + i
            p = p + elastic_endpoint_phi[idx] * joint_q[q_start + i]
            theta = theta + elastic_endpoint_psi[idx] * joint_q[q_start + i]
    angle = wp.length(theta)
    if angle > _SMALL_ANGLE_EPS:
        q = _quat_from_rotation_vector(theta, angle) * q
    return wp.transform(p, q)

@wp.func
def _coordinate_frame(data: JointCoordinateData, joint: int, body: int, is_parent_side: bool, X: wp.transform, elastic_q: wp.array[float]):
    if data.elastic_max_mode_count == 0:
        return X
    return eval_elastic_endpoint_xform(joint, body, is_parent_side, X, data.body_elastic_index, data.elastic_joint, data.elastic_mode_count, elastic_q, data.q_start, data.parent_elastic_endpoint, data.child_elastic_endpoint, data.elastic_endpoint_phi, data.elastic_endpoint_psi, data.elastic_max_mode_count)

@wp.func
def unwrap_joint_coordinate(coordinate: float, previous: float, component: int, linear_count: int):
    """Lift an angular coordinate to the revolution nearest its previous value.

    Linear coordinates pass through unchanged. The caller owns the continuous
    history and must sample angular motion in increments smaller than pi.
    Quaternion sign changes do not affect the result.
    """
    if component >= linear_count:
        delta = coordinate - previous
        if wp.abs(delta) >= wp.pi:
            delta = wp.atan2(wp.sin(delta), wp.cos(delta))
        return previous + delta
    return coordinate

@wp.func
def _continuous_euler_angles(angles: wp.vec3, previous: wp.vec3, angular_count: int):
    """Choose the equivalent Euler branch nearest the previous coordinates."""
    direct = wp.vec3()
    alternate = wp.vec3()
    other = wp.vec3(angles[0] + wp.pi, wp.pi - angles[1], angles[2] + wp.pi)
    direct_distance, alternate_distance = (float(0.0), float(0.0))
    for axis in range(angular_count):
        direct[axis] = unwrap_joint_coordinate(angles[axis], previous[axis], 0, 0)
        alternate[axis] = unwrap_joint_coordinate(other[axis], previous[axis], 0, 0)
        direct_step = direct[axis] - previous[axis]
        alternate_step = alternate[axis] - previous[axis]
        direct_distance += direct_step * direct_step
        alternate_distance += alternate_step * alternate_step
    if alternate_distance < direct_distance:
        return alternate
    return direct

@wp.func
def _euler_coordinate_gradient(a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, component: int):
    """Return a row of the inverse angular Jacobian, not a rotation axis."""
    numerator = wp.cross(a1, a2)
    axis = a0
    if component == 1:
        numerator = wp.cross(a2, a0)
        axis = a1
    elif component == 2:
        numerator = wp.cross(a0, a1)
        axis = a2
    denominator = wp.dot(axis, numerator)
    if wp.abs(denominator) < 1e-06:
        denominator = wp.where(denominator < 0.0, -1e-06, 1e-06)
    return numerator / denominator

@wp.func
def _twist_coordinate_gradient(axis: wp.vec3, rotation: wp.quat):
    """Differentiate the signed twist angle, including off-axis swing."""
    v = wp.vec3(rotation[0], rotation[1], rotation[2])
    w = rotation[3]
    s = wp.dot(axis, v)
    denominator = w * w + s * s
    if denominator <= 1e-12:
        return axis
    return (w * w * axis + w * wp.cross(v, axis) + s * v) / denominator

@wp.func
def _two_axis_coordinate(axis_0: wp.vec3, axis_1: wp.vec3, rotation: wp.quat, component: int):
    """Recover two orthogonal rotations without the artificial third Euler axis."""
    basis = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, wp.cross(axis_0, axis_1)))
    R = wp.quat_to_matrix(wp.quat_inverse(basis) * rotation * basis)
    angle = wp.atan2(R[2, 1], R[1, 1])
    denominator = wp.max(R[1, 1] * R[1, 1] + R[2, 1] * R[2, 1], 1e-12)
    gradient = wp.vec3(1.0, -R[0, 1] * R[1, 1] / denominator, -R[0, 1] * R[2, 1] / denominator)
    if component == 1:
        angle = wp.atan2(R[0, 2], R[0, 0])
        denominator = wp.max(R[0, 0] * R[0, 0] + R[0, 2] * R[0, 2], 1e-12)
        gradient = wp.vec3(0.0, (R[0, 0] * R[2, 2] - R[0, 2] * R[2, 0]) / denominator, (R[0, 2] * R[1, 0] - R[0, 0] * R[1, 2]) / denominator)
    return (angle, wp.quat_rotate(basis, gradient))

@wp.func
def invert_2d_rotational_dofs(axis_0: wp.vec3, axis_1: wp.vec3, q_p: wp.quat, q_c: wp.quat, w_err: wp.vec3):
    """
    Computes generalized joint position and velocity coordinates for a 2D rotational joint given the joint axes, relative orientations and angular velocity differences between the two bodies the joint connects.

    Decomposes in the order ``(axis_0, axis_0 x axis_1, axis_1)``, so the locked rotation is the middle angle
    and both joint coordinates cover their full range.
    """
    qs, qds = invert_3d_rotational_dofs(axis_0, wp.cross(axis_0, axis_1), axis_1, q_p, q_c, w_err)
    return (wp.vec2(qs[0], qs[2]), wp.vec2(qds[0], qds[2]))

@wp.func
def eval_joint_coordinate_in_frames(joint: int, component: int, body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], joint_type: wp.array[int], joint_parent: wp.array[int], joint_child: wp.array[int], X_pj: wp.transform, X_cj: wp.transform, joint_qd_start: wp.array[int], joint_dof_dim: wp.array2d[int], joint_axis: wp.array[wp.vec3], angular_reference: wp.vec3, continuous: bool):
    """Return one joint coordinate and its maximal-coordinate gradients for explicit body-local joint frames.

    ``X_pj`` and ``X_cj`` replace ``joint_X_p[joint]`` and ``joint_X_c[joint]``, so callers can measure
    the coordinate at attachment frames that move relative to their bodies, such as deformed reduced
    elastic endpoints. The gradients use the supplied anchors as lever arms.
    """
    type = joint_type[joint]
    parent = joint_parent[joint]
    child = joint_child[joint]
    X_wp = X_pj
    pose_p = X_wp
    if parent >= 0:
        pose_p = body_q[parent]
        X_wp = pose_p * X_wp
    pose_c = body_q[child]
    X_wc = pose_c * X_cj
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)
    rel_q = wp.quat_inverse(q_p) * q_c
    x_err = wp.transform_get_translation(X_wc) - wp.transform_get_translation(X_wp)
    x_err_p = wp.quat_rotate_inv(q_p, x_err)
    qd_start = joint_qd_start[joint]
    lin_axis_count = joint_dof_dim[joint, 0]
    ang_axis_count = joint_dof_dim[joint, 1]
    coordinate = float(0.0)
    linear_axis = wp.vec3(0.0)
    angular_covector = wp.vec3(0.0)
    if type == JointType.PRISMATIC:
        axis = joint_axis[qd_start]
        coordinate = wp.dot(x_err_p, axis)
        linear_axis = wp.quat_rotate(q_p, axis)
    elif type == JointType.REVOLUTE:
        axis = joint_axis[qd_start]
        coordinate = wp.quat_twist_angle_signed(axis, rel_q)
        angular_covector = wp.quat_rotate(q_p, _twist_coordinate_gradient(axis, rel_q))
    elif type == JointType.D6:
        if component < lin_axis_count:
            axis = joint_axis[qd_start + component]
            coordinate = wp.dot(x_err_p, axis)
            linear_axis = wp.quat_rotate(q_p, axis)
        else:
            angular_component = component - lin_axis_count
            angular_start = qd_start + lin_axis_count
            local_covector = wp.vec3(0.0)
            if ang_axis_count == 1:
                axis = joint_axis[angular_start]
                coordinate = wp.quat_twist_angle_signed(axis, rel_q)
                local_covector = _twist_coordinate_gradient(axis, rel_q)
            elif ang_axis_count == 2:
                axis_0 = joint_axis[angular_start + 0]
                axis_1 = joint_axis[angular_start + 1]
                if continuous:
                    coordinate, local_covector = _two_axis_coordinate(axis_0, axis_1, rel_q, angular_component)
                else:
                    coordinates_2, _unused_velocities_2 = invert_2d_rotational_dofs(axis_0, axis_1, q_p, q_c, wp.vec3(0.0))
                    coordinate = coordinates_2[angular_component]
                    axis_0_q, axis_1_q, axis_2_q = transform_3d_rotational_axes(axis_0, axis_1, wp.cross(axis_0, axis_1), coordinates_2[0], coordinates_2[1])
                    local_covector = _euler_coordinate_gradient(axis_0_q, axis_1_q, axis_2_q, angular_component)
            elif ang_axis_count == 3:
                axis_0 = joint_axis[angular_start + 0]
                axis_1 = joint_axis[angular_start + 1]
                axis_2 = joint_axis[angular_start + 2]
                coordinates_3, _unused_velocities_3 = invert_3d_rotational_dofs(axis_0, axis_1, axis_2, q_p, q_c, wp.vec3(0.0))
                if continuous:
                    coordinates_3 = _continuous_euler_angles(coordinates_3, angular_reference, 3)
                coordinate = coordinates_3[angular_component]
                axis_0_q, axis_1_q, axis_2_q = transform_3d_rotational_axes(axis_0, axis_1, axis_2, coordinates_3[0], coordinates_3[1])
                local_covector = _euler_coordinate_gradient(axis_0_q, axis_1_q, axis_2_q, angular_component)
            angular_covector = wp.quat_rotate(q_p, local_covector)
    r_p = wp.vec3(0.0)
    if parent >= 0:
        r_p = wp.transform_get_translation(X_wp) - wp.transform_point(pose_p, body_com[parent])
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(pose_c, body_com[child])
    gradient_parent = wp.spatial_vector(-linear_axis, -wp.cross(r_p + x_err, linear_axis) - angular_covector)
    gradient_child = wp.spatial_vector(linear_axis, wp.cross(r_c, linear_axis) + angular_covector)
    return (coordinate, gradient_parent, gradient_child)

@wp.func
def evaluate_coordinate(data: JointCoordinateData, joint: int, component: int, body_q: wp.array[wp.transform], elastic_q: wp.array[float]):
    """Evaluate a coordinate on the current turn, together with its pose gradients.

    Reduced elastic endpoints are deformed with the modal coordinates in ``elastic_q``, which must
    pair with ``body_q``; the coordinate then includes endpoint rotation ``psi * q`` and the gradients
    use the deformed anchors.
    """
    angular_reference = wp.vec3()
    angular_start = data.q_start[joint] + data.dof_dim[joint, 0]
    for axis in range(data.dof_dim[joint, 1]):
        angular_reference[axis] = data.history[angular_start + axis]
    X_pj = _coordinate_frame(data, joint, data.parent[joint], True, data.X_p[joint], elastic_q)
    X_cj = _coordinate_frame(data, joint, data.child[joint], False, data.X_c[joint], elastic_q)
    q, g_p, g_c = eval_joint_coordinate_in_frames(joint, component, body_q, data.body_com, data.joint_type, data.parent, data.child, X_pj, X_cj, data.qd_start, data.dof_dim, data.axis, angular_reference, True)
    q = unwrap_joint_coordinate(q, data.history[data.q_start[joint] + component], component, data.dof_dim[joint, 0])
    return (q, g_p, g_c)

@wp.func
def evaluate_coordinate(data: JointCoordinateData, joint: int, component: int, body_q: wp.array[wp.transform]):
    """Evaluate a coordinate at ``body_q`` with the current modal coordinates ``data.elastic_q``."""
    return evaluate_coordinate(data, joint, component, body_q, data.elastic_q)

@wp.func
def evaluate_coordinate_prev(data: JointCoordinateData, joint: int, component: int, body_q_prev: wp.array[wp.transform]):
    """Evaluate a coordinate at previous-step poses with the previous modal coordinates."""
    return evaluate_coordinate(data, joint, component, body_q_prev, data.elastic_q_prev)

@wp.func
def _evaluate_angular_coordinate(data: JointCoordinateData, joint: int, component: int, body_q: wp.array[wp.transform], body_q_prev: wp.array[wp.transform], dt: float):
    """Measure the continuous angle, step velocity, and world-space angular gradient."""
    q, _g_p, g_c = evaluate_coordinate(data, joint, component, body_q)
    q_prev, _g_p_prev, _g_c_prev = evaluate_coordinate_prev(data, joint, component, body_q_prev)
    delta = q - q_prev
    if wp.abs(delta) >= wp.pi:
        delta = wp.atan2(wp.sin(delta), wp.cos(delta))
    rate = delta / dt
    return (q, rate, wp.spatial_bottom(g_c))

@wp.struct
class JointAxisDriveLimit:
    """Drive and limit data for one joint DOF.

    ``material_*_ke`` retain authored stiffness for activation and compliant ALM.
    ``drive_ke``/``limit_ke`` are the solve stiffnesses, capped by the legacy
    ramping penalty. Drive damping may independently activate a drive row.
    """
    material_drive_ke: float
    material_limit_ke: float
    drive_ke: float
    limit_ke: float
    drive_kd: float
    limit_kd: float
    target_pos: float
    target_vel: float
    lower: float
    upper: float

_DRIVE_LIMIT_MODE_DRIVE = wp.constant(3)

_COMPLIANT_ALM_BILATERAL_MIN_MATERIAL_FRACTION = 0.9

_COMPLIANT_ALM_BILATERAL_MIN_RHO_OVER_K = wp.constant(_COMPLIANT_ALM_BILATERAL_MIN_MATERIAL_FRACTION / (1.0 - _COMPLIANT_ALM_BILATERAL_MIN_MATERIAL_FRACTION))

@wp.func
def _bilateral_auto_rho(row_support: float, material_k: float):
    """Return ``max(H, 9K)``, ensuring ``k_eff >= 0.9K`` in each local solve."""
    if material_k <= 0.0:
        return 0.0
    return wp.max(row_support, _COMPLIANT_ALM_BILATERAL_MIN_RHO_OVER_K * material_k)

@wp.func
def _drive_auto_rho(axis_support: float, material_k: float):
    """Bilateral auto-rho for one drive axis."""
    return _bilateral_auto_rho(axis_support, material_k)

@wp.func
def _spring_damper_step(error: float, rate_error: float, material_k: float, material_d: float, inv_dt: float):
    """Return ``(e_row, K + D/dt)`` so ``K*e + D*rate = (K + D/dt)*e_row``."""
    step_k = material_k + material_d * inv_dt
    if step_k <= 0.0:
        return (0.0, 0.0)
    return ((material_k * error + material_d * rate_error) / step_k, step_k)

@wp.func
def _evaluate_drive_axis(error: float, rate: float, target_rate: float, active_mode: int, drive_k: float, drive_d: float, axis_support: float, drive_lambda: float, inv_dt: float):
    """Compliant ALM bilateral drive force/Hessian."""
    if active_mode != _DRIVE_LIMIT_MODE_DRIVE or axis_support <= 0.0:
        return (0.0, 0.0)
    row_error, row_k = _spring_damper_step(error, rate - target_rate, drive_k, drive_d, inv_dt)
    if row_k <= 0.0:
        return (0.0, 0.0)
    rho = _drive_auto_rho(axis_support, row_k)
    s, k_eff, _a = _compliant_alm_coefficients(row_k, rho)
    return (k_eff * row_error + s * drive_lambda, k_eff)

_DRIVE_LIMIT_MODE_LIMIT_LOWER = wp.constant(1)

_DRIVE_LIMIT_MODE_LIMIT_UPPER = wp.constant(2)

@wp.func
def _evaluate_drive_limit_axis(error: float, rate: float, target_rate: float, active_mode: int, drive_k: float, drive_d: float, limit_k: float, limit_d: float, inv_dt: float):
    """Legacy mutex drive/limit spring: one active mode, no dual.

    The damper is an explicit additive term (not folded into the residual).
    """
    if active_mode == _DRIVE_LIMIT_MODE_DRIVE:
        return (drive_k * error + drive_d * (rate - target_rate), drive_k + drive_d * inv_dt)
    if active_mode == _DRIVE_LIMIT_MODE_LIMIT_LOWER or active_mode == _DRIVE_LIMIT_MODE_LIMIT_UPPER:
        return (limit_k * error + limit_d * rate, limit_k + limit_d * inv_dt)
    return (0.0, 0.0)

_DRIVE_LIMIT_MODE_NONE = wp.constant(0)

@wp.func
def _limit_auto_rho(axis_support: float, material_k: float):
    """Return support as limit rho, or zero if the row is off."""
    if axis_support <= 0.0 or material_k <= 0.0:
        return 0.0
    return axis_support

@wp.func
def _limit_signed_gap(q: float, lim_lower: float, lim_upper: float, bound: int):
    """Return ``q - bound`` for the selected limit, or zero if none."""
    if bound == _DRIVE_LIMIT_MODE_LIMIT_LOWER:
        return q - lim_lower
    if bound == _DRIVE_LIMIT_MODE_LIMIT_UPPER:
        return q - lim_upper
    return 0.0

MAXVAL = 10000000000.0

@wp.func
def _nearest_limit_bound(q: float, lim_lower: float, lim_upper: float, has_limits: bool):
    """Pick the nearer finite bound for the limit residual (not the violated one).

    Violation-gated selection chatters at a resting bound where gap is zero but
    the multiplier is nonzero. Proximity keeps the bound stable through that
    fixed point.
    """
    if not has_limits:
        return _DRIVE_LIMIT_MODE_NONE
    if lim_lower <= -MAXVAL:
        return _DRIVE_LIMIT_MODE_LIMIT_UPPER
    if lim_upper >= MAXVAL:
        return _DRIVE_LIMIT_MODE_LIMIT_LOWER
    if q - lim_lower <= lim_upper - q:
        return _DRIVE_LIMIT_MODE_LIMIT_LOWER
    return _DRIVE_LIMIT_MODE_LIMIT_UPPER

@wp.func
def _evaluate_limit_axis(q: float, lim_lower: float, lim_upper: float, has_limits: bool, limit_k: float, limit_d: float, rate: float, axis_support: float, limit_lambda: float, inv_dt: float):
    """Evaluate a compliant unilateral limit alongside the independent drive row.

    Keep only forces directed into the selected bound, allowing a resting limit
    to retain its multiplier while releasing forces directed away. Legacy mode
    does not call this function.
    """
    bound = _nearest_limit_bound(q, lim_lower, lim_upper, has_limits)
    if bound == _DRIVE_LIMIT_MODE_NONE or limit_k <= 0.0 or axis_support <= 0.0:
        return (0.0, 0.0)
    gap = _limit_signed_gap(q, lim_lower, lim_upper, bound)
    row_error, row_k = _spring_damper_step(gap, rate, limit_k, limit_d, inv_dt)
    rho = _limit_auto_rho(axis_support, row_k)
    s, k_eff, _a = _compliant_alm_coefficients(row_k, rho)
    force = k_eff * row_error + s * limit_lambda
    if bound == _DRIVE_LIMIT_MODE_LIMIT_LOWER and force >= 0.0:
        return (0.0, 0.0)
    if bound == _DRIVE_LIMIT_MODE_LIMIT_UPPER and force <= 0.0:
        return (0.0, 0.0)
    return (force, k_eff)

@wp.func
def resolve_drive_limit_mode(q: float, target_pos: float, lim_lower: float, lim_upper: float, has_drive: bool, has_limits: bool):
    """Resolve drive/limit priority and compute position error [m or rad].

    Limits take precedence: if q is outside [lower, upper], the active limit
    wins. Otherwise the drive engages with target clamped to the limit range.

    Returns:
        (mode, err_pos) -- active mode constant and signed position error.
    """
    mode = _DRIVE_LIMIT_MODE_NONE
    err_pos = float(0.0)
    drive_target = target_pos
    if has_limits:
        drive_target = wp.clamp(target_pos, lim_lower, lim_upper)
        if q < lim_lower:
            mode = _DRIVE_LIMIT_MODE_LIMIT_LOWER
            err_pos = q - lim_lower
        elif q > lim_upper:
            mode = _DRIVE_LIMIT_MODE_LIMIT_UPPER
            err_pos = q - lim_upper
    if mode == _DRIVE_LIMIT_MODE_NONE and has_drive:
        mode = _DRIVE_LIMIT_MODE_DRIVE
        err_pos = q - drive_target
    return (mode, err_pos)

@wp.func
def _resolve_active_drive_row(q: float, target_pos: float, lim_lower: float, lim_upper: float, has_drive: bool, has_limits: bool, use_compliant_alm: int):
    """Legacy mutex mode, or ALM drive mode with target clamped into limits.

    ALM clamps so the drive never fights the separate limit row.
    """
    if use_compliant_alm == 0:
        return resolve_drive_limit_mode(q, target_pos, lim_lower, lim_upper, has_drive, has_limits)
    if not has_drive:
        return (_DRIVE_LIMIT_MODE_NONE, 0.0)
    drive_target = target_pos
    if has_limits:
        drive_target = wp.clamp(target_pos, lim_lower, lim_upper)
    return (_DRIVE_LIMIT_MODE_DRIVE, q - drive_target)

@wp.func
def _eval_joint_axis_drive_limit(axis: JointAxisDriveLimit, q: float, rate: float, has_drive: bool, has_limits: bool, axis_support: float, drive_lambda: float, limit_lambda: float, use_compliant_alm: int, inv_dt: float):
    """One DOF drive/limit force and Hessian: legacy mutex or ALM superposition."""
    active_mode, err_pos = _resolve_active_drive_row(q, axis.target_pos, axis.lower, axis.upper, has_drive, has_limits, use_compliant_alm)
    if use_compliant_alm == 0:
        return _evaluate_drive_limit_axis(err_pos, rate, axis.target_vel, active_mode, axis.drive_ke, axis.drive_kd, axis.limit_ke, axis.limit_kd, inv_dt)
    f_drive, H_drive = _evaluate_drive_axis(err_pos, rate, axis.target_vel, active_mode, axis.drive_ke, axis.drive_kd, axis_support, drive_lambda, inv_dt)
    f_limit, H_limit = _evaluate_limit_axis(q, axis.lower, axis.upper, has_limits, axis.limit_ke, axis.limit_kd, rate, axis_support, limit_lambda, inv_dt)
    return (f_drive + f_limit, H_drive + H_limit)

@wp.func
def _limit_row_exists(limit_ke: float, lower: float, upper: float):
    """Whether an authored limit row exists (stiffness and a finite bound)."""
    return limit_ke > 0.0 and (lower > -MAXVAL or upper < MAXVAL)

@wp.func
def _load_joint_axis_drive_limit(dof_idx: int, target_q_idx: int, penalty_slot: int, joint_target_ke: wp.array[float], joint_target_kd: wp.array[float], joint_target_q: wp.array[float], joint_target_qd: wp.array[float], joint_limit_lower: wp.array[float], joint_limit_upper: wp.array[float], joint_limit_ke: wp.array[float], joint_limit_kd: wp.array[float], joint_penalty_k: wp.array[float], use_compliant_alm: int):
    """Gather one DOF's drive/limit coefficients from the model arrays."""
    axis = JointAxisDriveLimit()
    axis.material_drive_ke = joint_target_ke[dof_idx]
    axis.material_limit_ke = joint_limit_ke[dof_idx]
    axis.drive_kd = joint_target_kd[dof_idx]
    axis.limit_kd = joint_limit_kd[dof_idx]
    axis.target_pos = joint_target_q[target_q_idx]
    axis.target_vel = joint_target_qd[dof_idx]
    axis.lower = joint_limit_lower[dof_idx]
    axis.upper = joint_limit_upper[dof_idx]
    axis.drive_ke = axis.material_drive_ke
    axis.limit_ke = axis.material_limit_ke
    if use_compliant_alm == 0:
        penalty_ke = joint_penalty_k[penalty_slot]
        axis.drive_ke = wp.min(penalty_ke, axis.material_drive_ke)
        axis.limit_ke = wp.min(penalty_ke, axis.material_limit_ke)
    return axis

@wp.func
def _evaluate_sparse_drive_limit(dof_idx: int, target_q_idx: int, penalty_slot: int, q: float, rate: float, joint_target_ke: wp.array[float], joint_target_kd: wp.array[float], joint_target_q: wp.array[float], joint_target_vel: wp.array[float], joint_limit_lower: wp.array[float], joint_limit_upper: wp.array[float], joint_limit_ke: wp.array[float], joint_limit_kd: wp.array[float], joint_penalty_k: wp.array[float], joint_drive_limit_support: wp.array[float], joint_drive_lambda: wp.array[float], joint_limit_lambda: wp.array[float], joint_compliant_alm: int, dt: float):
    axis = _load_joint_axis_drive_limit(dof_idx, target_q_idx, penalty_slot, joint_target_ke, joint_target_kd, joint_target_q, joint_target_vel, joint_limit_lower, joint_limit_upper, joint_limit_ke, joint_limit_kd, joint_penalty_k, joint_compliant_alm)
    has_drive = _drive_row_applies_force(axis.material_drive_ke, axis.drive_kd)
    has_limits = _limit_row_exists(axis.material_limit_ke, axis.lower, axis.upper)
    if has_drive or has_limits:
        return _eval_joint_axis_drive_limit(axis, q, rate, has_drive, has_limits, joint_drive_limit_support[dof_idx], joint_drive_lambda[dof_idx], joint_limit_lambda[dof_idx], joint_compliant_alm, 1.0 / dt)
    return (float(0.0), float(0.0))

@wp.func
def _sparse_angular_drive_limit_row(joint_coordinates: JointCoordinateData, joint: int, coordinate: int, dof_idx: int, target_q_idx: int, penalty_slot: int, body_q: wp.array[wp.transform], body_q_prev: wp.array[wp.transform], joint_target_ke: wp.array[float], joint_target_kd: wp.array[float], joint_target_q: wp.array[float], joint_target_vel: wp.array[float], joint_limit_lower: wp.array[float], joint_limit_upper: wp.array[float], joint_limit_ke: wp.array[float], joint_limit_kd: wp.array[float], joint_penalty_k: wp.array[float], joint_drive_limit_support: wp.array[float], joint_drive_lambda: wp.array[float], joint_limit_lambda: wp.array[float], joint_compliant_alm: int, dt: float):
    """Return ``(force, hessian, angular_jacobian_world)`` of an angular DOF's drive and limit row.

    ``hessian`` is zero when the DOF has neither a drive nor a limit row.
    """
    has_drive = _drive_row_applies_force(joint_target_ke[dof_idx], joint_target_kd[dof_idx])
    has_limits = _limit_row_exists(joint_limit_ke[dof_idx], joint_limit_lower[dof_idx], joint_limit_upper[dof_idx])
    if has_drive or has_limits:
        theta_abs, dtheta_dt, angular_jacobian_world = _evaluate_angular_coordinate(joint_coordinates, joint, coordinate, body_q, body_q_prev, dt)
        force_scalar, hessian_scalar = _evaluate_sparse_drive_limit(dof_idx, target_q_idx, penalty_slot, theta_abs, dtheta_dt, joint_target_ke, joint_target_kd, joint_target_q, joint_target_vel, joint_limit_lower, joint_limit_upper, joint_limit_ke, joint_limit_kd, joint_penalty_k, joint_drive_limit_support, joint_drive_lambda, joint_limit_lambda, joint_compliant_alm, dt)
        return (force_scalar, hessian_scalar, angular_jacobian_world)
    return (float(0.0), float(0.0), wp.vec3(0.0))

@wp.func
def _sparse_joint_armature_row(joint_coordinates: JointCoordinateData, joint: int, qd_start: int, body_q: wp.array[wp.transform], joint_armature: wp.array[float], joint_armature_target: wp.array[float], dt: float):
    """Return ``(active, g_parent, g_child, force, hessian)`` of a revolute joint's armature row."""
    armature = joint_armature[qd_start]
    if armature > 0.0:
        q, gp, gc = evaluate_coordinate(joint_coordinates, joint, 0, body_q)
        armature_hessian = armature / (dt * dt)
        armature_force = armature_hessian * (q - joint_armature_target[qd_start])
        return (True, gp, gc, armature_force, armature_hessian)
    return (False, wp.spatial_vector(), wp.spatial_vector(), float(0.0), float(0.0))

@wp.func
def _sparse_joint_frame(elastic: SparseElasticJointData, joint: int, body: int, is_parent_side: bool, X: wp.transform, use_prev: bool) -> wp.transform:
    if elastic.enabled == 0:
        return X
    joint_q = elastic.joint_q
    if use_prev:
        joint_q = elastic.joint_q_prev
    return eval_elastic_endpoint_xform(joint, body, is_parent_side, X, elastic.body_elastic_index, elastic.elastic_joint, elastic.elastic_mode_count, joint_q, elastic.joint_q_start, elastic.parent_endpoint, elastic.child_endpoint, elastic.endpoint_phi, elastic.endpoint_psi, elastic.max_mode_count)

_JOINT_FRICTION_SMOOTHING_VELOCITY = wp.constant(0.01)

@wp.func
def _eval_joint_axis_friction(rate: float, friction: float, inv_dt: float):
    """Evaluate smooth Coulomb friction with a positive secant majorizer.

    The exact tanh derivative vanishes during sliding. Using it in an
    unrestricted Newton step can overshoot zero velocity and add energy.
    The secant stiffness bounds that step while preserving the friction law
    at convergence (including its small regularized creep near rest).
    """
    force = float(0.0)
    hessian = float(0.0)
    if friction > 0.0:
        inv_eps = 1.0 / _JOINT_FRICTION_SMOOTHING_VELOCITY
        direction = wp.tanh(rate * inv_eps)
        force = friction * direction
        slope = friction * inv_eps
        if wp.abs(rate) > 1e-08:
            slope = force / rate
        hessian = slope * inv_dt
    return (force, hessian)

@wp.func
def _sparse_joint_friction_row(joint_coordinates: JointCoordinateData, joint: int, component: int, lin_count: int, dof: int, body_q: wp.array[wp.transform], body_q_prev: wp.array[wp.transform], joint_friction: wp.array[float], joint_damping: wp.array[float], dt: float):
    """Return ``(active, g_parent, g_child, force, hessian)`` of a free DOF's friction and damping row."""
    friction = joint_friction[dof]
    damping = wp.max(joint_damping[dof], 0.0)
    if friction > 0.0 or damping > 0.0:
        q, gp, gc = evaluate_coordinate(joint_coordinates, joint, component, body_q)
        q_prev, _gp_prev, _gc_prev = evaluate_coordinate_prev(joint_coordinates, joint, component, body_q_prev)
        delta = q - q_prev
        if component >= lin_count:
            delta = wp.atan2(wp.sin(delta), wp.cos(delta))
        rate = delta / dt
        force, hessian = _eval_joint_axis_friction(rate, friction, 1.0 / dt)
        return (True, gp, gc, force + damping * rate, hessian + damping / dt)
    return (False, wp.spatial_vector(), wp.spatial_vector(), float(0.0), float(0.0))

@wp.struct
class _SparseJointGeometry:
    """Endpoint frames and structural coefficients of one non-ROD joint, shared by the sparse joint walkers."""
    child_anchor: wp.vec3
    child_anchor_prev: wp.vec3
    child_anchor_q: wp.quat
    child_anchor_q_prev: wp.quat
    child_rest_q: wp.quat
    parent_anchor: wp.vec3
    parent_anchor_prev: wp.vec3
    parent_anchor_q: wp.quat
    parent_anchor_q_prev: wp.quat
    parent_rest_q: wp.quat
    c_start: int
    qd_start: int
    lin_count: int
    ang_count: int
    P_lin: wp.mat33
    P_ang: wp.mat33
    lin_lambda: wp.vec3
    lin_C0: wp.vec3
    lin_alpha: float
    ang_lambda: wp.vec3
    ang_C0: wp.vec3
    ang_alpha: float
    k_linear: float
    k_angular: float
    kd_linear: float
    kd_angular: float

@wp.func
def _joint_projectors(jt: int, joint_axis: wp.array[wp.vec3], qd_start: int, lin_count: int, ang_count: int, parent_anchor_q: wp.quat):
    P_lin = wp.identity(3, float)
    P_ang = wp.identity(3, float)
    if jt == JointType.PRISMATIC:
        axis_w = wp.normalize(wp.quat_rotate(parent_anchor_q, joint_axis[qd_start]))
        P_lin = P_lin - wp.outer(axis_w, axis_w)
    elif jt == JointType.D6:
        if lin_count > 0:
            axis_l0 = wp.normalize(wp.quat_rotate(parent_anchor_q, joint_axis[qd_start]))
            P_lin = P_lin - wp.outer(axis_l0, axis_l0)
        if lin_count > 1:
            axis_l1 = wp.normalize(wp.quat_rotate(parent_anchor_q, joint_axis[qd_start + 1]))
            P_lin = P_lin - wp.outer(axis_l1, axis_l1)
        if lin_count > 2:
            axis_l2 = wp.normalize(wp.quat_rotate(parent_anchor_q, joint_axis[qd_start + 2]))
            P_lin = P_lin - wp.outer(axis_l2, axis_l2)
    if jt == JointType.REVOLUTE:
        axis = wp.normalize(joint_axis[qd_start])
        P_ang = P_ang - wp.outer(axis, axis)
    elif jt == JointType.D6:
        if ang_count > 0:
            axis_a0 = wp.normalize(joint_axis[qd_start + lin_count])
            P_ang = P_ang - wp.outer(axis_a0, axis_a0)
        if ang_count > 1:
            axis_a1 = wp.normalize(joint_axis[qd_start + lin_count + 1])
            P_ang = P_ang - wp.outer(axis_a1, axis_a1)
        if ang_count > 2:
            axis_a2 = wp.normalize(joint_axis[qd_start + lin_count + 2])
            P_ang = P_ang - wp.outer(axis_a2, axis_a2)
    return (P_lin, P_ang)

@wp.func
def _load_sparse_structural_coefficients(jt: int, c_start: int, joint_penalty_k: wp.array[float], joint_rho: wp.array[float], joint_material_k: wp.array[float], joint_penalty_kd: wp.array[float], joint_compliant_alm: int):
    solve_weight_linear = float(0.0)
    solve_weight_angular = float(0.0)
    material_k_linear = float(0.0)
    material_k_angular = float(0.0)
    kd_linear = float(0.0)
    kd_angular = float(0.0)
    if jt == JointType.BALL:
        solve_weight_linear = _load_solve_weight(joint_penalty_k, joint_rho, c_start, joint_compliant_alm)
        material_k_linear = joint_material_k[c_start]
        kd_linear = joint_penalty_kd[c_start]
    else:
        solve_weight_linear = _load_solve_weight(joint_penalty_k, joint_rho, c_start, joint_compliant_alm)
        solve_weight_angular = _load_solve_weight(joint_penalty_k, joint_rho, c_start + 1, joint_compliant_alm)
        material_k_linear = joint_material_k[c_start]
        material_k_angular = joint_material_k[c_start + 1]
        kd_linear = joint_penalty_kd[c_start]
        kd_angular = joint_penalty_kd[c_start + 1]
    return (solve_weight_linear, solve_weight_angular, material_k_linear, material_k_angular, kd_linear, kd_angular)

@wp.func
def _sparse_joint_geometry(joint: int, jt: int, parent: int, child: int, X_p: wp.transform, X_c: wp.transform, X_p_prev: wp.transform, X_c_prev: wp.transform, body_q: wp.array[wp.transform], body_q_prev: wp.array[wp.transform], body_q_rest: wp.array[wp.transform], joint_X_p: wp.array[wp.transform], joint_X_c: wp.array[wp.transform], joint_axis: wp.array[wp.vec3], joint_qd_start: wp.array[int], joint_constraint_start: wp.array[int], joint_penalty_k: wp.array[float], joint_rho: wp.array[float], joint_material_k: wp.array[float], joint_penalty_kd: wp.array[float], joint_dof_dim: wp.array2d[int], joint_lambda_lin: wp.array[wp.vec3], joint_lambda_ang: wp.array[wp.vec3], joint_C0_lin: wp.array[wp.vec3], joint_C0_ang: wp.array[wp.vec3], joint_is_hard: wp.array[wp.int32], avbd_alpha: float, joint_compliant_alm: int) -> _SparseJointGeometry:
    """Evaluate the anchor frames, projectors and structural coefficients of a non-ROD joint."""
    g = _SparseJointGeometry()
    child_pose = body_q[child]
    child_prev_pose = body_q_prev[child]
    child_rest_pose = body_q_rest[child]
    g.child_anchor = wp.transform_point(child_pose, wp.transform_get_translation(X_c))
    g.child_anchor_prev = wp.transform_point(child_prev_pose, wp.transform_get_translation(X_c_prev))
    g.child_anchor_q = wp.mul(wp.transform_get_rotation(child_pose), wp.transform_get_rotation(X_c))
    g.child_anchor_q_prev = wp.mul(wp.transform_get_rotation(child_prev_pose), wp.transform_get_rotation(X_c_prev))
    g.child_rest_q = wp.mul(wp.transform_get_rotation(child_rest_pose), wp.transform_get_rotation(joint_X_c[joint]))
    if parent >= 0:
        parent_pose = body_q[parent]
        parent_prev_pose = body_q_prev[parent]
        parent_rest_pose = body_q_rest[parent]
        g.parent_anchor = wp.transform_point(parent_pose, wp.transform_get_translation(X_p))
        g.parent_anchor_prev = wp.transform_point(parent_prev_pose, wp.transform_get_translation(X_p_prev))
        g.parent_anchor_q = wp.mul(wp.transform_get_rotation(parent_pose), wp.transform_get_rotation(X_p))
        g.parent_anchor_q_prev = wp.mul(wp.transform_get_rotation(parent_prev_pose), wp.transform_get_rotation(X_p_prev))
        g.parent_rest_q = wp.mul(wp.transform_get_rotation(parent_rest_pose), wp.transform_get_rotation(joint_X_p[joint]))
    else:
        g.parent_anchor = wp.transform_get_translation(X_p)
        g.parent_anchor_prev = g.parent_anchor
        g.parent_anchor_q = wp.transform_get_rotation(X_p)
        g.parent_anchor_q_prev = g.parent_anchor_q
        g.parent_rest_q = g.parent_anchor_q
    c_start = joint_constraint_start[joint]
    qd_start = joint_qd_start[joint]
    lin_count = int(0)
    ang_count = int(0)
    if jt == JointType.REVOLUTE:
        ang_count = 1
    elif jt == JointType.PRISMATIC:
        lin_count = 1
    elif jt == JointType.D6:
        lin_count = joint_dof_dim[joint, 0]
        ang_count = joint_dof_dim[joint, 1]
    g.c_start = c_start
    g.qd_start = qd_start
    g.lin_count = lin_count
    g.ang_count = ang_count
    P_lin = wp.identity(3, float)
    P_ang = wp.identity(3, float)
    if jt == JointType.REVOLUTE or jt == JointType.PRISMATIC or jt == JointType.D6:
        P_lin, P_ang = _joint_projectors(jt, joint_axis, qd_start, lin_count, ang_count, g.parent_anchor_q)
    g.P_lin = P_lin
    g.P_ang = P_ang
    lin_lambda = wp.vec3(0.0)
    lin_C0 = wp.vec3(0.0)
    lin_alpha = float(0.0)
    linear_hard = joint_is_hard[c_start] == 1
    if linear_hard or joint_compliant_alm == 1:
        lin_lambda = joint_lambda_lin[joint]
        lin_C0 = joint_C0_lin[joint]
        lin_alpha = avbd_alpha
    ang_lambda = wp.vec3(0.0)
    ang_C0 = wp.vec3(0.0)
    ang_alpha = float(0.0)
    ang_hard = int(0)
    if jt != JointType.BALL:
        ang_hard = joint_is_hard[c_start + 1]
        if ang_hard == 1 or joint_compliant_alm == 1:
            ang_lambda = joint_lambda_ang[joint]
            ang_C0 = joint_C0_ang[joint]
            ang_alpha = avbd_alpha
    solve_weight_linear, solve_weight_angular, material_k_linear, material_k_angular, kd_linear, kd_angular = _load_sparse_structural_coefficients(jt, c_start, joint_penalty_k, joint_rho, joint_material_k, joint_penalty_kd, joint_compliant_alm)
    k_linear, lin_lambda = _material_force_terms(solve_weight_linear, material_k_linear, lin_lambda, joint_compliant_alm)
    k_angular, ang_lambda = _material_force_terms(solve_weight_angular, material_k_angular, ang_lambda, joint_compliant_alm)
    g.lin_lambda = lin_lambda
    g.lin_C0 = lin_C0
    g.lin_alpha = lin_alpha
    g.ang_lambda = ang_lambda
    g.ang_C0 = ang_C0
    g.ang_alpha = ang_alpha
    g.k_linear = k_linear
    g.k_angular = k_angular
    g.kd_linear = kd_linear
    g.kd_angular = kd_angular
    return g

@wp.func
def _sparse_joint_supported(jt: int) -> bool:
    return jt == JointType.ROD or jt == JointType.BALL or jt == JointType.FIXED or (jt == JointType.REVOLUTE) or (jt == JointType.PRISMATIC) or (jt == JointType.D6)

@wp.func
def _sparse_linear_drive_limit_row(dof_idx: int, target_q_idx: int, penalty_slot: int, parent_anchor_q: wp.quat, C_vec: wp.vec3, C_vec_prev: wp.vec3, joint_axis: wp.array[wp.vec3], joint_target_ke: wp.array[float], joint_target_kd: wp.array[float], joint_target_q: wp.array[float], joint_target_vel: wp.array[float], joint_limit_lower: wp.array[float], joint_limit_upper: wp.array[float], joint_limit_ke: wp.array[float], joint_limit_kd: wp.array[float], joint_penalty_k: wp.array[float], joint_drive_limit_support: wp.array[float], joint_drive_lambda: wp.array[float], joint_limit_lambda: wp.array[float], joint_compliant_alm: int, dt: float):
    """Return ``(force, hessian, axis_world)`` of a linear DOF's drive and limit row.

    ``C_vec`` is the child-minus-parent anchor separation; ``hessian`` is zero when the DOF has
    neither a drive nor a limit row.
    """
    has_drive = _drive_row_applies_force(joint_target_ke[dof_idx], joint_target_kd[dof_idx])
    has_limits = _limit_row_exists(joint_limit_ke[dof_idx], joint_limit_lower[dof_idx], joint_limit_upper[dof_idx])
    if has_drive or has_limits:
        axis_world = wp.normalize(wp.quat_rotate(parent_anchor_q, joint_axis[dof_idx]))
        d_along = wp.dot(C_vec, axis_world)
        dd_dt = wp.dot((C_vec - C_vec_prev) / dt, axis_world)
        force_scalar, hessian_scalar = _evaluate_sparse_drive_limit(dof_idx, target_q_idx, penalty_slot, d_along, dd_dt, joint_target_ke, joint_target_kd, joint_target_q, joint_target_vel, joint_limit_lower, joint_limit_upper, joint_limit_ke, joint_limit_kd, joint_penalty_k, joint_drive_limit_support, joint_drive_lambda, joint_limit_lambda, joint_compliant_alm, dt)
        return (force_scalar, hessian_scalar, axis_world)
    return (float(0.0), float(0.0), wp.vec3(0.0))

@wp.func
def _diag_mul_mat33(d: wp.vec3, m: wp.mat33) -> wp.mat33:
    """Return diag(d) * m without building a dense diagonal matrix."""
    return wp.matrix_from_rows(d[0] * wp.vec3(m[0, 0], m[0, 1], m[0, 2]), d[1] * wp.vec3(m[1, 0], m[1, 1], m[1, 2]), d[2] * wp.vec3(m[2, 0], m[2, 1], m[2, 2]))

@wp.func
def _wrap_principal_angle(angle: float) -> float:
    """Wrap a bounded angular difference into ``[-pi, pi]``.

    Args:
        angle: Difference of two principal angles. The caller guarantees a
            value in ``[-2*pi, 2*pi]``.

    Returns:
        Equivalent principal angular difference.
    """
    if angle > wp.pi:
        angle -= 2.0 * wp.pi
    elif angle < -wp.pi:
        angle += 2.0 * wp.pi
    return angle

@wp.func
def assemble_geometric_rod_kappa_z(q_wp: wp.quat, kb_now_world: wp.vec3, twist_now: float, kb_rest_local: wp.vec3, twist_rest: float) -> wp.vec3:
    """Assemble [bend_x, bend_y, twist_z] for local +Z rod joints."""
    bend_now_local = wp.quat_rotate(wp.quat_inverse(q_wp), kb_now_world)
    bend_residual_local = bend_now_local - kb_rest_local
    twist_residual = _wrap_principal_angle(twist_now - twist_rest)
    return wp.vec3(bend_residual_local[0], bend_residual_local[1], twist_residual)

@wp.func
def rod_bend_twist_delta(kappa: wp.vec3, kappa_prev: wp.vec3) -> wp.vec3:
    """Return a temporal rod bend/twist strain increment.

    Args:
        kappa: Current ``[bend_x, bend_y, twist_z]`` strain.
        kappa_prev: Previous strain in the same representation.

    Returns:
        Bend increments from ordinary subtraction and the shortest signed
        principal-angle increment for twist.
    """
    return wp.vec3(kappa[0] - kappa_prev[0], kappa[1] - kappa_prev[1], _wrap_principal_angle(kappa[2] - kappa_prev[2]))

@wp.func
def evaluate_rod_bend_twist_force_hessian_z(q_wp: wp.quat, q_wc: wp.quat, kb_rest_local: wp.vec3, twist_rest: float, q_wp_prev: wp.quat, q_wc_prev: wp.quat, is_parent: bool, K_elastic_diag: wp.vec3, C0_force: wp.vec3, sigma0: wp.vec3, H_fric_diag: wp.vec3, lambda_projected: wp.vec3, K_damp_diag: wp.vec3, damping_active: bool, dt: float):
    """Bend/twist torque and Hessian for SolverVBD local +Z rod joints.

    In the fixed rod material basis, local angular operators are diagonal:
    ``[bend_x, bend_y, twist_z]``. Keep them as vec3 row scales in the hot path
    instead of building dense local matrices.
    """
    inv_dt = 1.0 / dt
    measure = measure_rod_bend_twist_z(q_wp, q_wc)
    kappa_now_vec = assemble_geometric_rod_kappa_z(q_wp, measure.kb_world, measure.twist, kb_rest_local, twist_rest)
    f_local = wp.cw_mul(K_elastic_diag, kappa_now_vec) - C0_force + sigma0 + lambda_projected
    H_local_diag = K_elastic_diag + H_fric_diag
    if damping_active:
        prev_measure = measure_rod_bend_twist_z(q_wp_prev, q_wc_prev)
        kappa_prev_vec = assemble_geometric_rod_kappa_z(q_wp_prev, prev_measure.kb_world, prev_measure.twist, kb_rest_local, twist_rest)
        dkappa_dt = rod_bend_twist_delta(kappa_now_vec, kappa_prev_vec) * inv_dt
        f_local = f_local + wp.cw_mul(K_damp_diag, dkappa_dt)
        H_local_diag = H_local_diag + inv_dt * K_damp_diag
    J_body = rod_bend_twist_jacobian_z_from_measure(measure, is_parent)
    H_aa = wp.transpose(J_body) * _diag_mul_mat33(H_local_diag, J_body)
    tau_world = -(wp.transpose(J_body) * f_local)
    return (tau_world, H_aa, kappa_now_vec, J_body)

@wp.func
def evaluate_rod_stretch_shear_force_hessian(X_wp: wp.transform, X_wc: wp.transform, X_wp_prev: wp.transform, X_wc_prev: wp.transform, parent_pose: wp.transform, child_pose: wp.transform, parent_com: wp.vec3, child_com: wp.vec3, is_parent: bool, k_diag: wp.vec3, C0_force_local: wp.vec3, lambda_local: wp.vec3, kd_diag: wp.vec3, damping_active: bool, dt: float):
    """Rod-joint stretch/shear anchor force, torque, and PSD Gauss-Newton self-Hessian.

    All inputs are parent-material: residual ``u = R_p^T (x_c - x_p) =
    [shear_x, shear_y, stretch_z]``, diagonal ``k_diag = (k_shear, k_shear,
    k_stretch)`` / ``kd_diag``, and local ``C0_force_local`` / ``lambda_local``.
    Elastic and AL are energy gradients, damping is dissipative in ``u``, and both
    bodies react at the shared child anchor, so the net internal wrench is zero.
    """
    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)
    q_wp = wp.transform_get_rotation(X_wp)
    if is_parent:
        com_w = wp.transform_point(parent_pose, parent_com)
    else:
        com_w = wp.transform_point(child_pose, child_com)
    r = x_c - com_w
    C_vec = x_c - x_p
    u = wp.quat_rotate_inv(q_wp, C_vec)
    psi = wp.cw_mul(k_diag, u) - C0_force_local + lambda_local
    k_s = k_diag[0]
    k_z = k_diag[2]
    h_s = k_s
    h_z = k_z
    if damping_active:
        inv_dt = 1.0 / dt
        x_p_prev = wp.transform_get_translation(X_wp_prev)
        x_c_prev = wp.transform_get_translation(X_wc_prev)
        u_prev = wp.quat_rotate_inv(wp.transform_get_rotation(X_wp_prev), x_c_prev - x_p_prev)
        psi = psi + wp.cw_mul(kd_diag, (u - u_prev) * inv_dt)
        h_s = h_s + kd_diag[0] * inv_dt
        h_z = h_z + kd_diag[2] * inv_dt
    f_world = wp.quat_rotate(q_wp, psi)
    force = f_world if is_parent else -f_world
    t = quat_rotate_local_z(q_wp)
    identity = wp.identity(3, float)
    K_eff = h_s * identity + (h_z - h_s) * wp.outer(t, t)
    H_ll = K_eff
    if is_parent:
        k_iso = wp.min(k_s, k_z)
        K_material = K_eff - k_iso * identity
        r_elastic = x_p - com_w
        rx_material = wp.skew(r)
        H_al = k_iso * wp.skew(r_elastic) + rx_material * K_material
        H_aa = k_iso * (wp.length_sq(r_elastic) * identity - wp.outer(r_elastic, r_elastic))
        H_aa = H_aa + wp.transpose(rx_material) * K_material * rx_material
    else:
        rx = wp.skew(r)
        H_al = rx * K_eff
        H_aa = wp.transpose(rx) * K_eff * rx
    torque = wp.cross(r, force)
    return (force, torque, H_ll, H_al, H_aa)

@wp.func
def evaluate_rod_joint_force_hessian(body_index: int, joint_index: int, body_q: wp.array[wp.transform], body_q_prev: wp.array[wp.transform], body_com: wp.array[wp.vec3], joint_parent: wp.array[int], joint_child: wp.array[int], X_pj: wp.transform, X_cj: wp.transform, X_pj_prev: wp.transform, X_cj_prev: wp.transform, joint_rod_rest_kb_local: wp.array[wp.vec3], joint_rod_rest_twist: wp.array[float], joint_constraint_start: wp.array[int], joint_penalty_k: wp.array[float], joint_rho: wp.array[float], joint_material_k: wp.array[float], joint_penalty_kd: wp.array[float], joint_sigma_start: wp.array[wp.vec3], joint_C_fric: wp.array[wp.vec3], joint_lambda_lin: wp.array[wp.vec3], joint_lambda_ang: wp.array[wp.vec3], joint_C0_lin: wp.array[wp.vec3], joint_C0_ang: wp.array[wp.vec3], joint_is_hard: wp.array[wp.int32], stab_alpha: float, joint_compliant_alm: int, dt: float):
    """Compute exact split ROD force and self-Hessian terms for one body.

    ``X_pj``/``X_cj`` are the body-local parent/child attachment frames at the current
    iterate and ``X_pj_prev``/``X_cj_prev`` those at the previous step. Rigid callers
    pass ``joint_X_p[joint]``/``joint_X_c[joint]`` for both; reduced elastic endpoints
    pass their deformed frames so the rod measures strain at the deformed attachment.
    """
    parent_index = joint_parent[joint_index]
    child_index = joint_child[joint_index]
    is_parent_body = parent_index >= 0 and body_index == parent_index
    if parent_index >= 0:
        parent_pose = body_q[parent_index]
        parent_pose_prev = body_q_prev[parent_index]
        parent_com = body_com[parent_index]
    else:
        parent_pose = wp.transform(wp.vec3(0.0), wp.quat_identity())
        parent_pose_prev = parent_pose
        parent_com = wp.vec3(0.0)
    child_pose = body_q[child_index]
    child_pose_prev = body_q_prev[child_index]
    child_com = body_com[child_index]
    X_wp = parent_pose * X_pj
    X_wc = child_pose * X_cj
    X_wp_prev = parent_pose_prev * X_pj_prev
    X_wc_prev = child_pose_prev * X_cj_prev
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    q_wp_prev = wp.transform_get_rotation(X_wp_prev)
    q_wc_prev = wp.transform_get_rotation(X_wc_prev)
    c_start = joint_constraint_start[joint_index]
    stretch_idx = c_start
    shear_idx = c_start + 1
    bend_idx = c_start + 2
    twist_idx = c_start + 3
    solve_weight_stretch = _load_solve_weight(joint_penalty_k, joint_rho, stretch_idx, joint_compliant_alm)
    solve_weight_shear = _load_solve_weight(joint_penalty_k, joint_rho, shear_idx, joint_compliant_alm)
    solve_weight_bend = _load_solve_weight(joint_penalty_k, joint_rho, bend_idx, joint_compliant_alm)
    solve_weight_twist = _load_solve_weight(joint_penalty_k, joint_rho, twist_idx, joint_compliant_alm)
    material_stretch = joint_material_k[stretch_idx]
    material_shear = joint_material_k[shear_idx]
    material_bend = joint_material_k[bend_idx]
    material_twist = joint_material_k[twist_idx]
    kd_stretch = joint_penalty_kd[stretch_idx]
    kd_shear = joint_penalty_kd[shear_idx]
    kd_bend = joint_penalty_kd[bend_idx]
    kd_twist = joint_penalty_kd[twist_idx]
    total_force = wp.vec3(0.0)
    total_torque = wp.vec3(0.0)
    total_H_ll = wp.mat33(0.0)
    total_H_al = wp.mat33(0.0)
    total_H_aa = wp.mat33(0.0)
    bend_stiff = _structural_row_has_stiffness(solve_weight_bend, material_bend, joint_compliant_alm)
    twist_stiff = _structural_row_has_stiffness(solve_weight_twist, material_twist, joint_compliant_alm)
    bend_active = bend_stiff or kd_bend > 0.0
    twist_active = twist_stiff or kd_twist > 0.0
    if bend_active or twist_active:
        lambda_ang = joint_lambda_ang[joint_index]
        C0_ang = joint_C0_ang[joint_index]
        bend_hard = bend_stiff and joint_compliant_alm == 0 and (joint_is_hard[bend_idx] == 1)
        twist_hard = twist_stiff and joint_compliant_alm == 0 and (joint_is_hard[twist_idx] == 1)
        bend_dual = wp.vec3(lambda_ang[0], lambda_ang[1], 0.0)
        twist_dual = wp.vec3(0.0, 0.0, lambda_ang[2])
        if joint_compliant_alm == 0 and (not bend_hard):
            bend_dual = wp.vec3(0.0)
        if joint_compliant_alm == 0 and (not twist_hard):
            twist_dual = wp.vec3(0.0)
        bend_primal_k, bend_dual_eff = _material_force_terms(solve_weight_bend, material_bend, bend_dual, joint_compliant_alm)
        twist_primal_k, twist_dual_eff = _material_force_terms(solve_weight_twist, material_twist, twist_dual, joint_compliant_alm)
        K_elastic_diag = wp.vec3(bend_primal_k, bend_primal_k, twist_primal_k)
        K_damp_diag = wp.vec3(kd_bend, kd_bend, kd_twist)
        bend_alpha = stab_alpha if joint_compliant_alm == 1 or bend_hard else 0.0
        twist_alpha = stab_alpha if joint_compliant_alm == 1 or twist_hard else 0.0
        C0_force = bend_primal_k * bend_alpha * wp.vec3(C0_ang[0], C0_ang[1], 0.0)
        C0_force = C0_force + twist_primal_k * twist_alpha * wp.vec3(0.0, 0.0, C0_ang[2])
        sigma = wp.vec3(0.0)
        H_fric_diag = wp.vec3(0.0)
        dahl_sigma = joint_sigma_start[joint_index]
        dahl_fric = joint_C_fric[joint_index]
        if bend_stiff and (not bend_hard):
            sigma = sigma + wp.vec3(dahl_sigma[0], dahl_sigma[1], 0.0)
            H_fric_diag = H_fric_diag + wp.vec3(dahl_fric[0], dahl_fric[1], 0.0)
        if twist_stiff and (not twist_hard):
            sigma = sigma + wp.vec3(0.0, 0.0, dahl_sigma[2])
            H_fric_diag = H_fric_diag + wp.vec3(0.0, 0.0, dahl_fric[2])
        rod_torque, rod_H_aa, _rod_kappa, _rod_J = evaluate_rod_bend_twist_force_hessian_z(q_wp, q_wc, joint_rod_rest_kb_local[joint_index], joint_rod_rest_twist[joint_index], q_wp_prev, q_wc_prev, is_parent_body, K_elastic_diag, C0_force, sigma, H_fric_diag, bend_dual_eff + twist_dual_eff, K_damp_diag, kd_bend > 0.0 or kd_twist > 0.0, dt)
        total_torque = total_torque + rod_torque
        total_H_aa = total_H_aa + rod_H_aa
    stretch_stiff = _structural_row_has_stiffness(solve_weight_stretch, material_stretch, joint_compliant_alm)
    shear_stiff = _structural_row_has_stiffness(solve_weight_shear, material_shear, joint_compliant_alm)
    stretch_active = stretch_stiff or kd_stretch > 0.0
    shear_active = shear_stiff or kd_shear > 0.0
    if stretch_active or shear_active:
        lambda_lin = joint_lambda_lin[joint_index]
        C0_lin = joint_C0_lin[joint_index]
        stretch_hard = stretch_stiff and joint_compliant_alm == 0 and (joint_is_hard[stretch_idx] == 1)
        shear_hard = shear_stiff and joint_compliant_alm == 0 and (joint_is_hard[shear_idx] == 1)
        stretch_dual = wp.vec3(0.0, 0.0, lambda_lin[2])
        shear_dual = wp.vec3(lambda_lin[0], lambda_lin[1], 0.0)
        if joint_compliant_alm == 0 and (not stretch_hard):
            stretch_dual = wp.vec3(0.0)
        if joint_compliant_alm == 0 and (not shear_hard):
            shear_dual = wp.vec3(0.0)
        stretch_primal_k, stretch_dual_eff = _material_force_terms(solve_weight_stretch, material_stretch, stretch_dual, joint_compliant_alm)
        shear_primal_k, shear_dual_eff = _material_force_terms(solve_weight_shear, material_shear, shear_dual, joint_compliant_alm)
        k_diag = wp.vec3(shear_primal_k, shear_primal_k, stretch_primal_k)
        kd_diag = wp.vec3(kd_shear, kd_shear, kd_stretch)
        stretch_alpha = stab_alpha if joint_compliant_alm == 1 or stretch_hard else 0.0
        shear_alpha = stab_alpha if joint_compliant_alm == 1 or shear_hard else 0.0
        C0_force_local = shear_primal_k * shear_alpha * wp.vec3(C0_lin[0], C0_lin[1], 0.0)
        C0_force_local = C0_force_local + stretch_primal_k * stretch_alpha * wp.vec3(0.0, 0.0, C0_lin[2])
        f_l, t_l, Hll_l, Hal_l, Haa_l = evaluate_rod_stretch_shear_force_hessian(X_wp, X_wc, X_wp_prev, X_wc_prev, parent_pose, child_pose, parent_com, child_com, is_parent_body, k_diag, C0_force_local, stretch_dual_eff + shear_dual_eff, kd_diag, kd_stretch > 0.0 or kd_shear > 0.0, dt)
        total_force = total_force + f_l
        total_torque = total_torque + t_l
        total_H_ll = total_H_ll + Hll_l
        total_H_al = total_H_al + Hal_l
        total_H_aa = total_H_aa + Haa_l
    return (total_force, total_torque, total_H_ll, total_H_al, total_H_aa)

@wp.kernel(enable_backward=False)
def restore_carry(values: wp.array[float], rhs: wp.array[float], dest: wp.array[float], dest_rhs: wp.array[float]):
    thread = wp.tid()
    base = thread // 32 * 256 + thread % 32
    for item in range(8):
        index = base + 32 * item
        if index < values.shape[0]:
            dest[index] = values[index]
        elif index < values.shape[0] + rhs.shape[0]:
            dest_rhs[index - values.shape[0]] = rhs[index - values.shape[0]]


@wp.kernel(enable_backward=False)
def restore_packets(values: wp.array[wp.vec4], rhs: wp.array[wp.vec4], dest: wp.array[wp.vec4], dest_rhs: wp.array[wp.vec4], tail_values: wp.array[float], tail_rhs: wp.array[float], tail_dest: wp.array[float], tail_dest_rhs: wp.array[float]):
    thread = wp.tid()
    value_packets = values.shape[0]
    total_packets = value_packets + rhs.shape[0]
    base = (thread // 32) * 128 + thread % 32
    for item in range(4):
        index = base + item * 32
        if index < value_packets:
            dest[index] = values[index]
        elif index < total_packets:
            dest_rhs[index - value_packets] = rhs[index - value_packets]
    if thread < tail_values.shape[0] % 4:
        tail_dest[value_packets * 4 + thread] = tail_values[value_packets * 4 + thread]
    if thread < tail_rhs.shape[0] % 4:
        tail_dest_rhs[rhs.shape[0] * 4 + thread] = tail_rhs[rhs.shape[0] * 4 + thread]

@wp.kernel(enable_backward=False)
def restore_and_index(values: wp.array[float], rhs: wp.array[float], dest: wp.array[float], dest_rhs: wp.array[float], value_packs: wp.array[wp.vec4], rhs_packs: wp.array[wp.vec4], dest_packs: wp.array[wp.vec4], dest_rhs_packs: wp.array[wp.vec4], joints: wp.array[int], starts: wp.array[int], offsets: wp.array[int], columns: wp.array[int], local: wp.array[int], kinds: wp.array[int], enabled: wp.array[bool], parents: wp.array[int], children: wp.array[int], constraints: wp.array[int], rates: wp.array[float], edges: wp.array[wp.vec4i], body: wp.array[wp.transform], prev: wp.array[wp.transform], parent_frame: wp.array[wp.transform], child_frame: wp.array[wp.transform], rest_bend: wp.array[wp.vec3], rest_twist: wp.array[float], geometry: wp.array[wp.vec3], copy_threads: int, count: int):
    thread = wp.tid()
    if thread < copy_threads:
        base = thread // 32 * 64 + thread % 32
        value_count = value_packs.shape[0]
        for item in range(2):
            index = base + 32 * item
            if index < value_count:
                dest_packs[index] = value_packs[index]
            elif index < value_count + rhs_packs.shape[0]:
                dest_rhs_packs[index - value_count] = rhs_packs[index - value_count]
        if thread < values.shape[0] % 4:
            dest[value_count*4 + thread] = values[value_count*4 + thread]
        if thread < rhs.shape[0] % 4:
            dest_rhs[rhs_packs.shape[0]*4 + thread] = rhs[rhs_packs.shape[0]*4 + thread]
    else:
        aligned_count = ((count + 31) // 32) * 32
        task = (thread - copy_threads) // aligned_count
        cursor = (thread - copy_threads) % aligned_count
        if cursor >= count:
            return
        joint = joints[cursor]
        if task != 0:
            if not enabled[joint] or kinds[joint] != JointType.ROD or children[joint] < 0:
                return
            parent = parents[joint]
            child = children[joint]
            first = constraints[joint]
            if rates[first] != 0.0 or rates[first+1] != 0.0 or rates[first+2] != 0.0 or rates[first+3] != 0.0:
                return
            parent_pose = wp.transform(wp.vec3(0.0), wp.quat_identity())
            child_pose = body[child]
            if parent >= 0:
                parent_pose = body[parent]
            parent_rotation = wp.transform_get_rotation(parent_pose * parent_frame[joint])
            child_rotation = wp.transform_get_rotation(child_pose * child_frame[joint])
            measure = measure_rod_bend_twist_z(parent_rotation, child_rotation)
            strain = assemble_geometric_rod_kappa_z(parent_rotation, measure.kb_world, measure.twist, rest_bend[joint], rest_twist[joint])
            geometry[6 * count + cursor] = strain
            gradient_child = rod_bend_twist_jacobian_z_from_measure(measure, False)
            for row in range(3):
                geometry[row * count + cursor] = wp.vec3(gradient_child[row,0], gradient_child[row,1], gradient_child[row,2])
            if parent >= 0:
                gradient_parent = rod_bend_twist_jacobian_z_from_measure(measure, True)
                for row in range(3):
                    geometry[(3+row) * count + cursor] = wp.vec3(gradient_parent[row,0], gradient_parent[row,1], gradient_parent[row,2])
            return
        if enabled[joint] and kinds[joint] == JointType.ROD and children[joint] >= 0:
            child = local[children[joint]]
            base_body = starts[cursor]
            slot_child = _find_block_slot(offsets, columns, base_body, child, child)
            slot_parent = int(-1)
            slot_cross = int(-1)
            if parents[joint] >= 0:
                parent = local[parents[joint]]
                slot_parent = _find_block_slot(offsets, columns, base_body, parent, parent)
                slot_cross = _find_block_slot(offsets, columns, base_body, wp.max(parent,child), wp.min(parent,child))
            first = constraints[joint]
            no_rates = rates[first] == 0.0 and rates[first+1] == 0.0 and rates[first+2] == 0.0 and rates[first+3] == 0.0
            edges[cursor] = wp.vec4i(slot_child, slot_parent, slot_cross, int(no_rates))

@wp.func
def scatter_endpoint(values: wp.array[float], rhs: wp.array[float], row: int, slot: int, force: wp.vec3, torque: wp.vec3, ll: wp.mat33, al: wp.mat33, aa: wp.mat33, stripe: int, mask: int):
    _add_rhs_scalar_atomic(rhs, row, _vec6_from_parts(force, torque), stripe, mask)
    if slot >= 0:
        for axis in range(3):
            for comp in range(3):
                index = axis * 6 + comp
                origin = slot * 36 + index
                if (index & mask) == stripe and ll[axis,comp] != 0.0:
                    wp.atomic_add(values, origin, ll[axis,comp])
                if ((index + 3) & mask) == stripe and al[comp,axis] != 0.0:
                    wp.atomic_add(values, origin + 3, al[comp,axis])
                if ((index + 18) & mask) == stripe and al[axis,comp] != 0.0:
                    wp.atomic_add(values, origin + 18, al[axis,comp])
                if ((index + 21) & mask) == stripe and aa[axis,comp] != 0.0:
                    wp.atomic_add(values, origin + 21, aa[axis,comp])

@wp.func
def scatter_link(values: wp.array[float], slot: int, transpose: bool, cross: mat66f, stripe: int, mask: int):
    if slot >= 0:
        for axis in range(6):
            for comp in range(6):
                index = axis * 6 + comp
                if (index & mask) == stripe and cross[axis,comp] != 0.0:
                    addr = comp*6 + axis if transpose else index
                    wp.atomic_add(values, slot*36 + addr, cross[axis,comp])

@wp.kernel(enable_backward=False)
def sparse_scatter(dt: float, articulation_joints: wp.array[wp.int32], articulation_joint_body_start: wp.array[wp.int32], articulation_block_row_offsets: wp.array[wp.int32], articulation_block_cols: wp.array[wp.int32], body_articulation_local: wp.array[wp.int32], body_q: wp.array[wp.transform], body_q_prev: wp.array[wp.transform], body_q_rest: wp.array[wp.transform], body_com: wp.array[wp.vec3], joint_type: wp.array[int], joint_enabled: wp.array[bool], joint_parent: wp.array[int], joint_child: wp.array[int], joint_X_p: wp.array[wp.transform], joint_X_c: wp.array[wp.transform], joint_axis: wp.array[wp.vec3], joint_rod_rest_kb_local: wp.array[wp.vec3], joint_rod_rest_twist: wp.array[float], joint_qd_start: wp.array[int], joint_target_q_start: wp.array[int], joint_constraint_start: wp.array[int], joint_penalty_k: wp.array[float], joint_rho: wp.array[float], joint_material_k: wp.array[float], joint_penalty_kd: wp.array[float], joint_sigma_start: wp.array[wp.vec3], joint_C_fric: wp.array[wp.vec3], joint_dof_dim: wp.array2d[int], joint_coordinates: JointCoordinateData, joint_friction: wp.array[float], joint_damping: wp.array[float], joint_target_ke: wp.array[float], joint_target_kd: wp.array[float], joint_armature: wp.array[float], joint_armature_target: wp.array[float], joint_target_q: wp.array[float], joint_target_vel: wp.array[float], joint_limit_lower: wp.array[float], joint_limit_upper: wp.array[float], joint_limit_ke: wp.array[float], joint_limit_kd: wp.array[float], joint_drive_limit_support: wp.array[float], joint_drive_lambda: wp.array[float], joint_limit_lambda: wp.array[float], joint_lambda_lin: wp.array[wp.vec3], joint_lambda_ang: wp.array[wp.vec3], joint_C0_lin: wp.array[wp.vec3], joint_C0_ang: wp.array[wp.vec3], joint_is_hard: wp.array[wp.int32], avbd_alpha: float, joint_compliant_alm: int, elastic: SparseElasticJointData, values_scalar: wp.array[float], rhs_scalar: wp.array[float], lane_mask: int):
    joint_cursor, stripe = wp.tid()
    joint = articulation_joints[joint_cursor]
    if not joint_enabled[joint]:
        return
    jt = joint_type[joint]
    if not _sparse_joint_supported(jt):
        return
    child = joint_child[joint]
    parent = joint_parent[joint]
    if child < 0:
        return
    body_start = articulation_joint_body_start[joint_cursor]
    child_local = body_articulation_local[child]
    parent_local = -1
    if parent >= 0:
        parent_local = body_articulation_local[parent]
    X_p = _sparse_joint_frame(elastic, joint, parent, True, joint_X_p[joint], False)
    X_c = _sparse_joint_frame(elastic, joint, child, False, joint_X_c[joint], False)
    X_p_prev = _sparse_joint_frame(elastic, joint, parent, True, joint_X_p[joint], True)
    X_c_prev = _sparse_joint_frame(elastic, joint, child, False, joint_X_c[joint], True)
    cross_slot = int(-1)
    if elastic.enabled != 0:
        cross_slot = elastic.joint_cross_slot[joint]
    if jt == JointType.ROD:
        endpoint_force, torque, ll, al, aa = evaluate_rod_joint_force_hessian(child, joint, body_q, body_q_prev, body_com, joint_parent, joint_child, X_p, X_c, X_p_prev, X_c_prev, joint_rod_rest_kb_local, joint_rod_rest_twist, joint_constraint_start, joint_penalty_k, joint_rho, joint_material_k, joint_penalty_kd, joint_sigma_start, joint_C_fric, joint_lambda_lin, joint_lambda_ang, joint_C0_lin, joint_C0_ang, joint_is_hard, avbd_alpha, joint_compliant_alm, dt)
        _add_rod_body_contribution_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, child_local, endpoint_force, torque, ll, al, aa, stripe, lane_mask)
        if parent >= 0:
            endpoint_force, torque, ll, al, aa = evaluate_rod_joint_force_hessian(parent, joint, body_q, body_q_prev, body_com, joint_parent, joint_child, X_p, X_c, X_p_prev, X_c_prev, joint_rod_rest_kb_local, joint_rod_rest_twist, joint_constraint_start, joint_penalty_k, joint_rho, joint_material_k, joint_penalty_kd, joint_sigma_start, joint_C_fric, joint_lambda_lin, joint_lambda_ang, joint_C0_lin, joint_C0_ang, joint_is_hard, avbd_alpha, joint_compliant_alm, dt)
            _add_rod_body_contribution_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, endpoint_force, torque, ll, al, aa, stripe, lane_mask)
            cross = _rod_cross_hessian(dt, joint, parent, child, body_q, body_com, X_p, X_c, joint_constraint_start, joint_penalty_k, joint_rho, joint_material_k, joint_penalty_kd, joint_C_fric, joint_is_hard, joint_compliant_alm, stripe, lane_mask)
            _add_rod_cross_block_scalar(values_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, cross, cross_slot, stripe, lane_mask)
        return
    c_start = joint_constraint_start[joint]
    qd_start = joint_qd_start[joint]
    lin_count = int(jt == JointType.PRISMATIC)
    ang_count = int(jt == JointType.REVOLUTE)
    if jt == JointType.D6:
        lin_count = joint_dof_dim[joint, 0]
        ang_count = joint_dof_dim[joint, 1]
    g = _sparse_joint_geometry(joint, jt, parent, child, X_p, X_c, X_p_prev, X_c_prev, body_q, body_q_prev, body_q_rest, joint_X_p, joint_X_c, joint_axis, joint_qd_start, joint_constraint_start, joint_penalty_k, joint_rho, joint_material_k, joint_penalty_kd, joint_dof_dim, joint_lambda_lin, joint_lambda_ang, joint_C0_lin, joint_C0_ang, joint_is_hard, avbd_alpha, joint_compliant_alm)
    if g.k_linear > 0.0 and (jt != JointType.D6 or lin_count < 3):
        _assemble_linear_joint_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, parent, child, g.parent_anchor, g.child_anchor, g.parent_anchor_prev, g.child_anchor_prev, body_q, body_com, g.k_linear, g.kd_linear, dt, g.P_lin, g.lin_lambda, g.lin_C0, g.lin_alpha, cross_slot, stripe, lane_mask)
    if g.k_angular > 0.0 and jt != JointType.BALL and (jt != JointType.D6 or ang_count < 3):
        sigma0 = wp.vec3(0.0)
        C_fric = wp.vec3(0.0)
        _kappa, _J_world = _assemble_angular_joint_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, parent, g.parent_anchor_q, g.child_anchor_q, g.parent_anchor_q_prev, g.child_anchor_q_prev, g.parent_rest_q, g.child_rest_q, g.k_angular, g.kd_angular, dt, g.P_ang, sigma0, C_fric, g.ang_lambda, g.ang_C0, g.ang_alpha, cross_slot, joint_axis, qd_start + lin_count, ang_count, stripe, lane_mask)
    if jt == JointType.REVOLUTE or jt == JointType.PRISMATIC or jt == JointType.D6:
        for component in range(lin_count + ang_count):
            active, gp, gc, force, hessian = _sparse_joint_friction_row(joint_coordinates, joint, component, lin_count, qd_start + component, body_q, body_q_prev, joint_friction, joint_damping, dt)
            if active:
                _assemble_coordinate_row_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, gp, gc, force, hessian, cross_slot, stripe, lane_mask)
    if jt == JointType.REVOLUTE:
        active, gp, gc, force, hessian = _sparse_joint_armature_row(joint_coordinates, joint, qd_start, body_q, joint_armature, joint_armature_target, dt)
        if active:
            _assemble_coordinate_row_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, gp, gc, force, hessian, cross_slot, stripe, lane_mask)
    if jt == JointType.REVOLUTE:
        force_scalar, hessian_scalar, angular_jacobian_world = _sparse_angular_drive_limit_row(joint_coordinates, joint, 0, qd_start, joint_target_q_start[joint], c_start + 2, body_q, body_q_prev, joint_target_ke, joint_target_kd, joint_target_q, joint_target_vel, joint_limit_lower, joint_limit_upper, joint_limit_ke, joint_limit_kd, joint_penalty_k, joint_drive_limit_support, joint_drive_lambda, joint_limit_lambda, joint_compliant_alm, dt)
        if hessian_scalar > 0.0:
            _assemble_angular_axis_row_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, parent, angular_jacobian_world, force_scalar, hessian_scalar, cross_slot, stripe, lane_mask)
    if jt == JointType.PRISMATIC or jt == JointType.D6:
        C_vec = g.child_anchor - g.parent_anchor
        C_vec_prev = g.child_anchor_prev - g.parent_anchor_prev
        target_q_base = joint_target_q_start[joint]
        for li in range(3):
            if li < lin_count:
                force_scalar, hessian_scalar, axis_world = _sparse_linear_drive_limit_row(qd_start + li, target_q_base + li, c_start + 2 + li, g.parent_anchor_q, C_vec, C_vec_prev, joint_axis, joint_target_ke, joint_target_kd, joint_target_q, joint_target_vel, joint_limit_lower, joint_limit_upper, joint_limit_ke, joint_limit_kd, joint_penalty_k, joint_drive_limit_support, joint_drive_lambda, joint_limit_lambda, joint_compliant_alm, dt)
                if hessian_scalar > 0.0:
                    _assemble_linear_axis_row_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, parent, child, g.parent_anchor, g.child_anchor, body_q, body_com, axis_world, force_scalar, hessian_scalar, cross_slot, stripe, lane_mask)
    if jt == JointType.D6:
        target_q_base = joint_target_q_start[joint]
        for ai in range(3):
            if ai < ang_count:
                force_scalar, hessian_scalar, angular_jacobian_world = _sparse_angular_drive_limit_row(joint_coordinates, joint, lin_count + ai, qd_start + lin_count + ai, target_q_base + lin_count + ai, c_start + 2 + lin_count + ai, body_q, body_q_prev, joint_target_ke, joint_target_kd, joint_target_q, joint_target_vel, joint_limit_lower, joint_limit_upper, joint_limit_ke, joint_limit_kd, joint_penalty_k, joint_drive_limit_support, joint_drive_lambda, joint_limit_lambda, joint_compliant_alm, dt)
                if hessian_scalar > 0.0:
                    _assemble_angular_axis_row_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, parent, angular_jacobian_world, force_scalar, hessian_scalar, cross_slot, stripe, lane_mask)


@wp.struct
class CompactCoord:
    body_com: wp.array[wp.vec3]
    joint_type: wp.array[int]
    parent: wp.array[int]
    child: wp.array[int]
    X_p: wp.array[wp.transform]
    X_c: wp.array[wp.transform]
    q_start: wp.array[int]
    qd_start: wp.array[int]
    dof_dim: wp.array2d[int]
    axis: wp.array[wp.vec3]
    history: wp.array[float]

@wp.func
def evaluate_coordinate_no_modal(data: CompactCoord, joint: int, component: int, body_q: wp.array[wp.transform]):
    """Evaluate a coordinate on the current turn, together with its pose gradients.

    Reduced elastic endpoints are deformed with the modal coordinates in ``elastic_q``, which must
    pair with ``body_q``; the coordinate then includes endpoint rotation ``psi * q`` and the gradients
    use the deformed anchors.
    """
    angular_reference = wp.vec3()
    angular_start = data.q_start[joint] + data.dof_dim[joint, 0]
    for axis in range(data.dof_dim[joint, 1]):
        angular_reference[axis] = data.history[angular_start + axis]
    X_pj = data.X_p[joint]
    X_cj = data.X_c[joint]
    q, g_p, g_c = eval_joint_coordinate_in_frames(joint, component, body_q, data.body_com, data.joint_type, data.parent, data.child, X_pj, X_cj, data.qd_start, data.dof_dim, data.axis, angular_reference, True)
    q = unwrap_joint_coordinate(q, data.history[data.q_start[joint] + component], component, data.dof_dim[joint, 0])
    return (q, g_p, g_c)

@wp.func
def evaluate_coordinate_prev_no_modal(data: CompactCoord, joint: int, component: int, body_q_prev: wp.array[wp.transform]):
    """Evaluate a coordinate at previous-step poses with the previous modal coordinates."""
    return evaluate_coordinate_no_modal(data, joint, component, body_q_prev)

@wp.func
def _evaluate_angular_coordinate_no_modal(data: CompactCoord, joint: int, component: int, body_q: wp.array[wp.transform], body_q_prev: wp.array[wp.transform], dt: float):
    """Measure the continuous angle, step velocity, and world-space angular gradient."""
    q, _g_p, g_c = evaluate_coordinate_no_modal(data, joint, component, body_q)
    q_prev, _g_p_prev, _g_c_prev = evaluate_coordinate_prev_no_modal(data, joint, component, body_q_prev)
    delta = q - q_prev
    if wp.abs(delta) >= wp.pi:
        delta = wp.atan2(wp.sin(delta), wp.cos(delta))
    rate = delta / dt
    return (q, rate, wp.spatial_bottom(g_c))

@wp.func
def _sparse_angular_drive_limit_row_no_modal(joint_coordinates: CompactCoord, joint: int, coordinate: int, dof_idx: int, target_q_idx: int, penalty_slot: int, body_q: wp.array[wp.transform], body_q_prev: wp.array[wp.transform], joint_target_ke: wp.array[float], joint_target_kd: wp.array[float], joint_target_q: wp.array[float], joint_target_vel: wp.array[float], joint_limit_lower: wp.array[float], joint_limit_upper: wp.array[float], joint_limit_ke: wp.array[float], joint_limit_kd: wp.array[float], joint_penalty_k: wp.array[float], joint_drive_limit_support: wp.array[float], joint_drive_lambda: wp.array[float], joint_limit_lambda: wp.array[float], joint_compliant_alm: int, dt: float):
    """Return ``(force, hessian, angular_jacobian_world)`` of an angular DOF's drive and limit row.

    ``hessian`` is zero when the DOF has neither a drive nor a limit row.
    """
    has_drive = _drive_row_applies_force(joint_target_ke[dof_idx], joint_target_kd[dof_idx])
    has_limits = _limit_row_exists(joint_limit_ke[dof_idx], joint_limit_lower[dof_idx], joint_limit_upper[dof_idx])
    if has_drive or has_limits:
        theta_abs, dtheta_dt, angular_jacobian_world = _evaluate_angular_coordinate_no_modal(joint_coordinates, joint, coordinate, body_q, body_q_prev, dt)
        force_scalar, hessian_scalar = _evaluate_sparse_drive_limit(dof_idx, target_q_idx, penalty_slot, theta_abs, dtheta_dt, joint_target_ke, joint_target_kd, joint_target_q, joint_target_vel, joint_limit_lower, joint_limit_upper, joint_limit_ke, joint_limit_kd, joint_penalty_k, joint_drive_limit_support, joint_drive_lambda, joint_limit_lambda, joint_compliant_alm, dt)
        return (force_scalar, hessian_scalar, angular_jacobian_world)
    return (float(0.0), float(0.0), wp.vec3(0.0))

@wp.func
def _sparse_joint_armature_row_no_modal(joint_coordinates: CompactCoord, joint: int, qd_start: int, body_q: wp.array[wp.transform], joint_armature: wp.array[float], joint_armature_target: wp.array[float], dt: float):
    """Return ``(active, g_parent, g_child, force, hessian)`` of a revolute joint's armature row."""
    armature = joint_armature[qd_start]
    if armature > 0.0:
        q, gp, gc = evaluate_coordinate_no_modal(joint_coordinates, joint, 0, body_q)
        armature_hessian = armature / (dt * dt)
        armature_force = armature_hessian * (q - joint_armature_target[qd_start])
        return (True, gp, gc, armature_force, armature_hessian)
    return (False, wp.spatial_vector(), wp.spatial_vector(), float(0.0), float(0.0))

@wp.func
def _sparse_joint_friction_row_no_modal(joint_coordinates: CompactCoord, joint: int, component: int, lin_count: int, dof: int, body_q: wp.array[wp.transform], body_q_prev: wp.array[wp.transform], joint_friction: wp.array[float], joint_damping: wp.array[float], dt: float):
    """Return ``(active, g_parent, g_child, force, hessian)`` of a free DOF's friction and damping row."""
    friction = joint_friction[dof]
    damping = wp.max(joint_damping[dof], 0.0)
    if friction > 0.0 or damping > 0.0:
        q, gp, gc = evaluate_coordinate_no_modal(joint_coordinates, joint, component, body_q)
        q_prev, _gp_prev, _gc_prev = evaluate_coordinate_prev_no_modal(joint_coordinates, joint, component, body_q_prev)
        delta = q - q_prev
        if component >= lin_count:
            delta = wp.atan2(wp.sin(delta), wp.cos(delta))
        rate = delta / dt
        force, hessian = _eval_joint_axis_friction(rate, friction, 1.0 / dt)
        return (True, gp, gc, force + damping * rate, hessian + damping / dt)
    return (False, wp.spatial_vector(), wp.spatial_vector(), float(0.0), float(0.0))

@wp.func
def load_gradient(geometry: wp.array[wp.vec3], cursor: int, count: int, parent: bool):
    start = cursor + 3*count if parent else cursor
    return wp.matrix_from_rows(geometry[start], geometry[start+count], geometry[start+2*count])

@wp.func
def shared_bending(q_wp: wp.quat, q_wc: wp.quat, kb_rest_local: wp.vec3, twist_rest: float, q_wp_prev: wp.quat, q_wc_prev: wp.quat, is_parent: bool, K_elastic_diag: wp.vec3, C0_force: wp.vec3, sigma0: wp.vec3, H_fric_diag: wp.vec3, lambda_projected: wp.vec3, K_damp_diag: wp.vec3, damping_active: bool, dt: float, has_parent: bool, geometry: wp.array[wp.vec3], cursor: int, count: int):
    """Bend/twist torque and Hessian for SolverVBD local +Z rod joints.

    In the fixed rod material basis, local angular operators are diagonal:
    ``[bend_x, bend_y, twist_z]``. Keep them as vec3 row scales in the hot path
    instead of building dense local matrices.
    """
    inv_dt = 1.0 / dt
    kappa_now_vec = geometry[6*count + cursor]
    f_local = wp.cw_mul(K_elastic_diag, kappa_now_vec) - C0_force + sigma0 + lambda_projected
    H_local_diag = K_elastic_diag + H_fric_diag
    if damping_active:
        prev_measure = measure_rod_bend_twist_z(q_wp_prev, q_wc_prev)
        kappa_prev_vec = assemble_geometric_rod_kappa_z(q_wp_prev, prev_measure.kb_world, prev_measure.twist, kb_rest_local, twist_rest)
        dkappa_dt = rod_bend_twist_delta(kappa_now_vec, kappa_prev_vec) * inv_dt
        f_local = f_local + wp.cw_mul(K_damp_diag, dkappa_dt)
        H_local_diag = H_local_diag + inv_dt * K_damp_diag
    J_body = load_gradient(geometry, cursor, count, is_parent)
    H_aa = wp.transpose(J_body) * _diag_mul_mat33(H_local_diag, J_body)
    tau_world = -(wp.transpose(J_body) * f_local)
    parent_tau = wp.vec3(0.0)
    parent_aa = wp.mat33(0.0)
    if has_parent:
        parent_jac = load_gradient(geometry, cursor, count, True)
        parent_aa = wp.transpose(parent_jac) * _diag_mul_mat33(H_local_diag, parent_jac)
        parent_tau = -(wp.transpose(parent_jac) * f_local)
    return (tau_world, H_aa, parent_tau, parent_aa)

@wp.func
def paired_bending(q_wp: wp.quat, q_wc: wp.quat, kb_rest_local: wp.vec3, twist_rest: float, q_wp_prev: wp.quat, q_wc_prev: wp.quat, is_parent: bool, K_elastic_diag: wp.vec3, C0_force: wp.vec3, sigma0: wp.vec3, H_fric_diag: wp.vec3, lambda_projected: wp.vec3, K_damp_diag: wp.vec3, damping_active: bool, dt: float, has_parent: bool):
    """Bend/twist torque and Hessian for SolverVBD local +Z rod joints.

    In the fixed rod material basis, local angular operators are diagonal:
    ``[bend_x, bend_y, twist_z]``. Keep them as vec3 row scales in the hot path
    instead of building dense local matrices.
    """
    inv_dt = 1.0 / dt
    measure = measure_rod_bend_twist_z(q_wp, q_wc)
    kappa_now_vec = assemble_geometric_rod_kappa_z(q_wp, measure.kb_world, measure.twist, kb_rest_local, twist_rest)
    f_local = wp.cw_mul(K_elastic_diag, kappa_now_vec) - C0_force + sigma0 + lambda_projected
    H_local_diag = K_elastic_diag + H_fric_diag
    if damping_active:
        prev_measure = measure_rod_bend_twist_z(q_wp_prev, q_wc_prev)
        kappa_prev_vec = assemble_geometric_rod_kappa_z(q_wp_prev, prev_measure.kb_world, prev_measure.twist, kb_rest_local, twist_rest)
        dkappa_dt = rod_bend_twist_delta(kappa_now_vec, kappa_prev_vec) * inv_dt
        f_local = f_local + wp.cw_mul(K_damp_diag, dkappa_dt)
        H_local_diag = H_local_diag + inv_dt * K_damp_diag
    J_body = rod_bend_twist_jacobian_z_from_measure(measure, is_parent)
    H_aa = wp.transpose(J_body) * _diag_mul_mat33(H_local_diag, J_body)
    tau_world = -(wp.transpose(J_body) * f_local)
    parent_tau = wp.vec3(0.0)
    parent_aa = wp.mat33(0.0)
    if has_parent:
        parent_jac = rod_bend_twist_jacobian_z_from_measure(measure, True)
        parent_aa = wp.transpose(parent_jac) * _diag_mul_mat33(H_local_diag, parent_jac)
        parent_tau = -(wp.transpose(parent_jac) * f_local)
    return (tau_world, H_aa, parent_tau, parent_aa)

@wp.func
def paired_rod_response(joint_index: int, body_q: wp.array[wp.transform], body_q_prev: wp.array[wp.transform], body_com: wp.array[wp.vec3], joint_parent: wp.array[int], joint_child: wp.array[int], X_pj: wp.transform, X_cj: wp.transform, X_pj_prev: wp.transform, X_cj_prev: wp.transform, joint_rod_rest_kb_local: wp.array[wp.vec3], joint_rod_rest_twist: wp.array[float], joint_constraint_start: wp.array[int], joint_penalty_k: wp.array[float], joint_rho: wp.array[float], joint_material_k: wp.array[float], joint_penalty_kd: wp.array[float], joint_sigma_start: wp.array[wp.vec3], joint_C_fric: wp.array[wp.vec3], joint_lambda_lin: wp.array[wp.vec3], joint_lambda_ang: wp.array[wp.vec3], joint_C0_lin: wp.array[wp.vec3], joint_C0_ang: wp.array[wp.vec3], joint_is_hard: wp.array[wp.int32], stab_alpha: float, joint_compliant_alm: int, dt: float):
    """Compute exact split ROD force and self-Hessian terms for one body.

    ``X_pj``/``X_cj`` are the body-local parent/child attachment frames at the current
    iterate and ``X_pj_prev``/``X_cj_prev`` those at the previous step. Rigid callers
    pass ``joint_X_p[joint]``/``joint_X_c[joint]`` for both; reduced elastic endpoints
    pass their deformed frames so the rod measures strain at the deformed attachment.
    """
    parent_index = joint_parent[joint_index]
    child_index = joint_child[joint_index]
    is_parent_body = parent_index >= 0 and child_index == parent_index
    if parent_index >= 0:
        parent_pose = body_q[parent_index]
        parent_pose_prev = body_q_prev[parent_index]
        parent_com = body_com[parent_index]
    else:
        parent_pose = wp.transform(wp.vec3(0.0), wp.quat_identity())
        parent_pose_prev = parent_pose
        parent_com = wp.vec3(0.0)
    child_pose = body_q[child_index]
    child_pose_prev = body_q_prev[child_index]
    child_com = body_com[child_index]
    X_wp = parent_pose * X_pj
    X_wc = child_pose * X_cj
    X_wp_prev = parent_pose_prev * X_pj_prev
    X_wc_prev = child_pose_prev * X_cj_prev
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    q_wp_prev = wp.transform_get_rotation(X_wp_prev)
    q_wc_prev = wp.transform_get_rotation(X_wc_prev)
    c_start = joint_constraint_start[joint_index]
    stretch_idx = c_start
    shear_idx = c_start + 1
    bend_idx = c_start + 2
    twist_idx = c_start + 3
    solve_weight_stretch = _load_solve_weight(joint_penalty_k, joint_rho, stretch_idx, joint_compliant_alm)
    solve_weight_shear = _load_solve_weight(joint_penalty_k, joint_rho, shear_idx, joint_compliant_alm)
    solve_weight_bend = _load_solve_weight(joint_penalty_k, joint_rho, bend_idx, joint_compliant_alm)
    solve_weight_twist = _load_solve_weight(joint_penalty_k, joint_rho, twist_idx, joint_compliant_alm)
    material_stretch = joint_material_k[stretch_idx]
    material_shear = joint_material_k[shear_idx]
    material_bend = joint_material_k[bend_idx]
    material_twist = joint_material_k[twist_idx]
    kd_stretch = joint_penalty_kd[stretch_idx]
    kd_shear = joint_penalty_kd[shear_idx]
    kd_bend = joint_penalty_kd[bend_idx]
    kd_twist = joint_penalty_kd[twist_idx]
    total_force = wp.vec3(0.0)
    total_force_p = wp.vec3(0.0)
    total_torque = wp.vec3(0.0)
    total_torque_p = wp.vec3(0.0)
    total_H_ll = wp.mat33(0.0)
    total_H_ll_p = wp.mat33(0.0)
    total_H_al = wp.mat33(0.0)
    total_H_al_p = wp.mat33(0.0)
    total_H_aa = wp.mat33(0.0)
    total_H_aa_p = wp.mat33(0.0)
    bend_stiff = _structural_row_has_stiffness(solve_weight_bend, material_bend, joint_compliant_alm)
    twist_stiff = _structural_row_has_stiffness(solve_weight_twist, material_twist, joint_compliant_alm)
    bend_active = bend_stiff or kd_bend > 0.0
    twist_active = twist_stiff or kd_twist > 0.0
    if bend_active or twist_active:
        lambda_ang = joint_lambda_ang[joint_index]
        C0_ang = joint_C0_ang[joint_index]
        bend_hard = bend_stiff and joint_compliant_alm == 0 and (joint_is_hard[bend_idx] == 1)
        twist_hard = twist_stiff and joint_compliant_alm == 0 and (joint_is_hard[twist_idx] == 1)
        bend_dual = wp.vec3(lambda_ang[0], lambda_ang[1], 0.0)
        twist_dual = wp.vec3(0.0, 0.0, lambda_ang[2])
        if joint_compliant_alm == 0 and (not bend_hard):
            bend_dual = wp.vec3(0.0)
        if joint_compliant_alm == 0 and (not twist_hard):
            twist_dual = wp.vec3(0.0)
        bend_primal_k, bend_dual_eff = _material_force_terms(solve_weight_bend, material_bend, bend_dual, joint_compliant_alm)
        twist_primal_k, twist_dual_eff = _material_force_terms(solve_weight_twist, material_twist, twist_dual, joint_compliant_alm)
        K_elastic_diag = wp.vec3(bend_primal_k, bend_primal_k, twist_primal_k)
        K_damp_diag = wp.vec3(kd_bend, kd_bend, kd_twist)
        bend_alpha = stab_alpha if joint_compliant_alm == 1 or bend_hard else 0.0
        twist_alpha = stab_alpha if joint_compliant_alm == 1 or twist_hard else 0.0
        C0_force = bend_primal_k * bend_alpha * wp.vec3(C0_ang[0], C0_ang[1], 0.0)
        C0_force = C0_force + twist_primal_k * twist_alpha * wp.vec3(0.0, 0.0, C0_ang[2])
        sigma = wp.vec3(0.0)
        H_fric_diag = wp.vec3(0.0)
        dahl_sigma = joint_sigma_start[joint_index]
        dahl_fric = joint_C_fric[joint_index]
        if bend_stiff and (not bend_hard):
            sigma = sigma + wp.vec3(dahl_sigma[0], dahl_sigma[1], 0.0)
            H_fric_diag = H_fric_diag + wp.vec3(dahl_fric[0], dahl_fric[1], 0.0)
        if twist_stiff and (not twist_hard):
            sigma = sigma + wp.vec3(0.0, 0.0, dahl_sigma[2])
            H_fric_diag = H_fric_diag + wp.vec3(0.0, 0.0, dahl_fric[2])
        rod_torque, rod_H_aa, torque_parent, haa_parent = paired_bending(q_wp, q_wc, joint_rod_rest_kb_local[joint_index], joint_rod_rest_twist[joint_index], q_wp_prev, q_wc_prev, is_parent_body, K_elastic_diag, C0_force, sigma, H_fric_diag, bend_dual_eff + twist_dual_eff, K_damp_diag, kd_bend > 0.0 or kd_twist > 0.0, dt, parent_index >= 0)
        total_torque = total_torque + rod_torque
        total_H_aa = total_H_aa + rod_H_aa
        total_torque_p = total_torque_p + torque_parent
        total_H_aa_p = total_H_aa_p + haa_parent
    stretch_stiff = _structural_row_has_stiffness(solve_weight_stretch, material_stretch, joint_compliant_alm)
    shear_stiff = _structural_row_has_stiffness(solve_weight_shear, material_shear, joint_compliant_alm)
    stretch_active = stretch_stiff or kd_stretch > 0.0
    shear_active = shear_stiff or kd_shear > 0.0
    if stretch_active or shear_active:
        lambda_lin = joint_lambda_lin[joint_index]
        C0_lin = joint_C0_lin[joint_index]
        stretch_hard = stretch_stiff and joint_compliant_alm == 0 and (joint_is_hard[stretch_idx] == 1)
        shear_hard = shear_stiff and joint_compliant_alm == 0 and (joint_is_hard[shear_idx] == 1)
        stretch_dual = wp.vec3(0.0, 0.0, lambda_lin[2])
        shear_dual = wp.vec3(lambda_lin[0], lambda_lin[1], 0.0)
        if joint_compliant_alm == 0 and (not stretch_hard):
            stretch_dual = wp.vec3(0.0)
        if joint_compliant_alm == 0 and (not shear_hard):
            shear_dual = wp.vec3(0.0)
        stretch_primal_k, stretch_dual_eff = _material_force_terms(solve_weight_stretch, material_stretch, stretch_dual, joint_compliant_alm)
        shear_primal_k, shear_dual_eff = _material_force_terms(solve_weight_shear, material_shear, shear_dual, joint_compliant_alm)
        k_diag = wp.vec3(shear_primal_k, shear_primal_k, stretch_primal_k)
        kd_diag = wp.vec3(kd_shear, kd_shear, kd_stretch)
        stretch_alpha = stab_alpha if joint_compliant_alm == 1 or stretch_hard else 0.0
        shear_alpha = stab_alpha if joint_compliant_alm == 1 or shear_hard else 0.0
        C0_force_local = shear_primal_k * shear_alpha * wp.vec3(C0_lin[0], C0_lin[1], 0.0)
        C0_force_local = C0_force_local + stretch_primal_k * stretch_alpha * wp.vec3(0.0, 0.0, C0_lin[2])
        f_l, t_l, Hll_l, Hal_l, Haa_l = evaluate_rod_stretch_shear_force_hessian(X_wp, X_wc, X_wp_prev, X_wc_prev, parent_pose, child_pose, parent_com, child_com, is_parent_body, k_diag, C0_force_local, stretch_dual_eff + shear_dual_eff, kd_diag, kd_stretch > 0.0 or kd_shear > 0.0, dt)
        total_force = total_force + f_l
        total_torque = total_torque + t_l
        total_H_ll = total_H_ll + Hll_l
        total_H_al = total_H_al + Hal_l
        total_H_aa = total_H_aa + Haa_l
        if parent_index >= 0:
            f_l_p, t_l_p, Hll_l_p, Hal_l_p, Haa_l_p = evaluate_rod_stretch_shear_force_hessian(X_wp, X_wc, X_wp_prev, X_wc_prev, parent_pose, child_pose, parent_com, child_com, True, k_diag, C0_force_local, stretch_dual_eff + shear_dual_eff, kd_diag, kd_stretch > 0.0 or kd_shear > 0.0, dt)
            total_force_p = total_force_p + f_l_p
            total_torque_p = total_torque_p + t_l_p
            total_H_ll_p = total_H_ll_p + Hll_l_p
            total_H_al_p = total_H_al_p + Hal_l_p
            total_H_aa_p = total_H_aa_p + Haa_l_p
    return (total_force, total_torque, total_H_ll, total_H_al, total_H_aa, total_force_p, total_torque_p, total_H_ll_p, total_H_al_p, total_H_aa_p)

@wp.func
def _rod_rate_free_arm_blocks(arm: wp.vec3, anchor: wp.vec3, world_k: wp.mat33, iso: float, isotropic: bool, parent_side: bool):
    """Angular-linear and angular-angular tiles. For S=skew(arm), S.T*K = -(S*K)."""
    skew_arm = wp.skew(arm)
    if parent_side:
        identity = wp.identity(3, float)
        anchor_al = iso * wp.skew(anchor)
        anchor_aa = iso * (wp.length_sq(anchor) * identity - wp.outer(anchor, anchor))
        if isotropic:
            return anchor_al, anchor_aa
        material_al = skew_arm * (world_k - iso * identity)
        return anchor_al + material_al, anchor_aa + (-material_al) * skew_arm
    if isotropic:
        child_al = iso * skew_arm
    else:
        child_al = skew_arm * world_k
    return child_al, (-child_al) * skew_arm

@wp.func
def isotropic_cross_entry(row: int, col: int, iso: float, child_arm: wp.vec3, parent_anchor: wp.vec3) -> float:
    if row < 3 and col < 3:
        return -iso if row == col else 0.0
    child_skew = wp.skew(child_arm)
    if row < 3:
        return iso * child_skew[row, col - 3]
    parent_skew = wp.skew(parent_anchor)
    if col < 3:
        return iso * parent_skew[col, row - 3]
    result = float(0.0)
    for axis in range(3):
        result = result + (iso * parent_skew[axis, row - 3]) * -child_skew[axis, col - 3]
    return result

@wp.func
def rate_free_cross(dt: float, joint: int, parent: int, child: int, body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], X_pj: wp.transform, X_cj: wp.transform, joint_constraint_start: wp.array[int], joint_penalty_k: wp.array[float], joint_rho: wp.array[float], joint_material_k: wp.array[float], joint_penalty_kd: wp.array[float], joint_C_fric: wp.array[wp.vec3], joint_is_hard: wp.array[wp.int32], joint_compliant_alm: int, stripe: int, lane_mask: int, geometry: wp.array[wp.vec3], cursor: int, count: int, values: wp.array[float], slot: int, flip: bool):
    """Return the parent-child Gauss-Newton block for one ROD joint."""
    if slot < 0:
        return
    aa = wp.mat33(0.0)
    c_start = joint_constraint_start[joint]
    stretch_idx = c_start
    shear_idx = c_start + 1
    bend_idx = c_start + 2
    twist_idx = c_start + 3
    stretch_weight = _load_solve_weight(joint_penalty_k, joint_rho, stretch_idx, joint_compliant_alm)
    shear_weight = _load_solve_weight(joint_penalty_k, joint_rho, shear_idx, joint_compliant_alm)
    bend_weight = _load_solve_weight(joint_penalty_k, joint_rho, bend_idx, joint_compliant_alm)
    twist_weight = _load_solve_weight(joint_penalty_k, joint_rho, twist_idx, joint_compliant_alm)
    k_stretch, _ = _material_force_terms(stretch_weight, joint_material_k[stretch_idx], wp.vec3(0.0), joint_compliant_alm)
    k_shear, _ = _material_force_terms(shear_weight, joint_material_k[shear_idx], wp.vec3(0.0), joint_compliant_alm)
    k_bend, _ = _material_force_terms(bend_weight, joint_material_k[bend_idx], wp.vec3(0.0), joint_compliant_alm)
    k_twist, _ = _material_force_terms(twist_weight, joint_material_k[twist_idx], wp.vec3(0.0), joint_compliant_alm)
    X_wp = body_q[parent] * X_pj
    X_wc = body_q[child] * X_cj
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    h_shear = k_shear
    h_stretch = k_stretch
    if h_shear > 0.0 or h_stretch > 0.0:
        tangent = wp.quat_rotate(q_wp, wp.vec3(0.0, 0.0, 1.0))
        K_world = h_shear * wp.identity(3, float) + (h_stretch - h_shear) * wp.outer(tangent, tangent)
        k_iso = wp.min(k_shear, k_stretch)
        K_material = K_world - k_iso * wp.identity(3, float)
        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        parent_com_world = wp.transform_point(body_q[parent], body_com[parent])
        child_com_world = wp.transform_point(body_q[child], body_com[child])
        r_parent_anchor = x_p - parent_com_world
        r_parent_material = x_c - parent_com_world
        r_child = x_c - child_com_world
        child_skew = wp.skew(r_child)
        anchor_skew = wp.skew(r_parent_anchor)
        upper_right = k_iso * child_skew
        lower_left = k_iso * wp.transpose(anchor_skew)
        if h_shear != h_stretch:
            upper_right = K_world * child_skew
            lower_left = lower_left + wp.transpose(wp.skew(r_parent_material)) * K_material
        aa = -(lower_left * child_skew)
        _rigid_add_tile(values, slot, 0, 0, -K_world, flip, stripe, lane_mask)
        _rigid_add_tile(values, slot, 0, 3, upper_right, flip, stripe, lane_mask)
        _rigid_add_tile(values, slot, 3, 0, lower_left, flip, stripe, lane_mask)
    bend_stiff = _structural_row_has_stiffness(bend_weight, joint_material_k[bend_idx], joint_compliant_alm)
    twist_stiff = _structural_row_has_stiffness(twist_weight, joint_material_k[twist_idx], joint_compliant_alm)
    bend_hard = bend_stiff and joint_compliant_alm == 0 and (joint_is_hard[bend_idx] == 1)
    twist_hard = twist_stiff and joint_compliant_alm == 0 and (joint_is_hard[twist_idx] == 1)
    friction = joint_C_fric[joint]
    h_bend_x = k_bend
    h_bend_y = h_bend_x
    h_twist = k_twist
    if bend_stiff and (not bend_hard):
        h_bend_x = h_bend_x + friction[0]
        h_bend_y = h_bend_y + friction[1]
    if twist_stiff and (not twist_hard):
        h_twist = h_twist + friction[2]
    if h_bend_x > 0.0 or h_bend_y > 0.0 or h_twist > 0.0:
        J_parent_angular = load_gradient(geometry, cursor, count, True)
        J_child_angular = load_gradient(geometry, cursor, count, False)
        H_local_diag = wp.vec3(h_bend_x, h_bend_y, h_twist)
        aa = aa + wp.transpose(_diag_mul_mat33(H_local_diag, J_parent_angular)) * J_child_angular
    _rigid_add_tile(values, slot, 3, 3, aa, flip, stripe, lane_mask)

@wp.func
def rate_free_response(joint_index: int, body_q: wp.array[wp.transform], body_q_prev: wp.array[wp.transform], body_com: wp.array[wp.vec3], joint_parent: wp.array[int], joint_child: wp.array[int], X_pj: wp.transform, X_cj: wp.transform, X_pj_prev: wp.transform, X_cj_prev: wp.transform, joint_rod_rest_kb_local: wp.array[wp.vec3], joint_rod_rest_twist: wp.array[float], joint_constraint_start: wp.array[int], joint_penalty_k: wp.array[float], joint_rho: wp.array[float], joint_material_k: wp.array[float], joint_penalty_kd: wp.array[float], joint_sigma_start: wp.array[wp.vec3], joint_C_fric: wp.array[wp.vec3], joint_lambda_lin: wp.array[wp.vec3], joint_lambda_ang: wp.array[wp.vec3], joint_C0_lin: wp.array[wp.vec3], joint_C0_ang: wp.array[wp.vec3], joint_is_hard: wp.array[wp.int32], stab_alpha: float, joint_compliant_alm: int, dt: float, geometry: wp.array[wp.vec3], cursor: int, count: int):
    """Paired undamped rod response. Previous poses and rates are not read."""
    parent_index = joint_parent[joint_index]
    child_index = joint_child[joint_index]
    is_parent_body = parent_index >= 0 and child_index == parent_index
    if parent_index >= 0:
        parent_pose = body_q[parent_index]
        parent_com = body_com[parent_index]
    else:
        parent_pose = wp.transform(wp.vec3(0.0), wp.quat_identity())
        parent_com = wp.vec3(0.0)
    child_pose = body_q[child_index]
    child_com = body_com[child_index]
    X_wp = parent_pose * X_pj
    X_wc = child_pose * X_cj
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    c_start = joint_constraint_start[joint_index]
    stretch_idx = c_start
    shear_idx = c_start + 1
    bend_idx = c_start + 2
    twist_idx = c_start + 3
    solve_weight_stretch = _load_solve_weight(joint_penalty_k, joint_rho, stretch_idx, joint_compliant_alm)
    solve_weight_shear = _load_solve_weight(joint_penalty_k, joint_rho, shear_idx, joint_compliant_alm)
    solve_weight_bend = _load_solve_weight(joint_penalty_k, joint_rho, bend_idx, joint_compliant_alm)
    solve_weight_twist = _load_solve_weight(joint_penalty_k, joint_rho, twist_idx, joint_compliant_alm)
    material_stretch = joint_material_k[stretch_idx]
    material_shear = joint_material_k[shear_idx]
    material_bend = joint_material_k[bend_idx]
    material_twist = joint_material_k[twist_idx]
    total_force = wp.vec3(0.0)
    total_force_p = wp.vec3(0.0)
    total_torque = wp.vec3(0.0)
    total_torque_p = wp.vec3(0.0)
    total_H_ll = wp.mat33(0.0)
    total_H_ll_p = wp.mat33(0.0)
    total_H_al = wp.mat33(0.0)
    total_H_al_p = wp.mat33(0.0)
    total_H_aa = wp.mat33(0.0)
    total_H_aa_p = wp.mat33(0.0)
    bend_stiff = _structural_row_has_stiffness(solve_weight_bend, material_bend, joint_compliant_alm)
    twist_stiff = _structural_row_has_stiffness(solve_weight_twist, material_twist, joint_compliant_alm)
    if bend_stiff or twist_stiff:
        lambda_ang = joint_lambda_ang[joint_index]
        C0_ang = joint_C0_ang[joint_index]
        bend_hard = bend_stiff and joint_compliant_alm == 0 and (joint_is_hard[bend_idx] == 1)
        twist_hard = twist_stiff and joint_compliant_alm == 0 and (joint_is_hard[twist_idx] == 1)
        bend_dual = wp.vec3(lambda_ang[0], lambda_ang[1], 0.0)
        twist_dual = wp.vec3(0.0, 0.0, lambda_ang[2])
        if joint_compliant_alm == 0 and (not bend_hard):
            bend_dual = wp.vec3(0.0)
        if joint_compliant_alm == 0 and (not twist_hard):
            twist_dual = wp.vec3(0.0)
        bend_primal_k, bend_dual_eff = _material_force_terms(solve_weight_bend, material_bend, bend_dual, joint_compliant_alm)
        twist_primal_k, twist_dual_eff = _material_force_terms(solve_weight_twist, material_twist, twist_dual, joint_compliant_alm)
        K_elastic_diag = wp.vec3(bend_primal_k, bend_primal_k, twist_primal_k)
        bend_alpha = stab_alpha if joint_compliant_alm == 1 or bend_hard else 0.0
        twist_alpha = stab_alpha if joint_compliant_alm == 1 or twist_hard else 0.0
        C0_force = bend_primal_k * bend_alpha * wp.vec3(C0_ang[0], C0_ang[1], 0.0)
        C0_force = C0_force + twist_primal_k * twist_alpha * wp.vec3(0.0, 0.0, C0_ang[2])
        sigma = wp.vec3(0.0)
        H_fric_diag = wp.vec3(0.0)
        dahl_sigma = joint_sigma_start[joint_index]
        dahl_fric = joint_C_fric[joint_index]
        if bend_stiff and (not bend_hard):
            sigma = sigma + wp.vec3(dahl_sigma[0], dahl_sigma[1], 0.0)
            H_fric_diag = H_fric_diag + wp.vec3(dahl_fric[0], dahl_fric[1], 0.0)
        if twist_stiff and (not twist_hard):
            sigma = sigma + wp.vec3(0.0, 0.0, dahl_sigma[2])
            H_fric_diag = H_fric_diag + wp.vec3(0.0, 0.0, dahl_fric[2])
        rod_torque, rod_H_aa, torque_parent, haa_parent = shared_bending(q_wp, q_wc, joint_rod_rest_kb_local[joint_index], joint_rod_rest_twist[joint_index], q_wp, q_wc, is_parent_body, K_elastic_diag, C0_force, sigma, H_fric_diag, bend_dual_eff + twist_dual_eff, wp.vec3(0.0), False, dt, parent_index >= 0, geometry, cursor, count)
        total_torque = total_torque + rod_torque
        total_H_aa = total_H_aa + rod_H_aa
        total_torque_p = total_torque_p + torque_parent
        total_H_aa_p = total_H_aa_p + haa_parent
    stretch_stiff = _structural_row_has_stiffness(solve_weight_stretch, material_stretch, joint_compliant_alm)
    shear_stiff = _structural_row_has_stiffness(solve_weight_shear, material_shear, joint_compliant_alm)
    if stretch_stiff or shear_stiff:
        lambda_lin = joint_lambda_lin[joint_index]
        C0_lin = joint_C0_lin[joint_index]
        stretch_hard = stretch_stiff and joint_compliant_alm == 0 and (joint_is_hard[stretch_idx] == 1)
        shear_hard = shear_stiff and joint_compliant_alm == 0 and (joint_is_hard[shear_idx] == 1)
        stretch_dual = wp.vec3(0.0, 0.0, lambda_lin[2])
        shear_dual = wp.vec3(lambda_lin[0], lambda_lin[1], 0.0)
        if joint_compliant_alm == 0 and (not stretch_hard):
            stretch_dual = wp.vec3(0.0)
        if joint_compliant_alm == 0 and (not shear_hard):
            shear_dual = wp.vec3(0.0)
        stretch_primal_k, stretch_dual_eff = _material_force_terms(solve_weight_stretch, material_stretch, stretch_dual, joint_compliant_alm)
        shear_primal_k, shear_dual_eff = _material_force_terms(solve_weight_shear, material_shear, shear_dual, joint_compliant_alm)
        k_diag = wp.vec3(shear_primal_k, shear_primal_k, stretch_primal_k)
        stretch_alpha = stab_alpha if joint_compliant_alm == 1 or stretch_hard else 0.0
        shear_alpha = stab_alpha if joint_compliant_alm == 1 or shear_hard else 0.0
        C0_force_local = shear_primal_k * shear_alpha * wp.vec3(C0_lin[0], C0_lin[1], 0.0)
        C0_force_local = C0_force_local + stretch_primal_k * stretch_alpha * wp.vec3(0.0, 0.0, C0_lin[2])
        parent_anchor = wp.transform_get_translation(X_wp)
        child_anchor = wp.transform_get_translation(X_wc)
        strain = wp.quat_rotate_inv(q_wp, child_anchor - parent_anchor)
        stress = wp.cw_mul(k_diag, strain) - C0_force_local + (stretch_dual_eff + shear_dual_eff)
        world_force = wp.quat_rotate(q_wp, stress)
        isotropic = shear_primal_k == stretch_primal_k
        total_H_ll = shear_primal_k * wp.identity(3, float)
        if not isotropic:
            tangent = quat_rotate_local_z(q_wp)
            total_H_ll = total_H_ll + (stretch_primal_k - shear_primal_k) * wp.outer(tangent, tangent)
        iso = wp.min(shear_primal_k, stretch_primal_k)
        if not is_parent_body:
            child_arm = child_anchor - wp.transform_point(child_pose, child_com)
            total_force = -world_force
            total_torque = total_torque + wp.cross(child_arm, total_force)
            total_H_al, child_aa = _rod_rate_free_arm_blocks(child_arm, wp.vec3(0.0), total_H_ll, iso, isotropic, False)
            total_H_aa = total_H_aa + child_aa
        if parent_index >= 0:
            parent_center = wp.transform_point(parent_pose, parent_com)
            parent_arm = child_anchor - parent_center
            total_force_p = world_force
            parent_torque = wp.cross(parent_arm, world_force)
            total_torque_p = total_torque_p + parent_torque
            total_H_ll_p = total_H_ll
            total_H_al_p, parent_aa = _rod_rate_free_arm_blocks(parent_arm, parent_anchor - parent_center, total_H_ll, iso, isotropic, True)
            total_H_aa_p = total_H_aa_p + parent_aa
            if is_parent_body:
                total_force = world_force
                total_torque = total_torque + parent_torque
                total_H_al = total_H_al_p
                total_H_aa = total_H_aa + parent_aa
    return (total_force, total_torque, total_H_ll, total_H_al, total_H_aa, total_force_p, total_torque_p, total_H_ll_p, total_H_al_p, total_H_aa_p)

@wp.kernel(enable_backward=False, cuda_max_registers=128)
def scatter_compliant_rigid(dt: float, articulation_joints: wp.array[wp.int32], articulation_joint_body_start: wp.array[wp.int32], articulation_block_row_offsets: wp.array[wp.int32], articulation_block_cols: wp.array[wp.int32], body_articulation_local: wp.array[wp.int32], body_q: wp.array[wp.transform], body_q_prev: wp.array[wp.transform], body_q_rest: wp.array[wp.transform], body_com: wp.array[wp.vec3], joint_type: wp.array[int], joint_enabled: wp.array[bool], joint_parent: wp.array[int], joint_child: wp.array[int], joint_X_p: wp.array[wp.transform], joint_X_c: wp.array[wp.transform], joint_axis: wp.array[wp.vec3], joint_rod_rest_kb_local: wp.array[wp.vec3], joint_rod_rest_twist: wp.array[float], joint_qd_start: wp.array[int], joint_target_q_start: wp.array[int], joint_constraint_start: wp.array[int], joint_penalty_k: wp.array[float], joint_rho: wp.array[float], joint_material_k: wp.array[float], joint_penalty_kd: wp.array[float], joint_sigma_start: wp.array[wp.vec3], joint_C_fric: wp.array[wp.vec3], joint_dof_dim: wp.array2d[int], coord_body_com: wp.array[wp.vec3], coord_joint_type: wp.array[int], coord_parent: wp.array[int], coord_child: wp.array[int], coord_X_p: wp.array[wp.transform], coord_X_c: wp.array[wp.transform], coord_q_start: wp.array[int], coord_qd_start: wp.array[int], coord_dof_dim: wp.array2d[int], coord_axis: wp.array[wp.vec3], coord_history: wp.array[float], joint_friction: wp.array[float], joint_damping: wp.array[float], joint_target_ke: wp.array[float], joint_target_kd: wp.array[float], joint_armature: wp.array[float], joint_armature_target: wp.array[float], joint_target_q: wp.array[float], joint_target_vel: wp.array[float], joint_limit_lower: wp.array[float], joint_limit_upper: wp.array[float], joint_limit_ke: wp.array[float], joint_limit_kd: wp.array[float], joint_drive_limit_support: wp.array[float], joint_drive_lambda: wp.array[float], joint_limit_lambda: wp.array[float], joint_lambda_lin: wp.array[wp.vec3], joint_lambda_ang: wp.array[wp.vec3], joint_C0_lin: wp.array[wp.vec3], joint_C0_ang: wp.array[wp.vec3], joint_is_hard: wp.array[wp.int32], avbd_alpha: float, values_scalar: wp.array[float], rhs_scalar: wp.array[float], edges: wp.array[wp.vec4i], geometry: wp.array[wp.vec3], joint_count: int, lane_mask: int):
    joint_coordinates = CompactCoord()
    joint_coordinates.body_com = coord_body_com
    joint_coordinates.joint_type = coord_joint_type
    joint_coordinates.parent = coord_parent
    joint_coordinates.child = coord_child
    joint_coordinates.X_p = coord_X_p
    joint_coordinates.X_c = coord_X_c
    joint_coordinates.q_start = coord_q_start
    joint_coordinates.qd_start = coord_qd_start
    joint_coordinates.dof_dim = coord_dof_dim
    joint_coordinates.axis = coord_axis
    joint_coordinates.history = coord_history
    joint_compliant_alm = int(1)
    task, joint_cursor, stripe = wp.tid()
    joint = articulation_joints[joint_cursor]
    if not joint_enabled[joint]:
        return
    jt = joint_type[joint]
    if not _sparse_joint_supported(jt):
        return
    child = joint_child[joint]
    parent = joint_parent[joint]
    if child < 0:
        return
    body_start = articulation_joint_body_start[joint_cursor]
    child_local = body_articulation_local[child]
    parent_local = -1
    if parent >= 0:
        parent_local = body_articulation_local[parent]
    X_p = joint_X_p[joint]
    X_c = joint_X_c[joint]
    X_p_prev = joint_X_p[joint]
    X_c_prev = joint_X_c[joint]
    cross_slot = int(-1)
    if jt == JointType.ROD:
        entry = edges[joint_cursor]
        no_rates = entry[3] != 0
        if no_rates:
            if task == 1:
                if parent >= 0:
                    rate_free_cross(dt, joint, parent, child, body_q, body_com, X_p, X_c, joint_constraint_start, joint_penalty_k, joint_rho, joint_material_k, joint_penalty_kd, joint_C_fric, joint_is_hard, joint_compliant_alm, stripe, lane_mask, geometry, joint_cursor, joint_count, values_scalar, entry[2], parent_local < child_local)
                return
            endpoint_force, torque, ll, al, aa, endpoint_force_p, torque_p, ll_p, al_p, aa_p = rate_free_response(joint, body_q, body_q_prev, body_com, joint_parent, joint_child, X_p, X_c, X_p_prev, X_c_prev, joint_rod_rest_kb_local, joint_rod_rest_twist, joint_constraint_start, joint_penalty_k, joint_rho, joint_material_k, joint_penalty_kd, joint_sigma_start, joint_C_fric, joint_lambda_lin, joint_lambda_ang, joint_C0_lin, joint_C0_ang, joint_is_hard, avbd_alpha, joint_compliant_alm, dt, geometry, joint_cursor, joint_count)
            scatter_endpoint(values_scalar, rhs_scalar, body_start + child_local, entry[0], endpoint_force, torque, ll, al, aa, stripe, lane_mask)
            if parent >= 0:
                scatter_endpoint(values_scalar, rhs_scalar, body_start + parent_local, entry[1], endpoint_force_p, torque_p, ll_p, al_p, aa_p, stripe, lane_mask)
            return
        if task != 0:
            return
        endpoint_force, torque, ll, al, aa, endpoint_force_p, torque_p, ll_p, al_p, aa_p = paired_rod_response(joint, body_q, body_q_prev, body_com, joint_parent, joint_child, X_p, X_c, X_p_prev, X_c_prev, joint_rod_rest_kb_local, joint_rod_rest_twist, joint_constraint_start, joint_penalty_k, joint_rho, joint_material_k, joint_penalty_kd, joint_sigma_start, joint_C_fric, joint_lambda_lin, joint_lambda_ang, joint_C0_lin, joint_C0_ang, joint_is_hard, avbd_alpha, joint_compliant_alm, dt)
        scatter_endpoint(values_scalar, rhs_scalar, body_start + child_local, entry[0], endpoint_force, torque, ll, al, aa, stripe, lane_mask)
        if parent >= 0:
            scatter_endpoint(values_scalar, rhs_scalar, body_start + parent_local, entry[1], endpoint_force_p, torque_p, ll_p, al_p, aa_p, stripe, lane_mask)
            cross = _rod_cross_hessian(dt, joint, parent, child, body_q, body_com, X_p, X_c, joint_constraint_start, joint_penalty_k, joint_rho, joint_material_k, joint_penalty_kd, joint_C_fric, joint_is_hard, joint_compliant_alm, stripe, lane_mask)
            scatter_link(values_scalar, entry[2], parent_local < child_local, cross, stripe, lane_mask)
        return
    if task != 0:
        return
    c_start = joint_constraint_start[joint]
    qd_start = joint_qd_start[joint]
    lin_count = int(jt == JointType.PRISMATIC)
    ang_count = int(jt == JointType.REVOLUTE)
    if jt == JointType.D6:
        lin_count = joint_dof_dim[joint, 0]
        ang_count = joint_dof_dim[joint, 1]
    g = _sparse_joint_geometry(joint, jt, parent, child, X_p, X_c, X_p_prev, X_c_prev, body_q, body_q_prev, body_q_rest, joint_X_p, joint_X_c, joint_axis, joint_qd_start, joint_constraint_start, joint_penalty_k, joint_rho, joint_material_k, joint_penalty_kd, joint_dof_dim, joint_lambda_lin, joint_lambda_ang, joint_C0_lin, joint_C0_ang, joint_is_hard, avbd_alpha, joint_compliant_alm)
    if g.k_linear > 0.0 and (jt != JointType.D6 or lin_count < 3):
        _assemble_linear_joint_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, parent, child, g.parent_anchor, g.child_anchor, g.parent_anchor_prev, g.child_anchor_prev, body_q, body_com, g.k_linear, g.kd_linear, dt, g.P_lin, g.lin_lambda, g.lin_C0, g.lin_alpha, cross_slot, stripe, lane_mask)
    if g.k_angular > 0.0 and jt != JointType.BALL and (jt != JointType.D6 or ang_count < 3):
        sigma0 = wp.vec3(0.0)
        C_fric = wp.vec3(0.0)
        _kappa, _J_world = _assemble_angular_joint_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, parent, g.parent_anchor_q, g.child_anchor_q, g.parent_anchor_q_prev, g.child_anchor_q_prev, g.parent_rest_q, g.child_rest_q, g.k_angular, g.kd_angular, dt, g.P_ang, sigma0, C_fric, g.ang_lambda, g.ang_C0, g.ang_alpha, cross_slot, joint_axis, qd_start + lin_count, ang_count, stripe, lane_mask)
    if jt == JointType.REVOLUTE or jt == JointType.PRISMATIC or jt == JointType.D6:
        for component in range(lin_count + ang_count):
            active, gp, gc, force, hessian = _sparse_joint_friction_row_no_modal(joint_coordinates, joint, component, lin_count, qd_start + component, body_q, body_q_prev, joint_friction, joint_damping, dt)
            if active:
                _assemble_coordinate_row_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, gp, gc, force, hessian, cross_slot, stripe, lane_mask)
    if jt == JointType.REVOLUTE:
        active, gp, gc, force, hessian = _sparse_joint_armature_row_no_modal(joint_coordinates, joint, qd_start, body_q, joint_armature, joint_armature_target, dt)
        if active:
            _assemble_coordinate_row_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, gp, gc, force, hessian, cross_slot, stripe, lane_mask)
    if jt == JointType.REVOLUTE:
        force_scalar, hessian_scalar, angular_jacobian_world = _sparse_angular_drive_limit_row_no_modal(joint_coordinates, joint, 0, qd_start, joint_target_q_start[joint], c_start + 2, body_q, body_q_prev, joint_target_ke, joint_target_kd, joint_target_q, joint_target_vel, joint_limit_lower, joint_limit_upper, joint_limit_ke, joint_limit_kd, joint_penalty_k, joint_drive_limit_support, joint_drive_lambda, joint_limit_lambda, joint_compliant_alm, dt)
        if hessian_scalar > 0.0:
            _assemble_angular_axis_row_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, parent, angular_jacobian_world, force_scalar, hessian_scalar, cross_slot, stripe, lane_mask)
    if jt == JointType.PRISMATIC or jt == JointType.D6:
        C_vec = g.child_anchor - g.parent_anchor
        C_vec_prev = g.child_anchor_prev - g.parent_anchor_prev
        target_q_base = joint_target_q_start[joint]
        for li in range(3):
            if li < lin_count:
                force_scalar, hessian_scalar, axis_world = _sparse_linear_drive_limit_row(qd_start + li, target_q_base + li, c_start + 2 + li, g.parent_anchor_q, C_vec, C_vec_prev, joint_axis, joint_target_ke, joint_target_kd, joint_target_q, joint_target_vel, joint_limit_lower, joint_limit_upper, joint_limit_ke, joint_limit_kd, joint_penalty_k, joint_drive_limit_support, joint_drive_lambda, joint_limit_lambda, joint_compliant_alm, dt)
                if hessian_scalar > 0.0:
                    _assemble_linear_axis_row_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, parent, child, g.parent_anchor, g.child_anchor, body_q, body_com, axis_world, force_scalar, hessian_scalar, cross_slot, stripe, lane_mask)
    if jt == JointType.D6:
        target_q_base = joint_target_q_start[joint]
        for ai in range(3):
            if ai < ang_count:
                force_scalar, hessian_scalar, angular_jacobian_world = _sparse_angular_drive_limit_row_no_modal(joint_coordinates, joint, lin_count + ai, qd_start + lin_count + ai, target_q_base + lin_count + ai, c_start + 2 + lin_count + ai, body_q, body_q_prev, joint_target_ke, joint_target_kd, joint_target_q, joint_target_vel, joint_limit_lower, joint_limit_upper, joint_limit_ke, joint_limit_kd, joint_penalty_k, joint_drive_limit_support, joint_drive_lambda, joint_limit_lambda, joint_compliant_alm, dt)
                if hessian_scalar > 0.0:
                    _assemble_angular_axis_row_scalar(values_scalar, rhs_scalar, articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, child_local, parent, angular_jacobian_world, force_scalar, hessian_scalar, cross_slot, stripe, lane_mask)

def _bind(desc, tensors, views, packed=False):
    if not isinstance(desc, dict):
        return desc
    if 'array' in desc:
        dtype = getattr(wp, desc['warp_type'].split('.')[-1].rstrip("'>"))
        key = (desc['array'], dtype, packed)
        if key not in views:
            views[key] = wp.from_torch(tensors[key[0]], dtype=dtype, requires_grad=False, return_ctype=packed)
        return views[key]
    struct_type = {'SparseElasticJointData':SparseElasticJointData, 'JointCoordinateData':JointCoordinateData}[desc['warp_struct']]
    result = struct_type()
    for name, value in desc['fields'].items():
        if value is not None:
            setattr(result, name, _bind(value, tensors, views))
    return result


def run(step_size_h, step_size_inv_h2, step_size_bdf2, articulation_joints, articulation_joint_body_start, articulation_block_row_offsets, articulation_block_cols, body_articulation_local, body_q, body_q_prev, body_q_rest, body_inertia_q, body_com, joint_type, joint_enabled, joint_parent, joint_child, joint_X_p, joint_X_c, joint_axis, joint_rod_rest_kb_local, joint_rod_rest_twist, joint_qd_start, joint_target_q_start, joint_constraint_start, joint_penalty_k, joint_rho, joint_material_k, joint_penalty_kd, joint_sigma_start, joint_C_fric, joint_dof_dim, joint_coordinates_body_com, joint_coordinates_joint_type, joint_coordinates_parent, joint_coordinates_child, joint_coordinates_X_p, joint_coordinates_X_c, joint_coordinates_q_start, joint_coordinates_qd_start, joint_coordinates_dof_dim, joint_coordinates_axis, joint_coordinates_world, joint_coordinates_history, joint_friction, joint_damping, joint_target_ke, joint_target_kd, joint_armature, joint_armature_target, armature_body_disp_ang, armature_pred_ang, armature_delta, joint_target_q, joint_target_vel, joint_limit_lower, joint_limit_upper, joint_limit_ke, joint_limit_kd, joint_drive_limit_support, joint_drive_lambda, joint_limit_lambda, joint_lambda_lin, joint_lambda_ang, joint_C0_lin, joint_C0_ang, joint_is_hard, elastic_body_elastic_index, elastic_elastic_joint, elastic_elastic_mode_count, elastic_joint_q_start, elastic_parent_endpoint, elastic_child_endpoint, elastic_endpoint_phi, elastic_endpoint_psi, elastic_joint_q, elastic_joint_q_prev, elastic_joint_cross_slot, values_scalar, rhs_scalar, dispatch, out_values_scalar, out_rhs_scalar):
    tensors = locals().copy()
    meta = json.loads(dispatch)
    params = meta['params']
    stream = wp.get_stream(str(values_scalar.device))
    device = str(values_scalar.device)
    carry = [wp.from_torch(value, dtype=wp.float32, requires_grad=False, return_ctype=True) for value in (values_scalar, rhs_scalar, out_values_scalar, out_rhs_scalar)]
    tensors['values_scalar'] = out_values_scalar
    tensors['rhs_scalar'] = out_rhs_scalar
    keys = ('dt', 'articulation_joints', 'articulation_joint_body_start', 'articulation_block_row_offsets', 'articulation_block_cols', 'body_articulation_local', 'body_q', 'body_q_prev', 'body_q_rest', 'body_com', 'joint_type', 'joint_enabled', 'joint_parent', 'joint_child', 'joint_X_p', 'joint_X_c', 'joint_axis', 'joint_rod_rest_kb_local', 'joint_rod_rest_twist', 'joint_qd_start', 'joint_target_q_start', 'joint_constraint_start', 'joint_penalty_k', 'joint_rho', 'joint_material_k', 'joint_penalty_kd', 'joint_sigma_start', 'joint_C_fric', 'joint_dof_dim', 'joint_coordinates', 'joint_friction', 'joint_damping', 'joint_target_ke', 'joint_target_kd', 'joint_armature', 'joint_armature_target', 'joint_target_q', 'joint_target_vel', 'joint_limit_lower', 'joint_limit_upper', 'joint_limit_ke', 'joint_limit_kd', 'joint_drive_limit_support', 'joint_drive_lambda', 'joint_limit_lambda', 'joint_lambda_lin', 'joint_lambda_ang', 'joint_C0_lin', 'joint_C0_ang', 'joint_is_hard', 'avbd_alpha', 'joint_compliant_alm', 'elastic', 'values_scalar', 'rhs_scalar')
    joint_count = meta['options']['dim']
    lanes = 1
    target_threads = 2560
    if joint_count >= 2048:
        rows = max(1, articulation_block_row_offsets.numel() - 1)
        target_threads = 16384 if articulation_block_cols.numel() > 3 * rows else 4096
    while lanes < 32 and joint_count * lanes < target_threads:
        lanes *= 2
    no_modal = not params['elastic']['fields']['enabled'] and not params['joint_coordinates']['fields']['elastic_max_mode_count']
    carry_tensors = (values_scalar, rhs_scalar, out_values_scalar, out_rhs_scalar)
    aligned = all(value.is_contiguous() and value.storage_offset() % 4 == 0 and value.numel() >= 4 for value in carry_tensors)
    scatter = scatter_compliant_rigid if aligned and no_modal and params['joint_compliant_alm'] == 1 else sparse_scatter
    views = {('values_scalar', wp.float32, True): carry[2], ('rhs_scalar', wp.float32, True): carry[3]}
    bound = {}
    for name in keys:
        if scatter is scatter_compliant_rigid and name in ('elastic', 'joint_compliant_alm'):
            continue
        if scatter is scatter_compliant_rigid and name == 'joint_coordinates':
            for field in ('body_com', 'joint_type', 'parent', 'child', 'X_p', 'X_c', 'q_start', 'qd_start', 'dof_dim', 'axis', 'history'):
                bound['coord_'+field] = _bind(params[name]['fields'][field], tensors, views, packed=True)
        else:
            bound[name] = _bind(params[name], tensors, views, packed=True)
    args = list(bound.values())
    extra = []
    if scatter is scatter_compliant_rigid:
        storage = torch.empty(25 * joint_count, device=values_scalar.device, dtype=torch.float32)
        edges = wp.from_torch(storage[:4*joint_count].view(torch.int32).view(-1, 4), dtype=wp.vec4i, requires_grad=False, return_ctype=True)
        packets = [wp.from_torch(value[:(value.numel()//4)*4].view(-1, 4), dtype=wp.vec4, requires_grad=False, return_ctype=True) for value in carry_tensors]
        packet_count = values_scalar.numel()//4 + rhs_scalar.numel()//4
        copy_threads = ((packet_count + 63) // 64) * 32
        topology = [bound[name] for name in ('articulation_joints', 'articulation_joint_body_start', 'articulation_block_row_offsets', 'articulation_block_cols', 'body_articulation_local', 'joint_type', 'joint_enabled', 'joint_parent', 'joint_child', 'joint_constraint_start', 'joint_penalty_kd')]
        geometry_storage = storage[4*joint_count:].view(-1, 3)
        geometry = wp.from_torch(geometry_storage, dtype=wp.vec3, requires_grad=False, return_ctype=True)
        angular_inputs = [bound[name] for name in ('body_q', 'body_q_prev', 'joint_X_p', 'joint_X_c', 'joint_rod_rest_kb_local', 'joint_rod_rest_twist')]
        wp.launch(restore_and_index, dim=copy_threads + 64*((joint_count+31)//32), inputs=carry + packets + topology + [edges] + angular_inputs + [geometry, copy_threads, joint_count], device=device, stream=stream, block_dim=32)
        extra = [edges, geometry, joint_count]
    else:
        wp.launch(restore_carry, dim=((values_scalar.numel() + rhs_scalar.numel() + 255) // 256)*32, inputs=carry, device=device, stream=stream, block_dim=32)
    shape = (2, joint_count, lanes) if scatter is scatter_compliant_rigid else (joint_count, lanes)
    wp.launch(scatter, dim=shape, inputs=args+extra+[lanes-1], device=device, stream=stream, block_dim=32)
