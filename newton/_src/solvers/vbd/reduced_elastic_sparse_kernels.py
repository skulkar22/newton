# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reduced elastic bodies inside the VBD block-sparse articulation solve.

Each elastic body ``b`` contributes a dense ``(6 + n)`` block assembled by
:func:`~newton._src.solvers.vbd.reduced_elastic_kernels.assemble_elastic_joints`
(frame rows in body axes about the rest COM, modal rows ``m``). Its joints
``j`` couple the modal coordinates to neighbor frames ``c`` through
``K_mc = G_j^T H_bc``, where ``H_bc`` is the joint's frame-frame cross block in
the sparse system and the columns of ``G_j`` are the body-frame twists that move
the deformed attachment like mode ``i``:
``G_ji = [phi_i + r_j x psi_i, psi_i]`` (world axes, ``r_j`` from the rest COM to
the deformed attachment).

The modal coordinates are eliminated exactly. With the column groups
``B = [A_mf, K_mc1, K_mc2, ...]`` (own frame first) and ``W = A_mm^-1 B``,
``z = A_mm^-1 r_m``, the frame system receives ``-B_g^T W_h`` on the block of
frames ``(g, h)`` and ``-B_g^T z`` on the right-hand side of frame ``g``.
After the sparse solve, ``dm = z - sum_h W_h dx_h``. The mode-mode coupling of two
directly jointed elastic bodies is the only term not represented.
"""

import warp as wp

from newton._src.solvers.vbd.joint_coordinates import eval_elastic_endpoint_xform
from newton._src.solvers.vbd.reduced_elastic_kernels import _so3_left_jacobian
from newton._src.solvers.vbd.rigid_sparse_articulation_kernels import (
    _scalar_block_get,
    _scalar_block_index,
    _scalar_block_set,
    _scalar_vec_get,
    _scalar_vec_index,
    _scalar_vec_set,
    vec6f,
)

wp.set_module_options({"enable_backward": False})

_STATIC_FRAME_DIAGONAL = wp.constant(1.0e30)


@wp.func
def _dense_entry(matrix: wp.array3d[float], elastic_index: int, i: int, j: int) -> float:
    """Return a symmetric dense-block entry from its stored upper triangle."""
    if j < i:
        return matrix[elastic_index, j, i]
    return matrix[elastic_index, i, j]


@wp.kernel
def assemble_elastic_sparse_frame_blocks(
    elastic_bodies: wp.array[wp.int32],
    elastic_body_row_start: wp.array[wp.int32],
    body_articulation_local: wp.array[wp.int32],
    articulation_diag_slots: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    elastic_body_block_grad: wp.array2d[float],
    elastic_body_block_matrix: wp.array3d[float],
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
):
    """Replace an elastic body's sparse diagonal block and right-hand side by its dense frame rows.

    The dense block already holds the frame inertia (including ``I(q)``), contact Hessians, and
    the self terms of every attached joint at the deformed attachments, so it supersedes the
    rigid inertia and joint self terms the sparse assembly wrote for this body. Frame rows are
    rotated from body to world axes. Under the displacement-form update the dense rows already
    take the inertial force from the accumulated displacement, as for rigid sparse bodies.
    """
    tid = wp.tid()
    body = elastic_bodies[tid]
    elastic_index = body_elastic_index[body]
    row = elastic_body_row_start[tid] + body_articulation_local[body]
    diag_slot = articulation_diag_slots[row]

    if body_inv_mass[body] == 0.0:
        for i in range(6):
            _scalar_vec_set(rhs_scalar, row, i, 0.0)
            for j in range(6):
                value = float(0.0)
                if i == j:
                    value = _STATIC_FRAME_DIAGONAL
                _scalar_block_set(values_scalar, diag_slot, i, j, value)
        return

    pose = body_q[body]
    rot = wp.transform_get_rotation(pose)
    R = wp.quat_to_matrix(rot)

    # World frame rows: A_w = T A T^T and r_w = -T g with T = diag(R, R).
    for bi in range(2):
        for bj in range(2):
            local = wp.mat33(0.0)
            for i in range(3):
                for j in range(3):
                    local[i, j] = _dense_entry(elastic_body_block_matrix, elastic_index, 3 * bi + i, 3 * bj + j)
            world = R * local * wp.transpose(R)
            for i in range(3):
                for j in range(3):
                    _scalar_block_set(values_scalar, diag_slot, 3 * bi + i, 3 * bj + j, world[i, j])

    force_local = wp.vec3(
        -elastic_body_block_grad[elastic_index, 0],
        -elastic_body_block_grad[elastic_index, 1],
        -elastic_body_block_grad[elastic_index, 2],
    )
    torque_local = wp.vec3(
        -elastic_body_block_grad[elastic_index, 3],
        -elastic_body_block_grad[elastic_index, 4],
        -elastic_body_block_grad[elastic_index, 5],
    )
    force = R * force_local
    torque = R * torque_local

    for i in range(3):
        _scalar_vec_set(rhs_scalar, row, i, force[i])
        _scalar_vec_set(rhs_scalar, row, 3 + i, torque[i])


@wp.func
def _modal_solve_inplace(
    matrix: wp.array3d[float],
    elastic_index: int,
    factor_diag: wp.array[float],
    factor_offset: int,
    definite: int,
    mode_count: int,
    x: wp.array[float],
    x_offset: int,
    x_stride: int,
):
    """Overwrite ``x`` with ``A_mm^-1 x`` using the factor from :func:`_factor_modal_block`.

    The strictly lower Cholesky factor is stored in the lower triangle of the dense modal block
    and its diagonal in ``factor_diag``. Without a factor (``definite == 0``) this is the Jacobi
    step of the local block solve's fallback.
    """
    if definite == 0:
        for i in range(mode_count):
            diagonal = matrix[elastic_index, 6 + i, 6 + i]
            value = float(0.0)
            if diagonal > 0.0:
                value = x[x_offset + i * x_stride] / diagonal
            x[x_offset + i * x_stride] = value
        return

    for i in range(mode_count):
        value = x[x_offset + i * x_stride]
        for k in range(i):
            value = value - matrix[elastic_index, 6 + i, 6 + k] * x[x_offset + k * x_stride]
        x[x_offset + i * x_stride] = value / factor_diag[factor_offset + i]
    for ii in range(mode_count):
        i = mode_count - 1 - ii
        value = x[x_offset + i * x_stride]
        for k in range(i + 1, mode_count):
            value = value - matrix[elastic_index, 6 + k, 6 + i] * x[x_offset + k * x_stride]
        x[x_offset + i * x_stride] = value / factor_diag[factor_offset + i]


@wp.func
def _factor_modal_block(
    matrix: wp.array3d[float],
    elastic_index: int,
    factor_diag: wp.array[float],
    factor_offset: int,
    mode_count: int,
) -> int:
    """Cholesky-factor the modal block in place below its stored upper triangle.

    Returns 1 for a positive definite block and 0 otherwise.
    """
    for j in range(mode_count):
        value = matrix[elastic_index, 6 + j, 6 + j]
        for k in range(j):
            l_jk = matrix[elastic_index, 6 + j, 6 + k]
            value = value - l_jk * l_jk
        if not (value > 0.0) or not wp.isfinite(value):
            return 0
        diagonal = wp.sqrt(value)
        factor_diag[factor_offset + j] = diagonal
        for i in range(j + 1, mode_count):
            value_ij = matrix[elastic_index, 6 + j, 6 + i]
            for k in range(j):
                value_ij = value_ij - matrix[elastic_index, 6 + i, 6 + k] * matrix[elastic_index, 6 + j, 6 + k]
            matrix[elastic_index, 6 + i, 6 + j] = value_ij / diagonal
    return 1


@wp.kernel
def build_elastic_sparse_schur_columns(
    elastic_bodies: wp.array[wp.int32],
    elastic_group_owner: wp.array[wp.int32],
    elastic_group_joint: wp.array[wp.int32],
    elastic_group_body: wp.array[wp.int32],
    elastic_group_neighbor_elastic: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    elastic_joint: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_max_mode_count: int,
    joint_q_start: wp.array[int],
    joint_parent: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_parent_elastic_endpoint: wp.array[wp.int32],
    joint_child_elastic_endpoint: wp.array[wp.int32],
    elastic_endpoint_phi: wp.array[wp.vec3],
    elastic_endpoint_psi: wp.array[wp.vec3],
    joint_cross_slot: wp.array[wp.int32],
    joint_q: wp.array[float],
    values_scalar: wp.array[float],
    elastic_body_block_matrix: wp.array3d[float],
    modal_b: wp.array[float],
):
    """Form one mode's row of a Schur column group ``B`` (world axes), stored as ``[group, mode, 6]``.

    The own-frame group is the frame-mode coupling ``A_mf`` of the dense block; a joint group is
    ``K_mc = G^T H_bc`` with the joint's captured cross block ``H_bc``.

    A joint between two elastic bodies also couples their modes, ``X = G_b^T H_bc G_c``, which
    the per-body elimination cannot represent. Dropping ``X`` alone can make the system
    indefinite, so the joint's modal self term ``P_b = -a^T K_x a`` (``a`` the attachment motion
    of each mode, ``K_x`` the joint's attachment-space cross block, ``-K_x`` its self block) is
    added once more to ``A_mm``. The modified matrix exceeds the exact one by the positive
    semidefinite ``[[P_b, -X], [-X^T, P_c]]``, a conservative step with an unchanged fixed point.
    """
    tid = wp.tid()
    max_modes = elastic_max_mode_count
    group = tid // max_modes
    mode = tid - group * max_modes
    body = elastic_bodies[elastic_group_owner[group]]
    elastic_index = body_elastic_index[body]
    mode_count = elastic_mode_count[elastic_index]
    if mode >= mode_count:
        return

    pose = body_q[body]
    R = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    row = (group * max_modes + mode) * 6
    joint = elastic_group_joint[group]
    if joint < 0:
        linear = wp.vec3(0.0)
        angular = wp.vec3(0.0)
        for axis in range(3):
            linear[axis] = elastic_body_block_matrix[elastic_index, axis, 6 + mode]
            angular[axis] = elastic_body_block_matrix[elastic_index, 3 + axis, 6 + mode]
        linear = R * linear
        angular = R * angular
        for axis in range(3):
            modal_b[row + axis] = linear[axis]
            modal_b[row + 3 + axis] = angular[axis]
        return

    is_parent = joint_parent[joint] == body
    endpoint = joint_child_elastic_endpoint[joint]
    X_rest = joint_X_c[joint]
    if is_parent:
        endpoint = joint_parent_elastic_endpoint[joint]
        X_rest = joint_X_p[joint]
    endpoint_xform = eval_elastic_endpoint_xform(
        joint,
        body,
        is_parent,
        X_rest,
        body_elastic_index,
        elastic_joint,
        elastic_mode_count,
        joint_q,
        joint_q_start,
        joint_parent_elastic_endpoint,
        joint_child_elastic_endpoint,
        elastic_endpoint_phi,
        elastic_endpoint_psi,
        max_modes,
    )
    r = wp.transform_point(pose, wp.transform_get_translation(endpoint_xform)) - wp.transform_point(
        pose, body_com[body]
    )

    owner_q_start = joint_q_start[elastic_joint[elastic_index]] + 7
    theta_local = wp.vec3(0.0)
    for k in range(mode_count):
        theta_local = theta_local + elastic_endpoint_psi[endpoint * max_modes + k] * joint_q[owner_q_start + k]

    # Body twist that moves the deformed attachment like this mode: [phi + r x psi, psi].
    phi = R * elastic_endpoint_phi[endpoint * max_modes + mode]
    psi = R * (_so3_left_jacobian(theta_local) * elastic_endpoint_psi[endpoint * max_modes + mode])
    twist_linear = phi + wp.cross(r, psi)
    slot = joint_cross_slot[joint]
    b_row = vec6f(0.0)
    for col in range(6):
        value = float(0.0)
        for k in range(3):
            h_lin = float(0.0)
            h_ang = float(0.0)
            # The capture slot stores parent rows and child columns.
            if is_parent:
                h_lin = _scalar_block_get(values_scalar, slot, k, col)
                h_ang = _scalar_block_get(values_scalar, slot, 3 + k, col)
            else:
                h_lin = _scalar_block_get(values_scalar, slot, col, k)
                h_ang = _scalar_block_get(values_scalar, slot, col, 3 + k)
            value = value + twist_linear[k] * h_lin + psi[k] * h_ang
        modal_b[row + col] = value
        b_row[col] = value

    if elastic_group_neighbor_elastic[group] == 0:
        return
    # Twist of the neighbor that moves the shared attachment like mode ``other`` of this body.
    neighbor = elastic_group_body[group]
    attachment = wp.transform_point(pose, wp.transform_get_translation(endpoint_xform))
    r_neighbor = attachment - wp.transform_point(body_q[neighbor], body_com[neighbor])
    for other in range(mode_count):
        phi_o = R * elastic_endpoint_phi[endpoint * max_modes + other]
        psi_o = R * (_so3_left_jacobian(theta_local) * elastic_endpoint_psi[endpoint * max_modes + other])
        twist_o = phi_o + wp.cross(r_neighbor, psi_o)
        value = -(
            b_row[0] * twist_o[0]
            + b_row[1] * twist_o[1]
            + b_row[2] * twist_o[2]
            + b_row[3] * psi_o[0]
            + b_row[4] * psi_o[1]
            + b_row[5] * psi_o[2]
        )
        if other == mode:
            wp.atomic_add(elastic_body_block_matrix, elastic_index, 6 + mode, 6 + mode, value)
        else:
            wp.atomic_add(
                elastic_body_block_matrix,
                elastic_index,
                6 + wp.min(mode, other),
                6 + wp.max(mode, other),
                0.5 * value,
            )


@wp.kernel
def factor_elastic_sparse_modes(
    elastic_bodies: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_max_mode_count: int,
    elastic_body_block_grad: wp.array2d[float],
    elastic_body_block_matrix: wp.array3d[float],
    factor_diag: wp.array[float],
    modal_definite: wp.array[wp.int32],
    modal_z: wp.array[float],
):
    """Factor one elastic body's modal block and solve ``z = A_mm^-1 r_m``."""
    tid = wp.tid()
    elastic_index = body_elastic_index[elastic_bodies[tid]]
    mode_count = elastic_mode_count[elastic_index]
    offset = tid * elastic_max_mode_count
    definite = _factor_modal_block(elastic_body_block_matrix, elastic_index, factor_diag, offset, mode_count)
    modal_definite[tid] = definite
    for mode in range(mode_count):
        modal_z[offset + mode] = -elastic_body_block_grad[elastic_index, 6 + mode]
    _modal_solve_inplace(
        elastic_body_block_matrix, elastic_index, factor_diag, offset, definite, mode_count, modal_z, offset, 1
    )


@wp.kernel
def solve_elastic_sparse_schur_columns(
    elastic_bodies: wp.array[wp.int32],
    elastic_group_owner: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_max_mode_count: int,
    elastic_body_block_matrix: wp.array3d[float],
    factor_diag: wp.array[float],
    modal_definite: wp.array[wp.int32],
    modal_b: wp.array[float],
    modal_w: wp.array[float],
):
    """Solve one column of ``W = A_mm^-1 B`` for a Schur column group."""
    tid = wp.tid()
    group = tid // 6
    col = tid - group * 6
    owner = elastic_group_owner[group]
    elastic_index = body_elastic_index[elastic_bodies[owner]]
    mode_count = elastic_mode_count[elastic_index]
    max_modes = elastic_max_mode_count
    for mode in range(mode_count):
        index = (group * max_modes + mode) * 6 + col
        modal_w[index] = modal_b[index]
    _modal_solve_inplace(
        elastic_body_block_matrix,
        elastic_index,
        factor_diag,
        owner * max_modes,
        modal_definite[owner],
        mode_count,
        modal_w,
        group * max_modes * 6 + col,
        6,
    )


@wp.kernel
def scatter_elastic_sparse_schur(
    elastic_bodies: wp.array[wp.int32],
    elastic_body_row_start: wp.array[wp.int32],
    elastic_group_owner: wp.array[wp.int32],
    elastic_group_local: wp.array[wp.int32],
    elastic_pair_left: wp.array[wp.int32],
    elastic_pair_right: wp.array[wp.int32],
    elastic_pair_slot: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_max_mode_count: int,
    modal_z: wp.array[float],
    modal_b: wp.array[float],
    modal_w: wp.array[float],
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
):
    """Add the modal Schur terms ``-B_g^T W_h`` and ``-B_g^T z`` to the sparse frame system.

    One thread per entry: the first ``36 * pair_count`` threads update the stored lower block
    triangle, the remaining ``6 * group_count`` threads the right-hand side.
    """
    tid = wp.tid()
    max_modes = elastic_max_mode_count
    pair_entries = elastic_pair_slot.shape[0] * 36
    if tid < pair_entries:
        pair = tid // 36
        entry = tid - pair * 36
        i = entry // 6
        j = entry - i * 6
        left = elastic_pair_left[pair]
        right = elastic_pair_right[pair]
        owner = elastic_group_owner[left]
        mode_count = elastic_mode_count[body_elastic_index[elastic_bodies[owner]]]
        accum = float(0.0)
        for mode in range(mode_count):
            accum = accum + modal_b[(left * max_modes + mode) * 6 + i] * modal_w[(right * max_modes + mode) * 6 + j]
        wp.atomic_add(values_scalar, _scalar_block_index(elastic_pair_slot[pair], i, j), -accum)
        return

    entry = tid - pair_entries
    group = entry // 6
    col = entry - group * 6
    owner = elastic_group_owner[group]
    mode_count = elastic_mode_count[body_elastic_index[elastic_bodies[owner]]]
    accum = float(0.0)
    for mode in range(mode_count):
        accum = accum + modal_b[(group * max_modes + mode) * 6 + col] * modal_z[owner * max_modes + mode]
    row = elastic_body_row_start[owner] + elastic_group_local[group]
    wp.atomic_add(rhs_scalar, _scalar_vec_index(row, col), -accum)


@wp.kernel
def update_elastic_sparse_modes(
    dt: float,
    relaxation: float,
    elastic_bodies: wp.array[wp.int32],
    elastic_body_row_start: wp.array[wp.int32],
    elastic_group_offsets: wp.array[wp.int32],
    elastic_group_local: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_max_mode_count: int,
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q_prev: wp.array[float],
    modal_z: wp.array[float],
    modal_w: wp.array[float],
    delta_scalar: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    """Recover the modal update ``dm = z - sum_h W_h dx_h`` from the solved frame updates.

    The modes take the same relaxation as the frames, and their velocities follow the local
    block solve's convention ``qdot = (q - q_prev) / dt``.
    """
    tid = wp.tid()
    max_modes = elastic_max_mode_count
    index = tid // max_modes
    mode = tid - index * max_modes
    elastic_index = body_elastic_index[elastic_bodies[index]]
    if mode >= elastic_mode_count[elastic_index]:
        return
    row_start = elastic_body_row_start[index]
    owner = elastic_joint[elastic_index]
    q_start = joint_q_start[owner] + 7 + mode
    qd_start = joint_qd_start[owner] + 6 + mode

    update = modal_z[index * max_modes + mode]
    for group in range(elastic_group_offsets[index], elastic_group_offsets[index + 1]):
        row = row_start + elastic_group_local[group]
        for col in range(6):
            update = update - modal_w[(group * max_modes + mode) * 6 + col] * _scalar_vec_get(delta_scalar, row, col)
    q_new = joint_q[q_start] + relaxation * update
    joint_q[q_start] = q_new
    joint_qd[qd_start] = (q_new - joint_q_prev[q_start]) * (1.0 / dt)
