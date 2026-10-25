# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assembly of the variable-size block system of mixed rigid/elastic articulations in the VBD solve.

A mixed articulation of a layout built with ``elastic_solve="coupled"`` has one sparse row per
body of size ``n_i = 6 + m_i`` (see
:mod:`~newton._src.solvers.vbd.rigid_sparse_articulation_variable_kernels` for the storage). Row
components ``0..2`` are the world-axes displacement ``dc`` of the body's center of mass
``c = transform_point(body_q, body_com)``, ``3..5`` the world rotation vector ``dtheta`` of the
update ``q_new = exp(dtheta) q`` (both as in the 6 x 6 kernels), and ``6..n_i - 1`` the
increments of the body's modal coordinates. The system is ``H dx = rhs`` with ``rhs`` the
negative gradient of the iterate's incremental potential.

Per iterate, :func:`assemble_articulation_variable_system` runs:

1. Zero the mixed part of the block values and right-hand side.
2. :func:`scatter_elastic_body_block_variable`: the body-axes block of every reduced elastic
   body, assembled by :func:`~newton._src.solvers.vbd.reduced_elastic_kernels.assemble_elastic_blocks`
   with ``joint_terms=False`` (inertia, modal mass, stiffness and damping, inertial coupling,
   contacts, external forces), rotated to world axes.
3. :func:`assemble_articulation_body_diagonal_variable`: the rigid bodies of mixed
   articulations, with the 6 x 6 kernels' body diagonal (inertia, contacts, external forces).
4. :func:`~newton._src.solvers.vbd.rigid_sparse_articulation_kernels.evaluate_articulation_joint_terms`:
   every joint of the mixed articulations once.
5. :func:`assemble_articulation_joint_terms_variable`: the joint terms projected onto both
   endpoints' rows, ``g_i += J_i^T w_i`` and ``H_ij += J_i^T K_ij J_j`` with
   ``J_e = [I_6 | G_e]`` and ``G_e`` the modal attachment twists of
   :func:`~newton._src.solvers.vbd.rigid_sparse_articulation_kernels.joint_endpoint_mode_twist`,
   gathered per output entry in joint order (no atomics).

Joint terms enter only through step 5, so neither the elastic blocks of step 2 nor the rigid
diagonals of step 3 may contain them.

Reference point of the elastic block. The local block solve applies its body-axes frame
increment ``(dx_l, dw_l)`` as ``c_new = c + R dx_l`` and ``rot_new = exp(R dw_l) rot`` and keeps
``c = transform_point(body_q, body_com)`` (see ``_apply_elastic_block_delta``); its frame force,
inertia, ``-skew(S q)`` coupling and joint moments are all taken about that same point (the
stored, pose-transformed rest COM). The variable rows use the same point and update
(``_apply_sparse_delta_value_to_body``), so the two coordinate sets differ only by axes:
``dx_w = T dx_l`` with ``T = diag(R, R, I_m)``. The world block is therefore ``T A T^T`` and the
right-hand side ``-T g`` with no point shift; for the frame rows this is the rotation the Schur
backend's ``assemble_elastic_sparse_frame_blocks`` applies.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import warp as wp

from newton._src.solvers.vbd.rigid_sparse_articulation_kernels import (
    JointEndpointTerms,
    SparseElasticJointData,
    _body_diagonal_contribution,
    evaluate_articulation_joint_terms,
    joint_endpoint_mode_count,
    joint_endpoint_mode_shape,
    joint_endpoint_mode_twist,
    vec6f,
)
from newton._src.solvers.vbd.rigid_sparse_articulation_variable_kernels import (
    vb_block_entry_index,
    vb_rhs_index,
)

wp.set_module_options({"enable_backward": False})

_STATIC_FRAME_DIAGONAL = wp.constant(1.0e30)


# ---------------------------------------------------------------------------
# Reduced elastic body blocks
# ---------------------------------------------------------------------------


@wp.func
def _elastic_block_entry(matrix: wp.array3d[float], elastic_index: int, i: int, j: int) -> float:
    """Return entry ``(i, j)`` of a symmetric body-axes elastic block from its stored upper triangle."""
    if j < i:
        return matrix[elastic_index, j, i]
    return matrix[elastic_index, i, j]


@wp.kernel
def scatter_elastic_body_block_variable(
    mixed_elastic_bodies: wp.array[wp.int32],
    mixed_elastic_body_rows: wp.array[wp.int32],
    articulation_row_dims: wp.array[wp.int32],
    articulation_row_scalar_offsets: wp.array[wp.int32],
    articulation_block_value_offsets: wp.array[wp.int32],
    articulation_block_dims: wp.array2d[wp.int32],
    articulation_diag_slots: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    elastic_body_block_grad: wp.array2d[float],
    elastic_body_block_matrix: wp.array3d[float],
    values: wp.array[float],
    rhs: wp.array[float],
):
    """Write row ``r`` of a reduced elastic body's diagonal block and right-hand side in world axes.

    Launch with ``dim = (mixed_elastic_body_count, articulation_max_row_dim)`` after
    :func:`~newton._src.solvers.vbd.reduced_elastic_kernels.assemble_elastic_blocks` (``joint_terms=False``)
    and the elastic contact assembly. The body-axes block ``A`` (stored upper triangle) and gradient
    ``g`` become ``T A T^T`` and ``-T g`` with ``T = diag(R, R, I_m)``, ``R`` the body rotation; see
    the module docstring for why no point shift is needed. The diagonal block and the row's
    right-hand side are overwritten, so this runs before any joint term is added. A static frame
    (``body_inv_mass == 0``) gets ``1e30 I_6`` on its frame block, zero frame-mode coupling and a zero
    frame right-hand side, as in the Schur backend; its modal rows are assembled normally.
    """
    entry, r = wp.tid()
    row = mixed_elastic_body_rows[entry]
    n = articulation_row_dims[row]
    if r >= n:
        return
    body = mixed_elastic_bodies[entry]
    elastic_index = body_elastic_index[body]
    slot = articulation_diag_slots[row]
    static_frame = body_inv_mass[body] == 0.0
    R = wp.quat_to_matrix(wp.transform_get_rotation(body_q[body]))

    if r < 6:
        bi = r // 3
        i = r - 3 * bi
        rhs_value = float(0.0)
        if not static_frame:
            for k in range(3):
                rhs_value = rhs_value - R[i, k] * elastic_body_block_grad[elastic_index, 3 * bi + k]
        rhs[vb_rhs_index(articulation_row_scalar_offsets, row, r)] = rhs_value
        for c in range(n):
            value = float(0.0)
            if static_frame:
                if c == r:
                    value = _STATIC_FRAME_DIAGONAL
            elif c < 6:
                bj = c // 3
                j = c - 3 * bj
                for k in range(3):
                    for kk in range(3):
                        value = (
                            value
                            + R[i, k]
                            * _elastic_block_entry(elastic_body_block_matrix, elastic_index, 3 * bi + k, 3 * bj + kk)
                            * R[j, kk]
                        )
            else:
                for k in range(3):
                    value = value + R[i, k] * _elastic_block_entry(
                        elastic_body_block_matrix, elastic_index, 3 * bi + k, c
                    )
            values[vb_block_entry_index(articulation_block_value_offsets, articulation_block_dims, slot, r, c)] = value
    else:
        rhs[vb_rhs_index(articulation_row_scalar_offsets, row, r)] = -elastic_body_block_grad[elastic_index, r]
        for c in range(n):
            value = float(0.0)
            if c < 6:
                if not static_frame:
                    bj = c // 3
                    j = c - 3 * bj
                    for kk in range(3):
                        value = (
                            value
                            + _elastic_block_entry(elastic_body_block_matrix, elastic_index, r, 3 * bj + kk) * R[j, kk]
                        )
            else:
                value = _elastic_block_entry(elastic_body_block_matrix, elastic_index, r, c)
            values[vb_block_entry_index(articulation_block_value_offsets, articulation_block_dims, slot, r, c)] = value


# ---------------------------------------------------------------------------
# Rigid bodies of mixed articulations
# ---------------------------------------------------------------------------


@wp.kernel
def assemble_articulation_body_diagonal_variable(
    dt: float,
    rigid_body_row_count: int,
    articulation_bodies: wp.array[wp.int32],
    articulation_diag_slots: wp.array[wp.int32],
    articulation_row_scalar_offsets: wp.array[wp.int32],
    articulation_block_value_offsets: wp.array[wp.int32],
    articulation_block_dims: wp.array2d[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    body_inertia_q: wp.array[wp.transform],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
    use_displacement: int,
    values: wp.array[float],
    rhs: wp.array[float],
):
    """Write the 6 x 6 diagonal block and right-hand side of each rigid body row of the mixed articulations.

    Launch with ``dim = articulation_body_count - rigid_body_row_count``; thread ``t`` handles row
    ``rigid_body_row_count + t`` and skips reduced elastic bodies (see
    :func:`scatter_elastic_body_block_variable`). The block is the one the 6 x 6 kernels assemble
    (``_body_diagonal_contribution``: inertia, contact and external Hessians and forces, the small
    angular regularization, ``1e30 I_6`` for static bodies), written at the variable offsets.
    """
    row = rigid_body_row_count + wp.tid()
    body = articulation_bodies[row]
    if body_elastic_index.shape[0] > 0:
        if body_elastic_index[body] >= 0:
            return
    diag, rhs_value = _body_diagonal_contribution(
        dt,
        row,
        articulation_bodies,
        body_q,
        body_mass,
        body_inv_mass,
        body_com,
        body_inertia,
        body_inertia_q,
        body_forces,
        body_torques,
        body_hessian_ll,
        body_hessian_al,
        body_hessian_aa,
        body_disp,
        body_disp_ang,
        use_displacement,
    )
    slot = articulation_diag_slots[row]
    for i in range(6):
        rhs[vb_rhs_index(articulation_row_scalar_offsets, row, i)] = rhs_value[i]
        for j in range(6):
            values[vb_block_entry_index(articulation_block_value_offsets, articulation_block_dims, slot, i, j)] = diag[
                i, j
            ]


# ---------------------------------------------------------------------------
# Joint terms
# ---------------------------------------------------------------------------


@wp.func
def _endpoint_twist(
    elastic: SparseElasticJointData,
    joint: int,
    is_parent_side: bool,
    rotation: wp.mat33,
    rotation_jacobian_world: wp.mat33,
    attachment_offset: wp.vec3,
    mode: int,
) -> vec6f:
    phi, psi = joint_endpoint_mode_shape(elastic, joint, is_parent_side, mode)
    return joint_endpoint_mode_twist(rotation, rotation_jacobian_world, attachment_offset, phi, psi)


@wp.func
def _joint_self_entry(
    elastic: SparseElasticJointData,
    joint: int,
    t: JointEndpointTerms,
    is_parent: bool,
    row_dim: int,
    body_q: wp.array[wp.transform],
    a: int,
    b: int,
) -> float:
    """Entry ``(a, b)`` of ``J_e^T K_ee J_e`` for one endpoint of an evaluated joint (0 past the endpoint's rows).

    ``row_dim`` is the size of the endpoint body's row; the endpoint projects onto its first
    ``6 + m_e`` components. The expressions are ``K``, ``(K g_b)[a]``, ``(K^T g_a)[b]`` and
    ``g_a . (K g_b)`` with ``g`` the mode twists of :func:`_endpoint_twist`.
    """
    body = t.child
    K = t.hessian_child
    offset = t.attachment_offset_child
    jacobian = t.rotation_jacobian_child
    if is_parent:
        body = t.parent
        K = t.hessian_parent
        offset = t.attachment_offset_parent
        jacobian = t.rotation_jacobian_parent
    n = 6 + wp.min(joint_endpoint_mode_count(elastic, joint, body, is_parent), row_dim - 6)
    if a >= n or b >= n:
        return 0.0
    if a < 6 and b < 6:
        return K[a, b]
    R = wp.quat_to_matrix(wp.transform_get_rotation(body_q[body]))
    RJ = R * jacobian
    if a < 6:
        g_col = _endpoint_twist(elastic, joint, is_parent, R, RJ, offset, b - 6)
        Kg = K * g_col
        return Kg[a]
    if b < 6:
        g_row = _endpoint_twist(elastic, joint, is_parent, R, RJ, offset, a - 6)
        gK = wp.transpose(K) * g_row
        return gK[b]
    g_col = _endpoint_twist(elastic, joint, is_parent, R, RJ, offset, b - 6)
    Kg = K * g_col
    g_row = _endpoint_twist(elastic, joint, is_parent, R, RJ, offset, a - 6)
    return wp.dot(g_row, Kg)


@wp.func
def _joint_self_rhs(
    elastic: SparseElasticJointData,
    joint: int,
    t: JointEndpointTerms,
    is_parent: bool,
    row_dim: int,
    body_q: wp.array[wp.transform],
    k: int,
) -> float:
    """Component ``k`` of ``J_e^T w_e`` for one endpoint of an evaluated joint (0 past the endpoint's rows)."""
    body = t.child
    wrench = t.wrench_child
    offset = t.attachment_offset_child
    jacobian = t.rotation_jacobian_child
    if is_parent:
        body = t.parent
        wrench = t.wrench_parent
        offset = t.attachment_offset_parent
        jacobian = t.rotation_jacobian_parent
    n = 6 + wp.min(joint_endpoint_mode_count(elastic, joint, body, is_parent), row_dim - 6)
    if k >= n:
        return 0.0
    if k < 6:
        return wrench[k]
    R = wp.quat_to_matrix(wp.transform_get_rotation(body_q[body]))
    RJ = R * jacobian
    g = _endpoint_twist(elastic, joint, is_parent, R, RJ, offset, k - 6)
    return wp.dot(g, wrench)


@wp.func
def _joint_cross_entry(
    elastic: SparseElasticJointData,
    joint: int,
    t: JointEndpointTerms,
    parent_dim: int,
    child_dim: int,
    body_q: wp.array[wp.transform],
    a: int,
    b: int,
) -> float:
    """Entry ``(a, b)`` (parent row ``a``, child column ``b``) of ``J_p^T H_pc J_c`` of an evaluated joint.

    0 past the endpoints' projected rows. The expressions are ``H``, ``(H g_c)[a]``,
    ``(H^T g_p)[b]`` and ``(H^T g_p) . g_c``.
    """
    parent_n = 6 + wp.min(joint_endpoint_mode_count(elastic, joint, t.parent, True), parent_dim - 6)
    child_n = 6 + wp.min(joint_endpoint_mode_count(elastic, joint, t.child, False), child_dim - 6)
    if a >= parent_n or b >= child_n:
        return 0.0
    H = t.hessian_cross
    if a < 6 and b < 6:
        return H[a, b]
    if a < 6:
        R_c = wp.quat_to_matrix(wp.transform_get_rotation(body_q[t.child]))
        RJ_c = R_c * t.rotation_jacobian_child
        g_col = _endpoint_twist(elastic, joint, False, R_c, RJ_c, t.attachment_offset_child, b - 6)
        Hg = H * g_col
        return Hg[a]
    R_p = wp.quat_to_matrix(wp.transform_get_rotation(body_q[t.parent]))
    RJ_p = R_p * t.rotation_jacobian_parent
    g_row = _endpoint_twist(elastic, joint, True, R_p, RJ_p, t.attachment_offset_parent, a - 6)
    gH = wp.transpose(H) * g_row
    if b < 6:
        return gH[b]
    R_c = wp.quat_to_matrix(wp.transform_get_rotation(body_q[t.child]))
    RJ_c = R_c * t.rotation_jacobian_child
    g_col = _endpoint_twist(elastic, joint, False, R_c, RJ_c, t.attachment_offset_child, b - 6)
    return wp.dot(gH, g_col)


@wp.func
def _last_le(offsets: wp.array[wp.int32], begin: int, end: int, value: int) -> int:
    """Largest ``i`` in ``[begin, end)`` with ``offsets[i] <= value`` (``offsets`` non-decreasing, ``offsets[begin] <= value``)."""
    lo = begin
    hi = end - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if offsets[mid] <= value:
            lo = mid
        else:
            hi = mid - 1
    return lo


def joint_term_entry_count(max_row_dim: int) -> int:
    """Size of the second launch dimension of :func:`assemble_articulation_joint_terms_variable`.

    A diagonal slot of row size ``n`` has ``n * n`` block entries followed by ``n`` right-hand side
    components; an off-diagonal slot has at most ``n * n`` entries.
    """
    return max_row_dim * (max_row_dim + 1)


@wp.kernel
def assemble_articulation_joint_terms_variable(
    rigid_articulation_count: int,
    rigid_joint_count: int,
    rigid_body_row_count: int,
    rigid_block_count: int,
    articulation_body_offsets: wp.array[wp.int32],
    articulation_joint_offsets: wp.array[wp.int32],
    mixed_articulation_joints: wp.array[wp.int32],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    articulation_row_dims: wp.array[wp.int32],
    articulation_row_scalar_offsets: wp.array[wp.int32],
    articulation_block_value_offsets: wp.array[wp.int32],
    articulation_block_dims: wp.array2d[wp.int32],
    body_articulation_local: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    elastic: SparseElasticJointData,
    terms: wp.array[JointEndpointTerms],
    values: wp.array[float],
    rhs: wp.array[float],
):
    """Add the projected joint terms to one entry of a mixed-articulation block slot or right-hand side.

    Launched by :func:`launch_articulation_joint_terms_variable` with
    ``dim = (block_count - rigid_block_count, joint_term_entry_count(articulation_max_row_dim))``
    after :func:`~newton._src.solvers.vbd.rigid_sparse_articulation_kernels.evaluate_articulation_joint_terms`
    on ``mixed_articulation_joints`` (``terms[j]`` belongs to ``mixed_articulation_joints[j]``).
    Thread ``(s, e)`` owns entry ``e`` of slot ``rigid_block_count + s`` (row-major), and entry
    ``R * C + k`` of a diagonal slot is component ``k`` of its row's right-hand side; threads
    past the slot's size return.

    With ``J_e = [I_6 | G_e]`` (``G_e`` empty for a rigid endpoint or an attachment without a modal
    sample) each evaluated joint of the slot's articulation adds, for an endpoint whose body owns
    the slot's row, its self block ``J_e^T K_ee J_e`` and right-hand side ``J_e^T w_e`` to the
    diagonal slot, and for a slot coupling its parent and child rows the cross block
    ``J_p^T H_pc J_c`` (or its transpose when the parent row comes first in the articulation's
    order, since only the lower block triangle is stored). A parent that is not a body of the
    articulation acts like the world. Each thread reads the value written by the body-block
    assembly and adds the joints' contributions in the articulation's joint order, so the result
    is deterministic and no atomics are used; every contribution is formed by the same
    expression for each entry (see :func:`_joint_self_entry`, :func:`_joint_self_rhs`,
    :func:`_joint_cross_entry`). Joints with ``evaluated == 0`` add nothing.
    """
    s, e = wp.tid()
    slot = rigid_block_count + s
    R = articulation_block_dims[slot, 0]
    C = articulation_block_dims[slot, 1]
    if e >= R * C + R:
        return
    row = _last_le(
        articulation_block_row_offsets, rigid_body_row_count, articulation_block_row_offsets.shape[0] - 1, slot
    )
    articulation = _last_le(
        articulation_body_offsets, rigid_articulation_count, articulation_body_offsets.shape[0] - 1, row
    )
    body_start = articulation_body_offsets[articulation]
    row_local = row - body_start
    col_local = articulation_block_cols[slot]
    diagonal = col_local == row_local
    if e >= R * C and not diagonal:
        return
    joint_begin = articulation_joint_offsets[articulation] - rigid_joint_count
    joint_end = articulation_joint_offsets[articulation + 1] - rigid_joint_count
    row_dim = articulation_row_dims[row]

    if e >= R * C:
        k = e - R * C
        index = vb_rhs_index(articulation_row_scalar_offsets, row, k)
        value = rhs[index]
        for j in range(joint_begin, joint_end):
            t = terms[j]
            if t.evaluated != 0:
                joint = mixed_articulation_joints[j]
                if t.parent >= 0:
                    if body_articulation_local[t.parent] == row_local:
                        value += _joint_self_rhs(elastic, joint, t, True, row_dim, body_q, k)
                if body_articulation_local[t.child] == row_local:
                    value += _joint_self_rhs(elastic, joint, t, False, row_dim, body_q, k)
        rhs[index] = value
        return

    a = e // C
    b = e - a * C
    index = articulation_block_value_offsets[slot] + e
    value = values[index]
    col_dim = articulation_row_dims[body_start + col_local]
    for j in range(joint_begin, joint_end):
        t = terms[j]
        if t.evaluated != 0:
            joint = mixed_articulation_joints[j]
            parent_local = int(-1)
            if t.parent >= 0:
                parent_local = body_articulation_local[t.parent]
            child_local = body_articulation_local[t.child]
            if diagonal:
                if parent_local == row_local:
                    value += _joint_self_entry(elastic, joint, t, True, row_dim, body_q, a, b)
                if child_local == row_local:
                    value += _joint_self_entry(elastic, joint, t, False, row_dim, body_q, a, b)
            elif parent_local >= 0:
                if parent_local == row_local and child_local == col_local:
                    value += _joint_cross_entry(elastic, joint, t, row_dim, col_dim, body_q, a, b)
                elif child_local == row_local and parent_local == col_local:
                    value += _joint_cross_entry(elastic, joint, t, col_dim, row_dim, body_q, b, a)
    values[index] = value


def launch_articulation_joint_terms_variable(
    layout,
    body_q: wp.array,
    elastic: SparseElasticJointData,
    terms: wp.array,
    values: wp.array,
    rhs: wp.array,
    device: wp.context.Devicelike = None,
) -> None:
    """Launch :func:`assemble_articulation_joint_terms_variable` over every block slot of the mixed articulations.

    ``terms`` must hold the evaluated terms of ``layout.mixed_articulation_joints``. Launch only;
    no-op without mixed articulations.
    """
    slot_count = layout.block_count - layout.rigid_block_count
    if layout.mixed_articulation_count == 0 or slot_count == 0:
        return
    wp.launch(
        assemble_articulation_joint_terms_variable,
        dim=(slot_count, joint_term_entry_count(layout.articulation_max_row_dim)),
        inputs=[
            layout.rigid_articulation_count,
            layout.rigid_joint_count,
            layout.rigid_body_row_count,
            layout.rigid_block_count,
            layout.articulation_body_offsets,
            layout.articulation_joint_offsets,
            layout.mixed_articulation_joints,
            layout.articulation_block_row_offsets,
            layout.articulation_block_cols,
            layout.articulation_row_dims,
            layout.articulation_row_scalar_offsets,
            layout.articulation_block_value_offsets,
            layout.articulation_block_dims,
            layout.body_articulation_local,
            body_q,
            elastic,
            terms,
        ],
        outputs=[values, rhs],
        device=device if device is not None else values.device,
    )


# ---------------------------------------------------------------------------
# Launch helper
# ---------------------------------------------------------------------------


def kernel_argument_labels(kernel) -> list[str]:
    """Return the argument names of a Warp ``kernel`` in signature order.

    Reads ``kernel.adj.args`` (Warp internal, verified with Warp 1.18.0) and falls back to the
    signature of the kernel's Python function (``kernel.func``).

    Raises:
        RuntimeError: If neither is available.
    """
    try:
        return [arg.label for arg in kernel.adj.args]
    except AttributeError:
        pass
    try:
        return list(inspect.signature(kernel.func).parameters)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(f"Cannot read the argument names of Warp kernel {kernel!r}.") from error


_JOINT_INPUTS_FROM_HELPER = ("dt", "joints", "body_articulation_local", "body_q", "body_com", "elastic", "terms")
JOINT_TERM_INPUT_NAMES = tuple(
    label
    for label in kernel_argument_labels(evaluate_articulation_joint_terms)
    if label not in _JOINT_INPUTS_FROM_HELPER
)
"""Keys of the ``joint_inputs`` mapping of :func:`assemble_articulation_variable_system`.

They are the arguments of :func:`~newton._src.solvers.vbd.rigid_sparse_articulation_kernels.evaluate_articulation_joint_terms`
that the helper does not derive from the layout or its other arguments; they mean the same as
for :func:`~newton._src.solvers.vbd.rigid_sparse_articulation_kernels.assemble_articulation_joints_scalar`.
"""


def assemble_articulation_variable_system(
    layout,
    model,
    *,
    dt: float,
    body_q: wp.array,
    body_inv_mass: wp.array,
    body_inertia_q: wp.array,
    body_forces: wp.array,
    body_torques: wp.array,
    body_hessian_ll: wp.array,
    body_hessian_al: wp.array,
    body_hessian_aa: wp.array,
    body_disp: wp.array,
    body_disp_ang: wp.array,
    use_displacement: int,
    elastic_body_block_grad: wp.array,
    elastic_body_block_matrix: wp.array,
    elastic: SparseElasticJointData,
    joint_inputs: Mapping[str, Any],
    terms: wp.array,
    values: wp.array,
    rhs: wp.array,
    device: wp.context.Devicelike = None,
) -> tuple[wp.array, wp.array]:
    """Assemble the variable-size block system of every mixed articulation of ``layout``.

    Zeroes the mixed part of ``values`` and ``rhs`` (the rigid prefix
    ``[0, 36 * rigid_block_count)`` and ``[0, 6 * rigid_body_row_count)`` is not touched), then
    writes the elastic body blocks, the rigid body diagonals, evaluates every joint of
    ``layout.mixed_articulation_joints`` once into ``terms`` and adds the projected joint terms.
    The result is ready for
    :func:`~newton._src.solvers.vbd.rigid_sparse_articulation_variable_kernels.launch_solve_articulation_variable`,
    which adds the frame regularization. Launches only, no allocation or host synchronization, so
    the call can be captured in a CUDA graph. No-op without mixed articulations.

    Args:
        layout: Layout built with ``elastic_solve="coupled"``.
        model: The model; ``body_mass``, ``body_com``, ``body_inertia`` and ``body_elastic_index``
            are read from it.
        dt: Time step [s].
        body_q: Body poses at the iterate, shape [body_count].
        body_inv_mass: Effective inverse masses (0 for static or kinematic frames), shape [body_count].
        body_inertia_q: Inertial target poses, shape [body_count].
        body_forces: Contact and external forces on rigid bodies [N], shape [body_count].
        body_torques: Contact and external torques about the COM [N m], shape [body_count].
        body_hessian_ll: Contact and external Hessian blocks (world axes), shape [body_count].
        body_hessian_al: Same, angular-linear block, shape [body_count].
        body_hessian_aa: Same, angular block, shape [body_count].
        body_disp: Accumulated COM displacement from the inertial target [m], shape [body_count].
        body_disp_ang: Accumulated rotation vector from the inertial target [rad], shape [body_count].
        use_displacement: Nonzero for the displacement-form inertial terms of the 6 x 6 kernels.
        elastic_body_block_grad: Body-axes gradient of every reduced elastic body, shape
            [elastic_body_count, width], assembled at this iterate by ``assemble_elastic_blocks``
            with ``solve_frame=True`` and ``joint_terms=False`` (plus the elastic contacts).
        elastic_body_block_matrix: Body-axes block (upper triangle) matching
            ``elastic_body_block_grad``, shape [elastic_body_count, width, width].
        elastic: Elastic endpoint data with ``enabled != 0`` and the current and previous modal
            coordinates.
        joint_inputs: Every argument of ``evaluate_articulation_joint_terms`` named in
            :data:`JOINT_TERM_INPUT_NAMES`; the helper supplies ``dt``, ``joints``,
            ``body_articulation_local``, ``body_q``, ``body_com``, ``elastic`` and ``terms``.
        terms: Scratch, shape [len(layout.mixed_articulation_joints)], dtype
            :class:`~newton._src.solvers.vbd.rigid_sparse_articulation_kernels.JointEndpointTerms`.
        values: Block values, shape [layout.articulation_value_count].
        rhs: Right-hand side, shape [layout.articulation_scalar_count].
        device: Launch device; defaults to the device of ``values``.

    Returns:
        ``(values, rhs)``.

    Raises:
        ValueError: If ``layout`` is not a coupled layout or ``joint_inputs`` has missing or unknown keys.
    """
    if layout.elastic_solve != "coupled":
        raise ValueError(
            f"assemble_articulation_variable_system needs a 'coupled' layout, got {layout.elastic_solve!r}."
        )
    missing = [name for name in JOINT_TERM_INPUT_NAMES if name not in joint_inputs]
    unknown = [name for name in joint_inputs if name not in JOINT_TERM_INPUT_NAMES]
    if missing or unknown:
        raise ValueError(f"joint_inputs: missing {missing}, unknown {unknown}.")
    if layout.mixed_articulation_count == 0:
        return values, rhs
    device = device if device is not None else values.device

    # Rows and slots of the rigid prefix are all 6 x 6, so the mixed part starts at these offsets.
    values[36 * layout.rigid_block_count :].zero_()
    rhs[6 * layout.rigid_body_row_count :].zero_()

    if layout.mixed_elastic_body_count > 0:
        wp.launch(
            scatter_elastic_body_block_variable,
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
                body_q,
                body_inv_mass,
                elastic_body_block_grad,
                elastic_body_block_matrix,
            ],
            outputs=[values, rhs],
            device=device,
        )

    wp.launch(
        assemble_articulation_body_diagonal_variable,
        dim=layout.articulation_body_count - layout.rigid_body_row_count,
        inputs=[
            dt,
            layout.rigid_body_row_count,
            layout.articulation_bodies,
            layout.articulation_diag_slots,
            layout.articulation_row_scalar_offsets,
            layout.articulation_block_value_offsets,
            layout.articulation_block_dims,
            model.body_elastic_index,
            body_q,
            model.body_mass,
            body_inv_mass,
            model.body_com,
            model.body_inertia,
            body_inertia_q,
            body_forces,
            body_torques,
            body_hessian_ll,
            body_hessian_al,
            body_hessian_aa,
            body_disp,
            body_disp_ang,
            use_displacement,
        ],
        outputs=[values, rhs],
        device=device,
    )

    joint_count = layout.mixed_articulation_joints.shape[0]
    if joint_count == 0:
        return values, rhs
    supplied = {
        "dt": dt,
        "joints": layout.mixed_articulation_joints,
        "body_articulation_local": layout.body_articulation_local,
        "body_q": body_q,
        "body_com": model.body_com,
        "elastic": elastic,
    }
    evaluator_inputs = [
        supplied[arg.label] if arg.label in supplied else joint_inputs[arg.label]
        for arg in evaluate_articulation_joint_terms.adj.args
        if arg.label != "terms"
    ]
    wp.launch(
        evaluate_articulation_joint_terms,
        dim=joint_count,
        inputs=evaluator_inputs,
        outputs=[terms],
        device=device,
    )
    launch_articulation_joint_terms_variable(layout, body_q, elastic, terms, values, rhs, device)
    return values, rhs
