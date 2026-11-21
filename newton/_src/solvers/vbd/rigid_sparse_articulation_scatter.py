# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Striped joint assembly for small rigid BDF1 articulations."""

import warp as wp

from newton._src.sim.enums import (
    JointType,
)
from newton._src.sim.joint_coordinates import (
    eval_joint_coordinate_in_frames,
    unwrap_joint_coordinate,
)
from newton._src.solvers.rod_kernels import (
    assemble_geometric_rod_kappa_z,
    measure_rod_bend_twist_z,
    quat_rotate_local_z,
    rod_bend_twist_delta,
    rod_bend_twist_jacobian_z_from_measure,
)
from newton._src.solvers.vbd.joint_coordinates import (
    JointCoordinateData,
)
from newton._src.solvers.vbd.rigid_sparse_articulation_kernels import (
    _anchor_offset,
    _angular_constraint_force_hessian,
    _evaluate_sparse_drive_limit,
    _joint_angular_jacobian_value,
    _joint_linear_jacobian_value,
    _linear_joint_residual,
    _scalar_block_index,
    _scalar_vec_index,
    _sparse_joint_geometry,
    _sparse_joint_supported,
    _sparse_linear_drive_limit_row,
    _vec6_from_parts,
    mat66f,
    vec6f,
)
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _diag_mul_mat33,
    _drive_row_applies_force,
    _eval_joint_axis_friction,
    _limit_row_exists,
    _load_solve_weight,
    _material_force_terms,
    _structural_row_has_stiffness,
    evaluate_rod_stretch_shear_force_hessian,
)

wp.set_module_options({"enable_backward": False})


@wp.func
def _add_rhs_scalar_atomic(rhs_scalar: wp.array[float], index: int, value: vec6f, stripe: int, lane_mask: int):
    for i in range(6):
        if i & lane_mask == stripe:
            wp.atomic_add(rhs_scalar, _scalar_vec_index(index, i), value[i])


@wp.func
def _find_block_slot(
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    row_local: int,
    col_local: int,
) -> int:
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
def _rigid_add_rhs3(rhs: wp.array[float], row: int, offset: int, value: wp.vec3, stripe: int, lane_mask: int):
    base = row * 6 + offset
    for component in range(3):
        if offset + component & lane_mask == stripe and value[component] != 0.0:
            wp.atomic_add(rhs, base + component, value[component])


@wp.func
def _rigid_add_tile(
    values: wp.array[float], slot: int, row0: int, col0: int, tile: wp.mat33, flip: bool, stripe: int, lane_mask: int
):
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
                if (
                    (col0 + col) * 6 + row0 + row if flip else (row0 + row) * 6 + col0 + col
                ) & lane_mask == stripe and value != 0.0:
                    wp.atomic_add(values, base + row * row_stride + col * col_stride, value)


@wp.func
def _assemble_angular_direct_pair_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    torque_parent: wp.vec3,
    H_aa: wp.mat33,
    cross_slot: int,
    stripe: int,
    lane_mask: int,
):
    has_parent = parent_body >= 0 and parent_local >= 0
    if has_parent:
        _rigid_add_rhs3(rhs_scalar, body_start + parent_local, 3, torque_parent, stripe, lane_mask)
        slot_pp = _find_block_slot(
            articulation_block_row_offsets, articulation_block_cols, body_start, parent_local, parent_local
        )
        _rigid_add_tile(values_scalar, slot_pp, 3, 3, H_aa, False, stripe, lane_mask)
    _rigid_add_rhs3(rhs_scalar, body_start + child_local, 3, -torque_parent, stripe, lane_mask)
    slot_cc = _find_block_slot(
        articulation_block_row_offsets, articulation_block_cols, body_start, child_local, child_local
    )
    _rigid_add_tile(values_scalar, slot_cc, 3, 3, H_aa, False, stripe, lane_mask)
    if has_parent:
        slot_pc = _find_block_slot(
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            wp.max(parent_local, child_local),
            wp.min(parent_local, child_local),
        )
        _rigid_add_tile(values_scalar, slot_pc, 3, 3, -H_aa, False, stripe, lane_mask)
        _rigid_add_tile(values_scalar, cross_slot, 3, 3, -H_aa, False, stripe, lane_mask)


@wp.func
def _assemble_angular_axis_row_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    angular_jacobian_world: wp.vec3,
    force_scalar: float,
    hessian_scalar: float,
    cross_slot: int,
    stripe: int,
    lane_mask: int,
):
    torque_parent = force_scalar * angular_jacobian_world
    H_aa = hessian_scalar * wp.outer(angular_jacobian_world, angular_jacobian_world)
    _assemble_angular_direct_pair_scalar(
        values_scalar,
        rhs_scalar,
        articulation_block_row_offsets,
        articulation_block_cols,
        body_start,
        parent_local,
        child_local,
        parent_body,
        torque_parent,
        H_aa,
        cross_slot,
        stripe,
        lane_mask,
    )


@wp.func
def _assemble_angular_joint_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    parent_anchor_q: wp.quat,
    child_anchor_q: wp.quat,
    parent_anchor_q_prev: wp.quat,
    child_anchor_q_prev: wp.quat,
    parent_rest_q: wp.quat,
    child_rest_q: wp.quat,
    stiffness: float,
    damping: float,
    dt: float,
    P: wp.mat33,
    sigma0: wp.vec3,
    C_fric: wp.vec3,
    lambda_ang: wp.vec3,
    C0_ang: wp.vec3,
    alpha: float,
    cross_slot: int,
    joint_axis: wp.array[wp.vec3],
    ang_start: int,
    ang_count: int,
    stripe: int,
    lane_mask: int,
):
    torque_parent, H_aa, kappa, J_world = _angular_constraint_force_hessian(
        parent_anchor_q,
        child_anchor_q,
        parent_anchor_q_prev,
        child_anchor_q_prev,
        parent_rest_q,
        child_rest_q,
        stiffness,
        P,
        sigma0,
        C_fric,
        lambda_ang,
        C0_ang,
        alpha,
        damping,
        dt,
        joint_axis,
        ang_start,
        ang_count,
    )
    _assemble_angular_direct_pair_scalar(
        values_scalar,
        rhs_scalar,
        articulation_block_row_offsets,
        articulation_block_cols,
        body_start,
        parent_local,
        child_local,
        parent_body,
        torque_parent,
        H_aa,
        cross_slot,
        stripe,
        lane_mask,
    )
    return (kappa, J_world)


@wp.func
def _add_coordinate_outer_scalar(
    values: wp.array[float],
    slot: int,
    left: vec6f,
    right: vec6f,
    hessian: float,
    transposed: bool,
    stripe: int,
    lane_mask: int,
):
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
def _assemble_coordinate_row_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    row_offsets: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    gradient_parent: wp.spatial_vector,
    gradient_child: wp.spatial_vector,
    force: float,
    hessian: float,
    cross_slot: int,
    stripe: int,
    lane_mask: int,
):
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
def _jac_column(projector: wp.mat33, arm: wp.vec3, parent_side: bool, angular: bool, dof: int) -> wp.vec3:
    column = wp.vec3(0.0)
    for row in range(3):
        if angular:
            column[row] = _joint_angular_jacobian_value(projector, parent_side, row, dof)
        else:
            column[row] = _joint_linear_jacobian_value(projector, arm, parent_side, row, dof)
    return column


@wp.func
def _column_gram(left: wp.vec3, scale: float, right: wp.vec3) -> float:
    result = float(0.0)
    for component in range(3):
        result = result + scale * left[component] * right[component]
    return result


@wp.func
def _column_rhs(gradient: wp.vec3, residual: wp.vec3, scale: float) -> float:
    result = float(0.0)
    for component in range(3):
        result = result + gradient[component] * residual[component]
    return -scale * result


@wp.func
def _write_nonzero(dest: wp.array[float], index: int, value: float):
    if value != 0.0:
        wp.atomic_add(dest, index, value)


@wp.func
def _assemble_constraint_pair_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    local_a: int,
    local_b: int,
    residual: wp.vec3,
    force_scale: float,
    hessian_scale: float,
    P: wp.mat33,
    r_a: wp.vec3,
    r_b: wp.vec3,
    is_parent_a: bool,
    is_parent_b: bool,
    angular_only: bool,
    cross_slot: int,
    stripe: int,
    lane_mask: int,
):
    if force_scale == 0.0 and hessian_scale <= 0.0:
        return
    width = lane_mask + 1
    if force_scale != 0.0:
        for dof in range(stripe, 6, width):
            grad_a = _jac_column(P, r_a, is_parent_a, angular_only, dof)
            grad_b = _jac_column(P, r_b, is_parent_b, angular_only, dof)
            _write_nonzero(rhs_scalar, (body_start + local_a) * 6 + dof, _column_rhs(grad_a, residual, force_scale))
            _write_nonzero(rhs_scalar, (body_start + local_b) * 6 + dof, _column_rhs(grad_b, residual, force_scale))
    if hessian_scale == 0.0:
        return
    slot_a = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, local_a, local_a)
    slot_b = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, local_b, local_b)
    slot_cross = _find_block_slot(
        articulation_block_row_offsets,
        articulation_block_cols,
        body_start,
        wp.max(local_a, local_b),
        wp.min(local_a, local_b),
    )
    for entry in range(stripe, 36, width):
        row = entry // 6
        col = entry % 6
        left_a = _jac_column(P, r_a, is_parent_a, angular_only, row)
        right_b = _jac_column(P, r_b, is_parent_b, angular_only, col)
        if slot_a >= 0:
            right_a = _jac_column(P, r_a, is_parent_a, angular_only, col)
            _write_nonzero(values_scalar, slot_a * 36 + entry, _column_gram(left_a, hessian_scale, right_a))
        if slot_b >= 0:
            left_b = _jac_column(P, r_b, is_parent_b, angular_only, row)
            _write_nonzero(values_scalar, slot_b * 36 + entry, _column_gram(left_b, hessian_scale, right_b))
        if slot_cross >= 0 or cross_slot >= 0:
            cross = _column_gram(left_a, hessian_scale, right_b)
            if slot_cross >= 0:
                dest = col * 6 + row if local_a < local_b else entry
                _write_nonzero(values_scalar, slot_cross * 36 + dest, cross)
            if cross_slot >= 0:
                _write_nonzero(values_scalar, cross_slot * 36 + entry, cross)


@wp.func
def _assemble_constraint_single_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    local_body: int,
    residual: wp.vec3,
    force_scale: float,
    hessian_scale: float,
    P: wp.mat33,
    r: wp.vec3,
    is_parent: bool,
    angular_only: bool,
    stripe: int,
    lane_mask: int,
):
    if force_scale == 0.0 and hessian_scale <= 0.0:
        return
    width = lane_mask + 1
    if force_scale != 0.0:
        for dof in range(stripe, 6, width):
            grad = _jac_column(P, r, is_parent, angular_only, dof)
            _write_nonzero(rhs_scalar, (body_start + local_body) * 6 + dof, _column_rhs(grad, residual, force_scale))
    if hessian_scale == 0.0:
        return
    slot = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, body_start, local_body, local_body)
    if slot >= 0:
        for entry in range(stripe, 36, width):
            row = entry // 6
            col = entry % 6
            left = _jac_column(P, r, is_parent, angular_only, row)
            right = _jac_column(P, r, is_parent, angular_only, col)
            _write_nonzero(values_scalar, slot * 36 + entry, _column_gram(left, hessian_scale, right))


@wp.func
def _assemble_linear_axis_row_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    child_body: int,
    parent_anchor: wp.vec3,
    child_anchor: wp.vec3,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    axis_world: wp.vec3,
    force_scalar: float,
    hessian_scalar: float,
    cross_slot: int,
    stripe: int,
    lane_mask: int,
):
    P = wp.outer(axis_world, axis_world)
    residual = axis_world
    r_child = _anchor_offset(body_q, body_com, child_body, child_anchor)
    if parent_body >= 0 and parent_local >= 0:
        r_parent = _anchor_offset(body_q, body_com, parent_body, parent_anchor)
        _assemble_constraint_pair_scalar(
            values_scalar,
            rhs_scalar,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            parent_local,
            child_local,
            residual,
            force_scalar,
            hessian_scalar,
            P,
            r_parent,
            r_child,
            True,
            False,
            False,
            cross_slot,
            stripe,
            lane_mask,
        )
    else:
        _assemble_constraint_single_scalar(
            values_scalar,
            rhs_scalar,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            child_local,
            residual,
            force_scalar,
            hessian_scalar,
            P,
            r_child,
            False,
            False,
            stripe,
            lane_mask,
        )


@wp.func
def _assemble_linear_joint_scalar(
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_start: int,
    parent_local: int,
    child_local: int,
    parent_body: int,
    child_body: int,
    parent_anchor: wp.vec3,
    child_anchor: wp.vec3,
    parent_anchor_prev: wp.vec3,
    child_anchor_prev: wp.vec3,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    stiffness: float,
    damping: float,
    dt: float,
    P: wp.mat33,
    lambda_lin: wp.vec3,
    C0_lin: wp.vec3,
    alpha: float,
    cross_slot: int,
    stripe: int,
    lane_mask: int,
):
    force_residual, hessian_scale = _linear_joint_residual(
        parent_anchor,
        child_anchor,
        parent_anchor_prev,
        child_anchor_prev,
        stiffness,
        damping,
        dt,
        P,
        lambda_lin,
        C0_lin,
        alpha,
    )
    r_child = _anchor_offset(body_q, body_com, child_body, child_anchor)
    if parent_body >= 0 and parent_local >= 0:
        r_parent = _anchor_offset(body_q, body_com, parent_body, parent_anchor)
        _assemble_constraint_pair_scalar(
            values_scalar,
            rhs_scalar,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            parent_local,
            child_local,
            force_residual,
            1.0,
            hessian_scale,
            P,
            r_parent,
            r_child,
            True,
            False,
            False,
            cross_slot,
            stripe,
            lane_mask,
        )
    else:
        _assemble_constraint_single_scalar(
            values_scalar,
            rhs_scalar,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            child_local,
            force_residual,
            1.0,
            hessian_scale,
            P,
            r_child,
            False,
            False,
            stripe,
            lane_mask,
        )


@wp.func
def _sparse_cross_element(
    row: int, col: int, material: wp.mat33, iso: float, child: wp.vec3, anchor: wp.vec3, parent: wp.vec3
) -> float:
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
def _rod_cross_hessian(
    dt: float,
    joint: int,
    parent: int,
    child: int,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    X_pj: wp.transform,
    X_cj: wp.transform,
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
    joint_rho: wp.array[float],
    joint_material_k: wp.array[float],
    joint_penalty_kd: wp.array[float],
    joint_C_fric: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    joint_compliant_alm: int,
    stripe: int,
    lane_mask: int,
):
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
    k_stretch, _ = _material_force_terms(
        stretch_weight, joint_material_k[stretch_idx], wp.vec3(0.0), joint_compliant_alm
    )
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
        for i in range(6):
            for j in range(6):
                if i * 6 + j & lane_mask == stripe:
                    cross[i, j] = _sparse_cross_element(
                        i, j, K_material, k_iso, r_child, r_parent_anchor, r_parent_material
                    )
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
def _scatter_endpoint(
    values: wp.array[float],
    rhs: wp.array[float],
    row: int,
    slot: int,
    force: wp.vec3,
    torque: wp.vec3,
    ll: wp.mat33,
    al: wp.mat33,
    aa: wp.mat33,
    stripe: int,
    mask: int,
):
    _add_rhs_scalar_atomic(rhs, row, _vec6_from_parts(force, torque), stripe, mask)
    if slot >= 0:
        for axis in range(3):
            for comp in range(3):
                index = axis * 6 + comp
                origin = slot * 36 + index
                if (index & mask) == stripe and ll[axis, comp] != 0.0:
                    wp.atomic_add(values, origin, ll[axis, comp])
                if ((index + 3) & mask) == stripe and al[comp, axis] != 0.0:
                    wp.atomic_add(values, origin + 3, al[comp, axis])
                if ((index + 18) & mask) == stripe and al[axis, comp] != 0.0:
                    wp.atomic_add(values, origin + 18, al[axis, comp])
                if ((index + 21) & mask) == stripe and aa[axis, comp] != 0.0:
                    wp.atomic_add(values, origin + 21, aa[axis, comp])


@wp.func
def _scatter_link(values: wp.array[float], slot: int, transpose: bool, cross: mat66f, stripe: int, mask: int):
    if slot >= 0:
        for axis in range(6):
            for comp in range(6):
                index = axis * 6 + comp
                if (index & mask) == stripe and cross[axis, comp] != 0.0:
                    addr = comp * 6 + axis if transpose else index
                    wp.atomic_add(values, slot * 36 + addr, cross[axis, comp])


@wp.func
def _evaluate_coordinate_no_modal(
    data: JointCoordinateData, joint: int, component: int, body_q: wp.array[wp.transform]
):
    angular_reference = wp.vec3()
    angular_start = data.q_start[joint] + data.dof_dim[joint, 0]
    for axis in range(data.dof_dim[joint, 1]):
        angular_reference[axis] = data.history[angular_start + axis]
    X_pj = data.X_p[joint]
    X_cj = data.X_c[joint]
    q, g_p, g_c = eval_joint_coordinate_in_frames(
        joint,
        component,
        body_q,
        data.body_com,
        data.joint_type,
        data.parent,
        data.child,
        X_pj,
        X_cj,
        data.qd_start,
        data.dof_dim,
        data.axis,
        angular_reference,
        True,
    )
    q = unwrap_joint_coordinate(q, data.history[data.q_start[joint] + component], component, data.dof_dim[joint, 0])
    return (q, g_p, g_c)


@wp.func
def _evaluate_angular_coordinate_no_modal(
    data: JointCoordinateData,
    joint: int,
    component: int,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    dt: float,
):
    """Measure the continuous angle, step velocity, and world-space angular gradient."""
    q, _g_p, g_c = _evaluate_coordinate_no_modal(data, joint, component, body_q)
    q_prev, _g_p_prev, _g_c_prev = _evaluate_coordinate_no_modal(data, joint, component, body_q_prev)
    delta = q - q_prev
    if wp.abs(delta) >= wp.pi:
        delta = wp.atan2(wp.sin(delta), wp.cos(delta))
    rate = delta / dt
    return (q, rate, wp.spatial_bottom(g_c))


@wp.func
def _sparse_angular_drive_limit_row_no_modal(
    joint_coordinates: JointCoordinateData,
    joint: int,
    coordinate: int,
    dof_idx: int,
    target_q_idx: int,
    penalty_slot: int,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    joint_penalty_k: wp.array[float],
    joint_drive_limit_support: wp.array[float],
    joint_drive_lambda: wp.array[float],
    joint_limit_lambda: wp.array[float],
    joint_compliant_alm: int,
    dt: float,
):
    """Return ``(force, hessian, angular_jacobian_world)`` of an angular DOF's drive and limit row.

    ``hessian`` is zero when the DOF has neither a drive nor a limit row.
    """
    has_drive = _drive_row_applies_force(joint_target_ke[dof_idx], joint_target_kd[dof_idx])
    has_limits = _limit_row_exists(joint_limit_ke[dof_idx], joint_limit_lower[dof_idx], joint_limit_upper[dof_idx])
    if has_drive or has_limits:
        theta_abs, dtheta_dt, angular_jacobian_world = _evaluate_angular_coordinate_no_modal(
            joint_coordinates, joint, coordinate, body_q, body_q_prev, dt
        )
        force_scalar, hessian_scalar = _evaluate_sparse_drive_limit(
            dof_idx,
            target_q_idx,
            penalty_slot,
            theta_abs,
            dtheta_dt,
            joint_target_ke,
            joint_target_kd,
            joint_target_q,
            joint_target_vel,
            joint_limit_lower,
            joint_limit_upper,
            joint_limit_ke,
            joint_limit_kd,
            joint_penalty_k,
            joint_drive_limit_support,
            joint_drive_lambda,
            joint_limit_lambda,
            joint_compliant_alm,
            dt,
        )
        return (force_scalar, hessian_scalar, angular_jacobian_world)
    return (float(0.0), float(0.0), wp.vec3(0.0))


@wp.func
def _sparse_joint_armature_row_no_modal(
    joint_coordinates: JointCoordinateData,
    joint: int,
    qd_start: int,
    body_q: wp.array[wp.transform],
    joint_armature: wp.array[float],
    joint_armature_target: wp.array[float],
    dt: float,
):
    """Return ``(active, g_parent, g_child, force, hessian)`` of a revolute joint's armature row."""
    armature = joint_armature[qd_start]
    if armature > 0.0:
        q, gp, gc = _evaluate_coordinate_no_modal(joint_coordinates, joint, 0, body_q)
        armature_hessian = armature / (dt * dt)
        armature_force = armature_hessian * (q - joint_armature_target[qd_start])
        return (True, gp, gc, armature_force, armature_hessian)
    return (False, wp.spatial_vector(), wp.spatial_vector(), float(0.0), float(0.0))


@wp.func
def _sparse_joint_friction_row_no_modal(
    joint_coordinates: JointCoordinateData,
    joint: int,
    component: int,
    lin_count: int,
    dof: int,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    joint_friction: wp.array[float],
    joint_damping: wp.array[float],
    dt: float,
):
    """Return ``(active, g_parent, g_child, force, hessian)`` of a free DOF's friction and damping row."""
    friction = joint_friction[dof]
    damping = wp.max(joint_damping[dof], 0.0)
    if friction > 0.0 or damping > 0.0:
        q, gp, gc = _evaluate_coordinate_no_modal(joint_coordinates, joint, component, body_q)
        q_prev, _gp_prev, _gc_prev = _evaluate_coordinate_no_modal(joint_coordinates, joint, component, body_q_prev)
        delta = q - q_prev
        if component >= lin_count:
            delta = wp.atan2(wp.sin(delta), wp.cos(delta))
        rate = delta / dt
        force, hessian = _eval_joint_axis_friction(rate, friction, 1.0 / dt)
        return (True, gp, gc, force + damping * rate, hessian + damping / dt)
    return (False, wp.spatial_vector(), wp.spatial_vector(), float(0.0), float(0.0))


@wp.func
def _load_gradient(geometry: wp.array[wp.vec3], cursor: int, count: int, parent: bool):
    start = cursor + 3 * count if parent else cursor
    return wp.matrix_from_rows(geometry[start], geometry[start + count], geometry[start + 2 * count])


@wp.func
def _shared_bending(
    q_wp: wp.quat,
    q_wc: wp.quat,
    kb_rest_local: wp.vec3,
    twist_rest: float,
    q_wp_prev: wp.quat,
    q_wc_prev: wp.quat,
    is_parent: bool,
    K_elastic_diag: wp.vec3,
    C0_force: wp.vec3,
    sigma0: wp.vec3,
    H_fric_diag: wp.vec3,
    lambda_projected: wp.vec3,
    K_damp_diag: wp.vec3,
    damping_active: bool,
    dt: float,
    has_parent: bool,
    geometry: wp.array[wp.vec3],
    cursor: int,
    count: int,
):
    """Bend/twist torque and Hessian for SolverVBD local +Z rod joints.

    In the fixed rod material basis, local angular operators are diagonal:
    ``[bend_x, bend_y, twist_z]``. Keep them as vec3 row scales in the hot path
    instead of building dense local matrices.
    """
    inv_dt = 1.0 / dt
    kappa_now_vec = geometry[6 * count + cursor]
    f_local = wp.cw_mul(K_elastic_diag, kappa_now_vec) - C0_force + sigma0 + lambda_projected
    H_local_diag = K_elastic_diag + H_fric_diag
    if damping_active:
        prev_measure = measure_rod_bend_twist_z(q_wp_prev, q_wc_prev)
        kappa_prev_vec = assemble_geometric_rod_kappa_z(
            q_wp_prev, prev_measure.kb_world, prev_measure.twist, kb_rest_local, twist_rest
        )
        dkappa_dt = rod_bend_twist_delta(kappa_now_vec, kappa_prev_vec) * inv_dt
        f_local = f_local + wp.cw_mul(K_damp_diag, dkappa_dt)
        H_local_diag = H_local_diag + inv_dt * K_damp_diag
    J_body = _load_gradient(geometry, cursor, count, is_parent)
    H_aa = wp.transpose(J_body) * _diag_mul_mat33(H_local_diag, J_body)
    tau_world = -(wp.transpose(J_body) * f_local)
    parent_tau = wp.vec3(0.0)
    parent_aa = wp.mat33(0.0)
    if has_parent:
        parent_jac = _load_gradient(geometry, cursor, count, True)
        parent_aa = wp.transpose(parent_jac) * _diag_mul_mat33(H_local_diag, parent_jac)
        parent_tau = -(wp.transpose(parent_jac) * f_local)
    return (tau_world, H_aa, parent_tau, parent_aa)


@wp.func
def _paired_bending(
    q_wp: wp.quat,
    q_wc: wp.quat,
    kb_rest_local: wp.vec3,
    twist_rest: float,
    q_wp_prev: wp.quat,
    q_wc_prev: wp.quat,
    is_parent: bool,
    K_elastic_diag: wp.vec3,
    C0_force: wp.vec3,
    sigma0: wp.vec3,
    H_fric_diag: wp.vec3,
    lambda_projected: wp.vec3,
    K_damp_diag: wp.vec3,
    damping_active: bool,
    dt: float,
    has_parent: bool,
):
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
        kappa_prev_vec = assemble_geometric_rod_kappa_z(
            q_wp_prev, prev_measure.kb_world, prev_measure.twist, kb_rest_local, twist_rest
        )
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
def _paired_rod_response(
    joint_index: int,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    X_pj: wp.transform,
    X_cj: wp.transform,
    X_pj_prev: wp.transform,
    X_cj_prev: wp.transform,
    joint_rod_rest_kb_local: wp.array[wp.vec3],
    joint_rod_rest_twist: wp.array[float],
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
    joint_rho: wp.array[float],
    joint_material_k: wp.array[float],
    joint_penalty_kd: wp.array[float],
    joint_sigma_start: wp.array[wp.vec3],
    joint_C_fric: wp.array[wp.vec3],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    stab_alpha: float,
    joint_compliant_alm: int,
    dt: float,
):
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
        bend_primal_k, bend_dual_eff = _material_force_terms(
            solve_weight_bend, material_bend, bend_dual, joint_compliant_alm
        )
        twist_primal_k, twist_dual_eff = _material_force_terms(
            solve_weight_twist, material_twist, twist_dual, joint_compliant_alm
        )
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
        rod_torque, rod_H_aa, torque_parent, haa_parent = _paired_bending(
            q_wp,
            q_wc,
            joint_rod_rest_kb_local[joint_index],
            joint_rod_rest_twist[joint_index],
            q_wp_prev,
            q_wc_prev,
            is_parent_body,
            K_elastic_diag,
            C0_force,
            sigma,
            H_fric_diag,
            bend_dual_eff + twist_dual_eff,
            K_damp_diag,
            kd_bend > 0.0 or kd_twist > 0.0,
            dt,
            parent_index >= 0,
        )
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
        stretch_primal_k, stretch_dual_eff = _material_force_terms(
            solve_weight_stretch, material_stretch, stretch_dual, joint_compliant_alm
        )
        shear_primal_k, shear_dual_eff = _material_force_terms(
            solve_weight_shear, material_shear, shear_dual, joint_compliant_alm
        )
        k_diag = wp.vec3(shear_primal_k, shear_primal_k, stretch_primal_k)
        kd_diag = wp.vec3(kd_shear, kd_shear, kd_stretch)
        stretch_alpha = stab_alpha if joint_compliant_alm == 1 or stretch_hard else 0.0
        shear_alpha = stab_alpha if joint_compliant_alm == 1 or shear_hard else 0.0
        C0_force_local = shear_primal_k * shear_alpha * wp.vec3(C0_lin[0], C0_lin[1], 0.0)
        C0_force_local = C0_force_local + stretch_primal_k * stretch_alpha * wp.vec3(0.0, 0.0, C0_lin[2])
        f_l, t_l, Hll_l, Hal_l, Haa_l = evaluate_rod_stretch_shear_force_hessian(
            X_wp,
            X_wc,
            X_wp_prev,
            X_wc_prev,
            parent_pose,
            child_pose,
            parent_com,
            child_com,
            is_parent_body,
            k_diag,
            C0_force_local,
            stretch_dual_eff + shear_dual_eff,
            kd_diag,
            kd_stretch > 0.0 or kd_shear > 0.0,
            dt,
        )
        total_force = total_force + f_l
        total_torque = total_torque + t_l
        total_H_ll = total_H_ll + Hll_l
        total_H_al = total_H_al + Hal_l
        total_H_aa = total_H_aa + Haa_l
        if parent_index >= 0:
            f_l_p, t_l_p, Hll_l_p, Hal_l_p, Haa_l_p = evaluate_rod_stretch_shear_force_hessian(
                X_wp,
                X_wc,
                X_wp_prev,
                X_wc_prev,
                parent_pose,
                child_pose,
                parent_com,
                child_com,
                True,
                k_diag,
                C0_force_local,
                stretch_dual_eff + shear_dual_eff,
                kd_diag,
                kd_stretch > 0.0 or kd_shear > 0.0,
                dt,
            )
            total_force_p = total_force_p + f_l_p
            total_torque_p = total_torque_p + t_l_p
            total_H_ll_p = total_H_ll_p + Hll_l_p
            total_H_al_p = total_H_al_p + Hal_l_p
            total_H_aa_p = total_H_aa_p + Haa_l_p
    return (
        total_force,
        total_torque,
        total_H_ll,
        total_H_al,
        total_H_aa,
        total_force_p,
        total_torque_p,
        total_H_ll_p,
        total_H_al_p,
        total_H_aa_p,
    )


@wp.func
def _rod_rate_free_arm_blocks(
    arm: wp.vec3, anchor: wp.vec3, world_k: wp.mat33, iso: float, isotropic: bool, parent_side: bool
):
    """Angular-linear and angular-angular tiles. For S=skew(arm), S.T*K = -(S*K)."""
    skew_arm = wp.skew(arm)
    if parent_side:
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
def _rate_free_cross(
    dt: float,
    joint: int,
    parent: int,
    child: int,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    X_pj: wp.transform,
    X_cj: wp.transform,
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
    joint_rho: wp.array[float],
    joint_material_k: wp.array[float],
    joint_penalty_kd: wp.array[float],
    joint_C_fric: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    joint_compliant_alm: int,
    stripe: int,
    lane_mask: int,
    geometry: wp.array[wp.vec3],
    cursor: int,
    count: int,
    values: wp.array[float],
    slot: int,
    flip: bool,
):
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
    k_stretch, _ = _material_force_terms(
        stretch_weight, joint_material_k[stretch_idx], wp.vec3(0.0), joint_compliant_alm
    )
    k_shear, _ = _material_force_terms(shear_weight, joint_material_k[shear_idx], wp.vec3(0.0), joint_compliant_alm)
    k_bend, _ = _material_force_terms(bend_weight, joint_material_k[bend_idx], wp.vec3(0.0), joint_compliant_alm)
    k_twist, _ = _material_force_terms(twist_weight, joint_material_k[twist_idx], wp.vec3(0.0), joint_compliant_alm)
    X_wp = body_q[parent] * X_pj
    X_wc = body_q[child] * X_cj
    q_wp = wp.transform_get_rotation(X_wp)
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
        J_parent_angular = _load_gradient(geometry, cursor, count, True)
        J_child_angular = _load_gradient(geometry, cursor, count, False)
        H_local_diag = wp.vec3(h_bend_x, h_bend_y, h_twist)
        aa = aa + wp.transpose(_diag_mul_mat33(H_local_diag, J_parent_angular)) * J_child_angular
    _rigid_add_tile(values, slot, 3, 3, aa, flip, stripe, lane_mask)


@wp.func
def _rate_free_response(
    joint_index: int,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    X_pj: wp.transform,
    X_cj: wp.transform,
    X_pj_prev: wp.transform,
    X_cj_prev: wp.transform,
    joint_rod_rest_kb_local: wp.array[wp.vec3],
    joint_rod_rest_twist: wp.array[float],
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
    joint_rho: wp.array[float],
    joint_material_k: wp.array[float],
    joint_penalty_kd: wp.array[float],
    joint_sigma_start: wp.array[wp.vec3],
    joint_C_fric: wp.array[wp.vec3],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    stab_alpha: float,
    joint_compliant_alm: int,
    dt: float,
    geometry: wp.array[wp.vec3],
    cursor: int,
    count: int,
):
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
        bend_primal_k, bend_dual_eff = _material_force_terms(
            solve_weight_bend, material_bend, bend_dual, joint_compliant_alm
        )
        twist_primal_k, twist_dual_eff = _material_force_terms(
            solve_weight_twist, material_twist, twist_dual, joint_compliant_alm
        )
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
        rod_torque, rod_H_aa, torque_parent, haa_parent = _shared_bending(
            q_wp,
            q_wc,
            joint_rod_rest_kb_local[joint_index],
            joint_rod_rest_twist[joint_index],
            q_wp,
            q_wc,
            is_parent_body,
            K_elastic_diag,
            C0_force,
            sigma,
            H_fric_diag,
            bend_dual_eff + twist_dual_eff,
            wp.vec3(0.0),
            False,
            dt,
            parent_index >= 0,
            geometry,
            cursor,
            count,
        )
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
        stretch_primal_k, stretch_dual_eff = _material_force_terms(
            solve_weight_stretch, material_stretch, stretch_dual, joint_compliant_alm
        )
        shear_primal_k, shear_dual_eff = _material_force_terms(
            solve_weight_shear, material_shear, shear_dual, joint_compliant_alm
        )
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
            total_H_al_p, parent_aa = _rod_rate_free_arm_blocks(
                parent_arm, parent_anchor - parent_center, total_H_ll, iso, isotropic, True
            )
            total_H_aa_p = total_H_aa_p + parent_aa
            if is_parent_body:
                total_force = world_force
                total_torque = total_torque + parent_torque
                total_H_al = total_H_al_p
                total_H_aa = total_H_aa + parent_aa
    return (
        total_force,
        total_torque,
        total_H_ll,
        total_H_al,
        total_H_aa,
        total_force_p,
        total_torque_p,
        total_H_ll_p,
        total_H_al_p,
        total_H_aa_p,
    )


@wp.kernel(enable_backward=False)
def prepare_joint_scatter(
    articulation_joints: wp.array[int],
    articulation_joint_body_start: wp.array[int],
    articulation_block_row_offsets: wp.array[int],
    articulation_block_cols: wp.array[int],
    body_articulation_local: wp.array[int],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_constraint_start: wp.array[int],
    joint_penalty_kd: wp.array[float],
    body_q: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_rod_rest_kb_local: wp.array[wp.vec3],
    joint_rod_rest_twist: wp.array[float],
    edges: wp.array[wp.vec4i],
    geometry: wp.array[wp.vec3],
):
    """Index rod blocks and cache undamped rod strain and both endpoint Jacobians."""
    task, cursor = wp.tid()
    count = edges.shape[0]
    joint = articulation_joints[cursor]
    if task != 0:
        if not joint_enabled[joint] or joint_type[joint] != JointType.ROD or joint_child[joint] < 0:
            return
        parent = joint_parent[joint]
        child = joint_child[joint]
        first = joint_constraint_start[joint]
        if (
            joint_penalty_kd[first] != 0.0
            or joint_penalty_kd[first + 1] != 0.0
            or joint_penalty_kd[first + 2] != 0.0
            or joint_penalty_kd[first + 3] != 0.0
        ):
            return
        parent_pose = wp.transform(wp.vec3(0.0), wp.quat_identity())
        child_pose = body_q[child]
        if parent >= 0:
            parent_pose = body_q[parent]
        parent_rotation = wp.transform_get_rotation(parent_pose * joint_X_p[joint])
        child_rotation = wp.transform_get_rotation(child_pose * joint_X_c[joint])
        measure = measure_rod_bend_twist_z(parent_rotation, child_rotation)
        strain = assemble_geometric_rod_kappa_z(
            parent_rotation,
            measure.kb_world,
            measure.twist,
            joint_rod_rest_kb_local[joint],
            joint_rod_rest_twist[joint],
        )
        geometry[6 * count + cursor] = strain
        gradient_child = rod_bend_twist_jacobian_z_from_measure(measure, False)
        for row in range(3):
            geometry[row * count + cursor] = wp.vec3(
                gradient_child[row, 0], gradient_child[row, 1], gradient_child[row, 2]
            )
        if parent >= 0:
            gradient_parent = rod_bend_twist_jacobian_z_from_measure(measure, True)
            for row in range(3):
                geometry[(3 + row) * count + cursor] = wp.vec3(
                    gradient_parent[row, 0], gradient_parent[row, 1], gradient_parent[row, 2]
                )
        return
    if joint_enabled[joint] and joint_type[joint] == JointType.ROD and joint_child[joint] >= 0:
        child = body_articulation_local[joint_child[joint]]
        base_body = articulation_joint_body_start[cursor]
        slot_child = _find_block_slot(articulation_block_row_offsets, articulation_block_cols, base_body, child, child)
        slot_parent = int(-1)
        slot_cross = int(-1)
        if joint_parent[joint] >= 0:
            parent = body_articulation_local[joint_parent[joint]]
            slot_parent = _find_block_slot(
                articulation_block_row_offsets, articulation_block_cols, base_body, parent, parent
            )
            slot_cross = _find_block_slot(
                articulation_block_row_offsets,
                articulation_block_cols,
                base_body,
                wp.max(parent, child),
                wp.min(parent, child),
            )
        first = joint_constraint_start[joint]
        no_rates = (
            joint_penalty_kd[first] == 0.0
            and joint_penalty_kd[first + 1] == 0.0
            and joint_penalty_kd[first + 2] == 0.0
            and joint_penalty_kd[first + 3] == 0.0
        )
        edges[cursor] = wp.vec4i(slot_child, slot_parent, slot_cross, int(no_rates))


@wp.func
def _scatter_joint(
    task: int,
    joint_cursor: int,
    stripe: int,
    dt: float,
    articulation_joints: wp.array[wp.int32],
    articulation_joint_body_start: wp.array[wp.int32],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    body_articulation_local: wp.array[wp.int32],
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
    joint_dof_dim: wp.array2d[int],
    joint_coordinates: JointCoordinateData,
    joint_friction: wp.array[float],
    joint_damping: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_armature: wp.array[float],
    joint_armature_target: wp.array[float],
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
    edges: wp.array[wp.vec4i],
    geometry: wp.array[wp.vec3],
    lane_mask: int,
    values_scalar: wp.array[float],
    rhs_scalar: wp.array[float],
):
    joint_compliant_alm = int(1)
    joint_count = edges.shape[0]
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
                    _rate_free_cross(
                        dt,
                        joint,
                        parent,
                        child,
                        body_q,
                        body_com,
                        X_p,
                        X_c,
                        joint_constraint_start,
                        joint_penalty_k,
                        joint_rho,
                        joint_material_k,
                        joint_penalty_kd,
                        joint_C_fric,
                        joint_is_hard,
                        joint_compliant_alm,
                        stripe,
                        lane_mask,
                        geometry,
                        joint_cursor,
                        joint_count,
                        values_scalar,
                        entry[2],
                        parent_local < child_local,
                    )
                return
            endpoint_force, torque, ll, al, aa, endpoint_force_p, torque_p, ll_p, al_p, aa_p = _rate_free_response(
                joint,
                body_q,
                body_q_prev,
                body_com,
                joint_parent,
                joint_child,
                X_p,
                X_c,
                X_p_prev,
                X_c_prev,
                joint_rod_rest_kb_local,
                joint_rod_rest_twist,
                joint_constraint_start,
                joint_penalty_k,
                joint_rho,
                joint_material_k,
                joint_penalty_kd,
                joint_sigma_start,
                joint_C_fric,
                joint_lambda_lin,
                joint_lambda_ang,
                joint_C0_lin,
                joint_C0_ang,
                joint_is_hard,
                avbd_alpha,
                joint_compliant_alm,
                dt,
                geometry,
                joint_cursor,
                joint_count,
            )
            _scatter_endpoint(
                values_scalar,
                rhs_scalar,
                body_start + child_local,
                entry[0],
                endpoint_force,
                torque,
                ll,
                al,
                aa,
                stripe,
                lane_mask,
            )
            if parent >= 0:
                _scatter_endpoint(
                    values_scalar,
                    rhs_scalar,
                    body_start + parent_local,
                    entry[1],
                    endpoint_force_p,
                    torque_p,
                    ll_p,
                    al_p,
                    aa_p,
                    stripe,
                    lane_mask,
                )
            return
        if task != 0:
            return
        endpoint_force, torque, ll, al, aa, endpoint_force_p, torque_p, ll_p, al_p, aa_p = _paired_rod_response(
            joint,
            body_q,
            body_q_prev,
            body_com,
            joint_parent,
            joint_child,
            X_p,
            X_c,
            X_p_prev,
            X_c_prev,
            joint_rod_rest_kb_local,
            joint_rod_rest_twist,
            joint_constraint_start,
            joint_penalty_k,
            joint_rho,
            joint_material_k,
            joint_penalty_kd,
            joint_sigma_start,
            joint_C_fric,
            joint_lambda_lin,
            joint_lambda_ang,
            joint_C0_lin,
            joint_C0_ang,
            joint_is_hard,
            avbd_alpha,
            joint_compliant_alm,
            dt,
        )
        _scatter_endpoint(
            values_scalar,
            rhs_scalar,
            body_start + child_local,
            entry[0],
            endpoint_force,
            torque,
            ll,
            al,
            aa,
            stripe,
            lane_mask,
        )
        if parent >= 0:
            _scatter_endpoint(
                values_scalar,
                rhs_scalar,
                body_start + parent_local,
                entry[1],
                endpoint_force_p,
                torque_p,
                ll_p,
                al_p,
                aa_p,
                stripe,
                lane_mask,
            )
            cross = _rod_cross_hessian(
                dt,
                joint,
                parent,
                child,
                body_q,
                body_com,
                X_p,
                X_c,
                joint_constraint_start,
                joint_penalty_k,
                joint_rho,
                joint_material_k,
                joint_penalty_kd,
                joint_C_fric,
                joint_is_hard,
                joint_compliant_alm,
                stripe,
                lane_mask,
            )
            _scatter_link(values_scalar, entry[2], parent_local < child_local, cross, stripe, lane_mask)
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
    g = _sparse_joint_geometry(
        joint,
        jt,
        parent,
        child,
        X_p,
        X_c,
        X_p_prev,
        X_c_prev,
        body_q,
        body_q_prev,
        body_q_rest,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_qd_start,
        joint_constraint_start,
        joint_penalty_k,
        joint_rho,
        joint_material_k,
        joint_penalty_kd,
        joint_dof_dim,
        joint_lambda_lin,
        joint_lambda_ang,
        joint_C0_lin,
        joint_C0_ang,
        joint_is_hard,
        avbd_alpha,
        joint_compliant_alm,
    )
    if g.k_linear > 0.0 and (jt != JointType.D6 or lin_count < 3):
        _assemble_linear_joint_scalar(
            values_scalar,
            rhs_scalar,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            parent_local,
            child_local,
            parent,
            child,
            g.parent_anchor,
            g.child_anchor,
            g.parent_anchor_prev,
            g.child_anchor_prev,
            body_q,
            body_com,
            g.k_linear,
            g.kd_linear,
            dt,
            g.P_lin,
            g.lin_lambda,
            g.lin_C0,
            g.lin_alpha,
            cross_slot,
            stripe,
            lane_mask,
        )
    if g.k_angular > 0.0 and jt != JointType.BALL and (jt != JointType.D6 or ang_count < 3):
        sigma0 = wp.vec3(0.0)
        C_fric = wp.vec3(0.0)
        _kappa, _J_world = _assemble_angular_joint_scalar(
            values_scalar,
            rhs_scalar,
            articulation_block_row_offsets,
            articulation_block_cols,
            body_start,
            parent_local,
            child_local,
            parent,
            g.parent_anchor_q,
            g.child_anchor_q,
            g.parent_anchor_q_prev,
            g.child_anchor_q_prev,
            g.parent_rest_q,
            g.child_rest_q,
            g.k_angular,
            g.kd_angular,
            dt,
            g.P_ang,
            sigma0,
            C_fric,
            g.ang_lambda,
            g.ang_C0,
            g.ang_alpha,
            cross_slot,
            joint_axis,
            qd_start + lin_count,
            ang_count,
            stripe,
            lane_mask,
        )
    if jt == JointType.REVOLUTE or jt == JointType.PRISMATIC or jt == JointType.D6:
        for component in range(lin_count + ang_count):
            active, gp, gc, force, hessian = _sparse_joint_friction_row_no_modal(
                joint_coordinates,
                joint,
                component,
                lin_count,
                qd_start + component,
                body_q,
                body_q_prev,
                joint_friction,
                joint_damping,
                dt,
            )
            if active:
                _assemble_coordinate_row_scalar(
                    values_scalar,
                    rhs_scalar,
                    articulation_block_row_offsets,
                    articulation_block_cols,
                    body_start,
                    parent_local,
                    child_local,
                    gp,
                    gc,
                    force,
                    hessian,
                    cross_slot,
                    stripe,
                    lane_mask,
                )
    if jt == JointType.REVOLUTE:
        active, gp, gc, force, hessian = _sparse_joint_armature_row_no_modal(
            joint_coordinates, joint, qd_start, body_q, joint_armature, joint_armature_target, dt
        )
        if active:
            _assemble_coordinate_row_scalar(
                values_scalar,
                rhs_scalar,
                articulation_block_row_offsets,
                articulation_block_cols,
                body_start,
                parent_local,
                child_local,
                gp,
                gc,
                force,
                hessian,
                cross_slot,
                stripe,
                lane_mask,
            )
    if jt == JointType.REVOLUTE:
        force_scalar, hessian_scalar, angular_jacobian_world = _sparse_angular_drive_limit_row_no_modal(
            joint_coordinates,
            joint,
            0,
            qd_start,
            joint_target_q_start[joint],
            c_start + 2,
            body_q,
            body_q_prev,
            joint_target_ke,
            joint_target_kd,
            joint_target_q,
            joint_target_vel,
            joint_limit_lower,
            joint_limit_upper,
            joint_limit_ke,
            joint_limit_kd,
            joint_penalty_k,
            joint_drive_limit_support,
            joint_drive_lambda,
            joint_limit_lambda,
            joint_compliant_alm,
            dt,
        )
        if hessian_scalar > 0.0:
            _assemble_angular_axis_row_scalar(
                values_scalar,
                rhs_scalar,
                articulation_block_row_offsets,
                articulation_block_cols,
                body_start,
                parent_local,
                child_local,
                parent,
                angular_jacobian_world,
                force_scalar,
                hessian_scalar,
                cross_slot,
                stripe,
                lane_mask,
            )
    if jt == JointType.PRISMATIC or jt == JointType.D6:
        C_vec = g.child_anchor - g.parent_anchor
        C_vec_prev = g.child_anchor_prev - g.parent_anchor_prev
        target_q_base = joint_target_q_start[joint]
        for li in range(3):
            if li < lin_count:
                force_scalar, hessian_scalar, axis_world = _sparse_linear_drive_limit_row(
                    qd_start + li,
                    target_q_base + li,
                    c_start + 2 + li,
                    g.parent_anchor_q,
                    C_vec,
                    C_vec_prev,
                    joint_axis,
                    joint_target_ke,
                    joint_target_kd,
                    joint_target_q,
                    joint_target_vel,
                    joint_limit_lower,
                    joint_limit_upper,
                    joint_limit_ke,
                    joint_limit_kd,
                    joint_penalty_k,
                    joint_drive_limit_support,
                    joint_drive_lambda,
                    joint_limit_lambda,
                    joint_compliant_alm,
                    dt,
                )
                if hessian_scalar > 0.0:
                    _assemble_linear_axis_row_scalar(
                        values_scalar,
                        rhs_scalar,
                        articulation_block_row_offsets,
                        articulation_block_cols,
                        body_start,
                        parent_local,
                        child_local,
                        parent,
                        child,
                        g.parent_anchor,
                        g.child_anchor,
                        body_q,
                        body_com,
                        axis_world,
                        force_scalar,
                        hessian_scalar,
                        cross_slot,
                        stripe,
                        lane_mask,
                    )
    if jt == JointType.D6:
        target_q_base = joint_target_q_start[joint]
        for ai in range(3):
            if ai < ang_count:
                force_scalar, hessian_scalar, angular_jacobian_world = _sparse_angular_drive_limit_row_no_modal(
                    joint_coordinates,
                    joint,
                    lin_count + ai,
                    qd_start + lin_count + ai,
                    target_q_base + lin_count + ai,
                    c_start + 2 + lin_count + ai,
                    body_q,
                    body_q_prev,
                    joint_target_ke,
                    joint_target_kd,
                    joint_target_q,
                    joint_target_vel,
                    joint_limit_lower,
                    joint_limit_upper,
                    joint_limit_ke,
                    joint_limit_kd,
                    joint_penalty_k,
                    joint_drive_limit_support,
                    joint_drive_lambda,
                    joint_limit_lambda,
                    joint_compliant_alm,
                    dt,
                )
                if hessian_scalar > 0.0:
                    _assemble_angular_axis_row_scalar(
                        values_scalar,
                        rhs_scalar,
                        articulation_block_row_offsets,
                        articulation_block_cols,
                        body_start,
                        parent_local,
                        child_local,
                        parent,
                        angular_jacobian_world,
                        force_scalar,
                        hessian_scalar,
                        cross_slot,
                        stripe,
                        lane_mask,
                    )
