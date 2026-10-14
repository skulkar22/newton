# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import functools

import warp as wp

from newton._src.math.spatial import quat_velocity
from newton._src.solvers.vbd.joint_coordinates import JointCoordinateData
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _SMALL_ANGLE_EPS,
    _SMALL_LENGTH_EPS,
    RigidForceElementAdjacencyInfo,
    _load_solve_weight,
    contact_surface_separation,
    evaluate_contact_point_world,
    evaluate_joint_force_hessian,
    evaluate_rigid_contact_from_collision,
    get_body_adjacent_joint_id,
    get_body_num_adjacent_joints,
)

wp.set_module_options({"enable_backward": False})

_NUM_ELASTIC_CONTACT_THREADS_PER_BODY = 32
"""Threads per reduced elastic body for modal contact accumulation."""


@wp.kernel
def copy_elastic_body_transforms_back(
    body_ids: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    body_q_in: wp.array[wp.transform],
    body_q_out: wp.array[wp.transform],
):
    """Copy the coupled frame solve result for elastic bodies in one color group."""
    body = body_ids[wp.tid()]
    if body_elastic_index[body] >= 0:
        body_q_out[body] = body_q_in[body]


@wp.func
def _so3_left_jacobian(theta: wp.vec3) -> wp.mat33:
    """Map a rotation-vector increment to a left-trivialized angular increment."""
    theta_sq = wp.dot(theta, theta)
    theta_skew = wp.skew(theta)
    theta_skew_sq = theta_skew * theta_skew
    if theta_sq < _SMALL_ANGLE_EPS * _SMALL_ANGLE_EPS:
        return wp.identity(3, float) + 0.5 * theta_skew + (1.0 / 6.0) * theta_skew_sq

    angle = wp.sqrt(theta_sq)
    a = (1.0 - wp.cos(angle)) / theta_sq
    b = (angle - wp.sin(angle)) / (theta_sq * angle)
    return wp.identity(3, float) + a * theta_skew + b * theta_skew_sq


@wp.func
def _elastic_endpoint_xform(
    joint_index: int,
    body_index: int,
    is_parent_side: bool,
    xform_rest: wp.transform,
    body_elastic_index: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    joint_q: wp.array[float],
    joint_q_start: wp.array[wp.int32],
    joint_parent_elastic_endpoint: wp.array[wp.int32],
    joint_child_elastic_endpoint: wp.array[wp.int32],
    elastic_endpoint_phi: wp.array[wp.vec3],
    elastic_endpoint_psi: wp.array[wp.vec3],
    elastic_max_mode_count: int,
):
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
    for mode in range(elastic_max_mode_count):
        if mode < mode_count:
            idx = endpoint * elastic_max_mode_count + mode
            p = p + elastic_endpoint_phi[idx] * joint_q[q_start + mode]
            theta = theta + elastic_endpoint_psi[idx] * joint_q[q_start + mode]

    angle = wp.length(theta)
    if angle > _SMALL_ANGLE_EPS:
        q = wp.quat_from_axis_angle(theta / angle, angle) * q

    return wp.transform(p, q)


@wp.kernel
def copy_elastic_joint_frame_to_body(
    elastic_body: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    elastic_index = wp.tid()
    body = elastic_body[elastic_index]
    joint = elastic_joint[elastic_index]
    q_start = joint_q_start[joint]
    qd_start = joint_qd_start[joint]

    body_q[body] = wp.transform(
        wp.vec3(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2]),
        wp.quat(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6]),
    )
    body_qd[body] = wp.spatial_vector(
        wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]),
        wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5]),
    )


@wp.kernel
def integrate_elastic_modes_implicit(
    dt: float,
    elastic_joint: wp.array[wp.int32],
    elastic_mode_start: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_mode_mass: wp.array[float],
    elastic_mode_stiffness: wp.array[float],
    elastic_mode_damping: wp.array[float],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_f: wp.array[float],
    joint_q_in: wp.array[float],
    joint_qd_in: wp.array[float],
    joint_q_out: wp.array[float],
    joint_qd_out: wp.array[float],
):
    elastic_index = wp.tid()
    joint = elastic_joint[elastic_index]
    q_start = joint_q_start[joint] + 7
    qd_start = joint_qd_start[joint] + 6
    mode_start = elastic_mode_start[elastic_index]
    mode_count = elastic_mode_count[elastic_index]

    for i in range(mode_count):
        mode = mode_start + i
        q_idx = q_start + i
        qd_idx = qd_start + i

        q = joint_q_in[q_idx]
        v = joint_qd_in[qd_idx]
        force = joint_f[qd_idx]
        mass = elastic_mode_mass[mode]
        stiffness = elastic_mode_stiffness[mode]
        damping = elastic_mode_damping[mode]

        denom = mass + dt * damping + dt * dt * stiffness
        if denom <= 0.0:
            joint_q_out[q_idx] = q
            joint_qd_out[qd_idx] = 0.0
        else:
            v_new = (mass * v + dt * (force - stiffness * q)) / denom
            q_new = q + dt * v_new
            joint_q_out[q_idx] = q_new
            joint_qd_out[qd_idx] = v_new


@wp.kernel
def copy_elastic_modes(
    elastic_joint: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_q_src: wp.array[float],
    joint_qd_src: wp.array[float],
    joint_q_dst: wp.array[float],
    joint_qd_dst: wp.array[float],
):
    elastic_index = wp.tid()
    joint = elastic_joint[elastic_index]
    q_start = joint_q_start[joint] + 7
    qd_start = joint_qd_start[joint] + 6
    mode_count = elastic_mode_count[elastic_index]

    for i in range(mode_count):
        joint_q_dst[q_start + i] = joint_q_src[q_start + i]
        joint_qd_dst[qd_start + i] = joint_qd_src[qd_start + i]


@wp.func
def _accumulate_elastic_contact_modes(
    dt: float,
    solve_frame: bool,
    contact_idx: int,
    body: int,
    elastic_index: int,
    body_R: wp.mat33,
    body_R_T: wp.mat33,
    elastic_joint: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_max_mode_count: int,
    body_elastic_index: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    rigid_contact_elastic_sample0: wp.array[wp.int32],
    rigid_contact_elastic_sample1: wp.array[wp.int32],
    contact_penalty_k: wp.array[float],
    contact_normal_rho: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_tangent_rho: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    stab_alpha: float,
    legacy_hard_contacts: int,
    contact_compliant_alm: int,
    friction_epsilon: float,
    joint_q_start: wp.array[wp.int32],
    elastic_shape_vertex_local: wp.array[wp.vec3],
    elastic_shape_vertex_phi: wp.array[wp.vec3],
    joint_q_prev: wp.array[float],
    joint_q: wp.array[float],
    elastic_body_block_grad: wp.array2d[float],
    elastic_body_block_matrix: wp.array3d[float],
):
    s0 = rigid_contact_shape0[contact_idx]
    s1 = rigid_contact_shape1[contact_idx]
    if s0 < 0 or s1 < 0:
        return

    b0 = shape_body[s0]
    b1 = shape_body[s1]
    elastic_sample0 = rigid_contact_elastic_sample0[contact_idx]
    elastic_sample1 = rigid_contact_elastic_sample1[contact_idx]

    elastic_body = -1
    elastic_sample = -1
    use_side0 = False
    if elastic_sample0 >= 0:
        elastic_body = b0
        elastic_sample = elastic_sample0
        use_side0 = True
    elif elastic_sample1 >= 0:
        elastic_body = b1
        elastic_sample = elastic_sample1

    if elastic_body != body or elastic_sample < 0:
        return

    cp0_local = rigid_contact_point0[contact_idx]
    cp1_local = rigid_contact_point1[contact_idx]
    cp0_world, cp0_world_prev = evaluate_contact_point_world(
        b0,
        cp0_local,
        elastic_sample0,
        body_q,
        body_q_prev,
        body_elastic_index,
        elastic_joint,
        elastic_mode_count,
        joint_q,
        joint_q_prev,
        joint_q_start,
        elastic_shape_vertex_local,
        elastic_shape_vertex_phi,
        elastic_max_mode_count,
    )
    cp1_world, cp1_world_prev = evaluate_contact_point_world(
        b1,
        cp1_local,
        elastic_sample1,
        body_q,
        body_q_prev,
        body_elastic_index,
        elastic_joint,
        elastic_mode_count,
        joint_q,
        joint_q_prev,
        joint_q_start,
        elastic_shape_vertex_local,
        elastic_shape_vertex_phi,
        elastic_max_mode_count,
    )

    contact_normal = rigid_contact_normal[contact_idx]
    C_n = -contact_surface_separation(
        cp0_world,
        cp1_world,
        contact_normal,
        rigid_contact_margin0[contact_idx],
        rigid_contact_margin1[contact_idx],
    )
    lam_vec = wp.vec3(0.0)
    lam_n = float(0.0)
    C_eff = C_n
    friction_c0 = wp.vec3(0.0)
    normal_solve_weight = _load_solve_weight(contact_penalty_k, contact_normal_rho, contact_idx, contact_compliant_alm)
    material_k = contact_material_ke[contact_idx]
    if legacy_hard_contacts == 0 and contact_compliant_alm == 0:
        normal_solve_weight = material_k
    tangent_solve_weight = contact_tangent_rho[contact_idx]

    if legacy_hard_contacts == 1 or contact_compliant_alm == 1:
        lam_vec = contact_lambda[contact_idx]
        lam_n = wp.dot(lam_vec, contact_normal)
        C0_vec = contact_C0[contact_idx]
        C0_n = wp.dot(contact_normal, C0_vec)
        C_eff = C_n - stab_alpha * C0_n
        friction_c0 = (1.0 - stab_alpha) * (C0_vec - contact_normal * C0_n)

    if C_n <= _SMALL_LENGTH_EPS and lam_n <= 0.0:
        return

    (
        force_0,
        _torque_0,
        h_ll_0,
        h_al_0,
        _h_aa_0,
        force_1,
        _torque_1,
        h_ll_1,
        h_al_1,
        _h_aa_1,
    ) = evaluate_rigid_contact_from_collision(
        b0,
        b1,
        body_q,
        body_q_prev,
        body_com,
        cp0_world,
        cp1_world,
        cp0_world_prev,
        cp1_world_prev,
        wp.vec3(0.0),  # mesh surface velocity is not applied to reduced elastic contacts
        rigid_contact_offset0[contact_idx],
        rigid_contact_offset1[contact_idx],
        1,
        contact_normal,
        C_eff,
        normal_solve_weight,
        material_k,
        tangent_solve_weight,
        contact_material_kd[contact_idx],
        lam_vec,
        contact_material_mu[contact_idx],
        friction_epsilon,
        legacy_hard_contacts,
        contact_compliant_alm,
        dt,
        friction_c0,
    )

    elastic_force = force_1
    elastic_h = h_ll_1
    elastic_h_al = h_al_1
    if use_side0:
        elastic_force = force_0
        elastic_h = h_ll_0
        elastic_h_al = h_al_0

    elastic_force_local = body_R_T * elastic_force
    elastic_h_local = body_R_T * (elastic_h * body_R)
    elastic_h_al_local = body_R_T * (elastic_h_al * body_R)
    mode_count = elastic_mode_count[elastic_index]
    max_modes = elastic_max_mode_count

    for i in range(mode_count):
        phi_i_local = elastic_shape_vertex_phi[elastic_sample * max_modes + i]
        wp.atomic_add(
            elastic_body_block_grad,
            elastic_index,
            6 + i,
            -wp.dot(elastic_force_local, phi_i_local),
        )

        if solve_frame and body_inv_mass[body] > 0.0:
            linear_cross = elastic_h_local * phi_i_local
            angular_cross = elastic_h_al_local * phi_i_local
            for axis in range(3):
                wp.atomic_add(
                    elastic_body_block_matrix,
                    elastic_index,
                    axis,
                    6 + i,
                    linear_cross[axis],
                )
                wp.atomic_add(
                    elastic_body_block_matrix,
                    elastic_index,
                    3 + axis,
                    6 + i,
                    angular_cross[axis],
                )

        for j in range(i, mode_count):
            phi_j_local = elastic_shape_vertex_phi[elastic_sample * max_modes + j]
            h_ij = wp.dot(phi_i_local, elastic_h_local * phi_j_local)
            wp.atomic_add(elastic_body_block_matrix, elastic_index, 6 + i, 6 + j, h_ij)


@wp.struct
class ElasticEndpointJointTerms:
    """Joint force and Hessian at each deformed elastic attachment, in the elastic body's axes.

    Entries are indexed like ``RigidForceElementAdjacencyInfo.body_adj_joints`` (one per body and
    adjacent joint) and written by :func:`evaluate_elastic_endpoint_joints` for the joints that attach
    at an elastic endpoint of the body.
    """

    force: wp.array[wp.vec3]
    """Joint force on the body [N]."""
    torque: wp.array[wp.vec3]
    """Joint torque on the body about its rest center of mass [N m]."""
    endpoint_torque: wp.array[wp.vec3]
    """Joint torque on the body about the deformed attachment [N m]."""
    hessian_ll: wp.array[wp.mat33]
    """Linear-linear joint Hessian block [N/m]."""
    hessian_al: wp.array[wp.mat33]
    """Angular-linear joint Hessian block about the rest center of mass [N]."""
    hessian_aa: wp.array[wp.mat33]
    """Angular-angular joint Hessian block about the rest center of mass [N m]."""
    endpoint_hessian_aa: wp.array[wp.mat33]
    """Angular-angular joint Hessian block about the deformed attachment [N m]."""
    rotation_jacobian: wp.array[wp.mat33]
    """Left Jacobian of the attachment's modal rotation vector."""


@wp.kernel
def evaluate_elastic_endpoint_joints(
    dt: float,
    body_ids: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    adjacency: RigidForceElementAdjacencyInfo,
    elastic_endpoint_phi: wp.array[wp.vec3],
    elastic_endpoint_psi: wp.array[wp.vec3],
    elastic_max_mode_count: int,
    body_elastic_index: wp.array[wp.int32],
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
    joint_qd_start: wp.array[wp.int32],
    joint_target_q_start: wp.array[wp.int32],
    joint_constraint_start: wp.array[wp.int32],
    joint_penalty_k: wp.array[float],
    joint_rho: wp.array[float],
    joint_material_k: wp.array[float],
    joint_penalty_kd: wp.array[float],
    joint_sigma_start: wp.array[wp.vec3],
    joint_C_fric: wp.array[wp.vec3],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
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
    stab_alpha: float,
    joint_compliant_alm: int,
    joint_dof_dim: wp.array2d[int],
    joint_coordinates: JointCoordinateData,
    joint_parent_elastic_endpoint: wp.array[wp.int32],
    joint_child_elastic_endpoint: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_q_prev: wp.array[float],
    joint_q: wp.array[float],
    endpoint_terms: ElasticEndpointJointTerms,
):
    """Evaluate every joint at a deformed elastic attachment once per body for the block assembly.

    Launch with one thread per body of ``body_ids``. The joint force and Hessian are expressed in the
    body's axes and also referred to the deformed attachment, which is how they enter the modal rows.
    """
    body = body_ids[wp.tid()]
    elastic_index = body_elastic_index[body]
    if elastic_index < 0:
        return

    q_start = joint_q_start[elastic_joint[elastic_index]] + 7
    mode_count = elastic_mode_count[elastic_index]
    body_R = wp.quat_to_matrix(wp.transform_get_rotation(body_q[body]))
    body_R_T = wp.transpose(body_R)

    adjacent_offset = adjacency.body_adj_joints_offsets[body]
    adjacent_joint_count = get_body_num_adjacent_joints(adjacency, body)
    for adjacent_joint in range(adjacent_joint_count):
        joint = get_body_adjacent_joint_id(adjacency, body, adjacent_joint)
        is_parent = joint_parent[joint] == body
        endpoint = joint_parent_elastic_endpoint[joint] if is_parent else joint_child_elastic_endpoint[joint]
        if endpoint < 0:
            continue

        joint_force, joint_torque, joint_H_ll, joint_H_al, joint_H_aa = evaluate_joint_force_hessian(
            body,
            joint,
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
            body_elastic_index,
            elastic_joint,
            elastic_mode_count,
            joint_q,
            joint_q_prev,
            joint_q_start,
            joint_parent_elastic_endpoint,
            joint_child_elastic_endpoint,
            elastic_endpoint_phi,
            elastic_endpoint_psi,
            elastic_max_mode_count,
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
            joint_target_qd,
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
            stab_alpha,
            joint_compliant_alm,
            joint_dof_dim,
            joint_coordinates,
            dt,
        )

        endpoint_xform = _elastic_endpoint_xform(
            joint,
            body,
            is_parent,
            joint_X_p[joint] if is_parent else joint_X_c[joint],
            body_elastic_index,
            elastic_joint,
            elastic_mode_count,
            joint_q,
            joint_q_start,
            joint_parent_elastic_endpoint,
            joint_child_elastic_endpoint,
            elastic_endpoint_phi,
            elastic_endpoint_psi,
            elastic_max_mode_count,
        )
        endpoint_world = body_q[body] * endpoint_xform
        endpoint_pos_world = wp.transform_get_translation(endpoint_world)
        com_world = wp.transform_point(body_q[body], body_com[body])
        r_world = endpoint_pos_world - com_world
        r_skew = wp.skew(r_world)

        linear_H_aa = wp.transpose(r_skew) * joint_H_ll * r_skew
        angular_torque = joint_torque - wp.cross(r_world, joint_force)
        angular_H_aa = joint_H_aa - linear_H_aa

        theta_local = wp.vec3(0.0)
        for mode in range(mode_count):
            theta_local = (
                theta_local + elastic_endpoint_psi[endpoint * elastic_max_mode_count + mode] * joint_q[q_start + mode]
            )

        entry = adjacent_offset + adjacent_joint
        endpoint_terms.force[entry] = body_R_T * joint_force
        endpoint_terms.torque[entry] = body_R_T * joint_torque
        endpoint_terms.endpoint_torque[entry] = body_R_T * angular_torque
        endpoint_terms.hessian_ll[entry] = body_R_T * (joint_H_ll * body_R)
        endpoint_terms.hessian_al[entry] = body_R_T * (joint_H_al * body_R)
        endpoint_terms.hessian_aa[entry] = body_R_T * (joint_H_aa * body_R)
        endpoint_terms.endpoint_hessian_aa[entry] = body_R_T * (angular_H_aa * body_R)
        endpoint_terms.rotation_jacobian[entry] = _so3_left_jacobian(theta_local)


@wp.kernel
def compute_elastic_mode_sums(
    dt: float,
    body_ids: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    elastic_mode_start: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_max_mode_count: int,
    body_com: wp.array[wp.vec3],
    elastic_mode_coupling_linear: wp.array[wp.vec3],
    elastic_mode_coupling_angular: wp.array[wp.vec3],
    elastic_mode_coupling_coriolis: wp.array[wp.vec3],
    elastic_mode_coupling_quadratic: wp.array[wp.mat33],
    elastic_mode_mass_matrix: wp.array[float],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_q: wp.array[float],
    joint_q_prev: wp.array[float],
    joint_qd_prev: wp.array[float],
    elastic_mode_torque_q: wp.array[wp.vec3],
    elastic_mode_coriolis_q: wp.array[wp.vec3],
    elastic_mode_quadratic_q: wp.array[wp.mat33],
    elastic_mode_mass_offdiagonal: wp.array[float],
):
    """Form the mode-pair sums of the reduced elastic inertia, one thread per body and mode ``i``.

    Launch with ``dim=(len(body_ids), elastic_max_mode_count)``. About the rest center of mass
    ``c``, with the modal coordinates ``q`` of ``joint_q`` and the step-start ``q_n``, ``qd_n``,
    writes per mode:

    - ``elastic_mode_torque_q``: ``G_i(q) = G_i + c x S_i - sum_j C_ij q_j`` [kg m],
    - ``elastic_mode_coriolis_q``: ``sum_j C_ij (q_j - q_n,j) / dt`` [kg/s],
    - ``elastic_mode_quadratic_q``: ``sum_j P_ij q_j`` [kg],
    - ``elastic_mode_mass_offdiagonal``: ``sum_{j != i} Mf_ij (q_j - q_n,j - dt qd_n,j)`` [kg].

    Each thread sums over ``j`` in the order of the per-body loops these sums replace.
    """
    slot, mode = wp.tid()
    body = body_ids[slot]
    elastic_index = body_elastic_index[body]
    if elastic_index < 0:
        return
    mode_count = elastic_mode_count[elastic_index]
    if mode >= mode_count:
        return

    owner_joint = elastic_joint[elastic_index]
    q_start = joint_q_start[owner_joint] + 7
    qd_start = joint_qd_start[owner_joint] + 6
    max_modes = elastic_max_mode_count
    pair_start = (elastic_index * max_modes + mode) * max_modes
    mode_data = elastic_mode_start[elastic_index] + mode
    inv_dt = 1.0 / dt

    coupling_linear = elastic_mode_coupling_linear[mode_data]
    torque = elastic_mode_coupling_angular[mode_data] + wp.cross(body_com[body], coupling_linear)
    coriolis = wp.vec3(0.0)
    quadratic = wp.mat33(0.0)
    mass_offdiagonal = float(0.0)
    for j in range(mode_count):
        pair = pair_start + j
        q_j = joint_q[q_start + j]
        q_prev_j = joint_q_prev[q_start + j]
        coriolis_ij = elastic_mode_coupling_coriolis[pair]
        coriolis = coriolis + coriolis_ij * ((q_j - q_prev_j) * inv_dt)
        torque = torque - coriolis_ij * q_j
        quadratic = quadratic + elastic_mode_coupling_quadratic[pair] * q_j
        if j != mode:
            mass_offdiagonal = mass_offdiagonal + elastic_mode_mass_matrix[pair] * (
                q_j - q_prev_j - dt * joint_qd_prev[qd_start + j]
            )

    elastic_mode_torque_q[mode_data] = torque
    elastic_mode_coriolis_q[mode_data] = coriolis
    elastic_mode_quadratic_q[mode_data] = quadratic
    elastic_mode_mass_offdiagonal[mode_data] = mass_offdiagonal


@wp.func
def _elastic_frame_accel_rel(
    dt: float,
    frame_dynamic: bool,
    body: int,
    body_qd_start: int,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    joint_qd_prev: wp.array[float],
) -> wp.vec3:
    """Return the world acceleration of the rest COM relative to gravity, ``a_c - g``.

    A dynamic frame takes it from the rigid row's displacement from its inertial target, so absolute
    positions cancel; a kinematic frame differences its prescribed poses.
    """
    inv_dt = 1.0 / dt
    body_com_local = body_com[body]
    if frame_dynamic:
        q_current = body_q[body]
        q_inertial = body_inertia_q[body]
        com_current = wp.transform_get_translation(q_current) + wp.quat_rotate(
            wp.transform_get_rotation(q_current), body_com_local
        )
        com_star = wp.transform_get_translation(q_inertial) + wp.quat_rotate(
            wp.transform_get_rotation(q_inertial), body_com_local
        )
        return (com_current - com_star) * (inv_dt * inv_dt) + wp.spatial_top(body_f[body]) * body_inv_mass[body]

    com_v_prev = wp.vec3(
        joint_qd_prev[body_qd_start + 0], joint_qd_prev[body_qd_start + 1], joint_qd_prev[body_qd_start + 2]
    )
    com_v = (
        wp.transform_point(body_q[body], body_com_local) - wp.transform_point(body_q_prev[body], body_com_local)
    ) * inv_dt
    return (com_v - com_v_prev) * inv_dt - gravity[wp.max(body_world[body], 0)]


@wp.func
def _elastic_frame_inertial_rows(
    dt: float,
    body: int,
    body_R: wp.mat33,
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_inertia_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    external_forces: wp.array[wp.vec3],
    external_torques: wp.array[wp.vec3],
    external_hessian_ll: wp.array[wp.mat33],
    external_hessian_al: wp.array[wp.mat33],
    external_hessian_aa: wp.array[wp.mat33],
):
    """Return the rigid inertial and external force, torque and Hessian blocks of a dynamic frame in body axes."""
    inv_dt = 1.0 / dt
    inv_dt_sq = inv_dt * inv_dt
    body_R_T = wp.transpose(body_R)
    body_com_local = body_com[body]
    q_current = body_q[body]
    q_inertial = body_inertia_q[body]
    pos_current = wp.transform_get_translation(q_current)
    rot_current = wp.transform_get_rotation(q_current)
    pos_star = wp.transform_get_translation(q_inertial)
    rot_star = wp.transform_get_rotation(q_inertial)
    com_current = pos_current + wp.quat_rotate(rot_current, body_com_local)
    com_star = pos_star + wp.quat_rotate(rot_star, body_com_local)

    inertial_coeff = body_mass[body] * inv_dt_sq
    frame_force_world = (com_star - com_current) * inertial_coeff + external_forces[body]

    q_delta = wp.mul(wp.quat_inverse(rot_current), rot_star)
    if q_delta[3] < 0.0:
        q_delta = wp.quat(-q_delta[0], -q_delta[1], -q_delta[2], -q_delta[3])
    axis_body, angle_body = wp.quat_to_axis_angle(q_delta)
    frame_torque_local = body_inertia[body] * (axis_body * angle_body * inv_dt_sq)
    frame_torque_world = wp.quat_rotate(rot_current, frame_torque_local) + external_torques[body]

    h_ll_world = external_hessian_ll[body] + inertial_coeff * wp.identity(3, float)
    h_al_world = external_hessian_al[body]
    h_aa_world = external_hessian_aa[body] + body_R * (body_inertia[body] * inv_dt_sq) * body_R_T

    force_local = body_R_T * frame_force_world
    torque_local = body_R_T * frame_torque_world
    h_ll_local = body_R_T * (h_ll_world * body_R)
    h_al_local = body_R_T * (h_al_world * body_R)
    h_aa_local = body_R_T * (h_aa_world * body_R)
    return force_local, torque_local, h_ll_local, h_al_local, h_aa_local


@wp.kernel
def assemble_elastic_blocks(
    dt: float,
    solve_frame: bool,
    body_ids: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    elastic_mode_start: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_mode_mass: wp.array[float],
    elastic_mode_stiffness: wp.array[float],
    elastic_mode_damping: wp.array[float],
    elastic_mode_coupling_linear: wp.array[wp.vec3],
    elastic_mode_coupling_centrifugal: wp.array[wp.mat33],
    elastic_mode_mass_matrix: wp.array[float],
    elastic_mode_torque_q: wp.array[wp.vec3],
    elastic_mode_coriolis_q: wp.array[wp.vec3],
    elastic_mode_quadratic_q: wp.array[wp.mat33],
    elastic_mode_mass_offdiagonal: wp.array[float],
    elastic_implicit_mass_coupling: wp.array[bool],
    adjacency: RigidForceElementAdjacencyInfo,
    joint_parent: wp.array[int],
    joint_parent_elastic_endpoint: wp.array[wp.int32],
    joint_child_elastic_endpoint: wp.array[wp.int32],
    elastic_endpoint_phi: wp.array[wp.vec3],
    elastic_endpoint_psi: wp.array[wp.vec3],
    endpoint_terms: ElasticEndpointJointTerms,
    elastic_max_mode_count: int,
    body_elastic_index: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_inertia_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    external_forces: wp.array[wp.vec3],
    external_torques: wp.array[wp.vec3],
    external_hessian_ll: wp.array[wp.mat33],
    external_hessian_al: wp.array[wp.mat33],
    external_hessian_aa: wp.array[wp.mat33],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_f: wp.array[float],
    joint_q_prev: wp.array[float],
    joint_qd_prev: wp.array[float],
    joint_q: wp.array[float],
    elastic_body_block_grad: wp.array2d[float],
    elastic_body_block_delta: wp.array2d[float],
    elastic_body_block_matrix: wp.array3d[float],
):
    """Assemble the coupled frame/modal block of each reduced elastic body, one thread per column.

    Launch with ``dim=(len(body_ids), elastic_max_mode_count + 6)`` after
    :func:`compute_elastic_mode_sums` at the current iterate and :func:`evaluate_elastic_endpoint_joints`
    for the same bodies.
    Column ``k`` owns ``grad[k]`` and every stored entry ``[r, k]``: the full ``6 x 6`` frame block
    and, for the modal columns, the frame/modal coupling and the upper modal triangle. The lower
    triangle outside the frame block is left zero (see :func:`symmetrize_elastic_blocks`). Rows past
    a body's mode count are identity rows.

    Modal rows: ``sum_j Mf_ij qdd_j + d_i qd_i + k_i q_i = Q_i`` with
    ``Q_i = f_i + S_i . (g_b - a_b) + G_i(q) . alpha_b - Omega^2 : (Bc_i + sum_j q_j P_ij)
    - 2 w_b . sum_j C_ij qd_j``, where ``a_b`` is the rest-COM acceleration, ``Bc_i = B_i - S_i outer c``
    and ``G_i(q) = G_i + c x S_i - sum_j C_ij q_j``, all about the rest COM ``c``. The frame rows add the
    deformation-dependent inertia ``I(q) = I0 + sum_i q_i I1_i + tr(D) 1 - D`` with
    ``D = sum_ij q_i q_j P_ij``.
    """
    slot, column = wp.tid()
    body = body_ids[slot]
    elastic_index = body_elastic_index[body]
    if elastic_index < 0:
        return

    max_modes = elastic_max_mode_count
    block_width = max_modes + 6
    owner_joint = elastic_joint[elastic_index]
    q_start = joint_q_start[owner_joint] + 7
    body_qd_start = joint_qd_start[owner_joint]
    qd_start = body_qd_start + 6
    mode_start = elastic_mode_start[elastic_index]
    mode_count = elastic_mode_count[elastic_index]

    inv_dt = 1.0 / dt
    inv_dt_sq = inv_dt * inv_dt
    body_rot = wp.transform_get_rotation(body_q[body])
    body_R = wp.quat_to_matrix(body_rot)
    body_R_T = wp.transpose(body_R)
    body_com_local = body_com[body]
    frame_dynamic = solve_frame and body_inv_mass[body] > 0.0
    implicit_coupling = frame_dynamic and elastic_implicit_mass_coupling[elastic_index]

    for row in range(block_width):
        elastic_body_block_matrix[elastic_index, row, column] = 0.0
    elastic_body_block_delta[elastic_index, column] = 0.0
    grad = float(0.0)

    adjacent_offset = adjacency.body_adj_joints_offsets[body]
    adjacent_joint_count = get_body_num_adjacent_joints(adjacency, body)
    mode = column - 6

    if column < 6:
        if frame_dynamic:
            force_local, torque_local, h_ll_local, h_al_local, h_aa_local = _elastic_frame_inertial_rows(
                dt,
                body,
                body_R,
                body_q,
                body_mass,
                body_inertia,
                body_inertia_q,
                body_com,
                external_forces,
                external_torques,
                external_hessian_ll,
                external_hessian_al,
                external_hessian_aa,
            )
            axis = column % 3
            if column < 3:
                grad = -force_local[axis]
                for i in range(3):
                    elastic_body_block_matrix[elastic_index, i, column] = h_ll_local[i, axis]
                    elastic_body_block_matrix[elastic_index, 3 + i, column] = h_al_local[i, axis]
            else:
                grad = -torque_local[axis]
                for i in range(3):
                    elastic_body_block_matrix[elastic_index, i, column] = h_al_local[axis, i]
                    elastic_body_block_matrix[elastic_index, 3 + i, column] = h_aa_local[i, axis]

            # Frame mass at the iterate: I(q) - I0 on the rotation block (I0 is assembled above) and
            # the translation-rotation coupling -skew(S q) from y = (r - c) + Phi q.
            s_q = wp.vec3(0.0)
            inertia_delta = wp.mat33(0.0)
            quadratic = wp.mat33(0.0)
            for i in range(mode_count):
                mode_data = mode_start + i
                q = joint_q[q_start + i]
                coupling_linear = elastic_mode_coupling_linear[mode_data]
                coupling_centrifugal = elastic_mode_coupling_centrifugal[mode_data] - wp.outer(
                    coupling_linear, body_com_local
                )
                s_q = s_q + coupling_linear * q
                quadratic = quadratic + elastic_mode_quadratic_q[mode_data] * q
                centrifugal_trace = coupling_centrifugal[0, 0] + coupling_centrifugal[1, 1] + coupling_centrifugal[2, 2]
                inertia_delta = inertia_delta + q * (
                    2.0 * centrifugal_trace * wp.identity(3, float)
                    - coupling_centrifugal
                    - wp.transpose(coupling_centrifugal)
                )
            quadratic_trace = quadratic[0, 0] + quadratic[1, 1] + quadratic[2, 2]
            inertia_delta = inertia_delta + (
                quadratic_trace * wp.identity(3, float) - 0.5 * (quadratic + wp.transpose(quadratic))
            )
            inertia_delta_hessian = inertia_delta * inv_dt_sq
            linear_angular_hessian = -wp.skew(s_q) * inv_dt_sq
            if column >= 3:
                for i in range(3):
                    elastic_body_block_matrix[elastic_index, 3 + i, column] = (
                        elastic_body_block_matrix[elastic_index, 3 + i, column] + inertia_delta_hessian[i, axis]
                    )
            if implicit_coupling:
                for i in range(3):
                    if column >= 3:
                        elastic_body_block_matrix[elastic_index, i, column] = (
                            elastic_body_block_matrix[elastic_index, i, column] + linear_angular_hessian[i, axis]
                        )
                    else:
                        elastic_body_block_matrix[elastic_index, 3 + i, column] = (
                            elastic_body_block_matrix[elastic_index, 3 + i, column] + linear_angular_hessian[axis, i]
                        )

            for adjacent_joint in range(adjacent_joint_count):
                joint = adjacency.body_adj_joints[adjacent_offset + adjacent_joint]
                is_parent = joint_parent[joint] == body
                endpoint = joint_parent_elastic_endpoint[joint] if is_parent else joint_child_elastic_endpoint[joint]
                if endpoint < 0:
                    continue
                entry = adjacent_offset + adjacent_joint
                H_ll_local = endpoint_terms.hessian_ll[entry]
                H_al_local = endpoint_terms.hessian_al[entry]
                if column < 3:
                    grad = grad - endpoint_terms.force[entry][axis]
                    for i in range(3):
                        elastic_body_block_matrix[elastic_index, i, column] = (
                            elastic_body_block_matrix[elastic_index, i, column] + H_ll_local[i, axis]
                        )
                        elastic_body_block_matrix[elastic_index, 3 + i, column] = (
                            elastic_body_block_matrix[elastic_index, 3 + i, column] + H_al_local[i, axis]
                        )
                else:
                    H_aa_local = endpoint_terms.hessian_aa[entry]
                    grad = grad - endpoint_terms.torque[entry][axis]
                    for i in range(3):
                        elastic_body_block_matrix[elastic_index, i, column] = (
                            elastic_body_block_matrix[elastic_index, i, column] + H_al_local[axis, i]
                        )
                        elastic_body_block_matrix[elastic_index, 3 + i, column] = (
                            elastic_body_block_matrix[elastic_index, 3 + i, column] + H_aa_local[i, axis]
                        )
        else:
            elastic_body_block_matrix[elastic_index, column, column] = 1.0
    elif mode < mode_count:
        mode_data = mode_start + mode
        q_idx = q_start + mode
        qd_idx = qd_start + mode
        q = joint_q[q_idx]
        q_prev = joint_q_prev[q_idx]
        v_prev = joint_qd_prev[qd_idx]
        mass = elastic_mode_mass[mode_data]
        stiffness = elastic_mode_stiffness[mode_data]
        damping = elastic_mode_damping[mode_data]
        coupling_linear = elastic_mode_coupling_linear[mode_data]
        coupling_torque = elastic_mode_torque_q[mode_data]
        coupling_centrifugal = elastic_mode_coupling_centrifugal[mode_data] - wp.outer(coupling_linear, body_com_local)

        omega_prev = wp.vec3(
            joint_qd_prev[body_qd_start + 3], joint_qd_prev[body_qd_start + 4], joint_qd_prev[body_qd_start + 5]
        )
        omega = quat_velocity(body_rot, wp.transform_get_rotation(body_q_prev[body]), dt)
        alpha_local = body_R_T * ((omega - omega_prev) * inv_dt)
        omega_local = body_R_T * omega
        omega_skew = wp.skew(omega_local)
        omega_skew_sq = omega_skew * omega_skew
        accel_rel_local = body_R_T * _elastic_frame_accel_rel(
            dt,
            frame_dynamic,
            body,
            body_qd_start,
            body_q,
            body_q_prev,
            body_inertia_q,
            body_inv_mass,
            body_com,
            body_world,
            gravity,
            body_f,
            joint_qd_prev,
        )

        force = (
            joint_f[qd_idx]
            - wp.dot(coupling_linear, accel_rel_local)
            + wp.dot(coupling_torque, alpha_local)
            - wp.ddot(omega_skew_sq, coupling_centrifugal + elastic_mode_quadratic_q[mode_data])
            - 2.0 * wp.dot(omega_local, elastic_mode_coriolis_q[mode_data])
        )

        h = stiffness
        grad = stiffness * q - force
        if mass > 0.0:
            h = h + mass * inv_dt_sq
            grad = grad + mass * inv_dt_sq * (q - q_prev - dt * v_prev)
        if damping > 0.0:
            h = h + damping * inv_dt
            grad = grad + damping * inv_dt * (q - q_prev)
        grad = grad + elastic_mode_mass_offdiagonal[mode_data] * inv_dt_sq
        elastic_body_block_matrix[elastic_index, column, column] = h

        pair_column = elastic_index * max_modes * max_modes + mode
        for i in range(mode):
            elastic_body_block_matrix[elastic_index, 6 + i, column] = (
                elastic_body_block_matrix[elastic_index, 6 + i, column]
                + elastic_mode_mass_matrix[pair_column + i * max_modes] * inv_dt_sq
            )

        if implicit_coupling:
            for axis in range(3):
                elastic_body_block_matrix[elastic_index, axis, column] = (
                    elastic_body_block_matrix[elastic_index, axis, column] + coupling_linear[axis] * inv_dt_sq
                )
                elastic_body_block_matrix[elastic_index, 3 + axis, column] = (
                    elastic_body_block_matrix[elastic_index, 3 + axis, column] - coupling_torque[axis] * inv_dt_sq
                )

        for adjacent_joint in range(adjacent_joint_count):
            joint = adjacency.body_adj_joints[adjacent_offset + adjacent_joint]
            is_parent = joint_parent[joint] == body
            endpoint = joint_parent_elastic_endpoint[joint] if is_parent else joint_child_elastic_endpoint[joint]
            if endpoint < 0:
                continue
            entry = adjacent_offset + adjacent_joint
            force_local = endpoint_terms.force[entry]
            angular_torque_local = endpoint_terms.endpoint_torque[entry]
            H_ll_local = endpoint_terms.hessian_ll[entry]
            angular_H_aa_local = endpoint_terms.endpoint_hessian_aa[entry]
            endpoint_rotation_jacobian = endpoint_terms.rotation_jacobian[entry]
            endpoint_mode_start = endpoint * max_modes
            phi_j_local = elastic_endpoint_phi[endpoint_mode_start + mode]
            psi_j_local = endpoint_rotation_jacobian * elastic_endpoint_psi[endpoint_mode_start + mode]
            grad = grad - wp.dot(force_local, phi_j_local) - wp.dot(angular_torque_local, psi_j_local)

            if frame_dynamic:
                linear_cross = H_ll_local * phi_j_local
                angular_cross = endpoint_terms.hessian_al[entry] * phi_j_local + angular_H_aa_local * psi_j_local
                for axis in range(3):
                    elastic_body_block_matrix[elastic_index, axis, column] = (
                        elastic_body_block_matrix[elastic_index, axis, column] + linear_cross[axis]
                    )
                    elastic_body_block_matrix[elastic_index, 3 + axis, column] = (
                        elastic_body_block_matrix[elastic_index, 3 + axis, column] + angular_cross[axis]
                    )

            linear_j = H_ll_local * phi_j_local
            angular_j = angular_H_aa_local * psi_j_local
            for i in range(mode + 1):
                phi_i_local = elastic_endpoint_phi[endpoint_mode_start + i]
                psi_i_local = endpoint_rotation_jacobian * elastic_endpoint_psi[endpoint_mode_start + i]
                elastic_body_block_matrix[elastic_index, 6 + i, column] = (
                    elastic_body_block_matrix[elastic_index, 6 + i, column]
                    + wp.dot(phi_i_local, linear_j)
                    + wp.dot(psi_i_local, angular_j)
                )
    else:
        elastic_body_block_matrix[elastic_index, column, column] = 1.0

    elastic_body_block_grad[elastic_index, column] = grad
    diagonal = elastic_body_block_matrix[elastic_index, column, column]
    elastic_body_block_matrix[elastic_index, column, column] = diagonal + 1.0e-9 * (wp.abs(diagonal) + 1.0)


@wp.func
def _elastic_excess_momentum(
    rot: wp.quat,
    omega: wp.vec3,
    com: wp.vec3,
    q_start: int,
    qd_start: int,
    mode_start: int,
    mode_count: int,
    elastic_mode_coupling_linear: wp.array[wp.vec3],
    elastic_mode_coupling_centrifugal: wp.array[wp.mat33],
    elastic_mode_torque_q: wp.array[wp.vec3],
    elastic_mode_quadratic_q: wp.array[wp.mat33],
    joint_q: wp.array[float],
    joint_q_ref: wp.array[float],
    joint_qd_ref: wp.array[float],
    inv_dt: float,
    use_difference: bool,
):
    """Return the elastic excess momenta of one reduced elastic body.

    With ``y = (r - c) + Phi q`` the sample offset from the rest center of mass
    ``c``, the world linear momentum is ``m v_c + P_e`` and the angular momentum
    about ``c`` (without the ``s x v_c`` transport term) is ``R I0 w_b + h_e``.

    Args:
        rot: Frame rotation.
        omega: World angular velocity [rad/s].
        com: Rest center of mass ``c`` in the body frame [m].
        q_start: Index of the first modal coordinate in ``joint_q``.
        qd_start: Index of the first modal velocity in ``joint_qd_ref``.
        mode_start: First per-mode coupling index of the body.
        mode_count: Number of modes of the body.
        elastic_mode_coupling_linear: ``S_i`` [kg].
        elastic_mode_coupling_centrifugal: ``B_i`` about the frame origin [kg m].
        elastic_mode_torque_q: ``G_i(q) = G_i + c x S_i - sum_j C_ij q_j`` at ``joint_q`` [kg m]
            (see :func:`compute_elastic_mode_sums`).
        elastic_mode_quadratic_q: ``sum_j P_ij q_j`` at ``joint_q`` [kg].
        joint_q: Modal coordinates of this time level.
        joint_q_ref: Modal coordinates of the step start (used when ``use_difference``).
        joint_qd_ref: Stored modal velocities (used otherwise).
        inv_dt: Inverse time step [1/s].
        use_difference: Use ``(q - q_ref) / dt`` instead of the stored velocity.

    Returns:
        ``P_e = w x (R S q) + R S qdot`` [kg m/s], ``h_e = R [(I(q) - I0) w_b - G(q) qdot]``
        [kg m^2/s], and ``s = R S q`` [kg m], all in world frame.
    """
    rot_matrix = wp.quat_to_matrix(rot)
    omega_local = wp.transpose(rot_matrix) * omega

    s_q = wp.vec3(0.0)
    s_qd = wp.vec3(0.0)
    g_qd = wp.vec3(0.0)
    inertia_delta = wp.mat33(0.0)
    quadratic = wp.mat33(0.0)
    for i in range(mode_count):
        mode_data = mode_start + i
        q_i = joint_q[q_start + i]
        qd_i = joint_qd_ref[qd_start + i]
        if use_difference:
            qd_i = (q_i - joint_q_ref[q_start + i]) * inv_dt
        coupling_linear = elastic_mode_coupling_linear[mode_data]
        s_q += coupling_linear * q_i
        s_qd += coupling_linear * qd_i

        centrifugal = elastic_mode_coupling_centrifugal[mode_data] - wp.outer(coupling_linear, com)
        trace = centrifugal[0, 0] + centrifugal[1, 1] + centrifugal[2, 2]
        inertia_delta += q_i * (2.0 * trace * wp.identity(3, float) - centrifugal - wp.transpose(centrifugal))
        quadratic += elastic_mode_quadratic_q[mode_data] * q_i
        g_qd += elastic_mode_torque_q[mode_data] * qd_i

    quadratic_trace = quadratic[0, 0] + quadratic[1, 1] + quadratic[2, 2]
    inertia_delta += quadratic_trace * wp.identity(3, float) - 0.5 * (quadratic + wp.transpose(quadratic))

    s_world = rot_matrix * s_q
    linear = wp.cross(omega, s_world) + rot_matrix * s_qd
    angular = rot_matrix * (inertia_delta * omega_local - g_qd)
    return linear, angular, s_world


@wp.kernel
def accumulate_elastic_frame_coupling(
    dt: float,
    elastic_body: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    elastic_mode_start: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_mode_coupling_linear: wp.array[wp.vec3],
    elastic_mode_coupling_centrifugal: wp.array[wp.mat33],
    elastic_mode_torque_q: wp.array[wp.vec3],
    elastic_mode_quadratic_q: wp.array[wp.mat33],
    elastic_mode_torque_q_prev: wp.array[wp.vec3],
    elastic_mode_quadratic_q_prev: wp.array[wp.mat33],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_inertia_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_f: wp.array[wp.spatial_vector],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_q: wp.array[float],
    joint_q_prev: wp.array[float],
    joint_qd_prev: wp.array[float],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
):
    """Accumulate the elastic part of the floating-frame momentum balance.

    The rigid frame rows integrate ``m v_c`` and ``R I0 w_b`` about the rest center of
    mass ``c``. This kernel adds the remaining terms of the exact balance laws as
    world wrenches, written as backward differences of the elastic excess momenta
    between the step start and the current iterate:

    ``F = -(P_e[n+1] - P_e[n]) / dt`` and
    ``tau = -(h_e[n+1] - h_e[n]) / dt - s[n+1] x (a_c - g)``,

    see :func:`_elastic_excess_momentum`. ``a_c - g`` is taken from the same
    displacement from the inertial target that the rigid row uses, so absolute
    positions and the frame translational velocity never enter. A free body
    therefore conserves its physical linear momentum and the result does not depend
    on a uniform velocity boost or on the choice of the frame origin.

    Launch with one thread per reduced elastic body after :func:`compute_elastic_mode_sums` at the
    iterate and at the step start (``joint_q = joint_q_prev``), which supply the mode-pair sums, so
    this kernel is linear in the mode count.

    Args:
        dt: Time step [s].
        elastic_body: Body index of each reduced elastic body.
        elastic_joint: Owner joint of each reduced elastic body.
        elastic_mode_start: First per-mode coupling index of each body.
        elastic_mode_count: Number of modes of each body.
        elastic_mode_coupling_linear: ``S_i`` [kg].
        elastic_mode_coupling_centrifugal: ``B_i`` about the frame origin [kg m].
        elastic_mode_torque_q: ``G_i(q)`` at the iterate [kg m].
        elastic_mode_quadratic_q: ``sum_j P_ij q_j`` at the iterate [kg].
        elastic_mode_torque_q_prev: ``G_i(q)`` at the step start [kg m].
        elastic_mode_quadratic_q_prev: ``sum_j P_ij q_j`` at the step start [kg].
        body_q: Current iterate of the body poses.
        body_q_prev: Body poses at the step start.
        body_inertia_q: Inertial target poses of the rigid predictor.
        body_com: Rest center of mass ``c`` in the body frame [m].
        body_inv_mass: Effective inverse body masses [1/kg].
        body_f: Body wrenches applied by the rigid predictor (world, at ``c``) [N, N m].
        joint_q_start: Joint coordinate start indices.
        joint_qd_start: Joint velocity start indices.
        joint_q: Current iterate of the joint coordinates.
        joint_q_prev: Joint coordinates at the step start.
        joint_qd_prev: Joint velocities at the step start.
        body_forces: Accumulated world forces [N].
        body_torques: Accumulated world torques about ``c`` [N m].
    """
    elastic_index = wp.tid()
    body = elastic_body[elastic_index]
    inv_mass = body_inv_mass[body]
    if inv_mass == 0.0:
        return

    owner_joint = elastic_joint[elastic_index]
    q_start = joint_q_start[owner_joint] + 7
    body_qd_start = joint_qd_start[owner_joint]
    qd_start = body_qd_start + 6
    mode_start = elastic_mode_start[elastic_index]
    mode_count = elastic_mode_count[elastic_index]

    inv_dt = 1.0 / dt
    com = body_com[body]
    q_current = body_q[body]
    q_prev = body_q_prev[body]
    rot = wp.transform_get_rotation(q_current)
    rot_prev = wp.transform_get_rotation(q_prev)
    omega = quat_velocity(rot, rot_prev, dt)
    omega_prev = wp.vec3(
        joint_qd_prev[body_qd_start + 3], joint_qd_prev[body_qd_start + 4], joint_qd_prev[body_qd_start + 5]
    )

    linear, angular, s_world = _elastic_excess_momentum(
        rot,
        omega,
        com,
        q_start,
        qd_start,
        mode_start,
        mode_count,
        elastic_mode_coupling_linear,
        elastic_mode_coupling_centrifugal,
        elastic_mode_torque_q,
        elastic_mode_quadratic_q,
        joint_q,
        joint_q_prev,
        joint_qd_prev,
        inv_dt,
        True,
    )
    linear_prev, angular_prev, _s_prev = _elastic_excess_momentum(
        rot_prev,
        omega_prev,
        com,
        q_start,
        qd_start,
        mode_start,
        mode_count,
        elastic_mode_coupling_linear,
        elastic_mode_coupling_centrifugal,
        elastic_mode_torque_q_prev,
        elastic_mode_quadratic_q_prev,
        joint_q_prev,
        joint_q_prev,
        joint_qd_prev,
        inv_dt,
        False,
    )

    # COM acceleration relative to gravity, from the rigid row's own displacement.
    q_inertial = body_inertia_q[body]
    com_current = wp.transform_point(q_current, com)
    com_star = wp.transform_point(q_inertial, com)
    accel_rel = (com_current - com_star) * (inv_dt * inv_dt) + wp.spatial_top(body_f[body]) * inv_mass

    force = -(linear - linear_prev) * inv_dt
    torque = -(angular - angular_prev) * inv_dt - wp.cross(s_world, accel_rel)

    wp.atomic_add(body_forces, body, force)
    wp.atomic_add(body_torques, body, torque)


@wp.kernel
def assemble_elastic_contacts(
    dt: float,
    solve_frame: bool,
    body_ids: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    elastic_max_mode_count: int,
    body_elastic_index: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    rigid_contact_max: int,
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    rigid_contact_elastic_sample0: wp.array[wp.int32],
    rigid_contact_elastic_sample1: wp.array[wp.int32],
    contact_penalty_k: wp.array[float],
    contact_normal_rho: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_tangent_rho: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    stab_alpha: float,
    legacy_hard_contacts: int,
    contact_compliant_alm: int,
    friction_epsilon: float,
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    body_contact_indices: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    elastic_shape_vertex_local: wp.array[wp.vec3],
    elastic_shape_vertex_phi: wp.array[wp.vec3],
    joint_q_prev: wp.array[float],
    joint_q: wp.array[float],
    elastic_body_block_grad: wp.array2d[float],
    elastic_body_block_matrix: wp.array3d[float],
):
    tid = wp.tid()
    body_group_index = tid // _NUM_ELASTIC_CONTACT_THREADS_PER_BODY
    thread_id_within_body = tid % _NUM_ELASTIC_CONTACT_THREADS_PER_BODY

    body = body_ids[body_group_index]
    elastic_index = body_elastic_index[body]
    if elastic_index < 0:
        return

    body_rot = wp.transform_get_rotation(body_q[body])
    body_R = wp.quat_to_matrix(body_rot)
    body_R_T = wp.transpose(body_R)

    contact_limit = rigid_contact_count[0]
    if contact_limit > rigid_contact_max:
        contact_limit = rigid_contact_max

    # Use the rigid per-body list when it is complete; otherwise stride the
    # full contact range so contact overflow never truncates reduced forces.
    use_contact_list = False
    contact_count = wp.int32(0)
    if body_contact_buffer_pre_alloc > 0:
        contact_count = body_contact_counts[body]
        use_contact_list = contact_count <= body_contact_buffer_pre_alloc

    if use_contact_list:
        contact_i = thread_id_within_body
        while contact_i < contact_count:
            contact_idx = body_contact_indices[body * body_contact_buffer_pre_alloc + contact_i]
            if contact_idx < contact_limit:
                _accumulate_elastic_contact_modes(
                    dt,
                    solve_frame,
                    contact_idx,
                    body,
                    elastic_index,
                    body_R,
                    body_R_T,
                    elastic_joint,
                    elastic_mode_count,
                    elastic_max_mode_count,
                    body_elastic_index,
                    body_inv_mass,
                    body_q,
                    body_q_prev,
                    body_com,
                    shape_body,
                    rigid_contact_shape0,
                    rigid_contact_shape1,
                    rigid_contact_point0,
                    rigid_contact_point1,
                    rigid_contact_offset0,
                    rigid_contact_offset1,
                    rigid_contact_normal,
                    rigid_contact_margin0,
                    rigid_contact_margin1,
                    rigid_contact_elastic_sample0,
                    rigid_contact_elastic_sample1,
                    contact_penalty_k,
                    contact_normal_rho,
                    contact_material_ke,
                    contact_material_kd,
                    contact_material_mu,
                    contact_tangent_rho,
                    contact_lambda,
                    contact_C0,
                    stab_alpha,
                    legacy_hard_contacts,
                    contact_compliant_alm,
                    friction_epsilon,
                    joint_q_start,
                    elastic_shape_vertex_local,
                    elastic_shape_vertex_phi,
                    joint_q_prev,
                    joint_q,
                    elastic_body_block_grad,
                    elastic_body_block_matrix,
                )

            contact_i += _NUM_ELASTIC_CONTACT_THREADS_PER_BODY
    else:
        contact_idx = thread_id_within_body
        while contact_idx < contact_limit:
            _accumulate_elastic_contact_modes(
                dt,
                solve_frame,
                contact_idx,
                body,
                elastic_index,
                body_R,
                body_R_T,
                elastic_joint,
                elastic_mode_count,
                elastic_max_mode_count,
                body_elastic_index,
                body_inv_mass,
                body_q,
                body_q_prev,
                body_com,
                shape_body,
                rigid_contact_shape0,
                rigid_contact_shape1,
                rigid_contact_point0,
                rigid_contact_point1,
                rigid_contact_offset0,
                rigid_contact_offset1,
                rigid_contact_normal,
                rigid_contact_margin0,
                rigid_contact_margin1,
                rigid_contact_elastic_sample0,
                rigid_contact_elastic_sample1,
                contact_penalty_k,
                contact_normal_rho,
                contact_material_ke,
                contact_material_kd,
                contact_material_mu,
                contact_tangent_rho,
                contact_lambda,
                contact_C0,
                stab_alpha,
                legacy_hard_contacts,
                contact_compliant_alm,
                friction_epsilon,
                joint_q_start,
                elastic_shape_vertex_local,
                elastic_shape_vertex_phi,
                joint_q_prev,
                joint_q,
                elastic_body_block_grad,
                elastic_body_block_matrix,
            )

            contact_idx += _NUM_ELASTIC_CONTACT_THREADS_PER_BODY


@wp.kernel
def symmetrize_elastic_blocks(
    body_ids: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    elastic_body_block_matrix: wp.array3d[float],
):
    """Mirror the assembled upper triangle of each block into its lower triangle.

    Assembly writes the frame/modal coupling and the modal-modal terms into the upper
    triangle only. Launch with ``dim=(len(body_ids), width, width)`` to symmetrize the
    leading ``width x width`` block of every elastic body in ``body_ids``.
    """
    body_slot, i, j = wp.tid()
    if j >= i:
        return
    elastic_index = body_elastic_index[body_ids[body_slot]]
    if elastic_index < 0:
        return
    elastic_body_block_matrix[elastic_index, i, j] = elastic_body_block_matrix[elastic_index, j, i]


@wp.func
def _elastic_block_entry(
    elastic_body_block_matrix: wp.array3d[float],
    elastic_index: int,
    row: int,
    column: int,
) -> float:
    """Read one entry of a symmetric block from its upper triangle."""
    if column < row:
        return elastic_body_block_matrix[elastic_index, column, row]
    return elastic_body_block_matrix[elastic_index, row, column]


@wp.func
def _store_elastic_block_delta(
    elastic_index: int,
    solution_is_finite: bool,
    elastic_body_block_grad: wp.array2d[float],
    elastic_body_block_matrix: wp.array3d[float],
    row: int,
    negative_solution: float,
    elastic_body_block_delta: wp.array2d[float],
):
    """Store one solved block entry, or its Jacobi fallback when the factorization failed."""
    value = negative_solution
    if not solution_is_finite:
        value = 0.0
        diagonal = elastic_body_block_matrix[elastic_index, row, row]
        if diagonal > 0.0:
            value = -elastic_body_block_grad[elastic_index, row] / diagonal
    elastic_body_block_delta[elastic_index, row] = value


@wp.func
def _apply_elastic_block_delta(
    width: int,
    dt: float,
    solve_frame: bool,
    body: int,
    elastic_index: int,
    elastic_joint: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_q_prev: wp.array[float],
    elastic_body_block_grad: wp.array2d[float],
    elastic_body_block_delta: wp.array2d[float],
    elastic_body_block_matrix: wp.array3d[float],
    elastic_body_block_initial_residual_norm: wp.array[float],
    elastic_body_block_solve_residual_norm: wp.array[float],
    elastic_body_block_applied_residual_norm: wp.array[float],
    elastic_body_block_update_norm: wp.array[float],
    elastic_body_block_update_max: wp.array[float],
    elastic_body_relaxation: float,
    body_q_new: wp.array[wp.transform],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    """Record optional metrics and apply a relaxed block increment to the frame and modes.

    The modal update reads and writes ``joint_q``, so only one thread per body may execute
    this. It reads the increment that the same thread wrote to ``elastic_body_block_delta``,
    so no barrier is required.
    """
    owner_joint = elastic_joint[elastic_index]
    q_start = joint_q_start[owner_joint] + 7
    qd_start = joint_qd_start[owner_joint] + 6
    mode_count = elastic_mode_count[elastic_index]

    if elastic_body_block_initial_residual_norm:
        initial_residual_sq = float(0.0)
        solve_residual_sq = float(0.0)
        applied_residual_sq = float(0.0)
        update_norm_sq = float(0.0)
        update_max = float(0.0)
        for row in range(width):
            if row < 6:
                row_is_solved = solve_frame
            else:
                row_is_solved = row - 6 < mode_count
            if row_is_solved:
                grad_i = elastic_body_block_grad[elastic_index, row]
                solve_residual_i = grad_i
                applied_residual_i = grad_i
                for j in range(width):
                    delta_j = elastic_body_block_delta[elastic_index, j]
                    h_ij = _elastic_block_entry(elastic_body_block_matrix, elastic_index, row, j)
                    solve_residual_i = solve_residual_i + h_ij * delta_j
                    applied_residual_i = applied_residual_i + h_ij * (elastic_body_relaxation * delta_j)

                applied_delta_i = elastic_body_relaxation * elastic_body_block_delta[elastic_index, row]
                initial_residual_sq = initial_residual_sq + grad_i * grad_i
                solve_residual_sq = solve_residual_sq + solve_residual_i * solve_residual_i
                applied_residual_sq = applied_residual_sq + applied_residual_i * applied_residual_i
                update_norm_sq = update_norm_sq + applied_delta_i * applied_delta_i
                update_max = wp.max(update_max, wp.abs(applied_delta_i))

        elastic_body_block_initial_residual_norm[elastic_index] = wp.sqrt(initial_residual_sq)
        elastic_body_block_solve_residual_norm[elastic_index] = wp.sqrt(solve_residual_sq)
        elastic_body_block_applied_residual_norm[elastic_index] = wp.sqrt(applied_residual_sq)
        elastic_body_block_update_norm[elastic_index] = wp.sqrt(update_norm_sq)
        elastic_body_block_update_max[elastic_index] = update_max

    inv_dt = 1.0 / dt
    q_current = body_q[body]
    if solve_frame and body_inv_mass[body] > 0.0:
        rot_current = wp.transform_get_rotation(q_current)
        body_R = wp.quat_to_matrix(rot_current)
        x_delta = wp.vec3(
            elastic_body_block_delta[elastic_index, 0],
            elastic_body_block_delta[elastic_index, 1],
            elastic_body_block_delta[elastic_index, 2],
        )
        w_delta = wp.vec3(
            elastic_body_block_delta[elastic_index, 3],
            elastic_body_block_delta[elastic_index, 4],
            elastic_body_block_delta[elastic_index, 5],
        )
        x_inc_world = body_R * (elastic_body_relaxation * x_delta)
        w_world = body_R * (elastic_body_relaxation * w_delta)
        angle = wp.length(w_world)
        if angle > _SMALL_ANGLE_EPS:
            dq_world = wp.quat_from_axis_angle(w_world / angle, angle)
        else:
            half_w = 0.5 * w_world
            dq_world = wp.normalize(wp.quat(half_w[0], half_w[1], half_w[2], 1.0))
        rot_new = wp.normalize(dq_world * rot_current)
        com_current = wp.transform_point(q_current, body_com[body])
        com_new = com_current + x_inc_world
        pos_new = com_new - wp.quat_rotate(rot_new, body_com[body])
        body_q_new[body] = wp.transform(pos_new, rot_new)
    else:
        body_q_new[body] = q_current

    for mode in range(mode_count):
        q_idx = q_start + mode
        qd_idx = qd_start + mode
        q_prev = joint_q_prev[q_idx]
        q_old = joint_q[q_idx]
        q_new = q_old + elastic_body_relaxation * elastic_body_block_delta[elastic_index, 6 + mode]
        joint_q[q_idx] = q_new
        joint_qd[qd_idx] = (q_new - q_prev) * inv_dt


def elastic_block_solve_tiled_block_dim(block_width: int) -> int:
    """Return the CUDA block size of the tiled block solve for ``block_width``.

    The tiled Cholesky slows down by one to two orders of magnitude once the block width
    exceeds a few times the thread count, so wider blocks use larger thread blocks. Measured
    on an RTX PRO 6000 Blackwell GPU with Warp 1.18; a width whose block, at this size, does
    not fit the shared memory left after the per-thread static reservation uses the dense
    solve instead of a smaller, much slower thread block.
    """
    if block_width <= 70:
        return 32
    if block_width <= 126:
        return 128
    return 256


@functools.cache
def create_solve_elastic_body_tiled(block_width: int):
    """Create the coupled frame/modal block solve kernel for one block width.

    The kernel solves the leading ``block_width x block_width`` sub-block of each body's
    ``elastic_body_block_matrix`` entry, which may be stored with a larger width. The
    sub-block must hold the full symmetric matrix (see :func:`symmetrize_elastic_blocks`);
    rows past the body's modal count must be identity rows. The block and the right-hand
    side are loaded into shared memory once, factorized and solved in place, so the
    kernel needs about ``4 * (block_width**2 + block_width)`` bytes of dynamic shared
    memory. Blocks that are not positive definite fall back to a Jacobi step.
    """
    width = int(block_width)

    def _solve_elastic_body_tiled(
        dt: float,
        solve_frame: bool,
        body_ids: wp.array[wp.int32],
        body_elastic_index: wp.array[wp.int32],
        elastic_joint: wp.array[wp.int32],
        elastic_mode_count: wp.array[wp.int32],
        body_inv_mass: wp.array[float],
        body_com: wp.array[wp.vec3],
        body_q: wp.array[wp.transform],
        joint_q_start: wp.array[wp.int32],
        joint_qd_start: wp.array[wp.int32],
        joint_q_prev: wp.array[float],
        elastic_body_block_grad: wp.array2d[float],
        elastic_body_block_delta: wp.array2d[float],
        elastic_body_block_matrix: wp.array3d[float],
        elastic_body_block_initial_residual_norm: wp.array[float],
        elastic_body_block_solve_residual_norm: wp.array[float],
        elastic_body_block_applied_residual_norm: wp.array[float],
        elastic_body_block_update_norm: wp.array[float],
        elastic_body_block_update_max: wp.array[float],
        elastic_body_relaxation: float,
        body_q_new: wp.array[wp.transform],
        joint_q: wp.array[float],
        joint_qd: wp.array[float],
    ):
        slot, lane = wp.tid()
        body = body_ids[slot]
        elastic_index = body_elastic_index[body]
        if elastic_index < 0:
            return

        factor = wp.tile_load(elastic_body_block_matrix[elastic_index], shape=(width, width), storage="shared")
        solution = wp.tile_load(elastic_body_block_grad[elastic_index], shape=width, storage="shared")
        wp.tile_cholesky_inplace(factor)
        wp.tile_cholesky_solve_inplace(factor, solution)

        # The update increments joint_q in place: one thread per body, or threads of a
        # multi-warp block that read an already updated coordinate would apply it twice.
        if lane != 0:
            return
        solution_is_finite = bool(True)
        for i in range(width):
            if not wp.isfinite(solution[i]):
                solution_is_finite = False
        for i in range(width):
            _store_elastic_block_delta(
                elastic_index,
                solution_is_finite,
                elastic_body_block_grad,
                elastic_body_block_matrix,
                i,
                -solution[i],
                elastic_body_block_delta,
            )

        _apply_elastic_block_delta(
            width,
            dt,
            solve_frame,
            body,
            elastic_index,
            elastic_joint,
            elastic_mode_count,
            body_inv_mass,
            body_com,
            body_q,
            joint_q_start,
            joint_qd_start,
            joint_q_prev,
            elastic_body_block_grad,
            elastic_body_block_delta,
            elastic_body_block_matrix,
            elastic_body_block_initial_residual_norm,
            elastic_body_block_solve_residual_norm,
            elastic_body_block_applied_residual_norm,
            elastic_body_block_update_norm,
            elastic_body_block_update_max,
            elastic_body_relaxation,
            body_q_new,
            joint_q,
            joint_qd,
        )

    _solve_elastic_body_tiled.__name__ = f"solve_elastic_body_tiled_{width}"
    _solve_elastic_body_tiled.__qualname__ = f"solve_elastic_body_tiled_{width}"
    return wp.kernel(enable_backward=False, module="unique")(_solve_elastic_body_tiled)


ELASTIC_BLOCK_DENSE_PANEL = 32
"""Panel size of the dense reduced elastic block solve."""

ELASTIC_BLOCK_DENSE_BLOCK_DIM = 128
"""CUDA threads per block of the dense reduced elastic block-solve kernels."""


def elastic_block_dense_width(block_width: int) -> int:
    """Return ``block_width`` rounded up to whole dense-solve panels."""
    panel = ELASTIC_BLOCK_DENSE_PANEL
    return (int(block_width) + panel - 1) // panel * panel


@wp.func
def _elastic_dense_lower_panel(pair: int) -> wp.vec2i:
    """Map a linear index to the ``(row, column)`` panel of a lower block triangle, row-major."""
    remaining = int(pair)
    row = int(0)
    while remaining > row:
        remaining -= row + 1
        row += 1
    return wp.vec2i(row, remaining)


@wp.func
def _elastic_dense_trailing_panel(pair: int, step: int, panel_count: int) -> wp.vec2i:
    """Map a linear index to the ``(row, column)`` panel of the trailing lower triangle after ``step``."""
    remaining = int(pair)
    column = int(step + 1)
    while remaining >= panel_count - column:
        remaining -= panel_count - column
        column += 1
    return wp.vec2i(column + remaining, column)


@wp.kernel(enable_backward=False)
def copy_elastic_blocks_to_dense_factor(
    body_ids: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    elastic_body_block_matrix: wp.array3d[float],
    factor: wp.array3d[float],
):
    """Copy the lower panel triangle of each body's block into its dense factor slot.

    Launch tiled with ``dim=(len(body_ids), panel_count * (panel_count + 1) // 2)``.
    """
    slot, pair = wp.tid()
    elastic_index = body_elastic_index[body_ids[slot]]
    if elastic_index < 0:
        return
    panel = _elastic_dense_lower_panel(pair)
    offset = (panel[0] * ELASTIC_BLOCK_DENSE_PANEL, panel[1] * ELASTIC_BLOCK_DENSE_PANEL)
    block = wp.tile_load(
        elastic_body_block_matrix[elastic_index],
        shape=(ELASTIC_BLOCK_DENSE_PANEL, ELASTIC_BLOCK_DENSE_PANEL),
        offset=offset,
        storage="shared",
    )
    wp.tile_store(factor[slot], block, offset=offset)


@wp.kernel(enable_backward=False)
def factor_elastic_dense_panel_column(
    body_ids: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    step: int,
    panel_count: int,
    factor: wp.array3d[float],
):
    """Factor diagonal panel ``step`` and solve the panels below it (one block per body).

    Overwrites ``A[step, step]`` with ``L[step, step]`` and every ``A[i, step]`` below it with
    ``L[i, step] = A[i, step] L[step, step]^-T``. The diagonal factor stays in shared memory,
    so no panel is read back after being stored in the same launch.
    """
    slot = wp.tid()
    if body_elastic_index[body_ids[slot]] < 0:
        return
    origin = step * ELASTIC_BLOCK_DENSE_PANEL
    diagonal = wp.tile_load(
        factor[slot],
        shape=(ELASTIC_BLOCK_DENSE_PANEL, ELASTIC_BLOCK_DENSE_PANEL),
        offset=(origin, origin),
        storage="shared",
    )
    wp.tile_cholesky_inplace(diagonal)
    wp.tile_store(factor[slot], diagonal, offset=(origin, origin))
    for row in range(step + 1, panel_count):
        below = wp.tile_load(
            factor[slot],
            shape=(ELASTIC_BLOCK_DENSE_PANEL, ELASTIC_BLOCK_DENSE_PANEL),
            offset=(row * ELASTIC_BLOCK_DENSE_PANEL, origin),
            storage="shared",
        )
        solved = wp.tile_lower_solve(diagonal, wp.tile_transpose(below))
        wp.tile_store(factor[slot], wp.tile_transpose(solved), offset=(row * ELASTIC_BLOCK_DENSE_PANEL, origin))


@wp.kernel(enable_backward=False)
def update_elastic_dense_trailing_panels(
    body_ids: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    step: int,
    panel_count: int,
    factor: wp.array3d[float],
):
    """Apply the rank-32 update of panel column ``step`` to one trailing lower panel.

    Launch tiled with ``dim=(len(body_ids), trailing_panel_count)``; every block updates
    ``A[i, j] -= L[i, step] L[j, step]^T`` for its own ``(i, j)`` with ``i >= j > step``.
    """
    slot, pair = wp.tid()
    if body_elastic_index[body_ids[slot]] < 0:
        return
    panel = _elastic_dense_trailing_panel(pair, step, panel_count)
    row = panel[0] * ELASTIC_BLOCK_DENSE_PANEL
    column = panel[1] * ELASTIC_BLOCK_DENSE_PANEL
    origin = step * ELASTIC_BLOCK_DENSE_PANEL
    shape = (ELASTIC_BLOCK_DENSE_PANEL, ELASTIC_BLOCK_DENSE_PANEL)
    target = wp.tile_load(factor[slot], shape=shape, offset=(row, column), storage="shared")
    left = wp.tile_load(factor[slot], shape=shape, offset=(row, origin), storage="shared")
    right = wp.tile_load(factor[slot], shape=shape, offset=(column, origin), storage="shared")
    wp.tile_matmul(left, wp.tile_transpose(right), target, alpha=-1.0)
    wp.tile_store(factor[slot], target, offset=(row, column))


@functools.cache
def create_solve_elastic_body_dense(block_width: int):
    """Create the substitution and update kernel of the dense block solve for one block width.

    The kernel reads the panel Cholesky factor produced by
    :func:`copy_elastic_blocks_to_dense_factor`, :func:`factor_elastic_dense_panel_column` and
    :func:`update_elastic_dense_trailing_panels`, solves for the block increment by forward and
    backward substitution over 32x32 panels with the right-hand side held in shared memory, and
    then applies the same Jacobi fallback, metrics and frame/modal update as the tiled solve.
    Rows between ``block_width`` and the panel-rounded width must be identity rows of the
    stored block.
    """
    width = int(block_width)
    dense_width = elastic_block_dense_width(width)
    panel = ELASTIC_BLOCK_DENSE_PANEL
    panel_count = dense_width // panel

    def _solve_elastic_body_dense(
        dt: float,
        solve_frame: bool,
        body_ids: wp.array[wp.int32],
        body_elastic_index: wp.array[wp.int32],
        elastic_joint: wp.array[wp.int32],
        elastic_mode_count: wp.array[wp.int32],
        body_inv_mass: wp.array[float],
        body_com: wp.array[wp.vec3],
        body_q: wp.array[wp.transform],
        joint_q_start: wp.array[wp.int32],
        joint_qd_start: wp.array[wp.int32],
        joint_q_prev: wp.array[float],
        elastic_body_block_grad: wp.array2d[float],
        elastic_body_block_delta: wp.array2d[float],
        elastic_body_block_matrix: wp.array3d[float],
        elastic_body_block_initial_residual_norm: wp.array[float],
        elastic_body_block_solve_residual_norm: wp.array[float],
        elastic_body_block_applied_residual_norm: wp.array[float],
        elastic_body_block_update_norm: wp.array[float],
        elastic_body_block_update_max: wp.array[float],
        elastic_body_relaxation: float,
        factor: wp.array3d[float],
        elastic_body_block_grad_column: wp.array3d[float],
        body_q_new: wp.array[wp.transform],
        joint_q: wp.array[float],
        joint_qd: wp.array[float],
    ):
        slot, lane = wp.tid()
        body = body_ids[slot]
        elastic_index = body_elastic_index[body]
        if elastic_index < 0:
            return

        solution = wp.tile_load(elastic_body_block_grad_column[elastic_index], shape=(dense_width, 1), storage="shared")
        # Forward substitution L y = g.
        for k in range(panel_count):
            solution_k = wp.tile_view(solution, offset=(k * panel, 0), shape=(panel, 1))
            for j in range(k):
                factor_kj = wp.tile_load(
                    factor[slot], shape=(panel, panel), offset=(k * panel, j * panel), storage="shared"
                )
                solution_j = wp.tile_view(solution, offset=(j * panel, 0), shape=(panel, 1))
                wp.tile_matmul(factor_kj, solution_j, solution_k, alpha=-1.0)
            factor_kk = wp.tile_load(
                factor[slot], shape=(panel, panel), offset=(k * panel, k * panel), storage="shared"
            )
            wp.tile_lower_solve_inplace(factor_kk, solution_k)
        # Backward substitution L^T x = y.
        for step in range(panel_count):
            k = panel_count - 1 - step
            solution_k = wp.tile_view(solution, offset=(k * panel, 0), shape=(panel, 1))
            for j in range(k + 1, panel_count):
                factor_jk = wp.tile_load(
                    factor[slot], shape=(panel, panel), offset=(j * panel, k * panel), storage="shared"
                )
                solution_j = wp.tile_view(solution, offset=(j * panel, 0), shape=(panel, 1))
                wp.tile_matmul(wp.tile_transpose(factor_jk), solution_j, solution_k, alpha=-1.0)
            factor_kk = wp.tile_load(
                factor[slot], shape=(panel, panel), offset=(k * panel, k * panel), storage="shared"
            )
            wp.tile_upper_solve_inplace(wp.tile_transpose(factor_kk), solution_k)

        # Single-thread update, as in the tiled solve.
        if lane != 0:
            return
        solution_is_finite = bool(True)
        for i in range(width):
            if not wp.isfinite(solution[i, 0]):
                solution_is_finite = False
        for i in range(width):
            _store_elastic_block_delta(
                elastic_index,
                solution_is_finite,
                elastic_body_block_grad,
                elastic_body_block_matrix,
                i,
                -solution[i, 0],
                elastic_body_block_delta,
            )

        _apply_elastic_block_delta(
            width,
            dt,
            solve_frame,
            body,
            elastic_index,
            elastic_joint,
            elastic_mode_count,
            body_inv_mass,
            body_com,
            body_q,
            joint_q_start,
            joint_qd_start,
            joint_q_prev,
            elastic_body_block_grad,
            elastic_body_block_delta,
            elastic_body_block_matrix,
            elastic_body_block_initial_residual_norm,
            elastic_body_block_solve_residual_norm,
            elastic_body_block_applied_residual_norm,
            elastic_body_block_update_norm,
            elastic_body_block_update_max,
            elastic_body_relaxation,
            body_q_new,
            joint_q,
            joint_qd,
        )

    _solve_elastic_body_dense.__name__ = f"solve_elastic_body_dense_{width}"
    _solve_elastic_body_dense.__qualname__ = f"solve_elastic_body_dense_{width}"
    # Keep the panel loops dynamic: Warp reserves shared memory for every tile expression, so
    # unrolling them would allocate one set of panel tiles per iteration.
    return wp.kernel(enable_backward=False, module="unique", module_options={"max_unroll": 1})(
        _solve_elastic_body_dense
    )


def launch_solve_elastic_body_dense(
    block_width: int,
    body_ids: wp.array,
    body_elastic_index: wp.array,
    elastic_body_block_matrix: wp.array,
    factor: wp.array,
    solve_inputs: list,
    solve_outputs: list,
    device,
    block_dim: int = ELASTIC_BLOCK_DENSE_BLOCK_DIM,
):
    """Launch the dense block solve for the bodies in ``body_ids``.

    The block matrix must be symmetric and stored with at least
    :func:`elastic_block_dense_width` rows and columns, with identity rows past
    ``block_width``. ``factor`` is scratch storage of shape ``(len(body_ids), W_d, W_d)``
    where ``W_d`` is the panel-rounded width. ``solve_inputs`` and ``solve_outputs`` are the
    arguments of :func:`create_solve_elastic_body_dense`.
    """
    body_count = body_ids.shape[0]
    if body_count == 0:
        return
    panel_count = elastic_block_dense_width(block_width) // ELASTIC_BLOCK_DENSE_PANEL
    wp.launch_tiled(
        copy_elastic_blocks_to_dense_factor,
        dim=[body_count, panel_count * (panel_count + 1) // 2],
        inputs=[body_ids, body_elastic_index, elastic_body_block_matrix],
        outputs=[factor],
        block_dim=block_dim,
        device=device,
    )
    for step in range(panel_count):
        wp.launch_tiled(
            factor_elastic_dense_panel_column,
            dim=[body_count],
            inputs=[body_ids, body_elastic_index, step, panel_count],
            outputs=[factor],
            block_dim=block_dim,
            device=device,
        )
        trailing_count = (panel_count - step - 1) * (panel_count - step) // 2
        if trailing_count > 0:
            wp.launch_tiled(
                update_elastic_dense_trailing_panels,
                dim=[body_count, trailing_count],
                inputs=[body_ids, body_elastic_index, step, panel_count],
                outputs=[factor],
                block_dim=block_dim,
                device=device,
            )
    wp.launch_tiled(
        create_solve_elastic_body_dense(block_width),
        dim=[body_count],
        inputs=solve_inputs,
        outputs=solve_outputs,
        block_dim=block_dim,
        device=device,
    )


@wp.kernel
def copy_body_frame_to_elastic_joint(
    elastic_body: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    elastic_index = wp.tid()
    body = elastic_body[elastic_index]
    joint = elastic_joint[elastic_index]
    q_start = joint_q_start[joint]
    qd_start = joint_qd_start[joint]

    X_wb = body_q[body]
    p = wp.transform_get_translation(X_wb)
    q = wp.transform_get_rotation(X_wb)
    v = body_qd[body]
    lin = wp.spatial_top(v)
    ang = wp.spatial_bottom(v)

    joint_q[q_start + 0] = p[0]
    joint_q[q_start + 1] = p[1]
    joint_q[q_start + 2] = p[2]
    joint_q[q_start + 3] = q[0]
    joint_q[q_start + 4] = q[1]
    joint_q[q_start + 5] = q[2]
    joint_q[q_start + 6] = q[3]

    joint_qd[qd_start + 0] = lin[0]
    joint_qd[qd_start + 1] = lin[1]
    joint_qd[qd_start + 2] = lin[2]
    joint_qd[qd_start + 3] = ang[0]
    joint_qd[qd_start + 4] = ang[1]
    joint_qd[qd_start + 5] = ang[2]
