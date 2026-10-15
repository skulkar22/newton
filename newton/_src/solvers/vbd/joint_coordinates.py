# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Continuous scalar joint coordinates shared by VBD drives, limits, and mimics."""

import warp as wp

from ...sim import JointType, Model, State
from ...sim.joint_coordinates import eval_joint_coordinate_in_frames, eval_joint_velocity, unwrap_joint_coordinate

wp.set_module_options({"enable_backward": False})

_Vec6 = wp.types.vector(length=6, dtype=wp.float32)
_SMALL_ANGLE_EPS = wp.constant(1.0e-7)


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
    # Reduced elastic endpoints: joints attached to an elastic body measure their coordinate at the
    # deformed attachment frame. ``elastic_q`` holds the modal coordinates that pair with the current
    # body poses and ``elastic_q_prev`` those that pair with the previous-step poses.
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


@wp.func
def eval_elastic_endpoint_xform(
    joint_index: int,
    body_index: int,
    is_parent_side: bool,
    xform_rest: wp.transform,
    body_elastic_index: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    joint_q: wp.array[float],
    joint_q_start: wp.array[int],
    joint_parent_elastic_endpoint: wp.array[wp.int32],
    joint_child_elastic_endpoint: wp.array[wp.int32],
    elastic_endpoint_phi: wp.array[wp.vec3],
    elastic_endpoint_psi: wp.array[wp.vec3],
    elastic_max_mode_count: int,
):
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
        q = wp.quat_from_axis_angle(theta / angle, angle) * q

    return wp.transform(p, q)


@wp.func
def _coordinate_frame(
    data: JointCoordinateData, joint: int, body: int, is_parent_side: bool, X: wp.transform, elastic_q: wp.array[float]
):
    if data.elastic_max_mode_count == 0:
        return X
    return eval_elastic_endpoint_xform(
        joint,
        body,
        is_parent_side,
        X,
        data.body_elastic_index,
        data.elastic_joint,
        data.elastic_mode_count,
        elastic_q,
        data.q_start,
        data.parent_elastic_endpoint,
        data.child_elastic_endpoint,
        data.elastic_endpoint_phi,
        data.elastic_endpoint_psi,
        data.elastic_max_mode_count,
    )


@wp.func
def evaluate_coordinate(
    data: JointCoordinateData,
    joint: int,
    component: int,
    body_q: wp.array[wp.transform],
    elastic_q: wp.array[float],
):
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
    return q, g_p, g_c


@wp.func
def evaluate_coordinate(data: JointCoordinateData, joint: int, component: int, body_q: wp.array[wp.transform]):
    """Evaluate a coordinate at ``body_q`` with the current modal coordinates ``data.elastic_q``."""
    return evaluate_coordinate(data, joint, component, body_q, data.elastic_q)


@wp.func
def evaluate_coordinate_prev(
    data: JointCoordinateData, joint: int, component: int, body_q_prev: wp.array[wp.transform]
):
    """Evaluate a coordinate at previous-step poses with the previous modal coordinates."""
    return evaluate_coordinate(data, joint, component, body_q_prev, data.elastic_q_prev)


@wp.kernel
def _begin_step(
    data: JointCoordinateData,
    joints: wp.array[int],
    body_q: wp.array[wp.transform],
    joint_q_seed: wp.array[float],
    rebaseline_mask: wp.array[bool],
    mimic_multipliers: wp.array[float],
):
    joint = joints[wp.tid()]
    start = data.q_start[joint]
    world = data.world[joint]
    slot = wp.where(world >= 0, world, rebaseline_mask.shape[0] - 1)
    count = data.dof_dim[joint, 0] + data.dof_dim[joint, 1]
    if rebaseline_mask[slot]:
        for component in range(count):
            data.history[start + component] = joint_q_seed[start + component]
    coordinates = _Vec6()
    for component in range(count):
        q, _g_p, _g_c = evaluate_coordinate(data, joint, component, body_q)
        coordinates[component] = q
        if mimic_multipliers:
            mimic_multipliers[data.qd_start[joint] + component] = 0.0
    for component in range(count):
        data.history[start + component] = coordinates[component]


@wp.kernel
def _end_step(
    data: JointCoordinateData,
    joints: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_q_mirror: wp.array[float],
    joint_qd_mirror: wp.array[float],
):
    joint = joints[wp.tid()]
    coordinates = _Vec6()
    for component in range(data.dof_dim[joint, 0] + data.dof_dim[joint, 1]):
        q, g_p, g_c = evaluate_coordinate(data, joint, component, body_q)
        qd = eval_joint_velocity(data.parent[joint], data.child[joint], g_p, g_c, body_qd)
        qi = data.q_start[joint] + component
        vi = data.qd_start[joint] + component
        coordinates[component] = q
        if joint_q:
            joint_q[qi] = q
        if joint_qd:
            joint_qd[vi] = qd
        if joint_q_mirror:
            joint_q_mirror[qi] = q
        if joint_qd_mirror:
            joint_qd_mirror[vi] = qd
    for component in range(data.dof_dim[joint, 0] + data.dof_dim[joint, 1]):
        data.history[data.q_start[joint] + component] = coordinates[component]


@wp.kernel
def _reset_coordinates(
    data: JointCoordinateData,
    joints: wp.array[int],
    model_joint_q: wp.array[float],
    model_joint_qd: wp.array[float],
    world_mask: wp.array[bool],
    reset_all: bool,
    world_count: int,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    joint = joints[wp.tid()]
    slot = wp.where(data.world[joint] >= 0, data.world[joint], world_count)
    selected = reset_all
    if not reset_all and slot < world_mask.shape[0]:
        selected = world_mask[slot]
    if selected:
        start = data.q_start[joint]
        for component in range(data.dof_dim[joint, 0] + data.dof_dim[joint, 1]):
            if joint_q:
                joint_q[start + component] = model_joint_q[start + component]
            if joint_qd:
                dof = data.qd_start[joint] + component
                joint_qd[dof] = model_joint_qd[dof]


class JointCoordinates:
    """Own continuous coordinates independently of any particular joint feature.

    History is fixed throughout each solve and committed from its final poses.
    Initial/reset turn counts come from State.joint_q; disabled joints are still
    tracked. Each angular coordinate must move less than pi per timestep.
    """

    def __init__(self, model: Model):
        self.model = model
        self.data = JointCoordinateData()
        self.data.body_com = model.body_com
        for field, source in (
            ("joint_type", "joint_type"),
            ("parent", "joint_parent"),
            ("child", "joint_child"),
            ("X_p", "joint_X_p"),
            ("X_c", "joint_X_c"),
            ("q_start", "joint_q_start"),
            ("qd_start", "joint_qd_start"),
            ("dof_dim", "joint_dof_dim"),
            ("axis", "joint_axis"),
            ("world", "joint_world"),
        ):
            setattr(self.data, field, getattr(model, source))
        supported = (JointType.PRISMATIC, JointType.REVOLUTE, JointType.D6)
        joints = [j for j, joint_type in enumerate(model.joint_type.numpy()) if joint_type in supported]
        self.joints = wp.array(joints, dtype=int, device=model.device)
        self.data.history = wp.clone(model.joint_q) if joints else wp.empty(0, dtype=float, device=model.device)
        elastic = bool(joints) and int(model.elastic_body_count) > 0
        self.data.elastic_max_mode_count = int(model.elastic_max_mode_count) if elastic else 0
        if elastic:
            for field, source in (
                ("body_elastic_index", "body_elastic_index"),
                ("elastic_joint", "elastic_joint"),
                ("elastic_mode_count", "elastic_mode_count"),
                ("parent_elastic_endpoint", "joint_parent_elastic_endpoint"),
                ("child_elastic_endpoint", "joint_child_elastic_endpoint"),
                ("elastic_endpoint_phi", "elastic_endpoint_phi"),
                ("elastic_endpoint_psi", "elastic_endpoint_psi"),
            ):
                setattr(self.data, field, getattr(model, source))
            self.set_elastic_coordinates(model.joint_q, model.joint_q)

    @property
    def has_elastic_endpoints(self) -> bool:
        """Whether coordinates depend on reduced elastic modal coordinates."""
        return self.data.elastic_max_mode_count > 0

    # Quote host-side array unions because Warp array annotations are not types on Python 3.10.
    def set_elastic_coordinates(self, joint_q: "wp.array[float] | None", joint_q_prev: "wp.array[float] | None"):
        """Select the modal coordinates that pair with current and previous body poses.

        Reduced elastic endpoints deform with these coordinates. Kernels receive
        :attr:`data` by value, so call this before launching kernels that evaluate coordinates.
        """
        if self.has_elastic_endpoints and joint_q is not None:
            self.data.elastic_q = joint_q
            self.data.elastic_q_prev = joint_q_prev if joint_q_prev is not None else joint_q

    def begin_step(self, state: State, rebaseline_mask: wp.array[bool], mimic_multipliers: "wp.array[float] | None"):
        if self.joints.size:
            wp.launch(
                _begin_step,
                dim=self.joints.size,
                inputs=[
                    self.data,
                    self.joints,
                    state.body_q,
                    state.joint_q if state.joint_q is not None else self.model.joint_q,
                    rebaseline_mask,
                ],
                outputs=[mimic_multipliers],
                device=self.model.device,
            )

    def end_step(self, state_in: State, state_out: State):
        if self.joints.size:
            wp.launch(
                _end_step,
                dim=self.joints.size,
                inputs=[self.data, self.joints, state_out.body_q, state_out.body_qd],
                outputs=[state_out.joint_q, state_out.joint_qd, state_in.joint_q, state_in.joint_qd],
                device=self.model.device,
            )

    def reset_coordinates(
        self, joint_q: "wp.array[float] | None", joint_qd: "wp.array[float] | None", world_mask: "wp.array[bool] | None"
    ):
        if self.joints.size:
            wp.launch(
                _reset_coordinates,
                dim=self.joints.size,
                inputs=[
                    self.data,
                    self.joints,
                    self.model.joint_q,
                    self.model.joint_qd,
                    world_mask,
                    world_mask is None,
                    self.model.world_count,
                ],
                outputs=[joint_q, joint_qd],
                device=self.model.device,
            )
