# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""VBD mimic constraints using the systems of the body update and force multipliers."""

import numpy as np
import warp as wp

from ...sim import JointType, Model
from .joint_coordinates import JointCoordinateData, evaluate_coordinate
from .rigid_sparse_articulation import RigidArticulationSparseLayout
from .rigid_sparse_articulation_kernels import _solve_lower66, _solve_upper66_from_lower, mat66f, vec6f
from .rigid_vbd_kernels import ldlt6_solve

wp.set_module_options({"enable_backward": False})

_Mat46 = wp.types.matrix(shape=(4, 6), dtype=wp.float32)


@wp.struct
class _JointData:
    coordinates: JointCoordinateData
    enabled: wp.array[bool]
    reference: wp.array[int]
    coeffs: wp.array[wp.vec2]


@wp.func
def _evaluate_row(data: _JointData, follower: int, component: int, body_q: wp.array[wp.transform]):
    reference = data.reference[follower]
    q_f, g_fp, g_fc = evaluate_coordinate(data.coordinates, follower, component, body_q)
    q_r, g_rp, g_rc = evaluate_coordinate(data.coordinates, reference, component, body_q)
    offset, multiplier = data.coeffs[follower][0], data.coeffs[follower][1]
    error = q_f - offset - multiplier * q_r
    bodies = wp.vec4i(
        data.coordinates.parent[follower],
        data.coordinates.child[follower],
        data.coordinates.parent[reference],
        data.coordinates.child[reference],
    )
    gradients = wp.matrix_from_rows(g_fp, g_fc, -multiplier * g_rp, -multiplier * g_rc)
    # Shared parents and serial joints must use the SUM of their gradients
    # before evaluating J H^-1 J^T; otherwise the cross terms are lost.
    for i in range(4):
        if bodies[i] >= 0:
            for j in range(i):
                if bodies[i] == bodies[j]:
                    gradients[j] = gradients[j] + gradients[i]
                    bodies[i] = -1
    return error, bodies, gradients


@wp.kernel
def _accumulate_reactions(
    data: _JointData,
    followers: wp.array[int],
    body_q: wp.array[wp.transform],
    multipliers: wp.array[float],
    forces: wp.array[wp.vec3],
    torques: wp.array[wp.vec3],
):
    follower = followers[wp.tid()]
    reference = data.reference[follower]
    if reference < 0 or not data.enabled[follower] or not data.enabled[reference]:
        return
    for component in range(data.coordinates.dof_dim[follower, 0] + data.coordinates.dof_dim[follower, 1]):
        _error, bodies, gradients = _evaluate_row(data, follower, component, body_q)
        multiplier = multipliers[data.coordinates.qd_start[follower] + component]
        for i in range(4):
            if bodies[i] >= 0:
                reaction = multiplier * gradients[i]
                wp.atomic_add(forces, bodies[i], wp.spatial_top(reaction))
                wp.atomic_add(torques, bodies[i], wp.spatial_bottom(reaction))


@wp.kernel
def _solve_mimics(
    data: _JointData,
    followers: wp.array[int],
    body_inv_mass: wp.array[float],
    h_ll: wp.array[wp.mat33],
    h_al: wp.array[wp.mat33],
    h_aa: wp.array[wp.mat33],
    body_q: wp.array[wp.transform],
    multipliers: wp.array[float],
):
    follower = followers[wp.tid()]
    reference = data.reference[follower]
    if reference < 0 or not data.enabled[follower] or not data.enabled[reference]:
        return
    # Rows in a color have disjoint bodies. Components within a joint are
    # sequential, so neither multi-axis joints nor shared leaders race.
    for component in range(data.coordinates.dof_dim[follower, 0] + data.coordinates.dof_dim[follower, 1]):
        error, bodies, gradients = _evaluate_row(data, follower, component, body_q)
        responses = _Mat46()
        compliance = float(0.0)
        for i in range(4):
            body = bodies[i]
            if body >= 0 and body_inv_mass[body] > 0.0:
                gradient = gradients[i]
                dx, dw = ldlt6_solve(
                    h_ll[body], h_aa[body], h_al[body], wp.spatial_top(gradient), wp.spatial_bottom(gradient)
                )
                response = wp.spatial_vector(dx, dw)
                responses[i] = response
                compliance += wp.dot(gradient, response)
        if compliance <= 0.0:
            continue
        delta_lambda = -error / compliance
        multipliers[data.coordinates.qd_start[follower] + component] += delta_lambda
        for i in range(4):
            body = bodies[i]
            if body >= 0 and body_inv_mass[body] > 0.0:
                delta = delta_lambda * responses[i]
                pose = body_q[body]
                rotation = wp.transform_get_rotation(pose)
                com = wp.transform_point(pose, data.coordinates.body_com[body])
                rotation_new = wp.normalize(rotation + 0.5 * wp.quat(wp.spatial_bottom(delta), 0.0) * rotation)
                position_new = (
                    com + wp.spatial_top(delta) - wp.quat_rotate(rotation_new, data.coordinates.body_com[body])
                )
                body_q[body] = wp.transform(position_new, rotation_new)


@wp.struct
class _ArticulationFactor:
    """Block pattern of the factored block-sparse articulation systems and their mimic rows.

    The pattern arrays are those of ``RigidArticulationSparseLayout``, restricted to the
    articulations solved by the 6 x 6 block kernels.
    """

    body_row: wp.array[wp.int32]
    """Sparse row of each body, or ``-1`` for a body outside the block-sparse articulations, shape [body_count]."""
    mimic_offsets: wp.array[wp.int32]
    """Start of each articulation's mimic followers in ``mimic_followers``, shape [articulation_count + 1]."""
    mimic_followers: wp.array[wp.int32]
    """Follower joints with a body in the articulation, in joint order per articulation."""
    body_offsets: wp.array[wp.int32]
    bodies: wp.array[wp.int32]
    row_offsets: wp.array[wp.int32]
    cols: wp.array[wp.int32]
    col_offsets: wp.array[wp.int32]
    col_rows: wp.array[wp.int32]
    col_slots: wp.array[wp.int32]
    diag_slots: wp.array[wp.int32]


@wp.func
def _to_vec6(v: wp.spatial_vector) -> vec6f:
    return vec6f(v[0], v[1], v[2], v[3], v[4], v[5])


@wp.func
def _to_spatial(v: vec6f) -> wp.spatial_vector:
    return wp.spatial_vector(v[0], v[1], v[2], v[3], v[4], v[5])


@wp.func
def _articulation_response(
    factor: _ArticulationFactor,
    values: wp.array[mat66f],
    articulation: int,
    bodies: wp.vec4i,
    gradients: _Mat46,
    body_inv_mass: wp.array[float],
    response: wp.array[vec6f],
):
    """Overwrite the articulation's rows of ``response`` with ``H^-1 J^T`` for one mimic row.

    ``values`` holds the block Cholesky factor ``L`` of the articulation's coupled system ``H``
    left by this iteration's block-sparse solve; ``J^T`` is the row's gradient on the
    articulation's dynamic bodies.
    """
    start = factor.body_offsets[articulation]
    end = factor.body_offsets[articulation + 1]
    for row in range(start, end):
        response[row] = vec6f(0.0)
    first = end
    for i in range(4):
        body = bodies[i]
        if body >= 0 and body_inv_mass[body] > 0.0:
            row = factor.body_row[body]
            if row >= 0:
                response[row] = _to_vec6(gradients[i])
                first = wp.min(first, row)
    # L z = J^T; rows before the first loaded row stay zero.
    for row in range(first, end):
        value = response[row]
        local_row = row - start
        for entry in range(factor.row_offsets[row], factor.row_offsets[row + 1]):
            local_col = factor.cols[entry]
            if local_col < local_row:
                value = value - values[entry] * response[start + local_col]
        response[row] = _solve_lower66(values[factor.diag_slots[row]], value)
    # L^T y = z
    for index in range(end - start):
        row = end - 1 - index
        value = response[row]
        for entry in range(factor.col_offsets[row], factor.col_offsets[row + 1]):
            value = value - wp.transpose(values[factor.col_slots[entry]]) * response[start + factor.col_rows[entry]]
        response[row] = _solve_upper66_from_lower(values[factor.diag_slots[row]], value)


@wp.func
def _move_body(data: _JointData, body: int, delta: wp.spatial_vector, body_q: wp.array[wp.transform]):
    pose = body_q[body]
    rotation = wp.transform_get_rotation(pose)
    com = wp.transform_point(pose, data.coordinates.body_com[body])
    rotation_new = wp.normalize(rotation + 0.5 * wp.quat(wp.spatial_bottom(delta), 0.0) * rotation)
    position_new = com + wp.spatial_top(delta) - wp.quat_rotate(rotation_new, data.coordinates.body_com[body])
    body_q[body] = wp.transform(position_new, rotation_new)


@wp.kernel
def _solve_mimics_coupled(
    data: _JointData,
    articulations: wp.array[int],
    body_inv_mass: wp.array[float],
    h_ll: wp.array[wp.mat33],
    h_al: wp.array[wp.mat33],
    h_aa: wp.array[wp.mat33],
    factor: _ArticulationFactor,
    values: wp.array[mat66f],
    response: wp.array[vec6f],
    body_q: wp.array[wp.transform],
    multipliers: wp.array[float],
):
    """Solve the mimic rows of block-sparse articulations against their coupled systems.

    The block-sparse solve answers a multiplier change with ``H^-1 J^T`` of the whole
    articulation, not of the row's bodies alone, so the row compliance ``J H^-1 J^T`` and the
    pose update use the articulation's factor. A row body outside the block-sparse
    articulations is jointed to no articulation body and keeps its own 6 x 6 block. One thread
    solves the rows of one articulation in sequence; the articulations of a launch share no
    such outside body.
    """
    articulation = articulations[wp.tid()]
    row_start = factor.body_offsets[articulation]
    row_end = factor.body_offsets[articulation + 1]
    for index in range(factor.mimic_offsets[articulation], factor.mimic_offsets[articulation + 1]):
        follower = factor.mimic_followers[index]
        reference = data.reference[follower]
        if reference < 0 or not data.enabled[follower] or not data.enabled[reference]:
            continue
        for component in range(data.coordinates.dof_dim[follower, 0] + data.coordinates.dof_dim[follower, 1]):
            error, bodies, gradients = _evaluate_row(data, follower, component, body_q)
            _articulation_response(factor, values, articulation, bodies, gradients, body_inv_mass, response)
            responses = _Mat46()
            compliance = float(0.0)
            for i in range(4):
                body = bodies[i]
                if body >= 0 and body_inv_mass[body] > 0.0:
                    gradient = gradients[i]
                    row = factor.body_row[body]
                    if row >= 0:
                        compliance += wp.dot(_to_vec6(gradient), response[row])
                    else:
                        dx, dw = ldlt6_solve(
                            h_ll[body], h_aa[body], h_al[body], wp.spatial_top(gradient), wp.spatial_bottom(gradient)
                        )
                        responses[i] = wp.spatial_vector(dx, dw)
                        compliance += wp.dot(gradient, responses[i])
            # A factor with clamped pivots can overflow the response; leave such a row alone.
            if not (compliance > 0.0 and wp.isfinite(compliance)):
                continue
            delta_lambda = -error / compliance
            multipliers[data.coordinates.qd_start[follower] + component] += delta_lambda
            # Every body of the articulation answers the multiplier change.
            for row in range(row_start, row_end):
                body = factor.bodies[row]
                if body_inv_mass[body] > 0.0:
                    _move_body(data, body, delta_lambda * _to_spatial(response[row]), body_q)
            for i in range(4):
                body = bodies[i]
                if body >= 0 and body_inv_mass[body] > 0.0:
                    if factor.body_row[body] < 0:
                        _move_body(data, body, delta_lambda * responses[i], body_q)


def _color_groups(items: list[int], item_units: list[set[int]], device) -> list[wp.array]:
    """Greedily color ``items`` so that the items of a color share no unit."""
    used_colors: dict[int, set[int]] = {}
    colors: list[list[int]] = []
    for item, units in zip(items, item_units, strict=True):
        unavailable = set().union(*(used_colors.get(unit, set()) for unit in units))
        color = 0
        while color in unavailable:
            color += 1
        if color == len(colors):
            colors.append([])
        colors[color].append(item)
        for unit in units:
            used_colors.setdefault(unit, set()).add(color)
    return [wp.array(group, dtype=int, device=device) for group in colors]


class JointMimicSolver:
    """Solve mimic rows once per VBD iteration, including their reaction forces.

    A mass-only projection discards friction/drive/contact stiffness and can
    satisfy the mimic equation while violating force balance. Instead, use
    the Schur complement J H^-1 J^T of VBD's assembled body blocks, and feed
    the resulting force multipliers into the next body sweep. Multipliers
    are reset each timestep; no additional iteration setting is needed.

    ``H`` is the system that the body update solves: the 6 x 6 block of each body in the local
    (Gauss-Seidel) solve, and the coupled matrix of the whole articulation in the block-sparse
    solves. The per-body blocks underestimate the coupled compliance of a row whose bodies are
    jointed to other bodies, and the multiplier iteration diverges once the coupled compliance
    exceeds about twice the estimate. With ``sparse_layout``, rows on block-sparse articulation
    bodies therefore take both the compliance and the pose update from the block Cholesky
    factor that the articulation solve of the same iteration leaves in its block storage: two
    block triangular solves over the articulation per row component and iteration, and one
    launch per iteration for all such rows unless articulations share a reference body outside
    the articulations.

    Args:
        model: Model whose mimic relationships are solved.
        coordinates: Joint coordinate data of the solver.
        sparse_layout: Layout of the block-sparse articulation solve, or ``None`` for the local solve.
    """

    def __init__(
        self,
        model: Model,
        coordinates: JointCoordinateData,
        sparse_layout: RigidArticulationSparseLayout | None = None,
    ):
        self.model = model
        self.data = _JointData()
        self.data.coordinates = coordinates
        self.data.enabled = model.joint_enabled
        self.data.reference = model.joint_mimic_joint
        self.data.coeffs = model.joint_mimic_coeffs
        self.multipliers = wp.zeros(model.joint_dof_count, dtype=float, device=model.device)
        references = model.joint_mimic_joint.numpy()
        types = model.joint_type.numpy()
        parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
        supported = (JointType.REVOLUTE, JointType.PRISMATIC, JointType.D6)
        followers = [j for j, r in enumerate(references) if r >= 0 and types[j] in supported and types[r] in supported]
        self.followers = wp.array(followers, dtype=int, device=model.device)

        def row_bodies(follower):
            reference = references[follower]
            joint_bodies = (parents[follower], children[follower], parents[reference], children[reference])
            return {int(body) for body in joint_bodies if body >= 0}

        # Rows on block-sparse articulation bodies move the whole articulation, so one thread
        # solves all rows of an articulation; the remaining rows are solved per row.
        self._factor = None
        self._response = None
        self._articulation_colors: list[wp.array] = []
        local_followers = followers
        if sparse_layout is not None and sparse_layout.rigid_body_row_count > 0 and followers:
            articulation_count = sparse_layout.rigid_articulation_count
            row_count = sparse_layout.rigid_body_row_count
            offsets = sparse_layout.articulation_body_offsets.numpy()[: articulation_count + 1]
            sparse_bodies = sparse_layout.articulation_bodies.numpy()[:row_count]
            body_row = np.full(model.body_count, -1, dtype=np.int32)
            body_row[sparse_bodies] = np.arange(row_count, dtype=np.int32)
            body_articulation = np.full(model.body_count, -1, dtype=np.int32)
            body_articulation[sparse_bodies] = np.repeat(
                np.arange(articulation_count, dtype=np.int32), np.diff(offsets)
            )
            articulation_followers: dict[int, list[int]] = {}
            articulation_units: dict[int, set[int]] = {}
            local_followers = []
            for follower in followers:
                bodies = row_bodies(follower)
                articulations = {int(body_articulation[body]) for body in bodies} - {-1}
                if len(articulations) > 1:
                    raise ValueError(
                        f"Mimic joint {follower} and its reference joint {int(references[follower])} lie in two "
                        "different articulations, which the block-sparse articulation solve does not support. "
                        "Use rigid_articulation_solve='local'."
                    )
                if not articulations:
                    local_followers.append(follower)
                    continue
                articulation = articulations.pop()
                articulation_followers.setdefault(articulation, []).append(follower)
                # Articulations conflict through the bodies outside the articulations that their rows share.
                outside = {body for body in bodies if body_articulation[body] < 0}
                articulation_units.setdefault(articulation, set()).update(outside)
            if articulation_followers:
                counts = np.zeros(articulation_count, dtype=np.int64)
                for articulation, rows in articulation_followers.items():
                    counts[articulation] = len(rows)
                mimic_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
                mimic_followers = [
                    f for articulation in sorted(articulation_followers) for f in articulation_followers[articulation]
                ]
                factor = _ArticulationFactor()
                factor.body_row = wp.array(body_row, dtype=wp.int32, device=model.device)
                factor.mimic_offsets = wp.array(mimic_offsets, dtype=wp.int32, device=model.device)
                factor.mimic_followers = wp.array(mimic_followers, dtype=wp.int32, device=model.device)
                factor.body_offsets = sparse_layout.articulation_body_offsets
                factor.bodies = sparse_layout.articulation_bodies
                factor.row_offsets = sparse_layout.articulation_block_row_offsets
                factor.cols = sparse_layout.articulation_block_cols
                factor.col_offsets = sparse_layout.articulation_block_col_offsets
                factor.col_rows = sparse_layout.articulation_block_col_rows
                factor.col_slots = sparse_layout.articulation_block_col_slots
                factor.diag_slots = sparse_layout.articulation_diag_slots
                self._factor = factor
                self._response = wp.zeros(row_count, dtype=vec6f, device=model.device)
                ordered = sorted(articulation_followers)
                self._articulation_colors = _color_groups(
                    ordered, [articulation_units[articulation] for articulation in ordered], model.device
                )
        # Greedy edge coloring of four-body constraints, independent of the
        # ordinary body coloring (which only includes structural joints).
        self.colors = _color_groups(local_followers, [row_bodies(f) for f in local_followers], model.device)

    def accumulate_reactions(self, body_q, forces, torques):
        wp.launch(
            _accumulate_reactions,
            dim=self.followers.size,
            inputs=[self.data, self.followers, body_q, self.multipliers],
            outputs=[forces, torques],
            device=self.model.device,
        )

    def solve(self, body_q, body_inv_mass, h_ll, h_al, h_aa, factor_values=None):
        """Solve every mimic row once.

        Args:
            body_q: Body poses of the iterate, updated in place.
            body_inv_mass: Effective inverse body masses.
            h_ll: Linear 3 x 3 blocks of the bodies' assembled 6 x 6 systems.
            h_al: Angular-linear blocks.
            h_aa: Angular blocks.
            factor_values: Block Cholesky factors that this iteration's block-sparse articulation
                solve left in its block storage. Required when the solver was built with a
                sparse layout and has rows on block-sparse articulation bodies.
        """
        for group in self._articulation_colors:
            wp.launch(
                _solve_mimics_coupled,
                dim=group.size,
                inputs=[
                    self.data,
                    group,
                    body_inv_mass,
                    h_ll,
                    h_al,
                    h_aa,
                    self._factor,
                    factor_values,
                    self._response,
                ],
                outputs=[body_q, self.multipliers],
                device=self.model.device,
            )
        for group in self.colors:
            wp.launch(
                _solve_mimics,
                dim=group.size,
                inputs=[self.data, group, body_inv_mass, h_ll, h_al, h_aa],
                outputs=[body_q, self.multipliers],
                device=self.model.device,
            )
