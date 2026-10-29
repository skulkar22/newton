# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host-side sparse articulation layout for the VBD rigid solver."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import warp as wp

from ...sim import JointType


@dataclass
class RigidArticulationSparseLevels:
    """Independent pivot levels and deterministic Schur-update reductions."""

    articulation_level_offsets: wp.array[wp.int32]
    level_body_offsets: wp.array[wp.int32]
    level_column_offsets: wp.array[wp.int32]
    level_column_slots: wp.array[wp.int32]
    level_column_diag_slots: wp.array[wp.int32]
    level_update_offsets: wp.array[wp.int32]
    level_update_dst_slots: wp.array[wp.int32]
    level_update_term_offsets: wp.array[wp.int32]
    level_update_left_slots: wp.array[wp.int32]
    level_update_right_slots: wp.array[wp.int32]


@dataclass
class RigidArticulationSparseLayout:
    """Static block-sparse articulation layout for rigid VBD solves."""

    articulation_body_offsets: wp.array[wp.int32]
    articulation_joint_offsets: wp.array[wp.int32]
    articulation_bodies: wp.array[wp.int32]
    articulation_joints: wp.array[wp.int32]
    articulation_joint_body_start: wp.array[wp.int32]
    articulation_block_row_offsets: wp.array[wp.int32]
    articulation_block_cols: wp.array[wp.int32]
    articulation_block_col_offsets: wp.array[wp.int32]
    articulation_block_col_rows: wp.array[wp.int32]
    articulation_block_col_slots: wp.array[wp.int32]
    articulation_factor_update_offsets: wp.array[wp.int32]
    articulation_factor_update_dst_slots: wp.array[wp.int32]
    articulation_factor_update_left_slots: wp.array[wp.int32]
    articulation_factor_update_right_slots: wp.array[wp.int32]
    articulation_diag_slots: wp.array[wp.int32]
    body_articulation_local: wp.array[wp.int32]
    local_body_color_groups: list[wp.array[wp.int32]]
    articulation_count: int
    articulation_body_count: int
    articulation_joint_count: int
    local_body_count: int
    block_count: int
    elastic_bodies: wp.array[wp.int32]
    """Reduced elastic bodies whose modes the Schur backend eliminates, shape [elastic_body_count].

    This and the following ``elastic_*`` group and pair fields, ``joint_cross_slot`` and
    ``cross_block_count`` serve only ``elastic_solve="schur"``; under ``"coupled"`` they are
    empty, ``-1`` or zero (see ``mixed_elastic_bodies``).
    """
    elastic_body_row_start: wp.array[wp.int32]
    """First sparse row of each elastic body's articulation, shape [elastic_body_count]."""
    elastic_group_offsets: wp.array[wp.int32]
    """Schur column groups of each elastic body, shape [elastic_body_count + 1].

    The first group of a body is its own frame; every further group is one joint
    that couples the body's modal coordinates to a neighbor frame.
    """
    elastic_group_joint: wp.array[wp.int32]
    """Joint of each Schur column group, or ``-1`` for the own frame, shape [elastic_group_count]."""
    elastic_group_local: wp.array[wp.int32]
    """Articulation-local row of the frame each group couples to, shape [elastic_group_count]."""
    elastic_group_owner: wp.array[wp.int32]
    """Index into ``elastic_bodies`` of the body owning each group, shape [elastic_group_count]."""
    elastic_group_body: wp.array[wp.int32]
    """Body whose frame each group couples to, shape [elastic_group_count]."""
    elastic_group_neighbor_elastic: wp.array[wp.int32]
    """Whether a joint group's neighbor is itself a reduced elastic body, shape [elastic_group_count]."""
    elastic_pair_left: wp.array[wp.int32]
    """Row group of each modal Schur block, shape [elastic_pair_count]."""
    elastic_pair_right: wp.array[wp.int32]
    """Column group of each modal Schur block, shape [elastic_pair_count]."""
    elastic_pair_slot: wp.array[wp.int32]
    """Sparse block slot receiving each modal Schur block, shape [elastic_pair_count]."""
    joint_cross_slot: wp.array[wp.int32]
    """Scratch block slot capturing a joint's parent-child Hessian block, or ``-1``, shape [joint_count]."""
    elastic_body_count: int
    elastic_group_count: int
    elastic_pair_count: int
    cross_block_count: int
    elastic_solve: str
    """How reduced elastic bodies inside articulations are solved: ``"schur"`` or ``"coupled"``."""
    rigid_articulation_count: int
    """Articulations ``[0, rigid_articulation_count)`` are solved by the 6 x 6 block kernels.

    Under ``"coupled"`` these are the articulations without reduced elastic bodies, listed before
    every mixed articulation, so the rigid kernels run on a prefix of the existing arrays with
    unchanged ``slot * 36`` and ``row * 6`` indexing. Under ``"schur"`` every articulation is in
    the prefix, because the Schur backend eliminates elastic modes onto 6 x 6 frame blocks.
    """
    rigid_body_row_count: int
    """Sparse rows of the rigid prefix, ``articulation_body_offsets[rigid_articulation_count]``."""
    rigid_joint_count: int
    """Joints of the rigid prefix, ``articulation_joint_offsets[rigid_articulation_count]``."""
    rigid_block_count: int
    """Block slots of the rigid prefix, ``articulation_block_row_offsets[rigid_body_row_count]``."""
    mixed_articulation_count: int
    """Articulations solved by the variable-size block backend (``"coupled"`` only).

    They are ``[rigid_articulation_count, articulation_count)`` and contain at least one reduced
    elastic body and at least two bodies.
    """
    articulation_row_dims: wp.array[wp.int32]
    """Logical block size ``6 + m_i`` of every sparse row, shape [articulation_body_count].

    ``m_i`` is the modal count of a reduced elastic body in a mixed articulation and 0 otherwise,
    so every row of the rigid prefix (and every row under ``"schur"``) has size 6.
    """
    articulation_row_scalar_offsets: wp.array[wp.int32]
    """Exclusive prefix sum of ``articulation_row_dims``, shape [articulation_body_count + 1].

    Component ``k`` of row ``r`` of a right-hand side or update vector is stored at
    ``articulation_row_scalar_offsets[r] + k``; for rows of size 6 this equals ``6 * r + k``.
    """
    articulation_block_value_offsets: wp.array[wp.int32]
    """First scalar of each block slot, shape [block_count + 1].

    Slot ``s`` stores a dense row-major ``articulation_block_dims[s, 0] x articulation_block_dims[s, 1]``
    block, entry ``(r, c)`` at ``articulation_block_value_offsets[s] + r * articulation_block_dims[s, 1] + c``;
    for 6 x 6 slots the offset equals ``36 * s``.
    """
    articulation_block_dims: wp.array2d[wp.int32]
    """Row and column size of each block slot, shape [block_count, 2]."""
    articulation_scalar_count: int
    """Scalars in a right-hand side vector over all rows, ``articulation_row_scalar_offsets[-1]``."""
    articulation_value_count: int
    """Scalars of all block slots, ``articulation_block_value_offsets[-1]``."""
    articulation_max_row_dim: int
    """Largest entry of ``articulation_row_dims``, or 0 without rows."""
    mixed_articulation_joints: wp.array[wp.int32]
    """Joints of the mixed articulations, ``articulation_joints[rigid_joint_count:]``, shape [articulation_joint_count - rigid_joint_count]."""
    mixed_elastic_bodies: wp.array[wp.int32]
    """Reduced elastic bodies of the mixed articulations in row order (``"coupled"`` only), shape [mixed_elastic_body_count]."""
    mixed_elastic_body_rows: wp.array[wp.int32]
    """Global sparse row of each entry of ``mixed_elastic_bodies``, shape [mixed_elastic_body_count]."""
    mixed_elastic_body_count: int
    mixed_slot_entry_offsets: wp.array[wp.int32]
    """Exclusive prefix sum of the gather entries of the mixed block slots, shape [block_count - rigid_block_count + 1].

    Mixed slot ``s`` (global slot ``rigid_block_count + s``) of size ``R x C`` has ``R * C`` entries,
    plus ``R`` right-hand side components for a diagonal slot; entry ``e`` of the slot is gather
    entry ``mixed_slot_entry_offsets[s] + e``. ``[0]`` without mixed slots (always under ``"schur"``).
    """
    mixed_slot_entry_count: int
    """Gather entries of all mixed block slots, ``mixed_slot_entry_offsets[-1]`` (0 under ``"schur"``)."""
    mixed_slot_rows: wp.array[wp.int32]
    """Global sparse row of each mixed block slot, shape [block_count - rigid_block_count]."""
    mixed_slot_joint_offsets: wp.array[wp.int32]
    """CSR offsets of ``mixed_slot_joints``, shape [block_count - rigid_block_count + 1]."""
    mixed_slot_joints: wp.array[wp.int32]
    """Joints that can add to each mixed block slot, as indices into ``mixed_articulation_joints``.

    For the diagonal slot of a row: the joints of the articulation with the row's body as child,
    or as parent; for an off-diagonal slot: the joints between the two rows' bodies (either
    direction); fill slots have none. Each list is in the articulation's joint order, so a gather
    over it adds the same contributions in the same order as one over all of the articulation's
    joints. The endpoints are taken from ``joint_parent`` / ``joint_child``, a superset of the
    evaluated endpoints.
    """
    mixed_joint_twist_offsets: wp.array[wp.int32]
    """Offsets of the modal attachment twists of the mixed joints' endpoints, shape [2 * len(mixed_articulation_joints) + 1].

    Entry ``2 * j + side`` (``side`` 0 for the parent, 1 for the child endpoint of
    ``mixed_articulation_joints[j]``) starts the endpoint's ``m`` twists, one per mode of a reduced
    elastic endpoint body with an endpoint sample (0 twists otherwise), in the scratch that
    :func:`~newton._src.solvers.vbd.rigid_sparse_articulation_variable_assembly.assemble_articulation_variable_system`
    fills each iterate. ``[0]`` without mixed joints (always under ``"schur"``).
    """
    mixed_twist_count: int
    """Endpoint mode twists of the mixed joints, ``mixed_joint_twist_offsets[-1]`` (0 under ``"schur"``)."""
    mixed_max_wide_column_rows: int
    """Largest ``n_k + b_k`` over the mixed rows with ``n_k > 6``, where ``b_k`` is the number of scalar rows of the column's off-diagonal blocks (0 without such rows)."""
    mixed_max_wide_column_footprint: int
    """Largest ``n_k * (n_k + max(b_k, 1))`` over the mixed rows with ``n_k > 6`` (0 without such rows).

    The scalars a warp of the variable-block solve stages to factor the column (see
    :func:`~newton._src.solvers.vbd.rigid_sparse_articulation_variable_kernels.warp_solve_preferred`).
    """
    standalone_elastic_bodies: wp.array[wp.int32]
    """Reduced elastic bodies left to the local solve (``"coupled"`` only), shape [standalone_elastic_body_count].

    An articulation whose only body is a reduced elastic body (all its joints attach to the world
    or are the body's owner joint) is not part of the sparse layout; the body joins
    ``local_body_color_groups`` and is solved by the local tiled or dense block solve.
    """
    standalone_elastic_body_count: int
    levels: RigidArticulationSparseLevels | None = None
    """Independent pivot levels for ``rigid_articulation_solve="block_sparse_levels"``, else ``None``.

    Levels cover the articulations solved by the 6 x 6 kernels, ``[0, rigid_articulation_count)``.
    Mixed articulations (``elastic_solve="coupled"`` only) keep their size-weighted order and get an
    empty level range; the variable-size block solve factors them.
    """


ELASTIC_SOLVE_MODES = ("schur", "coupled")
"""Supported values of the ``elastic_solve`` argument of :func:`build_rigid_articulation_sparse_layout`."""

_INT32_MAX = 2**31 - 1


def _symbolic_cholesky_pattern(body_count: int, edges: set[tuple[int, int]]) -> list[list[int]]:
    rows = [{i} for i in range(body_count)]
    for a, b in edges:
        row = max(a, b)
        col = min(a, b)
        rows[row].add(col)

    for k in range(body_count):
        later = [i for i in range(k + 1, body_count) if k in rows[i]]
        for i_index, i in enumerate(later):
            for j in later[:i_index]:
                rows[max(i, j)].add(min(i, j))

    return [sorted(row) for row in rows]


def _minimum_degree_order(body_count: int, edges: set[tuple[int, int]]) -> list[int]:
    adjacency = [set() for _ in range(body_count)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    remaining = set(range(body_count))
    order: list[int] = []
    for _ in range(body_count):
        pivot = min(remaining, key=lambda i: (len(adjacency[i] & remaining), i))
        neighbors = sorted((adjacency[pivot] & remaining) - {pivot})
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                adjacency[a].add(b)
                adjacency[b].add(a)
        remaining.remove(pivot)
        order.append(pivot)

    return order


def _independent_level_order(body_count: int, edges: set[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """Eliminate independent low-degree pivots together, including loop fill.

    On a chain this contracts alternating vertices each round. The same
    symbolic procedure remains valid for general graphs; dense separators
    eventually serialize. Low degree first limits avoidable fill.
    """
    adjacency = [set() for _ in range(body_count)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining = set(range(body_count))
    order, offsets = [], [0]
    while remaining:
        blocked = set()
        pivots = []
        for pivot in sorted(remaining, key=lambda i: (len(adjacency[i] & remaining), i)):
            if pivot not in blocked:
                pivots.append(pivot)
                blocked.add(pivot)
                blocked.update(adjacency[pivot] & remaining)
        for pivot in pivots:
            neighbors = sorted(adjacency[pivot] & remaining)
            for i, a in enumerate(neighbors):
                for b in neighbors[:i]:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
        remaining.difference_update(pivots)
        order.extend(pivots)
        offsets.append(len(order))
    return order, offsets


def _weighted_minimum_degree_order(body_count: int, edges: set[tuple[int, int]], dims: list[int]) -> list[int]:
    """Greedy elimination order for blocks of different sizes.

    Eliminating block ``k`` with remaining neighbors ``N`` applies one trailing update
    ``A_ab -= L_ak L_bk^T`` to every stored pair ``a >= b`` of ``N``, which costs
    ``n_k * sum_{a <= b in N} n_a * n_b = n_k * (S**2 + Q) / 2`` multiply-adds with
    ``S = sum n_a`` and ``Q = sum n_a**2``. The pivot minimizes that cost, ties broken by index.
    When all ``n`` are equal the cost increases strictly with ``|N|``, so the order equals
    :func:`_minimum_degree_order`, including its tie-breaking and its neighbor-set definition.
    """
    adjacency = [set() for _ in range(body_count)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    def cost(i: int, remaining: set[int]) -> tuple[int, int]:
        neighbors = adjacency[i] & remaining
        size_sum = sum(dims[a] for a in neighbors)
        size_square_sum = sum(dims[a] * dims[a] for a in neighbors)
        return dims[i] * (size_sum * size_sum + size_square_sum), i

    remaining = set(range(body_count))
    order: list[int] = []
    for _ in range(body_count):
        pivot = min(remaining, key=lambda i: cost(i, remaining))
        neighbors = sorted((adjacency[pivot] & remaining) - {pivot})
        for i, a in enumerate(neighbors):
            for b in neighbors[:i]:
                adjacency[a].add(b)
                adjacency[b].add(a)
        remaining.remove(pivot)
        order.append(pivot)

    return order


def _int32_array(values, device) -> wp.array[wp.int32]:
    return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)


def build_rigid_articulation_sparse_layout(
    model, device: wp.context.Devicelike, *, elastic_solve: str = "coupled", use_levels: bool = False
) -> RigidArticulationSparseLayout | None:
    """Build the static sparse layout from Newton articulation ranges.

    This host analysis runs once from :class:`SolverVBD` construction, never
    from the per-step or per-iteration path. Repeated articulation topologies,
    such as replicated RL environments, share cached ordering and fill analysis.

    Each articulation is one direct-factorization group: every joint in
    ``[articulation_start, articulation_end)`` is assembled into that group's
    block system, including loop-closing joints added through
    :meth:`~newton.ModelBuilder.add_articulation` with ``allow_closed_loops=True``.
    Bodies outside every articulation remain in the regular local VBD solve.
    Joints outside articulation ranges must not touch articulation bodies.

    Reduced elastic owner joints (:attr:`~newton.JointType.ELASTIC`) carry no
    constraint, add no graph edge, and may lie inside or outside an articulation
    range. An elastic body joins an articulation through its owner joint or any
    physical joint in the range.

    With ``elastic_solve="schur"`` every row is a 6 x 6 frame block. An elastic
    body's modal coordinates are eliminated onto its frame block, which couples
    every pair of the body's joint neighbors, so those pairs are added to the
    fill pattern.

    With ``elastic_solve="coupled"`` the articulations are classified. Rigid
    articulations (no elastic body) are listed first, in model order, and keep
    the 6 x 6 layout bit for bit. An articulation whose only body is elastic is
    left out of the layout and solved locally (``standalone_elastic_bodies``).
    The remaining (mixed) articulations follow the rigid prefix; their rows have
    size ``6 + m`` (``m`` modes of an elastic body, 0 for a rigid body), no
    elimination fill is forced, and the elimination order minimizes the
    size-weighted trailing-update cost (:func:`_weighted_minimum_degree_order`).
    The Schur-only fields (``elastic_*`` groups and pairs, ``joint_cross_slot``)
    are empty or ``-1``.

    Args:
        model: The model whose articulations are analyzed.
        device: Device of the layout arrays.
        elastic_solve: ``"coupled"`` (default, as in :class:`~newton.solvers.SolverVBD`) or
            ``"schur"``; see above.
        use_levels: Order the articulations solved by the 6 x 6 kernels (every articulation under
            ``"schur"``, the rigid prefix under ``"coupled"``) by independent pivot levels and fill
            :attr:`RigidArticulationSparseLayout.levels` for ``rigid_articulation_solve="block_sparse_levels"``.
            Mixed articulations keep the size-weighted order and are factored by the variable-size
            block solve, as without levels.

    Raises:
        ValueError: If ``elastic_solve`` is unknown, the articulation ranges are
            inconsistent with the joints, a joint on an elastic articulation body
            has no elastic endpoint sample, or a layout index exceeds int32.
    """

    if elastic_solve not in ELASTIC_SOLVE_MODES:
        raise ValueError(f"elastic_solve must be one of {ELASTIC_SOLVE_MODES}, got {elastic_solve!r}.")
    coupled = elastic_solve == "coupled"

    if model.body_count == 0:
        return None

    joint_parent = np.asarray(model.joint_parent.to("cpu").numpy(), dtype=np.int32)
    joint_child = np.asarray(model.joint_child.to("cpu").numpy(), dtype=np.int32)
    # Topology-only models (e.g. the level-ordering test fixtures) carry no joint types and so no
    # reduced elastic owner joints.
    if hasattr(model, "joint_type"):
        joint_type = np.asarray(model.joint_type.to("cpu").numpy(), dtype=np.int32)
        elastic_owner = joint_type == int(JointType.ELASTIC)
    else:
        elastic_owner = np.zeros((model.joint_count,), dtype=bool)
    if int(getattr(model, "elastic_body_count", 0)) > 0:
        body_elastic_index = np.asarray(model.body_elastic_index.to("cpu").numpy(), dtype=np.int32)
        elastic_mode_count = np.asarray(model.elastic_mode_count.to("cpu").numpy(), dtype=np.int32)
        parent_endpoint = np.asarray(model.joint_parent_elastic_endpoint.to("cpu").numpy(), dtype=np.int32)
        child_endpoint = np.asarray(model.joint_child_elastic_endpoint.to("cpu").numpy(), dtype=np.int32)
    else:
        body_elastic_index = np.full((model.body_count,), -1, dtype=np.int32)
        elastic_mode_count = np.zeros((0,), dtype=np.int32)
        parent_endpoint = np.full((model.joint_count,), -1, dtype=np.int32)
        child_endpoint = np.full((model.joint_count,), -1, dtype=np.int32)
    articulation_start = np.asarray(model.articulation_start.to("cpu").numpy(), dtype=np.int32)
    articulation_end = np.asarray(model.articulation_end.to("cpu").numpy(), dtype=np.int32)

    articulation_bodies_host: list[int] = []
    articulation_joints_host: list[int] = []
    articulation_joint_body_start_host: list[int] = []
    articulation_body_offsets_host = [0]
    articulation_joint_offsets_host = [0]
    block_row_offsets_host = [0]
    block_cols_host: list[int] = []
    block_col_rows_host: list[int] = []
    block_col_slots_host: list[int] = []
    block_col_offsets_host: list[int] = []
    factor_update_dst_slots_host: list[int] = []
    factor_update_left_slots_host: list[int] = []
    factor_update_right_slots_host: list[int] = []
    factor_update_offsets_host: list[int] = []
    diag_slots_host: list[int] = []
    row_dims_host: list[int] = []
    articulation_level_offsets = [0]
    level_body_offsets = [0]
    level_column_offsets = [0]
    level_column_slots, level_column_diag_slots = [], []
    level_update_offsets = [0]
    level_update_dst_slots, level_update_left_slots, level_update_right_slots = [], [], []
    level_update_term_offsets = [0]

    body_articulation_local_host = np.full((model.body_count,), -1, dtype=np.int32)
    for joint_idx in range(model.joint_count):
        parent = int(joint_parent[joint_idx])
        child = int(joint_child[joint_idx])
        if parent >= model.body_count or child >= model.body_count:
            raise ValueError(
                f"Joint {joint_idx} references an out-of-range body: parent={parent}, child={child}, "
                f"body_count={model.body_count}."
            )
        if parent < 0 and child < 0:
            raise ValueError(f"Joint {joint_idx} has no rigid-body endpoint.")

    # One group per articulation. The range is authoritative, so loop-closing joints
    # added with allow_closed_loops=True are factorized together with the tree joints.
    body_group = np.full((model.body_count,), -1, dtype=np.int32)
    articulation_groups: list[tuple[list[int], list[int]]] = []
    for articulation_id in range(model.articulation_count):
        joints = list(range(int(articulation_start[articulation_id]), int(articulation_end[articulation_id])))
        bodies: list[int] = []
        for joint_idx in joints:
            for body in (int(joint_parent[joint_idx]), int(joint_child[joint_idx])):
                if body >= 0 and body not in bodies:
                    if body_group[body] >= 0:
                        raise ValueError(
                            f"Body {body} appears in articulation {articulation_id} and articulation "
                            f"{int(body_group[body])}. The block-sparse articulation solve requires each body to "
                            f"belong to at most one articulation."
                        )
                    bodies.append(body)
        if not bodies:
            continue
        for body in bodies:
            body_group[body] = len(articulation_groups)
        articulation_groups.append((bodies, joints))

    # Under "coupled", list rigid articulations first (stable), drop articulations whose only
    # body is elastic to the local solve, and append the mixed articulations.
    standalone_elastic_bodies_host: list[int] = []
    rigid_group_count = len(articulation_groups)
    if coupled:
        rigid_groups: list[tuple[list[int], list[int]]] = []
        mixed_groups: list[tuple[list[int], list[int]]] = []
        for bodies, joints in articulation_groups:
            elastic_count = sum(1 for body in bodies if body_elastic_index[body] >= 0)
            if elastic_count == 0:
                rigid_groups.append((bodies, joints))
            elif len(bodies) == 1:
                standalone_elastic_bodies_host.append(bodies[0])
            else:
                mixed_groups.append((bodies, joints))
        articulation_groups = rigid_groups + mixed_groups
        rigid_group_count = len(rigid_groups)

    body_group.fill(-1)
    joint_group = np.full((model.joint_count,), -1, dtype=np.int32)
    for group, (bodies, joints) in enumerate(articulation_groups):
        for body in bodies:
            body_group[body] = group
        for joint_idx in joints:
            joint_group[joint_idx] = group

    # Without articulation groups every body, including a standalone elastic one, is solved locally.
    if not articulation_groups:
        return None

    for joint_idx in range(model.joint_count):
        if joint_group[joint_idx] >= 0 or elastic_owner[joint_idx]:
            continue
        parent = int(joint_parent[joint_idx])
        child = int(joint_child[joint_idx])
        parent_group = int(body_group[parent]) if parent >= 0 else -1
        child_group = int(body_group[child]) if child >= 0 else -1
        if parent_group >= 0 or child_group >= 0:
            raise ValueError(
                f"Joint {joint_idx} lies outside the declared articulation ranges but touches an articulation body "
                f"(parent={parent}, child={child}). The block-sparse VBD solve requires every joint touching an "
                f"articulation body to belong to that articulation. Include the joint in one articulation, using "
                f"ModelBuilder.add_articulation(..., allow_closed_loops=True) for a loop closure, or use "
                f"rigid_articulation_solve='local'."
            )

    local_body_color_groups: list[wp.array[wp.int32]] = []
    local_body_count = 0
    for color_group in model.body_color_groups:
        color_bodies = np.asarray(color_group.to("cpu").numpy(), dtype=np.int32)
        local_bodies = color_bodies[body_group[color_bodies] < 0]
        if local_bodies.size > 0:
            local_body_color_groups.append(wp.array(local_bodies, dtype=wp.int32, device=device))
            local_body_count += int(local_bodies.size)

    elastic_bodies_host: list[int] = []
    elastic_body_row_start_host: list[int] = []
    elastic_group_offsets_host = [0]
    elastic_group_joint_host: list[int] = []
    elastic_group_body_host: list[int] = []
    joint_cross_slot_host = np.full((model.joint_count,), -1, dtype=np.int32)
    elastic_group_owner_host: list[int] = []
    elastic_pair_left_host: list[int] = []
    elastic_pair_right_host: list[int] = []
    elastic_pair_slot_host: list[int] = []
    cross_joints: list[int] = []
    mixed_elastic_bodies_host: list[int] = []
    mixed_elastic_body_rows_host: list[int] = []

    symbolic_cache: dict[
        tuple[bool, str, bool, int, tuple[tuple[int, int], ...], tuple[int, ...]],
        tuple[list[int], list[list[int]], list[int]],
    ] = {}
    for group_index, (bodies, joints) in enumerate(articulation_groups):
        mixed = coupled and group_index >= rigid_group_count
        local_index = {body: i for i, body in enumerate(bodies)}
        edges: set[tuple[int, int]] = set()
        elastic_neighbors: dict[int, list[tuple[int, int]]] = {
            body: [] for body in bodies if body_elastic_index[body] >= 0
        }
        for joint_idx in joints:
            parent = int(joint_parent[joint_idx])
            child = int(joint_child[joint_idx])
            if elastic_owner[joint_idx]:
                continue
            for body, is_parent in ((parent, True), (child, False)):
                if body >= 0 and body in elastic_neighbors:
                    endpoint = parent_endpoint[joint_idx] if is_parent else child_endpoint[joint_idx]
                    if endpoint < 0:
                        raise ValueError(
                            f"Joint {joint_idx} touches reduced elastic body {body} without an elastic endpoint "
                            f"sample, so the block-sparse VBD solve cannot couple it to the body's modes. "
                            f"Use rigid_articulation_solve='local' for this model."
                        )
                    other = child if is_parent else parent
                    if other >= 0 and other != body:
                        elastic_neighbors[body].append((joint_idx, other))
            if parent >= 0 and child >= 0 and parent in local_index and child in local_index:
                edges.add((local_index[parent], local_index[child]))
        if not coupled:
            # Eliminating an elastic body's modes couples all of its neighbor frames.
            for neighbors in elastic_neighbors.values():
                others = sorted({local_index[other] for _joint, other in neighbors})
                for i, a in enumerate(others):
                    for b in others[:i]:
                        edges.add((a, b))

        if mixed:
            dims = [
                6 + int(elastic_mode_count[body_elastic_index[body]]) if body in elastic_neighbors else 6
                for body in bodies
            ]
        else:
            dims = [6] * len(bodies)
        # Mixed articulations are factored by the variable-size block solve, never by levels.
        group_levels = use_levels and not mixed
        topology_key = (
            group_levels,
            elastic_solve,
            mixed,
            len(bodies),
            tuple(sorted((min(a, b), max(a, b)) for a, b in edges)),
            tuple(dims),
        )
        symbolic = symbolic_cache.get(topology_key)
        if symbolic is None:
            levels: list[int] = []
            if group_levels:
                order, levels = _independent_level_order(len(bodies), edges)
            elif len(bodies) == 1:
                order = [0]
            elif mixed:
                order = _weighted_minimum_degree_order(len(bodies), edges, dims)
            else:
                order = _minimum_degree_order(len(bodies), edges)
            old_to_new = {old: new for new, old in enumerate(order)}
            ordered_edges = {(old_to_new[a], old_to_new[b]) for a, b in edges}
            pattern = _symbolic_cholesky_pattern(len(bodies), ordered_edges)
            symbolic = (order, pattern, levels)
            symbolic_cache[topology_key] = symbolic
        order, pattern, levels = symbolic
        ordered_bodies = [bodies[old] for old in order]
        row_dims_host.extend(dims[old] for old in order)

        body_start = len(articulation_bodies_host)
        articulation_bodies_host.extend(ordered_bodies)
        articulation_joints_host.extend(joints)
        articulation_joint_body_start_host.extend([body_start] * len(joints))
        articulation_body_offsets_host.append(len(articulation_bodies_host))
        articulation_joint_offsets_host.append(len(articulation_joints_host))

        local_index_ordered = {body: i for i, body in enumerate(ordered_bodies)}
        for local_body, body in enumerate(ordered_bodies):
            body_articulation_local_host[body] = local_body

        for local_row, row_cols in enumerate(pattern):
            row_start = len(block_cols_host)
            block_cols_host.extend(row_cols)
            diag_slots_host.append(row_start + row_cols.index(local_row))
            block_row_offsets_host.append(len(block_cols_host))

        block_lookup: dict[tuple[int, int], int] = {}
        local_row_offset_start = len(block_row_offsets_host) - len(pattern) - 1
        for local_row, row_cols in enumerate(pattern):
            row_start = block_row_offsets_host[local_row_offset_start + local_row]
            for local_col_index, local_col in enumerate(row_cols):
                block_lookup[(local_row, local_col)] = row_start + local_col_index

        for body in ordered_bodies:
            if body not in elastic_neighbors:
                continue
            if coupled:
                mixed_elastic_bodies_host.append(body)
                mixed_elastic_body_rows_host.append(body_start + local_index_ordered[body])
                continue
            elastic_index = len(elastic_bodies_host)
            elastic_bodies_host.append(body)
            elastic_body_row_start_host.append(body_start)
            group_begin = len(elastic_group_joint_host)
            group_locals = [local_index_ordered[body]]
            elastic_group_joint_host.append(-1)
            elastic_group_body_host.append(body)
            for joint_idx, other in elastic_neighbors[body]:
                if joint_cross_slot_host[joint_idx] < 0:
                    joint_cross_slot_host[joint_idx] = len(cross_joints)
                    cross_joints.append(joint_idx)
                elastic_group_joint_host.append(joint_idx)
                elastic_group_body_host.append(other)
                group_locals.append(local_index_ordered[other])
            elastic_group_offsets_host.append(len(elastic_group_joint_host))
            elastic_group_owner_host.extend([elastic_index] * len(group_locals))
            # Schur blocks of the stored lower block triangle, one per ordered group pair.
            for g, local_g in enumerate(group_locals):
                for h, local_h in enumerate(group_locals):
                    if local_g >= local_h:
                        elastic_pair_left_host.append(group_begin + g)
                        elastic_pair_right_host.append(group_begin + h)
                        elastic_pair_slot_host.append(block_lookup[(local_g, local_h)])

        for local_col in range(len(ordered_bodies)):
            block_col_offsets_host.append(len(block_col_rows_host))
            later_rows: list[tuple[int, int]] = []
            for local_row in range(local_col + 1, len(ordered_bodies)):
                slot = block_lookup.get((local_row, local_col))
                if slot is not None:
                    later_rows.append((local_row, slot))
                    block_col_rows_host.append(local_row)
                    block_col_slots_host.append(slot)

            factor_update_offsets_host.append(len(factor_update_dst_slots_host))
            for left_index, (left_row, left_slot) in enumerate(later_rows):
                for right_row, right_slot in later_rows[: left_index + 1]:
                    dst_slot = block_lookup[(left_row, right_row)]
                    factor_update_dst_slots_host.append(dst_slot)
                    factor_update_left_slots_host.append(left_slot)
                    factor_update_right_slots_host.append(right_slot)

        # Each level has independent pivots. Several pivots may update the
        # same remaining block, so group contributions by destination and
        # reduce them in fixed order instead of racing or using float atomics.
        for begin, end in itertools.pairwise(levels):
            updates = {}
            for local_col in range(begin, end):
                later = [
                    (row, block_lookup[(row, local_col)])
                    for row in range(local_col + 1, len(bodies))
                    if (row, local_col) in block_lookup
                ]
                assert all(row >= end for row, _ in later)
                for _, slot in later:
                    level_column_slots.append(slot)
                    level_column_diag_slots.append(block_lookup[(local_col, local_col)])
                for i, (left_row, left_slot) in enumerate(later):
                    for right_row, right_slot in later[: i + 1]:
                        dst = block_lookup[(left_row, right_row)]
                        updates.setdefault(dst, []).append((left_slot, right_slot))
            for dst, terms in sorted(updates.items()):
                level_update_dst_slots.append(dst)
                for left, right in terms:
                    level_update_left_slots.append(left)
                    level_update_right_slots.append(right)
                level_update_term_offsets.append(len(level_update_left_slots))
            level_body_offsets.append(body_start + end)
            level_column_offsets.append(len(level_column_slots))
            level_update_offsets.append(len(level_update_dst_slots))
        articulation_level_offsets.append(len(level_body_offsets) - 1)

    articulation_bodies_np = np.asarray(articulation_bodies_host, dtype=np.int32)
    articulation_joints_np = np.asarray(articulation_joints_host, dtype=np.int32)
    articulation_joint_body_start_np = np.asarray(articulation_joint_body_start_host, dtype=np.int32)
    articulation_body_offsets_np = np.asarray(articulation_body_offsets_host, dtype=np.int32)
    articulation_joint_offsets_np = np.asarray(articulation_joint_offsets_host, dtype=np.int32)
    block_row_offsets_np = np.asarray(block_row_offsets_host, dtype=np.int32)
    block_cols_np = np.asarray(block_cols_host, dtype=np.int32)
    block_col_offsets_host.append(len(block_col_rows_host))
    factor_update_offsets_host.append(len(factor_update_dst_slots_host))
    block_col_offsets_np = np.asarray(block_col_offsets_host, dtype=np.int32)
    block_col_rows_np = np.asarray(block_col_rows_host, dtype=np.int32)
    block_col_slots_np = np.asarray(block_col_slots_host, dtype=np.int32)
    factor_update_offsets_np = np.asarray(factor_update_offsets_host, dtype=np.int32)
    factor_update_dst_slots_np = np.asarray(factor_update_dst_slots_host, dtype=np.int32)
    factor_update_left_slots_np = np.asarray(factor_update_left_slots_host, dtype=np.int32)
    factor_update_right_slots_np = np.asarray(factor_update_right_slots_host, dtype=np.int32)
    diag_slots_np = np.asarray(diag_slots_host, dtype=np.int32)
    # Cross-capture blocks follow the factorized blocks in the same value array.
    block_count = len(block_cols_host)
    joint_cross_slot_host[joint_cross_slot_host >= 0] += block_count
    elastic_group_local_np = np.asarray(
        [body_articulation_local_host[body] for body in elastic_group_body_host], dtype=np.int32
    )

    # Variable-size block storage, computed in int64 and checked against int32 indexing.
    row_dims_np = np.asarray(row_dims_host, dtype=np.int64)
    row_scalar_offsets_np = np.zeros(len(row_dims_host) + 1, dtype=np.int64)
    np.cumsum(row_dims_np, out=row_scalar_offsets_np[1:])
    row_of_slot = np.repeat(np.arange(len(row_dims_host), dtype=np.int64), np.diff(block_row_offsets_np))
    row_body_start = np.repeat(
        articulation_body_offsets_np[:-1].astype(np.int64), np.diff(articulation_body_offsets_np)
    )
    block_dims_np = np.zeros((block_count, 2), dtype=np.int64)
    if block_count > 0:
        block_dims_np[:, 0] = row_dims_np[row_of_slot]
        block_dims_np[:, 1] = row_dims_np[row_body_start[row_of_slot] + block_cols_np.astype(np.int64)]
    block_value_offsets_np = np.zeros(block_count + 1, dtype=np.int64)
    np.cumsum(block_dims_np[:, 0] * block_dims_np[:, 1], out=block_value_offsets_np[1:])
    scalar_count = int(row_scalar_offsets_np[-1])
    value_count = int(block_value_offsets_np[-1])
    for name, count in (
        ("block values", value_count + 36 * len(cross_joints)),
        ("right-hand side scalars", scalar_count),
        ("factor updates", len(factor_update_dst_slots_host)),
    ):
        if count > _INT32_MAX:
            raise ValueError(
                f"The block-sparse articulation layout needs {count} {name}, which exceeds 32-bit indexing "
                f"({_INT32_MAX}). Reduce the articulation sizes or modal counts, or use rigid_articulation_solve='local'."
            )

    rigid_body_row_count = int(articulation_body_offsets_np[rigid_group_count])
    rigid_joint_count = int(articulation_joint_offsets_np[rigid_group_count])
    rigid_block_count = int(block_row_offsets_np[rigid_body_row_count])

    # Gather entries and joint lists of the mixed block slots (see assemble_articulation_joint_terms_variable).
    slot_entry_counts: list[int] = []
    slot_rows: list[int] = []
    slot_joint_offsets: list[int] = [0]
    slot_joints: list[int] = []
    if coupled:
        for articulation in range(rigid_group_count, len(articulation_groups)):
            body_start = int(articulation_body_offsets_np[articulation])
            joint_begin = int(articulation_joint_offsets_np[articulation])
            joint_end = int(articulation_joint_offsets_np[articulation + 1])
            endpoint_locals = []
            for j in range(joint_begin, joint_end):
                joint_idx = int(articulation_joints_np[j])
                parent = int(joint_parent[joint_idx])
                child = int(joint_child[joint_idx])
                parent_local = int(body_articulation_local_host[parent]) if parent >= 0 else -1
                child_local = int(body_articulation_local_host[child]) if child >= 0 else -1
                endpoint_locals.append((j - rigid_joint_count, parent_local, child_local))
            for row in range(body_start, int(articulation_body_offsets_np[articulation + 1])):
                row_local = row - body_start
                for slot in range(int(block_row_offsets_np[row]), int(block_row_offsets_np[row + 1])):
                    col_local = int(block_cols_np[slot])
                    rows, cols = int(block_dims_np[slot, 0]), int(block_dims_np[slot, 1])
                    slot_rows.append(row)
                    if col_local == row_local:
                        slot_entry_counts.append(rows * cols + rows)
                        slot_joints.extend(
                            j
                            for j, parent_local, child_local in endpoint_locals
                            if parent_local == row_local or child_local == row_local
                        )
                    else:
                        slot_entry_counts.append(rows * cols)
                        slot_joints.extend(
                            j
                            for j, parent_local, child_local in endpoint_locals
                            if parent_local >= 0 and {parent_local, child_local} == {row_local, col_local}
                        )
                    slot_joint_offsets.append(len(slot_joints))
    # Modal twist scratch of the mixed joints' endpoints (see assemble_articulation_joint_terms_variable).
    twist_counts: list[int] = []
    if coupled:
        for joint_idx in articulation_joints_np[rigid_joint_count:]:
            for body, endpoint in (
                (int(joint_parent[joint_idx]), int(parent_endpoint[joint_idx])),
                (int(joint_child[joint_idx]), int(child_endpoint[joint_idx])),
            ):
                elastic_index = int(body_elastic_index[body]) if body >= 0 else -1
                twist_counts.append(
                    int(elastic_mode_count[elastic_index]) if elastic_index >= 0 and endpoint >= 0 else 0
                )
    wide_rows = 0
    wide_footprint = 0
    if coupled:
        for row in range(rigid_body_row_count, len(row_dims_host)):
            n = int(row_dims_np[row])
            if n <= 6:
                continue
            body_start = int(row_body_start[row])
            below = sum(
                int(row_dims_np[body_start + int(local_row)])
                for local_row in block_col_rows_np[block_col_offsets_np[row] : block_col_offsets_np[row + 1]]
            )
            wide_rows = max(wide_rows, n + below)
            wide_footprint = max(wide_footprint, n * (n + max(below, 1)))
    twist_offsets = np.zeros(len(twist_counts) + 1, dtype=np.int64)
    np.cumsum(np.asarray(twist_counts, dtype=np.int64), out=twist_offsets[1:])
    slot_entry_offsets = np.zeros(len(slot_entry_counts) + 1, dtype=np.int64)
    np.cumsum(np.asarray(slot_entry_counts, dtype=np.int64), out=slot_entry_offsets[1:])
    if max(int(slot_entry_offsets[-1]), 4 * int(twist_offsets[-1])) > _INT32_MAX:
        raise ValueError(
            f"The block-sparse articulation layout needs {int(slot_entry_offsets[-1])} gather entries and "
            f"{4 * int(twist_offsets[-1])} endpoint twist vectors, which "
            f"exceeds 32-bit indexing ({_INT32_MAX}). Reduce the articulation sizes or modal counts, or use "
            f"rigid_articulation_solve='local'."
        )

    return RigidArticulationSparseLayout(
        articulation_body_offsets=wp.array(articulation_body_offsets_np, dtype=wp.int32, device=device),
        articulation_joint_offsets=wp.array(articulation_joint_offsets_np, dtype=wp.int32, device=device),
        articulation_bodies=wp.array(articulation_bodies_np, dtype=wp.int32, device=device),
        articulation_joints=wp.array(articulation_joints_np, dtype=wp.int32, device=device),
        articulation_joint_body_start=wp.array(articulation_joint_body_start_np, dtype=wp.int32, device=device),
        articulation_block_row_offsets=wp.array(block_row_offsets_np, dtype=wp.int32, device=device),
        articulation_block_cols=wp.array(block_cols_np, dtype=wp.int32, device=device),
        articulation_block_col_offsets=wp.array(block_col_offsets_np, dtype=wp.int32, device=device),
        articulation_block_col_rows=wp.array(block_col_rows_np, dtype=wp.int32, device=device),
        articulation_block_col_slots=wp.array(block_col_slots_np, dtype=wp.int32, device=device),
        articulation_factor_update_offsets=wp.array(factor_update_offsets_np, dtype=wp.int32, device=device),
        articulation_factor_update_dst_slots=wp.array(factor_update_dst_slots_np, dtype=wp.int32, device=device),
        articulation_factor_update_left_slots=wp.array(factor_update_left_slots_np, dtype=wp.int32, device=device),
        articulation_factor_update_right_slots=wp.array(factor_update_right_slots_np, dtype=wp.int32, device=device),
        articulation_diag_slots=wp.array(diag_slots_np, dtype=wp.int32, device=device),
        body_articulation_local=wp.array(body_articulation_local_host, dtype=wp.int32, device=device),
        local_body_color_groups=local_body_color_groups,
        articulation_count=len(articulation_groups),
        articulation_body_count=len(articulation_bodies_host),
        articulation_joint_count=len(articulation_joints_host),
        local_body_count=local_body_count,
        block_count=block_count,
        elastic_bodies=wp.array(np.asarray(elastic_bodies_host, dtype=np.int32), dtype=wp.int32, device=device),
        elastic_body_row_start=wp.array(
            np.asarray(elastic_body_row_start_host, dtype=np.int32), dtype=wp.int32, device=device
        ),
        elastic_group_offsets=wp.array(
            np.asarray(elastic_group_offsets_host, dtype=np.int32), dtype=wp.int32, device=device
        ),
        elastic_group_joint=wp.array(
            np.asarray(elastic_group_joint_host, dtype=np.int32), dtype=wp.int32, device=device
        ),
        elastic_group_local=wp.array(elastic_group_local_np, dtype=wp.int32, device=device),
        elastic_group_owner=wp.array(
            np.asarray(elastic_group_owner_host, dtype=np.int32), dtype=wp.int32, device=device
        ),
        elastic_group_body=wp.array(np.asarray(elastic_group_body_host, dtype=np.int32), dtype=wp.int32, device=device),
        elastic_group_neighbor_elastic=wp.array(
            np.asarray(
                [
                    int(joint >= 0 and body_elastic_index[body] >= 0)
                    for joint, body in zip(elastic_group_joint_host, elastic_group_body_host, strict=True)
                ],
                dtype=np.int32,
            ),
            dtype=wp.int32,
            device=device,
        ),
        elastic_pair_left=wp.array(np.asarray(elastic_pair_left_host, dtype=np.int32), dtype=wp.int32, device=device),
        elastic_pair_right=wp.array(np.asarray(elastic_pair_right_host, dtype=np.int32), dtype=wp.int32, device=device),
        elastic_pair_slot=wp.array(np.asarray(elastic_pair_slot_host, dtype=np.int32), dtype=wp.int32, device=device),
        joint_cross_slot=wp.array(joint_cross_slot_host, dtype=wp.int32, device=device),
        elastic_body_count=len(elastic_bodies_host),
        elastic_group_count=len(elastic_group_joint_host),
        elastic_pair_count=len(elastic_pair_slot_host),
        cross_block_count=len(cross_joints),
        elastic_solve=elastic_solve,
        rigid_articulation_count=rigid_group_count,
        rigid_body_row_count=rigid_body_row_count,
        rigid_joint_count=rigid_joint_count,
        rigid_block_count=rigid_block_count,
        mixed_articulation_count=len(articulation_groups) - rigid_group_count,
        articulation_row_dims=_int32_array(row_dims_np, device),
        articulation_row_scalar_offsets=_int32_array(row_scalar_offsets_np, device),
        articulation_block_value_offsets=_int32_array(block_value_offsets_np, device),
        articulation_block_dims=wp.array(block_dims_np.astype(np.int32), dtype=wp.int32, device=device),
        articulation_scalar_count=scalar_count,
        articulation_value_count=value_count,
        articulation_max_row_dim=int(row_dims_np.max()) if row_dims_np.size > 0 else 0,
        mixed_articulation_joints=_int32_array(articulation_joints_np[rigid_joint_count:], device),
        mixed_elastic_bodies=_int32_array(mixed_elastic_bodies_host, device),
        mixed_elastic_body_rows=_int32_array(mixed_elastic_body_rows_host, device),
        mixed_elastic_body_count=len(mixed_elastic_bodies_host),
        mixed_slot_entry_offsets=_int32_array(slot_entry_offsets, device),
        mixed_slot_entry_count=int(slot_entry_offsets[-1]),
        mixed_slot_rows=_int32_array(slot_rows, device),
        mixed_slot_joint_offsets=_int32_array(slot_joint_offsets, device),
        mixed_slot_joints=_int32_array(slot_joints, device),
        mixed_joint_twist_offsets=_int32_array(twist_offsets, device),
        mixed_twist_count=int(twist_offsets[-1]),
        mixed_max_wide_column_rows=wide_rows,
        mixed_max_wide_column_footprint=wide_footprint,
        standalone_elastic_bodies=_int32_array(standalone_elastic_bodies_host, device),
        standalone_elastic_body_count=len(standalone_elastic_bodies_host),
        levels=RigidArticulationSparseLevels(
            articulation_level_offsets=wp.array(articulation_level_offsets, dtype=wp.int32, device=device),
            level_body_offsets=wp.array(level_body_offsets, dtype=wp.int32, device=device),
            level_column_offsets=wp.array(level_column_offsets, dtype=wp.int32, device=device),
            level_column_slots=wp.array(level_column_slots, dtype=wp.int32, device=device),
            level_column_diag_slots=wp.array(level_column_diag_slots, dtype=wp.int32, device=device),
            level_update_offsets=wp.array(level_update_offsets, dtype=wp.int32, device=device),
            level_update_dst_slots=wp.array(level_update_dst_slots, dtype=wp.int32, device=device),
            level_update_term_offsets=wp.array(level_update_term_offsets, dtype=wp.int32, device=device),
            level_update_left_slots=wp.array(level_update_left_slots, dtype=wp.int32, device=device),
            level_update_right_slots=wp.array(level_update_right_slots, dtype=wp.int32, device=device),
        )
        if use_levels
        else None,
    )
