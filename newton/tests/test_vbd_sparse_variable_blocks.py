# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Variable-size block layout of the VBD block-sparse articulation solve.

Rows of a mixed articulation have size ``6 + m`` (``m`` modes of a reduced elastic body, 0 for
a rigid body). The layout tests use real models and synthetic topologies (a namespace with the
model attributes the layout builder reads), which allows arbitrary modal counts.
"""

import itertools
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.sim import JointType
from newton._src.solvers.vbd import rigid_sparse_articulation_kernels as legacy_kernels
from newton._src.solvers.vbd import rigid_sparse_articulation_variable_kernels as variable_kernels
from newton._src.solvers.vbd.joint_coordinates import JointCoordinateData
from newton._src.solvers.vbd.rigid_sparse_articulation import (
    _minimum_degree_order,
    _weighted_minimum_degree_order,
    build_rigid_articulation_sparse_layout,
)
from newton._src.solvers.vbd.rigid_sparse_articulation_variable_kernels import (
    VB_FACTOR_INDEFINITE,
    VB_FACTOR_OK,
    VB_SOLVE_NONFINITE,
    apply_articulation_variable_delta,
    launch_solve_articulation_variable,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _add_rigid_chain(builder, count: int, y: float, *, closed: bool = False, branch: bool = False):
    positions = [wp.vec3(0.25 * i, y, 1.5) for i in range(count)]
    bodies = [
        builder.add_link(
            xform=wp.transform(p, wp.quat_identity()),
            mass=1.0 + 0.2 * i,
            inertia=wp.mat33(np.diag([0.1, 0.2, 0.3]).astype(np.float32)),
        )
        for i, p in enumerate(positions)
    ]
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=bodies[0],
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(positions[0], wp.quat_identity()),
        )
    ]
    edges = [(i - 1, i) for i in range(1, count)]
    if closed:
        edges.append((count - 1, 0))
    for a, b in edges:
        mid = 0.5 * (positions[a] + positions[b])
        joints.append(
            builder.add_joint_revolute(
                parent=bodies[a],
                child=bodies[b],
                axis=(0.0, 1.0, 0.0),
                parent_xform=wp.transform(mid - positions[a], wp.quat_identity()),
                child_xform=wp.transform(mid - positions[b], wp.quat_identity()),
            )
        )
    if branch:
        root = positions[count // 2]
        p = wp.vec3(root[0], root[1] + 0.3, root[2])
        extra = builder.add_link(
            xform=wp.transform(p, wp.quat_identity()), mass=0.7, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.1)
        )
        mid = 0.5 * (root + p)
        joints.append(
            builder.add_joint_ball(
                parent=bodies[count // 2],
                child=extra,
                parent_xform=wp.transform(mid - root, wp.quat_identity()),
                child_xform=wp.transform(mid - p, wp.quat_identity()),
            )
        )
    builder.add_articulation(joints, allow_closed_loops=closed)
    return bodies


def _rigid_model(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    for i, (count, closed, branch) in enumerate(
        [(4, True, False), (1, False, False), (7, False, True), (5, True, True)]
    ):
        _add_rigid_chain(builder, count, 2.0 * i, closed=closed, branch=branch)
    builder.color()
    return builder.finalize(device=device)


def _beam_basis(length: float, mode_specs):
    beam = newton.ModalGeneratorBeam
    return beam(
        length=length, half_width_y=0.03, half_width_z=0.02, young_modulus=2.0e6, density=1000.0, mode_specs=mode_specs
    ).build()


def _beam_inertia(length: float):
    mass = 1000.0 * 4.0 * 0.03 * 0.02 * length
    inertia = np.diag(
        [
            mass * (0.03**2 + 0.02**2) / 3.0,
            mass * (length**2 / 12.0 + 0.02**2 / 3.0),
            mass * (length**2 / 12.0 + 0.03**2 / 3.0),
        ]
    )
    return mass, wp.mat33(*inertia.ravel())


def _mixed_model(device):
    """Rigid chain, a rigid base carrying an elastic leaf link, and a standalone elastic body.

    The rigid chain is added after the mixed articulation, so the coupled layout must move it
    to the rigid prefix.
    """
    beam = newton.ModalGeneratorBeam
    length = 0.6
    mass, inertia = _beam_inertia(length)
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))

    base = builder.add_link(
        xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(*np.diag([0.002, 0.02, 0.02]).ravel()),
    )
    link = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(0.4 + 0.5 * length, 0.0, 0.0), wp.quat_identity()),
        mass=mass,
        inertia=inertia,
        modal_basis=_beam_basis(
            length,
            [
                {"type": beam.Mode.AXIAL},
                {"type": beam.Mode.BENDING_Y, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
                {"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
            ],
        ),
    )
    root = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=(0.0, 0.0, 1.0),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    hinge = builder.add_joint_revolute(
        parent=base,
        child=link,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([root, hinge])

    standalone = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(0.0, 3.0, 0.0), wp.quat_identity()),
        mass=mass,
        inertia=inertia,
        modal_basis=_beam_basis(length, [{"type": beam.Mode.AXIAL}]),
    )
    pin = builder.add_joint_revolute(
        parent=-1,
        child=standalone,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(-0.5 * length, 3.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([pin])

    chain = _add_rigid_chain(builder, 3, 6.0, closed=False)
    builder.color()
    model = builder.finalize(device=device)
    return model, {"base": base, "link": link, "standalone": standalone, "chain": chain, "link_modes": 3}


def _synthetic_model(device, mode_counts, articulations, *, owner_joints_inside: bool = True):
    """Return a namespace with the attributes read by the layout builder.

    Args:
        mode_counts: Modal count per body, ``None`` for a rigid body.
        articulations: Per articulation, a list of ``(parent, child)`` physical joints
            (``-1`` is the world).
        owner_joints_inside: Put each elastic body's owner joint in its articulation range.
    """
    body_count = len(mode_counts)
    elastic_bodies = [b for b, m in enumerate(mode_counts) if m is not None]
    body_elastic_index = np.full(body_count, -1, dtype=np.int32)
    for i, b in enumerate(elastic_bodies):
        body_elastic_index[b] = i
    body_articulation = {}
    for a, joints in enumerate(articulations):
        for parent, child in joints:
            for b in (parent, child):
                if b >= 0:
                    body_articulation[b] = a

    parents, children, types, starts, ends = [], [], [], [], []
    owners_outside = []
    for a, joints in enumerate(articulations):
        starts.append(len(parents))
        for parent, child in joints:
            parents.append(parent)
            children.append(child)
            types.append(int(JointType.REVOLUTE))
        for b in elastic_bodies:
            if body_articulation.get(b) == a:
                if owner_joints_inside:
                    parents.append(-1)
                    children.append(b)
                    types.append(int(JointType.ELASTIC))
                else:
                    owners_outside.append(b)
        ends.append(len(parents))
    for b in owners_outside + [b for b in elastic_bodies if b not in body_articulation]:
        parents.append(-1)
        children.append(b)
        types.append(int(JointType.ELASTIC))

    parents_np = np.asarray(parents, dtype=np.int32)
    children_np = np.asarray(children, dtype=np.int32)
    types_np = np.asarray(types, dtype=np.int32)
    physical = types_np != int(JointType.ELASTIC)
    parent_endpoint = np.where(physical & (parents_np >= 0) & (body_elastic_index[parents_np] >= 0), 0, -1)
    child_endpoint = np.where(physical & (children_np >= 0) & (body_elastic_index[children_np] >= 0), 0, -1)

    def arr(values):
        return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

    return SimpleNamespace(
        body_count=body_count,
        joint_count=len(parents),
        joint_parent=arr(parents_np),
        joint_child=arr(children_np),
        joint_type=arr(types_np),
        elastic_body_count=len(elastic_bodies),
        body_elastic_index=arr(body_elastic_index),
        elastic_mode_count=arr([mode_counts[b] for b in elastic_bodies]),
        joint_parent_elastic_endpoint=arr(parent_endpoint),
        joint_child_elastic_endpoint=arr(child_endpoint),
        articulation_count=len(articulations),
        articulation_start=arr(starts),
        articulation_end=arr(ends),
        body_color_groups=[arr(np.arange(body_count))],
    )


def _layout_numpy(layout) -> dict:
    out = {}
    for name, value in vars(layout).items():
        if isinstance(value, wp.array):
            out[name] = value.numpy()
        elif isinstance(value, list):
            out[name] = [item.numpy() for item in value]
        else:
            out[name] = value
    return out


def _assert_layouts_equal(test, a, b, skip=()):
    na, nb = _layout_numpy(a), _layout_numpy(b)
    test.assertEqual(set(na), set(nb))
    for name in na:
        if name in skip:
            continue
        x, y = na[name], nb[name]
        if isinstance(x, list):
            test.assertEqual(len(x), len(y), name)
            for xi, yi in zip(x, y, strict=True):
                np.testing.assert_array_equal(xi, yi, err_msg=name)
        elif isinstance(x, np.ndarray):
            test.assertEqual(x.dtype, y.dtype, name)
            np.testing.assert_array_equal(x, y, err_msg=name)
        else:
            test.assertEqual(x, y, name)


def _model_edges(model, layout, articulation: int) -> set[tuple[int, int]]:
    """Ordered local graph edges of one articulation, from the model's physical joints."""
    parent = model.joint_parent.numpy()
    child = model.joint_child.numpy()
    joint_type = model.joint_type.numpy()
    local = layout.body_articulation_local.numpy()
    body_offsets = layout.articulation_body_offsets.numpy()
    bodies = set(layout.articulation_bodies.numpy()[body_offsets[articulation] : body_offsets[articulation + 1]])
    joint_offsets = layout.articulation_joint_offsets.numpy()
    edges = set()
    for joint in layout.articulation_joints.numpy()[joint_offsets[articulation] : joint_offsets[articulation + 1]]:
        p, c = int(parent[joint]), int(child[joint])
        if joint_type[joint] == int(JointType.ELASTIC) or p < 0 or c < 0 or p not in bodies or c not in bodies:
            continue
        a, b = int(local[p]), int(local[c])
        if a != b:
            edges.add((max(a, b), min(a, b)))
    return edges


def _dense_symbolic_elimination(count: int, edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Lower-triangle nonzeros of the Cholesky factor by boolean Gaussian elimination."""
    nz = np.eye(count, dtype=bool)
    for a, b in edges:
        nz[a, b] = nz[b, a] = True
    for k in range(count):
        rows = [i for i in range(k + 1, count) if nz[i, k]]
        for i in rows:
            for j in rows:
                nz[i, j] = True
    return {(i, j) for i in range(count) for j in range(i + 1) if nz[i, j]}


def _stored_pattern(layout, articulation: int) -> set[tuple[int, int]]:
    body_offsets = layout.articulation_body_offsets.numpy()
    row_offsets = layout.articulation_block_row_offsets.numpy()
    cols = layout.articulation_block_cols.numpy()
    start = body_offsets[articulation]
    pattern = set()
    for local_row in range(body_offsets[articulation + 1] - start):
        row = start + local_row
        for slot in range(row_offsets[row], row_offsets[row + 1]):
            pattern.add((local_row, int(cols[slot])))
    return pattern


def _check_variable_layout_consistency(test, layout):
    """Check prefix sums, dims and the CSC/update lists of every articulation."""
    dims = layout.articulation_row_dims.numpy()
    scalar_offsets = layout.articulation_row_scalar_offsets.numpy()
    value_offsets = layout.articulation_block_value_offsets.numpy()
    block_dims = layout.articulation_block_dims.numpy()
    body_offsets = layout.articulation_body_offsets.numpy()
    joint_offsets = layout.articulation_joint_offsets.numpy()
    row_offsets = layout.articulation_block_row_offsets.numpy()
    cols = layout.articulation_block_cols.numpy()
    diag = layout.articulation_diag_slots.numpy()
    col_offsets = layout.articulation_block_col_offsets.numpy()
    col_rows = layout.articulation_block_col_rows.numpy()
    col_slots = layout.articulation_block_col_slots.numpy()
    update_offsets = layout.articulation_factor_update_offsets.numpy()
    update_dst = layout.articulation_factor_update_dst_slots.numpy()
    update_left = layout.articulation_factor_update_left_slots.numpy()
    update_right = layout.articulation_factor_update_right_slots.numpy()

    rows = layout.articulation_body_count
    test.assertEqual(dims.shape, (rows,))
    test.assertEqual(scalar_offsets.shape, (rows + 1,))
    test.assertEqual(value_offsets.shape, (layout.block_count + 1,))
    test.assertEqual(block_dims.shape, (layout.block_count, 2))
    test.assertEqual(scalar_offsets[0], 0)
    np.testing.assert_array_equal(np.diff(scalar_offsets), dims)
    np.testing.assert_array_equal(np.diff(value_offsets), block_dims[:, 0] * block_dims[:, 1])
    test.assertEqual(layout.articulation_scalar_count, int(scalar_offsets[-1]))
    test.assertEqual(layout.articulation_value_count, int(value_offsets[-1]))
    test.assertEqual(layout.articulation_max_row_dim, int(dims.max()) if rows else 0)

    test.assertEqual(layout.rigid_body_row_count, int(body_offsets[layout.rigid_articulation_count]))
    test.assertEqual(layout.rigid_joint_count, int(joint_offsets[layout.rigid_articulation_count]))
    test.assertEqual(layout.rigid_block_count, int(row_offsets[layout.rigid_body_row_count]))
    test.assertEqual(layout.mixed_articulation_count, layout.articulation_count - layout.rigid_articulation_count)
    np.testing.assert_array_equal(dims[: layout.rigid_body_row_count], 6)
    prefix = layout.rigid_block_count
    np.testing.assert_array_equal(value_offsets[: prefix + 1], 36 * np.arange(prefix + 1))
    np.testing.assert_array_equal(
        scalar_offsets[: layout.rigid_body_row_count + 1], 6 * np.arange(layout.rigid_body_row_count + 1)
    )
    np.testing.assert_array_equal(
        layout.mixed_articulation_joints.numpy(), layout.articulation_joints.numpy()[layout.rigid_joint_count :]
    )

    for a in range(layout.articulation_count):
        start, end = int(body_offsets[a]), int(body_offsets[a + 1])
        lookup = {}
        for local_row in range(end - start):
            row = start + local_row
            for slot in range(row_offsets[row], row_offsets[row + 1]):
                local_col = int(cols[slot])
                test.assertLessEqual(local_col, local_row)
                lookup[(local_row, local_col)] = slot
                test.assertEqual(tuple(block_dims[slot]), (dims[row], dims[start + local_col]))
            test.assertEqual(int(cols[diag[row]]), local_row)
            test.assertGreaterEqual(diag[row], row_offsets[row])
            test.assertLess(diag[row], row_offsets[row + 1])
        for local_col in range(end - start):
            row = start + local_col
            expected_col = [(i, s) for (i, j), s in sorted(lookup.items()) if j == local_col and i > local_col]
            stored_col = list(
                zip(
                    col_rows[col_offsets[row] : col_offsets[row + 1]].tolist(),
                    col_slots[col_offsets[row] : col_offsets[row + 1]].tolist(),
                    strict=True,
                )
            )
            test.assertEqual(stored_col, expected_col)
            expected_updates = []
            for li, (left_row, left_slot) in enumerate(expected_col):
                for right_row, right_slot in expected_col[: li + 1]:
                    expected_updates.append((lookup[(left_row, right_row)], left_slot, right_slot))
            begin, finish = update_offsets[row], update_offsets[row + 1]
            stored_updates = list(
                zip(
                    update_dst[begin:finish].tolist(),
                    update_left[begin:finish].tolist(),
                    update_right[begin:finish].tolist(),
                    strict=True,
                )
            )
            test.assertEqual(stored_updates, expected_updates)


# ---------------------------------------------------------------------------
# Layout tests
# ---------------------------------------------------------------------------


def test_all_rigid_layout_identical_under_both_backends(test, device):
    """A rigid-only model has the same layout under both backends, with 6 x 6 offsets."""
    model = _rigid_model(device)
    schur = build_rigid_articulation_sparse_layout(model, device)
    explicit_schur = build_rigid_articulation_sparse_layout(model, device, elastic_solve="schur")
    coupled = build_rigid_articulation_sparse_layout(model, device, elastic_solve="coupled")
    _assert_layouts_equal(test, schur, explicit_schur)
    _assert_layouts_equal(test, schur, coupled, skip=("elastic_solve",))
    test.assertEqual(coupled.elastic_solve, "coupled")
    test.assertEqual(coupled.rigid_articulation_count, coupled.articulation_count)
    test.assertEqual(coupled.mixed_articulation_count, 0)
    test.assertEqual(coupled.rigid_body_row_count, coupled.articulation_body_count)
    test.assertEqual(coupled.rigid_block_count, coupled.block_count)
    np.testing.assert_array_equal(
        coupled.articulation_block_value_offsets.numpy(), 36 * np.arange(coupled.block_count + 1)
    )
    np.testing.assert_array_equal(
        coupled.articulation_row_scalar_offsets.numpy(), 6 * np.arange(coupled.articulation_body_count + 1)
    )
    np.testing.assert_array_equal(coupled.articulation_block_dims.numpy(), 6)
    test.assertEqual(coupled.articulation_value_count, 36 * coupled.block_count)
    test.assertEqual(coupled.mixed_elastic_body_count, 0)
    test.assertEqual(coupled.standalone_elastic_body_count, 0)
    _check_variable_layout_consistency(test, coupled)


def test_unknown_elastic_solve_raises(test, device):
    """Reject an unknown ``elastic_solve`` value when building the layout."""
    model = _rigid_model(device)
    with test.assertRaisesRegex(ValueError, "elastic_solve"):
        build_rigid_articulation_sparse_layout(model, device, elastic_solve="dense")


def test_mixed_layout_partition_and_dims(test, device):
    """Real model: rigid prefix first, standalone elastic body left local, variable dims."""
    model, ids = _mixed_model(device)
    schur = build_rigid_articulation_sparse_layout(model, device)
    coupled = build_rigid_articulation_sparse_layout(model, device, elastic_solve="coupled")

    # Schur keeps the model order, includes the standalone body, and uses 6 x 6 rows only.
    test.assertEqual(schur.articulation_count, 3)
    test.assertEqual(schur.rigid_articulation_count, 3)
    test.assertEqual(schur.mixed_articulation_count, 0)
    np.testing.assert_array_equal(schur.articulation_row_dims.numpy(), 6)
    test.assertEqual(set(schur.elastic_bodies.numpy().tolist()), {ids["link"], ids["standalone"]})
    test.assertEqual(schur.standalone_elastic_body_count, 0)
    _check_variable_layout_consistency(test, schur)

    test.assertEqual(coupled.articulation_count, 2)
    test.assertEqual(coupled.rigid_articulation_count, 1)
    test.assertEqual(coupled.mixed_articulation_count, 1)
    bodies = coupled.articulation_bodies.numpy()
    test.assertEqual(set(bodies[: coupled.rigid_body_row_count].tolist()), set(ids["chain"]))
    test.assertEqual(set(bodies[coupled.rigid_body_row_count :].tolist()), {ids["base"], ids["link"]})
    dims = coupled.articulation_row_dims.numpy()
    for row, body in enumerate(bodies):
        test.assertEqual(dims[row], 6 + ids["link_modes"] if body == ids["link"] else 6)
    np.testing.assert_array_equal(coupled.mixed_elastic_bodies.numpy(), [ids["link"]])
    np.testing.assert_array_equal(coupled.mixed_elastic_body_rows.numpy(), np.nonzero(bodies == ids["link"])[0])
    test.assertEqual(coupled.articulation_max_row_dim, 6 + ids["link_modes"])

    # The standalone elastic body is local; Schur-only fields are empty.
    np.testing.assert_array_equal(coupled.standalone_elastic_bodies.numpy(), [ids["standalone"]])
    test.assertEqual(coupled.body_articulation_local.numpy()[ids["standalone"]], -1)
    local_bodies = np.concatenate([g.numpy() for g in coupled.local_body_color_groups])
    test.assertIn(ids["standalone"], local_bodies.tolist())
    test.assertNotIn(ids["standalone"], bodies.tolist())
    test.assertEqual(coupled.elastic_body_count, 0)
    test.assertEqual(coupled.elastic_group_count, 0)
    test.assertEqual(coupled.elastic_pair_count, 0)
    test.assertEqual(coupled.cross_block_count, 0)
    np.testing.assert_array_equal(coupled.joint_cross_slot.numpy(), -1)
    test.assertEqual(coupled.joint_cross_slot.shape, (model.joint_count,))
    _check_variable_layout_consistency(test, coupled)

    # The rigid prefix equals the layout of the rigid chain alone.
    rigid_offsets = coupled.articulation_block_value_offsets.numpy()
    test.assertEqual(rigid_offsets[coupled.rigid_block_count], 36 * coupled.rigid_block_count)


def test_synthetic_mixed_layout_consistency(test, device):
    """Synthetic mixed topologies: loops, adjacent elastic bodies, heterogeneous modes, replication."""
    mode_counts = [None, 25, None, 1, 157, None, None, 58, 0, None, 7, 26, 25, None, 10]
    articulations = [
        # Mixed: rigid base, elastic chain with a loop closure over a rigid coupler.
        [(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 1)],
        # Rigid only, listed after a mixed one.
        [(-1, 5), (5, 6)],
        # Two adjacent elastic links (58 modes and a zero-mode elastic body) and a rigid tip.
        [(-1, 7), (7, 8), (8, 9)],
        # Same graph as above with other modal counts: the cache must not reuse its order.
        [(-1, 10), (10, 11), (11, 13)],
        # Standalone elastic body pinned to the world.
        [(-1, 12)],
        # A second standalone elastic body.
        [(-1, 14)],
    ]
    model = _synthetic_model(device, mode_counts, articulations, owner_joints_inside=False)
    layout = build_rigid_articulation_sparse_layout(model, device, elastic_solve="coupled")
    _check_variable_layout_consistency(test, layout)
    # Rigid articulations first, then the mixed ones in model order; standalone ones dropped.
    test.assertEqual(layout.rigid_articulation_count, 1)
    test.assertEqual(layout.mixed_articulation_count, 3)
    np.testing.assert_array_equal(layout.standalone_elastic_bodies.numpy(), [12, 14])
    bodies = layout.articulation_bodies.numpy()
    body_offsets = layout.articulation_body_offsets.numpy()
    test.assertEqual(set(bodies[body_offsets[0] : body_offsets[1]].tolist()), {5, 6})
    test.assertEqual(set(bodies[body_offsets[1] : body_offsets[2]].tolist()), {0, 1, 2, 3, 4})
    dims = layout.articulation_row_dims.numpy()
    for row, body in enumerate(bodies):
        test.assertEqual(dims[row], 6 + (mode_counts[body] or 0))
    test.assertEqual(layout.articulation_max_row_dim, 163)
    elastic_rows = layout.mixed_elastic_body_rows.numpy()
    np.testing.assert_array_equal(bodies[elastic_rows], layout.mixed_elastic_bodies.numpy())
    test.assertEqual(
        sorted(layout.mixed_elastic_bodies.numpy().tolist()), [b for b in range(12) if mode_counts[b] is not None]
    )
    # The two same-graph articulations differ only in dims and are ordered by their own costs.
    order_a = [int(bodies[r]) for r in range(body_offsets[2], body_offsets[3])]
    order_b = [int(bodies[r]) for r in range(body_offsets[3], body_offsets[4])]
    test.assertEqual(order_a, [9, 7, 8])
    test.assertEqual(order_b, [13, 11, 10])


def test_symbolic_pattern_matches_independent_elimination(test, device):
    """Stored fill equals a boolean Gaussian elimination of the ordered joint graph."""
    rng = np.random.default_rng(3)
    cases = []
    for trial in range(6):
        count = 3 + trial * 2
        modes = [None if rng.random() < 0.5 else int(rng.integers(0, 40)) for _ in range(count)]
        modes[0] = 12
        joints = [(-1, 0)] + [(int(rng.integers(0, i)), i) for i in range(1, count)]
        for _ in range(trial):
            a, b = sorted(rng.choice(count, size=2, replace=False).tolist())
            joints.append((a, b))
        cases.append((modes, joints))
    for modes, joints in cases:
        model = _synthetic_model(device, modes, [joints])
        for backend in ("schur", "coupled"):
            layout = build_rigid_articulation_sparse_layout(model, device, elastic_solve=backend)
            edges = _model_edges(model, layout, 0)
            if backend == "schur":
                # The Schur backend adds a clique over each elastic body's joint neighbors.
                test.assertTrue(_stored_pattern(layout, 0) >= _dense_symbolic_elimination(len(modes), edges))
                continue
            test.assertEqual(_stored_pattern(layout, 0), _dense_symbolic_elimination(len(modes), edges))
    rigid = _rigid_model(device)
    layout = build_rigid_articulation_sparse_layout(rigid, device, elastic_solve="coupled")
    for a in range(layout.articulation_count):
        count = int(np.diff(layout.articulation_body_offsets.numpy())[a])
        test.assertEqual(_stored_pattern(layout, a), _dense_symbolic_elimination(count, _model_edges(rigid, layout, a)))


def test_coupled_layout_adds_no_elastic_clique(test, device):
    """An elastic hub with three rigid neighbors needs no fill under "coupled".

    With a zero-mode hub all blocks have size 6, so the leaves are eliminated first and the
    factor has no fill; the Schur backend adds the neighbor clique of the elastic hub.
    """
    model = _synthetic_model(device, [None, 0, None, None], [[(-1, 0), (0, 1), (1, 2), (1, 3)]])
    schur = build_rigid_articulation_sparse_layout(model, device, elastic_solve="schur")
    coupled = build_rigid_articulation_sparse_layout(model, device, elastic_solve="coupled")
    test.assertEqual(coupled.block_count, 4 + 3)
    test.assertGreater(schur.block_count, coupled.block_count)


def test_weighted_order_reduces_to_minimum_degree(test, device):
    """With equal block sizes the weighted order equals the minimum-degree order."""
    rng = np.random.default_rng(11)
    for count in range(1, 14):
        for _trial in range(8):
            density = rng.uniform(0.1, 0.7)
            edges = {(a, b) for a, b in itertools.combinations(range(count), 2) if rng.random() < density} | {
                (i, i) for i in range(count) if rng.random() < 0.1
            }
            reference = _minimum_degree_order(count, set(edges))
            for size in (6, 1, 40):
                test.assertEqual(_weighted_minimum_degree_order(count, set(edges), [size] * count), reference)
    # Path rigid - elastic - rigid: eliminating the large middle block first is cheaper.
    test.assertEqual(_minimum_degree_order(3, {(0, 1), (1, 2)}), [0, 1, 2])
    test.assertEqual(_weighted_minimum_degree_order(3, {(0, 1), (1, 2)}, [6, 106, 6]), [1, 0, 2])


def test_layout_rejects_int32_overflow(test, device):
    """A layout whose block values exceed 32-bit indexing raises instead of wrapping."""
    model = _synthetic_model(device, [40000, 40000], [[(-1, 0), (0, 1)]])
    with test.assertRaisesRegex(ValueError, "32-bit"):
        build_rigid_articulation_sparse_layout(model, device, elastic_solve="coupled")
    # The Schur backend stores 6 x 6 blocks and is unaffected.
    build_rigid_articulation_sparse_layout(model, device, elastic_solve="schur")


# ---------------------------------------------------------------------------
# Variable-block factorization and solve
# ---------------------------------------------------------------------------


def _variable_solvers(device):
    """Variable-block solve launchers to check on ``device``, as ``(name, launcher)`` pairs.

    Both kernels run on every device: the serial kernel uses one thread per articulation, the
    cooperative kernel one CTA per articulation (``block_dim = 1`` on the CPU, where its tile
    operations run serially).
    """
    return [
        ("serial", variable_kernels.launch_solve_articulation_variable_serial),
        ("cooperative", variable_kernels.launch_solve_articulation_variable_cooperative),
    ]


def _articulation_rows(layout, articulation: int) -> tuple[int, int]:
    offsets = layout.articulation_body_offsets.numpy()
    return int(offsets[articulation]), int(offsets[articulation + 1])


def _random_system(layout, model, mode_counts, rng, *, edge_scale: float = 0.5, scale_decades: float = 0.0):
    """Random SPD system on each mixed articulation's joint graph, in model body order.

    Returns ``(H, rhs)`` dicts per mixed articulation, where ``H`` is in layout (eliminated)
    row order and ``P`` the permutation such that ``H = P H_model P^T``. With ``scale_decades``
    the matrix is scaled symmetrically, ``S H S``, by a diagonal ``S`` with entries spread
    log-uniformly over ``10**[-scale_decades, scale_decades]`` (stored per layout row as ``scale``).
    """
    bodies = layout.articulation_bodies.numpy()
    dims_all = layout.articulation_row_dims.numpy()
    parent = model.joint_parent.numpy()
    child = model.joint_child.numpy()
    joint_type = model.joint_type.numpy()
    joint_offsets = layout.articulation_joint_offsets.numpy()
    joints = layout.articulation_joints.numpy()
    systems = {}
    for a in range(layout.rigid_articulation_count, layout.articulation_count):
        start, end = _articulation_rows(layout, a)
        model_bodies = sorted(int(b) for b in bodies[start:end])
        dims = [6 + (mode_counts[b] or 0) for b in model_bodies]
        offsets = np.concatenate([[0], np.cumsum(dims)])
        size = int(offsets[-1])
        index = {b: i for i, b in enumerate(model_bodies)}
        H = np.zeros((size, size))
        for i, n in enumerate(dims):
            M = rng.normal(size=(n, n))
            H[offsets[i] : offsets[i + 1], offsets[i] : offsets[i + 1]] += M @ M.T / n + np.eye(n)
        for joint in joints[joint_offsets[a] : joint_offsets[a + 1]]:
            p, c = int(parent[joint]), int(child[joint])
            if joint_type[joint] == int(JointType.ELASTIC) or p < 0 or c < 0:
                continue
            idx = np.concatenate(
                [
                    np.arange(offsets[index[p]], offsets[index[p] + 1]),
                    np.arange(offsets[index[c]], offsets[index[c] + 1]),
                ]
            )
            E = rng.normal(size=(min(len(idx), 12), len(idx))) * edge_scale
            H[np.ix_(idx, idx)] += E.T @ E
        # Permutation into layout row order.
        perm = np.concatenate(
            [np.arange(offsets[index[int(b)]], offsets[index[int(b)] + 1]) for b in bodies[start:end]]
        )
        np.testing.assert_array_equal([dims[index[int(b)]] for b in bodies[start:end]], dims_all[start:end])
        scale = 10.0 ** rng.uniform(-scale_decades, scale_decades, size=size)
        H_model = scale[:, None] * H * scale[None, :]
        H_layout = H_model[np.ix_(perm, perm)]
        systems[a] = {
            "H_model": H_model,
            "perm": perm,
            "H": H_layout,
            "rhs": rng.normal(size=size) * scale[perm],
            "scale": scale[perm],
        }
    return systems


def _scatter_system(layout, systems, rng):
    """Write the lower block triangle of each system into flat value and rhs arrays.

    Rigid-prefix entries and fill slots get sentinels/zeros so untouched storage can be checked.
    """
    value_offsets = layout.articulation_block_value_offsets.numpy()
    block_dims = layout.articulation_block_dims.numpy()
    scalar_offsets = layout.articulation_row_scalar_offsets.numpy()
    row_offsets = layout.articulation_block_row_offsets.numpy()
    cols = layout.articulation_block_cols.numpy()
    values = rng.normal(size=layout.articulation_value_count)
    rhs = rng.normal(size=layout.articulation_scalar_count)
    for a, system in systems.items():
        start, end = _articulation_rows(layout, a)
        base = scalar_offsets[start]
        for row in range(start, end):
            for slot in range(row_offsets[row], row_offsets[row + 1]):
                col_row = start + int(cols[slot])
                r0, c0 = scalar_offsets[row] - base, scalar_offsets[col_row] - base
                R, C = block_dims[slot]
                block = system["H"][r0 : r0 + R, c0 : c0 + C]
                values[value_offsets[slot] : value_offsets[slot + 1]] = block.ravel()
        rhs[scalar_offsets[start] : scalar_offsets[end]] = system["rhs"]
    return values, rhs


def _dense_factor(layout, values, articulation: int) -> np.ndarray:
    """Assemble the dense lower factor of one articulation from factored block values."""
    value_offsets = layout.articulation_block_value_offsets.numpy()
    block_dims = layout.articulation_block_dims.numpy()
    scalar_offsets = layout.articulation_row_scalar_offsets.numpy()
    row_offsets = layout.articulation_block_row_offsets.numpy()
    cols = layout.articulation_block_cols.numpy()
    start, end = _articulation_rows(layout, articulation)
    base = scalar_offsets[start]
    size = scalar_offsets[end] - base
    L = np.zeros((size, size))
    for row in range(start, end):
        for slot in range(row_offsets[row], row_offsets[row + 1]):
            col_row = start + int(cols[slot])
            r0, c0 = scalar_offsets[row] - base, scalar_offsets[col_row] - base
            R, C = block_dims[slot]
            block = np.asarray(values[value_offsets[slot] : value_offsets[slot + 1]], dtype=np.float64).reshape(R, C)
            if col_row == row:
                block = np.tril(block)
            L[r0 : r0 + R, c0 : c0 + C] = block
    return L


def _reference_layout_solve(layout, values, rhs, articulation: int) -> np.ndarray:
    """Float64 right-looking block Cholesky that walks the layout's column and update lists."""
    value_offsets = layout.articulation_block_value_offsets.numpy()
    block_dims = layout.articulation_block_dims.numpy()
    scalar_offsets = layout.articulation_row_scalar_offsets.numpy()
    diag = layout.articulation_diag_slots.numpy()
    col_offsets = layout.articulation_block_col_offsets.numpy()
    col_rows = layout.articulation_block_col_rows.numpy()
    col_slots = layout.articulation_block_col_slots.numpy()
    update_offsets = layout.articulation_factor_update_offsets.numpy()
    update_dst = layout.articulation_factor_update_dst_slots.numpy()
    update_left = layout.articulation_factor_update_left_slots.numpy()
    update_right = layout.articulation_factor_update_right_slots.numpy()
    row_offsets = layout.articulation_block_row_offsets.numpy()
    cols = layout.articulation_block_cols.numpy()
    start, end = _articulation_rows(layout, articulation)

    blocks = {}
    for row in range(start, end):
        for slot in range(row_offsets[row], row_offsets[row + 1]):
            R, C = block_dims[slot]
            blocks[slot] = np.asarray(values[value_offsets[slot] : value_offsets[slot + 1]], dtype=np.float64).reshape(
                R, C
            )
    for row in range(start, end):
        blocks[diag[row]] = np.linalg.cholesky(blocks[diag[row]])
        L_kk = blocks[diag[row]]
        for entry in range(col_offsets[row], col_offsets[row + 1]):
            slot = col_slots[entry]
            blocks[slot] = np.linalg.solve(L_kk, blocks[slot].T).T
        for entry in range(update_offsets[row], update_offsets[row + 1]):
            blocks[update_dst[entry]] = (
                blocks[update_dst[entry]] - blocks[update_left[entry]] @ blocks[update_right[entry]].T
            )

    def seg(r):
        return slice(scalar_offsets[r] - scalar_offsets[start], scalar_offsets[r + 1] - scalar_offsets[start])

    x = np.asarray(rhs[scalar_offsets[start] : scalar_offsets[end]], dtype=np.float64).copy()
    for row in range(start, end):
        for slot in range(row_offsets[row], row_offsets[row + 1]):
            if slot != diag[row]:
                x[seg(row)] -= blocks[slot] @ x[seg(start + cols[slot])]
        x[seg(row)] = np.linalg.solve(blocks[diag[row]], x[seg(row)])
    for row in range(end - 1, start - 1, -1):
        for entry in range(col_offsets[row], col_offsets[row + 1]):
            x[seg(row)] -= blocks[col_slots[entry]].T @ x[seg(start + col_rows[entry])]
        x[seg(row)] = np.linalg.solve(blocks[diag[row]].T, x[seg(row)])
    return x


def _run_solve(launcher, layout, values_np, rhs_np, device, frame_regularization: float = 0.0, delta_fill=None):
    values = wp.array(np.asarray(values_np, dtype=np.float32), dtype=float, device=device)
    rhs = wp.array(np.asarray(rhs_np, dtype=np.float32), dtype=float, device=device)
    delta_init = np.full(layout.articulation_scalar_count, 7.0 if delta_fill is None else delta_fill, dtype=np.float32)
    delta = wp.array(delta_init, dtype=float, device=device)
    diag_backup = wp.zeros(layout.articulation_scalar_count, dtype=float, device=device)
    status = wp.full(max(layout.mixed_articulation_count, 1), -1, dtype=wp.int32, device=device)
    failures = wp.zeros(1, dtype=wp.int32, device=device)
    launcher(layout, frame_regularization, rhs, values, diag_backup, status, failures, delta, device=device)
    return {
        "values": values.numpy(),
        "delta": delta.numpy(),
        "diag_backup": diag_backup.numpy(),
        "status": status.numpy(),
        "failures": int(failures.numpy()[0]),
        "rhs": rhs.numpy(),
    }


# Rigid prefix articulation, then mixed articulations whose rows have sizes 6, 7, 31, 32, 33, 64
# and 163: an adjacent elastic chain, a loop of elastic links, and a branched loop with a
# zero-mode elastic body. Modal counts per body (``None`` for rigid).
_SOLVE_MODE_COUNTS = [None, None, 157, 58, 1, None, 25, 26, 27, None, 0, None, 1, 58]
_SOLVE_ARTICULATIONS = [
    [(-1, 0), (0, 1)],
    [(-1, 2), (2, 3), (3, 4), (4, 5)],
    [(-1, 6), (6, 7), (7, 8), (8, 9), (9, 6)],
    [(-1, 10), (10, 11), (10, 12), (12, 13), (11, 13)],
]


def _solve_fixture(device, seed: int = 5, scale_decades: float = 0.0):
    model = _synthetic_model(device, _SOLVE_MODE_COUNTS, _SOLVE_ARTICULATIONS)
    layout = build_rigid_articulation_sparse_layout(model, device, elastic_solve="coupled")
    rng = np.random.default_rng(seed)
    systems = _random_system(layout, model, _SOLVE_MODE_COUNTS, rng, scale_decades=scale_decades)
    values, rhs = _scatter_system(layout, systems, rng)
    return model, layout, systems, values, rhs


def test_reference_layout_solve_matches_dense(test, device):
    """A float64 solve that walks the layout's lists equals numpy.linalg.solve."""
    _model, layout, systems, values, rhs = _solve_fixture(device)
    test.assertEqual(sorted(set(layout.articulation_row_dims.numpy().tolist())), [6, 7, 31, 32, 33, 64, 163])
    for a, system in systems.items():
        x = _reference_layout_solve(layout, values, rhs, a)
        expected = np.linalg.solve(system["H"], system["rhs"])
        np.testing.assert_allclose(x, expected, rtol=0.0, atol=1.0e-10 * np.linalg.norm(expected))


def test_variable_solve_random_spd(test, device):
    """Random SPD mixed systems: solution vs numpy and factor ``L L^T`` vs ``P H P^T``.

    The third system is scaled over four decades (condition number about 1e8); errors are
    measured in the scaled coordinates, where float32 Cholesky is accurate.
    """
    for name, launcher in _variable_solvers(device):
        for seed, decades in ((5, 0.0), (6, 0.0), (7, 2.0)):
            _model, layout, systems, values, rhs = _solve_fixture(device, seed, decades)
            out = _run_solve(launcher, layout, values, rhs, device)
            test.assertEqual(out["failures"], 0, name)
            np.testing.assert_array_equal(out["status"], VB_FACTOR_OK)
            scalar_offsets = layout.articulation_row_scalar_offsets.numpy()
            for a, system in systems.items():
                start, end = _articulation_rows(layout, a)
                delta = out["delta"][scalar_offsets[start] : scalar_offsets[end]].astype(np.float64)
                expected = np.linalg.solve(system["H"], system["rhs"])
                scale = system["scale"]
                error = np.linalg.norm(scale * (delta - expected)) / np.linalg.norm(scale * expected)
                test.assertLess(error, 1.0e-4, f"{name} articulation {a}")
                # The factor reproduces the permuted matrix.
                P = np.eye(len(system["perm"]))[system["perm"]]
                L = _dense_factor(layout, out["values"], a) / scale[:, None]
                H_unscaled = (P @ system["H_model"] @ P.T) / np.outer(scale, scale)
                factor_error = np.linalg.norm(L @ L.T - H_unscaled) / np.linalg.norm(H_unscaled)
                test.assertLess(factor_error, 1.0e-5, f"{name} articulation {a}")
            # The rigid prefix is left untouched.
            rigid_values = layout.articulation_block_value_offsets.numpy()[layout.rigid_block_count]
            np.testing.assert_array_equal(out["values"][:rigid_values], values[:rigid_values].astype(np.float32))
            np.testing.assert_array_equal(out["delta"][: scalar_offsets[layout.rigid_body_row_count]], 7.0)


def test_variable_solve_frame_regularization(test, device):
    """The backend adds the frame regularization to the six frame diagonal entries of mixed rows."""
    regularization = 0.75
    for name, launcher in _variable_solvers(device):
        _model, layout, systems, values, rhs = _solve_fixture(device, 9)
        out = _run_solve(launcher, layout, values, rhs, device, frame_regularization=regularization)
        test.assertEqual(out["failures"], 0, name)
        scalar_offsets = layout.articulation_row_scalar_offsets.numpy()
        dims = layout.articulation_row_dims.numpy()
        for a, system in systems.items():
            start, end = _articulation_rows(layout, a)
            base = scalar_offsets[start]
            H = system["H"].copy()
            for row in range(start, end):
                for k in range(6):
                    H[scalar_offsets[row] - base + k, scalar_offsets[row] - base + k] += regularization
                np.testing.assert_allclose(
                    out["diag_backup"][scalar_offsets[row] : scalar_offsets[row] + dims[row]],
                    np.diag(H)[scalar_offsets[row] - base : scalar_offsets[row] - base + dims[row]],
                    rtol=1.0e-6,
                )
            expected = np.linalg.solve(H, system["rhs"])
            delta = out["delta"][scalar_offsets[start] : scalar_offsets[end]]
            test.assertLess(np.linalg.norm(delta - expected) / np.linalg.norm(expected), 1.0e-4, name)


def test_variable_solve_failure_takes_jacobi_step(test, device):
    """Indefinite or non-finite systems set the status, count once, and take a finite Jacobi step."""
    for name, launcher in _variable_solvers(device):
        _model, layout, systems, values, rhs = _solve_fixture(device, 13)
        value_offsets = layout.articulation_block_value_offsets.numpy()
        block_dims = layout.articulation_block_dims.numpy()
        scalar_offsets = layout.articulation_row_scalar_offsets.numpy()
        diag = layout.articulation_diag_slots.numpy()
        mixed = list(range(layout.rigid_articulation_count, layout.articulation_count))
        # Articulation 1: an indefinite modal diagonal block in its second row.
        start, _end = _articulation_rows(layout, mixed[0])
        row = start + 1
        n = int(block_dims[diag[row], 0])
        block = values[value_offsets[diag[row]] : value_offsets[diag[row] + 1]].reshape(n, n)
        block[6:, 6:] -= 1.0e3 * np.eye(n - 6)
        values[value_offsets[diag[row]] : value_offsets[diag[row] + 1]] = block.ravel()
        # Articulation 2: a NaN in an off-diagonal block.
        start, end = _articulation_rows(layout, mixed[1])
        row_offsets = layout.articulation_block_row_offsets.numpy()
        off_slot = next(s for r in range(start, end) for s in range(row_offsets[r], row_offsets[r + 1]) if s != diag[r])
        values[value_offsets[off_slot]] = np.nan
        # Articulation 3: a valid matrix with a non-finite right-hand side entry.
        start, end = _articulation_rows(layout, mixed[2])
        rhs[scalar_offsets[start] + 2] = np.inf

        out = _run_solve(launcher, layout, values, rhs, device)
        np.testing.assert_array_equal(out["status"], [VB_FACTOR_INDEFINITE, VB_FACTOR_INDEFINITE, VB_SOLVE_NONFINITE])
        test.assertEqual(out["failures"], 3, name)
        test.assertTrue(np.isfinite(out["delta"]).all())
        for a in mixed:
            start, end = _articulation_rows(layout, a)
            segment = slice(scalar_offsets[start], scalar_offsets[end])
            d, r = out["diag_backup"][segment], out["rhs"][segment]
            usable = (d > 0.0) & np.isfinite(d) & np.isfinite(r)
            expected = np.where(usable, r / np.where(usable, d, 1.0), 0.0).astype(np.float32)
            np.testing.assert_allclose(out["delta"][segment], expected, rtol=1.0e-6, atol=0.0)
            test.assertTrue((~usable).any() or a != mixed[0])

        # A healthy articulation next to a failing one is still solved exactly.
        _model, layout, systems, values, rhs = _solve_fixture(device, 13)
        start, _end = _articulation_rows(layout, mixed[0])
        row = start + 1
        block = values[value_offsets[diag[row]] : value_offsets[diag[row] + 1]].reshape(n, n)
        block[6:, 6:] -= 1.0e3 * np.eye(n - 6)
        values[value_offsets[diag[row]] : value_offsets[diag[row] + 1]] = block.ravel()
        out = _run_solve(launcher, layout, values, rhs, device)
        np.testing.assert_array_equal(out["status"], [VB_FACTOR_INDEFINITE, VB_FACTOR_OK, VB_FACTOR_OK])
        test.assertEqual(out["failures"], 1, name)
        for a in mixed[1:]:
            start, end = _articulation_rows(layout, a)
            expected = np.linalg.solve(systems[a]["H"], systems[a]["rhs"])
            delta = out["delta"][scalar_offsets[start] : scalar_offsets[end]]
            test.assertLess(np.linalg.norm(delta - expected) / np.linalg.norm(expected), 1.0e-4, name)


def _launch_legacy_serial_solve(layout, values, rhs, delta, body_count, device):
    """Run the 6 x 6 serial kernel on preassembled values (``assemble=False``)."""
    body_q = wp.zeros(body_count, dtype=wp.transform, device=device)
    body_q_new = wp.zeros(body_count, dtype=wp.transform, device=device)
    body_vec = wp.zeros(body_count, dtype=wp.vec3, device=device)
    zeros_float = wp.zeros(body_count, dtype=float, device=device)
    wp.launch(
        legacy_kernels.solve_articulation_sparse_serial_scalar,
        dim=layout.articulation_count,
        inputs=[
            1.0,
            layout.articulation_joints,
            layout.articulation_joint_body_start,
            layout.articulation_block_row_offsets,
            layout.articulation_block_cols,
            layout.body_articulation_local,
            body_q,
            *([None] * 3),
            body_vec,
            *([None] * 19),
            JointCoordinateData(),
            *([None] * 20),
            0.0,
            0,
            legacy_kernels.SparseElasticJointData(),
            layout.articulation_body_offsets,
            layout.articulation_joint_offsets,
            layout.articulation_bodies,
            layout.articulation_diag_slots,
            None,
            zeros_float,
            *([None] * 6),
            1.0,
            False,
            False,
            None,
            None,
            values,
            rhs,
            delta,
            body_vec,
            body_vec,
            0,
            body_q_new,
        ],
        device=device,
    )


def test_all_six_mixed_matches_legacy_serial_kernel(test, device):
    """A mixed articulation whose blocks are all 6 x 6 matches the legacy serial kernel."""
    # Zero-mode elastic bodies make a mixed articulation with size-6 rows only.
    mode_counts = [None, None, 0, None, 0, None, 0]
    articulations = [[(-1, 0), (0, 1)], [(-1, 2), (2, 3), (3, 4), (4, 2), (3, 5), (5, 6)]]
    model = _synthetic_model(device, mode_counts, articulations)
    layout = build_rigid_articulation_sparse_layout(model, device, elastic_solve="coupled")
    test.assertEqual(layout.mixed_articulation_count, 1)
    np.testing.assert_array_equal(layout.articulation_row_dims.numpy(), 6)
    rng = np.random.default_rng(21)
    systems = _random_system(layout, model, mode_counts, rng)
    values, rhs = _scatter_system(layout, systems, rng)
    # Make the rigid prefix SPD too, since the legacy kernel factors every articulation.
    for slot in layout.articulation_diag_slots.numpy()[: layout.rigid_body_row_count]:
        values[36 * slot : 36 * slot + 36] = (10.0 * np.eye(6)).ravel()
    for slot in range(layout.rigid_block_count):
        if slot not in set(layout.articulation_diag_slots.numpy().tolist()):
            values[36 * slot : 36 * slot + 36] = 0.1
    scalar_offsets = layout.articulation_row_scalar_offsets.numpy()
    mixed = slice(scalar_offsets[layout.rigid_body_row_count], scalar_offsets[-1])
    mixed_values = slice(36 * layout.rigid_block_count, 36 * layout.block_count)

    for name, launcher in _variable_solvers(device):
        out = _run_solve(launcher, layout, values, rhs, device)
        test.assertEqual(out["failures"], 0, name)
        legacy_values = wp.array(values.astype(np.float32), dtype=float, device=device)
        legacy_rhs = wp.array(rhs.astype(np.float32), dtype=float, device=device)
        legacy_delta = wp.zeros(layout.articulation_scalar_count, dtype=float, device=device)
        _launch_legacy_serial_solve(
            layout,
            legacy_values.reshape((layout.block_count, 6, 6)).view(legacy_kernels.mat66f),
            legacy_rhs.reshape((layout.articulation_body_count, 6)).view(legacy_kernels.vec6f),
            legacy_delta.reshape((layout.articulation_body_count, 6)).view(legacy_kernels.vec6f),
            model.body_count,
            device,
        )
        expected = legacy_delta.numpy()[mixed]
        np.testing.assert_allclose(out["delta"][mixed], expected, rtol=0.0, atol=2.0e-6 * np.abs(expected).max())
        np.testing.assert_allclose(
            out["values"][mixed_values],
            legacy_values.numpy()[mixed_values],
            rtol=0.0,
            atol=2.0e-6 * np.abs(legacy_values.numpy()[mixed_values]).max(),
        )


# ---------------------------------------------------------------------------
# Cooperative backend
# ---------------------------------------------------------------------------


def _solver_outputs_equal(test, a, b, msg):
    for key in ("values", "delta", "diag_backup", "status", "failures"):
        np.testing.assert_array_equal(a[key], b[key], err_msg=f"{msg}: {key}")


def test_cooperative_matches_serial(test, device):
    """The cooperative and serial kernels give the same factor and update up to float32 rounding.

    Rows of size 6 take the cooperative lane path, which repeats the serial arithmetic, so an
    all-6 system agrees bitwise; wider rows take the 32 x 32 tile path.
    """
    serial = variable_kernels.launch_solve_articulation_variable_serial
    cooperative = variable_kernels.launch_solve_articulation_variable_cooperative
    for seed, decades in ((5, 0.0), (7, 2.0)):
        _model, layout, systems, values, rhs = _solve_fixture(device, seed, decades)
        a = _run_solve(serial, layout, values, rhs, device, frame_regularization=0.25)
        b = _run_solve(cooperative, layout, values, rhs, device, frame_regularization=0.25)
        np.testing.assert_array_equal(a["status"], b["status"])
        test.assertEqual(a["failures"], b["failures"])
        np.testing.assert_array_equal(a["diag_backup"], b["diag_backup"])
        scalar_offsets = layout.articulation_row_scalar_offsets.numpy()
        for art, system in systems.items():
            start, end = _articulation_rows(layout, art)
            segment = slice(scalar_offsets[start], scalar_offsets[end])
            scale = system["scale"]
            difference = np.linalg.norm(scale * (a["delta"][segment] - b["delta"][segment]))
            test.assertLess(difference / np.linalg.norm(scale * a["delta"][segment]), 1.0e-5, f"articulation {art}")
        # Untouched storage is identical.
        rigid_values = layout.articulation_block_value_offsets.numpy()[layout.rigid_block_count]
        np.testing.assert_array_equal(b["values"][:rigid_values], a["values"][:rigid_values])
        np.testing.assert_array_equal(b["delta"][: scalar_offsets[layout.rigid_body_row_count]], 7.0)

    mode_counts = [None, None, 0, None, 0, None, 0]
    articulations = [[(-1, 0), (0, 1)], [(-1, 2), (2, 3), (3, 4), (4, 2), (3, 5), (5, 6)]]
    model = _synthetic_model(device, mode_counts, articulations)
    layout = build_rigid_articulation_sparse_layout(model, device, elastic_solve="coupled")
    rng = np.random.default_rng(21)
    values, rhs = _scatter_system(layout, _random_system(layout, model, mode_counts, rng), rng)
    a = _run_solve(serial, layout, values, rhs, device, frame_regularization=0.25)
    b = _run_solve(cooperative, layout, values, rhs, device, frame_regularization=0.25)
    _solver_outputs_equal(test, a, b, "all-6 system")


def test_cooperative_failure_matches_serial(test, device):
    """Indefinite, NaN and non-finite-solve articulations get the serial status, count and Jacobi step."""
    _model, layout, _systems, values, rhs = _solve_fixture(device, 13)
    value_offsets = layout.articulation_block_value_offsets.numpy()
    block_dims = layout.articulation_block_dims.numpy()
    scalar_offsets = layout.articulation_row_scalar_offsets.numpy()
    diag = layout.articulation_diag_slots.numpy()
    row_offsets = layout.articulation_block_row_offsets.numpy()
    mixed = list(range(layout.rigid_articulation_count, layout.articulation_count))
    # Indefinite modal block in a wide (tile path) row; the failure is in a later panel.
    start, end = _articulation_rows(layout, mixed[0])
    row = max(range(start, end), key=lambda r: block_dims[diag[r], 0])
    n = int(block_dims[diag[row], 0])
    test.assertGreater(n, 64)
    block = values[value_offsets[diag[row]] : value_offsets[diag[row] + 1]].reshape(n, n)
    block[n - 3, n - 3] -= 1.0e4
    values[value_offsets[diag[row]] : value_offsets[diag[row] + 1]] = block.ravel()
    # A NaN in an off-diagonal block.
    start, end = _articulation_rows(layout, mixed[1])
    off_slot = next(s for r in range(start, end) for s in range(row_offsets[r], row_offsets[r + 1]) if s != diag[r])
    values[value_offsets[off_slot]] = np.nan
    # A valid matrix with a non-finite right-hand side entry.
    start, _end = _articulation_rows(layout, mixed[2])
    rhs[scalar_offsets[start] + 2] = np.inf

    serial = _run_solve(variable_kernels.launch_solve_articulation_variable_serial, layout, values, rhs, device)
    cooperative = _run_solve(
        variable_kernels.launch_solve_articulation_variable_cooperative, layout, values, rhs, device
    )
    np.testing.assert_array_equal(
        cooperative["status"], [VB_FACTOR_INDEFINITE, VB_FACTOR_INDEFINITE, VB_SOLVE_NONFINITE]
    )
    np.testing.assert_array_equal(cooperative["status"], serial["status"])
    test.assertEqual(cooperative["failures"], 3)
    test.assertTrue(np.isfinite(cooperative["delta"]).all())
    # The Jacobi step depends only on the saved diagonal and the right-hand side.
    np.testing.assert_array_equal(cooperative["diag_backup"], serial["diag_backup"])
    np.testing.assert_array_equal(cooperative["delta"], serial["delta"])


def test_cooperative_shared_memory_under_budget(test, device):
    """The cooperative CTA needs at most the documented budget and fits the device."""
    required = variable_kernels.cooperative_shared_memory_bytes(device)
    test.assertGreater(required, 0)
    test.assertLessEqual(required, variable_kernels.VB_COOPERATIVE_SHARED_MEMORY_BUDGET)
    test.assertLessEqual(required, wp.get_device(device).max_shared_memory_per_block)


def test_cooperative_graph_capture_and_run_to_run(test, device):
    """Graph replays and repeated launches of the cooperative solve are bitwise identical."""
    _model, layout, _systems, values_np, rhs_np = _solve_fixture(device, 11)
    pristine = wp.array(values_np.astype(np.float32), dtype=float, device=device)
    values = wp.empty_like(pristine)
    rhs = wp.array(rhs_np.astype(np.float32), dtype=float, device=device)
    delta = wp.zeros(layout.articulation_scalar_count, dtype=float, device=device)
    diag_backup = wp.zeros_like(delta)
    status = wp.full(layout.mixed_articulation_count, -1, dtype=wp.int32, device=device)
    failures = wp.zeros(1, dtype=wp.int32, device=device)

    def solve():
        wp.copy(values, pristine)
        failures.zero_()
        variable_kernels.launch_solve_articulation_variable(
            layout, 0.5, rhs, values, diag_backup, status, failures, delta, device=device
        )

    def snapshot():
        return [array.numpy().copy() for array in (values, delta, diag_backup, status, failures)]

    solve()
    reference = snapshot()
    test.assertEqual(int(reference[4][0]), 0)
    solve()
    for got, expected in zip(snapshot(), reference, strict=True):
        np.testing.assert_array_equal(got, expected)
    with wp.ScopedCapture(device=device) as capture:
        solve()
    for _ in range(3):
        # Clear the outputs so each replay has to rewrite them (the rigid prefix stays zero).
        delta.zero_()
        diag_backup.zero_()
        status.fill_(-1)
        wp.capture_launch(capture.graph)
        for got, expected in zip(snapshot(), reference, strict=True):
            np.testing.assert_array_equal(got, expected)


def _batch_fixture(device, count: int, seed: int):
    """``count`` mixed articulations of 2 to 5 bodies with heterogeneous modal counts (chains, loops)."""
    rng = np.random.default_rng(seed)
    choices = [None, 0, 1, 8, 26, 33, 70]
    mode_counts, articulations = [], []
    for _ in range(count):
        size = int(rng.integers(2, 6))
        first = len(mode_counts)
        modes = [choices[int(i)] for i in rng.integers(0, len(choices), size=size)]
        if all(m is None for m in modes):
            modes[0] = 8
        mode_counts.extend(modes)
        joints = [(-1, first)] + [(first + i, first + i + 1) for i in range(size - 1)]
        if size > 2 and rng.random() < 0.5:
            joints.append((first + size - 1, first))
        articulations.append(joints)
    model = _synthetic_model(device, mode_counts, articulations)
    layout = build_rigid_articulation_sparse_layout(model, device, elastic_solve="coupled")
    systems = _random_system(layout, model, mode_counts, rng)
    values, rhs = _scatter_system(layout, systems, rng)
    return layout, systems, values, rhs


def test_cooperative_batch_of_heterogeneous_articulations(test, device):
    """One launch solves 256 heterogeneous mixed articulations, each matching numpy."""
    layout, systems, values, rhs = _batch_fixture(device, 256, 31)
    test.assertEqual(layout.mixed_articulation_count, 256)
    dims = set(layout.articulation_row_dims.numpy().tolist())
    test.assertTrue({6, 7, 14, 32, 39, 76} <= dims, dims)
    scalar_offsets = layout.articulation_row_scalar_offsets.numpy()
    for name, launcher in _variable_solvers(device):
        out = _run_solve(launcher, layout, values, rhs, device)
        test.assertEqual(out["failures"], 0, name)
        np.testing.assert_array_equal(out["status"], VB_FACTOR_OK)
        worst = 0.0
        for a, system in systems.items():
            start, end = _articulation_rows(layout, a)
            delta = out["delta"][scalar_offsets[start] : scalar_offsets[end]].astype(np.float64)
            expected = np.linalg.solve(system["H"], system["rhs"])
            worst = max(worst, np.linalg.norm(delta - expected) / np.linalg.norm(expected))
        test.assertLess(worst, 1.0e-4, name)


def test_dispatch_selects_backend_by_device(test, device):
    """``launch_solve_articulation_variable`` runs the cooperative kernel on CUDA and the serial one on CPU."""
    _model, layout, _systems, values, rhs = _solve_fixture(device, 5)
    expected_launcher = (
        variable_kernels.launch_solve_articulation_variable_cooperative
        if wp.get_device(device).is_cuda
        else variable_kernels.launch_solve_articulation_variable_serial
    )
    dispatched = _run_solve(launch_solve_articulation_variable, layout, values, rhs, device)
    expected = _run_solve(expected_launcher, layout, values, rhs, device)
    _solver_outputs_equal(test, dispatched, expected, "dispatch")


# ---------------------------------------------------------------------------
# Update application
# ---------------------------------------------------------------------------


def _np_quat_mul(a, b):
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


def _np_quat_rotate(q, v):
    qv = np.asarray(q[:3], dtype=np.float64)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + float(q[3]) * v)


def _launch_apply(layout, model, state, delta_np, joint_q_prev_np, relaxation, dt, use_displacement, device):
    body_q_new = wp.array(np.full((model.body_count, 7), 3.0, dtype=np.float32), dtype=wp.transform, device=device)
    joint_q = wp.clone(state.joint_q)
    joint_qd = wp.array(np.full(model.joint_dof_count, 5.0, dtype=np.float32), dtype=float, device=device)
    body_disp = wp.array(
        np.random.default_rng(1).normal(size=(model.body_count, 3)).astype(np.float32) * 1e-3,
        dtype=wp.vec3,
        device=device,
    )
    body_disp_ang = wp.zeros(model.body_count, dtype=wp.vec3, device=device)
    inertia_q = wp.clone(state.body_q)
    wp.launch(
        apply_articulation_variable_delta,
        dim=layout.articulation_body_count - layout.rigid_body_row_count,
        inputs=[
            dt,
            relaxation,
            layout.rigid_body_row_count,
            layout.articulation_bodies,
            layout.articulation_row_dims,
            layout.articulation_row_scalar_offsets,
            wp.array(delta_np.astype(np.float32), dtype=float, device=device),
            state.body_q,
            model.body_inv_mass,
            model.body_com,
            inertia_q,
            use_displacement,
            model.body_elastic_index,
            model.elastic_joint,
            model.joint_q_start,
            model.joint_qd_start,
            wp.array(joint_q_prev_np.astype(np.float32), dtype=float, device=device),
        ],
        outputs=[body_disp, body_disp_ang, body_q_new, joint_q, joint_qd],
        device=device,
    )
    return body_q_new, joint_q, joint_qd, body_disp, body_disp_ang, inertia_q


def test_apply_variable_delta(test, device):
    """Frame (displacement form via the 6 x 6 helper) and modal updates, relaxation applied once."""
    model, ids = _mixed_model(device)
    layout = build_rigid_articulation_sparse_layout(model, device, elastic_solve="coupled")
    state = model.state()
    rng = np.random.default_rng(4)
    delta = rng.normal(size=layout.articulation_scalar_count) * 0.05
    joint_q0 = state.joint_q.numpy()
    joint_q_prev = joint_q0 + rng.normal(size=joint_q0.shape) * 1.0e-3
    dt = 1.0 / 240.0
    bodies = layout.articulation_bodies.numpy()
    scalar_offsets = layout.articulation_row_scalar_offsets.numpy()
    dims = layout.articulation_row_dims.numpy()
    body_q0 = state.body_q.numpy().astype(np.float64)
    com = model.body_com.numpy().astype(np.float64)
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    owner = model.elastic_joint.numpy()[model.body_elastic_index.numpy()[ids["link"]]]
    untouched_bodies = [b for b in range(model.body_count) if b not in bodies[layout.rigid_body_row_count :]]

    for relaxation in (0.65, 1.0):
        body_q_new, joint_q, joint_qd, _disp, _ang, _q_star = _launch_apply(
            layout, model, state, delta, joint_q_prev, relaxation, dt, 0, device
        )
        body_q_new, joint_q, joint_qd = body_q_new.numpy(), joint_q.numpy(), joint_qd.numpy()
        for row in range(layout.rigid_body_row_count, layout.articulation_body_count):
            body = int(bodies[row])
            d = delta[scalar_offsets[row] : scalar_offsets[row] + dims[row]].astype(np.float32).astype(np.float64)
            pos, rot = body_q0[body, :3], body_q0[body, 3:]
            w = relaxation * d[3:6]
            angle = np.linalg.norm(w)
            dq = np.concatenate([np.sin(0.5 * angle) * w / angle, [np.cos(0.5 * angle)]])
            rot_new = _np_quat_mul(dq, rot)
            rot_new /= np.linalg.norm(rot_new)
            com_new = pos + _np_quat_rotate(rot, com[body]) + relaxation * d[:3]
            pos_new = com_new - _np_quat_rotate(rot_new, com[body])
            np.testing.assert_allclose(body_q_new[body, :3], pos_new, atol=2.0e-6)
            np.testing.assert_allclose(np.abs(np.dot(body_q_new[body, 3:], rot_new)), 1.0, atol=1.0e-6)
            if body == ids["link"]:
                modes = slice(q_start[owner] + 7, q_start[owner] + 7 + ids["link_modes"])
                q_new = joint_q0[modes] + relaxation * d[6:]
                np.testing.assert_allclose(joint_q[modes], q_new, atol=1.0e-6)
                dofs = slice(qd_start[owner] + 6, qd_start[owner] + 6 + ids["link_modes"])
                np.testing.assert_allclose(joint_qd[dofs], (q_new - joint_q_prev[modes]) / dt, rtol=1.0e-4, atol=1.0e-3)
        # Rigid-prefix and standalone bodies and every other coordinate are untouched.
        np.testing.assert_array_equal(body_q_new[untouched_bodies], 3.0)
        link_modes = set(range(q_start[owner] + 7, q_start[owner] + 7 + ids["link_modes"]))
        others = [i for i in range(len(joint_q0)) if i not in link_modes]
        np.testing.assert_array_equal(joint_q[others], joint_q0[others])
        link_dofs = set(range(qd_start[owner] + 6, qd_start[owner] + 6 + ids["link_modes"]))
        np.testing.assert_array_equal(joint_qd[[i for i in range(len(joint_qd)) if i not in link_dofs]], 5.0)

    # The displacement-form frame update equals the 6 x 6 kernel's, bit for bit.
    relaxation = 0.65
    body_q_new, _joint_q, _joint_qd, disp, disp_ang, inertia_q = _launch_apply(
        layout, model, state, delta, joint_q_prev, relaxation, dt, 1, device
    )
    frames = np.zeros((layout.articulation_body_count, 6), dtype=np.float32)
    for row in range(layout.articulation_body_count):
        frames[row] = delta[scalar_offsets[row] : scalar_offsets[row] + 6]
    legacy_q_new = wp.array(np.full((model.body_count, 7), 3.0, dtype=np.float32), dtype=wp.transform, device=device)
    legacy_disp = wp.array(
        np.random.default_rng(1).normal(size=(model.body_count, 3)).astype(np.float32) * 1e-3,
        dtype=wp.vec3,
        device=device,
    )
    legacy_disp_ang = wp.zeros(model.body_count, dtype=wp.vec3, device=device)
    wp.launch(
        legacy_kernels.apply_articulation_sparse_delta_scalar,
        dim=layout.articulation_body_count,
        inputs=[
            layout.articulation_bodies,
            state.body_q,
            model.body_inv_mass,
            model.body_com,
            relaxation,
            wp.array(frames.ravel(), dtype=float, device=device),
            inertia_q,
            legacy_disp,
            legacy_disp_ang,
            1,
        ],
        outputs=[legacy_q_new],
        device=device,
    )
    mixed_bodies = bodies[layout.rigid_body_row_count :]
    np.testing.assert_array_equal(body_q_new.numpy()[mixed_bodies], legacy_q_new.numpy()[mixed_bodies])
    np.testing.assert_array_equal(disp.numpy()[mixed_bodies], legacy_disp.numpy()[mixed_bodies])
    np.testing.assert_array_equal(disp_ang.numpy()[mixed_bodies], legacy_disp_ang.numpy()[mixed_bodies])


def test_variable_kernels_honor_deterministic_option(test, device):
    """SolverVBD forwards its deterministic mode to the variable-block kernel module."""
    original = wp.get_module_options(module=variable_kernels)
    try:
        wp.set_module_options({"deterministic": wp.DeterministicMode.NOT_GUARANTEED}, module=variable_kernels)
        newton.solvers.SolverVBD(
            _rigid_model(device),
            rigid_articulation_solve="block_sparse_joints",
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        options = wp.get_module_options(module=variable_kernels)
        test.assertEqual(options["deterministic"], wp.DeterministicMode.RUN_TO_RUN)
    finally:
        wp.set_module_options(original, module=variable_kernels)


class TestVBDSparseVariableBlocks(unittest.TestCase):
    pass


cpu_devices = [d for d in devices if not wp.get_device(d).is_cuda]
cuda_devices = [d for d in devices if wp.get_device(d).is_cuda]

for _name, _func in [
    ("test_all_rigid_layout_identical_under_both_backends", test_all_rigid_layout_identical_under_both_backends),
    ("test_unknown_elastic_solve_raises", test_unknown_elastic_solve_raises),
    ("test_mixed_layout_partition_and_dims", test_mixed_layout_partition_and_dims),
    ("test_synthetic_mixed_layout_consistency", test_synthetic_mixed_layout_consistency),
    ("test_symbolic_pattern_matches_independent_elimination", test_symbolic_pattern_matches_independent_elimination),
    ("test_coupled_layout_adds_no_elastic_clique", test_coupled_layout_adds_no_elastic_clique),
    ("test_weighted_order_reduces_to_minimum_degree", test_weighted_order_reduces_to_minimum_degree),
    ("test_layout_rejects_int32_overflow", test_layout_rejects_int32_overflow),
    ("test_reference_layout_solve_matches_dense", test_reference_layout_solve_matches_dense),
]:
    # Host-side analysis: one device suffices.
    add_function_test(TestVBDSparseVariableBlocks, _name, _func, devices=cpu_devices)

for _name, _func in [
    ("test_variable_solve_random_spd", test_variable_solve_random_spd),
    ("test_variable_solve_frame_regularization", test_variable_solve_frame_regularization),
    ("test_variable_solve_failure_takes_jacobi_step", test_variable_solve_failure_takes_jacobi_step),
    ("test_all_six_mixed_matches_legacy_serial_kernel", test_all_six_mixed_matches_legacy_serial_kernel),
    ("test_apply_variable_delta", test_apply_variable_delta),
    ("test_variable_kernels_honor_deterministic_option", test_variable_kernels_honor_deterministic_option),
    ("test_cooperative_matches_serial", test_cooperative_matches_serial),
    ("test_cooperative_failure_matches_serial", test_cooperative_failure_matches_serial),
    ("test_cooperative_batch_of_heterogeneous_articulations", test_cooperative_batch_of_heterogeneous_articulations),
    ("test_dispatch_selects_backend_by_device", test_dispatch_selects_backend_by_device),
]:
    add_function_test(TestVBDSparseVariableBlocks, _name, _func, devices=devices)

for _name, _func in [
    ("test_cooperative_shared_memory_under_budget", test_cooperative_shared_memory_under_budget),
    ("test_cooperative_graph_capture_and_run_to_run", test_cooperative_graph_capture_and_run_to_run),
]:
    # Shared memory and graph capture are CUDA concepts.
    add_function_test(TestVBDSparseVariableBlocks, _name, _func, devices=cuda_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
