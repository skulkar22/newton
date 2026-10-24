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
from newton._src.solvers.vbd.rigid_sparse_articulation import (
    _minimum_degree_order,
    _weighted_minimum_degree_order,
    build_rigid_articulation_sparse_layout,
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


class TestVBDSparseVariableBlocks(unittest.TestCase):
    pass


cpu_devices = [d for d in devices if not wp.get_device(d).is_cuda]

for _name, _func in [
    ("test_all_rigid_layout_identical_under_both_backends", test_all_rigid_layout_identical_under_both_backends),
    ("test_unknown_elastic_solve_raises", test_unknown_elastic_solve_raises),
    ("test_mixed_layout_partition_and_dims", test_mixed_layout_partition_and_dims),
    ("test_synthetic_mixed_layout_consistency", test_synthetic_mixed_layout_consistency),
    ("test_symbolic_pattern_matches_independent_elimination", test_symbolic_pattern_matches_independent_elimination),
    ("test_coupled_layout_adds_no_elastic_clique", test_coupled_layout_adds_no_elastic_clique),
    ("test_weighted_order_reduces_to_minimum_degree", test_weighted_order_reduces_to_minimum_degree),
    ("test_layout_rejects_int32_overflow", test_layout_rejects_int32_overflow),
]:
    # Host-side analysis: one device suffices.
    add_function_test(TestVBDSparseVariableBlocks, _name, _func, devices=cpu_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
