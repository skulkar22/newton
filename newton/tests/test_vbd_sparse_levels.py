# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent numerical checks for level-scheduled block Cholesky."""

import itertools
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.vbd.rigid_sparse_articulation import build_rigid_articulation_sparse_layout
from newton._src.solvers.vbd.rigid_sparse_levels_kernels import solve_articulation_sparse_levels
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _layout(groups, device, use_levels=True):
    parents, children, starts, ends = [], [], [], []
    body_count = 0
    for count, edges in groups:
        starts.append(len(parents))
        parents.extend([-1, *(body_count + a for a, _ in edges)])
        children.extend([body_count, *(body_count + b for _, b in edges)])
        ends.append(len(parents))
        body_count += count

    def array(values):
        return wp.array(values, dtype=wp.int32, device=device)

    model = SimpleNamespace(
        body_count=body_count,
        joint_count=len(parents),
        articulation_count=len(groups),
        joint_parent=array(parents),
        joint_child=array(children),
        articulation_start=array(starts),
        articulation_end=array(ends),
        body_color_groups=[array(range(body_count))],
    )
    return build_rigid_articulation_sparse_layout(model, device, use_levels=use_levels)


def _cases():
    def chain(n):
        return (n, [(i, i + 1) for i in range(n - 1)])

    return {
        "single": [(1, [])],
        "chain": [chain(65)],
        "ring": [(17, [(i, (i + 1) % 17) for i in range(17)])],
        "ladder": [(24, [(i, i + 1) for i in range(23) if i != 11] + [(i, i + 12) for i in range(12)])],
        "dense": [(9, [(i, j) for i in range(9) for j in range(i)])],
        "mixed": [chain(17), (1, []), chain(6), chain(17)],
    }


def test_levels_symbolic(test, device):
    """Cover fill edges, independent pivots, and offsets across repeated articulations."""
    for name, groups in _cases().items():
        with test.subTest(case=name):
            layout = _layout(groups, device)
            levels = layout.levels
            offsets = levels.level_body_offsets.numpy()
            cols = layout.articulation_block_cols.numpy()
            rows = layout.articulation_block_row_offsets.numpy()
            bodies = layout.articulation_body_offsets.numpy()
            level_starts = levels.articulation_level_offsets.numpy()
            for art in range(len(groups)):
                test.assertEqual(offsets[level_starts[art]], bodies[art])
                test.assertEqual(offsets[level_starts[art + 1]], bodies[art + 1])
                for begin, end in zip(
                    offsets[level_starts[art] : level_starts[art + 1]],
                    offsets[level_starts[art] + 1 : level_starts[art + 1] + 1],
                    strict=True,
                ):
                    for row in range(begin, end):
                        predecessors = bodies[art] + cols[rows[row] : rows[row + 1] - 1]
                        test.assertTrue(np.all(predecessors < begin))
            dst = levels.level_update_dst_slots.numpy()
            update_offsets = levels.level_update_offsets.numpy()
            for begin, end in itertools.pairwise(update_offsets):
                test.assertEqual(len(set(dst[begin:end])), end - begin)
            if name == "chain":
                test.assertLessEqual(len(offsets) - 1, 8)
            test.assertIsNone(_layout(groups, device, use_levels=False).levels)


def _random_system(groups, layout, device):
    """Random SPD block system in the layout's block storage: dense FP64 matrix, rhs, values, rhs array."""
    count = layout.articulation_body_count
    rng = np.random.default_rng(103)
    matrix = np.eye(6 * count)
    base = 0
    for size, edges in groups:
        for a, b in edges:
            indices = np.r_[6 * (base + a) + np.arange(6), 6 * (base + b) + np.arange(6)]
            block = rng.normal(size=(8, 12))
            matrix[np.ix_(indices, indices)] += block.T @ block
        base += size
    order = (6 * layout.articulation_bodies.numpy()[:, None] + np.arange(6)).reshape(-1)
    matrix = matrix[np.ix_(order, order)].astype(np.float32).astype(np.float64)
    rhs_host = rng.normal(size=count * 6).astype(np.float32)
    rows = layout.articulation_block_row_offsets.numpy()
    cols = layout.articulation_block_cols.numpy()
    offsets = layout.articulation_body_offsets.numpy()
    blocks = np.zeros((layout.block_count, 6, 6), dtype=np.float32)
    for row in range(count):
        base = offsets[np.searchsorted(offsets, row, side="right") - 1]
        for slot in range(rows[row], rows[row + 1]):
            col = base + cols[slot]
            blocks[slot] = matrix[6 * row : 6 * row + 6, 6 * col : 6 * col + 6]
    original = wp.array(blocks.reshape(-1), dtype=float, device=device)
    rhs = wp.array(rhs_host, dtype=float, device=device)
    return matrix, rhs_host, original, rhs


def _levels_inputs(layout):
    """Static inputs of ``solve_articulation_sparse_levels`` (everything before values and rhs)."""
    inputs = [
        getattr(layout, "articulation_" + key)
        for key in (
            "body_offsets",
            "block_row_offsets",
            "block_cols",
            "block_col_offsets",
            "block_col_rows",
            "block_col_slots",
            "diag_slots",
        )
    ]
    inputs += [layout.levels.articulation_level_offsets]
    inputs += [
        getattr(layout.levels, "level_" + key)
        for key in (
            "body_offsets",
            "column_offsets",
            "column_slots",
            "column_diag_slots",
            "update_offsets",
            "update_dst_slots",
            "update_term_offsets",
            "update_left_slots",
            "update_right_slots",
        )
    ]
    return inputs


def _solve_levels(layout, original, rhs, device, block_dim=128):
    """One launch of the cooperative levels kernel; returns the factor and the solution."""
    values = wp.clone(original)
    solution = wp.zeros_like(rhs)
    wp.launch(
        solve_articulation_sparse_levels,
        block_dim * layout.articulation_count,
        inputs=[*_levels_inputs(layout), values, rhs],
        outputs=[solution],
        device=device,
        block_dim=block_dim,
    )
    return values, solution


def test_levels_dense_oracle(test, device):
    """Compare FP32 CUDA solves with a dense FP64 oracle, including captured replays."""
    for name, groups in _cases().items():
        with test.subTest(case=name):
            layout = _layout(groups, device)
            matrix, rhs_host, original, rhs = _random_system(groups, layout, device)
            expected = np.linalg.solve(matrix, rhs_host.astype(np.float64))
            values = wp.empty_like(original)
            solution = wp.empty_like(rhs)
            inputs = [*_levels_inputs(layout), values, rhs]

            def solve(values=values, original=original, layout=layout, inputs=inputs, solution=solution):
                wp.copy(values, original)
                wp.launch(
                    solve_articulation_sparse_levels,
                    128 * layout.articulation_count,
                    inputs=inputs,
                    outputs=[solution],
                    device=device,
                    block_dim=128,
                )

            solve()
            actual = solution.numpy()
            np.testing.assert_allclose(actual, expected, rtol=2.0e-4, atol=2.0e-6)
            backward_error = np.linalg.norm(matrix @ actual - rhs_host, np.inf) / (
                np.linalg.norm(matrix, np.inf) * np.linalg.norm(actual, np.inf) + np.linalg.norm(rhs_host, np.inf)
            )
            test.assertLess(backward_error, 1.0e-6)
            with wp.ScopedCapture(device=device) as capture:
                solve()
            for _ in range(3):
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(solution.numpy(), actual)


def test_levels_block_dim_matches_default(test, device):
    """Any CUDA block size gives the factor and solution of the 128-thread launch bit for bit.

    Each thread owns whole tasks and no phase reduces across threads, so the block size
    only changes how the tasks of a level are strided over the threads.
    """
    for name, groups in _cases().items():
        layout = _layout(groups, device)
        _matrix, _rhs_host, original, rhs = _random_system(groups, layout, device)
        ref_values, ref_solution = _solve_levels(layout, original, rhs, device)
        for block_dim in (32, 256, 512):
            with test.subTest(case=name, block_dim=block_dim):
                values, solution = _solve_levels(layout, original, rhs, device, block_dim)
                np.testing.assert_array_equal(solution.numpy(), ref_solution.numpy())
                np.testing.assert_array_equal(values.numpy(), ref_values.numpy())


class TestVBDSparseLevels(unittest.TestCase):
    pass


add_function_test(TestVBDSparseLevels, "test_levels_symbolic", test_levels_symbolic, devices=get_test_devices())
add_function_test(
    TestVBDSparseLevels,
    "test_levels_dense_oracle",
    test_levels_dense_oracle,
    devices=[device for device in get_test_devices() if device.is_cuda],
)
add_function_test(
    TestVBDSparseLevels,
    "test_levels_block_dim_matches_default",
    test_levels_block_dim_matches_default,
    devices=[device for device in get_test_devices() if device.is_cuda],
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
