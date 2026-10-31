# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent numerical checks for level-scheduled block Cholesky."""

import itertools
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.vbd.rigid_sparse_articulation import build_rigid_articulation_sparse_layout
from newton._src.solvers.vbd.rigid_sparse_level_launch_kernels import (
    build_level_launch_plan,
    launch_level_scheduled_solve,
)
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
        inputs=[*_levels_inputs(layout), values, rhs, wp.zeros(1, dtype=wp.int32, device=device)],
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
            inputs = [*_levels_inputs(layout), values, rhs, wp.zeros(1, dtype=wp.int32, device=device)]

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


def _star(chains: int, length: int):
    """Chains that start and end at one hub body (loop closures): the hub is a dense separator.

    The hub row collects a fill block from every chain and the hub diagonal receives one Schur
    term per adjacent pivot of a level, so the per-level plan splits both.
    """
    edges = []
    for chain_index in range(chains):
        first = 1 + chain_index * length
        edges.append((0, first))
        edges.extend((first + i, first + i + 1) for i in range(length - 1))
        edges.append((first + length - 1, 0))
    return (1 + chains * length, edges)


def _launch_cases():
    def chain(n):
        return (n, [(i, i + 1) for i in range(n - 1)])

    cases = _cases()
    # Levels with more pivot rows than the default tail threshold, in several articulations.
    cases["long"] = [chain(300), chain(40), (17, [(i, (i + 1) % 17) for i in range(17)]), chain(257)]
    cases["star"] = [_star(12, 40), chain(9)]
    return cases


# (split_entries, split_terms) of the per-level plan: defaults, no splitting, split almost everything.
_SPLITS = {
    "default": {},
    "nosplit": {"split_entries": 0, "split_terms": 0},
    "split2": {"split_entries": 2, "split_terms": 1},
}


def _plan_is_unsplit(plan) -> bool:
    return (
        plan.update.long.shape[0] + plan.forward.long.shape[0] + plan.backward.long.shape[0] == 0
        and plan.tail_split_entries == 0
    )


def _assert_launch_matches(test, plan, values, solution, ref_values, ref_solution, label):
    """Bitwise where no task is split (same operations, same order), 1e-5 relative otherwise."""
    if _plan_is_unsplit(plan):
        np.testing.assert_array_equal(solution.numpy(), ref_solution.numpy(), err_msg=label)
        np.testing.assert_array_equal(values.numpy(), ref_values.numpy(), err_msg=label)
    else:
        np.testing.assert_allclose(solution.numpy(), ref_solution.numpy(), rtol=1.0e-5, atol=1.0e-7, err_msg=label)
        np.testing.assert_allclose(values.numpy(), ref_values.numpy(), rtol=1.0e-5, atol=1.0e-7, err_msg=label)


def test_level_launch_plan_covers_tasks(test, device):
    """Every row, column entry and update of the leading levels appears once, at its level; chunks tile long tasks."""
    for name, groups in _launch_cases().items():
        layout = _layout(groups, device)
        levels = layout.levels
        level_starts = levels.articulation_level_offsets.numpy()
        body_offsets = levels.level_body_offsets.numpy()
        column_offsets = levels.level_column_offsets.numpy()
        update_offsets = levels.level_update_offsets.numpy()
        term_offsets = levels.level_update_term_offsets.numpy()
        row_offsets = layout.articulation_block_row_offsets.numpy()
        col_offsets = layout.articulation_block_col_offsets.numpy()
        articulation_offsets = layout.articulation_body_offsets.numpy()
        for tail_rows in (0, 4, 16, 2**30):
            with test.subTest(case=name, tail_rows=tail_rows):
                plan = build_level_launch_plan(layout, device, tail_rows=tail_rows)
                test.assertEqual(plan.level_count, int(np.max(np.diff(level_starts))))
                test.assertEqual(sum(plan.rows_per_level), layout.articulation_body_count)
                if tail_rows == 0:
                    test.assertFalse(plan.has_tail)
                tail = [rows < tail_rows for rows in plan.rows_per_level]
                test.assertEqual(plan.leading_level_count, tail.index(True) if True in tail else plan.level_count)
                rows = plan.rows.numpy()
                expected_rows, expected_columns, expected_updates = [], [], []
                for local in range(plan.leading_level_count):
                    for a in range(layout.rigid_articulation_count):
                        g = level_starts[a] + local
                        if g < level_starts[a + 1]:
                            expected_rows.extend(range(body_offsets[g], body_offsets[g + 1]))
                            expected_columns.extend(range(column_offsets[g], column_offsets[g + 1]))
                            expected_updates.extend(range(update_offsets[g], update_offsets[g + 1]))
                    level_rows = rows[plan.row_offsets[local] : plan.row_offsets[local + 1]]
                    test.assertEqual(len(set(level_rows.tolist())), len(level_rows))
                np.testing.assert_array_equal(rows, np.asarray(expected_rows, dtype=np.int32))
                np.testing.assert_array_equal(plan.columns.numpy(), np.asarray(expected_columns, dtype=np.int32))
                for split_name, split in _SPLITS.items():
                    split_plan = build_level_launch_plan(layout, device, tail_rows=tail_rows, **split)
                    label = f"{name} tail_rows={tail_rows} {split_name}"
                    _check_phase(test, split_plan.update, expected_updates, split_plan, term_offsets, None, label)
                    _check_phase(
                        test, split_plan.forward, expected_rows, split_plan, row_offsets, articulation_offsets, label
                    )
                    _check_phase(
                        test, split_plan.backward, expected_rows, split_plan, col_offsets, articulation_offsets, label
                    )
                    if name == "star" and split_name == "default" and tail_rows == 0:
                        # The hub row and the hub diagonal take the chunked paths at the default thresholds.
                        test.assertGreater(split_plan.forward.long.shape[0], 0)
                        test.assertGreater(split_plan.update.long.shape[0], 0)
                    if split_name == "nosplit":
                        test.assertTrue(_plan_is_unsplit(split_plan))


def _check_phase(test, phase, expected_tasks, plan, entry_offsets, articulation_offsets, label):
    """Short and long tasks partition the phase's tasks per level; chunks tile each long task's entries."""
    short, long = phase.short.numpy(), phase.long.numpy()
    long_chunks = phase.long_chunk_offsets.numpy()
    chunk_begin, chunk_end, chunk_owner = phase.chunk_begin.numpy(), phase.chunk_end.numpy(), phase.chunk_owner.numpy()
    collected = []
    for level in range(plan.leading_level_count):
        level_short = short[phase.short_offsets[level] : phase.short_offsets[level + 1]].tolist()
        level_long = long[phase.long_offsets[level] : phase.long_offsets[level + 1]].tolist()
        collected.extend(level_short + level_long)
        for t in range(phase.long_offsets[level], phase.long_offsets[level + 1]):
            chunks = range(long_chunks[t], long_chunks[t + 1])
            test.assertGreater(len(chunks), 1, label)
            test.assertTrue(
                all(phase.chunk_offsets[level] <= c < phase.chunk_offsets[level + 1] for c in chunks), label
            )
            bounds = [entry_offsets[long[t]]]
            for c in chunks:
                test.assertEqual(chunk_begin[c], bounds[-1], label)
                test.assertLess(chunk_begin[c], chunk_end[c], label)
                test.assertEqual(chunk_owner[c], long[t], label)
                bounds.append(chunk_end[c])
            test.assertEqual(bounds[-1], entry_offsets[long[t] + 1], label)
    test.assertEqual(sorted(collected), sorted(expected_tasks), label)
    if articulation_offsets is not None:
        for row, start in zip(short, phase.short_aux.numpy(), strict=True):
            test.assertEqual(start, articulation_offsets[np.searchsorted(articulation_offsets, row, side="right") - 1])
        for row, start in zip(chunk_owner, phase.chunk_aux.numpy(), strict=True):
            test.assertEqual(start, articulation_offsets[np.searchsorted(articulation_offsets, row, side="right") - 1])


def test_level_launch_serial_matches_levels(test, device):
    """CPU: the per-level launches with one-thread blocks reproduce the one-thread levels kernel.

    ``SolverVBD`` keeps the serial solve on CPU; this checks the task decomposition itself, since
    every Warp CPU thread of a block runs to completion in order.
    """
    for name, groups in _launch_cases().items():
        layout = _layout(groups, device)
        matrix, rhs_host, original, rhs = _random_system(groups, layout, device)
        ref_values, ref_solution = _solve_levels(layout, original, rhs, device, block_dim=1)
        expected = np.linalg.solve(matrix, rhs_host.astype(np.float64))
        np.testing.assert_allclose(ref_solution.numpy(), expected, rtol=2.0e-4, atol=2.0e-6)
        for tail_rows, split_name in itertools.product((0, 4, 16), _SPLITS):
            with test.subTest(case=name, tail_rows=tail_rows, split=split_name):
                plan = build_level_launch_plan(
                    layout, device, tail_rows=tail_rows, tail_block_dim=1, **_SPLITS[split_name]
                )
                values = wp.clone(original)
                solution = wp.zeros_like(rhs)
                launch_level_scheduled_solve(
                    plan, layout, values, rhs, solution, device, wp.zeros(1, dtype=wp.int32, device=device), block_dim=1
                )
                label = f"serial {name} tail_rows={tail_rows} split={split_name}"
                _assert_launch_matches(test, plan, values, solution, ref_values, ref_solution, label)


def test_level_launch_matches_levels(test, device):
    """CUDA: the per-level launches match the 128-thread cooperative kernel, also under graph replay."""
    for name, groups in _launch_cases().items():
        layout = _layout(groups, device)
        matrix, rhs_host, original, rhs = _random_system(groups, layout, device)
        ref_values, ref_solution = _solve_levels(layout, original, rhs, device)
        expected = np.linalg.solve(matrix, rhs_host.astype(np.float64))
        for tail_rows, split_name in itertools.product((0, 4, 16), _SPLITS):
            with test.subTest(case=name, tail_rows=tail_rows, split=split_name):
                plan = build_level_launch_plan(layout, device, tail_rows=tail_rows, **_SPLITS[split_name])
                values = wp.empty_like(original)
                solution = wp.zeros_like(rhs)

                def solve(plan=plan, layout=layout, values=values, solution=solution, original=original, rhs=rhs):
                    wp.copy(values, original)
                    launch_level_scheduled_solve(
                        plan, layout, values, rhs, solution, device, wp.zeros(1, dtype=wp.int32, device=device)
                    )

                solve()
                actual = solution.numpy()
                label = f"kernel {name} tail_rows={tail_rows} split={split_name} launches={plan.launch_count()}"
                _assert_launch_matches(test, plan, values, solution, ref_values, ref_solution, label)
                np.testing.assert_allclose(actual, expected, rtol=2.0e-4, atol=2.0e-6)
                with wp.ScopedCapture(device=device) as capture:
                    solve()
                for _ in range(3):
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(solution.numpy(), actual)


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
add_function_test(
    TestVBDSparseLevels,
    "test_level_launch_plan_covers_tasks",
    test_level_launch_plan_covers_tasks,
    devices=get_test_devices(),
)
add_function_test(
    TestVBDSparseLevels,
    "test_level_launch_serial_matches_levels",
    test_level_launch_serial_matches_levels,
    devices=[device for device in get_test_devices() if device.is_cpu],
)
add_function_test(
    TestVBDSparseLevels,
    "test_level_launch_matches_levels",
    test_level_launch_matches_levels,
    devices=[device for device in get_test_devices() if device.is_cuda],
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
