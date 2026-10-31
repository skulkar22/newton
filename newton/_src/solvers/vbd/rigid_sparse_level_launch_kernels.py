# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Level-scheduled block Cholesky as one CUDA launch per elimination level.

:func:`~newton._src.solvers.vbd.rigid_sparse_levels_kernels.solve_articulation_sparse_levels`
runs one thread block per articulation and separates the phases of a level with block barriers,
so one long articulation uses one multiprocessor. This module performs the same operations in
the same order as ordinary launches: one launch per (phase, level) over that level's tasks of
every articulation, so the large early levels spread over the whole GPU. Kernel boundaries
replace the block barriers.

Each task is the loop body of the matching phase of the cooperative kernel:

- diagonal: one thread per pivot row, the same 6 x 6 Cholesky;
- column: one thread per (column entry, block row), the same right solve;
- update: one thread per (destination block, scalar entry), the same serial loop over the
  destination's terms in the host-built order (one writer per entry, no atomics);
- forward and backward substitution: one thread per row, the same serial loop over the row's blocks.

A dense separator, such as a body that every branch of a loop-closed tree meets at, has a row
with hundreds of fill blocks and update destinations that receive one Schur term per adjacent
pivot of a level. One thread walking such a row or term list would be the critical path of its
launch. The plan therefore marks substitution rows with more than ``split_entries`` entries and
destinations with more than ``split_terms`` terms as long and cuts their entries into about
``sqrt(n)`` fixed chunks: the phase launch sums each chunk into a scratch buffer (6 threads per
substitution chunk, 36 per update chunk) and a second launch subtracts the chunk partials in
chunk order. Short tasks and chunks of a level share one launch. The summation order is fixed by
the plan, so the result is deterministic run to run, but a split task sums its terms in a
different order than the serial loop and is not bit-identical to the cooperative kernel.
Unsplit tasks are.

Levels with fewer pivot rows (summed over articulations) than ``LEVEL_LAUNCH_TAIL_ROWS`` form a tail that
:func:`solve_articulation_sparse_levels_tail` solves in one launch, one block per articulation,
as the cooperative kernel does; long substitution rows in the tail are summed by all threads of
the block and reduced in a fixed-order tree. The tail is a suffix of every articulation's levels,
so the launch order is: factor the leading levels, forward-substitute them, solve the tail
(factor, forward, backward), back-substitute the leading levels. Forward substitution of a level
needs only the factor of that level and earlier ones, and the tail factor writes no block of a
leading-level row, so this order reads the same data as the cooperative kernel.

All launch sizes are fixed when the plan is built, so the launches can be captured in a CUDA
graph. Extension points: a pivot guard or a modified diagonal factorization belongs in
:func:`level_launch_diag` and the tail kernel's diagonal phase; a row or column scaling of the
system belongs in :func:`launch_level_scheduled_solve` before the factor launches and after the
back substitution. Neither touches the cooperative kernel.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from .rigid_sparse_articulation import RigidArticulationSparseLayout
from .rigid_sparse_articulation_kernels import (
    _block_sync,
    _scalar_block_get,
    _scalar_block_set,
    _scalar_vec_get,
    _scalar_vec_load,
    _scalar_vec_set,
    _solve_lower66,
    _solve_upper66_from_lower,
    vec6f,
)
from .rigid_sparse_levels_kernels import (
    _cholesky66_scalar,
    _load_block66_scalar,
    _right_solve_lower_transpose66_scalar_row,
)

wp.set_module_options({"enable_backward": False})

LEVEL_LAUNCH_BLOCK_DIM = 128
"""Threads per block of the per-level launches (any value works; the tasks are independent)."""

LEVEL_LAUNCH_TAIL_ROWS = 16
"""Levels with fewer pivot rows than this (summed over articulations) form the cooperative tail.

A per-level launch costs its launch latency whatever its size, so the smallest levels are cheaper
inside one cooperative block. On the 31-arm rod array (5054 bodies, one articulation, 40 x 32)
thresholds of 0 / 16 / 32 measured 408.0 / 400.5 / 409.9 ms per frame; above 16 the single-block
tail serializes too much work.
"""

LEVEL_LAUNCH_SPLIT_ENTRIES = 32
"""Substitution rows with more block entries than this are split into chunks."""

LEVEL_LAUNCH_SPLIT_TERMS = 16
"""Update destinations with more Schur terms than this are split into chunks."""


# ---------------------------------------------------------------------------
# Per-task helpers: the loop bodies of the matching phases of
# ``solve_articulation_sparse_levels``, shared by the per-level and tail kernels.
# ---------------------------------------------------------------------------


@wp.func
def _level_update_entry(
    values: wp.array[float],
    dst: int,
    term_begin: int,
    term_end: int,
    level_update_left_slots: wp.array[wp.int32],
    level_update_right_slots: wp.array[wp.int32],
    i: int,
    j: int,
):
    value = _scalar_block_get(values, dst, i, j)
    for term in range(term_begin, term_end):
        left = level_update_left_slots[term]
        right = level_update_right_slots[term]
        accum = float(0.0)
        for k in range(6):
            accum = accum + _scalar_block_get(values, left, i, k) * _scalar_block_get(values, right, j, k)
        value = value - accum
    _scalar_block_set(values, dst, i, j, value)


@wp.func
def _level_forward_row(
    row: int,
    body_start: int,
    block_row_offsets: wp.array[wp.int32],
    block_cols: wp.array[wp.int32],
    diag_slots: wp.array[wp.int32],
    values: wp.array[float],
    rhs: wp.array[float],
    delta: wp.array[float],
):
    vector_value = _scalar_vec_load(rhs, row)
    for slot in range(block_row_offsets[row], block_row_offsets[row + 1]):
        col = body_start + block_cols[slot]
        if col < row:
            vector_value = vector_value - _load_block66_scalar(values, slot) * _scalar_vec_load(delta, col)
    result = _solve_lower66(_load_block66_scalar(values, diag_slots[row]), vector_value)
    for i in range(6):
        _scalar_vec_set(delta, row, i, result[i])


@wp.func
def _level_backward_row(
    row: int,
    body_start: int,
    block_col_offsets: wp.array[wp.int32],
    block_col_rows: wp.array[wp.int32],
    block_col_slots: wp.array[wp.int32],
    diag_slots: wp.array[wp.int32],
    values: wp.array[float],
    delta: wp.array[float],
):
    vector_value = _scalar_vec_load(delta, row)
    for entry in range(block_col_offsets[row], block_col_offsets[row + 1]):
        col = body_start + block_col_rows[entry]
        slot = block_col_slots[entry]
        vector_value = vector_value - wp.transpose(_load_block66_scalar(values, slot)) * _scalar_vec_load(delta, col)
    result = _solve_upper66_from_lower(_load_block66_scalar(values, diag_slots[row]), vector_value)
    for i in range(6):
        _scalar_vec_set(delta, row, i, result[i])


# ---------------------------------------------------------------------------
# Per-level kernels: one thread per task of one level across all articulations.
# ---------------------------------------------------------------------------


@wp.kernel
def level_launch_diag(
    task_begin: int,
    task_rows: wp.array[wp.int32],
    diag_slots: wp.array[wp.int32],
    values: wp.array[float],
):
    """Factor the diagonal block of each pivot row of one level."""
    row = task_rows[task_begin + wp.tid()]
    _cholesky66_scalar(values, diag_slots[row])


@wp.kernel
def level_launch_column(
    task_begin: int,
    task_columns: wp.array[wp.int32],
    level_column_slots: wp.array[wp.int32],
    level_column_diag_slots: wp.array[wp.int32],
    values: wp.array[float],
):
    """Right-solve each off-diagonal block row of the level's columns (6 tasks per block)."""
    task = wp.tid()
    local_entry = task // 6
    component = task - local_entry * 6
    entry = task_columns[task_begin + local_entry]
    _right_solve_lower_transpose66_scalar_row(
        values, level_column_slots[entry], level_column_diag_slots[entry], component
    )


@wp.kernel
def level_launch_update(
    short_begin: int,
    short_count: int,
    short_updates: wp.array[wp.int32],
    level_update_dst_slots: wp.array[wp.int32],
    level_update_term_offsets: wp.array[wp.int32],
    level_update_left_slots: wp.array[wp.int32],
    level_update_right_slots: wp.array[wp.int32],
    chunk_begin: int,
    chunk_term_begin: wp.array[wp.int32],
    chunk_term_end: wp.array[wp.int32],
    values: wp.array[float],
    update_partial: wp.array[float],
):
    """Apply the level's short Schur updates and sum the chunks of its long ones (36 tasks each).

    Tasks ``[0, 36 * short_count)`` update one scalar entry of a short destination with the
    serial term loop of the cooperative kernel. The remaining tasks sum one chunk of a long
    destination's terms into ``update_partial``; :func:`level_launch_update_reduce` applies them.
    """
    task = wp.tid()
    local = task // 36
    component = task - local * 36
    i = component // 6
    j = component - i * 6
    if local < short_count:
        update = short_updates[short_begin + local]
        _level_update_entry(
            values,
            level_update_dst_slots[update],
            level_update_term_offsets[update],
            level_update_term_offsets[update + 1],
            level_update_left_slots,
            level_update_right_slots,
            i,
            j,
        )
        return
    chunk = chunk_begin + local - short_count
    partial = float(0.0)
    for term in range(chunk_term_begin[chunk], chunk_term_end[chunk]):
        left = level_update_left_slots[term]
        right = level_update_right_slots[term]
        accum = float(0.0)
        for k in range(6):
            accum = accum + _scalar_block_get(values, left, i, k) * _scalar_block_get(values, right, j, k)
        partial = partial + accum
    update_partial[chunk * 36 + component] = partial


@wp.kernel
def level_launch_update_reduce(
    task_begin: int,
    long_updates: wp.array[wp.int32],
    long_chunk_offsets: wp.array[wp.int32],
    level_update_dst_slots: wp.array[wp.int32],
    update_partial: wp.array[float],
    values: wp.array[float],
):
    """Subtract a long destination's chunk partials in chunk order (36 tasks per destination)."""
    task = wp.tid()
    local_update = task // 36
    component = task - local_update * 36
    i = component // 6
    j = component - i * 6
    long_index = task_begin + local_update
    dst = level_update_dst_slots[long_updates[long_index]]
    value = _scalar_block_get(values, dst, i, j)
    for chunk in range(long_chunk_offsets[long_index], long_chunk_offsets[long_index + 1]):
        value = value - update_partial[chunk * 36 + component]
    _scalar_block_set(values, dst, i, j, value)


@wp.kernel
def level_launch_forward(
    short_begin: int,
    short_count: int,
    short_rows: wp.array[wp.int32],
    short_row_body_start: wp.array[wp.int32],
    block_row_offsets: wp.array[wp.int32],
    block_cols: wp.array[wp.int32],
    diag_slots: wp.array[wp.int32],
    values: wp.array[float],
    rhs: wp.array[float],
    chunk_begin: int,
    chunk_rows: wp.array[wp.int32],
    chunk_row_body_start: wp.array[wp.int32],
    chunk_entry_begin: wp.array[wp.int32],
    chunk_entry_end: wp.array[wp.int32],
    delta: wp.array[float],
    vector_partial: wp.array[float],
):
    """Forward-substitute the level's short rows and sum the chunks of its long rows.

    Tasks ``[0, short_count)`` solve one short row each, as the cooperative kernel does. The
    remaining tasks, 6 per chunk (one per output component, so neighboring threads read
    neighboring block entries), sum ``(L_rc delta_c)_i`` over one chunk of a long row's slots
    ``c < row`` into ``vector_partial``; :func:`level_launch_forward_reduce` finishes those rows.
    """
    task = wp.tid()
    if task < short_count:
        index = short_begin + task
        _level_forward_row(
            short_rows[index],
            short_row_body_start[index],
            block_row_offsets,
            block_cols,
            diag_slots,
            values,
            rhs,
            delta,
        )
        return
    local_chunk = (task - short_count) // 6
    i = task - short_count - local_chunk * 6
    chunk = chunk_begin + local_chunk
    row = chunk_rows[chunk]
    body_start = chunk_row_body_start[chunk]
    partial = float(0.0)
    for slot in range(chunk_entry_begin[chunk], chunk_entry_end[chunk]):
        col = body_start + block_cols[slot]
        if col < row:
            accum = float(0.0)
            for j in range(6):
                accum = accum + _scalar_block_get(values, slot, i, j) * _scalar_vec_get(delta, col, j)
            partial = partial + accum
    vector_partial[chunk * 6 + i] = partial


@wp.kernel
def level_launch_forward_reduce(
    task_begin: int,
    long_rows: wp.array[wp.int32],
    long_chunk_offsets: wp.array[wp.int32],
    diag_slots: wp.array[wp.int32],
    values: wp.array[float],
    rhs: wp.array[float],
    vector_partial: wp.array[float],
    delta: wp.array[float],
):
    """Finish the forward substitution of a long row from its chunk partials (chunk order)."""
    long_index = task_begin + wp.tid()
    row = long_rows[long_index]
    vector_value = _scalar_vec_load(rhs, row)
    for chunk in range(long_chunk_offsets[long_index], long_chunk_offsets[long_index + 1]):
        vector_value = vector_value - _scalar_vec_load(vector_partial, chunk)
    result = _solve_lower66(_load_block66_scalar(values, diag_slots[row]), vector_value)
    for i in range(6):
        _scalar_vec_set(delta, row, i, result[i])


@wp.kernel
def level_launch_backward(
    short_begin: int,
    short_count: int,
    short_rows: wp.array[wp.int32],
    short_row_body_start: wp.array[wp.int32],
    block_col_offsets: wp.array[wp.int32],
    block_col_rows: wp.array[wp.int32],
    block_col_slots: wp.array[wp.int32],
    diag_slots: wp.array[wp.int32],
    values: wp.array[float],
    chunk_begin: int,
    chunk_rows: wp.array[wp.int32],
    chunk_row_body_start: wp.array[wp.int32],
    chunk_entry_begin: wp.array[wp.int32],
    chunk_entry_end: wp.array[wp.int32],
    delta: wp.array[float],
    vector_partial: wp.array[float],
):
    """Back-substitute the level's short rows and sum the chunks of its long rows (as the forward pass)."""
    task = wp.tid()
    if task < short_count:
        index = short_begin + task
        _level_backward_row(
            short_rows[index],
            short_row_body_start[index],
            block_col_offsets,
            block_col_rows,
            block_col_slots,
            diag_slots,
            values,
            delta,
        )
        return
    local_chunk = (task - short_count) // 6
    i = task - short_count - local_chunk * 6
    chunk = chunk_begin + local_chunk
    body_start = chunk_row_body_start[chunk]
    partial = float(0.0)
    for entry in range(chunk_entry_begin[chunk], chunk_entry_end[chunk]):
        col = body_start + block_col_rows[entry]
        slot = block_col_slots[entry]
        accum = float(0.0)
        for j in range(6):
            accum = accum + _scalar_block_get(values, slot, j, i) * _scalar_vec_get(delta, col, j)
        partial = partial + accum
    vector_partial[chunk * 6 + i] = partial


@wp.kernel
def level_launch_backward_reduce(
    task_begin: int,
    long_rows: wp.array[wp.int32],
    long_chunk_offsets: wp.array[wp.int32],
    diag_slots: wp.array[wp.int32],
    values: wp.array[float],
    vector_partial: wp.array[float],
    delta: wp.array[float],
):
    """Finish the back substitution of a long row from its chunk partials (chunk order)."""
    long_index = task_begin + wp.tid()
    row = long_rows[long_index]
    vector_value = _scalar_vec_load(delta, row)
    for chunk in range(long_chunk_offsets[long_index], long_chunk_offsets[long_index + 1]):
        vector_value = vector_value - _scalar_vec_load(vector_partial, chunk)
    result = _solve_upper66_from_lower(_load_block66_scalar(values, diag_slots[row]), vector_value)
    for i in range(6):
        _scalar_vec_set(delta, row, i, result[i])


# ---------------------------------------------------------------------------
# Tail kernel: the cooperative solve restricted to the small trailing levels.
# ---------------------------------------------------------------------------


@wp.func
def _tail_reduce_lanes(tail_partial: wp.array[float], scratch: int, lane: int, threads: int, partial: vec6f):
    """Sum the block's 6-vectors into ``tail_partial[scratch]`` with a fixed-order tree (all lanes call)."""
    for i in range(6):
        _scalar_vec_set(tail_partial, scratch + lane, i, partial[i])
    _block_sync()
    count = threads
    while count > 1:
        half = (count + 1) // 2
        if lane < count - half:
            value = _scalar_vec_load(tail_partial, scratch + lane) + _scalar_vec_load(
                tail_partial, scratch + lane + half
            )
            for i in range(6):
                _scalar_vec_set(tail_partial, scratch + lane, i, value[i])
        _block_sync()
        count = half


@wp.kernel
def solve_articulation_sparse_levels_tail(
    level_start: int,
    articulation_body_offsets: wp.array[wp.int32],
    block_row_offsets: wp.array[wp.int32],
    block_cols: wp.array[wp.int32],
    block_col_offsets: wp.array[wp.int32],
    block_col_rows: wp.array[wp.int32],
    block_col_slots: wp.array[wp.int32],
    diag_slots: wp.array[wp.int32],
    articulation_level_offsets: wp.array[wp.int32],
    level_body_offsets: wp.array[wp.int32],
    level_column_offsets: wp.array[wp.int32],
    level_column_slots: wp.array[wp.int32],
    level_column_diag_slots: wp.array[wp.int32],
    level_update_offsets: wp.array[wp.int32],
    level_update_dst_slots: wp.array[wp.int32],
    level_update_term_offsets: wp.array[wp.int32],
    level_update_left_slots: wp.array[wp.int32],
    level_update_right_slots: wp.array[wp.int32],
    values: wp.array[float],
    rhs: wp.array[float],
    split_entries: int,
    tail_partial: wp.array[float],
    delta: wp.array[float],
):
    """Factor, forward- and back-solve each articulation's levels ``>= level_start`` in one block.

    Same work split as ``solve_articulation_sparse_levels`` restricted to the tail levels. The
    leading levels must already be factored and forward-substituted; their back substitution
    follows this launch. With ``split_entries > 0``, a substitution row with more than
    ``split_entries`` entries is summed by all threads of the block and reduced in
    ``tail_partial`` (``6 * block_dim`` floats per articulation) in a fixed order.
    """
    tid = wp.tid()
    threads = wp.block_dim()
    articulation = tid // threads
    lane = tid - articulation * threads
    body_start = articulation_body_offsets[articulation]
    last = articulation_level_offsets[articulation + 1]
    first = wp.min(articulation_level_offsets[articulation] + level_start, last)

    for level in range(first, last):
        begin = level_body_offsets[level]
        end = level_body_offsets[level + 1]
        for row in range(begin + lane, end, threads):
            _cholesky66_scalar(values, diag_slots[row])
        _block_sync()

        column_begin = level_column_offsets[level]
        column_end = level_column_offsets[level + 1]
        for task in range(column_begin * 6 + lane, column_end * 6, threads):
            entry = task // 6
            component = task - entry * 6
            _right_solve_lower_transpose66_scalar_row(
                values, level_column_slots[entry], level_column_diag_slots[entry], component
            )
        _block_sync()

        update_begin = level_update_offsets[level]
        update_end = level_update_offsets[level + 1]
        for task in range(update_begin * 36 + lane, update_end * 36, threads):
            update = task // 36
            component = task - update * 36
            i = component // 6
            j = component - i * 6
            _level_update_entry(
                values,
                level_update_dst_slots[update],
                level_update_term_offsets[update],
                level_update_term_offsets[update + 1],
                level_update_left_slots,
                level_update_right_slots,
                i,
                j,
            )
        _block_sync()

    scratch = articulation * threads
    for level in range(first, last):
        begin = level_body_offsets[level]
        end = level_body_offsets[level + 1]
        for row in range(begin + lane, end, threads):
            if split_entries <= 0 or block_row_offsets[row + 1] - block_row_offsets[row] <= split_entries:
                _level_forward_row(row, body_start, block_row_offsets, block_cols, diag_slots, values, rhs, delta)
        if split_entries > 0:
            for row in range(begin, end):
                if block_row_offsets[row + 1] - block_row_offsets[row] > split_entries:
                    partial = vec6f(0.0)
                    for slot in range(block_row_offsets[row] + lane, block_row_offsets[row + 1], threads):
                        col = body_start + block_cols[slot]
                        if col < row:
                            partial = partial + _load_block66_scalar(values, slot) * _scalar_vec_load(delta, col)
                    _tail_reduce_lanes(tail_partial, scratch, lane, threads, partial)
                    if lane == 0:
                        vector_value = _scalar_vec_load(rhs, row) - _scalar_vec_load(tail_partial, scratch)
                        result = _solve_lower66(_load_block66_scalar(values, diag_slots[row]), vector_value)
                        for i in range(6):
                            _scalar_vec_set(delta, row, i, result[i])
                    _block_sync()
        _block_sync()

    for reverse_level in range(last - first):
        level = last - reverse_level - 1
        begin = level_body_offsets[level]
        end = level_body_offsets[level + 1]
        for row in range(begin + lane, end, threads):
            if split_entries <= 0 or block_col_offsets[row + 1] - block_col_offsets[row] <= split_entries:
                _level_backward_row(
                    row, body_start, block_col_offsets, block_col_rows, block_col_slots, diag_slots, values, delta
                )
        if split_entries > 0:
            for row in range(begin, end):
                if block_col_offsets[row + 1] - block_col_offsets[row] > split_entries:
                    partial = vec6f(0.0)
                    for entry in range(block_col_offsets[row] + lane, block_col_offsets[row + 1], threads):
                        col = body_start + block_col_rows[entry]
                        slot = block_col_slots[entry]
                        partial = partial + wp.transpose(_load_block66_scalar(values, slot)) * _scalar_vec_load(
                            delta, col
                        )
                    _tail_reduce_lanes(tail_partial, scratch, lane, threads, partial)
                    if lane == 0:
                        vector_value = _scalar_vec_load(delta, row) - _scalar_vec_load(tail_partial, scratch)
                        result = _solve_upper66_from_lower(_load_block66_scalar(values, diag_slots[row]), vector_value)
                        for i in range(6):
                            _scalar_vec_set(delta, row, i, result[i])
                    _block_sync()
        _block_sync()


# ---------------------------------------------------------------------------
# Host-side plan and launch sequence.
# ---------------------------------------------------------------------------


def _chunk_bounds(begin: int, end: int, split: int) -> list[int]:
    """Chunk boundaries of ``[begin, end)``: one chunk up to ``split`` entries, else about ``sqrt(n)`` chunks.

    ``split <= 0`` never splits. Chunks of ``ceil(sqrt(n))`` entries balance the chunk pass and
    the serial reduction over the chunk partials.
    """
    count = end - begin
    if split <= 0 or count <= split:
        return [begin, end]
    size = max(2, int(np.ceil(np.sqrt(count))))
    return [*range(begin, end, size), end]


class _PhaseTaskBuilder:
    """Collects the short and long tasks of one phase, level by level (host build helper)."""

    def __init__(self):
        self.short, self.short_aux = [], []
        self.long, self.long_chunk_offsets = [], [0]
        self.chunk_owner, self.chunk_aux, self.chunk_begin, self.chunk_end = [], [], [], []
        self.short_offsets, self.long_offsets, self.chunk_offsets = [0], [0], [0]

    def add(self, task: int, aux: int, begin: int, end: int, split: int):
        bounds = _chunk_bounds(begin, end, split)
        if len(bounds) == 2:
            self.short.append(task)
            self.short_aux.append(aux)
            return
        self.long.append(task)
        for lo, hi in itertools.pairwise(bounds):
            self.chunk_begin.append(lo)
            self.chunk_end.append(hi)
            self.chunk_owner.append(task)
            self.chunk_aux.append(aux)
        self.long_chunk_offsets.append(len(self.chunk_owner))

    def close_level(self):
        self.short_offsets.append(len(self.short))
        self.long_offsets.append(len(self.long))
        self.chunk_offsets.append(len(self.chunk_owner))

    def build(self, device, partial_width: int) -> LevelLaunchPhaseTasks:
        def ints(values):
            return wp.array(np.asarray(values, dtype=np.int32).reshape(-1), dtype=wp.int32, device=device)

        return LevelLaunchPhaseTasks(
            short=ints(self.short),
            short_aux=ints(self.short_aux),
            long=ints(self.long),
            long_chunk_offsets=ints(self.long_chunk_offsets),
            chunk_owner=ints(self.chunk_owner),
            chunk_aux=ints(self.chunk_aux),
            chunk_begin=ints(self.chunk_begin),
            chunk_end=ints(self.chunk_end),
            partial=wp.zeros(partial_width * max(len(self.chunk_owner), 1), dtype=float, device=device),
            short_offsets=self.short_offsets,
            long_offsets=self.long_offsets,
            chunk_offsets=self.chunk_offsets,
        )


@dataclass
class LevelLaunchPhaseTasks:
    """Tasks of one phase (update, forward or backward) of the leading levels, level by level.

    A short task keeps the one-thread path of the cooperative kernel. A long task (row or
    destination) owns the chunks ``[long_chunk_offsets[t], long_chunk_offsets[t + 1])``; chunk
    ``c`` covers the entries ``[chunk_begin[c], chunk_end[c])`` and belongs to ``chunk_owner[c]``.
    ``*_aux`` is the first articulation row of a substitution task's articulation (unused by
    updates). The host offset lists give each level's ``[begin, end)`` in the three task lists.
    """

    short: wp.array[wp.int32]
    short_aux: wp.array[wp.int32]
    long: wp.array[wp.int32]
    long_chunk_offsets: wp.array[wp.int32]
    chunk_owner: wp.array[wp.int32]
    chunk_aux: wp.array[wp.int32]
    chunk_begin: wp.array[wp.int32]
    chunk_end: wp.array[wp.int32]
    partial: wp.array[float]
    """Scratch of the chunk partial sums: 36 floats per update chunk, 6 per substitution chunk."""
    short_offsets: list[int] = field(default_factory=list)
    long_offsets: list[int] = field(default_factory=list)
    chunk_offsets: list[int] = field(default_factory=list)

    def level_range(self, level: int) -> tuple[int, int, int, int, int, int]:
        """``(short_begin, short_count, chunk_begin, chunk_count, long_begin, long_count)`` of a level."""
        return (
            self.short_offsets[level],
            self.short_offsets[level + 1] - self.short_offsets[level],
            self.chunk_offsets[level],
            self.chunk_offsets[level + 1] - self.chunk_offsets[level],
            self.long_offsets[level],
            self.long_offsets[level + 1] - self.long_offsets[level],
        )


@dataclass
class RigidArticulationLevelLaunchPlan:
    """Static task lists of the per-level launches of the level-scheduled solve.

    Level ``l`` means the ``l``-th level of every articulation that has at least ``l + 1``
    levels; the task lists concatenate, per level, the matching level ranges of all
    articulations. Levels ``>= leading_level_count`` form the tail.
    """

    articulation_count: int
    """Articulations solved by the 6 x 6 level kernels (the layout's rigid prefix)."""
    level_count: int
    """Largest per-articulation level count."""
    leading_level_count: int
    """Leading levels solved by per-level launches; the rest is the tail."""
    rows_per_level: list[int]
    """Pivot rows of every level summed over articulations (all ``level_count`` levels)."""
    rows: wp.array[wp.int32]
    """Articulation rows of the leading levels, level by level (diagonal factor tasks)."""
    row_offsets: list[int]
    columns: wp.array[wp.int32]
    """Indices into ``level_column_slots`` of the leading levels, level by level."""
    column_offsets: list[int]
    update: LevelLaunchPhaseTasks
    forward: LevelLaunchPhaseTasks
    backward: LevelLaunchPhaseTasks
    tail_block_dim: int
    """Threads per block of the tail launch."""
    tail_split_entries: int
    """Tail substitution rows with more entries than this are summed by the whole block (0: none)."""
    tail_partial: wp.array[float]
    """Scratch of the tail's long rows, ``6 * tail_block_dim`` floats per articulation (or 6)."""

    @property
    def has_tail(self) -> bool:
        return self.leading_level_count < self.level_count

    def launch_count(self) -> int:
        """Kernel launches per solve (static)."""

        def nonempty(offsets, level):
            return int(offsets[level + 1] > offsets[level])

        count = 0
        for level in range(self.leading_level_count):
            count += nonempty(self.row_offsets, level) + nonempty(self.column_offsets, level)
            for phase in (self.update, self.forward, self.backward):
                count += max(nonempty(phase.short_offsets, level), nonempty(phase.chunk_offsets, level))
                count += nonempty(phase.long_offsets, level)
        return count + int(self.has_tail)


def build_level_launch_plan(
    layout: RigidArticulationSparseLayout,
    device: wp.context.Devicelike,
    *,
    tail_rows: int = LEVEL_LAUNCH_TAIL_ROWS,
    split_entries: int = LEVEL_LAUNCH_SPLIT_ENTRIES,
    split_terms: int = LEVEL_LAUNCH_SPLIT_TERMS,
    tail_block_dim: int = 128,
) -> RigidArticulationLevelLaunchPlan:
    """Build the static task lists of the per-level launches from a layout built with levels.

    Args:
        layout: Sparse layout built with ``use_levels=True``; its device arrays are read once here.
        device: Device of the task arrays.
        tail_rows: Levels from the first one with fewer pivot rows (summed over articulations)
            than this form the tail. ``0`` launches every level separately.
        split_entries: Substitution rows with more block entries than this are split into chunks.
            ``0`` disables splitting.
        split_terms: Update destinations with more Schur terms than this are split. ``0`` disables.
        tail_block_dim: Threads per block of the tail launch; sizes the tail's scratch.
    """
    levels = layout.levels
    if levels is None:
        raise ValueError("The sparse layout carries no levels; build it with use_levels=True.")
    articulation_count = layout.rigid_articulation_count
    level_starts = levels.articulation_level_offsets.numpy()
    body_offsets = levels.level_body_offsets.numpy()
    column_offsets = levels.level_column_offsets.numpy()
    update_offsets = levels.level_update_offsets.numpy()
    term_offsets = levels.level_update_term_offsets.numpy()
    articulation_body_offsets = layout.articulation_body_offsets.numpy()
    block_row_offsets = layout.articulation_block_row_offsets.numpy()
    block_col_offsets = layout.articulation_block_col_offsets.numpy()

    level_counts = [int(level_starts[a + 1] - level_starts[a]) for a in range(articulation_count)]
    level_count = max(level_counts, default=0)
    rows_per_level = [0] * level_count
    for a, count in enumerate(level_counts):
        for local in range(count):
            g = int(level_starts[a]) + local
            rows_per_level[local] += int(body_offsets[g + 1] - body_offsets[g])
    leading_level_count = level_count
    for local, rows in enumerate(rows_per_level):
        if rows < tail_rows:
            leading_level_count = local
            break

    rows, columns = [], []
    row_offset_list, column_offset_list = [0], [0]
    update, forward, backward = _PhaseTaskBuilder(), _PhaseTaskBuilder(), _PhaseTaskBuilder()
    for local in range(leading_level_count):
        for a, count in enumerate(level_counts):
            if local >= count:
                continue
            g = int(level_starts[a]) + local
            body_start = int(articulation_body_offsets[a])
            level_rows = range(int(body_offsets[g]), int(body_offsets[g + 1]))
            rows.extend(level_rows)
            columns.extend(range(int(column_offsets[g]), int(column_offsets[g + 1])))
            for dst in range(int(update_offsets[g]), int(update_offsets[g + 1])):
                update.add(dst, 0, int(term_offsets[dst]), int(term_offsets[dst + 1]), split_terms)
            for row in level_rows:
                forward.add(
                    row, body_start, int(block_row_offsets[row]), int(block_row_offsets[row + 1]), split_entries
                )
                backward.add(
                    row, body_start, int(block_col_offsets[row]), int(block_col_offsets[row + 1]), split_entries
                )
        row_offset_list.append(len(rows))
        column_offset_list.append(len(columns))
        for phase in (update, forward, backward):
            phase.close_level()

    # The tail sums long rows with the whole block; it needs scratch only if one exists.
    tail_split_entries = 0
    if leading_level_count < level_count and split_entries > 0:
        for a, count in enumerate(level_counts):
            if count <= leading_level_count:
                continue
            tail_rows_ = np.arange(
                int(body_offsets[int(level_starts[a]) + leading_level_count]),
                int(body_offsets[int(level_starts[a + 1])]),
            )
            longest = max(
                int(np.max(block_row_offsets[tail_rows_ + 1] - block_row_offsets[tail_rows_])),
                int(np.max(block_col_offsets[tail_rows_ + 1] - block_col_offsets[tail_rows_])),
            )
            if longest > split_entries:
                tail_split_entries = split_entries
                break
    tail_scratch = 6 * articulation_count * tail_block_dim if tail_split_entries > 0 else 6

    def ints(values):
        return wp.array(np.asarray(values, dtype=np.int32).reshape(-1), dtype=wp.int32, device=device)

    return RigidArticulationLevelLaunchPlan(
        articulation_count=articulation_count,
        level_count=level_count,
        leading_level_count=leading_level_count,
        rows_per_level=rows_per_level,
        rows=ints(rows),
        row_offsets=row_offset_list,
        columns=ints(columns),
        column_offsets=column_offset_list,
        update=update.build(device, 36),
        forward=forward.build(device, 6),
        backward=backward.build(device, 6),
        tail_block_dim=tail_block_dim,
        tail_split_entries=tail_split_entries,
        tail_partial=wp.zeros(tail_scratch, dtype=float, device=device),
    )


def launch_level_scheduled_solve(
    plan: RigidArticulationLevelLaunchPlan,
    layout: RigidArticulationSparseLayout,
    values: wp.array[float],
    rhs: wp.array[float],
    delta: wp.array[float],
    device: wp.context.Devicelike,
    *,
    block_dim: int = LEVEL_LAUNCH_BLOCK_DIM,
) -> None:
    """Solve ``A delta = rhs`` for the level-ordered articulations with per-level launches.

    Equivalent to one launch of ``solve_articulation_sparse_levels`` over
    ``plan.articulation_count`` blocks. ``values`` is overwritten by the factor. All launch sizes
    come from the plan, so the sequence can be captured in a CUDA graph. ``block_dim`` is the
    block size of the per-level launches; the tail uses ``plan.tail_block_dim``.
    """
    levels = layout.levels

    def launch(kernel, dim, inputs, outputs):
        if dim > 0:
            wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=device, block_dim=block_dim)

    def offsets_range(offsets, level):
        return offsets[level], offsets[level + 1] - offsets[level]

    for level in range(plan.leading_level_count):
        begin, count = offsets_range(plan.row_offsets, level)
        launch(level_launch_diag, count, [begin, plan.rows, layout.articulation_diag_slots], [values])
        begin, count = offsets_range(plan.column_offsets, level)
        launch(
            level_launch_column,
            6 * count,
            [begin, plan.columns, levels.level_column_slots, levels.level_column_diag_slots],
            [values],
        )
        short_begin, short_count, chunk_begin, chunk_count, long_begin, long_count = plan.update.level_range(level)
        launch(
            level_launch_update,
            36 * (short_count + chunk_count),
            [
                short_begin,
                short_count,
                plan.update.short,
                levels.level_update_dst_slots,
                levels.level_update_term_offsets,
                levels.level_update_left_slots,
                levels.level_update_right_slots,
                chunk_begin,
                plan.update.chunk_begin,
                plan.update.chunk_end,
            ],
            [values, plan.update.partial],
        )
        launch(
            level_launch_update_reduce,
            36 * long_count,
            [
                long_begin,
                plan.update.long,
                plan.update.long_chunk_offsets,
                levels.level_update_dst_slots,
                plan.update.partial,
            ],
            [values],
        )

    for level in range(plan.leading_level_count):
        short_begin, short_count, chunk_begin, chunk_count, long_begin, long_count = plan.forward.level_range(level)
        launch(
            level_launch_forward,
            short_count + 6 * chunk_count,
            [
                short_begin,
                short_count,
                plan.forward.short,
                plan.forward.short_aux,
                layout.articulation_block_row_offsets,
                layout.articulation_block_cols,
                layout.articulation_diag_slots,
                values,
                rhs,
                chunk_begin,
                plan.forward.chunk_owner,
                plan.forward.chunk_aux,
                plan.forward.chunk_begin,
                plan.forward.chunk_end,
            ],
            [delta, plan.forward.partial],
        )
        launch(
            level_launch_forward_reduce,
            long_count,
            [
                long_begin,
                plan.forward.long,
                plan.forward.long_chunk_offsets,
                layout.articulation_diag_slots,
                values,
                rhs,
                plan.forward.partial,
            ],
            [delta],
        )

    if plan.has_tail:
        wp.launch(
            solve_articulation_sparse_levels_tail,
            dim=plan.articulation_count * plan.tail_block_dim,
            inputs=[
                plan.leading_level_count,
                layout.articulation_body_offsets,
                layout.articulation_block_row_offsets,
                layout.articulation_block_cols,
                layout.articulation_block_col_offsets,
                layout.articulation_block_col_rows,
                layout.articulation_block_col_slots,
                layout.articulation_diag_slots,
                levels.articulation_level_offsets,
                levels.level_body_offsets,
                levels.level_column_offsets,
                levels.level_column_slots,
                levels.level_column_diag_slots,
                levels.level_update_offsets,
                levels.level_update_dst_slots,
                levels.level_update_term_offsets,
                levels.level_update_left_slots,
                levels.level_update_right_slots,
                values,
                rhs,
                plan.tail_split_entries,
                plan.tail_partial,
            ],
            outputs=[delta],
            device=device,
            block_dim=plan.tail_block_dim,
        )

    for level in reversed(range(plan.leading_level_count)):
        short_begin, short_count, chunk_begin, chunk_count, long_begin, long_count = plan.backward.level_range(level)
        launch(
            level_launch_backward,
            short_count + 6 * chunk_count,
            [
                short_begin,
                short_count,
                plan.backward.short,
                plan.backward.short_aux,
                layout.articulation_block_col_offsets,
                layout.articulation_block_col_rows,
                layout.articulation_block_col_slots,
                layout.articulation_diag_slots,
                values,
                chunk_begin,
                plan.backward.chunk_owner,
                plan.backward.chunk_aux,
                plan.backward.chunk_begin,
                plan.backward.chunk_end,
            ],
            [delta, plan.backward.partial],
        )
        launch(
            level_launch_backward_reduce,
            long_count,
            [
                long_begin,
                plan.backward.long,
                plan.backward.long_chunk_offsets,
                layout.articulation_diag_slots,
                values,
                plan.backward.partial,
            ],
            [delta],
        )
