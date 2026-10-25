# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Variable-size block Cholesky for mixed rigid/elastic articulations in the VBD solve.

A mixed articulation (see :class:`~newton._src.solvers.vbd.rigid_sparse_articulation.RigidArticulationSparseLayout`
with ``elastic_solve="coupled"``) has one sparse row per body of size ``n_i = 6 + m_i``:
six frame components followed by the ``m_i`` modal coordinates of a reduced elastic body
(``m_i = 0`` for a rigid body). The backend solves ``H dx = rhs`` per articulation, where
``rhs`` is the negative gradient, with a direct block Cholesky factorization ``H = L L^T``
in the elimination order and fill pattern stored by the layout.

Storage, shared by every device and by the assembly kernels:

- Block values: one flat ``float32`` array of ``layout.articulation_value_count`` scalars.
  Slot ``s`` (row ``i``, column ``j <= i`` in the articulation's local order) holds the dense
  row-major ``R x C`` block ``H_ij`` with ``R = articulation_block_dims[s, 0]`` and
  ``C = articulation_block_dims[s, 1]``; entry ``(r, c)`` is at
  ``articulation_block_value_offsets[s] + r * C + c`` (:func:`vb_block_entry_index`).
  Only the lower block triangle is stored. Diagonal blocks are assembled as full symmetric
  square blocks; the factorization overwrites the lower triangle of every stored block with
  ``L`` and zeroes the strict upper triangle of diagonal blocks. For an all-6 layout the
  offsets are ``36 * s``, the legacy ``values_scalar`` layout.
- Vectors (right-hand side, update, diagonal backup): flat ``float32`` arrays of
  ``layout.articulation_scalar_count`` scalars; component ``k`` of row ``r`` is at
  ``articulation_row_scalar_offsets[r] + k`` (:func:`vb_rhs_index`); for size-6 rows this is
  ``6 * r + k``. Components ``0..2`` are the world-axes displacement of the body's COM,
  ``3..5`` its world-axes rotation vector, and ``6..n_i - 1`` its modal coordinates in the
  basis order, matching the displacement-form update of the 6 x 6 kernels.
- Rows and slots are global: articulation ``a`` owns rows
  ``[articulation_body_offsets[a], articulation_body_offsets[a + 1])`` and the slots of
  those rows. Mixed articulations are ``[rigid_articulation_count, articulation_count)``; the
  rigid prefix is solved by the 6 x 6 kernels and its part of the arrays is not touched here.
- Per mixed articulation ``t = a - rigid_articulation_count``: ``factor_status[t]`` and a
  shared ``int32`` failure counter.

The factorization is right-looking. For each column ``k``: factor ``L_kk`` in place, form
``L_ik = H_ik L_kk^-T`` for the column's entries (``articulation_block_col_*``), and apply the
trailing updates ``H_ab -= L_ak L_bk^T`` listed in ``articulation_factor_update_*``. The
column and update lists are the ones the 6 x 6 kernels use. No float atomics are used in
the factorization or the solve.

Failure policy: a pivot that is not positive or not finite, or a non-finite solution, sets
``factor_status`` (:data:`VB_FACTOR_INDEFINITE`, :data:`VB_SOLVE_NONFINITE`), increments the
failure counter, and replaces the articulation's update by a scalar Jacobi step
``dx_k = rhs_k / d_k`` from the diagonal ``d`` saved before factorization (0 where ``d_k`` is
not positive and finite, or ``rhs_k`` is not finite), so the update is always finite.

Frame regularization: the backend adds ``frame_regularization`` to the six frame diagonal
entries of every row of a mixed articulation before factorization (the 6 x 6 kernels apply
``rigid_articulation_diagonal_regularization`` to the rigid prefix through the body Hessians).

Backends (:func:`launch_solve_articulation_variable` picks one by device):

- CPU: :func:`solve_articulation_variable_serial`, one thread per mixed articulation.
- CUDA: :data:`solve_articulation_variable_cooperative`, one CTA of
  :data:`VB_COOPERATIVE_BLOCK_DIM` (256) threads per mixed articulation, the same elimination
  order, column and update lists, regularization and failure policy. Column ``k`` takes the
  *lane path* when ``n_k = 6``: every lane factors ``L_kk`` in registers (the same operations as
  the serial kernel, so the same bits) and lane 0 stores it; the column solves give one block
  row and the trailing updates one destination entry to each lane, numbered across all blocks
  of the column. In the substitutions each lane owns one vector component of the block row for
  the off-diagonal products, and a 6 x 6 diagonal solve runs on lane 0 in registers. The lane
  path repeats the serial arithmetic, so all-6 systems agree bitwise with the serial kernel.
  Wider columns take the *wide path*, right-looking over 32-wide panels of the whole column
  (diagonal block and column blocks): the panel is factored with ``tile_cholesky_inplace`` on a
  32 x 32 shared-memory view (:func:`_vb_view`, zero filled past the block edge, padded
  diagonal set to the identity), or by warp 0 for a last panel of at most 16 columns; every
  valid pivot is checked, since ``tile_cholesky_inplace`` reports no failure. Each thread then
  solves whole rows below the panel in registers (:func:`_vb_panel_trsm`), and one batch of
  block updates (:func:`_vb_update_batch`) applies the rank-32 update to the trailing columns.
  After the column, a second batch applies its trailing updates ``H_ab -= L_ak L_bk^T`` with
  the full inner size. The batches run 64 x 64 tiles of all their blocks on groups of 64
  threads, with every entry accumulated in registers and written once. In the substitutions a
  wide diagonal block is solved with 32 x 32 tile panels (``tile_lower_solve_inplace``,
  ``tile_matmul``). Bodies are not padded to the largest modal count; only the last partial
  panel or tile of a block is. Control flow is uniform across the CTA: branches depend on block
  sizes or on ``factor_status`` and pivot flags, which single lanes write and all lanes read
  after a barrier. An explicit ``__syncthreads`` follows every phase that stores to global
  memory (``tile_store`` issues none). The kernel neither allocates nor synchronizes with the
  host, so it can be captured in a CUDA graph; the wide path is accurate to float32 rounding,
  not bitwise equal to the serial kernel (its summation order differs). On the CPU the
  cooperative kernel runs with ``block_dim = 1`` (for testing); the native helpers then run
  their serial branch.

Shared memory of the cooperative kernel: the native helpers reserve about 45 KiB of static
shared memory (staging of the block updates for four groups, their job table, and two 32 x 33
panels). The tile operations add dynamic shared memory: one 32 x 32 ``float32`` tile is 4 KiB,
and every tile expression in a function frame owns its own buffer, so each tile phase is a
separate ``wp.func`` (a CTA needs the largest phase frame instead of the sum) and every loop
around a tile operation has a runtime trip count, so loop unrolling cannot duplicate tiles.
:func:`cooperative_shared_memory_bytes` returns the sum, which must stay under the design
budget :data:`VB_COOPERATIVE_SHARED_MEMORY_BUDGET` (64 KiB), and raises if it exceeds the
device limit; the CUDA launcher calls it once per device. With 256 threads the kernel uses up to
255 registers per thread, so one CTA runs per SM; the kernel targets few large articulations
(one CTA each) rather than many small ones. The cooperative kernel has its own Warp module
(``module="unique"``), which keeps its libmathdx build (tens of seconds on a cold kernel cache)
out of the module of the light kernels; it uses no float atomics, so it does not depend on the
deterministic mode.
"""

import logging

import warp as wp

from newton._src.solvers.vbd.rigid_sparse_articulation_kernels import (
    _apply_sparse_delta_value_to_body,
    _block_sync,
    mat66f,
    vec6f,
)

wp.set_module_options({"enable_backward": False})

_logger = logging.getLogger(__name__)

VB_FACTOR_OK = wp.constant(0)
"""``factor_status`` of an articulation whose factorization and solve succeeded."""
VB_FACTOR_INDEFINITE = wp.constant(1)
"""``factor_status`` of an articulation with a pivot that is not positive or not finite."""
VB_SOLVE_NONFINITE = wp.constant(2)
"""``factor_status`` of an articulation whose triangular solves produced a non-finite update."""


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


@wp.func
def vb_block_entry_index(
    block_value_offsets: wp.array[wp.int32],
    block_dims: wp.array2d[wp.int32],
    slot: int,
    r: int,
    c: int,
) -> int:
    """Return the flat index of entry ``(r, c)`` of block slot ``slot``."""
    return block_value_offsets[slot] + r * block_dims[slot, 1] + c


@wp.func
def vb_rhs_index(row_scalar_offsets: wp.array[wp.int32], row: int, k: int) -> int:
    """Return the flat index of component ``k`` of sparse row ``row`` in a vector."""
    return row_scalar_offsets[row] + k


# ---------------------------------------------------------------------------
# Offset-based dense routines on row-major blocks
# ---------------------------------------------------------------------------


@wp.func
def _vb_pivot_ok(value: float) -> bool:
    return value > 0.0 and wp.isfinite(value)


@wp.func
def _vb_cholesky66_inplace(values: wp.array[float], offset: int) -> int:
    """Factor the 6 x 6 row-major block at ``offset`` in place; return 1, or 0 on a bad pivot."""
    ok = int(1)
    for j in range(6):
        if ok == 1:
            pivot = values[offset + j * 6 + j]
            for p in range(6):
                if p < j:
                    pivot = pivot - values[offset + j * 6 + p] * values[offset + j * 6 + p]
            if _vb_pivot_ok(pivot):
                d = wp.sqrt(pivot)
                values[offset + j * 6 + j] = d
                for i in range(6):
                    if i > j:
                        value = values[offset + i * 6 + j]
                        for p in range(6):
                            if p < j:
                                value = value - values[offset + i * 6 + p] * values[offset + j * 6 + p]
                        values[offset + i * 6 + j] = value / d
                        values[offset + j * 6 + i] = 0.0
            else:
                ok = 0
    return ok


@wp.func
def _vb_load66(values: wp.array[float], offset: int) -> mat66f:
    """Return the 6 x 6 row-major block at ``offset``."""
    A = mat66f(0.0)
    for i in range(6):
        for j in range(6):
            A[i, j] = values[offset + i * 6 + j]
    return A


@wp.func
def _vb_cholesky66_registers(A: mat66f):
    """Factor a 6 x 6 block held in registers; return ``(L, ok)``.

    The same operations, in the same order, as :func:`_vb_cholesky66_inplace`, so ``L`` matches
    the in-place factor bitwise (also when a pivot fails: the remaining entries are unchanged).
    """
    ok = int(1)
    for j in range(6):
        if ok == 1:
            pivot = A[j, j]
            for p in range(6):
                if p < j:
                    pivot = pivot - A[j, p] * A[j, p]
            if _vb_pivot_ok(pivot):
                d = wp.sqrt(pivot)
                A[j, j] = d
                for i in range(6):
                    if i > j:
                        value = A[i, j]
                        for p in range(6):
                            if p < j:
                                value = value - A[i, p] * A[j, p]
                        A[i, j] = value / d
                        A[j, i] = 0.0
            else:
                ok = 0
    return A, ok


@wp.func
def _vb_cholesky_inplace(values: wp.array[float], offset: int, n: int) -> int:
    """Factor the ``n x n`` row-major block at ``offset`` in place (left-looking, by column).

    Returns 1 on success and 0 at the first pivot that is not positive or not finite.
    """
    if n == 6:
        return _vb_cholesky66_inplace(values, offset)
    ok = int(1)
    for j in range(n):
        if ok == 1:
            pivot = values[offset + j * n + j]
            for p in range(j):
                pivot = pivot - values[offset + j * n + p] * values[offset + j * n + p]
            if _vb_pivot_ok(pivot):
                d = wp.sqrt(pivot)
                values[offset + j * n + j] = d
                for i in range(j + 1, n):
                    value = values[offset + i * n + j]
                    for p in range(j):
                        value = value - values[offset + i * n + p] * values[offset + j * n + p]
                    values[offset + i * n + j] = value / d
                    values[offset + j * n + i] = 0.0
            else:
                ok = 0
    return ok


@wp.func
def _vb_right_solve_lower_transpose(values: wp.array[float], offset: int, rows: int, cols: int, diag_offset: int):
    """Overwrite the ``rows x cols`` block ``A`` at ``offset`` with ``A L^-T`` (``L`` at ``diag_offset``)."""
    for r in range(rows):
        row_offset = offset + r * cols
        for c in range(cols):
            value = values[row_offset + c]
            for p in range(c):
                value = value - values[row_offset + p] * values[diag_offset + c * cols + p]
            values[row_offset + c] = value / values[diag_offset + c * cols + c]


@wp.func
def _vb_subtract_outer(
    values: wp.array[float],
    dst_offset: int,
    left_offset: int,
    right_offset: int,
    left_rows: int,
    right_rows: int,
    inner: int,
    lower_only: bool,
):
    """``D -= A B^T`` for ``A`` (``left_rows x inner``), ``B`` (``right_rows x inner``), ``D`` (``left_rows x right_rows``)."""
    for r in range(left_rows):
        c_end = right_rows
        if lower_only:
            c_end = r + 1
        for c in range(c_end):
            accum = float(0.0)
            for p in range(inner):
                accum = accum + values[left_offset + r * inner + p] * values[right_offset + c * inner + p]
            values[dst_offset + r * right_rows + c] = values[dst_offset + r * right_rows + c] - accum


@wp.func
def _vb_subtract_block_vec(
    values: wp.array[float], offset: int, rows: int, cols: int, x: wp.array[float], x_offset: int, y_offset: int
):
    """``y -= A x`` with ``A`` (``rows x cols``) at ``offset``; ``x`` and ``y`` are segments of the same vector."""
    for r in range(rows):
        accum = float(0.0)
        for c in range(cols):
            accum = accum + values[offset + r * cols + c] * x[x_offset + c]
        x[y_offset + r] = x[y_offset + r] - accum


@wp.func
def _vb_subtract_block_transpose_vec(
    values: wp.array[float], offset: int, rows: int, cols: int, x: wp.array[float], x_offset: int, y_offset: int
):
    """``y -= A^T x`` with ``A`` (``rows x cols``) at ``offset``; ``x`` has ``rows`` and ``y`` ``cols`` entries."""
    for c in range(cols):
        accum = float(0.0)
        for r in range(rows):
            accum = accum + values[offset + r * cols + c] * x[x_offset + r]
        x[y_offset + c] = x[y_offset + c] - accum


@wp.func
def _vb_lower_solve(values: wp.array[float], diag_offset: int, n: int, x: wp.array[float], x_offset: int):
    """Solve ``L y = x`` in place for the factored ``n x n`` diagonal block."""
    for i in range(n):
        value = x[x_offset + i]
        for j in range(i):
            value = value - values[diag_offset + i * n + j] * x[x_offset + j]
        x[x_offset + i] = value / values[diag_offset + i * n + i]


@wp.func
def _vb_lower_solve66_registers(values: wp.array[float], diag_offset: int, x: wp.array[float], x_offset: int):
    """:func:`_vb_lower_solve` for a 6 x 6 block with the operands in registers (same operations and order)."""
    L = _vb_load66(values, diag_offset)
    y = vec6f(x[x_offset], x[x_offset + 1], x[x_offset + 2], x[x_offset + 3], x[x_offset + 4], x[x_offset + 5])
    for i in range(6):
        value = y[i]
        for j in range(6):
            if j < i:
                value = value - L[i, j] * y[j]
        y[i] = value / L[i, i]
    for i in range(6):
        x[x_offset + i] = y[i]


@wp.func
def _vb_upper_solve66_registers(values: wp.array[float], diag_offset: int, x: wp.array[float], x_offset: int):
    """:func:`_vb_upper_solve` for a 6 x 6 block with the operands in registers (same operations and order)."""
    L = _vb_load66(values, diag_offset)
    y = vec6f(x[x_offset], x[x_offset + 1], x[x_offset + 2], x[x_offset + 3], x[x_offset + 4], x[x_offset + 5])
    for ii in range(6):
        i = 5 - ii
        value = y[i]
        for j in range(6):
            if j > i:
                value = value - L[j, i] * y[j]
        y[i] = value / L[i, i]
    for i in range(6):
        x[x_offset + i] = y[i]


@wp.func
def _vb_upper_solve(values: wp.array[float], diag_offset: int, n: int, x: wp.array[float], x_offset: int):
    """Solve ``L^T y = x`` in place for the factored ``n x n`` diagonal block."""
    for ii in range(n):
        i = n - 1 - ii
        value = x[x_offset + i]
        for j in range(i + 1, n):
            value = value - values[diag_offset + j * n + i] * x[x_offset + j]
        x[x_offset + i] = value / values[diag_offset + i * n + i]


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def solve_articulation_variable_serial(
    rigid_articulation_count: int,
    frame_regularization: float,
    articulation_body_offsets: wp.array[wp.int32],
    articulation_row_dims: wp.array[wp.int32],
    articulation_row_scalar_offsets: wp.array[wp.int32],
    articulation_block_value_offsets: wp.array[wp.int32],
    articulation_block_dims: wp.array2d[wp.int32],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    articulation_block_col_offsets: wp.array[wp.int32],
    articulation_block_col_rows: wp.array[wp.int32],
    articulation_block_col_slots: wp.array[wp.int32],
    articulation_factor_update_offsets: wp.array[wp.int32],
    articulation_factor_update_dst_slots: wp.array[wp.int32],
    articulation_factor_update_left_slots: wp.array[wp.int32],
    articulation_factor_update_right_slots: wp.array[wp.int32],
    articulation_diag_slots: wp.array[wp.int32],
    rhs: wp.array[float],
    values: wp.array[float],
    diag_backup: wp.array[float],
    factor_status: wp.array[wp.int32],
    failure_count: wp.array[wp.int32],
    delta: wp.array[float],
):
    """Factor and solve one mixed articulation per thread (right-looking block Cholesky).

    Launch with ``dim = mixed_articulation_count``. ``values`` is factored in place; ``rhs`` is
    read only. See the module docstring for storage, regularization and failure handling.
    """
    tid = wp.tid()
    articulation = rigid_articulation_count + tid
    body_start = articulation_body_offsets[articulation]
    body_end = articulation_body_offsets[articulation + 1]

    # Regularize the frame rows and keep the diagonal for the Jacobi fallback.
    for row in range(body_start, body_end):
        n = articulation_row_dims[row]
        diag_offset = articulation_block_value_offsets[articulation_diag_slots[row]]
        vec_offset = articulation_row_scalar_offsets[row]
        for k in range(n):
            if k < 6:
                values[diag_offset + k * n + k] = values[diag_offset + k * n + k] + frame_regularization
            diag_backup[vec_offset + k] = values[diag_offset + k * n + k]

    ok = int(1)
    status = VB_FACTOR_OK
    for row_k in range(body_start, body_end):
        if ok == 1:
            n_k = articulation_row_dims[row_k]
            diag_offset = articulation_block_value_offsets[articulation_diag_slots[row_k]]
            ok = _vb_cholesky_inplace(values, diag_offset, n_k)
            if ok == 1:
                for entry in range(articulation_block_col_offsets[row_k], articulation_block_col_offsets[row_k + 1]):
                    slot = articulation_block_col_slots[entry]
                    _vb_right_solve_lower_transpose(
                        values,
                        articulation_block_value_offsets[slot],
                        articulation_block_dims[slot, 0],
                        n_k,
                        diag_offset,
                    )
                for entry in range(
                    articulation_factor_update_offsets[row_k], articulation_factor_update_offsets[row_k + 1]
                ):
                    dst = articulation_factor_update_dst_slots[entry]
                    left = articulation_factor_update_left_slots[entry]
                    right = articulation_factor_update_right_slots[entry]
                    _vb_subtract_outer(
                        values,
                        articulation_block_value_offsets[dst],
                        articulation_block_value_offsets[left],
                        articulation_block_value_offsets[right],
                        articulation_block_dims[left, 0],
                        articulation_block_dims[right, 0],
                        n_k,
                        left == right,
                    )
            else:
                status = VB_FACTOR_INDEFINITE

    if ok == 1:
        for row in range(body_start, body_end):
            vec_offset = articulation_row_scalar_offsets[row]
            for k in range(articulation_row_dims[row]):
                delta[vec_offset + k] = rhs[vec_offset + k]
        # Forward substitution with the stored lower block rows.
        for row in range(body_start, body_end):
            n = articulation_row_dims[row]
            vec_offset = articulation_row_scalar_offsets[row]
            diag_slot = articulation_diag_slots[row]
            for slot in range(articulation_block_row_offsets[row], articulation_block_row_offsets[row + 1]):
                if slot != diag_slot:
                    col_row = body_start + articulation_block_cols[slot]
                    _vb_subtract_block_vec(
                        values,
                        articulation_block_value_offsets[slot],
                        n,
                        articulation_row_dims[col_row],
                        delta,
                        articulation_row_scalar_offsets[col_row],
                        vec_offset,
                    )
            _vb_lower_solve(values, articulation_block_value_offsets[diag_slot], n, delta, vec_offset)
        # Back substitution with the column lists (L^T).
        for rr in range(body_end - body_start):
            row = body_end - 1 - rr
            n = articulation_row_dims[row]
            vec_offset = articulation_row_scalar_offsets[row]
            for entry in range(articulation_block_col_offsets[row], articulation_block_col_offsets[row + 1]):
                slot = articulation_block_col_slots[entry]
                below = body_start + articulation_block_col_rows[entry]
                _vb_subtract_block_transpose_vec(
                    values,
                    articulation_block_value_offsets[slot],
                    articulation_row_dims[below],
                    n,
                    delta,
                    articulation_row_scalar_offsets[below],
                    vec_offset,
                )
            _vb_upper_solve(
                values, articulation_block_value_offsets[articulation_diag_slots[row]], n, delta, vec_offset
            )
        begin = articulation_row_scalar_offsets[body_start]
        end = articulation_row_scalar_offsets[body_end]
        for i in range(begin, end):
            if not wp.isfinite(delta[i]):
                ok = 0
                status = VB_SOLVE_NONFINITE

    factor_status[tid] = status
    if ok == 0:
        wp.atomic_add(failure_count, 0, 1)
        begin = articulation_row_scalar_offsets[body_start]
        end = articulation_row_scalar_offsets[body_end]
        for i in range(begin, end):
            d = diag_backup[i]
            r = rhs[i]
            step = float(0.0)
            if _vb_pivot_ok(d) and wp.isfinite(r):
                step = r / d
            delta[i] = step


# ---------------------------------------------------------------------------
# Cooperative CUDA kernel (one CTA per mixed articulation)
# ---------------------------------------------------------------------------

VB_COOPERATIVE_BLOCK_DIM = 256
"""Threads per CTA of :data:`solve_articulation_variable_cooperative` on CUDA devices.

A multiple of 64, at least 64: the block updates run in groups of 64 threads (at most four
groups are used) and narrow panel factorizations on warp 0.
"""

VB_COOPERATIVE_SHARED_MEMORY_BUDGET = 64 * 1024
"""Design budget in bytes for the shared memory of one cooperative CTA (see the module docstring)."""

_VB_STATIC_SHARED_MEMORY_ESTIMATE = 4 * (2 * 4 * 16 * 68 + 64 * 9 + 2 * 32 * 33) + 8
"""Static shared memory of the native helpers in bytes (update staging, job table, two panels, flags)."""

_VB_TILE = wp.constant(32)
_VB_SMALL_PANEL = wp.constant(16)
_VB_LANE_MAX_DIM = wp.constant(6)


@wp.func
def _vb_view(values: wp.array[float], offset: int, rows: int, cols: int):
    """Row-major ``rows x cols`` view of ``values[offset : offset + rows * cols]``.

    The view aliases ``values`` (no copy); it is a valid tile source and target whose runtime
    shape bounds ``tile_load`` zero fill and ``tile_store`` clipping. The layout guarantees that
    every block lies inside the array (int32 offsets, checked when the layout is built).
    """
    return wp.array(ptr=values.ptr + wp.uint64(offset) * wp.uint64(4), shape=(rows, cols), dtype=float)


@wp.func
def _vb_pad_entry(index: float, valid: float) -> float:
    """Diagonal padding of a partial panel: 1 for ``index >= valid``, else 0.

    Used as ``tile_map(_vb_pad_entry, tile_arange(32), valid)`` and added to the diagonal of a
    zero-filled partial diagonal panel, which makes the padded part the identity, so Cholesky
    and triangular solves stay finite and leave the valid part unchanged. The tile is built
    inline at each use: returning a register tile from a ``wp.func`` faults on CUDA in Warp 1.18.
    """
    if index >= valid:
        return 1.0
    return 0.0


_VB_UPDATE_PANEL = wp.constant(0)
_VB_UPDATE_COLUMN = wp.constant(1)


@wp.func_native("""
// Job j of a batch: D_j -= A_j B_j^T (lower triangle only when lower), all of inner size `inner`.
//   mode 0 (panel update of a wide column at panel offset o, inner = 32): job 0 is the trailing
//   part of the diagonal block, rows/cols o + 32 .. n - 1 (lower); job 1 + e is the trailing
//   part of the column's off-diagonal block col_slots[begin + e], against the same panel rows.
//   mode 1 (trailing updates of column k, inner = n): job e is update begin + e of the
//   dst/left/right slot lists.
auto job = [&](int j, int& dst, int& dst_ld, int& rows, int& cols, int& left, int& right, int& ld, int& lower) {
    if (mode == 0) {
        const int rest = n - o - 32;
        const int panel_rows = diag_offset + (o + 32) * n + o;
        cols = rest;
        dst_ld = n;
        ld = n;
        right = panel_rows;
        if (j == 0) {
            dst = panel_rows + 32;
            rows = rest;
            left = panel_rows;
            lower = 1;
        } else {
            const int slot = col_slots.data[begin + j - 1];
            dst = value_offsets.data[slot] + o + 32;
            rows = *wp::address(block_dims, slot, 0);
            left = value_offsets.data[slot] + o;
            lower = 0;
        }
    } else {
        const int dst_slot = dst_slots.data[begin + j];
        const int left_slot = left_slots.data[begin + j];
        const int right_slot = right_slots.data[begin + j];
        dst = value_offsets.data[dst_slot];
        dst_ld = *wp::address(block_dims, dst_slot, 1);
        rows = *wp::address(block_dims, left_slot, 0);
        cols = *wp::address(block_dims, right_slot, 0);
        left = value_offsets.data[left_slot];
        right = value_offsets.data[right_slot];
        ld = n;
        lower = left_slot == right_slot ? 1 : 0;
    }
};
const int inner = mode == 0 ? 32 : n;
const int jobs = mode == 0 ? 1 + (end - begin) : end - begin;
float* v = values.data;
#if defined(__CUDA_ARCH__)
// Groups of 64 threads each update one 64 x 64 tile of one job per round; a thread owns the
// 8 x 8 entries at rows {4 mi + q, 32 + 4 mi + q} and columns {4 mj + s, 32 + 4 mj + s} of its
// group's tile (contiguous float4 reads from shared memory). The inner dimension is staged in
// chunks of 16 through per-group shared buffers (transposed, zero filled past the edges); all
// groups advance chunk by chunk between CTA barriers. Every entry is accumulated in registers
// over the whole inner dimension and written once. Loads are issued in batches (all staging
// loads of a thread, then all destination loads) so their latencies overlap.
constexpr int TILE = 64;
constexpr int CHUNK = 16;
constexpr int STRIDE = 68;
constexpr int MAX_GROUPS = 4;
constexpr int MAX_JOBS = 64;
constexpr int LOADS = TILE * CHUNK / TILE;
__shared__ __align__(16) float vb_batch_a[MAX_GROUPS][CHUNK * STRIDE];
__shared__ __align__(16) float vb_batch_b[MAX_GROUPS][CHUNK * STRIDE];
__shared__ int vb_batch_job[MAX_JOBS][9];
const int groups = wp::min(MAX_GROUPS, threads / TILE);
const int group = lane / TILE;
const int gl = lane - group * TILE;
const int mi = gl >> 3;
const int mj = gl & 7;
const bool member = group < groups;
for (int job_base = 0; job_base < jobs; job_base += MAX_JOBS) {
    const int job_end = wp::min(jobs, job_base + MAX_JOBS);
    __syncthreads();
    for (int j = job_base + lane; j < job_end; j += threads) {
        int* d = vb_batch_job[j - job_base];
        job(j, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
        const int tr = (d[2] + TILE - 1) / TILE;
        const int tc = (d[3] + TILE - 1) / TILE;
        d[8] = (d[2] > 0 && d[3] > 0) ? (d[7] ? tr * (tr + 1) / 2 : tr * tc) : 0;
    }
    __syncthreads();
    int tiles = 0;
    for (int j = 0; j < job_end - job_base; ++j)
        tiles += vb_batch_job[j][8];
    for (int base = 0; base < tiles; base += groups) {
        int t = base + group;
        const bool active = member && t < tiles;
        int dst = 0, dst_ld = 0, rows = 0, cols = 0, left = 0, right = 0, ld = 0, lower = 0, r0 = 0, c0 = 0;
        if (active) {
            for (int j = 0; j < job_end - job_base; ++j) {
                const int* d = vb_batch_job[j];
                if (t < d[8]) {
                    dst = d[0]; dst_ld = d[1]; rows = d[2]; cols = d[3];
                    left = d[4]; right = d[5]; ld = d[6]; lower = d[7];
                    if (lower) {
                        int ti = 0;
                        while (t > ti) {
                            t -= ti + 1;
                            ++ti;
                        }
                        r0 = ti * TILE;
                        c0 = t * TILE;
                    } else {
                        const int tc = (cols + TILE - 1) / TILE;
                        r0 = (t / tc) * TILE;
                        c0 = (t % tc) * TILE;
                    }
                    break;
                }
                t -= d[8];
            }
        }
        float acc[64];
#pragma unroll
        for (int e = 0; e < 64; ++e)
            acc[e] = 0.0f;
        float* sa = vb_batch_a[member ? group : 0];
        float* sb = vb_batch_b[member ? group : 0];
        for (int k0 = 0; k0 < inner; k0 += CHUNK) {
            if (active) {
                float ra[LOADS];
                float rb[LOADS];
#pragma unroll
                for (int u = 0; u < LOADS; ++u) {
                    const int f = gl + u * TILE;
                    const int i = f / CHUNK;
                    const int p = f - i * CHUNK;
                    const bool k_ok = k0 + p < inner;
                    ra[u] = (k_ok && r0 + i < rows) ? v[left + (r0 + i) * ld + k0 + p] : 0.0f;
                    rb[u] = (k_ok && c0 + i < cols) ? v[right + (c0 + i) * ld + k0 + p] : 0.0f;
                }
#pragma unroll
                for (int u = 0; u < LOADS; ++u) {
                    const int f = gl + u * TILE;
                    const int i = f / CHUNK;
                    const int p = f - i * CHUNK;
                    sa[p * STRIDE + i] = ra[u];
                    sb[p * STRIDE + i] = rb[u];
                }
            }
            __syncthreads();
            if (active) {
#pragma unroll 4
                for (int p = 0; p < CHUNK; ++p) {
                    const float4 a0 = *reinterpret_cast<const float4*>(sa + p * STRIDE + 4 * mi);
                    const float4 a1 = *reinterpret_cast<const float4*>(sa + p * STRIDE + 32 + 4 * mi);
                    const float4 b0 = *reinterpret_cast<const float4*>(sb + p * STRIDE + 4 * mj);
                    const float4 b1 = *reinterpret_cast<const float4*>(sb + p * STRIDE + 32 + 4 * mj);
                    const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                    const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                    for (int q = 0; q < 8; ++q)
#pragma unroll
                        for (int s = 0; s < 8; ++s)
                            acc[8 * q + s] = fmaf(av[q], bv[s], acc[8 * q + s]);
                }
            }
            __syncthreads();
        }
        if (active) {
            float old[64];
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const int r = r0 + (q < 4 ? 4 * mi + q : 32 + 4 * mi + q - 4);
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    const int c = c0 + (s < 4 ? 4 * mj + s : 32 + 4 * mj + s - 4);
                    old[8 * q + s] = (r < rows && c < cols && (!lower || c <= r)) ? v[dst + r * dst_ld + c] : 0.0f;
                }
            }
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const int r = r0 + (q < 4 ? 4 * mi + q : 32 + 4 * mi + q - 4);
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    const int c = c0 + (s < 4 ? 4 * mj + s : 32 + 4 * mj + s - 4);
                    if (r < rows && c < cols && (!lower || c <= r))
                        v[dst + r * dst_ld + c] = old[8 * q + s] - acc[8 * q + s];
                }
            }
        }
    }
}
__syncthreads();
#else
// One thread (the CPU runs the cooperative kernel with block_dim = 1): plain dot products.
for (int j = 0; j < jobs; ++j) {
    int dst, dst_ld, rows, cols, left, right, ld, lower;
    job(j, dst, dst_ld, rows, cols, left, right, ld, lower);
    for (int r = 0; r < rows; ++r) {
        const int c_end = lower ? wp::min(cols, r + 1) : cols;
        for (int c = 0; c < c_end; ++c) {
            float sum = 0.0f;
            for (int p = 0; p < inner; ++p)
                sum += v[left + r * ld + p] * v[right + c * ld + p];
            v[dst + r * dst_ld + c] -= sum;
        }
    }
}
#endif
""")
def _vb_update_batch(
    values: wp.array[float],
    mode: int,
    n: int,
    o: int,
    diag_offset: int,
    begin: int,
    end: int,
    col_slots: wp.array[wp.int32],
    dst_slots: wp.array[wp.int32],
    left_slots: wp.array[wp.int32],
    right_slots: wp.array[wp.int32],
    value_offsets: wp.array[wp.int32],
    block_dims: wp.array2d[wp.int32],
    lane: int,
    threads: int,
):
    """Apply a batch of block updates ``D -= A B^T`` of a wide column (all threads of the CTA; ends with a barrier).

    ``mode == _VB_UPDATE_PANEL``: the rank-32 update after the panel at column ``o`` of the
    ``n``-wide column whose diagonal block is at ``diag_offset``, on the trailing columns of the
    diagonal block (lower triangle) and of the column's off-diagonal blocks
    ``col_slots[begin:end]``. ``mode == _VB_UPDATE_COLUMN``: the trailing updates
    ``H_dst -= L_left L_right^T`` (lower triangle when ``left == right``) of the update list
    entries ``[begin, end)`` with inner size ``n``. Destinations of a batch are disjoint and do not
    overlap its operands.

    On CUDA, groups of 64 threads (at most four, so ``threads`` should be a multiple of 64) update
    64 x 64 tiles of all jobs of the batch concurrently, staging the inner dimension through
    shared memory in chunks of 16 and accumulating each entry in registers over the whole inner
    dimension, so each entry is written once. On the CPU (one thread) every entry is a dot
    product in index order.
    """
    ...


@wp.func_native("""
// Rows below the factored 32-wide panel at column o of a wide column: rows o + 32 .. n - 1 of the
// diagonal block (stride n, at diag_offset), then the rows of the off-diagonal blocks
// col_slots[col_begin:col_end] (stride n). Each row takes x := x L_pp^-T on its panel columns by
// forward substitution; one row per thread, the row's panel entries are held in registers.
float* v = values.data;
const int w = wp::min(32, n - o);
const int below = wp::max(n - o - 32, 0);
int total = below;
for (int e = col_begin; e < col_end; ++e)
    total += *wp::address(block_dims, col_slots.data[e], 0);
#if defined(__CUDA_ARCH__)
__shared__ float vb_trsm_l[32 * 33];
for (int f = lane; f < 32 * 32; f += threads) {
    const int c = f >> 5;
    const int p = f & 31;
    vb_trsm_l[c * 33 + p] = (c < w && p <= c) ? v[diag_offset + (o + c) * n + o + p] : 0.0f;
}
__syncthreads();
#define VB_TRSM_L(c, p) vb_trsm_l[(c) * 33 + (p)]
#else
#define VB_TRSM_L(c, p) v[diag_offset + (o + (c)) * n + o + (p)]
#endif
for (int item = lane; item < total; item += threads) {
    int row_offset = 0;
    if (item < below) {
        row_offset = diag_offset + (o + 32 + item) * n + o;
    } else {
        int r = item - below;
        for (int e = col_begin; e < col_end; ++e) {
            const int slot = col_slots.data[e];
            const int block_rows = *wp::address(block_dims, slot, 0);
            if (r < block_rows) {
                row_offset = value_offsets.data[slot] + r * n + o;
                break;
            }
            r -= block_rows;
        }
    }
    float x[32];
#pragma unroll
    for (int c = 0; c < 32; ++c)
        x[c] = c < w ? v[row_offset + c] : 0.0f;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
        if (c < w) {
            float value = x[c];
#pragma unroll
            for (int p = 0; p < c; ++p)
                value = value - x[p] * VB_TRSM_L(c, p);
            x[c] = value / VB_TRSM_L(c, c);
        }
    }
#pragma unroll
    for (int c = 0; c < 32; ++c)
        if (c < w)
            v[row_offset + c] = x[c];
}
#undef VB_TRSM_L
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _vb_panel_trsm(
    values: wp.array[float],
    diag_offset: int,
    n: int,
    o: int,
    col_begin: int,
    col_end: int,
    col_slots: wp.array[wp.int32],
    value_offsets: wp.array[wp.int32],
    block_dims: wp.array2d[wp.int32],
    lane: int,
    threads: int,
):
    """Solve the rows below a factored panel of a wide column, ``x := x L_pp^-T`` (all threads; ends with a barrier).

    ``L_pp`` is the factored ``w x w`` diagonal panel (``w = min(32, n - o)``) at ``(o, o)`` of the
    ``n x n`` diagonal block at ``diag_offset``. The rows are rows ``o + 32 .. n - 1`` of the
    diagonal block followed by all rows of the column's off-diagonal blocks
    ``col_slots[col_begin:col_end]``, restricted to columns ``o .. o + w - 1``. On CUDA the
    panel is staged in shared memory and each thread solves whole rows in registers; on the
    CPU one thread solves every row in turn.
    """
    ...


@wp.func_native("""
// Factor the w x w diagonal panel (w = n - o <= 32) at (o, o) of the n x n block at diag_offset
// in place and return 1, or 0 if a pivot is not positive and finite.
float* v = values.data;
const int w = wp::min(32, n - o);
#if defined(__CUDA_ARCH__)
// The CTA stages the panel in shared memory; warp 0 factors it right-looking (lane i owns row
// i; one column per step, separated by warp barriers) and writes it back.
__shared__ float vb_small_l[32 * 33];
__shared__ int vb_small_ok;
for (int f = lane; f < 32 * 32; f += threads) {
    const int i = f >> 5;
    const int p = f & 31;
    vb_small_l[i * 33 + p] = (i < w && p <= i) ? v[diag_offset + (o + i) * n + o + p] : 0.0f;
}
__syncthreads();
if (lane < 32) {
    int ok = 1;
    for (int j = 0; j < w; ++j) {
        const float pivot = vb_small_l[j * 33 + j];
        if (!(pivot > 0.0f && wp::isfinite(pivot)))
            ok = 0;
        const float d = wp::sqrt(pivot);
        __syncwarp();
        if (lane == j)
            vb_small_l[j * 33 + j] = d;
        else if (lane > j && lane < w)
            vb_small_l[lane * 33 + j] = vb_small_l[lane * 33 + j] / d;
        __syncwarp();
        if (lane > j && lane < w) {
            const float l_ij = vb_small_l[lane * 33 + j];
            for (int p = j + 1; p <= lane; ++p)
                vb_small_l[lane * 33 + p] = vb_small_l[lane * 33 + p] - l_ij * vb_small_l[p * 33 + j];
        }
        __syncwarp();
    }
    if (lane < w)
        for (int p = 0; p <= lane; ++p)
            v[diag_offset + (o + lane) * n + o + p] = vb_small_l[lane * 33 + p];
    if (lane == 0)
        vb_small_ok = ok;
}
__syncthreads();
return vb_small_ok;
#else
int ok = 1;
for (int j = 0; j < w && ok; ++j) {
    float pivot = v[diag_offset + (o + j) * n + o + j];
    for (int p = 0; p < j; ++p)
        pivot -= v[diag_offset + (o + j) * n + o + p] * v[diag_offset + (o + j) * n + o + p];
    if (!(pivot > 0.0f && wp::isfinite(pivot))) {
        ok = 0;
        break;
    }
    const float d = wp::sqrt(pivot);
    v[diag_offset + (o + j) * n + o + j] = d;
    for (int i = j + 1; i < w; ++i) {
        float value = v[diag_offset + (o + i) * n + o + j];
        for (int p = 0; p < j; ++p)
            value -= v[diag_offset + (o + i) * n + o + p] * v[diag_offset + (o + j) * n + o + p];
        v[diag_offset + (o + i) * n + o + j] = value / d;
    }
}
return ok;
#endif
""")
def _vb_panel_cholesky_small(values: wp.array[float], diag_offset: int, n: int, o: int, lane: int, threads: int) -> int:
    """Factor a narrow last panel (``n - o <= 32``) of a wide diagonal block in place (all threads; ends with a barrier).

    Only the lower triangle of the panel is read and written. Every pivot is checked; returns 1,
    or 0 if one is not positive and finite. On CUDA warp 0 factors the panel staged in shared
    memory (cheaper than the tile Cholesky for narrow panels); on the CPU one thread factors it
    left-looking in place. Needs at least 32 threads on CUDA.
    """
    ...


@wp.func
def _vb_wide_panel(
    values: wp.array[float],
    diag_offset: int,
    n: int,
    o: int,
    col_begin: int,
    col_end: int,
    articulation_block_col_slots: wp.array[wp.int32],
    articulation_block_value_offsets: wp.array[wp.int32],
    articulation_block_dims: wp.array2d[wp.int32],
    lane: int,
    threads: int,
) -> int:
    """Factor the 32-wide panel at column ``o`` of a wide column and solve the rows below it.

    A panel of at most :data:`_VB_SMALL_PANEL` columns (the last panel of a block) is factored by
    :func:`_vb_panel_cholesky_small`; a wider one in shared memory with ``tile_cholesky_inplace``
    (a partial panel is padded to the identity), checking every valid pivot since
    ``tile_cholesky_inplace`` reports no failure, and stored. Then :func:`_vb_panel_trsm` solves
    every row below the panel in the column. Returns 1, or 0 on a bad pivot (rows are then not
    solved).
    """
    ok = int(1)
    if n - o <= _VB_SMALL_PANEL:
        ok = _vb_panel_cholesky_small(values, diag_offset, n, o, lane, threads)
    else:
        D = _vb_view(values, diag_offset, n, n)
        akk = wp.tile_load(D, shape=(_VB_TILE, _VB_TILE), offset=(o, o), storage="shared", bounds_check=True)
        lkk = wp.tile_diag_add(akk, wp.tile_map(_vb_pad_entry, wp.tile_arange(_VB_TILE, dtype=float), float(n - o)))
        wp.tile_cholesky_inplace(lkk)
        for i in range(wp.min(_VB_TILE, n - o)):
            if not _vb_pivot_ok(lkk[i, i]):
                ok = 0
        wp.tile_store(D, lkk, offset=(o, o), bounds_check=True)
        # Required although the tests do not detect its removal (final2 mutation M2): tile_store
        # may take Warp's vectorized path, where thread t stores entries 4t..4t+3, while
        # _vb_panel_trsm has thread f % threads load entry f; the race needs warps to drift
        # apart by more than the store latency, which no test provokes reliably.
        _block_sync()
    if ok == 1:
        _vb_panel_trsm(
            values,
            diag_offset,
            n,
            o,
            col_begin,
            col_end,
            articulation_block_col_slots,
            articulation_block_value_offsets,
            articulation_block_dims,
            lane,
            threads,
        )
    return ok


@wp.func
def _vb_wide_factor_column(
    values: wp.array[float],
    diag_offset: int,
    n: int,
    col_begin: int,
    col_end: int,
    articulation_block_col_slots: wp.array[wp.int32],
    articulation_block_value_offsets: wp.array[wp.int32],
    articulation_block_dims: wp.array2d[wp.int32],
    lane: int,
    threads: int,
) -> int:
    """Factor the diagonal block of a wide column and form its column blocks ``L_ik = H_ik L_kk^-T``.

    Right-looking over 32-wide panels of the combined column (the ``n x n`` diagonal block and the
    column's off-diagonal blocks below it): :func:`_vb_wide_panel`, then the rank-32 update of the
    trailing columns of the diagonal block (lower triangle) and of every column block in one
    :func:`_vb_update_batch`. Returns 1, or 0 at the first bad pivot. The strict upper triangle
    of the diagonal block is left unspecified.
    """
    panels = (n + _VB_TILE - 1) // _VB_TILE
    for k in range(panels):
        o = k * _VB_TILE
        if (
            _vb_wide_panel(
                values,
                diag_offset,
                n,
                o,
                col_begin,
                col_end,
                articulation_block_col_slots,
                articulation_block_value_offsets,
                articulation_block_dims,
                lane,
                threads,
            )
            == 0
        ):
            return 0
        if n - o - _VB_TILE > 0:
            _vb_update_batch(
                values,
                _VB_UPDATE_PANEL,
                n,
                o,
                diag_offset,
                col_begin,
                col_end,
                articulation_block_col_slots,
                articulation_block_col_slots,
                articulation_block_col_slots,
                articulation_block_col_slots,
                articulation_block_value_offsets,
                articulation_block_dims,
                lane,
                threads,
            )
    return 1


@wp.func
def _vb_tile_lower_solve_vec(L: wp.array2d[float], y: wp.array2d[float]):
    """Solve ``L y' = y`` in place panel by panel (``y`` is an ``n x 1`` view)."""
    n = L.shape[0]
    panels = (n + _VB_TILE - 1) // _VB_TILE
    for k in range(panels):
        o = k * _VB_TILE
        yk = wp.tile_load(y, shape=(_VB_TILE, 1), offset=(o, 0), storage="shared", bounds_check=True)
        for j in range(k):
            yj = wp.tile_load(y, shape=(_VB_TILE, 1), offset=(j * _VB_TILE, 0), storage="shared", bounds_check=True)
            lkj = wp.tile_load(
                L, shape=(_VB_TILE, _VB_TILE), offset=(o, j * _VB_TILE), storage="shared", bounds_check=True
            )
            wp.tile_matmul(lkj, yj, yk, alpha=-1.0)
        lkk = wp.tile_diag_add(
            wp.tile_load(L, shape=(_VB_TILE, _VB_TILE), offset=(o, o), storage="shared", bounds_check=True),
            wp.tile_map(_vb_pad_entry, wp.tile_arange(_VB_TILE, dtype=float), float(n - o)),
        )
        wp.tile_lower_solve_inplace(lkk, yk)
        wp.tile_store(y, yk, offset=(o, 0), bounds_check=True)
        # Required although the tests do not detect its removal (final2 mutation M2b): the next
        # panel loads this part of y into tiles whose thread mapping is Warp-internal, and its
        # loads reuse the shared buffers this panel's store still reads; not provably redundant.
        _block_sync()


@wp.func
def _vb_tile_upper_solve_vec(L: wp.array2d[float], y: wp.array2d[float]):
    """Solve ``L^T y' = y`` in place panel by panel (``y`` is an ``n x 1`` view)."""
    n = L.shape[0]
    panels = (n + _VB_TILE - 1) // _VB_TILE
    for s in range(panels):
        k = panels - 1 - s
        o = k * _VB_TILE
        yk = wp.tile_load(y, shape=(_VB_TILE, 1), offset=(o, 0), storage="shared", bounds_check=True)
        for j in range(k + 1, panels):
            yj = wp.tile_load(y, shape=(_VB_TILE, 1), offset=(j * _VB_TILE, 0), storage="shared", bounds_check=True)
            ljk = wp.tile_load(
                L, shape=(_VB_TILE, _VB_TILE), offset=(j * _VB_TILE, o), storage="shared", bounds_check=True
            )
            wp.tile_matmul(wp.tile_transpose(ljk), yj, yk, alpha=-1.0)
        lkk = wp.tile_diag_add(
            wp.tile_load(L, shape=(_VB_TILE, _VB_TILE), offset=(o, o), storage="shared", bounds_check=True),
            wp.tile_map(_vb_pad_entry, wp.tile_arange(_VB_TILE, dtype=float), float(n - o)),
        )
        wp.tile_upper_solve_inplace(wp.tile_transpose(lkk), yk)
        wp.tile_store(y, yk, offset=(o, 0), bounds_check=True)
        _block_sync()


@wp.func
def _vb_first_lane_item(lane: int, base: int, threads: int) -> int:
    """First item index ``i >= 0`` of a block whose items are numbered from ``base`` with ``(base + i) % threads == lane``."""
    return ((lane - base) % threads + threads) % threads


@wp.kernel(enable_backward=False, module="unique")
def solve_articulation_variable_cooperative(
    rigid_articulation_count: int,
    frame_regularization: float,
    articulation_body_offsets: wp.array[wp.int32],
    articulation_row_dims: wp.array[wp.int32],
    articulation_row_scalar_offsets: wp.array[wp.int32],
    articulation_block_value_offsets: wp.array[wp.int32],
    articulation_block_dims: wp.array2d[wp.int32],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    articulation_block_col_offsets: wp.array[wp.int32],
    articulation_block_col_rows: wp.array[wp.int32],
    articulation_block_col_slots: wp.array[wp.int32],
    articulation_factor_update_offsets: wp.array[wp.int32],
    articulation_factor_update_dst_slots: wp.array[wp.int32],
    articulation_factor_update_left_slots: wp.array[wp.int32],
    articulation_factor_update_right_slots: wp.array[wp.int32],
    articulation_diag_slots: wp.array[wp.int32],
    rhs: wp.array[float],
    values: wp.array[float],
    diag_backup: wp.array[float],
    factor_status: wp.array[wp.int32],
    failure_count: wp.array[wp.int32],
    delta: wp.array[float],
):
    """Factor and solve one mixed articulation per CTA (right-looking block Cholesky).

    Launch with :func:`warp.launch_tiled`, ``dim = mixed_articulation_count`` and
    ``block_dim = VB_COOPERATIVE_BLOCK_DIM``. Same arguments, storage, elimination order,
    regularization and failure handling as :func:`solve_articulation_variable_serial`; see the
    module docstring for the lane and wide paths.
    """
    tid, lane = wp.tid()
    threads = wp.block_dim()
    articulation = rigid_articulation_count + tid
    body_start = articulation_body_offsets[articulation]
    body_end = articulation_body_offsets[articulation + 1]
    begin = articulation_row_scalar_offsets[body_start]
    end = articulation_row_scalar_offsets[body_end]

    # ``factor_status[tid]`` doubles as the CTA-uniform failure flag: it is written by single
    # lanes and read by every lane after a barrier.
    if lane == 0:
        factor_status[tid] = VB_FACTOR_OK

    # Regularize the frame rows and keep the diagonal for the Jacobi fallback; one lane per
    # component of the articulation.
    for i in range(begin + lane, end, threads):
        row = body_start
        while articulation_row_scalar_offsets[row + 1] <= i:
            row += 1
        n = articulation_row_dims[row]
        diag_offset = articulation_block_value_offsets[articulation_diag_slots[row]]
        k = i - articulation_row_scalar_offsets[row]
        if k < 6:
            values[diag_offset + k * n + k] = values[diag_offset + k * n + k] + frame_regularization
        diag_backup[i] = values[diag_offset + k * n + k]
    _block_sync()

    for row_k in range(body_start, body_end):
        n_k = articulation_row_dims[row_k]
        diag_offset = articulation_block_value_offsets[articulation_diag_slots[row_k]]
        col_begin = articulation_block_col_offsets[row_k]
        col_end = articulation_block_col_offsets[row_k + 1]
        if n_k <= _VB_LANE_MAX_DIM:
            # Every lane factors the 6 x 6 block in registers (identical operations, so identical
            # bits); lane 0 stores it once every lane has read the block, and each lane solves its
            # rows of the column blocks with it, L_ik = H_ik L_kk^-T, one row per lane numbered
            # across all blocks of the column.
            Hkk = _vb_load66(values, diag_offset)
            _block_sync()
            Lkk, ok = _vb_cholesky66_registers(Hkk)
            if lane == 0:
                for i in range(6):
                    for j in range(6):
                        values[diag_offset + i * 6 + j] = Lkk[i, j]
                if ok == 0:
                    factor_status[tid] = VB_FACTOR_INDEFINITE
            if ok == 1:
                base = int(0)
                for entry in range(col_begin, col_end):
                    slot = articulation_block_col_slots[entry]
                    rows = articulation_block_dims[slot, 0]
                    offset = articulation_block_value_offsets[slot]
                    for r in range(_vb_first_lane_item(lane, base, threads), rows, threads):
                        row_offset = offset + r * 6
                        x = vec6f(
                            values[row_offset],
                            values[row_offset + 1],
                            values[row_offset + 2],
                            values[row_offset + 3],
                            values[row_offset + 4],
                            values[row_offset + 5],
                        )
                        for c in range(6):
                            value = x[c]
                            for p in range(6):
                                if p < c:
                                    value = value - x[p] * Lkk[c, p]
                            x[c] = value / Lkk[c, c]
                        for c in range(6):
                            values[row_offset + c] = x[c]
                    base += rows
        else:
            ok = _vb_wide_factor_column(
                values,
                diag_offset,
                n_k,
                articulation_block_col_offsets[row_k],
                articulation_block_col_offsets[row_k + 1],
                articulation_block_col_slots,
                articulation_block_value_offsets,
                articulation_block_dims,
                lane,
                threads,
            )
            if ok == 0 and lane == 0:
                factor_status[tid] = VB_FACTOR_INDEFINITE
            for e in range(lane, n_k * n_k, threads):
                r = e // n_k
                if e - r * n_k > r:
                    values[diag_offset + e] = 0.0
        _block_sync()
        if factor_status[tid] != VB_FACTOR_OK:
            break

        # Trailing updates H_ab -= L_ak L_bk^T; destinations within a column are distinct.
        update_begin = articulation_factor_update_offsets[row_k]
        update_end = articulation_factor_update_offsets[row_k + 1]
        if n_k <= _VB_LANE_MAX_DIM:
            # One lane per destination entry, numbered across all updates of the column.
            base = int(0)
            for entry in range(update_begin, update_end):
                dst = articulation_factor_update_dst_slots[entry]
                left = articulation_factor_update_left_slots[entry]
                right = articulation_factor_update_right_slots[entry]
                left_rows = articulation_block_dims[left, 0]
                right_rows = articulation_block_dims[right, 0]
                dst_offset = articulation_block_value_offsets[dst]
                left_offset = articulation_block_value_offsets[left]
                right_offset = articulation_block_value_offsets[right]
                count = left_rows * right_rows
                for e in range(_vb_first_lane_item(lane, base, threads), count, threads):
                    r = e // right_rows
                    c = e - r * right_rows
                    if left != right or c <= r:
                        accum = float(0.0)
                        for p in range(n_k):
                            accum = accum + values[left_offset + r * n_k + p] * values[right_offset + c * n_k + p]
                        values[dst_offset + e] = values[dst_offset + e] - accum
                base += count
        elif update_end > update_begin:
            _vb_update_batch(
                values,
                _VB_UPDATE_COLUMN,
                n_k,
                0,
                diag_offset,
                update_begin,
                update_end,
                articulation_block_col_slots,
                articulation_factor_update_dst_slots,
                articulation_factor_update_left_slots,
                articulation_factor_update_right_slots,
                articulation_block_value_offsets,
                articulation_block_dims,
                lane,
                threads,
            )
        _block_sync()

    if factor_status[tid] == VB_FACTOR_OK:
        for i in range(begin + lane, end, threads):
            delta[i] = rhs[i]
        _block_sync()
        # Forward substitution with the stored lower block rows; one lane per component.
        for row in range(body_start, body_end):
            n = articulation_row_dims[row]
            vec_offset = articulation_row_scalar_offsets[row]
            diag_slot = articulation_diag_slots[row]
            slot_begin = articulation_block_row_offsets[row]
            slot_end = articulation_block_row_offsets[row + 1]
            for r in range(lane, n, threads):
                for slot in range(slot_begin, slot_end):
                    if slot != diag_slot:
                        col_row = body_start + articulation_block_cols[slot]
                        cols = articulation_row_dims[col_row]
                        x_offset = articulation_row_scalar_offsets[col_row]
                        offset = articulation_block_value_offsets[slot] + r * cols
                        accum = float(0.0)
                        for c in range(cols):
                            accum = accum + values[offset + c] * delta[x_offset + c]
                        delta[vec_offset + r] = delta[vec_offset + r] - accum
            _block_sync()
            diag_offset = articulation_block_value_offsets[diag_slot]
            if n <= _VB_LANE_MAX_DIM:
                if lane == 0:
                    _vb_lower_solve66_registers(values, diag_offset, delta, vec_offset)
            else:
                _vb_tile_lower_solve_vec(_vb_view(values, diag_offset, n, n), _vb_view(delta, vec_offset, n, 1))
            _block_sync()
        # Back substitution with the column lists (L^T); one lane per component.
        for rr in range(body_end - body_start):
            row = body_end - 1 - rr
            n = articulation_row_dims[row]
            vec_offset = articulation_row_scalar_offsets[row]
            col_begin = articulation_block_col_offsets[row]
            col_end = articulation_block_col_offsets[row + 1]
            for c in range(lane, n, threads):
                for entry in range(col_begin, col_end):
                    slot = articulation_block_col_slots[entry]
                    below = body_start + articulation_block_col_rows[entry]
                    rows = articulation_row_dims[below]
                    x_offset = articulation_row_scalar_offsets[below]
                    offset = articulation_block_value_offsets[slot]
                    accum = float(0.0)
                    for r in range(rows):
                        accum = accum + values[offset + r * n + c] * delta[x_offset + r]
                    delta[vec_offset + c] = delta[vec_offset + c] - accum
            _block_sync()
            diag_offset = articulation_block_value_offsets[articulation_diag_slots[row]]
            if n <= _VB_LANE_MAX_DIM:
                if lane == 0:
                    _vb_upper_solve66_registers(values, diag_offset, delta, vec_offset)
            else:
                _vb_tile_upper_solve_vec(_vb_view(values, diag_offset, n, n), _vb_view(delta, vec_offset, n, 1))
            _block_sync()
        for i in range(begin + lane, end, threads):
            if not wp.isfinite(delta[i]):
                factor_status[tid] = VB_SOLVE_NONFINITE
        _block_sync()

    if factor_status[tid] != VB_FACTOR_OK:
        if lane == 0:
            wp.atomic_add(failure_count, 0, 1)
        for i in range(begin + lane, end, threads):
            d = diag_backup[i]
            b = rhs[i]
            step = float(0.0)
            if _vb_pivot_ok(d) and wp.isfinite(b):
                step = b / d
            delta[i] = step


@wp.kernel
def apply_articulation_variable_delta(
    dt: float,
    update_relaxation: float,
    rigid_body_row_count: int,
    articulation_bodies: wp.array[wp.int32],
    articulation_row_dims: wp.array[wp.int32],
    articulation_row_scalar_offsets: wp.array[wp.int32],
    delta: wp.array[float],
    body_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_inertia_q: wp.array[wp.transform],
    use_displacement: int,
    body_elastic_index: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q_prev: wp.array[float],
    body_disp: wp.array[wp.vec3],
    body_disp_ang: wp.array[wp.vec3],
    body_q_new: wp.array[wp.transform],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    """Apply the solved update of one mixed-articulation row per thread.

    Launch with ``dim = articulation_body_count - rigid_body_row_count``. The frame takes the
    displacement-form (or absolute) update of the 6 x 6 kernels; an elastic body's modes take
    ``q += update_relaxation * dq`` and ``qd = (q - q_prev) / dt``, the local block solve's
    convention. The relaxation is applied once to each component.
    """
    row = rigid_body_row_count + wp.tid()
    offset = articulation_row_scalar_offsets[row]
    frame = vec6f(
        delta[offset], delta[offset + 1], delta[offset + 2], delta[offset + 3], delta[offset + 4], delta[offset + 5]
    )
    _apply_sparse_delta_value_to_body(
        row,
        articulation_bodies,
        body_q,
        body_inv_mass,
        body_com,
        update_relaxation,
        frame,
        body_inertia_q,
        body_disp,
        body_disp_ang,
        use_displacement,
        body_q_new,
    )
    mode_count = articulation_row_dims[row] - 6
    if mode_count > 0:
        owner = elastic_joint[body_elastic_index[articulation_bodies[row]]]
        q_start = joint_q_start[owner] + 7
        qd_start = joint_qd_start[owner] + 6
        inv_dt = 1.0 / dt
        for mode in range(mode_count):
            q_new = joint_q[q_start + mode] + update_relaxation * delta[offset + 6 + mode]
            joint_q[q_start + mode] = q_new
            joint_qd[qd_start + mode] = (q_new - joint_q_prev[q_start + mode]) * inv_dt


_cooperative_shared_memory: dict[str, int] = {}


# The two helpers below read Warp internals that have no public equivalent: the metadata of a
# loaded module, the kernel hooks and the runtime's query for a kernel's static shared memory.
# They were verified against Warp 1.18.0; when an internal is missing they return ``None`` and
# cooperative_shared_memory_bytes() falls back to the declared sizes.


def _loaded_dynamic_shared_memory(module_exec) -> int | None:
    """Return the largest dynamic shared memory [bytes] of the tile functions of a loaded module, or ``None``.

    Reads the ``*_smem_bytes`` entries of ``module_exec.meta`` (Warp internal, verified with Warp 1.18.0).
    """
    try:
        items = list(module_exec.meta.items())
        return max((int(value) for key, value in items if key.endswith("_smem_bytes")), default=0)
    except (AttributeError, TypeError, ValueError):
        return None


def _loaded_static_shared_memory(module_exec, kernel, device) -> int | None:
    """Return the static (``__shared__``) memory [bytes] of a kernel loaded on a CUDA device, or ``None``.

    Uses ``module_exec.get_kernel_hooks`` and the runtime's ``wp_cuda_get_kernel_static_shared_memory``
    (Warp internals, verified with Warp 1.18.0).
    """
    try:
        hooks = module_exec.get_kernel_hooks(kernel)
        static = int(
            wp._src.context.runtime.core.wp_cuda_get_kernel_static_shared_memory(device.context, hooks.forward)
        )
    except (AttributeError, TypeError, ValueError):
        return None
    return static if static >= 0 else None


def cooperative_shared_memory_bytes(device=None) -> int:
    """Load :data:`solve_articulation_variable_cooperative` on a CUDA ``device`` and return its shared memory per CTA.

    The sum of the dynamic shared memory of the tile operations (from the module metadata) and the
    static shared memory of the native helpers (queried from the loaded kernel). Without the static
    query the sizes declared in the helpers (``_VB_STATIC_SHARED_MEMORY_ESTIMATE``) are used, and
    without the module metadata the design budget :data:`VB_COOPERATIVE_SHARED_MEMORY_BUDGET`;
    either fallback is logged at debug level.

    Raises:
        ValueError: If ``device`` is not a CUDA device.
        RuntimeError: If the requirement exceeds the device's shared memory per block.
    """
    device = wp.get_device(device)
    if not device.is_cuda:
        raise ValueError(f"cooperative_shared_memory_bytes needs a CUDA device, got {device.alias}")
    cached = _cooperative_shared_memory.get(device.alias)
    if cached is not None:
        return cached
    kernel = solve_articulation_variable_cooperative
    module_exec = kernel.module.load(device, block_dim=VB_COOPERATIVE_BLOCK_DIM)
    dynamic = _loaded_dynamic_shared_memory(module_exec)
    if dynamic is None:
        _logger.debug(
            "Warp module metadata is unavailable; assuming the design budget of %d bytes as the shared memory "
            "of solve_articulation_variable_cooperative on %s.",
            VB_COOPERATIVE_SHARED_MEMORY_BUDGET,
            device.alias,
        )
        required = VB_COOPERATIVE_SHARED_MEMORY_BUDGET
    else:
        static = _loaded_static_shared_memory(module_exec, kernel, device)
        if static is None:
            _logger.debug(
                "The static shared memory query is unavailable; using the %d bytes declared by the native helpers "
                "of solve_articulation_variable_cooperative on %s.",
                _VB_STATIC_SHARED_MEMORY_ESTIMATE,
                device.alias,
            )
            static = _VB_STATIC_SHARED_MEMORY_ESTIMATE
        required = dynamic + static
    limit = device.max_shared_memory_per_block
    if required > limit:
        raise RuntimeError(
            f"solve_articulation_variable_cooperative needs {required} bytes of shared memory per CTA, "
            f"more than the {limit} bytes available on {device.alias}"
        )
    _cooperative_shared_memory[device.alias] = required
    return required


def _solve_inputs(layout, frame_regularization: float, rhs: wp.array) -> list:
    return [
        layout.rigid_articulation_count,
        frame_regularization,
        layout.articulation_body_offsets,
        layout.articulation_row_dims,
        layout.articulation_row_scalar_offsets,
        layout.articulation_block_value_offsets,
        layout.articulation_block_dims,
        layout.articulation_block_row_offsets,
        layout.articulation_block_cols,
        layout.articulation_block_col_offsets,
        layout.articulation_block_col_rows,
        layout.articulation_block_col_slots,
        layout.articulation_factor_update_offsets,
        layout.articulation_factor_update_dst_slots,
        layout.articulation_factor_update_left_slots,
        layout.articulation_factor_update_right_slots,
        layout.articulation_diag_slots,
        rhs,
    ]


def launch_solve_articulation_variable_serial(
    layout,
    frame_regularization: float,
    rhs: wp.array,
    values: wp.array,
    diag_backup: wp.array,
    factor_status: wp.array,
    failure_count: wp.array,
    delta: wp.array,
    device=None,
) -> None:
    """Launch :func:`solve_articulation_variable_serial` on any device.

    Same arguments as :func:`launch_solve_articulation_variable`.
    """
    if layout.mixed_articulation_count == 0:
        return
    wp.launch(
        solve_articulation_variable_serial,
        dim=layout.mixed_articulation_count,
        inputs=_solve_inputs(layout, frame_regularization, rhs),
        outputs=[values, diag_backup, factor_status, failure_count, delta],
        device=device if device is not None else values.device,
    )


def launch_solve_articulation_variable_cooperative(
    layout,
    frame_regularization: float,
    rhs: wp.array,
    values: wp.array,
    diag_backup: wp.array,
    factor_status: wp.array,
    failure_count: wp.array,
    delta: wp.array,
    device=None,
) -> None:
    """Launch :data:`solve_articulation_variable_cooperative`, one CTA per mixed articulation.

    Uses ``block_dim = VB_COOPERATIVE_BLOCK_DIM`` on CUDA (the first call per device checks the
    shared memory, see :func:`cooperative_shared_memory_bytes`) and ``block_dim = 1`` on the
    CPU, where the tile operations and the native helpers run serially (a debugging path). Same
    arguments as :func:`launch_solve_articulation_variable`.
    """
    if layout.mixed_articulation_count == 0:
        return
    device = wp.get_device(device if device is not None else values.device)
    block_dim = 1
    if device.is_cuda:
        if VB_COOPERATIVE_BLOCK_DIM < 64 or VB_COOPERATIVE_BLOCK_DIM % 64 != 0:
            raise ValueError(
                f"VB_COOPERATIVE_BLOCK_DIM must be a positive multiple of 64, got {VB_COOPERATIVE_BLOCK_DIM}"
            )
        cooperative_shared_memory_bytes(device)
        block_dim = VB_COOPERATIVE_BLOCK_DIM
    wp.launch_tiled(
        solve_articulation_variable_cooperative,
        dim=[layout.mixed_articulation_count],
        inputs=_solve_inputs(layout, frame_regularization, rhs),
        outputs=[values, diag_backup, factor_status, failure_count, delta],
        device=device,
        block_dim=block_dim,
    )


def launch_solve_articulation_variable(
    layout,
    frame_regularization: float,
    rhs: wp.array,
    values: wp.array,
    diag_backup: wp.array,
    factor_status: wp.array,
    failure_count: wp.array,
    delta: wp.array,
    device=None,
) -> None:
    """Factor and solve every mixed articulation of ``layout`` (no-op without mixed articulations).

    Launches the cooperative kernel (one CTA per articulation) on CUDA devices and the serial
    kernel (one thread per articulation) on the CPU. Neither allocates memory nor synchronizes
    with the host, so the call can be captured in a CUDA graph once the kernel is loaded (call
    :func:`cooperative_shared_memory_bytes` or launch once before capturing).

    Args:
        layout: Layout built with ``elastic_solve="coupled"``.
        frame_regularization: Value added to each frame diagonal entry of mixed rows.
        rhs: Right-hand side, shape [articulation_scalar_count], read only.
        values: Block values, shape [articulation_value_count], factored in place.
        diag_backup: Scratch for the regularized diagonal, shape [articulation_scalar_count].
        factor_status: Status per mixed articulation, shape [mixed_articulation_count].
        failure_count: ``int32`` counter, shape [1]; incremented once per failed articulation.
        delta: Solved update, shape [articulation_scalar_count]; only mixed rows are written.
        device: Launch device; defaults to the device of ``values``.
    """
    device = wp.get_device(device if device is not None else values.device)
    if device.is_cuda:
        launcher = launch_solve_articulation_variable_cooperative
    else:
        launcher = launch_solve_articulation_variable_serial
    launcher(layout, frame_regularization, rhs, values, diag_backup, factor_status, failure_count, delta, device)
