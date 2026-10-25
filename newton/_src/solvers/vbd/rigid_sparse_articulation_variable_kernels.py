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
  :data:`VB_COOPERATIVE_BLOCK_DIM` (128) threads per mixed articulation, the same elimination
  order, column and update lists, regularization and failure policy. Column ``k`` takes the
  *lane path* when ``n_k = 6``: lane 0 factors ``L_kk``; the column solves give one block row
  and the trailing updates one destination entry to each lane, numbered across all blocks of
  the column. In the substitutions each lane owns one vector component of the block row for the
  off-diagonal products, and a 6 x 6 diagonal solve runs on lane 0. The lane path
  repeats the serial arithmetic, so all-6 systems agree bitwise with the serial kernel. Wider
  columns take the *tile path*: fixed 32 x 32 shared-memory panels of runtime-shaped views into
  ``values`` (:func:`_vb_view`), zero filled past the block edge, with the padded diagonal of a
  partial panel set to the identity each time the panel is loaded; panel Cholesky
  (``tile_cholesky_inplace``), panel triangular solves, and ``tile_matmul`` updates. Bodies
  are not padded to the largest modal count; only the last partial panel of a block is. Every
  valid diagonal entry of every factored panel is checked, since ``tile_cholesky_inplace``
  reports no failure. Control flow is uniform across the CTA: branches depend on block sizes or
  on ``factor_status``, which single lanes write and all lanes read after a barrier. An explicit
  ``__syncthreads`` follows every phase that stores to global memory (``tile_store`` issues
  none). The kernel neither allocates nor synchronizes with the host, so it can be captured in
  a CUDA graph; the tile path is accurate to float32 rounding, not bitwise equal to the serial
  kernel. On the CPU the cooperative kernel runs with ``block_dim = 1`` (for testing).

Shared memory of the cooperative kernel: one 32 x 32 ``float32`` tile is 4 KiB, and every
tile expression in a function frame owns its own buffer. Each tile phase is therefore a
separate ``wp.func``, so a CTA needs the largest phase frame instead of the sum, and every loop
around a tile operation has a runtime trip count, so loop unrolling cannot duplicate tiles
(compiling with ``max_unroll = 1`` gives the same size; the default unrolling keeps the lane
path fast). The panel Cholesky frame holds seven panels plus 128 bytes: 28 800 bytes per CTA
with Warp 1.18, under the design budget :data:`VB_COOPERATIVE_SHARED_MEMORY_BUDGET` (32 KiB;
three CTAs fit in 100 KiB per SM). :func:`cooperative_shared_memory_bytes` reads the
requirement from the compiled module and raises if it exceeds the device limit; the CUDA
launcher calls it once per device. The cooperative kernel has its own Warp module
(``module="unique"``), which keeps its libmathdx build (tens of seconds on a cold kernel
cache) out of the module of the light kernels; it uses no float atomics, so it does not
depend on the deterministic mode.
"""

import warp as wp

from newton._src.solvers.vbd.rigid_sparse_articulation_kernels import (
    _apply_sparse_delta_value_to_body,
    _block_sync,
    vec6f,
)

wp.set_module_options({"enable_backward": False})

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

VB_COOPERATIVE_BLOCK_DIM = 128
"""Threads per CTA of :data:`solve_articulation_variable_cooperative` on CUDA devices."""

VB_COOPERATIVE_SHARED_MEMORY_BUDGET = 32 * 1024
"""Design budget in bytes for the shared memory of one cooperative CTA (see the module docstring)."""

_VB_TILE = wp.constant(32)
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


@wp.func
def _vb_tile_potrf(A: wp.array2d[float]) -> int:
    """Right-looking panel Cholesky of the square view ``A`` in place (all threads of the CTA).

    Every valid diagonal entry of each factored panel is checked, since ``tile_cholesky_inplace``
    reports no failure. Returns 1 if every pivot is positive and finite, else 0 (the factor is then
    incomplete). The strict upper triangle of ``A`` is left unspecified.
    """
    n = A.shape[0]
    panels = (n + _VB_TILE - 1) // _VB_TILE
    ok = int(1)
    for k in range(panels):
        o = k * _VB_TILE
        akk = wp.tile_load(A, shape=(_VB_TILE, _VB_TILE), offset=(o, o), storage="shared", bounds_check=True)
        lkk = wp.tile_diag_add(akk, wp.tile_map(_vb_pad_entry, wp.tile_arange(_VB_TILE, dtype=float), float(n - o)))
        wp.tile_cholesky_inplace(lkk)
        for i in range(wp.min(_VB_TILE, n - o)):
            if not _vb_pivot_ok(lkk[i, i]):
                ok = 0
        wp.tile_store(A, lkk, offset=(o, o), bounds_check=True)
        if ok == 0:
            break
        for i in range(k + 1, panels):
            aik = wp.tile_load(
                A, shape=(_VB_TILE, _VB_TILE), offset=(i * _VB_TILE, o), storage="shared", bounds_check=True
            )
            x = wp.tile_lower_solve(lkk, wp.tile_transpose(aik))
            wp.tile_store(A, wp.tile_transpose(x), offset=(i * _VB_TILE, o), bounds_check=True)
        _block_sync()
        for j in range(k + 1, panels):
            for i in range(j, panels):
                cij = wp.tile_load(
                    A,
                    shape=(_VB_TILE, _VB_TILE),
                    offset=(i * _VB_TILE, j * _VB_TILE),
                    storage="shared",
                    bounds_check=True,
                )
                li = wp.tile_load(
                    A, shape=(_VB_TILE, _VB_TILE), offset=(i * _VB_TILE, o), storage="shared", bounds_check=True
                )
                lj = wp.tile_load(
                    A, shape=(_VB_TILE, _VB_TILE), offset=(j * _VB_TILE, o), storage="shared", bounds_check=True
                )
                wp.tile_matmul(li, wp.tile_transpose(lj), cij, alpha=-1.0)
                wp.tile_store(A, cij, offset=(i * _VB_TILE, j * _VB_TILE), bounds_check=True)
        _block_sync()
    _block_sync()
    return ok


@wp.func
def _vb_tile_right_solve_lower_transpose(L: wp.array2d[float], B: wp.array2d[float]):
    """``B := B L^-T`` panel by panel (``L`` factored ``n_k x n_k``, ``B`` is ``n_i x n_k``)."""
    nk = L.shape[0]
    col_panels = (nk + _VB_TILE - 1) // _VB_TILE
    row_panels = (B.shape[0] + _VB_TILE - 1) // _VB_TILE
    for c in range(col_panels):
        o = c * _VB_TILE
        lcc = wp.tile_diag_add(
            wp.tile_load(L, shape=(_VB_TILE, _VB_TILE), offset=(o, o), storage="shared", bounds_check=True),
            wp.tile_map(_vb_pad_entry, wp.tile_arange(_VB_TILE, dtype=float), float(nk - o)),
        )
        for r in range(row_panels):
            brc = wp.tile_load(
                B, shape=(_VB_TILE, _VB_TILE), offset=(r * _VB_TILE, o), storage="shared", bounds_check=True
            )
            for p in range(c):
                brp = wp.tile_load(
                    B,
                    shape=(_VB_TILE, _VB_TILE),
                    offset=(r * _VB_TILE, p * _VB_TILE),
                    storage="shared",
                    bounds_check=True,
                )
                lcp = wp.tile_load(
                    L, shape=(_VB_TILE, _VB_TILE), offset=(o, p * _VB_TILE), storage="shared", bounds_check=True
                )
                wp.tile_matmul(brp, wp.tile_transpose(lcp), brc, alpha=-1.0)
            x = wp.tile_lower_solve(lcc, wp.tile_transpose(brc))
            wp.tile_store(B, wp.tile_transpose(x), offset=(r * _VB_TILE, o), bounds_check=True)
        _block_sync()


@wp.func
def _vb_tile_subtract_outer(C: wp.array2d[float], A: wp.array2d[float], B: wp.array2d[float], lower_only: int):
    """``C -= A B^T`` panel by panel; with ``lower_only`` only panels on or below the diagonal."""
    row_panels = (C.shape[0] + _VB_TILE - 1) // _VB_TILE
    col_panels = (C.shape[1] + _VB_TILE - 1) // _VB_TILE
    inner_panels = (A.shape[1] + _VB_TILE - 1) // _VB_TILE
    for r in range(row_panels):
        c_end = col_panels
        if lower_only != 0:
            c_end = wp.min(col_panels, r + 1)
        for c in range(c_end):
            crc = wp.tile_load(
                C,
                shape=(_VB_TILE, _VB_TILE),
                offset=(r * _VB_TILE, c * _VB_TILE),
                storage="shared",
                bounds_check=True,
            )
            for p in range(inner_panels):
                arp = wp.tile_load(
                    A,
                    shape=(_VB_TILE, _VB_TILE),
                    offset=(r * _VB_TILE, p * _VB_TILE),
                    storage="shared",
                    bounds_check=True,
                )
                bcp = wp.tile_load(
                    B,
                    shape=(_VB_TILE, _VB_TILE),
                    offset=(c * _VB_TILE, p * _VB_TILE),
                    storage="shared",
                    bounds_check=True,
                )
                wp.tile_matmul(arp, wp.tile_transpose(bcp), crc, alpha=-1.0)
            wp.tile_store(C, crc, offset=(r * _VB_TILE, c * _VB_TILE), bounds_check=True)


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
    module docstring for the lane and tile paths.
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

    # Regularize the frame rows and keep the diagonal for the Jacobi fallback.
    for row in range(body_start, body_end):
        n = articulation_row_dims[row]
        diag_offset = articulation_block_value_offsets[articulation_diag_slots[row]]
        vec_offset = articulation_row_scalar_offsets[row]
        for k in range(lane, n, threads):
            if k < 6:
                values[diag_offset + k * n + k] = values[diag_offset + k * n + k] + frame_regularization
            diag_backup[vec_offset + k] = values[diag_offset + k * n + k]
    _block_sync()

    for row_k in range(body_start, body_end):
        n_k = articulation_row_dims[row_k]
        diag_offset = articulation_block_value_offsets[articulation_diag_slots[row_k]]
        if n_k <= _VB_LANE_MAX_DIM:
            if lane == 0:
                if _vb_cholesky_inplace(values, diag_offset, n_k) == 0:
                    factor_status[tid] = VB_FACTOR_INDEFINITE
        else:
            ok = _vb_tile_potrf(_vb_view(values, diag_offset, n_k, n_k))
            if ok == 0 and lane == 0:
                factor_status[tid] = VB_FACTOR_INDEFINITE
            for e in range(lane, n_k * n_k, threads):
                r = e // n_k
                if e - r * n_k > r:
                    values[diag_offset + e] = 0.0
        _block_sync()
        if factor_status[tid] != VB_FACTOR_OK:
            break

        # Column blocks: L_ik = H_ik L_kk^-T.
        col_begin = articulation_block_col_offsets[row_k]
        col_end = articulation_block_col_offsets[row_k + 1]
        if n_k <= _VB_LANE_MAX_DIM:
            # One lane per block row, numbered across all blocks of the column.
            base = int(0)
            for entry in range(col_begin, col_end):
                slot = articulation_block_col_slots[entry]
                rows = articulation_block_dims[slot, 0]
                offset = articulation_block_value_offsets[slot]
                for r in range(_vb_first_lane_item(lane, base, threads), rows, threads):
                    row_offset = offset + r * n_k
                    for c in range(n_k):
                        value = values[row_offset + c]
                        for p in range(c):
                            value = value - values[row_offset + p] * values[diag_offset + c * n_k + p]
                        values[row_offset + c] = value / values[diag_offset + c * n_k + c]
                base += rows
        else:
            for entry in range(col_begin, col_end):
                slot = articulation_block_col_slots[entry]
                _vb_tile_right_solve_lower_transpose(
                    _vb_view(values, diag_offset, n_k, n_k),
                    _vb_view(values, articulation_block_value_offsets[slot], articulation_block_dims[slot, 0], n_k),
                )
        _block_sync()

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
        else:
            for entry in range(update_begin, update_end):
                dst = articulation_factor_update_dst_slots[entry]
                left = articulation_factor_update_left_slots[entry]
                right = articulation_factor_update_right_slots[entry]
                left_rows = articulation_block_dims[left, 0]
                right_rows = articulation_block_dims[right, 0]
                lower_only = int(0)
                if left == right:
                    lower_only = 1
                _vb_tile_subtract_outer(
                    _vb_view(values, articulation_block_value_offsets[dst], left_rows, right_rows),
                    _vb_view(values, articulation_block_value_offsets[left], left_rows, n_k),
                    _vb_view(values, articulation_block_value_offsets[right], right_rows, n_k),
                    lower_only,
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
                    _vb_lower_solve(values, diag_offset, n, delta, vec_offset)
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
                    _vb_upper_solve(values, diag_offset, n, delta, vec_offset)
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


def cooperative_shared_memory_bytes(device=None) -> int:
    """Load :data:`solve_articulation_variable_cooperative` on a CUDA ``device`` and return its shared memory per CTA.

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
    module_exec = solve_articulation_variable_cooperative.module.load(device, block_dim=VB_COOPERATIVE_BLOCK_DIM)
    required = max(
        (int(value) for key, value in module_exec.meta.items() if key.endswith("_smem_bytes")),
        default=0,
    )
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
    CPU, where the tile operations run serially (a debugging path). Same arguments as
    :func:`launch_solve_articulation_variable`.
    """
    if layout.mixed_articulation_count == 0:
        return
    device = wp.get_device(device if device is not None else values.device)
    block_dim = 1
    if device.is_cuda:
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
