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
the factorization or the solve; :func:`vb_add_block_entry_atomic` serves assembly only.

Failure policy: a pivot that is not positive or not finite, or a non-finite solution, sets
``factor_status`` (:data:`VB_FACTOR_INDEFINITE`, :data:`VB_SOLVE_NONFINITE`), increments the
failure counter, and replaces the articulation's update by a scalar Jacobi step
``dx_k = rhs_k / d_k`` from the diagonal ``d`` saved before factorization (0 where ``d_k`` is
not positive and finite, or ``rhs_k`` is not finite), so the update is always finite.

Frame regularization: the backend adds ``frame_regularization`` to the six frame diagonal
entries of every row of a mixed articulation before factorization (the 6 x 6 kernels apply
``rigid_articulation_diagonal_regularization`` to the rigid prefix through the body Hessians).

The serial kernel here runs one thread per mixed articulation and is the CPU backend; the
helpers and the layout are device agnostic, so a cooperative CUDA kernel can use the same
storage.
"""

import warp as wp

from newton._src.solvers.vbd.rigid_sparse_articulation_kernels import (
    _apply_sparse_delta_value_to_body,
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


@wp.func
def vb_add_block_entry_atomic(
    values: wp.array[float],
    block_value_offsets: wp.array[wp.int32],
    block_dims: wp.array2d[wp.int32],
    slot: int,
    r: int,
    c: int,
    value: float,
):
    """Atomically add ``value`` to entry ``(r, c)`` of block slot ``slot`` (no-op for ``slot < 0``)."""
    if slot >= 0:
        wp.atomic_add(values, vb_block_entry_index(block_value_offsets, block_dims, slot, r, c), value)


@wp.func
def vb_add_rhs_atomic(rhs: wp.array[float], row_scalar_offsets: wp.array[wp.int32], row: int, k: int, value: float):
    """Atomically add ``value`` to component ``k`` of sparse row ``row``."""
    wp.atomic_add(rhs, vb_rhs_index(row_scalar_offsets, row, k), value)


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

    Launches :func:`solve_articulation_variable_serial` (one thread per articulation) on every
    device; this is the single dispatch point for a cooperative CUDA kernel.

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
    if layout.mixed_articulation_count == 0:
        return
    wp.launch(
        solve_articulation_variable_serial,
        dim=layout.mixed_articulation_count,
        inputs=[
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
        ],
        outputs=[values, diag_backup, factor_status, failure_count, delta],
        device=device if device is not None else values.device,
    )
