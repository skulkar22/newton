# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional solve metrics of reduced elastic bodies in the coupled articulation backend.

The coupled backend (:mod:`~newton._src.solvers.vbd.rigid_sparse_articulation_variable_kernels`)
factors each mixed articulation's assembled system ``H dx = rhs`` in place. With
``elastic_solver_metrics=True`` the solver keeps a copy of the assembled values and right-hand
side of the mixed articulations, taken before the backend adds its frame regularization and
factors, and :func:`evaluate_articulation_variable_elastic_metrics` evaluates, per reduced
elastic body, the same five quantities the local block solve reports, over the rows of the
articulation system that belong to the body (see
:meth:`~newton.solvers.SolverVBD.elastic_mode_solve_metrics`).
"""

import warp as wp

from newton._src.solvers.vbd.rigid_sparse_articulation_variable_kernels import VB_FACTOR_OK

wp.set_module_options({"enable_backward": False})


@wp.kernel
def evaluate_articulation_variable_elastic_metrics(
    rigid_articulation_count: int,
    frame_regularization: float,
    update_relaxation: float,
    mixed_elastic_bodies: wp.array[wp.int32],
    mixed_elastic_body_rows: wp.array[wp.int32],
    mixed_elastic_body_articulations: wp.array[wp.int32],
    articulation_body_offsets: wp.array[wp.int32],
    articulation_row_dims: wp.array[wp.int32],
    articulation_row_scalar_offsets: wp.array[wp.int32],
    articulation_block_value_offsets: wp.array[wp.int32],
    articulation_block_row_offsets: wp.array[wp.int32],
    articulation_block_cols: wp.array[wp.int32],
    articulation_block_col_offsets: wp.array[wp.int32],
    articulation_block_col_rows: wp.array[wp.int32],
    articulation_block_col_slots: wp.array[wp.int32],
    body_elastic_index: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    assembled_values: wp.array[float],
    assembled_rhs: wp.array[float],
    delta: wp.array[float],
    factor_status: wp.array[wp.int32],
    initial_residual_norm: wp.array[float],
    solve_residual_norm: wp.array[float],
    applied_residual_norm: wp.array[float],
    update_norm: wp.array[float],
    update_max: wp.array[float],
):
    """Write the solve metrics of one reduced elastic body of a mixed articulation per thread.

    Launch with ``dim = mixed_elastic_body_count`` after the backend's factor-and-solve.
    ``assembled_values`` and ``assembled_rhs`` hold the system as assembled (lower block triangle,
    full symmetric diagonal blocks, no regularization); ``delta`` is the unrelaxed solution. For
    row component ``k`` of the body's row (frame components only when the frame is dynamic, plus
    every modal component), with ``Hd = (H + frame_regularization * I_frame) delta`` over the whole
    articulation:

    - ``initial_residual_norm``: ``|rhs_k|``, the negative gradient before the update;
    - ``solve_residual_norm``: ``|Hd_k - rhs_k|``, the linear residual of the unrelaxed update;
    - ``applied_residual_norm``: ``|relaxation * Hd_k - rhs_k|``, the same for the applied update;
    - ``update_norm`` and ``update_max``: 2-norm and max-norm of ``relaxation * delta_k``.

    Norms are 2-norms over the components. A body whose articulation failed to factor or solve
    (``factor_status != VB_FACTOR_OK``) gets NaN in every metric.
    """
    entry = wp.tid()
    body = mixed_elastic_bodies[entry]
    elastic_index = body_elastic_index[body]
    articulation = mixed_elastic_body_articulations[entry]
    nan = wp.nan
    if factor_status[articulation - rigid_articulation_count] != VB_FACTOR_OK:
        initial_residual_norm[elastic_index] = nan
        solve_residual_norm[elastic_index] = nan
        applied_residual_norm[elastic_index] = nan
        update_norm[elastic_index] = nan
        update_max[elastic_index] = nan
        return

    row = mixed_elastic_body_rows[entry]
    body_start = articulation_body_offsets[articulation]
    n = articulation_row_dims[row]
    row_offset = articulation_row_scalar_offsets[row]
    first = int(0)
    if body_inv_mass[body] == 0.0:
        first = 6

    initial_sq = float(0.0)
    solve_sq = float(0.0)
    applied_sq = float(0.0)
    update_sq = float(0.0)
    largest = float(0.0)
    for k in range(first, n):
        hd = float(0.0)
        if k < 6:
            hd = frame_regularization * delta[row_offset + k]
        # Stored blocks (row, c) with c <= row; the diagonal block is stored in full.
        for slot in range(articulation_block_row_offsets[row], articulation_block_row_offsets[row + 1]):
            col_row = body_start + articulation_block_cols[slot]
            col_dim = articulation_row_dims[col_row]
            col_offset = articulation_row_scalar_offsets[col_row]
            value_offset = articulation_block_value_offsets[slot] + k * col_dim
            for c in range(col_dim):
                hd += assembled_values[value_offset + c] * delta[col_offset + c]
        # Blocks (i, row) with i > row enter transposed.
        for col_entry in range(articulation_block_col_offsets[row], articulation_block_col_offsets[row + 1]):
            slot = articulation_block_col_slots[col_entry]
            below = body_start + articulation_block_col_rows[col_entry]
            below_offset = articulation_row_scalar_offsets[below]
            value_offset = articulation_block_value_offsets[slot]
            for i in range(articulation_row_dims[below]):
                hd += assembled_values[value_offset + i * n + k] * delta[below_offset + i]
        b = assembled_rhs[row_offset + k]
        solve_k = hd - b
        applied_k = update_relaxation * hd - b
        update_k = update_relaxation * delta[row_offset + k]
        initial_sq += b * b
        solve_sq += solve_k * solve_k
        applied_sq += applied_k * applied_k
        update_sq += update_k * update_k
        largest = wp.max(largest, wp.abs(update_k))

    initial_residual_norm[elastic_index] = wp.sqrt(initial_sq)
    solve_residual_norm[elastic_index] = wp.sqrt(solve_sq)
    applied_residual_norm[elastic_index] = wp.sqrt(applied_sq)
    update_norm[elastic_index] = wp.sqrt(update_sq)
    update_max[elastic_index] = largest
