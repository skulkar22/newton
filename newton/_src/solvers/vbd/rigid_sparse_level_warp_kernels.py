# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp-cooperative level solve for sparse plans with split Schur updates."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import warp as wp

from . import rigid_sparse_level_launch_kernels as native
from .rigid_sparse_articulation_kernels import (
    _block_sync,
    _warp_sync,
)

if TYPE_CHECKING:
    from .rigid_sparse_articulation import RigidArticulationSparseLayout
    from .rigid_sparse_level_launch_kernels import LevelLaunchPhaseTasks, RigidArticulationLevelLaunchPlan

wp.set_module_options({"block_dim": 128, "enable_backward": False, "fast_math": False, "fuse_fp": True})

ia = wp.array[wp.int32]


fa = wp.array[wp.float32]


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffffu, value, src);
#else
    return value;
#endif
""")
def shfl(value: float, src: int) -> float:
    pass


@wp.struct
class Sparse:
    rowptr: ia
    cols: ia
    colptr: ia
    rows: ia
    slots: ia
    diag: ia


@wp.struct
class Tasks:
    short_ids: ia
    short_aux: ia


@wp.func
def column(vals: wp.array[float], slot: int, diag: int, lane: int):
    """Each eight-lane group solves one row, and groups 0 and 1 solve a second."""
    row = lane >> 3
    col = lane & 7
    offset = 36 * slot + 6 * row + col
    factor_row = 36 * diag + 6 * col
    first = float(0.0)
    second = float(0.0)
    if col < 6:
        first = vals[offset]
        if row < 2:
            second = vals[offset + 24]

    for step in range(6):
        coeff = float(0.0)
        if col >= step and col < 6:
            coeff = vals[factor_row + step]
        if col == step:
            first = first / coeff
            second = second / coeff
        if step < 5:
            src = 8 * row + step
            solved_first = shfl(first, src)
            solved_second = shfl(second, src)
            if col > step:
                first = first - solved_first * coeff
                second = second - solved_second * coeff

    if col < 6:
        vals[offset] = first
        if row < 2:
            vals[offset + 24] = second


@wp.func
def _subst_rhs(data: fa, row: int, lane: int) -> float:
    value = float(0.0)
    if lane < 6:
        value = data[6 * row + lane]
    return value


@wp.func
def _subst_entries(
    vals: fa,
    delta: fa,
    row: int,
    body_start: int,
    neighbors: ia,
    slots: ia,
    first: int,
    last: int,
    initial: float,
    lane: int,
    upper: bool,
    chunked: bool,
) -> float:
    """Four 8-lane groups compute independent dots; lanes 0..5 consume them in entry order."""
    group = lane // 8
    component = lane % 8
    total = initial
    for batch in range(first, last, 4):
        entry = batch + group
        payload = float(0.0)
        if entry < last:
            neighbor = body_start + neighbors[entry]
            if upper or neighbor < row:
                if component < 6:
                    address = 36 * entry + 6 * component
                    stride = int(1)
                    if upper:
                        address = 36 * slots[entry] + component
                        stride = 6
                    source = 6 * neighbor
                    product = vals[address] * delta[source]
                    payload = product
                    if chunked:
                        payload = 0.0 + product
                    for inner in range(5):
                        payload = payload + vals[address + (inner + 1) * stride] * delta[source + inner + 1]
                elif component == 6:
                    payload = 1.0

        for item in range(4):
            if batch + item < last:
                present = float(1.0)
                if not upper:
                    present = shfl(payload, 8 * item + 6)
                term = payload
                if item != 0:
                    term = shfl(payload, 8 * item + component)
                if present != 0.0:
                    if chunked:
                        total = total + term
                    else:
                        total = total - term
    return total


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    if (data.strides[0] == sizeof(float) && !(reinterpret_cast<size_t>(data.data) & 7u)) {
        const float2 pair = *reinterpret_cast<const float2*>(data.data + start);
        return wp::vec2(pair.x, pair.y);
    }
#endif
    return wp::vec2(wp::index(data, start), wp::index(data, start + 1));
""")
def _front_load_pair(data: fa, start: int) -> wp.vec2:
    pass


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    if (data.strides[0] == sizeof(float) && !(reinterpret_cast<size_t>(data.data) & 7u)) {
        *reinterpret_cast<float2*>(data.data + start) = make_float2(first, second);
        return;
    }
#endif
    wp::index(data, start) = first;
    wp::index(data, start + 1) = second;
""")
def _front_store_pair(data: fa, start: int, first: float, second: float):
    pass


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __drcp_rn(static_cast<double>(factor));
#else
    return 1.0 / static_cast<double>(factor);
#endif
""")
def _front_inverse(factor: float) -> wp.float64:
    pass


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    const double product = __dmul_rn(static_cast<double>(value), inverse);
    const float candidate = __double2float_rn(product);
    if (fabsf(candidate) >= 0x1p-125f) return candidate;
    if (value == 0.0f && isfinite(inverse) && inverse != 0.0) return candidate;
    return __fdiv_rn(value, factor);
#else
    return value / factor;
#endif
""")
def _front_quotient(value: float, factor: float, inverse: wp.float64) -> float:
    pass


@wp.func
def _subst_finish(vals: fa, delta: fa, row: int, diag: int, residual: float, lane: int, upper: bool):
    """Prepare the six inverses and all coefficient routes outside the substitution chain."""
    offset = int(0)
    if lane < 6:
        offset = 7 * lane
    elif lane < 31:
        packed = lane - 6
        offset = 6 * (1 + packed // 5) + packed % 5
    factor = vals[36 * diag + offset]
    divisor = wp.where(lane < 6, factor, 1.0)
    inverse = _front_inverse(divisor)
    coefficients = wp.spatial_vector()
    rank = wp.where(upper, 5 - lane, lane)
    for stage in range(5):
        pivot = wp.where(upper, 5 - stage, stage)
        owner = wp.where(upper, 6 + 5 * (pivot - 1) + lane, 6 + 5 * (lane - 1) + pivot)
        coefficients[stage] = shfl(factor, owner & 31)
    value = residual
    output = float(0.0)
    for stage in range(5):
        pivot = wp.where(upper, 5 - stage, stage)
        quotient = float(0.0)
        if rank == stage:
            quotient = _front_quotient(value, divisor, inverse)
            output = quotient
        solved = shfl(quotient, pivot)
        value = value - coefficients[stage] * solved
    if rank == 5:
        output = _front_quotient(value, divisor, inverse)
    if lane < 6:
        delta[6 * row + lane] = output


@wp.func
def factor_front(
    layout: Sparse, vals: fa, sol: fa, piv: ia, row: int, lane: int, residual: float, source: fa, offset: int
):
    """One register per column, an independent diagonal recurrence, and one RHS row in lane 30."""
    group = lane // 6
    local = lane % 6
    begin = layout.colptr[row]
    end = layout.colptr[row + 1]
    slot = layout.diag[row]
    valid = group == 0
    if group > 0 and group <= 4 and begin + group <= end:
        slot = layout.slots[begin + group - 1]
        valid = True
    start = 36 * slot + 6 * local
    pair01 = wp.vec2()
    pair23 = wp.vec2()
    pair45 = wp.vec2()
    if valid:
        address = wp.where(offset < 0, start, offset + 36 * group + 6 * local)
        pair01 = _front_load_pair(source, address)
        pair23 = _front_load_pair(source, address + 2)
        pair45 = _front_load_pair(source, address + 4)
    col0 = wp.where(lane == 30, shfl(residual, 0), pair01[0])
    col1 = wp.where(lane == 30, shfl(residual, 1), pair01[1])
    col2 = wp.where(lane == 30, shfl(residual, 2), pair23[0])
    col3 = wp.where(lane == 30, shfl(residual, 3), pair23[1])
    col4 = wp.where(lane == 30, shfl(residual, 4), pair45[0])
    col5 = wp.where(lane == 30, shfl(residual, 5), pair45[1])
    diagonal = wp.where(
        lane < 2,
        wp.where(lane == 0, col0, col1),
        wp.where(lane < 4, wp.where(lane == 2, col2, col3), wp.where(lane == 4, col4, col5)),
    )
    guard = wp.max(1.0e-30, 1.0e-6 * wp.abs(diagonal))
    clamped = False
    accepted = diagonal > guard
    root = wp.sqrt(wp.where(accepted, diagonal, guard))
    clamped = wp.where(lane == 0, not accepted, clamped)
    pivot = shfl(root, 0)
    solved = _front_quotient(col0, pivot, _front_inverse(pivot))
    col0 = wp.where(lane == 0, pivot, solved)
    diagonal = diagonal - solved * solved
    accepted = diagonal > guard
    root = wp.sqrt(wp.where(accepted, diagonal, guard))
    col1 = col1 - solved * shfl(solved, 1)
    col2 = col2 - solved * shfl(solved, 2)
    col3 = col3 - solved * shfl(solved, 3)
    col4 = col4 - solved * shfl(solved, 4)
    col5 = col5 - solved * shfl(solved, 5)

    clamped = wp.where(lane == 1, not accepted, clamped)
    pivot = shfl(root, 1)
    solved = _front_quotient(col1, pivot, _front_inverse(pivot))
    col1 = wp.where(lane == 1, pivot, solved)
    diagonal = diagonal - solved * solved
    accepted = diagonal > guard
    root = wp.sqrt(wp.where(accepted, diagonal, guard))
    col2 = col2 - solved * shfl(solved, 2)
    col3 = col3 - solved * shfl(solved, 3)
    col4 = col4 - solved * shfl(solved, 4)
    col5 = col5 - solved * shfl(solved, 5)

    clamped = wp.where(lane == 2, not accepted, clamped)
    pivot = shfl(root, 2)
    solved = _front_quotient(col2, pivot, _front_inverse(pivot))
    col2 = wp.where(lane == 2, pivot, solved)
    diagonal = diagonal - solved * solved
    accepted = diagonal > guard
    root = wp.sqrt(wp.where(accepted, diagonal, guard))
    col3 = col3 - solved * shfl(solved, 3)
    col4 = col4 - solved * shfl(solved, 4)
    col5 = col5 - solved * shfl(solved, 5)

    clamped = wp.where(lane == 3, not accepted, clamped)
    pivot = shfl(root, 3)
    solved = _front_quotient(col3, pivot, _front_inverse(pivot))
    col3 = wp.where(lane == 3, pivot, solved)
    diagonal = diagonal - solved * solved
    accepted = diagonal > guard
    root = wp.sqrt(wp.where(accepted, diagonal, guard))
    col4 = col4 - solved * shfl(solved, 4)
    col5 = col5 - solved * shfl(solved, 5)

    clamped = wp.where(lane == 4, not accepted, clamped)
    pivot = shfl(root, 4)
    solved = _front_quotient(col4, pivot, _front_inverse(pivot))
    col4 = wp.where(lane == 4, pivot, solved)
    diagonal = diagonal - solved * solved
    accepted = diagonal > guard
    root = wp.sqrt(wp.where(accepted, diagonal, guard))
    col5 = col5 - solved * shfl(solved, 5)

    clamped = wp.where(lane == 5, not accepted, clamped)
    pivot = shfl(root, 5)
    solved = _front_quotient(col5, pivot, _front_inverse(pivot))
    col5 = wp.where(lane == 5, pivot, solved)

    if clamped:
        wp.atomic_add(piv, 0, 1)
    if valid:
        _front_store_pair(vals, start, col0, wp.where(lane < 1, 0.0, col1))
        _front_store_pair(vals, start + 2, wp.where(lane < 2, 0.0, col2), wp.where(lane < 3, 0.0, col3))
        _front_store_pair(vals, start + 4, wp.where(lane < 4, 0.0, col4), wp.where(lane < 5, 0.0, col5))
    if lane == 30:
        _front_store_pair(sol, 6 * row, col0, col1)
        _front_store_pair(sol, 6 * row + 2, col2, col3)
        _front_store_pair(sol, 6 * row + 4, col4, col5)


@wp.func
def register_front(layout: Sparse, vals: fa, sol: fa, piv: ia, row: int, lane: int, residual: float):
    factor_front(layout, vals, sol, piv, row, lane, residual, vals, -1)


@wp.func
def backward(
    vals: fa,
    delta: fa,
    row: int,
    body_start: int,
    col_offsets: ia,
    col_rows: ia,
    col_slots: ia,
    diag_slots: ia,
    lane: int,
):
    value = _subst_rhs(delta, row, lane)
    value = _subst_entries(
        vals,
        delta,
        row,
        body_start,
        col_rows,
        col_slots,
        col_offsets[row],
        col_offsets[row + 1],
        value,
        lane,
        True,
        False,
    )
    _subst_finish(vals, delta, row, diag_slots[row], value, lane, True)


@wp.func
def row_front(layout: Sparse, vals: fa, sol: fa, piv: ia, row: int, lane: int, residual: float):
    register_front(layout, vals, sol, piv, row, lane, residual)
    begin = layout.colptr[row]
    end = layout.colptr[row + 1]
    diag = layout.diag[row]
    if begin + 4 < end:
        _warp_sync()
        for entry in range(begin + 4, end):
            column(vals, layout.slots[entry], diag, lane)


@wp.kernel
def first_front(layout: Sparse, tasks: Tasks, vals: fa, rhs: fa, sol: fa, piv: ia, first: int):
    tid = wp.tid()
    lane = tid & 31
    row = tasks.short_ids[first + tid // 32]
    row_front(layout, vals, sol, piv, row, lane, _subst_rhs(rhs, row, lane))


@wp.kernel
def broad_front(
    layout: Sparse,
    tasks: Tasks,
    vals: fa,
    rhs: fa,
    sol: fa,
    piv: ia,
    short_begin: int,
):
    tid = wp.tid()
    work = tid // 128
    local = tid & 127
    lane = tid & 31
    idx = short_begin + work
    row = tasks.short_ids[idx]
    base = tasks.short_aux[idx]
    if local < 32:
        value = _subst_rhs(rhs, row, lane)
        value = _subst_entries(
            vals,
            sol,
            row,
            base,
            layout.cols,
            layout.cols,
            layout.rowptr[row],
            layout.rowptr[row + 1],
            value,
            lane,
            False,
            False,
        )
        register_front(layout, vals, sol, piv, row, lane, value)
    begin = layout.colptr[row] + 4
    end = layout.colptr[row + 1]
    if begin < end:
        _block_sync()
        diag = layout.diag[row] * 36
        for job in range(local, 6 * (end - begin), 128):
            slot = layout.slots[begin + job // 6]
            address = slot * 36 + 6 * (job % 6)
            solved = wp.spatial_vector()
            for col in range(6):
                value = vals[address + col]
                for inner in range(6):
                    if inner < col:
                        value -= solved[inner] * vals[diag + 6 * col + inner]
                solved[col] = value / vals[diag + 7 * col]
            for col in range(6):
                vals[address + col] = solved[col]


@wp.kernel
def back_sub(layout: Sparse, tasks: Tasks, vals: fa, sol: fa, short_begin: int):
    tid = wp.tid()
    work = tid // 32
    lane = tid % 32
    idx = short_begin + work
    backward(
        vals,
        sol,
        tasks.short_ids[idx],
        tasks.short_aux[idx],
        layout.colptr,
        layout.rows,
        layout.slots,
        layout.diag,
        lane,
    )


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0x7u << (threadIdx.x & 28u), value, source, 4);
#else
    return value;
#endif
""")
def _sback_share(value: float, source: int) -> float:
    pass


@wp.func_native("return fmaf(coeff, weight, accum);")
def _sback_fma(coeff: float, weight: float, accum: float) -> float:
    pass


@wp.func
def _sback_axpy(coeff: wp.vec2, weight: float, accum: wp.vec2) -> wp.vec2:
    return wp.vec2(_sback_fma(coeff[0], weight, accum[0]), _sback_fma(coeff[1], weight, accum[1]))


@wp.func
def _sback_dot(layout: Sparse, vals: fa, sol: fa, base: int, entry: int, column: int) -> wp.vec2:
    """Two columns of block.T @ x; a product, then five ordered FMAs per column."""
    block = 36 * layout.slots[entry] + column
    other = 6 * (base + layout.rows[entry])
    vec01 = _front_load_pair(sol, other)
    vec23 = _front_load_pair(sol, other + 2)
    vec45 = _front_load_pair(sol, other + 4)
    mat0 = _front_load_pair(vals, block)
    mat1 = _front_load_pair(vals, block + 6)
    mat2 = _front_load_pair(vals, block + 12)
    mat3 = _front_load_pair(vals, block + 18)
    mat4 = _front_load_pair(vals, block + 24)
    mat5 = _front_load_pair(vals, block + 30)
    dot = mat0 * vec01[0]
    dot = _sback_axpy(mat1, vec01[1], dot)
    dot = _sback_axpy(mat2, vec23[0], dot)
    dot = _sback_axpy(mat3, vec23[1], dot)
    dot = _sback_axpy(mat4, vec45[0], dot)
    return _sback_axpy(mat5, vec45[1], dot)


@wp.func
def _sback_solve_pair(
    residual: wp.vec2, even: float, odd_row: wp.vec2, inv_even: wp.float64, inv_odd: wp.float64
) -> wp.vec2:
    odd = _front_quotient(residual[1], odd_row[1], inv_odd)
    updated = _sback_fma(odd_row[0], -odd, residual[0])
    return wp.vec2(_front_quotient(updated, even, inv_even), odd)


@wp.func
def reverse_local(layout: Sparse, vals: fa, sol: fa, forward_data: fa, base: int, row: int, pair: int):
    column = 2 * pair
    diag = 36 * layout.diag[row]
    entry = layout.colptr[row]
    end = layout.colptr[row + 1]
    addr = 6 * row + column
    residual = _front_load_pair(forward_data, addr)

    even = vals[diag + 7 * column]
    odd_row = _front_load_pair(vals, diag + 7 * column + 6)
    inv_even = _front_inverse(even)
    inv_odd = _front_inverse(odd_row[1])
    from5 = wp.vec2()
    from4 = wp.vec2()
    from3 = wp.vec2()
    from2 = wp.vec2()
    if pair < 2:
        from5 = _front_load_pair(vals, diag + 30 + column)
        from4 = _front_load_pair(vals, diag + 24 + column)
    if pair == 0:
        from3 = _front_load_pair(vals, diag + 18 + column)
        from2 = _front_load_pair(vals, diag + 12 + column)

    while end - entry >= 2:
        first = _sback_dot(layout, vals, sol, base, entry, column)
        second = _sback_dot(layout, vals, sol, base, entry + 1, column)
        residual = (residual - first) - second
        entry += 2
    if entry < end:
        residual = residual - _sback_dot(layout, vals, sol, base, entry, column)

    solved = wp.vec2()
    if pair == 2:
        solved = _sback_solve_pair(residual, even, odd_row, inv_even, inv_odd)
    pivot5 = _sback_share(solved[1], 2)
    pivot4 = _sback_share(solved[0], 2)
    if pair < 2:
        residual = _sback_axpy(from5, -pivot5, residual)
        residual = _sback_axpy(from4, -pivot4, residual)
    if pair == 1:
        solved = _sback_solve_pair(residual, even, odd_row, inv_even, inv_odd)
    pivot3 = _sback_share(solved[1], 1)
    pivot2 = _sback_share(solved[0], 1)
    if pair == 0:
        residual = _sback_axpy(from3, -pivot3, residual)
        residual = _sback_axpy(from2, -pivot2, residual)
        solved = _sback_solve_pair(residual, even, odd_row, inv_even, inv_odd)
    return solved


@wp.kernel
def short_leading_back(layout: Sparse, tasks: Tasks, vals: fa, sol: fa, begin: int, count: int):
    """Three live lanes per four-thread group own column pairs (0,1), (2,3), (4,5)."""
    tid = wp.tid()
    work = tid >> 2
    pair = tid & 3
    if work >= count or pair == 3:
        return
    index = begin + work
    row = tasks.short_ids[index]
    base = tasks.short_aux[index]
    solved = reverse_local(layout, vals, sol, sol, base, row, pair)
    _front_store_pair(sol, 6 * row + 2 * pair, solved[0], solved[1])


class _Buffer(Enum):
    VALUES = 0
    RHS = 1
    DELTA = 2
    PIVOT_CLAMPED = 3


def _bind_phase(phase: "LevelLaunchPhaseTasks") -> Tasks:
    tasks = Tasks()
    tasks.short_ids = phase.short
    tasks.short_aux = phase.short_aux
    return tasks


@dataclass
class WarpLevelSchedule:
    """Kernel arguments fixed by a layout, with slots for the four per-call arrays."""

    layout: "RigidArticulationSparseLayout"
    commands: list

    def launch(self, values, rhs, delta, pivot_clamped, device):
        buffers = (values, rhs, delta, pivot_clamped)
        for kernel, dim, arguments, bindings in self.commands:
            inputs = arguments.copy()
            for index, buffer in bindings:
                inputs[index] = buffers[buffer]
            wp.launch(kernel, dim=dim, inputs=inputs, device=device, block_dim=128)


def build_warp_schedule(
    plan: "RigidArticulationLevelLaunchPlan", layout: "RigidArticulationSparseLayout"
) -> WarpLevelSchedule | None:
    """Prepare the six-kernel schedule for the leading split Schur case."""
    cut = plan.leading_level_count
    if (
        plan.articulation_count != 1
        or cut == 0
        or not 0 < plan.level_count - cut < 8
        or plan.tail_split_entries <= 0
        or plan.tail_block_dim != 128
        or plan.forward.long.shape[0]
        or plan.backward.long.shape[0]
        or any(plan.update.level_range(level)[-1] == 0 for level in range(cut))
    ):
        return None
    schedule = WarpLevelSchedule(layout, [])
    sparse = Sparse()
    sparse.rowptr = layout.articulation_block_row_offsets
    sparse.cols = layout.articulation_block_cols
    sparse.colptr = layout.articulation_block_col_offsets
    sparse.rows = layout.articulation_block_col_rows
    sparse.slots = layout.articulation_block_col_slots
    sparse.diag = layout.articulation_diag_slots
    levels = layout.levels

    def record(kernel, dim, args):
        if dim:
            bindings = [(index, arg.value) for index, arg in enumerate(args) if isinstance(arg, _Buffer)]
            schedule.commands.append((kernel, dim, args, bindings))

    fwd = _bind_phase(plan.forward)
    bwd = _bind_phase(plan.backward)
    values, rhs, delta, pivot_clamped = _Buffer
    for level in range(cut):
        first, count, *_ = plan.forward.level_range(level)
        if level == 0:
            record(first_front, 32 * count, [sparse, fwd, values, rhs, delta, pivot_clamped, first])
        else:
            record(broad_front, 128 * count, [sparse, fwd, values, rhs, delta, pivot_clamped, first])
        short_begin, short_count, chunk_begin, chunk_count, long_begin, long_count = plan.update.level_range(level)
        record(
            native.level_launch_update,
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
                values,
                plan.update.partial,
            ],
        )
        record(
            native.level_launch_update_reduce,
            36 * long_count,
            [
                long_begin,
                plan.update.long,
                plan.update.long_chunk_offsets,
                levels.level_update_dst_slots,
                plan.update.partial,
                values,
            ],
        )
    record(
        native.solve_articulation_sparse_levels_tail,
        plan.articulation_count * plan.tail_block_dim,
        [
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
            pivot_clamped,
            delta,
        ],
    )
    for level in reversed(range(cut)):
        first, count, *_ = plan.backward.level_range(level)
        if count >= 256:
            record(short_leading_back, 4 * count, [sparse, bwd, values, delta, first, count])
        else:
            record(back_sub, 32 * count, [sparse, bwd, values, delta, first])
    return schedule
