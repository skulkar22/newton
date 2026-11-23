# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp-cooperative level solve for sparse plans with split Schur updates."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import warp as wp

from .rigid_sparse_articulation_kernels import (
    _block_sync,
    _scalar_vec_load,
    _solve_lower66,
    _solve_upper66_from_lower,
    _warp_sync,
    vec6f,
)
from .rigid_sparse_levels_kernels import _load_block66_scalar

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
    base: ia
    rowptr: ia
    cols: ia
    colptr: ia
    rows: ia
    slots: ia
    diag: ia
    artlev: ia
    levrow: ia
    levcol: ia
    levupd: ia
    cslot: ia
    cdiag: ia
    dest: ia
    ptr: ia
    left: ia
    right: ia


@wp.struct
class Tasks:
    short_ids: ia
    short_aux: ia
    long_ids: ia
    long_chunk_offsets: ia
    chunk_begin: ia
    chunk_end: ia
    chunk_aux: ia


@wp.func
def chol(vals: wp.array[float], slot: int, piv: wp.array[wp.int32], lane: int):
    """Packed lower triangle: lane row*(row+1)//2 + col owns one residual."""
    row = int(lane >= 1) + int(lane >= 3) + int(lane >= 6) + int(lane >= 10) + int(lane >= 15)
    row_start = row * (row + 1) // 2
    col = wp.min(lane, 20) - row_start
    col_start = col * (col + 1) // 2
    base = 36 * slot
    original = float(0.0)
    if lane < 21:
        original = vals[base + 6 * row + col]
    value = original

    for step in range(6):
        pivot_lane = step * (step + 1) // 2 + step
        if lane == pivot_lane:
            floor = wp.max(1.0e-6 * wp.abs(original), 1.0e-30)
            if not (value > floor):
                wp.atomic_add(piv, 0, 1)
                value = floor
            value = wp.sqrt(value)
        if step < 5:
            pivot = shfl(value, pivot_lane)
            if col == step and row > step:
                value = value / pivot
            left = shfl(value, row_start + step)
            right = shfl(value, col_start + step)
            if col > step:
                value = value - left * right

    if lane < 21:
        vals[base + 6 * row + col] = value
        if col != row:
            vals[base + 6 * col + row] = 0.0


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
def update(
    vals: wp.array[float],
    dst: int,
    begin: int,
    end: int,
    left_slots: wp.array[wp.int32],
    right_slots: wp.array[wp.int32],
    lane: int,
):
    """Four lanes per operand row hold columns 0..3, with columns 4..5 in extra registers."""
    row = lane >> 2
    col = lane & 3
    entry = row * 6 + col
    offset = 36 * dst + entry
    first = float(0.0)
    second = float(0.0)
    if row < 6:
        first = vals[offset]
        if col < 2:
            second = vals[offset + 4]

    for term in range(begin, end):
        left_base = 36 * left_slots[term] + entry
        right_base = 36 * right_slots[term] + entry
        left_first = float(0.0)
        left_second = float(0.0)
        right_first = float(0.0)
        right_second = float(0.0)
        if row < 6:
            left_first = vals[left_base]
            right_first = vals[right_base]
            if col < 2:
                left_second = vals[left_base + 4]
                right_second = vals[right_base + 4]
        dot_first = float(0.0)
        dot_second = float(0.0)
        for inner in range(6):
            left_reg = left_first
            right_reg = right_first
            if inner >= 4:
                left_reg = left_second
                right_reg = right_second
            component = inner & 3
            left = shfl(left_reg, 4 * row + component)
            right_lo = shfl(right_reg, 4 * col + component)
            right_hi = shfl(right_reg, 16 + 4 * (col & 1) + component)
            dot_first = dot_first + left * right_lo
            dot_second = dot_second + left * right_hi
        first = first - dot_first
        second = second - dot_second

    if row < 6:
        vals[offset] = first
        if col < 2:
            vals[offset + 4] = second


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
def forward(
    vals: fa, rhs: fa, delta: fa, row: int, body_start: int, row_offsets: ia, cols: ia, diag_slots: ia, lane: int
):
    value = _subst_rhs(rhs, row, lane)
    value = _subst_entries(
        vals, delta, row, body_start, cols, cols, row_offsets[row], row_offsets[row + 1], value, lane, False, False
    )
    _subst_finish(vals, delta, row, diag_slots[row], value, lane, False)


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
def backward_long(
    vals: fa,
    delta: fa,
    row: int,
    body_start: int,
    col_offsets: ia,
    col_rows: ia,
    col_slots: ia,
    diag_slots: ia,
    chunk_first: int,
    chunk_last: int,
    chunk_begin: ia,
    chunk_end: ia,
    lane: int,
):
    value = _subst_rhs(delta, row, lane)
    for chunk in range(chunk_first, chunk_last):
        partial = _subst_entries(
            vals,
            delta,
            row,
            body_start,
            col_rows,
            col_slots,
            chunk_begin[chunk],
            chunk_end[chunk],
            0.0,
            lane,
            True,
            True,
        )
        value = value - partial
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


@wp.func
def front_short(layout: Sparse, vals: fa, rhs: fa, sol: fa, piv: ia, row: int, base: int, lane: int):
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
    row_front(layout, vals, sol, piv, row, lane, value)


@wp.func
def _split_reduce(scratch: wp.array[wp.float32], base: int, orig_threads: int, tid: int):
    _block_sync()
    count = orig_threads
    while count > 1:
        half = (count + 1) // 2
        for scalar in range(tid, 6 * (count - half), wp.block_dim()):
            address = base + scalar
            scratch[address] = scratch[address] + scratch[address + 6 * half]
        _block_sync()
        count = half


@wp.func
def forward_split(
    vals: wp.array[wp.float32],
    rhs: wp.array[wp.float32],
    delta: wp.array[wp.float32],
    row: int,
    body_start: int,
    row_offsets: wp.array[wp.int32],
    cols: wp.array[wp.int32],
    diag_slots: wp.array[wp.int32],
    scratch: wp.array[wp.float32],
    art: int,
    orig_threads: int,
    tid: int,
):
    base = 6 * art * orig_threads
    for logical_lane in range(tid, orig_threads, wp.block_dim()):
        partial = vec6f()
        for slot in range(row_offsets[row] + logical_lane, row_offsets[row + 1], orig_threads):
            col = body_start + cols[slot]
            if col < row:
                partial = partial + _load_block66_scalar(vals, slot) * _scalar_vec_load(delta, col)
        for component in range(6):
            scratch[base + 6 * logical_lane + component] = partial[component]
    _split_reduce(scratch, base, orig_threads, tid)
    if tid == 0:
        residual = _scalar_vec_load(rhs, row) - _scalar_vec_load(scratch, art * orig_threads)
        factor = _load_block66_scalar(vals, diag_slots[row])
        result = _solve_lower66(factor, residual)
        for component in range(6):
            delta[6 * row + component] = result[component]
    _block_sync()


@wp.func
def backward_split(
    vals: wp.array[wp.float32],
    delta: wp.array[wp.float32],
    row: int,
    body_start: int,
    col_offsets: wp.array[wp.int32],
    col_rows: wp.array[wp.int32],
    col_slots: wp.array[wp.int32],
    diag_slots: wp.array[wp.int32],
    scratch: wp.array[wp.float32],
    art: int,
    orig_threads: int,
    tid: int,
):
    base = 6 * art * orig_threads
    for logical_lane in range(tid, orig_threads, wp.block_dim()):
        partial = vec6f()
        for entry in range(col_offsets[row] + logical_lane, col_offsets[row + 1], orig_threads):
            col = body_start + col_rows[entry]
            block = _load_block66_scalar(vals, col_slots[entry])
            partial = partial + wp.transpose(block) * _scalar_vec_load(delta, col)
        for component in range(6):
            scratch[base + 6 * logical_lane + component] = partial[component]
    _split_reduce(scratch, base, orig_threads, tid)
    if tid == 0:
        residual = _scalar_vec_load(delta, row) - _scalar_vec_load(scratch, art * orig_threads)
        factor = _load_block66_scalar(vals, diag_slots[row])
        result = _solve_upper66_from_lower(factor, residual)
        for component in range(6):
            delta[6 * row + component] = result[component]
    _block_sync()


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
    short_count: int,
    long_begin: int,
):
    tid = wp.tid()
    work = tid // 128
    local = tid & 127
    lane = tid & 31
    row = int(0)
    base = int(0)
    first = int(0)
    last = int(0)
    if work < short_count:
        idx = short_begin + work
        row = tasks.short_ids[idx]
        base = tasks.short_aux[idx]
    else:
        idx = long_begin + work - short_count
        row = tasks.long_ids[idx]
        first = tasks.long_chunk_offsets[idx]
        last = tasks.long_chunk_offsets[idx + 1]
        base = tasks.chunk_aux[first]
    if local < 32:
        value = _subst_rhs(rhs, row, lane)
        if work < short_count:
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
        else:
            for chunk in range(first, last):
                value -= _subst_entries(
                    vals,
                    sol,
                    row,
                    base,
                    layout.cols,
                    layout.cols,
                    tasks.chunk_begin[chunk],
                    tasks.chunk_end[chunk],
                    0.0,
                    lane,
                    False,
                    True,
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
def back_sub(layout: Sparse, tasks: Tasks, vals: fa, sol: fa, short_begin: int, short_count: int, long_begin: int):
    tid = wp.tid()
    work = tid // 32
    lane = tid % 32
    if work < short_count:
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
    else:
        idx = long_begin + work - short_count
        first = tasks.long_chunk_offsets[idx]
        last = tasks.long_chunk_offsets[idx + 1]
        backward_long(
            vals,
            sol,
            tasks.long_ids[idx],
            tasks.chunk_aux[first],
            layout.colptr,
            layout.rows,
            layout.slots,
            layout.diag,
            first,
            last,
            tasks.chunk_begin,
            tasks.chunk_end,
            lane,
        )


@wp.kernel
def cooperative_tail(
    layout: Sparse, vals: fa, rhs: fa, sol: fa, piv: ia, scratch: fa, cut: int, split: int, orig_threads: int
):
    tid = wp.tid()
    threads = wp.block_dim()
    art = tid // threads
    local = tid % threads
    group = local // 32
    groups = threads // 32
    lane = local % 32
    base = layout.base[art]
    end = layout.artlev[art + 1]
    start = wp.min(layout.artlev[art] + cut, end)
    for lev in range(start, end):
        r0 = layout.levrow[lev]
        r1 = layout.levrow[lev + 1]
        if split <= 0:
            for row in range(r0 + group, r1, groups):
                front_short(layout, vals, rhs, sol, piv, row, base, lane)
            _block_sync()
            for ident in range(layout.levupd[lev] + group, layout.levupd[lev + 1], groups):
                update(
                    vals, layout.dest[ident], layout.ptr[ident], layout.ptr[ident + 1], layout.left, layout.right, lane
                )
            _block_sync()
            continue
        for row in range(r0 + group, r1, groups):
            chol(vals, layout.diag[row], piv, lane)
        _block_sync()
        for task in range(layout.levcol[lev] + group, layout.levcol[lev + 1], groups):
            column(vals, layout.cslot[task], layout.cdiag[task], lane)
        for row in range(r0 + group, r1, groups):
            if split <= 0 or layout.rowptr[row + 1] - layout.rowptr[row] <= split:
                forward(vals, rhs, sol, row, base, layout.rowptr, layout.cols, layout.diag, lane)
        if split > 0:
            for row in range(r0, r1):
                if layout.rowptr[row + 1] - layout.rowptr[row] > split:
                    forward_split(
                        vals,
                        rhs,
                        sol,
                        row,
                        base,
                        layout.rowptr,
                        layout.cols,
                        layout.diag,
                        scratch,
                        art,
                        orig_threads,
                        local,
                    )
        _block_sync()
        for ident in range(layout.levupd[lev] + group, layout.levupd[lev + 1], groups):
            update(vals, layout.dest[ident], layout.ptr[ident], layout.ptr[ident + 1], layout.left, layout.right, lane)
        _block_sync()
    for off in range(end - start):
        lev = end - off - 1
        r0 = layout.levrow[lev]
        r1 = layout.levrow[lev + 1]
        for row in range(r0 + group, r1, groups):
            if split <= 0 or layout.colptr[row + 1] - layout.colptr[row] <= split:
                backward(vals, sol, row, base, layout.colptr, layout.rows, layout.slots, layout.diag, lane)
        if split > 0:
            for row in range(r0, r1):
                if layout.colptr[row + 1] - layout.colptr[row] > split:
                    backward_split(
                        vals,
                        sol,
                        row,
                        base,
                        layout.colptr,
                        layout.rows,
                        layout.slots,
                        layout.diag,
                        scratch,
                        art,
                        orig_threads,
                        local,
                    )
        _block_sync()


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    if (data.strides[0] == sizeof(float) && !(reinterpret_cast<size_t>(data.data) & 15u)) {
        const float4 words = *reinterpret_cast<const float4*>(data.data + start);
        return wp::vec4(words.x, words.y, words.z, words.w);
    }
#endif
    return wp::vec4(wp::index(data, start), wp::index(data, start+1),
                    wp::index(data, start+2), wp::index(data, start+3));
""")
def _schur_load4(data: fa, start: int) -> wp.vec4:
    """Four logical entries. All call-site starts are multiples of four floats."""
    pass


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    if (data.strides[0] == sizeof(float) && !(reinterpret_cast<size_t>(data.data) & 7u)) {
        const float2 words = *reinterpret_cast<const float2*>(data.data + start);
        return wp::vec2(words.x, words.y);
    }
#endif
    return wp::vec2(wp::index(data, start), wp::index(data, start+1));
""")
def _schur_load2(data: fa, start: int) -> wp.vec2:
    """Two logical entries, starting at an even float index."""
    pass


@wp.func
def _schur_outer(acc: wp.vec4, left0: float, left1: float, right0: float, right1: float):
    return wp.vec4(acc[0] + left0 * right0, acc[1] + left0 * right1, acc[2] + left1 * right0, acc[3] + left1 * right1)


@wp.func
def _schur_tile_dot(layout: Sparse, vals: fa, term: int, row_pair: int, col_pair: int):
    """One thread owns four outputs. Two contiguous operand rows fit in three float4s."""
    left = 36 * layout.left[term] + 12 * row_pair
    right = 36 * layout.right[term] + 12 * col_pair
    left_head = _schur_load4(vals, left)
    left_mid = _schur_load4(vals, left + 4)
    left_tail = _schur_load4(vals, left + 8)
    right_head = _schur_load4(vals, right)
    right_mid = _schur_load4(vals, right + 4)
    right_tail = _schur_load4(vals, right + 8)
    dot = wp.vec4()
    dot = _schur_outer(dot, left_head[0], left_mid[2], right_head[0], right_mid[2])
    dot = _schur_outer(dot, left_head[1], left_mid[3], right_head[1], right_mid[3])
    dot = _schur_outer(dot, left_head[2], left_tail[0], right_head[2], right_tail[0])
    dot = _schur_outer(dot, left_head[3], left_tail[1], right_head[3], right_tail[1])
    dot = _schur_outer(dot, left_mid[0], left_tail[2], right_mid[0], right_tail[2])
    return _schur_outer(dot, left_mid[1], left_tail[3], right_mid[1], right_tail[3])


@wp.func
def _schur_short(layout: Sparse, tasks: Tasks, vals: fa, short_begin: int, work: int):
    task = work // 9
    tile = work % 9
    row_pair = tile // 3
    col_pair = tile % 3
    ident = tasks.short_ids[short_begin + task]
    first = layout.ptr[ident]
    last = layout.ptr[ident + 1]
    dst = 36 * layout.dest[ident] + 12 * row_pair + 2 * col_pair
    upper = _schur_load2(vals, dst)
    lower = _schur_load2(vals, dst + 6)
    value = wp.vec4(upper[0], upper[1], lower[0], lower[1])
    if first < last:
        value = value - _schur_tile_dot(layout, vals, first, row_pair, col_pair)
        if first + 1 < last:
            value = value - _schur_tile_dot(layout, vals, first + 1, row_pair, col_pair)
            for term in range(first + 2, last):
                value = value - _schur_tile_dot(layout, vals, term, row_pair, col_pair)
    vals[dst] = value[0]
    vals[dst + 1] = value[1]
    vals[dst + 6] = value[2]
    vals[dst + 7] = value[3]


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffu << (threadIdx.x & 24u), value, source, 8);
#else
    return value;
#endif
""")
def _schur_gather8(value: float, source: int) -> float:
    """Only the eight lanes owning this chunk must participate."""
    pass


@wp.func
def _schur_row_dot(layout: Sparse, vals: fa, term: int, row: int, col: int):
    left = 36 * layout.left[term] + 6 * row
    right = 36 * layout.right[term] + 6 * col
    accum = float(0.0)
    for pair in range(3):
        lhs = _schur_load2(vals, left + 2 * pair)
        rhs = _schur_load2(vals, right + 2 * pair)
        accum = accum + lhs[0] * rhs[0]
        accum = accum + lhs[1] * rhs[1]
    return accum


@wp.func
def _schur_long(layout: Sparse, tasks: Tasks, vals: fa, long_begin: int, tid: int):
    """One warp per destination scalar; four chunks by eight terms in flight."""
    warp_id = tid // 32
    index = long_begin + warp_id // 36
    elem = warp_id % 36
    lane = tid & 31
    cohort = lane >> 3
    member = lane & 7
    row = elem // 6
    col = elem % 6
    dst = 36 * layout.dest[tasks.long_ids[index]] + elem
    first = tasks.long_chunk_offsets[index]
    last = tasks.long_chunk_offsets[index + 1]
    value = vals[dst]
    for batch in range(first, last, 4):
        chunk = batch + cohort
        partial = float(0.0)
        if chunk < last:
            begin = tasks.chunk_begin[chunk]
            end = tasks.chunk_end[chunk]
            for base in range(begin, end, 8):
                term = base + member
                dot = float(0.0)
                if term < end:
                    dot = _schur_row_dot(layout, vals, term, row, col)
                for source in range(8):
                    if base + source < end:
                        partial = partial + _schur_gather8(dot, source)
        for part in range(4):
            if batch + part < last:
                value = value - shfl(partial, 8 * part)
    if lane == 0:
        vals[dst] = value


@wp.kernel
def split_updates(
    layout: Sparse, tasks: Tasks, vals: fa, short_begin: int, short_count: int, long_begin: int, long_count: int
):
    tid = wp.tid()
    prefix = long_count * (36 * 32)
    if tid < prefix:
        _schur_long(layout, tasks, vals, long_begin, tid)
    elif tid - prefix < 9 * short_count:
        _schur_short(layout, tasks, vals, short_begin, tid - prefix)


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
    tasks.long_ids = phase.long
    tasks.long_chunk_offsets = phase.long_chunk_offsets
    tasks.chunk_begin = phase.chunk_begin
    tasks.chunk_end = phase.chunk_end
    tasks.chunk_aux = phase.chunk_aux
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
    sparse.base = layout.articulation_body_offsets
    sparse.rowptr = layout.articulation_block_row_offsets
    sparse.cols = layout.articulation_block_cols
    sparse.colptr = layout.articulation_block_col_offsets
    sparse.rows = layout.articulation_block_col_rows
    sparse.slots = layout.articulation_block_col_slots
    sparse.diag = layout.articulation_diag_slots
    levels = layout.levels
    sparse.artlev = levels.articulation_level_offsets
    sparse.levrow = levels.level_body_offsets
    sparse.levcol = levels.level_column_offsets
    sparse.levupd = levels.level_update_offsets
    sparse.cslot = levels.level_column_slots
    sparse.cdiag = levels.level_column_diag_slots
    sparse.dest = levels.level_update_dst_slots
    sparse.ptr = levels.level_update_term_offsets
    sparse.left = levels.level_update_left_slots
    sparse.right = levels.level_update_right_slots

    def record(kernel, dim, args):
        if dim:
            bindings = [(index, arg.value) for index, arg in enumerate(args) if isinstance(arg, _Buffer)]
            schedule.commands.append((kernel, dim, args, bindings))

    fwd, upd, bwd = (_bind_phase(phase) for phase in (plan.forward, plan.update, plan.backward))
    val, rhs, delta, piv = _Buffer
    for level in range(cut):
        first, count, *_ = plan.forward.level_range(level)
        if level == 0:
            record(first_front, 32 * count, [sparse, fwd, val, rhs, delta, piv, first])
        else:
            record(broad_front, 128 * count, [sparse, fwd, val, rhs, delta, piv, first, count, 0])
        short, shorts, _, _, long, longs = plan.update.level_range(level)
        record(split_updates, 1152 * longs + 9 * shorts, [sparse, upd, val, short, shorts, long, longs])

    record(
        cooperative_tail,
        128 * plan.articulation_count,
        [sparse, val, rhs, delta, piv, plan.tail_partial, cut, plan.tail_split_entries, plan.tail_block_dim],
    )
    for level in reversed(range(cut)):
        first, count, *_ = plan.backward.level_range(level)
        if count >= 256:
            record(short_leading_back, 4 * count, [sparse, bwd, val, delta, first, count])
        else:
            record(back_sub, 32 * count, [sparse, bwd, val, delta, first, count, 0])
    return schedule
