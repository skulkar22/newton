# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA warp-cooperative front, Schur update and substitution kernels for rigid sparse levels.

The static host schedule binds Newton's layout and allocates the selected algorithm's workspace
when the level plan is built. Each solve runs its numeric and symbolic kernels on the caller's stream.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import warp as wp

if TYPE_CHECKING:
    from .rigid_sparse_articulation import RigidArticulationSparseLayout
    from .rigid_sparse_level_launch_kernels import LevelLaunchPhaseTasks, RigidArticulationLevelLaunchPlan

wp.set_module_options({"block_dim": 128, "enable_backward": False, "fast_math": False, "fuse_fp": True})

ia = wp.array[wp.int32]
fa = wp.array[wp.float32]
eight = wp.types.vector(length=8, dtype=wp.float32)


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffffu, value, src);
#else
    return value;
#endif
""")
def shfl(value: float, src: int) -> float:
    pass


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
""")
def sync_block():
    pass


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __syncwarp();
#endif
""")
def sync_warp():
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
        sync_warp()
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
def _split_vec_load(data: wp.array[wp.float32], index: int):
    vector = wp.spatial_vector()
    for component in range(6):
        vector[component] = data[6 * index + component]
    return vector


@wp.func
def _split_block_load(vals: wp.array[wp.float32], slot: int):
    block = wp.spatial_matrix()
    for block_row in range(6):
        for block_col in range(6):
            block[block_row, block_col] = vals[36 * slot + 6 * block_row + block_col]
    return block


@wp.func
def _split_reduce(scratch: wp.array[wp.float32], base: int, orig_threads: int, tid: int):
    sync_block()
    count = orig_threads
    while count > 1:
        half = (count + 1) // 2
        for scalar in range(tid, 6 * (count - half), wp.block_dim()):
            address = base + scalar
            scratch[address] = scratch[address] + scratch[address + 6 * half]
        sync_block()
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
        partial = wp.spatial_vector()
        for slot in range(row_offsets[row] + logical_lane, row_offsets[row + 1], orig_threads):
            col = body_start + cols[slot]
            if col < row:
                partial = partial + _split_block_load(vals, slot) * _split_vec_load(delta, col)
        for component in range(6):
            scratch[base + 6 * logical_lane + component] = partial[component]
    _split_reduce(scratch, base, orig_threads, tid)
    if tid == 0:
        residual = _split_vec_load(rhs, row) - _split_vec_load(scratch, art * orig_threads)
        factor = _split_block_load(vals, diag_slots[row])
        result = wp.spatial_vector()
        for component in range(6):
            value = residual[component]
            for prior in range(6):
                if prior < component:
                    value = value - factor[component, prior] * result[prior]
            result[component] = value / factor[component, component]
        for component in range(6):
            delta[6 * row + component] = result[component]
    sync_block()


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
        partial = wp.spatial_vector()
        for entry in range(col_offsets[row] + logical_lane, col_offsets[row + 1], orig_threads):
            col = body_start + col_rows[entry]
            block = _split_block_load(vals, col_slots[entry])
            partial = partial + wp.transpose(block) * _split_vec_load(delta, col)
        for component in range(6):
            scratch[base + 6 * logical_lane + component] = partial[component]
    _split_reduce(scratch, base, orig_threads, tid)
    if tid == 0:
        residual = _split_vec_load(delta, row) - _split_vec_load(scratch, art * orig_threads)
        factor = _split_block_load(vals, diag_slots[row])
        result = wp.spatial_vector()
        for reverse_component in range(6):
            component = 5 - reverse_component
            value = residual[component]
            for reverse_prior in range(6):
                prior = 5 - reverse_prior
                if prior > component:
                    value = value - factor[prior, component] * result[prior]
            result[component] = value / factor[component, component]
        for component in range(6):
            delta[6 * row + component] = result[component]
    sync_block()


@wp.kernel
def first_front(layout: Sparse, tasks: Tasks, vals: fa, rhs: fa, sol: fa, piv: ia, first: int):
    tid = wp.tid()
    lane = tid & 31
    row = tasks.short_ids[first + tid // 32]
    row_front(layout, vals, sol, piv, row, lane, _subst_rhs(rhs, row, lane))


@wp.kernel
def eliminate(
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
    work = tid // 32
    lane = tid % 32
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
    if work < short_count:
        front_short(layout, vals, rhs, sol, piv, row, base, lane)
    else:
        value = _subst_rhs(rhs, row, lane)
        for chunk in range(first, last):
            partial = _subst_entries(
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
            value -= partial
        row_front(layout, vals, sol, piv, row, lane, value)


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
        sync_block()
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
def apply_updates(layout: Sparse, tasks: Tasks, vals: fa, short_begin: int, short_count: int, long_begin: int):
    tid = wp.tid()
    work = tid // 32
    lane = tid % 32
    if work < short_count:
        ident = tasks.short_ids[short_begin + work]
        update(vals, layout.dest[ident], layout.ptr[ident], layout.ptr[ident + 1], layout.left, layout.right, lane)
    else:
        idx = long_begin + work - short_count
        dst = layout.dest[tasks.long_ids[idx]]
        for elem in range(lane, 36, 32):
            value = vals[dst * 36 + elem]
            for chunk in range(tasks.long_chunk_offsets[idx], tasks.long_chunk_offsets[idx + 1]):
                total = float(0.0)
                for term in range(tasks.chunk_begin[chunk], tasks.chunk_end[chunk]):
                    left = layout.left[term] * 36 + (elem // 6) * 6
                    right = layout.right[term] * 36 + (elem % 6) * 6
                    dot = float(0.0)
                    for comp in range(6):
                        dot += vals[left + comp] * vals[right + comp]
                    total += dot
                value -= total
            vals[dst * 36 + elem] = value


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
            sync_block()
            for ident in range(layout.levupd[lev] + group, layout.levupd[lev + 1], groups):
                update(
                    vals, layout.dest[ident], layout.ptr[ident], layout.ptr[ident + 1], layout.left, layout.right, lane
                )
            sync_block()
            continue
        for row in range(r0 + group, r1, groups):
            chol(vals, layout.diag[row], piv, lane)
        sync_block()
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
        sync_block()
        for ident in range(layout.levupd[lev] + group, layout.levupd[lev + 1], groups):
            update(vals, layout.dest[ident], layout.ptr[ident], layout.ptr[ident + 1], layout.left, layout.right, lane)
        sync_block()
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
        sync_block()


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __shared__ float storage[1728];
    return wp::array_t<float>(storage, 1728);
#else
    return wp::array_t<float>();
#endif
""")
def dense_tail_tile() -> fa:
    pass


@wp.func
def dense_tail_sparse(layout: Sparse, vals: fa, piv: ia, start: int, stop: int, tid: int):
    group = tid >> 5
    lane = tid & 31
    for level in range(start, stop):
        for row in range(layout.levrow[level] + group, layout.levrow[level + 1], 4):
            chol(vals, layout.diag[row], piv, lane)
        sync_block()
        for task in range(layout.levcol[level] + group, layout.levcol[level + 1], 4):
            column(vals, layout.cslot[task], layout.cdiag[task], lane)
        sync_block()
        for task in range(layout.levupd[level] + group, layout.levupd[level + 1], 4):
            update(vals, layout.dest[task], layout.ptr[task], layout.ptr[task + 1], layout.left, layout.right, lane)
        sync_block()


@wp.func
def dense_tail_tri_row(index: int):
    return int((wp.sqrt(float(8 * index + 1)) - 1.0) * 0.5)


@wp.kernel
def dense_tail_prepare(layout: Sparse, work: ia, cut: int, size: int):
    """One CTA per level verifies the actual columns/Schur terms before using a dense suffix."""
    tid = wp.tid()
    row = tid >> 7
    local = tid & 127
    stop = layout.artlev[1]
    start = wp.min(layout.artlev[0] + cut, stop)
    first = layout.levrow[start]
    row_map = 1 + row * size
    if stop - start != size or layout.levrow[stop] != first + size:
        if local == 0:
            work[row_map + row] = -1
        return

    level = start + row
    actual = first + row
    head = layout.rowptr[actual]
    end = layout.rowptr[actual + 1]
    base = end - 1 - row
    remaining = size - 1 - row
    column_begin = layout.levcol[level]
    update_begin = layout.levupd[level]
    updates = remaining * (remaining + 1) // 2
    if (
        base < head
        or layout.levrow[level] != actual
        or layout.diag[actual] != end - 1
        or layout.levcol[level + 1] - column_begin != remaining
        or layout.levupd[level + 1] - update_begin != updates
    ):
        if local == 0:
            work[row_map + row] = -1
        return

    valid = bool(True)
    for slot in range(head + local, end, 128):
        col = layout.base[0] + layout.cols[slot]
        if (slot < base and col >= first) or (slot >= base and col != first + slot - base):
            valid = False
    if local < size:
        work[row_map + local] = wp.where(local <= row, base + local, -1)
    if layout.colptr[actual + 1] - layout.colptr[actual] != remaining:
        valid = False
    if local < remaining:
        outer = row + 1 + local
        source = layout.rowptr[first + outer + 1] - 1 - outer + row
        if layout.cslot[column_begin + local] != source or layout.cdiag[column_begin + local] != end - 1:
            valid = False
        entry = layout.colptr[actual] + local
        if entry >= layout.colptr[actual + 1]:
            valid = False
        elif layout.base[0] + layout.rows[entry] != first + outer or layout.slots[entry] != source:
            valid = False
    for offset in range(local, updates, 128):
        outer = dense_tail_tri_row(offset)
        inner = offset - outer * (outer + 1) // 2
        dst_row = row + 1 + outer
        dst_col = row + 1 + inner
        left = layout.rowptr[first + dst_row + 1] - 1 - dst_row
        right = layout.rowptr[first + dst_col + 1] - 1 - dst_col
        task = update_begin + offset
        term = layout.ptr[task]
        if layout.ptr[task + 1] != term + 1 or layout.dest[task] != left + dst_col:
            valid = False
        elif layout.left[term] != left + row or layout.right[term] != right + row:
            valid = False
    sync_block()
    if not valid:
        wp.atomic_min(work, row_map + row, -1)


@wp.kernel
def dense_tail_substitute(
    layout: Sparse, vals: fa, rhs: fa, sol: fa, scratch: fa, cut: int, split: int, orig_threads: int
):
    """After all factors are published, use the original CSR/CSC substitution ordering."""
    tid = wp.tid()
    lane = tid & 31
    group = tid >> 5
    base = layout.base[0]
    stop = layout.artlev[1]
    start = wp.min(layout.artlev[0] + cut, stop)
    for level in range(start, stop):
        first = layout.levrow[level]
        last = layout.levrow[level + 1]
        for row in range(first + group, last, 4):
            if split <= 0 or layout.rowptr[row + 1] - layout.rowptr[row] <= split:
                forward(vals, rhs, sol, row, base, layout.rowptr, layout.cols, layout.diag, lane)
        if split > 0:
            for row in range(first, last):
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
                        0,
                        orig_threads,
                        tid,
                    )
        sync_block()
    for reverse in range(stop - start):
        level = stop - reverse - 1
        first = layout.levrow[level]
        last = layout.levrow[level + 1]
        for row in range(first + group, last, 4):
            if split <= 0 or layout.colptr[row + 1] - layout.colptr[row] <= split:
                backward(vals, sol, row, base, layout.colptr, layout.rows, layout.slots, layout.diag, lane)
        if split > 0:
            for row in range(first, last):
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
                        0,
                        orig_threads,
                        tid,
                    )
        sync_block()


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __shared__ float tile[294];
    return wp::array_t<float>(tile,294);
#else
    return wp::array_t<float>();
#endif
""")
def sub_tile() -> fa:
    pass


def dense_tail_size(plan, cut):
    """Returns zero for schedules that must stay on the general tail path."""
    count = plan.level_count
    rows = plan.rows_per_level
    if (
        plan.articulation_count == 1
        and 0 <= cut < count
        and count - cut <= 48
        and len(rows) == count
        and all(value == 1 for value in rows[cut:])
    ):
        return count - cut
    return 0


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __shared__ float packed[4*48*36];
    return wp::array_t<float>(packed,4*48*36);
#else
    return wp::array_t<float>();
#endif
""")
def substitution_stage() -> fa:
    pass


@wp.func
def external_dot(layout: Sparse, vals: fa, sol: fa, base: int, slot: int, component: int):
    address = slot * 36 + 6 * component
    neighbor = 6 * (base + layout.cols[slot])
    total = float(0.0)
    for part in range(6):
        total += vals[address + part] * sol[neighbor + part]
    return total


@wp.func
def staged_term(factors: fa, state: fa, base: int, other: int, component: int, upper: bool):
    address = 36 * (base + other) + 6 * component
    stride = int(1)
    if upper:
        address = 36 * (base + other) + component
        stride = 6
    total = float(0.0)
    for index in range(6):
        total += factors[address + stride * index] * state[6 * other + index]
    return total


@wp.func
def substitution_run(
    layout: Sparse,
    vals: fa,
    rhs: fa,
    boundary: fa,
    work: ia,
    first: int,
    size: int,
    split: int,
    factors: fa,
    state: fa,
    tid: int,
    upper: bool,
):
    component = wp.min(tid // 16, 5)
    member = tid & 15
    lane = tid & 31
    for panel in range(0, size, 4):
        count = wp.min(4, size - panel)
        for quad in range(tid, 9 * size * count, 128):
            block = quad // 9
            part = block // size
            other = block % size
            row = wp.where(upper, size - 1 - panel - part, panel + part)
            if (upper and other >= row) or (not upper and other <= row):
                slot = work[1 + wp.max(row, other) * size] + wp.min(row, other)
                _small_copy4(factors, 4 * quad, vals, 36 * slot + 4 * (quad % 9))
        sync_block()
        for part in range(count):
            row = wp.where(upper, size - 1 - panel - part, panel + part)
            base = part * size
            previous = int(0)
            span = row
            start = int(0)
            extent = int(0)
            initial = float(0.0)
            if upper:
                start = row + 1
                span = size - start
                extent = span
                initial = state[6 * row + component]
            else:
                previous = work[1 + row * size] - layout.rowptr[first + row]
                extent = previous + span + 1
                initial = rhs[6 * (first + row) + component]
            result = initial
            if split <= 0 or extent <= split:
                if not upper:
                    result = boundary[128 * (6 * row + component)]
                for block in range(0, span, 16):
                    term = float(0.0)
                    if block + member < span:
                        term = staged_term(factors, state, base, start + block + member, component, upper)
                    for source in range(wp.min(16, span - block)):
                        result -= shfl(term, (lane & 16) + source)
            else:
                partial = eight()
                for stripe in range(8):
                    logical = member + 16 * stripe
                    value = float(0.0)
                    if not upper:
                        value = boundary[128 * (6 * row + component) + logical]
                    pos = (logical - previous) & 127
                    if pos < span:
                        value += staged_term(factors, state, base, start + pos, component, upper)
                    partial[stripe] = value
                value = ((partial[0] + partial[4]) + (partial[2] + partial[6])) + (
                    (partial[1] + partial[5]) + (partial[3] + partial[7])
                )
                for distance in range(4):
                    value += shfl(value, lane ^ (8 >> distance))
                result -= value
            if tid < 96 and member == 0:
                state[288 + component] = result
            sync_block()
            if tid < 32:
                _subst_finish(factors, state, row, base + row, state[288 + wp.min(lane, 5)], lane, upper)
            sync_block()


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __shared__ int storage[1925];
    return wp::array_t<int>(storage + offset, count);
#else
    return wp::array_t<int>();
#endif
""")
def small_ints(offset: int, count: int) -> ia:
    pass


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __shared__ float storage[4800];
    return wp::array_t<float>(storage + offset, count);
#else
    return wp::array_t<float>();
#endif
""")
def small_floats(offset: int, count: int) -> fa:
    pass


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    int total = width;
    #pragma unroll
    for (int dist = 1; dist < 32; dist *= 2) {
        int other = __shfl_up_sync(0xffffffffu, total, dist);
        if (lane >= dist) total += other;
    }
    return total - width;
#else
    return 0;
#endif
""")
def small_exclusive(width: int, lane: int) -> int:
    pass


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __syncthreads_or(bad);
#else
    return bad;
#endif
""")
def small_any(bad: int) -> int:
    pass


@wp.func
def small_layout() -> Sparse:
    """Views into one CTA's arena. Slots [1285,1797) and [1797,1925) are the two slot maps."""
    packed = Sparse()
    packed.rowptr = small_ints(0, 33)
    packed.colptr = small_ints(33, 33)
    packed.diag = small_ints(66, 32)
    packed.levrow = small_ints(98, 17)
    packed.levupd = small_ints(115, 17)
    packed.cols = small_ints(132, 128)
    packed.rows = small_ints(260, 128)
    packed.slots = small_ints(388, 128)
    packed.dest = small_ints(516, 128)
    packed.ptr = small_ints(644, 129)
    packed.left = small_ints(773, 256)
    packed.right = small_ints(1029, 256)
    return packed


@wp.func
def small_map(mapping: ia, slot: int, first: int, stop: int) -> int:
    result = int(-1)
    if first <= slot and slot < stop:
        result = mapping[slot - first]
    return result


@wp.func
def small_updates(packed: Sparse, blocks: fa, begin: int, end: int, tid: int):
    """One thread per destination scalar; retain the two nested accumulation orders."""
    for job in range(36 * begin + tid, 36 * end, 128):
        task = job // 36
        scalar = job % 36
        dst = 36 * packed.dest[task] + scalar
        value = blocks[dst]
        for term in range(packed.ptr[task], packed.ptr[task + 1]):
            left = 36 * packed.left[term] + 6 * (scalar // 6)
            right = 36 * packed.right[term] + 6 * (scalar % 6)
            dot = float(0.0)
            for inner in range(6):
                dot += blocks[left + inner] * blocks[right + inner]
            value -= dot
        blocks[dst] = value


@wp.func
def small_unpacked(layout: Sparse, vals: fa, rhs: fa, sol: fa, piv: ia, base: int, start: int, stop: int, tid: int):
    """Unsplit fallback for a whole CTA. Executes the explicit column task schedule."""
    dense_tail_sparse(layout, vals, piv, start, stop, tid)
    group = tid >> 5
    lane = tid & 31
    for level in range(start, stop):
        for row in range(layout.levrow[level] + group, layout.levrow[level + 1], 4):
            forward(vals, rhs, sol, row, base, layout.rowptr, layout.cols, layout.diag, lane)
        sync_block()
    for reverse in range(stop - start):
        level = stop - reverse - 1
        for row in range(layout.levrow[level] + group, layout.levrow[level + 1], 4):
            backward(vals, sol, row, base, layout.colptr, layout.rows, layout.slots, layout.diag, lane)
        sync_block()


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffu << (threadIdx.x & 24u),data,source,8);
#else
    return data;
#endif
""")
def group_share(data: float, source: int) -> float:
    pass


def small_tail_launch(launch, plan, layout, vals, rhs, sol, piv, cut):
    """Solve short tails in shared memory; each block validates its articulation's footprint."""
    count = plan.level_count
    arts = plan.articulation_count
    rows = plan.rows_per_level
    if (
        arts <= 0
        or not 0 <= cut < count
        or count - cut > 16
        or len(rows) != count
        or plan.tail_split_entries > 0
        or sum(rows[cut:]) > 32 * arts
    ):
        return False
    launch(small_sparse_tail, 128 * arts, [layout, vals, rhs, sol, piv, cut])
    return True


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


@wp.kernel
def short_updates(layout: Sparse, tasks: Tasks, vals: fa, short_begin: int):
    _schur_short(layout, tasks, vals, short_begin, wp.tid())


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


def launch_schur(launch, layout, tasks, vals, short_begin, short_count, long_begin, long_count):
    """Schedule short and split Schur updates in one launch per level."""
    if long_count:
        launch(
            split_updates,
            (36 * 32) * long_count + 9 * short_count,
            [layout, tasks, vals, short_begin, short_count, long_begin, long_count],
        )
    elif short_count:
        launch(short_updates, 9 * short_count, [layout, tasks, vals, short_begin])


@wp.kernel
def update_and_gather(
    layout: Sparse,
    tasks: Tasks,
    vals: fa,
    rhs: fa,
    sol: fa,
    short_begin: int,
    long_begin: int,
    long_count: int,
    width: int,
    arts: int,
    cut: int,
):
    tid = wp.tid()
    gather_size = 32 * arts * width
    if tid < gather_size:
        warp_id = tid >> 5
        lane = tid & 31
        art = warp_id // width
        end = layout.artlev[art + 1]
        start = wp.min(layout.artlev[art] + cut, end)
        first = layout.levrow[start]
        row = first + warp_id % width
        if row < layout.levrow[end]:
            result = _subst_entries(
                vals,
                sol,
                first,
                layout.base[art],
                layout.cols,
                layout.cols,
                layout.rowptr[row],
                layout.rowptr[row + 1],
                _subst_rhs(rhs, row, lane),
                lane,
                False,
                False,
            )
            if lane < 6:
                sol[6 * row + lane] = result
    elif tid < gather_size + 1152 * long_count:
        _schur_long(layout, tasks, vals, long_begin, tid - gather_size)
    else:
        _schur_short(layout, tasks, vals, short_begin, tid - gather_size - 1152 * long_count)


@wp.func
def _branch_read_row(values: fa, address: int):
    head = _schur_load2(values, address)
    middle = _schur_load2(values, address + 2)
    tail = _schur_load2(values, address + 4)
    return wp.spatial_vector(head[0], head[1], middle[0], middle[1], tail[0], tail[1])


@wp.func
def _branch_wide(layout: Sparse, vals: fa, residual: fa, sol: fa, piv: ia, row: int, local: int):
    """Four replicated diagonal-register groups solve up to 4*(4+5) block columns at once."""
    lane = local & 31
    warp = local >> 5
    group = lane // 6
    component = lane % 6
    begin = layout.colptr[row]
    end = layout.colptr[row + 1]
    diag = layout.diag[row]
    diagonal = lane < 6
    first_entry = begin + 4 * warp + group - 1
    first_column = group > 0 and group <= 4 and first_entry < end
    second_entry = begin + 16 + 5 * warp + group
    second_column = group < 5 and second_entry < end
    address = 36 * diag + 6 * component
    extra_address = int(0)
    primary = wp.spatial_vector()
    extra = wp.spatial_vector()
    if first_column:
        address = 36 * layout.slots[first_entry] + 6 * component
    if diagonal or first_column:
        primary = _branch_read_row(vals, address)
    if local == 30:
        primary = _branch_read_row(residual, 6 * row)
    if second_column:
        extra_address = 36 * layout.slots[second_entry] + 6 * component
        extra = _branch_read_row(vals, extra_address)
    floor = wp.max(1.0e-6 * wp.abs(primary[component]), 1.0e-30)
    for step in range(6):
        value = primary[step]
        if local == step and not (value > floor):
            wp.atomic_add(piv, 0, 1)
        guarded = wp.where(value > floor, value, floor)
        pivot = wp.sqrt(shfl(guarded, step))
        solved = value / pivot
        if lane == step:
            solved = pivot
        extra_solved = extra[step] / pivot
        primary[step] = solved
        extra[step] = extra_solved
        for col in range(6):
            if col > step:
                coefficient = shfl(solved, col)
                primary[col] = primary[col] - solved * coefficient
                extra[col] = extra[col] - extra_solved * coefficient
    sync_block()
    if local < 6 or first_column:
        for col in range(6):
            value = primary[col]
            if diagonal and col > component:
                value = 0.0
            vals[address + col] = value
    if second_column:
        for col in range(6):
            vals[extra_address + col] = extra[col]
    if local == 30:
        for col in range(6):
            sol[6 * row + col] = primary[col]
    if end > begin + 36:
        sync_block()
        for entry in range(begin + 36 + warp, end, 4):
            column(vals, layout.slots[entry], diag, lane)


@wp.kernel
def branch_front(
    layout: Sparse,
    tasks: Tasks,
    vals: fa,
    residual: fa,
    sol: fa,
    piv: ia,
    long_begin: int,
    long_count: int,
    short_begin: int,
):
    tid = wp.tid()
    if tid < 128 * long_count:
        row = tasks.long_ids[long_begin + tid // 128]
        _branch_wide(layout, vals, residual, sol, piv, row, tid & 127)
    else:
        row = tasks.short_ids[short_begin + (tid - 128 * long_count) // 32]
        row_front(layout, vals, sol, piv, row, tid & 31, _subst_rhs(residual, row, tid & 31))


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __shared__ float ordered_chunks[16 * 6];
    return wp::array_t<float>(ordered_chunks, 16 * 6);
#else
    return wp::array_t<float>();
#endif
""")
def _branch_partials() -> fa:
    pass


@wp.func
def _branch_long_residual(
    layout: Sparse, tasks: Tasks, vals: fa, source: fa, sol: fa, index: int, local: int, upper: bool
) -> float:
    """Up to sixteen independent chunk sums at once; consume in original order."""
    row = tasks.long_ids[index]
    component = local & 7
    group = local >> 3
    scratch = _branch_partials()
    first = tasks.long_chunk_offsets[index]
    stop = tasks.long_chunk_offsets[index + 1]
    result = _subst_rhs(source, row, local)
    for batch in range(first, stop, 16):
        chunk = batch + group
        if chunk < stop and component < 6:
            base = tasks.chunk_aux[chunk]
            total = float(0.0)
            for entry in range(tasks.chunk_begin[chunk], tasks.chunk_end[chunk]):
                other = int(0)
                address = int(0)
                if upper:
                    other = base + layout.rows[entry]
                    address = 36 * layout.slots[entry] + component
                else:
                    other = base + layout.cols[entry]
                    address = 36 * entry + 6 * component
                if upper or other < row:
                    vector = _branch_read_row(sol, 6 * other)
                    dot = float(0.0)
                    if upper:
                        for inner in range(6):
                            dot += vals[address + inner * 6] * vector[inner]
                    else:
                        block = _branch_read_row(vals, address)
                        for inner in range(6):
                            dot += block[inner] * vector[inner]
                    total += dot
            scratch[6 * group + component] = total
        sync_block()
        if local < 6:
            for part in range(wp.min(stop - batch, 16)):
                result -= scratch[6 * part + local]
        if stop - batch > 16:
            sync_block()
    return result


@wp.kernel
def branch_update_forward(
    layout: Sparse,
    updates: Tasks,
    forwards: Tasks,
    vals: fa,
    rhs: fa,
    sol: fa,
    long_forward_begin: int,
    long_forward_count: int,
    long_update_begin: int,
    long_update_count: int,
    short_forward_begin: int,
    short_forward_count: int,
    short_update_begin: int,
):
    tid = wp.tid()
    after_long_forward = tid - 128 * long_forward_count
    after_long_update = after_long_forward - 1152 * long_update_count
    after_short_forward = after_long_update - 32 * short_forward_count
    if after_long_forward < 0:
        idx = long_forward_begin + tid // 128
        result = _branch_long_residual(layout, forwards, vals, rhs, sol, idx, tid & 127, False)
        if tid & 127 < 6:
            sol[6 * forwards.long_ids[idx] + (tid & 127)] = result
    elif after_long_update < 0:
        _schur_long(layout, updates, vals, long_update_begin, after_long_forward)
    elif after_short_forward < 0:
        lane = tid & 31
        idx = short_forward_begin + after_long_update // 32
        row = forwards.short_ids[idx]
        result = _subst_entries(
            vals,
            sol,
            row,
            forwards.short_aux[idx],
            layout.cols,
            layout.cols,
            layout.rowptr[row],
            layout.rowptr[row + 1],
            _subst_rhs(rhs, row, lane),
            lane,
            False,
            False,
        )
        if lane < 6:
            sol[6 * row + lane] = result
    else:
        _schur_short(layout, updates, vals, short_update_begin, after_short_forward)


@wp.kernel
def branch_backward(
    layout: Sparse, tasks: Tasks, vals: fa, sol: fa, long_begin: int, long_count: int, short_begin: int
):
    tid = wp.tid()
    local = tid & 127
    lane = tid & 31
    if tid < 128 * long_count:
        idx = long_begin + tid // 128
        row = tasks.long_ids[idx]
        result = _branch_long_residual(layout, tasks, vals, sol, sol, idx, local, True)
        if local < 32:
            _subst_finish(vals, sol, row, layout.diag[row], result, lane, True)
    else:
        idx = short_begin + (tid - 128 * long_count) // 32
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


def branch_enabled(plan):
    """Use the pipelined launch layout for large leading schedules with both kinds of fan-in."""
    cut = plan.leading_level_count
    if cut < 2 or plan.row_offsets[cut] - plan.row_offsets[0] < 512:
        return False
    upd = plan.update.long_offsets
    bwd = plan.backward.long_offsets
    fwd = plan.forward.long_offsets
    return upd[cut] > upd[0] and bwd[cut] > bwd[0] and fwd[1] == fwd[0]


def _branch_ranges(scheme, level):
    shorts, longs, _tasks = scheme
    return shorts[level], shorts[level + 1] - shorts[level], longs[level], longs[level + 1] - longs[level]


def branch_level(launch, layout, fwd, upd, bwd, vals, rhs, sol, piv, level, cut):
    short_row, short_rows, long_row, long_rows = _branch_ranges(bwd, level)
    residual = rhs if level == 0 else sol
    if long_rows:
        launch(
            branch_front,
            128 * long_rows + 32 * short_rows,
            [layout, bwd.tasks, vals, residual, sol, piv, long_row, long_rows, short_row],
        )
    else:
        launch(first_front, 32 * short_rows, [layout, bwd.tasks, vals, residual, sol, piv, short_row])
    short_dst, short_dsts, long_dst, long_dsts = _branch_ranges(upd, level)
    if level + 1 == cut:
        launch_schur(launch, layout, upd.tasks, vals, short_dst, short_dsts, long_dst, long_dsts)
        return
    short_fwd, short_fwds, long_fwd, long_fwds = _branch_ranges(fwd, level + 1)
    launch(
        branch_update_forward,
        128 * long_fwds + 1152 * long_dsts + 32 * short_fwds + 9 * short_dsts,
        [
            layout,
            upd.tasks,
            fwd.tasks,
            vals,
            rhs,
            sol,
            long_fwd,
            long_fwds,
            long_dst,
            long_dsts,
            short_fwd,
            short_fwds,
            short_dst,
        ],
    )


def branch_reverse(launch, layout, bwd, vals, sol, level):
    short_row, short_rows, long_row, long_rows = _branch_ranges(bwd, level)
    if not long_rows:
        return False
    launch(
        branch_backward,
        128 * long_rows + 32 * short_rows,
        [layout, bwd.tasks, vals, sol, long_row, long_rows, short_row],
    )
    return True


@wp.func
def panel_rank6(residual: wp.spatial_vector, tile: fa, stage: int, lane: int, column_group: int):
    """Subtract one block product (including its own ordered six-term dot)."""
    sums = wp.spatial_vector()
    for component in range(6):
        base = stage + 32 * component
        left = tile[base + lane]
        for col in range(6):
            sums[col] += left * tile[base + 6 * column_group + col]
    return residual - sums


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return static_cast<int>(__ballot_sync(0xffffffffu, keep != 0));
#else
    return keep;
#endif
""")
def _small_ballot(keep: int) -> int:
    pass


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __popc(static_cast<unsigned>(bits) & ((1u << lane) - 1u));
#else
    return 0;
#endif
""")
def _small_before(bits: int, lane: int) -> int:
    pass


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __popc(static_cast<unsigned>(bits));
#else
    return 0;
#endif
""")
def _small_pop(bits: int) -> int:
    pass


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    return __shfl_sync(0xffffffffu, value, source);
#else
    return value;
#endif
""")
def _small_broadcast(value: int, source: int) -> int:
    pass


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    if (dst.strides[0] == sizeof(float) && src.strides[0] == sizeof(float) &&
        !((reinterpret_cast<size_t>(dst.data + di) | reinterpret_cast<size_t>(src.data + si)) & 15u)) {
        *reinterpret_cast<float4*>(dst.data + di) = *reinterpret_cast<const float4*>(src.data + si);
        return;
    }
#endif
    for (int k = 0; k < 4; ++k) wp::index(dst, di+k) = wp::index(src, si+k);
""")
def _small_copy4(dst: fa, di: int, src: fa, si: int):
    pass


@wp.func
def _small_rank(mapping: ia, index: int, span: int, total: int) -> int:
    rank = total
    if index < span:
        code = mapping[index]
        rank = wp.where(code < 0, -1 - code, code)
    return rank


@wp.func
def small_stage(
    layout: Sparse,
    packed: Sparse,
    vals: fa,
    rhs: fa,
    sol: fa,
    blocks: fa,
    vectors: fa,
    inverse: ia,
    base: int,
    start: int,
    stop: int,
    tid: int,
) -> bool:
    """Compact contiguous slot stripes, preserving CSR order and the gathered RHS split."""
    first = layout.levrow[start]
    last = layout.levrow[stop]
    count = last - first
    slot0 = layout.rowptr[first]
    slot1 = layout.rowptr[last]
    span = slot1 - slot0
    edge0 = layout.colptr[first]
    edges = layout.colptr[last] - edge0
    column0 = layout.levcol[start]
    update0 = layout.levupd[start]
    update1 = layout.levupd[stop]
    term0 = layout.ptr[update0]
    terms = layout.ptr[update1] - term0
    if (
        count > 32
        or stop - start > 16
        or span > 512
        or edges > 128
        or update1 - update0 > 128
        or terms > 256
        or layout.levcol[stop] - column0 != edges
    ):
        return False

    mapping = small_ints(1285, 512)
    words = (span + 31) // 32
    lane = tid & 31
    warp = tid >> 5
    bad = int(0)
    for word in range(warp, words, 4):
        index = 32 * word + lane
        other = int(-1)
        if index < span:
            other = layout.cols[slot0 + index] - (first - base)
            mapping[index] = other
        bits = _small_ballot(int(other >= 0))
        if lane == 0:
            packed.right[word] = bits
    if tid <= stop - start:
        level = start + tid
        row = layout.levrow[level]
        packed.levrow[tid] = row - first
        packed.levupd[tid] = layout.levupd[level] - update0
        if layout.levcol[level] - column0 != layout.colptr[row] - edge0:
            bad = 1
    if tid <= count:
        packed.colptr[tid] = layout.colptr[first + tid] - edge0
    for scalar in range(tid, 6 * count, 128):
        vectors[scalar] = sol[6 * first + scalar]
    sync_block()

    width = int(0)
    if lane < words:
        width = _small_pop(packed.right[lane])
    prefix = small_exclusive(width, lane)
    total = _small_broadcast(prefix, 16)
    if total > 128:
        return False
    for word in range(warp, words, 4):
        index = 32 * word + lane
        rank = _small_broadcast(prefix, word) + _small_before(packed.right[word], lane)
        if index < span:
            other = mapping[index]
            if other >= 0:
                mapping[index] = rank
                packed.cols[rank] = other
                inverse[rank] = slot0 + index
                if other >= count:
                    bad = 1
            else:
                mapping[index] = -1 - rank
    sync_block()

    if tid == 0:
        packed.rowptr[count] = total
    if tid < count:
        row = first + tid
        begin = layout.rowptr[row] - slot0
        end = layout.rowptr[row + 1] - slot0
        lower = _small_rank(mapping, begin, span, total)
        upper = _small_rank(mapping, end, span, total)
        diagonal = small_map(mapping, layout.diag[row], slot0, slot1)
        packed.rowptr[tid] = lower
        packed.diag[tid] = diagonal
        suffix = end - (upper - lower)
        if _small_rank(mapping, suffix, span, total) != lower or diagonal < lower or diagonal != upper - 1:
            bad = 1
        elif packed.cols[diagonal] != tid:
            bad = 1
        for edge in range(packed.colptr[tid], packed.colptr[tid + 1]):
            entry = edge0 + edge
            slot = layout.slots[entry]
            other = layout.rows[entry] - (first - base)
            mapped = small_map(mapping, slot, slot0, slot1)
            packed.slots[edge] = mapped
            packed.rows[edge] = other
            if (
                other <= tid
                or other >= count
                or mapped < 0
                or layout.cslot[column0 + edge] != slot
                or layout.cdiag[column0 + edge] != layout.diag[row]
            ):
                bad = 1
            elif (
                packed.cols[mapped] != tid
                or slot < layout.rowptr[first + other]
                or slot >= layout.rowptr[first + other + 1]
            ):
                bad = 1
    for entry in range(tid, update1 - update0 + 1, 128):
        task = update0 + entry
        packed.ptr[entry] = layout.ptr[task] - term0
        if task < update1:
            mapped = small_map(mapping, layout.dest[task], slot0, slot1)
            packed.dest[entry] = mapped
            if mapped < 0:
                bad = 1
    for entry in range(tid, terms, 128):
        left = small_map(mapping, layout.left[term0 + entry], slot0, slot1)
        right = small_map(mapping, layout.right[term0 + entry], slot0, slot1)
        packed.left[entry] = left
        packed.right[entry] = right
        if left < 0 or right < 0:
            bad = 1
    for quad in range(tid, 9 * total, 128):
        _small_copy4(blocks, 4 * quad, vals, 36 * inverse[quad // 9] + 4 * (quad % 9))
    return small_any(bad) == 0


@wp.func
def group_front(packed: Sparse, blocks: fa, vectors: fa, piv: ia, row: int, lane: int):
    """One warp advances the factor, RHS and up to four child panels on one division stream."""
    residual = _subst_rhs(vectors, row, lane)
    begin = packed.rowptr[row]
    end = packed.diag[row]
    if begin < end:
        residual = _subst_entries(
            blocks, vectors, row, 0, packed.cols, packed.cols, begin, end, residual, lane, False, False
        )
    row_front(packed, blocks, vectors, piv, row, lane, residual)


@wp.func
def group_backward(packed: Sparse, blocks: fa, vectors: fa, row: int, comp: int):
    """Prefetch the transposed factor column and prepare its divisor before the edge dots."""
    diag = 36 * packed.diag[row]
    column = wp.spatial_vector()
    value = float(0.0)
    factor = float(1.0)
    if comp < 6:
        factor = blocks[diag + 7 * comp]
        value = vectors[6 * row + comp]
        for col in range(1, 6):
            column[col] = blocks[diag + 6 * col + comp]
    inverse = _front_inverse(factor)
    if comp < 6:
        for edge in range(packed.colptr[row], packed.colptr[row + 1]):
            other = 6 * packed.rows[edge]
            offset = 36 * packed.slots[edge] + comp
            dot = blocks[offset] * vectors[other]
            for inner in range(1, 6):
                dot += blocks[offset + 6 * inner] * vectors[other + inner]
            value -= dot
    for rev in range(6):
        col = 5 - rev
        if comp == col:
            value = _front_quotient(value, factor, inverse)
        if rev < 5:
            solved = group_share(value, col)
            if comp < col:
                value -= column[col] * solved
    if comp < 6:
        vectors[6 * row + comp] = value


@wp.kernel
def small_sparse_tail(layout: Sparse, vals: fa, rhs: fa, sol: fa, piv: ia, cut: int):
    tid = wp.tid()
    art = tid // 128
    local = tid % 128
    base = layout.base[art]
    stop = layout.artlev[art + 1]
    start = wp.min(layout.artlev[art] + cut, stop)
    if start == stop:
        return
    packed = small_layout()
    blocks = small_floats(0, 4608)
    vectors = small_floats(4608, 192)
    inverse = small_ints(1797, 128)
    if not small_stage(layout, packed, vals, rhs, sol, blocks, vectors, inverse, base, start, stop, local):
        small_unpacked(layout, vals, rhs, sol, piv, base, start, stop, local)
        return

    levels = stop - start
    count = packed.levrow[levels]
    terminal = (
        packed.levupd[levels - 1] == packed.levupd[levels]
        and packed.colptr[packed.levrow[levels - 1]] == packed.colptr[count]
    )
    first = layout.levrow[start]
    lane = local & 31
    comp = local & 7
    for level in range(levels):
        finish = terminal and level + 1 == levels
        for row in range(packed.levrow[level] + (local >> 5), packed.levrow[level + 1], 4):
            group_front(packed, blocks, vectors, piv, row, lane)
            if finish:
                sync_warp()
                if lane < 8:
                    group_backward(packed, blocks, vectors, row, comp)
                    if comp < 6:
                        sol[6 * (first + row) + comp] = vectors[6 * row + comp]
        sync_block()
        begin = packed.levupd[level]
        end = packed.levupd[level + 1]
        if begin < end:
            small_updates(packed, blocks, begin, end, local)
            sync_block()

    for reverse in range(levels - int(terminal)):
        level = levels - int(terminal) - 1 - reverse
        for row in range(packed.levrow[level] + (local >> 3), packed.levrow[level + 1], 16):
            group_backward(packed, blocks, vectors, row, comp)
            if comp < 6:
                sol[6 * (first + row) + comp] = vectors[6 * row + comp]
        if level > 0:
            sync_block()
    for quad in range(local, 9 * packed.rowptr[count], 128):
        _small_copy4(vals, 36 * inverse[quad // 9] + 4 * (quad % 9), blocks, 4 * quad)


@wp.func
def lookahead_column(residual: wp.spatial_vector, group: int, lane: int, piv: ia, owner: bool):
    """Distributed six-column solve. Carry each pivot candidate's diagonal in its own lane."""
    result = residual
    base_lane = 6 * group
    diagonal = float(0.0)
    for component in range(6):
        if lane == base_lane + component:
            diagonal = residual[component]
    floor = wp.max(1.0e-6 * wp.abs(diagonal), 1.0e-30)
    clamped = bool(False)
    for step in range(6):
        source = base_lane + step
        accepted = diagonal > floor
        clamped = clamped or ((lane == source) and not accepted)
        pivot = shfl(wp.sqrt(wp.where(accepted, diagonal, floor)), source)
        quotient = result[step] / pivot
        result[step] = wp.where(lane == source, pivot, wp.where(lane > source, quotient, 0.0))
        if step < 5:
            diagonal = diagonal - quotient * quotient
            for col in range(6):
                if col > step:
                    coefficient = shfl(quotient, base_lane + col)
                    result[col] = result[col] - quotient * coefficient
    if owner and clamped:
        wp.atomic_add(piv, 0, 1)
    return result


@wp.func
def lookahead_load_row(data: fa, start: int):
    first = _front_load_pair(data, start)
    middle = _front_load_pair(data, start + 2)
    last = _front_load_pair(data, start + 4)
    return wp.spatial_vector(first[0], first[1], middle[0], middle[1], last[0], last[1])


@wp.func
def lookahead_store_row(data: fa, start: int, row: wp.spatial_vector):
    _front_store_pair(data, start, row[0], row[1])
    _front_store_pair(data, start + 2, row[2], row[3])
    _front_store_pair(data, start + 4, row[4], row[5])


@wp.func
def lookahead_strip(vals: fa, piv: ia, work: ia, pending: fa, size: int, panel: int, strip: int, local: int, tile: fa):
    group = local >> 5
    lane = local & 31
    width = wp.min(size - panel, 4)
    wing = panel + 4 + strip
    row = wp.where(lane < 24, panel + lane // 6, wing)
    live = lane < 6 * width or (lane >= 24 and lane < 30 and wing < size)
    publish = live and group < width and row >= panel + group
    slot0 = int(0)
    if live:
        slot0 = work[1 + row * size]

    stash = 6 * (24 * group + lane)
    if strip == 0 and panel > 0 and lane < 24 and lane // 6 >= group:
        old_row = panel - 4 + lane // 6
        old_slot = work[1 + old_row * size] + panel - 4 + group
        lookahead_store_row(vals, 36 * old_slot + 6 * (lane % 6), lookahead_load_row(pending, stash))

    address = 36 * (slot0 + panel + group) + 6 * (lane % 6)
    residual = wp.spatial_vector()
    if publish:
        residual = lookahead_load_row(vals, address)
    if panel > 0:
        if live:
            source = 36 * (slot0 + panel - 4 + group) + 6 * (lane % 6)
            previous = lookahead_load_row(vals, source)
            for col in range(6):
                tile[192 * group + 32 * col + lane] = previous[col]
        sync_block()
        if publish:
            for prior in range(4):
                residual = panel_rank6(residual, tile, 192 * prior, lane, group)

    for step in range(4):
        if step < width:
            if group == step:
                residual = lookahead_column(residual, group, lane, piv, strip == 0)
                if step + 1 < width:
                    for col in range(6):
                        tile[768 + 192 * step + 32 * col + lane] = residual[col]
            if step + 1 < width:
                sync_block()
                if group > step and publish:
                    residual = panel_rank6(residual, tile, 768 + 192 * step, lane, group)

    if publish:
        if lane >= 24 or (strip == 0 and panel + width == size):
            lookahead_store_row(vals, address, residual)
        elif strip == 0:
            lookahead_store_row(pending, stash, residual)


@wp.func
def lookahead_dot(vals: fa, left: int, right: int):
    """One thread, four independent ordered dots; each operand is two consecutive rows."""
    left_head = _schur_load4(vals, left)
    left_mid = _schur_load4(vals, left + 4)
    left_end = _schur_load4(vals, left + 8)
    right_head = _schur_load4(vals, right)
    right_mid = _schur_load4(vals, right + 4)
    right_end = _schur_load4(vals, right + 8)
    sums = wp.vec4()
    sums = _schur_outer(sums, left_head[0], left_mid[2], right_head[0], right_mid[2])
    sums = _schur_outer(sums, left_head[1], left_mid[3], right_head[1], right_mid[3])
    sums = _schur_outer(sums, left_head[2], left_end[0], right_head[2], right_end[0])
    sums = _schur_outer(sums, left_head[3], left_end[1], right_head[3], right_end[1])
    sums = _schur_outer(sums, left_mid[0], left_end[2], right_mid[0], right_end[2])
    return _schur_outer(sums, left_mid[1], left_end[3], right_mid[1], right_end[3])


@wp.func
def lookahead_prefix(vals: fa, work: ia, size: int, panel: int, block: int, local: int, dots: fa):
    """Prepare one destination of the NEXT panel, through (but not including) this panel."""
    first = panel + 4
    width = wp.min(4, size - first)
    triangle = width * (width + 1) // 2
    row = int(0)
    col = int(0)
    if block < triangle:
        row = dense_tail_tri_row(block)
        col = block - row * (row + 1) // 2
    else:
        row = width + (block - triangle) // width
        col = (block - triangle) % width
    left = work[1 + (first + row) * size]
    right = work[1 + (first + col) * size]
    address = 36 * (left + first + col) + local
    value = float(0.0)
    if local < 36:
        value = vals[address]
    for job in range(local, 9 * panel, 128):
        prior = job // 9
        row_pair = (job % 9) // 3
        col_pair = job % 3
        sums = lookahead_dot(vals, 36 * (left + prior) + 12 * row_pair, 36 * (right + prior) + 12 * col_pair)
        target = 36 * prior + 12 * row_pair + 2 * col_pair
        dots[target] = sums[0]
        dots[target + 1] = sums[1]
        dots[target + 6] = sums[2]
        dots[target + 7] = sums[3]
    sync_block()
    if local < 36:
        for prior in range(panel):
            value = value - dots[36 * prior + local]
        vals[address] = value


@wp.kernel
def lookahead_factor_step(
    layout: Sparse, vals: fa, piv: ia, work: ia, pending: fa, cut: int, size: int, panel: int, strips: int
):
    tid = wp.tid()
    cta = tid >> 7
    local = tid & 127
    if panel == 0:
        bad = int(0)
        if local < size:
            bad = int(work[1 + local * size + local] < 0)
        reject = small_any(bad)
        if tid == 0:
            work[0] = wp.where(reject == 0, 2, 0)
        if reject != 0:
            if cta == 0:
                stop = layout.artlev[1]
                start = wp.min(layout.artlev[0] + cut, stop)
                dense_tail_sparse(layout, vals, piv, start, stop, local)
            return
    elif work[0] == 0:
        return
    tile = dense_tail_tile()
    if cta < strips:
        lookahead_strip(vals, piv, work, pending, size, panel, cta, local, tile)
    else:
        lookahead_prefix(vals, work, size, panel, cta - strips, local, tile)


def panel_launch_factor(launch, layout, vals, piv, work, cut, size, device):
    """Factor the validated singleton tail in four-column panels."""
    pending = wp.empty(4 * 24 * 6, dtype=float, device=device)
    launch(dense_tail_prepare, 128 * size, [layout, work, cut, size])
    for panel in range(0, size, 4):
        trailing = max(0, size - panel - 4)
        strips = max(1, trailing)
        width = min(4, trailing)
        ahead = width * trailing - width * (width - 1) // 2 if panel else 0
        launch(
            lookahead_factor_step, 128 * (strips + ahead), [layout, vals, piv, work, pending, cut, size, panel, strips]
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


@wp.func
def _wave_strip(
    vals: fa, piv: ia, work: ia, incoming: fa, outgoing: fa, size: int, panel: int, strip: int, local: int, tile: fa
):
    group = local >> 5
    lane = local & 31
    width = wp.min(size - panel, 4)
    wing = panel + 4 + strip
    row = wp.where(lane < 24, panel + lane // 6, wing)
    live = lane < 6 * width or (lane >= 24 and lane < 30 and wing < size)
    publish = live and group < width and row >= panel + group
    slot0 = int(0)
    if live:
        slot0 = work[1 + row * size]

    stash = 6 * (24 * group + lane)
    if strip == 0 and panel > 0 and lane < 24 and lane // 6 >= group:
        old_row = panel - 4 + lane // 6
        old_slot = work[1 + old_row * size] + panel - 4 + group
        lookahead_store_row(vals, 36 * old_slot + 6 * (lane % 6), lookahead_load_row(incoming, stash))

    address = 36 * (slot0 + panel + group) + 6 * (lane % 6)
    residual = wp.spatial_vector()
    if publish:
        residual = lookahead_load_row(vals, address)
    if panel > 0:
        if live:
            source = 36 * (slot0 + panel - 4 + group) + 6 * (lane % 6)
            previous = lookahead_load_row(vals, source)
            for col in range(6):
                tile[192 * group + 32 * col + lane] = previous[col]
        sync_block()
        if publish:
            for prior in range(4):
                residual = panel_rank6(residual, tile, 192 * prior, lane, group)

    for step in range(4):
        if step < width:
            if group == step:
                residual = lookahead_column(residual, group, lane, piv, strip == 0)
                if step + 1 < width:
                    for col in range(6):
                        tile[768 + 192 * step + 32 * col + lane] = residual[col]
            if step + 1 < width:
                sync_block()
                if group > step and publish:
                    residual = panel_rank6(residual, tile, 768 + 192 * step, lane, group)

    if publish:
        if lane >= 24 or (strip == 0 and panel + width == size):
            lookahead_store_row(vals, address, residual)
        elif strip == 0:
            lookahead_store_row(outgoing, stash, residual)


@wp.func
def _wave_boundary(
    layout: Sparse, vals: fa, rhs: fa, sol: fa, work: ia, boundary: fa, cut: int, size: int, split: int, tid: int
):
    local = tid & 127
    component = (tid >> 7) % 6
    index = tid // (6 * 128)
    first = layout.levrow[layout.artlev[0] + cut]
    row = first + index
    base = layout.base[0]
    head = layout.rowptr[row]
    end = work[1 + index * size]
    if split > 0 and end - head + index + 1 > split:
        partial = float(0.0)
        for entry in range(head + local, end, 128):
            partial += external_dot(layout, vals, sol, base, entry, component)
        boundary[tid] = partial
    elif local < 32:
        result = rhs[6 * row + component]
        for block in range(head, end, 32):
            term = float(0.0)
            if block + local < end:
                term = external_dot(layout, vals, sol, base, block + local, component)
            for lane in range(wp.min(32, end - block)):
                result -= shfl(term, lane)
        if local == 0:
            boundary[tid] = result


@wp.func
def _wave_sparse_solve(
    layout: Sparse, vals: fa, rhs: fa, sol: fa, scratch: fa, cut: int, split: int, orig_threads: int, tid: int
):
    """After all factors are published, use the original CSR/CSC substitution ordering."""
    lane = tid & 31
    group = tid >> 5
    base = layout.base[0]
    stop = layout.artlev[1]
    start = wp.min(layout.artlev[0] + cut, stop)
    for level in range(start, stop):
        first = layout.levrow[level]
        last = layout.levrow[level + 1]
        for row in range(first + group, last, 4):
            if split <= 0 or layout.rowptr[row + 1] - layout.rowptr[row] <= split:
                forward(vals, rhs, sol, row, base, layout.rowptr, layout.cols, layout.diag, lane)
        if split > 0:
            for row in range(first, last):
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
                        0,
                        orig_threads,
                        tid,
                    )
        sync_block()
    for reverse in range(stop - start):
        level = stop - reverse - 1
        first = layout.levrow[level]
        last = layout.levrow[level + 1]
        for row in range(first + group, last, 4):
            if split <= 0 or layout.colptr[row + 1] - layout.colptr[row] <= split:
                backward(vals, sol, row, base, layout.colptr, layout.rows, layout.slots, layout.diag, lane)
        if split > 0:
            for row in range(first, last):
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
                        0,
                        orig_threads,
                        tid,
                    )
        sync_block()


@wp.func
def _wave_forward(
    layout: Sparse,
    vals: fa,
    rhs: fa,
    sol: fa,
    work: ia,
    boundary: fa,
    incoming: fa,
    tile: fa,
    state: fa,
    first: int,
    size: int,
    split: int,
    panel: int,
    local: int,
    published: bool,
):
    """Solve one panel. Older wings are final; the immediate panel core may still be private."""
    component = wp.min(local >> 4, 5)
    member = local & 15
    lane = local & 31
    for scalar in range(local, 6 * panel, 128):
        state[scalar] = sol[6 * first + scalar]
    count = wp.min(4, size - panel)
    for part in range(count):
        row = panel + part
        slot0 = work[1 + row * size]
        for quad in range(local, 9 * (row + 1), 128):
            other = quad // 9
            if published or other < panel:
                _small_copy4(tile, 4 * quad, vals, 36 * slot0 + 4 * quad)
            else:
                core = 4 * (other - panel) + part
                _small_copy4(tile, 4 * quad, incoming, 36 * core + 4 * (quad % 9))
        sync_block()
        previous = slot0 - layout.rowptr[first + row]
        address = 128 * (6 * row + component)
        residual = float(0.0)
        if split <= 0 or previous + row + 1 <= split:
            residual = boundary[address]
            for begin in range(0, row, 16):
                term = float(0.0)
                if begin + member < row:
                    term = staged_term(tile, state, 0, begin + member, component, False)
                for source in range(wp.min(16, row - begin)):
                    residual -= shfl(term, (lane & 16) + source)
        else:
            partial = eight()
            for stripe in range(8):
                logical = member + 16 * stripe
                value = boundary[address + logical]
                other = (logical - previous) & 127
                if other < row:
                    value += staged_term(tile, state, 0, other, component, False)
                partial[stripe] = value
            subtotal = ((partial[0] + partial[4]) + (partial[2] + partial[6])) + (
                (partial[1] + partial[5]) + (partial[3] + partial[7])
            )
            for stage in range(4):
                subtotal += shfl(subtotal, lane ^ (8 >> stage))
            residual = rhs[6 * (first + row) + component] - subtotal
        if local < 96 and member == 0:
            state[288 + component] = residual
        sync_block()
        if local < 32:
            _subst_finish(tile, state, row, row, state[288 + wp.min(lane, 5)], lane, False)
        sync_block()
    if not published:
        for scalar in range(local, 6 * count, 128):
            sol[6 * (first + panel) + scalar] = state[6 * panel + scalar]


@wp.kernel
def _wave_step(
    layout: Sparse,
    vals: fa,
    rhs: fa,
    sol: fa,
    piv: ia,
    work: ia,
    boundary: fa,
    incoming: fa,
    outgoing: fa,
    cut: int,
    size: int,
    split: int,
    panel: int,
    strips: int,
    side: int,
):
    tid = wp.tid()
    block = tid >> 7
    local = tid & 127
    if panel == 0:
        bad = int(0)
        if local < size:
            bad = int(work[1 + local * size + local] < 0)
        reject = small_any(bad)
        if tid == 0:
            work[0] = wp.where(reject == 0, 2, 0)
        if reject != 0:
            if block == 0:
                stop = layout.artlev[1]
                start = wp.min(layout.artlev[0] + cut, stop)
                dense_tail_sparse(layout, vals, piv, start, stop, local)
            return
    elif work[0] == 0:
        return
    tile = dense_tail_tile()
    if block < strips:
        _wave_strip(vals, piv, work, incoming, outgoing, size, panel, block, local, tile)
    elif block < side:
        lookahead_prefix(vals, work, size, panel, block - strips, local, tile)
    elif panel == 0:
        _wave_boundary(layout, vals, rhs, sol, work, boundary, cut, size, split, tid - 128 * side)
    else:
        first = layout.levrow[layout.artlev[0] + cut]
        _wave_forward(
            layout,
            vals,
            rhs,
            sol,
            work,
            boundary,
            incoming,
            tile,
            sub_tile(),
            first,
            size,
            split,
            panel - 4,
            local,
            False,
        )


@wp.kernel
def _wave_finish(
    layout: Sparse,
    vals: fa,
    rhs: fa,
    sol: fa,
    work: ia,
    boundary: fa,
    scratch: fa,
    cut: int,
    size: int,
    split: int,
    last_panel: int,
):
    local = wp.tid()
    if work[0] == 0:
        _wave_sparse_solve(layout, vals, rhs, sol, scratch, cut, split, 128, local)
    else:
        first = layout.levrow[layout.artlev[0] + cut]
        factors = substitution_stage()
        state = sub_tile()
        _wave_forward(
            layout, vals, rhs, sol, work, boundary, vals, factors, state, first, size, split, last_panel, local, True
        )
        substitution_run(layout, vals, rhs, boundary, work, first, size, split, factors, state, local, True)
        for scalar in range(local, 6 * size, 128):
            sol[6 * first + scalar] = state[scalar]


def dense_wave_launch(launch, plan, layout, vals, rhs, sol, piv, scratch, cut, size, device):
    """The caller's dense_tail_size gate ensures one articulation and 8..48 singleton levels."""

    def floats(count):
        return wp.empty(count, dtype=float, device=device)

    work = wp.empty(1 + size * size, dtype=wp.int32, device=device)
    split = plan.tail_split_entries
    threads = plan.tail_block_dim
    if threads != 128:
        panel_launch_factor(launch, layout, vals, piv, work, cut, size, device)
        launch(dense_tail_substitute, 128, [layout, vals, rhs, sol, scratch, cut, split, threads])
        return
    boundary = floats(6 * 128 * size)
    incoming = floats(4 * 4 * 36)
    outgoing = floats(4 * 4 * 36)
    launch(dense_tail_prepare, 128 * size, [layout, work, cut, size])
    for panel in range(0, size, 4):
        trailing = max(0, size - panel - 4)
        strips = max(1, trailing)
        width = min(4, trailing)
        ahead = width * trailing - width * (width - 1) // 2 if panel else 0
        side = strips + ahead
        blocks = side + (1 if panel else 6 * size)
        launch(
            _wave_step,
            128 * blocks,
            [layout, vals, rhs, sol, piv, work, boundary, incoming, outgoing, cut, size, split, panel, strips, side],
        )
        incoming, outgoing = outgoing, incoming
    launch(
        _wave_finish, 128, [layout, vals, rhs, sol, work, boundary, scratch, cut, size, split, ((size - 1) // 4) * 4]
    )


class _Buffer(Enum):
    VALUES = 0
    RHS = 1
    DELTA = 2
    PIVOT_CLAMPED = 3


class _Phase(NamedTuple):
    short_offsets: list[int]
    long_offsets: list[int]
    tasks: Tasks


def _bind_phase(phase: "LevelLaunchPhaseTasks") -> _Phase:
    tasks = Tasks()
    tasks.short_ids = phase.short
    tasks.short_aux = phase.short_aux
    tasks.long_ids = phase.long
    tasks.long_chunk_offsets = phase.long_chunk_offsets
    tasks.chunk_begin = phase.chunk_begin
    tasks.chunk_end = phase.chunk_end
    tasks.chunk_aux = phase.chunk_aux
    return _Phase(phase.short_offsets, phase.long_offsets, tasks)


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
    plan: "RigidArticulationLevelLaunchPlan", layout: "RigidArticulationSparseLayout", device: "wp.context.Devicelike"
) -> WarpLevelSchedule | None:
    """Build a fused schedule for plans that perform at least one split reduction."""
    if plan.tail_split_entries == 0 and all(
        phase.long.shape[0] == 0 for phase in (plan.update, plan.forward, plan.backward)
    ):
        # Unsplit plans guarantee bitwise parity with the cooperative kernel.
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

    _schedule_launches(record, plan, sparse, device)
    return schedule


def _schedule_launches(launch, plan, layout, device):
    fwd, upd, bwd = (_bind_phase(phase) for phase in (plan.forward, plan.update, plan.backward))
    val, b, sol, piv = _Buffer
    tmp = plan.tail_partial

    def scheduled(kernel, scheme, level, buffers):
        short, long, tasks = scheme
        first = short[level]
        size = short[level + 1] - first
        long_first = long[level]
        total = size + long[level + 1] - long_first
        if kernel is apply_updates:
            long_count = long[level + 1] - long_first
            launch_schur(launch, layout, tasks, buffers[0], first, size, long_first, long_count)
            return
        if kernel is eliminate and upd.long_offsets[level + 1] > upd.long_offsets[level]:
            launch(broad_front, 128 * total, [layout, tasks, *buffers, first, size, long_first])
            return
        launch(kernel, 32 * total, [layout, tasks, *buffers, first, size, long_first])

    cut = plan.leading_level_count
    arts = plan.articulation_count
    widths = plan.rows_per_level
    if arts > 0 and len(widths) == plan.level_count and plan.tail_split_entries <= 0 and 0 < cut < 6:
        while (
            cut > 0
            and widths[cut - 1] <= 4 * arts
            and len(widths) - cut + 1 <= 16
            and sum(widths[cut - 1 :]) <= 32 * arts
        ):
            if (
                any(scheme.long_offsets[cut - 1] != scheme.long_offsets[cut] for scheme in (fwd, upd, bwd))
                or dense_tail_size(plan, cut - 1) >= 8
            ):
                break
            cut -= 1
    tail_rows = plan.rows_per_level[cut:]
    gather = (
        cut < plan.level_count
        and arts > 0
        and len(tail_rows) <= 16
        and len(plan.rows_per_level) == plan.level_count
        and plan.tail_split_entries <= 0
        and sum(tail_rows) <= 32 * arts
        and dense_tail_size(plan, cut) < 8
    )
    width = min(32, sum(tail_rows)) if gather else 0
    branched = branch_enabled(plan) and not gather
    for level in range(cut):
        if branched:
            branch_level(launch, layout, fwd, upd, bwd, val, b, sol, piv, level, cut)
            continue
        pipelined = fwd.long_offsets[level + 1] == fwd.long_offsets[level] and (
            level == 0 or upd.long_offsets[level] == upd.long_offsets[level - 1]
        )
        if pipelined:
            launch(
                first_front,
                32 * (fwd.short_offsets[level + 1] - fwd.short_offsets[level]),
                [layout, fwd.tasks, val, b if level == 0 else sol, sol, piv, fwd.short_offsets[level]],
            )
        else:
            scheduled(eliminate, fwd, level, [val, b, sol, piv])
        if width and level + 1 == cut:
            first, shorts, long_first, longs = _branch_ranges(upd, level)
            launch(
                update_and_gather,
                32 * arts * width + 9 * shorts + 1152 * longs,
                [layout, upd.tasks, val, b, sol, first, long_first, longs, width, arts, cut],
            )
        elif (
            level + 1 < cut
            and fwd.long_offsets[level + 2] == fwd.long_offsets[level + 1]
            and upd.long_offsets[level + 1] == upd.long_offsets[level]
        ):
            first, shorts, long_first, longs = _branch_ranges(upd, level)
            rows, row_count, _, _ = _branch_ranges(fwd, level + 1)
            launch(
                branch_update_forward,
                9 * shorts + 32 * row_count,
                [layout, upd.tasks, fwd.tasks, val, b, sol, 0, 0, 0, 0, rows, row_count, first],
            )
        else:
            scheduled(apply_updates, upd, level, [val])
    if cut < plan.level_count:
        size = dense_tail_size(plan, cut)
        if size >= 8:
            dense_wave_launch(launch, plan, layout, val, b, sol, piv, tmp, cut, size, device)
        elif not small_tail_launch(launch, plan, layout, val, b, sol, piv, cut):
            launch(
                cooperative_tail,
                128 * plan.articulation_count,
                [layout, val, b, sol, piv, tmp, cut, plan.tail_split_entries, plan.tail_block_dim],
            )
    for level in reversed(range(cut)):
        if (
            bwd.long_offsets[level] == bwd.long_offsets[level + 1]
            and bwd.short_offsets[level + 1] - bwd.short_offsets[level] >= 256
        ):
            begin = bwd.short_offsets[level]
            count = bwd.short_offsets[level + 1] - begin
            launch(short_leading_back, 4 * count, [layout, bwd.tasks, val, sol, begin, count])
            continue
        if branched and branch_reverse(launch, layout, bwd, val, sol, level):
            continue
        scheduled(back_sub, bwd, level, [val, sol])
