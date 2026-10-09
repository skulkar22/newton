# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Packed workspace and route assembly for direct tendon material transfer."""

import warp as wp

from ..sim.tendon import TendonLinkType
from .tendon_alm_material import TendonALMMaterialState, project_alm_material
from .tendon_material import solve_tendon_material_component
from .tendon_material_nonlinear import TendonMaterialNonlinearState, solve_tendon_material_nonlinear_component


@wp.struct
class TendonMaterialState:
    enabled: bool
    segment_alm: bool
    legacy_alm_damping: bool
    process_changed_routes: bool
    changed_routes: wp.array[int]
    alm: TendonALMMaterialState
    nonlinear_enabled: bool
    nonlinear: TendonMaterialNonlinearState
    ea_low: float
    ea_ratio: float
    transition_strain: float
    transition_width: float
    raw_compliance: wp.array[float]
    initial: wp.array[float]
    compliance: wp.array[float]
    cap: wp.array[float]
    upper: wp.array[float]
    output: wp.array[float]
    status: wp.array[int]
    faces: wp.array[int]
    valid: wp.array[int]
    pieces: wp.array[int]
    count: wp.array[int]
    ids: wp.array[int]
    edge_ids: wp.array[int]
    next_link: wp.array[int]
    incoming: wp.array[int]
    failure: wp.array[int]
    failure_component: wp.array[int]
    failure_seg: wp.array[int]
    failure_length: wp.array[float]
    failure_compliance: wp.array[float]
    failure_tension: wp.array[float]
    failure_upper: wp.array[float]


@wp.func
def fail_tendon_material(state: TendonMaterialState, code: int, tendon: int, component: int):
    # One thread owns a tendon. Latch the first failure, including during graph replay.
    if state.failure[tendon] == 0:
        state.failure[tendon] = code
        state.failure_component[tendon] = component
        wp.printf(
            "ERROR: direct tendon material solve failed (tendon=%d, component=%d, status=%d). "
            "Discard this simulation state and call solver.check_tendon_material() for details.\n",
            tendon,
            component,
            code,
        )


@wp.func
def reject_tendon_material(state: TendonMaterialState, code: int, tendon: int, component: int, latch: int) -> bool:
    """Report a rejected component, latching the failure only on an accepted pose.

    VBD re-solves the material every iteration, including iteration 0, whose pose is the raw
    inertial predictor and is discarded one iteration later. Latching a rejection taken on such
    a trial pose would freeze the tendon's rest lengths for the rest of the simulation. A trial
    rejection therefore leaves this iteration's rest lengths in place and is re-tried next
    iteration; the accepted pose at the end of the step still latches, so a genuine infeasibility
    is reported within the same step.
    """
    if latch != 0:
        fail_tendon_material(state, code, tendon, component)
    return False


@wp.func
def record_tendon_material_failure_span(
    state: TendonMaterialState,
    tendon: int,
    row: int,
    count: int,
    seg_length: wp.array[float],
    seg_compliance: wp.array[float],
    latch: int,
):
    """Record the span carrying the tightest bound of a failing component.

    A component's index alone does not say which of its spans the solve could not satisfy, and
    that span's own numbers are what a diagnosis needs. Report the span whose demanded extension
    exceeds its rest-length bound by the most, or the most loaded one when all bounds hold.
    """
    if latch == 0:
        return
    if state.failure_seg[tendon] >= 0:
        return
    worst = int(-1)
    margin = float(0.0)
    for slot in range(count):
        index = row + slot
        violation = state.initial[index] - state.upper[index]
        if worst < 0 or violation > margin:
            worst = index
            margin = violation
    if worst < 0:
        return
    seg = state.ids[worst]
    if seg < 0:
        return
    state.failure_seg[tendon] = seg
    state.failure_length[tendon] = seg_length[seg]
    state.failure_compliance[tendon] = seg_compliance[seg]
    # Linear-law tension of the extension the component was asked to hold. The sigmoid law
    # reports its own tension through the same ratio, which is exact only at low strain.
    state.failure_tension[tendon] = state.initial[worst] / state.compliance[worst]
    state.failure_upper[tendon] = state.upper[worst]


@wp.func
def project_tendon_component(state: TendonMaterialState, row: int, count: int, tendon: int, latch: int) -> bool:
    if state.count[row] != count:
        state.valid[row] = 0
    state.count[row] = count
    if state.segment_alm:
        if not project_alm_material(
            state.alm,
            state.ids,
            state.initial,
            state.compliance,
            state.cap,
            state.upper,
            state.output,
            state.status,
            state.faces,
            state.valid,
            state.pieces,
            row,
            count,
            state.ea_low,
            state.ea_ratio,
            state.transition_strain,
            state.transition_width,
        ):
            return reject_tendon_material(state, state.status[row], tendon, row, latch)
        return True
    if state.nonlinear_enabled:
        solve_tendon_material_nonlinear_component(
            state.nonlinear,
            count,
            1,
            row,
            1,
            1,
            state.ea_low,
            state.ea_ratio,
            state.transition_strain,
            state.transition_width,
            1.0e-6,
            64,
            1.0e-5,
        )
        if state.status[row] < 0:
            return reject_tendon_material(state, state.status[row] - 200, tendon, row, latch)
        return True
    # Each component packs into its original segment interval; inactive route slots
    # need no padding. Storage remains O(total segments), not 32 slots per segment.
    solve_tendon_material_component(
        state.initial,
        state.compliance,
        state.cap,
        state.upper,
        state.output,
        state.status,
        state.faces,
        state.valid,
        state.pieces,
        count,
        1,
        row,
        1,
        1,
    )
    if state.status[row] < 0:
        return reject_tendon_material(state, state.status[row], tendon, row, latch)
    return True


@wp.func
def transfer_tendon_material_direct(
    state: TendonMaterialState,
    tendon: int,
    link_start: int,
    num_links: int,
    seg_offset: int,
    num_segs: int,
    tendon_link_type: wp.array[int],
    tendon_link_active: wp.array[bool],
    cone_seg_l: wp.array[int],
    cone_seg_r: wp.array[int],
    cap_ratio: wp.array[float],
    seg_active: wp.array[int],
    seg_compliance: wp.array[float],
    seg_damping: wp.array[float],
    seg_length: wp.array[float],
    seg_stretch: wp.array[float],
    rolling_delta_l: wp.array[float],
    rolling_delta_r: wp.array[float],
    apply_rolling_transfer: int,
    apply_pinhole_slip: int,
    min_rest: float,
    latch_failure: int,
) -> bool:
    for s in range(num_segs):
        seg = seg_offset + s
        # Check authored inputs too: VBD's per-span floor and route merging
        # must not turn invalid coefficients into an apparently valid solve.
        raw_compliance = state.raw_compliance[seg]
        if not wp.isfinite(raw_compliance) or raw_compliance < 1.0e-25:
            return reject_tendon_material(state, -1, tendon, seg_offset, latch_failure)
        state.next_link[seg] = -1
        state.incoming[seg] = -1

    for i in range(1, num_links - 1):
        link = link_start + i
        kind = tendon_link_type[link]
        rolling = kind == int(TendonLinkType.ROLLING) and tendon_link_active[link]
        pinhole = kind == int(TendonLinkType.PINHOLE)
        if not ((apply_rolling_transfer != 0 and rolling) or (apply_pinhole_slip != 0 and pinhole)):
            continue
        left = cone_seg_l[link]
        right = cone_seg_r[link]
        if left < 0 or right < 0:
            continue
        if left < seg_offset or right >= seg_offset + num_segs or left >= right:
            return reject_tendon_material(state, -102, tendon, seg_offset, latch_failure)
        if seg_active[left] == 0 or seg_active[right] == 0 or state.next_link[left] >= 0 or state.incoming[right] >= 0:
            return reject_tendon_material(state, -102, tendon, seg_offset, latch_failure)
        for skipped in range(left + 1, right):
            if seg_active[skipped] != 0:
                return reject_tendon_material(state, -102, tendon, seg_offset, latch_failure)
        state.next_link[left] = link
        state.incoming[right] = link
        if rolling:
            # The no-slip trial carries the full rolling motion, including material
            # exchanged with the arc. Project friction AFTER constructing this trial.
            # Keep the two modes separate to match the validated float32 evaluation.
            common = 0.5 * (rolling_delta_r[left] + rolling_delta_l[right])
            differential = 0.5 * (rolling_delta_r[left] - rolling_delta_l[right])
            seg_stretch[left] = seg_stretch[left] - common
            seg_stretch[right] = seg_stretch[right] - common
            seg_stretch[left] = seg_stretch[left] - differential
            seg_stretch[right] = seg_stretch[right] + differential

    row = int(-1)
    count = int(0)
    previous = int(-1)
    for s in range(num_segs):
        seg = seg_offset + s
        if seg_active[seg] == 0:
            continue
        link = int(-1)
        if previous >= 0:
            link = state.next_link[previous]
        if link < 0:
            if count > 0:
                if not project_tendon_component(state, row, count, tendon, latch_failure):
                    record_tendon_material_failure_span(
                        state, tendon, row, count, seg_length, seg_compliance, latch_failure
                    )
                    return False
            row = seg
            count = 0
        if count >= 32:
            return reject_tendon_material(state, -103, tendon, row, latch_failure)
        index = row + count
        if state.ids[index] != seg:
            state.valid[row] = 0
        state.ids[index] = seg
        if state.segment_alm:
            state.alm.reference[index] = seg_stretch[seg]
        compliance = seg_compliance[seg]
        offset = compliance * seg_damping[seg]
        state.initial[index] = seg_stretch[seg] + offset
        state.compliance[index] = compliance
        state.upper[index] = (seg_length[seg] - min_rest) + offset
        if state.nonlinear_enabled:
            state.nonlinear.reference[index] = seg_stretch[seg]
            state.nonlinear.length[index] = seg_length[seg]
            state.nonlinear.damping[index] = seg_damping[seg]
        if count > 0:
            edge = index - 1
            if state.edge_ids[edge] != link:
                state.valid[row] = 0
            state.edge_ids[edge] = link
            state.cap[edge] = cap_ratio[link]
        count += 1
        previous = seg
    if count > 0:
        if not project_tendon_component(state, row, count, tendon, latch_failure):
            record_tendon_material_failure_span(state, tendon, row, count, seg_length, seg_compliance, latch_failure)
            return False

    # Publish only after every component on this tendon has passed its certificate.
    row = int(-1)
    count = int(0)
    previous = int(-1)
    for s in range(num_segs):
        seg = seg_offset + s
        if seg_active[seg] == 0:
            continue
        if previous < 0:
            row = seg
            count = 0
        elif state.next_link[previous] < 0:
            row = seg
            count = 0
        if state.nonlinear_enabled or state.segment_alm:
            seg_stretch[seg] = state.output[row + count]
            if state.segment_alm and state.alm.force[row + count] == 0.0:
                # Release a certified zero-force row exactly. Reconstructing a
                # float rest length can erase its tiny negative extension; using
                # that rounded extension for dual ascent would otherwise leave
                # a positive multiplier decaying indefinitely towards underflow.
                state.alm.multiplier[seg] = 0.0
        else:
            offset = seg_compliance[seg] * seg_damping[seg]
            seg_stretch[seg] = state.output[row + count] - offset
        count += 1
        previous = seg
    return True
