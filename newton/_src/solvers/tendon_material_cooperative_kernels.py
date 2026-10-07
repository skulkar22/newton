# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One-warp-per-tendon CUDA integration of nonlinear material projection.

The leader prepares the routed, full-rolling reference and publishes certified
extensions. All lanes participate in each connected component's force solve.
"""

import warp as wp

from ..sim.tendon import TendonLinkType
from .tendon_alm_material import project_alm_material_cooperative
from .tendon_kernels import oriented_wrap_arc_length, tendon_segment_length_rate, tendon_segment_length_rate_from_poses
from .tendon_material_cooperative import solve_tendon_material_nonlinear_component as project_cooperative
from .tendon_material_cooperative import warp_broadcast
from .tendon_material_state import (
    TendonMaterialState,
    record_tendon_material_failure_span,
    reject_tendon_material,
)


@wp.func
def record_component(state: TendonMaterialState, row: int, count: int, tendon: int):
    if state.count[row] != count:
        state.valid[row] = 0
    state.count[row] = count


@wp.func
def prepare_material(
    tendon_id: int,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    tendon_start: wp.array[int],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_orientation: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_rest_length_step: wp.array[float],
    seg_route_rest_length: wp.array[float],
    seg_stretch: wp.array[float],
    seg_damping_tension: wp.array[float],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    tendon_link_active: wp.array[bool],
    tendon_link_active_step: wp.array[bool],
    tendon_link_route_rest_length: wp.array[float],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_length: wp.array[float],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_rolling_delta_l: wp.array[float],
    seg_rolling_delta_r: wp.array[float],
    tendon_link_cone_seg_l: wp.array[int],
    tendon_link_cone_seg_r: wp.array[int],
    tendon_link_cap_ratio: wp.array[float],
    tendon_cone_sweep_count: wp.array[int],
    damping_from_pose_delta: int,
    dt: float,
    apply_rolling_transfer: int,
    apply_pinhole_slip: int,
    adaptive_cone_sweeps: int,
    tendon_max_sweeps: int,
    tendon_settle_tol: float,
    sigmoid_ea_low: float,
    sigmoid_ea_ratio: float,
    sigmoid_transition_strain: float,
    sigmoid_transition_width: float,
    direct: TendonMaterialState,
) -> int:
    if direct.process_changed_routes and direct.changed_routes[tendon_id] == 0:
        return 0
    if direct.failure[tendon_id] != 0:
        return 0
    link_start = tendon_start[tendon_id]
    link_end = tendon_start[tendon_id + 1]
    num_links = link_end - link_start
    num_segs = num_links - 1
    tendon_cone_sweep_count[tendon_id] = 0
    if num_segs < 1:
        return 0
    seg_offset = link_start - tendon_id
    min_rest = 1e-06
    for s in range(num_segs):
        seg = seg_offset + s
        seg_damping_tension[seg] = 0.0
        seg_rest_length[seg] = seg_route_rest_length[seg]
    for i in range(1, num_links - 1):
        link_idx = link_start + i
        if tendon_link_type[link_idx] != int(TendonLinkType.ROLLING):
            continue
        if tendon_link_route_rest_length[link_idx] <= 0.0:
            continue
        if not tendon_link_active[link_idx]:
            continue
        if tendon_link_active_step[link_idx]:
            continue
        seg_left = seg_offset + i - 1
        seg_right = seg_left + 1
        body = tendon_link_body[link_idx]
        pose = body_q[body]
        center = wp.transform_point(pose, tendon_link_offset[link_idx])
        normal = wp.transform_vector(pose, tendon_link_axis[link_idx])
        arc_rest = oriented_wrap_arc_length(
            seg_attachment_r[seg_left],
            seg_attachment_l[seg_right],
            center,
            tendon_link_radius[link_idx],
            normal,
            tendon_link_orientation[link_idx],
        )
        free_rest = seg_rest_length_step[seg_left] - arc_rest
        if free_rest < 2.0 * min_rest:
            free_rest = 2.0 * min_rest
        len_l = seg_length[seg_left]
        len_r = seg_length[seg_right]
        free_len = wp.max(len_l + len_r, 1e-08)
        rest_l = wp.max(min_rest, free_rest * len_l / free_len)
        rest_r = wp.max(min_rest, free_rest - rest_l)
        seg_rest_length[seg_left] = rest_l
        seg_rest_length[seg_right] = rest_r
    if apply_rolling_transfer != 0 or apply_pinhole_slip != 0:
        for s_snap in range(num_segs):
            seg = seg_offset + s_snap
            if seg_active[seg] != 0:
                len_snap = seg_length[seg]
                seg_stretch[seg] = len_snap - seg_rest_length[seg]
                if seg_active_damping[seg] != 0.0:
                    if damping_from_pose_delta != 0:
                        seg_damping_tension[seg] = seg_active_damping[seg] * tendon_segment_length_rate_from_poses(
                            dt,
                            body_q,
                            body_q_prev,
                            body_com,
                            tendon_link_body,
                            tendon_link_type,
                            tendon_link_offset,
                            tendon_link_axis,
                            seg_active_link_l[seg],
                            seg_active_link_r[seg],
                            seg_attachment_l_local[seg],
                            seg_attachment_r_local[seg],
                            seg_attachment_l[seg],
                            seg_attachment_r[seg],
                        )
                    else:
                        seg_damping_tension[seg] = seg_active_damping[seg] * tendon_segment_length_rate(
                            body_q,
                            body_qd,
                            body_com,
                            tendon_link_body,
                            tendon_link_type,
                            tendon_link_offset,
                            tendon_link_axis,
                            seg_active_link_l[seg],
                            seg_active_link_r[seg],
                            seg_attachment_l[seg],
                            seg_attachment_r[seg],
                        )
        return 1
    return 0


@wp.func
def pack_material(
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
    latch: int,
) -> bool:
    for s in range(num_segs):
        seg = seg_offset + s
        raw_compliance = state.raw_compliance[seg]
        if not wp.isfinite(raw_compliance) or raw_compliance < 1e-25:
            return reject_tendon_material(state, -1, tendon, seg_offset, latch)
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
            return reject_tendon_material(state, -102, tendon, seg_offset, latch)
        if (
            seg_active[left] == 0
            or seg_active[right] == 0
            or state.next_link[left] >= 0
            or (state.incoming[right] >= 0)
        ):
            return reject_tendon_material(state, -102, tendon, seg_offset, latch)
        for skipped in range(left + 1, right):
            if seg_active[skipped] != 0:
                return reject_tendon_material(state, -102, tendon, seg_offset, latch)
        state.next_link[left] = link
        state.incoming[right] = link
        if rolling:
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
                record_component(state, row, count, tendon)
                if state.failure[tendon] != 0:
                    return False
            row = seg
            count = 0
        if count >= 32:
            return reject_tendon_material(state, -103, tendon, row, latch)
        index = row + count
        if state.ids[index] != seg:
            state.valid[row] = 0
        state.ids[index] = seg
        if state.segment_alm:
            state.alm.reference[index] = seg_stretch[seg]
        state.initial[index] = seg_stretch[seg] + seg_compliance[seg] * seg_damping[seg]
        state.compliance[index] = seg_compliance[seg]
        state.upper[index] = seg_length[seg] - min_rest + seg_compliance[seg] * seg_damping[seg]
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
        record_component(state, row, count, tendon)
        if state.failure[tendon] != 0:
            return False
    return True


@wp.func
def publish_material(
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
) -> bool:
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
                state.alm.multiplier[seg] = 0.0
        else:
            seg_stretch[seg] = state.output[row + count] - seg_compliance[seg] * seg_damping[seg]
        count += 1
        previous = seg
    return True


@wp.func
def transfer_cooperative(
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
    lane: int,
    latch: int,
) -> bool:
    ready = int(0)
    if lane == 0:
        ready = int(
            pack_material(
                state,
                tendon,
                link_start,
                num_links,
                seg_offset,
                num_segs,
                tendon_link_type,
                tendon_link_active,
                cone_seg_l,
                cone_seg_r,
                cap_ratio,
                seg_active,
                seg_compliance,
                seg_damping,
                seg_length,
                seg_stretch,
                rolling_delta_l,
                rolling_delta_r,
                apply_rolling_transfer,
                apply_pinhole_slip,
                min_rest,
                latch,
            )
        )
    ready = warp_broadcast(ready)
    if ready == 0:
        return False
    row = int(-1)
    previous = int(-1)
    for s in range(num_segs):
        seg = seg_offset + s
        if seg_active[seg] == 0:
            continue
        if previous < 0:
            row = seg
        elif state.next_link[previous] < 0:
            row = seg
        else:
            previous = seg
            continue
        count = state.count[row]
        if state.segment_alm:
            project_alm_material_cooperative(
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
                lane,
            )
        else:
            project_cooperative(
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
                min_rest,
                64,
                1e-05,
                lane,
            )
        failed = int(0)
        if lane == 0:
            if state.status[row] < 0:
                code = state.status[row]
                if not state.segment_alm:
                    code -= 200
                reject_tendon_material(state, code, tendon, row, latch)
                record_tendon_material_failure_span(state, tendon, row, count, seg_length, seg_compliance, latch)
                failed = 1
        failed = warp_broadcast(failed)
        if failed != 0:
            return False
        previous = seg
    if lane == 0:
        publish_material(
            state,
            tendon,
            link_start,
            num_links,
            seg_offset,
            num_segs,
            tendon_link_type,
            tendon_link_active,
            cone_seg_l,
            cone_seg_r,
            cap_ratio,
            seg_active,
            seg_compliance,
            seg_damping,
            seg_length,
            seg_stretch,
            rolling_delta_l,
            rolling_delta_r,
            apply_rolling_transfer,
            apply_pinhole_slip,
            min_rest,
        )
    ready = warp_broadcast(1)
    return True


@wp.kernel(module="unique", enable_backward=False)
def solve_tendon_material_cooperative(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    tendon_start: wp.array[int],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_orientation: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_rest_length_step: wp.array[float],
    seg_route_rest_length: wp.array[float],
    seg_stretch: wp.array[float],
    seg_damping_tension: wp.array[float],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    tendon_link_active: wp.array[bool],
    tendon_link_active_step: wp.array[bool],
    tendon_link_route_rest_length: wp.array[float],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_length: wp.array[float],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_rolling_delta_l: wp.array[float],
    seg_rolling_delta_r: wp.array[float],
    tendon_link_cone_seg_l: wp.array[int],
    tendon_link_cone_seg_r: wp.array[int],
    tendon_link_cap_ratio: wp.array[float],
    tendon_cone_sweep_count: wp.array[int],
    damping_from_pose_delta: int,
    dt: float,
    apply_rolling_transfer: int,
    apply_pinhole_slip: int,
    adaptive_cone_sweeps: int,
    tendon_max_sweeps: int,
    tendon_settle_tol: float,
    sigmoid_ea_low: float,
    sigmoid_ea_ratio: float,
    sigmoid_transition_strain: float,
    sigmoid_transition_width: float,
    direct: TendonMaterialState,
    latch_failure: int,
    link_material_transfer: wp.array[float],
):
    # Launch-compatible with the scalar kernel, including its ``latch_failure`` semantics: a
    # rejection taken on a discarded trial pose keeps that iteration's rest lengths and is
    # re-tried, and only the accepted pose latches.
    tendon_id = wp.tid() // 32
    lane = wp.tid() % 32
    ready = int(0)
    if lane == 0:
        ready = prepare_material(
            tendon_id,
            body_q,
            body_qd,
            body_q_prev,
            body_com,
            tendon_start,
            tendon_link_body,
            tendon_link_type,
            tendon_link_radius,
            tendon_link_orientation,
            tendon_link_offset,
            tendon_link_axis,
            seg_rest_length,
            seg_rest_length_step,
            seg_route_rest_length,
            seg_stretch,
            seg_damping_tension,
            seg_active,
            seg_active_link_l,
            seg_active_link_r,
            seg_active_compliance,
            seg_active_damping,
            tendon_link_active,
            tendon_link_active_step,
            tendon_link_route_rest_length,
            seg_attachment_l,
            seg_attachment_r,
            seg_length,
            seg_attachment_l_local,
            seg_attachment_r_local,
            seg_rolling_delta_l,
            seg_rolling_delta_r,
            tendon_link_cone_seg_l,
            tendon_link_cone_seg_r,
            tendon_link_cap_ratio,
            tendon_cone_sweep_count,
            damping_from_pose_delta,
            dt,
            apply_rolling_transfer,
            apply_pinhole_slip,
            adaptive_cone_sweeps,
            tendon_max_sweeps,
            tendon_settle_tol,
            sigmoid_ea_low,
            sigmoid_ea_ratio,
            sigmoid_transition_strain,
            sigmoid_transition_width,
            direct,
        )
    ready = warp_broadcast(ready)
    if ready == 0:
        return
    link_start = tendon_start[tendon_id]
    num_links = tendon_start[tendon_id + 1] - link_start
    num_segs = num_links - 1
    seg_offset = link_start - tendon_id
    min_rest = 1e-06
    if not transfer_cooperative(
        direct,
        tendon_id,
        link_start,
        num_links,
        seg_offset,
        num_segs,
        tendon_link_type,
        tendon_link_active,
        tendon_link_cone_seg_l,
        tendon_link_cone_seg_r,
        tendon_link_cap_ratio,
        seg_active,
        seg_active_compliance,
        seg_damping_tension,
        seg_length,
        seg_stretch,
        seg_rolling_delta_l,
        seg_rolling_delta_r,
        apply_rolling_transfer,
        apply_pinhole_slip,
        min_rest,
        lane,
        latch_failure,
    ):
        return
    if lane == 0:
        for s_wb in range(num_segs):
            seg = seg_offset + s_wb
            if seg_active[seg] != 0:
                len_wb = seg_length[seg]
                seg_rest_length[seg] = wp.max(len_wb - seg_stretch[seg], min_rest)
