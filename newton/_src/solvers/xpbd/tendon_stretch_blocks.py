# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shape-independent XPBD stretch rows with condensed sliding material modes."""

import warp as wp

from ..tendon_kernels import tendon_material_tangent, tendon_material_tension


@wp.struct
class TendonStretchRow:
    body_l: int
    body_r: int
    linear: wp.vec3
    angular_l: wp.vec3
    angular_r: wp.vec3
    residual: float
    compliance: float
    motion_scale: float
    tension: float
    weight: float
    next: int


@wp.kernel
def prepare_stretch_rows(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    link_body: wp.array[int],
    active: wp.array[int],
    link_l: wp.array[int],
    link_r: wp.array[int],
    attachment_l: wp.array[wp.vec3],
    attachment_r: wp.array[wp.vec3],
    rest: wp.array[float],
    stretch: wp.array[float],
    compliance: wp.array[float],
    damping: wp.array[float],
    dt: float,
    ea_low: float,
    ea_ratio: float,
    transition_strain: float,
    transition_width: float,
    rows: wp.array[TendonStretchRow],
    material_tension: wp.array[float],
):
    """Build physical contact Jacobians without knowing the roller shape."""
    seg = wp.tid()
    row = TendonStretchRow()
    row.body_l = -1
    row.body_r = -1
    row.next = -1
    material_tension[seg] = 0.0
    if active[seg] != 0:
        bl = link_body[link_l[seg]]
        br = link_body[link_r[seg]]
        pl = attachment_l[seg]
        pr = attachment_r[seg]
        length = wp.length(pr - pl)
        if length > 1.0e-8:
            row.body_l = bl
            row.body_r = br
            row.linear = (pr - pl) / length
            row.angular_l = -wp.cross(pl - wp.transform_point(body_q[bl], body_com[bl]), row.linear)
            row.angular_r = wp.cross(pr - wp.transform_point(body_q[br], body_com[br]), row.linear)
            rate = wp.dot(row.linear, wp.spatial_top(body_qd[br]) - wp.spatial_top(body_qd[bl]))
            rate += wp.dot(row.angular_l, wp.spatial_bottom(body_qd[bl]))
            rate += wp.dot(row.angular_r, wp.spatial_bottom(body_qd[br]))
            force = tendon_material_tension(
                length, rest[seg], compliance[seg], ea_low, ea_ratio, transition_strain, transition_width
            )
            if ea_low <= 0.0:
                # Keep the material solve's small stretch, rather than losing it
                # by subtracting rounded lengths again on a stiff cable.
                force = wp.max(stretch[seg], 0.0) / wp.max(compliance[seg], 1.0e-30)
            row.compliance = compliance[seg]
            row.residual = stretch[seg]
            row.motion_scale = dt + compliance[seg] * damping[seg]
            if ea_low > 0.0:
                tangent = tendon_material_tangent(
                    length, rest[seg], compliance[seg], ea_low, ea_ratio, transition_strain, transition_width
                )
                # Material transfer varies rest length, not geometric length.
                # For T(strain), -dT/drest = dT/dlength * length/rest.
                row.compliance = wp.max(rest[seg], 1.0e-8) / (wp.max(length, 1.0e-8) * tangent)
                row.residual = row.compliance * force
                row.motion_scale = row.compliance * (dt * tangent + damping[seg])
            row.residual += row.compliance * damping[seg] * rate
            row.tension = wp.max(force + damping[seg] * rate, 0.0)
            if ea_low <= 0.0:
                # Classify the capstan branch with the same signed elastic
                # term as the residual, not a separately clamped spring force.
                row.tension = wp.max(row.residual / wp.max(row.compliance, 1.0e-30), 0.0)
            row.weight = 1.0
            material_tension[seg] = force
    rows[seg] = row


@wp.func
def _body_row(row: TendonStretchRow, body: int):
    linear = wp.vec3(0.0)
    angular = wp.vec3(0.0)
    if body == row.body_l:
        linear -= row.linear
        angular += row.angular_l
    if body == row.body_r:
        linear += row.linear
        angular += row.angular_r
    return linear, angular


@wp.func
def _inverse_mass_product(
    a: TendonStretchRow,
    b: TendonStretchRow,
    body_q: wp.array[wp.transform],
    inv_mass: wp.array[float],
    inv_inertia: wp.array[wp.mat33],
):
    result = float(0.0)
    for side in range(2):
        body = a.body_l if side == 0 else a.body_r
        if body < 0 or (side == 1 and a.body_l == a.body_r):
            continue
        la, aa = _body_row(a, body)
        lb, ab = _body_row(b, body)
        rotation = wp.transform_get_rotation(body_q[body])
        aa = wp.quat_rotate_inv(rotation, aa)
        ab = wp.quat_rotate_inv(rotation, ab)
        result += inv_mass[body] * wp.dot(la, lb) + wp.dot(aa, inv_inertia[body] * ab)
    return result


@wp.kernel
def solve_stretch_blocks(
    body_q: wp.array[wp.transform],
    inv_mass: wp.array[float],
    inv_inertia: wp.array[wp.mat33],
    tendon_start: wp.array[int],
    cone_l: wp.array[int],
    cone_r: wp.array[int],
    cap_ratio: wp.array[float],
    rest: wp.array[float],
    dt: float,
    relaxation: float,
    settle_tol: float,
    rows: wp.array[TendonStretchRow],
    impulse: wp.array[float],
    delta_impulse: wp.array[float],
    body_deltas: wp.array[wp.spatial_vector],
):
    """Condense capstan-bound spans into one impulse mode, leaving sticking spans independent.

    Sliding fixes neighboring tension ratios. Summing the corresponding material
    equations eliminates their internal transfer. The resulting scalar solve uses
    all cross terms of J M^-1 J^T, including repeated bodies and opposing moments.
    No dense matrix or shape-specific spin correction is needed.
    """
    tendon = wp.tid()
    first = tendon_start[tendon] - tendon
    end = tendon_start[tendon + 1] - tendon - 1
    for seg in range(first, end):
        delta_impulse[seg] = 0.0
        if rows[seg].body_l < 0:
            impulse[seg] = 0.0

    # Record the active sliding connections. Depleted spans cannot freely
    # exchange material, even if their contact has zero friction.
    for link in range(tendon_start[tendon] + 1, tendon_start[tendon + 1] - 1):
        left = cone_l[link]
        right = cone_r[link]
        if left < first or right >= end or left < 0 or right <= left:
            continue
        l = rows[left]
        r = rows[right]
        if l.body_l < 0 or r.body_l < 0 or rest[left] <= 1.001e-6 or rest[right] <= 1.001e-6:
            continue
        ratio = cap_ratio[link]
        scale = wp.max(l.tension, r.tension)
        tolerance = wp.max(4.0 * settle_tol, 2.0e-4) * scale
        factor = float(0.0)
        if ratio == 1.0:
            factor = 1.0
        elif scale > 1.0e-8:
            if wp.abs(l.tension - ratio * r.tension) <= tolerance:
                factor = 1.0 / ratio
            elif wp.abs(r.tension - ratio * l.tension) <= tolerance:
                factor = ratio
        if factor > 0.0:
            l.next = right
            r.weight = factor
            rows[left] = l
            rows[right] = r

    first_group = first
    while first_group < end:
        if rows[first_group].body_l < 0:
            first_group += 1
            continue
        last = first_group
        root = rows[first_group]
        root.weight = 0.0
        rows[first_group] = root
        maximum = float(0.0)
        while rows[last].next >= 0:
            next_seg = rows[last].next
            row = rows[next_seg]
            row.weight = wp.log(row.weight) + rows[last].weight
            maximum = wp.max(maximum, row.weight)
            rows[next_seg] = row
            last = next_seg
        # Log weights avoid overflowing products of many capstan ratios.
        for seg in range(first_group, last + 1):
            row = rows[seg]
            row.weight = wp.exp(row.weight - maximum)
            rows[seg] = row
        rhs = float(0.0)
        denominator = float(0.0)
        for i in range(first_group, last + 1):
            a = rows[i]
            if a.body_l < 0:
                continue
            rhs -= a.residual
            denominator += a.compliance * a.weight / dt
            for j in range(first_group, last + 1):
                b = rows[j]
                if b.body_l < 0:
                    continue
                mass = a.motion_scale * _inverse_mass_product(a, b, body_q, inv_mass, inv_inertia)
                denominator += mass * b.weight
                rhs += mass * impulse[j]
        if denominator <= 0.0:
            if wp.abs(rhs) > 1.0e-9:
                wp.printf(
                    "ERROR: Tendon %d has a singular or non-monotone sliding stretch block at segment %d.\n",
                    tendon,
                    first_group,
                )
            first_group = last + 1
            continue
        target = wp.min(rhs / denominator, 0.0)
        for seg in range(first_group, last + 1):
            row = rows[seg]
            if row.body_l < 0:
                continue
            delta = relaxation * (row.weight * target - impulse[seg])
            impulse[seg] += delta
            delta_impulse[seg] = delta
            delta_l = wp.spatial_vector(-row.linear * delta, row.angular_l * delta)
            delta_r = wp.spatial_vector(row.linear * delta, row.angular_r * delta)
            if row.body_l == row.body_r:
                wp.atomic_add(body_deltas, row.body_l, delta_l + delta_r)
            else:
                wp.atomic_add(body_deltas, row.body_l, delta_l)
                wp.atomic_add(body_deltas, row.body_r, delta_r)
        first_group = last + 1
