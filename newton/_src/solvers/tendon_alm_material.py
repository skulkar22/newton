# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental nonlinear ALM material projection with a full-force certificate.

Each inner solve is the global finite-friction linear projection, not a material
sweep. Re-evaluate the secant law and damping engagement until the candidate satisfies
the actual ALM forces and transfer directions. Nonconvergence is explicit.
"""

import warp as wp

from .tendon_material import TendonMaterialStatus, solve_tendon_material_component
from .tendon_material_cooperative import warp_broadcast
from .tendon_material_nonlinear import _secant_compliance_float


@wp.struct
class TendonALMMaterialState:
    reference: wp.array[float]
    extension: wp.array[float]
    offset: wp.array[float]
    force: wp.array[float]
    iterations: wp.array[int]
    length: wp.array[float]
    physical_compliance: wp.array[float]
    damping: wp.array[float]
    damping_force: wp.array[float]
    multiplier: wp.array[float]
    penalty: wp.array[float]
    dt: float


@wp.func
def alm_damping_rate(extension: float, rate: float, dt: float) -> float:
    """Limit positive damping to the taut extension created during this step.

    On an already taut span this is the usual length rate. On engagement it is
    extension/dt, not the slack travel divided by dt. Negative damping can only
    reduce the unilateral pull and is retained to release stored multipliers.
    """
    return wp.min(rate, wp.max(extension, 0.0) / dt)


@wp.func
def alm_material_coefficients(
    state: TendonALMMaterialState,
    seg: int,
    extension: float,
    ea_low: float,
    ea_ratio: float,
    knee: float,
    width: float,
):
    """Express the segment ALM row as T=max((e+offset)/compliance,0)."""
    physical_c = wp.max(state.physical_compliance[seg], 1.0e-8)
    if ea_low > 0.0:
        physical_c = _secant_compliance_float(state.length[seg], extension, ea_low, ea_ratio, knee, width)
    damping_force = wp.min(state.damping_force[seg], 0.0)
    compliance = physical_c
    offset = physical_c * damping_force
    rho = state.penalty[seg]
    metric = 1.0 + physical_c * state.damping[seg] / state.dt
    if rho > 0.0:
        compliance = physical_c + metric / rho
        offset = offset + metric * state.multiplier[seg] / rho
    if state.damping_force[seg] > 0.0 and extension >= 0.0:
        if state.damping_force[seg] > state.damping[seg] * extension / state.dt:
            # Engagement is affine in extension. Include its slope in the inner
            # solve instead of iterating a frozen damping-force offset.
            compliance = compliance / metric
            offset = offset / metric
        else:
            offset += physical_c * state.damping_force[seg]
    return compliance, offset


@wp.func
def alm_material_linearization(
    state: TendonALMMaterialState,
    seg: int,
    extension: float,
    ea_low: float,
    ea_ratio: float,
    knee: float,
    width: float,
):
    """Linearize the full ALM force, including its extension-dependent metric.

    Freezing only the sigmoid secant compliance is not a Newton step: stored
    multipliers are also multiplied by that compliance through the ALM metric.
    Near the knee that iteration can cycle even for a monotone force law.
    """
    c, offset = alm_material_coefficients(state, seg, extension, ea_low, ea_ratio, knee, width)
    if ea_low <= 0.0:
        return c, offset
    length = state.length[seg]
    rest = wp.max(length - extension, 1.0e-8)
    rest_derivative = float(-1.0)
    if length - extension <= 1.0e-8:
        rest_derivative = 0.0
    strain = wp.max(extension, 0.0) / rest
    strain_derivative = float(0.0)
    if extension > 0.0:
        strain_derivative = (rest - extension * rest_derivative) / (rest * rest)
    transition = wp.tanh((strain - knee) / width)
    delta = 0.5 * ea_low * (ea_ratio - 1.0)
    ea = ea_low + delta * (1.0 + transition)
    ea_derivative = delta * (1.0 - transition * transition) * strain_derivative / width
    physical_c = rest / ea
    physical_derivative = rest_derivative / ea - physical_c * ea_derivative / ea
    metric = 1.0 + physical_c * state.damping[seg] / state.dt
    metric_derivative = physical_derivative * state.damping[seg] / state.dt
    c_derivative = physical_derivative
    offset_derivative = physical_derivative * wp.min(state.damping_force[seg], 0.0)
    rho = state.penalty[seg]
    if rho > 0.0:
        c_derivative += metric_derivative / rho
        offset_derivative += metric_derivative * state.multiplier[seg] / rho
    if state.damping_force[seg] > 0.0 and extension >= 0.0:
        if state.damping_force[seg] > state.damping[seg] * extension / state.dt:
            c_derivative = (c_derivative - c * metric_derivative) / metric
            offset_derivative = (offset_derivative - offset * metric_derivative) / metric
        else:
            offset_derivative += physical_derivative * state.damping_force[seg]
    force = (extension + offset) / c
    tangent = (1.0 + offset_derivative - force * c_derivative) / c
    # A positive tangent is needed by the convex inner projection. An unsuitable
    # local tangent retains the positive secant preconditioner, not a sweep solve;
    # only the exact-force certificate below can accept its candidate.
    if tangent > 0.0 and wp.isfinite(tangent):
        return 1.0 / tangent, force / tangent - extension
    return c, offset


@wp.func_native("""
#ifdef __CUDA_ARCH__
return __shfl_sync(0xffffffffu, value, 0);
#else
return value;
#endif
""")
def _broadcast_float(value: float) -> float: ...


def _make_project_alm_material(cooperative: bool):
    """Share the force law and certificate between scalar and full-warp execution."""

    @wp.func
    def project_alm_material(
        state: TendonALMMaterialState,
        ids: wp.array[int],
        initial: wp.array[float],
        compliance: wp.array[float],
        cap: wp.array[float],
        upper: wp.array[float],
        output: wp.array[float],
        status: wp.array[int],
        faces: wp.array[int],
        valid: wp.array[int],
        pieces: wp.array[int],
        row: int,
        count: int,
        ea_low: float,
        ea_ratio: float,
        knee: float,
        width: float,
        lane: int = 0,
    ) -> bool:
        stride = int(1)
        if wp.static(cooperative):
            stride = 32
        reference_sum = float(0.0)
        for slot in range(count):
            reference_sum += state.reference[row + slot]
        for slot in range(lane, count, stride):
            state.extension[row + slot] = state.reference[row + slot]
        if wp.static(cooperative):
            warp_broadcast(0)
        for iteration in range(128):
            if lane == 0:
                state.iterations[row] = iteration + 1
            for slot in range(lane, count, stride):
                i = row + slot
                seg = ids[i]
                trial_extension = state.extension[i]
                if iteration == 0 and reference_sum <= 0.0:
                    # Try the all-slack branch first when the material inventory permits it.
                    # Publication still requires the full-force certificate below.
                    trial_extension = wp.min(trial_extension, 0.0)
                c, offset = alm_material_linearization(state, seg, trial_extension, ea_low, ea_ratio, knee, width)
                initial[i] = state.reference[i] + offset
                compliance[i] = c
                state.offset[i] = offset
                upper[i] = state.length[seg] - 1.0e-6 + offset
            if wp.static(cooperative):
                warp_broadcast(0)
            if lane == 0:
                solve_tendon_material_component(
                    initial,
                    compliance,
                    cap,
                    upper,
                    output,
                    status,
                    faces,
                    valid,
                    pieces,
                    count,
                    1,
                    row,
                    1,
                    1,
                )
            if wp.static(cooperative):
                warp_broadcast(0)
            if status[row] == -3:
                # A tangent's unconstrained root can cross the physical rest-length
                # boundary even when the nonlinear root is feasible. Obtain its
                # direction, then limit the outer step below; never publish it.
                for slot in range(lane, count, stride):
                    upper[row + slot] = 1.0e30
                if wp.static(cooperative):
                    warp_broadcast(0)
                if lane == 0:
                    solve_tendon_material_component(
                        initial, compliance, cap, upper, output, status, faces, valid, pieces, count, 1, row, 1, 1
                    )
                if wp.static(cooperative):
                    warp_broadcast(0)
            if status[row] < 0:
                return False
            force_scale = float(1.0e-5)
            flow_scale = float(1.0e-12)
            if lane == 0:
                for slot in range(count):
                    i = row + slot
                    # Use the linearized offset before the parallel evaluation
                    # replaces it with the actual force-row offset below.
                    physical_extension = output[i] - state.offset[i]
                    flow_scale += wp.abs(state.reference[i]) + wp.abs(physical_extension) + wp.abs(state.offset[i])
            if wp.static(cooperative):
                warp_broadcast(0)
            for slot in range(lane, count, stride):
                i = row + slot
                seg = ids[i]
                output[i] = output[i] - state.offset[i]
                c, offset = alm_material_coefficients(state, seg, output[i], ea_low, ea_ratio, knee, width)
                state.force[i] = wp.max((output[i] + offset) / c, 0.0)
                if status[row] == int(TendonMaterialStatus.SLACK_UNCHANGED) or status[row] == int(
                    TendonMaterialStatus.SLACK_REDISTRIBUTED
                ):
                    # The inner solution has zero force. Undoing its offset can
                    # leave a positive roundoff residue, including subnormals.
                    # Retain zero only within the local arithmetic error bound;
                    # a genuinely positive nonlinear force still needs iteration.
                    roundoff = 8.0 * 1.1920928955078125e-7 * (wp.abs(output[i]) + wp.abs(offset)) / c
                    if state.force[i] <= roundoff:
                        state.force[i] = 0.0
                # Keep the actual row coefficients for a float backward-error bound.
                # A small net force may subtract a large viscous/dual offset, so its
                # relative rounding error cannot be bounded by the net force alone.
                compliance[i] = c
                state.offset[i] = offset
                upper[i] = state.length[seg] - 1.0e-6 + offset
            if wp.static(cooperative):
                warp_broadcast(0)
            certified = bool(True)
            step_fraction = float(0.5)
            adjusted_metric = bool(False)
            if lane == 0:
                for slot in range(count):
                    i = row + slot
                    force_scale = wp.max(force_scale, state.force[i])
                force_tol = 2.0e-5 * force_scale
                flow_tol = 2.0e-6 * flow_scale
                flow = float(0.0)
                for slot in range(count):
                    i = row + slot
                    flow += state.reference[i] - output[i]
                    if not wp.isfinite(output[i]) or not wp.isfinite(state.force[i]):
                        certified = False
                    bound = state.length[ids[i]] - 1.0e-6
                    if output[i] > bound:
                        certified = False
                        direction = output[i] - state.extension[i]
                        if direction > 0.0:
                            step_fraction = wp.min(
                                step_fraction, 0.9 * wp.max(bound - state.extension[i], 0.0) / direction
                            )
                        seg = ids[i]
                        if state.penalty[seg] > 0.0 and state.extension[i] >= bound - 0.01 * wp.abs(bound):
                            # A finite ALM penalty puts an artificial force ceiling on
                            # a short span, even when the physical material is feasible.
                            # Once Newton reaches that ceiling, strengthen this row's
                            # metric and restart the projection. Body/dual evaluation
                            # shares this array, so it sees the same adjusted force law.
                            physical_c = wp.max(state.physical_compliance[seg], 1.0e-8)
                            if ea_low > 0.0:
                                physical_c = _secant_compliance_float(
                                    state.length[seg], bound, ea_low, ea_ratio, knee, width
                                )
                            limit = (1.0 / physical_c + state.damping[seg] / state.dt) / 1.1920928955078125e-7
                            rho = wp.min(2.0 * state.penalty[seg], limit)
                            if rho > state.penalty[seg]:
                                state.penalty[seg] = rho
                                adjusted_metric = True
                    if slot + 1 < count:
                        tl = state.force[i]
                        tr = state.force[i + 1]
                        ratio = cap[i]
                        left_roundoff = (
                            8.0 * 1.1920928955078125e-7 * (wp.abs(output[i]) + wp.abs(state.offset[i])) / compliance[i]
                        )
                        right_roundoff = (
                            8.0
                            * 1.1920928955078125e-7
                            * (wp.abs(output[i + 1]) + wp.abs(state.offset[i + 1]))
                            / compliance[i + 1]
                        )
                        positive_tol = force_tol + left_roundoff + ratio * right_roundoff
                        negative_tol = force_tol + right_roundoff + ratio * left_roundoff
                        if tl > ratio * tr + positive_tol or tr > ratio * tl + negative_tol:
                            certified = False
                        if flow > flow_tol and wp.abs(tl - ratio * tr) > positive_tol:
                            certified = False
                        if flow < -flow_tol and wp.abs(tr - ratio * tl) > negative_tol:
                            certified = False
                if wp.abs(flow) > flow_tol:
                    certified = False
            if wp.static(cooperative):
                certified = bool(warp_broadcast(int(certified)))
                adjusted_metric = bool(warp_broadcast(int(adjusted_metric)))
                step_fraction = _broadcast_float(step_fraction)
            if certified:
                return True
            if adjusted_metric:
                for slot in range(lane, count, stride):
                    state.extension[row + slot] = state.reference[row + slot]
                continue
            # Damping limits steps that cross a sigmoid knee or a unilateral engagement boundary.
            for slot in range(lane, count, stride):
                i = row + slot
                state.extension[i] += step_fraction * (output[i] - state.extension[i])
        if lane == 0:
            status[row] = -120
        return False

    return project_alm_material


project_alm_material = _make_project_alm_material(False)
project_alm_material_cooperative = _make_project_alm_material(True)
