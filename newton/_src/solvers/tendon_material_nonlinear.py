# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact-block sigmoid projection and shared numerical certificates.

A feasible log-force active set enforces finite-friction capstan constraints.
Flat force branches choose minimum squared net material transfer. Every trial
uses the same full-rolling reference, never an earlier slip candidate. Only
certified force/flow equilibrium satisfying conservation and rest bounds may
publish an extension. Float32 storage error is reported separately and never
relaxes the physical solve tolerance. Failure does not fall back to sweeps.
"""

from enum import IntEnum

import warp as wp


class TendonMaterialNonlinearStatus(IntEnum):
    """Result of the experimental nonlinear component projection."""

    CONVERGED = 1
    CONVERGED_ROUNDED = 2
    BAD_INPUT = -1
    INNER_FAILURE = -2
    UNSUPPORTED_SLACK = -3
    MAX_ITERATIONS = -4
    NUMERICAL_FAILURE = -5
    BOUND_INCOMPATIBLE = -6


@wp.struct
class TendonMaterialNonlinearState:
    reference: wp.array[float]
    length: wp.array[float]
    damping: wp.array[float]
    cap: wp.array[float]
    output: wp.array[float]
    candidate: wp.array[float]
    tension: wp.array[wp.float64]
    exact_tension: wp.array[wp.float64]
    endpoint: wp.array[wp.float64]
    log_force: wp.array[wp.float64]
    target_log_force: wp.array[wp.float64]
    plateau: wp.array[int]
    corrected: wp.array[wp.float64]
    faces: wp.array[int]
    valid: wp.array[int]
    status: wp.array[int]
    inner_status: wp.array[int]
    outer_iterations: wp.array[int]
    residual: wp.array[float]
    publication_residual: wp.array[float]
    storage_error: wp.array[wp.vec4d]


def allocate_tendon_material_nonlinear_state(
    span_count: int, edge_count: int, component_count: int, *, device=None
) -> TendonMaterialNonlinearState:
    """Allocate reusable packed input, candidate, cache, and accepted storage."""
    state = TendonMaterialNonlinearState()
    for name in (
        "reference",
        "length",
        "damping",
        "output",
        "candidate",
    ):
        setattr(state, name, wp.zeros(span_count, dtype=float, device=device))
    state.tension = wp.zeros(span_count, dtype=wp.float64, device=device)
    state.exact_tension = wp.zeros(span_count, dtype=wp.float64, device=device)
    state.endpoint = wp.zeros(span_count, dtype=wp.float64, device=device)
    state.log_force = wp.zeros(span_count, dtype=wp.float64, device=device)
    state.target_log_force = wp.zeros(span_count, dtype=wp.float64, device=device)
    state.corrected = wp.zeros(span_count, dtype=wp.float64, device=device)
    state.plateau = wp.zeros(span_count, dtype=int, device=device)
    state.cap = wp.zeros(edge_count, dtype=float, device=device)
    state.faces = wp.zeros(edge_count, dtype=int, device=device)
    for name in ("valid", "status", "inner_status", "outer_iterations"):
        setattr(state, name, wp.zeros(component_count, dtype=int, device=device))
    state.residual = wp.zeros(component_count, dtype=float, device=device)
    state.publication_residual = wp.zeros(component_count, dtype=float, device=device)
    state.storage_error = wp.zeros(component_count, dtype=wp.vec4d, device=device)
    return state


@wp.func
def _secant_compliance(
    length: wp.float64,
    extension: wp.float64,
    ea_low: wp.float64,
    ea_ratio: wp.float64,
    transition_strain: wp.float64,
    transition_width: wp.float64,
) -> wp.float64:
    rest = wp.max(length - extension, wp.float64(1.0e-8))
    strain = wp.max(extension, wp.float64(0.0)) / rest
    transition = wp.tanh((strain - transition_strain) / transition_width)
    ea = ea_low * (wp.float64(1.0) + (ea_ratio - wp.float64(1.0)) * wp.float64(0.5) * (wp.float64(1.0) + transition))
    return rest / ea


@wp.func_native("""
union { float f; unsigned int u; } bits;
bits.f = static_cast<float>(value);
if (double(bits.f) > value) --bits.u;
return bits.f;
""")
def _round_nonnegative_down(value: wp.float64) -> float:
    """Round a nonnegative physical upper bound without enlarging it."""
    ...


@wp.func_native("""
if (!(value > 0.0)) return 0.0;
union { double f; unsigned long long u; } bits;
bits.f = value;
--bits.u;
return bits.f;
""")
def _next_double_down(value: wp.float64) -> wp.float64:
    """Take one representable nonnegative step below a zero-force endpoint."""
    ...


@wp.func
def _inverse_strain_hint(
    target: float, ea_low: float, ea_ratio: float, knee: float, width: float, hint: float
) -> float:
    """Cheap starting guess only; the double solve still determines acceptance."""
    lower = target / (ea_low * ea_ratio)
    upper = target / ea_low
    strain = 0.5 * (lower + upper)
    if hint >= lower and hint <= upper:
        strain = hint
    for _iteration in range(8):
        transition = wp.tanh((strain - knee) / width)
        delta = 0.5 * ea_low * (ea_ratio - 1.0)
        ea = ea_low + delta * (1.0 + transition)
        slope = delta * (1.0 - transition * transition) / width
        error = ea * strain - target
        if wp.abs(error) <= 5.0e-5 * target:
            break
        if error < 0.0:
            lower = strain
        else:
            upper = strain
        trial = strain - error / (ea + strain * slope)
        if not wp.isfinite(trial) or trial < lower or trial > upper:
            trial = 0.5 * (lower + upper)
        strain = trial
    return strain


@wp.func
def _inverse_extension_and_tangent(
    target: wp.float64,
    length: wp.float64,
    ea_low: wp.float64,
    ea_ratio: wp.float64,
    transition_strain: wp.float64,
    transition_width: wp.float64,
    strain_hint: wp.float64,
) -> wp.vec2d:
    """Return extension and its force derivative at fixed geometric length."""
    if target <= wp.float64(0.0):
        transition = wp.tanh(-transition_strain / transition_width)
        ea = ea_low + wp.float64(0.5) * ea_low * (ea_ratio - wp.float64(1.0)) * (wp.float64(1.0) + transition)
        return wp.vec2d(wp.float64(0.0), ea / wp.max(length, wp.float64(1.0e-8)))
    lower = target / (ea_low * ea_ratio)
    upper = target / ea_low
    strain = wp.float64(0.5) * (lower + upper)
    if strain_hint >= lower and strain_hint <= upper:
        strain = strain_hint
    # Float precision supplies only a guess. The double bracket, refinement,
    # and stopping criterion below remain authoritative.
    coarse = _inverse_strain_hint(
        float(target), float(ea_low), float(ea_ratio), float(transition_strain), float(transition_width), float(strain)
    )
    if wp.isfinite(coarse) and wp.float64(coarse) >= lower and wp.float64(coarse) <= upper:
        strain = wp.float64(coarse)
    previous_step = upper - lower
    last_step = previous_step
    evaluated_strain = strain
    ea = ea_low
    derivative = wp.float64(0.0)
    for _iteration in range(64):
        evaluated_strain = strain
        transition = wp.tanh((strain - transition_strain) / transition_width)
        delta = wp.float64(0.5) * ea_low * (ea_ratio - wp.float64(1.0))
        ea = ea_low + delta * (wp.float64(1.0) + transition)
        derivative = delta * (wp.float64(1.0) - transition * transition) / transition_width
        residual = ea * strain - target
        if wp.abs(residual) <= wp.float64(7.2e-15) * target:
            break
        if residual < wp.float64(0.0):
            lower = strain
        else:
            upper = strain
        newton_step = residual / (ea + strain * derivative)
        trial = strain - newton_step
        # Two-step progress prevents the knee's Newton cycle without rejecting
        # an excellent proposal just because it approaches a bracket endpoint.
        # Keep both prior steps: a single-step test can repeatedly reject the
        # exact root at the endpoint of a linear, saturated material branch.
        if (
            not wp.isfinite(trial)
            or trial < lower
            or trial > upper
            or wp.abs(newton_step) > wp.float64(0.5) * previous_step
        ):
            trial = wp.float64(0.5) * (lower + upper)
        previous_step = last_step
        last_step = wp.abs(trial - strain)
        strain = trial
    extension = length * (strain / (wp.float64(1.0) + strain))
    if length - extension < wp.float64(1.0e-8):
        extension = strain * wp.float64(1.0e-8)
    # A capped loop may leave the last Newton proposal unevaluated.
    if strain != evaluated_strain:
        transition = wp.tanh((strain - transition_strain) / transition_width)
        delta = wp.float64(0.5) * ea_low * (ea_ratio - wp.float64(1.0))
        ea = ea_low + delta * (wp.float64(1.0) + transition)
        derivative = delta * (wp.float64(1.0) - transition * transition) / transition_width
    rest = wp.max(length - extension, wp.float64(1.0e-8))
    tangent = length / (rest * rest) * (ea + strain * derivative)
    if length - extension <= wp.float64(1.0e-8):
        tangent = (ea + strain * derivative) / wp.float64(1.0e-8)
    return wp.vec2d(extension, tangent)


@wp.func
def _inverse_extension(
    target: wp.float64,
    length: wp.float64,
    ea_low: wp.float64,
    ea_ratio: wp.float64,
    transition_strain: wp.float64,
    transition_width: wp.float64,
) -> wp.float64:
    return _inverse_extension_and_tangent(
        target, length, ea_low, ea_ratio, transition_strain, transition_width, wp.float64(-1.0)
    )[0]


@wp.func_native("""
double prefix[33], fitted[33], lower[33], upper[33], sums[33], value[33];
int start[33], end[33], weight[33];
prefix[0] = 0.0;
double scale = 1.0e-30;
for (int i = 0; i < n; ++i) {
    const double ref = reference[offset+i], u = endpoint[offset+i];
    prefix[i+1] = prefix[i] + (ref-u);
    scale += wp::abs(ref) + wp::abs(u);
}
const double roundoff = 128.0*n*2.220446049250313e-16*scale;
int blocks = 0, first = 0;
while (first <= n) {
    int last = first;
    while (last < n && plateau[offset+last] == 0) ++last;
    double lo = -1.0e300, hi = 1.0e300, sum = 0.0;
    int w = 0;
    for (int node = first; node <= last; ++node) {
        if (node == 0) { lo = wp::max(lo,0.0); hi = wp::min(hi,0.0); }
        else if (node == n) { lo = wp::max(lo,-prefix[n]); hi = wp::min(hi,-prefix[n]); }
        else {
            sum -= prefix[node];
            ++w;
            const int sign = faces[edge_offset+node-1];
            // Equal-force capstan faces do not restrict the sign of transfer.
            if (restrict_signs && cap[edge_offset+node-1] != 1.0f) {
                if (sign >= 0) lo = wp::max(lo,-prefix[node]);
                if (sign <= 0) hi = wp::min(hi,-prefix[node]);
            }
        }
    }
    if (lo > hi) {
        if (lo-hi > roundoff) return 0;
        lo = hi = 0.5*(lo+hi);
    }
    start[blocks] = first;
    end[blocks] = last;
    lower[blocks] = lo;
    upper[blocks] = hi;
    sums[blocks] = sum;
    weight[blocks] = w;
    value[blocks] = w ? wp::max(lo,wp::min(hi,sum/w)) : lo;
    ++blocks;
    while (blocks > 1 && value[blocks-2] > value[blocks-1]) {
        const int left = blocks-2, right = blocks-1;
        lower[left] = wp::max(lower[left],lower[right]);
        upper[left] = wp::min(upper[left],upper[right]);
        if (lower[left] > upper[left]) {
            if (lower[left]-upper[left] > roundoff) return 0;
            lower[left] = upper[left] = 0.5*(lower[left]+upper[left]);
        }
        sums[left] += sums[right];
        weight[left] += weight[right];
        end[left] = end[right];
        value[left] = weight[left]
            ? wp::max(lower[left],wp::min(upper[left],sums[left]/weight[left])) : lower[left];
        --blocks;
    }
    first = last+1;
}
for (int b = 0; b < blocks; ++b) {
    if (!wp::isfinite(value[b])) return -1;
    for (int node = start[b]; node <= end[b]; ++node) fitted[node] = value[b];
}
for (int i = 0; i < n; ++i) {
    // Expand reference - (flow_r-flow_l) using prefix[i+1]-prefix[i]
    // = reference-endpoint. This retains tiny taut endpoints exactly when
    // adjacent fitted values coincide, even if the transferred slack is large.
    const double result = endpoint[offset+i] - (fitted[i+1]-fitted[i]);
    if (!wp::isfinite(result)) return -1;
    if (plateau[offset+i] ? result > endpoint[offset+i]+roundoff
                          : wp::abs(result-endpoint[offset+i]) > roundoff) return 0;
    output[offset+i] = result;
}
return 1;
""")
def _allocate_plateau_block(
    reference: wp.array[float],
    endpoint: wp.array[wp.float64],
    plateau: wp.array[int],
    cap: wp.array[float],
    faces: wp.array[int],
    output: wp.array[wp.float64],
    offset: int,
    edge_offset: int,
    n: int,
    restrict_signs: int = 1,
) -> int:
    """Choose minimum squared net transfer at fixed force and capstan faces."""
    ...


@wp.func
def _measure_candidate(
    state: TendonMaterialNonlinearState,
    n: int,
    offset: int,
    edge_offset: int,
    ea_low: float,
    ea_ratio: float,
    transition_strain: float,
    transition_width: float,
    min_rest: float,
    tolerance: float,
    exact: bool = False,
) -> wp.vec2d:
    """Measure true-law residual and squared net transfer for a face trial."""
    peak = wp.float64(1.0e-20)
    minimum_compliance = wp.float64(1.0e300)
    scale = wp.float64(1.0e-12)
    for i in range(n):
        index = offset + i
        extension = wp.float64(state.candidate[index])
        if exact:
            extension = state.corrected[index]
        length = wp.float64(state.length[index])
        if not wp.isfinite(extension) or length - extension < wp.float64(min_rest):
            return wp.vec2d(wp.float64(1.0e30), wp.float64(1.0e300))
        compliance = _secant_compliance(
            length,
            extension,
            wp.float64(ea_low),
            wp.float64(ea_ratio),
            wp.float64(transition_strain),
            wp.float64(transition_width),
        )
        tension = wp.max(
            wp.max(extension, wp.float64(0.0)) / compliance + wp.float64(state.damping[index]), wp.float64(0.0)
        )
        if not wp.isfinite(tension):
            return wp.vec2d(wp.float64(1.0e30), wp.float64(1.0e300))
        state.tension[index] = tension
        peak = wp.max(peak, tension)
        minimum_compliance = wp.min(minimum_compliance, compliance)
        scale += wp.abs(extension) + wp.abs(wp.float64(state.reference[index]))
    flow = wp.float64(0.0)
    objective = wp.float64(0.0)
    error = wp.float64(0.0)
    # A prefix flow only selects a capstan face once it exceeds the accuracy the
    # allocation itself can promise. Conservation holds to the same prefix
    # roundoff the block allocator uses, which scales with the largest packed
    # extension rather than with the force tolerance. Without that floor, a
    # slack component whose peak tension is pure damping demands a saturated
    # face at a stationary block boundary and can never be certified.
    flow_tolerance = wp.max(
        wp.float64(tolerance) * peak * minimum_compliance,
        wp.float64(128.0 * 2.220446049250313e-16) * wp.float64(n) * scale,
    )
    for i in range(n):
        extension = wp.float64(state.candidate[offset + i])
        if exact:
            extension = state.corrected[offset + i]
        flow += wp.float64(state.reference[offset + i]) - extension
        if i < n - 1:
            left = state.tension[offset + i]
            right = state.tension[offset + i + 1]
            cap = wp.float64(state.cap[edge_offset + i])
            positive = cap * right - left
            negative = cap * left - right
            error = wp.max(error, wp.max(-positive, -negative))
            if flow > flow_tolerance:
                error = wp.max(error, wp.abs(positive))
            elif flow < -flow_tolerance:
                error = wp.max(error, wp.abs(negative))
            objective += flow * flow
    return wp.vec2d(wp.max(error / peak, wp.abs(flow) / scale), objective)


@wp.func
def _secant_compliance_float(
    length: wp.float32,
    extension: wp.float32,
    ea_low: wp.float32,
    ea_ratio: wp.float32,
    transition_strain: wp.float32,
    transition_width: wp.float32,
) -> wp.float32:
    rest = wp.max(length - extension, wp.float32(1.0e-8))
    strain = wp.max(extension, wp.float32(0.0)) / rest
    transition = wp.tanh((strain - transition_strain) / transition_width)
    ea = ea_low * (wp.float32(1.0) + (ea_ratio - wp.float32(1.0)) * wp.float32(0.5) * (wp.float32(1.0) + transition))
    return rest / ea


@wp.func
def _initialize_component(
    state: TendonMaterialNonlinearState,
    n: int,
    tendon: int,
    span_stride: int,
    edge_stride: int,
    ea_low: float,
    ea_ratio: float,
    transition_strain: float,
    transition_width: float,
    min_rest: float,
    tolerance: float,
) -> bool:
    """Handle feasible references and zero force before choosing a taut guess.

    Return true if the invocation is complete, including explicit failure.
    Coefficient initialization never changes the original material reference.
    """
    offset = tendon * span_stride
    edge_offset = tendon * edge_stride
    peak = wp.float64(1e-20)
    total = wp.float64(0.0)
    all_nonpositive_damping = bool(True)
    for i in range(n):
        index = offset + i
        extension = wp.float64(state.reference[index])
        length = wp.float64(state.length[index])
        damping = wp.float64(state.damping[index])
        force = wp.float64(
            wp.max(
                wp.max(state.reference[index], 0.0)
                / _secant_compliance_float(
                    state.length[index], state.reference[index], ea_low, ea_ratio, transition_strain, transition_width
                )
                + state.damping[index],
                0.0,
            )
        )
        if not wp.isfinite(force):
            state.status[tendon] = int(TendonMaterialNonlinearStatus.NUMERICAL_FAILURE)
            state.valid[tendon] = 0
            return True
        state.tension[index] = force
        peak = wp.max(peak, force)
        total += extension
        all_nonpositive_damping = all_nonpositive_damping and damping <= wp.float64(0.0)
    force_error = wp.float64(0.0)
    for i in range(n - 1):
        left = state.tension[offset + i]
        right = state.tension[offset + i + 1]
        cap = wp.float64(state.cap[edge_offset + i])
        force_error = wp.max(force_error, wp.max(left - cap * right, right - cap * left))
    reference_is_feasible = bool(False)
    if force_error <= wp.float64(tolerance) * peak:
        for i in range(n):
            state.corrected[offset + i] = wp.float64(state.reference[offset + i])
            state.candidate[offset + i] = state.reference[offset + i]
        checked = _measure_candidate(
            state,
            n,
            offset,
            edge_offset,
            ea_low,
            ea_ratio,
            transition_strain,
            transition_width,
            min_rest,
            tolerance,
            True,
        )
        reference_is_feasible = checked[0] <= wp.float64(tolerance)
        peak = wp.float64(1e-20)
        force_error = checked[0]
        for i in range(n):
            peak = wp.max(peak, state.tension[offset + i])
        force_error *= peak
    if reference_is_feasible:
        for i in range(n):
            state.output[offset + i] = state.reference[offset + i]
            state.corrected[offset + i] = wp.float64(state.reference[offset + i])
            state.exact_tension[offset + i] = state.tension[offset + i]
        state.status[tendon] = int(TendonMaterialNonlinearStatus.CONVERGED)
        state.residual[tendon] = float(force_error / peak)
        state.publication_residual[tendon] = state.residual[tendon]
        state.storage_error[tendon] = wp.vec4d(wp.float64(0.0))
        state.valid[tendon] = 0
        return True
    if all_nonpositive_damping:
        slack_total = wp.float64(0.0)
        for i in range(n):
            index = offset + i
            length = wp.float64(state.length[index])
            target = -wp.float64(state.damping[index])
            endpoint = _inverse_extension(
                target,
                length,
                wp.float64(ea_low),
                wp.float64(ea_ratio),
                wp.float64(transition_strain),
                wp.float64(transition_width),
            )
            endpoint = wp.min(endpoint, length - wp.float64(min_rest))
            zero_force = bool(False)
            for _endpoint_iteration in range(8):
                rest = wp.max(length - endpoint, wp.float64(1e-08))
                strain = endpoint / rest
                transition = wp.tanh((strain - wp.float64(transition_strain)) / wp.float64(transition_width))
                ea = wp.float64(ea_low) * (
                    wp.float64(1.0)
                    + (wp.float64(ea_ratio) - wp.float64(1.0)) * wp.float64(0.5) * (wp.float64(1.0) + transition)
                )
                elastic = wp.max(endpoint / (rest / ea), wp.max(ea * strain, ea * endpoint / rest))
                error = elastic - target
                if wp.isfinite(error) and error <= wp.float64(0.0):
                    zero_force = True
                    break
                if not wp.isfinite(error):
                    break
                derivative = _elastic_derivative(
                    length,
                    endpoint,
                    wp.float64(ea_low),
                    wp.float64(ea_ratio),
                    wp.float64(transition_strain),
                    wp.float64(transition_width),
                )
                trial = endpoint - error / derivative
                endpoint = wp.max(wp.float64(0.0), _next_double_down(wp.min(endpoint, trial)))
            if not zero_force:
                state.status[tendon] = int(TendonMaterialNonlinearStatus.NUMERICAL_FAILURE)
                state.valid[tendon] = 0
                return True
            state.endpoint[index] = endpoint
            state.plateau[index] = 1
            slack_total += wp.float64(state.reference[index]) - endpoint
        if slack_total <= wp.float64(0.0):
            state.outer_iterations[tendon] = 1
            allocated = _allocate_plateau_block(
                state.reference,
                state.endpoint,
                state.plateau,
                state.cap,
                state.faces,
                state.corrected,
                offset,
                edge_offset,
                n,
                0,
            )
            state.valid[tendon] = 0
            if allocated <= 0:
                state.status[tendon] = int(TendonMaterialNonlinearStatus.NUMERICAL_FAILURE)
                return True
            for i in range(n):
                index = offset + i
                state.corrected[index] = wp.min(state.corrected[index], state.endpoint[index])
                state.candidate[index] = float(state.corrected[index])
            if not _certify_exact_publication(
                state,
                n,
                tendon,
                offset,
                edge_offset,
                ea_low,
                ea_ratio,
                transition_strain,
                transition_width,
                min_rest,
                tolerance,
            ):
                state.status[tendon] = int(TendonMaterialNonlinearStatus.NUMERICAL_FAILURE)
                return True
            for i in range(n):
                if state.exact_tension[offset + i] != wp.float64(0.0):
                    state.status[tendon] = int(TendonMaterialNonlinearStatus.NUMERICAL_FAILURE)
                    return True
            for i in range(n):
                state.output[offset + i] = state.candidate[offset + i]
            state.status[tendon] = int(TendonMaterialNonlinearStatus.CONVERGED)
            if state.publication_residual[tendon] > tolerance:
                state.status[tendon] = int(TendonMaterialNonlinearStatus.CONVERGED_ROUNDED)
            return True
    return False


@wp.func
def _elastic_derivative(
    length: wp.float64,
    extension: wp.float64,
    ea_low: wp.float64,
    ea_ratio: wp.float64,
    transition_strain: wp.float64,
    transition_width: wp.float64,
) -> wp.float64:
    rest = wp.max(length - extension, wp.float64(1.0e-8))
    strain = wp.max(extension, wp.float64(0.0)) / rest
    transition = wp.tanh((strain - transition_strain) / transition_width)
    delta = wp.float64(0.5) * ea_low * (ea_ratio - wp.float64(1.0))
    ea = ea_low + delta * (wp.float64(1.0) + transition)
    slope = delta * (wp.float64(1.0) - transition * transition) / transition_width
    if length - extension <= wp.float64(1.0e-8):
        return (ea + strain * slope) / wp.float64(1.0e-8)
    return length / (rest * rest) * (ea + strain * slope)


@wp.func_native("""
double l[32], b[32], w[32], e[32], floor[32], force[32], tangent[32], direction[32];
double capacity=0.0;
for (int i=0; i<n; ++i) {
    l[i]=length[offset+i]; b[i]=damping[offset+i]; w[i]=weights[offset+i];
    floor[i]=endpoints[offset+i];
    capacity += l[i]-floor[i];
}
if (capacity <= total-floor_sum) return wp::vec2d(-1.0, lower);
for (int i=0; i<n; ++i) e[i]=floor[i]+(total-floor_sum)*(l[i]-floor[i])/capacity;
auto evaluate = [&](int i, double x, bool accurate, double &f, double &k) {
    if (accurate) {
        double r=wp::max(l[i]-x,1.0e-8), s=wp::max(x,0.0)/r;
        double t=wp::tanh((s-knee)/width), d=0.5*ea_low*(ea_ratio-1.0);
        double ea=ea_low+d*(1.0+t), ds=d*(1.0-t*t)/width;
        f=ea*s;
        k=(l[i]-x<=1.0e-8 ? 1.0/1.0e-8 : l[i]/(r*r))*(ea+s*ds);
    } else {
        float ll=static_cast<float>(l[i]), xx=static_cast<float>(x), r=wp::max(ll-xx,1.0e-8f), s=wp::max(xx,0.0f)/r;
        float t=wp::tanh((s-static_cast<float>(knee))/static_cast<float>(width)), d=0.5f*static_cast<float>(ea_low)*(static_cast<float>(ea_ratio)-1.0f);
        float ea=static_cast<float>(ea_low)+d*(1.0f+t), ds=d*(1.0f-t*t)/static_cast<float>(width);
        f=static_cast<double>(ea*s);
        k=static_cast<double>((ll-xx<=1.0e-8f ? 1.0f/1.0e-8f : ll/(r*r))*(ea+s*ds));
    }
};
double base=0.5*(lower+upper);
bool accurate=false, converged=false;
for (int iteration=0; iteration<64; ++iteration) {
    if (iteration>=6) accurate=true;
    double mass=0.0,num=0.0,den=0.0,error=0.0,force_scale=1.0e-300;
    for (int i=0; i<n; ++i) {
        double elastic=0.0;
        evaluate(i,e[i],accurate,elastic,tangent[i]);
        force[i]=elastic+b[i]; mass+=e[i];
        num+=force[i]/tangent[i]; den+=w[i]/tangent[i];
        error=wp::max(error,wp::abs((force[i]-w[i]*base)/w[i]));
        force_scale=wp::max(force_scale,(wp::abs(elastic)+wp::abs(b[i]))/w[i]);
    }
    if (iteration==0) {
        base=wp::max(lower,(num+total-mass)/den);
        error=0.0;
        for (int i=0;i<n;++i) error=wp::max(error,wp::abs((force[i]-w[i]*base)/w[i]));
    }
    if (!accurate && error<=2.0e-5*force_scale) {accurate=true; continue;}
    if (accurate && error<=7.2e-15*force_scale && wp::abs(mass-total)<=32.0*2.220446049250313e-16*(scale+wp::abs(mass))) {
        converged=true; break;
    }
    double proposed=(num+total-mass)/den, step=1.0;
    if (proposed<lower) step=wp::min(step,0.99*(base-lower)/(base-proposed));
    for (int i=0; i<n; ++i) {
        direction[i]=(w[i]*proposed-force[i])/tangent[i];
        if (direction[i]<0.0) step=wp::min(step,0.99*(e[i]-floor[i])/(-direction[i]));
        else if (direction[i]>0.0) step=wp::min(step,0.99*(l[i]-e[i])/direction[i]);
    }
    bool accepted=false;
    for (int line=0; line<32; ++line) {
        double trial_base=base+step*(proposed-base), trial_error=0.0;
        for (int i=0; i<n; ++i) {
            double elastic=0.0,k=0.0;
            evaluate(i,e[i]+step*direction[i],accurate,elastic,k);
            trial_error=wp::max(trial_error,wp::abs((elastic+b[i])/w[i]-trial_base));
        }
        if (trial_error<=(1.0-0.0001*step)*error || trial_error<=7.2e-15*force_scale) {accepted=true; break;}
        step*=0.5;
    }
    if (!accepted) {
        if (!accurate) {accurate=true; continue;}
        return wp::vec2d(-1.0,base);
    }
    for (int i=0; i<n; ++i) e[i]+=step*direction[i];
    base+=step*(proposed-base);
}
if (!converged) return wp::vec2d(-1.0,base);
for (int i=0; i<n; ++i) endpoints[offset+i]=e[i];
return wp::vec2d(1.0,base);
""")
def _coupled_block_local(
    length: wp.array[float],
    damping: wp.array[float],
    weights: wp.array[wp.float64],
    endpoints: wp.array[wp.float64],
    offset: int,
    n: int,
    total: wp.float64,
    floor_sum: wp.float64,
    scale: wp.float64,
    lower: wp.float64,
    upper: wp.float64,
    ea_low: wp.float64,
    ea_ratio: wp.float64,
    knee: wp.float64,
    width: wp.float64,
) -> wp.vec2d:
    """Use local temporary storage while solving a fixed friction block."""
    ...


@wp.func
def _exact_block_targets(
    state: TendonMaterialNonlinearState,
    n: int,
    offset: int,
    edge_offset: int,
    ea_low: float,
    ea_ratio: float,
    transition_strain: float,
    transition_width: float,
    min_rest: float,
) -> int:
    """Minimize each fixed log-force block from the original material sum."""
    first = int(0)
    while first < n:
        last = first
        while last < n - 1 and state.faces[edge_offset + last] != 0:
            last += 1
        if first == last:
            # Conservation fixes an isolated span to its original reference.
            # Its force is direct; inverting it through a nested mass root is
            # redundant and can lose the exact sticking balance to roundoff.
            index = offset + first
            reference = wp.float64(state.reference[index])
            length = wp.float64(state.length[index])
            damping = wp.float64(state.damping[index])
            compliance = _secant_compliance(
                length,
                reference,
                wp.float64(ea_low),
                wp.float64(ea_ratio),
                wp.float64(transition_strain),
                wp.float64(transition_width),
            )
            force = wp.max(wp.max(reference, wp.float64(0.0)) / compliance + damping, wp.float64(0.0))
            state.tension[index] = wp.float64(1.0)
            state.plateau[index] = int(force == wp.max(damping, wp.float64(0.0)))
            endpoint = reference
            if state.plateau[index] != 0:
                endpoint = _inverse_extension(
                    wp.max(-damping, wp.float64(0.0)),
                    length,
                    wp.float64(ea_low),
                    wp.float64(ea_ratio),
                    wp.float64(transition_strain),
                    wp.float64(transition_width),
                )
                endpoint = wp.min(endpoint, length - wp.float64(min_rest))
            state.endpoint[index] = endpoint
            state.target_log_force[index] = wp.log(wp.max(force, wp.float64(1.0e-300)))
            first = last + 1
            continue
        weight = wp.float64(1.0)
        lower = wp.float64(0.0)
        upper = wp.float64(0.0)
        total = wp.float64(0.0)
        scale = wp.float64(1.0e-300)
        for i in range(first, last + 1):
            index = offset + i
            if i > first:
                cap = wp.float64(state.cap[edge_offset + i - 1])
                if state.faces[edge_offset + i - 1] > 0:
                    weight /= cap
                else:
                    weight *= cap
            state.tension[index] = weight
            damping = wp.float64(state.damping[index])
            lower = wp.max(lower, wp.max(damping, wp.float64(0.0)) / weight)
            reference = wp.float64(state.reference[index])
            total += reference
            scale += wp.abs(reference)
        base = lower
        upper_floor_sum = wp.float64(0.0)
        for i in range(first, last + 1):
            index = offset + i
            target = wp.max(base * state.tension[index] - wp.float64(state.damping[index]), wp.float64(0.0))
            # EA(strain) >= EA_low gives an upper bound on the zero/floor
            # extension. Include the constitutive rest clamp in that bound.
            bound = wp.max(
                wp.float64(state.length[index]) * (target / (wp.float64(ea_low) + target)),
                wp.float64(1.0e-8) * (target / wp.float64(ea_low)),
            )
            upper_floor_sum += bound
        provably_taut = total > upper_floor_sum + wp.float64(128.0 * 2.220446049250313e-16) * (scale + upper_floor_sum)
        floor_sum = wp.float64(0.0)
        for i in range(first, last + 1):
            index = offset + i
            endpoint = wp.float64(0.0)
            if not provably_taut:
                endpoint = _inverse_extension(
                    wp.max(base * state.tension[index] - wp.float64(state.damping[index]), wp.float64(0.0)),
                    wp.float64(state.length[index]),
                    wp.float64(ea_low),
                    wp.float64(ea_ratio),
                    wp.float64(transition_strain),
                    wp.float64(transition_width),
                )
            state.endpoint[index] = endpoint
            state.target_log_force[index] = endpoint
            floor_sum += endpoint
        evaluated_base = base
        if total > floor_sum:
            local_result = _coupled_block_local(
                state.length,
                state.damping,
                state.tension,
                state.endpoint,
                offset + first,
                last - first + 1,
                total,
                floor_sum,
                scale,
                lower,
                upper,
                wp.float64(ea_low),
                wp.float64(ea_ratio),
                wp.float64(transition_strain),
                wp.float64(transition_width),
            )
            if local_result[0] < wp.float64(0.0):
                return -1
            base = local_result[1]
            evaluated_base = base
        for i in range(first, last + 1):
            index = offset + i
            force = base * state.tension[index]
            floor = wp.max(wp.float64(state.damping[index]), wp.float64(0.0))
            roundoff = wp.float64(64.0 * 2.220446049250313e-16) * wp.max(force, floor)
            state.plateau[index] = int(wp.abs(force - floor) <= roundoff)
            # Reuse only endpoints evaluated at this exact base. A capped mass
            # loop can leave its final proposal unevaluated, requiring inversion.
            endpoint = state.endpoint[index]
            if base != evaluated_base:
                endpoint = _inverse_extension(
                    wp.max(force - wp.float64(state.damping[index]), wp.float64(0.0)),
                    wp.float64(state.length[index]),
                    wp.float64(ea_low),
                    wp.float64(ea_ratio),
                    wp.float64(transition_strain),
                    wp.float64(transition_width),
                )
            upper_extension = wp.float64(state.length[index]) - wp.float64(min_rest)
            if state.plateau[index] != 0:
                endpoint = wp.min(endpoint, upper_extension)
            elif endpoint > upper_extension:
                return -1
            if not wp.isfinite(endpoint) or not wp.isfinite(force):
                return -1
            state.endpoint[index] = endpoint
            # A zero-force block can only coexist with another zero block in
            # the feasible finite-friction solution. This finite log target
            # reaches an intervening capstan face before zero is approached.
            state.target_log_force[index] = wp.log(wp.max(force, wp.float64(1.0e-300)))
        first = last + 1
    return 1


@wp.func
def _allocate_exact_faces(state: TendonMaterialNonlinearState, n: int, offset: int, edge_offset: int) -> int:
    """Allocate stationary blocks; return a wrong-sign edge to release plus one."""
    first = int(0)
    release = int(-1)
    worst = wp.float64(0.0)
    while first < n:
        last = first
        while last < n - 1 and state.faces[edge_offset + last] != 0:
            last += 1
        allocated = _allocate_plateau_block(
            state.reference,
            state.endpoint,
            state.plateau,
            state.cap,
            state.faces,
            state.corrected,
            offset + first,
            edge_offset + first,
            last - first + 1,
            1,
        )
        if allocated <= 0:
            allocated = _allocate_plateau_block(
                state.reference,
                state.endpoint,
                state.plateau,
                state.cap,
                state.faces,
                state.corrected,
                offset + first,
                edge_offset + first,
                last - first + 1,
                0,
            )
            if allocated <= 0:
                return -1
            flow = wp.float64(0.0)
            for i in range(first, last):
                flow += wp.float64(state.reference[offset + i]) - state.corrected[offset + i]
                signed = flow * wp.float64(state.faces[edge_offset + i])
                if state.cap[edge_offset + i] != 1.0 and signed < worst:
                    worst = signed
                    release = i
        first = last + 1
    return release + 1


@wp.func
def _certify_exact_publication(
    state: TendonMaterialNonlinearState,
    n: int,
    tendon: int,
    offset: int,
    edge_offset: int,
    ea_low: float,
    ea_ratio: float,
    transition_strain: float,
    transition_width: float,
    min_rest: float,
    tolerance: float,
) -> bool:
    """Certify the double allocation before checking prescribed float storage.

    No storage allowance enters the exact force, flow, mass, or rest checks.
    Publication must be exactly float32(corrected), finite, and rest-feasible.
    For delta_e=published-exact and eta=T(published)-T(exact), the measured
    storage_error contains maximum |eta| [N], maximum internal prefix |sum
    delta_e| [m], |sum delta_e| [m], and sum |delta_e| [m]. The last value bounds
    every prefix-flow error and the mass error; a capstan gap error is bounded
    by |eta_left|+cap*|eta_right| (or the reversed expression). These are
    numerical forward differences under the same double force evaluation,
    not formally outward-rounded intervals for the transcendental law.
    """
    state.publication_residual[tendon] = 1.0e30
    state.storage_error[tendon] = wp.vec4d(wp.float64(0.0))
    exact_measurement = _measure_candidate(
        state,
        n,
        offset,
        edge_offset,
        ea_low,
        ea_ratio,
        transition_strain,
        transition_width,
        min_rest,
        tolerance,
        True,
    )
    state.residual[tendon] = float(exact_measurement[0])
    if not wp.isfinite(exact_measurement[0]) or exact_measurement[0] > wp.float64(tolerance):
        return False
    for i in range(n):
        index = offset + i
        state.exact_tension[index] = state.tension[index]
        published = state.candidate[index]
        if (
            not wp.isfinite(published)
            or published != float(state.corrected[index])
            or wp.float64(state.length[index]) - wp.float64(published) < wp.float64(min_rest)
        ):
            return False
    published_measurement = _measure_candidate(
        state,
        n,
        offset,
        edge_offset,
        ea_low,
        ea_ratio,
        transition_strain,
        transition_width,
        min_rest,
        tolerance,
    )
    if not wp.isfinite(published_measurement[0]) or published_measurement[0] >= wp.float64(1.0e30):
        return False
    state.publication_residual[tendon] = float(published_measurement[0])
    force_error = wp.float64(0.0)
    prefix_error = wp.float64(0.0)
    flow_error = wp.float64(0.0)
    bound = wp.float64(0.0)
    for i in range(n):
        index = offset + i
        delta = wp.float64(state.candidate[index]) - state.corrected[index]
        prefix_error += delta
        bound += wp.abs(delta)
        force_error = wp.max(force_error, wp.abs(state.tension[index] - state.exact_tension[index]))
        if i < n - 1:
            flow_error = wp.max(flow_error, wp.abs(prefix_error))
    state.storage_error[tendon] = wp.vec4d(force_error, flow_error, wp.abs(prefix_error), bound)
    return True


@wp.func
def _finish_exact_faces(
    state: TendonMaterialNonlinearState,
    n: int,
    tendon: int,
    offset: int,
    edge_offset: int,
    ea_low: float,
    ea_ratio: float,
    transition_strain: float,
    transition_width: float,
    min_rest: float,
    tolerance: float,
) -> int:
    """Return zero only for a certified candidate; positive means release edge+1."""
    # Include weak equalities before allocating minimum net transfer. A cached
    # sticking split cannot choose a larger transfer than the cold solution.
    for i in range(n - 1):
        if state.faces[edge_offset + i] == 0:
            difference = state.target_log_force[offset + i] - state.target_log_force[offset + i + 1]
            bound = wp.log(wp.float64(state.cap[edge_offset + i]))
            roundoff = wp.float64(128.0 * 2.220446049250313e-16) * (wp.float64(1.0) + wp.abs(difference))
            if wp.abs(difference - bound) <= roundoff:
                state.faces[edge_offset + i] = 1
            elif wp.abs(difference + bound) <= roundoff:
                state.faces[edge_offset + i] = -1
    release = _allocate_exact_faces(state, n, offset, edge_offset)
    if release != 0:
        return release
    for i in range(n):
        state.candidate[offset + i] = float(state.corrected[offset + i])
    if _certify_exact_publication(
        state,
        n,
        tendon,
        offset,
        edge_offset,
        ea_low,
        ea_ratio,
        transition_strain,
        transition_width,
        min_rest,
        tolerance,
    ):
        return 0
    return -1


@wp.func
def _solve_log_force_faces(
    state: TendonMaterialNonlinearState,
    n: int,
    allow_warm: int,
    tendon: int,
    span_stride: int,
    edge_stride: int,
    ea_low: float,
    ea_ratio: float,
    transition_strain: float,
    transition_width: float,
    min_rest: float,
    max_iterations: int,
    tolerance: float,
):
    """Feasible primal active-set solve of the convex log-force formulation.

    Fixed faces define blocks T_i=w_i*t. Each block minimizes its convex
    potential by conserving original material through the inverse true force
    law. A step toward these minimizers stops at the first inactive capstan
    constraint. Only at a block minimum may a wrong-sign flow release a face.
    This avoids infeasible full-step merge/split cycling. Flat-force inverse
    intervals are allocated by the exact minimum-net-transfer PAVA.
    """
    offset = tendon * span_stride
    edge_offset = tendon * edge_stride
    peak = wp.float64(1.0e-300)
    for i in range(n):
        peak = wp.max(peak, state.tension[offset + i])
    attempts = int(0)
    state.status[tendon] = int(TendonMaterialNonlinearStatus.MAX_ITERATIONS)
    cache_valid = allow_warm != 0 and state.valid[tendon] != 0
    for i in range(n - 1):
        cache_valid = cache_valid and wp.abs(state.faces[edge_offset + i]) <= 1
    if cache_valid:
        attempts = 1
        state.outer_iterations[tendon] = attempts
        cached = _exact_block_targets(
            state,
            n,
            offset,
            edge_offset,
            ea_low,
            ea_ratio,
            transition_strain,
            transition_width,
            min_rest,
        )
        if cached > 0:
            cached = _finish_exact_faces(
                state,
                n,
                tendon,
                offset,
                edge_offset,
                ea_low,
                ea_ratio,
                transition_strain,
                transition_width,
                min_rest,
                tolerance,
            )
            if cached == 0:
                for i in range(n):
                    state.output[offset + i] = state.candidate[offset + i]
                state.status[tendon] = int(TendonMaterialNonlinearStatus.CONVERGED)
                if state.publication_residual[tendon] > tolerance:
                    state.status[tendon] = int(TendonMaterialNonlinearStatus.CONVERGED_ROUNDED)
                state.valid[tendon] = 1
                return
        # The cache is only a face guess. An unsuccessful trial neither
        # publishes an allocation nor latches a failure; restart feasible search
        # from the original reference's peak, saved before scratch was changed.
    for i in range(n):
        state.log_force[offset + i] = wp.log(peak)
    for i in range(n - 1):
        state.faces[edge_offset + i] = 0
    state.valid[tendon] = 0
    for iteration in range(max_iterations - attempts):
        state.outer_iterations[tendon] = attempts + iteration + 1
        result = _exact_block_targets(
            state,
            n,
            offset,
            edge_offset,
            ea_low,
            ea_ratio,
            transition_strain,
            transition_width,
            min_rest,
        )
        if result < 0:
            state.status[tendon] = int(TendonMaterialNonlinearStatus.BOUND_INCOMPATIBLE)
            return
        step = wp.float64(1.0)
        blocking_edge = int(-1)
        blocking_face = int(0)
        for i in range(n - 1):
            if state.faces[edge_offset + i] == 0:
                current = state.log_force[offset + i] - state.log_force[offset + i + 1]
                target = state.target_log_force[offset + i] - state.target_log_force[offset + i + 1]
                bound = wp.log(wp.float64(state.cap[edge_offset + i]))
                trial_step = wp.float64(1.0)
                face = int(0)
                if target > bound:
                    trial_step = wp.max(wp.float64(0.0), (bound - current) / (target - current))
                    face = 1
                elif target < -bound:
                    trial_step = wp.max(wp.float64(0.0), (-bound - current) / (target - current))
                    face = -1
                if trial_step < step:
                    step = trial_step
                    blocking_edge = i
                    blocking_face = face
        for i in range(n):
            index = offset + i
            state.log_force[index] += step * (state.target_log_force[index] - state.log_force[index])
        if blocking_edge >= 0:
            state.faces[edge_offset + blocking_edge] = blocking_face
        else:
            release = _finish_exact_faces(
                state,
                n,
                tendon,
                offset,
                edge_offset,
                ea_low,
                ea_ratio,
                transition_strain,
                transition_width,
                min_rest,
                tolerance,
            )
            if release < 0:
                state.status[tendon] = int(TendonMaterialNonlinearStatus.NUMERICAL_FAILURE)
                return
            if release > 0:
                state.faces[edge_offset + release - 1] = 0
            else:
                for i in range(n):
                    state.output[offset + i] = state.candidate[offset + i]
                state.status[tendon] = int(TendonMaterialNonlinearStatus.CONVERGED)
                if state.publication_residual[tendon] > tolerance:
                    state.status[tendon] = int(TendonMaterialNonlinearStatus.CONVERGED_ROUNDED)
                state.valid[tendon] = 1
                return


@wp.func
def solve_tendon_material_nonlinear_component(
    state: TendonMaterialNonlinearState,
    n: int,
    allow_warm: int,
    tendon: int,
    span_stride: int,
    edge_stride: int,
    ea_low: float,
    ea_ratio: float,
    transition_strain: float,
    transition_width: float,
    min_rest: float,
    max_iterations: int,
    tolerance: float,
):
    """Project one component using a bounded experimental nonlinear method.

    Args:
        state: Packed arrays. Populate reference, length, damping, and cap.
            Reference is the full-rolling signed extension before any slip [m].
            Damping is a frozen additive tension [N]. Output is the accepted
            absolute signed extension [m], never a damping-shifted extension.
            Candidate and other scratch arrays must not alias output or inputs.
        n: Number of spans, between one and 32.
        allow_warm: Whether to try cached capstan active faces.
        tendon: Component status/cache index and base offset multiplier.
        span_stride: Span offset multiplier; use one for packed components.
        edge_stride: Edge offset multiplier; use one for packed components.
        ea_low: Positive low-strain axial stiffness [N].
        ea_ratio: High-to-low stiffness ratio, at least one.
        transition_strain: Center strain of the sigmoid stiffness transition.
        transition_width: Positive sigmoid transition width in strain.
        min_rest: Minimum permitted rest length [m].
        max_iterations: Maximum exact-block active-set
            iterations, between one and 64. Scalar roots have separate bounds.
        tolerance: Relative true-law force and material-conservation tolerance.
            Applies to corrected double-precision extensions, not
            to the independently reported float32 publication residual.

    Cache faces are only guesses and are validated by the inner solve after
    coefficient changes. Invalidate valid when component route identities change.
    The solve preserves a feasible reference and handles zero tension. The
    original reference still defines every trial and its net-transfer direction.
    Negative status leaves output unchanged and invalidates the scratch cache.
    The caller decides how to report a terminal failure; no failure is latched.

    Corrected and exact_tension describe the strictly certified
    double allocation on every successful return. Output is its prescribed
    float32 rounding and still satisfies the hard rest bound. Residual reports
    the exact solve; publication_residual reports the raw stored-state residual.
    Storage_error reports (max force difference [N], max internal prefix-flow
    difference [m], total mass difference [m], sum absolute extension rounding
    errors [m]). CONVERGED_ROUNDED qualifies success when the raw publication
    residual exceeds tolerance; it does not claim strict float32 equilibrium.
    """
    offset = tendon * span_stride
    edge_offset = tendon * edge_stride
    state.status[tendon] = int(TendonMaterialNonlinearStatus.BAD_INPUT)
    state.inner_status[tendon] = 0
    state.outer_iterations[tendon] = 0
    state.residual[tendon] = 1e30
    state.publication_residual[tendon] = 1e30
    state.storage_error[tendon] = wp.vec4d(wp.float64(0.0))
    if (
        n < 1
        or n > 32
        or max_iterations < 1
        or (max_iterations > 64)
        or (not wp.isfinite(ea_low))
        or (ea_low <= 0.0)
        or (not wp.isfinite(ea_ratio))
        or (ea_ratio < 1.0)
        or (not wp.isfinite(transition_strain))
        or (not wp.isfinite(transition_width))
        or (transition_width <= 0.0)
        or (not wp.isfinite(min_rest))
        or (min_rest < 0.0)
        or (not wp.isfinite(tolerance))
        or (tolerance <= 0.0)
    ):
        state.valid[tendon] = 0
        return
    # Full rolling can put the no-slip trial beyond an individual rest bound;
    # slip must be allowed to repair it. Certify the projected rest lengths,
    # not the trial. A short geometric span can also contain valid slack.
    for i in range(n):
        index = offset + i
        reference = state.reference[index]
        length = state.length[index]
        damping = state.damping[index]
        if not wp.isfinite(reference) or not wp.isfinite(length) or length < 0.0 or (not wp.isfinite(damping)):
            state.valid[tendon] = 0
            return
        if i < n - 1:
            cap = state.cap[edge_offset + i]
            if not wp.isfinite(cap) or cap < 1.0 or cap > 4.0:
                state.valid[tendon] = 0
                return
    if _initialize_component(
        state,
        n,
        tendon,
        span_stride,
        edge_stride,
        ea_low,
        ea_ratio,
        transition_strain,
        transition_width,
        min_rest,
        tolerance,
    ):
        return
    _solve_log_force_faces(
        state,
        n,
        allow_warm,
        tendon,
        span_stride,
        edge_stride,
        ea_low,
        ea_ratio,
        transition_strain,
        transition_width,
        min_rest,
        max_iterations,
        tolerance,
    )
    return
