# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference checks for damping, nonlinear material and changing ALM routes."""

import unittest
from itertools import product

import numpy as np
import warp as wp

import newton
from newton._src.solvers.tendon_alm_material import (
    TendonALMMaterialState,
    alm_material_coefficients,
    alm_material_linearization,
    project_alm_material,
)
from newton._src.solvers.tendon_kernels import tendon_material_tangent
from newton._src.solvers.vbd.tendon_kernels import (
    _tendon_alm_body_stiffness,
    _tendon_alm_row,
    _tendon_secant_stiffness,
)
from newton.tests.test_tendon_capstan import _oriented_route_material_length, build_force_driven_dynamic_route
from newton.tests.test_tendon_material_integration import _chain_model
from newton.tests.test_tendon_material_vbd import _body_force, _simulate_pulley
from newton.tests.test_tendon_segment_alm import _dual_step, _fixture
from newton.tests.test_tendon_segment_alm import test_graph_matches_eager as _graph_matches_eager
from newton.tests.unittest_utils import add_function_test, get_test_devices

SIGMOID = {
    "tendon_sigmoid_ea_low": 500.0,
    "tendon_sigmoid_ea_ratio": 10.0,
    "tendon_sigmoid_transition_strain": 0.01,
    "tendon_sigmoid_transition_width": 0.003,
}


@wp.kernel(enable_backward=False)
def _body_tangent_probe(values: wp.array[float], forces: wp.array[float], tangents: wp.array[float], rho: float):
    i = wp.tid()
    length = values[i]
    rest = 0.04
    extension = length - rest
    rate = (length - 0.0403) / (1.0 / 1200.0)
    k_sec = _tendon_secant_stiffness(length, rest, 5.0e-6, 500.0, 10.0, 0.01, 0.003)
    k_tangent = tendon_material_tangent(length, rest, 5.0e-6, 500.0, 10.0, 0.01, 0.003)
    force, _ = _tendon_alm_row(k_sec, 5.0, 1.0 / 1200.0, extension, rate, rho, 4.0, 2)
    forces[i] = force
    tangents[i] = _tendon_alm_body_stiffness(k_sec, k_tangent, 5.0, 1.0 / 1200.0, extension, rho, 4.0, force)


def test_nonlinear_body_tangent(test, device):
    """Differentiate body motion at fixed rest length, not material motion at fixed geometry."""
    for rho in (0.0, 5000.0, 1.0e6):
        centers = np.array([0.04035, 0.0404, 0.0405], np.float32)
        x = np.stack([centers - 2e-7, centers, centers + 2e-7], axis=1).ravel()
        force, tangent = [wp.zeros(len(x), device=device) for _ in range(2)]
        wp.launch(_body_tangent_probe, len(x), inputs=[wp.array(x, device=device), force, tangent, rho], device=device)
        f = force.numpy().reshape(-1, 3).astype(float)
        h = tangent.numpy().reshape(-1, 3)[:, 1]
        xx = x.reshape(-1, 3).astype(float)
        numerical = (f[:, 2] - f[:, 0]) / (xx[:, 2] - xx[:, 0])
        np.testing.assert_allclose(h, numerical, rtol=5e-3, atol=0.1)


def test_slack_cancellation_has_exact_zero_force(test, device):
    """Offset cancellation must not resurrect a certified slack force as a tiny dual."""
    for lam in np.geomspace(1e-8, 1e-15, 32):
        state, _, status = _frozen_projection(
            device,
            [0.0] * 4,
            [0.02, 0.015, 0.03, 0.01],
            [0.0, 0.0, lam, 0.0],
            [-1e-5, 0.0, 0.0, 0.0],
            [1.1] * 4,
        )
        test.assertGreaterEqual(status, 0)
        np.testing.assert_array_equal(state.force.numpy(), 0.0)


@wp.kernel(enable_backward=False)
def _project_frozen(
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
    count: int,
):
    project_alm_material(
        state,
        ids,
        initial,
        compliance,
        cap,
        upper,
        output,
        status,
        faces,
        valid,
        pieces,
        0,
        count,
        500.0,
        10.0,
        0.01,
        0.003,
    )


def _frozen_projection(device, reference, length, lam, damping_force, caps, rho=9406.884):
    n = len(reference)
    state = TendonALMMaterialState()
    state.dt = 1 / 1200
    for name in ("extension", "offset", "force"):
        setattr(state, name, wp.zeros(n, device=device))
    state.iterations = wp.zeros(n, dtype=int, device=device)
    state.reference = wp.array(reference, dtype=float, device=device)
    state.length = wp.array(length, dtype=float, device=device)
    state.physical_compliance = wp.full(n, 5e-6, device=device)
    state.damping = wp.full(n, 5.0, device=device)
    state.damping_force = wp.array(damping_force, dtype=float, device=device)
    state.penalty = wp.array(np.broadcast_to(rho, (n,)).copy(), dtype=float, device=device)
    state.multiplier = wp.array(lam, dtype=float, device=device)
    initial, compliance, upper, output = [wp.zeros(n, device=device) for _ in range(4)]
    status, faces, valid, pieces = [wp.zeros(n, dtype=int, device=device) for _ in range(4)]
    wp.launch(
        _project_frozen,
        dim=1,
        inputs=[
            state,
            wp.array(np.arange(n), dtype=int, device=device),
            initial,
            compliance,
            wp.array(caps, dtype=float, device=device),
            upper,
            output,
            status,
            faces,
            valid,
            pieces,
            n,
        ],
        device=device,
    )
    return state, output.numpy().astype(float), int(status.numpy()[0])


def test_sigmoid_knee_projection(test, device):
    """Project stored duals across a steep sigmoid knee without secant cycling."""
    # Frozen Toy4-like state: nine short pinhole spans followed by frictional rollers.
    reference = np.array(
        [-0.00017587] + [8.02e-6] * 9 + [0.000158527, 0.000129347, 0.000616001, 0.000184701, 0.000077494], np.float32
    )
    length = np.array(
        [0.01753155] + [0.000644] * 9 + [0.01997498, 0.01585055, 0.01946313, 0.02987846, 0.01069240], np.float32
    )
    lam = np.array([0] + [15.899] * 9 + [13.3283, 15.1183, 40.064, 19.0157, 10.4593], np.float32)
    damping_force = np.array([-1.94579] + [0] * 11 + [1.90892, 0.291096, 0.340378], np.float32)
    caps = np.array([1] * 10 + [1.0965482, 1.1968346, 1.0948426, 1.5735564, 1], np.float32)
    state, e, status = _frozen_projection(device, reference, length, lam, damping_force, caps)
    test.assertGreaterEqual(status, 0)
    rest = length.astype(float) - e
    strain = np.maximum(e, 0) / rest
    physical_c = rest / (500 * (1 + 4.5 * (1 + np.tanh((strain - 0.01) / 0.003))))
    metric = 1 + physical_c * 5 / state.dt
    damping = np.minimum(damping_force, 5 * np.maximum(e, 0) / state.dt)
    force = np.maximum((e + physical_c * damping + metric * lam / 9406.884) / (physical_c + metric / 9406.884), 0)
    np.testing.assert_allclose(state.force.numpy(), force, atol=1e-4, rtol=2e-5)
    test.assertAlmostEqual(float(e.sum()), float(reference.astype(float).sum()), delta=2e-8)
    test.assertTrue(np.all(rest >= 1e-6))
    flow = np.cumsum(reference - e)[:-1]
    tol = 2e-5 * max(force)
    test.assertTrue(np.all(force[:-1] <= caps[:-1] * force[1:] + tol))
    test.assertTrue(np.all(force[1:] <= caps[:-1] * force[:-1] + tol))
    for i in np.flatnonzero(abs(flow) > 1e-7):
        left, right = (force[i], caps[i] * force[i + 1]) if flow[i] > 0 else (force[i + 1], caps[i] * force[i])
        test.assertAlmostEqual(left, right, delta=tol)


def test_short_movable_span_metric(test, device):
    """A 104 um span must not inherit an artificial 4 N ceiling from its ALM metric."""
    reference = [0.00048289448] + [1.891e-6] * 9 + [0.0001382893, -0.0000190205, 0.0001055974, 0.0000600813]
    length = [0.05943925] + [0.00024613] * 9 + [0.018, 0.00010401336, 0.016977936, 0.01137444]
    lam = [12.780825] + [0.0] * 9 + [10.198462, 0.0, 5.3336334, 3.8903127]
    rho = [6946.406] + [0.0] * 9 + [73756.44, 37817.945, 49979.39, 41820.92]
    damping = [0.18543] + [0.0] * 9 + [-0.0015036, -0.1285062, 0.05305, 0.2576957]
    caps = np.array([1.0] * 10 + [1.2870318, 1.8418641, 1.3448842, 1.0])
    state, e, status = _frozen_projection(device, reference, length, lam, damping, caps, rho=rho)
    test.assertGreaterEqual(status, 0)
    test.assertGreater(state.penalty.numpy()[11], rho[11])
    test.assertTrue(np.all(np.array(length) - e >= 0.999e-6))
    test.assertAlmostEqual(sum(e), sum(reference), delta=2e-8)
    force = state.force.numpy().astype(float)
    rest = np.array(length) - e
    strain = np.maximum(e, 0.0) / rest
    physical_c = rest / (500.0 * (1.0 + 4.5 * (1.0 + np.tanh((strain - 0.01) / 0.003))))
    metric = 1.0 + physical_c * 5.0 / state.dt
    damping_force = np.minimum(damping, 5.0 * np.maximum(e, 0.0) / state.dt)
    offset = physical_c * damping_force
    compliance = physical_c.copy()
    new_rho = state.penalty.numpy().astype(float)
    movable = new_rho > 0.0
    offset[movable] += metric[movable] * np.array(lam)[movable] / new_rho[movable]
    compliance[movable] += metric[movable] / new_rho[movable]
    np.testing.assert_allclose(force, np.maximum((e + offset) / compliance, 0.0), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(force[:11], force[0], rtol=5e-5)
    test.assertTrue(np.all(force[:-1] <= caps[:-1] * force[1:] + 1e-3))
    test.assertTrue(np.all(force[1:] <= caps[:-1] * force[:-1] + 1e-3))


def test_cancelling_damping_projection(test, device):
    """Resolve small net tensions when elasticity and damping nearly cancel."""
    reference = np.array([0.000415076] + [0.0] * 9 + [0.0, -0.000274644, 0.0, 0.0000617973], np.float32)
    length = np.array([0.04346313] + [0.00024613] * 9 + [0.018, 0.0275703, 0.01697792, 0.01153069], np.float32)
    damping_force = np.array([2.490476] + [0.0] * 10 + [-7.802517, 0.2162303, 1.481095], np.float32)
    caps = np.array([1.0] * 10 + [1.1389692, 1.2307099, 1.3421429, 1.0], np.float32)
    state, e, status = _frozen_projection(
        device, reference, length, np.zeros(14, np.float32), damping_force, caps, rho=0.0
    )
    test.assertGreaterEqual(status, 0)
    # Independent scalar inversion of the physical force on every span. For
    # this sliding route the first eleven forces agree, followed by +,-,- faces.
    weights = np.ones(14)
    weights[11] = 1 / caps[10]
    weights[12] = weights[11] * caps[11]
    weights[13] = weights[12] * caps[12]

    def inverse_force(base):
        lo = np.full(14, -0.1)
        hi = length.astype(float) - 1e-6
        for _ in range(60):
            mid = (lo + hi) / 2
            rest = length - mid
            strain = np.maximum(mid, 0) / rest
            elastic = 500 * (1 + 4.5 * (1 + np.tanh((strain - 0.01) / 0.003))) * strain
            f = elastic + np.minimum(damping_force, 5 * np.maximum(mid, 0) / state.dt)
            above = f > base * weights
            hi = np.where(above, mid, hi)
            lo = np.where(above, lo, mid)
        return (lo + hi) / 2

    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        target = inverse_force(mid)
        if target.sum() > reference.astype(float).sum():
            hi = mid
        else:
            lo = mid
    np.testing.assert_allclose(e, target, atol=2e-9, rtol=2e-5)
    np.testing.assert_allclose(state.force.numpy(), mid * weights, atol=2e-5, rtol=2e-5)
    test.assertAlmostEqual(float(e.sum()), float(reference.astype(float).sum()), delta=2e-8)


@wp.kernel
def _engagement_rows(
    state: TendonALMMaterialState,
    extension: wp.array[float],
    rate: wp.array[float],
    result: wp.array[wp.vec3],
):
    i = wp.tid()
    e = extension[i]
    c, offset = alm_material_coefficients(state, i, e, 0.0, 1.0, 0.01, 0.003)
    body, _ = _tendon_alm_row(5000.0, 100.0, state.dt, e, rate[i], state.penalty[i], state.multiplier[i], 2)
    shared, _ = _tendon_alm_row(5000.0, 100.0, state.dt, e, rate[i], state.penalty[i], state.multiplier[i], 1)
    result[i] = wp.vec3(wp.max(body, 0.0), wp.max((e + offset) / c, 0.0), wp.max(shared, 0.0))


def test_continuous_damping_rows(test, device):
    """Match material/body forces through engagement while preserving the shared-ALM law."""
    samples = np.asarray(
        list(
            product(
                (-1.0e-3, -1.0e-9, 0.0, 1.0e-9, 1.0e-5, 2.0e-5, 1.0e-3), (-0.02, 0.0, 0.02), (0.0, 1.0e4), (0.0, 2.0)
            )
        ),
        dtype=np.float32,
    )
    samples = samples[(samples[:, 2] > 0) | (samples[:, 3] == 0)]
    e, rate, rho, lam = samples.T.astype(float)
    n = len(samples)
    state = TendonALMMaterialState()
    state.dt = 0.001
    state.length = wp.full(n, 0.1, device=device)
    state.physical_compliance = wp.full(n, 1 / 5000, device=device)
    state.damping = wp.full(n, 100.0, device=device)
    state.damping_force = wp.array((100 * rate).astype(np.float32), device=device)
    state.penalty = wp.array(rho.astype(np.float32), device=device)
    state.multiplier = wp.array(lam.astype(np.float32), device=device)
    result = wp.zeros(n, dtype=wp.vec3, device=device)
    wp.launch(
        _engagement_rows,
        dim=n,
        inputs=[
            state,
            wp.array(e.astype(np.float32), device=device),
            wp.array(rate.astype(np.float32), device=device),
            result,
        ],
        device=device,
    )
    effective_rate = np.minimum(rate, np.maximum(e, 0) / state.dt)
    row_k = 5000 + 100 / state.dt
    gain = np.divide(row_k, rho, out=np.zeros_like(rho), where=rho > 0)
    expected = np.maximum((5000 * e + 100 * effective_rate + gain * lam) / (1 + gain), 0)
    shared = np.maximum((5000 * e + 100 * np.where(e > 0, rate, 0) + gain * lam) / (1 + gain), 0)
    actual = result.numpy()
    np.testing.assert_allclose(actual[:, 0], expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(actual[:, 1], expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(actual[:, 2], shared, atol=2e-6, rtol=2e-6)
    test.assertTrue(np.all(effective_rate * rate >= 0), "Damping must not inject energy")


@wp.kernel
def _linearization_values(state: TendonALMMaterialState, extension: wp.array[float], result: wp.array[wp.vec2]):
    i = wp.tid()
    e = extension[i]
    c, offset = alm_material_linearization(state, i, e, 500.0, 10.0, 0.01, 0.003)
    result[i] = wp.vec2((e + offset) / c, 1.0 / c)


def test_nonlinear_force_linearization(test, device):
    """Check the Newton tangent against independent double-precision finite differences."""
    rng = np.random.default_rng(20260922)
    n = 128
    state = TendonALMMaterialState()
    state.dt = 1 / 1200
    length = rng.uniform(0.001, 0.2, n).astype(np.float32)
    strain = rng.uniform(-0.01, 0.03, n)
    e = (length * strain / (1 + strain)).astype(np.float32)
    rho = rng.choice([0.0, 1e3, 1e4, 1e5], n).astype(np.float32)
    lam = np.where(rho > 0, rng.uniform(0, 40, n), 0).astype(np.float32)
    damping = rng.choice([0.0, 5.0, 100.0], n).astype(np.float32)
    viscous = (damping * rng.uniform(-0.2, 0.2, n)).astype(np.float32)
    for key, value in (
        ("length", length),
        ("penalty", rho),
        ("multiplier", lam),
        ("damping", damping),
        ("damping_force", viscous),
    ):
        setattr(state, key, wp.array(value, device=device))
    state.physical_compliance = wp.full(n, 1e-6, device=device)
    result = wp.zeros(n, dtype=wp.vec2, device=device)
    wp.launch(_linearization_values, dim=n, inputs=[state, wp.array(e, device=device), result], device=device)

    def force(extension):
        rest = length.astype(float) - extension
        strain = np.maximum(extension, 0) / rest
        c = rest / (500 * (1 + 4.5 * (1 + np.tanh((strain - 0.01) / 0.003))))
        metric = 1 + c * damping / state.dt
        b = np.minimum(viscous, damping * np.maximum(extension, 0) / state.dt)
        invrho = np.divide(1.0, rho, out=np.zeros(n), where=rho > 0)
        return (extension + c * b + metric * lam * invrho) / (c + metric * invrho)

    e = e.astype(float)
    h = length.astype(float) * 1e-7
    derivative = (force(e + h) - force(e - h)) / (2 * h)
    actual = result.numpy()
    np.testing.assert_allclose(actual[:, 0], force(e), atol=3e-4, rtol=1e-4)
    test.assertTrue(np.all(derivative > 0))
    np.testing.assert_allclose(actual[:, 1], derivative, atol=0.01, rtol=3e-4)


def _elastic(length, rest, compliance, nonlinear):
    extension = np.maximum(length - rest, 0.0)
    if not nonlinear:
        return extension / compliance
    strain = extension / rest
    return 500.0 * (1.0 + 4.5 * (1.0 + np.tanh((strain - 0.01) / 0.003))) * strain


def _set_rest(solver, lengths, target, nonlinear):
    compliance = solver.tendon_seg_active_compliance.numpy().astype(float)
    low, high = 0.5 * lengths, lengths.copy()
    for _ in range(60):
        mid = 0.5 * (low + high)
        tension = _elastic(lengths, mid, compliance, nonlinear)
        low = np.where(tension > target, mid, low)
        high = np.where(tension > target, high, mid)
    rest = (0.5 * (low + high)).astype(np.float32)
    solver.tendon_seg_rest_length.assign(rest)
    solver.tendon_seg_route_rest_length.assign(rest)
    solver._snapshot_tendon_step_state()
    return rest


def _set_rates(model, solver, state, bodies, rates):
    """Prescribe endpoint translation rates along the two fixed current span directions."""
    prev = state.body_q.numpy().copy()
    delta = solver.tendon_seg_attachment_r.numpy() - solver.tendon_seg_attachment_l.numpy()
    direction = delta / np.linalg.norm(delta, axis=1)[:, None]
    prev[bodies[0], :3] += direction[0] * (rates[0] / 60.0)
    prev[bodies[1], :3] -= direction[1] * (rates[1] / 60.0)
    solver.body_q_prev.assign(prev)


def test_frozen_damped_and_nonlinear(test, device):
    """Recover independent physical segment loads, including signed damping, without averaging."""
    for nonlinear in (False, True):
        for damped in (False, True):
            with test.subTest(nonlinear=nonlinear, damped=damped):
                damping = np.array([100.0, 30.0]) if damped else np.zeros(2)
                model, solver, state, bodies, length, _ = _fixture(
                    device,
                    compliance=(1.0e-3, 1.0e-3),
                    damping=damping,
                    material=SIGMOID if nonlinear else None,
                )
                length = length.astype(float)
                rest = _set_rest(solver, length, np.array([10.0, 20.0]), nonlinear)
                solver._step_tendon_alm_state(1.0 / 60.0)
                rate = np.array([0.03, -0.01])
                _set_rates(model, solver, state, bodies, rate)
                physical = _elastic(length, rest, model.tendon_seg_compliance.numpy(), nonlinear) + damping * rate
                for _ in range(1024):
                    solver._update_tendon_routing(state, 1.0 / 60.0, False)
                    _dual_step(model, solver, state)
                solver._update_tendon_routing(state, 1.0 / 60.0, True)
                solver.check_tendon_material()
                np.testing.assert_allclose(solver.tendon_seg_rest_length.numpy(), rest, atol=2.0e-7, rtol=0.0)
                np.testing.assert_allclose(solver.tendon_seg_alm_lambda.numpy(), physical, atol=0.04, rtol=0.0)
                for body in bodies:
                    args = {"previous_poses": solver.body_q_prev.numpy(), "dt": 1.0 / 60.0}
                    actual, _ = _body_force(model, solver, body, None, **args)
                    solver.tendon_alm = False
                    expected, _ = _body_force(model, solver, body, None, **args)
                    solver.tendon_alm = True
                    np.testing.assert_allclose(actual, expected, atol=0.04, rtol=0.0)


def test_damped_slack(test, device):
    """Keep damped slack spans force-free and conserve material through mixed slack/taut states."""
    for target in ((-10.0, -20.0), (10.0, -20.0)):
        model, solver, state, bodies, _, rest = _fixture(device, target=target, damping=(100.0, 100.0))
        _set_rates(model, solver, state, bodies, (0.3, 0.3))
        for _ in range(128):
            solver._update_tendon_routing(state, 1.0 / 60.0, False)
            _dual_step(model, solver, state)
        solver._update_tendon_routing(state, 1.0 / 60.0, True)
        solver.check_tendon_material()
        after = solver.tendon_seg_rest_length.numpy()
        test.assertAlmostEqual(float(after.astype(float).sum()), float(rest.astype(float).sum()), delta=5.0e-7)
        if sum(target) <= 0:
            np.testing.assert_allclose(solver.tendon_seg_alm_lambda.numpy(), 0.0, atol=0.03, rtol=0.0)


def test_damping_engagement_projection(test, device):
    """Resolve slack-to-taut engagement without a discontinuous damping impulse."""
    model, solver, state, bodies, lengths, rest = _fixture(
        device,
        target=(20.0, -10.0),
        damping=(100.0, 100.0),
    )
    # Freeze equal metrics to isolate the damping-engagement law from mass differences.
    solver.tendon_seg_alm_k.fill_(1.0e4)
    _set_rates(model, solver, state, bodies, (0.3, 0.3))
    solver._update_tendon_cone_rows(model, state.body_q, False)
    cap = float(solver.tendon_link_cap_ratio.numpy()[1])
    total_extension = float((lengths.astype(float) - rest).sum())
    # The right span must receive material transfer before becoming taut. With equal
    # rho/C/D and zero duals, its positive-side force jumps to the equivalent of 30 N.
    # The largest possible taut left/right force ratio is (30 + sum(e)/C)/30 < cap.
    # At e_right <= 0 its force is zero instead. Neither side satisfies the sliding face.
    test.assertLess((30.0 + total_extension / 1.0e-6) / 30.0, cap)
    np.testing.assert_allclose(solver.tendon_seg_alm_k.numpy()[0], solver.tendon_seg_alm_k.numpy()[1])
    for _ in range(1024):
        solver._update_tendon_routing(state, 1.0 / 60.0, True)
        solver.check_tendon_material()
        _dual_step(model, solver, state)
    # Both spans engage during this step: their viscous force is D*extension/dt.
    expected_e = total_extension * np.array([cap, 1.0]) / (cap + 1.0)
    expected_t = expected_e * (1.0e6 + 100.0 * 60.0)
    np.testing.assert_allclose(solver.tendon_seg_rest_length.numpy(), lengths - expected_e, atol=2.0e-7, rtol=0)
    np.testing.assert_allclose(solver.tendon_seg_alm_lambda.numpy(), expected_t, atol=0.09, rtol=0)


def test_zero_force_releases_dual_exactly(test, device):
    """Clear a released multiplier rather than decaying into an unrepresentable slack correction."""
    rest = np.ones(15, dtype=np.float32)
    rest[[8, 13, 14]] += np.array([1.0e-5, 2.0e-5, 3.0e-5], dtype=np.float32)
    model = _chain_model(device, 15, rest_lengths=rest)
    solver = newton.solvers.SolverVBD(
        model,
        tendon_material_direct=True,
        tendon_alm=True,
        tendon_alm_per_segment=True,
        **SIGMOID,
    )
    state = model.state()
    solver.body_q_prev.assign(state.body_q)
    solver.tendon_seg_alm_k.fill_(9408.0)
    lam = np.zeros(15, dtype=np.float32)
    lam[11] = 1.0e-4
    solver.tendon_seg_alm_lambda.assign(lam)
    solver._update_tendon_routing(state, 1.0 / 60.0, True)
    solver.check_tendon_material()
    np.testing.assert_array_equal(solver._tendon_material_state.alm.force.numpy(), 0.0)
    np.testing.assert_array_equal(solver.tendon_seg_alm_lambda.numpy(), 0.0)
    for _ in range(256):
        _dual_step(model, solver, state)
        solver._update_tendon_routing(state, 1.0 / 60.0, True)
        solver.check_tendon_material()
    test.assertAlmostEqual(
        float(solver.tendon_seg_rest_length.numpy().astype(float).sum()), float(rest.astype(float).sum()), delta=2.0e-6
    )


def test_unstretched_dual_release_precision(test, device):
    model, solver, state, _, _, _ = _fixture(device, target=(0.0, 0.0))
    solver.tendon_seg_alm_lambda.assign(np.array([1e-12, 10.0], np.float32))
    _dual_step(model, solver, state)
    lam = solver.tendon_seg_alm_lambda.numpy()
    test.assertEqual(lam[0], 0.0)
    test.assertGreater(lam[1], 0.0, "resolvable stored tension must not be discarded")
    test.assertLess(lam[1], 10.0)


def test_accepted_dynamic_route_release(test, device, capture=False):
    """A crossed dynamic roller is detached in the same accepted step, without losing material."""
    model, body, link = build_force_driven_dynamic_route(device)
    model.body_color_groups = [wp.array([i], dtype=int, device=device) for i in range(model.body_count)]
    solver = newton.solvers.SolverVBD(
        model,
        tendon_alm=True,
        tendon_alm_per_segment=True,
        tendon_material_direct=True,
        tendon_alm_min_stiffness_ratio=0.0,
    )
    state = model.state()
    poses = state.body_q.numpy()
    poses[body, 0] = 0.099
    state.body_q.assign(poses)
    solver.body_q_prev.assign(state.body_q)
    solver._snapshot_tendon_step_state()
    solver._update_tendon_link_active(model, state.body_q)
    solver._prepare_tendon_route(model, state.body_q, 1e-8)
    solver._update_tendon_routing(state, 1 / 1200, False)
    solver._snapshot_tendon_step_state()
    solver._prepare_tendon_route(model, state.body_q, 1e-8)
    test.assertTrue(solver.tendon_link_active.numpy()[link])
    initial = _oriented_route_material_length(solver, model, state, link)
    graph = None
    if capture:
        solver._finalize_tendon_routing(state, 1 / 1200)
        with wp.ScopedCapture(device=device) as captured:
            solver._finalize_tendon_routing(state, 1 / 1200)
        graph = captured.graph
    poses[body, 0] = 0.1001
    state.body_q.assign(poses)
    if graph is None:
        solver._finalize_tendon_routing(state, 1 / 1200)
    else:
        wp.capture_launch(graph)
    solver.check_tendon_material()
    test.assertFalse(solver.tendon_link_active.numpy()[link])
    test.assertEqual(np.count_nonzero(solver.tendon_seg_active.numpy()), 1)
    test.assertAlmostEqual(_oriented_route_material_length(solver, model, state, link), initial, delta=2e-7)
    # Finalization is idempotent and the next step must not merge the arc again.
    rest = solver.tendon_seg_rest_length.numpy().copy()
    if graph is None:
        solver._finalize_tendon_routing(state, 1 / 1200)
    else:
        wp.capture_launch(graph)
    np.testing.assert_allclose(solver.tendon_seg_rest_length.numpy(), rest, atol=2e-7, rtol=0)


def test_accepted_dynamic_route_graph(test, device):
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires CUDA")
    test_accepted_dynamic_route_release(test, device, capture=True)


def test_immobile_span_has_no_alm_regularization(test, device):
    """Do not give a short, rigidly routed span an artificial finite-force ceiling."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        bodies = [
            builder.add_body(
                xform=wp.transform(p=(x, 0.0, 0.0)),
                mass=2.0 if x > 0 else 1.0,
                inertia=wp.mat33(np.eye(3, dtype=np.float32)),
            )
            for x in (-0.1, 0.0, 0.1)
        ]
        builder.add_tendon()
        for body, offset, kind in (
            (bodies[0], 0.0, newton.TendonLinkType.ATTACHMENT),
            (bodies[1], 0.0, newton.TendonLinkType.PINHOLE),
            (bodies[1], 0.001, newton.TendonLinkType.PINHOLE),
            (bodies[2], 0.0, newton.TendonLinkType.ATTACHMENT),
        ):
            builder.add_tendon_link(
                body=body, offset=wp.vec3(offset, 0.0, 0.0), link_type=kind, compliance=1e-6, mu=0.2
            )
        builder.color()
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverVBD(
            model,
            tendon_alm=True,
            tendon_alm_per_segment=True,
            tendon_alm_min_stiffness_ratio=0.0,
            tendon_material_direct=True,
            **SIGMOID,
        )
        state = model.state()
        solver.body_q_prev.assign(state.body_q)
        # These low-level calls bypass step()'s route-inventory initialization.
        solver.tendon_seg_route_rest_length.assign(solver.tendon_seg_rest_length)
        solver._snapshot_tendon_step_state()
        solver._step_tendon_alm_state(0.1)
        rho = solver.tendon_seg_alm_k.numpy()
        # Collinear endpoint moments vanish: w_left=1+1, w_right=1+1/2.
        test.assertAlmostEqual(float(rho[0]), 250.0, delta=0.001)
        test.assertAlmostEqual(float(rho[2]), 1000 / 3, delta=0.001)
        test.assertEqual(float(rho[1]), 0)
        solver.tendon_seg_alm_lambda.fill_(20.0)
        solver._step_tendon_alm_state(0.1)
        test.assertEqual(float(solver.tendon_seg_alm_lambda.numpy()[1]), 0)
        solver._update_tendon_routing(state, 0.1, True)
        solver.check_tendon_material()


def test_multisegment_projection(test, device):
    """Compare 8/32-span frictional chains to an ordinary nonlinear/linear reference."""
    for count in (8, 32):
        for nonlinear in (False, True):
            with test.subTest(count=count, nonlinear=nonlinear), wp.ScopedDevice(device):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                bodies = [
                    builder.add_body(
                        xform=wp.transform(p=(0.2 * i, 0.05 * (-1) ** i, 0.0)),
                        mass=1.0,
                        inertia=wp.mat33(np.eye(3, dtype=np.float32)),
                    )
                    for i in range(count + 1)
                ]
                builder.add_tendon()
                for i, body in enumerate(bodies):
                    builder.add_tendon_link(
                        body=body,
                        link_type=newton.TendonLinkType.ATTACHMENT
                        if i in (0, count)
                        else newton.TendonLinkType.PINHOLE,
                        compliance=1.0e-3,
                        damping=10.0,
                        mu=0.2,
                    )
                builder.color()
                model = builder.finalize(device=device)
                material = SIGMOID if nonlinear else {}
                solver = newton.solvers.SolverVBD(
                    model,
                    iterations=32,
                    tendon_material_direct=True,
                    tendon_alm=True,
                    tendon_alm_per_segment=True,
                    tendon_alm_min_stiffness_ratio=0.0,
                    **material,
                )
                reference = newton.solvers.SolverVBD(model, iterations=32, tendon_material_direct=True, **material)
                state = model.state()
                length = solver.tendon_seg_length.numpy().astype(float)
                target = np.resize(np.array([10.0, 60.0, 15.0, 45.0]), count)
                rest = _set_rest(solver, length, target, nonlinear)
                reference.tendon_seg_rest_length.assign(rest)
                reference.tendon_seg_route_rest_length.assign(rest)
                reference._snapshot_tendon_step_state()
                reference.body_q_prev.assign(state.body_q)
                solver.body_q_prev.assign(state.body_q)
                solver._step_tendon_alm_state(1.0 / 60.0)
                test.assertGreater(float(solver.tendon_seg_alm_k.numpy().min()), 0.0)
                reference._update_tendon_routing(state, 1.0 / 60.0, True)
                reference.check_tendon_material()
                for _ in range(512):
                    solver._update_tendon_routing(state, 1.0 / 60.0, False)
                    _dual_step(model, solver, state)
                solver._update_tendon_routing(state, 1.0 / 60.0, True)
                solver.check_tendon_material()
                actual = solver.tendon_seg_rest_length.numpy().astype(float)
                np.testing.assert_allclose(actual, reference.tendon_seg_rest_length.numpy(), atol=5.0e-6, rtol=0)
                test.assertAlmostEqual(float(actual.sum()), float(rest.astype(float).sum()), delta=2.0e-6)
                physical = _elastic(length, actual, model.tendon_seg_compliance.numpy(), nonlinear)
                np.testing.assert_allclose(solver.tendon_seg_alm_lambda.numpy(), physical, atol=0.04, rtol=0)
                caps = solver.tendon_link_cap_ratio.numpy()[1:-1]
                test.assertTrue(np.all(physical[:-1] <= caps * physical[1:] + 0.04))
                test.assertTrue(np.all(physical[1:] <= caps * physical[:-1] + 0.04))


def test_nonlinear_damped_graph(test, device):
    """Keep nonlinear damping and per-segment duals identical under graph replay."""
    _graph_matches_eager(test, device, material=SIGMOID, damping=(10.0, 10.0))


def test_dynamic_route(test, device):
    """Preserve inventory and reset stale duals during repeated activation and deactivation."""
    for nonlinear in (False, True):
        for damped in (False, True):
            with test.subTest(nonlinear=nonlinear, damped=damped), wp.ScopedDevice(device):
                model, body, link = build_force_driven_dynamic_route(device)
                model.tendon_link_mu.fill_(0.2)
                model.tendon_seg_damping.fill_(10.0 if damped else 0.0)
                model.body_color_groups = [wp.array([i], dtype=int) for i in range(model.body_count)]
                solver = newton.solvers.SolverVBD(
                    model,
                    iterations=32,
                    tendon_material_direct=True,
                    tendon_alm=True,
                    tendon_alm_per_segment=True,
                    tendon_alm_min_stiffness_ratio=0.0,
                    **(SIGMOID if nonlinear else {}),
                )
                state, next_state = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                initial = _oriented_route_material_length(solver, model, state, link)
                for x, expected in ((0.099, True), (0.099, True), (0.25, False), (0.099, True), (0.25, False)):
                    q = state.body_q.numpy()
                    q[body, 0] = x
                    state.body_q.assign(q)
                    state.body_qd.zero_()
                    solver.body_q_prev.assign(state.body_q)
                    state.clear_forces()
                    solver.step(state, next_state, model.control(), None, 1.0 / 1200.0)
                    state, next_state = next_state, state
                    solver.check_tendon_material()
                    test.assertEqual(bool(solver.tendon_link_active.numpy()[link]), expected)
                    total = _oriented_route_material_length(solver, model, state, link)
                    test.assertAlmostEqual(total, initial, delta=2.0e-6)
                    active = solver.tendon_seg_active.numpy().astype(bool)
                    np.testing.assert_array_equal(solver.tendon_seg_alm_lambda.numpy()[~active], 0.0)
                    test.assertTrue(np.isfinite(state.body_q.numpy()).all())


def test_nonlinear_sliding(test, device):
    """Reach the physical sigmoid capstan boundary in both transfer directions with damping."""
    for reverse in (False, True):
        for damped in (False, True):
            with test.subTest(reverse=reverse, damped=damped):
                damping = np.array([100.0, 30.0]) if damped else np.zeros(2)
                model, solver, state, bodies, length, _ = _fixture(device, damping=damping, material=SIGMOID)
                length = length.astype(float)
                target = np.array([10.0, 60.0]) if reverse else np.array([60.0, 10.0])
                rest = _set_rest(solver, length, target, True)
                solver._step_tendon_alm_state(1.0 / 60.0)
                rate = np.array([0.03, -0.01])
                _set_rates(model, solver, state, bodies, rate)
                solver._update_tendon_cone_rows(model, state.body_q, False)
                cap = float(solver.tendon_link_cap_ratio.numpy()[1])
                # Independent scalar bisection: conserve rest length and saturate the selected cone face.
                lo, hi = rest[0] - 0.1, rest[0] + 0.1
                for _ in range(70):
                    mid = 0.5 * (lo + hi)
                    candidate = np.array([mid, rest.astype(float).sum() - mid])
                    tension = _elastic(length, candidate, model.tendon_seg_compliance.numpy(), True) + damping * rate
                    residual = tension[0] - tension[1] / cap if reverse else tension[0] - cap * tension[1]
                    if residual > 0:
                        lo = mid
                    else:
                        hi = mid
                expected = np.array([mid, rest.astype(float).sum() - mid])
                for _ in range(1024):
                    solver._update_tendon_routing(state, 1.0 / 60.0, False)
                    _dual_step(model, solver, state)
                solver._update_tendon_routing(state, 1.0 / 60.0, True)
                solver.check_tendon_material()
                actual = solver.tendon_seg_rest_length.numpy()
                np.testing.assert_allclose(actual, expected, atol=5.0e-7, rtol=0.0)
                physical = _elastic(length, actual, model.tendon_seg_compliance.numpy(), True) + damping * rate
                np.testing.assert_allclose(solver.tendon_seg_alm_lambda.numpy(), physical, atol=0.025, rtol=0.0)


class TestTendonSegmentALMExtended(unittest.TestCase):
    """Check physical references separately from the solver's reported tensions."""

    def test_nonlinear_damped_motion(self):
        """Match a converged ordinary VBD reference through nonlinear motion and reversal.

        The per-segment ALM row damps the free-span stretch rate with roller-axis spin removed. The
        ordinary VBD row defaults to contact-point damping, a different damping law, so the reference
        uses the ALM row's damping rate; the comparison then isolates the ALM row from the penalty row.
        """

        def alm_damping_rate(solver):
            solver._tendon_material_state.legacy_alm_damping = True

        for nonlinear in (False, True):
            material = SIGMOID if nonlinear else {}
            reference = _simulate_pulley(
                iterations=128,
                duration=0.05,
                velocity=2.0,
                preload=1.0,
                damping=10.0,
                solver_options=material,
                configure_solver=alm_damping_rate,
            )
            result = _simulate_pulley(
                iterations=128,
                duration=0.05,
                velocity=2.0,
                preload=1.0,
                damping=10.0,
                solver_options=material
                | {
                    "tendon_alm": True,
                    "tendon_alm_per_segment": True,
                    "tendon_alm_min_stiffness_ratio": 0.0,
                },
            )
            np.testing.assert_allclose(result.angle, reference.angle, atol=2.0e-5, rtol=0.0)
            np.testing.assert_allclose(result.velocity, reference.velocity, atol=2.0e-3, rtol=0.0)


for _device in get_test_devices():
    for _test in (
        test_frozen_damped_and_nonlinear,
        test_nonlinear_sliding,
        test_dynamic_route,
        test_damped_slack,
        test_damping_engagement_projection,
        test_zero_force_releases_dual_exactly,
        test_unstretched_dual_release_precision,
        test_accepted_dynamic_route_release,
        test_accepted_dynamic_route_graph,
        test_continuous_damping_rows,
        test_sigmoid_knee_projection,
        test_short_movable_span_metric,
        test_cancelling_damping_projection,
        test_immobile_span_has_no_alm_regularization,
        test_nonlinear_force_linearization,
        test_nonlinear_body_tangent,
        test_slack_cancellation_has_exact_zero_force,
        test_multisegment_projection,
        test_nonlinear_damped_graph,
    ):
        add_function_test(TestTendonSegmentALMExtended, _test.__name__, _test, devices=[_device])

if __name__ == "__main__":
    unittest.main()
