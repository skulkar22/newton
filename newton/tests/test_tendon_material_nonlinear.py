# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Certify nonlinear capstan forces, least transfer, and rounded publication."""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.tendon_material_cooperative import (
    solve_tendon_material_nonlinear_component as project_cooperative,
)
from newton._src.solvers.tendon_material_nonlinear import (
    TendonMaterialNonlinearState,
    TendonMaterialNonlinearStatus,
    _certify_exact_publication,
    _elastic_derivative,
    _inverse_extension,
    _inverse_extension_and_tangent,
    allocate_tendon_material_nonlinear_state,
    solve_tendon_material_nonlinear_component,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel(enable_backward=False)
def _check_publication(state: TendonMaterialNonlinearState, n: int, min_rest: float, tolerance: float):
    good = _certify_exact_publication(state, n, 0, 0, 0, 100.0, 1.0, 0.1, 0.05, min_rest, tolerance)
    state.status[0] = int(TendonMaterialNonlinearStatus.NUMERICAL_FAILURE)
    if good:
        for i in range(n):
            state.output[i] = state.candidate[i]
        state.status[0] = int(TendonMaterialNonlinearStatus.CONVERGED)
        if state.publication_residual[0] > tolerance:
            state.status[0] = int(TendonMaterialNonlinearStatus.CONVERGED_ROUNDED)


@wp.kernel(enable_backward=False)
def _invert_force(
    force: wp.array[wp.float64], length: wp.array[wp.float64], ea_low: wp.float64, result: wp.array[wp.float64]
):
    i = wp.tid()
    result[i] = _inverse_extension(force[i], length[i], ea_low, wp.float64(10.0), wp.float64(0.01), wp.float64(0.003))


@wp.kernel(enable_backward=False)
def _invert_force_with_tangent(inputs: wp.array2d[wp.float64], results: wp.array2d[wp.float64]):
    i = wp.tid()
    inverse = _inverse_extension_and_tangent(
        inputs[i, 0], inputs[i, 1], inputs[i, 2], inputs[i, 3], inputs[i, 4], inputs[i, 5], inputs[i, 6]
    )
    results[i, 0] = inverse[0]
    results[i, 1] = inverse[1]
    results[i, 2] = _elastic_derivative(
        inputs[i, 1], inverse[0], inputs[i, 2], inputs[i, 3], inputs[i, 4], inputs[i, 5]
    )


@wp.kernel(enable_backward=False)
def _project_exact(
    state: TendonMaterialNonlinearState,
    n: int,
    ea_low: float,
    ea_ratio: float,
    transition_strain: float,
    transition_width: float,
    min_rest: float,
    max_iterations: int,
    tolerance: float,
):
    solve_tendon_material_nonlinear_component(
        state,
        n,
        1,
        wp.tid(),
        n,
        n - 1,
        ea_low,
        ea_ratio,
        transition_strain,
        transition_width,
        min_rest,
        max_iterations,
        tolerance,
    )


@wp.kernel(enable_backward=False)
def _project_cooperative(
    state: TendonMaterialNonlinearState,
    n: int,
    ea_low: float,
    ea_ratio: float,
    transition_strain: float,
    transition_width: float,
    min_rest: float,
    max_iterations: int,
    tolerance: float,
):
    project_cooperative(
        state,
        n,
        1,
        wp.tid() // 32,
        n,
        n - 1,
        ea_low,
        ea_ratio,
        transition_strain,
        transition_width,
        min_rest,
        max_iterations,
        tolerance,
        wp.tid() % 32,
    )


def _launch(kernel, *, dim, inputs, device):
    if kernel is _project_exact and wp.get_device(device).is_cuda:
        wp.launch(_project_cooperative, dim=dim * 32, block_dim=32, inputs=inputs, device=device)
    else:
        wp.launch(kernel, dim=dim, inputs=inputs, device=device)


def _fixture(device, reference, lengths, caps, damping=0.0, ratio=1.0, iterations=64):
    reference = np.atleast_2d(reference).astype(np.float32)
    count, n = reference.shape
    state = allocate_tendon_material_nonlinear_state(count * n, count * (n - 1), count, device=device)
    for name, value, shape in (
        ("reference", reference, (count, n)),
        ("length", lengths, (count, n)),
        ("damping", damping, (count, n)),
        ("cap", caps, (count, n - 1)),
    ):
        getattr(state, name).assign(np.broadcast_to(value, shape).astype(np.float32).ravel())
    state.output.fill_(-123.0)
    args = [state, n, 100.0, ratio, 0.1, 0.05, 1.0e-5, iterations, 2.0e-6]
    return state, args, count


def _assert_certificate(
    test,
    state,
    n,
    *,
    ratio=1.0,
    ea_low=100.0,
    transition_strain=0.1,
    transition_width=0.05,
    tolerance=4.0e-6,
    allow_slack=False,
    exact=False,
):
    reference = state.reference.numpy().astype(np.float64).reshape(-1, n)
    published = state.output.numpy().astype(np.float64).reshape(-1, n)
    output = state.corrected.numpy().reshape(-1, n) if exact else published
    lengths = state.length.numpy().astype(np.float64).reshape(-1, n)
    damping = state.damping.numpy().astype(np.float64).reshape(-1, n)
    caps = state.cap.numpy().astype(np.float64).reshape(-1, n - 1)
    if exact:
        test.assertTrue(np.all(state.status.numpy() > 0))
        np.testing.assert_array_equal(published.astype(np.float32), output.astype(np.float32))
        delta = published - output
        prefix = np.cumsum(delta, axis=1)
        measured = np.column_stack(
            (
                np.max(
                    np.abs(state.tension.numpy().reshape(-1, n) - state.exact_tension.numpy().reshape(-1, n)), axis=1
                ),
                np.max(np.abs(prefix[:, :-1]), axis=1) if n > 1 else np.zeros(len(output)),
                np.abs(prefix[:, -1]),
                np.sum(np.abs(delta), axis=1),
            )
        )
        np.testing.assert_allclose(state.storage_error.numpy(), measured, rtol=2.0e-12, atol=1.0e-30)
        test.assertTrue(np.all(lengths - published >= 1.0e-5))
    else:
        np.testing.assert_array_equal(state.status.numpy(), int(TendonMaterialNonlinearStatus.CONVERGED))
    if not allow_slack:
        test.assertTrue(np.all(output >= 0.0))
    test.assertTrue(np.all(lengths - output >= 1.0e-5))
    if exact:
        # The public kernel parameters are float32 even though this certificate
        # evaluates the constitutive law in double precision. Near cancellation,
        # substituting unrounded Python constants describes a different law.
        ea_low, ratio, transition_strain, transition_width = (
            float(np.float32(value)) for value in (ea_low, ratio, transition_strain, transition_width)
        )
    strain = np.maximum(output, 0.0) / np.maximum(lengths - output, 1.0e-8)
    ea = ea_low * (1.0 + (ratio - 1.0) * 0.5 * (1.0 + np.tanh((strain - transition_strain) / transition_width)))
    tension = np.maximum(ea * strain + damping, 0.0)
    for row in range(len(output)):
        peak = max(float(tension[row].max()), 1.0e-20)
        flow_tolerance = tolerance * peak * float(np.min((lengths[row] - output[row]) / ea[row]))
        scale = max(float(np.abs(reference[row]).sum() + np.abs(output[row]).sum()), 1.0e-12)
        test.assertLessEqual(abs(float((output[row] - reference[row]).sum())), tolerance * scale)
        flow = np.cumsum(reference[row] - output[row])[:-1]
        left, right = tension[row, :-1], tension[row, 1:]
        positive_gap = caps[row] * right - left
        negative_gap = caps[row] * left - right
        test.assertTrue(np.all(positive_gap >= -tolerance * peak))
        test.assertTrue(np.all(negative_gap >= -tolerance * peak))
        test.assertTrue(np.all(np.abs(positive_gap[flow > flow_tolerance]) <= tolerance * peak))
        test.assertTrue(np.all(np.abs(negative_gap[flow < -flow_tolerance]) <= tolerance * peak))


def _analytic_two_span(reference, lengths, cap, damping):
    """Solve the active capstan face's constant-EA variable-rest quadratic."""
    total = float(np.sum(np.asarray(reference, dtype=np.float32), dtype=np.float64))
    l0, l1 = np.asarray(lengths, dtype=np.float32).astype(np.float64)
    d0, d1 = np.asarray(damping, dtype=np.float32).astype(np.float64)
    cap = float(np.float32(cap))
    ea = 100.0
    base = l1 - total
    difference = d0 - cap * d1
    a = ea * (1.0 - cap) - difference
    b = ea * base + cap * ea * (total + l0) + difference * (l0 - base)
    c = -cap * ea * total * l0 + difference * l0 * base
    if abs(a) < 1.0e-12:
        extension = -c / b
    else:
        # This stable root lies in the physical interval for these taut cases.
        extension = -2.0 * c / (b + math.sqrt(b * b - 4.0 * a * c))
    return np.array([extension, total - extension])


def _check_known_sigmoid_solutions(test, device, project=_project_exact):
    """Converge at full relaxation for both slip signs, loads, and damping offsets."""
    lengths = np.array([0.003, 0.05])
    for ea_low, strain_left in ((500.0, 0.00897), (1.0e5, 1.0e-4), (1.0e5, 0.00897), (500.0, 0.03)):
        strains = strain_left * np.array([1.0, 0.96])
        elastic = ea_low * (1.0 + 4.5 * (1.0 + np.tanh((strains - 0.01) / 0.003))) * strains
        expected = lengths * strains / (1.0 + strains)
        for damped in (False, True):
            damping = elastic * np.array([0.07, -0.03]) if damped else np.zeros(2)
            cap = (elastic[0] + damping[0]) / (elastic[1] + damping[1])
            transfer = 0.2 * min(expected)
            reference = expected + np.array([transfer, -transfer])
            state, inputs, count = _fixture(
                device,
                [reference, reference[::-1]],
                [lengths, lengths[::-1]],
                [cap],
                [damping, damping[::-1]],
                ratio=10.0,
            )
            inputs[2] = ea_low
            inputs[4] = 0.01
            inputs[5] = 0.003
            _launch(project, dim=count, inputs=inputs, device=device)
            _assert_certificate(
                test,
                state,
                2,
                ratio=10.0,
                ea_low=ea_low,
                transition_strain=0.01,
                transition_width=0.003,
                exact=project is _project_exact,
            )
            np.testing.assert_allclose(state.output.numpy().reshape(2, 2), [expected, expected[::-1]], rtol=4.0e-6)


def _check_feasible_force_floors_are_unchanged(test, device, project=_project_exact):
    """Preserve feasible zero and positive force floors exactly."""
    for reference, damping, cap in (
        ([-0.02, -0.2], [0.0, 0.0], 1.2),
        ([0.0, 0.0], [0.0, 0.0], 1.2),
        ([-0.02, -0.2], [2.0, 1.0], 2.0),
        ([0.08, 0.12], [-10.0, -20.0], 1.2),
    ):
        state, inputs, count = _fixture(device, reference, [1.0, 1.0], [cap], damping)
        _launch(project, dim=count, inputs=inputs, device=device)
        test.assertEqual(state.status.numpy()[0], TendonMaterialNonlinearStatus.CONVERGED)
        np.testing.assert_array_equal(state.output.numpy(), state.reference.numpy())
        test.assertEqual(state.outer_iterations.numpy()[0], 0)
        if project is _project_exact:
            np.testing.assert_array_equal(state.corrected.numpy(), state.reference.numpy().astype(np.float64))
            np.testing.assert_array_equal(state.storage_error.numpy(), np.zeros((1, 4)))
            test.assertEqual(state.residual.numpy()[0], state.publication_residual.numpy()[0])


def _check_zero_force_least_transfer(test, device, project=_project_exact):
    """Project zero-force material with exact damping-dependent extension caps."""
    for reference, damping, expected in (
        ([0.03, -0.05], [0.0, 0.0], [0.0, -0.02]),
        ([0.14, 0.08], [-10.0, -20.0], [1.0 / 11.0, 0.22 - 1.0 / 11.0]),
    ):
        state, inputs, count = _fixture(device, reference, [1.0, 1.0], [1.2], damping)
        _launch(project, dim=count, inputs=inputs, device=device)
        if project is _project_exact:
            test.assertGreater(state.status.numpy()[0], 0)
            test.assertEqual(state.inner_status.numpy()[0], 0)
            output = state.corrected.numpy()
        else:
            test.assertEqual(state.status.numpy()[0], TendonMaterialNonlinearStatus.CONVERGED)
            test.assertEqual(state.inner_status.numpy()[0], 4)
            output = state.output.numpy().astype(np.float64)
        tension = np.maximum(100.0 * np.maximum(output, 0.0) / (1.0 - output) + damping, 0.0)
        np.testing.assert_array_equal(tension, np.zeros(2))
        np.testing.assert_allclose(output, expected, rtol=3.0e-6, atol=1.0e-9)
        test.assertLess(abs(float(np.sum(output - state.reference.numpy()))), 1.0e-7)
        test.assertEqual(state.valid.numpy()[0], 0)
        if project is _project_exact:
            exact_expected = np.asarray(expected).copy()
            exact_expected[1] = float(state.reference.numpy().astype(np.float64).sum()) - exact_expected[0]
            np.testing.assert_allclose(output, exact_expected, rtol=0.0, atol=8.0e-16)
            np.testing.assert_array_equal(state.output.numpy(), output.astype(np.float32))
            np.testing.assert_array_equal(state.exact_tension.numpy(), np.zeros(2))
            _assert_certificate(test, state, 2, allow_slack=True, exact=True)


def _check_multiple_plateaus_choose_least_net_flow(test, device, project=_project_exact):
    """Minimize cumulative transfer rather than regularized span displacement."""
    for caps, damping, force in (([1.0, 1.0], [5.0, 5.0, 0.0], 5.0), ([1.25, 1.25], [5.0, 4.0, 0.0], 3.2)):
        state, inputs, count = _fixture(device, [-0.01, -0.02, 0.0], [1.0, 1.0, 1.0], caps, damping)
        _launch(project, dim=count, inputs=inputs, device=device)
        _assert_certificate(test, state, 3, allow_slack=True, exact=project is _project_exact)
        moved = force / (100.0 + force)
        np.testing.assert_allclose(state.output.numpy(), [-0.01, -0.02 - moved, moved], rtol=2.0e-6)
        test.assertEqual(state.output.numpy()[0], state.reference.numpy()[0])


def _check_graph_force_floor_transitions(test, device, project=_project_exact):
    """Replay changing slack, taut, and positive-floor states in one graph."""
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")
    state, inputs, count = _fixture(device, [-0.01, 0.0], [1.0, 1.0], [1.2], [5.0, 0.0])
    _launch(project, dim=count, inputs=inputs, device=device)
    with wp.ScopedCapture(device=device) as capture:
        _launch(project, dim=count, inputs=inputs, device=device)
    for reference, damping in (([0.03, -0.05], [0.0, 0.0]), ([-0.02, 0.2], [1.0, -0.5]), ([-0.01, 0.0], [5.0, 0.0])):
        state.reference.assign(np.asarray(reference, dtype=np.float32))
        state.damping.assign(np.asarray(damping, dtype=np.float32))
        wp.capture_launch(capture.graph)
        _assert_certificate(test, state, 2, allow_slack=True, exact=project is _project_exact)


def test_exact_releases_incompatible_floor_face(test, device):
    """Release a capstan face that incorrectly couples distinct positive floors."""
    reference = np.array(
        [0.0, -1.6282759862651375e-10, -5.999388485999901e-12, 3.953993035765713e-11, -4.551884563719355e-11]
    )
    lengths = np.array(
        [0.05000000447034836, 0.015850551426410675, 0.017233693972229958, 0.026472575962543488, 0.010692402720451355]
    )
    damping = np.array(
        [
            -1.4271119397335497e-8,
            2.553629833632054e-12,
            1.1460289783826738e-7,
            4.735301217806409e-8,
            1.2353015677035728e-7,
        ]
    )
    caps = np.array([1.0323779582977295, 1.0203989744186401, 1.032899260520935, 1.0499523878097534])
    expected = np.array([12.166881311, 3.520062083, -148.691650595, 3.717650446, -45.518845637]) * 1.0e-12
    for reflect in (False, True):
        order = slice(None, None, -1) if reflect else slice(None)
        state, inputs, count = _fixture(
            device, reference[order], lengths[order], caps[order], damping[order], ratio=10.0
        )
        inputs[2], inputs[4], inputs[5] = 500.0, 0.01, 0.003
        _launch(_project_exact, dim=count, inputs=inputs, device=device)
        _assert_certificate(
            test,
            state,
            5,
            ratio=10.0,
            ea_low=500.0,
            transition_strain=0.01,
            transition_width=0.003,
            allow_slack=True,
            exact=True,
        )
        np.testing.assert_allclose(state.output.numpy(), expected[order], rtol=0.0, atol=1.0e-17)
        np.testing.assert_array_equal(state.faces.numpy(), [0, 1, 1, 1] if reflect else [-1, -1, -1, 0])
        unchanged = 0 if reflect else 4
        test.assertEqual(state.output.numpy()[unchanged], state.reference.numpy()[unchanged])


def test_exact_known_sigmoid_solutions(test, device):
    _check_known_sigmoid_solutions(test, device, _project_exact)


def test_exact_rounding_qualified_force_tolerance(test, device):
    """Separate a strict exact solution from a measurable float32 force limitation."""
    state, inputs, count = _fixture(
        device,
        [-8.465591236017644e-5, 0.00022895466827321798],
        [0.07578499615192413, 0.04852620139718056],
        [1.6499861478805542],
        [-4.402872562408447, 0.004285832867026329],
        ratio=10.0,
    )
    inputs[2], inputs[4], inputs[5], inputs[8] = 500.0, 0.01, 0.003, 1.0e-5
    _launch(_project_exact, dim=count, inputs=inputs, device=device)
    _assert_certificate(
        test,
        state,
        2,
        ratio=10.0,
        ea_low=500.0,
        transition_strain=0.01,
        transition_width=0.003,
        tolerance=1.0e-5,
        allow_slack=True,
        exact=True,
    )
    test.assertEqual(state.status.numpy()[0], TendonMaterialNonlinearStatus.CONVERGED_ROUNDED)
    test.assertLessEqual(state.residual.numpy()[0], 1.0e-5)
    test.assertGreater(state.publication_residual.numpy()[0], 1.0e-5)
    test.assertGreater(state.storage_error.numpy()[0, 0], 8.0e-8)


def test_exact_publication_rejects_bad_math_and_non_nearest_storage(test, device):
    """Storage diagnostics cannot excuse an invalid exact allocation or arbitrary publication."""
    reference, lengths, cap = [0.3, 0.1], [1.0, 1.0], 1.4
    expected = _analytic_two_span(reference, lengths, cap, [0.0, 0.0])
    for mutation in ("none", "wrong_face", "wrong_force", "mass", "nan_exact", "non_nearest", "nan_stored"):
        state, _inputs, _count = _fixture(device, reference, lengths, [cap])
        corrected = expected.copy()
        if mutation == "wrong_face":
            corrected = corrected[::-1].copy()
        elif mutation == "wrong_force":
            corrected += np.array([0.03, -0.03])
        elif mutation == "mass":
            corrected += 0.01
        elif mutation == "nan_exact":
            corrected[0] = np.nan
        published = corrected.astype(np.float32)
        if mutation == "non_nearest":
            published[0] = np.nextafter(published[0], np.float32(np.inf))
        elif mutation == "nan_stored":
            published[0] = np.nan
        state.corrected.assign(corrected)
        state.candidate.assign(published)
        _launch(_check_publication, dim=1, inputs=[state, 2, 1.0e-5, 2.0e-6], device=device)
        if mutation == "none":
            _assert_certificate(test, state, 2, exact=True)
        else:
            test.assertEqual(state.status.numpy()[0], TendonMaterialNonlinearStatus.NUMERICAL_FAILURE)
            np.testing.assert_array_equal(state.output.numpy(), [-123.0, -123.0])


def test_exact_publication_preserves_hard_rest_bound(test, device):
    """Even prescribed nearest rounding cannot cross the physical rest bound."""
    minimum_rest = float(np.float32(0.09999999))
    exact = 1.0 - minimum_rest
    nearest = np.float32(exact)
    reference = np.nextafter(nearest, np.float32(-np.inf))
    test.assertLess(1.0 - float(nearest), minimum_rest)
    state, _inputs, _count = _fixture(device, [reference], [1.0], [])
    state.corrected.assign(np.array([exact], dtype=np.float64))
    state.candidate.assign(np.array([nearest], dtype=np.float32))
    _launch(_check_publication, dim=1, inputs=[state, 1, minimum_rest, 2.0e-6], device=device)
    test.assertLessEqual(state.residual.numpy()[0], 2.0e-6)
    test.assertEqual(state.status.numpy()[0], TendonMaterialNonlinearStatus.NUMERICAL_FAILURE)
    np.testing.assert_array_equal(state.output.numpy(), [-123.0])


def test_exact_repairs_overbound_rolling_trial(test, device):
    """Project a full-rolling trial before enforcing individual span rest bounds."""
    # Both spans start with extension 0.002 m and rest length 0.098 m.
    # Full rolling transfers 0.11 m: its no-slip trial crosses a rest bound,
    # but capstan slip can restore valid rest lengths without losing material.
    reference = np.array([-0.108, 0.112], dtype=np.float32)
    length = float(np.float32(0.1))
    total = float(reference.astype(float).sum())

    def tension(extension):
        strain = extension / (length - extension)
        return (
            100.0
            * (1.0 + 4.5 * (1.0 + math.tanh((strain - float(np.float32(0.1))) / float(np.float32(0.05)))))
            * strain
        )

    # Independent scalar root of T_right = 2 T_left on the slipping face.
    lower, upper = 0.0, total
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if tension(total - middle) > 2.0 * tension(middle):
            lower = middle
        else:
            upper = middle
    left = 0.5 * (lower + upper)
    expected = np.array([left, total - left])
    state, inputs, count = _fixture(device, [reference, reference[::-1]], [length, length], [2.0], ratio=10.0)
    original = state.reference.numpy().copy()
    for _ in range(2):
        # Check both cold and cached-face solves and both rolling directions.
        _launch(_project_exact, dim=count, inputs=inputs, device=device)
        _assert_certificate(test, state, 2, ratio=10.0, exact=True)
        np.testing.assert_allclose(state.corrected.numpy().reshape(2, 2), [expected, expected[::-1]], rtol=2.0e-6)
        np.testing.assert_array_equal(state.reference.numpy(), original)


def test_exact_rejects_invalid_geometry_and_infeasible_inventory(test, device):
    """Accepting trial extensions must not permit invalid geometry or final rest lengths."""
    for lengths, reference in (
        ([-0.1, 0.1], [-0.2, 0.0]),
        ([np.nan, 0.1], [0.0, 0.0]),
        ([np.inf, 0.1], [0.0, 0.0]),
        ([0.1, 0.1], [np.nan, 0.0]),
        ([0.1, 0.1], [np.inf, 0.0]),
        # There is not enough material for even the sum of minimum rest lengths.
        ([0.1, 0.1], [0.1, 0.1]),
        ([0.1], [0.11]),
    ):
        with test.subTest(lengths=lengths, reference=reference):
            n = len(lengths)
            state, inputs, count = _fixture(device, reference, lengths, np.full(n - 1, 2.0), ratio=10.0)
            _launch(_project_exact, dim=count, inputs=inputs, device=device)
            test.assertLess(state.status.numpy()[0], 0)
            test.assertEqual(state.valid.numpy()[0], 0)
            np.testing.assert_array_equal(state.output.numpy(), np.full(n, -123.0))


def test_exact_cache_revalidates_inputs_and_preserves_failure(test, device):
    """Re-solve cached faces for changed inputs; a rejected cache cannot publish."""
    state, inputs, count = _fixture(device, [0.3, 0.1], [1.0, 1.0], [1.4])
    _launch(_project_exact, dim=count, inputs=inputs, device=device)
    _assert_certificate(test, state, 2, exact=True)
    test.assertGreater(state.outer_iterations.numpy()[0], 1)
    test.assertEqual(state.valid.numpy()[0], 1)
    changed = {
        "reference": [0.32, 0.09],
        "length": [1.05, 0.9],
        "damping": [0.5, -0.2],
        "cap": [1.3],
    }
    for name, value in changed.items():
        getattr(state, name).assign(np.asarray(value, dtype=np.float32))
    _launch(_project_exact, dim=count, inputs=inputs, device=device)
    _assert_certificate(test, state, 2, exact=True)
    test.assertEqual(state.outer_iterations.numpy()[0], 1)
    expected = _analytic_two_span(changed["reference"], changed["length"], changed["cap"][0], changed["damping"])
    np.testing.assert_allclose(state.output.numpy(), expected, rtol=3.0e-6)
    accepted = state.output.numpy().copy()
    for name in ("reference", "length", "damping"):
        getattr(state, name).assign(np.asarray(changed[name][::-1], dtype=np.float32))
    inputs[7] = 1
    _launch(_project_exact, dim=count, inputs=inputs, device=device)
    test.assertEqual(state.status.numpy()[0], TendonMaterialNonlinearStatus.MAX_ITERATIONS)
    test.assertEqual(state.outer_iterations.numpy()[0], 1)
    np.testing.assert_array_equal(state.output.numpy(), accepted)
    np.testing.assert_array_equal(state.reference.numpy(), np.asarray(changed["reference"][::-1], dtype=np.float32))
    inputs[7] = 64
    _launch(_project_exact, dim=count, inputs=inputs, device=device)
    _assert_certificate(test, state, 2, exact=True)
    np.testing.assert_allclose(state.output.numpy(), expected[::-1], rtol=3.0e-6)
    _launch(_project_exact, dim=count, inputs=inputs, device=device)
    test.assertEqual(state.outer_iterations.numpy()[0], 1)


def test_exact_storage_conservation_error_is_explicit(test, device):
    """Report a coarse plateau's storage error without moving the exact elastic solution."""
    reference = np.array(
        [-0.0019891858100891113, 5.9190933825448155e-6, 1.897531001304742e-5, 4.888146122539183e-6, 7.86988948675571e-7]
    )
    lengths = np.array(
        [0.015161670744419098, 0.015850558876991272, 0.03149638697504997, 0.028864741325378418, 0.010692405514419079]
    )
    damping = np.array(
        [0.08134332299232483, 0.024075232446193695, 0.07983347773551941, -0.0908929854631424, 0.07099602371454239]
    )
    caps = np.array([1.032394528388977, 1.213971734046936, 1.0385125875473022, 1.2021379470825195])
    expected = np.array(
        [
            -0.001973408184428859,
            1.8755591109966482e-6,
            1.3767429189572911e-6,
            1.0752621827925233e-5,
            7.86988948675571e-7,
        ]
    )
    for reflect in (False, True):
        order = slice(None, None, -1) if reflect else slice(None)
        state, inputs, count = _fixture(
            device, reference[order], lengths[order], caps[order], damping[order], ratio=10.0
        )
        inputs[2], inputs[4], inputs[5], inputs[8] = 500.0, 0.01, 0.003, 1.0e-5
        _launch(_project_exact, dim=count, inputs=inputs, device=device)
        _assert_certificate(
            test,
            state,
            5,
            ratio=10.0,
            ea_low=500.0,
            transition_strain=0.01,
            transition_width=0.003,
            tolerance=1.0e-5,
            allow_slack=True,
            exact=True,
        )
        np.testing.assert_allclose(state.output.numpy(), expected[order], rtol=0.0, atol=1.0e-10)
        unchanged = 0 if reflect else 4
        test.assertEqual(state.output.numpy()[unchanged], state.reference.numpy()[unchanged])


def test_inverse_hint_and_reused_tangent(test, device):
    """Keep double-precision force and tangent accuracy despite approximate guesses."""
    rows = []
    strains = np.concatenate(([0.0], np.geomspace(1.0e-12, 1.0, 25), [0.009, 0.01, 0.011]))
    for low, ratio, width in (
        (500.0, 10.0, 0.003),
        (500.0, 10.0, 0.0001),
        (0.01, 2.0, 0.003),
        (1.0e7, 1000.0, 0.003),
        (500.0, 1.0, 0.003),
    ):
        for length in (5.0e-9, 5.0e-8, 0.001, 0.1, 100.0):
            for strain in strains:
                force = low * (1.0 + (ratio - 1.0) * 0.5 * (1.0 + np.tanh((strain - 0.01) / width))) * strain
                for hint in (-1.0, strain * 0.9, strain * 1.1, np.nan, np.inf):
                    rows.append([force, length, low, ratio, 0.01, width, hint])
    data = np.asarray(rows, dtype=np.float64)
    output = wp.empty((len(data), 3), dtype=wp.float64, device=device)
    _launch(
        _invert_force_with_tangent,
        dim=len(data),
        inputs=[wp.array(data, dtype=wp.float64, device=device), output],
        device=device,
    )
    result = output.numpy()
    test.assertTrue(np.all(np.isfinite(result)))
    test.assertTrue(np.all(result[:, 1] > 0.0))
    extension = result[:, 0]
    rest = np.maximum(data[:, 1] - extension, 1.0e-8)
    strain = np.maximum(extension, 0.0) / rest
    force = data[:, 2] * (1.0 + (data[:, 3] - 1.0) * 0.5 * (1.0 + np.tanh((strain - data[:, 4]) / data[:, 5]))) * strain
    np.testing.assert_allclose(force, data[:, 0], rtol=2.0e-12, atol=0.0)
    np.testing.assert_allclose(result[:, 1], result[:, 2], rtol=2.0e-12, atol=0.0)


def test_inverse_progress_across_sigmoid_knee(test, device):
    """A bracketed Newton step must not alternate indefinitely near its endpoints."""
    targets = np.concatenate((np.geomspace(1.0e-8, 1.0e4, 64), [26.868515014648438, 25.850]))
    lengths = np.linspace(0.01, 1.0, len(targets))
    for ea_low in (500.0, 1.0e5):
        force = targets * (ea_low / 500.0)
        result = wp.zeros(len(force), dtype=wp.float64, device=device)
        _launch(
            _invert_force,
            dim=len(force),
            device=device,
            inputs=[
                wp.array(force, dtype=wp.float64, device=device),
                wp.array(lengths, dtype=wp.float64, device=device),
                ea_low,
                result,
            ],
        )
        extension = result.numpy()
        strain = extension / np.maximum(lengths - extension, 1.0e-8)
        actual = ea_low * (1.0 + 4.5 * (1.0 + np.tanh((strain - 0.01) / 0.003))) * strain
        np.testing.assert_allclose(actual, force, rtol=1.0e-12, atol=1.0e-20)


def test_exact_equal_force_knee_inverse(test, device):
    """Invert a negative damping span at the knee while another span is on its floor."""
    reference = np.array([-0.002359138336032629, -0.0003570818225853145, 0.0006572366692125797])
    lengths = np.array([0.0859721228480339, 0.09283291548490524, 0.05050825700163841])
    damping = np.array([10.228294372558594, -16.640220642089844, 8.7294340133667])
    for reflect in (False, True):
        order = slice(None, None, -1) if reflect else slice(None)
        state, inputs, count = _fixture(
            device, reference[order], lengths[order], [1.0, 1.0], damping[order], ratio=10.0
        )
        inputs[2], inputs[4], inputs[5] = 500.0, 0.01, 0.003
        _launch(_project_exact, dim=count, inputs=inputs, device=device)
        _assert_certificate(
            test,
            state,
            3,
            ratio=10.0,
            ea_low=500.0,
            transition_strain=0.01,
            transition_width=0.003,
            allow_slack=True,
            exact=True,
        )
        np.testing.assert_allclose(state.tension.numpy(), np.full(3, damping.max()), rtol=2.0e-6)
        test.assertAlmostEqual(float(state.output.numpy()[1]), 0.0009135047410666881, delta=1.0e-10)


def test_exact_force_floor_regimes(test, device):
    _check_feasible_force_floors_are_unchanged(test, device, _project_exact)
    _check_zero_force_least_transfer(test, device, _project_exact)
    _check_multiple_plateaus_choose_least_net_flow(test, device, _project_exact)


def test_exact_zero_force_retains_double_allocation(test, device):
    """Keep canonical zero-force transfer before recording float publication."""
    reference = np.array(
        [
            -0.0003521330654621124,
            2.546155286609064e-9,
            -8.458433740088367e-7,
            -0.0008104796288534999,
            4.490648741750647e-9,
        ]
    )
    lengths = np.array(
        [0.01596725545823574, 0.015850551426410675, 0.031485892832279205, 0.028883151710033417, 0.010692404583096504]
    )
    damping = np.array(
        [
            -0.17254510521888733,
            -0.00015818909741938114,
            -0.000738076283596456,
            -0.00014784227823838592,
            -0.00011458260996732861,
        ]
    )
    caps = np.array([1.0323879718780518, 1.213707447052002, 1.0384721755981445, 1.202047348022461])
    # Only the final span exceeds its zero-force cap. Its monotone inverse
    # fixes the unique minimum transfer; no active-face oracle is required.
    lower, upper = 0.0, float(reference[-1])
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        rest = lengths[-1] - middle
        strain = middle / rest
        ea = 500.0 * (1.0 + 4.5 * (1.0 + math.tanh((strain - float(np.float32(0.01))) / float(np.float32(0.003)))))
        if middle / (rest / ea) + damping[-1] <= 0.0:
            lower = middle
        else:
            upper = middle
    expected = reference.copy()
    expected[-1] = lower
    expected[-2] += reference[-1] - lower
    for reflect in (False, True):
        order = slice(None, None, -1) if reflect else slice(None)
        state, inputs, count = _fixture(
            device, reference[order], lengths[order], caps[order], damping[order], ratio=10.0
        )
        inputs[2], inputs[4], inputs[5], inputs[6], inputs[8] = 500.0, 0.01, 0.003, 1.0e-6, 1.0e-5
        _launch(_project_exact, dim=count, inputs=inputs, device=device)
        test.assertGreater(state.status.numpy()[0], 0)
        corrected = state.corrected.numpy()
        expected_flow = np.cumsum(reference[order] - expected[order])[:-1]
        actual_flow = np.cumsum(reference[order] - corrected)[:-1]
        np.testing.assert_allclose(actual_flow, expected_flow, rtol=0.0, atol=1.0e-17)
        np.testing.assert_array_equal(state.exact_tension.numpy(), np.zeros(5))
        np.testing.assert_array_equal(state.output.numpy(), corrected.astype(np.float32))
        test.assertGreater(state.storage_error.numpy()[0, 1], 1.0e-11)
        test.assertLess(state.residual.numpy()[0], 1.0e-13)
        test.assertEqual(state.valid.numpy()[0], 0)


def test_exact_near_slack_positive_floor_preserves_taut_endpoints(test, device):
    """Retain tiny taut extensions when plateau allocation moves much larger slack."""
    # Frozen Toy3 VBD frame 56, cable B (packed row 4), prior validated fixture
    # with A:R0 down 5 mm. Negative damping nearly cancels elastic force in the
    # first four spans; the last span has a strictly positive force plateau.
    # Reconstructing taut extensions from differences of large prefix flows
    # rejected this state despite feasible rest bounds and exact conservation.
    reference = [
        -1.4808028936386108e-7,
        -3.2372682312598045e-10,
        -0.0002795840264298022,
        -7.786706390788822e-9,
        -8.195613077077724e-10,
    ]
    lengths = [
        0.02019013836979866,
        0.016829736530780792,
        0.029587481170892715,
        0.027207329869270325,
        0.0010380364255979657,
    ]
    damping = [
        -0.003086121752858162,
        -1.8243259773953469e-6,
        -0.00010252638458041474,
        -5.8755256759468466e-5,
        3.5606708553848065e-11,
    ]
    caps = [1.0202298164367676, 1.2079423666000366, 1.0258269309997559, 1.1533946990966797]
    state, inputs, count = _fixture(device, reference, lengths, caps, damping, ratio=10.0)
    inputs[2], inputs[4], inputs[5], inputs[6], inputs[8] = 500.0, 0.01, 0.003, 1.0e-6, 1.0e-5
    original = state.reference.numpy().copy()
    for _ in range(2):
        # Exercise both the cold solve and the accepted face cache without
        # changing the original reference or allowing a weaker certificate.
        _launch(_project_exact, dim=count, inputs=inputs, device=device)
        test.assertGreater(
            state.status.numpy()[0],
            0,
            f"Near-slack exact projection rejected: residual={state.residual.numpy()[0]}",
        )
        test.assertLessEqual(state.residual.numpy()[0], 1.0e-5)
        _assert_certificate(
            test,
            state,
            5,
            ratio=10.0,
            ea_low=500.0,
            transition_strain=0.01,
            transition_width=0.003,
            tolerance=1.0e-5,
            allow_slack=True,
            exact=True,
        )
        np.testing.assert_array_equal(state.reference.numpy(), original)
        np.testing.assert_array_equal(state.faces.numpy(), [-1, -1, -1, -1])
        test.assertTrue(np.all(state.corrected.numpy()[:4] > 0.0))
        test.assertLess(state.corrected.numpy()[-1], 0.0)
        test.assertEqual(state.exact_tension.numpy()[-1], float(np.float32(damping[-1])))


def test_exact_isolated_sticking_span(test, device):
    """Keep an isolated middle span exact between two independently slipping blocks."""
    for middle_reference, middle_damping in ((-0.03, 4.0), (1.0 / 26.0, 0.0), (1.0 / 11.0, -6.0)):
        reference = [-0.01, 0.0, middle_reference, 0.0, -0.02]
        damping = [5.0, 0.0, middle_damping, 0.0, 5.0]
        state, inputs, count = _fixture(device, reference, np.ones(5), np.full(4, 1.2), damping)
        _launch(_project_exact, dim=count, inputs=inputs, device=device)
        _assert_certificate(test, state, 5, allow_slack=True, exact=True)
        np.testing.assert_allclose(state.output.numpy(), [-0.05, 0.04, middle_reference, 0.04, -0.06], rtol=3.0e-6)
        np.testing.assert_array_equal(state.faces.numpy(), [1, 0, 0, -1])
        test.assertEqual(state.output.numpy()[2], state.reference.numpy()[2])


def test_exact_graph_force_floor_transitions(test, device):
    _check_graph_force_floor_transitions(test, device, _project_exact)


def test_exact_captured_slip_and_rounding(test, device):
    """Resolve a missed slipping face and conserve rounded sticking blocks."""
    cases = (
        (
            [0.00023899783263914287, 1.4968591131037101e-6, 5.8072666433872655e-5, 7.257749530253932e-5],
            [0.05877023935317993, 0.019199756905436516, 0.016977934166789055, 0.0257235337048769],
            [1.3854022026062012, -1.7581076622009277, 0.086367167532444, 0.4040565490722656],
            [1.0307284593582153, 1.1079387664794922, 1.0117360353469849],
            [0.00012053576772224856, 0.00010778308148672987, 6.191170963147683e-5, 8.09142946482033e-5],
            [1, 1, 1],
        ),
        (
            [0.0, -1.6282758474872594e-10, -2.3213356570295218e-8, 1.0198323435739454e-10, -2.0436777270482764e-12],
            [
                0.05000000447034836,
                0.015850551426410675,
                0.017233695834875107,
                0.02647257223725319,
                0.010692404583096504,
            ],
            [
                1.5186168988989834e-10,
                1.6955226217712886e-14,
                1.29458703668206e-6,
                -3.474525271940365e-7,
                5.503952706931159e-7,
            ],
            [1.0323779582977295, 1.0203990936279297, 1.0328991413116455, 1.0499522686004639],
            [1.21486792e-10, 3.97645238e-11, -2.35374355e-8, 8.56316583e-11, 1.43078984e-11],
            [-1, -1, 0, 1],
        ),
    )
    for reference, lengths, damping, caps, expected, faces in cases:
        for reflect in (False, True):
            order = slice(None, None, -1) if reflect else slice(None)
            state, inputs, count = _fixture(
                device,
                np.asarray(reference)[order],
                np.asarray(lengths)[order],
                np.asarray(caps)[order],
                np.asarray(damping)[order],
                ratio=10.0,
            )
            inputs[2], inputs[4], inputs[5], inputs[8] = 500.0, 0.01, 0.003, 1.0e-5
            _launch(_project_exact, dim=count, inputs=inputs, device=device)
            _assert_certificate(
                test,
                state,
                len(reference),
                ratio=10.0,
                ea_low=500.0,
                transition_strain=0.01,
                transition_width=0.003,
                tolerance=1.0e-5,
                allow_slack=True,
                exact=True,
            )
            np.testing.assert_allclose(state.output.numpy(), np.asarray(expected)[order], rtol=1.0e-5, atol=1.0e-17)
            expected_faces = -np.asarray(faces)[::-1] if reflect else faces
            np.testing.assert_array_equal(state.faces.numpy(), expected_faces)
            original = state.reference.numpy().copy()
            _launch(_project_exact, dim=count, inputs=inputs, device=device)
            np.testing.assert_array_equal(state.reference.numpy(), original)


class TestTendonMaterialNonlinear(unittest.TestCase):
    pass


for _test in (
    test_exact_releases_incompatible_floor_face,
    test_exact_known_sigmoid_solutions,
    test_exact_rounding_qualified_force_tolerance,
    test_exact_publication_rejects_bad_math_and_non_nearest_storage,
    test_exact_publication_preserves_hard_rest_bound,
    test_exact_repairs_overbound_rolling_trial,
    test_exact_rejects_invalid_geometry_and_infeasible_inventory,
    test_exact_cache_revalidates_inputs_and_preserves_failure,
    test_exact_storage_conservation_error_is_explicit,
    test_inverse_hint_and_reused_tangent,
    test_inverse_progress_across_sigmoid_knee,
    test_exact_equal_force_knee_inverse,
    test_exact_force_floor_regimes,
    test_exact_zero_force_retains_double_allocation,
    test_exact_near_slack_positive_floor_preserves_taut_endpoints,
    test_exact_isolated_sticking_span,
    test_exact_graph_force_floor_transitions,
    test_exact_captured_slip_and_rounding,
):
    add_function_test(TestTendonMaterialNonlinear, _test.__name__, _test, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
