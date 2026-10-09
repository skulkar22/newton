# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the opt-in direct material solver through normal solver entry points."""

import unittest
from itertools import product

import numpy as np
import warp as wp

import newton
from newton._src.solvers.tendon_kernels import solve_tendon_material
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _add_chain(builder, body, span_count, *, attachments=(), rest_lengths=None, height=0.0):
    """Add a straight route with optional internal attachment boundaries."""
    builder.add_tendon()
    for index in range(span_count + 1):
        kind = newton.TendonLinkType.PINHOLE
        if index in (0, span_count) or index in attachments:
            kind = newton.TendonLinkType.ATTACHMENT
        builder.add_tendon_link(
            body=body,
            link_type=int(kind),
            offset=(float(index), 0.0, height),
            compliance=1.0e-3,
            rest_length=0.99 if rest_lengths is None or index == 0 else rest_lengths[index - 1],
            mu=0.1,
        )


def _chain_model(device, span_count=2, *, attachments=(), rest_lengths=None, requires_grad=False):
    """Build a stationary chain whose material redistribution is directly observable."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(mass=0.0, is_kinematic=True)
    _add_chain(builder, body, span_count, attachments=attachments, rest_lengths=rest_lengths)
    builder.color()
    return builder.finalize(device=device, requires_grad=requires_grad)


def _step(model, solver, state_in=None):
    """Run one ordinary XPBD step and return the accepted state."""
    if state_in is None:
        state_in = model.state()
    state_out = model.state()
    solver.step(state_in, state_out, model.control(), None, 1.0 / 240.0)
    return state_out


def test_direct_allocates_nonlinear_material(test, device):
    """Allocate the experimental nonlinear projection only in opt-in direct mode."""
    model = _chain_model(device)
    solver = newton.solvers.SolverXPBD(model, tendon_material_direct=True, tendon_sigmoid_ea_low=2000.0)
    test.assertTrue(solver._tendon_material_state.nonlinear_enabled)
    legacy = newton.solvers.SolverXPBD(model, tendon_sigmoid_ea_low=2000.0)
    test.assertFalse(legacy._tendon_material_state.enabled)


def test_direct_rejects_gradient_model(test, device):
    """Reject differentiable models before allocating the direct solve workspace."""
    model = _chain_model(device, requires_grad=True)
    with test.assertRaisesRegex(ValueError, "differentiable simulation"):
        newton.solvers.SolverXPBD(model, tendon_material_direct=True)


def test_direct_rejects_invalid_compliance(test, device):
    """Reject zero, negative, nonfinite, and unsupported tiny compliance values."""
    model = _chain_model(device)
    for compliance in (0.0, -1.0, np.nan, np.inf, 1.0e-26):
        with test.subTest(compliance=compliance):
            model.tendon_seg_compliance.assign(np.array([compliance, 1.0e-3], dtype=np.float32))
            with test.assertRaisesRegex(ValueError, "finite segment compliance"):
                newton.solvers.SolverXPBD(model, tendon_material_direct=True)


def test_direct_rejects_oversized_component(test, device):
    """Reject one 33-span material component without rejecting the legacy solver."""
    model = _chain_model(device, 33)
    with test.assertRaisesRegex(ValueError, "at most 32 authored spans"):
        newton.solvers.SolverXPBD(model, tendon_material_direct=True)
    legacy = newton.solvers.SolverXPBD(model)
    test.assertFalse(legacy._tendon_material_state.enabled)


def test_direct_accepts_capacity_boundary(test, device):
    """Solve a full 32-span component through the ordinary XPBD step."""
    model = _chain_model(device, 32)
    solver = newton.solvers.SolverXPBD(model, iterations=2, tendon_material_direct=True)
    initial = solver.tendon_seg_rest_length.numpy().astype(np.float64).sum()
    _step(model, solver)
    solver.check_tendon_material()
    test.assertEqual(int(solver._tendon_material_state.count.numpy()[0]), 32)
    rest = solver.tendon_seg_rest_length.numpy()
    test.assertTrue(np.isfinite(rest).all())
    test.assertAlmostEqual(float(rest.astype(np.float64).sum()), float(initial), delta=2.0e-6)


def test_direct_attachment_components_are_independent(test, device):
    """Allow more than 32 total spans and preserve each attachment-bounded inventory."""
    rest_lengths = np.concatenate((np.tile([0.97, 1.01], 8), [0.98], np.tile([0.95, 1.01], 8), [0.96]))
    model = _chain_model(device, 34, attachments=(17,), rest_lengths=rest_lengths)
    solver = newton.solvers.SolverXPBD(model, iterations=2, tendon_material_direct=True)
    initial = solver.tendon_seg_rest_length.numpy().astype(np.float64)
    _step(model, solver)
    solver.check_tendon_material()
    workspace = solver._tendon_material_state
    np.testing.assert_array_equal(workspace.count.numpy()[[0, 17]], [17, 17])
    np.testing.assert_array_equal(workspace.ids.numpy(), np.arange(34))
    test.assertEqual(workspace.initial.size, model.tendon_segment_count)
    rest = solver.tendon_seg_rest_length.numpy().astype(np.float64)
    for start, stop in ((0, 17), (17, 34)):
        expected_rest = float(initial[start:stop].mean())
        np.testing.assert_allclose(rest[start:stop], expected_rest, rtol=0.0, atol=2.0e-7)
        test.assertAlmostEqual(float(rest[start:stop].sum()), float(initial[start:stop].sum()), delta=2.0e-6)


def test_direct_is_disabled_by_default(test, device):
    """Keep default XPBD and VBD on the existing material kernel without direct storage."""
    model = _chain_model(device)
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        with test.subTest(solver=solver_type.__name__):
            solver = solver_type(model, iterations=1)
            test.assertFalse(solver._tendon_material_state.enabled)
            test.assertIs(solver._tendon_material_kernel, solve_tendon_material)
            test.assertIsNone(solver._tendon_material_state.initial)
            solver.check_tendon_material()


def test_direct_handles_model_without_tendons(test, device):
    """Allow opt-in direct mode on a model with no tendon components."""
    for has_body in (False, True):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        if has_body:
            builder.add_body(mass=1.0, inertia=wp.diag(wp.vec3(1.0)))
        builder.color()
        model = builder.finalize(device=device)
        for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
            with test.subTest(solver=solver_type.__name__, has_body=has_body):
                solver = solver_type(model, iterations=2, tendon_material_direct=True)
                state = model.state()
                expected = state.body_q.numpy().copy() if has_body else None
                solver.step(state, model.state(), model.control(), None, 1.0 / 240.0)
                solver.check_tendon_material()
                test.assertEqual(solver._tendon_material_state.failure.size, 0)
                if has_body:
                    np.testing.assert_array_equal(state.body_q.numpy(), expected)


def test_direct_rejects_gradient_state_override(test, device):
    """Reject gradient-enabled input or output states even on a non-gradient model."""
    model = _chain_model(device)
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        solver = solver_type(model, tendon_material_direct=True)
        for input_grad, output_grad in ((True, False), (False, True)):
            with test.subTest(solver=solver_type.__name__, input_grad=input_grad, output_grad=output_grad):
                state_in = model.state(requires_grad=input_grad)
                state_out = model.state(requires_grad=output_grad)
                with test.assertRaisesRegex(ValueError, "differentiable simulation"):
                    solver.step(state_in, state_out, model.control(), None, 1.0 / 240.0)


def test_direct_rejects_runtime_mode_change(test, device):
    """Require reconstruction when either enabling or disabling direct mode after initialization."""
    model = _chain_model(device)
    for enabled in (False, True):
        with test.subTest(initially_enabled=enabled):
            solver = newton.solvers.SolverXPBD(model, tendon_material_direct=enabled)
            solver.tendon_material_direct = not enabled
            with test.assertRaisesRegex(ValueError, "Reconstruct SolverXPBD.*tendon_material_direct"):
                _step(model, solver)
            test.assertEqual(solver._tendon_material_state.enabled, enabled)


def test_direct_rejects_runtime_nonlinear_material(test, device):
    """Reject enabling nonlinear material after constructing a linear direct solver."""
    model = _chain_model(device)
    solver = newton.solvers.SolverXPBD(model, tendon_material_direct=True)
    rest = solver.tendon_seg_rest_length.numpy().copy()
    solver.tendon_sigmoid_ea_low = 2000.0
    with test.assertRaisesRegex(ValueError, "Reconstruct.*material law"):
        _step(model, solver)
    np.testing.assert_array_equal(solver.tendon_seg_rest_length.numpy(), rest)


def test_direct_retains_short_positive_pinhole_spans(test, device):
    """Keep a nonzero sub-10-micron pinhole span in the connected material solve."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(mass=0.0, is_kinematic=True)
    builder.add_tendon()
    for index, x in enumerate((0.0, 1.0, 1.000005, 2.0)):
        builder.add_tendon_link(
            body=body,
            link_type=newton.TendonLinkType.ATTACHMENT if index in (0, 3) else newton.TendonLinkType.PINHOLE,
            offset=(x, 0.0, 0.0),
            compliance=1.0e-8 if index == 2 else 1.0e-3,
            rest_length=4.9e-6 if index == 2 else 0.99,
            mu=0.1,
        )
    builder.color()
    model = builder.finalize(device=device)
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        with test.subTest(solver=solver_type.__name__):
            solver = solver_type(model, iterations=2, tendon_material_direct=True)
            length = solver.tendon_seg_length.numpy().astype(float)
            initial_rest = solver.tendon_seg_rest_length.numpy().astype(float)
            compliance = model.tendon_seg_compliance.numpy().astype(float)
            # The collinear pinholes have no wrap, so all three taut spans have
            # the same tension and their sum of elastic extensions is fixed.
            expected_tension = float(np.sum(length - initial_rest) / np.sum(compliance))
            expected_rest = length - compliance * expected_tension
            test.assertGreater(expected_rest[1], 1.0e-6)
            _step(model, solver)
            solver.check_tendon_material()
            test.assertEqual(int(solver._tendon_material_state.count.numpy()[0]), 3)
            np.testing.assert_array_equal(solver._tendon_material_state.ids.numpy(), [0, 1, 2])
            np.testing.assert_array_equal(solver.tendon_link_cone_seg_l.numpy()[1:3], [0, 1])
            np.testing.assert_array_equal(solver.tendon_link_cone_seg_r.numpy()[1:3], [1, 2])
            rest = solver.tendon_seg_rest_length.numpy().astype(float)
            np.testing.assert_allclose(rest, expected_rest, atol=2.0e-7, rtol=0.0)
            test.assertAlmostEqual(float(rest.sum()), float(initial_rest.sum()), delta=2.0e-7)
            np.testing.assert_allclose((length - rest) / compliance, expected_tension, atol=2.0e-3, rtol=0.0)


def test_direct_vbd_rejects_invalid_raw_runtime_compliance(test, device):
    """Reject invalid authored coefficients before VBD's active-route floor can hide them."""
    for merged in (False, True):
        for invalid, rebind in product((np.nan, -1.0, 0.0, 1.0e-26), (False, True)):
            with test.subTest(merged=merged, invalid=invalid, rebind=rebind):
                if not merged:
                    model = _chain_model(device)
                else:
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    base = builder.add_body(mass=0.0, is_kinematic=True)
                    roller = builder.add_body(xform=wp.transform(p=(0.25, 0.0, 0.0)), mass=0.0, is_kinematic=True)
                    builder.add_tendon()
                    for index, z in enumerate((-0.5, 0.0, 0.5)):
                        builder.add_tendon_link(
                            body=roller if index == 1 else base,
                            link_type=newton.TendonLinkType.ROLLING if index == 1 else newton.TendonLinkType.ATTACHMENT,
                            offset=(0.0, 0.0, z),
                            axis=(0.0, 1.0, 0.0),
                            radius=0.1 if index == 1 else 0.0,
                            orientation=1,
                            dynamic=index == 1,
                            compliance=1.0e-3,
                            rest_length=0.45,
                            mu=0.1,
                        )
                    builder.color()
                    model = builder.finalize(device=device)
                solver = newton.solvers.SolverVBD(model, iterations=2, tendon_material_direct=True)
                if merged:
                    test.assertFalse(bool(solver.tendon_link_active.numpy()[1]))
                    test.assertEqual(int(solver.tendon_seg_active.numpy()[1]), 0)
                compliance = model.tendon_seg_compliance.numpy()
                # In the bypass case this slot is inactive, but contributes to
                # the merged span and must still be checked before flooring.
                compliance[1] = invalid
                if rebind:
                    model.tendon_seg_compliance = wp.array(compliance, dtype=float, device=device)
                else:
                    model.tendon_seg_compliance.assign(compliance)
                _step(model, solver)
                with test.assertRaisesRegex(RuntimeError, r"BAD_INPUT.*tendon 0"):
                    solver.check_tendon_material()


def _two_tendon_model(device):
    """Build a straight route and a separate right-angle pinhole route."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(mass=0.0, is_kinematic=True)
    _add_chain(builder, body, 2, height=2.0)
    builder.add_tendon()
    for index, point in enumerate(((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))):
        builder.add_tendon_link(
            body=body,
            link_type=int(newton.TendonLinkType.PINHOLE if index == 1 else newton.TendonLinkType.ATTACHMENT),
            offset=point,
            compliance=1.0e-3,
            rest_length=0.99,
            mu=0.1,
        )
    return builder.finalize(device=device)


def test_direct_runtime_failure_remains_latched(test, device):
    """Report an unsupported runtime cap ratio and retain the failure after input repair."""
    model = _two_tendon_model(device)
    solver = newton.solvers.SolverXPBD(model, iterations=2, tendon_material_direct=True)
    state = _step(model, solver)
    solver.check_tendon_material()
    mu = model.tendon_link_mu.numpy()
    mu[4] = 1.0
    model.tendon_link_mu.assign(mu)
    state = _step(model, solver, state)
    test.assertGreater(float(solver.tendon_link_cap_ratio.numpy()[4]), 4.0)
    with test.assertRaisesRegex(RuntimeError, r"BAD_INPUT.*tendon 1.*segment 2"):
        solver.check_tendon_material()
    failure = solver._tendon_material_state.failure.numpy().copy()
    test.assertEqual(int(failure[0]), 0)
    test.assertLess(int(failure[1]), 0)
    mu[4] = 0.1
    model.tendon_link_mu.assign(mu)
    _step(model, solver, state)
    np.testing.assert_array_equal(solver._tendon_material_state.failure.numpy(), failure)
    with test.assertRaisesRegex(RuntimeError, "Discard this step/frame/batch"):
        solver.check_tendon_material()


def test_direct_reports_failure_after_graph_replay(test, device, *, sigmoid=False):
    """Capture the direct path and observe later parameter failures outside graph replay."""
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")
    model = _two_tendon_model(device)
    solver = newton.solvers.SolverXPBD(
        model, iterations=2, tendon_material_direct=True, tendon_sigmoid_ea_low=500.0 if sigmoid else 0.0
    )
    state_in, state_out = model.state(), model.state()
    control = model.control()
    solver.step(state_in, state_out, control, None, 1.0 / 240.0)
    solver.check_tendon_material()
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_in, state_out, control, None, 1.0 / 240.0)
    wp.capture_launch(capture.graph)
    solver.check_tendon_material()
    mu = model.tendon_link_mu.numpy()
    mu[4] = 1.0
    model.tendon_link_mu.assign(mu)
    wp.capture_launch(capture.graph)
    with test.assertRaisesRegex(RuntimeError, r"BAD_INPUT.*tendon 1.*segment 2"):
        solver.check_tendon_material()


def test_direct_dynamic_route_repacking(test, device, *, solver_type=newton.solvers.SolverXPBD, sigmoid=False):
    """Repack repeatedly changing active spans without touching adjacent components or tendons."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = builder.add_body(mass=0.0, is_kinematic=True)
    roller = builder.add_body(xform=wp.transform(p=(0.25, 0.0, 0.0)), mass=0.0, is_kinematic=True)
    builder.add_tendon()
    points = ((0.0, 0.0, -0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (0.4, 0.0, 1.0), (0.8, 0.0, 1.0))
    rests = (-1.0, 0.45, 0.45, 0.5, 0.39, 0.38)
    kinds = (
        newton.TendonLinkType.ATTACHMENT,
        newton.TendonLinkType.ROLLING,
        newton.TendonLinkType.PINHOLE,
        newton.TendonLinkType.ATTACHMENT,
        newton.TendonLinkType.PINHOLE,
        newton.TendonLinkType.ATTACHMENT,
    )
    for index, (point, rest, kind) in enumerate(zip(points, rests, kinds, strict=True)):
        builder.add_tendon_link(
            body=roller if index == 1 else base,
            link_type=int(kind),
            offset=point,
            axis=(0.0, 1.0, 0.0),
            radius=0.1 if index == 1 else 0.0,
            orientation=1,
            dynamic=index == 1,
            compliance=1.0e-3,
            rest_length=rest,
            mu=0.1,
        )
    _add_chain(builder, base, 2, height=3.0, rest_lengths=[0.97, 0.99])
    builder.color()
    model = builder.finalize(device=device)
    solver = solver_type(
        model, iterations=3, tendon_material_direct=True, tendon_sigmoid_ea_low=500.0 if sigmoid else 0.0
    )
    state = model.state()
    unaffected_rest = None
    for x, active in (
        (0.25, False),
        (0.05, True),
        (0.05, True),
        (0.25, False),
        (0.25, False),
        (0.05, True),
        (0.25, False),
    ):
        poses = state.body_q.numpy()
        poses[roller, 0] = x
        state.body_q.assign(poses)
        state = _step(model, solver, state)
        solver.check_tendon_material()
        test.assertEqual(bool(solver.tendon_link_active.numpy()[1]), active)
        workspace = solver._tendon_material_state
        ids = workspace.ids.numpy()
        count = int(workspace.count.numpy()[0])
        np.testing.assert_array_equal(ids[:count], [0, 1, 2] if active else [0, 2])
        np.testing.assert_array_equal(ids[3:], [3, 4, 5, 6])
        np.testing.assert_array_equal(workspace.count.numpy()[[3, 5]], [2, 2])
        rest = solver.tendon_seg_rest_length.numpy()
        test.assertTrue(np.isfinite(rest).all())
        test.assertTrue(np.all(rest[solver.tendon_seg_active.numpy() != 0] >= 1.0e-6))
        tension = np.maximum(
            (solver.tendon_seg_length.numpy() - rest) / solver.tendon_seg_active_compliance.numpy(), 0.0
        )
        left = solver.tendon_link_cone_seg_l.numpy()
        right = solver.tendon_link_cone_seg_r.numpy()
        caps = solver.tendon_link_cap_ratio.numpy()
        for link in np.flatnonzero((left >= 0) & (right >= 0)):
            test.assertLessEqual(float(tension[left[link]] - caps[link] * tension[right[link]]), 2.0e-3)
            test.assertLessEqual(float(tension[right[link]] - caps[link] * tension[left[link]]), 2.0e-3)
        if unaffected_rest is None:
            unaffected_rest = rest[3:].copy()
        else:
            np.testing.assert_allclose(rest[3:], unaffected_rest, rtol=0.0, atol=2.0e-7)


class TestTendonMaterialIntegration(unittest.TestCase):
    """Validate public opt-in behavior and packed route integration."""


devices = get_test_devices()
for test_function in (
    test_direct_allocates_nonlinear_material,
    test_direct_rejects_gradient_model,
    test_direct_rejects_invalid_compliance,
    test_direct_rejects_oversized_component,
    test_direct_accepts_capacity_boundary,
    test_direct_attachment_components_are_independent,
    test_direct_is_disabled_by_default,
    test_direct_handles_model_without_tendons,
    test_direct_rejects_gradient_state_override,
    test_direct_rejects_runtime_mode_change,
    test_direct_rejects_runtime_nonlinear_material,
    test_direct_retains_short_positive_pinhole_spans,
    test_direct_vbd_rejects_invalid_raw_runtime_compliance,
    test_direct_runtime_failure_remains_latched,
    test_direct_reports_failure_after_graph_replay,
    test_direct_dynamic_route_repacking,
):
    add_function_test(
        TestTendonMaterialIntegration,
        test_function.__name__,
        test_function,
        devices=devices,
        check_output=test_function
        not in (
            test_direct_runtime_failure_remains_latched,
            test_direct_reports_failure_after_graph_replay,
            test_direct_vbd_rejects_invalid_raw_runtime_compliance,
        ),
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
