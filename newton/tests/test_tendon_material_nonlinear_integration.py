# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the nonlinear experiment through both body solvers."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.tendon_kernels import solve_tendon_material_direct
from newton._src.solvers.tendon_material_cooperative_kernels import solve_tendon_material_cooperative
from newton._src.solvers.vbd.tendon_kernels import _direct_tendon_span_tension
from newton.tests.test_tendon_material_integration import (
    _chain_model,
)
from newton.tests.test_tendon_material_integration import (
    test_direct_dynamic_route_repacking as _check_route_repacking,
)
from newton.tests.test_tendon_material_integration import (
    test_direct_reports_failure_after_graph_replay as _check_graph_failure,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _trial_tension(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    link_body: wp.array[int],
    link_type: wp.array[int],
    link_offset: wp.array[wp.vec3],
    link_axis: wp.array[wp.vec3],
    rest: wp.array[float],
    att_l: wp.array[wp.vec3],
    att_r: wp.array[wp.vec3],
    compliance: wp.array[float],
    damping: wp.array[float],
    left: wp.array[int],
    right: wp.array[int],
    result: wp.array[float],
):
    seg = wp.tid()
    result[seg] = _direct_tendon_span_tension(
        seg,
        1.0 / 240.0,
        body_q,
        body_q,
        body_com,
        link_body,
        link_type,
        link_offset,
        link_axis,
        rest,
        att_l,
        att_r,
        compliance,
        damping,
        left,
        right,
        500.0,
        10.0,
        0.01,
        0.003,
        damping,  # Unused ALM arrays: this test exercises the constitutive path.
        damping,
        0,
    )


def test_nonlinear_solver_entry(test, device):
    """Conserve material through ordinary XPBD and VBD nonlinear direct steps."""
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        for rest_lengths in ([0.97, 1.01], [1.01, 1.02], [0.99, 0.99]):
            with test.subTest(solver=solver_type.__name__, rest=rest_lengths):
                model = _chain_model(device, rest_lengths=rest_lengths)
                solver = solver_type(
                    model,
                    iterations=2,
                    tendon_material_direct=True,
                    tendon_sigmoid_ea_low=500.0,
                    tendon_sigmoid_ea_ratio=10.0,
                    tendon_sigmoid_transition_strain=0.01,
                    tendon_sigmoid_transition_width=0.003,
                )
                state, next_state = model.state(), model.state()
                expected = solve_tendon_material_cooperative if device.is_cuda else solve_tendon_material_direct
                test.assertIs(solver._tendon_material_kernel, expected)
                test.assertEqual(solver._tendon_material_lanes, 32 if device.is_cuda else 1)
                initial_rest = solver.tendon_seg_rest_length.numpy().astype(float)
                solver.step(state, next_state, model.control(), None, 1.0 / 240.0)
                solver.check_tendon_material()
                rest = solver.tendon_seg_rest_length.numpy().astype(float)
                test.assertTrue(np.isfinite(rest).all())
                test.assertGreaterEqual(rest.min(), 1.0e-6)
                test.assertAlmostEqual(rest.sum(), initial_rest.sum(), delta=5.0e-7)
                test.assertGreater(int(solver._tendon_material_state.status.numpy()[0]), 0)


def test_nonlinear_short_slack_span(test, device):
    """Keep valid slack material even when geometric length is below the rest floor."""
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        for length in (0.0, 5.0e-7):
            with test.subTest(solver=solver_type.__name__, length=length):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                body = builder.add_body(mass=0.0, is_kinematic=True)
                builder.add_tendon()
                for x in (0.0, length):
                    builder.add_tendon_link(
                        body=body,
                        link_type=newton.TendonLinkType.ATTACHMENT,
                        offset=(x, 0.0, 0.0),
                        rest_length=2.0e-6,
                        compliance=1.0e-3,
                    )
                builder.color()
                model = builder.finalize(device=device)
                solver = solver_type(
                    model,
                    iterations=1,
                    tendon_material_direct=True,
                    tendon_sigmoid_ea_low=500.0,
                    tendon_sigmoid_ea_ratio=10.0,
                    tendon_sigmoid_transition_strain=0.01,
                    tendon_sigmoid_transition_width=0.003,
                )
                original_rest = solver.tendon_seg_rest_length.numpy().copy()
                state, next_state = model.state(), model.state()
                control = model.control()
                for _ in range(3):
                    solver.step(state, next_state, control, None, 1.0e-3)
                    solver.check_tendon_material()
                    state, next_state = next_state, state
                    np.testing.assert_array_equal(solver.tendon_seg_rest_length.numpy(), original_rest)
                    np.testing.assert_array_equal(
                        solver.tendon_seg_length.numpy(), np.array([length], dtype=np.float32)
                    )
                    np.testing.assert_array_equal(solver.tendon_seg_material_tension.numpy(), [0.0])
                    np.testing.assert_array_equal(solver.tendon_seg_damping_tension.numpy(), [0.0])
                    np.testing.assert_array_equal(state.body_q.numpy(), model.body_q.numpy())


def test_vbd_adjacent_trial_uses_sigmoid(test, device):
    """Evaluate VBD direct friction limiting with sigmoid rather than linear tension."""
    model = _chain_model(device, rest_lengths=[0.99, 0.995])
    solver = newton.solvers.SolverVBD(model, iterations=1)
    result = wp.zeros(model.tendon_segment_count, device=device)
    wp.launch(
        _trial_tension,
        dim=model.tendon_segment_count,
        device=device,
        inputs=[
            model.body_q,
            model.body_com,
            model.tendon_link_body,
            model.tendon_link_type,
            model.tendon_link_offset,
            model.tendon_link_axis,
            solver.tendon_seg_rest_length,
            solver.tendon_seg_attachment_l_local,
            solver.tendon_seg_attachment_r_local,
            solver.tendon_seg_active_compliance,
            solver.tendon_seg_active_damping,
            solver.tendon_seg_active_link_l,
            solver.tendon_seg_active_link_r,
            result,
        ],
    )
    length = solver.tendon_seg_length.numpy().astype(float)
    rest = solver.tendon_seg_rest_length.numpy().astype(float)
    strain = np.maximum(length - rest, 0.0) / rest
    expected = 500.0 * (1.0 + 4.5 * (1.0 + np.tanh((strain - 0.01) / 0.003))) * strain
    np.testing.assert_allclose(result.numpy(), expected, rtol=2.0e-6, atol=1.0e-6)
    test.assertGreater(np.max(np.abs(expected - np.maximum(length - rest, 0.0) / 1.0e-3)), 1.0)


def test_nonlinear_curved_finite_friction(test, device):
    """Preserve inventory and reach the true sigmoid capstan bound on a fixed pulley."""
    radius, mu = 0.1, 0.2
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        with test.subTest(solver=solver_type.__name__):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            bodies = [
                builder.add_body(xform=wp.transform(p=(x, 0.0, 0.0)), mass=0.0, is_kinematic=True)
                for x in (-0.4, 0.0, 0.4)
            ]
            builder.add_tendon()
            for index, body in enumerate(bodies):
                rolling = index == 1
                builder.add_tendon_link(
                    body=body,
                    link_type=newton.TendonLinkType.ROLLING if rolling else newton.TendonLinkType.ATTACHMENT,
                    radius=radius if rolling else 0.0,
                    orientation=1,
                    axis=(0.0, 0.0, 1.0),
                    mu=mu,
                    compliance=1.0e-3,
                    damping=0.0,
                    rest_length=-1.0,
                )
            builder.color()
            model = builder.finalize(device=device)
            solver = solver_type(
                model,
                iterations=2,
                tendon_material_direct=True,
                tendon_sigmoid_ea_low=500.0,
                tendon_sigmoid_ea_ratio=10.0,
                tendon_sigmoid_transition_strain=0.01,
                tendon_sigmoid_transition_width=0.003,
            )
            length = solver.tendon_seg_length.numpy().astype(float)
            initial_rest = (length / np.array([1.015, 1.005])).astype(np.float32)
            solver.tendon_seg_rest_length.assign(initial_rest)
            expected_wrap = 2.0 * math.asin(radius / 0.4)
            expected_cap = math.exp(mu * expected_wrap)
            initial_inventory = float(initial_rest.astype(float).sum()) + radius * expected_wrap
            initial_strain = (length - initial_rest) / initial_rest
            initial_tension = 500.0 * (1.0 + 4.5 * (1.0 + np.tanh((initial_strain - 0.01) / 0.003))) * initial_strain
            test.assertGreater(initial_tension[0] / initial_tension[1], expected_cap)
            test.assertGreater(expected_wrap, 0.0)
            test.assertGreater(expected_cap, 1.0)
            state, next_state = model.state(), model.state()
            control = model.control()
            for step in range(3):
                solver.step(state, next_state, control, None, 1.0 / 240.0)
                solver.check_tendon_material()
                state, next_state = next_state, state
                length = solver.tendon_seg_length.numpy().astype(float)
                rest = solver.tendon_seg_rest_length.numpy().astype(float)
                strain = np.maximum(length - rest, 0.0) / rest
                tension = 500.0 * (1.0 + 4.5 * (1.0 + np.tanh((strain - 0.01) / 0.003))) * strain
                attachment_l = solver.tendon_seg_attachment_r.numpy()[0].astype(float)
                attachment_r = solver.tendon_seg_attachment_l.numpy()[1].astype(float)
                wrap = math.atan2(float(np.cross(attachment_l, attachment_r)[2]), float(attachment_l @ attachment_r))
                with test.subTest(step=step):
                    test.assertAlmostEqual(wrap, expected_wrap, delta=2.0e-6)
                    test.assertAlmostEqual(float(solver.tendon_link_cap_ratio.numpy()[1]), expected_cap, delta=2.0e-6)
                    test.assertTrue(np.isfinite(tension).all())
                    test.assertGreater(float(tension.min()), 0.0)
                    test.assertGreaterEqual(float(rest.min()), 1.0e-6)
                    test.assertLessEqual(float(tension[0] / tension[1]), expected_cap + 2.0e-4)
                    test.assertLessEqual(float(tension[1] / tension[0]), expected_cap + 2.0e-4)
                    test.assertAlmostEqual(float(tension[0] / tension[1]), expected_cap, delta=2.0e-4)
                    test.assertAlmostEqual(float(rest.sum()) + radius * wrap, initial_inventory, delta=5.0e-7)
                    np.testing.assert_array_equal(state.body_q.numpy(), model.body_q.numpy())
            test.assertGreater(float(rest[0]), float(initial_rest[0]))
            test.assertLess(float(rest[1]), float(initial_rest[1]))


def test_nonlinear_material_law_is_immutable(test, device):
    """Reject changing any packed nonlinear law parameter before advancing either solver."""
    parameters = {
        "tendon_sigmoid_ea_low": 500.0,
        "tendon_sigmoid_ea_ratio": 10.0,
        "tendon_sigmoid_transition_strain": 0.01,
        "tendon_sigmoid_transition_width": 0.003,
    }
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        model = _chain_model(device)
        solver = solver_type(model, iterations=2, tendon_material_direct=True, **parameters)
        state, next_state = model.state(), model.state()
        original_body = state.body_q.numpy().copy()
        original_rest = solver.tendon_seg_rest_length.numpy().copy()
        for name, value in parameters.items():
            for changed in [0.0, value * 2.0] if name == "tendon_sigmoid_ea_low" else [value * 2.0]:
                with test.subTest(solver=solver_type.__name__, parameter=name, value=changed):
                    setattr(solver, name, changed)
                    with test.assertRaisesRegex(ValueError, "Reconstruct.*direct tendon material law"):
                        solver.step(state, next_state, model.control(), None, 1.0 / 240.0)
                    np.testing.assert_array_equal(state.body_q.numpy(), original_body)
                    np.testing.assert_array_equal(solver.tendon_seg_rest_length.numpy(), original_rest)
                    setattr(solver, name, value)
        solver.step(state, next_state, model.control(), None, 1.0 / 240.0)
        solver.check_tendon_material()


def test_nonlinear_route_repacking(test, device):
    """Repack nonlinear components across repeated roller activation in both solvers."""
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        with test.subTest(solver=solver_type.__name__):
            _check_route_repacking(test, device, solver_type=solver_type, sigmoid=True)


def test_nonlinear_graph_failure(test, device):
    """Latch cooperative projection failures during graph replay without fallback."""
    _check_graph_failure(test, device, sigmoid=True)


class TestTendonMaterialNonlinearIntegration(unittest.TestCase):
    """Exercise the isolated experimental material integration."""


for _test in (
    test_nonlinear_solver_entry,
    test_nonlinear_short_slack_span,
    test_vbd_adjacent_trial_uses_sigmoid,
    test_nonlinear_curved_finite_friction,
    test_nonlinear_material_law_is_immutable,
    test_nonlinear_route_repacking,
    test_nonlinear_graph_failure,
):
    add_function_test(
        TestTendonMaterialNonlinearIntegration,
        _test.__name__,
        _test,
        devices=get_test_devices(),
        # The graph-failure test drives a latched failure on purpose and the solver reports it on
        # stdout; the integration module registers the same function with check_output=False too.
        check_output=_test is not test_nonlinear_graph_failure,
    )

if __name__ == "__main__":
    unittest.main()
