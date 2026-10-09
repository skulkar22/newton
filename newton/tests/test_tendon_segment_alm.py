# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent finite-friction checks for the segment-ALM experiment."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.tendon_kernels import update_duals_tendon
from newton.tests.test_tendon_capstan import build_kinematic_pulley_atwood
from newton.tests.test_tendon_material_vbd import _body_force, _simulate_pulley
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _fixture(
    device,
    *,
    per_segment=True,
    target=(10.0, 20.0),
    mu=0.35,
    compliance=(1.0e-6, 1.0e-6),
    damping=(0.0, 0.0),
    material=None,
):
    with wp.ScopedDevice(device):
        model, left, right, pulley = build_kinematic_pulley_atwood(mu=mu, mass_right=2.0)
        model.tendon_seg_damping.assign(np.asarray(damping, dtype=np.float32))
        model.tendon_seg_compliance.assign(np.asarray(compliance, dtype=np.float32))
        model.body_color_groups = [wp.array([i], dtype=int) for i in range(model.body_count)]
        solver = newton.solvers.SolverVBD(
            model,
            iterations=32,
            tendon_alm=True,
            tendon_alm_per_segment=per_segment,
            tendon_alm_min_stiffness_ratio=0.0,
            tendon_material_direct=True,
            **(material or {}),
        )
        state = model.state()
        wp.copy(solver.body_q_prev, model.body_q)
        lengths = solver.tendon_seg_length.numpy()
        rest = lengths - np.asarray(target, dtype=np.float32) * model.tendon_seg_compliance.numpy()
        solver.tendon_seg_rest_length.assign(rest)
        solver.tendon_seg_route_rest_length.assign(rest)
        solver._snapshot_tendon_step_state()
        solver._step_tendon_alm_state(1.0 / 60.0)
        return model, solver, state, (left, right, pulley), lengths, rest


def _dual_step(model, solver, state):
    wp.launch(
        update_duals_tendon,
        dim=model.tendon_count,
        inputs=[
            1.0 / 60.0,
            state.body_q,
            solver.body_q_prev,
            model.body_com,
            model.tendon_start,
            model.tendon_link_body,
            model.tendon_link_type,
            model.tendon_link_offset,
            model.tendon_link_axis,
            solver.tendon_seg_attachment_l_local,
            solver.tendon_seg_attachment_r_local,
            solver.tendon_seg_rest_length,
            solver.tendon_seg_active_compliance,
            solver.tendon_seg_active_damping,
            solver.tendon_seg_active,
            solver.tendon_seg_active_link_l,
            solver.tendon_seg_active_link_r,
            solver.tendon_sigmoid_ea_low,
            solver.tendon_sigmoid_ea_ratio,
            solver.tendon_sigmoid_transition_strain,
            solver.tendon_sigmoid_transition_width,
            solver.tendon_alm_lambda,
            solver.tendon_alm_k,
            solver.tendon_seg_alm_lambda,
            solver.tendon_seg_alm_k,
            int(solver.tendon_alm_per_segment),
        ],
        device=model.device,
    )


def test_frozen_sticking(test, device):
    """Retain unequal 10 N / 20 N loads and their roller torque inside the sticking cone."""
    model, solver, state, bodies, lengths, rest = _fixture(device)
    physical = (lengths.astype(float) - rest) / model.tendon_seg_compliance.numpy()
    test.assertGreater(float(solver.tendon_seg_alm_k.numpy().min()), 0.0)
    for _ in range(512):
        solver._update_tendon_routing(state, 1.0 / 60.0, False)
        _dual_step(model, solver, state)
    solver.check_tendon_material()
    np.testing.assert_allclose(solver.tendon_seg_rest_length.numpy(), rest, atol=2.0e-7, rtol=0.0)
    np.testing.assert_allclose(solver.tendon_seg_alm_lambda.numpy(), physical, atol=0.025, rtol=0.0)
    for body in bodies:
        actual, _ = _body_force(model, solver, body, None)
        solver.tendon_alm = False
        expected, _ = _body_force(model, solver, body, None)
        solver.tendon_alm = True
        np.testing.assert_allclose(actual, expected, atol=0.025, rtol=0.0)


def test_shifted_projection(test, device):
    """Project unequal ALM offsets onto the capstan cone while conserving physical cable."""
    for reverse in (False, True):
        model, solver, state, _, lengths, rest = _fixture(device)
        compliance = np.array([1.0e-6, 3.0e-6], dtype=np.float32)
        rho = np.array([1.0e5, 2.0e5], dtype=np.float32)
        lam = np.array([100.0, 1.0], dtype=np.float32)
        if reverse:
            lam = lam[::-1].copy()
        model.tendon_seg_compliance.assign(compliance)
        solver.tendon_seg_active_compliance.assign(compliance)
        solver.tendon_seg_alm_lambda.assign(lam)
        solver.tendon_seg_alm_k.assign(rho)
        solver._update_tendon_cone_rows(model, state.body_q, False)
        cap = float(solver.tendon_link_cap_ratio.numpy()[1])
        c_eff = compliance.astype(float) + 1.0 / rho.astype(float)
        offset = lam.astype(float) / rho
        shifted = lengths.astype(float) - rest + offset
        test.assertGreater(max(shifted / c_eff) / min(shifted / c_eff), cap)
        ratio = np.array([1.0, cap]) if reverse else np.array([cap, 1.0])
        expected_force = ratio * shifted.sum() / np.dot(c_eff, ratio)
        expected_rest = lengths - (c_eff * expected_force - offset)
        solver._update_tendon_routing(state, 1.0 / 60.0, True)
        solver.check_tendon_material()
        actual_rest = solver.tendon_seg_rest_length.numpy()
        np.testing.assert_allclose(actual_rest, expected_rest, atol=1.2e-7, rtol=0.0)
        test.assertAlmostEqual(float(actual_rest.astype(float).sum()), float(rest.astype(float).sum()), delta=1.2e-7)
        actual_force = (lengths.astype(float) - actual_rest + offset) / c_eff
        np.testing.assert_allclose(actual_force, expected_force, atol=0.015, rtol=0.0)


def test_unequal_compliance_sticking(test, device):
    """Recover a sticking state after transient ALM force imbalance on unequal stiffnesses."""
    for compliance in ((1.0e-6, 1.0e-5), (1.0e-5, 1.0e-6)):
        model, solver, state, _, lengths, rest = _fixture(device, compliance=compliance)
        physical = (lengths.astype(float) - rest) / model.tendon_seg_compliance.numpy()
        for _ in range(1024):
            solver._update_tendon_routing(state, 1.0 / 60.0, False)
            _dual_step(model, solver, state)
        solver.check_tendon_material()
        np.testing.assert_allclose(solver.tendon_seg_rest_length.numpy(), rest, atol=2.0e-7, rtol=0.0)
        np.testing.assert_allclose(solver.tendon_seg_alm_lambda.numpy(), physical, atol=0.15, rtol=0.0)


def test_roller_reaction_uses_alm_force(test, device):
    """Endpoint tractions must balance at a common origin, including off-cone trials."""
    for target in (np.array([10.0, 20.0]), np.array([80.0, 5.0]), np.array([5.0, 80.0])):
        model, solver, _, bodies, lengths, rest = _fixture(device)
        compliance = model.tendon_seg_compliance.numpy().astype(float)
        stretch = lengths.astype(float) - rest
        physical = stretch / compliance
        rho = np.full(2, 1.0e5)
        lam = (1.0 + compliance * rho) * target - rho * stretch
        solver.tendon_seg_alm_lambda.assign(lam.astype(np.float32))
        solver.tendon_seg_alm_k.assign(rho.astype(np.float32))
        wrenches = np.array([_body_force(model, solver, b, physical)[0] for b in range(model.body_count)])
        positions = model.body_q.numpy()[:, :3]  # fixture COM offsets are zero
        np.testing.assert_allclose(wrenches[:, 0].sum(axis=0), 0.0, atol=2.0e-5, rtol=0.0)
        moments = wrenches[:, 1] + np.cross(positions, wrenches[:, 0])
        np.testing.assert_allclose(moments.sum(axis=0), 0.0, atol=2.0e-5, rtol=0.0)
        radius = float(model.tendon_link_radius.numpy()[1])
        test.assertAlmostEqual(
            float(np.linalg.norm(wrenches[bodies[2], 1])), radius * abs(target[0] - target[1]), delta=0.003
        )


def test_slack_releases_multiplier(test, device):
    """Release a positive multiplier on slack segments without inventing cable length."""
    model, solver, state, _, _, rest = _fixture(device, target=(-10.0, -20.0))
    solver.tendon_seg_alm_lambda.fill_(20.0)
    for _ in range(256):
        solver._update_tendon_routing(state, 1.0 / 60.0, False)
        _dual_step(model, solver, state)
    solver.check_tendon_material()
    np.testing.assert_array_equal(solver.tendon_seg_alm_lambda.numpy(), 0.0)
    test.assertAlmostEqual(
        float(solver.tendon_seg_rest_length.numpy().astype(float).sum()), float(rest.astype(float).sum()), delta=2.0e-7
    )


def test_graph_matches_eager(test, device, *, material=None, damping=(0.0, 0.0)):
    """Replay complete VBD steps with mutable per-segment duals and match eager execution."""
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires CUDA")
    results = []
    for capture in (False, True):
        model, solver, state, _, _, _ = _fixture(device, material=material, damping=damping)
        next_state = model.state()
        control = model.control()

        def two_steps(state=state, solver=solver, next_state=next_state, control=control):
            state.clear_forces()
            solver.step(state, next_state, control, None, 0.001)
            next_state.clear_forces()
            solver.step(next_state, state, control, None, 0.001)

        two_steps()
        if capture:
            with wp.ScopedCapture(device=device) as graph:
                two_steps()
            for _ in range(3):
                wp.capture_launch(graph.graph)
        else:
            for _ in range(3):
                two_steps()
        solver.check_tendon_material()
        result = (state.body_q.numpy(), solver.tendon_seg_rest_length.numpy(), solver.tendon_seg_alm_lambda.numpy())
        for values in result:
            test.assertTrue(np.isfinite(values).all())
        results.append(result)
    for eager, captured in zip(*results, strict=True):
        np.testing.assert_allclose(captured, eager, atol=2.0e-6, rtol=2.0e-6)


class TestTendonSegmentALM(unittest.TestCase):
    """Exercise the opt-in linear, undamped segment-ALM mode."""

    def test_rejects_missing_prerequisites(self):
        """Reject per-segment ALM without the coupled material/force implementation."""
        with wp.ScopedDevice("cpu"):
            model, _, _, _ = build_kinematic_pulley_atwood()
            model.tendon_seg_damping.zero_()
            args = {"tendon_alm": True, "tendon_alm_per_segment": True, "tendon_material_direct": True}
            for key in ("tendon_alm", "tendon_material_direct"):
                with self.assertRaisesRegex(ValueError, "requires tendon_alm and tendon_material_direct"):
                    newton.solvers.SolverVBD(model, **(args | {key: False}))

    def test_sticking_dynamics_and_reversal(self):
        """Match an implicit torsional oscillator through both directions of sticking motion."""
        result = _simulate_pulley(
            iterations=32,
            solver_options={
                "tendon_alm": True,
                "tendon_alm_per_segment": True,
                "tendon_alm_min_stiffness_ratio": 0.0,
            },
        )
        omega_sq = 2.0 * result.radius**2 / (result.compliance * result.inertia)
        expected_angle = np.zeros_like(result.angle)
        expected_velocity = np.zeros_like(result.velocity)
        expected_velocity[0] = result.velocity[0]
        for i in range(1, len(expected_angle)):
            expected_angle[i] = (expected_angle[i - 1] + result.dt * expected_velocity[i - 1]) / (
                1.0 + result.dt**2 * omega_sq
            )
            expected_velocity[i] = expected_velocity[i - 1] - result.dt * omega_sq * expected_angle[i]
        np.testing.assert_allclose(result.angle, expected_angle, atol=1.0e-5, rtol=0.0)
        np.testing.assert_allclose(result.transfer(), result.radius * result.angle, atol=4.0e-7, rtol=0.0)
        self.assertLessEqual(float(np.max(result.energy()) - result.energy()[0]), 2.0e-6)

    def test_sliding_dynamics_and_reversal(self):
        """Recover the capstan-limited stopping angle and nonnegative friction loss in both directions."""
        for sign in (-1, 1):
            result = _simulate_pulley(
                iterations=32,
                duration=0.065,
                velocity=2.0 * sign,
                solver_options={
                    "tendon_alm": True,
                    "tendon_alm_per_segment": True,
                    "tendon_alm_min_stiffness_ratio": 0.0,
                },
            )
            omega_sq = 2.0 * result.radius**2 / (result.compliance * result.inertia)
            beta = math.tanh(0.5 * result.wrap[0])
            slip_angle = result.compliance * float(np.mean(result.tension[0])) * beta / result.radius
            expected_peak = slip_angle + (4.0 - omega_sq * slip_angle**2) / (2.0 * omega_sq * slip_angle)
            turns = np.flatnonzero((sign * result.velocity[:-1] > 0.0) & (sign * result.velocity[1:] <= 0.0))
            self.assertGreater(len(turns), 0)
            i = turns[0]
            peak = max(abs(result.angle[i]), abs(result.angle[i + 1]))
            self.assertLess(abs(peak / expected_peak - 1.0), 0.04)
            self.assertGreater(float(result.slip_heat()[i]), 0.0)
            self.assertLessEqual(float(np.max(result.energy() + result.slip_heat()) - result.energy()[0]), 2.0e-6)


for _device in get_test_devices():
    for _test in (
        test_frozen_sticking,
        test_shifted_projection,
        test_roller_reaction_uses_alm_force,
        test_slack_releases_multiplier,
        test_graph_matches_eager,
        test_unequal_compliance_sticking,
    ):
        add_function_test(TestTendonSegmentALM, _test.__name__, _test, devices=[_device])

if __name__ == "__main__":
    unittest.main()
