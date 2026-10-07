# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen-geometry tests for direct finite-friction material transfer."""

import unittest

import numpy as np
import warp as wp

import newton


class _MaterialFixture:
    """Own real solver state with prescribed geometry and rolling increments."""

    def __init__(self, kinds, extension, *, compliance=None, cap=2.0, dynamic=()):
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_tendon()
        for i, kind in enumerate(kinds):
            body = builder.add_body(xform=wp.transform(p=wp.vec3(float(i), 0.0, 0.0)), mass=0.0, is_kinematic=True)
            builder.add_tendon_link(
                body=body,
                link_type=kind,
                axis=(0.0, 0.0, 1.0),
                radius=0.1 if kind == newton.TendonLinkType.ROLLING else 0.0,
                compliance=0.001,
                rest_length=0.1,
                mu=0.1,
                dynamic=i in dynamic,
            )
        self.model = builder.finalize(device="cpu")
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=1, tendon_material_direct=True)
        self.state = self.model.state()
        n = len(kinds) - 1
        solver = self.solver
        solver.tendon_seg_length.fill_(0.1)
        self.set_extension(extension)
        solver.tendon_seg_active.fill_(1)
        solver.tendon_seg_active_compliance.assign(
            np.full(n, 0.001, dtype=np.float32) if compliance is None else np.asarray(compliance, dtype=np.float32)
        )
        solver.tendon_seg_active_damping.zero_()
        solver.tendon_seg_active_link_l.assign(np.arange(n, dtype=np.int32))
        solver.tendon_seg_active_link_r.assign(np.arange(1, n + 1, dtype=np.int32))
        solver.tendon_link_active.fill_(True)
        solver.tendon_link_active_step.fill_(True)
        solver.tendon_link_route_rest_length.zero_()
        solver.tendon_seg_rolling_delta_l.zero_()
        solver.tendon_seg_rolling_delta_r.zero_()
        cone_l = np.full(n + 1, -1, dtype=np.int32)
        cone_r = cone_l.copy()
        for i in range(1, n):
            if kinds[i] != newton.TendonLinkType.ATTACHMENT:
                cone_l[i], cone_r[i] = i - 1, i
        solver.tendon_link_cone_seg_l.assign(cone_l)
        solver.tendon_link_cone_seg_r.assign(cone_r)
        solver.tendon_link_cap_ratio.fill_(cap)

    def set_extension(self, extension):
        """Change the step's material snapshot without reusing prior solve output."""
        rest = self.solver.tendon_seg_length.numpy() - np.asarray(extension, dtype=np.float32)
        self.solver.tendon_seg_route_rest_length.assign(rest)
        self.solver.tendon_seg_rest_length_step.assign(rest)

    def set_motion(self, differential, *, common=None):
        """Prescribe the two tangent increments for each interior link."""
        differential = np.asarray(differential, dtype=np.float32)
        common = np.zeros_like(differential) if common is None else np.asarray(common, dtype=np.float32)
        self.solver.tendon_seg_rolling_delta_r.assign(np.r_[common + differential, np.float32(0)])
        self.solver.tendon_seg_rolling_delta_l.assign(np.r_[np.float32(0), common - differential])

    def solve(self, *, rolling=True, pinhole=True):
        """Launch the production material kernel without moving any bodies."""
        model, solver, state = self.model, self.solver, self.state
        wp.launch(
            kernel=solver._tendon_material_kernel,
            dim=model.tendon_count,
            inputs=[
                state.body_q,
                state.body_qd,
                state.body_q,
                model.body_com,
                model.tendon_start,
                model.tendon_link_body,
                model.tendon_link_type,
                model.tendon_link_radius,
                model.tendon_link_orientation,
                model.tendon_link_offset,
                model.tendon_link_axis,
                solver.tendon_seg_rest_length,
                solver.tendon_seg_rest_length_step,
                solver.tendon_seg_route_rest_length,
                solver.tendon_seg_stretch,
                solver.tendon_seg_damping_tension,
                solver.tendon_seg_active,
                solver.tendon_seg_active_link_l,
                solver.tendon_seg_active_link_r,
                solver.tendon_seg_active_compliance,
                solver.tendon_seg_active_damping,
                solver.tendon_link_active,
                solver.tendon_link_active_step,
                solver.tendon_link_route_rest_length,
                solver.tendon_seg_attachment_l,
                solver.tendon_seg_attachment_r,
                solver.tendon_seg_length,
                solver.tendon_seg_attachment_l_local,
                solver.tendon_seg_attachment_r_local,
                solver.tendon_seg_rolling_delta_l,
                solver.tendon_seg_rolling_delta_r,
                solver.tendon_link_cone_seg_l,
                solver.tendon_link_cone_seg_r,
                solver.tendon_link_cap_ratio,
                solver.tendon_cone_sweep_count,
                0,
                1.0 / 60.0,
                int(rolling),
                int(pinhole),
                1,
                solver.tendon_max_sweeps,
                solver.tendon_settle_tol,
                0.0,
                0.0,
                0.0,
                0.0,
                solver._tendon_material_state,
                1,  # latch a rejection: this fixture solves an accepted pose, not a trial pose
                solver._tendon_link_material_transfer,
            ],
            device="cpu",
        )
        solver.check_tendon_material()
        return solver.tendon_seg_stretch.numpy().copy()


class TestTendonMaterialTransfer(unittest.TestCase):
    """Verify rolling trials, material components, and capstan projection together."""

    def setUp(self):
        """Provide short link-type names for the authored test routes."""
        self.attachment = newton.TendonLinkType.ATTACHMENT
        self.roller = newton.TendonLinkType.ROLLING
        self.pinhole = newton.TendonLinkType.PINHOLE

    def fixture(self, extension, *, kinds=None, compliance=None, dynamic=()):
        """Create an attachment-ended rolling route unless explicitly specified."""
        if kinds is None:
            kinds = [self.attachment] + [self.roller] * (len(extension) - 1) + [self.attachment]
        return _MaterialFixture(kinds, extension, compliance=compliance, dynamic=dynamic)

    def assert_cones(self, fixture, stretch):
        """Check total tension bounds on all active material connections."""
        solver = fixture.solver
        tension = np.maximum(stretch / solver.tendon_seg_active_compliance.numpy(), 0.0)
        kinds = fixture.model.tendon_link_type.numpy()
        active = solver.tendon_link_active.numpy()
        caps = solver.tendon_link_cap_ratio.numpy()
        left = solver.tendon_link_cone_seg_l.numpy()
        right = solver.tendon_link_cone_seg_r.numpy()
        for link, (l, r) in enumerate(zip(left, right, strict=True)):
            if l < 0 or r < 0 or kinds[link] == self.attachment:
                continue
            if kinds[link] == self.roller and not active[link]:
                continue
            self.assertLessEqual(float(tension[l] - caps[link] * tension[r]), 5.0e-5)
            self.assertLessEqual(float(tension[r] - caps[link] * tension[l]), 5.0e-5)

    def test_sticking_transports_full_rolling_motion(self):
        """Keep the full no-slip trial when its finite-friction cones are feasible."""
        fixture = self.fixture([0.01, 0.01])
        fixture.set_motion([0.001])
        result = fixture.solve()
        np.testing.assert_allclose(result, [0.009, 0.011], atol=1.0e-8, rtol=0)
        self.assert_cones(fixture, result)

    def test_sliding_returns_to_either_capstan_face(self):
        """Project excessive rolling motion onto either finite-friction cone face."""
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                fixture = self.fixture([0.01, 0.01])
                fixture.set_motion([direction * 0.008])
                result = fixture.solve()
                expected = np.array([0.02 / 3, 0.04 / 3])
                if direction < 0:
                    expected = expected[::-1]
                np.testing.assert_allclose(result, expected, atol=1.0e-8, rtol=0)
                self.assert_cones(fixture, result)
                self.assertAlmostEqual(float(result.sum()), 0.02, delta=1.0e-8)

    def test_rolling_has_no_post_projection_transfer_or_trial_clamp(self):
        """Keep feasible projected states unchanged, even for an over-bound trial."""
        for extension, differential, expected in (
            ([0.01, 0.005], -0.003, [0.01, 0.005]),
            ([0.002, 0.002], 0.11, [0.004 / 3, 0.008 / 3]),
        ):
            with self.subTest(differential=differential):
                fixture = self.fixture(extension)
                fixture.set_motion([differential])
                result = fixture.solve()
                np.testing.assert_allclose(result, expected, atol=1.0e-8, rtol=0)
                self.assert_cones(fixture, result)
                self.assertGreater(float(fixture.solver.tendon_seg_rest_length.numpy().min()), 1.0e-6)
                if differential > 0.1:
                    scratch = fixture.solver._tendon_material_state
                    self.assertGreater(float(scratch.initial.numpy()[1]), float(scratch.upper.numpy()[1]))
                    self.assertAlmostEqual(float(scratch.initial.numpy()[1]), 0.112, delta=1.0e-8)

    def test_multiple_rollers_account_for_common_arc_exchange_once(self):
        """Conserve free-span plus wrapped inventory under simultaneous roller motion."""
        fixture = self.fixture([0.009, 0.008, 0.01])
        common = np.array([0.0002, -0.0003])
        fixture.set_motion([0.001, -0.0015], common=common)
        before = fixture.solver.tendon_seg_route_rest_length.numpy().astype(float).sum()
        result = fixture.solve()
        np.testing.assert_allclose(result, [0.0078, 0.0106, 0.0088], atol=1.0e-8, rtol=0)
        self.assert_cones(fixture, result)
        after = fixture.solver.tendon_seg_rest_length.numpy().astype(float).sum()
        self.assertAlmostEqual(float(after - before), float(2.0 * common.sum()), delta=2.0e-8)

    def test_internal_attachment_splits_rolling_and_pinhole_components(self):
        """Prevent material transfer across an attachment inside a mixed route."""
        fixture = self.fixture(
            [0.012, 0.001, 0.001, 0.001, 0.014],
            kinds=[self.attachment, self.roller, self.pinhole, self.attachment, self.roller, self.attachment],
        )
        # Increments at the pinhole and attachment are irrelevant to transport.
        fixture.set_motion([0.001, 0.04, -0.04, -0.002])
        result = fixture.solve()
        np.testing.assert_allclose(result, [0.008, 0.004, 0.002, 0.005, 0.01], atol=1.0e-8, rtol=0)
        self.assert_cones(fixture, result)
        np.testing.assert_allclose([result[:3].sum(), result[3:].sum()], [0.014, 0.015], atol=1.0e-8, rtol=0)

    def test_inactive_roller_is_bypassed_in_mixed_route(self):
        """Ignore a bypassed roller and its stale slot while projecting active spans."""
        fixture = self.fixture(
            [0.01, 0.01, 0.02, 0.01, 0.01],
            kinds=[self.attachment, self.roller, self.roller, self.pinhole, self.roller, self.attachment],
            dynamic=(2,),
        )
        solver = fixture.solver
        solver.tendon_link_active.assign(np.array([True, True, False, True, True, True]))
        solver.tendon_link_active_step.assign(np.array([True, True, False, True, True, True]))
        solver.tendon_seg_active.assign(np.array([1, 1, 0, 1, 1], dtype=np.int32))
        solver.tendon_seg_active_link_r.assign(np.array([1, 3, 3, 4, 5], dtype=np.int32))
        solver.tendon_link_cone_seg_l.assign(np.array([-1, 0, 1, 1, 3, -1], dtype=np.int32))
        solver.tendon_link_cone_seg_r.assign(np.array([-1, 1, 2, 3, 4, -1], dtype=np.int32))
        fixture.set_motion([0.001, 99.0, 0.0, -0.002], common=[0.0002, 0.0, 0.0, 0.0001])
        inactive_rest = float(solver.tendon_seg_route_rest_length.numpy()[2])
        result = fixture.solve()
        np.testing.assert_allclose(result[[0, 1, 3, 4]], [0.0088, 0.0108, 0.0119, 0.0079], atol=1.0e-8, rtol=0)
        self.assertEqual(float(solver.tendon_seg_rest_length.numpy()[2]), inactive_rest)
        self.assert_cones(fixture, result)

    def test_repeated_calls_rebuild_from_the_step_snapshot(self):
        """Rebuild every rolling trial without accumulating prior iteration output."""
        fixture = self.fixture([0.01, 0.01])
        fixture.set_motion([0.001])
        expected = fixture.solve()
        rest = fixture.solver.tendon_seg_rest_length.numpy().copy()
        for _ in range(8):
            fixture.solver.tendon_seg_rest_length.fill_(-123.0)
            fixture.solver.tendon_seg_stretch.fill_(456.0)
            np.testing.assert_array_equal(fixture.solve(), expected)
            np.testing.assert_array_equal(fixture.solver.tendon_seg_rest_length.numpy(), rest)
        fixture.set_extension([0.015, 0.005])
        changed = fixture.solve()
        np.testing.assert_allclose(changed, [0.04 / 3, 0.02 / 3], atol=1.0e-8, rtol=0)
        self.assert_cones(fixture, changed)

    def test_unequal_compliance_constrains_tension_not_stretch(self):
        """Apply capstan ratios to tension when adjacent span compliances differ."""
        fixture = self.fixture([0.01, 0.02], compliance=[0.001, 0.002])
        fixture.set_motion([0.015])
        result = fixture.solve()
        np.testing.assert_allclose(result, [0.006, 0.024], atol=1.0e-8, rtol=0)
        self.assert_cones(fixture, result)

    def test_slack_trial_only_redistributes_when_a_span_tightens(self):
        """Retain slack history, remove local trial tension, and later handle taut cable."""
        fixture = self.fixture([-0.02, -0.01])
        fixture.set_motion([0.003])
        np.testing.assert_allclose(fixture.solve(), [-0.023, -0.007], atol=1.0e-8, rtol=0)
        fixture.set_motion([0.015])
        np.testing.assert_allclose(fixture.solve(), [-0.03, 0.0], atol=1.0e-8, rtol=0)
        fixture.set_extension([0.01, 0.005])
        result = fixture.solve()
        np.testing.assert_allclose(result, [0.005, 0.01], atol=1.0e-8, rtol=0)
        self.assert_cones(fixture, result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
