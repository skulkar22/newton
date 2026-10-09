# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent dynamics checks for finite-friction tendon material transfer."""

import math
import unittest
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton


@dataclass
class _PulleyTrajectory:
    radius: float
    compliance: float
    inertia: float
    dt: float
    angle: np.ndarray
    velocity: np.ndarray
    center: np.ndarray
    rest: np.ndarray
    tension: np.ndarray
    impulse: np.ndarray
    wrap: np.ndarray

    def energy(self):
        """Compute mechanical energy independently of the material solver."""
        return 0.5 * self.inertia * self.velocity**2 + 0.5 * self.compliance * np.sum(self.tension**2, axis=1)

    def transfer(self):
        """Measure signed cable transport from the two stored rest lengths."""
        return 0.5 * ((self.rest[:, 1] - self.rest[0, 1]) - (self.rest[:, 0] - self.rest[0, 0]))

    def slip_heat(self):
        """Integrate friction work using accepted pose and material increments."""
        difference = self.tension[:, 0] - self.tension[:, 1]
        # Trapezoidal angular velocity would count implicit integration error
        # as slip; use the actual accepted angular displacement instead.
        slip = self.radius * np.diff(self.angle) - np.diff(self.transfer())
        return np.r_[0.0, np.cumsum(0.5 * (difference[1:] + difference[:-1]) * slip)]


def _make_pulley():
    """Build an undriven hinged pulley between two world-fixed anchors."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    anchors = [
        builder.add_body(xform=wp.transform(p=wp.vec3(x, 0.0, 0.0)), mass=0.0, is_kinematic=True) for x in (-0.4, 0.4)
    ]
    inertia = 0.01
    pulley = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(inertia, 0.0, 0.0, 0.0, inertia, 0.0, 0.0, 0.0, inertia),
        lock_inertia=True,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=pulley,
        axis=newton.Axis.Z,
        target_ke=0.0,
        target_kd=0.0,
    )
    builder.add_articulation([joint])
    builder.add_tendon()
    for body, kind in (
        (anchors[0], newton.TendonLinkType.ATTACHMENT),
        (pulley, newton.TendonLinkType.ROLLING),
        (anchors[1], newton.TendonLinkType.ATTACHMENT),
    ):
        builder.add_tendon_link(
            body=body,
            link_type=kind,
            radius=0.1 if kind == newton.TendonLinkType.ROLLING else 0.0,
            orientation=1,
            mu=1.0,
            axis=(0.0, 0.0, 1.0),
            compliance=0.001,
            damping=0.0,
            rest_length=-1.0,
        )
    return builder.finalize(device="cpu"), pulley


def _simulate_pulley(*, dt=0.001, iterations=8, duration=0.18, velocity=0.2, preload=10.0):
    """Record the accepted CPU trajectory without retangenting its geometry."""
    with wp.ScopedDevice("cpu"):
        model, pulley = _make_pulley()
        solver = newton.solvers.SolverXPBD(
            model,
            iterations=iterations,
            joint_linear_relaxation=1.0,
            joint_angular_relaxation=1.0,
            angular_damping=0.0,
            tendon_settle_tol=0.0,
            tendon_material_direct=True,
        )
        state, next_state, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_velocity = state.body_qd.numpy().copy()
        body_velocity[pulley, 5] = velocity
        state.body_qd.assign(body_velocity)
        solver.tendon_seg_rest_length.assign(
            solver.tendon_seg_rest_length.numpy() - np.float32(preload) * model.tendon_seg_compliance.numpy()
        )

        bodies = model.tendon_link_body.numpy()
        compliance = model.tendon_seg_compliance.numpy().astype(float)
        records = []

        def record():
            poses = state.body_q.numpy().astype(float)
            left = solver.tendon_seg_active_link_l.numpy()
            right = solver.tendon_seg_active_link_r.numpy()
            local_l = solver.tendon_seg_attachment_l_local.numpy().astype(float)
            local_r = solver.tendon_seg_attachment_r_local.numpy().astype(float)

            def world(link, local):
                pose = poses[bodies[link]]
                q = pose[3:6]
                return pose[:3] + local + 2.0 * np.cross(q, np.cross(q, local) + pose[6] * local)

            length = np.array(
                [
                    np.linalg.norm(world(r, b) - world(l, a))
                    for l, r, a, b in zip(left, right, local_l, local_r, strict=True)
                ]
            )
            rest = solver.tendon_seg_rest_length.numpy().astype(float)
            wrap = abs(np.arctan2(np.cross(local_r[0], local_l[1])[2], np.dot(local_r[0], local_l[1])))
            records.append(
                (
                    2.0 * np.arctan2(poses[pulley, 5], poses[pulley, 6]),
                    float(state.body_qd.numpy()[pulley, 5]),
                    poses[pulley, :3],
                    rest,
                    np.maximum((length - rest) / compliance, 0.0),
                    -solver.tendon_seg_lambda.numpy().astype(float),
                    wrap,
                )
            )

        record()
        for _ in range(round(duration / dt)):
            state.clear_forces()
            solver.step(state, next_state, control, None, dt)
            state, next_state = next_state, state
            record()
        solver.check_tendon_material()

        return _PulleyTrajectory(
            radius=float(model.tendon_link_radius.numpy()[1]),
            compliance=float(compliance[0]),
            inertia=float(model.body_inertia.numpy()[pulley, 2, 2]),
            dt=dt,
            **{
                field: np.asarray(values)
                for field, values in zip(
                    ("angle", "velocity", "center", "rest", "tension", "impulse", "wrap"),
                    zip(*records, strict=True),
                    strict=True,
                )
            },
        )


class TestTendonMaterialDynamics(unittest.TestCase):
    """Check transport and reaction against a pulley with a known physical model."""

    def test_sticking_matches_oscillator_and_full_reaction_torque(self):
        """Recover the independently derived sticking oscillator and its torque."""
        result = _simulate_pulley()
        frequency_sq = 2.0 * result.radius**2 / (result.compliance * result.inertia)
        expected_angle = np.zeros_like(result.angle)
        expected_velocity = np.zeros_like(result.velocity)
        expected_velocity[0] = result.velocity[0]
        for i in range(1, len(expected_angle)):
            expected_angle[i] = (expected_angle[i - 1] + result.dt * expected_velocity[i - 1]) / (
                1.0 + result.dt**2 * frequency_sq
            )
            expected_velocity[i] = expected_velocity[i - 1] - result.dt * frequency_sq * expected_angle[i]

        # Sticking transfers R*angle, giving stiffness R²*(1/c_left+1/c_right).
        # Compare with backward Euler, not an undamped sinusoid: integration
        # dissipates energy even though this fixture has no physical damping.
        np.testing.assert_allclose(result.angle, expected_angle, atol=1.0e-5, rtol=0)
        np.testing.assert_allclose(result.transfer(), result.radius * result.angle, atol=3.0e-7, rtol=0)
        np.testing.assert_allclose(result.center, 0.0, atol=1.0e-8, rtol=0)
        cap = np.exp(result.wrap)  # The fixture's finite friction coefficient is one.
        beta = (cap - 1.0) / (cap + 1.0)
        friction_margin = beta * result.tension.sum(axis=1) - np.abs(np.diff(result.tension, axis=1)[:, 0])
        self.assertGreater(float(friction_margin.min()), 3.0)

        positive_crossings = np.flatnonzero((result.angle[:-1] <= 0.0) & (result.angle[1:] > 0.0))
        positive_crossings = positive_crossings[positive_crossings > 0]
        self.assertGreater(len(positive_crossings), 0, "The pulley did not complete its expected oscillation")
        i = positive_crossings[0]
        period = result.dt * (i - result.angle[i] / (result.angle[i + 1] - result.angle[i]))
        self.assertAlmostEqual(period / (2.0 * math.pi / math.sqrt(frequency_sq)), 1.0, delta=0.002)

        impulse_torque = result.radius * (result.impulse[1:, 1] - result.impulse[1:, 0]) / result.dt
        measured_torque = result.inertia * np.diff(result.velocity) / result.dt
        nonzero = np.abs(impulse_torque) > 0.1 * np.max(np.abs(impulse_torque))
        np.testing.assert_allclose(measured_torque[nonzero], impulse_torque[nonzero], rtol=0.003, atol=1.0e-5)
        self.assertLessEqual(float(np.max(result.energy()) - result.energy()[0]), 1.0e-6)

    def test_iterations_do_not_accumulate_rolling_transport(self):
        """Keep the converged pulley history unchanged when iterations increase."""
        few = _simulate_pulley(duration=0.025, iterations=8)
        many = _simulate_pulley(duration=0.025, iterations=32)
        for field in ("angle", "velocity", "rest"):
            with self.subTest(field=field):
                np.testing.assert_allclose(getattr(few, field), getattr(many, field), atol=1.0e-7, rtol=0)

    def test_sliding_dissipates_energy_and_refines_stopping_angle(self):
        """Approach the finite-friction stopping angle from either sliding direction."""
        for direction in (-1, 1):
            errors = []
            heat_errors = []
            for dt in (0.001, 0.0005):
                with self.subTest(direction=direction, dt=dt):
                    result = _simulate_pulley(dt=dt, duration=0.065, velocity=direction * 2.0)
                    frequency_sq = 2.0 * result.radius**2 / (result.compliance * result.inertia)
                    beta = math.tanh(0.5 * result.wrap[0])
                    slip_angle = result.compliance * float(np.mean(result.tension[0])) * beta / result.radius
                    speed_at_slip_sq = result.velocity[0] ** 2 - frequency_sq * slip_angle**2
                    self.assertGreater(speed_at_slip_sq, 0.0)
                    acceleration = frequency_sq * slip_angle
                    expected_peak = slip_angle + speed_at_slip_sq / (2.0 * acceleration)
                    expected_heat = 0.5 * result.inertia * speed_at_slip_sq

                    turns = np.flatnonzero(
                        (direction * result.velocity[:-1] > 0.0) & (direction * result.velocity[1:] <= 0.0)
                    )
                    self.assertGreater(len(turns), 0, "The capstan torque failed to stop the pulley")
                    i = turns[0]
                    peak = i if abs(result.angle[i]) > abs(result.angle[i + 1]) else i + 1
                    relative_error = abs(abs(result.angle[peak]) / expected_peak - 1.0)
                    errors.append(relative_error)
                    self.assertLess(relative_error, 0.04)

                    heat = result.slip_heat()
                    self.assertGreater(float(heat[peak]), 0.0)
                    heat_errors.append(abs(float(heat[peak]) - expected_heat))
                    self.assertLessEqual(float(np.max(result.energy() + heat) - result.energy()[0]), 1.0e-6)
            with self.subTest(direction=direction, comparison="refinement"):
                self.assertEqual(len(errors), 2)
                self.assertEqual(len(heat_errors), 2)
                self.assertLess(errors[1], 0.7 * errors[0])
                self.assertLess(errors[1], 0.02)
                self.assertLess(heat_errors[1], heat_errors[0])

    def test_unloaded_rotation_does_not_create_tension(self):
        """Let an unloaded pulley rotate without manufacturing cable tension."""
        result = _simulate_pulley(duration=0.01, preload=0.0)
        np.testing.assert_allclose(result.tension, 0.0, atol=1.0e-4, rtol=0)
        np.testing.assert_allclose(result.velocity, result.velocity[0], atol=1.0e-6, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
