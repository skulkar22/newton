# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Coaxial impact reference independent of tendon forces and material solvers."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _impact(device, iterations, reverse, sign, ke, kd, enabled=True, graph=False):
    inertia = np.array([1.0e-5, 1.0e-6])
    dt = 1.0 / 1200.0
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    bodies = [builder.add_link(mass=0.01, inertia=wp.mat33(np.eye(3) * value), lock_inertia=True) for value in inertia]
    root = builder.add_joint_revolute(-1, bodies[0], axis=newton.Axis.Z, target_ke=0.0, target_kd=0.0)
    hinge = builder.add_joint_revolute(
        bodies[0],
        bodies[1],
        axis=newton.Axis.Z,
        target_ke=0.0,
        target_kd=0.0,
        limit_lower=-0.05 if sign < 0 else -1.0,
        limit_upper=0.05 if sign > 0 else 1.0,
        limit_ke=ke,
        limit_kd=kd,
    )
    builder.add_articulation([root, hinge])
    model = builder.finalize(device=device)
    model.body_color_groups = [wp.array([i], dtype=int, device=device) for i in (bodies[::-1] if reverse else bodies)]
    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_avbd_beta=1.0e6,
        rigid_joint_linear_k_start=1.0e7,
        rigid_joint_angular_k_start=1.0e5,
        rigid_joint_limit_alm=enabled,
    )
    state, out = model.state(), model.state()
    control = model.control()
    velocity = state.body_qd.numpy()
    velocity[1, 5] = sign * 10.0
    state.body_qd.assign(velocity)
    history = []
    reference = []
    omega = np.array([0.0, sign * 10.0])
    angles = np.zeros(2)
    for _ in range(14):
        predicted = angles + dt * omega
        g = sign * (predicted[1] - predicted[0]) - 0.05
        g_prev = max(sign * (angles[1] - angles[0]) - 0.05, 0.0)
        # Closed-form backward Euler for the unilateral Kelvin-Voigt stop.
        stiffness = ke + 0.5 * kd / dt
        traction = max((stiffness * g - 0.5 * kd / dt * g_prev) / (1 + stiffness * dt**2 * np.sum(1 / inertia)), 0)
        omega += sign * dt * traction * np.array([1 / inertia[0], -1 / inertia[1]])
        angles += dt * omega
        reference.append(omega.copy())
        if graph:
            with wp.ScopedCapture(device=device) as capture:
                solver.step(state, out, control, None, dt)
            wp.capture_launch(capture.graph)
        else:
            solver.step(state, out, control, None, dt)
        history.append(out.body_qd.numpy()[:, 5].copy())
        state, out = out, state
    return np.asarray(history), np.asarray(reference), inertia


def test_revolute_limit_momentum(self, device):
    for iterations in (32, 256):
        for reverse in (False, True):
            for sign in (-1, 1):
                with self.subTest(iterations=iterations, reverse=reverse, sign=sign):
                    actual, reference, inertia = _impact(device, iterations, reverse, sign, 1.0e8, 2.0e4)
                    np.testing.assert_allclose(actual, reference, atol=0.002, rtol=0.002)
                    np.testing.assert_allclose(actual @ inertia, sign * 1.0e-5, atol=2.0e-8)
                    self.assertLessEqual(float(np.max(0.5 * actual**2 @ inertia)), 5.01e-5)


def test_revolute_limit_compliant_reference(self, device):
    for ke, kd in ((1.0, 0.0), (1000.0, 0.0), (1.0e8, 0.0), (1.0, 0.01)):
        for reverse in (False, True):
            with self.subTest(ke=ke, kd=kd, reverse=reverse):
                actual, reference, inertia = _impact(device, 32, reverse, 1, ke, kd)
                np.testing.assert_allclose(actual, reference, atol=0.003, rtol=0.003)
                np.testing.assert_allclose(actual @ inertia, 1.0e-5, atol=3.0e-8)


class TestVBDCompliantLimits(unittest.TestCase):
    pass


def test_revolute_limit_graph(self, device):
    if not device.is_cuda:
        self.skipTest("CUDA graph capture requires a CUDA device")
    eager, _, _ = _impact(device, 32, True, 1, 1.0e8, 2.0e4)
    captured, _, _ = _impact(device, 32, True, 1, 1.0e8, 2.0e4, graph=True)
    np.testing.assert_allclose(captured, eager, atol=1.0e-5, rtol=1.0e-5)


for _device in get_test_devices():
    add_function_test(TestVBDCompliantLimits, "test_revolute_limit_graph", test_revolute_limit_graph, devices=[_device])
    add_function_test(
        TestVBDCompliantLimits, "test_revolute_limit_momentum", test_revolute_limit_momentum, devices=[_device]
    )
    add_function_test(
        TestVBDCompliantLimits,
        "test_revolute_limit_compliant_reference",
        test_revolute_limit_compliant_reference,
        devices=[_device],
    )


if __name__ == "__main__":
    unittest.main()
