# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check optional direct loading of explicit torques in VBD solves."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.test_reduced_elastic_sparse_articulation import _make_leaf_link_model
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _fixed_pair(device, dt, *, explicit=None, iterations=40):
    """Apply equal and opposite torques to a fixed pair with unequal inertias."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = np.array([1.0e-4, 0.1])
    bodies = [builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * value)) for value in inertia]
    root = builder.add_joint_free(child=bodies[0])
    fixed = builder.add_joint_fixed(bodies[0], bodies[1])
    builder.add_articulation([root, fixed])
    builder.color()
    model = builder.finalize(device=device)
    kwargs = {} if explicit is None else {"rigid_explicit_torque_in_solve": explicit}
    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_articulation_solve="block_sparse_joints",
        rigid_articulation_relaxation=1.0,
        rigid_joint_angular_ke=1.0e6,
        **kwargs,
    )
    state, output = model.state(), model.state()
    forces = np.array([[0, 0, 0, 0, 0, 50], [0, 0, 0, 0, 0, -50]], dtype=np.float32)
    state.body_f.assign(forces)
    solver.step(state, output, model.control(), None, dt)
    np.testing.assert_array_equal(state.body_f.numpy(), forces)
    return inertia, output


def test_vbd_explicit_torque_momentum(test, device):
    """Conserve pair momentum across time steps when the free predictor would wrap."""
    for dt in (0.00125, 0.0025, 0.005, 0.01):
        for iterations in (5, 40):
            with test.subTest(dt=dt, iterations=iterations):
                inertia, output = _fixed_pair(device, dt, explicit=True, iterations=iterations)
                test.assertLess(abs(float(inertia @ output.body_qd.numpy()[:, 5])), 1.0e-6)
                pose = output.body_q.numpy()
                angle = 2.0 * np.arctan2(pose[:, 5], pose[:, 6])
                reduced = np.prod(inertia) / np.sum(inertia)
                expected = 50.0 * dt**2 / (reduced + 1.0e6 * dt**2)
                test.assertAlmostEqual(float(angle[0] - angle[1]), expected, delta=1.0e-6)


def test_vbd_explicit_torque_default(test, device):
    """Preserve the original predictor unless the option is enabled."""
    _inertia, default = _fixed_pair(device, 0.01)
    _inertia, disabled = _fixed_pair(device, 0.01, explicit=False)
    np.testing.assert_array_equal(default.body_q.numpy(), disabled.body_q.numpy())
    np.testing.assert_array_equal(default.body_qd.numpy(), disabled.body_qd.numpy())


def test_vbd_explicit_torque_actuation(test, device):
    """Load explicit actuator and external torques once while retaining translation and armature."""
    dt = 0.01
    for solve in ("local", "block_sparse_joints"):
        for armature in (0.0, 0.4) if solve == "block_sparse_joints" else (0.0,):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.3))
            joint = builder.add_joint_revolute(-1, body, axis=newton.Axis.Z, armature=armature)
            builder.add_articulation([joint])
            builder.color()
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverVBD(
                model, iterations=40, rigid_articulation_solve=solve, rigid_explicit_torque_in_solve=True
            )
            state, output = model.state(), model.state()
            state.body_f.assign(np.array([[0, 0, 0, 0, 0, 1.0]], dtype=np.float32))
            control = model.control()
            control.joint_f.fill_(5.2)
            solver.step(state, output, control, None, dt)
            test.assertAlmostEqual(float(output.joint_qd.numpy()[0]), 6.2 * dt / (0.3 + armature), delta=2.0e-5)
            test.assertAlmostEqual(float(state.body_f.numpy()[0, 5]), 1.0)


def test_vbd_explicit_torque_free_body(test, device):
    """Recover small-step free-body motion without moving translational force out of the predictor."""
    dt = 0.001
    for solve in ("local", "block_sparse_joints"):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_body(mass=2.0, inertia=wp.mat33(np.eye(3) * 0.3))
        builder.color()
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=8,
            rigid_articulation_solve=solve,
            rigid_articulation_relaxation=1.0,
            rigid_explicit_torque_in_solve=True,
        )
        state, output = model.state(), model.state()
        state.body_f.assign(np.array([[4.0, 0, 0, 0, 0, 3.0]], dtype=np.float32))
        solver.step(state, output, model.control(), None, dt)
        test.assertAlmostEqual(float(output.body_q.numpy()[0, 0]), 2.0 * dt**2, delta=1.0e-8)
        np.testing.assert_allclose(output.body_qd.numpy()[0], [2.0 * dt, 0, 0, 0, 0, 10.0 * dt], atol=1.0e-6)


def test_vbd_explicit_torque_elastic(test, device):
    """Retain the converged elastic response when rotational predictions are small."""
    for solve, backend in (("local", "coupled"), ("block_sparse_joints", "schur"), ("block_sparse_joints", "coupled")):
        series = []
        for explicit in (False, True):
            model, state, _hinge = _make_leaf_link_model(device, gravity=0.0)
            solver = newton.solvers.SolverVBD(
                model,
                iterations=80,
                rigid_articulation_solve=solve,
                rigid_articulation_elastic_solve=backend,
                rigid_explicit_torque_in_solve=explicit,
            )
            torque = np.zeros((model.body_count, 6), dtype=np.float32)
            torque[:, 5] = [-0.1, 0.1]
            state.body_f.assign(torque)
            output = model.state()
            solver.step(state, output, model.control(), None, 0.001)
            series.append(np.concatenate((output.body_q.numpy().ravel(), output.joint_q.numpy())))
        np.testing.assert_allclose(series[0], series[1], atol=1.0e-5)


class TestSolverVBDExplicitTorque(unittest.TestCase):
    pass


for test_function in (
    test_vbd_explicit_torque_momentum,
    test_vbd_explicit_torque_default,
    test_vbd_explicit_torque_actuation,
    test_vbd_explicit_torque_free_body,
    test_vbd_explicit_torque_elastic,
):
    add_function_test(TestSolverVBDExplicitTorque, test_function.__name__, test_function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
