# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check requested drive targets independently of joint-limit reactions."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.test_reduced_elastic_sparse_articulation import _make_leaf_link_model
from newton.tests.test_solver_vbd_drive_effort import _DT, _SOLVES, _step
from newton.tests.test_solver_vbd_joint_forces import _make_joint_model
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_vbd_drive_target_outside_limits(test, device):
    """Use the commanded target before either stop becomes active."""
    for solve in _SOLVES:
        for alm in (False, True):
            for joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
                for direction in (-1.0, 1.0):
                    with test.subTest(solve=solve, alm=alm, joint_type=joint_type, direction=direction):
                        model = _make_joint_model(
                            device,
                            joint_type,
                            target_ke=4.0,
                            effort_limit=10.0,
                            limit_lower=-0.2,
                            limit_upper=0.2,
                            limit_ke=200.0,
                            limit_kd=0.0,
                        )
                        _solver, output, effort, _wrench = _step(model, alm=alm, solve=solve, target=direction)
                        velocity = direction * _DT * 4.0 / (1.0 + 4.0 * _DT**2)
                        test.assertAlmostEqual(float(output.joint_qd.numpy()[0]), velocity, delta=2.0e-5)
                        test.assertAlmostEqual(float(effort[0]), velocity / _DT, delta=2.0e-4)


def test_vbd_drive_target_stop_balance(test, device):
    """Balance motor and stop forces with saturated and unsaturated drives."""
    for solve in _SOLVES:
        for alm in (False, True):
            for joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
                for direction in (-1.0, 1.0):
                    for bound in (2.0, 10.0):
                        with test.subTest(
                            solve=solve, alm=alm, joint_type=joint_type, direction=direction, bound=bound
                        ):
                            model = _make_joint_model(
                                device,
                                joint_type,
                                target_ke=4.0,
                                target_kd=2.0,
                                effort_limit=bound,
                                limit_lower=-0.2,
                                limit_upper=0.2,
                                limit_ke=200.0,
                                limit_kd=0.0,
                            )
                            motor = direction * min(bound, (4.0 * (1.0 - 0.2) + 2.0 * 0.1) / (1.0 + 4.0 / 200.0))
                            position = direction * 0.2 + motor / 200.0
                            model.joint_q.fill_(position)
                            _solver, output, effort, wrench = _step(
                                model, alm=alm, solve=solve, target=direction, target_velocity=direction * 0.1
                            )
                            test.assertAlmostEqual(float(output.joint_q.numpy()[0]), position, delta=2.0e-5)
                            test.assertAlmostEqual(float(output.joint_qd.numpy()[0]), 0.0, delta=2.0e-5)
                            test.assertAlmostEqual(float(effort[0]), motor, delta=2.0e-4)
                            np.testing.assert_allclose(wrench, 0.0, atol=2.0e-4)


def test_vbd_drive_target_continuous_angle(test, device):
    """Keep an out-of-range angular target on the same continuous turn."""
    for solve in _SOLVES:
        for alm in (False, True):
            model = _make_joint_model(
                device,
                newton.JointType.REVOLUTE,
                target_ke=4.0,
                effort_limit=10.0,
                limit_lower=2.9,
                limit_upper=3.3,
                limit_ke=200.0,
                limit_kd=0.0,
            )
            position = (4.0 * 4.0 + 200.0 * 3.3) / 204.0
            model.joint_q.fill_(position)
            _solver, output, effort, _wrench = _step(model, alm=alm, solve=solve, target=4.0)
            test.assertAlmostEqual(float(output.joint_q.numpy()[0]), position, delta=2.0e-5)
            test.assertAlmostEqual(float(output.joint_qd.numpy()[0]), 0.0, delta=5.0e-5)
            test.assertAlmostEqual(float(effort[0]), 4.0 * (4.0 - position), delta=2.0e-4)


def test_vbd_elastic_drive_target(test, device):
    """Preserve requested targets at a deformed elastic attachment."""
    for backend in ("schur", "coupled"):
        for alm in (False, True):
            model, state, hinge = _make_leaf_link_model(device, gravity=0.0)
            dof = int(model.joint_qd_start.numpy()[hinge])
            for array, value in (
                (model.joint_target_ke, 4.0),
                (model.joint_limit_lower, -0.2),
                (model.joint_limit_upper, 0.2),
                (model.joint_limit_ke, 200.0),
            ):
                values = array.numpy()
                values[dof] = value
                array.assign(values)
            model.joint_effort_limit.fill_(10.0)
            control = model.control()
            control.joint_target_q.fill_(1.0)
            solver = newton.solvers.SolverVBD(
                model,
                iterations=64,
                rigid_compliant_alm=alm,
                rigid_avbd_alpha=0.0,
                rigid_articulation_solve="block_sparse_joints",
                rigid_articulation_elastic_solve=backend,
            )
            output = model.state()
            previous = wp.clone(state.body_q)
            previous_velocity = wp.clone(state.joint_qd)
            solver.step(state, output, control, None, _DT)
            effort = wp.empty_like(output.joint_qd)
            wrench = wp.empty(model.joint_count, dtype=wp.spatial_vector, device=device)
            solver.eval_joint_forces(
                output,
                wrench,
                body_q_prev=previous,
                dt=_DT,
                control=control,
                joint_effort=effort,
                joint_qd_prev=previous_velocity,
            )
            q = float(output.joint_q.numpy()[int(model.joint_q_start.numpy()[hinge])])
            test.assertAlmostEqual(float(effort.numpy()[dof]), 4.0 * (1.0 - q), delta=2.0e-3)


class TestSolverVBDDriveTargets(unittest.TestCase):
    pass


for test_function in (
    test_vbd_drive_target_outside_limits,
    test_vbd_drive_target_stop_balance,
    test_vbd_drive_target_continuous_angle,
    test_vbd_elastic_drive_target,
):
    add_function_test(TestSolverVBDDriveTargets, test_function.__name__, test_function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
