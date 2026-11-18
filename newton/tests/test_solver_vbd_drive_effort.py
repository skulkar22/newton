# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check actuator saturation separately from passive joint reactions."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.test_reduced_elastic_sparse_articulation import _make_leaf_link_model
from newton.tests.test_solver_vbd_joint_forces import _make_joint_model
from newton.tests.unittest_utils import add_function_test, get_test_devices

_DT = 1.0 / 240.0
_SOLVES = ("local", "block_sparse_joints")


def _step(model, *, alm, solve, target=0.0, target_velocity=0.0, feedforward=0.0):
    """Advance one step and reconstruct the motor and total joint efforts."""
    solver = newton.solvers.SolverVBD(
        model, iterations=32, rigid_compliant_alm=alm, rigid_avbd_alpha=0.0, rigid_articulation_solve=solve
    )
    state, output = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control = model.control()
    control.joint_target_q.fill_(target)
    control.joint_target_qd.fill_(target_velocity)
    control.joint_f.fill_(feedforward)
    previous = wp.clone(state.body_q)
    previous_velocity = wp.clone(state.joint_qd)
    solver.step(state, output, control, None, _DT)
    effort = wp.empty_like(output.joint_qd)
    wrench = wp.empty(model.joint_count, dtype=wp.spatial_vector, device=model.device)
    solver.eval_joint_forces(
        output,
        wrench,
        body_q_prev=previous,
        dt=_DT,
        control=control,
        joint_effort=effort,
        joint_qd_prev=previous_velocity,
    )
    return solver, output, effort.numpy(), wrench.numpy()


def test_vbd_drive_effort_saturation(test, device):
    """Bound both spring and velocity-drive effort in each rigid solve."""
    for solve in _SOLVES:
        for alm in (False, True):
            for joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
                for damping_drive in (False, True):
                    for limit in (0.0, 2.0):
                        for direction in (-1.0, 1.0):
                            with test.subTest(
                                solve=solve,
                                alm=alm,
                                joint_type=joint_type,
                                damping=damping_drive,
                                limit=limit,
                                direction=direction,
                            ):
                                model = _make_joint_model(
                                    device,
                                    joint_type,
                                    effort_limit=limit,
                                    target_ke=0.0 if damping_drive else 1000.0,
                                    target_kd=100.0 if damping_drive else 0.0,
                                )
                                solver, output, effort, _wrench = _step(
                                    model,
                                    alm=alm,
                                    solve=solve,
                                    target=direction,
                                    target_velocity=direction if damping_drive else 0.0,
                                )
                                expected = direction * limit
                                test.assertAlmostEqual(float(effort[0]), expected, delta=2.0e-4)
                                test.assertAlmostEqual(float(output.joint_qd.numpy()[0]), _DT * expected, delta=2.0e-5)
                                test.assertLessEqual(abs(float(solver.joint_drive_lambda.numpy()[0])), limit + 1.0e-5)


def test_vbd_drive_effort_unsaturated(test, device):
    """Retain the implicit PD response below a finite or unlimited effort bound."""
    for solve in _SOLVES:
        for alm in (False, True):
            for limit in (10.0, float("inf")):
                model = _make_joint_model(
                    device, newton.JointType.PRISMATIC, effort_limit=limit, target_ke=4.0, target_kd=2.0
                )
                _solver, output, effort, _wrench = _step(model, alm=alm, solve=solve, target=0.5)
                velocity = _DT * 2.0 / (1.0 + 2.0 * _DT + 4.0 * _DT**2)
                test.assertAlmostEqual(float(output.joint_qd.numpy()[0]), velocity, delta=2.0e-5)
                test.assertAlmostEqual(float(effort[0]), velocity / _DT, delta=2.0e-4)


def test_vbd_drive_effort_stiff_target(test, device):
    """Converge a stiff motor that starts saturated near its target."""
    for solve in _SOLVES:
        for alm in (False, True):
            model = _make_joint_model(device, newton.JointType.PRISMATIC, effort_limit=2.0, target_ke=1.0e8)
            _solver, output, effort, _wrench = _step(model, alm=alm, solve=solve, target=1.0e-6)
            velocity = _DT * 100.0 / (1.0 + 1.0e8 * _DT**2)
            test.assertAlmostEqual(float(output.joint_qd.numpy()[0]), velocity, delta=1.0e-6)
            test.assertAlmostEqual(float(effort[0]), velocity / _DT, delta=2.0e-4)


def test_vbd_drive_effort_passive_terms(test, device):
    """Keep passive damping and feedforward effort outside the motor cap."""
    for solve in _SOLVES:
        for alm in (False, True):
            model = _make_joint_model(
                device, newton.JointType.PRISMATIC, effort_limit=0.0, target_ke=1000.0, damping=4.0
            )
            model.joint_qd.fill_(1.0)
            _solver, output, effort, wrench = _step(model, alm=alm, solve=solve, target=1.0, feedforward=8.0)
            velocity = (1.0 + _DT * 8.0) / (1.0 + _DT * 4.0)
            test.assertAlmostEqual(float(output.joint_qd.numpy()[0]), velocity, delta=2.0e-5)
            # Reported actuator effort includes the separately applied feedforward term.
            test.assertAlmostEqual(float(effort[0]), 8.0, delta=2.0e-4)
            test.assertAlmostEqual(float(wrench[0, 0]), 8.0 - 4.0 * velocity, delta=2.0e-4)


def test_vbd_drive_effort_limit_reaction(test, device):
    """Enforce a stop even when the motor effort limit is zero."""
    for solve in _SOLVES:
        for alm in (False, True):
            model = _make_joint_model(
                device,
                newton.JointType.PRISMATIC,
                effort_limit=0.0,
                target_ke=1000.0,
                limit_lower=-0.2,
                limit_upper=0.2,
                limit_ke=200.0,
                limit_kd=0.0,
            )
            model.joint_q.fill_(0.3)
            _solver, output, effort, wrench = _step(model, alm=alm, solve=solve, target=0.2)
            expected = -200.0 * (float(output.joint_q.numpy()[0]) - 0.2)
            test.assertLess(expected, -10.0)
            test.assertAlmostEqual(float(effort[0]), 0.0, delta=2.0e-4)
            test.assertAlmostEqual(float(wrench[0, 0]), expected, delta=0.01)


def test_vbd_drive_effort_holding(test, device):
    """Balance a saturated drive against an equal external load."""
    for solve in _SOLVES:
        for alm in (False, True):
            model = _make_joint_model(device, newton.JointType.REVOLUTE, effort_limit=2.0, target_ke=1000.0)
            solver, output, effort, wrench = _step(model, alm=alm, solve=solve, target=1.0, feedforward=-2.0)
            test.assertAlmostEqual(float(output.joint_qd.numpy()[0]), 0.0, delta=2.0e-5)
            test.assertAlmostEqual(float(effort[0]), 0.0, delta=2.0e-4)
            test.assertAlmostEqual(float(wrench[0, 5]), 0.0, delta=2.0e-4)
            if alm:
                test.assertAlmostEqual(float(solver.joint_drive_lambda.numpy()[0]), -2.0, delta=2.0e-4)


def test_vbd_d6_drive_effort(test, device):
    """Bound linear and angular D6 motors independently."""
    for solve in _SOLVES:
        for alm in (False, True):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
            config = newton.ModelBuilder.JointDofConfig
            joint = builder.add_joint_d6(
                -1,
                body,
                linear_axes=[config(axis=newton.Axis.X, target_ke=1000.0, effort_limit=2.0)],
                angular_axes=[config(axis=newton.Axis.Z, target_ke=1000.0, effort_limit=3.0)],
            )
            builder.add_articulation([joint])
            builder.color()
            model = builder.finalize(device=device)
            _solver, output, effort, _wrench = _step(model, alm=alm, solve=solve, target=1.0)
            np.testing.assert_allclose(effort, [2.0, 3.0], atol=2.0e-4)
            np.testing.assert_allclose(output.joint_qd.numpy(), _DT * np.array([2.0, 3.0]), atol=2.0e-5)


def test_vbd_drive_effort_live_update(test, device):
    """Read a smaller bound without retaining the previous ALM traction."""
    for solve in _SOLVES:
        model = _make_joint_model(device, newton.JointType.REVOLUTE, effort_limit=2.0, target_ke=1000.0)
        solver, state, _effort, _wrench = _step(model, alm=True, solve=solve, target=1.0)
        control = model.control()
        control.joint_target_q.fill_(1.0)
        for limit in (0.5, 0.0):
            model.joint_effort_limit.fill_(limit)
            output = model.state()
            previous_velocity = float(state.joint_qd.numpy()[0])
            solver.step(state, output, control, None, _DT)
            test.assertLessEqual(abs(float(solver.joint_drive_lambda.numpy()[0])), limit + 1.0e-5)
            test.assertAlmostEqual(float(output.joint_qd.numpy()[0]), previous_velocity + _DT * limit, delta=2.0e-5)
            state = output


def test_vbd_elastic_drive_effort(test, device):
    """Apply the same motor bound to deformed elastic joint endpoints."""
    for backend in ("schur", "coupled"):
        for alm in (False, True):
            for limit in (0.0, 0.05):
                model, state, hinge = _make_leaf_link_model(device, gravity=0.0)
                dof = int(model.joint_qd_start.numpy()[hinge])
                gains = np.zeros(model.joint_dof_count, dtype=np.float32)
                gains[dof] = 1000.0
                model.joint_target_ke.assign(gains)
                model.joint_effort_limit.fill_(limit)
                control = model.control()
                control.joint_target_q.fill_(1.0)
                solver = newton.solvers.SolverVBD(
                    model,
                    iterations=32,
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
                test.assertAlmostEqual(float(effort.numpy()[dof]), limit, delta=2.0e-4)
                test.assertLessEqual(abs(float(solver.joint_drive_lambda.numpy()[dof])), limit + 1.0e-5)


class TestSolverVBDDriveEffort(unittest.TestCase):
    pass


for test_function in (
    test_vbd_drive_effort_saturation,
    test_vbd_drive_effort_unsaturated,
    test_vbd_drive_effort_stiff_target,
    test_vbd_drive_effort_passive_terms,
    test_vbd_drive_effort_limit_reaction,
    test_vbd_drive_effort_holding,
    test_vbd_d6_drive_effort,
    test_vbd_drive_effort_live_update,
    test_vbd_elastic_drive_effort,
):
    add_function_test(TestSolverVBDDriveEffort, test_function.__name__, test_function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
