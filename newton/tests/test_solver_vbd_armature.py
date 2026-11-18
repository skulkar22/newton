# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check relative-coordinate rotor inertia in local and sparse VBD."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

_DT = 0.01


def _step(
    device,
    solve,
    *,
    parent_kind="dynamic",
    armature=0.4,
    torque=0.0,
    ke=0.0,
    kd=0.0,
    velocities=(0.0, 0.0),
    rotated=False,
    iterations=80,
    alm=True,
    external=0.0,
):
    """Step two aligned spherical inertias about a common hinge."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.7) if rotated else wp.quat_identity()
    pose = wp.transform(wp.vec3(0.0), rotation)
    parent = -1
    joints = []
    if parent_kind != "world":
        parent = builder.add_link(
            xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3) * 0.2), is_kinematic=parent_kind == "kinematic"
        )
        joints.append(builder.add_joint_free(child=parent))
    child = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3) * 0.3))
    joint = builder.add_joint_revolute(
        parent,
        child,
        axis=newton.Axis.Y,
        parent_xform=pose if parent < 0 else wp.transform_identity(),
        target_ke=ke,
        target_kd=kd,
        limit_ke=0.0,
        limit_kd=0.0,
        armature=armature,
    )
    joints.append(joint)
    builder.add_articulation(joints)
    builder.color()
    model = builder.finalize(device=device)
    state, output = model.state(), model.state()
    axis = np.asarray(wp.quat_rotate(rotation, wp.vec3(0.0, 1.0, 0.0)))
    velocity = np.zeros((model.body_count, 6), dtype=np.float32)
    if parent >= 0:
        velocity[parent, 3:] = axis * velocities[0]
    velocity[child, 3:] = axis * velocities[1]
    state.body_qd.assign(velocity)
    forces = np.zeros_like(velocity)
    forces[child, 3:] = axis * external
    state.body_f.assign(forces)
    control = model.control()
    dof = int(model.joint_qd_start.numpy()[joint])
    targets = control.joint_target_q.numpy()
    targets[int(model.joint_target_q_start.numpy()[joint])] = 0.001
    control.joint_target_q.assign(targets)
    efforts = control.joint_f.numpy()
    efforts[dof] = torque
    control.joint_f.assign(efforts)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_articulation_solve=solve,
        rigid_compliant_alm=alm,
        rigid_articulation_relaxation=1.0,
        rigid_avbd_alpha=0.0,
    )
    solver.step(state, output, control, None, _DT)
    angular = output.body_qd.numpy()[:, 3:] @ axis
    rates = np.array([angular[parent] if parent >= 0 else 0.0, angular[child]])
    return rates, model, solver, control, output, dof


def test_vbd_revolute_armature_torque(test, device):
    """Match augmented inertia and conserve pair momentum for explicit torques."""
    for solve in ("local", "block_sparse_joints"):
        for parent_kind in ("world", "kinematic", "dynamic"):
            for rotated in (False, True):
                for armature in (0.0, 0.4):
                    with test.subTest(solve=solve, parent=parent_kind, rotated=rotated, armature=armature):
                        rates, *_ = _step(
                            device, solve, parent_kind=parent_kind, rotated=rotated, armature=armature, torque=5.2
                        )
                        inertia = 0.12 if parent_kind == "dynamic" else 0.3
                        test.assertAlmostEqual(
                            float(rates[1] - rates[0]), 5.2 * _DT / (inertia + armature), delta=2.0e-5
                        )
                        if parent_kind == "dynamic":
                            test.assertAlmostEqual(float(0.2 * rates[0] + 0.3 * rates[1]), 0.0, delta=2.0e-6)
                        else:
                            test.assertAlmostEqual(float(rates[0]), 0.0, delta=2.0e-6)


def test_vbd_revolute_armature_drive(test, device):
    """Use the same relative inertia for implicit position and velocity drives."""
    for solve in ("local", "block_sparse_joints"):
        for alm in (False, True):
            rates, *_ = _step(device, solve, ke=1000.0, kd=2.0, alm=alm)
            expected = _DT / (0.12 + 0.4 + 2.0 * _DT + 1000.0 * _DT**2)
            test.assertAlmostEqual(float(rates[1] - rates[0]), expected, delta=2.0e-5)
            test.assertAlmostEqual(float(0.2 * rates[0] + 0.3 * rates[1]), 0.0, delta=2.0e-6)


def test_vbd_revolute_armature_unforced(test, device):
    """Preserve common-mode and relative motion in the absence of applied torque."""
    for solve in ("local", "block_sparse_joints"):
        for velocities in ((0.3, 0.3), (-0.1, 0.2)):
            rates, *_ = _step(device, solve, velocities=velocities)
            np.testing.assert_allclose(rates, velocities, atol=2.0e-5, rtol=0.0)


def test_vbd_revolute_armature_balance(test, device):
    """Keep the force-free rotor reference independent of balanced external torques."""
    for solve in ("local", "block_sparse_joints"):
        rates, *_ = _step(device, solve, parent_kind="world", torque=-20.0, external=20.0)
        np.testing.assert_allclose(rates, 0.0, atol=2.0e-5)


def test_vbd_local_armature_convergence(test, device):
    """Converge local rotor coupling toward the block-sparse result."""
    reference = _step(device, "block_sparse_joints", ke=1000.0)[0]
    errors = [np.linalg.norm(_step(device, "local", ke=1000.0, iterations=n)[0] - reference) for n in (1, 5, 20, 80)]
    test.assertGreater(errors[0], 1.0e-4)
    test.assertLess(errors[1], errors[0])
    test.assertLess(errors[2], errors[1])
    test.assertLess(errors[-1], 2.0e-5)


def test_vbd_local_armature_live_updates(test, device):
    """Read an armature value enabled after solver construction."""
    rates, model, solver, control, state, dof = _step(device, "local", armature=0.0, torque=5.2)
    values = model.joint_armature.numpy()
    values[dof] = 0.4
    model.joint_armature.assign(values)
    output = model.state()
    solver.step(state, output, control, None, _DT)
    expected = rates[1] - rates[0] + 5.2 * _DT / (0.12 + 0.4)
    test.assertAlmostEqual(float(output.joint_qd.numpy()[dof]), expected, delta=2.0e-5)


class TestSolverVBDArmature(unittest.TestCase):
    pass


for test_function in (
    test_vbd_revolute_armature_torque,
    test_vbd_revolute_armature_drive,
    test_vbd_revolute_armature_unforced,
    test_vbd_revolute_armature_balance,
    test_vbd_local_armature_convergence,
    test_vbd_local_armature_live_updates,
):
    add_function_test(TestSolverVBDArmature, test_function.__name__, test_function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
