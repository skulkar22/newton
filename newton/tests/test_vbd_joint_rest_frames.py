# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverVBD joint rest frames decoupled from the model's joint frames."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()

_ANGLE = math.radians(20.0)


def _build_pair(device):
    """Build a world-fixed link with a second link welded to it 0.5 m along x."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(np.eye(3) * 1.0e-2)
    a = builder.add_link(xform=wp.transform_identity(), mass=1.0, inertia=inertia)
    b = builder.add_link(xform=wp.transform((0.5, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=inertia)
    j0 = builder.add_joint_fixed(parent=-1, child=a)
    j1 = builder.add_joint_fixed(
        parent=a,
        child=b,
        parent_xform=wp.transform((0.25, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform((-0.25, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j0, j1])
    builder.color()
    return builder.finalize(device=device), b, j1


def _rotate_parent_frame(model, joint):
    """Rotate the joint's parent frame about z by ``_ANGLE``, as an endpoint frame update would."""
    X_p = model.joint_X_p.numpy()
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), _ANGLE)
    X_p[joint, 3:7] = np.array([q[0], q[1], q[2], q[3]])
    model.joint_X_p.assign(X_p)


def _child_yaw(model, solver, body, steps=60):
    """Step the model and return the child body's rotation about z [rad]."""
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, 1.0e-3)
        state_0, state_1 = state_1, state_0
    q = state_0.body_q.numpy()[body, 3:7].astype(np.float64)
    return 2.0 * math.atan2(q[2], q[3])


def _run(device, solve, pin_rest):
    """Rotate the parent frame of the welded joint, optionally after pinning the rest frames, and simulate."""
    model, body, joint = _build_pair(device)
    solver = newton.solvers.SolverVBD(model, iterations=20, rigid_articulation_solve=solve)
    if pin_rest:
        solver.set_joint_rest_frames(joint_X_p_rest=model.joint_X_p, joint_X_c_rest=model.joint_X_c)
    _rotate_parent_frame(model, joint)
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    return _child_yaw(model, solver, body)


def _rest_frames_follow_model_by_default(test, device):
    """Keep the child at rest when a joint frame edit moves the current and rest frames together."""
    for solve in ("local", "block_sparse_joints"):
        with test.subTest(solve=solve):
            yaw = _run(device, solve, pin_rest=False)
            test.assertAlmostEqual(yaw, 0.0, delta=1.0e-3)


def _pinned_rest_frames_keep_the_original_target(test, device):
    """Drive the child to the original relative rotation when the rest frames are pinned.

    Pinning is checked on the local and the block-sparse articulation solves; without the
    pinned frames the kernels would measure the error against the rotated frame and leave
    the child at rest.
    """
    for solve in ("local", "block_sparse_joints"):
        with test.subTest(solve=solve):
            yaw = _run(device, solve, pin_rest=True)
            # The child frame keeps its original relative rotation to the rotated parent frame.
            test.assertAlmostEqual(yaw, _ANGLE, delta=2.0e-3)


def _set_joint_rest_frames_copies_and_resets(test, device):
    """Copy pinned frames, follow the model again after a reset, and reject wrongly sized arrays."""
    model, _body, joint = _build_pair(device)
    solver = newton.solvers.SolverVBD(model, iterations=2)
    solver.set_joint_rest_frames(joint_X_p_rest=model.joint_X_p)
    test.assertIsNot(solver.joint_X_p_rest, model.joint_X_p)
    test.assertIs(solver.joint_X_c_rest, model.joint_X_c)
    before = solver.joint_X_p_rest.numpy().copy()
    _rotate_parent_frame(model, joint)
    np.testing.assert_array_equal(solver.joint_X_p_rest.numpy(), before)
    solver.set_joint_rest_frames()
    test.assertIs(solver.joint_X_p_rest, model.joint_X_p)
    with test.assertRaises(ValueError):
        solver.set_joint_rest_frames(joint_X_p_rest=wp.zeros(model.joint_count + 1, dtype=wp.transform, device=device))


class TestVBDJointRestFrames(unittest.TestCase):
    pass


add_function_test(
    TestVBDJointRestFrames,
    "test_rest_frames_follow_model_by_default",
    _rest_frames_follow_model_by_default,
    devices=devices,
)
add_function_test(
    TestVBDJointRestFrames,
    "test_pinned_rest_frames_keep_the_original_target",
    _pinned_rest_frames_keep_the_original_target,
    devices=devices,
)
add_function_test(
    TestVBDJointRestFrames,
    "test_set_joint_rest_frames_copies_and_resets",
    _set_joint_rest_frames_copies_and_resets,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
