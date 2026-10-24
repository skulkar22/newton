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
    """Build a world-fixed link with a second link welded to it 0.5 m along x.

    Returns:
        The model, the indices of the first and second links, and the indices of the world
        joint and the joint between the links.
    """
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
    return builder.finalize(device=device), (a, b), (j0, j1)


def _rotated_frames(frames, joint):
    """Return a copy of ``frames`` with the joint's rotation set to ``_ANGLE`` about z."""
    X = frames.numpy().copy()
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), _ANGLE)
    X[joint, 3:7] = np.array([q[0], q[1], q[2], q[3]])
    return X


def _rotate_parent_frame(model, joint):
    """Rotate the joint's parent frame about z by ``_ANGLE``, as an endpoint frame update would."""
    model.joint_X_p.assign(_rotated_frames(model.joint_X_p, joint))


def _yaws(model, solver, steps=60):
    """Step the model and return the rotation of each body about z [rad]."""
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, 1.0e-3)
        state_0, state_1 = state_1, state_0
    q = state_0.body_q.numpy()[:, 3:7].astype(np.float64)
    return 2.0 * np.arctan2(q[:, 2], q[:, 3])


# Each case rotates one joint frame about z by _ANGLE. With the rest frames pinned to the original
# frames, the joint must restore the original relative rotation, which turns the listed body by the
# expected yaw [rad]: (side, joint index into (j0, j1), body index into (a, b), expected pinned yaw).
_FRAME_EDITS = {
    "link_parent_frame": ("parent", 1, 1, _ANGLE),
    "world_parent_frame": ("parent", 0, 0, _ANGLE),
    "link_child_frame": ("child", 1, 1, -_ANGLE),
}


def _run(device, solve, pin_rest, edit="link_parent_frame", rebind=False):
    """Rotate one joint frame, optionally after pinning the rest frames, simulate, and return a yaw.

    Args:
        device: Warp device.
        solve: Articulation solve of :class:`~newton.solvers.SolverVBD`.
        pin_rest: Whether to pin the original joint frames as rest frames first.
        edit: Key of :data:`_FRAME_EDITS` that selects the rotated frame.
        rebind: Whether to replace the model's frame array instead of assigning into it.

    Returns:
        The rotation about z [rad] of the body that the edit moves when pinned.
    """
    model, bodies, joints = _build_pair(device)
    side, joint, body, _expected = _FRAME_EDITS[edit]
    solver = newton.solvers.SolverVBD(model, iterations=20, rigid_articulation_solve=solve)
    if pin_rest:
        solver.set_joint_rest_frames(joint_X_p_rest=model.joint_X_p, joint_X_c_rest=model.joint_X_c)
    attribute = "joint_X_p" if side == "parent" else "joint_X_c"
    frames = _rotated_frames(getattr(model, attribute), joints[joint])
    if rebind:
        setattr(model, attribute, wp.array(frames, dtype=wp.transform, device=device))
    else:
        getattr(model, attribute).assign(frames)
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    return float(_yaws(model, solver)[bodies[body]])


def _rest_frames_follow_model_by_default(test, device):
    """Keep the bodies at rest when a joint frame edit moves the current and rest frames together.

    The model's frames are followed whether they are edited in place or the model's array is
    replaced after the solver was created.
    """
    for solve in ("local", "block_sparse_joints"):
        for edit in _FRAME_EDITS:
            for rebind in (False, True):
                with test.subTest(solve=solve, edit=edit, rebind=rebind):
                    yaw = _run(device, solve, pin_rest=False, edit=edit, rebind=rebind)
                    test.assertAlmostEqual(yaw, 0.0, delta=1.0e-3)


def _pinned_rest_frames_keep_the_original_target(test, device):
    """Drive the bodies to the original relative rotation when the rest frames are pinned.

    Pinning is checked on the local and the block-sparse articulation solves, for the parent
    frame of a joint between links, the parent frame of a joint to the world, and the child
    frame of a joint; without the pinned frames the kernels would measure the error against
    the rotated frame and leave the bodies at rest.
    """
    for solve in ("local", "block_sparse_joints"):
        for edit, (_side, _joint, _body, expected) in _FRAME_EDITS.items():
            with test.subTest(solve=solve, edit=edit):
                yaw = _run(device, solve, pin_rest=True, edit=edit)
                test.assertAlmostEqual(yaw, expected, delta=2.0e-3)


def _set_joint_rest_frames_copies_and_resets(test, device):
    """Copy pinned frames into stable buffers, follow the model again after a reset, and validate input."""
    model, _bodies, joints = _build_pair(device)
    joint = joints[1]
    solver = newton.solvers.SolverVBD(model, iterations=2)
    solver.set_joint_rest_frames(joint_X_p_rest=model.joint_X_p)
    test.assertIsNot(solver.joint_X_p_rest, model.joint_X_p)
    test.assertIs(solver.joint_X_c_rest, model.joint_X_c)
    pinned = solver.joint_X_p_rest
    before = pinned.numpy().copy()
    _rotate_parent_frame(model, joint)
    np.testing.assert_array_equal(solver.joint_X_p_rest.numpy(), before)

    # Pinning again copies into the same buffer, so captured launches see the new frames.
    solver.set_joint_rest_frames(joint_X_p_rest=model.joint_X_p)
    test.assertIs(solver.joint_X_p_rest, pinned)
    test.assertEqual(solver.joint_X_p_rest.ptr, pinned.ptr)
    np.testing.assert_array_equal(solver.joint_X_p_rest.numpy(), model.joint_X_p.numpy())

    solver.set_joint_rest_frames()
    test.assertIs(solver.joint_X_p_rest, model.joint_X_p)
    solver.set_joint_rest_frames(joint_X_p_rest=wp.array(before, dtype=wp.transform, device=device))
    test.assertEqual(solver.joint_X_p_rest.ptr, pinned.ptr)
    solver.set_joint_rest_frames()

    # An unpinned side resolves the model's array at use, also after the array is replaced.
    replacement = wp.clone(model.joint_X_c)
    model.joint_X_c = replacement
    test.assertIs(solver.joint_X_c_rest, replacement)

    wrong_inputs = {
        "too long": wp.zeros(model.joint_count + 1, dtype=wp.transform, device=device),
        "vec3 dtype": wp.zeros(model.joint_count, dtype=wp.vec3, device=device),
        "numpy": model.joint_X_p.numpy(),
    }
    for label, array in wrong_inputs.items():
        with test.subTest(input=label):
            with test.assertRaises(ValueError):
                solver.set_joint_rest_frames(joint_X_p_rest=array)
            with test.assertRaises(ValueError):
                solver.set_joint_rest_frames(joint_X_c_rest=array)
    test.assertIs(solver.joint_X_p_rest, model.joint_X_p)


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
