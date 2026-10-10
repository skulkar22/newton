# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal joint drives must not move a free system's center of mass in the sparse VBD solve.

A free two-link articulation (1 kg + 2 kg) driven by an internal sinusoidal hinge target in zero
gravity has no external force, so its center of mass must stay fixed. Accumulating absolute float32
body positions during the sparse solve turned rounding into unbalanced linear momentum (several mm
of drift over 2 s at dt = 1 ms); the displacement-form update keeps it below 1 um.
"""

import unittest

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _com_drift(device, dt, seconds):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    a = b.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.1), lock_inertia=True)
    c = b.add_link(mass=2.0, inertia=wp.mat33(np.eye(3) * 0.2), lock_inertia=True)
    root = b.add_joint_free(a)
    hinge = b.add_joint_revolute(
        a,
        c,
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
        axis=newton.Axis.Z,
        target_ke=300.0,
        target_kd=8.0,
    )
    b.add_articulation([root, hinge])
    b.color()
    previous = newton.use_coord_layout_targets
    newton.use_coord_layout_targets = True
    try:
        model = b.finalize(device=device)
    finally:
        newton.use_coord_layout_targets = previous
    solver = newton.solvers.SolverVBD(
        model,
        iterations=12,
        rigid_articulation_solve="block_sparse_joints",
        rigid_articulation_relaxation=0.65,
        rigid_joint_linear_ke=1.0e6,
        rigid_joint_angular_ke=1.0e6,
        rigid_joint_linear_kd=500.0,
        rigid_joint_angular_kd=500.0,
    )
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    mass = model.body_mass.numpy().astype(np.float64)
    com0 = (mass[:, None] * s0.body_q.numpy()[:, :3]).sum(0) / mass.sum()
    control = model.control()
    target = control.joint_target_q.numpy()
    qi = b.joint_q_start[hinge]
    for step in range(round(seconds / dt)):
        target[qi] = 0.3 * np.sin(2.0 * np.pi * 0.6 * step * dt)
        control.joint_target_q.assign(target)
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
    com = (mass[:, None] * s0.body_q.numpy()[:, :3]).sum(0) / mass.sum()
    return float(np.linalg.norm(com - com0))


def test_vbd_sparse_internal_drive_conserves_com(test, device):
    drift = _com_drift(device, dt=0.001, seconds=2.0)
    test.assertLess(drift, 1.0e-5, f"COM drift {drift:.3e} m")


class TestSolverVBDMomentum(unittest.TestCase):
    pass


add_function_test(
    TestSolverVBDMomentum,
    "test_vbd_sparse_internal_drive_conserves_com",
    test_vbd_sparse_internal_drive_conserves_com,
    devices=get_test_devices(),
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
