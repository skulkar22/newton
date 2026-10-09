# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the circular penalty formulation against the signed damping contract."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import Axis, TendonLinkType
from newton.tests.test_tendon_vbd import _make_tendon_vbd_solver
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_damped_tendon_matches_unilateral_force_law(test, device):
    """Match implicit signed spring-plus-damper force, including its inactive branch."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        anchor = builder.add_body(mass=0.0, is_kinematic=True)
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.25, 0.0, 0.0)),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_tendon()
        builder.add_tendon_link(body=anchor, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=body,
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=1.0e-3,
            damping=100.0,
            rest_length=1.0,
        )
        builder.color()
        model = builder.finalize()
        dt = 1.0 / 120.0
        for solver_name in ("xpbd", "vbd"):
            for velocity in (-3.0, 0.0, 3.0, 30.0):
                with test.subTest(solver=solver_name, velocity=velocity):
                    if solver_name == "xpbd":
                        solver = newton.solvers.SolverXPBD(model, iterations=20, joint_linear_relaxation=1.0)
                    else:
                        solver = _make_tendon_vbd_solver(model)
                    state_0, state_1 = model.state(), model.state()
                    state_0.body_qd.assign(
                        [wp.spatial_vector(0.0), wp.spatial_vector(velocity, 0.0, 0.0, 0.0, 0.0, 0.0)]
                    )
                    solver.step(state_0, state_1, model.control(), None, dt)

                    # x = x_free - dt^2*T/m, T = max((x-rest)/C + d*(x-x0)/dt, 0).
                    # XPBD's signed row lets fast extension transmit damping tension
                    # before geometric tautness. VBD keeps a slack span force-free
                    # (see test_vbd_slack_damped_tendon_remains_force_free).
                    free_length = 0.25 + dt * velocity
                    trial_tension = (free_length - 1.0) / 1.0e-3 + 100.0 * velocity
                    if solver_name == "vbd" and free_length <= 1.0:
                        trial_tension = 0.0
                    tension = max(trial_tension, 0.0) / (1.0 + dt * dt / 1.0e-3 + dt * 100.0)
                    expected_velocity = velocity - dt * tension
                    test.assertAlmostEqual(float(state_1.body_qd.numpy()[body][0]), expected_velocity, delta=1.0e-4)
                    test.assertEqual(float(solver.tendon_seg_material_tension.numpy()[0]), 0.0)
                    reported_lambda = float(solver.tendon_seg_lambda.numpy()[0])
                    if solver_name == "vbd":
                        test.assertEqual(reported_lambda, 0.0, "VBD must not report a synthetic XPBD multiplier")
                    else:
                        test.assertAlmostEqual(-reported_lambda / dt, tension, delta=1.0e-3)


class TestTendonCircularDamping(unittest.TestCase):
    pass


for device in get_test_devices():
    add_function_test(
        TestTendonCircularDamping,
        "test_damped_tendon_matches_unilateral_force_law",
        test_damped_tendon_matches_unilateral_force_law,
        devices=[device],
    )

if __name__ == "__main__":
    unittest.main()
