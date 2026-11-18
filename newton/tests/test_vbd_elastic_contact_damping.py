# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-sided contact damping on contacts between rigid shapes and reduced elastic bodies."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests._reduced_elastic_fixtures import elastic_free_joint_starts
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


def _elastic_box_on_ground_velocity(
    test, device, vertical_velocity: float, bidirectional: bool, iterations: int
) -> float:
    """Step a penetrating dynamic elastic box once and return its vertical velocity after the step.

    The box (half extent 5 cm, one massive mode) starts 1 cm into a ground plane with ``kd > 0``,
    moving at ``vertical_velocity``. Every contact row has an elastic sample on the box side.
    """

    def downward_shape_fn(_x):
        return np.array([[0.0, 0.0, -1.0]], dtype=np.float32)

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.ke = 1000.0
    cfg.kd = 50.0
    cfg.mu = 0.0
    cfg.margin = 0.0
    cfg.gap = 0.0

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis="Z")
    builder.add_ground_plane(cfg=cfg)
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.04), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3, dtype=np.float32)),
        mode_count=1,
        mode_mass=[1.0],
        mode_stiffness=[100.0],
        mode_damping=[0.0],
        mode_q=[0.0],
        mode_shape_fn=downward_shape_fn,
    )
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    _q_start, qd_start = elastic_free_joint_starts(model, body)
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[qd_start + 2] = vertical_velocity
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    pipeline = newton.CollisionPipeline(model, broad_phase="explicit", deterministic=True)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    count = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
    elastic_rows = (contacts.rigid_contact_elastic_sample0.numpy()[:count] >= 0) | (
        contacts.rigid_contact_elastic_sample1.numpy()[:count] >= 0
    )
    test.assertGreater(count, 0)
    test.assertTrue(bool(elastic_rows.all()), "expected only rigid-elastic contact rows")

    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_compliant_alm=True,
        rigid_avbd_contact_alpha=0.0,
        rigid_body_contact_buffer_size=128,
        elastic_body_relaxation=1.0,
        rigid_contact_damping_bidirectional=bidirectional,
    )
    test.assertIs(solver.rigid_contact_damping_bidirectional, bidirectional)
    solver.step(state_0, state_1, model.control(), contacts, 0.01)
    velocity = float(state_1.body_qd.numpy()[body][2])
    test.assertTrue(np.isfinite(velocity))
    return velocity


def test_elastic_contact_damping_bidirectional(test, device):
    """The flag also damps a separating rigid-elastic contact and leaves a closing one unchanged.

    The flag reaches two assemblies of the same contact rows (the rigid per-body contact accumulation
    and the elastic block); passing it to only one of them moves the result far outside the asserted
    band. Measured on CPU and CUDA, separating at 0.1 m/s with 4 iterations: 0.329 m/s with
    closing-only damping, 0.121 m/s with two-sided damping (-0.83 and 0.74 m/s when the flag reaches
    only the rigid or only the elastic assembly). Closing at -0.1 m/s with 1 iteration (no iterate
    separates yet): -0.0248 m/s for both settings.
    """
    closing_only = _elastic_box_on_ground_velocity(test, device, 0.1, False, iterations=4)
    two_sided = _elastic_box_on_ground_velocity(test, device, 0.1, True, iterations=4)
    test.assertGreater(closing_only, 0.0)
    test.assertGreater(two_sided, 0.0)
    test.assertLess(two_sided, 0.5 * closing_only)

    closing_only = _elastic_box_on_ground_velocity(test, device, -0.1, False, iterations=1)
    two_sided = _elastic_box_on_ground_velocity(test, device, -0.1, True, iterations=1)
    test.assertLess(closing_only, 0.0)
    test.assertAlmostEqual(two_sided, closing_only, delta=1.0e-5)


class TestVBDElasticContactDamping(unittest.TestCase):
    pass


add_function_test(
    TestVBDElasticContactDamping,
    "test_elastic_contact_damping_bidirectional",
    test_elastic_contact_damping_bidirectional,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
