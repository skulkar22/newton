# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def source_builder():
    """Build an equilibrium cable with an explicit fixed anchor body."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    anchor = builder.add_body(xform=wp.transform((0.2, 0.0, 0.0), wp.quat_identity()), mass=0.0)
    body = builder.add_body(xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)
    builder.add_tendon()
    builder.add_tendon_link(anchor, link_type=newton.TendonLinkType.ATTACHMENT)
    builder.add_tendon_link(
        body, link_type=newton.TendonLinkType.ATTACHMENT, offset=(0.1, 0.0, 0.0), compliance=0.01, rest_length=0.9
    )
    builder.color()
    return builder


def test_replicated_tendon_indices_and_transforms(test, device):
    """Preserve tendon topology and equilibrium under rotated world replication."""
    source = source_builder()
    transform = wp.transform((2.0, 3.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2.0))
    target = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    target.replicate(source, 2, xforms=[wp.transform_identity(), transform])
    target.color()
    model = target.finalize(device=device)
    test.assertEqual(model.tendon_count, 2)
    np.testing.assert_array_equal(model.tendon_start.numpy(), [0, 2, 4])
    np.testing.assert_array_equal(model.tendon_link_body.numpy(), [0, 1, 2, 3])
    offsets = model.tendon_link_offset.numpy()
    np.testing.assert_array_equal(offsets[0], offsets[2])
    np.testing.assert_array_equal(offsets[1], offsets[3])
    np.testing.assert_allclose(model.tendon_seg_compliance.numpy(), [0.01, 0.01])
    np.testing.assert_allclose(model.tendon_seg_rest_length.numpy(), [0.9, 0.9])
    solver = newton.solvers.SolverVBD(model, iterations=2, tendon_material_direct=True)
    a, b = model.state(), model.state()
    for _ in range(3):
        solver.step(a, b, model.control(), None, 1.0 / 120.0)
        a, b = b, a
    solver.check_tendon_material()
    test.assertTrue(np.isfinite(a.body_q.numpy()).all())
    # Identical equilibrium copies stay at their independently transformed positions.
    np.testing.assert_allclose(
        a.body_q.numpy()[:, :3], [[0.2, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 3.2, 0.0], [2.0, 4.0, 0.0]], atol=2.0e-5
    )


def test_add_builder_preserves_existing_tendons(test, device):
    """Offset appended links without altering the existing tendon."""
    source = source_builder()
    target = source_builder()
    target.add_builder(source)
    target.color()
    model = target.finalize(device=device)
    np.testing.assert_array_equal(model.tendon_start.numpy(), [0, 2, 4])
    np.testing.assert_array_equal(model.tendon_link_body.numpy(), [0, 1, 2, 3])


def test_unsupported_modes_preserve_input_state(test, device):
    """Reject unsupported solver combinations and leave a rejected reset untouched."""
    model = source_builder().finalize(device=device)
    for kwargs in (
        {"rigid_articulation_solve": "block_sparse_joints"},
        {"integrate_with_external_rigid_solver": True},
        {"rigid_joint_limit_alm": True},
    ):
        with test.assertRaises(ValueError):
            newton.solvers.SolverVBD(model, **kwargs)
    solver = newton.solvers.SolverVBD(model)
    state = model.state()
    changed = state.body_q.numpy().copy()
    changed[:, 0] += 0.5
    state.body_q.assign(changed)
    with test.assertRaisesRegex(NotImplementedError, "Tendon history reset"):
        solver.reset(state)
    np.testing.assert_array_equal(state.body_q.numpy(), changed)


class TestTendonModelCopy(unittest.TestCase):
    pass


for device in get_test_devices():
    add_function_test(
        TestTendonModelCopy,
        "test_unsupported_modes_preserve_input_state",
        test_unsupported_modes_preserve_input_state,
        devices=[device],
    )
    add_function_test(
        TestTendonModelCopy,
        "test_replicated_tendon_indices_and_transforms",
        test_replicated_tendon_indices_and_transforms,
        devices=[device],
    )
    add_function_test(
        TestTendonModelCopy,
        "test_add_builder_preserves_existing_tendons",
        test_add_builder_preserves_existing_tendons,
        devices=[device],
    )

if __name__ == "__main__":
    unittest.main(verbosity=2)
