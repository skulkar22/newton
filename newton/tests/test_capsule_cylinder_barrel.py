# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the bounded analytic barrel witness and its general-query fallbacks."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.core.types import MAXVAL
from newton._src.geometry.collision_primitive import _collide_capsule_cylinder_barrel
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def query_barrel(
    positions: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    handled: wp.array[int],
    distances: wp.array[float],
    centers: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
):
    i = wp.tid()
    ok, distance, center, normal = _collide_capsule_cylinder_barrel(
        positions[i], axes[i], 0.25, 0.5, wp.vec3(0.0), wp.vec3(0.0, 0.0, 1.0), 3.75, 1.5
    )
    handled[i] = int(ok)
    distances[i] = distance
    centers[i] = center
    normals[i] = normal


def test_barrel_guards(test, device):
    """Preserve parallel, cap/rim and core-intersection fallback; handle endpoints."""
    positions = [
        (3.95, 0.0, 0.3),  # interior barrel witness
        (4.45, 0.0, 0.3),  # core endpoint is the closest witness
        (4.1, 0.0, 0.3),  # separated but analytically handled
        (3.95, 0.0, 0.3),  # parallel: preserve the general manifold
        (3.95, 0.0, 1.5),  # exact rim: fallback
        (3.95, 0.0, 1.6),  # outside barrel: fallback
        (3.7, 0.0, 0.3),  # core intersects cylinder: fallback
    ]
    axes = [
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    n = len(positions)
    with wp.ScopedDevice(device):
        handled = wp.zeros(n, dtype=int)
        distances = wp.zeros(n, dtype=float)
        centers = wp.zeros(n, dtype=wp.vec3)
        normals = wp.zeros(n, dtype=wp.vec3)
        wp.launch(
            query_barrel,
            n,
            [wp.array(positions, dtype=wp.vec3), wp.array(axes, dtype=wp.vec3), handled, distances, centers, normals],
        )
    np.testing.assert_array_equal(handled.numpy(), [1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(distances.numpy()[:3], [-0.05, -0.05, 0.1], atol=1e-6)
    np.testing.assert_allclose(normals.numpy()[:3], [[-1.0, 0.0, 0.0]] * 3, atol=1e-6)
    np.testing.assert_allclose(centers.numpy()[:3], [[3.725, 0.0, 0.3], [3.725, 0.0, 0.3], [3.8, 0.0, 0.3]], atol=1e-6)
    test.assertTrue(np.all(distances.numpy()[3:] >= MAXVAL * 0.99))


def test_pipeline_dispatch(test, device):
    """Use analytic barrel contacts in either shape order; keep general fallbacks."""
    with wp.ScopedDevice(device):
        for reversed_order in (False, True):
            for kind in ("barrel", "parallel", "rim", "core", "rounded"):
                with test.subTest(reversed_order=reversed_order, kind=kind):
                    builder = newton.ModelBuilder()
                    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, gap=0.00025)
                    cylinder_body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
                    capsule_body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
                    cylinder_kwargs = {"body": cylinder_body, "radius": 0.00375, "half_height": 0.0015, "cfg": cfg}
                    if kind == "rounded":
                        cylinder_kwargs["barrel_radius"] = 0.02
                    capsule_kwargs = {"body": capsule_body, "radius": 0.00025, "half_height": 0.0005, "cfg": cfg}
                    if reversed_order:
                        builder.add_shape_capsule(**capsule_kwargs)
                        builder.add_shape_cylinder(**cylinder_kwargs)
                    else:
                        builder.add_shape_cylinder(**cylinder_kwargs)
                        builder.add_shape_capsule(**capsule_kwargs)
                    model = builder.finalize()
                    state = model.state()
                    poses = model.body_q.numpy()
                    poses[capsule_body, :3] = (0.00395, 0.0, 0.0003)
                    axis = wp.vec3(0.0, 0.0, 1.0) if kind == "parallel" else wp.vec3(0.0, 1.0, 0.0)
                    poses[capsule_body, 3:] = np.asarray(wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), axis))
                    if kind == "rim":
                        poses[capsule_body, 2] = 0.0016
                    elif kind == "core":
                        poses[capsule_body, 0] = 0.0037
                    state.body_q.assign(poses)
                    pipeline = newton.CollisionPipeline(model)
                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts)
                    count = int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0])
                    test.assertEqual(count, 0 if kind == "barrel" else 1)
                    if kind == "barrel":
                        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
                        a = int(contacts.rigid_contact_shape0.numpy()[0])
                        b = int(contacts.rigid_contact_shape1.numpy()[0])
                        bodies = model.shape_body.numpy()
                        x0 = np.asarray(
                            wp.transform_point(
                                wp.transform(*poses[bodies[a]]), wp.vec3(*contacts.rigid_contact_point0.numpy()[0])
                            )
                        )
                        x1 = np.asarray(
                            wp.transform_point(
                                wp.transform(*poses[bodies[b]]), wp.vec3(*contacts.rigid_contact_point1.numpy()[0])
                            )
                        )
                        normal = contacts.rigid_contact_normal.numpy()[0]
                        gap = float(
                            (x1 - x0) @ normal
                            - contacts.rigid_contact_margin0.numpy()[0]
                            - contacts.rigid_contact_margin1.numpy()[0]
                        )
                        test.assertAlmostEqual(gap, -0.00005, delta=1e-8)
                        expected = (
                            np.array([1.0, 0.0, 0.0]) if bodies[a] == cylinder_body else np.array([-1.0, 0.0, 0.0])
                        )
                        np.testing.assert_allclose(normal, expected, atol=1e-6)


class TestCapsuleCylinderBarrel(unittest.TestCase):
    """The private shortcut must never claim the general cap/manifold cases."""


add_function_test(TestCapsuleCylinderBarrel, "test_barrel_guards", test_barrel_guards, devices=get_test_devices())
add_function_test(
    TestCapsuleCylinderBarrel, "test_pipeline_dispatch", test_pipeline_dispatch, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
