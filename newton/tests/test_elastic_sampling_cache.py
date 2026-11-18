# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check reuse of replicated elastic surface samples during construction."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton


def _basis(torsion=False):
    """Build a lumped basis whose interpolated surface points must stay massless."""
    points = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.1, 0.1) for z in (-0.05, 0.05)], dtype=np.float32)
    phi = np.cross([1.0, 0.0, 0.0], points) if torsion else points
    return newton.ModalBasis(
        sample_points=points, sample_phi=phi[:, None, :], sample_mass=np.full(8, 0.125), mode_stiffness=[10.0]
    )


def _scene(bases, transforms=None, scales=None, *, different_meshes=False):
    """Replicate a sampled mesh across worlds, preserving each body-local transform."""
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 9), np.linspace(-0.1, 0.1, 9))
    vertices = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.05))).astype(np.float32)
    indices = []
    for j in range(8):
        for i in range(8):
            a = 9 * j + i
            indices.extend((a, a + 1, a + 10, a, a + 10, a + 9))
    mesh = newton.Mesh(vertices, np.asarray(indices, dtype=np.int32))
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for index, basis in enumerate(bases):
        template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = template.add_body_elastic(modal_basis=basis)
        source = newton.Mesh(vertices + 0.03 * index, np.asarray(indices, dtype=np.int32)) if different_meshes else mesh
        template.add_shape_mesh(
            body,
            mesh=source,
            xform=None if transforms is None else transforms[index],
            scale=None if scales is None else scales[index],
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )
        scene.add_world(template, xform=wp.transform(wp.vec3(2.0 * index, 0.0, 0.0), wp.quat_identity()))
    return scene, vertices


def _assert_samples(test, model):
    """Match model-local positions, basis indices and material values for every shape."""
    positions = model.elastic_shape_vertex_local.numpy()
    indices = model.elastic_shape_vertex_sample.numpy()
    phi = model.elastic_shape_vertex_phi.numpy().reshape(-1, model.elastic_max_mode_count, 3)
    for row, body in enumerate(model.elastic_shape_body.numpy()):
        elastic = int(model.body_elastic_index.numpy()[body])
        basis = model.modal_bases[int(model.elastic_basis.numpy()[elastic])]
        begin = int(model.elastic_shape_vertex_start.numpy()[row])
        end = begin + int(model.elastic_shape_vertex_count.numpy()[row])
        np.testing.assert_allclose(basis.sample_points[indices[begin:end]], positions[begin:end], atol=1.0e-7)
        np.testing.assert_array_equal(basis.sample_phi[indices[begin:end]], phi[begin:end])
        test.assertAlmostEqual(float(np.sum(basis.sample_mass)), 1.0)
        np.testing.assert_array_equal(basis.sample_mass[8:], 0.0)


class TestElasticSamplingCache(unittest.TestCase):
    def test_replicated_surface_samples(self):
        """Sample identical replicated surfaces once without changing their material indices."""
        basis = _basis()
        before_mass = basis.mode_mass_matrix.copy()
        builder, vertices = _scene([basis] * 4)
        with mock.patch.object(basis, "add_samples", wraps=basis.add_samples) as sampled:
            model = builder.finalize(device="cpu")
        surface_calls = [call for call in sampled.call_args_list if len(call.args[0]) == len(vertices)]
        self.assertEqual(len(surface_calls), 1)
        _assert_samples(self, model)
        np.testing.assert_array_equal(basis.mode_mass_matrix, before_mass)

    def test_distinct_bases(self):
        """Keep coincident geometry with different modal fields in separate cache entries."""
        first, second = _basis(), _basis(torsion=True)
        builder, _vertices = _scene([first, second, first, second])
        _assert_samples(self, builder.finalize(device="cpu"))

    def test_distinct_meshes(self):
        """Keep separate mesh geometry when basis and shape transforms coincide."""
        basis = _basis()
        builder, vertices = _scene([basis] * 2, different_meshes=True)
        model = builder.finalize(device="cpu")
        _assert_samples(self, model)
        begin = int(model.elastic_shape_vertex_start.numpy()[1])
        np.testing.assert_allclose(
            model.elastic_shape_vertex_local.numpy()[begin : begin + len(vertices)], vertices + 0.03, atol=1.0e-7
        )

    def test_distinct_transforms_and_scales(self):
        """Retain body-local placement and scaling when surfaces share one mesh."""
        basis = _basis()
        shifted = wp.transform(wp.vec3(0.03, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.2))
        transforms = [wp.transform_identity(), shifted, wp.transform_identity(), shifted]
        scales = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)]
        builder, vertices = _scene([basis] * 4, transforms, scales)
        model = builder.finalize(device="cpu")
        _assert_samples(self, model)
        positions = model.elastic_shape_vertex_local.numpy()
        for index, (transform, scale) in enumerate(zip(transforms, scales, strict=True)):
            expected = np.array([wp.transform_point(transform, wp.vec3(point * scale)) for point in vertices])
            begin = int(model.elastic_shape_vertex_start.numpy()[index])
            np.testing.assert_allclose(positions[begin : begin + len(vertices)], expected, atol=1.0e-7)

    def test_repeated_finalization(self):
        """Start a fresh sampling cache for each model construction."""
        basis = _basis()
        builder, vertices = _scene([basis] * 3)
        with mock.patch.object(basis, "add_samples", wraps=basis.add_samples) as sampled:
            first = builder.finalize(device="cpu")
            second = builder.finalize(device="cpu")
        surface_calls = [call for call in sampled.call_args_list if len(call.args[0]) == len(vertices)]
        self.assertEqual(len(surface_calls), 2)
        np.testing.assert_array_equal(first.elastic_shape_vertex_phi.numpy(), second.elastic_shape_vertex_phi.numpy())


if __name__ == "__main__":
    unittest.main(verbosity=2)
