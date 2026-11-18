# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Active-set contact linearization of the coupled block-sparse articulation step."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.rigid_vbd_kernels import evaluate_rigid_contact_from_collision
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()
SPARSE_MODES = (
    ("block_sparse_joints", {}),
    ("block_sparse_levels", {}),
    ("block_sparse_levels", {"rigid_articulation_levels_launch": "level"}),
)


@wp.kernel
def _evaluate_normal_contact(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    penetration: float,
    rho: float,
    material_k: float,
    lam_n: float,
    active_set_linear: int,
    force_a: wp.array[wp.vec3],
    hessian_a: wp.array[wp.mat33],
):
    """One body against the world along +z, no friction, no damping."""
    n = wp.vec3(0.0, 0.0, 1.0)
    f_a, _t_a, h_ll_a, _h_al_a, _h_aa_a, _f_b, _t_b, _h_ll_b, _h_al_b, _h_aa_b = evaluate_rigid_contact_from_collision(
        0,
        -1,
        body_q,
        body_q,
        body_com,
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        0,
        n,
        penetration,
        rho,
        material_k,
        rho,
        0.0,
        n * lam_n,
        0.0,
        1.0e-3,
        0,
        1,
        1.0e-3,
        wp.vec3(0.0),
        active_set_linear,
    )
    force_a[0] = f_a
    hessian_a[0] = h_ll_a


def _normal_contact_response(device, penetration, lam_n, active_set_linear, rho=1.0e5, material_k=1.0e8):
    body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
    body_com = wp.zeros(1, dtype=wp.vec3, device=device)
    force = wp.zeros(1, dtype=wp.vec3, device=device)
    hessian = wp.zeros(1, dtype=wp.mat33, device=device)
    wp.launch(
        _evaluate_normal_contact,
        dim=1,
        inputs=[body_q, body_com, penetration, rho, material_k, lam_n, active_set_linear],
        outputs=[force, hessian],
        device=device,
    )
    return force.numpy()[0], hessian.numpy()[0]


def test_active_set_keeps_separated_compressive_contact_linear(test, device):
    """A separated contact with a compressive multiplier: clamped -> released; active set -> linear pull-back."""
    rho, k = 1.0e5, 1.0e8
    s, k_eff = k / (k + rho), k * rho / (k + rho)
    lam_n = 10.0
    penetration = -1.5 * s * lam_n / k_eff  # k_eff * C + s * lam < 0 but above -s * lam: the clamped force is zero
    f_clamped, h_clamped = _normal_contact_response(device, penetration, lam_n, 0, rho, k)
    test.assertEqual(float(np.abs(f_clamped).max()), 0.0)
    test.assertEqual(float(np.abs(h_clamped).max()), 0.0)
    f_linear, h_linear = _normal_contact_response(device, penetration, lam_n, 1, rho, k)
    expected = k_eff * penetration + s * lam_n
    test.assertLess(expected, 0.0)
    # The function returns the reaction on body a, -f_n * n.
    test.assertAlmostEqual(float(f_linear[2]), -expected, delta=abs(expected) * 1.0e-5)
    test.assertAlmostEqual(float(h_linear[2, 2]), k_eff, delta=k_eff * 1.0e-5)
    test.assertEqual(float(np.abs(h_linear[:2, :]).max()), 0.0)


def test_active_set_tensile_force_is_bounded_by_the_multiplier(test, device):
    """Far past separation the linear force stops at -lambda: the penalty may at most cancel the multiplier."""
    rho, k = 1.0e5, 1.0e8
    s, k_eff = k / (k + rho), k * rho / (k + rho)
    lam_n = 10.0
    f_linear, h_linear = _normal_contact_response(device, -50.0 * s * lam_n / k_eff, lam_n, 1, rho, k)
    test.assertAlmostEqual(float(f_linear[2]), s * lam_n, delta=s * lam_n * 1.0e-5)
    test.assertAlmostEqual(float(h_linear[2, 2]), k_eff, delta=k_eff * 1.0e-5)


def test_active_set_matches_clamped_when_compressive(test, device):
    """With the force compressive the two linearizations are identical."""
    for penetration, lam_n in ((1.0e-4, 0.0), (1.0e-4, 5.0), (-1.0e-6, 5.0)):
        f0, h0 = _normal_contact_response(device, penetration, lam_n, 0)
        f1, h1 = _normal_contact_response(device, penetration, lam_n, 1)
        np.testing.assert_array_equal(f0, f1)
        np.testing.assert_array_equal(h0, h1)


def test_active_set_releases_without_multiplier(test, device):
    """A separated contact with no multiplier is released in both linearizations."""
    f, h = _normal_contact_response(device, -1.0e-4, 0.0, 1)
    test.assertEqual(float(np.abs(f).max()), 0.0)
    test.assertEqual(float(np.abs(h).max()), 0.0)


def _box_on_ground(device, mode, standalone=False, **solver_kw):
    """A free box in an articulation resting on the ground; optionally a second, standalone box far away."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    builder.add_ground_plane()
    body = builder.add_link(
        xform=wp.transform((0.0, 0.0, 0.12), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.01),
    )
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_articulation([builder.add_joint_free(child=body)])
    other = None
    if standalone:
        # A link without a joint stays outside every articulation and is solved by the local sweep.
        other = builder.add_link(
            xform=wp.transform((5.0, 0.0, 0.12), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.01),
        )
        builder.add_shape_box(other, hx=0.1, hy=0.1, hz=0.1)
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model, iterations=8, rigid_compliant_alm=True, rigid_articulation_solve=mode, **solver_kw
    )
    return model, pipeline, contacts, solver, body, other


def _settle(model, pipeline, contacts, solver, steps=20, dt=1.0 / 120.0):
    state_in, state_out = model.state(), model.state()
    for _ in range(steps):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in
    return state_in


def test_contact_active_set_box_rests(test, device, mode="block_sparse_levels", launch=None):
    kw = dict(launch or {})
    model, pipeline, contacts, solver, body, _ = _box_on_ground(
        device, mode, rigid_articulation_contact_active_set=True, **kw
    )
    test.assertTrue(solver.rigid_articulation_contact_active_set)
    pose = _settle(model, pipeline, contacts, solver).body_q.numpy()[body]
    test.assertTrue(np.isfinite(pose).all())
    test.assertGreater(float(pose[2]), 0.08)


def test_contact_active_set_leaves_local_bodies_unchanged(test, device, mode="block_sparse_levels", launch=None):
    """A standalone body on the local sweep follows the same path with the option on and off, bit for bit."""
    kw = dict(launch or {})
    poses = []
    for active in (False, True):
        model, pipeline, contacts, solver, _, other = _box_on_ground(
            device, mode, standalone=True, rigid_articulation_contact_active_set=active, **kw
        )
        test.assertLess(int(solver.rigid_articulation_sparse_layout.body_articulation_local.numpy()[other]), 0)
        poses.append(_settle(model, pipeline, contacts, solver).body_q.numpy()[other].copy())
    np.testing.assert_array_equal(poses[0], poses[1])
    test.assertGreater(float(poses[1][2]), 0.08)


class TestVBDSparseContactStep(unittest.TestCase):
    def test_rejects_invalid_options(self):
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.01))
        builder.add_articulation([builder.add_joint_free(child=body)])
        builder.color()
        model = builder.finalize(device="cpu")
        with self.assertRaises(ValueError):
            newton.solvers.SolverVBD(
                model, rigid_articulation_solve="block_sparse_levels", rigid_articulation_contact_active_set=1
            )
        with self.assertRaises(ValueError):
            newton.solvers.SolverVBD(model, rigid_articulation_solve="local", rigid_articulation_contact_active_set=True)

    def test_default_keeps_previous_behaviour(self):
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.01))
        builder.add_articulation([builder.add_joint_free(child=body)])
        builder.color()
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverVBD(model, rigid_articulation_solve="block_sparse_levels")
        self.assertFalse(solver.rigid_articulation_contact_active_set)


class TestVBDSparseContactStepDevices(unittest.TestCase):
    pass


for _test in (
    test_active_set_keeps_separated_compressive_contact_linear,
    test_active_set_tensile_force_is_bounded_by_the_multiplier,
    test_active_set_matches_clamped_when_compressive,
    test_active_set_releases_without_multiplier,
):
    add_function_test(TestVBDSparseContactStepDevices, _test.__name__, _test, devices=devices)

for _test in (
    test_contact_active_set_box_rests,
    test_contact_active_set_leaves_local_bodies_unchanged,
):
    for mode, launch in SPARSE_MODES:
        suffix = mode + ("_level_launch" if launch else "")
        add_function_test(
            TestVBDSparseContactStepDevices, f"{_test.__name__}_{suffix}", _test, devices=devices, mode=mode, launch=launch
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
