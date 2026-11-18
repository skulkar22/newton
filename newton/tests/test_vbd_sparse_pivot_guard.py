# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pivot guard of the 6 x 6 block Cholesky used by the block-sparse articulation solves."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.rigid_sparse_articulation_kernels import (
    SPARSE_PIVOT_FLOOR_ABSOLUTE,
    SPARSE_PIVOT_FLOOR_RELATIVE,
    _cholesky66_scalar,
    _cholesky66_serial,
    mat66f,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


@wp.func
def _cholesky66_unguarded(A_in: mat66f) -> mat66f:
    """The factorization as it was before the guard, for the bitwise-neutrality check."""
    L = mat66f(0.0)
    for i in range(6):
        for j in range(6):
            if j <= i:
                value = A_in[i, j]
                for k in range(6):
                    if k < j:
                        value = value - L[i, k] * L[j, k]
                if i == j:
                    L[i, j] = wp.sqrt(value)
                else:
                    L[i, j] = value / L[j, j]
    return L


@wp.kernel
def _factor_blocks(
    blocks: wp.array[mat66f],
    pivot_clamped: wp.array[wp.int32],
    guarded: wp.array[mat66f],
    unguarded: wp.array[mat66f],
    scalar_values: wp.array[float],
):
    i = wp.tid()
    guarded[i] = _cholesky66_serial(blocks[i], pivot_clamped)
    unguarded[i] = _cholesky66_unguarded(blocks[i])
    _cholesky66_scalar(scalar_values, i, pivot_clamped)


def _factor(device, blocks_host):
    blocks_host = np.asarray(blocks_host, dtype=np.float32)
    n = blocks_host.shape[0]
    blocks = wp.array(blocks_host, dtype=mat66f, device=device)
    scalar = wp.array(blocks_host.reshape(-1), dtype=float, device=device)
    pivot_clamped = wp.zeros(1, dtype=wp.int32, device=device)
    guarded = wp.zeros(n, dtype=mat66f, device=device)
    unguarded = wp.zeros(n, dtype=mat66f, device=device)
    wp.launch(_factor_blocks, dim=n, inputs=[blocks, pivot_clamped], outputs=[guarded, unguarded, scalar], device=device)
    return guarded.numpy(), unguarded.numpy(), scalar.numpy().reshape(n, 6, 6), int(pivot_clamped.numpy()[0])


def _spd_blocks(count, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        a = rng.standard_normal((6, 6))
        a = a @ a.T + 6.0 * np.eye(6)
        scale = 10.0 ** rng.uniform(-6.0, 6.0, size=6)  # the real blocks span many orders of magnitude
        out.append((scale[:, None] * a * scale[None, :]).astype(np.float32))
    return np.stack(out)


def test_pivot_guard_is_bitwise_neutral_on_spd_blocks(test, device):
    blocks = _spd_blocks(64)
    guarded, unguarded, scalar, clamped = _factor(device, blocks)
    test.assertEqual(clamped, 0)
    np.testing.assert_array_equal(guarded, unguarded)
    # The in-place scalar variant writes the same lower factor (upper part zeroed).
    np.testing.assert_array_equal(np.tril(scalar), guarded)
    test.assertTrue(np.isfinite(guarded).all())


def test_pivot_guard_clamps_indefinite_block_without_nan(test, device):
    blocks = _spd_blocks(2)
    blocks[0, 5, 5] = -1.0  # a negative assembled diagonal entry: the last pivot is negative
    blocks[1] = np.eye(6, dtype=np.float32)
    blocks[1, 0, 1] = blocks[1, 1, 0] = 1.5  # positive diagonal, but the second Schur pivot is 1 - 1.5^2 < 0
    guarded, unguarded, scalar, clamped = _factor(device, blocks)
    # One clamp per block in each of the two factorization variants.
    test.assertEqual(clamped, 4)
    test.assertTrue(np.isfinite(guarded).all())
    test.assertTrue(np.isfinite(scalar).all())
    test.assertFalse(np.isfinite(unguarded).all())
    floor0 = max(SPARSE_PIVOT_FLOOR_RELATIVE * abs(float(blocks[0, 5, 5])), SPARSE_PIVOT_FLOOR_ABSOLUTE)
    test.assertAlmostEqual(float(guarded[0, 5, 5]) ** 2, floor0, delta=floor0 * 1.0e-5)
    np.testing.assert_array_equal(np.tril(scalar), guarded)


def test_solver_counts_no_clamped_pivots_on_a_healthy_model(test, device, mode="block_sparse_levels", launch=None):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    builder.add_ground_plane()
    body = builder.add_link(
        xform=wp.transform((0.0, 0.0, 0.12), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.01),
    )
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_articulation([builder.add_joint_free(child=body)])
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model, iterations=4, rigid_compliant_alm=True, rigid_articulation_solve=mode, **dict(launch or {})
    )
    state_in, state_out = model.state(), model.state()
    for _ in range(5):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, 1.0 / 120.0)
        state_in, state_out = state_out, state_in
    test.assertEqual(int(solver.rigid_articulation_pivot_clamped.numpy()[0]), 0)
    test.assertTrue(np.isfinite(state_in.body_q.numpy()).all())


class TestVBDSparsePivotGuard(unittest.TestCase):
    def test_local_solver_has_no_counter(self):
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.01))
        builder.add_articulation([builder.add_joint_free(child=body)])
        builder.color()
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverVBD(model, rigid_articulation_solve="local")
        self.assertIsNone(solver.rigid_articulation_pivot_clamped)


class TestVBDSparsePivotGuardDevices(unittest.TestCase):
    pass


for _test in (test_pivot_guard_is_bitwise_neutral_on_spd_blocks, test_pivot_guard_clamps_indefinite_block_without_nan):
    add_function_test(TestVBDSparsePivotGuardDevices, _test.__name__, _test, devices=devices)
for mode, launch in (
    ("block_sparse_joints", {}),
    ("block_sparse_levels", {}),
    ("block_sparse_levels", {"rigid_articulation_levels_launch": "level"}),
):
    add_function_test(
        TestVBDSparsePivotGuardDevices,
        f"test_solver_counts_no_clamped_pivots_on_a_healthy_model_{mode}{'_level_launch' if launch else ''}",
        test_solver_counts_no_clamped_pivots_on_a_healthy_model,
        devices=devices,
        mode=mode,
        launch=launch,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
