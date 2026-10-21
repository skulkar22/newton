# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for sparse finite-element input and the ARPACK path of ``ModalGeneratorFEM``.

These are setup-time NumPy/SciPy tests; they do not launch Warp kernels and
are registered once, independent of the test device.
"""

import importlib.util
import itertools
import math
import sys
import time
import tracemalloc
import unittest
from unittest import mock

import numpy as np

import newton
from newton._src.sim import modal_sparse
from newton._src.sim.modal import _estimate_sample_psi, _nearest_neighbors
from newton.tests.unittest_utils import add_function_test

HAS_SCIPY = importlib.util.find_spec("scipy") is not None


def _sparse_spring_lattice(shape, spacing, spring_stiffness, node_mass, braced=True, soft_split=None):
    """Assemble a lumped-mass spring lattice directly in sparse form.

    The springs match ``_spring_lattice`` of ``test_reduced_elastic_body``:
    axial springs of stiffness ``spring_stiffness / length`` along the grid
    edges and, when ``braced``, the face and body diagonals.

    Args:
        shape: Node counts along ``x``, ``y``, and ``z``.
        spacing: Node spacing [m].
        spring_stiffness: Axial stiffness scale [N] divided by spring length.
        node_mass: Lumped node mass [kg].
        braced: Whether to add face and body diagonal springs.
        soft_split: Optional ``(x_index, ratio)``; springs crossing the plane
            between ``x_index - 1`` and ``x_index`` are scaled by ``ratio``.

    Returns:
        Node positions [m], CSC mass matrix [kg], and CSC stiffness matrix [N/m].
    """
    from scipy import sparse

    grid = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), axis=-1).reshape((-1, 3))
    points = grid.astype(np.float64) * spacing
    points -= points.mean(axis=0)
    node_count = points.shape[0]
    index = np.arange(node_count).reshape(shape)
    if braced:
        offsets = [offset for offset in itertools.product((-1, 0, 1), repeat=3) if offset > (0, 0, 0)]
    else:
        offsets = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    rows, cols, values = [], [], []
    for offset in offsets:
        first = tuple(slice(max(0, -o), n - max(0, o)) for o, n in zip(offset, shape, strict=True))
        second = tuple(slice(max(0, o), n - max(0, -o)) for o, n in zip(offset, shape, strict=True))
        a = index[first].reshape(-1)
        b = index[second].reshape(-1)
        if a.size == 0:
            continue
        delta = np.asarray(offset, dtype=np.float64) * spacing
        length = float(np.linalg.norm(delta))
        direction = delta / length
        scale = np.full(a.size, spring_stiffness / length)
        if soft_split is not None:
            split, ratio = soft_split
            crossing = (grid[a, 0] < split) != (grid[b, 0] < split)
            scale[crossing] *= ratio
        block = np.outer(direction, direction)
        for i in range(3):
            for j in range(3):
                for p, q, sign in ((a, a, 1.0), (b, b, 1.0), (a, b, -1.0), (b, a, -1.0)):
                    rows.append(3 * p + i)
                    cols.append(3 * q + j)
                    values.append(sign * block[i, j] * scale)
    size = 3 * node_count
    stiffness = sparse.csc_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(size, size)
    )
    stiffness.sum_duplicates()
    stiffness.eliminate_zeros()
    mass = sparse.identity(size, format="csc") * node_mass
    return points, sparse.csc_matrix(mass), stiffness


def _truss_stiffness(points, bars, bar_stiffness):
    """Assemble the dense stiffness matrix [N/m] of axial bars between ``points``."""
    points = np.asarray(points, dtype=np.float64)
    stiffness = np.zeros((3 * points.shape[0], 3 * points.shape[0]), dtype=np.float64)
    for (a, b), k in zip(bars, bar_stiffness, strict=True):
        direction = points[b] - points[a]
        direction /= np.linalg.norm(direction)
        block = k * np.outer(direction, direction)
        for i, j, sign in ((a, a, 1.0), (b, b, 1.0), (a, b, -1.0), (b, a, -1.0)):
            stiffness[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += sign * block
    return stiffness


def _build_pair(points, mass, stiffness, **kwargs):
    """Build the same model with the dense and the ARPACK eigensolver."""
    dense = newton.ModalGeneratorFEM(points, mass, stiffness, eigensolver="dense", **kwargs)
    arpack = newton.ModalGeneratorFEM(points, mass, stiffness, eigensolver="arpack", **kwargs)
    return dense, dense.build(), arpack, arpack.build()


def _assert_paths_agree(test, mass, dense, dense_basis, arpack, arpack_basis):
    """Assert that the dense and ARPACK builds agree up to the sign of each mode."""
    np.testing.assert_allclose(arpack.eigenvalues, dense.eigenvalues, rtol=1.0e-8)
    test.assertEqual(arpack.rigid_mode_count, dense.rigid_mode_count)
    mass = modal_sparse.to_dense(mass)
    # Mass-weighted MAC of mass-normalized modes, and a per-mode sign for the sampled values.
    overlap = np.einsum("di,de,ei->i", dense.modal_matrix, mass, arpack.modal_matrix)
    test.assertGreaterEqual(float(np.min(np.abs(overlap))), 1.0 - 1.0e-8)
    sign = np.sign(overlap)
    np.testing.assert_allclose(
        arpack.modal_matrix * sign, dense.modal_matrix, atol=1.0e-6 * np.abs(dense.modal_matrix).max()
    )
    np.testing.assert_allclose(arpack_basis.mode_mass_matrix, dense_basis.mode_mass_matrix, atol=1.0e-6)
    np.testing.assert_allclose(arpack_basis.mode_mass, dense_basis.mode_mass, rtol=1.0e-6)
    np.testing.assert_allclose(arpack_basis.mode_stiffness, dense_basis.mode_stiffness, rtol=1.0e-6)
    test.assertAlmostEqual(arpack_basis.rigid_mass, dense_basis.rigid_mass, places=12)
    np.testing.assert_allclose(arpack_basis.rigid_com, dense_basis.rigid_com, atol=1.0e-12)
    np.testing.assert_allclose(arpack_basis.rigid_inertia, dense_basis.rigid_inertia, rtol=1.0e-10, atol=1.0e-16)
    if dense_basis.sample_psi is not None:
        scale = float(np.max(np.abs(dense_basis.sample_psi)))
        np.testing.assert_allclose(
            arpack_basis.sample_psi * sign[None, :, None], dense_basis.sample_psi, atol=1.0e-5 * scale
        )


def test_fem_arpack_matches_dense_free_free(test, device):
    """Match the dense eigensolver with ARPACK shift-invert on a free-free lattice."""
    points, mass, stiffness = _sparse_spring_lattice((6, 4, 3), 0.01, 2.0e9, 8.0e-3)
    dense, dense_basis, arpack, arpack_basis = _build_pair(
        points, mass.toarray(), stiffness.toarray(), mode_count=10, derive_psi=True
    )
    test.assertEqual(dense.rigid_mode_count, 6)
    _assert_paths_agree(test, mass, dense, dense_basis, arpack, arpack_basis)
    np.testing.assert_allclose(arpack.rigid_modal_matrix, dense.rigid_modal_matrix, atol=1.0e-12)


def test_fem_arpack_matches_dense_clamped(test, device):
    """Match the dense eigensolver with ARPACK on a clamped lattice with sparse input."""
    points, mass, stiffness = _sparse_spring_lattice((6, 4, 3), 0.01, 2.0e9, 8.0e-3)
    kwargs = {"mode_count": 8, "derive_psi": True, "fixed_node_indices": [0, 1, 3], "discard_mode_count": 1}
    dense, dense_basis, arpack, arpack_basis = _build_pair(points, mass, stiffness, **kwargs)
    test.assertEqual(dense.rigid_mode_count, 0)
    _assert_paths_agree(test, mass, dense, dense_basis, arpack, arpack_basis)
    np.testing.assert_array_equal(arpack.modal_matrix[:3], 0.0)


def test_fem_sparse_input_dense_path_identical(test, device):
    """Give bitwise identical dense-path results for sparse CSR and dense matrix input."""
    points, mass, stiffness = _sparse_spring_lattice((4, 3, 3), 0.01, 2.0e9, 8.0e-3)
    damping = 1.0e-4 * stiffness + 0.3 * mass
    kwargs = {"mode_count": 6, "derive_psi": True, "eigensolver": "dense"}
    reference = newton.ModalGeneratorFEM(
        points, mass.toarray(), stiffness.toarray(), damping_matrix=damping.toarray(), **kwargs
    )
    candidate = newton.ModalGeneratorFEM(
        points, mass.tocsr(), stiffness.tocsr(), damping_matrix=damping.tocsr(), **kwargs
    )
    reference_basis = reference.build()
    candidate_basis = candidate.build()
    np.testing.assert_array_equal(candidate.eigenvalues, reference.eigenvalues)
    np.testing.assert_array_equal(candidate.modal_matrix, reference.modal_matrix)
    np.testing.assert_array_equal(candidate.rigid_modal_matrix, reference.rigid_modal_matrix)
    for name in ("sample_phi", "sample_psi", "mode_mass_matrix", "mode_damping", "mode_coupling_quadratic"):
        np.testing.assert_array_equal(getattr(candidate_basis, name), getattr(reference_basis, name))
    np.testing.assert_array_equal(candidate_basis.rigid_inertia, reference_basis.rigid_inertia)

    # "auto" keeps small dense input on the dense path and moves sparse input to ARPACK.
    test.assertFalse(newton.ModalGeneratorFEM(points, mass.toarray(), stiffness.toarray())._uses_arpack(108))
    test.assertTrue(newton.ModalGeneratorFEM(points, mass, stiffness)._uses_arpack(108))
    large = {"dense_dof_limit": 100}
    test.assertTrue(
        newton.ModalGeneratorFEM(points, mass.toarray(), stiffness.toarray(), mode_count=4, **large)._uses_arpack(108)
    )
    # Without mode_count, which ARPACK needs, large dense input keeps the dense path as before.
    test.assertFalse(newton.ModalGeneratorFEM(points, mass.toarray(), stiffness.toarray(), **large)._uses_arpack(108))

    # Craig-Bampton reduced matrices are small; sparse input is densified.
    coerced = newton.ModalGeneratorCraigBampton._coerce_square_matrix(stiffness.tocoo(), 108, "stiffness_matrix")
    test.assertIsInstance(coerced, np.ndarray)
    np.testing.assert_array_equal(coerced, 0.5 * (stiffness.toarray() + stiffness.toarray().T))
    with test.assertRaisesRegex(ValueError, r"stiffness_matrix must have shape \(9, 9\), got \(10, 10\)"):
        newton.ModalGeneratorFEM(points[:3], mass[:9, :9], stiffness[:10, :10])


def test_fem_arpack_wide_spectrum_keeps_soft_modes(test, device):
    """Keep elastic modes 1e9-1e10 below the stiffest one on the ARPACK path at default settings."""
    # The fixtures of test_reduced_elastic_body: small enough that ARPACK falls back to a dense solve.
    line = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    stiffness = _truss_stiffness(line, [(0, 1), (1, 2)], [1.0, 1.0e9])
    fixed = [0, 1, 2, 4, 5, 7, 8]
    reference = newton.ModalGeneratorFEM(line, np.eye(9), stiffness, fixed_dof_indices=fixed, eigensolver="dense")
    reference.build()
    generator = newton.ModalGeneratorFEM(
        line, np.eye(9), stiffness, fixed_dof_indices=fixed, mode_count=2, eigensolver="arpack"
    )
    test.assertEqual(generator.build().mode_count, 2)
    np.testing.assert_allclose(generator.eigenvalues, reference.eigenvalues, rtol=1.0e-6)
    test.assertGreater(float(generator.eigenvalues[1] / generator.eigenvalues[0]), 1.0e9)

    tet = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    bars = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    stiffness = _truss_stiffness(tet, bars, [1.0e10] * 5 + [1.0])
    reference = newton.ModalGeneratorFEM(tet, np.eye(12), stiffness, eigensolver="dense")
    reference.build()
    generator = newton.ModalGeneratorFEM(tet, np.eye(12), stiffness, mode_count=6, eigensolver="arpack")
    test.assertEqual(generator.build().mode_count, 6)
    test.assertEqual(generator.rigid_mode_count, 6)
    np.testing.assert_allclose(generator.eigenvalues, reference.eigenvalues, rtol=1.0e-6)

    # Two stiff blocks joined by springs 1e-10 as stiff: six soft joint modes far below the
    # stiff spectrum, solved by ARPACK proper (k much smaller than the DOF count).
    points, mass, stiffness = _sparse_spring_lattice((8, 3, 3), 0.01, 2.0e9, 8.0e-3, soft_split=(4, 1.0e-10))
    dense, _, arpack, _ = _build_pair(points, mass, stiffness, mode_count=10)
    test.assertEqual(arpack.rigid_mode_count, 6)
    test.assertGreater(float(arpack.eigenvalues[-1] / arpack.eigenvalues[0]), 1.0e9)
    np.testing.assert_allclose(arpack.eigenvalues, dense.eigenvalues, rtol=1.0e-5)
    test.assertLess(float(arpack.eigenvalues[5] / arpack.eigenvalues[6]), 1.0e-6)


def test_fem_arpack_rejects_invalid_models(test, device):
    """Reject mechanisms, leftover rigid motion, and inconsistent rigid modes on the ARPACK path."""
    kwargs = {"mode_count": 4, "eigensolver": "arpack"}
    points, mass, stiffness = _sparse_spring_lattice((4, 2, 2), 0.1, 1.0e3, 1.0e-2, braced=False)
    with test.assertRaisesRegex(ValueError, "zero-energy mechanism"):
        newton.ModalGeneratorFEM(points, mass, stiffness, **kwargs).build()

    points, mass, stiffness = _sparse_spring_lattice((4, 2, 2), 0.1, 1.0e3, 1.0e-2)
    with test.assertRaisesRegex(ValueError, "do not remove all rigid-body motion"):
        newton.ModalGeneratorFEM(points, mass, stiffness, fixed_node_indices=[0], **kwargs).build()

    broken = stiffness.tolil()
    broken[0, 0] += 1.0e3
    with test.assertRaisesRegex(ValueError, "does not preserve rigid-body motion"):
        newton.ModalGeneratorFEM(points, mass, broken.tocsc(), **kwargs).build()

    # A free-free model needs a negative shift; test_fem_helpers covers the Sturm check of a high shift.
    sigma = 0.5 * modal_sparse.generalized_eigenvalue_bound(stiffness, mass)
    with test.assertRaisesRegex(ValueError, "arpack_shift must be negative"):
        newton.ModalGeneratorFEM(points, mass, stiffness, arpack_shift=sigma, **kwargs).build()

    with test.assertRaisesRegex(ValueError, "mode_count is required"):
        newton.ModalGeneratorFEM(points, mass, stiffness, eigensolver="arpack").build()

    indefinite = mass.tolil()
    indefinite[5, 5] = -1.0e-2
    with test.assertRaisesRegex(ValueError, "mass_matrix must be symmetric positive definite"):
        newton.ModalGeneratorFEM(points, indefinite.tocsc(), stiffness, **kwargs).build()
    with test.assertRaisesRegex(ValueError, "eigensolver must be"):
        newton.ModalGeneratorFEM(points, mass, stiffness, eigensolver="lobpcg")


def test_fem_arpack_rejects_invalid_shifts(test, device):
    """Reject a non-negative shift on a free-free model and a shift at an eigenvalue before ARPACK runs."""
    points, mass, stiffness = _sparse_spring_lattice((6, 4, 3), 0.01, 2.0e9, 8.0e-3)
    kwargs = {"mode_count": 6, "eigensolver": "arpack"}
    reference = newton.ModalGeneratorFEM(points, mass, stiffness, eigensolver="dense")
    reference.build()
    for sigma in (0.0, 0.5 * float(reference.eigenvalues[0])):
        with test.assertRaisesRegex(ValueError, "arpack_shift must be negative for a free-free model"):
            newton.ModalGeneratorFEM(points, mass, stiffness, arpack_shift=sigma, **kwargs).build()

    clamped = {"fixed_node_indices": [0, 1, 2, 3, 4, 5]}
    reference = newton.ModalGeneratorFEM(points, mass, stiffness, eigensolver="dense", **clamped)
    reference.build()
    with test.assertRaisesRegex(ValueError, "numerically singular for sigma"):
        newton.ModalGeneratorFEM(
            points, mass, stiffness, arpack_shift=float(reference.eigenvalues[0]), **clamped, **kwargs
        ).build()
    generator = newton.ModalGeneratorFEM(
        points, mass, stiffness, arpack_shift=0.5 * float(reference.eigenvalues[0]), **clamped, **kwargs
    )
    generator.build()
    np.testing.assert_allclose(generator.eigenvalues, reference.eigenvalues[:6], rtol=1.0e-8)

    # SuperLU factors the singular free-free stiffness without error; only the pivot ratio reveals it.
    test.assertTrue(modal_sparse.ShiftedFactorization(stiffness, mass, 0.0).numerically_singular)
    sigma = modal_sparse.default_shift(stiffness, mass)
    test.assertFalse(modal_sparse.ShiftedFactorization(stiffness, mass, sigma).numerically_singular)


def test_fem_arpack_sturm_check_never_skipped_silently(test, device):
    """Warn whenever the Sturm sequence check cannot run, instead of returning unverified modes silently."""
    from scipy import sparse

    # Twelve identical clamped columns: the lowest eigenvalue has multiplicity 24, more than the
    # mode count plus every margin the gap search adds, so no gap follows the retained modes.
    points, mass, stiffness = _sparse_spring_lattice((2, 2, 10), 0.01, 2.0e9, 8.0e-3)
    copies = 12
    points = np.vstack([points + np.array([0.1 * copy, 0.0, 0.0]) for copy in range(copies)])
    mass = sparse.block_diag([mass] * copies, format="csc")
    stiffness = sparse.block_diag([stiffness] * copies, format="csc")
    base = np.flatnonzero(np.isclose(points[:, 2], points[:, 2].min()))
    reference = newton.ModalGeneratorFEM(points, mass, stiffness, fixed_node_indices=base, eigensolver="dense")
    reference.build()
    generator = newton.ModalGeneratorFEM(
        points, mass, stiffness, fixed_node_indices=base, mode_count=4, eigensolver="arpack"
    )
    with test.assertWarnsRegex(UserWarning, "Sturm sequence check was skipped: no relative eigenvalue gap"):
        generator.build()
    np.testing.assert_allclose(generator.eigenvalues, reference.eigenvalues[:4], rtol=1.0e-8)

    # Without a symmetric factorization the inertia is unavailable, which also warns.
    points, mass, stiffness = _sparse_spring_lattice((6, 4, 3), 0.01, 2.0e9, 8.0e-3)
    with mock.patch.object(modal_sparse, "count_eigenvalues_below", return_value=-1):
        with test.assertWarnsRegex(UserWarning, "Sturm sequence check was skipped: the factorization"):
            newton.ModalGeneratorFEM(points, mass, stiffness, mode_count=4, eigensolver="arpack").build()


def test_fem_arpack_residual_check(test, device):
    """Reject inaccurate ARPACK eigenpairs, which a nearly singular shifted factorization returns silently."""
    points, mass, stiffness = _sparse_spring_lattice((10, 8, 2), 0.01, 2.0e9, 8.0e-3)
    rigid = modal_sparse.rigid_vectors(points, 3, np.mean(points, axis=0))
    deflation = np.linalg.qr(rigid)[0] / math.sqrt(8.0e-3)
    # sigma = 0 factors the singular free-free stiffness: ARPACK reports convergence, but the pairs it
    # returns include negative eigenvalues with residuals of order 0.1.
    with test.assertRaisesRegex(ValueError, "inaccurate eigenpair near sigma = 0"):
        modal_sparse.generalized_eigsh(stiffness, mass, 12, sigma=0.0, deflation=deflation)
    sigma = modal_sparse.default_shift(stiffness, mass)
    factorization = modal_sparse.ShiftedFactorization(stiffness, mass, sigma)
    values, vectors = modal_sparse.generalized_eigsh(
        stiffness, mass, 12, sigma=sigma, deflation=deflation, factorization=factorization
    )
    errors = modal_sparse.eigenpair_backward_errors(stiffness, mass, values, vectors, factorization.norm)
    test.assertLess(float(np.max(errors)), 1.0e3 * np.finfo(np.float64).eps)
    test.assertGreater(float(values[0]), 0.0)


def test_fem_arpack_diagnoses_indefinite_matrices(test, device):
    """Tell an indefinite mass, an indefinite stiffness, and spurious eigenvalues apart from missed ones."""
    from scipy import sparse

    kwargs = {"mode_count": 4, "eigensolver": "arpack"}
    # Removing mass along the stiffest mode makes M indefinite in a direction that the inertia of
    # K - sigma M at sigma <= 0 cannot see; the Sturm count then disagrees with ARPACK.
    points, mass, stiffness = _sparse_spring_lattice((12, 12, 2), 0.01, 2.0e9, 8.0e-3)
    stiffest = np.linalg.eigh(stiffness.toarray())[1][:, -1]
    indefinite = sparse.csc_matrix(mass.toarray() - 1.5 * 8.0e-3 * np.outer(stiffest, stiffest))
    with test.assertRaisesRegex(ValueError, "mass_matrix must be symmetric positive definite, but its factorization"):
        newton.ModalGeneratorFEM(points, indefinite, stiffness, **kwargs).build()
    test.assertEqual(modal_sparse.nonpositive_pivot_count(indefinite), 1)

    points, mass, stiffness = _sparse_spring_lattice((6, 4, 3), 0.01, 2.0e9, 8.0e-3)
    clamped = {"fixed_node_indices": [0, 1, 2, 3, 4, 5]}
    reference = newton.ModalGeneratorFEM(points, mass, stiffness, eigensolver="dense", **clamped)
    reference.build()
    softened = (stiffness - 1.5 * float(reference.eigenvalues[0]) * mass).tocsc()
    with test.assertRaisesRegex(ValueError, "stiffness_matrix must be positive semi-definite on free DOFs"):
        newton.ModalGeneratorFEM(points, mass, softened, **clamped, **kwargs).build()

    # More eigenvalues than the Sturm count, with a positive-definite mass, are spurious, not missed.
    with mock.patch.object(modal_sparse, "count_eigenvalues_below", return_value=7):
        with test.assertRaisesRegex(ValueError, "more than the 1 of the Sturm sequence count.*spurious"):
            newton.ModalGeneratorFEM(points, mass, stiffness, **kwargs).build()


def test_fem_mode_sign_convention(test, device):
    """Give the dense and ARPACK paths the same mode signs: the largest-magnitude entry is positive."""
    points, mass, stiffness = _sparse_spring_lattice((6, 4, 3), 0.01, 2.0e9, 8.0e-3)
    for kwargs in ({"mode_count": 10}, {"mode_count": 8, "fixed_node_indices": [0, 1, 3], "discard_mode_count": 1}):
        dense, _, arpack, _ = _build_pair(points, mass, stiffness, **kwargs)
        np.testing.assert_allclose(
            arpack.modal_matrix, dense.modal_matrix, atol=1.0e-6 * np.abs(dense.modal_matrix).max()
        )
        for modes in (dense.modal_matrix, arpack.modal_matrix):
            magnitude = np.abs(modes)
            first_peak = np.argmax(magnitude >= (1.0 - 1.0e-6) * magnitude.max(axis=0), axis=0)
            test.assertTrue(np.all(modes[first_peak, np.arange(modes.shape[1])] > 0.0))


def test_fem_arpack_mechanisms_discarded(test, device):
    """Grow the ARPACK request until it spans the complete zero-energy subspace."""
    points, mass, stiffness = _sparse_spring_lattice((4, 2, 2), 0.1, 1.0e3, 1.0e-2, braced=False)
    dense = newton.ModalGeneratorFEM(points, mass, stiffness, eigensolver="dense")
    with test.assertRaisesRegex(ValueError, r"has (\d+) zero-energy mechanism") as caught:
        dense.build()
    mechanism_count = int(str(caught.exception).split()[3])
    test.assertGreater(mechanism_count, 4 + 2)
    kwargs = {"mode_count": 4, "discard_mode_count": mechanism_count}
    reference, _, generator, _ = _build_pair(points, mass, stiffness, **kwargs)
    np.testing.assert_allclose(generator.eigenvalues, reference.eigenvalues, rtol=1.0e-8)


def test_fem_arpack_large_sparse_model(test, device):
    """Build a 30k-DOF free-free lattice from sparse matrices, which the dense path cannot handle.

    The dense path would form several ``[28800, 28800]`` float64 matrices (6.6 GB each) and an
    ``O(n^3)`` eigensolve (about 1e13 flops); the ARPACK path keeps the matrices sparse.
    """
    shape = (40, 20, 12)
    points, mass, stiffness = _sparse_spring_lattice(shape, 0.01, 2.0e9, 8.0e-3)
    node_count = points.shape[0]
    test.assertEqual(3 * node_count, 28800)
    tracemalloc.start()
    try:
        start = time.perf_counter()
        generator = newton.ModalGeneratorFEM(
            points, mass, stiffness, mode_count=12, derive_psi=True, sample_node_indices=np.arange(node_count)
        )
        basis = generator.build()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    test.assertEqual(basis.mode_count, 12)
    test.assertEqual(generator.rigid_mode_count, 6)
    test.assertTrue(np.all(generator.frequencies > 0.0))
    test.assertTrue(np.all(np.diff(generator.eigenvalues) >= 0.0))
    test.assertTrue(np.all(np.isfinite(basis.sample_psi)))
    np.testing.assert_allclose(basis.mode_mass_matrix, np.eye(12), atol=1.0e-5)
    rigid = generator.rigid_modal_matrix
    np.testing.assert_allclose(rigid.T @ (mass @ generator.modal_matrix), 0.0, atol=1.0e-8)
    residual = stiffness @ generator.modal_matrix - (mass @ generator.modal_matrix) * generator.eigenvalues
    test.assertLess(float(np.max(np.abs(residual))), 1.0e-6 * float(np.max(generator.eigenvalues)) * 8.0e-3)
    # Peak NumPy-tracked memory, dominated by transient copies of the sparse LU factor U, stays far
    # below a single dense [n, n] float64 matrix (6.6 GB).
    test.assertLess(peak, 2.5e9)
    test.assertLess(elapsed, 120.0)


def test_fem_helpers(test, device):
    """Check the shared dense/sparse helpers against dense references."""
    from scipy import sparse

    rng = np.random.default_rng(3)
    dense = rng.uniform(-1.0, 1.0, size=(12, 12))
    dense[np.abs(dense) < 0.5] = 0.0
    matrix = sparse.csr_array(dense)
    vectors = rng.uniform(-1.0, 1.0, size=(12, 3))
    test.assertTrue(modal_sparse.is_sparse(matrix))
    test.assertFalse(modal_sparse.is_sparse(dense))
    symmetric = modal_sparse.coerce_symmetric_matrix(matrix, 12, "m")
    test.assertTrue(modal_sparse.is_sparse(symmetric))
    np.testing.assert_array_equal(symmetric.toarray(), 0.5 * (dense + dense.T))
    test.assertAlmostEqual(modal_sparse.matrix_inf_norm(matrix), np.linalg.norm(dense, ord=np.inf))
    test.assertAlmostEqual(modal_sparse.matrix_one_norm(matrix), np.linalg.norm(dense, ord=1))
    rows = np.array([1, 4, 7])
    np.testing.assert_array_equal(modal_sparse.submatrix(matrix, rows, rows).toarray(), dense[np.ix_(rows, rows)])
    np.testing.assert_allclose(modal_sparse.bilinear(vectors, matrix, vectors), vectors.T @ dense @ vectors)
    np.testing.assert_allclose(
        modal_sparse.quadratic_form_diag(vectors, matrix), np.diag(vectors.T @ dense @ vectors), atol=1.0e-12
    )
    with test.assertRaisesRegex(ValueError, "non-finite"):
        modal_sparse.coerce_symmetric_matrix(sparse.csr_matrix(np.full((2, 2), np.nan)), 2, "m")

    positions = rng.uniform(-1.0, 1.0, size=(5, 3))
    center = np.array([0.1, -0.2, 0.3])
    omega = np.array([0.3, -0.7, 0.2])
    velocity = np.array([1.0, 2.0, -0.5])
    for dofs_per_node in (3, 6):
        rigid = modal_sparse.rigid_vectors(positions, dofs_per_node, center)
        field = (rigid @ np.concatenate((velocity, omega))).reshape((5, dofs_per_node))
        np.testing.assert_allclose(field[:, :3], velocity + np.cross(omega, positions - center), atol=1.0e-14)
        if dofs_per_node == 6:
            np.testing.assert_allclose(field[:, 3:], np.tile(omega, (5, 1)), atol=1.0e-14)

    # ARPACK matches a dense generalized solve, and non-convergence becomes a ValueError.
    _, mass, stiffness = _sparse_spring_lattice((6, 4, 3), 0.01, 2.0e9, 8.0e-3)
    sigma = modal_sparse.default_shift(stiffness, mass)
    test.assertLess(sigma, 0.0)
    factorization = modal_sparse.ShiftedFactorization(stiffness, mass, sigma)
    test.assertEqual(factorization.negative_pivot_count, 0)
    values, vectors = modal_sparse.generalized_eigsh(stiffness, mass, 12, sigma=sigma, factorization=factorization)
    reference = np.linalg.eigvalsh(stiffness.toarray() / 8.0e-3)[:12]
    np.testing.assert_allclose(values[6:], reference[6:], rtol=1.0e-10)
    np.testing.assert_allclose(values[:6], 0.0, atol=1.0e-12 * reference[-1] * 1.0e3)
    np.testing.assert_allclose(vectors.T @ (mass @ vectors), np.eye(12), atol=1.0e-10)
    # The factorization inertia counts the eigenvalues below a positive shift.
    shift = 0.5 * (reference[8] + reference[9])
    test.assertEqual(modal_sparse.ShiftedFactorization(stiffness, mass, shift).negative_pivot_count, 9)
    test.assertLessEqual(modal_sparse.generalized_eigenvalue_bound(stiffness, mass), math.inf)
    test.assertGreaterEqual(
        modal_sparse.generalized_eigenvalue_bound(stiffness, mass),
        float(np.max(np.linalg.eigvalsh(stiffness.toarray() / 8.0e-3))),
    )
    with test.assertRaisesRegex(ValueError, "ARPACK did not converge"):
        modal_sparse.generalized_eigsh(stiffness, mass, 12, sigma=sigma, maxiter=1, ncv=13)

    # A positive shift between higher modes makes ARPACK skip the lowest ones; the Sturm sequence
    # count detects them and a locked restart recovers them, or the search fails loudly.
    fixed = np.r_[0:6, 9:12]
    free = np.setdiff1d(np.arange(stiffness.shape[0]), fixed)
    clamped_stiffness = stiffness[free][:, free]
    clamped_mass = mass[free][:, free]
    reference = np.linalg.eigvalsh(clamped_stiffness.toarray() / 8.0e-3)
    recovered, _ = modal_sparse.lowest_generalized_eigenpairs(
        clamped_stiffness, clamped_mass, 4, sigma=0.5 * (reference[4] + reference[5]), zero_threshold=0.0
    )
    np.testing.assert_allclose(recovered[:4], reference[:4], rtol=1.0e-8)
    with test.assertRaisesRegex(ValueError, "Sturm sequence count"):
        modal_sparse.lowest_generalized_eigenpairs(
            clamped_stiffness, clamped_mass, 4, sigma=0.5 * (reference[6] + reference[7]), zero_threshold=0.0
        )


def test_fem_neighbors_and_coincident_samples(test, device):
    """Match brute-force neighbors with the k-d tree and copy samples that coincide with nodes."""
    rng = np.random.default_rng(5)
    cloud = rng.uniform(-1.0, 1.0, size=(2600, 3))
    grid = np.stack(np.meshgrid(*(np.arange(n) for n in (16, 14, 12)), indexing="ij"), axis=-1).reshape((-1, 3))
    for points in (cloud, grid.astype(np.float64) * 0.01):
        neighbors = _nearest_neighbors(points, 12)
        for point in range(0, points.shape[0], 97):
            others = np.delete(np.arange(points.shape[0]), point)
            dist2 = np.sum((points[others] - points[point]) ** 2, axis=1)
            np.testing.assert_array_equal(neighbors[point], others[np.lexsort((others, dist2))[:12]])

    # A rigid rotation field has psi equal to its rotation vector on the k-d tree path.
    omega = np.array([0.02, -0.01, 0.03])
    phi = np.cross(omega, cloud)[:, None, :]
    np.testing.assert_allclose(_estimate_sample_psi(cloud, phi)[:, 0], np.tile(omega, (2600, 1)), atol=1.0e-6)

    points, mass, stiffness = _sparse_spring_lattice((4, 3, 3), 0.01, 2.0e9, 8.0e-3)
    nodes = np.array([7, 2, 30, 11])
    samples = np.vstack((points[nodes], points[:2] + 0.002))
    kwargs = {"mode_count": 5, "derive_psi": True, "eigensolver": "dense"}
    indexed = newton.ModalGeneratorFEM(
        points, mass, stiffness, sample_points=points[nodes], sample_node_indices=nodes, **kwargs
    )
    detected = newton.ModalGeneratorFEM(points, mass, stiffness, sample_points=samples, **kwargs)
    indexed_basis = indexed.build()
    detected_basis = detected.build()
    np.testing.assert_array_equal(detected_basis.sample_phi[:4], indexed_basis.sample_phi)
    np.testing.assert_array_equal(detected_basis.sample_psi[:4], indexed_basis.sample_psi)
    nodal = newton.ModalBasis(
        sample_points=points.astype(np.float32),
        sample_phi=np.transpose(detected.modal_matrix.reshape((-1, 3, 5)), (0, 2, 1)),
        sample_psi=np.zeros((points.shape[0], 5, 3), dtype=np.float32),
    )
    for sample in (4, 5):
        np.testing.assert_array_equal(detected_basis.sample_phi[sample], nodal.evaluate(samples[sample])[0])


def test_fem_dense_path_without_scipy(test, device):
    """Keep the dense path working without SciPy and explain how to enable the ARPACK path."""
    tet = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    bars = [(a, b) for a in range(5) for b in range(a + 1, 5)]
    stiffness = _truss_stiffness(tet, bars, [1.0e3] * len(bars))
    blocked = {name: None for name in list(sys.modules) if name == "scipy" or name.startswith("scipy.")}
    blocked["scipy"] = None
    with mock.patch.dict(sys.modules, blocked):
        generator = newton.ModalGeneratorFEM(
            tet, np.eye(15), stiffness, mode_count=4, derive_psi=True, dense_dof_limit=4
        )
        test.assertFalse(generator._uses_arpack(15))
        basis = generator.build()
        test.assertEqual(basis.mode_count, 4)
        test.assertEqual(generator.rigid_mode_count, 6)
        with test.assertRaisesRegex(ImportError, "pip install scipy"):
            newton.ModalGeneratorFEM(tet, np.eye(15), stiffness, mode_count=4, eigensolver="arpack").build()


class TestModalFemSparse(unittest.TestCase):
    pass


_TESTS = (
    test_fem_arpack_matches_dense_free_free,
    test_fem_arpack_matches_dense_clamped,
    test_fem_sparse_input_dense_path_identical,
    test_fem_arpack_wide_spectrum_keeps_soft_modes,
    test_fem_arpack_rejects_invalid_models,
    test_fem_arpack_rejects_invalid_shifts,
    test_fem_arpack_sturm_check_never_skipped_silently,
    test_fem_arpack_residual_check,
    test_fem_arpack_diagnoses_indefinite_matrices,
    test_fem_mode_sign_convention,
    test_fem_arpack_mechanisms_discarded,
    test_fem_arpack_large_sparse_model,
    test_fem_helpers,
    test_fem_neighbors_and_coincident_samples,
)

add_function_test(
    TestModalFemSparse, "test_fem_dense_path_without_scipy", test_fem_dense_path_without_scipy, devices=None
)
if HAS_SCIPY:
    for _test in _TESTS:
        add_function_test(TestModalFemSparse, _test.__name__, _test, devices=None)
else:
    for _test in _TESTS:
        setattr(
            TestModalFemSparse,
            _test.__name__,
            unittest.skip("SciPy is required for sparse FEM matrices")(lambda self: None),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
