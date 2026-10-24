# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pluggable eigensolver backends of the finite-element modal generators.

These are setup-time NumPy/SciPy tests; they do not launch Warp kernels and
are registered once, independent of the test device.
"""

import importlib.util
import unittest
import warnings

import numpy as np
import warp as wp
import warp.sparse as wps

import newton
from newton._src.sim import modal_sparse
from newton.tests.test_modal_fem_sparse import _sparse_spring_lattice
from newton.tests.test_reduced_elastic_body import _spring_lattice
from newton.tests.unittest_utils import add_function_test

HAS_SCIPY = importlib.util.find_spec("scipy") is not None


class _RecordingBackend:
    """Backend that records its calls and returns the dense ``scipy.linalg.eigh`` spectrum outside ``deflation``."""

    name = "recording"
    supports_inertia_count = False
    device = "cpu"

    def __init__(self, inertia_offset=None):
        self.calls = []
        self.inertia_offset = inertia_offset

    def solve(
        self, stiffness, mass, count, *, deflation=None, sigma=None, tol=0.0, factorization=None, zero_threshold=0.0
    ):
        from scipy import linalg

        self.calls.append(
            {
                "stiffness": stiffness,
                "mass": mass,
                "count": count,
                "deflation": deflation,
                "sigma": sigma,
                "tol": tol,
                "factorization": factorization,
                "zero_threshold": zero_threshold,
            }
        )
        stiffness = modal_sparse.to_dense(stiffness)
        mass = modal_sparse.to_dense(mass)
        complement = np.eye(stiffness.shape[0])
        if deflation is not None and deflation.shape[1] > 0:
            complement = linalg.null_space((mass @ deflation).T)
        values, vectors = linalg.eigh(complement.T @ stiffness @ complement, complement.T @ mass @ complement)
        vectors = complement @ vectors
        if self.inertia_offset is None:
            return newton.ModalEigenResult(values, vectors, backend=self.name)
        # Return a partial spectrum with a deliberately wrong Sturm sequence count.
        deflated = 0 if deflation is None else deflation.shape[1]
        shift = 0.5 * (values[count - 1] + values[count])
        return newton.ModalEigenResult(
            values[:count],
            vectors[:, :count],
            inertia_count=deflated + count + self.inertia_offset,
            backend=self.name,
            inertia_shift=shift,
        )


def _clamped_problem():
    """Return CSC stiffness [N/m] and mass [kg] of a lattice with its ``x = min`` face clamped, and its points."""
    points, mass, stiffness = _sparse_spring_lattice((6, 4, 3), 0.01, 2.0e9, 8.0e-3)
    clamped = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].min()))
    fixed = np.zeros(mass.shape[0], dtype=bool)
    fixed[(3 * clamped[:, None] + np.arange(3)).reshape(-1)] = True
    free = np.flatnonzero(~fixed)
    return points, clamped, modal_sparse.submatrix(stiffness, free, free), modal_sparse.submatrix(mass, free, free)


def _rigid_deflation(points, mass):
    """Return the ``M``-orthonormal rigid-body motions of a lattice."""
    center = np.mean(points, axis=0)
    rigid = modal_sparse.rigid_vectors(points, 3, center)
    gram = rigid.T @ modal_sparse.matmul(mass, rigid)
    values, rotation = np.linalg.eigh(0.5 * (gram + gram.T))
    return rigid @ (rotation / np.sqrt(values))


def _assert_same_eigenpairs(test, mass, values, vectors, reference_values, reference_vectors):
    """Assert equal eigenvalues (relative ``1e-8``) and ``M``-weighted principal angles below ``1e-6`` rad."""
    np.testing.assert_allclose(values, reference_values, rtol=1.0e-8)
    cosines = np.linalg.svd(vectors.T @ modal_sparse.matmul(mass, reference_vectors), compute_uv=False)
    angles = np.arccos(np.clip(cosines, -1.0, 1.0))
    test.assertLess(float(np.max(angles)), 1.0e-6)


def test_backend_protocol_and_resolution(test, device):
    """Resolve names and enum members to the same backends, and accept duck-typed backends unchanged."""
    for backend in (
        newton.ModalDenseBackend(),
        newton.ModalScipyArpackBackend(),
        newton.ModalSubspaceIterationBackend(),
        modal_sparse.WarpackBackend(),
        _RecordingBackend(),
    ):
        test.assertIsInstance(backend, newton.ModalEigensolverBackend)
        test.assertIs(modal_sparse.resolve_eigensolver(backend), backend)
    for member in newton.ModalEigensolver:
        test.assertIs(
            type(modal_sparse.resolve_eigensolver(member)), type(modal_sparse.resolve_eigensolver(member.value))
        )
    test.assertIsInstance(modal_sparse.resolve_eigensolver("dense"), newton.ModalDenseBackend)
    test.assertIsInstance(modal_sparse.resolve_eigensolver("arpack"), newton.ModalScipyArpackBackend)
    test.assertIsInstance(modal_sparse.resolve_eigensolver("auto"), newton.ModalScipyArpackBackend)
    subspace = modal_sparse.resolve_eigensolver(newton.ModalEigensolver.SUBSPACE, margin=4)
    test.assertIsInstance(subspace, newton.ModalSubspaceIterationBackend)
    test.assertEqual(subspace.margin, 4)
    test.assertTrue(subspace.supports_inertia_count)
    with test.assertRaisesRegex(ValueError, "eigensolver must be"):
        modal_sparse.resolve_eigensolver("lobpcg")
    with test.assertRaisesRegex(TypeError, "eigensolver must be"):
        modal_sparse.resolve_eigensolver(3)
    with test.assertRaisesRegex(TypeError, "eigensolver must be"):
        newton.ModalGeneratorFEM(np.zeros((2, 3)), np.eye(6), np.eye(6), eigensolver=object())


def test_enum_and_string_select_identical_paths(test, device):
    """Build bit-identical bases from an eigensolver name and from the matching ``ModalEigensolver`` member."""
    points, mass, stiffness = _sparse_spring_lattice((5, 3, 3), 0.01, 2.0e9, 8.0e-3)
    for name in ("dense", "arpack", "auto"):
        by_name = newton.ModalGeneratorFEM(points, mass, stiffness, mode_count=6, eigensolver=name)
        by_enum = newton.ModalGeneratorFEM(
            points, mass, stiffness, mode_count=6, eigensolver=newton.ModalEigensolver(name)
        )
        test.assertEqual(by_enum.eigensolver, name)
        basis_name = by_name.build()
        basis_enum = by_enum.build()
        np.testing.assert_array_equal(by_enum.eigenvalues, by_name.eigenvalues)
        np.testing.assert_array_equal(by_enum.modal_matrix, by_name.modal_matrix)
        np.testing.assert_array_equal(basis_enum.sample_phi, basis_name.sample_phi)
    # An explicit default ARPACK backend runs the same computation as the name.
    explicit = newton.ModalGeneratorFEM(
        points, mass, stiffness, mode_count=6, eigensolver=newton.ModalScipyArpackBackend()
    )
    explicit.build()
    np.testing.assert_array_equal(explicit.eigenvalues, by_name.eigenvalues)
    np.testing.assert_array_equal(explicit.modal_matrix, by_name.modal_matrix)


def test_fake_backend_drives_fem_generator(test, device):
    """Call a recording backend with the free-DOF CSC matrices, rigid deflation, and shift, and use its result."""
    points, mass, stiffness = _sparse_spring_lattice((5, 3, 3), 0.01, 2.0e9, 8.0e-3)
    reference = newton.ModalGeneratorFEM(points, mass, stiffness, eigensolver="dense")
    reference_basis = reference.build()
    backend = _RecordingBackend()
    # Nine modes end at a spectral gap; the eighth and ninth eigenvalues are equal.
    generator = newton.ModalGeneratorFEM(points, mass, stiffness, mode_count=9, eigensolver=backend)
    test.assertIs(generator.eigensolver, backend)
    with warnings.catch_warnings():
        # The backend returns the full spectrum, so no Sturm sequence warning may be raised.
        warnings.simplefilter("error")
        basis = generator.build()
    test.assertEqual(len(backend.calls), 1)
    call = backend.calls[0]
    test.assertEqual(call["count"], 9)
    test.assertTrue(modal_sparse.is_sparse(call["stiffness"]))
    test.assertTrue(modal_sparse.is_sparse(call["mass"]))
    test.assertEqual(call["deflation"].shape, (mass.shape[0], 6))
    test.assertLess(call["sigma"], 0.0)
    test.assertIsInstance(call["factorization"], modal_sparse.ShiftedFactorization)
    test.assertEqual(call["factorization"].sigma, call["sigma"])
    test.assertGreater(call["zero_threshold"], 0.0)
    np.testing.assert_allclose(generator.eigenvalues, reference.eigenvalues[:9], rtol=1.0e-8)
    np.testing.assert_allclose(basis.mode_stiffness, reference_basis.mode_stiffness[:9], rtol=1.0e-8)
    test.assertEqual(generator.rigid_mode_count, 6)

    # A partial spectrum whose Sturm sequence count disagrees is rejected.
    for offset in (-1, 1):
        wrong = newton.ModalGeneratorFEM(
            points, mass, stiffness, mode_count=9, eigensolver=_RecordingBackend(inertia_offset=offset)
        )
        with test.assertRaisesRegex(ValueError, "Sturm sequence count"):
            wrong.build()


def test_fake_backend_drives_craig_bampton_generator(test, device):
    """Solve the fixed-interface eigenproblem of a Craig-Bampton reduction through a recording backend."""
    from scipy import sparse

    points, mass, stiffness = _spring_lattice((3, 3, 3), 0.1, 1000.0, 0.01)
    top = np.flatnonzero(np.isclose(points[:, 2], points[:, 2].max()))
    interfaces = [{"node_indices": top, "coupling": "rbe3"}]
    matrices = (sparse.csc_matrix(mass), sparse.csc_matrix(stiffness))
    reference = newton.ModalGeneratorCraigBamptonFEM(
        points, *matrices, interfaces, fixed_mode_count=8, dofs_per_node=3, eigensolver="arpack"
    )
    reference.reduce()
    backend = _RecordingBackend()
    generator = newton.ModalGeneratorCraigBamptonFEM(
        points, *matrices, interfaces, fixed_mode_count=8, dofs_per_node=3, eigensolver=backend
    )
    generator.reduce()
    test.assertEqual(generator.report["eigensolver"], "recording")
    test.assertEqual(len(backend.calls), 1)
    test.assertEqual(backend.calls[0]["count"], 8)
    test.assertIsNone(backend.calls[0]["deflation"])
    test.assertEqual(backend.calls[0]["sigma"], 0.0)
    np.testing.assert_allclose(
        generator.fixed_interface_eigenvalues, reference.fixed_interface_eigenvalues, rtol=1.0e-9
    )
    # Repeated fixed-interface eigenvalues leave the basis of their eigenspaces arbitrary, so compare
    # the interface blocks and the rotation-invariant Gram matrix of the mass coupling.
    scale = float(np.max(np.abs(reference.reduced_stiffness_matrix)))
    np.testing.assert_allclose(
        generator.reduced_stiffness_matrix[:6, :6],
        reference.reduced_stiffness_matrix[:6, :6],
        rtol=0.0,
        atol=1.0e-8 * scale,
    )
    np.testing.assert_allclose(
        generator.reduced_mass_matrix[:6, :6], reference.reduced_mass_matrix[:6, :6], rtol=0.0, atol=1.0e-9
    )
    coupling = generator.reduced_mass_matrix[:6, 6:]
    reference_coupling = reference.reduced_mass_matrix[:6, 6:]
    np.testing.assert_allclose(coupling @ coupling.T, reference_coupling @ reference_coupling.T, atol=1.0e-9)

    subspace = newton.ModalGeneratorCraigBamptonFEM(
        points, *matrices, interfaces, fixed_mode_count=8, dofs_per_node=3, eigensolver="subspace"
    )
    subspace.reduce()
    test.assertEqual(subspace.report["eigensolver"], "subspace")
    np.testing.assert_allclose(subspace.fixed_interface_eigenvalues, reference.fixed_interface_eigenvalues, rtol=1.0e-8)


def test_subspace_iteration_matches_arpack(test, device):
    """Agree with ARPACK to relative ``1e-8`` with and without rigid deflation, cold and warm started."""
    points, mass, stiffness = _sparse_spring_lattice((6, 4, 3), 0.01, 2.0e9, 8.0e-3)
    _, _, clamped_stiffness, clamped_mass = _clamped_problem()
    count = 8
    cases = (
        ("free-free, deflated", stiffness, mass, _rigid_deflation(points, mass)),
        ("clamped", clamped_stiffness, clamped_mass, None),
    )
    for label, k, m, deflation in cases:
        with test.subTest(label):
            sigma = modal_sparse.default_shift(k, m)
            arpack = newton.ModalScipyArpackBackend().solve(k, m, count, deflation=deflation, sigma=sigma)
            reference_values = arpack.values[:count]
            reference_vectors = arpack.vectors[:, :count]
            # The mode count must end at a gap for the subspace comparison to be well posed.
            test.assertGreater((arpack.values[count] - arpack.values[count - 1]) / arpack.values[count], 1.0e-3)

            cold = newton.ModalSubspaceIterationBackend(margin=4, sturm_check=True)
            result = cold.solve(k, m, count, deflation=deflation, sigma=sigma)
            test.assertEqual(result.backend, "subspace")
            _assert_same_eigenpairs(test, m, result.values, result.vectors, reference_values, reference_vectors)
            deflated = 0 if deflation is None else deflation.shape[1]
            test.assertEqual(result.inertia_count, deflated + count)
            test.assertLess(result.values[-1], result.inertia_shift)
            if deflation is not None:
                np.testing.assert_allclose(deflation.T @ modal_sparse.matmul(m, result.vectors), 0.0, atol=1.0e-10)
            test.assertGreater(cold.last_iterations, 2)

            # Warm start from slightly perturbed modes: a couple of iterations suffice.
            rng = np.random.default_rng(7)
            previous = (
                reference_vectors
                + 1.0e-3 * rng.standard_normal(reference_vectors.shape) * np.abs(reference_vectors).max()
            )
            warm = newton.ModalSubspaceIterationBackend(margin=4, previous_vectors=previous)
            warm_result = warm.solve(k, m, count, deflation=deflation, sigma=sigma)
            _assert_same_eigenpairs(
                test, m, warm_result.values, warm_result.vectors, reference_values, reference_vectors
            )
            test.assertLess(warm.last_iterations, cold.last_iterations)
            test.assertIs(warm.previous_vectors, warm_result.vectors)

    # Exactly one iteration without a convergence requirement returns a result and its residuals.
    fixed = newton.ModalSubspaceIterationBackend(iterations=1, require_convergence=False)
    fixed.solve(clamped_stiffness, clamped_mass, count)
    test.assertEqual(fixed.last_iterations, 1)
    test.assertEqual(fixed.last_residuals.shape, (count,))
    with test.assertRaisesRegex(ValueError, "Subspace iteration did not converge"):
        newton.ModalSubspaceIterationBackend(max_iterations=1, margin=0).solve(clamped_stiffness, clamped_mass, count)


def test_subspace_backend_drives_fem_generator(test, device):
    """Build the same free-free basis with ``eigensolver="subspace"`` as with ARPACK."""
    points, mass, stiffness = _sparse_spring_lattice((5, 3, 3), 0.01, 2.0e9, 8.0e-3)
    arpack = newton.ModalGeneratorFEM(points, mass, stiffness, mode_count=9, eigensolver="arpack")
    arpack.build()
    subspace = newton.ModalGeneratorFEM(
        points, mass, stiffness, mode_count=9, eigensolver=newton.ModalEigensolver.SUBSPACE
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        subspace.build()
    test.assertIsInstance(subspace.eigensolver, newton.ModalSubspaceIterationBackend)
    np.testing.assert_allclose(subspace.eigenvalues, arpack.eigenvalues, rtol=1.0e-8)


def test_warpack_backend_placeholder(test, device):
    """Raise ``NotImplementedError`` from the Warpack placeholder, and run a subclass without a host factorization."""
    points, mass, stiffness = _sparse_spring_lattice((5, 3, 3), 0.01, 2.0e9, 8.0e-3)
    placeholder = modal_sparse.WarpackBackend()
    test.assertFalse(placeholder.supports_inertia_count)
    test.assertNotEqual(placeholder.device, "cpu")
    with test.assertRaisesRegex(NotImplementedError, "Warpack backend is not available in this build"):
        placeholder.solve(stiffness, mass, 4)
    with test.assertRaisesRegex(NotImplementedError, "Warpack backend is not available in this build"):
        newton.ModalGeneratorFEM(points, mass, stiffness, mode_count=4, eigensolver=placeholder).build()

    class PartialDense(modal_sparse.WarpackBackend):
        """Device-tagged backend that returns only the wanted eigenpairs and no Sturm sequence count."""

        def __init__(self):
            super().__init__(device="cuda:0")
            self.factorizations = []

        def solve(
            self, stiffness, mass, count, *, deflation=None, sigma=None, tol=0.0, factorization=None, zero_threshold=0.0
        ):
            self.factorizations.append(factorization)
            full = newton.ModalDenseBackend().solve(stiffness, mass, count, deflation=deflation)
            return newton.ModalEigenResult(full.values[: count + 1], full.vectors[:, : count + 1], backend="partial")

    backend = PartialDense()
    generator = newton.ModalGeneratorFEM(points, mass, stiffness, mode_count=4, eigensolver=backend)
    with test.assertWarnsRegex(UserWarning, "Sturm sequence check was skipped"):
        generator.build()
    test.assertEqual(backend.factorizations, [None])
    reference = newton.ModalGeneratorFEM(points, mass, stiffness, mode_count=4, eigensolver="arpack")
    reference.build()
    test.assertEqual(generator.eigenvalues.shape, (4,))
    np.testing.assert_allclose(generator.eigenvalues, reference.eigenvalues, rtol=1.0e-8)


def test_backend_result_validation_and_bsr_input(test, device):
    """Reject malformed backend results, and convert a Warp BSR matrix to the host CSC form."""
    _, _, stiffness = _sparse_spring_lattice((3, 3, 2), 0.01, 2.0e9, 8.0e-3)
    size = stiffness.shape[0]
    values = np.arange(1.0, 5.0)
    with test.assertRaisesRegex(ValueError, "eigenvectors of shape"):
        modal_sparse.validate_eigen_result(
            newton.ModalEigenResult(values, np.zeros((size, 3))),
            size=size,
            count=4,
            deflated=0,
            supports_inertia_count=True,
        )
    with test.assertRaisesRegex(ValueError, "fewer than the 4 requested"):
        modal_sparse.validate_eigen_result(
            newton.ModalEigenResult(values[:3], np.zeros((size, 3))),
            size=size,
            count=4,
            deflated=0,
            supports_inertia_count=True,
        )
    with test.assertRaisesRegex(ValueError, "non-finite"):
        modal_sparse.validate_eigen_result(
            newton.ModalEigenResult(np.array([np.nan]), np.zeros((size, 1))),
            size=size,
            count=1,
            deflated=0,
            supports_inertia_count=True,
        )

    coo = stiffness.tocoo()
    with wp.ScopedDevice("cpu"):
        bsr = wps.bsr_from_triplets(
            size,
            size,
            wp.array(coo.row.astype(np.int32)),
            wp.array(coo.col.astype(np.int32)),
            wp.array(coo.data.astype(np.float64)),
        )
    test.assertTrue(modal_sparse.is_bsr_matrix(bsr))
    test.assertFalse(modal_sparse.is_bsr_matrix(stiffness))
    host = modal_sparse.to_backend_matrix(bsr)
    test.assertTrue(modal_sparse.is_sparse(host))
    np.testing.assert_allclose(host.toarray(), stiffness.toarray(), rtol=0.0, atol=0.0)
    test.assertIs(modal_sparse.to_backend_matrix(bsr, device="cuda:0"), bsr)
    test.assertIs(modal_sparse.to_backend_matrix(stiffness), stiffness)


class TestModalEigensolverBackend(unittest.TestCase):
    pass


_TESTS = (
    test_backend_protocol_and_resolution,
    test_enum_and_string_select_identical_paths,
    test_fake_backend_drives_fem_generator,
    test_fake_backend_drives_craig_bampton_generator,
    test_subspace_iteration_matches_arpack,
    test_subspace_backend_drives_fem_generator,
    test_warpack_backend_placeholder,
    test_backend_result_validation_and_bsr_input,
)

if HAS_SCIPY:
    for _test in _TESTS:
        add_function_test(TestModalEigensolverBackend, _test.__name__, _test, devices=None)
else:
    for _test in _TESTS:
        setattr(
            TestModalEigensolverBackend,
            _test.__name__,
            unittest.skip("SciPy is required for the eigensolver backends")(lambda self: None),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
