# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dense/sparse matrix helpers for finite-element modal generators.

The helpers accept either dense :class:`numpy.ndarray` matrices or SciPy sparse
matrices and arrays. SciPy is an optional dependency, so it is imported only
when a sparse code path runs. Dense inputs follow the exact operation order of
the original dense generator code, so dense results are unchanged.
"""

from __future__ import annotations

import enum
import importlib
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

SHIFT_FRACTION_DEFAULT = 1.0e-6
"""Default magnitude of the ARPACK shift relative to ``||K||_1 / ||M||_1`` [dimensionless]."""

SYMMETRY_TOLERANCE = 1.0e-8
"""Largest relative asymmetry ``||A - A^T||_F / ||A||_F`` of an input matrix that is treated as roundoff [dimensionless]."""

_ARPACK_START_SEED = 0x5EED
"""Seed of the deterministic ARPACK starting vector."""

SINGULAR_PIVOT_RATIO = 1.0e-12
"""Pivot magnitude relative to the largest one at or below which a factorization is numerically singular [dimensionless].

Factoring the singular stiffness matrix of a free-free model leaves pivots of
roundoff size, ``1e-16`` to ``1e-13`` of the largest one, while the shifted
matrices of well-posed models give ratios above ``1e-8``.
"""

_RESIDUAL_TOLERANCE_FACTOR = 10.0
"""Allowed backward error of an ARPACK eigenpair relative to ``max(tol, _RESIDUAL_EPS_MULTIPLE * eps)``."""

_RESIDUAL_EPS_MULTIPLE = 1.0e3
"""Multiple of machine epsilon that replaces the ARPACK tolerance ``0`` (machine precision) in the residual check."""


def is_sparse(matrix: Any) -> bool:
    """Return whether ``matrix`` is a SciPy sparse matrix or array.

    Without a loaded :mod:`scipy.sparse` module no object can be sparse, so
    this check never imports SciPy.
    """
    sparse = sys.modules.get("scipy.sparse")
    return sparse is not None and bool(sparse.issparse(matrix))


def has_scipy() -> bool:
    """Return whether SciPy, needed for sparse matrices and the ARPACK eigensolver, can be imported."""
    try:
        import scipy  # noqa: F401
    except ImportError:
        return False
    return True


def _import_scipy(module: str):
    """Import a SciPy submodule lazily, raising an :class:`ImportError` with an install hint if SciPy is missing."""
    try:
        return importlib.import_module(f"scipy.{module}")
    except ImportError as exc:
        raise ImportError(
            "Sparse finite-element matrices and the ARPACK eigensolver require SciPy; "
            "install it with `pip install scipy` or the `newton[importers]` extra"
        ) from exc


def to_csc(matrix: Any) -> Any:
    """Return ``matrix`` as a canonical float64 CSC sparse matrix."""
    sparse = _import_scipy("sparse")
    result = sparse.csc_matrix(matrix, dtype=np.float64)
    result.sum_duplicates()
    result.sort_indices()
    return result


def to_dense(matrix: Any) -> np.ndarray:
    """Return ``matrix`` as a dense float64 array, without copying dense float64 input."""
    if is_sparse(matrix):
        # C order, as for dense input, so BLAS reductions round identically.
        return np.ascontiguousarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def relative_asymmetry(matrix: Any) -> float:
    """Return ``||A - A^T||_F / ||A||_F`` of a dense or sparse square matrix, or ``0`` for a zero matrix."""
    if is_sparse(matrix):
        csc = to_csc(matrix)
        norm = float(np.linalg.norm(csc.data))
        difference = float(np.linalg.norm(to_csc(csc - csc.T).data))
    else:
        array = np.asarray(matrix, dtype=np.float64)
        norm = float(np.linalg.norm(array))
        difference = float(np.linalg.norm(array - array.T))
    return difference / norm if norm > 0.0 else 0.0


def check_symmetric(matrix: Any, name: str) -> None:
    """Raise if the relative asymmetry of ``matrix`` exceeds :data:`SYMMETRY_TOLERANCE`.

    Finite-element tools often export symmetric matrices as one triangle.
    Symmetrizing such a matrix would silently halve its off-diagonal terms,
    so only roundoff-sized asymmetry is accepted.
    """
    asymmetry = relative_asymmetry(matrix)
    if asymmetry > SYMMETRY_TOLERANCE:
        raise ValueError(
            f"{name} is not symmetric: its relative asymmetry ||A - A^T||_F / ||A||_F is {asymmetry:.3g}, above "
            f"{SYMMETRY_TOLERANCE:g}. A matrix stored as one triangle must be expanded to the full matrix first, "
            "for example to A + A^T - diag(A); scipy.io.mmread and newton.utils.load_sparse_matrix expand Matrix "
            "Market files declared 'symmetric', but not 'general' files that hold one triangle"
        )


def coerce_symmetric_matrix(matrix: Any, size: int, name: str, *, keep_sparse: bool = True) -> Any:
    """Validate a square symmetric matrix and return its symmetric part in float64.

    Asymmetry up to :data:`SYMMETRY_TOLERANCE`, such as assembly roundoff, is
    removed; larger asymmetry raises (see :func:`check_symmetric`).

    Args:
        matrix: Dense array-like or SciPy sparse matrix.
        size: Required row and column count.
        name: Argument name used in error messages.
        keep_sparse: Whether sparse input stays sparse (CSC). Otherwise it is
            densified before symmetrization, which matches dense input exactly.

    Returns:
        ``0.5 * (A + A^T)`` as a dense array, or as a CSC matrix for sparse
        input with ``keep_sparse``.

    Raises:
        ValueError: If the shape is not ``(size, size)``, sparse entries are
            not finite, or the matrix is not symmetric.
    """
    if is_sparse(matrix):
        if tuple(matrix.shape) != (size, size):
            raise ValueError(f"{name} must have shape ({size}, {size}), got {tuple(matrix.shape)}")
        if not keep_sparse:
            array = np.ascontiguousarray(matrix.toarray(), dtype=np.float64)
            check_symmetric(array, name)
            return 0.5 * (array + array.T)
        csc = to_csc(matrix)
        if not np.all(np.isfinite(csc.data)):
            raise ValueError(f"{name} contains non-finite entries")
        check_symmetric(csc, name)
        return to_csc(0.5 * (csc + csc.T))
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {array.shape}")
    check_symmetric(array, name)
    return 0.5 * (array + array.T)


def matrix_inf_norm(matrix: Any) -> float:
    """Return the infinity norm (maximum absolute row sum) of a dense or sparse matrix."""
    if is_sparse(matrix):
        if matrix.shape[0] == 0:
            return 0.0
        return float(np.max(np.asarray(abs(matrix).sum(axis=1)).reshape(-1)))
    return float(np.linalg.norm(matrix, ord=np.inf))


def matrix_one_norm(matrix: Any) -> float:
    """Return the one norm (maximum absolute column sum) of a dense or sparse matrix."""
    if is_sparse(matrix):
        if matrix.shape[1] == 0:
            return 0.0
        return float(np.max(np.asarray(abs(matrix).sum(axis=0)).reshape(-1)))
    return float(np.linalg.norm(matrix, ord=1))


def submatrix(matrix: Any, rows: np.ndarray, cols: np.ndarray) -> Any:
    """Return ``matrix[rows][:, cols]``; sparse input stays CSC and full selections are not copied."""
    if is_sparse(matrix):
        n_rows, n_cols = matrix.shape
        if rows.shape[0] == n_rows and cols.shape[0] == n_cols:
            if np.array_equal(rows, np.arange(n_rows)) and np.array_equal(cols, np.arange(n_cols)):
                return matrix
        return to_csc(to_csc(matrix)[rows, :][:, cols])
    return matrix[np.ix_(rows, cols)]


def matmul(matrix: Any, vectors: np.ndarray) -> np.ndarray:
    """Return ``matrix @ vectors`` as a dense array for dense or sparse ``matrix``."""
    return np.asarray(matrix @ vectors)


def bilinear(left: np.ndarray, matrix: Any, right: np.ndarray) -> np.ndarray:
    """Return ``left^T A right`` without densifying a sparse ``A``.

    Dense input is evaluated as ``left.T @ A @ right``, the original operation
    order; sparse input as ``left.T @ (A @ right)``.
    """
    if is_sparse(matrix):
        return left.T @ matmul(matrix, right)
    return left.T @ matrix @ right


def quadratic_form(vector: np.ndarray, matrix: Any) -> float:
    """Return ``v^T A v`` for a single vector ``v``."""
    if is_sparse(matrix):
        return float(vector @ matmul(matrix, vector))
    return float(vector @ matrix @ vector)


def quadratic_form_diag(vectors: np.ndarray, matrix: Any) -> np.ndarray:
    """Return ``diag(X^T A X)`` without forming an ``[n, n]`` intermediate."""
    if is_sparse(matrix):
        return np.einsum("di,di->i", vectors, matmul(matrix, vectors))
    return np.einsum("di,de,ei->i", vectors, matrix, vectors)


def max_row_nonzeros(matrix: Any) -> int:
    """Return the largest number of stored or nonzero entries in any row."""
    if is_sparse(matrix):
        csr = matrix.tocsr()
        if csr.shape[0] == 0:
            return 0
        return int(np.max(np.diff(csr.indptr)))
    if matrix.shape[0] == 0:
        return 0
    return int(np.max(np.count_nonzero(matrix, axis=1)))


def rigid_vectors(positions: np.ndarray, dofs_per_node: int, center: np.ndarray) -> np.ndarray:
    """Return nodal rigid translations and rotations about ``center``, shape ``[node_count * dofs_per_node, 6]``.

    Columns ``0..2`` translate every node along ``x``, ``y``, and ``z``;
    columns ``3..5`` rotate the body about ``center`` by ``omega x (x - c)``.
    With ``dofs_per_node == 6`` each node additionally carries rotational DOFs
    ``[rx, ry, rz]`` after its translations, which equal the unit rotation.

    Args:
        positions: Node positions [m], shape ``[node_count, 3]``.
        dofs_per_node: Number of DOFs per node, 3 or 6.
        center: Rotation center [m], shape ``[3]``.

    Returns:
        Rigid-body displacement fields [m per unit translation or rad].
    """
    if dofs_per_node not in (3, 6):
        raise ValueError(f"dofs_per_node must be 3 or 6, got {dofs_per_node}")
    positions = np.asarray(positions, dtype=np.float64)
    node_count = positions.shape[0]
    rigid = np.zeros((node_count, dofs_per_node, 6), dtype=np.float64)
    rigid[:, :3, :3] = np.eye(3)
    arm = positions - np.asarray(center, dtype=np.float64)
    for axis in range(3):
        unit = np.zeros(3, dtype=np.float64)
        unit[axis] = 1.0
        rigid[:, :3, 3 + axis] = np.cross(unit, arm)
    if dofs_per_node == 6:
        rigid[:, 3:, 3:] = np.eye(3)
    return rigid.reshape((node_count * dofs_per_node, 6))


def generalized_eigenvalue_bound(stiffness: Any, mass: Any) -> float:
    """Estimate the largest generalized eigenvalue of ``K phi = lambda M phi`` [1/s^2].

    Returns the Gershgorin-type bound ``max_i sum_j |K_ij| / M_ii``. It is an
    upper bound for a diagonal (lumped) mass matrix; for a consistent mass
    matrix it is an estimate, which suffices as the scale of a relative
    zero-energy threshold when it is of the same order of magnitude.

    On three-DOF lattices with lumped or consistent mass the bound is 1.2 to
    1.9 times the largest eigenvalue. Consistent-mass shells with rotational
    DOFs inflate it, because their rotary-inertia diagonals are small against
    the off-diagonal coupling: on a six-DOF shell panel it is 41 times the
    largest eigenvalue, which puts a threshold of ``1e-12`` relative to it at
    about 0.6 Hz, close to genuine soft modes. A tighter scale, for example
    from a few power iterations on ``M^-1 K``, is needed before such models
    rely on a threshold relative to this bound.
    """
    if is_sparse(stiffness):
        row_sums = np.asarray(abs(stiffness).sum(axis=1)).reshape(-1)
    else:
        row_sums = np.sum(np.abs(stiffness), axis=1)
    diagonal = np.asarray(mass.diagonal(), dtype=np.float64).reshape(-1)
    if row_sums.shape[0] == 0:
        return 0.0
    return float(np.max(row_sums / np.maximum(diagonal, np.finfo(np.float64).tiny)))


def default_shift(stiffness: Any, mass: Any, shift_fraction: float = SHIFT_FRACTION_DEFAULT) -> float:
    """Return a negative ARPACK shift ``sigma`` [1/s^2] for a possibly singular stiffness matrix.

    Shift-invert Lanczos iterates with ``(K - sigma M)^-1 M`` and converges
    first to the eigenvalues closest to ``sigma``. For a free-free model ``K``
    is only positive semi-definite, so ``sigma = 0`` would make the operator
    singular. A mildly negative ``sigma`` keeps ``K - sigma M`` symmetric
    positive definite (so a factorization without pivoting exists and its
    inertia certifies definiteness), turns the rigid-body modes into the
    largest eigenvalues ``1 / |sigma|`` of the inverse operator, and keeps the
    lowest elastic modes, which follow them, well resolved. ``|sigma|`` is
    ``shift_fraction`` of ``||K||_1 / ||M||_1``, an estimate of the largest
    eigenvalue, so the shifted matrix stays well conditioned
    (condition number about ``1 / shift_fraction``) independent of units.

    Args:
        stiffness: Stiffness matrix [N/m].
        mass: Mass matrix [kg].
        shift_fraction: Shift magnitude relative to ``||K||_1 / ||M||_1``.

    Returns:
        The shift ``sigma`` [1/s^2], always negative.
    """
    mass_norm = matrix_one_norm(mass)
    stiffness_norm = matrix_one_norm(stiffness)
    if mass_norm > 0.0 and stiffness_norm > 0.0:
        return -float(shift_fraction) * stiffness_norm / mass_norm
    return -1.0


def _symmetric_splu(matrix: Any) -> Any:
    """Factor a sparse symmetric matrix with SuperLU, symmetric ordering, and diagonal pivots."""
    sparse_linalg = _import_scipy("sparse.linalg")

    return sparse_linalg.splu(matrix, permc_spec="COLAMD", diag_pivot_thresh=0.0, options={"SymmetricMode": True})


def nonpositive_pivot_count(matrix: Any) -> int:
    """Return the number of non-positive eigenvalues of a symmetric sparse matrix, or ``-1``.

    Counts the non-positive pivots of a symmetric factorization (Sylvester's
    law), for example to certify that a mass matrix is positive definite.
    An exactly singular matrix counts one zero pivot; ``-1`` means SuperLU
    pivoted off the diagonal, so the inertia is unavailable.
    """
    try:
        lu = _symmetric_splu(to_csc(matrix))
    except RuntimeError:
        return 1
    if not np.array_equal(lu.perm_r, lu.perm_c):
        return -1
    return int(np.count_nonzero(np.asarray(lu.U.diagonal(), dtype=np.float64) <= 0.0))


def _check_mass_definite(mass: Any, failure: str) -> None:
    """Raise if ``mass`` has non-positive pivots, naming the eigensolver ``failure`` that it explains."""
    mass_pivots = nonpositive_pivot_count(mass)
    if mass_pivots > 0:
        raise ValueError(
            f"mass_matrix must be symmetric positive definite, but its factorization has {mass_pivots} "
            f"non-positive pivots: {failure}"
        )


class ShiftedFactorization:
    """Sparse LU factorization of ``K - sigma M`` for shift-invert eigensolves.

    The matrix is factored with SuperLU in symmetric mode (a COLAMD fill-reducing
    ordering applied to rows and columns alike, with diagonal pivots), so for a
    symmetric matrix the diagonal of
    ``U`` is the ``D`` of an ``L D L^T`` factorization and its signs give the
    matrix inertia (Sylvester's law): ``negative_pivot_count`` is the number of
    generalized eigenvalues below ``sigma`` when ``M`` is positive definite.
    SuperLU raises only for exactly zero pivots; ``pivot_ratio``, the
    smallest pivot magnitude relative to the largest, reveals a numerically
    singular matrix (see :attr:`numerically_singular`), for example ``K``
    of a free-free model factored at ``sigma = 0``. ``norm`` is the one norm
    of ``K - sigma M`` [N/m], the scale of eigenpair residuals.

    Args:
        stiffness: Sparse stiffness matrix [N/m], shape ``[n, n]``.
        mass: Sparse mass matrix [kg], shape ``[n, n]``.
        sigma: Shift [1/s^2].

    Raises:
        ValueError: If the shifted matrix is exactly singular.
    """

    def __init__(self, stiffness: Any, mass: Any, sigma: float):
        self.sigma = float(sigma)
        shifted = to_csc(to_csc(stiffness) - self.sigma * to_csc(mass))
        self.size = int(shifted.shape[0])
        self.norm = matrix_one_norm(shifted)
        try:
            self.lu = _symmetric_splu(shifted)
        except RuntimeError as exc:
            raise ValueError(
                f"stiffness_matrix - sigma * mass_matrix is singular for sigma = {self.sigma:.6g}; "
                "choose a different arpack_shift"
            ) from exc
        self.symmetric_pivoting = bool(np.array_equal(self.lu.perm_r, self.lu.perm_c))
        # SuperLU exposes U only as a copy; keep just its diagonal.
        pivots = np.asarray(self.lu.U.diagonal(), dtype=np.float64)
        magnitude = np.abs(pivots)
        largest = float(np.max(magnitude, initial=0.0))
        self.pivot_ratio = float(np.min(magnitude)) / largest if largest > 0.0 else float(self.size == 0)
        self.negative_pivot_count = -1
        if self.symmetric_pivoting:
            self.negative_pivot_count = int(np.count_nonzero(pivots <= 0.0))

    @property
    def numerically_singular(self) -> bool:
        """Whether the smallest pivot is at most :data:`SINGULAR_PIVOT_RATIO` of the largest."""
        return self.pivot_ratio <= SINGULAR_PIVOT_RATIO

    @property
    def fill_nonzeros(self) -> int:
        """Number of stored entries in the ``L`` and ``U`` factors (builds temporary copies)."""
        return int(self.lu.L.nnz + self.lu.U.nnz)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Return ``(K - sigma M)^-1 rhs``."""
        return self.lu.solve(np.asarray(rhs, dtype=np.float64))

    def operator(self) -> Any:
        """Return the inverse as a :class:`scipy.sparse.linalg.LinearOperator` for ``eigsh(OPinv=...)``."""
        sparse_linalg = _import_scipy("sparse.linalg")

        return sparse_linalg.LinearOperator((self.size, self.size), matvec=self.solve, dtype=np.float64)


def eigenpair_backward_errors(
    stiffness: Any, mass: Any, values: np.ndarray, vectors: np.ndarray, scale: float
) -> np.ndarray:
    """Return the relative residual ``||K phi - lambda M phi||_2 / (scale ||phi||_2)`` of each eigenpair.

    With ``scale = ||K - sigma M||`` this is the backward error that
    shift-invert Lanczos controls: a pair that ARPACK reports converged to
    relative accuracy ``tol`` has a residual of about ``tol`` or roundoff,
    independent of the conditioning of ``K - sigma M``.

    Args:
        stiffness: Stiffness matrix [N/m], shape ``[n, n]``.
        mass: Mass matrix [kg], shape ``[n, n]``.
        values: Eigenvalues [1/s^2], shape ``[k]``.
        vectors: Eigenvectors, shape ``[n, k]``.
        scale: Matrix norm [N/m] that normalizes the residual.

    Returns:
        Relative residuals [dimensionless], shape ``[k]``.
    """
    residual = matmul(stiffness, vectors) - matmul(mass, vectors) * values
    denominator = max(float(scale), np.finfo(np.float64).tiny) * np.linalg.norm(vectors, axis=0)
    return np.linalg.norm(residual, axis=0) / np.maximum(denominator, np.finfo(np.float64).tiny)


def arpack_start_vector(size: int, seed: int = 0) -> np.ndarray:
    """Return a deterministic, generic ARPACK starting vector of unit length.

    A constant vector would lie in the rigid-translation subspace of lattice
    models and span an invariant Krylov subspace, so a seeded pseudo-random
    vector is used instead; it keeps results reproducible across runs.

    Args:
        size: Vector length.
        seed: Offset of the fixed seed, for independent restarts.
    """
    vector = np.random.default_rng(_ARPACK_START_SEED + int(seed)).uniform(-1.0, 1.0, size)
    return vector / max(float(np.linalg.norm(vector)), np.finfo(np.float64).tiny)


def _deflation_projector(mass: Any, deflation: np.ndarray | None):
    """Return ``x -> (I - Y Y^T M) x`` for ``M``-orthonormal columns ``Y``, or ``None`` without deflation."""
    if deflation is None or deflation.shape[1] == 0:
        return None
    mass_deflation = matmul(mass, deflation)

    def project(vector: np.ndarray) -> np.ndarray:
        return vector - deflation @ (mass_deflation.T @ vector)

    return project


def _dense_deflated_eigh(stiffness: Any, mass: Any, deflation: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return all eigenpairs of the pencil restricted to the ``M``-orthogonal complement of ``deflation``."""
    dense_linalg = _import_scipy("linalg")

    stiffness = to_dense(stiffness)
    mass = to_dense(mass)
    if deflation is None or deflation.shape[1] == 0:
        return dense_linalg.eigh(stiffness, mass)
    # Orthonormal basis of the null space of Y^T M: the M-orthogonal complement of span(Y).
    complement = dense_linalg.null_space((mass @ deflation).T)
    values, vectors = dense_linalg.eigh(complement.T @ stiffness @ complement, complement.T @ mass @ complement)
    return values, complement @ vectors


def generalized_eigsh(
    stiffness: Any,
    mass: Any,
    k: int,
    *,
    sigma: float,
    tol: float = 0.0,
    maxiter: int | None = None,
    ncv: int | None = None,
    v0: np.ndarray | None = None,
    factorization: ShiftedFactorization | None = None,
    deflation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``k`` generalized eigenpairs of ``K phi = lambda M phi`` closest to ``sigma``.

    Uses ARPACK shift-invert Lanczos (:func:`scipy.sparse.linalg.eigsh` with
    ``which="LM"``) and an explicit sparse LU of ``K - sigma M`` passed as
    ``OPinv``. Known eigenvectors ``Y`` in ``deflation`` are locked: the
    operator is restricted to their ``M``-orthogonal complement by applying
    ``I - Y Y^T M`` after each solve, so exactly repeated eigenvalues already
    found (for example rigid-body modes) are not sought again. ARPACK requires
    ``k`` below the complement dimension minus one; larger requests fall back
    to a dense generalized solve.

    Args:
        stiffness: Symmetric stiffness matrix [N/m], shape ``[n, n]``.
        mass: Symmetric positive-definite mass matrix [kg], shape ``[n, n]``.
        k: Number of eigenpairs.
        sigma: Shift [1/s^2].
        tol: ARPACK relative accuracy; ``0`` requests machine precision.
        maxiter: Maximum ARPACK restart iterations; ``None`` uses SciPy's default.
        ncv: Number of Lanczos vectors; ``None`` uses SciPy's default.
        v0: Starting vector; ``None`` uses :func:`arpack_start_vector`.
        factorization: Reusable factorization of ``K - sigma M``.
        deflation: ``M``-orthonormal eigenvectors to exclude, shape ``[n, d]``.

    Returns:
        Eigenvalues [1/s^2] in ascending order, shape ``[k]``, and the
        ``M``-orthonormal eigenvectors, shape ``[n, k]``.

    Raises:
        ValueError: If ARPACK does not converge, the shifted matrix is
            singular, or a returned eigenpair has a relative residual (see
            :func:`eigenpair_backward_errors`) above
            ``10 * max(tol, 1000 * eps)``, scaled by the ratio of its
            distance ``|lambda - sigma|`` to the smallest one.
    """
    sparse_linalg = _import_scipy("sparse.linalg")

    size = int(stiffness.shape[0])
    deflated = 0 if deflation is None else int(deflation.shape[1])
    k = min(int(k), size - deflated)
    if k <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros((size, 0), dtype=np.float64)
    if k >= size - deflated - 1:
        values, vectors = _dense_deflated_eigh(stiffness, mass, deflation)
        order = np.argsort(np.abs(values - sigma), kind="stable")[:k]
        order = order[np.argsort(values[order], kind="stable")]
        return np.asarray(values[order], dtype=np.float64), np.asarray(vectors[:, order], dtype=np.float64)

    if factorization is None:
        factorization = ShiftedFactorization(stiffness, mass, sigma)
    project = _deflation_projector(mass, deflation)
    if project is None:
        inverse = factorization.operator()
    else:

        def solve_projected(rhs: np.ndarray) -> np.ndarray:
            return project(factorization.solve(np.asarray(rhs, dtype=np.float64).reshape(-1)))

        inverse = sparse_linalg.LinearOperator((size, size), matvec=solve_projected, dtype=np.float64)
    if v0 is None:
        v0 = arpack_start_vector(size)
    if project is not None:
        v0 = project(np.asarray(v0, dtype=np.float64))
    try:
        values, vectors = sparse_linalg.eigsh(
            to_csc(stiffness),
            k=k,
            M=to_csc(mass),
            sigma=factorization.sigma,
            which="LM",
            OPinv=inverse,
            tol=tol,
            maxiter=maxiter,
            ncv=ncv,
            v0=v0,
        )
    except sparse_linalg.ArpackNoConvergence as exc:
        raise ValueError(
            f"ARPACK did not converge for {k} eigenpairs near sigma = {factorization.sigma:.6g} "
            f"({len(exc.eigenvalues)} converged); increase arpack_maxiter or arpack_ncv, or choose an "
            "arpack_shift closer to the lowest wanted eigenvalues"
        ) from exc
    # ARPACK checks convergence of the Ritz values of the inverse operator only. Solves with a nearly
    # singular factor can return unconverged or spurious pairs without an error, so check the pencil.
    # Lanczos resolves a Ritz value theta_i = 1 / (lambda_i - sigma) only to roundoff of the largest
    # one, so the attainable residual grows with |lambda_i - sigma| / min_j |lambda_j - sigma|.
    errors = eigenpair_backward_errors(stiffness, mass, values, vectors, factorization.norm)
    distance = np.abs(values - factorization.sigma)
    spread = np.maximum(distance / max(float(np.min(distance)), np.finfo(np.float64).tiny), 1.0)
    accuracy = max(float(tol), _RESIDUAL_EPS_MULTIPLE * np.finfo(np.float64).eps)
    tolerance = _RESIDUAL_TOLERANCE_FACTOR * accuracy * spread
    worst = int(np.argmax(errors / tolerance))
    if not errors[worst] <= tolerance[worst]:
        failure = (
            f"ARPACK returned an inaccurate eigenpair near sigma = {factorization.sigma:.6g}: eigenvalue "
            f"{values[worst]:.6g} 1/s^2 has relative residual {errors[worst]:.3g} (||K phi - lambda M phi|| / "
            f"(||K - sigma M||_1 ||phi||)), above the tolerance {tolerance[worst]:.3g}"
        )
        # Lanczos needs the M inner product; an indefinite mass matrix also produces such pairs.
        _check_mass_definite(mass, failure)
        raise ValueError(
            f"{failure}. The shifted matrix is close to singular or ARPACK stopped early; choose a different "
            "arpack_shift or a smaller arpack_tol"
        )
    order = np.argsort(values, kind="stable")
    return np.asarray(values[order], dtype=np.float64), np.asarray(vectors[:, order], dtype=np.float64)


def count_eigenvalues_below(stiffness: Any, mass: Any, shift: float) -> int:
    """Return the number of generalized eigenvalues below ``shift`` (Sturm sequence count), or ``-1``.

    Uses the inertia of a symmetric factorization of ``K - shift M``. Returns
    ``-1`` when SuperLU had to pivot off the diagonal, so the inertia is
    unavailable.
    """
    try:
        factorization = ShiftedFactorization(stiffness, mass, shift)
    except ValueError:
        # The shift hit an eigenvalue exactly; any nearby shift in the same gap counts the same.
        factorization = ShiftedFactorization(stiffness, mass, shift * (1.0 + 1.0e-9) + 1.0e-300)
    return factorization.negative_pivot_count


_STURM_MIN_RELATIVE_GAP = 1.0e-6
"""Smallest relative eigenvalue gap in which a Sturm check shift is placed [dimensionless]."""

_STURM_GAP_SEARCHES = 4
"""Number of times the request grows past a repeated eigenvalue to find a gap for the Sturm check."""


def lowest_generalized_eigenpairs(
    stiffness: Any,
    mass: Any,
    count: int,
    *,
    sigma: float,
    zero_threshold: float,
    margin: int = 2,
    deflation: np.ndarray | None = None,
    tol: float = 0.0,
    maxiter: int | None = None,
    ncv: int | None = None,
    factorization: ShiftedFactorization | None = None,
    max_restarts: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the lowest generalized eigenpairs outside ``deflation``, verified by a Sturm sequence count.

    Computes ``count + margin`` eigenpairs with :func:`generalized_eigsh`
    and returns all of them in ascending order:

    - When every computed eigenvalue is at most ``zero_threshold``, the
      zero-energy subspace may be larger than the request, which is doubled
      until a positive-energy eigenvalue appears.
    - Lanczos can miss copies of exactly repeated eigenvalues. A shift
      ``tau`` is placed in the largest relative gap after the first
      ``count`` eigenvalues (computing more eigenpairs when a repeated
      eigenvalue fills the margin), and the inertia of ``K - tau M`` gives
      the true number of eigenvalues below it. Missing eigenpairs are
      sought again with all found ones locked, up to ``max_restarts``
      times. The check costs a second sparse factorization.

    Args:
        stiffness: Symmetric stiffness matrix [N/m], shape ``[n, n]``.
        mass: Symmetric positive-definite mass matrix [kg], shape ``[n, n]``.
        count: Number of lowest eigenpairs wanted.
        sigma: ARPACK shift [1/s^2], at or below the lowest wanted eigenvalue.
        zero_threshold: Largest eigenvalue treated as zero energy [1/s^2].
        margin: Extra eigenpairs computed beyond ``count``.
        deflation: ``M``-orthonormal eigenvectors excluded from the search, with
            eigenvalues below ``zero_threshold``, shape ``[n, d]``.
        tol: ARPACK relative accuracy; ``0`` requests machine precision.
        maxiter: Maximum ARPACK restart iterations.
        ncv: Number of Lanczos vectors.
        factorization: Reusable factorization of ``K - sigma M``.
        max_restarts: Number of searches for eigenpairs missed by the first one.

    Returns:
        Ascending eigenvalues [1/s^2] and ``M``-orthonormal eigenvectors,
        excluding ``deflation``.

    Raises:
        ValueError: If ARPACK does not converge, eigenvalues below the check
            shift remain missing, ARPACK returns more eigenvalues below it
            than the Sturm sequence count (spurious eigenpairs), or that
            excess comes from an indefinite mass matrix.

    Warns:
        UserWarning: If the Sturm sequence check is skipped, because a
            repeated eigenvalue fills every computed eigenpair after the
            wanted ones or the factorization pivoted off the diagonal.
    """
    values, vectors, _, _ = _lowest_generalized_eigenpairs(
        stiffness,
        mass,
        count,
        sigma=sigma,
        zero_threshold=zero_threshold,
        margin=margin,
        deflation=deflation,
        tol=tol,
        maxiter=maxiter,
        ncv=ncv,
        factorization=factorization,
        max_restarts=max_restarts,
    )
    return values, vectors


def _lowest_generalized_eigenpairs(
    stiffness: Any,
    mass: Any,
    count: int,
    *,
    sigma: float,
    zero_threshold: float,
    margin: int,
    deflation: np.ndarray | None,
    tol: float,
    maxiter: int | None,
    ncv: int | None,
    factorization: ShiftedFactorization | None,
    max_restarts: int,
) -> tuple[np.ndarray, np.ndarray, float | None, int | None]:
    """Implement :func:`lowest_generalized_eigenpairs` and also return its Sturm check.

    Returns:
        The eigenvalues and eigenvectors, the check shift ``tau`` [1/s^2], and
        the number of eigenvalues of the whole pencil below it (deflated
        directions included). Both are ``None`` when no check was made, either
        because the whole spectrum outside ``deflation`` was computed or because
        the check was skipped with a warning.
    """
    size = int(stiffness.shape[0])
    deflated = 0 if deflation is None else int(deflation.shape[1])
    available = size - deflated
    if factorization is None:
        factorization = ShiftedFactorization(stiffness, mass, sigma)

    def solve(request: int, locked: np.ndarray | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return generalized_eigsh(
            stiffness,
            mass,
            request,
            sigma=sigma,
            tol=tol,
            maxiter=maxiter,
            ncv=ncv,
            v0=arpack_start_vector(size, seed),
            factorization=factorization,
            deflation=locked,
        )

    count = int(count)
    margin = max(int(margin), 1)
    request = count + margin
    while True:
        request = min(request, available)
        values, vectors = solve(request, deflation, 0)
        if request >= available or np.any(values > zero_threshold):
            break
        request *= 2

    # Sturm check in the widest relative gap after the last wanted eigenvalue. A repeated eigenvalue
    # spanning the whole margin leaves no gap, so the request grows until one appears; every result,
    # including that of the last search, is tested.
    first = max(count - 1, 0)
    tau = None
    for search in range(_STURM_GAP_SEARCHES + 1):
        if values.shape[0] >= available or values.shape[0] <= count:
            # The whole spectrum outside deflation was computed, so no eigenvalue can be missing.
            return values, vectors, None, None
        lower = values[first:-1]
        upper = values[first + 1 :]
        relative_gap = (upper - lower) / np.maximum(np.abs(upper), np.finfo(np.float64).tiny)
        gap = int(np.argmax(relative_gap))
        if relative_gap[gap] >= _STURM_MIN_RELATIVE_GAP and upper[gap] > zero_threshold:
            tau = 0.5 * (lower[gap] + upper[gap])
            break
        if search == _STURM_GAP_SEARCHES:
            break
        request = min(values.shape[0] + 2 * margin, available)
        values, vectors = solve(request, deflation, 0)
    if tau is None:
        warnings.warn(
            f"The Sturm sequence check was skipped: no relative eigenvalue gap of at least "
            f"{_STURM_MIN_RELATIVE_GAP:g} follows eigenvalue {count} within the {values.shape[0]} computed "
            "eigenpairs, because a repeated eigenvalue fills them, so missed copies of it cannot be ruled out. "
            "Choose a mode count that ends at a gap in the spectrum",
            stacklevel=3,
        )
        return values, vectors, None, None
    below = count_eigenvalues_below(stiffness, mass, tau)
    if below < 0:
        warnings.warn(
            f"The Sturm sequence check was skipped: the factorization of stiffness_matrix - {tau:.6g} * "
            "mass_matrix pivoted off the diagonal, so its inertia does not count the eigenvalues below the shift",
            stacklevel=3,
        )
        return values, vectors, None, None
    expected = below - deflated
    for restart in range(1, max_restarts + 1):
        found = int(np.count_nonzero(values < tau))
        if found >= expected:
            break
        locked_vectors = vectors[:, values < tau]
        locked = locked_vectors if deflation is None else np.hstack((deflation, locked_vectors))
        extra_values, extra_vectors = solve(expected - found + int(margin), locked, restart)
        keep = values < tau
        values = np.concatenate((values[keep], extra_values))
        vectors = np.hstack((vectors[:, keep], extra_vectors))
        order = np.argsort(values, kind="stable")
        values = values[order]
        vectors = vectors[:, order]
    found = int(np.count_nonzero(values < tau))
    if found < expected:
        raise ValueError(
            f"ARPACK found {found} eigenvalues below {tau:.6g} 1/s^2, but the Sturm sequence count of "
            f"stiffness_matrix - {tau:.6g} * mass_matrix gives {expected}; repeated or clustered eigenvalues "
            "were missed. Increase arpack_ncv, or set arpack_shift at or below the lowest wanted eigenvalue"
        )
    if found > expected:
        # Sylvester's law counts eigenvalues only for a positive-definite mass matrix, which the inertia
        # at sigma <= 0 cannot certify in directions that the stiffness dominates.
        _check_mass_definite(
            mass,
            f"ARPACK found {found} eigenvalues below {tau:.6g} 1/s^2 where the Sturm sequence count, which "
            f"assumes a positive-definite mass matrix, gives {expected}",
        )
        raise ValueError(
            f"ARPACK found {found} eigenvalues below {tau:.6g} 1/s^2, more than the {expected} of the Sturm "
            f"sequence count of stiffness_matrix - {tau:.6g} * mass_matrix, so some returned eigenpairs are "
            "spurious. Check that stiffness_matrix is symmetric positive semi-definite, and choose an "
            "arpack_shift away from the eigenvalues"
        )
    return values, vectors, tau, below


# ----------------------------------------------------------------------------------------------------------------
# Pluggable eigensolver backends
# ----------------------------------------------------------------------------------------------------------------


class ModalEigensolver(enum.Enum):
    """Built-in eigensolvers of the finite-element modal generators.

    .. experimental::

    Each member's value is the string that the ``eigensolver`` argument of
    :class:`~newton.ModalGeneratorFEM` and
    :class:`~newton.ModalGeneratorCraigBamptonFEM` also accepts.
    """

    AUTO = "auto"
    """Let the generator choose ``DENSE`` or ``ARPACK`` from the input format and size."""

    DENSE = "dense"
    """Full spectrum with a Cholesky reduction and a dense symmetric eigensolve; see :class:`~newton.ModalDenseBackend`."""

    ARPACK = "arpack"
    """Sparse shift-invert Lanczos with a Sturm sequence check; see :class:`~newton.ModalScipyArpackBackend`."""

    SUBSPACE = "subspace"
    """Shift-invert block subspace iteration with Rayleigh-Ritz; see :class:`~newton.ModalSubspaceIterationBackend`."""


@dataclass
class ModalEigenResult:
    """Lowest generalized eigenpairs of ``K phi = lambda M phi`` returned by a :class:`~newton.ModalEigensolverBackend`.

    .. experimental::
    """

    values: np.ndarray
    """Eigenvalues [1/s^2] in ascending order, shape ``[k]``."""
    vectors: np.ndarray
    """``M``-orthonormal eigenvectors, shape ``[n, k]``; excludes the deflated directions."""
    inertia_count: int | None = None
    """Number of eigenvalues of the whole pencil, deflated directions included, below :attr:`~newton.ModalEigenResult.inertia_shift`.

    It is a Sturm sequence count (the inertia of a factorization of
    ``K - inertia_shift * M``), or ``None`` when the backend did not compute one.
    """
    backend: str = ""
    """Name of the backend that produced the result."""
    wall_s: float = 0.0
    """Wall-clock time of the solve [s]."""
    inertia_shift: float | None = None
    """Shift [1/s^2] of :attr:`inertia_count`, placed in a spectral gap above the wanted eigenvalues."""


@runtime_checkable
class ModalEigensolverBackend(Protocol):
    """Interface of a generalized symmetric eigensolver used by the finite-element modal generators.

    .. experimental::

    A backend computes the lowest eigenpairs of ``K phi = lambda M phi`` for a
    symmetric positive semi-definite stiffness ``K`` [N/m] and a symmetric
    positive-definite mass ``M`` [kg]. Any object with these attributes and
    method is accepted; subclassing is not required. The generators pass host
    matrices: SciPy CSC matrices, or dense :class:`numpy.ndarray` for
    :class:`~newton.ModalDenseBackend`. A device backend (``device`` other than ``"cpu"``)
    may convert them itself or accept
    :class:`warp.sparse.BsrMatrix` input from its own callers. Generators
    build and validate a SuperLU factorization of ``K - sigma M`` only for
    ``"cpu"`` backends.

    Contract of :meth:`~newton.ModalEigensolverBackend.solve`:

    - Return at least ``count`` eigenpairs outside ``deflation`` (fewer only
      when fewer exist) in ascending order, ``M``-orthonormal and
      ``M``-orthogonal to ``deflation``. Extra pairs above them are allowed.
    - When every computed eigenvalue is at most ``zero_threshold``, compute
      more, so that zero-energy mechanisms cannot hide the elastic spectrum.
    - Either set :attr:`~newton.ModalEigensolverBackend.supports_inertia_count` and return
      :attr:`~newton.ModalEigenResult.inertia_count` with :attr:`~newton.ModalEigenResult.inertia_shift`
      whenever only part of the spectrum is returned, or set
      :attr:`~newton.ModalEigensolverBackend.supports_inertia_count` to ``False``; callers then skip the Sturm
      sequence check with a warning. A backend that supports the count may
      return ``None`` after warning itself why the check was skipped; callers
      then also warn that the check was skipped.
    - Raise :class:`ValueError` when the solve fails to converge.
    """

    name: str
    """Short backend name, recorded in :attr:`~newton.ModalEigenResult.backend` and generator reports."""
    supports_inertia_count: bool
    """Whether :meth:`~newton.ModalEigensolverBackend.solve` returns a Sturm sequence count for partial spectra."""
    device: str
    """Device that runs the solve, ``"cpu"`` for host backends."""

    def solve(
        self,
        stiffness: Any,
        mass: Any,
        count: int,
        *,
        deflation: np.ndarray | None = None,
        sigma: float | None = None,
        tol: float = 0.0,
        factorization: ShiftedFactorization | None = None,
        zero_threshold: float = 0.0,
    ) -> ModalEigenResult:
        """Return the ``count`` lowest generalized eigenpairs outside ``deflation``.

        Args:
            stiffness: Symmetric stiffness matrix [N/m], shape ``[n, n]``.
            mass: Symmetric positive-definite mass matrix [kg], shape ``[n, n]``.
            count: Number of lowest eigenpairs wanted.
            deflation: ``M``-orthonormal zero-energy eigenvectors to exclude,
                such as rigid-body motions, shape ``[n, d]``.
            sigma: Shift [1/s^2] at or below the lowest wanted eigenvalue;
                ``None`` uses ``modal_sparse.default_shift``.
            tol: Relative accuracy; ``0`` requests machine precision.
            factorization: Optional reusable factorization of
                ``K - sigma M``; backends may ignore it.
            zero_threshold: Largest eigenvalue treated as zero energy [1/s^2].

        Returns:
            The computed eigenpairs.
        """
        ...


def is_bsr_matrix(matrix: Any) -> bool:
    """Return whether ``matrix`` looks like a :class:`warp.sparse.BsrMatrix` (duck-typed, never imports Warp)."""
    return all(hasattr(matrix, attribute) for attribute in ("nrow", "ncol", "block_shape", "offsets", "columns"))


def to_backend_matrix(matrix: Any, *, device: str = "cpu") -> Any:
    """Return ``matrix`` in a form that a backend on ``device`` accepts.

    For ``"cpu"``, a :class:`warp.sparse.BsrMatrix` is copied to a SciPy CSC
    matrix, and dense arrays and SciPy matrices pass unchanged. For any other
    device the matrix passes unchanged; a device backend converts host input
    itself.

    Args:
        matrix: Dense array, SciPy sparse matrix, or Warp BSR matrix.
        device: Device of the consuming backend.
    """
    if device != "cpu" or not is_bsr_matrix(matrix):
        return matrix
    sparse = _import_scipy("sparse")
    offsets = np.asarray(matrix.offsets.numpy(), dtype=np.int64)[: int(matrix.nrow) + 1]
    nonzeros = int(offsets[-1])
    columns = np.asarray(matrix.columns.numpy(), dtype=np.int64)[:nonzeros]
    rows_per_block, cols_per_block = (int(size) for size in matrix.block_shape)
    values = np.asarray(matrix.values.numpy(), dtype=np.float64)[:nonzeros]
    values = values.reshape((nonzeros, rows_per_block, cols_per_block))
    shape = (int(matrix.nrow) * rows_per_block, int(matrix.ncol) * cols_per_block)
    return to_csc(sparse.bsr_matrix((values, columns, offsets), shape=shape))


def cholesky_eigh(stiffness: np.ndarray, mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return all generalized eigenpairs of dense ``K``, ``M`` by a Cholesky reduction, without SciPy.

    Args:
        stiffness: Dense symmetric stiffness matrix [N/m], shape ``[n, n]``.
        mass: Dense symmetric positive-definite mass matrix [kg], shape ``[n, n]``.

    Returns:
        Ascending eigenvalues [1/s^2] and ``M``-orthonormal eigenvectors.

    Raises:
        ValueError: If ``mass`` is not positive definite.
    """
    try:
        chol = np.linalg.cholesky(mass)
    except np.linalg.LinAlgError as exc:
        raise ValueError("mass_matrix must be symmetric positive definite on free DOFs") from exc

    inv_chol_stiffness = np.linalg.solve(chol, stiffness)
    standard = np.linalg.solve(chol, inv_chol_stiffness.T).T
    standard = 0.5 * (standard + standard.T)
    eigenvalues, eigenvectors = np.linalg.eigh(standard)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return eigenvalues, np.linalg.solve(chol.T, eigenvectors)


def _mass_orthonormalize(vectors: np.ndarray, mass: Any, relative_cutoff: float = 1.0e-12) -> np.ndarray:
    """Return an ``M``-orthonormal basis of the span of ``vectors``, dropping numerically dependent directions."""
    # Column scaling and a second pass (as in CholeskyQR2) keep the Gram-matrix method accurate when the
    # block columns differ in magnitude by many decades, as after a shift-invert solve.
    for _ in range(2):
        if vectors.shape[1] == 0:
            return vectors
        norms = np.sqrt(np.maximum(quadratic_form_diag(vectors, mass), 0.0))
        vectors = vectors[:, norms > 0.0] / norms[norms > 0.0]
        gram = vectors.T @ matmul(mass, vectors)
        values, rotation = np.linalg.eigh(0.5 * (gram + gram.T))
        keep = values > relative_cutoff * max(float(np.max(values, initial=0.0)), np.finfo(np.float64).tiny)
        vectors = vectors @ (rotation[:, keep] / np.sqrt(values[keep]))
    return vectors


class ModalDenseBackend:
    """Dense full-spectrum eigensolver: Cholesky reduction and :func:`numpy.linalg.eigh`.

    .. experimental::

    Densifies sparse input and returns every eigenpair outside ``deflation``,
    ignoring ``count``, so no Sturm sequence check is needed. Memory grows
    as ``n^2`` and time as ``n^3``; use it for small models. It needs no
    SciPy.
    """

    name: str = "dense"
    supports_inertia_count: bool = False
    device: str = "cpu"

    def solve(
        self,
        stiffness: Any,
        mass: Any,
        count: int,
        *,
        deflation: np.ndarray | None = None,
        sigma: float | None = None,
        tol: float = 0.0,
        factorization: ShiftedFactorization | None = None,
        zero_threshold: float = 0.0,
    ) -> ModalEigenResult:
        """Return all eigenpairs outside ``deflation``; see :meth:`~newton.ModalEigensolverBackend.solve`."""
        del count, sigma, tol, factorization, zero_threshold
        start = time.perf_counter()
        stiffness = to_dense(stiffness)
        mass = to_dense(mass)
        if deflation is None or deflation.shape[1] == 0:
            values, vectors = cholesky_eigh(stiffness, mass)
        else:
            # Orthonormal basis of the null space of Y^T M: the M-orthogonal complement of span(Y).
            constraint = (mass @ deflation).T
            _, singular, right = np.linalg.svd(constraint, full_matrices=True)
            rank = int(np.count_nonzero(singular > singular[0] * max(constraint.shape) * np.finfo(np.float64).eps))
            complement = right[rank:].T
            values, reduced = cholesky_eigh(complement.T @ stiffness @ complement, complement.T @ mass @ complement)
            vectors = complement @ reduced
        return ModalEigenResult(values, vectors, backend=self.name, wall_s=time.perf_counter() - start)


class ModalScipyArpackBackend:
    """SciPy ARPACK shift-invert Lanczos backend with a Sturm sequence check.

    .. experimental::

    Wraps ``modal_sparse.lowest_generalized_eigenpairs``: computes ``count + margin``
    eigenpairs on a SuperLU factorization of ``K - sigma M``, deflates the
    given zero-energy vectors, and verifies with the inertia of a second
    factorization that no eigenvalue below a shift in the spectral gap after
    the wanted ones was missed. Needs SciPy.

    Args:
        margin: Extra eigenpairs computed beyond the wanted ones.
        maxiter: Maximum ARPACK restart iterations; ``None`` uses SciPy's default.
        ncv: Number of Lanczos vectors; ``None`` uses SciPy's default.
        max_restarts: Searches for eigenpairs missed by the first one.
    """

    name: str = "arpack"
    supports_inertia_count: bool = True
    device: str = "cpu"

    def __init__(self, *, margin: int = 2, maxiter: int | None = None, ncv: int | None = None, max_restarts: int = 2):
        self.margin = int(margin)
        self.maxiter = None if maxiter is None else int(maxiter)
        self.ncv = None if ncv is None else int(ncv)
        self.max_restarts = int(max_restarts)

    def solve(
        self,
        stiffness: Any,
        mass: Any,
        count: int,
        *,
        deflation: np.ndarray | None = None,
        sigma: float | None = None,
        tol: float = 0.0,
        factorization: ShiftedFactorization | None = None,
        zero_threshold: float = 0.0,
    ) -> ModalEigenResult:
        """Return the lowest eigenpairs outside ``deflation``; see :meth:`~newton.ModalEigensolverBackend.solve`."""
        start = time.perf_counter()
        if sigma is None:
            sigma = factorization.sigma if factorization is not None else default_shift(stiffness, mass)
        if factorization is not None and factorization.sigma != float(sigma):
            factorization = None
        values, vectors, tau, below = _lowest_generalized_eigenpairs(
            stiffness,
            mass,
            count,
            sigma=float(sigma),
            zero_threshold=float(zero_threshold),
            margin=self.margin,
            deflation=deflation,
            tol=float(tol),
            maxiter=self.maxiter,
            ncv=self.ncv,
            factorization=factorization,
            max_restarts=self.max_restarts,
        )
        return ModalEigenResult(
            values,
            vectors,
            inertia_count=below,
            backend=self.name,
            wall_s=time.perf_counter() - start,
            inertia_shift=tau,
        )


class ModalSubspaceIterationBackend:
    """Shift-invert block subspace iteration with Rayleigh-Ritz, warm-started from previous eigenvectors.

    .. experimental::

    Each iteration applies ``(I - Y Y^T M) (K - sigma M)^-1 M`` to a block of
    ``count + margin`` vectors, with ``Y`` the deflated vectors, and
    ``M``-orthonormalizes it; a Rayleigh-Ritz projection then gives the Ritz
    pairs. The block starts from ``previous_vectors`` (for example the modes
    of a nearby configuration, which makes one or two iterations enough)
    padded with deterministic random vectors. Previous vectors of a
    different length are ignored. The factorization passed to
    :meth:`~newton.ModalEigensolverBackend.solve` is reused when its shift matches. Needs SciPy.

    At least ``iterations`` iterations run. With ``require_convergence``,
    iterations continue until every wanted pair has the relative residual
    ``||K phi - lambda M phi|| / (||K - sigma M||_1 ||phi||)`` that
    ``modal_sparse.generalized_eigsh`` accepts, and :class:`ValueError` is raised after
    ``max_iterations``; otherwise exactly ``iterations`` iterations run and
    :attr:`~newton.ModalSubspaceIterationBackend.last_residuals` records the accuracy reached.

    The backend is stateful. With ``warm_start``, each solve replaces
    ``previous_vectors`` with a copy of the returned eigenvectors, so a
    backend reused on a slowly changing model warm-starts itself; results
    then depend on the order of the solves. Do not share a warm-started
    backend between generators or between unrelated problems of equal
    size. Without ``warm_start``, ``previous_vectors`` is never changed by a
    solve, so repeated solves are independent.

    Args:
        iterations: Minimum number of block iterations.
        margin: Extra block vectors beyond the wanted eigenpairs; a larger
            margin speeds up convergence of the highest wanted pair.
        previous_vectors: Start vectors of the next solve, shape ``[n, p]``;
            copied.
        warm_start: Whether each solve stores a copy of its eigenvectors as
            ``previous_vectors`` for the next solve.
        max_iterations: Iteration limit with ``require_convergence``.
        require_convergence: Whether to iterate to convergence and raise
            when it is not reached.
        sturm_check: Whether to return a Sturm sequence count (one extra
            factorization) at a shift in the gap after the wanted eigenvalues.
        seed: Seed of the random padding vectors.
    """

    name: str = "subspace"
    device: str = "cpu"

    def __init__(
        self,
        *,
        iterations: int = 1,
        margin: int = 2,
        previous_vectors: np.ndarray | None = None,
        warm_start: bool = False,
        max_iterations: int = 500,
        require_convergence: bool = True,
        sturm_check: bool = False,
        seed: int = 0,
    ):
        self.iterations = int(iterations)
        if self.iterations < 1:
            raise ValueError(f"iterations must be at least 1, got {self.iterations}")
        self.margin = int(margin)
        if self.margin < 0:
            raise ValueError(f"margin must be non-negative, got {self.margin}")
        self.previous_vectors = None if previous_vectors is None else np.array(previous_vectors, dtype=np.float64)
        self.warm_start = bool(warm_start)
        self.max_iterations = max(int(max_iterations), self.iterations)
        self.require_convergence = bool(require_convergence)
        self.sturm_check = bool(sturm_check)
        self.seed = int(seed)
        self.last_iterations = 0
        """Number of block iterations of the last solve."""
        self.last_residuals = np.zeros(0, dtype=np.float64)
        """Relative residuals of the eigenpairs of the last solve [dimensionless]."""

    @property
    def supports_inertia_count(self) -> bool:
        """Whether :meth:`~newton.ModalEigensolverBackend.solve` returns a Sturm sequence count, as set by ``sturm_check``."""
        return self.sturm_check

    def solve(
        self,
        stiffness: Any,
        mass: Any,
        count: int,
        *,
        deflation: np.ndarray | None = None,
        sigma: float | None = None,
        tol: float = 0.0,
        factorization: ShiftedFactorization | None = None,
        zero_threshold: float = 0.0,
    ) -> ModalEigenResult:
        """Return the lowest eigenpairs outside ``deflation``; see :meth:`~newton.ModalEigensolverBackend.solve`."""
        start = time.perf_counter()
        size = int(stiffness.shape[0])
        deflated = 0 if deflation is None else int(deflation.shape[1])
        available = size - deflated
        count = min(int(count), available)
        if count <= 0:
            return ModalEigenResult(
                np.zeros(0), np.zeros((size, 0)), backend=self.name, wall_s=time.perf_counter() - start
            )
        if sigma is None:
            sigma = factorization.sigma if factorization is not None else default_shift(stiffness, mass)
        if factorization is None or factorization.sigma != float(sigma):
            factorization = ShiftedFactorization(stiffness, mass, float(sigma))
        project = _deflation_projector(mass, deflation)

        block = min(count + self.margin, available)
        previous = self.previous_vectors
        if previous is None or previous.ndim != 2 or previous.shape[0] != size:
            previous = np.zeros((size, 0), dtype=np.float64)
        rng = np.random.default_rng(_ARPACK_START_SEED + self.seed)
        pad = max(block - previous.shape[1], 0)
        vectors = np.hstack((previous, rng.uniform(-1.0, 1.0, (size, pad))))
        if project is not None:
            vectors = project(vectors)
        vectors = _mass_orthonormalize(vectors, mass)

        accuracy = max(float(tol), _RESIDUAL_EPS_MULTIPLE * np.finfo(np.float64).eps)
        iteration = 0
        while True:
            vectors = factorization.solve(matmul(mass, vectors))
            if project is not None:
                vectors = project(vectors)
            vectors = _mass_orthonormalize(vectors, mass)
            iteration += 1
            if iteration < self.iterations:
                continue
            reduced_stiffness = vectors.T @ matmul(stiffness, vectors)
            reduced_mass = vectors.T @ matmul(mass, vectors)
            ritz_values, ritz_vectors = cholesky_eigh(
                0.5 * (reduced_stiffness + reduced_stiffness.T), 0.5 * (reduced_mass + reduced_mass.T)
            )
            ritz = vectors @ ritz_vectors
            wanted = min(count, ritz.shape[1])
            errors = eigenpair_backward_errors(
                stiffness, mass, ritz_values[:wanted], ritz[:, :wanted], factorization.norm
            )
            distance = np.abs(ritz_values[:wanted] - factorization.sigma)
            spread = np.maximum(distance / max(float(np.min(distance)), np.finfo(np.float64).tiny), 1.0)
            converged = bool(np.all(errors <= _RESIDUAL_TOLERANCE_FACTOR * accuracy * spread))
            if converged or not self.require_convergence or iteration >= self.max_iterations:
                break
            vectors = ritz
        self.last_iterations = iteration
        self.last_residuals = errors
        if self.require_convergence and not converged:
            raise ValueError(
                f"Subspace iteration did not converge in {iteration} iterations: the largest relative residual "
                f"of the {wanted} wanted eigenpairs is {float(np.max(errors)):.3g}. Increase max_iterations or "
                "margin, or pass previous_vectors closer to the wanted modes"
            )
        if wanted < count:
            raise ValueError(
                f"Subspace iteration kept only {wanted} independent vectors for {count} wanted eigenpairs; the "
                "block lost rank. Increase margin"
            )

        inertia_count = None
        inertia_shift = None
        if self.sturm_check and ritz_values.shape[0] > count and count < available:
            lower, upper = float(ritz_values[count - 1]), float(ritz_values[count])
            if (upper - lower) >= _STURM_MIN_RELATIVE_GAP * max(abs(upper), np.finfo(np.float64).tiny):
                inertia_shift = 0.5 * (lower + upper)
                below = count_eigenvalues_below(stiffness, mass, inertia_shift)
                inertia_count = None if below < 0 else below
            if inertia_count is None:
                warnings.warn(
                    "The Sturm sequence check of the subspace iteration was skipped: no relative gap of at least "
                    f"{_STURM_MIN_RELATIVE_GAP:g} follows eigenvalue {count}, or the factorization pivoted off "
                    "the diagonal",
                    stacklevel=2,
                )
                inertia_shift = None
        values = np.asarray(ritz_values[:count], dtype=np.float64)
        result_vectors = np.ascontiguousarray(ritz[:, :count])
        if self.warm_start:
            self.previous_vectors = result_vectors.copy()
        return ModalEigenResult(
            values,
            result_vectors,
            inertia_count=inertia_count,
            backend=self.name,
            wall_s=time.perf_counter() - start,
            inertia_shift=inertia_shift,
        )


class WarpackBackend:
    """Placeholder for a GPU eigensolver backend (Warpack with cuDSS, or FSAI-CG with LOBPCG).

    .. experimental::

    This build contains no GPU eigensolver: :meth:`~newton.ModalEigensolverBackend.solve` raises
    :class:`NotImplementedError`. The class documents the contract that such
    a backend fulfills, so it can be subclassed, or replaced by any object
    that satisfies :class:`~newton.ModalEigensolverBackend`, and passed as the
    ``eigensolver`` of a finite-element modal generator:

    - Inputs are host SciPy CSC matrices from the generators;
      :func:`to_backend_matrix` leaves them unchanged for a device other
      than ``"cpu"``, and direct callers may pass
      :class:`warp.sparse.BsrMatrix`. ``deflation`` holds the rigid-body
      vectors of a free-free model (``M``-orthonormal, host
      :class:`numpy.ndarray`), which must be projected out. ``sigma`` is a
      shift at or below the lowest wanted eigenvalue; the host
      ``factorization`` is ``None`` for a non-``"cpu"`` device.
    - The output is a :class:`~newton.ModalEigenResult` in float64 host arrays with
      ascending eigenvalues and ``M``-orthonormal eigenvectors.
    - Either return :attr:`~newton.ModalEigenResult.inertia_count` (for example from the
      inertia of a cuDSS ``L D L^T`` factorization of ``K - tau M``) or
      leave :attr:`~newton.ModalEigensolverBackend.supports_inertia_count` ``False``; callers then skip the
      Sturm sequence check with a warning, so missed repeated eigenvalues
      are not detected.

    Args:
        device: Warp device that would run the solve.
    """

    name: str = "warpack"
    supports_inertia_count: bool = False

    def __init__(self, *, device: str = "cuda:0"):
        self.device = str(device)

    def solve(
        self,
        stiffness: Any,
        mass: Any,
        count: int,
        *,
        deflation: np.ndarray | None = None,
        sigma: float | None = None,
        tol: float = 0.0,
        factorization: ShiftedFactorization | None = None,
        zero_threshold: float = 0.0,
    ) -> ModalEigenResult:
        """Raise :class:`NotImplementedError`; a subclass implements the GPU solve."""
        raise NotImplementedError("Warpack backend is not available in this build")


_EIGENSOLVER_CHOICES = "'auto', 'dense', 'arpack', or 'subspace', a ModalEigensolver, or a ModalEigensolverBackend"


def eigensolver_kind(
    spec: str | ModalEigensolver | ModalEigensolverBackend,
) -> ModalEigensolver | ModalEigensolverBackend:
    """Return the :class:`ModalEigensolver` named by ``spec``, or ``spec`` itself if it is a backend.

    Raises:
        ValueError: If ``spec`` is a string that names no eigensolver.
        TypeError: If ``spec`` is neither a name, a :class:`ModalEigensolver`,
            nor a :class:`~newton.ModalEigensolverBackend`.
    """
    if isinstance(spec, ModalEigensolver):
        return spec
    if isinstance(spec, str):
        try:
            return ModalEigensolver(spec)
        except ValueError:
            raise ValueError(f"eigensolver must be {_EIGENSOLVER_CHOICES}, got '{spec}'") from None
    if isinstance(spec, ModalEigensolverBackend):
        return spec
    raise TypeError(f"eigensolver must be {_EIGENSOLVER_CHOICES}, got {type(spec).__name__}")


def resolve_eigensolver(
    spec: str | ModalEigensolver | ModalEigensolverBackend,
    *,
    margin: int = 2,
    maxiter: int | None = None,
    ncv: int | None = None,
) -> ModalEigensolverBackend:
    """Return the eigensolver backend selected by ``spec``.

    .. experimental::

    A backend instance is returned unchanged. ``"subspace"`` resolves to a
    :class:`~newton.ModalSubspaceIterationBackend` without ``warm_start``,
    so every solve is cold-started; it iterates to convergence and returns
    a Sturm sequence count. ``"auto"`` resolves to
    :class:`~newton.ModalScipyArpackBackend` when SciPy is installed and to
    :class:`~newton.ModalDenseBackend` otherwise; the generators make a finer choice from
    the input format and size before they resolve.

    Args:
        spec: Eigensolver name, :class:`ModalEigensolver`, or backend.
        margin: Extra eigenpairs of the ARPACK and subspace backends.
        maxiter: Maximum ARPACK restart iterations.
        ncv: Number of ARPACK Lanczos vectors.

    Returns:
        A backend that satisfies :class:`~newton.ModalEigensolverBackend`.

    Raises:
        ValueError: If ``spec`` is a string that names no eigensolver.
        TypeError: If ``spec`` has an unsupported type.
    """
    kind = eigensolver_kind(spec)
    if not isinstance(kind, ModalEigensolver):
        return kind
    if kind is ModalEigensolver.AUTO:
        kind = ModalEigensolver.ARPACK if has_scipy() else ModalEigensolver.DENSE
    if kind is ModalEigensolver.DENSE:
        return ModalDenseBackend()
    if kind is ModalEigensolver.ARPACK:
        return ModalScipyArpackBackend(margin=margin, maxiter=maxiter, ncv=ncv)
    # A one-shot build cannot rely on a warm start, so the named choice verifies its result and keeps no
    # state between builds.
    return ModalSubspaceIterationBackend(margin=margin, sturm_check=True, warm_start=False)


def validate_eigen_result(
    result: ModalEigenResult, *, size: int, count: int, deflated: int, supports_inertia_count: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Check a :class:`~newton.ModalEigenResult` against the backend contract and return its sorted eigenpairs.

    Verifies shapes and finiteness, that at least ``min(count, size - deflated)``
    eigenpairs were returned, and the Sturm sequence count when the result
    carries one. A partial spectrum without a count warns that the check was
    skipped, whether or not the backend supports counting.

    Args:
        result: Backend output.
        size: Matrix dimension ``n``.
        count: Number of eigenpairs requested.
        deflated: Number of deflated directions.
        supports_inertia_count: The backend's :attr:`~newton.ModalEigensolverBackend.supports_inertia_count`.

    Returns:
        Ascending eigenvalues [1/s^2] and the matching eigenvectors.

    Raises:
        ValueError: If the result violates the contract or the Sturm count
            disagrees with the returned eigenvalues.
    """
    values = np.asarray(result.values, dtype=np.float64).reshape(-1)
    vectors = np.asarray(result.vectors, dtype=np.float64)
    name = result.backend or "eigensolver"
    if vectors.ndim != 2 or vectors.shape != (size, values.shape[0]):
        raise ValueError(
            f"The {name} backend returned eigenvectors of shape {vectors.shape} for {values.shape[0]} eigenvalues "
            f"of a size-{size} problem"
        )
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(vectors))):
        raise ValueError(f"The {name} backend returned non-finite eigenpairs")
    available = size - int(deflated)
    wanted = min(int(count), available)
    if values.shape[0] < wanted:
        raise ValueError(f"The {name} backend returned {values.shape[0]} eigenpairs, fewer than the {wanted} requested")
    order = np.argsort(values, kind="stable")
    values = values[order]
    vectors = vectors[:, order]
    if values.shape[0] >= available:
        return values, vectors
    if result.inertia_count is not None:
        if result.inertia_shift is None:
            raise ValueError(f"The {name} backend returned an inertia_count without its inertia_shift")
        expected = int(result.inertia_count) - int(deflated)
        found = int(np.count_nonzero(values < result.inertia_shift))
        if found != expected:
            raise ValueError(
                f"The {name} backend returned {found} eigenvalues below {result.inertia_shift:.6g} 1/s^2, but the "
                f"Sturm sequence count gives {expected}; eigenpairs were missed or are spurious"
            )
    elif not supports_inertia_count:
        warnings.warn(
            f"The Sturm sequence check was skipped: the {name} eigensolver backend does not report an inertia "
            "count, so missed copies of repeated eigenvalues cannot be ruled out",
            stacklevel=3,
        )
    else:
        warnings.warn(
            f"The Sturm sequence check was skipped: the {name} eigensolver backend supports an inertia count but "
            "returned none for a partial spectrum, so missed copies of repeated eigenvalues cannot be ruled out",
            stacklevel=3,
        )
    return values, vectors
