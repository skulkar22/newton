# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dense/sparse matrix helpers for finite-element modal generators.

The helpers accept either dense :class:`numpy.ndarray` matrices or SciPy sparse
matrices and arrays. SciPy is an optional dependency, so it is imported only
when a sparse code path runs. Dense inputs follow the exact operation order of
the original dense generator code, so dense results are unchanged.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np

SHIFT_FRACTION_DEFAULT = 1.0e-6
"""Default magnitude of the ARPACK shift relative to ``||K||_1 / ||M||_1`` [dimensionless]."""

_ARPACK_START_SEED = 0x5EED
"""Seed of the deterministic ARPACK starting vector."""


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


def coerce_symmetric_matrix(matrix: Any, size: int, name: str, *, keep_sparse: bool = True) -> Any:
    """Validate a square matrix and return its symmetric part in float64.

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
        ValueError: If the shape is not ``(size, size)`` or sparse entries are
            not finite.
    """
    if is_sparse(matrix):
        if tuple(matrix.shape) != (size, size):
            raise ValueError(f"{name} must have shape ({size}, {size}), got {tuple(matrix.shape)}")
        if not keep_sparse:
            array = np.ascontiguousarray(matrix.toarray(), dtype=np.float64)
            return 0.5 * (array + array.T)
        csc = to_csc(matrix)
        if not np.all(np.isfinite(csc.data)):
            raise ValueError(f"{name} contains non-finite entries")
        return to_csc(0.5 * (csc + csc.T))
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {array.shape}")
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
    matrix it is an estimate of the same order of magnitude, which suffices
    as the scale of a relative zero-energy threshold.
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


class ShiftedFactorization:
    """Sparse LU factorization of ``K - sigma M`` for shift-invert eigensolves.

    The matrix is factored with SuperLU in symmetric mode (a COLAMD fill-reducing
    ordering applied to rows and columns alike, with diagonal pivots), so for a
    symmetric matrix the diagonal of
    ``U`` is the ``D`` of an ``L D L^T`` factorization and its signs give the
    matrix inertia (Sylvester's law): ``negative_pivot_count`` is the number of
    generalized eigenvalues below ``sigma`` when ``M`` is positive definite.

    Args:
        stiffness: Sparse stiffness matrix [N/m], shape ``[n, n]``.
        mass: Sparse mass matrix [kg], shape ``[n, n]``.
        sigma: Shift [1/s^2].

    Raises:
        ValueError: If the shifted matrix is exactly singular.
    """

    def __init__(self, stiffness: Any, mass: Any, sigma: float):
        sparse_linalg = _import_scipy("sparse.linalg")

        self.sigma = float(sigma)
        shifted = to_csc(to_csc(stiffness) - self.sigma * to_csc(mass))
        self.size = int(shifted.shape[0])
        try:
            self.lu = sparse_linalg.splu(
                shifted,
                permc_spec="COLAMD",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
        except RuntimeError as exc:
            raise ValueError(
                f"stiffness_matrix - sigma * mass_matrix is singular for sigma = {self.sigma:.6g}; "
                "choose a different arpack_shift"
            ) from exc
        self.symmetric_pivoting = bool(np.array_equal(self.lu.perm_r, self.lu.perm_c))
        self.negative_pivot_count = -1
        if self.symmetric_pivoting:
            # SuperLU exposes U only as a copy; keep just its diagonal.
            pivots = np.asarray(self.lu.U.diagonal(), dtype=np.float64)
            self.negative_pivot_count = int(np.count_nonzero(pivots <= 0.0))

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
        ValueError: If ARPACK does not converge or the shifted matrix is singular.
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
        ValueError: If ARPACK does not converge, or eigenvalues below the
            check shift remain missing.
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
    # spanning the whole margin leaves no gap, so the request grows until one appears.
    first = max(count - 1, 0)
    tau = None
    for _ in range(_STURM_GAP_SEARCHES):
        if values.shape[0] >= available or values.shape[0] <= count:
            return values, vectors
        lower = values[first:-1]
        upper = values[first + 1 :]
        relative_gap = (upper - lower) / np.maximum(np.abs(upper), np.finfo(np.float64).tiny)
        gap = int(np.argmax(relative_gap))
        if relative_gap[gap] >= _STURM_MIN_RELATIVE_GAP and upper[gap] > zero_threshold:
            tau = 0.5 * (lower[gap] + upper[gap])
            break
        request = min(values.shape[0] + 2 * margin, available)
        values, vectors = solve(request, deflation, 0)
    if tau is None:
        return values, vectors
    below = count_eigenvalues_below(stiffness, mass, tau)
    if below < 0:
        return values, vectors
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
    if found != expected:
        raise ValueError(
            f"ARPACK found {found} eigenvalues below {tau:.6g} 1/s^2, but the Sturm sequence count of "
            f"stiffness_matrix - {tau:.6g} * mass_matrix gives {expected}; repeated or clustered eigenvalues "
            "were missed. Increase arpack_ncv, or set arpack_shift at or below the lowest wanted eigenvalue"
        )
    return values, vectors
