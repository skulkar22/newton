# ruff: noqa: TID253
"""Eigensolver seam of the re-ROM controller, on top of Newton's ``newton.ModalEigensolverBackend``.

The controller's ``eigensolver`` option is a name, a :class:`newton.ModalEigensolver` or any object that
satisfies :class:`newton.ModalEigensolverBackend` (for example :class:`newton.ModalSubspaceIterationBackend`, a
Warpack/cuDSS backend, or a test double). :func:`resolve_backend` maps it to a backend:

- ``"arpack"`` / ``"auto"`` / ``ModalEigensolver.ARPACK`` / ``AUTO``: :class:`RestrictedArpackBackend`, ARPACK
  shift-invert with the *exact* inverse restricted to the M-orthogonal complement of the deflated rigid
  motions and a Haynsworth inertia count (the M1 re-ROM path);
- ``"subspace"`` / ``ModalEigensolver.SUBSPACE``: :class:`RestrictedSubspaceBackend`, warm-started block
  inverse iteration with the shared restricted factorization (the WP3a path);
- ``"dense"`` / ``ModalEigensolver.DENSE``: ``modal_sparse.resolve_eigensolver("dense")``, i.e. :class:`newton.ModalDenseBackend`;
- a backend instance: returned unchanged (validated with ``isinstance(x, newton.ModalEigensolverBackend)``).

The two restricted backends exist because the tangent stiffness of a pre-stressed body does not have the
rigid motions in its null space: projecting them out after a plain shift-invert solve (what Newton's
generic backends do) is not the inverse of the restricted pencil. Generic backends still work: they see
``(K_T, M)`` on the free DOFs with the M-orthonormal rigid motions as ``deflation``; a backend exposing a
``previous_vectors`` attribute (Newton's :class:`~newton.ModalSubspaceIterationBackend`) is warm-started with
the previous basis before every solve.

:class:`ShiftedRestrictedSolver` is the one factorization per update shared by the eigensolve (ARPACK
shift-invert operator and Sturm check, or the subspace iteration), the static Ritz vectors and the modal
derivatives (:func:`shared_restricted_solver`, ``BasisOptions.shared_factorization``). The legacy path
(``shared_factorization=False``) factors ARPACK's shift and the Ritz shift separately and reproduces the
recorded runs bit for bit.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import newton
from newton._src.sim import modal_sparse

__all__ = [
    "RestrictedArpackBackend",
    "RestrictedSubspaceBackend",
    "ShiftedRestrictedSolver",
    "resolve_backend",
    "shared_restricted_solver",
    "solve_restricted_eigen",
]


class ShiftedRestrictedSolver:
    """Inertia-relief / clamped static solve with ONE sparse factorization of ``K - sigma M``.

    Returns ``x`` with ``Y^T M x = 0`` and ``(I - M Y Y^T)(K x - b) = 0`` for ``Y`` M-orthonormal rigid
    motions (``None``: no deflation, a clamped body). The tiny shift ``sigma = 1e-3 default_shift(K, M)``
    keeps ``K - sigma M`` nonsingular for the rigid translations; the restriction is exact
    (``x = A^-1 b' - A^-1 B (B^T A^-1 B)^-1 B^T A^-1 b'``, ``A = K - sigma M``, ``B = M Y``,
    ``b' = b - B Y^T b``). ``factor_time`` [s] records the factorization cost; ``sigma`` is exposed so the
    object can be passed as the ``factorization`` of a backend.

    Args:
        K, M: Free-DOF stiffness and mass (sparse).
        Y: M-orthonormal rigid motions (n, d) or ``None``.
        symmetric: ``False`` (legacy): a pivoting SciPy SuperLU of ``A`` (``splu`` defaults).
            ``True``: :class:`newton._src.sim.modal_sparse.ShiftedFactorization` (symmetric ordering,
            diagonal pivots), whose inertia count lets :class:`RestrictedArpackBackend` reuse the same
            factorization for its shift-invert operator (one factorization per update, see
            :func:`shared_restricted_solver`).
    """

    def __init__(self, K, M, Y: np.ndarray | None, *, symmetric: bool = False):
        M = sp.csr_matrix(M)
        if Y is None:
            Y = np.zeros((K.shape[0], 0))
        self.Y = Y
        self.B = np.asarray(M @ Y)
        Kc = sp.csc_matrix(K)
        self.sigma = modal_sparse.default_shift(Kc, M) * 1.0e-3
        self.symmetric = bool(symmetric)
        t_f = time.perf_counter()
        if self.symmetric:
            self.factorization = modal_sparse.ShiftedFactorization(Kc, M, self.sigma)
            self.lu = self.factorization.lu
        else:
            self.factorization = None
            self.lu = spla.splu(sp.csc_matrix(Kc - self.sigma * M))
        self.factor_time = time.perf_counter() - t_f
        B = self.B
        self.AinvB = self.lu.solve(B) if B.shape[1] else np.zeros_like(B)
        S6 = B.T @ self.AinvB
        self.S6 = 0.5 * (S6 + S6.T)
        self.S6inv = np.linalg.inv(self.S6) if B.shape[1] else np.zeros((0, 0))

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Restricted static solve (vector or block of columns)."""
        b = np.asarray(b, dtype=np.float64)
        b = b - self.B @ (self.Y.T @ b)  # remove the rigid-body (inertia relief) part of the load
        x = self.lu.solve(b)
        return x - self.AinvB @ (self.S6inv @ (self.B.T @ x))

    __call__ = solve


def shared_restricted_solver(K, M, Y: np.ndarray | None) -> ShiftedRestrictedSolver:
    """The update's single factorization: a symmetric :class:`ShiftedRestrictedSolver` (with inertia count).

    Passed as ``factorization`` to :class:`RestrictedArpackBackend` (shift-invert operator and Sturm check at
    the same shift) and :class:`RestrictedSubspaceBackend` (block inverse iteration), and used for the static
    Ritz vectors and the modal derivatives of the same update.
    """
    return ShiftedRestrictedSolver(K, M, Y, symmetric=True)


class RestrictedArpackBackend:
    """ARPACK shift-invert on the M-orthogonal complement of ``deflation`` with an exact restricted inverse.

    Satisfies :class:`newton.ModalEigensolverBackend`. Up to ``dense_dof_limit`` free DOFs (or when the request
    is close to the full size) the complement is solved densely (QR complement + LAPACK). Otherwise
    ``x = A^-1 b - A^-1 B (B^T A^-1 B)^-1 B^T A^-1 b`` (``A = K - sigma M``, ``B = M Y``); eigenvalues below
    ``sigma`` are detected by the Haynsworth inertia count ``neg(Z^T A Z) = neg(A) + neg(-B^T A^-1 B) - d``
    and sigma is lowered until none remain. :attr:`last_info` holds the path, final shift, inertia count,
    factorization time and the max restricted residual of the last solve.
    """

    name = "arpack-restricted"
    supports_inertia_count = True
    device = "cpu"

    def __init__(self, *, dense_dof_limit: int = 0):
        self.dense_dof_limit = int(dense_dof_limit)
        self.last_info: dict[str, Any] = {}

    def solve(
        self,
        stiffness,
        mass,
        count: int,
        *,
        deflation: np.ndarray | None = None,
        sigma: float | None = None,
        tol: float = 0.0,
        factorization=None,
        zero_threshold: float = 0.0,
    ) -> newton.ModalEigenResult:
        """Return the ``count`` lowest eigenpairs outside ``deflation``.

        ``sigma`` is ignored. ``factorization``: a symmetric :class:`ShiftedRestrictedSolver` (from
        :func:`shared_restricted_solver`) on the same pencil is reused as the first shift-invert operator
        (its shift and inertia count); only when its Sturm check finds eigenvalues below that shift is a
        new, lower shift factored. ``None`` (legacy): factor at ``default_shift(K, M)``.
        """
        del sigma, tol, zero_threshold
        shared = None
        if isinstance(factorization, ShiftedRestrictedSolver) and factorization.factorization is not None:
            shared = factorization
        t0 = time.perf_counter()
        K, M, Y, k = stiffness, mass, deflation, int(count)
        n = int(K.shape[0])
        M = sp.csr_matrix(M)
        d = 0 if Y is None else int(Y.shape[1])
        if Y is None:
            Y = np.zeros((n, 0))
        info: dict[str, Any] = {}
        B = np.asarray(M @ Y)
        if n - d <= max(self.dense_dof_limit, k + 2):
            Kd = K.toarray() if sp.issparse(K) else np.asarray(K)
            Md = M.toarray()
            if d:
                Q, _ = np.linalg.qr(B, mode="complete")
                Z = Q[:, d:]
                Kz = Z.T @ Kd @ Z
                Mz = Z.T @ Md @ Z
            else:
                Z = np.eye(n)
                Kz, Mz = Kd, Md
            Kz = 0.5 * (Kz + Kz.T)
            Mz = 0.5 * (Mz + Mz.T)
            vals, vecs = sla.eigh(Kz, Mz, subset_by_index=[0, min(k, n - d) - 1])
            V = Z @ vecs
            info.update(path="dense", sigma=None, below_sigma=0)
            self.last_info = info
            return newton.ModalEigenResult(vals, V, backend=self.name, wall_s=time.perf_counter() - t0)

        K = sp.csc_matrix(K)
        sigma = shared.sigma if shared is not None else modal_sparse.default_shift(K, M)
        scale = modal_sparse.matrix_one_norm(K) / max(modal_sparse.matrix_one_norm(M), 1.0e-300)
        tries = []
        fact = None
        below = -1
        info["shared_factorization"] = False
        for attempt in range(12):
            if attempt == 0 and shared is not None:
                fact = shared.factorization
                AinvB = shared.AinvB
                S6 = shared.S6 if d else np.zeros((0, 0))
                info["shared_factorization"] = True
            else:
                info["shared_factorization"] = False
                t_f = time.perf_counter()
                try:
                    fact = modal_sparse.ShiftedFactorization(K, M, sigma)
                    info["t_factor"] = info.get("t_factor", 0.0) + time.perf_counter() - t_f
                except ValueError:
                    sigma *= 1.37
                    continue
                AinvB = np.column_stack([fact.solve(B[:, i]) for i in range(d)]) if d else np.zeros((n, 0))
                S6 = B.T @ AinvB if d else np.zeros((0, 0))
                S6 = 0.5 * (S6 + S6.T)
            negA = fact.negative_pivot_count
            if negA < 0:
                below = -1
                tries.append((sigma, -1))
                break
            below = negA + int(np.count_nonzero(np.linalg.eigvalsh(-S6) < 0.0)) - d if d else negA
            tries.append((sigma, below))
            if below <= 0:
                below = max(below, 0)
                break
            sigma = sigma - scale * (10.0 ** (attempt - 5))
        info.update(path="arpack", sigma=sigma, below_sigma=below, sigma_tries=tries)
        if d:
            S6inv = np.linalg.inv(S6)

        def solve(b):
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            x = fact.solve(b)
            if d:
                x = x - AinvB @ (S6inv @ (B.T @ x))
            return x

        op = spla.LinearOperator((n, n), matvec=solve, dtype=np.float64)
        v0 = solve(modal_sparse.arpack_start_vector(n))
        vals, vecs = spla.eigsh(K, k=k, M=M, sigma=sigma, which="LM", OPinv=op, v0=v0, tol=0.0)
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        # residual of the restricted problem: (I - B Y^T)(K v - lam M v)
        R = K @ vecs - (M @ vecs) * vals
        R = R - B @ (Y.T @ R)
        info["max_rel_residual"] = float(np.max(np.linalg.norm(R, axis=0) / (fact.norm * np.linalg.norm(vecs, axis=0))))
        self.last_info = info
        return newton.ModalEigenResult(
            vals,
            vecs,
            inertia_count=None if below < 0 else int(below) + d,
            backend=self.name,
            wall_s=time.perf_counter() - t0,
            inertia_shift=float(sigma),
        )


class RestrictedSubspaceBackend:
    """Warm-started subspace iteration (Bathe) with the shared restricted factorization.

    Satisfies :class:`newton.ModalEigensolverBackend`. The start block is :attr:`previous_vectors` (the previous
    basis on the free DOFs) with the deflated directions removed, padded with ``margin`` deterministic
    normal random vectors (seed :attr:`seed`); ``iterations`` block inverse iterations
    ``X <- solve(M X)`` with the :class:`ShiftedRestrictedSolver` passed as ``factorization`` (built here
    when ``None``), then Rayleigh-Ritz. Returns all Ritz pairs of the block in ascending order;
    :attr:`last_info` holds the residuals.
    """

    name = "subspace-restricted"
    supports_inertia_count = False
    device = "cpu"

    def __init__(
        self, *, iterations: int = 1, margin: int = 2, previous_vectors: np.ndarray | None = None, seed: int = 1234
    ):
        self.iterations = max(int(iterations), 1)
        self.margin = int(margin)
        self.previous_vectors = previous_vectors
        self.seed = int(seed)
        self.last_info: dict[str, Any] = {}

    def solve(
        self,
        stiffness,
        mass,
        count: int,
        *,
        deflation: np.ndarray | None = None,
        sigma: float | None = None,
        tol: float = 0.0,
        factorization=None,
        zero_threshold: float = 0.0,
    ) -> newton.ModalEigenResult:
        """Return the Ritz pairs of the warm-started block (at least ``count`` when the block allows)."""
        del sigma, tol, zero_threshold
        t0 = time.perf_counter()
        Kf, Mf, Yf = stiffness, mass, deflation
        solve = factorization if factorization is not None else ShiftedRestrictedSolver(Kf, Mf, Yf)
        X = self.previous_vectors
        if X is None or X.shape[0] != Kf.shape[0]:
            X = np.zeros((Kf.shape[0], 0))
        X = np.array(X, dtype=np.float64, copy=True)
        pad = max(self.margin, int(count) - X.shape[1])
        if pad > 0:
            rng = np.random.default_rng(self.seed)
            X = np.column_stack([X, rng.standard_normal((X.shape[0], pad))])
        B = None
        if Yf is not None:
            B = np.asarray(Mf @ Yf)
            X = X - Yf @ (B.T @ X)
        for _ in range(self.iterations):
            X = solve(np.asarray(Mf @ X))
            Gx = X.T @ (Mf @ X)
            w, U = np.linalg.eigh(0.5 * (Gx + Gx.T))
            keep = w > 1.0e-14 * float(np.max(w))
            X = X @ (U[:, keep] / np.sqrt(w[keep]))
        Kx = X.T @ (Kf @ X)
        Mx = X.T @ (Mf @ X)
        vals, vecs = sla.eigh(0.5 * (Kx + Kx.T), 0.5 * (Mx + Mx.T))
        V = X @ vecs
        R = Kf @ V - (Mf @ V) * vals
        if B is not None:
            R = R - B @ (Yf.T @ R)
        knorm = modal_sparse.matrix_one_norm(sp.csc_matrix(Kf))
        rel = np.linalg.norm(R, axis=0) / (knorm * np.linalg.norm(V, axis=0))
        self.last_info = {
            "path": "subspace",
            "iterations": self.iterations,
            "block": int(X.shape[1]),
            "rel_residual": rel.tolist(),
            "max_rel_residual": float(np.max(rel)),
        }
        self.previous_vectors = V
        return newton.ModalEigenResult(vals, V, backend=self.name, wall_s=time.perf_counter() - t0)


def resolve_backend(spec, *, margin: int = 2, iterations: int = 1, dense_dof_limit: int = 0):
    """Map the controller's ``eigensolver`` option to a backend (see the module docstring)."""
    if isinstance(spec, (str, newton.ModalEigensolver)):
        kind = newton.ModalEigensolver(spec) if isinstance(spec, str) else spec
        if kind in (newton.ModalEigensolver.AUTO, newton.ModalEigensolver.ARPACK):
            return RestrictedArpackBackend(dense_dof_limit=dense_dof_limit)
        if kind is newton.ModalEigensolver.SUBSPACE:
            return RestrictedSubspaceBackend(iterations=iterations, margin=margin)
        return modal_sparse.resolve_eigensolver(kind, margin=margin)
    if isinstance(spec, newton.ModalEigensolverBackend):
        return modal_sparse.resolve_eigensolver(spec)
    raise TypeError(
        "eigensolver must be 'auto', 'arpack', 'subspace', 'dense', a newton.ModalEigensolver or a "
        f"newton.ModalEigensolverBackend, got {type(spec).__name__}"
    )


def solve_restricted_eigen(
    backend,
    Kf,
    Mf,
    Yf: np.ndarray | None,
    count: int,
    *,
    previous: np.ndarray | None,
    shared_solver,
    seed: int,
    share_arpack: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run ``backend`` on the free-DOF pencil with the rigid motions ``Yf`` deflated.

    ``previous`` is the previous basis on the free DOFs (warm start of backends with a
    ``previous_vectors`` attribute); ``shared_solver`` is a zero-argument callable returning the update's
    :class:`ShiftedRestrictedSolver` (built lazily: always for the restricted subspace backend, and for the
    restricted ARPACK backend with ``share_arpack``, which then reuses it as its shift-invert factorization).
    Returns ``(values, vectors, info)`` in ascending order.
    """
    info: dict[str, Any] = {}
    if isinstance(backend, RestrictedSubspaceBackend):
        backend.previous_vectors = previous
        backend.seed = seed
        res = backend.solve(Kf, Mf, count, deflation=Yf, factorization=shared_solver())
        info.update(backend.last_info)
        return np.asarray(res.values), np.asarray(res.vectors), info
    if isinstance(backend, RestrictedArpackBackend):
        res = backend.solve(Kf, Mf, count, deflation=Yf, factorization=shared_solver() if share_arpack else None)
        info.update(backend.last_info)
        return np.asarray(res.values), np.asarray(res.vectors), info
    # generic newton.ModalEigensolverBackend (Newton built-ins, Warpack, test doubles)
    if hasattr(backend, "previous_vectors") and previous is not None:
        prev = previous
        if Yf is not None:
            prev = prev - Yf @ (Yf.T @ (Mf @ prev))
        backend.previous_vectors = prev
    device = getattr(backend, "device", "cpu")
    K_in = modal_sparse.to_backend_matrix(sp.csc_matrix(Kf), device=device)
    M_in = modal_sparse.to_backend_matrix(sp.csc_matrix(Mf), device=device)
    t0 = time.perf_counter()
    res = backend.solve(K_in, M_in, int(count), deflation=Yf)
    values, vectors = modal_sparse.validate_eigen_result(
        res,
        size=int(Kf.shape[0]),
        count=int(count),
        deflated=0 if Yf is None else int(Yf.shape[1]),
        supports_inertia_count=bool(getattr(backend, "supports_inertia_count", False)),
    )
    info.update(
        path=str(res.backend or getattr(backend, "name", type(backend).__name__)),
        inertia_count=res.inertia_count,
        backend_wall_s=float(res.wall_s or (time.perf_counter() - t0)),
    )
    return values, vectors, info
