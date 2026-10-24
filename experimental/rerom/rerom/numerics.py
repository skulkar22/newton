# ruff: noqa: TID253
"""Shared re-ROM numerics (host, numpy/scipy): one implementation of each building block.

Conventions: nodal DOF layout ``x_flat[3 n + i]``; nodal bases are ``(3N, m)`` matrices; quaternions
are xyzw (Newton/Warp order); ``M`` is the consistent mass matrix of the body (sparse).

Contents
--------
quaternions        :func:`quat_mul`, :func:`quat_conj`, :func:`quat_from_rotvec`, :func:`quat_to_matrix`,
                   :func:`matrix_to_quat`, :func:`rotvec_of_matrix`, :func:`quat_angle`, :func:`skew`
nodal layout       :func:`node_dofs`, :func:`to_nodal`, :func:`idw_rows`, :func:`rbe2_rows`
rigid motions      :func:`nodal_rigid_vectors`, :func:`rigid_properties`, :func:`m_orthonormal_rigid`
M-orthonormal      :class:`MOrthonormalBlock` (Gram-Schmidt accumulation with drop tolerance and fill),
blocks             :func:`m_orthonormalize`
enrichment         :func:`ritz_enrichment` (load / velocity / acceleration static Ritz vectors and the
                   block-Krylov continuation ``K^-1 M r_k``)
re-diagonalization :func:`rayleigh_ritz` (``Phi^T M Phi = I``, ``Phi^T K Phi`` diagonal, Newton's
                   ``_orient_modes`` sign rule)
modal derivatives  :func:`tangent_derivative`, :func:`static_modal_derivatives`
state projection   :func:`split_rigid_velocity`, :func:`project_modal_velocity`,
                   :func:`constrain_frame_velocity`
diagnostics        :func:`compare_eigen`, :func:`imex_spectral_radius`
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from newton._src.sim import modal_sparse
from newton._src.sim.modal import _orient_modes

__all__ = [
    "MOrthonormalBlock",
    "compare_eigen",
    "constrain_frame_velocity",
    "host_sparse",
    "idw_rows",
    "imex_spectral_radius",
    "m_orthonormal_rigid",
    "m_orthonormalize",
    "matrix_to_quat",
    "nodal_rigid_vectors",
    "node_dofs",
    "project_modal_velocity",
    "quat_angle",
    "quat_conj",
    "quat_from_rotvec",
    "quat_mul",
    "quat_to_matrix",
    "rayleigh_ritz",
    "rbe2_rows",
    "rigid_properties",
    "ritz_enrichment",
    "rotvec_of_matrix",
    "skew",
    "split_rigid_velocity",
    "static_modal_derivatives",
    "tangent_derivative",
    "to_nodal",
]


# ----------------------------------------------------------------------------------------------
# quaternions (xyzw) and rotations
# ----------------------------------------------------------------------------------------------


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a b`` of xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse of a unit quaternion), xyzw."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_from_rotvec(theta: np.ndarray) -> np.ndarray:
    """Unit xyzw quaternion of the rotation vector ``theta`` [rad]."""
    angle = float(np.linalg.norm(theta))
    if angle < 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = theta / angle
    s = math.sin(0.5 * angle)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle)])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of an xyzw quaternion (normalized on the fly)."""
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 0.0 else 0.0
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1.0 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1.0 - s * (x * x + y * y)],
        ]
    )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit xyzw quaternion (largest-pivot branch, numerically stable)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def rotvec_of_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation vector [rad] of a rotation matrix."""
    c = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    th = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return 0.5 * v
    return th * v / (2.0 * math.sin(th))


def quat_angle(q: np.ndarray) -> float:
    """Rotation angle [rad] of a (unit) xyzw quaternion, in [0, pi]."""
    return 2.0 * math.atan2(float(np.linalg.norm(q[:3])), abs(float(q[3])))


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix ``[v]x``."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


# ----------------------------------------------------------------------------------------------
# nodal layout
# ----------------------------------------------------------------------------------------------


def node_dofs(nodes: np.ndarray) -> np.ndarray:
    """DOF indices ``3 n + i`` of the given nodes (flattened, node-major)."""
    nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
    return (3 * nodes[:, None] + np.arange(3)[None, :]).reshape(-1)


def to_nodal(Phi: np.ndarray, n: int) -> np.ndarray:
    """(3N, m) -> (N, m, 3)."""
    return np.transpose(Phi.reshape(n, 3, -1), (0, 2, 1))


def idw_rows(prior_points: np.ndarray, queries: np.ndarray, eps: float) -> np.ndarray:
    """Inverse-distance weights exactly as ``ModalBasis._evaluate_many`` (float32 distances), (q, s)."""
    prior = np.asarray(prior_points, dtype=np.float32)
    rows = np.zeros((queries.shape[0], prior.shape[0]), dtype=np.float64)
    eps2 = eps * eps
    exact_tol2 = max(eps2, 1.0e-14)
    for i, point in enumerate(np.asarray(queries, dtype=np.float32)):
        delta = prior - point[None, :]
        dist2 = np.einsum("sc,sc->s", delta, delta)
        nearest = int(np.argmin(dist2))
        if float(dist2[nearest]) <= exact_tol2:
            rows[i, nearest] = 1.0
            continue
        w = (1.0 / (dist2 + np.float32(eps2))).astype(np.float32)
        w /= np.sum(w)
        rows[i] = w
    return rows


def rbe2_rows(points: np.ndarray, position: np.ndarray) -> np.ndarray:
    """(3F, 6) rigid rows: nodal translation of face points for frame motion ``[t, theta]`` about ``position``."""
    return np.vstack([np.hstack((np.eye(3), -skew(p - position))) for p in points])


# ----------------------------------------------------------------------------------------------
# rigid motions
# ----------------------------------------------------------------------------------------------


def nodal_rigid_vectors(x: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Return the 6 nodal rigid motions (3 translations, 3 rotations about ``center``), (3N, 6)."""
    n = x.shape[0]
    Y = np.zeros((3 * n, 6))
    Y[:, :3] = np.tile(np.eye(3), (n, 1))
    d = x - center[None, :]
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        Y[:, 3 + k] = np.cross(e, d).reshape(-1)
    return Y


def rigid_properties(mass_matrix, x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Rigid mass [kg], COM [m] and inertia about the COM [kg m^2] of nodes ``x`` (N,3) under ``M``.

    Same projection as ``ModalGeneratorFEM._rigid_properties`` (spatial mass of the rigid fields).
    """
    x = np.asarray(x, dtype=np.float64)
    Y0 = nodal_rigid_vectors(x, np.zeros(3))
    spatial = Y0.T @ (mass_matrix @ Y0)
    mass = float(np.trace(spatial[:3, :3]) / 3.0)
    com_skew = -spatial[:3, 3:] / mass
    com = np.array([com_skew[2, 1], com_skew[0, 2], com_skew[1, 0]])
    inertia = 0.5 * (spatial[3:, 3:] + spatial[3:, 3:].T)
    inertia -= mass * ((com @ com) * np.eye(3) - np.outer(com, com))
    return mass, com, inertia


def m_orthonormal_rigid(Y: np.ndarray, M) -> np.ndarray:
    """M-orthonormalize the rigid vectors ``Y`` (3N, 6) by a Cholesky factor of their Gram matrix."""
    G = Y.T @ (M @ Y)
    Lg = np.linalg.cholesky(0.5 * (G + G.T))
    return Y @ np.linalg.inv(Lg).T


# ----------------------------------------------------------------------------------------------
# M-orthonormal blocks
# ----------------------------------------------------------------------------------------------


def m_orthonormalize(V: np.ndarray, M, rel_cutoff: float = 1.0e-12) -> np.ndarray:
    """M-orthonormal basis of ``span(V)`` from the eigen-decomposition of the Gram matrix.

    Directions with Gram eigenvalue ``<= rel_cutoff * max`` are dropped.
    """
    Gv = V.T @ (M @ V)
    w, Qv = np.linalg.eigh(0.5 * (Gv + Gv.T))
    keep = w > rel_cutoff * float(np.max(w))
    return V @ (Qv[:, keep] / np.sqrt(w[keep]))


class MOrthonormalBlock:
    """Accumulates an M-orthonormal block by two-pass classical Gram-Schmidt.

    Args:
        M: Mass matrix of the space (sparse or dense).
        initial: M-orthonormal starting columns ``(n, k)`` (``None``: empty).
        drop_tol: A candidate is dropped when its norm after orthogonalization falls below
            ``drop_tol`` times its norm before.

    ``log`` records ``(name, ratio)`` for every candidate (ratio 0 for a zero candidate);
    ``accepted`` and ``dropped`` list the candidate names.
    """

    def __init__(self, M, initial: np.ndarray | None = None, *, n: int | None = None, drop_tol: float = 1.0e-6):
        self.M = M
        if initial is None:
            if n is None:
                raise ValueError("pass initial columns or the space dimension n")
            initial = np.zeros((n, 0))
        self.columns = initial
        self.drop_tol = float(drop_tol)
        self.log: list[tuple[str, float]] = []
        self.accepted: list[str] = []
        self.dropped: list[str] = []

    @property
    def count(self) -> int:
        """Number of columns."""
        return int(self.columns.shape[1])

    def accept(self, name: str, v: np.ndarray) -> np.ndarray | None:
        """Orthogonalize ``v`` against the block, append it normalized and return it (``None`` if dropped)."""
        M = self.M
        nrm0 = math.sqrt(max(float(v @ (M @ v)), 0.0))
        if nrm0 == 0.0:
            self.log.append((name, 0.0))
            self.dropped.append(name)
            return None
        current = self.columns
        for _ in range(2):
            v = v - current @ (current.T @ (M @ v))
        nrm = math.sqrt(max(float(v @ (M @ v)), 0.0))
        ratio = nrm / nrm0
        self.log.append((name, ratio))
        if ratio < self.drop_tol:
            self.dropped.append(name)
            return None
        v = v / nrm
        self.columns = np.column_stack([current, v])
        self.accepted.append(name)
        return v

    def fill(self, V: np.ndarray, start: int, target: int, abs_tol: float = 1.0e-6) -> None:
        """Append columns ``V[:, start:]`` (M-orthonormal) until ``target`` columns, skipping dependent ones."""
        M = self.M
        n_fill = target - self.count
        j = start
        while n_fill > 0 and j < V.shape[1]:
            v = V[:, j].copy()
            current = self.columns
            for _ in range(2):
                v = v - current @ (current.T @ (M @ v))
            nrm = math.sqrt(max(float(v @ (M @ v)), 0.0))
            if nrm > abs_tol:
                self.columns = np.column_stack([current, v / nrm])
                n_fill -= 1
            j += 1


def ritz_enrichment(
    block: MOrthonormalBlock,
    candidates: list[tuple[str, np.ndarray]],
    solve: Callable[[np.ndarray], np.ndarray],
    n_ritz: int,
    *,
    krylov: bool,
    M,
) -> list[np.ndarray]:
    """Append up to ``n_ritz`` static Ritz vectors ``solve(b)`` of the load candidates to ``block``.

    ``candidates`` are ``(name, load)`` in priority order (typically ``f_int``, ``f_ext``, ``M v``,
    ``M a``); zero loads are skipped. With ``krylov`` and fewer than ``n_ritz`` accepted vectors, the
    block-Krylov continuation ``solve(M r_k)`` (Wilson-Yuan-Dickens load-dependent Ritz vectors) fills
    up, at most ``4 n_ritz`` attempts. Returns the accepted vectors.
    """
    ritz_vecs: list[np.ndarray] = []
    for name, b in candidates:
        if len(ritz_vecs) >= n_ritz:
            break
        if not np.any(b):
            block.log.append((name, 0.0))
            continue
        v = block.accept(name, solve(b))
        if v is not None:
            ritz_vecs.append(v)
    if krylov:
        k_src = 0
        attempts = 0
        while len(ritz_vecs) < n_ritz and k_src < len(ritz_vecs) and attempts < 4 * n_ritz:
            v = block.accept(f"K^-1 M r{k_src}", solve(M @ ritz_vecs[k_src]))
            if v is not None:
                ritz_vecs.append(v)
            k_src += 1
            attempts += 1
    return ritz_vecs


# ----------------------------------------------------------------------------------------------
# Rayleigh-Ritz re-diagonalization
# ----------------------------------------------------------------------------------------------


def rayleigh_ritz(B: np.ndarray, K, M) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate the basis ``B`` so that ``Phi^T M Phi = I`` and ``Phi^T K Phi = diag(lam)``.

    Signs follow Newton's ``_orient_modes`` rule (largest-magnitude entry of each column positive).
    Returns ``(lam, Phi, Q)`` with ``Phi = B Q`` (``Q`` includes the sign flips).
    """
    Kr = B.T @ (K @ B)
    Mr = B.T @ (M @ B)
    lam, Q = sla.eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T))
    Phi0 = B @ Q
    Phi = _orient_modes(Phi0)
    if Phi.shape[1]:
        piv = np.argmax(np.abs(Phi0), axis=0)
        cols = np.arange(Phi.shape[1])
        sign = np.where(Phi[piv, cols] * Phi0[piv, cols] < 0.0, -1.0, 1.0)
        Q = Q * sign
    return lam, Phi, Q


# ----------------------------------------------------------------------------------------------
# static modal derivatives
# ----------------------------------------------------------------------------------------------


def host_sparse(K):
    """SciPy CSR copy of a tangent: SciPy/dense input, or a ``warp.sparse.BsrMatrix`` (via ``newton.to_backend_matrix``)."""
    if modal_sparse.is_bsr_matrix(K):
        K = modal_sparse.to_backend_matrix(K, device="cpu")
    return sp.csr_matrix(K)


def tangent_derivative(assembler, x0: np.ndarray, phi: np.ndarray, h: float):
    """Directional derivative ``dK_T/dq`` along ``phi`` by a central difference of the tangent (step ``h``)."""
    Kp = host_sparse(assembler.evaluate(x0 + h * phi, need_tangent=True)[2])
    Km = host_sparse(assembler.evaluate(x0 - h * phi, need_tangent=True)[2])
    return (Kp - Km) / (2.0 * h)


def fd_step(phi: np.ndarray, scale: float, char_length: float) -> float:
    """Finite-difference step giving a peak displacement of ``scale * char_length`` along ``phi``."""
    return scale * char_length / max(float(np.max(np.abs(phi))), 1.0e-30)


def static_modal_derivatives(
    assembler,
    x0: np.ndarray,
    Phi: np.ndarray,
    k: int,
    solve_restricted: Callable[[np.ndarray], np.ndarray],
    free: np.ndarray,
    *,
    fd_scale: float,
    char_length: float,
) -> np.ndarray:
    """Static modal derivatives ``Theta_ij = -K_T^-1 (dK_T/dq_i) phi_j`` of the first ``k`` columns, (3N, k, k).

    ``solve_restricted`` acts on the ``free`` DOFs (clamped DOFs stay zero); ``Theta`` is symmetric in i, j.
    """
    n = Phi.shape[0]
    Theta = np.zeros((n, k, k))
    for i in range(k):
        phi_i = Phi[:, i]
        dK = tangent_derivative(assembler, x0, phi_i, fd_step(phi_i, fd_scale, char_length))
        for j in range(i, k):
            th = np.zeros(n)
            th[free] = -solve_restricted((dK @ Phi[:, j])[free])
            Theta[:, i, j] = th
            Theta[:, j, i] = th
    return Theta


# ----------------------------------------------------------------------------------------------
# momentum-consistent state projection
# ----------------------------------------------------------------------------------------------


def split_rigid_velocity(u: np.ndarray, Y: np.ndarray, M) -> tuple[np.ndarray, np.ndarray]:
    """M-orthogonal projection of the nodal velocity ``u`` onto the rigid motions ``Y`` (3N, 6).

    Returns ``(coeff, u_rigid)`` with ``coeff = [t, w]`` (translation, angular velocity about the
    centre of ``Y``) and ``u_rigid = Y coeff``. The remainder ``u - u_rigid`` carries no linear or
    angular momentum about that centre.
    """
    Gr = Y.T @ (M @ Y)
    coeff = np.linalg.solve(Gr, Y.T @ (M @ u))
    return coeff, Y @ coeff


def project_modal_velocity(u_el: np.ndarray, Phi: np.ndarray, M, mode_mass_matrix: np.ndarray) -> np.ndarray:
    """M-orthogonal projection of an elastic nodal velocity onto ``span(Phi)``: modal velocities ``qd``."""
    a = Phi.T @ (M @ u_el)
    return np.linalg.solve(mode_mass_matrix, a)


def constrain_frame_velocity(qd: np.ndarray, F: np.ndarray, fdot: np.ndarray) -> np.ndarray:
    """Minimal-norm correction of ``qd`` so that the interface-frame velocity ``F qd`` equals ``fdot``."""
    corr = F.T @ np.linalg.solve(F @ F.T, fdot - F @ qd)
    return qd + corr


# ----------------------------------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------------------------------


def compare_eigen(vals_ref, V_ref, vals, V, M, n: int) -> dict:
    """Accuracy of an approximate eigenbasis (vals, V) against a reference on the lowest n modes.

    Principal angles between the M-orthonormalized spans, per-mode angle (|v_i^T M v_ref_i|) and
    relative eigenvalue error.
    """
    n = int(min(n, V_ref.shape[1], V.shape[1]))

    def orth(A):
        G = A.T @ (M @ A)
        w, U = np.linalg.eigh(0.5 * (G + G.T))
        return A @ (U / np.sqrt(np.maximum(w, 1.0e-300)))

    A = orth(V_ref[:, :n])
    Bm = orth(V[:, :n])
    cos = np.clip(np.linalg.svd(A.T @ (M @ Bm), compute_uv=False), -1.0, 1.0)
    angles = np.arccos(cos)
    per_mode = []
    for i in range(n):
        a = V_ref[:, i] / math.sqrt(max(float(V_ref[:, i] @ (M @ V_ref[:, i])), 1e-300))
        b = V[:, i] / math.sqrt(max(float(V[:, i] @ (M @ V[:, i])), 1e-300))
        per_mode.append(float(np.arccos(min(abs(float(a @ (M @ b))), 1.0))))
    vr = np.asarray(vals_ref[:n])
    rel = np.abs(np.asarray(vals[:n]) - vr) / np.maximum(np.abs(vr), 1e-300)
    return {
        "n": n,
        "max_principal_angle": float(np.max(angles)),
        "principal_angles": angles.tolist(),
        "per_mode_angle": per_mode,
        "eig_rel_err": rel.tolist(),
        "max_eig_rel_err": float(np.max(rel)),
    }


def imex_spectral_radius(lam: np.ndarray, K_exp: np.ndarray, h: float, damping: np.ndarray | None = None) -> float:
    """Spectral radius of one linearized IMEX step (unit modal mass).

    Linear part implicit Euler (VBD), remainder Jacobian ``K_exp`` explicit (lagged):
    ``(I + h C + h^2 Lambda) q1 = (I + h C - h^2 K_exp) q0 + h v0``, ``v1 = (q1 - q0) / h``.
    A value above 1 predicts growth of the lagged scheme.
    """
    m = len(lam)
    C = np.zeros((m, m)) if damping is None else np.diag(damping)
    A = np.eye(m) + h * C + h * h * np.diag(lam)
    Ainv = np.linalg.inv(A)
    Bq = Ainv @ (np.eye(m) + h * C - h * h * K_exp)
    Bv = Ainv * h
    top = np.hstack([Bq, Bv])
    bot = np.hstack([(Bq - np.eye(m)) / h, Bv / h])
    return float(np.max(np.abs(np.linalg.eigvals(np.vstack([top, bot])))))
