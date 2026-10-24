# ruff: noqa: TID253, PLW2901  (moved as-is from the experiment package)
"""WP3a: hyper-reduced modal internal force on an element subset (host-side numpy).

``ReducedElementForce`` evaluates, for the current reference nodes ``r_ref`` and nodal basis ``Phi``
(3N, m), the weighted element sums

    W_S(q) = sum_e w_e W_e(x),   F_S(q) = sum_e w_e Phi_e^T f_e(x),   K_S(q) = sum_e w_e Phi_e^T K_e(x) Phi_e

with ``x = r_ref + Phi q`` restricted to the element nodes, without any global assembly.

Optional quadratic manifold (``set_manifold``): ``x = r_ref + Phi q + 1/2 Theta:(p, p)`` with
``p = S^T q`` (k manifold coordinates, ``S`` = first k unit vectors at an update, rotated by folds) and
static modal derivatives ``Theta`` (3N, k, k); the force is projected on the tangent
``P(q) = Phi + (Theta p) S^T`` and the tangent includes the curvature term ``S (Theta_ab . f) S^T``. With all
elements and unit weights these are exactly ``W(x)``, ``Phi^T f_int(x)`` and ``Phi^T K_T(x) Phi``.
Element quantities come from ``NumpyTetAssembler.element_terms`` (WP-A, same kernels as
``evaluate``).

Element subsets:

* ``full_set(asm)``: all elements, unit weights (exact; used to prove the mechanics).
* ``stratified_volume_sample(asm, n)``: first-cut cubature. Elements ordered along a key (centroid
  x by default), split into ``n`` strata of equal volume, one random element per stratum with weight
  ``stratum volume / element volume`` (unbiased for any element-additive integrand).
* ``ecsw_greedy(G, b, tol)``: energy-conserving sampling and weighting (Farhat et al.): sparse
  non-negative least squares ``min ||G w - b||, w >= 0`` by greedy active-set selection, stopped at
  ``||G w - b|| <= tol ||b||``. ``ecsw_training_matrix`` builds ``G`` from training configurations and
  test bases (rows = projected element forces).
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import scipy.optimize as sopt

__all__ = [
    "ReducedElementForce",
    "ecsw_greedy",
    "ecsw_training_matrix",
    "full_set",
    "stratified_volume_sample",
]


class ReducedElementForce:
    """Weighted element-subset evaluation of the modal energy / force / tangent (see module doc)."""

    def __init__(self, assembler, element_ids=None, weights=None):
        self.asm = assembler
        n_el = int(assembler.mesh.tets.shape[0])
        if element_ids is None:
            self.ids = None
            self.n_el = n_el
            self.tets = assembler.mesh.tets
            self.w = np.ones(n_el) if weights is None else np.asarray(weights, dtype=np.float64)
        else:
            self.ids = np.asarray(element_ids, dtype=np.int64)
            self.n_el = int(self.ids.size)
            self.tets = assembler.mesh.tets[self.ids]
            self.w = np.ones(self.n_el) if weights is None else np.asarray(weights, dtype=np.float64)
        if self.w.shape != (self.n_el,):
            raise ValueError(f"weights must have shape ({self.n_el},), got {self.w.shape}")
        self.m = 0
        self.Theta_e = None
        self.S = None
        self.timings = {"force": 0.0, "tangent": 0.0, "n_force": 0, "n_tangent": 0}

    @property
    def is_full(self) -> bool:
        return self.ids is None and bool(np.all(self.w == 1.0))

    def set_basis(self, r_ref: np.ndarray, Phi: np.ndarray) -> None:
        r_ref = np.asarray(r_ref, dtype=np.float64).reshape(-1, 3)
        N = r_ref.shape[0]
        m = int(Phi.shape[1])
        self.m = m
        self.x0 = r_ref[self.tets]  # (E,4,3)
        self.Phi_e = np.ascontiguousarray(
            np.asarray(Phi, dtype=np.float64).reshape(N, 3, m)[self.tets].reshape(-1, 12, m)
        )
        self._refresh_weighted()

    def set_manifold(self, Theta: np.ndarray | None) -> None:
        """Static modal derivatives (3N, k, k) of the first k basis columns (None: linear kinematics)."""
        if Theta is None:
            self.Theta_e, self.S = None, None
            return
        k = int(Theta.shape[1])
        N3 = Theta.shape[0]
        self.Theta_e = np.ascontiguousarray(
            np.asarray(Theta, dtype=np.float64).reshape(N3 // 3, 3, k, k)[self.tets].reshape(-1, 12, k, k)
        )
        self.S = np.eye(self.m)[:, :k]

    def _refresh_weighted(self):
        self.wPhi = (self.w[:, None, None] * self.Phi_e).reshape(-1, self.m)  # (E*12, m)
        self.Phi_flat = self.Phi_e.reshape(-1, self.m)

    def rotate(self, Q: np.ndarray) -> None:
        """Basis change ``Phi <- Phi Q`` (fold)."""
        self.Phi_e = np.ascontiguousarray(self.Phi_e @ Q)
        if self.S is not None:
            self.S = Q.T @ self.S
        self._refresh_weighted()

    def positions(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        x = self.x0 + (self.Phi_flat @ q).reshape(-1, 4, 3)
        if self.Theta_e is not None:
            p = self.S.T @ q
            x = x + 0.5 * np.einsum("eiab,a,b->ei", self.Theta_e, p, p).reshape(-1, 4, 3)
        return x

    def evaluate(
        self, q: np.ndarray, *, need_tangent: bool = False, need_energy: bool = False, manifold_curvature: bool = True
    ):
        """Return ``(W_S or None, F_S (m,), K_S (m,m) or None)`` at ``x = r_ref + Phi q`` (+ manifold term).

        ``manifold_curvature=False`` leaves the ``Theta . f`` curvature term out of ``K_S``.
        """
        t0 = time.perf_counter()
        q = np.asarray(q, dtype=np.float64)
        x_e = self.positions(q)
        We, fe, Ke = self.asm.element_terms(x_e, self.ids, need_tangent=need_tangent)
        W = float(self.w @ We) if need_energy else None
        K = None
        if self.Theta_e is None:
            P_e, wP = self.Phi_e, self.wPhi
        else:
            p = self.S.T @ q
            P_e = self.Phi_e + np.einsum("eiab,b->eia", self.Theta_e, p) @ self.S.T
            wP = (self.w[:, None, None] * P_e).reshape(-1, self.m)
        F = wP.T @ fe.reshape(-1)
        if need_tangent:
            KP = Ke.reshape(-1, 12, 12) @ P_e  # (E,12,m)
            K = wP.T @ KP.reshape(-1, self.m)
            if self.Theta_e is not None and manifold_curvature:
                wf = self.w[:, None] * fe.reshape(-1, 12)
                C = np.einsum("eiab,ei->ab", self.Theta_e, wf)
                K = K + self.S @ C @ self.S.T
            K = 0.5 * (K + K.T)
            self.timings["tangent"] += time.perf_counter() - t0
            self.timings["n_tangent"] += 1
        else:
            self.timings["force"] += time.perf_counter() - t0
            self.timings["n_force"] += 1
        return W, F, K


# ----------------------------------------------------------------------------------------------
# element subsets
# ----------------------------------------------------------------------------------------------


def full_set(assembler):
    """All elements, unit weights."""
    n = int(assembler.mesh.tets.shape[0])
    return np.arange(n), np.ones(n)


def stratified_volume_sample(assembler, n: int, *, seed: int = 0, key: np.ndarray | None = None):
    """``n`` strata of equal volume along ``key`` (default: centroid x); one random element each.

    Returns (element_ids, weights) with ``weight = stratum volume / element volume``.
    """
    vol = np.asarray(assembler.vol, dtype=np.float64)
    n_el = vol.size
    if n >= n_el:
        return full_set(assembler)
    if key is None:
        key = assembler.mesh.nodes[assembler.mesh.tets].mean(axis=1)[:, 0]
    order = np.argsort(key, kind="stable")
    cum = np.cumsum(vol[order])
    edges = np.linspace(0.0, cum[-1], n + 1)
    strata = np.clip(np.searchsorted(cum, edges[1:-1], side="left"), 0, n_el - 1)
    bounds = np.concatenate([[0], strata + 1, [n_el]])
    rng = np.random.default_rng(seed)
    ids, w = [], []
    for a, b in itertools.pairwise(bounds):
        if b <= a:
            continue
        members = order[a:b]
        pick = int(rng.choice(members, p=vol[members] / vol[members].sum()))
        ids.append(pick)
        # importance weight for a volume-proportional pick: stratum volume / element volume
        w.append(float(vol[members].sum() / vol[pick]))
    ids = np.asarray(ids, dtype=np.int64)
    w = np.asarray(w)
    # merge duplicates (cannot happen with disjoint strata, kept for safety)
    uniq, inv = np.unique(ids, return_inverse=True)
    return uniq, np.bincount(inv, weights=w)


def ecsw_training_matrix(assembler, configs, bases, *, increments=None, normalize=True):
    """ECSW training matrix.

    ``configs``: list of body-frame node arrays (N,3) or flat (3N,); ``bases``: list of (3N, m_k) test
    bases, one per config. Row block k = ``Phi_k^T f_e(x_k)`` for every element e (columns).
    ``increments``: optional list of (x_ref_k, K-free) pairs is not used; the force blocks already
    contain the nonlinear part. Each block is scaled by its norm when ``normalize``.
    Returns (G (rows, E), b = G @ 1).
    """
    tets = assembler.mesh.tets
    blocks = []
    for x, Phi in zip(configs, bases, strict=False):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
        _, fe, _ = assembler.element_terms(x[tets], None, need_tangent=False)  # (E,4,3)
        m = Phi.shape[1]
        Pe = np.asarray(Phi).reshape(-1, 3, m)[tets].reshape(-1, 12, m)  # (E,12,m)
        blk = np.einsum("eim,ei->me", Pe, fe.reshape(-1, 12))  # (m, E)
        if normalize:
            nrm = np.linalg.norm(blk.sum(axis=1))
            if nrm > 0:
                blk = blk / nrm
        blocks.append(blk)
    G = np.vstack(blocks)
    return G, G.sum(axis=1)


def ecsw_greedy(G: np.ndarray, b: np.ndarray, tol: float = 1.0e-2, max_elements: int | None = None, verbose=False):
    """Sparse NNLS (ECSW, Farhat-Chapman-Avery): greedy element selection + NNLS on the active set.

    Returns (element_ids, weights, info) with ``||G[:, ids] w - b|| <= tol ||b||`` (or max_elements).
    """
    n_el = G.shape[1]
    max_elements = n_el if max_elements is None else int(max_elements)
    bn = float(np.linalg.norm(b))
    active: list[int] = []
    w = np.zeros(0)
    r = b.copy()
    hist = []
    t0 = time.perf_counter()
    while len(active) < max_elements:
        res = float(np.linalg.norm(r)) / bn
        hist.append(res)
        if res <= tol:
            break
        score = G.T @ r
        if active:
            score[np.asarray(active)] = -np.inf
        j = int(np.argmax(score))
        if score[j] <= 0:
            break
        active.append(j)
        w_new, _ = sopt.nnls(G[:, active], b, maxiter=50 * len(active) + 100)
        keep = w_new > 0
        if not np.all(keep):
            active = [a for a, k in zip(active, keep, strict=False) if k]
            w_new = w_new[keep]
        w = w_new
        r = b - G[:, active] @ w
        if verbose and len(active) % 50 == 0:
            print(f"  ecsw: {len(active)} elements, residual {np.linalg.norm(r) / bn:.3e}", flush=True)
    info = {
        "n_elements": len(active),
        "residual": float(np.linalg.norm(r)) / bn,
        "history": hist,
        "seconds": time.perf_counter() - t0,
    }
    order = np.argsort(active)
    return np.asarray(active, dtype=np.int64)[order], np.asarray(w)[order], info
