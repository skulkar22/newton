# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp (CPU/GPU) total-Lagrangian tet4 assembler: energy, internal force and exact tangent.

:class:`WarpTetAssembler` is a device implementation of the numpy reference
``adaptive_rom.fe_nonlinear.NonlinearAssembler`` with the same definitions:

* DOF layout ``x_flat[3 * n + i]`` (coordinate ``i`` of node ``n``); total-Lagrangian with respect
  to the rest nodes ``X``; ``F = sum_a x_a g_a^T`` is constant per tet, with ``g_a`` the rest
  shape-function gradients (``g_1..g_3`` = rows of ``Dm^-1``, ``g_0 = -(g_1 + g_2 + g_3)``).
* Energies per unit rest volume (``lambda``, ``mu`` from Young's modulus and Poisson ratio):

  - ``"stvk"``: ``Psi = mu E:E + lambda/2 tr(E)^2``, ``E = (F^T F - I)/2``, ``P = F S``,
    ``S = lambda tr(E) I + 2 mu E``.
  - ``"neo_hookean"``: ``Psi = mu/2 (tr(F^T F) - 3) - mu ln J + lambda/2 (ln J)^2``,
    ``P = mu (F - F^-T) + lambda ln J F^-T``. This is the compressible (log-J) neo-Hookean model
    of the reference assembler, not the Smith et al. "stable" neo-Hookean model; inverted
    elements (``J <= 0``) raise :class:`FloatingPointError` as in the reference.

* ``f_int = dW/dx`` (``M a + f_int(x) = f_ext``), element force ``f_a = V P g_a``.
* ``K_T = d f_int / dx``, element block ``K_ab[i, k] = V A_ijkl g_aj g_bl`` with ``A = dP/dF``
  exact (material and geometric parts):

  - StVK: ``(g_a . S g_b) I + lambda (F g_a)(F g_b)^T + mu (g_a . g_b) F F^T
    + mu (F g_b)(F g_a)^T``.
  - neo-Hookean (``H = F^-T``): ``mu (g_a . g_b) I + (mu - lambda ln J) (H g_b)(H g_a)^T
    + lambda (H g_a)(H g_b)^T``.

* Consistent mass ``M_e = rho V / 20 (1 + delta_ab) I_3``, lumped mass = row sums of ``M``.

Precision: all kernel math and accumulation are float64. Positions may be supplied as float64
or float32 (numpy or Warp ``vec3f``); float32 input is widened to float64 before the kernel, so
the result is the float64 evaluation at the rounded positions (finite-difference and identity
checks need float64 input).

Global assembly: the 3x3-block sparsity pattern (node-pair adjacency, rows and columns sorted)
and the target block of every ``(tet, a, b)`` pair are built once on the host. Two scatter
strategies are available, selected by ``scatter`` (default ``"auto"``: ``"atomic"`` on the CPU
device, where Warp runs the threads of a launch serially so atomics are deterministic and the
single pass is fastest, ``"gather"`` on CUDA devices):

* ``"gather"``: one kernel writes element force and tangent blocks to per-element
  buffers; a second pass sums, per block and per node, its contributions in a fixed order. No
  atomics, bitwise deterministic run to run, and every block is written (no zeroing pass).
* ``"atomic"``: one kernel adds element blocks to the BSR values and node forces with float64
  atomic adds. Fewer passes and no per-element buffer, but the summation order (and therefore
  the last bits) can change between runs on a GPU.

This module depends only on numpy and Warp (SciPy is imported lazily by the CSR/CSC helpers and
the ``mass_matrix`` property).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp
import warp.sparse as wps

__all__ = ["WarpTetAssembler", "WarpTetMesh"]

MATERIALS = {"stvk": 0, "neo_hookean": 1}
SCATTER_MODES = ("auto", "gather", "atomic")


def _lame_parameters(young: float, poisson: float) -> tuple[float, float]:
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    return lam, mu


@dataclass
class WarpTetMesh:
    """Minimal rest tet mesh (the subset of ``adaptive_rom.fe_nonlinear.TetMesh`` used here)."""

    nodes: np.ndarray
    tets: np.ndarray
    density: float
    young: float
    poisson: float
    node_sets: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.nodes = np.ascontiguousarray(self.nodes, dtype=np.float64)
        self.tets = np.ascontiguousarray(self.tets, dtype=np.int64)

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def num_tets(self) -> int:
        return int(self.tets.shape[0])

    @property
    def lame(self) -> tuple[float, float]:
        return _lame_parameters(self.young, self.poisson)

    def tet_volumes(self) -> np.ndarray:
        X = self.nodes[self.tets]
        Dm = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=-1)
        return np.linalg.det(Dm) / 6.0

    def total_mass(self) -> float:
        return float(self.density * self.tet_volumes().sum())


# ----------------------------------------------------------------------------------------------
# Warp functions and kernels
# ----------------------------------------------------------------------------------------------


@wp.func
def _deformation_gradient(
    x: wp.array[wp.vec3d],
    conn: wp.array2d[wp.int32],
    t: int,
    G: wp.array2d[wp.vec3d],
    e: int,
) -> wp.mat33d:
    F = wp.outer(x[conn[t, 0]], G[e, 0])
    for a in range(1, 4):
        F = F + wp.outer(x[conn[t, a]], G[e, a])
    return F


@wp.func
def _constitutive(F: wp.mat33d, lam: wp.float64, mu: wp.float64, material: int):
    """Return (P, psi, A, c, J): A = S (StVK) or F^-T (neo-Hookean), c = ln J (neo-Hookean)."""
    half = wp.float64(0.5)
    I = wp.identity(n=3, dtype=wp.float64)
    J = wp.determinant(F)
    if material == 0:
        E = half * (wp.transpose(F) @ F - I)
        trE = wp.trace(E)
        S = lam * trE * I + wp.float64(2.0) * mu * E
        psi = mu * wp.ddot(E, E) + half * lam * trE * trE
        return F @ S, psi, S, wp.float64(0.0), J
    if J <= wp.float64(0.0):
        return F, wp.float64(0.0), I, wp.float64(0.0), J
    H = wp.transpose(wp.inverse(F))
    c = wp.log(J)
    P = mu * (F - H) + lam * c * H
    psi = half * mu * (wp.ddot(F, F) - wp.float64(3.0)) - mu * c + half * lam * c * c
    return P, psi, H, c, J


@wp.func
def _tangent_block(
    ga: wp.vec3d,
    gb: wp.vec3d,
    F: wp.mat33d,
    A: wp.mat33d,
    c: wp.float64,
    lam: wp.float64,
    mu: wp.float64,
    material: int,
) -> wp.mat33d:
    """3x3 block K_ab[i, k] = A_ijkl g_aj g_bl (per unit volume)."""
    I = wp.identity(n=3, dtype=wp.float64)
    gg = wp.dot(ga, gb)
    if material == 0:
        Fga = F @ ga
        Fgb = F @ gb
        return (
            wp.dot(ga, A @ gb) * I
            + lam * wp.outer(Fga, Fgb)
            + mu * gg * (F @ wp.transpose(F))
            + mu * wp.outer(Fgb, Fga)
        )
    Hga = A @ ga
    Hgb = A @ gb
    return mu * gg * I + (mu - lam * c) * wp.outer(Hgb, Hga) + lam * wp.outer(Hga, Hgb)


@wp.kernel
def _element_terms_kernel(
    x: wp.array[wp.vec3d],
    conn: wp.array2d[wp.int32],
    elem_ids: wp.array[wp.int32],
    G: wp.array2d[wp.vec3d],
    vol: wp.array[wp.float64],
    lam: wp.float64,
    mu: wp.float64,
    material: int,
    need_tangent: int,
    We: wp.array[wp.float64],
    fe: wp.array[wp.vec3d],
    Ke: wp.array[wp.mat33d],
    inverted: wp.array[wp.int32],
):
    """Element energy, force (fe[4 t + a]) and tangent blocks (Ke[16 t + 4 a + b]) of tet t."""
    t = wp.tid()
    e = elem_ids[t]
    V = vol[e]
    F = _deformation_gradient(x, conn, t, G, e)
    P, psi, A, c, J = _constitutive(F, lam, mu, material)
    if material == 1 and J <= wp.float64(0.0):
        wp.atomic_max(inverted, 0, 1)
    We[t] = V * psi
    for a in range(4):
        fe[4 * t + a] = V * (P @ G[e, a])
    if need_tangent != 0:
        for a in range(4):
            ga = G[e, a]
            for b in range(4):
                Ke[16 * t + 4 * a + b] = V * _tangent_block(ga, G[e, b], F, A, c, lam, mu, material)


@wp.kernel
def _atomic_assembly_kernel(
    x: wp.array[wp.vec3d],
    conn: wp.array2d[wp.int32],
    G: wp.array2d[wp.vec3d],
    vol: wp.array[wp.float64],
    block_index: wp.array2d[wp.int32],
    lam: wp.float64,
    mu: wp.float64,
    material: int,
    need_tangent: int,
    We: wp.array[wp.float64],
    f: wp.array[wp.vec3d],
    K_values: wp.array[wp.mat33d],
    inverted: wp.array[wp.int32],
):
    """Element terms of tet e added directly to the global force and BSR values with atomics."""
    e = wp.tid()
    V = vol[e]
    F = _deformation_gradient(x, conn, e, G, e)
    P, psi, A, c, J = _constitutive(F, lam, mu, material)
    if material == 1 and J <= wp.float64(0.0):
        wp.atomic_max(inverted, 0, 1)
    We[e] = V * psi
    for a in range(4):
        wp.atomic_add(f, conn[e, a], V * (P @ G[e, a]))
    if need_tangent != 0:
        for a in range(4):
            ga = G[e, a]
            for b in range(4):
                blk = V * _tangent_block(ga, G[e, b], F, A, c, lam, mu, material)
                wp.atomic_add(K_values, block_index[e, 4 * a + b], blk)


@wp.kernel
def _gather_vec3d_kernel(
    src: wp.array[wp.vec3d],
    ptr: wp.array[wp.int32],
    ids: wp.array[wp.int32],
    dst: wp.array[wp.vec3d],
):
    i = wp.tid()
    s = wp.vec3d()
    for j in range(ptr[i], ptr[i + 1]):
        s = s + src[ids[j]]
    dst[i] = s


@wp.kernel
def _gather_mat33d_kernel(
    src: wp.array[wp.mat33d],
    ptr: wp.array[wp.int32],
    ids: wp.array[wp.int32],
    dst: wp.array[wp.mat33d],
):
    i = wp.tid()
    s = wp.mat33d()
    for j in range(ptr[i], ptr[i + 1]):
        s = s + src[ids[j]]
    dst[i] = s


@wp.kernel
def _widen_vec3f_kernel(src: wp.array[wp.vec3f], dst: wp.array[wp.vec3d]):
    i = wp.tid()
    v = src[i]
    dst[i] = wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


@wp.func
def _polar_rotation(F: wp.mat33d) -> wp.mat33d:
    """Rotation factor R of F = R U (det R = +1).

    det F > 0: scaled Newton iteration X <- (gamma X + X^-T / gamma) / 2 with Higham's Frobenius
    scaling, converged to float64 roundoff. det F <= 0: SVD with the column of the smallest
    singular value flipped (the reference's convention).
    """
    if wp.determinant(F) > wp.float64(0.0):
        X = F
        for _it in range(60):
            Xi = wp.inverse(X)
            gamma = wp.sqrt(wp.sqrt(wp.ddot(Xi, Xi) / wp.ddot(X, X)))
            Xn = wp.float64(0.5) * (gamma * X + wp.transpose(Xi) / gamma)
            d = Xn - X
            X = Xn
            if wp.ddot(d, d) < wp.float64(1.0e-30):
                break
        return X
    U = wp.mat33d()
    sigma = wp.vec3d()
    Vm = wp.mat33d()
    wp.svd3(F, U, sigma, Vm)
    k = int(0)
    if sigma[1] < sigma[k]:
        k = 1
    if sigma[2] < sigma[k]:
        k = 2
    D = wp.identity(n=3, dtype=wp.float64)
    D[k, k] = wp.float64(-1.0)
    R = U @ D @ wp.transpose(Vm)
    if wp.determinant(R) < wp.float64(0.0):  # det U V^T was already +1 (det F == 0 corner case)
        R = U @ wp.transpose(Vm)
    return R


@wp.kernel
def _rotation_kernel(
    x: wp.array[wp.vec3d],
    x_ref: wp.array[wp.vec3d],
    has_ref: int,
    conn: wp.array2d[wp.int32],
    elem_ids: wp.array[wp.int32],
    G: wp.array2d[wp.vec3d],
    angle: wp.array[wp.float64],
):
    t = wp.tid()
    e = elem_ids[t]
    F = _deformation_gradient(x, conn, e, G, e)
    if has_ref != 0:
        F = F @ wp.inverse(_deformation_gradient(x_ref, conn, e, G, e))
    R = _polar_rotation(F)
    half = wp.float64(0.5)
    w = wp.vec3d(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])
    angle[t] = wp.atan2(half * wp.length(w), half * (wp.trace(R) - wp.float64(1.0)))


# ----------------------------------------------------------------------------------------------
# Host-side pattern helpers
# ----------------------------------------------------------------------------------------------


def _host_copy(a: wp.array) -> np.ndarray:
    """Host numpy copy of a Warp array (``numpy()`` aliases device memory on the CPU device)."""
    out = a.numpy()
    return out.copy() if a.device.is_cpu else out


def _csr_offsets(keys: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.bincount(keys, minlength=n))]).astype(np.int32)


class WarpTetAssembler:
    """Warp tet4 assembler for StVK / compressible neo-Hookean (see the module docstring).

    Args:
        mesh_nodes: Rest node positions ``X``, shape ``(N, 3)``.
        tets: Positively oriented tetrahedra, shape ``(E, 4)``.
        density: Mass density [kg/m^3].
        young: Young's modulus [Pa].
        poisson: Poisson ratio [-].
        material: ``"stvk"`` or ``"neo_hookean"``.
        device: Warp device for kernels and device arrays (``None``: Warp's default device).
        scatter: Global assembly strategy: ``"gather"`` (deterministic on every device),
            ``"atomic"`` (deterministic only on the CPU device) or ``"auto"`` (atomic on CPU,
            gather on CUDA).

    Attributes:
        mesh: Rest mesh (:class:`WarpTetMesh`, or the object given to :meth:`from_mesh`).
        lam, mu: Lame parameters. ``vol`` (E,) rest volumes, ``G`` (E, 4, 3) shape gradients.
        X_flat: Rest positions, (3N,) float64.
        lumped_mass: (3N,) row sums of the consistent mass matrix.
        mass_matrix: Consistent mass, SciPy CSR (built on first access);
            :attr:`mass_matrix_bsr` holds it as a device :class:`warp.sparse.BsrMatrix`.
        nnz_blocks, block_rows, block_columns, block_offsets: The 3x3 block pattern (host).
        block_index: (E, 16) block index of the element block ``(a, b)`` at ``4 a + b``.
    """

    def __init__(
        self,
        mesh_nodes,
        tets,
        density: float,
        young: float,
        poisson: float,
        material: str = "stvk",
        device: Any = None,
        *,
        scatter: str = "auto",
        _mesh: Any = None,
    ):
        if material not in MATERIALS:
            raise ValueError(f"unknown material {material!r}")
        if scatter not in SCATTER_MODES:
            raise ValueError(f"unknown scatter mode {scatter!r}; expected one of {SCATTER_MODES}")
        self.mesh = _mesh if _mesh is not None else WarpTetMesh(mesh_nodes, tets, density, young, poisson)
        nodes = np.ascontiguousarray(mesh_nodes, dtype=np.float64)
        tets = np.ascontiguousarray(tets, dtype=np.int64)
        self.material = material
        self.device = wp.get_device(device)
        if scatter == "auto":
            scatter = "atomic" if self.device.is_cpu else "gather"
        self.scatter = scatter
        self.density = float(density)
        self.lam, self.mu = _lame_parameters(float(young), float(poisson))
        self.n_nodes = int(nodes.shape[0])
        self.n_dof = 3 * self.n_nodes
        self.n_tets = int(tets.shape[0])
        E, N = self.n_tets, self.n_nodes

        # rest geometry, computed exactly as the numpy reference
        X = nodes[tets]
        Dm = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=-1)
        self.vol = np.linalg.det(Dm) / 6.0
        if np.any(self.vol <= 0):
            raise ValueError("mesh has non-positive tet volumes")
        Dm_inv = np.linalg.inv(Dm)
        G = np.empty((E, 4, 3))
        G[:, 1:, :] = Dm_inv
        G[:, 0, :] = -Dm_inv.sum(axis=1)
        self.G = G
        self.X_flat = nodes.reshape(-1).copy()

        # 3x3 block pattern: node pairs (n_a, n_b) of every tet, sorted by (row, column)
        rows = np.repeat(tets, 4, axis=1)  # (E,16): tets[:, a] at 4a+b
        cols = np.tile(tets, (1, 4))  # (E,16): tets[:, b] at 4a+b
        ukey, inv = np.unique(rows.astype(np.int64) * N + cols, return_inverse=True)
        self.nnz_blocks = int(ukey.size)
        self.block_rows = (ukey // N).astype(np.int32)
        self.block_columns = (ukey % N).astype(np.int32)
        self.block_offsets = _csr_offsets(self.block_rows, N)
        self.block_index = inv.reshape(E, 16).astype(np.int32)
        # gather lists: element contributions per block / per node, in ascending element order
        flat_blocks = self.block_index.ravel()
        self._block_ptr = _csr_offsets(flat_blocks, self.nnz_blocks)
        self._block_ids = np.argsort(flat_blocks, kind="stable").astype(np.int32)
        flat_nodes = tets.ravel()
        self._node_ptr = _csr_offsets(flat_nodes, N)
        self._node_ids = np.argsort(flat_nodes, kind="stable").astype(np.int32)

        # consistent and lumped mass (reference definition), assembled on the block pattern
        me = (np.ones((4, 4)) + np.eye(4)).reshape(-1)  # (16,) at 4a+b
        mass_scalar = np.zeros(self.nnz_blocks)
        np.add.at(mass_scalar, flat_blocks, ((self.density * self.vol / 20.0)[:, None] * me[None, :]).ravel())
        self._mass_block_scalar = mass_scalar  # block = scalar * I3
        row_sum = np.bincount(self.block_rows, weights=mass_scalar, minlength=N)
        self.lumped_mass = np.repeat(row_sum, 3)
        self._mass_csr = None
        self._csr_cache = None
        self._csc_cache = None

        # device data
        dev = self.device
        self._tets_wp = wp.array(tets.astype(np.int32), dtype=wp.int32, device=dev)
        self._G_wp = wp.array(G, dtype=wp.vec3d, device=dev)
        self._vol_wp = wp.array(self.vol, dtype=wp.float64, device=dev)
        self._all_ids_wp = wp.array(np.arange(E, dtype=np.int32), dtype=wp.int32, device=dev)
        self._block_index_wp = wp.array(self.block_index, dtype=wp.int32, device=dev)
        self._block_ptr_wp = wp.array(self._block_ptr, dtype=wp.int32, device=dev)
        self._block_ids_wp = wp.array(self._block_ids, dtype=wp.int32, device=dev)
        self._node_ptr_wp = wp.array(self._node_ptr, dtype=wp.int32, device=dev)
        self._node_ids_wp = wp.array(self._node_ids, dtype=wp.int32, device=dev)
        self._x_wp = wp.zeros(N, dtype=wp.vec3d, device=dev)
        self._xref_wp = wp.zeros(N, dtype=wp.vec3d, device=dev)
        self._We_wp = wp.zeros(E, dtype=wp.float64, device=dev)
        self._f_wp = wp.zeros(N, dtype=wp.vec3d, device=dev)
        self._inverted_wp = wp.zeros(1, dtype=wp.int32, device=dev)
        if scatter == "gather":
            self._fe_wp = wp.zeros(4 * E, dtype=wp.vec3d, device=dev)
            self._Ke_wp = wp.zeros(16 * E, dtype=wp.mat33d, device=dev)
        self._K = self._new_bsr()
        self._K0 = None
        self._mass_bsr = None

    @classmethod
    def from_mesh(cls, mesh, material: str = "stvk", device: Any = None, **kwargs) -> WarpTetAssembler:
        """Build from a ``TetMesh``-like object (``nodes``, ``tets``, ``density``, ``young``, ``poisson``)."""
        return cls(
            mesh.nodes, mesh.tets, mesh.density, mesh.young, mesh.poisson, material, device, _mesh=mesh, **kwargs
        )

    # -- sparse helpers --------------------------------------------------------------------
    def _new_bsr(self) -> wps.BsrMatrix:
        """A BSR matrix with the cached block pattern (values zero)."""
        dev = self.device
        K = wps.bsr_zeros(self.n_nodes, self.n_nodes, block_type=wp.mat33d, device=dev)
        wps.bsr_set_from_triplets(
            K,
            wp.array(self.block_rows, dtype=wp.int32, device=dev),
            wp.array(self.block_columns, dtype=wp.int32, device=dev),
            wp.zeros(self.nnz_blocks, dtype=wp.mat33d, device=dev),
            prune_numerical_zeros=False,
        )
        offsets = K.offsets.numpy()[: self.n_nodes + 1]
        columns = K.columns.numpy()[: self.nnz_blocks]
        if not (np.array_equal(offsets, self.block_offsets) and np.array_equal(columns, self.block_columns)):
            raise RuntimeError("warp.sparse produced an unexpected block pattern")
        return K

    def _values_view(self, K: wps.BsrMatrix):
        return K.values[: self.nnz_blocks]

    def _scalar_permutation(self, fmt: str):
        """Cached (indptr, indices, perm) so that CSR/CSC data = block_values.ravel()[perm]."""
        cache = self._csr_cache if fmt == "csr" else self._csc_cache
        if cache is None:
            from scipy import sparse

            ids = np.arange(1, 9 * self.nnz_blocks + 1, dtype=np.float64).reshape(-1, 3, 3)
            B = sparse.bsr_matrix((ids, self.block_columns, self.block_offsets), shape=(self.n_dof, self.n_dof))
            S = B.tocsr() if fmt == "csr" else B.tocsc()
            S.sort_indices()
            if S.nnz != 9 * self.nnz_blocks:
                raise RuntimeError("scalar pattern lost entries")
            cache = (S.indptr.copy(), S.indices.copy(), S.data.astype(np.int64) - 1)
            if fmt == "csr":
                self._csr_cache = cache
            else:
                self._csc_cache = cache
        return cache

    def tangent_values_device(self) -> wp.array:
        """Device BSR values ``(nnz_blocks,)`` ``mat33d`` of the last assembled tangent (no copy).

        The internal buffer of :meth:`evaluate` / :meth:`evaluate_device`, on the pattern
        :attr:`block_offsets` / :attr:`block_columns`; the next assembly overwrites it.
        """
        return self._values_view(self._K)

    def block_values_numpy(self, K: wps.BsrMatrix) -> np.ndarray:
        """Device-to-host copy of the BSR values, shape (nnz_blocks, 3, 3) float64."""
        return _host_copy(self._values_view(K))

    def to_scipy_csr(self, K: wps.BsrMatrix):
        """SciPy CSR copy of a matrix on this assembler's pattern (one value copy; pattern cached)."""
        from scipy import sparse

        indptr, indices, perm = self._scalar_permutation("csr")
        data = self.block_values_numpy(K).reshape(-1)[perm]
        return sparse.csr_matrix((data, indices, indptr), shape=(self.n_dof, self.n_dof))

    def to_scipy_csc(self, K: wps.BsrMatrix):
        """SciPy CSC copy of a matrix on this assembler's pattern (one value copy; pattern cached)."""
        return self.scipy_from_block_values(self.block_values_numpy(K), "csc")

    def scipy_from_block_values(self, values: np.ndarray, fmt: str = "csc"):
        """SciPy CSR/CSC matrix from host block values ``(nnz_blocks, 3, 3)`` (or flat) on the cached pattern.

        For callers that gathered the BSR values themselves (e.g. packed with other data in one
        device-to-host copy).
        """
        from scipy import sparse

        indptr, indices, perm = self._scalar_permutation(fmt)
        data = np.asarray(values, dtype=np.float64).reshape(-1)[perm]
        cls = sparse.csc_matrix if fmt == "csc" else sparse.csr_matrix
        return cls((data, indices, indptr), shape=(self.n_dof, self.n_dof))

    def _convert(self, K: wps.BsrMatrix, fmt: str):
        if fmt == "bsr":
            return K
        if fmt == "csr":
            return self.to_scipy_csr(K)
        if fmt == "csc":
            return self.to_scipy_csc(K)
        raise ValueError(f"unknown tangent_format {fmt!r}; expected 'bsr', 'csr' or 'csc'")

    @property
    def mass_matrix(self):
        """Consistent mass matrix, SciPy CSR (reference definition)."""
        if self._mass_csr is None:
            self._mass_csr = self.to_scipy_csr(self.mass_matrix_bsr)
        return self._mass_csr

    @property
    def mass_matrix_bsr(self) -> wps.BsrMatrix:
        """Consistent mass matrix as a device BSR matrix (3x3 blocks ``m I_3``)."""
        if self._mass_bsr is None:
            M = self._new_bsr()
            blocks = self._mass_block_scalar[:, None, None] * np.eye(3)[None]
            self._values_view(M).assign(wp.array(blocks, dtype=wp.mat33d, device=self.device))
            self._mass_bsr = M
        return self._mass_bsr

    # -- input handling --------------------------------------------------------------------
    def _load_positions(self, x, dst: wp.array, n: int) -> wp.array:
        """Return float64 positions (n vec3d) on the device; copies into ``dst`` unless already vec3d."""
        if isinstance(x, wp.array):
            if x.dtype == wp.vec3d and x.device == self.device and x.shape == (n,):
                return x
            if x.dtype == wp.vec3f and x.shape == (n,):
                src = x if x.device == self.device else x.to(self.device)
                wp.launch(_widen_vec3f_kernel, dim=n, inputs=[src], outputs=[dst], device=self.device)
                return dst
            x = x.numpy()
        arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(n, 3))
        dst.assign(arr)
        return dst

    # -- energy / force / tangent ----------------------------------------------------------
    def evaluate(
        self,
        x_flat,
        *,
        need_tangent: bool = True,
        tangent_format: str = "bsr",
        device_output: bool = False,
        copy_tangent: bool = True,
    ):
        """Return ``(W, f_int, K_T)`` at current positions (``K_T`` is ``None`` without tangent).

        Args:
            x_flat: Positions, (3N,) or (N, 3) numpy (float64/float32) or a Warp ``vec3d``/``vec3f``
                array of N entries (device-resident ``vec3d`` input is used without a copy).
            need_tangent: Assemble ``K_T``.
            tangent_format: ``"bsr"`` (:class:`warp.sparse.BsrMatrix` on the device),
                ``"csr"`` or ``"csc"`` (SciPy, one device-to-host copy of the values).
            device_output: Return ``f_int`` as a Warp ``vec3d`` array (N,) on the device (an internal
                buffer overwritten by the next call) instead of a (3N,) numpy array.
            copy_tangent: For ``"bsr"``, return a copy; ``False`` returns the internal matrix, which
                the next call overwrites.

        Raises:
            FloatingPointError: neo-Hookean with an inverted element (``det F <= 0``).
        """
        self.evaluate_device(x_flat, need_tangent=need_tangent)
        self._check_inverted()
        W = float(np.sum(self._We_wp.numpy()))
        f_int = self._f_wp if device_output else _host_copy(self._f_wp).reshape(-1)
        if not need_tangent:
            return W, f_int, None
        if tangent_format == "bsr":
            K = wps.bsr_copy(self._K) if copy_tangent else self._K
        else:
            K = self._convert(self._K, tangent_format)
        return W, f_int, K

    def evaluate_device(self, x: wp.array, *, need_tangent: bool = True):
        """Assemble at device positions without any host synchronization.

        Args:
            x: Positions, Warp ``vec3d`` array (N,) on :attr:`device`.
            need_tangent: Assemble ``K_T``.

        Returns:
            ``(element_energy (E,) float64, f_int (N,) vec3d, K_T BsrMatrix or None, inverted (1,) int32)``:
            internal buffers overwritten by the next call. The total energy is the sum of
            ``element_energy`` and ``inverted[0] != 0`` flags inverted elements (neo-Hookean); both stay on
            the device (no host reduction, no check).
        """
        E, N, dev = self.n_tets, self.n_nodes, self.device
        x = self._load_positions(x, self._x_wp, N)
        self._inverted_wp.zero_()
        lam, mu, mat = wp.float64(self.lam), wp.float64(self.mu), MATERIALS[self.material]
        tan = int(bool(need_tangent))
        Kv = self._values_view(self._K)
        if self.scatter == "gather":
            wp.launch(
                _element_terms_kernel,
                dim=E,
                inputs=[x, self._tets_wp, self._all_ids_wp, self._G_wp, self._vol_wp, lam, mu, mat, tan],
                outputs=[self._We_wp, self._fe_wp, self._Ke_wp, self._inverted_wp],
                device=dev,
            )
            wp.launch(
                _gather_vec3d_kernel,
                dim=N,
                inputs=[self._fe_wp, self._node_ptr_wp, self._node_ids_wp],
                outputs=[self._f_wp],
                device=dev,
            )
            if need_tangent:
                wp.launch(
                    _gather_mat33d_kernel,
                    dim=self.nnz_blocks,
                    inputs=[self._Ke_wp, self._block_ptr_wp, self._block_ids_wp],
                    outputs=[Kv],
                    device=dev,
                )
        else:
            self._f_wp.zero_()
            if need_tangent:
                Kv.zero_()
            wp.launch(
                _atomic_assembly_kernel,
                dim=E,
                inputs=[x, self._tets_wp, self._G_wp, self._vol_wp, self._block_index_wp, lam, mu, mat, tan],
                outputs=[self._We_wp, self._f_wp, Kv, self._inverted_wp],
                device=dev,
            )
        return self._We_wp, self._f_wp, (Kv if need_tangent else None), self._inverted_wp

    def _check_inverted(self):
        if self.material == "neo_hookean" and int(self._inverted_wp.numpy()[0]) != 0:
            raise FloatingPointError("inverted element(s): det F <= 0")

    def element_terms(self, x_elem, element_ids=None, *, need_tangent: bool = True, device_output: bool = False):
        """Per-element energy, force and tangent on an element subset (no global assembly; for ECSW).

        Args:
            x_elem: Current positions of the element nodes, (E_sel, 4, 3) numpy, or a Warp ``vec3d``
                array of ``4 * E_sel`` entries (node ``a`` of element ``t`` at ``4 t + a``).
            element_ids: Indices of the selected elements (``None``: all, in mesh order).
            device_output: Return Warp arrays ``We`` (E_sel,), ``fe`` (4 E_sel,) ``vec3d`` and ``Ke``
                (16 E_sel,) ``mat33d`` (block ``(a, b)`` at ``16 t + 4 a + b``) instead of numpy.
        Returns:
            ``(W_e (E_sel,), f_e (E_sel, 4, 3), K_e (E_sel, 4, 3, 4, 3) or None)``, the layout of the
            reference ``NonlinearAssembler.element_terms``.
        """
        dev = self.device
        ids = np.arange(self.n_tets) if element_ids is None else np.asarray(element_ids).reshape(-1)
        n_sel = int(ids.size)
        if n_sel == 0:
            empty = np.zeros((0,))
            return empty, np.zeros((0, 4, 3)), (np.zeros((0, 4, 3, 4, 3)) if need_tangent else None)
        x = self._load_positions(x_elem, wp.zeros(4 * n_sel, dtype=wp.vec3d, device=dev), 4 * n_sel)
        conn = wp.array(np.arange(4 * n_sel, dtype=np.int32).reshape(n_sel, 4), dtype=wp.int32, device=dev)
        ids_wp = wp.array(ids.astype(np.int32), dtype=wp.int32, device=dev)
        We = wp.zeros(n_sel, dtype=wp.float64, device=dev)
        fe = wp.zeros(4 * n_sel, dtype=wp.vec3d, device=dev)
        Ke = wp.zeros(16 * n_sel if need_tangent else 1, dtype=wp.mat33d, device=dev)
        self._inverted_wp.zero_()
        wp.launch(
            _element_terms_kernel,
            dim=n_sel,
            inputs=[
                x,
                conn,
                ids_wp,
                self._G_wp,
                self._vol_wp,
                wp.float64(self.lam),
                wp.float64(self.mu),
                MATERIALS[self.material],
                int(bool(need_tangent)),
            ],
            outputs=[We, fe, Ke, self._inverted_wp],
            device=dev,
        )
        self._check_inverted()
        if device_output:
            return We, fe, (Ke if need_tangent else None)
        K = None
        if need_tangent:
            K = Ke.numpy().reshape(n_sel, 4, 4, 3, 3).transpose(0, 1, 3, 2, 4).copy()
        return We.numpy(), fe.numpy().reshape(n_sel, 4, 3), K

    def linear_stiffness(self, tangent_format: str = "bsr"):
        """K_T at the rest configuration (linear elastic stiffness with lambda, mu); cached."""
        if self._K0 is None:
            self._K0 = self.evaluate(self.X_flat, need_tangent=True)[2]
        return self._convert(self._K0, tangent_format)

    # -- per-element kinematic measures ----------------------------------------------------
    def element_rotations(
        self, x_flat, element_ids=None, reference_flat=None, *, device_output: bool = False, out: wp.array = None
    ):
        """Rotation angle [rad] in [0, pi] of the polar factor R of F = R U per element.

        F is relative to the rest nodes; with ``reference_flat`` the relative gradient
        ``F(x) F(reference)^-1`` is used (rotation since that reference). The angle is
        ``atan2(|axial(R)|, (tr R - 1) / 2)``, accurate also near 0 and pi.

        ``element_ids`` may be a Warp ``int32`` array on :attr:`device` (no upload); ``out`` an optional
        preallocated ``float64`` output of the sample size (no allocation). Device-resident ``vec3d``
        positions/reference and both options make the call free of host copies (per-step triggers).
        """
        dev, N = self.device, self.n_nodes
        x = self._load_positions(x_flat, self._x_wp, N)
        has_ref = reference_flat is not None
        xr = self._load_positions(reference_flat, self._xref_wp, N) if has_ref else x
        if isinstance(element_ids, wp.array):
            ids_wp = element_ids
            n_ids = int(element_ids.shape[0])
        else:
            ids = np.arange(self.n_tets) if element_ids is None else np.asarray(element_ids).reshape(-1)
            n_ids = int(ids.size)
            ids_wp = (
                self._all_ids_wp if element_ids is None else wp.array(ids.astype(np.int32), dtype=wp.int32, device=dev)
            )
        if out is None:
            out = wp.zeros(n_ids, dtype=wp.float64, device=dev)
        if n_ids:
            wp.launch(
                _rotation_kernel,
                dim=n_ids,
                inputs=[x, xr, int(has_ref), self._tets_wp, ids_wp, self._G_wp],
                outputs=[out],
                device=dev,
            )
        return out if device_output else out.numpy()
