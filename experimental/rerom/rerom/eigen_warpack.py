# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: TID253
"""Warpack eigensolver backends of the re-ROM seam (optional, experimental).

`Warpack <https://github.com/alecjacobson/warpack>`_ is Alec Jacobson's device-resident sparse eigensolver
library for NVIDIA Warp (MIT licence). This module drives two of its solvers on the re-ROM pencil
``(K_T, M)`` restricted to the free DOFs and, for free bodies, to the M-orthogonal complement of the rigid
motions ``Y``:

- :class:`WarpackKrylovBackend`: ``warpack.GeneralizedKrylovSchur`` in shift-invert mode (M inner product,
  thick restart), cold start. The primary path.
- :class:`WarpackLOBPCGBackend`: ``warpack.BlockLOBPCG`` warm-started from the previous basis, with the
  shift-invert inverse as its preconditioner and the rigid motions as constraints.

Both satisfy :class:`newton.ModalEigensolverBackend` and accept the update's shared factorization
(:class:`WarpackRestrictedSolver`) through ``solve(..., factorization=...)``.

Sparse factorization. Warpack's own ``CuDSSInverse`` needs the ``warp-cudss`` binding, which is not
installed here. :class:`CudssFactor` provides the same Warpack inverse protocol (``n``, ``device``,
``width``, ``apply(rows, out)``) on cuDSS 0.8 through ``nvmath-python``'s low-level bindings. It works
directly on Warp device arrays, with the symbolic analysis once per pattern, numeric refactorization per
update, arbitrary RHS blocks and the ``L D L^T`` inertia. It is always ``MatrixType.SYMMETRIC``: the cuDSS
SPD type can hang on an indefinite ``K - sigma M``.

Rigid deflation. Warpack's ``MassOrthogonalComplementInverse`` and LOBPCG ``constraints`` assume the
excluded space is invariant under ``K``. A pre-stressed ``K_T`` does not annihilate the rigid rotations, so
the pencil is posed with the projected stiffness ``K_r = P^T K P`` (``P = I - Y B^T``, ``B = M Y``), for which
``Y`` is exactly invariant (``K_r Y = 0``). Its shift-invert inverse on the complement is the exact restricted
inverse ``x = A^-1 b' - A^-1 B (B^T A^-1 B)^-1 B^T A^-1 b'``, ``A = K - sigma M``, ``b' = b - B Y^T b``, the
same algebra as :class:`rerom.eigen.ShiftedRestrictedSolver`. The restricted eigenpairs of ``(K_r, M)`` are
those of ``(I - B Y^T)(K v - lambda M v) = 0``.

Device hand-off. :class:`WarpackContext` maps the 3x3 BSR tangent of a
:class:`rerom.assembly_warp.WarpTetAssembler` (device values, before any host copy) to the symmetrized
free-node BSR used by Warpack and to the scalar CSR values of ``K - sigma M`` used by cuDSS, with all index
maps built once per pattern. The symmetrization ``0.5 (K + K^T)`` matches the host path bit for bit.

Sturm count. From the cuDSS ``L D L^T`` inertia at a shift ``tau`` in a gap above the returned pairs, with the
Haynsworth correction ``neg(A) + neg(-B^T A^-1 B) - d`` for the restriction, reported like
:class:`rerom.eigen.RestrictedArpackBackend` (``inertia_count = below + d``).

Requires ``warpack`` and ``nvmath-python`` with cuDSS (``nvidia-cudss-cu13``); both are imported lazily.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import numpy as np
import scipy.sparse as sp
import warp as wp

import newton

from .eigen import ShiftedRestrictedSolver

__all__ = [
    "CudssFactor",
    "WarpackContext",
    "WarpackKrylovBackend",
    "WarpackLOBPCGBackend",
    "WarpackRestrictedSolver",
]

_TILE = 256


def _require():
    """Import Warpack and the cuDSS bindings (raise an ImportError with install hints)."""
    try:
        import warpack  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "rerom.eigen_warpack needs Warpack (Alec Jacobson, MIT licence, https://github.com/alecjacobson/warpack); "
            "install it from a checkout with `pip install <warpack checkout>`"
        ) from exc
    try:
        from nvmath.bindings import cudss  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "rerom.eigen_warpack needs cuDSS through nvmath-python: `pip install nvmath-python[cu13]` "
            "(brings nvidia-cudss-cu13 0.8)"
        ) from exc
    return warpack, cudss


# ----------------------------------------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------------------------------------
@wp.kernel
def _symmetrize_blocks(
    src: wp.array[wp.mat33d],
    a: wp.array[wp.int32],
    b: wp.array[wp.int32],
    dst: wp.array[wp.mat33d],
):
    i = wp.tid()
    dst[i] = (src[a[i]] + wp.transpose(src[b[i]])) * wp.float64(0.5)


@wp.kernel
def _shifted_values(
    k: wp.array[wp.mat33d],
    m: wp.array[wp.mat33d],
    flat: wp.array[wp.int32],
    sigma: wp.float64,
    out: wp.array[wp.float64],
):
    e = wp.tid()
    f = flat[e]
    blk = f // 9
    rc = f - 9 * blk
    r = rc // 3
    c = rc - 3 * r
    out[e] = k[blk][r, c] - sigma * m[blk][r, c]


@wp.kernel
def _abs_row_sums(k: wp.array[wp.mat33d], block_row: wp.array[wp.int32], out: wp.array[wp.float64]):
    i = wp.tid()
    blk = k[i]
    r0 = 3 * block_row[i]
    for r in range(3):
        wp.atomic_add(out, r0 + r, wp.abs(blk[r, 0]) + wp.abs(blk[r, 1]) + wp.abs(blk[r, 2]))


@wp.kernel
def _row_dots(a: wp.array2d[wp.float64], x: wp.array2d[wp.float64], out: wp.array2d[wp.float64]):
    """out[i, j] += a[i] . x[j] (out zeroed by the caller)."""
    i, j, chunk = wp.tid()
    ta = wp.tile_load(a[i], shape=_TILE, offset=chunk * _TILE)
    tx = wp.tile_load(x[j], shape=_TILE, offset=chunk * _TILE)
    wp.tile_atomic_add(out[i], wp.tile_sum(wp.tile_map(wp.mul, ta, tx)), offset=j)


@wp.kernel
def _sub_combination(u: wp.array2d[wp.float64], c: wp.array2d[wp.float64], y: wp.array2d[wp.float64]):
    """y[r] -= sum_k c[k, r] u[k]."""
    r, i = wp.tid()
    s = y[r, i]
    for k in range(u.shape[0]):
        s -= u[k, i] * c[k, r]
    y[r, i] = s


@wp.kernel
def _small_matmul(a: wp.array2d[wp.float64], c: wp.array2d[wp.float64], out: wp.array2d[wp.float64]):
    i, j = wp.tid()
    s = wp.float64(0.0)
    for k in range(a.shape[1]):
        s += a[i, k] * c[k, j]
    out[i, j] = s


@wp.kernel
def _residual_rows(
    kx: wp.array2d[wp.float64],
    mx: wp.array2d[wp.float64],
    lam: wp.array[wp.float64],
    out: wp.array2d[wp.float64],
):
    r, i = wp.tid()
    out[r, i] = kx[r, i] - lam[r] * mx[r, i]


@wp.kernel
def _row_norm2(x: wp.array2d[wp.float64], out: wp.array[wp.float64]):
    r, chunk = wp.tid()
    t = wp.tile_load(x[r], shape=_TILE, offset=chunk * _TILE)
    wp.tile_atomic_add(out, wp.tile_sum(wp.tile_map(wp.mul, t, t)), offset=r)


def _launch_dots(a, x, out, device):
    out.zero_()
    wp.launch_tiled(
        _row_dots,
        dim=(a.shape[0], x.shape[0], (a.shape[1] + _TILE - 1) // _TILE),
        inputs=[a, x, out],
        block_dim=128,
        device=device,
    )


# ----------------------------------------------------------------------------------------------------------
# cuDSS on Warp arrays (Warpack inverse protocol)
# ----------------------------------------------------------------------------------------------------------
_R64F = 1  # CUDA_R_64F
_R32I = 10  # CUDA_R_32I


def _threading_lib() -> str | None:
    import glob  # noqa: PLC0415
    import os  # noqa: PLC0415

    try:
        import nvidia  # noqa: PLC0415
    except ImportError:
        return None
    for base in nvidia.__path__:
        for sub in ("cu13", "cu12"):
            hits = sorted(glob.glob(os.path.join(base, sub, "lib", "libcudss_mtlayer_gomp.so*")))
            if hits:
                return hits[0]
    return None


class _CudssHandles:
    handles: ClassVar[dict[str, int]] = {}

    @classmethod
    def get(cls, device) -> int:
        key = str(device)
        if key not in cls.handles:
            _, cudss = _require()
            with wp.ScopedDevice(device):
                h = cudss.create()
                lib = _threading_lib()
                if lib is not None:
                    cudss.set_threading_layer(h, lib)
            cls.handles[key] = h
        return cls.handles[key]


class CudssFactor:
    """cuDSS ``L D L^T`` (``MatrixType.SYMMETRIC``) factorization of a scalar CSR matrix held in Warp arrays.

    The matrix descriptor points at ``indptr`` / ``indices`` / ``values`` (device, int32 / int32 / float64);
    the caller writes new ``values`` and calls :meth:`factorize` (numeric phase only, the symbolic analysis of
    the first call is kept). :meth:`apply` solves for any number of contiguous FP64 rows (``(w, n)`` row-major
    = cuDSS column-major ``n x w``) on the current Warp stream and is graph capturable once warmed for that
    width. Implements the Warpack inverse protocol (``n``, ``device``, ``width = 1``,
    ``requires_static_outer_capture``, ``apply``).
    """

    requires_static_outer_capture = True
    width = 1

    def __init__(self, indptr: wp.array, indices: wp.array, values: wp.array, n: int, device):
        _, cudss = _require()
        self._cudss = cudss
        self.n = int(n)
        self.device = wp.get_device(device)
        self.indptr, self.indices, self.values = indptr, indices, values
        self.handle = _CudssHandles.get(self.device)
        with wp.ScopedDevice(self.device):
            self.config = cudss.config_create()
            self.data = cudss.data_create(self.handle)
            self._A = cudss.matrix_create_csr(
                self.n,
                self.n,
                int(indices.shape[0]),
                indptr.ptr,
                0,
                indices.ptr,
                values.ptr,
                _R32I,
                _R32I,
                _R64F,
                int(cudss.MatrixType.SYMMETRIC),
                int(cudss.MatrixViewType.FULL),
                int(cudss.IndexBase.ZERO),
            )
            self._dummy = wp.zeros((1, self.n), dtype=wp.float64, device=self.device)
        self._dense: dict[int, tuple[int, int]] = {}
        self.analyzed = False
        self.factor_count = 0
        self.timings: dict[str, float] = {}
        self.info = 0
        self.inertia: tuple[int, int] | None = None
        self.perturbed_pivots = 0
        self.negative_pivot_count = -1
        self.sigma = float("nan")

    def _desc(self, w: int) -> tuple[int, int]:
        if w not in self._dense:
            c = self._cudss
            dx = c.matrix_create_dn(self.n, w, self.n, self._dummy.ptr, _R64F, int(c.Layout.COL_MAJOR))
            db = c.matrix_create_dn(self.n, w, self.n, self._dummy.ptr, _R64F, int(c.Layout.COL_MAJOR))
            self._dense[w] = (dx, db)
        return self._dense[w]

    def _exec(self, phase, dx, db):
        c = self._cudss
        c.set_stream(self.handle, wp.get_stream(self.device).cuda_stream)
        c.execute(self.handle, int(phase), self.config, self.data, self._A, dx, db)

    def _get(self, param, dtype, length):
        a = np.zeros(length, dtype=dtype)
        w = np.zeros(1, dtype=np.uint64)
        self._cudss.data_get(self.handle, self.data, int(param), a.ctypes.data, a.nbytes, w.ctypes.data)
        return a

    def analyze(self):
        """Reordering + symbolic factorization (once per pattern; synchronizes)."""
        dx, db = self._desc(1)
        wp.synchronize_device(self.device)
        t0 = time.perf_counter()
        self._exec(self._cudss.Phase.ANALYSIS, dx, db)
        wp.synchronize_device(self.device)
        self.timings["analysis"] = time.perf_counter() - t0
        self.analyzed = True

    def factorize(self, sigma: float = float("nan")):
        """Numeric factorization of the current ``values``; reads INFO, NPIVOTS and INERTIA (synchronizes)."""
        if not self.analyzed:
            self.analyze()
        c = self._cudss
        dx, db = self._desc(1)
        t0 = time.perf_counter()
        self._exec(c.Phase.FACTORIZATION, dx, db)
        wp.synchronize_device(self.device)
        self.timings["factorization"] = time.perf_counter() - t0
        self.factor_count += 1
        self.sigma = float(sigma)
        self.info = int(self._get(c.DataParam.INFO, np.int32, 1)[0])
        if self.info != 0:
            raise ValueError(f"cuDSS factorization of K - sigma M (sigma = {sigma:.6g}) reported info = {self.info}")
        self.perturbed_pivots = int(self._get(c.DataParam.NPIVOTS, np.int32, 1)[0])
        pos, neg = (int(v) for v in self._get(c.DataParam.INERTIA, np.int32, 2))
        self.inertia = (pos, neg)
        self.negative_pivot_count = neg + (self.n - pos - neg)  # zero pivots count as non-positive

    def apply(self, x: wp.array, y: wp.array):
        """``y = A^-1 x`` for contiguous FP64 rows ``(w, n)`` (distinct arrays)."""
        w = int(x.shape[0])
        dx, db = self._desc(w)
        self._cudss.matrix_set_values(db, x.ptr)
        self._cudss.matrix_set_values(dx, y.ptr)
        self._exec(self._cudss.Phase.SOLVE, dx, db)

    def free(self):
        c = getattr(self, "_cudss", None)
        if c is None:
            return
        for dx, db in self._dense.values():
            c.matrix_destroy(dx)
            c.matrix_destroy(db)
        self._dense = {}
        if getattr(self, "_A", None):
            c.matrix_destroy(self._A)
        if getattr(self, "data", None):
            c.data_destroy(self.handle, self.data)
        if getattr(self, "config", None):
            c.config_destroy(self.config)
        self._A = self.data = self.config = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


# ----------------------------------------------------------------------------------------------------------
# pencil structure (host index maps, built once per pattern)
# ----------------------------------------------------------------------------------------------------------
class _BlockStructure:
    """Free-node 3x3 block pattern derived from a symmetric source block pattern, plus the scalar CSR maps."""

    def __init__(self, src_offsets: np.ndarray, src_columns: np.ndarray, n_src_nodes: int, free_nodes: np.ndarray):
        src_offsets = np.asarray(src_offsets, dtype=np.int64)
        src_columns = np.asarray(src_columns, dtype=np.int64)
        nsrc = int(n_src_nodes)
        free_nodes = np.asarray(free_nodes, dtype=np.int64)
        if free_nodes.size and np.any(np.diff(free_nodes) <= 0):
            raise ValueError("free nodes must be strictly increasing")
        rows = np.repeat(np.arange(nsrc, dtype=np.int64), np.diff(src_offsets))
        keys = rows * nsrc + src_columns
        if np.any(np.diff(keys) <= 0):
            raise ValueError("source block pattern must be sorted CSR without duplicates")
        tkeys = src_columns * nsrc + rows
        trans_pos = np.searchsorted(keys, tkeys)
        trans_pos = np.minimum(trans_pos, keys.size - 1)
        if not np.array_equal(keys[trans_pos], tkeys):
            raise ValueError("source block pattern is not structurally symmetric")
        node_map = -np.ones(nsrc, dtype=np.int64)
        node_map[free_nodes] = np.arange(free_nodes.size)
        keep = (node_map[rows] >= 0) & (node_map[src_columns] >= 0)
        self.n_nodes = int(free_nodes.size)
        self.n = 3 * self.n_nodes
        self.src_nnzb = int(keys.size)
        self.src_a = np.nonzero(keep)[0].astype(np.int32)
        self.src_b = trans_pos[keep].astype(np.int32)
        frows = node_map[rows[keep]]
        self.columns = node_map[src_columns[keep]].astype(np.int32)
        self.block_rows = frows.astype(np.int32)
        self.offsets = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.add.at(self.offsets, frows + 1, 1)
        self.offsets = np.cumsum(self.offsets).astype(np.int32)
        self.nnzb = int(self.columns.size)
        self.keys = frows * self.n_nodes + self.columns.astype(np.int64)
        if not np.all(np.isin(np.arange(self.n_nodes) * (self.n_nodes + 1), self.keys)):
            raise ValueError("block pattern lacks diagonal blocks")
        ids = np.arange(1, 9 * self.nnzb + 1, dtype=np.float64).reshape(-1, 3, 3)
        S = sp.bsr_matrix((ids, self.columns, self.offsets), shape=(self.n, self.n)).tocsr()
        S.sort_indices()
        if S.nnz != 9 * self.nnzb:
            raise RuntimeError("scalar expansion lost entries")
        self.indptr = S.indptr.astype(np.int32)
        self.indices = S.indices.astype(np.int32)
        self.flat = (S.data.astype(np.int64) - 1).astype(np.int32)

    def host_blocks(self, A) -> np.ndarray:
        """Block values (nnzb, 3, 3) of a host sparse matrix on this (free) pattern; raises on entries outside."""
        B = sp.bsr_matrix(sp.csr_matrix(A), blocksize=(3, 3))
        B.sort_indices()
        n_nodes = self.n_nodes
        rows = np.repeat(np.arange(B.shape[0] // 3, dtype=np.int64), np.diff(B.indptr))
        keys = rows * n_nodes + B.indices.astype(np.int64)
        pos = np.searchsorted(self.keys, keys)
        pos = np.minimum(pos, self.keys.size - 1)
        if not np.array_equal(self.keys[pos], keys):
            raise ValueError("matrix has blocks outside the cached pattern")
        out = np.zeros((self.nnzb, 3, 3), dtype=np.float64)
        out[pos] = B.data
        return out


def _host_block_pattern(A) -> tuple[np.ndarray, np.ndarray, int]:
    B = sp.bsr_matrix(sp.csr_matrix(A), blocksize=(3, 3))
    B.sort_indices()
    return B.indptr.copy(), B.indices.copy(), B.shape[0] // 3


# ----------------------------------------------------------------------------------------------------------
# pencil on the device
# ----------------------------------------------------------------------------------------------------------
class _Pencil:
    """Device state of one restricted pencil: K, M (free-node BSR), shifted CSR values, two cuDSS factors,
    rigid rows Y, B = M Y, and the restriction data of the primary factor."""

    def __init__(self, structure: _BlockStructure, device):
        import warp.sparse as wps  # noqa: PLC0415

        warpack, _ = _require()
        self.s = structure
        self.device = wp.get_device(device)
        self.n = structure.n
        dev = self.device
        rows_wp = wp.array(structure.block_rows, dtype=wp.int32, device=dev)
        cols_wp = wp.array(structure.columns, dtype=wp.int32, device=dev)

        def bsr():
            A = wps.bsr_zeros(structure.n_nodes, structure.n_nodes, block_type=wp.mat33d, device=dev)
            wps.bsr_set_from_triplets(
                A, rows_wp, cols_wp, wp.zeros(structure.nnzb, dtype=wp.mat33d, device=dev), prune_numerical_zeros=False
            )
            off = A.offsets.numpy()[: structure.n_nodes + 1]
            col = A.columns.numpy()[: structure.nnzb]
            if not (np.array_equal(off, structure.offsets) and np.array_equal(col, structure.columns)):
                raise RuntimeError("warp.sparse produced an unexpected block pattern")
            return A

        self.K = bsr()
        self.M = bsr()
        self.Kv = self.K.values[: structure.nnzb]
        self.Mv = self.M.values[: structure.nnzb]
        self.K_op = warpack.SparseOperator(self.K)
        self.M_op = warpack.SparseOperator(self.M)
        self.block_rows = rows_wp
        self.src_a = wp.array(structure.src_a, dtype=wp.int32, device=dev)
        self.src_b = wp.array(structure.src_b, dtype=wp.int32, device=dev)
        self.flat = wp.array(structure.flat, dtype=wp.int32, device=dev)
        self.indptr = wp.array(structure.indptr, dtype=wp.int32, device=dev)
        self.indices = wp.array(structure.indices, dtype=wp.int32, device=dev)
        self.values = {
            name: wp.zeros(structure.indices.size, dtype=wp.float64, device=dev) for name in ("primary", "aux")
        }
        self.factors = {
            name: CudssFactor(self.indptr, self.indices, self.values[name], self.n, dev) for name in ("primary", "aux")
        }
        self.mass_host = None
        self.m_norm = None
        self.k_norm_value = float("nan")
        self.norm_ratio = 1.0
        self.d = 0
        self.Y = self.B = None
        self.restriction: dict[str, dict] = {}
        self._rowsum = wp.zeros(self.n, dtype=wp.float64, device=dev)
        self._scratch: dict[tuple, wp.array] = {}

    # -- values ------------------------------------------------------------------------------------
    def set_mass_host(self, M) -> bool:
        """Upload ``M`` (free DOFs) once; returns True when it changed."""
        Mc = sp.csr_matrix(M)
        if self.mass_host is not None and self.mass_host.nnz == Mc.nnz:
            old = self.mass_host
            if (
                np.array_equal(old.indptr, Mc.indptr)
                and np.array_equal(old.indices, Mc.indices)
                and np.array_equal(old.data, Mc.data)
            ):
                return False
        self.mass_host = Mc.copy()
        self.Mv.assign(self.s.host_blocks(Mc))
        self.m_norm = float(np.max(np.asarray(abs(Mc).sum(axis=0)).reshape(-1)))
        return True

    def set_stiffness_device(self, src_blocks: wp.array):
        """K = 0.5 (K_src + K_src^T) restricted to the free nodes, from device source block values."""
        if src_blocks.shape[0] < self.s.src_nnzb:
            raise ValueError("device tangent has fewer blocks than the cached source pattern")
        wp.launch(_symmetrize_blocks, self.s.nnzb, [src_blocks, self.src_a, self.src_b, self.Kv], device=self.device)

    def set_stiffness_host(self, K):
        self.Kv.assign(self.s.host_blocks(K))

    def k_norm(self) -> float:
        """One-norm of K (row abs sums; K symmetric) on the device (one host read)."""
        self._rowsum.zero_()
        wp.launch(_abs_row_sums, self.s.nnzb, [self.Kv, self.block_rows, self._rowsum], device=self.device)
        return float(np.max(self._rowsum.numpy()))

    # -- deflation ---------------------------------------------------------------------------------
    def set_deflation(self, Y: np.ndarray | None):
        d = 0 if Y is None else int(Y.shape[1])
        if d == 0:
            self.d = 0
            return
        rows = np.ascontiguousarray(np.asarray(Y, dtype=np.float64).T)
        if self.Y is None or self.Y.shape[0] != d:
            self.Y = wp.array(rows, dtype=wp.float64, device=self.device)
            self.B = wp.zeros_like(self.Y)
        else:
            self.Y.assign(rows)
        self.M_op.apply(self.Y, self.B)
        self.d = d

    def scratch(self, name: str, shape: tuple) -> wp.array:
        key = (name, *shape)
        if key not in self._scratch:
            self._scratch[key] = wp.zeros(shape, dtype=wp.float64, device=self.device)
        return self._scratch[key]

    # -- factorizations ----------------------------------------------------------------------------
    def factor(self, name: str, sigma: float) -> CudssFactor:
        """Numeric factorization of ``K - sigma M`` into factor ``name`` (+ restriction data)."""
        f = self.factors[name]
        wp.launch(
            _shifted_values,
            self.values[name].shape[0],
            [self.Kv, self.Mv, self.flat, float(sigma), self.values[name]],
            device=self.device,
        )
        f.factorize(sigma)
        info: dict[str, Any] = {"negA": f.negative_pivot_count}
        if self.d:
            d = self.d
            AinvB = self.scratch(f"AinvB_{name}", (d, self.n))
            f.apply(self.B, AinvB)
            S = self.scratch(f"S_{name}", (d, d))
            _launch_dots(self.B, AinvB, S, self.device)
            S_h = S.numpy()
            S_h = 0.5 * (S_h + S_h.T)
            Sinv = np.linalg.inv(S_h)
            Sinv_d = self.scratch(f"Sinv_{name}", (d, d))
            Sinv_d.assign(Sinv)
            info.update(AinvB=AinvB, S=S_h, Sinv=Sinv_d, negS=int(np.count_nonzero(np.linalg.eigvalsh(-S_h) < 0.0)))
        else:
            info["negS"] = 0
        info["below"] = info["negA"] + info["negS"] - self.d
        self.restriction[name] = info
        return f


class _RestrictedInverse:
    """Warpack inverse protocol: exact restricted inverse of ``K - sigma M`` on the M-complement of ``Y``."""

    requires_static_outer_capture = True
    width = 1

    def __init__(self, pencil: _Pencil, name: str):
        self.p = pencil
        self.name = name
        self.n, self.device = pencil.n, pencil.device

    def apply(self, x: wp.array, y: wp.array):
        p = self.p
        f = p.factors[self.name]
        if not p.d:
            f.apply(x, y)
            return
        w, d, dev = int(x.shape[0]), p.d, self.device
        r = p.restriction[self.name]
        t = p.scratch("ri_t", (w, self.n))
        c = p.scratch("ri_c", (d, w))
        c2 = p.scratch("ri_c2", (d, w))
        wp.copy(t, x)
        _launch_dots(p.Y, t, c, dev)
        wp.launch(_sub_combination, (w, self.n), [p.B, c, t], device=dev)  # b' = b - B Y^T b
        f.apply(t, y)
        _launch_dots(p.B, y, c, dev)
        wp.launch(_small_matmul, (d, w), [r["Sinv"], c, c2], device=dev)
        wp.launch(_sub_combination, (w, self.n), [r["AinvB"], c2, y], device=dev)


class _ProjectedStiffness:
    """``K_r = P^T K P`` with ``P = I - Y B^T`` (Warpack operator protocol); ``K`` itself when ``d = 0``."""

    def __init__(self, pencil: _Pencil):
        self.p = pencil
        self.n, self.device = pencil.n, pencil.device

    def apply(self, x: wp.array, y: wp.array):
        p = self.p
        if not p.d:
            p.K_op.apply(x, y)
            return
        w, d, dev = int(x.shape[0]), p.d, self.device
        t = p.scratch("ks_t", (w, self.n))
        c = p.scratch("ks_c", (d, w))
        wp.copy(t, x)
        _launch_dots(p.B, t, c, dev)
        wp.launch(_sub_combination, (w, self.n), [p.Y, c, t], device=dev)  # P x
        p.K_op.apply(t, y)
        _launch_dots(p.Y, y, c, dev)
        wp.launch(_sub_combination, (w, self.n), [p.B, c, y], device=dev)  # P^T K P x


# ----------------------------------------------------------------------------------------------------------
# shared restricted solver (controller factory product)
# ----------------------------------------------------------------------------------------------------------
class WarpackRestrictedSolver(ShiftedRestrictedSolver):
    """The update's restricted solver on the GPU: one cuDSS factorization of ``K - sigma M`` at the Ritz shift
    ``sigma = 1e-3 default_shift(K, M)`` serving the eigensolve, the static Ritz vectors and the modal derivatives.

    Duck-compatible with :class:`rerom.eigen.ShiftedRestrictedSolver` (``sigma``, ``factor_time``, ``solve``,
    ``factorization`` with ``negative_pivot_count`` / ``solve`` / ``norm``, ``AinvB`` / ``S6`` as host arrays on
    demand), so :class:`rerom.eigen.RestrictedArpackBackend` and :class:`rerom.eigen.RestrictedSubspaceBackend`
    can also use it (one host round trip per solve).
    """

    def __init__(self, pencil: _Pencil, sigma: float, factor_time: float, timings: dict):
        # no super().__init__: the factorization lives on the device
        self.pencil = pencil
        self.sigma = float(sigma)
        self.factor_time = float(factor_time)
        self.timings = dict(timings)
        self.symmetric = True
        self.inverse = _RestrictedInverse(pencil, "primary")
        self.factorization = _HostFactorView(pencil, "primary")
        self.lu = self.factorization

    @property
    def Y(self):
        return np.zeros((self.pencil.n, 0)) if not self.pencil.d else self.pencil.Y.numpy().T

    @property
    def B(self):
        return np.zeros((self.pencil.n, 0)) if not self.pencil.d else self.pencil.B.numpy().T

    @property
    def AinvB(self):
        r = self.pencil.restriction["primary"]
        return np.zeros((self.pencil.n, 0)) if not self.pencil.d else r["AinvB"].numpy().T

    @property
    def S6(self):
        r = self.pencil.restriction["primary"]
        return np.zeros((0, 0)) if not self.pencil.d else r["S"]

    @property
    def S6inv(self):
        return np.linalg.inv(self.S6) if self.pencil.d else np.zeros((0, 0))

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Restricted static solve of a host vector or (n, k) block (one upload, one download)."""
        b = np.asarray(b, dtype=np.float64)
        vec = b.ndim == 1
        rows = np.ascontiguousarray(b.reshape(self.pencil.n, -1).T)
        w = rows.shape[0]
        x = self.pencil.scratch("solve_x", (w, self.pencil.n))
        y = self.pencil.scratch("solve_y", (w, self.pencil.n))
        x.assign(rows)
        self.inverse.apply(x, y)
        out = y.numpy().T
        return out[:, 0].copy() if vec else np.ascontiguousarray(out)

    __call__ = solve


class _HostFactorView:
    """``ShiftedFactorization`` duck of one cuDSS factor (unrestricted ``A^-1`` with host round trips)."""

    symmetric_pivoting = True

    def __init__(self, pencil: _Pencil, name: str):
        self.p, self.name = pencil, name
        f = pencil.factors[name]
        self.sigma = f.sigma
        self.size = pencil.n
        self.negative_pivot_count = f.negative_pivot_count
        # one-norm bound of A = K - sigma M (||K||_1 + |sigma| ||M||_1)
        self.norm = float(pencil.k_norm_value + abs(f.sigma) * pencil.m_norm)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=np.float64)
        vec = rhs.ndim == 1
        rows = np.ascontiguousarray(rhs.reshape(self.size, -1).T)
        x = self.p.scratch("hv_x", rows.shape)
        y = self.p.scratch("hv_y", rows.shape)
        x.assign(rows)
        self.p.factors[self.name].apply(x, y)
        out = y.numpy().T
        return out[:, 0].copy() if vec else np.ascontiguousarray(out)


class WarpackContext:
    """Persistent GPU state of one body's re-ROM pencil, and the controller's restricted-solver factory.

    One instance per controller (body): ``BasisOptions(restricted_solver_factory=ctx)`` and
    ``eigensolver=WarpackKrylovBackend(ctx)`` (or :class:`WarpackLOBPCGBackend`). The block pattern, the scalar
    CSR maps, the cuDSS symbolic analyses and the Warpack solver workspaces are built on the first call and
    kept; later updates only refresh values (numeric factorization).

    Args:
        device: Warp CUDA device.
        assembler: Optional :class:`rerom.assembly_warp.WarpTetAssembler`; enables the device hand-off of its BSR
            tangent (``device_tangent``) without host copies.
        free_dofs: Free DOFs of the controller (whole nodes), required with ``assembler``.
    """

    def __init__(self, *, device="cuda:0", assembler=None, free_dofs: np.ndarray | None = None):
        _require()
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("WarpackContext needs a CUDA device")
        self.assembler = assembler
        self.free_dofs = None if free_dofs is None else np.asarray(free_dofs, dtype=np.int64)
        self.pencil: _Pencil | None = None
        self.solver: WarpackRestrictedSolver | None = None
        self.setup_s: dict[str, float] = {}
        self.update_count = 0
        self.last_timings: dict[str, float] = {}

    # -- structure ---------------------------------------------------------------------------------
    def _free_nodes(self, n_src_nodes: int) -> np.ndarray:
        if self.free_dofs is None:
            return np.arange(n_src_nodes)
        fd = self.free_dofs
        nodes = fd[::3] // 3
        if fd.size % 3 or not np.array_equal(fd, (3 * nodes[:, None] + np.arange(3)[None, :]).reshape(-1)):
            raise ValueError("free DOFs must be whole nodes for the 3x3 block pencil")
        return nodes

    def _ensure(self, Kf, use_device: bool):
        if self.pencil is not None:
            return
        t0 = time.perf_counter()
        if use_device:
            asm = self.assembler
            structure = _BlockStructure(
                asm.block_offsets, asm.block_columns, asm.n_nodes, self._free_nodes(asm.n_nodes)
            )
        else:
            off, col, nn = _host_block_pattern(Kf)
            structure = _BlockStructure(off, col, nn, np.arange(nn))
        t1 = time.perf_counter()
        self.pencil = _Pencil(structure, self.device)
        wp.synchronize_device(self.device)
        self.setup_s = {"pattern_maps": t1 - t0, "device_alloc": time.perf_counter() - t1}

    def __call__(self, Kf, Mf, Yf, *, device_tangent: wp.array | None = None) -> WarpackRestrictedSolver:
        """Refresh the pencil to ``(Kf, Mf, Yf)`` and factor ``K - sigma M`` at the Ritz shift.

        ``device_tangent``: the assembler's device BSR values of the full, unsymmetrized ``K_T`` (used instead
        of the host ``Kf`` when the context was built with that assembler).
        """
        t0 = time.perf_counter()
        use_device = device_tangent is not None and self.assembler is not None
        self._ensure(Kf, use_device)
        p = self.pencil
        if p.n != Kf.shape[0]:
            raise ValueError(f"pencil size {p.n} != matrix size {Kf.shape[0]}")
        tim: dict[str, float] = {}
        ta = time.perf_counter()
        p.set_mass_host(Mf)
        if use_device:
            p.set_stiffness_device(device_tangent)
        else:
            p.set_stiffness_host(Kf)
        p.set_deflation(Yf)
        knorm = p.k_norm()
        p.k_norm_value = knorm
        p.norm_ratio = knorm / p.m_norm if knorm > 0 and p.m_norm > 0 else 1.0
        sigma = -1.0e-6 * knorm / p.m_norm * 1.0e-3 if knorm > 0 and p.m_norm > 0 else -1.0e-3
        tim["handoff"] = time.perf_counter() - ta
        tf = time.perf_counter()
        first = not p.factors["primary"].analyzed
        if first:
            wp.launch(
                _shifted_values,
                p.values["primary"].shape[0],
                [p.Kv, p.Mv, p.flat, float(sigma), p.values["primary"]],
                device=self.device,
            )
            p.factors["primary"].analyze()
            tim["analysis"] = p.factors["primary"].timings["analysis"]
        p.factor("primary", sigma)
        tim["numeric_factor"] = p.factors["primary"].timings["factorization"]
        tim["restriction"] = time.perf_counter() - tf - tim["numeric_factor"] - tim.get("analysis", 0.0)
        self.update_count += 1
        self.last_timings = tim
        self.solver = WarpackRestrictedSolver(p, sigma, time.perf_counter() - t0, tim)
        return self.solver


# ----------------------------------------------------------------------------------------------------------
# backends
# ----------------------------------------------------------------------------------------------------------
def _sturm_tau(vals: np.ndarray, count: int, rel_gap: float = 1.0e-6) -> tuple[float | None, int]:
    """Shift in the first gap at or after pair ``count`` (``vals`` ascending); returns (tau, pairs below)."""
    for j in range(max(count, 1), vals.size):
        lo, hi = float(vals[j - 1]), float(vals[j])
        if hi - lo > rel_gap * max(abs(hi), abs(lo), 1.0e-300):
            return 0.5 * (lo + hi), j
    return None, 0


class _WarpackBackendBase:
    accepts_restricted_factorization = True
    supports_inertia_count = True

    def __init__(
        self,
        context: WarpackContext | None,
        *,
        tol: float,
        guard: int,
        sturm_check: bool,
        residual_tol: float,
        seed: int,
    ):
        self.context = context if context is not None else WarpackContext()
        self.device = str(self.context.device)
        self.tol = float(tol)
        self.guard = int(guard)
        self.sturm_check = bool(sturm_check)
        self.residual_tol = float(residual_tol)
        self.seed = int(seed)
        self.floor_factor = 8.0
        self.last_info: dict[str, Any] = {}
        self._solvers: dict[tuple, Any] = {}

    def _accepted_count(self, p: _Pencil, values: wp.array, residuals: wp.array) -> tuple[int, float]:
        """Pairs whose Warpack residual is below ``max(tol, floor_factor eps (||K||_1/||M||_1) / |lambda|)``.

        The second term is the roundoff floor of the original-equation residual of a soft mode (computing
        ``K x`` loses ``eps ||K|| ||x||`` against ``||K x|| = lambda ||M x||``); no solver gets below it, so a strict
        ``tol`` would never stop on the lowest modes of a soft body. Returns (count, max residual / bound).
        """
        lam = np.abs(values.numpy())
        res = residuals.numpy()
        bound = np.maximum(
            self.tol,
            self.floor_factor * np.finfo(np.float64).eps * p.norm_ratio / np.maximum(lam, np.finfo(np.float64).tiny),
        )
        ok = np.isfinite(res) & (res <= bound)
        return int(np.count_nonzero(ok)), float(np.max(res / bound))

    def _shared(self, stiffness, mass, deflation, factorization) -> WarpackRestrictedSolver:
        if isinstance(factorization, WarpackRestrictedSolver) and factorization.pencil is self.context.pencil:
            return factorization
        return self.context(sp.csr_matrix(stiffness), sp.csr_matrix(mass), deflation)

    def _check(self, p: _Pencil, vals: np.ndarray, X: wp.array) -> dict:
        """Gate metrics on the device: max ``||(I - B Y^T)(K v - lam M v)|| / ||K v||`` and M-orthonormality."""
        k = int(X.shape[0])
        dev = p.device
        kx = p.scratch("chk_kx", (k, p.n))
        mx = p.scratch("chk_mx", (k, p.n))
        r = p.scratch("chk_r", (k, p.n))
        lam = wp.array(vals, dtype=wp.float64, device=dev)
        p.K_op.apply(X, kx)
        p.M_op.apply(X, mx)
        wp.launch(_residual_rows, (k, p.n), [kx, mx, lam, r], device=dev)
        if p.d:
            c = p.scratch("chk_c", (p.d, k))
            _launch_dots(p.Y, r, c, dev)
            wp.launch(_sub_combination, (k, p.n), [p.B, c, r], device=dev)
        nr = wp.zeros(k, dtype=wp.float64, device=dev)
        nk = wp.zeros(k, dtype=wp.float64, device=dev)
        chunks = (p.n + _TILE - 1) // _TILE
        wp.launch_tiled(_row_norm2, dim=(k, chunks), inputs=[r, nr], block_dim=128, device=dev)
        wp.launch_tiled(_row_norm2, dim=(k, chunks), inputs=[kx, nk], block_dim=128, device=dev)
        G = p.scratch("chk_g", (k, k))
        _launch_dots(X, mx, G, dev)
        rel = np.sqrt(nr.numpy()) / np.sqrt(np.maximum(nk.numpy(), 1.0e-300))
        g = G.numpy()
        out = {
            "rel_residual": rel.tolist(),
            "max_rel_residual": float(np.max(rel)),
            "m_orth": float(np.max(np.abs(g - np.eye(k)))),
        }
        if p.d:
            cy = p.scratch("chk_cy", (p.d, k))
            _launch_dots(p.B, X, cy, dev)
            out["max_rigid_overlap"] = float(np.max(np.abs(cy.numpy())))
        return out

    def _sturm(self, p: _Pencil, vals: np.ndarray, count: int, info: dict) -> tuple[int | None, float | None]:
        if not self.sturm_check:
            return None, None
        tau, j = _sturm_tau(vals, count)
        if tau is None:
            info["sturm"] = {"skipped": "no resolved gap above the requested pairs"}
            return None, None
        ts = time.perf_counter()
        aux = p.factor("aux", tau)
        r = p.restriction["aux"]
        info["sturm"] = {
            "tau": tau,
            "pairs_below": j,
            "negA": r["negA"],
            "negS": r["negS"],
            "below": r["below"],
            "perturbed_pivots": aux.perturbed_pivots,
            "t_s": time.perf_counter() - ts,
        }
        info["t_factor"] = info.get("t_factor", 0.0) + aux.timings["factorization"]
        if aux.perturbed_pivots:
            info["sturm"]["warning"] = "perturbed pivots: count not certified"
        return int(r["below"]) + p.d, tau

    def _finish(self, p, vals_dev, X, count, t0, info) -> newton.ModalEigenResult:
        vals = np.asarray(vals_dev, dtype=np.float64)
        order = np.argsort(vals, kind="stable")
        vals = vals[order]
        Xh = X.numpy()[order]
        Xs = wp.array(np.ascontiguousarray(Xh), dtype=wp.float64, device=p.device)
        tc = time.perf_counter()
        info.update(self._check(p, vals, Xs))
        inertia_count, tau = self._sturm(p, vals, count, info)
        info["t_check_s"] = time.perf_counter() - tc
        if info["max_rel_residual"] > self.residual_tol or not np.all(np.isfinite(vals)):
            self.last_info = info
            raise ValueError(
                f"{self.name}: eigenpairs not accepted (max relative residual {info['max_rel_residual']:.3e} > "
                f"{self.residual_tol:.1e}, status {info.get('status')}, converged {info.get('converged')})"
            )
        self.last_info = info
        return newton.ModalEigenResult(
            vals,
            np.ascontiguousarray(Xh.T),
            inertia_count=inertia_count,
            backend=self.name,
            wall_s=time.perf_counter() - t0,
            inertia_shift=tau,
        )


class WarpackKrylovBackend(_WarpackBackendBase):
    """``warpack.GeneralizedKrylovSchur`` (shift-invert, M inner product, thick restart), cold start.

    The shift is the shared factorization's Ritz shift. When the Sturm count at that shift finds restricted
    eigenvalues below it (indefinite tangent), the shift is lowered (as in
    :class:`rerom.eigen.RestrictedArpackBackend`) on the auxiliary factor until none remain, so negative
    eigenvalues are found and returned. ``guard`` extra pairs are converged so that the Sturm shift can be placed
    in a resolved gap above the requested ones. The restart loop runs eagerly with one host read of the
    converged count per restart (cuDSS needs a fixed outer budget for capture; see :meth:`capture`).

    Args:
        context: The body's :class:`WarpackContext` (shared with the restricted-solver factory).
        ncv: Krylov basis size (default ``max(2k, k + 8)``, ``k = count + guard``).
        tol: Warpack stopping tolerance (its residual normalization ``max(1, ||Kx|| + |lam| ||Mx||)``).
        max_restarts: Restart budget; exhausting it raises ``ValueError``.
        residual_tol: Acceptance bound on ``||(I - B Y^T)(K v - lam M v)|| / ||K v||``.
    """

    name = "warpack-krylov-schur"

    def __init__(
        self,
        context: WarpackContext | None = None,
        *,
        ncv: int | None = None,
        tol: float = 1.0e-8,
        max_restarts: int = 100,
        guard: int = 2,
        sturm_check: bool = True,
        residual_tol: float = 1.0e-6,
        seed: int = 42,
    ):
        super().__init__(context, tol=tol, guard=guard, sturm_check=sturm_check, residual_tol=residual_tol, seed=seed)
        self._start: dict[tuple, wp.array] = {}
        self.ncv = ncv
        self.max_restarts = int(max_restarts)

    def solver_for(self, p: _Pencil, k: int, inverse_name: str):
        warpack, _ = _require()
        ncv = self.ncv if self.ncv is not None else max(2 * k, k + 8)
        ncv = min(ncv, p.n - p.d)
        key = (id(p), k, ncv, inverse_name)
        if key not in self._solvers:
            # start vector OP(random): in the range of the restricted inverse, hence M-orthogonal to Y (a plain
            # random start keeps rigid components in the Krylov basis and stalls the free-body residuals)
            v0 = wp.zeros((1, p.n), dtype=wp.float64, device=p.device)
            self._solvers[key] = warpack.GeneralizedKrylovSchur(
                _ProjectedStiffness(p),
                p.M_op,
                _RestrictedInverse(p, inverse_name),
                k,
                ncv=ncv,
                tol=self.tol,
                target=0.0,
                initial_vector=v0,
            )
            self._start[key] = v0
        return self._solvers[key]

    def _seed_start(self, p: _Pencil, key_solver, inverse_name: str):
        v0 = self._start[key_solver]
        rng = np.random.default_rng(self.seed)
        r = p.scratch("ks_r", (1, p.n))
        r.assign(rng.standard_normal((1, p.n)))
        _RestrictedInverse(p, inverse_name).apply(r, v0)

    def solve(
        self,
        stiffness,
        mass,
        count: int,
        *,
        deflation=None,
        sigma=None,
        tol: float = 0.0,
        factorization=None,
        zero_threshold: float = 0.0,
    ) -> newton.ModalEigenResult:
        del sigma, tol, zero_threshold
        t0 = time.perf_counter()
        shared = self._shared(stiffness, mass, deflation, factorization)
        p = shared.pencil
        count = min(int(count), p.n - p.d)
        k = min(count + self.guard, p.n - p.d - 1)
        info: dict[str, Any] = {"path": self.name, "sigma": shared.sigma, "t_factor": 0.0}
        # Sturm check at the operator shift: lower it while restricted eigenvalues lie below
        below = p.restriction["primary"]["below"]
        inverse_name, sigma_op = "primary", shared.sigma
        scale = shared.pencil.k_norm() / p.m_norm if below > 0 else 1.0
        tries = [(sigma_op, below)]
        attempt = 0
        while below > 0 and attempt < 12:
            sigma_op = sigma_op - scale * (10.0 ** (attempt - 5))
            p.factor("aux", sigma_op)
            info["t_factor"] += p.factors["aux"].timings["factorization"]
            below = p.restriction["aux"]["below"]
            tries.append((sigma_op, below))
            inverse_name = "aux"
            attempt += 1
        if below > 0:
            raise ValueError(f"{self.name}: could not place the shift below the restricted spectrum ({tries})")
        info.update(operator_sigma=sigma_op, sigma_tries=tries, below_sigma=tries[0][1])
        ts = time.perf_counter()
        ks = self.solver_for(p, k, inverse_name)
        info["t_solver_setup_s"] = time.perf_counter() - ts
        te = time.perf_counter()
        key = next(kk for kk, v in self._solvers.items() if v is ks)
        self._seed_start(p, key, inverse_name)
        ks.initialize(self.seed)
        restarts, conv, status, acc, ratio = 0, 0, 0, 0, float("inf")
        while restarts < self.max_restarts:
            ks.step()
            ks.finalize()
            restarts += 1
            conv, status = int(ks.converged.numpy()[0]), int(ks.status.numpy()[0])
            acc, ratio = self._accepted_count(p, ks.eigenvalues, ks.residuals)
            if status != 0 or conv >= k or acc >= k:
                break
        info["t_eigen_s"] = time.perf_counter() - te
        info.update(
            restarts=restarts,
            converged=conv,
            accepted_floor_aware=acc,
            max_residual_over_bound=ratio,
            k=k,
            ncv=ks.m,
            status=status,
            warpack_max_residual=float(np.max(ks.residuals.numpy())),
        )
        if status != 0 or max(conv, acc) < k:
            self.last_info = info
            raise ValueError(
                f"{self.name}: not converged (status {status}, converged {conv}/{k} after {restarts} restarts)"
            )
        return self._finish(p, ks.eigenvalues.numpy(), ks.eigenvectors, count, t0, info)


class WarpackLOBPCGBackend(_WarpackBackendBase):
    """``warpack.BlockLOBPCG`` warm-started from :attr:`previous_vectors`, preconditioned by the shared restricted
    shift-invert inverse, rigid motions as constraints of the projected pencil ``(K_r, M)``.

    The warm start is the update analogue of :class:`rerom.eigen.RestrictedSubspaceBackend` but iterates to
    ``tol``. As Warpack documents, a warm start is no completeness proof: the Sturm count (on by default) checks
    that no restricted eigenvalue below the returned ones was missed (``validate_eigen_result`` then raises).

    Args:
        width: Block width (default ``k + 4``, ``k = count + guard``; ``3 width <= n`` is required).
        max_iterations: Iteration budget; exhausting it raises ``ValueError``.
    """

    name = "warpack-lobpcg"

    def __init__(
        self,
        context: WarpackContext | None = None,
        *,
        width: int | None = None,
        tol: float = 1.0e-8,
        max_iterations: int = 200,
        guard: int = 2,
        sturm_check: bool = True,
        residual_tol: float = 1.0e-6,
        seed: int = 42,
        previous_vectors: np.ndarray | None = None,
    ):
        super().__init__(context, tol=tol, guard=guard, sturm_check=sturm_check, residual_tol=residual_tol, seed=seed)
        self.width = width
        self.max_iterations = int(max_iterations)
        self.previous_vectors = previous_vectors
        self._start: dict[tuple, wp.array] = {}

    def solver_for(self, p: _Pencil, k: int, n_start: int):
        warpack, _ = _require()
        width = self.width if self.width is not None else k + 4
        width = max(width, k)
        key = (id(p), k, width, n_start, p.d)
        if key not in self._solvers:
            start = None
            if n_start:
                start = wp.zeros((min(n_start, width), p.n), dtype=wp.float64, device=p.device)
            self._start[key] = start
            self._solvers[key] = warpack.BlockLOBPCG(
                _ProjectedStiffness(p),
                k,
                width=width,
                mass=p.M_op,
                preconditioner=_RestrictedInverse(p, "primary"),
                initial_vectors=start,
                constraints=p.Y if p.d else None,
                tol=self.tol,
            )
        return self._solvers[key], self._start[key]

    def solve(
        self,
        stiffness,
        mass,
        count: int,
        *,
        deflation=None,
        sigma=None,
        tol: float = 0.0,
        factorization=None,
        zero_threshold: float = 0.0,
    ) -> newton.ModalEigenResult:
        del sigma, tol, zero_threshold
        t0 = time.perf_counter()
        shared = self._shared(stiffness, mass, deflation, factorization)
        p = shared.pencil
        count = min(int(count), p.n - p.d)
        k = min(count + self.guard, p.n - p.d - 1)
        info: dict[str, Any] = {
            "path": self.name,
            "sigma": shared.sigma,
            "t_factor": 0.0,
            "below_sigma": p.restriction["primary"]["below"],
        }
        X = self.previous_vectors
        n_start = 0 if X is None or X.shape[0] != p.n else int(X.shape[1])
        ts = time.perf_counter()
        lob, start = self.solver_for(p, k, n_start)
        if start is not None:
            start.assign(np.ascontiguousarray(np.asarray(X, dtype=np.float64).T[: start.shape[0]]))
        info["t_solver_setup_s"] = time.perf_counter() - ts
        te = time.perf_counter()
        lob.initialize(self.seed)
        its, conv, status, acc, ratio = 0, 0, 0, 0, float("inf")
        while its < self.max_iterations:
            lob.step()
            its += 1
            conv, status = int(lob.converged.numpy()[0]), int(lob.status.numpy()[0])
            acc, ratio = self._accepted_count(p, lob.eigenvalues, lob.residuals)
            if status != 0 or conv >= k or acc >= k:
                break
        info["t_eigen_s"] = time.perf_counter() - te
        info.update(
            iterations=its,
            converged=conv,
            accepted_floor_aware=acc,
            max_residual_over_bound=ratio,
            k=k,
            width=lob.b,
            status=status,
            warm_rows=n_start,
            warpack_max_residual=float(np.max(lob.residuals.numpy())),
        )
        if status != 0 or max(conv, acc) < k:
            self.last_info = info
            raise ValueError(
                f"{self.name}: not converged (status {status}, converged {conv}/{k} after {its} iterations)"
            )
        res = self._finish(p, lob.eigenvalues.numpy(), lob.eigenvectors, count, t0, info)
        self.previous_vectors = np.asarray(res.vectors)
        return res
