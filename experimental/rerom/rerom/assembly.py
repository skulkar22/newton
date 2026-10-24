# ruff: noqa: TID253
"""Tangent assembly for re-ROM: the ``TangentAssembler`` protocol and the numpy tet4 assembler.

Contents
--------
TangentAssembler        protocol the controller needs from a nonlinear assembler
TetMesh                 rest mesh (nodes, tets, density, Young's modulus, Poisson ratio, node sets)
make_beam_mesh          structured box beam, hex grid split into 6 Kuhn tetrahedra
make_flexure_mesh       bulky notched block (double-notch flexure hinge with a thin web)
NumpyTetAssembler       total-Lagrangian StVK / compressible neo-Hookean energy, internal force,
                        exact consistent tangent (material + geometric), consistent and lumped
                        mass, per-element polar-decomposition rotation angles (host, numpy/scipy)
WarpTetAssembler        slot for the GPU assembler of work package WP-D (not in this package yet)

Conventions
-----------
* DOF layout: x_flat[3*n + i] is coordinate i (x, y, z) of node n.
* Total-Lagrangian w.r.t. ``mesh.nodes`` (rest configuration X).  F = dx/dX is constant per tet.
* f_int = dW/dx (the force the body exerts *against* deformation; equation of motion
  M a + f_int(x) = f_ext).  K_T = d f_int / dx, symmetric.
* All quantities SI, float64.
"""

from __future__ import annotations

import enum
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp

__all__ = [
    "HostTangentAssembler",
    "Material",
    "NumpyTetAssembler",
    "TangentAssembler",
    "TetMesh",
    "default_assembler",
    "lame_parameters",
    "make_beam_mesh",
    "make_flexure_mesh",
]


@runtime_checkable
class TangentAssembler(Protocol):
    """What :class:`rerom.ReROMController` needs from a nonlinear (total-Lagrangian) assembler.

    DOF layout ``x_flat[3 n + i]``; ``f_int = dW/dx``; ``K_T = df_int/dx`` (symmetric).
    ``element_terms`` is needed only by the hyper-reduced IMEX remainder and the energy trigger,
    ``element_rotations`` only by :class:`rerom.RotationTrigger`.
    """

    mass_matrix: Any
    """Consistent mass matrix (sparse, 3N x 3N) [kg]."""
    mesh: Any
    """Rest mesh with ``nodes`` (N, 3) and ``tets`` (E, 4)."""

    def evaluate(self, x_flat: np.ndarray, *, need_tangent: bool = True) -> tuple[float, np.ndarray, Any]:
        """Return ``(W, f_int, K_T)`` at the nodal positions ``x_flat`` (3N,); ``K_T`` is None without tangent."""
        ...

    def element_terms(self, x_elem: np.ndarray, element_ids=None, *, need_tangent: bool = True):
        """Return per-element ``(W_e (E,), f_e (E,4,3), K_e (E,4,3,4,3) or None)`` for element nodes ``x_elem``."""
        ...

    def element_rotations(self, x_flat: np.ndarray, element_ids=None, reference_flat=None) -> np.ndarray:
        """Return the rotation angle [rad] of each element relative to the rest (or ``reference_flat``)."""
        ...


# ----------------------------------------------------------------------------------------------
# Mesh
# ----------------------------------------------------------------------------------------------


def lame_parameters(young: float, poisson: float) -> tuple[float, float]:
    """Return (lambda, mu) from Young's modulus and Poisson ratio."""
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    return lam, mu


@dataclass
class TetMesh:
    """Linear tetrahedral mesh in its rest configuration.

    nodes   (N,3) float64 rest positions X
    tets    (E,4) int64 node indices, positively oriented (det[X1-X0, X2-X0, X3-X0] > 0)
    density [kg/m^3], young [Pa], poisson [-]
    node_sets  named node-index arrays (e.g. "root", "tip", "tip_center")
    meta    free-form generation parameters
    """

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

    # aliases used by the WP-C harness
    @property
    def young_modulus(self) -> float:
        return self.young

    @property
    def poisson_ratio(self) -> float:
        return self.poisson

    @property
    def lame(self) -> tuple[float, float]:
        return lame_parameters(self.young, self.poisson)

    def tet_volumes(self) -> np.ndarray:
        X = self.nodes[self.tets]
        Dm = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=-1)
        return np.linalg.det(Dm) / 6.0

    def total_mass(self) -> float:
        return float(self.density * self.tet_volumes().sum())

    def nodes_where(self, predicate: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Indices of rest nodes X with predicate(X) true (predicate gets the (N,3) array)."""
        return np.nonzero(predicate(self.nodes))[0]


# Kuhn (Freudenthal) split of a unit cube into 6 tetrahedra sharing the 000-111 diagonal.
# Every cube is split identically, so the triangulation is conforming across shared faces.
_CUBE_CORNER = {(bx, by, bz): bx * 4 + by * 2 + bz for bx in (0, 1) for by in (0, 1) for bz in (0, 1)}


def _kuhn_tets() -> np.ndarray:
    out = []
    for perm in itertools.permutations(range(3)):
        c = [0, 0, 0]
        path = [tuple(c)]
        for ax in perm:
            c[ax] = 1
            path.append(tuple(c))
        out.append([_CUBE_CORNER[p] for p in path])
    return np.asarray(out, dtype=np.int64)


_KUHN = _kuhn_tets()


def _structured_tet_mesh(xs, ys, zs, keep_cell=None, mirrored=True):
    """Tensor-product grid with coordinates xs, ys, zs; each kept hex cell -> 6 Kuhn tets.

    mirrored=True (default) reflects the Kuhn pattern of cell (i,j,k) in each direction whose
    index is odd.  The triangulation stays conforming (the diagonal of a face normal to axis d
    depends only on the flags of the two other axes, which neighbours across that face share)
    and, for an even cell count across a symmetry plane, the mesh is mirror symmetric.  The
    plain Kuhn split (mirrored=False) is anisotropic: a square cantilever under vertical
    gravity drifted sideways by ~6% of its vertical deflection.

    Grid node (ix, iy, iz) has provisional id (ix*(ny+1) + iy)*(nz+1) + iz (z fastest, x slowest).
    Unused nodes (only when cells are removed) are dropped and the remaining nodes renumbered in
    the same order.  Returns nodes, tets, grid index (M,3) of each kept node.
    """
    xs, ys, zs = (np.asarray(a, dtype=np.float64) for a in (xs, ys, zs))
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    IX, IY, IZ = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    gid = lambda ix, iy, iz: (ix * (ny + 1) + iy) * (nz + 1) + iz  # noqa: E731
    nodes = np.stack([xs[IX.ravel()], ys[IY.ravel()], zs[IZ.ravel()]], axis=1)
    grid_idx = np.stack([IX.ravel(), IY.ravel(), IZ.ravel()], axis=1)

    cx, cy, cz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    cx, cy, cz = cx.ravel(), cy.ravel(), cz.ravel()
    if keep_cell is not None:
        centers = np.stack(
            [0.5 * (xs[cx] + xs[cx + 1]), 0.5 * (ys[cy] + ys[cy + 1]), 0.5 * (zs[cz] + zs[cz + 1])], axis=1
        )
        mask = np.asarray(keep_cell(centers), dtype=bool)
        cx, cy, cz = cx[mask], cy[mask], cz[mask]
    corners = np.empty((cx.size, 8), dtype=np.int64)
    if mirrored:
        fx, fy, fz = cx % 2, cy % 2, cz % 2  # reflect the local cube in every odd-index direction
    else:
        fx = fy = fz = 0
    for (bx, by, bz), k in _CUBE_CORNER.items():
        corners[:, k] = gid(cx + (bx ^ fx), cy + (by ^ fy), cz + (bz ^ fz))
    tets = corners[:, _KUHN].reshape(-1, 4)

    # orientation fix
    X = nodes[tets]
    vol = np.linalg.det(np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=-1))
    neg = vol < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]

    used = np.unique(tets)
    if used.size != nodes.shape[0]:
        remap = -np.ones(nodes.shape[0], dtype=np.int64)
        remap[used] = np.arange(used.size)
        nodes, grid_idx, tets = nodes[used], grid_idx[used], remap[tets]
    return nodes, tets, grid_idx


def make_beam_mesh(
    length: float,
    width: float,
    height: float,
    nx: int,
    ny: int,
    nz: int,
    density: float = 1000.0,
    young: float = 1.0e7,
    poisson: float = 0.3,
    mirrored: bool = True,
) -> TetMesh:
    """Box beam along +x: x in [0, length], y in [-width/2, width/2], z in [-height/2, height/2].

    Hex grid nx*ny*nz, each hex -> 6 Kuhn tets (mirrored pattern, see _structured_tet_mesh) (6*nx*ny*nz tets, (nx+1)(ny+1)(nz+1) nodes).
    Node ordering: id = (ix*(ny+1) + iy)*(nz+1) + iz, i.e. z fastest, x slowest; nodes of the
    cross-section ix are the contiguous block [ix*(ny+1)*(nz+1), (ix+1)*(ny+1)*(nz+1)).
    Node sets: "root" (x=0 face), "tip" (x=length face), "tip_center" (nodes on the tip face
    closest to the beam axis; 1 node if ny, nz even, else up to 4), "axis" (nodes closest to
    the centre line, ordered by x when ny, nz are even).
    """
    xs = np.linspace(0.0, length, nx + 1)
    ys = np.linspace(-0.5 * width, 0.5 * width, ny + 1)
    zs = np.linspace(-0.5 * height, 0.5 * height, nz + 1)
    nodes, tets, _gidx = _structured_tet_mesh(xs, ys, zs, mirrored=mirrored)
    mesh = TetMesh(nodes, tets, density, young, poisson)
    tol = 1e-9 * max(length, width, height)
    root = np.nonzero(np.abs(nodes[:, 0]) < tol)[0]
    tip = np.nonzero(np.abs(nodes[:, 0] - length) < tol)[0]
    r2 = nodes[:, 1] ** 2 + nodes[:, 2] ** 2
    axis_mask = np.abs(r2 - r2.min()) < tol * tol + 1e-12 * max(width, height) ** 2
    axis = np.nonzero(axis_mask)[0]
    axis = axis[np.argsort(nodes[axis, 0], kind="stable")]
    mesh.node_sets = {
        "root": root,
        "tip": tip,
        "tip_center": np.intersect1d(tip, axis),
        "axis": axis,
    }
    mesh.meta = {
        "kind": "beam",
        "length": length,
        "width": width,
        "height": height,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "split": "kuhn6_mirrored" if mirrored else "kuhn6",
        "density": density,
        "young": young,
        "poisson": poisson,
    }
    return mesh


def _graded(a: float, b: float, n: int) -> np.ndarray:
    return np.linspace(a, b, max(int(n), 1) + 1)


def make_flexure_mesh(
    block_length: float = 0.1,
    block_width: float = 0.02,
    block_height: float = 0.02,
    notch_start: float = 0.02,
    notch_length: float = 0.03,
    web_thickness: float = 0.002,
    web_offset: float = 0.0,
    h_block: float = 0.004,
    n_web_z: int = 2,
    n_notch_x: int | None = None,
    ny: int | None = None,
    density: float = 1200.0,
    young: float = 3.5e7,
    poisson: float = 0.35,
    mirrored: bool = True,
) -> TetMesh:
    """Bulky notched block with a flexure hinge (double-sided rectangular notch, thin web).

    Geometry: block x in [0, block_length], y in [-W/2, W/2], z in [-H/2, H/2].  Material is
    removed for x in [notch_start, notch_start+notch_length] and |z - web_offset| > t/2, leaving
    a web of thickness t = web_thickness (centred at z = web_offset).  Notch depths are
    H/2 - t/2 + web_offset (bottom) and H/2 - t/2 - web_offset (top), reported in meta.
    Bending under a load on the distal block localises in the web (hinge).

    Discretisation: tensor-product grid whose lines include the notch and web boundaries.
    Block regions use spacing ~h_block; the web uses n_web_z cells through its thickness and
    n_notch_x cells along its length (default: spacing ~ 2*t/n_web_z... at least 4 cells).
    Node ordering as in make_beam_mesh (z fastest, x slowest) after dropping unused nodes.
    Node sets: "root" (x=0 face), "tip" (x=block_length face), "tip_center" (tip node nearest
    the block axis), "web" (nodes inside the web), "distal" (nodes with x >= notch end).
    """
    L, W, H, t = block_length, block_width, block_height, web_thickness
    x0, x1 = notch_start, notch_start + notch_length
    z0, z1 = web_offset - 0.5 * t, web_offset + 0.5 * t
    if not (0 < x0 < x1 < L):
        raise ValueError("notch must lie strictly inside the block")
    if not (-0.5 * H < z0 < z1 < 0.5 * H):
        raise ValueError("web must lie strictly inside the block height")
    if n_notch_x is None:
        n_notch_x = max(4, int(np.ceil(notch_length / max(t / n_web_z * 3.0, 1e-12))))
    if ny is None:
        ny = max(2, 2 * int(np.ceil(0.5 * W / h_block)))  # even -> mirror symmetric about y = 0

    def seg(a, b, h):
        return _graded(a, b, max(1, int(round((b - a) / h))))

    xs = np.unique(np.concatenate([seg(0, x0, h_block), _graded(x0, x1, n_notch_x), seg(x1, L, h_block)]))
    zs = np.unique(np.concatenate([seg(-0.5 * H, z0, h_block), _graded(z0, z1, n_web_z), seg(z1, 0.5 * H, h_block)]))
    ys = np.linspace(-0.5 * W, 0.5 * W, ny + 1)

    def keep(c):
        in_notch = (c[:, 0] > x0) & (c[:, 0] < x1)
        in_web = (c[:, 2] > z0) & (c[:, 2] < z1)
        return ~in_notch | in_web

    nodes, tets, _ = _structured_tet_mesh(xs, ys, zs, keep, mirrored=mirrored)
    mesh = TetMesh(nodes, tets, density, young, poisson)
    tol = 1e-9 * max(L, W, H)
    tip = np.nonzero(np.abs(nodes[:, 0] - L) < tol)[0]
    d = nodes[tip, 1] ** 2 + (nodes[tip, 2] - 0.0) ** 2
    mesh.node_sets = {
        "root": np.nonzero(np.abs(nodes[:, 0]) < tol)[0],
        "tip": tip,
        "tip_center": tip[np.abs(d - d.min()) < tol * tol + 1e-12 * H * H],
        "web": np.nonzero((nodes[:, 0] >= x0 - tol) & (nodes[:, 0] <= x1 + tol))[0],
        "distal": np.nonzero(nodes[:, 0] >= x1 - tol)[0],
    }
    mesh.meta = {
        "kind": "flexure",
        "block_length": L,
        "block_width": W,
        "block_height": H,
        "notch_start": x0,
        "notch_length": notch_length,
        "web_thickness": t,
        "web_offset": web_offset,
        "notch_depth_bottom": 0.5 * H - 0.5 * t + web_offset,
        "notch_depth_top": 0.5 * H - 0.5 * t - web_offset,
        "h_block": h_block,
        "n_web_z": n_web_z,
        "n_notch_x": n_notch_x,
        "ny": ny,
        "split": "kuhn6_mirrored" if mirrored else "kuhn6",
        "density": density,
        "young": young,
        "poisson": poisson,
    }
    return mesh


# ----------------------------------------------------------------------------------------------
# Assembler
# ----------------------------------------------------------------------------------------------


class NumpyTetAssembler:
    """Total-Lagrangian tet4 assembly for StVK or compressible neo-Hookean material.

    Energies per unit rest volume (lambda, mu = Lame parameters from mesh.young, mesh.poisson):
      StVK:        Psi = mu E:E + lambda/2 tr(E)^2,   E = (F^T F - I)/2,  P = F S,
                   S = lambda tr(E) I + 2 mu E
      neo_hookean: Psi = mu/2 (tr(F^T F) - 3) - mu ln J + lambda/2 (ln J)^2,  J = det F,
                   P = mu (F - F^{-T}) + lambda ln J F^{-T}
    Both linearise to isotropic linear elasticity with (lambda, mu) at F = I.

    Element internal force: f_a = V P g_a (g_a = rest gradient of shape function a).
    Element tangent: K_{ai,bk} = V A_{ijkl} g_aj g_bl with A = dP/dF (exact; includes the
    geometric/initial-stress term delta_ik S_lj for StVK).
    """

    def __init__(self, mesh: TetMesh, material: str = "stvk"):
        if material not in ("stvk", "neo_hookean"):
            raise ValueError(f"unknown material {material!r}")
        self.mesh = mesh
        self.material = material
        self.lam, self.mu = mesh.lame
        self.n_nodes = mesh.num_nodes
        self.n_dof = 3 * self.n_nodes
        tets = mesh.tets
        X = mesh.nodes[tets]  # (E,4,3)
        Dm = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=-1)  # columns
        self.vol = np.linalg.det(Dm) / 6.0
        if np.any(self.vol <= 0):
            raise ValueError("mesh has non-positive tet volumes")
        Dm_inv = np.linalg.inv(Dm)  # (E,3,3); F = Ds Dm_inv, Ds columns x_a - x_0
        G = np.empty((tets.shape[0], 4, 3))
        G[:, 1:, :] = Dm_inv  # g_a (a=1..3) = row a-1 of Dm_inv
        G[:, 0, :] = -Dm_inv.sum(axis=1)
        self.G = G
        self.X_flat = mesh.nodes.reshape(-1).copy()

        # sparse pattern for 12x12 element blocks
        dofs = (3 * tets[:, :, None] + np.arange(3)[None, None, :]).reshape(-1, 12)  # (E,12)
        self._dofs = dofs
        rows = np.broadcast_to(dofs[:, :, None], (dofs.shape[0], 12, 12)).ravel()
        cols = np.broadcast_to(dofs[:, None, :], (dofs.shape[0], 12, 12)).ravel()
        key = rows.astype(np.int64) * self.n_dof + cols
        ukey, inv = np.unique(key, return_inverse=True)
        self._scatter = inv.ravel()
        self._nnz = ukey.size
        r = ukey // self.n_dof
        self._indices = (ukey % self.n_dof).astype(np.int32)
        self._indptr = np.zeros(self.n_dof + 1, dtype=np.int64)
        np.add.at(self._indptr, r + 1, 1)
        self._indptr = np.cumsum(self._indptr).astype(np.int32)

        # consistent mass: M_e = rho V / 20 (1 + delta_ab) I3
        me = np.ones((4, 4)) + np.eye(4)
        Me = np.einsum("e,ab,ik->eaibk", mesh.density * self.vol / 20.0, me, np.eye(3)).reshape(-1, 12, 12)
        self.mass_matrix = self.assemble_blocks(Me)
        self.lumped_mass = np.asarray(self.mass_matrix.sum(axis=1)).ravel()  # = rho V/4 per tet node
        self._K0 = None

    # -- assembly helpers ------------------------------------------------------------------
    def assemble_blocks(self, Ke: np.ndarray) -> sp.csr_matrix:
        """Assemble (E,12,12) element blocks (DOF order a*3+i) into a (3N,3N) CSR matrix."""
        data = np.bincount(self._scatter, weights=Ke.reshape(-1), minlength=self._nnz)
        return sp.csr_matrix((data, self._indices.copy(), self._indptr.copy()), shape=(self.n_dof, self.n_dof))

    def assemble_vector(self, fe: np.ndarray) -> np.ndarray:
        """Assemble (E,4,3) element vectors into (3N,)."""
        return np.bincount(self._dofs.ravel(), weights=fe.reshape(-1), minlength=self.n_dof)

    def deformation_gradient(self, x_flat: np.ndarray, element_ids=None) -> np.ndarray:
        x = np.asarray(x_flat, dtype=np.float64).reshape(-1, 3)
        tets, G = (
            (self.mesh.tets, self.G) if element_ids is None else (self.mesh.tets[element_ids], self.G[element_ids])
        )
        return np.einsum("eai,eaj->eij", x[tets], G)

    # -- energy / force / tangent ----------------------------------------------------------
    def evaluate(self, x_flat: np.ndarray, *, need_tangent: bool = True):
        """Return (W, f_int, K_T) at current positions x_flat (3N,) w.r.t. rest mesh.nodes.

        K_T is None when need_tangent is False.  For neo-Hookean, raises FloatingPointError if
        any element has det F <= 0.
        """
        x = np.asarray(x_flat, dtype=np.float64).reshape(-1, 3)
        We, fe, K = self.element_terms(x[self.mesh.tets], None, need_tangent=need_tangent)
        W = float(np.sum(We))
        f_int = self.assemble_vector(fe)
        if not need_tangent:
            return W, f_int, None
        K_T = self.assemble_blocks(K.reshape(-1, 12, 12))
        return W, f_int, K_T

    def element_terms(self, x_elem: np.ndarray, element_ids=None, *, need_tangent: bool = True):
        """Per-element energy, force and tangent on an element subset (no global assembly).

        Args:
            x_elem: current positions of the element nodes, (E_sel, 4, 3).
            element_ids: indices of the selected elements (``None``: all, in mesh order).
        Returns:
            ``(W_e (E_sel,), f_e (E_sel, 4, 3), K_e (E_sel, 4, 3, 4, 3) or None)``; summing/assembling
            them over all elements gives exactly :meth:`evaluate` (which is implemented with this).
        """
        G, V = (self.G, self.vol) if element_ids is None else (self.G[element_ids], self.vol[element_ids])
        F = np.einsum("eai,eaj->eij", np.asarray(x_elem, dtype=np.float64), G)
        lam, mu = self.lam, self.mu
        I3 = np.eye(3)
        if self.material == "stvk":
            C = np.einsum("eki,ekj->eij", F, F)
            E = 0.5 * (C - I3)
            trE = np.trace(E, axis1=1, axis2=2)
            S = lam * trE[:, None, None] * I3 + 2.0 * mu * E
            P = np.einsum("eik,ekj->eij", F, S)
            psi = mu * np.einsum("eij,eij->e", E, E) + 0.5 * lam * trE**2
        else:
            J = np.linalg.det(F)
            if np.any(J <= 0):
                raise FloatingPointError(f"inverted element(s): min det F = {J.min():.3e}")
            FinvT = np.transpose(np.linalg.inv(F), (0, 2, 1))
            lnJ = np.log(J)
            P = mu * (F - FinvT) + (lam * lnJ)[:, None, None] * FinvT
            psi = 0.5 * mu * (np.einsum("eij,eij->e", F, F) - 3.0) - mu * lnJ + 0.5 * lam * lnJ**2
        We = V * psi
        fe = V[:, None, None] * np.einsum("eij,eaj->eai", P, G)  # (E,4,3)
        if not need_tangent:
            return We, fe, None

        GG = np.einsum("eaj,ebj->eab", G, G)  # g_a . g_b
        if self.material == "stvk":
            FG = np.einsum("eij,eaj->eai", F, G)  # (F g_a)_i
            GSG = np.einsum("eaj,ejl,ebl->eab", G, S, G)  # g_a . S g_b  (S symmetric)
            FFt = np.einsum("eik,ejk->eij", F, F)
            K = (
                np.einsum("eab,ik->eaibk", GSG, I3)  # geometric (initial stress)
                + lam * np.einsum("eai,ebk->eaibk", FG, FG)
                + mu * np.einsum("eab,eik->eaibk", GG, FFt)
                + mu * np.einsum("ebi,eak->eaibk", FG, FG)
            )
        else:
            HG = np.einsum("eij,eaj->eai", FinvT, G)  # (F^{-T} g_a)_i
            K = (
                mu * np.einsum("eab,ik->eaibk", GG, I3)
                + (mu - lam * lnJ)[:, None, None, None, None] * np.einsum("ebi,eak->eaibk", HG, HG)
                + lam * np.einsum("eai,ebk->eaibk", HG, HG)
            )
        K *= V[:, None, None, None, None]
        return We, fe, K

    def linear_stiffness(self) -> sp.csr_matrix:
        """K_T at the rest configuration (= linear elastic stiffness with lambda, mu). Cached."""
        if self._K0 is None:
            self._K0 = self.evaluate(self.X_flat, need_tangent=True)[2]
        return self._K0

    # -- per-element kinematic measures ----------------------------------------------------
    def element_rotations(self, x_flat: np.ndarray, element_ids=None, reference_flat=None) -> np.ndarray:
        """Rotation angle [rad] of the polar factor R of F per element (F = R U).

        F is relative to mesh.nodes; with reference_flat the relative gradient
        F_rel = F(x) F(reference)^{-1} is used instead (rotation since that reference).
        Returns (E_sel,) angles in [0, pi].
        """
        F = self.deformation_gradient(x_flat, element_ids)
        if reference_flat is not None:
            F = F @ np.linalg.inv(self.deformation_gradient(reference_flat, element_ids))
        U, _, Vt = np.linalg.svd(F)
        d = np.sign(np.linalg.det(U @ Vt))
        U[:, :, 2] *= d[:, None]
        R = U @ Vt
        c = 0.5 * (np.trace(R, axis1=1, axis2=2) - 1.0)
        return np.arccos(np.clip(c, -1.0, 1.0))

    def element_green_strain(self, x_flat: np.ndarray, element_ids=None) -> np.ndarray:
        """Frobenius norm of the Green-Lagrange strain E per element."""
        F = self.deformation_gradient(x_flat, element_ids)
        E = 0.5 * (np.einsum("eki,ekj->eij", F, F) - np.eye(3))
        return np.sqrt(np.einsum("eij,eij->e", E, E))


# ----------------------------------------------------------------------------------------------
# WP-D slot: WarpTetAssembler
# ----------------------------------------------------------------------------------------------
# A GPU implementation of TangentAssembler (Warp kernel per tet: StVK / stable neo-Hookean element
# force and full 12x12 tangent, float64 accumulation into a warp.sparse.BsrMatrix with a cached
# sparsity pattern, element-subset evaluation for ECSW, polar-decomposition rotation kernel) is
# added here by work package WP-D and verified against NumpyTetAssembler. Until then only the numpy
# assembler exists; nothing in the controller depends on the concrete class.


class Material(enum.Enum):
    """Hyperelastic material models of the tet4 assemblers (values are the assembler ``material`` strings)."""

    STVK = "stvk"
    """Saint Venant-Kirchhoff: ``Psi = mu E:E + lambda/2 tr(E)^2``."""
    COMPRESSIBLE_NEO_HOOKEAN = "neo_hookean"
    """Compressible log-J neo-Hookean: ``Psi = mu/2 (tr C - 3) - mu ln J + lambda/2 (ln J)^2``.

    Not the Smith et al. "stable" neo-Hookean model; undefined for inverted elements (``J <= 0``)."""


def default_assembler(mesh: TetMesh, material: str | Material = Material.STVK, *, device=None):
    """The default :class:`TangentAssembler` for ``mesh``.

    :class:`rerom.assembly_warp.WarpTetAssembler` on a CUDA device when one is available (or on ``device``
    when given), returning host SciPy tangents (``tangent_format="csc"``) so the host eigensolvers can use
    them; otherwise :class:`NumpyTetAssembler`.
    """
    material = Material(material).value
    import warp as wp  # noqa: PLC0415

    if device is None and wp.is_cuda_available():
        device = "cuda:0"
    if device is None:
        return NumpyTetAssembler(mesh, material)
    from .assembly_warp import WarpTetAssembler  # noqa: PLC0415

    return HostTangentAssembler(WarpTetAssembler.from_mesh(mesh, material, device=device))


class HostTangentAssembler:
    """Wraps a device assembler (``evaluate(..., tangent_format=...)``) so ``evaluate`` returns a SciPy CSC tangent.

    All other attributes are forwarded. ``inner`` is the wrapped assembler.
    """

    def __init__(self, inner, tangent_format: str = "csc"):
        self.inner = inner
        self.tangent_format = tangent_format

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @property
    def mass_matrix(self):
        """Consistent mass (SciPy CSR)."""
        return self.inner.mass_matrix

    @property
    def mesh(self):
        """Rest mesh of the wrapped assembler."""
        return self.inner.mesh

    def element_terms(self, x_elem, element_ids=None, *, need_tangent: bool = True):
        """Per-element ``(W_e, f_e, K_e)`` (host arrays)."""
        return self.inner.element_terms(x_elem, element_ids, need_tangent=need_tangent)

    def element_rotations(self, x_flat, element_ids=None, reference_flat=None):
        """Per-element rotation angles [rad] (host array)."""
        return self.inner.element_rotations(x_flat, element_ids, reference_flat=reference_flat)

    def evaluate(self, x_flat, *, need_tangent: bool = True):
        """``(W, f_int, K_T)`` with a host sparse ``K_T``."""
        return self.inner.evaluate(x_flat, need_tangent=need_tangent, tangent_format=self.tangent_format)

    def linear_stiffness(self):
        """Rest tangent (host sparse)."""
        return self.inner.linear_stiffness(tangent_format=self.tangent_format)
