# ruff: noqa: TID253, B007, PLC0415  (moved as-is from the experiment package)
"""WP4 (v2): Craig-Bampton ROM / re-ROM of thin-solid (C3D8I) shell panels coupled to CalculiX.

Pieces
------
* :class:`PanelCcxModel` -- :class:`rerom.ccx.CcxModel` plus the decks WP4 needs: linear or
  NLGEOM static with *RIGID BODY pads for the released clamps (the same RBE2 pads the ROM keeps), and
  an ``evaluate`` that can also return the total strain energy (``*EL PRINT, TOTALS=ONLY`` ELSE).
* :class:`CcxAssembler` -- adapter so a :class:`CcxModel` satisfies the ``ReROMController``
  assembler interface (``mass_matrix``, ``evaluate(x_flat, need_tangent) -> (W, f_int, K_T)``) in the
  3-DOF solid-node layout.  The ccx 6N layout ``q[6 i:6 i + 6] = (bottom, top)`` *is* the 3-DOF layout
  of solid node ``2 i`` / ``2 i + 1``, so no permutation is needed.  W is ccx's ELSE (7 digits) when
  ``energy=True``; otherwise NaN.
* :class:`SolidCB` -- fixed-interface Craig-Bampton reduction on 3-DOF solid nodes: weld rows
  eliminated exactly (dependent = welded node b), each clamp pad (both solid nodes of its shell nodes)
  an RBE2 patch of a six-DOF frame, constraint modes ``Psi = -K_ii^-1 K_ib`` and fixed-interface
  eigenmodes of ``(K_ii, M_ii)`` with ONE SuperLU factorization of ``K_ii`` (ARPACK shift-invert at
  sigma = 0), plus optional static Ritz vectors ``K_ii^-1 F_i`` (interfaces held), M-orthogonalised
  against the eigenmodes.  Reduced coordinates: ``[6 x frames, modes]`` (Newton's
  ``ModalGeneratorCraigBampton`` order).  At rest it reproduces ``ModalGeneratorCraigBamptonFEM``
  (tests/test_panel_rerom.py).  About a deformed state it is called with the deformed node positions
  and the tangent ``K_T`` (total Lagrangian: ``M`` is the reference mass and does not change).
* :class:`PanelScene` -- the Newton scene of the HMG example (one elastic body, one FIXED joint per
  clamp at its interface frame, joints enabled per configuration) built from a reduction, with a
  float64 read-out of the nodal displacement from the joint state.
* :class:`PanelReROM` -- re-ROM driver: residual-mismatch trigger evaluated with ccx on the recovered
  field (plain or director-rotated), basis rebuild about the recovered state (constraint modes and
  fixed-interface modes from ``K_T``, 2 static Ritz vectors, pre-stress ``g0 = Phi^T f_int(u_ref)``
  as modal ``joint_f``), Newton model rebuild and momentum-consistent state transfer.

Why a model rebuild instead of ``ReROMController``'s in-place rewrite: the Craig-Bampton body's joint
endpoints are the basis' interface attachment frames, which the WP-B controller refuses to re-derive
(``NotImplementedError`` in ``ReROMController.__init__``), and its re-basis is a free/fixed-node
eigenbasis, not a Craig-Bampton basis.  A rebuild is exact and its cost is reported separately
("rewrite").  The adapter is still provided (and tested with the controller on a free plate).

Sign conventions: f_int = dW/du, M a + f_int = f_ext; Newton modal ``joint_f`` enters as
``k q = joint_f + ...`` (verified in WP-B, ``test_joint_f_sign``), so the pre-stress is
``joint_f[modal rows] = -g0``.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .ccx import CcxModel, parse_dat_blocks

__all__ = [
    "CCX_VOIGT_ORDER",
    "CcxAssembler",
    "PanelCcxModel",
    "PanelReROM",
    "PanelScene",
    "SolidCB",
    "director_rotated",
    "interface_list",
    "pad_solid_nodes",
    "rigid_fit",
]

# Verified by tests/test_panel_rerom.py::test_ccx_voigt_order (uniform-shear patch test):
# ccx *EL PRINT E columns are exx, eyy, ezz, exy, exz, eyz with TENSORIAL shear, Green-Lagrange under NLGEOM.
CCX_VOIGT_ORDER = ("xx", "yy", "zz", "xy", "zx", "yz")


def _skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rigid_rows(points, p):
    """(3 n, 6): u_j = t + theta x (x_j - p)."""
    return np.vstack([np.hstack((np.eye(3), -_skew(x - p))) for x in points])


def rigid_fit(points, disp, p):
    """Least-squares rigid frame motion (t at p, theta) of a patch displacement."""
    A = rigid_rows(points, p)
    return np.linalg.lstsq(A, disp.reshape(-1), rcond=None)[0]


def rigidify_pads(rest_points, u, pads):
    """Replace the displacement of each pad (solid node indices) by its best-fit FINITE rigid motion
    (Kabsch).  Linearised RBE2 increments about successive references leave O(dtheta^2 r) non-rigidity in
    the released pads (~1-6 um at 0.01-0.02 rad), which the stiff pad elements turn into spurious internal
    forces; the ccx reference uses exact *RIGID BODY pads."""
    out = np.asarray(u, float).reshape(-1, 3).copy()
    for nodes in pads:
        X = rest_points[nodes]
        x = X + out[nodes]
        cX, cx = X.mean(0), x.mean(0)
        H = (X - cX).T @ (x - cx)
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        out[nodes] = (X - cX) @ R.T + cx - X
    return out.reshape(-1)


def pad_solid_nodes(shell_rows):
    """Solid node indices (bottom 2 i, top 2 i + 1) of shell node rows."""
    r = np.asarray(shell_rows, np.int64)
    return np.sort(np.concatenate([2 * r, 2 * r + 1]))


def interface_list(model: CcxModel, centre_from="package"):
    """One interface per clamp: dict(name, solid nodes (both layers), position)."""
    out = []
    for c in model.mesh.clamps:
        nodes = pad_solid_nodes(c["nodes"])
        pos = (
            np.asarray(c["centre"], float)
            if ("centre" in c and centre_from == "package")
            else model.mesh.points[c["nodes"]].mean(axis=0)
        )
        out.append({"name": c["clamp_id"], "nodes": nodes, "position": pos, "active": c.get("active", {})})
    return out


def director_rotated(model: CcxModel, q_disp, q_ref=None, keep_stretch=True):
    """Director-rotated recovery of a field that is linear about the reference state ``q_ref``.

    Keeps the mid-surface displacement of ``q_disp``.  With d_ref = n + (top - bot)_ref / t the reference
    director and d_lin = n + (top - bot)/t the linearly recovered one, the fibre is ROTATED onto d_lin
    instead of being stretched by the superposed (linear) rotation increment, which adds the spurious
    length |d_lin| - e_ref.d_lin ~ |d_ref| dtheta^2 / 2 (e_ref = d_ref / |d_ref|):
        d_new = s e_lin,  e_lin = d_lin / |d_lin|,  s = e_ref . d_lin   (keep_stretch=True)
    i.e. the fibre stretch component along the reference fibre (the physical Poisson thickness change
    of the increment) is kept.  ``keep_stretch=False``: s = |d_ref| (no thickness change of the increment).
    ``q_ref=None`` means the rest state (d_ref = n)."""
    u_mid, dd = model.mid_and_director(q_disp)
    n = model.normals
    d_ref = n if q_ref is None else n + model.mid_and_director(q_ref)[1]
    e_ref = d_ref / np.linalg.norm(d_ref, axis=1)[:, None]
    d_lin = n + dd
    unit = d_lin / np.linalg.norm(d_lin, axis=1)[:, None]
    s = np.einsum("ij,ij->i", e_ref, d_lin) if keep_stretch else np.linalg.norm(d_ref, axis=1)
    h = 0.5 * model.thickness[:, None]
    out = np.empty((model.mesh.n_nodes, 2, 3))
    out[:, 0] = u_mid - h * (s[:, None] * unit - n)
    out[:, 1] = u_mid + h * (s[:, None] * unit - n)
    return out.reshape(-1)


# ----------------------------------------------------------------------------------------------
# CalculiX decks
# ----------------------------------------------------------------------------------------------
class PanelCcxModel(CcxModel):
    """CcxModel with rigid pads, linear/NLGEOM statics and energy output."""

    def __init__(self, mesh, workdir, **kw):
        super().__init__(mesh, workdir, **kw)
        # clamp centres for rigid-body reference nodes
        for c in mesh.clamps:
            c.setdefault("centre", mesh.points[c["nodes"]].mean(axis=0))

    def _rigid_cards(self, clamp_ids):
        """*RIGID BODY per released clamp pad (all solid nodes of the 3x3 shell patch)."""
        L = []
        nid = 2 * self.mesh.n_nodes
        extra_nodes = []
        for c in self.mesh.clamps:
            if c["clamp_id"] not in clamp_ids:
                continue
            ref, rot = nid + 1, nid + 2
            nid += 2
            p = np.asarray(c["centre"], float)
            extra_nodes.append(f"{ref}, {p[0]:.13e}, {p[1]:.13e}, {p[2]:.13e}")
            extra_nodes.append(f"{rot}, {p[0]:.13e}, {p[1]:.13e}, {p[2]:.13e}")
            sn = pad_solid_nodes(c["nodes"]) + 1
            L += [f"*NSET, NSET=PAD_{c['clamp_id']}"] + [
                ", ".join(str(v) for v in sn[i : i + 8]) for i in range(0, len(sn), 8)
            ]
            L += [f"*RIGID BODY, NSET=PAD_{c['clamp_id']}, REF NODE={ref}, ROT NODE={rot}"]
        return (["*NODE", *extra_nodes] if extra_nodes else []) + L

    def static(
        self, fixed_dofs, g_vec, *, nlgeom=True, rigid_clamps=(), welds=True, n_incr=10, controls=True, label="static"
    ):
        """Gravity static (linear or NLGEOM); fixed 6N DOFs held at zero, released pads rigid."""
        fixed = np.unique(np.asarray(fixed_dofs, np.int64))
        L = [f"** {label}", *self._mesh_cards(welds=welds), *self._rigid_cards(set(rigid_clamps))]
        g = np.asarray(g_vec, float)
        gn = np.linalg.norm(g)
        L += ["*BOUNDARY"]
        L += [f"{2 * (k // 6) + (k % 6) // 3 + 1}, {k % 3 + 1}, {k % 3 + 1}" for k in fixed]
        if nlgeom:
            L += ["*STEP, NLGEOM, INC=1000", "*STATIC", f"{1.0 / n_incr}, 1., 1e-6, {1.0 / n_incr}"]
        else:
            L += ["*STEP", "*STATIC"]
        L += ["*DLOAD", f"EALL, GRAV, {gn:.13e}, {g[0] / gn:.13e}, {g[1] / gn:.13e}, {g[2] / gn:.13e}"]
        if controls and nlgeom:
            # tighten the force-residual criterion (default 0.5 % of the mean force) to 1e-6
            L += ["*CONTROLS, PARAMETERS=FIELD", "1e-6, 1e-6, 1e-8, , 0.02, 1e-5, 1e-3, 1e-8"]
        L += ["*NODE PRINT, NSET=NALL, TOTALS=NO", "U", "*END STEP"]
        job, dt = self._run(label, L)
        blocks = parse_dat_blocks(self.workdir / f"{job}.dat")
        U = blocks["displacements"][-1][1]
        u = np.zeros((2 * self.mesh.n_nodes, 3))
        m = U[:, 0].astype(int) <= 2 * self.mesh.n_nodes
        u[U[m, 0].astype(int) - 1] = U[m, 1:4]
        sta = (self.workdir / f"{job}.sta").read_text() if (self.workdir / f"{job}.sta").exists() else ""
        n_inc = sum(1 for ln in sta.splitlines() if ln.strip()[:1].isdigit())
        self._cleanup(job)
        return {"u": u.reshape(-1), "wall": dt, "job": job, "increments": n_inc}

    def release_dynamic(
        self,
        active_first,
        active_final,
        g_vec,
        *,
        t_end,
        dt,
        damping=(0.0, 0.0),
        alpha=-0.05,
        welds=True,
        out_every=10,
        n_incr_static=10,
        label="dyn",
    ):
        """Two-step ccx history: (1) NLGEOM static under gravity with the clamps ``active_first`` held
        (released-to-be pads rigid with their reference/rotation nodes fixed); (2) *DYNAMIC (HHT, DIRECT,
        NLGEOM) from that state with only ``active_final`` held, the other pads rigid and free.
        ``damping = (alpha_M, beta_K)``: Rayleigh damping (*DAMPING material card).
        Returns t (T,), u (T, 6N) of step 2 (every ``out_every`` increments) and the step-1 state."""
        ids = {c["clamp_id"] for c in self.mesh.clamps}
        released = sorted(ids - set(active_final))
        L = [f"** {label}", *self._mesh_cards(welds=welds)]
        if damping != (0.0, 0.0):
            # material cards: "*DENSITY", value -> insert damping after the value line
            L2 = []
            i = 0
            while i < len(L):
                L2.append(L[i])
                if L[i].startswith("*DENSITY"):
                    L2.append(L[i + 1])
                    L2.append(f"*DAMPING, ALPHA={damping[0]:.6e}, BETA={damping[1]:.6e}")
                    i += 2
                    continue
                i += 1
            L = L2
        L += self._rigid_cards(set(released))
        n2 = 2 * self.mesh.n_nodes
        ref_nodes = {cid: (n2 + 2 * k + 1, n2 + 2 * k + 2) for k, cid in enumerate(released)}
        fix_final = self.clamp_dofs(list(active_final))
        bc = lambda dofs: [f"{2 * (k // 6) + (k % 6) // 3 + 1}, {k % 3 + 1}, {k % 3 + 1}" for k in dofs]  # noqa: E731
        g = np.asarray(g_vec, float)
        gn = np.linalg.norm(g)
        grav = ["*DLOAD", f"EALL, GRAV, {gn:.13e}, {g[0] / gn:.13e}, {g[1] / gn:.13e}, {g[2] / gn:.13e}"]
        L += ["*BOUNDARY", *bc(fix_final)]
        L += [
            "*STEP, NLGEOM, INC=1000",
            "*STATIC",
            f"{1.0 / n_incr_static}, 1., 1e-6, {1.0 / n_incr_static}",
            "*BOUNDARY",
        ]
        for cid in released:
            if cid in set(active_first):
                r, q = ref_nodes[cid]
                L += [f"{r}, 1, 3", f"{q}, 1, 3"]
        L += [*grav, "*NODE PRINT, NSET=NALL", "U", "*EL PRINT, ELSET=EALL, TOTALS=ONLY", "ELSE", "*END STEP"]
        L += [
            "*STEP, NLGEOM, INC=1000000",
            f"*DYNAMIC, ALPHA={alpha}, DIRECT",
            f"{dt}, {t_end}",
            "*BOUNDARY, OP=NEW",
            *bc(fix_final),
            *grav,
        ]
        L += [f"*NODE PRINT, NSET=NALL, FREQUENCY={out_every}", "U", "*END STEP"]
        job, wall = self._run(label, L)
        blocks = parse_dat_blocks(self.workdir / f"{job}.dat")
        times, U = [], []
        for t, arr in blocks["displacements"]:
            u = np.zeros((n2, 3))
            m = arr[:, 0].astype(int) <= n2
            u[arr[m, 0].astype(int) - 1] = arr[m, 1:4]
            times.append(t)
            U.append(u.reshape(-1))
        self._cleanup(job)
        # the first block is the end of step 1 (static, total time 1); step-2 times continue from 1
        t = np.array(times)
        return {"t": t, "u": np.array(U), "wall": wall, "job": job}

    def evaluate_ext(self, q_disp, tangent=True, strains=True, energy=False, nlgeom=True):
        """``evaluate`` plus optional total strain energy W (ccx ELSE summed, 7 digits)."""
        if not energy:
            return self.evaluate(q_disp, tangent=tangent, strains=strains, nlgeom=nlgeom)
        q = np.asarray(q_disp, float).reshape(-1, 3)
        L = ["** prescribed-state evaluation + energy", *self._mesh_cards(welds=False)]
        L += ["*STEP" + (", NLGEOM" if nlgeom else ""), "*STATIC", "1., 1.", "*BOUNDARY"]
        L += [f"{k + 1}, {d + 1}, {d + 1}, {q[k, d]:.13e}" for k in range(len(q)) for d in range(3)]
        L += ["*NODE PRINT, NSET=NALL, TOTALS=NO", "RF"]
        if strains:
            L += ["*EL PRINT, ELSET=EALL", "E"]
        L += ["*EL PRINT, ELSET=EALL, TOTALS=ONLY", "ELSE"]
        L += ["*END STEP"]
        if tangent:
            L += ["*STEP, PERTURBATION", "*FREQUENCY, SOLVER=MATRIXSTORAGE", "*BOUNDARY, OP=NEW", "*END STEP"]
        job, dt = self._run("evalW", L)
        txt = (self.workdir / f"{job}.dat").read_text().splitlines()
        W = float("nan")
        for i, ln in enumerate(txt):
            if "total internal energy" in ln:
                for j in range(i + 1, min(i + 4, len(txt))):
                    v = txt[j].split()
                    if v:
                        W = float(v[-1].replace("D", "E"))
                        break
        blocks = parse_dat_blocks(self.workdir / f"{job}.dat")
        rf = blocks["forces"][-1][1]
        f = np.zeros((2 * self.mesh.n_nodes, 3))
        f[rf[:, 0].astype(int) - 1] = rf[:, 1:4]
        out = {"f_int": f.reshape(-1), "wall": dt, "job": job, "W": W}
        if strains and "strains" in blocks:
            e = blocks["strains"][-1][1]
            E = np.zeros((len(self.mesh.quads), int(e[:, 1].max()), 6))
            E[e[:, 0].astype(int) - 1, e[:, 1].astype(int) - 1] = e[:, 2:8]
            out["strain"] = E
        if tangent:
            K, _, real, target, _lab = self._read_storage(job, need_mass=False)
            out["K_T"], _ = self._condense(K, None, real, target)
        self._cleanup(job)
        return out


class CcxAssembler:
    """ReROMController assembler interface on a CcxModel (3-DOF solid-node layout).

    ``evaluate(x_flat, need_tangent)`` takes POSITIONS (rest + displacement, as the WP-A assembler
    does) and returns ``(W, f_int, K_T)``; ``W`` comes from ccx's ELSE output when ``energy=True``
    (one extra print, 7 digits) and is NaN otherwise -- the controller only needs W for
    ``nonlinear_strain_energy`` and the ``EnergyMismatchTrigger`` (which also needs a tet mesh and
    is therefore not usable with this adapter; use the residual trigger of :class:`PanelReROM`).
    """

    def __init__(self, model: PanelCcxModel, mass_matrix, energy=False):
        self.model = model
        self.mass_matrix = sp.csr_matrix(mass_matrix)
        self.energy = bool(energy)
        self.rest = model.rest_q.copy()
        self.calls = 0

    def evaluate(self, x_flat, need_tangent=True):
        self.calls += 1
        u = np.asarray(x_flat, float).reshape(-1) - self.rest
        if self.energy:
            r = self.model.evaluate_ext(u, tangent=need_tangent, strains=False, energy=True)
        else:
            r = self.model.evaluate(u, tangent=need_tangent, strains=False)
            r["W"] = float("nan")
        return r["W"], r["f_int"], (r.get("K_T") if need_tangent else None)


# ----------------------------------------------------------------------------------------------
# Craig-Bampton on solid nodes
# ----------------------------------------------------------------------------------------------
@dataclass
class CBResult:
    T: np.ndarray  # (n_dof, 6 nI + m) full recovery (3-DOF solid layout)
    Mr: np.ndarray
    Kr: np.ndarray
    interface_positions: np.ndarray
    interface_names: list
    n_frames: int
    n_modes: int
    n_ritz: int
    eigenvalues: np.ndarray
    timings: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


class SolidCB:
    """Fixed-interface Craig-Bampton on 3-DOF nodes with exact weld elimination and RBE2 pads."""

    def __init__(self, points, weld_C, interfaces):
        self.points = np.asarray(points, float)  # (n_nodes, 3) positions used for RBE2 lever arms
        self.n = 3 * len(self.points)
        self.interfaces = interfaces
        self._build_S(sp.csr_matrix(weld_C) if weld_C is not None else None)

    def _build_S(self, C):
        n = self.n
        dep_coef = {}
        if C is not None and C.shape[0]:
            for r in range(C.shape[0]):
                cols = C.indices[C.indptr[r] : C.indptr[r + 1]]
                vals = C.data[C.indptr[r] : C.indptr[r + 1]]
                k = int(np.argmax(vals == 1.0))
                d = int(cols[k])
                if d in dep_coef:
                    raise ValueError("DOF dependent in two weld rows")
                dep_coef[d] = [(int(c), -float(v)) for j, (c, v) in enumerate(zip(cols, vals, strict=False)) if j != k]
        for d, terms in dep_coef.items():
            if any(c in dep_coef for c, _ in terms):
                raise ValueError("chained weld rows are not supported")
        pad = {}
        for k, itf in enumerate(self.interfaces):
            for node in itf["nodes"]:
                for a in range(3):
                    dof = 3 * int(node) + a
                    if dof in dep_coef:
                        raise ValueError("a pad DOF is weld-dependent; not supported")
                    pad[dof] = (k, a, int(node))
        interior = np.array([d for d in range(n) if d not in dep_coef and d not in pad], np.int64)
        nI = len(self.interfaces)
        ni = len(interior)
        col_of = {int(d): j for j, d in enumerate(interior)}
        # expression of every independent (non-dependent) DOF in [interior; frames]
        rows, cols, vals = [], [], []
        A_cache = {}

        def expr(dof):
            if dof in col_of:
                return [(col_of[dof], 1.0)]
            k, a, node = pad[dof]
            key = (k, node)
            if key not in A_cache:
                A_cache[key] = np.hstack((np.eye(3), -_skew(self.points[node] - self.interfaces[k]["position"])))
            A = A_cache[key]
            return [(ni + 6 * k + j, A[a, j]) for j in range(6) if A[a, j] != 0.0]

        for dof in range(n):
            if dof in dep_coef:
                acc = {}
                for c, v in dep_coef[dof]:
                    for cc, vv in expr(c):
                        acc[cc] = acc.get(cc, 0.0) + v * vv
                items = acc.items()
            else:
                items = expr(dof)
            for c, v in items:
                rows.append(dof)
                cols.append(c)
                vals.append(v)
        self.S = sp.csr_matrix((vals, (rows, cols)), shape=(n, ni + 6 * nI))
        self.ni, self.nI = ni, nI

    def reduce(
        self, K, M, n_modes, ritz_loads=(), ritz_drop_tol=1e-6, eig_tol=1e-10, names=None, rigid_project_tol=1e-9
    ):
        """``rigid_project_tol``: about a stressed state K_T R_rot = (omega x f_int) != 0 (relative ~1e-6 at 5 g),
        which Newton's ModalGeneratorCraigBampton rejects above 1e-7; above this tolerance the reduced
        tangent is replaced by P^T K_r P (P = M-orthogonal projector onto the elastic complement of the six
        rigid motions of the current positions).  The change is of the order of the residual itself."""
        tm = {}
        t0 = time.perf_counter()
        S = self.S
        Kz = (S.T @ (sp.csr_matrix(K) @ S)).tocsc()
        Mz = (S.T @ (sp.csr_matrix(M) @ S)).tocsc()
        Kz = 0.5 * (Kz + Kz.T)
        Mz = 0.5 * (Mz + Mz.T)
        ni = self.ni
        Kii = Kz[:ni, :ni].tocsc()
        Kib = Kz[:ni, ni:].toarray()
        Mii = Mz[:ni, :ni].tocsc()
        tm["projection_in"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        lu = spla.splu(Kii, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
        tm["factorization"] = time.perf_counter() - t0
        diagU = lu.U.diagonal()
        neg = int(np.count_nonzero(diagU <= 0))
        t0 = time.perf_counter()
        Psi = -lu.solve(Kib)
        tm["constraint_modes"] = time.perf_counter() - t0
        n_ritz_req = len(ritz_loads)
        m_eig = n_modes - n_ritz_req
        t0 = time.perf_counter()
        op = spla.LinearOperator(Kii.shape, matvec=lu.solve, dtype=float)
        vals, vecs = spla.eigsh(Kii, k=m_eig + 2, M=Mii, sigma=0.0, which="LM", OPinv=op, tol=eig_tol)
        o = np.argsort(vals)
        vals, vecs = vals[o][:m_eig], vecs[:, o][:, :m_eig]
        mm = np.einsum("ij,ij->j", vecs, Mii @ vecs)
        vecs = vecs / np.sqrt(mm)
        tm["eigensolve"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        basis = [vecs]
        kept = 0
        for F in ritz_loads:
            Fz = S.T @ np.asarray(F, float)
            r = lu.solve(Fz[:ni])
            B = np.hstack(basis)
            for _ in range(2):  # twice-iterated M-Gram-Schmidt
                r = r - B @ (B.T @ (Mii @ r))
            nr = math.sqrt(max(float(r @ (Mii @ r)), 0.0))
            ref = math.sqrt(float(lu.solve(Fz[:ni]) @ (Mii @ lu.solve(Fz[:ni]))))
            if nr > ritz_drop_tol * max(ref, 1e-300):
                basis.append((r / nr)[:, None])
                kept += 1
        if kept < n_ritz_req:  # fill with more eigenmodes is not possible after the solve: keep fewer columns
            pass
        Phi = np.hstack(basis)
        tm["ritz"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        nb = 6 * self.nI
        m = Phi.shape[1]
        Tz = np.zeros((Kz.shape[0], nb + m))
        Tz[:ni, :nb] = Psi
        Tz[ni:, :nb] = np.eye(nb)
        Tz[:ni, nb:] = Phi
        Kr = Tz.T @ (Kz @ Tz)
        Mr = Tz.T @ (Mz @ Tz)
        Kr = 0.5 * (Kr + Kr.T)
        Mr = 0.5 * (Mr + Mr.T)
        T = np.asarray(S @ Tz)
        tm["projection_out"] = time.perf_counter() - t0
        pos = np.array([itf["position"] for itf in self.interfaces])
        R = np.zeros((nb + m, 6))
        for k, p in enumerate(pos):
            R[6 * k : 6 * k + 3, :3] = np.eye(3)
            R[6 * k : 6 * k + 3, 3:] = -_skew(p)
            R[6 * k + 3 : 6 * k + 6, 3:] = np.eye(3)
        rig = float(
            np.linalg.norm(Kr @ R, ord=np.inf)
            / (max(np.linalg.norm(Kr, ord=np.inf), 1) * max(np.linalg.norm(R, ord=np.inf), 1))
        )
        rig_before = rig
        if rigid_project_tol is not None and rig > rigid_project_tol:
            Kp = rigid_free_project(Kr, R, Mr)
            rel_change = float(np.linalg.norm(Kp - Kr) / np.linalg.norm(Kr))
            Kr = Kp
            rig = float(
                np.linalg.norm(Kr @ R, ord=np.inf)
                / (max(np.linalg.norm(Kr, ord=np.inf), 1) * max(np.linalg.norm(R, ord=np.inf), 1))
            )
        else:
            rel_change = 0.0
        info = {
            "negative_pivots": neg,
            "rigid_stiffness_residual": rig,
            "rigid_residual_before_projection": rig_before,
            "rigid_projection_rel_change": rel_change,
            "interior": ni,
            "ritz_kept": kept,
            "fixed_interface_hz": (np.sqrt(np.maximum(vals, 0)) / (2 * np.pi)).tolist()[:12],
        }
        self.lu, self.Kz, self.Mz, self.Tz = lu, Kz, Mz, Tz
        return CBResult(
            T=T,
            Mr=Mr,
            Kr=Kr,
            interface_positions=pos,
            interface_names=names or [itf["name"] for itf in self.interfaces],
            n_frames=self.nI,
            n_modes=m,
            n_ritz=kept,
            eigenvalues=vals,
            timings=tm,
            info=info,
        )


def rigid_free_project(Kr, R, Mr):
    """Remove the rigid-motion residual of a reduced tangent: Kr <- P^T Kr P with the M-orthogonal
    projector onto the elastic complement (used only when K_T R != 0 exceeds Newton's 1e-7 check)."""
    G = R.T @ Mr @ R
    P = np.eye(Kr.shape[0]) - R @ np.linalg.solve(G, R.T @ Mr)
    out = P.T @ Kr @ P
    return 0.5 * (out + out.T)


# ----------------------------------------------------------------------------------------------
# Newton scene
# ----------------------------------------------------------------------------------------------
def modal_damping(Mr, Kr, zeta):
    """Uniform modal damping in reduced coordinates, zero on the six rigid modes (HMG solve_newton)."""
    import scipy.linalg as la

    w, V = la.eigh(Kr, Mr)
    Phi = V[:, 6:]
    D = Mr @ Phi @ np.diag(2.0 * zeta * np.sqrt(np.maximum(w[6:], 0.0))) @ Phi.T @ Mr
    return 0.5 * (D + D.T)


class PanelScene:
    """HMG-style Newton scene of one CB elastic body with FIXED clamp joints (package frame shifted
    by ``shift`` so the float32 solver sees planform-centred coordinates)."""

    def __init__(
        self,
        cb: CBResult,
        ref_points,
        *,
        gravity,
        shift,
        zeta=1.0,
        iterations=10,
        ke=1e10,
        ka=1e9,
        device="cpu",
        sample_mass=None,
        joint_clamps=None,
    ):
        import warp as wp

        import newton

        self.newton, self.wp = newton, wp
        t0 = time.perf_counter()
        self.cb = cb
        self.shift = np.asarray(shift, float)
        self.ref_points = np.asarray(ref_points, float)  # package frame, (n_nodes, 3)
        pts_c = self.ref_points - self.shift
        pos_c = cb.interface_positions - self.shift
        D = modal_damping(cb.Mr, cb.Kr, zeta) if zeta > 0 else None
        self.gen = newton.ModalGeneratorCraigBampton(
            pos_c,
            cb.Mr,
            cb.Kr,
            pts_c,
            cb.T,
            damping_matrix=D,
            interface_names=list(cb.interface_names),
            sample_mass=sample_mass,
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            self.basis = self.gen.build()
        self.build_warnings = [str(r.message)[:200] for r in rec]
        t_basis = time.perf_counter() - t0
        t0 = time.perf_counter()
        builder = newton.ModelBuilder(gravity=tuple(float(v) for v in gravity))
        self.body = builder.add_body_elastic(xform=wp.transform_identity(), modal_basis=self.basis, label="panel")
        self.joints = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Adding a FIXED joint.*")
            for k, p in enumerate(pos_c):
                xf = wp.transform(wp.vec3(*[float(v) for v in p]), wp.quat_identity())
                self.joints.append(
                    builder.add_joint_fixed(
                        parent=-1, child=self.body, parent_xform=xf, child_xform=xf, label=cb.interface_names[k]
                    )
                )
        builder.color()
        self.device = wp.get_device(device)
        self.model = builder.finalize(device=self.device)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.solver = newton.solvers.SolverVBD(
                self.model, iterations=iterations, rigid_joint_linear_ke=ke, rigid_joint_angular_ke=ka
            )
        self.s0, self.s1, self.control = self.model.state(), self.model.state(), self.model.control()
        ej = int(self.model.elastic_joint.numpy()[0])
        self.qs = int(self.model.joint_q_start.numpy()[ej])
        self.qds = int(self.model.joint_qd_start.numpy()[ej])
        self.nm = int(self.basis.mode_count)
        self.Phi_nodal = cb.T @ self.gen.modal_matrix  # (n_dof, nm), float64
        self.pts_c = pts_c
        self.timings = {"basis_build": t_basis, "model_build": time.perf_counter() - t0}
        if joint_clamps is not None:
            self.set_active(joint_clamps)

    def set_active(self, active_idx):
        en = self.model.joint_enabled.numpy()
        for k, j in enumerate(self.joints):
            en[j] = k in set(active_idx)
        self.model.joint_enabled.assign(en)
        self.solver.notify_model_changed(self.newton.ModelFlags.JOINT_PROPERTIES)
        self.active = list(active_idx)

    def step(self, dt):
        self.solver.step(self.s0, self.s1, self.control, None, dt)
        self.s0, self.s1 = self.s1, self.s0

    def read(self):
        """(frame translation, quat xyzw, eta, eta_dot, frame twist) as float64."""
        q = self.s0.joint_q.numpy()[self.qs : self.qs + 7 + self.nm].astype(np.float64)
        qd = self.s0.joint_qd.numpy()[self.qds : self.qds + 6 + self.nm].astype(np.float64)
        return q[:3], q[3:7], q[7:], qd[6:], qd[:6]

    def displacement(self):
        """Nodal displacement relative to the scene's reference points, package frame (n_dof,)."""
        tr, quat, eta, _, _ = self.read()
        loc = self.pts_c + (self.Phi_nodal @ eta).reshape(-1, 3)
        u = np.asarray(quat[:3])[None, :]
        w = quat[3]
        world = loc + 2.0 * np.cross(u, np.cross(u, loc) + w * loc) + tr
        return (world - self.pts_c).reshape(-1)

    def nodal_velocity(self):
        """Nodal velocity (package/world axes) = frame twist rigid part + Phi eta_dot (small rotation)."""
        _tr, _quat, eta, etad, tw = self.read()
        loc = self.pts_c + (self.Phi_nodal @ eta).reshape(-1, 3)
        v = (self.Phi_nodal @ etad).reshape(-1, 3)
        # joint_qd[:3] is the world velocity of the rest COM; angular part world omega
        com = np.asarray(self.model.body_com.numpy()[self.body], float)
        v = v + tw[:3] + np.cross(tw[3:], loc - com)
        return v.reshape(-1)

    def set_modal(self, eta=None, eta_dot=None):
        jq = self.s0.joint_q.numpy()
        jd = self.s0.joint_qd.numpy()
        if eta is not None:
            jq[self.qs + 7 : self.qs + 7 + self.nm] = eta
        if eta_dot is not None:
            jd[self.qds + 6 : self.qds + 6 + self.nm] = eta_dot
        self.s0.joint_q.assign(jq)
        self.s0.joint_qd.assign(jd)
        self.s1.joint_q.assign(jq)
        self.s1.joint_qd.assign(jd)

    def set_modal_force(self, f_modal):
        jf = self.control.joint_f.numpy()
        jf[self.qds + 6 : self.qds + 6 + self.nm] = f_modal
        self.control.joint_f.assign(jf)


# ----------------------------------------------------------------------------------------------
# re-ROM driver
# ----------------------------------------------------------------------------------------------
class GroundedScene(PanelScene):
    """Newton scene of a GROUNDED Craig-Bampton body: the active clamp frames are eliminated (held),
    the remaining coordinates (released frames' constraint modes + fixed-interface modes + Ritz
    vectors) are diagonalised by the clamped eigenproblem of (K_c, M_c), and the frame is held by FIXED
    joints at the active clamp centres, which are basis attachment frames with zero mode values.

    Used for the re-ROM updates: about a pre-stressed state the free-free tangent is not a valid
    floating-frame stiffness (K_T R_rot = omega x f_int != 0 and the free-free stressed body has negative
    elastic eigenvalues, see notes/v2_wp4_panel.md), while the clamped tangent is positive definite.
    The rigid mass properties and the linear/angular coupling integrals are exact from the consistent
    mass M (as ModalGeneratorFEM computes them); no sample_mass (no centrifugal/quadratic integrals),
    as in the HMG static example."""

    def __init__(
        self,
        cb: CBResult,
        ref_points,
        active,
        M,
        *,
        gravity,
        shift,
        zeta=1.0,
        iterations=10,
        ke=1e10,
        ka=1e9,
        device="cpu",
    ):
        import scipy.linalg as la
        import warp as wp

        import newton
        from newton._src.sim.modal import ModalBasis

        self.newton, self.wp = newton, wp
        t0 = time.perf_counter()
        self.cb = cb
        self.shift = np.asarray(shift, float)
        self.ref_points = np.asarray(ref_points, float)
        pts_c = self.ref_points - self.shift
        keep = np.ones(cb.T.shape[1], bool)
        for k in active:
            keep[6 * k : 6 * k + 6] = False
        Kc, Mc = cb.Kr[np.ix_(keep, keep)], cb.Mr[np.ix_(keep, keep)]
        lam, V = la.eigh(Kc, Mc)
        self.clamped_eigenvalues = lam
        if lam[0] <= 0:
            raise ValueError(f"clamped reduced tangent is not positive definite (lowest {lam[0]:.3g})")
        Phi = cb.T[:, keep] @ V  # (n_dof, m) M-orthonormal
        self.Phi_nodal = Phi
        self.modal_matrix = V
        self.keep = keep
        n = len(pts_c)
        m = Phi.shape[1]
        MPhi = sp.csr_matrix(M) @ Phi
        Rg = np.zeros((3 * n, 6))
        Rg[:, :3] = np.tile(np.eye(3), (n, 1))
        for j, x in enumerate(pts_c):
            Rg[3 * j : 3 * j + 3, 3:] = -_skew(x)
        rm = Rg.T @ MPhi
        spatial = Rg.T @ (sp.csr_matrix(M) @ Rg)
        mass = float(np.trace(spatial[:3, :3]) / 3.0)
        cs = -spatial[:3, 3:] / mass
        com = np.array([cs[2, 1], cs[0, 2], cs[1, 0]])
        I0 = 0.5 * (spatial[3:, 3:] + spatial[3:, 3:].T)
        inertia = I0 - mass * ((com @ com) * np.eye(3) - np.outer(com, com))
        phi_s = np.transpose(Phi.reshape(n, 3, m), (0, 2, 1)).astype(np.float32)
        pos_c = cb.interface_positions - self.shift
        self.basis = ModalBasis(
            sample_points=pts_c.astype(np.float32),
            sample_phi=phi_s,
            mode_mass=np.ones(m),
            mode_stiffness=lam,
            mode_damping=2.0 * zeta * np.sqrt(lam),
            mode_coupling_linear=rm[:3].T,
            mode_coupling_angular=-rm[3:].T,
            mode_mass_matrix=Phi.T @ MPhi,
            rigid_mass=mass,
            rigid_com=com,
            rigid_inertia=inertia,
            frame_points=pos_c[active].astype(np.float32),
            frame_phi=np.zeros((len(active), m, 3), np.float32),
            frame_psi=np.zeros((len(active), m, 3), np.float32),
            frame_labels=[cb.interface_names[k] for k in active],
            label="panel_grounded",
        )
        self.build_warnings = []
        t_basis = time.perf_counter() - t0
        t0 = time.perf_counter()
        builder = newton.ModelBuilder(gravity=tuple(float(v) for v in gravity))
        self.body = builder.add_body_elastic(xform=wp.transform_identity(), modal_basis=self.basis, label="panel")
        self.joints = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Adding a FIXED joint.*")
            for k in active:
                xf = wp.transform(wp.vec3(*[float(v) for v in pos_c[k]]), wp.quat_identity())
                self.joints.append(
                    builder.add_joint_fixed(
                        parent=-1, child=self.body, parent_xform=xf, child_xform=xf, label=cb.interface_names[k]
                    )
                )
        builder.color()
        self.device = wp.get_device(device)
        self.model = builder.finalize(device=self.device)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.solver = newton.solvers.SolverVBD(
                self.model, iterations=iterations, rigid_joint_linear_ke=ke, rigid_joint_angular_ke=ka
            )
        self.s0, self.s1, self.control = self.model.state(), self.model.state(), self.model.control()
        ej = int(self.model.elastic_joint.numpy()[0])
        self.qs = int(self.model.joint_q_start.numpy()[ej])
        self.qds = int(self.model.joint_qd_start.numpy()[ej])
        self.nm = m
        self.pts_c = pts_c
        self.active = list(active)
        self.timings = {"basis_build": t_basis, "model_build": time.perf_counter() - t0}

    def set_active(self, active_idx):
        if sorted(active_idx) != sorted(self.active):
            raise ValueError("a grounded scene is built for one clamp set; rebuild it for another")


class PanelReROM:
    """Fixed-basis (``rerom=False``, HMG free-free CB + one FIXED joint per clamp) or re-ROM
    (``rerom=True``) Newton run of a CB panel body (see module doc).

    Re-ROM: the body is a :class:`GroundedScene`; it is rebuilt (forced update) whenever the active
    clamp set changes and whenever the residual mismatch exceeds ``tol``.  ``recovery``: "plain"
    (u_ref + Phi eta) or "rotated" (director-rotated, inextensible fibres) for the state at which
    f_int / K_T are evaluated and at which the new reference is placed.
    """

    def __init__(
        self,
        ccx: PanelCcxModel,
        M,
        K0,
        weld_C,
        interfaces,
        *,
        n_modes,
        n_ritz=2,
        gravity,
        shift,
        zeta=1.0,
        iterations=10,
        recovery="rotated",
        tol=0.02,
        check_every=10,
        min_interval=10,
        device="cpu",
        ke=1e10,
        ka=1e9,
        initial_cb: CBResult | None = None,
        rerom=True,
        log=print,
        tol_full=0.002,
        settle_tol=1e-3,
        rigid_pads=True,
    ):
        self.ccx = ccx
        self.M = sp.csr_matrix(M)
        self.C = weld_C
        self.itf0 = [dict(it) for it in interfaces]
        self.n_modes, self.n_ritz = int(n_modes), int(n_ritz)
        self.g = np.asarray(gravity, float)
        self.shift = shift
        self.zeta, self.iterations, self.device, self.ke, self.ka = zeta, iterations, device, ke, ka
        self.recovery = recovery
        self.rerom = bool(rerom)
        self.tol_full, self.settle_tol = float(tol_full), float(settle_tol)
        self.rigid_pads = bool(rigid_pads)
        self._full_lu = None
        self._u_prev_check = None
        self.tol, self.check_every, self.min_interval = float(tol), int(check_every), int(min_interval)
        self.log = log
        self.rest_pts = ccx.solid_points.copy()
        self.f_g = ccx.gravity_load(self.M, self.g)
        self.u_ref = np.zeros(ccx.n_dof)
        self.f_ref = np.zeros(ccx.n_dof)
        self.K0 = sp.csr_matrix(K0)
        self.K_ref = self.K0
        self.events = []
        self.checks = []
        self.active = None
        self.steps_since = 0
        t0 = time.perf_counter()
        if initial_cb is None:
            initial_cb = SolidCB(self.rest_pts, weld_C, self.itf0).reduce(K0, self.M, n_modes)
        self.cb = initial_cb
        self.scene = None
        if not self.rerom:
            self.scene = PanelScene(
                self.cb,
                self.rest_pts,
                gravity=self.g,
                shift=shift,
                zeta=zeta,
                iterations=iterations,
                device=device,
                ke=ke,
                ka=ka,
            )
        self.g0 = np.zeros(self.cb.T.shape[1])
        self.init_wall = time.perf_counter() - t0

    # -- state helpers ---------------------------------------------------------------------------
    def total_displacement(self):
        return self.u_ref + self.scene.displacement()

    def eval_state(self, u_plain):
        u = director_rotated(self.ccx, u_plain, q_ref=self.u_ref) if self.recovery == "rotated" else u_plain
        if self.rigid_pads and self.active is not None:
            pads = [it["nodes"] for k, it in enumerate(self.itf0) if k not in set(self.active)]
            u = rigidify_pads(self.rest_pts, u, pads)
        return u

    def set_active(self, active_idx, step=0, t=0.0):
        changed = self.active is None or sorted(active_idx) != sorted(self.active)
        self.active = list(active_idx)
        if not self.rerom:
            self.scene.set_active(active_idx)
            return
        if self.scene is None:  # first configuration: grounded CB about the rest state, no ccx call
            t0 = time.perf_counter()
            self.scene = GroundedScene(
                self.cb,
                self.rest_pts,
                self.active,
                self.M,
                gravity=self.g,
                shift=self.shift,
                zeta=self.zeta,
                iterations=self.iterations,
                ke=self.ke,
                ka=self.ka,
                device=self.device,
            )
            self._factor_full(SolidCB(self.rest_pts, self.C, self.itf0), self.K0)
            self.init_wall += time.perf_counter() - t0
        elif changed:
            self.update(step, t, "clamp_change")

    def _factor_full(self, scb, K_T):
        """Factorise the constrained FULL tangent (welds and pads eliminated, active frames held) of the
        new reference, for the out-of-basis (modified-Newton) residual check."""
        t0 = time.perf_counter()
        keep = np.ones(scb.S.shape[1], bool)
        for j in self.active:
            keep[scb.ni + 6 * j : scb.ni + 6 * j + 6] = False
        Sf = scb.S[:, keep].tocsc()
        Kf = (Sf.T @ (sp.csr_matrix(K_T) @ Sf)).tocsc()
        self._full_lu = (
            Sf,
            spla.splu(Kf, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True}),
        )
        return time.perf_counter() - t0

    def full_step(self, u_eval, f_int):
        """Modified-Newton step of the full constrained model with the last reference tangent,
        relative to the current max displacement (out-of-basis convergence indicator)."""
        Sf, lu = self._full_lu
        du = Sf @ lu.solve(Sf.T @ (self.f_g - f_int))
        return float(np.abs(du).max() / max(np.abs(u_eval).max(), 1e-300))

    def _moved_since_ref(self, u_plain):
        return float(np.abs(u_plain - self.u_ref).max() / max(np.abs(u_plain).max(), 1e-300))

    def _free_coords(self):
        mask = np.ones(self.cb.T.shape[1], bool)
        for k in self.active or []:
            mask[6 * k : 6 * k + 6] = False
        return mask

    def mismatch(self):
        """Mismatch of the linearisation about u_ref at the current state (trigger quantity).

        r = T_c^T (f_int(u_eval) - f_int(u_ref) - K_T(u_ref) d)  (T_c = CB basis without the held frames,
        d = recovered displacement since the reference, u_eval = plain or director-rotated recovery).
        Primary metric (returned first): the displacement correction it implies,
        e = max|T_c K_c^-1 r| / max|u_total| (a reduced Newton step relative to the current deflection).
        A force norm ||r|| / ||T_c^T f_g|| is far too sensitive for thin solids (membrane/thickness
        directions turn sub-micron differences into large forces) and is reported only.
        Also returned: the energy mismatch |dW_nl - dW_q| / (|f_ref.d| + |d K d|/2) with the trapezoid
        dW_nl ~ (f_ref + f).du / 2."""
        t0 = time.perf_counter()
        d = self.scene.displacement()
        u_plain = self.u_ref + d
        u_eval = self.eval_state(u_plain)
        r = self.ccx.evaluate(u_eval, tangent=False, strains=False)
        t_ccx = time.perf_counter() - t0
        lin = self.K_ref @ d
        free = self._free_coords()
        Tc = self.cb.T[:, free]
        resid_red = Tc.T @ (r["f_int"] - self.f_ref - lin)
        fg = Tc.T @ self.f_g
        e_force = float(np.linalg.norm(resid_red) / max(np.linalg.norm(fg), 1e-300))
        Kc = self.cb.Kr[np.ix_(free, free)]
        delta = Tc @ np.linalg.solve(Kc, resid_red)
        e = float(np.abs(delta).max() / max(np.abs(u_plain).max(), 1e-300))
        du = u_eval - self.u_ref
        dWnl = 0.5 * float((self.f_ref + r["f_int"]) @ du)
        dWq = float(self.f_ref @ d) + 0.5 * float(d @ lin)
        e_W = abs(dWnl - dWq) / max(abs(float(self.f_ref @ d)) + abs(0.5 * float(d @ lin)), 1e-30)
        self._last_force_mismatch = e_force
        return e, e_W, t_ccx, time.perf_counter() - t0, u_plain, u_eval, r["f_int"]

    def update(self, step, t, reason, u_eval=None):
        """Rebuild the CB basis about the current recovered state and rebuild the (grounded) Newton scene."""
        tw = {}
        T0 = time.perf_counter()
        v = self.scene.nodal_velocity()
        if u_eval is None:
            u_eval = self.eval_state(self.total_displacement())
        t0 = time.perf_counter()
        r = self.ccx.evaluate(u_eval, tangent=True, strains=False)
        tw["ccx_eval"] = time.perf_counter() - t0
        f_int, K_T = r["f_int"], r["K_T"]
        pts_new = self.rest_pts + u_eval.reshape(-1, 3)
        itf = []
        for it in self.itf0:
            fr = rigid_fit(self.rest_pts[it["nodes"]], u_eval.reshape(-1, 3)[it["nodes"]], it["position"])
            itf.append(dict(it, position=it["position"] + fr[:3]))
        t0 = time.perf_counter()
        scb = SolidCB(pts_new, self.C, itf)
        tw["elimination"] = time.perf_counter() - t0
        ritz = [self.f_g, self.f_g - f_int][: self.n_ritz]
        cb = scb.reduce(K_T, self.M, self.n_modes, ritz_loads=ritz, rigid_project_tol=None)
        for k, v_ in cb.timings.items():
            tw[k] = v_
        tw["full_factorization"] = self._factor_full(scb, K_T)
        t0 = time.perf_counter()
        old_scene = self.scene
        scene = GroundedScene(
            cb,
            pts_new,
            self.active,
            self.M,
            gravity=self.g,
            shift=self.shift,
            zeta=self.zeta,
            iterations=self.iterations,
            ke=self.ke,
            ka=self.ka,
            device=self.device,
        )
        tw["basis_build"] = scene.timings["basis_build"]
        tw["model_build"] = scene.timings["model_build"]
        t1 = time.perf_counter()
        eta_dot = scene.Phi_nodal.T @ (self.M @ v)  # M-orthogonal velocity projection
        g0_eta = scene.Phi_nodal.T @ f_int  # pre-stress
        scene.set_modal(eta=np.zeros(scene.nm), eta_dot=eta_dot)
        scene.set_modal_force(-g0_eta)
        tw["state_transfer"] = time.perf_counter() - t1
        tw["rewrite_total"] = time.perf_counter() - t0
        tw["total"] = time.perf_counter() - T0
        ke_old = 0.5 * float(v @ (self.M @ v))
        v_new = scene.Phi_nodal @ eta_dot
        ke_new = 0.5 * float(v_new @ (self.M @ v_new))
        resid = float(
            np.linalg.norm(scene.Phi_nodal.T @ (self.f_g - f_int)) / np.linalg.norm(scene.Phi_nodal.T @ self.f_g)
        )
        self.scene, self.cb = scene, cb
        self.u_ref, self.f_ref, self.K_ref, self.g0 = u_eval.copy(), f_int.copy(), K_T, g0_eta
        ev = {
            "step": int(step),
            "time": float(t),
            "reason": reason,
            "interval": int(self.steps_since),
            "wall": tw,
            "ritz_kept": cb.n_ritz,
            "negative_pivots": cb.info["negative_pivots"],
            "modes": int(scene.nm),
            "rigid_residual_free_free": cb.info["rigid_residual_before_projection"],
            "clamped_hz_first": (np.sqrt(scene.clamped_eigenvalues[:3]) / (2 * np.pi)).tolist(),
            "modal_residual_at_ref": resid,
            "ke_before": ke_old,
            "ke_after": ke_new,
            "max_sag_ref_mm": 1e3 * float(np.abs(self.ccx.mid_and_director(u_eval)[0][:, 2]).max()),
        }
        self.events.append(ev)
        self.steps_since = 0
        del old_scene
        self.log(
            f"  update {len(self.events)} at step {step} ({reason}): total {tw['total']:.2f}s "
            f"ccx {tw['ccx_eval']:.2f} fact {tw['factorization']:.3f} cmodes {tw['constraint_modes']:.2f} "
            f"eig {tw['eigensolve']:.2f} rewrite {tw['rewrite_total']:.2f}; sag_ref {ev['max_sag_ref_mm']:.4f} mm, "
            f"ritz {cb.n_ritz}, f1 {ev['clamped_hz_first'][0]:.2f} Hz, resid@ref {resid:.2e}, KE {ke_old:.3e}->{ke_new:.3e}"
        )
        return ev

    def maybe_update(self, step, t, force=False):
        self.steps_since += 1
        if not self.rerom:
            return None
        if not force and (self.steps_since < self.min_interval or self.steps_since % self.check_every != 0):
            return None
        e, eW, t_ccx, t_all, u_plain, u_eval, f_eval = self.mismatch()
        t0 = time.perf_counter()
        e_full = self.full_step(u_eval, f_eval)
        moved = (
            float("inf")
            if self._u_prev_check is None
            else float(np.abs(u_plain - self._u_prev_check).max() / max(np.abs(u_plain).max(), 1e-300))
        )
        self._u_prev_check = u_plain
        settled = moved < self.settle_tol
        self.checks.append(
            {
                "step": int(step),
                "time": float(t),
                "mismatch": e,
                "energy_mismatch": eW,
                "force_mismatch": self._last_force_mismatch,
                "full_step": e_full,
                "moved": moved,
                "moved_since_ref": self._moved_since_ref(u_plain),
                "wall_ccx": t_ccx,
                "wall": t_all + time.perf_counter() - t0,
            }
        )
        if force:
            reason = "forced"
        elif e > self.tol:
            reason = "mismatch"
        elif settled and self._moved_since_ref(u_plain) > self.tol_full:
            # settled away from the reference: re-linearise at the (reduced) equilibrium.  A residual check
            # with the stale reference tangent (e_full) underestimates the remaining error here because the
            # sagging panel softens (see notes/v2_wp4_panel.md), so the move itself is the criterion.
            reason = "settled_move"
        else:
            return None
        self._u_prev_check = None
        return self.update(step, t, reason, u_eval=u_eval)
