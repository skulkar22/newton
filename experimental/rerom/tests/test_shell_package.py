# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shell packages with triangles and offset weld ties (rerom.ccx, rerom.panel).

Synthetic packages: a cambered triangle plate A, a strip B above it (offset by the mid-surface gap) welded to A
by rigid offset links (each B node to the A triangle below it, rows like a package's C.mtx: 6 per weld incl.
rotations), and RBE2 clamp pads, one of which contains welded B nodes. CalculiX cases are skipped without
``ccx`` (``$REROM_CCX``)."""

# ruff: noqa: TID253

from __future__ import annotations

import _fixtures  # noqa: F401  (puts the rerom package on sys.path)

# isort: split

import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from rerom.ccx import CCX_DEFAULT, ShellMesh, rotvec_to_matrix
from rerom.panel import GroundedScene, IndefiniteTangentError, PanelCcxModel, PanelReROM, SolidCB, interface_list

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from shell_package import nr_static

HAVE_CCX = Path(CCX_DEFAULT).exists()
TA, TB = 0.0012, 0.0015
MAT = {"E": 7.0e10, "nu": 0.33, "rho": 2700.0}


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def tri_grid(nx, ny, lx, ly, x0=0.0, z=lambda x, y: 0.0):
    xs, ys = np.linspace(x0, x0 + lx, nx + 1), np.linspace(0.0, ly, ny + 1)
    P = np.array([[x, y, z(x, y)] for y in ys for x in xs])
    idx = lambda i, j: j * (nx + 1) + i  # noqa: E731
    T = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            T += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    return P, np.array(T)


def write_package(path, *, camber=0.004, pad_weld=True):
    """Write a two-panel triangle package; returns dict with node ids of interest."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    lx, ly = 0.24, 0.08
    zA = lambda x, y: camber * (1.0 - (2.0 * x / lx - 1.0) ** 2)  # noqa: E731
    PA, TA_ = tri_grid(12, 4, lx, ly, z=zA)
    gap = 0.5 * (TA + TB)
    # strip B over x in [0.09, 0.15], on the mid-surface of A shifted by the gap along the local normal (~z)
    PB, TB_ = tri_grid(3, 4, 0.06, ly, x0=0.09, z=lambda x, y: zA(x, y) + gap)
    nA = len(PA)
    P = np.vstack([PA, PB])
    T = np.vstack([TA_, TB_ + nA])
    panel = ["A"] * nA + ["B"] * len(PB)
    # welds: B nodes of the columns x = 0.11 and x = 0.15 (some of the latter inside pad PB)
    rows, cols, vals = [], [], []
    r = 0
    welded = []
    for k in range(len(PB)):
        xb = PB[k]
        if not (abs(xb[0] - 0.11) < 1e-9 or abs(xb[0] - 0.15) < 1e-9):
            continue
        # A triangle below (planar projection) and barycentric weights
        for tri in TA_:
            X = PA[tri]
            M2 = np.array([[X[1, 0] - X[0, 0], X[2, 0] - X[0, 0]], [X[1, 1] - X[0, 1], X[2, 1] - X[0, 1]]])
            l12 = np.linalg.solve(M2, xb[:2] - X[0, :2])
            w = np.array([1.0 - l12.sum(), l12[0], l12[1]])
            if np.all(w > -1e-12):
                break
        keep = w > 1e-12  # 2-node (edge) or 3-node master sets, as in production packages
        tri, w = np.asarray(tri)[keep], w[keep] / w[keep].sum()
        b = nA + k
        welded.append(b)
        for a in range(3):  # u_b - sum w_i (u_i + theta_i x d_i) = 0, d_i = x_b - x_i
            rows.append(r), cols.append(6 * b + a), vals.append(1.0)
            for i, wi in zip(tri, w, strict=True):
                d = P[b] - P[i]
                rows.append(r), cols.append(6 * i + a), vals.append(-wi)
                S = -_skew(d)  # theta x d = -[d]x theta
                for c in range(3):
                    if S[a, c] != 0.0:
                        rows.append(r), cols.append(6 * i + 3 + c), vals.append(-wi * S[a, c])
            r += 1
        for a in range(3):  # theta_b - sum w_i theta_i = 0
            rows.append(r), cols.append(6 * b + 3 + a), vals.append(1.0)
            for i, wi in zip(tri, w, strict=True):
                rows.append(r), cols.append(6 * i + 3 + a), vals.append(-wi)
            r += 1
    C = sp.csr_matrix((vals, (rows, cols)), shape=(r, 6 * len(P)))
    scipy.io.mmwrite(path / "C.mtx", C)
    nid = np.arange(1, len(P) + 1) + 1000
    with open(path / "nodes.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["node_id", "x_m", "y_m", "z_m", "panel_id"])
        for k in range(len(P)):
            wr.writerow([nid[k], *P[k], panel[k]])
    with open(path / "elements.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["element_id", "panel_id", "n1", "n2", "n3", "n4"])
        for e, t in enumerate(T):
            wr.writerow([e + 1, panel[t[0]], *nid[t], ""])
    with open(path / "panels.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "panel_id",
                "part_name",
                "material_code",
                "thickness_m",
                "density_kg_m3",
                "youngs_modulus_pa",
                "poisson_ratio",
            ]
        )
        wr.writerow(["A", "A", "m", TA, MAT["rho"], MAT["E"], MAT["nu"]])
        wr.writerow(["B", "B", "m", TB, MAT["rho"], MAT["E"], MAT["nu"]])

    def near(x0, panel_rows):
        return [k for k in panel_rows if abs(P[k, 0] - x0) < 0.021 and abs(P[k, 1] - 0.04) < 0.021]

    clamps = {
        "L": ("A", near(0.0, range(nA)), 1, 1),
        "R": ("A", near(0.24, range(nA)), 1, 0),
        "PB": ("B", near(0.15, range(nA, len(P))) if pad_weld else near(0.09, range(nA, len(P))), 1, 0),
    }
    with open(path / "clamps.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "clamp_id",
                "panel_id",
                "centre_x_m",
                "centre_y_m",
                "centre_z_m",
                "normal_x",
                "normal_y",
                "normal_z",
                "pad_type",
                "constrained_dofs",
                "active_full",
                "active_reduced",
            ]
        )
        for cid, (pid, ks, af, ar) in clamps.items():
            c = P[ks].mean(0)
            wr.writerow([cid, pid, *c, 0, 0, 1, "clamp", 123456, af, ar])
    with open(path / "clamp_nodes.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["clamp_id", "node_id"])
        for cid, (_pid, ks, _af, _ar) in clamps.items():
            for k in ks:
                wr.writerow([cid, nid[k]])
    return {
        "n_nodes": len(P),
        "n_tris": len(T),
        "welded_rows": welded,
        "pad_rows": {k: v[1] for k, v in clamps.items()},
    }


def stiff_pad_package(src, dst, clamp_ids, factor=1.0e3):
    """Copy a package with the elements inside the given clamp pads made `factor` times stiffer (own panel ids), so
    a solver can carry those pads as near-rigid elements instead of MPCs."""
    src, dst = Path(src), Path(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    with open(src / "clamp_nodes.csv") as fh:
        pad = {r["node_id"] for r in csv.DictReader(fh) if r["clamp_id"] in set(clamp_ids)}
    with open(src / "elements.csv") as fh:
        els = list(csv.DictReader(fh))
    with open(dst / "elements.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(els[0].keys()))
        wr.writeheader()
        for e in els:
            nodes = {e[k] for k in ("n1", "n2", "n3", "n4") if e.get(k)}
            wr.writerow(e | {"panel_id": "S" + e["panel_id"]} if nodes <= pad else e)
    with open(src / "panels.csv") as fh:
        panels = list(csv.DictReader(fh))
    with open(dst / "panels.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(panels[0].keys()))
        wr.writeheader()
        for p in panels:
            wr.writerow(p)
            wr.writerow(
                p | {"panel_id": "S" + p["panel_id"], "youngs_modulus_pa": float(p["youngs_modulus_pa"]) * factor}
            )


class TestShellPackageLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="rerom_pkg_"))
        cls.info = write_package(cls.work / "pkg")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_triangles_need_split(self):
        with self.assertRaises(ValueError):
            ShellMesh.from_shell_package(self.work / "pkg")

    def test_split_and_refine_counts_and_area(self):
        """Every source node keeps its row and id; the split is conforming and area-preserving."""
        for refine in (0, 1):
            m = ShellMesh.from_shell_package(self.work / "pkg", triangles="split", refine=refine)
            nT = self.info["n_tris"]
            self.assertEqual(len(m.quads), 3 * nT * 4**refine)
            self.assertEqual(m.orig_count, self.info["n_nodes"])
            np.testing.assert_array_equal(m.node_ids[: m.orig_count], np.arange(1, m.orig_count + 1) + 1000)
            _, area = m.element_normals()
            X = m.points[: m.orig_count]
            T = m.source_tris
            a_src = 0.5 * np.linalg.norm(np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]), axis=1).sum()
            self.assertLess(abs(area.sum() / a_src - 1.0), 1e-3)  # planar quads of a slightly curved surface
            # conforming: every interior edge shared by exactly two quads, boundary edges by one
            edges = {}
            for q in m.quads:
                for i in range(4):
                    e = tuple(sorted((q[i], q[(i + 1) % 4])))
                    edges[e] = edges.get(e, 0) + 1
            self.assertLessEqual(max(edges.values()), 2)
            # pads grow: every source pad node stays, every new pad node interpolates pad nodes only
            for c in m.clamps:
                self.assertTrue(set(c["source_nodes"].tolist()) <= set(c["nodes"].tolist()))
                for i in c["nodes"]:
                    if i >= m.orig_count:
                        self.assertTrue(all(p in set(c["nodes"].tolist()) for p in m.parents[i]))
            self.assertIsNotNone(m.shell_C)
            self.assertEqual(m.welds, [])

    def test_ties_are_rigid_motion_consistent(self):
        """Mapped ties vanish for rigid translation and (linearised) rigid rotation, incl. drilling terms."""
        m = ShellMesh.from_shell_package(self.work / "pkg", triangles="split")
        model = PanelCcxModel(m, self.work / "rig")
        C = model.weld_rows()
        deps = model.weld_dependents
        self.assertEqual(C.shape[0], 5 * len(self.info["welded_rows"]))
        self.assertEqual(len(set(deps.tolist())), len(deps))
        X = model.rest_q.reshape(-1, 3)
        scale = np.abs(C).max() * np.abs(X).max()
        for k in range(3):
            t = np.zeros(3)
            t[k] = 1.0
            self.assertLess(np.abs(C @ np.tile(t, len(X))).max(), 1e-12 * np.abs(C).max())
            w = np.zeros(3)
            w[k] = 1.0
            q = np.cross(w, X).reshape(-1)  # linearised rigid rotation of every solid node
            self.assertLess(np.abs(C @ q).max(), 1e-10 * scale)
        # dependent DOFs: identity block, and welds inside pad PB take their dependents from the master panel
        pad_nodes = set()
        for c in m.clamps:
            pad_nodes.update(int(v) for v in c["nodes"])
        self.assertTrue(all((d // 6) not in pad_nodes for d in deps))
        np.testing.assert_allclose(C[np.arange(len(deps)), deps].A.ravel(), 1.0)


@unittest.skipUnless(HAVE_CCX, f"CalculiX not found ({CCX_DEFAULT}); set REROM_CCX")
class TestShellPackageCcx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="rerom_pkg_ccx_"))
        cls.info = write_package(cls.work / "pkg")
        cls.mesh = ShellMesh.from_shell_package(cls.work / "pkg", triangles="split", refine=1)
        cls.model = PanelCcxModel(cls.mesh, cls.work / "ccx")
        cls.K, cls.M = cls.model.rest_matrices()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def _active(self, cfg):
        return [c["clamp_id"] for c in self.mesh.clamps if c["active"][cfg] == 1]

    def test_plate_element_accuracy(self):
        """Split + one refinement (the element used for the two-panel model): a clamped square triangle plate under
        self-weight at a 5 mm element size deflects within 1 % of Kirchhoff's 0.00126 q a^4 / D."""
        d = self.work / "plate"
        d.mkdir(exist_ok=True)
        a, n = 0.2, 40
        P, T = tri_grid(n, n, a, a)
        edge = [k for k in range(len(P)) if min(P[k, 0], P[k, 1], a - P[k, 0], a - P[k, 1]) < 1e-12]
        with open(d / "nodes.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["node_id", "x_m", "y_m", "z_m", "panel_id"])
            for k in range(len(P)):
                wr.writerow([k + 1, *P[k], "A"])
        with open(d / "elements.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["element_id", "panel_id", "n1", "n2", "n3", "n4"])
            for e, t in enumerate(T):
                wr.writerow([e + 1, "A", *(t + 1), ""])
        with open(d / "panels.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(
                [
                    "panel_id",
                    "part_name",
                    "material_code",
                    "thickness_m",
                    "density_kg_m3",
                    "youngs_modulus_pa",
                    "poisson_ratio",
                ]
            )
            wr.writerow(["A", "A", "m", TA, MAT["rho"], MAT["E"], MAT["nu"]])
        with open(d / "clamps.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["clamp_id", "panel_id", "active_full"])
            wr.writerow(["EDGE", "A", 1])
        with open(d / "clamp_nodes.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["clamp_id", "node_id"])
            for k in edge:
                wr.writerow(["EDGE", k + 1])
        m = ShellMesh.from_shell_package(d, triangles="split", refine=1)
        model = PanelCcxModel(m, self.work / "plate_ccx")
        r = model.static(model.clamp_dofs(["EDGE"]), [0, 0, -9.81], nlgeom=False, welds=False, label="plate")
        w = -model.mid_and_director(r["u"])[0][:, 2].min()
        D = MAT["E"] * TA**3 / (12 * (1 - MAT["nu"] ** 2))
        w_ex = 0.00126 * MAT["rho"] * 9.81 * TA * a**4 / D
        self.assertLess(abs(w / w_ex - 1.0), 0.01, f"w {w:.6g} vs Kirchhoff {w_ex:.6g}")

    def test_ccx_ties_equal_python_elimination(self):
        """ccx linear static with *EQUATION ties and rigid pads = Python solve with SolidCB elimination of the
        same exported K, under identical point loads (*CLOAD; a body load would differ between ccx's consistent
        element load and M g with the condensed mass), both configurations incl. a pad that contains welded nodes.
        Also: the ccx solution satisfies the mapped ties."""
        model, K = self.model, self.K
        C, deps = model.weld_rows(), model.weld_dependents
        itf = interface_list(model)
        X = model.solid_points
        rng = np.random.default_rng(7)
        F = np.zeros_like(X)
        # loads on both panels, away from the pads: a few solid nodes, all three directions
        cand = [k for k in range(len(X)) if all(np.linalg.norm(X[k] - it["position"]) > 0.03 for it in itf)]
        for k in rng.choice(cand, 12, replace=False):
            F[k] = rng.uniform(-1.0, 1.0, 3)
        f = F.reshape(-1)
        for cfg in ("full", "reduced"):
            active = self._active(cfg)
            released = [c["clamp_id"] for c in self.mesh.clamps if c["clamp_id"] not in active]
            r = model.static(
                model.clamp_dofs(active),
                [0.0, 0.0, 0.0],
                nlgeom=False,
                rigid_clamps=released,
                label=f"lin_{cfg}",
                nodal_forces=f,
            )
            scb = SolidCB(model.solid_points, C, itf, weld_dependents=deps)
            keep = np.ones(scb.S.shape[1], bool)
            for k, it in enumerate(itf):
                if it["name"] in active:
                    keep[scb.ni + 6 * k : scb.ni + 6 * k + 6] = False
            S = scb.S[:, keep].tocsc()
            u = S @ spla.spsolve((S.T @ K @ S).tocsc(), S.T @ f)
            rel = np.abs(r["u"] - u).max() / np.abs(u).max()
            self.assertLess(rel, 1e-5, cfg)  # ccx prints 7 significant digits
            self.assertLess(np.abs(C @ r["u"]).max(), 1e-6 * np.abs(r["u"]).max())

    def test_indefinite_tangent_clip(self):
        """A clamped tangent with negative eigenvalues stops the grounded scene by default; with clip_negative the
        scene keeps the tangent's eigenvectors and raises only the non-positive eigenvalues to the smallest
        positive one."""
        model, K, M = self.model, self.K, self.M
        itf = interface_list(model)
        active = [k for k, c in enumerate(self.mesh.clamps) if c["active"]["reduced"] == 1]
        scb = SolidCB(model.solid_points, model.weld_rows(), itf, weld_dependents=model.weld_dependents)
        cb0 = scb.reduce(K, M, 12, rigid_project_tol=None)
        keep = np.ones(cb0.T.shape[1], bool)
        for k in active:
            keep[6 * k : 6 * k + 6] = False
        import scipy.linalg as sla

        lam0 = sla.eigh(cb0.Kr[np.ix_(keep, keep)], cb0.Mr[np.ix_(keep, keep)], eigvals_only=True)
        alpha = 0.5 * (lam0[1] + lam0[2])  # about two clamped eigenvalues below zero after the shift
        cb = scb.reduce(sp.csr_matrix(K - alpha * M), M, 12, rigid_project_tol=None)
        kw = {"gravity": [0.0, 0.0, -9.81], "shift": model.mesh.points.mean(0)}
        with self.assertRaises(IndefiniteTangentError):
            GroundedScene(cb, model.solid_points, active, M, **kw)
        sc = GroundedScene(cb, model.solid_points, active, M, clip_negative=True, **kw)
        lam = sc.clamped_eigenvalues
        self.assertEqual(sc.clipped, int(np.count_nonzero(lam <= 0)))
        self.assertGreaterEqual(sc.clipped, 1)
        stiff = np.asarray(sc.basis.mode_stiffness, float)
        floor = lam[lam > 0].min()
        np.testing.assert_allclose(stiff[lam > 0], lam[lam > 0], rtol=1e-6)  # stored as float32
        np.testing.assert_allclose(stiff[lam <= 0], floor, rtol=1e-6)

    def test_settle_force_guard(self):
        """settle_force_tol: once the re-ROM run has settled, a projected force mismatch above the threshold triggers a
        'settled_unbalanced' update; with a near-zero threshold it fires, and the settled sag still matches CalculiX
        NLGEOM with stiff pads within 0.1 %."""
        model, K, M = self.model, self.K, self.M
        ref = self._references()
        itf = interface_list(model)
        full = [k for k, c in enumerate(self.mesh.clamps) if c["active"]["full"] == 1]
        red = [k for k, c in enumerate(self.mesh.clamps) if c["active"]["reduced"] == 1]
        run = PanelReROM(
            model,
            M,
            K,
            model.weld_rows(),
            itf,
            n_modes=24,
            gravity=self.G10,
            shift=model.mesh.points.mean(0),
            iterations=10,
            tol=0.01,
            check_every=10,
            min_interval=10,
            log=lambda s: None,
            weld_dependents=model.weld_dependents,
            settle_force_tol=1e-9,
        )
        step = 0
        for act in (full, red):
            run.set_active(act, step=step)
            for _ in range(1500):
                run.scene.step(1e-3)
                step += 1
                run.maybe_update(step, step * 1e-3)
        reasons = [e["reason"] for e in run.events]
        self.assertIn("settled_unbalanced", reasons)
        s = self._sag(run.eval_state(run.total_displacement()))
        self.assertLess(
            abs(s / self._sag(ref["nl"]) - 1.0), 1e-3, f"re-ROM {s:.6g} vs NLGEOM {self._sag(ref['nl']):.6g}"
        )

    def test_tangent_matches_finite_difference(self):
        """K_T from ccx at a deformed state matches central differences of f_int (NLGEOM)."""
        model = self.model
        rng = np.random.default_rng(3)
        X = model.rest_q.reshape(-1, 3)
        Q = rotvec_to_matrix(np.array([0.05, -0.08, 0.03]))
        u0 = (X @ Q.T - X).reshape(-1) + 2e-4 * np.sin(40.0 * X[:, 0]).repeat(3) * np.tile([0, 0, 1.0], len(X))
        v = rng.standard_normal(u0.shape)
        v *= 1e-5 / np.abs(v).max()  # ccx prints forces to 7 significant digits
        KT = model.evaluate(u0, tangent=True, strains=False)["K_T"]
        fp = model.evaluate(u0 + v, tangent=False, strains=False)["f_int"]
        fm = model.evaluate(u0 - v, tangent=False, strains=False)["f_int"]
        fd = (fp - fm) / 2.0
        self.assertLess(np.linalg.norm(KT @ v - fd) / np.linalg.norm(fd), 1e-3)

    # 10 g: NLGEOM vs linear differs by 0.75 % and the two nonlinear references agree to 0.011 %. Above ~15 g the
    # cambered strip nears snap-through and the full-order NR no longer converges (measured at 20 and 30 g).
    G10 = np.array([0.0, 0.0, -9.81 * 10.0])

    def _references(self):
        """Released configuration at 10 g, computed once: CalculiX NLGEOM and linear statics with the released pads
        as stiff elements (E x 1000, no MPC), and the full-order Newton-Raphson with exact rigid pad frames.
        CalculiX's *RIGID BODY pads are not used: with NLGEOM and the *EQUATION ties the increments shrink until
        CalculiX stops (exit 201)."""
        cls = type(self)
        if getattr(cls, "_refs", None) is None:
            model, active = self.model, self._active("reduced")
            released = [c["clamp_id"] for c in self.mesh.clamps if c["clamp_id"] not in active]
            stiff_pad_package(self.work / "pkg", self.work / "pkg_stiff", released)
            ms = ShellMesh.from_shell_package(self.work / "pkg_stiff", triangles="split", refine=1)
            rm = PanelCcxModel(ms, self.work / "ccx_stiff")
            nl = rm.static(rm.clamp_dofs(active), self.G10, nlgeom=True, n_incr=40, controls=False, label="nl_ref")
            lin = rm.static(rm.clamp_dofs(active), self.G10, nlgeom=False, label="lin_ref")
            itf = interface_list(model)
            red = [k for k, c in enumerate(self.mesh.clamps) if c["active"]["reduced"] == 1]
            u_nr, hist = nr_static(
                model,
                self.K,
                self.M,
                model.weld_rows(),
                model.weld_dependents,
                itf,
                red,
                self.G10,
                steps=(0.25, 0.5, 0.75, 1.0),
                tol=3e-7,
                log=lambda s: None,
            )
            cls._refs = {"nl": nl["u"], "lin": lin["u"], "nr": u_nr, "nr_hist": hist, "released": released}
        return cls._refs

    def _sag(self, u):
        return -self.model.mid_and_director(u)[0][:, 2].min()  # same nodes in the stiff-pad model

    def test_nr_equilibrium_rigid_pads_and_ties(self):
        """Full-order nonlinear equilibrium (NR, 10 g): released pads stay rigid, the ties hold, and the sag agrees
        with CalculiX NLGEOM with stiff pads within 0.05 % (two independent nonlinear references)."""
        ref = self._references()
        model, released = self.model, ref["released"]
        u_nr = ref["nr"]
        self.assertFalse(any(h.get("not_converged") for h in ref["nr_hist"]), "NR did not converge")
        self.assertLess(
            abs(self._sag(u_nr) / self._sag(ref["nl"]) - 1.0),
            5e-4,
            f"NR {self._sag(u_nr):.6g} vs stiff-pad NLGEOM {self._sag(ref['nl']):.6g}",
        )
        self.assertLess(np.abs(model.weld_rows() @ u_nr).max(), 1e-9 * np.abs(u_nr).max())
        X = model.solid_points
        u = u_nr.reshape(-1, 3)
        for c in self.mesh.clamps:
            if c["clamp_id"] not in released:
                continue
            nodes = np.concatenate([2 * c["nodes"], 2 * c["nodes"] + 1])
            A, B = X[nodes], X[nodes] + u[nodes]
            ca, cb = A.mean(0), B.mean(0)
            Uu, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
            R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ Uu.T))]) @ Uu.T
            resid = np.abs((A - ca) @ R.T + cb - B).max()
            self.assertLess(resid, 1e-6 * np.abs(u[nodes]).max(), c["clamp_id"])

    def test_rerom_matches_nlgeom_static(self):
        """End to end: release R and PB at 10 g (clear geometric nonlinearity); the settled re-ROM sag matches
        CalculiX NLGEOM with stiff pads (see _references) within 0.1 %, the fixed basis does not."""
        model, K, M = self.model, self.K, self.M
        g = self.G10
        ref = self._references()
        sag = self._sag
        s_nl, s_lin = sag(ref["nl"]), sag(ref["lin"])
        self.assertGreater(abs(s_nl / s_lin - 1.0), 0.005)  # nonlinear well beyond the 0.1 % acceptance band
        itf = interface_list(model)
        full = [k for k, c in enumerate(self.mesh.clamps) if c["active"]["full"] == 1]
        red = [k for k, c in enumerate(self.mesh.clamps) if c["active"]["reduced"] == 1]
        out = {}
        # the fixed basis needs dt = 0.25 ms on this small model (at 1 ms it diverges right after the release, also
        # at 1 g); re-ROM rebuilds its basis at the release and runs stably at 1 ms
        for method, dt in (("fixed", 2.5e-4), ("rerom", 1e-3)):
            run = PanelReROM(
                model,
                M,
                K,
                model.weld_rows(),
                itf,
                n_modes=24,
                gravity=g,
                shift=model.mesh.points.mean(0),
                iterations=10,
                tol=0.01,
                check_every=10,
                min_interval=10,
                rerom=(method == "rerom"),
                log=lambda s: None,
                weld_dependents=model.weld_dependents,
            )
            step = 0
            for act in (full, red):
                run.set_active(act, step=step)
                for _ in range(int(round(1.5 / dt))):
                    run.scene.step(dt)
                    step += 1
                    run.maybe_update(step, step * dt)
            u = run.total_displacement()
            out[method] = sag(run.eval_state(u) if method == "rerom" else u)
        err_rerom = abs(out["rerom"] / s_nl - 1.0)
        err_fixed = abs(out["fixed"] / s_nl - 1.0)
        self.assertLess(err_rerom, 1e-3, f"re-ROM {out['rerom']:.6g} vs NLGEOM {s_nl:.6g}")
        self.assertGreater(
            err_fixed, 5.0 * err_rerom, f"fixed {out['fixed']:.6g}, re-ROM {out['rerom']:.6g}, NLGEOM {s_nl:.6g}"
        )


if __name__ == "__main__":
    unittest.main()
