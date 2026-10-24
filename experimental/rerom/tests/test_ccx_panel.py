# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CalculiX-coupled pieces (rerom.ccx, rerom.panel); skipped without a ``ccx`` executable (``$REROM_CCX``)."""

# ruff: noqa: TID253

from __future__ import annotations

import _fixtures  # noqa: F401  (puts the rerom package on sys.path)

# isort: split

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warp as wp
from _fixtures import free_controller
from rerom.ccx import CCX_DEFAULT, CcxModel, make_plate_mesh, rotvec_to_matrix
from rerom.panel import CcxAssembler, PanelCcxModel, SolidCB, interface_list
from test_controller import _relative_diffs, _snapshot

import newton

HAVE_CCX = Path(CCX_DEFAULT).exists()


def clamped_plate(nx=8, ny=3):
    m = make_plate_mesh(nx=nx, ny=ny, lx=0.24, ly=0.09, t=0.0015, camber=0.01)
    P = m.points

    def rows(x0):
        return np.flatnonzero((np.abs(P[:, 0] - x0) < 0.031) & (np.abs(P[:, 1] - 0.045) < 0.031))

    m.clamps = [
        {"clamp_id": "L", "nodes": rows(0.0), "active": {"c": 1}, "centre": P[rows(0.0)].mean(0)},
        {"clamp_id": "M", "nodes": rows(0.12), "active": {"c": 0}, "centre": P[rows(0.12)].mean(0)},
        {"clamp_id": "R", "nodes": rows(0.24), "active": {"c": 1}, "centre": P[rows(0.24)].mean(0)},
    ]
    return m


@unittest.skipUnless(HAVE_CCX, f"CalculiX not found ({CCX_DEFAULT}); set REROM_CCX")
class TestCcxPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="rerom_ccx_"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_rigid_rotation_invariance(self):
        """A 40-degree rigid rotation gives f_int = 0 and K_T = Q K Q^T (CalculiX NLGEOM)."""
        m = make_plate_mesh(nx=8, ny=3, lx=0.3, ly=0.08, t=0.002, camber=0.015)
        model = CcxModel(m, self.work / "plate")
        K, _ = model.rest_matrices()
        Q = rotvec_to_matrix(np.array([0.3, -0.5, 0.4]))
        X = model.rest_q.reshape(-1, 3)
        u = (X @ Q.T - X).reshape(-1)
        ev = model.evaluate(u, tangent=True, strains=True)
        self.assertLess(np.linalg.norm(ev["f_int"]), 1e-8 * np.linalg.norm(K @ u))
        Qb = sp.kron(sp.identity(2 * m.n_nodes), sp.csr_matrix(Q)).tocsr()
        self.assertLess(abs(ev["K_T"] - Qb @ K @ Qb.T).max(), 1e-8 * abs(K).max())

    def test_solid_cb_matches_newton_generator(self):
        mesh = clamped_plate()
        c = PanelCcxModel(mesh, self.work / "cb")
        K, M = c.rest_matrices()
        itf = interface_list(c)
        cb = SolidCB(c.solid_points, c.weld_rows(), itf).reduce(K, M, 10)
        specs = [
            {
                "name": i["name"],
                "node_indices": i["nodes"].tolist(),
                "coupling": "rbe2",
                "position": i["position"].tolist(),
            }
            for i in itf
        ]
        gen = newton.ModalGeneratorCraigBamptonFEM(c.solid_points, M, K, specs, fixed_mode_count=10, dofs_per_node=3)
        gen.reduce()
        np.testing.assert_allclose(cb.eigenvalues, gen.fixed_interface_eigenvalues, rtol=1e-8)

    def test_ccx_assembler_null_update(self):
        """CcxAssembler satisfies the TangentAssembler interface: a null update of a clamped thin-solid plate
        reproduces every rewritten Newton array."""
        mesh = make_plate_mesh(nx=8, ny=2, lx=0.24, ly=0.06, t=0.002)
        c = PanelCcxModel(mesh, self.work / "adapter")
        K, M = c.rest_matrices()
        asm = CcxAssembler(c, M, energy=True)
        X = c.solid_points
        root = np.flatnonzero(X[:, 0] < 1e-9)
        gen = newton.ModalGeneratorFEM(X, M, K, mode_count=6, damping_ratio=0.02, fixed_node_indices=root)
        basis = gen.build()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body_elastic(xform=wp.transform_identity(), modal_basis=basis, lock_inertia=True)
        anchor = wp.transform(wp.vec3(*X[root[0]]), wp.quat_identity())
        builder.add_joint_fixed(parent=-1, child=body, parent_xform=anchor, child_xform=anchor)
        builder.color()
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverVBD(model, iterations=8)
        snap = _snapshot(model)

        class _Mesh:
            nodes = X

        asm.mesh = getattr(asm, "mesh", _Mesh)
        ctrl = free_controller(model, solver, body, asm, gen, fixed=root)
        ctrl.relinearize(model.state(), model.control())
        for k, d in _relative_diffs(snap, model, float(M.sum()) / 3.0, ctrl.char_length).items():
            self.assertLess(d, 2.0e-5, k)
        _, f, KT = asm.evaluate(X.reshape(-1), need_tangent=True)
        self.assertLess(np.abs(f).max(), 1e-9)
        self.assertLess(spla.norm(KT - K) / spla.norm(K), 1e-9)


if __name__ == "__main__":
    unittest.main()
