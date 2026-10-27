# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of rerom.eigen_warpack (Warpack eigensolver backends, cuDSS factorization, restricted-solver factory).

Skipped when Warpack, nvmath-python (cuDSS) or a CUDA device is missing; the import-error test always runs.
"""

from __future__ import annotations

import _fixtures  # noqa: F401  (puts the rerom package on sys.path)

# isort: split

import importlib.util
import sys
import unittest
import warnings
from unittest import mock

import numpy as np
import rerom
import scipy.linalg as sla  # noqa: TID253
import scipy.sparse as sp  # noqa: TID253
import warp as wp
from _fixtures import HAVE_CUDA, beam_assembler, build_cantilever, free_controller
from rerom import eigen_warpack as ew
from rerom.assembly_warp import WarpTetAssembler
from rerom.eigen import RestrictedArpackBackend, ShiftedRestrictedSolver
from rerom.numerics import m_orthonormal_rigid, nodal_rigid_vectors, rigid_properties

import newton
from newton._src.sim import modal_sparse

HAVE_WARPACK = importlib.util.find_spec("warpack") is not None and importlib.util.find_spec("nvmath") is not None
GPU = "cuda:0"
SKIP = unittest.skipUnless(HAVE_CUDA and HAVE_WARPACK, "needs CUDA, warpack and nvmath-python (cuDSS)")


class TestSturmPerturbedPivots(unittest.TestCase):
    def test_perturbed_pivots_skip_the_count(self):
        """A Sturm factorization with perturbed pivots returns no inertia count (check skipped, recorded), an
        unperturbed one returns ``below + d`` (stubbed pencil; no CUDA, Warpack or cuDSS needed)."""
        from types import SimpleNamespace  # noqa: PLC0415

        backend = object.__new__(ew.WarpackKrylovBackend)
        backend.sturm_check = True
        vals = np.array([1.0, 2.0, 3.0, 10.0, 11.0])
        for pivots, expected in ((2, (None, None)), (0, (3 + 6, 6.5))):
            with self.subTest(perturbed_pivots=pivots):
                aux = SimpleNamespace(perturbed_pivots=pivots, timings={"factorization": 0.0})
                p = SimpleNamespace(
                    d=6, factor=lambda name, tau, a=aux: a, restriction={"aux": {"negA": 0, "negS": 0, "below": 3}}
                )
                info = {}
                self.assertEqual(backend._sturm(p, vals, 3, info), expected)
                self.assertEqual(info["sturm"]["perturbed_pivots"], pivots)
                self.assertEqual("warning" in info["sturm"], pivots > 0)


def _pencil(kind: str, young=1.0e9):
    """Pre-stressed beam tangent (rigid rotations not in its null space) on the free DOFs."""
    asm = beam_assembler(young=young)
    X = asm.mesh.nodes
    x = X.copy()
    x[:, 2] += 0.02 * (X[:, 0] / np.ptp(X[:, 0])) ** 2  # bent reference
    _, _, K = asm.evaluate(x.reshape(-1))
    K = sp.csr_matrix(K)
    K = sp.csr_matrix(0.5 * (K + K.T))
    M = sp.csr_matrix(asm.mass_matrix)
    if kind == "clamped":
        fixed = np.zeros(K.shape[0], bool)
        for a in asm.mesh.node_sets["root"]:
            fixed[3 * a : 3 * a + 3] = True
        free = np.nonzero(~fixed)[0]
        return asm, x, sp.csr_matrix(K[free][:, free]), sp.csr_matrix(M[free][:, free]), None, free
    _, com, _ = rigid_properties(M, x)
    Y = m_orthonormal_rigid(nodal_rigid_vectors(x, com), M)
    return asm, x, K, M, Y, np.arange(K.shape[0])


def _dense_restricted(K, M, Y, count):
    Kd, Md = K.toarray(), M.toarray()
    if Y is not None:
        Q, _ = np.linalg.qr(M @ Y, mode="complete")
        Z = Q[:, Y.shape[1] :]
        Kd, Md = Z.T @ Kd @ Z, Z.T @ Md @ Z
    else:
        Z = np.eye(K.shape[0])
    vals, vecs = sla.eigh(0.5 * (Kd + Kd.T), 0.5 * (Md + Md.T), subset_by_index=[0, count - 1])
    return vals, Z @ vecs


def _max_angle(V1, V2, M):
    def orth(V):
        G = V.T @ (M @ V)
        w, U = np.linalg.eigh(0.5 * (G + G.T))
        return V @ (U / np.sqrt(w))

    Q1, Q2 = orth(V1), orth(V2)
    R = Q2 - Q1 @ (Q1.T @ (M @ Q2))
    s2 = np.linalg.eigvalsh(R.T @ (M @ R))
    return float(np.arcsin(min(1.0, np.sqrt(max(float(np.max(s2)), 0.0)))))


class TestImportError(unittest.TestCase):
    def test_missing_warpack_message(self):
        with mock.patch.dict(sys.modules, {"warpack": None}):
            with self.assertRaisesRegex(ImportError, "Warpack"):
                ew._require()


@SKIP
class TestCudssFactor(unittest.TestCase):
    def test_solve_inertia_and_refactor_reuse(self):
        _, _, K, M, Y, _ = _pencil("clamped")
        ctx = ew.WarpackContext(device=GPU)
        s = ctx(K, M, Y)
        p = s.pencil
        f = p.factors["primary"]
        self.assertTrue(f.analyzed)
        analysis_time = f.timings["analysis"]
        # solve: cuDSS L D L^T of K - sigma M against SciPy
        A = (K - s.sigma * M).tocsc()
        b = np.random.default_rng(0).standard_normal((3, K.shape[0]))
        x = wp.array(b, dtype=wp.float64, device=GPU)
        y = wp.zeros_like(x)
        f.apply(x, y)
        r = A @ y.numpy().T - b.T
        self.assertLess(np.max(np.linalg.norm(r, axis=0) / np.linalg.norm(b, axis=1)), 1e-10)
        # inertia at a shift between eigenvalues 3 and 4 == SuperLU symmetric-mode count == 3
        vals, _ = _dense_restricted(K, M, None, 6)
        tau = 0.5 * (vals[2] + vals[3])
        p.factor("aux", tau)
        self.assertEqual(p.factors["aux"].negative_pivot_count, 3)
        self.assertEqual(modal_sparse.ShiftedFactorization(K.tocsc(), M, tau).negative_pivot_count, 3)
        self.assertEqual(p.factors["aux"].perturbed_pivots, 0)
        # refactor with new values: numeric phase only (analysis kept), solves follow the new matrix
        K2 = sp.csr_matrix(K * 1.5)
        count0 = f.factor_count
        s2 = ctx(K2, M, Y)
        self.assertIs(s2.pencil, p)
        self.assertEqual(f.timings["analysis"], analysis_time)
        self.assertEqual(f.factor_count, count0 + 1)
        f.apply(x, y)
        A2 = (K2 - s2.sigma * M).tocsc()
        r = A2 @ y.numpy().T - b.T
        self.assertLess(np.max(np.linalg.norm(r, axis=0) / np.linalg.norm(b, axis=1)), 1e-10)

    def test_restricted_solver_matches_host(self):
        for kind in ("clamped", "free"):
            with self.subTest(kind=kind):
                _, _, K, M, Y, _ = _pencil(kind)
                g = ew.WarpackContext(device=GPU)(K, M, Y)
                h = ShiftedRestrictedSolver(K, M, Y, symmetric=True)
                self.assertAlmostEqual(g.sigma / h.sigma, 1.0, places=12)
                b = np.random.default_rng(1).standard_normal((K.shape[0], 2))
                xg, xh = g.solve(b), h.solve(b)
                self.assertLess(np.max(np.abs(xg - xh)) / np.max(np.abs(xh)), 1e-9)
                if Y is not None:
                    self.assertLess(np.max(np.abs(Y.T @ (M @ xg))), 1e-9 * np.max(np.abs(xg)))


@SKIP
class TestWarpackEigen(unittest.TestCase):
    def _check(self, backend_cls, kind, count=10, **kw):
        _, _, K, M, Y, _ = _pencil(kind)
        ref = RestrictedArpackBackend().solve(K.tocsc(), M, count + 4, deflation=Y)
        ctx = ew.WarpackContext(device=GPU)
        b = backend_cls(ctx, **kw)
        if backend_cls is ew.WarpackLOBPCGBackend:
            b.previous_vectors = np.asarray(ref.vectors)[:, :count] + 1e-3 * np.random.default_rng(2).standard_normal(
                (K.shape[0], count)
            )
        self.assertIsInstance(b, newton.ModalEigensolverBackend)
        res = b.solve(K, M, count, deflation=Y)
        d = 0 if Y is None else Y.shape[1]
        vals, V = modal_sparse.validate_eigen_result(
            res, size=K.shape[0], count=count, deflated=d, supports_inertia_count=True
        )
        self.assertIsNotNone(res.inertia_count)
        rv = np.asarray(ref.values)
        np.testing.assert_allclose(vals[:count], rv[:count], rtol=1e-7)
        self.assertLess(np.max(np.abs(V.T @ (M @ V) - np.eye(V.shape[1]))), 1e-10)
        if Y is not None:
            self.assertLess(np.max(np.abs(Y.T @ (M @ V))), 1e-9)
        mp = count
        while mp < rv.size and rv[mp] - rv[mp - 1] <= 1e-6 * abs(rv[mp]):
            mp += 1
        self.assertLess(_max_angle(np.asarray(ref.vectors)[:, :mp], V[:, :mp], M), 1e-5)
        R = K @ V - (M @ V) * vals
        if Y is not None:
            R = R - (M @ Y) @ (Y.T @ R)
        self.assertLess(np.max(np.linalg.norm(R, axis=0) / np.linalg.norm(K @ V, axis=0)), 1e-6)
        below = int(res.inertia_count) - d
        self.assertEqual(below, int(np.count_nonzero(rv < res.inertia_shift)))
        return b

    def test_krylov_clamped(self):
        self._check(ew.WarpackKrylovBackend, "clamped")

    def test_krylov_free(self):
        self._check(ew.WarpackKrylovBackend, "free")

    def test_lobpcg_warm_clamped(self):
        self._check(ew.WarpackLOBPCGBackend, "clamped")

    def test_lobpcg_warm_free(self):
        self._check(ew.WarpackLOBPCGBackend, "free")

    def test_indefinite_reports_negative_eigenvalues(self):
        """K - alpha M with alpha between eigenvalues 2 and 3: two negative restricted eigenvalues are returned."""
        _, _, K, M, _, _ = _pencil("clamped")
        vals0, _ = _dense_restricted(K, M, None, 8)
        alpha = 0.5 * (vals0[1] + vals0[2])
        Ki = sp.csr_matrix(K - alpha * M)
        ref_vals, ref_V = _dense_restricted(Ki, M, None, 8)
        self.assertEqual(int(np.count_nonzero(ref_vals < 0.0)), 2)
        b = ew.WarpackKrylovBackend(ew.WarpackContext(device=GPU))
        res = b.solve(Ki, M, 6)
        vals, V = modal_sparse.validate_eigen_result(
            res, size=Ki.shape[0], count=6, deflated=0, supports_inertia_count=True
        )
        self.assertGreater(b.last_info["below_sigma"], 0)  # the Ritz shift sat above two eigenvalues
        self.assertEqual(int(np.count_nonzero(vals < 0.0)), 2)
        np.testing.assert_allclose(vals[:6], ref_vals[:6], rtol=1e-7, atol=1e-7 * abs(ref_vals[5]))
        self.assertLess(_max_angle(ref_V[:, :6], V[:, :6], M), 1e-5)


@SKIP
class TestControllerIntegration(unittest.TestCase):
    def test_device_handoff_matches_host_bitwise(self):
        """WarpTetAssembler BSR values -> symmetrized free-node BSR: identical to the host 0.5 (K + K^T)[free][:, free]."""
        asm_np, x, _, _, _, free = _pencil("clamped")
        mesh = asm_np.mesh
        asm = WarpTetAssembler(
            mesh.nodes, mesh.tets, mesh.density, mesh.young, mesh.poisson, "stvk", device=GPU, _mesh=mesh
        )
        _, _, Kb = asm.evaluate(x.reshape(-1), tangent_format="bsr")
        K = asm.to_scipy_csr(Kb)
        K = 0.5 * (K + K.T)
        Kf = sp.csr_matrix(K[free][:, free])
        Mf = sp.csr_matrix(asm.mass_matrix)[free][:, free]
        ctx = ew.WarpackContext(device=GPU, assembler=asm, free_dofs=free)
        ctx(Kf, Mf, None, device_tangent=asm.tangent_values_device())
        dev_blocks = ctx.pencil.Kv.numpy()
        host_blocks = ctx.pencil.s.host_blocks(Kf)
        self.assertTrue(np.array_equal(dev_blocks, host_blocks))

    def test_controller_update_matches_arpack(self):
        """One update of a clamped cantilever (device path, Warp assembler on CUDA): Warpack vs the default ARPACK."""
        results = {}
        for name in ("arpack", "warpack-ks", "warpack-lobpcg"):
            asm_np = beam_assembler(young=1.0e8)
            mesh = asm_np.mesh
            asm = WarpTetAssembler(
                mesh.nodes, mesh.tets, mesh.density, mesh.young, mesh.poisson, "stvk", device=GPU, _mesh=mesh
            )
            gen, _, model, body, fixed = build_cantilever(asm_np, 8, device=GPU)
            solver = newton.solvers.SolverVBD(model, iterations=4)
            factory = eig = None
            if name != "arpack":
                n = 3 * mesh.num_nodes
                fx = np.zeros(n, bool)
                for a in fixed:
                    fx[3 * a : 3 * a + 3] = True
                factory = ew.WarpackContext(device=GPU, assembler=asm, free_dofs=np.nonzero(~fx)[0])
                eig = ew.WarpackKrylovBackend(factory) if name == "warpack-ks" else ew.WarpackLOBPCGBackend(factory)
            basis = rerom.BasisOptions(n_ritz=2, restricted_solver_factory=factory)
            ctrl = free_controller(
                model,
                solver,
                body,
                asm,
                gen,
                fixed=fixed,
                basis=basis,
                **({"eigensolver": eig} if eig is not None else {}),
            )
            state = model.state()
            jq = state.joint_q.numpy()
            jq[ctrl.q_start + 7 : ctrl.q_start + 9] = [0.02, -0.01]  # a deformed reference
            state.joint_q.assign(jq)
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # no skipped-Sturm warning
                info = ctrl.relinearize(state)
            results[name] = (np.asarray(info["eigen_raw"]), ctrl.Phi.copy(), info)
            if factory is not None:
                self.assertIsNotNone(info["eigen_info"].get("inertia_count"))
                self.assertGreater(info["timings"]["factorization"], 0.0)
                # one context, one factorization object kept across updates (numeric refactor only)
                p = factory.pencil
                f = p.factors["primary"]
                n_fact = f.factor_count
                ctrl.relinearize(state)
                self.assertIs(factory.pencil, p)
                self.assertEqual(f.factor_count, n_fact + 1)
        lam_a, Phi_a, _ = results["arpack"]
        M = sp.csr_matrix(beam_assembler(young=1.0e8).mass_matrix)
        for name in ("warpack-ks", "warpack-lobpcg"):
            lam, Phi, _ = results[name]
            np.testing.assert_allclose(lam[:8], lam_a[:8], rtol=1e-7)
            self.assertLess(_max_angle(Phi_a, Phi, M), 1e-5)


if __name__ == "__main__":
    unittest.main()
