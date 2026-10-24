# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of the IMEX hyper-reduced nonlinear modal remainder, tangent folding and the residual trigger (CPU)."""

from __future__ import annotations

import _fixtures  # noqa: F401  (puts the rerom package on sys.path)

# isort: split

import unittest

import numpy as np
import rerom
from _fixtures import LinearElementAssembler, beam_assembler, build_cantilever, free_controller
from rerom.hyper_reduction import ReducedElementForce, ecsw_greedy, ecsw_training_matrix, stratified_volume_sample
from rerom.numerics import imex_spectral_radius

import newton

YOUNG = 2.0e6  # soft 12x2x2 beam: visible sag under gravity within ~50 steps
DT = 2.0e-3


def _setup(asm=None, modes=8, **kw):
    asm = asm if asm is not None else beam_assembler(young=YOUNG)
    gen, _, model, body, fixed = build_cantilever(asm, modes, damping_ratio=0.0, gravity=-9.81, clamped=True)
    solver = newton.solvers.SolverVBD(model, iterations=16)
    control = model.control()
    ctrl = free_controller(model, solver, body, asm, gen, fixed=fixed, **kw)
    return asm, model, solver, control, ctrl


def _steps(model, solver, control, ctrl, n, s0=None, s1=None):
    s0 = model.state() if s0 is None else s0
    s1 = model.state() if s1 is None else s1
    for _ in range(n):
        ctrl.apply_prestress(control, s0)
        s0.clear_forces()
        solver.step(s0, s1, control, None, DT)
        s0, s1 = s1, s0
        ctrl.after_step(s0, control, DT)
    return s0, s1


def _no_trigger():
    return rerom.RotationTrigger(theta_tol=None)


class TestImex(unittest.TestCase):
    def test_null_behaviour(self):
        """The IMEX path with ``interval=None`` (remainder only at updates, where it is 0) reproduces the plain
        re-ROM bit for bit, updates included."""
        runs = []
        for nl in (None, rerom.NonlinearForceOptions(interval=None)):
            trig = rerom.RotationTrigger(theta_tol=np.radians(0.5))
            _, model, solver, control, ctrl = _setup(n_ritz=2, trigger=trig, nonlinear_force=nl)
            s0, _ = _steps(model, solver, control, ctrl, 60)
            runs.append((s0.joint_q.numpy().copy(), s0.joint_qd.numpy().copy(), len(ctrl.stats)))
        self.assertGreater(runs[0][2], 0)
        self.assertEqual(runs[0][2], runs[1][2])
        np.testing.assert_array_equal(runs[0][0], runs[1][0])
        np.testing.assert_array_equal(runs[0][1], runs[1][1])

    def test_full_set_force_matches_assembly(self):
        """Full element set, unit weights: F_S = Phi^T f_int(x), K_S = Phi^T K_T Phi, W_S = W."""
        asm = beam_assembler(young=YOUNG)
        rng = np.random.default_rng(0)
        Phi = np.linalg.qr(rng.standard_normal((asm.n_dof, 5)))[0]
        r = asm.mesh.nodes + 0.01 * rng.standard_normal(asm.mesh.nodes.shape)
        q = 0.05 * rng.standard_normal(5)
        hr = ReducedElementForce(asm)
        hr.set_basis(r, Phi)
        W, F, K = hr.evaluate(q, need_tangent=True, need_energy=True)
        W_a, f_a, K_a = asm.evaluate(r.reshape(-1) + Phi @ q, need_tangent=True)
        self.assertLess(np.linalg.norm(F - Phi.T @ f_a) / np.linalg.norm(Phi.T @ f_a), 1e-12)
        KK = Phi.T @ (K_a @ Phi)
        self.assertLess(np.linalg.norm(K - KK) / np.linalg.norm(KK), 1e-12)
        self.assertLess(abs(W - W_a), 1e-12 * abs(W_a))

    def test_remainder_zero_for_linear_assembler(self):
        lin = LinearElementAssembler(beam_assembler(young=YOUNG))
        _, model, solver, control, ctrl = _setup(
            asm=lin, n_ritz=0, trigger=_no_trigger(), nonlinear_force=rerom.NonlinearForceOptions()
        )
        s0, s1 = model.state(), model.state()
        ratios = []
        for _ in range(50):
            s0, s1 = _steps(model, solver, control, ctrl, 1, s0, s1)
            n_nl, _, n_tot = ctrl.cached_remainder_norms()
            ratios.append(n_nl / max(n_tot, 1e-30))
        _, model2, solver2, control2, ctrl2 = _setup(asm=lin, n_ritz=0, trigger=_no_trigger())
        s2, _ = _steps(model2, solver2, control2, ctrl2, 50)
        q1, q2 = ctrl.read_modes(s0)[0], ctrl2.read_modes(s2)[0]
        self.assertGreater(np.linalg.norm(q1), 1e-4)
        self.assertLess(max(ratios), 1e-9)
        self.assertLessEqual(np.max(np.abs(q1 - q2)), 1e-6 * np.max(np.abs(q2)))

    def test_fold_invariants(self):
        """A fold only rotates the basis: field, momentum and the force function are unchanged; afterwards
        M_red = I, K_red is the diagonal folded tangent, the remainder and its Jacobian are zero."""
        _, model, solver, control, ctrl = _setup(
            n_ritz=0, trigger=_no_trigger(), nonlinear_force=rerom.NonlinearForceOptions()
        )
        s0, s1 = _steps(model, solver, control, ctrl, 60)
        x_before = ctrl.full_field(s0).copy()
        mom_before = ctrl.nodal_momentum(s0)
        r0 = ctrl.nl_evaluate(ctrl.read_modes(s0)[0], need_tangent=True)
        Phi0 = ctrl.Phi.copy()
        ctrl.fold(s0, control)
        r1 = ctrl.nl_evaluate(ctrl.read_modes(s0)[0], need_tangent=True)
        lay = ctrl.layout
        m, Mx = ctrl.m, lay.Mx
        mm = model.elastic_mode_mass_matrix.numpy()[lay.e * Mx * Mx : (lay.e + 1) * Mx * Mx].reshape(Mx, Mx)[:m, :m]
        k_model = model.elastic_mode_stiffness.numpy()[lay.mode_slice]
        T = r1["T"]
        Q = Phi0.T @ (ctrl.M @ ctrl.Phi)
        mom_after = ctrl.nodal_momentum(s0)
        lam_scale = np.max(np.abs(ctrl.lam))
        self.assertLess(np.max(np.abs(ctrl.Phi.T @ (ctrl.M @ ctrl.Phi) - np.eye(m))), 1e-10)
        self.assertLess(np.max(np.abs(mm - np.eye(m))), 1e-5)
        self.assertLess(np.max(np.abs(k_model - ctrl.lam)) / lam_scale, 1e-6)
        self.assertLess(np.max(np.abs(T - np.diag(np.diag(T)))) / np.max(np.abs(np.diag(T))), 1e-5)
        self.assertLess(np.max(np.abs(np.diag(T) - ctrl.lam)) / lam_scale, 1e-5)
        field = np.max(np.abs(ctrl.full_field(s0) - x_before)) / np.max(np.abs(x_before - ctrl.r_ref))
        self.assertLess(field, 1e-5)
        self.assertGreater(np.linalg.norm(r0["f_nl"]) / np.linalg.norm(r0["f_tot"]), 1e-6)
        self.assertLess(np.linalg.norm(r1["f_nl"]) / np.linalg.norm(r1["f_tot"]), 1e-5)
        self.assertLess(np.linalg.norm(r1["K_exp"], 2) / lam_scale, 1e-5)
        self.assertLess(np.linalg.norm(Q @ r1["f_tot"] - r0["f_tot"]) / np.linalg.norm(r0["f_tot"]), 1e-5)
        self.assertLess(np.linalg.norm(mom_after["P"] - mom_before["P"]) / np.linalg.norm(mom_before["P"]), 1e-5)
        s0, s1 = _steps(model, solver, control, ctrl, 20, s0, s1)
        self.assertTrue(np.all(np.isfinite(ctrl.world_field(s0))))

    def test_fold_interval_runs(self):
        _, model, solver, control, ctrl = _setup(
            n_ritz=2,
            trigger=rerom.RotationTrigger(theta_tol=np.radians(5.0)),
            nonlinear_force=rerom.NonlinearForceOptions(fold_interval=5),
        )
        s0, _ = _steps(model, solver, control, ctrl, 40)
        self.assertGreaterEqual(len(ctrl.folds), 1)
        self.assertTrue(np.all(np.isfinite(ctrl.world_field(s0))))

    def test_residual_trigger_and_spectral_radius(self):
        lam = np.array([10.0, 1.0e4])
        self.assertLessEqual(imex_spectral_radius(lam, np.zeros((2, 2)), 1e-3), 1.0)
        self.assertGreater(imex_spectral_radius(lam, np.diag([0.0, 1.0e8]), 1e-3), 1.0)
        _, model, solver, control, ctrl = _setup(
            n_ritz=2,
            nonlinear_force=rerom.NonlinearForceOptions(),
            trigger=rerom.ResidualTrigger(tol=1e-6, rho_max=None, theta_cap=None),
        )
        _steps(model, solver, control, ctrl, 30)
        self.assertIn("residual", [st["trigger"] for st in ctrl.stats])
        lin = LinearElementAssembler(beam_assembler(young=YOUNG))
        _, model2, solver2, control2, ctrl2 = _setup(
            asm=lin,
            n_ritz=0,
            nonlinear_force=rerom.NonlinearForceOptions(),
            trigger=rerom.ResidualTrigger(tol=1e-6, rho_max=1.0, theta_cap=None),
        )
        _steps(model2, solver2, control2, ctrl2, 30)
        self.assertEqual(len(ctrl2.stats), 0)

    def test_sampling_helpers(self):
        asm = beam_assembler(young=YOUNG)
        ids, w = stratified_volume_sample(asm, 10, seed=3)
        self.assertEqual(ids.size, 10)
        self.assertTrue(np.all(w > 0))
        self.assertLess(abs(float(w @ asm.vol[ids]) - float(asm.vol.sum())), 1e-12 * float(asm.vol.sum()))
        rng = np.random.default_rng(1)
        X = asm.mesh.nodes
        cfgs = [X + 0.02 * rng.standard_normal(X.shape) for _ in range(4)]
        bases = [np.linalg.qr(rng.standard_normal((asm.n_dof, 4)))[0] for _ in range(4)]
        G, b = ecsw_training_matrix(asm, cfgs, bases)
        _, w2, info = ecsw_greedy(G, b, tol=1e-3)
        self.assertLessEqual(info["residual"], 1e-3)
        self.assertTrue(np.all(w2 > 0))

    def test_manifold_force_is_energy_gradient(self):
        asm = beam_assembler(young=YOUNG)
        rng = np.random.default_rng(2)
        m, k = 5, 3
        Phi = np.linalg.qr(rng.standard_normal((asm.n_dof, m)))[0]
        Theta = 0.3 * rng.standard_normal((asm.n_dof, k, k))
        Theta = 0.5 * (Theta + Theta.transpose(0, 2, 1))
        hr = ReducedElementForce(asm)
        hr.set_basis(asm.mesh.nodes, Phi)
        hr.set_manifold(Theta)
        hr.rotate(np.linalg.qr(rng.standard_normal((m, m)))[0])
        q = 0.02 * rng.standard_normal(m)
        _, F, K = hr.evaluate(q, need_tangent=True, need_energy=True)
        h = 1e-6
        E = np.eye(m)
        Ffd = np.array(
            [
                (hr.evaluate(q + h * e, need_energy=True)[0] - hr.evaluate(q - h * e, need_energy=True)[0]) / (2 * h)
                for e in E
            ]
        )
        Kfd = np.column_stack([(hr.evaluate(q + h * e)[1] - hr.evaluate(q - h * e)[1]) / (2 * h) for e in E])
        self.assertLess(np.linalg.norm(F - Ffd) / np.linalg.norm(F), 1e-6)
        self.assertLess(np.linalg.norm(K - 0.5 * (Kfd + Kfd.T)) / np.linalg.norm(K), 1e-5)

    def test_manifold_with_linear_assembler(self):
        lin = LinearElementAssembler(beam_assembler(young=YOUNG))
        _, model, solver, control, ctrl = _setup(
            asm=lin, n_ritz=0, trigger=_no_trigger(), nonlinear_force=rerom.NonlinearForceOptions(manifold=2)
        )
        _steps(model, solver, control, ctrl, 20)
        n_nl, _, n_tot = ctrl.cached_remainder_norms()
        self.assertLess(np.max(np.abs(ctrl._Theta)), 1e-8)
        self.assertLess(n_nl / n_tot, 1e-9)


if __name__ == "__main__":
    unittest.main()
