# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Raw vs operational tangent spectrum (IndefiniteGuard.CLAMP) and the energies of the operational model.

Every consumer of the modal stiffness (solver arrays, damping, ROM strain energy, the IMEX linear term and
remainder, event energies) must describe the same operational model; the raw spectrum is kept for diagnostics.
"""

from __future__ import annotations

import _fixtures  # noqa: F401  (puts the rerom package on sys.path)

# isort: split

import unittest
import warnings

import numpy as np
import rerom
from _fixtures import (
    HAVE_CUDA,
    PrestressedQuadraticAssembler,
    beam_assembler,
    build_cantilever,
    build_free_beam,
    free_controller,
    run,
    set_modal_coordinates,
)

import newton

FREE_FLIGHT_YOUNG = 1.0e7
COMPRESSION = np.array([0.98, 1.0, 1.0])  # 2 % axial compression: indefinite restricted tangent
DAMPING_RATIO = 0.05
PATHS = (rerom.UpdatePath.HOST, rerom.UpdatePath.DEVICE)


def _devices():
    return ["cpu", "cuda:0"] if HAVE_CUDA else ["cpu"]


def _dyadic_q(m, scale=2.0**-7):
    """Modal coordinates exactly representable in float32 (central differences of a quadratic stay exact)."""
    return scale * np.array([(-1.0) ** i * 2.0 ** -(i % 3) for i in range(m)])


def _clamped_update(path, *, device="cpu", override=True, nonlinear_force=None, m=6):
    """Free beam re-linearized at an indefinite (2 % compressed) tangent with IndefiniteGuard.CLAMP."""
    inner = beam_assembler(young=FREE_FLIGHT_YOUNG)
    x_c = inner.mesh.nodes * COMPRESSION
    asm = inner if override else PrestressedQuadraticAssembler(inner, x_c)
    gen, _b, model, state, body = build_free_beam(inner, m, damping_ratio=DAMPING_RATIO, shape=False, device=device)
    solver = newton.solvers.SolverVBD(model, iterations=4)
    policy = rerom.UpdatePolicy(guard_indefinite=rerom.IndefiniteGuard.CLAMP, path=path)
    ctrl = free_controller(model, solver, body, asm, gen, policy=policy, nonlinear_force=nonlinear_force)
    control = model.control()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = ctrl.relinearize(
            state, control, stiffness_override=inner.evaluate(x_c.reshape(-1)) if override else None
        )
    return ctrl, model, state, info


def _written(model, m):
    mm = model.elastic_mode_mass.numpy()[:m].astype(np.float64)
    return (
        model.elastic_mode_stiffness.numpy()[:m].astype(np.float64) / mm,
        model.elastic_mode_damping.numpy()[:m].astype(np.float64),
        mm,
    )


def _fd_gradient(ctrl, state, q, h):
    """Central-difference gradient of ``ctrl.rom_strain_energy`` at ``q`` (float32-exact steps)."""
    g = np.zeros(ctrl.m)
    for i in range(ctrl.m):
        e = np.zeros(ctrl.m)
        e[i] = h
        set_modal_coordinates(state, ctrl, q=q + e)
        wp_ = ctrl.rom_strain_energy(state)
        set_modal_coordinates(state, ctrl, q=q - e)
        wm = ctrl.rom_strain_energy(state)
        g[i] = (wp_ - wm) / (2.0 * h)
    set_modal_coordinates(state, ctrl, q=q)
    return g


class TestOperationalSpectrum(unittest.TestCase):
    def test_clamp_writes_and_stores_the_operational_spectrum(self):
        """CLAMP (stiffness_override): the controller's ``lam`` is what the solver steps, ``lam_raw`` keeps the
        negative raw values, and the damping is computed from the operational spectrum."""
        for device in _devices():
            for path in PATHS:
                with self.subTest(device=device, path=path.value):
                    ctrl, model, _state, info = _clamped_update(path, device=device)
                    m = ctrl.m
                    lam_w, c_w, mm = _written(model, m)
                    raw = np.asarray(info["eigenvalues"])
                    self.assertTrue(np.any(raw < 0.0))
                    self.assertTrue(np.all(ctrl.lam > 0.0))
                    np.testing.assert_allclose(ctrl.lam, lam_w, rtol=2.0e-6)
                    c_expected = 2.0 * DAMPING_RATIO * np.sqrt(mm * ctrl.lam * mm)
                    np.testing.assert_allclose(c_w, c_expected, rtol=2.0e-6)
                    np.testing.assert_array_equal(raw, ctrl.lam_raw)
                    np.testing.assert_allclose(info["eigenvalues_operational"], ctrl.lam)
                    self.assertLess(info["lam_raw_min"], 0.0)
                    self.assertGreater(info["lam_operational_min"], 0.0)
                    self.assertEqual(info["clamp"]["count"], int(np.count_nonzero(ctrl.lam_raw <= 0.0)))

    def test_energy_gradient_is_the_operational_force(self):
        """Away from q = 0 the ROM strain energy's gradient is ``g0 + Lambda_op q`` (what the solver steps)."""
        for path in PATHS:
            with self.subTest(path=path.value):
                ctrl, model, state, _ = _clamped_update(path)
                m = ctrl.m
                lam_w, _, _ = _written(model, m)
                q = _dyadic_q(m)
                grad = _fd_gradient(ctrl, state, q, 2.0**-9)
                force = ctrl.g0 + lam_w * q
                scale = float(np.max(np.abs(force)))
                np.testing.assert_allclose(grad, force, atol=1.0e-5 * scale)

    def test_clamp_with_imex_uses_one_operational_split(self):
        """CLAMP + IMEX: ``f_tot'(0) = Lambda_op`` (the remainder does not restore the raw negative stiffness),
        ``f_lin`` uses the operational spectrum, and the model energy is the potential of ``f_tot``."""
        for path in PATHS:
            with self.subTest(path=path.value):
                ctrl, model, state, info = _clamped_update(
                    path, override=False, nonlinear_force=rerom.NonlinearForceOptions(interval=1)
                )
                m = ctrl.m
                lam_w, _, _ = _written(model, m)
                self.assertTrue(np.any(np.asarray(info["eigenvalues"]) < 0.0))
                scale = float(np.max(np.abs(lam_w)))
                r0 = ctrl.nl_evaluate(np.zeros(m), need_tangent=True)
                np.testing.assert_allclose(r0["T"], np.diag(lam_w), atol=1.0e-6 * scale)
                self.assertLess(float(np.max(np.abs(r0["K_exp"]))), 1.0e-6 * scale)
                q = _dyadic_q(m)
                r = ctrl.nl_evaluate(q)
                np.testing.assert_allclose(r["f_lin"], ctrl.g0 + lam_w * q, rtol=1.0e-6)
                # quadratic assembler: the operational remainder vanishes identically
                self.assertLess(float(np.linalg.norm(r["f_nl"])), 1.0e-6 * float(np.linalg.norm(r["f_lin"])))
                grad = _fd_gradient(ctrl, state, q, 2.0**-9)
                np.testing.assert_allclose(grad, r["f_tot"], atol=1.0e-5 * float(np.max(np.abs(r["f_tot"]))))

    def test_keep_guard_is_unchanged(self):
        """KEEP: operational == raw (negative values written as they are)."""
        inner = beam_assembler(young=FREE_FLIGHT_YOUNG)
        gen, _b, model, state, body = build_free_beam(inner, 6, shape=False)
        solver = newton.solvers.SolverVBD(model, iterations=4)
        ctrl = free_controller(model, solver, body, inner, gen, policy=rerom.UpdatePolicy(path=rerom.UpdatePath.HOST))
        override = inner.evaluate((inner.mesh.nodes * COMPRESSION).reshape(-1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctrl.relinearize(state, stiffness_override=override)
        self.assertTrue(np.any(ctrl.lam < 0.0))
        np.testing.assert_array_equal(ctrl.lam, ctrl.lam_raw)


class TestUpdateEnergies(unittest.TestCase):
    def _sagging_cantilever(self, path, nonlinear_force):
        asm = beam_assembler(young=2.0e6)  # soft: large sag, visible nonlinear remainder
        gen, _b, model, body, fixed = build_cantilever(asm, 8, damping_ratio=0.0, clamped=True)
        solver = newton.solvers.SolverVBD(model, iterations=16)
        control = model.control()
        ctrl = free_controller(
            model,
            solver,
            body,
            asm,
            gen,
            fixed=fixed,
            n_ritz=2,
            trigger=rerom.RotationTrigger(theta_tol=None),
            nonlinear_force=nonlinear_force,
            policy=rerom.UpdatePolicy(path=path),
        )
        s0, _ = run(solver, model.state(), model.state(), control, 60, 2.0e-3, ctrl, state_arg=True)
        return ctrl, s0, control

    def test_imex_update_reports_the_model_energy(self):
        """With the IMEX remainder (full element set) the update reports the corrected model's energy, which is the
        nonlinear energy at the update point, next to the quadratic model energy, which is not."""
        for path in PATHS:
            with self.subTest(path=path.value):
                ctrl, s0, control = self._sagging_cantilever(path, rerom.NonlinearForceOptions(interval=1))
                q, _ = ctrl.read_modes(s0)
                quadratic = ctrl.W_ref + float(ctrl.g0 @ q) + 0.5 * float(np.sum(ctrl.lam * q * q))
                model_energy = ctrl.rom_strain_energy(s0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    info = ctrl.relinearize(s0, control)
                W = info["W_nonlinear"]
                self.assertEqual(info["W_rom_quadratic"], quadratic)
                self.assertEqual(info["W_rom"], info["W_rom_quadratic"])
                self.assertAlmostEqual(info["W_rom_model"], model_energy, delta=1.0e-12 * abs(W))
                # quadratic model off by ~57 % here; the corrected model reproduces W to roundoff (~1e-8 relative)
                self.assertLess(abs(info["W_rel_error_model"]), 1.0e-6)
                self.assertGreater(abs(info["W_rel_error"]), 0.1)

    def test_plain_update_model_energy_is_quadratic(self):
        ctrl, s0, control = self._sagging_cantilever(rerom.UpdatePath.HOST, None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = ctrl.relinearize(s0, control)
        self.assertEqual(info["W_rom_model"], info["W_rom_quadratic"])
        self.assertEqual(info["W_rel_error_model"], info["W_rel_error"])


class TestFoldGuard(unittest.TestCase):
    def _folding(self, path, guard):
        asm = beam_assembler(young=2.0e6)
        gen, _b, model, body, fixed = build_cantilever(asm, 8, damping_ratio=0.0, clamped=True)
        solver = newton.solvers.SolverVBD(model, iterations=16)
        control = model.control()
        ctrl = free_controller(
            model,
            solver,
            body,
            asm,
            gen,
            fixed=fixed,
            n_ritz=2,
            trigger=rerom.RotationTrigger(theta_tol=None),
            nonlinear_force=rerom.NonlinearForceOptions(interval=1),
            policy=rerom.UpdatePolicy(path=path, guard_indefinite=guard),
        )
        s0, _ = run(solver, model.state(), model.state(), control, 30, 2.0e-3, ctrl, state_arg=True)
        # injected indefinite hyper-reduced tangent (a negative direction the fold re-diagonalizes)
        ctrl._nl_D = ctrl._nl_D.copy()
        ctrl._nl_D[0, 0] -= 3.0 * ctrl.lam[0]
        return ctrl, model, s0, control

    def test_clamp_fold_writes_no_negative_stiffness(self):
        """CLAMP + fold: eig(T) has a negative value, the written stiffness stays positive, lam_raw keeps the raw
        value, f_tot and the model energy at q_n are unchanged and f_tot'(q_n) is the operational spectrum."""
        for path in PATHS:
            with self.subTest(path=path.value):
                ctrl, model, s0, control = self._folding(path, rerom.IndefiniteGuard.CLAMP)
                q, _ = ctrl.read_modes(s0)
                q = q.astype(np.float64)
                f_before = ctrl.nl_evaluate(q)["f_tot"]
                w_before = ctrl.rom_energies(q)[1]
                Phi_old = ctrl.Phi.copy()
                info = ctrl.fold(s0, control)
                m = ctrl.m
                lam_w, _, _ = _written(model, m)
                self.assertTrue(np.all(lam_w > 0.0))
                self.assertLess(info["lam_min"], 0.0)
                self.assertLess(float(ctrl.lam_raw.min()), 0.0)
                np.testing.assert_allclose(ctrl.lam, lam_w, rtol=2.0e-6)
                self.assertEqual(ctrl.clamp_log[-1]["source"], "fold")
                Q = Phi_old.T @ (ctrl.M @ ctrl.Phi)  # the fold rotation (M-orthonormal bases)
                q_rot = Q.T @ q
                r = ctrl.nl_evaluate(q_rot, need_tangent=True)
                scale = float(np.max(np.abs(ctrl.lam)))
                np.testing.assert_allclose(r["T"], np.diag(ctrl.lam), atol=1.0e-6 * scale)
                np.testing.assert_allclose(r["f_tot"], Q.T @ f_before, atol=1.0e-8 * float(np.linalg.norm(f_before)))
                # the cached (lagged) remainder is the remainder at q_n, so the step force is f_tot(q_n)
                fscale = float(np.linalg.norm(r["f_tot"]))
                np.testing.assert_allclose(ctrl._fnl, r["f_nl"], atol=1.0e-9 * fscale)
                f_step = ctrl.g0 + ctrl.lam * q_rot + ctrl._fnl
                np.testing.assert_allclose(f_step, r["f_tot"], atol=1.0e-9 * fscale)
                np.testing.assert_allclose(-ctrl.modal_joint_force(), ctrl.g0 + ctrl._fnl, atol=1.0e-12 * fscale)
                self.assertAlmostEqual(ctrl.rom_energies(q_rot)[1], w_before, delta=1.0e-9 * max(abs(w_before), 1.0))

    def test_reject_guard_skips_the_fold(self):
        for path in PATHS:
            with self.subTest(path=path.value):
                ctrl, model, s0, control = self._folding(path, rerom.IndefiniteGuard.REJECT)
                k0 = model.elastic_mode_stiffness.numpy().copy()
                Phi0, jq0 = ctrl.Phi.copy(), s0.joint_q.numpy().copy()
                info = ctrl.fold(s0, control)
                self.assertEqual(info["skipped"], "negative_stiffness")
                np.testing.assert_array_equal(model.elastic_mode_stiffness.numpy(), k0)
                np.testing.assert_array_equal(ctrl.Phi, Phi0)
                np.testing.assert_array_equal(s0.joint_q.numpy(), jq0)
                self.assertEqual(ctrl.steps_since_fold, 0)


if __name__ == "__main__":
    unittest.main()
