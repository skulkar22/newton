# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""State transfer at an update: constrained transfer in the reduced mass metric, signed energy diagnostics,
velocity enrichment."""

from __future__ import annotations

import _fixtures  # noqa: F401  (puts the rerom package on sys.path)

# isort: split

import unittest
import warnings

import numpy as np
import rerom
from _fixtures import HAVE_CUDA, beam_assembler, build_cantilever, build_free_beam, free_controller, run
from rerom.numerics import constrain_frame_velocity, transfer_modal_velocity

import newton

FREE_FLIGHT_YOUNG = 1.0e7
CANTILEVER_YOUNG = 1.0e8
PATHS = (rerom.UpdatePath.HOST, rerom.UpdatePath.DEVICE)


def _devices():
    return ["cpu", "cuda:0"] if HAVE_CUDA else ["cpu"]


def _free_flight(asm, m, device="cpu"):
    gen, _basis, model, state, body = build_free_beam(
        asm,
        m,
        mode_qd=([0.4, 0.0, 0.2, 0.0, 0.05] + [0.0] * (m - 5)),
        twist=[0.2, -0.1, 0.3, 0.5, 0.4, 2.0],
        device=device,
    )
    jq = state.joint_q.numpy()
    jq[7] = 0.08 / np.max(np.abs(gen.modal_matrix[:, 0]))
    state.joint_q.assign(jq)
    return gen, model, state, body


def _switch(path, device="cpu", *, n_ritz=0, basis=None, steps=40):
    """Evolve a spinning, vibrating free beam and re-linearize it once (the review's free switch)."""
    asm = beam_assembler(young=FREE_FLIGHT_YOUNG)
    gen, model, state, body = _free_flight(asm, 8, device=device)
    solver = newton.solvers.SolverVBD(model, iterations=16)
    control = model.control()
    ctrl = free_controller(
        model, solver, body, asm, gen, n_ritz=n_ritz, basis=basis, policy=rerom.UpdatePolicy(path=path)
    )
    s0, _s1 = run(solver, state, model.state(), control, steps, 1.0e-3, ctrl)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = ctrl.relinearize(s0, control)
    return ctrl, info


def _random_spd(rng, m):
    A = rng.standard_normal((m, m))
    return A @ A.T + m * np.eye(m)


class TestConstrainedTransferMetric(unittest.TestCase):
    def test_identity_metric_reproduces_the_euclidean_correction(self):
        """G = I (M-orthonormal basis) gives the original Euclidean minimal-norm correction; G = None is it bit for
        bit."""
        rng = np.random.default_rng(3)
        m, k = 12, 6
        F = rng.standard_normal((k, m))
        qd = rng.standard_normal(m)
        fdot = rng.standard_normal(k)
        old = qd + F.T @ np.linalg.solve(F @ F.T, fdot - F @ qd)
        np.testing.assert_array_equal(constrain_frame_velocity(qd, F, fdot), old)
        np.testing.assert_allclose(constrain_frame_velocity(qd, F, fdot, G=np.eye(m)), old, rtol=1e-12, atol=1e-12)
        _a, info = transfer_modal_velocity(F.T @ fdot, np.eye(m), np.eye(m), np.eye(m), F=F, fdot=fdot)
        self.assertEqual(info["metric"], "identity")
        self.assertLess(info["interface_velocity_mismatch"], 1e-12)

    def test_general_metric_gives_the_constrained_minimizer(self):
        """Random SPD G: the correction satisfies F a = fdot and the KKT conditions of min 1/2 |a - a0|_G^2."""
        rng = np.random.default_rng(5)
        m, k = 10, 6
        G = _random_spd(rng, m)
        F = rng.standard_normal((k, m))
        a0 = rng.standard_normal(m)
        fdot = rng.standard_normal(k)
        a = constrain_frame_velocity(a0, F, fdot, G=G)
        kkt = np.block([[G, F.T], [F, np.zeros((k, k))]])
        ref = np.linalg.solve(kkt, np.concatenate([G @ a0, fdot]))[:m]
        np.testing.assert_allclose(a, ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(F @ a, fdot, atol=1e-10)
        # any other feasible point has a larger G-distance to a0
        _, _, Vt = np.linalg.svd(F)
        z = Vt[k:].T @ rng.standard_normal(m - k)
        d = a - a0
        self.assertLess(float(d @ G @ d), float((d + z) @ G @ (d + z)))
        # the Euclidean formula is not the G-minimizer
        e = constrain_frame_velocity(a0, F, fdot)
        self.assertGreater(float(np.max(np.abs(e - a))), 1e-3)

    def test_ill_conditioned_interface_map_is_flagged(self):
        """A rank-deficient interface map raises instead of producing an arbitrarily large velocity."""
        rng = np.random.default_rng(11)
        m = 8
        F = rng.standard_normal((6, m))
        F[5] = F[4] * (1.0 + 1e-14)
        qd = rng.standard_normal(m)
        fdot = rng.standard_normal(6)
        with self.assertRaises(ValueError):
            constrain_frame_velocity(qd, F, fdot, cond_max=1e12)
        info = {}
        constrain_frame_velocity(qd, F[:5], fdot[:5], cond_max=1e12, info=info)
        self.assertLess(info["schur_condition"], 1e12)


class TestTransferEnergyDiagnostics(unittest.TestCase):
    def test_affine_interface_correction_reports_an_energy_gain(self):
        """The review's counterexample (M = I, old velocity (1, 0), unit basis (0.1, sqrt(0.99)), interface velocity
        1): the constrained transfer preserves the interface velocity and *raises* the kinetic energy by 49.5; the
        residual energy is 49.5 too, so it must not be reported as a loss."""
        u = np.array([1.0, 0.0])
        Phi = np.array([[0.1], [np.sqrt(0.99)]])
        M = np.eye(2)
        qd, info = transfer_modal_velocity(u, Phi, M, Phi.T @ Phi, F=Phi[:1], fdot=np.array([1.0]), energies=True)
        self.assertAlmostEqual(float(qd[0]), 10.0, places=10)
        self.assertAlmostEqual(info["KE_before"], 0.5, places=12)
        self.assertAlmostEqual(info["KE_after"], 50.0, places=10)
        self.assertAlmostEqual(info["delta_KE"], 49.5, places=10)
        self.assertGreater(info["delta_KE"], 0.0)
        self.assertAlmostEqual(info["velocity_projection_residual_energy"], 49.5, places=10)
        self.assertLess(info["interface_velocity_mismatch"], 1e-14)

    def test_same_span_rotation_and_sign_flip_are_lossless(self):
        """A rotation / sign flip within the same span is an exact coordinate change: zero residual and zero
        energy change for any velocity in the span (non-identity M)."""
        rng = np.random.default_rng(2)
        n, m = 12, 4
        M = np.diag(rng.uniform(0.5, 2.0, n))
        B = rng.standard_normal((n, m))
        w, Q = np.linalg.eigh(B.T @ M @ B)
        Phi = B @ (Q / np.sqrt(w))  # M-orthonormal
        u = Phi @ rng.standard_normal(m)
        for name, R in (
            ("rotation", np.linalg.qr(rng.standard_normal((m, m)))[0]),
            ("sign_flip", np.diag([1.0, -1.0, -1.0, 1.0])),
        ):
            with self.subTest(case=name):
                P2 = Phi @ R
                qd, info = transfer_modal_velocity(u, P2, M, P2.T @ M @ P2, energies=True)
                np.testing.assert_allclose(P2 @ qd, u, atol=1e-12)
                self.assertLess(info["velocity_projection_residual_energy"], 1e-24)
                self.assertLess(abs(info["delta_KE"]), 1e-12 * info["KE_before"])

    def test_free_switch_identity_and_momentum(self):
        """Free body (actual controller switch): delta_KE = -residual to 1e-5 of KE_before, momentum preserved to
        1e-5 (relative), no position change; host and device paths."""
        for device in _devices():
            for path in PATHS:
                with self.subTest(device=device, path=path.value):
                    _, info = _switch(path, device)
                    self.assertEqual(info["transfer_mode"], "free")
                    self.assertGreater(info["velocity_projection_residual_energy"], 0.0)
                    self.assertLess(info["delta_KE"], 0.0)
                    self.assertLess(
                        abs(info["delta_KE"] + info["velocity_projection_residual_energy"]), 1.0e-5 * info["KE_before"]
                    )
                    self.assertAlmostEqual(info["delta_KE"], info["KE_after"] - info["KE_before"], places=12)
                    self.assertEqual(info["KE_lost_projection"], info["velocity_projection_residual_energy"])
                    self.assertLess(info["dP_rel"], 1.0e-5)
                    self.assertLess(info["dL_rel"], 1.0e-5)
                    self.assertLess(info["position_change_max"], 1.0e-6)
                    self.assertIsNone(info["interface_velocity_mismatch"])

    def test_prescribed_frame_mode_is_reported(self):
        """A clamped basis keeps the frame twist: the transfer mode says so, delta_KE is still the signed change."""
        asm = beam_assembler(young=CANTILEVER_YOUNG)
        for path in PATHS:
            with self.subTest(path=path.value):
                gen, _b, model, body, fixed = build_cantilever(asm, 8, damping_ratio=0.3, clamped=True)
                solver = newton.solvers.SolverVBD(model, iterations=8)
                state, _ = run(solver, model.state(), model.state(), model.control(), 40, 2.0e-3)
                ctrl = free_controller(
                    model, solver, body, asm, gen, fixed=fixed, n_ritz=2, policy=rerom.UpdatePolicy(path=path)
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    info = ctrl.relinearize(state, model.control())
                self.assertEqual(info["transfer_mode"], "prescribed_frame")
                self.assertAlmostEqual(info["delta_KE"], info["KE_after"] - info["KE_before"], places=12)


if __name__ == "__main__":
    unittest.main()
