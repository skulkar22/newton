# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of rerom.ReROMController with the free-body adapter (CPU; one CUDA case when available).

Run from the Newton checkout: ``uv run python -m unittest discover experimental/rerom/tests``.
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
    LinearStubAssembler,
    beam_assembler,
    build_cantilever,
    build_free_beam,
    free_controller,
    run,
)
from rerom.numerics import m_orthonormal_rigid, nodal_rigid_vectors, rigid_properties

import newton

FREE_FLIGHT_YOUNG = 1.0e7  # soft enough that the modes survive implicit-Euler damping at dt = 1 ms
CANTILEVER_YOUNG = 1.0e8  # Euler-Bernoulli tip sag ~ 9 % of L for the 1 m x 0.06 x 0.04 beam

_ARRAYS = [
    "elastic_mode_mass",
    "elastic_mode_stiffness",
    "elastic_mode_damping",
    "elastic_mode_coupling_linear",
    "elastic_mode_coupling_angular",
    "elastic_mode_coupling_centrifugal",
    "elastic_mode_coupling_coriolis",
    "elastic_mode_coupling_quadratic",
    "elastic_mode_mass_matrix",
    "elastic_endpoint_phi",
    "elastic_endpoint_psi",
    "elastic_render_point_phi",
    "elastic_render_point_local",
    "elastic_shape_vertex_phi",
    "elastic_shape_vertex_local",
    "joint_X_p",
    "joint_X_c",
    "body_com",
    "body_mass",
    "body_inertia",
    "body_inv_mass",
    "body_inv_inertia",
]


def _snapshot(model):
    return {k: getattr(model, k).numpy().copy() for k in _ARRAYS if getattr(model, k) is not None}


def _relative_diffs(snap, model, mass, length):
    """Max abs difference relative to the array magnitude, with a physical floor for arrays that are zero up to
    roundoff for mass-orthonormal free-free modes (S ~ sqrt(m), G and B ~ sqrt(m) L)."""
    floors = {
        "elastic_mode_coupling_linear": np.sqrt(mass),
        "elastic_mode_coupling_angular": np.sqrt(mass) * length,
        "elastic_mode_coupling_centrifugal": np.sqrt(mass) * length,
    }
    out = {}
    for k, v in snap.items():
        new = getattr(model, k).numpy()
        diff = float(np.max(np.abs(new.astype(np.float64) - v))) if v.size else 0.0
        out[k] = diff / max(float(np.max(np.abs(v))) if v.size else 0.0, floors.get(k, 1.0e-12))
    return out


def _free_flight(asm, m, device="cpu"):
    gen, _basis, model, state, body = build_free_beam(
        asm,
        m,
        mode_qd=([0.4, 0.0, 0.2, 0.0, 0.05] + [0.0] * (m - 5)),
        twist=[0.2, -0.1, 0.3, 0.5, 0.4, 2.0],
        device=device,
    )
    jq = state.joint_q.numpy()
    jq[7] = 0.08 / np.max(np.abs(gen.modal_matrix[:, 0]))  # first bending mode, ~8 % L peak displacement
    state.joint_q.assign(jq)
    return gen, model, state, body


class TestJointForceSign(unittest.TestCase):
    def test_modal_joint_f_equilibrium_is_plus_f_over_k(self):
        """Newton's modal row is ``k q - joint_f``: static equilibrium ``q = +joint_f / k`` (pre-stress sign)."""
        asm = beam_assembler()
        _, _, model, state, _ = build_free_beam(asm, 1, damping_ratio=1.0, shape=False)
        k = float(model.elastic_mode_stiffness.numpy()[0])
        target = 1.0e-3
        control = model.control()
        jf = control.joint_f.numpy()
        jf[6] = target * k
        control.joint_f.assign(jf)
        solver = newton.solvers.SolverVBD(model, iterations=8)
        s0, _ = run(solver, state, model.state(), control, 300, 1.0e-3)
        self.assertAlmostEqual(float(s0.joint_q.numpy()[7]) / target, 1.0, delta=1.0e-2)


class TestNullUpdate(unittest.TestCase):
    def _null_arrays(self, case):
        asm = beam_assembler()
        if case == "free":
            gen, _, model, state, body = build_free_beam(asm, 6, damping_ratio=0.02, derive_psi=True)
            fixed = None
        else:
            gen, _, model, body, fixed = build_cantilever(asm, 6, damping_ratio=0.02, clamped=(case == "cantilever"))
            state = model.state()
        solver = newton.solvers.SolverVBD(model, iterations=8)
        snap = _snapshot(model)
        ctrl = free_controller(model, solver, body, asm, gen, fixed=fixed)
        ctrl.relinearize(state, model.control())
        return _relative_diffs(snap, model, float(asm.mass_matrix.sum()) / 3.0, ctrl.char_length)

    def test_null_update_reproduces_arrays(self):
        """At q = 0 every rewritten array matches the generator/builder values to <= 2e-5 relative (float32)."""
        for case in ("free", "cantilever", "free_joint"):
            with self.subTest(case=case):
                for k, d in self._null_arrays(case).items():
                    self.assertLess(d, 2.0e-5, f"{k}: relative difference {d:.3e}")

    def _trajectory(self, update, device="cpu"):
        asm = beam_assembler()
        gen, _, model, state, body = build_free_beam(
            asm,
            6,
            mode_q=[0.0] * 6,
            mode_qd=[0.05, 0.0, 0.02, 0.0, 0.0, 0.0],
            twist=[0.1, 0.0, 0.05, 0.0, 0.3, 1.0],
            device=device,
        )
        solver = newton.solvers.SolverVBD(model, iterations=8)
        control = model.control()
        ctrl = free_controller(model, solver, body, asm, gen)
        if update:
            ctrl.relinearize(state, control)
        s0, _ = run(solver, state, model.state(), control, 200, 1.0e-3, ctrl)
        return s0.joint_q.numpy().copy(), s0.joint_qd.numpy().copy(), ctrl.world_field(s0)

    def test_null_update_leaves_trajectory_unchanged(self):
        """A spinning, vibrating free beam: a null update at t = 0 leaves the 200-step trajectory unchanged."""
        a, b = self._trajectory(False), self._trajectory(True)
        self.assertLess(float(np.max(np.abs(a[2] - b[2]))), 1.0e-6)
        self.assertLess(float(np.max(np.abs(a[0] - b[0]))), 1.0e-6)
        self.assertLess(float(np.max(np.abs(a[1] - b[1]))), 1.0e-4)

    @unittest.skipUnless(HAVE_CUDA, "CUDA device not available")
    def test_null_update_trajectory_cuda(self):
        """Same null-update check with the Newton model on the GPU (host-side controller)."""
        a, b = self._trajectory(False, "cuda:0"), self._trajectory(True, "cuda:0")
        self.assertLess(float(np.max(np.abs(a[2] - b[2]))), 1.0e-5)


class TestMomentumAcrossSwitch(unittest.TestCase):
    def test_switch_conserves_momentum(self):
        """Mid-flight update of a spinning, vibrating free beam: the M-orthogonal split conserves linear and angular
        momentum, never adds kinetic energy, and the solver keeps conserving afterwards."""
        for n_ritz in (0, 2):
            with self.subTest(n_ritz=n_ritz):
                asm = beam_assembler(young=FREE_FLIGHT_YOUNG)
                gen, model, state, body = _free_flight(asm, 8)
                solver = newton.solvers.SolverVBD(model, iterations=16)
                control = model.control()
                ctrl = free_controller(model, solver, body, asm, gen, n_ritz=n_ritz)
                dt = 1.0e-3
                s0, s1 = state, model.state()
                hist = []
                for _ in range(40):
                    s0, s1 = run(solver, s0, s1, control, 1, dt, ctrl)
                    hist.append(ctrl.nodal_momentum(s0))
                info = ctrl.relinearize(s0, control)
                s0, s1 = run(solver, s0, s1, control, 100, dt, ctrl)
                post = ctrl.nodal_momentum(s0)
                P0, L0 = np.asarray(info["P_before"]), np.asarray(info["L_before"])
                self.assertLess(info["dP"] / np.linalg.norm(P0), 1.0e-5)
                self.assertLess(info["dL"] / np.linalg.norm(L0), 1.0e-5)
                self.assertLessEqual(info["KE_after"], info["KE_before"] * (1.0 + 1.0e-6))
                drift_P_pre = np.linalg.norm(hist[-1]["P"] - hist[0]["P"]) / np.linalg.norm(hist[0]["P"])
                drift_L_pre = np.linalg.norm(hist[-1]["L"] - hist[0]["L"]) / np.linalg.norm(hist[0]["L"])
                self.assertLess(np.linalg.norm(post["P"] - P0) / np.linalg.norm(P0), max(10 * drift_P_pre, 1e-4))
                self.assertLess(np.linalg.norm(post["L"] - L0) / np.linalg.norm(L0), max(10 * drift_L_pre, 1e-3))
                if n_ritz:
                    self.assertGreater(info["ritz_kept"], 0)


class TestCantileverPrestress(unittest.TestCase):
    def test_settled_cantilever_stays_in_equilibrium(self):
        """A settled sagging cantilever stays put after an update with the pre-stress g0, springs back without it."""
        for clamped in (True, False):
            for prestress in (True, False):
                with self.subTest(clamped=clamped, prestress=prestress):
                    asm = beam_assembler(young=CANTILEVER_YOUNG)
                    gen, _, model, body, fixed = build_cantilever(asm, 8, damping_ratio=0.5, clamped=clamped)
                    solver = newton.solvers.SolverVBD(model, iterations=16)
                    control = model.control()
                    dt = 2.0e-3
                    s0, s1 = run(solver, model.state(), model.state(), control, 1000, dt)
                    ctrl = free_controller(model, solver, body, asm, gen, fixed=fixed)
                    x_before = ctrl.world_field(s0)
                    sag = float(np.max(np.abs(x_before[:, 2] - asm.mesh.nodes[:, 2])))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ctrl.relinearize(s0, control)
                    if not prestress:
                        ctrl.g0[:] = 0.0
                    s0, s1 = run(solver, s0, s1, control, 300, dt, ctrl)
                    drift = float(np.max(np.linalg.norm(ctrl.world_field(s0) - x_before, axis=1))) / sag
                    if prestress:
                        self.assertLess(drift, 0.05)
                    else:
                        self.assertGreater(drift, 0.3)


class _CountingTrigger(rerom.Trigger):
    """Criterion that always fires (gating tests)."""

    def check(self, ctrl, state, steps_since, metrics):
        return True, "always", metrics


class TestTriggerGating(unittest.TestCase):
    def _setup(self, trigger, policy=None):
        asm = LinearStubAssembler(beam_assembler(young=FREE_FLIGHT_YOUNG))
        gen, _, model, state, body = build_free_beam(asm.inner, 6, shape=False)
        solver = newton.solvers.SolverVBD(model, iterations=4)
        ctrl = free_controller(model, solver, body, asm, gen, trigger=trigger, policy=policy)
        return model, solver, ctrl, state

    def _fire_steps(self, ctrl, model, solver, state, steps=30, hook=None):
        control = model.control()
        s0, s1 = state, model.state()
        fired = []
        for i in range(1, steps + 1):
            if hook is not None:
                hook(ctrl, i)
            s0, s1 = run(solver, s0, s1, control, 1, 1.0e-3, ctrl)
            if ctrl.after_step(s0, control, 1.0e-3):
                fired.append(i)
        return fired

    def test_min_interval_budget_and_cadence(self):
        trig = _CountingTrigger(options=rerom.TriggerOptions(min_interval=3, max_updates=4))
        model, solver, ctrl, state = self._setup(trig)
        self.assertEqual(self._fire_steps(ctrl, model, solver, state), [3, 6, 9, 12])
        trig = _CountingTrigger(options=rerom.TriggerOptions(min_interval=1, check_every=4))
        model, solver, ctrl, state = self._setup(trig)
        self.assertEqual(self._fire_steps(ctrl, model, solver, state, 12), [4, 8, 12])

    def test_max_interval_fires_without_criterion(self):
        trig = rerom.RotationTrigger(theta_tol=None, options=rerom.TriggerOptions(max_interval=7))
        model, solver, ctrl, state = self._setup(trig)
        fired = self._fire_steps(ctrl, model, solver, state, 21)
        self.assertEqual(fired, [7, 14, 21])
        self.assertTrue(all(st["trigger"] == "max_interval" for st in ctrl.stats))

    def test_freeze_hooks_and_requested_update(self):
        trig = _CountingTrigger(options=rerom.TriggerOptions(min_interval=2))
        policy = rerom.UpdatePolicy(freeze_in_contact=True, freeze_after_release=True)
        model, solver, ctrl, state = self._setup(trig, policy)

        def hook(c, i):
            c.set_phase(in_contact=5 <= i <= 10, released=i > 20)
            if i == 25:
                c.request_update()

        fired = self._fire_steps(ctrl, model, solver, state, 30, hook)
        self.assertEqual(fired, [2, 4, 11, 13, 15, 17, 19, 25])
        self.assertEqual(ctrl.stats[-1]["trigger"], "requested")
        trig = _CountingTrigger(options=rerom.TriggerOptions(min_interval=1))
        model, solver, ctrl, state = self._setup(trig, rerom.UpdatePolicy(post_release_min_interval=5))
        fired = self._fire_steps(ctrl, model, solver, state, 14, lambda c, i: c.set_phase(released=i > 4))
        self.assertEqual(fired, [1, 2, 3, 4, 9, 14])

    def test_rotation_trigger_fires_on_a_vibrating_beam(self):
        asm = beam_assembler(young=FREE_FLIGHT_YOUNG)
        gen, _, model, state, body = build_free_beam(asm, 6, twist=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], shape=False)
        jq = state.joint_q.numpy()
        jq[7] = 0.08 / np.max(np.abs(gen.modal_matrix[:, 0]))
        state.joint_q.assign(jq)
        solver = newton.solvers.SolverVBD(model, iterations=16)
        trig = rerom.RotationTrigger(
            theta_tol=np.radians(1.0), options=rerom.TriggerOptions(max_interval=150, min_interval=5)
        )
        ctrl = free_controller(model, solver, body, asm, gen, trigger=trig, n_ritz=1)
        self._fire_steps(ctrl, model, solver, state, 300)
        self.assertTrue(ctrl.stats)
        self.assertTrue(any(st["trigger"] == "rotation" for st in ctrl.stats))
        self.assertTrue(all(st["interval"] >= 5 for st in ctrl.stats))

    def test_energy_trigger_and_modal_derivatives(self):
        """The energy-mismatch trigger fires; modal derivatives add k(k+1)/2 vectors and Krylov fills n_ritz;
        momentum is still conserved by the switch."""
        asm = beam_assembler(young=FREE_FLIGHT_YOUNG)
        gen, model, state, body = _free_flight(asm, 12)
        solver = newton.solvers.SolverVBD(model, iterations=16)
        trig = rerom.EnergyMismatchTrigger(tol=0.05, options=rerom.TriggerOptions(min_interval=3))
        ctrl = free_controller(
            model, solver, body, asm, gen, trigger=trig, basis=rerom.BasisOptions(n_ritz=4, modal_derivatives=2)
        )
        self._fire_steps(ctrl, model, solver, state, 40)
        self.assertTrue(ctrl.stats)
        st = ctrl.stats[0]
        self.assertEqual(st["md_kept"], 3)
        self.assertEqual(st["ritz_kept"], 4)
        self.assertLess(st["dP"] / np.linalg.norm(st["P_before"]), 1e-5)
        self.assertLess(st["dL"] / np.linalg.norm(st["L_before"]), 1e-5)


class TestEigensolvers(unittest.TestCase):
    def test_restricted_arpack_matches_dense_and_counts_negative_eigenvalues(self):
        """Restricted ARPACK agrees with the dense complement solve; beyond buckling the negative eigenvalues are
        detected by the inertia count at the first shift, not crashed on."""
        asm = beam_assembler(young=FREE_FLIGHT_YOUNG)
        for label, stretch in (("rest", 1.0), ("compressed_2pct", 0.98)):
            with self.subTest(case=label):
                x = asm.mesh.nodes * np.array([stretch, 1.0, 1.0])
                _, _, K = asm.evaluate(x.reshape(-1))
                M = asm.mass_matrix
                _, c, _ = rigid_properties(M, x)
                Y = m_orthonormal_rigid(nodal_rigid_vectors(x, c), M)
                dense = rerom.RestrictedArpackBackend(dense_dof_limit=10**6).solve(K, M, 8, deflation=Y)
                arp = rerom.RestrictedArpackBackend()
                res = arp.solve(K, M, 8, deflation=Y)
                np.testing.assert_allclose(
                    res.values, dense.values, rtol=1e-6, atol=1e-6 * np.max(np.abs(dense.values))
                )
                if stretch == 1.0:
                    self.assertTrue(np.all(dense.values > 0.0))
                else:
                    self.assertGreaterEqual(int(np.count_nonzero(dense.values < 0.0)), 1)
                    self.assertGreaterEqual(arp.last_info["sigma_tries"][0][1], 1)

    def _update_with(self, eigensolver, basis=None, steps=40):
        asm = beam_assembler(young=2.0e6)
        gen, _, model, body, fixed = build_cantilever(asm, 8, damping_ratio=0.0, clamped=True)
        solver = newton.solvers.SolverVBD(model, iterations=16)
        control = model.control()
        ctrl = free_controller(
            model,
            solver,
            body,
            asm,
            gen,
            fixed=fixed,
            eigensolver=eigensolver,
            basis=basis or rerom.BasisOptions(n_ritz=2),
            trigger=rerom.RotationTrigger(theta_tol=None),
        )
        s0, _ = run(solver, model.state(), model.state(), control, steps, 2.0e-3, ctrl)
        return ctrl, ctrl.relinearize(s0, control)

    def test_subspace_against_arpack(self):
        """Warm-started restricted subspace iteration vs restricted ARPACK at a deformed state (diagnostic compare);
        a second iteration does not make it worse."""
        out = {}
        for iters in (1, 2):
            _, info = self._update_with(
                "subspace", rerom.BasisOptions(n_ritz=2, subspace_iterations=iters, eigen_compare=True)
            )
            out[iters] = info["eigen_compare"]
        self.assertLessEqual(out[2]["max_principal_angle"], out[1]["max_principal_angle"] + 1e-12)
        self.assertLess(out[2]["max_principal_angle"], 0.05)
        self.assertLess(out[2]["max_eig_rel_err"], 1e-2)

    def test_newton_backends_against_restricted_arpack(self):
        """Newton's ScipyArpackBackend and SubspaceIterationBackend (warm-started through previous_vectors) give the
        same eigenvalues as the restricted ARPACK path on a clamped body (no deflation)."""
        _, ref = self._update_with("arpack")
        sub = newton.SubspaceIterationBackend(iterations=2, margin=2)
        for backend in (newton.ScipyArpackBackend(), sub, newton.ModalEigensolver.DENSE):
            with self.subTest(backend=str(backend)):
                _ctrl, info = self._update_with(backend)
                np.testing.assert_allclose(info["eigen_raw"][:6], ref["eigen_raw"][:6], rtol=1e-6)
                np.testing.assert_allclose(info["eigenvalues"], ref["eigenvalues"], rtol=1e-5)
        self.assertIsNotNone(sub.previous_vectors)

    def test_fake_backend_accepted_end_to_end(self):
        """Any object satisfying newton.EigenSolverBackend plugs in (Warpack seam): it is called with the free-DOF
        pencil and the M-orthonormal rigid deflation, and its eigenpairs reach the model."""

        class FakeBackend:
            name = "fake"
            supports_inertia_count = False
            device = "cpu"

            def __init__(self):
                self.calls = []
                self.inner = newton.DenseBackend()

            def solve(
                self,
                stiffness,
                mass,
                count,
                *,
                deflation=None,
                sigma=None,
                tol=0.0,
                factorization=None,
                zero_threshold=0.0,
            ):
                self.calls.append((stiffness.shape, count, None if deflation is None else deflation.shape))
                res = self.inner.solve(stiffness, mass, count, deflation=deflation)
                return newton.EigenResult(res.values[:count], res.vectors[:, :count], backend=self.name)

        fake = FakeBackend()
        self.assertIsInstance(fake, newton.EigenSolverBackend)
        asm = beam_assembler(young=FREE_FLIGHT_YOUNG)
        gen, model, state, body = _free_flight(asm, 8)
        solver = newton.solvers.SolverVBD(model, iterations=8)
        control = model.control()
        ctrl = free_controller(model, solver, body, asm, gen, n_ritz=2, eigensolver=fake)
        s0, _ = run(solver, state, model.state(), control, 10, 1.0e-3, ctrl)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = ctrl.relinearize(s0, control)
        self.assertEqual(len(fake.calls), 1)
        n = 3 * asm.mesh.num_nodes
        self.assertEqual(fake.calls[0], ((n, n), 10, (n, 6)))
        self.assertEqual(info["eigen_info"]["path"], "fake")
        k_model = model.elastic_mode_stiffness.numpy()[:8]
        np.testing.assert_allclose(k_model, np.asarray(info["eigenvalues"]), rtol=1e-5)
        self.assertLess(info["dP"] / np.linalg.norm(info["P_before"]), 1e-5)
        s0, _ = run(solver, s0, model.state(), control, 10, 1.0e-3, ctrl)
        self.assertTrue(np.all(np.isfinite(ctrl.world_field(s0))))

    def test_warpack_stub_raises(self):
        asm = beam_assembler(young=FREE_FLIGHT_YOUNG)
        gen, model, state, body = _free_flight(asm, 6)
        solver = newton.solvers.SolverVBD(model, iterations=4)
        ctrl = free_controller(model, solver, body, asm, gen, eigensolver=newton.WarpackBackend(device="cpu"))
        with self.assertRaises(NotImplementedError):
            ctrl.relinearize(state)

    def test_indefinite_guard_clamps_written_stiffness(self):
        asm = beam_assembler(young=FREE_FLIGHT_YOUNG)
        gen, model, state, body = _free_flight(asm, 6)
        solver = newton.solvers.SolverVBD(model, iterations=4)
        policy = rerom.UpdatePolicy(guard_indefinite=rerom.IndefiniteGuard.CLAMP)
        ctrl = free_controller(model, solver, body, asm, gen, policy=policy)
        x = asm.mesh.nodes * np.array([0.98, 1.0, 1.0])
        override = asm.evaluate(x.reshape(-1))
        info = ctrl.relinearize(state, stiffness_override=override)
        lam = np.asarray(info["eigenvalues"])
        self.assertTrue(np.any(lam < 0.0))
        k_model = model.elastic_mode_stiffness.numpy()[:6]
        self.assertTrue(np.all(k_model > 0.0))
        self.assertEqual(ctrl.clamp_log[-1]["count"], int(np.count_nonzero(lam <= 0.0)))


if __name__ == "__main__":
    unittest.main()
