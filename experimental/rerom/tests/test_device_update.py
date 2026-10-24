# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of the device-side update (rerom.update_kernels), the shared factorization and graph capture.

CPU cases always run; CUDA cases run when a device is present.
"""

from __future__ import annotations

import _fixtures  # noqa: F401  (puts the rerom package on sys.path)

# isort: split

import unittest
import warnings

import numpy as np
import rerom
import scipy.sparse.linalg as spla  # noqa: TID253
import warp as wp
from _fixtures import HAVE_CUDA, beam_assembler, build_cantilever, build_free_beam, free_controller, run
from rerom.assembly_warp import WarpTetAssembler
from rerom.numerics import to_nodal
from rerom.update_kernels import count_host_transfers, reference_reduced_inertia

import newton
from newton._src.sim import modal_sparse
from newton._src.sim.modal import _reduced_inertia_from_pairs

FREE_FLIGHT_YOUNG = 1.0e7
CANTILEVER_YOUNG = 1.0e8

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
    "body_q",
    "body_com",
    "body_mass",
    "body_inertia",
    "body_inv_mass",
    "body_inv_inertia",
]


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


def _relative(a, b, floor):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), floor)


def _devices():
    return ["cpu", "cuda:0"] if HAVE_CUDA else ["cpu"]


def _assembler_for(asm, device):
    """Device assembler on the model device (the device path then assembles and triggers on the device)."""
    return WarpTetAssembler.from_mesh(asm.mesh, device=device)


class TestDeviceIntegrals(unittest.TestCase):
    def test_integral_kernels_match_reduced_inertia_from_pairs(self):
        """Pair / coupling integrals of the kernels match Newton's ``_reduced_inertia_from_pairs`` (float64 1e-12)."""
        asm = beam_assembler(young=FREE_FLIGHT_YOUNG)
        rng = np.random.default_rng(7)
        for device in _devices():
            for cpu_blas in (True, False) if device == "cpu" else (False,):
                with self.subTest(device=device, cpu_blas=cpu_blas):
                    gen, model, _state, body = _free_flight(asm, 6, device=device)
                    solver = newton.solvers.SolverVBD(model, iterations=2)
                    ctrl = free_controller(model, solver, body, asm, gen)
                    dev = rerom.update_kernels.DeviceUpdater(ctrl, cpu_blas=cpu_blas)
                    N, m = ctrl.N, ctrl.m
                    Phi = gen.modal_matrix[:, :m] + 0.1 * rng.standard_normal((3 * N, m))
                    x = asm.mesh.nodes + 0.01 * rng.standard_normal((N, 3))
                    dev.upload_basis(Phi, x, np.zeros(3))
                    dev.x.assign(x)
                    dev._integrals()
                    ref = reference_reduced_inertia(x, to_nodal(Phi, N), to_nodal(ctrl.M @ Phi, N))
                    pairs = dev.pairs.numpy().reshape(3 * m, 3 * m)
                    integ = dev.integrals.numpy().reshape(m, 15)
                    self.assertLess(_relative(pairs, ref["pairs"], 1e-300), 1.0e-12)
                    self.assertLess(_relative(integ[:, :3], ref["linear"], 1e-300), 1.0e-12)
                    self.assertLess(_relative(integ[:, 3:6], ref["angular"], 1e-300), 1.0e-12)
                    self.assertLess(_relative(integ[:, 6:].reshape(m, 3, 3), ref["centrifugal"], 1e-300), 1.0e-12)
                    # Newton's float32 integrals agree to float32 rounding
                    nw = _reduced_inertia_from_pairs(x, to_nodal(Phi, N), to_nodal(ctrl.M @ Phi, N))
                    quad = pairs.reshape(m, 3, m, 3).transpose(0, 2, 1, 3)
                    np.testing.assert_allclose(
                        quad.astype(np.float32), nw.mode_coupling_quadratic, rtol=2e-7, atol=1e-6 * np.abs(quad).max()
                    )


class TestDeviceMatchesHost(unittest.TestCase):
    def _compare(self, case, device):
        asm = beam_assembler(young=FREE_FLIGHT_YOUNG if case == "free" else CANTILEVER_YOUNG)
        out = {}
        for path in (rerom.UpdatePath.HOST, rerom.UpdatePath.DEVICE):
            if case == "free":
                gen, model, state, body = _free_flight(asm, 8, device=device)
                fixed = None
            else:
                gen, _b, model, body, fixed = build_cantilever(
                    asm, 8, damping_ratio=0.3, clamped=(case == "clamped"), device=device
                )
                solver = newton.solvers.SolverVBD(model, iterations=8)
                state, _s1 = run(solver, model.state(), model.state(), model.control(), 40, 2.0e-3)
            solver = newton.solvers.SolverVBD(model, iterations=4)
            a = asm if path is rerom.UpdatePath.HOST else _assembler_for(asm, device)
            # free-free + gravity: the restricted solve of the (rigid) gravity load is pure roundoff, so the
            # f_ext Ritz vector would be noise; compare with the f_int vector only
            n_ritz = 1 if case == "endpoint" else 2
            ctrl = free_controller(
                model, solver, body, a, gen, fixed=fixed, n_ritz=n_ritz, policy=rerom.UpdatePolicy(path=path)
            )
            control = model.control()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info = ctrl.relinearize(state, control)
            out[path] = (
                info,
                {k: getattr(model, k).numpy().astype(np.float64) for k in _ARRAYS if getattr(model, k) is not None},
                state.joint_q.numpy().astype(np.float64),
                state.joint_qd.numpy().astype(np.float64),
                control.joint_f.numpy().astype(np.float64),
            )
        (ih, ah, qh, qdh, fh), (idv, adv, qdv, qddv, fdv) = out[rerom.UpdatePath.HOST], out[rerom.UpdatePath.DEVICE]
        mass = float(adv["body_mass"].max())
        floors = {
            "elastic_mode_coupling_linear": np.sqrt(mass),
            "elastic_mode_coupling_angular": np.sqrt(mass),
            "elastic_mode_coupling_centrifugal": np.sqrt(mass),
            "elastic_mode_coupling_coriolis": mass,
        }
        for k in ah:
            self.assertLess(_relative(adv[k], ah[k], floors.get(k, 1.0e-12)), 2.0e-5, k)
        self.assertLess(_relative(qdv, qh, 1.0), 1.0e-6)
        self.assertLess(_relative(qddv, qdh, 1.0e-3), 1.0e-5)
        self.assertLess(_relative(fdv, fh, 1.0e-6), 1.0e-5)
        for k in ("KE_before", "KE_after", "KE_elastic_before"):
            self.assertAlmostEqual(idv[k], ih[k], delta=1.0e-7 * max(abs(ih[k]), 1e-12), msg=k)
        self.assertAlmostEqual(idv["KE_lost_projection"], ih["KE_lost_projection"], delta=1.0e-6 * ih["KE_before"])
        np.testing.assert_allclose(idv["P_after"], ih["P_after"], atol=1.0e-7 * (1.0 + np.abs(ih["P_after"]).max()))
        np.testing.assert_allclose(idv["L_after"], ih["L_after"], atol=1.0e-7 * (1.0 + np.abs(ih["L_after"]).max()))
        self.assertEqual(idv["update_path"], "device")
        self.assertTrue(idv["solver_refresh_in_place"])

    def test_free_beam(self):
        """Free flight (render points, shape vertices, rigid split): device update == host update."""
        for device in _devices():
            with self.subTest(device=device):
                self._compare("free", device)

    def test_cantilever_with_endpoint(self):
        """Free-free basis held by a fixed joint (endpoint frames, counter-rotation, derived psi)."""
        for device in _devices():
            with self.subTest(device=device):
                self._compare("endpoint", device)

    def test_clamped_cantilever(self):
        """Clamped basis (frame twist kept, constrained projection)."""
        for device in _devices():
            with self.subTest(device=device):
                self._compare("clamped", device)


class TestPerStep(unittest.TestCase):
    def test_device_rotation_trigger_matches_host(self):
        """The device rotation trigger reports the host metrics and fires at the same step."""
        asm = beam_assembler(young=FREE_FLIGHT_YOUNG)
        for device in _devices():
            with self.subTest(device=device):
                logs = {}
                for path in (rerom.UpdatePath.HOST, rerom.UpdatePath.DEVICE):
                    gen, model, s0, body = _free_flight(asm, 6, device=device)
                    s1 = model.state()
                    solver = newton.solvers.SolverVBD(model, iterations=4)
                    trig = rerom.RotationTrigger(
                        theta_tol=np.radians(2.0),
                        disp_tol=0.5,
                        options=rerom.TriggerOptions(max_updates=1, n_sample_elements=64),
                    )
                    ctrl = free_controller(
                        model,
                        solver,
                        body,
                        _assembler_for(asm, device),
                        gen,
                        trigger=trig,
                        policy=rerom.UpdatePolicy(path=path),
                    )
                    control = model.control()
                    s0, s1 = run(solver, s0, s1, control, 25, 1.0e-3, ctrl, after=True)
                    logs[path] = [
                        (e.get("trig_theta_max"), e.get("trig_disp_rel"), e.get("update", False)) for e in ctrl.step_log
                    ]
                host, devl = logs[rerom.UpdatePath.HOST], logs[rerom.UpdatePath.DEVICE]
                self.assertEqual([e[2] for e in host], [e[2] for e in devl])
                self.assertTrue(any(e[2] for e in host))
                th = np.array([e[0] for e in host if e[0] is not None])
                td = np.array([e[0] for e in devl if e[0] is not None])
                np.testing.assert_allclose(td, th, rtol=1.0e-5, atol=1.0e-9)

    def test_prestress_kernel_and_change_detection(self):
        """Modal rows of joint_f = modal_force - g0 (float32); host edits of g0 / modal_force are picked up."""
        asm = beam_assembler(young=CANTILEVER_YOUNG)
        gen, _b, model, body, fixed = build_cantilever(asm, 6, damping_ratio=0.3, clamped=True)
        solver = newton.solvers.SolverVBD(model, iterations=8)
        control = model.control()
        s0, _s1 = run(solver, model.state(), model.state(), control, 30, 2.0e-3)
        ctrl = free_controller(model, solver, body, asm, gen, fixed=fixed)
        ctrl.relinearize(s0, control)
        rows = slice(ctrl.qd_start + 6, ctrl.qd_start + 6 + ctrl.m)
        np.testing.assert_array_equal(control.joint_f.numpy()[rows], (-ctrl.g0).astype(np.float32))
        control.joint_f.zero_()
        with count_host_transfers() as n:
            ctrl.apply_prestress(control)
        self.assertEqual(n, {"d2h": 0, "h2d": 0, "h2d_alloc": 0})
        np.testing.assert_array_equal(control.joint_f.numpy()[rows], (-ctrl.g0).astype(np.float32))
        ctrl.g0[:] = 0.0
        ctrl.modal_force = np.arange(ctrl.m, dtype=np.float64)
        ctrl.apply_prestress(control)
        np.testing.assert_array_equal(control.joint_f.numpy()[rows], np.arange(ctrl.m, dtype=np.float32))


class TestHostRoundTrips(unittest.TestCase):
    def test_one_gather_one_scatter(self):
        """A device update makes one device-to-host gather (+ one diagnostics read) and one host-to-device copy."""
        asm = beam_assembler(young=FREE_FLIGHT_YOUNG)
        for device in _devices():
            for diagnostics in (True, False):
                with self.subTest(device=device, diagnostics=diagnostics):
                    gen, model, state, body = _free_flight(asm, 6, device=device)
                    solver = newton.solvers.SolverVBD(model, iterations=2)
                    ctrl = free_controller(
                        model,
                        solver,
                        body,
                        _assembler_for(asm, device),
                        gen,
                        n_ritz=2,
                        policy=rerom.UpdatePolicy(notify=False, diagnostics=diagnostics),
                    )
                    ctrl.relinearize(state, model.control())  # warm-up (kernel loads, trigger sample upload)
                    ctrl.trigger.evaluate(ctrl, state, 5)
                    with count_host_transfers() as n:
                        ctrl.relinearize(state, model.control())
                    self.assertEqual(n["d2h"], 2 if diagnostics else 1)
                    self.assertEqual(n["h2d"], 1)
                    self.assertEqual(n["h2d_alloc"], 0)
                    with count_host_transfers() as n:
                        ctrl.poll(state, 1.0e-3)  # velocity history + trigger check
                    self.assertEqual(n["d2h"], 1)  # the 3-value trigger flag
                    self.assertEqual(n["h2d"] + n["h2d_alloc"], 0)


class TestSharedFactorization(unittest.TestCase):
    def _relinearize(self, eigensolver, shared):
        asm = beam_assembler(young=FREE_FLIGHT_YOUNG)
        gen, model, state, body = _free_flight(asm, 8)
        solver = newton.solvers.SolverVBD(model, iterations=2)
        ctrl = free_controller(
            model,
            solver,
            body,
            asm,
            gen,
            basis=rerom.BasisOptions(n_ritz=2, shared_factorization=shared),
            eigensolver=eigensolver,
        )
        count = {"n": 0}
        orig = spla.splu

        def splu(*a, **k):
            count["n"] += 1
            return orig(*a, **k)

        spla.splu = splu
        try:
            info = ctrl.relinearize(state, model.control())
        finally:
            spla.splu = orig
        return info, count["n"]

    def test_one_factorization_per_update(self):
        for eigensolver in ("arpack", "subspace"):
            with self.subTest(eigensolver=eigensolver):
                info_s, n_s = self._relinearize(eigensolver, True)
                info_l, n_l = self._relinearize(eigensolver, False)
                self.assertEqual(n_s, 1)
                self.assertEqual(n_l, 2 if eigensolver == "arpack" else 1)
                if eigensolver == "arpack":
                    self.assertTrue(info_s["eigen_info"]["shared_factorization"])
                    self.assertFalse(info_l["eigen_info"]["shared_factorization"])
                np.testing.assert_allclose(info_s["eigenvalues"], info_l["eigenvalues"], rtol=1.0e-8)

    def test_arpack_lowers_the_shift_below_negative_eigenvalues(self):
        """With eigenvalues below the shared shift the Sturm check rejects it and a lower shift is factored."""
        asm = beam_assembler(young=FREE_FLIGHT_YOUNG)
        K = asm.linear_stiffness().tocsc()
        M = asm.mass_matrix.tocsc()
        root = np.asarray(asm.mesh.node_sets["root"])
        clamped = np.zeros(K.shape[0], dtype=bool)
        clamped[(3 * root[:, None] + np.arange(3)).reshape(-1)] = True
        free = np.nonzero(~clamped)[0]
        lam = spla.eigsh(K[free][:, free], k=3, M=M[free][:, free], sigma=0.0, which="LM")[0]
        Kf = (K - 1.5 * float(np.sort(lam)[1]) * M)[free][:, free]  # two eigenvalues become negative
        Mf = M[free][:, free]
        shared = rerom.eigen.shared_restricted_solver(Kf, Mf, None)
        backend = rerom.RestrictedArpackBackend()
        res = backend.solve(Kf, Mf, 6, factorization=shared)
        self.assertFalse(backend.last_info["shared_factorization"])
        ref = rerom.RestrictedArpackBackend().solve(Kf, Mf, 6)
        np.testing.assert_allclose(res.values, ref.values, rtol=1.0e-8)
        self.assertLess(float(res.values[0]), 0.0)
        self.assertIsInstance(shared.factorization, modal_sparse.ShiftedFactorization)


@unittest.skipUnless(HAVE_CUDA, "CUDA graph capture needs a GPU")
class TestGraphCapture(unittest.TestCase):
    def test_captured_step_graph_survives_an_update(self):
        """Capture prestress + two steps, update between replays: the graph matches an uncaptured run, sees the
        update, and the solver's mass-solve arrays keep their allocation."""
        device = "cuda:0"
        asm = beam_assembler(young=CANTILEVER_YOUNG)
        dt = 2.0e-3
        results = {}
        for captured, update in ((False, True), (True, True), (True, False)):
            gen, _b, model, body, _fixed = build_cantilever(asm, 8, damping_ratio=0.3, clamped=False, device=device)
            solver = newton.solvers.SolverVBD(model, iterations=8)
            ctrl = free_controller(model, solver, body, _assembler_for(asm, device), gen, n_ritz=2)
            s0, s1 = model.state(), model.state()
            control = model.control()

            def steps(s0=s0, s1=s1, control=control, solver=solver, ctrl=ctrl):
                ctrl.apply_prestress(control)
                s0.clear_forces()
                solver.step(s0, s1, control, None, dt)
                ctrl.apply_prestress(control)
                s1.clear_forces()
                solver.step(s1, s0, control, None, dt)

            steps()
            if captured:
                with wp.ScopedCapture(device=device) as cap:
                    steps()

                def go(graph=cap.graph):
                    wp.capture_launch(graph)
            else:
                go = steps
            for _ in range(30):
                go()
            ptrs = [
                solver.elastic_implicit_mass_coupling.ptr,
                solver.elastic_reduced_mobility.frame_inv_schur.ptr,
                solver.elastic_reduced_mobility.frame_modal_mass.ptr,
                solver.elastic_reduced_mobility.modal_inv_mass.ptr,
            ]
            if update:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    info = ctrl.relinearize(s0, control)
                self.assertTrue(info["solver_refresh_in_place"])
                self.assertEqual(
                    ptrs,
                    [
                        solver.elastic_implicit_mass_coupling.ptr,
                        solver.elastic_reduced_mobility.frame_inv_schur.ptr,
                        solver.elastic_reduced_mobility.frame_modal_mass.ptr,
                        solver.elastic_reduced_mobility.modal_inv_mass.ptr,
                    ],
                )
                _flags, fresh = solver._build_elastic_mass_solve_data(model)
                np.testing.assert_array_equal(
                    solver.elastic_reduced_mobility.frame_inv_schur.numpy(), fresh.frame_inv_schur.numpy()
                )
            for _ in range(30):
                go()
            results[(captured, update)] = s0.joint_q.numpy().copy()
        np.testing.assert_array_equal(results[(True, True)], results[(False, True)])
        self.assertGreater(float(np.max(np.abs(results[(True, True)] - results[(True, False)]))), 1.0e-4)


if __name__ == "__main__":
    unittest.main()
