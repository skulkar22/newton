# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of the TangentAssembler implementations: numpy reference and the Warp assembler inside the controller."""

from __future__ import annotations

import _fixtures  # noqa: F401  (puts the rerom package on sys.path)

# isort: split

import unittest

import numpy as np
import rerom
from _fixtures import HAVE_CUDA, beam_mesh, build_free_beam, free_controller, run
from rerom.assembly import HostTangentAssembler, Material
from rerom.assembly_warp import WarpTetAssembler

import newton


def _rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _small_mesh():
    return rerom.make_beam_mesh(0.3, 0.1, 0.08, 3, 2, 2, density=1000.0, young=1.0e6, poisson=0.3)


def _deformed(mesh, rng, amp=0.08):
    X = mesh.nodes
    x = X.copy()
    x[:, 2] += 0.6 * X[:, 0] ** 2
    x[:, 0] *= 1.05
    x = x @ _rot([0.3, 1.0, -0.2], 0.9).T + np.array([0.1, -0.2, 0.3])
    return (x + amp * 0.03 * rng.standard_normal(x.shape)).reshape(-1)


class TestNumpyAssembler(unittest.TestCase):
    def test_force_and_tangent_are_derivatives(self):
        for material in Material:
            with self.subTest(material=material.value):
                rng = np.random.default_rng(1)
                mesh = _small_mesh()
                asm = rerom.NumpyTetAssembler(mesh, material.value)
                self.assertIsInstance(asm, rerom.TangentAssembler)
                x = _deformed(mesh, rng)
                _, f, K = asm.evaluate(x)
                v = rng.standard_normal(x.size)
                h = 1e-6
                dW = (
                    asm.evaluate(x + h * v, need_tangent=False)[0] - asm.evaluate(x - h * v, need_tangent=False)[0]
                ) / (2 * h)
                self.assertLess(abs(dW - f @ v) / abs(f @ v), 1e-6)
                df = (
                    asm.evaluate(x + h * v, need_tangent=False)[1] - asm.evaluate(x - h * v, need_tangent=False)[1]
                ) / (2 * h)
                self.assertLess(np.linalg.norm(df - K @ v) / np.linalg.norm(K @ v), 1e-6)

    def test_rigid_motion_zero_energy_and_force(self):
        mesh = _small_mesh()
        asm = rerom.NumpyTetAssembler(mesh, "stvk")
        x = (mesh.nodes @ _rot([1, 2, 3], 1.3).T + np.array([0.3, -1.0, 2.0])).reshape(-1)
        W, f, _ = asm.evaluate(x)
        self.assertLess(abs(W), 1e-12)
        self.assertLess(np.max(np.abs(f)), 1e-9)


class TestWarpAssembler(unittest.TestCase):
    def _check_against_numpy(self, device):
        mesh = beam_mesh(young=1.0e7)
        ref = rerom.NumpyTetAssembler(mesh, "stvk")
        wasm = HostTangentAssembler(WarpTetAssembler.from_mesh(mesh, "stvk", device=device))
        self.assertIsInstance(wasm, rerom.TangentAssembler)
        self.assertIsInstance(wasm.inner, rerom.TangentAssembler)  # BSR tangents are accepted too (host_sparse)
        rng = np.random.default_rng(4)
        x = (mesh.nodes + 1e-3 * rng.standard_normal(mesh.nodes.shape)).reshape(-1)
        W0, f0, K0 = ref.evaluate(x)
        W1, f1, K1 = wasm.evaluate(x)
        self.assertLess(abs(W1 - W0), 1e-10 * abs(W0))
        self.assertLess(np.max(np.abs(f1 - f0)), 1e-10 * np.max(np.abs(f0)))
        self.assertLess(abs(K1 - K0).max(), 1e-10 * abs(K0).max())
        return mesh, ref, wasm

    def test_warp_cpu_matches_numpy_and_drives_the_controller(self):
        """WarpTetAssembler (CPU, host CSC tangent) equals the numpy reference and gives the same update."""
        _mesh, ref, wasm = self._check_against_numpy("cpu")
        infos = []
        for asm in (ref, wasm, wasm.inner):  # numpy, Warp with host CSC, Warp with device BSR tangents
            gen, _, model, state, body = build_free_beam(ref, 6, mode_qd=[0.3, 0.0, 0.1, 0.0, 0.0, 0.0], shape=False)
            jq = state.joint_q.numpy()
            jq[7] = 0.05 / np.max(np.abs(gen.modal_matrix[:, 0]))
            state.joint_q.assign(jq)
            solver = newton.solvers.SolverVBD(model, iterations=8)
            control = model.control()
            trig = rerom.EnergyMismatchTrigger(tol=1e9)  # builds its sampled sub-assembler from the same class
            ctrl = free_controller(model, solver, body, asm, gen, n_ritz=2, trigger=trig)
            s0, _ = run(solver, state, model.state(), control, 5, 1e-3, ctrl, after=True)
            infos.append(ctrl.relinearize(s0, control))
        for other in infos[1:]:
            np.testing.assert_allclose(other["eigenvalues"], infos[0]["eigenvalues"], rtol=1e-8)
        self.assertAlmostEqual(
            infos[1]["W_nonlinear"], infos[0]["W_nonlinear"], delta=1e-10 * abs(infos[0]["W_nonlinear"])
        )

    @unittest.skipUnless(HAVE_CUDA, "CUDA device not available")
    def test_warp_cuda_matches_numpy(self):
        self._check_against_numpy("cuda:0")

    def test_default_assembler(self):
        mesh = beam_mesh()
        asm = rerom.default_assembler(mesh, device="cpu")
        self.assertIsInstance(asm, HostTangentAssembler)
        self.assertIsInstance(asm.inner, WarpTetAssembler)
        asm = rerom.default_assembler(mesh)
        if HAVE_CUDA:
            self.assertEqual(str(asm.inner.device), "cuda:0")
        else:
            self.assertIsInstance(asm, rerom.NumpyTetAssembler)


if __name__ == "__main__":
    unittest.main()
