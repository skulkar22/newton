# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of the full FE references (Newmark FullFEReference, prescribed rigid base motion) and the strain library."""

from __future__ import annotations

import _fixtures  # noqa: F401  (puts the rerom package on sys.path)

# isort: split

import math
import unittest

import numpy as np
import rerom
from _fixtures import beam_mesh
from rerom import strain as S
from rerom.fe_reference import best_fit_rotation


def _rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def small_mesh(young=1.0e8):
    m = rerom.make_beam_mesh(0.1, 0.02, 0.02, 10, 2, 2, density=1100.0, young=young, poisson=0.35)
    m.node_sets["clamp"] = np.asarray(m.node_sets["root"], np.int64)
    return m


class TestBaseMotion(unittest.TestCase):
    def test_derivatives_and_rest_ends(self):
        mot = rerom.BaseMotion(rot_amp=math.radians(60.0), freq_hz=2.0, pivot=(0.01, -0.02, 0.03))
        X = np.array([[0.5, 0.0, 0.0], [0.1, 0.0125, -0.0125], [0.0, 0.0, 0.0]])
        h = 1e-5
        err_v = err_a = 0.0
        for t in np.linspace(0.05, 3.4, 37):
            _, v, a = mot.nodal(X, t)
            xp, vp, _ = mot.nodal(X, t + h)
            xm, vm, _ = mot.nodal(X, t - h)
            err_v = max(err_v, float(np.abs((xp - xm) / (2 * h) - v).max()))
            err_a = max(err_a, float(np.abs((vp - vm) / (2 * h) - a).max()))
        self.assertLess(err_v, 1e-6)
        self.assertLess(err_a, 1e-4)
        _, v0, a0 = mot.nodal(X, 0.0)
        _, v1, a1 = mot.nodal(X, 3.2)
        self.assertEqual(np.abs(np.r_[v0, v1, a0, a1]).max(), 0.0)

    def test_rigid_prescribed_all_nodes(self):
        mesh = small_mesh()
        asm = rerom.NumpyTetAssembler(mesh, "stvk")
        mot = rerom.BaseMotion(rot_amp=math.radians(60.0), env_times=(0.0, 0.1, 0.25, 0.35))
        ref = rerom.PrescribedMotionReference(asm, mot, clamped_nodes=np.arange(mesh.nodes.shape[0]))
        r = ref.run(dt=1e-3, steps=200)
        xe = np.stack([mot.nodal(mesh.nodes, t)[0] for t in r["times"]])
        self.assertLess(r["strain"].max(), 1e-20)
        self.assertLess(float(np.abs(r["x_hist"] - xe).max()), 1e-14)


class TestFullFEReference(unittest.TestCase):
    def test_free_flight_conserves_momentum(self):
        mesh = beam_mesh(nx=6, young=1.0e6)
        asm = rerom.NumpyTetAssembler(mesh, "stvk")
        ref = rerom.FullFEReference(asm, gravity=(0.0, 0.0, 0.0))
        X = mesh.nodes
        v0 = np.cross(np.array([0.0, 0.5, 2.0]), X - X.mean(axis=0)) + np.array([0.1, 0.0, 0.2])
        res = ref.run(x0=X.reshape(-1), v0=v0.reshape(-1), dt=2e-3, steps=50, record_velocity=True)
        M = asm.mass_matrix
        P = [(M @ v.reshape(-1)).reshape(-1, 3).sum(axis=0) for v in res["v_hist"]]
        self.assertLess(np.max(np.abs(P[-1] - P[0])), 1e-9 * np.linalg.norm(P[0]))

    def test_best_fit_rotation(self):
        X = np.random.default_rng(0).standard_normal((20, 3))
        R = _rot([0.2, 1.0, 0.4], 0.7)
        R_fit, ang = best_fit_rotation(X, X @ R.T + 1.0)
        np.testing.assert_allclose(R_fit, R, atol=1e-12)
        self.assertAlmostEqual(ang, 0.7, places=12)


class TestStrain(unittest.TestCase):
    def setUp(self):
        m = rerom.make_beam_mesh(length=0.2, width=0.04, height=0.03, nx=6, ny=2, nz=2)
        self.X0, self.tets = np.asarray(m.nodes, float), np.asarray(m.tets, np.int64)

    def test_rigid_motion_zero_green(self):
        kin = S.TetKinematics(self.X0, self.tets)
        Xs = np.stack([self.X0 @ _rot([1, 2, 3], a).T + np.array([0.3, -1.0, 2.0]) * a for a in (0.0, 0.7, 1.4)])
        series = S.strain_series(kin, Xs, np.arange(3.0))
        self.assertLess(np.max(np.abs(series.tensor)), 1e-12)
        self.assertLess(np.max(np.abs(S.strain_series(kin, Xs, np.arange(3.0), "biot").tensor)), 1e-12)

    def test_uniform_stretch_and_metrics(self):
        kin = S.TetKinematics(self.X0, self.tets)
        F = np.diag([1.2, 0.9, 1.05])
        F[0, 2] = 0.1
        X = self.X0 @ (_rot([0.2, -1, 0.5], 1.1) @ F).T + 0.5
        E = S.strain_series(kin, X[None], [0.0]).tensor[0]
        np.testing.assert_allclose(E, (0.5 * (F.T @ F - np.eye(3)))[None].repeat(E.shape[0], 0), atol=1e-12)
        a = S.strain_series(kin, np.stack([self.X0, X]), [0.0, 1.0])
        c = S.compare_strain(a, a)
        self.assertEqual(c.metrics["rms_err"], 0.0)
        self.assertEqual(c.metrics["max_err"], 0.0)

    def test_matches_assembler_green_strain(self):
        asm = rerom.NumpyTetAssembler(rerom.TetMesh(self.X0, self.tets, 1000.0, 1e6, 0.3), "stvk")
        X = self.X0 + 1e-3 * np.random.default_rng(1).standard_normal(self.X0.shape)
        E = S.strain_series(S.TetKinematics(self.X0, self.tets), X[None], [0.0]).tensor[0]
        np.testing.assert_allclose(S.frobenius(E), asm.element_green_strain(X.ravel()), rtol=1e-10, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
