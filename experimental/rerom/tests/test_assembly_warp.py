# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for rerom.assembly_warp.WarpTetAssembler (CPU and, when available, CUDA).

Run from the Newton checkout with its environment, e.g.::

    uv run python -m unittest discover -s experimental/rerom/tests -p test_assembly_warp.py -v

The identity tests against the numpy reference use ``rerom.assembly.NumpyTetAssembler`` when the
unified package provides it, otherwise ``adaptive_rom.fe_nonlinear.NonlinearAssembler`` from the
directory in ``ADAPTIVE_ROM_ROOT`` (default: the exp-re-rom workspace four levels up). They are
skipped, with the reason, when neither is importable; every other test is self-contained.
"""

from __future__ import annotations

import importlib
import os
import sys
import unittest

import numpy as np
import warp as wp

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from rerom.assembly_warp import WarpTetAssembler, WarpTetMesh  # noqa: E402

wp.config.quiet = True

MATERIALS = ("stvk", "neo_hookean")
SCATTERS = ("gather", "atomic")


def _load_reference():
    """Return the numpy reference assembler class, or (None, reason)."""
    try:
        return importlib.import_module("rerom.assembly").NumpyTetAssembler, ""
    except (ImportError, AttributeError):
        pass
    root = os.environ.get("ADAPTIVE_ROM_ROOT", os.path.abspath(os.path.join(_PKG_ROOT, "..", "..", "..")))
    if root not in sys.path:
        sys.path.append(root)
    try:
        return importlib.import_module("adaptive_rom.fe_nonlinear").NonlinearAssembler, ""
    except ImportError as exc:
        return None, f"numpy reference assembler not importable ({exc}); set ADAPTIVE_ROM_ROOT"


NumpyReference, _REFERENCE_REASON = _load_reference()

_KUHN = np.array(
    [[0, 4, 6, 7], [0, 4, 5, 7], [0, 2, 6, 7], [0, 2, 3, 7], [0, 1, 5, 7], [0, 1, 3, 7]], dtype=np.int64
)  # corner id = 4 bx + 2 by + bz; six tets sharing the 000-111 diagonal


def _box_mesh(nx, ny, nz, size=(0.3, 0.1, 0.08), jitter=0.0, seed=0, young=1.0e6, poisson=0.3, density=1000.0):
    """Structured box split into Kuhn tets (orientation fixed); optional interior node jitter."""
    xs, ys, zs = (np.linspace(0.0, s, n + 1) for s, n in zip(size, (nx, ny, nz), strict=True))
    IX, IY, IZ = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    nodes = np.stack([xs[IX.ravel()], ys[IY.ravel()], zs[IZ.ravel()]], axis=1)
    if jitter:
        h = min(s / n for s, n in zip(size, (nx, ny, nz), strict=True))
        nodes = nodes + jitter * h * np.random.default_rng(seed).uniform(-1, 1, nodes.shape)
    gid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [gid(i + bx, j + by, k + bz) for bx in (0, 1) for by in (0, 1) for bz in (0, 1)]
                tets.extend([[c[q] for q in t] for t in _KUHN])
    tets = np.asarray(tets, dtype=np.int64)
    X = nodes[tets]
    neg = np.linalg.det(np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=-1)) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return WarpTetMesh(nodes, tets, density, young, poisson)


def _rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _deformed_state(mesh, rng, amp=0.08):
    """Large rotation + bending + stretch + random perturbation (det F > 0)."""
    X = mesh.nodes
    x = X.copy()
    x[:, 2] += 0.6 * X[:, 0] ** 2
    x[:, 0] *= 1.05
    x = x @ _rot([0.3, 1.0, -0.2], 0.9).T + np.array([0.1, -0.2, 0.3])
    x += amp * 0.03 * rng.standard_normal(x.shape)
    return x.reshape(-1)


def _devices():
    devs = ["cpu"]
    if wp.is_cuda_available():
        devs.append("cuda:0")
    return devs


class _AssemblerTests:
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        cls.mesh = _box_mesh(3, 2, 2, jitter=0.15)

    def make(self, material="stvk", scatter="auto", mesh=None):
        return WarpTetAssembler.from_mesh(mesh or self.mesh, material, self.device, scatter=scatter)

    # -- derivative identities -----------------------------------------------------------
    def test_force_is_energy_gradient(self):
        for material in MATERIALS:
            with self.subTest(material=material):
                rng = np.random.default_rng(1)
                asm = self.make(material)
                x = _deformed_state(self.mesh, rng)
                _, f, _ = asm.evaluate(x, need_tangent=False)
                errs = []
                for _ in range(5):
                    d = rng.standard_normal(x.size)
                    eps = 1e-6
                    Wp = asm.evaluate(x + eps * d, need_tangent=False)[0]
                    Wm = asm.evaluate(x - eps * d, need_tangent=False)[0]
                    fd = (Wp - Wm) / (2 * eps)
                    errs.append(abs(fd - f @ d) / abs(f @ d))
                print(f"[{self.device} {material}] max rel err dW/dx: {max(errs):.2e}")
                self.assertLess(max(errs), 1e-5)  # central FD, eps = 1e-6: truncation/roundoff floor ~1e-7

    def test_tangent_is_force_jacobian(self):
        for material in MATERIALS:
            for scatter in SCATTERS:
                with self.subTest(material=material, scatter=scatter):
                    rng = np.random.default_rng(2)
                    asm = self.make(material, scatter)
                    x = _deformed_state(self.mesh, rng)
                    _, _, K = asm.evaluate(x, tangent_format="csr")
                    self.assertLessEqual(abs(K - K.T).max(), 1e-12 * abs(K).max())
                    errs = []
                    for _ in range(5):
                        d = rng.standard_normal(x.size)
                        eps = 1e-6
                        fp = asm.evaluate(x + eps * d, need_tangent=False)[1]
                        fm = asm.evaluate(x - eps * d, need_tangent=False)[1]
                        fd = (fp - fm) / (2 * eps)
                        errs.append(np.linalg.norm(fd - K @ d) / np.linalg.norm(K @ d))
                    print(f"[{self.device} {material} {scatter}] max rel err K d vs FD: {max(errs):.2e}")
                    self.assertLess(max(errs), 1e-6)

    def test_rigid_motion_zero_energy_and_force(self):
        R = _rot([1.0, -2.0, 0.5], 2.3)
        x = (self.mesh.nodes @ R.T + np.array([3.0, -1.0, 2.0])).reshape(-1)
        fscale = self.mesh.young * np.ptp(self.mesh.nodes, axis=0).max() ** 2
        for material in MATERIALS:
            with self.subTest(material=material):
                W, f, K = self.make(material).evaluate(x, tangent_format="csr")
                self.assertLess(abs(W), 1e-11 * fscale)
                self.assertLess(np.abs(f).max(), 1e-9 * fscale)
                xc = x.reshape(-1, 3)
                modes = [np.tile(e, self.mesh.num_nodes) for e in np.eye(3)]
                modes += [np.cross(w, xc).reshape(-1) for w in np.eye(3)]
                for m in modes:
                    self.assertLess(np.linalg.norm(K @ m), 1e-8 * abs(K).max() * np.linalg.norm(m))

    def test_linear_stiffness_is_linear_elasticity(self):
        """K_T at rest equals an independent B^T D B assembly (Voigt, engineering shear)."""
        from scipy import sparse

        for material in MATERIALS:
            with self.subTest(material=material):
                asm = self.make(material)
                lam, mu = asm.lam, asm.mu
                D = np.zeros((6, 6))
                D[:3, :3] = lam
                D[:3, :3] += 2 * mu * np.eye(3)
                D[3:, 3:] = mu * np.eye(3)
                rows, cols, vals = [], [], []
                for e, tet in enumerate(self.mesh.tets):
                    B = np.zeros((6, 12))
                    for a in range(4):
                        gx, gy, gz = asm.G[e, a]
                        B[:, 3 * a : 3 * a + 3] = [
                            [gx, 0, 0],
                            [0, gy, 0],
                            [0, 0, gz],
                            [gy, gx, 0],
                            [0, gz, gy],
                            [gz, 0, gx],
                        ]
                    dofs = (3 * tet[:, None] + np.arange(3)).ravel()
                    rows.append(np.repeat(dofs, 12))
                    cols.append(np.tile(dofs, 12))
                    vals.append((asm.vol[e] * B.T @ D @ B).ravel())
                Klin = sparse.coo_matrix(
                    (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(asm.n_dof,) * 2
                ).tocsr()
                K0 = asm.linear_stiffness("csr")
                self.assertLess(abs(K0 - Klin).max() / abs(Klin).max(), 1e-12)
                self.assertIs(asm.linear_stiffness(), asm.linear_stiffness())

    # -- identities against the numpy reference ------------------------------------------
    @unittest.skipIf(NumpyReference is None, _REFERENCE_REASON)
    def test_matches_numpy_reference(self):
        rng = np.random.default_rng(3)
        x = _deformed_state(self.mesh, rng)
        for material in MATERIALS:
            ref = NumpyReference(self.mesh, material)
            W0, f0, K0 = ref.evaluate(x)
            for scatter in SCATTERS:
                with self.subTest(material=material, scatter=scatter):
                    W, f, K = self.make(material, scatter).evaluate(x, tangent_format="csr")
                    eW = abs(W - W0) / abs(W0)
                    ef = np.abs(f - f0).max() / np.abs(f0).max()
                    eK = abs(K - K0).max() / abs(K0).max()
                    print(f"[{self.device} {material} {scatter}] vs numpy: W {eW:.1e} f {ef:.1e} K {eK:.1e}")
                    self.assertLess(eW, 1e-10)
                    self.assertLess(ef, 1e-10)
                    self.assertLess(eK, 1e-10)
                    self.assertTrue(np.array_equal(K.indptr, K0.indptr) and np.array_equal(K.indices, K0.indices))

    @unittest.skipIf(NumpyReference is None, _REFERENCE_REASON)
    def test_mass_matches_numpy_reference(self):
        ref = NumpyReference(self.mesh)
        asm = self.make()
        M0, M = ref.mass_matrix, asm.mass_matrix
        self.assertLess(abs(M - M0).max() / abs(M0).max(), 1e-14)
        np.testing.assert_allclose(asm.lumped_mass, ref.lumped_mass, rtol=1e-14)
        self.assertLess(abs(asm.to_scipy_csr(asm.mass_matrix_bsr) - M0).max() / abs(M0).max(), 1e-14)

    @unittest.skipIf(NumpyReference is None, _REFERENCE_REASON)
    def test_element_rotations_match_numpy_reference(self):
        ref = NumpyReference(self.mesh)
        asm = self.make()
        rng = np.random.default_rng(4)
        x = _deformed_state(self.mesh, rng)
        x_ref = (self.mesh.nodes @ _rot([0.0, 0.0, 1.0], 0.4).T).reshape(-1)
        np.testing.assert_allclose(asm.element_rotations(x), ref.element_rotations(x), atol=1e-9)
        np.testing.assert_allclose(
            asm.element_rotations(x, reference_flat=x_ref), ref.element_rotations(x, reference_flat=x_ref), atol=1e-9
        )
        sel = np.array([0, 5, 7, 11])
        np.testing.assert_allclose(asm.element_rotations(x, sel), ref.element_rotations(x, sel), atol=1e-9)

    @unittest.skipIf(NumpyReference is None, _REFERENCE_REASON)
    def test_element_terms_match_numpy_reference(self):
        rng = np.random.default_rng(5)
        x = _deformed_state(self.mesh, rng).reshape(-1, 3)
        sel = np.array([2, 9, 17, 30, 31, 70])
        for material in MATERIALS:
            with self.subTest(material=material):
                We0, fe0, Ke0 = NumpyReference(self.mesh, material).element_terms(x[self.mesh.tets[sel]], sel)
                We, fe, Ke = self.make(material).element_terms(x[self.mesh.tets[sel]], sel)
                np.testing.assert_allclose(We, We0, rtol=1e-12)
                self.assertLess(np.abs(fe - fe0).max() / np.abs(fe0).max(), 1e-12)
                self.assertLess(np.abs(Ke - Ke0).max() / np.abs(Ke0).max(), 1e-12)

    # -- self-contained checks -----------------------------------------------------------
    def test_mass(self):
        asm = self.make()
        M = asm.mass_matrix
        m_tot = self.mesh.total_mass()
        one_x = np.tile([1.0, 0.0, 0.0], self.mesh.num_nodes)
        self.assertAlmostEqual(one_x @ M @ one_x / m_tot, 1.0, places=12)
        self.assertAlmostEqual(asm.lumped_mass.sum() / (3 * m_tot), 1.0, places=12)
        np.testing.assert_allclose(asm.lumped_mass, np.asarray(M.sum(axis=1)).ravel(), rtol=1e-14)
        # element mass rho V / 20 (1 + delta_ab): lumped share rho V / 4 per tet node
        lumped = np.zeros(self.mesh.num_nodes)
        np.add.at(lumped, self.mesh.tets.ravel(), np.repeat(self.mesh.density * asm.vol / 4.0, 4))
        np.testing.assert_allclose(asm.lumped_mass.reshape(-1, 3), np.repeat(lumped[:, None], 3, axis=1), rtol=1e-13)
        self.assertGreater(np.linalg.eigvalsh(M.toarray()).min(), 0.0)

    def test_element_rotations_analytic(self):
        asm = self.make()
        R = _rot([0.2, 1.0, 0.4], 0.7)
        x = (self.mesh.nodes @ R.T).reshape(-1)
        np.testing.assert_allclose(asm.element_rotations(x), 0.7, atol=1e-12)
        xs = (self.mesh.nodes * [1.2, 0.9, 1.1]).reshape(-1)
        np.testing.assert_allclose(asm.element_rotations(xs), 0.0, atol=1e-12)
        np.testing.assert_allclose(asm.element_rotations(x, reference_flat=x), 0.0, atol=1e-12)
        # stretch after rotation, measured from the rotated reference: zero; from rest: 0.7
        xsr = ((self.mesh.nodes * [1.2, 0.9, 1.1]) @ R.T).reshape(-1)
        np.testing.assert_allclose(asm.element_rotations(xsr), 0.7, atol=1e-12)
        R2 = _rot([1.0, 0.0, 0.0], 3.0)
        np.testing.assert_allclose(asm.element_rotations((self.mesh.nodes @ R2.T).reshape(-1)), 3.0, atol=1e-12)
        self.assertEqual(asm.element_rotations(x, np.array([0, 5, 7])).shape, (3,))

    def test_element_rotations_reflection(self):
        """det F < 0: the rotation of the reflected-polar convention (flip of the smallest axis)."""
        asm = self.make()
        x = (self.mesh.nodes * [1.0, 1.0, -0.5]).reshape(-1)  # reflection through z: R = I
        np.testing.assert_allclose(asm.element_rotations(x), 0.0, atol=1e-10)

    def test_element_subset_matches_full(self):
        rng = np.random.default_rng(6)
        for material in MATERIALS:
            with self.subTest(material=material):
                asm = self.make(material)
                x = _deformed_state(self.mesh, rng)
                W, f, K = asm.evaluate(x, tangent_format="csr")
                xe = x.reshape(-1, 3)[self.mesh.tets]
                We, fe, Ke = asm.element_terms(xe)
                self.assertAlmostEqual(We.sum() / W, 1.0, places=13)
                f_sum = np.zeros_like(x)
                np.add.at(f_sum, (3 * self.mesh.tets[:, :, None] + np.arange(3)).ravel(), fe.ravel())
                self.assertLess(np.abs(f_sum - f).max() / np.abs(f).max(), 1e-13)
                dofs = (3 * self.mesh.tets[:, :, None] + np.arange(3)).reshape(-1, 12)
                Kd = np.zeros((asm.n_dof, asm.n_dof))
                np.add.at(Kd, (dofs[:, :, None], dofs[:, None, :]), Ke.reshape(-1, 12, 12))
                self.assertLess(np.abs(Kd - K.toarray()).max() / abs(K).max(), 1e-13)
                sel = np.array([4, 1, 40])
                We_s, fe_s, Ke_s = asm.element_terms(xe[sel], sel)
                np.testing.assert_array_equal(We_s, We[sel])
                np.testing.assert_array_equal(fe_s, fe[sel])
                np.testing.assert_array_equal(Ke_s, Ke[sel])
                self.assertIsNone(asm.element_terms(xe[sel], sel, need_tangent=False)[2])

    def test_scatter_modes_agree_and_gather_is_deterministic(self):
        rng = np.random.default_rng(7)
        x = _deformed_state(self.mesh, rng)
        a_g, a_a = self.make("neo_hookean", "gather"), self.make("neo_hookean", "atomic")
        _, f1, K1 = a_g.evaluate(x)
        _, f2, K2 = a_g.evaluate(x)
        _, f3, K3 = a_a.evaluate(x)
        v1, v2, v3 = (a_g.block_values_numpy(K) for K in (K1, K2, K3))
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(v1, v2)
        self.assertLess(np.abs(f3 - f1).max() / np.abs(f1).max(), 1e-14)
        self.assertLess(np.abs(v3 - v1).max() / np.abs(v1).max(), 1e-14)

    def test_formats_and_device_inputs(self):
        rng = np.random.default_rng(8)
        asm = self.make("stvk")
        x = _deformed_state(self.mesh, rng)
        W, f, Kb = asm.evaluate(x)
        Kr = asm.to_scipy_csr(Kb)
        Kc = asm.to_scipy_csc(Kb)
        self.assertEqual(Kc.format, "csc")
        self.assertEqual(abs(Kr - Kc).max(), 0.0)
        # BSR product agrees with the CSR product
        y = wp.zeros(asm.n_nodes, dtype=wp.vec3d, device=self.device)
        d = rng.standard_normal(asm.n_dof)
        import warp.sparse as wps  # noqa: PLC0415

        wps.bsr_mv(Kb, wp.array(d.reshape(-1, 3), dtype=wp.vec3d, device=self.device), y)
        self.assertLess(np.abs(y.numpy().ravel() - Kr @ d).max() / np.abs(Kr @ d).max(), 1e-13)
        # returned copies are not overwritten by later calls
        asm.evaluate(asm.X_flat)
        np.testing.assert_array_equal(asm.to_scipy_csr(Kb).data, Kr.data)
        # device-resident float64 input and device output
        x_wp = wp.array(x.reshape(-1, 3), dtype=wp.vec3d, device=self.device)
        W2, f2, _ = asm.evaluate(x_wp, need_tangent=False, device_output=True)
        self.assertEqual(W2, W)
        np.testing.assert_array_equal(f2.numpy().ravel(), f)
        # float32 input = float64 evaluation at the rounded positions
        x32 = x.astype(np.float32)
        W3, f3, _ = asm.evaluate(x32.astype(np.float64), need_tangent=False)
        W4, f4, _ = asm.evaluate(wp.array(x32.reshape(-1, 3), dtype=wp.vec3f, device=self.device), need_tangent=False)
        W5, f5, _ = asm.evaluate(x32, need_tangent=False)
        self.assertEqual(W4, W3)
        self.assertEqual(W5, W3)
        np.testing.assert_array_equal(f4, f3)
        np.testing.assert_array_equal(f5, f3)

    def test_inverted_neo_hookean_raises(self):
        asm = self.make("neo_hookean")
        x = (self.mesh.nodes * [1.0, 1.0, -1.0]).reshape(-1)
        with self.assertRaises(FloatingPointError):
            asm.evaluate(x)
        with self.assertRaises(FloatingPointError):
            asm.element_terms(x.reshape(-1, 3)[self.mesh.tets[:2]], np.array([0, 1]))
        asm.evaluate(asm.X_flat)  # the flag is reset per call

    def test_auto_scatter(self):
        expected = "atomic" if wp.get_device(self.device).is_cpu else "gather"
        self.assertEqual(self.make().scatter, expected)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.make("mooney")
        with self.assertRaises(ValueError):
            self.make(scatter="sorted")
        with self.assertRaises(ValueError):
            self.make().evaluate(self.make().X_flat, tangent_format="coo")
        bad = WarpTetMesh(self.mesh.nodes, self.mesh.tets[:, [0, 2, 1, 3]], 1.0, 1.0, 0.3)
        with self.assertRaises(ValueError):
            WarpTetAssembler.from_mesh(bad, device=self.device)


class TestWarpTetAssemblerCPU(_AssemblerTests, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
class TestWarpTetAssemblerCUDA(_AssemblerTests, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
