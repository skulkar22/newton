# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""State transfer at an update: constrained transfer in the reduced mass metric, signed energy diagnostics,
velocity enrichment."""

from __future__ import annotations

import _fixtures  # noqa: F401  (puts the rerom package on sys.path)

# isort: split

import unittest

import numpy as np
from rerom.numerics import constrain_frame_velocity, transfer_modal_velocity


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


if __name__ == "__main__":
    unittest.main()
