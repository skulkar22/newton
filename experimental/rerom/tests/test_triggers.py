# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Trigger defaults, the composite (OR) trigger, validity checks that bypass the gating and the record of
suppressed accuracy alarms."""

from __future__ import annotations

import _fixtures  # noqa: F401  (puts the rerom package on sys.path)

# isort: split

import math
import unittest
from types import SimpleNamespace

import numpy as np
import rerom
from _fixtures import LinearStubAssembler, beam_assembler, build_free_beam, free_controller, run, set_modal_coordinates

import newton

FREE_FLIGHT_YOUNG = 1.0e7


def _field_stub(field, amplitude=1.0):
    """Minimal controller view of a prescribed field ``x = X + amplitude * field`` (review's probe pattern)."""
    mesh = rerom.make_beam_mesh(1.0, 0.1, 0.1, 4, 1, 1, density=1000.0, young=1.0e6, poisson=0.3)
    asm = rerom.NumpyTetAssembler(mesh, "stvk")
    X = mesh.nodes.copy()
    phi = field(X)
    return SimpleNamespace(
        assembler=asm,
        r_ref=X,
        Phi=phi.reshape(-1, 1),
        char_length=1.0,
        device_updater=None,
        read_modes=lambda state: (np.array([amplitude]), np.zeros(1)),
    )


def _rotation_field(theta):
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return lambda X: X @ R.T - X


def _extension_field(X):
    phi = np.zeros_like(X)
    phi[:, 0] = X[:, 0]
    return phi


class _AlwaysTrigger(rerom.Trigger):
    def check(self, ctrl, state, steps_since, metrics):
        metrics["always"] = 1.0
        return True, "always", metrics


class TestRotationDefault(unittest.TestCase):
    def test_default_is_half_a_degree_and_four_is_reproducible(self):
        """The class default is 0.5 deg; ``theta_tol=radians(4)`` reproduces the former default behaviour."""
        self.assertAlmostEqual(rerom.RotationTrigger().theta_tol, math.radians(0.5))
        stub = _field_stub(_rotation_field(math.radians(1.0)))
        opts = rerom.TriggerOptions(n_sample_elements=None)
        fire, reason, m = rerom.RotationTrigger(options=opts).check(stub, None, 1, {})
        self.assertTrue(fire)
        self.assertEqual(reason, "rotation")
        self.assertAlmostEqual(m["theta_max"], math.radians(1.0), places=6)
        fire4, _, _ = rerom.RotationTrigger(theta_tol=math.radians(4.0), options=opts).check(stub, None, 1, {})
        self.assertFalse(fire4)


class TestCompositeTrigger(unittest.TestCase):
    def test_pure_extension_is_missed_by_rotation_and_caught_by_the_composite(self):
        """Uniform axial extension (zero polar rotation): a 0.2 deg rotation trigger does not fire; the composite
        with the energy Taylor-defect trigger does, and reports both metrics and which child fired."""
        stub = _field_stub(_extension_field, amplitude=0.10)
        opts = rerom.TriggerOptions(n_sample_elements=None)
        rot = rerom.RotationTrigger(theta_tol=math.radians(0.2), options=opts)
        energy = rerom.EnergyMismatchTrigger(tol=0.05, options=opts)
        fire, _, m = rot.check(stub, None, 1, {})
        self.assertFalse(fire)
        self.assertLess(m["theta_max"], 1e-12)
        comp = rerom.CompositeTrigger(triggers=[rot, energy], options=opts)
        fire, reason, m = comp.check(stub, None, 1, {})
        self.assertTrue(fire)
        self.assertEqual(reason, "energy")
        self.assertEqual(m["fired"], ["energy"])
        self.assertIn("theta_max", m)
        self.assertGreater(m["energy_mismatch"], 0.05)

    def test_composite_through_the_controller(self):
        """A composite of a never-firing and an always-firing child fires through the controller's poll."""
        asm = LinearStubAssembler(beam_assembler(young=FREE_FLIGHT_YOUNG))
        gen, _, model, state, body = build_free_beam(asm.inner, 6, shape=False)
        solver = newton.solvers.SolverVBD(model, iterations=4)
        comp = rerom.CompositeTrigger(triggers=[rerom.RotationTrigger(theta_tol=None), _AlwaysTrigger()])
        ctrl = free_controller(model, solver, body, asm, gen, trigger=comp)
        fire, reason, metrics = ctrl.poll(state, 1.0e-3)
        self.assertTrue(fire)
        self.assertEqual(reason, "always")
        self.assertEqual(metrics["fired"], ["always"])


class TestValidityAndSuppression(unittest.TestCase):
    def _controller(self, trigger, path=rerom.UpdatePath.HOST, policy_kw=None):
        asm = beam_assembler(young=FREE_FLIGHT_YOUNG)
        gen, _, model, state, body = build_free_beam(asm, 6, shape=False)
        solver = newton.solvers.SolverVBD(model, iterations=4)
        policy = rerom.UpdatePolicy(path=path, **(policy_kw or {}))
        ctrl = free_controller(model, solver, body, asm, gen, trigger=trigger, policy=policy)
        return ctrl, model, solver, state

    def test_validity_fires_under_budget_exhaustion_and_freeze(self):
        """Non-finite modal state and inverted elements raise a validity alarm although the update budget is
        exhausted (max_updates=0) and the controller is frozen; the rotation criterion alone stays gated."""
        for path in (rerom.UpdatePath.HOST, rerom.UpdatePath.DEVICE):
            with self.subTest(path=path.value):
                trig = rerom.CompositeTrigger(
                    triggers=[rerom.RotationTrigger(), rerom.ValidityTrigger(options=rerom.TriggerOptions())],
                    options=rerom.TriggerOptions(max_updates=0),
                )
                ctrl, _model, _solver, state = self._controller(trig, path, {"freeze_in_contact": True})
                ctrl.set_phase(in_contact=True)
                m = ctrl.m
                # bend the first mode until elements invert in the linear field
                phi0 = ctrl.Phi[:, 0]
                amp = 1.0
                while ctrl.min_det_f(ctrl.r_ref.reshape(-1) + amp * phi0) > 0.0:
                    amp *= 2.0
                q = np.zeros(m)
                q[0] = amp
                set_modal_coordinates(state, ctrl, q=q)
                fire, reason, metrics = ctrl.poll(state, 1.0e-3)
                self.assertTrue(fire)
                self.assertEqual(reason, "validity_det_f")
                self.assertLessEqual(metrics["det_f_min"], 0.0)
                q[1] = np.nan
                set_modal_coordinates(state, ctrl, q=q)
                fire, reason, _ = ctrl.poll(state, 1.0e-3)
                self.assertTrue(fire)
                self.assertEqual(reason, "validity_non_finite")
                # a valid but strongly rotated state: rotation alarm, gated by the budget -> recorded, not fired
                set_modal_coordinates(state, ctrl, q=np.r_[0.5 * amp / 8.0, np.zeros(m - 1)])
                n0 = ctrl.accuracy_unverified["count"]
                fire, reason, _ = ctrl.poll(state, 1.0e-3)
                self.assertFalse(fire)
                self.assertEqual(ctrl.accuracy_unverified["count"], n0 + 1)
                self.assertEqual(ctrl.accuracy_unverified["last_step"], ctrl.step_count)
                self.assertEqual(ctrl.accuracy_unverified["last_reason"], "rotation")
                self.assertEqual(ctrl.accuracy_unverified["suppressed_by"], "budget")

    def test_min_interval_suppression_is_recorded(self):
        """An alarm inside the minimum interval does not fire but is counted as accuracy-unverified."""
        trig = _AlwaysTrigger(options=rerom.TriggerOptions(min_interval=3))
        asm = LinearStubAssembler(beam_assembler(young=FREE_FLIGHT_YOUNG))
        gen, _, model, state, body = build_free_beam(asm.inner, 6, shape=False)
        solver = newton.solvers.SolverVBD(model, iterations=4)
        ctrl = free_controller(model, solver, body, asm, gen, trigger=trig)
        control = model.control()
        s0, s1 = state, model.state()
        fired = []
        for i in range(1, 10):
            s0, s1 = run(solver, s0, s1, control, 1, 1.0e-3, ctrl)
            if ctrl.after_step(s0, control, 1.0e-3):
                fired.append(i)
        self.assertEqual(fired, [3, 6, 9])
        self.assertEqual(ctrl.accuracy_unverified["count"], 6)
        self.assertEqual(ctrl.accuracy_unverified["suppressed_by"], "min_interval")
        trig.options.audit_suppressed = False
        ctrl.accuracy_unverified["count"] = 0
        for _ in range(3):
            s0, s1 = run(solver, s0, s1, control, 1, 1.0e-3, ctrl)
            ctrl.after_step(s0, control, 1.0e-3)
        self.assertEqual(ctrl.accuracy_unverified["count"], 0)


if __name__ == "__main__":
    unittest.main()
