# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Transactional updates: a rejected candidate leaves the Model, the State, the control, the controller record
and the adapter bit-identical (host and device update paths, CPU and CUDA); validation reason codes; back-off
and the rejection limit."""

from __future__ import annotations

import _fixtures  # noqa: F401  (puts the rerom package on sys.path)

# isort: split

import unittest
import warnings

import numpy as np
import rerom
import warp as wp
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
from rerom import craig_bampton as CB
from rerom.assembly_warp import WarpTetAssembler

import newton

FREE_FLIGHT_YOUNG = 1.0e7
CANTILEVER_YOUNG = 1.0e8
PATHS = (rerom.UpdatePath.HOST, rerom.UpdatePath.DEVICE)
FORCED_STAGES = ("basis", "interface_map", "commit")

_MODEL_ARRAYS = [
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
_CTRL_ATTRS = [
    "r_ref",
    "Phi",
    "com",
    "g0",
    "lam",
    "lam_raw",
    "W_ref",
    "steps_since_update",
    "steps_since_fold",
    "last_joint_rest_mismatch",
    "_qdd_est",
    "_dev_hist_ready",
    "_dev_qdd_valid",
]
_DEV_BUFFERS = ["phi", "r_ref", "com", "prestress", "qd_hist", "qdd"]


def _devices():
    return ["cpu", "cuda:0"] if HAVE_CUDA else ["cpu"]


def _np(a):
    return None if a is None else np.array(a.numpy() if hasattr(a, "numpy") else a, copy=True)


def _snapshot(ctrl, state, control):
    """Everything an update may write: Model arrays, State, control, controller record, adapter, device buffers."""
    model = ctrl.model
    snap = {f"model.{k}": _np(getattr(model, k)) for k in _MODEL_ARRAYS if getattr(model, k, None) is not None}
    for k in ("joint_q", "joint_qd", "body_q", "body_qd"):
        if getattr(state, k, None) is not None:
            snap[f"state.{k}"] = _np(getattr(state, k))
    snap["control.joint_f"] = _np(control.joint_f)
    for k in _CTRL_ATTRS:
        v = getattr(ctrl, k, None)
        snap[f"ctrl.{k}"] = None if v is None else np.array(v, copy=True)
    snap["ctrl.stats"] = len(ctrl.stats)
    snap["ctrl.clamp_log"] = len(ctrl.clamp_log)
    snap["ctrl._qd_hist"] = None if ctrl._qd_hist is None else np.array(ctrl._qd_hist[0], copy=True)
    ad = ctrl.adapter
    for k in ("tip_pos", "frame_map", "offset"):
        if hasattr(ad, k):
            v = getattr(ad, k)
            snap[f"adapter.{k}"] = None if v is None else np.array(v, copy=True)
    dev = ctrl.device_updater
    if dev is not None:
        for k in _DEV_BUFFERS:
            snap[f"dev.{k}"] = _np(getattr(dev, k))
        snap["dev.prestress_host"] = None if dev.prestress_host is None else dev.prestress_host.copy()
    return snap


def _assert_identical(test, a, b):
    test.assertEqual(sorted(a), sorted(b))
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            test.assertTrue(
                va is not None and vb is not None and va.shape == vb.shape and np.array_equal(va, vb, equal_nan=True),
                msg=f"{k} changed by a rejected update",
            )
        else:
            test.assertEqual(va, vb, msg=f"{k} changed by a rejected update")


def _held_free_beam(path, device):
    """Free-free basis held by a fixed joint: endpoint frames, counter-rotation, render points, derived psi."""
    asm = beam_assembler(young=CANTILEVER_YOUNG)
    gen, _b, model, body, _fixed = build_cantilever(asm, 8, damping_ratio=0.3, clamped=False, device=device)
    solver = newton.solvers.SolverVBD(model, iterations=8)
    control = model.control()
    s0, _s1 = run(solver, model.state(), model.state(), control, 30, 2.0e-3)
    a = asm if path is rerom.UpdatePath.HOST else WarpTetAssembler.from_mesh(asm.mesh, device=device)
    ctrl = free_controller(model, solver, body, a, gen, n_ritz=1, policy=rerom.UpdatePolicy(path=path))
    return ctrl, s0, control, solver


def _offset_cb_beam(path, device):
    """One Craig-Bampton beam with offset pivots (the adapter mutates its offset, tip position and frame map)."""
    mesh = rerom.make_beam_mesh(0.4, 0.02, 0.02, 20, 2, 2, density=1100.0, young=3.0e7, poisson=0.35)
    root, tip = mesh.node_sets["root"], mesh.node_sets["tip"]
    seg = CB.segment_from_mesh(mesh, root, tip, mode_count=10)
    d = np.array([0.0, 0.0, 0.03])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    seg.body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(*seg.origin.astype(np.float32)), wp.quat_identity()),
        modal_basis=CB.modal_basis_for(seg, "beam"),
    )
    tip_world = seg.origin + seg.tip_point + d
    link = builder.add_body(xform=wp.transform(wp.vec3(*tip_world.astype(np.float32)), wp.quat_identity()), mass=0.01)
    builder.add_shape_box(link, hx=0.01, hy=0.01, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=100.0))

    def tf(p):
        return wp.transform(wp.vec3(*np.asarray(p, dtype=np.float32)), wp.quat_identity())

    builder.add_joint_fixed(parent=-1, child=seg.body, parent_xform=tf(seg.origin + d), child_xform=tf(d))
    builder.add_joint_fixed(
        parent=seg.body, child=link, parent_xform=tf(seg.tip_point + d), child_xform=tf(np.zeros(3))
    )
    builder.color()
    model = builder.finalize(device=device)
    CB.install_interface_endpoint_maps(model, [seg], offset=d)
    solver = newton.solvers.SolverVBD(model, iterations=10)
    adapter = rerom.InterfaceBodyAdapter(seg, endpoint_offset=d)
    ctrl = rerom.ReROMController(
        model,
        solver,
        seg.body,
        seg.asm,
        adapter,
        basis=rerom.BasisOptions(n_ritz=2),
        trigger=rerom.RotationTrigger(theta_tol=None),
        policy=rerom.UpdatePolicy(rest_rotation=rerom.RestRotationMode.JOINT_TREE, path=path),
    )
    control = model.control()
    s0, _ = run(solver, model.state(), model.state(), control, 60, 2.0e-3, ctrl)
    return ctrl, s0, control, solver


class TestForcedRejection(unittest.TestCase):
    def _check_case(self, build):
        for device in _devices():
            for path in PATHS:
                with self.subTest(device=device, path=path.value):
                    ctrl, state, control, _ = build(path, device)
                    twin, tstate, tcontrol, _ = build(path, device)
                    before = _snapshot(ctrl, state, control)
                    for stage in FORCED_STAGES:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            info = ctrl.relinearize(state, control, debug={"reject_at": stage})
                        _assert_identical(self, before, _snapshot(ctrl, state, control))
                        self.assertFalse(info["accepted"])
                        self.assertEqual((info["reason"], info["stage"]), ("forced", stage))
                    self.assertEqual(ctrl.rejection_count, len(FORCED_STAGES))
                    self.assertEqual(ctrl.update_count, 0)
                    # the accepted update after the rejections equals an update without them, bit for bit
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.assertTrue(ctrl.relinearize(state, control)["accepted"])
                        self.assertTrue(twin.relinearize(tstate, tcontrol)["accepted"])
                    a, b = _snapshot(ctrl, state, control), _snapshot(twin, tstate, tcontrol)
                    _assert_identical(self, a, b)
                    self.assertEqual(ctrl.consecutive_rejections, 0)

    def test_free_body_with_endpoints(self):
        self._check_case(_held_free_beam)

    def test_interface_body_with_offset(self):
        self._check_case(_offset_cb_beam)


class TestValidation(unittest.TestCase):
    def _free(self, path, *, asm=None, policy_kw=None, device="cpu", warp=False):
        inner = beam_assembler(young=FREE_FLIGHT_YOUNG)
        gen, _b, model, state, body = build_free_beam(inner, 6, shape=False, device=device)
        solver = newton.solvers.SolverVBD(model, iterations=4)
        a = asm if asm is not None else inner
        if warp:
            a = WarpTetAssembler.from_mesh(inner.mesh, device=device)
        policy = rerom.UpdatePolicy(path=path, **(policy_kw or {}))
        ctrl = free_controller(model, solver, body, a, gen, policy=policy)
        return ctrl, model, state, inner

    def _relin(self, ctrl, state, control=None, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ctrl.relinearize(state, control, **kw)

    def test_inverted_reference_is_rejected(self):
        """det F <= 0 in the candidate reference is rejected (also when a rotation criterion would be silent);
        the device path with device assembly no longer raises."""
        cases = [(p, False) for p in PATHS] + [(rerom.UpdatePath.DEVICE, True)]
        for path, warp in cases:
            with self.subTest(path=path.value, warp=warp):
                ctrl, model, state, _ = self._free(path, warp=warp)
                amp = 1.0
                while ctrl.min_det_f(ctrl.r_ref.reshape(-1) + amp * ctrl.Phi[:, 0]) > 0.0:
                    amp *= 2.0
                set_modal_coordinates(state, ctrl, q=np.r_[amp, np.zeros(ctrl.m - 1)])
                control = model.control()
                before = _snapshot(ctrl, state, control)
                info = self._relin(ctrl, state, control)
                self.assertEqual((info["accepted"], info["reason"]), (False, "inverted_element"))
                _assert_identical(self, before, _snapshot(ctrl, state, control))

    def test_reject_guard_refuses_a_negative_tangent(self):
        """IndefiniteGuard.REJECT refuses a materially negative restricted tangent; KEEP accepts it."""
        for path in PATHS:
            with self.subTest(path=path.value):
                inner = beam_assembler(young=FREE_FLIGHT_YOUNG)
                asm = PrestressedQuadraticAssembler(inner, inner.mesh.nodes * np.array([0.98, 1.0, 1.0]))
                ctrl, model, state, _ = self._free(
                    path, asm=asm, policy_kw={"guard_indefinite": rerom.IndefiniteGuard.REJECT}
                )
                control = model.control()
                before = _snapshot(ctrl, state, control)
                info = self._relin(ctrl, state, control)
                self.assertEqual((info["accepted"], info["reason"]), (False, "negative_stiffness"))
                _assert_identical(self, before, _snapshot(ctrl, state, control))
                ctrl_keep, _, state_keep, _ = self._free(path, asm=asm)
                self.assertTrue(self._relin(ctrl_keep, state_keep)["accepted"])
                self.assertLess(ctrl_keep.lam_raw.min(), 0.0)

    def test_non_finite_and_transfer_budget(self):
        """A non-finite internal force and a transfer beyond the energy budget are rejected with their codes."""
        for path in PATHS:
            with self.subTest(path=path.value):
                ctrl, _model, state, inner = self._free(path)
                W, f, K = inner.evaluate(inner.mesh.nodes.reshape(-1))
                f = f.copy()
                f[3] = np.nan
                info = self._relin(ctrl, state, stiffness_override=(W, f, K))
                self.assertEqual(info["reason"], "non_finite")
                # spinning, vibrating beam: a free switch loses some kinetic energy; a zero budget refuses it
                jqd = state.joint_qd.numpy()
                jqd[:6] = [0.2, -0.1, 0.3, 0.5, 0.4, 2.0]
                jqd[6:8] = [0.4, 0.2]
                state.joint_qd.assign(jqd)
                set_modal_coordinates(state, ctrl, q=np.r_[0.02, np.zeros(ctrl.m - 1)])
                ctrl.policy.validation.transfer_energy_budget = 0.0
                info = self._relin(ctrl, state)
                self.assertEqual((info["accepted"], info["reason"]), (False, "transfer_energy"))
                ctrl.policy.validation.transfer_energy_budget = 1.0e-2
                ctrl.request_update()  # clears the back-off
                info = self._relin(ctrl, state)
                self.assertTrue(info["accepted"])
                self.assertLess(abs(info["delta_KE"]), 1.0e-2 * info["KE_before"])

    def test_ill_conditioned_interface_transfer_is_rejected(self):
        for path in PATHS:
            with self.subTest(path=path.value):
                ctrl, state, control, _ = _offset_cb_beam(path, "cpu")
                ctrl.policy.validation.interface_cond_max = 1.0  # any real interface map is worse
                before = _snapshot(ctrl, state, control)
                info = self._relin(ctrl, state, control)
                self.assertEqual(
                    (info["accepted"], info["reason"], info["stage"]), (False, "interface_constraint", "transfer")
                )
                _assert_identical(self, before, _snapshot(ctrl, state, control))

    def test_backoff_limit_and_request(self):
        """After a rejection the trigger backs off (suppressed alarms are recorded); after the rejection limit
        updates stop until request_update()."""
        trig = rerom.RotationTrigger(theta_tol=0.0)  # fires on any rotation
        ctrl, model, state, _inner = self._free(rerom.UpdatePath.HOST)
        ctrl.trigger = trig
        v = ctrl.policy.validation
        v.backoff_steps, v.max_consecutive_rejections = 3, 2
        v.transfer_energy_budget = 0.0  # every switch of a moving beam is refused
        solver = newton.solvers.SolverVBD(model, iterations=4)
        ctrl.solver = solver
        jqd = state.joint_qd.numpy()
        jqd[6:8] = [0.4, 0.2]
        state.joint_qd.assign(jqd)
        control = model.control()
        s0, s1 = state, model.state()
        attempts = []
        for i in range(1, 16):
            s0, s1 = run(solver, s0, s1, control, 1, 1.0e-3, ctrl)
            n = ctrl.rejection_count
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertFalse(ctrl.after_step(s0, control, 1.0e-3))
            if ctrl.rejection_count > n:
                attempts.append(i)
        self.assertEqual(attempts, [1, 4])  # back-off of 3 steps, then blocked after 2 rejections
        self.assertEqual(ctrl.update_suppression(), "rejection_limit")
        self.assertGreater(ctrl.accuracy_unverified["count"], 0)
        self.assertIn(ctrl.accuracy_unverified["suppressed_by"], ("rejection_backoff", "rejection_limit"))
        self.assertEqual(ctrl.step_log[0]["update_rejected"], "transfer_energy")
        v.transfer_energy_budget = None
        ctrl.request_update()
        self.assertIsNone(ctrl.update_suppression())
        s0, s1 = run(solver, s0, s1, control, 1, 1.0e-3, ctrl)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertTrue(ctrl.after_step(s0, control, 1.0e-3))
        self.assertEqual(ctrl.consecutive_rejections, 0)


if __name__ == "__main__":
    unittest.main()
