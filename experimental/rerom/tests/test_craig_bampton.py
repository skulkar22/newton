# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of Craig-Bampton interface bodies: chain statics, interface coincidence and momentum across updates,
null update, rest-rotation propagation, endpoint offsets (CPU)."""

from __future__ import annotations

import _fixtures  # noqa: F401  (puts the rerom package on sys.path)

# isort: split

import copy
import unittest

import numpy as np
import rerom
import warp as wp
from _fixtures import run
from rerom import craig_bampton as CB
from rerom.numerics import quat_from_rotvec, quat_to_matrix

import newton

RHO, E, NU = 1100.0, 3.0e7, 0.35


def small_beam(nx=40, ny=2, nz=2, L=0.8, h=0.02):
    mesh = rerom.make_beam_mesh(L, h, h, nx, ny, nz, density=RHO, young=E, poisson=NU)
    return mesh, np.asarray(mesh.nodes), np.asarray(mesh.tets)


def _chain(n_seg=3, keep_velocity=False):
    _, nodes, tets = small_beam()
    chain = rerom.build_cb_chain(
        nodes, tets, RHO, E, NU, n_segments=n_seg, mode_count=12, n_ritz_initial=1, device="cpu"
    )
    solver = newton.solvers.SolverVBD(chain.model, iterations=10)
    mfc = rerom.MultiFrameController(chain, solver, theta_deg=None, keep_velocity=keep_velocity)
    return chain, solver, mfc


def _swing(chain, solver, mfc, steps, s0=None, s1=None):
    s0 = chain.model.state() if s0 is None else s0
    s1 = chain.model.state() if s1 is None else s1
    ctl = chain.model.control()
    for _ in range(steps):
        mfc.apply_prestress(ctl)
        s0.clear_forces()
        solver.step(s0, s1, ctl, None, 2e-3)
        s0, s1 = s1, s0
    return s0, s1, ctl


class TestChainStatics(unittest.TestCase):
    def test_cb_chain_linear_static_vs_full_fe(self):
        """Fixed-basis CB chain (RBE2 cuts, exact welds): linear static sag within 3 % of full FE, and equal to
        Newton's ModalGeneratorCraigBamptonFEM with RBE2 interfaces."""
        mesh, nodes, tets = small_beam()
        g = np.array([0.0, 0.0, -9.81])
        u_ref = CB.full_fe_linear_static(nodes, tets, RHO, E, NU, g, mesh.node_sets["root"])
        tip = mesh.node_sets["tip"]
        z_ref = u_ref.reshape(-1, 3)[tip, 2].mean()
        for n_seg in (2, 3):
            with self.subTest(n_seg=n_seg):
                segs = CB.make_segments(nodes, tets, RHO, E, NU, n_seg)
                for s in segs:
                    s.Phi, s.lam, s.frame_map, s.basis_info = CB.cb_basis(
                        s.local_nodes, s.asm.linear_stiffness(), s.M, s.root_ids, s.tip_ids, s.tip_point, 12
                    )
                    if s.has_tip:
                        self.assertLess(s.basis_info["frame_map_residual"], 1e-12)
                    np.testing.assert_allclose(s.Phi.T @ (s.M @ s.Phi), np.eye(12), atol=1e-9)
                u, _ = CB.linear_chain_static(segs, nodes.shape[0], g)
                z = u.reshape(-1, 3)[tip, 2].mean()
                self.assertLess(abs(z / z_ref - 1.0), 0.03)
                un, info = CB.newton_cb_chain_static(segs, nodes.shape[0], g, coupling="rbe2", fixed_mode_count=6)
                self.assertLess(abs(un.reshape(-1, 3)[tip, 2].mean() / z - 1.0), 1e-5)
                self.assertLess(info["interface_node_gap"], 1e-9 * abs(z_ref))


class TestChainUpdates(unittest.TestCase):
    def test_null_update_reproduces_chain(self):
        """Updating at rest reproduces masses, COMs, rest targets and the static gravity response of the chain."""
        chain, _, mfc = _chain()
        s0 = chain.model.state()
        ctl = chain.model.control()
        m = chain.model
        mass0, com0 = m.body_mass.numpy().copy(), m.body_com.numpy().copy()
        mfc.update(s0, ctl, [0, 1, 2])
        np.testing.assert_allclose(m.body_mass.numpy(), mass0, rtol=1e-6)
        np.testing.assert_allclose(m.body_com.numpy(), com0, atol=1e-7)
        self.assertLess(mfc.rest_target_error(), 1e-7)
        g = np.array([0.0, 0.0, -9.81])
        new_segs = []
        for c, s in zip(mfc.ctrls, chain.segments, strict=True):
            s2 = copy.copy(s)
            s2.Phi, s2.lam, s2.frame_map = c.Phi, c.lam, c.adapter.frame_map
            new_segs.append(s2)
            if c.adapter.frame_map is not None:
                np.testing.assert_allclose(c.adapter.tip_pos, s.tip_point, atol=1e-12)
        u0, _ = CB.linear_chain_static(chain.segments, chain.n_nodes, g)
        u1, _ = CB.linear_chain_static(new_segs, chain.n_nodes, g)
        self.assertLess(np.max(np.abs(u1 - u0)), 1e-6 * np.max(np.abs(u0)))

    def test_interface_coincidence_after_update(self):
        """After updating one / all bodies of a swinging chain the interface frames and the shared face nodes stay
        where they were (float32), every joint keeps its rest target, and the joint mismatch does not jump."""
        for which in ([1], [0, 1, 2]):
            with self.subTest(which=which):
                chain, solver, mfc = _chain()
                s0, s1, ctl = _swing(chain, solver, mfc, 120)
                poses0 = mfc.frame_poses(s0)
                X0, _, _ = mfc.full_field(s0)
                gap0, ang0, _, _ = mfc.interface_mismatch(s0)
                mfc.update(s0, ctl, which)
                poses1 = mfc.frame_poses(s0)
                X1, _, _ = mfc.full_field(s0)
                for j in poses0:
                    for a, b in zip(poses0[j], poses1[j], strict=True):
                        np.testing.assert_allclose(a, b, atol=2e-6)
                self.assertLess(np.max(np.abs(X1 - X0)), 2e-6)
                self.assertLess(mfc.rest_target_error(), 1e-6)
                s0, s1, _ = _swing(chain, solver, mfc, 30, s0, s1)
                gap2, ang2, _, _ = mfc.interface_mismatch(s0)
                self.assertLess(gap2, 5.0 * gap0 + 1e-5)
                self.assertLess(ang2, 5.0 * ang0 + 1e-4)

    def test_mean_counter_rotation_alone_fails_on_a_chain_body(self):
        """Control: the single-body mean counter-rotation leaves rest-target errors of the order of half the
        tip-frame rotation on a chain body (unequal endpoint rotations); the joint-tree propagation fixes them."""
        chain, solver, mfc = _chain()
        s0, _, ctl = _swing(chain, solver, mfc, 120)
        c = mfc.ctrls[1]
        q, _ = c.read_modes(s0)
        tip_theta = np.linalg.norm(c.adapter.frame_map[3:] @ q)
        import warnings  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c.relinearize(s0, ctl)
        self.assertGreater(mfc.rest_target_error(), 0.25 * tip_theta)
        self.assertGreater(0.25 * tip_theta, 1e-4)
        mfc.fix_rest_rotations()
        self.assertLess(mfc.rest_target_error(), 1e-6)

    def test_assembly_momentum_across_update(self):
        """Chain momentum across a simultaneous update: preserved to the projection residual; to float32 when the old
        elastic velocity is kept as a basis vector; interface velocities preserved exactly."""
        for keep_velocity in (False, True):
            with self.subTest(keep_velocity=keep_velocity):
                chain, solver, mfc = _chain(keep_velocity=keep_velocity)
                s0, _, ctl = _swing(chain, solver, mfc, 100)
                P0, L0 = mfc.assembly_momentum(s0)
                infos = mfc.update(s0, ctl, [0, 1, 2])
                P1, L1 = mfc.assembly_momentum(s0)
                dP = np.linalg.norm(P1 - P0) / np.linalg.norm(P0)
                dL = np.linalg.norm(L1 - L0) / np.linalg.norm(L0)
                self.assertLess(max(i["tip_frame_velocity_change"] or 0.0 for i in infos), 1e-9)
                ke = max(i["KE_lost_fraction_total"] for i in infos)
                tol = (1e-5, 1e-4) if keep_velocity else (2e-2, 2e-2)
                self.assertLess(dP, tol[0])
                self.assertLess(dL, tol[0])
                self.assertLess(ke, tol[1])

    def test_after_step_triggers(self):
        chain, solver, _ = _chain()
        mfc = rerom.MultiFrameController(chain, solver, theta_deg=0.3, sync=rerom.UpdateSync.SYNC)
        s0, s1 = chain.model.state(), chain.model.state()
        ctl = chain.model.control()
        events = []
        for _ in range(80):
            mfc.apply_prestress(ctl)
            s0.clear_forces()
            solver.step(s0, s1, ctl, None, 2e-3)
            s0, s1 = s1, s0
            events += mfc.after_step(s0, ctl, 2e-3)
        self.assertTrue(events)
        self.assertEqual(len(events) % 3, 0)  # sync: all bodies update together
        self.assertLess(mfc.rest_target_error(), 1e-6)


class TestEndpointOffset(unittest.TestCase):
    def test_offset_endpoints_follow_and_rest_targets_hold(self):
        """One CB beam whose joints sit at offset pivots (root pivot -> world, tip pivot -> rigid link): an update
        keeps both pivots where they were and every joint's rest target (joint-tree propagation)."""
        mesh, _, _ = small_beam(nx=20, L=0.4)
        root, tip = mesh.node_sets["root"], mesh.node_sets["tip"]
        seg = CB.segment_from_mesh(mesh, root, tip, mode_count=10)
        d = np.array([0.0, 0.0, 0.03])
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        seg.body = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(*seg.origin.astype(np.float32)), wp.quat_identity()),
            modal_basis=CB.modal_basis_for(seg, "beam"),
        )
        tip_world = seg.origin + seg.tip_point + d
        link = builder.add_body(
            xform=wp.transform(wp.vec3(*tip_world.astype(np.float32)), wp.quat_identity()), mass=0.01
        )
        builder.add_shape_box(link, hx=0.01, hy=0.01, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=100.0))
        tf = lambda p: wp.transform(wp.vec3(*np.asarray(p, dtype=np.float32)), wp.quat_identity())  # noqa: E731
        builder.add_joint_fixed(parent=-1, child=seg.body, parent_xform=tf(seg.origin + d), child_xform=tf(d))
        builder.add_joint_fixed(
            parent=seg.body, child=link, parent_xform=tf(seg.tip_point + d), child_xform=tf(np.zeros(3))
        )
        builder.color()
        model = builder.finalize(device="cpu")
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
            policy=rerom.UpdatePolicy(rest_rotation=rerom.RestRotationMode.JOINT_TREE),
        )
        self.assertEqual(adapter.ep_is_tip.count(True), 1)
        control = model.control()
        s0, s1 = run(solver, model.state(), model.state(), control, 100, 2e-3, ctrl)

        def pivots(state):
            bq = state.body_q.numpy()
            Xp = model.joint_X_p.numpy()
            Mx = int(model.elastic_max_mode_count)
            ep_phi = model.elastic_endpoint_phi.numpy()
            ep_psi = model.elastic_endpoint_psi.numpy()
            q, _ = ctrl.read_modes(state)
            out = []
            for ep in ctrl.endpoints:
                rows = ep.endpoint * Mx + np.arange(ctrl.m)
                X = Xp[ep.joint] if ep.side == 0 else model.joint_X_c.numpy()[ep.joint]
                pos = X[:3].astype(np.float64) + ep_phi[rows].astype(np.float64).T @ q
                rot = quat_to_matrix(quat_from_rotvec(ep_psi[rows].astype(np.float64).T @ q)) @ quat_to_matrix(X[3:7])
                Rb = quat_to_matrix(bq[seg.body, 3:7])
                out.append((bq[seg.body, :3] + Rb @ pos, Rb @ rot))
            return out

        before = pivots(s0)
        ctrl.relinearize(s0, control)
        after = pivots(s0)
        for (p0, R0), (p1, R1) in zip(before, after, strict=True):
            np.testing.assert_allclose(p1, p0, atol=2e-6)
            np.testing.assert_allclose(R1, R0, atol=2e-6)
        self.assertLess(ctrl.rest_propagator.error(), 1e-6)
        s0, _ = run(solver, s0, s1, control, 30, 2e-3, ctrl)
        self.assertTrue(np.all(np.isfinite(ctrl.world_field(s0))))


if __name__ == "__main__":
    unittest.main()
