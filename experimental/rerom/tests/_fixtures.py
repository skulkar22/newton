# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared builders of the re-ROM tests (small beams, free and clamped Newton models)."""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import numpy as np
import rerom
import warp as wp

import newton

wp.config.quiet = True

DEVICE = "cpu"
HAVE_CUDA = wp.is_cuda_available()


class LinearStubAssembler:
    """f_int = K (x - X), K_T = K: isolates pre-stress and projection logic from the nonlinearity."""

    def __init__(self, inner):
        self.inner = inner
        self.mesh = inner.mesh
        self.mass_matrix = inner.mass_matrix
        self.K = inner.linear_stiffness()
        self.X = inner.mesh.nodes.reshape(-1)

    def evaluate(self, x_flat, *, need_tangent=True):
        u = np.asarray(x_flat) - self.X
        f = self.K @ u
        return 0.5 * float(u @ f), f, (self.K if need_tangent else None)

    def element_rotations(self, x_flat, element_ids=None, reference_flat=None):
        return self.inner.element_rotations(x_flat, element_ids, reference_flat=reference_flat)

    def linear_stiffness(self):
        return self.K


class LinearElementAssembler(LinearStubAssembler):
    """Linear stub with per-element terms: f_e = K_e0 (x_e - X_e), K_e = K_e0."""

    def __init__(self, inner):
        super().__init__(inner)
        X = inner.mesh.nodes
        _, _, K = inner.element_terms(X[inner.mesh.tets], None, need_tangent=True)
        self.Ke0 = K.reshape(-1, 12, 12)
        self.Xe = X[inner.mesh.tets]

    def element_terms(self, x_elem, element_ids=None, *, need_tangent=True):
        sl = slice(None) if element_ids is None else element_ids
        Ke = self.Ke0[sl]
        u = (np.asarray(x_elem) - self.Xe[sl]).reshape(-1, 12)
        fe = np.einsum("eij,ej->ei", Ke, u)
        We = 0.5 * np.einsum("ei,ei->e", u, fe)
        return We, fe.reshape(-1, 4, 3), (Ke.reshape(-1, 4, 3, 4, 3) if need_tangent else None)


class PrestressedQuadraticAssembler(LinearStubAssembler):
    """Quadratic energy about the rest mesh with the stress and tangent of a prescribed deformed state.

    ``W(x) = W_c + f_c.u + 1/2 u^T K_c u`` with ``u = x - X`` and ``(W_c, f_c, K_c)`` the element terms of the
    inner assembler at ``x_c`` (for example a 2 % axially compressed beam, whose restricted tangent is
    indefinite). Element and global terms are consistent, so a re-linearization anywhere sees the same
    (indefinite) tangent without a ``stiffness_override`` and the IMEX hyper-reduced force is exactly linear.
    """

    def __init__(self, inner, x_c):
        super().__init__(inner)
        Xc = np.asarray(x_c, dtype=np.float64).reshape(-1, 3)
        We, fe, Ke = inner.element_terms(Xc[inner.mesh.tets], None, need_tangent=True)
        self.We0, self.fe0 = We, fe.reshape(-1, 12)
        self.Ke0 = Ke.reshape(-1, 12, 12)
        self.Xe = inner.mesh.nodes[inner.mesh.tets]
        self.W0, self.f0, self.K = inner.evaluate(Xc.reshape(-1))

    def evaluate(self, x_flat, *, need_tangent=True):
        u = np.asarray(x_flat) - self.X
        f = self.f0 + self.K @ u
        return self.W0 + float(self.f0 @ u) + 0.5 * float(u @ (self.K @ u)), f, (self.K if need_tangent else None)

    def element_terms(self, x_elem, element_ids=None, *, need_tangent=True):
        sl = slice(None) if element_ids is None else element_ids
        Ke, fe0 = self.Ke0[sl], self.fe0[sl]
        u = (np.asarray(x_elem) - self.Xe[sl]).reshape(-1, 12)
        Ku = np.einsum("eij,ej->ei", Ke, u)
        fe = fe0 + Ku
        We = self.We0[sl] + np.einsum("ei,ei->e", fe0, u) + 0.5 * np.einsum("ei,ei->e", u, Ku)
        return We, fe.reshape(-1, 4, 3), (Ke.reshape(-1, 4, 3, 4, 3) if need_tangent else None)


def set_modal_coordinates(state, ctrl, q=None, qd=None):
    """Write the body's modal coordinates / velocities into ``state`` (float32)."""
    if q is not None:
        jq = state.joint_q.numpy()
        jq[ctrl.q_start + 7 : ctrl.q_start + 7 + ctrl.m] = q
        state.joint_q.assign(jq)
    if qd is not None:
        jqd = state.joint_qd.numpy()
        jqd[ctrl.qd_start + 6 : ctrl.qd_start + 6 + ctrl.m] = qd
        state.joint_qd.assign(jqd)


def beam_mesh(nx=12, ny=2, nz=2, length=1.0, width=0.06, height=0.04, young=1.0e9, density=1000.0):
    return rerom.make_beam_mesh(length, width, height, nx, ny, nz, density=density, young=young, poisson=0.3)


def beam_assembler(young=1.0e9, material="stvk", **kw):
    return rerom.NumpyTetAssembler(beam_mesh(young=young, **kw), material)


def fem_basis(asm, mode_count, *, damping_ratio=0.0, derive_psi=False, eigensolver="auto"):
    gen = newton.ModalGeneratorFEM(
        asm.mesh.nodes,
        asm.mass_matrix,
        asm.linear_stiffness(),
        mode_count=mode_count,
        damping_ratio=damping_ratio,
        derive_psi=derive_psi,
        eigensolver=eigensolver,
    )
    return gen, gen.build()


def build_free_beam(
    asm,
    mode_count,
    *,
    mode_q=None,
    mode_qd=None,
    twist=None,
    shape=True,
    damping_ratio=0.0,
    derive_psi=False,
    device=DEVICE,
):
    gen, basis = fem_basis(asm, mode_count, damping_ratio=damping_ratio, derive_psi=derive_psi)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        modal_basis=basis,
        mode_q=mode_q,
        mode_qd=mode_qd,
        lock_inertia=True,
    )
    if shape:
        L, w, h = (float(np.ptp(asm.mesh.nodes[:, k])) for k in range(3))
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
        builder.add_shape_box(
            body,
            xform=wp.transform(wp.vec3(0.5 * L, 0.0, 0.0), wp.quat_identity()),
            hx=0.5 * L,
            hy=0.5 * w,
            hz=0.5 * h,
            cfg=cfg,
        )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    if twist is not None:
        jqd = state.joint_qd.numpy()
        jqd[:6] = twist
        state.joint_qd.assign(jqd)
    return gen, basis, model, state, body


def build_cantilever(asm, mode_count, *, damping_ratio=0.3, gravity=-9.81, clamped=True, device=DEVICE):
    """Cantilever along +x, root face at x = 0, gravity along z.

    clamped=True: basis with the root-face nodes fixed and the frame held by a fixed joint at the root centre;
    clamped=False: free-free basis with derived psi, clamped only through the fixed joint.
    Returns ``(gen, basis, model, body, fixed_nodes or None)``.
    """
    root_nodes = asm.mesh.node_sets["root"]
    idx = int(np.argmin(np.linalg.norm(asm.mesh.nodes, axis=1)))
    anchor = asm.mesh.nodes[idx]
    if clamped:
        gen = newton.ModalGeneratorFEM(
            asm.mesh.nodes,
            asm.mass_matrix,
            asm.linear_stiffness(),
            mode_count=mode_count,
            damping_ratio=damping_ratio,
            fixed_node_indices=root_nodes,
        )
        basis = gen.build()
    else:
        gen, basis = fem_basis(asm, mode_count, damping_ratio=damping_ratio, derive_psi=True)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    body = builder.add_body_elastic(xform=wp.transform_identity(), modal_basis=basis, lock_inertia=True)
    builder.add_joint_fixed(
        parent=-1,
        child=body,
        parent_xform=wp.transform(wp.vec3(*anchor), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(*anchor), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    return gen, basis, model, body, (root_nodes if clamped else None)


def free_controller(model, solver, body, asm, gen, *, fixed=None, **kw):
    """ReROMController with a FreeBodyAdapter (``kw``: controller keywords; ``n_ritz`` shortcut)."""
    n_ritz = kw.pop("n_ritz", 0)
    basis = kw.pop("basis", None) or rerom.BasisOptions(n_ritz=n_ritz)
    adapter = rerom.FreeBodyAdapter(gen.modal_matrix, asm.mesh.nodes, fixed_nodes=fixed)
    return rerom.ReROMController(model, solver, body, asm, adapter, basis=basis, **kw)


def run(solver, s0, s1, control, steps, dt, ctrl=None, after=False, state_arg=False):
    for _ in range(steps):
        if ctrl is not None:
            ctrl.apply_prestress(control, s0 if state_arg else None)
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        if after:
            ctrl.after_step(s0, control, dt)
    return s0, s1
