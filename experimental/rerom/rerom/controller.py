# ruff: noqa: TID253
"""Re-linearized ROM (re-ROM) controller for Newton reduced elastic bodies (host-side, no kernel changes).

:class:`ReROMController` re-linearizes ONE reduced elastic body of a live ``newton.Model`` + ``SolverVBD``
in place. It keeps the full-field record outside the Model (rest nodes ``r0``, current reference ``r_ref``,
nodal basis ``Phi`` (3N, m) in the body frame, consistent mass ``M``, a :class:`~rerom.TangentAssembler`).
When its trigger fires it evaluates ``W, f_int, K_T`` at ``r_new = r_ref + Phi q`` (total-Lagrangian w.r.t.
``r0``), lets the :class:`BodyAdapter` build the new basis and project the state, rebuilds every ROM array
of the body (modal mass/stiffness/damping, the five inertia integrals, modal mass matrix, rigid
mass/COM/inertia, endpoint/render/shape samples and their body-frame rest positions, joint endpoint
frames), writes them in place (:mod:`rerom.model_io`), resets ``q = 0`` with a momentum-consistent velocity
and supplies the pre-stress ``g0 = Phi^T f_int(r_new)`` through the modal rows of ``control.joint_f``
(``joint_f = modal_force - g0``: Newton's modal row is ``k q - joint_f``, so equilibrium is ``q = joint_f/k``).

Body adapters (strategy):

- :class:`FreeBodyAdapter`: free-floating or node-clamped bodies (``ModalGeneratorFEM`` bases): restricted
  tangent eigenmodes on the M-orthogonal complement of the rigid motions of ``r_new`` (via the eigensolver
  seam, :mod:`rerom.eigen`), static Ritz / Krylov vectors, optional modal derivatives, Rayleigh-Ritz; the
  nodal velocity splits M-orthogonally into the rigid motion of ``r_new`` (frame twist about the new rest
  COM) and the new modal span. Clamped bases keep the frame twist.
- :class:`InterfaceBodyAdapter`: Craig-Bampton bodies with an interface frame (root face = body frame, tip
  face = RBE2 frame, :mod:`rerom.craig_bampton`): constraint modes from ``K_T`` with interfaces held,
  fixed-interface tangent modes, interface-held Ritz vectors; the frame twist is kept and the modal velocity
  is projected so that the tip-frame velocity is preserved exactly. Optional endpoint offset (yoke pivots).

:class:`MultiFrameController` aggregates one interface controller per CB body of a chain.

Conventions (Newton): ``joint_q = [p(3), quat xyzw(4), q(m)]``, ``joint_qd = [v_c(3), omega(3), qd(m)]``
(world velocity of the rest COM, world angular velocity); ``x_world = p + R (r_ref + Phi q)``; nodal layout
``x_flat[3 n + i]``. The pose origin ``p`` is never moved by an update; ``body_com`` changes.
"""

from __future__ import annotations

import enum
import math
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sp

import newton
from newton._src.sim.modal import _estimate_sample_psi, _reduced_inertia_from_pairs

from . import model_io as io
from .craig_bampton import cb_basis
from .eigen import (
    RestrictedArpackBackend,
    RestrictedSubspaceBackend,
    ShiftedRestrictedSolver,
    resolve_backend,
    shared_restricted_solver,
    solve_restricted_eigen,
)
from .hyper_reduction import ReducedElementForce
from .numerics import (
    MOrthonormalBlock,
    compare_eigen,
    constrain_frame_velocity,
    fd_step,
    host_sparse,
    m_orthonormal_rigid,
    m_orthonormalize,
    nodal_rigid_vectors,
    project_modal_velocity,
    quat_from_rotvec,
    quat_to_matrix,
    rayleigh_ritz,
    rigid_properties,
    ritz_enrichment,
    rotvec_of_matrix,
    split_rigid_velocity,
    static_modal_derivatives,
    tangent_derivative,
    to_nodal,
)
from .options import (
    BasisOptions,
    IndefiniteGuard,
    NonlinearForceOptions,
    RestRotationMode,
    TriggerOptions,
    UpdatePath,
    UpdatePolicy,
)
from .triggers import RotationTrigger
from .update_kernels import DeviceUpdater

__all__ = [
    "BodyAdapter",
    "FreeBodyAdapter",
    "InterfaceBodyAdapter",
    "MultiFrameController",
    "ReROMController",
    "UpdateSync",
]


# ----------------------------------------------------------------------------------------------
# body adapters
# ----------------------------------------------------------------------------------------------


class BodyAdapter:
    """Strategy for the body-type specific part of an update.

    An adapter supplies the initial nodal basis and rest nodes, the clamped nodes, and implements
    :meth:`rebuild` (new basis, endpoint samples and projected state about ``r_new``). Everything else
    (bookkeeping, triggers, Model writes, pre-stress, IMEX remainder) is shared by the controller.
    """

    full_modes: np.ndarray
    rest_nodes: np.ndarray
    fixed_nodes: np.ndarray | None = None
    base_sample_count: int | None = None
    damping_ratio: float | None = None
    derive_psi: bool | None = None

    def attach(self, ctrl: ReROMController) -> None:
        """Called once by the controller after its bookkeeping is set up."""

    def rebuild(self, ctrl: ReROMController, ctx: dict) -> dict:
        """Return the new basis and state for the update context ``ctx`` (see :meth:`ReROMController.relinearize`)."""
        raise NotImplementedError


class FreeBodyAdapter(BodyAdapter):
    """Free-floating or node-clamped body with a nodal basis (e.g. ``ModalGeneratorFEM.modal_matrix``).

    Args:
        full_modes: Nodal basis the body was built with, (3N, m).
        rest_nodes: Original rest nodes r0 (N, 3) in the body frame (total-Lagrangian reference).
        fixed_nodes: Nodes clamped in the body frame (constrained basis, e.g. ``fixed_node_indices``): no rigid
            motion is deflated or transferred to the frame; the frame twist is kept and only the modal
            velocity is projected.
        base_sample_count: Basis samples produced by the generator before the builder appended
            endpoint/render/shape samples (default: N when the first N samples are the rest nodes).
        external_force_fn: ``f(ctrl, state) -> (3N,)`` body-frame external nodal force for the load Ritz
            vector; ``None`` uses gravity ``M tile(R^T g)``.
    """

    def __init__(
        self,
        full_modes: np.ndarray,
        rest_nodes: np.ndarray,
        *,
        fixed_nodes: np.ndarray | None = None,
        base_sample_count: int | None = None,
        external_force_fn: Callable[[ReROMController, Any], np.ndarray] | None = None,
    ):
        self.full_modes = np.asarray(full_modes, dtype=np.float64)
        self.rest_nodes = np.asarray(rest_nodes, dtype=np.float64)
        self.fixed_nodes = fixed_nodes
        self.base_sample_count = base_sample_count
        self.external_force_fn = external_force_fn

    def rebuild(self, ctrl, ctx):
        o = ctrl.basis_options
        n, N, m = ctrl.n_dof, ctrl.N, ctrl.m
        M = ctrl.M
        K_T, f_int, x_new = ctx["K_T"], ctx["f_int"], ctx["x_new"]
        timings = ctx["timings"]
        debug = ctx["debug"]
        t1 = time.perf_counter()

        # ---- restricted eigenproblem ------------------------------------------------------------
        mass_new, com_new, I_new = rigid_properties(M, x_new)
        Yraw = nodal_rigid_vectors(x_new, com_new)
        free = ctrl.free_dofs
        Kf = K_T[free][:, free]
        Mf = M[free][:, free]
        if ctrl.constrained:
            Yf = None
            k_eig = min(m + o.eig_margin, free.size)
        else:
            Yf = m_orthonormal_rigid(Yraw, M)
            k_eig = min(m + o.eig_margin, n - 6)
        do_rebasis = o.rebasis if ctx["rebasis"] is None else bool(ctx["rebasis"])
        shared: dict[str, Any] = {}
        factor = {"t": 0.0}

        def get_solver():
            # ONE restricted factorization shared by the subspace iteration, Ritz vectors and modal derivatives
            if "f" not in shared:
                # the symmetric factorization (with inertia count) is needed only to share it with ARPACK;
                # the subspace iteration always shared the pivoting LU
                shared["f"] = (
                    shared_restricted_solver(Kf, Mf, Yf)
                    if o.shared_factorization and isinstance(ctrl.eigen_backend, RestrictedArpackBackend)
                    else ShiftedRestrictedSolver(Kf, Mf, Yf)
                )
                factor["t"] += shared["f"].factor_time
            return shared["f"]

        backend = ctrl.eigen_backend
        t_cmp = 0.0
        eig_compare = None
        t_e0 = time.perf_counter()
        f_before_eig = factor["t"]
        if do_rebasis:
            previous = ctrl.Phi[free]
            seed = 1234 + ctrl.step_count
            vals, Vf, eig_info = solve_restricted_eigen(
                backend,
                Kf,
                Mf,
                Yf,
                k_eig,
                previous=previous,
                shared_solver=get_solver,
                seed=seed,
                share_arpack=o.shared_factorization,
            )
            f_in_eig = factor["t"] - f_before_eig
            if o.eigen_compare:
                tc = time.perf_counter()
                f_before = factor["t"]
                n_cmp = m - o.n_ritz - o.modal_derivatives * (o.modal_derivatives + 1) // 2
                if isinstance(backend, RestrictedSubspaceBackend):
                    ref = RestrictedArpackBackend(dense_dof_limit=o.dense_dof_limit)
                    vals_o, Vf_o, _ = solve_restricted_eigen(
                        ref, Kf, Mf, Yf, k_eig, previous=previous, shared_solver=get_solver, seed=seed
                    )
                    eig_compare = compare_eigen(vals_o, Vf_o, vals, Vf, Mf, n_cmp)
                else:
                    alt = RestrictedSubspaceBackend(iterations=o.subspace_iterations, margin=o.eig_margin)
                    vals_o, Vf_o, _ = solve_restricted_eigen(
                        alt, Kf, Mf, Yf, k_eig, previous=previous, shared_solver=get_solver, seed=seed
                    )
                    eig_compare = compare_eigen(vals, Vf, vals_o, Vf_o, Mf, n_cmp)
                t_cmp = time.perf_counter() - tc
                # a factorization made for the comparison is reused later; keep its time in the split
                t_cmp -= factor["t"] - f_before
            V = np.zeros((n, Vf.shape[1]))
            V[free] = Vf
        else:
            # plain re-referencing: old basis, rigid motions of r_new removed, M-orthonormalized
            V = ctrl.Phi.copy()
            if Yf is not None:
                V = V - Yf @ (Yf.T @ (M @ V))
            V = m_orthonormalize(V, M, 1.0e-12)
            Kv = V.T @ (K_T @ V)
            vals, Qk = np.linalg.eigh(0.5 * (Kv + Kv.T))
            V = V @ Qk
            eig_info = {"path": "rebasis_off", "t_factor": 0.0}
            f_in_eig = 0.0
        t_e1 = time.perf_counter()

        # ---- Ritz / modal-derivative augmentation -------------------------------------------------
        ritz_kept = 0
        md_kept = 0
        ritz_info: list = []
        basis = V[:, :m]
        kmd = int(o.modal_derivatives)
        n_md_target = kmd * (kmd + 1) // 2
        n_extra_target = n_md_target + o.n_ritz
        if n_extra_target > 0:
            if n_extra_target >= m:
                raise ValueError(f"n_ritz + k(k+1)/2 modal derivatives = {n_extra_target} must be < mode_count {m}")
            solve_f = get_solver()

            def solve_r(b):
                x = np.zeros(n)
                x[free] = solve_f(np.asarray(b)[free])
                return x

            n_eig = m - n_extra_target
            block = MOrthonormalBlock(M, V[:, :n_eig].copy(), drop_tol=o.ritz_drop_tol)
            if kmd > 0:
                t_md = time.perf_counter()
                x0 = x_new.reshape(-1)
                for i in range(kmd):
                    phi_i = V[:, i]
                    dK = tangent_derivative(ctrl.assembler, x0, phi_i, fd_step(phi_i, o.md_fd_scale, ctrl.char_length))
                    for j in range(i, kmd):
                        if md_kept >= n_md_target:
                            break
                        theta = -solve_r(dK @ V[:, j])
                        if block.accept(f"md_{i}{j}", theta) is not None:
                            md_kept += 1
                timings["modal_derivatives"] = time.perf_counter() - t_md
            cands = [("f_int", f_int)]
            if self.external_force_fn is not None:
                f_ext = np.asarray(self.external_force_fn(ctrl, ctx["state"]), dtype=np.float64)
            else:
                g = ctx["arrays"].get("gravity")[max(ctrl.layout.world, 0)].astype(np.float64)
                f_ext = M @ np.tile(ctx["R"].T @ g, N)
            cands.append(("f_ext", f_ext))
            cands.append(("M v", M @ (ctrl.Phi @ ctx["qd"])))
            if ctrl._qdd_est is not None:
                cands.append(("M a", M @ (ctrl.Phi @ ctrl._qdd_est)))
            ritz_vecs = ritz_enrichment(block, cands, solve_r, o.n_ritz, krylov=o.ritz_krylov, M=M)
            ritz_kept = len(ritz_vecs)
            block.fill(V, n_eig, m)
            ritz_info = block.log
            basis = block.columns[:, :m]
            if basis.shape[1] < m:
                raise RuntimeError("could not assemble m independent basis vectors")
        _, Phi_new, _ = rayleigh_ritz(basis, K_T, M)
        if debug.get("keep_basis"):
            Phi_new = ctrl.Phi.copy()
        mass_modes = M @ Phi_new
        mode_mass_matrix = Phi_new.T @ mass_modes
        mode_mass = np.diag(mode_mass_matrix).copy()
        lam = np.diag(Phi_new.T @ (K_T @ Phi_new)) / mode_mass
        t2 = time.perf_counter()
        f_outside = factor["t"] - f_in_eig
        timings["eigen"] = t2 - t1 - t_cmp
        timings["factorization"] = float(eig_info.get("t_factor", 0.0)) + factor["t"]
        timings["eigensolve"] = (t_e1 - t_e0) - float(eig_info.get("t_factor", 0.0)) - f_in_eig - t_cmp
        timings["projection"] = (t2 - t1) - (t_e1 - t_e0) - f_outside - timings.get("modal_derivatives", 0.0)
        if t_cmp:
            timings["eigen_compare_excluded"] = t_cmp

        # ---- samples -----------------------------------------------------------------------------
        nodal_phi = to_nodal(Phi_new, N)
        nodal_psi = _estimate_sample_psi(x_new.astype(np.float32), nodal_phi) if ctrl.derive_psi else None
        lay = ctrl.layout
        ep_idx = np.array([ep.sample for ep in lay.endpoints], dtype=np.int64)
        ep_phi = ctrl.map_samples(ep_idx, nodal_phi) if ep_idx.size else np.zeros((0, m, 3))
        ep_psi = (
            ctrl.map_samples(ep_idx, nodal_psi) if (ep_idx.size and nodal_psi is not None) else np.zeros_like(ep_phi)
        )
        if debug.get("keep_endpoint_psi") and ep_idx.size:
            arr = ctx["arrays"].get("elastic_endpoint_psi")
            ep_psi = np.stack([arr[ep.endpoint * lay.Mx + np.arange(m)] for ep in lay.endpoints])
        damping = 2.0 * ctrl.damping_ratio * np.sqrt(np.maximum(mode_mass * lam * mode_mass, 0.0))

        # ---- state projection: rigid part -> frame twist (about com_new), remainder -> qd ----------
        if ctx.get("device_projection"):
            # device path (rerom.update_kernels): only the rigid Gram matrix about com_new is formed here
            dynamic = not (ctrl.constrained or not lay.frame_dynamic)
            return {
                "Phi": Phi_new,
                "mass_modes": mass_modes,
                "mode_mass_matrix": mode_mass_matrix,
                "mode_mass": mode_mass,
                "lam": lam,
                "damping": damping,
                "rigid": (mass_new, com_new, I_new),
                "ep_phi": ep_phi,
                "ep_psi": ep_psi,
                "map_shapes": True,
                "projection": {"mode": 0 if dynamic else 1, "G": (Yraw.T @ (M @ Yraw)) if dynamic else None},
                "v_c": None,
                "omega": None,
                "qd": None,
                "u_el_old": None,
                "t_samples": t2,
                "solver": get_solver,
                "info": {
                    "eigen_raw": np.asarray(vals).tolist(),
                    "negative_eigenvalues": int(np.count_nonzero(np.asarray(vals) < 0.0)),
                    "eigen_info": {k: v for k, v in eig_info.items() if k != "sigma_tries"},
                    "ritz": ritz_info,
                    "ritz_kept": ritz_kept,
                    "md_kept": md_kept,
                    **({"eigen_compare": eig_compare} if eig_compare is not None else {}),
                },
            }
        R, u_local, qd = ctx["R"], ctx["u_local"], ctx["qd"]
        if ctrl.constrained or not lay.frame_dynamic:
            omega_l = R.T @ ctx["omega"]
            t_vec = R.T @ ctx["v_c"] + np.cross(omega_l, com_new - ctrl.com)
            w_vec = omega_l
            u_rig = u_local - ctrl.Phi @ qd
        else:
            coeff, u_rig = split_rigid_velocity(u_local, Yraw, M)
            t_vec, w_vec = coeff[:3], coeff[3:]
        u_el_old = u_local - u_rig
        qd_new = project_modal_velocity(u_el_old, Phi_new, M, mode_mass_matrix)
        return {
            "Phi": Phi_new,
            "mass_modes": mass_modes,
            "mode_mass_matrix": mode_mass_matrix,
            "mode_mass": mode_mass,
            "lam": lam,
            "damping": damping,
            "rigid": (mass_new, com_new, I_new),
            "ep_phi": ep_phi,
            "ep_psi": ep_psi,
            "map_shapes": True,
            "v_c": R @ t_vec,
            "omega": R @ w_vec,
            "qd": qd_new,
            "u_el_old": u_el_old,
            "t_samples": t2,
            "solver": get_solver,
            "info": {
                "eigen_raw": np.asarray(vals).tolist(),
                "negative_eigenvalues": int(np.count_nonzero(np.asarray(vals) < 0.0)),
                "eigen_info": {k: v for k, v in eig_info.items() if k != "sigma_tries"},
                "ritz": ritz_info,
                "ritz_kept": ritz_kept,
                "md_kept": md_kept,
                **({"eigen_compare": eig_compare} if eig_compare is not None else {}),
            },
        }


class InterfaceBodyAdapter(BodyAdapter):
    """Craig-Bampton body with an interface frame: root face = body frame, optional RBE2 tip frame.

    Args:
        segment: :class:`rerom.craig_bampton.CBSegment` the body was built from (rest nodes, root/tip node
            sets, tip point, basis ``Phi`` and frame map ``F``).
        gravity: World gravity vector for the ``f_ext`` Ritz load (``None``: the model gravity).
        keep_velocity: Add the old elastic velocity (minus its tip-frame part) as an interface-held vector.
        endpoint_offset: Joint endpoints sit at ``tip_point + d`` / ``d`` (e.g. yoke pivots) instead of the
            interface frames; the offset follows the tip frame linearly at every update. With an offset
            exactly one tip endpoint must exist.
        endpoint_tol: Distance tolerance [m] of the tip-endpoint detection.
    """

    damping_ratio = 0.0
    derive_psi = False

    def __init__(self, segment, *, gravity=None, keep_velocity: bool = False, endpoint_offset=None, endpoint_tol=None):
        self.seg = segment
        self.full_modes = np.asarray(segment.Phi, dtype=np.float64)
        self.rest_nodes = np.asarray(segment.local_nodes, dtype=np.float64)
        self.fixed_nodes = segment.root_ids
        self.gravity = None if gravity is None else np.asarray(gravity, dtype=np.float64)
        self.keep_velocity = bool(keep_velocity)
        self.offset = None if endpoint_offset is None else np.asarray(endpoint_offset, dtype=np.float64)
        self.endpoint_tol = float(
            endpoint_tol if endpoint_tol is not None else (1e-5 if self.offset is not None else 1e-6)
        )
        self.frame_map = None if segment.frame_map is None else segment.frame_map.copy()
        self.tip_pos = None if segment.tip_point is None else segment.tip_point.astype(np.float64).copy()
        self.ep_is_tip: list[bool] = []

    def attach(self, ctrl):
        seg = self.seg
        d = np.zeros(3) if self.offset is None else self.offset
        self.ep_is_tip = []
        for ep in ctrl.layout.endpoints:
            pt = np.asarray(ctrl.basis.sample_points[ep.sample], dtype=np.float64)
            self.ep_is_tip.append(bool(seg.has_tip and np.linalg.norm(pt - (seg.tip_point + d)) < self.endpoint_tol))
        if self.offset is not None and sum(self.ep_is_tip) != 1:
            raise ValueError(f"expected one tip endpoint at the offset, found {self.ep_is_tip}")

    def tip_frame_body(self, q):
        """Body-frame tip-frame position and rotation vector (linear, as the Newton endpoint) at ``q``."""
        if self.frame_map is None:
            return None, None
        f = self.frame_map @ q
        return self.tip_pos + f[:3], f[3:]

    def rebuild(self, ctrl, ctx):
        o = ctrl.basis_options
        N, m = ctrl.N, ctrl.m
        M = ctrl.M
        q, qd, R = ctx["q"], ctx["qd"], ctx["R"]
        K_T, f_int, x_new = ctx["K_T"], ctx["f_int"], ctx["x_new"]
        timings = ctx["timings"]
        t1 = time.perf_counter()
        u_el_old = ctrl.Phi @ qd
        fdot = self.frame_map @ qd if self.frame_map is not None else None
        tip_pos_new = None
        if self.frame_map is not None:
            tip_pos_new = self.tip_pos + self.frame_map[:3] @ q
        if self.gravity is not None:
            g = self.gravity
        else:
            g = ctx["arrays"].get("gravity")[max(ctrl.layout.world, 0)].astype(np.float64)
        f_ext = M @ np.tile(R.T @ g, N)
        loads = [("f_int", f_int), ("f_ext", f_ext)][: o.n_ritz]
        extra = [("velocity", u_el_old.copy())] if self.keep_velocity else []
        backend = ctrl.eigen_backend if ctrl.eigensolver_explicit_backend else None
        Phi_new, lam, F_new, binfo = cb_basis(
            x_new,
            K_T,
            M,
            self.seg.root_ids,
            self.seg.tip_ids,
            tip_pos_new,
            m,
            ritz_loads=loads,
            extra_vectors=extra,
            eig_margin=o.eig_margin,
            drop_tol=o.ritz_drop_tol,
            eigensolver=backend,
        )
        t2 = time.perf_counter()
        timings["factorization"] = binfo["t_factor"]
        timings["eigensolve"] = binfo["t_eig"]
        timings["ritz"] = binfo["t_ritz"]
        timings["projection"] = binfo["t_rayleigh"]
        timings["basis"] = t2 - t1

        mass_modes = M @ Phi_new
        mode_mass_matrix = Phi_new.T @ mass_modes
        mode_mass = np.diag(mode_mass_matrix).copy()
        # state projection: frame twist kept, interface velocities preserved
        qd_new = project_modal_velocity(u_el_old, Phi_new, M, mode_mass_matrix)
        if F_new is not None:
            qd_new = constrain_frame_velocity(qd_new, F_new, fdot)
        mass_new, com_new, I_new = rigid_properties(M, x_new)
        omega_l = R.T @ ctx["omega"]
        t_vec = R.T @ ctx["v_c"] + np.cross(omega_l, com_new - ctrl.com)

        lay = ctrl.layout
        ep_phi = np.zeros((len(lay.endpoints), m, 3))
        ep_psi = np.zeros((len(lay.endpoints), m, 3))
        for i, is_tip in enumerate(self.ep_is_tip):
            if is_tip:
                ep_phi[i] = F_new[:3].T
                ep_psi[i] = F_new[3:].T
        if self.offset is not None:
            # the offset follows the tip frame (linearised like Newton's endpoint update at this q)
            self.offset = self.offset + np.cross(self.frame_map[3:] @ q, self.offset)
            for i, is_tip in enumerate(self.ep_is_tip):
                if is_tip:
                    ep_phi[i] = ep_phi[i] + np.cross(ep_psi[i], self.offset)
        if F_new is not None:
            self.tip_pos = tip_pos_new
            self.frame_map = F_new
        fdot_after = (F_new @ qd_new) if F_new is not None else None
        return {
            "Phi": Phi_new,
            "mass_modes": mass_modes,
            "mode_mass_matrix": mode_mass_matrix,
            "mode_mass": mode_mass,
            "lam": lam,
            "damping": np.zeros(m),
            "rigid": (mass_new, com_new, I_new),
            "ep_phi": ep_phi,
            "ep_psi": ep_psi,
            "map_shapes": True,
            "projection": {"mode": 2, "G": None},
            "v_c": R @ t_vec,
            "omega": ctx["omega"],
            "qd": qd_new,
            "u_el_old": u_el_old,
            "t_samples": t2,
            "solver": None,
            "info": {
                "segment": self.seg.index,
                "negative_eigenvalues": int(np.count_nonzero(lam < 0)),
                "held_names": binfo.get("held_names"),
                "frame_map_residual": binfo.get("frame_map_residual"),
                "tip_frame_velocity_change": None if fdot is None else float(np.max(np.abs(fdot_after - fdot))),
            },
        }


class _GatheredArrays:
    """``ModelArrays``-like read access for the adapters on the device path: ``gravity`` comes from the
    update's packed gather; any other array falls back to a host gather (diagnostic switches only)."""

    def __init__(self, model, gravity: np.ndarray, world: int):
        self._gravity = np.zeros((max(world, 0) + 1, 3))
        self._gravity[max(world, 0)] = gravity
        self._fallback = io.ModelArrays(model)

    def get(self, name: str) -> np.ndarray:
        if name == "gravity":
            return self._gravity
        return self._fallback.get(name)


# ----------------------------------------------------------------------------------------------
# controller
# ----------------------------------------------------------------------------------------------


class ReROMController:
    """Host-side re-linearization of one reduced elastic body (see the module docstring).

    Args:
        model: Live ``newton.Model``.
        solver: ``SolverVBD`` stepping the model.
        body_index: Newton body index of the elastic body.
        assembler: A :class:`~rerom.TangentAssembler` on the body's rest mesh (nodes in the body frame).
        adapter: :class:`FreeBodyAdapter` or :class:`InterfaceBodyAdapter`.
        mode_count: Number of modes (must equal the body's Newton mode count; default: basis columns).
        basis: :class:`~rerom.BasisOptions`.
        trigger: A :class:`~rerom.Trigger` (default: :class:`~rerom.RotationTrigger`, 4 deg).
        policy: :class:`~rerom.UpdatePolicy`.
        eigensolver: ``"arpack"`` (default), ``"subspace"``, ``"dense"``, ``"auto"``, a
            ``newton.ModalEigensolver`` or a ``newton.EigenSolverBackend`` instance (see :mod:`rerom.eigen`).
        nonlinear_force: :class:`~rerom.NonlinearForceOptions` to enable the IMEX hyper-reduced remainder
            (free-body adapter only).
    """

    def __init__(
        self,
        model,
        solver,
        body_index: int,
        assembler,
        adapter: BodyAdapter,
        *,
        mode_count: int | None = None,
        basis: BasisOptions | None = None,
        trigger=None,
        policy: UpdatePolicy | None = None,
        eigensolver: str | newton.ModalEigensolver | newton.EigenSolverBackend = "arpack",
        nonlinear_force: NonlinearForceOptions | None = None,
    ):
        self.model = model
        self.solver = solver
        self.assembler = assembler
        self.adapter = adapter
        self.basis_options = basis if basis is not None else BasisOptions()
        self.policy = policy if policy is not None else UpdatePolicy()
        self.trigger = trigger if trigger is not None else RotationTrigger()
        o = self.basis_options

        self.r0 = np.asarray(adapter.rest_nodes, dtype=np.float64).reshape(-1, 3).copy()
        self.N = int(self.r0.shape[0])
        self.n_dof = 3 * self.N
        Phi = np.asarray(adapter.full_modes, dtype=np.float64)
        if Phi.shape[0] != self.n_dof:
            raise ValueError(f"full_modes must have {self.n_dof} rows, got {Phi.shape}")
        self.m = int(Phi.shape[1] if mode_count is None else mode_count)
        self.Phi = Phi[:, : self.m].copy()
        self.r_ref = self.r0.copy()
        self.M = sp.csr_matrix(assembler.mass_matrix)
        self.char_length = float(np.max(np.ptp(self.r0, axis=0)))
        fixed = np.zeros(self.n_dof, dtype=bool)
        if adapter.fixed_nodes is not None:
            fn = np.asarray(adapter.fixed_nodes, dtype=np.int64).reshape(-1)
            fixed[(3 * fn[:, None] + np.arange(3)[None, :]).reshape(-1)] = True
        self.constrained = bool(np.any(fixed))
        self.free_dofs = np.nonzero(~fixed)[0]

        # eigensolver seam
        self.eigensolver_explicit_backend = not isinstance(eigensolver, (str, newton.ModalEigensolver))
        self.eigen_backend = resolve_backend(
            eigensolver, margin=o.eig_margin, iterations=o.subspace_iterations, dense_dof_limit=o.dense_dof_limit
        )

        # Newton bookkeeping
        self.layout = io.BodyLayout.from_model(model, body_index, self.m)
        lay = self.layout
        self.body = lay.body
        self.q_start, self.qd_start = lay.q_start, lay.qd_start
        self.basis = model.modal_bases[lay.basis_index]
        if self.basis.frame_count > 0:
            for ep in lay.endpoints:
                if self.basis.find_frame(self.basis.sample_points[ep.sample]) >= 0:
                    raise NotImplementedError(
                        "joint endpoint uses a prescribed ModalBasis attachment frame; cannot re-derive it"
                    )

        # sample -> node interpolation weights (replays the builder's add_samples batches)
        self.sample_weights, self._sample_count_mapped = io.sample_weight_replay(
            model, lay, self.basis, self.r0, adapter.base_sample_count
        )
        used = np.unique(
            np.concatenate(
                [np.array([ep.sample for ep in lay.endpoints], dtype=np.int64), lay.render_samples, lay.shape_samples]
            )
        )
        used = used[used >= 0]
        ref = self.basis.sample_phi[used].astype(np.float64)
        scale = max(float(np.max(np.abs(ref))) if ref.size else 0.0, 1.0e-30)
        recon = self.map_samples(used, to_nodal(self.Phi, self.N))
        self.sample_map_error = float(np.max(np.abs(recon - ref))) / scale if used.size else 0.0
        if self.sample_map_error > 1.0e-4:
            warnings.warn(
                f"sample->node interpolation replay reproduces the builder samples only to "
                f"{self.sample_map_error:.2e} (relative); render/contact/endpoint samples will be approximate",
                stacklevel=2,
            )
        derive_psi = adapter.derive_psi if adapter.derive_psi is not None else o.derive_psi
        if derive_psi is None:
            derive_psi = bool(np.any(self.basis.sample_psi != 0.0))
        self.derive_psi = bool(derive_psi)

        # damping policy
        mm, kk, cc = io.read_modal_diagonals(model, lay)
        damping_ratio = adapter.damping_ratio if adapter.damping_ratio is not None else o.damping_ratio
        if damping_ratio is None:
            ok = (kk > 0) & (mm > 0)
            ratios = cc[ok] / (2.0 * np.sqrt(mm[ok] * kk[ok])) if np.any(ok) else np.zeros(1)
            damping_ratio = float(np.mean(ratios)) if ratios.size else 0.0
            self.damping_ratio_spread = float(np.ptp(ratios)) if ratios.size else 0.0
        else:
            self.damping_ratio_spread = 0.0
        self.damping_ratio = float(damping_ratio)

        # mass offsets (armature etc.) between the model and the mass matrix
        m0, _, I0 = rigid_properties(self.M, self.r0)
        body_mass, body_inertia, body_com = io.read_body_inertial(model, self.body)
        self.rigid_mass0 = m0
        self.mass_offset = body_mass - m0
        self.inertia_offset = body_inertia - I0
        self.com = body_com

        # ROM energy bookkeeping / pre-stress
        self.g0 = np.zeros(self.m)
        self.lam = kk / np.maximum(mm, 1.0e-30)
        self.W_ref = 0.0
        self._dev: DeviceUpdater | None = None
        self.modal_force = np.zeros(self.m)  # user modal joint_f added to -g0 by apply_prestress
        self.stats: list[dict] = []
        self.steps_since_update = 0
        self.step_count = 0
        self.last_dt: float | None = None
        self.last_joint_rest_mismatch = 0.0
        self.clamp_log: list[dict] = []
        self._qd_hist: tuple[np.ndarray, float] | None = None
        self._qdd_est: np.ndarray | None = None
        self._view: tuple[Any, np.ndarray, np.ndarray] | None = None
        self._released = False
        self._in_contact = False
        self._update_requested = False

        # IMEX hyper-reduced remainder, folding
        self.nl_options = nonlinear_force
        self.nl_model: ReducedElementForce | None = None
        self.folds: list[dict] = []
        self.steps_since_fold = 0
        self.step_log: list[dict] = []
        self._fnl = np.zeros(self.m)
        self._fnl_step = -1
        self._nl_last: dict = {}
        self._Theta = None
        self.nl_manifold = 0
        self.nl_interval: int | None = None
        self.fold_interval: int | None = None
        if nonlinear_force is not None:
            if not isinstance(adapter, FreeBodyAdapter):
                raise NotImplementedError("nonlinear_force is implemented for FreeBodyAdapter only")
            if not hasattr(assembler, "element_terms"):
                raise TypeError("nonlinear_force needs an assembler with element_terms()")
            nf = nonlinear_force
            self.nl_interval = None if nf.interval is None else max(int(nf.interval), 1)
            self.fold_interval = None if nf.fold_interval is None else max(int(nf.fold_interval), 1)
            self.nl_model = ReducedElementForce(assembler, nf.element_ids, nf.weights)
            self.nl_manifold = int(nf.manifold)
            if self.nl_manifold > 0:
                _, _, K0_ = assembler.evaluate(self.r_ref.reshape(-1), need_tangent=True)
                K0_ = host_sparse(K0_)
                K0_ = 0.5 * (K0_ + K0_.T)
                free = self.free_dofs
                Yf0 = None
                if not self.constrained:
                    _, c0_, _ = rigid_properties(self.M, self.r_ref)
                    Yf0 = m_orthonormal_rigid(nodal_rigid_vectors(self.r_ref, c0_), self.M)
                solve_f0 = ShiftedRestrictedSolver(K0_[free][:, free], self.M[free][:, free], Yf0)
                self._Theta = self._static_mds(self.r_ref.reshape(-1), self.Phi, solve_f0)
            self._init_nl_model()

        # rest rotations
        self.rest_propagator: io.JointRestPropagator | None = None
        if self.policy.rest_rotation is RestRotationMode.JOINT_TREE:
            self.rest_propagator = io.JointRestPropagator(model, self.policy.rest_root)
        elif self.policy.rest_rotation is RestRotationMode.REST_FRAMES:
            if not hasattr(solver, "set_joint_rest_frames"):
                raise RuntimeError(
                    "RestRotationMode.REST_FRAMES needs SolverVBD.set_joint_rest_frames (per-joint rest-frame patch)"
                )
            solver.set_joint_rest_frames(joint_X_p_rest=model.joint_X_p, joint_X_c_rest=model.joint_X_c)
        adapter.attach(self)

        # device-side update (rerom.update_kernels)
        self._dev_hist_ready = False
        self._dev_qdd_valid = False
        self.last_refresh_in_place: bool | None = None
        if self.policy.path is UpdatePath.DEVICE:
            self._dev = DeviceUpdater(self)
            self._dev.set_prestress(self.modal_joint_force())

    # ------------------------------------------------------------------------------------------
    # properties / phase hooks
    # ------------------------------------------------------------------------------------------
    @property
    def update_count(self) -> int:
        """Number of updates made so far."""
        return len(self.stats)

    @property
    def device_updater(self) -> DeviceUpdater | None:
        """The :class:`rerom.update_kernels.DeviceUpdater` (``None`` on the host path)."""
        return self._dev

    @property
    def endpoints(self):
        """Joint endpoints on the body (:class:`rerom.model_io.EndpointRef`)."""
        return self.layout.endpoints

    def set_phase(self, *, released: bool | None = None, in_contact: bool | None = None) -> None:
        """Inform the update policy about the scene phase (freeze hooks of :class:`~rerom.UpdatePolicy`)."""
        if released is not None:
            self._released = bool(released)
        if in_contact is not None:
            self._in_contact = bool(in_contact)

    def request_update(self) -> None:
        """Force an update at the next :meth:`poll` (reason ``"requested"``), bypassing the trigger."""
        self._update_requested = True

    def frozen(self) -> bool:
        """Whether the update policy currently forbids updates."""
        p = self.policy
        if p.freeze_after_release and self._released:
            return True
        if p.freeze_in_contact and self._in_contact:
            return True
        return p.freeze_from_step is not None and self.step_count >= p.freeze_from_step

    def effective_min_interval(self, trigger_min_interval: int) -> int:
        """The trigger's minimum interval, or the policy's post-release one once released."""
        p = self.policy
        if self._released and p.post_release_min_interval is not None:
            return int(p.post_release_min_interval)
        return int(trigger_min_interval)

    # ------------------------------------------------------------------------------------------
    # state access (one gather per controller call)
    # ------------------------------------------------------------------------------------------
    def _state_arrays(self, state) -> tuple[np.ndarray, np.ndarray]:
        if self._view is not None and self._view[0] is state:
            if self._view[1] is None:  # lazy bind: gather on first use
                jq, jqd = io.read_joint_state(state)
                self._view = (state, jq, jqd)
            return self._view[1], self._view[2]
        return io.read_joint_state(state)

    def _bind(self, state, view=None) -> None:
        """Cache ``(joint_q, joint_qd)`` of ``state`` for the duration of a controller call.

        On the device path the gather is deferred to the first host read (none for device triggers).
        """
        if view is None and self._dev is not None:
            self._view = (state, None, None)
            return
        jq, jqd = view if view is not None else io.read_joint_state(state)
        self._view = (state, jq, jqd)

    def _unbind(self) -> None:
        self._view = None

    def read_modes(self, state) -> tuple[np.ndarray, np.ndarray]:
        """Modal coordinates and velocities ``(q, qd)`` of the body (float64)."""
        jq, jqd = self._state_arrays(state)
        q = jq[self.q_start + 7 : self.q_start + 7 + self.m].astype(np.float64)
        qd = jqd[self.qd_start + 6 : self.qd_start + 6 + self.m].astype(np.float64)
        return q, qd

    def read_frame(self, state) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(p, quat xyzw, v_c world, omega world)``."""
        jq, jqd = self._state_arrays(state)
        a = jq[self.q_start : self.q_start + 7].astype(np.float64)
        b = jqd[self.qd_start : self.qd_start + 6].astype(np.float64)
        return a[:3], a[3:7], b[:3], b[3:6]

    def full_field(self, state, manifold: bool = False) -> np.ndarray:
        """Body-frame node positions ``r_ref + Phi q``, (N, 3) (what Newton displays/attaches).

        ``manifold=True`` (with the IMEX quadratic manifold): ``+ 1/2 Theta:(p, p)``.
        """
        q, _ = self.read_modes(state)
        x = self.r_ref + (self.Phi @ q).reshape(-1, 3)
        if manifold and self.nl_model is not None and self.nl_model.S is not None and self._Theta is not None:
            p = self.nl_model.S.T @ q
            x = x + 0.5 * np.einsum("iab,a,b->i", self._Theta, p, p).reshape(-1, 3)
        return x

    def world_field(self, state) -> np.ndarray:
        """World node positions (N, 3)."""
        p, quat, _, _ = self.read_frame(state)
        R = quat_to_matrix(quat)
        return p[None, :] + self.full_field(state) @ R.T

    def local_velocity(self, state, com: np.ndarray | None = None) -> np.ndarray:
        """Body-frame nodal velocity (3N,) ``R^T v_c + w x (x - c) + Phi qd``."""
        _, quat, v_c, omega = self.read_frame(state)
        R = quat_to_matrix(quat)
        c = self.com if com is None else com
        _, qd = self.read_modes(state)
        x = self.full_field(state)
        u = (R.T @ v_c)[None, :] + np.cross((R.T @ omega)[None, :], x - c[None, :])
        return u.reshape(-1) + self.Phi @ qd

    def nodal_momentum(self, state) -> dict:
        """World linear and angular momentum (about the world origin) and kinetic energy of the nodal field."""
        p, quat, _, _ = self.read_frame(state)
        R = quat_to_matrix(quat)
        x = self.full_field(state)
        u = self.local_velocity(state)
        Mu = (self.M @ u).reshape(-1, 3)
        P_l = Mu.sum(axis=0)
        L_l = np.cross(x, Mu).sum(axis=0)
        P = R @ P_l
        L = R @ L_l + np.cross(p, P)
        return {"P": P, "L": L, "KE": 0.5 * float(u @ (self.M @ u))}

    def rom_strain_energy(self, state) -> float:
        """Strain energy the current ROM assigns (``W_ref + g0.q + 1/2 sum lambda q^2``, or the potential of
        the hyper-reduced modal force with ``nonlinear_force``)."""
        q, _ = self.read_modes(state)
        if self.nl_model is not None:
            W, _, _ = self.nl_model.evaluate(q, need_energy=True)
            return self.W_ref + float(self._nl_c @ q) + 0.5 * float(q @ (self._nl_D @ q)) + W - self._nl_W0
        return self.W_ref + float(self.g0 @ q) + 0.5 * float(np.sum(self.lam * q * q))

    def nonlinear_strain_energy(self, state) -> float:
        """Assembler (nonlinear, total-Lagrangian) strain energy of the current full field."""
        return float(self.assembler.evaluate(self.full_field(state).reshape(-1), need_tangent=False)[0])

    def map_samples(self, idx: np.ndarray, nodal: np.ndarray) -> np.ndarray:
        """Return ``W[idx] @ nodal`` for a nodal field (N, m, 3) -> (len(idx), m, 3)."""
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return np.zeros((0, *nodal.shape[1:]))
        out = self.sample_weights[idx] @ nodal.reshape(self.N, -1)
        return np.asarray(out).reshape(idx.size, *nodal.shape[1:])

    # ------------------------------------------------------------------------------------------
    # IMEX hyper-reduced remainder and tangent folding
    # ------------------------------------------------------------------------------------------
    def _init_nl_model(self, K_T=None) -> dict:
        """(Re)build ``f_tot(q) = c + F_S(q) + D q`` about ``r_ref``/``Phi`` with ``c = g0 - F_S(0)``,
        ``D = diag(Lambda) - K_S(0)`` (``K_S(0) = Phi^T K_T Phi`` from the assembled tangent for the full set)."""
        t0 = time.perf_counter()
        hr = self.nl_model
        hr.set_basis(self.r_ref, self.Phi)
        hr.set_manifold(self._Theta)
        if self._Theta is not None:
            W0, F0, K0 = hr.evaluate(np.zeros(self.m), need_tangent=True, need_energy=True, manifold_curvature=False)
        elif K_T is not None and hr.is_full:
            W0, F0, _ = hr.evaluate(np.zeros(self.m), need_energy=True)
            K0 = self.Phi.T @ (K_T @ self.Phi)
            K0 = 0.5 * (K0 + K0.T)
        else:
            W0, F0, K0 = hr.evaluate(np.zeros(self.m), need_tangent=True, need_energy=True)
        self._nl_c = self.g0 - F0
        self._nl_D = np.diag(self.lam) - K0
        self._nl_W0 = float(W0)
        self._fnl = np.zeros(self.m)
        self._fnl_step = self.step_count
        sc = max(float(np.max(np.abs(self.lam))), 1.0e-30)
        return {
            "nl_setup": time.perf_counter() - t0,
            "nl_D_rel": float(np.max(np.abs(self._nl_D))) / sc,
            "nl_c_norm": float(np.linalg.norm(self._nl_c)),
        }

    def nl_evaluate(self, q: np.ndarray, *, need_tangent: bool = False) -> dict:
        """Hyper-reduced modal force at ``q``: ``f_tot``, ``f_lin = g0 + Lambda q``, remainder ``f_nl`` and
        optionally the remainder Jacobian ``K_exp`` and the full tangent ``T``."""
        if self.nl_model is None:
            raise RuntimeError("nonlinear_force is not enabled")
        q = np.asarray(q, dtype=np.float64)
        _, F, K = self.nl_model.evaluate(q, need_tangent=need_tangent)
        f_tot = self._nl_c + F + self._nl_D @ q
        f_lin = self.g0 + self.lam * q
        out = {"f_tot": f_tot, "f_lin": f_lin, "f_nl": f_tot - f_lin}
        if need_tangent:
            T = K + self._nl_D
            out["T"] = 0.5 * (T + T.T)
            out["K_exp"] = out["T"] - np.diag(self.lam)
        return out

    def cached_remainder_norms(self) -> tuple[float, float, float] | None:
        """``(|f_nl|, |f_lin|, |f_tot|)`` when the remainder was evaluated at the current step, else ``None``."""
        if self._fnl_step != self.step_count:
            return None
        last = self._nl_last
        return last["f_nl_norm"], last["f_lin_norm"], last["f_tot_norm"]

    def _nl_refresh(self, state) -> None:
        t0 = time.perf_counter()
        q, _ = self.read_modes(state)
        r = self.nl_evaluate(q)
        self._fnl = r["f_nl"]
        self._fnl_step = self.step_count
        self._nl_last = {
            "step": self.step_count,
            "f_nl_norm": float(np.linalg.norm(r["f_nl"])),
            "f_lin_norm": float(np.linalg.norm(r["f_lin"])),
            "f_tot_norm": float(np.linalg.norm(r["f_tot"])),
            "t_nl": time.perf_counter() - t0,
        }

    def _nl_due(self) -> bool:
        return (
            self.nl_model is not None
            and self.nl_interval is not None
            and self._fnl_step != self.step_count
            and (self.step_count - self._fnl_step) >= self.nl_interval
        )

    def modal_joint_force(self) -> np.ndarray:
        """Modal rows of ``control.joint_f``: ``modal_force - g0`` (minus the lagged remainder with IMEX)."""
        f = self.modal_force - self.g0
        if self.nl_model is not None:
            f = f - self._fnl
        return f

    def apply_prestress(self, control, state=None) -> None:
        """Write the pre-stress (and the due IMEX remainder at ``state``) into ``control.joint_f``.

        Device path: one kernel into the modal rows (graph-capturable; the IMEX remainder, when due, is
        evaluated on the host and uploaded first).
        """
        if state is not None and self._nl_due():
            self._nl_refresh(state)
        if self._dev is not None:
            # host-side change detection (user modal_force, g0 edits, IMEX remainder): upload only on change
            self._dev.set_prestress(self.modal_joint_force(), only_if_changed=True)
            self._dev.write_prestress(control)
            return
        io.write_modal_joint_f(control, [(self.layout, self.modal_joint_force())])

    # ------------------------------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------------------------------
    def poll(self, state, dt: float | None = None, *, view=None) -> tuple[bool, str, dict]:
        """Advance the step counters, record history and evaluate the trigger (no update)."""
        self.step_count += 1
        self.steps_since_update += 1
        self.steps_since_fold += 1
        if dt is not None:
            self.last_dt = float(dt)
        self._bind(state, view)
        try:
            if self._dev is not None:
                # velocity history on the device (the M a Ritz candidate reads it at the next update)
                self._dev.poll(state, dt)
                self._dev_qdd_valid = self._dev_hist_ready and dt is not None
                self._dev_hist_ready = dt is not None
            else:
                _, qd = self.read_modes(state)
                if self._qd_hist is not None and dt is not None:
                    self._qdd_est = (qd - self._qd_hist[0]) / dt
                self._qd_hist = (qd, dt)
            if self._nl_due():
                self._nl_refresh(state)
            if self._update_requested:
                self._update_requested = False
                return True, "requested", {}
            return self.trigger.evaluate(self, state, self.steps_since_update)
        finally:
            self._unbind()

    def after_step(self, state, control, dt: float) -> bool:
        """Record history, evaluate the trigger, re-linearize when it fires (fold otherwise when due).

        Returns True on an update.
        """
        log: dict[str, Any] = {"step": self.step_count + 1}
        t_tr = time.perf_counter()
        nl_before = self._fnl_step
        fire, reason, metrics = self.poll(state, dt)
        if self._fnl_step != nl_before and self._fnl_step == self.step_count and self._nl_last:
            log.update({k: v for k, v in self._nl_last.items() if k != "step"})
        log["t_trigger"] = time.perf_counter() - t_tr - float(log.get("t_nl", 0.0))
        log.update({f"trig_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        if fire:
            info = self.relinearize(state, control)
            info["trigger"] = reason
            info["trigger_metrics"] = metrics
            log["update"] = True
            log["t_update"] = info["timings"]["total"]
            self.step_log.append(log)
            return True
        if self.fold_interval is not None and self.steps_since_fold >= self.fold_interval:
            finfo = self.fold(state, control)
            log["t_fold"] = finfo["timings"]["total"]
        self.step_log.append(log)
        return False

    # ------------------------------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------------------------------
    def relinearize(
        self, state, control=None, *, stiffness_override=None, rebasis: bool | None = None, debug=None
    ) -> dict:
        """Re-linearize at ``r_new = r_ref + Phi q`` and rewrite the Model/state in place.

        Args:
            state: Current state (rewritten: ``q = 0``, projected velocities).
            control: When given, the new pre-stress is written to ``control.joint_f``.
            stiffness_override: ``(W, f_int, K)`` replacing the assembler evaluation (tests).
            rebasis: Overrides :attr:`BasisOptions.rebasis` for this call (free-body adapter).
            debug: Diagnostics switches ``keep_basis`` / ``keep_endpoint_psi`` (free-body adapter).

        Returns the per-update diagnostics (also appended to :attr:`stats`): wall-clock split ``timings``,
        eigenvalues, kinetic energy before/after and lost in the projection, nonlinear vs ROM strain energy,
        momentum before/after.
        """
        if self._dev is not None:
            return self._relinearize_device(state, control, stiffness_override, rebasis, debug)
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        arrays = io.ModelArrays(self.model)
        self._bind(state)
        try:
            q, qd = self.read_modes(state)
            _p, quat, v_c, omega = self.read_frame(state)
            R = quat_to_matrix(quat)
            x_new = self.r_ref + (self.Phi @ q).reshape(-1, 3)
            before = self.nodal_momentum(state)
            u_local = self.local_velocity(state)
        finally:
            self._unbind()

        # ---- 1. nonlinear evaluation ---------------------------------------------------------------
        if stiffness_override is None:
            W_nl, f_int, K_T = self.assembler.evaluate(x_new.reshape(-1), need_tangent=True)
        else:
            W_nl, f_int, K_T = stiffness_override
        f_int = np.asarray(f_int, dtype=np.float64)
        K_T = host_sparse(K_T)
        K_T = 0.5 * (K_T + K_T.T)
        t1 = time.perf_counter()
        timings["assembly"] = t1 - t0

        # ---- 2. basis + state projection (adapter) ------------------------------------------------
        ctx = {
            "state": state,
            "q": q,
            "qd": qd,
            "R": R,
            "v_c": v_c,
            "omega": omega,
            "x_new": x_new,
            "u_local": u_local,
            "f_int": f_int,
            "K_T": K_T,
            "timings": timings,
            "arrays": arrays,
            "rebasis": rebasis,
            "debug": debug or {},
        }
        res = self.adapter.rebuild(self, ctx)
        Phi_new, mode_mass, lam = res["Phi"], res["mode_mass"], res["lam"]
        mass_new, com_new, I_new = res["rigid"]
        g0 = Phi_new.T @ f_int
        t2 = time.perf_counter()

        # ---- 3. integrals / samples -----------------------------------------------------------------
        nodal_phi = to_nodal(Phi_new, self.N)
        integrals = _reduced_inertia_from_pairs(x_new, nodal_phi, to_nodal(res["mass_modes"], self.N))
        rp_phi = self.map_samples(self.layout.render_samples, nodal_phi)
        sh_phi = self.map_samples(self.layout.shape_samples, nodal_phi) if self.layout.shape_samples.size else None
        t3 = time.perf_counter()
        timings["integrals"] = t3 - t2 + (t2 - res["t_samples"] if "t_samples" in res else 0.0)

        # kinetic energy bookkeeping
        qd_new = res["qd"]
        u_el_old = res["u_el_old"]
        M = self.M
        resid = u_el_old - Phi_new @ qd_new
        ke_lost = 0.5 * float(resid @ (M @ resid))
        ke_total = 0.5 * float(u_local @ (M @ u_local))
        ke_el = 0.5 * float(u_el_old @ (M @ u_el_old))
        W_rom = self.W_ref + float(self.g0 @ q) + 0.5 * float(np.sum(self.lam * q * q))

        # ---- 4. rewrite model (one gather / one scatter per array) and state -------------------------
        lam_written = self._guard_indefinite(lam)
        io.write_modal_arrays(
            arrays, self.layout, mode_mass, lam_written, res["damping"], res["mode_mass_matrix"], integrals
        )
        if self.layout.endpoints:
            rotations = io.move_endpoint_frames(arrays, self.layout, q, res["ep_phi"], res["ep_psi"])
            self.last_joint_rest_mismatch = 0.0
            if self.policy.rest_rotation is RestRotationMode.COUNTER_ROTATE:
                self.last_joint_rest_mismatch = io.counter_rotate_body_rest(arrays, self.layout, rotations)
        io.shift_sample_rest_positions(arrays, self.layout, q, rp_phi, sh_phi if res["map_shapes"] else None)
        io.write_rigid_properties(
            arrays, self.layout, mass_new + self.mass_offset, com_new, I_new + self.inertia_offset
        )
        arrays.flush()
        if self.rest_propagator is not None:
            self.rest_propagator.fix()
        io.write_body_state(state, self.layout, q=0.0, v_c=res["v_c"], omega=res["omega"], qd=qd_new)

        # controller record
        com_old = self.com
        self.r_ref = x_new
        self.Phi = Phi_new
        self.com = com_new
        self.g0 = g0
        self.lam = lam
        self.W_ref = float(W_nl)
        self._qd_hist = (qd_new, self._qd_hist[1]) if self._qd_hist is not None else None
        self._qdd_est = None
        if self.nl_model is not None and self.nl_manifold > 0:
            t_md = time.perf_counter()
            self._Theta = self._static_mds(x_new.reshape(-1), Phi_new, res["solver"]())
            timings["nl_manifold_md"] = time.perf_counter() - t_md
        nl_info = self._init_nl_model(K_T) if self.nl_model is not None else {}
        if nl_info:
            timings["nl_setup"] = nl_info["nl_setup"]
        self.steps_since_fold = 0
        if control is not None:
            self.apply_prestress(control)
        self._notify()
        t4 = time.perf_counter()
        timings["rewrite"] = t4 - t3 - timings.get("nl_setup", 0.0) - timings.get("nl_manifold_md", 0.0)
        timings["total"] = t4 - t0 - timings.get("eigen_compare_excluded", 0.0)

        after = self.nodal_momentum(state)
        info = {
            "body": self.body,
            "step": self.step_count,
            "interval": self.steps_since_update,
            "timings": timings,
            "eigenvalues": lam.tolist(),
            "g0_norm": float(np.linalg.norm(g0)),
            "W_nonlinear": float(W_nl),
            "W_rom": W_rom,
            "W_rel_error": (W_rom - float(W_nl)) / max(abs(float(W_nl)), 1.0e-30),
            "KE_before": before["KE"],
            "KE_after": after["KE"],
            "KE_lost_projection": ke_lost,
            "KE_elastic_before": ke_el,
            "KE_lost_fraction_total": ke_lost / max(ke_total, 1.0e-30),
            "KE_lost_fraction_elastic": ke_lost / max(ke_el, 1.0e-30),
            "P_before": before["P"].tolist(),
            "P_after": after["P"].tolist(),
            "L_before": before["L"].tolist(),
            "L_after": after["L"].tolist(),
            "dP": float(np.linalg.norm(after["P"] - before["P"])),
            "dL": float(np.linalg.norm(after["L"] - before["L"])),
            "q_norm_before": float(np.linalg.norm(q)),
            "com_shift": float(np.linalg.norm(com_new - com_old)),
            **res["info"],
        }
        if nl_info:
            info["nl_model"] = nl_info
        self.stats.append(info)
        self.steps_since_update = 0
        self.trigger.on_update(self)
        return info

    def _relinearize_device(self, state, control, stiffness_override, rebasis, debug) -> dict:
        """Device-side :meth:`relinearize` (:attr:`UpdatePath.DEVICE`, see :mod:`rerom.update_kernels`)."""
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        dev = self._dev
        lay = self.layout

        # ---- 1. recovery + assembly on the device, ONE packed device-to-host copy -------------------
        g = dev.gather(state, need_assembly=stiffness_override is None)
        q, qd, v_c, omega = g.q, g.qd, g.v_c, g.omega
        R = quat_to_matrix(g.quat)
        x_new = g.x_new.copy()
        self._qdd_est = g.qdd.copy() if self._dev_qdd_valid else None
        if stiffness_override is not None:
            W_nl, f_int, K_T = stiffness_override
        elif g.f_int is not None:
            if g.inverted:
                raise FloatingPointError("inverted element(s): det F <= 0")
            W_nl, f_int, K_T = g.W, g.f_int, dev.device_assembler.scipy_from_block_values(g.K_values, "csr")
        else:  # host assembler: evaluate at the gathered reference
            W_nl, f_int, K_T = self.assembler.evaluate(x_new.reshape(-1), need_tangent=True)
        f_int = np.asarray(f_int, dtype=np.float64)
        K_T = host_sparse(K_T)
        K_T = 0.5 * (K_T + K_T.T)
        t1 = time.perf_counter()
        timings["assembly"] = t1 - t0

        # ---- 2. basis (host; the adapter skips the state projection) --------------------------------
        ctx = {
            "state": state,
            "q": q,
            "qd": qd,
            "R": R,
            "v_c": v_c,
            "omega": omega,
            "x_new": x_new,
            "u_local": None,
            "f_int": f_int,
            "K_T": K_T,
            "timings": timings,
            "arrays": _GatheredArrays(self.model, g.gravity, lay.world),
            "rebasis": rebasis,
            "debug": debug or {},
            "device_projection": True,
        }
        res = self.adapter.rebuild(self, ctx)
        Phi_new, mode_mass, lam = res["Phi"], res["mode_mass"], res["lam"]
        mass_new, com_new, I_new = res["rigid"]
        g0 = Phi_new.T @ f_int
        t2 = time.perf_counter()

        # ---- 3. ONE packed host-to-device copy, rewrite kernels in place ------------------------------
        lam_written = self._guard_indefinite(lam)
        proj = res["projection"]
        mode = int(proj["mode"])
        mass_total = mass_new + self.mass_offset
        inertia_total = I_new + self.inertia_offset
        dynamic = lay.frame_dynamic
        dev.pack_inbox(
            Phi=Phi_new,
            com_new=com_new,
            G=proj.get("G"),
            mass=mass_total,
            inertia=inertia_total,
            inv_mass=(1.0 / mass_total if mass_total > 0 else 0.0) if dynamic else 0.0,
            inv_inertia=np.linalg.inv(inertia_total) if dynamic else np.zeros((3, 3)),
            mode_mass=mode_mass,
            stiffness=lam_written * mode_mass,
            damping=res["damping"],
            prestress=self.modal_force - g0,
            qd_given=res["qd"] if mode == 2 else None,
            v_c_given=res["v_c"] if mode == 2 else None,
            omega_given=res["omega"] if mode == 2 else None,
            ep_phi=res["ep_phi"],
            ep_psi=res["ep_psi"],
        )
        diag = dev.rewrite(
            state,
            control,
            mode=mode,
            counter_rotate=self.policy.rest_rotation is RestRotationMode.COUNTER_ROTATE,
            diagnostics=self.policy.diagnostics,
        )
        t3 = time.perf_counter()
        timings["integrals"] = t3 - t2 + (t2 - res["t_samples"] if "t_samples" in res else 0.0)
        W_rom = self.W_ref + float(self.g0 @ q) + 0.5 * float(np.sum(self.lam * q * q))

        # ---- 4. host bookkeeping, rest rotations, solver refresh ---------------------------------------
        self.last_joint_rest_mismatch = 0.0
        if diag is not None and lay.endpoints and self.policy.rest_rotation is RestRotationMode.COUNTER_ROTATE:
            self.last_joint_rest_mismatch = float(diag[9])
            if self.last_joint_rest_mismatch > 1.0e-6:
                warnings.warn(
                    f"elastic endpoints of body {lay.body} rotated differently (spread "
                    f"{self.last_joint_rest_mismatch:.2e} rad); joint rest rotations are only approximately "
                    "preserved by RestRotationMode.COUNTER_ROTATE",
                    stacklevel=3,
                )
        if self.rest_propagator is not None:
            self.rest_propagator.fix()
        com_old = self.com
        self.r_ref = x_new
        self.Phi = Phi_new
        self.com = com_new
        self.g0 = g0
        self.lam = lam
        self.W_ref = float(W_nl)
        self._qd_hist = None
        self._qdd_est = None
        self._dev_hist_ready = True  # history = projected qd (written by the state kernel)
        self._dev_qdd_valid = False
        if self.nl_model is not None and self.nl_manifold > 0:
            t_md = time.perf_counter()
            self._Theta = self._static_mds(x_new.reshape(-1), Phi_new, res["solver"]())
            timings["nl_manifold_md"] = time.perf_counter() - t_md
        nl_info = self._init_nl_model(K_T) if self.nl_model is not None else {}
        if nl_info:
            timings["nl_setup"] = nl_info["nl_setup"]
        self.steps_since_fold = 0
        self._notify()
        t4 = time.perf_counter()
        timings["rewrite"] = t4 - t3 - timings.get("nl_setup", 0.0) - timings.get("nl_manifold_md", 0.0)
        timings["total"] = t4 - t0 - timings.get("eigen_compare_excluded", 0.0)

        # ---- diagnostics (momenta in world, about the world origin) ---------------------------------
        p = g.p

        def world(Pl, Ll):
            P = R @ Pl
            return P, R @ Ll + np.cross(p, P)

        P_b, L_b = world(g.moments[:3], g.moments[3:6])
        ke_total = 0.5 * float(g.moments[6])
        if diag is not None:
            P_a, L_a = world(diag[:3], diag[3:6])
            ke_after, ke_el, ke_lost = 0.5 * float(diag[6]), float(diag[7]), float(diag[8])
        else:
            nan3 = np.full(3, np.nan)
            P_a, L_a = nan3, nan3
            ke_after = ke_el = ke_lost = float("nan")
        info = {
            "body": self.body,
            "step": self.step_count,
            "interval": self.steps_since_update,
            "timings": timings,
            "eigenvalues": lam.tolist(),
            "g0_norm": float(np.linalg.norm(g0)),
            "W_nonlinear": float(W_nl),
            "W_rom": W_rom,
            "W_rel_error": (W_rom - float(W_nl)) / max(abs(float(W_nl)), 1.0e-30),
            "KE_before": ke_total,
            "KE_after": ke_after,
            "KE_lost_projection": ke_lost,
            "KE_elastic_before": ke_el,
            "KE_lost_fraction_total": ke_lost / max(ke_total, 1.0e-30),
            "KE_lost_fraction_elastic": ke_lost / max(ke_el, 1.0e-30),
            "P_before": P_b.tolist(),
            "P_after": P_a.tolist(),
            "L_before": L_b.tolist(),
            "L_after": L_a.tolist(),
            "dP": float(np.linalg.norm(P_a - P_b)),
            "dL": float(np.linalg.norm(L_a - L_b)),
            "q_norm_before": float(np.linalg.norm(q)),
            "com_shift": float(np.linalg.norm(com_new - com_old)),
            "update_path": "device",
            "solver_refresh_in_place": self.last_refresh_in_place,
            **res["info"],
        }
        if nl_info:
            info["nl_model"] = nl_info
        self.stats.append(info)
        self.steps_since_update = 0
        self.trigger.on_update(self)
        return info

    def _guard_indefinite(self, lam: np.ndarray) -> np.ndarray:
        if self.policy.guard_indefinite is IndefiniteGuard.CLAMP and np.any(lam <= 0.0):
            pos = lam[lam > 0.0]
            floor = self.policy.clamp_floor_ratio * float(pos.min()) if pos.size else 1.0
            self.clamp_log.append(
                {
                    "step": self.step_count,
                    "lam_min": float(lam.min()),
                    "floor": floor,
                    "count": int(np.count_nonzero(lam <= 0.0)),
                }
            )
            return np.maximum(lam, floor)
        return lam

    def _notify(self) -> None:
        if not self.policy.notify:
            return
        if self._dev is not None and hasattr(self.solver, "refresh_elastic_mass_solve_data"):
            self.last_refresh_in_place = bool(self.solver.refresh_elastic_mass_solve_data())
            if self.layout.endpoints or self.rest_propagator is not None:
                self.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
            return
        flags = newton.ModelFlags.BODY_INERTIAL_PROPERTIES
        if self.layout.endpoints or self.rest_propagator is not None:
            flags |= newton.ModelFlags.JOINT_PROPERTIES
        self.solver.notify_model_changed(flags)

    def _static_mds(self, x0: np.ndarray, Phi: np.ndarray, solve_f) -> np.ndarray:
        return static_modal_derivatives(
            self.assembler,
            x0,
            Phi,
            int(self.nl_manifold),
            solve_f,
            self.free_dofs,
            fd_scale=self.basis_options.md_fd_scale,
            char_length=self.char_length,
        )

    def fold(self, state, control=None) -> dict:
        """Tangent folding: re-diagonalize the hyper-reduced tangent at the current ``q`` and rotate the basis.

        ``T = K_S(q_n) + D``; ``Phi <- Phi Q``, ``q <- Q^T q``, ``Lambda <- eig T``,
        ``g0 <- Q^T (f_tot(q_n) - T q_n)``; modal arrays rewritten in place, no assembly or eigensolve.
        """
        if self.nl_model is None:
            raise RuntimeError("fold needs nonlinear_force")
        t0 = time.perf_counter()
        m = self.m
        q, qd = self.read_modes(state)
        r = self.nl_evaluate(q, need_tangent=True)
        T = r["T"]
        t1 = time.perf_counter()
        lam_new, Q = np.linalg.eigh(T)  # Phi^T M Phi = I: standard problem
        for j in range(m):  # keep an identity-like Q (readable logs)
            if Q[np.argmax(np.abs(Q[:, j])), j] < 0:
                Q[:, j] = -Q[:, j]
        g_new = Q.T @ (r["f_tot"] - T @ q)
        q_new = Q.T @ q
        qd_new = Q.T @ qd
        self._nl_c = Q.T @ self._nl_c
        self._nl_D = Q.T @ self._nl_D @ Q
        self._nl_D = 0.5 * (self._nl_D + self._nl_D.T)
        self.nl_model.rotate(Q)
        Phi_new = self.Phi @ Q
        t2 = time.perf_counter()
        mass_modes = self.M @ Phi_new
        mode_mass_matrix = Phi_new.T @ mass_modes
        mode_mass = np.diag(mode_mass_matrix).copy()
        integrals = _reduced_inertia_from_pairs(self.r_ref, to_nodal(Phi_new, self.N), to_nodal(mass_modes, self.N))
        t3 = time.perf_counter()
        damping = 2.0 * self.damping_ratio * np.sqrt(np.maximum(mode_mass * lam_new * mode_mass, 0.0))
        arrays = io.ModelArrays(self.model)
        io.write_modal_arrays(arrays, self.layout, mode_mass, lam_new, damping, mode_mass_matrix, integrals)
        io.rotate_sample_rows(arrays, self.layout, Q)
        arrays.flush()
        io.write_body_state(state, self.layout, q=q_new, qd=qd_new)
        self.Phi = Phi_new
        self.lam = lam_new
        self.g0 = g_new
        self._fnl = np.zeros(m)
        self._fnl_step = self.step_count
        if self._qd_hist is not None:
            self._qd_hist = (Q.T @ self._qd_hist[0], self._qd_hist[1])
        if self._qdd_est is not None:
            self._qdd_est = Q.T @ self._qdd_est
        if self._dev is not None:
            self._dev.upload_basis(self.Phi, self.r_ref, self.com)
            self._dev.set_prestress(self.modal_joint_force())
            self._dev_hist_ready = False
            self._dev_qdd_valid = False
        if control is not None:
            self.apply_prestress(control)
        self._notify()
        t4 = time.perf_counter()
        info = {
            "step": self.step_count,
            "since_update": self.steps_since_update,
            "timings": {
                "tangent": t1 - t0,
                "eig_rotate": t2 - t1,
                "integrals": t3 - t2,
                "rewrite": t4 - t3,
                "total": t4 - t0,
            },
            "f_nl_norm_before": float(np.linalg.norm(r["f_nl"])),
            "K_exp_norm": float(np.linalg.norm(r["K_exp"], 2)),
            "lam_min": float(lam_new.min()),
            "lam_max": float(lam_new.max()),
            "rotation_offdiag": float(np.linalg.norm(Q - np.diag(np.diag(Q)))),
        }
        self.folds.append(info)
        self.steps_since_fold = 0
        return info


# ----------------------------------------------------------------------------------------------
# multi-body aggregate
# ----------------------------------------------------------------------------------------------


class UpdateSync(enum.Enum):
    """Update coupling of the bodies of a :class:`MultiFrameController`."""

    INDEPENDENT = "independent"
    """Each body updates on its own trigger."""
    SYNC = "sync"
    """All bodies update when any trigger fires."""


class MultiFrameController:
    """One :class:`ReROMController` (:class:`InterfaceBodyAdapter`) per CB body of a chain.

    Per step: one gather of the joint state for all triggers, the updates of the bodies that fire, one
    rest-rotation propagation over the joint tree from the world, one ``joint_f`` write and one solver
    notification.

    Args:
        chain: :class:`rerom.craig_bampton.CBChain`.
        solver: ``SolverVBD``.
        basis: :class:`~rerom.BasisOptions` of every body (``n_ritz``: interface-held Ritz loads).
        theta_deg: Element-rotation trigger per body [deg] (``None``: never by rotation).
        max_interval: Maximum steps between updates of a body.
        n_sample_elements: Rotation-trigger sample per body (default ``max(64, 512 // n_bodies)``).
        sync: :class:`UpdateSync`.
        keep_velocity: Passed to every :class:`InterfaceBodyAdapter`.
    """

    def __init__(
        self,
        chain,
        solver,
        *,
        basis: BasisOptions | None = None,
        theta_deg: float | None = 0.5,
        max_interval: int | None = None,
        n_sample_elements: int | None = None,
        sync: UpdateSync = UpdateSync.INDEPENDENT,
        keep_velocity: bool = False,
    ):
        self.chain = chain
        self.model = chain.model
        self.solver = solver
        self.sync = UpdateSync(sync)
        n_seg = len(chain.segments)
        if n_sample_elements is None:
            n_sample_elements = max(64, 512 // n_seg)
        gvec = np.array([0.0, 0.0, -chain.gravity])
        self.ctrls: list[ReROMController] = []
        for s in chain.segments:
            trig = RotationTrigger(
                theta_tol=math.radians(theta_deg) if theta_deg is not None else None,
                options=TriggerOptions(max_interval=max_interval, n_sample_elements=n_sample_elements),
            )
            adapter = InterfaceBodyAdapter(s, gravity=gvec, keep_velocity=keep_velocity)
            self.ctrls.append(
                ReROMController(
                    chain.model,
                    solver,
                    s.body,
                    s.asm,
                    adapter,
                    mode_count=s.Phi.shape[1],
                    basis=basis if basis is not None else BasisOptions(n_ritz=2),
                    trigger=trig,
                    policy=UpdatePolicy(notify=False, rest_rotation=RestRotationMode.COUNTER_ROTATE),
                )
            )
        self.rest = io.JointRestPropagator(chain.model)
        self.events: list[dict] = []
        self.step_count = 0

    # ---- rest rotations --------------------------------------------------------------------------
    def fix_rest_rotations(self) -> None:
        """Propagate rest rotations from the world so every joint keeps its original rest target."""
        self.rest.fix()

    def rest_target_error(self) -> float:
        """Max angle [rad] between the current and the original rest relative rotations over the joints."""
        return self.rest.error()

    # ---- stepping --------------------------------------------------------------------------------
    def _all_device(self) -> bool:
        return all(c.device_updater is not None for c in self.ctrls)

    def apply_prestress(self, control) -> None:
        """Pre-stress of every body (one kernel per body on the device path, else one ``joint_f`` gather and
        scatter)."""
        if self._all_device():
            for c in self.ctrls:
                c.apply_prestress(control)
            return
        io.write_modal_joint_f(control, [(c.layout, c.modal_joint_force()) for c in self.ctrls])

    def update(self, state, control, bodies, reasons=None) -> list[dict]:
        """Update the given controllers (indices), fix rest rotations, write pre-stress and notify once."""
        infos = []
        for k, i in enumerate(bodies):
            c = self.ctrls[i]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info = c.relinearize(state, control)
            if reasons is not None:
                info["trigger"], info["trigger_metrics"] = reasons[k]
            infos.append(info)
        if infos:
            t = time.perf_counter()
            self.fix_rest_rotations()
            self.apply_prestress(control)
            if self._all_device() and hasattr(self.solver, "refresh_elastic_mass_solve_data"):
                self.solver.refresh_elastic_mass_solve_data()
                self.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
            else:
                self.solver.notify_model_changed(
                    newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.JOINT_PROPERTIES
                )
            dt_fix = time.perf_counter() - t
            err = self.rest_target_error()
            for info in infos:
                info["timings"]["notify_fix"] = dt_fix / len(infos)
                info["rest_target_error"] = err
            self.events.extend(infos)
        return infos

    def after_step(self, state, control, dt) -> list[dict]:
        """Evaluate every body's trigger (one state gather); update the bodies that fire."""
        self.step_count += 1
        view = None if self._all_device() else io.read_joint_state(state)
        fire = []
        for i, c in enumerate(self.ctrls):
            ok, reason, metrics = c.poll(state, dt, view=view)
            if ok:
                fire.append((i, reason, metrics))
        if fire and self.sync is UpdateSync.SYNC:
            fired = {i for i, _, _ in fire}
            fire += [(i, "sync", {}) for i in range(len(self.ctrls)) if i not in fired]
        return self.update(state, control, [i for i, _, _ in fire], [(r, mt) for _, r, mt in fire])

    # ---- diagnostics -----------------------------------------------------------------------------
    def frame_poses(self, state) -> dict:
        """World poses of the joint frames ``{joint: (pos_p, R_p, pos_c, R_c)}`` with elastic endpoint deformation."""
        m = self.model
        bq = state.body_q.numpy()
        jq = state.joint_q.numpy()
        Xp = m.joint_X_p.numpy()
        Xc = m.joint_X_c.numpy()
        out = {}
        by_body = {c.body: c for c in self.ctrls}
        for j, p, c in self.rest.order:
            jp_, jc_ = int(m.joint_parent.numpy()[j]), int(m.joint_child.numpy()[j])
            del p, c
            poses = []
            for b, X in ((jp_, Xp[j]), (jc_, Xc[j])):
                pos = X[:3].astype(np.float64)
                rot = quat_to_matrix(X[3:7].astype(np.float64))
                if b >= 0 and b in by_body:
                    ctl = by_body[b]
                    ad = ctl.adapter
                    q = jq[ctl.q_start + 7 : ctl.q_start + 7 + ctl.m].astype(np.float64)
                    if ad.frame_map is not None and np.linalg.norm(pos - ad.tip_pos) < 1e-5:
                        f = ad.frame_map @ q
                        pos = pos + f[:3]
                        rot = quat_to_matrix(quat_from_rotvec(f[3:])) @ rot
                if b >= 0:
                    P = bq[b, :3].astype(np.float64)
                    Rb = quat_to_matrix(bq[b, 3:7].astype(np.float64))
                    pos = P + Rb @ pos
                    rot = Rb @ rot
                poses.extend([pos, rot])
            out[j] = tuple(poses)
        return out

    def interface_mismatch(self, state):
        """Max position gap [m] and rotation mismatch [rad] between the two frames of each joint."""
        gaps, angs = [], []
        for pp, Rp, pc, Rc in self.frame_poses(state).values():
            gaps.append(float(np.linalg.norm(pp - pc)))
            angs.append(float(np.linalg.norm(rotvec_of_matrix(Rp.T @ Rc))))
        return max(gaps), max(angs), gaps, angs

    def segment_world_fields(self, state):
        """Per-segment world node positions and velocities (lists)."""
        jq = state.joint_q.numpy()
        jqd = state.joint_qd.numpy()
        com = self.model.body_com.numpy()
        X, V = [], []
        for c in self.ctrls:
            q = jq[c.q_start + 7 : c.q_start + 7 + c.m].astype(np.float64)
            qd = jqd[c.qd_start + 6 : c.qd_start + 6 + c.m].astype(np.float64)
            p = jq[c.q_start : c.q_start + 3].astype(np.float64)
            R = quat_to_matrix(jq[c.q_start + 3 : c.q_start + 7].astype(np.float64))
            v_c = jqd[c.qd_start : c.qd_start + 3].astype(np.float64)
            w = jqd[c.qd_start + 3 : c.qd_start + 6].astype(np.float64)
            rl = c.r_ref + (c.Phi @ q).reshape(-1, 3)
            xw = p + rl @ R.T
            cw = R @ com[c.body].astype(np.float64)
            vw = v_c + np.cross(w, rl @ R.T - cw) + (c.Phi @ qd).reshape(-1, 3) @ R.T
            X.append(xw)
            V.append(vw)
        return X, V

    def full_field(self, state):
        """Global world positions/velocities (interface nodes averaged), kinetic energy, node gap at interfaces."""
        X, V = self.segment_world_fields(state)
        n = self.chain.n_nodes
        Xg = np.zeros((n, 3))
        Vg = np.zeros((n, 3))
        cnt = np.zeros(n)
        ke = 0.0
        for s, x, v in zip(self.chain.segments, X, V, strict=True):
            Xg[s.global_ids] += x
            Vg[s.global_ids] += v
            cnt[s.global_ids] += 1
            ke += 0.5 * float(v.ravel() @ (s.M @ v.ravel()))
        gap = 0.0
        for a in range(len(X) - 1):
            sa, sb = self.chain.segments[a], self.chain.segments[a + 1]
            common, ia, ib = np.intersect1d(sa.global_ids, sb.global_ids, return_indices=True)
            if common.size:
                gap = max(gap, float(np.max(np.linalg.norm(X[a][ia] - X[a + 1][ib], axis=1))))
        Xg /= cnt[:, None]
        Vg /= cnt[:, None]
        return Xg, Vg, {"kinetic": ke, "node_gap": gap, "segment_fields": X}

    def assembly_momentum(self, state):
        """World linear momentum and angular momentum about the world origin of all segments (nodal)."""
        X, V = self.segment_world_fields(state)
        P = np.zeros(3)
        L = np.zeros(3)
        for s, x, v in zip(self.chain.segments, X, V, strict=True):
            Mv = (s.M @ v.ravel()).reshape(-1, 3)
            P += Mv.sum(axis=0)
            L += np.cross(x, Mv).sum(axis=0)
        return P, L

    def rom_elastic_energy(self, state) -> float:
        """Sum of the bodies' ROM strain energies."""
        jq = state.joint_q.numpy()
        e = 0.0
        for c in self.ctrls:
            q = jq[c.q_start + 7 : c.q_start + 7 + c.m].astype(np.float64)
            e += c.W_ref + float(c.g0 @ q) + 0.5 * float(np.sum(c.lam * q * q))
        return e
