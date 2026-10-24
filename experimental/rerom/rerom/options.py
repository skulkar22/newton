"""Keyword-only option groups and enums of the re-ROM controller."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BasisOptions",
    "IndefiniteGuard",
    "NonlinearForceOptions",
    "RestRotationMode",
    "TriggerOptions",
    "UpdatePolicy",
]


class RestRotationMode(enum.Enum):
    """How the controller keeps SolverVBD's joint rest relative rotations after moving endpoint frames.

    SolverVBD measures a joint's angular error against ``rot(model.body_q[p] X_p)^-1 rot(model.body_q[c] X_c)``.
    An update moves an elastic endpoint frame ``X`` to its deformed orientation, which would move that
    target too.
    """

    COUNTER_ROTATE = "counter_rotate"
    """Counter-rotate the body's rest rotation in ``model.body_q`` by the (mean) endpoint rotation.

    Exact only when all endpoints of the body rotate equally (a warning reports the spread otherwise).
    """
    KEEP = "keep"
    """Leave ``model.body_q`` alone (an outer object fixes rest rotations, or the solver keeps its own)."""
    JOINT_TREE = "joint_tree"
    """Re-propagate the rest rotations of ``model.body_q`` over the joint graph from a root (see
    :class:`rerom.model_io.JointRestPropagator`); exact for trees, raises on kinematic loops."""
    REST_FRAMES = "rest_frames"
    """Pin the build-time joint frames once with ``SolverVBD.set_joint_rest_frames`` (requires the
    per-joint rest-frame patch) and leave ``model.body_q`` alone; works with kinematic loops."""


class IndefiniteGuard(enum.Enum):
    """What to do with non-positive tangent eigenvalues of the updated basis."""

    KEEP = "keep"
    """Write them as they are (counted in ``info["negative_eigenvalues"]``)."""
    CLAMP = "clamp"
    """Raise them to ``UpdatePolicy.clamp_floor_ratio`` times the smallest positive one (SolverVBD needs a
    positive modal stiffness for some joint solves); clamps are logged in ``controller.clamp_log``."""


@dataclass(kw_only=True)
class BasisOptions:
    """Composition of the updated basis.

    Attributes:
        n_ritz: Static Ritz vectors (load ``f_int``, ``f_ext``, velocity ``M v``, acceleration ``M a``,
            then block-Krylov ``K^-1 M r_k`` with ``ritz_krylov``); the eigenmode count is
            ``mode_count - n_ritz_kept - md_kept``.
        modal_derivatives: ``k > 0`` adds the ``k(k+1)/2`` static modal derivatives of the ``k`` lowest
            tangent modes (central difference of the tangent with step ``md_fd_scale * char_length``).
        ritz_krylov: Continue with block-Krylov vectors when the candidates give fewer than ``n_ritz``.
        rebasis: ``False`` skips the eigen-solve: the old basis is re-orthogonalized against the rigid
            motions of the new reference, M-orthonormalized and rotated (plain re-referencing).
        ritz_drop_tol: Relative norm below which an orthogonalized candidate is dropped.
        md_fd_scale: Peak finite-difference displacement of the modal derivatives / char. length.
        eig_margin: Extra eigenpairs (ARPACK) or block vectors (subspace) beyond the mode count.
        subspace_iterations: Block iterations of the warm-started subspace eigensolver.
        dense_dof_limit: Free DOF count up to which the restricted eigenproblem is solved densely.
        derive_psi: Estimate angular samples (``psi``) of the new modes; ``None``: when the basis had psi.
        damping_ratio: Modal damping ratio of the new modes; ``None`` infers the mean initial ratio.
        eigen_compare: Also run the other built-in restricted eigensolver and record principal angles
            (diagnostic; its time is excluded from ``timings["total"]``).
    """

    n_ritz: int = 2
    modal_derivatives: int = 0
    ritz_krylov: bool = True
    rebasis: bool = True
    ritz_drop_tol: float = 1.0e-6
    md_fd_scale: float = 1.0e-3
    eig_margin: int = 2
    subspace_iterations: int = 1
    dense_dof_limit: int = 0
    derive_psi: bool | None = None
    damping_ratio: float | None = None
    eigen_compare: bool = False


@dataclass(kw_only=True)
class TriggerOptions:
    """Gating shared by all triggers (checked in this order before the trigger's own criterion).

    Attributes:
        min_interval: Never fire before this many steps since the last update.
        max_updates: Update budget; no trigger fires once the controller made this many updates.
        max_interval: Fire unconditionally after this many steps (``None``: never).
        check_every: Evaluate the criterion only every ``check_every`` steps.
        n_sample_elements: Deterministic element sample of sampled criteria (``None``: all elements).
        element_ids: Explicit element sample (overrides ``n_sample_elements``).
    """

    min_interval: int = 1
    max_updates: int | None = None
    max_interval: int | None = None
    check_every: int = 1
    n_sample_elements: int | None = 512
    element_ids: np.ndarray | None = None


@dataclass(kw_only=True)
class UpdatePolicy:
    """When updates are allowed and how they are written.

    The freeze hooks are driven by the caller through :meth:`rerom.ReROMController.set_phase`
    (``released``, ``in_contact``); :meth:`rerom.ReROMController.request_update` forces one update at the
    next check (for example once a settled state is reached).

    Attributes:
        freeze_after_release: No updates once ``set_phase(released=True)`` was called.
        freeze_in_contact: No updates while ``set_phase(in_contact=True)``.
        freeze_from_step: No updates from this controller step on.
        post_release_min_interval: Minimum update interval once released (overrides the trigger's).
        guard_indefinite: Handling of non-positive tangent eigenvalues.
        clamp_floor_ratio: Floor of :attr:`IndefiniteGuard.CLAMP` relative to the smallest positive one.
        rest_rotation: Joint rest-rotation handling after endpoint frames moved.
        rest_root: Root body of :attr:`RestRotationMode.JOINT_TREE` (``None``: the world).
        notify: Call ``solver.notify_model_changed`` after a rewrite.
    """

    freeze_after_release: bool = False
    freeze_in_contact: bool = False
    freeze_from_step: int | None = None
    post_release_min_interval: int | None = None
    guard_indefinite: IndefiniteGuard = IndefiniteGuard.KEEP
    clamp_floor_ratio: float = 0.05
    rest_rotation: RestRotationMode = RestRotationMode.COUNTER_ROTATE
    rest_root: int | None = None
    notify: bool = True


@dataclass(kw_only=True)
class NonlinearForceOptions:
    """IMEX hyper-reduced nonlinear modal remainder (free-body adapter only).

    The modal internal force seen by the step becomes ``g0 + Lambda q_{n+1} + f_nl(q_n)`` with the lagged
    remainder ``f_nl(q) = f_tot(q) - (g0 + Lambda q)`` and ``f_tot(q) = c + F_S(q) + D q``,
    ``F_S(q) = sum_e w_e Phi_e^T f_e(r_ref + Phi q)`` on an element subset; ``c`` and ``D`` are set at each
    update so that ``f_tot(0) = g0`` and ``f_tot'(0) = Lambda``.

    Attributes:
        element_ids: Element subset of ``F_S`` (``None``: all elements).
        weights: Element weights (``None``: ones).
        interval: Evaluate the remainder every ``interval`` steps (``None``: only at updates, where it is
            zero, which reproduces the plain re-ROM).
        fold_interval: Re-diagonalize the hyper-reduced tangent every ``fold_interval`` steps without an
            update (rotate ``Phi``, ``q``; no assembly or eigensolve).
        manifold: ``k > 0`` evaluates the remainder on the quadratic manifold of the static modal
            derivatives of the ``k`` lowest modes.
    """

    element_ids: np.ndarray | None = None
    weights: np.ndarray | None = None
    interval: int | None = 1
    fold_interval: int | None = None
    manifold: int = 0
