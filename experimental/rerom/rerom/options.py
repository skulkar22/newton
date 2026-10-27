"""Keyword-only option groups and enums of the re-ROM controller."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BasisOptions",
    "IndefiniteGuard",
    "NonlinearForceOptions",
    "RejectReason",
    "RestRotationMode",
    "TransferMode",
    "TriggerOptions",
    "UpdatePath",
    "UpdatePolicy",
    "ValidationOptions",
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


class UpdatePath(enum.Enum):
    """Where the non-eigensolver part of an update runs (see :mod:`rerom.update_kernels`)."""

    DEVICE = "device"
    """Device-side update: one packed gather (state, ``r_new``, ``f_int``, ``K_T`` values) and one packed scatter
    per update; recovery, inertia integrals, sample rewrites, rigid properties and the state projection run as
    Warp kernels writing in place; the solver's mass-solve data are refreshed in place
    (``SolverVBD.refresh_elastic_mass_solve_data``, CUDA graphs stay valid); per-step pre-stress, velocity
    history and the rotation trigger run on the device. Results agree with :attr:`HOST` to roundoff
    (different summation order), not bit for bit."""
    HOST = "host"
    """Legacy host path (numpy writes through :class:`rerom.model_io.ModelArrays`,
    ``solver.notify_model_changed``); with ``BasisOptions(shared_factorization=False)`` it reproduces the
    recorded M1/WP3a runs bit for bit."""


class IndefiniteGuard(enum.Enum):
    """What to do with non-positive tangent eigenvalues of the updated basis.

    The controller keeps two spectra: the *raw* one (``controller.lam_raw``, the restricted tangent in the new
    basis, logged with its negative values) and the *operational* one (``controller.lam``), which is what the
    solver steps. Every consumer (written modal stiffness, modal damping, ROM strain energy, the IMEX linear
    term and remainder, :class:`~rerom.ResidualTrigger`) uses the operational spectrum.
    """

    KEEP = "keep"
    """Write them as they are (counted in ``info["negative_eigenvalues"]``); operational == raw."""
    CLAMP = "clamp"
    """Raise them to ``UpdatePolicy.clamp_floor_ratio`` times the smallest positive one (SolverVBD needs a
    positive modal stiffness for some joint solves); clamps are logged in ``controller.clamp_log``.

    The clamped spectrum *is* the operational model: the ROM energy, the damping and the IMEX split
    (``f_tot'(0) = Lambda_op``) all use it, so the remainder never restores the clamped negative stiffness.
    The physical (raw) tangent is no longer represented in the clamped directions; this is a regularization,
    not a physically accurate treatment of an unstable state."""
    REJECT = "reject"
    """Reject the candidate update (:attr:`RejectReason.NEGATIVE_STIFFNESS`) when the raw restricted spectrum
    (eigensolve and Rayleigh-Ritz values) has an eigenvalue below
    ``-ValidationOptions.negative_stiffness_tol * max|lambda|``; the previous basis stays in place. Smaller
    (roundoff-level) negative values are written as they are, like :attr:`KEEP`. A negative restricted tangent
    eigenvalue is a conservative rejection signal for an intended stable equilibrium, not a proof of a physical
    instability (contact or constraints may stabilize the structure)."""


class RejectReason(enum.Enum):
    """Deterministic reason codes of a rejected candidate update (``controller.rejections``)."""

    NON_FINITE = "non_finite"
    """A candidate quantity (modal state, reference, force, tangent, basis, spectrum, inertia, maps, transferred
    state) is not finite."""
    INVERTED_ELEMENT = "inverted_element"
    """``min det F`` of the candidate reference is at or below ``ValidationOptions.det_f_floor``."""
    NON_POSITIVE_MASS = "non_positive_mass"
    """Rigid mass, rigid inertia (dynamic frame) or a modal mass of the candidate is not positive."""
    BASIS_RANK = "basis_rank"
    """The candidate basis is rank deficient (Gram eigenvalue at or below ``ValidationOptions.rank_tol``)."""
    BASIS_NOT_M_ORTHONORMAL = "basis_not_m_orthonormal"
    """``max |Phi^T M Phi - I|`` exceeds ``ValidationOptions.orthonormality_tol``."""
    EIGEN_RESIDUAL = "eigen_residual"
    """The eigensolver residual exceeds ``ValidationOptions.eigen_residual_tol`` (free-body adapter)."""
    NEGATIVE_STIFFNESS = "negative_stiffness"
    """:attr:`IndefiniteGuard.REJECT` and a materially negative restricted tangent eigenvalue."""
    INTERFACE_CONSTRAINT = "interface_constraint"
    """The interface-constrained transfer is singular or ill-conditioned
    (``cond(F G^-1 F^T) > ValidationOptions.interface_cond_max``)."""
    TRANSFER_ENERGY = "transfer_energy"
    """``|delta_KE|`` of the transfer exceeds ``ValidationOptions.transfer_energy_budget * KE_before``."""
    FORCED = "forced"
    """Rejection forced through ``relinearize(debug={"reject_at": stage})`` (tests, diagnostics)."""


class TransferMode(enum.Enum):
    """How the state was transferred to the new basis at an update (``info["transfer_mode"]``)."""

    FREE = "free"
    """Free dynamic frame: M-orthogonal split into the rigid motions of the new reference and the new modal
    span. Momentum is preserved and the nodal kinetic energy can only decrease, by
    ``velocity_projection_residual_energy`` (then equal to ``-delta_KE``)."""
    PRESCRIBED_FRAME = "prescribed_frame"
    """Constrained basis or kinematic frame: the frame twist is kept and only the modal velocity is projected.
    ``velocity_projection_residual_energy`` is the energy norm of the modal velocity error, not the kinetic
    energy change (the old elastic and rigid velocities need not be M-orthogonal)."""
    INTERFACE_CONSTRAINED = "interface_constrained"
    """Craig-Bampton interface body: frame twist kept, modal velocity projected and then corrected so that the
    interface-frame velocity is preserved. The correction is affine (a nonzero prescribed interface velocity),
    so the kinetic energy can increase; ``delta_KE`` is the signed change."""


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
        shared_factorization: One factorization of ``K_T - sigma M`` per update (``sigma`` = the Ritz shift
            ``1e-3 default_shift``) serves the eigensolve, the static Ritz vectors and the modal derivatives.
            With ARPACK it is a symmetric factorization whose inertia count gives the Sturm check at
            ``sigma``; ARPACK factors a lower shift only when that check finds eigenvalues below ``sigma``.
            The subspace iteration always shared its (pivoting) factorization, so the option does not change
            it. ``False`` reproduces the legacy ARPACK path bit for bit (ARPACK at ``default_shift``, a
            separate pivoting LU for the Ritz solves).
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
    shared_factorization: bool = True


@dataclass(kw_only=True)
class TriggerOptions:
    """Gating shared by all triggers (checked in this order before the trigger's own criterion).

    Validity checks (:class:`~rerom.ValidityTrigger`, also inside a :class:`~rerom.CompositeTrigger`) run
    before any gating and are never suppressed by it.

    Attributes:
        min_interval: Never fire before this many steps since the last update.
        max_updates: Update budget; no trigger fires once the controller made this many updates.
        max_interval: Fire unconditionally after this many steps (``None``: never).
        check_every: Evaluate the criterion only every ``check_every`` steps.
        n_sample_elements: Deterministic element sample of sampled criteria (``None``: all elements).
        element_ids: Explicit element sample (overrides ``n_sample_elements``).
        audit_suppressed: When the minimum interval, the budget, a freeze hook or a rejection back-off suppresses
            the trigger, still evaluate the accuracy criterion (on the ``check_every`` cadence) and record a fired
            alarm in ``controller.accuracy_unverified`` instead of dropping it silently. The criterion is read-only,
            so this changes no trajectory, only the cost of suppressed steps.
    """

    min_interval: int = 1
    max_updates: int | None = None
    max_interval: int | None = None
    check_every: int = 1
    n_sample_elements: int | None = 512
    element_ids: np.ndarray | None = None
    audit_suppressed: bool = True


@dataclass(kw_only=True)
class ValidationOptions:
    """Checks of a candidate update before it is committed (see :meth:`rerom.ReROMController.relinearize`).

    A candidate (reference, basis, spectra, pre-stress, inertia, sample/endpoint maps, transferred state and
    adapter state) is built without touching the Model, the State, the controller or the adapter. A failed
    check rejects it with a :class:`RejectReason`: nothing is written, the rejection is logged in
    ``controller.rejections`` and no new attempt is made for ``backoff_steps`` steps (also not for validity
    alarms). After ``max_consecutive_rejections`` rejections without an accepted update, attempts stop until
    :meth:`rerom.ReROMController.request_update`. Suppressed accuracy alarms are recorded in
    ``controller.accuracy_unverified``.

    Retaining the old basis only prevents this rewrite; it does not make the already advanced state accurate.
    Rolling the solver state back and retrying with a smaller step (a checkpoint of the state, contacts and
    solver caches) is not implemented.

    Attributes:
        det_f_floor: Reject when the minimum ``det F`` over all elements of the candidate reference is at or
            below this value (``None``: no check; the device path then raises on inverted elements as before).
        orthonormality_tol: Maximum ``|Phi^T M Phi - I|`` of the candidate basis.
        rank_tol: Minimum eigenvalue of the candidate's modal Gram matrix ``Phi^T M Phi``.
        eigen_residual_tol: Maximum relative residual ``|K v - lam M v| / max(|K v|, |lam| |M v|)`` of the
            eigenpairs the basis was built from (free-body adapter; ``None``: not computed, the default).
        negative_stiffness_tol: Scale of :attr:`IndefiniteGuard.REJECT`: reject when an eigenvalue is below
            ``-negative_stiffness_tol * max|lambda|``.
        interface_cond_max: Maximum condition number of ``F G^-1 F^T`` in the interface-constrained transfer.
        transfer_energy_budget: Reject when ``|delta_KE| > budget * KE_before`` for the nodal kinetic energy of the
            transfer (``None``: no check, the default). On the device path the transfer is then also evaluated on
            the host (O(N m) per update).
        backoff_steps: Steps without a new attempt after a rejection.
        max_consecutive_rejections: Stop attempting after this many rejections in a row (``None``: never).
    """

    det_f_floor: float | None = 0.0
    orthonormality_tol: float = 1.0e-6
    rank_tol: float = 1.0e-10
    eigen_residual_tol: float | None = None
    negative_stiffness_tol: float = 1.0e-8
    interface_cond_max: float = 1.0e12
    transfer_energy_budget: float | None = None
    backoff_steps: int = 10
    max_consecutive_rejections: int | None = 5


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
        notify: Refresh the solver after a rewrite (``SolverVBD.refresh_elastic_mass_solve_data`` on the device
            path, ``solver.notify_model_changed`` on the host path; ``JOINT_PROPERTIES`` is notified when joint
            frames moved).
        path: :class:`UpdatePath` (default :attr:`UpdatePath.DEVICE`).
        diagnostics: Device path: read the momentum / kinetic-energy bookkeeping of each update (one small
            extra device-to-host copy). ``False`` leaves those ``info`` entries ``nan``.
        validation: :class:`ValidationOptions` of the transactional update.
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
    path: UpdatePath = UpdatePath.DEVICE
    diagnostics: bool = True
    validation: ValidationOptions = field(default_factory=ValidationOptions)


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
