"""Update triggers of the re-ROM controller.

Every trigger shares the gating of :class:`~rerom.options.TriggerOptions` (minimum interval, update
budget, the controller's freeze hooks and rejection back-off, maximum interval, check cadence) and then
applies its own criterion. ``evaluate(ctrl, state, steps_since) -> (fire, reason, metrics)``.

Two kinds of alarm are distinguished:

- *validity* (:meth:`Trigger.check_validity`, :class:`ValidityTrigger`): the current state is not a valid
  configuration of the model (non-finite modal state, inverted or collapsing elements). Validity alarms run
  before and bypass every gate.
- *accuracy* (:meth:`Trigger.check`): the linearization is getting stale. When a gate suppresses the trigger,
  the criterion is still evaluated (``TriggerOptions.audit_suppressed``) and a fired alarm is recorded in
  ``controller.accuracy_unverified`` rather than dropped.

:class:`CompositeTrigger` combines triggers with OR logic (for example a cheap rotation proxy, an energy
Taylor-defect check and a validity check).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .numerics import host_sparse, imex_spectral_radius
from .options import TriggerOptions

__all__ = [
    "CompositeTrigger",
    "EnergyMismatchTrigger",
    "ResidualTrigger",
    "RotationTrigger",
    "Trigger",
    "ValidityTrigger",
]


def _element_sample(options: TriggerOptions, ctrl) -> np.ndarray | None:
    """Element sample of ``options`` (``element_ids``, else a deterministic ``n_sample_elements`` sample;
    ``None``: all elements)."""
    if options.element_ids is not None:
        return np.asarray(options.element_ids)
    n_el = int(ctrl.assembler.mesh.tets.shape[0]) if hasattr(ctrl.assembler, "mesh") else None
    n_s = options.n_sample_elements
    if n_el is None or n_s is None or n_el <= n_s:
        return None
    return np.unique(np.linspace(0, n_el - 1, n_s).astype(np.int64))


@dataclass(kw_only=True)
class Trigger:
    """Base class: validity checks, shared gating, then :meth:`check` (the trigger's own accuracy criterion)."""

    options: TriggerOptions = field(default_factory=TriggerOptions)

    def evaluate(self, ctrl, state, steps_since: int) -> tuple[bool, str, dict]:
        """Validity checks (never gated), then the gating of :attr:`options` and the controller's freeze hooks and
        rejection back-off, then :meth:`check`.

        A suppressed trigger still evaluates its criterion when ``options.audit_suppressed`` (on the
        ``check_every`` cadence) and reports a fired alarm to ``ctrl.note_accuracy_unverified``; it does not fire.
        """
        metrics: dict[str, Any] = {}
        o = self.options
        reason = self.check_validity(ctrl, state, metrics)
        if reason:
            return True, reason, metrics
        suppressed = self._suppression(ctrl, steps_since)
        if suppressed is not None:
            if o.audit_suppressed and steps_since % max(o.check_every, 1) == 0:
                fire, why, metrics = self.check(ctrl, state, steps_since, metrics)
                if fire:
                    note = getattr(ctrl, "note_accuracy_unverified", None)
                    if note is not None:
                        note(why, suppressed, metrics)
            return False, "", metrics
        if o.max_interval is not None and steps_since >= o.max_interval:
            return True, "max_interval", metrics
        if steps_since % max(o.check_every, 1) != 0:
            return False, "", metrics
        return self.check(ctrl, state, steps_since, metrics)

    def _suppression(self, ctrl, steps_since: int) -> str | None:
        """Name of the gate that suppresses the trigger now (``None``: not suppressed)."""
        o = self.options
        if steps_since < ctrl.effective_min_interval(o.min_interval):
            return "min_interval"
        if o.max_updates is not None and ctrl.update_count >= o.max_updates:
            return "budget"
        if ctrl.frozen():
            return "freeze"
        blocked = getattr(ctrl, "update_suppression", None)
        return blocked() if blocked is not None else None

    def check_validity(self, ctrl, state, metrics: dict) -> str | None:
        """Validity criterion, evaluated before and regardless of the gating: the alarm reason, or ``None``."""
        return None

    def check(self, ctrl, state, steps_since: int, metrics: dict) -> tuple[bool, str, dict]:
        """The trigger's own criterion (gating already passed)."""
        return False, "", metrics

    def on_update(self, ctrl) -> None:
        """Called by the controller after every update (refresh reference data)."""


@dataclass(kw_only=True)
class RotationTrigger(Trigger):
    """Fires when the max element rotation of the current field relative to ``r_ref`` exceeds ``theta_tol``
    [rad], or the max nodal modal displacement ``|Phi q|`` exceeds ``disp_tol * char_length``.

    Element rotations are evaluated on ``options.element_ids`` or a deterministic sample of
    ``options.n_sample_elements`` elements (all elements when the mesh is smaller). On the device update
    path with a device tangent assembler the check runs as kernels (:meth:`DeviceUpdater.rotation_check
    <rerom.update_kernels.DeviceUpdater.rotation_check>`) with one 3-value device-to-host read per check.

    The rotation angle is a cheap *proxy* for the linearization error, and the tolerance it needs depends on
    geometry and slenderness (the spurious membrane energy of a linearly rotated segment scales like
    ``(L/h)^2 theta^2``), mode content and time step; no angle is universally safe. The default 0.5 deg
    follows the package's slender-beam evidence: on the branch's own coarse beam example (189 nodes, 12 modes,
    StVK, E = 10 MPa, gravity release, 0.6 s against backward-Euler FE on the same mesh) the former 4 deg
    default gave 16.95 % L tip RMS error with 12 updates, 0.5 deg gave 0.22 % L with 109 updates and 0.2 deg
    0.03 % L with 228 updates (external re-ROM review, 26 Oct 2026, E1; dt, VBD iteration and eigensolver
    controls did not change the 4 deg result). Pass ``theta_tol=math.radians(4.0)`` for the former default.
    A rotation criterion is blind to stretch without rotation (pure extension has zero polar rotation):
    combine it with :class:`EnergyMismatchTrigger` and :class:`ValidityTrigger` in a :class:`CompositeTrigger`
    where that matters.
    """

    theta_tol: float | None = math.radians(0.5)
    disp_tol: float | None = None
    char_length: float | None = None
    _sample: np.ndarray | None = field(default=None, init=False, repr=False)

    def check(self, ctrl, state, steps_since, metrics):
        dev = getattr(ctrl, "device_updater", None)
        if dev is not None and (dev.device_assembly or self.theta_tol is None):
            return self._check_device(ctrl, dev, state, metrics)
        q = ctrl.read_modes(state)[0]
        u = (ctrl.Phi @ q).reshape(-1, 3)
        if self.disp_tol is not None:
            L = self.char_length if self.char_length is not None else ctrl.char_length
            disp = float(np.max(np.linalg.norm(u, axis=1))) / L
            metrics["disp_rel"] = disp
            if disp > self.disp_tol:
                return True, "displacement", metrics
        if self.theta_tol is not None and hasattr(ctrl.assembler, "element_rotations"):
            ids = self.sample_ids(ctrl)
            x = (ctrl.r_ref + u).reshape(-1)
            try:
                ang = ctrl.assembler.element_rotations(x, ids, reference_flat=ctrl.r_ref.reshape(-1))
            except TypeError:  # assembler without reference support: total angle difference
                ang = np.abs(
                    ctrl.assembler.element_rotations(x, ids)
                    - ctrl.assembler.element_rotations(ctrl.r_ref.reshape(-1), ids)
                )
            theta = float(np.max(ang)) if ang.size else 0.0
            metrics["theta_max"] = theta
            if theta > self.theta_tol:
                return True, "rotation", metrics
        return False, "", metrics

    def _check_device(self, ctrl, dev, state, metrics):
        """Same criterion on the device (no ``joint_q`` host read; one 3-value flag read)."""
        L = self.char_length if self.char_length is not None else ctrl.char_length
        has_theta = self.theta_tol is not None and hasattr(ctrl.assembler, "element_rotations")
        code, theta, disp = dev.rotation_check(
            state, self.sample_ids(ctrl) if has_theta else None, self.theta_tol if has_theta else None, self.disp_tol, L
        )
        if self.disp_tol is not None:
            metrics["disp_rel"] = disp
            if code == 1.0:
                return True, "displacement", metrics
        if has_theta:
            metrics["theta_max"] = theta
            if code == 2.0:
                return True, "rotation", metrics
        return False, "", metrics

    def sample_ids(self, ctrl):
        """Element sample (``None``: all elements)."""
        if self.options.element_ids is not None:
            return self.options.element_ids
        if self._sample is not None:
            return self._sample
        self._sample = _element_sample(self.options, ctrl)
        return self._sample


@dataclass(kw_only=True)
class EnergyMismatchTrigger(Trigger):
    """Fires when the nonlinear strain energy of a sampled element subset departs from its quadratic
    (ROM-consistent) Taylor model about ``r_ref``.

    On sampled elements S: ``dW_nl = W_S(x) - W_S(r_ref)`` and ``dW_q = f_S(r_ref).u + 1/2 u^T K_S(r_ref) u``
    with ``u = x - r_ref``, ``x = r_ref + Phi q``; fires when
    ``|dW_nl - dW_q| > tol * (|f_S.u| + |1/2 u^T K_S u|)``. The sampled quantities come from a
    sub-assembler on the sampled elements (``type(assembler)(submesh, material)``), so a check costs
    O(|S|); the reference data are refreshed at every update.
    """

    tol: float = 0.05
    floor_abs: float = 1.0e-12
    _ids: np.ndarray | None = field(default=None, init=False, repr=False)
    _sub: Any = field(default=None, init=False, repr=False)
    _ref: Any = field(default=None, init=False, repr=False)

    def _build(self, ctrl):
        asm = ctrl.assembler
        mesh = asm.mesh
        n_el = int(mesh.tets.shape[0])
        ids = self.options.element_ids
        if ids is None:
            n_s = self.options.n_sample_elements
            if n_s is None or n_el <= n_s:
                ids = np.arange(n_el)
            else:
                ids = np.unique(np.linspace(0, n_el - 1, n_s).astype(np.int64))
        self._ids = ids
        sub_mesh = copy.copy(mesh)
        sub_mesh.tets = np.ascontiguousarray(mesh.tets[ids])
        inner = getattr(asm, "inner", asm)
        material = getattr(asm, "material", "stvk")
        if hasattr(type(inner), "from_mesh"):  # device assembler: same class and device on the submesh
            self._sub = type(inner).from_mesh(sub_mesh, material, device=getattr(inner, "device", None))
        else:
            self._sub = type(asm)(sub_mesh, material)

    def on_update(self, ctrl):
        if self._sub is None:
            self._build(ctrl)
        W, f, K = self._sub.evaluate(ctrl.r_ref.reshape(-1), need_tangent=True)
        self._ref = (float(W), np.asarray(f), host_sparse(K))

    def check(self, ctrl, state, steps_since, metrics):
        if self._ref is None:
            self.on_update(ctrl)
        W0, f0, K0 = self._ref
        q = ctrl.read_modes(state)[0]
        u = ctrl.Phi @ q
        x = ctrl.r_ref.reshape(-1) + u
        W = float(self._sub.evaluate(x, need_tangent=False)[0])
        dW_nl = W - W0
        lin = float(f0 @ u)
        quad = 0.5 * float(u @ (K0 @ u))
        dW_q = lin + quad
        # scale |f.u| + |1/2 u K u| avoids spurious firing when the linear term crosses zero
        ratio = abs(dW_nl - dW_q) / max(abs(lin) + abs(quad), self.floor_abs)
        metrics["energy_mismatch"] = ratio
        if ratio > self.tol:
            return True, "energy", metrics
        return False, "", metrics


@dataclass(kw_only=True)
class ValidityTrigger(Trigger):
    """Validity alarm: non-finite modal state, or ``det F`` of a sampled element at or below ``det_f_floor``.

    The check runs before the gating of :class:`~rerom.options.TriggerOptions` and is never suppressed by the
    minimum interval, the update budget, a freeze hook or a rejection back-off. It has no accuracy criterion of
    its own; use it inside a :class:`CompositeTrigger`. ``det F`` is evaluated on the linear field
    ``r_ref + Phi q`` against the rest mesh on ``options.element_ids`` or ``options.n_sample_elements`` elements
    (``n_sample_elements=None``: all), one host read of the modal state per check (also on the device path).
    A large element rotation is not an inversion test; ``det F`` is.

    Attributes:
        det_f_floor: Alarm when the minimum ``det F`` is at or below this value (``None``: no ``det F`` check).
        check_finite: Alarm on non-finite modal coordinates or velocities.
    """

    det_f_floor: float | None = 0.0
    check_finite: bool = True
    _sample: np.ndarray | None = field(default=None, init=False, repr=False)
    _sampled: bool = field(default=False, init=False, repr=False)

    def check_validity(self, ctrl, state, metrics):
        q, qd = ctrl.read_modes(state)
        if self.check_finite and not (np.all(np.isfinite(q)) and np.all(np.isfinite(qd))):
            metrics["finite"] = 0.0
            return "validity_non_finite"
        if self.det_f_floor is not None:
            if not self._sampled:
                self._sample = _element_sample(self.options, ctrl)
                self._sampled = True
            det_min = ctrl.min_det_f(ctrl.r_ref.reshape(-1) + ctrl.Phi @ q, self._sample)
            if det_min is not None:
                metrics["det_f_min"] = det_min
                if not det_min > self.det_f_floor:
                    return "validity_det_f"
        return None


@dataclass(kw_only=True)
class CompositeTrigger(Trigger):
    """OR of child triggers under one gating (this trigger's :attr:`options`; the children's own gating options
    are not used).

    Validity checks of all children run first and bypass the gating. Then every child's accuracy criterion is
    evaluated (no short circuit), so the metrics of all children are reported; ``metrics["fired"]`` lists the
    reasons that fired and the returned reason joins them with ``"+"``. Metric names that collide between
    children are prefixed with the child index (``"1.theta_max"``).

    Attributes:
        triggers: Child triggers, for example ``[RotationTrigger(), EnergyMismatchTrigger(), ValidityTrigger()]``.
    """

    triggers: list[Trigger] = field(default_factory=list)

    @staticmethod
    def _merge(metrics: dict, child: dict, index: int) -> None:
        for k, v in child.items():
            metrics[k if k not in metrics else f"{index}.{k}"] = v

    def check_validity(self, ctrl, state, metrics):
        reasons = []
        for i, t in enumerate(self.triggers):
            m: dict = {}
            r = t.check_validity(ctrl, state, m)
            self._merge(metrics, m, i)
            if r:
                reasons.append(r)
        if reasons:
            metrics["fired"] = reasons
            return "+".join(reasons)
        return None

    def check(self, ctrl, state, steps_since, metrics):
        fired = []
        for i, t in enumerate(self.triggers):
            fire, reason, m = t.check(ctrl, state, steps_since, {})
            self._merge(metrics, m, i)
            if fire:
                fired.append(reason)
        metrics["fired"] = fired
        return bool(fired), "+".join(fired), metrics

    def on_update(self, ctrl):
        for t in self.triggers:
            t.on_update(ctrl)


@dataclass(kw_only=True)
class ResidualTrigger(Trigger):
    """Residual / stability trigger for the IMEX remainder (``nonlinear_force`` set on the controller).

    Fires when (any of)
      * the spectral radius of the linearized IMEX step (:func:`rerom.numerics.imex_spectral_radius`,
        remainder Jacobian of the sampled tangent, every ``stability_every`` steps) exceeds ``rho_max``;
      * ``||f_nl|| / max(||g0 + Lambda q||, ||f_tot||, floor_abs) > tol``;
      * the element rotation since ``r_ref`` exceeds ``theta_cap`` (safety cap).
    ``None`` disables a criterion.
    """

    tol: float | None = 0.05
    rho_max: float | None = 1.0
    stability_every: int = 1
    theta_cap: float | None = math.radians(10.0)
    floor_abs: float = 1.0e-9
    _rot: RotationTrigger | None = field(default=None, init=False, repr=False)

    def check(self, ctrl, state, steps_since, metrics):
        if ctrl.nl_model is None:
            raise RuntimeError("ResidualTrigger needs the controller's nonlinear_force option")
        q, _ = ctrl.read_modes(state)
        need_t = self.rho_max is not None and steps_since % max(self.stability_every, 1) == 0
        cached = ctrl.cached_remainder_norms()
        if cached is not None and not need_t:
            nf, nl, nt = cached
        else:
            r = ctrl.nl_evaluate(q, need_tangent=need_t)
            nf, nl, nt = (float(np.linalg.norm(r[k])) for k in ("f_nl", "f_lin", "f_tot"))
        ratio = nf / max(nl, nt, self.floor_abs)
        metrics["residual_ratio"] = ratio
        if need_t:
            damp = 2.0 * ctrl.damping_ratio * np.sqrt(np.maximum(ctrl.lam, 0.0))
            try:
                rho = imex_spectral_radius(ctrl.lam, r["K_exp"], ctrl.last_dt or 1.0e-3, damp)
            except (np.linalg.LinAlgError, ValueError):
                rho = float("inf")
            if not np.isfinite(rho):
                rho = float("inf")
            metrics["rho"] = rho
            if rho > self.rho_max:
                return True, "stability", metrics
        if self.tol is not None and ratio > self.tol:
            return True, "residual", metrics
        if self.theta_cap is not None:
            if self._rot is None:
                self._rot = RotationTrigger(
                    theta_tol=self.theta_cap,
                    options=TriggerOptions(min_interval=0, n_sample_elements=self.options.n_sample_elements),
                )
            fire, _, m2 = self._rot.check(ctrl, state, steps_since, {})
            metrics.update(m2)
            if fire:
                return True, "rotation_cap", metrics
        return False, "", metrics
