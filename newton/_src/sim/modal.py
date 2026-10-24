# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reduced modal basis utilities for elastic links."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

import numpy as np

from . import modal_sparse

if TYPE_CHECKING:
    from scipy.sparse import sparray, spmatrix

_PAIR_CHUNK_BYTES = 256 * 1024 * 1024
"""Upper bound [bytes] for each float64 GEMM operand chunk in :func:`_reduced_inertia_from_pairs`."""


class _ReducedInertia(NamedTuple):
    """Reduced floating-frame inertia integrals of a sampled modal basis.

    Units assume displacement-valued modal coordinates, so ``phi`` is
    dimensionless. Every array is ``float32``; rigid properties are ``None``
    unless a lumped sample mass was supplied.
    """

    mode_coupling_quadratic: np.ndarray
    """``P[i, j] = sum_s phi_i(s) outer (m phi_j)(s)`` [kg], shape ``[mode_count, mode_count, 3, 3]``."""
    mode_mass_matrix: np.ndarray
    """``Mf[i, j] = trace(P[i, j])`` [kg], shape ``[mode_count, mode_count]``."""
    mode_mass: np.ndarray
    """Diagonal of ``mode_mass_matrix`` [kg], shape ``[mode_count]``."""
    mode_coupling_linear: np.ndarray
    """``S[i] = sum_s (m phi_i)(s)`` [kg], shape ``[mode_count, 3]``."""
    mode_coupling_angular: np.ndarray
    """``G[i] = sum_s (m phi_i)(s) cross x_s`` [kg m], shape ``[mode_count, 3]``."""
    mode_coupling_centrifugal: np.ndarray
    """``B[i] = sum_s (m phi_i)(s) outer x_s`` [kg m], shape ``[mode_count, 3, 3]``."""
    mode_coupling_coriolis: np.ndarray
    """``C[i, j] = sum_s phi_j(s) cross (m phi_i)(s)`` [kg], shape ``[mode_count, mode_count, 3]``."""
    rigid_mass: float | None
    """Total lumped mass [kg]."""
    rigid_com: np.ndarray | None
    """Lumped center of mass in the sample frame [m], shape ``[3]``."""
    rigid_inertia: np.ndarray | None
    """Lumped rotational inertia about ``rigid_com`` [kg m^2], shape ``[3, 3]``."""


def _mass_matrix_from_quadratic(quadratic: np.ndarray) -> np.ndarray:
    """Return ``Mf[i, j] = trace(P[i, j])`` [kg] from a ``[M, M, 3, 3]`` pair tensor."""
    return np.einsum("ijcc->ij", quadratic)


def _coriolis_from_quadratic(quadratic: np.ndarray) -> np.ndarray:
    """Return ``C[i, j] = sum_s phi_j cross (m phi_i)`` from ``P[i, j] = sum_s phi_i outer (m phi_j)``.

    ``P[j, i][c, d] = sum_s phi_j[c] (m phi_i)[d]``, so each cross-product
    component is an antisymmetric difference of two ``P[j, i]`` entries.
    """
    p_ji = np.swapaxes(quadratic, 0, 1)
    coriolis = np.empty((*quadratic.shape[:2], 3), dtype=quadratic.dtype)
    coriolis[..., 0] = p_ji[..., 1, 2] - p_ji[..., 2, 1]
    coriolis[..., 1] = p_ji[..., 2, 0] - p_ji[..., 0, 2]
    coriolis[..., 2] = p_ji[..., 0, 1] - p_ji[..., 1, 0]
    return coriolis


def _mass_coupling_ratio(mass_matrix: np.ndarray | None) -> float:
    """Return ``max_{i != j} |Mf_ij| / sqrt(Mf_ii Mf_jj)``, or ``0.0`` when undefined."""
    if mass_matrix is None or mass_matrix.shape[0] < 2:
        return 0.0
    matrix = np.asarray(mass_matrix, dtype=np.float64)
    diagonal = np.sqrt(np.maximum(np.diag(matrix), 0.0))
    scale = np.outer(diagonal, diagonal)
    off_diagonal = np.abs(matrix - np.diag(np.diag(matrix)))
    ratio = np.divide(off_diagonal, scale, out=np.zeros_like(off_diagonal), where=scale > 0.0)
    return float(np.max(ratio))


def _reduced_inertia_from_pairs(
    points: np.ndarray,
    phi: np.ndarray,
    mass_phi: np.ndarray | None = None,
    mass: np.ndarray | None = None,
    *,
    chunk_bytes: int = _PAIR_CHUNK_BYTES,
) -> _ReducedInertia:
    """Return the reduced inertia integrals of a sampled modal basis.

    The mode-pair tensor ``P[i, j] = sum_s phi_i(s) outer (m phi_j)(s)`` is
    accumulated in ``float64`` as one GEMM per sample chunk, so memory stays
    ``O(mode_count^2 + chunk * mode_count)`` instead of materializing a
    ``[sample_count, mode_count, mode_count, 3]`` temporary. The modal mass
    matrix and Coriolis integral are derived from ``P``. ``mass_phi`` is the
    mass-weighted mode field: ``mass[:, None, None] * phi`` for lumped masses,
    or ``(M Phi)`` reshaped to ``[sample_count, mode_count, 3]`` for a
    consistent mass matrix.

    Args:
        points: Sample points [m], shape ``[sample_count, 3]``.
        phi: Mode samples [m/m], shape ``[sample_count, mode_count, 3]``.
        mass_phi: Mass-weighted mode samples [kg], shape
            ``[sample_count, mode_count, 3]``. Formed per chunk from ``mass``
            when omitted.
        mass: Lumped sample masses [kg], shape ``[sample_count]``. Required when
            ``mass_phi`` is omitted; also yields the rigid mass properties.
        chunk_bytes: Upper bound [bytes] for each chunked ``float64`` operand.

    Returns:
        The reduced inertia integrals, stored as ``float32``.
    """
    if mass_phi is None and mass is None:
        raise ValueError("Either mass_phi or mass must be provided")
    points64 = np.asarray(points, dtype=np.float64)
    sample_count, mode_count = int(phi.shape[0]), int(phi.shape[1])
    mass64 = None if mass is None else np.asarray(mass, dtype=np.float64).reshape((-1,))

    width = 3 * mode_count
    pairs = np.zeros((width, width), dtype=np.float64)
    linear = np.zeros((mode_count, 3), dtype=np.float64)
    angular = np.zeros((mode_count, 3), dtype=np.float64)
    centrifugal = np.zeros((mode_count, 3, 3), dtype=np.float64)
    chunk = max(1, int(chunk_bytes) // max(8 * width, 1))
    for start in range(0, sample_count, chunk):
        stop = min(start + chunk, sample_count)
        phi_chunk = np.asarray(phi[start:stop], dtype=np.float64)
        if mass_phi is None:
            mass_phi_chunk = mass64[start:stop, None, None] * phi_chunk
        else:
            mass_phi_chunk = np.asarray(mass_phi[start:stop], dtype=np.float64)
        points_chunk = points64[start:stop]
        # [3M, s] @ [s, 3M] -> pairs[(i, c), (j, d)] = sum_s phi_i[c] (m phi_j)[d]
        pairs += phi_chunk.transpose(1, 2, 0).reshape((width, stop - start)) @ mass_phi_chunk.reshape(
            (stop - start, width)
        )
        linear += mass_phi_chunk.sum(axis=0)
        angular += np.cross(mass_phi_chunk, points_chunk[:, None, :]).sum(axis=0)
        centrifugal += np.einsum("smc,sd->mcd", mass_phi_chunk, points_chunk)

    quadratic = pairs.reshape((mode_count, 3, mode_count, 3)).transpose(0, 2, 1, 3)
    mass_matrix = _mass_matrix_from_quadratic(quadratic)
    coriolis = _coriolis_from_quadratic(quadratic)

    rigid_mass = rigid_com = rigid_inertia = None
    if mass64 is not None and float(np.sum(mass64)) > 0.0:
        rigid_mass = float(np.sum(mass64))
        com = (mass64 @ points64) / rigid_mass
        offsets = points64 - com
        inertia = (
            np.sum(mass64 * np.sum(offsets * offsets, axis=1)) * np.eye(3) - (mass64[:, None] * offsets).T @ offsets
        )
        rigid_com = com.astype(np.float32)
        rigid_inertia = inertia.astype(np.float32)

    return _ReducedInertia(
        mode_coupling_quadratic=quadratic.astype(np.float32),
        mode_mass_matrix=mass_matrix.astype(np.float32),
        mode_mass=np.diag(mass_matrix).astype(np.float32),
        mode_coupling_linear=linear.astype(np.float32),
        mode_coupling_angular=angular.astype(np.float32),
        mode_coupling_centrifugal=centrifugal.astype(np.float32),
        mode_coupling_coriolis=coriolis.astype(np.float32),
        rigid_mass=rigid_mass,
        rigid_com=rigid_com,
        rigid_inertia=rigid_inertia,
    )


class ModalBasis:
    """Sampled linear modal basis for a reduced elastic body.

    .. experimental::

    The basis stores displacement samples ``phi`` at body-local material points.
    A deformation at point ``X`` is evaluated linearly as ``sum_i phi_i(X) q_i``.
    Unknown points are evaluated by inverse-distance interpolation from the
    stored samples, keeping the runtime representation independent of how the
    modes were generated. The inertia integrals below define the reduced mass
    matrix of the floating frame and modes; see :ref:`Reduced Elastic Links`.

    Units below assume displacement-valued modal coordinates ``q`` [m], so
    ``phi`` is dimensionless [m/m]. Other normalizations are equivalent: the
    reduced dynamics are invariant under ``phi -> phi / a``, ``q -> a q``,
    ``mode_mass -> mode_mass / a^2``, and ``mode_stiffness -> mode_stiffness / a^2``,
    with ``mode_damping`` and the mode-pair integrals (``mode_mass_matrix``,
    ``mode_coupling_coriolis``, ``mode_coupling_quadratic``) also scaling by
    ``1 / a^2`` and the single-mode integrals (``mode_coupling_linear``,
    ``_angular``, ``_centrifugal``) by ``1 / a``.

    Args:
        sample_points: Body-local sample points [m], shape ``[sample_count, 3]``.
        sample_phi: Translational mode values [m per modal coordinate], shape
            ``[sample_count, mode_count, 3]``.
        sample_psi: Optional angular mode values [rad per modal coordinate], shape
            ``[sample_count, mode_count, 3]``. Each entry is the body-local
            infinitesimal rotation vector of the material frame at the sample for
            a unit modal coordinate. Defaults to zeros, which disables rotational
            joint coupling.
        sample_mass: Optional per-sample lumped mass [kg], shape ``[sample_count]``.
            When supplied, ``mode_mass``, ``mode_mass_matrix``, the rigid mass
            properties, and any unspecified inertia coupling integrals
            (``mode_coupling_linear``, ``_angular``, ``_centrifugal``,
            ``_coriolis``, ``_quadratic``) are computed from it.
        mode_mass: Optional modal mass [kg], shape ``[mode_count]``. An explicit
            value always takes precedence; when omitted it is the diagonal of
            ``mode_mass_matrix``, i.e. ``sum_s mass_s |phi_i(x_s)|^2`` for lumped
            sample masses. An explicit diagonal also defines a diagonal
            ``mode_mass_matrix`` unless ``mode_mass_matrix`` is given. A
            warning is emitted when the sampled integrals show off-diagonal
            mass coupling, or a diagonal differing from the explicit value by
            a relative amount from the lumped ``sample_mass`` integral, above
            ``mode_coupling_tolerance``.
        mode_stiffness: Modal stiffness values [N/m], shape ``[mode_count]``.
        mode_damping: Modal damping values [N s/m], shape ``[mode_count]``.
        mode_coupling_linear: Optional linear coupling integral
            ``S_i = sum_s mass_s phi_i(x_s)`` [kg], shape ``[mode_count, 3]``.
            Couples floating-frame translational acceleration and gravity into
            each mode. ``None`` when unavailable.
        mode_coupling_angular: Optional angular coupling integral
            ``sum_s mass_s (phi_i(x_s) cross x_s)`` [kg m], shape
            ``[mode_count, 3]``. Couples floating-frame angular acceleration
            into each mode. ``None`` when unavailable.
        mode_coupling_centrifugal: Optional centrifugal coupling integral
            ``sum_s mass_s (phi_i(x_s) outer x_s)`` [kg m], shape
            ``[mode_count, 3, 3]``. Couples the floating-frame centrifugal
            acceleration into each mode. ``None`` when unavailable.
        mode_coupling_coriolis: Optional Coriolis coupling integral
            ``sum_s mass_s (phi_j(x_s) cross phi_i(x_s))`` [kg], shape
            ``[mode_count, mode_count, 3]``. Couples the floating-frame angular
            velocity with the modal velocities. Antisymmetric in ``(i, j)`` for
            lumped masses, so it vanishes for a single mode. ``None`` when
            unavailable. Derived from ``mode_coupling_quadratic`` when omitted.
        label: Optional basis label.
        interpolation_epsilon: Distance regularizer [m] used for inverse-distance
            interpolation.
        mode_coupling_quadratic: Optional mode-pair inertia integral
            ``P_ij = sum_s phi_i(x_s) outer (mass_s phi_j(x_s))`` [kg], shape
            ``[mode_count, mode_count, 3, 3]``. Its trace is the modal mass
            matrix and its antisymmetric part the Coriolis integral; it also
            provides the deformation-dependent inertia terms. ``None`` when
            unavailable.
        mode_mass_matrix: Optional full modal mass matrix [kg], shape
            ``[mode_count, mode_count]``. When omitted it is the trace of
            ``mode_coupling_quadratic``, or diagonal when ``mode_mass`` is
            explicit. It must be symmetric, its diagonal must match an explicit
            ``mode_mass``, and an indefinite matrix emits a warning.
        rigid_mass: Optional rigid body mass [kg] represented by the basis.
            Defaults to ``sum(sample_mass)``.
        rigid_com: Optional rigid center of mass [m] in the basis frame, shape
            ``[3]``. Defaults to the lumped sample-mass centroid.
        rigid_inertia: Optional rigid rotational inertia [kg m^2] about
            ``rigid_com``, shape ``[3, 3]``. Defaults to the lumped sample-mass
            inertia.
        mode_coupling_tolerance: Relative off-diagonal modal mass coupling
            ``max_{i != j} |Mf_ij| / sqrt(Mf_ii Mf_jj)`` above which an explicit
            diagonal ``mode_mass`` emits a warning.
        frame_points: Optional body-local attachment frame origins [m], shape
            ``[frame_count, 3]``. Attachment frames are named connection points,
            such as Craig-Bampton interface frames, whose mode values are
            prescribed rather than interpolated. They carry no mass, never
            enter the inertia integrals or the inverse-distance interpolation,
            and take precedence over material samples for joint endpoints that
            coincide with them.
        frame_phi: Translational mode values of the attachment frames [m per
            modal coordinate], shape ``[frame_count, mode_count, 3]``. Required
            when ``frame_points`` is given.
        frame_psi: Optional angular mode values of the attachment frames [rad per
            modal coordinate], shape ``[frame_count, mode_count, 3]``. Defaults to
            zeros.
        frame_labels: Optional attachment frame labels, length ``frame_count``.
    """

    def __init__(
        self,
        sample_points: Sequence[Sequence[float]] | np.ndarray | None = None,
        sample_phi: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        sample_psi: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        sample_mass: Sequence[float] | np.ndarray | None = None,
        mode_mass: Sequence[float] | np.ndarray | None = None,
        mode_stiffness: Sequence[float] | np.ndarray | None = None,
        mode_damping: Sequence[float] | np.ndarray | None = None,
        mode_coupling_linear: Sequence[Sequence[float]] | np.ndarray | None = None,
        mode_coupling_angular: Sequence[Sequence[float]] | np.ndarray | None = None,
        mode_coupling_centrifugal: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        mode_coupling_coriolis: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        label: str | None = None,
        interpolation_epsilon: float = 1.0e-8,
        *,
        mode_coupling_quadratic: np.ndarray | None = None,
        mode_mass_matrix: Sequence[Sequence[float]] | np.ndarray | None = None,
        rigid_mass: float | None = None,
        rigid_com: Sequence[float] | np.ndarray | None = None,
        rigid_inertia: Sequence[Sequence[float]] | np.ndarray | None = None,
        mode_coupling_tolerance: float = 1.0e-2,
        frame_points: Sequence[Sequence[float]] | np.ndarray | None = None,
        frame_phi: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        frame_psi: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        frame_labels: Sequence[str | None] | None = None,
    ):
        self.label = label
        self.interpolation_epsilon = float(interpolation_epsilon)
        self.mode_coupling_tolerance = float(mode_coupling_tolerance)
        if self.mode_coupling_tolerance < 0.0:
            raise ValueError(f"mode_coupling_tolerance must be non-negative, got {self.mode_coupling_tolerance}")

        if sample_points is None:
            points = np.zeros((0, 3), dtype=np.float32)
        else:
            points = np.asarray(sample_points, dtype=np.float32)
            if points.ndim == 1:
                points = points.reshape((1, 3))
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"sample_points must have shape [sample_count, 3], got {points.shape}")

        if sample_phi is None:
            mode_count = self._infer_mode_count(mode_mass, mode_stiffness, mode_damping)
            phi = np.zeros((points.shape[0], mode_count, 3), dtype=np.float32)
        else:
            phi = np.asarray(sample_phi, dtype=np.float32)
            if phi.ndim == 2 and points.shape[0] == 1:
                phi = phi.reshape((1, phi.shape[0], 3))
            if phi.ndim != 3 or phi.shape[0] != points.shape[0] or phi.shape[2] != 3:
                raise ValueError(
                    "sample_phi must have shape [sample_count, mode_count, 3], "
                    f"got {phi.shape} for {points.shape[0]} sample points"
                )
            mode_count = int(phi.shape[1])

        if sample_psi is None:
            psi = np.zeros((points.shape[0], mode_count, 3), dtype=np.float32)
        else:
            psi = np.asarray(sample_psi, dtype=np.float32)
            if psi.ndim == 2 and points.shape[0] == 1:
                psi = psi.reshape((1, psi.shape[0], 3))
            if psi.ndim != 3 or psi.shape != (points.shape[0], mode_count, 3):
                raise ValueError(
                    "sample_psi must have shape [sample_count, mode_count, 3], "
                    f"got {psi.shape} for {points.shape[0]} sample points and {mode_count} modes"
                )

        self.sample_points = np.array(points, dtype=np.float32, copy=True)
        self.sample_phi = np.array(phi, dtype=np.float32, copy=True)
        self.sample_psi = np.array(psi, dtype=np.float32, copy=True)
        self.sample_mass = self._coerce_sample_mass(sample_mass, points.shape[0])

        self.mode_mass = self._coerce_mode_array(mode_mass, mode_count, 1.0, "mode_mass")
        self.mode_stiffness = self._coerce_mode_array(mode_stiffness, mode_count, 0.0, "mode_stiffness")
        self.mode_damping = self._coerce_mode_array(mode_damping, mode_count, 0.0, "mode_damping")
        self.mode_coupling_linear = self._coerce_coupling(mode_coupling_linear, mode_count, "mode_coupling_linear")
        self.mode_coupling_angular = self._coerce_coupling(mode_coupling_angular, mode_count, "mode_coupling_angular")
        self.mode_coupling_centrifugal = self._coerce_coupling_matrix(
            mode_coupling_centrifugal, mode_count, "mode_coupling_centrifugal"
        )
        self.mode_coupling_coriolis = self._coerce_coupling_pairs(
            mode_coupling_coriolis, mode_count, "mode_coupling_coriolis"
        )
        self.mode_coupling_quadratic = self._coerce_coupling_quadratic(
            mode_coupling_quadratic, mode_count, "mode_coupling_quadratic"
        )
        explicit_mass_matrix = self._coerce_mode_mass_matrix(mode_mass_matrix, mode_count)
        self.rigid_mass = None if rigid_mass is None else float(rigid_mass)
        self.rigid_com = self._coerce_fixed_shape(rigid_com, (3,), "rigid_com")
        self.rigid_inertia = self._coerce_fixed_shape(rigid_inertia, (3, 3), "rigid_inertia")

        lumped = None
        if self.sample_mass is not None and (
            any(
                value is None
                for value in (
                    self.mode_coupling_linear,
                    self.mode_coupling_angular,
                    self.mode_coupling_centrifugal,
                    self.mode_coupling_coriolis,
                    self.mode_coupling_quadratic,
                    self.rigid_mass,
                    self.rigid_com,
                    self.rigid_inertia,
                )
            )
            or (mode_mass is None and explicit_mass_matrix is None)
        ):
            lumped = _reduced_inertia_from_pairs(self.sample_points, self.sample_phi, mass=self.sample_mass)
            if self.rigid_mass is None:
                self.rigid_mass = lumped.rigid_mass
            if self.rigid_com is None:
                self.rigid_com = lumped.rigid_com
            if self.rigid_inertia is None:
                self.rigid_inertia = lumped.rigid_inertia
            if mode_count == 0:
                lumped = None

        # Explicit mode-pair integrals take precedence; the Coriolis integral and
        # modal mass matrix default to values derived from the same pair tensor.
        derived_mass_matrix = None
        if self.mode_coupling_quadratic is not None:
            quadratic64 = self.mode_coupling_quadratic.astype(np.float64)
            derived_mass_matrix = _mass_matrix_from_quadratic(quadratic64).astype(np.float32)
            if self.mode_coupling_coriolis is None:
                self.mode_coupling_coriolis = _coriolis_from_quadratic(quadratic64).astype(np.float32)
        elif lumped is not None:
            self.mode_coupling_quadratic = lumped.mode_coupling_quadratic
            derived_mass_matrix = lumped.mode_mass_matrix
        if lumped is not None:
            if self.mode_coupling_linear is None:
                self.mode_coupling_linear = lumped.mode_coupling_linear
            if self.mode_coupling_angular is None:
                self.mode_coupling_angular = lumped.mode_coupling_angular
            if self.mode_coupling_centrifugal is None:
                self.mode_coupling_centrifugal = lumped.mode_coupling_centrifugal
            if self.mode_coupling_coriolis is None:
                self.mode_coupling_coriolis = lumped.mode_coupling_coriolis

        self._mode_mass_coupling_ratio = _mass_coupling_ratio(
            derived_mass_matrix if derived_mass_matrix is not None else explicit_mass_matrix
        )
        if explicit_mass_matrix is not None:
            if mode_mass is None:
                self.mode_mass = np.array(np.diag(explicit_mass_matrix), dtype=np.float32, copy=True)
            else:
                diagonal = np.diag(explicit_mass_matrix).astype(np.float64)
                scale = max(float(np.max(np.abs(diagonal), initial=0.0)), 1.0e-30)
                if not np.allclose(diagonal, self.mode_mass, rtol=1.0e-5, atol=1.0e-6 * scale):
                    raise ValueError("mode_mass must match the diagonal of mode_mass_matrix")
            self.mode_mass_matrix = explicit_mass_matrix
        elif mode_mass is not None:
            self.mode_mass_matrix = None if derived_mass_matrix is None else np.diag(self.mode_mass)
            # Only lumped sample masses define the mass distribution here; explicit pair integrals
            # come with their own (possibly analytic) modal masses from the caller.
            if derived_mass_matrix is not None and lumped is not None and mode_coupling_quadratic is None:
                derived = np.diag(derived_mass_matrix).astype(np.float64)
                explicit = np.asarray(self.mode_mass, dtype=np.float64)
                deviation = np.abs(explicit - derived) / np.maximum(np.abs(derived), 1.0e-30)
                if derived.shape[0] > 0 and float(np.max(deviation)) > self.mode_coupling_tolerance:
                    warnings.warn(
                        "ModalBasis explicit mode_mass differs from the modal mass integrated from the samples "
                        f"(largest relative difference {float(np.max(deviation)):.3g} > mode_coupling_tolerance "
                        f"{self.mode_coupling_tolerance:.3g}). The inertia coupling integrals come from the samples, "
                        "so an inconsistent modal mass can make the reduced mass indefinite; omit mode_mass to use "
                        "the sampled value.",
                        stacklevel=2,
                    )
            if self._mode_mass_coupling_ratio > self.mode_coupling_tolerance:
                warnings.warn(
                    "ModalBasis modes are not mass-orthogonal under the sampled mass distribution "
                    f"(relative off-diagonal modal mass {self._mode_mass_coupling_ratio:.3g} > "
                    f"mode_coupling_tolerance {self.mode_coupling_tolerance:.3g}); the explicit diagonal "
                    "mode_mass discards this coupling. Pass mode_mass_matrix or omit mode_mass to keep it.",
                    stacklevel=2,
                )
        elif derived_mass_matrix is not None:
            self.mode_mass_matrix = derived_mass_matrix
            self.mode_mass = np.array(np.diag(derived_mass_matrix), dtype=np.float32, copy=True)
        else:
            self.mode_mass_matrix = None

        self.frame_points = np.zeros((0, 3), dtype=np.float32)
        """Body-local attachment frame origins [m], shape ``[frame_count, 3]``."""
        self.frame_phi = np.zeros((0, mode_count, 3), dtype=np.float32)
        """Attachment frame translational mode values [m per modal coordinate], shape ``[frame_count, mode_count, 3]``."""
        self.frame_psi = np.zeros((0, mode_count, 3), dtype=np.float32)
        """Attachment frame angular mode values [rad per modal coordinate], shape ``[frame_count, mode_count, 3]``."""
        self.frame_labels: list[str | None] = []
        """Attachment frame labels, length ``frame_count``; ``None`` for unlabeled frames."""
        if frame_points is not None:
            frames = np.asarray(frame_points, dtype=np.float32).reshape((-1, 3))
            if frame_phi is None:
                raise ValueError("frame_phi is required when frame_points is given")
            frame_count = int(frames.shape[0])
            phi_values = np.asarray(frame_phi, dtype=np.float32)
            if phi_values.shape != (frame_count, mode_count, 3):
                raise ValueError(f"frame_phi must have shape ({frame_count}, {mode_count}, 3), got {phi_values.shape}")
            psi_values = np.zeros_like(phi_values) if frame_psi is None else np.asarray(frame_psi, dtype=np.float32)
            if psi_values.shape != (frame_count, mode_count, 3):
                raise ValueError(f"frame_psi must have shape ({frame_count}, {mode_count}, 3), got {psi_values.shape}")
            labels = [None] * frame_count if frame_labels is None else list(frame_labels)
            if len(labels) != frame_count:
                raise ValueError(f"frame_labels must have length {frame_count}, got {len(labels)}")
            self.frame_points = np.array(frames, dtype=np.float32, copy=True)
            self.frame_phi = np.array(phi_values, dtype=np.float32, copy=True)
            self.frame_psi = np.array(psi_values, dtype=np.float32, copy=True)
            self.frame_labels = [None if label is None else str(label) for label in labels]
        elif frame_phi is not None or frame_psi is not None or frame_labels is not None:
            raise ValueError("frame_phi, frame_psi, and frame_labels require frame_points")

    @property
    def mode_mass_coupling_ratio(self) -> float:
        """Largest relative off-diagonal modal mass coupling.

        ``max_{i != j} |Mf_ij| / sqrt(Mf_ii Mf_jj)`` of the modal mass matrix
        integrated from ``mode_coupling_quadratic`` (or of an explicit
        ``mode_mass_matrix`` when no pair integral exists). ``0.0`` for fewer
        than two modes or when no full modal mass matrix is known. Mass-orthogonal
        modes give ``0.0``.
        """
        return self._mode_mass_coupling_ratio

    @property
    def mode_count(self) -> int:
        """Number of modal coordinates."""
        return int(self.sample_phi.shape[1])

    @property
    def sample_count(self) -> int:
        """Number of stored sample points."""
        return int(self.sample_points.shape[0])

    @property
    def frame_count(self) -> int:
        """Number of attachment frames."""
        return int(self.frame_points.shape[0])

    @staticmethod
    def _infer_mode_count(*arrays: Sequence[float] | np.ndarray | None) -> int:
        mode_count = 0
        for values in arrays:
            if values is None:
                continue
            count = int(np.asarray(values, dtype=np.float32).reshape((-1,)).shape[0])
            if mode_count not in (0, count):
                raise ValueError(f"mode property lengths disagree: {mode_count} and {count}")
            mode_count = count
        return mode_count

    @staticmethod
    def _coerce_mode_array(
        values: Sequence[float] | np.ndarray | None,
        mode_count: int,
        default: float,
        name: str,
    ) -> np.ndarray:
        if values is None:
            return np.full(mode_count, default, dtype=np.float32)
        array = np.asarray(values, dtype=np.float32).reshape((-1,))
        if array.shape[0] != mode_count:
            raise ValueError(f"{name} must have length {mode_count}, got {array.shape[0]}")
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _coerce_sample_mass(
        values: Sequence[float] | np.ndarray | None,
        sample_count: int,
    ) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32).reshape((-1,))
        if array.shape[0] != sample_count:
            raise ValueError(f"sample_mass must have length {sample_count}, got {array.shape[0]}")
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _coerce_coupling(
        values: Sequence[Sequence[float]] | np.ndarray | None,
        mode_count: int,
        name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (mode_count, 3):
            raise ValueError(f"{name} must have shape ({mode_count}, 3), got {array.shape}")
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _coerce_coupling_matrix(
        values: Sequence[Sequence[Sequence[float]]] | np.ndarray | None,
        mode_count: int,
        name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (mode_count, 3, 3):
            raise ValueError(f"{name} must have shape ({mode_count}, 3, 3), got {array.shape}")
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _coerce_coupling_pairs(
        values: Sequence[Sequence[Sequence[float]]] | np.ndarray | None,
        mode_count: int,
        name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (mode_count, mode_count, 3):
            raise ValueError(f"{name} must have shape ({mode_count}, {mode_count}, 3), got {array.shape}")
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _coerce_coupling_quadratic(
        values: np.ndarray | None,
        mode_count: int,
        name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (mode_count, mode_count, 3, 3):
            raise ValueError(f"{name} must have shape ({mode_count}, {mode_count}, 3, 3), got {array.shape}")
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _coerce_mode_mass_matrix(
        values: Sequence[Sequence[float]] | np.ndarray | None,
        mode_count: int,
    ) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (mode_count, mode_count):
            raise ValueError(f"mode_mass_matrix must have shape ({mode_count}, {mode_count}), got {array.shape}")
        scale = float(np.max(np.abs(array), initial=0.0))
        asymmetry = float(np.max(np.abs(array - array.T), initial=0.0))
        if asymmetry > 1.0e-5 * scale:
            raise ValueError(
                f"mode_mass_matrix must be symmetric; its largest asymmetry {asymmetry:.3g} exceeds 1e-5 of its "
                f"largest entry {scale:.3g}"
            )
        array = 0.5 * (array + array.T)
        if mode_count > 0:
            min_eigenvalue = float(np.linalg.eigvalsh(array)[0])
            if min_eigenvalue < -1.0e-6 * scale:
                warnings.warn(
                    f"mode_mass_matrix is indefinite (smallest eigenvalue {min_eigenvalue:.3g} kg); a modal mass "
                    "matrix must be positive semidefinite, and an indefinite one disables the implicit "
                    "frame/mode mass coupling of the elastic bodies that use it",
                    stacklevel=3,
                )
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _coerce_fixed_shape(
        values: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None,
        shape: tuple[int, ...],
        name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        return np.array(array, dtype=np.float32, copy=True)

    def copy(self) -> ModalBasis:
        """Return a deep copy of this basis."""
        return ModalBasis(
            sample_points=self.sample_points,
            sample_phi=self.sample_phi,
            sample_psi=self.sample_psi,
            sample_mass=self.sample_mass,
            mode_mass=self.mode_mass,
            mode_stiffness=self.mode_stiffness,
            mode_damping=self.mode_damping,
            mode_coupling_linear=self.mode_coupling_linear,
            mode_coupling_angular=self.mode_coupling_angular,
            mode_coupling_centrifugal=self.mode_coupling_centrifugal,
            mode_coupling_coriolis=self.mode_coupling_coriolis,
            label=self.label,
            interpolation_epsilon=self.interpolation_epsilon,
            mode_coupling_quadratic=self.mode_coupling_quadratic,
            mode_mass_matrix=self.mode_mass_matrix,
            rigid_mass=self.rigid_mass,
            rigid_com=self.rigid_com,
            rigid_inertia=self.rigid_inertia,
            mode_coupling_tolerance=self.mode_coupling_tolerance,
            frame_points=self.frame_points,
            frame_phi=self.frame_phi,
            frame_psi=self.frame_psi,
            frame_labels=self.frame_labels,
        )

    def find_frame(self, point: Sequence[float] | np.ndarray, tolerance: float = 1.0e-7) -> int:
        """Return the index of the attachment frame at ``point``, or ``-1`` if none exists.

        When several frames lie within ``tolerance``, the nearest one is
        returned, and the first of equally near frames.

        Args:
            point: Body-local point [m], shape ``[3]``.
            tolerance: Matching tolerance [m].

        Returns:
            The attachment frame index, or ``-1``.
        """
        if self.frame_count == 0:
            return -1
        query = np.asarray(point, dtype=np.float32).reshape((3,))
        dist2 = np.sum((self.frame_points - query.reshape((1, 3))) ** 2, axis=1)
        nearest = int(np.argmin(dist2))
        return nearest if float(dist2[nearest]) <= tolerance * tolerance else -1

    def add_frame(
        self,
        point: Sequence[float] | np.ndarray,
        phi: Sequence[Sequence[float]] | np.ndarray,
        *,
        psi: Sequence[Sequence[float]] | np.ndarray | None = None,
        label: str | None = None,
    ) -> int:
        """Add an attachment frame with prescribed mode values and return its index.

        Attachment frames carry no mass and are never interpolated. Joint
        endpoints that coincide with a frame use its ``phi`` and ``psi`` instead
        of the interpolated material sample, so a frame can represent a
        connection whose motion differs from the material point at the same
        location, such as a distributing (RBE3-style) interface.

        Args:
            point: Body-local frame origin [m], shape ``[3]``.
            phi: Translational mode values [m per modal coordinate], shape
                ``[mode_count, 3]``.
            psi: Angular mode values [rad per modal coordinate], shape
                ``[mode_count, 3]``. Defaults to zeros.
            label: Optional frame label.

        Returns:
            The attachment frame index.
        """
        point_value = np.asarray(point, dtype=np.float32).reshape((1, 3))
        phi_value = np.asarray(phi, dtype=np.float32)
        if phi_value.shape != (self.mode_count, 3):
            raise ValueError(f"phi must have shape ({self.mode_count}, 3), got {phi_value.shape}")
        psi_value = np.zeros_like(phi_value) if psi is None else np.asarray(psi, dtype=np.float32)
        if psi_value.shape != (self.mode_count, 3):
            raise ValueError(f"psi must have shape ({self.mode_count}, 3), got {psi_value.shape}")
        frame_index = self.frame_count
        self.frame_points = np.concatenate((self.frame_points, point_value), axis=0)
        self.frame_phi = np.concatenate((self.frame_phi, phi_value[None]), axis=0)
        self.frame_psi = np.concatenate((self.frame_psi, psi_value[None]), axis=0)
        self.frame_labels.append(None if label is None else str(label))
        return frame_index

    def find_sample(self, point: Sequence[float] | np.ndarray, tolerance: float = 1.0e-7) -> int:
        """Return the index of a matching sample point, or ``-1`` if none exists.

        Args:
            point: Body-local point [m], shape ``[3]``.
            tolerance: Matching tolerance [m].

        Returns:
            The local sample index, or ``-1``.
        """
        if self.sample_count == 0:
            return -1
        query = np.asarray(point, dtype=np.float32).reshape((3,))
        dist2 = np.sum((self.sample_points - query.reshape((1, 3))) ** 2, axis=1)
        nearest = int(np.argmin(dist2))
        return nearest if float(dist2[nearest]) <= tolerance * tolerance else -1

    def add_sample(
        self,
        point: Sequence[float] | np.ndarray,
        phi: Sequence[Sequence[float]] | np.ndarray | None = None,
        psi: Sequence[Sequence[float]] | np.ndarray | None = None,
        tolerance: float = 1.0e-7,
    ) -> int:
        """Add a body-local sample and return its local index.

        If ``point`` already exists, its existing index is returned. When ``phi``
        or ``psi`` is not provided, that value is interpolated from the current
        samples.

        Args:
            point: Body-local point [m], shape ``[3]``.
            phi: Translational mode values [m per modal coordinate], shape
                ``[mode_count, 3]``.
            psi: Angular mode values [rad per modal coordinate], shape
                ``[mode_count, 3]``.
            tolerance: Duplicate-point tolerance [m].

        Returns:
            The local sample index.
        """
        phi_batch = None if phi is None else np.asarray(phi, dtype=np.float32).reshape((1, self.mode_count, 3))
        psi_batch = None if psi is None else np.asarray(psi, dtype=np.float32).reshape((1, self.mode_count, 3))
        return int(
            self.add_samples(np.asarray(point, dtype=np.float32).reshape((1, 3)), phi_batch, psi_batch, tolerance)[0]
        )

    def add_samples(
        self,
        points: Sequence[Sequence[float]] | np.ndarray,
        phi: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        psi: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        tolerance: float = 1.0e-7,
    ) -> np.ndarray:
        """Add body-local samples in one batch and return their local indices.

        Duplicate points reuse the first matching sample. Missing mode values are
        interpolated from the samples that existed before this call. If the basis
        carries lumped sample masses, appended interpolation samples receive zero
        mass so they do not alter the basis's inertial integrals.

        Args:
            points: Body-local points [m], shape ``[sample_count, 3]``.
            phi: Translational mode values [m per modal coordinate], shape
                ``[sample_count, mode_count, 3]``.
            psi: Angular mode values [rad per modal coordinate], shape
                ``[sample_count, mode_count, 3]``.
            tolerance: Duplicate-point tolerance [m].

        Returns:
            Local sample indices, shape ``[sample_count]``.
        """
        queries = np.asarray(points, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape((1, 3))
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError(f"points must have shape [sample_count, 3], got {queries.shape}")
        if tolerance < 0.0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")

        query_count = int(queries.shape[0])
        expected_shape = (query_count, self.mode_count, 3)
        phi_values = None if phi is None else np.asarray(phi, dtype=np.float32)
        psi_values = None if psi is None else np.asarray(psi, dtype=np.float32)
        if phi_values is not None and phi_values.shape != expected_shape:
            raise ValueError(f"phi must have shape {expected_shape}, got {phi_values.shape}")
        if psi_values is not None and psi_values.shape != expected_shape:
            raise ValueError(f"psi must have shape {expected_shape}, got {psi_values.shape}")
        if query_count == 0:
            return np.zeros(0, dtype=np.int32)

        all_points = list(self.sample_points)
        result = np.empty(query_count, dtype=np.int32)
        new_query_indices: list[int] = []

        # A spatial hash keeps duplicate detection linear in the common case while
        # retaining the Euclidean tolerance used by find_sample().
        cell_size = tolerance if tolerance > 0.0 else 1.0
        buckets: dict[tuple[int, int, int], list[int]] = {}

        def cell(point: np.ndarray) -> tuple[int, int, int]:
            if tolerance == 0.0:
                return tuple(int(value) for value in point.view(np.int32))
            return tuple(int(value) for value in np.floor(point / cell_size))

        for sample_index, point in enumerate(all_points):
            buckets.setdefault(cell(point), []).append(sample_index)

        neighbor_offsets = (
            [(0, 0, 0)]
            if tolerance == 0.0
            else [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        )
        tolerance2 = tolerance * tolerance
        for query_index, query in enumerate(queries):
            query_cell = cell(query)
            nearest = -1
            nearest_dist2 = math.inf
            for dx, dy, dz in neighbor_offsets:
                candidate_cell = (query_cell[0] + dx, query_cell[1] + dy, query_cell[2] + dz)
                for sample_index in buckets.get(candidate_cell, ()):
                    delta = all_points[sample_index] - query
                    dist2 = float(np.dot(delta, delta))
                    if dist2 <= tolerance2 and dist2 < nearest_dist2:
                        nearest = sample_index
                        nearest_dist2 = dist2
            if nearest >= 0:
                result[query_index] = nearest
                continue

            sample_index = len(all_points)
            result[query_index] = sample_index
            new_query_indices.append(query_index)
            all_points.append(query)
            buckets.setdefault(query_cell, []).append(sample_index)

        if not new_query_indices:
            return result

        new_queries = queries[new_query_indices]
        interpolated_phi, interpolated_psi = self._evaluate_many(new_queries)
        new_phi = interpolated_phi if phi_values is None else phi_values[new_query_indices]
        new_psi = interpolated_psi if psi_values is None else psi_values[new_query_indices]
        self.sample_points = np.concatenate((self.sample_points, new_queries), axis=0)
        self.sample_phi = np.concatenate((self.sample_phi, new_phi), axis=0)
        self.sample_psi = np.concatenate((self.sample_psi, new_psi), axis=0)
        if self.sample_mass is not None:
            self.sample_mass = np.concatenate((self.sample_mass, np.zeros(len(new_query_indices), dtype=np.float32)))
        return result

    def _evaluate_many(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate interpolation points in bounded-memory NumPy batches."""
        query_count = int(points.shape[0])
        if self.mode_count == 0 or self.sample_count == 0:
            zeros = np.zeros((query_count, self.mode_count, 3), dtype=np.float32)
            return zeros, zeros.copy()

        phi = np.empty((query_count, self.mode_count, 3), dtype=np.float32)
        psi = np.empty_like(phi)
        chunk_size = max(1, min(query_count, 1_000_000 // self.sample_count))
        eps2 = self.interpolation_epsilon * self.interpolation_epsilon
        exact_tol2 = max(eps2, 1.0e-14)
        for start in range(0, query_count, chunk_size):
            stop = min(start + chunk_size, query_count)
            delta = self.sample_points[None, :, :] - points[start:stop, None, :]
            dist2 = np.einsum("qsc,qsc->qs", delta, delta)
            nearest = np.argmin(dist2, axis=1)
            exact = dist2[np.arange(stop - start), nearest] <= exact_tol2
            weights = (1.0 / (dist2 + eps2)).astype(np.float32)
            weights /= np.sum(weights, axis=1, keepdims=True)
            phi[start:stop] = np.einsum("qs,smc->qmc", weights, self.sample_phi)
            psi[start:stop] = np.einsum("qs,smc->qmc", weights, self.sample_psi)
            if np.any(exact):
                phi[start:stop][exact] = self.sample_phi[nearest[exact]]
                psi[start:stop][exact] = self.sample_psi[nearest[exact]]
        return phi, psi

    def evaluate(self, point: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse-distance interpolate the modal samples at a body-local point.

        Args:
            point: Body-local point [m], shape ``[3]``.

        Returns:
            The translational and angular mode values ``(phi, psi)`` at ``point``,
            each shape ``[mode_count, 3]``, in [m] and [rad] per modal coordinate.
        """
        if self.mode_count == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        if self.sample_count == 0:
            return np.zeros((self.mode_count, 3), dtype=np.float32), np.zeros((self.mode_count, 3), dtype=np.float32)

        query = np.asarray(point, dtype=np.float32).reshape((3,))
        delta = self.sample_points - query.reshape((1, 3))
        dist2 = np.sum(delta * delta, axis=1)
        nearest = int(np.argmin(dist2))
        exact_tol2 = max(self.interpolation_epsilon * self.interpolation_epsilon, 1.0e-14)
        if float(dist2[nearest]) <= exact_tol2:
            return (
                np.array(self.sample_phi[nearest], dtype=np.float32, copy=True),
                np.array(self.sample_psi[nearest], dtype=np.float32, copy=True),
            )

        eps2 = self.interpolation_epsilon * self.interpolation_epsilon
        weights = (1.0 / (dist2 + eps2)).astype(np.float32)
        weights /= np.sum(weights)
        return (
            np.einsum("s,smc->mc", weights, self.sample_phi).astype(np.float32),
            np.einsum("s,smc->mc", weights, self.sample_psi).astype(np.float32),
        )

    def mass_orthonormalize(self, *, tolerance: float = 1.0e-6) -> ModalBasis:
        """Return a copy whose modes are orthonormal under the modal mass matrix.

        With the Cholesky factor ``Mf = L L^T`` of :attr:`mode_mass_matrix`
        (``diag(mode_mass)`` when no full matrix is known), the new modes are
        ``Phi' = Phi L^-T``. Sample, attachment-frame, and inertia-integral
        values are transformed consistently, the new modal mass matrix is the
        identity, and the physical displacement field is unchanged when the
        modal coordinates are mapped as ``q' = L^T q``. Rigid mass properties and
        sample masses are unchanged.

        :class:`ModalBasis` stores diagonal stiffness and damping, while the
        transformed matrices are ``L^-1 K L^-T`` and ``L^-1 D L^-T``. These stay
        diagonal only when ``Mf`` is diagonal (a pure rescaling) or when the
        stiffness and damping are proportional to ``Mf`` (including zero), so
        any other combination raises :class:`ValueError`. The utility is
        optional: the solver consumes a full :attr:`mode_mass_matrix` directly.

        A :class:`UserWarning` is emitted when the modal coordinates change,
        because initial ``mode_q``/``mode_qd`` values and any stored modal
        trajectories must then be mapped with ``L^T``.

        Args:
            tolerance: Relative tolerance on the off-diagonal part of the
                transformed stiffness and damping, measured against their
                largest diagonal entry, and on the deviation of ``L`` from the
                identity used to decide whether the coordinates change.

        Returns:
            A new mass-orthonormal basis. ``L`` is
            ``numpy.linalg.cholesky(mode_mass_matrix)`` of this basis.
        """
        mode_count = self.mode_count
        mass_matrix = (
            np.diag(self.mode_mass.astype(np.float64))
            if self.mode_mass_matrix is None
            else np.asarray(self.mode_mass_matrix, dtype=np.float64)
        )
        mass_matrix = 0.5 * (mass_matrix + mass_matrix.T)
        try:
            factor = np.linalg.cholesky(mass_matrix) if mode_count > 0 else np.zeros((0, 0))
        except np.linalg.LinAlgError as exc:
            raise ValueError("mass_orthonormalize requires a positive-definite modal mass matrix") from exc
        # transform[i, j] = (L^-T)[i, j], so phi'_j = sum_i phi_i transform[i, j].
        transform = np.linalg.inv(factor).T if mode_count > 0 else np.zeros((0, 0))

        def congruent(values: np.ndarray, name: str) -> np.ndarray:
            matrix = transform.T @ np.diag(values.astype(np.float64)) @ transform
            diagonal = np.diag(matrix).copy()
            scale = max(float(np.max(np.abs(diagonal), initial=0.0)), np.finfo(np.float64).tiny)
            off_diagonal = float(np.max(np.abs(matrix - np.diag(diagonal)), initial=0.0))
            if off_diagonal > tolerance * scale:
                raise ValueError(
                    f"mass_orthonormalize would make {name} non-diagonal (relative off-diagonal "
                    f"{off_diagonal / scale:.3g} > tolerance {tolerance:.3g}); the diagonal {name} of "
                    "ModalBasis cannot represent it. Use the full mode_mass_matrix instead."
                )
            return diagonal

        stiffness = congruent(self.mode_stiffness, "mode_stiffness")
        damping = congruent(self.mode_damping, "mode_damping")

        def modes(values: np.ndarray) -> np.ndarray:
            return np.einsum("smc,mn->snc", values.astype(np.float64), transform).astype(np.float32)

        def single(values: np.ndarray | None) -> np.ndarray | None:
            if values is None:
                return None
            return np.einsum("m...,mn->n...", values.astype(np.float64), transform).astype(np.float32)

        def pairs(values: np.ndarray | None) -> np.ndarray | None:
            if values is None:
                return None
            return np.einsum("ab...,ai,bj->ij...", values.astype(np.float64), transform, transform).astype(np.float32)

        if mode_count > 0 and float(np.max(np.abs(factor - np.eye(mode_count)))) > tolerance:
            warnings.warn(
                "ModalBasis.mass_orthonormalize rescales the modal coordinates (q' = L^T q); map initial "
                "mode_q/mode_qd and stored modal trajectories accordingly.",
                stacklevel=2,
            )

        return ModalBasis(
            sample_points=self.sample_points,
            sample_phi=modes(self.sample_phi),
            sample_psi=modes(self.sample_psi),
            sample_mass=self.sample_mass,
            mode_mass=np.ones(mode_count, dtype=np.float32),
            mode_stiffness=stiffness,
            mode_damping=damping,
            mode_coupling_linear=single(self.mode_coupling_linear),
            mode_coupling_angular=single(self.mode_coupling_angular),
            mode_coupling_centrifugal=single(self.mode_coupling_centrifugal),
            mode_coupling_coriolis=pairs(self.mode_coupling_coriolis),
            label=self.label,
            interpolation_epsilon=self.interpolation_epsilon,
            mode_coupling_quadratic=pairs(self.mode_coupling_quadratic),
            mode_mass_matrix=np.eye(mode_count, dtype=np.float32),
            rigid_mass=self.rigid_mass,
            rigid_com=self.rigid_com,
            rigid_inertia=self.rigid_inertia,
            mode_coupling_tolerance=self.mode_coupling_tolerance,
            frame_points=self.frame_points,
            frame_phi=modes(self.frame_phi),
            frame_psi=modes(self.frame_psi),
            frame_labels=self.frame_labels,
        )


_NEIGHBOR_CANDIDATE_EXTRA = 8
"""Extra k-d tree candidates per point beyond ``k + 1``, so that distance ties at the ``k``-th neighbor are seen."""

_NEIGHBOR_BRUTE_FORCE_LIMIT = 2048
"""Largest point count whose neighbors :func:`_nearest_neighbors` finds by brute force."""


def _nearest_neighbors(points: np.ndarray, k: int) -> np.ndarray:
    """Return the ``k`` nearest other points of every point, shape ``[point_count, k]``.

    Up to :data:`_NEIGHBOR_BRUTE_FORCE_LIMIT` points, or without SciPy, a
    brute-force ``O(N^2 log N)`` search sorts all squared distances, which
    keeps the historical results, including how ties at the ``k``-th
    neighbor of regular grids are broken. Larger point sets use a k-d tree
    (:class:`scipy.spatial.cKDTree`) in ``O(N log N)`` and order neighbors by
    squared distance, then by index, so ties are broken deterministically.

    Args:
        points: Point positions [m], shape ``[point_count, 3]``, float64.
        k: Neighbor count, at most ``point_count - 1``.

    Returns:
        Neighbor indices, nearest first, excluding the point itself.
    """
    count = points.shape[0]
    neighbors = np.empty((count, k), dtype=np.int64)
    tree = None
    if count > _NEIGHBOR_BRUTE_FORCE_LIMIT:
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(points)
        except ImportError:
            tree = None
    if tree is None:
        for point in range(count):
            delta = points - points[point]
            neighbors[point] = np.argsort(np.sum(delta * delta, axis=1))[1 : k + 1]
        return neighbors

    def ordered(point: int, candidates: np.ndarray) -> np.ndarray:
        candidates = candidates[candidates != point]
        delta = points[candidates] - points[point]
        dist2 = np.sum(delta * delta, axis=1)
        return candidates[np.lexsort((candidates, dist2))[:k]]

    query_count = min(count, k + 1 + _NEIGHBOR_CANDIDATE_EXTRA)
    _, candidates = tree.query(points, k=query_count)
    candidates = np.asarray(candidates, dtype=np.int64).reshape((count, query_count))
    for point in range(count):
        row = candidates[point]
        chosen = ordered(point, row)
        if query_count < count and k > 0:
            # Every point outside the candidates is at least as far as the farthest candidate; when that
            # candidate ties with the k-th neighbor, gather all tied points with a radius query.
            delta = points[row] - points[point]
            kth_delta = points[chosen[-1]] - points[point]
            kth = float(kth_delta @ kth_delta)
            if float(np.max(np.sum(delta * delta, axis=1))) <= kth * (1.0 + 1.0e-9):
                radius = math.sqrt(kth) * (1.0 + 1.0e-6) + np.finfo(np.float64).tiny
                chosen = ordered(point, np.asarray(tree.query_ball_point(points[point], radius), dtype=np.int64))
        neighbors[point] = chosen
    return neighbors


def _estimate_sample_psi(
    sample_points: Sequence[Sequence[float]] | np.ndarray,
    sample_phi: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    neighbor_count: int = 12,
    regularization: float = 1.0e-9,
) -> np.ndarray:
    """Estimate angular mode samples from a translational displacement point cloud.

    The body-local rotation of the material frame is the skew part of the
    displacement gradient, i.e. the half-curl of the mode shape:
    ``psi_i = 0.5 * curl(phi_i)``. For generators that only carry sampled (or
    nodal) displacement values, with no analytic shape or mesh connectivity, this
    fits a local affine displacement gradient over each sample's nearest
    neighbors by least squares and returns the axial vector of its skew part.

    Args:
        sample_points: Body-local sample points [m], shape ``[sample_count, 3]``.
        sample_phi: Translational mode samples [m per modal coordinate], shape
            ``[sample_count, mode_count, 3]``.
        neighbor_count: Number of nearest neighbors used for each local fit.
        regularization: Relative Tikhonov term added to the normal equations so
            that degenerate (collinear or coplanar) neighborhoods stay stable;
            directions the geometry cannot resolve fall back toward zero.

    Returns:
        Angular mode samples [rad per modal coordinate], shape
        ``[sample_count, mode_count, 3]``.
    """
    points = np.asarray(sample_points, dtype=np.float64)
    phi = np.asarray(sample_phi, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"sample_points must have shape [sample_count, 3], got {points.shape}")
    sample_count = points.shape[0]
    if phi.ndim != 3 or phi.shape[0] != sample_count or phi.shape[2] != 3:
        raise ValueError(
            "sample_phi must have shape [sample_count, mode_count, 3], "
            f"got {phi.shape} for {sample_count} sample points"
        )

    mode_count = phi.shape[1]
    psi = np.zeros((sample_count, mode_count, 3), dtype=np.float32)
    if sample_count < 4 or mode_count == 0:
        # An affine 3D gradient needs at least four non-degenerate samples.
        return psi

    k = min(int(neighbor_count), sample_count - 1)
    neighbors = _nearest_neighbors(points, k)
    degenerate = 0
    for s in range(sample_count):
        order = neighbors[s]
        dx = points[order] - points[s]  # [k, 3]
        gram = dx.T @ dx  # [3, 3]
        scale = np.trace(gram) / 3.0
        if scale <= 0.0:
            continue
        if np.linalg.svd(dx, compute_uv=False)[-1] <= 1.0e-6 * math.sqrt(scale):
            degenerate += 1
        gram_inv = np.linalg.inv(gram + regularization * scale * np.eye(3))
        du = phi[order] - phi[s]  # [k, mode, 3]
        # grad[m] = (du_mᵀ dx)(dxᵀ dx)⁻¹  ->  grad[m, i, j] = d(phi_m)_i / dx_j
        grad = np.einsum("kmi,kj->mij", du, dx) @ gram_inv
        psi[s, :, 0] = 0.5 * (grad[:, 2, 1] - grad[:, 1, 2])
        psi[s, :, 1] = 0.5 * (grad[:, 0, 2] - grad[:, 2, 0])
        psi[s, :, 2] = 0.5 * (grad[:, 1, 0] - grad[:, 0, 1])

    if degenerate:
        warnings.warn(
            f"derive_psi: {degenerate}/{sample_count} sample neighborhoods are rank-deficient "
            "(collinear or coplanar); the unresolved rotation directions are damped toward zero. "
            "Provide a fuller 3D sample distribution for accurate angular coupling.",
            stacklevel=2,
        )
    return psi


class ModalGeneratorSampled:
    """Build a :class:`ModalBasis` from externally supplied sampled modes.

    .. experimental::

    The modes are used as supplied, in any normalization. Without
    ``sample_mass`` the basis has no inertia coupling integrals and no rigid
    mass properties.

    Args:
        sample_points: Body-local sample points [m], shape ``[sample_count, 3]``.
        sample_phi: Translational mode samples [m per modal coordinate], shape
            ``[sample_count, mode_count, 3]``.
        sample_psi: Optional angular mode samples [rad per modal coordinate],
            shape ``[sample_count, mode_count, 3]``. Pass these when known;
            otherwise set ``derive_psi`` to estimate them from ``sample_phi``.
        mode_mass: Optional per-mode modal mass [kg]. Defaults to the diagonal
            of the sampled modal mass matrix when ``sample_mass`` is given,
            else ``1``.
        mode_stiffness: Optional per-mode modal stiffness [N/m].
        mode_damping: Optional per-mode modal damping [N s/m].
        derive_psi: When ``True`` and ``sample_psi`` is not given, estimate the
            angular samples from the displacement gradient of ``sample_phi``.
        label: Optional basis label.
        sample_mass: Optional per-sample lumped mass [kg], shape
            ``[sample_count]``, forwarded to :class:`ModalBasis` to compute the
            modal mass matrix, the inertia coupling integrals, and the rigid
            mass properties.
    """

    def __init__(
        self,
        sample_points: Sequence[Sequence[float]] | np.ndarray,
        sample_phi: Sequence[Sequence[Sequence[float]]] | np.ndarray,
        sample_psi: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        mode_mass: Sequence[float] | np.ndarray | None = None,
        mode_stiffness: Sequence[float] | np.ndarray | None = None,
        mode_damping: Sequence[float] | np.ndarray | None = None,
        derive_psi: bool = False,
        label: str | None = None,
        *,
        sample_mass: Sequence[float] | np.ndarray | None = None,
    ):
        self.sample_points = np.asarray(sample_points, dtype=np.float32)
        self.sample_phi = np.asarray(sample_phi, dtype=np.float32)
        self.sample_psi = None if sample_psi is None else np.asarray(sample_psi, dtype=np.float32)
        self.mode_mass = None if mode_mass is None else np.asarray(mode_mass, dtype=np.float32)
        self.mode_stiffness = None if mode_stiffness is None else np.asarray(mode_stiffness, dtype=np.float32)
        self.mode_damping = None if mode_damping is None else np.asarray(mode_damping, dtype=np.float32)
        self.derive_psi = bool(derive_psi)
        self.label = label
        self.sample_mass = None if sample_mass is None else np.asarray(sample_mass, dtype=np.float32)

    def build(self) -> ModalBasis:
        """Build the sampled modal basis."""
        sample_psi = self.sample_psi
        if sample_psi is None and self.derive_psi:
            sample_psi = _estimate_sample_psi(self.sample_points, self.sample_phi)
        return ModalBasis(
            sample_points=self.sample_points,
            sample_phi=self.sample_phi,
            sample_psi=sample_psi,
            sample_mass=self.sample_mass,
            mode_mass=self.mode_mass,
            mode_stiffness=self.mode_stiffness,
            mode_damping=self.mode_damping,
            label=self.label,
        )


# ``damping_projection`` stays a closed set of strings rather than an enum: it is a setup-time
# option shared by ModalGeneratorCraigBampton and ModalGeneratorFEM, and the categorical options of
# the modal generator family (ModalGeneratorBeam.Mode and .Boundary) are string-valued as well. The
# Literal annotation documents and type-checks the allowed values.
_DAMPING_PROJECTIONS = ("diagonal", "error")


def _normalize_eigensolver(
    eigensolver: str | modal_sparse.ModalEigensolver | modal_sparse.EigenSolverBackend,
) -> str | modal_sparse.EigenSolverBackend:
    """Return ``"auto"``, ``"dense"``, or ``"arpack"`` for a built-in selection, or the backend to dispatch to.

    ``"subspace"`` and :attr:`~newton.ModalEigensolver.SUBSPACE` resolve to a
    default :class:`~newton.SubspaceIterationBackend`.
    """
    kind = modal_sparse.eigensolver_kind(eigensolver)
    if not isinstance(kind, modal_sparse.ModalEigensolver):
        return kind
    if kind is modal_sparse.ModalEigensolver.SUBSPACE:
        return modal_sparse.resolve_eigensolver(kind)
    return kind.value


def _validate_damping_projection(damping_projection: str) -> str:
    value = str(damping_projection)
    if value not in _DAMPING_PROJECTIONS:
        raise ValueError(f"damping_projection must be one of {_DAMPING_PROJECTIONS}, got '{value}'")
    return value


def _project_modal_damping(generator: Any, modal_damping: np.ndarray, source: str) -> np.ndarray:
    """Reduce a projected modal damping matrix to its diagonal.

    Records the off-diagonal ratio in ``generator.damping_off_diagonal_ratio``
    before applying ``generator.damping_projection``.

    Args:
        generator: Generator providing ``damping_projection`` and
            ``damping_coupling_tolerance``.
        modal_damping: Damping matrix projected onto mass-normalized modes
            [N s/m per modal coordinate], shape ``[mode_count, mode_count]``.
        source: Generator name used in diagnostics.

    Returns:
        The non-negative diagonal modal damping [N s/m per modal coordinate].
    """
    tolerance = generator.damping_coupling_tolerance
    modal_damping = 0.5 * (modal_damping + modal_damping.T)
    diagonal = np.diag(modal_damping).copy()
    generator.damping_off_diagonal_ratio = 0.0
    if diagonal.shape[0] == 0:
        return diagonal
    off_diagonal = modal_damping - np.diag(diagonal)
    diagonal_scale = max(float(np.max(np.abs(diagonal))), np.finfo(np.float64).eps)
    ratio = float(np.max(np.abs(off_diagonal))) / diagonal_scale
    generator.damping_off_diagonal_ratio = ratio
    if ratio > tolerance:
        message = (
            f"{source} damping is not diagonal in the retained elastic modal basis "
            f"(off-diagonal ratio {ratio:.3g} exceeds damping_coupling_tolerance {tolerance:.3g})"
        )
        if generator.damping_projection == "error":
            raise ValueError(message)
        warnings.warn(f"{message}; discarding the off-diagonal terms", stacklevel=3)
    return _clamp_modal_diagonal(diagonal, tolerance * diagonal_scale, f"{source} damping", stacklevel=4)


_MODE_SIGN_TIE_TOLERANCE = 1.0e-6
"""Relative magnitude within which mode entries tie for the largest one in :func:`_orient_modes` [dimensionless]."""


def _orient_modes(modes: np.ndarray) -> np.ndarray:
    """Return ``modes`` with each column's sign chosen so that its largest-magnitude entry is positive.

    Entries within :data:`_MODE_SIGN_TIE_TOLERANCE` of the largest magnitude
    tie, and the first of them decides, so that eigensolvers whose results
    differ by roundoff choose the same sign, also for symmetric modes whose
    largest entries have equal magnitude and opposite signs.
    """
    if modes.shape[1] == 0:
        return modes
    magnitude = np.abs(modes)
    peak = np.max(magnitude, axis=0)
    pivot = np.argmax(magnitude >= (1.0 - _MODE_SIGN_TIE_TOLERANCE) * peak, axis=0)
    sign = np.where(modes[pivot, np.arange(modes.shape[1])] < 0.0, -1.0, 1.0)
    return modes * sign


def _clamp_modal_diagonal(values: np.ndarray, tolerance: float, name: str, stacklevel: int = 3) -> np.ndarray:
    """Clamp negative projected modal values to zero, warning beyond the roundoff ``tolerance``."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] > 0 and float(np.min(values)) < -tolerance:
        warnings.warn(
            f"{name} has negative modal diagonal entries (minimum {float(np.min(values)):.3g}); clamping them to zero",
            stacklevel=stacklevel,
        )
    return np.maximum(values, 0.0)


class ModalGeneratorCraigBampton:
    """Build a Newton modal basis from Craig-Bampton reduced matrices.

    .. experimental::

    The generator accepts already-loaded arrays. Input coordinates must contain
    six DOFs per interface in ``[tx, ty, tz, rx, ry, rz]`` order, followed by
    any fixed-interface modes. A setup-time generalized eigensolve separates the
    six global rigid-body modes and transforms all remaining coordinates into
    ``reduced_dof_count - 6`` mass-normalized elastic modes. The eigensolve
    diagonalizes the reduced mass and stiffness matrices exactly, so importing
    dense reduced matrices does not add matrix bandwidth to Newton's runtime
    elastic solve.

    Damping is not part of the eigenproblem: ``damping_matrix`` is only projected
    onto the elastic modes. Because :class:`ModalBasis` stores diagonal modal
    damping, the off-diagonal part of the projected damping matrix is discarded
    (``damping_projection="diagonal"``, with a warning above
    ``damping_coupling_tolerance``) or rejected (``damping_projection="error"``).
    The projection is exact only for damping that the undamped modes
    diagonalize, such as modal or Rayleigh damping. :meth:`build` records the
    measured coupling in ``damping_off_diagonal_ratio``.

    The reduced matrices determine the exact modal mass matrix, the rigid mass
    properties, and the rigid-to-modal linear and angular inertia coupling. They
    do not contain the higher-order centrifugal, Coriolis, and mode-pair
    integrals used by Newton's nonlinear floating-frame coupling. Supply
    ``sample_mass`` to integrate those terms with lumped masses over the
    recovered ``sample_phi``; the exact reduced-matrix values keep precedence.
    Without ``sample_mass`` they remain unavailable, so the body's rotational
    inertia stays frozen at its rigid value (no deformation-dependent inertia).

    Each interface becomes a named attachment frame of the returned basis
    (``frame_points``, ``frame_phi``, ``frame_psi``, ``frame_labels``) whose mode
    values are the interface DOF rows of the modal matrix; joint endpoints placed
    at an interface position use these values. The material samples used for
    contacts and rendering depend on how the interface relates to the recovery
    samples:

    - No recovery sample within ``interface_match_tolerance``: the interface is
      also appended as a massless material sample with the interface values.
    - A coincident recovery sample whose recovered translation matches the
      interface translation within ``interface_recovery_tolerance`` (relative
      to the largest modal matrix entry), as for a direct (RBE2-style)
      interface node: the sample keeps its recovered ``phi`` and takes the
      exact interface rotation as its ``psi``.
    - A coincident recovery sample whose recovered motion differs, as for a
      distributing (RBE3-style) interface whose reference point lies on a
      physical node: the sample keeps its recovered motion unchanged and a
      warning is emitted. Joints use the interface frame; contacts and
      rendering use the physical sample.

    :meth:`build` records the chosen sample index of each interface in
    ``interface_sample_indices`` and the relative deviation of each coincident
    interface in ``interface_recovery_deviation``.

    Args:
        interface_positions: Body-local interface-frame positions [m], shape
            ``[interface_count, 3]``. Their order defines the six-DOF blocks in
            the reduced matrices.
        mass_matrix: Craig-Bampton reduced mass matrix, shape
            ``[reduced_dof_count, reduced_dof_count]``. The first
            ``6 * interface_count`` rows and columns are the interface block.
            Dense or SciPy sparse; the reduced matrices are always densified.
        stiffness_matrix: Craig-Bampton reduced stiffness matrix, with the same
            shape and coordinate order as ``mass_matrix``.
        sample_points: Body-local recovery sample points [m], shape
            ``[sample_count, 3]``.
        recovery_matrix: Translational recovery matrix mapping reduced
            coordinates to interleaved sample displacements, shape
            ``[3 * sample_count, reduced_dof_count]``.
        damping_matrix: Optional Craig-Bampton reduced damping matrix, with the
            same shape and coordinate order as ``mass_matrix``.
        interface_names: Optional unique names for the interface frames. Defaults
            to ``interface_0``, ``interface_1``, and so on.
        rigid_mode_tolerance: Relative eigenvalue tolerance used to identify the
            six global rigid-body modes and reject additional zero-energy modes.
        damping_coupling_tolerance: Relative off-diagonal damping magnitude,
            measured against the largest diagonal entry, above which
            ``damping_projection`` warns or raises.
        label: Optional basis label.
        sample_mass: Optional lumped mass [kg] for each recovery sample point,
            shape ``[sample_count]``. Enables the centrifugal, Coriolis, and
            mode-pair inertia integrals of the returned basis. Interface frames
            appended as extra samples receive zero mass.
        damping_projection: How to reduce the projected modal damping matrix to
            the diagonal damping stored by :class:`ModalBasis`. ``"diagonal"``
            discards off-diagonal terms, warning when their relative magnitude
            exceeds ``damping_coupling_tolerance``; ``"error"`` raises instead.
        interface_match_tolerance: Distance [m] within which an interface
            position coincides with a recovery sample point.
        interface_recovery_tolerance: Largest relative deviation between the
            interface translation rows and the recovered translation of a
            coincident sample for which the interface is treated as direct
            (RBE2-style) rather than distributing.
    """

    def __init__(
        self,
        interface_positions: Sequence[Sequence[float]] | np.ndarray,
        mass_matrix: Sequence[Sequence[float]] | np.ndarray,
        stiffness_matrix: Sequence[Sequence[float]] | np.ndarray,
        sample_points: Sequence[Sequence[float]] | np.ndarray,
        recovery_matrix: Sequence[Sequence[float]] | np.ndarray,
        damping_matrix: Sequence[Sequence[float]] | np.ndarray | None = None,
        interface_names: Sequence[str] | None = None,
        rigid_mode_tolerance: float = 1.0e-8,
        damping_coupling_tolerance: float = 1.0e-5,
        label: str | None = None,
        *,
        sample_mass: Sequence[float] | np.ndarray | None = None,
        damping_projection: Literal["diagonal", "error"] = "diagonal",
        interface_match_tolerance: float = 1.0e-7,
        interface_recovery_tolerance: float = 1.0e-6,
    ):
        self.interface_positions = np.asarray(interface_positions, dtype=np.float64)
        if self.interface_positions.ndim != 2 or self.interface_positions.shape[1] != 3:
            raise ValueError(
                f"interface_positions must have shape [interface_count, 3], got {self.interface_positions.shape}"
            )
        self.interface_count = int(self.interface_positions.shape[0])
        if self.interface_count < 1:
            raise ValueError("At least one interface is required")

        if interface_names is None:
            names = tuple(f"interface_{i}" for i in range(self.interface_count))
        else:
            names = tuple(str(name) for name in interface_names)
            if len(names) != self.interface_count:
                raise ValueError(f"interface_names must have length {self.interface_count}, got {len(names)}")
            if any(not name for name in names):
                raise ValueError("interface_names cannot contain empty names")
            if len(set(names)) != len(names):
                raise ValueError("interface_names must be unique")
        self.interface_names = names

        mass = np.asarray(mass_matrix, dtype=np.float64)
        if mass.ndim != 2 or mass.shape[0] != mass.shape[1]:
            raise ValueError(f"mass_matrix must be square, got {mass.shape}")
        self.reduced_dof_count = int(mass.shape[0])
        self.interface_dof_count = 6 * self.interface_count
        if self.reduced_dof_count < self.interface_dof_count:
            raise ValueError(
                f"mass_matrix must provide at least {self.interface_dof_count} interface DOFs, "
                f"got {self.reduced_dof_count}"
            )
        self.mass_matrix = self._coerce_square_matrix(mass, self.reduced_dof_count, "mass_matrix")
        self.stiffness_matrix = self._coerce_square_matrix(stiffness_matrix, self.reduced_dof_count, "stiffness_matrix")
        self.damping_matrix = (
            None
            if damping_matrix is None
            else self._coerce_square_matrix(damping_matrix, self.reduced_dof_count, "damping_matrix")
        )

        self.sample_points = np.asarray(sample_points, dtype=np.float64)
        if self.sample_points.ndim != 2 or self.sample_points.shape[1] != 3:
            raise ValueError(f"sample_points must have shape [sample_count, 3], got {self.sample_points.shape}")
        if self.sample_points.shape[0] == 0:
            raise ValueError("At least one recovery sample point is required")
        self.recovery_matrix = np.asarray(recovery_matrix, dtype=np.float64)
        expected_recovery_shape = (3 * self.sample_points.shape[0], self.reduced_dof_count)
        if self.recovery_matrix.shape != expected_recovery_shape:
            raise ValueError(
                f"recovery_matrix must have shape {expected_recovery_shape}, got {self.recovery_matrix.shape}"
            )
        self.sample_mass = None
        if sample_mass is not None:
            self.sample_mass = np.asarray(sample_mass, dtype=np.float64).reshape((-1,))
            if self.sample_mass.shape[0] != self.sample_points.shape[0]:
                raise ValueError(
                    f"sample_mass must have length {self.sample_points.shape[0]}, got {self.sample_mass.shape[0]}"
                )
            if np.any(self.sample_mass < 0.0):
                raise ValueError("sample_mass must be non-negative")

        self.rigid_mode_tolerance = float(rigid_mode_tolerance)
        if self.rigid_mode_tolerance <= 0.0:
            raise ValueError(f"rigid_mode_tolerance must be positive, got {self.rigid_mode_tolerance}")
        self.damping_coupling_tolerance = float(damping_coupling_tolerance)
        if self.damping_coupling_tolerance < 0.0:
            raise ValueError(f"damping_coupling_tolerance must be non-negative, got {self.damping_coupling_tolerance}")
        self.damping_projection = _validate_damping_projection(damping_projection)
        self.interface_match_tolerance = float(interface_match_tolerance)
        if self.interface_match_tolerance < 0.0:
            raise ValueError(f"interface_match_tolerance must be non-negative, got {self.interface_match_tolerance}")
        self.interface_recovery_tolerance = float(interface_recovery_tolerance)
        if self.interface_recovery_tolerance < 0.0:
            raise ValueError(
                f"interface_recovery_tolerance must be non-negative, got {self.interface_recovery_tolerance}"
            )
        self.label = label

        self.fixed_interface_mode_count = self.reduced_dof_count - self.interface_dof_count
        self.discarded_mode_count = 0
        self.rigid_mode_count = 6
        self.mode_count = self.reduced_dof_count - self.rigid_mode_count
        self.mass = 0.0
        self.com = np.zeros(3, dtype=np.float64)
        self.inertia = np.zeros((3, 3), dtype=np.float64)
        self.spatial_mass = np.zeros((6, 6), dtype=np.float64)
        self.frequencies = np.zeros(0, dtype=np.float64)
        self.rigid_matrix = np.zeros((self.reduced_dof_count, 6), dtype=np.float64)
        self.modal_matrix = np.zeros((self.reduced_dof_count, self.mode_count), dtype=np.float64)
        self.damping_off_diagonal_ratio = 0.0
        self.interface_sample_indices: dict[str, int] = {}
        self.interface_recovery_deviation: dict[str, float] = {}

    @staticmethod
    def _coerce_square_matrix(
        matrix: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray,
        size: int,
        name: str,
    ) -> np.ndarray:
        """Return the dense symmetric part of a square matrix; reduced sparse matrices are densified."""
        return modal_sparse.coerce_symmetric_matrix(matrix, size, name, keep_sparse=False)

    @staticmethod
    def _skew(value: np.ndarray) -> np.ndarray:
        x, y, z = value
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)

    def _rigid_map(self) -> np.ndarray:
        rigid = np.zeros((self.reduced_dof_count, 6), dtype=np.float64)
        for interface, position in enumerate(self.interface_positions):
            start = 6 * interface
            rigid[start : start + 3, :3] = np.eye(3)
            rigid[start : start + 3, 3:] = -self._skew(position)
            rigid[start + 3 : start + 6, 3:] = np.eye(3)
        return rigid

    def _extract_rigid_properties(self, spatial_mass: np.ndarray) -> None:
        translation_mass = spatial_mass[:3, :3]
        mass = float(np.trace(translation_mass) / 3.0)
        scale = max(float(np.max(np.abs(spatial_mass))), 1.0)
        if mass <= 0.0:
            raise ValueError(f"The Craig-Bampton mass matrix has non-positive rigid mass {mass}")
        if np.max(np.abs(translation_mass - mass * np.eye(3))) > 1.0e-7 * scale:
            raise ValueError("The Craig-Bampton mass matrix does not contain a physical rigid translational block")

        com_skew = -spatial_mass[:3, 3:] / mass
        if np.max(np.abs(com_skew + com_skew.T)) > 1.0e-7 * scale:
            raise ValueError(
                "The Craig-Bampton mass matrix does not contain a physical rigid translation-rotation block"
            )
        com = np.array([com_skew[2, 1], com_skew[0, 2], com_skew[1, 0]], dtype=np.float64)

        inertia_origin = 0.5 * (spatial_mass[3:, 3:] + spatial_mass[3:, 3:].T)
        parallel_axis = mass * ((com @ com) * np.eye(3) - np.outer(com, com))
        inertia = inertia_origin - parallel_axis
        if np.min(np.linalg.eigvalsh(inertia)) < -1.0e-7 * scale:
            raise ValueError("The Craig-Bampton mass matrix produces a non-positive rigid inertia")

        self.mass = mass
        self.com = com
        self.inertia = inertia

    def build(self) -> ModalBasis:
        """Build the mass-normalized elastic modal basis."""
        mass = self.mass_matrix
        stiffness = self.stiffness_matrix
        damping = self.damping_matrix
        rigid = self._rigid_map()

        stiffness_scale = max(float(np.linalg.norm(stiffness, ord=np.inf)), 1.0)
        rigid_scale = max(float(np.linalg.norm(rigid, ord=np.inf)), 1.0)
        rigid_stiffness_ratio = float(np.linalg.norm(stiffness @ rigid, ord=np.inf)) / (stiffness_scale * rigid_scale)
        if rigid_stiffness_ratio > 1.0e-7:
            raise ValueError(
                "The Craig-Bampton stiffness matrix does not preserve the six rigid-body modes "
                f"(relative residual {rigid_stiffness_ratio:.3g})"
            )
        if damping is not None:
            damping_scale = max(float(np.linalg.norm(damping, ord=np.inf)), 1.0)
            rigid_damping_ratio = float(np.linalg.norm(damping @ rigid, ord=np.inf)) / (damping_scale * rigid_scale)
            if rigid_damping_ratio > 1.0e-7:
                raise ValueError(
                    "The Craig-Bampton damping matrix does not preserve the six rigid-body modes "
                    f"(relative residual {rigid_damping_ratio:.3g})"
                )

        try:
            chol = np.linalg.cholesky(mass)
        except np.linalg.LinAlgError as exc:
            raise ValueError("The Craig-Bampton mass matrix must be positive definite") from exc

        inv_chol_stiffness = np.linalg.solve(chol, stiffness)
        standard = np.linalg.solve(chol, inv_chol_stiffness.T).T
        standard = 0.5 * (standard + standard.T)
        eigenvalues, eigenvectors = np.linalg.eigh(standard)
        eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        rigid_threshold = self.rigid_mode_tolerance * eigenvalue_scale
        if float(np.max(np.abs(eigenvalues[: self.rigid_mode_count]))) > rigid_threshold:
            raise ValueError(
                "The Craig-Bampton system does not contain exactly six global rigid-body modes "
                f"within relative tolerance {self.rigid_mode_tolerance:.3g}"
            )
        if self.mode_count > 0 and float(eigenvalues[self.rigid_mode_count]) <= rigid_threshold:
            raise ValueError(
                "The Craig-Bampton system contains additional zero-energy modes beyond the six global rigid-body modes"
            )

        modal_matrix = np.linalg.solve(chol.T, eigenvectors[:, self.rigid_mode_count :])
        for mode in range(self.mode_count):
            pivot = int(np.argmax(np.abs(modal_matrix[:, mode])))
            if modal_matrix[pivot, mode] < 0.0:
                modal_matrix[:, mode] *= -1.0

        modal_mass = modal_matrix.T @ mass @ modal_matrix
        modal_stiffness = modal_matrix.T @ stiffness @ modal_matrix
        mode_mass = np.diag(modal_mass)
        mode_stiffness = _clamp_modal_diagonal(np.diag(modal_stiffness), rigid_threshold, "Craig-Bampton stiffness")

        if damping is None:
            mode_damping = np.zeros(self.mode_count, dtype=np.float64)
            self.damping_off_diagonal_ratio = 0.0
        else:
            mode_damping = _project_modal_damping(self, modal_matrix.T @ damping @ modal_matrix, "Craig-Bampton")

        sample_rigid = np.hstack(
            (
                np.tile(np.eye(3), (self.sample_points.shape[0], 1)),
                np.vstack([-self._skew(point) for point in self.sample_points]),
            )
        )
        recovery_scale = max(float(np.max(np.abs(sample_rigid))), 1.0)
        recovery_rigid_error = float(np.max(np.abs(self.recovery_matrix @ rigid - sample_rigid)))
        if recovery_rigid_error > 1.0e-6 * recovery_scale:
            raise ValueError(
                "recovery_matrix is inconsistent with rigid motion of sample_points "
                f"(maximum error {recovery_rigid_error:.3g})"
            )

        recovered_modes = self.recovery_matrix @ modal_matrix
        sample_phi = np.transpose(
            recovered_modes.reshape((self.sample_points.shape[0], 3, self.mode_count)),
            (0, 2, 1),
        )
        sample_psi = _estimate_sample_psi(self.sample_points, sample_phi)
        basis_points = np.array(self.sample_points, dtype=np.float32, copy=True)
        basis_phi = np.array(sample_phi, dtype=np.float32, copy=True)
        basis_psi = np.array(sample_psi, dtype=np.float32, copy=True)
        basis_mass = None if self.sample_mass is None else np.array(self.sample_mass, dtype=np.float32, copy=True)

        self.interface_sample_indices = {}
        self.interface_recovery_deviation = {}
        frames: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        distributing: list[str] = []
        mode_scale = max(float(np.max(np.abs(modal_matrix), initial=0.0)), np.finfo(np.float64).eps)
        for interface, (name, point) in enumerate(zip(self.interface_names, self.interface_positions, strict=True)):
            start = 6 * interface
            interface_phi = modal_matrix[start : start + 3].T
            interface_psi = modal_matrix[start + 3 : start + 6].T
            frames.append((name, point, interface_phi, interface_psi))
            distances = np.linalg.norm(self.sample_points - point, axis=1)
            matching = np.nonzero(distances <= self.interface_match_tolerance)[0]
            if matching.shape[0] > 0:
                # The interface frame coincides with a recovery sample. A direct (RBE2-style)
                # interface moves that material point exactly, so the sample reuses the exact
                # interface rotation. A distributing (RBE3-style) interface follows a weighted
                # average of other nodes, so the material sample keeps its recovered motion and
                # joints use the attachment frame instead.
                sample_index = int(matching[np.argmin(distances[matching])])
                deviation = float(np.max(np.abs(interface_phi - sample_phi[sample_index]), initial=0.0)) / mode_scale
                self.interface_recovery_deviation[name] = deviation
                if deviation <= self.interface_recovery_tolerance:
                    basis_psi[sample_index] = interface_psi.astype(np.float32)
                else:
                    distributing.append(name)
            else:
                sample_index = int(basis_points.shape[0])
                basis_points = np.vstack((basis_points, point.astype(np.float32)))
                basis_phi = np.concatenate((basis_phi, interface_phi.astype(np.float32)[None]), axis=0)
                basis_psi = np.concatenate((basis_psi, interface_psi.astype(np.float32)[None]), axis=0)
                if basis_mass is not None:
                    basis_mass = np.append(basis_mass, np.float32(0.0))
            self.interface_sample_indices[name] = sample_index

        if distributing:
            worst = max(self.interface_recovery_deviation[name] for name in distributing)
            warnings.warn(
                f"Craig-Bampton interfaces {distributing} coincide with recovery samples whose recovered "
                f"motion differs from the interface DOFs (maximum relative deviation {worst:.3g} > "
                f"interface_recovery_tolerance {self.interface_recovery_tolerance:.3g}). Treating them as "
                "distributing interfaces: joints attached here use the interface frame, contacts and "
                "rendering use the physical recovery sample.",
                stacklevel=2,
            )

        spatial_mass = rigid.T @ mass @ rigid
        self._extract_rigid_properties(spatial_mass)
        rigid_modal_mass = rigid.T @ mass @ modal_matrix
        coupling_linear = rigid_modal_mass[:3].T
        coupling_angular = -rigid_modal_mass[3:].T

        self.spatial_mass = np.array(spatial_mass, dtype=np.float64, copy=True)
        self.rigid_matrix = np.array(rigid, dtype=np.float64, copy=True)
        self.modal_matrix = np.array(modal_matrix, dtype=np.float64, copy=True)
        self.frequencies = np.sqrt(np.maximum(mode_stiffness / np.maximum(mode_mass, 1.0e-12), 0.0)) / (2.0 * math.pi)

        return ModalBasis(
            sample_points=basis_points,
            sample_phi=basis_phi,
            sample_psi=basis_psi,
            sample_mass=basis_mass,
            mode_mass=mode_mass,
            mode_stiffness=mode_stiffness,
            mode_damping=mode_damping,
            mode_coupling_linear=coupling_linear,
            mode_coupling_angular=coupling_angular,
            label=self.label,
            mode_mass_matrix=modal_mass,
            rigid_mass=self.mass,
            rigid_com=self.com,
            rigid_inertia=self.inertia,
            frame_points=np.asarray([frame[1] for frame in frames], dtype=np.float32),
            frame_phi=np.asarray([frame[2] for frame in frames], dtype=np.float32),
            frame_psi=np.asarray([frame[3] for frame in frames], dtype=np.float32),
            frame_labels=[frame[0] for frame in frames],
        )


class ModalGeneratorPOD:
    """Build linear modal basis functions with proper orthogonal decomposition.

    .. experimental::

    Each retained mode is a right singular vector of the snapshot matrix scaled
    to a unit maximum sample displacement, so the modal coordinates are
    displacement-valued [m]. Modal stiffness is a heuristic derived from
    ``stiffness_scale`` and the singular values; calibrate it for physical
    results.

    Args:
        sample_points: Rest sample points [m], shape ``[sample_count, 3]``.
        displacements: Exemplar displacement fields [m], shape
            ``[snapshot_count, sample_count, 3]``.
        deformed_points: Exemplar deformed points [m], shape
            ``[snapshot_count, sample_count, 3]``. Used only when
            ``displacements`` is not provided.
        mode_count: Number of POD modes to retain.
        total_mass: Total mass [kg] represented by the samples. If provided,
            it is lumped uniformly over the samples to form the modal masses,
            the inertia coupling integrals, and the rigid mass properties.
            Otherwise every modal mass is ``1`` and no coupling integrals are
            formed.
        stiffness_scale: Heuristic stiffness scale [N/m].
        damping_ratio: Modal damping ratio.
        subtract_mean: Whether to remove the mean exemplar displacement before
            computing POD modes.
        derive_psi: Whether to estimate angular mode samples from the
            displacement gradient of the POD modes, enabling joint rotational
            coupling.
        label: Optional basis label.
    """

    def __init__(
        self,
        sample_points: Sequence[Sequence[float]] | np.ndarray,
        displacements: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        deformed_points: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        mode_count: int | None = None,
        total_mass: float | None = None,
        stiffness_scale: float = 1.0,
        damping_ratio: float = 0.0,
        subtract_mean: bool = False,
        derive_psi: bool = False,
        label: str | None = None,
    ):
        self.sample_points = np.asarray(sample_points, dtype=np.float32)
        if self.sample_points.ndim != 2 or self.sample_points.shape[1] != 3:
            raise ValueError(f"sample_points must have shape [sample_count, 3], got {self.sample_points.shape}")

        if displacements is None:
            if deformed_points is None:
                raise ValueError("Either displacements or deformed_points must be provided")
            deformed = np.asarray(deformed_points, dtype=np.float32)
            displacements = deformed - self.sample_points.reshape((1, self.sample_points.shape[0], 3))

        self.displacements = np.asarray(displacements, dtype=np.float32)
        if self.displacements.ndim != 3 or self.displacements.shape[1:] != self.sample_points.shape:
            raise ValueError(
                f"displacements must have shape [snapshot_count, sample_count, 3], got {self.displacements.shape}"
            )

        self.mode_count = mode_count
        self.total_mass = total_mass
        self.stiffness_scale = float(stiffness_scale)
        self.damping_ratio = float(damping_ratio)
        self.subtract_mean = bool(subtract_mean)
        self.derive_psi = bool(derive_psi)
        self.label = label

    def build(self) -> ModalBasis:
        """Build a sampled POD modal basis."""
        snapshot_count = int(self.displacements.shape[0])
        if snapshot_count == 0:
            raise ValueError("At least one displacement snapshot is required")

        matrix = self.displacements.reshape((snapshot_count, -1)).astype(np.float64)
        if self.subtract_mean:
            matrix = matrix - np.mean(matrix, axis=0, keepdims=True)

        _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        mode_count = int(vt.shape[0]) if self.mode_count is None else self.mode_count
        mode_count = min(int(mode_count), int(vt.shape[0]))
        if mode_count < 0:
            raise ValueError(f"mode_count must be non-negative, got {mode_count}")

        modes = vt[:mode_count].reshape((mode_count, self.sample_points.shape[0], 3)).astype(np.float32)
        for i in range(mode_count):
            scale = float(np.max(np.linalg.norm(modes[i], axis=1)))
            if scale > 0.0:
                modes[i] /= scale

        sample_phi = np.transpose(modes, (1, 0, 2))
        sample_psi = _estimate_sample_psi(self.sample_points, sample_phi) if self.derive_psi else None
        mode_mass = self._estimate_mode_mass(sample_phi)
        mode_stiffness = self._estimate_mode_stiffness(mode_count, singular_values)
        mode_damping = np.zeros(mode_count, dtype=np.float32)
        if self.damping_ratio > 0.0:
            for i in range(mode_count):
                mode_damping[i] = (
                    2.0
                    * self.damping_ratio
                    * math.sqrt(max(float(mode_mass[i]), 0.0) * max(float(mode_stiffness[i]), 0.0))
                )

        # Lump the total mass uniformly over the samples (as the modal-mass
        # estimate does) so the basis can form the coupling integrals.
        sample_mass = None
        if self.total_mass is not None and self.sample_points.shape[0] > 0:
            sample_mass = np.full(
                self.sample_points.shape[0],
                float(self.total_mass) / float(self.sample_points.shape[0]),
                dtype=np.float32,
            )

        return ModalBasis(
            sample_points=self.sample_points,
            sample_phi=sample_phi,
            sample_psi=sample_psi,
            sample_mass=sample_mass,
            mode_mass=mode_mass,
            mode_stiffness=mode_stiffness,
            mode_damping=mode_damping,
            label=self.label,
        )

    def _estimate_mode_mass(self, sample_phi: np.ndarray) -> np.ndarray:
        mode_count = int(sample_phi.shape[1])
        if self.total_mass is None or self.sample_points.shape[0] == 0:
            return np.ones(mode_count, dtype=np.float32)

        sample_mass = float(self.total_mass) / float(self.sample_points.shape[0])
        mode_mass = np.zeros(mode_count, dtype=np.float32)
        for i in range(mode_count):
            norm2 = np.sum(sample_phi[:, i, :] * sample_phi[:, i, :], axis=1)
            mode_mass[i] = float(sample_mass * np.sum(norm2))
        return mode_mass

    def _estimate_mode_stiffness(self, mode_count: int, singular_values: np.ndarray) -> np.ndarray:
        if mode_count == 0:
            return np.zeros(0, dtype=np.float32)

        stiffness = np.zeros(mode_count, dtype=np.float32)
        if singular_values.shape[0] > 0 and singular_values[0] > 0.0:
            energy = singular_values[:mode_count] / singular_values[0]
        else:
            energy = np.ones(mode_count, dtype=np.float64)

        for i in range(mode_count):
            # POD modes are ordered by explained displacement energy. Keep the
            # default heuristic gentle so higher-order exemplar modes are not
            # accidentally locked out before users provide calibrated values.
            stiffness[i] = float(self.stiffness_scale * max(energy[i], 1.0e-6) / ((i + 1) * (i + 1)))
        return stiffness


class ModalGeneratorFEM:
    """Build linear modal modes from nodal finite-element matrices.

    .. experimental::

    This generator solves the generalized eigenvalue problem
    ``K phi = lambda M phi`` on translational nodal DOFs, mass-normalizes the
    selected eigenvectors, and returns a sampled :class:`ModalBasis`. With
    ``mass_matrix`` in [kg], the modal coordinates have units [kg^(1/2) m],
    ``mode_mass`` is ``1``, and ``mode_stiffness`` is ``lambda`` [1/s^2]. The
    inertia coupling integrals and rigid mass properties are computed exactly
    from the consistent nodal mass matrix.

    Rigid-body modes are identified by their mass-weighted overlap with the
    analytic rigid-body translations and rotations of the nodes, not by
    eigenvalue magnitude, so they are removed reliably even when roundoff lifts
    their eigenvalues for stiff materials. A free-free model (no fixed DOFs by
    default) must contain exactly one rigid-body mode per independent rigid
    motion, normally six. A constrained model must not retain any zero-energy
    mode. Remaining zero-energy mechanisms are rejected unless they are skipped
    deliberately with ``discard_mode_count``.

    An eigenvalue counts as zero energy when it is at most
    ``eigenvalue_tolerance`` times the largest eigenvalue magnitude, so models
    whose elastic spectrum spans many decades, such as a soft mount in series
    with a stiff member, keep all their elastic modes. The threshold never drops
    below the accuracy of the double-precision eigensolver,
    ``8 * free_dof_count * eps`` relative to the largest eigenvalue. For a
    constrained model, an unconstrained mechanism is reported as remaining
    rigid-body motion only when it overlaps a rigid motion that the fixed DOFs
    admit.

    The sign of each mode is chosen so that its largest-magnitude nodal
    entry is positive, on both eigensolver paths. Within a repeated
    eigenvalue the basis remains arbitrary and can differ between the paths.

    Damping is not part of the eigenproblem. ``damping_matrix`` is projected onto
    the selected modes and only its diagonal is stored; see
    ``damping_projection``. Unlike :class:`ModalGeneratorCraigBampton`, the
    generator does not reject damping that acts on rigid-body motion, such as
    mass-proportional Rayleigh damping of a free-free model; that part is
    dropped with the rigid-body modes, so the floating frame is undamped.

    After :meth:`build`, ``rigid_mode_count`` and ``rigid_modal_matrix``
    (``[3 * node_count, rigid_mode_count]``, mass-orthonormal rigid-body nodal
    motions; empty for constrained models) describe the removed rigid-body
    subspace, and ``damping_off_diagonal_ratio`` records the discarded damping
    coupling.

    The matrices may be dense arrays or SciPy sparse matrices. The eigenproblem
    is solved either densely (Cholesky reduction and a full symmetric
    eigensolve over all free DOFs) or, for large models, with ARPACK
    shift-invert Lanczos (:func:`scipy.sparse.linalg.eigsh`) on a sparse LU
    factorization of ``K - sigma M``, which computes only the
    ``discard_mode_count + mode_count`` lowest non-rigid modes plus a small
    margin. On that path the rigid-body motions of a free-free model, which
    the stiffness check proves to be zero-energy eigenvectors, are deflated
    from the Lanczos operator rather than sought as a six-fold repeated
    eigenvalue, and a Sturm sequence count (the inertia of a second
    factorization of ``K - tau M`` above the retained modes) verifies that
    no eigenvalue was missed; missed ones are sought again with the found
    modes locked, and a remaining mismatch raises. The zero-energy threshold
    is relative to the bound ``max_i sum_j |K_ij| / M_ii`` on the largest
    eigenvalue instead of the computed spectrum, and its floor is
    ``8 * max_row_nonzeros * eps`` instead of ``8 * free_dof_count * eps``.
    Mass positive definiteness is certified there by the inertia of the
    factorization of ``K - sigma M`` (for ``sigma <= 0``), the mass diagonal,
    and the modal masses of the computed modes, which detects an indefinite
    mass matrix except in directions that the stiffness dominates. For large
    models, pass ``sample_node_indices``: otherwise every sample point that
    does not coincide with a node is interpolated from all nodes.

    Args:
        node_positions: Rest node positions [m], shape ``[node_count, 3]``.
        mass_matrix: Nodal mass matrix [kg], shape
            ``[3 * node_count, 3 * node_count]``; dense or SciPy sparse.
        stiffness_matrix: Nodal stiffness matrix [N/m], shape
            ``[3 * node_count, 3 * node_count]``; dense or SciPy sparse.
        damping_matrix: Optional nodal damping matrix [N s/m], shape
            ``[3 * node_count, 3 * node_count]``; dense or SciPy sparse.
        sample_points: Body-local points [m] where the returned basis is
            sampled. Defaults to ``node_positions``.
        sample_node_indices: Optional node index for each sample point. When
            provided, sample values are copied exactly from those FEM nodes.
            Otherwise sample points that coincide exactly with distinct nodes
            copy those nodes (detected with a k-d tree when SciPy is
            available), and other points are interpolated from nodal modes,
            which costs ``O(sample_count * node_count)``.
        mode_count: Number of elastic modes to retain. Required when the
            ARPACK eigensolver is used.
        fixed_node_indices: Nodes whose translational DOFs are fixed during the
            eigen solve. Their modal displacement is zero in the returned basis.
        fixed_dof_indices: Individual nodal DOF indices fixed during the solve.
        discard_mode_count: Number of lowest elastic modes to skip after
            rigid-body modes have been removed; rigid modes of free-free models
            are detected automatically.
        eigenvalue_tolerance: Eigenvalue threshold relative to the largest
            eigenvalue magnitude, at or below which a non-rigid mode is treated
            as a zero-energy mechanism. The default is set by the accuracy of
            the double-precision eigensolver; values below
            ``8 * free_dof_count * eps`` are raised to that floor.
        damping_ratio: Modal damping ratio used when ``damping_matrix`` is not
            provided.
        derive_psi: Whether to estimate angular mode samples from the
            displacement gradient of the nodal modes, enabling joint rotational
            coupling.
        label: Optional basis label.
        free_free: Whether the model is an unconstrained free body whose
            rigid-body modes must be removed. ``None`` treats the model as
            free-free when no DOFs are fixed. A free-free stiffness matrix must
            annihilate rigid-body motion.
        damping_projection: How to reduce the projected modal damping matrix to
            the diagonal damping stored by :class:`ModalBasis`. ``"diagonal"``
            discards off-diagonal terms, warning when their relative magnitude
            exceeds ``damping_coupling_tolerance``; ``"error"`` raises instead.
        damping_coupling_tolerance: Relative off-diagonal damping magnitude
            allowed before ``damping_projection`` warns or raises.
        eigensolver: Generalized eigensolver: a name, a
            :class:`~newton.ModalEigensolver`, or an
            :class:`~newton.EigenSolverBackend` instance. ``"dense"`` solves
            the full spectrum with LAPACK, densifying sparse input.
            ``"arpack"`` uses sparse ARPACK shift-invert and requires
            ``mode_count`` and SciPy. ``"auto"`` selects ``"arpack"`` when any
            input matrix is sparse, or when the free DOF count exceeds
            ``dense_dof_limit``, ``mode_count`` is set, and SciPy is
            installed; otherwise ``"dense"``, which does not need SciPy.
            ``"subspace"`` uses a default
            :class:`~newton.SubspaceIterationBackend`. A backend instance
            receives the free-DOF matrices in SciPy CSC form, the rigid-body
            motions of a free-free model as ``deflation``, ``arpack_shift``
            as ``sigma``, and ``arpack_tol`` as ``tol``, and requires
            ``mode_count``; its result is checked as by
            :func:`~newton._src.sim.modal_sparse.validate_eigen_result`.
        arpack_shift: ARPACK shift ``sigma`` [1/s^2]. ``None`` uses a small
            negative shift, ``-1e-6 * ||K||_1 / ||M||_1``, which keeps
            ``K - sigma M`` positive definite for free-free models. A
            free-free model requires a negative shift, because its rigid-body
            eigenvalues are zero. For a constrained model, a shift just below
            the lowest wanted eigenvalue speeds up convergence for very wide
            spectra; modes skipped by a higher shift are detected by the Sturm
            sequence count. A shift at an eigenvalue, which makes
            ``K - sigma M`` numerically singular, raises.
        arpack_tol: ARPACK relative eigenvalue accuracy; ``0`` requests
            machine precision.
        arpack_maxiter: Maximum ARPACK restart iterations; ``None`` uses
            SciPy's default. Ignored by backend instances.
        arpack_ncv: Number of ARPACK Lanczos vectors; ``None`` uses SciPy's
            default. Ignored by backend instances.
        dense_dof_limit: Largest free DOF count that ``eigensolver="auto"``
            solves densely for dense input.
    """

    _ARPACK_MARGIN: ClassVar[int] = 2
    """Extra eigenpairs computed beyond the retained ones so the last retained mode is converged."""

    def __init__(
        self,
        node_positions: Sequence[Sequence[float]] | np.ndarray,
        mass_matrix: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray,
        stiffness_matrix: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray,
        damping_matrix: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray | None = None,
        sample_points: Sequence[Sequence[float]] | np.ndarray | None = None,
        sample_node_indices: Sequence[int] | np.ndarray | None = None,
        mode_count: int | None = None,
        fixed_node_indices: Sequence[int] | np.ndarray | None = None,
        fixed_dof_indices: Sequence[int] | np.ndarray | None = None,
        discard_mode_count: int = 0,
        eigenvalue_tolerance: float = 1.0e-12,
        damping_ratio: float = 0.0,
        derive_psi: bool = False,
        label: str | None = None,
        *,
        free_free: bool | None = None,
        damping_projection: Literal["diagonal", "error"] = "diagonal",
        damping_coupling_tolerance: float = 1.0e-5,
        eigensolver: Literal["auto", "dense", "arpack", "subspace"]
        | modal_sparse.ModalEigensolver
        | modal_sparse.EigenSolverBackend = "auto",
        arpack_shift: float | None = None,
        arpack_tol: float = 0.0,
        arpack_maxiter: int | None = None,
        arpack_ncv: int | None = None,
        dense_dof_limit: int = 2000,
    ):
        self.eigensolver = _normalize_eigensolver(eigensolver)
        """``"auto"``, ``"dense"``, ``"arpack"``, or the :class:`~newton.EigenSolverBackend` that solves the eigenproblem."""
        self.arpack_shift = None if arpack_shift is None else float(arpack_shift)
        self.arpack_tol = float(arpack_tol)
        if self.arpack_tol < 0.0:
            raise ValueError(f"arpack_tol must be non-negative, got {self.arpack_tol}")
        self.arpack_maxiter = None if arpack_maxiter is None else int(arpack_maxiter)
        self.arpack_ncv = None if arpack_ncv is None else int(arpack_ncv)
        self.dense_dof_limit = int(dense_dof_limit)
        self._node_positions64 = np.asarray(node_positions, dtype=np.float64)
        self.node_positions = np.asarray(node_positions, dtype=np.float32)
        if self.node_positions.ndim != 2 or self.node_positions.shape[1] != 3:
            raise ValueError(f"node_positions must have shape [node_count, 3], got {self.node_positions.shape}")

        self.node_count = int(self.node_positions.shape[0])
        self.dof_count = 3 * self.node_count
        self.mass_matrix = self._coerce_square_matrix(mass_matrix, self.dof_count, "mass_matrix")
        self.stiffness_matrix = self._coerce_square_matrix(stiffness_matrix, self.dof_count, "stiffness_matrix")
        self.damping_matrix = (
            None
            if damping_matrix is None
            else self._coerce_square_matrix(damping_matrix, self.dof_count, "damping_matrix")
        )

        if sample_points is None:
            self.sample_points = np.array(self.node_positions, dtype=np.float32, copy=True)
        else:
            self.sample_points = np.asarray(sample_points, dtype=np.float32)
            if self.sample_points.ndim != 2 or self.sample_points.shape[1] != 3:
                raise ValueError(f"sample_points must have shape [sample_count, 3], got {self.sample_points.shape}")

        self.sample_node_indices = None
        if sample_node_indices is not None:
            sample_node_indices = np.asarray(sample_node_indices, dtype=np.int32).reshape((-1,))
            if sample_node_indices.shape[0] != self.sample_points.shape[0]:
                raise ValueError(
                    "sample_node_indices must have one entry per sample point, "
                    f"got {sample_node_indices.shape[0]} for {self.sample_points.shape[0]} samples"
                )
            if np.any(sample_node_indices < 0) or np.any(sample_node_indices >= self.node_count):
                raise ValueError("sample_node_indices contains out-of-range node indices")
            self.sample_node_indices = sample_node_indices

        self.mode_count = None if mode_count is None else int(mode_count)
        if self.mode_count is not None and self.mode_count < 0:
            raise ValueError(f"mode_count must be non-negative, got {self.mode_count}")

        fixed: set[int] = set()
        if fixed_node_indices is not None:
            for node_value in np.asarray(fixed_node_indices, dtype=np.int32).reshape((-1,)):
                node = int(node_value)
                if node < 0 or node >= self.node_count:
                    raise ValueError(f"fixed_node_indices contains out-of-range node index {node}")
                fixed.update((3 * node, 3 * node + 1, 3 * node + 2))
        if fixed_dof_indices is not None:
            for dof_value in np.asarray(fixed_dof_indices, dtype=np.int32).reshape((-1,)):
                dof = int(dof_value)
                if dof < 0 or dof >= self.dof_count:
                    raise ValueError(f"fixed_dof_indices contains out-of-range DOF index {dof}")
                fixed.add(dof)
        self.fixed_dof_indices = np.asarray(sorted(fixed), dtype=np.int32)
        self.discard_mode_count = int(discard_mode_count)
        if self.discard_mode_count < 0:
            raise ValueError(f"discard_mode_count must be non-negative, got {self.discard_mode_count}")
        self.eigenvalue_tolerance = float(eigenvalue_tolerance)
        if self.eigenvalue_tolerance < 0.0:
            raise ValueError(f"eigenvalue_tolerance must be non-negative, got {self.eigenvalue_tolerance}")
        self.damping_ratio = float(damping_ratio)
        self.derive_psi = bool(derive_psi)
        self.label = label

        self.free_free = self.fixed_dof_indices.shape[0] == 0 if free_free is None else bool(free_free)
        if self.free_free and self.fixed_dof_indices.shape[0] > 0:
            raise ValueError("free_free=True cannot be combined with fixed nodes or DOFs")
        self.damping_projection = _validate_damping_projection(damping_projection)
        self.damping_coupling_tolerance = float(damping_coupling_tolerance)
        if self.damping_coupling_tolerance < 0.0:
            raise ValueError(f"damping_coupling_tolerance must be non-negative, got {self.damping_coupling_tolerance}")

        self.eigenvalues = np.zeros(0, dtype=np.float64)
        self.frequencies = np.zeros(0, dtype=np.float64)
        self.modal_matrix = np.zeros((self.dof_count, 0), dtype=np.float64)
        self.rigid_mode_count = 0
        self.rigid_modal_matrix = np.zeros((self.dof_count, 0), dtype=np.float64)
        self.damping_off_diagonal_ratio = 0.0

    @staticmethod
    def _coerce_square_matrix(
        matrix: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray, size: int, name: str
    ) -> np.ndarray | spmatrix:
        """Return the symmetric part of a square matrix; sparse input stays sparse (CSC)."""
        return modal_sparse.coerce_symmetric_matrix(matrix, size, name)

    def _admissible_rigid_vectors(self, mass: np.ndarray | spmatrix) -> np.ndarray:
        """Return the rigid motions that leave every fixed DOF at rest, shape ``[dof_count, k]``.

        For a free model these are all rigid translations and rotations. For a
        constrained model they span the null space of the rigid vectors
        restricted to the fixed DOFs, which is empty when the constraints
        remove all rigid-body motion.
        """
        rigid = self._rigid_vectors(mass)
        if self.fixed_dof_indices.shape[0] == 0:
            return rigid
        # Normalize the columns so the rank test does not depend on the body size.
        rigid = rigid / np.maximum(np.linalg.norm(rigid, axis=0), np.finfo(np.float64).tiny)
        # Only the 6x6 right singular vectors are needed; skip the [fixed, fixed] left ones when the
        # thin factorization already yields all of them.
        fixed_rigid = rigid[self.fixed_dof_indices]
        _, singular_values, right = np.linalg.svd(fixed_rigid, full_matrices=fixed_rigid.shape[0] < 6)
        rank = int(np.count_nonzero(singular_values > 1.0e-10))
        return rigid @ right[rank:].T

    def _rigid_vectors(self, mass: np.ndarray | spmatrix) -> np.ndarray:
        """Return nodal rigid translations and rotations about the mass center, shape ``[dof_count, 6]``."""
        positions = self._node_positions64
        translation = np.tile(np.eye(3, dtype=np.float64), (self.node_count, 1))
        total_mass = float(np.trace(modal_sparse.bilinear(translation, mass, translation))) / 3.0
        if total_mass > 0.0:
            center = modal_sparse.bilinear(translation, mass, positions.reshape(-1)) / total_mass
        else:
            center = np.mean(positions, axis=0)
        rigid = np.zeros((self.dof_count, 6), dtype=np.float64)
        rigid[:, :3] = translation
        for axis in range(3):
            unit = np.zeros(3, dtype=np.float64)
            unit[axis] = 1.0
            rigid[:, 3 + axis] = np.cross(unit, positions - center).reshape(-1)
        return rigid

    @staticmethod
    def _mass_orthonormal_columns(vectors: np.ndarray, mass: np.ndarray | spmatrix) -> np.ndarray:
        """Return an ``M``-orthonormal basis of the column span of ``vectors``, dropping dependent directions."""
        if vectors.shape[1] == 0:
            return vectors
        gram = modal_sparse.bilinear(vectors, mass, vectors)
        gram = 0.5 * (gram + gram.T)
        values, rotation = np.linalg.eigh(gram)
        scale = max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
        keep = values > 1.0e-10 * scale
        return vectors @ (rotation[:, keep] / np.sqrt(values[keep]))

    def _uses_arpack(self, free_count: int) -> bool:
        """Return whether :meth:`build` solves a sparse eigenproblem through a backend for ``free_count`` free DOFs."""
        if not isinstance(self.eigensolver, str):
            return True
        if self.eigensolver == "auto":
            matrices = (self.mass_matrix, self.stiffness_matrix, self.damping_matrix)
            if any(modal_sparse.is_sparse(matrix) for matrix in matrices):
                return True
            # Dense input stays dense without mode_count (ARPACK needs it) or SciPy, as before.
            return free_count > self.dense_dof_limit and self.mode_count is not None and modal_sparse.has_scipy()
        return self.eigensolver == "arpack"

    def _eigen_backend(self) -> modal_sparse.EigenSolverBackend:
        """Return the backend of the sparse path: the given instance, or ARPACK with this generator's options."""
        if not isinstance(self.eigensolver, str):
            return self.eigensolver
        return modal_sparse.ScipyArpackBackend(
            margin=self._ARPACK_MARGIN, maxiter=self.arpack_maxiter, ncv=self.arpack_ncv
        )

    def _solve_dense(self, mass: np.ndarray, stiffness: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Solve the full generalized eigenproblem on the free DOFs with LAPACK.

        Returns:
            Ascending eigenvalues [1/s^2], ``M``-orthonormal eigenvectors, the
            zero-energy threshold [1/s^2], and its relative tolerance.
        """
        eigenvalues, eigen_modes = modal_sparse.cholesky_eigh(stiffness, mass)
        # Relative to the largest eigenvalue: a genuine elastic mode far below a stiff one stays
        # elastic, while eigenvalues within the eigensolver's roundoff of zero are zero energy.
        eigenvalue_scale = float(np.max(np.abs(eigenvalues)))
        relative_tolerance = max(self.eigenvalue_tolerance, 8.0 * mass.shape[0] * np.finfo(np.float64).eps)
        return eigenvalues, eigen_modes, relative_tolerance * eigenvalue_scale, relative_tolerance

    def _solve_arpack(
        self, mass: spmatrix, stiffness: spmatrix, rigid_basis: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Solve for the lowest generalized eigenpairs on the free DOFs with an eigensolver backend.

        The default backend is ARPACK shift-invert. The rigid-body motions
        ``rigid_basis`` of a free-free model are exact zero-energy
        eigenvectors once the stiffness check has passed, so they are
        deflated from the backend's operator and prepended to the result
        instead of being sought as a six-fold repeated eigenvalue.
        ``discard_mode_count + mode_count`` further eigenpairs, plus a small
        margin, are computed and verified with a Sturm sequence count when
        the backend provides one; see
        :func:`~newton._src.sim.modal_sparse.validate_eigen_result`.

        Returns:
            Ascending eigenvalues [1/s^2], ``M``-orthonormal eigenvectors, the
            zero-energy threshold [1/s^2], and its relative tolerance.
        """
        backend = self._eigen_backend()
        if self.mode_count is None:
            solver = "ARPACK" if backend.name == "arpack" else f"'{backend.name}'"
            raise ValueError(
                f"mode_count is required by the {solver} eigensolver, which computes only the lowest modes; "
                "set mode_count, or use eigensolver='dense' for the full spectrum of a small model"
            )
        mass_diagonal = np.asarray(mass.diagonal(), dtype=np.float64)
        if not np.all(mass_diagonal > 0.0):
            raise ValueError("mass_matrix must be symmetric positive definite on free DOFs")
        sigma = modal_sparse.default_shift(stiffness, mass) if self.arpack_shift is None else self.arpack_shift
        rigid_count = 0 if rigid_basis is None else int(rigid_basis.shape[1])
        if rigid_count > 0 and sigma >= 0.0:
            # The rigid-body eigenvalues are zero: sigma = 0 makes K - sigma M singular, and solves near it
            # amplify roundoff in the deflated rigid directions into spurious eigenpairs.
            raise ValueError(
                f"arpack_shift must be negative for a free-free model, got {sigma:.6g}: its {rigid_count} rigid-body "
                "modes have eigenvalue 0, so stiffness_matrix - arpack_shift * mass_matrix is singular or "
                "indefinite. Omit arpack_shift to use the default negative shift"
            )
        # Host backends share a validated SuperLU factorization; a device backend factors on its own device.
        factorization = None
        if backend.device == "cpu":
            factorization = self._checked_factorization(mass, stiffness, sigma)

        # Gershgorin-type bound on the largest eigenvalue: computing it would cost a second solve, and only
        # its order of magnitude sets the relative threshold. The floor is the roundoff of a sparse Rayleigh
        # quotient, which grows with the nonzeros per row rather than with the DOF count. The bound is
        # tight for these three-DOF nodes but loose for six-DOF shells; see generalized_eigenvalue_bound.
        eigenvalue_scale = modal_sparse.generalized_eigenvalue_bound(stiffness, mass)
        row_nonzeros = max(modal_sparse.max_row_nonzeros(stiffness), 1)
        relative_tolerance = max(self.eigenvalue_tolerance, 8.0 * row_nonzeros * np.finfo(np.float64).eps)
        zero_threshold = relative_tolerance * eigenvalue_scale

        count = self.discard_mode_count + self.mode_count
        try:
            result = backend.solve(
                stiffness,
                mass,
                count,
                deflation=rigid_basis,
                sigma=sigma,
                tol=self.arpack_tol,
                factorization=factorization,
                zero_threshold=zero_threshold,
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError("mass_matrix must be symmetric positive definite on free DOFs") from exc
        eigenvalues, eigen_modes = modal_sparse.validate_eigen_result(
            result,
            size=int(stiffness.shape[0]),
            count=count,
            deflated=rigid_count,
            supports_inertia_count=bool(backend.supports_inertia_count),
        )
        modal_mass = modal_sparse.quadratic_form_diag(eigen_modes, mass)
        if not np.all(modal_mass > 0.0):
            raise ValueError("mass_matrix must be symmetric positive definite on free DOFs")
        if rigid_basis is not None and rigid_basis.shape[1] > 0:
            eigenvalues = np.concatenate((modal_sparse.quadratic_form_diag(rigid_basis, stiffness), eigenvalues))
            eigen_modes = np.hstack((rigid_basis, eigen_modes))
            order = np.argsort(eigenvalues, kind="stable")
            eigenvalues = eigenvalues[order]
            eigen_modes = eigen_modes[:, order]
        return eigenvalues, eigen_modes, zero_threshold, relative_tolerance

    @staticmethod
    def _checked_factorization(mass: spmatrix, stiffness: spmatrix, sigma: float) -> modal_sparse.ShiftedFactorization:
        """Factor ``K - sigma M`` and raise if it is singular or reveals an indefinite matrix."""
        factorization = modal_sparse.ShiftedFactorization(stiffness, mass, sigma)
        if factorization.numerically_singular:
            raise ValueError(
                f"stiffness_matrix - sigma * mass_matrix is numerically singular for sigma = {sigma:.6g} (smallest "
                f"pivot {factorization.pivot_ratio:.3g} of the largest): sigma lies at an eigenvalue, or the "
                "stiffness matrix has a mechanism that the shift does not separate. Choose a different arpack_shift"
            )
        if sigma <= 0.0 and factorization.negative_pivot_count > 0:
            # K - sigma M with sigma <= 0 is indefinite only if M or K is; factor M to tell which.
            mass_pivots = modal_sparse.nonpositive_pivot_count(mass)
            if mass_pivots == 0:
                raise ValueError(
                    "stiffness_matrix must be positive semi-definite on free DOFs: stiffness_matrix - sigma * "
                    f"mass_matrix has {factorization.negative_pivot_count} non-positive pivots for "
                    f"sigma = {sigma:.6g} although mass_matrix is positive definite"
                )
            cause = (
                f"mass_matrix has {mass_pivots}"
                if mass_pivots > 0
                else "also check that stiffness_matrix is positive semi-definite"
            )
            raise ValueError(
                "mass_matrix must be symmetric positive definite on free DOFs: stiffness_matrix - sigma * "
                f"mass_matrix has {factorization.negative_pivot_count} non-positive pivots for "
                f"sigma = {sigma:.6g}; {cause}"
            )
        return factorization

    @staticmethod
    def _check_rigid_motion_preserved(stiffness: np.ndarray | spmatrix, rigid_free: np.ndarray) -> None:
        """Raise if a free-free ``stiffness`` does not annihilate the rigid-body motions ``rigid_free``."""
        stiffness_scale = max(modal_sparse.matrix_inf_norm(stiffness), 1.0)
        rigid_scale = max(float(np.linalg.norm(rigid_free, ord=np.inf)), 1.0)
        rigid_stiffness_ratio = float(np.linalg.norm(modal_sparse.matmul(stiffness, rigid_free), ord=np.inf)) / (
            stiffness_scale * rigid_scale
        )
        if rigid_stiffness_ratio > 1.0e-7:
            raise ValueError(
                "The free-free stiffness_matrix does not preserve rigid-body motion "
                f"(relative residual {rigid_stiffness_ratio:.3g}); fix the stiffness model or "
                "constrain the model with fixed_node_indices or fixed_dof_indices"
            )

    def _coincident_sample_nodes(self) -> np.ndarray:
        """Return the node index of each sample point that coincides with exactly one node, else ``-1``.

        Uses a k-d tree when SciPy is available; otherwise every entry is ``-1``.
        """
        coincident = np.full(self.sample_points.shape[0], -1, dtype=np.int64)
        if self.sample_points.shape[0] == 0 or self.node_count == 0:
            return coincident
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return coincident
        neighbors = min(2, self.node_count)
        distance, index = cKDTree(self.node_positions.astype(np.float64)).query(
            self.sample_points.astype(np.float64), k=neighbors
        )
        distance = np.asarray(distance, dtype=np.float64).reshape((self.sample_points.shape[0], neighbors))
        index = np.asarray(index, dtype=np.int64).reshape((self.sample_points.shape[0], neighbors))
        exact = distance[:, 0] == 0.0
        if neighbors > 1:
            exact &= distance[:, 1] > 0.0
        coincident[exact] = index[exact, 0]
        return coincident

    def build(self) -> ModalBasis:
        """Build a sampled modal basis.

        Returns:
            The sampled modal basis of the selected elastic modes.

        Raises:
            ValueError: If the mass matrix is not positive definite on the free
                DOFs, a free-free stiffness matrix does not preserve rigid-body
                motion, the rigid-body mode count is inconsistent, zero-energy
                mechanisms remain, projected damping coupling is rejected,
                ``mode_count`` is missing on the ARPACK path, or ARPACK does
                not converge.
        """
        free_dofs = np.ones(self.dof_count, dtype=bool)
        free_dofs[self.fixed_dof_indices] = False
        free = np.nonzero(free_dofs)[0]
        if free.shape[0] == 0:
            raise ValueError("At least one free DOF is required to build FEM modes")

        use_arpack = self._uses_arpack(int(free.shape[0]))
        convert = modal_sparse.to_csc if use_arpack else modal_sparse.to_dense
        mass_matrix = convert(self.mass_matrix)
        stiffness_matrix = convert(self.stiffness_matrix)
        damping_matrix = None if self.damping_matrix is None else convert(self.damping_matrix)

        mass = modal_sparse.submatrix(mass_matrix, free, free)
        stiffness = modal_sparse.submatrix(stiffness_matrix, free, free)
        rigid_free = self._admissible_rigid_vectors(mass_matrix)[free]
        rigid_basis = self._mass_orthonormal_columns(rigid_free, mass)
        if use_arpack:
            if self.free_free:
                self._check_rigid_motion_preserved(stiffness, rigid_free)
            eigenvalues, eigen_modes, zero_threshold, relative_tolerance = self._solve_arpack(
                mass, stiffness, rigid_basis if self.free_free else None
            )
        else:
            eigenvalues, eigen_modes, zero_threshold, relative_tolerance = self._solve_dense(mass, stiffness)

        zero_indices = np.nonzero(eigenvalues <= zero_threshold)[0]
        if zero_indices.shape[0] > 1 and rigid_basis.shape[1] > 0:
            # Rigid modes and mechanisms share one degenerate zero-energy eigenspace, in which the
            # eigensolver returns arbitrary mixtures. Rotate that subspace so leading vectors align
            # with rigid-body motion before the per-mode overlap test.
            zero_modes = eigen_modes[:, zero_indices]
            _, _, alignment = np.linalg.svd(modal_sparse.bilinear(rigid_basis, mass, zero_modes))
            zero_modes = zero_modes @ alignment.T
            eigen_modes[:, zero_indices] = zero_modes
            eigenvalues[zero_indices] = modal_sparse.quadratic_form_diag(zero_modes, stiffness)
        rigid_overlap = np.sum((rigid_basis.T @ modal_sparse.matmul(mass, eigen_modes)) ** 2, axis=0)
        if self.free_free:
            if not use_arpack:
                self._check_rigid_motion_preserved(stiffness, rigid_free)
            rigid_mask = rigid_overlap > 0.5
            expected_rigid_count = int(rigid_basis.shape[1])
            if int(np.count_nonzero(rigid_mask)) != expected_rigid_count:
                raise ValueError(
                    f"Expected {expected_rigid_count} rigid-body modes in the free-free model, identified "
                    f"{int(np.count_nonzero(rigid_mask))}; check that mass_matrix and stiffness_matrix describe a "
                    "single connected body"
                )
        else:
            rigid_mask = np.zeros(eigenvalues.shape[0], dtype=bool)

        elastic_indices = np.nonzero(~rigid_mask)[0]
        mechanism = eigenvalues[elastic_indices] <= zero_threshold
        mechanism_count = int(np.count_nonzero(mechanism))
        if mechanism_count > self.discard_mode_count:
            if not self.free_free and np.any(rigid_overlap[elastic_indices[mechanism]] > 0.5):
                raise ValueError(
                    f"The constrained model has {mechanism_count} zero-energy modes dominated by rigid-body motion; "
                    "the fixed nodes or DOFs do not remove all rigid-body motion. Fix more DOFs, or build an "
                    "unconstrained body with free_free=True"
                )
            raise ValueError(
                f"The model has {mechanism_count} zero-energy mechanism modes with eigenvalues at or below "
                f"{zero_threshold:.3g} 1/s^2 (relative tolerance {relative_tolerance:.3g}); fix the "
                f"stiffness model, or set discard_mode_count >= {mechanism_count} to drop them deliberately"
            )

        valid_indices = elastic_indices[self.discard_mode_count :]
        if self.mode_count is not None:
            valid_indices = valid_indices[: self.mode_count]
        if valid_indices.shape[0] == 0:
            raise ValueError("No eigenmodes remain after filtering")

        # LAPACK and ARPACK return arbitrary signs; a fixed convention makes both paths reproducible.
        free_modes = _orient_modes(eigen_modes[:, valid_indices])
        full_modes = np.zeros((self.dof_count, valid_indices.shape[0]), dtype=np.float64)
        full_modes[free, :] = free_modes

        mode_mass = np.zeros(valid_indices.shape[0], dtype=np.float64)
        for i in range(valid_indices.shape[0]):
            phi = full_modes[:, i]
            modal_mass = modal_sparse.quadratic_form(phi, mass_matrix)
            if modal_mass <= 0.0:
                raise ValueError(f"Mode {i} has non-positive modal mass {modal_mass}")
            full_modes[:, i] *= 1.0 / math.sqrt(modal_mass)
            phi = full_modes[:, i]
            mode_mass[i] = modal_sparse.quadratic_form(phi, mass_matrix)
        mode_stiffness = _clamp_modal_diagonal(
            modal_sparse.quadratic_form_diag(full_modes, stiffness_matrix),
            zero_threshold,
            "FEM stiffness",
        )
        if damping_matrix is not None:
            mode_damping = _project_modal_damping(
                self, modal_sparse.bilinear(full_modes, damping_matrix, full_modes), "FEM"
            )
        else:
            self.damping_off_diagonal_ratio = 0.0
            mode_damping = np.zeros(valid_indices.shape[0], dtype=np.float64)
            if self.damping_ratio > 0.0:
                mode_damping = 2.0 * self.damping_ratio * np.sqrt(mode_mass * mode_stiffness)

        rigid_modal_matrix = np.zeros((self.dof_count, rigid_basis.shape[1] if self.free_free else 0))
        if self.free_free:
            rigid_modal_matrix[free, :] = rigid_basis
        self.rigid_mode_count = int(rigid_modal_matrix.shape[1])
        self.rigid_modal_matrix = rigid_modal_matrix

        nodal_phi = np.transpose(full_modes.reshape((self.node_count, 3, -1)), (0, 2, 1))
        nodal_psi = _estimate_sample_psi(self.node_positions, nodal_phi) if self.derive_psi else None
        if self.sample_node_indices is None:
            # Sample points on a node copy it exactly, as evaluate() would; only the others are interpolated.
            coincident = self._coincident_sample_nodes()
            interpolated = np.nonzero(coincident < 0)[0]
            sample_count = self.sample_points.shape[0]
            sample_phi = np.zeros((sample_count, valid_indices.shape[0], 3), dtype=np.float32)
            sample_psi = np.zeros_like(sample_phi) if self.derive_psi else None
            matched = np.nonzero(coincident >= 0)[0]
            sample_phi[matched] = nodal_phi[coincident[matched]]
            if sample_psi is not None:
                sample_psi[matched] = nodal_psi[coincident[matched]]
            if interpolated.shape[0] > 0:
                nodal_basis = ModalBasis(
                    sample_points=self.node_positions,
                    sample_phi=nodal_phi,
                    sample_psi=nodal_psi,
                    mode_mass=mode_mass,
                    mode_stiffness=mode_stiffness,
                    mode_damping=mode_damping,
                    label=self.label,
                )
                for sample in interpolated:
                    phi_value, psi_value = nodal_basis.evaluate(self.sample_points[sample])
                    sample_phi[sample] = phi_value
                    if sample_psi is not None:
                        sample_psi[sample] = psi_value
        else:
            sample_phi = nodal_phi[self.sample_node_indices]
            sample_psi = None if nodal_psi is None else nodal_psi[self.sample_node_indices]

        self.eigenvalues = mode_stiffness / np.maximum(mode_mass, 1.0e-12)
        self.frequencies = np.sqrt(np.maximum(self.eigenvalues, 0.0)) / (2.0 * math.pi)
        self.modal_matrix = np.array(full_modes, dtype=np.float64, copy=True)

        # Exact inertia integrals from the consistent mass matrix: the
        # mass-weighted mode field (M Phi) replaces the lumped mass_s phi_s.
        mass_modes = modal_sparse.matmul(mass_matrix, full_modes)
        mass_phi = np.transpose(mass_modes.reshape((self.node_count, 3, -1)), (0, 2, 1))  # [node, mode, 3]
        integrals = _reduced_inertia_from_pairs(self.node_positions, nodal_phi, mass_phi)
        rigid_mass, rigid_com, rigid_inertia = self._rigid_properties(mass_matrix)

        return ModalBasis(
            sample_points=self.sample_points,
            sample_phi=sample_phi,
            sample_psi=sample_psi,
            mode_mass=mode_mass,
            mode_stiffness=mode_stiffness,
            mode_damping=mode_damping,
            mode_coupling_linear=integrals.mode_coupling_linear,
            mode_coupling_angular=integrals.mode_coupling_angular,
            mode_coupling_centrifugal=integrals.mode_coupling_centrifugal,
            mode_coupling_coriolis=integrals.mode_coupling_coriolis,
            label=self.label,
            mode_coupling_quadratic=integrals.mode_coupling_quadratic,
            mode_mass_matrix=full_modes.T @ mass_modes,
            rigid_mass=rigid_mass,
            rigid_com=rigid_com,
            rigid_inertia=rigid_inertia,
        )

    def _rigid_properties(
        self, mass_matrix: np.ndarray | spmatrix
    ) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
        """Return the rigid mass [kg], center of mass [m], and inertia about it [kg m^2].

        The spatial mass ``[T R]^T M [T R]`` is projected from the nodal mass
        matrix with rigid translation ``T`` and rotation ``R`` fields. Returns
        ``None`` values when ``mass_matrix`` is not a physical nodal mass matrix.
        """
        positions = self.node_positions.astype(np.float64)
        rigid = np.zeros((self.dof_count, 6), dtype=np.float64)
        rigid[:, :3] = np.tile(np.eye(3), (self.node_count, 1))
        for axis in range(3):
            unit = np.zeros(3, dtype=np.float64)
            unit[axis] = 1.0
            rigid[:, 3 + axis] = np.cross(unit, positions).reshape((-1,))
        spatial_mass = modal_sparse.bilinear(rigid, mass_matrix, rigid)
        tolerance = 1.0e-7 * max(float(np.max(np.abs(spatial_mass))), 1.0)
        mass = float(np.trace(spatial_mass[:3, :3]) / 3.0)
        com_skew = -spatial_mass[:3, 3:] / mass if mass > 0.0 else np.zeros((3, 3))
        if (
            mass <= 0.0
            or np.max(np.abs(spatial_mass[:3, :3] - mass * np.eye(3))) > tolerance
            or np.max(np.abs(com_skew + com_skew.T)) > tolerance
        ):
            return None, None, None
        com = np.array([com_skew[2, 1], com_skew[0, 2], com_skew[1, 0]], dtype=np.float64)
        inertia = 0.5 * (spatial_mass[3:, 3:] + spatial_mass[3:, 3:].T)
        inertia -= mass * ((com @ com) * np.eye(3) - np.outer(com, com))
        return mass, com, inertia


# Roots of ``cos(x) cosh(x) = -1``: clamped-free Euler-Bernoulli eigenvalues ``beta_n L``.
_CANTILEVER_BETA_L = (
    1.8751040687119611,
    4.6940911329741746,
    7.8547574382376126,
    10.995540734875467,
    14.137168391046471,
)


def _cantilever_beta_l(order: int) -> float:
    """Return the clamped-free bending eigenvalue ``beta_n L`` of 1-based ``order``."""
    if order <= len(_CANTILEVER_BETA_L):
        return _CANTILEVER_BETA_L[order - 1]
    x = (float(order) - 0.5) * math.pi
    # Newton iterations on cos(x) + 1 / cosh(x) = 0, written without overflow for large x.
    for _ in range(4):
        decay = math.exp(-x)
        sech = 2.0 * decay / (1.0 + decay * decay)
        tanh = (1.0 - decay * decay) / (1.0 + decay * decay)
        x -= (math.cos(x) + sech) / (-math.sin(x) - tanh * sech)
    return x


def _cantilever_eigenfunction(s: np.ndarray, beta_l: float, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate an un-normalized clamped-free bending eigenfunction and its slope.

    ``W(s) = cosh(beta s) - cos(beta s) - sigma (sinh(beta s) - sin(beta s))`` with
    ``sigma = (cosh(beta L) + cos(beta L)) / (sinh(beta L) + sin(beta L))``.
    The growing hyperbolic terms nearly cancel for high orders, so
    ``cosh(x) - sigma sinh(x)`` is evaluated as
    ``((1 - sigma) exp(x) + (1 + sigma) exp(-x)) / 2`` with ``(1 - sigma) exp(x)``
    formed from ``exp(x - beta L)``.

    Args:
        s: Arc-length coordinates from the clamped root [m].
        beta_l: Dimensionless eigenvalue ``beta L``.
        length: Beam length [m].

    Returns:
        ``(W, dW/ds)`` evaluated at ``s``, with ``W(L) = 2 (-1)^(n + 1)``.
    """
    beta = beta_l / length
    decay = math.exp(-beta_l)
    sin_l = math.sin(beta_l)
    cos_l = math.cos(beta_l)
    denominator = 1.0 - decay * decay + 2.0 * decay * sin_l
    sigma = (1.0 + decay * decay + 2.0 * decay * cos_l) / denominator
    # (1 - sigma) exp(beta L) and (1 + sigma), both of order one.
    grow_coefficient = 2.0 * (sin_l - cos_l - decay) / denominator
    decay_coefficient = 2.0 * (1.0 + decay * (sin_l + cos_l)) / denominator

    x = beta * np.asarray(s, dtype=np.float64)
    grow = grow_coefficient * np.exp(x - beta_l)
    shrink = decay_coefficient * np.exp(-x)
    w = 0.5 * (grow + shrink) - np.cos(x) + sigma * np.sin(x)
    dw = beta * (0.5 * (grow - shrink) + np.sin(x) + sigma * np.cos(x))
    return w, dw


class ModalGeneratorBeam:
    """Build sampled linear modes for a straight Euler-Bernoulli beam.

    .. experimental::

    The beam lies along local ``x`` with its midpoint at the body origin. Fixed
    (clamped) boundaries are at the root ``x = -length / 2``; arc length
    ``s = x + length / 2`` runs from the root to the free tip. Cantilever and
    fixed-free modes are normalized to unit tip deflection or rotation;
    pinned-pinned and linear modes have unit peak amplitude. Modal masses and
    stiffnesses are the exact Euler-Bernoulli centerline integrals of each shape
    for the uniform section; cross-section rotary inertia is neglected.

    The floating-frame inertia integrals of the returned basis
    (``mode_coupling_linear``, ``_angular``, ``_centrifugal``, ``_coriolis``,
    and ``_quadratic``) and its rigid mass, center of mass (the origin), and
    inertia are integrated over the beam volume by Gauss quadrature, independent
    of the sample points passed to :meth:`build`. The section is represented by
    the rectangle with the same area and second moments ``area_moment_y`` and
    ``area_moment_z``. Like the modal mass, these integrals neglect the rotary
    inertia of bending (a relative effect of order ``(h / L)^2`` for section
    height ``h``), so the trace of each mode-pair integral ``P_ii`` equals
    ``mode_mass``. Torsion modal mass uses ``polar_moment``, whereas the volume
    integrals use ``area_moment_y + area_moment_z``; the two differ when an
    explicit ``polar_moment`` (e.g. a torsion constant of a non-circular
    section) is given, and a ``polar_moment`` below ``area_moment_y +
    area_moment_z`` can make the reduced mass matrix indefinite.

    Mode specifications are dictionaries with a ``type`` from :class:`Mode` and
    optional ``order`` (1-based, default ``1``) and ``boundary`` from
    :class:`Boundary`. Supported combinations:

    - Bending: ``CANTILEVER_TIP`` (default), ``CANTILEVER_STATIC`` (order 1 only),
      and ``PINNED_PINNED``.
    - Torsion: ``FIXED_FREE`` (default) and ``LINEAR``.
    - Axial: ``LINEAR`` (default, order 1 only) and ``FIXED_FREE``.

    Args:
        length: Beam length [m].
        half_width_y: Half width in local ``y`` [m].
        half_width_z: Half width in local ``z`` [m].
        mode_specs: Sequence of mode dictionaries, or bare mode type names. Supported
            ``type`` values are ``"axial"``, ``"bending_y"``, ``"bending_z"``, and
            ``"torsion"``.
        sample_count: Number of samples along the beam length.
        density: Density [kg/m^3].
        young_modulus: Young's modulus [Pa].
        shear_modulus: Shear modulus [Pa]. Defaults to ``young_modulus / 2.6``.
        area: Cross-section area [m^2]. Defaults to the rectangular section.
        area_moment_y: Second moment about local ``y`` [m^4].
        area_moment_z: Second moment about local ``z`` [m^4].
        polar_moment: Polar second moment [m^4].
        damping_ratio: Modal damping ratio.
        label: Optional basis label.
    """

    class Mode:
        """Beam mode type names."""

        AXIAL = "axial"
        BENDING_Y = "bending_y"
        BENDING_Z = "bending_z"
        TORSION = "torsion"

    class Boundary:
        """Beam boundary names used by mode specifications."""

        PINNED_PINNED = "pinned-pinned"
        """Bending ``sin(n pi s / L)``, simply supported at both ends."""
        CANTILEVER_TIP = "cantilever-tip"
        """Clamped-free Euler-Bernoulli bending eigenfunctions.

        ``cosh(beta s) - cos(beta s) - sigma (sinh(beta s) - sin(beta s))`` for
        every order, with zero root deflection and slope and unit tip deflection.
        The modes are mass- and stiffness-orthogonal.
        """
        CANTILEVER_STATIC = "cantilever-static"
        """Order-1 static tip-load bending shape ``s^2 (3 L - s) / (2 L^3)``.

        Its stiffness ``3 E I / L^3`` reproduces the exact static tip-load
        deflection; its modal mass is ``33 rho A L / 140``.
        """
        FIXED_FREE = "fixed-free"
        """Torsion or axial ``sin((n - 1/2) pi s / L)``, fixed at the root."""
        LINEAR = "linear"
        """Axial stretch ``x / L`` about the midpoint (order 1 only), or torsion ``(s / L)^n``."""

    _SUPPORTED_BOUNDARIES: ClassVar[dict[str, tuple[str, ...]]] = {
        Mode.AXIAL: (Boundary.LINEAR, Boundary.FIXED_FREE),
        Mode.BENDING_Y: (Boundary.CANTILEVER_TIP, Boundary.CANTILEVER_STATIC, Boundary.PINNED_PINNED),
        Mode.BENDING_Z: (Boundary.CANTILEVER_TIP, Boundary.CANTILEVER_STATIC, Boundary.PINNED_PINNED),
        Mode.TORSION: (Boundary.FIXED_FREE, Boundary.LINEAR),
    }

    def __init__(
        self,
        length: float,
        half_width_y: float = 0.0,
        half_width_z: float = 0.0,
        mode_specs: Sequence[dict[str, Any] | str] | None = None,
        sample_count: int = 33,
        density: float = 1000.0,
        young_modulus: float = 1.0e6,
        shear_modulus: float | None = None,
        area: float | None = None,
        area_moment_y: float | None = None,
        area_moment_z: float | None = None,
        polar_moment: float | None = None,
        damping_ratio: float = 0.0,
        label: str | None = None,
    ):
        if length <= 0.0:
            raise ValueError(f"length must be positive, got {length}")
        self.length = float(length)
        self.half_width_y = float(half_width_y)
        self.half_width_z = float(half_width_z)
        self.mode_specs = [self._coerce_mode_spec(spec) for spec in (mode_specs or [self.Mode.BENDING_Z])]
        self.sample_count = max(2, int(sample_count))
        self.density = float(density)
        self.young_modulus = float(young_modulus)
        self.shear_modulus = float(shear_modulus) if shear_modulus is not None else self.young_modulus / 2.6
        self.area = float(area) if area is not None else max(4.0 * self.half_width_y * self.half_width_z, 0.0)
        self.area_moment_y = (
            float(area_moment_y)
            if area_moment_y is not None
            else (4.0 / 3.0) * self.half_width_y * self.half_width_z**3
        )
        self.area_moment_z = (
            float(area_moment_z)
            if area_moment_z is not None
            else (4.0 / 3.0) * self.half_width_z * self.half_width_y**3
        )
        self.polar_moment = (
            float(polar_moment) if polar_moment is not None else max(self.area_moment_y + self.area_moment_z, 0.0)
        )
        self.damping_ratio = float(damping_ratio)
        self.label = label

    @classmethod
    def _coerce_mode_spec(cls, spec: dict[str, Any] | str) -> dict[str, Any]:
        result = {"type": spec} if isinstance(spec, str) else dict(spec)
        if "type" not in result:
            raise ValueError("Each beam mode spec must include a 'type' value")
        mode_type = str(result["type"])
        if mode_type not in cls._SUPPORTED_BOUNDARIES:
            raise ValueError(f"Unsupported beam mode type '{mode_type}'")
        order = int(result.get("order", 1))
        if order < 1:
            raise ValueError(f"Beam mode order must be at least 1, got {order}")
        boundary = str(result.get("boundary", cls._default_boundary(mode_type)))
        if boundary not in cls._SUPPORTED_BOUNDARIES[mode_type]:
            raise ValueError(f"Unsupported boundary '{boundary}' for beam mode type '{mode_type}'")
        if order > 1 and boundary == cls.Boundary.CANTILEVER_STATIC:
            raise ValueError(
                "The cantilever-static bending shape supports only order 1; use the cantilever-tip "
                "eigenfunctions for higher orders"
            )
        if order > 1 and mode_type == cls.Mode.AXIAL and boundary == cls.Boundary.LINEAR:
            raise ValueError("The linear axial shape supports only order 1; use the fixed-free axial boundary")
        return result

    def build(self, sample_points: Sequence[Sequence[float]] | np.ndarray | None = None) -> ModalBasis:
        """Build a sampled beam modal basis.

        Args:
            sample_points: Optional body-local sample points [m], shape
                ``[sample_count, 3]``. If omitted, centerline and rectangular
                cross-section samples are generated.

        Returns:
            A sampled modal basis.
        """
        points = self._sample_points() if sample_points is None else np.asarray(sample_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"sample_points must have shape [sample_count, 3], got {points.shape}")

        sample_phi, sample_psi = self._evaluate_modes(points)

        mode_mass = np.zeros(len(self.mode_specs), dtype=np.float32)
        mode_stiffness = np.zeros(len(self.mode_specs), dtype=np.float32)
        mode_damping = np.zeros(len(self.mode_specs), dtype=np.float32)
        for i, spec in enumerate(self.mode_specs):
            mode_mass[i], mode_stiffness[i] = self._mode_properties(spec)
            if self.damping_ratio > 0.0:
                mode_damping[i] = (
                    2.0
                    * self.damping_ratio
                    * math.sqrt(max(float(mode_mass[i]), 0.0) * max(float(mode_stiffness[i]), 0.0))
                )

        inertia = self._reduced_inertia()
        coupling: dict[str, Any] = {}
        if inertia is not None:
            coupling = {
                "mode_coupling_linear": inertia.mode_coupling_linear,
                "mode_coupling_angular": inertia.mode_coupling_angular,
                "mode_coupling_centrifugal": inertia.mode_coupling_centrifugal,
                "mode_coupling_coriolis": inertia.mode_coupling_coriolis,
                "mode_coupling_quadratic": inertia.mode_coupling_quadratic,
                "rigid_mass": inertia.rigid_mass,
                "rigid_com": inertia.rigid_com,
                "rigid_inertia": inertia.rigid_inertia,
            }

        return ModalBasis(
            sample_points=points,
            sample_phi=sample_phi.astype(np.float32),
            sample_psi=sample_psi.astype(np.float32),
            mode_mass=mode_mass,
            mode_stiffness=mode_stiffness,
            mode_damping=mode_damping,
            label=self.label,
            **coupling,
        )

    def _section_quadrature(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return 2x2 Gauss points and weights over the equivalent rectangular section.

        The rectangle ``[-hy, hy] x [-hz, hz]`` with ``hy = sqrt(3 I_z / A)`` and
        ``hz = sqrt(3 I_y / A)`` has the section's area ``A`` and second moments
        ``I_y = int z^2 dA`` and ``I_z = int y^2 dA``, so the rule integrates
        ``1``, ``y``, ``z``, ``y^2``, ``z^2``, and ``y z`` exactly. For the default
        rectangular section it is the section itself.

        Returns:
            Section coordinates ``y`` and ``z`` [m] and area weights [m^2], each
            shape ``[4]``.
        """
        y_offset = math.sqrt(max(self.area_moment_z, 0.0) / self.area)
        z_offset = math.sqrt(max(self.area_moment_y, 0.0) / self.area)
        y = np.array([-y_offset, y_offset, -y_offset, y_offset], dtype=np.float64)
        z = np.array([-z_offset, -z_offset, z_offset, z_offset], dtype=np.float64)
        return y, z, np.full(4, 0.25 * self.area, dtype=np.float64)

    def _reduced_inertia(self) -> _ReducedInertia | None:
        """Integrate the reduced floating-frame inertia of the beam volume.

        Uses composite 4-point Gauss-Legendre quadrature on ``4 * sample_count``
        equal panels along the beam and the 2x2 rule of
        :meth:`_section_quadrature` over the section, with point masses
        ``density * w_x * w_yz``. The integrals therefore do not depend on the
        sample points passed to :meth:`build`, which typically describe a render
        or contact surface rather than the beam volume. Bending modes enter
        with their transverse deflection only (no section rotation), matching
        the centerline modal mass that neglects rotary inertia, so ``trace(P_ii)``
        equals the modal mass and the reduced mass matrix stays positive
        definite for a frame with the rigid properties of the basis. Rigid
        properties are the exact values of the uniform prism centered at the
        origin.

        Returns:
            The reduced inertia integrals, or ``None`` when the beam has no mass.
        """
        if self.density <= 0.0 or self.area <= 0.0:
            return None
        nodes, weights = np.polynomial.legendre.leggauss(4)
        panel_count = 4 * self.sample_count
        panel_length = self.length / panel_count
        panel_centers = -0.5 * self.length + panel_length * (np.arange(panel_count, dtype=np.float64) + 0.5)
        x = (panel_centers[:, None] + 0.5 * panel_length * nodes[None, :]).reshape((-1,))
        x_weights = np.tile(0.5 * panel_length * weights, panel_count)
        y, z, section_weights = self._section_quadrature()

        points = np.empty((x.shape[0], y.shape[0], 3), dtype=np.float64)
        points[..., 0] = x[:, None]
        points[..., 1] = y[None, :]
        points[..., 2] = z[None, :]
        points = points.reshape((-1, 3))
        mass = (self.density * x_weights[:, None] * section_weights[None, :]).reshape((-1,))
        phi, _ = self._evaluate_modes(points)
        # Drop the section-rotation (axial) part of bending modes so the inertial field neglects
        # rotary inertia exactly like the centerline modal mass. Otherwise the coupling integrals
        # would carry inertia that the modal mass lacks and the reduced mass could turn indefinite.
        for mode_index, spec in enumerate(self.mode_specs):
            if spec["type"] in (self.Mode.BENDING_Y, self.Mode.BENDING_Z):
                phi[:, mode_index, 0] = 0.0
        integrals = _reduced_inertia_from_pairs(points, phi, mass=mass)

        rigid_mass = self.density * self.area * self.length
        axial = self.density * self.area * self.length**3 / 12.0
        rigid_inertia = np.diag(
            [
                self.density * self.length * (self.area_moment_y + self.area_moment_z),
                axial + self.density * self.length * self.area_moment_y,
                axial + self.density * self.length * self.area_moment_z,
            ]
        )
        return integrals._replace(
            rigid_mass=float(rigid_mass),
            rigid_com=np.zeros(3, dtype=np.float32),
            rigid_inertia=rigid_inertia.astype(np.float32),
        )

    def _sample_points(self) -> np.ndarray:
        yz: list[tuple[float, float]] = [(0.0, 0.0)]
        if self.half_width_y > 0.0:
            yz.extend([(-self.half_width_y, 0.0), (self.half_width_y, 0.0)])
        if self.half_width_z > 0.0:
            yz.extend([(0.0, -self.half_width_z), (0.0, self.half_width_z)])
        if self.half_width_y > 0.0 and self.half_width_z > 0.0:
            yz.extend(
                [
                    (-self.half_width_y, -self.half_width_z),
                    (self.half_width_y, -self.half_width_z),
                    (self.half_width_y, self.half_width_z),
                    (-self.half_width_y, self.half_width_z),
                ]
            )

        unique_yz = list(dict.fromkeys(yz))
        points: list[tuple[float, float, float]] = []
        for x in np.linspace(-0.5 * self.length, 0.5 * self.length, self.sample_count):
            for y, z in unique_yz:
                points.append((float(x), y, z))
        return np.asarray(points, dtype=np.float32)

    def _evaluate_modes(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate every configured mode at body-local points.

        Args:
            points: Body-local points [m], shape ``[point_count, 3]``.

        Returns:
            ``(phi, psi)`` in float64, each with shape ``[point_count, mode_count, 3]``:
            translational mode values [m per modal coordinate] and body-local
            infinitesimal rotations [rad per modal coordinate].
        """
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        phi = np.zeros((points.shape[0], len(self.mode_specs), 3), dtype=np.float64)
        psi = np.zeros_like(phi)
        for mode_index, spec in enumerate(self.mode_specs):
            phi[:, mode_index], psi[:, mode_index] = self._evaluate_mode_points(spec, points)
        return phi, psi

    def _evaluate_mode_points(self, spec: dict[str, Any], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mode_type = str(spec["type"])
        order = int(spec.get("order", 1))
        boundary = str(spec.get("boundary", self._default_boundary(mode_type)))
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        s = np.clip(x + 0.5 * self.length, 0.0, self.length)
        zero = np.zeros_like(x)
        phi = np.zeros((points.shape[0], 3), dtype=np.float64)
        psi = np.zeros_like(phi)

        if mode_type == self.Mode.AXIAL:
            if boundary == self.Boundary.LINEAR:
                phi[:, 0] = x / self.length
            else:
                phi[:, 0], _ = self._fixed_free_shape(s, order)
            return phi, psi

        if mode_type == self.Mode.BENDING_Y:
            deflection, slope = self._bending_shape(s, order, boundary)
            phi[:] = np.stack((-y * slope, deflection, zero), axis=1)
            psi[:, 2] = slope
            return phi, psi

        if mode_type == self.Mode.BENDING_Z:
            deflection, slope = self._bending_shape(s, order, boundary)
            phi[:] = np.stack((-z * slope, zero, deflection), axis=1)
            psi[:, 1] = -slope
            return phi, psi

        if mode_type == self.Mode.TORSION:
            theta, _ = self._torsion_shape(s, order, boundary)
            phi[:] = np.stack((zero, -z * theta, y * theta), axis=1)
            psi[:, 0] = theta
            return phi, psi

        raise ValueError(f"Unsupported beam mode type '{mode_type}'")

    @classmethod
    def _default_boundary(cls, mode_type: str) -> str:
        if mode_type == cls.Mode.TORSION:
            return cls.Boundary.FIXED_FREE
        if mode_type == cls.Mode.AXIAL:
            return cls.Boundary.LINEAR
        return cls.Boundary.CANTILEVER_TIP

    def _bending_shape(self, s: np.ndarray, order: int, boundary: str) -> tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=np.float64)
        if boundary == self.Boundary.PINNED_PINNED:
            k = float(order) * math.pi / self.length
            return np.sin(k * s), k * np.cos(k * s)

        if boundary == self.Boundary.CANTILEVER_TIP:
            beta_l = _cantilever_beta_l(order)
            tip, _ = _cantilever_eigenfunction(np.array(self.length), beta_l, self.length)
            deflection, slope = _cantilever_eigenfunction(s, beta_l, self.length)
            return deflection / float(tip), slope / float(tip)

        if boundary == self.Boundary.CANTILEVER_STATIC:
            if order != 1:
                raise ValueError("The cantilever-static bending shape supports only order 1")
            deflection = (s * s * (3.0 * self.length - s)) / (2.0 * self.length**3)
            slope = (3.0 * s * (2.0 * self.length - s)) / (2.0 * self.length**3)
            return deflection, slope

        raise ValueError(f"Unsupported bending boundary '{boundary}'")

    def _fixed_free_shape(self, s: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
        # sin((n - 1/2) pi) = (-1)^(n + 1), so dividing by it gives a unit tip value.
        k = (float(order) - 0.5) * math.pi / self.length
        tip = -1.0 if order % 2 == 0 else 1.0
        return tip * np.sin(k * s), tip * k * np.cos(k * s)

    def _torsion_shape(self, s: np.ndarray, order: int, boundary: str) -> tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=np.float64)
        if boundary == self.Boundary.LINEAR:
            power = max(1, order)
            xi = s / self.length
            theta = xi**power
            dtheta = float(power) * xi ** (power - 1) / self.length
            return theta, dtheta

        if boundary == self.Boundary.FIXED_FREE:
            return self._fixed_free_shape(s, order)

        raise ValueError(f"Unsupported torsion boundary '{boundary}'")

    def _mode_properties(self, spec: dict[str, Any]) -> tuple[float, float]:
        mode_type = str(spec["type"])
        order = int(spec.get("order", 1))
        boundary = str(spec.get("boundary", self._default_boundary(mode_type)))
        mass_per_length = self.density * self.area

        if mode_type == self.Mode.AXIAL:
            ea = self.young_modulus * self.area
            if boundary == self.Boundary.LINEAR:
                mass = mass_per_length * self.length / 12.0
                stiffness = ea / self.length if self.area > 0.0 else 0.0
                return mass, stiffness
            k = (float(order) - 0.5) * math.pi / self.length
            return mass_per_length * self.length / 2.0, ea * k * k * self.length / 2.0

        if mode_type in (self.Mode.BENDING_Y, self.Mode.BENDING_Z):
            inertia = self.area_moment_z if mode_type == self.Mode.BENDING_Y else self.area_moment_y
            ei = self.young_modulus * inertia
            if boundary == self.Boundary.PINNED_PINNED:
                k = float(order) * math.pi / self.length
                mass = mass_per_length * self.length / 2.0
                stiffness = ei * k**4 * self.length / 2.0
                return mass, stiffness
            if boundary == self.Boundary.CANTILEVER_STATIC:
                mass = mass_per_length * self.length * (33.0 / 140.0)
                stiffness = 3.0 * ei / (self.length**3)
                return mass, stiffness
            # The un-normalized eigenfunction satisfies int_0^L W^2 ds = L and
            # int_0^L W''^2 ds = beta^4 L; the unit-tip shape divides W by W(L) = +-2.
            beta_l = _cantilever_beta_l(order)
            tip, _ = _cantilever_eigenfunction(np.array(self.length), beta_l, self.length)
            tip_squared = float(tip) ** 2
            beta = beta_l / self.length
            mass = mass_per_length * self.length / tip_squared
            stiffness = ei * beta**4 * self.length / tip_squared
            return mass, stiffness

        if mode_type == self.Mode.TORSION:
            rotary_mass = self.density * self.polar_moment
            gj = self.shear_modulus * self.polar_moment
            if boundary == self.Boundary.LINEAR:
                mass = rotary_mass * self.length / float(2 * order + 1)
                stiffness = gj * float(order * order) / (self.length * float(2 * order - 1))
                return mass, stiffness
            k = (float(order) - 0.5) * math.pi / self.length
            mass = rotary_mass * self.length / 2.0
            stiffness = gj * k * k * self.length / 2.0
            return mass, stiffness

        raise ValueError(f"Unsupported beam mode type '{mode_type}'")


__all__ = [
    "ModalBasis",
    "ModalGeneratorBeam",
    "ModalGeneratorCraigBampton",
    "ModalGeneratorFEM",
    "ModalGeneratorPOD",
    "ModalGeneratorSampled",
]
