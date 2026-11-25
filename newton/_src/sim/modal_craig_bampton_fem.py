# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-interface Craig-Bampton reduction of full finite-element models."""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

import numpy as np

from . import modal_sparse
from .modal import ModalBasis, ModalGeneratorCraigBampton
from .modal_constraints import (
    assemble_triplets,
    check_transformation_rank,
    constraint_transformation,
    transformation_rigid_residual,
)

if TYPE_CHECKING:
    from scipy.sparse import sparray, spmatrix

_COUPLINGS = {"node": "node", "rbe2": "rbe2", "direct": "rbe2", "rbe3": "rbe3", "distributing": "rbe3"}
"""Accepted interface coupling names and the coupling they select."""

_RANK_TOLERANCE = 1.0e-9
"""Relative singular-value threshold for the rigid-fit rank of an interface patch."""

_DOF_COMPONENTS = ("ux", "uy", "uz", "rx", "ry", "rz")
"""Names of the nodal DOF components, node-major."""

_FRAME_COMPONENTS = ("tx", "ty", "tz", "rx", "ry", "rz")
"""Names of the six interface frame coordinates."""


class _Interface(NamedTuple):
    name: str
    nodes: np.ndarray
    position: np.ndarray
    coupling: str
    weights: np.ndarray


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _patch_rigid_rows(points: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Return ``A`` with ``u_j = A_j f`` for rigid frame motion ``f = [t, theta]``, shape ``[3 * node_count, 6]``."""
    return np.vstack([np.hstack((np.eye(3), -_skew(point - position))) for point in points])


class ModalGeneratorCraigBamptonFEM:
    """Build a Craig-Bampton modal basis directly from full finite-element matrices.

    .. experimental::

    This generator performs the fixed-interface Craig-Bampton reduction that
    :class:`ModalGeneratorCraigBampton` otherwise expects as pre-computed
    reduced matrices. It keeps the full model sparse: linear multi-point
    constraints are eliminated first (see :class:`ConstraintElimination`),
    each interface becomes a six-DOF frame ``[tx, ty, tz, rx, ry, rz]`` at
    ``position``, the static constraint modes
    ``Psi = -K_ii^-1 K_ib`` are solved with one sparse factorization of the
    interior stiffness, and the lowest fixed-interface modes are found with
    ARPACK shift-invert Lanczos and verified with a Sturm sequence count, as in
    :class:`ModalGeneratorFEM`. The reduced matrices
    ``T^T M T``, ``T^T K T``, and ``T^T D T`` of
    ``T = [[I, 0], [Psi, Phi]]`` and the translational recovery of the sample
    nodes are passed to :class:`ModalGeneratorCraigBampton`, whose
    :meth:`~ModalGeneratorCraigBampton.build` separates the six rigid-body
    modes. The reduced matrices therefore have ``6 * interface_count +
    fixed_mode_count`` coordinates, and the basis has six fewer modes.

    A truncated set of fixed-interface modes reproduces the static response
    to a load on the interior, such as gravity, only approximately.
    ``static_correction_loads`` adds, for each given nodal load ``f``, the
    fixed-interface static response ``K_ii^-1 f_i`` to the interior basis,
    after the fixed-interface modes: ``T = [[I, 0, 0], [Psi, Phi, X]]``. Each
    vector is made mass-orthogonal to the retained modes and to the earlier
    vectors, dropped when the retained basis already represents it, and
    normalized like the modes. The exact response to ``f`` is
    ``u_i = Psi u_b + K_ii^-1 f_i`` for any motion ``u_b`` of the interface
    frames, so the reduced static solution for every combination of the
    given loads equals the full finite-element one for any
    ``fixed_mode_count`` when the model is supported statically through its
    interface frames: frames held, released, or on springs. This does not
    extend to loads balanced by inertia (a free body under acceleration),
    and ``rigid_consistency="project"`` changes the reduced stiffness, so
    the solution is then exact only as far as the projected residual is
    small (measured on a drifting grid: 1e-14 with every frame held, 1.5e-4
    with one frame released). The reduced model then has
    ``6 * interface_count + fixed_mode_count + static_correction_count``
    coordinates. The vectors are not eigenvectors: each is a stiff
    coordinate whose Rayleigh frequency lies above every retained
    fixed-interface mode. These frequencies are reported separately, and
    ``fixed_interface_modes``, ``fixed_interface_eigenvalues``,
    ``fixed_interface_frequencies``, and the eigenpair residual cover the
    fixed-interface modes only.

    Interfaces are mappings with the keys:

    - ``node_indices`` (required): the interface nodes.
    - ``coupling``: how the nodes follow the frame. ``"node"`` uses the six
      DOFs of a single node as the frame (``dofs_per_node=6`` only).
      ``"rbe2"`` (alias ``"direct"``) ties every node rigidly to the frame:
      ``u_j = u_f + theta_f x (x_j - p)`` and, for six-DOF nodes,
      ``theta_j = theta_f``. ``"rbe3"`` (alias ``"distributing"``) defines the
      frame as the weighted least-squares rigid fit of the node translations,
      leaving the nodes free to deform. Defaults to ``"rbe2"``.
    - ``position``: frame position ``p`` [m]. Defaults to the node position
      for ``"node"`` and to the centroid of the nodes otherwise.
    - ``name``: unique frame label. Defaults to ``interface_<index>``.
    - ``weights``: ``"rbe3"`` node weights, positive. Defaults to equal weights.

    A rigid frame needs six independent DOFs: with three-DOF nodes, or for
    ``"rbe3"``, at least three non-collinear nodes. Nodes of ``"node"`` and
    ``"rbe2"`` interfaces cannot belong to another interface.

    The interior stiffness ``K_ii`` (every interface frame fixed) must be
    positive definite; a remaining mechanism, such as a part without a tie,
    raises. The default ARPACK shift ``sigma = 0`` lets the factorization of
    ``K_ii`` serve both the constraint modes and the eigensolve.

    The sparse path factors and solves the symmetrically equilibrated
    matrices ``K_s = D K_ii D`` and ``M_s = D M_ii D`` with
    ``D = diag(K_ii)^-1/2``, which have the eigenvalues of the original
    pencil, so the units of the DOFs (translations next to rotations) and
    stiffness contrasts that show on the diagonal do not affect the
    factorization or its pivot tests. A contrast that the diagonal does not
    show, such as a stiff part coupled to a soft host, remains in ``K_s``.
    A mechanism is detected by its energy ratio in this metric,
    ``y^T K_s y / (||K_s||_1 y^T y)`` with ``phi = D y``: the strain energy
    of a motion relative to the largest one the equilibrated stiffness can
    hold. It is evaluated for every computed fixed-interface mode and for the
    softest direction of ``K_s`` (a few inverse iterations, or all
    eigenvalues on the dense path), and a ratio at or below
    ``max(eigenvalue_tolerance, 8 n eps)``, with ``n`` the largest number of
    nonzeros in a row of ``K_ii``, raises. The ratio does not involve the
    mass matrix or the largest eigenvalue, so stiff, light parts and small
    rotary inertia, which raise the top of the spectrum by many orders of
    magnitude, do not turn a low elastic mode into an apparent mechanism.
    It is the reciprocal condition number of the equilibrated stiffness, so
    it also refuses a sound model that double precision cannot resolve,
    such as a very slender, finely meshed beam (the ratio falls with the
    fourth power of the element size); ``eigenvalue_tolerance=0`` lowers the
    limit to the roundoff floor. The pivot test of the factor keeps its own
    limit of ``1e-12``.

    The energy ratio does not see a coordinate whose whole stiffness is
    small, such as a node that hangs on soft springs: equilibration scales
    its diagonal to one. Two further checks cover it. The relative tie
    stiffness of a coordinate ``i`` is ``(sum_j |K_ij| / sqrt(K_ii
    K_jj))^2`` over its neighbors ``j``, for a single spring the spring
    stiffness over the stiffness of what it is attached to; a value at or
    below the same tolerance raises, and one below ``1e-6`` warns with the
    lowest fixed-interface frequency. The rule sees a single weakly tied
    coordinate, not a soft cluster: several nodes joined to each other and
    to the model by equally soft springs have ordinary tie ratios. A
    coordinate with a stiffness diagonal and no coupling at all, such as a
    rotation stabilized by a diagonal penalty, is uncoupled rather than
    weakly tied: it is kept and reported with one warning. A returned mode
    whose eigenpair residual exceeds ``1e-3`` is not an eigenvector and
    raises; a residual above ``1e-5`` warns.

    Finally :meth:`reduce` runs the acceptance checks of
    :meth:`ModalGeneratorCraigBampton.build` on the reduced matrices: rigid
    invariance, exactly six rigid-body modes, no further zero-energy mode,
    rigid recovery, and a physical rigid mass block. A model that
    :meth:`build` would refuse for one of them, a soft cluster among them,
    raises here, after ``report`` is set. The softest elastic mode must lie
    above ``rigid_mode_tolerance`` times the largest eigenvalue of the
    reduced model; ``report["reduced_elastic_eigenvalue_ratio"]`` is that
    ratio. A stiff reduced coordinate, such as a small rigid interface
    patch, lowers it, so for a sound model that comes close to the limit
    ``rigid_mode_tolerance`` may be lowered (the rigid-body eigenvalues of
    a healthy reduced model lie near ``1e-14`` of the largest one).

    After :meth:`reduce` (called by :meth:`build`), the generator exposes
    ``transformation`` (``u = T_z z`` for the coordinates ``z`` after
    constraint and interface elimination, interior coordinates first and the
    ``6 * interface_count`` frame coordinates last), ``constraint_modes``,
    ``fixed_interface_modes``, ``fixed_interface_eigenvalues`` [1/s^2],
    ``fixed_interface_frequencies`` [Hz], ``static_correction_vectors``,
    ``static_correction_frequencies`` [Hz], ``static_correction_load_indices``
    (the load of each retained vector), ``reduction_basis`` (the
    Craig-Bampton matrix ``T`` in ``z`` coordinates), the reduced matrices,
    ``recovery_matrix`` and, for six-DOF nodes, ``rotation_recovery_matrix``
    of the sample nodes, :meth:`full_recovery_matrix`, ``craig_bampton`` (the
    :class:`ModalGeneratorCraigBampton`), and ``report`` with sizes,
    residuals, conditioning, the zero-energy decision
    (``zero_energy_decision``, ``zero_energy_ratio_min`` and
    ``zero_energy_tie_ratio_min`` against ``zero_energy_ratio_tolerance``,
    ``interior_stiffness_pivot_ratio`` against
    ``interior_stiffness_pivot_limit``, ``reduced_elastic_eigenvalue_ratio``
    against ``reduced_elastic_eigenvalue_limit``, and the range of the
    equilibration scaling), and stage timings.

    The reduced model must reproduce rigid-body motion: zero stiffness and
    damping on the rigid motion ``R`` of the interface frames, a recovery
    that maps ``R`` to the rigid motion of the samples, and a rigid block
    ``R^T M R`` with a physical mass and center of mass. A stiffness matrix
    that resists rigid motion slightly, as the shells of a non-conformal
    (for example ray-cast) mid-surface mesh can, leaves a small residual
    that :meth:`ModalGeneratorCraigBampton.build` rejects with the default
    ``rigid_consistency="check"``. ``rigid_consistency="project"`` removes
    it in :meth:`reduce` with the smallest (Frobenius-norm) corrections that
    make the model exactly consistent: ``K <- P^T K P`` and ``D <- P^T D P``
    with the orthogonal projector ``P = I - R R^+``, ``X <- X + (X_R - X R)
    R^+`` for the sample and full recoveries ``X`` with their exact rigid
    motion ``X_R``, and ``M <- M + (R^+)^T (S - R^T M R) R^+``, where ``S``
    is ``R^T M R`` with an isotropic translational mass and a skew-symmetric
    mass moment. It raises instead when a residual exceeds
    ``rigid_projection_limit`` or when the projection moves an elastic
    eigenvalue of the reduced model by more than
    ``rigid_projection_eigenvalue_tolerance``: such a residual is a modeling
    error, such as a support left in the stiffness matrix, not discretization
    drift. ``report["rigid_projection"]`` records the residuals before and
    after the projection and the largest eigenvalue shift; the reduced
    matrices, recoveries, and :meth:`to_arrays` are the projected ones.

    Args:
        node_positions: Rest node positions [m], shape ``[node_count, 3]``.
        mass_matrix: Nodal mass matrix [kg, kg m^2 for rotational DOFs],
            shape ``[dofs_per_node * node_count, dofs_per_node * node_count]``,
            node-major DOFs; dense or SciPy sparse.
        stiffness_matrix: Nodal stiffness matrix [N/m, N m/rad], with the
            shape of ``mass_matrix``; dense or SciPy sparse. It must preserve
            rigid-body motion (free-free model).
        interfaces: Interface specifications as described above. Their order
            defines the six-DOF blocks of the reduced matrices.
        fixed_mode_count: Number of fixed-interface modes to retain.
        static_correction_loads: Optional nodal loads [N, N m for rotational
            DOFs] whose static response the reduced model reproduces
            exactly, shape ``[dof_count, load_count]`` or ``[dof_count]``,
            node-major DOFs as for ``stiffness_matrix``; dense or SciPy
            sparse. Only the direction of each load matters. Each load adds
            at most one coordinate, as described above.
        damping_matrix: Optional nodal damping matrix [N s/m], with the shape
            of ``mass_matrix``; dense or SciPy sparse.
        dofs_per_node: DOFs per node: 3 (``[ux, uy, uz]``) or 6
            (``[ux, uy, uz, rx, ry, rz]``).
        constraints: Optional multi-point constraint matrix ``C`` with
            ``C u = 0``, eliminated before the reduction; see
            :class:`ConstraintElimination`. Mutually exclusive with
            ``transformation``.
        transformation: Optional transformation ``T`` with ``u = T q``, shape
            ``[dof_count, independent_count]``, applied instead of
            ``constraints``.
        sample_node_indices: Nodes used as recovery samples of the basis.
            Defaults to all nodes; pass a subset for large models.
        sample_mass: Lumped sample masses [kg] for the centrifugal, Coriolis,
            and mode-pair inertia integrals; see
            :class:`ModalGeneratorCraigBampton`. ``"lumped"`` uses the
            row-sum lumped translational mass of each sample node, which
            covers the whole body only when every node is a sample. ``None``
            leaves those integrals unavailable.
        label: Optional basis label.
        eigensolver: Fixed-interface eigensolver. ``"dense"`` uses LAPACK on
            dense matrices and does not need SciPy. ``"arpack"`` keeps the
            matrices sparse and needs SciPy. ``"auto"`` selects ``"arpack"``
            when any input matrix is sparse or the DOF count exceeds
            ``dense_dof_limit`` and SciPy is installed.
        arpack_shift: ARPACK shift ``sigma`` [1/s^2]. ``None`` uses ``0``.
        arpack_tol: ARPACK relative eigenvalue accuracy; ``0`` requests
            machine precision.
        arpack_maxiter: Maximum ARPACK restart iterations; ``None`` uses
            SciPy's default.
        arpack_ncv: Number of ARPACK Lanczos vectors; ``None`` uses SciPy's
            default.
        dense_dof_limit: Largest DOF count that ``eigensolver="auto"`` reduces
            densely for dense input.
        constraint_tolerance: Relative tolerance of the constraint
            elimination; see :class:`ConstraintElimination`.
        eigenvalue_tolerance: Smallest energy ratio of an interior motion
            in the equilibrated stiffness metric (see above), at or below
            which the interior is reported as a mechanism. Values below the
            roundoff of the ratio, ``8 n eps``, are raised to it.
        rigid_mode_tolerance: Passed to :class:`ModalGeneratorCraigBampton`.
        damping_coupling_tolerance: Passed to :class:`ModalGeneratorCraigBampton`.
        damping_projection: Passed to :class:`ModalGeneratorCraigBampton`.
        interface_match_tolerance: Passed to :class:`ModalGeneratorCraigBampton`.
        interface_recovery_tolerance: Passed to :class:`ModalGeneratorCraigBampton`.
        rigid_recovery_tolerance: Passed to :class:`ModalGeneratorCraigBampton`.
        rigid_consistency: ``"check"`` rejects a reduced model that does not
            reproduce rigid-body motion; ``"project"`` projects a small
            rigid-body residual out, as described above.
        rigid_projection_limit: Largest relative rigid-body residual of the
            reduced stiffness, damping, recovery, or rigid mass block that
            ``rigid_consistency="project"`` removes; a larger one raises.
        rigid_projection_eigenvalue_tolerance: Largest relative change of any
            elastic eigenvalue of the reduced model that the projection may
            cause; a larger one raises.
    """

    _ARPACK_MARGIN: ClassVar[int] = 2
    """Extra eigenpairs computed beyond the retained ones so the last retained mode is converged."""

    _SOLVE_CHUNK_COLUMNS: ClassVar[int] = 64
    """Number of constraint-mode right-hand sides solved at once."""

    _STATIC_CORRECTION_SPAN_TOLERANCE: ClassVar[float] = 1.0e-10
    """Mass norm of a static response after orthogonalization, relative to its norm before, below which
    the retained basis already represents it and no correction vector is added."""

    _ARPACK_RESTARTS: ClassVar[int] = 8
    """Searches for eigenpairs that the Sturm sequence count shows missing. Each finds further copies of a
    repeated eigenvalue, such as that of many identical uncoupled coordinates."""

    _SOFT_TIE_RATIO: ClassVar[float] = 1.0e-6
    """Relative tie stiffness of a coordinate below which :meth:`reduce` warns of a soft part."""

    _EIGEN_RESIDUAL_WARNING: ClassVar[float] = 1.0e-5
    """Relative residual of a fixed-interface eigenpair above which :meth:`reduce` warns."""

    _EIGEN_RESIDUAL_LIMIT: ClassVar[float] = 1.0e-3
    """Relative residual of a fixed-interface eigenpair above which :meth:`reduce` raises: the returned
    vector is not an eigenvector. Resolved modes of ill-conditioned but sound models stay below ``2e-4``."""

    _DENSE_EIGENVALUE_AGREEMENT: ClassVar[float] = 1.0e-8
    """Relative gap between a dense eigenvalue and the Rayleigh quotient of its vector above which the
    dense path solves the inverse eigenproblem instead."""

    _RIGID_RESIDUAL_TOLERANCE: ClassVar[float] = 1.0e-7
    """Relative rigid-body residual of the reduced stiffness above which :meth:`reduce` raises.

    The same threshold as in :meth:`ModalGeneratorCraigBampton.build`.
    """

    def __init__(
        self,
        node_positions: Sequence[Sequence[float]] | np.ndarray,
        mass_matrix: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray,
        stiffness_matrix: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray,
        interfaces: Sequence[Mapping[str, Any]],
        *,
        fixed_mode_count: int,
        static_correction_loads: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray | None = None,
        damping_matrix: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray | None = None,
        dofs_per_node: Literal[3, 6] = 6,
        constraints: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray | None = None,
        transformation: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray | None = None,
        sample_node_indices: Sequence[int] | np.ndarray | None = None,
        sample_mass: Literal["lumped"] | Sequence[float] | np.ndarray | None = None,
        label: str | None = None,
        eigensolver: Literal["auto", "dense", "arpack"] = "auto",
        arpack_shift: float | None = None,
        arpack_tol: float = 0.0,
        arpack_maxiter: int | None = None,
        arpack_ncv: int | None = None,
        dense_dof_limit: int = 2000,
        constraint_tolerance: float = 1.0e-10,
        eigenvalue_tolerance: float = 1.0e-12,
        rigid_mode_tolerance: float = 1.0e-8,
        damping_coupling_tolerance: float = 1.0e-5,
        damping_projection: Literal["diagonal", "error"] = "diagonal",
        interface_match_tolerance: float = 1.0e-7,
        interface_recovery_tolerance: float = 1.0e-6,
        rigid_recovery_tolerance: float = 1.0e-6,
        rigid_consistency: Literal["check", "project"] = "check",
        rigid_projection_limit: float = 1.0e-3,
        rigid_projection_eigenvalue_tolerance: float = 1.0e-2,
    ):
        if int(dofs_per_node) not in (3, 6):
            raise ValueError(f"dofs_per_node must be 3 or 6, got {dofs_per_node}")
        self.dofs_per_node = int(dofs_per_node)
        self.node_positions = np.asarray(node_positions, dtype=np.float64)
        if self.node_positions.ndim != 2 or self.node_positions.shape[1] != 3:
            raise ValueError(f"node_positions must have shape [node_count, 3], got {self.node_positions.shape}")
        self.node_count = int(self.node_positions.shape[0])
        self.dof_count = self.dofs_per_node * self.node_count

        if eigensolver not in ("auto", "dense", "arpack"):
            raise ValueError(f"eigensolver must be 'auto', 'dense', or 'arpack', got '{eigensolver}'")
        self.eigensolver = eigensolver
        self.arpack_shift = None if arpack_shift is None else float(arpack_shift)
        self.arpack_tol = float(arpack_tol)
        if self.arpack_tol < 0.0:
            raise ValueError(f"arpack_tol must be non-negative, got {self.arpack_tol}")
        self.arpack_maxiter = None if arpack_maxiter is None else int(arpack_maxiter)
        self.arpack_ncv = None if arpack_ncv is None else int(arpack_ncv)
        self.dense_dof_limit = int(dense_dof_limit)
        self.constraint_tolerance = float(constraint_tolerance)
        self.eigenvalue_tolerance = float(eigenvalue_tolerance)
        if self.eigenvalue_tolerance < 0.0:
            raise ValueError(f"eigenvalue_tolerance must be non-negative, got {self.eigenvalue_tolerance}")
        self.fixed_mode_count = int(fixed_mode_count)
        if self.fixed_mode_count < 0:
            raise ValueError(f"fixed_mode_count must be non-negative, got {self.fixed_mode_count}")
        self.label = label
        self._craig_bampton_options = {
            "rigid_mode_tolerance": rigid_mode_tolerance,
            "damping_coupling_tolerance": damping_coupling_tolerance,
            "damping_projection": damping_projection,
            "interface_match_tolerance": interface_match_tolerance,
            "interface_recovery_tolerance": interface_recovery_tolerance,
            "rigid_recovery_tolerance": rigid_recovery_tolerance,
        }
        if rigid_consistency not in ("check", "project"):
            raise ValueError(f"rigid_consistency must be 'check' or 'project', got '{rigid_consistency}'")
        self.rigid_consistency = rigid_consistency
        self.rigid_projection_limit = float(rigid_projection_limit)
        if not self.rigid_projection_limit >= 0.0:
            raise ValueError(f"rigid_projection_limit must be non-negative, got {self.rigid_projection_limit}")
        self.rigid_projection_eigenvalue_tolerance = float(rigid_projection_eigenvalue_tolerance)
        if not self.rigid_projection_eigenvalue_tolerance >= 0.0:
            raise ValueError(
                "rigid_projection_eigenvalue_tolerance must be non-negative, "
                f"got {self.rigid_projection_eigenvalue_tolerance}"
            )

        size = self.dof_count
        self.mass_matrix = modal_sparse.coerce_symmetric_matrix(mass_matrix, size, "mass_matrix")
        self.stiffness_matrix = modal_sparse.coerce_symmetric_matrix(stiffness_matrix, size, "stiffness_matrix")
        self.damping_matrix = (
            None
            if damping_matrix is None
            else modal_sparse.coerce_symmetric_matrix(damping_matrix, size, "damping_matrix")
        )
        self.static_correction_loads: np.ndarray | None = None
        if static_correction_loads is not None:
            loads = np.array(modal_sparse.to_dense(static_correction_loads), dtype=np.float64)
            if loads.ndim == 1:
                loads = loads[:, None]
            if loads.ndim != 2 or loads.shape[0] != size:
                raise ValueError(
                    f"static_correction_loads must have shape [{size}, load_count] or [{size}], got "
                    f"{tuple(np.shape(static_correction_loads))}"
                )
            if not np.all(np.isfinite(loads)):
                raise ValueError("static_correction_loads contains non-finite entries")
            self.static_correction_loads = loads
        if constraints is not None and transformation is not None:
            raise ValueError("Pass at most one of constraints and transformation")
        self.constraints = constraints
        self.input_transformation = transformation

        self.interfaces = self._coerce_interfaces(interfaces)
        self.interface_count = len(self.interfaces)
        self.interface_names = tuple(interface.name for interface in self.interfaces)
        self.interface_positions = np.array([interface.position for interface in self.interfaces], dtype=np.float64)

        if sample_node_indices is None:
            self.sample_node_indices = np.arange(self.node_count, dtype=np.int64)
        else:
            self.sample_node_indices = np.asarray(sample_node_indices, dtype=np.int64).reshape(-1)
            if self.sample_node_indices.shape[0] == 0:
                raise ValueError("sample_node_indices must contain at least one node")
            if np.any(self.sample_node_indices < 0) or np.any(self.sample_node_indices >= self.node_count):
                raise ValueError("sample_node_indices contains out-of-range node indices")
        self.sample_points = self.node_positions[self.sample_node_indices]
        if isinstance(sample_mass, str):
            if sample_mass != "lumped":
                raise ValueError(f"sample_mass must be 'lumped', an array, or None, got '{sample_mass}'")
        elif sample_mass is not None:
            sample_mass = np.asarray(sample_mass, dtype=np.float64).reshape(-1)
            if sample_mass.shape[0] != self.sample_node_indices.shape[0]:
                raise ValueError(
                    f"sample_mass must have length {self.sample_node_indices.shape[0]}, got {sample_mass.shape[0]}"
                )
        self.sample_mass = sample_mass

        self.craig_bampton: ModalGeneratorCraigBampton | None = None
        self.report: dict[str, Any] = {}
        self._zero_energy: dict[str, Any] = {}
        self._interior_solve: Any = None
        self._interior_labels: tuple[np.ndarray, int, np.ndarray | None, bool] | None = None
        self._rigid_projection: tuple[np.ndarray, np.ndarray] | None = None

    def _coerce_interfaces(self, interfaces: Sequence[Mapping[str, Any]]) -> list[_Interface]:
        """Validate the interface specifications and fill in their defaults."""
        if len(interfaces) == 0:
            raise ValueError("At least one interface is required")
        result: list[_Interface] = []
        exclusive: dict[int, str] = {}
        for index, spec in enumerate(interfaces):
            unknown = set(spec) - {"name", "node_indices", "position", "coupling", "weights"}
            if unknown:
                raise ValueError(f"Interface {index} has unknown keys {sorted(unknown)}")
            name = str(spec.get("name", f"interface_{index}"))
            if "node_indices" not in spec:
                raise ValueError(f"Interface '{name}' requires node_indices")
            nodes = np.asarray(spec["node_indices"], dtype=np.int64).reshape(-1)
            if nodes.shape[0] == 0:
                raise ValueError(f"Interface '{name}' requires at least one node")
            if np.any(nodes < 0) or np.any(nodes >= self.node_count):
                raise ValueError(f"Interface '{name}' contains out-of-range node indices")
            if np.unique(nodes).shape[0] != nodes.shape[0]:
                raise ValueError(f"Interface '{name}' lists a node more than once")
            coupling_name = str(spec.get("coupling", "rbe2"))
            if coupling_name not in _COUPLINGS:
                raise ValueError(
                    f"Interface '{name}' coupling must be one of {sorted(_COUPLINGS)}, got '{coupling_name}'"
                )
            coupling = _COUPLINGS[coupling_name]
            points = self.node_positions[nodes]
            if coupling == "node":
                if nodes.shape[0] != 1:
                    raise ValueError(
                        f"Interface '{name}' with coupling 'node' needs exactly one node, got {nodes.shape[0]}"
                    )
                if self.dofs_per_node != 6:
                    raise ValueError(
                        f"Interface '{name}' with coupling 'node' needs dofs_per_node=6: a three-DOF node cannot "
                        "define a six-DOF frame. Use coupling 'rbe2' or 'rbe3' with at least three non-collinear nodes"
                    )
                default_position = points[0]
            else:
                default_position = np.mean(points, axis=0)
            position = np.asarray(spec.get("position", default_position), dtype=np.float64).reshape(-1)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError(f"Interface '{name}' position must be three finite values")
            if coupling == "node" and np.max(np.abs(position - points[0])) > 0.0:
                raise ValueError(f"Interface '{name}' with coupling 'node' must be positioned at its node")
            weights = np.ones(nodes.shape[0], dtype=np.float64)
            if "weights" in spec:
                if coupling != "rbe3":
                    raise ValueError(f"Interface '{name}': weights apply only to coupling 'rbe3'")
                weights = np.asarray(spec["weights"], dtype=np.float64).reshape(-1)
                if weights.shape[0] != nodes.shape[0] or not np.all(weights > 0.0) or not np.all(np.isfinite(weights)):
                    raise ValueError(f"Interface '{name}' weights must be positive, one per node")
            if coupling == "rbe3" or (coupling == "rbe2" and self.dofs_per_node == 3):
                singular = np.linalg.svd(_patch_rigid_rows(points, position), compute_uv=False)
                if singular.shape[0] < 6 or singular[5] <= _RANK_TOLERANCE * max(singular[0], 1.0):
                    raise ValueError(
                        f"Interface '{name}' with coupling '{coupling_name}' needs at least three non-collinear "
                        "nodes to define a six-DOF frame"
                    )
            if coupling in ("node", "rbe2"):
                for node in nodes.tolist():
                    if node in exclusive:
                        raise ValueError(
                            f"Node {node} belongs to interfaces '{exclusive[node]}' and '{name}'; a node of a 'node' "
                            "or 'rbe2' interface cannot belong to another interface"
                        )
                    exclusive[node] = name
            result.append(_Interface(name, nodes, position, coupling, weights))
        names = [interface.name for interface in result]
        if len(set(names)) != len(names):
            raise ValueError("Interface names must be unique")
        for interface in result:
            if interface.coupling == "rbe3":
                shared = [int(node) for node in interface.nodes if int(node) in exclusive]
                if shared:
                    raise ValueError(
                        f"Node {shared[0]} belongs to interfaces '{exclusive[shared[0]]}' and '{interface.name}'; a "
                        "node of a 'node' or 'rbe2' interface cannot belong to another interface"
                    )
        return result

    def _uses_sparse(self) -> bool:
        """Return whether the reduction keeps the matrices sparse and uses ARPACK."""
        if self.eigensolver == "auto":
            matrices = (self.mass_matrix, self.stiffness_matrix, self.damping_matrix, self.constraints)
            if any(modal_sparse.is_sparse(matrix) for matrix in matrices):
                return True
            if modal_sparse.is_sparse(self.input_transformation):
                return True
            return self.dof_count > self.dense_dof_limit and modal_sparse.has_scipy()
        return self.eigensolver == "arpack"

    def _describe_dof(self, dof: int) -> str:
        """Return ``"DOF <dof> (node <node> <component>)"`` for an original DOF index."""
        node, component = divmod(int(dof), self.dofs_per_node)
        return f"DOF {int(dof)} (node {node} {_DOF_COMPONENTS[component]})"

    def _describe_coordinate(
        self, coordinate: int, coordinate_count: int, independent_dofs: np.ndarray | None, transformed: bool
    ) -> str:
        """Name an augmented coordinate ``[q, f]`` by the user's DOF, interface frame, or transformation column."""
        if coordinate >= coordinate_count:
            index, component = divmod(coordinate - coordinate_count, 6)
            return f"interface '{self.interfaces[index].name}' frame {_FRAME_COMPONENTS[component]}"
        if independent_dofs is not None:
            return self._describe_dof(int(independent_dofs[coordinate]))
        if transformed:
            return f"transformation column {coordinate}"
        return self._describe_dof(coordinate)

    def _describe_interior(self, index: int) -> str:
        """Name interior coordinate ``index`` of the last elimination for error messages."""
        if self._interior_labels is None:
            return f"interior coordinate {int(index)}"
        independent, coordinate_count, independent_dofs, transformed = self._interior_labels
        return self._describe_coordinate(int(independent[int(index)]), coordinate_count, independent_dofs, transformed)

    def _interface_rows(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, list[tuple[int, int]]]:
        """Return the interface constraint triplets ``G u - H f = 0`` on DOFs ``u`` and frame coordinates ``f``.

        Returns:
            Row, DOF, and value arrays of ``G``; row, frame-coordinate, and
            value arrays of ``H``; the row count; and, per row, the interface
            index and the coupled DOF (``"node"`` and ``"rbe2"``) or ``-1 -
            frame component`` (``"rbe3"``), for error messages.
        """
        dofs = self.dofs_per_node
        g_rows, g_cols, g_values = [], [], []
        h_rows, h_cols, h_values = [], [], []
        origins: list[tuple[int, int]] = []
        row = 0
        for index, interface in enumerate(self.interfaces):
            frame = 6 * index
            points = self.node_positions[interface.nodes]
            rigid = _patch_rigid_rows(points, interface.position)
            if interface.coupling in ("node", "rbe2"):
                for local, node in enumerate(interface.nodes.tolist()):
                    for axis in range(3):
                        g_rows.append(row)
                        g_cols.append(dofs * node + axis)
                        g_values.append(1.0)
                        coefficients = rigid[3 * local + axis]
                        nonzero = np.flatnonzero(coefficients)
                        h_rows.extend([row] * nonzero.shape[0])
                        h_cols.extend((frame + nonzero).tolist())
                        h_values.extend(coefficients[nonzero].tolist())
                        origins.append((index, dofs * node + axis))
                        row += 1
                    if dofs == 6:
                        for axis in range(3):
                            g_rows.append(row)
                            g_cols.append(dofs * node + 3 + axis)
                            g_values.append(1.0)
                            h_rows.append(row)
                            h_cols.append(frame + 3 + axis)
                            h_values.append(1.0)
                            origins.append((index, dofs * node + 3 + axis))
                            row += 1
            else:
                weights = np.repeat(interface.weights, 3)
                weighted = rigid.T * weights
                fit = np.linalg.solve(weighted @ rigid, weighted)
                translation_dofs = (dofs * interface.nodes[:, None] + np.arange(3)).reshape(-1)
                for axis in range(6):
                    nonzero = np.flatnonzero(fit[axis])
                    g_rows.extend([row] * nonzero.shape[0])
                    g_cols.extend(translation_dofs[nonzero].tolist())
                    g_values.extend(fit[axis, nonzero].tolist())
                    h_rows.append(row)
                    h_cols.append(frame + axis)
                    h_values.append(1.0)
                    origins.append((index, -1 - axis))
                    row += 1
        return (
            np.asarray(g_rows, dtype=np.int64),
            np.asarray(g_cols, dtype=np.int64),
            np.asarray(g_values, dtype=np.float64),
            np.asarray(h_rows, dtype=np.int64),
            np.asarray(h_cols, dtype=np.int64),
            np.asarray(h_values, dtype=np.float64),
            row,
            origins,
        )

    def _coordinate_transformation(self, sparse: bool) -> tuple[Any, dict[str, Any] | None, dict[str, Any]]:
        """Return ``T_z`` with ``u = T_z z``, frame coordinates last, and the elimination reports."""
        size = self.dof_count
        frame_count = 6 * self.interface_count
        mpc_report = None
        base = None
        independent_dofs = None
        eliminated = np.zeros(size, dtype=bool)
        if self.constraints is not None:
            result = constraint_transformation(
                self.constraints, size, tolerance=self.constraint_tolerance, sparse=sparse
            )
            base = result.transformation
            mpc_report = result.report
            independent_dofs = result.independent_dofs
            eliminated[result.dependent_dofs] = True
        elif self.input_transformation is not None:
            base = self.input_transformation
            base = modal_sparse.to_csc(base) if sparse else np.asarray(modal_sparse.to_dense(base))
            if base.ndim != 2 or base.shape[0] != size:
                raise ValueError(f"transformation must have shape [{size}, independent_count], got {base.shape}")
            check_transformation_rank(base)
            mpc_report = {"independent_dofs": int(base.shape[1]), "dependent_dofs": None}
        coordinate_count = size if base is None else int(base.shape[1])

        g_rows, g_cols, g_values, h_rows, h_cols, h_values, row_count, origins = self._interface_rows()
        interface_dofs = assemble_triplets(g_rows, g_cols, g_values, (row_count, size), sparse)
        if base is not None:
            interface_dofs = interface_dofs @ base
        if sparse:
            interface_dofs = interface_dofs.tocoo()
            u_rows, u_cols, u_values = interface_dofs.row, interface_dofs.col, interface_dofs.data
        else:
            u_rows, u_cols = np.nonzero(interface_dofs)
            u_values = interface_dofs[u_rows, u_cols]
        extended = assemble_triplets(
            np.concatenate((u_rows, h_rows)),
            np.concatenate((u_cols, coordinate_count + h_cols)),
            np.concatenate((u_values, -h_values)),
            (row_count, coordinate_count + frame_count),
            sparse,
        )

        # Name the user's DOFs and the interfaces in errors, not the augmented coordinates [q, f].
        def describe_coordinate(coordinate: int) -> str:
            return self._describe_coordinate(coordinate, coordinate_count, independent_dofs, base is not None)

        def describe_row(row: int) -> str:
            index, dof = origins[row]
            name = self.interfaces[index].name
            if dof < 0:
                return f"The rigid-fit row {_FRAME_COMPONENTS[-1 - dof]} of interface '{name}'"
            text = f"The coupling of interface '{name}' to {self._describe_dof(dof)}"
            return f"{text}, which the constraints eliminate," if eliminated[dof] else text

        result = constraint_transformation(
            extended,
            coordinate_count + frame_count,
            retained_dofs=np.arange(coordinate_count, coordinate_count + frame_count),
            tolerance=self.constraint_tolerance,
            sparse=sparse,
            dof_label=describe_coordinate,
            row_label=describe_row,
        )
        independent = result.independent_dofs
        if not np.array_equal(independent[-frame_count:], np.arange(coordinate_count, coordinate_count + frame_count)):
            raise RuntimeError("Interface frame coordinates were not retained as independent coordinates")
        # Interior coordinate i is augmented coordinate independent[i]; name it in mechanism errors.
        self._interior_labels = (independent, coordinate_count, independent_dofs, base is not None)
        transform = result.transformation[:coordinate_count]
        if base is not None:
            transform = base @ transform
        if sparse:
            transform = modal_sparse.to_csc(transform)
        return transform, mpc_report, result.report

    def _interior_scaling(self, stiffness_ii) -> np.ndarray:
        """Return the equilibration scaling ``diag(K_ii)^-1/2``; a non-positive diagonal entry is a mechanism."""
        diagonal = np.asarray(stiffness_ii.diagonal(), dtype=np.float64).reshape(-1)
        invalid = np.flatnonzero(~(diagonal > 0.0) | ~np.isfinite(diagonal))
        if invalid.shape[0] > 0:
            first = int(invalid[0])
            raise ValueError(
                self._mechanism_message(
                    detail=f"the interior stiffness diagonal of {self._describe_interior(first)} is "
                    f"{diagonal[first]:.6g}; {invalid.shape[0]} interior coordinates have a non-positive diagonal"
                )
            )
        return 1.0 / np.sqrt(diagonal)

    def _check_interior_mass(self, mass_ii) -> None:
        """Raise unless the interior mass matrix is positive definite."""
        diagonal = np.asarray(mass_ii.diagonal(), dtype=np.float64).reshape(-1)
        invalid = np.flatnonzero(~(diagonal > 0.0) | ~np.isfinite(diagonal))
        if invalid.shape[0] > 0:
            first = int(invalid[0])
            raise ValueError(
                "mass_matrix must be positive definite on the interior DOFs: the mass diagonal of "
                f"{self._describe_interior(first)} is {diagonal[first]:.6g}"
            )
        if not modal_sparse.is_sparse(mass_ii) or mass_ii.nnz == mass_ii.shape[0]:
            # Dense matrices are factored by the caller; a diagonal matrix is certified by its diagonal.
            return
        # The inertia of the equilibrated matrix is that of M_ii (Sylvester's law).
        pivots = modal_sparse.nonpositive_pivot_count(modal_sparse.scale_symmetric(mass_ii, 1.0 / np.sqrt(diagonal)))
        if pivots > 0:
            raise ValueError(
                "mass_matrix must be positive definite on the interior DOFs: its factorization has "
                f"{pivots} non-positive pivots"
            )
        if pivots < 0:
            warnings.warn(
                "The positive definiteness of the interior mass matrix could not be verified: its factorization "
                "pivoted off the diagonal, so its inertia is unavailable",
                stacklevel=3,
            )

    def _energy_ratio_tolerance(self, stiffness_ii) -> float:
        """Return the smallest accepted energy ratio, ``max(eigenvalue_tolerance, 8 n eps)``.

        ``n`` is the largest number of nonzeros in a row of ``K_ii`` on both paths: the roundoff of a
        quadratic form grows with it.
        """
        row_nonzeros = max(modal_sparse.max_row_nonzeros(stiffness_ii), 1)
        return max(self.eigenvalue_tolerance, 8.0 * row_nonzeros * np.finfo(np.float64).eps)

    def _weakest_tie(self, stiffness_z, interior_count: int) -> tuple[float, int, np.ndarray]:
        """Return the smallest relative tie stiffness of an interior coordinate, that coordinate, and the
        uncoupled interior coordinates.

        The relative tie stiffness of coordinate ``i`` is ``(sum_{j != i} |K_ij| / sqrt(K_ii K_jj))^2``
        over all coordinates ``j``, interface frames included. For a coordinate on a single spring it is
        the spring stiffness over the stiffness of the coordinate it is attached to, and it does not
        depend on the units of the DOFs. Coordinates with a non-positive diagonal are left to the
        interior solve, which names them. A coordinate without any off-diagonal entry is uncoupled, not
        weakly tied: it is listed separately and does not enter the minimum.
        """
        diagonal = np.asarray(stiffness_z.diagonal(), dtype=np.float64).reshape(-1)
        positive = (diagonal > 0.0) & np.isfinite(diagonal)
        scale = np.zeros_like(diagonal)
        scale[positive] = 1.0 / np.sqrt(diagonal[positive])
        scaled = abs(modal_sparse.scale_symmetric(stiffness_z, scale))
        row_sums = np.asarray(scaled.sum(axis=1)).reshape(-1) - np.asarray(scaled.diagonal()).reshape(-1)
        uncoupled = np.flatnonzero((positive & (row_sums == 0.0))[:interior_count])
        coupling = np.where(positive & (row_sums > 0.0), row_sums, math.inf)[:interior_count]
        if coupling.shape[0] == 0 or not np.any(np.isfinite(coupling)):
            return math.inf, -1, uncoupled
        weakest = int(np.argmin(coupling))
        return float(coupling[weakest]) ** 2, weakest, uncoupled

    def _zero_energy_check(
        self,
        scaling: np.ndarray,
        stiffness_norm: float,
        tolerance: float,
        softest_value: float,
        softest_vector: np.ndarray,
        values: np.ndarray,
        mode_energy: np.ndarray,
        mode_norm_squared: np.ndarray,
    ) -> float:
        """Decide whether the interior has a zero-energy mechanism from equilibrated energy ratios.

        Args:
            scaling: Equilibration scaling ``d = diag(K_ii)^-1/2``.
            stiffness_norm: One norm of the equilibrated stiffness ``K_s``.
            tolerance: Smallest accepted energy ratio.
            softest_value: Estimate of the smallest eigenvalue of ``K_s``.
            softest_vector: Unit vector ``y`` of that estimate; may be empty.
            values: Computed fixed-interface eigenvalues [1/s^2].
            mode_energy: ``y_j^T K_s y_j`` of each computed mode ``phi_j = d y_j``.
            mode_norm_squared: ``y_j^T y_j`` of each computed mode.

        Returns:
            The zero-energy threshold [1/s^2] of the mode with the smallest
            ratio: the eigenvalue at which its ratio would equal ``tolerance``.

        Raises:
            ValueError: If a ratio is at or below ``tolerance``.
        """
        tiny = np.finfo(np.float64).tiny
        norm = max(float(stiffness_norm), tiny)
        stiffness_ratio = float(softest_value) / norm
        mode_ratios = np.asarray(mode_energy, dtype=np.float64) / (norm * np.maximum(mode_norm_squared, tiny))
        mode_ratio_min = float(np.min(mode_ratios)) if mode_ratios.shape[0] else math.inf
        threshold = 0.0
        scale = 0.0
        if mode_ratios.shape[0]:
            worst = int(np.argmin(mode_ratios))
            # lambda_j = ratio_j * scale_j with scale_j = ||K_s|| y_j^T y_j / y_j^T M_s y_j.
            scale = float(values[worst]) / mode_ratios[worst] if mode_ratios[worst] > 0.0 else math.inf
            threshold = tolerance * scale
        ratio_min = min(stiffness_ratio, mode_ratio_min)
        self._zero_energy = {
            "zero_energy_decision": "equilibrated_energy_ratio",
            "zero_energy_ratio_min": ratio_min if math.isfinite(ratio_min) else None,
            "zero_energy_ratio_tolerance": float(tolerance),
            "zero_energy_mode_ratio_min": mode_ratio_min if math.isfinite(mode_ratio_min) else None,
            "zero_energy_stiffness_ratio": stiffness_ratio if math.isfinite(stiffness_ratio) else None,
            "zero_energy_scale": float(scale),
            "zero_energy_scale_source": "equilibrated_energy_ratio",
            "interior_stiffness_scaling_min": float(np.min(scaling, initial=math.inf)) if scaling.shape[0] else None,
            "interior_stiffness_scaling_max": float(np.max(scaling, initial=0.0)) if scaling.shape[0] else None,
            "interior_stiffness_pivot_limit": modal_sparse.SINGULAR_PIVOT_RATIO,
        }
        if stiffness_ratio <= tolerance:
            detail = (
                f"the softest direction of the equilibrated interior stiffness has the energy ratio "
                f"{stiffness_ratio:.3g}, at or below the tolerance {tolerance:.3g}"
            )
            if softest_vector.shape[0] == scaling.shape[0] and scaling.shape[0] > 0:
                largest = int(np.argmax(np.abs(scaling * softest_vector)))
                detail += f"; its motion is largest at {self._describe_interior(largest)}"
            raise ValueError(self._mechanism_message(detail=detail))
        if mode_ratios.shape[0] and mode_ratios[worst] <= tolerance:
            raise ValueError(
                self._mechanism_message(
                    float(values[worst]),
                    threshold,
                    detail=f"energy ratio {mode_ratios[worst]:.3g} in the equilibrated stiffness metric against "
                    f"the tolerance {tolerance:.3g}",
                )
            )
        return threshold

    def _interior_dense(self, stiffness_ii, stiffness_ib, mass_ii, count: int):
        """Solve the constraint modes and fixed-interface modes densely."""
        scaling = self._interior_scaling(stiffness_ii)
        self._check_interior_mass(mass_ii)
        try:
            chol_k = np.linalg.cholesky(stiffness_ii)
        except np.linalg.LinAlgError as exc:
            raise ValueError(self._mechanism_message()) from exc
        self._interior_solve = lambda rhs: np.linalg.solve(chol_k.T, np.linalg.solve(chol_k, rhs))
        constraint_modes = -np.linalg.solve(chol_k.T, np.linalg.solve(chol_k, stiffness_ib))
        # Pivots of the equilibrated factor: the Cholesky factor of D K D is D L.
        diagonal = (scaling * np.abs(np.diag(chol_k))) ** 2
        pivot_ratio = float(np.min(diagonal) / np.max(diagonal)) if diagonal.shape[0] else 1.0
        try:
            chol_m = np.linalg.cholesky(mass_ii)
        except np.linalg.LinAlgError as exc:
            raise ValueError("mass_matrix must be positive definite on the interior DOFs") from exc
        standard = np.linalg.solve(chol_m, np.linalg.solve(chol_m, stiffness_ii).T).T
        values, vectors = np.linalg.eigh(0.5 * (standard + standard.T))
        modes = np.linalg.solve(chol_m.T, vectors[:, :count])
        values = values[:count]
        # The standard form resolves an eigenvalue only to roundoff of the largest one. When a retained
        # eigenvalue disagrees with the Rayleigh quotient of its vector, solve the inverse problem
        # L_k^-1 M L_k^-T w = (1 / lambda) w instead, which resolves the lowest eigenvalues to roundoff
        # of themselves.
        rayleigh = modal_sparse.quadratic_form_diag(modes, stiffness_ii) / np.maximum(
            modal_sparse.quadratic_form_diag(modes, mass_ii), np.finfo(np.float64).tiny
        )
        if np.any(np.abs(values - rayleigh) > self._DENSE_EIGENVALUE_AGREEMENT * np.abs(rayleigh)):
            inverse = np.linalg.solve(chol_k, np.linalg.solve(chol_k, mass_ii).T).T
            flexibility, vectors = np.linalg.eigh(0.5 * (inverse + inverse.T))
            flexibility, vectors = flexibility[::-1][:count], vectors[:, ::-1][:, :count]
            if not np.all(flexibility > 0.0):
                raise ValueError("mass_matrix must be positive definite on the interior DOFs")
            values = 1.0 / flexibility
            modes = np.linalg.solve(chol_k.T, vectors) * np.sqrt(values)

        stiffness_s = modal_sparse.scale_symmetric(stiffness_ii, scaling)
        stiffness_s = 0.5 * (stiffness_s + stiffness_s.T)
        size = stiffness_ii.shape[0]
        tolerance = self._energy_ratio_tolerance(stiffness_ii)
        softest_value, softest_vector = math.inf, np.zeros(0)
        if size > 0:
            try:
                from scipy import linalg as dense_linalg

                softest_values, softest_vectors = dense_linalg.eigh(stiffness_s, subset_by_index=(0, 0))
            except ImportError:
                # NumPy has no subset driver; the dense path must work without SciPy.
                softest_values, softest_vectors = np.linalg.eigh(stiffness_s)
            softest_value, softest_vector = float(softest_values[0]), softest_vectors[:, 0]
        # The softest eigenvalue of K_s bounds the ratio of every motion from below, so it decides; the
        # ratios of the retained modes phi = D y are reported, from y^T K_s y = phi^T K_ii phi.
        equilibrated = modes / scaling[:, None]
        threshold = self._zero_energy_check(
            scaling,
            modal_sparse.matrix_one_norm(stiffness_s),
            tolerance,
            softest_value,
            softest_vector,
            values,
            modal_sparse.quadratic_form_diag(modes, stiffness_ii),
            np.einsum("di,di->i", equilibrated, equilibrated),
        )
        return constraint_modes, values, modes, pivot_ratio, threshold

    def _interior_sparse(self, stiffness_ii, stiffness_ib, mass_ii, count: int):
        """Solve the constraint modes and fixed-interface modes with SuperLU and ARPACK.

        Works on the equilibrated matrices ``K_s = D K_ii D`` and
        ``M_s = D M_ii D`` with ``D = diag(K_ii)^-1/2`` and maps the constraint
        modes and eigenvectors back with ``D``; the eigenvalues are those of
        the original pencil.
        """
        sigma = 0.0 if self.arpack_shift is None else self.arpack_shift
        scaling = self._interior_scaling(stiffness_ii)
        self._check_interior_mass(mass_ii)
        stiffness_s = modal_sparse.scale_symmetric(stiffness_ii, scaling)
        mass_s = modal_sparse.scale_symmetric(mass_ii, scaling)
        try:
            factorization = modal_sparse.ShiftedFactorization(stiffness_s, mass_s, sigma)
        except ValueError as exc:
            raise ValueError(self._mechanism_message()) from exc
        if sigma <= 0.0 and factorization.negative_pivot_count != 0:
            pivots = factorization.negative_pivot_count
            detail = (
                f"the factorization of the equilibrated interior stiffness has {pivots} non-positive pivots"
                if pivots > 0
                else "the factorization of the equilibrated interior stiffness pivoted off the diagonal"
            )
            raise ValueError(self._mechanism_message(detail=detail))
        if sigma == 0.0:
            stiffness_factorization = factorization
        else:
            if factorization.numerically_singular:
                raise ValueError(
                    f"The interior stiffness_matrix - sigma * mass_matrix is numerically singular for sigma = "
                    f"{sigma:.6g} (smallest pivot {factorization.pivot_ratio:.3g} of the largest): sigma lies at "
                    "a fixed-interface eigenvalue. Choose a different arpack_shift"
                )
            try:
                stiffness_factorization = modal_sparse.ShiftedFactorization(stiffness_s, mass_s, 0.0)
            except ValueError as exc:
                raise ValueError(self._mechanism_message()) from exc
            if stiffness_factorization.negative_pivot_count != 0:
                raise ValueError(self._mechanism_message())
        # SuperLU raises only for exactly zero pivots; a roundoff-sized pivot also reveals a mechanism.
        # The pivots of the equilibrated matrix lie between its smallest eigenvalue and one, so their
        # ratio does not depend on the units of the DOFs.
        if stiffness_factorization.numerically_singular:
            raise ValueError(
                self._mechanism_message(
                    detail=f"the smallest pivot of the equilibrated interior stiffness is "
                    f"{stiffness_factorization.pivot_ratio:.3g} of the largest, at or below the limit "
                    f"{modal_sparse.SINGULAR_PIVOT_RATIO:g} of the factorization, which eigenvalue_tolerance "
                    "does not change"
                )
            )
        pivot_ratio = stiffness_factorization.pivot_ratio

        def interior_solve(rhs: np.ndarray) -> np.ndarray:
            """Return ``K_ii^-1 rhs`` for the columns of ``rhs`` through the equilibrated factor."""
            result = np.empty_like(rhs)
            for start in range(0, rhs.shape[1], self._SOLVE_CHUNK_COLUMNS):
                stop = min(start + self._SOLVE_CHUNK_COLUMNS, rhs.shape[1])
                scaled_rhs = np.ascontiguousarray(scaling[:, None] * rhs[:, start:stop])
                result[:, start:stop] = scaling[:, None] * stiffness_factorization.solve(scaled_rhs)
            return result

        self._interior_solve = interior_solve
        rhs = stiffness_ib.toarray() if modal_sparse.is_sparse(stiffness_ib) else np.asarray(stiffness_ib)
        constraint_modes = -interior_solve(rhs)

        # A mechanism is a direction without strain energy. Its energy ratio in the equilibrated metric,
        # y^T K_s y / (||K_s||_1 y^T y), is at roundoff level whatever the mass matrix is, while a threshold
        # relative to the largest eigenvalue of (K, M) rises above genuine elastic modes when small rotary
        # inertia or a stiff, light part lifts the top of the spectrum. The softest direction of K_s covers
        # motions with almost no mass, which lie far above the computed modes.
        stiffness_norm = modal_sparse.matrix_one_norm(stiffness_s)
        tolerance = self._energy_ratio_tolerance(stiffness_ii)
        softest_value, softest_vector = modal_sparse.smallest_eigenvalue_estimate(
            stiffness_s, stiffness_factorization.solve
        )
        empty = np.zeros(0)
        if count == 0 or softest_value <= tolerance * stiffness_norm:
            threshold = self._zero_energy_check(
                scaling, stiffness_norm, tolerance, softest_value, softest_vector, empty, empty, empty
            )
            return constraint_modes, np.zeros(0), np.zeros((stiffness_ii.shape[0], 0)), pivot_ratio, threshold
        try:
            values, vectors = modal_sparse.lowest_generalized_eigenpairs(
                stiffness_s,
                mass_s,
                count,
                sigma=sigma,
                zero_threshold=0.0,
                margin=self._ARPACK_MARGIN,
                tol=self.arpack_tol,
                maxiter=self.arpack_maxiter,
                ncv=self.arpack_ncv,
                factorization=factorization,
                max_restarts=self._ARPACK_RESTARTS,
            )
        except ValueError as exc:
            if sigma != 0.0 or "inaccurate eigenpair" not in str(exc):
                raise
            # At the default shift the factor is that of the stiffness alone: an eigenpair that it cannot
            # resolve lies too far below the others, as for a part that is almost free.
            raise ValueError(
                self._mechanism_message(detail=f"the eigensolver cannot resolve the lowest modes: {exc}")
            ) from exc
        threshold = self._zero_energy_check(
            scaling,
            stiffness_norm,
            tolerance,
            softest_value,
            softest_vector,
            values,
            modal_sparse.quadratic_form_diag(vectors, stiffness_s),
            np.einsum("di,di->i", vectors, vectors),
        )
        return constraint_modes, values[:count], scaling[:, None] * vectors[:, :count], pivot_ratio, threshold

    @staticmethod
    def _mechanism_message(
        lowest: float | None = None, threshold: float | None = None, detail: str | None = None
    ) -> str:
        message = (
            "The interior stiffness matrix (every interface frame fixed) is singular, not positive definite, or "
            "ill-conditioned beyond the tolerance: the model has a zero-energy mechanism, for example a part "
            "that is not tied to the rest or an unrestrained DOF without stiffness, or stiffness contrasts "
            "that double precision does not resolve. Check constraints, interfaces, and the stiffness matrix; "
            "eigenvalue_tolerance sets the smallest accepted energy ratio"
        )
        if lowest is not None and threshold is not None:

            def hertz(value: float) -> float:
                return math.sqrt(max(value, 0.0)) / (2.0 * math.pi)

            message += (
                f" (lowest fixed-interface eigenvalue {lowest:.6g} 1/s^2 = {hertz(lowest):.4g} Hz is at or below "
                f"the zero-energy threshold {threshold:.6g} 1/s^2 = {hertz(threshold):.4g} Hz"
            )
            message += f"; {detail})" if detail else ")"
        elif detail:
            message += f" ({detail})"
        return message

    def _static_correction_vectors(
        self, transform, stiffness_ii, mass_ii, modes: np.ndarray, solve: Any = None
    ) -> tuple[np.ndarray, np.ndarray, list[int], list[float], float]:
        """Return the static correction vectors of ``static_correction_loads`` on the interior coordinates.

        The fixed-interface response ``K_ii^-1 f_i`` of each load is orthogonalized twice in the mass
        inner product against ``modes`` and the vectors already retained, dropped when its remaining
        mass norm is below :attr:`_STATIC_CORRECTION_SPAN_TOLERANCE` of the original one, normalized to
        unit modal mass, and oriented like the modes. ``solve`` returns ``K_ii^-1 b`` for the columns of
        ``b``; without it ``K_ii`` is factored here.

        Returns:
            The vectors, shape ``[interior_count, static_correction_count]``, their Rayleigh quotients
            [1/s^2], the load index of each vector, the relative remaining norm of every load, and the
            largest deviation of the vectors from mass-orthonormality against the modes and each other.

        Warns:
            UserWarning: If a load has no part on the interior coordinates (it is zero or acts on
                interface DOFs only) and so adds no vector.
        """
        interior_count = int(stiffness_ii.shape[0])
        vectors: list[np.ndarray] = []
        mass_vectors: list[np.ndarray] = []
        indices: list[int] = []
        relative_norms: list[float] = []
        loads = self.static_correction_loads
        if loads is not None and loads.shape[1] > 0 and interior_count > 0:
            interior_loads = np.ascontiguousarray(np.asarray(transform.T @ loads)[:interior_count])
            if solve is None:
                # A subclass that overrides the interior solve, as wrappers written for earlier versions
                # do, does not provide the factorization of K_ii: factor it here.
                if modal_sparse.is_sparse(stiffness_ii):
                    scaling = self._interior_scaling(stiffness_ii)
                    factor = modal_sparse.ShiftedFactorization(
                        modal_sparse.scale_symmetric(stiffness_ii, scaling),
                        modal_sparse.scale_symmetric(mass_ii, scaling),
                        0.0,
                    )

                    def solve(rhs: np.ndarray) -> np.ndarray:
                        return scaling[:, None] * factor.solve(np.ascontiguousarray(scaling[:, None] * rhs))

                else:

                    def solve(rhs: np.ndarray) -> np.ndarray:
                        return np.linalg.solve(stiffness_ii, rhs)

            responses = np.asarray(solve(interior_loads), dtype=np.float64)
            mass_modes = modal_sparse.matmul(mass_ii, modes)
            for column in range(responses.shape[1]):
                vector = np.array(responses[:, column], dtype=np.float64)
                norm_before = math.sqrt(max(modal_sparse.quadratic_form(vector, mass_ii), 0.0))
                norm_after = 0.0
                mass_vector = vector
                if norm_before > 0.0 and math.isfinite(norm_before):
                    for _ in range(2):
                        vector -= modes @ (mass_modes.T @ vector)
                        for kept, mass_kept in zip(vectors, mass_vectors, strict=True):
                            vector -= kept * float(mass_kept @ vector)
                    mass_vector = modal_sparse.matmul(mass_ii, vector)
                    norm_after = math.sqrt(max(float(vector @ mass_vector), 0.0))
                relative = norm_after / norm_before if norm_before > 0.0 and math.isfinite(norm_before) else 0.0
                relative_norms.append(relative)
                if norm_before == 0.0:
                    warnings.warn(
                        f"static_correction_loads column {column} has no load on the interior coordinates (it "
                        "is zero or acts on interface DOFs only) and adds no correction vector",
                        stacklevel=3,
                    )
                if not relative >= self._STATIC_CORRECTION_SPAN_TOLERANCE:
                    continue
                sign = -1.0 if vector[int(np.argmax(np.abs(vector)))] < 0.0 else 1.0
                vectors.append(vector * (sign / norm_after))
                mass_vectors.append(mass_vector * (sign / norm_after))
                indices.append(column)
        elif loads is not None:
            relative_norms = [0.0] * loads.shape[1]
        orthogonality = 0.0
        if vectors:
            stacked = np.column_stack(vectors)
            mass_stacked = np.column_stack(mass_vectors)
            gram = stacked.T @ mass_stacked - np.eye(stacked.shape[1])
            orthogonality = max(float(np.max(np.abs(gram))), float(np.max(np.abs(modes.T @ mass_stacked), initial=0.0)))
        else:
            stacked = np.zeros((interior_count, 0), dtype=np.float64)
        values = modal_sparse.quadratic_form_diag(stacked, stiffness_ii)
        return stacked, values, indices, relative_norms, orthogonality

    def _lumped_sample_mass(self) -> np.ndarray:
        """Return the row-sum lumped translational mass [kg] of each sample node."""
        dofs = self.dofs_per_node
        translation = np.zeros((self.dof_count, 3), dtype=np.float64)
        for axis in range(3):
            translation[axis::dofs, axis] = 1.0
        row_sums = modal_sparse.matmul(self.mass_matrix, translation)
        nodal = np.stack([row_sums[axis::dofs, axis] for axis in range(3)], axis=1).mean(axis=1)
        return nodal[self.sample_node_indices]

    def reduce(self) -> ModalGeneratorCraigBampton:
        """Reduce the model and return the :class:`ModalGeneratorCraigBampton` of its reduced matrices.

        Returns:
            The generator of the reduced model; :meth:`build` builds its basis.

        Raises:
            ValueError: If the constraints or interfaces are inconsistent, the
                interior has a mechanism, the reduced mass matrix is not
                positive definite, the reduced stiffness matrix does not
                preserve the six rigid-body modes, for example because a
                constraint grounds the body, or the reduced model fails an
                acceptance check of :meth:`ModalGeneratorCraigBampton.build`.
        """
        timings: dict[str, float] = {}
        clock = time.perf_counter()
        sparse = self._uses_sparse()
        mass, stiffness, damping = self.mass_matrix, self.stiffness_matrix, self.damping_matrix
        if sparse:
            mass, stiffness = modal_sparse.to_csc(mass), modal_sparse.to_csc(stiffness)
            damping = None if damping is None else modal_sparse.to_csc(damping)
        else:
            mass, stiffness = modal_sparse.to_dense(mass), modal_sparse.to_dense(stiffness)
            damping = None if damping is None else modal_sparse.to_dense(damping)

        transform, mpc_report, interface_report = self._coordinate_transformation(sparse)
        frame_count = 6 * self.interface_count
        coordinate_count = int(transform.shape[1])
        interior_count = coordinate_count - frame_count
        if self.fixed_mode_count > interior_count:
            raise ValueError(
                f"fixed_mode_count {self.fixed_mode_count} exceeds the {interior_count} interior coordinates"
            )
        timings["elimination"] = time.perf_counter() - clock

        clock = time.perf_counter()

        def congruence(matrix):
            product = transform.T @ (matrix @ transform)
            if sparse:
                return modal_sparse.to_csc(0.5 * (product + product.T))
            product = np.asarray(product)
            return 0.5 * (product + product.T)

        stiffness_z = congruence(stiffness)
        mass_z = congruence(mass)
        damping_z = None if damping is None else congruence(damping)
        interior = np.arange(interior_count)
        boundary = np.arange(interior_count, coordinate_count)
        stiffness_ii = modal_sparse.submatrix(stiffness_z, interior, interior)
        stiffness_ib = modal_sparse.submatrix(stiffness_z, interior, boundary)
        mass_ii = modal_sparse.submatrix(mass_z, interior, interior)
        timings["projection"] = time.perf_counter() - clock

        clock = time.perf_counter()
        tie_ratio, tie_coordinate, uncoupled = self._weakest_tie(stiffness_z, interior_count)
        if uncoupled.shape[0] > 0:
            names = ", ".join(self._describe_interior(index) for index in uncoupled[:3])
            warnings.warn(
                f"{uncoupled.shape[0]} interior coordinates have a stiffness diagonal but no coupling to any "
                f"other coordinate ({names}{', ...' if uncoupled.shape[0] > 3 else ''}): each moves on its own "
                "stiffness, independent of the rest of the model",
                stacklevel=2,
            )
        tie_tolerance = self._energy_ratio_tolerance(stiffness_ii)
        if tie_ratio <= tie_tolerance:
            raise ValueError(
                self._mechanism_message(
                    detail=f"{self._describe_interior(tie_coordinate)} is tied to the rest of the model with "
                    f"{tie_ratio:.3g} of the stiffness of the coordinates it is tied to, at or below the "
                    f"tolerance {tie_tolerance:.3g}"
                )
            )
        solve = self._interior_sparse if sparse else self._interior_dense
        self._zero_energy = {}
        self._interior_solve = None
        constraint_modes, eigenvalues, modes, pivot_ratio, threshold = solve(
            stiffness_ii, stiffness_ib, mass_ii, self.fixed_mode_count
        )
        # The factorization of K_ii serves the correction vectors below and is not kept on the generator.
        interior_solve, self._interior_solve = self._interior_solve, None
        modal_mass = modal_sparse.quadratic_form_diag(modes, mass_ii)
        if not np.all(modal_mass > 0.0):
            raise ValueError("mass_matrix must be positive definite on the interior DOFs")
        modes = modes / np.sqrt(modal_mass)
        for mode in range(modes.shape[1]):
            pivot = int(np.argmax(np.abs(modes[:, mode])))
            if modes[pivot, mode] < 0.0:
                modes[:, mode] *= -1.0
        (
            corrections,
            correction_values,
            correction_indices,
            correction_norms,
            correction_orthogonality,
        ) = self._static_correction_vectors(transform, stiffness_ii, mass_ii, modes, interior_solve)
        del interior_solve
        timings["interior_solve"] = time.perf_counter() - clock

        clock = time.perf_counter()
        mode_stop = frame_count + modes.shape[1]
        reduced_count = mode_stop + corrections.shape[1]
        basis = np.zeros((coordinate_count, reduced_count), dtype=np.float64)
        basis[:interior_count, :frame_count] = constraint_modes
        basis[:interior_count, frame_count:mode_stop] = modes
        basis[:interior_count, mode_stop:] = corrections
        basis[interior_count:, :frame_count] = np.eye(frame_count)

        def reduce_matrix(matrix):
            reduced = basis.T @ modal_sparse.matmul(matrix, basis)
            return 0.5 * (reduced + reduced.T)

        reduced_mass = reduce_matrix(mass_z)
        reduced_stiffness = reduce_matrix(stiffness_z)
        reduced_damping = None if damping_z is None else reduce_matrix(damping_z)
        try:
            np.linalg.cholesky(reduced_mass)
        except np.linalg.LinAlgError as exc:
            raise ValueError("The reduced mass matrix is not positive definite") from exc

        stiffness_ib_dense = modal_sparse.to_dense(stiffness_ib)
        constraint_residual_matrix = modal_sparse.matmul(stiffness_ii, constraint_modes) + stiffness_ib_dense
        constraint_mode_residual = float(np.linalg.norm(constraint_residual_matrix)) / max(
            float(np.linalg.norm(stiffness_ib_dense)), np.finfo(np.float64).tiny
        )
        stiffness_modes = modal_sparse.matmul(stiffness_ii, modes)
        eigen_residual = 0.0
        if modes.shape[1] > 0:
            residual = stiffness_modes - modal_sparse.matmul(mass_ii, modes) * eigenvalues
            eigen_residual = float(
                np.max(np.linalg.norm(residual, axis=0) / np.maximum(np.linalg.norm(stiffness_modes, axis=0), 1e-300))
            )
        if eigen_residual > self._EIGEN_RESIDUAL_LIMIT:
            raise ValueError(
                self._mechanism_message(
                    detail=f"a returned fixed-interface mode is not an eigenvector: its relative residual "
                    f"||K phi - lambda M phi|| / ||K phi|| is {eigen_residual:.3g}, above the limit "
                    f"{self._EIGEN_RESIDUAL_LIMIT:g}; the eigensolver cannot resolve a mode this far below the "
                    "others"
                )
            )
        if eigen_residual > self._EIGEN_RESIDUAL_WARNING:
            warnings.warn(
                f"The fixed-interface eigenpairs have a relative residual of {eigen_residual:.3g}; the lowest "
                "modes are poorly resolved against the stiffest ones",
                stacklevel=2,
            )
        rigid = np.zeros((reduced_count, 6), dtype=np.float64)
        for index, position in enumerate(self.interface_positions):
            rigid[6 * index : 6 * index + 3, :3] = np.eye(3)
            rigid[6 * index : 6 * index + 3, 3:] = -_skew(position)
            rigid[6 * index + 3 : 6 * index + 6, 3:] = np.eye(3)
        rigid_residual = float(np.linalg.norm(reduced_stiffness @ rigid, ord=np.inf)) / (
            max(float(np.linalg.norm(reduced_stiffness, ord=np.inf)), 1.0)
            * max(float(np.linalg.norm(rigid, ord=np.inf)), 1.0)
        )
        if rigid_residual > self._RIGID_RESIDUAL_TOLERANCE:
            # Reject here rather than in build(), so that save_npz() never exports such a model; tell
            # constraints that resist rigid motion apart from a stiffness matrix that does. Constraints that
            # ground the body are rejected even when rigid_consistency="project".
            message = (
                "The reduced stiffness matrix does not preserve the six rigid-body modes (relative residual "
                f"{rigid_residual:.3g})"
            )
            motion_residual = transformation_rigid_residual(transform, self.node_positions, self.dofs_per_node)
            if motion_residual > self._RIGID_RESIDUAL_TOLERANCE:
                raise ValueError(
                    f"{message}: the constraints do not admit rigid-body motion (relative residual "
                    f"{motion_residual:.3g}). A constraint grounds the body, such as an SPC (u = 0), or ties nodes at "
                    "different positions without the lever arm of a rigid link; a Craig-Bampton model must stay "
                    "free-free, with supports applied through its interfaces"
                )
            if self.rigid_consistency == "check":
                raise ValueError(
                    f"{message}: check that stiffness_matrix is the free-free stiffness of the model; a small "
                    "residual of a non-conformal mesh can be removed with rigid_consistency='project'"
                )

        dofs = self.dofs_per_node
        translation_rows = (dofs * self.sample_node_indices[:, None] + np.arange(3)).reshape(-1)
        transform_rows = transform.tocsr() if sparse else transform
        recovery = np.asarray(transform_rows[translation_rows] @ basis)
        rotation_recovery = None
        if dofs == 6:
            rotation_rows = translation_rows + 3
            rotation_recovery = np.asarray(transform_rows[rotation_rows] @ basis)
        self._rigid_projection = None
        projection_report = None
        if self.rigid_consistency == "project":
            (
                reduced_stiffness,
                reduced_mass,
                reduced_damping,
                recovery,
                rotation_recovery,
                projection_report,
            ) = self._project_rigid_consistency(
                rigid, reduced_stiffness, reduced_mass, reduced_damping, recovery, rotation_recovery
            )
        # The test with which ModalGeneratorCraigBampton.build() separates the rigid-body modes: an elastic
        # mode at or below rigid_mode_tolerance of the largest eigenvalue of the reduced model cannot be
        # told from them, so such a model is refused here instead of being returned or exported.
        elastic_ratio = None
        if reduced_count > 6:
            factor = np.linalg.cholesky(reduced_mass)
            standard = np.linalg.solve(factor, np.linalg.solve(factor, reduced_stiffness).T).T
            reduced_values = np.linalg.eigvalsh(0.5 * (standard + standard.T))
            reduced_scale = max(float(np.max(np.abs(reduced_values))), 1.0)
            elastic_ratio = float(reduced_values[6]) / reduced_scale
            rigid_mode_tolerance = float(self._craig_bampton_options["rigid_mode_tolerance"])
            if elastic_ratio <= rigid_mode_tolerance:
                raise ValueError(
                    self._mechanism_message(
                        detail=f"the softest elastic mode of the reduced model, at "
                        f"{math.sqrt(max(float(reduced_values[6]), 0.0)) / (2.0 * math.pi):.4g} Hz, has "
                        f"{elastic_ratio:.3g} of its largest eigenvalue, at or below rigid_mode_tolerance "
                        f"{rigid_mode_tolerance:.3g}: it cannot be separated from the six rigid-body modes"
                    )
                )
        if tie_ratio <= self._SOFT_TIE_RATIO:
            message = (
                f"{self._describe_interior(tie_coordinate)} is tied to the rest of the model with only "
                f"{tie_ratio:.3g} of the stiffness of the coordinates it is tied to: a soft part"
            )
            if eigenvalues.shape[0] > 0:
                lowest = math.sqrt(max(float(eigenvalues[0]), 0.0)) / (2.0 * math.pi)
                highest = math.sqrt(max(float(eigenvalues[-1]), 0.0)) / (2.0 * math.pi)
                message += (
                    f". The lowest fixed-interface mode lies at {lowest:.4g} Hz, a factor "
                    f"{highest / max(lowest, np.finfo(np.float64).tiny):.3g} below the highest retained one at "
                    f"{highest:.4g} Hz"
                )
            warnings.warn(f"{message}. Check the tie if the part is meant to be attached", stacklevel=2)
        timings["reduction"] = time.perf_counter() - clock

        self.transformation = transform
        self.interior_count = interior_count
        self.constraint_modes = constraint_modes
        self.fixed_interface_modes = modes
        self.fixed_interface_eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        self.fixed_interface_frequencies = np.sqrt(np.maximum(self.fixed_interface_eigenvalues, 0.0)) / (2.0 * math.pi)
        self.static_correction_vectors = corrections
        self.static_correction_count = int(corrections.shape[1])
        self.static_correction_frequencies = np.sqrt(np.maximum(correction_values, 0.0)) / (2.0 * math.pi)
        self.static_correction_load_indices = np.asarray(correction_indices, dtype=np.int64)
        self.reduction_basis = basis
        self.reduced_mass_matrix = reduced_mass
        self.reduced_stiffness_matrix = reduced_stiffness
        self.reduced_damping_matrix = reduced_damping
        self.recovery_matrix = recovery
        self.rotation_recovery_matrix = rotation_recovery

        sample_mass = self._lumped_sample_mass() if isinstance(self.sample_mass, str) else self.sample_mass
        self.craig_bampton = ModalGeneratorCraigBampton(
            self.interface_positions,
            reduced_mass,
            reduced_stiffness,
            self.sample_points,
            recovery,
            damping_matrix=reduced_damping,
            interface_names=self.interface_names,
            label=self.label,
            sample_mass=sample_mass,
            rigid_mode_tolerance=self._craig_bampton_options["rigid_mode_tolerance"],
            damping_coupling_tolerance=self._craig_bampton_options["damping_coupling_tolerance"],
            damping_projection=self._craig_bampton_options["damping_projection"],
            interface_match_tolerance=self._craig_bampton_options["interface_match_tolerance"],
            interface_recovery_tolerance=self._craig_bampton_options["interface_recovery_tolerance"],
            rigid_recovery_tolerance=self._craig_bampton_options["rigid_recovery_tolerance"],
        )
        self.report = {
            "dof_count": self.dof_count,
            "node_count": self.node_count,
            "interface_count": self.interface_count,
            "coordinate_count": coordinate_count,
            "interior_count": interior_count,
            "reduced_dof_count": reduced_count,
            "eigensolver": "arpack" if sparse else "dense",
            "constraint_elimination": mpc_report,
            "interface_elimination": interface_report,
            "fixed_interface_eigenvalues": self.fixed_interface_eigenvalues.tolist(),
            "fixed_interface_frequencies_hz": self.fixed_interface_frequencies.tolist(),
            "static_correction_load_count": len(correction_norms),
            "static_correction_count": self.static_correction_count,
            "static_correction_load_indices": list(correction_indices),
            "static_correction_dropped": sorted(set(range(len(correction_norms))) - set(correction_indices)),
            "static_correction_relative_norms": list(correction_norms),
            "static_correction_frequencies_hz": self.static_correction_frequencies.tolist(),
            "static_correction_orthogonality": correction_orthogonality,
            "zero_energy_threshold": float(threshold),
            # Replaced by the interior solve; a subclass that overrides it leaves these defaults.
            "zero_energy_scale": math.nan,
            "zero_energy_scale_source": "",
            "constraint_mode_residual": constraint_mode_residual,
            "eigen_residual": eigen_residual,
            "rigid_stiffness_residual": rigid_residual,
            "rigid_consistency": self.rigid_consistency,
            "rigid_projection": projection_report,
            "interior_stiffness_pivot_ratio": pivot_ratio,
            "zero_energy_tie_ratio_min": tie_ratio if math.isfinite(tie_ratio) else None,
            "zero_energy_tie_coordinate": self._describe_interior(tie_coordinate) if tie_coordinate >= 0 else None,
            "uncoupled_interior_count": int(uncoupled.shape[0]),
            "reduced_elastic_eigenvalue_ratio": elastic_ratio,
            "reduced_elastic_eigenvalue_limit": float(self._craig_bampton_options["rigid_mode_tolerance"]),
            "reduced_mass_condition": float(np.linalg.cond(reduced_mass)),
            "timings_s": timings,
        }
        self.report.update(self._zero_energy)
        # The remaining acceptance checks of build(), on the same matrices and with the same code, so that a
        # model that could not be built is refused here. The report above stays available for diagnosis.
        try:
            self.craig_bampton._check_buildable()
        except ValueError as exc:
            raise ValueError(f"The reduced model cannot be built: {exc}") from exc
        return self.craig_bampton

    @staticmethod
    def _rigid_residuals(
        rigid, stiffness, mass, damping, recovery, sample_rigid
    ) -> tuple[dict[str, float], np.ndarray]:
        """Return the relative rigid-body residuals of a reduced model and the physical target of ``R^T M R``.

        The stiffness and damping residuals are those of :meth:`ModalGeneratorCraigBampton.build`, the recovery
        residual is its recovery error relative to the sample rigid motion, and the mass residual is the largest
        deviation of ``R^T M R`` from an isotropic translational mass and a skew-symmetric mass moment, relative
        to its largest entry.
        """
        rigid_scale = max(float(np.linalg.norm(rigid, ord=np.inf)), 1.0)

        def stiffness_like(matrix):
            scale = max(float(np.linalg.norm(matrix, ord=np.inf)), 1.0)
            return float(np.linalg.norm(matrix @ rigid, ord=np.inf)) / (scale * rigid_scale)

        spatial = rigid.T @ mass @ rigid
        target = spatial.copy()
        target[:3, :3] = float(np.trace(spatial[:3, :3]) / 3.0) * np.eye(3)
        moment = 0.5 * (spatial[:3, 3:] - spatial[:3, 3:].T)
        target[:3, 3:] = moment
        target[3:, :3] = moment.T
        residuals = {
            "stiffness": stiffness_like(stiffness),
            "damping": 0.0 if damping is None else stiffness_like(damping),
            "recovery": float(np.max(np.abs(recovery @ rigid - sample_rigid)))
            / max(float(np.max(np.abs(sample_rigid))), 1.0),
            "mass": float(np.max(np.abs(spatial - target))) / max(float(np.max(np.abs(spatial))), 1.0),
        }
        return residuals, target

    def _project_rigid_consistency(self, rigid, stiffness, mass, damping, recovery, rotation_recovery):
        """Project a small rigid-body residual out of the reduced model (``rigid_consistency="project"``).

        Returns:
            The projected stiffness, mass, damping, recovery, and rotation recovery, and the projection report.

        Raises:
            ValueError: If a residual exceeds ``rigid_projection_limit``, the projected mass matrix is not
                positive definite, or an elastic eigenvalue moves by more than
                ``rigid_projection_eigenvalue_tolerance``.
        """
        sample_rigid = modal_sparse.rigid_vectors(self.sample_points, 3, np.zeros(3))
        before, target = self._rigid_residuals(rigid, stiffness, mass, damping, recovery, sample_rigid)
        worst = max(before, key=before.get)
        if before[worst] > self.rigid_projection_limit:
            raise ValueError(
                f"The reduced model resists rigid-body motion beyond rigid_projection_limit "
                f"{self.rigid_projection_limit:.3g} (relative {worst} residual {before[worst]:.3g}); this is not "
                "discretization drift that rigid_consistency='project' may remove. Check that stiffness_matrix is "
                "the free-free stiffness of the model, without supports or grounding springs"
            )

        rigid_pinv = np.linalg.pinv(rigid)
        projector = np.eye(rigid.shape[0]) - rigid @ rigid_pinv

        def project(matrix):
            projected = projector.T @ matrix @ projector
            return 0.5 * (projected + projected.T)

        new_stiffness = project(stiffness)
        new_damping = None if damping is None else project(damping)
        new_recovery = recovery + (sample_rigid - recovery @ rigid) @ rigid_pinv
        new_rotation_recovery = None
        if rotation_recovery is not None:
            rotation_rigid = np.zeros_like(sample_rigid)
            rotation_rigid[:, 3:] = np.tile(np.eye(3), (self.sample_points.shape[0], 1))
            new_rotation_recovery = rotation_recovery + (rotation_rigid - rotation_recovery @ rigid) @ rigid_pinv
        new_mass = mass + rigid_pinv.T @ (target - rigid.T @ mass @ rigid) @ rigid_pinv
        new_mass = 0.5 * (new_mass + new_mass.T)

        def eigenvalues(matrix_k, matrix_m):
            factor = np.linalg.cholesky(matrix_m)
            standard = np.linalg.solve(factor, np.linalg.solve(factor, matrix_k).T).T
            return np.linalg.eigvalsh(0.5 * (standard + standard.T))

        try:
            projected_values = eigenvalues(new_stiffness, new_mass)
        except np.linalg.LinAlgError as exc:
            raise ValueError("The rigid-consistency projection made the reduced mass matrix indefinite") from exc
        original_values = eigenvalues(stiffness, mass)
        rigid_count = rigid.shape[1]
        shift = np.abs(projected_values[rigid_count:] - original_values[rigid_count:]) / np.maximum(
            np.abs(projected_values[rigid_count:]), np.finfo(np.float64).tiny
        )
        largest_shift = float(np.max(shift, initial=0.0))
        if largest_shift > self.rigid_projection_eigenvalue_tolerance:
            mode = int(np.argmax(shift))
            raise ValueError(
                f"The rigid-consistency projection moves elastic eigenvalue {mode} of the reduced model from "
                f"{original_values[rigid_count + mode]:.6g} to {projected_values[rigid_count + mode]:.6g} 1/s^2 "
                f"(relative {largest_shift:.3g} > rigid_projection_eigenvalue_tolerance "
                f"{self.rigid_projection_eigenvalue_tolerance:.3g}); the rigid-body residual is not separated from "
                "the elastic modes, so it is not removed"
            )

        after, _ = self._rigid_residuals(rigid, new_stiffness, new_mass, new_damping, new_recovery, sample_rigid)
        self._rigid_projection = (rigid, rigid_pinv)
        report = {f"{name}_residual_before": value for name, value in before.items()}
        report.update({f"{name}_residual_after": value for name, value in after.items()})
        report["max_elastic_eigenvalue_shift"] = largest_shift
        return new_stiffness, new_mass, new_damping, new_recovery, new_rotation_recovery, report

    def build(self) -> ModalBasis:
        """Reduce the model and build the mass-normalized elastic modal basis.

        Returns:
            The basis with ``6 * interface_count + fixed_mode_count +
            static_correction_count - 6`` elastic modes.
        """
        return self.reduce().build()

    def _require_reduced(self) -> None:
        if self.craig_bampton is None:
            raise RuntimeError("Call reduce() or build() first")

    def full_recovery_matrix(self) -> np.ndarray:
        """Return the recovery of every nodal DOF from the reduced coordinates, shape ``[dof_count, reduced_dof_count]``.

        The matrix is dense; for large models prefer ``recovery_matrix`` and
        ``rotation_recovery_matrix``, which cover the sample nodes only. With
        ``rigid_consistency="project"`` it carries the same rigid-motion
        correction as those matrices.
        """
        self._require_reduced()
        recovery = np.asarray(self.transformation @ self.reduction_basis)
        if self._rigid_projection is not None:
            rigid, rigid_pinv = self._rigid_projection
            full_rigid = modal_sparse.rigid_vectors(self.node_positions, self.dofs_per_node, np.zeros(3))
            recovery = recovery + (full_rigid - recovery @ rigid) @ rigid_pinv
        return recovery

    def to_arrays(self, *, full_recovery: bool = True) -> dict[str, np.ndarray]:
        """Return the reduced model as arrays named like a Craig-Bampton import file.

        The keys ``interface_positions``, ``interface_names``, ``mass_matrix``,
        ``stiffness_matrix``, ``sample_points``, and ``recovery_matrix`` are the
        :class:`ModalGeneratorCraigBampton` arguments. ``damping_matrix`` is
        present with a damping matrix, ``full_recovery`` with
        ``full_recovery``, ``interfaces`` (one node index per interface) when
        every interface has a single node, and ``fixed_interface_frequencies_hz``,
        ``sample_node_indices``, ``interface_node_indices``, and
        ``interface_node_offsets`` (CSR-style offsets into the node indices)
        always. With ``static_correction_loads``, ``static_correction_frequencies_hz``
        and ``static_correction_load_indices`` describe the coordinates that
        follow the fixed-interface modes.

        Args:
            full_recovery: Whether to include the dense full recovery matrix.
        """
        self._require_reduced()
        arrays = {
            "interface_positions": np.array(self.interface_positions, dtype=np.float64),
            "interface_names": np.array(self.interface_names),
            "mass_matrix": self.reduced_mass_matrix,
            "stiffness_matrix": self.reduced_stiffness_matrix,
            "sample_points": np.array(self.sample_points, dtype=np.float64),
            "recovery_matrix": self.recovery_matrix,
            "fixed_interface_frequencies_hz": self.fixed_interface_frequencies,
            "sample_node_indices": self.sample_node_indices,
            "interface_node_indices": np.concatenate([interface.nodes for interface in self.interfaces]),
            "interface_node_offsets": np.cumsum([0] + [interface.nodes.shape[0] for interface in self.interfaces]),
        }
        if self.static_correction_loads is not None:
            arrays["static_correction_frequencies_hz"] = self.static_correction_frequencies
            arrays["static_correction_load_indices"] = self.static_correction_load_indices
        if self.reduced_damping_matrix is not None:
            arrays["damping_matrix"] = self.reduced_damping_matrix
        if full_recovery:
            arrays["full_recovery"] = self.full_recovery_matrix()
        if all(interface.nodes.shape[0] == 1 for interface in self.interfaces):
            arrays["interfaces"] = np.array([int(interface.nodes[0]) for interface in self.interfaces], dtype=np.int64)
        return arrays

    def save_npz(self, path: str | Path, *, full_recovery: bool = True, **extra_arrays: np.ndarray) -> None:
        """Write :meth:`to_arrays` and ``extra_arrays``, such as mesh triangles, to a compressed ``.npz`` file.

        Args:
            path: Output file path.
            full_recovery: Whether to include the dense full recovery matrix.
            extra_arrays: Additional named arrays to store.
        """
        arrays = self.to_arrays(full_recovery=full_recovery)
        arrays.update(extra_arrays)
        np.savez_compressed(path, **arrays)
