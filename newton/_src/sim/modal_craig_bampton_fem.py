# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-interface Craig-Bampton reduction of full finite-element models."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

import numpy as np

from . import modal_sparse
from .modal import ModalBasis, ModalGeneratorCraigBampton
from .modal_constraints import assemble_triplets, constraint_transformation

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

    After :meth:`reduce` (called by :meth:`build`), the generator exposes
    ``transformation`` (``u = T_z z`` for the coordinates ``z`` after
    constraint and interface elimination, interior coordinates first and the
    ``6 * interface_count`` frame coordinates last), ``constraint_modes``,
    ``fixed_interface_modes``, ``fixed_interface_eigenvalues`` [1/s^2],
    ``fixed_interface_frequencies`` [Hz], ``reduction_basis`` (the
    Craig-Bampton matrix ``T`` in ``z`` coordinates), the reduced matrices,
    ``recovery_matrix`` and, for six-DOF nodes, ``rotation_recovery_matrix``
    of the sample nodes, :meth:`full_recovery_matrix`, ``craig_bampton`` (the
    :class:`ModalGeneratorCraigBampton`), and ``report`` with sizes,
    residuals, conditioning, and stage timings.

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
        eigenvalue_tolerance: Fixed-interface eigenvalue threshold, relative
            to an estimate of the largest eigenvalue, at or below which the
            interior is reported as a mechanism. Values below the eigensolver
            accuracy are raised to it, as in :class:`ModalGeneratorFEM`.
        rigid_mode_tolerance: Passed to :class:`ModalGeneratorCraigBampton`.
        damping_coupling_tolerance: Passed to :class:`ModalGeneratorCraigBampton`.
        damping_projection: Passed to :class:`ModalGeneratorCraigBampton`.
        interface_match_tolerance: Passed to :class:`ModalGeneratorCraigBampton`.
        interface_recovery_tolerance: Passed to :class:`ModalGeneratorCraigBampton`.
    """

    _ARPACK_MARGIN: ClassVar[int] = 2
    """Extra eigenpairs computed beyond the retained ones so the last retained mode is converged."""

    _SOLVE_CHUNK_COLUMNS: ClassVar[int] = 64
    """Number of constraint-mode right-hand sides solved at once."""

    def __init__(
        self,
        node_positions: Sequence[Sequence[float]] | np.ndarray,
        mass_matrix: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray,
        stiffness_matrix: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray,
        interfaces: Sequence[Mapping[str, Any]],
        *,
        fixed_mode_count: int,
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
        }

        size = self.dof_count
        self.mass_matrix = modal_sparse.coerce_symmetric_matrix(mass_matrix, size, "mass_matrix")
        self.stiffness_matrix = modal_sparse.coerce_symmetric_matrix(stiffness_matrix, size, "stiffness_matrix")
        self.damping_matrix = (
            None
            if damping_matrix is None
            else modal_sparse.coerce_symmetric_matrix(damping_matrix, size, "damping_matrix")
        )
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
            if coordinate >= coordinate_count:
                index, component = divmod(coordinate - coordinate_count, 6)
                return f"interface '{self.interfaces[index].name}' frame {_FRAME_COMPONENTS[component]}"
            if independent_dofs is not None:
                return self._describe_dof(int(independent_dofs[coordinate]))
            if base is not None:
                return f"transformation column {coordinate}"
            return self._describe_dof(coordinate)

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
        transform = result.transformation[:coordinate_count]
        if base is not None:
            transform = base @ transform
        if sparse:
            transform = modal_sparse.to_csc(transform)
        return transform, mpc_report, result.report

    def _interior_dense(self, stiffness_ii, stiffness_ib, mass_ii, count: int):
        """Solve the constraint modes and fixed-interface modes densely."""
        try:
            chol_k = np.linalg.cholesky(stiffness_ii)
        except np.linalg.LinAlgError as exc:
            raise ValueError(self._mechanism_message()) from exc
        constraint_modes = -np.linalg.solve(chol_k.T, np.linalg.solve(chol_k, stiffness_ib))
        diagonal = np.abs(np.diag(chol_k)) ** 2
        pivot_ratio = float(np.min(diagonal) / np.max(diagonal)) if diagonal.shape[0] else 1.0
        try:
            chol_m = np.linalg.cholesky(mass_ii)
        except np.linalg.LinAlgError as exc:
            raise ValueError("mass_matrix must be positive definite on the interior DOFs") from exc
        standard = np.linalg.solve(chol_m, np.linalg.solve(chol_m, stiffness_ii).T).T
        values, vectors = np.linalg.eigh(0.5 * (standard + standard.T))
        relative = max(self.eigenvalue_tolerance, 8.0 * stiffness_ii.shape[0] * np.finfo(np.float64).eps)
        threshold = relative * float(np.max(np.abs(values), initial=0.0))
        if values.shape[0] > 0 and values[0] <= threshold:
            raise ValueError(self._mechanism_message())
        modes = np.linalg.solve(chol_m.T, vectors[:, :count])
        return constraint_modes, values[:count], modes, pivot_ratio, threshold

    def _interior_sparse(self, stiffness_ii, stiffness_ib, mass_ii, count: int):
        """Solve the constraint modes and fixed-interface modes with SuperLU and ARPACK."""
        sigma = 0.0 if self.arpack_shift is None else self.arpack_shift
        try:
            factorization = modal_sparse.ShiftedFactorization(stiffness_ii, mass_ii, sigma)
        except ValueError as exc:
            raise ValueError(self._mechanism_message()) from exc
        if sigma <= 0.0 and factorization.negative_pivot_count != 0:
            raise ValueError(self._mechanism_message())
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
                stiffness_factorization = modal_sparse.ShiftedFactorization(stiffness_ii, mass_ii, 0.0)
            except ValueError as exc:
                raise ValueError(self._mechanism_message()) from exc
            if stiffness_factorization.negative_pivot_count != 0:
                raise ValueError(self._mechanism_message())
        # SuperLU raises only for exactly zero pivots; a roundoff-sized pivot also reveals a mechanism.
        if stiffness_factorization.numerically_singular:
            raise ValueError(self._mechanism_message())
        pivot_ratio = stiffness_factorization.pivot_ratio

        rhs = stiffness_ib.toarray() if modal_sparse.is_sparse(stiffness_ib) else np.asarray(stiffness_ib)
        constraint_modes = np.empty_like(rhs)
        for start in range(0, rhs.shape[1], self._SOLVE_CHUNK_COLUMNS):
            stop = min(start + self._SOLVE_CHUNK_COLUMNS, rhs.shape[1])
            constraint_modes[:, start:stop] = -stiffness_factorization.solve(np.ascontiguousarray(rhs[:, start:stop]))

        # The bound overestimates the largest eigenvalue of consistent-mass six-DOF shells by up to about
        # 40 times (see generalized_eigenvalue_bound), which raises the mechanism threshold accordingly.
        bound = modal_sparse.generalized_eigenvalue_bound(stiffness_ii, mass_ii)
        row_nonzeros = max(modal_sparse.max_row_nonzeros(stiffness_ii), 1)
        relative = max(self.eigenvalue_tolerance, 8.0 * row_nonzeros * np.finfo(np.float64).eps)
        threshold = relative * bound
        if count == 0:
            return constraint_modes, np.zeros(0), np.zeros((stiffness_ii.shape[0], 0)), pivot_ratio, threshold
        values, modes = modal_sparse.lowest_generalized_eigenpairs(
            stiffness_ii,
            mass_ii,
            count,
            sigma=sigma,
            zero_threshold=threshold,
            margin=self._ARPACK_MARGIN,
            tol=self.arpack_tol,
            maxiter=self.arpack_maxiter,
            ncv=self.arpack_ncv,
            factorization=factorization,
        )
        if values[0] <= threshold:
            raise ValueError(self._mechanism_message())
        return constraint_modes, values[:count], modes[:, :count], pivot_ratio, threshold

    @staticmethod
    def _mechanism_message() -> str:
        return (
            "The interior stiffness matrix (every interface frame fixed) is singular or not positive definite: "
            "the model has a zero-energy mechanism, for example a part that is not tied to the rest or an "
            "unrestrained DOF without stiffness. Check constraints, interfaces, and the stiffness matrix"
        )

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
                interior has a mechanism, or the reduced mass matrix is not
                positive definite.
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
        solve = self._interior_sparse if sparse else self._interior_dense
        constraint_modes, eigenvalues, modes, pivot_ratio, threshold = solve(
            stiffness_ii, stiffness_ib, mass_ii, self.fixed_mode_count
        )
        modal_mass = modal_sparse.quadratic_form_diag(modes, mass_ii)
        if not np.all(modal_mass > 0.0):
            raise ValueError("mass_matrix must be positive definite on the interior DOFs")
        modes = modes / np.sqrt(modal_mass)
        for mode in range(modes.shape[1]):
            pivot = int(np.argmax(np.abs(modes[:, mode])))
            if modes[pivot, mode] < 0.0:
                modes[:, mode] *= -1.0
        timings["interior_solve"] = time.perf_counter() - clock

        clock = time.perf_counter()
        reduced_count = frame_count + modes.shape[1]
        basis = np.zeros((coordinate_count, reduced_count), dtype=np.float64)
        basis[:interior_count, :frame_count] = constraint_modes
        basis[:interior_count, frame_count:] = modes
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
        rigid = np.zeros((reduced_count, 6), dtype=np.float64)
        for index, position in enumerate(self.interface_positions):
            rigid[6 * index : 6 * index + 3, :3] = np.eye(3)
            rigid[6 * index : 6 * index + 3, 3:] = -_skew(position)
            rigid[6 * index + 3 : 6 * index + 6, 3:] = np.eye(3)
        rigid_residual = float(np.linalg.norm(reduced_stiffness @ rigid, ord=np.inf)) / (
            max(float(np.linalg.norm(reduced_stiffness, ord=np.inf)), 1.0)
            * max(float(np.linalg.norm(rigid, ord=np.inf)), 1.0)
        )

        dofs = self.dofs_per_node
        translation_rows = (dofs * self.sample_node_indices[:, None] + np.arange(3)).reshape(-1)
        transform_rows = transform.tocsr() if sparse else transform
        recovery = np.asarray(transform_rows[translation_rows] @ basis)
        rotation_recovery = None
        if dofs == 6:
            rotation_rows = translation_rows + 3
            rotation_recovery = np.asarray(transform_rows[rotation_rows] @ basis)
        timings["reduction"] = time.perf_counter() - clock

        self.transformation = transform
        self.interior_count = interior_count
        self.constraint_modes = constraint_modes
        self.fixed_interface_modes = modes
        self.fixed_interface_eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        self.fixed_interface_frequencies = np.sqrt(np.maximum(self.fixed_interface_eigenvalues, 0.0)) / (2.0 * math.pi)
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
            "zero_energy_threshold": float(threshold),
            "constraint_mode_residual": constraint_mode_residual,
            "eigen_residual": eigen_residual,
            "rigid_stiffness_residual": rigid_residual,
            "interior_stiffness_pivot_ratio": pivot_ratio,
            "reduced_mass_condition": float(np.linalg.cond(reduced_mass)),
            "timings_s": timings,
        }
        return self.craig_bampton

    def build(self) -> ModalBasis:
        """Reduce the model and build the mass-normalized elastic modal basis.

        Returns:
            The basis with ``6 * interface_count + fixed_mode_count - 6`` elastic modes.
        """
        return self.reduce().build()

    def _require_reduced(self) -> None:
        if self.craig_bampton is None:
            raise RuntimeError("Call reduce() or build() first")

    def full_recovery_matrix(self) -> np.ndarray:
        """Return the recovery of every nodal DOF from the reduced coordinates, shape ``[dof_count, reduced_dof_count]``.

        The matrix is dense; for large models prefer ``recovery_matrix`` and
        ``rotation_recovery_matrix``, which cover the sample nodes only.
        """
        self._require_reduced()
        return np.asarray(self.transformation @ self.reduction_basis)

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
        always.

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
