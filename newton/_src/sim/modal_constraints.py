# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Elimination of homogeneous linear multi-point constraints from finite-element matrices.

The constraints ``C u = 0`` are removed with a coordinate transformation
``u = T q`` whose columns span the null space of ``C``. Every column of ``T``
belongs to an original DOF that stays an independent coordinate; the rows of
the dependent DOFs express them in terms of the independent ones. Dense inputs
are processed with NumPy only; sparse inputs keep ``T`` sparse and need SciPy.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from . import modal_sparse

if TYPE_CHECKING:
    from scipy.sparse import sparray, spmatrix

_SOLVE_CHUNK_COLUMNS = 256
"""Number of right-hand sides solved at once when forming ``-C_s^-1 C_m``."""

_TRANSFORMATION_RANK_TOLERANCE = 1.0e-10
"""Squared sine of the angle to the other columns below which a sparse ``T`` column counts as dependent."""

_GRAM_REGULARIZATION = 1.0e-13
"""Diagonal shift of the normalized Gram matrix ``T^T T`` that keeps exactly dependent columns factorable."""


class _Transformation(NamedTuple):
    """Result of :func:`constraint_transformation`."""

    transformation: Any
    """Transformation ``T`` [dimensionless], shape ``[dof_count, independent_count]``; CSC or dense."""
    independent_dofs: np.ndarray
    """Original DOF index of each column of ``T``, ascending."""
    dependent_dofs: np.ndarray
    """Original DOF indices eliminated by the constraints, ascending."""
    report: dict[str, Any]
    """Counts and residuals of the elimination."""


class _UnionFind:
    """Disjoint sets over ``0..size-1`` whose representative is the smallest member."""

    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(self, first: int, second: int) -> bool:
        """Join the sets of ``first`` and ``second``; return ``False`` if they were already joined."""
        a = self.find(first)
        b = self.find(second)
        if a == b:
            return False
        if a < b:
            self.parent[b] = a
        else:
            self.parent[a] = b
        return True

    def labels(self) -> np.ndarray:
        """Return the representative of every item."""
        parent = self.parent
        while True:
            grand = parent[parent]
            if np.array_equal(grand, parent):
                return parent.copy()
            parent = grand


def _canonical_entries(constraints: Any, dof_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return the row-major, duplicate-free nonzero entries ``(rows, cols, values, row_count)`` of ``constraints``."""
    if modal_sparse.is_sparse(constraints):
        coo = constraints.tocoo()
        shape = tuple(int(v) for v in coo.shape)
        rows = np.asarray(coo.row, dtype=np.int64)
        cols = np.asarray(coo.col, dtype=np.int64)
        values = np.asarray(coo.data, dtype=np.float64)
    else:
        array = np.asarray(constraints, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"constraints must be a 2D matrix, got shape {array.shape}")
        shape = array.shape
        rows, cols = np.nonzero(array)
        rows = rows.astype(np.int64)
        cols = cols.astype(np.int64)
        values = array[rows, cols]
    if shape[1] != dof_count:
        raise ValueError(f"constraints must have {dof_count} columns, one per DOF, got shape {shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("constraints contains non-finite entries")
    order = np.lexsort((cols, rows))
    rows, cols, values = rows[order], cols[order], values[order]
    if rows.shape[0] > 0:
        first = np.ones(rows.shape[0], dtype=bool)
        first[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
        starts = np.flatnonzero(first)
        values = np.add.reduceat(values, starts)
        rows, cols = rows[starts], cols[starts]
    keep = values != 0.0
    return rows[keep], cols[keep], values[keep], int(shape[0])


def _row_starts(rows: np.ndarray, row_count: int) -> np.ndarray:
    """Return CSR-style offsets of row-sorted entries."""
    return np.searchsorted(rows, np.arange(row_count + 1))


def _exact_duplicate_count(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, row_count: int) -> int:
    """Return the number of non-empty rows that repeat an earlier row exactly."""
    starts = _row_starts(rows, row_count)
    seen: set[bytes] = set()
    duplicates = 0
    for row in range(row_count):
        begin, end = starts[row], starts[row + 1]
        if begin == end:
            continue
        key = cols[begin:end].tobytes() + values[begin:end].tobytes()
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def assemble_triplets(
    rows: np.ndarray, cols: np.ndarray, values: np.ndarray, shape: tuple[int, int], sparse: bool
) -> Any:
    """Assemble triplets into a CSC matrix (``sparse``) or a dense array, summing duplicates."""
    if sparse:
        scipy_sparse = modal_sparse._import_scipy("sparse")
        return modal_sparse.to_csc(scipy_sparse.coo_matrix((values, (rows, cols)), shape=shape))
    dense = np.zeros(shape, dtype=np.float64)
    np.add.at(dense, (rows, cols), values)
    return dense


def _select_pivots(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    row_count: int,
    col_count: int,
    retained: np.ndarray,
    tolerance: float,
    row_ids: np.ndarray,
    describe_row: Callable[[int], str],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Choose one dependent column per independent constraint row by greedy pivoted elimination.

    Rows whose largest-magnitude non-retained entry lies in a column that no
    other remaining row uses are pivoted on that column directly; this peels
    rigid links (RBE2-style rows) without any elimination. The remaining rows
    are split into blocks connected through non-retained columns, and each
    block is eliminated densely row by row, pivoting on the largest-magnitude
    remaining non-retained column.

    Args:
        rows: Row index of each nonzero entry, sorted.
        cols: Column index of each entry.
        values: Value of each entry.
        row_count: Number of rows.
        col_count: Number of columns.
        retained: Mask of columns that must stay independent, shape ``[col_count]``.
        tolerance: Relative magnitude below which an eliminated row is zero.
        row_ids: Original row index of each row, for error messages.
        describe_row: Description of an original row index, for error messages.

    Returns:
        Kept row indices, their pivot columns (aligned), and the number of
        dependent rows that were dropped.

    Raises:
        ValueError: If a row can only be satisfied by making a retained column dependent.
    """
    starts = _row_starts(rows, row_count)
    row_scale = np.zeros(row_count, dtype=np.float64)
    if rows.shape[0] > 0:
        np.maximum.at(row_scale, rows, np.abs(values))
    free_entry = ~retained[cols]

    def inconsistent(row: int) -> ValueError:
        return ValueError(
            f"{describe_row(int(row_ids[row]))} cannot be satisfied without making a retained DOF dependent; "
            "the constraints relate retained DOFs to each other (for example two interface frames tied together, "
            "a node attached to two interfaces, or an interface DOF that a constraint fixes or eliminates)"
        )

    pivot_rows: list[np.ndarray] = []
    pivot_cols: list[np.ndarray] = []
    active = np.zeros(row_count, dtype=bool)
    active[rows] = True
    # Rows without entries, for example ties already implied by other ties, are dependent.
    dropped = row_count - int(np.count_nonzero(active))
    # A non-empty row without free entries constrains retained columns only.
    has_free = np.zeros(row_count, dtype=bool)
    has_free[rows[free_entry]] = True
    bad = np.flatnonzero(active & ~has_free)
    if bad.shape[0] > 0:
        raise inconsistent(int(bad[0]))

    while True:
        entry_active = active[rows] & free_entry
        candidate_rows = rows[entry_active]
        if candidate_rows.shape[0] == 0:
            break
        candidate_cols = cols[entry_active]
        magnitude = np.abs(values[entry_active])
        uses = np.bincount(candidate_cols, minlength=col_count)
        # Largest free entry per row: sort by row, then by decreasing magnitude.
        order = np.lexsort((-magnitude, candidate_rows))
        first = np.ones(order.shape[0], dtype=bool)
        first[1:] = candidate_rows[order[1:]] != candidate_rows[order[:-1]]
        best = order[first]
        # A negligible pivot would make the row's other DOFs explode; the block path reports it.
        peel = (uses[candidate_cols[best]] == 1) & (magnitude[best] > tolerance * row_scale[candidate_rows[best]])
        if not np.any(peel):
            break
        peeled_rows = candidate_rows[best[peel]]
        pivot_rows.append(peeled_rows)
        pivot_cols.append(candidate_cols[best[peel]])
        active[peeled_rows] = False

    remaining = np.flatnonzero(active)
    if remaining.shape[0] > 0:
        components = _UnionFind(col_count)
        first_free = np.full(row_count, -1, dtype=np.int64)
        for row in remaining:
            begin, end = starts[row], starts[row + 1]
            free_cols = cols[begin:end][free_entry[begin:end]]
            first_free[row] = free_cols[0]
            for col in free_cols[1:]:
                components.union(int(free_cols[0]), int(col))
        labels = components.labels()
        row_label = labels[first_free[remaining]]
        order = np.argsort(row_label, kind="stable")
        boundaries = np.flatnonzero(np.diff(row_label[order])) + 1
        for block in np.split(remaining[order], boundaries):
            block_entries = np.concatenate([np.arange(starts[row], starts[row + 1]) for row in block])
            block_cols, local_cols = np.unique(cols[block_entries], return_inverse=True)
            local_rows = np.repeat(np.arange(block.shape[0]), starts[block + 1] - starts[block])
            work = np.zeros((block.shape[0], block_cols.shape[0]), dtype=np.float64)
            work[local_rows, local_cols] = values[block_entries]
            free = ~retained[block_cols]
            for local, row in enumerate(block):
                residual = work[local]
                threshold = tolerance * row_scale[row]
                free_residual = np.where(free, np.abs(residual), 0.0)
                pivot = int(np.argmax(free_residual))
                if free_residual[pivot] <= threshold:
                    if float(np.max(np.abs(residual))) <= threshold:
                        dropped += 1
                        continue
                    raise inconsistent(int(row))
                pivot_rows.append(np.array([row], dtype=np.int64))
                pivot_cols.append(np.array([block_cols[pivot]], dtype=np.int64))
                free[pivot] = False
                below = work[local + 1 :]
                if below.shape[0] > 0:
                    below -= np.outer(below[:, pivot] / residual[pivot], residual)
                    below[:, pivot] = 0.0
    if not pivot_rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), dropped
    return np.concatenate(pivot_rows), np.concatenate(pivot_cols), dropped


def constraint_transformation(
    constraints: Any,
    dof_count: int,
    *,
    retained_dofs: Sequence[int] | np.ndarray | None = None,
    tolerance: float = 1.0e-10,
    sparse: bool | None = None,
    dof_label: Callable[[int], str] | None = None,
    row_label: Callable[[int], str] | None = None,
) -> _Transformation:
    """Return a transformation ``u = T q`` whose columns span the null space of ``constraints``.

    Rows with exactly two nonzeros of equal magnitude and opposite sign tie two
    DOFs together. They are grouped by connected components (union-find), so
    chains such as ``a = b = c`` and redundant cycles are handled, and each
    group keeps one independent DOF: a retained DOF if the group has one,
    otherwise the smallest index. This part of ``T`` is Boolean. All other
    rows are expressed in the tie-reduced coordinates; one dependent DOF per
    independent row is chosen by greedy pivoted elimination (see
    :func:`_select_pivots`), and ``T`` receives the rows ``-C_s^-1 C_m`` from an
    LU factorization of the square constraint block ``C_s`` of the dependent
    DOFs. Linearly dependent rows, including exact duplicates, are dropped
    with a warning.

    Args:
        constraints: Constraint matrix ``C`` with ``C u = 0``, shape
            ``[constraint_count, dof_count]``; dense or SciPy sparse.
        dof_count: Number of DOFs (columns of ``C``).
        retained_dofs: DOFs that must stay independent coordinates.
        tolerance: Relative magnitude below which an eliminated row counts as
            linearly dependent; also the relative tolerance of the tie test.
        sparse: Whether ``T`` is assembled as a CSC matrix. ``None`` follows
            the type of ``constraints``.
        dof_label: Description of a column (DOF) index for error messages.
            ``None`` uses ``"DOF <index>"``.
        row_label: Description of a constraint row index for error messages.
            ``None`` uses ``"Constraint row <index>"``.

    Returns:
        The transformation, the independent and dependent DOF indices, and a report.

    Raises:
        ValueError: If the constraints are malformed or would make a retained DOF dependent.
    """
    dof_count = int(dof_count)
    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if sparse is None:
        sparse = modal_sparse.is_sparse(constraints)
    rows, cols, values, row_count = _canonical_entries(constraints, dof_count)

    retained = np.zeros(dof_count, dtype=bool)
    if retained_dofs is not None:
        retained_index = np.asarray(retained_dofs, dtype=np.int64).reshape(-1)
        if np.any(retained_index < 0) or np.any(retained_index >= dof_count):
            raise ValueError("retained_dofs contains out-of-range DOF indices")
        retained[retained_index] = True

    starts = _row_starts(rows, row_count)
    row_nonzeros = np.diff(starts)
    empty_rows = int(np.count_nonzero(row_nonzeros == 0))
    duplicate_rows = _exact_duplicate_count(rows, cols, values, row_count)

    # Tie rows: two entries a, b with a + b == 0 (relative to the tolerance).
    two = np.flatnonzero(row_nonzeros == 2)
    first_value = values[starts[two]]
    second_value = values[starts[two] + 1]
    is_tie = np.abs(first_value + second_value) <= tolerance * np.maximum(np.abs(first_value), np.abs(second_value))
    tie_rows = two[is_tie]
    tie_mask = np.zeros(row_count, dtype=bool)
    tie_mask[tie_rows] = True

    groups = _UnionFind(dof_count)
    redundant_ties = 0
    for a, b in zip(cols[starts[tie_rows]].tolist(), cols[starts[tie_rows] + 1].tolist(), strict=True):
        if not groups.union(a, b):
            redundant_ties += 1
    root = groups.labels()
    retained_per_group = np.bincount(root[retained], minlength=dof_count)
    conflict = np.flatnonzero(retained_per_group > 1)
    if conflict.shape[0] > 0:
        members = np.flatnonzero((root == conflict[0]) & retained)
        links = np.flatnonzero((root == conflict[0]) & ~retained)[:4]
        if dof_label is None:
            described = str(members.tolist())
            through = f" through DOFs {links.tolist()}" if links.shape[0] else ""
        else:
            described = ", ".join(dof_label(int(member)) for member in members)
            through = f" through {', '.join(dof_label(int(link)) for link in links)}" if links.shape[0] else ""
        raise ValueError(
            f"Tie constraints connect the retained DOFs {described}{through}; "
            "retained DOFs, such as interface frame DOFs, cannot be tied to each other"
        )
    group_lead = np.arange(dof_count, dtype=np.int64)
    retained_index = np.flatnonzero(retained)
    group_lead[root[retained_index]] = retained_index
    roots = np.unique(root)
    leads = group_lead[roots]
    order = np.argsort(leads, kind="stable")
    tie_leads = leads[order]
    column_of_root = np.empty(dof_count, dtype=np.int64)
    column_of_root[roots[order]] = np.arange(roots.shape[0])
    tie_column = column_of_root[root]
    tie_count = int(roots.shape[0])
    retained_columns = np.zeros(tie_count, dtype=bool)
    retained_columns[tie_column[retained]] = True

    # General rows in tie-reduced coordinates.
    general = ~tie_mask[rows]
    general_rows = rows[general]
    general_cols = tie_column[cols[general]]
    general_values = values[general]
    general_ids = np.unique(general_rows)
    compact = np.searchsorted(general_ids, general_rows)
    ordering = np.lexsort((general_cols, compact))
    compact, general_cols, general_values = compact[ordering], general_cols[ordering], general_values[ordering]
    if compact.shape[0] > 0:
        first = np.ones(compact.shape[0], dtype=bool)
        first[1:] = (compact[1:] != compact[:-1]) | (general_cols[1:] != general_cols[:-1])
        segment = np.flatnonzero(first)
        scale = np.zeros(general_ids.shape[0], dtype=np.float64)
        np.maximum.at(scale, compact, np.abs(general_values))
        general_values = np.add.reduceat(general_values, segment)
        compact, general_cols = compact[segment], general_cols[segment]
        # Entries cancelled by ties (for example u_a - u_b on two tied DOFs) vanish.
        keep = np.abs(general_values) > tolerance * scale[compact]
        compact, general_cols, general_values = compact[keep], general_cols[keep], general_values[keep]
    general_count = int(general_ids.shape[0])

    pivot_rows, pivot_cols, dependent_general = _select_pivots(
        compact,
        general_cols,
        general_values,
        general_count,
        tie_count,
        retained_columns,
        tolerance,
        general_ids,
        row_label if row_label is not None else (lambda row: f"Constraint row {row}"),
    )
    dependent_rows = redundant_ties + dependent_general
    if dependent_rows > 0:
        warnings.warn(
            f"Dropped {dependent_rows} linearly dependent constraint rows ({duplicate_rows} exact duplicates)",
            stacklevel=3,
        )

    is_pivot = np.zeros(tie_count, dtype=bool)
    is_pivot[pivot_cols] = True
    kept = np.flatnonzero(~is_pivot)
    kept_position = np.full(tie_count, -1, dtype=np.int64)
    kept_position[kept] = np.arange(kept.shape[0])
    independent_count = int(kept.shape[0])

    # T2 in tie-reduced coordinates: identity rows for independent columns, -C_s^-1 C_m for pivots.
    t_rows = [kept]
    t_cols = [np.arange(independent_count)]
    t_values = [np.ones(independent_count)]
    pivot_count = int(pivot_rows.shape[0])
    if pivot_count > 0:
        row_position = np.full(general_count, -1, dtype=np.int64)
        row_position[pivot_rows] = np.arange(pivot_count)
        pivot_position = np.full(tie_count, -1, dtype=np.int64)
        pivot_position[pivot_cols] = np.arange(pivot_count)
        in_system = row_position[compact] >= 0
        system_rows = row_position[compact[in_system]]
        system_cols = general_cols[in_system]
        system_values = general_values[in_system]
        on_pivot = is_pivot[system_cols]
        follower_rows, follower_cols, follower_values = _solve_dependent(
            (system_rows[on_pivot], pivot_position[system_cols[on_pivot]], system_values[on_pivot]),
            (system_rows[~on_pivot], kept_position[system_cols[~on_pivot]], system_values[~on_pivot]),
            pivot_count,
            independent_count,
            sparse,
        )
        t_rows.append(pivot_cols[follower_rows])
        t_cols.append(follower_cols)
        t_values.append(follower_values)
    reduced_rows = np.concatenate(t_rows)
    reduced_cols = np.concatenate(t_cols)
    reduced_values = np.concatenate(t_values)
    # T = T1 T2 with Boolean T1: every DOF copies the row of its tie column.
    by_row = np.argsort(reduced_rows, kind="stable")
    reduced_rows, reduced_cols, reduced_values = reduced_rows[by_row], reduced_cols[by_row], reduced_values[by_row]
    reduced_starts = _row_starts(reduced_rows, tie_count)
    counts = np.diff(reduced_starts)[tie_column]
    entry = np.repeat(reduced_starts[tie_column], counts) + (
        np.arange(int(np.sum(counts))) - np.repeat(np.cumsum(counts) - counts, counts)
    )
    transformation = assemble_triplets(
        np.repeat(np.arange(dof_count), counts),
        reduced_cols[entry],
        reduced_values[entry],
        (dof_count, independent_count),
        sparse,
    )
    independent_dofs = tie_leads[kept]
    dependent_mask = np.ones(dof_count, dtype=bool)
    dependent_mask[independent_dofs] = False

    constraint_matrix = assemble_triplets(rows, cols, values, (row_count, dof_count), sparse)
    residual_matrix = constraint_matrix @ transformation
    constraint_scale = float(np.max(np.abs(values), initial=0.0))
    if sparse:
        residual = float(np.max(np.abs(residual_matrix.data), initial=0.0))
    else:
        residual = float(np.max(np.abs(residual_matrix), initial=0.0))
    report = {
        "constraint_rows": row_count,
        "empty_rows": empty_rows,
        "tie_rows": int(tie_rows.shape[0]),
        "general_rows": general_count,
        "dependent_rows": dependent_rows,
        "duplicate_rows": duplicate_rows,
        "independent_dofs": independent_count,
        "dependent_dofs": dof_count - independent_count,
        "constraint_residual": residual / constraint_scale if constraint_scale > 0.0 else 0.0,
    }
    return _Transformation(transformation, independent_dofs, np.flatnonzero(dependent_mask), report)


def check_transformation_rank(transformation: Any) -> None:
    """Raise if a transformation ``T`` (``u = T q``) has linearly dependent columns.

    Dependent columns make ``T^T K T`` and ``T^T M T`` singular, which would
    otherwise surface as a mechanism or an indefinite mass. A dense ``T`` is
    checked with its singular values. For a sparse ``T`` the Gram matrix
    ``T^T T`` with unit diagonal is factored symmetrically; each pivot is the
    squared sine of the angle between a column and the columns eliminated
    before it, and pivots at or below
    :data:`_TRANSFORMATION_RANK_TOLERANCE` count as dependent columns.

    Args:
        transformation: Transformation [dimensionless], shape
            ``[dof_count, independent_count]``; dense or CSC.

    Raises:
        ValueError: If the rank of ``T`` is below its column count.
    """
    column_count = int(transformation.shape[1])
    if column_count == 0:
        return
    if modal_sparse.is_sparse(transformation):
        scipy_sparse = modal_sparse._import_scipy("sparse")
        gram = modal_sparse.to_csc(transformation.T @ transformation)
        diagonal = np.asarray(gram.diagonal(), dtype=np.float64)
        scale = np.zeros(column_count, dtype=np.float64)
        np.divide(1.0, np.sqrt(diagonal), out=scale, where=diagonal > 0.0)
        normalized = scipy_sparse.diags(scale) @ gram @ scipy_sparse.diags(scale)
        normalized = modal_sparse.to_csc(normalized + _GRAM_REGULARIZATION * scipy_sparse.identity(column_count))
        pivots = np.abs(np.asarray(modal_sparse._symmetric_splu(normalized).U.diagonal(), dtype=np.float64))
        rank = column_count - int(np.count_nonzero(pivots <= _TRANSFORMATION_RANK_TOLERANCE))
    else:
        rank = int(np.linalg.matrix_rank(np.asarray(transformation, dtype=np.float64)))
    if rank < column_count:
        raise ValueError(
            f"transformation has linearly dependent columns: its rank is {rank}, below its {column_count} "
            "columns (independent_count). Each column must be an independent coordinate; remove the dependent "
            "columns, such as duplicates"
        )


def transformation_rigid_residual(
    transformation: Any,
    node_positions: np.ndarray,
    dofs_per_node: int,
    independent_dofs: np.ndarray | None = None,
) -> float:
    """Return how well ``T`` reproduces rigid-body motion, as a relative maximum error.

    With known ``independent_dofs`` the coordinates of a rigid motion ``R``
    are ``R`` restricted to them; otherwise they are the least-squares fit
    ``(T^T T)^-1 T^T R``. See :meth:`ConstraintElimination.rigid_motion_residual`.

    Args:
        transformation: Transformation ``T`` [dimensionless], shape
            ``[dof_count, coordinate_count]``; dense or CSC.
        node_positions: Node positions [m], shape ``[dof_count / dofs_per_node, 3]``.
        dofs_per_node: Number of DOFs per node, 3 or 6, node-major.
        independent_dofs: Original DOF of each column of ``T``, if each
            column is a unit row of ``T``.

    Returns:
        ``max |T R_q - R| / max |R|`` [dimensionless].
    """
    positions = np.asarray(node_positions, dtype=np.float64)
    rigid = modal_sparse.rigid_vectors(positions, dofs_per_node, np.mean(positions, axis=0))
    if independent_dofs is not None:
        rigid_q = rigid[independent_dofs]
    elif modal_sparse.is_sparse(transformation):
        sparse_linalg = modal_sparse._import_scipy("sparse.linalg")
        gram = modal_sparse.to_csc(transformation.T @ transformation)
        rigid_q = sparse_linalg.splu(gram).solve(np.asarray(transformation.T @ rigid))
    else:
        rigid_q = np.linalg.lstsq(transformation, rigid, rcond=None)[0]
    error = np.asarray(transformation @ rigid_q) - rigid
    return float(np.max(np.abs(error))) / max(float(np.max(np.abs(rigid))), np.finfo(np.float64).tiny)


def _solve_dependent(
    pivot_block: tuple[np.ndarray, np.ndarray, np.ndarray],
    lead_block: tuple[np.ndarray, np.ndarray, np.ndarray],
    pivot_count: int,
    independent_count: int,
    sparse: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the nonzero triplets of ``X = -C_s^-1 C_m``.

    Args:
        pivot_block: Triplets of the square block ``C_s`` (rows, pivot positions, values).
        lead_block: Triplets of ``C_m`` (rows, independent positions, values).
        pivot_count: Order of ``C_s``.
        independent_count: Number of independent coordinates (columns of ``C_m``).
        sparse: Whether to factor ``C_s`` with SuperLU instead of dense LAPACK.

    Returns:
        Row (pivot position), column (independent position), and value arrays.
    """
    support = np.unique(lead_block[1])
    if support.shape[0] == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float64)
    support_position = np.searchsorted(support, lead_block[1])
    shape_s = (pivot_count, pivot_count)
    out_rows, out_cols, out_values = [], [], []
    if sparse:
        sparse_linalg = modal_sparse._import_scipy("sparse.linalg")
        factor = sparse_linalg.splu(assemble_triplets(*pivot_block, shape_s, True))
        lead = assemble_triplets(lead_block[0], support_position, lead_block[2], (pivot_count, support.shape[0]), True)

        def solve(rhs: np.ndarray) -> np.ndarray:
            return factor.solve(rhs)
    else:
        dense_s = assemble_triplets(*pivot_block, shape_s, False)
        lead = assemble_triplets(lead_block[0], support_position, lead_block[2], (pivot_count, support.shape[0]), False)

        def solve(rhs: np.ndarray) -> np.ndarray:
            return np.linalg.solve(dense_s, rhs)

    for start in range(0, support.shape[0], _SOLVE_CHUNK_COLUMNS):
        stop = min(start + _SOLVE_CHUNK_COLUMNS, support.shape[0])
        rhs = lead[:, start:stop]
        rhs = rhs.toarray() if sparse else np.asarray(rhs)
        block = -solve(np.ascontiguousarray(rhs, dtype=np.float64))
        if not np.all(np.isfinite(block)):
            raise ValueError("The dependent-DOF constraint block is singular; the constraints are ill-conditioned")
        local_rows, local_cols = np.nonzero(block)
        out_rows.append(local_rows)
        out_cols.append(support[start + local_cols])
        out_values.append(block[local_rows, local_cols])
    return np.concatenate(out_rows), np.concatenate(out_cols), np.concatenate(out_values)


class ConstraintElimination:
    """Eliminate homogeneous linear multi-point constraints from finite-element matrices.

    .. experimental::

    The constraints ``C u = 0``, such as node-to-node ties between meshes,
    rigid links, or interpolation (distributing) elements, are removed with a
    coordinate transformation ``u = T q``. Each column of ``T`` belongs to an
    original DOF that stays an independent coordinate (``independent_dofs``);
    its row in ``T`` is a unit row. The other DOFs (``dependent_dofs``) are
    expressed in terms of the independent ones, so ``C T = 0``. The
    transformed matrices are ``T^T M T``, ``T^T K T``, and ``T^T D T``; loads
    transform as ``T^T f`` and displacements are recovered as ``T q``.

    Two elimination paths are combined:

    - Rows with exactly two nonzeros of equal magnitude and opposite sign tie
      two DOFs (``u_a = u_b``). Tied DOFs are grouped with a union-find, which
      handles chains (``a = b = c``) and redundant cycles; each group keeps
      its smallest DOF index, or a ``retained_dofs`` entry, as the independent
      DOF. This part of ``T`` is Boolean, so ties keep the matrices as sparse
      as the mesh. A scaled pair such as ``u_a - 2 u_b = 0`` uses the general
      path.
    - All other rows are expressed in the tie-reduced coordinates. One
      dependent DOF per independent row is chosen by greedy pivoted
      elimination: a row whose largest-magnitude entry lies in a DOF used by
      no other row (as in rigid links) is pivoted there directly, and the
      remaining rows are eliminated densely in blocks connected through
      shared DOFs, pivoting on the largest-magnitude remaining entry. ``T``
      receives the rows ``-C_s^-1 C_m``, solved with an LU factorization of
      the square block ``C_s`` of the dependent DOFs (SuperLU for sparse
      input). A single block coupling very many rows is eliminated densely,
      so general constraints should be local, as finite-element MPCs are.

    Linearly dependent rows, including exact duplicates and redundant tie
    cycles, are dropped with a warning; ``report`` records their count. Rows
    that could only be satisfied by making a retained DOF dependent raise.

    Alternatively, pass ``transformation`` directly, for example a ``T`` that
    an external pre-processor produced; ``independent_dofs`` and
    ``dependent_dofs`` are then ``None``.

    Dense inputs are processed with NumPy only. When any input is a SciPy
    sparse matrix, ``T`` and the transformed matrices are CSC matrices.

    Args:
        mass_matrix: Mass matrix [kg], shape ``[dof_count, dof_count]``; dense
            or SciPy sparse.
        stiffness_matrix: Stiffness matrix [N/m], with the shape of
            ``mass_matrix``.
        constraints: Constraint matrix ``C`` with ``C u = 0``, shape
            ``[constraint_count, dof_count]``; dense or SciPy sparse.
            Mutually exclusive with ``transformation``.
        transformation: Transformation ``T`` with ``u = T q``, shape
            ``[dof_count, independent_count]``; dense or SciPy sparse.
        damping_matrix: Optional damping matrix [N s/m], with the shape of
            ``mass_matrix``.
        retained_dofs: DOF indices that must remain independent coordinates,
            such as interface DOFs.
        tolerance: Relative magnitude, measured against the largest entry of
            the original row, below which an eliminated row counts as linearly
            dependent. Also the relative tolerance of the tie test.

    Raises:
        ValueError: If shapes mismatch, both or neither of ``constraints`` and
            ``transformation`` are given, entries are not finite, the
            constraints relate retained DOFs to each other, or
            ``transformation`` has linearly dependent columns.
    """

    def __init__(
        self,
        mass_matrix: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray,
        stiffness_matrix: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray,
        constraints: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray | None = None,
        *,
        transformation: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray | None = None,
        damping_matrix: Sequence[Sequence[float]] | np.ndarray | spmatrix | sparray | None = None,
        retained_dofs: Sequence[int] | np.ndarray | None = None,
        tolerance: float = 1.0e-10,
    ):
        if (constraints is None) == (transformation is None):
            raise ValueError("Pass exactly one of constraints and transformation")
        size = int(np.shape(mass_matrix)[0])
        self.dof_count = size
        self.tolerance = float(tolerance)
        matrices = (mass_matrix, stiffness_matrix, damping_matrix, constraints, transformation)
        sparse = any(modal_sparse.is_sparse(matrix) for matrix in matrices)
        mass = modal_sparse.coerce_symmetric_matrix(mass_matrix, size, "mass_matrix")
        stiffness = modal_sparse.coerce_symmetric_matrix(stiffness_matrix, size, "stiffness_matrix")
        damping = (
            None
            if damping_matrix is None
            else modal_sparse.coerce_symmetric_matrix(damping_matrix, size, "damping_matrix")
        )
        if sparse:
            mass, stiffness = modal_sparse.to_csc(mass), modal_sparse.to_csc(stiffness)
            damping = None if damping is None else modal_sparse.to_csc(damping)

        self.constraints = None
        self.independent_dofs: np.ndarray | None = None
        self.dependent_dofs: np.ndarray | None = None
        if constraints is not None:
            result = constraint_transformation(
                constraints, size, retained_dofs=retained_dofs, tolerance=self.tolerance, sparse=sparse
            )
            transform = result.transformation
            self.independent_dofs = result.independent_dofs
            self.dependent_dofs = result.dependent_dofs
            self.report = dict(result.report)
            self.constraints = modal_sparse.to_csc(constraints) if sparse else np.asarray(constraints, np.float64)
        else:
            if retained_dofs is not None:
                raise ValueError("retained_dofs applies to constraints, not to a given transformation")
            transform = modal_sparse.to_csc(transformation) if sparse else np.asarray(transformation, np.float64)
            if transform.ndim != 2 or transform.shape[0] != size:
                raise ValueError(f"transformation must have shape [{size}, independent_count], got {transform.shape}")
            values = transform.data if sparse else transform
            if not np.all(np.isfinite(values)):
                raise ValueError("transformation contains non-finite entries")
            check_transformation_rank(transform)
            self.report = {"independent_dofs": int(transform.shape[1]), "dependent_dofs": None}

        self.transformation = transform
        self.reduced_mass_matrix = self._congruence(mass)
        self.reduced_stiffness_matrix = self._congruence(stiffness)
        self.reduced_damping_matrix = None if damping is None else self._congruence(damping)

    def _congruence(self, matrix: Any) -> Any:
        """Return the symmetric part of ``T^T A T``."""
        transform = self.transformation
        product = transform.T @ (matrix @ transform)
        if modal_sparse.is_sparse(product):
            return modal_sparse.to_csc(0.5 * (product + product.T))
        product = np.asarray(product)
        return 0.5 * (product + product.T)

    @property
    def independent_count(self) -> int:
        """Number of independent coordinates (columns of ``transformation``)."""
        return int(self.transformation.shape[1])

    def recover(self, coordinates: np.ndarray) -> np.ndarray:
        """Return the full DOF vector ``T q`` [m or rad] for independent coordinates ``q``."""
        return np.asarray(self.transformation @ np.asarray(coordinates, dtype=np.float64))

    def reduce_load(self, load: np.ndarray) -> np.ndarray:
        """Return the generalized load ``T^T f`` [N, N·m] for a full DOF load vector ``f``."""
        return np.asarray(self.transformation.T @ np.asarray(load, dtype=np.float64))

    def rigid_motion_residual(self, node_positions: np.ndarray, dofs_per_node: int = 6) -> float:
        """Return how well ``T`` reproduces rigid-body motion, as a relative maximum error.

        Constraints that are compatible with rigid motion, such as ties
        between coincident nodes and rigid links, admit every rigid-body
        displacement ``R``: there is an ``R_q`` with ``R = T R_q``. With
        known independent DOFs ``R_q`` is ``R`` restricted to them; for a given
        ``transformation`` it is the least-squares solution. A residual near
        roundoff means the transformed model keeps the six rigid-body modes.

        Args:
            node_positions: Node positions [m], shape ``[node_count, 3]``.
            dofs_per_node: Number of DOFs per node, 3 (``[ux, uy, uz]``) or 6
                (``[ux, uy, uz, rx, ry, rz]``), node-major.

        Returns:
            ``max |T R_q - R| / max |R|`` [dimensionless].
        """
        positions = np.asarray(node_positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] * dofs_per_node != self.dof_count:
            raise ValueError(
                f"node_positions must have shape [{self.dof_count // max(dofs_per_node, 1)}, 3] "
                f"for {self.dof_count} DOFs with dofs_per_node={dofs_per_node}, got {positions.shape}"
            )
        return transformation_rigid_residual(self.transformation, positions, dofs_per_node, self.independent_dofs)
