# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Loading of sparse matrices exported by finite-element tools."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scipy.sparse import csc_matrix


def load_sparse_matrix(path: str | os.PathLike[str], *, symmetric: bool = False) -> csc_matrix:
    """Load a sparse matrix from a Matrix Market or SciPy ``.npz`` file.

    Matrix Market files (``.mtx``) are read with :func:`scipy.io.mmread`,
    which expands storage declared ``symmetric`` to the full matrix; a file
    declared ``general`` that holds only one triangle is read as that
    triangle. ``.npz`` files must have been written with
    :func:`scipy.sparse.save_npz`. Duplicate entries are summed.

    Args:
        path: Path of a ``.mtx`` or ``.npz`` file.
        symmetric: Whether the matrix must be symmetric, as finite-element
            mass, stiffness, and damping matrices are. A relative asymmetry
            ``||A - A^T||_F / ||A||_F`` above ``1e-8``, as in a single
            stored triangle, then raises.

    Returns:
        The matrix as a float64 :class:`scipy.sparse.csc_matrix` with sorted
        indices.

    Raises:
        ImportError: If SciPy is not installed.
        ValueError: If the file extension is not ``.mtx`` or ``.npz``, or
            ``symmetric`` is set and the matrix is not symmetric.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".mtx", ".npz"):
        raise ValueError(f"load_sparse_matrix supports .mtx or .npz files, got '{path.name}'")
    try:
        from scipy import io, sparse
    except ImportError as exc:
        raise ImportError(
            "Loading sparse matrices requires SciPy; install it with `pip install scipy` or the "
            "`newton[importers]` extra"
        ) from exc
    if suffix == ".npz":
        matrix = sparse.load_npz(path)
    else:
        # Request a sparse array where supported; the sparse-matrix default is deprecated.
        options = {"spmatrix": False} if "spmatrix" in inspect.signature(io.mmread).parameters else {}
        matrix = io.mmread(path, **options)
    result = sparse.csc_matrix(matrix, dtype=np.float64)
    result.sum_duplicates()
    result.sort_indices()
    if symmetric:
        from ..sim.modal_sparse import check_symmetric  # noqa: PLC0415

        check_symmetric(result, path.name)
    return result
