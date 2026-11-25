# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``ConstraintElimination``, ``ModalGeneratorCraigBamptonFEM``, and ``load_sparse_matrix``.

These are setup-time NumPy/SciPy tests; they do not launch Warp kernels and
are registered once, independent of the test device. The reference-panel tests use
the finite-element matrices of the FE/ROM example panel when its folder is
available (set ``NEWTON_CB_PANEL_DIR`` to the folder that contains
``model/panel`` and ``rom/panel``) and are skipped otherwise.
"""

import importlib.util
import inspect
import json
import math
import os
import pickle
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

import newton
import newton.utils
from newton._src.sim import modal_sparse
from newton.tests.test_reduced_elastic_body import _lattice_craig_bampton, _spring_lattice
from newton.tests.unittest_utils import add_function_test

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

REF_DIR = Path(os.environ.get("NEWTON_CB_PANEL_DIR", "")).expanduser()
REF_FILES = (
    "model/panel/M.npz",
    "model/panel/K.npz",
    "model/panel/geometry.npz",
    "rom/panel/cb_48.npz",
    "rom/panel/reduction_report.json",
    "rom/panel/static_four_clamps_48.npz",
    "rom/panel/static_two_clamps_48.npz",
)
HAS_PANEL_DATA = (
    HAS_SCIPY and bool(os.environ.get("NEWTON_CB_PANEL_DIR")) and all((REF_DIR / name).is_file() for name in REF_FILES)
)
REF_INTERFACE_NAMES = ("left_front", "left_back", "right_front", "right_back")


def _skip_without_panel_data(test):
    if not HAS_PANEL_DATA:
        test.skipTest("Reference panel data not available; set NEWTON_CB_PANEL_DIR to the FE/ROM example panel folder")


def _load_panel():
    """Return the reference panel node positions [m], CSC mass [kg], CSC stiffness [N/m], and geometry arrays."""
    from scipy import sparse

    with np.load(REF_DIR / "model/panel/geometry.npz") as data:
        geometry = dict(data)
    mass = sparse.load_npz(REF_DIR / "model/panel/M.npz")
    stiffness = sparse.load_npz(REF_DIR / "model/panel/K.npz")
    return np.asarray(geometry["points"], dtype=np.float64), mass, stiffness, geometry


def _translation_rows(node_count, dofs_per_node=6):
    return (np.arange(node_count)[:, None] * dofs_per_node + np.arange(3)).reshape(-1)


def _gravity_load(mass, node_count, dofs_per_node=6, gravity=-9.81):
    """Return the consistent nodal gravity load ``M g`` [N] along ``z``."""
    acceleration = np.zeros(node_count * dofs_per_node)
    acceleration[2::dofs_per_node] = gravity
    return np.asarray(mass @ acceleration)


def _tie_rows(first_nodes, second_nodes, node_count, dofs_per_node=6):
    """Return CSR rows ``u_a - u_b = 0`` for all DOFs of each node pair."""
    from scipy import sparse

    k = np.arange(dofs_per_node)
    first = (np.asarray(first_nodes)[:, None] * dofs_per_node + k).reshape(-1)
    second = (np.asarray(second_nodes)[:, None] * dofs_per_node + k).reshape(-1)
    count = first.shape[0]
    rows = np.concatenate((np.arange(count), np.arange(count)))
    return sparse.csr_matrix(
        (np.concatenate((np.ones(count), -np.ones(count))), (rows, np.concatenate((first, second)))),
        shape=(count, node_count * dofs_per_node),
    )


def _tiled_panel(nx, ny):
    """Tile the reference panel ``nx`` by ``ny`` times and tie coincident edge nodes.

    The copies are shifted by the exact panel extent, so edge nodes coincide
    bitwise. Each node is tied to the first node at its position (a star, so
    no tie is redundant).

    Returns:
        Node positions [m], block-diagonal CSC mass [kg] and stiffness [N/m],
        CSR tie rows, and the first-node index of every node.
    """
    from scipy import sparse

    points, mass, stiffness, _ = _load_panel()
    length = float(points[:, 0].max() - points[:, 0].min())
    width = float(points[:, 1].max() - points[:, 1].min())
    tiled = np.vstack([points + np.array([i * length, j * width, 0.0]) for j in range(ny) for i in range(nx)])
    copies = nx * ny
    mass_tiled = sparse.block_diag([mass] * copies, format="csc")
    stiffness_tiled = sparse.block_diag([stiffness] * copies, format="csc")
    _, first, inverse = np.unique(tiled, axis=0, return_index=True, return_inverse=True)
    lead = first[inverse.reshape(-1)]
    duplicate = np.flatnonzero(lead != np.arange(tiled.shape[0]))
    ties = _tie_rows(duplicate, lead[duplicate], tiled.shape[0])
    return tiled, mass_tiled, stiffness_tiled, ties, lead


def _patch(points, lead, corner, half_x=0.1, half_y=0.102):
    """Return the distinct nodes within a rectangle around ``corner`` (one per coincident group)."""
    delta = np.abs(points[:, :2] - np.asarray(corner)[:2])
    nodes = np.flatnonzero((delta[:, 0] < half_x) & (delta[:, 1] < half_y))
    return np.unique(lead[nodes])


def _reduced_static(generator, load, fixed_interfaces):
    """Solve the reduced static problem with the listed interface frames fixed; return the full DOF field."""
    reduced_count = generator.reduced_mass_matrix.shape[0]
    fixed = (np.asarray(fixed_interfaces)[:, None] * 6 + np.arange(6)).reshape(-1)
    free = np.setdiff1d(np.arange(reduced_count), fixed)
    recovery = generator.full_recovery_matrix()
    reduced_load = recovery.T @ load
    coordinates = np.zeros(reduced_count)
    coordinates[free] = np.linalg.solve(generator.reduced_stiffness_matrix[np.ix_(free, free)], reduced_load[free])
    return recovery @ coordinates


def _relative(error, reference):
    return float(np.linalg.norm(error) / np.linalg.norm(reference))


def test_cb_fem_panel_reference(test, device):
    """Reproduce the stored 48-mode Craig-Bampton reduction of the reference panel from its full FE matrices.

    The stored reference uses four single-node, six-DOF interfaces at the panel
    corners and fixed-interface modes from SciPy's ``eigsh``. Fixed-interface
    frequencies, interface blocks, the reduced spectrum, the rigid mass, and
    the four-clamp gravity statics must agree to ``1e-6``.
    """
    _skip_without_panel_data(test)
    from scipy import linalg

    points, mass, stiffness, geometry = _load_panel()
    report = json.loads((REF_DIR / "rom/panel/reduction_report.json").read_text())
    with np.load(REF_DIR / "rom/panel/cb_48.npz") as data:
        stored = dict(data)
    interfaces = [
        {"name": name, "node_indices": [int(node)], "coupling": "node"}
        for name, node in zip(REF_INTERFACE_NAMES, geometry["interfaces"], strict=True)
    ]
    generator = newton.ModalGeneratorCraigBamptonFEM(points, mass, stiffness, interfaces, fixed_mode_count=48)
    craig_bampton = generator.reduce()
    test.assertEqual(generator.report["eigensolver"], "arpack")
    test.assertEqual(generator.report["dof_count"], 2394)
    test.assertEqual(generator.reduction_basis.shape, (2394, 72))

    np.testing.assert_allclose(
        generator.fixed_interface_frequencies, report["fixed_interface_frequencies_hz"], rtol=1.0e-6
    )
    reduced_mass = generator.reduced_mass_matrix
    reduced_stiffness = generator.reduced_stiffness_matrix
    for ours, theirs in ((reduced_mass, stored["mass_matrix"]), (reduced_stiffness, stored["stiffness_matrix"])):
        test.assertLess(_relative(ours[:24, :24] - theirs[:24, :24], theirs[:24, :24]), 1.0e-6)
    # The interface-mode mass coupling agrees up to mode sign; the stiffness coupling vanishes.
    test.assertAlmostEqual(
        float(np.linalg.norm(reduced_mass[:24, 24:]) / np.linalg.norm(stored["mass_matrix"][:24, 24:])),
        1.0,
        delta=1.0e-6,
    )
    test.assertLess(float(np.linalg.norm(reduced_stiffness[:24, 24:])), 1.0e-9 * np.linalg.norm(reduced_stiffness))
    ours = linalg.eigh(reduced_stiffness, reduced_mass, eigvals_only=True)
    theirs = linalg.eigh(stored["stiffness_matrix"], stored["mass_matrix"], eigvals_only=True)
    np.testing.assert_allclose(ours[6:], theirs[6:], rtol=1.0e-6)
    test.assertLess(float(np.max(np.abs(ours[:6]))), 1.0e-8 * float(ours[-1]))
    test.assertLess(generator.report["constraint_mode_residual"], 1.0e-10)
    test.assertLess(generator.report["rigid_stiffness_residual"], 1.0e-9)

    basis = craig_bampton.build()
    test.assertEqual(basis.mode_count, 66)
    test.assertAlmostEqual(craig_bampton.mass, report["mass_kg"], delta=1.0e-9 * report["mass_kg"])
    test.assertEqual(basis.frame_labels, list(REF_INTERFACE_NAMES))

    load = _gravity_load(mass, points.shape[0])
    translations = _translation_rows(points.shape[0])
    with np.load(REF_DIR / "rom/panel/static_four_clamps_48.npz") as data:
        static = dict(data)
    displacement = _reduced_static(generator, load, [0, 1, 2, 3])
    test.assertLess(_relative((displacement - static["rom"])[translations], static["rom"][translations]), 1.0e-6)
    field_error = _relative((displacement - static["full"])[translations], static["full"][translations])
    four_clamps = next(case for case in report["cases"] if case["case"] == "four_clamps")
    test.assertAlmostEqual(field_error, four_clamps["displacement_relative_l2"], delta=1.0e-6)
    test.assertAlmostEqual(field_error, 0.0069, delta=0.0001)
    with np.load(REF_DIR / "rom/panel/static_two_clamps_48.npz") as data:
        static_two = dict(data)
    displacement = _reduced_static(generator, load, [0, 1])
    test.assertLess(
        _relative((displacement - static_two["rom"])[translations], static_two["rom"][translations]), 1.0e-6
    )

    # The arrays follow the field names of the stored file, so the existing import contract holds.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "cb_48.npz"
        generator.save_npz(path, triangles=geometry["triangles"])
        with np.load(path) as data:
            saved = dict(data)
        for name, value in stored.items():
            if name in ("damping_matrix", "beta_seconds"):
                continue
            test.assertIn(name, saved)
            test.assertEqual(saved[name].shape, value.shape, name)
        np.testing.assert_array_equal(saved["interfaces"], stored["interfaces"])
        np.testing.assert_array_equal(saved["interface_names"], stored["interface_names"])
        np.testing.assert_allclose(saved["recovery_matrix"][:, :24], stored["recovery_matrix"][:, :24], atol=1.0e-9)
        reloaded = newton.ModalGeneratorCraigBampton(
            **{
                name: saved[name]
                for name in (
                    "interface_positions",
                    "mass_matrix",
                    "stiffness_matrix",
                    "sample_points",
                    "recovery_matrix",
                )
            },
            interface_names=saved["interface_names"].tolist(),
        ).build()
        np.testing.assert_array_equal(reloaded.mode_stiffness, basis.mode_stiffness)


def _two_panels():
    """Return two reference panels side by side in ``x`` with their 17 shared edge nodes tied (102 tie rows)."""
    from scipy import sparse

    points, mass, stiffness, _ = _load_panel()
    node_count = points.shape[0]
    length = float(points[:, 0].max() - points[:, 0].min())
    both = np.vstack((points, points + np.array([length, 0.0, 0.0])))
    first = np.flatnonzero(points[:, 0] == points[:, 0].max())
    second = node_count + np.flatnonzero(points[:, 0] == points[:, 0].min())
    first = first[np.argsort(points[first, 1])]
    second = second[np.argsort(both[second, 1])]
    np.testing.assert_array_equal(both[first], both[second])
    ties = _tie_rows(first, second, both.shape[0])
    return (
        both,
        sparse.block_diag([mass, mass], format="csc"),
        sparse.block_diag([stiffness, stiffness], format="csc"),
        ties,
        node_count,
    )


def _jacobi_scaled_solve(matrix, rhs, refinements=3):
    """Solve a sparse system with SuperLU after symmetric Jacobi scaling, with iterative refinement."""
    from scipy import sparse
    from scipy.sparse import linalg as sparse_linalg

    diagonal = np.abs(matrix.diagonal())
    scale = np.ones_like(diagonal)
    scale[diagonal > 0.0] = 1.0 / np.sqrt(diagonal[diagonal > 0.0])
    scaled = sparse.csc_matrix(sparse.diags(scale) @ matrix @ sparse.diags(scale))
    factor = sparse_linalg.splu(scaled)
    scaled_rhs = scale * rhs
    solution = factor.solve(scaled_rhs)
    for _ in range(refinements):
        solution += factor.solve(scaled_rhs - scaled @ solution)
    return scale * solution


def test_constraint_elimination_two_panel_tie(test, device):
    """Tie two reference panels along a shared edge and match the Lagrange-multiplier (KKT) statics.

    The 102 tie rows take the Boolean fast path: every row of ``T`` has a
    single unit entry. The tied model keeps exactly six rigid-body modes, and
    its static gravity response with one edge clamped equals the solution of
    the original constrained system with Lagrange multipliers.
    """
    _skip_without_panel_data(test)
    from scipy import sparse
    from scipy.sparse import linalg as sparse_linalg

    points, mass, stiffness, ties, node_count = _two_panels()
    elimination = newton.ConstraintElimination(mass, stiffness, ties)
    transform = elimination.transformation
    test.assertEqual(transform.shape, (4788, 4686))
    test.assertEqual(elimination.report["tie_rows"], 102)
    test.assertEqual(elimination.report["general_rows"], 0)
    test.assertEqual(elimination.report["constraint_residual"], 0.0)
    test.assertTrue(np.all(np.diff(transform.tocsr().indptr) == 1))
    test.assertTrue(np.all(transform.data == 1.0))
    np.testing.assert_array_equal(transform[elimination.independent_dofs].toarray(), np.eye(4686))
    test.assertEqual(elimination.rigid_motion_residual(points, 6), 0.0)

    # Exactly six zero-energy modes: the seventh eigenvalue is elastic and far from zero.
    eigenvalues = sparse_linalg.eigsh(
        elimination.reduced_stiffness_matrix,
        k=8,
        M=elimination.reduced_mass_matrix,
        sigma=-1.0,
        which="LM",
        v0=np.ones(4686),
    )[0]
    eigenvalues = np.sort(eigenvalues)
    test.assertLess(float(np.max(np.abs(eigenvalues[:6]))), 1.0e-4)
    test.assertGreater(float(eigenvalues[6]), 1.0e2)

    load = _gravity_load(mass, points.shape[0])
    clamped = (np.flatnonzero(points[:node_count, 0] == points[:, 0].min())[:, None] * 6 + np.arange(6)).reshape(-1)
    free_q = np.flatnonzero(~np.isin(elimination.independent_dofs, clamped))
    reduced_load = elimination.reduce_load(load)
    coordinates = np.zeros(4686)
    coordinates[free_q] = _jacobi_scaled_solve(
        elimination.reduced_stiffness_matrix[free_q][:, free_q], reduced_load[free_q]
    )
    displacement = elimination.recover(coordinates)

    free = np.setdiff1d(np.arange(stiffness.shape[0]), clamped)
    constraint_free = ties[:, free]
    kkt = sparse.bmat([[stiffness[free][:, free], constraint_free.T], [constraint_free, None]], format="csc")
    solution = _jacobi_scaled_solve(kkt, np.concatenate((load[free], np.zeros(ties.shape[0]))))
    reference = np.zeros(stiffness.shape[0])
    reference[free] = solution[: free.shape[0]]
    test.assertLess(_relative(displacement - reference, reference), 1.0e-8)


def test_cb_fem_two_panel_rbe2_statics(test, device):
    """Reduce the tied two-panel model with RBE2 corner interfaces and compare gravity statics to full FE.

    Each outer corner carries an RBE2 frame at the centroid of its 3x3 node
    patch. The full-FE reference solves the same constrained model (ties and
    rigid patches) without modal truncation.
    """
    _skip_without_panel_data(test)
    from scipy.sparse import linalg as sparse_linalg

    points, mass, stiffness, ties, _ = _two_panels()
    lead = np.arange(points.shape[0])
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    corners = ((x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max))
    interfaces = [
        {"name": f"corner_{index}", "node_indices": _patch(points, lead, corner), "coupling": "rbe2"}
        for index, corner in enumerate(corners)
    ]
    test.assertEqual([len(interface["node_indices"]) for interface in interfaces], [9, 9, 9, 9])
    generator = newton.ModalGeneratorCraigBamptonFEM(
        points, mass, stiffness, interfaces, fixed_mode_count=24, constraints=ties
    )
    basis = generator.build()
    test.assertEqual(basis.mode_count, 42)
    test.assertEqual(generator.report["constraint_elimination"]["tie_rows"], 102)
    test.assertLess(generator.report["constraint_mode_residual"], 1.0e-10)

    load = _gravity_load(mass, points.shape[0])
    translations = _translation_rows(points.shape[0])
    transform = generator.transformation
    stiffness_z = (transform.T @ stiffness @ transform).tocsc()
    load_z = transform.T @ load
    frames = np.arange(generator.interior_count, transform.shape[1])
    errors = {}
    for fixed in ([0, 1, 2, 3], [0, 1]):
        fixed_z = frames[(np.asarray(fixed)[:, None] * 6 + np.arange(6)).reshape(-1)]
        free_z = np.setdiff1d(np.arange(transform.shape[1]), fixed_z)
        coordinates = np.zeros(transform.shape[1])
        coordinates[free_z] = sparse_linalg.splu(stiffness_z[free_z][:, free_z]).solve(load_z[free_z])
        full = transform @ coordinates
        reduced = _reduced_static(generator, load, fixed)
        errors[len(fixed)] = _relative((reduced - full)[translations], full[translations])
    test.assertLess(errors[4], 0.02)
    test.assertLess(errors[2], 0.02)


def _hand_built_reduction(points, mass, stiffness, interface_nodes, fixed_mode_count, distributing):
    """Return a dense hand-built single-interface Craig-Bampton reduction of a three-DOF model.

    The interface frame sits at the centroid of ``interface_nodes``. For a
    distributing interface its coordinates are the least-squares rigid fit of
    the nodes; for a rigid (RBE2) interface the nodes follow the frame rigidly.
    """
    node_count = points.shape[0]
    center = points[interface_nodes].mean(axis=0)

    def rigid_rows(nodes):
        rows = []
        for node in nodes:
            x, y, z = points[node] - center
            rows.append(np.array([[1, 0, 0, 0, z, -y], [0, 1, 0, -z, 0, x], [0, 0, 1, y, -x, 0]], dtype=np.float64))
        return np.vstack(rows)

    rigid_all = rigid_rows(range(node_count))
    if distributing:
        return _lattice_craig_bampton(False, fixed_mode_count)[3:6]
    interface_dofs = (np.asarray(interface_nodes)[:, None] * 3 + np.arange(3)).reshape(-1)
    interior = np.setdiff1d(np.arange(3 * node_count), interface_dofs)
    coordinates = np.zeros((3 * node_count, 6 + interior.shape[0]))
    coordinates[interface_dofs, :6] = rigid_rows(interface_nodes)
    coordinates[interior, 6:] = np.eye(interior.shape[0])
    mass_z = coordinates.T @ mass @ coordinates
    stiffness_z = coordinates.T @ stiffness @ coordinates
    constraint_modes = -np.linalg.solve(stiffness_z[6:, 6:], stiffness_z[6:, :6])
    chol = np.linalg.cholesky(mass_z[6:, 6:])
    standard = np.linalg.solve(chol, np.linalg.solve(chol, stiffness_z[6:, 6:]).T).T
    _, vectors = np.linalg.eigh(0.5 * (standard + standard.T))
    modes = np.linalg.solve(chol.T, vectors[:, :fixed_mode_count])
    basis = np.zeros((6 + interior.shape[0], 6 + fixed_mode_count))
    basis[:6, :6] = np.eye(6)
    basis[6:, :6] = constraint_modes
    basis[6:, 6:] = modes
    recovery = coordinates @ basis
    np.testing.assert_allclose(recovery[:, :6], rigid_all, atol=1.0e-9)
    return recovery.T @ mass @ recovery, recovery.T @ stiffness @ recovery, recovery


def _assert_same_reduction(test, generator, reference, tolerance):
    """Compare a single-interface reduction with a reference up to mode sign and rotation within eigenspaces."""
    reduced_mass, reduced_stiffness, recovery = reference
    mass, stiffness = generator.reduced_mass_matrix, generator.reduced_stiffness_matrix
    scale_m = float(np.max(np.abs(reduced_mass)))
    scale_k = float(np.max(np.abs(reduced_stiffness)))
    np.testing.assert_allclose(mass[:6, :6], reduced_mass[:6, :6], rtol=0.0, atol=tolerance * scale_m)
    np.testing.assert_allclose(stiffness[:6, :6], reduced_stiffness[:6, :6], rtol=0.0, atol=tolerance * scale_k)
    np.testing.assert_allclose(
        np.diag(stiffness)[6:], np.diag(reduced_stiffness)[6:], rtol=0.0, atol=tolerance * scale_k
    )
    np.testing.assert_allclose(mass[6:, 6:], np.eye(mass.shape[0] - 6), rtol=0.0, atol=tolerance)
    # Invariant under sign flips and rotations within repeated eigenvalues.
    np.testing.assert_allclose(
        mass[:6, 6:] @ mass[6:, :6], reduced_mass[:6, 6:] @ reduced_mass[6:, :6], rtol=0.0, atol=tolerance * scale_m**2
    )
    ours = generator.full_recovery_matrix()
    np.testing.assert_allclose(ours[:, :6], recovery[:, :6], rtol=0.0, atol=tolerance)
    np.testing.assert_allclose(
        ours[:, 6:] @ ours[:, 6:].T, recovery[:, 6:] @ recovery[:, 6:].T, rtol=0.0, atol=tolerance
    )


def test_cb_fem_lattice_rbe3_and_rbe2_match_hand_built(test, device):
    """Match dense hand-built single-interface reductions of a three-DOF lattice with RBE3 and RBE2 interfaces.

    The RBE3 reference is ``_lattice_craig_bampton`` of the reduced elastic
    body tests; its 8-mode cut does not split a repeated eigenvalue. Both the
    dense and the ARPACK paths are checked.
    """
    from scipy import sparse

    points, mass, stiffness = _spring_lattice((3, 3, 3), 0.1, 1000.0, 0.01)
    top = np.flatnonzero(np.isclose(points[:, 2], points[:, 2].max()))
    for coupling, distributing in (("rbe3", True), ("rbe2", False)):
        reference = _hand_built_reduction(points, mass, stiffness, top, 8, distributing)
        for eigensolver, matrices in (
            ("dense", (mass, stiffness)),
            ("arpack", (sparse.csc_matrix(mass), sparse.csc_matrix(stiffness))),
        ):
            generator = newton.ModalGeneratorCraigBamptonFEM(
                points,
                *matrices,
                [{"node_indices": top, "coupling": coupling}],
                fixed_mode_count=8,
                dofs_per_node=3,
                eigensolver=eigensolver,
            )
            generator.reduce()
            test.assertEqual(generator.report["eigensolver"], eigensolver)
            _assert_same_reduction(test, generator, reference, 1.0e-8)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Craig-Bampton interfaces", category=UserWarning)
                basis = generator.craig_bampton.build()
            test.assertEqual(basis.mode_count, 8)


def test_cb_fem_transformation_matches_constraints(test, device):
    """Give the same result for ``constraints`` and the equivalent ``transformation``, dense or sparse."""
    from scipy import sparse

    points, mass, stiffness = _spring_lattice((4, 2, 2), 0.1, 1000.0, 0.01)
    node_count = points.shape[0]
    # Tie the lattice to a coincident copy: the copy carries half the mass and stiffness.
    both = np.vstack((points, points))
    mass_both = sparse.csc_matrix(np.kron(np.eye(2), 0.5 * mass))
    stiffness_both = sparse.csc_matrix(np.kron(np.eye(2), 0.5 * stiffness))
    ties = _tie_rows(np.arange(node_count), node_count + np.arange(node_count), 2 * node_count, 3)
    elimination = newton.ConstraintElimination(mass_both, stiffness_both, ties)
    left = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].min()))
    right = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].max()))
    interfaces = [{"node_indices": left}, {"node_indices": right, "coupling": "rbe3"}]
    single = newton.ModalGeneratorCraigBamptonFEM(
        points, mass, stiffness, interfaces, fixed_mode_count=6, dofs_per_node=3, eigensolver="dense"
    )
    single.reduce()
    for kwargs in (
        {"constraints": ties},
        {"transformation": elimination.transformation},
        {"constraints": ties.toarray(), "eigensolver": "dense"},
    ):
        generator = newton.ModalGeneratorCraigBamptonFEM(
            both, mass_both, stiffness_both, interfaces, fixed_mode_count=6, dofs_per_node=3, **kwargs
        )
        generator.reduce()
        np.testing.assert_allclose(generator.fixed_interface_eigenvalues, single.fixed_interface_eigenvalues, rtol=1e-9)
        np.testing.assert_allclose(
            generator.reduced_mass_matrix[:12, :12], single.reduced_mass_matrix[:12, :12], rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            generator.reduced_stiffness_matrix[:12, :12],
            single.reduced_stiffness_matrix[:12, :12],
            rtol=0.0,
            atol=1e-9 * float(np.max(np.abs(single.reduced_stiffness_matrix))),
        )


def test_constraint_elimination_general_path(test, device):
    """Eliminate tie chains, scaled pairs, rigid links, and averages, and drop redundant rows with a warning."""
    from scipy import sparse

    size = 12
    rows = [
        {0: 1.0, 1: -1.0},  # tie chain 0 = 1 = 2
        {1: 2.0, 2: -2.0},
        {3: 1.0, 4: -2.0},  # scaled pair: general path
        {5: 1.0, 6: -1.0, 7: 0.5},  # rigid link with a lever arm
        {8: 0.25, 9: 0.25, 10: 0.5, 11: -1.0},  # weighted average
        {0: -3.0, 2: 3.0},  # redundant tie closing the chain into a cycle
        {3: 1.0, 4: -2.0},  # exact duplicate
    ]
    dense = np.zeros((len(rows), size))
    for index, row in enumerate(rows):
        for col, value in row.items():
            dense[index, col] = value
    for constraints in (dense, sparse.csr_matrix(dense)):
        with test.assertWarnsRegex(UserWarning, r"Dropped 2 linearly dependent constraint rows \(1 exact duplicates\)"):
            elimination = newton.ConstraintElimination(np.eye(size), np.eye(size), constraints, retained_dofs=[11])
        transform = elimination.transformation
        transform = transform.toarray() if sparse.issparse(transform) else transform
        test.assertEqual(sparse.issparse(elimination.transformation), sparse.issparse(constraints))
        np.testing.assert_allclose(dense @ transform, 0.0, atol=1.0e-15)
        test.assertEqual(elimination.report["tie_rows"], 3)
        test.assertEqual(elimination.report["general_rows"], 4)
        test.assertEqual(elimination.report["dependent_rows"], 2)
        test.assertEqual(elimination.independent_count, size - 5)
        test.assertIn(11, elimination.independent_dofs.tolist())
        np.testing.assert_array_equal(transform[elimination.independent_dofs], np.eye(size - 5))
        np.testing.assert_array_equal(
            np.sort(np.concatenate((elimination.independent_dofs, elimination.dependent_dofs))), np.arange(size)
        )
        reduced_mass = elimination.reduced_mass_matrix
        reduced_mass = reduced_mass.toarray() if sparse.issparse(reduced_mass) else reduced_mass
        np.testing.assert_allclose(reduced_mass, transform.T @ transform, atol=1.0e-15)
        coordinates = np.arange(1.0, size - 4)
        np.testing.assert_allclose(elimination.recover(coordinates), transform @ coordinates)
        np.testing.assert_allclose(elimination.reduce_load(np.ones(size)), transform.T @ np.ones(size))


def test_cb_fem_error_paths(test, device):
    """Report inconsistent constraints, mechanisms, three-DOF single-node frames, and mismatched sizes clearly."""
    from scipy import sparse

    points, mass, stiffness = _spring_lattice((3, 2, 2), 0.1, 1000.0, 0.01)
    node_count = points.shape[0]
    dof_count = 3 * node_count
    left = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].min()))

    # Constraints that tie two retained DOFs, or relate retained DOFs only, are inconsistent.
    with test.assertRaisesRegex(ValueError, "retained DOFs"):
        newton.ConstraintElimination(np.eye(4), np.eye(4), [[1.0, -1.0, 0.0, 0.0]], retained_dofs=[0, 1])
    with test.assertRaisesRegex(ValueError, "without making a retained DOF dependent"):
        newton.ConstraintElimination(np.eye(4), np.eye(4), [[1.0, 2.0, 0.0, 0.0]], retained_dofs=[0, 1])
    with test.assertRaisesRegex(ValueError, "without making a retained DOF dependent"):
        newton.ConstraintElimination(
            np.eye(4), np.eye(4), [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]], retained_dofs=[0, 1]
        )
    with test.assertRaisesRegex(ValueError, "exactly one of constraints and transformation"):
        newton.ConstraintElimination(np.eye(4), np.eye(4))
    with test.assertRaisesRegex(ValueError, "constraints must have 4 columns"):
        newton.ConstraintElimination(np.eye(4), np.eye(4), np.ones((1, 3)))
    with test.assertRaisesRegex(ValueError, "non-finite"):
        newton.ConstraintElimination(np.eye(4), np.eye(4), [[1.0, np.nan, 0.0, 0.0]])
    with test.assertRaisesRegex(ValueError, "stiffness_matrix must have shape"):
        newton.ConstraintElimination(np.eye(4), np.eye(3), [[1.0, -1.0, 0.0, 0.0]])

    def make(**kwargs):
        options = {"fixed_mode_count": 4, "dofs_per_node": 3}
        options.update(kwargs)
        interfaces = options.pop("interfaces", [{"node_indices": left}])
        matrices = options.pop("matrices", (mass, stiffness))
        return newton.ModalGeneratorCraigBamptonFEM(points, *matrices, interfaces, **options)

    with test.assertRaisesRegex(ValueError, "coupling 'node' needs dofs_per_node=6"):
        make(interfaces=[{"node_indices": [0], "coupling": "node"}])
    with test.assertRaisesRegex(ValueError, "at least three non-collinear nodes"):
        make(interfaces=[{"node_indices": [0, 1], "coupling": "rbe2"}])
    with test.assertRaisesRegex(ValueError, "at least three non-collinear nodes"):
        make(interfaces=[{"node_indices": left[:2], "coupling": "rbe3"}])
    with test.assertRaisesRegex(ValueError, "cannot belong to another interface"):
        make(interfaces=[{"node_indices": left}, {"node_indices": left[:3], "coupling": "rbe3"}])
    with test.assertRaisesRegex(ValueError, "coupling must be one of"):
        make(interfaces=[{"node_indices": left, "coupling": "glue"}])
    with test.assertRaisesRegex(ValueError, "mass_matrix must have shape"):
        make(matrices=(np.eye(dof_count - 3), stiffness))
    with test.assertRaisesRegex(ValueError, "constraints must have 36 columns"):
        make(constraints=np.ones((1, dof_count + 1))).reduce()
    with test.assertRaisesRegex(ValueError, "at most one of constraints and transformation"):
        make(constraints=np.eye(dof_count)[:1], transformation=np.eye(dof_count))
    with test.assertRaisesRegex(ValueError, "fixed_mode_count 40 exceeds"):
        make(fixed_mode_count=40).reduce()

    # Two lattices without a tie: with the interface fixed, the second one is a free mechanism.
    pair_points = np.vstack((points, points + np.array([1.0, 0.0, 0.0])))
    pair_mass = np.kron(np.eye(2), mass)
    pair_stiffness = np.kron(np.eye(2), stiffness)
    for eigensolver, matrices in (
        ("dense", (pair_mass, pair_stiffness)),
        ("arpack", (sparse.csc_matrix(pair_mass), sparse.csc_matrix(pair_stiffness))),
    ):
        generator = newton.ModalGeneratorCraigBamptonFEM(
            pair_points,
            *matrices,
            [{"node_indices": left}],
            fixed_mode_count=4,
            dofs_per_node=3,
            eigensolver=eigensolver,
        )
        with test.assertRaisesRegex(ValueError, "zero-energy mechanism"):
            generator.reduce()
    # An interface frame tied to a node of another rigid interface over-constrains it.
    ties = np.zeros((3, dof_count))
    ties[np.arange(3), 3 * left[0] + np.arange(3)] = 1.0
    ties[np.arange(3), 3 * (node_count - 1) + np.arange(3)] = -1.0
    right = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].max()))
    with test.assertRaisesRegex(ValueError, "retained DOF"):
        make(interfaces=[{"node_indices": left}, {"node_indices": right}], constraints=ties).reduce()


def test_fe_matrices_must_be_symmetric(test, device):
    """Reject triangular-storage or asymmetric matrices instead of silently using their symmetric part."""
    from scipy import io, sparse

    points, mass, stiffness = _spring_lattice((3, 2, 2), 0.1, 1000.0, 0.01)
    dof_count = mass.shape[0]
    left = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].min()))
    interfaces = [{"node_indices": left}]
    # A consistent-type mass with off-diagonal coupling, still positive definite.
    coupled = mass + 1.0e-3 * (np.eye(dof_count, k=3) + np.eye(dof_count, k=-3))
    options = {"fixed_mode_count": 4, "dofs_per_node": 3}
    upper_mass = sparse.triu(sparse.csc_matrix(coupled), format="csc")
    upper_stiffness = sparse.triu(sparse.csc_matrix(stiffness), format="csc")
    with test.assertRaisesRegex(ValueError, "mass_matrix is not symmetric.*expanded"):
        newton.ModalGeneratorCraigBamptonFEM(points, upper_mass, stiffness, interfaces, **options)
    with test.assertRaisesRegex(ValueError, "stiffness_matrix is not symmetric"):
        newton.ModalGeneratorCraigBamptonFEM(points, coupled, upper_stiffness.toarray(), interfaces, **options)
    with test.assertRaisesRegex(ValueError, "damping_matrix is not symmetric"):
        newton.ModalGeneratorCraigBamptonFEM(
            points, coupled, stiffness, interfaces, damping_matrix=1.0e-3 * upper_stiffness, **options
        )
    ties = np.zeros((1, dof_count))
    ties[0, [0, 3]] = [1.0, -1.0]
    with test.assertRaisesRegex(ValueError, "mass_matrix is not symmetric"):
        newton.ConstraintElimination(upper_mass, stiffness, ties)
    with test.assertRaisesRegex(ValueError, "stiffness_matrix is not symmetric"):
        newton.ModalGeneratorFEM(points, coupled, upper_stiffness, mode_count=4)

    # Roundoff-sized asymmetry is accepted and removed.
    noise = np.random.default_rng(1).uniform(-1.0, 1.0, stiffness.shape) * 1.0e-12 * np.abs(stiffness).max()
    perturbed = stiffness + noise
    generator = newton.ModalGeneratorCraigBamptonFEM(points, coupled, perturbed, interfaces, **options)
    np.testing.assert_array_equal(generator.stiffness_matrix, 0.5 * (perturbed + perturbed.T))

    # A Matrix Market file declared "general" that holds one triangle is not expanded on load.
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        io.mmwrite(folder / "upper.mtx", upper_mass)
        io.mmwrite(folder / "full.mtx", sparse.csc_matrix(coupled), symmetry="symmetric")
        test.assertEqual(newton.utils.load_sparse_matrix(folder / "upper.mtx").nnz, upper_mass.nnz)
        with test.assertRaisesRegex(ValueError, "upper.mtx is not symmetric"):
            newton.utils.load_sparse_matrix(folder / "upper.mtx", symmetric=True)
        full = newton.utils.load_sparse_matrix(folder / "full.mtx", symmetric=True)
        np.testing.assert_allclose(full.toarray(), coupled)


def test_cb_fem_constraint_errors_name_interfaces(test, device):
    """Name interfaces, nodes, and original DOFs in constraint conflicts, not internal coordinate indices."""
    points, mass, stiffness = _spring_lattice((3, 2, 2), 0.1, 1000.0, 0.01)
    dof_count = mass.shape[0]
    left = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].min()))
    right = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].max()))
    # Frames at a node: that node's translations follow the frame through pure tie rows.
    interfaces = [
        {"name": "left", "node_indices": left, "position": points[left[0]]},
        {"name": "right", "node_indices": right, "position": points[right[0]]},
    ]

    def reduce(constraints):
        return newton.ModalGeneratorCraigBamptonFEM(
            points, mass, stiffness, interfaces, fixed_mode_count=4, dofs_per_node=3, constraints=constraints
        ).reduce()

    # An SPC on a DOF of an interface node leaves its coupling row with frame coordinates only.
    dof = 3 * int(left[1]) + 2
    spc = np.zeros((1, dof_count))
    spc[0, dof] = 1.0
    pattern = rf"coupling of interface 'left' to DOF {dof} \(node {int(left[1])} uz\), which the constraints eliminate"
    with test.assertRaisesRegex(ValueError, pattern):
        reduce(spc)

    # Tying the two frame nodes ties the two frames: name the frame coordinates and the shared DOF.
    ties = np.zeros((3, dof_count))
    ties[np.arange(3), 3 * int(left[0]) + np.arange(3)] = 1.0
    ties[np.arange(3), 3 * int(right[0]) + np.arange(3)] = -1.0
    pattern = (
        rf"retained DOFs interface 'left' frame tx, interface 'right' frame tx through DOF {3 * int(left[0])} "
        rf"\(node {int(left[0])} ux\)"
    )
    with test.assertRaisesRegex(ValueError, pattern):
        reduce(ties)


def test_transformation_rank_checked(test, device):
    """Report a transformation with dependent columns as such, not as a mechanism or a singular mass."""
    from scipy import sparse

    points, mass, stiffness = _spring_lattice((3, 2, 2), 0.1, 1000.0, 0.01)
    dof_count = mass.shape[0]
    left = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].min()))
    duplicated = np.hstack((np.eye(dof_count), np.eye(dof_count)[:, :1]))
    pattern = rf"transformation has linearly dependent columns: its rank is {dof_count}, below its {dof_count + 1}"
    for transform in (duplicated, sparse.csc_matrix(duplicated)):
        with test.assertRaisesRegex(ValueError, pattern):
            newton.ConstraintElimination(mass, stiffness, transformation=transform)
        with test.assertRaisesRegex(ValueError, pattern):
            newton.ModalGeneratorCraigBamptonFEM(
                points,
                mass,
                stiffness,
                [{"node_indices": left}],
                fixed_mode_count=4,
                dofs_per_node=3,
                transformation=transform,
            ).reduce()
        # A full-rank transformation, for example from scaled and combined columns, is accepted.
        newton.ConstraintElimination(mass, stiffness, transformation=transform[:, :dof_count] * 3.0)


def test_cb_fem_reduce_rejects_non_rigid_constraints(test, device):
    """Reject constraints that ground the body or tie distant nodes in reduce(), not only in build()."""
    points, mass, stiffness = _spring_lattice((4, 2, 2), 0.1, 1000.0, 0.01)
    dof_count = mass.shape[0]
    left = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].min()))
    right = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].max()))
    interfaces = [{"node_indices": left, "coupling": "rbe3"}, {"node_indices": right}]
    # An SPC on a node of the RBE3 patch grounds the body.
    spc = np.zeros((1, dof_count))
    spc[0, 3 * int(left[1]) + 2] = 1.0
    # A tie between nodes at different positions, without the lever arm, resists rigid rotation.
    tie = np.zeros((3, dof_count))
    interior = np.flatnonzero(~np.isin(np.arange(points.shape[0]), np.concatenate((left, right))))
    tie[np.arange(3), 3 * int(interior[0]) + np.arange(3)] = 1.0
    tie[np.arange(3), 3 * int(interior[-1]) + np.arange(3)] = -1.0
    for constraints in (spc, tie):
        generator = newton.ModalGeneratorCraigBamptonFEM(
            points, mass, stiffness, interfaces, fixed_mode_count=4, dofs_per_node=3, constraints=constraints
        )
        with test.assertRaisesRegex(ValueError, "constraints do not admit rigid-body motion"):
            generator.reduce()


def test_cb_fem_large_tiled_model(test, device):
    """Reduce a 59,850-DOF tiling of 5x5 reference panels with 8 RBE2 interfaces and 30 modes within 120 s on CPU."""
    _skip_without_panel_data(test)
    points, mass, stiffness, ties, lead = _tiled_panel(5, 5)
    test.assertEqual(stiffness.shape[0], 59850)
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_mid, y_mid = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    corners = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]
    corners += [(x_mid, y_min), (x_mid, y_max), (x_min, y_mid), (x_max, y_mid)]
    interfaces = [{"node_indices": _patch(points, lead, corner)} for corner in corners]
    start = time.perf_counter()
    generator = newton.ModalGeneratorCraigBamptonFEM(
        points,
        mass,
        stiffness,
        interfaces,
        fixed_mode_count=30,
        constraints=ties,
        sample_node_indices=np.unique(lead)[::5],
    )
    basis = generator.build()
    elapsed = time.perf_counter() - start
    test.assertEqual(basis.mode_count, 6 * 8 + 30 - 6)
    test.assertLess(generator.report["constraint_mode_residual"], 1.0e-8)
    test.assertLess(generator.report["eigen_residual"], 1.0e-6)
    test.assertTrue(np.all(np.diff(generator.fixed_interface_eigenvalues) >= 0.0))
    test.assertLess(elapsed, 120.0)


def test_cb_fem_dense_path_without_scipy(test, device):
    """Reduce a dense three-DOF lattice with RBE2 and RBE3 interfaces and a tie without SciPy."""
    points, mass, stiffness = _spring_lattice((3, 3, 2), 0.1, 1000.0, 0.01)
    node_count = points.shape[0]
    both = np.vstack((points, points))
    mass_both = np.kron(np.eye(2), 0.5 * mass)
    stiffness_both = np.kron(np.eye(2), 0.5 * stiffness)
    ties = np.hstack((np.eye(3 * node_count), -np.eye(3 * node_count)))
    left = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].min()))
    right = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].max()))
    interfaces = [{"node_indices": left}, {"node_indices": right, "coupling": "rbe3"}]
    blocked = {name: None for name in list(sys.modules) if name == "scipy" or name.startswith("scipy.")}
    blocked["scipy"] = None
    with mock.patch.dict(sys.modules, blocked):
        reference = newton.ModalGeneratorCraigBamptonFEM(
            points, mass, stiffness, interfaces, fixed_mode_count=5, dofs_per_node=3, sample_mass="lumped"
        )
        reference_basis = reference.build()
        tied = newton.ModalGeneratorCraigBamptonFEM(
            both, mass_both, stiffness_both, interfaces, fixed_mode_count=5, dofs_per_node=3, constraints=ties
        )
        tied.reduce()
        test.assertEqual(tied.report["eigensolver"], "dense")
        with test.assertRaisesRegex(ImportError, "pip install scipy"):
            newton.ModalGeneratorCraigBamptonFEM(
                points, mass, stiffness, interfaces, fixed_mode_count=5, dofs_per_node=3, eigensolver="arpack"
            ).reduce()
    test.assertEqual(reference_basis.mode_count, 12 + 5 - 6)
    np.testing.assert_allclose(reference.craig_bampton.sample_mass.sum(), np.trace(mass) / 3.0, rtol=1.0e-12)
    np.testing.assert_allclose(tied.fixed_interface_eigenvalues, reference.fixed_interface_eigenvalues, rtol=1.0e-9)


def test_load_sparse_matrix(test, device):
    """Load Matrix Market and SciPy ``.npz`` matrices as canonical float64 CSC matrices."""
    from scipy import io, sparse

    matrix = sparse.random(7, 7, density=0.4, random_state=3, format="coo")
    symmetric = (matrix + matrix.T).tocsc()
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        io.mmwrite(folder / "a.mtx", symmetric, symmetry="symmetric")
        sparse.save_npz(folder / "a.npz", symmetric.tocsr())
        for filename in ("a.mtx", "a.npz"):
            loaded = newton.utils.load_sparse_matrix(folder / filename)
            test.assertEqual(loaded.format, "csc")
            test.assertEqual(loaded.dtype, np.float64)
            np.testing.assert_allclose(loaded.toarray(), symmetric.toarray())
        (folder / "a.txt").write_text("1 2 3\n")
        with test.assertRaisesRegex(ValueError, r"\.mtx or \.npz"):
            newton.utils.load_sparse_matrix(folder / "a.txt")


def _frame_beam(element_count=40, length=1.0, rotary_fraction=1.0e-6, offset=(0.0, 0.0, 0.0)):
    """Return node positions, lumped mass, and stiffness of a straight six-DOF frame-element beam along x.

    Steel, 10 mm square section. The rotary inertia of each node is ``rotary_fraction`` times its
    translational mass [kg m^2 per kg], small as in shell models with drilling-ratio rotary inertia;
    the Gershgorin bound then exceeds the largest eigenvalue by orders of magnitude.
    """
    from scipy import sparse

    young, shear, density, side = 2.0e11, 8.0e10, 7850.0, 0.01
    area, inertia, torsion = side * side, side**4 / 12.0, 0.1406 * side**4
    step = length / element_count
    node_count = element_count + 1
    points = np.zeros((node_count, 3))
    points[:, 0] = np.linspace(0.0, length, node_count)
    points += np.asarray(offset, dtype=np.float64)
    a, t, b = young * area / step, shear * torsion / step, young * inertia
    element = np.zeros((12, 12))
    for i, j, value in ((0, 0, a), (0, 6, -a), (3, 3, t), (3, 9, -t)):
        element[i, j] += value
    element[6, 6] += a
    element[9, 9] += t
    # Bending about z (v, theta_z = DOFs 1, 5, 7, 11) and about y (w, theta_y = DOFs 2, 4, 8, 10).
    for (v1, r1, v2, r2), sign in (((1, 5, 7, 11), 1.0), ((2, 4, 8, 10), -1.0)):
        k = (
            b
            / step**3
            * np.array(
                [
                    [12.0, sign * 6.0 * step, -12.0, sign * 6.0 * step],
                    [sign * 6.0 * step, 4.0 * step**2, -sign * 6.0 * step, 2.0 * step**2],
                    [-12.0, -sign * 6.0 * step, 12.0, -sign * 6.0 * step],
                    [sign * 6.0 * step, 2.0 * step**2, -sign * 6.0 * step, 4.0 * step**2],
                ]
            )
        )
        index = np.array([v1, r1, v2, r2])
        element[np.ix_(index, index)] += k
    element = np.triu(element) + np.triu(element, 1).T
    stiffness = sparse.lil_matrix((6 * node_count, 6 * node_count))
    nodal_mass = np.zeros(node_count)
    for e in range(element_count):
        dofs = np.arange(6 * e, 6 * e + 12)
        stiffness[np.ix_(dofs, dofs)] = stiffness[np.ix_(dofs, dofs)].toarray() + element
        nodal_mass[e : e + 2] += 0.5 * density * area * step
    diagonal = np.repeat(nodal_mass, 6) * np.tile([1.0, 1.0, 1.0] + [rotary_fraction] * 3, node_count)
    return points, sparse.diags(diagonal).tocsc(), stiffness.tocsc()


def _interior_rotation_units(node_count, scale):
    """Return the diagonal of the congruence that measures the rotations of all nodes but node 0 in other units."""
    diagonal = np.ones(6 * node_count)
    for component in (3, 4, 5):
        diagonal[6 + component :: 6] = scale
    return diagonal


def _rescaled(matrix, diagonal):
    """Return the congruence ``D A D`` of a sparse matrix as CSC."""
    from scipy import sparse

    return sparse.csc_matrix(sparse.diags(diagonal) @ matrix @ sparse.diags(diagonal))


def _lowest_eigenvalues(stiffness, mass, count):
    """Return the lowest generalized eigenvalues of dense ``(K, M)``, accurate for any spread of the spectrum.

    The inverse problem ``M x = (1 / lambda) K x`` resolves them to roundoff of themselves; the direct one
    only to roundoff of the largest eigenvalue.
    """
    from scipy import linalg

    return 1.0 / linalg.eigh(mass, stiffness, eigvals_only=True)[::-1][:count]


def _energy_ratios(stiffness_ii, modes, eigenvalues):
    """Return ``lambda_j / (||K_s||_1 phi_j^T diag(K_ii) phi_j)`` of M-orthonormal modes, with ``K_s = D K_ii D``."""
    diagonal = stiffness_ii.diagonal()
    scale = 1.0 / np.sqrt(diagonal)
    norm = modal_sparse.matrix_one_norm(_rescaled(stiffness_ii, scale))
    return eigenvalues / (norm * np.einsum("di,d,di->i", modes, diagonal, modes))


def test_cb_fem_small_rotary_inertia_is_not_a_mechanism(test, device):
    """Keep genuine elastic modes of six-DOF models whose largest eigenvalue lies far above the lowest ones.

    Small rotary inertia lifts the top of the spectrum: a zero-energy threshold relative to the largest
    eigenvalue, or to its Gershgorin bound, then rises above genuine fixed-interface modes (welded shell
    models: a 27.9 Hz mode against a 46.6 Hz threshold, a 69.1 Hz mode against 308 Hz). The decision uses the
    energy ratio in the equilibrated stiffness metric, which does not involve the mass matrix, and a part that
    is not tied to the rest is still rejected.
    """
    from scipy import linalg, sparse

    points, mass, stiffness = _frame_beam()
    clamp = [{"name": "root", "node_indices": [0], "coupling": "node"}]
    interior = np.arange(6, stiffness.shape[0])
    stiffness_ii = stiffness[interior][:, interior]
    mass_ii = mass[interior][:, interior]
    exact = linalg.eigh(stiffness_ii.toarray(), mass_ii.toarray(), eigvals_only=True)
    bound = modal_sparse.generalized_eigenvalue_bound(stiffness_ii, mass_ii)
    test.assertGreater(bound / exact[-1], 50.0)

    generator = newton.ModalGeneratorCraigBamptonFEM(
        points, mass, stiffness, clamp, fixed_mode_count=4, eigensolver="arpack"
    )
    generator.reduce()
    np.testing.assert_allclose(generator.fixed_interface_eigenvalues, exact[:4], rtol=1.0e-6)
    report = generator.report
    test.assertEqual(report["zero_energy_decision"], "equilibrated_energy_ratio")
    test.assertEqual(report["zero_energy_scale_source"], "equilibrated_energy_ratio")
    tolerance = report["zero_energy_ratio_tolerance"]
    test.assertEqual(tolerance, 1.0e-12)
    # The reported ratios follow their definition: the mode ratio covers the retained modes and the margin,
    # and the stiffness ratio is an upper estimate of the smallest eigenvalue of K_s over its norm.
    ratios = _energy_ratios(stiffness_ii, generator.fixed_interface_modes, generator.fixed_interface_eigenvalues)
    test.assertLessEqual(report["zero_energy_mode_ratio_min"], float(np.min(ratios)) * (1.0 + 1.0e-9))
    np.testing.assert_allclose(report["zero_energy_mode_ratio_min"], float(np.min(ratios)), rtol=1.0e-6)
    scale = 1.0 / np.sqrt(stiffness_ii.diagonal())
    equilibrated = _rescaled(stiffness_ii, scale)
    softest = float(np.linalg.eigvalsh(equilibrated.toarray())[0]) / modal_sparse.matrix_one_norm(equilibrated)
    test.assertGreaterEqual(report["zero_energy_stiffness_ratio"], softest * (1.0 - 1.0e-9))
    test.assertLess(report["zero_energy_stiffness_ratio"], 10.0 * softest)
    test.assertEqual(
        report["zero_energy_ratio_min"],
        min(report["zero_energy_mode_ratio_min"], report["zero_energy_stiffness_ratio"]),
    )
    test.assertGreater(report["zero_energy_ratio_min"], 1.0e3 * tolerance)
    test.assertGreater(report["zero_energy_threshold"], 0.0)
    test.assertLess(report["zero_energy_threshold"], 1.0e-3 * exact[0])
    np.testing.assert_allclose(report["interior_stiffness_scaling_min"], float(np.min(scale)), rtol=1.0e-12)
    np.testing.assert_allclose(report["interior_stiffness_scaling_max"], float(np.max(scale)), rtol=1.0e-12)

    # The dense path decides in the same way and agrees.
    dense = newton.ModalGeneratorCraigBamptonFEM(
        points, mass.toarray(), stiffness.toarray(), clamp, fixed_mode_count=4, eigensolver="dense"
    )
    dense.reduce()
    test.assertEqual(dense.report["zero_energy_decision"], "equilibrated_energy_ratio")
    np.testing.assert_allclose(dense.fixed_interface_eigenvalues, exact[:4], rtol=1.0e-6)
    np.testing.assert_allclose(dense.report["zero_energy_stiffness_ratio"], softest, rtol=1.0e-6)
    np.testing.assert_allclose(dense.report["zero_energy_mode_ratio_min"], float(np.min(ratios)), rtol=1.0e-6)

    # Rotary inertia of 1e-12 of the translational mass puts the largest eigenvalue more than 1e13 above
    # the first mode: 1e-12 of it is far above the first modes, which are unchanged and must be kept.
    _, light_mass, _ = _frame_beam(rotary_fraction=1.0e-12)
    light_ii = light_mass[interior][:, interior]
    reference = _lowest_eigenvalues(stiffness_ii.toarray(), light_ii.toarray(), 4)
    largest = float(linalg.eigh(stiffness_ii.toarray(), light_ii.toarray(), eigvals_only=True)[-1])
    test.assertGreater(largest / reference[0], 1.0e13)
    test.assertGreater(1.0e-12 * largest, reference[3])
    for eigensolver in ("arpack", "dense"):
        matrices = (light_mass, stiffness) if eigensolver == "arpack" else (light_mass.toarray(), stiffness.toarray())
        # The reduced mass matrix of this model has a condition number of 2e12, and the rigid-body
        # eigenvalues of the dense reduction reach 2e-7 of the largest one: rigid_mode_tolerance, which
        # reduce() applies as build() does, is raised from its default 1e-8.
        generator = newton.ModalGeneratorCraigBamptonFEM(
            points, *matrices, clamp, fixed_mode_count=4, eigensolver=eigensolver, rigid_mode_tolerance=1.0e-6
        )
        generator.reduce()
        np.testing.assert_allclose(generator.fixed_interface_eigenvalues, reference, rtol=1.0e-8)
        test.assertGreater(generator.report["zero_energy_ratio_min"], 1.0e3 * tolerance)

    # generalized_eigenvalue_scale, the scale of the ModalGeneratorFEM threshold, tightens the bound.
    scale, source = modal_sparse.generalized_eigenvalue_scale(stiffness_ii, mass_ii)
    test.assertEqual(source, "power_iteration")
    test.assertGreaterEqual(scale, exact[-1])
    test.assertLessEqual(scale, 2.0 * exact[-1] * (1.0 + 1.0e-9))
    np.testing.assert_allclose(
        modal_sparse.generalized_eigenvalue_scale(stiffness_ii.toarray(), mass_ii.toarray())[0], scale, rtol=1.0e-6
    )
    # Without a factorable mass matrix the scale falls back to the bound (infinite for a zero mass diagonal).
    singular = sparse.lil_matrix(mass_ii)
    singular[0, 0] = 0.0
    with np.errstate(over="ignore"):
        fallback = modal_sparse.generalized_eigenvalue_scale(stiffness_ii, singular.tocsc())
    test.assertEqual(fallback[1], "gershgorin_bound")

    # A second beam that is not tied to the clamped one is a mechanism and is still rejected.
    second_points, second_mass, second_stiffness = _frame_beam(offset=(0.0, 0.1, 0.0))
    pair = (
        np.vstack((points, second_points)),
        sparse.block_diag((mass, second_mass), format="csc"),
        sparse.block_diag((stiffness, second_stiffness), format="csc"),
    )
    for eigensolver in ("arpack", "dense"):
        matrices = pair[1:] if eigensolver == "arpack" else (pair[1].toarray(), pair[2].toarray())
        generator = newton.ModalGeneratorCraigBamptonFEM(
            pair[0], *matrices, clamp, fixed_mode_count=4, eigensolver=eigensolver
        )
        with test.assertRaisesRegex(ValueError, "zero-energy mechanism"):
            generator.reduce()


def _beam_with_appendix(stiffness_translation=1.0e9, stiffness_rotation=1.0e5, appendix_mass=1.0e-13):
    """Return :func:`_frame_beam` with a stiff, light node on six springs at its tip.

    The extra node coincides with the tip, so the springs preserve rigid-body motion. Its eigenvalues,
    about ``2 k / m``, lie more than 1e13 above the first beam modes, which it leaves unchanged to 1e-9:
    the springs are soft enough that adding them to the tip stiffness does not round it visibly.
    """
    from scipy import sparse

    points, mass, stiffness = _frame_beam()
    node_count = points.shape[0]
    size = 6 * node_count
    tip = 6 * (node_count - 1)
    extended = sparse.lil_matrix((size + 6, size + 6))
    extended[:size, :size] = stiffness
    for component in range(6):
        k = stiffness_translation if component < 3 else stiffness_rotation
        a, b = tip + component, size + component
        extended[a, a] += k
        extended[b, b] += k
        extended[a, b] -= k
        extended[b, a] -= k
    appendix = appendix_mass * np.array([1.0, 1.0, 1.0, 1.0e-4, 1.0e-4, 1.0e-4])
    extended_mass = sparse.diags(np.concatenate((mass.diagonal(), appendix))).tocsc()
    return np.vstack((points, points[-1:])), extended_mass, extended.tocsc()


def test_cb_fem_bad_scaling_is_not_a_mechanism(test, device):
    """Accept positive definite models whose DOF units or stiff parts spread the matrices over 13 decades.

    The pivots of an unequilibrated factor then span more than twelve decades, the limit of the pivot test,
    and the largest eigenvalue lies more than 1e13 above the first mode. Both cases reproduce the
    well-scaled model.
    """
    from scipy import linalg

    points, mass, stiffness = _frame_beam()
    clamp = [{"name": "root", "node_indices": [0], "coupling": "node"}]
    interior = np.arange(6, stiffness.shape[0])
    translations = _translation_rows(points.shape[0])
    reference = newton.ModalGeneratorCraigBamptonFEM(
        points, mass, stiffness, clamp, fixed_mode_count=6, eigensolver="arpack"
    )
    reference.reduce()

    def assert_same(generator, sample_rows, tolerance):
        np.testing.assert_allclose(
            generator.fixed_interface_eigenvalues, reference.fixed_interface_eigenvalues, rtol=tolerance
        )
        for ours, theirs in (
            (generator.reduced_stiffness_matrix, reference.reduced_stiffness_matrix),
            (generator.reduced_mass_matrix, reference.reduced_mass_matrix),
        ):
            scale = float(np.max(np.abs(theirs)))
            np.testing.assert_allclose(ours[:6, :6], theirs[:6, :6], rtol=0.0, atol=tolerance * scale)
            np.testing.assert_allclose(np.diag(ours)[6:], np.diag(theirs)[6:], rtol=tolerance, atol=tolerance)
            # Invariant under the sign of each mode.
            np.testing.assert_allclose(
                ours[:6, 6:] @ ours[6:, :6], theirs[:6, 6:] @ theirs[6:, :6], rtol=0.0, atol=tolerance * scale**2
            )
        ours = generator.full_recovery_matrix()[sample_rows]
        theirs = reference.full_recovery_matrix()[translations]
        np.testing.assert_allclose(ours[:, :6], theirs[:, :6], rtol=0.0, atol=tolerance)
        np.testing.assert_allclose(
            ours[:, 6:] @ ours[:, 6:].T, theirs[:, 6:] @ theirs[:, 6:].T, rtol=0.0, atol=tolerance
        )

    # Rotations of the interior nodes measured in units of 1e5 rad: a congruence of the same model.
    units = _interior_rotation_units(points.shape[0], 1.0e-5)
    scaled_mass, scaled_stiffness = _rescaled(mass, units), _rescaled(stiffness, units)
    diagonal = scaled_stiffness.diagonal()[interior]
    test.assertGreater(float(np.max(diagonal) / np.min(diagonal)), 1.0e13)
    unscaled_factor = modal_sparse.ShiftedFactorization(
        scaled_stiffness[interior][:, interior], scaled_mass[interior][:, interior], 0.0
    )
    test.assertTrue(unscaled_factor.numerically_singular)
    for eigensolver in ("arpack", "dense"):
        matrices = (scaled_mass, scaled_stiffness)
        if eigensolver == "dense":
            matrices = (scaled_mass.toarray(), scaled_stiffness.toarray())
        generator = newton.ModalGeneratorCraigBamptonFEM(
            points, *matrices, clamp, fixed_mode_count=6, eigensolver=eigensolver
        )
        generator.reduce()
        assert_same(generator, translations, 1.0e-8)
        report = generator.report
        test.assertGreater(report["interior_stiffness_scaling_max"] / report["interior_stiffness_scaling_min"], 3.0e6)
        test.assertGreater(report["interior_stiffness_pivot_ratio"], 1.0e-9)
        # The energy ratio is invariant under the units of the DOFs.
        np.testing.assert_allclose(
            report["zero_energy_mode_ratio_min"], reference.report["zero_energy_mode_ratio_min"], rtol=1.0e-6
        )
        if eigensolver == "arpack":
            np.testing.assert_allclose(
                report["zero_energy_ratio_min"], reference.report["zero_energy_ratio_min"], rtol=1.0e-6
            )
        generator.craig_bampton.build()

    # A stiff, light appendix lifts the largest eigenvalue more than 1e13 above the first mode.
    appendix_points, appendix_mass, appendix_stiffness = _beam_with_appendix()
    appendix_interior = np.arange(6, appendix_stiffness.shape[0])
    largest = float(
        linalg.eigh(
            appendix_stiffness[appendix_interior][:, appendix_interior].toarray(),
            appendix_mass[appendix_interior][:, appendix_interior].toarray(),
            eigvals_only=True,
        )[-1]
    )
    test.assertGreater(largest / reference.fixed_interface_eigenvalues[0], 1.0e13)
    for eigensolver in ("arpack", "dense"):
        matrices = (appendix_mass, appendix_stiffness)
        if eigensolver == "dense":
            matrices = (appendix_mass.toarray(), appendix_stiffness.toarray())
        generator = newton.ModalGeneratorCraigBamptonFEM(
            appendix_points, *matrices, clamp, fixed_mode_count=6, eigensolver=eigensolver
        )
        generator.reduce()
        assert_same(generator, translations, 1.0e-8)
        test.assertGreater(
            generator.report["zero_energy_ratio_min"], 1.0e3 * generator.report["zero_energy_ratio_tolerance"]
        )
        generator.craig_bampton.build()


def test_cb_fem_mechanisms_are_rejected_in_any_scaling(test, device):
    """Reject parts without a tie in any DOF units, DOFs without stiffness, ties of roundoff stiffness, and an
    indefinite mass matrix, and accept a soft tie."""
    from scipy import sparse

    points, mass, stiffness = _frame_beam()
    node_count = points.shape[0]
    clamp = [{"name": "root", "node_indices": [0], "coupling": "node"}]

    def reduce(points, mass, stiffness, eigensolver, **options):
        if eigensolver == "dense":
            mass, stiffness = mass.toarray(), stiffness.toarray()
        generator = newton.ModalGeneratorCraigBamptonFEM(
            points, mass, stiffness, clamp, fixed_mode_count=4, eigensolver=eigensolver, **options
        )
        generator.reduce()
        return generator

    # A second beam without a tie, with the rotations of all interior nodes in units of 1e5 rad.
    second_points, second_mass, second_stiffness = _frame_beam(offset=(0.0, 0.1, 0.0))
    units = _interior_rotation_units(2 * node_count, 1.0e-5)
    pair_points = np.vstack((points, second_points))
    pair_mass = _rescaled(sparse.block_diag((mass, second_mass), format="csc"), units)
    pair_stiffness = _rescaled(sparse.block_diag((stiffness, second_stiffness), format="csc"), units)
    for eigensolver in ("arpack", "dense"):
        with test.assertRaisesRegex(ValueError, "zero-energy mechanism"):
            reduce(pair_points, pair_mass, pair_stiffness, eigensolver)

    # A DOF without stiffness: the error names it.
    dof = 6 * 20 + 3
    released = sparse.lil_matrix(stiffness)
    released[dof, :] = 0.0
    released[:, dof] = 0.0
    pattern = rf"zero-energy mechanism.*interior stiffness diagonal of DOF {dof} \(node 20 rx\) is 0"
    for eigensolver in ("arpack", "dense"):
        with test.assertRaisesRegex(ValueError, pattern):
            reduce(points, mass, released.tocsc(), eigensolver)

    # A short beam hangs on the tip of the clamped one through six springs, each a fraction of the tip
    # stiffness: 1e-14 of it is roundoff, so the beam is free, while 1e-6 is a soft but genuine tie.
    hung_points, hung_mass, hung_stiffness = _frame_beam(element_count=8, length=0.2, offset=(1.0, 0.0, 0.0))
    tied_points = np.vstack((points, hung_points))
    tied_mass = sparse.block_diag((mass, hung_mass), format="csc")
    tip, root = 6 * (node_count - 1), 6 * node_count
    for fraction, accepted in ((1.0e-14, False), (1.0e-6, True)):
        tied = sparse.lil_matrix(sparse.block_diag((stiffness, hung_stiffness)))
        for component in range(6):
            k = fraction * stiffness[tip + component, tip + component]
            a, b = tip + component, root + component
            tied[a, a] += k
            tied[b, b] += k
            tied[a, b] -= k
            tied[b, a] -= k
        for eigensolver in ("arpack", "dense"):
            if accepted:
                generator = reduce(tied_points, tied_mass, tied.tocsc(), eigensolver)
                report = generator.report
                test.assertGreater(report["zero_energy_ratio_min"], 10.0 * report["zero_energy_ratio_tolerance"])
                test.assertLess(report["zero_energy_ratio_min"], 1.0e-6)
            else:
                with test.assertRaisesRegex(ValueError, "zero-energy mechanism"):
                    reduce(tied_points, tied_mass, tied.tocsc(), eigensolver)
    # The soft tie falls below a tolerance that the user raises above its energy ratio: the softest
    # direction of the equilibrated stiffness decides on both paths.
    for eigensolver in ("arpack", "dense"):
        with test.assertRaisesRegex(ValueError, "softest direction of the equilibrated interior stiffness"):
            reduce(tied_points, tied_mass, tied.tocsc(), eigensolver, eigenvalue_tolerance=1.0e-5)
    # If the inverse iteration of the sparse path missed that direction, the lowest computed mode decides.
    interior_count = tied.shape[0] - 6
    with mock.patch.object(modal_sparse, "smallest_eigenvalue_estimate", return_value=(1.0, np.zeros(interior_count))):
        with test.assertRaisesRegex(
            ValueError, r"lowest fixed-interface eigenvalue .* Hz is at or below the zero-energy threshold"
        ):
            reduce(tied_points, tied_mass, tied.tocsc(), "arpack", eigenvalue_tolerance=1.0e-5)
    # The roundoff tie leaves non-positive pivots in the equilibrated factor of the sparse path.
    roundoff = sparse.lil_matrix(sparse.block_diag((stiffness, hung_stiffness)))
    for component in range(6):
        k = 1.0e-14 * stiffness[tip + component, tip + component]
        a, b = tip + component, root + component
        roundoff[a, a] += k
        roundoff[b, b] += k
        roundoff[a, b] -= k
        roundoff[b, a] -= k
    with test.assertRaisesRegex(ValueError, r"equilibrated interior stiffness has \d+ non-positive pivots"):
        reduce(tied_points, tied_mass, roundoff.tocsc(), "arpack")
    # The pivot-ratio limit of the factorization is a separate decision with its own message.
    with mock.patch.object(modal_sparse, "SINGULAR_PIVOT_RATIO", 1.0):
        with test.assertRaisesRegex(
            ValueError, "smallest pivot of the equilibrated interior stiffness is .* at or below the limit 1 of"
        ):
            reduce(points, mass, stiffness, "arpack")

    # The interior mass matrix must be positive definite: a zero diagonal and an indefinite coupling raise.
    massless = sparse.lil_matrix(mass)
    massless[dof, dof] = 0.0
    indefinite = sparse.lil_matrix(mass)
    indefinite[dof, dof + 6] = indefinite[dof + 6, dof] = 2.0 * mass[dof, dof]
    for eigensolver in ("arpack", "dense"):
        with test.assertRaisesRegex(ValueError, rf"positive definite on the interior DOFs.*DOF {dof} \(node 20 rx\)"):
            reduce(points, massless.tocsc(), stiffness, eigensolver)
        with test.assertRaisesRegex(ValueError, "mass_matrix must be positive definite on the interior DOFs"):
            reduce(points, indefinite.tocsc(), stiffness, eigensolver)


def _beam_with_hung_node(fraction):
    """Return :func:`_frame_beam` with a node of the tip mass on six springs at its tip.

    Each spring has ``fraction`` of the tip stiffness in its DOF. The whole stiffness of the node is its
    tie, so its equilibrated diagonal is one whatever the fraction is.
    """
    from scipy import sparse

    points, mass, stiffness = _frame_beam()
    size = stiffness.shape[0]
    tip = size - 6
    extended = sparse.lil_matrix((size + 6, size + 6))
    extended[:size, :size] = stiffness
    for component in range(6):
        k = fraction * stiffness[tip + component, tip + component]
        a, b = tip + component, size + component
        extended[a, a] += k
        extended[b, b] += k
        extended[a, b] -= k
        extended[b, a] -= k
    extended_mass = sparse.diags(np.concatenate((mass.diagonal(), mass.diagonal()[tip:]))).tocsc()
    return np.vstack((points, points[-1:])), extended_mass, extended.tocsc()


def test_cb_fem_soft_and_free_nodes(test, device):
    """Refuse a node whose ties are roundoff against what it hangs on, and warn of a soft but resolved one.

    The energy ratio does not see such a node, because its whole stiffness is small. Ties of 1e-12 or less of
    the neighboring stiffness are refused by the relative tie stiffness on both paths; reduce() also refuses
    what build() could not separate from the rigid-body modes. Ties of 1e-8 and 1e-10 give modes at 1.8 Hz
    and 0.18 Hz below the 8.2 Hz of the beam: the model reduces and builds, with a warning.
    """
    from scipy import sparse

    clamp = [{"name": "root", "node_indices": [0], "coupling": "node"}]

    def generator(fraction, eigensolver, **options):
        points, mass, stiffness = _beam_with_hung_node(fraction)
        if eigensolver == "dense":
            mass, stiffness = mass.toarray(), stiffness.toarray()
        return newton.ModalGeneratorCraigBamptonFEM(
            points, mass, stiffness, clamp, fixed_mode_count=8, eigensolver=eigensolver, **options
        )

    for eigensolver in ("arpack", "dense"):
        for fraction, frequency in ((1.0e-8, 1.8), (1.0e-10, 0.18)):
            soft = generator(fraction, eigensolver)
            with test.assertWarnsRegex(
                UserWarning, r"DOF 246 \(node 41 ux\) is tied to the rest of the model with only .* a soft part"
            ) as caught:
                soft.reduce()
            test.assertRegex(str(caught.warning), r"lowest fixed-interface mode lies at .* Hz, a factor")
            np.testing.assert_allclose(soft.fixed_interface_frequencies[0], frequency, rtol=0.1)
            np.testing.assert_allclose(soft.report["zero_energy_tie_ratio_min"], fraction, rtol=1.0e-6)
            test.assertEqual(soft.report["zero_energy_tie_coordinate"], "DOF 246 (node 41 ux)")
            test.assertGreater(soft.report["reduced_elastic_eigenvalue_ratio"], 1.0e-8)
            test.assertEqual(soft.report["reduced_elastic_eigenvalue_limit"], 1.0e-8)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="derive_psi", category=UserWarning)
                soft.craig_bampton.build()
        for fraction in (1.0e-12, 1.0e-13, 1.0e-14):
            with test.assertRaisesRegex(
                ValueError, r"zero-energy mechanism.*node 41 .* is tied to the rest of the model with"
            ):
                generator(fraction, eigensolver).reduce()

    # With the tolerance at its roundoff floor the 1e-13 tie passes the tie test, but the reduced model
    # could not be built: reduce() refuses it, on the sparse path already in the eigensolver.
    with test.assertRaisesRegex(ValueError, "zero-energy mechanism.*cannot be separated from the six rigid-body"):
        generator(1.0e-13, "dense", eigenvalue_tolerance=0.0).reduce()
    with test.assertRaisesRegex(ValueError, "zero-energy mechanism.*(cannot resolve|cannot be separated)"):
        generator(1.0e-13, "arpack", eigenvalue_tolerance=0.0).reduce()

    # A rotational DOF cut out of the model and left with a negligible stiffness of its own is uncoupled,
    # which warns and names it; its mode at 3e-8 Hz is then refused. On the sparse path the reduced
    # stiffness no longer preserves rigid motion; on the dense path the returned mode is not an eigenvector.
    points, mass, stiffness = _frame_beam()
    dof = 6 * 20 + 3
    cut = sparse.lil_matrix(stiffness)
    original = cut[dof, dof]
    cut[dof, :] = 0.0
    cut[:, dof] = 0.0
    cut[dof, dof] = 1.0e-25 * original
    for eigensolver, message in (
        ("arpack", "does not preserve the six rigid-body modes"),
        ("dense", "zero-energy mechanism.*a returned fixed-interface mode is not an eigenvector"),
    ):
        matrices = (mass, cut.tocsc()) if eigensolver == "arpack" else (mass.toarray(), cut.toarray())
        isolated = newton.ModalGeneratorCraigBamptonFEM(
            points, *matrices, clamp, fixed_mode_count=4, eigensolver=eigensolver
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with test.assertRaisesRegex(ValueError, message):
                isolated.reduce()
        test.assertTrue(
            any(
                f"DOF {dof} (node 20 rx)" in str(item.message) and "no coupling" in str(item.message) for item in caught
            )
        )

    # A node that hangs on the interface node only is tied through the interface frame: the frame columns
    # count, so it is neither uncoupled nor weakly tied.
    points, mass, stiffness = _frame_beam()
    size = stiffness.shape[0]
    hung = sparse.lil_matrix((size + 6, size + 6))
    hung[:size, :size] = stiffness
    for component in range(6):
        k = 1.0e-3 * stiffness[component, component]
        hung[component, component] += k
        hung[size + component, size + component] += k
        hung[component, size + component] -= k
        hung[size + component, component] -= k
    hung_mass = sparse.diags(np.concatenate((mass.diagonal(), mass.diagonal()[:6]))).tocsc()
    on_frame = newton.ModalGeneratorCraigBamptonFEM(
        np.vstack((points, points[:1])), hung_mass, hung.tocsc(), clamp, fixed_mode_count=4
    )
    on_frame.reduce()
    test.assertEqual(on_frame.report["uncoupled_interior_count"], 0)
    test.assertGreater(on_frame.report["zero_energy_tie_ratio_min"], 1.0e-4)


def test_cb_fem_node_enclosed_by_a_large_patch(test, device):
    """Keep a node whose only neighbors are the nodes of a large rigid interface patch, without a warning.

    Its coordinates couple to the interface frame only, with their whole stiffness. The frame diagonal
    gathers the stiffness of every patch node, so a tie ratio measured against it would fall with the
    patch size (120 and 360 nodes here) and report a firmly held node as a soft part.
    """
    ratios = {}
    for size in (12, 20):
        points, mass, stiffness = _frame_grid(nx=size, ny=size)
        node = np.arange(size * size).reshape(size, size)
        enclosed = int(node[size // 2, size // 2])
        patch = np.setdiff1d(node[: size - 1, : size - 1].reshape(-1), [enclosed])
        test.assertEqual(patch.shape[0], (size - 1) ** 2 - 1)
        interfaces = [{"name": "patch", "node_indices": patch, "coupling": "rbe2"}]
        for eigensolver in ("arpack", "dense"):
            matrices = (mass, stiffness) if eigensolver == "arpack" else (mass.toarray(), stiffness.toarray())
            generator = newton.ModalGeneratorCraigBamptonFEM(
                points, *matrices, interfaces, fixed_mode_count=4, eigensolver=eigensolver
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                generator.reduce()
            test.assertEqual(generator.report["uncoupled_interior_count"], 0)
            ratios[size, eigensolver] = generator.report["zero_energy_tie_ratio_min"]
            # The enclosed node is held by the frame with its whole stiffness in every DOF.
            interior = generator.interior_count
            stiffness_z = generator.transformation.T @ stiffness @ generator.transformation
            stiffness_z = stiffness_z.toarray() if hasattr(stiffness_z, "toarray") else np.asarray(stiffness_z)
            rows = np.flatnonzero(
                np.abs(
                    np.asarray(generator.transformation[6 * enclosed : 6 * enclosed + 6, :interior].sum(axis=0))
                ).reshape(-1)
                > 0.0
            )
            test.assertEqual(rows.shape[0], 6)
            for row in rows:
                test.assertLess(float(np.sum(np.abs(stiffness_z[row, :interior])) - stiffness_z[row, row]), 1.0e-9)
                support = float(np.sum(np.abs(stiffness_z[row, interior:]))) / stiffness_z[row, row]
                test.assertGreaterEqual(support, 1.0 - 1.0e-9)
    for value in ratios.values():
        test.assertGreater(value, 0.1)
    np.testing.assert_allclose(ratios[20, "arpack"], ratios[12, "arpack"], rtol=1.0e-9)
    np.testing.assert_allclose(ratios[20, "dense"], ratios[12, "dense"], rtol=1.0e-9)


def _beam_with_soft_cluster(fractions, mutual):
    """Return :func:`_frame_beam` with coincident extra nodes on its tip, each on six springs of the given
    fraction of the tip stiffness and joined to the next by springs of ``mutual`` times the tip stiffness."""
    from scipy import sparse

    points, mass, stiffness = _frame_beam()
    size = stiffness.shape[0]
    tip = size - 6
    extra = len(fractions)
    extended = sparse.lil_matrix((size + 6 * extra, size + 6 * extra))
    extended[:size, :size] = stiffness

    def spring(a, b, k):
        extended[a, a] += k
        extended[b, b] += k
        extended[a, b] -= k
        extended[b, a] -= k

    for index, fraction in enumerate(fractions):
        for component in range(6):
            spring(
                tip + component, size + 6 * index + component, fraction * stiffness[tip + component, tip + component]
            )
    for index in range(extra - 1):
        for component in range(6):
            spring(
                size + 6 * index + component,
                size + 6 * (index + 1) + component,
                mutual * stiffness[tip + component, tip + component],
            )
    diagonal = np.concatenate([mass.diagonal()] + [mass.diagonal()[tip:]] * extra)
    return np.vstack([points] + [points[-1:]] * extra), sparse.diags(diagonal).tocsc(), extended.tocsc()


def test_cb_fem_reduce_applies_build_checks(test, device):
    """Refuse in reduce() what build() could not build, and keep uncoupled DOFs with a warning.

    Soft clusters of nodes pass the tie rule and the energy ratio; a stiff, heavy appendix spoils the rigid
    mass block. Each of these reduced and then failed in build(). A lattice whose interior rotations have a
    diagonal stiffness only reduces as before.
    """
    from scipy import sparse

    clamp = [{"name": "root", "node_indices": [0], "coupling": "node"}]
    cases = (
        ("dense", _beam_with_soft_cluster([1.0e-10, 1.0e-10], 1.0e-10), "exactly six global rigid-body modes"),
        ("arpack", _beam_with_soft_cluster([1.0e-14, 1.0e-14], 1.0e-14), "exactly six global rigid-body modes"),
        ("arpack", _beam_with_soft_cluster([1.0e-11] * 3, 1.0e-11), "exactly six global rigid-body modes"),
        ("arpack", _beam_with_appendix(1.0e12, 1.0e8, 1.0e-3), "physical rigid translational block"),
    )
    for index, (eigensolver, (points, mass, stiffness), message) in enumerate(cases):
        matrices = (mass, stiffness) if eigensolver == "arpack" else (mass.toarray(), stiffness.toarray())
        generator = newton.ModalGeneratorCraigBamptonFEM(
            points, *matrices, clamp, fixed_mode_count=8 if index < 3 else 6, eigensolver=eigensolver
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with test.assertRaisesRegex(ValueError, f"The reduced model cannot be built: .*{message}"):
                generator.reduce()
        # The report of the refused model stays available.
        test.assertIn("reduced_elastic_eigenvalue_ratio", generator.report)
        if index == 1:
            # The lowest modes of this cluster are poorly resolved, which is reported as well.
            test.assertTrue(any("relative residual" in str(item.message) for item in caught))

    # Six-DOF nodes on a translational spring lattice; the interior rotations have a diagonal stiffness only.
    points, lattice_mass, lattice_stiffness = _spring_lattice((4, 3, 3), 0.1, 1000.0, 0.01)
    node_count = points.shape[0]
    left = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].min()))
    right = np.flatnonzero(np.isclose(points[:, 0], points[:, 0].max()))
    interfaces = [{"node_indices": left}, {"node_indices": right}]
    interface_nodes = set(np.concatenate((left, right)).tolist())
    translations = _translation_rows(node_count)
    mean_stiffness = float(np.mean(np.diag(lattice_stiffness)))
    mean_mass = float(np.mean(np.diag(lattice_mass)))
    for fraction in (1.0, 1.0e-3, 1.0e-9):
        stiffness = np.zeros((6 * node_count, 6 * node_count))
        mass = np.zeros((6 * node_count, 6 * node_count))
        stiffness[np.ix_(translations, translations)] = lattice_stiffness
        mass[np.ix_(translations, translations)] = lattice_mass
        for node in range(node_count):
            for component in (3, 4, 5):
                mass[6 * node + component, 6 * node + component] = 1.0e-4 * mean_mass
                if node not in interface_nodes:
                    stiffness[6 * node + component, 6 * node + component] = fraction * mean_stiffness * 1.0e-2
        reference = None
        for eigensolver in ("dense", "arpack"):
            matrices = (
                (mass, stiffness) if eigensolver == "dense" else (sparse.csc_matrix(mass), sparse.csc_matrix(stiffness))
            )
            generator = newton.ModalGeneratorCraigBamptonFEM(
                points, *matrices, interfaces, fixed_mode_count=6, eigensolver=eigensolver
            )
            with test.assertWarnsRegex(
                UserWarning, r"\d+ interior coordinates have a stiffness diagonal but no coupling"
            ):
                generator.reduce()
            test.assertGreater(generator.report["uncoupled_interior_count"], 0)
            if reference is None:
                transform = np.asarray(generator.transformation)
                interior = generator.interior_count
                stiffness_ii = (transform.T @ stiffness @ transform)[:interior, :interior]
                mass_ii = (transform.T @ mass @ transform)[:interior, :interior]
                reference = _lowest_eigenvalues(stiffness_ii, mass_ii, 6)
            np.testing.assert_allclose(generator.fixed_interface_eigenvalues, reference, rtol=1.0e-8)


def test_cb_fem_generator_pickles(test, device):
    """Pickle the generator before and after reduce() on both paths, and default the report of an old override."""
    points, mass, stiffness = _frame_grid()
    interfaces = _grid_interfaces("rbe2")
    load = _gravity_load(mass, points.shape[0])
    for eigensolver in ("arpack", "dense"):
        matrices = (mass, stiffness) if eigensolver == "arpack" else (mass.toarray(), stiffness.toarray())
        generator = newton.ModalGeneratorCraigBamptonFEM(
            points, *matrices, interfaces, fixed_mode_count=6, eigensolver=eigensolver, static_correction_loads=load
        )
        fresh = pickle.loads(pickle.dumps(generator))
        generator.reduce()
        # The factorization of the interior stiffness is released with reduce().
        test.assertIsNone(generator._interior_solve)
        restored = pickle.loads(pickle.dumps(generator))
        np.testing.assert_array_equal(restored.reduced_stiffness_matrix, generator.reduced_stiffness_matrix)
        test.assertEqual(restored._describe_interior(0), generator._describe_interior(0))
        fresh.reduce()
        np.testing.assert_array_equal(fresh.reduced_stiffness_matrix, generator.reduced_stiffness_matrix)

    # An interior solve written for the former five-value contract still reduces; the report keeps every key.
    class Overridden(newton.ModalGeneratorCraigBamptonFEM):
        def _interior_sparse(self, *args):
            result = super()._interior_sparse(*args)
            self._zero_energy = {}
            return result

    overridden = Overridden(points, mass, stiffness, interfaces, fixed_mode_count=6)
    overridden.reduce()
    test.assertTrue(math.isnan(overridden.report["zero_energy_scale"]))
    test.assertEqual(overridden.report["zero_energy_scale_source"], "")
    test.assertGreater(overridden.report["zero_energy_threshold"], 0.0)


def test_cb_fem_equilibration_keeps_well_scaled_results(test, device):
    """Reproduce an unequilibrated dense reduction of well-scaled models to 1e-10 on the sparse path."""
    from scipy import linalg, sparse

    lattice_points, lattice_mass, lattice_stiffness = _spring_lattice((4, 3, 3), 0.1, 1000.0, 0.01)
    left = np.flatnonzero(np.isclose(lattice_points[:, 0], lattice_points[:, 0].min()))
    right = np.flatnonzero(np.isclose(lattice_points[:, 0], lattice_points[:, 0].max()))
    grid_points, grid_mass, grid_stiffness = _frame_grid()
    cases = (
        (
            lattice_points,
            sparse.csc_matrix(lattice_mass),
            sparse.csc_matrix(lattice_stiffness),
            [{"node_indices": left}, {"node_indices": right, "coupling": "rbe3"}],
            3,
            7,
        ),
        (grid_points, grid_mass, grid_stiffness, _grid_interfaces("rbe2"), 6, 6),
        (grid_points, grid_mass, grid_stiffness, _grid_interfaces("rbe3"), 6, 6),
    )
    for points, mass, stiffness, interfaces, dofs, count in cases:
        generator = newton.ModalGeneratorCraigBamptonFEM(
            points, mass, stiffness, interfaces, fixed_mode_count=count, dofs_per_node=dofs, eigensolver="arpack"
        )
        generator.reduce()
        transform = generator.transformation
        interior = generator.interior_count
        frames = 6 * len(interfaces)
        stiffness_z = (transform.T @ stiffness @ transform).toarray()
        mass_z = (transform.T @ mass @ transform).toarray()
        constraint_modes = -np.linalg.solve(stiffness_z[:interior, :interior], stiffness_z[:interior, interior:])
        values, vectors = linalg.eigh(stiffness_z[:interior, :interior], mass_z[:interior, :interior])
        test.assertGreater(values[count] - values[count - 1], 1.0e-6 * values[count])
        # Signs and the rotation within repeated eigenvalues as chosen by the generator.
        modes = vectors[:, :count]
        modes = modes @ (modes.T @ mass_z[:interior, :interior] @ generator.fixed_interface_modes)
        basis = np.zeros((interior + frames, frames + count))
        basis[:interior, :frames] = constraint_modes
        basis[:interior, frames:] = modes
        basis[interior:, :frames] = np.eye(frames)
        recovery = (transform @ basis)[(np.arange(points.shape[0])[:, None] * dofs + np.arange(3)).reshape(-1)]
        np.testing.assert_allclose(generator.fixed_interface_eigenvalues, values[:count], rtol=1.0e-10)
        for name, ours, theirs in (
            ("constraint modes", generator.constraint_modes, constraint_modes),
            ("stiffness", generator.reduced_stiffness_matrix, basis.T @ stiffness_z @ basis),
            ("mass", generator.reduced_mass_matrix, basis.T @ mass_z @ basis),
            ("recovery", generator.recovery_matrix, recovery),
        ):
            error = float(np.max(np.abs(ours - theirs))) / float(np.max(np.abs(theirs)))
            test.assertLess(error, 1.0e-10, name)
        frequencies = np.sqrt(values[:count]) / (2.0 * math.pi)
        np.testing.assert_allclose(generator.fixed_interface_frequencies, frequencies, rtol=1.0e-10)


def _frame_grid(nx=8, ny=5, spacing=0.1, rotary_fraction=1.0e-3):
    """Return node positions, lumped mass, and stiffness of a flat six-DOF grillage of frame elements in the xy plane.

    Steel, 10 mm square section, ``nx`` by ``ny`` nodes; a plate-like free-free model whose stiffness preserves
    rigid-body motion exactly.
    """
    from scipy import sparse

    young, shear, density, side = 2.0e11, 8.0e10, 7850.0, 0.01
    area, inertia, torsion = side * side, side**4 / 12.0, 0.1406 * side**4
    a, t, b, s = young * area / spacing, shear * torsion / spacing, young * inertia, spacing
    local = np.zeros((12, 12))
    for i, j, value in ((0, 0, a), (0, 6, -a), (6, 6, a), (3, 3, t), (3, 9, -t), (9, 9, t)):
        local[i, j] += value
    for (v1, r1, v2, r2), sign in (((1, 5, 7, 11), 1.0), ((2, 4, 8, 10), -1.0)):
        k = (b / s**3) * np.array(
            [
                [12.0, sign * 6.0 * s, -12.0, sign * 6.0 * s],
                [sign * 6.0 * s, 4.0 * s * s, -sign * 6.0 * s, 2.0 * s * s],
                [-12.0, -sign * 6.0 * s, 12.0, -sign * 6.0 * s],
                [sign * 6.0 * s, 2.0 * s * s, -sign * 6.0 * s, 4.0 * s * s],
            ]
        )
        index = np.array([v1, r1, v2, r2])
        local[np.ix_(index, index)] += k
    local = np.triu(local) + np.triu(local, 1).T
    # Elements along y: local axes (y, -x, z) in global coordinates.
    rotation = np.kron(np.eye(4), np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    along_y = rotation.T @ local @ rotation
    node = np.arange(nx * ny).reshape(ny, nx)
    points = np.zeros((nx * ny, 3))
    points[:, 0] = np.tile(np.arange(nx) * spacing, ny)
    points[:, 1] = np.repeat(np.arange(ny) * spacing, nx)
    stiffness = sparse.lil_matrix((6 * nx * ny, 6 * nx * ny))
    nodal_mass = np.zeros(nx * ny)
    pairs = [(node[j, i], node[j, i + 1], local) for j in range(ny) for i in range(nx - 1)]
    pairs += [(node[j, i], node[j + 1, i], along_y) for j in range(ny - 1) for i in range(nx)]
    for first, second, element in pairs:
        dofs = np.concatenate((np.arange(6 * first, 6 * first + 6), np.arange(6 * second, 6 * second + 6)))
        stiffness[np.ix_(dofs, dofs)] = stiffness[np.ix_(dofs, dofs)].toarray() + element
        nodal_mass[[first, second]] += 0.5 * density * area * spacing
    diagonal = np.repeat(nodal_mass, 6) * np.tile([1.0, 1.0, 1.0] + [rotary_fraction] * 3, nx * ny)
    return points, sparse.diags(diagonal).tocsc(), stiffness.tocsc()


def _grid_interfaces(coupling, nx=8, ny=5):
    """Two four-node corner patches of :func:`_frame_grid` with the given coupling."""
    node = np.arange(nx * ny).reshape(ny, nx)
    return [
        {"name": "near", "node_indices": node[:2, :2].reshape(-1), "coupling": coupling},
        {"name": "far", "node_indices": node[-2:, -2:].reshape(-1), "coupling": coupling},
    ]


def _rotation_drift(stiffness, relative):
    """Return a diagonal stiffness on the rotational DOFs, ``relative`` times their mean diagonal [N m/rad].

    It resists rigid rotation only, like the rotational rigid-body residual of shells on a non-conformal mesh.
    """
    from scipy import sparse

    diagonal = stiffness.diagonal()
    rotational = np.zeros(diagonal.shape[0], dtype=bool)
    rotational[3::6] = rotational[4::6] = rotational[5::6] = True
    return sparse.diags(np.where(rotational, relative * float(np.mean(diagonal[rotational])), 0.0)).tocsc()


_GRID_DRIFT = 1.0e-6
"""Relative rotational drift stiffness of the :func:`_frame_grid` rigid-consistency tests."""

_GRID_GROUND_SPRING = 1.0e4
"""Translational ground spring [N/m] of the rigid-consistency guard test."""


def _external_rigid_projection(craig_bampton):
    """Project a Craig-Bampton model onto rigid consistency outside the generator, on its public attributes.

    The algebra of an external correction applied between ``reduce()`` and ``build()``, kept
    separate from the generator code as the reference for ``rigid_consistency="project"``.
    """
    rigid = craig_bampton._rigid_map()
    rigid_pinv = np.linalg.pinv(rigid)
    projector = np.eye(craig_bampton.reduced_dof_count) - rigid @ rigid_pinv
    stiffness = projector.T @ craig_bampton.stiffness_matrix @ projector
    stiffness = 0.5 * (stiffness + stiffness.T)
    points = craig_bampton.sample_points
    sample_rigid = np.hstack(
        (np.tile(np.eye(3), (points.shape[0], 1)), np.vstack([-craig_bampton._skew(p) for p in points]))
    )
    recovery = craig_bampton.recovery_matrix + (sample_rigid - craig_bampton.recovery_matrix @ rigid) @ rigid_pinv
    spatial = rigid.T @ craig_bampton.mass_matrix @ rigid
    target = spatial.copy()
    target[:3, :3] = np.trace(spatial[:3, :3]) / 3.0 * np.eye(3)
    moment = 0.5 * (spatial[:3, 3:] - spatial[:3, 3:].T)
    target[:3, 3:] = moment
    target[3:, :3] = moment.T
    mass = craig_bampton.mass_matrix + rigid_pinv.T @ (target - spatial) @ rigid_pinv
    return stiffness, 0.5 * (mass + mass.T), recovery


def test_cb_fem_rigid_consistency_projection(test, device):
    """Remove a small rigid-body residual with rigid_consistency="project", as an external projection would.

    The default check rejects the model in ModalGeneratorCraigBampton.build() (on this stiff grillage at the
    rigid-mode gate, which precedes the recovery gate). The projected generator matches the external projection
    (external correction algebra) to 1e-10, reports the residuals before and after, and its full recovery
    reproduces rigid motion and agrees with the sample recovery.
    """
    points, mass, stiffness = _frame_grid()
    drifting = stiffness + _rotation_drift(stiffness, _GRID_DRIFT)
    for coupling in ("rbe2", "rbe3"):
        interfaces = _grid_interfaces(coupling)
        checked = newton.ModalGeneratorCraigBamptonFEM(
            points, mass, drifting, interfaces, fixed_mode_count=6, sample_mass="lumped"
        )
        with test.assertRaisesRegex(ValueError, "six global rigid-body modes|recovery_matrix is inconsistent"):
            checked.build()
        test.assertIsNone(checked.report["rigid_projection"])
        reference = checked.craig_bampton
        stiffness_ref, mass_ref, recovery_ref = _external_rigid_projection(reference)
        reference.stiffness_matrix, reference.mass_matrix, reference.recovery_matrix = (
            stiffness_ref,
            mass_ref,
            recovery_ref,
        )
        reference.build()

        projected = newton.ModalGeneratorCraigBamptonFEM(
            points, mass, drifting, interfaces, fixed_mode_count=6, sample_mass="lumped", rigid_consistency="project"
        )
        projected.build()
        for name, value, expected in (
            ("stiffness", projected.reduced_stiffness_matrix, stiffness_ref),
            ("mass", projected.reduced_mass_matrix, mass_ref),
            ("recovery", projected.recovery_matrix, recovery_ref),
            ("frequencies", projected.craig_bampton.frequencies, reference.frequencies),
        ):
            error = float(np.max(np.abs(value - expected))) / float(np.max(np.abs(expected)))
            test.assertLess(error, 1.0e-10, f"{coupling} {name}")

        report = projected.report["rigid_projection"]
        test.assertGreater(report["recovery_residual_before"], 1.0e-6)
        test.assertLess(report["recovery_residual_before"], projected.rigid_projection_limit)
        for name in ("stiffness", "recovery", "mass"):
            test.assertLess(report[f"{name}_residual_after"], 1.0e-12, f"{coupling} {name}")
        test.assertLess(report["max_elastic_eigenvalue_shift"], projected.rigid_projection_eigenvalue_tolerance)

        # The full recovery carries the same correction: it reproduces rigid motion exactly and its sample rows
        # are the recovery matrices.
        full = projected.full_recovery_matrix()
        rigid = projected.craig_bampton._rigid_map()
        exact = modal_sparse.rigid_vectors(points, 6, np.zeros(3))
        test.assertLess(float(np.max(np.abs(full @ rigid - exact))), 1.0e-12)
        np.testing.assert_allclose(full[_translation_rows(points.shape[0])], projected.recovery_matrix, atol=1.0e-12)
        np.testing.assert_allclose(
            full[_translation_rows(points.shape[0]) + 3], projected.rotation_recovery_matrix, atol=1.0e-12
        )
        exported = projected.to_arrays(full_recovery=False)
        np.testing.assert_array_equal(exported["stiffness_matrix"], projected.reduced_stiffness_matrix)
        np.testing.assert_array_equal(exported["recovery_matrix"], projected.recovery_matrix)


def test_cb_fem_rigid_consistency_guards(test, device):
    """Refuse to project a residual that is a modeling error rather than discretization drift."""
    from scipy import sparse

    points, mass, stiffness = _frame_grid()
    interfaces = _grid_interfaces("rbe3")
    middle = int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))

    def generator(matrix, **options):
        options.setdefault("rigid_consistency", "project")
        return newton.ModalGeneratorCraigBamptonFEM(points, mass, matrix, interfaces, fixed_mode_count=6, **options)

    # A support left in the stiffness matrix: a translational spring to ground at an interior node.
    grounded = sparse.lil_matrix(stiffness)
    for dof in range(6 * middle, 6 * middle + 3):
        grounded[dof, dof] += _GRID_GROUND_SPRING
    grounded = grounded.tocsc()
    with test.assertRaisesRegex(ValueError, "beyond rigid_projection_limit"):
        generator(grounded).reduce()
    # Below an (unreasonably) relaxed limit the projection is still refused: it would change the elastic
    # eigenvalues or make the mass matrix indefinite.
    with test.assertRaisesRegex(ValueError, "moves elastic eigenvalue|mass matrix indefinite"):
        generator(grounded, rigid_projection_limit=1.0).reduce()
    # Small drift passes the default guards but not a zero eigenvalue tolerance.
    drifting = stiffness + _rotation_drift(stiffness, _GRID_DRIFT)
    generator(drifting).reduce()
    with test.assertRaisesRegex(ValueError, "moves elastic eigenvalue"):
        generator(drifting, rigid_projection_eigenvalue_tolerance=0.0).reduce()
    # Constraints that ground the body are rejected in either mode.
    spc = np.zeros((1, stiffness.shape[0]))
    spc[0, 6 * middle] = 1.0
    with test.assertRaisesRegex(ValueError, "constraints do not admit rigid-body motion"):
        generator(stiffness, constraints=spc).reduce()
    with test.assertRaisesRegex(ValueError, "rigid_consistency must be 'check' or 'project'"):
        generator(stiffness, rigid_consistency="ignore")
    with test.assertRaisesRegex(ValueError, "rigid_projection_limit must be non-negative"):
        generator(stiffness, rigid_projection_limit=-1.0)
    with test.assertRaisesRegex(ValueError, "rigid_projection_eigenvalue_tolerance must be non-negative"):
        generator(stiffness, rigid_projection_eigenvalue_tolerance=float("nan"))
    with test.assertRaisesRegex(ValueError, "rigid_recovery_tolerance must be non-negative"):
        generator(stiffness, rigid_recovery_tolerance=-1.0).reduce()


def test_cb_fem_rigid_consistency_defaults(test, device):
    """Keep the default behavior: check mode, the former tolerances, and no change to consistent models."""
    fem = inspect.signature(newton.ModalGeneratorCraigBamptonFEM).parameters
    test.assertEqual(fem["rigid_consistency"].default, "check")
    test.assertEqual(fem["rigid_recovery_tolerance"].default, 1.0e-6)
    test.assertEqual(fem["rigid_mode_tolerance"].default, 1.0e-8)
    craig_bampton = inspect.signature(newton.ModalGeneratorCraigBampton).parameters
    test.assertEqual(craig_bampton["rigid_recovery_tolerance"].default, 1.0e-6)

    points, mass, stiffness = _frame_grid()
    for coupling in ("rbe2", "rbe3"):
        interfaces = _grid_interfaces(coupling)
        default = newton.ModalGeneratorCraigBamptonFEM(points, mass, stiffness, interfaces, fixed_mode_count=6)
        default.build()
        test.assertEqual(default.report["rigid_consistency"], "check")
        test.assertIsNone(default.report["rigid_projection"])
        # On a model that preserves rigid motion the projection changes nothing beyond roundoff.
        projected = newton.ModalGeneratorCraigBamptonFEM(
            points, mass, stiffness, interfaces, fixed_mode_count=6, rigid_consistency="project"
        )
        projected.build()
        for value, expected in (
            (projected.reduced_stiffness_matrix, default.reduced_stiffness_matrix),
            (projected.reduced_mass_matrix, default.reduced_mass_matrix),
            (projected.recovery_matrix, default.recovery_matrix),
            (projected.craig_bampton.frequencies, default.craig_bampton.frequencies),
        ):
            test.assertLess(float(np.max(np.abs(value - expected))) / float(np.max(np.abs(expected))), 1.0e-9)
        test.assertLess(projected.report["rigid_projection"]["max_elastic_eigenvalue_shift"], 1.0e-9)

    # The recovery tolerance of the Craig-Bampton build, formerly fixed at 1e-6, is now a parameter with the same
    # default: a drift whose recovery error (about 1e-7) passes the default fails a tighter tolerance.
    drifting = stiffness + _rotation_drift(stiffness, 1.0e-8)
    interfaces = _grid_interfaces("rbe2")
    newton.ModalGeneratorCraigBamptonFEM(points, mass, drifting, interfaces, fixed_mode_count=6).build()
    tight = newton.ModalGeneratorCraigBamptonFEM(
        points, mass, drifting, interfaces, fixed_mode_count=6, rigid_recovery_tolerance=1.0e-8
    )
    with test.assertRaisesRegex(ValueError, "recovery_matrix is inconsistent.*rigid_recovery_tolerance 1e-08"):
        tight.build()


def _full_fe_static(generator, stiffness, load, fixed_interfaces):
    """Solve the full constrained finite-element static problem with the listed interface frames fixed.

    Uses the coordinates of the generator (constraints and interface couplings eliminated) without any
    modal truncation; returns the full DOF field.
    """
    from scipy import sparse
    from scipy.sparse import linalg as sparse_linalg

    transform = sparse.csc_matrix(generator.transformation)
    stiffness_z = sparse.csc_matrix(transform.T @ sparse.csc_matrix(stiffness) @ transform)
    load_z = transform.T @ load
    frames = np.arange(generator.interior_count, transform.shape[1])
    fixed_z = frames[(np.asarray(fixed_interfaces)[:, None] * 6 + np.arange(6)).reshape(-1)]
    free_z = np.setdiff1d(np.arange(transform.shape[1]), fixed_z)
    coordinates = np.zeros(transform.shape[1])
    coordinates[free_z] = sparse_linalg.splu(sparse.csc_matrix(stiffness_z[free_z][:, free_z])).solve(load_z[free_z])
    return transform @ coordinates


def test_cb_fem_static_correction_reproduces_statics(test, device):
    """Reproduce the full finite-element statics of the given loads exactly with static correction vectors.

    With every interface fixed and with one released, at 4, 8, and 16 fixed-interface modes, the reduced
    gravity response equals the full one to 1e-9 with the correction vector. Without it the truncation
    error exceeds 1e-3 of the field with every interface fixed; with one released it is smaller relative
    to the larger field (above 1e-5), because the constraint modes carry most of that response. Three
    gravity axes make every gravity direction exact.
    """
    points, mass, stiffness = _frame_grid()
    node_count = points.shape[0]
    interfaces = _grid_interfaces("rbe2")
    frames = 6 * len(interfaces)
    load = _gravity_load(mass, node_count)
    translations = _translation_rows(node_count)

    def static_errors(generator, nodal_load):
        errors = []
        for fixed in ([0, 1], [0]):
            full = _full_fe_static(generator, stiffness, nodal_load, fixed)
            reduced = _reduced_static(generator, nodal_load, fixed)
            errors.append(_relative((reduced - full)[translations], full[translations]))
        return errors

    for count in (4, 8, 16):
        plain = newton.ModalGeneratorCraigBamptonFEM(points, mass, stiffness, interfaces, fixed_mode_count=count)
        plain.reduce()
        test.assertEqual(plain.static_correction_count, 0)
        test.assertEqual(plain.report["static_correction_load_count"], 0)
        test.assertNotIn("static_correction_frequencies_hz", plain.to_arrays(full_recovery=False))
        truncated = static_errors(plain, load)
        test.assertGreater(truncated[0], 1.0e-3)
        test.assertGreater(truncated[1], 1.0e-5)

        corrected = newton.ModalGeneratorCraigBamptonFEM(
            points, mass, stiffness, interfaces, fixed_mode_count=count, static_correction_loads=load
        )
        basis = corrected.build()
        for error in static_errors(corrected, load):
            test.assertLess(error, 1.0e-9)
        test.assertEqual(basis.mode_count, frames + count + 1 - 6)
        report = corrected.report
        test.assertEqual(report["reduced_dof_count"], frames + count + 1)
        test.assertEqual(report["static_correction_load_count"], 1)
        test.assertEqual(report["static_correction_count"], 1)
        test.assertEqual(report["static_correction_load_indices"], [0])
        test.assertEqual(report["static_correction_dropped"], [])
        test.assertEqual(len(report["fixed_interface_frequencies_hz"]), count)
        test.assertEqual(corrected.fixed_interface_modes.shape[1], count)
        test.assertEqual(corrected.static_correction_vectors.shape, (corrected.interior_count, 1))
        test.assertLess(report["eigen_residual"], 1.0e-8)
        # The fixed-interface modes and their block of the reduced matrices are those of the plain model.
        np.testing.assert_allclose(
            corrected.fixed_interface_eigenvalues, plain.fixed_interface_eigenvalues, rtol=1.0e-12
        )
        for ours, theirs in (
            (corrected.reduced_stiffness_matrix, plain.reduced_stiffness_matrix),
            (corrected.reduced_mass_matrix, plain.reduced_mass_matrix),
        ):
            scale = float(np.max(np.abs(theirs)))
            np.testing.assert_allclose(ours[:-1, :-1], theirs, rtol=0.0, atol=1.0e-12 * scale)
        # The vector is stiffness-orthogonal to the constraint modes and the fixed-interface modes,
        # mass-orthogonal to the modes, of unit modal mass, and stiffer than every retained mode.
        reduced_stiffness, reduced_mass = corrected.reduced_stiffness_matrix, corrected.reduced_mass_matrix
        test.assertLess(float(np.max(np.abs(reduced_stiffness[-1, :-1]))), 1.0e-9 * reduced_stiffness[-1, -1])
        test.assertLess(float(np.max(np.abs(reduced_mass[-1, frames:-1]))), 1.0e-12)
        test.assertAlmostEqual(reduced_mass[-1, -1], 1.0, delta=1.0e-12)
        frequency = corrected.static_correction_frequencies[0]
        test.assertGreater(frequency, corrected.fixed_interface_frequencies[-1])
        np.testing.assert_allclose(reduced_stiffness[-1, -1], (2.0 * math.pi * frequency) ** 2, rtol=1.0e-12)
        test.assertEqual(report["static_correction_frequencies_hz"], [frequency])
        arrays = corrected.to_arrays(full_recovery=False)
        np.testing.assert_array_equal(arrays["static_correction_frequencies_hz"], [frequency])
        np.testing.assert_array_equal(arrays["static_correction_load_indices"], [0])
        test.assertEqual(arrays["fixed_interface_frequencies_hz"].shape, (count,))
        test.assertEqual(arrays["mass_matrix"].shape, (frames + count + 1, frames + count + 1))
        test.assertEqual(arrays["recovery_matrix"].shape, (3 * node_count, frames + count + 1))

    # The dense and the sparse path agree.
    sparse_path = newton.ModalGeneratorCraigBamptonFEM(
        points, mass, stiffness, interfaces, fixed_mode_count=8, static_correction_loads=load, eigensolver="arpack"
    )
    sparse_path.reduce()
    dense_path = newton.ModalGeneratorCraigBamptonFEM(
        points,
        mass.toarray(),
        stiffness.toarray(),
        interfaces,
        fixed_mode_count=8,
        static_correction_loads=load,
        eigensolver="dense",
    )
    dense_path.reduce()
    np.testing.assert_allclose(
        dense_path.static_correction_frequencies, sparse_path.static_correction_frequencies, rtol=1.0e-8
    )
    for error in static_errors(dense_path, load):
        test.assertLess(error, 1.0e-9)
    for fixed in ([0, 1], [0]):
        ours, theirs = _reduced_static(dense_path, load, fixed), _reduced_static(sparse_path, load, fixed)
        test.assertLess(_relative((ours - theirs)[translations], theirs[translations]), 1.0e-9)

    # Three gravity axes make every gravity direction exact.
    axes = np.zeros((6 * node_count, 3))
    for axis in range(3):
        acceleration = np.zeros(6 * node_count)
        acceleration[axis::6] = 1.0
        axes[:, axis] = mass @ acceleration
    three = newton.ModalGeneratorCraigBamptonFEM(
        points, mass, stiffness, interfaces, fixed_mode_count=4, static_correction_loads=axes
    )
    basis = three.build()
    test.assertEqual(three.static_correction_count, 3)
    test.assertEqual(basis.mode_count, frames + 4 + 3 - 6)
    np.testing.assert_allclose(three.reduced_mass_matrix[frames + 4 :, frames + 4 :], np.eye(3), rtol=0.0, atol=1.0e-12)
    for direction in ((0.3, -0.5, 0.8), (1.0, 0.0, 0.0), (-0.6, 0.7, 0.1)):
        for error in static_errors(three, axes @ np.asarray(direction)):
            test.assertLess(error, 1.0e-9)


def test_cb_fem_static_correction_span_and_options(test, device):
    """Drop loads that the basis already represents, keep the model unchanged without loads, and validate input."""
    points, mass, stiffness = _frame_grid()
    node_count = points.shape[0]
    interfaces = _grid_interfaces("rbe2")
    frames = 6 * len(interfaces)
    load = _gravity_load(mass, node_count)
    options = {"fixed_mode_count": 6}
    plain = newton.ModalGeneratorCraigBamptonFEM(points, mass, stiffness, interfaces, **options)
    plain.reduce()

    # The interior coordinates of RBE2 interfaces without constraints are nodal DOFs, so M_ii phi_j, placed
    # on them, is a nodal load whose fixed-interface response is phi_j / lambda_j: already in the span.
    interior = plain.interior_count
    selection = plain.transformation[:, :interior]
    np.testing.assert_array_equal((selection.T @ selection).toarray(), np.eye(interior))
    mass_ii = (plain.transformation.T @ mass @ plain.transformation)[:interior, :interior]
    in_span = selection @ (mass_ii @ plain.fixed_interface_modes[:, 2])
    loads = np.column_stack((in_span, load, -2.5 * load, np.zeros(6 * node_count)))
    generator = newton.ModalGeneratorCraigBamptonFEM(
        points, mass, stiffness, interfaces, static_correction_loads=loads, **options
    )
    # A load without a part on the interior coordinates is reported with a warning.
    with test.assertWarnsRegex(UserWarning, "static_correction_loads column 3 has no load on the interior"):
        generator.reduce()
    basis = generator.craig_bampton.build()
    report = generator.report
    test.assertEqual(report["static_correction_load_count"], 4)
    test.assertEqual(report["static_correction_count"], 1)
    test.assertEqual(report["static_correction_load_indices"], [1])
    test.assertEqual(report["static_correction_dropped"], [0, 2, 3])
    norms = report["static_correction_relative_norms"]
    test.assertLess(norms[0], 1.0e-10)
    test.assertGreater(norms[1], 1.0e-3)
    test.assertLess(norms[2], 1.0e-10)
    test.assertEqual(norms[3], 0.0)
    test.assertEqual(basis.mode_count, frames + 6 + 1 - 6)
    np.testing.assert_array_equal(generator.static_correction_load_indices, [1])
    # The same vector as for the gravity load alone.
    single = newton.ModalGeneratorCraigBamptonFEM(
        points, mass, stiffness, interfaces, static_correction_loads=load, **options
    )
    single.reduce()
    np.testing.assert_allclose(
        generator.static_correction_vectors, single.static_correction_vectors, rtol=0.0, atol=1.0e-12
    )

    # No loads, or loads that are all dropped, leave the reduced model as it is without the option.
    for empty in (np.zeros((6 * node_count, 0)), in_span):
        generator = newton.ModalGeneratorCraigBamptonFEM(
            points, mass, stiffness, interfaces, static_correction_loads=empty, **options
        )
        generator.reduce()
        test.assertEqual(generator.static_correction_count, 0)
        np.testing.assert_array_equal(generator.reduced_stiffness_matrix, plain.reduced_stiffness_matrix)
        np.testing.assert_array_equal(generator.reduced_mass_matrix, plain.reduced_mass_matrix)
        np.testing.assert_array_equal(generator.recovery_matrix, plain.recovery_matrix)
        test.assertEqual(generator.to_arrays(full_recovery=False)["static_correction_frequencies_hz"].shape, (0,))

    # With rigid_consistency="project" the correction coordinate keeps its stiffness: the rigid motion of
    # the interface frames does not involve it.
    drifting = stiffness + _rotation_drift(stiffness, _GRID_DRIFT)
    projected = newton.ModalGeneratorCraigBamptonFEM(
        points, mass, drifting, interfaces, static_correction_loads=load, rigid_consistency="project", **options
    )
    basis = projected.build()
    test.assertEqual(basis.mode_count, frames + 6 + 1 - 6)
    np.testing.assert_allclose(
        projected.reduced_stiffness_matrix[-1, -1],
        (2.0 * math.pi * projected.static_correction_frequencies[0]) ** 2,
        rtol=1.0e-9,
    )

    # An interior solve overridden without the factorization closure still gets its correction vectors.
    class Overridden(newton.ModalGeneratorCraigBamptonFEM):
        def _interior_sparse(self, *args):
            result = super()._interior_sparse(*args)
            self._interior_solve = None
            return result

    overridden = Overridden(points, mass, stiffness, interfaces, static_correction_loads=load, **options)
    overridden.reduce()
    np.testing.assert_allclose(
        overridden.static_correction_vectors, single.static_correction_vectors, rtol=0.0, atol=1.0e-9
    )

    with test.assertRaisesRegex(
        ValueError, rf"static_correction_loads must have shape \[{6 * node_count}, load_count\]"
    ):
        newton.ModalGeneratorCraigBamptonFEM(
            points, mass, stiffness, interfaces, static_correction_loads=np.zeros((3, 6 * node_count)), **options
        )
    with test.assertRaisesRegex(ValueError, "static_correction_loads contains non-finite"):
        newton.ModalGeneratorCraigBamptonFEM(
            points, mass, stiffness, interfaces, static_correction_loads=np.full(6 * node_count, np.nan), **options
        )


def test_cb_fem_static_correction_conditions(test, device):
    """Keep the static correction exact with constraints, distributing interfaces, sprung frames, and no modes.

    Also: a load close to an earlier one needs the second orthogonalization pass, and the built modal basis
    spans the same static solution.
    """
    from scipy import sparse

    points, mass, stiffness = _frame_grid()
    node_count = points.shape[0]
    translations = _translation_rows(node_count)
    load = _gravity_load(mass, node_count)

    def relative_error(generator, nodal_load, fixed, matrix=stiffness):
        full = _full_fe_static(generator, matrix, nodal_load, fixed)
        reduced = _reduced_static(generator, nodal_load, fixed)
        rows = _translation_rows(generator.node_count)
        return _relative((reduced - full)[rows], full[rows])

    # Distributing interfaces and no fixed-interface modes at all.
    for coupling, count in (("rbe3", 4), ("rbe2", 0), ("rbe3", 0)):
        generator = newton.ModalGeneratorCraigBamptonFEM(
            points,
            mass,
            stiffness,
            _grid_interfaces(coupling),
            fixed_mode_count=count,
            static_correction_loads=load,
        )
        generator.reduce()
        test.assertEqual(generator.static_correction_count, 1)
        for fixed in ([0, 1], [0]):
            test.assertLess(relative_error(generator, load, fixed), 1.0e-9, f"{coupling} {count} {fixed}")

    interfaces = _grid_interfaces("rbe2")
    frames = 6 * len(interfaces)
    generator = newton.ModalGeneratorCraigBamptonFEM(
        points, mass, stiffness, interfaces, fixed_mode_count=4, static_correction_loads=load
    )
    craig_bampton = generator.reduce()

    # Every interface frame on springs instead of held.
    transform = generator.transformation
    interior = generator.interior_count
    stiffness_z = (transform.T @ stiffness @ transform).toarray()
    springs = 1.0e5 * np.eye(frames)
    stiffness_z[interior:, interior:] += springs
    full = transform @ np.linalg.solve(stiffness_z, transform.T @ load)
    recovery = generator.full_recovery_matrix()
    sprung = generator.reduced_stiffness_matrix.copy()
    sprung[:frames, :frames] += springs
    reduced = recovery @ np.linalg.solve(sprung, recovery.T @ load)
    test.assertLess(_relative((reduced - full)[translations], full[translations]), 1.0e-9)

    # The built basis spans the same solution: solve in rigid and modal coordinates with one frame held.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="derive_psi", category=UserWarning)
        basis = craig_bampton.build()
    test.assertEqual(basis.mode_count, frames + 4 + 1 - 6)
    coordinates = np.hstack((craig_bampton.rigid_matrix, craig_bampton.modal_matrix))
    modal_stiffness = coordinates.T @ generator.reduced_stiffness_matrix @ coordinates
    held = coordinates[:6]
    saddle = np.block([[modal_stiffness, held.T], [held, np.zeros((6, 6))]])
    rhs = np.concatenate((coordinates.T @ (recovery.T @ load), np.zeros(6)))
    built = recovery @ (coordinates @ np.linalg.solve(saddle, rhs)[: coordinates.shape[1]])
    full = _full_fe_static(generator, stiffness, load, [0])
    test.assertLess(_relative((built - full)[translations], full[translations]), 1.0e-8)

    # A second load that differs from the first by 1e-7 leaves a response almost inside the span. One
    # orthogonalization pass would leave it 1e-9 away from mass-orthogonality; two reach roundoff.
    other = np.zeros(6 * node_count)
    other[0::6] = 1.0
    close = np.column_stack(
        (load, load + 1.0e-7 * np.linalg.norm(load) / np.linalg.norm(mass @ other) * (mass @ other))
    )
    generator = newton.ModalGeneratorCraigBamptonFEM(
        points, mass, stiffness, interfaces, fixed_mode_count=4, static_correction_loads=close
    )
    generator.reduce()
    test.assertEqual(generator.static_correction_count, 2)
    test.assertLess(generator.report["static_correction_relative_norms"][1], 1.0e-5)
    test.assertLess(generator.report["static_correction_orthogonality"], 1.0e-12)
    for column in range(2):
        for fixed in ([0, 1], [0]):
            test.assertLess(relative_error(generator, close[:, column], fixed), 1.0e-8)

    # Constraints: a coincident node carries a mass and is tied to an interior node. The load is given in
    # the original DOF numbering, on the dependent node as well; the reference solves the constrained
    # problem with Lagrange multipliers.
    host = int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    tied_points = np.vstack((points, points[host : host + 1]))
    size = 6 * node_count
    tied_stiffness = sparse.block_diag((stiffness, sparse.csc_matrix((6, 6))), format="csc")
    tied_mass = sparse.block_diag((mass, 5.0 * mass[6 * host : 6 * host + 6, 6 * host : 6 * host + 6]), format="csc")
    ties = _tie_rows([node_count], [host], node_count + 1)
    tied_load = _gravity_load(tied_mass, node_count + 1)
    test.assertGreater(float(np.abs(tied_load[size:]).max()), 0.0)
    generator = newton.ModalGeneratorCraigBamptonFEM(
        tied_points,
        tied_mass,
        tied_stiffness,
        interfaces,
        fixed_mode_count=4,
        constraints=ties,
        static_correction_loads=tied_load,
    )
    generator.reduce()
    test.assertEqual(generator.static_correction_count, 1)
    patch_nodes = np.concatenate([np.asarray(interface["node_indices"]) for interface in interfaces])
    clamped = (patch_nodes[:, None] * 6 + np.arange(6)).reshape(-1)
    free = np.setdiff1d(np.arange(size + 6), clamped)
    constraint_free = ties.toarray()[:, free]
    saddle = np.block(
        [
            [tied_stiffness.toarray()[np.ix_(free, free)], constraint_free.T],
            [constraint_free, np.zeros((ties.shape[0], ties.shape[0]))],
        ]
    )
    reference = np.zeros(size + 6)
    reference[free] = np.linalg.solve(saddle, np.concatenate((tied_load[free], np.zeros(ties.shape[0]))))[
        : free.shape[0]
    ]
    reduced = _reduced_static(generator, tied_load, [0, 1])
    rows = _translation_rows(node_count + 1)
    test.assertLess(_relative((reduced - reference)[rows], reference[rows]), 1.0e-9)
    plain = newton.ModalGeneratorCraigBamptonFEM(
        tied_points, tied_mass, tied_stiffness, interfaces, fixed_mode_count=4, constraints=ties
    )
    plain.reduce()
    truncated = _reduced_static(plain, tied_load, [0, 1])
    test.assertGreater(_relative((truncated - reference)[rows], reference[rows]), 1.0e-3)


class TestModalCraigBamptonFem(unittest.TestCase):
    pass


_TESTS = (
    test_cb_fem_panel_reference,
    test_constraint_elimination_two_panel_tie,
    test_cb_fem_two_panel_rbe2_statics,
    test_cb_fem_lattice_rbe3_and_rbe2_match_hand_built,
    test_cb_fem_transformation_matches_constraints,
    test_constraint_elimination_general_path,
    test_cb_fem_error_paths,
    test_fe_matrices_must_be_symmetric,
    test_cb_fem_constraint_errors_name_interfaces,
    test_transformation_rank_checked,
    test_cb_fem_reduce_rejects_non_rigid_constraints,
    test_cb_fem_small_rotary_inertia_is_not_a_mechanism,
    test_cb_fem_bad_scaling_is_not_a_mechanism,
    test_cb_fem_mechanisms_are_rejected_in_any_scaling,
    test_cb_fem_soft_and_free_nodes,
    test_cb_fem_reduce_applies_build_checks,
    test_cb_fem_node_enclosed_by_a_large_patch,
    test_cb_fem_generator_pickles,
    test_cb_fem_equilibration_keeps_well_scaled_results,
    test_cb_fem_static_correction_reproduces_statics,
    test_cb_fem_static_correction_span_and_options,
    test_cb_fem_static_correction_conditions,
    test_cb_fem_rigid_consistency_projection,
    test_cb_fem_rigid_consistency_guards,
    test_cb_fem_rigid_consistency_defaults,
    test_cb_fem_large_tiled_model,
    test_load_sparse_matrix,
)

add_function_test(
    TestModalCraigBamptonFem, "test_cb_fem_dense_path_without_scipy", test_cb_fem_dense_path_without_scipy, devices=None
)
if HAS_SCIPY:
    for _test in _TESTS:
        add_function_test(TestModalCraigBamptonFem, _test.__name__, _test, devices=None)
else:
    for _test in _TESTS:
        setattr(
            TestModalCraigBamptonFem,
            _test.__name__,
            unittest.skip("SciPy is required for sparse FEM matrices")(lambda self: None),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
