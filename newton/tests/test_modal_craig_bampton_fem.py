# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``ConstraintElimination``, ``ModalGeneratorCraigBamptonFEM``, and ``load_sparse_matrix``.

These are setup-time NumPy/SciPy tests; they do not launch Warp kernels and
are registered once, independent of the test device. The HMG panel tests use
the finite-element matrices of the HMG FE/ROM example when its folder is
available (set ``NEWTON_HMG_PANEL_DIR`` to the folder that contains
``model/panel`` and ``rom/panel``) and are skipped otherwise.
"""

import importlib.util
import json
import os
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
from newton.tests.test_reduced_elastic_body import _lattice_craig_bampton, _spring_lattice
from newton.tests.unittest_utils import add_function_test

HAS_SCIPY = importlib.util.find_spec("scipy") is not None

HMG_DIR = Path(os.environ.get("NEWTON_HMG_PANEL_DIR", "")).expanduser()
HMG_FILES = (
    "model/panel/M.npz",
    "model/panel/K.npz",
    "model/panel/geometry.npz",
    "rom/panel/cb_48.npz",
    "rom/panel/reduction_report.json",
    "rom/panel/static_four_clamps_48.npz",
    "rom/panel/static_two_clamps_48.npz",
)
HAS_HMG = (
    HAS_SCIPY and bool(os.environ.get("NEWTON_HMG_PANEL_DIR")) and all((HMG_DIR / name).is_file() for name in HMG_FILES)
)
HMG_INTERFACE_NAMES = ("left_front", "left_back", "right_front", "right_back")


def _skip_without_hmg(test):
    if not HAS_HMG:
        test.skipTest("HMG panel data not available; set NEWTON_HMG_PANEL_DIR to the HMG FE/ROM example folder")


def _load_panel():
    """Return the HMG panel node positions [m], CSC mass [kg], CSC stiffness [N/m], and geometry arrays."""
    from scipy import sparse

    with np.load(HMG_DIR / "model/panel/geometry.npz") as data:
        geometry = dict(data)
    mass = sparse.load_npz(HMG_DIR / "model/panel/M.npz")
    stiffness = sparse.load_npz(HMG_DIR / "model/panel/K.npz")
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
    """Tile the HMG panel ``nx`` by ``ny`` times and tie coincident edge nodes.

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


def test_cb_fem_hmg_panel_reference(test, device):
    """Reproduce the stored 48-mode Craig-Bampton reduction of the HMG panel from its full FE matrices.

    The stored reference uses four single-node, six-DOF interfaces at the panel
    corners and fixed-interface modes from SciPy's ``eigsh``. Fixed-interface
    frequencies, interface blocks, the reduced spectrum, the rigid mass, and
    the four-clamp gravity statics must agree to ``1e-6``.
    """
    _skip_without_hmg(test)
    from scipy import linalg

    points, mass, stiffness, geometry = _load_panel()
    report = json.loads((HMG_DIR / "rom/panel/reduction_report.json").read_text())
    with np.load(HMG_DIR / "rom/panel/cb_48.npz") as data:
        stored = dict(data)
    interfaces = [
        {"name": name, "node_indices": [int(node)], "coupling": "node"}
        for name, node in zip(HMG_INTERFACE_NAMES, geometry["interfaces"], strict=True)
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
    test.assertEqual(basis.frame_labels, list(HMG_INTERFACE_NAMES))

    load = _gravity_load(mass, points.shape[0])
    translations = _translation_rows(points.shape[0])
    with np.load(HMG_DIR / "rom/panel/static_four_clamps_48.npz") as data:
        static = dict(data)
    displacement = _reduced_static(generator, load, [0, 1, 2, 3])
    test.assertLess(_relative((displacement - static["rom"])[translations], static["rom"][translations]), 1.0e-6)
    field_error = _relative((displacement - static["full"])[translations], static["full"][translations])
    four_clamps = next(case for case in report["cases"] if case["case"] == "four_clamps")
    test.assertAlmostEqual(field_error, four_clamps["displacement_relative_l2"], delta=1.0e-6)
    test.assertAlmostEqual(field_error, 0.0069, delta=0.0001)
    with np.load(HMG_DIR / "rom/panel/static_two_clamps_48.npz") as data:
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
    """Return two HMG panels side by side in ``x`` with their 17 shared edge nodes tied (102 tie rows)."""
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
    """Tie two HMG panels along a shared edge and match the Lagrange-multiplier (KKT) statics.

    The 102 tie rows take the Boolean fast path: every row of ``T`` has a
    single unit entry. The tied model keeps exactly six rigid-body modes, and
    its static gravity response with one edge clamped equals the solution of
    the original constrained system with Lagrange multipliers.
    """
    _skip_without_hmg(test)
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
    _skip_without_hmg(test)
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


def test_cb_fem_large_tiled_model(test, device):
    """Reduce a 59,850-DOF tiling of 5x5 HMG panels with 8 RBE2 interfaces and 30 modes within 120 s on CPU."""
    _skip_without_hmg(test)
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


class TestModalCraigBamptonFem(unittest.TestCase):
    pass


_TESTS = (
    test_cb_fem_hmg_panel_reference,
    test_constraint_elimination_two_panel_tie,
    test_cb_fem_two_panel_rbe2_statics,
    test_cb_fem_lattice_rbe3_and_rbe2_match_hand_built,
    test_cb_fem_transformation_matches_constraints,
    test_constraint_elimination_general_path,
    test_cb_fem_error_paths,
    test_fe_matrices_must_be_symmetric,
    test_cb_fem_constraint_errors_name_interfaces,
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
