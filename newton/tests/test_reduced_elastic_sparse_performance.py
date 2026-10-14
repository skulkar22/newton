# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Modal factorization, fallbacks, and validation of reduced elastic bodies in the VBD block-sparse solve."""

import unittest

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()

_SPARSE = "block_sparse_joints"
_DT = 1.0 / 480.0


def _beam_basis(mode_count: int):
    """Build a beam basis with ``mode_count`` bending, torsion, and axial modes of rising order."""
    beam = newton.ModalGeneratorBeam
    kinds = (
        (beam.Mode.BENDING_Y, beam.Boundary.PINNED_PINNED),
        (beam.Mode.BENDING_Z, beam.Boundary.PINNED_PINNED),
        (beam.Mode.TORSION, beam.Boundary.FIXED_FREE),
        (beam.Mode.AXIAL, beam.Boundary.FIXED_FREE),
    )
    specs = []
    order = 1
    while len(specs) < mode_count:
        for mode_type, boundary in kinds:
            if len(specs) < mode_count:
                specs.append({"type": mode_type, "boundary": boundary, "order": order})
        order += 1
    return beam(
        length=0.6,
        half_width_y=0.03,
        half_width_z=0.02,
        young_modulus=2.0e6,
        density=1000.0,
        mode_specs=specs,
        sample_count=max(64, 6 * mode_count),
    ).build()


def _make_beam_links_model(device, mode_counts):
    """Build a rigid base hinged to the world carrying one elastic beam leaf link per entry of ``mode_counts``."""
    length = 0.6
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    base = builder.add_link(
        xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(*np.diag([0.002, 0.02, 0.02]).ravel()),
    )
    # Each elastic link creates its owner joint, so create the links before the articulation's joints.
    links = [
        builder.add_link_elastic(
            xform=wp.transform(wp.vec3(0.4 + 0.5 * length, 0.0, 0.3 * k), wp.quat_identity()),
            modal_basis=_beam_basis(mode_count),
        )
        for k, mode_count in enumerate(mode_counts)
    ]
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=base,
            axis=(0.0, 0.0, 1.0),
            child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
        )
    ]
    for k, link in enumerate(links):
        joints.append(
            builder.add_joint_revolute(
                parent=base,
                child=link,
                axis=(0.0, 0.0, 1.0),
                parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.3 * k), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[model.joint_qd_start.numpy()[joints[0]]] = 2.0
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


def _step(model, state_0, solver, steps: int):
    state_1 = model.state()
    control = model.control()
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, _DT)
        state_0, state_1 = state_1, state_0
    return state_0


def _modal_schur_reference(solver):
    """Return float64 ``z = A_mm^-1 r_m`` and ``W = A_mm^-1 B`` from the solver's last assembled data.

    ``A_mm`` is read from the stored upper triangle of each body's dense block, ``r_m`` is the
    negated modal gradient, and ``B`` the Schur column groups. Also returns the device's ``z`` and
    ``W`` restricted to the rows each body uses.
    """
    model = solver.model
    layout = solver.rigid_articulation_sparse_layout
    max_modes = int(model.elastic_max_mode_count)
    matrix = solver.elastic_body_block_matrix.numpy().astype(np.float64)
    grad = solver.elastic_body_block_grad.numpy().astype(np.float64)
    b = solver._sparse_elastic_b.numpy().reshape(-1, max_modes, 6).astype(np.float64)
    z = solver._sparse_elastic_z.numpy().reshape(-1, max_modes)
    w = solver._sparse_elastic_w.numpy().reshape(-1, max_modes, 6)
    owners = layout.elastic_group_owner.numpy()
    elastic_index = model.body_elastic_index.numpy()
    counts = model.elastic_mode_count.numpy()
    z_ref, z_got, w_ref, w_got = [], [], [], []
    for slot, body in enumerate(layout.elastic_bodies.numpy()):
        e = int(elastic_index[body])
        n = int(counts[e])
        upper = np.triu(matrix[e, 6 : 6 + n, 6 : 6 + n])
        a = upper + np.triu(upper, 1).T
        z_ref.append(np.linalg.solve(a, -grad[e, 6 : 6 + n]))
        z_got.append(z[slot, :n])
        for group in np.nonzero(owners == slot)[0]:
            w_ref.append(np.linalg.solve(a, b[group, :n]))
            w_got.append(w[group, :n])
    return (
        np.concatenate(z_ref),
        np.concatenate(z_got),
        np.concatenate([x.ravel() for x in w_ref]),
        np.concatenate([x.ravel() for x in w_got]),
    )


def _assert_modal_schur_matches_reference(test, solver, rtol: float, label: str):
    z_ref, z, w_ref, w = _modal_schur_reference(solver)
    test.assertTrue(np.all(solver._sparse_elastic_definite.numpy() == 1), label)
    test.assertGreater(float(np.max(np.abs(z_ref))), 0.0, label)
    test.assertGreater(float(np.max(np.abs(w_ref))), 0.0, label)
    np.testing.assert_allclose(z, z_ref, rtol=rtol, atol=rtol * float(np.max(np.abs(z_ref))), err_msg=label)
    np.testing.assert_allclose(w, w_ref, rtol=rtol, atol=rtol * float(np.max(np.abs(w_ref))), err_msg=label)


def test_sparse_modal_solve_matches_float64_reference(test, device):
    """Reproduce a float64 serial solve of the modal Schur data with the cooperative factorization.

    A 64-mode elastic link's modal block ``A_mm`` is factored in one shared-memory tile, and all
    right-hand sides of the body (``r_m`` and every Schur column group ``B``) are solved in the
    same pass. The resulting ``z = A_mm^-1 r_m`` and ``W = A_mm^-1 B`` match a float64 numpy solve
    of the same assembled data to ``rtol = 1e-4``.
    """
    model, state = _make_beam_links_model(device, [64])
    solver = newton.solvers.SolverVBD(model, iterations=2, rigid_articulation_solve=_SPARSE)
    test.assertEqual(solver._sparse_elastic_modal_solve[0], "tiled")
    _step(model, state, solver, 3)
    _assert_modal_schur_matches_reference(test, solver, 1.0e-4, "tiled, 64 modes")


def test_sparse_modal_dense_solve_matches_tiled(test, device):
    """Match the dense panel modal solve to a float64 reference and to the tiled solve.

    Two links with 40 and 23 modes exercise the identity padding of the modal block to the
    largest mode count and to whole 32-row panels.
    """
    trajectories = {}
    for elastic_block_solver, path in (("auto", "tiled"), ("dense", "dense")):
        model, state = _make_beam_links_model(device, [40, 23])
        solver = newton.solvers.SolverVBD(
            model, iterations=4, rigid_articulation_solve=_SPARSE, elastic_block_solver=elastic_block_solver
        )
        test.assertEqual(solver._sparse_elastic_modal_solve[0], path)
        state = _step(model, state, solver, 20)
        _assert_modal_schur_matches_reference(test, solver, 1.0e-4, path)
        trajectories[path] = np.concatenate((state.body_q.numpy().ravel(), state.joint_q.numpy()))
    test.assertTrue(np.isfinite(trajectories["dense"]).all())
    np.testing.assert_allclose(trajectories["dense"], trajectories["tiled"], atol=1.0e-5, rtol=0.0)


class TestReducedElasticSparsePerformance(unittest.TestCase):
    pass


for _name, _test in (
    ("test_sparse_modal_solve_matches_float64_reference", test_sparse_modal_solve_matches_float64_reference),
    ("test_sparse_modal_dense_solve_matches_tiled", test_sparse_modal_dense_solve_matches_tiled),
):
    add_function_test(TestReducedElasticSparsePerformance, _name, _test, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
