# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Modal factorization, fallbacks, and validation of reduced elastic bodies in the VBD block-sparse solve."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton.tests import test_reduced_elastic_sparse_articulation as sparse_tests
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

devices = get_test_devices()
cuda_devices = get_cuda_test_devices()

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


def _make_inconsistent_leaf_link_model(device):
    """Build a hinged elastic leaf link whose modal block has no Cholesky factor.

    The basis holds three beam modes, a duplicate of the axial mode, and a decoupled mode with
    a zero shape and negative modal stiffness, so ``A_mm`` is indefinite while every physical
    coordinate keeps bounded dynamics. The duplicated axial mode moves the hinge attachment like
    the axial mode, which makes ``A_ff - B^T diag(A_mm)^-1 B`` indefinite under stiff joints.
    """
    length = 0.6
    beam = newton.ModalGeneratorBeam
    base_basis = beam(
        length=length,
        half_width_y=0.03,
        half_width_z=0.02,
        young_modulus=2.0e6,
        density=1000.0,
        mode_specs=[
            {"type": beam.Mode.AXIAL},
            {"type": beam.Mode.BENDING_Y, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
            {"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
        ],
    ).build()
    zero = np.zeros((base_basis.sample_count, 1, 3), dtype=np.float32)
    basis = newton.ModalBasis(
        sample_points=base_basis.sample_points,
        sample_phi=np.concatenate((base_basis.sample_phi, base_basis.sample_phi[:, :1], zero), axis=1),
        sample_psi=np.concatenate((base_basis.sample_psi, base_basis.sample_psi[:, :1], zero), axis=1),
        sample_mass=base_basis.sample_mass,
        mode_mass=[*base_basis.mode_mass, base_basis.mode_mass[0], 1.0e-3],
        mode_stiffness=[*base_basis.mode_stiffness, base_basis.mode_stiffness[0], -1.0e4],
        mode_damping=[*base_basis.mode_damping, 0.0, 0.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    base = builder.add_link(
        xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(*np.diag([0.002, 0.02, 0.02]).ravel()),
    )
    link = builder.add_link_elastic(
        xform=wp.transform(wp.vec3(0.4 + 0.5 * length, 0.0, 0.0), wp.quat_identity()),
        mass=base_basis.rigid_mass,
        com=wp.vec3(*base_basis.rigid_com),
        inertia=wp.mat33(*np.asarray(base_basis.rigid_inertia).ravel()),
        modal_basis=basis,
        mode_q=[0.0, 0.0, 0.01, 0.0, 0.0],
    )
    root = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=(0.0, 0.0, 1.0),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    hinge = builder.add_joint_revolute(
        parent=base,
        child=link,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * length, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([root, hinge])
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[model.joint_qd_start.numpy()[root]] = 2.0
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


def test_sparse_modal_fallback_keeps_frames_definite(test, device):
    """Keep the sparse solve finite when an elastic body's modal block has no Cholesky factor.

    Eliminating the modes with ``diag(A_mm)`` instead of ``A_mm`` makes the frame Schur
    complement indefinite for this fixture, and the frame factorization then returned NaN for
    the whole articulation. The fallback instead solves the frames without the modal Schur terms
    and recovers the modes with a diagonal step afterwards, like the local block solve's Jacobi
    fallback: both stay finite and close to each other, and the sparse solve warns once.
    """
    steps = 120
    stiff = {"rigid_joint_linear_ke": 1.0e6, "rigid_joint_angular_ke": 1.0e6}
    trajectories = {}
    for mode in ("local", _SPARSE):
        with warnings.catch_warnings():
            # The fixture is deliberately inconsistent; its construction warnings are expected.
            warnings.simplefilter("ignore")
            model, state = _make_inconsistent_leaf_link_model(device)
            solver = newton.solvers.SolverVBD(model, iterations=24, rigid_articulation_solve=mode, **stiff)
        state_1 = model.state()
        control = model.control()
        series = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(steps):
                solver.step(state, state_1, control, None, _DT)
                state, state_1 = state_1, state
                series.append(state.body_q.numpy().ravel().copy())
        trajectories[mode] = np.asarray(series)
        messages = [str(w.message) for w in caught if "could not Cholesky-factor" in str(w.message)]
        if mode == _SPARSE:
            test.assertEqual(solver._sparse_elastic_definite.numpy().tolist(), [0])
            test.assertEqual(len(messages), 1)
        else:
            test.assertEqual(messages, [])
        test.assertTrue(np.isfinite(trajectories[mode]).all(), mode)
    test.assertGreater(float(np.max(np.abs(trajectories[_SPARSE][-1] - trajectories[_SPARSE][0]))), 0.1)
    np.testing.assert_allclose(trajectories[_SPARSE], trajectories["local"], atol=2.0e-2, rtol=0.0)


def test_sparse_modal_fallback_checks_frame_schur(test, device):
    """Fall back when the modal block factors but the body's whole frame-and-mode block is indefinite.

    An inconsistent dense modal mass leaves ``A_mm`` positive definite while the reduced mass is
    indefinite, so the exact frame Schur complement ``A_ff - B^T A_mm^-1 B`` is indefinite too.
    That body is solved with the fallback instead of returning NaN for the whole articulation.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, state, _hinge = sparse_tests._make_leaf_link_model(device)
        mode_count = model.elastic_max_mode_count
        mass = model.elastic_mode_mass_matrix.numpy().reshape(mode_count, mode_count).copy()
        mass[1, 2] = mass[2, 1] = 2.0 * np.sqrt(mass[1, 1] * mass[2, 2])
        model.elastic_mode_mass_matrix.assign(mass.reshape(-1))
        solver = newton.solvers.SolverVBD(model, iterations=24, rigid_articulation_solve=_SPARSE)
    base_start = state.body_q.numpy()[0, :3].copy()
    state_1 = model.state()
    solver.step(state, state_1, model.control(), None, _DT)
    test.assertEqual(solver._sparse_elastic_definite.numpy().tolist(), [0])
    test.assertTrue(np.isfinite(state_1.body_q.numpy()).all())
    test.assertTrue(np.isfinite(state_1.joint_q.numpy()).all())
    test.assertLess(float(np.max(np.abs(state_1.body_q.numpy()[0, :3] - base_start))), 0.05)


def test_sparse_elastic_revalidates_narrow_model_flags(test, device, backend):
    """Reject unsupported elastic joint terms notified with the narrow model-change flags.

    Armature is notified with ``JOINT_DOF_INERTIAL_PROPERTIES``, damping and friction with
    ``JOINT_DOF_FORCE_PROPERTIES``, and mimic relationships with ``CONSTRAINT_PROPERTIES``; none of
    these sets the broad ``JOINT_DOF_PROPERTIES`` or ``JOINT_PROPERTIES`` bit. The local solve keeps
    accepting the same changes. Both ``rigid_articulation_elastic_solve`` backends revalidate.
    """
    flags = newton.ModelFlags
    for attribute, flag in (
        ("joint_armature", flags.JOINT_DOF_INERTIAL_PROPERTIES),
        ("joint_damping", flags.JOINT_DOF_FORCE_PROPERTIES),
        ("joint_friction", flags.JOINT_DOF_FORCE_PROPERTIES),
    ):
        for mode in (_SPARSE, "local"):
            model, _state, hinge = sparse_tests._make_leaf_link_model(device)
            solver = newton.solvers.SolverVBD(
                model, iterations=2, rigid_articulation_solve=mode, rigid_articulation_elastic_solve=backend
            )
            values = getattr(model, attribute).numpy()
            values[model.joint_qd_start.numpy()[hinge]] = 0.01
            getattr(model, attribute).assign(values)
            if mode == _SPARSE:
                with test.assertRaisesRegex(ValueError, attribute, msg=f"{attribute}, {flag.name}"):
                    solver.notify_model_changed(flag)
            else:
                solver.notify_model_changed(flag)

    model, _state, hinge = sparse_tests._make_leaf_link_model(device)
    solver = newton.solvers.SolverVBD(
        model, iterations=2, rigid_articulation_solve=_SPARSE, rigid_articulation_elastic_solve=backend
    )
    joint_type = model.joint_type.numpy()
    root = next(
        joint
        for joint, parent in enumerate(model.joint_parent.numpy())
        if parent == -1 and joint_type[joint] == int(newton.JointType.REVOLUTE)
    )
    # The world hinge follows the elastic-attached hinge.
    mimic = model.joint_mimic_joint.numpy()
    mimic[root] = hinge
    model.joint_mimic_joint.assign(mimic)
    with test.assertRaisesRegex(NotImplementedError, "joint mimics"):
        solver.notify_model_changed(flags.CONSTRAINT_PROPERTIES)


def _elastic_chain_series(device, mode: str, iterations: int, steps: int = 120):
    """Return the per-step ``[body_q, joint_q]`` series of the two-link elastic chain."""
    model = sparse_tests._make_elastic_chain_model(device)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    solver = sparse_tests._solver(model, mode, iterations=iterations)
    series = []
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, _DT)
        state_0, state_1 = state_1, state_0
        series.append(np.concatenate((state_0.body_q.numpy().ravel(), state_0.joint_q.numpy())))
    return np.asarray(series)


def test_sparse_elastic_elastic_chain_converges_to_local(test, device):
    """Converge to the local solve's fixed point for directly jointed elastic links, more slowly.

    The per-body elimination cannot represent the mode-mode coupling of a hinge between two
    elastic links and adds a conservative positive semidefinite term instead, which keeps the
    fixed point but slows convergence. Over 0.25 s the sparse trajectory is about 8.5e-4 from a
    converged 384-iteration local reference at 96 iterations, where the local solve is within
    2e-5, and about 1e-5 at 384 iterations; the bounds assert 2e-3 and 5e-5.
    """
    reference = _elastic_chain_series(device, "local", 384)
    for iterations, bound in ((96, 2.0e-3), (384, 5.0e-5)):
        sparse = _elastic_chain_series(device, _SPARSE, iterations)
        test.assertTrue(np.isfinite(sparse).all())
        error = float(np.max(np.abs(sparse - reference)))
        test.assertLess(error, bound, f"{iterations} iterations")


def test_sparse_modal_fallback_warns_after_graph_replay(test, device):
    """Count fallback factorizations in a captured graph and report them outside capture."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, state_0 = _make_inconsistent_leaf_link_model(device)
        solver = newton.solvers.SolverVBD(model, iterations=8, rigid_articulation_solve=_SPARSE)
        state_1 = model.state()
        control = model.control()
        # Load the kernels before capture; this step's count is only mirrored to the host.
        solver.step(state_0, state_1, control, None, _DT)
    solver._sparse_elastic_indefinite_count.zero_()
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_1, state_0, control, None, _DT)
        solver.step(state_0, state_1, control, None, _DT)
    for _ in range(3):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    with test.assertWarnsRegex(UserWarning, r"could not Cholesky-factor .* \(48 times so far\)"):
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    test.assertTrue(np.isfinite(state_1.body_q.numpy()).all())


class TestReducedElasticSparsePerformance(unittest.TestCase):
    pass


for _name, _test in (
    ("test_sparse_modal_solve_matches_float64_reference", test_sparse_modal_solve_matches_float64_reference),
    ("test_sparse_modal_dense_solve_matches_tiled", test_sparse_modal_dense_solve_matches_tiled),
    ("test_sparse_modal_fallback_keeps_frames_definite", test_sparse_modal_fallback_keeps_frames_definite),
    ("test_sparse_modal_fallback_checks_frame_schur", test_sparse_modal_fallback_checks_frame_schur),
    ("test_sparse_elastic_elastic_chain_converges_to_local", test_sparse_elastic_elastic_chain_converges_to_local),
):
    # Schur-backend internals and convergence (the default rigid_articulation_elastic_solve).
    add_function_test(TestReducedElasticSparsePerformance, _name, _test, devices=devices)

for _backend in ("schur", "coupled"):
    add_function_test(
        TestReducedElasticSparsePerformance,
        f"test_sparse_elastic_revalidates_narrow_model_flags_{_backend}",
        test_sparse_elastic_revalidates_narrow_model_flags,
        devices=devices,
        backend=_backend,
    )

add_function_test(
    TestReducedElasticSparsePerformance,
    "test_sparse_modal_fallback_warns_after_graph_replay",
    test_sparse_modal_fallback_warns_after_graph_replay,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
