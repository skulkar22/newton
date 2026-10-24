# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of :meth:`SolverVBD.refresh_elastic_mass_solve_data` (in-place inverse reduced mass refresh)."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices

devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()

_INERTIAL_ARRAYS = (
    "body_mass",
    "body_inv_mass",
    "body_inertia",
    "body_inv_inertia",
    "elastic_mode_mass",
    "elastic_mode_stiffness",
    "elastic_mode_mass_matrix",
    "elastic_mode_coupling_linear",
    "elastic_mode_coupling_angular",
    "elastic_mode_coupling_centrifugal",
    "elastic_mode_coupling_coriolis",
    "elastic_mode_coupling_quadratic",
)


def _elastic_model(device):
    """A free reduced elastic body with two lumped-mass sampled modes on a 3x3x3 point grid."""
    axis = np.linspace(0.0, 0.2, 3)
    points = np.stack(np.meshgrid(axis, 0.5 * axis, 0.5 * axis, indexing="ij"), axis=-1).reshape(-1, 3)
    phi = np.zeros((points.shape[0], 2, 3))
    phi[:, 0, 2] = (points[:, 0] / 0.2) ** 2  # bending-like
    phi[:, 1, 1] = points[:, 0] * points[:, 2] / 0.02  # twist-like
    basis = newton.ModalGeneratorSampled(
        sample_points=points,
        sample_phi=phi,
        mode_stiffness=[40.0, 60.0],
        sample_mass=np.full(points.shape[0], 0.05),
    ).build()
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        modal_basis=basis,
        mode_qd=[0.3, -0.2],
    )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    jqd = state.joint_qd.numpy()
    jqd[:6] = [0.1, 0.0, 0.05, 0.2, 0.4, 1.0]
    state.joint_qd.assign(jqd)
    return model, state, body


def _scale_inertia(model, factor: float):
    """Scale every inertial property of the elastic body consistently (keeps M(0) positive definite)."""
    for name in _INERTIAL_ARRAYS:
        array = getattr(model, name)
        values = array.numpy()
        if name in ("body_inv_mass", "body_inv_inertia"):
            values = values / factor
        elif name != "elastic_mode_stiffness":
            values = values * factor
        array.assign(values)


def _mass_solve_arrays(solver):
    mobility = solver.elastic_reduced_mobility
    return {
        "flags": solver.elastic_implicit_mass_coupling,
        "frame_inv_schur": mobility.frame_inv_schur,
        "frame_modal_mass": mobility.frame_modal_mass,
        "modal_inv_mass": mobility.modal_inv_mass,
        "modal_inv_mass_dense": mobility.modal_inv_mass_dense,
    }


def test_refresh_writes_in_place(test, device):
    """The refresh keeps every allocation and reproduces a fresh build of the mass-solve data."""
    model, _state, _body = _elastic_model(device)
    solver = newton.solvers.SolverVBD(model, iterations=4)
    before = {k: (a.ptr, a.numpy().copy()) for k, a in _mass_solve_arrays(solver).items()}
    _scale_inertia(model, 2.0)
    test.assertTrue(solver.refresh_elastic_mass_solve_data())
    after = _mass_solve_arrays(solver)
    flags, fresh = solver._build_elastic_mass_solve_data(model)
    expected = {
        "flags": flags,
        "frame_inv_schur": fresh.frame_inv_schur,
        "frame_modal_mass": fresh.frame_modal_mass,
        "modal_inv_mass": fresh.modal_inv_mass,
        "modal_inv_mass_dense": fresh.modal_inv_mass_dense,
    }
    test.assertEqual(int(solver.elastic_reduced_mobility.modal_mass_is_dense), int(fresh.modal_mass_is_dense))
    for name, array in after.items():
        test.assertEqual(array.ptr, before[name][0], name)
        np.testing.assert_array_equal(array.numpy(), expected[name].numpy(), err_msg=name)
    # the inverse of a twice heavier body is half as large
    np.testing.assert_allclose(after["frame_inv_schur"].numpy(), 0.5 * before["frame_inv_schur"][1], rtol=1.0e-5)
    eff = solver.body_inv_mass_effective.numpy()
    np.testing.assert_allclose(eff, model.body_inv_mass.numpy(), rtol=0.0)


def test_refresh_reallocates_on_storage_switch(test, device):
    """A modal mass matrix that turns dense cannot be refreshed in place; the data are reallocated."""
    model, _state, _body = _elastic_model(device)
    solver = newton.solvers.SolverVBD(model, iterations=4)
    mobility = solver.elastic_reduced_mobility
    max_modes = int(model.elastic_max_mode_count)
    mass_matrix = model.elastic_mode_mass_matrix.numpy().reshape(-1, max_modes, max_modes)
    dense_before = int(mobility.modal_mass_is_dense)
    if dense_before:
        mass_matrix[0] = np.diag(np.diag(mass_matrix[0]))
    else:
        mass_matrix[0, 0, 1] = mass_matrix[0, 1, 0] = 0.05 * np.sqrt(mass_matrix[0, 0, 0] * mass_matrix[0, 1, 1])
    model.elastic_mode_mass_matrix.assign(mass_matrix.reshape(-1))
    ptr = mobility.frame_inv_schur.ptr
    test.assertFalse(solver.refresh_elastic_mass_solve_data())
    test.assertNotEqual(int(solver.elastic_reduced_mobility.modal_mass_is_dense), dense_before)
    test.assertNotEqual(solver.elastic_reduced_mobility.frame_inv_schur.ptr, ptr)


def test_captured_step_graph_stays_valid(test, device):
    """A step graph captured before an inertia change replays like an uncaptured run after the refresh."""
    dt = 2.0e-3
    results = {}
    for captured in (False, True):
        model, s0, _body = _elastic_model(device)
        s1 = model.state()
        control = model.control()
        solver = newton.solvers.SolverVBD(model, iterations=4)

        def two_steps(s0=s0, s1=s1, control=control, solver=solver):
            s0.clear_forces()
            solver.step(s0, s1, control, None, dt)
            s1.clear_forces()
            solver.step(s1, s0, control, None, dt)

        two_steps()  # load the kernels before a capture
        if captured:
            with wp.ScopedCapture(device=device) as capture:
                two_steps()

            def run(graph=capture.graph):
                wp.capture_launch(graph)
        else:
            run = two_steps
        for _ in range(5):
            run()
        _scale_inertia(model, 3.0)
        test.assertTrue(solver.refresh_elastic_mass_solve_data())
        for _ in range(20):
            run()
        results[captured] = (s0.joint_q.numpy().copy(), s0.joint_qd.numpy().copy())
    np.testing.assert_array_equal(results[True][0], results[False][0])
    np.testing.assert_array_equal(results[True][1], results[False][1])


class TestVBDElasticMassRefresh(unittest.TestCase):
    pass


add_function_test(
    TestVBDElasticMassRefresh, "test_refresh_writes_in_place", test_refresh_writes_in_place, devices=devices
)
add_function_test(
    TestVBDElasticMassRefresh,
    "test_refresh_reallocates_on_storage_switch",
    test_refresh_reallocates_on_storage_switch,
    devices=devices,
)
add_function_test(
    TestVBDElasticMassRefresh,
    "test_captured_step_graph_stays_valid",
    test_captured_step_graph_stays_valid,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
