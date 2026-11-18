# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check passive forces at reduced-elastic joint attachments."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _model(device, parent_elastic, *, damping, friction, psi=1.0, linear=False, d6=False):
    """Build two unit inertias and one unit modal inertia with a scalar joint."""
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[psi if linear or d6 else 0.0, 0.0, 0.0]]],
        sample_psi=[[[0.0, 0.0, 0.0 if linear and not d6 else psi]]],
        mode_mass=[1.0],
        mode_stiffness=[0.0],
        mode_damping=[0.0],
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    elastic = builder.add_link_elastic(mass=1.0, inertia=wp.mat33(np.eye(3)), modal_basis=basis)
    rigid = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
    parent, child = (elastic, rigid) if parent_elastic else (rigid, elastic)
    root = builder.elastic_joint[0] if parent_elastic else builder.add_joint_free(child=rigid)
    if d6:
        config = newton.ModelBuilder.JointDofConfig
        hinge = builder.add_joint_d6(
            parent,
            child,
            linear_axes=[config(axis=newton.Axis.X, damping=damping, friction=friction)],
            angular_axes=[config(axis=newton.Axis.Z, damping=damping, friction=friction)],
        )
    else:
        add_joint = builder.add_joint_prismatic if linear else builder.add_joint_revolute
        hinge = add_joint(
            parent,
            child,
            axis=newton.Axis.X if linear else newton.Axis.Z,
            damping=damping,
            friction=friction,
            armature=0.0,
            target_ke=0.0,
            target_kd=0.0,
        )
    builder.add_articulation([root, hinge])
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    owner = int(model.elastic_joint.numpy()[0])
    modal_start = int(model.joint_qd_start.numpy()[owner]) + 6
    velocity = state.joint_qd.numpy()
    velocity[modal_start] = 0.8
    state.joint_qd.assign(velocity)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state, elastic, rigid, modal_start


def test_vbd_elastic_passive_response(test, device, *, alm=True):
    """Match scalar implicit damping and sliding friction, including modal reactions."""
    dt = 0.01
    for solve, backend in (("local", "coupled"), ("block_sparse_joints", "schur"), ("block_sparse_joints", "coupled")):
        for parent_elastic in (False, True):
            for linear in (False, True):
                for damping, friction in ((4.0, 0.0), (0.0, 0.2), (4.0, 0.2)):
                    with test.subTest(
                        solve=solve,
                        backend=backend,
                        parent_elastic=parent_elastic,
                        linear=linear,
                        damping=damping,
                        friction=friction,
                    ):
                        model, state, elastic, rigid, modal = _model(
                            device, parent_elastic, damping=damping, friction=friction, linear=linear
                        )
                        solver = newton.solvers.SolverVBD(
                            model,
                            iterations=64,
                            rigid_articulation_solve=solve,
                            rigid_articulation_elastic_solve=backend,
                            rigid_avbd_alpha=0.0,
                            rigid_compliant_alm=alm,
                        )
                        output = model.state()
                        solver.step(state, output, model.control(), None, dt)
                        # Each of the two frames and the mode contributes unit inverse inertia.
                        relative_velocity = (0.8 - 3.0 * dt * friction) / (1.0 + 3.0 * dt * damping)
                        reaction = damping * relative_velocity + friction
                        component = 0 if linear else 5
                        frame_velocities = output.body_qd.numpy()
                        test.assertAlmostEqual(
                            float(frame_velocities[elastic, component]), -dt * reaction, delta=2.0e-5
                        )
                        test.assertAlmostEqual(float(frame_velocities[rigid, component]), dt * reaction, delta=2.0e-5)
                        test.assertAlmostEqual(float(output.joint_qd.numpy()[modal]), 0.8 - dt * reaction, delta=2.0e-5)
                        final_energy = 0.5 * (np.sum(frame_velocities**2) + float(output.joint_qd.numpy()[modal]) ** 2)
                        test.assertLess(final_energy, 0.5 * 0.8**2)


def test_vbd_elastic_d6_passive_response(test, device, *, alm=True):
    """Load one mode through both free D6 coordinates without losing their cross terms."""
    dt = 0.001
    for solve, backend in (("local", "coupled"), ("block_sparse_joints", "schur"), ("block_sparse_joints", "coupled")):
        for parent_elastic in (False, True):
            model, state, elastic, rigid, modal = _model(device, parent_elastic, damping=4.0, friction=0.2, d6=True)
            solver = newton.solvers.SolverVBD(
                model,
                iterations=64,
                rigid_articulation_solve=solve,
                rigid_articulation_elastic_solve=backend,
                rigid_avbd_alpha=0.0,
                rigid_compliant_alm=alm,
            )
            output = model.state()
            solver.step(state, output, model.control(), None, dt)
            # The two coordinate rows share the mode, giving a row-sum inverse inertia of four.
            relative_velocity = (0.8 - 4.0 * dt * 0.2) / (1.0 + 4.0 * dt * 4.0)
            reaction = 4.0 * relative_velocity + 0.2
            velocity = output.body_qd.numpy()
            np.testing.assert_allclose(velocity[elastic, [0, 5]], -dt * reaction, atol=2.0e-5)
            np.testing.assert_allclose(velocity[rigid, [0, 5]], dt * reaction, atol=2.0e-5)
            test.assertAlmostEqual(float(output.joint_qd.numpy()[modal]), 0.8 - 2.0 * dt * reaction, delta=2.0e-5)


def test_vbd_elastic_passive_rigid_limit(test, device, *, alm=True):
    """Recover two rigid inertias when the mode does not move the attachment."""
    dt = 0.01
    for solve, backend in (("local", "coupled"), ("block_sparse_joints", "schur"), ("block_sparse_joints", "coupled")):
        for parent_elastic in (False, True):
            model, state, elastic, rigid, modal = _model(device, parent_elastic, damping=4.0, friction=0.2, psi=0.0)
            velocity = state.joint_qd.numpy()
            velocity[modal - 1] = 0.8
            velocity[int(model.joint_qd_start.numpy()[-2])] = -0.8 if parent_elastic else 0.8
            state.joint_qd.assign(velocity)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver = newton.solvers.SolverVBD(
                model,
                iterations=64,
                rigid_articulation_solve=solve,
                rigid_articulation_elastic_solve=backend,
                rigid_avbd_alpha=0.0,
                rigid_compliant_alm=alm,
            )
            output = model.state()
            solver.step(state, output, model.control(), None, dt)
            relative_velocity = (0.8 - 2.0 * dt * 0.2) / (1.0 + 2.0 * dt * 4.0)
            reaction = 4.0 * relative_velocity + 0.2
            velocity = output.body_qd.numpy()
            test.assertAlmostEqual(float(velocity[elastic, 5]), 0.8 - dt * reaction, delta=2.0e-5)
            test.assertAlmostEqual(float(velocity[rigid, 5]), dt * reaction, delta=2.0e-5)
            test.assertAlmostEqual(float(output.joint_qd.numpy()[modal]), 0.8, delta=2.0e-5)


class TestSolverVBDElasticPassive(unittest.TestCase):
    pass


for test_function in (
    test_vbd_elastic_passive_response,
    test_vbd_elastic_d6_passive_response,
    test_vbd_elastic_passive_rigid_limit,
):
    for alm in (False, True):
        add_function_test(
            TestSolverVBDElasticPassive,
            f"{test_function.__name__}_{'alm' if alm else 'legacy'}",
            test_function,
            devices=get_test_devices(),
            alm=alm,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
