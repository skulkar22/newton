# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _advance_reference(
    angles: wp.array[float],
    increment: float,
    flip_sign: bool,
    body_q: wp.array[wp.transform],
):
    pair = wp.tid()
    angle = angles[pair] + increment
    angles[pair] = angle
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
    if flip_sign:
        rotation = -rotation
    body_q[2 * pair] = wp.transform(wp.vec3(0.0), rotation)


@wp.kernel
def _prescribe_d6_reference(linear: float, angular: float, body_q: wp.array[wp.transform]):
    body_q[1] = wp.transform(wp.vec3(0.0, 0.0, linear), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angular))


class _MimicRig:
    def __init__(
        self,
        device,
        *,
        ratio=0.01,
        offset=0.03,
        initial=0.0,
        angular_follower=False,
        d6=False,
        worlds=False,
        kinematic=True,
        solve="local",
    ):
        self.ratio, self.offset = ratio, offset
        self.angular_follower = angular_follower
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for world in (0, 1, -1) if worlds else (-1,):
            if world >= 0:
                builder.begin_world()
            parent = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True, is_kinematic=kinematic)
            child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
            cfg = newton.ModelBuilder.JointDofConfig(
                axis=newton.Axis.Z, target_ke=0.0, target_kd=0.0, limit_ke=0.0, limit_kd=0.0
            )
            if d6:
                reference = builder.add_joint_d6(-1, parent, angular_axes=[cfg])
            else:
                reference = builder.add_joint(newton.JointType.REVOLUTE, -1, parent, angular_axes=[cfg])
            if angular_follower:
                follower = builder.add_joint(newton.JointType.REVOLUTE, -1, child, angular_axes=[cfg])
            else:
                follower = builder.add_joint(newton.JointType.PRISMATIC, -1, child, linear_axes=[cfg])
            builder.add_articulation([reference, follower])
            builder.set_joint_mimic(follower, reference, coeffs=(offset, ratio))
            builder.joint_q[-2:] = [initial, offset + ratio * initial]
            if world >= 0:
                builder.end_world()
        builder.color()
        self.model = builder.finalize(device=device)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.state, self.next_state = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverVBD(
            self.model, iterations=12, rigid_compliant_alm=True, rigid_articulation_solve=solve
        )
        self.angles = wp.full(3 if worlds else 1, initial, dtype=float, device=device)

    def step(self, increment=0.1, flip_sign=False):
        """Prescribe the reference pose and solve the follower without setting its coordinate."""
        wp.launch(
            _advance_reference,
            dim=self.angles.size,
            inputs=[self.angles, increment, flip_sign],
            outputs=[self.state.body_q],
            device=self.model.device,
        )
        self.solver.step(self.state, self.next_state, self.control, None, 1.0 / 240.0)
        self.state, self.next_state = self.next_state, self.state

    def check(self):
        """Compare actual follower poses against the commanded, accumulated reference angles."""
        targets = self.offset + self.ratio * self.angles.numpy()
        poses = self.state.body_q.numpy()[1::2]
        if self.angular_follower:
            for pose, angle in zip(poses, targets, strict=True):
                expected = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(angle)))
                np.testing.assert_allclose(abs(np.dot(pose[3:], expected)), 1.0, atol=2.0e-6)
        else:
            np.testing.assert_allclose(poses[:, 2], targets, atol=2.0e-5)


def test_vbd_mimic_multiturn_translation(test, device):
    """Extend a lead screw continuously over several turns in either direction."""
    for d6 in (False, True):
        for ratio in (0.01, -0.02):
            with test.subTest(d6=d6, ratio=ratio):
                rig = _MimicRig(device, ratio=ratio, d6=d6)
                for direction in (1.0, -1.0, -1.0):
                    for step in range(100):
                        rig.step(direction * 0.1, flip_sign=step % 2 == 0)
                        rig.check()


def test_vbd_mimic_fractional_angular_ratio(test, device):
    """Preserve fractional gear ratios when either angular joint crosses a turn."""
    for ratio in (0.5, -1.5):
        rig = _MimicRig(device, ratio=ratio, angular_follower=True)
        for _ in range(180):
            rig.step()
            rig.check()


def test_vbd_mimic_multiturn_dynamics(test, device):
    """Keep a driven lead screw coupled across turns with friction and damping on both joints."""
    rig = _MimicRig(device, kinematic=False)
    rig.model.joint_friction.assign([0.2, 2.0])
    rig.model.joint_damping.assign([0.2, 20.0])
    speed = 12.0
    rig.state.joint_qd.assign([speed, rig.ratio * speed])
    newton.eval_fk(rig.model, rig.state.joint_q, rig.state.joint_qd, rig.state)
    # Balance both joints' dissipation to maintain approximately constant speed.
    torque = 0.2 + rig.ratio * 2.0 + (0.2 + rig.ratio**2 * 20.0) * speed
    rig.control.joint_f.assign([torque, 0.0])
    angle, previous = 0.0, 0.0
    for _ in range(300):
        rig.solver.step(rig.state, rig.next_state, rig.control, None, 1.0 / 240.0)
        rig.state, rig.next_state = rig.next_state, rig.state
        poses = rig.state.body_q.numpy()
        wrapped = 2.0 * math.atan2(poses[0, 5], poses[0, 6])
        angle += math.remainder(wrapped - previous, 2.0 * math.pi)
        previous = wrapped
        test.assertAlmostEqual(float(poses[1, 2]), rig.offset + rig.ratio * angle, delta=2.0e-5)
    test.assertGreater(angle, 4.0 * math.pi)
    newton.eval_ik(rig.model, rig.state, rig.state.joint_q, rig.state.joint_qd)
    velocities = rig.state.joint_qd.numpy()
    np.testing.assert_allclose(velocities[1], rig.ratio * velocities[0], atol=2.0e-5)


def test_vbd_mimic_initial_turns(test, device):
    """Seed turns from authored joint coordinates, including non-model initial states."""
    for initial in (-4.0 * math.pi - 0.2, 6.0 * math.pi + 0.2):
        rig = _MimicRig(device, initial=initial)
        rig.step(0.0, flip_sign=True)
        rig.check()
        rig = _MimicRig(device)
        rig.state.joint_q.assign([initial, rig.offset + rig.ratio * initial])
        newton.eval_fk(rig.model, rig.state.joint_q, rig.model.joint_qd, rig.state)
        rig.angles.fill_(initial)
        rig.step(0.0)
        rig.check()


def test_vbd_mimic_d6_coordinates(test, device):
    """Track D6 linear and angular coordinates independently, with differing q/qd offsets."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    # A preceding FREE joint ensures coordinate indices differ from velocity indices.
    builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), is_kinematic=True)
    joints = []
    for kinematic in (True, False):
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), is_kinematic=kinematic)
        cfg = newton.ModelBuilder.JointDofConfig(
            axis=newton.Axis.Z, target_ke=0.0, target_kd=0.0, limit_ke=0.0, limit_kd=0.0
        )
        joints.append(builder.add_joint_d6(-1, body, linear_axes=[cfg], angular_axes=[cfg]))
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0], coeffs=(0.0, 0.5))
    initial = 4.0 * math.pi + 0.2
    builder.joint_q[-4:] = [2.0, initial, 1.0, 0.5 * initial]
    builder.color()
    model = builder.finalize(device=device)
    state, next_state = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver = newton.solvers.SolverVBD(model, iterations=12, rigid_compliant_alm=True)
    for step in range(200):
        linear, angular = 2.0 + 0.03 * step, initial + 0.1 * step
        wp.launch(_prescribe_d6_reference, dim=1, inputs=[linear, angular], outputs=[state.body_q], device=device)
        solver.step(state, next_state, None, None, 1.0 / 240.0)
        state, next_state = next_state, state
        pose = state.body_q.numpy()[2]
        test.assertAlmostEqual(float(pose[2]), 0.5 * linear, delta=2.0e-5)
        expected = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * angular))
        np.testing.assert_allclose(abs(np.dot(pose[3:], expected)), 1.0, atol=2.0e-6)


def test_vbd_mimic_masked_reset(test, device):
    """Reseed selected worlds after pose edits without disturbing other turn histories."""
    rig = _MimicRig(device, worlds=True)
    for _ in range(100):
        rig.step()
    for world in (0, 2):
        selected = np.zeros(3, dtype=bool)
        selected[world] = True
        mask = wp.array(selected, dtype=bool, device=device)
        rig.solver.reset(rig.state, world_mask=mask, flags=0)
        angle = -4.0 * math.pi - 0.2
        q = rig.state.joint_q.numpy()
        q[2 * world : 2 * world + 2] = [angle, rig.offset + rig.ratio * angle]
        rig.state.joint_q.assign(q)
        # Author after reset: the next step must use these poses and turn seeds.
        newton.eval_fk(rig.model, rig.state.joint_q, rig.model.joint_qd, rig.state, mask=mask)
        angles = rig.angles.numpy()
        angles[world] = angle
        rig.angles.assign(angles)
        rig.step()
        rig.check()

    rig.solver.reset(rig.state)
    rig.angles.zero_()
    rig.step()
    rig.check()


def test_vbd_mimic_disabled_history(test, device):
    """Keep counting turns while mimic enforcement is disabled and after notification."""
    rig = _MimicRig(device)
    for _ in range(100):
        rig.step()
    rig.model.joint_enabled.assign([False, True])
    rig.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    for _ in range(100):
        rig.step()
    rig.model.joint_enabled.fill_(True)
    rig.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    rig.step()
    rig.check()


def test_vbd_mimic_graph_replay(test, device):
    """Retain turn history across graph replays, including a captured episode reset."""
    rig = _MimicRig(device)
    rig.step(0.0)  # Load kernels before capture.
    rig.solver.reset(rig.state)
    graph = None
    reset_graph = None
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            rig.step()
            rig.step()
        graph = capture.graph
        with wp.ScopedCapture(device=device) as capture:
            rig.solver.reset(rig.state)
            rig.angles.zero_()
        reset_graph = capture.graph
    for episode in range(2):
        if episode:
            if reset_graph is None:
                rig.solver.reset(rig.state)
                rig.angles.zero_()
            else:
                wp.capture_launch(reset_graph)
        for _ in range(100):
            if graph is None:
                rig.step()
                rig.step()
            else:
                wp.capture_launch(graph)
            rig.check()


def _build_jointed_mimic_mechanism(device, topology, copies=1):
    """Build hanging hinge mechanisms whose mimic rows act on bodies that joints couple to other bodies.

    Each of the ``copies`` is its own articulation. Returns the model, the hinges as
    ``(parent, child, parent anchor)`` and the mimic rows as
    ``(follower coordinate, reference coordinate, offset, ratio)``.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    inertia = wp.mat33(np.eye(3) * 1.0e-2)
    hinges = []
    mimics = []
    initial = []

    def link(index, **kwargs):
        return builder.add_link(mass=1.0 + index, com=wp.vec3(0.0, 0.0, -0.25), inertia=inertia, **kwargs)

    def hinge(parent, child, anchor):
        hinges.append((parent, child, anchor))
        return builder.add_joint_revolute(
            parent, child, axis=newton.Axis.Y, parent_xform=wp.transform(anchor, wp.quat_identity())
        )

    def chain(count, x):
        links = [link(i) for i in range(count)]
        return [
            hinge(-1 if i == 0 else links[i - 1], links[i], (x, 0.0, 1.0) if i == 0 else (0.0, 0.0, -0.5))
            for i in range(count)
        ]

    def free_hinge(x):
        # A hinge whose body is outside every articulation.
        anchor = (x, 0.0, 1.0)
        body = builder.add_link(
            xform=wp.transform(anchor, wp.quat_identity()), mass=1.0, com=wp.vec3(0.3, 0.0, 0.0), inertia=inertia
        )
        return hinge(-1, body, anchor)

    shared_reference = free_hinge(-1.0) if topology == "shared_free_reference" else None
    if shared_reference is not None:
        initial.append((shared_reference, 0.0))
    for copy in range(copies):
        x = float(copy)
        if topology == "serial":
            # Follower and reference joints share the middle link.
            joints = chain(3, x)
            builder.add_articulation(joints)
            mimics.append((joints[2], joints[1], 0.1, 0.5))
            coordinates = [0.4, 0.2, 0.2]
        elif topology == "serial_far":
            # Follower and reference joints share no body.
            joints = chain(4, x)
            builder.add_articulation(joints)
            mimics.append((joints[3], joints[1], 0.1, 0.5))
            coordinates = [0.4, 0.2, 0.2, 0.2]
        elif topology == "gripper":
            # Two fingers on a swinging base, closing symmetrically.
            links = [link(i) for i in range(3)]
            joints = [
                hinge(-1, links[0], (x, 0.0, 1.0)),
                hinge(links[0], links[1], (0.1, 0.0, -0.5)),
                hinge(links[0], links[2], (-0.1, 0.0, -0.5)),
            ]
            builder.add_articulation(joints)
            mimics.append((joints[2], joints[1], 0.0, -1.0))
            coordinates = [0.4, 0.2, -0.2]
        elif topology == "two_followers":
            # Two rows of one articulation that share the reference joint's bodies.
            joints = chain(4, x)
            builder.add_articulation(joints)
            mimics.extend([(joints[2], joints[1], 0.1, 0.5), (joints[3], joints[1], 0.0, -0.5)])
            coordinates = [0.4, 0.2, 0.2, -0.1]
        elif topology in ("free_reference", "shared_free_reference"):
            # The reference hinge carries a body outside every articulation; with
            # "shared_free_reference" every copy follows the same one.
            reference = shared_reference
            if reference is None:
                reference = free_hinge(x + 0.5)
                initial.append((reference, 0.0))
            joints = chain(3, x)
            builder.add_articulation(joints)
            mimics.append((joints[2], reference, 0.1, 0.5))
            coordinates = [0.4, 0.2, 0.1]
        else:
            raise ValueError(topology)
        initial.extend(zip(joints, coordinates, strict=True))
    for follower, reference, offset, ratio in mimics:
        builder.set_joint_mimic(follower, reference, coeffs=(offset, ratio))
    for joint, coordinate in initial:
        builder.joint_q[builder.joint_q_start[joint]] = coordinate
    builder.color()
    model = builder.finalize(device=device)
    starts = model.joint_q_start.numpy()
    rows = [
        (int(starts[follower]), int(starts[reference]), offset, ratio) for follower, reference, offset, ratio in mimics
    ]
    return model, hinges, rows


def _hinge_separation(body_q, hinges):
    """Return the largest distance [m] between a hinge's parent anchor and its child body origin."""
    separation = 0.0
    for parent, child, anchor in hinges:
        point = np.asarray(anchor)
        if parent >= 0:
            point = np.asarray(wp.transform_point(wp.transform(*body_q[parent]), wp.vec3(*anchor)))
        separation = max(separation, float(np.linalg.norm(body_q[child, :3] - point)))
    return separation


def _run_jointed_mimic_mechanism(device, topology, iterations, copies=1, **solver_options):
    """Simulate 30 steps and return the coordinates, hinge separation, mimic error, residuals and solver."""
    model, hinges, rows = _build_jointed_mimic_mechanism(device, topology, copies)
    solver = newton.solvers.SolverVBD(model, iterations=iterations, rigid_compliant_alm=True, **solver_options)
    observables = solver.observables({solver.ObservableKind.SOLVER_RESIDUAL_HISTORY})
    state, next_state = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    separation = 0.0
    for _ in range(30):
        solver.step(state, next_state, None, None, 4.0e-3, observables=observables)
        state, next_state = next_state, state
        separation = max(separation, _hinge_separation(state.body_q.numpy(), hinges))
    coordinates = state.joint_q.numpy()
    mimic_error = max(abs(coordinates[f] - offset - ratio * coordinates[r]) for f, r, offset, ratio in rows)
    return coordinates, separation, mimic_error, observables.solver_residual_history.numpy()[:, -1], solver


def _sparse_solves(device):
    solves = [{"rigid_articulation_solve": "block_sparse_joints"}, {"rigid_articulation_solve": "block_sparse_levels"}]
    if device.is_cuda:
        # The CPU keeps the serial solve under this option.
        solves.append({"rigid_articulation_solve": "block_sparse_levels", "rigid_articulation_levels_launch": "level"})
    return solves


def _check_against_local_solve(test, device, topology, copies=1):
    """Compare the block-sparse solves at 16 and 64 iterations with the local solve at 64 iterations.

    Returns the block-sparse solvers.
    """
    solvers = []
    expected, expected_separation, _error, _residual, _solver = _run_jointed_mimic_mechanism(
        device, topology, 64, copies
    )
    for options in _sparse_solves(device):
        for iterations in (16, 64):
            with test.subTest(topology=topology, copies=copies, iterations=iterations, **options):
                coordinates, separation, mimic_error, residual, solver = _run_jointed_mimic_mechanism(
                    device, topology, iterations, copies, **options
                )
                # The hinges sag by about 1 mm under gravity in the local solve as well [m].
                test.assertLess(separation, expected_separation + 1.0e-5)
                # Two rows of one articulation converge on each other over the iterations [rad].
                test.assertLess(mimic_error, 1.0e-4 if iterations == 16 else 1.0e-6)
                np.testing.assert_allclose(coordinates, expected, atol=5.0e-4)
                # The iteration converges within the step instead of growing.
                test.assertLess(residual[-1], 1.0e-4 * residual[0])
                solvers.append(solver)
    return solvers


def test_vbd_mimic_block_sparse_jointed_bodies(test, device):
    """Solve mimics between jointed bodies with the articulation's coupled system.

    The block-sparse solves answer a mimic multiplier with the whole articulation. With the row
    compliance taken from the row's own body blocks, the multiplier overshot and these
    mechanisms came apart by 0.6 m to tens of meters within 30 steps.
    """
    for topology in ("serial", "serial_far", "gripper", "two_followers", "free_reference"):
        _check_against_local_solve(test, device, topology)


def test_vbd_mimic_block_sparse_many_articulations(test, device):
    """Solve the mimic rows of several articulations per launch, also with per-level launches."""
    for solver in _check_against_local_solve(test, device, "gripper", copies=8):
        plan = solver._rigid_articulation_level_launch_plan
        if plan is not None:
            # The fingers of the eight grippers fill a level that is factored by per-level launches.
            test.assertGreater(plan.leading_level_count, 0)
        test.assertEqual(len(solver._mimic_solver._articulation_colors), 1)
    # Articulations that follow the same body outside the articulations are solved one after another.
    for solver in _check_against_local_solve(test, device, "shared_free_reference", copies=3):
        test.assertEqual(len(solver._mimic_solver._articulation_colors), 3)


def test_vbd_mimic_block_sparse_kinematic_reference(test, device):
    """Follow a prescribed reference over several turns in the block-sparse modes."""
    for solve in ("block_sparse_joints", "block_sparse_levels"):
        for d6 in (False, True):
            with test.subTest(solve=solve, d6=d6):
                rig = _MimicRig(device, d6=d6, solve=solve)
                for step in range(100):
                    rig.step(0.1, flip_sign=step % 2 == 0)
                    rig.check()


def test_vbd_mimic_block_sparse_graph_replay(test, device):
    """Replay captured block-sparse steps with a mimic between jointed bodies."""
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")
    for options in _sparse_solves(device):
        with test.subTest(**options):
            results = []
            for capture in (False, True):
                model, _hinges, _rows = _build_jointed_mimic_mechanism(device, "gripper", copies=8)
                solver = newton.solvers.SolverVBD(model, iterations=16, rigid_compliant_alm=True, **options)
                state, next_state = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                # Two steps return the result to ``state``.
                solver.step(state, next_state, None, None, 4.0e-3)
                solver.step(next_state, state, None, None, 4.0e-3)
                if capture:
                    # Capturing records the two steps without running them.
                    with wp.ScopedCapture(device=device) as scope:
                        solver.step(state, next_state, None, None, 4.0e-3)
                        solver.step(next_state, state, None, None, 4.0e-3)
                    for _ in range(14):
                        wp.capture_launch(scope.graph)
                else:
                    for _ in range(14):
                        solver.step(state, next_state, None, None, 4.0e-3)
                        solver.step(next_state, state, None, None, 4.0e-3)
                results.append(state.joint_q.numpy())
            np.testing.assert_allclose(results[1], results[0], atol=1.0e-5)


class TestSolverVBDMimic(unittest.TestCase):
    pass


for test_function in (
    test_vbd_mimic_multiturn_translation,
    test_vbd_mimic_fractional_angular_ratio,
    test_vbd_mimic_multiturn_dynamics,
    test_vbd_mimic_initial_turns,
    test_vbd_mimic_d6_coordinates,
    test_vbd_mimic_masked_reset,
    test_vbd_mimic_disabled_history,
    test_vbd_mimic_graph_replay,
    test_vbd_mimic_block_sparse_jointed_bodies,
    test_vbd_mimic_block_sparse_many_articulations,
    test_vbd_mimic_block_sparse_kinematic_reference,
    test_vbd_mimic_block_sparse_graph_replay,
):
    add_function_test(TestSolverVBDMimic, test_function.__name__, test_function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
