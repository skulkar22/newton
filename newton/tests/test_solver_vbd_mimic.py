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


def _build_jointed_mimic_mechanism(device, topology, copies=1, extras=None):
    """Build hanging hinge mechanisms whose mimic rows act on bodies that joints couple to other bodies.

    Each of the ``copies`` is its own articulation. Returns the model, the hinges as
    ``(parent, child, parent anchor)`` and the mimic rows as
    ``(follower coordinate, reference coordinate, offset, ratio)``. If ``extras`` is a dict, it
    receives ``"joint_f"`` (generalized forces by DOF), ``"contact"`` (whether the model has a
    ground plane) and ``"control"`` (whether the run needs the model's control for its drives).
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    inertia = wp.mat33(np.eye(3) * 1.0e-2)
    hinges = []
    mimics = []
    initial = []
    joint_f = {}
    needs_control = False
    contact = False

    def link(index, **kwargs):
        kwargs.setdefault("mass", 1.0 + index)
        kwargs.setdefault("com", wp.vec3(0.0, 0.0, -0.25))
        kwargs.setdefault("inertia", inertia)
        return builder.add_link(**kwargs)

    def hinge(parent, child, anchor, axis=newton.Axis.Y, **kwargs):
        hinges.append((parent, child, anchor))
        return builder.add_joint_revolute(
            parent, child, axis=axis, parent_xform=wp.transform(anchor, wp.quat_identity()), **kwargs
        )

    def chain(count, x, z=1.0, joint_options=None):
        joint_options = joint_options or {}
        links = [link(i) for i in range(count)]
        return [
            hinge(
                -1 if i == 0 else links[i - 1],
                links[i],
                (x, 0.0, z) if i == 0 else (0.0, 0.0, -0.5),
                **joint_options.get(i, {}),
            )
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
        elif topology == "chain":
            # The serial chain without a mimic.
            joints = chain(3, x)
            builder.add_articulation(joints)
            coordinates = [0.4, 0.2, 0.2]
        elif topology == "lead_screw":
            # A lead screw on a swinging base: the screw turns on the base, the nut slides along the
            # screw (prismatic follows revolute, 0.01 m/rad) and carries a hanging load. A torque
            # drives the screw.
            base = link(1)
            screw = link(0, mass=0.5, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3) * 1.0e-3))
            nut = link(0, mass=1.0, com=wp.vec3(0.02, 0.0, 0.0))
            load = link(1)
            joints = [
                hinge(-1, base, (x, 0.0, 1.0)),
                hinge(base, screw, (0.0, 0.0, -0.5), axis=newton.Axis.Z),
                builder.add_joint_prismatic(
                    screw, nut, axis=newton.Axis.Z, parent_xform=wp.transform((0.0, 0.0, -0.1), wp.quat_identity())
                ),
                hinge(nut, load, (0.0, 0.0, -0.1)),
            ]
            builder.add_articulation(joints)
            mimics.append((joints[2], joints[1], 0.02, 0.01))
            coordinates = [0.3, 0.0, 0.02, 0.2]
            joint_f[builder.joint_qd_start[joints[1]]] = 0.2
        elif topology in ("reference_limit", "follower_limit"):
            # Without the limit the reference joint settles near 0.12 rad and the follower near 0.16 rad,
            # so the reference limit [0.15, 0.25] or the follower limit [0.18, 0.22] is active.
            limit = {"limit_ke": 1.0e4, "limit_kd": 1.0e1}
            if topology == "reference_limit":
                options = {1: {"limit_lower": 0.15, "limit_upper": 0.25, **limit}}
            else:
                options = {2: {"limit_lower": 0.18, "limit_upper": 0.22, **limit}}
            joints = chain(3, x, joint_options=options)
            builder.add_articulation(joints)
            mimics.append((joints[2], joints[1], 0.1, 0.5))
            coordinates = [0.4, 0.2, 0.2]
        elif topology in ("reference_drive", "follower_drive"):
            # A stiff position drive toward -0.3 rad on the reference joint, or on the follower where it
            # fights the mimic.
            drive = {
                "target_ke": 1.0e4,
                "target_kd": 1.0e2,
                "target_pos": -0.3,
                "actuator_mode": newton.JointTargetMode.POSITION,
            }
            joints = chain(3, x, joint_options={1 if topology == "reference_drive" else 2: drive})
            builder.add_articulation(joints)
            mimics.append((joints[2], joints[1], 0.1, 0.5))
            coordinates = [0.4, 0.2, 0.2]
            needs_control = True
        elif topology == "follower_contact":
            # The follower's link carries a sphere that lands on the ground and stays in contact.
            joints = chain(3, x, z=1.45)
            builder.add_shape_sphere(
                builder.joint_child[joints[2]], xform=wp.transform((0.0, 0.0, -0.4), wp.quat_identity()), radius=0.1
            )
            builder.add_articulation(joints)
            mimics.append((joints[2], joints[1], 0.1, 0.5))
            coordinates = [0.4, 0.2, 0.2]
            contact = True
        else:
            raise ValueError(topology)
        initial.extend(zip(joints, coordinates, strict=True))
    for follower, reference, offset, ratio in mimics:
        builder.set_joint_mimic(follower, reference, coeffs=(offset, ratio))
    for joint, coordinate in initial:
        builder.joint_q[builder.joint_q_start[joint]] = coordinate
    if contact:
        builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    if extras is not None:
        extras.update(joint_f=joint_f, contact=contact, control=needs_control)
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
    extras = {}
    model, hinges, rows = _build_jointed_mimic_mechanism(device, topology, copies, extras)
    solver = newton.solvers.SolverVBD(model, iterations=iterations, rigid_compliant_alm=True, **solver_options)
    observables = solver.observables({solver.ObservableKind.SOLVER_RESIDUAL_HISTORY})
    state, next_state = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = None
    if extras["joint_f"] or extras["control"]:
        control = model.control()
        if extras["joint_f"]:
            joint_f = np.zeros(model.joint_dof_count, dtype=np.float32)
            for dof, value in extras["joint_f"].items():
                joint_f[dof] = value
            control.joint_f.assign(joint_f)
    pipeline = contacts = None
    if extras["contact"]:
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()
    separation = 0.0
    for _ in range(30):
        if pipeline is not None:
            pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, 4.0e-3, observables=observables)
        state, next_state = next_state, state
        separation = max(separation, _hinge_separation(state.body_q.numpy(), hinges))
    coordinates = state.joint_q.numpy()
    mimic_error = max(
        (abs(coordinates[f] - offset - ratio * coordinates[r]) for f, r, offset, ratio in rows), default=0.0
    )
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


def _check_against_converged_reference(
    test, device, topology, reference_iterations, coordinate_tolerance, extra_solves=(), iterations=(16, 64)
):
    """Compare the block-sparse solves with a converged local solve and return its coordinates.

    ``coordinate_tolerance`` maps the iteration count to the bound on the joint coordinates [rad or m].
    The mimic error must stay below 1e-6 [rad or m] (measured at most 7.5e-8 on these mechanisms), the
    hinges may not open more than 1e-4 m beyond the reference (measured at most 4.7e-5 m, at 16
    iterations with an active limit, where the local solve opens 5.0e-5 m), and every pose must stay
    finite.
    """
    expected, expected_separation, _error, _residual, _solver = _run_jointed_mimic_mechanism(
        device, topology, reference_iterations, rigid_articulation_solve="local"
    )
    for options in [*_sparse_solves(device), *extra_solves]:
        for count in iterations:
            with test.subTest(topology=topology, iterations=count, **options):
                coordinates, separation, mimic_error, _residual, _solver = _run_jointed_mimic_mechanism(
                    device, topology, count, **options
                )
                test.assertTrue(np.isfinite(coordinates).all())
                test.assertLess(separation, expected_separation + 1.0e-4)
                test.assertLess(mimic_error, 1.0e-6)
                np.testing.assert_allclose(coordinates, expected, atol=coordinate_tolerance[count])
    return expected


def test_vbd_mimic_block_sparse_lead_screw(test, device):
    """Drive a lead screw (prismatic nut following a revolute screw) on a swinging base.

    The local solve converges slowly on this serial mechanism: after 30 steps of 4 ms its screw angle at
    64 iterations is 0.44 rad from its value at 4096 iterations, and at 1024 iterations still 2.0e-4
    (CPU and CUDA); 2048 and 4096 iterations agree within 1.2e-5 (CUDA). The reference is therefore the
    local solve at 4096 iterations. Measured block-sparse difference from it at 16 and 64 iterations:
    at most 5.7e-5 (CPU and CUDA, all backends).
    """
    _check_against_converged_reference(test, device, "lead_screw", 4096, {16: 2.0e-4, 64: 2.0e-4})


def test_vbd_mimic_block_sparse_active_limits(test, device):
    """Hold the reference joint, or the follower, at an active joint limit.

    Reference: the local solve at 1024 iterations (within 3.1e-7 of 4096 iterations on CPU, 1.6e-6 on
    CUDA). Limits converge slowly in both solves: at 16 iterations the local solve is 2.2e-3 from the
    reference and the block-sparse solves 2.2e-3; at 64 iterations 2.1e-4 and 2.1e-4 (CPU and CUDA).
    """
    expected = _check_against_converged_reference(test, device, "reference_limit", 1024, {16: 6.0e-3, 64: 6.0e-4})
    # Without the limit the reference joint ends near 0.12 rad; the limit holds it at 0.15 rad.
    test.assertGreater(expected[1], 0.149)
    expected = _check_against_converged_reference(test, device, "follower_limit", 1024, {16: 6.0e-3, 64: 6.0e-4})
    # Without the limit the follower ends near 0.16 rad; its limit holds it at 0.18 rad and the mimic
    # moves the reference with it.
    test.assertGreater(expected[2], 0.179)


def test_vbd_mimic_block_sparse_stiff_drive(test, device):
    """Drive the reference joint, or the follower against the mimic, with a stiff position drive.

    Reference: the local solve at 1024 iterations (within 4.3e-6 of 4096 iterations). Measured
    block-sparse difference from it: at most 3.9e-6 (CPU and CUDA).
    """
    expected = _check_against_converged_reference(test, device, "reference_drive", 1024, {16: 5.0e-5, 64: 5.0e-5})
    test.assertLess(expected[1], -0.25)
    expected = _check_against_converged_reference(test, device, "follower_drive", 1024, {16: 5.0e-5, 64: 5.0e-5})
    test.assertLess(expected[2], -0.25)


def test_vbd_mimic_block_sparse_follower_contact(test, device):
    """Land the follower's link on the ground, also with the contact active set.

    Reference: the local solve at 1024 iterations (within 1.1e-5 of 4096 iterations). Measured
    block-sparse difference from it: at most 1.4e-5 (CPU and CUDA, with and without the active set).
    """
    active_set = [{**options, "rigid_articulation_contact_active_set": True} for options in _sparse_solves(device)]
    expected = _check_against_converged_reference(
        test, device, "follower_contact", 1024, {16: 1.0e-4, 64: 1.0e-4}, extra_solves=active_set
    )
    # The ground stops the swing: the same chain without the ground ends 4e-2 rad elsewhere.
    free, _separation, _error, _residual, _solver = _run_jointed_mimic_mechanism(
        device, "serial", 64, rigid_articulation_solve="local"
    )
    test.assertGreater(np.abs(expected - free).max(), 1.0e-2)


def _set_joint_mimic(model, follower, reference, coeffs=(0.1, 0.5)):
    """Change a mimic in the model arrays, as a user does before ``notify_model_changed``."""
    references = model.joint_mimic_joint.numpy()
    references[follower] = reference
    model.joint_mimic_joint.assign(references)
    values = model.joint_mimic_coeffs.numpy()
    values[follower] = coeffs
    model.joint_mimic_coeffs.assign(values)


def test_vbd_mimic_block_sparse_rejected_model_change(test, device):
    """Reject a mimic across two articulations through ``notify_model_changed`` and refuse to step.

    ``ModelBuilder`` rejects such a mimic, so the model arrays are changed directly. The rejected
    change keeps the solver's previous mimic state, and ``step`` raises until a later
    ``notify_model_changed`` accepts the model.
    """
    for solve in ("block_sparse_joints", "block_sparse_levels"):
        with test.subTest(solve=solve):
            model, _hinges, _rows = _build_jointed_mimic_mechanism(device, "chain", copies=2)
            solver = newton.solvers.SolverVBD(
                model, iterations=16, rigid_compliant_alm=True, rigid_articulation_solve=solve
            )
            state, next_state = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            solver.step(state, next_state, None, None, 4.0e-3)
            test.assertIsNone(solver._mimic_solver)
            test.assertEqual(solver._use_body_displacement, 1)

            # Joints 0-2 form the first articulation, joints 3-5 the second.
            _set_joint_mimic(model, 5, 1)
            with test.assertRaisesRegex(ValueError, "two different articulations"):
                solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
            test.assertFalse(solver._has_joint_mimics)
            test.assertIsNone(solver._mimic_solver)
            test.assertEqual(solver._use_body_displacement, 1)
            with test.assertRaisesRegex(RuntimeError, "rejected the model's joint mimics"):
                solver.step(next_state, state, None, None, 4.0e-3)

            # A mimic inside the second articulation is accepted and enforced.
            _set_joint_mimic(model, 5, -1)
            _set_joint_mimic(model, 5, 4)
            solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
            test.assertTrue(solver._has_joint_mimics)
            test.assertIsNotNone(solver._mimic_solver)
            test.assertEqual(solver._use_body_displacement, 0)
            for _ in range(30):
                solver.step(state, next_state, None, None, 4.0e-3)
                state, next_state = next_state, state
            coordinates = state.joint_q.numpy()
            # Measured 2.2e-8 to 6.7e-8 rad on CPU and CUDA; without the mimic this error is 2.1e-2 rad.
            test.assertLess(abs(coordinates[5] - 0.1 - 0.5 * coordinates[4]), 1.0e-6)


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
    test_vbd_mimic_block_sparse_rejected_model_change,
    test_vbd_mimic_block_sparse_lead_screw,
    test_vbd_mimic_block_sparse_active_limits,
    test_vbd_mimic_block_sparse_stiff_drive,
    test_vbd_mimic_block_sparse_follower_contact,
):
    add_function_test(TestSolverVBDMimic, test_function.__name__, test_function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
