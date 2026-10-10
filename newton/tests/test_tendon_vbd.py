# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for routed tendons in the VBD rigid solver."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.examples.cable.example_tendon_3d_routing import Example as Routing3DExample
from newton.examples.cable.example_tendon_cable_machine import Example as CableMachineExample
from newton.examples.cable.example_tendon_capstan_friction import Example as DynamicCapstanExample
from newton.examples.cable.example_tendon_capstan_kinematic import Example as KinematicCapstanExample
from newton.examples.cable.example_tendon_compound_pulley import Example as CompoundPulleyExample
from newton.examples.cable.example_tendon_equilibrium import Example as EquilibriumExample
from newton.examples.cable.example_tendon_gear_pulley import Example as GearPulleyExample
from newton.examples.cable.example_tendon_mujoco_switch import Example as MujocoSwitchExample
from newton.examples.cable.example_tendon_pinhole import Example as PinholeExample
from newton.examples.cable.example_tendon_pulley import Example as PulleyExample
from newton.examples.cable.example_tendon_rolling_pulley import Example as RollingPulleyExample
from newton.examples.cable.example_tendon_xy_table import Example as XYTableExample
from newton.examples.cable.example_tendon_xy_table import _desired_table_xy
from newton.tests.test_tendon_capstan import (
    SIGMOID_TENDON_MATERIAL,
    _moving_rolling_route_material_error,
    _oriented_route_material_length,
    build_dynamic_pulley_atwood,
    build_force_driven_dynamic_route,
    build_kinematic_pulley_atwood,
    build_kinematic_rolling_transport,
    build_motorized_pulley_drive,
    build_pinhole_atwood,
    build_sigmoid_pinhole_route,
    build_sigmoid_single_span,
    build_simple_cable_gravity,
    build_slack_pinhole_route,
    sigmoid_tendon_tangent,
    sigmoid_tendon_tension,
)
from newton.tests.test_tendon_equilibrium import build_atwood_equal_weights
from newton.tests.unittest_utils import find_nan_members, sanitize_identifier

TENDON_VBD_SOLVER_KWARGS = {
    "iterations": 20,
    "tendon_max_sweeps": 32,
    "rigid_avbd_beta": 1.0e6,
    "rigid_joint_linear_k_start": 1.0e7,
    "rigid_joint_angular_k_start": 1.0e5,
    "rigid_joint_linear_kd": 5.0e-2,
    "rigid_joint_angular_kd": 2.0e-2,
}


def add_test(cls, name, devices, test_fn):
    for device in devices:
        test_name = f"test_{sanitize_identifier(name)}_{sanitize_identifier(device)}"

        def test_method(self, d=device, fn=test_fn):
            return fn(self, d)

        expected_failure_devices = getattr(test_fn, "__unittest_expected_failure_devices__", set())
        if getattr(test_fn, "__unittest_expecting_failure__", False) or device in expected_failure_devices:
            test_method = unittest.expectedFailure(test_method)
        setattr(cls, test_name, test_method)


def _set_serial_body_coloring(model):
    model.body_color_groups = [
        wp.array([body], dtype=wp.int32, device=model.device) for body in range(model.body_count)
    ]


def _hinge_y_angle(body_q, body_idx):
    q = body_q[body_idx]
    return float(2.0 * np.arctan2(float(q[4]), float(q[6])))


def _hinge_z_angle(body_q, body_idx):
    q = body_q[body_idx]
    return float(2.0 * np.arctan2(float(q[5]), float(q[6])))


# Compliant ALM (the default rigid path) ignores the legacy k_start/beta knobs in TENDON_VBD_SOLVER_KWARGS and
# uses rigid_joint_linear_ke=1e5 N/m, which lets the 500 kg rolling-pulley revolute anchor sag ~5-10 cm.
STIFF_REVOLUTE_ANCHOR_OVERRIDES = {"rigid_joint_linear_ke": 1.0e7, "rigid_joint_angular_ke": 1.0e5}


def _make_tendon_vbd_solver(model, solver_overrides=None):
    return newton.solvers.SolverVBD(model, **{**TENDON_VBD_SOLVER_KWARGS, **(solver_overrides or {})})


def _make_example_args(num_frames):
    return type(
        "Args",
        (),
        {"headless": True, "record": False, "num_frames": num_frames, "episode_frames": None, "test": True},
    )()


def _install_vbd_example_solver(example, solver_overrides=None):
    _set_serial_body_coloring(example.model)
    solver_kwargs = dict(TENDON_VBD_SOLVER_KWARGS)

    is_xy_table = all(hasattr(example, attr) for attr in ("p2_dof_start", "p6_dof_start", "_apply_cable_pretension"))
    if is_xy_table:
        example.model.joint_target_kd[example.p2_dof_start : example.p2_dof_start + 1].fill_(200.0)
        example.model.joint_target_kd[example.p6_dof_start : example.p6_dof_start + 1].fill_(200.0)
        solver_kwargs["iterations"] = 30

    if solver_overrides is not None:
        solver_kwargs.update(solver_overrides)
    example.solver = newton.solvers.SolverVBD(example.model, **solver_kwargs)
    if is_xy_table:
        example._apply_cable_pretension(0.99995)


def _assert_example_members_have_no_nans(test, example):
    for member_name in ("state_0", "state_1", "model", "control", "contacts"):
        if not hasattr(example, member_name):
            continue
        nan_members = find_nan_members(getattr(example, member_name))
        test.assertFalse(nan_members, f"NaN members found in {member_name}: {nan_members}")


def _run_vbd_example_assertions(test, device, example_cls, num_frames, solver_overrides=None):
    with wp.ScopedDevice(device):
        example = example_cls(None, _make_example_args(num_frames))
        _install_vbd_example_solver(example, solver_overrides)

        has_post_step = hasattr(example, "test_post_step")
        has_final = hasattr(example, "test_final")
        if not (has_post_step or has_final):
            test.fail(f"{example_cls.__module__}.Example does not define example assertions")

        for _ in range(num_frames):
            example.step()
            if has_post_step:
                example.test_post_step()

        if has_final:
            example.test_final()
        _assert_example_members_have_no_nans(test, example)


def _make_vbd_example_assertion_test(example_cls, num_frames, *, skip_cpu=False, solver_overrides=None):
    def test_fn(test, device, cls=example_cls, frames=num_frames, cpu_skip=skip_cpu, overrides=solver_overrides):
        if cpu_skip and not wp.get_device(device).is_cuda:
            test.skipTest("VBD example-assertion regression is CUDA-only because the scene is expensive")
        _run_vbd_example_assertions(test, device, cls, frames, overrides)

    return test_fn


def build_single_span_tendon():
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))

    anchor = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)),
        mass=0.0,
        is_kinematic=True,
    )
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),
        mass=1.0,
    )
    builder.add_shape_sphere(anchor, radius=0.01)
    builder.add_shape_sphere(body, radius=0.01)

    builder.add_tendon()
    builder.add_tendon_link(
        body=anchor,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
    )
    builder.add_tendon_link(
        body=body,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        compliance=1.0e-5,
        damping=0.0,
        rest_length=0.5,
    )

    builder.color()
    model = builder.finalize()
    return model, body


def build_fixed_rolling_chain(compliance=1.0e-5):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    positions = [(-1.0, 0.0, 0.0), (0.0, 0.25, 0.0), (1.0, -0.25, 0.0), (2.0, 0.25, 0.0)]
    bodies = [
        builder.add_body(mass=0.0, is_kinematic=True),
        builder.add_body(mass=0.0, is_kinematic=True),
        builder.add_body(mass=0.0, is_kinematic=True),
        builder.add_body(
            xform=wp.transform(p=wp.vec3(*positions[-1])),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        ),
    ]

    builder.add_tendon()
    builder.add_tendon_link(
        body=bodies[0],
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=positions[0],
    )
    builder.add_tendon_link(
        body=bodies[1],
        link_type=int(TendonLinkType.ROLLING),
        radius=0.1,
        orientation=-1,
        offset=positions[1],
        compliance=compliance,
    )
    builder.add_tendon_link(
        body=bodies[2],
        link_type=int(TendonLinkType.ROLLING),
        radius=0.1,
        orientation=1,
        offset=positions[2],
        compliance=compliance,
    )
    builder.add_tendon_link(
        body=bodies[3],
        link_type=int(TendonLinkType.ATTACHMENT),
        compliance=compliance,
    )
    builder.color()
    return builder.finalize(), bodies[-1]


def build_dynamic_rolling_route():
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
    lower = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.5)),
        mass=0.0,
        is_kinematic=True,
    )
    upper = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5)),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    candidate = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.25, 0.0, 0.0)),
        mass=0.0,
        is_kinematic=True,
    )

    builder.add_tendon()
    builder.add_tendon_link(body=lower, link_type=int(TendonLinkType.ATTACHMENT), axis=(0.0, 1.0, 0.0))
    candidate_link = builder.add_tendon_link(
        body=candidate,
        link_type=int(TendonLinkType.ROLLING),
        radius=0.1,
        orientation=1,
        dynamic=True,
        axis=(0.0, 1.0, 0.0),
        compliance=1.0e-4,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=upper,
        link_type=int(TendonLinkType.ATTACHMENT),
        axis=(0.0, 1.0, 0.0),
        compliance=1.0e-4,
        rest_length=-1.0,
    )
    builder.color()
    return builder.finalize(), candidate, candidate_link, lower, upper


def run_vbd_model(model, num_frames=80, substeps=12, fps=60):
    _set_serial_body_coloring(model)
    dt = 1.0 / fps / substeps
    solver = _make_tendon_vbd_solver(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

    return state_0, solver


def run_vbd_motorized_model(model, drive_joint, target=2.0, num_frames=70, substeps=10, fps=60):
    _set_serial_body_coloring(model)
    dt = 1.0 / fps / substeps
    solver = _make_tendon_vbd_solver(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    q_start = int(model.joint_q_start.numpy()[drive_joint])
    for _ in range(num_frames):
        control.joint_target_q[q_start : q_start + 1].fill_(target)
        for _ in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

    return state_0, solver


def _run_dynamic_pulley(mu, device, num_frames=40):
    model, left_idx, right_idx, pulley_idx = build_dynamic_pulley_atwood(mu=mu)
    _set_serial_body_coloring(model)

    solver = _make_tendon_vbd_solver(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    substeps = 12
    dt = 1.0 / 60.0 / substeps
    last_angle = _hinge_y_angle(state_0.body_q.numpy(), pulley_idx)
    theta = 0.0

    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

        angle = _hinge_y_angle(state_0.body_q.numpy(), pulley_idx)
        theta += float((angle - last_angle + np.pi) % (2.0 * np.pi) - np.pi)
        last_angle = angle

    body_q = state_0.body_q.numpy()
    return body_q, left_idx, right_idx, theta


def _dynamic_capstan_metrics(device, mu, num_frames=40):
    model, left_idx, right_idx, pulley_idx = build_dynamic_pulley_atwood(mu=mu)
    _set_serial_body_coloring(model)
    substeps = 12
    dt = 1.0 / 60.0 / substeps
    solver = _make_tendon_vbd_solver(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    body_q = state_0.body_q.numpy()
    last_angle = _hinge_y_angle(body_q, pulley_idx)
    theta = 0.0

    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        angle = _hinge_y_angle(body_q, pulley_idx)
        theta += float((angle - last_angle + np.pi) % (2.0 * np.pi) - np.pi)
        last_angle = angle

    left_travel = float(body_q[left_idx][2]) - 2.0
    right_travel = 2.0 - float(body_q[right_idx][2])
    cable_travel = 0.5 * (left_travel + right_travel)
    radius = float(model.tendon_link_radius.numpy()[1])
    rim_travel = theta * radius
    slip = abs(cable_travel - rim_travel)
    return body_q, left_travel, right_travel, cable_travel, theta, rim_travel, slip


def _dynamic_capstan_example_metrics(device, num_frames=40):
    example = DynamicCapstanExample(None, None)
    _set_serial_body_coloring(example.model)
    example.solver = _make_tendon_vbd_solver(example.model)
    for _ in range(num_frames):
        example.step()
    theta = np.array(example._pulley_rotation_history[-1], dtype=np.float64)
    left_z = np.array(example._left_z_history[-1], dtype=np.float64)
    right_z = np.array(example._right_z_history[-1], dtype=np.float64)
    cable_travel = 0.5 * ((left_z - 2.0) + (2.0 - right_z))
    rim_travel = theta * example.pulley_radius
    slip = np.abs(cable_travel - rim_travel)
    return tuple(example.mus), theta, cable_travel, rim_travel, slip


def _dynamic_capstan_example_history(device, num_frames=100):
    example = DynamicCapstanExample(None, None)
    _set_serial_body_coloring(example.model)
    example.solver = _make_tendon_vbd_solver(example.model)
    for _ in range(num_frames):
        example.step()

    theta = np.asarray(example._pulley_rotation_history, dtype=np.float64)
    left_z = np.asarray(example._left_z_history, dtype=np.float64)
    right_z = np.asarray(example._right_z_history, dtype=np.float64)
    return example, theta, left_z, right_z


def _kinematic_capstan_metrics(device, mu, num_frames=40):
    model, left_idx, right_idx, pulley_idx = build_kinematic_pulley_atwood(mu=mu)
    state, _ = run_vbd_model(model, num_frames=num_frames)
    body_q = state.body_q.numpy()
    left_travel = float(body_q[left_idx][2]) - 2.0
    right_travel = 2.0 - float(body_q[right_idx][2])
    theta = _hinge_y_angle(body_q, pulley_idx)
    return body_q, left_travel, right_travel, theta


class TestTendonVBD(unittest.TestCase):
    pass


def test_vbd_tendon_stretch_pulls_toward_anchor(test, device):
    """A taut VBD tendon segment should pull the dynamic endpoint toward rest length."""
    with wp.ScopedDevice(device):
        model, body_idx = build_single_span_tendon()
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        initial_x = float(state_0.body_q.numpy()[body_idx][0])
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        final_x = float(state_1.body_q.numpy()[body_idx][0])

        test.assertLess(final_x, initial_x - 0.1, f"Tendon should pull endpoint inward: {initial_x} -> {final_x}")
        test.assertTrue(np.isfinite(state_1.body_q.numpy()).all(), "Non-finite VBD single-span state")


def test_vbd_tendon_zero_compliance_uses_floor(test, device):
    """VBD should approximate zero tendon compliance with its stiffness floor."""
    with wp.ScopedDevice(device):
        results = []
        for compliance in (0.0, 1.0e-8):
            model, body_idx = build_single_span_tendon()
            model.tendon_seg_compliance.fill_(compliance)
            solver = _make_tendon_vbd_solver(model)
            state_0 = model.state()
            state_1 = model.state()
            solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
            results.append(
                (
                    state_1.body_q.numpy()[body_idx].copy(),
                    float(solver.tendon_seg_active_compliance.numpy()[0]),
                )
            )

        test.assertTrue(np.isfinite(results[0][0]).all())
        np.testing.assert_allclose(results[0][0], results[1][0], rtol=0.0, atol=1.0e-6)
        test.assertEqual(results[0][1], results[1][1])


def test_vbd_rolling_chain_tension_matches_accepted_pose(test, device):
    """VBD material relaxation should converge at the final accepted rigid pose."""
    with wp.ScopedDevice(device):
        model, slider = build_fixed_rolling_chain()
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()

        direction = solver.tendon_seg_attachment_r.numpy()[-1] - solver.tendon_seg_attachment_l.numpy()[-1]
        direction /= np.linalg.norm(direction)
        body_f = np.zeros((model.body_count, 6), dtype=np.float32)
        body_f[slider, :3] = 10.0 * direction
        state_0.body_f.assign(wp.array(body_f, dtype=wp.spatial_vector))
        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)

        lengths = np.linalg.norm(
            solver.tendon_seg_attachment_r.numpy() - solver.tendon_seg_attachment_l.numpy(), axis=1
        )
        rest = solver.tendon_seg_rest_length.numpy()
        compliance = solver.tendon_seg_active_compliance.numpy()
        tension = np.maximum(lengths - rest, 0.0) / compliance
        # At this stiffness a single float32 length ULP is already about
        # 0.012 N. Include length/rest quantization and the configured relative
        # material tolerance instead of demanding a sub-ULP tension spread.
        tension_quantum = np.maximum(np.spacing(lengths), np.spacing(rest)) / compliance
        tension_tolerance = 2.0 * np.max(tension_quantum) + solver.tendon_settle_tol * np.max(tension)
        test.assertGreater(np.min(tension), 1.0, f"Every rolling-chain span should carry load: {tension}")
        test.assertLess(
            np.ptp(tension), tension_tolerance, f"Frictionless rolling-chain tensions did not converge: {tension}"
        )
        # One free mass, initially at rest, is pulled along the last tangent.
        # The three compliant spans act in series: T = F / (1 + m sum(c) / dt^2).
        expected_tension = 10.0 / (1.0 + float(model.body_mass.numpy()[slider]) * np.sum(compliance) * 60.0**2)
        np.testing.assert_allclose(tension, expected_tension, rtol=0.0, atol=tension_tolerance)
        np.testing.assert_allclose(
            solver.tendon_seg_material_tension.numpy(), tension, rtol=0.0, atol=np.max(tension_quantum)
        )

        # The two fixed rollers bend in opposite directions. The regression
        # must not obtain its result from an unsupported negative wrap.
        centers = model.tendon_link_offset.numpy()[1:-1]
        incoming = solver.tendon_seg_attachment_r.numpy()[:-1] - centers
        outgoing = solver.tendon_seg_attachment_l.numpy()[1:] - centers
        wrap = np.arctan2(np.cross(incoming, outgoing)[:, 2], np.sum(incoming * outgoing, axis=1))
        wrap *= model.tendon_link_orientation.numpy()[1:-1]
        test.assertTrue(np.all((wrap > 0.0) & (wrap <= np.pi)), f"Invalid fixture wraps: {wrap}")
        test.assertGreater(
            int(solver.tendon_cone_sweep_count.numpy()[0]),
            4,
            "The regression must exercise adaptive material relaxation beyond the legacy VBD sweep count",
        )


def test_vbd_dynamic_tendon_route_switches_inside_solver(test, device):
    """VBD should update dynamic routing and its active segment endpoints inside ``step``."""
    with wp.ScopedDevice(device):
        model, candidate, candidate_link, lower, upper = build_dynamic_rolling_route()
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        rest_length = solver.tendon_seg_rest_length.numpy()
        rest_length[0] -= 0.02
        solver.tendon_seg_rest_length.assign(rest_length)

        body_colors = np.full(model.body_count, -1, dtype=np.int32)
        for color, group in enumerate(model.body_color_groups):
            body_colors[group.numpy()] = color
        test.assertNotEqual(body_colors[lower], body_colors[upper], "The possible bypass span must be colored")

        body_q = state_0.body_q.numpy()
        active_history = []
        segment_history = []
        material_tension_history = []
        first_upper_z = None
        for x in (0.25, 0.05, -0.25, 0.25):
            body_q[candidate, :3] = (x, 0.0, 0.0)
            state_0.body_q.assign(body_q)
            solver.step(state_0, state_1, control, None, 1.0 / 120.0)
            active_history.append(bool(solver.tendon_link_active.numpy()[candidate_link]))
            segment_history.append(solver.tendon_seg_active.numpy().tolist())
            material_tension_history.append(solver.tendon_seg_material_tension.numpy().tolist())
            if first_upper_z is None:
                first_upper_z = float(state_1.body_q.numpy()[upper][2])
            state_0, state_1 = state_1, state_0
            body_q = state_0.body_q.numpy()

        test.assertEqual(active_history, [False, True, True, False])
        test.assertEqual(segment_history, [[1, 0], [1, 1], [1, 1], [1, 0]])
        test.assertEqual(material_tension_history[0][1], 0.0)
        test.assertEqual(material_tension_history[-1][1], 0.0)
        test.assertLess(first_upper_z, 0.499, "The inactive bypass span must load its new right endpoint")
        test.assertTrue(np.isfinite(state_0.body_q.numpy()).all())


def test_vbd_dynamic_route_activation_uses_accepted_pose(test, device):
    """Route activation should not use the VBD inertial predictor pose."""
    with wp.ScopedDevice(device):
        model, candidate, candidate_link = build_force_driven_dynamic_route(device)
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        test.assertFalse(solver.tendon_link_active.numpy()[candidate_link])
        body_qd = state_0.body_qd.numpy()
        body_qd[candidate, 0] = -24.0
        state_0.body_qd.assign(body_qd)

        solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)

        test.assertLess(abs(float(state_1.body_q.numpy()[candidate, 0])), 0.1)
        test.assertFalse(
            solver.tendon_link_active.numpy()[candidate_link],
            "The predictor crossed the cable, but activation belongs to the accepted step-start pose",
        )


def test_vbd_tendon_coloring_separates_segment_endpoints(test, device):
    """Rigid-body coloring should include tendon force-element connectivity."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body_0 = builder.add_body(mass=1.0)
        body_1 = builder.add_body(mass=1.0)
        builder.add_shape_sphere(body_0, radius=0.01)
        builder.add_shape_sphere(body_1, radius=0.01)
        builder.add_tendon()
        builder.add_tendon_link(body=body_0, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=body_1,
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=1.0e-3,
        )
        builder.color()
        model = builder.finalize()

        body_colors = np.full(model.body_count, -1, dtype=int)
        for color, group in enumerate(model.body_color_groups):
            body_colors[group.numpy()] = color
        test.assertNotEqual(body_colors[body_0], body_colors[body_1])
        _make_tendon_vbd_solver(model)

        model.body_color_groups = [wp.array([body_0, body_1], dtype=wp.int32, device=model.device)]
        with test.assertRaisesRegex(ValueError, "does not separate tendon segment endpoints"):
            _make_tendon_vbd_solver(model)


def test_vbd_tendon_compliance_uses_physical_tension(test, device):
    """A compliant VBD tendon should settle to the same physical equilibrium at different time steps."""
    with wp.ScopedDevice(device):
        results = []
        for dt in (1.0 / 60.0, 1.0 / 120.0):
            model, body_idx, mass, compliance, initial_z = build_simple_cable_gravity()
            model.tendon_seg_damping.fill_(100.0)
            _set_serial_body_coloring(model)
            solver = _make_tendon_vbd_solver(model)
            state_0 = model.state()
            state_1 = model.state()
            control = model.control()
            contacts = model.contacts()

            load = mass * 9.81
            expected_z = initial_z - load * compliance
            for _ in range(round(1.5 / dt)):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

            attachment_l = solver.tendon_seg_attachment_l.numpy()[0]
            attachment_r = solver.tendon_seg_attachment_r.numpy()[0]
            rest_length = float(solver.tendon_seg_rest_length.numpy()[0])
            stretch = float(np.linalg.norm(attachment_r - attachment_l)) - rest_length
            material_tension = stretch / compliance
            final_z = float(state_0.body_q.numpy()[body_idx][2])
            final_vz = float(state_0.body_qd.numpy()[body_idx][2])
            results.append((material_tension, final_z, final_vz))

            test.assertAlmostEqual(
                material_tension,
                load,
                delta=0.05 * load,
                msg=f"VBD material tension should balance the load: {material_tension:.3f} vs {load:.3f}",
            )
            test.assertAlmostEqual(
                final_z,
                expected_z,
                delta=5.0e-3,
                msg=f"VBD equilibrium position changed: {final_z:.6f} vs {expected_z:.6f}",
            )
            test.assertAlmostEqual(final_vz, 0.0, delta=2.0e-2, msg=f"VBD equilibrium velocity changed: {final_vz:.6f}")

        test.assertAlmostEqual(
            results[0][0],
            results[1][0],
            delta=0.05 * mass * 9.81,
            msg=f"VBD material tension should be time-step independent: {results}",
        )


def test_vbd_sigmoid_material_projection_matches_capstan_cone(test, device):
    """VBD material transfer must use the nonlinear law selected for its body solve."""
    with wp.ScopedDevice(device):
        mu = 0.1
        material = SIGMOID_TENDON_MATERIAL
        model, initial_rest = build_sigmoid_pinhole_route(mu)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            tendon_max_sweeps=32,
            rigid_avbd_beta=1.0e6,
            **material,
        )
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)

        lengths = np.linalg.norm(
            solver.tendon_seg_attachment_r.numpy() - solver.tendon_seg_attachment_l.numpy(), axis=1
        )
        rest = solver.tendon_seg_rest_length.numpy()
        tension = sigmoid_tendon_tension(lengths, rest, material)
        target_ratio = np.exp(mu * np.pi / 2.0)

        np.testing.assert_allclose(np.sum(rest), np.sum(initial_rest), rtol=1.0e-6, atol=1.0e-6)
        test.assertAlmostEqual(float(tension[0] / tension[1]), target_ratio, delta=0.01)


def test_vbd_sigmoid_stretch_uses_material_tangent(test, device):
    """One VBD iteration should match the nonlinear force and tangent solve."""
    with wp.ScopedDevice(device):
        material = SIGMOID_TENDON_MATERIAL
        model, body, rest_length = build_sigmoid_single_span()
        solver = newton.solvers.SolverVBD(model, iterations=1, rigid_avbd_beta=1.0e6, **material)
        state_0, state_1 = model.state(), model.state()
        dt = 1.0 / 60.0

        solver.step(state_0, state_1, model.control(), None, dt)

        tension = sigmoid_tendon_tension(1.0, rest_length, material)
        tangent = sigmoid_tendon_tangent(1.0, rest_length, material)
        mass = float(model.body_mass.numpy()[body])
        expected_x = 1.0 - tension / (mass / (dt * dt) + tangent)
        expected_tension = sigmoid_tendon_tension(expected_x, rest_length, material)

        test.assertAlmostEqual(float(state_1.body_q.numpy()[body][0]), expected_x, delta=1.0e-5)
        np.testing.assert_allclose(solver.tendon_seg_material_tension.numpy()[0], expected_tension, rtol=2.0e-5)


def test_vbd_quasi_static_roller_conserves_cable_inventory(test, device):
    """A taut cable over a near-static dynamic roller must not leak inventory step by step.

    The roller sits off its body origin on a revolute arm, so its contact points round-trip through
    float32 body-local coordinates every step. Rolling transfer must telescope across steps instead of
    re-booking that round-trip error as cable sliding over the roller.
    """
    with wp.ScopedDevice(device):
        for arm_angle in (0.83, 2.21):
            radius = 0.006
            pivot = (-0.0612, 0.0237)
            offset = 0.0473
            builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
            q = wp.quat(0.0, 0.0, float(np.sin(0.5 * arm_angle)), float(np.cos(0.5 * arm_angle)))
            arm = builder.add_link(
                xform=wp.transform(p=wp.vec3(pivot[0], pivot[1], 0.0), q=q),
                mass=0.02,
                inertia=wp.mat33(np.eye(3) * 2.0e-6),
            )
            joint = builder.add_joint_revolute(
                parent=-1,
                child=arm,
                axis=(0.0, 0.0, 1.0),
                limit_lower=arm_angle - 0.05,
                limit_upper=arm_angle + 0.05,
                limit_ke=1.0e4,
                limit_kd=5.0,
                parent_xform=wp.transform(p=wp.vec3(pivot[0], pivot[1], 0.0), q=q),
            )
            builder.add_articulation([joint])
            center = (pivot[0] + np.cos(arm_angle) * offset, pivot[1] + np.sin(arm_angle) * offset)
            anchors = [
                builder.add_body(
                    xform=wp.transform(p=wp.vec3(center[0] + side * 0.04, center[1] - radius - 0.003, 0.0)),
                    mass=0.0,
                    is_kinematic=True,
                )
                for side in (-1.0, 1.0)
            ]
            builder.add_tendon()
            builder.add_tendon_link(body=anchors[0], link_type=int(TendonLinkType.ATTACHMENT), axis=(0.0, 0.0, 1.0))
            builder.add_tendon_link(
                body=arm,
                link_type=int(TendonLinkType.ROLLING),
                radius=radius,
                orientation=1,
                mu=0.25,
                offset=(offset, 0.0, 0.0),
                axis=(0.0, 0.0, 1.0),
                compliance=2.0e-5,
                damping=5.0,
                rest_length=0.038,
            )
            builder.add_tendon_link(
                body=anchors[1],
                link_type=int(TendonLinkType.ATTACHMENT),
                axis=(0.0, 0.0, 1.0),
                compliance=2.0e-5,
                damping=5.0,
                rest_length=0.038,
            )
            model = builder.finalize()
            _set_serial_body_coloring(model)
            solver = newton.solvers.SolverVBD(
                model,
                iterations=32,
                tendon_alm=True,
                tendon_alm_per_segment=True,
                tendon_material_direct=True,
                tendon_alm_min_stiffness_ratio=0.0,
            )
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
            dt = 1.0 / 1200.0
            for _ in range(600):
                solver.step(state_0, state_1, model.control(), None, dt)
                state_0, state_1 = state_1, state_0
            test.assertTrue(bool(solver.tendon_link_active.numpy()[1]), "The roller should stay wrapped")
            inventory_0 = _oriented_route_material_length(solver, model, state_0, 1)
            for _ in range(3000):
                solver.step(state_0, state_1, model.control(), None, dt)
                state_0, state_1 = state_1, state_0
            drift = _oriented_route_material_length(solver, model, state_0, 1) - inventory_0
            # The pre-fix world-frame transfer leaked ~1.5-5 nm per step here (several um over 3000 steps).
            test.assertLess(abs(drift), 1.0e-6, f"arm_angle={arm_angle}: cable inventory drifted by {drift:.3e} m")


def test_vbd_slack_damped_tendon_remains_force_free(test, device):
    """Damping must not make a slack tendon transmit tension."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        anchor = builder.add_body(mass=0.0, is_kinematic=True)
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.25, 0.0, 0.0)),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_tendon()
        builder.add_tendon_link(body=anchor, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=body,
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=1.0e-3,
            damping=100.0,
            rest_length=1.0,
        )
        builder.color()
        model = builder.finalize()
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        state_0.body_qd.assign(
            [
                wp.spatial_vector(0.0),
                wp.spatial_vector(30.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ]
        )

        dt = 1.0 / 120.0
        solver.step(state_0, state_1, model.control(), None, dt)

        final_vx = float(state_1.body_qd.numpy()[body][0])
        material_tension = float(solver.tendon_seg_material_tension.numpy()[0])
        reported_lambda = float(solver.tendon_seg_lambda.numpy()[0])
        test.assertAlmostEqual(final_vx, 30.0, delta=1.0e-4)
        test.assertEqual(material_tension, 0.0)
        test.assertEqual(reported_lambda, 0.0, "VBD must not report a synthetic XPBD multiplier")


def test_vbd_tendon_diagnostics_match_final_pose(test, device):
    """Reported tendon geometry and damping should use the completed VBD pose."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        body_l = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.5, 0.0, 0.0)),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        body_r = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.5, 0.0, 0.0)),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        compliance = 1.0e-5
        damping = 1.0
        initial_length = 1.0
        rest_length = 0.5
        builder.add_tendon()
        builder.add_tendon_link(body=body_l, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=body_r,
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=compliance,
            damping=damping,
            rest_length=rest_length,
        )
        builder.color()
        model = builder.finalize()
        solver = newton.solvers.SolverVBD(model, iterations=1, rigid_avbd_beta=1.0e6)
        state_0 = model.state()
        state_1 = model.state()

        dt = 1.0 / 60.0
        solver.step(state_0, state_1, model.control(), None, dt)

        body_q = state_1.body_q.numpy()
        final_length = float(np.linalg.norm(body_q[body_r][:3] - body_q[body_l][:3]))
        attachment_l = solver.tendon_seg_attachment_l.numpy()[0]
        attachment_r = solver.tendon_seg_attachment_r.numpy()[0]
        attachment_length = float(np.linalg.norm(attachment_r - attachment_l))
        length_rate = (final_length - initial_length) / dt
        material_tension = max((final_length - rest_length) / compliance, 0.0)
        total_tension = max(material_tension + damping * length_rate, 0.0)
        reported_material_tension = float(solver.tendon_seg_material_tension.numpy()[0])
        reported_lambda = float(solver.tendon_seg_lambda.numpy()[0])
        routing_damping_tension = float(solver.tendon_seg_damping_tension.numpy()[0])
        test.assertLess(final_length, 0.75, "The solve must move the endpoints enough to expose stale diagnostics")
        test.assertGreater(total_tension, 0.0, "The regression must exercise a loaded tendon")
        test.assertAlmostEqual(attachment_length, final_length, delta=1.0e-6)
        test.assertAlmostEqual(reported_material_tension, material_tension, delta=1.0e-3)
        test.assertAlmostEqual(
            routing_damping_tension,
            damping * length_rate,
            delta=max(1.0e-3, 1.0e-4 * abs(damping * length_rate)),
            msg="VBD material routing must use the same pose-difference damping rate as its force evaluation",
        )
        test.assertEqual(reported_lambda, 0.0, "VBD must not report a synthetic XPBD multiplier")


def test_vbd_nonrolling_rotation_uses_discrete_length_rate(test, device):
    """Non-rolling damping should use the discrete free-span length change."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        rotating = builder.add_body(mass=0.0, is_kinematic=True)
        fixed = builder.add_body(
            xform=wp.transform(p=wp.vec3(2.0, 1.0, 0.0)),
            mass=0.0,
            is_kinematic=True,
        )
        damping = 10.0
        builder.add_tendon()
        builder.add_tendon_link(
            body=rotating,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(1.0, 0.0, 0.0),
        )
        builder.add_tendon_link(
            body=fixed,
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=1.0e-3,
            damping=damping,
            rest_length=-1.0,
        )
        builder.color()
        model = builder.finalize()
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        dt = 1.0 / 60.0
        # SolverVBD baselines rigid pose history from the first step's input State,
        # so establish the unrotated baseline before teleporting the kinematic body.
        solver.step(state_0, state_1, model.control(), None, dt)
        state_0, state_1 = state_1, state_0

        body_q = state_0.body_q.numpy()
        angle = 0.5 * np.pi
        body_q[rotating, 3:] = np.array([0.0, 0.0, np.sin(0.5 * angle), np.cos(0.5 * angle)])
        state_0.body_q.assign(body_q)

        solver.step(state_0, state_1, model.control(), None, dt)

        initial_length = np.sqrt(2.0)
        final_length = 2.0
        expected_damping_tension = damping * (final_length - initial_length) / dt
        test.assertAlmostEqual(
            float(solver.tendon_seg_damping_tension.numpy()[0]),
            expected_damping_tension,
            delta=1.0e-3,
        )


def test_vbd_rolling_spin_does_not_add_damping_tension(test, device):
    """Pure roller-axis spin should transport material without stretching the free spans."""
    with wp.ScopedDevice(device):
        model, pulley = build_kinematic_rolling_transport(mu=10.0, pre_stretch=0.08)
        model.tendon_seg_damping.fill_(100.0)
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        # Establish pose history first; a first step would report zero length rate for any motion.
        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0
        initial_rest = solver.tendon_seg_rest_length.numpy().copy()

        body_q = state_0.body_q.numpy()
        angle = 0.4
        body_q[pulley, 3:] = np.array([0.0, 0.0, np.sin(0.5 * angle), np.cos(0.5 * angle)], dtype=np.float32)
        state_0.body_q.assign(body_q)

        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)

        rest_delta = solver.tendon_seg_rest_length.numpy() - initial_rest
        damping_tension = solver.tendon_seg_damping_tension.numpy()
        test.assertGreater(
            float(np.max(np.abs(rest_delta))),
            1.0e-4,
            f"Prescribed rolling spin should still transport material: delta={rest_delta}",
        )
        np.testing.assert_allclose(
            damping_tension,
            0.0,
            atol=1.0e-4,
            err_msg="Roller-axis spin must not change free-span length or create damping tension",
        )


def test_vbd_frictionless_off_center_roller_retains_body_torque(test, device):
    """A frictionless roller still transmits force through its off-center axis."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        anchor_l = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.5, 1.0, 0.0)),
            mass=0.0,
            is_kinematic=True,
        )
        roller = builder.add_body(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            lock_inertia=True,
        )
        anchor_r = builder.add_body(
            xform=wp.transform(p=wp.vec3(1.5, 1.0, 0.0)),
            mass=0.0,
            is_kinematic=True,
        )

        builder.add_tendon()
        builder.add_tendon_link(body=anchor_l, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=roller,
            link_type=int(TendonLinkType.ROLLING),
            radius=0.2,
            orientation=1,
            axis=(0.0, 0.0, 1.0),
            offset=(0.5, 0.0, 0.0),
            mu=0.0,
            compliance=1.0e-2,
            rest_length=1.3,
        )
        builder.add_tendon_link(
            body=anchor_r,
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=1.0e-2,
            rest_length=1.3,
        )
        builder.color()
        model = builder.finalize()
        _set_serial_body_coloring(model)

        solver_kwargs = dict(TENDON_VBD_SOLVER_KWARGS)
        solver_kwargs["iterations"] = 1
        solver = newton.solvers.SolverVBD(model, **solver_kwargs)
        state_0 = model.state()
        state_1 = model.state()
        solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)

        tension = solver.tendon_seg_material_tension.numpy()
        angle = _hinge_z_angle(state_1.body_q.numpy(), roller)
        np.testing.assert_allclose(tension[0], tension[1], rtol=1.0e-4, atol=1.0e-4)
        test.assertGreater(
            angle,
            4.0e-4,
            "The resultant tendon force at an off-center roller must retain its moment about the body COM",
        )


def test_vbd_same_body_rolling_span_applies_net_torque(test, device):
    """A rolling span with both endpoints on one body should retain its net torque."""
    with wp.ScopedDevice(device):
        initial_angle = 0.23
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        anchor = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.8, -0.5, 0.0)),
            mass=0.0,
            is_kinematic=True,
        )
        body = builder.add_body(
            xform=wp.transform(q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), initial_angle)),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            lock_inertia=True,
        )

        builder.add_tendon()
        builder.add_tendon_link(body=anchor, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=body,
            link_type=int(TendonLinkType.ROLLING),
            radius=0.12,
            orientation=-1,
            mu=0.0,
            offset=(0.1, 0.15, 0.0),
            compliance=1.0e-3,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=body,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.5, -0.3, 0.0),
            compliance=1.0e-3,
            rest_length=-1.0,
        )
        builder.color()
        model = builder.finalize()
        _set_serial_body_coloring(model)

        solver_kwargs = dict(TENDON_VBD_SOLVER_KWARGS)
        solver_kwargs.update(iterations=1, tendon_max_sweeps=1)
        solver = newton.solvers.SolverVBD(model, **solver_kwargs)
        span_lengths = np.linalg.norm(
            solver.tendon_seg_attachment_r.numpy() - solver.tendon_seg_attachment_l.numpy(), axis=1
        )
        rest_lengths = span_lengths.copy()
        rest_lengths -= 0.01
        solver.tendon_seg_rest_length.assign(rest_lengths)

        state_0, state_1 = model.state(), model.state()
        initial_pose = state_0.body_q.numpy()[body].copy()
        dt = 0.002
        # Equal 10 N tensions cancel the internal span's force and moment.
        # The net wrench is the external span's force at its tangent point.
        point = solver.tendon_seg_attachment_r.numpy()[0]
        direction = point - solver.tendon_seg_attachment_l.numpy()[0]
        force = -10.0 * direction / np.linalg.norm(direction)
        torque = np.cross(point - initial_pose[:3], force)[2]
        solver.step(state_0, state_1, model.control(), None, dt)

        final_pose = state_1.body_q.numpy()[body]
        np.testing.assert_allclose(final_pose[:3] - initial_pose[:3], force * dt**2, rtol=0.03, atol=2.0e-8)
        test.assertAlmostEqual(
            _hinge_z_angle(state_1.body_q.numpy(), body) - initial_angle, float(torque * dt**2), delta=2.0e-7
        )
        test.assertGreater(float(solver.tendon_seg_material_tension.numpy()[1]), 1.0)


def test_vbd_tendon_cuda_graph_capture(test, device):
    """Native VBD tendon force accumulation should remain CUDA graph-capturable."""
    device = wp.get_device(device)
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture requires the Warp memory pool")

    with wp.ScopedDevice(device):
        model, body_idx, _, _, initial_z = build_simple_cable_gravity()
        model.tendon_seg_damping.fill_(100.0)
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        dt = 1.0 / 120.0

        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

        with wp.ScopedCapture(device=device) as capture:
            solver.step(state_0, state_1, control, None, dt)
            solver.step(state_1, state_0, control, None, dt)

        for _ in range(10):
            wp.capture_launch(capture.graph)

        body_q = state_0.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Captured VBD tendon steps produced non-finite poses")
        test.assertLess(float(body_q[body_idx][2]), initial_z, "Captured VBD tendon should respond to gravity")


def test_vbd_dynamic_tendon_cuda_graph_capture(test, device):
    """VBD dynamic active-set updates should remain CUDA graph-capturable."""
    device = wp.get_device(device)
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture requires the Warp memory pool")

    with wp.ScopedDevice(device):
        warm_model, _, _, _, _ = build_dynamic_rolling_route()
        warm_solver = _make_tendon_vbd_solver(warm_model)
        warm_0 = warm_model.state()
        warm_1 = warm_model.state()
        warm_solver.step(warm_0, warm_1, warm_model.control(), None, 1.0 / 120.0)

        model, candidate, candidate_link, _, _ = build_dynamic_rolling_route()
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        for state in (state_0, state_1):
            body_q = state.body_q.numpy()
            body_q[candidate, :3] = (0.05, 0.0, 0.0)
            state.body_q.assign(body_q)

        with wp.ScopedCapture(device=device) as capture:
            solver.step(state_0, state_1, control, None, 1.0 / 120.0)
            solver.step(state_1, state_0, control, None, 1.0 / 120.0)

        wp.capture_launch(capture.graph)
        test.assertTrue(solver.tendon_link_active.numpy()[candidate_link])


def test_vbd_dynamic_tendon_switch_example(test, device):
    """VBD should reproduce the existing dynamic-switch example active-set behavior."""
    device = wp.get_device(device)
    if not device.is_cuda:
        test.skipTest("The full dynamic-switch example regression runs on CUDA")

    with wp.ScopedDevice(device):
        example = MujocoSwitchExample(None, None)
        _set_serial_body_coloring(example.model)
        example.solver = _make_tendon_vbd_solver(example.model)
        for _ in range(220):
            example.step()
        example.test_final()


def test_vbd_pinhole_slip_atwood(test, device):
    """A pinhole is a frictionless slip waypoint: heavy descends, light rises."""
    with wp.ScopedDevice(device):
        model, left_idx, right_idx = build_pinhole_atwood()
        state, _ = run_vbd_model(model, num_frames=120)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite VBD pinhole Atwood state")

        left_z = float(body_q[left_idx][2])
        right_z = float(body_q[right_idx][2])
        test.assertGreater(left_z, 2.05, f"Light side should rise through VBD pinhole slip: z={left_z:.4f}")
        test.assertLess(right_z, 1.95, f"Heavy side should descend through VBD pinhole slip: z={right_z:.4f}")


def test_vbd_slack_pinhole_does_not_redistribute(test, device):
    """Pinholes should transfer only taut excess, not proportionally repartition slack."""
    with wp.ScopedDevice(device):
        model = build_slack_pinhole_route()
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        initial = solver.tendon_seg_rest_length.numpy().copy()
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        final = solver.tendon_seg_rest_length.numpy()

        np.testing.assert_allclose(
            final,
            initial,
            rtol=0.0,
            atol=1.0e-6,
            err_msg=f"VBD slack pinhole rest lengths should not be repartitioned: {initial} -> {final}",
        )


def test_vbd_dynamic_pulley_uses_angular_jacobian(test, device):
    """High friction recovers the no-slip angular Jacobian baseline."""
    with wp.ScopedDevice(device):
        model, left_idx, right_idx, pulley_idx = build_dynamic_pulley_atwood(mu=10.0)
        state, _ = run_vbd_model(model, num_frames=60)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite VBD dynamic pulley Atwood state")

        left_z = float(body_q[left_idx][2])
        right_z = float(body_q[right_idx][2])
        theta = abs(_hinge_y_angle(body_q, pulley_idx))

        test.assertGreater(left_z, 2.05, f"VBD light side should rise: z={left_z:.4f}")
        test.assertLess(right_z, 1.95, f"VBD heavy side should descend: z={right_z:.4f}")
        test.assertGreater(theta, 0.05, f"VBD pulley should rotate from no-slip coupling: theta={theta:.4f}")


def test_vbd_pulley_inertia_limit_locks_cable_travel(test, device):
    """As pulley inertia tends to infinity, no-slip cable travel tends to zero."""
    with wp.ScopedDevice(device):
        radius = 0.15
        model, _, _, pulley_idx = build_dynamic_pulley_atwood(
            mu=10.0,
            mass_left=1.0,
            mass_right=3.0,
            pulley_mass=20000.0,
            pulley_radius=radius,
        )
        state, _ = run_vbd_model(model, num_frames=60)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite VBD high-inertia pulley state")

        theta = abs(_hinge_y_angle(body_q, pulley_idx))
        rim_travel = theta * radius

        test.assertLess(
            rim_travel,
            5.0e-3,
            f"VBD high-inertia no-slip pulley should lock cable travel: R*theta={rim_travel:.6f}, theta={theta:.6f}",
        )


def test_vbd_dynamic_capstan_mu_controls_pulley_rotation(test, device):
    """Dynamic capstan: zero mu slips, mid mu grips partially, high mu approaches no-slip."""
    with wp.ScopedDevice(device):
        low = _dynamic_capstan_metrics(device, mu=0.0)
        mid = _dynamic_capstan_metrics(device, mu=0.04)
        high = _dynamic_capstan_metrics(device, mu=10.0)

        for label, metrics in [("low", low), ("mid", mid), ("high", high)]:
            body_q, left_travel, right_travel, *_ = metrics
            test.assertTrue(np.isfinite(body_q).all(), f"Non-finite VBD dynamic capstan state for {label} mu")
            test.assertGreater(right_travel, 0.03, f"VBD heavy side should descend for {label} mu: {right_travel:.5f}")
            test.assertGreater(left_travel, -0.03, f"VBD light side should not sink for {label} mu: {left_travel:.5f}")

        _, _, _, _, theta_low, _, slip_low = low
        _, _, _, _, theta_mid, rim_mid, slip_mid = mid
        _, _, _, cable_high, theta_high, rim_high, slip_high = high

        test.assertLess(abs(theta_low), 0.035, f"Zero-mu VBD dynamic pulley should not rotate: theta={theta_low:.5f}")
        test.assertGreater(
            theta_mid, theta_low + 0.02, f"Mid-mu VBD pulley should rotate in cable direction: {theta_mid:.5f}"
        )
        test.assertGreater(
            theta_high, theta_mid + 0.03, f"High-mu VBD pulley should rotate more than mid mu: {theta_high:.5f}"
        )
        test.assertLess(
            abs(cable_high - rim_high),
            0.06,
            f"High-mu VBD dynamic capstan should approach no-slip: cable={cable_high:.5f}, rim={rim_high:.5f}",
        )
        test.assertGreater(
            slip_low, slip_mid, f"VBD dynamic slip should decrease from low to mid mu: {slip_low:.5f} <= {slip_mid:.5f}"
        )
        test.assertGreater(
            slip_mid,
            slip_high,
            f"VBD dynamic slip should decrease from mid to high mu: {slip_mid:.5f} <= {slip_high:.5f}",
        )
        test.assertGreater(rim_mid, 0.0, f"VBD mid-mu rim travel should be positive: {rim_mid:.5f}")


def test_vbd_dynamic_capstan_example_friction_modes(test, device):
    """Distinguish frictionless slip from the example's two sticking cases."""
    with wp.ScopedDevice(device):
        mus, theta, _cable_travel, rim_travel, slip = _dynamic_capstan_example_metrics(device)
        test.assertEqual(mus[0], 0.0, f"VBD dynamic capstan example should keep zero friction first: {mus}")
        test.assertGreater(mus[1], 0.0, f"VBD dynamic capstan example mid friction should be finite: {mus}")
        test.assertGreaterEqual(
            mus[2], 10.0, f"VBD dynamic capstan example high friction should be no-slip-like: {mus}"
        )

        theta_low, theta_mid, theta_high = theta
        rim_mid = rim_travel[1]
        slip_mid = slip[1]
        slip_high = slip[2]
        test.assertLess(abs(theta_low), 0.08, f"VBD example zero-mu pulley should not rotate: theta={theta_low:.5f}")
        test.assertGreater(
            theta_mid, 0.25, f"VBD example mid-mu pulley should rotate in cable direction: {theta_mid:.5f}"
        )
        # Both finite coefficients support sticking in this setup. Friction
        # sets a limit, not a multiplier on the transported cable length.
        np.testing.assert_allclose(theta_mid, theta_high, rtol=0.03, atol=0.003)
        test.assertAlmostEqual(rim_mid, float(rim_travel[2]), delta=0.01)
        test.assertLess(
            slip_high,
            0.03,
            f"VBD example high-mu pulley should be no-slip/locked: "
            f"theta={theta_high:.5f}, slip={slip_high:.5f}, all_slip={slip}",
        )
        test.assertLess(
            slip_mid,
            slip[0],
            f"VBD example mid-mu slip should be below zero-mu slip: mid={slip_mid:.5f}, low={slip[0]:.5f}",
        )


def test_vbd_dynamic_capstan_example_rotation_transients(test, device):
    """Rendered VBD dynamic capstan trajectories should stay ordered and non-reversing."""
    with wp.ScopedDevice(device):
        example, theta, left_z, right_z = _dynamic_capstan_example_history(device, num_frames=60)
        dtheta = np.diff(theta, axis=0)

        test.assertTrue(np.isfinite(theta).all(), "Non-finite VBD dynamic capstan rotation history")
        test.assertTrue(np.isfinite(left_z).all(), "Non-finite VBD dynamic capstan payload history")
        # Only cable-driven frames isolate capstan friction: once the rising light payload (box half-extent
        # 0.06 m) reaches the pulley bottom, rigid shape contact can legitimately spin the zero-mu rim.
        cable_only = left_z[:, 0] + 0.06 < example.pulley_z - example.pulley_radius - 0.02
        test.assertGreater(np.count_nonzero(cable_only), 40)
        test.assertLess(
            np.max(np.abs(theta[cable_only, 0])),
            0.08,
            f"Zero-mu VBD dynamic pulley should remain nearly stationary while only the cable acts: {theta[:, 0]}",
        )
        test.assertGreaterEqual(
            np.min(dtheta[:, 2]),
            -5.0e-3,
            f"High-mu VBD dynamic pulley should not reverse direction: min_dtheta={np.min(dtheta[:, 2]):.5f}",
        )
        test.assertLessEqual(
            np.max(theta[:, 1] - theta[:, 2]),
            0.10,
            f"Mid-mu VBD dynamic pulley should not overtake the high-mu pulley: "
            f"max_mid_minus_high={np.max(theta[:, 1] - theta[:, 2]):.5f}, final={theta[-1]}",
        )
        test.assertLessEqual(
            np.max(dtheta[:, 1]),
            0.16,
            f"Mid-mu VBD dynamic pulley should not develop a delayed angular-velocity spike: "
            f"max_dtheta={np.max(dtheta[:, 1]):.5f}",
        )

        light_limit = example.pulley_z + example.pulley_radius - 0.02
        test.assertLess(
            np.max(left_z),
            light_limit,
            f"VBD dynamic capstan light payloads should not crest over the pulleys: "
            f"max_z={np.max(left_z, axis=0)}, limit={light_limit:.5f}",
        )

        cable_travel = 0.5 * ((left_z[-1] - 2.0) + (2.0 - right_z[-1]))
        rim_travel = theta[-1] * example.pulley_radius
        slip = np.abs(cable_travel - rim_travel)
        test.assertAlmostEqual(float(slip[1]), float(slip[2]), delta=0.01)


def test_vbd_kinematic_capstan_mu_controls_slip_and_locking(test, device):
    """Kinematic capstan: zero mu slips, mid mu slips less, high mu locks."""
    with wp.ScopedDevice(device):
        low = _kinematic_capstan_metrics(device, mu=0.0)
        mid = _kinematic_capstan_metrics(device, mu=0.08)
        high = _kinematic_capstan_metrics(device, mu=10.0)

        for label, metrics in [("low", low), ("mid", mid), ("high", high)]:
            body_q, _, _, theta = metrics
            test.assertTrue(np.isfinite(body_q).all(), f"Non-finite VBD kinematic capstan state for {label} mu")
            test.assertLess(abs(theta), 1.0e-5, f"VBD kinematic pulley should not rotate for {label} mu: {theta:.6f}")

        _, left_low, right_low, _ = low
        _, _, right_mid, _ = mid
        _, _, right_high, _ = high

        test.assertGreater(right_low, 0.18, f"Zero-mu VBD kinematic capstan should freely slip: dz={right_low:.5f}")
        test.assertGreater(left_low, 0.08, f"Zero-mu VBD light side should rise through slip: dz={left_low:.5f}")
        test.assertGreater(
            right_low, right_mid + 0.03, f"VBD mid mu should slip less than zero mu: {right_low:.5f} vs {right_mid:.5f}"
        )
        test.assertGreater(
            right_mid,
            right_high + 0.03,
            f"VBD high mu should lock more than mid mu: {right_mid:.5f} vs {right_high:.5f}",
        )
        test.assertLess(
            right_high, 0.06, f"High-mu VBD kinematic capstan should lock cable motion: dz={right_high:.5f}"
        )


def test_vbd_unloaded_motorized_pulley_does_not_drive_slider(test, device):
    """Finite friction cannot grip an unloaded cable."""
    with wp.ScopedDevice(device):
        model, slider_idx, pulley_idx, drive_joint = build_motorized_pulley_drive(mu=10.0)
        # Isolate force-free transport from stiff motor/joint convergence transients.
        model.tendon_seg_compliance.fill_(1.0e-4)
        state, _ = run_vbd_motorized_model(model, drive_joint)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite VBD motorized pulley state")

        slider_x = float(body_q[slider_idx][0])
        theta = abs(_hinge_z_angle(body_q, pulley_idx))

        test.assertGreater(theta, 0.5, f"VBD drive pulley should rotate under its target: theta={theta:.4f}")
        test.assertAlmostEqual(slider_x, -0.4, delta=1.0e-5)


def test_vbd_frictionless_motorized_pulley_does_not_drive_slider(test, device):
    """With mu=0, pulley spin should not inject cable sliding through rolling transfer."""
    with wp.ScopedDevice(device):
        model, slider_idx, pulley_idx, drive_joint = build_motorized_pulley_drive(mu=0.0)
        state, _ = run_vbd_motorized_model(model, drive_joint)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite VBD frictionless motorized pulley state")

        slider_x = float(body_q[slider_idx][0])
        theta = abs(_hinge_z_angle(body_q, pulley_idx))

        test.assertGreater(theta, 0.5, f"VBD frictionless drive pulley should still rotate: theta={theta:.4f}")
        test.assertLess(
            abs(slider_x + 0.4),
            0.02,
            f"VBD frictionless pulley spin should not pull cable/slider: x={slider_x:.4f}",
        )


def test_vbd_unloaded_motorized_pulley_preserves_rest(test, device):
    """An unloaded spinning pulley must not inject stretch during startup."""
    with wp.ScopedDevice(device):
        model, slider_idx, pulley_idx, drive_joint = build_motorized_pulley_drive(mu=10.0)
        # Isolate force-free transport from stiff motor/joint convergence transients.
        model.tendon_seg_compliance.fill_(1.0e-4)
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        q_start = int(model.joint_q_start.numpy()[drive_joint])
        initial_x = float(state_0.body_q.numpy()[slider_idx][0])
        initial_rest = solver.tendon_seg_rest_length.numpy().copy()
        dt = 1.0 / 60.0 / 10.0
        for _ in range(30):
            control.joint_target_q[q_start : q_start + 1].fill_(1.0)
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

        body_q = state_0.body_q.numpy()
        slider_dx = float(body_q[slider_idx][0]) - initial_x
        theta = abs(_hinge_z_angle(body_q, pulley_idx))

        test.assertGreater(theta, 0.1, f"VBD pulley should have started rotating: theta={theta:.4f}")
        test.assertAlmostEqual(slider_dx, 0.0, delta=1.0e-5)
        np.testing.assert_allclose(solver.tendon_seg_rest_length.numpy(), initial_rest, atol=1.0e-6, rtol=0)


def test_vbd_motorized_pulley_updates_rest_in_first_step(test, device):
    """A pre-tensioned cable should grip during the first step of pulley rotation."""
    with wp.ScopedDevice(device):
        model, _, pulley_idx, drive_joint = build_motorized_pulley_drive(mu=10.0)
        model.tendon_seg_compliance.fill_(1.0e-3)
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        solver.tendon_seg_rest_length.assign(solver.tendon_seg_rest_length.numpy() - 0.01)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        q_start = int(model.joint_q_start.numpy()[drive_joint])
        initial_rest = solver.tendon_seg_rest_length.numpy().copy()
        control.joint_target_q[q_start : q_start + 1].fill_(1.0)

        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0 / 10.0)

        body_q = state_1.body_q.numpy()
        theta = abs(_hinge_z_angle(body_q, pulley_idx))
        rest_delta = solver.tendon_seg_rest_length.numpy() - initial_rest

        test.assertGreater(theta, 1.0e-3, f"VBD drive pulley should rotate in the first step: theta={theta:.6f}")
        test.assertGreater(
            float(np.max(np.abs(rest_delta))),
            1.0e-4,
            f"VBD pulley rotation should transfer rolling rest length in the first step: delta={rest_delta}",
        )


def test_vbd_moving_rolling_route_conserves_material(test, device):
    """VBD should conserve material while a rolling link's wrap angle changes."""
    with wp.ScopedDevice(device):

        def solver_factory(model):
            _set_serial_body_coloring(model)
            return _make_tendon_vbd_solver(model)

        cases = ((0.0, None), (0.2, None), (100.0, None), (100.0, 1))
        for mu, short_segment in cases:
            with test.subTest(mu=mu, short_segment=short_segment):
                error, min_rest = _moving_rolling_route_material_error(solver_factory, mu, short_segment)
                test.assertLess(
                    error,
                    1.0e-4,
                    f"VBD cable material drifted by {error} m for mu={mu}, short_segment={short_segment}",
                )
                if short_segment is not None:
                    test.assertGreaterEqual(min_rest, 0.99e-6)


def test_vbd_rolling_transfer_saturates_at_zero_span(test, device):
    """Bound rolling transport without creating material at a depleted span."""
    with wp.ScopedDevice(device):
        # Both sides remain taut even at the floor: total free rest is 0.24 m,
        # versus a 0.387 m geometric length on each side. No unloaded grip.
        model, pulley = build_kinematic_rolling_transport(mu=100.0)
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        solver.tendon_seg_rest_length.fill_(0.12)
        state_0, state_1 = model.state(), model.state()
        for angle in (0.5, 1.0, 1.5, 2.0):
            body_q = state_0.body_q.numpy()
            body_q[pulley, 3:] = [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]
            state_0.body_q.assign(body_q)
            solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
            rest = solver.tendon_seg_rest_length.numpy()
            test.assertGreaterEqual(float(rest.min()), 0.99e-6)
            test.assertAlmostEqual(float(rest.sum()), 0.24, delta=1.0e-6)
        test.assertLess(float(rest.min()), 1.1e-6)


def test_vbd_equal_weight_atwood(test, device):
    """Equal masses over a high-friction pulley should remain close to equilibrium."""
    with wp.ScopedDevice(device):
        model, left_idx, right_idx, pulley_idx = build_atwood_equal_weights()
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        dt = 1.0 / 60.0 / 16

        bq0 = state_0.body_q.numpy()
        y_left_0 = float(bq0[left_idx][2])
        y_right_0 = float(bq0[right_idx][2])

        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

        att_r = solver.tendon_seg_attachment_r.numpy()
        att_l = solver.tendon_seg_attachment_l.numpy()
        pulley_z = float(state_0.body_q.numpy()[pulley_idx][2])
        test.assertGreater(
            att_r[0][2],
            pulley_z,
            f"VBD cable should wrap over pulley: left tangent z={att_r[0][2]:.3f} <= center z={pulley_z:.3f}",
        )
        test.assertGreater(
            att_l[1][2],
            pulley_z,
            f"VBD cable should wrap over pulley: right tangent z={att_l[1][2]:.3f} <= center z={pulley_z:.3f}",
        )

        for _ in range(40):
            for _ in range(16):
                state_0.clear_forces()
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

        bq = state_0.body_q.numpy()
        y_left = float(bq[left_idx][2])
        y_right = float(bq[right_idx][2])

        drift_left = abs(y_left - y_left_0)
        drift_right = abs(y_right - y_right_0)
        drift_diff = abs((y_left - y_left_0) - (y_right - y_right_0))

        test.assertLess(drift_left, 0.05, f"VBD left weight drifted {drift_left:.4f} m")
        test.assertLess(drift_right, 0.05, f"VBD right weight drifted {drift_right:.4f} m")
        test.assertLess(drift_diff, 0.02, f"VBD asymmetric drift: {drift_diff:.4f} m")
        test.assertTrue(np.isfinite(bq).all(), "Non-finite VBD equal-weight body positions")


def test_vbd_compound_pulley_stays_balanced(test, device):
    """The VBD compound-pulley render path must not pass while visibly unstable."""
    # The stiff mechanism needs more nonlinear iterations after its weights
    # enter ground contact to satisfy the long-run pulley-windup bound.
    # Compliant-ALM structural joints default to 1e5 N/m; the 80 kg pulleys would sag ~9 mm and bounce,
    # counter-rotating the stick-coupled pulleys. Author a stiff axle under the compliant-ALM contract.
    _run_vbd_example_assertions(
        test, device, CompoundPulleyExample, 220, {"iterations": 32, "rigid_joint_linear_ke": 1.0e6}
    )


def test_vbd_rolling_pulley_revolute_axis_stays_stiff(test, device):
    """VBD tendon scenes need stiff revolute anchors so routing geometry stays coherent."""
    with wp.ScopedDevice(device):
        args = type("Args", (), {"headless": True, "record": False, "num_frames": 24, "episode_frames": None})()
        example = RollingPulleyExample(None, args)
        _set_serial_body_coloring(example.model)
        example.solver = _make_tendon_vbd_solver(example.model, STIFF_REVOLUTE_ANCHOR_OVERRIDES)

        body_q0 = example.state_0.body_q.numpy().copy()
        pulley_idx = example.pulley_idx
        max_pulley_drift = 0.0

        for _ in range(24):
            example.step()
            body_q = example.state_0.body_q.numpy()
            test.assertTrue(np.isfinite(body_q).all(), "Non-finite VBD rolling-pulley state")
            drift = float(np.linalg.norm(body_q[pulley_idx][:3] - body_q0[pulley_idx][:3]))
            max_pulley_drift = max(max_pulley_drift, drift)

        test.assertLess(
            max_pulley_drift,
            5.0e-3,
            f"VBD rolling-pulley revolute anchor drift is too soft: {max_pulley_drift:.6f} m",
        )


def test_vbd_gear_pulley_direction_and_mechanical_advantage(test, device):
    """The VBD gear example must match the XPBD direction: free light weight falls."""
    with wp.ScopedDevice(device):
        example = GearPulleyExample(None, None)
        _set_serial_body_coloring(example.model)
        example.solver = _make_tendon_vbd_solver(example.model)

        for _ in range(example._direction_validation_frames):
            example.step()

        lower_positions = np.asarray(example._lower_history, dtype=np.float64)
        right_positions = np.asarray(example._right_history, dtype=np.float64)
        rotations = np.asarray(example._pulley_rotation_history, dtype=np.float64)

        sample = len(lower_positions) - 1
        lower_rise = float(lower_positions[sample, 2]) - example._initial_lower_z
        right_drop = example._initial_right_z - float(right_positions[sample, 2])
        right_travel = float(np.linalg.norm(right_positions[sample] - right_positions[0]))
        travel_ratio = right_travel / max(lower_rise, 1.0e-6)
        final_rim_travel = np.abs(rotations[sample] * example._pulley_radii)

        test.assertGreater(lower_rise, 0.05, f"VBD gear moving block should rise: dz={lower_rise:.4f}")
        test.assertGreater(
            right_drop,
            0.20,
            f"VBD gear free counterweight should fall, not rise: dz={right_drop:.4f}",
        )
        test.assertGreater(travel_ratio, 2.5, f"VBD gear should amplify free-weight travel: ratio={travel_ratio:.3f}")
        test.assertLess(
            travel_ratio, 5.5, f"VBD gear travel ratio is not the expected block-and-tackle range: {travel_ratio:.3f}"
        )
        inactive = [example._pulley_names[i] for i, travel in enumerate(final_rim_travel) if float(travel) <= 0.025]
        test.assertFalse(inactive, f"All VBD gear pulleys should rotate: inactive={inactive}, rim={final_rim_travel}")


def test_vbd_xy_table_tracks_reference_prefix(test, device):
    """VBD XY table should follow the same reference direction as the XPBD example."""
    if not wp.get_device(device).is_cuda:
        test.skipTest("XY table VBD reference-prefix regression is CUDA-only because the scene is expensive")

    with wp.ScopedDevice(device):
        example = XYTableExample(None, None)
        _set_serial_body_coloring(example.model)

        example.model.joint_target_kd[example.p2_dof_start : example.p2_dof_start + 1].fill_(200.0)
        example.model.joint_target_kd[example.p6_dof_start : example.p6_dof_start + 1].fill_(200.0)

        solver_kwargs = dict(TENDON_VBD_SOLVER_KWARGS)
        solver_kwargs["iterations"] = 30
        solver_kwargs["rigid_tendon_relaxation"] = 0.6
        example.solver = newton.solvers.SolverVBD(example.model, **solver_kwargs)
        example._apply_cable_pretension(0.99995)

        for _ in range(30):
            example.step()
            example.test_post_step()

        # example.test_final() encodes whole-run windows (e.g. a 2.5 s RMS window and full-run guide-pulley
        # rotation); on this 0.5 s prefix it measures only the start-up transient. The prefix bounds are below.
        table_xy = np.asarray(example._table_xy_history, dtype=np.float64)
        drive_rot = np.asarray(example._drive_rotation_history, dtype=np.float64)
        sample_times = (np.arange(len(table_xy), dtype=np.float64) + 1.0) * example.frame_dt
        reference_xy = np.asarray([_desired_table_xy(float(t)) for t in sample_times], dtype=np.float64)
        reference_error = np.linalg.norm(table_xy - reference_xy, axis=1)
        reference_rms = float(np.sqrt(np.mean(reference_error * reference_error)))
        reference_max = float(np.max(reference_error))
        y_drift = float(np.max(np.abs(table_xy[:, 1])))
        drive_pair_error = float(np.max(np.abs(drive_rot[:, 0] - drive_rot[:, 1])))

        test.assertLess(reference_rms, 0.008, f"VBD XY table prefix RMS drifted from reference: {reference_rms:.5f}")
        test.assertLess(reference_max, 0.012, f"VBD XY table prefix max error too high: {reference_max:.5f}")
        test.assertLess(y_drift, 0.004, f"VBD XY table should stay on the X reference prefix: y={y_drift:.5f}")
        test.assertLess(
            drive_pair_error,
            0.02,
            f"VBD XY table drive pulleys should stay coupled in the symmetric prefix: error={drive_pair_error:.5f}",
        )


VBD_EXAMPLE_ASSERTION_CASES = (
    ("vbd_tendon_pulley_example_assertions", PulleyExample, 100, True, set()),
    ("vbd_tendon_pinhole_example_assertions", PinholeExample, 100, True, set()),
    # 150 frames: the light weight reaches the pulley crown at frame ~154 with XPBD and VBD alike.
    ("vbd_tendon_rolling_pulley_example_assertions", RollingPulleyExample, 150, True, set()),
    ("vbd_tendon_equilibrium_example_assertions", EquilibriumExample, 100, True, set()),
    ("vbd_tendon_gear_pulley_example_assertions", GearPulleyExample, 180, True, set()),
    ("vbd_tendon_cable_machine_example_assertions", CableMachineExample, 100, True, set()),
    ("vbd_tendon_3d_routing_example_assertions", Routing3DExample, 70, True, set()),
    ("vbd_tendon_capstan_friction_example_assertions", DynamicCapstanExample, 100, True, set()),
    ("vbd_tendon_capstan_kinematic_example_assertions", KinematicCapstanExample, 100, True, set()),
)


def test_vbd_tendon_alm_matches_penalty_equilibrium(test, device):
    """The compliant-ALM stretch row must reach the same static equilibrium as the penalty row, with the
    multiplier equal to the material tension."""
    with wp.ScopedDevice(device):
        results = {}
        for alm in (False, True):
            model, body_idx, mass, compliance, initial_z = build_simple_cable_gravity()
            model.tendon_seg_damping.fill_(100.0)
            _set_serial_body_coloring(model)
            solver = newton.solvers.SolverVBD(
                model, **TENDON_VBD_SOLVER_KWARGS, tendon_alm=alm, tendon_alm_min_stiffness_ratio=0.0
            )
            state_0 = model.state()
            state_1 = model.state()
            control = model.control()
            contacts = model.contacts()
            dt = 1.0 / 120.0
            for _ in range(round(1.5 / dt)):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

            attachment_l = solver.tendon_seg_attachment_l.numpy()[0]
            attachment_r = solver.tendon_seg_attachment_r.numpy()[0]
            stretch = float(np.linalg.norm(attachment_r - attachment_l)) - float(
                solver.tendon_seg_rest_length.numpy()[0]
            )
            results[alm] = {
                "z": float(state_0.body_q.numpy()[body_idx][2]),
                "material_tension": stretch / compliance,
                "reported_tension": float(solver.tendon_seg_material_tension.numpy()[0])
                + float(solver.tendon_seg_damping_tension.numpy()[0]),
                "multiplier": float(solver.tendon_seg_lambda.numpy()[0]),
                "alm_lambda": float(solver.tendon_seg_alm_lambda.numpy()[0]) if alm else 0.0,
            }

        load = mass * 9.81
        expected_z = initial_z - load * compliance
        for alm, r in results.items():
            test.assertAlmostEqual(r["material_tension"], load, delta=0.05 * load, msg=f"alm={alm}: {r}")
            test.assertAlmostEqual(r["z"], expected_z, delta=5.0e-3, msg=f"alm={alm}: {r}")
            test.assertAlmostEqual(r["reported_tension"], load, delta=0.05 * load, msg=f"alm={alm}: {r}")
        test.assertAlmostEqual(results[True]["z"], results[False]["z"], delta=1.0e-3, msg=str(results))
        test.assertEqual(results[False]["multiplier"], 0.0, "penalty mode must not report a multiplier")
        test.assertAlmostEqual(
            results[True]["alm_lambda"],
            load,
            delta=0.02 * load,
            msg=f"ALM multiplier should carry the static tension: {results[True]}",
        )
        test.assertEqual(results[True]["multiplier"], results[True]["alm_lambda"])


def test_vbd_tendon_alm_releases_slack_segment(test, device):
    """A stale stretch multiplier on a slack segment must release without pushing on the bodies."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        anchor = builder.add_body(mass=0.0, is_kinematic=True)
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.25, 0.0, 0.0)),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_tendon()
        builder.add_tendon_link(body=anchor, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=body,
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=1.0e-3,
            damping=100.0,
            rest_length=1.0,
        )
        builder.color()
        model = builder.finalize()
        solver = newton.solvers.SolverVBD(
            model, **TENDON_VBD_SOLVER_KWARGS, tendon_alm=True, tendon_alm_min_stiffness_ratio=0.0
        )
        state_0 = model.state()
        state_1 = model.state()
        dt = 1.0 / 120.0
        # First step registers the route identity of the multiplier; then plant a stale tension on it.
        solver.step(state_0, state_1, model.control(), None, dt)
        state_0, state_1 = state_1, state_0
        # Plant the stale tension where the row keeps it (the per-segment array is a broadcast copy rewritten each step).
        solver.tendon_alm_lambda.fill_(50.0)
        solver.tendon_seg_alm_lambda.fill_(50.0)
        state_0.body_qd.assign(
            [
                wp.spatial_vector(0.0),
                wp.spatial_vector(30.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ]
        )
        solver.step(state_0, state_1, model.control(), None, dt)

        final_vx = float(state_1.body_qd.numpy()[body][0])
        test.assertAlmostEqual(final_vx, 30.0, delta=1.0e-4, msg="slack segment must transmit no force")
        test.assertEqual(float(solver.tendon_seg_alm_lambda.numpy()[0]), 0.0, "stale multiplier must release")
        # Same contract as the legacy slack test: no material tension is reported for a slack span.
        test.assertEqual(float(solver.tendon_seg_material_tension.numpy()[0]), 0.0)
        test.assertEqual(float(solver.tendon_seg_lambda.numpy()[0]), 0.0)


def test_vbd_tendon_alm_shared_multiplier_matches_material_law(test, device):
    """On a routed cable whose segments differ in stiffness, the shared multiplier must settle on the material
    tension of every segment (the ascent averages force residuals, not stretch)."""
    with wp.ScopedDevice(device):
        model, slider = build_fixed_rolling_chain()
        compliance = model.tendon_seg_compliance.numpy()
        compliance[1] *= 8.0  # one span eight times softer: K differs across the tendon
        model.tendon_seg_compliance.assign(compliance)
        model.tendon_seg_damping.fill_(50.0)
        solver = newton.solvers.SolverVBD(
            model, **TENDON_VBD_SOLVER_KWARGS, tendon_alm=True, tendon_alm_min_stiffness_ratio=0.0
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        direction = solver.tendon_seg_attachment_r.numpy()[-1] - solver.tendon_seg_attachment_l.numpy()[-1]
        direction /= np.linalg.norm(direction)
        body_f = np.zeros((model.body_count, 6), dtype=np.float32)
        body_f[slider, :3] = 10.0 * direction
        dt = 1.0 / 120.0
        for _ in range(240):
            state_0.body_f.assign(wp.array(body_f, dtype=wp.spatial_vector))
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0

        lengths = np.linalg.norm(
            solver.tendon_seg_attachment_r.numpy() - solver.tendon_seg_attachment_l.numpy(), axis=1
        )
        material = np.maximum(lengths - solver.tendon_seg_rest_length.numpy(), 0.0) / np.maximum(
            solver.tendon_seg_active_compliance.numpy(), 1.0e-30
        )
        lam = float(solver.tendon_alm_lambda.numpy()[0])
        test.assertAlmostEqual(lam, 10.0, delta=0.5, msg=f"multiplier should carry the 10 N load: {lam}")
        for seg, t_law in enumerate(material):
            test.assertAlmostEqual(
                t_law, 10.0, delta=0.5, msg=f"segment {seg}: material-law tension {t_law:.3f} should equal the load"
            )


def test_vbd_tendon_alm_two_tendons_keep_separate_multipliers(test, device):
    """Two cables in one model each settle on their own load (per-tendon segment ranges and multipliers)."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, -9.81))
        masses = (1.0, 3.0)
        compliance = 1.0e-3
        for i, mass in enumerate(masses):
            anchor = builder.add_body(xform=wp.transform(p=wp.vec3(float(i), 0.0, 0.0)), mass=0.0, is_kinematic=True)
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(float(i), 0.0, -0.5)),
                mass=mass,
                inertia=wp.mat33(np.eye(3) * 0.01 * mass),
            )
            builder.add_tendon()
            builder.add_tendon_link(body=anchor, link_type=int(TendonLinkType.ATTACHMENT))
            builder.add_tendon_link(
                body=body,
                link_type=int(TendonLinkType.ATTACHMENT),
                compliance=compliance,
                damping=50.0,
                rest_length=0.5,
            )
        builder.color()
        model = builder.finalize()
        solver = newton.solvers.SolverVBD(
            model, **TENDON_VBD_SOLVER_KWARGS, tendon_alm=True, tendon_alm_min_stiffness_ratio=0.0
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        dt = 1.0 / 120.0
        for _ in range(240):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0

        lam = solver.tendon_alm_lambda.numpy()
        lengths = np.linalg.norm(
            solver.tendon_seg_attachment_r.numpy() - solver.tendon_seg_attachment_l.numpy(), axis=1
        )
        material = np.maximum(lengths - solver.tendon_seg_rest_length.numpy(), 0.0) / compliance
        for t, mass in enumerate(masses):
            load = mass * 9.81
            test.assertAlmostEqual(
                float(lam[t]), load, delta=0.05 * load, msg=f"tendon {t}: multiplier {lam[t]:.3f} vs load {load:.3f}"
            )
            test.assertAlmostEqual(
                float(material[t]), load, delta=0.05 * load, msg=f"tendon {t}: material {material[t]:.3f}"
            )


def test_vbd_tendon_alm_guarded_tendon_matches_flag_off(test, device):
    """A tendon the stiffness guard leaves on the legacy row must reproduce the flag-off run bit for bit.

    A routed cable with a nonlinear law is the sensitive case: the legacy row linearises the material with its
    tangent, so any leak of the ALM row's secant into the guarded-off path shows up in the trajectory."""
    laws = {
        "linear": {},
        "sigmoid": {
            "tendon_sigmoid_ea_low": 2000.0,
            "tendon_sigmoid_ea_ratio": 10.0,
            "tendon_sigmoid_transition_strain": 0.002,
            "tendon_sigmoid_transition_width": 0.001,
        },
    }
    with wp.ScopedDevice(device):
        for law_name, law_kwargs in laws.items():
            trajectories = {}
            for alm in (False, True):
                model, slider = build_fixed_rolling_chain()
                model.tendon_seg_damping.fill_(50.0)
                _set_serial_body_coloring(model)
                # An impossibly high ratio forces the guard to keep the legacy row for every tendon.
                solver = newton.solvers.SolverVBD(
                    model,
                    **TENDON_VBD_SOLVER_KWARGS,
                    **law_kwargs,
                    tendon_alm=alm,
                    tendon_alm_min_stiffness_ratio=1.0e12,
                )
                state_0 = model.state()
                state_1 = model.state()
                control = model.control()
                direction = solver.tendon_seg_attachment_r.numpy()[-1] - solver.tendon_seg_attachment_l.numpy()[-1]
                direction /= np.linalg.norm(direction)
                dt = 1.0 / 120.0
                frames = []
                for step in range(240):
                    # 10 N for one second, then released so the cable goes slack.
                    body_f = np.zeros((model.body_count, 6), dtype=np.float32)
                    body_f[slider, :3] = (10.0 if step < 120 else 0.0) * direction
                    state_0.body_f.assign(wp.array(body_f, dtype=wp.spatial_vector))
                    solver.step(state_0, state_1, control, None, dt)
                    state_0, state_1 = state_1, state_0
                    frames.append(
                        np.concatenate(
                            [
                                state_0.body_q.numpy().ravel(),
                                state_0.body_qd.numpy().ravel(),
                                solver.tendon_seg_material_tension.numpy(),
                                solver.tendon_seg_damping_tension.numpy(),
                                solver.tendon_seg_lambda.numpy(),
                            ]
                        )
                    )
                trajectories[alm] = np.array(frames)
                if alm:
                    test.assertTrue(
                        np.all(solver.tendon_alm_mode.numpy() == 0), "guard must have chosen the legacy row"
                    )
                    test.assertTrue(np.all(solver.tendon_seg_alm_k.numpy() == 0.0))
                    test.assertTrue(np.all(solver.tendon_seg_alm_lambda.numpy() == 0.0))

            test.assertTrue(
                np.array_equal(trajectories[True], trajectories[False]),
                f"{law_name} law: the guarded-off ALM run differs from the flag-off run, max |delta| = "
                f"{np.max(np.abs(trajectories[True] - trajectories[False]))}",
            )


def test_vbd_tendon_alm_multiplier_shared_across_frictional_roller(test, device):
    """The ALM multiplier stays one number per tendon across a live frictional roller.

    The material transfer bands the material row across the roller (the Euler-Eytelwein ratio), and the row applies
    the compliance-weighted mean of that row on both sides: the band lives in the rest lengths, not in the span
    forces. Carrying the band on the row was measured on the planar two-cable rig with the direct material solve and
    rejected both ways (one multiplier per run of spans between frictional rollers converges to a band-edge allocation
    left by a transient, 10 N -> 140-283 N; per-span copies scaled by the material profile chatter, hold ripple
    65 deg), so this pins the shared level: equal on both sides, equal to the compliance-weighted mean of the banded
    material row, while the material row itself keeps the capstan differential.
    """
    with wp.ScopedDevice(device):
        model, _left_idx, _right_idx, _pulley_idx = build_kinematic_pulley_atwood(mu=10.0)
        _set_serial_body_coloring(model)
        solver = newton.solvers.SolverVBD(
            model, **TENDON_VBD_SOLVER_KWARGS, tendon_alm=True, tendon_alm_min_stiffness_ratio=0.0
        )
        state_0, state_1 = model.state(), model.state()
        control, contacts = model.control(), model.contacts()
        dt = 1.0 / 60.0 / 12.0
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        for _ in range(40):
            for _ in range(12):
                state_0.clear_forces()
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

        lam = solver.tendon_seg_alm_lambda.numpy()
        length = np.linalg.norm(solver.tendon_seg_attachment_r.numpy() - solver.tendon_seg_attachment_l.numpy(), axis=1)
        compliance = solver.tendon_seg_active_compliance.numpy()
        stretch = length - solver.tendon_seg_rest_length.numpy()
        material = np.maximum(stretch, 0.0) / compliance
        cap_ratio = float(solver.tendon_link_cap_ratio.numpy()[1])

        test.assertGreater(cap_ratio, 1.0, "the pulley must be a live frictional link")
        test.assertGreater(
            float(material.max()) / float(material.min()), 1.2, f"material row must be banded: {material}"
        )
        test.assertLessEqual(float(material.max()) / float(material.min()), cap_ratio * 1.01, f"band: {material}")
        test.assertEqual(float(lam[0]), float(lam[1]), f"one multiplier per tendon across the roller: {lam}")
        mean = float(np.sum(compliance * material) / np.sum(compliance))
        test.assertAlmostEqual(
            float(lam[0]), mean, delta=0.05 * mean, msg=f"level {lam[0]:.3f} N vs compliance-weighted mean {mean:.3f} N"
        )


def test_vbd_tendon_alm_disabled_leaves_state_untouched(test, device):
    """With the default flag the ALM arrays stay zero and no multiplier is reported."""
    with wp.ScopedDevice(device):
        model, _body = build_single_span_tendon()
        _state, solver = run_vbd_model(model, num_frames=10, substeps=4)
        test.assertFalse(solver.tendon_alm)
        test.assertIsNone(solver.tendon_seg_alm_lambda, "no ALM state is allocated when the row is off")
        test.assertIsNone(solver.tendon_alm_lambda)
        test.assertTrue(np.all(solver.tendon_seg_lambda.numpy() == 0.0))


devices = ["cpu"]
if wp.is_cuda_available():
    devices.append("cuda:0")

add_test(
    TestTendonVBD,
    "vbd_tendon_alm_matches_penalty_equilibrium",
    devices,
    test_vbd_tendon_alm_matches_penalty_equilibrium,
)
add_test(TestTendonVBD, "vbd_tendon_alm_releases_slack_segment", devices, test_vbd_tendon_alm_releases_slack_segment)
add_test(
    TestTendonVBD,
    "vbd_tendon_alm_shared_multiplier_matches_material_law",
    devices,
    test_vbd_tendon_alm_shared_multiplier_matches_material_law,
)
add_test(
    TestTendonVBD,
    "vbd_tendon_alm_two_tendons_keep_separate_multipliers",
    devices,
    test_vbd_tendon_alm_two_tendons_keep_separate_multipliers,
)
add_test(
    TestTendonVBD,
    "vbd_tendon_alm_guarded_tendon_matches_flag_off",
    devices,
    test_vbd_tendon_alm_guarded_tendon_matches_flag_off,
)
add_test(
    TestTendonVBD,
    "vbd_tendon_alm_multiplier_shared_across_frictional_roller",
    devices,
    test_vbd_tendon_alm_multiplier_shared_across_frictional_roller,
)
add_test(
    TestTendonVBD,
    "vbd_tendon_alm_disabled_leaves_state_untouched",
    devices,
    test_vbd_tendon_alm_disabled_leaves_state_untouched,
)
add_test(TestTendonVBD, "vbd_tendon_stretch_pulls_toward_anchor", devices, test_vbd_tendon_stretch_pulls_toward_anchor)
add_test(
    TestTendonVBD,
    "vbd_tendon_zero_compliance_uses_floor",
    devices,
    test_vbd_tendon_zero_compliance_uses_floor,
)
add_test(
    TestTendonVBD,
    "vbd_rolling_chain_tension_matches_accepted_pose",
    devices,
    test_vbd_rolling_chain_tension_matches_accepted_pose,
)
add_test(
    TestTendonVBD,
    "vbd_dynamic_tendon_route_switches_inside_solver",
    devices,
    test_vbd_dynamic_tendon_route_switches_inside_solver,
)
add_test(
    TestTendonVBD,
    "vbd_dynamic_route_activation_uses_accepted_pose",
    devices,
    test_vbd_dynamic_route_activation_uses_accepted_pose,
)
add_test(
    TestTendonVBD,
    "vbd_tendon_coloring_separates_segment_endpoints",
    devices,
    test_vbd_tendon_coloring_separates_segment_endpoints,
)
add_test(
    TestTendonVBD,
    "vbd_tendon_compliance_uses_physical_tension",
    devices,
    test_vbd_tendon_compliance_uses_physical_tension,
)
add_test(
    TestTendonVBD,
    "vbd_sigmoid_material_projection_matches_capstan_cone",
    devices,
    test_vbd_sigmoid_material_projection_matches_capstan_cone,
)
add_test(
    TestTendonVBD,
    "vbd_sigmoid_stretch_uses_material_tangent",
    devices,
    test_vbd_sigmoid_stretch_uses_material_tangent,
)
add_test(
    TestTendonVBD,
    "vbd_slack_damped_tendon_remains_force_free",
    devices,
    test_vbd_slack_damped_tendon_remains_force_free,
)
add_test(
    TestTendonVBD,
    "vbd_quasi_static_roller_conserves_cable_inventory",
    devices,
    test_vbd_quasi_static_roller_conserves_cable_inventory,
)
add_test(
    TestTendonVBD,
    "vbd_tendon_diagnostics_match_final_pose",
    devices,
    test_vbd_tendon_diagnostics_match_final_pose,
)
add_test(
    TestTendonVBD,
    "vbd_nonrolling_rotation_uses_discrete_length_rate",
    devices,
    test_vbd_nonrolling_rotation_uses_discrete_length_rate,
)
add_test(
    TestTendonVBD,
    "vbd_rolling_spin_does_not_add_damping_tension",
    devices,
    test_vbd_rolling_spin_does_not_add_damping_tension,
)
add_test(
    TestTendonVBD,
    "vbd_frictionless_off_center_roller_retains_body_torque",
    devices,
    test_vbd_frictionless_off_center_roller_retains_body_torque,
)
add_test(
    TestTendonVBD,
    "vbd_same_body_rolling_span_applies_net_torque",
    devices,
    test_vbd_same_body_rolling_span_applies_net_torque,
)
add_test(TestTendonVBD, "vbd_tendon_cuda_graph_capture", devices, test_vbd_tendon_cuda_graph_capture)
add_test(TestTendonVBD, "vbd_dynamic_tendon_cuda_graph_capture", devices, test_vbd_dynamic_tendon_cuda_graph_capture)
add_test(TestTendonVBD, "vbd_dynamic_tendon_switch_example", devices, test_vbd_dynamic_tendon_switch_example)
add_test(TestTendonVBD, "vbd_pinhole_slip_atwood", devices, test_vbd_pinhole_slip_atwood)
add_test(
    TestTendonVBD, "vbd_slack_pinhole_does_not_redistribute", devices, test_vbd_slack_pinhole_does_not_redistribute
)
add_test(
    TestTendonVBD, "vbd_dynamic_pulley_uses_angular_jacobian", devices, test_vbd_dynamic_pulley_uses_angular_jacobian
)
add_test(
    TestTendonVBD,
    "vbd_pulley_inertia_limit_locks_cable_travel",
    devices,
    test_vbd_pulley_inertia_limit_locks_cable_travel,
)
add_test(
    TestTendonVBD,
    "vbd_dynamic_capstan_mu_controls_pulley_rotation",
    devices,
    test_vbd_dynamic_capstan_mu_controls_pulley_rotation,
)
add_test(
    TestTendonVBD,
    "vbd_dynamic_capstan_example_friction_modes",
    devices,
    test_vbd_dynamic_capstan_example_friction_modes,
)
add_test(
    TestTendonVBD,
    "vbd_dynamic_capstan_example_rotation_transients",
    devices,
    test_vbd_dynamic_capstan_example_rotation_transients,
)
add_test(
    TestTendonVBD,
    "vbd_kinematic_capstan_mu_controls_slip_and_locking",
    devices,
    test_vbd_kinematic_capstan_mu_controls_slip_and_locking,
)
add_test(
    TestTendonVBD,
    "vbd_unloaded_motorized_pulley_does_not_drive_slider",
    devices,
    test_vbd_unloaded_motorized_pulley_does_not_drive_slider,
)
add_test(
    TestTendonVBD,
    "vbd_frictionless_motorized_pulley_does_not_drive_slider",
    devices,
    test_vbd_frictionless_motorized_pulley_does_not_drive_slider,
)
add_test(
    TestTendonVBD,
    "vbd_unloaded_motorized_pulley_preserves_rest",
    devices,
    test_vbd_unloaded_motorized_pulley_preserves_rest,
)
add_test(
    TestTendonVBD,
    "vbd_motorized_pulley_updates_rest_in_first_step",
    devices,
    test_vbd_motorized_pulley_updates_rest_in_first_step,
)
add_test(
    TestTendonVBD,
    "vbd_moving_rolling_route_conserves_material",
    devices,
    test_vbd_moving_rolling_route_conserves_material,
)
add_test(
    TestTendonVBD,
    "vbd_rolling_transfer_saturates_at_zero_span",
    devices,
    test_vbd_rolling_transfer_saturates_at_zero_span,
)
add_test(TestTendonVBD, "vbd_equal_weight_atwood", devices, test_vbd_equal_weight_atwood)
add_test(TestTendonVBD, "vbd_compound_pulley_stays_balanced", devices, test_vbd_compound_pulley_stays_balanced)
add_test(
    TestTendonVBD,
    "vbd_rolling_pulley_revolute_axis_stays_stiff",
    devices,
    test_vbd_rolling_pulley_revolute_axis_stays_stiff,
)
add_test(
    TestTendonVBD,
    "vbd_gear_pulley_direction_and_mechanical_advantage",
    devices,
    test_vbd_gear_pulley_direction_and_mechanical_advantage,
)
add_test(TestTendonVBD, "vbd_xy_table_tracks_reference_prefix", devices, test_vbd_xy_table_tracks_reference_prefix)
for name, example_cls, num_frames, skip_cpu, expected_failure_devices in VBD_EXAMPLE_ASSERTION_CASES:
    overrides = STIFF_REVOLUTE_ANCHOR_OVERRIDES if example_cls is RollingPulleyExample else None
    test_fn = _make_vbd_example_assertion_test(example_cls, num_frames, skip_cpu=skip_cpu, solver_overrides=overrides)
    test_fn.__unittest_expected_failure_devices__ = expected_failure_devices
    add_test(TestTendonVBD, name, devices, test_fn)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
