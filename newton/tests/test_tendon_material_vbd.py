# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical and solver-entry-point checks for direct tendon material in VBD."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.tendon_material_cooperative_kernels import solve_tendon_material_cooperative
from newton._src.solvers.vbd.tendon_kernels import (
    TendonForceElementAdjacencyInfo,
    evaluate_tendon_force_hessians,
)
from newton.tests.test_tendon_capstan import _oriented_route_material_length, build_force_driven_dynamic_route
from newton.tests.test_tendon_material_dynamics import _make_pulley, _PulleyTrajectory
from newton.tests.test_tendon_material_integration import _add_chain, _chain_model, _two_tendon_model
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _make_solver(model, *, iterations=8, direct=True, settle_iterations=4, **solver_options):
    """Use stiff undamped hinges so cable torque controls the free rotation."""
    model.body_color_groups = [wp.array([body], dtype=int, device=model.device) for body in range(model.body_count)]
    return newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        tendon_material_direct_settle_iterations=settle_iterations,
        rigid_joint_linear_ke=1.0e8,
        rigid_joint_angular_ke=1.0e8,
        rigid_joint_linear_k_start=1.0e8,
        rigid_joint_angular_k_start=1.0e8,
        tendon_settle_tol=0.0,
        tendon_material_direct=direct,
        **solver_options,
    )


def _simulate_pulley(
    *, dt=0.001, iterations=8, duration=0.18, velocity=0.2, preload=10.0, mu=1.0, solver_options=None, damping=0.0
):
    """Observe accepted poses and independent spring energy in the VBD path."""
    with wp.ScopedDevice("cpu"):
        model, pulley = _make_pulley()
        model.tendon_link_mu.fill_(mu)
        model.tendon_seg_damping.fill_(damping)
        solver = _make_solver(model, iterations=iterations, **(solver_options or {}))
        state, next_state, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_velocity = state.body_qd.numpy().copy()
        body_velocity[pulley, 5] = velocity
        state.body_qd.assign(body_velocity)
        solver.tendon_seg_rest_length.assign(
            solver.tendon_seg_rest_length.numpy() - np.float32(preload) * model.tendon_seg_compliance.numpy()
        )
        compliance = model.tendon_seg_compliance.numpy().astype(float)
        records = []

        def record():
            poses = state.body_q.numpy().astype(float)
            link_bodies = model.tendon_link_body.numpy()
            left = solver.tendon_seg_active_link_l.numpy()
            right = solver.tendon_seg_active_link_r.numpy()
            local_l = solver.tendon_seg_attachment_l_local.numpy().astype(float)
            local_r = solver.tendon_seg_attachment_r_local.numpy().astype(float)

            def world(link, local):
                pose = poses[link_bodies[link]]
                q = pose[3:6]
                return pose[:3] + local + 2.0 * np.cross(q, np.cross(q, local) + pose[6] * local)

            length = np.array(
                [
                    np.linalg.norm(world(r, b) - world(l, a))
                    for l, r, a, b in zip(left, right, local_l, local_r, strict=True)
                ]
            )
            rest = solver.tendon_seg_rest_length.numpy().astype(float)
            wrap = abs(np.arctan2(np.cross(local_r[0], local_l[1])[2], np.dot(local_r[0], local_l[1])))
            records.append(
                (
                    2.0 * np.arctan2(poses[pulley, 5], poses[pulley, 6]),
                    float(state.body_qd.numpy()[pulley, 5]),
                    poses[pulley, :3],
                    rest,
                    np.maximum((length - rest) / compliance, 0.0),
                    np.zeros(model.tendon_segment_count),
                    wrap,
                )
            )

        record()
        for _ in range(round(duration / dt)):
            state.clear_forces()
            solver.step(state, next_state, control, None, dt)
            state, next_state = next_state, state
            record()
        solver.check_tendon_material()
        return _PulleyTrajectory(
            radius=float(model.tendon_link_radius.numpy()[1]),
            compliance=float(compliance[0]),
            inertia=float(model.body_inertia.numpy()[pulley, 2, 2]),
            dt=dt,
            **{
                field: np.asarray(values)
                for field, values in zip(
                    ("angle", "velocity", "center", "rest", "tension", "impulse", "wrap"),
                    zip(*records, strict=True),
                    strict=True,
                )
            },
        )


@wp.kernel
def _evaluate_body(
    body: int,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    adjacency: TendonForceElementAdjacencyInfo,
    link_body: wp.array[int],
    link_type: wp.array[int],
    link_radius: wp.array[float],
    link_mu: wp.array[float],
    link_offset: wp.array[wp.vec3],
    link_axis: wp.array[wp.vec3],
    link_seg_left: wp.array[int],
    rest: wp.array[float],
    local_l: wp.array[wp.vec3],
    local_r: wp.array[wp.vec3],
    compliance: wp.array[float],
    damping: wp.array[float],
    active: wp.array[int],
    active_l: wp.array[int],
    active_r: wp.array[int],
    alm_lambda: wp.array[float],
    alm_k: wp.array[float],
    alm_enabled: int,
    direct: bool,
    cone_l: wp.array[int],
    cone_r: wp.array[int],
    cap: wp.array[float],
    dt: float,
    ea_low: float,
    ea_ratio: float,
    knee: float,
    width: float,
    vectors: wp.array[wp.vec3],
    matrices: wp.array[wp.mat33],
):
    force, torque, h_ll, h_al, h_aa = evaluate_tendon_force_hessians(
        body,
        dt,
        body_q,
        body_q_prev,
        body_com,
        adjacency,
        link_body,
        link_type,
        link_radius,
        link_mu,
        link_offset,
        link_axis,
        link_seg_left,
        rest,
        local_l,
        local_r,
        compliance,
        damping,
        active,
        active_l,
        active_r,
        ea_low,
        ea_ratio,
        knee,
        width,
        alm_lambda,
        alm_k,
        alm_enabled,
        direct,
        cone_l,
        cone_r,
        cap,
    )
    vectors[0] = force
    vectors[1] = torque
    matrices[0] = h_ll
    matrices[1] = h_al
    matrices[2] = h_aa


def _body_force(model, solver, body, tension, *, direct=True, previous_poses=None, dt=0.001):
    """Set a specified elastic load, then evaluate only the body force element."""
    # The normal VBD iteration updates these rows before its body solve.
    solver._update_tendon_cone_rows(model, model.body_q, False)
    length = np.linalg.norm(solver.tendon_seg_attachment_r.numpy() - solver.tendon_seg_attachment_l.numpy(), axis=1)
    if tension is not None:
        solver.tendon_seg_rest_length.assign(length - np.asarray(tension) * model.tendon_seg_compliance.numpy())
    vectors = wp.empty(2, dtype=wp.vec3, device=model.device)
    matrices = wp.empty(3, dtype=wp.mat33, device=model.device)
    previous = model.body_q
    if previous_poses is not None:
        previous = wp.array(previous_poses, dtype=wp.transform, device=model.device)
    wp.launch(
        _evaluate_body,
        dim=1,
        inputs=[
            body,
            model.body_q,
            previous,
            model.body_com,
            solver.tendon_adjacency,
            model.tendon_link_body,
            model.tendon_link_type,
            model.tendon_link_radius,
            model.tendon_link_mu,
            model.tendon_link_offset,
            model.tendon_link_axis,
            solver.tendon_link_seg_left,
            solver.tendon_seg_rest_length,
            solver.tendon_seg_attachment_l_local,
            solver.tendon_seg_attachment_r_local,
            solver.tendon_seg_active_compliance,
            solver.tendon_seg_active_damping,
            solver.tendon_seg_active,
            solver.tendon_seg_active_link_l,
            solver.tendon_seg_active_link_r,
            solver.tendon_seg_alm_lambda,
            solver.tendon_seg_alm_k,
            (2 if solver.tendon_alm_per_segment else 1) if solver.tendon_alm else 0,
            direct,
            solver.tendon_link_cone_seg_l,
            solver.tendon_link_cone_seg_r,
            solver.tendon_link_cap_ratio,
            dt,
            solver.tendon_sigmoid_ea_low,
            solver.tendon_sigmoid_ea_ratio,
            solver.tendon_sigmoid_transition_strain,
            solver.tendon_sigmoid_transition_width,
        ],
        outputs=[vectors, matrices],
        device=model.device,
    )
    return vectors.numpy(), matrices.numpy()


def _offset_pulley(*, same_body_attachment=False, mu=0.1):
    """Place a roller away from its body's COM, optionally ending its cable on that body."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    anchor_l = builder.add_body(xform=wp.transform(p=(-0.5, 1.0, 0.0)), mass=0.0, is_kinematic=True)
    body = builder.add_body(mass=1.0, com=(0.1, -0.1, 0.0), inertia=wp.mat33(np.eye(3)), lock_inertia=True)
    anchor_r = body
    if not same_body_attachment:
        anchor_r = builder.add_body(xform=wp.transform(p=(1.5, 1.0, 0.0)), mass=0.0, is_kinematic=True)
    builder.add_tendon()
    builder.add_tendon_link(body=anchor_l, link_type=newton.TendonLinkType.ATTACHMENT)
    builder.add_tendon_link(
        body=body,
        link_type=newton.TendonLinkType.ROLLING,
        radius=0.2,
        orientation=1,
        offset=(0.5, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        mu=mu,
        compliance=1.0e-3,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=anchor_r,
        link_type=newton.TendonLinkType.ATTACHMENT,
        offset=(1.5, 1.0, 0.0) if same_body_attachment else (0.0, 0.0, 0.0),
        compliance=1.0e-3,
        rest_length=-1.0,
    )
    builder.color()
    return builder.finalize(device="cpu"), body


def _unequal_pinhole_chain():
    """Build a 1.0 m / 0.2 m pinhole chain whose equal-tension split no allocation can hold."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(mass=0.0, is_kinematic=True)
    builder.add_tendon()
    for index, offset in enumerate(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.2, 0.0, 0.0))):
        builder.add_tendon_link(
            body=body,
            link_type=int(newton.TendonLinkType.ATTACHMENT if index in (0, 2) else newton.TendonLinkType.PINHOLE),
            offset=offset,
            compliance=1.0e-3,
            rest_length=0.99,
            mu=0.1,
        )
    builder.color()
    return builder.finalize(device="cpu")


class TestTendonMaterialVBD(unittest.TestCase):
    """Check the normal VBD path against independently derived physical results."""

    def test_sticking_matches_implicit_oscillator(self):
        """Recover full sticking stiffness rather than an extra beta-scaled torque."""
        result = _simulate_pulley()
        omega_sq = 2.0 * result.radius**2 / (result.compliance * result.inertia)
        expected_angle = np.zeros_like(result.angle)
        expected_velocity = np.zeros_like(result.velocity)
        expected_velocity[0] = result.velocity[0]
        for i in range(1, len(expected_angle)):
            expected_angle[i] = (expected_angle[i - 1] + result.dt * expected_velocity[i - 1]) / (
                1.0 + result.dt**2 * omega_sq
            )
            expected_velocity[i] = expected_velocity[i - 1] - result.dt * omega_sq * expected_angle[i]
        np.testing.assert_allclose(result.angle, expected_angle, atol=1.0e-5, rtol=0.0)
        np.testing.assert_allclose(result.transfer(), result.radius * result.angle, atol=4.0e-7, rtol=0.0)
        np.testing.assert_allclose(result.center, 0.0, atol=1.0e-6, rtol=0.0)
        crossing = np.flatnonzero((result.angle[:-1] <= 0.0) & (result.angle[1:] > 0.0))
        crossing = crossing[crossing > 0]
        self.assertGreater(len(crossing), 0)
        i = crossing[0]
        period = result.dt * (i - result.angle[i] / (result.angle[i + 1] - result.angle[i]))
        self.assertAlmostEqual(period / (2.0 * math.pi / math.sqrt(omega_sq)), 1.0, delta=0.003)
        self.assertLessEqual(float(np.max(result.energy()) - result.energy()[0]), 2.0e-6)

    def test_sliding_dissipates_and_refines(self):
        """Approach the capstan-limited stopping angle and dissipate sliding energy."""
        for direction in (-1, 1):
            errors = []
            for dt in (0.001, 0.0005):
                with self.subTest(direction=direction, dt=dt):
                    result = _simulate_pulley(dt=dt, duration=0.065, velocity=direction * 2.0)
                    omega_sq = 2.0 * result.radius**2 / (result.compliance * result.inertia)
                    beta = math.tanh(0.5 * result.wrap[0])
                    slip_angle = result.compliance * float(np.mean(result.tension[0])) * beta / result.radius
                    speed_at_slip_sq = result.velocity[0] ** 2 - omega_sq * slip_angle**2
                    expected_peak = slip_angle + speed_at_slip_sq / (2.0 * omega_sq * slip_angle)
                    turns = np.flatnonzero(
                        (direction * result.velocity[:-1] > 0.0) & (direction * result.velocity[1:] <= 0.0)
                    )
                    self.assertGreater(len(turns), 0)
                    i = turns[0]
                    peak = i if abs(result.angle[i]) > abs(result.angle[i + 1]) else i + 1
                    errors.append(abs(abs(result.angle[peak]) / expected_peak - 1.0))
                    self.assertLess(errors[-1], 0.04)
                    heat = result.slip_heat()
                    self.assertGreater(float(heat[peak]), 0.0)
                    self.assertLessEqual(float(np.max(result.energy() + heat) - result.energy()[0]), 2.0e-6)
            self.assertLess(errors[1], 0.7 * errors[0])

    def test_zero_friction_does_not_resist_spin(self):
        """Keep a preloaded frictionless pulley freely rotating as a control."""
        result = _simulate_pulley(duration=0.02, mu=0.0)
        np.testing.assert_allclose(result.velocity, result.velocity[0], atol=2.0e-5, rtol=0.0)
        np.testing.assert_allclose(result.tension[:, 0], result.tension[:, 1], atol=1.0e-4, rtol=0.0)
        np.testing.assert_allclose(result.transfer(), 0.0, atol=2.0e-7, rtol=0.0)

    def test_more_iterations_do_not_accumulate_transport(self):
        """Rebuild each material trial from step-start state rather than advecting again."""
        few = _simulate_pulley(duration=0.025, iterations=8)
        many = _simulate_pulley(duration=0.025, iterations=32)
        for field in ("angle", "velocity", "rest"):
            with self.subTest(field=field):
                np.testing.assert_allclose(getattr(few, field), getattr(many, field), atol=2.0e-6, rtol=0.0)

    def test_trial_force_limits_spin_without_a_second_beta_factor(self):
        """Saturate a trial imbalance at the capstan torque, retaining the full sticking torque."""
        with wp.ScopedDevice("cpu"):
            model, pulley = _make_pulley()
            solver = _make_solver(model)
            theta = 2.0 * math.asin(0.1 / 0.4)
            beta = math.tanh(0.5 * theta)
            for tension in ([11.0, 10.0], [30.0, 10.0], [10.0, 30.0]):
                with self.subTest(tension=tension):
                    vectors, _ = _body_force(model, solver, pulley, tension)
                    difference = tension[1] - tension[0]
                    expected = 0.1 * math.copysign(min(abs(difference), beta * sum(tension)), difference)
                    self.assertAlmostEqual(float(vectors[1, 2]), expected, delta=1.0e-5)

    def test_off_center_and_same_body_forces_match_load_moments(self):
        """Retain COM leverage, full sticking torque, and the attachment's own moment."""
        for same_body in (False, True):
            for mu in (0.0, 0.1):
                with self.subTest(same_body=same_body, mu=mu), wp.ScopedDevice("cpu"):
                    model, body = _offset_pulley(same_body_attachment=same_body, mu=mu)
                    solver = _make_solver(model)
                    tension = np.array([10.5 if mu else 10.0, 10.0])
                    # These finite-friction loads are inside the cone, not on a sliding face.
                    vectors, matrices = _body_force(model, solver, body, tension)
                    self.assertLessEqual(max(tension) / min(tension), float(solver.tendon_link_cap_ratio.numpy()[1]))
                    left = solver.tendon_seg_attachment_l.numpy().astype(float)
                    right = solver.tendon_seg_attachment_r.numpy().astype(float)
                    direction_l = (left[0] - right[0]) / np.linalg.norm(left[0] - right[0])
                    direction_r = (right[1] - left[1]) / np.linalg.norm(right[1] - left[1])
                    force_l, force_r = tension[0] * direction_l, tension[1] * direction_r
                    center = model.tendon_link_offset.numpy()[1].astype(float)
                    com = model.body_com.numpy()[body].astype(float)
                    expected_force = force_l + force_r
                    if mu == 0.0:
                        # With no tangential traction all roller forces act through its axis.
                        expected_torque = np.cross(center - com, expected_force)
                    else:
                        expected_torque = np.cross(right[0] - com, force_l) + np.cross(left[1] - com, force_r)
                    if same_body:
                        expected_force -= force_r
                        expected_torque += np.cross(right[1] - com, -force_r)
                    np.testing.assert_allclose(vectors[0], expected_force, atol=4.0e-4, rtol=0.0)
                    np.testing.assert_allclose(vectors[1], expected_torque, atol=2.0e-4, rtol=0.0)
                    h_ll, h_al, h_aa = matrices
                    hessian = np.block([[h_ll, h_al.T], [h_al, h_aa]])
                    np.testing.assert_allclose(hessian, hessian.T, atol=1.0e-6, rtol=0.0)
                    self.assertGreaterEqual(float(np.linalg.eigvalsh(hessian).min()), -2.0e-4)

    def test_limiter_uses_current_pose_damping_on_both_sides(self):
        """Limit the current total spring-damper load, ignoring stale tension diagnostics."""
        with wp.ScopedDevice("cpu"):
            model, pulley = _make_pulley()
            model.tendon_seg_damping.assign(np.array([20.0, 5.0], dtype=np.float32))
            solver = _make_solver(model)
            solver.tendon_seg_material_tension.fill_(1234.0)
            solver.tendon_seg_damping_tension.fill_(-567.0)
            left = solver.tendon_seg_attachment_l.numpy().astype(float)
            right = solver.tendon_seg_attachment_r.numpy().astype(float)
            outward = [left[0] - right[0], right[1] - left[1]]
            outward = [direction / np.linalg.norm(direction) for direction in outward]
            bodies = model.tendon_link_body.numpy()
            beta = math.tanh(math.asin(0.1 / 0.4))
            for rates in ((0.3, -0.2), (-0.2, 0.3)):
                with self.subTest(length_rates=rates):
                    previous = model.body_q.numpy().copy()
                    for body, rate, direction in zip((bodies[0], bodies[2]), rates, outward, strict=True):
                        previous[body, :3] -= 0.001 * rate * direction
                    total = np.array([10.0, 10.0]) + np.array([20.0, 5.0]) * rates
                    difference = total[1] - total[0]
                    expected = 0.1 * math.copysign(min(abs(difference), beta * sum(total)), difference)
                    vectors, _ = _body_force(model, solver, pulley, [10.0, 10.0], previous_poses=previous)
                    self.assertAlmostEqual(float(vectors[1, 2]), expected, delta=1.0e-4)

    def test_dynamic_routes_repack_and_preserve_component_inventory(self):
        """Rebuild direct components across activation without changing unrelated routes."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            base = builder.add_body(mass=0.0, is_kinematic=True)
            roller = builder.add_body(xform=wp.transform(p=(0.25, 0.0, 0.0)), mass=0.0, is_kinematic=True)
            builder.add_tendon()
            points = ((0.0, 0.0, -0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0))
            for index, point in enumerate(points):
                kind = newton.TendonLinkType.PINHOLE
                if index in (0, 3):
                    kind = newton.TendonLinkType.ATTACHMENT
                elif index == 1:
                    kind = newton.TendonLinkType.ROLLING
                builder.add_tendon_link(
                    body=roller if index == 1 else base,
                    link_type=kind,
                    offset=point,
                    axis=(0.0, 1.0, 0.0),
                    radius=0.1 if index == 1 else 0.0,
                    orientation=1,
                    dynamic=index == 1,
                    compliance=1.0e-3,
                    rest_length=(-1.0, 0.45, 0.45, 0.5)[index],
                    mu=0.1,
                )
            _add_chain(builder, base, 2, height=3.0, rest_lengths=[0.97, 0.99])
            builder.color()
            model = builder.finalize(device="cpu")
            solver = _make_solver(model, iterations=3)
            state, state_out, control = model.state(), model.state(), model.control()
            initial_inventory = None
            for x, active in ((0.25, False), (0.05, True), (0.05, True), (0.25, False), (0.05, True), (0.25, False)):
                poses = state.body_q.numpy()
                poses[roller, 0] = x
                state.body_q.assign(poses)
                solver.step(state, state_out, control, None, 0.001)
                state, state_out = state_out, state
                solver.check_tendon_material()
                self.assertEqual(bool(solver.tendon_link_active.numpy()[1]), active)
                workspace = solver._tendon_material_state
                count = int(workspace.count.numpy()[0])
                np.testing.assert_array_equal(workspace.ids.numpy()[:count], [0, 1, 2] if active else [0, 2])
                np.testing.assert_array_equal(workspace.ids.numpy()[3:], [3, 4])
                rest = solver.tendon_seg_rest_length.numpy().astype(float)
                active_spans = solver.tendon_seg_active.numpy()[:3] != 0
                arc = 0.0
                if active:
                    local_r = solver.tendon_seg_attachment_r_local.numpy()[0].astype(float)
                    local_l = solver.tendon_seg_attachment_l_local.numpy()[1].astype(float)
                    angle = abs(np.arctan2(np.cross(local_r, local_l)[1], np.dot(local_r, local_l)))
                    arc = 0.1 * angle
                inventory = float(rest[:3][active_spans].sum()) + arc
                if initial_inventory is None:
                    initial_inventory = inventory
                self.assertAlmostEqual(inventory, initial_inventory, delta=2.0e-6)
                np.testing.assert_allclose(rest[3:], [0.98, 0.98], atol=2.0e-7, rtol=0.0)
                tension = solver.tendon_seg_material_tension.numpy()
                cone_l = solver.tendon_link_cone_seg_l.numpy()
                cone_r = solver.tendon_link_cone_seg_r.numpy()
                cap = solver.tendon_link_cap_ratio.numpy()
                for link in np.flatnonzero((cone_l >= 0) & (cone_r >= 0)):
                    self.assertLessEqual(float(tension[cone_l[link]] - cap[link] * tension[cone_r[link]]), 2.0e-3)
                    self.assertLessEqual(float(tension[cone_r[link]] - cap[link] * tension[cone_l[link]]), 2.0e-3)

    def test_trial_pose_rejection_is_retried_and_not_latched(self):
        """Latch a direct-material rejection on the accepted pose only, not on an iteration pose.

        VBD re-solves the material every iteration, and iteration 0 runs on the raw inertial
        predictor pose. On toy4 that pose transiently collapses a bypass span by 93% and
        over-stretches the actuator span by 16 mm; latching there froze the tendon's rest
        lengths for the rest of the run. An unequal-span chain with no material left in either
        span reproduces the same infeasible component deterministically.
        """
        with wp.ScopedDevice("cpu"):
            model = _unequal_pinhole_chain()
            solver = _make_solver(model, iterations=1)
            state = model.state()
            workspace = solver._tendon_material_state

            def set_route(rest):
                solver.tendon_seg_route_rest_length.assign(np.full(model.tendon_segment_count, rest, dtype=np.float32))

            # Equal tensions across the frictionless pinhole split the component's 1.2 m of
            # extension evenly, so the 0.2 m span cannot hold its 0.6 m share: BOUND_INCOMPATIBLE.
            set_route(1.0e-6)
            solver._update_tendon_routing(state, 0.001, False)
            np.testing.assert_array_equal(workspace.failure.numpy(), np.zeros(model.tendon_count, dtype=np.int32))
            solver.check_tendon_material()

            # The next iteration must still be solved, not skipped by a latched failure.
            set_route(0.99)
            solver._update_tendon_routing(state, 0.001, False)
            solver.check_tendon_material()
            self.assertFalse(
                np.allclose(solver.tendon_seg_rest_length.numpy(), 0.99, atol=1.0e-6),
                "a trial rejection must not stop later iterations from being solved",
            )

            # The accepted pose at the end of the step still reports the same infeasibility.
            set_route(1.0e-6)
            solver._update_tendon_routing(state, 0.001, True)
            with self.assertRaisesRegex(RuntimeError, "BOUND_INCOMPATIBLE"):
                solver.check_tendon_material()

    def test_cooperative_kernel_latches_and_reports_the_offending_span(self):
        """Apply the accepted-pose latch rule in the cooperative kernel, and name the bad span.

        The cooperative kernel is only selected on CUDA, but its warp primitives fall back to
        scalars off the device (as in ``test_tendon_material_cooperative``), so the same launch
        shape exercises it here. Without the rule, a rejection taken on the discarded predictor
        pose freezes the tendon's material state for the rest of the simulation.
        """
        with wp.ScopedDevice("cpu"):
            model = _unequal_pinhole_chain()
            solver = newton.solvers.SolverVBD(
                model,
                iterations=1,
                tendon_settle_tol=0.0,
                tendon_material_direct=True,
                tendon_sigmoid_ea_low=500.0,
                tendon_sigmoid_ea_ratio=10.0,
                tendon_sigmoid_transition_strain=0.01,
                tendon_sigmoid_transition_width=0.003,
            )
            solver._tendon_material_kernel = solve_tendon_material_cooperative
            solver._tendon_material_lanes = 32
            solver._tendon_material_block_dim = 32
            state = model.state()
            workspace = solver._tendon_material_state

            def set_route(rest):
                solver.tendon_seg_route_rest_length.assign(np.full(model.tendon_segment_count, rest, dtype=np.float32))

            set_route(1.0e-6)
            solver._update_tendon_routing(state, 0.001, False)
            np.testing.assert_array_equal(workspace.failure.numpy(), np.zeros(model.tendon_count, dtype=np.int32))
            solver.check_tendon_material()

            # The accepted pose latches, and reports the span that carries the tightest bound.
            solver._update_tendon_routing(state, 0.001, True)
            self.assertLess(int(workspace.failure.numpy()[0]), 0)
            span = int(workspace.failure_seg.numpy()[0])
            self.assertIn(span, (0, 1))
            with self.assertRaisesRegex(RuntimeError, f"Span {span} carries the tightest bound"):
                solver.check_tendon_material()
            self.assertAlmostEqual(float(workspace.failure_length.numpy()[0]), (1.0, 0.2)[span], delta=1.0e-6)
            self.assertAlmostEqual(float(workspace.failure_compliance.numpy()[0]), 1.0e-3, delta=1.0e-9)
            self.assertGreater(float(workspace.failure_tension.numpy()[0]), 0.0)
            self.assertLess(float(workspace.failure_upper.numpy()[0]), (1.0, 0.2)[span])

    def test_direct_material_holds_the_opening_iterations(self):
        """Transfer no material over the opening iteration poses of a step.

        The direct solve returns an exact allocation for whatever geometry it is handed, so the
        opening poses of a step -- iteration 0 is the raw inertial predictor, and the iterations
        that undo it are still far from the accepted pose -- produce kilonewton-scale ones on
        toy4. Sweeps never reach them because each pass is bounded. The rest lengths are still
        re-based on the step-start route, so the force kernel reads the spans it evaluates.
        """
        model = _chain_model("cpu")
        for iterations, hold, expected in ((1, 4, 0), (3, 4, 2), (8, 4, 4), (32, 4, 4), (32, 0, 0), (32, 7, 7)):
            with self.subTest(iterations=iterations, hold=hold):
                solver = _make_solver(model, iterations=iterations, settle_iterations=hold)
                self.assertEqual(solver._tendon_direct_settle_iterations(), expected)
        with self.assertRaisesRegex(ValueError, "settle_iterations must be non-negative"):
            _make_solver(model, settle_iterations=-1)

        solver = _make_solver(model, iterations=8)
        state = model.state()
        solver.tendon_seg_route_rest_length.assign(np.full(model.tendon_segment_count, 0.5, dtype=np.float32))
        held = solver.tendon_seg_rest_length.numpy().copy()
        stretch = solver.tendon_seg_stretch.numpy().copy()

        solver._update_tendon_routing(state, 0.001, False, skip_material=True)
        np.testing.assert_array_equal(solver.tendon_seg_rest_length.numpy(), 0.5)
        np.testing.assert_array_equal(solver.tendon_seg_stretch.numpy(), stretch)
        # The route geometry itself is still refreshed for the body solve.
        self.assertTrue(np.all(solver.tendon_seg_length.numpy() > 0.0))

        solver._update_tendon_routing(state, 0.001, False)
        self.assertFalse(
            np.allclose(solver.tendon_seg_rest_length.numpy(), held, atol=1.0e-6),
            "an iteration past the hold must solve the material",
        )

    def test_runtime_mode_and_nonlinear_changes_are_rejected(self):
        """Keep the experimental direct mode and selected material law immutable."""
        model = _chain_model("cpu")
        nonlinear = newton.solvers.SolverVBD(model, tendon_material_direct=True, tendon_sigmoid_ea_low=2000.0)
        self.assertTrue(nonlinear._tendon_material_state.nonlinear_enabled)
        for enabled in (False, True):
            with self.subTest(initially_enabled=enabled):
                solver = _make_solver(model, direct=enabled)
                solver.tendon_material_direct = not enabled
                with self.assertRaisesRegex(ValueError, "Reconstruct SolverVBD.*tendon_material_direct"):
                    solver.step(model.state(), model.state(), model.control(), None, 0.001)
        solver = _make_solver(model)
        solver.tendon_sigmoid_ea_low = 2000.0
        with self.assertRaisesRegex(ValueError, "Reconstruct.*material law"):
            solver.step(model.state(), model.state(), model.control(), None, 0.001)

    def test_direct_hold_reads_merged_rest_length_at_deactivation(self):
        """The hold iterations of the step that bypasses a roller must see its merged span's rest length.

        ``prepare_tendon_route`` rewires the two spans of a deactivated candidate into one and writes
        the merged rest length to the route ledger; only the material kernel re-bases the rest lengths
        the force kernel reads from it. Skipped for the settle hold, the merged span was evaluated
        against the rest length of one half-span: a phantom stretch of half the span (17.7 mm, 795 N
        to 1.7 kN on toy4's cable A) that injected 1.1 J into a 20 g arm in one substep.
        """
        dt = 1.0 / 1200.0
        model, candidate, candidate_link = build_force_driven_dynamic_route("cpu")
        solver = _make_solver(model)
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        radius = float(model.tendon_link_radius.numpy()[candidate_link])
        for _ in range(4):
            _teleport(solver, state_0, candidate, radius - 2.0e-3)
            solver.step(state_0, state_1, model.control(), None, dt)
            state_0, state_1 = state_1, state_0
        self.assertTrue(solver.tendon_link_active.numpy()[candidate_link])
        driving = float(np.max(solver.tendon_seg_material_tension.numpy()))
        self.assertGreater(driving, 0.0)

        # Step-start pose clear of the chord: this step deactivates the candidate and merges its spans.
        _teleport(solver, state_0, candidate, 0.25)
        tension = _hold_iteration_tension(solver, model, state_0, dt)
        self.assertFalse(solver.tendon_link_active.numpy()[candidate_link])
        np.testing.assert_array_equal(
            solver.tendon_seg_rest_length.numpy(), solver.tendon_seg_route_rest_length.numpy()
        )
        # The merged straight span is shorter than the wrapped route it replaces, so it cannot
        # carry more than the tension of the accepted pose (the half-span rest gave 50 N here).
        self.assertLessEqual(tension, driving)

    def test_direct_hold_splits_rest_length_at_activation(self):
        """The hold iterations of the step that activates a roller must see its split spans' rest lengths.

        The split of the merged bypass rest length onto the two new spans also runs only in the
        material kernel; skipped, the new span kept its 1 um placeholder rest length against its full
        length (18.25 mm and 839 N on toy4's cable A at re-activation).
        """
        dt = 1.0 / 1200.0
        model, candidate, candidate_link = build_force_driven_dynamic_route("cpu")
        solver = _make_solver(model)
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        for _ in range(4):
            solver.step(state_0, state_1, model.control(), None, dt)
            state_0, state_1 = state_1, state_0
        self.assertFalse(solver.tendon_link_active.numpy()[candidate_link])
        material_before = _oriented_route_material_length(solver, model, state_0, candidate_link)

        # Step-start pose 1 mm inside the chord: this step activates the candidate and splits the span.
        radius = float(model.tendon_link_radius.numpy()[candidate_link])
        _teleport(solver, state_0, candidate, radius - 1.0e-3)
        tension = _hold_iteration_tension(solver, model, state_0, dt)
        self.assertTrue(solver.tendon_link_active.numpy()[candidate_link])
        material_hold = _oriented_route_material_length(solver, model, state_0, candidate_link)
        # The split leaves both spans with the strain of the accepted route (the placeholder gave 50 N).
        solver._update_tendon_routing(state_0, dt, False)
        driving = _span_tension(solver)
        self.assertGreater(driving, 0.0)
        self.assertLessEqual(tension, 2.0 * driving)
        self.assertAlmostEqual(material_hold, material_before, delta=1.0e-6)


def _teleport(solver, state, body, x):
    """Place a dynamic body at rest: ``body_q``, the solver's ``body_q_prev`` and ``body_qd`` together."""
    body_q = state.body_q.numpy()
    body_q[body, 0] = x
    state.body_q.assign(body_q)
    state.body_qd.zero_()
    solver.body_q_prev.assign(state.body_q)


def _span_tension(solver):
    """Largest linear-law tension the force kernel derives from the current rest lengths."""
    active = solver.tendon_seg_active.numpy().astype(bool)
    stretch = solver.tendon_seg_length.numpy() - solver.tendon_seg_rest_length.numpy()
    return float(np.max(np.maximum(stretch[active], 0.0) / solver.tendon_seg_active_compliance.numpy()[active]))


def _hold_iteration_tension(solver, model, state, dt):
    """Run the step-start route update and one settle-hold routing update, as ``step`` does."""
    solver._snapshot_tendon_step_state()
    solver._update_tendon_link_active(model, state.body_q)
    solver._prepare_tendon_route(model, state.body_q, 1.0e-8)
    solver._update_tendon_routing(state, dt, False, skip_material=True)
    return _span_tension(solver)


def test_direct_vbd_failure_after_graph_replay(test, device):
    """Keep direct material capturable and report invalid runtime data after replay."""
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")
    with wp.ScopedDevice(device):
        model = _two_tendon_model(device)
        solver = _make_solver(model, iterations=2)
        state_in, state_out, control = model.state(), model.state(), model.control()
        solver.step(state_in, state_out, control, None, 0.001)
        solver.check_tendon_material()
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state_in, state_out, control, None, 0.001)
        wp.capture_launch(capture.graph)
        solver.check_tendon_material()
        mu = model.tendon_link_mu.numpy()
        mu[4] = 1.0
        model.tendon_link_mu.assign(mu)
        wp.capture_launch(capture.graph)
        with test.assertRaisesRegex(RuntimeError, r"BAD_INPUT.*tendon 1.*segment 2"):
            solver.check_tendon_material()


add_function_test(
    TestTendonMaterialVBD,
    "test_direct_vbd_failure_after_graph_replay",
    test_direct_vbd_failure_after_graph_replay,
    devices=get_test_devices(),
    check_output=False,  # The invalid runtime input deliberately emits a device-side error.
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
