# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent algebra checks for the shape-independent XPBD material reduction."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.xpbd.tendon_stretch_blocks import TendonStretchRow, prepare_stretch_rows, solve_stretch_blocks
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_damped_row_uses_signed_total_tension(test, device):
    """Use the same signed Kelvin-Voigt tension for capstan classification and the XPBD residual."""
    stretch = np.array([0.0006, -0.001, -0.0001, 0.0, -0.001, 0.0006, 0.0006])
    damping_force = np.array([0.0, 80.0, 80.0, 80.0, 20.0, -40.0, -20.0])
    count = len(stretch)
    velocity = np.zeros((count + 1, 6))
    velocity[1:, 1] = np.cumsum(damping_force / 80.0)

    def array(values, dtype=float):
        return wp.array(values, dtype=dtype, device=device)

    rows = wp.empty(count, dtype=TendonStretchRow, device=device)
    material = wp.empty(count, dtype=float, device=device)
    wp.launch(
        prepare_stretch_rows,
        dim=count,
        inputs=[
            array([wp.transform(p=(0, i, 0)) for i in range(count + 1)], wp.transform),
            array(velocity, wp.spatial_vector),
            array(np.zeros((count + 1, 3)), wp.vec3),
            array(np.arange(count + 1), int),
            array(np.ones(count), int),
            array(np.arange(count), int),
            array(np.arange(1, count + 1), int),
            array([(0, i, 0) for i in range(count)], wp.vec3),
            array([(0, i + 1, 0) for i in range(count)], wp.vec3),
            array(1.0 - stretch),
            array(stretch),
            array(np.full(count, 2.0e-5)),
            array(np.full(count, 80.0)),
            0.001,
            0.0,
            20.0,
            0.001,
            0.001,
        ],
        outputs=[rows, material],
        device=device,
    )
    result = rows.numpy()
    expected = np.maximum(stretch / 2.0e-5 + damping_force, 0.0)
    np.testing.assert_allclose(result["tension"], expected, atol=1.0e-5)
    np.testing.assert_allclose(np.maximum(result["residual"] / result["compliance"], 0.0), expected, atol=1.0e-5)
    np.testing.assert_allclose(material.numpy(), np.maximum(stretch, 0.0) / 2.0e-5, atol=1.0e-5)


def test_reduced_impulse_reference(test, device):
    """Match independent dense linear algebra for sliding, sticking, and repeated bodies."""
    for ratio, forces, weights in (
        (1.0, (4.0, 4.0), (1.0, 1.0)),
        (1.7, (4.0, 6.8), (1.0, 1.7)),
        (3.0, (4.0, 6.0), None),
    ):
        for same_body in (False, True):
            dt = 0.002
            compliance = np.array([1.0e-5, 7.0e-5])
            damping = np.array([0.2, 2.0])
            old = np.array([-0.001, -0.003])
            rows = []
            jacobian = np.zeros((2, 18))
            for i in range(2):
                row = TendonStretchRow()
                row.body_l = i
                row.body_r = i + 1 if not same_body else 1
                row.linear = wp.vec3(0, 1 if i == 0 else -1, 0)
                row.angular_l = wp.vec3(0, 0, 0.06 if i == 1 else 0)
                row.angular_r = wp.vec3(0, 0, -0.06)
                if i == 1 and not same_body:
                    row.angular_r = wp.vec3(0)
                row.residual = compliance[i] * forces[i]
                row.compliance = compliance[i]
                row.motion_scale = dt + compliance[i] * damping[i]
                row.tension = forces[i]
                row.weight = 1.0
                row.next = -1
                rows.append(row)
                jacobian[i, row.body_l * 6 : row.body_l * 6 + 3] -= np.array(row.linear)
                jacobian[i, row.body_l * 6 + 3 : row.body_l * 6 + 6] += np.array(row.angular_l)
                jacobian[i, row.body_r * 6 : row.body_r * 6 + 3] += np.array(row.linear)
                jacobian[i, row.body_r * 6 + 3 : row.body_r * 6 + 6] += np.array(row.angular_r)
            inverse = np.diag([0] * 6 + [0, 0, 0, 0, 0, 5000] + [10, 10, 10, 0, 0, 0])
            coupling = (dt + compliance * damping)[:, None] * (jacobian @ inverse @ jacobian.T)
            residual = compliance * forces
            if weights is not None:
                w = np.array(weights)
                target = ((coupling @ old).sum() - residual.sum()) / (
                    (compliance * w).sum() / dt + (coupling @ w).sum()
                )
                expected = w * min(target, 0.0)
            else:
                expected = np.minimum((np.diag(coupling) * old - residual) / (compliance / dt + np.diag(coupling)), 0.0)
            poses = wp.array([wp.transform_identity()] * 3, dtype=wp.transform, device=device)
            impulses = wp.array(old, dtype=float, device=device)
            deltas = wp.zeros(2, dtype=float, device=device)
            loads = wp.zeros(3, dtype=wp.spatial_vector, device=device)
            wp.launch(
                solve_stretch_blocks,
                dim=1,
                inputs=[
                    poses,
                    wp.array([0.0, 0.0, 10.0], dtype=float, device=device),
                    wp.array(
                        [wp.mat33(0), wp.mat33(0, 0, 0, 0, 0, 0, 0, 0, 5000), wp.mat33(0)],
                        dtype=wp.mat33,
                        device=device,
                    ),
                    wp.array([0, 3], dtype=int, device=device),
                    wp.array([-1, 0, -1], dtype=int, device=device),
                    wp.array([-1, 1, -1], dtype=int, device=device),
                    wp.array([1.0, ratio, 1.0], dtype=float, device=device),
                    wp.array([0.3, 0.4], dtype=float, device=device),
                    dt,
                    1.0,
                    1.0e-5,
                ],
                outputs=[wp.array(rows, dtype=TendonStretchRow, device=device), impulses, deltas, loads],
                device=device,
            )
            np.testing.assert_allclose(impulses.numpy(), expected, rtol=2.0e-5, atol=1.0e-8)
            np.testing.assert_allclose(loads.numpy().ravel(), jacobian.T @ (expected - old), rtol=3.0e-5, atol=1.0e-8)
            if ratio == 1.0 and not same_body:
                # Initial impulses can be unequal. The correction removes that
                # old angular impulse rather than merely overwriting diagnostics.
                total = jacobian.T @ old + loads.numpy().ravel()
                test.assertAlmostEqual(total[11], 0.0, delta=1.0e-8)


class TestTendonStretchBlocks(unittest.TestCase):
    pass


def test_long_repeated_body_blocks(test, device):
    """Match dense references across long routes, sticking boundaries, and depleted spans."""
    rng = np.random.default_rng(2976)
    for count in (1, 8, 32, 128):
        with test.subTest(count=count):
            dt = 0.002
            compliance = rng.uniform(1e-5, 1e-4, count)
            damping = rng.uniform(0.0, 2.0, count)
            old = rng.uniform(-0.004, -0.001, count)
            forces = np.where(np.arange(count) % 2, 6.0, 4.0)
            rest = np.full(count, 0.3)
            if count > 1:
                rest[count // 2] = 1e-6
            ratio = np.full(count + 1, 1.5)
            if count > 1:
                ratio[count // 4 + 1] = 3.0
            jacobian = np.zeros((count, 24))
            rows = []
            for i in range(count):
                row = TendonStretchRow()
                row.body_l = 0
                row.body_r = 1 + i % 3
                row.linear = wp.vec3(0, 1, 0)
                row.angular_l = wp.vec3(0)
                row.angular_r = wp.vec3(0, 0, 0.02 + 0.01 * (i % 3))
                row.compliance = compliance[i]
                row.residual = compliance[i] * forces[i]
                row.motion_scale = dt + compliance[i] * damping[i]
                row.tension = forces[i]
                row.weight = 1.0
                row.next = -1
                rows.append(row)
                j = row.body_r * 6
                jacobian[i, j : j + 3] = np.array(row.linear)
                jacobian[i, j + 3 : j + 6] = np.array(row.angular_r)
            inverse = np.diag([0] * 6 + [1, 1, 1, 0, 0, 20] * 3)
            matrix = (dt + compliance * damping)[:, None] * (jacobian @ inverse @ jacobian.T)
            groups = [[0]]
            for i in range(1, count):
                if rest[i - 1] > 1.001e-6 and rest[i] > 1.001e-6 and ratio[i] == 1.5:
                    groups[-1].append(i)
                else:
                    groups.append([i])
            expected = np.zeros(count)
            for group in groups:
                indices = np.array(group)
                block = matrix[np.ix_(indices, indices)]
                weights = forces[indices] / max(forces[indices])
                target = ((block @ old[indices]).sum() - (compliance[indices] * forces[indices]).sum()) / (
                    (compliance[indices] * weights).sum() / dt + (block @ weights).sum()
                )
                expected[indices] = weights * min(target, 0.0)
            impulses = wp.array(old, dtype=float, device=device)
            loads = wp.zeros(4, dtype=wp.spatial_vector, device=device)
            cone_l = np.arange(-1, count, dtype=np.int32)
            cone_r = np.arange(count + 1, dtype=np.int32)
            cone_r[0] = -1
            cone_l[-1] = -1
            cone_r[-1] = -1
            wp.launch(
                solve_stretch_blocks,
                dim=1,
                inputs=[
                    wp.array([wp.transform_identity()] * 4, dtype=wp.transform, device=device),
                    wp.array([0.0, 1.0, 1.0, 1.0], dtype=float, device=device),
                    wp.array([wp.mat33(0)] + [wp.mat33(0, 0, 0, 0, 0, 0, 0, 0, 20)] * 3, dtype=wp.mat33, device=device),
                    wp.array([0, count + 1], dtype=int, device=device),
                    wp.array(cone_l, dtype=int, device=device),
                    wp.array(cone_r, dtype=int, device=device),
                    wp.array(ratio, dtype=float, device=device),
                    wp.array(rest, dtype=float, device=device),
                    dt,
                    1.0,
                    1e-5,
                ],
                outputs=[
                    wp.array(rows, dtype=TendonStretchRow, device=device),
                    impulses,
                    wp.zeros(count, dtype=float, device=device),
                    loads,
                ],
                device=device,
            )
            np.testing.assert_allclose(impulses.numpy(), expected, rtol=5e-5, atol=1e-8)
            # Include the opposite force on the kinematic anchor in the reference.
            jacobian[:, 1] = -1
            np.testing.assert_allclose(loads.numpy().ravel(), jacobian.T @ (expected - old), rtol=5e-5, atol=1e-7)


def _check_sliding_atwood_acceleration(test, device, solver_name, damping):
    with wp.ScopedDevice(device):
        for mu in (0.0, 0.1, 0.2):
            with test.subTest(mu=mu, solver=solver_name, damping=damping):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
                masses = (1.0, 3.0)
                left = builder.add_body(
                    xform=wp.transform(p=wp.vec3(-0.001, 0, 0)), mass=masses[0], inertia=wp.mat33(np.eye(3))
                )
                roller = builder.add_body(xform=wp.transform(p=wp.vec3(0, 0, 0.01)), mass=0.0, is_kinematic=True)
                right = builder.add_body(
                    xform=wp.transform(p=wp.vec3(0.001, 0, 0)), mass=masses[1], inertia=wp.mat33(np.eye(3))
                )
                builder.add_tendon()
                builder.add_tendon_link(body=left, link_type=newton.TendonLinkType.ATTACHMENT)
                builder.add_tendon_link(
                    body=roller,
                    link_type=newton.TendonLinkType.ROLLING,
                    radius=0.001,
                    axis=(0, 1, 0),
                    mu=mu,
                    compliance=1e-6,
                    damping=damping,
                )
                builder.add_tendon_link(
                    body=right, link_type=newton.TendonLinkType.ATTACHMENT, compliance=1e-6, damping=damping
                )
                builder.color()
                model = builder.finalize()
                if solver_name == "xpbd":
                    solver = newton.solvers.SolverXPBD(model, iterations=32, joint_linear_relaxation=1.0)
                else:
                    solver = newton.solvers.SolverVBD(model, iterations=256, tendon_settle_tol=1.0e-6)
                a, b = model.state(), model.state()
                dt = 0.01
                solver.step(a, b, None, None, dt)
                ratio = np.exp(mu * np.pi)
                # T_right = ratio*T_left; the two masses have equal opposite
                # acceleration in the inextensible limit. Include the small
                # compliance term for the implicit first-step reference.
                tension = (
                    2 * 9.81 / (1 / masses[0] + ratio / masses[1] + 1e-6 * (1 + ratio) / (dt * (dt + 1e-6 * damping)))
                )
                expected = np.array([tension / masses[0] - 9.81, 9.81 - ratio * tension / masses[1]])
                actual = b.body_qd.numpy()[[left, right], 2] / dt * [1, -1]
                np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=1e-3)
                if solver_name == "xpbd":
                    np.testing.assert_allclose(
                        -solver.tendon_seg_lambda.numpy() / dt, [tension, ratio * tension], atol=2e-3, rtol=1e-3
                    )


def test_sliding_atwood_acceleration(test, device):
    """Match finite-friction Atwood acceleration without collisions or changing lever arms."""
    _check_sliding_atwood_acceleration(test, device, "xpbd", 0.0)


def test_damped_sliding_atwood_acceleration(test, device):
    """Match independent implicit Atwood dynamics with damping and finite friction in both solvers."""
    for solver_name in ("xpbd", "vbd"):
        _check_sliding_atwood_acceleration(test, device, solver_name, 80.0)


add_function_test(
    TestTendonStretchBlocks,
    "test_damped_row_uses_signed_total_tension",
    test_damped_row_uses_signed_total_tension,
    devices=get_test_devices(),
)
add_function_test(
    TestTendonStretchBlocks,
    "test_reduced_impulse_reference",
    test_reduced_impulse_reference,
    devices=get_test_devices(),
)
add_function_test(
    TestTendonStretchBlocks,
    "test_long_repeated_body_blocks",
    test_long_repeated_body_blocks,
    devices=get_test_devices(),
)
add_function_test(
    TestTendonStretchBlocks,
    "test_damped_sliding_atwood_acceleration",
    test_damped_sliding_atwood_acceleration,
    devices=get_test_devices(),
)
add_function_test(
    TestTendonStretchBlocks,
    "test_sliding_atwood_acceleration",
    test_sliding_atwood_acceleration,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
