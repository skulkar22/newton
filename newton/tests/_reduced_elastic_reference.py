# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Float64 reference dynamics for reduced elastic (floating frame of reference) tests.

The reference describes a lumped-mass body by its samples ``x_s = t + R (r_s + Phi_s q)``,
with ``t`` the world position of the sample-frame origin, ``R`` the frame rotation and ``q``
the modal coordinates. Its state is ``[t (3), quat xyzw (4), q (n), v_b (3), w_b (3), qdot (n)]``
with ``v_b`` the body-axes velocity of the frame origin and ``w_b`` the body-axes angular
velocity. Accelerations follow from Kane's equations ``sum_s m_s J_s^T (a_s - g) = -K q`` with
``J_s = [1, -skew(y_s), Phi_s]`` and ``y_s = r_s + Phi_s q``; no inertia term is dropped.

Newton's reduced elastic joint uses ``joint_q = [origin (3), quat xyzw (4), q]`` and
``joint_qd = [world velocity of the rest center of mass c (3), world angular velocity (3), qdot]``.
:func:`newton_to_reference` and :func:`reference_to_newton` convert between the two.
"""

import numpy as np


def rod_fixture(mode: str = "cos", nx: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the 44-sample one-mode rod used by the frame-coupling tests.

    The rod spans ``x in [-0.5, 0.5]`` m with four samples per station at ``y, z = +-0.02`` m,
    a total mass of 1 kg, and one mode along ``y``: ``cos`` (``S != 0``, ``G = 0``),
    ``cos_ortho`` (``S = 0``), or ``cos_offset`` (``S != 0``, ``G != 0``).

    Args:
        mode: Mode-shape name.
        nx: Number of stations along ``x``.

    Returns:
        Sample points [m] ``[s, 3]``, sample masses [kg] ``[s]``, and mode samples ``[s, 1, 3]``,
        all ``float32``.
    """
    points = np.array(
        [
            [x, y, z]
            for x in np.linspace(-0.5, 0.5, nx)
            for y, z in ((-0.02, -0.02), (-0.02, 0.02), (0.02, -0.02), (0.02, 0.02))
        ],
        dtype=np.float32,
    )
    mass = np.full(len(points), 1.0 / len(points), dtype=np.float32)
    phi = np.zeros((len(points), 1, 3), dtype=np.float32)
    if mode == "cos":
        phi[:, 0, 1] = np.cos(np.pi * points[:, 0])
    elif mode == "cos_ortho":
        phi[:, 0, 1] = np.cos(np.pi * points[:, 0])
        phi[:, 0, 1] -= np.sum(mass * phi[:, 0, 1]) / np.sum(mass)
    elif mode == "cos_offset":
        phi[:, 0, 1] = np.cos(np.pi * (points[:, 0] + 0.15))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return points, mass, phi


def lumped_inertia(points: np.ndarray, mass: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Return the lumped rotational inertia [kg m^2] of ``points`` about ``center``."""
    offsets = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)
    return np.sum(m * np.sum(offsets * offsets, axis=1)) * np.eye(3) - (m[:, None] * offsets).T @ offsets


def skew(v: np.ndarray) -> np.ndarray:
    """Return the cross-product matrix of ``v``."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Return the rotation matrix of a unit quaternion ``(x, y, z, w)``."""
    x, y, z, w = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Hamilton product ``a b`` of quaternions ``(x, y, z, w)``."""
    av, aw = np.asarray(a[:3], dtype=np.float64), float(a[3])
    bv, bw = np.asarray(b[:3], dtype=np.float64), float(b[3])
    return np.r_[aw * bv + bw * av + np.cross(av, bv), aw * bw - av @ bv]


def quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Return the quaternion ``(x, y, z, w)`` of a rotation vector [rad]."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1.0e-12:
        return np.r_[0.5 * rotvec, 1.0] / np.linalg.norm(np.r_[0.5 * rotvec, 1.0])
    return np.r_[np.sin(0.5 * angle) * rotvec / angle, np.cos(0.5 * angle)]


def rotvec_from_quat(quat: np.ndarray) -> np.ndarray:
    """Return the rotation vector [rad] of a quaternion ``(x, y, z, w)``."""
    quat = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    if quat[3] < 0.0:
        quat = -quat
    sin_half = float(np.linalg.norm(quat[:3]))
    if sin_half < 1.0e-12:
        return 2.0 * quat[:3]
    angle = 2.0 * np.arctan2(sin_half, quat[3])
    return angle * quat[:3] / sin_half


def rotation_angle_between(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    """Return the angle [rad] of ``R_a^T R_b``."""
    inv_a = np.r_[-np.asarray(quat_a[:3], dtype=np.float64), float(quat_a[3])]
    return float(np.linalg.norm(rotvec_from_quat(quat_multiply(inv_a, quat_b))))


def reduced_mass_brute_force(points: np.ndarray, mass: np.ndarray, phi: np.ndarray, center: np.ndarray, q: np.ndarray):
    """Return ``M(q) = sum_s m_s J_s^T J_s`` for velocities ``(v_c, w_b, qdot)`` in body axes.

    ``J_s = [1, -skew(y_s), Phi_s]`` with ``y_s = (r_s - c) + Phi_s q``, i.e. the generalized
    velocities are the body-axes velocity of the rest point ``c``, the body-axes angular velocity,
    and the modal velocities.
    """
    points = np.asarray(points, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    y = points - np.asarray(center, dtype=np.float64) + np.einsum("smc,m->sc", phi, q)
    width = 6 + phi.shape[1]
    matrix = np.zeros((width, width))
    for s in range(len(points)):
        jacobian = np.concatenate([np.eye(3), -skew(y[s]), phi[s].T], axis=1)
        matrix += float(mass[s]) * jacobian.T @ jacobian
    return matrix


def kane_derivative(
    points: np.ndarray,
    mass: np.ndarray,
    phi: np.ndarray,
    stiffness: np.ndarray,
    state: np.ndarray,
    gravity: np.ndarray | None = None,
) -> np.ndarray:
    """Return the time derivative of a reference state (see the module docstring)."""
    n = phi.shape[1]
    q = state[7 : 7 + n]
    vel = state[7 + n :]
    v_b, w_b, qd = vel[:3], vel[3:6], vel[6:]
    y = points + np.einsum("smc,m->sc", phi, q)
    u = np.einsum("smc,m->sc", phi, qd)
    R = quat_to_matrix(state[3:7])
    jac = np.concatenate(
        [np.broadcast_to(np.eye(3), (len(y), 3, 3)), -np.stack([skew(r) for r in y]), phi.transpose(0, 2, 1)], axis=2
    )
    matrix = np.einsum("sci,s,scj->ij", jac, mass, jac)
    bias = np.cross(w_b, v_b) + np.cross(w_b, np.cross(w_b, y)) + 2.0 * np.cross(w_b, u)
    load = -bias
    if gravity is not None:
        load = load + R.T @ np.asarray(gravity, dtype=np.float64)
    rhs = np.einsum("sci,s,sc->i", jac, mass, load)
    rhs[6:] -= np.asarray(stiffness, dtype=np.float64) * q
    accel = np.linalg.solve(matrix, rhs)
    quat = state[3:7]
    quat_dot = 0.5 * np.r_[quat[3] * w_b + np.cross(quat[:3], w_b), -quat[:3] @ w_b]
    return np.r_[R @ v_b, quat_dot, qd, accel]


def integrate_reference(
    points: np.ndarray,
    mass: np.ndarray,
    phi: np.ndarray,
    stiffness: np.ndarray,
    state: np.ndarray,
    duration: float,
    dt: float,
    gravity: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate a reference state with fixed-step fourth-order Runge-Kutta."""
    points = np.asarray(points, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    x = np.asarray(state, dtype=np.float64).copy()
    for _ in range(round(duration / dt)):
        k1 = kane_derivative(points, mass, phi, stiffness, x, gravity)
        k2 = kane_derivative(points, mass, phi, stiffness, x + 0.5 * dt * k1, gravity)
        k3 = kane_derivative(points, mass, phi, stiffness, x + 0.5 * dt * k2, gravity)
        k4 = kane_derivative(points, mass, phi, stiffness, x + dt * k3, gravity)
        x = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x[3:7] /= np.linalg.norm(x[3:7])
    return x


def newton_to_reference(joint_q: np.ndarray, joint_qd: np.ndarray, com: np.ndarray, mode_count: int) -> np.ndarray:
    """Convert one free reduced elastic joint state to a reference state."""
    joint_q = np.asarray(joint_q, dtype=np.float64)
    joint_qd = np.asarray(joint_qd, dtype=np.float64)
    R = quat_to_matrix(joint_q[3:7])
    omega = joint_qd[3:6]
    v_origin = joint_qd[:3] - np.cross(omega, R @ np.asarray(com, dtype=np.float64))
    return np.r_[joint_q[:7], joint_q[7 : 7 + mode_count], R.T @ v_origin, R.T @ omega, joint_qd[6 : 6 + mode_count]]


def reference_to_newton(state: np.ndarray, com: np.ndarray, mode_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a reference state to ``(joint_q, joint_qd)`` of a free reduced elastic joint."""
    R = quat_to_matrix(state[3:7])
    n = mode_count
    v_b, w_b = state[7 + n : 10 + n], state[10 + n : 13 + n]
    omega = R @ w_b
    v_com = R @ v_b + np.cross(omega, R @ np.asarray(com, dtype=np.float64))
    return state[: 7 + n].copy(), np.r_[v_com, omega, state[13 + n : 13 + 2 * n]]


def physical_state(
    points: np.ndarray, mass: np.ndarray, phi: np.ndarray, com: np.ndarray, joint_q: np.ndarray, joint_qd: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the sampled body's center of mass [m], linear momentum [kg m/s] and angular momentum about it [kg m^2/s].

    Uses Newton's convention ``v_s = v_c + w x R (y_s - c) + R Phi_s qdot``.
    """
    joint_q = np.asarray(joint_q, dtype=np.float64)
    joint_qd = np.asarray(joint_qd, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)
    n = phi.shape[1]
    R = quat_to_matrix(joint_q[3:7])
    y = np.asarray(points, dtype=np.float64) + np.einsum("smc,m->sc", phi, joint_q[7 : 7 + n])
    x = joint_q[:3] + y @ R.T
    v = (
        joint_qd[:3]
        + np.cross(joint_qd[3:6], (y - np.asarray(com, dtype=np.float64)) @ R.T)
        + np.einsum("smc,m->sc", phi, joint_qd[6 : 6 + n]) @ R.T
    )
    center = (m[:, None] * x).sum(0) / m.sum()
    momentum = (m[:, None] * v).sum(0)
    angular = (m[:, None] * np.cross(x - center, v)).sum(0)
    return center, momentum, angular
