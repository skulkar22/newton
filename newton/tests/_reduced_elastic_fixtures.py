# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Model fixtures shared by the reduced elastic test modules."""

import math

import numpy as np
import warp as wp

import newton
from newton.tests import _reduced_elastic_reference as elastic_reference


def elastic_free_joint_starts(model, body):
    """Return the ``(joint_q_start, joint_qd_start)`` of the joint attaching ``body`` to the world."""
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    for j in range(len(joint_child)):
        if int(joint_child[j]) == body and int(joint_parent[j]) == -1:
            return int(model.joint_q_start.numpy()[j]), int(model.joint_qd_start.numpy()[j])
    raise RuntimeError("no free joint")


def solve_fourbar(theta2: float, a: float, b: float, c: float, d: float) -> tuple[float, float]:
    """Return open-branch coupler and rocker angles for a planar four-bar."""
    k1 = d / a
    k2 = d / c
    k3 = (a * a - b * b + c * c + d * d) / (2.0 * a * c)
    A = k1 - math.cos(theta2)
    B = -math.sin(theta2)
    C = k2 * math.cos(theta2) - k3
    denom = math.sqrt(A * A + B * B)
    theta4 = math.atan2(B, A) + math.acos(np.clip(C / denom, -1.0, 1.0))
    cx = d + c * math.cos(theta4) - a * math.cos(theta2)
    cy = c * math.sin(theta4) - a * math.sin(theta2)
    theta3 = math.atan2(cy, cx)
    return theta3, theta4


def rod_model(
    device,
    mode: str = "cos",
    *,
    com=(0.0, 0.0, 0.0),
    position=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    spin=(0.0, 0.0, 0.0),
    q0: float = 0.02,
    qd0: float = 0.1,
    stiffness: float = 30.0,
    gravity=(0.0, 0.0, 0.0),
    rigid_control: bool = False,
):
    """Build the one-mode reference rod (see :func:`_reduced_elastic_reference.rod_fixture`).

    The physical body does not depend on ``com``: the samples are shifted by ``com`` and the frame
    origin is placed at ``position - com``, so the rest COM starts at ``position``. With
    ``rigid_control`` a plain rigid body with the rod's mass and inertia is added at the same position.
    """
    points, mass, phi = elastic_reference.rod_fixture(mode)
    offset = np.asarray(com, dtype=np.float32)
    local_points = (points + offset).astype(np.float32)
    inertia = elastic_reference.lumped_inertia(points, mass, np.zeros(3))
    basis = newton.ModalBasis(
        sample_points=local_points,
        sample_phi=phi,
        sample_mass=mass,
        mode_stiffness=[stiffness],
        mode_damping=[0.0],
    )
    builder = newton.ModelBuilder(gravity=gravity)
    origin = np.asarray(position, dtype=np.float32) - offset
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(*origin), wp.quat_identity()),
        com=wp.vec3(*offset),
        mass=float(mass.sum()),
        inertia=wp.mat33(*inertia.ravel()),
        modal_basis=basis,
        mode_q=[q0],
        mode_qd=[qd0],
    )
    rigid = -1
    if rigid_control:
        rigid = builder.add_body(
            xform=wp.transform(wp.vec3(*np.asarray(position, dtype=np.float32)), wp.quat_identity()),
            mass=float(mass.sum()),
            inertia=wp.mat33(*inertia.ravel()),
        )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    q_start, qd_start = elastic_free_joint_starts(model, body)
    joint_qd = state.joint_qd.numpy()
    joint_qd[qd_start : qd_start + 3] = velocity
    joint_qd[qd_start + 3 : qd_start + 6] = spin
    if rigid_control:
        _rigid_q_start, rigid_qd_start = elastic_free_joint_starts(model, rigid)
        joint_qd[rigid_qd_start : rigid_qd_start + 3] = velocity
        joint_qd[rigid_qd_start + 3 : rigid_qd_start + 6] = spin
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    data = {
        "points": local_points,
        "mass": mass,
        "phi": phi,
        "com": offset.astype(np.float64),
        "q_start": q_start,
        "qd_start": qd_start,
        "rigid": rigid,
    }
    return model, state, data


def rod_joint_state(state, data):
    """Return the elastic body's ``(joint_q, joint_qd)`` as float64 arrays."""
    q_start, qd_start = data["q_start"], data["qd_start"]
    return (
        state.joint_q.numpy()[q_start : q_start + 8].astype(np.float64),
        state.joint_qd.numpy()[qd_start : qd_start + 7].astype(np.float64),
    )


def rod_physical(state, data):
    """Return the rod's sampled COM, linear momentum, and angular momentum about the COM."""
    joint_q, joint_qd = rod_joint_state(state, data)
    return elastic_reference.physical_state(data["points"], data["mass"], data["phi"], data["com"], joint_q, joint_qd)


def grid_surface_mesh(half_x: float, half_y: float, count_x: int, count_y: int) -> newton.Mesh:
    """Return a flat triangle grid of ``count_x x count_y`` vertices in the local ``z = 0`` plane."""
    xs = np.linspace(-half_x, half_x, count_x, dtype=np.float32)
    ys = np.linspace(-half_y, half_y, count_y, dtype=np.float32)
    vertices = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float32)
    indices = []
    for j in range(count_y - 1):
        for i in range(count_x - 1):
            a = j * count_x + i
            c = a + count_x
            indices += [a, a + 1, c + 1, a, c + 1, c]
    return newton.Mesh(vertices, np.asarray(indices, dtype=np.int32), compute_inertia=False)


def elastic_plank_on_ground(device, *, ground: str = "mesh", count_x: int = 5, count_y: int = 3, height: float = 0.0):
    """Build a reduced elastic plank above a static ground.

    The plank is a 0.4 m x 0.2 m x 0.01 m beam basis from :class:`newton.ModalGeneratorBeam` with
    two vertical bending modes and one torsion mode (0.8 kg). Its collision mesh is a flat grid of
    ``count_x x count_y`` vertices on its mid plane, ``height`` [m] above a ground surface at
    ``z = 0``, so each grid vertex makes one elastic contact. ``ground`` selects a two-triangle
    static mesh (``"mesh"``), a ground plane (``"plane"``), or a static box (``"box"``).

    Returns:
        The finalized model and the plank body index.
    """
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.ke = 1.0e5
    cfg.kd = 0.0
    cfg.mu = 0.5
    cfg.gap = 0.01
    builder = newton.ModelBuilder()
    if ground == "mesh":
        builder.add_shape_mesh(-1, mesh=grid_surface_mesh(1.0, 1.0, 2, 2), cfg=cfg)
    elif ground == "plane":
        builder.add_ground_plane(cfg=cfg)
    elif ground == "box":
        builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()), hx=1.0, hy=1.0, hz=0.05, cfg=cfg
        )
    else:
        raise ValueError(f"unknown ground {ground!r}")
    beam = newton.ModalGeneratorBeam
    basis = beam(
        length=0.4,
        half_width_y=0.1,
        half_width_z=0.005,
        mode_specs=[
            {"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 1},
            {"type": beam.Mode.BENDING_Z, "boundary": beam.Boundary.PINNED_PINNED, "order": 2},
            {"type": beam.Mode.TORSION, "boundary": beam.Boundary.LINEAR},
        ],
        young_modulus=1.0e7,
        damping_ratio=0.05,
    ).build()
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity()), modal_basis=basis
    )
    builder.add_shape_mesh(body, mesh=grid_surface_mesh(0.2, 0.1, count_x, count_y), cfg=cfg)
    builder.color()
    return builder.finalize(device=device), body
