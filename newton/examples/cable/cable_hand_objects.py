# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deformable objects for the cable-driven hand: headphones with a flexible band and a soft foam ball.

Helpers of the examples ``cable_driven_hand_elastic`` and ``cable_driven_hand_soft``. Both objects are
grasped by the hand of ``cable_hand`` with massless tendons and are solved by the same
:class:`~newton.solvers.SolverVBD` as the hand:

- **Headphones** (``headphones_*``): one reduced elastic body. The headband is a circular arc of a flat
  strip; its mode shapes come from a small space-frame model of the arc center line with the two ear
  cups as rigid masses on the ends. The lowest free-free modes are sampled on the collision meshes and
  on a set of mass points and handed to :class:`newton.ModalGeneratorSampled`. The ear cups move rigidly
  with the band ends, so the whole object is one body with ``6 + mode_count`` coordinates.
- **Soft object** (``soft_*``): a tetrahedral mesh of particles (finite elements), a plain soft
  ball of foam. The pads of the hand and the support are the only shapes that collide with the
  particles, and the contact acts on both sides. Contact acts at the particle radius, so the surface
  to draw is the display surface of :func:`create_soft_display_surface`, not the particle centers.

In both scenes the object is free: no joint, weld or attachment ties it to the hand. It is held by the
contact and the friction of the pads. Lengths are in meters, forces in newtons, angles in radians.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.examples.cable import cable_hand as ch

# ---------------------------------------------------------------------------
# Headphones (one reduced elastic body: flexible headband, rigid ear cups)
# ---------------------------------------------------------------------------

HEADPHONES = {
    # headband: strip on a circular arc in the body y-z plane, width along body x, crown at +z
    "band_radius": 0.0595,  # center line radius [m]
    "band_lean": math.radians(9.0),  # the ends stop this far above the horizontal [rad]
    "band_width": 0.026,  # along x [m]
    "band_thickness": 0.005,  # radial [m]
    "band_density": 1100.0,  # [kg/m^3]
    "band_young": 20.0e6,  # [Pa]
    "band_poisson": 0.40,
    # crown cushion: soft pad under the inner crown; part of the collision mesh, no mass, no stiffness
    "cushion_depth": 0.025,  # at the crown [m]
    "cushion_half_angle": math.radians(40.0),  # cos^2 taper to zero at this arc angle from the crown [rad]
    # ear cups: rigid cylinders (axis along body y) hanging on a yoke below each band end
    "cup_radius": 0.036,  # [m]
    "cup_depth": 0.026,  # [m]
    "cup_mass": 0.045,  # each, yoke included [kg]
    "yoke_length": 0.040,  # band end to cup center, along -z [m]
    # reduction
    "mode_count": 6,
    "damping_ratio": 0.15,
    "elements": 48,
}
"""Dimensions and material of the headphones: a flexible plastic headband of 24 g and two ear cups of 45 g.

The band modulus of 20 MPa is chosen for a band that bends visibly in the hand (the ear cups come 18 mm
closer in the hold, 15 % of their distance) and has settled at the end of the hold. At half of it the
band still swings at the end of the hold; at 2.5 times (50 MPa) the cups come 7 mm closer."""

HEADPHONES_SUPPORT_HEIGHT = 0.020
"""Height of the lowest point of the ear cups above the table when the headphones sit in their cradles [m]."""

HEADPHONES_CRADLE_ANGLE = math.radians(35.0)
"""Angle from the vertical at which the two cradle faces touch an ear cup [rad]."""

HEADPHONES_CONTACT_BUFFER = 256
"""Contact buffer per body for the headphones scene: the band meets the pads at many mesh vertices."""

HEADPHONES_HOLD_TENSION = 20.0
"""Cable tension of the four fingers in the hold of the headphones [N]. The headphones weigh 1.1 N; at
the 40 N of the rigid objects the little finger creeps along the cushion and snaps over in the hold."""

HEADPHONES_FRICTION_EPSILON = 1.0e-3
"""Slip speed below which the solver's friction is regularized [m/s] (``friction_epsilon``). At the
solver default of 1e-2 the fingertips creep on the cushion and the band keeps closing in the hold."""


@dataclass
class HeadphonesModel:
    """Reduced model and meshes of the headphones in the body frame, see :func:`headphones_reduced_model`.

    Body frame: origin at the center of the headband arc, the band in the
    ``y-z`` plane with the crown at ``+z``, the band width along ``x``.
    """

    basis: newton.ModalBasis
    """Modal basis: mass points, mesh vertices, mode values, modal stiffness and damping."""
    frequencies: np.ndarray
    """Natural frequencies of the retained free-free modes [Hz], shape ``(mode_count,)``."""
    mode_stiffness: np.ndarray
    """Modal stiffness [N/m], shape ``(mode_count,)``; the modal coordinates are displacements [m]."""
    mass: float
    """Total mass [kg]."""
    com: np.ndarray
    """Center of mass in the body frame [m], shape ``(3,)``."""
    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    """Vertices [m], shape ``(n, 3)``, and triangle indices, shape ``(3 m,)``, of ``"band"`` (with the
    crown cushion), ``"cup_left"`` (``+y``), ``"cup_right"``, ``"yoke_left"`` and ``"yoke_right"``."""
    cup_centers: np.ndarray = None
    """Centers of the left and right ear cup in the body frame [m], shape ``(2, 3)``."""
    cup_phi: np.ndarray = None
    """Mode values at the two cup centers, shape ``(2, mode_count, 3)``."""
    crown: np.ndarray = None
    """Top of the crown (outer surface) in the body frame [m], shape ``(3,)``."""
    bottom: float = 0.0
    """Body ``z`` of the lowest point of the ear cups [m]."""


def _frame_element_stiffness(length: float, ea: float, gj: float, ei_y: float, ei_z: float) -> np.ndarray:
    """Stiffness of a two-node beam element in its own axes; node DoFs ``u, v, w, rx, ry, rz``."""
    k = np.zeros((12, 12))
    l2, l3 = length * length, length**3
    for a, b, value in ((0, 6, ea / length), (3, 9, gj / length)):
        k[a, a] = k[b, b] = value
        k[a, b] = k[b, a] = -value
    # bending with deflection v (rotation rz) and with deflection w (rotation ry)
    for d, r, ei, sign in ((1, 5, ei_z, 1.0), (2, 4, ei_y, -1.0)):
        d2, r2 = d + 6, r + 6
        k[d, d] = k[d2, d2] = 12.0 * ei / l3
        k[d, d2] = k[d2, d] = -12.0 * ei / l3
        k[r, r] = k[r2, r2] = 4.0 * ei / length
        k[r, r2] = k[r2, r] = 2.0 * ei / length
        for i, j, s in ((d, r, 1.0), (d, r2, 1.0), (d2, r, -1.0), (d2, r2, -1.0)):
            k[i, j] = k[j, i] = sign * s * 6.0 * ei / l2
    return k


def _skew(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors by a quaternion ``x, y, z, w``."""
    return v + 2.0 * np.cross(q[..., :3], np.cross(q[..., :3], v) + q[..., 3:4] * v)


def headphones_cushion_depth(angle, params: dict[str, Any] | None = None):
    """Depth of the crown cushion below the inner band surface.

    Args:
        angle: Arc angle from the crown [rad], a number or an array.
        params: Parameters that replace those of :data:`HEADPHONES`.

    Returns:
        The depth [m], with the shape of ``angle``.
    """
    p = {**HEADPHONES, **(params or {})}
    u = np.clip(np.abs(angle) / p["cushion_half_angle"], 0.0, 1.0)
    return p["cushion_depth"] * np.cos(0.5 * math.pi * u) ** 2


def headphones_reduced_model(
    stiffness_scale: float = 1.0,
    params: dict[str, Any] | None = None,
    width_points: int = 7,
    side_points: int = 5,
    cup_segments: int = 40,
) -> HeadphonesModel:
    """Build the modal basis and the meshes of the headphones.

    The center line of the headband is a chain of beam elements with the
    section stiffness of the strip (soft bending across the thickness, stiff
    bending across the width, torsion of a thin rectangle) and lumped masses.
    Each ear cup is a rigid mass on the end node. The lowest free-free modes
    of this model (the six rigid motions removed) are the mode shapes; they
    are scaled so that the largest displacement of a center line point or a
    cup center is one, which makes a modal coordinate a displacement [m].
    Free-free modes carry no net momentum, so the weight of the object is a
    force at its center of mass and does not load the modes; the band flexes
    under the contact forces of whatever holds it. Every mesh
    vertex and every mass point belongs to one node and moves rigidly with
    it, so plane sections of the band stay plane and the cups stay rigid.

    The mode shapes are linear: the model suits deflections of some
    millimeters to a centimeter and does not stiffen when the band is
    stretched flat. It is meant to look right and stay stable, not to
    predict deflections.

    Args:
        stiffness_scale: Factor on the Young's modulus of the band (250 makes a practically rigid band).
        params: Parameters that replace those of :data:`HEADPHONES`.
        width_points: Mesh vertices across the band width on the outer and on the inner surface.
        side_points: Mesh vertices on each side face of the band and cushion, edges not counted.
        cup_segments: Vertices around an ear cup.

    Returns:
        The reduced model.
    """
    p = {**HEADPHONES, **(params or {})}
    radius, width, thickness = p["band_radius"], p["band_width"], p["band_thickness"]
    young = p["band_young"] * stiffness_scale
    shear = young / (2.0 * (1.0 + p["band_poisson"]))
    n_el = int(p["elements"])
    n_node = n_el + 1
    n_mode = int(p["mode_count"])
    angle = np.linspace(p["band_lean"], math.pi - p["band_lean"], n_node)  # from the +y end over the crown
    e_r = np.stack([np.zeros(n_node), np.cos(angle), np.sin(angle)], axis=1)  # radial, outward
    center = radius * e_r
    x_axis = np.array([1.0, 0.0, 0.0])

    # stiffness and mass of the frame model
    area = width * thickness
    ei_soft = young * width * thickness**3 / 12.0
    ei_stiff = young * thickness * width**3 / 12.0
    gj = shear * width * thickness**3 / 3.0 * (1.0 - 0.63 * thickness / width)
    stiffness = np.zeros((6 * n_node, 6 * n_node))
    mass = np.zeros((6 * n_node, 6 * n_node))
    node_mass = np.zeros(n_node)
    for e in range(n_el):
        d = center[e + 1] - center[e]
        length = float(np.linalg.norm(d))
        e1 = d / length
        e2 = x_axis
        e3 = np.cross(e1, e2)
        rot = np.kron(np.eye(4), np.stack([e1, e2, e3]))
        dofs = np.r_[6 * e : 6 * e + 12]
        k_local = _frame_element_stiffness(length, young * area, gj, ei_soft, ei_stiff)
        stiffness[np.ix_(dofs, dofs)] += rot.T @ k_local @ rot
        node_mass[e : e + 2] += 0.5 * p["band_density"] * area * length
    # lumped band mass: four points per node reproduce the mass and the second moments of the section
    a_off = width / math.sqrt(12.0)
    b_off = thickness / math.sqrt(12.0)
    mass_points, mass_values, mass_node, mass_offset = [], [], [], []
    for k in range(n_node):
        for sa in (-1.0, 1.0):
            for sb in (-1.0, 1.0):
                offset = sa * a_off * x_axis + sb * b_off * e_r[k]
                mass_points.append(center[k] + offset)
                mass_values.append(0.25 * node_mass[k])
                mass_node.append(k)
                mass_offset.append(offset)
    # ear cups: six points each reproduce the mass and the inertia of a solid cylinder with its axis along y
    cup_r, cup_h, cup_m = p["cup_radius"], p["cup_depth"], p["cup_mass"]
    arm = np.sqrt(3.0 * np.array([cup_r**2 / 4.0, cup_h**2 / 12.0, cup_r**2 / 4.0]))
    cup_centers = np.stack([center[k] - np.array([0.0, 0.0, p["yoke_length"]]) for k in (0, n_node - 1)])
    for side, k in enumerate((0, n_node - 1)):
        for axis in range(3):
            for s in (-1.0, 1.0):
                offset = cup_centers[side] - center[k] + s * arm[axis] * np.eye(3)[axis]
                mass_points.append(center[k] + offset)
                mass_values.append(cup_m / 6.0)
                mass_node.append(k)
                mass_offset.append(offset)
    mass_points = np.array(mass_points)
    mass_values = np.array(mass_values)
    mass_node = np.array(mass_node)
    mass_offset = np.array(mass_offset)
    for m, k, offset in zip(mass_values, mass_node, mass_offset, strict=True):
        # velocity of the point: v + omega x offset
        b = np.hstack([np.eye(3), -_skew(offset)])
        dofs = np.r_[6 * k : 6 * k + 6]
        mass[np.ix_(dofs, dofs)] += m * b.T @ b

    # free-free modes: K x = lambda M x through the Cholesky factor of M
    chol = np.linalg.cholesky(mass)
    reduced = np.linalg.solve(chol, np.linalg.solve(chol, stiffness).T).T
    eigenvalues, vectors = np.linalg.eigh(0.5 * (reduced + reduced.T))
    shapes = np.linalg.solve(chol.T, vectors[:, 6 : 6 + n_mode])  # the first six are the rigid motions
    lam = eigenvalues[6 : 6 + n_mode]
    u = shapes.reshape(n_node, 6, n_mode)[:, :3, :]  # node displacement per mode
    th = shapes.reshape(n_node, 6, n_mode)[:, 3:, :]  # node rotation per mode

    def mode_values(nodes, offsets):
        """Displacement of points that ride on nodes: ``u + theta x offset``, shape ``(n, mode_count, 3)``."""
        nodes = np.asarray(nodes)
        offsets = np.asarray(offsets, dtype=float)
        un = np.transpose(u[nodes], (0, 2, 1))
        tn = np.transpose(th[nodes], (0, 2, 1))
        return un + np.cross(tn, offsets[:, None, :])

    ends = np.array([0, n_node - 1])
    cup_phi = mode_values(ends, cup_centers - center[ends])
    # scale: the largest displacement of a center line node or a cup center is one
    scale = np.maximum(np.linalg.norm(cup_phi, axis=2).max(axis=0), np.linalg.norm(u, axis=1).max(axis=0))
    # sign: the +y cup moves toward +y, else +x, else +z (whichever component is largest)
    for i in range(n_mode):
        j = int(np.argmax(np.abs(cup_phi[0, i])))
        if cup_phi[0, i, j] < 0.0:
            scale[i] = -scale[i]
    u = u / scale
    th = th / scale
    cup_phi = cup_phi / scale[None, :, None]
    shapes = shapes / scale
    mode_stiffness = np.einsum("ai,ab,bi->i", shapes, stiffness, shapes)
    mode_mass = np.einsum("ai,ab,bi->i", shapes, mass, shapes)
    mode_damping = 2.0 * p["damping_ratio"] * np.sqrt(mode_stiffness * mode_mass)

    # meshes; every vertex rides on one node
    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    vertex_node: dict[str, np.ndarray] = {}
    vertex_offset: dict[str, np.ndarray] = {}

    # band with the crown cushion: one closed tube, a ring of vertices per node
    depth = headphones_cushion_depth(angle - 0.5 * math.pi, p)
    across = np.linspace(-0.5 * width, 0.5 * width, width_points)
    ring_offsets = []
    for k in range(n_node):
        top = 0.5 * thickness
        bottom = -0.5 * thickness - depth[k]
        side = np.linspace(top, bottom, side_points + 2)[1:-1]
        ring = [(a, top) for a in across]  # outer surface, -x to +x
        ring += [(0.5 * width, b) for b in side]  # +x side, outer to inner
        ring += [(a, bottom) for a in across[::-1]]  # inner surface, +x to -x
        ring += [(-0.5 * width, b) for b in side[::-1]]  # -x side, inner to outer
        ring_offsets.append([a * x_axis + b * e_r[k] for a, b in ring])
    ring_offsets = np.array(ring_offsets)
    n_ring = ring_offsets.shape[1]
    nodes_band = np.repeat(np.arange(n_node), n_ring)
    offsets_band = ring_offsets.reshape(-1, 3)
    triangles = []
    for k in range(n_node - 1):
        for j in range(n_ring):
            a0, a1 = k * n_ring + j, k * n_ring + (j + 1) % n_ring
            b0, b1 = a0 + n_ring, a1 + n_ring
            triangles += [(a0, a1, b1), (a0, b1, b0)]
    # end caps: a fan around the center line point
    cap0, cap1 = n_node * n_ring, n_node * n_ring + 1
    nodes_band = np.concatenate([nodes_band, [0, n_node - 1]])
    offsets_band = np.concatenate([offsets_band, np.zeros((2, 3))])
    for j in range(n_ring):
        triangles.append((cap0, (j + 1) % n_ring, j))
        last = (n_node - 1) * n_ring
        triangles.append((cap1, last + j, last + (j + 1) % n_ring))
    band_vertices = center[nodes_band] + offsets_band
    band_indices = np.array(triangles, dtype=np.int32)
    # orient outward: the first triangle's normal must point away from the center line
    tri = band_vertices[band_indices[0]]
    if np.dot(np.cross(tri[1] - tri[0], tri[2] - tri[0]), tri.mean(axis=0) - center[0]) < 0.0:
        band_indices = band_indices[:, ::-1]
    meshes["band"] = (band_vertices, band_indices.reshape(-1))
    vertex_node["band"] = nodes_band
    vertex_offset["band"] = offsets_band

    # ear cups: closed cylinders along y with a ring at each face and one in the middle
    ring_angle = np.linspace(0.0, 2.0 * math.pi, cup_segments, endpoint=False)
    circle = np.stack([np.cos(ring_angle), np.zeros(cup_segments), np.sin(ring_angle)], axis=1)
    for side, (name, k) in enumerate((("left", 0), ("right", n_node - 1))):
        local = []
        for y in (-0.5 * cup_h, 0.0, 0.5 * cup_h):
            local += [cup_r * c + np.array([0.0, y, 0.0]) for c in circle]
        local += [np.array([0.0, -0.5 * cup_h, 0.0]), np.array([0.0, 0.5 * cup_h, 0.0])]
        local = np.array(local)
        tris = []
        n = cup_segments
        for row in range(2):
            for j in range(n):
                a0, a1 = row * n + j, row * n + (j + 1) % n
                tris += [(a0, a0 + n, a1 + n), (a0, a1 + n, a1)]
        for j in range(n):
            tris.append((3 * n, j, (j + 1) % n))
            tris.append((3 * n + 1, 2 * n + (j + 1) % n, 2 * n + j))
        tris = np.array(tris, dtype=np.int32)
        t0 = local[tris[0]]
        if np.dot(np.cross(t0[1] - t0[0], t0[2] - t0[0]), t0.mean(axis=0)) < 0.0:
            tris = tris[:, ::-1]
        offsets = cup_centers[side] - center[k] + local
        meshes[f"cup_{name}"] = (center[k] + offsets, tris.reshape(-1))
        vertex_node[f"cup_{name}"] = np.full(len(local), k)
        vertex_offset[f"cup_{name}"] = offsets
        # yoke: a slim bar from the band end to the cup center (drawn only)
        hx, hy = 0.5 * width, 0.002
        corners = np.array(
            [(sx * hx, sy * hy, z) for z in (0.0, -p["yoke_length"]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        )
        quads = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
        ytris = np.array([t for q in quads for t in ((q[0], q[2], q[1]), (q[0], q[3], q[2]))], dtype=np.int32)
        meshes[f"yoke_{name}"] = (center[k] + corners, ytris.reshape(-1))
        vertex_node[f"yoke_{name}"] = np.full(len(corners), k)
        vertex_offset[f"yoke_{name}"] = corners

    # samples: mass points first, then the massless mesh vertices
    points = [mass_points]
    values = [mode_values(mass_node, mass_offset)]
    rotations = [np.transpose(th[mass_node], (0, 2, 1))]
    masses = [mass_values]
    for name, (vertices, _) in meshes.items():
        points.append(np.asarray(vertices, dtype=np.float32))
        values.append(mode_values(vertex_node[name], vertex_offset[name]))
        rotations.append(np.transpose(th[vertex_node[name]], (0, 2, 1)))
        masses.append(np.zeros(len(vertices)))
        meshes[name] = (points[-1], meshes[name][1])
    basis = newton.ModalGeneratorSampled(
        sample_points=np.concatenate(points),
        sample_phi=np.concatenate(values),
        sample_psi=np.concatenate(rotations),
        sample_mass=np.concatenate(masses),
        mode_stiffness=mode_stiffness,
        mode_damping=mode_damping,
        label="headphones",
    ).build()
    total = float(mass_values.sum())
    return HeadphonesModel(
        basis=basis,
        frequencies=np.sqrt(np.maximum(lam, 0.0)) / (2.0 * math.pi),
        mode_stiffness=mode_stiffness,
        mass=total,
        com=(mass_values[:, None] * mass_points).sum(axis=0) / total,
        meshes=meshes,
        cup_centers=cup_centers,
        cup_phi=cup_phi,
        crown=np.array([0.0, 0.0, radius + 0.5 * thickness]),
        bottom=float(cup_centers[0, 2] - cup_r),
    )


@dataclass
class HeadphonesScene:
    """Indices and geometry of the headphones in a grasp scene, see :func:`add_headphones_scene`."""

    pads: ch.HandPads
    """Pads of the hand."""
    reduced: HeadphonesModel
    """Reduced model and meshes of the headphones."""
    object_body: int
    """Body index of the headphones, a free reduced elastic body."""
    object_xform: wp.transform
    """World pose of the headphones frame at build time."""
    object_mass: float
    """Mass of the headphones [kg]."""
    shapes: dict[str, int]
    """Shape index of ``"band"``, ``"cup_left"``, ``"cup_right"``, ``"yoke_left"`` and ``"yoke_right"``."""
    table_shape: int
    """Shape index of the table, -1 without one."""
    cradle_shapes: list[int]
    """Shape indices of the static cradle faces and posts under the ear cups."""
    q_start: int = -1
    """Index of the first coordinate of the headphones in ``joint_q`` (position, quaternion, modal coordinates)."""
    qd_start: int = -1
    """Index of the first velocity of the headphones in ``joint_qd``."""
    mode_count: int = 0
    """Number of modal coordinates."""
    mesh_samples: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    """Per mesh name, the rest positions of its vertices in the body frame [m], shape ``(n, 3)``, and
    their mode values, shape ``(n, mode_count, 3)``; filled by :func:`headphones_mesh_vertices`."""


def headphones_rest_xform(support_height: float = HEADPHONES_SUPPORT_HEIGHT, params: dict[str, Any] | None = None):
    """World pose of the headphones sitting in their cradles: upright above the origin, the band along world ``y``.

    Args:
        support_height: Height of the lowest point of the ear cups above the table [m].
        params: Parameters that replace those of :data:`HEADPHONES`.

    Returns:
        The pose of the headphones frame (the center of the band arc).
    """
    p = {**HEADPHONES, **(params or {})}
    bottom = p["band_radius"] * math.sin(p["band_lean"]) - p["yoke_length"] - p["cup_radius"]
    return wp.transform(wp.vec3(0.0, 0.0, support_height - bottom), wp.quat_identity())


def add_headphones_scene(
    builder: newton.ModelBuilder,
    spec: ch.HandSpec,
    pads: ch.HandPads,
    reduced: HeadphonesModel | None = None,
    object_xform: wp.transform | None = None,
    table_half_size: tuple[float, float, float] = (0.2, 0.2, 0.01),
    contact_ke: float | None = None,
) -> HeadphonesScene:
    """Add the table, two cradles and the headphones (one reduced elastic body) to a builder.

    The headphones are a free reduced elastic body in an articulation of
    their own, so :class:`~newton.solvers.SolverVBD` treats them with the
    local block solve of the frame and the modal coordinates. Contacts are
    generated between the vertices of their meshes and the rigid shapes
    (pads, cradles, table). Shapes and friction follow
    ``cable_hand.add_grasp_scene``.

    Each ear cup rests on two inclined static faces that keep it from rolling.

    Args:
        builder: The model builder; ``z`` is up and the table top is ``z = 0``.
        spec: The hand spec.
        pads: The pads of the hand, from ``cable_hand.add_hand_pads``.
        reduced: The reduced model; the default of :func:`headphones_reduced_model` if ``None``.
        object_xform: World pose of the headphones frame; :func:`headphones_rest_xform` if ``None``.
        table_half_size: Half extents of the table box [m].
        contact_ke: Contact stiffness [N/m]; the pad contact stiffness of the spec if ``None``.

    Returns:
        The scene. Call :func:`finish_headphones_scene` after finalizing the model.
    """
    if reduced is None:
        reduced = headphones_reduced_model()
    if object_xform is None:
        object_xform = headphones_rest_xform()
    surface_mu = spec.pads.table_mu**2 / spec.pads.mu
    surface_cfg = ch.grasp_shape_config(spec, surface_mu, contact_ke=contact_ke)
    table_shape = builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -table_half_size[2]), wp.quat_identity()),
        hx=table_half_size[0],
        hy=table_half_size[1],
        hz=table_half_size[2],
        cfg=surface_cfg,
        color=(0.78, 0.78, 0.76),
        label="grasp/table",
    )
    cradle_shapes = []
    cup_r = HEADPHONES["cup_radius"]
    alpha = HEADPHONES_CRADLE_ANGLE
    half = (0.010, 0.016, 0.003)
    for side, cup in enumerate(reduced.cup_centers):
        c = np.array(wp.transform_point(object_xform, wp.vec3(*cup.tolist())), dtype=float)
        for sign in (-1.0, 1.0):
            normal = np.array([-sign * math.sin(alpha), 0.0, math.cos(alpha)])
            touch = c - cup_r * normal
            position = touch - half[2] * normal
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -sign * alpha)
            cradle_shapes.append(
                builder.add_shape_box(
                    body=-1,
                    xform=wp.transform(wp.vec3(*position.tolist()), rot),
                    hx=half[0],
                    hy=half[1],
                    hz=half[2],
                    cfg=surface_cfg,
                    color=(0.45, 0.45, 0.45),
                    label=f"grasp/cradle{side}{'ab'[sign > 0]}",
                )
            )
        # post under the cradle (keeps clear of the cup)
        top = c[2] - cup_r - 0.004
        if top > 0.002:
            cradle_shapes.append(
                builder.add_shape_box(
                    body=-1,
                    xform=wp.transform(wp.vec3(c[0], c[1], 0.5 * top), wp.quat_identity()),
                    hx=0.006,
                    hy=half[1],
                    hz=0.5 * top,
                    cfg=surface_cfg,
                    color=(0.45, 0.45, 0.45),
                    label=f"grasp/cradle_post{side}",
                )
            )
    body = builder.add_body_elastic(xform=object_xform, modal_basis=reduced.basis, label="grasp/headphones")
    cfg = ch.grasp_shape_config(spec, spec.pads.mu, contact_ke=contact_ke)
    visual = ch.grasp_shape_config(spec, spec.pads.mu, contact_ke=contact_ke)
    visual.has_shape_collision = False
    visual.has_particle_collision = False
    colors = {"band": (0.16, 0.17, 0.2), "cup": (0.1, 0.25, 0.6), "yoke": (0.7, 0.7, 0.72)}
    shapes = {}
    for name, (vertices, indices) in reduced.meshes.items():
        kind = name.split("_")[0]
        shapes[name] = builder.add_shape_mesh(
            body=body,
            mesh=newton.Mesh(vertices, indices, compute_inertia=False),
            cfg=visual if kind == "yoke" else cfg,
            color=colors[kind],
            label=f"headphones/{name}",
        )
    return HeadphonesScene(
        pads=pads,
        reduced=reduced,
        object_body=body,
        object_xform=object_xform,
        object_mass=reduced.mass,
        shapes=shapes,
        table_shape=table_shape,
        cradle_shapes=cradle_shapes,
        mode_count=reduced.basis.mode_count,
    )


def finish_headphones_scene(model: newton.Model, scene: HeadphonesScene) -> HeadphonesScene:
    """Look up the coordinates of the headphones in a finalized model.

    Args:
        model: The finalized model.
        scene: The scene from :func:`add_headphones_scene`.

    Returns:
        The scene with ``q_start`` and ``qd_start`` set.
    """
    elastic = int(model.body_elastic_index.numpy()[scene.object_body])
    joint = int(model.elastic_joint.numpy()[elastic])
    scene.q_start = int(model.joint_q_start.numpy()[joint])
    scene.qd_start = int(model.joint_qd_start.numpy()[joint])
    return scene


def _add_tendon_hand(builder: newton.ModelBuilder, spec: ch.HandSpec, wrist: str, palm_xform, visible_pads: bool):
    """Add the hand with massless tendons and its pads, as ``cable_hand.build_tendon_hand_grasp_model`` does."""
    hand = ch.add_hand_skeleton(builder, spec, wrist, palm_xform, 0.0)
    tendons = ch.add_hand_tendons(builder, spec, hand)
    ch.add_hand_articulation(builder, hand)
    pads = ch.add_hand_pads(builder, spec, hand, visible=visible_pads)
    return hand, tendons, pads


def build_hand_headphones_model(
    spec: ch.HandSpec | None = None,
    reduced: HeadphonesModel | None = None,
    object_xform: wp.transform | None = None,
    palm_xform: wp.transform | None = None,
    wrist: str = "kinematic",
    device=None,
    visuals: bool = False,
) -> tuple[newton.Model, ch.HandSkeleton, dict[str, Any], HeadphonesScene]:
    """Build and finalize the hand with massless tendons, pads, a table, two cradles and the headphones.

    The hand is that of ``cable_hand.build_tendon_hand_grasp_model``. The
    model has no gravity: pass ``gravity=(0, 0, -GRAVITY)`` to
    ``cable_hand.create_hand_control``, which also applies the weight of the
    headphones. The headphones are a free body: nothing but contact connects
    them to the hand, the cradles or the table.

    Args:
        spec: The hand spec; the default spec if ``None``.
        reduced: The reduced model of the headphones; the default if ``None``.
        object_xform: World pose of the headphones frame; in the cradles if ``None``.
        palm_xform: World pose of the palm frame at the start; identity if ``None``.
        wrist: ``"kinematic"`` or ``"fixed"``.
        device: Device of the model.
        visuals: If ``True``, attach the shipped visual meshes of the hand and hide the pad capsules.

    Returns:
        The finalized model, the hand skeleton, the tendon indices per actuator key and the scene.
    """
    if spec is None:
        spec = ch.default_hand_spec()
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    hand, tendons, pads = _add_tendon_hand(builder, spec, wrist, palm_xform, not visuals)
    scene = add_headphones_scene(builder, spec, pads, reduced, object_xform)
    if visuals:
        ch.add_hand_visuals(builder, hand, spec=spec)
    builder.color()
    model = builder.finalize(device=device)
    return model, hand, tendons, finish_headphones_scene(model, scene)


def make_hand_headphones_solver(model: newton.Model, **overrides) -> newton.solvers.SolverVBD:
    """Create the solver for the hand with the headphones.

    The settings of ``cable_hand.make_tendon_hand_grasp_solver`` with a
    contact buffer of :data:`HEADPHONES_CONTACT_BUFFER` per body and the
    friction regularization :data:`HEADPHONES_FRICTION_EPSILON`. Create the
    collision pipeline (``cable_hand.create_grasp_collision_pipeline``) first.

    Args:
        model: The model from :func:`build_hand_headphones_model`.
        **overrides: Solver keyword arguments that replace the defaults.

    Returns:
        The solver.
    """
    overrides.setdefault("rigid_body_contact_buffer_size", HEADPHONES_CONTACT_BUFFER)
    overrides.setdefault("friction_epsilon", HEADPHONES_FRICTION_EPSILON)
    return ch.make_tendon_hand_grasp_solver(model, **overrides)


HEADPHONES_GRASP = {
    # Hook grasp from above: the palm faces down, the fingers point along world +x across the band and the
    # thumb side is world +y. The flat outer surface of the crown (26 mm wide) lies against the two palm
    # pads, which are 26 mm apart; the four fingers curl under the crown cushion. The thumb stays out.
    "palm_axes": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "center_z": 0.0,
    "standoff": 0.001,
    "digits": ("index", "middle", "ring", "little"),
    "tilt": math.radians(30.0),
    "tilt_axis": (1.0, 0.0, 0.0),
    "shake": 0.015,
}
"""Grasp pose of the headphones: palm axes in the world frame at the grasp, palm ``z`` of the crown [m],
gap between the palm pads and the crown at the grasp pose [m], the digits that close, tilt angle [rad]
and world axis, and the amplitude of the vertical shake [m]."""

HEADPHONES_GRASP_PHASES = {
    "lead_in": 0.2,
    "approach": 0.8,
    "close": 1.5,
    "settle": 0.5,
    "lift": 1.0,
    "hold": 1.2,
    "shake_up": 0.25,
    "shake_down": 0.35,
    "shake_back": 0.25,
    "shake_hold": 0.9,
    "tilt": 0.8,
    "tilt_hold": 0.6,
    "untilt": 0.8,
    "lower": 1.0,
    "rest": 0.3,
    "open": 0.6,
    "wait": 0.6,
    "retreat": 0.8,
    "end": 0.3,
}
"""Duration of every phase of the headphones grasp [s], in the order they are played."""


def headphones_grasp_plan(
    spec: ch.HandSpec,
    reduced: HeadphonesModel | None = None,
    lift: float = 0.1,
    approach: float = 0.08,
    phases: dict[str, float] | None = None,
) -> ch.GraspPlan:
    """Keyframes of the headphones grasp: approach from above, close, lift, hold, shake, tilt, lower, open, retreat.

    The plan follows ``cable_hand.grasp_plan``. The palm comes down on
    the crown of the headband and stops just short of touching, the
    four fingers close under the crown cushion at :data:`HEADPHONES_HOLD_TENSION`
    and press the band against the palm pads, and the hand lifts the
    headphones out of their cradles. In the lifted hold the hand moves up and
    down once (the ear cups lag and the band opens and closes) and rolls about
    the finger direction and back (one cup hangs lower than the other). Then
    it sets the headphones back into the cradles, opens and retreats. Pose,
    tilt and shake are those of :data:`HEADPHONES_GRASP`.

    Args:
        spec: The hand spec.
        reduced: The reduced model of the headphones; the default if ``None``.
        lift: Height of the lift [m].
        approach: Height of the start pose above the grasp pose [m].
        phases: Durations that replace those of :data:`HEADPHONES_GRASP_PHASES` [s].

    Returns:
        The plan; ``object_xform`` is the start pose of the headphones frame and ``pedestal`` is ``None``.
    """
    if reduced is None:
        reduced = headphones_reduced_model()
    pose = HEADPHONES_GRASP
    hold_tension = HEADPHONES_HOLD_TENSION
    tilt, shake = pose["tilt"], pose["shake"]
    durations = dict(HEADPHONES_GRASP_PHASES)
    durations.update(phases or {})
    object_xform = headphones_rest_xform()
    crown_world = np.array(wp.transform_point(object_xform, wp.vec3(*reduced.crown.tolist())), dtype=float)
    rotation = np.array(pose["palm_axes"], dtype=float).T
    a, b = spec.palm.pads[0], spec.palm.pads[1]
    crown_palm = np.array(
        [
            0.5 * (a.p0[0] + b.p0[0]),
            0.5 * (a.p0[1] + b.p0[1]) + 0.5 * (a.radius + b.radius) + pose["standoff"],
            pose["center_z"],
        ]
    )
    origin = crown_world - rotation @ crown_palm
    up = np.array([0.0, 0.0, lift])
    bump = np.array([0.0, 0.0, shake])

    def palm(position, quat=None):
        pose7 = np.array(ch.frame_to_transform(position, rotation), dtype=float)
        if quat is not None:
            pose7[3:] = quat
        return pose7

    grasp = palm(origin)
    start = palm(origin - approach * rotation[:, 1])
    lifted = palm(origin + up)
    turn = np.array(wp.quat_from_axis_angle(wp.vec3(*pose["tilt_axis"]), float(tilt)), dtype=float)
    pivot = crown_world + up
    tilted = palm(
        pivot + _rotate(turn, origin + up - pivot),
        np.array(wp.quat(*turn.tolist()) * wp.quat(*grasp[3:].tolist()), dtype=float),
    )
    rest = spec.cable.tension_rest
    fingers = dict.fromkeys(pose["digits"], hold_tension)
    changes = {
        "lead_in": {},
        "approach": {"wrist": grasp},
        "close": {"tension": fingers},
        "settle": {},
        "lift": {"wrist": lifted},
        "hold": {},
        "shake_up": {"wrist": palm(origin + up + bump)},
        "shake_down": {"wrist": palm(origin + up - bump)},
        "shake_back": {"wrist": lifted},
        "shake_hold": {},
        "tilt": {"wrist": tilted},
        "tilt_hold": {},
        "untilt": {"wrist": lifted},
        "lower": {"wrist": grasp},
        "rest": {},
        "open": {"tension": rest},
        "wait": {},
        "retreat": {"wrist": start},
        "end": {},
    }
    keyframes = [ch.HandKeyframe(0.0, wrist=start, opposition=0.0, tension=rest)]
    windows = {}
    t = 0.0
    for name, change in changes.items():
        windows[name] = (t, t + durations[name])
        t += durations[name]
        keyframes.append(ch.HandKeyframe(t, **change))
    return ch.GraspPlan(
        object_name="headphones",
        object_xform=object_xform,
        pedestal=None,
        palm_start=wp.transform(*start.tolist()),
        palm_grasp=wp.transform(*grasp.tolist()),
        opposition=0.0,
        hold_tension=float(hold_tension),
        lift=float(lift),
        tilt=float(tilt),
        keyframes=keyframes,
        phases=windows,
    )


def _pose_in_palm(palm: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Pose (position, quaternion ``x, y, z, w``) of a frame in the palm frame."""
    inverse = np.concatenate([-palm[3:6], palm[6:]])
    return np.concatenate(
        [
            _rotate(inverse, pose[:3] - palm[:3]),
            np.array(wp.quat(*inverse.tolist()) * wp.quat(*pose[3:].tolist()), dtype=float),
        ]
    )


def headphones_state(
    model: newton.Model, state: newton.State, hand: ch.HandSkeleton, scene: HeadphonesScene
) -> dict[str, Any]:
    """Read the headphones from a state.

    Args:
        model: The model.
        state: The state to read.
        hand: The hand skeleton.
        scene: The scene.

    Returns:
        ``"object_pose"``: world pose of the headphones frame (position [m], quaternion), shape ``(7,)``;
        ``"object_velocity"`` [m/s] and ``"object_angular_velocity"`` [rad/s], shape ``(3,)``;
        ``"object_in_palm"``: pose of the headphones frame in the palm frame, shape ``(7,)``
        (the reference for ``cable_hand.grasp_slip``);
        ``"mode_q"`` [m] and ``"mode_qd"`` [m/s]: modal coordinates and their rates, shape ``(mode_count,)``;
        ``"cup_centers"`` [m]: world positions of the two deformed cup centers, shape ``(2, 3)``;
        ``"cup_distance"`` [m]: distance between them;
        ``"cup_spread"`` [m]: that distance minus its rest value (positive: the band is opened).
    """
    body_q = state.body_q.numpy().astype(float)
    body_qd = state.body_qd.numpy().astype(float)
    pose = body_q[scene.object_body]
    n = scene.mode_count
    q = state.joint_q.numpy()[scene.q_start + 7 : scene.q_start + 7 + n].astype(float)
    qd = state.joint_qd.numpy()[scene.qd_start + 6 : scene.qd_start + 6 + n].astype(float)
    reduced = scene.reduced
    local = reduced.cup_centers + np.einsum("smc,m->sc", reduced.cup_phi, q)
    cups = pose[:3] + _rotate(pose[3:], local)
    rest = float(np.linalg.norm(reduced.cup_centers[0] - reduced.cup_centers[1]))
    distance = float(np.linalg.norm(local[0] - local[1]))
    return {
        "object_pose": pose,
        "object_velocity": body_qd[scene.object_body, :3],
        "object_angular_velocity": body_qd[scene.object_body, 3:],
        "object_in_palm": _pose_in_palm(body_q[hand.palm], pose),
        "mode_q": q,
        "mode_qd": qd,
        "cup_centers": cups,
        "cup_distance": distance,
        "cup_spread": distance - rest,
    }


def headphones_mesh_vertices(
    model: newton.Model, state: newton.State, scene: HeadphonesScene, names=("band", "cup_left", "cup_right")
) -> dict[str, np.ndarray]:
    """World positions of the vertices of the headphones' meshes for a state, deformed with the band.

    A vertex sits at ``p + R (r + Phi q)``: its rest position in the body frame plus the mode shapes
    times the modal coordinates, moved by the body pose. These are the points on which the headphones
    make contact, and the points the drawn headphones are built on (:class:`HeadphonesLook`).

    Args:
        model: The model.
        state: The state to read.
        scene: The scene.
        names: Mesh names, keys of ``scene.shapes``.

    Returns:
        Per name, the vertices [m] in the order of ``scene.reduced.meshes[name][0]``, shape ``(n, 3)``.
    """
    if not scene.mesh_samples:
        shapes = model.elastic_shape_shape.numpy()
        starts = model.elastic_shape_vertex_start.numpy()
        counts = model.elastic_shape_vertex_count.numpy()
        local = model.elastic_shape_vertex_local.numpy().astype(float)
        phi = model.elastic_shape_vertex_phi.numpy().astype(float).reshape(-1, int(model.elastic_max_mode_count), 3)
        for name, shape in scene.shapes.items():
            entry = int(np.where(shapes == shape)[0][0])
            rows = slice(int(starts[entry]), int(starts[entry]) + int(counts[entry]))
            scene.mesh_samples[name] = (local[rows], phi[rows, : scene.mode_count])
    pose = state.body_q.numpy().astype(float)[scene.object_body]
    q = state.joint_q.numpy()[scene.q_start + 7 : scene.q_start + 7 + scene.mode_count].astype(float)
    out = {}
    for name in names:
        rest, phi = scene.mesh_samples[name]
        out[name] = pose[:3] + _rotate(pose[3:], rest + np.einsum("vmc,m->vc", phi, q))
    return out


def headphones_crown_in_palm(
    model: newton.Model, state: newton.State, hand: ch.HandSkeleton, scene: HeadphonesScene
) -> np.ndarray:
    """Position of the crown point of the headband in the palm frame: the measure of slip in the grip.

    The crown point is the band vertex nearest to the top of the crown, a material point under the palm
    pads. The pose of the body frame is a mean over the whole body: with a soft band it shifts against
    the crown when the ear cups swing, without any slip at the grip, so slip is judged at this point.

    Args:
        model: The model.
        state: The state to read.
        hand: The hand skeleton.
        scene: The scene.

    Returns:
        The point in the palm frame [m], shape ``(3,)``.
    """
    rest = scene.reduced.meshes["band"][0]
    index = int(np.argmin(np.linalg.norm(rest - scene.reduced.crown, axis=1)))
    crown = headphones_mesh_vertices(model, state, scene, names=("band",))["band"][index]
    palm = state.body_q.numpy().astype(float)[hand.palm]
    return _rotate(np.concatenate([-palm[3:6], palm[6:]]), crown - palm[:3])


def _headphones_contact_forces(
    solver, state: newton.State, state_prev: newton.State, contacts, dt: float, scene: HeadphonesScene
) -> dict[str, Any]:
    """Contact forces on the headphones from the last solver step.

    Call it outside a captured graph, after a solver step that used ``contacts``.

    Args:
        solver: The solver.
        state: State after the last solver step.
        state_prev: State the last solver step started from.
        contacts: The contacts the last solver step used.
        dt: Substep length [s].
        scene: The scene.

    Returns:
        ``"pad_force"`` [N]: force of each pad on the headphones in the world
        frame, shape ``(pad_count, 3)`` in the order of ``scene.pads.keys``;
        ``"support_force"`` [N]: force of the table and the cradles, shape ``(3,)``;
        ``"contact_count"``: number of contacts on the headphones.
    """
    # the joint coordinates carry the modal coordinates: the contact points are those of the deformed body
    _, _, _, _, force, count = solver.collect_rigid_contact_forces(
        state.body_q, state_prev.body_q, contacts, dt, joint_q=state.joint_q, joint_q_prev=state_prev.joint_q
    )
    n = int(count.numpy()[0])
    force = force.numpy()[:n].astype(float)
    shape0 = contacts.rigid_contact_shape0.numpy()[:n]
    shape1 = contacts.rigid_contact_shape1.numpy()[:n]
    own = list(scene.shapes.values())
    on1 = np.isin(shape1, own)
    on0 = np.isin(shape0, own)
    on_object = np.where(on1[:, None], force, -force)
    other = np.where(on1, shape0, shape1)
    touching = on0 | on1
    pads = scene.pads
    pad_force = np.zeros((len(pads.keys), 3))
    for i, key in enumerate(pads.keys):
        pad_force[i] = on_object[touching & (other == pads.shapes[key])].sum(axis=0)
    support = touching & np.isin(other, [scene.table_shape, *scene.cradle_shapes])
    return {
        "pad_force": pad_force,
        "support_force": on_object[support].sum(axis=0),
        "contact_count": int(touching.sum()),
    }


# ---------------------------------------------------------------------------
# Headphones, display meshes (what the example draws; numpy only, no part in the simulation)
# ---------------------------------------------------------------------------
# The simulated headphones are a flat strip on an arc with a crown cushion (one collision mesh, a ring of
# vertices per node of the band) and two cylinders for the ear cups. This block draws a finished product on
# exactly those surfaces:
#
# - headband: a steel spine set flush between two wrapped rails on the outside, a wrapped carrier under them
#   and the crown cushion, smooth tubes swept through the deformed rings of the collision mesh, so the drawn
#   band bends with the simulated one and its faces lie on the simulated contact faces (corners rounded by
#   0.3 to 0.8 mm, every simulated vertex within 0.5 mm of the drawn surface);
# - at each band end a hinge block, a steel fork around the upper half of the cup with two pivots, the cup
#   shell with a trim ring and a cap, a seam, and the ear cushion as a separate ring. Shell and cushion keep
#   the radius of the simulated cylinder where the cup lies in its cradle;
# - the stand: a base plate with two V saddles; the rubber pads of the saddles are the simulated cradle faces.
#
# Use:
#     look = HeadphonesLook(band_rest, cup_left_rest, cup_right_rest)      # collision mesh vertices at rest
#     parts = look.parts(band_now, cup_left_now, cup_right_now)             # the same vertices, deformed
#     stand = headphones_stand_parts(cup_centers_world, look.cup_radius)    # static
# Every part is a dict: "name", "vertices" (n, 3) float32 [m], "triangles" (m, 3) int32, "normals" (n, 3)
# float32, "material" (a key of HEADPHONES_MATERIALS). The frame of the output is that of the input vertices.

_MM = 1.0e-3

HEADPHONES_MATERIALS = {
    # tag: base color (linear RGB), roughness, metallic
    "steel": {"color": (0.56, 0.57, 0.58), "roughness": 0.34, "metallic": 1.0},  # spring strip, forks
    "shell": {"color": (0.030, 0.031, 0.034), "roughness": 0.46, "metallic": 0.0},  # cup shells, hinge blocks
    "cap": {"color": (0.055, 0.056, 0.060), "roughness": 0.26, "metallic": 0.0},  # outer face of the cups
    "wrap": {"color": (0.050, 0.050, 0.054), "roughness": 0.62, "metallic": 0.0},  # carrier under the strip
    "fabric": {"color": (0.140, 0.133, 0.127), "roughness": 0.96, "metallic": 0.0},  # crown and ear cushions
    "cloth": {"color": (0.022, 0.022, 0.024), "roughness": 1.0, "metallic": 0.0},  # speaker cover
    "trim": {"color": (0.012, 0.012, 0.013), "roughness": 0.5, "metallic": 0.0},  # seams, collars, pivot pins
    "accent": {"color": (0.60, 0.40, 0.22), "roughness": 0.30, "metallic": 1.0},  # one warm ring on each cup
    "stand": {"color": (0.30, 0.30, 0.31), "roughness": 0.55, "metallic": 0.0},  # base plate and saddles
    "rubber": {"color": (0.020, 0.020, 0.021), "roughness": 0.85, "metallic": 0.0},  # saddle pads
}
"""Look of every material tag of the drawn headphones and their stand: keyword arguments for the viewer."""


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1.0e-12)


def _orient(vertices, triangles, normals):
    """Wind every triangle so that it faces the way its vertex normals point."""
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    face = np.cross(b - a, c - a)
    flip = np.einsum("ij,ij->i", face, normals[triangles].sum(axis=1)) < 0.0
    out = triangles.copy()
    out[flip] = out[flip][:, ::-1]
    return out


def _part(name, material, vertices, triangles, normals):
    return {
        "name": name,
        "material": material,
        "vertices": np.asarray(vertices, dtype=np.float32),
        "triangles": np.asarray(triangles, dtype=np.int32),
        "normals": np.asarray(_unit(normals), dtype=np.float32),
    }


def _merge_arrays(pieces):
    """Join (vertices, triangles, normals) pieces into one set of arrays."""
    vs, ts, ns, start = [], [], [], 0
    for v, t, n in pieces:
        vs.append(v)
        ts.append(t + start)
        ns.append(n)
        start += len(v)
    return np.concatenate(vs), np.concatenate(ts), np.concatenate(ns)


def _grid_triangles(rows, cols, close_cols):
    """Triangles of a grid of rows x cols vertices (row-major); the columns wrap around if close_cols."""
    tris = []
    last = cols if close_cols else cols - 1
    for i in range(rows - 1):
        for j in range(last):
            j2 = (j + 1) % cols
            a, b, c, d = i * cols + j, i * cols + j2, (i + 1) * cols + j2, (i + 1) * cols + j
            tris += [(a, b, c), (a, c, d)]
    return np.array(tris, dtype=np.int32)


def _arc(center, radii, a0, a1, n):
    """Points on an elliptical arc in a profile plane, angles in degrees."""
    a = np.radians(np.linspace(a0, a1, n))
    rr = (radii, radii) if np.isscalar(radii) else radii
    return np.stack([center[0] + rr[0] * np.cos(a), center[1] + rr[1] * np.sin(a)], axis=1)


def _lathe(runs, segments=96):
    """Surface of revolution about the local axis 1 (coordinates: radius, height).

    Each run is a polyline (n, 2) that is smooth in itself; runs meet in creases. A run is walked
    counterclockwise in the (radius, height) plane, so the outward normal is (d height, -d radius).
    Returns pieces of vertices, triangles and normals with the axis along local y.
    """
    ang = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    cos, sin = np.cos(ang), np.sin(ang)
    pieces = []
    for polyline in runs:
        run = np.asarray(polyline, dtype=float)
        d = np.gradient(run, axis=0)
        n2 = _unit(np.stack([d[:, 1], -d[:, 0]], axis=1))
        height = np.repeat(run[:, 1:2], segments, axis=1)
        v = np.stack([run[:, 0:1] * cos, height, run[:, 0:1] * sin], axis=2).reshape(-1, 3)
        up = np.repeat(n2[:, 1:2], segments, axis=1)
        n = np.stack([n2[:, 0:1] * cos, up, n2[:, 0:1] * sin], axis=2).reshape(-1, 3)
        t = _grid_triangles(len(run), segments, True)
        # drop the degenerate triangles on the axis
        area = np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
        t = t[area > 1.0e-14]
        pieces.append((v, _orient(v, t, n), n))
    return pieces


def _section(a0, a1, b0, b1, r_low, r_high, seg=5, flat=0):
    """Rounded rectangles [a0, a1] x [b0, b1] for a sweep, one per ring.

    ``b0``, ``b1`` and the corner radii of the low side (``b0``) and the high side (``b1``) may be arrays
    over the rings. ``flat`` extra points are set on every flat side. Returns points and normals
    (rings, n, 2).
    """
    b0, b1, r_low, r_high = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (b0, b1, r_low, r_high))
    m = max(len(b0), len(b1), len(r_low), len(r_high))
    b0, b1, r_low, r_high = (np.broadcast_to(v, (m,)) for v in (b0, b1, r_low, r_high))
    arcs, normals = [], []
    for corner, (sa, sb) in enumerate(((1, -1), (1, 1), (-1, 1), (-1, -1))):  # counterclockwise from low right
        r = r_low if sb < 0 else r_high
        ca = (a1 if sa > 0 else a0) - sa * r
        cb = (b1 if sb > 0 else b0) - sb * r
        ang = np.radians(np.linspace(-90.0 + 90.0 * corner, 90.0 * corner, seg + 1))
        arcs.append(np.stack([ca[:, None] + r[:, None] * np.cos(ang), cb[:, None] + r[:, None] * np.sin(ang)], axis=2))
        normals.append(np.broadcast_to(np.stack([np.cos(ang), np.sin(ang)], axis=1), (m, seg + 1, 2)))
    pts, nrm = [], []
    w = (np.arange(1, flat + 1) / (flat + 1.0))[None, :, None]
    for corner in range(4):
        pts.append(arcs[corner])
        nrm.append(normals[corner])
        if flat:
            start, end = arcs[corner][:, -1:, :], arcs[(corner + 1) % 4][:, :1, :]
            pts.append(start + w * (end - start))
            nrm.append(np.broadcast_to(normals[corner][:, -1:, :], (m, flat, 2)))
    return np.concatenate(pts, axis=1), np.concatenate(nrm, axis=1)


def _sweep(origin, ex, er, pts, nrm, scale=None):
    """Tube through rings: ring k is origin[k] + a * ex[k] + b * er[k] for the section points (a, b)."""
    m, n = len(origin), pts.shape[1]
    pts = np.broadcast_to(pts, (m, n, 2))
    nrm = np.broadcast_to(nrm, (m, n, 2))
    if scale is not None:
        pts = pts * np.asarray(scale)[:, None, None]
    exu, eru = _unit(ex), _unit(er)
    v = origin[:, None, :] + pts[..., 0:1] * ex[:, None, :] + pts[..., 1:2] * er[:, None, :]
    nv = nrm[..., 0:1] * exu[:, None, :] + nrm[..., 1:2] * eru[:, None, :]
    v, nv = v.reshape(-1, 3), nv.reshape(-1, 3)
    tris = [_grid_triangles(m, n, True)]
    vs, ns = [v], [nv]
    # flat end caps
    for ring, tangent in ((0, _unit(origin[0] - origin[1])), (m - 1, _unit(origin[-1] - origin[-2]))):
        start = sum(len(x) for x in vs)
        rim = v[ring * n : (ring + 1) * n]
        vs.append(np.concatenate([rim, rim.mean(axis=0, keepdims=True)]))
        ns.append(np.tile(tangent, (n + 1, 1)))
        tris.append(np.array([(start + n, start + j, start + (j + 1) % n) for j in range(n)], dtype=np.int32))
    v, nv, t = np.concatenate(vs), np.concatenate(ns), np.concatenate(tris)
    return v, t, nv


def _round_box(center, half, radius, n=8):
    """Box with rounded edges in local coordinates: vertices, triangles, normals."""
    center, half = np.asarray(center, dtype=float), np.asarray(half, dtype=float)
    core = half - radius
    # grid lines per axis: n steps across each rounding (even in angle), one cell across the flat
    step = radius * np.tan(np.radians(np.linspace(0.0, 45.0, n + 1)))
    lines = [np.concatenate([-(core[i] + step[::-1]), core[i] + step]) for i in range(3)]
    vs, ns, ts, start = [], [], [], 0
    for axis in range(3):
        a1, a2 = (axis + 1) % 3, (axis + 2) % 3
        for sign in (-1.0, 1.0):
            q = np.zeros((len(lines[a1]), len(lines[a2]), 3))
            q[..., axis] = sign * half[axis]
            q[..., a1] = lines[a1][:, None]
            q[..., a2] = lines[a2][None, :]
            inner = np.clip(q, -core, core)
            normal = _unit(q - inner)
            vs.append((center + inner + radius * normal).reshape(-1, 3))
            ns.append(normal.reshape(-1, 3))
            ts.append(_grid_triangles(len(lines[a1]), len(lines[a2]), False) + start)
            start += len(lines[a1]) * len(lines[a2])
    v, nv, t = np.concatenate(vs), np.concatenate(ns), np.concatenate(ts)
    return v, _orient(v, t, nv), nv


def _extrude(poly, y0, y1, inside):
    """Prism of a polygon in the x-z plane between y0 and y1 with flat faces.

    ``inside`` is a point that sees every corner (the caps are fans around it).
    """
    poly = np.asarray(poly, dtype=float)
    n = len(poly)
    vs, ns, ts, start = [], [], [], 0
    middle = np.array([inside[0], 0.5 * (y0 + y1), inside[1]])
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        quad = np.array([[p[0], y0, p[1]], [q[0], y0, q[1]], [q[0], y1, q[1]], [p[0], y1, p[1]]])
        normal = np.cross(quad[1] - quad[0], quad[3] - quad[0])
        if np.dot(normal, quad.mean(axis=0) - middle) < 0.0:
            normal = -normal
        vs.append(quad)
        ns.append(np.tile(normal, (4, 1)))
        ts.append(np.array([(0, 1, 2), (0, 2, 3)], dtype=np.int32) + start)
        start += 4
    for y, sign in ((y0, -1.0), (y1, 1.0)):
        cap = np.concatenate([np.stack([poly[:, 0], np.full(n, y), poly[:, 1]], axis=1), [[inside[0], y, inside[1]]]])
        vs.append(cap)
        ns.append(np.tile([0.0, sign, 0.0], (n + 1, 1)))
        ts.append(np.array([(n, i, (i + 1) % n) for i in range(n)], dtype=np.int32) + start)
        start += n + 1
    v, nv, t = np.concatenate(vs), np.concatenate(ns), np.concatenate(ts)
    return v, _orient(v, t, nv), nv


def _upsample(p, sub):
    """Cubic interpolation (Catmull-Rom) of the rows of p, ``sub`` pieces per interval."""
    p = np.asarray(p, dtype=float)
    flat = p.reshape(len(p), -1)
    m = np.empty_like(flat)
    m[1:-1] = 0.5 * (flat[2:] - flat[:-2])
    m[0], m[-1] = flat[1] - flat[0], flat[-1] - flat[-2]
    u = (np.arange(sub) / sub)[None, :, None]
    h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
    h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
    body = h00 * flat[:-1, None] + h10 * m[:-1, None] + h01 * flat[1:, None] + h11 * m[1:, None]
    out = np.concatenate([body.reshape(-1, flat.shape[1]), flat[-1:]])
    return out.reshape((-1, *p.shape[1:]))


class HeadphonesLook:
    """Display parts of the headphones on the collision meshes of the simulated ones.

    Args:
        band_rest: Vertices of the band collision mesh at rest [m], shape ``(nodes * ring + 2, 3)``: per
            node a ring of ``2 * width_points + 2 * side_points`` vertices (outer face, +x side, inner
            face, -x side).
        cup_left_rest: Vertices of the left cup cylinder at rest [m], shape ``(3 * cup_segments + 2, 3)``:
            three rings along the axis, then the two face centers.
        cup_right_rest: The same for the right cup.
        width_points: Mesh resolution of :func:`headphones_reduced_model`.
        side_points: Mesh resolution of :func:`headphones_reduced_model`.
        cup_segments: Mesh resolution of :func:`headphones_reduced_model`.
        subdivide: Display rings per interval of band nodes.
        segments: Segments around the turned parts of a cup.
    """

    SPRING = 1.2 * _MM  # thickness of the steel strip, the outer layer of the band
    SPINE = 7.5 * _MM  # half width of the visible steel spine

    def __init__(
        self,
        band_rest,
        cup_left_rest,
        cup_right_rest,
        width_points=7,
        side_points=5,
        cup_segments=40,
        subdivide=4,
        segments=128,
    ):
        self.ring = 2 * width_points + 2 * side_points
        self.nodes = (len(band_rest) - 2) // self.ring
        self.i_out_mid, self.i_out_low, self.i_out_high = width_points // 2, 0, width_points - 1
        self.i_in_mid = width_points + side_points + width_points // 2
        self.seg = cup_segments
        self.sub = subdivide
        self.segments = segments
        rings = np.asarray(band_rest, dtype=float)[: self.nodes * self.ring].reshape(self.nodes, self.ring, 3)
        self.width = float(np.linalg.norm(rings[:, self.i_out_high] - rings[:, self.i_out_low], axis=1).mean())
        self.total = np.linalg.norm(rings[:, self.i_out_mid] - rings[:, self.i_in_mid], axis=1)  # strip and cushion
        self.thickness = float(self.total.min())
        self.depth = self.total - self.thickness  # crown cushion below the strip, per node
        cup = np.asarray(cup_left_rest, dtype=float)
        self.cup_radius = 0.5 * float(np.linalg.norm(cup[self.seg] - cup[self.seg + self.seg // 2]))
        self.cup_depth = float(np.linalg.norm(cup[3 * self.seg + 1] - cup[3 * self.seg]))
        # which way is "out" (away from the head) along the cup axis: the left cup sits at +y of the band
        mid = rings[:, self.i_out_mid].mean(axis=0)
        self.out_sign = []
        self.band_end_local = []
        for vertices, node in ((cup_left_rest, 0), (cup_right_rest, self.nodes - 1)):
            o, x, y, z = self._cup_frame(np.asarray(vertices, dtype=float))
            s = 1.0 if np.dot(o - mid, y) > 0.0 else -1.0
            self.out_sign.append(s)
            end = self._band_frames(rings)[0][node]
            self.band_end_local.append(np.linalg.solve(np.stack([x, s * y, z], axis=1), end - o))
        self._cup_local = self._build_cup_parts(np.mean(self.band_end_local, axis=0))
        self._band_triangles = {}

    def _cup_frame(self, cup):
        """Center and axes of a cup from its cylinder vertices (the axes carry the cup's size)."""
        s = self.seg
        o = cup[s : 2 * s].mean(axis=0)
        x = (cup[s] - cup[s + s // 2]) / (2.0 * self.cup_radius)
        z = (cup[s + s // 4] - cup[s + 3 * s // 4]) / (2.0 * self.cup_radius)
        y = (cup[3 * s + 1] - cup[3 * s]) / self.cup_depth
        return o, x, y, z

    def _band_frames(self, rings):
        """Center line point, width axis and radial axis per node, from the ring vertices.

        The axes are not normalized: they carry the small stretch of the linear mode shapes, so the drawn
        faces stay on the simulated ones.
        """
        er = (rings[:, self.i_out_mid] - rings[:, self.i_in_mid]) / self.total[:, None]
        ex = (rings[:, self.i_out_high] - rings[:, self.i_out_low]) / self.width
        origin = rings[:, self.i_out_mid] - 0.5 * self.thickness * er
        return origin, ex, er

    def _build_cup_parts(self, band_end):
        """Parts that ride on a band end, in the cup frame, built once.

        Cup frame: x along the band width, y out of the head, z toward the band end. Returns a list of
        (name, material, vertices, triangles, normals).
        """
        radius, h = self.cup_radius, 0.5 * self.cup_depth
        parts = []

        def lathe(name, material, runs):
            parts.append((name, material, *_merge_arrays(_lathe(runs, self.segments))))

        seam0, seam1 = -3.6 * _MM, -4.4 * _MM  # the seam between shell and ear cushion (a recessed ring)
        face = 30.8 * _MM  # radius where the outer face steps down to the trim ring
        # shell: a short chamfer at the seam, the cylindrical land, a 2.2 mm fillet, the flat rim of the outer
        # face (the fillet is small enough that a cup tilted in its cradle stays within 0.5 mm of the
        # simulated sharp rim)
        fillet = 2.2 * _MM
        shell = np.concatenate(
            [
                [(radius, seam0 + 0.6 * _MM)],
                [(radius, h - fillet)],
                _arc((radius - fillet, h - fillet), fillet, 0.0, 90.0, 14)[1:],
                [(face, h)],
            ]
        )
        shell = np.concatenate([np.linspace(shell[0], shell[1], 6), shell[2:]])
        chamfer = np.array([(radius - 0.6 * _MM, seam0), (radius, seam0 + 0.6 * _MM)])
        lathe("shell", "shell", [chamfer, shell, np.array([(face, h), (face, h - 1.0 * _MM)])])
        # trim ring: a slim warm metal ring with a round top, set just below the rim
        ring_mid, ring_half = face - 0.85 * _MM, 0.85 * _MM
        lathe("trim_ring", "accent", [_arc((ring_mid, h - 1.0 * _MM), (ring_half, 0.8 * _MM), 0.0, 180.0, 13)])
        # cap: a shallow glossy dome inside the trim ring
        r_cap = ring_mid - ring_half
        rr = np.linspace(r_cap, 0.0, 28)
        lathe("cap", "cap", [np.stack([rr, h - 1.0 * _MM + 0.85 * _MM * (1.0 - (rr / r_cap) ** 2)], axis=1)])
        # seam: a dark ring recessed by 0.6 mm between shell and cushion
        lathe("seam", "trim", [np.array([(radius - 0.6 * _MM, seam1), (radius - 0.6 * _MM, seam0)])])
        # ear cushion: a pillow ring; its outer wall keeps the cup radius, the face is the inner face of the cup
        hole, floor = 19.5 * _MM, -6.0 * _MM
        cushion = np.concatenate(
            [
                np.linspace((hole, floor), (hole, -h + 3.5 * _MM), 5),
                _arc((hole + 3.5 * _MM, -h + 3.5 * _MM), 3.5 * _MM, 180.0, 270.0, 12)[1:],
                # outer shoulder: small enough that a cup tilted by 11 deg in its cradle (band still bent after
                # the set-down) shows less than 0.5 mm between the drawn shoulder and the simulated sharp rim
                _arc((radius - 3.0 * _MM, -h + 2.6 * _MM), (3.0 * _MM, 2.6 * _MM), 270.0, 360.0, 14),
                np.linspace((radius, -h + 2.6 * _MM), (radius, seam1 - 0.8 * _MM), 5)[1:],
                _arc((radius - 0.8 * _MM, seam1 - 0.8 * _MM), 0.8 * _MM, 0.0, 90.0, 6)[1:],
            ]
        )
        lathe("ear_cushion", "fabric", [cushion])
        lathe("speaker_cover", "cloth", [np.array([(0.0, floor), (0.5 * hole, floor), (hole, floor)])])

        def to_xyz(pieces, axis):
            """Turn lathe output (axis along local y) so that its axis points along +x or -x."""
            out = []
            for v, t, n in pieces:
                v2 = np.stack([axis * v[:, 1], v[:, 0], axis * v[:, 2]], axis=1)
                n2 = np.stack([axis * n[:, 1], n[:, 0], axis * n[:, 2]], axis=1)
                out.append((v2, _orient(v2, t, n2), n2))
            return out

        # fork: an arc of flat steel around the upper half of the cup, clear of it by 1.4 mm, with rounded tips
        y_fork, gap, bar = 3.0 * _MM, 1.4 * _MM, 3.2 * _MM
        r_mid = radius + gap + 0.5 * bar
        theta = np.radians(np.linspace(-7.0, 187.0, 195))
        er = np.stack([np.cos(theta), np.zeros_like(theta), np.sin(theta)], axis=1)
        origin = r_mid * er + np.array([0.0, y_fork, 0.0])
        ex = np.tile([0.0, 1.0, 0.0], (len(theta), 1))
        arc_len = r_mid * (theta - theta[0])
        from_end = np.minimum(arc_len, arc_len[-1] - arc_len)
        nose = 3.0 * _MM
        scale = np.sqrt(np.clip(1.0 - (np.clip(nose - from_end, 0.0, nose) / nose) ** 2, 0.02, 1.0))
        pts, nrm = _section(-3.25 * _MM, 3.25 * _MM, -0.5 * bar, 0.5 * bar, 0.9 * _MM, 0.9 * _MM)
        v, t, n = _sweep(origin, ex, er, pts, nrm, scale=scale)
        parts.append(("fork", "steel", v, _orient(v, t, n), n))
        # pivots: a dark pin from the fork into the shell and a small cap on the outside of the fork, both sides
        pins, caps = [], []
        top = r_mid + 0.5 * bar
        for axis in (1.0, -1.0):
            pin = _lathe([np.array([(2.1 * _MM, radius - 0.5 * _MM), (2.1 * _MM, r_mid)])], 32)
            cap_run = np.concatenate(
                [
                    [(2.7 * _MM, top - 0.1 * _MM)],
                    [(2.7 * _MM, top + 0.3 * _MM)],
                    _arc((2.2 * _MM, top + 0.3 * _MM), 0.5 * _MM, 0.0, 90.0, 6)[1:],
                    [(0.0, top + 0.8 * _MM)],
                ]
            )
            cap = _lathe([cap_run], 32)
            for group, pieces in ((pins, pin), (caps, cap)):
                for v, t, n in to_xyz(pieces, axis):
                    group.append((v + np.array([0.0, y_fork, 0.0]), t, n))
        parts.append(("pivot_pins", "trim", *_merge_arrays(pins)))
        parts.append(("pivot_caps", "steel", *_merge_arrays(caps)))
        # hinge block: where the band end meets the crown of the fork; a dark collar where the strip enters it
        z_end = float(band_end[2])
        lean = float(band_end[1])
        low, high = radius + 0.6 * _MM, z_end + 7.4 * _MM
        block = (0.5 * self.width + 2.6 * _MM, 6.2 * _MM, 0.5 * (high - low))
        v, t, n = _round_box((0.0, 0.4 * _MM + lean, 0.5 * (low + high)), block, 3.2 * _MM)
        parts.append(("hinge_block", "shell", v, t, n))
        slope = math.tan(math.radians(9.0))  # the band leans toward the head as it rises
        collar = (0.5 * self.width + 1.2 * _MM, 3.5 * _MM, 0.9 * _MM)
        v, t, n = _round_box((0.0, lean - (high - z_end) * slope, high + 0.1 * _MM), collar, 0.8 * _MM)
        parts.append(("band_collar", "trim", v, t, n))
        return parts

    def band_frames(self, band):
        """Display rings of the band: center line, width axis, radial axis and cushion depth, upsampled."""
        rings = np.asarray(band, dtype=float)[: self.nodes * self.ring].reshape(self.nodes, self.ring, 3)
        origin, ex, er = self._band_frames(rings)
        depth = np.clip(_upsample(self.depth, self.sub), 0.0, None)
        return _upsample(origin, self.sub), _upsample(ex, self.sub), _upsample(er, self.sub), depth

    def parts(self, band, cup_left, cup_right):
        """All display parts for the given (deformed) collision mesh vertices.

        Args:
            band: Vertices of the band collision mesh [m], in the order of the rest mesh.
            cup_left: Vertices of the left cup cylinder [m].
            cup_right: Vertices of the right cup cylinder [m].

        Returns:
            The parts (dicts, see the block comment above); the triangles of a part never change.
        """
        origin, ex, er, depth = self.band_frames(band)
        half, t = 0.5 * self.width, self.thickness
        out = []
        # outer face: a steel spine set flush between two wrapped rails (grooves of 0.3 mm); below them the
        # carrier. All of them end on the simulated faces: outer face at +t/2, inner face at -t/2, sides at
        # half the width.
        spine, groove, top, low = self.SPINE, 0.3 * _MM, 0.5 * t, 0.5 * t - self.SPRING
        tubes = (
            ("band_spring", "steel", (-spine, spine, low - 0.2 * _MM, top, 0.3 * _MM, 0.3 * _MM)),
            ("band_rail_a", "wrap", (-half, -spine - groove, low - 0.2 * _MM, top, 0.3 * _MM, 0.45 * _MM)),
            ("band_rail_b", "wrap", (spine + groove, half, low - 0.2 * _MM, top, 0.3 * _MM, 0.45 * _MM)),
            ("band_carrier", "wrap", (-half, half, -0.5 * t, low, 0.5 * _MM, 0.2 * _MM)),
        )
        for name, material, (a0, a1, b0, b1, r0, r1) in tubes:
            pts, nrm = _section(a0, a1, b0, b1, r0, r1, flat=5)
            v, tri, n = _sweep(origin, ex, er, pts, nrm)
            out.append(self._band_part(name, material, v, tri, n))
        # crown cushion: where it has depth; 0.2 mm narrower than the band on each side, bottom corners 0.8 mm
        # (the sharp bottom corner of the simulated section is then 0.48 mm from the drawn one)
        rows = np.where(depth > 0.25 * _MM)[0]
        if len(rows) > 1:
            rows = slice(int(rows[0]), int(rows[-1]) + 1)
            d = depth[rows]
            low_radius, high_radius = np.minimum(0.8 * _MM, 0.45 * d), np.minimum(0.4 * _MM, 0.45 * d)
            pts, nrm = _section(
                -half + 0.2 * _MM, half - 0.2 * _MM, -0.5 * t - d, -0.5 * t + 0.0 * d, low_radius, high_radius, flat=9
            )
            v, tri, n = _sweep(origin[rows], ex[rows], er[rows], pts, nrm)
            out.append(self._band_part("band_cushion", "fabric", v, tri, n))
        for side, cup in enumerate((cup_left, cup_right)):
            o, x, y, z = self._cup_frame(np.asarray(cup, dtype=float))
            s = self.out_sign[side]
            axes = np.stack([x, s * y, z], axis=0)
            unit = _unit(axes)
            tag = ("left", "right")[side]
            for name, material, v, tri, n in self._cup_local:
                out.append(_part(f"{name}_{tag}", material, o + v @ axes, tri if s > 0 else tri[:, ::-1], n @ unit))
        return out

    def _band_part(self, name, material, v, tri, n):
        key = (name, len(v))
        if key not in self._band_triangles:
            self._band_triangles[key] = _orient(v, tri, n)
        return _part(name, material, v, self._band_triangles[key], n)


def headphones_stand_parts(
    cup_centers,
    cup_radius,
    table_z=0.0,
    cradle_angle=HEADPHONES_CRADLE_ANGLE,
    face_half=(0.010, 0.016),
    post_clearance=0.004,
    plate=0.005,
    pad=0.0016,
):
    """A made stand for the headphones: a base plate and two V saddles with rubber pads.

    The top faces of the four pads are the cradle faces of the simulation (planes tangent to an ear cup at
    ``cradle_angle`` from the vertical, ``face_half`` wide), so a cup that rests in the simulated cradle
    rests on the drawn pads. Base plate and saddle bars are display only.

    Args:
        cup_centers: World positions of the two ear cup centers at rest [m], shape ``(2, 3)``; the band lies
            along world ``y`` and ``z`` is up.
        cup_radius: Radius of the ear cups [m].
        table_z: Height of the table top [m].
        cradle_angle: Angle of the cradle faces from the vertical [rad].
        face_half: Half length of a cradle face along its slope and half width along ``y`` [m].
        post_clearance: Clearance of the saddle valley below the cup [m] (the simulated post).
        plate: Thickness of the base plate [m].
        pad: Thickness of the rubber pads [m].

    Returns:
        The parts, in world coordinates.
    """
    c = np.asarray(cup_centers, dtype=float)
    sa, ca = math.sin(cradle_angle), math.cos(cradle_angle)
    out = []
    # base plate: a rounded rectangle with a small fillet on the top edge
    y_half = 0.5 * abs(c[0, 1] - c[1, 1]) + face_half[1] + 0.012
    x_half = cup_radius * sa + face_half[0] * ca + 0.012
    corner = 0.014
    cx, cy = float(c[:, 0].mean()), float(c[:, 1].mean())
    outline, normal = _section(-x_half, x_half, -y_half, y_half, corner, corner, seg=14)
    outline, normal = outline[0], normal[0]
    fillet = 0.0015
    angles = np.radians(np.linspace(0.0, 90.0, 7))[1:]
    profile = [(0.0, 0.0), (0.0, plate - fillet)]
    profile += [(fillet * (1.0 - math.cos(a)), plate - fillet + fillet * math.sin(a)) for a in angles]
    profile_n = [(1.0, 0.0), (1.0, 0.0)] + [(math.cos(a), math.sin(a)) for a in angles]
    vs, ns = [], []
    for (inset, z), (nh, nz) in zip(profile, profile_n, strict=True):
        p = outline - inset * normal
        vs.append(np.stack([cx + p[:, 0], cy + p[:, 1], np.full(len(p), table_z + z)], axis=1))
        ns.append(np.stack([nh * normal[:, 0], nh * normal[:, 1], np.full(len(p), nz)], axis=1))
    m = len(outline)
    tris = [_grid_triangles(len(profile), m, True)]
    start = len(profile) * m
    top = np.concatenate([vs[-1], [[cx, cy, table_z + plate]]])
    tris.append(np.array([(start + m, start + j, start + (j + 1) % m) for j in range(m)], dtype=np.int32))
    v = np.concatenate([*vs, top])
    n = np.concatenate([*ns, np.tile([0.0, 0.0, 1.0], (m + 1, 1))])
    t = np.concatenate(tris)
    out.append(_part("stand_base", "stand", v, _orient(v, t, n), n))
    for side in range(2):
        o = c[side]
        valley = o[2] - cup_radius - post_clearance
        tag = ("left", "right")[side]
        pads, prisms = [], []
        y0, y1 = o[1] - face_half[1], o[1] + face_half[1]
        stem, sheet = 0.006, 0.004  # half width of the post (the simulated one), thickness of the two arms
        for sign in (1.0, -1.0):
            normal = np.array([-sign * sa, ca])  # of the cradle face, in (x, z)
            tangent = np.array([sign * ca, sa])  # up the slope, away from the middle
            touch = np.array([o[0], o[2]]) - cup_radius * normal
            seat = touch - pad * normal  # the arm surface under the pad
            u_low = (o[0] + sign * 0.5 * stem - seat[0]) / tangent[0]  # the arm starts inside the post
            p1 = seat + u_low * tangent
            p2 = seat + (face_half[0] + 0.0015) * tangent
            arm = np.array([p1, p2, p2 - sheet * normal, p1 - sheet * normal])
            prisms.append(_extrude(arm, y0, y1, arm.mean(axis=0)))
            u_valley = (valley - seat[1]) / tangent[1]
            flare = abs((seat + u_valley * tangent)[0] - o[0])
            # rubber pad: its top face is the simulated cradle face
            v, t, n = _round_box((0.0, 0.0, -0.5 * pad), (face_half[0], face_half[1], 0.5 * pad), 0.0004, n=4)
            # columns: slope, y, normal
            rot = np.array([[tangent[0], 0.0, normal[0]], [0.0, 1.0, 0.0], [tangent[1], 0.0, normal[1]]])
            pads.append((np.array([touch[0], o[1], touch[1]]) + v @ rot.T, t, n @ rot.T))
        # post: the simulated one (12 mm wide, its top 4 mm below the cup), flared at the top into the two arms
        base = table_z + plate - 0.0003
        post = np.array(
            [
                (o[0] - flare, valley),
                (o[0] + flare, valley),
                (o[0] + stem, valley - 0.005),
                (o[0] + stem, base),
                (o[0] - stem, base),
                (o[0] - stem, valley - 0.005),
            ]
        )
        prisms.append(_extrude(post, y0, y1, (o[0], 0.5 * (valley + base))))
        v, t, n = _merge_arrays(prisms)
        out.append(_part(f"stand_saddle_{tag}", "stand", v, t, n))
        pv, pt, pn = _merge_arrays([(v, _orient(v, t, n), n) for v, t, n in pads])
        out.append(_part(f"stand_pads_{tag}", "rubber", pv, pt, pn))
    return out


# ---------------------------------------------------------------------------
# Soft objects (procedural tetrahedral meshes, material, scene, read-back, grasp plan)
# ---------------------------------------------------------------------------

SOFT_BALL_RADIUS = 0.028
"""Radius of the soft object, a plain foam ball of the kind a hand squeezes [m] (its contact surface,
which is also what is drawn)."""

SOFT_BALL_CELLS = 8
"""Number of mesh cells across the soft ball (even); each cell is six tetrahedra."""

SOFT_SKIN_FACTOR = 4.0
"""Factor on the moduli of the surface tetrahedra of a soft object: a denser skin over a softer core.

A fingertip pad is as large as one mesh cell. With one material throughout, the surface elements under
a pad collapse and invert before the object as a whole is squeezed; with the stiffer skin the load
spreads."""

SOFT_PARTICLE_RADIUS = 0.003
"""Contact radius of the particles of a soft object [m]. Contact acts on spheres of this radius around
the particles, so the surface of the object (its stated size, and what is drawn, see
:func:`create_soft_display_surface`) lies this far outside the mesh of particle centers."""

SOFT_DENSITY = 300.0
"""Density of the soft objects [kg/m^3]: a self-skinned foam ball. The mass is carried by the mesh of
particle centers (radius 25 mm), 19 g: 210 kg/m^3 over the volume of the 56 mm ball."""

SOFT_YOUNG = 3.0e4
"""Young's modulus of the core of the soft objects [Pa], a soft foam; the skin is
:data:`SOFT_SKIN_FACTOR` times stiffer. Softer cores (10 and 20 kPa) slip out of the squeezing hand."""

SOFT_POISSON = 0.40
"""Poisson's ratio of the soft objects (below the 0.5 of rubber: linear tetrahedra lock near 0.5)."""

SOFT_DAMPING = 1.0e-2
"""Viscous damping of the tetrahedra (``k_damp`` of the builder): a foam that does not wobble."""

SOFT_CONTACT_KE = 2.0e3
"""Particle side of the particle-shape contact stiffness [N/m]; the solver averages it with the shape's."""

SOFT_CONTACT_KD = 1.0e-4
"""Particle side of the particle-shape contact damping [N*s/m]; the solver averages it with the shape's."""

SOFT_PAD_CONTACT_KE = 2.0e3
"""Pad side of the pad-particle contact stiffness [N/m] in a scene with a soft object.

The solver uses the mean of the two sides. With the pad stiffness of the rigid grasp scenes (1e5 N/m)
that mean is 51 kN/m, a thousand times what one particle of the foam (26 mg, a few hundred N/m of
material stiffness) can balance within a solver step: the particles under a pad are then slaved to the
pad, the iterations between the links and the particles do not settle, and the fingers chatter on the
ball by several degrees from frame to frame. With both sides at 2 kN/m a pad that carries 2 N on four
particles sinks in by a quarter of a millimeter and the grasp is still."""

SOFT_SEAT_CONTACT_KE = 2.0e3
"""Contact stiffness of the table, the post and the seat balls under a soft object [N/m], for the same
reason as :data:`SOFT_PAD_CONTACT_KE`: on a seat of 1e5 N/m the resting ball rattles by a tenth of a
millimeter for seconds."""

SOFT_SEAT_SETTLE_SHIFT = (0.0, -0.000764, -0.000878)
"""Where the soft ball rests in its seat, against the pose at which the seat balls are tangent to its
nominal surface [m] (measured). The contact surface is made of the spheres around the particles, so the
seat balls sink in between them and the ball rolls a little until it sits on four particles. The ball is
built at this pose so that it does not drop and roll at the start. The value was measured for the default
ball (radius, mesh and material) in the default seat and holds only for those: another ball or seat
settles from this pose to its own."""

SOFT_SEAT_SETTLE_TURN = 0.03756
"""Rotation about the world x axis that goes with :data:`SOFT_SEAT_SETTLE_SHIFT` [rad] (measured for the
default ball and seat, and valid only for those)."""

SOFT_PAD_CONTACT_KD = 2.0
"""Pad side of the pad-particle contact damping [N*s/m] in a scene with a soft object (the shape default
of 100 N*s/m gives a mean of 50 N*s/m, as stiff per step as the contact itself)."""

SOFT_CONTACT_GAP = 0.004
"""Detection distance of particle-shape contacts beyond the particle radius [m]."""

SOFT_FRICTION_EPSILON = 1.0e-3
"""Slip speed below which the solver's friction is regularized [m/s] (``friction_epsilon``). At the
solver default of 1e-2 a seated soft object creeps on its post at a few millimeters per second."""

SOFT_SET_DOWN_SHIFT = (0.0126, 0.005)
"""World ``(x, y)`` shift of the palm at the set-down of a soft object against the pick-up pose [m].
The closing hand moves the ball about 13 mm toward the fingers and 5 mm toward the little finger, and the
ball is still there in the hand at the set-down; the shift brings it down over the middle of its seat
(measured: within 0.1 mm), so that it does not roll back and rock when the palm lets go."""

SOFT_GRASP_PHASES = {"squeeze": 1.0, "squeeze_hold": 1.0, "relax": 1.0, "recover": 0.6, "ease": 0.6}
"""Duration of the phases a soft grasp adds to ``cable_hand.GRASP_PHASES`` [s]: one squeeze pulse in the
lifted hold (``"squeeze"``, ``"squeeze_hold"``, ``"relax"``), ``"recover"`` after the last pulse, and
``"ease"`` before the retreat (the palm rises by :data:`SOFT_EASE_HEIGHT` off the released object)."""

SOFT_TAKE_PHASES = {"close": 2.5, "oppose": 1.4}
"""Durations of ``cable_hand.GRASP_PHASES`` that a soft grasp replaces [s]: the fingers close on the ball
in 2.5 s (1.0 s for a rigid object) and the thumb turns onto it in 1.4 s (0.7 s). At the faster pace a
fingertip that meets the light ball is thrown back by several degrees, and the thumb base strikes the
ball and the whole grasp rings at about 7 Hz for half a second."""

SOFT_EASE_HEIGHT = 0.006
"""Height by which the palm rises slowly off a released soft object before the hand retreats [m]."""

SOFT_END_REST = 1.0
"""Duration of the last phase of a soft grasp [s]: the released object comes to rest while the hand is away."""

SOFT_HOLD_TENSION = 25.0
"""Cable tension of the four fingers in the hold of a soft object [N]. Below about 20 N the fingers do
not reach under the ball before the thumb comes, and the thumb pushes it off its seat."""

SOFT_SQUEEZE_TENSION = 45.0
"""Cable tension of the four fingers at the top of a squeeze pulse [N]. The tensions follow the command
within a few hundredths of a newton, so the highest tension of the run is this value (limit 60 N)."""

SOFT_THUMB_TENSION = 12.0
"""Cable tension of the thumb while it holds a soft object [N], as for the rigid ball."""

SOFT_THUMB_SQUEEZE_TENSION = 30.0
"""Cable tension of the thumb at the top of a squeeze pulse [N]. With 45 N on the fingers and 30 N on the
thumb the gap between the thumb pad and the fingertip pads closes by 29 %."""

SOFT_SET_DOWN_CLEARANCE = -0.008
"""Height above the pick-up height at which the hand stops when it lowers a soft object into its seat [m].
The closing hand draws the ball 7 mm up into the palm, so the hand goes 8 mm lower than it picked the
ball up: the ball then sits in its seat, pressed on lightly, before the hand lets go. (With 4.5 mm it is
released 3 mm above the seat and drops into it.)"""


def lame_parameters(young: float, poisson: float) -> tuple[float, float]:
    """Lame parameters of an isotropic material.

    Args:
        young: Young's modulus [Pa].
        poisson: Poisson's ratio.

    Returns:
        ``k_mu`` and ``k_lambda`` [Pa] for :meth:`~newton.ModelBuilder.add_soft_mesh`.
    """
    return young / (2.0 * (1.0 + poisson)), young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))


def soft_hand_spec(spec: ch.HandSpec | None = None) -> ch.HandSpec:
    """Hand spec for the scene with the soft ball: the given spec with the ball at the foam ball's size.

    The soft helpers take the size of the ball from the ball of the spec, as the ball grasp of
    ``cable_hand.grasp_plan`` does, so one spec places the hand, the post and the ball. The hand itself
    is unchanged.

    Args:
        spec: The hand spec; the default spec if ``None``.

    Returns:
        A copy of the spec whose ``objects["ball"]["radius"]`` is :data:`SOFT_BALL_RADIUS`.
    """
    spec = copy.copy(ch.default_hand_spec() if spec is None else spec)
    spec.objects = {**spec.objects, "ball": {**spec.objects["ball"], "radius": SOFT_BALL_RADIUS}}
    return spec


def _kuhn_tets() -> tuple:
    """The six tetrahedra of the Kuhn split of a unit cube, as corner triples; all of positive volume."""
    tets = []
    for perm in itertools.permutations(range(3)):
        corner = [0, 0, 0]
        tet = [tuple(corner)]
        for axis in perm:
            corner[axis] = 1
            tet.append(tuple(corner))
        # odd permutations are mirrored: swap two vertices so that every tetrahedron has positive volume
        if sum(1 for a in range(3) for b in range(a + 1, 3) if perm[a] > perm[b]) % 2:
            tet[2], tet[3] = tet[3], tet[2]
        tets.append(tuple(tet))
    return tuple(tets)


_KUHN_TETS = _kuhn_tets()


def tet_volumes(vertices, tets) -> np.ndarray:
    """Signed volumes of tetrahedra.

    Args:
        vertices: Vertex positions [m], shape ``(vertex_count, 3)``.
        tets: Vertex indices, shape ``(tet_count, 4)``.

    Returns:
        The volumes [m^3], shape ``(tet_count,)``; negative for an inverted element.
    """
    v = np.asarray(vertices, dtype=float)
    t = np.asarray(tets, dtype=int)
    a, b, c, d = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]], v[t[:, 3]]
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def tet_surface_triangles(tets) -> np.ndarray:
    """Boundary faces of a tetrahedral mesh, wound so that the normals point outward.

    Args:
        tets: Vertex indices of positively oriented tetrahedra, shape ``(tet_count, 4)``.

    Returns:
        The triangles, shape ``(triangle_count, 3)``.
    """
    t = np.asarray(tets, dtype=int)
    faces = np.concatenate([t[:, [0, 2, 1]], t[:, [0, 1, 3]], t[:, [1, 2, 3]], t[:, [0, 3, 2]]])
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse.ravel()] == 1]


def ball_tet_mesh(radius: float, cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedral mesh of a ball, with every surface vertex exactly on the sphere.

    A cube of ``cells`` cells per edge is split into six tetrahedra per cell (Kuhn split, conforming
    across cells) and mapped onto the ball: a grid point ``p`` of the cube ``[-1, 1]^3`` goes to
    ``radius * p * max(|p_i|) / |p|``, so the shells of the cube become concentric spheres. The mesh is
    structured and smooth, with no staircase on the surface.

    Args:
        radius: Radius of the ball [m].
        cells: Number of cells across the ball; even, so that one vertex is the center.

    Returns:
        Vertices [m], shape ``((cells + 1)^3, 3)``, and tetrahedra, shape ``(6 cells^3, 4)``.
    """
    n = int(cells)
    if n < 2 or n % 2:
        raise ValueError(f"cells must be even and at least 2, got {cells}")
    axis = np.linspace(-1.0, 1.0, n + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    length = np.linalg.norm(grid, axis=1)
    scale = np.divide(np.abs(grid).max(axis=1), length, out=np.zeros_like(length), where=length > 0.0)
    vertices = radius * grid * scale[:, None]
    i, j, k = (c.reshape(-1) for c in np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"))
    # The split of a cell is mirrored per octant so that the diagonal shared by its six tetrahedra points
    # away from the center: with one split everywhere, the cells near the other diagonals of the cube
    # get tetrahedra with all four vertices on one shell, which the map flattens to slivers.
    low = [c < n // 2 for c in (i, j, k)]
    odd = (low[0].astype(int) + low[1] + low[2]) % 2 == 1
    tets = []
    for tet in _KUHN_TETS:
        corners = []
        for a, b, c in tet:
            ia, jb, kc = (np.where(flip, 1 - d, d) for flip, d in zip(low, (a, b, c), strict=True))
            corners.append(((i + ia) * (n + 1) + j + jb) * (n + 1) + k + kc)
        corners = np.stack(corners, axis=1)
        corners[odd] = corners[odd][:, [0, 1, 3, 2]]  # a mirrored tetrahedron: restore the orientation
        tets.append(corners)
    return vertices, np.stack(tets, axis=1).reshape(-1, 4)


def soft_object_mesh(spec: ch.HandSpec, cells: int = SOFT_BALL_CELLS) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedral mesh of the soft ball in the object frame (origin at the center of the ball).

    These are the particle centers: the mesh surface lies :data:`SOFT_PARTICLE_RADIUS` inside the
    contact surface, a sphere with the radius of the ball of the spec, so the particles touch a shape
    where the surface of the ball is.

    Args:
        spec: The hand spec, see :func:`soft_hand_spec`.
        cells: Number of mesh cells across the ball (even).

    Returns:
        Vertices [m], shape ``(vertex_count, 3)``, and tetrahedra, shape ``(tet_count, 4)``.
    """
    return ball_tet_mesh(float(spec.objects["ball"]["radius"]) - SOFT_PARTICLE_RADIUS, cells)


@wp.kernel
def particle_weight_kernel(
    particle_mass: wp.array[float],
    gravity: wp.vec3,
    particle_f: wp.array[wp.vec3],
):
    """Add the weight of every particle to the particle forces."""
    p = wp.tid()
    particle_f[p] = particle_f[p] + particle_mass[p] * gravity


class ParticleWeight:
    """Gravity on the particles as particle forces, see :func:`create_particle_weight`."""

    def __init__(self, model: newton.Model, gravity):
        self.model = model
        self.gravity = wp.vec3(*[float(v) for v in gravity])

    def apply(self, state: newton.State):
        """Add the weights to the particle forces of a state.

        Args:
            state: State whose ``particle_f`` receives the weights.
        """
        wp.launch(
            particle_weight_kernel,
            dim=self.model.particle_count,
            inputs=[self.model.particle_mass, self.gravity],
            outputs=[state.particle_f],
            device=self.model.device,
        )


def create_particle_weight(model: newton.Model, gravity=(0.0, 0.0, -ch.GRAVITY)) -> ParticleWeight:
    """Create the helper that applies gravity to the particles as particle forces.

    The hand models are built without model gravity and get the weight of their bodies as body forces
    (``cable_hand.create_body_weight``); the particles of a soft object get theirs the same way.

    Args:
        model: The finalized model, built with zero gravity.
        gravity: Gravity vector in the world frame [m/s^2].

    Returns:
        The helper; call :meth:`ParticleWeight.apply` in every substep after ``HandControl.step``
        (which clears the forces). A kernel launch; safe inside a captured graph.
    """
    return ParticleWeight(model, gravity)


@dataclass
class SoftObjectScene:
    """Indices and geometry of a grasp scene with a soft object, see :func:`add_soft_object_scene`."""

    pads: ch.HandPads
    """Pads of the hand."""
    table_shape: int
    """Shape index of the table (a static box with its top at ``z = 0``), -1 without a table."""
    pedestal_shape: int
    """Shape index of the post under the object, -1 without one."""
    seat_shapes: list[int]
    """Shape indices of the three seat balls on the rim of the post; empty without a post."""
    pedestal_height: float
    """Height of the lowest point of the object above the table at the start [m], zero without a post."""
    radius: float
    """Radius of the ball [m]: its contact surface, one particle radius outside the particle centers."""
    object_xform: wp.transform
    """World pose of the object frame at build time."""
    particle_start: int
    """Index of the first particle of the object."""
    particle_count: int
    """Number of particles of the object."""
    tet_start: int
    """Index of the first tetrahedron of the object."""
    tet_count: int
    """Number of tetrahedra of the object."""
    rest_vertices: np.ndarray
    """Rest positions of the particles in the object frame [m], shape ``(particle_count, 3)``."""
    tets: np.ndarray
    """Particle indices of the tetrahedra relative to ``particle_start``, shape ``(tet_count, 4)``."""
    triangles: np.ndarray
    """Surface triangles (outward normals), indices relative to ``particle_start``, shape ``(n, 3)``."""
    particle_radius: float
    """Contact radius of the particles [m]."""
    object_mass: float
    """Mass of the object [kg]."""
    material: dict[str, float] = field(default_factory=dict)
    """``"young"`` [Pa], ``"poisson"``, ``"density"`` [kg/m^3], ``"k_mu"``, ``"k_lambda"`` [Pa], ``"k_damp"``."""


def add_soft_object_scene(
    builder: newton.ModelBuilder,
    spec: ch.HandSpec,
    pads: ch.HandPads,
    object_xform: wp.transform | None = None,
    table_half_size: tuple[float, float, float] = (0.2, 0.2, 0.01),
    pedestal: tuple[float, float] | None = None,
    young: float = SOFT_YOUNG,
    poisson: float = SOFT_POISSON,
    density: float = SOFT_DENSITY,
    damping: float = SOFT_DAMPING,
    cells: int = SOFT_BALL_CELLS,
    skin: float | None = None,
) -> SoftObjectScene:
    """Add the table, an optional post and the soft ball to a builder that holds the hand and its pads.

    Call it after the hand, its cables and its pads were added and before ``builder.color()``: every
    shape that is in the builder at that point and is not a pad is excluded from particle contact.
    The pads and the support collide with the particles. The friction coefficient of a particle-shape
    pair is the geometric mean of ``Model.soft_contact_mu`` and the shape's;
    :func:`build_hand_soft_object_model` sets the former to the pad coefficient, which gives pad-object
    pairs ``spec.pads.mu`` and object-support pairs ``spec.pads.table_mu`` as in
    ``cable_hand.add_grasp_scene``.

    The ball rests in a seat on a post: three small balls on the rim of the post, as for the rigid ball,
    placed so that the ball starts in touch without overlap. On a post the ball is built at its measured
    rest pose in the seat (:data:`SOFT_SEAT_SETTLE_SHIFT`, :data:`SOFT_SEAT_SETTLE_TURN`). That pose was
    measured for the default radius, mesh and material and holds only for those; with other values the
    ball settles a little at the start. The object is a set of free particles: nothing but contact
    connects it to the hand or the support.

    Args:
        builder: The model builder; ``z`` is up and the table top is ``z = 0``.
        spec: The hand spec.
        pads: The pads of the hand, from ``cable_hand.add_hand_pads``.
        object_xform: World pose of the object frame (upright); above the origin with the underside
            on the table, or at the pedestal height, if ``None``.
        table_half_size: Half extents of the table box [m].
        pedestal: Radius of the post [m] and, used when ``object_xform`` is ``None``, the height of the
            underside of the object above the table [m]; or ``None`` for no post.
        young: Young's modulus [Pa].
        poisson: Poisson's ratio.
        density: Density [kg/m^3].
        damping: Viscous damping of the tetrahedra.
        cells: Number of mesh cells across the ball (even).
        skin: Factor on the moduli of the tetrahedra that have a face on the surface (a denser skin
            over a softer core); :data:`SOFT_SKIN_FACTOR` if ``None``.

    Returns:
        The scene.
    """
    particle_bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    pad_shapes = set(pads.shapes.values())
    for shape in range(builder.shape_count):
        if shape not in pad_shapes:
            builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~particle_bit
    vertices, tets = soft_object_mesh(spec, cells)
    ball = float(spec.objects["ball"]["radius"])
    underside = -ball  # the lowest point of the contact surface, in the object frame
    surface_mu = spec.pads.table_mu**2 / spec.pads.mu
    # table, post and seat balls keep the default contact damping of a shape (only the pads get
    # SOFT_PAD_CONTACT_KD); the recorded grasp was accepted with it
    surface_cfg = ch.grasp_shape_config(spec, surface_mu, contact_ke=SOFT_SEAT_CONTACT_KE)
    if object_xform is None:
        height = (0.0 if pedestal is None else float(pedestal[1])) - underside
        object_xform = wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity())
    position = wp.transform_get_translation(object_xform)
    table_shape = builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -table_half_size[2]), wp.quat_identity()),
        hx=table_half_size[0],
        hy=table_half_size[1],
        hz=table_half_size[2],
        cfg=surface_cfg,
        color=(0.78, 0.78, 0.76),
        label="grasp/table",
    )
    pedestal_shape = -1
    seat_shapes = []
    pedestal_height = 0.0
    if pedestal is not None:
        # The seat of the rigid ball: the post is a little lower than the underside of the ball and carries
        # three small balls on its rim, on which the ball rests. (A ball on a flat top rolls off at the
        # slightest push, and a flat on the ball is of no use: the ball turns in the hand.)
        pedestal_height = float(position[2]) + underside
        radius = float(pedestal[0])
        drop = math.sqrt((ball + ch.GRASP_SEAT_BALL_RADIUS) ** 2 - radius**2)
        post_height = pedestal_height + ball - drop
        for i in range(3):
            angle = (0.25 + i / 3.0) * 2.0 * math.pi
            seat = (position[0] + radius * math.cos(angle), position[1] + radius * math.sin(angle), post_height)
            seat_shapes.append(
                builder.add_shape_sphere(
                    body=-1,
                    xform=wp.transform(wp.vec3(*seat), wp.quat_identity()),
                    radius=ch.GRASP_SEAT_BALL_RADIUS,
                    cfg=surface_cfg,
                    color=(0.45, 0.45, 0.45),
                    label=f"grasp/seat{i}",
                )
            )
        pedestal_shape = builder.add_shape_cylinder(
            body=-1,
            xform=wp.transform(wp.vec3(position[0], position[1], 0.5 * post_height), wp.quat_identity()),
            radius=radius,
            half_height=0.5 * post_height,
            cfg=surface_cfg,
            color=(0.45, 0.45, 0.45),
            label="grasp/pedestal",
        )
    k_mu, k_lambda = lame_parameters(young, poisson)
    particle_start = builder.particle_count
    tet_start = builder.tet_count
    rotation = wp.transform_get_rotation(object_xform)
    if pedestal is not None:
        # the ball starts where it rests in its seat (see SOFT_SEAT_SETTLE_SHIFT)
        position = wp.vec3(*(np.array(position, dtype=float) + np.array(SOFT_SEAT_SETTLE_SHIFT)).tolist())
        rotation = rotation * wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), SOFT_SEAT_SETTLE_TURN)
    builder.add_soft_mesh(
        pos=position,
        rot=rotation,
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=vertices.tolist(),
        indices=tets.flatten().tolist(),
        density=density,
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=damping,
        particle_radius=SOFT_PARTICLE_RADIUS,
        label="grasp/soft_ball",
    )
    skin = SOFT_SKIN_FACTOR if skin is None else skin
    if skin != 1.0:
        # a denser skin: the tetrahedra with a face on the surface are stiffer than the core
        faces = np.sort(
            np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [1, 2, 3]], tets[:, [0, 3, 2]]]), 1
        )
        _, inverse, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
        outer = (counts[inverse.ravel()] == 1).reshape(4, -1).any(axis=0)
        for tet in np.nonzero(outer)[0]:
            builder.tet_materials[tet_start + int(tet)] = (skin * k_mu, skin * k_lambda, damping)
    return SoftObjectScene(
        pads=pads,
        table_shape=table_shape,
        pedestal_shape=pedestal_shape,
        seat_shapes=seat_shapes,
        pedestal_height=pedestal_height,
        radius=ball,
        object_xform=object_xform,
        particle_start=particle_start,
        particle_count=builder.particle_count - particle_start,
        tet_start=tet_start,
        tet_count=builder.tet_count - tet_start,
        rest_vertices=vertices,
        tets=tets,
        triangles=tet_surface_triangles(tets),
        particle_radius=SOFT_PARTICLE_RADIUS,
        object_mass=float(np.sum(builder.particle_mass[particle_start:])),
        material={
            "young": young,
            "poisson": poisson,
            "density": density,
            "k_mu": k_mu,
            "k_lambda": k_lambda,
            "k_damp": damping,
        },
    )


def build_hand_soft_object_model(
    spec: ch.HandSpec | None = None,
    object_xform: wp.transform | None = None,
    palm_xform: wp.transform | None = None,
    wrist: str = "kinematic",
    device=None,
    pedestal: tuple[float, float] | None = None,
    visuals: bool = False,
    **material,
) -> tuple[newton.Model, ch.HandSkeleton, dict[str, Any], SoftObjectScene]:
    """Build and finalize the hand with massless tendons, pads, a table, a post and the soft ball.

    The hand is that of ``cable_hand.build_tendon_hand_grasp_model``. The model has no gravity: pass
    ``gravity=(0, 0, -GRAVITY)`` to ``cable_hand.create_hand_control`` and apply a
    :func:`create_particle_weight` helper after the hand control in every substep. Use
    :func:`create_soft_object_collision_pipeline` and :func:`make_hand_soft_object_solver`.

    Args:
        spec: The hand spec; :func:`soft_hand_spec` if ``None``.
        object_xform: World pose of the object frame, see :func:`add_soft_object_scene`.
        palm_xform: World pose of the palm frame at the start; identity if ``None``.
        wrist: ``"kinematic"`` or ``"fixed"``.
        device: Device of the model.
        pedestal: Radius and height [m] of the post under the object, or ``None``.
        visuals: If ``True``, attach the shipped visual meshes of the hand and hide the pad capsules.
        **material: ``young``, ``poisson``, ``density``, ``damping``, ``cells`` and ``skin`` of
            :func:`add_soft_object_scene`.

    Returns:
        The finalized model, the hand skeleton, the tendon indices per actuator key and the scene.
    """
    if spec is None:
        spec = soft_hand_spec()
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    hand, tendons, pads = _add_tendon_hand(builder, spec, wrist, palm_xform, not visuals)
    scene = add_soft_object_scene(builder, spec, pads, object_xform, pedestal=pedestal, **material)
    if visuals:
        ch.add_hand_visuals(builder, hand, spec=spec)
        # the visual meshes were added after the scene: they do not touch the particles either
        particle_bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        keep = {*pads.shapes.values(), scene.table_shape, scene.pedestal_shape, *scene.seat_shapes}
        for shape in range(builder.shape_count):
            if shape not in keep:
                builder.shape_flags[shape] = int(builder.shape_flags[shape]) & ~particle_bit
    # the pads touch nothing but the particles in this scene: their side of the contact is as soft as the
    # particles' side (see SOFT_PAD_CONTACT_KE)
    for shape in pads.shapes.values():
        builder.shape_material_ke[shape] = SOFT_PAD_CONTACT_KE
        builder.shape_material_kd[shape] = SOFT_PAD_CONTACT_KD
    builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_ke = SOFT_CONTACT_KE
    model.soft_contact_kd = SOFT_CONTACT_KD
    model.soft_contact_mu = spec.pads.mu
    return model, hand, tendons, scene


def create_soft_object_collision_pipeline(model: newton.Model, **options) -> newton.CollisionPipeline:
    """Create the collision pipeline of a grasp scene with a soft object.

    The pipeline of ``cable_hand.create_grasp_collision_pipeline`` with the particle-shape detection
    distance of :data:`SOFT_CONTACT_GAP`. Create it before the solver.

    Args:
        model: The finalized model.
        **options: Further keyword arguments of :class:`newton.CollisionPipeline`.

    Returns:
        The collision pipeline.
    """
    options.setdefault("soft_contact_gap", SOFT_CONTACT_GAP)
    return newton.CollisionPipeline(model, contact_matching="latest", **options)


def make_hand_soft_object_solver(model: newton.Model, **overrides) -> newton.solvers.SolverVBD:
    """Create the solver for the hand with a soft object.

    The settings of ``cable_hand.make_tendon_hand_grasp_solver``: the particles are solved with the
    iterations and the step of the hand. The friction regularization is :data:`SOFT_FRICTION_EPSILON`.

    Args:
        model: The model from :func:`build_hand_soft_object_model`.
        **overrides: Solver keyword arguments that replace the defaults.

    Returns:
        The solver.
    """
    overrides.setdefault("friction_epsilon", SOFT_FRICTION_EPSILON)
    return ch.make_tendon_hand_grasp_solver(model, **overrides)


def soft_object_state(scene: SoftObjectScene, particle_q) -> dict[str, Any]:
    """Shape measures of the soft object from particle positions.

    Args:
        scene: The scene.
        particle_q: Particle positions of the model [m], shape ``(particle_count, 3)`` (numpy).

    Returns:
        ``"center"`` [m]: centroid of the particles, shape ``(3,)``;
        ``"volume"`` [m^3]: sum of the tetrahedron volumes;
        ``"volume_ratio_min"``: smallest ratio of a tetrahedron's volume to its rest volume (zero or
        negative for an inverted element);
        ``"rotation"``: best-fit rotation from the rest shape, shape ``(3, 3)``;
        ``"deformation"`` [m]: per-particle distance from the best-fit rigid placement of the rest
        shape, shape ``(particle_count,)``.
    """
    x = np.asarray(particle_q, dtype=float)[scene.particle_start : scene.particle_start + scene.particle_count]
    rest = scene.rest_vertices
    volumes = tet_volumes(x, scene.tets)
    center = x.mean(axis=0)
    rest_center = rest.mean(axis=0)
    u, _, vt = np.linalg.svd((x - center).T @ (rest - rest_center))
    rotation = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    fitted = (rest - rest_center) @ rotation.T + center
    return {
        "center": center,
        "volume": float(volumes.sum()),
        "volume_ratio_min": float((volumes / tet_volumes(rest, scene.tets)).min()),
        "rotation": rotation,
        "deformation": np.linalg.norm(x - fitted, axis=-1),
    }


def soft_object_in_palm(scene: SoftObjectScene, particle_q, palm_q) -> np.ndarray:
    """Pose of the best-fit rigid placement of the soft object in the palm frame.

    Args:
        scene: The scene.
        particle_q: Particle positions of the model [m] (numpy).
        palm_q: Pose of the palm body (position [m] and quaternion ``x, y, z, w``).

    Returns:
        Position of the centroid [m] and orientation quaternion in the palm frame, shape ``(7,)``:
        the reference for ``cable_hand.grasp_slip``.
    """
    state = soft_object_state(scene, particle_q)
    quat = np.array(wp.quat_from_matrix(wp.mat33(*state["rotation"].flatten().tolist())), dtype=float)
    return _pose_in_palm(np.asarray(palm_q, dtype=float), np.concatenate([state["center"], quat]))


def _soft_object_extent(scene: SoftObjectScene, particle_q, palm_q) -> np.ndarray:
    """Size of the soft object along the axes of the palm frame.

    Args:
        scene: The scene.
        particle_q: Particle positions of the model [m] (numpy).
        palm_q: Pose of the palm body (position [m] and quaternion ``x, y, z, w``).

    Returns:
        Extent along palm ``x`` (finger direction), ``y`` (palm normal: between the palm pads and the
        fingers that reach under the object) and ``z`` (toward the thumb) [m], including the
        particle radius on both sides, shape ``(3,)``.
    """
    x = np.asarray(particle_q, dtype=float)[scene.particle_start : scene.particle_start + scene.particle_count]
    q = np.asarray(palm_q, dtype=float)
    inverse = np.concatenate([-q[3:6], q[6:]])
    local = _rotate(inverse, x - q[:3])
    return local.max(axis=0) - local.min(axis=0) + 2.0 * scene.particle_radius


SOFT_GRIP_PADS = (("thumb/link2",), ("index/link3", "middle/link3"))
"""The two groups of pads a soft object is squeezed between in the ball grasp: the middle pad of the
thumb on one side, the fingertip pads of the index and the middle finger on the other."""


def soft_grip_line(scene: SoftObjectScene, body_q, pads=SOFT_GRIP_PADS) -> tuple[np.ndarray, np.ndarray, float]:
    """Line between two opposing groups of pads.

    Args:
        scene: The scene.
        body_q: Body poses of the model (numpy).
        pads: Two tuples of pad keys (``scene.pads.keys``), one per side.

    Returns:
        The mean axis midpoint of the first group [m], the unit vector from it to that of the second
        group, and the gap between the pad surfaces along that line [m] (the distance of the two
        points minus the mean pad radius of each group).
    """
    body_q = np.asarray(body_q, dtype=float)
    ends, radii = [], []
    for group in pads:
        index = [scene.pads.keys.index(key) for key in group]
        q = body_q[scene.pads.body[index]]
        local = 0.5 * (scene.pads.p0[index] + scene.pads.p1[index])
        ends.append((q[:, :3] + _rotate(q[:, 3:], local)).mean(axis=0))
        radii.append(float(scene.pads.radius[index].mean()))
    axis = ends[1] - ends[0]
    length = float(np.linalg.norm(axis))
    return ends[0], axis / length, length - radii[0] - radii[1]


def soft_grip_particles(
    scene: SoftObjectScene, particle_q, body_q, pads=SOFT_GRIP_PADS, radius: float = 0.010
) -> np.ndarray:
    """Particles of the soft object that lie between two opposing groups of pads.

    Args:
        scene: The scene.
        particle_q: Particle positions of the model [m] (numpy).
        body_q: Body poses of the model (numpy).
        pads: Two tuples of pad keys (``scene.pads.keys``), one per side.
        radius: Radius of the tube around the line of :func:`soft_grip_line` in which particles count [m].

    Returns:
        Indices of those particles, relative to ``scene.particle_start``.
    """
    x = np.asarray(particle_q, dtype=float)[scene.particle_start : scene.particle_start + scene.particle_count]
    start, axis, _ = soft_grip_line(scene, body_q, pads)
    along = (x - start) @ axis
    return np.nonzero(np.linalg.norm(x - start - along[:, None] * axis, axis=1) < radius)[0]


def soft_grip_width(scene: SoftObjectScene, particle_q, body_q, particles, pads=SOFT_GRIP_PADS) -> float:
    """Thickness of the soft object between two opposing groups of pads.

    The thickness is the extent of a fixed set of particles along the line between the pads
    (:func:`soft_grip_line`). Choose the set once, with :func:`soft_grip_particles` while the object is
    held, and follow it: the same material is measured in the hold and in a squeeze.

    Args:
        scene: The scene.
        particle_q: Particle positions of the model [m] (numpy).
        body_q: Body poses of the model (numpy).
        particles: Indices of the particles to measure, relative to ``scene.particle_start``.
        pads: Two tuples of pad keys (``scene.pads.keys``), one per side.

    Returns:
        The thickness, particle radius included on both sides [m].
    """
    x = np.asarray(particle_q, dtype=float)[scene.particle_start : scene.particle_start + scene.particle_count]
    start, axis, _ = soft_grip_line(scene, body_q, pads)
    along = (x[np.asarray(particles, dtype=int)] - start) @ axis
    return float(along.max() - along.min()) + 2.0 * scene.particle_radius


def _soft_object_contact_forces(model: newton.Model, scene: SoftObjectScene, state: newton.State, contacts):
    """Normal contact forces on the particles of the soft object, from the penalty law of the solver.

    Friction is not included: the normal force of a particle-shape contact is the mean of
    ``Model.soft_contact_ke`` and the shape's stiffness times the overlap.

    Args:
        model: The model.
        scene: The scene.
        state: The state the contacts were detected for.
        contacts: The contacts of the collision pipeline.

    Returns:
        ``"pad_force"`` [N]: normal force of each pad on the object in the world frame, shape
        ``(pad_count, 3)`` in the order of ``scene.pads.keys``;
        ``"support_force"`` [N]: normal force of the table and the post, shape ``(3,)``;
        ``"pad_contacts"`` and ``"support_contacts"``: number of contacts with overlap.
    """
    n = int(contacts.soft_contact_count.numpy()[0])
    pad_force = np.zeros((len(scene.pads.keys), 3))
    result = {"pad_force": pad_force, "support_force": np.zeros(3), "pad_contacts": 0, "support_contacts": 0}
    if n == 0:
        return result
    particle = contacts.soft_contact_particle.numpy()[:n]
    shape = contacts.soft_contact_shape.numpy()[:n]
    body_pos = contacts.soft_contact_body_pos.numpy()[:n].astype(float)
    normal = contacts.soft_contact_normal.numpy()[:n].astype(float)
    shape_body = model.shape_body.numpy()[shape]
    body_q = state.body_q.numpy().astype(float)
    world = body_pos.copy()
    on_body = shape_body >= 0
    if on_body.any():
        q = body_q[shape_body[on_body]]
        world[on_body] = q[:, :3] + _rotate(q[:, 3:], body_pos[on_body])
    x = state.particle_q.numpy().astype(float)[particle]
    radius = model.particle_radius.numpy()[particle] + model.shape_margin.numpy()[shape]
    depth = np.maximum(radius - np.einsum("ij,ij->i", normal, x - world), 0.0)
    ke = 0.5 * (float(model.soft_contact_ke) + model.shape_material_ke.numpy()[shape])
    force = (ke * depth)[:, None] * normal
    for i, key in enumerate(scene.pads.keys):
        pad_force[i] = force[shape == scene.pads.shapes[key]].sum(axis=0)
    support = np.isin(shape, [scene.table_shape, scene.pedestal_shape, *scene.seat_shapes])
    result["support_force"] = force[support].sum(axis=0)
    result["pad_contacts"] = int(((depth > 0.0) & ~support).sum())
    result["support_contacts"] = int(((depth > 0.0) & support).sum())
    return result


@dataclass
class SoftDisplaySurface:
    """A fine surface to draw a soft object with, see :func:`create_soft_display_surface`."""

    triangles: np.ndarray
    """Triangles of the display surface (outward normals), shape ``(triangle_count, 3)``."""
    tets: np.ndarray
    """Per display vertex, the model indices of the four particles of the tetrahedron that carries
    it, shape ``(vertex_count, 4)``."""
    weights: np.ndarray
    """Per display vertex, its barycentric weights in that tetrahedron at rest (they sum to one; a
    weight may be slightly negative for a vertex just outside the mesh), shape ``(vertex_count, 4)``."""
    offset: float
    """Distance by which the surface is drawn outside the carried points, along its normal [m]: the
    contact radius of the particles."""


def _icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit sphere of ``20 * 4**subdivisions`` triangles with outward normals: vertices and triangles."""
    t = 0.5 * (1.0 + math.sqrt(5.0))
    vertices = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
        (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]  # fmt: skip
    triangles = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
        (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
        (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    vertices = [np.array(v, dtype=float) / math.sqrt(1.0 + t * t) for v in vertices]
    for _ in range(subdivisions):
        middle: dict[tuple[int, int], int] = {}

        def midpoint(a, b, middle=middle):
            key = (min(a, b), max(a, b))
            if key not in middle:
                m = vertices[a] + vertices[b]
                vertices.append(m / np.linalg.norm(m))
                middle[key] = len(vertices) - 1
            return middle[key]

        split = []
        for a, b, c in triangles:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            split += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        triangles = split
    return np.array(vertices), np.array(triangles, dtype=int)


def create_soft_display_surface(scene: SoftObjectScene, subdivisions: int = 4) -> SoftDisplaySurface:
    """Embed a fine sphere in the tetrahedral mesh of the soft ball, to draw the ball with.

    The simulation mesh is coarse, and its particles are centers of contact spheres: contact acts
    :data:`SOFT_PARTICLE_RADIUS` outside them. Drawing the triangles between the particles therefore
    shows a faceted ball that hangs one particle radius off everything it touches. The display surface
    is a sphere of ``20 * 4**subdivisions`` triangles (5120 by default) whose vertices are carried by
    the tetrahedra (barycentric weights, computed here once at rest) and then moved outward by the
    particle radius along the surface normal: it is the contact surface of the deformed ball, smooth,
    and at rest it is the sphere of ``scene.radius``. It costs nothing in the simulation.

    Args:
        scene: The scene with the soft ball.
        subdivisions: Number of subdivisions of the icosahedron the sphere is made from.

    Returns:
        The display surface; get its vertices for a state with :func:`soft_display_surface_vertices`.
    """
    directions, triangles = _icosphere(subdivisions)
    # the points that the tetrahedra carry lie on the mesh of particle centers: the same ball, smaller by
    # the particle radius
    carried = (scene.radius - scene.particle_radius) * directions
    rest = scene.rest_vertices
    origin = rest[scene.tets[:, 0]]
    edges = np.stack([rest[scene.tets[:, i]] - origin for i in (1, 2, 3)], axis=2)
    inverse = np.linalg.inv(edges)
    tets = np.empty(len(carried), dtype=int)
    weights = np.empty((len(carried), 4))
    for start in range(0, len(carried), 256):
        points = carried[start : start + 256]
        partial = np.einsum("tij,ptj->pti", inverse, points[:, None, :] - origin[None, :, :])
        full = np.concatenate([1.0 - partial.sum(axis=2, keepdims=True), partial], axis=2)
        best = np.argmax(full.min(axis=2), axis=1)  # the tetrahedron that contains the point, or the nearest
        tets[start : start + 256] = best
        weights[start : start + 256] = full[np.arange(len(points)), best]
    return SoftDisplaySurface(
        triangles=triangles,
        tets=scene.tets[tets] + scene.particle_start,
        weights=weights,
        offset=scene.particle_radius,
    )


def soft_display_surface_vertices(surface: SoftDisplaySurface, particle_q) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and normals of the display surface of a soft object for given particle positions.

    Each vertex is carried by its tetrahedron (the barycentric combination of four particles) and then
    moved outward by ``surface.offset``, the contact radius of the particles, along the normal of the
    carried surface. The result is the surface on which the object touches the pads and its support:
    drawn with it, the object shows no gap where it is in contact. (A pad that rests between two
    particles sinks up to about 1 mm into this surface: the contact spheres leave valleys between
    them.)

    Args:
        surface: The display surface from :func:`create_soft_display_surface`.
        particle_q: Particle positions of the model [m], shape ``(particle_count, 3)`` (numpy).

    Returns:
        The vertices [m] and their unit normals, both of shape ``(vertex_count, 3)``; the triangles are
        ``surface.triangles``.
    """
    x = np.asarray(particle_q, dtype=float)
    carried = np.einsum("vi,vij->vj", surface.weights, x[surface.tets])
    a, b, c = (carried[surface.triangles[:, i]] for i in range(3))
    face = np.cross(b - a, c - a)  # area-weighted face normals
    normals = np.zeros_like(carried)
    for i in range(3):
        np.add.at(normals, surface.triangles[:, i], face)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    return carried + surface.offset * normals, normals


@dataclass
class SoftObjectPlan:
    """A scripted grasp of the soft ball, see :func:`soft_object_grasp_plan`."""

    object_xform: wp.transform
    """World pose of the object frame at the start."""
    pedestal: tuple[float, float] | None
    """Radius of the post under the object and height of the object's underside above the table [m]."""
    palm_start: wp.transform
    """World pose of the palm frame at the start (build the hand there)."""
    palm_grasp: wp.transform
    """World pose of the palm frame at the grasp."""
    opposition: float
    """Thumb opposition angle of the grasp [rad]."""
    hold_tension: float
    """Cable tension of the hold [N]."""
    squeeze_tension: float
    """Cable tension at the top of a squeeze pulse [N]."""
    lift: float
    """Height of the lift [m]."""
    keyframes: list[ch.HandKeyframe]
    """The keyframes for ``cable_hand.create_hand_script``."""
    phases: dict[str, tuple[float, float]]
    """Start and end time of every phase [s]; the pulses are ``"squeeze1"``, ``"squeeze_hold1"``, ``"relax1"``, ..."""


def soft_object_grasp_plan(
    spec: ch.HandSpec,
    hold_tension: float = SOFT_HOLD_TENSION,
    squeeze_tension: float = SOFT_SQUEEZE_TENSION,
    thumb_tension: float = SOFT_THUMB_TENSION,
    thumb_squeeze_tension: float | None = SOFT_THUMB_SQUEEZE_TENSION,
    pulses: int = 1,
    phases: dict[str, float] | None = None,
    set_down_shift: tuple[float, float] = SOFT_SET_DOWN_SHIFT,
    **plan_options,
) -> SoftObjectPlan:
    """Keyframes of the grasp of the soft ball: the ball grasp with squeeze pulses in place of the tilt.

    The hand takes the soft ball like the ball of the spec (from above), lifts and holds it. In the lifted
    hold the cables of the fingers ramp to ``squeeze_tension`` and back to ``hold_tension``, ``pulses``
    times; the thumb keeps ``thumb_tension`` (or ramps to ``thumb_squeeze_tension`` with the fingers) and
    acts as the jaw the fingers squeeze against. Then the object is lowered into its seat and released as in
    ``cable_hand.grasp_plan``: the thumb lets go and turns back, then the fingers open, then the palm
    eases off and the hand retreats.

    Args:
        spec: The hand spec, see :func:`soft_hand_spec`.
        hold_tension: Cable tension of the fingers in the hold [N].
        squeeze_tension: Cable tension of the fingers at the top of a squeeze pulse [N].
        thumb_tension: Cable tension of the thumb from the grasp to the release [N].
        thumb_squeeze_tension: Cable tension of the thumb at the top of a squeeze pulse [N];
            ``thumb_tension`` if ``None``.
        pulses: Number of squeeze pulses.
        phases: Durations that replace those of ``cable_hand.GRASP_PHASES`` and :data:`SOFT_GRASP_PHASES` [s].
        set_down_shift: World ``(x, y)`` by which the palm is shifted at the set-down against the pose
            at which it took the object [m]: the opposite of how far the object has moved in the hand.
        **plan_options: Further keyword arguments of ``cable_hand.grasp_plan`` (``lift``, ``approach``,
            ``opposition``, ``center_z``, ``set_down_clearance``, ...).

    Returns:
        The plan.
    """
    durations = dict(SOFT_GRASP_PHASES)
    durations.update({k: v for k, v in (phases or {}).items() if k in SOFT_GRASP_PHASES})
    base_phases = {
        "end": SOFT_END_REST,
        **SOFT_TAKE_PHASES,
        **{k: v for k, v in (phases or {}).items() if k not in SOFT_GRASP_PHASES},
    }
    plan_options.setdefault("set_down_clearance", SOFT_SET_DOWN_CLEARANCE)
    base = ch.grasp_plan(spec, "ball", hold_tension=hold_tension, tilt=0.0, phases=base_phases, **plan_options)
    close = base.keyframes[1 + list(base.phases).index("close")].tension
    digits = list(close)
    squeeze: dict[str, Any] = dict.fromkeys(digits, squeeze_tension)
    relax: dict[str, Any] = dict.fromkeys(digits, hold_tension)
    if thumb_squeeze_tension is not None:
        squeeze["thumb"] = thumb_squeeze_tension
        relax["thumb"] = thumb_tension
    pulse = []
    for i in range(1, pulses + 1):
        pulse.append((f"squeeze{i}", durations["squeeze"], {"tension": squeeze}))
        pulse.append((f"squeeze_hold{i}", durations["squeeze_hold"], {}))
        pulse.append((f"relax{i}", durations["relax"], {"tension": relax}))
    pulse.append(("recover", durations["recover"], {}))
    keyframes = [base.keyframes[0]]
    lowered = base.keyframes[0].wrist
    windows: dict[str, tuple[float, float]] = {}
    t = 0.0

    def add(name, duration, **change):
        nonlocal t
        windows[name] = (t, t + duration)
        t += duration
        keyframes.append(ch.HandKeyframe(t, **change))

    for (name, (t0, t1)), key in zip(base.phases.items(), base.keyframes[1:], strict=True):
        if name in ("tilt_hold", "untilt"):
            continue
        if name == "tilt":
            for pulse_name, pulse_duration, change in pulse:
                add(pulse_name, pulse_duration, **change)
        elif name == "thumb":
            add(name, t1 - t0, tension={"thumb": thumb_tension})
        elif name == "retreat":
            # the palm pads still lie on the object: take them off slowly, then retreat
            eased = np.array(lowered, dtype=float)
            eased[2] += SOFT_EASE_HEIGHT
            add("ease", durations["ease"], wrist=eased)
            add(name, t1 - t0, wrist=key.wrist)
        elif name == "lower":
            # the object has moved in the hand since it was picked up: the hand sets it down beside where
            # it took it, so that the object comes down over its seat
            lowered = np.array(key.wrist, dtype=float)
            lowered[:2] += np.asarray(set_down_shift, dtype=float)
            add(name, t1 - t0, wrist=lowered)
        else:
            add(name, t1 - t0, wrist=key.wrist, opposition=key.opposition, tension=key.tension)
    return SoftObjectPlan(
        object_xform=base.object_xform,
        pedestal=base.pedestal,
        palm_start=base.palm_start,
        palm_grasp=base.palm_grasp,
        opposition=base.opposition,
        hold_tension=float(hold_tension),
        squeeze_tension=float(squeeze_tension),
        lift=base.lift,
        keyframes=keyframes,
        phases=windows,
    )
