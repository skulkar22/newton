# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visual meshes of the cable-driven hand example.

The hand of the cable-driven hand examples is built from a design specification
(link sections, knuckle radii, cable stations, palm boxes, actuator boxes, pads).
This module turns that specification into triangle meshes that show the hand as a
machined product: side plates with windows, slotted covers, flanged sheaves, soft
pads, the actuators and their carriages. The meshes are purely visual; collision
shapes are created elsewhere from the same specification.

All geometry is generated in the frame of the body that carries it, in meters:

- ``palm``: origin at the center of the wrist end of the palm, +x distal,
  +y palmar, +z toward the thumb (right hand).
- ``thumb_column``: the thumb digit frame (origin on the thumb MCP axis, +x along
  the straight thumb, +y palmar, +z the flexion axis); it turns with the
  opposition hinge.
- ``<digit>/link1`` .. ``<digit>/link3``: origin on the proximal joint axis of the
  link, +x toward the next joint, +y palmar, +z the flexion axis.
- ``<digit>/slider``: parallel to the digit frame (the palm-mounted finger frame,
  or the thumb digit frame) and coincident with it at zero travel; it moves along
  the pull direction of the slider. It carries the carriage of the digit's first
  (long) cable.
- ``<digit>/slider_<cable name>``: the same for every further cable of a digit
  (``index/slider_knuckle`` and ``middle/slider_knuckle``).

A digit of the specification gives its cables as a list ``cables`` (each with
``stations``, ``slider``, ``plane_z`` and the envelope of its linear unit).

The module needs only NumPy to generate geometry. ``pxr`` is imported inside the
two functions that write and read the USD asset.

Regenerate the shipped asset with:

- ``python -m newton.examples.cable.cable_hand_visuals``

The palm is one modeled body with heel, wrist stem and flange, and a thenar
lobe with a turned thumb seat.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from dataclasses import dataclass, field

import numpy as np

ASSET_NAME = "cable_hand/cable_hand_visuals.usd"
"""Location of the shipped asset below the examples asset directory."""

FINGERS = ("index", "middle", "ring", "little")

MATERIALS: dict[str, dict] = {
    "frame": {"color": (0.085, 0.09, 0.10), "roughness": 0.38, "metallic": 1.0},
    "cover": {"color": (0.82, 0.81, 0.78), "roughness": 0.45, "metallic": 0.0},
    "tip": {"color": (0.82, 0.81, 0.78), "roughness": 0.45, "metallic": 0.0},
    "housing": {"color": (0.82, 0.81, 0.78), "roughness": 0.45, "metallic": 0.0},
    "pad": {"color": (0.16, 0.17, 0.18), "roughness": 0.85, "metallic": 0.0},
    "sheave": {"color": (0.83, 0.62, 0.26), "roughness": 0.28, "metallic": 1.0},
    "guide": {"color": (0.72, 0.73, 0.75), "roughness": 0.25, "metallic": 1.0},
    "pin": {"color": (0.62, 0.63, 0.65), "roughness": 0.3, "metallic": 1.0},
    "actuator": {"color": (0.25, 0.26, 0.28), "roughness": 0.5, "metallic": 1.0},
    "carriage": {"color": (0.85, 0.33, 0.08), "roughness": 0.4, "metallic": 1.0},
}
"""Default look per material tag: base color, roughness and metallic."""

# Visual-only dimensions [m]. They are not part of the design specification; every
# part built with them stays inside the envelope the specification defines.
_PLATE = 1.6e-3  # thickness of a side plate
_RUN = 0.25e-3  # running clearance between plates of neighboring bodies
_GAP = 0.2e-3  # set-back of an end face from its joint plane
_BOSS = 0.45  # axle boss radius as a fraction of the knuckle radius
_BITE = 0.5e-3  # clearance around the boss of the neighboring body
_STRUT = 2.2e-3  # width of the plate left around a window
_FLOOR = 1.0e-3  # thickness of the palmar floor of a link
_BEVEL = 0.3e-3  # edge break on outer faces
_PAD_SINK = 0.2e-3  # depth by which a pad is let into its floor
_HUB = 3.6e-3  # half length of a sheave hub
_PIN = 1.2e-3  # radius of the visible axle pins
_COVER = 1.0e-3  # thickness of the palm covers
_CHEEK = 1.4e-3  # least thickness of a knuckle cheek of the palm
_PALM_FRONT = 20.8e-3  # distance of the palm front wall behind an MCP axis
_CHEEK_BACK = 24.0e-3  # distance a knuckle cheek reaches behind its MCP axis
_SLOT = 3.4e-3  # half width of the cover slot above a carriage
_SEG = 24  # polygon segments of a full circle


@dataclass
class VisualPart:
    """One visual mesh of the hand, in the frame of the body that carries it.

    ``vertices`` and ``indices`` describe the mesh before ``pos`` and ``quat`` are
    applied. Parts that share a ``prototype`` name share the same arrays, so a
    renderer can instance them.
    """

    name: str
    """Part name, unique within its body."""
    body: str
    """Body key: ``palm``, ``thumb_column``, ``<digit>/link1`` .. ``link3`` or ``<digit>/slider``."""
    vertices: np.ndarray
    """Vertex positions [m], shape ``(n, 3)``, ``float32``. Hard edges have split vertices."""
    indices: np.ndarray
    """Triangle vertex indices, shape ``(m, 3)``, ``int32``, counterclockwise seen from outside."""
    material: str
    """Material tag, a key of :data:`MATERIALS`."""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Position of the mesh frame in the body frame [m]."""
    quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    """Orientation of the mesh frame in the body frame, quaternion ``(x, y, z, w)``."""
    prototype: str | None = None
    """Name of the shared mesh for instanced parts, else ``None``."""


# ----------------------------------------------------------------------------- small helpers
def _rot_axis(axis, angle):
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def _quat_from_matrix(m):
    """Quaternion ``(x, y, z, w)`` of a rotation matrix."""
    m = np.asarray(m, dtype=float)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _frame_from_z(axis):
    """Rotation matrix whose third column is ``axis`` (deterministic choice of the other two)."""
    z = np.asarray(axis, dtype=float)
    z = z / np.linalg.norm(z)
    ref = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _merge(meshes):
    """Concatenate ``(vertices, triangles)`` pairs."""
    meshes = [m for m in meshes if m is not None and len(m[1])]
    verts, tris, offset = [], [], 0
    for v, t in meshes:
        verts.append(np.asarray(v, dtype=float))
        tris.append(np.asarray(t, dtype=np.int64) + offset)
        offset += len(v)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(verts), np.vstack(tris)


def _moved(mesh, rot=None, pos=None):
    v, t = mesh
    v = np.asarray(v, dtype=float)
    if rot is not None:
        v = v @ np.asarray(rot, dtype=float).T
    if pos is not None:
        v = v + np.asarray(pos, dtype=float)
    return v, t


def _mirrored_z(mesh):
    v, t = mesh
    return v * np.array([1.0, 1.0, -1.0]), np.asarray(t)[:, ::-1]


def _both_sides(mesh):
    return _merge([mesh, _mirrored_z(mesh)])


def _planform(mesh):
    """Map a mesh extruded from ``(u, v)`` along ``w`` to ``x = v``, ``y = w``, ``z = u``."""
    v, t = mesh
    return v[:, [1, 2, 0]], t


def _arc(center, radius, a0, a1, seg=_SEG):
    """Points on a circle from angle ``a0`` to ``a1`` (both ends included)."""
    n = max(2, math.ceil(abs(a1 - a0) / (2.0 * math.pi / seg)))
    a = np.linspace(a0, a1, n + 1)
    return np.column_stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)])


def _area(loop):
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _clean_loop(loop, ccw):
    """Drop repeated points and give the loop the requested orientation."""
    loop = np.asarray(loop, dtype=float)
    keep = np.linalg.norm(loop - np.roll(loop, 1, axis=0), axis=1) > 1.0e-9
    loop = loop[keep]
    if (_area(loop) > 0.0) != ccw:
        loop = loop[::-1]
    return loop


# ----------------------------------------------------------------------------- convex polygon tools
def _clip_halfplane(poly, normal, offset):
    """Part of a polygon with ``normal . p <= offset`` (Sutherland-Hodgman)."""
    out = []
    n = len(poly)
    if n == 0:
        return np.zeros((0, 2))
    d = poly @ np.asarray(normal, dtype=float) - offset
    for i in range(n):
        j = (i + 1) % n
        if d[i] <= 0.0:
            out.append(poly[i])
        if (d[i] < 0.0 < d[j]) or (d[j] < 0.0 < d[i]):
            out.append(poly[i] + (poly[j] - poly[i]) * (d[i] / (d[i] - d[j])))
    return np.array(out) if out else np.zeros((0, 2))


def _inset_convex(poly, distance):
    """Convex polygon moved inward by ``distance`` on every side."""
    poly = _clean_loop(poly, True)
    out = poly.copy()
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = (b - a) / np.linalg.norm(b - a)
        outward = np.array([e[1], -e[0]])
        out = _clip_halfplane(out, outward, float(outward @ a) - distance)
        if len(out) < 3:
            return np.zeros((0, 2))
    return out


def _round_corners(poly, radius, steps=3):
    """Polygon with every corner replaced by a fillet of ``steps`` points."""
    poly = _clean_loop(poly, True)
    n = len(poly)
    out = []
    for i in range(n):
        p, v, q = poly[i - 1], poly[i], poly[(i + 1) % n]
        e0, e1 = v - p, q - v
        l0, l1 = np.linalg.norm(e0), np.linalg.norm(e1)
        turn = math.acos(float(np.clip(e0 @ e1 / (l0 * l1), -1.0, 1.0)))
        d = min(radius * math.tan(0.5 * turn), 0.45 * l0, 0.45 * l1)
        a, b = v - e0 / l0 * d, v + e1 / l1 * d
        out += [(1.0 - t) ** 2 * a + 2.0 * t * (1.0 - t) * v + t * t * b for t in np.linspace(0.0, 1.0, steps)]
    return np.array(out)


def _convex_hull(points):
    """Convex hull of 2-D points, counterclockwise (monotone chain)."""
    pts = sorted({(round(float(p[0]), 12), round(float(p[1]), 12)) for p in points})

    def half(seq):
        h = []
        for p in seq:
            while (
                len(h) >= 2
                and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0.0
            ):
                h.pop()
            h.append(p)
        return h[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def _rounded_rect(lo, hi, radius):
    return _round_corners(np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]), radius)


def _inside_convex(poly, p):
    e = np.roll(poly, -1, axis=0) - poly
    d = p - poly
    return bool(np.all(e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0] >= 0.0))


# ----------------------------------------------------------------------------- triangulation
def _segments_block(a, b, edges):
    """True if the open segment ``a``-``b`` touches any of the edges (edges that share an end point are skipped)."""
    if len(edges) == 0:
        return False
    c, d = edges[:, 0], edges[:, 1]
    tol = 1.0e-10
    shares = (
        (np.abs(c - a).max(axis=1) < tol)
        | (np.abs(c - b).max(axis=1) < tol)
        | (np.abs(d - a).max(axis=1) < tol)
        | (np.abs(d - b).max(axis=1) < tol)
    )

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    hit = (o1 * o2 <= 0.0) & (o3 * o4 <= 0.0) & ~shares
    return bool(hit.any())


def _point_in_ring(pts, p):
    x, y = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = (y > p[1]) != (yn > p[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x + (p[1] - y) * (xn - x) / (yn - y)
    return bool(np.count_nonzero(cross & (p[0] < xi)) % 2)


def _triangulate(loops):
    """Triangulate a polygon with holes by ear clipping.

    Args:
        loops: Outer loop (counterclockwise) followed by hole loops (clockwise).

    Returns:
        Triangles as indices into the concatenated loop points, shape ``(m, 3)``.
    """
    pts = np.vstack(loops)
    offsets = np.cumsum([0] + [len(lp) for lp in loops])
    ring = list(range(offsets[0], offsets[1]))
    holes = [list(range(offsets[i], offsets[i + 1])) for i in range(1, len(loops))]
    holes.sort(key=lambda h: (-float(pts[h, 0].max()), h[0]))
    for n_done, hole in enumerate(holes):
        edges = [np.stack([pts[ring], pts[ring[1:] + ring[:1]]], axis=1)]
        edges += [np.stack([pts[h], pts[h[1:] + h[:1]]], axis=1) for h in holes[n_done:]]
        edges = np.vstack(edges)
        dist = ((pts[hole][:, None, :] - pts[ring][None, :, :]) ** 2).sum(axis=-1)
        chosen = None
        for flat in np.argsort(dist, axis=None, kind="stable"):
            hi, ri = divmod(int(flat), len(ring))
            a, b = pts[hole[hi]], pts[ring[ri]]
            if _segments_block(a, b, edges):
                continue
            if not _point_in_ring(pts[ring], b + 1.0e-3 * (a - b)):
                continue
            chosen = (hi, ri)
            break
        if chosen is None:
            raise ValueError("no bridge found for a hole")
        hi, ri = chosen
        ring = ring[: ri + 1] + hole[hi:] + hole[: hi + 1] + ring[ri:]

    tris = []
    k = 0
    stall = 0
    while len(ring) > 3:
        n = len(ring)
        k %= n
        ia, ib, ic = ring[k - 1], ring[k], ring[(k + 1) % n]
        a, b, c = pts[ia], pts[ib], pts[ic]
        cr = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        ear = cr > 1.0e-16
        if ear:
            p = pts[ring]
            same = (
                (np.abs(p - a).max(axis=1) < 1.0e-10)
                | (np.abs(p - b).max(axis=1) < 1.0e-10)
                | (np.abs(p - c).max(axis=1) < 1.0e-10)
            )
            e1 = (b[0] - a[0]) * (p[:, 1] - a[1]) - (b[1] - a[1]) * (p[:, 0] - a[0])
            e2 = (c[0] - b[0]) * (p[:, 1] - b[1]) - (c[1] - b[1]) * (p[:, 0] - b[0])
            e3 = (a[0] - c[0]) * (p[:, 1] - c[1]) - (a[1] - c[1]) * (p[:, 0] - c[0])
            ear = not bool(((e1 >= -1.0e-16) & (e2 >= -1.0e-16) & (e3 >= -1.0e-16) & ~same).any())
        if ear or stall > n:
            if cr > 1.0e-16:
                tris.append((ia, ib, ic))
            del ring[k]
            stall = 0
        else:
            k += 1
            stall += 1
    tris.append(tuple(ring))
    return np.array(tris, dtype=np.int64)


# ----------------------------------------------------------------------------- mesh primitives
def _miter(loop):
    """Per-point offset vectors that move a loop into the material on its left by a unit distance."""
    e = np.roll(loop, -1, axis=0) - loop
    e /= np.linalg.norm(e, axis=1)[:, None]
    left = np.column_stack([-e[:, 1], e[:, 0]])
    prev = np.roll(left, 1, axis=0)
    den = np.maximum(1.0 + np.sum(left * prev, axis=1), 0.4)
    return (left + prev) / den[:, None]


def _sharp(loop, limit=0.5236):
    e = np.roll(loop, -1, axis=0) - loop
    e /= np.linalg.norm(e, axis=1)[:, None]
    prev = np.roll(e, 1, axis=0)
    return np.arccos(np.clip(np.sum(e * prev, axis=1), -1.0, 1.0)) > limit


def _extrude(outer, z0, z1, holes=(), bevel0=0.0, bevel1=0.0, caps=True):
    """Prism of a polygon in the xy plane between ``z0`` and ``z1``.

    Smooth runs of the outline share wall vertices; corners sharper than 30 degrees
    and the edges of the two end faces are split, so the prism shades as machined.

    Args:
        outer: Outline, shape ``(n, 2)``.
        z0: Lower end [m].
        z1: Upper end [m].
        holes: Outlines of through holes.
        bevel0: Edge break at the lower end [m].
        bevel1: Edge break at the upper end [m].
        caps: Whether to close the two ends.

    Returns:
        Tuple of vertices ``(n, 3)`` and triangles ``(m, 3)``.
    """
    loops = [_clean_loop(outer, True)] + [_clean_loop(h, False) for h in holes]
    pts = np.vstack(loops)
    inset = np.vstack([_miter(lp) for lp in loops])
    levels = [(z0, bevel0), (z0 + bevel0, 0.0)] if bevel0 > 0.0 else [(z0, 0.0)]
    levels += [(z1 - bevel1, 0.0), (z1, bevel1)] if bevel1 > 0.0 else [(z1, 0.0)]
    meshes = []

    def ring(level, index=None):
        z, c = level
        xy = pts + c * inset
        if index is not None:
            xy = xy[index]
        return np.column_stack([xy, np.full(len(xy), z)])

    if caps:
        tri = _triangulate(loops)
        meshes.append((ring(levels[0]), tri[:, ::-1]))
        meshes.append((ring(levels[-1]), tri))
    offset = 0
    for lp in loops:
        n = len(lp)
        sharp = _sharp(lp)
        index, start, end = [], [], []
        for i in range(n):
            end.append(len(index))
            index.append(offset + i)
            if sharp[i]:
                index.append(offset + i)
            start.append(len(index) - 1)
        m = len(index)
        quads = []
        for i in range(n):
            a, b = start[i], end[(i + 1) % n]
            quads += [(a, b, m + b), (a, m + b, m + a)]
        quads = np.array(quads, dtype=np.int64)
        for lo, hi in itertools.pairwise(levels):
            meshes.append((np.vstack([ring(lo, index), ring(hi, index)]), quads))
        offset += n
    return _merge(meshes)


def _box(lo, hi, bevel=0.0):
    """Axis-aligned box with broken edges on its two z faces."""
    rect = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    return _extrude(rect, lo[2], hi[2], bevel0=bevel, bevel1=bevel)


def _lathe(profile, segments=_SEG):
    """Body of revolution about z from a profile of ``(radius, z)`` points.

    The profile runs counterclockwise in the ``(radius, z)`` half plane, so that
    the outside lies on its right. Every profile corner is a hard edge.
    """
    profile = np.asarray(profile, dtype=float)
    ang = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    cs = np.column_stack([np.cos(ang), np.sin(ang)])
    nxt = np.roll(np.arange(segments), -1)
    idx = np.arange(segments)
    meshes = []
    for (r1, z1), (r2, z2) in itertools.pairwise(profile):
        lo = np.column_stack([r1 * cs, np.full(segments, z1)])
        hi = np.column_stack([r2 * cs, np.full(segments, z2)])
        tris = []
        if r2 > 1.0e-12:
            tris.append(np.column_stack([idx, segments + nxt, segments + idx]))
        if r1 > 1.0e-12:
            tris.append(np.column_stack([idx, nxt, segments + nxt]))
        if tris:
            meshes.append((np.vstack([lo, hi]), np.vstack(tris)))
    return _merge(meshes)


def _rod(p0, p1, radius, segments=16):
    """Closed cylinder between two points."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    h = float(np.linalg.norm(p1 - p0))
    mesh = _lathe([(0.0, 0.0), (radius, 0.0), (radius, h), (0.0, h)], segments)
    return _moved(mesh, _frame_from_z(p1 - p0), p0)


def _loft(rings, start_cap=True, end_point=None):
    """Smooth skin over rings of equal point count that advance along +x.

    Ring points run counterclockwise about +x (from +y toward +z).
    """
    rings = [np.asarray(r, dtype=float) for r in rings]
    k = len(rings[0])
    nxt = np.roll(np.arange(k), -1)
    idx = np.arange(k)
    verts = np.vstack(rings)
    tris = []
    for i in range(len(rings) - 1):
        a, b = i * k, (i + 1) * k
        tris.append(np.column_stack([a + idx, a + nxt, b + nxt]))
        tris.append(np.column_stack([a + idx, b + nxt, b + idx]))
    meshes = []
    if end_point is not None:
        a = (len(rings) - 1) * k
        verts = np.vstack([verts, np.asarray(end_point, dtype=float)[None, :]])
        tris.append(np.column_stack([a + idx, a + nxt, np.full(k, len(verts) - 1)]))
    meshes.append((verts, np.vstack(tris)))
    if start_cap:
        cap = np.vstack([rings[0].mean(axis=0)[None, :], rings[0]])
        meshes.append((cap, np.column_stack([np.zeros(k, dtype=np.int64), 1 + nxt, 1 + idx])))
    return _merge(meshes)


def _pad_lens(x0, x1, y_axis, radius, y_cut, across=8, along=4):
    """Part of a capsule (axis along x at height ``y_axis``) that lies beyond the plane ``y = y_cut``."""
    h = y_cut - y_axis
    if not 0.0 < h < radius:
        raise ValueError("pad plane does not cut the capsule")
    reach = math.sqrt(radius * radius - h * h)
    beta = np.linspace(0.0, 0.5 * math.pi, along + 1)[1:]
    xs = np.concatenate([x0 - reach * np.cos(beta), x1 + reach * np.cos(beta[::-1])])
    rings, edge = [], []
    for x in xs:
        dx = max(x0 - x, x - x1, 0.0)
        rho = math.sqrt(radius * radius - dx * dx)
        th = math.acos(min(1.0, h / rho))
        phi = np.linspace(-th, th, across + 1)
        rings.append(np.column_stack([np.full(across + 1, x), y_axis + rho * np.cos(phi), rho * np.sin(phi)]))
        edge.append(rho * math.sin(th))
    n, k = len(rings), across + 1
    verts = np.vstack([*rings, [[x0 - reach, y_cut, 0.0]], [[x1 + reach, y_cut, 0.0]]])
    tris = []
    for i in range(n - 1):
        for j in range(across):
            a, b = i * k + j, (i + 1) * k + j
            tris += [(a, a + 1, b + 1), (a, b + 1, b)]
    p0, p1 = n * k, n * k + 1
    for j in range(across):
        tris += [(p0, j + 1, j), ((n - 1) * k + j, (n - 1) * k + j + 1, p1)]
    # flat back face
    back = np.array(
        [[x0 - reach, y_cut, 0.0]]
        + [[x, y_cut, s * e] for x, e in zip(xs, edge, strict=True) for s in (-1, 1)]
        + [[x1 + reach, y_cut, 0.0]]
    )
    bt = [(0, 1, 2), (2 * n + 1, 2 * n, 2 * n - 1)]
    for i in range(n - 1):
        a = 1 + 2 * i
        bt += [(a, a + 2, a + 3), (a, a + 3, a + 1)]
    return _merge([(verts, np.array(tris)), (back, np.array(bt))])


# ----------------------------------------------------------------------------- spec helpers
def _flat_spec(spec):
    """Specification in which every digit also offers ``stations`` and ``slider`` of its first (long) cable.

    The parts that follow the long cable (link windows, knuckle cheeks, slots) read these two keys.
    """
    if all("stations" in digit for digit in spec["digits"].values()):
        return spec
    digits = {}
    for name, digit in spec["digits"].items():
        if "stations" in digit:
            digits[name] = digit
        else:
            first = digit["cables"][0]
            digits[name] = {**digit, "stations": first["stations"], "slider": first["slider"]}
    return {**spec, "digits": digits}


def _digit_cables(spec, name):
    """Cables of a digit, long cable first.

    Every entry has ``name``, ``plane_z`` (offset of the cable plane from the digit mid-plane), ``stations``,
    ``slider``, the body key of its carriage (``"<digit>/slider"`` for the first cable,
    ``"<digit>/slider_<cable name>"`` for every further one) and, for a finger, the envelope of its linear
    unit (``box``, ``front_face_x``, ``rear_face_x``, ``dorsal_face_y``, ``z_center`` in the digit frame).
    """
    digit, act = spec["digits"][name], spec["actuator"]
    out = []
    for k, cable in enumerate(digit["cables"]):
        unit = cable.get("actuator_unit")
        if unit is None and name in FINGERS:
            anchor = cable["slider"]["rest_anchor"]
            front = anchor[0] + act["finger_unit_front_offset"]
            unit = {
                "box": list(act["finger_unit_box"]),
                "front_face_x": front,
                "rear_face_x": front - act["finger_unit_box"][0],
                "dorsal_face_y": anchor[1] - act["finger_unit_cable_line_above_bottom"],
                "z_center": 0.0,
            }
        out.append(
            {
                "name": cable.get("name", "flexor"),
                "plane_z": cable.get("plane_z", 0.0),
                "stations": cable["stations"],
                "slider": cable["slider"],
                "unit": unit,
                "body": f"{name}/slider" if k == 0 else f"{name}/slider_{cable['name']}",
            }
        )
    return out


def _digit_frame(digit, phi=0.0):
    """Origin and axes (columns) of a digit frame in the palm frame."""
    m = digit["mount"]
    if "hinge_axis" not in m:
        return np.array(m["frame_origin"], dtype=float), np.column_stack([m["frame_x"], m["frame_y"], m["frame_z"]])
    rot = _rot_axis(m["hinge_axis"], phi)
    hp = np.array(m["hinge_point"], dtype=float)
    origin = hp + rot @ (np.array(m["frame_origin_at_phi0"]) - hp)
    return origin, rot @ np.column_stack([m["frame_x_at_phi0"], m["frame_y_at_phi0"], m["frame_z_at_phi0"]])


def _station(digit, name):
    for st in digit["stations"]:
        if st["name"] == name:
            return st
    return None


def _link_polygon(lk):
    """Convex section of a link in its flexion plane, counterclockwise."""
    length, yd, yp = lk["shaft_end"], lk["y_dorsal"], lk["y_palmar"]
    cp, cd = lk["palmar_chamfer_proximal"], lk["palmar_chamfer_distal"]
    return np.array([[0.0, yd], [length, yd], [length, 0.0], [length - cd, yp], [cp, yp], [0.0, 0.0]])


def _windows(poly, x_lo, x_hi, guide_x=None):
    """Window outlines in a side plate: the inset section, split by a strut at the guide pin."""
    base = _inset_convex(poly, _STRUT)
    spans = [(x_lo, x_hi)]
    if guide_x is not None:
        spans = [(x_lo, guide_x - 0.5 * _STRUT - 0.5e-3), (guide_x + 0.5 * _STRUT + 0.5e-3, x_hi)]
    out = []
    for lo, hi in spans:
        if hi - lo < 2.5e-3 or len(base) < 3:
            continue
        w = _clip_halfplane(_clip_halfplane(base, (-1.0, 0.0), -lo), (1.0, 0.0), hi)
        if len(w) >= 3 and abs(_area(w)) > 8.0e-6:
            out.append(_round_corners(w, 1.2e-3))
    return out


def _boss_radius(knuckle_radius):
    return _BOSS * knuckle_radius


# ----------------------------------------------------------------------------- pulleys, pins, carriages, actuators
def build_pulley_mesh(spec: dict, radius: float):
    """Build a flanged, grooved sheave as a body of revolution about its z axis.

    Args:
        spec: Hand specification.
        radius: Radius of the cable center line on the sheave [m].

    Returns:
        Tuple of vertices ``(n, 3)`` and triangles ``(m, 3)``.
    """
    hw = spec["station_hardware"]
    rc = spec["cable"]["radius"]
    rf = radius + hw["flange_height_above_centerline"]  # flange rim
    rb = radius - rc  # groove bottom
    gw = hw["groove_half_width"]
    ht = gw + hw["flange_thickness"]
    rh = max(1.6 * hw["pin_radius"], 0.3 * radius)  # hub
    ri = rb - 0.7e-3  # inner radius of the rim
    web = 0.6 * ht
    profile = [
        (0.0, -_HUB),
        (rh, -_HUB),
        (rh, -web),
        (ri, -web),
        (ri + 0.3e-3, -ht),
        (rf, -ht),
        (rf, -gw),
        (rb, -0.9 * rc),
        (rb, 0.9 * rc),
        (rf, gw),
        (rf, ht),
        (ri + 0.3e-3, ht),
        (ri, web),
        (rh, web),
        (rh, _HUB),
        (0.0, _HUB),
    ]
    return _lathe(profile, 28 if radius > 6.0e-3 else _SEG)


def build_finger_actuator_mesh(spec: dict):
    """Build the folded linear actuator of a finger.

    The mesh frame sits on the cable line at the front face of the unit: +x toward
    the finger, +y palmar, the unit extends along -x.
    """
    act = spec["actuator"]
    lx, ly, lz = act["finger_unit_box"]
    cl = act["finger_unit_cable_line_above_bottom"]
    travel = act["stroke"] + 2.0 * act["carriage_half_length"] + act["finger_unit_front_offset"] + 2.0e-3
    channel = 2.0 * cl + 1.0e-3  # depth of the open carriage channel
    half = 0.5 * lz
    rail = 0.5 * lz - _SLOT + 0.4e-3
    parts = [
        _box((-lx, -cl + channel, -half), (0.0, -cl + ly, half), _BEVEL),  # motor and gear body
        _box((-lx, -cl, -half), (-travel, -cl + channel, half), _BEVEL),  # rear block
        _box((-travel, -cl, half - rail), (0.0, -cl + channel, half)),
        _box((-travel, -cl, -half), (0.0, -cl + channel, -half + rail)),
        _rod((-travel, 0.6 * channel - cl + 0.4e-3, 0.0), (0.0, 0.6 * channel - cl + 0.4e-3, 0.0), 0.9e-3),  # screw
    ]
    return _merge(parts)


def build_carriage_mesh(spec: dict, round_body: bool = False):
    """Build an actuator carriage with its cable ferrule.

    The mesh frame sits on the cable anchor. A finger carriage is a block with +x
    toward the finger; the thumb carriage (``round_body``) is a nut about its z
    axis, which points toward the thumb.
    """
    act = spec["actuator"]
    hl = act["carriage_half_length"]
    cl = act["finger_unit_cable_line_above_bottom"]
    if round_body:
        r = 0.45 * act["thumb_unit_cylinder"]["diameter"]
        return _lathe([(0.0, -hl), (r, -hl), (r, hl), (0.9e-3, hl), (0.9e-3, hl + 1.5e-3), (0.0, hl + 1.5e-3)], 16)
    block = _box((-hl, -cl + 0.6e-3, -(_SLOT - 0.9e-3)), (hl, cl + 0.5e-3, _SLOT - 0.9e-3), 0.2e-3)
    return _merge([block, _rod((hl, 0.0, 0.0), (hl + 1.5e-3, 0.0, 0.0), 0.9e-3)])


def build_thumb_actuator_mesh(spec: dict):
    """Build the in-line thumb actuator about its z axis.

    The mesh frame sits on the hinge point with +z along the slider line toward the
    thumb; the unit lies at negative z.
    """
    act = spec["actuator"]
    r = 0.5 * act["thumb_unit_cylinder"]["diameter"]
    front = -act["thumb_unit_front_behind_hinge_point"]
    rear = front - act["thumb_unit_cylinder"]["length"]
    motor = rear + act["thumb_unit_motor_length"]
    b = _BEVEL
    parts = [
        _lathe([(0.0, rear), (r - b, rear), (r, rear + b), (r, motor - b), (r - b, motor), (0.0, motor)], 20),
        _lathe(
            [(2.0e-3, front - 1.5e-3), (r, front - 1.5e-3), (r, front), (2.0e-3, front), (2.0e-3, front - 1.5e-3)], 20
        ),
    ]
    for s in (-1.0, 1.0):
        parts.append(_rod((s * 0.7 * r, 0.0, motor), (s * 0.7 * r, 0.0, front - 1.5e-3), 1.0e-3))
    return _merge(parts)


# ----------------------------------------------------------------------------- links
def _plate_outline(lk, rb_prox, rb_dist, bite):
    """Outline of a side plate: the link section with an axle boss at the distal joint and either an axle
    boss (``bite`` false) or a clearance cut around the neighbor's boss (``bite`` true) at the proximal joint."""
    length, yd, yp = lk["length"], lk["y_dorsal"], lk["y_palmar"]
    cp, cd = lk["palmar_chamfer_proximal"], lk["palmar_chamfer_distal"]
    pts = [np.array([[_GAP, yd], [length - _GAP, yd]])]
    pts.append(_arc((length, 0.0), rb_dist, -(0.5 * math.pi + math.asin(_GAP / rb_dist)), math.atan2(yp, -cd)))
    pts.append(np.array([[length - cd, yp], [cp, yp]]))
    ap = math.atan2(yp, cp)
    end = -math.acos(_GAP / rb_prox) if bite else 2.0 * math.pi - math.acos(_GAP / rb_prox)
    pts.append(_arc((0.0, 0.0), rb_prox, ap, end))
    return np.vstack(pts)


def _neck_outline(lk, rb_prox, x_end):
    """Outline of an inner ear: the proximal axle boss and the link section up to ``x_end``."""
    yd, yp, cp = lk["y_dorsal"], lk["y_palmar"], lk["palmar_chamfer_proximal"]
    pts = [np.array([[_GAP, yd], [x_end, yd], [x_end, yp if x_end >= cp else yp * x_end / cp]])]
    if x_end > cp:
        pts.append(np.array([[cp, yp]]))
    pts.append(_arc((0.0, 0.0), rb_prox, math.atan2(yp, cp), 2.0 * math.pi - math.acos(_GAP / rb_prox)))
    return np.vstack(pts)


def _ear_layers(parent_width):
    """z range of the inner ears of a link whose parent has the given width."""
    z1 = 0.5 * parent_width - _PLATE - _RUN
    return z1 - _PLATE, z1


def _nose_start(digit):
    """x of the rear face of the fingertip: just beyond the tip anchor."""
    lk = digit["links"][2]
    return max(_station(digit, "T")["pos"][0] + 0.3e-3, lk["palmar_chamfer_proximal"])


def _roof(x0, x1, y0, half_width, thickness):
    """Dorsal cover strip of a link with a slot above the cable."""
    rect = np.array([[-half_width, x0], [half_width, x0], [half_width, x1], [-half_width, x1]])
    holes = []
    if x1 - x0 > 9.0e-3:
        holes.append(_rounded_rect((-1.6e-3, x0 + 2.5e-3), (1.6e-3, x1 - 2.5e-3), 1.5e-3))
    return _planform(_extrude(rect, y0, y0 + thickness, holes=holes))


def build_link_meshes(spec: dict, digit_name: str, link_index: int) -> dict[str, tuple]:
    """Build the meshes of one link in the link frame.

    Args:
        spec: Hand specification.
        digit_name: Digit name, a key of ``spec["digits"]``.
        link_index: 0 for the proximal, 1 for the middle, 2 for the distal link.

    Returns:
        Mapping from part name to ``(vertices, triangles, material)``. The inner ears of
        the middle and distal link come as ``ears``; they are identical for digits with
        the same link section.
    """
    spec = _flat_spec(spec)
    digit = spec["digits"][digit_name]
    links = digit["links"]
    lk = links[link_index]
    hw = spec["station_hardware"]
    width = lk["width"]
    zo0 = 0.5 * width - _PLATE
    out = {}
    rbp = _boss_radius(lk["knuckle_radius_proximal"])
    pad = digit["pads"][link_index]
    y_cut = lk["y_palmar"] - _PAD_SINK
    out["pad"] = (*_pad_lens(pad["p0"][0], pad["p1"][0], pad["p0"][1], pad["radius"], y_cut), "pad")
    if link_index > 0:
        zi0, zi1 = _ear_layers(links[link_index - 1]["width"])
        bite = _boss_radius(links[link_index - 1]["knuckle_radius_distal"]) + _BITE
        x_neck = _nose_start(digit) + 1.5e-3 if link_index == 2 else bite + 3.0e-3
        ears = _extrude(_neck_outline(lk, rbp, x_neck), zi0, zi1)
        out["ears"] = (*_both_sides(ears), "frame")

    if link_index == 2:
        # fingertip: a rounded nose behind which the cable ends
        xs = _nose_start(digit)
        x_end, r_tip = lk["distal_disk_x"], lk["knuckle_radius_distal"]
        yc, hy, hz = 0.5 * (lk["y_dorsal"] + lk["y_palmar"]), 0.5 * (lk["y_palmar"] - lk["y_dorsal"]), 0.5 * width

        def section(x, sy, sz):
            rc = 0.3 * min(sy, sz)
            pts = []
            for cy, cz, a0 in ((1, 1, 0.0), (-1, 1, 0.5), (-1, -1, 1.0), (1, -1, 1.5)):
                a = math.pi * (a0 + np.linspace(0.0, 0.5, 7))
                pts.append(
                    np.column_stack(
                        [np.full(7, x), yc + cy * (sy - rc) + rc * np.cos(a), cz * (sz - rc) + rc * np.sin(a)]
                    )
                )
            return np.vstack(pts)

        rings = [section(xs, hy, hz), section(x_end, hy, hz)]
        for t in np.linspace(0.0, 0.5 * math.pi, 12)[1:-1]:
            rings.append(section(x_end + r_tip * math.sin(t), hy * math.cos(t), hz * math.cos(t) ** 0.7))
        tip = _loft(rings, end_point=(x_end + r_tip, yc, 0.0))
        # lip that carries the palmar face back to the chamfer, between and through the inner ears
        yp, cp = lk["y_palmar"], lk["palmar_chamfer_proximal"]
        r_sheave = digit["joints"][2]["sheave_radius"] + hw["flange_height_above_centerline"] + 0.4e-3
        x_lip = max(cp, math.sqrt(max(r_sheave**2 - (yp - _FLOOR) ** 2, 0.0))) + 0.3e-3
        if xs - x_lip > 1.0e-3:
            lip = np.array([[x_lip, yp - _FLOOR], [xs, yp - _FLOOR], [xs, yp], [x_lip, yp]])
            tip = _merge([tip, _extrude(lip, -zi1, zi1)])
        out["tip"] = (*tip, "tip")
        anchor = _station(digit, "T")["pos"]
        out["anchor"] = (*_rod((anchor[0] - 1.5e-3, anchor[1], 0.0), (xs, anchor[1], 0.0), 0.9e-3), "pin")
        return out

    length = lk["length"]
    rbd = _boss_radius(lk["knuckle_radius_distal"])
    guide = next(st for st in digit["stations"] if st["body"] == f"link{link_index + 1}" and st["kind"] == "guide")
    if link_index == 0:
        outline = _plate_outline(lk, rbp, rbd, bite=False)
        x_lo = rbp + 1.8e-3
    else:
        outline = _plate_outline(lk, bite, rbd, bite=True)
        x_lo = bite + 1.8e-3
    windows = _windows(_link_polygon(lk), x_lo, length - rbd - 1.8e-3, guide["pos"][0])
    plate = _extrude(outline, zo0, 0.5 * width, holes=windows, bevel1=_BEVEL)
    out["frame"] = (*_both_sides(plate), "frame")

    # covers: slotted roof on the dorsal side, floor under the pad
    yd, yp = lk["y_dorsal"], lk["y_palmar"]
    r_sheave = digit["joints"][link_index]["sheave_radius"] + hw["flange_height_above_centerline"] + 0.4e-3
    inner = abs(yd) - hw["shell_wall"]
    x_roof = math.sqrt(max(r_sheave * r_sheave - inner * inner, 0.0)) + 1.0e-3
    roof = _roof(x_roof, length - _GAP, yd, zo0, hw["shell_wall"])
    cp, cd = lk["palmar_chamfer_proximal"], lk["palmar_chamfer_distal"]
    k = (yp - _FLOOR) / yp
    floor = _extrude(
        np.array([[cp * k, yp - _FLOOR], [length - cd * k, yp - _FLOOR], [length - cd, yp], [cp, yp]]), -zo0, zo0
    )
    out["cover"] = (*_merge([roof, floor]), "cover")

    # pins: the guide pin and the axle of the distal joint
    half = 0.5 * width + 0.3e-3
    pins = [
        _rod((guide["pos"][0], guide["pos"][1], -half), (guide["pos"][0], guide["pos"][1], half), _PIN),
        _rod((length, 0.0, -half), (length, 0.0, half), _PIN),
    ]
    out["pins"] = (*_merge(pins), "pin")
    return out


# ----------------------------------------------------------------------------- thumb column
def build_thumb_column_meshes(spec: dict) -> dict[str, tuple]:
    """Build the thumb column (the yoke that carries the thumb MCP joint) in the thumb digit frame.

    Returns:
        Mapping from part name to ``(vertices, triangles, material)``.
    """
    thumb = spec["digits"]["thumb"]
    lk = thumb["links"][0]
    col = spec["palm"]["thumb_column"]
    sl = thumb["slider"]
    tp = np.array(sl["line_tangent_point"], dtype=float)  # where the cable meets the first sheave
    along = -np.array(sl["pull_direction"], dtype=float)  # hinge line direction, toward the thumb
    side = np.array([along[1], -along[0]])
    rc = col["radius"]
    half = 0.5 * lk["width"]
    back = col["root_behind_hinge_point"] - col["root_running_clearance"]
    sweep = max(abs(lk["y_dorsal"]), lk["knuckle_radius_proximal"]) + 0.4e-3  # reach of link 1 about the MCP axis

    def to_frame(pts):
        return tp[None, :] + pts[:, :1] * along[None, :] + pts[:, 1:] * side[None, :]

    # rear block between the hinge bearing and the MCP joint, in (along, across) coordinates of the hinge line
    flange = 1.2e-3
    rect = to_frame(np.array([[-back + flange, -rc], [0.0, -rc], [0.0, rc], [-back + flange, rc]]))
    rect = _clean_loop(rect, True)
    # the block is the part of the rectangle outside the circle swept by link 1
    samples = []
    for i in range(len(rect)):
        a, b = rect[i], rect[(i + 1) % len(rect)]
        for t in np.linspace(0.0, 1.0, 40, endpoint=False):
            samples.append(a + t * (b - a))
    samples = np.array(samples)
    outside = np.linalg.norm(samples, axis=1) >= sweep
    # walk the rectangle; replace the run inside the circle by the arc between its two crossings
    first = int(np.argmax(outside & ~np.roll(outside, 1)))  # first sample after re-entering the outside
    order = np.roll(np.arange(len(samples)), -first)
    kept = samples[order][outside[order]]
    a0 = math.atan2(kept[-1][1], kept[-1][0])
    a1 = math.atan2(kept[0][1], kept[0][0])
    while a1 > a0:
        a1 -= 2.0 * math.pi
    block = np.vstack([kept, _arc((0.0, 0.0), sweep, a0, a1)[1:-1]])
    slot = 1.2e-3
    body = _extrude(block, slot, half, bevel1=_BEVEL)
    # cheeks inside the ears of link 1
    zi0, zi1 = _ear_layers(lk["width"])
    rb = _boss_radius(lk["knuckle_radius_proximal"])
    hull = _convex_hull(np.vstack([_arc((0.0, 0.0), rb, 0.0, 2.0 * math.pi)[:-1], kept[:: max(1, len(kept) // 12)]]))
    cheek = _extrude(hull, zi0, zi1)
    # bearing flange about the hinge line
    ring = _lathe([(2.0e-3, 0.0), (rc + 0.5e-3, 0.0), (rc + 0.5e-3, flange), (2.0e-3, flange), (2.0e-3, 0.0)], _SEG)
    rot = np.column_stack([[side[0], side[1], 0.0], [0.0, 0.0, 1.0], [along[0], along[1], 0.0]])
    if np.linalg.det(rot) < 0.0:
        rot[:, 0] *= -1.0
    origin = tp - back * along
    ring = _moved(ring, rot, (origin[0], origin[1], 0.0))
    out = {"frame": (*_merge([_both_sides(body), _both_sides(cheek), ring]), "frame")}
    pin_half = half + 0.3e-3
    out["pins"] = (*_rod((0.0, 0.0, -pin_half), (0.0, 0.0, pin_half), _PIN), "pin")
    return out


# ----------------------------------------------------------------------------- palm
def _column_range(spec, name):
    """x and z range of a finger column in its digit frame; z is widened to hold the knuckle cheeks."""
    box = next(b for b in spec["palm"]["solid_boxes"] if b["name"] == f"{name}_column")
    lk = spec["digits"][name]["links"][0]
    need = 0.5 * lk["width"] + _RUN + _CHEEK
    z0, z1 = box["z"][0] - box["pivot"][2], box["z"][1] - box["pivot"][2]
    return box["x"][0] - box["pivot"][0], min(z0, -need), max(z1, need)


def build_screw_mesh():
    """Build a button-head screw: the head sits on the plane z = 0 and rises along +z."""
    return _lathe([(1.7e-3, 0.0), (1.7e-3, 0.3e-3), (1.2e-3, 0.55e-3), (0.0, 0.55e-3)], 16)


def build_palm_meshes(spec: dict) -> dict[str, tuple]:
    """Build the palm with forearm stub and thenar housing in the palm frame.

    The palm is one modeled body with a heel, a wrist stem and a flange (see
    :func:`_modeled_palm`).

    Args:
        spec: Hand specification.

    Returns:
        Mapping from part name to ``(vertices, triangles, material)``.
    """
    return _modeled_palm(_flat_spec(spec))


def _palm_cheeks(spec):
    """Knuckle cheeks of the palm (section of the palm in the flexion plane of each finger, with an axle
    boss) and the pins of the stations they carry, as two lists of meshes in the palm frame."""
    palm = spec["palm"]
    y_dorsal, y_palmar = palm["thickness_y"]
    setback = palm["palmar_face_setback"]
    frame = []
    pins = []
    y_top, y_bot = y_dorsal + _COVER, y_palmar - _COVER
    for name in FINGERS:
        digit = spec["digits"][name]
        origin, axes = _digit_frame(digit)
        lk = digit["links"][0]
        _, z0, z1 = _column_range(spec, name)
        rb = _boss_radius(lk["knuckle_radius_proximal"])
        inner = 0.5 * lk["width"] + _RUN
        x_chamfer = setback * y_bot / y_palmar
        section = np.array(
            [[-_CHEEK_BACK, y_top], [0.0, y_top], [0.0, 0.0], [-x_chamfer, y_bot], [-_CHEEK_BACK, y_bot]]
        )
        outline = np.vstack(
            [
                np.array([[-_CHEEK_BACK, y_top], [-_GAP, y_top]]),
                _arc((0.0, 0.0), rb, -(0.5 * math.pi + math.asin(_GAP / rb)), math.atan2(y_palmar, -setback)),
                np.array([[-x_chamfer, y_bot], [-_CHEEK_BACK, y_bot]]),
            ]
        )
        g0 = _station(digit, "G0")
        windows = _windows(section, -_CHEEK_BACK + 0.5e-3, -rb - 1.8e-3, g0["pos"][0] if g0 else None)
        for lo, hi, bevel in ((inner, z1, True), (z0, -inner, False)):
            cheek = _extrude(
                outline, lo, hi, holes=windows, bevel1=_BEVEL if bevel else 0.0, bevel0=0.0 if bevel else _BEVEL
            )
            frame.append(_moved(cheek, axes, origin))
        for st in (g0, _station(digit, "S1")):
            if st is None:
                continue
            p = origin + axes[:, 0] * st["pos"][0] + axes[:, 1] * st["pos"][1]
            pins.append(_rod(p + axes[:, 2] * (z0 - 0.3e-3), p + axes[:, 2] * (z1 + 0.3e-3), _PIN))
    return frame, pins


def _palm_pads_and_motor(spec, out):
    """Add the two palm pads and the opposition motor to a mapping of palm parts."""
    palm = spec["palm"]
    act = spec["actuator"]
    y_palmar = palm["thickness_y"][1]
    # palm pads: capsules across the palm, let into the palmar cover
    for pad in palm["pads"]:
        p0, p1 = np.array(pad["p0"], dtype=float), np.array(pad["p1"], dtype=float)
        lens = _pad_lens(min(p0[2], p1[2]), max(p0[2], p1[2]), p0[1], pad["radius"], y_palmar - _PAD_SINK)
        v, t = lens
        v = np.column_stack([p0[0] - v[:, 2], v[:, 1], v[:, 0]])
        if p1[1] != p0[1]:  # a pad may rise toward one end
            v[:, 1] += (p1[1] - p0[1]) * (v[:, 2] - p0[2]) / (p1[2] - p0[2])
        out[f"pad_{pad['name']}"] = (v, t, "pad")

    # opposition motor
    motor = act["opposition_motor"]
    half = 0.5 * np.array(motor["size"], dtype=float)
    # the motor lies along the tilted hinge line and carries its own axes
    turn = np.column_stack(motor["axes"]) if motor.get("axes") else _rot_axis((0.0, 1.0, 0.0), motor.get("yaw") or 0.0)
    out["opposition_motor"] = (*_moved(_box(-half, half, _BEVEL), turn, motor["center"]), "actuator")
    return out


def build_bracket_mesh(points, z_top: float, posts=(), margin: float = 2.5e-3, thickness: float = 2.0e-3):
    """Build a base bracket for a single digit on a test stand: a back plate with posts.

    The plate lies in the flexion plane behind the parts it carries. Its
    outline is the convex outline of ``points`` grown by ``margin``; every post
    is a cylinder that rises from the plate along +z.

    Args:
        points: Points ``(x, y)`` the plate must cover [m], shape ``(n, 2)``.
        z_top: Height of the plate's upper face [m].
        posts: One ``(x, y, z_end, radius)`` per post [m]; a post runs from the plate to ``z_end``.
        margin: Distance by which the plate outline is grown [m].
        thickness: Plate thickness [m].

    Returns:
        Tuple of vertices ``(n, 3)`` and triangles ``(m, 3)``.
    """
    hull = _convex_hull(np.asarray(points, dtype=float)[:, :2])
    ring = [margin * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)]
    outline = _convex_hull([p + d for p in hull for d in ring])
    meshes = [_extrude(outline, z_top - thickness, z_top, bevel0=_BEVEL, bevel1=_BEVEL)]
    for x, y, z_end, radius in posts:
        if z_end > z_top:
            meshes.append(_rod((x, y, z_top), (x, y, z_end), radius))
    return _merge(meshes)


# ----------------------------------------------------------------------------- curved shells: helpers
_FORM_NECK_R = 8.8e-3  # radius of the neck that the collar of the thumb column turns on
_FORM_SEAT = (-15.6e-3, -15.0e-3, -11.4e-3)  # along the hinge line: end of the lobe, seat face, end of the neck
_FORM_COLLAR = (9.4e-3, 11.8e-3, -14.5e-3, -10.8e-3)  # collar of the thumb column: radii and ends along the hinge line


def _smooth(t):
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _weld(verts, tris):
    """Merge coincident vertices, drop collapsed triangles and unused vertices."""
    verts = np.asarray(verts, dtype=float)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    _, first, inverse = np.unique(np.round(verts, 7), axis=0, return_index=True, return_inverse=True)
    tris = first[inverse.reshape(-1)][tris]
    tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    tris = tris[np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1.0e-12]
    used = np.unique(tris)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[tris]


def _grid_mesh(points, skip=(), wrap=False):
    """Smooth skin over a grid of points: rows advance along an axis, columns run counterclockwise about it.

    Args:
        points: Grid points, shape ``(rows, columns, 3)``.
        skip: Quads ``(row, column)`` to leave open.
        wrap: Whether the last column joins the first.
    """
    points = np.asarray(points, dtype=float)
    ni, nj = points.shape[:2]
    skip = set(skip)
    tris = []
    for i in range(ni - 1):
        for j in range(nj if wrap else nj - 1):
            if (i, j) in skip:
                continue
            a, b = i * nj + j, i * nj + (j + 1) % nj
            tris += [(a, b, b + nj), (a, b + nj, a + nj)]
    return _weld(points.reshape(-1, 3), tris)


def _hole_lip(points, i0, i1, j0, j1, offset):
    """Wall around the opening left by skipping the quads ``i0..i1-1`` x ``j0..j1-1`` of a grid; ``offset``
    points into the housing."""
    loop = [(i0, j) for j in range(j0, j1)] + [(i, j1) for i in range(i0, i1)]
    loop += [(i1, j) for j in range(j1, j0, -1)] + [(i, j0) for i in range(i1, i0, -1)]
    top = np.array([points[i][j] for i, j in loop], dtype=float)
    n = len(loop)
    verts = np.vstack([top, top + np.asarray(offset, dtype=float)])
    tris = []
    for k in range(n):
        k1 = (k + 1) % n
        tris += [(k, k1, n + k1), (k, n + k1, n + k)]
    tris = np.array(tris, dtype=np.int64)
    a, b, c = verts[tris[0, 0]], verts[tris[0, 1]], verts[tris[0, 2]]
    if np.cross(b - a, c - a) @ (top.mean(axis=0) - (a + b + c) / 3.0) < 0.0:
        tris = tris[:, ::-1]
    return _weld(verts, tris)


def _thumb_hinge(spec):
    """Hinge point, hinge line direction (toward the thumb) and a lateral axis of the thumb mount (palm frame)."""
    mount = spec["digits"]["thumb"]["mount"]
    hinge = np.array(mount["hinge_point"], dtype=float)
    line = np.array(mount["slider_line_direction"], dtype=float)
    side = np.cross(line, [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    return hinge, line, side


class _FingerFan:
    """The fan of the fingers in the palm plane: the lines that run along the fingers and the line of the
    knuckle axes, as functions of the palm coordinates."""

    def __init__(self, spec):
        self.frames = {name: _digit_frame(spec["digits"][name]) for name in FINGERS}
        self.reach = max(0.5 * spec["digits"][n]["links"][0]["width"] for n in FINGERS) + _RUN + _CHEEK
        zs = [float(self.frames[n][0][2]) for n in FINGERS]
        xs = [float(self.frames[n][0][0]) for n in FINGERS]
        order = np.argsort(zs)
        zs, xs = [zs[k] for k in order], [xs[k] for k in order]
        lo, hi = self.frames[FINGERS[-1]][1][:, 2], self.frames[FINGERS[0]][1][:, 2]
        self._knuckle = (
            [zs[0] - 0.1, *zs, zs[-1] + 0.1],
            [xs[0] - 0.1 * lo[0] / lo[2], *xs, xs[-1] + 0.1 * hi[0] / hi[2]],
        )

    def _fan_raw(self, name, offset, x):
        """z of the line parallel to a finger at a lateral offset, at the palm coordinate x."""
        origin, axes = self.frames[name]
        s = (x - origin[0] - offset * axes[0, 2]) / axes[0, 0]
        return float(origin[2] + s * axes[2, 0] + offset * axes[2, 2])

    def knuckle(self, z):
        """x of the line through the knuckle axes."""
        return float(np.interp(z, *self._knuckle))

    def palm_x(self, name, x_digit):
        """Palm x of a point on a finger axis given by its x in the digit frame."""
        origin, axes = self.frames[name]
        return float(origin[0] + x_digit * axes[0, 0])


def _thumb_collar(spec):
    """Collar of the thumb column in the thumb digit frame: a skirt about the hinge line that turns on the
    neck of the thenar seat, with a thin web and a hub that carry it."""
    sl = spec["digits"]["thumb"]["slider"]
    tp = np.array(sl["line_tangent_point"], dtype=float)
    along = -np.array(sl["pull_direction"], dtype=float)
    side = np.array([along[1], -along[0]])
    col = spec["palm"]["thumb_column"]
    back = col["root_behind_hinge_point"] - col["root_running_clearance"]
    r_in, r_out, z0, z1 = _FORM_COLLAR
    r_hub = col["radius"] + 0.4e-3
    profile = [
        (r_in, z0),
        (r_out - 0.4e-3, z0),
        (r_out, z0 + 0.4e-3),
        (r_out, z1 - 0.4e-3),
        (r_out - 0.4e-3, z1),
        (r_hub, z1),
        (r_hub, -back),
        (r_hub - 0.9e-3, -back),
        (r_hub - 0.9e-3, z1 - 0.3e-3),
        (r_in, z1 - 0.3e-3),
        (r_in, z0),
    ]
    rot = np.column_stack([[side[0], side[1], 0.0], [0.0, 0.0, 1.0], [along[0], along[1], 0.0]])
    if np.linalg.det(rot) < 0.0:
        rot[:, 0] *= -1.0
    return _moved(_lathe(profile, 32), rot, (tp[0], tp[1], 0.0))


# ----------------------------------------------------------------------------- modeled palm
# One body from the wrist flange to the knuckles. Its section is two superellipse halves that meet at a waist
# line (the parting line): the dorsal half is crowned and carries a knuckle ridge, the palmar half is cupped
# between the two palm pads, swells under the little finger and rises into a thenar lobe around the thumb
# hinge line. The lobe ends in a turned seat for the thumb column. Everything is visual.
_FORM_MARGIN = 4.5e-3  # distance of the shell edge outside the knuckle cheeks of the outer fingers
_FORM_THUMB_SIDE = (2.5e-3, 35.0e-3, 20.0e-3)  # shell edge on the thumb side behind the index knuckle: distance
# outside the index column, and the x range over which it grows to the distance at the knuckles
_FORM_P = 6.0  # exponent of the section halves (2 = ellipse, large = rectangle)
_FORM_WAIST = -3.5e-3  # y of the parting line: the middle of the chassis
_FORM_DORSAL = 1.0e-3  # dorsal shell above the dorsal face of the chassis, at the middle of the section
_FORM_PALMAR = 1.2e-3  # palmar shell above the palmar face of the chassis
_FORM_CROWN = 1.0e-3  # extra dorsal crown at mid length
_FORM_RIDGE = (1.2e-3, 20.0e-3)  # height of the knuckle ridge and its length behind the knuckle axes
_FORM_CUP = (
    1.8e-3,
    59.0e-3,
    13.0e-3,
    38.0e-3,
)  # palmar cup between the palm pads: depth, center x, half length, half width
_FORM_SWELL = (
    1.8e-3,
    22.0e-3,
    26.0e-3,
    14.0e-3,
)  # ulnar swell: height, center x, half length, half width from the ulnar edge
_FORM_BAND = (1.3e-3, 0.7e-3)  # half height and recess of the dark band between the two shells
_FORM_LOBE = (14.5e-3, 8.0e-3, -105.0e-3, -36.0e-3, 1.2e-3)  # thenar lobe: radius at the seat and at the tail,
# start and full radius along the hinge line (behind the hinge point), wall around the thumb actuator
_FORM_BOSS = -26.0e-3  # rear end of the turned thumb seat along the hinge line
_FORM_SEAM = (0.5e-3, 3.5e-3, 0.8e-3)  # wrist seam: distance behind the heel step, length and recess
_FORM_REAR = 4.0e-3  # the stem starts this far ahead of the rear end of the forearm stub
_FORM_HEEL = (2.0e-3, 9.0e-3, 4.0e-3, 5.0e-3)  # heel of the palm: clearance behind the last finger unit, length
# of the step down to the stem, wall beside the opposition motor, and how far behind the wrist plane the
# outline starts to narrow
_FORM_FLANGE = (23.0e-3, 21.5e-3)  # radii of the two steps of the wrist flange
_FORM_COLUMN = (0.5e-3, 4.0e-3)  # step between the heights of two finger columns: distance of the blend from the
# border between the columns, and its width
_FORM_BULKHEAD = (1.2e-3, 7.5e-3, 2.5e-3)  # front bulkhead of a finger column: thickness, the cable window left
# open on its dorsal side, and its distance from the palmar face of the chassis
_FORM_PAD = (6.5e-3, 0.8e-3)  # palm pad panels: distance of the outline from the capsule axis (across the pad
# and beyond its ends), and how far below the shell surface the skirt ends
_FORM_PAD_TOP = 5.0e-3  # the panel is the capsule's own upper surface out to this distance from the capsule
# axis (38 deg about the axis for the 8 mm pads), so that an object that touches the capsule touches the panel
_FORM_EDGE_T = (0.012, 0.04, 0.09, 0.16, 0.26, 0.38, 0.52)  # section parameter of the columns that round an edge
_FORM_FILL = (6, 2)  # extra dorsal columns between the outer slot lines and the edges: thumb side, ulnar side
_FORM_PALMAR_COLUMNS = 34


def _softmax(a, b, k):
    return 0.5 * (a + b + math.sqrt((a - b) ** 2 + k * k))


class _PalmForm:
    """Surface of the palm: outline, waist and the two height fields, as functions of the palm
    coordinates. :func:`_modeled_palm` samples it; the checks of the design use it to test what lies inside."""

    def __init__(self, spec):
        spec = _flat_spec(spec)
        palm = spec["palm"]
        self.spec = spec
        self.y_dorsal, self.y_palmar = palm["thickness_y"]
        self.o = _FingerFan(spec)
        self.stub = next(b for b in palm["solid_boxes"] if b["name"] == "forearm_stub")
        self.hinge, self.axis, self.side = _thumb_hinge(spec)
        self.flat = math.hypot(self.axis[0], self.axis[2])
        self.x_rear = self.stub["x"][0] + _FORM_REAR
        self.x_tip = max(float(self.o.frames[n][0][0]) for n in FINGERS)
        self.a_end = _FORM_SEAT[0]
        # heel of the palm: behind the rearmost finger actuator (with two units in line on a finger, that is
        # the end of its knuckle unit); behind the heel the body is a slim stem around the opposition motor
        # and the tail of the thumb actuator
        act = spec["actuator"]
        ends = []
        for name in FINGERS:
            origin, axes = self.o.frames[name]
            rear = min(cable["unit"]["rear_face_x"] for cable in _digit_cables(spec, name))
            ends.append(float(origin[0] + rear * axes[0, 0]))
        self.x_heel = min(*ends, 0.0) - _FORM_HEEL[0]
        self.x_stem = self.x_heel - _FORM_HEEL[1]
        motor = act["opposition_motor"]
        turn = (
            np.column_stack(motor["axes"]) if motor.get("axes") else _rot_axis((0.0, 1.0, 0.0), motor.get("yaw") or 0.0)
        )
        half = 0.5 * np.array(motor["size"], dtype=float)
        corners = np.array(
            [np.array(motor["center"]) + turn @ (np.array(s) * half) for s in itertools.product((-1, 1), repeat=3)]
        )
        self.stem_z = (float(corners[:, 2].min()) - _FORM_HEEL[2], float(corners[:, 2].max()) + _FORM_HEEL[2])
        # heights of the stem above and below the waist: the motor plus a wall
        self.stem_palmar = float(corners[:, 1].max()) + _FORM_LOBE[4] - _FORM_WAIST
        self.stem_dorsal = _FORM_WAIST - float(corners[:, 1].min()) + _FORM_DORSAL + _FORM_LOBE[4]
        self._u0, self._r0 = self._base_ulnar(0.0), self._base_radial(0.0)

    # outline ------------------------------------------------------------------------------------------
    def _to_stem(self, x):
        """0 at the stem, 1 from just behind the wrist plane forward (outline and arch)."""
        return _smooth((x - self.x_stem) / (-_FORM_HEEL[3] - self.x_stem))

    def _to_palm(self, x):
        """0 at the stem, 1 from the heel of the palm forward (thickness)."""
        return _smooth((x - self.x_stem) / (self.x_heel - self.x_stem))

    def _base_ulnar(self, x):
        z0 = self.o._fan_raw(FINGERS[-1], -self.o.reach, max(x, 0.0)) - _FORM_MARGIN
        return z0 if x >= 0.0 else self.stem_z[0] + (z0 - self.stem_z[0]) * self._to_stem(x)

    def _base_radial(self, x):
        # beside the thumb the edge lies closer to the index column, so that the thumb can swing across the palm
        margin = _FORM_THUMB_SIDE[0] + (_FORM_MARGIN - _FORM_THUMB_SIDE[0]) * _smooth(
            (x - _FORM_THUMB_SIDE[1]) / _FORM_THUMB_SIDE[2]
        )
        z0 = self.o._fan_raw(FINGERS[0], self.o.reach, max(x, 0.0)) + margin
        return z0 if x >= 0.0 else self.stem_z[1] + (z0 - self.stem_z[1]) * self._to_stem(x)

    def waist(self, x, z, side=1.0):
        """y of the middle of the section: it follows the height of the finger roots across the palm.

        Every finger column keeps its own height up to the border with its neighbor (its knuckle cheeks
        reach that far); the step between two columns is blended over a short width beside the border. For the
        palmar surface (``side`` > 0) the blend lies in the column that sits lower, for the dorsal surface in
        the other one, so that both shells cover the cheeks of both columns.
        """
        knots = sorted((self.o._fan_raw(n, 0.0, x), float(self.o.frames[n][0][1])) for n in FINGERS)
        gap, width = _FORM_COLUMN
        y = knots[0][1]
        for (z0, y0), (z1, y1) in itertools.pairwise(knots):
            border = 0.5 * (z0 + z1)
            toward_z1 = (y0 > y1) == (side > 0.0)
            a, b = (border + gap, border + gap + width) if toward_z1 else (border - gap - width, border - gap)
            if z <= a:
                break
            y = y1 if z >= b else y0 + (y1 - y0) * _smooth((z - a) / (b - a))
        return _FORM_WAIST + y * self._to_stem(x)

    def lobe_radius(self, a):
        """Radius of the thenar lobe at the place ``a`` along the hinge line (negative behind the hinge point)."""
        r_seat, r_tail, a_tail, a_full, _ = _FORM_LOBE
        if a <= a_tail - 8.0e-3 or a >= self.a_end:
            return 0.0
        if a < a_tail:
            return r_tail * _smooth((a - a_tail + 8.0e-3) / 8.0e-3)
        if a <= a_full:
            return r_tail + (r_seat - r_tail) * _smooth((a - a_tail) / (a_full - a_tail))
        u = (a - a_full) / (self.a_end - a_full)
        return r_seat * math.sqrt(max(1.0 - u * u, 0.0))

    def _along(self, x, z):
        dx, dz = x - self.hinge[0], z - self.hinge[2]
        a = (dx * self.axis[0] + dz * self.axis[2]) / (self.flat * self.flat)
        d = (-dx * self.axis[2] + dz * self.axis[0]) / self.flat
        return a, d

    def lobe(self, x, z):
        """Palmar and dorsal height of the lobe at a point of the palm plane, or ``None`` outside it."""
        a, d = self._along(x, z)
        r = self.lobe_radius(a)
        if abs(d) >= r:
            return None
        h = math.sqrt(r * r - d * d)
        yc = self.hinge[1] + a * self.axis[1]
        return yc + h, yc - h

    def _lobe_edge(self, x):
        """z of the outer edge of the lobe on the row ``x`` and the place along the hinge line, or ``None``."""
        z_ax = self.hinge[2] + (x - self.hinge[0]) * self.axis[2] / self.axis[0]
        z = z_ax
        for _ in range(4):
            a, _ = self._along(x, z)
            z = z_ax + self.lobe_radius(a) * self.flat / self.axis[0]
        a, _ = self._along(x, z_ax)
        return (z, a) if self.lobe_radius(a) > 0.0 or z > z_ax else None

    def ulnar(self, x):
        return self._base_ulnar(x)

    def radial(self, x):
        base = self._base_radial(x)
        edge = self._lobe_edge(x)
        if edge is None:
            return base
        z, a = edge
        # in plan the nose of the lobe turns back to the palm edge around the rear end of the turned seat: the
        # thumb unit stays covered up to the seat, and the collar of the thumb column turns free in front of it
        a_turn = _FORM_BOSS - 4.0e-3
        taper = 1.0 - _smooth((a - a_turn) / (self.a_end - a_turn))
        return base + max(_softmax(base, z + _FORM_LOBE[4], 2.5e-3) - base, 0.0) * taper

    def section(self, x):
        zu, zr = self.ulnar(x), self.radial(x)
        return 0.5 * (zu + zr), 0.5 * (zr - zu)

    # heights ------------------------------------------------------------------------------------------
    def fan(self, name, offset, x):
        """z of the line parallel to a finger. It is straight as far back as the heel of the palm (the finger
        units lie along it); from the heel to the stem it moves to its place between the base edges."""
        straight = self.o._fan_raw(name, offset, x)
        if x >= self.x_heel:
            return straight
        f = (self.o._fan_raw(name, offset, 0.0) - self._u0) / (self._r0 - self._u0)
        u, r = self._base_ulnar(x), self._base_radial(x)
        mapped = u + f * (r - u)
        return mapped + (straight - mapped) * self._to_palm(x)

    def height_dorsal(self, x, z):
        """Height of the dorsal surface above the waist at the middle of the section (before the edge profile)."""
        h = (_FORM_WAIST - self.y_dorsal) + _FORM_DORSAL
        h = self.stem_dorsal + (h - self.stem_dorsal) * self._to_palm(x)
        h += _FORM_CROWN * max(math.sin(math.pi * (x - self.x_rear) / (self.x_tip - self.x_rear)), 0.0)
        d = self.o.knuckle(z) - x
        if 0.0 < d < _FORM_RIDGE[1]:
            bump = 0.0
            for name in FINGERS:
                t = abs(z - self.fan(name, 0.0, x)) / 8.5e-3
                if t < 1.0:
                    bump = max(bump, math.cos(0.5 * math.pi * t) ** 2)
            h += _FORM_RIDGE[0] * math.sin(math.pi * d / _FORM_RIDGE[1]) ** 2 * (0.5 + 0.5 * bump)
        return h

    def height_palmar(self, x, z):
        h = (self.y_palmar - _FORM_WAIST) + _FORM_PALMAR
        h = self.stem_palmar + (h - self.stem_palmar) * self._to_palm(x)  # the heel of the palm steps down to the stem
        depth, cx, ax, az = _FORM_CUP
        tx, tz = abs(x - cx) / ax, abs(z - 0.5 * (self._u0 + self._r0)) / az
        if tx < 1.0 and tz < 1.0:
            h -= depth * math.cos(0.5 * math.pi * tx) ** 2 * (1.0 - tz**4)
        height, sx, bx, bz = _FORM_SWELL
        tx, tz = abs(x - sx) / bx, abs(z - (self.ulnar(x) + bz)) / bz
        if tx < 1.0 and tz < 1.0:
            h += height * math.cos(0.5 * math.pi * tx) ** 2 * math.cos(0.5 * math.pi * tz) ** 2
        lobe = self.lobe(x, z)
        if lobe is not None:
            h = _softmax(h, lobe[0] + _FORM_LOBE[4] - self.waist(x, z), 1.5e-3)
        return h

    def profile(self, u):
        return (1.0 - min(abs(u), 1.0) ** _FORM_P) ** (1.0 / _FORM_P)

    def surfaces(self, x, z):
        """y of the dorsal and of the palmar surface at a point of the palm plane, or ``None`` outside the outline."""
        zc, a = self.section(x)
        u = (z - zc) / a
        if abs(u) >= 1.0:
            return None
        s = self.profile(u)
        return self.waist(x, z, -1.0) - self.height_dorsal(x, z) * s, self.waist(x, z, 1.0) + self.height_palmar(
            x, z
        ) * s

    def point_t(self, x, t, shrink=0.0):
        """Point of the section at the parameter ``t``: 0 = thumb-side waist, pi/2 = dorsal middle, pi = ulnar waist."""
        zc, a = self.section(x)
        c, s = math.cos(t), math.sin(t)
        z = zc + (a - shrink) * math.copysign(abs(c) ** (2.0 / _FORM_P), c)
        k = abs(s) ** (2.0 / _FORM_P)
        zq = min(max(z, zc - a + 1.0e-6), zc + a - 1.0e-6)
        if s >= 0.0:
            y = self.waist(x, zq, -1.0) - self.height_dorsal(x, zq) * k
        else:
            y = self.waist(x, zq, 1.0) + self.height_palmar(x, zq) * k
        return np.array([x, y, z])

    def point_z(self, x, z, dorsal):
        y = self.surfaces(x, z)
        if y is None:
            zc, a = self.section(x)
            z = min(max(z, zc - a + 1.0e-5), zc + a - 1.0e-5)
            y = self.surfaces(x, z)
        return np.array([x, y[0] if dorsal else y[1], z])

    def band_t(self, x, dorsal, side):
        """Parameter at which the surface is the half height of the band away from the waist."""
        zc, a = self.section(x)
        z = zc + side * (a - 1.0e-4)
        h = self.height_dorsal(x, z) if dorsal else self.height_palmar(x, z)
        t = math.asin(min((_FORM_BAND[0] / h) ** (0.5 * _FORM_P), 1.0))
        return t if dorsal else -t

    def edge_u(self):
        """Lateral coordinate (fraction of the half width) of the innermost edge column."""
        return math.cos(_FORM_EDGE_T[-1]) ** (2.0 / _FORM_P)


def _side_screws(spec):
    """Position and outward normal (palm frame) of the six fasteners on the dark band of the palm."""
    form = _PalmForm(spec)
    out = []
    for sign, edge in ((-1.0, form.ulnar), (1.0, form.radial)):
        for x in (30.0e-3, 50.0e-3, 70.0e-3):
            slope = (edge(x + 1.0e-3) - edge(x - 1.0e-3)) / 2.0e-3
            normal = sign * np.array([-slope, 0.0, 1.0]) / math.hypot(slope, 1.0)
            out.append((np.array([x, form.waist(x, edge(x)), edge(x)]) - _FORM_BAND[1] * normal, normal))
    return out


def _modeled_palm(spec):
    """Build the palm (see the notes at the top of this section).

    Parts: ``shell_dorsal`` and ``shell_palmar`` (cover), ``parting_band`` (frame), ``wrist_stem`` (frame),
    ``wrist_seam`` (frame), ``wrist_flange`` (housing), ``thenar_seat`` (frame), the knuckle cheeks, bulkheads
    and pins, the two palm pads as soft panels on the palmar shell, and the opposition motor.
    """
    spec = _flat_spec(spec)
    act = spec["actuator"]
    form = _PalmForm(spec)
    o = form.o
    out = {}
    setback = spec["palm"]["palmar_face_setback"] + 0.8e-3
    pi = math.pi

    # columns: ("t", parameter) on the rounded edges, ("z", function of x) across the faces
    def line(name, offset):
        return lambda x: form.fan(name, offset, x)

    def edge_z(side):
        return lambda x: form.section(x)[0] + side * form.section(x)[1] * form.edge_u()

    def between(f, g, w):
        return lambda x: (1.0 - w) * f(x) + w * g(x)

    fans, names = [], []
    for name in FINGERS:  # from the thumb side to the ulnar side
        for k, off in enumerate((_SLOT, 0.0, -_SLOT)):
            fans.append(line(name, off))
            names.append((name, k))
    dorsal_z, dorsal_names = [], []
    for k in range(_FORM_FILL[0]):
        dorsal_z.append(between(edge_z(1.0), fans[0], (k + 1.0) / (_FORM_FILL[0] + 1.0)))
        dorsal_names.append(None)
    for k, f in enumerate(fans):
        dorsal_z.append(f)
        dorsal_names.append(names[k])
        if k % 3 == 2 and k + 1 < len(fans):
            dorsal_z.append(between(f, fans[k + 1], 0.5))
            dorsal_names.append(None)
    for k in range(_FORM_FILL[1]):
        dorsal_z.append(between(fans[-1], edge_z(-1.0), (k + 1.0) / (_FORM_FILL[1] + 1.0)))
        dorsal_names.append(None)

    def ordered(f, k, n):
        """Column kept between the two edges and in its order, where the outline is narrower than the fan."""
        return lambda x: min(max(f(x), edge_z(-1.0)(x) + (n - k) * 0.05e-3), edge_z(1.0)(x) - (k + 1) * 0.05e-3)

    dorsal_z = [ordered(f, k, len(dorsal_z)) for k, f in enumerate(dorsal_z)]
    palmar_z = [
        between(edge_z(-1.0), edge_z(1.0), (k + 1.0) / (_FORM_PALMAR_COLUMNS + 1.0))
        for k in range(_FORM_PALMAR_COLUMNS)
    ]

    def columns_dorsal(with_waist):
        cols = [("t", 0.0)] if with_waist else [("band", (True, 1.0))]
        cols += [("t", t) for t in _FORM_EDGE_T] + [("zd", f) for f in dorsal_z]
        cols += [("t", pi - t) for t in reversed(_FORM_EDGE_T)]
        return cols + ([] if with_waist else [("band", (True, -1.0))])

    def columns_palmar(with_waist):
        cols = [("t", pi)] if with_waist else [("band", (False, -1.0))]
        cols += [("t", pi + t) for t in _FORM_EDGE_T] + [("zp", f) for f in palmar_z]
        cols += [("t", 2.0 * pi - t) for t in reversed(_FORM_EDGE_T)]
        return cols + ([] if with_waist else [("band", (False, 1.0))])

    def column_point(col, x, shrink=0.0):
        kind, value = col
        if kind == "t":
            return form.point_t(x, value, shrink)
        if kind == "band":
            dorsal, side = value
            t = form.band_t(x, dorsal, side)
            return form.point_t(x, t if side > 0.0 else pi - t, shrink)
        return form.point_z(x, value(x), kind == "zd")

    def column_front(col):
        kind = col[0]
        x = 85.0e-3
        for _ in range(5):
            p = column_point(col, x)
            zc, a = form.section(x)
            u = abs(p[2] - zc) / a
            xk = o.knuckle(p[2])
            palmar = kind == "zp" or (kind == "t" and math.sin(col[1]) < 0.0) or (kind == "band" and not col[1][0])
            face = xk - (setback if palmar else 0.5e-3)
            x = face + _smooth((u - 0.80) / 0.17) * (xk - _PALM_FRONT - face)
        return x

    def grid(cols, stations, lip_front=False, lip_rear=False, shrink=0.0):
        fronts = [column_front(c) for c in cols]
        rows = [[column_point(c, min(x, f), shrink) for c, f in zip(cols, fronts, strict=True)] for x in stations]

        def lip(row, x):
            zc, _ = form.section(x)
            center = np.array([x, form.waist(x, zc), zc])
            return [
                p + 1.2e-3 * (np.array([p[0], center[1], center[2]]) - p) / np.linalg.norm(p[1:] - center[1:])
                for p in row
            ]

        if lip_rear:
            rows.insert(0, lip(rows[0], stations[0]))
        if lip_front:
            rows.append(lip(rows[-1], stations[-1]))
        return np.array(rows, dtype=float)

    # the wrist seam lies just behind the heel of the palm, where the stem begins
    seam1 = form.x_stem - _FORM_SEAM[0]
    seam0, seam_recess = seam1 - _FORM_SEAM[1], _FORM_SEAM[2]
    # rows: 2 mm apart over the heel step and around the nose of the lobe, 4 mm elsewhere
    shell_x = [seam1]
    for stop, pitch in (
        (form.x_heel + 2.0e-3, 2.0e-3),
        (-2.0e-3, 4.0e-3),
        (10.0e-3, 2.0e-3),
        (form.x_tip + 4.0e-3, 4.0e-3),
    ):
        while shell_x[-1] + pitch <= stop + 1.0e-9:
            shell_x.append(shell_x[-1] + pitch)
    sleeve_x = [form.x_rear + 4.0e-3 * k for k in range(int((seam0 - 4.0e-3 - form.x_rear) / 4.0e-3) + 1)]
    sleeve_x += [seam0 - 2.0e-3, seam0]

    def inside_seat(points):
        """Quads of a grid that lie inside the turned thumb seat: left open, so that no shell surface crosses
        the hinge line where the thumb unit, its carriage and the cable run into the seat."""
        centers = 0.25 * (points[:-1, :-1] + points[1:, :-1] + points[:-1, 1:] + points[1:, 1:])
        rel = centers - form.hinge
        along = rel @ form.axis
        radial = np.linalg.norm(rel - along[..., None] * form.axis, axis=-1)
        hit = (along > _FORM_BOSS + 1.0e-3) & (radial < _FORM_COLLAR[1] - 1.0e-3)
        return [(int(i), int(j)) for i, j in zip(*np.nonzero(hit), strict=True)]

    def station_span(x0, x1):
        """Rows of the shells that enclose the range ``x0..x1`` (first row is the rear lip)."""
        lo = max(i for i, x in enumerate(shell_x) if x <= x0 + 1.0e-9)
        return lo + 1, min(i for i, x in enumerate(shell_x) if x >= x1 - 1.0e-9) + 1

    # dorsal shell with a slot over the travel of every finger carriage (two slots in line on a two-cable finger)
    cols_d = columns_dorsal(False)
    top = grid(cols_d, shell_x, lip_front=True, lip_rear=True)
    skip, lips = [], []
    offset = 1 + len(_FORM_EDGE_T)
    for name in FINGERS:
        j0, j1 = offset + dorsal_names.index((name, 0)), offset + dorsal_names.index((name, 2))
        for cable in _digit_cables(spec, name):
            sl = cable["slider"]
            x_rear = sl["rest_anchor"][0] - (sl["travel_max"] + 2.0 * act["carriage_half_length"] + 1.0e-3)
            x_front = sl["rest_anchor"][0] + act["finger_unit_front_offset"] - 1.0e-3
            i0, i1 = station_span(o.palm_x(name, x_rear), o.palm_x(name, x_front))
            skip += [(i, j) for i in range(i0, i1) for j in range(j0, j1)]
            lips.append(_hole_lip(top, i0, i1, j0, j1, (0.0, 1.2e-3, 0.0)))
    out["shell_dorsal"] = (*_merge([_grid_mesh(top, skip + inside_seat(top)), *lips]), "cover")

    # palmar shell
    tub = grid(columns_palmar(False), shell_x, lip_front=True, lip_rear=True)
    out["shell_palmar"] = (*_grid_mesh(tub, inside_seat(tub)), "cover")

    # dark band between the shells, recessed, on both sides
    bands = []
    for side in (1.0, -1.0):
        cols = [("band", (True, side)), ("band", (False, side))]
        if side < 0.0:
            cols = cols[::-1]
        outer = grid(cols, shell_x)
        inner = grid(cols, shell_x, shrink=_FORM_BAND[1])
        strip = np.stack([outer[:, 0], inner[:, 0], inner[:, 1], outer[:, 1]], axis=1)
        strip = strip if side > 0.0 else strip[:, ::-1]
        bands.append(_grid_mesh(strip, inside_seat(strip)))
    out["parting_band"] = (*_merge(bands), "frame")

    # wrist stem (around the opposition motor and the tail of the thumb actuator) and the wrist seam
    loop = columns_dorsal(True) + columns_palmar(True)
    sleeve = grid(loop, sleeve_x, lip_front=True)
    out["wrist_stem"] = (*_grid_mesh(sleeve, wrap=True), "frame")
    seam = grid(loop, [seam0 - 0.5e-3, seam0, seam1, seam1 + 0.5e-3], shrink=seam_recess)
    out["wrist_seam"] = (*_grid_mesh(seam, wrap=True), "frame")

    # turned seat of the thumb column at the nose of the lobe
    pod_end, seat, neck_end = _FORM_SEAT
    r_end = _FORM_COLLAR[1]
    # hollow: the thumb unit, its carriage and the cable run through the bore
    bore = 0.5 * act["thumb_unit_cylinder"]["diameter"] + 0.8e-3
    profile = [(bore, _FORM_BOSS), (r_end, _FORM_BOSS), (r_end, pod_end), (r_end - 1.0e-3, seat), (_FORM_NECK_R, seat)]
    profile += [(_FORM_NECK_R, neck_end), (2.5e-3, neck_end)]
    turn = np.column_stack([form.side, np.cross(form.axis, form.side), form.axis])
    out["thenar_seat"] = (*_moved(_lathe(profile, 40), turn, form.hinge), "frame")

    # round wrist flange at the rear end of the stem: two steps, centered on the stem
    rear = sleeve[0]
    center = np.array(
        [form.x_rear, 0.5 * (rear[:, 1].min() + rear[:, 1].max()), 0.5 * (rear[:, 2].min() + rear[:, 2].max())]
    )
    r0, r1 = _FORM_FLANGE
    profile = [(0.0, -4.0e-3), (r0 - 1.0e-3, -4.0e-3), (r0, -3.0e-3), (r0, 0.5e-3), (r1, 1.5e-3), (r1, 3.0e-3)]
    profile += [(r1 - 1.5e-3, 4.5e-3), (0.0, 4.5e-3)]
    turn = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # lathe axis along palm +x
    out["wrist_flange"] = (*_moved(_lathe(profile, 64), turn, center), "housing")

    # knuckle cheeks, and a bulkhead that closes the front of every finger column below the cable window
    cheeks, pins = _palm_cheeks(spec)
    y_dorsal, y_palmar = spec["palm"]["thickness_y"]
    for name in FINGERS:
        origin, axes = _digit_frame(spec["digits"][name])
        # as wide as the finger link, so that its corners stay inside the rounded shell
        half = 0.5 * spec["digits"][name]["links"][0]["width"]
        wall = _box(
            (-_PALM_FRONT - _FORM_BULKHEAD[0], y_dorsal + _FORM_BULKHEAD[1], -half),
            (-_PALM_FRONT, y_palmar - _FORM_BULKHEAD[2], half),
            _BEVEL,
        )
        cheeks.append(_moved(wall, axes, origin))
    out["frame"] = (*_merge(cheeks), "frame")
    out["pins"] = (*_merge(pins), "pin")
    _palm_pads_and_motor(spec, out)

    # Palm pads: soft panels over the pad capsules of the specification. An object that rests on the palm
    # touches the collision capsule of a pad, so the drawn panel is that capsule's upper surface: the top of
    # the cylinder and of its two end caps, out to `_FORM_PAD_TOP` from the capsule axis. Beyond that a skirt
    # runs down into the palmar shell and ends below its surface. Where the shell stands higher than the
    # capsule (at the ulnar end of both pads) the panel lies under the shell and the shell is what shows.
    outline, sink = _FORM_PAD
    top = min(_FORM_PAD_TOP, 0.95 * outline)
    rings = (1.0, 0.94, 0.87, top / outline, 0.6 * top / outline, 0.3 * top / outline, 0.0)  # outermost first
    for pad in spec["palm"]["pads"]:
        p0, p1 = np.array(pad["p0"], dtype=float), np.array(pad["p1"], dtype=float)
        if p0[2] > p1[2]:
            p0, p1 = p1, p0
        radius = float(pad["radius"])

        def capsule_top(x, z, p0=p0, p1=p1, radius=radius):
            """Height of the capsule's upper surface over the palm at ``(x, z)``, within its radius."""
            z_axis = min(max(z, p0[2]), p1[2])
            y_axis = p0[1] + (p1[1] - p0[1]) * (z_axis - p0[2]) / (p1[2] - p0[2])
            offset = (x - p0[0]) ** 2 + (z - z_axis) ** 2
            return y_axis + math.sqrt(max(radius * radius - offset, 0.0))

        panel = []
        for rho in rings:
            distance = rho * outline  # from the capsule axis, measured in the palm plane
            row = []
            for k in range(40):
                # a stadium around the axis: ten points on each long side and on each end
                if k < 10:
                    dx, z = distance, p0[2] + (p1[2] - p0[2]) * k / 10.0
                elif k < 20:
                    angle = pi * (k - 10) / 10.0
                    dx, z = distance * math.cos(angle), p1[2] + distance * math.sin(angle)
                elif k < 30:
                    dx, z = -distance, p1[2] + (p0[2] - p1[2]) * (k - 20) / 10.0
                else:
                    angle = pi * (k - 30) / 10.0
                    dx, z = -distance * math.cos(angle), p0[2] - distance * math.sin(angle)
                x = p0[0] + dx
                if distance <= top + 1.0e-12:
                    y = capsule_top(x, z)
                else:
                    # skirt: from the capsule surface at the edge of the top down to below the shell
                    scale = top / distance
                    z_axis = min(max(z, p0[2]), p1[2])
                    y_edge = capsule_top(p0[0] + dx * scale, z_axis + (z - z_axis) * scale)
                    blend = _smooth((distance - top) / (outline - top))
                    y = y_edge + (form.surfaces(x, z)[1] - sink - y_edge) * blend
                row.append((x, y, z))
            panel.append(row)
        out[f"pad_{pad['name']}"] = (*_grid_mesh(np.array(panel), wrap=True), "pad")
    return out


# ----------------------------------------------------------------------------- assembly
def body_keys(spec: dict) -> list[str]:
    """Return the body keys that visual parts refer to, in a fixed order."""
    keys = ["palm", "thumb_column"]
    for name, digit in spec["digits"].items():
        keys += [f"{name}/link1", f"{name}/link2", f"{name}/link3", f"{name}/slider"]
        # a digit with several cables has one more slider body per further cable: "<digit>/slider_<cable name>"
        keys += [f"{name}/slider_{cable['name']}" for cable in digit.get("cables", [])[1:]]
    return keys


def build_hand_visuals(spec: dict) -> list[VisualPart]:
    """Build every visual part of the hand.

    Args:
        spec: Hand specification as a plain dictionary (``digits``, ``palm``,
            ``actuator``, ``station_hardware``, ``cable``), as returned by
            ``HandSpec.to_dict()`` of the hand module.

    Returns:
        The visual parts, in a fixed order. Lengths are in meters, in the frame of
        the body named by :attr:`VisualPart.body`.
    """
    parts: list[VisualPart] = []
    protos: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    proto_keys: dict[tuple, str] = {}

    def as_arrays(mesh):
        v, t = mesh[0], mesh[1]
        return np.ascontiguousarray(v, dtype=np.float32), np.ascontiguousarray(t, dtype=np.int32)

    def add(name, body, mesh, material, pos=None, rot=None):
        v, t = as_arrays(mesh)
        parts.append(VisualPart(name, body, v, t, material, *_pose(pos, rot)))

    def _pose(pos, rot):
        p = np.zeros(3) if pos is None else np.array(pos, dtype=float)
        q = np.array([0.0, 0.0, 0.0, 1.0]) if rot is None else _quat_from_matrix(rot)
        return p, q

    def add_instance(name, body, base, key, make, material, pos=None, rot=None):
        """Add an instance of a shared mesh; ``key`` holds the numbers the mesh depends on."""
        key = (base, *[round(float(k), 9) for k in key])
        if key not in proto_keys:
            taken = sum(1 for k in proto_keys if k[0] == base)
            proto_keys[key] = base if taken == 0 else f"{base}_{taken + 1}"
            protos[proto_keys[key]] = as_arrays(make())
        proto = proto_keys[key]
        v, t = protos[proto]
        parts.append(VisualPart(name, body, v, t, material, *_pose(pos, rot), prototype=proto))

    def add_pulley(body, st, pos, rot=None):
        radius = st["radius"]
        add_instance(
            st["name"],
            body,
            f"pulley_{radius * 1.0e3:.1f}mm".replace(".", "p"),
            (radius,),
            lambda: build_pulley_mesh(spec, radius),
            st["kind"],
            pos,
            rot,
        )

    spec = _flat_spec(spec)
    act = spec["actuator"]
    cl = act["finger_unit_cable_line_above_bottom"]

    # palm
    for name, (v, t, material) in build_palm_meshes(spec).items():
        add(name, "palm", (v, t), material)
    for finger in FINGERS:
        digit = spec["digits"][finger]
        origin, axes = _digit_frame(digit)
        for k, cable in enumerate(_digit_cables(spec, finger)):
            # one linear unit per cable; a finger with two cables carries two units in line
            unit = cable["unit"]
            stroke = cable["slider"]["travel_max"]
            front = origin + axes[:, 0] * unit["front_face_x"] + axes[:, 1] * (unit["dorsal_face_y"] + cl)
            front = front + axes[:, 2] * unit.get("z_center", 0.0)
            sized = {**spec, "actuator": {**act, "finger_unit_box": list(unit["box"]), "stroke": stroke}}
            same = unit["box"][0] == act["finger_unit_box"][0] and stroke == act["stroke"]
            add_instance(
                f"{finger}_actuator" if k == 0 else f"{finger}_{cable['name']}_actuator",
                "palm",
                "finger_actuator",
                () if same else (unit["box"][0], stroke),
                lambda sized=sized: build_finger_actuator_mesh(sized),
                "actuator",
                front,
                axes,
            )
        for cable in _digit_cables(spec, finger):
            for st in cable["stations"]:
                if st["body"] == "base" and st["kind"] in ("sheave", "guide"):
                    pos = origin + axes[:, 0] * st["pos"][0] + axes[:, 1] * st["pos"][1] + axes[:, 2] * cable["plane_z"]
                    add_pulley("palm", {**st, "name": f"{finger}_{st['name']}"}, pos, axes)
    for k, (pos, normal) in enumerate(_side_screws(spec)):
        add_instance(f"screw_{k + 1:02d}", "palm", "screw", (), build_screw_mesh, "pin", pos, _frame_from_z(normal))
    thumb = spec["digits"]["thumb"]
    mount = thumb["mount"]
    ax = np.array(mount["slider_line_direction"], dtype=float)
    side = np.cross(ax, [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    add_instance(
        "thumb_actuator",
        "palm",
        "thumb_actuator",
        (),
        lambda: build_thumb_actuator_mesh(spec),
        "actuator",
        mount["hinge_point"],
        np.column_stack([side, np.cross(ax, side), ax]),
    )

    # thumb column
    for name, (v, t, material) in build_thumb_column_meshes(spec).items():
        add(name, "thumb_column", (v, t), material)
    add("collar", "thumb_column", _thumb_collar(spec), "frame")
    for st in thumb["stations"]:
        if st["body"] == "base" and st["kind"] in ("sheave", "guide"):
            add_pulley("thumb_column", st, (st["pos"][0], st["pos"][1], 0.0))

    # digits
    for name, digit in spec["digits"].items():
        for i in range(3):
            body = f"{name}/link{i + 1}"
            lk = digit["links"][i]
            for part, (v, t, material) in build_link_meshes(spec, name, i).items():
                if part == "ears":
                    parent = digit["links"][i - 1]
                    key = (
                        i,
                        lk["y_dorsal"],
                        lk["y_palmar"],
                        lk["palmar_chamfer_proximal"],
                        lk["knuckle_radius_proximal"],
                        parent["width"],
                        parent["knuckle_radius_distal"],
                        _nose_start(digit) if i == 2 else 0.0,
                    )
                    add_instance(part, body, f"link{i + 1}_ears", key, lambda v=v, t=t: (v, t), material)
                else:
                    add(part, body, (v, t), material)
            for cable in _digit_cables(spec, name):
                for st in cable["stations"]:
                    if st["body"] == f"link{i + 1}" and st["kind"] in ("sheave", "guide"):
                        add_pulley(body, st, (st["pos"][0], st["pos"][1], cable["plane_z"]))
        # carriages: one per cable, on the slider body of that cable
        sl = digit["slider"]
        anchor = np.array([sl["rest_anchor"][0], sl["rest_anchor"][1], 0.0])
        to_digit = -np.array([sl["pull_direction"][0], sl["pull_direction"][1], 0.0])
        if name in FINGERS:
            for cable in _digit_cables(spec, name):
                sl = cable["slider"]
                anchor = np.array([sl["rest_anchor"][0], sl["rest_anchor"][1], cable["unit"].get("z_center", 0.0)])
                to_digit = -np.array([sl["pull_direction"][0], sl["pull_direction"][1], 0.0])
                rot = np.column_stack([to_digit, np.cross([0.0, 0.0, 1.0], to_digit), [0.0, 0.0, 1.0]])
                add_instance(
                    "carriage",
                    cable["body"],
                    "carriage",
                    (),
                    lambda: build_carriage_mesh(spec),
                    "carriage",
                    anchor,
                    rot,
                )
        else:
            rot = _frame_from_z(to_digit)
            add_instance(
                "carriage",
                f"{name}/slider",
                "carriage_round",
                (),
                lambda: build_carriage_mesh(spec, round_body=True),
                "carriage",
                anchor,
                rot,
            )
    return parts


# ----------------------------------------------------------------------------- asset
def _asset_path(path=None) -> str:
    if path is not None:
        return os.fspath(path)
    import newton.examples  # noqa: PLC0415

    return newton.examples.get_asset(ASSET_NAME)


def write_visuals_usd(parts: list[VisualPart], path: str | os.PathLike | None = None) -> str:
    """Write visual parts to a USD file.

    The stage holds one ``Xform`` per body (``/cable_hand/<body>`` with ``/`` in the
    body key as a path separator) and one ``Mesh`` per part below it. Shared meshes
    are stored once below ``/cable_hand/prototypes`` and referenced; the look of
    every material tag is stored as a preview surface below ``/cable_hand/looks``.

    Args:
        parts: The visual parts.
        path: Output file. Defaults to the shipped asset location.

    Returns:
        The path written.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt  # noqa: PLC0415

    path = _asset_path(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/cable_hand")
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().documentation = (
        "Visual meshes of the cable-driven hand example. Units: meters. Each mesh is given in the frame of the "
        "body it is listed under. Generated by newton/examples/cable/cable_hand_visuals.py."
    )

    looks = {}
    stage.DefinePrim("/cable_hand/looks", "Scope")
    for tag in sorted({p.material for p in parts}):
        look = MATERIALS.get(tag, MATERIALS["frame"])
        material = UsdShade.Material.Define(stage, f"/cable_hand/looks/{tag}")
        shader = UsdShade.Shader.Define(stage, f"/cable_hand/looks/{tag}/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*look["color"]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(look["roughness"]))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(look["metallic"]))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        looks[tag] = material

    def author_mesh(prim_path, vertices, indices):
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(vertices, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(indices), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(indices, dtype=np.int32).ravel()))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
        return mesh

    stage.CreateClassPrim("/cable_hand/prototypes")
    done = set()
    for part in parts:
        if part.prototype is not None and part.prototype not in done:
            author_mesh(f"/cable_hand/prototypes/{part.prototype}", part.vertices, part.indices)
            done.add(part.prototype)

    for part in parts:
        body_path = "/cable_hand/" + part.body
        for depth in range(1, part.body.count("/") + 2):
            UsdGeom.Xform.Define(stage, "/cable_hand/" + "/".join(part.body.split("/")[:depth]))
        prim_path = f"{body_path}/{part.name}"
        if part.prototype is None:
            mesh = author_mesh(prim_path, part.vertices, part.indices)
        else:
            mesh = UsdGeom.Mesh.Define(stage, prim_path)
            mesh.GetPrim().GetReferences().AddInternalReference(f"/cable_hand/prototypes/{part.prototype}")
        if np.any(part.pos != 0.0) or np.any(part.quat != np.array([0.0, 0.0, 0.0, 1.0])):
            mesh.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, part.pos)))
            x, y, z, w = map(float, part.quat)
            mesh.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
        look = MATERIALS.get(part.material, MATERIALS["frame"])
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*look["color"])])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(looks[part.material])
    stage.GetRootLayer().Export(path)
    return path


def load_visuals_usd(path: str | os.PathLike | None = None) -> list[VisualPart]:
    """Load visual parts from a USD file written by :func:`write_visuals_usd`.

    Args:
        path: Input file. Defaults to the shipped asset.

    Returns:
        The visual parts in the order they were written. Instanced parts share
        their vertex and index arrays.
    """
    from pxr import Usd, UsdGeom, UsdShade  # noqa: PLC0415

    stage = Usd.Stage.Open(_asset_path(path))
    root = stage.GetDefaultPrim()
    shared: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def read_mesh(prim):
        mesh = UsdGeom.Mesh(prim)
        vertices = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32).reshape(-1, 3)
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
        return vertices, indices

    parts = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        rel = str(prim.GetPath().MakeRelativePath(root.GetPath()))
        body, name = rel.rsplit("/", 1)
        proto = None
        for spec in prim.GetPrimStack():
            for ref in spec.referenceList.GetAddedOrExplicitItems():
                proto = ref.primPath.name
        if proto is None:
            vertices, indices = read_mesh(prim)
        else:
            if proto not in shared:
                shared[proto] = read_mesh(prim)
            vertices, indices = shared[proto]
        pos, quat = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                pos = np.array(op.Get(), dtype=float)
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                q = op.Get()
                quat = np.array([*q.GetImaginary(), q.GetReal()], dtype=float)
        material = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterialPath().name
        parts.append(VisualPart(name, body, vertices, indices, material, pos, quat, proto))
    return parts


def add_visuals_to_builder(builder, parts: list[VisualPart], bodies: dict[str, int], materials: dict | None = None):
    """Attach visual parts to bodies of a model builder as visual-only mesh shapes.

    Args:
        builder: The :class:`newton.ModelBuilder` to add shapes to.
        parts: The visual parts.
        bodies: Body index for every body key used by the parts. Parts whose body
            key is missing are skipped.
        materials: Look per material tag (``color``, ``roughness``, ``metallic`` and
            optionally ``opacity``). Defaults to :data:`MATERIALS`.

    Returns:
        Mapping from ``"<body>/<part name>"`` to the shape index.
    """
    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415

    materials = MATERIALS if materials is None else materials
    cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False, density=0.0)
    meshes: dict[tuple, newton.Mesh] = {}
    shapes = {}
    for part in parts:
        if part.body not in bodies:
            continue
        look = materials.get(part.material, MATERIALS["frame"])
        key = (id(part.vertices), part.material)
        if key not in meshes:
            meshes[key] = newton.Mesh(
                part.vertices,
                part.indices.ravel(),
                compute_inertia=False,
                is_solid=False,
                color=tuple(look["color"]),
                roughness=look.get("roughness"),
                metallic=look.get("metallic"),
                opacity=look.get("opacity"),
            )
        shapes[f"{part.body}/{part.name}"] = builder.add_shape_mesh(
            bodies[part.body],
            xform=wp.transform(wp.vec3(*map(float, part.pos)), wp.quat(*map(float, part.quat))),
            mesh=meshes[key],
            cfg=cfg,
            label=f"visual/{part.body}/{part.name}",
        )
    return shapes


def _default_spec() -> dict:
    from newton.examples.cable.cable_hand import default_hand_spec  # noqa: PLC0415

    return default_hand_spec().to_dict()


def main(argv=None):
    """Regenerate the shipped asset and print its part and triangle counts."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--output", default=None, help="Output USD file (default: the shipped asset).")
    args = parser.parse_args(argv)
    parts = build_hand_visuals(_default_spec())
    path = write_visuals_usd(parts, args.output)
    unique = {id(p.vertices): p for p in parts}
    print(
        f"{len(parts)} parts, {len(unique)} meshes, "
        f"{sum(len(p.vertices) for p in unique.values())} stored vertices, "
        f"{sum(len(p.indices) for p in unique.values())} stored triangles, "
        f"{sum(len(p.indices) for p in parts)} triangles in the hand, "
        f"{os.path.getsize(path)} bytes: {os.path.basename(path)}"
    )


if __name__ == "__main__":
    main()
