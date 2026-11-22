# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the cable-driven hand examples.

The hand has four fingers and a thumb. Every digit has three flexion joints
(MCP, PIP, DIP) held straight by return springs and is closed by cables, each
pulled by its own linear actuator. The ring finger, the little finger and the
thumb have one cable that runs over a pulley at each joint to the fingertip.
The index and the middle finger have two: a knuckle cable that bends the first
joint and ends on the proximal link, and a curl cable that bends the two outer
joints and passes the first joint on a small idler. The joint torques are the
moment-arm matrix times the cable tensions (:func:`moment_arm_matrix`,
:func:`frictionless_row`). A digit carries a list of cables
(:class:`DigitCableSpec`); functions that work on one cable take a ``cable``
argument, which may be left out on a digit with one cable.
The same digit is modeled twice in the examples, once with a routed rod and
once with a massless tendon. This module holds everything the two examples
are built from:

- the design numbers (:class:`HandSpec`, :func:`default_hand_spec`),
- planar kinematics and the tangent-line/arc cable route (:func:`digit_route`),
- rod data (segment count, barrel rule, start pose) and tendon data (link list
  with per-span compliance),
- closed-form statics used as the reference in tests (frictionless, capstan
  friction, gravity),
- the rigid skeleton of one digit (:func:`add_digit_skeleton`) and of the hand
  (:func:`add_hand_skeleton`),
- the control blocks shared by both cable models: actuator governor, tension
  staircase, spring preload, wrist driver and hand control
  (:func:`create_actuator_governor`, :func:`create_hand_control`),
- the rod cable model (:func:`add_rod_cable`, :func:`create_rod_solver`,
  :class:`RodTensionProbe`) and the massless tendon model
  (:func:`add_digit_tendon`, :func:`make_tendon_solver`), each for one digit
  and for the hand,
- drawing helpers (tension colors, cable pieces, visual meshes),
- grasping: pads, objects and table, the keyframe script, grasp plans and the
  grasp read-back (:func:`grasp_plan`, :func:`grasp_state`),
- the pose library and the keyframes of the choreography
  (:func:`hand_poses`, :func:`hand_choreography`).

Frames
------
Digit frame: origin on the MCP axis in the digit mid-plane, ``x`` distal along
the straight digit, ``y`` palmar, ``z = x cross y`` is the flexion axis; a
positive joint angle is flexion. Link frame: origin on the link's proximal
joint axis, ``x`` toward the next joint, ``y`` palmar; parallel to the digit
frame at zero joint angles. Station coordinates are ``(x, y)`` in the frame of
the body that carries the station, in the plane ``z = plane_z`` of the cable
(zero on a digit with one cable). Every slider body frame coincides with the
digit frame at zero slider travel. Palm frame: origin at
the center of the wrist end of the palm, ``x`` distal, ``y`` palmar,
``z = x cross y``; this is a right hand, the thumb side is ``+z``.

Mass and inertia
----------------
Each body's mass comes from the design's mass budget, and its inertia is that
of a simple solid carrying that mass: a link is a uniform box of its envelope
with the center of mass at mid length, a carriage a cube, the thumb column a
cylinder, the palm a set of boxes. The material tags of the visual parts set
only the look; rod segments take their mass from the cord's linear density.
Compared with the part meshes (glass-filled nylon frames, brass and steel
pulleys), the moment of inertia of each link about its flexion axis is within
11 %. The real centers of mass of the first two links sit 5 to 13 mm further
out, because the pulleys are at the distal ends: the gravity torque at a
joint is up to 17 % larger and the inertia reflected at a joint up to 19 %
larger than in the model; against the return spring preloads the gravity
torque changes by at most 2.6 % of the preload. The thumb column is outside
this comparison: its mass budget covers parts that have no mesh.

All quantities are in SI units (m, rad, kg, N, N*m, s).
"""

from __future__ import annotations

import dataclasses
import math
import types
import typing
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.examples.cable import cable_hand_visuals
from newton.examples.cable.cable import (
    get_tendon_attachment_worlds,
    get_tendon_cable_lines,
    get_tendon_total_lengths,
)

BODY_KEYS = ("base", "link1", "link2", "link3")
"""Keys of the bodies that carry stations, in kinematic order."""

GRAVITY = 9.81
"""Gravitational acceleration [m/s^2]."""

_MM = 1.0e-3
_DEG = math.pi / 180.0


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CableSpec:
    """Cable material and working tensions."""

    diameter: float
    """Cord diameter [m]."""
    radius: float
    """Cord radius [m]."""
    EA: float
    """Axial stiffness [N]."""
    linear_density: float
    """Mass per length [kg/m]."""
    bend_rigidity: float
    """Bending rigidity [N*m^2]."""
    twist_rigidity: float
    """Twisting rigidity [N*m^2]."""
    tension_rest: float
    """Smallest tension kept by the actuator [N]."""
    tension_hold: float
    """Grasp hold tension [N]."""
    tension_max: float
    """Tension limit [N]."""
    friction_default: float
    anchor_compliance: float
    """Total compliance of the two end anchors [m/N]."""
    hinge_compliance: float
    """Opening of one flexion hinge per newton of cable tension at that hinge [m/N].

    The inverse of the structural joint stiffness of the solver (:data:`STRUCTURAL_JOINT_KE`):
    every hinge carries about one cable tension. It adds to the slider travel.
    """


@dataclass
class StationHardwareSpec:
    """Dimensions shared by all pulleys and guides [m]."""

    flange_height_above_centerline: float
    flange_rim_gap_min: float
    shell_wall: float
    groove_half_width: float
    flange_thickness: float
    pin_radius: float


@dataclass
class LinkSpec:
    """One phalanx: outer dimensions [m] and mass [kg] in the link frame."""

    name: str
    length: float
    width: float
    y_dorsal: float
    y_palmar: float
    mass: float
    knuckle_radius_proximal: float
    knuckle_radius_distal: float
    distal_disk_x: float
    shaft_end: float
    palmar_chamfer_proximal: float
    palmar_chamfer_distal: float


@dataclass
class JointSpec:
    """One flexion joint with its return spring."""

    name: str
    lower: float
    """Lower limit [rad]."""
    upper: float
    """Upper limit [rad]."""
    sheave_radius: float
    """Moment arm about the joint of the cable that runs to the fingertip, the first cable of the digit [m].

    On a digit with several cables every cable has its own arms
    (:attr:`DigitCableSpec.moment_arms`, :func:`moment_arm_matrix`).
    """
    spring_preload: float
    """Spring torque at the lower limit, toward extension [N*m]."""
    spring_stiffness: float
    """Spring stiffness [N*m/rad]."""
    spring_damping: float
    """Joint damping [N*m*s/rad]."""
    limit_ke: float
    """Limit stiffness [N*m/rad].

    The stops are soft on purpose. Tendons need the ``local`` articulation
    solve, which solves one body at a time. A loaded joint stop is a row of
    stiffness ``limit_ke + limit_kd / dt`` between two links whose inertia over
    ``dt^2`` is thousands of times smaller, so with stiff stops (1e4 N*m/rad)
    the links on a loaded stop hold back the whole chain. The default of
    300 N*m/rad keeps the stop deflection at the tension limit below 0.1 deg.
    Both cable models and the opposition hinge use it.
    """
    limit_kd: float
    """Limit damping [N*m*s/rad]."""


@dataclass
class StationSpec:
    """One station of the cable route: an end anchor, a redirect guide or a joint sheave."""

    name: str
    kind: str
    """``"anchor"``, ``"guide"`` or ``"sheave"``."""
    body: str
    """Key of the carrying body: ``"slider"``, ``"base"``, ``"link1"``, ``"link2"`` or ``"link3"``."""
    pos: tuple[float, float]
    """Center ``(x, y)`` in the frame of the carrying body [m]."""
    radius: float
    """Center-line radius of the cable on the station [m]; zero for an anchor."""
    sense: int
    """+1: the cable runs counterclockwise about digit ``+z`` from actuator to tip, -1: clockwise, 0: anchor."""
    joint: int | None = None
    """Index of the joint a sheave is concentric with (sheaves only)."""


@dataclass
class SliderSpec:
    """Linear actuator carriage of one digit, in the digit frame."""

    rest_anchor: tuple[float, float]
    """Cable anchor at zero travel [m]."""
    pull_direction: tuple[float, float]
    """Unit direction of positive travel (pulling the cable)."""
    line_tangent_point: tuple[float, float]
    """Point where the slider line touches the first station [m]."""
    travel_min: float
    """Lower travel limit [m]."""
    travel_max: float
    """Upper travel limit [m]."""
    mass: float
    """Carriage mass [kg]."""


@dataclass
class PadSpec:
    """Contact pad capsule of one link, axis from ``p0`` to ``p1`` in the link frame [m]."""

    body: str
    p0: tuple[float, float]
    p1: tuple[float, float]
    radius: float


@dataclass
class FingerMountSpec:
    """Placement of a finger's digit frame in the palm frame."""

    base_body: str
    frame_origin: tuple[float, float, float]
    """MCP position in the palm frame [m]."""
    splay: float
    """Rotation of the digit frame about palm ``+y`` [rad]; positive turns digit ``+x`` toward ``-z``."""
    frame_x: tuple[float, float, float]
    frame_y: tuple[float, float, float]
    frame_z: tuple[float, float, float]


@dataclass
class ThumbMountSpec:
    """Opposition hinge of the thumb and the thumb's digit frame at zero opposition, in the palm frame."""

    base_body: str
    hinge_point: tuple[float, float, float]
    """Point on the hinge line where the cable meets its first station [m]."""
    hinge_axis: tuple[float, float, float]
    """Rotation axis of positive opposition."""
    slider_line_direction: tuple[float, float, float]
    """Cable travel direction toward the thumb (the hinge line is the thumb slider line)."""
    hinge_lower: float
    """Lower opposition limit [rad]."""
    hinge_upper: float
    """Upper opposition limit [rad]."""
    frame_origin_at_phi0: tuple[float, float, float]
    """MCP position at zero opposition [m]."""
    frame_x_at_phi0: tuple[float, float, float]
    frame_y_at_phi0: tuple[float, float, float]
    frame_z_at_phi0: tuple[float, float, float]


SINGLE_CABLE_NAME = "flexor"
"""Name of the cable of a digit that has one cable (the flexor from the actuator to the fingertip)."""


@dataclass
class DigitCableSpec:
    """One cable of a digit: its stations from the actuator to the end anchor and its actuator carriage.

    The last station is the end anchor; its ``body`` and ``pos`` say where the
    cable ends (the distal link for a flexor or curl cable, the proximal link
    for a knuckle cable). The first station is the anchor on the cable's own
    carriage, the body :attr:`slider_body`.
    """

    name: str
    """Cable name, unique within the digit: ``"flexor"``, ``"curl"`` or ``"knuckle"``."""
    role: str
    """What the cable does: ``"flexor"`` (all three joints, one-cable digit), ``"curl"`` or ``"knuckle"``."""
    slider_body: str
    """Body key of the cable's carriage: ``"slider"`` for the first cable of a digit,
    ``"slider_<cable name>"`` for every further cable."""
    plane_z: float
    """Offset of the cable plane from the digit mid-plane along digit ``+z`` [m]."""
    moment_arms: tuple[float, ...]
    """Moment arm of the cable about each joint (MCP, PIP, DIP) [m]: the centerline radius of the
    cable's sheave on that joint, zero where the cable has none."""
    stations: list[StationSpec]
    slider: SliderSpec
    actuator_unit: dict[str, Any] | None = None
    """Envelope of the cable's linear unit in the digit frame [m] (fingers only): ``box``
    (length, height, width), ``front_face_x``, ``rear_face_x``, ``dorsal_face_y``, ``palmar_face_y``,
    ``z_center``. The visual meshes are built from it; the simulation does not read it."""
    reference_rate_limit: float | None = None
    """Largest rate at which the governor moves the reference of this cable's actuator [m/s]; ``None``
    leaves the actuator's speed limit (minus the governor's margin) as the only limit. The curl cable
    of a two-cable finger is ramped more slowly than the actuator allows, so that the knuckle carriage,
    which must take up more cable per radian of the first joint, keeps its cable taut."""
    rod_segment_length: float | None = None
    """Target length of the rod segments of this cable [m]; ``None`` takes the common target of the rod
    spec. The thumb's cable uses 2 mm segments: with 1 mm segments on the thumb's light column the rod
    hand does not start."""


@dataclass
class DigitSpec:
    """One digit: three links, three joints, its cables (each with stations and a slider), the pads and the mount.

    The cable that runs to the fingertip comes first. See
    :meth:`HandSpec.to_dict` for the two layouts a digit is written in;
    :meth:`HandSpec.from_dict` reads both.
    """

    links: list[LinkSpec]
    joints: list[JointSpec]
    cables: list[DigitCableSpec]
    pads: list[PadSpec]
    mount: FingerMountSpec | ThumbMountSpec

    @property
    def cable_names(self) -> list[str]:
        """Names of the cables in the order of :attr:`cables`."""
        return [cable.name for cable in self.cables]

    def cable(self, cable: int | str | None = None) -> DigitCableSpec:
        """Return one cable of the digit.

        Args:
            cable: Cable name or index into :attr:`cables`; ``None`` selects
                the only cable of a one-cable digit.

        Returns:
            The cable.

        Raises:
            ValueError: If ``cable`` is ``None`` and the digit has several
                cables, or if the digit has no cable of that name.
        """
        if cable is None:
            if len(self.cables) != 1:
                raise ValueError(
                    f"this digit has {len(self.cables)} cables ({', '.join(self.cable_names)}); "
                    "name the cable (cable=...) or use the functions that take all cables of a digit"
                )
            return self.cables[0]
        if isinstance(cable, str):
            for item in self.cables:
                if item.name == cable:
                    return item
            raise ValueError(f"no cable {cable!r}; this digit has {', '.join(self.cable_names)}")
        return self.cables[cable]

    def cable_index(self, cable: int | str | None = None) -> int:
        """Index of a cable in :attr:`cables`, see :meth:`cable`.

        Args:
            cable: Cable name or index; ``None`` selects the only cable of a one-cable digit.

        Returns:
            The index.
        """
        return next(i for i, item in enumerate(self.cables) if item is self.cable(cable))

    @property
    def stations(self) -> list[StationSpec]:
        """Stations of the only cable of a one-cable digit (raises for a digit with several cables)."""
        return self.cable().stations

    @property
    def slider(self) -> SliderSpec:
        """Slider of the only cable of a one-cable digit (raises for a digit with several cables)."""
        return self.cable().slider


@dataclass
class PalmBoxSpec:
    """One box of the palm solid: ``x`` and ``z`` ranges before a yaw about palm ``+y`` around ``pivot`` [m, rad]."""

    name: str
    x: tuple[float, float]
    z: tuple[float, float]
    yaw: float | None = None
    pivot: tuple[float, float, float] | None = None
    y: tuple[float, float] | None = None
    """Height range of this box [m]; the palm's ``thickness_y`` if ``None``."""
    axes: list[tuple[float, float, float]] | None = None
    """Box frame: the box ``x``, ``y`` and ``z`` axes in the palm frame. With ``axes`` the three ranges
    are measured from ``pivot`` along these axes (a point is ``pivot + x ax + y ay + z az``) and
    ``yaw`` is not used."""


@dataclass
class ThumbColumnSpec:
    """Rotating thumb root on the opposition hinge."""

    mass: float
    radius: float
    root_behind_hinge_point: float
    root_ahead_of_hinge_point: float
    root_running_clearance: float


@dataclass
class PalmPadSpec:
    """Contact pad capsule on the palm, in the palm frame [m]."""

    name: str
    p0: tuple[float, float, float]
    p1: tuple[float, float, float]
    radius: float


@dataclass
class PalmSpec:
    """Palm solid, thumb column and palm pads."""

    mass: float
    """Palm mass including the actuators [kg]."""
    thickness_y: tuple[float, float]
    """Palm ``y`` range about the plane of the finger MCP axes [m]."""
    solid_boxes: list[PalmBoxSpec]
    palmar_face_setback: float
    thumb_column: ThumbColumnSpec
    pads: list[PalmPadSpec]


@dataclass
class ActuatorSpec:
    """Linear actuators, their governor limits and the thumb opposition drive."""

    stroke: float
    """Usable slider travel [m]."""
    speed_limit: float
    """Largest rate of the slider reference [m/s]."""
    force_limit: float
    """Largest drive force [N]."""
    force_min: float
    """Smallest drive force, the rest tension [N]."""
    kp: float
    """Slider position gain [N/m]."""
    kd: float
    """Slider velocity gain [N*s/m]."""
    stop_ke: float
    """Stiffness of the end stops of the slider stroke [N/m]."""
    stop_kd: float
    """Damping of the end stops of the slider stroke [N*s/m]."""
    slider_mass: float
    """Carriage mass [kg]."""
    opposition: dict[str, Any]
    carriage_half_length: float
    finger_unit_box: tuple[float, float, float]
    finger_unit_front_offset: float
    finger_unit_cable_line_above_bottom: float
    thumb_unit_cylinder: dict[str, Any]
    thumb_unit_front_behind_hinge_point: float
    thumb_unit_motor_length: float
    opposition_motor: dict[str, Any]


@dataclass
class PadContactSpec:
    """Contact parameters shared by all pads."""

    mu: float
    ke: float
    """Contact stiffness of pads, objects and the table in a grasp scene [N/m].

    The default is 1e5 N/m. The tendon hand needs the ``local`` articulation
    solve, which solves one body at a time. A contact of stiffness ``k``
    between the pads and a light object converges in the 16 iterations of
    :func:`tendon_solver_settings` only while ``k dt^2`` stays below the
    object's mass: at 1e6 N/m and the 0.83 ms substep it is 0.69 kg against
    0.15 kg for the can, and the can was thrown out of the closing fingers in
    two of eight runs (30 N hold tension on the CPU, 40 N on CUDA; pad forces
    of 180 to 260 N within one frame). At 1e5 N/m (0.07 kg) the same runs
    hold, and so does the 30 N run at 1e6 N/m with 32 iterations. A 15 N pad
    force indents the contact by 0.15 mm at 1e5 N/m.
    """
    gap: float
    """Contact gap [m]."""
    proud: float
    """Height of the pad surface above the palmar face [m]."""
    table_mu: float


@dataclass
class RodSpec:
    """Parameters of the routed-rod cable model."""

    radius: float
    """Rod radius [m]."""
    segment_length_target: float
    """Target segment length [m]; the actual length is cable length / segment count."""
    contact_ke: float
    """Rod contact stiffness [N/m]."""
    contact_kd: float
    contact_gap: float
    """Rod contact gap [m]."""
    settings_for_1mm: dict[str, Any] | None = None
    """Solver settings that 1 mm segments need: ``reference_inertia`` [kg*m^2] (see
    :func:`rod_inertia_reference`) and the ``seating_stroke_tensions`` [N] every cable is run
    through before anything is measured."""


@dataclass
class TendonSpec:
    """Parameters of the massless-tendon cable model."""

    axis_in_digit_frame: tuple[float, float, float]
    orientation_guides: int
    orientation_sheaves: int
    damping: float


@dataclass
class StaircaseSpec:
    """Tension staircase used by the reference tables and the single-digit tests."""

    tensions: tuple[float, ...]
    """Tension levels on the way up [N]; the same levels are visited on the way down."""
    hold: float
    """Hold time per level [s]."""


@dataclass
class HandSpec:
    """All design numbers of the cable-driven hand (SI units).

    :meth:`to_dict` and :meth:`from_dict` convert the spec to and from nested
    dicts with the field names as keys.
    """

    cable: CableSpec
    station_hardware: StationHardwareSpec
    digits: dict[str, DigitSpec]
    palm: PalmSpec
    actuator: ActuatorSpec
    pads: PadContactSpec
    objects: dict[str, dict[str, Any]]
    rod: RodSpec
    tendon: TendonSpec
    staircase: StaircaseSpec

    def to_dict(self) -> dict[str, Any]:
        """Return the spec as nested dicts and lists of plain Python values.

        Returns:
            The spec with the field names as keys.
        """
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandSpec:
        """Build a spec from nested dicts.

        Args:
            data: The spec as returned by :meth:`to_dict`; keys that are not
                spec fields (notes, derived tables) are ignored.

        Returns:
            The spec.
        """
        names = {f.name for f in dataclasses.fields(cls)}
        return _from_plain(cls, {key: value for key, value in data.items() if key in names})


def _to_plain(value):
    if dataclasses.is_dataclass(value):
        out = {}
        for f in dataclasses.fields(value):
            item = getattr(value, f.name)
            if item is not None:
                out[f.name] = _to_plain(item)
        return out
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _from_plain(tp, value):
    if dataclasses.is_dataclass(tp):
        hints = typing.get_type_hints(tp)
        # a file may keep notes and derived values next to the design numbers; they are not spec fields
        value = {key: item for key, item in value.items() if key in hints}
        return tp(**{key: _from_plain(hints[key], item) for key, item in value.items()})
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (types.UnionType, typing.Union):
        if value is None:
            return None
        options = [arg for arg in args if arg is not type(None)]
        for option in options:
            if dataclasses.is_dataclass(option) and set(value) <= {f.name for f in dataclasses.fields(option)}:
                return _from_plain(option, value)
        classes = [option for option in options if dataclasses.is_dataclass(option)]
        if classes and isinstance(value, dict):
            # keys that are not fields (notes): the class that knows most of the keys
            return _from_plain(
                max(classes, key=lambda c: len(set(value) & {f.name for f in dataclasses.fields(c)})), value
            )
        return _from_plain(options[0], value)
    if origin is tuple:
        return tuple(_from_plain(args[0], item) for item in value)
    if origin is list:
        return [_from_plain(args[0], item) for item in value]
    if origin is dict:
        return {key: _from_plain(args[1], item) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Default spec: the design rules
# ---------------------------------------------------------------------------
# The rules below work in millimeters, degrees and grams and are converted to SI
# when the spec is assembled.

_SHEAVE_R = (8.0, 6.5, 5.0)  # joint sheave centerline radii = cable moment arms [mm]
_GUIDE_R = 5.0  # redirect guide centerline radius [mm]
_FLANGE_H = 1.2  # flange rim above the cable centerline [mm]
_FLANGE_GAP = 0.3  # smallest gap between two flange rims [mm]
_WALL = 0.8  # shell wall outside a flange [mm]
_Q_MAX = (90.0, 100.0, 70.0)  # joint ranges [deg]
_CHAMFER_RELIEF = 2.0  # palmar chamfer relief per side at full flexion [deg]

_THETA_IN_S1 = 88.0  # inclination of the span G0 -> S1 [deg]
_THETA_IN_S2 = 68.0  # inclination of the span G1 -> S2 [deg]
_THETA_OUT_S2 = 60.0  # inclination of the span S2 -> G2 [deg]
_TIP_ANCHOR = (7.5, -3.2)  # tip anchor in the distal-link frame [mm]
_REST_GAP = 12.0  # free span slider anchor -> first station at rest, fingers [mm]
_THUMB_REST_GAP = 15.0  # the same for the thumb [mm]
_THUMB_GAMMA = 88.0  # angle between the thumb slider line (the opposition axis) and thumb x [deg]

_Y_DORSAL = (-10.5, -8.5, -6.0)  # link cross-sections, the same on every digit [mm]
_Y_PALMAR = (8.5, 7.0, 6.0)
_WIDTH = (16.0, 15.0, 14.0)
_PAD_R = (7.0, 6.5, 6.0)
_TIP_R = 6.0  # fingertip rounding radius [mm]
_PAD_PROUD = 1.5  # pad surface above the palmar face [mm]
_PAD_MARGIN = 0.5  # pad-to-bisector margin at full flexion [mm]
_TIP_PAD_BACK = 6.0  # distal pad end center this far before the fingertip end [mm]

# name: (link lengths [mm], link masses [g])
_DIGITS = {
    "index": ((45.0, 28.0, 22.0), (8.7, 5.0, 3.3)),
    "middle": ((50.0, 31.0, 23.0), (9.4, 5.4, 3.2)),
    "ring": ((46.0, 29.0, 22.0), (8.4, 4.9, 3.2)),
    "little": ((36.0, 27.0, 20.0), (6.1, 4.2, 2.7)),
    "thumb": ((42.0, 32.0, 27.0), (9.1, 6.2, 4.2)),
}

# finger mounts in the palm frame: MCP origin (x, y, z) [mm], splay about +y [deg]
_FINGER_MOUNT = {
    "index": ((92.0, 0.0, 34.0), -5.0),
    "middle": ((95.0, 0.0, 10.5), 0.0),
    "ring": ((90.0, 1.5, -10.5), 5.0),
    "little": ((82.5, 4.5, -31.5), 10.0),
}

_THUMB_HINGE_POINT = (22.0, 4.0, 56.0)  # where the thumb slider line touches its first station [mm]
_THUMB_AXIS_CANT = 18.0  # hinge axis yaw: its distal end swings toward the thumb side [deg]
_THUMB_AXIS_TILT = 6.0  # hinge axis elevation toward palmar [deg]
_THUMB_PHI_MAX = 130.0  # opposition range [deg]
# box frame of the thenar housing and of the opposition motor: along the tilted hinge line, up, across
_THENAR_AXES = ((0.945847, 0.104528, 0.307324), (-0.099412, 0.994522, -0.032301), (-0.309017, 0.0, 0.951057))
_UNITS_REACH = 125.0  # the index and middle columns reach to the rear face of the knuckle units [mm]

_PALM_Y = (-17.0, 10.0)  # palm thickness range about the MCP axis plane [mm]
_STUB_LEN = 90.0
_STUB_Z = (-25.5, 32.5)
_THENAR_HALF_W = 8.5
_ROOT_BEHIND = 10.5
_ROOT_AHEAD = 10.5
_PALM_PAD_X = (72.0, 46.0)
_FINGER_UNIT = (62.0, 24.0, 12.0)  # folded finger actuator: length (digit x), height (y), width (z) [mm]
_FINGER_UNIT_FRONT = 4.3
_FINGER_UNIT_CABLE = 2.16
_THUMB_UNIT_DIA = 12.0
_THUMB_UNIT_LEN = 94.0
_THUMB_UNIT_FRONT = 11.5
_OPP_MOTOR = (36.0, 16.0, 16.0)

_SPRING_DAMPING = (1.0e-3, 4.0e-4, 1.5e-4)  # joint damping [N*m*s/rad]
_LIMIT_KE = 300.0  # joint stop stiffness [N*m/rad]: 0.07 deg of deflection at the tension limit
_LIMIT_KD = 0.01  # joint stop damping [N*m*s/rad]
STRUCTURAL_JOINT_KE = 1.0e5
"""Structural joint stiffness of the solver for both cable models [N/m and N*m/rad].

Each flexion hinge opens by the tension of every cable that passes it over this
stiffness: measured on the tendon hand, 0.40 mm at 40 N and 0.60 mm at 60 N for
one cable, twice that at the first joint of a two-cable finger. The hinge
compliance of the spec (``spec.cable.hinge_compliance``, part of the closed-form
travel) is its inverse, and :func:`rod_solver_settings` and
:func:`tendon_solver_settings` both pass it to the solver.
"""
_T_START = (6.0, 10.0, 16.0)  # frictionless cable tension at which each joint starts to move [N]
_T_FULL = (14.0, 22.0, 30.0)  # ... and at which it is fully flexed [N]

_CABLE_DIAMETER = 1.0  # [mm]
_T_REST = 2.0  # [N]
_T_MAX = 60.0  # [N]
_T_HOLD = 40.0  # [N]
_STROKE = 42.0  # [mm]
_ROD_SEGMENT = 1.0  # rod segment length target [mm]

# The digits below carry two cables. The knuckle cable runs over its own sheave on the first
# joint and ends on the proximal link; the curl cable keeps the route of the single flexor but passes the
# first joint on a small idler. The two cable planes lie either side of the digit mid-plane.
_TWO_CABLE_DIGITS = ("index", "middle")
_CABLE_ORDER = ("curl", "knuckle")  # order of the cables, sliders and actuators of a two-cable digit
_CURL_R = (5.0, 6.5, 5.0)  # curl cable: idler on the first joint, sheaves on the outer joints [mm]
_KNUCKLE_R = 8.0  # knuckle cable sheave on the first joint [mm]

THUMB_ROD_SEGMENT_LENGTH = 0.002
"""Target length of the rod segments of the thumb's cable [m] (the finger cables use the 1 mm of the rod spec).

The full hand with 1 mm rod segments on every cable does not start: the thumb's rod on its light column
diverges in the first frames. With 2 mm segments on the thumb's cable it starts every time."""

CURL_REFERENCE_RATE_LIMIT = 0.024
"""Reference rate of the curl cable's actuator on a two-cable finger [m/s] (the actuators allow 0.030).

When the curl cable bends the first joint, the knuckle carriage must take up 8 mm of cable per radian
while the curl carriage pays in 5 mm per radian. With both references at the actuator limit the knuckle
cable goes slack during a fast curl step (lowest tension 0.00 N in a step to 60 N); with the curl
reference at 24 mm/s it stays at about 1 N, at the price of curl moves that take about 20 % longer."""
_CURL_THETA_IN_S1 = 95.0  # inclination of the curl span G0 -> S1 [deg]
_KNUCKLE_ANCHOR = (30.0, -8.0)  # end of the knuckle cable in the proximal-link frame [mm]
# front face of the two actuator units behind the first joint axis (the knuckle unit sits behind the curl
# unit); a slider anchor at rest is `_FINGER_UNIT_FRONT` behind the front face of its unit [mm]
_UNIT_FRONT_TWO_CABLE = {"knuckle": 81.0, "curl": 21.0}
_UNIT_LENGTH_TWO_CABLE = {"knuckle": 42.0, "curl": 57.0}  # unit lengths: stroke + 20 mm [mm]
_KNUCKLE_UNIT_Z = {"index": 1.5, "middle": -1.0}  # sideways place of the knuckle unit in its finger (packaging) [mm]
_CABLE_PLANE_Z = {"knuckle": -1.5, "curl": 1.5}  # cable planes about the digit mid-plane [mm]
_KNUCKLE_SPRING = (0.120, 0.080)  # first-joint spring: preload [N*m] and stiffness [N*m/rad]
_KNUCKLE_SPRING_DAMPING = 1.4e-3  # first-joint damping of a two-cable digit [N*m*s/rad]
_CURL_T_START = (4.0, 8.0)  # curl tension at which the second / third joint starts to move [N]
_CURL_T_FULL = (12.0, 18.0)  # ... and at which it is fully flexed [N]
_STROKE_TWO_CABLE = {"knuckle": 22.0, "curl": 37.0}  # actuator strokes [mm]


def _m(value):
    """Convert millimeters to meters, rounded to 1e-9 m."""
    if isinstance(value, (tuple, list)):
        return tuple(round(v * _MM, 9) for v in value)
    return round(value * _MM, 9)


def _default_digit(name: str) -> tuple[DigitSpec, tuple[float, float]]:
    """Apply the route, shell, pad and spring rules to one digit.

    Returns the digit (without its mount) and the slider-line tangent point in
    millimeters, which places the thumb on its hinge. The digits of
    ``_TWO_CABLE_DIGITS`` get a knuckle cable and a curl cable.
    """
    lengths, masses = _DIGITS[name]
    thumb = name == "thumb"
    two_cables = name in _TWO_CABLE_DIGITS
    # the radii of the cable that runs to the fingertip (the flexor, or the curl cable of a two-cable digit)
    r1, r2, r3 = _CURL_R if two_cables else _SHEAVE_R
    theta_in_s1 = _CURL_THETA_IN_S1 if two_cables else _THETA_IN_S1
    rg = _GUIDE_R
    l1, l2, _l3 = lengths
    # every guide sits as close to its sheave as the flange rims allow
    d01 = r1 + rg + 2 * _FLANGE_H + _FLANGE_GAP
    d12 = r2 + rg + 2 * _FLANGE_H + _FLANGE_GAP
    d23 = r3 + rg + 2 * _FLANGE_H + _FLANGE_GAP

    # guides are turned about their sheave until the connecting span has the wanted inclination
    b1 = theta_in_s1 - math.degrees(math.asin((r1 + rg) / d01))
    g0 = (-d01 * math.cos(b1 * _DEG), -d01 * math.sin(b1 * _DEG))  # base frame
    b3 = _THETA_IN_S2 - math.degrees(math.asin((r2 + rg) / d12))
    g1 = (l1 - d12 * math.cos(b3 * _DEG), -d12 * math.sin(b3 * _DEG))  # link-1 frame
    b4 = _THETA_OUT_S2 - math.degrees(math.asin((r2 + rg) / d12))
    g2 = (d12 * math.cos(b4 * _DEG), -d12 * math.sin(b4 * _DEG))  # link-2 frame
    if math.hypot(l2 - g2[0], g2[1]) < d23 - 1e-9:
        raise ValueError(f"{name}: middle link too short for G2 next to S3")
    if math.hypot(g1[0], g1[1]) < d01 - 1e-9:
        raise ValueError(f"{name}: proximal link too short for G1 next to S1")

    # slider line. Fingers: parallel to digit x, tangent to the dorsal side of G0. Thumb: the line is
    # the opposition hinge axis and runs straight onto S1 (no G0).
    gamma = _THUMB_GAMMA if thumb else 0.0
    ux, uy = math.cos(gamma * _DEG), math.sin(gamma * _DEG)  # cable travel direction toward the first station
    if thumb:
        tangent_pt = (-r1 * uy, r1 * ux)
    else:
        tangent_pt = (g0[0] + rg * uy, g0[1] - rg * ux)
    gap0 = _THUMB_REST_GAP if thumb else _REST_GAP
    anchor_rest = (tangent_pt[0] - gap0 * ux, tangent_pt[1] - gap0 * uy)
    if two_cables:
        # the curl unit keeps its place behind the first joint; its anchor follows from the unit
        anchor_rest = (-(_UNIT_FRONT_TWO_CABLE["curl"] + _FINGER_UNIT_FRONT), tangent_pt[1])

    stations = [
        StationSpec("A", "anchor", "slider", _m(anchor_rest), 0.0, 0),
        StationSpec("G0", "guide", "base", _m(g0), _m(rg), 1),
        StationSpec("S1", "sheave", "base", _m((0.0, 0.0)), _m(r1), -1, joint=0),
        StationSpec("G1", "guide", "link1", _m(g1), _m(rg), 1),
        StationSpec("S2", "sheave", "link1", _m((l1, 0.0)), _m(r2), -1, joint=1),
        StationSpec("G2", "guide", "link2", _m(g2), _m(rg), 1),
        StationSpec("S3", "sheave", "link2", _m((l2, 0.0)), _m(r3), -1, joint=2),
        StationSpec("T", "anchor", "link3", _m(_TIP_ANCHOR), 0.0, 0),
    ]
    if thumb:
        del stations[1]

    # links: palmar chamfers keep the two sides of a joint apart at full flexion
    cham = [(180.0 - q) / 2.0 - _CHAMFER_RELIEF for q in _Q_MAX]
    knuckle = [r + _FLANGE_H + _WALL for r in _SHEAVE_R]
    links = []
    for i in range(3):
        cp_prox = _Y_PALMAR[i] / math.tan(cham[i] * _DEG)
        cp_dist = _Y_PALMAR[i] / math.tan(cham[i + 1] * _DEG) if i < 2 else 0.0
        shaft_end = lengths[i] if i < 2 else lengths[i] - _TIP_R
        links.append(
            LinkSpec(
                name=("proximal", "middle", "distal")[i],
                length=_m(lengths[i]),
                width=_m(_WIDTH[i]),
                y_dorsal=_m(_Y_DORSAL[i]),
                y_palmar=_m(_Y_PALMAR[i]),
                mass=round(masses[i] * 1.0e-3, 9),
                knuckle_radius_proximal=_m(knuckle[i]),
                knuckle_radius_distal=_m(knuckle[i + 1] if i < 2 else _TIP_R),
                distal_disk_x=_m(shaft_end),
                shaft_end=_m(shaft_end),
                palmar_chamfer_proximal=_m(cp_prox),
                palmar_chamfer_distal=_m(cp_dist),
            )
        )

    # pads: one capsule per link, ends pulled back so that the pad stays on its own side of the
    # joint bisector at full flexion
    pads = []
    for i in range(3):
        rho = _PAD_R[i]
        yc = _Y_PALMAR[i] + _PAD_PROUD - rho
        a_p = (180.0 - _Q_MAX[i]) / 2.0 * _DEG
        x0 = (rho + _PAD_MARGIN + math.cos(a_p) * yc) / math.sin(a_p)
        if i < 2:
            a_d = (180.0 - _Q_MAX[i + 1]) / 2.0 * _DEG
            x1 = lengths[i] - (rho + _PAD_MARGIN + math.cos(a_d) * yc) / math.sin(a_d)
        else:
            x1 = lengths[i] - _TIP_PAD_BACK
        x1 = max(x1, x0)
        pads.append(PadSpec(body=f"link{i + 1}", p0=_m((x0, yc)), p1=_m((x1, yc)), radius=_m(rho)))

    # springs: sized from the cable tensions at which a joint starts to move and is fully flexed. On a
    # two-cable digit the outer joints are sized from the curl tension and the first joint from its torques.
    arms = (r1, r2, r3)
    if two_cables:
        spring_start = [_KNUCKLE_SPRING[0], _CURL_T_START[0] * r2 * _MM, _CURL_T_START[1] * r3 * _MM]
        spring_range = [
            _KNUCKLE_SPRING[1] * _Q_MAX[0] * _DEG,
            (_CURL_T_FULL[0] - _CURL_T_START[0]) * r2 * _MM,
            (_CURL_T_FULL[1] - _CURL_T_START[1]) * r3 * _MM,
        ]
        damping = (_KNUCKLE_SPRING_DAMPING, *_SPRING_DAMPING[1:])
    else:
        spring_start = [_T_START[j] * _SHEAVE_R[j] * _MM for j in range(3)]
        spring_range = [(_T_FULL[j] - _T_START[j]) * _SHEAVE_R[j] * _MM for j in range(3)]
        damping = _SPRING_DAMPING
    joints = []
    for j in range(3):
        joints.append(
            JointSpec(
                name=("mcp", "pip", "dip")[j],
                lower=round(0.0 * _DEG, 9),
                upper=round(_Q_MAX[j] * _DEG, 9),
                sheave_radius=_m(arms[j]),
                spring_preload=spring_start[j],
                spring_stiffness=spring_range[j] / (_Q_MAX[j] * _DEG),
                spring_damping=damping[j],
                limit_ke=_LIMIT_KE,
                limit_kd=_LIMIT_KD,
            )
        )
    slider = SliderSpec(
        rest_anchor=_m(anchor_rest),
        pull_direction=(round(-ux, 12), round(-uy, 12)),
        line_tangent_point=_m(tangent_pt),
        travel_min=_m(-2.0),
        travel_max=_m(_STROKE_TWO_CABLE["curl"] if two_cables else _STROKE),
        mass=round(6.0 * 1.0e-3, 9),
    )
    if not two_cables:
        cables = [DigitCableSpec(SINGLE_CABLE_NAME, SINGLE_CABLE_NAME, "slider", 0.0, _m(arms), stations, slider)]
        if thumb:
            cables[0].rod_segment_length = THUMB_ROD_SEGMENT_LENGTH
    else:
        # knuckle cable: slider, guide G0k next to the sheave S1k on the first joint (placed by the rule of
        # G0), end anchor Tk on the dorsal wall of the proximal link
        rk = _KNUCKLE_R
        d01k = rk + rg + 2 * _FLANGE_H + _FLANGE_GAP
        b1k = _THETA_IN_S1 - math.degrees(math.asin((rk + rg) / d01k))
        g0k = (-d01k * math.cos(b1k * _DEG), -d01k * math.sin(b1k * _DEG))
        tangent_k = (g0k[0], g0k[1] - rg)
        anchor_k = (-(_UNIT_FRONT_TWO_CABLE["knuckle"] + _FINGER_UNIT_FRONT), tangent_k[1])
        knuckle = DigitCableSpec(
            "knuckle",
            "knuckle",
            "slider_knuckle",
            _m(_CABLE_PLANE_Z["knuckle"]),
            _m((rk, 0.0, 0.0)),
            [
                StationSpec("Ak", "anchor", "slider_knuckle", _m(anchor_k), 0.0, 0),
                StationSpec("G0k", "guide", "base", _m(g0k), _m(rg), 1),
                StationSpec("S1k", "sheave", "base", _m((0.0, 0.0)), _m(rk), -1, joint=0),
                StationSpec("Tk", "anchor", "link1", _m(_KNUCKLE_ANCHOR), 0.0, 0),
            ],
            SliderSpec(
                rest_anchor=_m(anchor_k),
                pull_direction=(-1.0, 0.0),
                line_tangent_point=_m(tangent_k),
                travel_min=_m(-2.0),
                travel_max=_m(_STROKE_TWO_CABLE["knuckle"]),
                mass=round(6.0 * 1.0e-3, 9),
            ),
        )
        curl = DigitCableSpec("curl", "curl", "slider", _m(_CABLE_PLANE_Z["curl"]), _m(arms), stations, slider)
        curl.reference_rate_limit = CURL_REFERENCE_RATE_LIMIT
        by_name = {"knuckle": knuckle, "curl": curl}
        cables = [by_name[key] for key in _CABLE_ORDER]
    if not thumb:
        # envelope of the linear unit of every finger cable, in the digit frame (the visual meshes read it)
        for cable in cables:
            ax, ay = cable.slider.rest_anchor
            length = _UNIT_LENGTH_TWO_CABLE[cable.name] if two_cables else _FINGER_UNIT[0]
            front = round(ax + _FINGER_UNIT_FRONT * _MM, 9)
            dorsal = round(ay - _FINGER_UNIT_CABLE * _MM, 9)
            unit = {
                "box": [length * _MM, _FINGER_UNIT[1] * _MM, _FINGER_UNIT[2] * _MM],
                "front_face_x": front,
                "rear_face_x": round(front - length * _MM, 9),
                "dorsal_face_y": dorsal,
                "palmar_face_y": round(dorsal + _FINGER_UNIT[1] * _MM, 9),
                "z_center": _m(_KNUCKLE_UNIT_Z[name]) if cable.name == "knuckle" else 0.0,
            }
            cable.actuator_unit = unit
    digit = DigitSpec(links=links, joints=joints, cables=cables, pads=pads, mount=None)
    return digit, tangent_pt


def _thumb_axis() -> tuple[float, float, float]:
    """Unit direction of the thumb slider line (cable travel toward the thumb) in the palm frame."""
    cant, tilt = _THUMB_AXIS_CANT * _DEG, _THUMB_AXIS_TILT * _DEG
    return (math.cos(cant) * math.cos(tilt), math.sin(tilt), math.sin(cant) * math.cos(tilt))


def _default_thumb_mount(tangent_pt: tuple[float, float]) -> ThumbMountSpec:
    gamma = _THUMB_GAMMA * _DEG
    axis = _thumb_axis()
    hinge_point, phi_max = _THUMB_HINGE_POINT, _THUMB_PHI_MAX
    # radial reference at zero opposition: the part of +z (thumb side) perpendicular to the axis, so the
    # straight thumb lies in the palm plane and points away from the palm
    dot = axis[2]
    er = [0.0 - dot * axis[0], 0.0 - dot * axis[1], 1.0 - dot * axis[2]]
    n = math.sqrt(sum(v * v for v in er))
    er = [v / n for v in er]
    xt = [math.cos(gamma) * axis[i] + math.sin(gamma) * er[i] for i in range(3)]
    yt = [math.sin(gamma) * axis[i] - math.cos(gamma) * er[i] for i in range(3)]
    zt = [xt[1] * yt[2] - xt[2] * yt[1], xt[2] * yt[0] - xt[0] * yt[2], xt[0] * yt[1] - xt[1] * yt[0]]
    # positive opposition lifts the thumb toward the palmar side (+y): rotation axis = er cross y
    lift = (er[1] * 0.0 - er[2] * 1.0, er[2] * 0.0 - er[0] * 0.0, er[0] * 1.0 - er[1] * 0.0)
    sgn = 1.0 if sum(lift[i] * axis[i] for i in range(3)) > 0.0 else -1.0
    px, py = tangent_pt
    origin = [hinge_point[i] - px * xt[i] - py * yt[i] for i in range(3)]
    return ThumbMountSpec(
        base_body="thumb_column",
        hinge_point=_m(hinge_point),
        hinge_axis=tuple(round(sgn * v, 12) for v in axis),
        slider_line_direction=tuple(round(v, 12) for v in axis),
        hinge_lower=0.0,
        hinge_upper=round(phi_max * _DEG, 9),
        frame_origin_at_phi0=_m(origin),
        frame_x_at_phi0=tuple(round(v, 12) for v in xt),
        frame_y_at_phi0=tuple(round(v, 12) for v in yt),
        frame_z_at_phi0=tuple(round(v, 12) for v in zt),
    )


def _default_finger_mount(name: str) -> FingerMountSpec:
    (x, y, z), splay = _FINGER_MOUNT[name]
    s = splay * _DEG
    return FingerMountSpec(
        base_body="palm",
        frame_origin=_m((x, y, z)),
        splay=round(s, 9),
        frame_x=(round(math.cos(s), 12), 0.0, round(-math.sin(s), 12)),
        frame_y=(0.0, 1.0, 0.0),
        frame_z=(round(math.sin(s), 12), 0.0, round(math.cos(s), 12)),
    )


def _default_palm() -> PalmSpec:
    """Palm solid: every finger column has its own height range, the thenar housing its own frame."""
    m = _MM
    y0, y1 = _PALM_Y
    boxes = []
    for name, ((x, y, z), splay) in _FINGER_MOUNT.items():
        # a column of a two-cable finger reaches back to the rear face of its knuckle unit
        length = _UNITS_REACH if name in _TWO_CABLE_DIGITS else x / math.cos(splay * _DEG)
        z0 = -9.0 if name == "little" else -10.5
        z1 = 9.0 if name == "index" else 10.5
        boxes.append(
            PalmBoxSpec(
                name=f"{name}_column",
                x=(round((x - length) * m, 6), x * m),
                z=(round((z + z0) * m, 9), round((z + z1) * m, 9)),
                yaw=round(splay * _DEG, 6),
                pivot=(x * m, y * m, z * m),
                y=(round((y0 + y) * m, 9), round((y1 + y) * m, 9)),
            )
        )
    boxes.append(PalmBoxSpec(name="forearm_stub", x=(-_STUB_LEN * m, 0.0), z=(_STUB_Z[0] * m, _STUB_Z[1] * m)))
    hinge = _THUMB_HINGE_POINT
    boxes.append(
        PalmBoxSpec(
            name="thenar_housing",
            x=(-(_THUMB_UNIT_FRONT + _THUMB_UNIT_LEN + 2.0) * m, -(_ROOT_BEHIND + 0.5) * m),
            z=(-_THENAR_HALF_W * m, _THENAR_HALF_W * m),
            pivot=(hinge[0] * m, hinge[1] * m, hinge[2] * m),
            y=(-_THENAR_HALF_W * m, _THENAR_HALF_W * m),
            axes=[tuple(row) for row in _THENAR_AXES],
        )
    )
    return PalmSpec(
        mass=0.301,
        thickness_y=(_PALM_Y[0] * m, _PALM_Y[1] * m),
        solid_boxes=boxes,
        palmar_face_setback=_PALM_Y[1] / math.tan((45.0 - _CHAMFER_RELIEF) * _DEG) * m,
        thumb_column=ThumbColumnSpec(
            mass=0.020,
            radius=0.007,
            root_behind_hinge_point=_ROOT_BEHIND * m,
            root_ahead_of_hinge_point=_ROOT_AHEAD * m,
            root_running_clearance=0.5 * m,
        ),
        pads=[
            PalmPadSpec(
                "distal_palm",
                (_PALM_PAD_X[0] * m, 5.5 * m, -20.0 * m),
                (_PALM_PAD_X[0] * m, 4.0 * m, 36.0 * m),
                8.0 * m,
            ),
            PalmPadSpec(
                "heel", (_PALM_PAD_X[1] * m, 5.5 * m, -22.0 * m), (_PALM_PAD_X[1] * m, 4.0 * m, 25.0 * m), 8.0 * m
            ),
        ],
    )


def _default_actuator() -> ActuatorSpec:
    cant = _THUMB_AXIS_CANT * _DEG
    hx, hy, hz = (v * _MM for v in _THUMB_HINGE_POINT)
    ax = _thumb_axis()
    # opposition motor: beside the tail of the thumb unit, 15 mm toward the palm center; it follows the
    # tilted hinge line and carries the box frame of the thenar housing
    s_mid = -(_THUMB_UNIT_FRONT + _THUMB_UNIT_LEN - 0.5 * _OPP_MOTOR[0]) * _MM
    inward = (math.sin(cant), 0.0, -math.cos(cant))
    opp_center = [hx + s_mid * ax[0] + 0.015 * inward[0], hy + s_mid * ax[1], hz + s_mid * ax[2] + 0.015 * inward[2]]
    opposition_motor = {
        "center": [round(v, 6) for v in opp_center],
        "size": [v * _MM for v in _OPP_MOTOR],
        "yaw": round(-cant, 6),
        "axes": [list(row) for row in _THENAR_AXES],
    }
    return ActuatorSpec(
        stroke=_STROKE * _MM,
        speed_limit=0.030,
        force_limit=_T_MAX,
        force_min=_T_REST,
        kp=2.0e4,
        kd=20.0,
        # end stops of the stroke: the gains of the slider drive; no run reaches a stop
        stop_ke=2.0e4,
        stop_kd=20.0,
        slider_mass=0.006,
        opposition={
            # direct-drive hinge on the thumb column
            "ke": 20.0,
            "kd": 0.2,
            "torque_limit": 0.8,
            "speed_limit": 3.0,
        },
        carriage_half_length=0.003,
        finger_unit_box=(_FINGER_UNIT[0] * _MM, _FINGER_UNIT[1] * _MM, _FINGER_UNIT[2] * _MM),
        finger_unit_front_offset=_FINGER_UNIT_FRONT * _MM,
        finger_unit_cable_line_above_bottom=_FINGER_UNIT_CABLE * _MM,
        thumb_unit_cylinder={"diameter": _THUMB_UNIT_DIA * _MM, "length": _THUMB_UNIT_LEN * _MM},
        thumb_unit_front_behind_hinge_point=_THUMB_UNIT_FRONT * _MM,
        thumb_unit_motor_length=0.040,
        opposition_motor=opposition_motor,
    )


def default_hand_spec() -> HandSpec:
    """Return the design of the cable-driven hand.

    Station positions, link shells, pads and springs are generated from a small
    set of design rules (guide next to its sheave, span inclinations, tension
    thresholds of the springs), the rest are plain numbers.

    The index and the middle finger carry two cables, each on its own linear
    actuator: a short knuckle cable that bends the first joint and a curl
    cable that bends the two outer joints and passes the first joint on a
    small idler. The ring finger, the little finger and the thumb have one
    flexor cable over all three joints.

    Returns:
        A new :class:`HandSpec`; the caller may modify it freely.
    """
    cable = CableSpec(
        diameter=_CABLE_DIAMETER * _MM,
        radius=0.5 * _CABLE_DIAMETER * _MM,
        EA=2000.0,  # braided polyester cord
        linear_density=0.79e-3,
        bend_rigidity=1.0e-8,
        twist_rigidity=1.0e-8,
        tension_rest=_T_REST,
        tension_hold=_T_HOLD,
        tension_max=_T_MAX,
        friction_default=0.0,
        anchor_compliance=2.0e-5,
        hinge_compliance=1.0 / STRUCTURAL_JOINT_KE,
    )
    station_hardware = StationHardwareSpec(
        flange_height_above_centerline=_FLANGE_H * _MM,
        flange_rim_gap_min=_FLANGE_GAP * _MM,
        shell_wall=_WALL * _MM,
        groove_half_width=0.8 * _MM,
        flange_thickness=0.6 * _MM,
        pin_radius=1.5 * _MM,
    )
    digits = {}
    for name in _DIGITS:
        digit, tangent_pt = _default_digit(name)
        if name == "thumb":
            digit.mount = _default_thumb_mount(tangent_pt)
        else:
            digit.mount = _default_finger_mount(name)
        digits[name] = digit
    return HandSpec(
        cable=cable,
        station_hardware=station_hardware,
        digits=digits,
        palm=_default_palm(),
        actuator=_default_actuator(),
        pads=PadContactSpec(mu=1.0, ke=1.0e5, gap=0.002, proud=_PAD_PROUD * _MM, table_mu=0.5),
        objects={
            # Power grasp of the can: palm z of the can center and the commanded thumb opposition of the
            # grasp plan. The can is 110 mm long and the hand spans 95 mm from the little finger to the
            # thumb, so the can sits toward the thumb: the thumb pad meets its side 7 to 13 mm below the
            # rim. With the center at 10 mm the thumb landed on top of the can.
            "can": {
                "radius": 0.032,
                "height": 0.110,
                "mass": 0.150,
                "grasp_center_z": 0.025,
                "grasp_opposition": 70.0 * _DEG,
            },
            "ball": {"radius": 0.028, "mass": 0.080},
            "block": {"size": [0.030, 0.030, 0.060], "mass": 0.040},
        },
        rod=RodSpec(
            radius=0.5 * _CABLE_DIAMETER * _MM,
            segment_length_target=_ROD_SEGMENT * _MM,
            contact_ke=1.0e8,
            contact_kd=0.0,
            contact_gap=0.5 * _CABLE_DIAMETER * _MM,
            settings_for_1mm={"reference_inertia": 2.5e-07, "seating_stroke_tensions": [2.0, 16.0, 2.0]},
        ),
        tendon=TendonSpec(
            axis_in_digit_frame=(0.0, 0.0, 1.0),
            orientation_guides=1,
            orientation_sheaves=-1,
            damping=1.0,
        ),
        staircase=StaircaseSpec(
            tensions=(2.0, 8.0, 12.0, 16.0, 20.0, 26.0, 32.0, 60.0),
            hold=0.5,
        ),
    )


# ---------------------------------------------------------------------------
# Kinematics and cable route (planar, in the digit frame)
# ---------------------------------------------------------------------------


def link_frames(digit: DigitSpec, q) -> tuple[np.ndarray, np.ndarray]:
    """Planar frames of the base and the three links in the digit frame.

    Args:
        digit: The digit.
        q: Joint angles (MCP, PIP, DIP) [rad].

    Returns:
        Frame origins [m], shape ``(4, 2)``, and frame angles about digit
        ``+z`` [rad], shape ``(4,)``, in the order of :data:`BODY_KEYS`.
    """
    origins = np.zeros((4, 2))
    angles = np.zeros(4)
    angles[1] = q[0]
    for i in (1, 2):
        a = angles[i]
        origins[i + 1] = origins[i] + digit.links[i - 1].length * np.array([math.cos(a), math.sin(a)])
        angles[i + 1] = a + q[i]
    return origins, angles


def frame_point(origins: np.ndarray, angles: np.ndarray, body: str, point) -> np.ndarray:
    """Map a point from a body frame to the digit frame.

    Args:
        origins: Frame origins from :func:`link_frames` [m].
        angles: Frame angles from :func:`link_frames` [rad].
        body: One of :data:`BODY_KEYS`.
        point: Point ``(x, y)`` in the body frame [m].

    Returns:
        The point in the digit frame [m], shape ``(2,)``.
    """
    i = BODY_KEYS.index(body)
    c, s = math.cos(angles[i]), math.sin(angles[i])
    return origins[i] + np.array([c * point[0] - s * point[1], s * point[0] + c * point[1]])


def station_centers(digit: DigitSpec, q=(0.0, 0.0, 0.0), travel: float = 0.0, cable=None) -> np.ndarray:
    """Centers of all stations of one cable of a digit in the digit frame.

    Args:
        digit: The digit.
        q: Joint angles [rad].
        travel: Travel of the cable's slider along its pull direction [m].
        cable: Cable name or index; ``None`` for the only cable of a one-cable digit.

    Returns:
        Station centers [m], shape ``(station_count, 2)``, actuator end first.
        The cable lies in the plane ``z = plane_offset`` of the digit frame.
    """
    cable = digit.cable(cable)
    origins, angles = link_frames(digit, q)
    centers = []
    for station in cable.stations:
        if station.body == cable.slider_body:
            centers.append(np.array(station.pos) + travel * np.array(cable.slider.pull_direction))
        else:
            centers.append(frame_point(origins, angles, station.body, station.pos))
    return np.array(centers)


@dataclass
class CableRoute:
    """Tangent-line and arc route of a cable through an ordered list of circles and end points.

    Index ``k`` runs over the stations; span ``k`` connects station ``k`` to
    station ``k + 1``.
    """

    centers: np.ndarray
    """Station centers [m], shape ``(n, 2)``."""
    radii: np.ndarray
    """Cable centerline radius on each station [m], shape ``(n,)``; zero for an end point."""
    senses: np.ndarray
    """Winding sense per station (+1 counterclockwise, -1 clockwise, 0 end point), shape ``(n,)``."""
    p_in: np.ndarray
    """Point where the cable arrives on each station [m], shape ``(n, 2)``; the first row is the start point."""
    p_out: np.ndarray
    """Point where the cable leaves each station [m], shape ``(n, 2)``; the last row is the end point."""
    spans: np.ndarray
    """Free span lengths [m], shape ``(n - 1,)``."""
    wraps: np.ndarray
    """Wrap angle on each station [rad], shape ``(n,)``; zero on the two end points."""
    pieces: list[tuple[float, int, int]] = field(default_factory=list)
    """Path pieces ``(start arc length, kind, index)``: kind 0 is span ``index``, kind 1 the arc on
    station ``index``."""

    @property
    def length(self) -> float:
        """Total centerline length [m]."""
        return float(np.sum(self.spans) + np.sum(self.wraps * self.radii))

    @property
    def total_wrap(self) -> float:
        """Sum of all wrap angles [rad]."""
        return float(np.sum(self.wraps))

    def point_at(self, s: float) -> np.ndarray:
        """Return the centerline point at an arc length from the start.

        Args:
            s: Arc length from the actuator end [m]; clamped to the route.

        Returns:
            The point ``(x, y)`` [m].
        """
        s = min(max(s, 0.0), self.length)
        piece = self.pieces[0]
        for candidate in self.pieces:
            if candidate[0] > s:
                break
            piece = candidate
        start, kind, k = piece
        if kind == 0:
            direction = (self.p_in[k + 1] - self.p_out[k]) / self.spans[k]
            return self.p_out[k] + (s - start) * direction
        rel = self.p_in[k] - self.centers[k]
        angle = math.atan2(rel[1], rel[0]) + self.senses[k] * (s - start) / self.radii[k]
        return self.centers[k] + self.radii[k] * np.array([math.cos(angle), math.sin(angle)])

    def sample(self, count: int) -> np.ndarray:
        """Return centerline points at equal arc-length spacing, both ends included.

        Args:
            count: Number of points.

        Returns:
            The points [m], shape ``(count, 2)``.
        """
        return np.array([self.point_at(s) for s in np.linspace(0.0, self.length, count)])


def route_through(centers, radii, senses) -> CableRoute:
    """Build the tangent-line and arc route through a list of stations.

    Args:
        centers: Station centers [m], shape ``(n, 2)``; the first and last are the cable ends.
        radii: Cable centerline radius on each station [m]; zero for the ends.
        senses: +1 where the cable runs counterclockwise about ``+z``, -1 clockwise, 0 for the ends.

    Returns:
        The route.

    Raises:
        ValueError: If two neighboring stations have no common tangent.
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    senses = np.asarray(senses, dtype=float)
    n = len(centers)
    p_in = centers.copy()
    p_out = centers.copy()
    spans = np.zeros(n - 1)
    for k in range(n - 1):
        d = centers[k + 1] - centers[k]
        dist = math.hypot(d[0], d[1])
        a = senses[k + 1] * radii[k + 1] - senses[k] * radii[k]
        if dist <= abs(a):
            raise ValueError(f"no common tangent between stations {k} and {k + 1}")
        psi = math.atan2(d[1], d[0]) - math.asin(a / dist)  # direction angle of the span
        normal = np.array([math.sin(psi), -math.cos(psi)])
        p_out[k] = centers[k] + senses[k] * radii[k] * normal
        p_in[k + 1] = centers[k + 1] + senses[k + 1] * radii[k + 1] * normal
        spans[k] = np.linalg.norm(p_in[k + 1] - p_out[k])
    # wrap: signed angle from the arriving to the leaving tangent point about +z, times the sense
    wraps = np.zeros(n)
    for k in range(1, n - 1):
        ul = (p_in[k] - centers[k]) / radii[k]
        ur = (p_out[k] - centers[k]) / radii[k]
        wraps[k] = math.atan2(ul[0] * ur[1] - ul[1] * ur[0], ul[0] * ur[0] + ul[1] * ur[1]) * senses[k]
    pieces = []
    s = 0.0
    for k in range(n - 1):
        pieces.append((s, 0, k))
        s += spans[k]
        if 0 < k + 1 < n - 1 and wraps[k + 1] > 0.0:
            pieces.append((s, 1, k + 1))
            s += wraps[k + 1] * radii[k + 1]
    return CableRoute(centers, radii, senses, p_in, p_out, spans, wraps, pieces)


def digit_route(digit: DigitSpec, q=(0.0, 0.0, 0.0), travel: float = 0.0, radii=None, cable=None) -> CableRoute:
    """Route of one cable of a digit at one pose, in the digit frame.

    Args:
        digit: The digit.
        q: Joint angles [rad].
        travel: Travel of the cable's slider along its pull direction [m].
        radii: Optional radius per station [m] that replaces the centerline
            radii (used for the rod start pose).
        cable: Cable name or index; ``None`` for the only cable of a one-cable digit.

    Returns:
        The route from the slider anchor to the end anchor.
    """
    stations = digit.cable(cable).stations
    if radii is None:
        radii = [station.radius for station in stations]
    senses = [station.sense for station in stations]
    return route_through(station_centers(digit, q, travel, cable), radii, senses)


def joint_sheave_stations(digit: DigitSpec, cable=None) -> dict[int, int]:
    """Stations of one cable that are joint sheaves.

    Args:
        digit: The digit.
        cable: Cable name or index; ``None`` for the only cable of a one-cable digit.

    Returns:
        Station index (into the cable's stations) per joint index, for the
        joints the cable has a sheave on, in joint order.
    """
    stations = digit.cable(cable).stations
    found = {station.joint: k for k, station in enumerate(stations) if station.kind == "sheave"}
    return dict(sorted(found.items()))


def sheave_station_indices(digit: DigitSpec, cable=None) -> list[int]:
    """Station index of the sheave of each joint (MCP, PIP, DIP) on a cable that passes all three joints.

    Args:
        digit: The digit.
        cable: Cable name or index; ``None`` for the only cable of a one-cable digit.

    Returns:
        One index into the cable's stations per joint.

    Raises:
        ValueError: If the cable has no sheave on one of the joints (use
            :func:`joint_sheave_stations` for such a cable).
    """
    sheaves = joint_sheave_stations(digit, cable)
    if len(sheaves) != len(digit.joints):
        raise ValueError(f"cable {digit.cable(cable).name!r} has sheaves on joints {list(sheaves)} only")
    return list(sheaves.values())


def moment_arm_matrix(digit: DigitSpec) -> np.ndarray:
    """Moment arm of every cable about every joint.

    The arm of a cable about a joint is the centerline radius of the cable's
    sheave on that joint, and zero where the cable has none.

    Args:
        digit: The digit.

    Returns:
        Arms [m], shape ``(cable_count, 3)``: row ``c`` is cable ``c`` of
        ``digit.cables``, column ``j`` is joint ``j``. Joint torques are
        ``arms.T @ tensions``, slider travels by take-up are ``arms @ q``.
    """
    return np.array([cable.moment_arms for cable in digit.cables], dtype=float)


def cable_takeup(digit: DigitSpec, q, cable=None) -> float:
    """Length of one cable taken up by its joint sheaves, ``sum(r_j q_j)``.

    Args:
        digit: The digit.
        q: Joint angles [rad].
        cable: Cable name or index; ``None`` for the only cable of a one-cable digit.

    Returns:
        The take-up [m].
    """
    arms = moment_arm_matrix(digit)[digit.cable_index(cable)]
    return float(sum(arms[j] * q[j] for j in range(len(digit.joints))))


def digit_frame_in_palm(spec: HandSpec, name: str, opposition: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Pose of a digit frame in the palm frame.

    A finger frame is fixed in the palm: its origin is the mount's
    ``frame_origin`` and it is turned by ``splay`` about palm ``+y`` (positive
    turns digit ``+x`` toward ``-z``). The thumb frame is fixed in the thumb
    column: its pose at zero opposition is turned about the hinge line (through
    ``hinge_point`` along ``hinge_axis``) by the opposition angle.

    Args:
        spec: The hand spec.
        name: Digit name.
        opposition: Thumb opposition angle [rad]; ignored for fingers.

    Returns:
        The origin [m], shape ``(3,)``, and the rotation matrix whose columns
        are the digit ``x``, ``y`` and ``z`` axes in the palm frame.
    """
    mount = spec.digits[name].mount
    if isinstance(mount, ThumbMountSpec):
        origin = np.array(mount.frame_origin_at_phi0)
        rot = np.column_stack([mount.frame_x_at_phi0, mount.frame_y_at_phi0, mount.frame_z_at_phi0])
        hinge = np.array(mount.hinge_point)
        turn = _axis_angle_matrix(np.array(mount.hinge_axis), opposition)
        return hinge + turn @ (origin - hinge), turn @ rot
    s = mount.splay
    rot = np.array([[math.cos(s), 0.0, math.sin(s)], [0.0, 1.0, 0.0], [-math.sin(s), 0.0, math.cos(s)]])
    return np.array(mount.frame_origin), rot


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


def lift_points(points, origin, rotation) -> np.ndarray:
    """Lift planar digit-frame points into a 3D frame.

    Args:
        points: Points ``(x, y)`` in the digit frame [m], shape ``(n, 2)`` or ``(2,)``.
        origin: Digit frame origin in the target frame [m], shape ``(3,)``.
        rotation: Digit frame axes as matrix columns, shape ``(3, 3)``.

    Returns:
        The points in the target frame [m], shape ``(n, 3)`` or ``(3,)``.
    """
    points = np.asarray(points, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    return np.asarray(origin, dtype=float) + points @ rotation[:, :2].T


def frame_to_transform(origin, rotation) -> wp.transform:
    """Convert a frame given by origin and rotation matrix into a :class:`warp.transform`.

    Args:
        origin: Origin of the frame [m].
        rotation: Rotation matrix with the frame axes as columns.

    Returns:
        The transform.
    """
    rot = np.asarray(rotation, dtype=float)
    quat = wp.quat_from_matrix(wp.mat33(*rot.flatten().tolist()))
    return wp.transform(wp.vec3(*np.asarray(origin, dtype=float).tolist()), wp.normalize(quat))


# ---------------------------------------------------------------------------
# Rod data
# ---------------------------------------------------------------------------


@dataclass
class RodStation:
    """Radii of one contact station for the rod model [m]."""

    station: str
    centerline_radius: float
    collision_barrel_radius: float
    """Radius of the collision cylinder the rod wraps on."""
    start_vertex_radius: float
    """Distance of the wrapped rod vertices from the station axis in the start pose."""
    flange_radius: float
    """Rim radius of the two flanges."""


@dataclass
class RodStartPose:
    """Start pose of a rod routed through the stations of a digit."""

    points: np.ndarray
    """Rod vertices from the slider anchor to the tip anchor [m], shape ``(segments + 1, 2)`` or ``(.., 3)``."""
    travel: float
    """Slider travel at which the first vertex is the slider anchor [m]."""
    segment_length: float
    """Distance between consecutive vertices [m]."""


def rod_segments(spec: HandSpec, name: str, cable=None) -> tuple[int, float]:
    """Number of rod segments of one cable of a digit and their length.

    The count is the cable length at extension divided by the target segment
    length, rounded; the actual segment length divides the cable exactly.

    Args:
        spec: The hand spec.
        name: Digit name.
        cable: Cable name or index; ``None`` for the only cable of a one-cable digit.

    Returns:
        Segment count and segment length [m].
    """
    length = digit_route(spec.digits[name], cable=cable).length
    count = round(length / rod_segment_target(spec, name, cable))
    return count, length / count


def rod_segment_target(spec: HandSpec, name: str, cable=None) -> float:
    """Target rod segment length of one cable of a digit.

    Args:
        spec: The hand spec.
        name: Digit name.
        cable: Cable name or index; ``None`` for the only cable of a one-cable digit.

    Returns:
        The cable's own ``rod_segment_length`` if the spec gives one,
        otherwise the common target ``spec.rod.segment_length_target`` [m].
    """
    own = spec.digits[name].cable(cable).rod_segment_length
    return float(spec.rod.segment_length_target if own is None else own)


def rod_barrel_radii(centerline_radius: float, segment_length: float, rod_radius: float) -> tuple[float, float]:
    """Barrel rule of the rod model.

    A chain of straight segments wrapped on a barrel is a polygon. With the
    segment mid-points at ``b = (l / 2) / tan(l / (2 R))`` from the axis the
    chain takes up exactly ``R`` of cable per radian of joint rotation.

    Args:
        centerline_radius: Nominal cable centerline radius ``R`` [m].
        segment_length: Actual rod segment length ``l`` [m].
        rod_radius: Rod radius [m].

    Returns:
        Collision barrel radius ``b - rod_radius`` [m] and start-pose vertex
        radius ``sqrt(b^2 + (l / 2)^2)`` [m].
    """
    b = 0.5 * segment_length / math.tan(segment_length / (2.0 * centerline_radius))
    return b - rod_radius, math.hypot(b, 0.5 * segment_length)


def rod_stations(spec: HandSpec, name: str, cable=None) -> list[RodStation]:
    """Collision barrel, start-pose vertex and flange radii of every contact station of one cable of a digit.

    Args:
        spec: The hand spec.
        name: Digit name.
        cable: Cable name or index; ``None`` for the only cable of a one-cable digit.

    Returns:
        One entry per station with a radius, in route order.
    """
    _, seg = rod_segments(spec, name, cable)
    out = []
    for station in spec.digits[name].cable(cable).stations:
        if station.radius == 0.0:
            continue
        barrel, vertex = rod_barrel_radii(station.radius, seg, spec.cable.radius)
        flange = station.radius + spec.station_hardware.flange_height_above_centerline
        out.append(RodStation(station.name, station.radius, barrel, vertex, flange))
    return out


def rod_start_pose(
    spec: HandSpec, name: str, q=(0.0, 0.0, 0.0), travel: float | None = None, cable=None
) -> RodStartPose:
    """Routed start pose of the rod of one cable of a digit, in the digit frame.

    The vertices lie on the tangent-line and arc route whose station radii are
    the start-pose vertex radii, one segment length apart (straight-line
    distance), so the rod starts unstretched, no vertex is inside a barrel and
    the mid-point of every wrapped segment touches its barrel. The vertices are
    laid from the tip anchor toward the actuator. By default the slider travel
    is solved so that the first vertex is the slider anchor. At extension that
    travel is about 0.01 mm (the polygon route and the ideal route differ
    slightly in length); at a flexed pose it is the cable take-up of that pose.

    Args:
        spec: The hand spec.
        name: Digit name.
        q: Joint angles [rad].
        travel: Slider travel [m]. If ``None``, it is solved as described
            above. With a given travel the first vertex lies on the line of
            the first span, in front of or behind the slider anchor.
        cable: Cable name or index; ``None`` for the only cable of a one-cable digit.

    Returns:
        The start pose with ``segments + 1`` planar vertices ordered from the
        slider anchor to the end anchor.
    """
    digit = spec.digits[name]
    count, seg = rod_segments(spec, name, cable)
    stations = digit.cable(cable).stations
    radii = [0.0 if s.radius == 0.0 else rod_barrel_radii(s.radius, seg, spec.cable.radius)[1] for s in stations]

    def lay(slider_travel):
        route = digit_route(digit, q, slider_travel, radii, cable)
        first_direction = (route.p_in[1] - route.p_out[0]) / route.spans[0]

        def point(s):
            # behind the slider anchor the route continues straight along the first span
            return route.p_out[0] + s * first_direction if s < 0.0 else route.point_at(s)

        s = route.length
        points = [point(s)]
        for _ in range(count):
            # next vertex toward the actuator at straight-line distance `seg`; a chord is never longer than its arc
            prev = points[-1]
            hi, lo = s - seg, s - 1.1 * seg
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if np.linalg.norm(point(mid) - prev) > seg:
                    lo = mid
                else:
                    hi = mid
            s = 0.5 * (lo + hi)
            points.append(point(s))
        return np.array(points[::-1]), s

    if travel is None:
        # `s` is the route length between the slider anchor and the first vertex. The first span lies (almost)
        # along the slider line, so `s` changes one to one with the travel and the iteration converges at once.
        travel = 0.0
        for _ in range(20):
            points, s = lay(travel)
            if abs(s) < 1.0e-13:
                break
            travel -= s
    points, s = lay(travel)
    return RodStartPose(points=points, travel=float(travel), segment_length=seg)


def rod_start_pose_3d(
    spec: HandSpec, name: str, origin, rotation, q=(0.0, 0.0, 0.0), travel=None, cable=None
) -> RodStartPose:
    """Routed rod start pose lifted into a 3D frame, in the plane of its cable.

    Args:
        spec: The hand spec.
        name: Digit name.
        origin: Digit frame origin in the target frame [m].
        rotation: Digit frame axes as matrix columns in the target frame.
        q: Joint angles [rad].
        travel: Slider travel [m], see :func:`rod_start_pose`.
        cable: Cable name or index; ``None`` for the only cable of a one-cable digit.

    Returns:
        The start pose with vertices of shape ``(segments + 1, 3)``.
    """
    pose = rod_start_pose(spec, name, q, travel, cable)
    plane_z = spec.digits[name].cable(cable).plane_z
    points = lift_points(pose.points, origin, rotation)
    if plane_z != 0.0:
        points = points + plane_z * np.asarray(rotation, dtype=float)[:, 2]
    return RodStartPose(points, pose.travel, pose.segment_length)


# ---------------------------------------------------------------------------
# Tendon data
# ---------------------------------------------------------------------------


@dataclass
class TendonLink:
    """One link of the massless tendon of a digit, in the order actuator to tip."""

    station: str
    type: str
    """``"ATTACHMENT"`` or ``"ROLLING"``, the names of the tendon link types."""
    body: str
    """Key of the carrying body: ``"slider"`` or one of :data:`BODY_KEYS`."""
    offset: tuple[float, float, float]
    """Link center in the body frame [m]."""
    radius: float
    """Rolling radius [m]; zero for an attachment."""
    orientation: int
    """+1 for guides, -1 for joint sheaves, with the link axis along digit ``+z``; +1 on attachments."""
    compliance: float
    """Compliance of the span that ends at this link [m/N]; zero on the first link."""


def tendon_links(spec: HandSpec, name: str, cable=None) -> list[TendonLink]:
    """Ordered tendon link list of one cable of a digit.

    The wrapped parts carry no compliance in the tendon model, so the free
    spans share the whole cable compliance in proportion to their length at
    extension: ``span / sum(spans) * (cable length / EA + anchor compliance)``.
    The link centers lie in the cable plane (``z = plane_z`` in the body frames).

    Args:
        spec: The hand spec.
        name: Digit name.
        cable: Cable name or index; ``None`` for the only cable of a one-cable digit.

    Returns:
        One link per station of the cable, actuator end first.
    """
    digit = spec.digits[name]
    cable_spec = digit.cable(cable)
    route = digit_route(digit, cable=cable)
    total = route.length / spec.cable.EA + spec.cable.anchor_compliance
    span_sum = sum(float(v) for v in route.spans)
    out = []
    for k, station in enumerate(cable_spec.stations):
        rolling = station.radius != 0.0
        out.append(
            TendonLink(
                station=station.name,
                type="ROLLING" if rolling else "ATTACHMENT",
                body=station.body,
                offset=(station.pos[0], station.pos[1], cable_spec.plane_z),
                radius=station.radius,
                orientation=int(station.sense) if rolling else 1,
                compliance=(total * float(route.spans[k - 1]) / span_sum) if k > 0 else 0.0,
            )
        )
    return out


def _digit_geometry(spec: HandSpec, name: str, cable=None) -> dict[str, Any]:
    """Derived geometry of one cable of a digit: length, segments, spans, wraps, rod radii and tendon links.

    Args:
        spec: The hand spec.
        name: Digit name.
        cable: Cable name or index; ``None`` for the only cable of a one-cable digit.

    Returns:
        The derived geometry of the cable route by name.
    """
    digit = spec.digits[name]
    q_full = [joint.upper for joint in digit.joints]
    r0 = digit_route(digit, cable=cable)
    rf = digit_route(digit, q_full, cable=cable)
    count, seg = rod_segments(spec, name, cable)
    return {
        "cable_length": r0.length,
        "segments": count,
        "segment_length": seg,
        "free_spans": [float(v) for v in r0.spans],
        "wrap_extension": [float(v) for v in r0.wraps],
        "wrap_full_flexion": [float(v) for v in rf.wraps],
        "total_wrap_extension": r0.total_wrap,
        "total_wrap_full_flexion": rf.total_wrap,
        "takeup_full_flexion": cable_takeup(digit, q_full, cable),
        "rod_stations": [_to_plain(station) for station in rod_stations(spec, name, cable)],
        "tendon_links": [_to_plain(link) for link in tendon_links(spec, name, cable)],
    }


# ---------------------------------------------------------------------------
# Closed-form statics
# ---------------------------------------------------------------------------
# Return spring at joint j: torque = preload_j + k_j q_j, toward extension. Sheave S_j is carried by the
# body on the palm side of joint j, so the cable torque about joint j is r_j times the tension of the span
# that LEAVES S_j.


def cable_tensions(digit: DigitSpec, tension) -> list[float]:
    """Normalize a tension argument to one tension per cable of a digit.

    Args:
        digit: The digit.
        tension: One tension [N] (a digit with one cable only), one tension
            per cable in the order of ``digit.cables``, or a dict by cable name.

    Returns:
        One tension per cable [N].

    Raises:
        ValueError: If the number of tensions does not match the cables.
    """
    if isinstance(tension, dict):
        if set(tension) != set(digit.cable_names):
            raise ValueError(f"tensions for {sorted(tension)} given, the digit has {digit.cable_names}")
        return [float(tension[name]) for name in digit.cable_names]
    if np.ndim(tension) == 0:
        if len(digit.cables) != 1:
            raise ValueError(
                f"one tension given for a digit with {len(digit.cables)} cables ({', '.join(digit.cable_names)}); "
                "give one tension per cable"
            )
        return [float(tension)]
    if len(tension) != len(digit.cables):
        raise ValueError(f"{len(tension)} tensions given for {len(digit.cables)} cables")
    return [float(v) for v in tension]


def cable_joint_torques(digit: DigitSpec, tension) -> list[float]:
    """Flexion torque of the cables about every joint, ``sum_c arm_cj T_c``.

    Args:
        digit: The digit.
        tension: Cable tension(s) [N], see :func:`cable_tensions`.

    Returns:
        One torque per joint [N*m].
    """
    tensions = cable_tensions(digit, tension)
    arms = moment_arm_matrix(digit)
    return [float(sum(t * arms[c, j] for c, t in enumerate(tensions))) for j in range(len(digit.joints))]


def frictionless_angles(digit: DigitSpec, tension) -> list[float]:
    """Joint angles at given cable tensions without friction and gravity.

    Args:
        digit: The digit.
        tension: Cable tension [N] of a one-cable digit, or one tension per
            cable (see :func:`cable_tensions`).

    Returns:
        ``clip((sum_c T_c r_cj - preload_j) / k_j, lower_j, upper_j)`` per joint [rad].
    """
    torque = cable_joint_torques(digit, tension)
    return [
        float(np.clip((torque[i] - j.spring_preload) / j.spring_stiffness, j.lower, j.upper))
        for i, j in enumerate(digit.joints)
    ]


def frictionless_row(spec: HandSpec, name: str, tension) -> dict[str, Any]:
    """Frictionless equilibrium of a digit at given cable tensions.

    The slider of a cable travels by the cable's take-up ``sum_j r_cj q_j``
    (on a digit with several cables this couples the sliders: a cable that
    passes a joint on a sheave follows that joint whichever cable moves it),
    by the stretch of the cable and of its anchors, and by the opening of the
    flexion hinges the cable passes. A hinge carries the tensions of all
    cables that have a sheave on it.

    Args:
        spec: The hand spec.
        name: Digit name.
        tension: Cable tension [N] of a one-cable digit, or one tension per
            cable (see :func:`cable_tensions`).

    Returns:
        A dict with ``T`` [N], ``q`` [rad], ``takeup`` [m], ``cable_stretch``
        [m], ``anchor_stretch`` [m], ``hinge_stretch`` [m] (the opening of the
        flexion hinges the cable passes) and the slider ``travel`` [m]; stretch
        and travel are measured from the state with every cable at the rest
        tension. With one tension given as a number the entries other than
        ``q`` are numbers, otherwise lists with one value per cable.
    """
    digit = spec.digits[name]
    cable = spec.cable
    tensions = cable_tensions(digit, tension)
    q = frictionless_angles(digit, tensions)
    arms = moment_arm_matrix(digit)
    # opening of every hinge: the tensions (above rest) of the cables with a sheave on it
    opening = [
        cable.hinge_compliance * sum(t - cable.tension_rest for c, t in enumerate(tensions) if arms[c, j] != 0.0)
        for j in range(len(digit.joints))
    ]
    rows = {key: [] for key in ("takeup", "cable_stretch", "anchor_stretch", "hinge_stretch", "travel")}
    for c, t in enumerate(tensions):
        takeup = cable_takeup(digit, q, c)
        stretch = (t - cable.tension_rest) * digit_route(digit, cable=c).length / cable.EA
        anchor = (t - cable.tension_rest) * cable.anchor_compliance
        hinge = sum(opening[j] for j in range(len(digit.joints)) if arms[c, j] != 0.0)
        rows["takeup"].append(takeup)
        rows["cable_stretch"].append(float(stretch))
        rows["anchor_stretch"].append(float(anchor))
        rows["hinge_stretch"].append(float(hinge))
        rows["travel"].append(float(takeup + stretch + anchor + hinge))
    if np.ndim(tension) == 0 and not isinstance(tension, dict):
        return {"T": tension, "q": q, **{key: value[0] for key, value in rows.items()}}
    return {"T": tensions, "q": q, **rows}


def cable_tensions_for_knuckle(
    spec: HandSpec, name: str, curl_tension: float, knuckle_angle: float | None = None, knuckle_torque: float = 0.0
) -> dict[str, float]:
    """Tensions of a two-cable digit for a wanted first-joint state at a given curl tension.

    The curl cable passes the first joint on its idler and adds
    ``arm_curl * curl_tension`` there. The knuckle tension is lowered by that
    cross term so that the first joint sees the wanted torque:
    ``T_knuckle = (spring torque at knuckle_angle + knuckle_torque - arm_curl * T_curl) / arm_knuckle``.

    Args:
        spec: The hand spec.
        name: Name of a digit with a ``"curl"`` and a ``"knuckle"`` cable.
        curl_tension: Tension of the curl cable [N]; it sets the two outer joints.
        knuckle_angle: Wanted first-joint angle [rad]; ``None`` keeps the
            first joint on its lower stop with the spring preload just balanced.
        knuckle_torque: Extra flexion torque wanted at the first joint beyond
            the spring torque [N*m], e.g. to press with the fingertip.

    Returns:
        ``{"curl": T_curl, "knuckle": T_knuckle}`` [N], each clipped to the
        range from the rest tension to the tension limit.
    """
    digit = spec.digits[name]
    arms = moment_arm_matrix(digit)
    curl, knuckle = digit.cable_index("curl"), digit.cable_index("knuckle")
    joint = digit.joints[0]
    lo, hi = spec.cable.tension_rest, spec.cable.tension_max
    curl_tension = float(np.clip(curl_tension, lo, hi))
    angle = joint.lower if knuckle_angle is None else knuckle_angle
    wanted = joint.spring_preload + joint.spring_stiffness * (angle - joint.lower) + knuckle_torque
    tension = (wanted - arms[curl, 0] * curl_tension) / arms[knuckle, 0]
    return {"curl": curl_tension, "knuckle": float(np.clip(tension, lo, hi))}


_TWO_CABLE_POSES = ("straight", "knuckle_only", "hook", "full_fist", "pinch_ready")
"""Names of the poses of :func:`two_cable_poses`."""

TWO_CABLE_STAIRCASE = {
    "knuckle": (2.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 60.0),
    "curl": (2.0, 6.0, 9.0, 12.0, 15.0, 18.0, 26.0, 34.0, 60.0),
}
"""Tension levels on the way up of the staircase on each cable alone [N] (the other cable at rest)."""

TWO_CABLE_GRID = {"knuckle": (2.0, 16.0, 22.0, 28.0, 40.0), "curl": (2.0, 6.0, 10.0, 14.0, 18.0, 30.0)}
"""Tension levels of the grid of combined tensions [N]; the curl tension runs fastest."""

TWO_CABLE_KNUCKLE_ONLY_ANGLE = math.pi / 3.0
"""First-joint angle of the pose ``knuckle_only`` [rad]."""

TWO_CABLE_PINCH_READY = (0.654498, 2.0)
"""Pose ``pinch_ready``: second-joint angle [rad] and the knuckle tension beyond a fully bent first joint [N]."""


def two_cable_poses(spec: HandSpec, name: str) -> dict[str, dict[str, float]]:
    """Named poses of a two-cable digit as cable tensions.

    ``straight``: both cables at rest. ``knuckle_only``: the first joint at
    60 deg with straight outer joints. ``hook``: straight first joint, outer
    joints fully curled (the knuckle cable at rest; the curl tension stays
    below the level that moves the first joint). ``full_fist``: both cables
    at the hold tension, every joint on its upper stop. ``pinch_ready``: the
    first joint held on its upper stop with a margin, the second joint
    partly bent, the fingertip joint straight.

    Args:
        spec: The hand spec.
        name: Name of a digit with a ``"curl"`` and a ``"knuckle"`` cable.

    Returns:
        ``{"curl": T, "knuckle": T}`` [N] per pose name; feed them to
        :func:`frictionless_row` for the reference angles and travels.
    """
    digit = spec.digits[name]
    arms = moment_arm_matrix(digit)
    curl = digit.cable_index("curl")
    rest, hold = spec.cable.tension_rest, spec.cable.tension_hold

    def curl_tension_for(joint_index, angle):
        joint = digit.joints[joint_index]
        return (joint.spring_preload + joint.spring_stiffness * (angle - joint.lower)) / arms[curl, joint_index]

    pinch = cable_tensions_for_knuckle(spec, name, curl_tension_for(1, TWO_CABLE_PINCH_READY[0]), digit.joints[0].upper)
    pinch["knuckle"] += TWO_CABLE_PINCH_READY[1]
    return {
        "straight": {"curl": rest, "knuckle": rest},
        "knuckle_only": cable_tensions_for_knuckle(spec, name, rest, TWO_CABLE_KNUCKLE_ONLY_ANGLE),
        "hook": {"curl": max(curl_tension_for(j, digit.joints[j].upper) for j in (1, 2)), "knuckle": rest},
        "full_fist": {"curl": hold, "knuckle": hold},
        "pinch_ready": pinch,
    }


def _two_cable_reference(spec: HandSpec, name: str) -> dict[str, list[dict[str, Any]]]:
    """Reference tables of a two-cable digit: each cable alone, combined tensions and the named poses.

    Args:
        spec: The hand spec.
        name: Name of a digit with a ``"curl"`` and a ``"knuckle"`` cable.

    Returns:
        ``{"knuckle_staircase": rows, "curl_staircase": rows, "grid": rows, "poses": rows}``,
        the reference tables of the finger. Each row is a :func:`frictionless_row`
        (lists in the order of ``digit.cables``) with ``T_by_cable`` (the
        tensions by cable name) and, for the poses, ``pose``. A staircase
        lists its levels on the way up, with the other cable at the rest
        tension; run it up and back down over the same levels.
    """
    rest = spec.cable.tension_rest

    def row(tension, **extra):
        return dict(frictionless_row(spec, name, tension), T_by_cable=dict(tension), **extra)

    def stairs(cable):
        other = "curl" if cable == "knuckle" else "knuckle"
        return [row({cable: t, other: rest}) for t in TWO_CABLE_STAIRCASE[cable]]

    grid = [(tc, tk) for tk in TWO_CABLE_GRID["knuckle"] for tc in TWO_CABLE_GRID["curl"]]
    return {
        "knuckle_staircase": stairs("knuckle"),
        "curl_staircase": stairs("curl"),
        "grid": [row({"curl": tc, "knuckle": tk}) for tc, tk in grid],
        "poses": [row(tension, pose=pose) for pose, tension in two_cable_poses(spec, name).items()],
    }


class CapstanStatics:
    """Statics of one digit with capstan friction on every contact station.

    Walking from the actuator, the tension is multiplied by ``exp(-mu wrap)``
    at every station while loading (the cable slides toward the actuator) and
    by ``exp(+mu wrap)`` while unloading. The wrap on the sheave of joint j is
    its wrap at extension minus ``q_j``, so the three joint equations are solved
    one after the other. When the actuator starts to pay out, the slip reverses
    from the actuator end first: the tension of a span is the smaller of its
    value at the peak and the reversed-capstan value, and a joint stays where
    it was at the peak until the reversed front has passed its sheave.

    Args:
        spec: The hand spec.
        name: Digit name.
    """

    LOADING = -1.0
    UNLOADING = 1.0

    def __init__(self, spec: HandSpec, name: str):
        self.digit = spec.digits[name]
        if len(self.digit.cables) != 1:
            # With two cables on the first joint the stick-slip state depends on the order in which the
            # two tensions were changed, so there is no table of one loading and one unloading branch.
            raise ValueError(
                f"{name}: capstan friction statics cover digits with one cable; this digit has "
                f"{len(self.digit.cables)} ({', '.join(self.digit.cable_names)}). Use the frictionless tables."
            )
        self.cable = spec.cable
        route = digit_route(self.digit)
        self.wrap0 = [float(v) for v in route.wraps]
        self.span0 = [float(v) for v in route.spans]
        self.length = route.length
        self.sheaves = sheave_station_indices(self.digit)

    def wraps(self, q) -> list[float]:
        """Wrap angle on every station at given joint angles.

        Args:
            q: Joint angles [rad].

        Returns:
            One wrap angle per station [rad]; zero at the two end anchors.
        """
        w = list(self.wrap0)
        for j, k in enumerate(self.sheaves):
            w[k] = self.wrap0[k] - q[j]
        return w

    def solve(self, tension: float, mu: float, sign: float, q_cap=None) -> list[float]:
        """Joint angles at an actuator tension.

        Args:
            tension: Actuator tension [N].
            mu: Friction coefficient.
            sign: :attr:`LOADING` or :attr:`UNLOADING`.
            q_cap: Joint angles at the peak [rad]; a joint cannot exceed them
                while unloading.

        Returns:
            Joint angles [rad].
        """
        q = [0.0, 0.0, 0.0]
        for j, k in enumerate(self.sheaves):
            joint = self.digit.joints[j]

            def residual(qj, j=j, k=k, joint=joint):
                trial = list(q)
                trial[j] = qj
                w = self.wraps(trial)
                # tension of the span leaving S_j: all stations up to and including S_j have acted
                leaving = tension * math.exp(sign * mu * sum(w[1 : k + 1]))
                return joint.sheave_radius * leaving - joint.spring_preload - joint.spring_stiffness * qj

            lo, hi = joint.lower, joint.upper
            if residual(lo) <= 0.0:
                qj = lo
            elif residual(hi) >= 0.0:
                qj = hi
            else:
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if residual(mid) > 0.0:
                        lo = mid
                    else:
                        hi = mid
                qj = 0.5 * (lo + hi)
            if q_cap is not None:
                qj = min(qj, q_cap[j])
            q[j] = qj
        return q

    def tensions(self, tension: float, mu: float, sign: float, q, peak=None) -> list[float]:
        """Tension of every span [N], actuator side first.

        Args:
            tension: Actuator tension [N].
            mu: Friction coefficient.
            sign: :attr:`LOADING` or :attr:`UNLOADING`.
            q: Joint angles [rad].
            peak: Span tensions at the peak [N] (unloading only).

        Returns:
            One tension per free span [N].
        """
        w = self.wraps(q)
        out = [tension]
        for k in range(1, len(self.wrap0) - 1):
            out.append(out[-1] * math.exp(sign * mu * w[k]))
        if peak is not None:
            out = [min(a, b) for a, b in zip(out, peak, strict=True)]
        return out

    def stretch(self, span_tensions, q) -> float:
        """Cable stretch, anchor deflection and hinge opening [m] for given span tensions.

        An arc carries the mean of its two spans; each anchor carries the
        tension of its end span; the hinge of joint j opens by the hinge
        compliance times the tension of the span that leaves its sheave.

        Args:
            span_tensions: Tension of every free span [N], see :meth:`tensions`.
            q: Joint angles [rad].

        Returns:
            The sum of the three parts [m].
        """
        ea, anchor = self.cable.EA, self.cable.anchor_compliance
        w = self.wraps(q)
        spans = list(self.span0)
        spans[0] += cable_takeup(self.digit, q)  # the actuator span grows by the take-up
        total = sum(t * s for t, s in zip(span_tensions, spans, strict=True)) / ea
        for k in range(1, len(self.wrap0) - 1):
            total += 0.5 * (span_tensions[k - 1] + span_tensions[k]) * self.digit.stations[k].radius * w[k] / ea
        hinge = self.cable.hinge_compliance * sum(span_tensions[k] for k in self.sheaves)
        return total + 0.5 * anchor * (span_tensions[0] + span_tensions[-1]) + hinge

    def row(self, tension: float, mu: float, sign: float, q, stretch_ref: float, peak=None) -> dict[str, Any]:
        """One table row: angles, take-up, stretch, slider travel, tip tension and the joint span tensions.

        Args:
            tension: Actuator tension [N].
            mu: Friction coefficient.
            sign: :attr:`LOADING` or :attr:`UNLOADING`.
            q: Joint angles [rad].
            stretch_ref: Stretch of the reference state [m], subtracted from the stretch of this row.
            peak: Span tensions at the peak [N] (unloading only).

        Returns:
            A dict with ``T`` [N], ``q`` [rad], ``takeup`` [m], ``stretch`` [m],
            ``travel`` [m], ``tip_tension`` [N] and ``joint_span_tension`` [N].
        """
        takeup = cable_takeup(self.digit, q)
        ten = self.tensions(tension, mu, sign, q, peak)
        stretch = self.stretch(ten, q) - stretch_ref
        return {
            "T": tension,
            "q": [float(v) for v in q],
            "takeup": takeup,
            "stretch": float(stretch),
            "travel": float(takeup + stretch),
            "tip_tension": float(ten[-1]),
            "joint_span_tension": [float(ten[k]) for k in self.sheaves],
        }


def friction_tables(spec: HandSpec, name: str, tensions, mu: float) -> dict[str, list[dict[str, Any]]]:
    """Loading and unloading equilibria of a digit with capstan friction.

    Args:
        spec: The hand spec.
        name: Digit name.
        tensions: Actuator tension levels on the way up [N]; the last one is the peak.
        mu: Friction coefficient.

    Returns:
        ``{"loading": rows, "unloading": rows}``; the unloading rows start at
        the peak and visit the levels in reverse. Travel is measured from the
        loading state at the rest tension.
    """
    cap = CapstanStatics(spec, name)
    rest = spec.cable.tension_rest
    q_rest = cap.solve(rest, mu, cap.LOADING)
    ref = cap.stretch(cap.tensions(rest, mu, cap.LOADING, q_rest), q_rest)
    loading = [cap.row(t, mu, cap.LOADING, cap.solve(t, mu, cap.LOADING), ref) for t in tensions]
    q_peak = loading[-1]["q"]
    peak = cap.tensions(tensions[-1], mu, cap.LOADING, q_peak)
    unloading = [
        cap.row(t, mu, cap.UNLOADING, cap.solve(t, mu, cap.UNLOADING, q_cap=q_peak), ref, peak=peak)
        for t in reversed(tensions)
    ]
    return {"loading": loading, "unloading": unloading}


def tip_tension(spec: HandSpec, name: str, tension: float, mu: float, q=None) -> float:
    """Tension that reaches the tip anchor while loading [N].

    Args:
        spec: The hand spec.
        name: Digit name.
        tension: Actuator tension [N].
        mu: Friction coefficient.
        q: Joint angles [rad]; the loading equilibrium at this tension if ``None``.

    Returns:
        The tension of the last span [N].
    """
    cap = CapstanStatics(spec, name)
    if q is None:
        q = cap.solve(tension, mu, cap.LOADING)
    return cap.tensions(tension, mu, cap.LOADING, q)[-1]


def gravity_angles(digit: DigitSpec, tension: float, gravity_direction, gravity: float = GRAVITY) -> list[float]:
    """Frictionless joint angles with gravity acting in the flexion plane.

    Link centers of mass are at mid-length on the link ``x`` axis. The
    actuator carriage hangs on the cable: its weight component along the pull
    direction adds to the drive force, so the cable carries
    ``tension + slider.mass * gravity * dot(pull_direction, gravity_direction)``
    (less than the command when the digit hangs from its base). On a digit
    with several cables this holds for every cable and its own carriage.

    Args:
        digit: The digit.
        tension: Commanded cable tension, the drive force of the actuator [N];
            one per cable on a digit with several cables (see :func:`cable_tensions`).
        gravity_direction: Unit direction of gravity ``(x, y)`` in the digit
            frame, e.g. ``(0, 1)`` for a horizontal digit with the palm down.
        gravity: Gravitational acceleration [m/s^2].

    Returns:
        Joint angles [rad].
    """
    lengths = [link.length for link in digit.links]
    masses = [link.mass for link in digit.links]
    gx, gy = gravity_direction
    tension = [
        t + c.slider.mass * gravity * (c.slider.pull_direction[0] * gx + c.slider.pull_direction[1] * gy)
        for t, c in zip(cable_tensions(digit, tension), digit.cables, strict=True)
    ]
    cable_torque = cable_joint_torques(digit, tension)
    q = frictionless_angles(digit, tension)
    for _ in range(400):
        origins, angles = link_frames(digit, q)
        coms = [
            origins[i + 1] + 0.5 * lengths[i] * np.array([math.cos(angles[i + 1]), math.sin(angles[i + 1])])
            for i in range(3)
        ]
        new = []
        for j, joint in enumerate(digit.joints):
            torque = 0.0
            for i in range(j, 3):
                arm = coms[i] - origins[j + 1]
                torque += masses[i] * gravity * (arm[0] * gy - arm[1] * gx)
            angle = (cable_torque[j] - joint.spring_preload + torque) / joint.spring_stiffness
            new.append(float(np.clip(angle, joint.lower, joint.upper)))
        if max(abs(a - b) for a, b in zip(new, q, strict=True)) < 1.0e-13:
            q = new
            break
        q = [0.5 * (a + b) for a, b in zip(new, q, strict=True)]
    return q


def staircase_levels(spec: HandSpec) -> list[float]:
    """Tension levels of the staircase, up and then the same levels back down.

    Args:
        spec: The hand spec.

    Returns:
        The levels [N]; the peak appears once.
    """
    up = list(spec.staircase.tensions)
    return up + up[-2::-1]


def staircase_table(spec: HandSpec, name: str, mu: float = 0.0) -> list[dict[str, Any]]:
    """Reference equilibria of a digit at every level of the tension staircase.

    Args:
        spec: The hand spec.
        name: Digit name.
        mu: Cable friction coefficient.

    Returns:
        One row per level of :func:`staircase_levels` with at least ``T`` [N],
        ``q`` [rad], ``takeup`` [m], ``travel`` [m], ``tip_tension`` [N] and
        ``branch`` (``"loading"`` or ``"unloading"``).
    """
    tensions = list(spec.staircase.tensions)
    if mu == 0.0:
        up = [dict(frictionless_row(spec, name, t), tip_tension=t, branch="loading") for t in tensions]
        down = [dict(row, branch="unloading") for row in up[-2::-1]]
        return up + down
    tables = friction_tables(spec, name, tensions, mu)
    up = [dict(row, branch="loading") for row in tables["loading"]]
    down = [dict(row, branch="unloading") for row in tables["unloading"][1:]]
    return up + down


# ---------------------------------------------------------------------------
# Rigid skeleton shared by both cable models
# ---------------------------------------------------------------------------


@dataclass
class DigitSkeleton:
    """Indices and controller data of one digit added by :func:`add_digit_skeleton`.

    Sign conventions: a positive flexion joint coordinate is flexion (rotation
    about digit ``+z``); a positive slider coordinate is travel along the pull
    direction, i.e. it pulls the cable and closes the digit. The return
    springs act toward extension.
    """

    name: str
    bodies: dict[str, int]
    """Body index per key: ``"base"``, ``"link1"``, ``"link2"``, ``"link3"`` and the sliders.

    ``"slider"`` is the carriage of the first cable; the carriage of every
    further cable has the key ``"slider_<cable name>"`` (see :meth:`slider_key`).
    """
    base_xform_digit: wp.transform
    """Pose of the digit frame in the base body frame."""
    joint_flexion: tuple[int, int, int]
    """Joint indices of MCP, PIP, DIP."""
    cable_names: tuple[str, ...]
    """Names of the digit's cables; the sliders below are in this order."""
    joint_sliders: tuple[int, ...]
    """Joint index of the slider of every cable."""
    q_flexion: tuple[int, int, int]
    """Indices of the flexion angles in ``joint_q``."""
    q_sliders: tuple[int, ...]
    """Index of the slider travel of every cable in ``joint_q``."""
    dof_flexion: tuple[int, int, int]
    """Indices of the flexion joints in ``joint_qd``, ``joint_f`` and the drive arrays."""
    dof_sliders: tuple[int, ...]
    """Index of the slider of every cable in ``joint_qd``, ``joint_f`` and the drive arrays."""
    flexion_lower: np.ndarray
    """Lower joint limits [rad], shape ``(3,)``."""
    flexion_upper: np.ndarray
    """Upper joint limits [rad], shape ``(3,)``."""
    spring_preload: np.ndarray
    """Spring torque at the lower limit, toward extension [N*m], shape ``(3,)``."""
    spring_stiffness: np.ndarray
    """Spring stiffness, set as the drive ``target_ke`` [N*m/rad], shape ``(3,)``."""
    spring_damping: np.ndarray
    """Joint damping, set as the drive ``target_kd`` [N*m*s/rad], shape ``(3,)``."""
    spring_target_q: np.ndarray
    """Drive target of the springs, the lower joint limits [rad], shape ``(3,)``."""
    joint_f_bias: np.ndarray
    """Constant ``Control.joint_f`` entries that apply the preload, ``-spring_preload`` [N*m], shape ``(3,)``."""
    sheave_radius: np.ndarray
    """Moment arm per joint of the first cable, the one that runs to the fingertip [m], shape ``(3,)``."""
    moment_arms: np.ndarray
    """Moment arm of every cable about every joint [m], shape ``(cable_count, 3)``."""
    slider_travel_mins: tuple[float, ...]
    """Lower slider limit of every cable [m]."""
    slider_travel_maxs: tuple[float, ...]
    """Upper slider limit of every cable [m]."""
    slider_kp: float
    """Slider drive stiffness [N/m]."""
    slider_kd: float
    """Slider drive damping [N*s/m]."""
    base_joint: int | None = None
    """Joint that holds the base body, if the digit was added with its own fixed base."""

    @property
    def joints(self) -> list[int]:
        """All joints of the digit in the order they were added."""
        return [*self.joint_flexion, *self.joint_sliders]

    def cable_index(self, cable: int | str | None = None) -> int:
        """Index of a cable in :attr:`cable_names`.

        Args:
            cable: Cable name or index; ``None`` selects the only cable of a one-cable digit.

        Returns:
            The index.

        Raises:
            ValueError: If ``cable`` is ``None`` and the digit has several cables.
        """
        if cable is None:
            if len(self.cable_names) != 1:
                raise ValueError(
                    f"{self.name} has {len(self.cable_names)} cables ({', '.join(self.cable_names)}); "
                    "name the cable or use the per-cable tuples of the skeleton"
                )
            return 0
        if isinstance(cable, str):
            return self.cable_names.index(cable)
        return int(cable)

    def slider_key(self, cable: int | str | None = None) -> str:
        """Key of a cable's slider in :attr:`bodies`.

        Args:
            cable: Cable name or index; ``None`` selects the only cable of a one-cable digit.

        Returns:
            ``"slider"`` for the first cable, ``"slider_<cable name>"`` for every further cable.
        """
        index = self.cable_index(cable)
        return "slider" if index == 0 else f"slider_{self.cable_names[index]}"

    def slider_body(self, cable: int | str | None = None) -> int:
        """Body index of a cable's slider.

        Args:
            cable: Cable name or index; ``None`` selects the only cable of a one-cable digit.

        Returns:
            The body index.
        """
        return self.bodies[self.slider_key(cable)]

    @property
    def joint_slider(self) -> int:
        """Joint index of the slider of a one-cable digit (raises for a digit with several cables)."""
        return self.joint_sliders[self.cable_index()]

    @property
    def q_slider(self) -> int:
        """Index of the slider travel of a one-cable digit in ``joint_q`` (raises for several cables)."""
        return self.q_sliders[self.cable_index()]

    @property
    def dof_slider(self) -> int:
        """Index of the slider of a one-cable digit in ``joint_qd`` and ``joint_f`` (raises for several cables)."""
        return self.dof_sliders[self.cable_index()]

    @property
    def slider_travel_min(self) -> float:
        """Lower slider limit of a one-cable digit [m] (raises for several cables)."""
        return self.slider_travel_mins[self.cable_index()]

    @property
    def slider_travel_max(self) -> float:
        """Upper slider limit of a one-cable digit [m] (raises for several cables)."""
        return self.slider_travel_maxs[self.cable_index()]


def _box_inertia(mass: float, lx: float, ly: float, lz: float) -> wp.mat33:
    """Inertia of a solid box with edge lengths ``lx, ly, lz`` [m] about its center [kg*m^2]."""
    c = mass / 12.0
    return wp.mat33(
        c * (ly * ly + lz * lz), 0.0, 0.0, 0.0, c * (lx * lx + lz * lz), 0.0, 0.0, 0.0, c * (lx * lx + ly * ly)
    )


def _planar_transform(point, angle: float) -> wp.transform:
    return wp.transform(
        wp.vec3(float(point[0]), float(point[1]), 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(angle))
    )


def add_digit_skeleton(
    builder: newton.ModelBuilder,
    spec: HandSpec,
    name: str,
    base: int,
    base_xform_digit: wp.transform | None = None,
    q=(0.0, 0.0, 0.0),
    travel: float = 0.0,
    label: str | None = None,
) -> DigitSkeleton:
    """Add the rigid skeleton of one digit to a builder.

    Adds the three links on revolute MCP, PIP and DIP joints about the flexion
    axis and the slider on a prismatic joint along the pull direction. Body
    frames are the link frames of the module docstring; the slider body frame
    coincides with the digit frame at zero travel, so every station offset of
    the spec is valid in the frame of its body.

    The return springs are set up for solvers that clamp drive targets into the
    joint limits and ignore passive joint stiffness: the spring stiffness is the
    drive ``target_ke`` with ``target_q`` at the lower limit, the damping is
    ``target_kd``, and the preload is returned as
    :attr:`DigitSkeleton.joint_f_bias` for the caller to write into
    ``Control.joint_f`` (see :func:`write_spring_preload`). The slider drive
    uses the actuator gains with a zero target; the actuator governor moves the
    target at run time.

    The joints are not added to an articulation; the caller does that after it
    has added the cable.

    Args:
        builder: The model builder.
        spec: The hand spec.
        name: Digit name.
        base: Index of the body that carries the digit (palm or thumb column), or -1 for the world.
        base_xform_digit: Pose of the digit frame in the base body frame; identity if ``None``.
        q: Initial joint angles [rad].
        travel: Initial slider travel [m]; one value for all sliders or one per cable.
        label: Label prefix of the new bodies and joints; the digit name if ``None``.

    Returns:
        The indices and controller data of the digit.
    """
    digit = spec.digits[name]
    label = name if label is None else label
    if base_xform_digit is None:
        base_xform_digit = wp.transform_identity()
    base_world = builder.body_q[base] if base >= 0 else wp.transform_identity()
    digit_world = wp.transform_multiply(wp.transform(*base_world), base_xform_digit)
    origins, angles = link_frames(digit, q)
    z_axis = wp.vec3(0.0, 0.0, 1.0)

    bodies = {"base": base}
    joint_flexion = []
    parent = base
    for i, (link, joint) in enumerate(zip(digit.links, digit.joints, strict=True)):
        key = BODY_KEYS[i + 1]
        # Link inertia: solid box of the link's length, height and width; center of mass at mid-length on
        # the link x axis (the convention of the closed-form gravity table).
        body = builder.add_link(
            xform=wp.transform_multiply(digit_world, _planar_transform(origins[i + 1], angles[i + 1])),
            com=wp.vec3(0.5 * link.length, 0.0, 0.0),
            inertia=_box_inertia(link.mass, link.length, link.y_palmar - link.y_dorsal, link.width),
            mass=link.mass,
            label=f"{label}_{key}",
        )
        if i == 0:
            parent_xform = base_xform_digit
        else:
            parent_xform = wp.transform(wp.vec3(digit.links[i - 1].length, 0.0, 0.0), wp.quat_identity())
        joint_index = builder.add_joint_revolute(
            parent=parent,
            child=body,
            parent_xform=parent_xform,
            child_xform=wp.transform_identity(),
            axis=z_axis,
            limit_lower=joint.lower,
            limit_upper=joint.upper,
            limit_ke=joint.limit_ke,
            limit_kd=joint.limit_kd,
            target_q=joint.lower,
            target_ke=joint.spring_stiffness,
            target_kd=joint.spring_damping,
            actuator_mode=newton.JointTargetMode.POSITION,
            friction=0.0,
            label=f"{label}_{joint.name}",
        )
        builder.joint_q[builder.joint_q_start[joint_index]] = float(q[i])
        bodies[key] = body
        joint_flexion.append(joint_index)
        parent = body

    # Sliders: one carriage of a linear actuator per cable, on the base along the cable's pull direction.
    # The first cable's carriage keeps the names of a one-cable digit ("slider"); further ones carry the cable name.
    travels = [float(travel)] * len(digit.cables) if np.ndim(travel) == 0 else [float(v) for v in travel]
    if len(travels) != len(digit.cables):
        raise ValueError(f"{name}: {len(travels)} slider travels given for {len(digit.cables)} cables")
    half = spec.actuator.carriage_half_length
    joint_sliders = []
    for c, (cable, cable_travel) in enumerate(zip(digit.cables, travels, strict=True)):
        slider = cable.slider
        pull = slider.pull_direction
        slider_key = cable.slider_body
        if slider_key != ("slider" if c == 0 else f"slider_{cable.name}"):
            raise ValueError(f"{name}: the carriage of cable {c} must be 'slider' (first) or 'slider_<cable name>'")
        slider_label = f"{label}_{slider_key}"
        slider_body = builder.add_link(
            xform=wp.transform_multiply(
                digit_world, _planar_transform((cable_travel * pull[0], cable_travel * pull[1]), 0.0)
            ),
            com=wp.vec3(slider.rest_anchor[0], slider.rest_anchor[1], cable.plane_z),
            inertia=_box_inertia(slider.mass, 2.0 * half, 2.0 * half, 2.0 * half),
            mass=slider.mass,
            label=slider_label,
        )
        joint_slider = builder.add_joint_prismatic(
            parent=base,
            child=slider_body,
            parent_xform=base_xform_digit,
            child_xform=wp.transform_identity(),
            axis=wp.vec3(pull[0], pull[1], 0.0),
            limit_lower=slider.travel_min,
            limit_upper=slider.travel_max,
            limit_ke=spec.actuator.stop_ke,
            limit_kd=spec.actuator.stop_kd,
            target_q=0.0,
            target_ke=spec.actuator.kp,
            target_kd=spec.actuator.kd,
            actuator_mode=newton.JointTargetMode.POSITION,
            friction=0.0,
            label=f"{slider_label}_axis",
        )
        builder.joint_q[builder.joint_q_start[joint_slider]] = cable_travel
        bodies[slider_key] = slider_body
        joint_sliders.append(joint_slider)

    # The station collision shapes of the rod model are added by add_rod_stations(), the pad collision
    # shapes of a grasp scene by add_hand_pads().

    preload = np.array([joint.spring_preload for joint in digit.joints])
    lower = np.array([joint.lower for joint in digit.joints])
    return DigitSkeleton(
        name=name,
        bodies=bodies,
        base_xform_digit=base_xform_digit,
        joint_flexion=tuple(joint_flexion),
        cable_names=tuple(digit.cable_names),
        joint_sliders=tuple(joint_sliders),
        q_flexion=tuple(builder.joint_q_start[j] for j in joint_flexion),
        q_sliders=tuple(builder.joint_q_start[j] for j in joint_sliders),
        dof_flexion=tuple(builder.joint_qd_start[j] for j in joint_flexion),
        dof_sliders=tuple(builder.joint_qd_start[j] for j in joint_sliders),
        flexion_lower=lower,
        flexion_upper=np.array([joint.upper for joint in digit.joints]),
        spring_preload=preload,
        spring_stiffness=np.array([joint.spring_stiffness for joint in digit.joints]),
        spring_damping=np.array([joint.spring_damping for joint in digit.joints]),
        spring_target_q=lower.copy(),
        joint_f_bias=-preload,
        sheave_radius=np.array([joint.sheave_radius for joint in digit.joints]),
        moment_arms=moment_arm_matrix(digit),
        slider_travel_mins=tuple(cable.slider.travel_min for cable in digit.cables),
        slider_travel_maxs=tuple(cable.slider.travel_max for cable in digit.cables),
        slider_kp=spec.actuator.kp,
        slider_kd=spec.actuator.kd,
    )


def write_spring_preload(joint_f: np.ndarray, skeletons) -> np.ndarray:
    """Write the spring preload of the given digits into a ``joint_f`` array.

    Args:
        joint_f: Generalized joint forces [N or N*m], shape ``(joint_dof_count,)``; modified in place.
        skeletons: One :class:`DigitSkeleton` or a sequence of them.

    Returns:
        ``joint_f``.
    """
    if isinstance(skeletons, DigitSkeleton):
        skeletons = [skeletons]
    for skeleton in skeletons:
        joint_f[list(skeleton.dof_flexion)] = skeleton.joint_f_bias
    return joint_f


def add_single_finger(
    builder: newton.ModelBuilder,
    spec: HandSpec,
    name: str = "index",
    base: str = "kinematic",
    base_xform: wp.transform | None = None,
    q=(0.0, 0.0, 0.0),
    travel: float = 0.0,
) -> DigitSkeleton:
    """Add a base body and the skeleton of one digit to a builder, without articulation.

    The base body frame is the digit frame. With the default pose the flexion
    axis is the world ``+z`` axis, so with gravity along ``-z`` gravity does
    not load the joints and the closed-form tables without gravity apply.

    Args:
        builder: The model builder.
        spec: The hand spec.
        name: Digit name.
        base: ``"kinematic"`` for a kinematic base body the caller may move, or
            ``"fixed"`` for a dynamic base body on a fixed joint to the world.
        base_xform: World pose of the base body; identity if ``None``.
        q: Initial joint angles [rad].
        travel: Initial slider travel [m].

    Returns:
        The digit skeleton; the caller adds the cable and then the
        articulation (see :func:`build_single_finger_model`).
    """
    if base_xform is None:
        base_xform = wp.transform_identity()
    if base == "kinematic":
        base_body = builder.add_link(xform=base_xform, is_kinematic=True, label=f"{name}_base")
    elif base == "fixed":
        size = 0.02
        mass = spec.palm.thumb_column.mass if name == "thumb" else spec.palm.mass
        base_body = builder.add_link(
            xform=base_xform, mass=mass, inertia=_box_inertia(mass, size, size, size), label=f"{name}_base"
        )
    else:
        raise ValueError(f"base must be 'kinematic' or 'fixed', got {base!r}")
    fixed_joint = None
    if base == "fixed":
        fixed_joint = builder.add_joint_fixed(
            parent=-1,
            child=base_body,
            parent_xform=base_xform,
            child_xform=wp.transform_identity(),
            label=f"{name}_base_fixed",
        )
    skeleton = add_digit_skeleton(builder, spec, name, base_body, q=q, travel=travel)
    skeleton.base_joint = fixed_joint
    return skeleton


def build_single_finger_model(
    spec: HandSpec | None = None,
    name: str = "index",
    base: str = "kinematic",
    base_xform: wp.transform | None = None,
    q=(0.0, 0.0, 0.0),
    travel: float = 0.0,
    gravity=(0.0, 0.0, -GRAVITY),
    device=None,
) -> tuple[newton.Model, DigitSkeleton]:
    """Build and finalize the skeleton-only model of one digit.

    This model has no cable; the rod and tendon single-finger examples call
    :func:`add_single_finger`, add their cable and then create the articulation
    themselves.

    Args:
        spec: The hand spec; the default spec if ``None``.
        name: Digit name.
        base: ``"kinematic"`` or ``"fixed"``, see :func:`add_single_finger`.
        base_xform: World pose of the base body; identity if ``None``.
        q: Initial joint angles [rad].
        travel: Initial slider travel [m].
        gravity: Gravity vector in the world frame [m/s^2]; the default is along ``-z``.
        device: Device of the model.

    Returns:
        The finalized model and the digit skeleton.
    """
    if spec is None:
        spec = default_hand_spec()
    builder = newton.ModelBuilder(gravity=gravity)
    skeleton = add_single_finger(builder, spec, name, base, base_xform, q, travel)
    joints = skeleton.joints
    if skeleton.base_joint is not None:
        joints = [skeleton.base_joint, *joints]
    builder.add_articulation(joints, label=f"{name}_finger")
    builder.color()
    model = builder.finalize(device=device)
    return model, skeleton


# ---------------------------------------------------------------------------
# Actuator governor, staircase command and spring preload (shared by both cable models)
# ---------------------------------------------------------------------------
# The solver has no joint effort or speed limits, so they live in the governor: one small state `s_ref`
# per actuator, updated every substep from the measured slider travel. Everything here is a kernel
# launch on device arrays, so it can run inside a captured CUDA graph.

GOVERNOR_FORCE = 0
"""Governor mode: the command is a cable tension [N]."""

GOVERNOR_POSITION = 1
"""Governor mode: the command is a slider travel [m]."""

GOVERNOR_SPEED_FILTER = 0.02
"""Default filter time of the carriage speed guard [s]; zero switches the guard off."""

GOVERNOR_SPEED_MARGIN = 0.002
"""Default margin of the reference rate below the speed limit [m/s]."""


@wp.kernel
def governor_kernel(
    body_q: wp.array[wp.transform],
    slider_body: wp.array[int],
    base_body: wp.array[int],
    base_xform_digit: wp.array[wp.transform],
    pull_axis: wp.array[wp.vec3],
    target_index: wp.array[int],
    mode: wp.array[int],
    command: wp.array[float],
    kp: wp.array[float],
    force_min: float,
    force_limit: float,
    rate_limit: wp.array[float],
    speed_filter: float,
    dt: float,
    s_ref: wp.array[float],
    travel: wp.array[float],
    drive_force: wp.array[float],
    ref_rate: wp.array[float],
    speed_excess: wp.array[wp.vec2],
    joint_target_q: wp.array[float],
):
    """Reference-clamp rule of the linear actuators, one thread per actuator."""
    i = wp.tid()
    speed_limit = rate_limit[i]  # largest reference rate of this actuator
    # measured travel: slider origin relative to the digit frame, along the pull direction
    digit = base_xform_digit[i]
    if base_body[i] >= 0:
        digit = wp.transform_multiply(body_q[base_body[i]], digit)
    offset = wp.transform_get_translation(body_q[slider_body[i]]) - wp.transform_get_translation(digit)
    s = wp.dot(offset, wp.quat_rotate(wp.transform_get_rotation(digit), pull_axis[i]))

    lo = s + force_min / kp[i]  # reference that gives the rest tension
    hi = s + force_limit / kp[i]  # reference that gives the force limit
    goal_in = command[i]
    if mode[i] == 0:
        goal_in = s + command[i] / kp[i]
    goal = wp.clamp(goal_in, lo, hi)
    # Carriage speed guard. The carriage moves with the reference plus the rate at which its lag changes
    # (the lag is the drive force over kp). The second part is the excess. Its envelope (a peak that decays
    # over `speed_filter` seconds, kept per direction) is taken off the allowed reference rate, so that the
    # carriage itself, not only the reference, stays inside the speed limit.
    up = speed_limit
    down = speed_limit
    if speed_filter > 0.0:
        excess_now = (s - travel[i]) / dt - ref_rate[i]
        decay = wp.max(1.0 - dt / speed_filter, 0.0)
        envelope = speed_excess[i]
        envelope = wp.vec2(wp.max(excess_now, envelope[0] * decay), wp.max(-excess_now, envelope[1] * decay))
        speed_excess[i] = envelope
        up = wp.max(speed_limit - envelope[0], 0.0)
        down = wp.max(speed_limit - envelope[1], 0.0)
    ref = s_ref[i] + wp.clamp(goal - s_ref[i], -down * dt, up * dt)  # speed limit
    ref = wp.min(ref, hi)  # the force limit wins over the speed limit
    ref_rate[i] = (ref - s_ref[i]) / dt
    s_ref[i] = ref
    travel[i] = s
    drive_force[i] = kp[i] * (ref - s)
    joint_target_q[target_index[i]] = ref


@wp.kernel
def staircase_command_kernel(
    clock: wp.array[float],
    levels: wp.array2d[float],
    period: float,
    blend: float,
    cyclic: int,
    dt: float,
    command: wp.array[float],
    level_index: wp.array[int],
):
    """Write the staircase command of the current time into every command entry and advance the clock.

    ``levels`` has one row per level and one column per command entry, or one column for all entries.
    At the start of a level the command moves from the previous level to the new one over ``blend``
    seconds with the minimum-jerk curve (zero rate at both ends); with ``blend = 0`` it steps. With
    ``cyclic`` the staircase starts again after its last level, blending from the last level to the first.
    The time is the update count times ``dt`` (the count is kept in ``level_index[1]``); a clock summed up in
    single precision would run about 1 % off after a minute of 0.4 ms steps.
    """
    step = level_index[1]
    t = float(step) * dt
    count = levels.shape[0]
    n = int(t / period)
    k = wp.min(n, count - 1)
    if cyclic != 0:
        k = n % count
    previous = k
    if n > 0 and (cyclic != 0 or n < count):
        previous = (k + count - 1) % count
    u = 1.0
    if blend > 0.0:
        u = wp.clamp((t - float(n) * period) / blend, 0.0, 1.0)
        u = u * u * u * (10.0 + u * (6.0 * u - 15.0))
    for i in range(command.shape[0]):
        column = wp.min(i, levels.shape[1] - 1)
        command[i] = levels[previous, column] + u * (levels[k, column] - levels[previous, column])
    level_index[0] = k
    level_index[1] = step + 1
    clock[0] = float(step + 1) * dt


@wp.kernel
def spring_preload_kernel(
    dof: wp.array[int],
    bias: wp.array[float],
    joint_f: wp.array[float],
):
    """Write the constant spring preload into the generalized joint forces."""
    i = wp.tid()
    joint_f[dof[i]] = bias[i]


class ActuatorGovernor:
    """Device arrays and the per-substep update of the actuator governor.

    The rule, with ``s`` the measured slider travel::

        lo   = s + force_min / kp
        hi   = s + force_limit / kp
        goal = clamp(goal_in, lo, hi)    # goal_in = command (position mode) or s + command / kp (force mode)
        s_ref += clamp(goal - s_ref, -down * dt, +up * dt)    # up = down = speed_limit without the guard
        s_ref  = min(s_ref, hi)
        slider drive target = s_ref

    The drive force ``kp (s_ref - s)`` stays between the rest tension and the
    force limit, and the reference is rate-limited in both directions. The
    rate limit is per actuator (:attr:`rate_limit`): the actuator's speed
    limit minus a margin, or the lower limit a cable of the spec asks for (the
    curl cable of a two-cable finger). It limits the reference; the carriage
    follows and can exceed it for single substeps. The
    slider travel is measured from the body poses, so the rule also holds on a
    moving base.

    Carriage speed guard (:attr:`speed_filter` > 0): the carriage speed is the
    reference rate plus the rate at which the drive lag ``s_ref - s`` shrinks,
    so a carriage on a rate-limited reference still overshoots the speed limit
    when its load drops. The guard measures that excess (carriage speed minus
    the last reference rate), keeps its envelope per direction (a peak that
    decays over :attr:`speed_filter` seconds) and takes the envelope off the
    allowed reference rate in the same direction:
    ``up = speed_limit - envelope_up``, ``down = speed_limit - envelope_down``.

    Create it with :func:`create_actuator_governor`, write :attr:`command`
    (and :attr:`mode`) whenever the command changes and call :meth:`step` once
    per substep before the solver step.
    """

    def __init__(self):
        self.count: int = 0
        """Number of actuators."""
        self.actuators: list[tuple[str, str]] = []
        """Digit name and cable name of every actuator, in the order of the arrays."""
        self.slider_body: wp.array = None
        self.base_body: wp.array = None
        self.base_xform_digit: wp.array = None
        self.pull_axis: wp.array = None
        self.target_index: wp.array = None
        self.kp: wp.array = None
        self.mode: wp.array = None
        """Mode per actuator: :data:`GOVERNOR_FORCE` or :data:`GOVERNOR_POSITION`."""
        self.command: wp.array = None
        """Command per actuator: cable tension [N] in force mode, slider travel [m] in position mode."""
        self.s_ref: wp.array = None
        """Governor state: reference travel per actuator [m]."""
        self.travel: wp.array = None
        """Measured slider travel at the last update [m]."""
        self.drive_force: wp.array = None
        """Drive force ``kp (s_ref - s)`` at the last update [N]."""
        self.ref_rate: wp.array = None
        """Rate of the reference at the last update [m/s]."""
        self.speed_excess: wp.array = None
        """Envelope of carriage speed minus reference rate, upward and downward [m/s] (state of the speed guard)."""
        self.force_min: float = 0.0
        self.force_limit: float = 0.0
        self.speed_limit: float = 0.0
        self.speed_filter: float = 0.0
        """Filter time of the carriage speed guard [s]; zero switches the guard off."""
        self.speed_margin: float = 0.0
        """Margin of the reference rate below the speed limit [m/s]; covers the first instant of a load drop."""
        self.rate_limit: wp.array = None
        """Largest reference rate per actuator [m/s]: the speed limit minus the margin, or the cable's own
        ``reference_rate_limit`` where that is lower (see :func:`actuator_reference_rate`)."""

    def step(self, state: newton.State, control: newton.Control, dt: float):
        """Update the references from the body poses of ``state`` and write the slider drive targets.

        Args:
            state: State at the start of the substep.
            control: Control whose ``joint_target_q`` receives the references.
            dt: Substep length [s].
        """
        wp.launch(
            governor_kernel,
            dim=self.count,
            inputs=[
                state.body_q,
                self.slider_body,
                self.base_body,
                self.base_xform_digit,
                self.pull_axis,
                self.target_index,
                self.mode,
                self.command,
                self.kp,
                self.force_min,
                self.force_limit,
                self.rate_limit,
                self.speed_filter,
                dt,
            ],
            outputs=[
                self.s_ref,
                self.travel,
                self.drive_force,
                self.ref_rate,
                self.speed_excess,
                control.joint_target_q,
            ],
            device=self.s_ref.device,
        )

    def set_command(self, command, mode: int | None = None):
        """Write the command of the actuators and optionally their mode.

        The command is written as given: a new value is a step. The governor
        only limits the rate of its travel reference, which follows a step
        within a few hundredths of a second. Commands to a two-cable digit
        should be blended (:class:`StaircaseCommand`, :class:`HandScript`): a
        stepped knuckle command that drives the first joint onto its stop
        while the curl cable is at the rest tension lets the curl cable dip
        to about 0.4 N for a moment.

        Args:
            command: One value for all actuators or one per actuator: a cable
                tension [N] in force mode, a slider travel [m] in position mode.
            mode: :data:`GOVERNOR_FORCE` or :data:`GOVERNOR_POSITION` for all
                actuators; unchanged if ``None``.
        """
        values = np.broadcast_to(np.asarray(command, dtype=np.float32), (self.count,))
        self.command.assign(np.ascontiguousarray(values))
        if mode is not None:
            self.mode.fill_(int(mode))


def actuator_reference_rate(
    spec: HandSpec, name: str, cable: int | str = 0, speed_margin: float = GOVERNOR_SPEED_MARGIN
) -> float:
    """Largest rate at which the governor moves the reference of one cable's actuator.

    The actuator's speed limit minus the governor's margin, or the cable's own
    ``reference_rate_limit`` where the spec gives a lower one (the curl cable
    of a two-cable finger, :data:`CURL_REFERENCE_RATE_LIMIT`). The limit is on
    the rate of the reference; the carriage follows it and can exceed it for
    single substeps.

    Args:
        spec: The hand spec.
        name: Digit name.
        cable: Index or name of the cable in the digit.
        speed_margin: Margin of the reference rate below the speed limit [m/s].

    Returns:
        The reference rate limit [m/s].
    """
    digit = spec.digits[name]
    item = digit.cable(cable) if isinstance(cable, str) else digit.cables[cable]
    rate = spec.actuator.speed_limit - speed_margin
    if item.reference_rate_limit is not None:
        rate = min(rate, float(item.reference_rate_limit))
    return rate


def create_actuator_governor(
    model: newton.Model,
    spec: HandSpec,
    skeletons,
    mode: int = GOVERNOR_FORCE,
    command: float | None = None,
    speed_filter: float = GOVERNOR_SPEED_FILTER,
    speed_margin: float = GOVERNOR_SPEED_MARGIN,
) -> ActuatorGovernor:
    """Allocate the governor of the given digits on the model's device.

    The reference starts at the slider travel stored in ``model.joint_q`` and
    the command at the rest tension (force mode) or at that travel (position
    mode).

    Args:
        model: The finalized model.
        spec: The hand spec (actuator limits and gains).
        skeletons: One :class:`DigitSkeleton` or a sequence of them; one actuator per
            cable, in the order of :attr:`ActuatorGovernor.actuators`.
        mode: :data:`GOVERNOR_FORCE` or :data:`GOVERNOR_POSITION` for all actuators.
        command: Initial command; the rest tension or the start travel if ``None``.
        speed_filter: Filter time of the carriage speed guard [s]; zero limits the reference rate only.
        speed_margin: Margin of the reference rate below the speed limit [m/s].

    Returns:
        The governor.
    """
    if isinstance(skeletons, DigitSkeleton):
        skeletons = [skeletons]
    device = model.device
    joint_q = model.joint_q.numpy()
    target_start = model.joint_target_q_start.numpy()
    gov = ActuatorGovernor()
    # one actuator per cable: digits in the given order, the cables of a digit in the order of the spec
    actuators = [(s, c) for s in skeletons for c in range(len(s.cable_names))]
    gov.actuators = [(s.name, s.cable_names[c]) for s, c in actuators]
    gov.count = len(actuators)
    gov.slider_body = wp.array([s.slider_body(c) for s, c in actuators], dtype=int, device=device)
    gov.base_body = wp.array([s.bodies["base"] for s, _ in actuators], dtype=int, device=device)
    gov.base_xform_digit = wp.array([s.base_xform_digit for s, _ in actuators], dtype=wp.transform, device=device)
    axes = [(*spec.digits[s.name].cables[c].slider.pull_direction, 0.0) for s, c in actuators]
    gov.pull_axis = wp.array(axes, dtype=wp.vec3, device=device)
    gov.target_index = wp.array([int(target_start[s.joint_sliders[c]]) for s, c in actuators], dtype=int, device=device)
    gov.kp = wp.array([s.slider_kp for s, _ in actuators], dtype=float, device=device)
    start = np.array([joint_q[s.q_sliders[c]] for s, c in actuators], dtype=np.float32)
    gov.s_ref = wp.array(start, dtype=float, device=device)
    gov.travel = wp.array(start, dtype=float, device=device)
    gov.drive_force = wp.zeros(gov.count, dtype=float, device=device)
    gov.ref_rate = wp.zeros(gov.count, dtype=float, device=device)
    gov.speed_excess = wp.zeros(gov.count, dtype=wp.vec2, device=device)
    gov.speed_filter = float(speed_filter)
    gov.speed_margin = float(speed_margin)
    gov.mode = wp.full(gov.count, int(mode), dtype=int, device=device)
    if command is None:
        initial = np.full(gov.count, spec.actuator.force_min, dtype=np.float32) if mode == GOVERNOR_FORCE else start
    else:
        initial = np.full(gov.count, command, dtype=np.float32)
    gov.command = wp.array(initial, dtype=float, device=device)
    gov.force_min = float(spec.actuator.force_min)
    gov.force_limit = float(spec.actuator.force_limit)
    gov.speed_limit = float(spec.actuator.speed_limit)
    rates = [actuator_reference_rate(spec, s.name, c, speed_margin) for s, c in actuators]
    gov.rate_limit = wp.array(rates, dtype=float, device=device)
    return gov


class StaircaseCommand:
    """Tension staircase on a device clock: every level lasts ``period`` seconds.

    At the start of a level the command moves from the previous level to the
    new one over :attr:`blend` seconds with the minimum-jerk curve of the
    keyframe scripts (:func:`minimum_jerk`), then stays. A two-cable digit
    should be commanded this way: a stepped knuckle command that drives the
    first joint onto its stop while the curl cable is at the rest tension
    lets the curl cable dip to about 0.4 N for a moment.

    Create it with :func:`create_staircase_command` and call :meth:`step` once
    per substep before the governor.
    """

    def __init__(self):
        self.levels: wp.array = None
        """Command values, shape ``(level_count, 1)`` (one value for all actuators) or
        ``(level_count, actuator_count)`` (one value per actuator, e.g. per cable of a two-cable digit)."""
        self.period: float = 0.0
        """Time per level [s]: the transition allowance plus the hold."""
        self.hold: float = 0.0
        """Hold time at the end of every level [s]."""
        self.blend: float = 0.0
        """Time over which the command moves from one level to the next [s]; zero steps."""
        self.cyclic: bool = False
        """Whether the staircase starts again after its last level (blending from the last level to the
        first); otherwise the last level is kept."""
        self.clock: wp.array = None
        """Elapsed time [s], shape ``(1,)``."""
        self.level_index: wp.array = None
        """Index of the level written at the last update and the number of updates so far, shape ``(2,)``."""

    @property
    def duration(self) -> float:
        """Length of the whole staircase [s]."""
        return self.period * self.levels.shape[0]

    def step(self, command: wp.array, dt: float):
        """Write the current command into a command array and advance the clock.

        Args:
            command: Array that receives the command in every entry, e.g. ``ActuatorGovernor.command``.
            dt: Substep length [s].
        """
        wp.launch(
            staircase_command_kernel,
            dim=1,
            inputs=[self.clock, self.levels, self.period, self.blend, int(self.cyclic), dt],
            outputs=[command, self.level_index],
            device=self.clock.device,
        )

    def hold_window(self, level: int) -> tuple[float, float]:
        """Start and end time of the hold of a level.

        Args:
            level: Index of the level.

        Returns:
            Start and end of the hold [s] from the start of the staircase.
        """
        return (level + 1) * self.period - self.hold, (level + 1) * self.period

    def evaluate(self, time: float) -> np.ndarray:
        """Command values at a time, computed on the host with the rule of the kernel.

        Args:
            time: Time from the start of the staircase [s].

        Returns:
            One value per column of :attr:`levels`.
        """
        levels = self.levels.numpy()
        count = len(levels)
        n = int(time / self.period)
        k = n % count if self.cyclic else min(n, count - 1)
        previous = (k + count - 1) % count if n > 0 and (self.cyclic or n < count) else k
        u = 1.0 if self.blend <= 0.0 else float(minimum_jerk((time - n * self.period) / self.blend))
        return levels[previous] + u * (levels[k] - levels[previous])


STAIRCASE_BLEND = 0.6
"""Default time over which a staircase command moves from one level to the next [s]: the blend time of
most moves of the hand choreography."""


def create_staircase_command(
    spec: HandSpec,
    levels=None,
    hold: float | None = None,
    transition: float = 0.5,
    device=None,
    blend: float | None = None,
    cyclic: bool = False,
) -> StaircaseCommand:
    """Allocate a staircase command.

    Each level lasts ``transition + hold``. The command moves to the new level
    over ``blend`` seconds (minimum-jerk curve), the governor then needs the
    rest of the transition time to move the slider at its reference rate, and
    the level is held.

    Args:
        spec: The hand spec.
        levels: Command values; :func:`staircase_levels` (tensions up and back down [N]) if ``None``.
            One value per level commands all actuators alike; one row per level with one value per
            actuator (in the order of :attr:`ActuatorGovernor.actuators`) commands every cable by itself.
        hold: Hold time per level [s]; the staircase hold of the spec if ``None``.
        transition: Time allowed for the move to a new level [s].
        device: Device of the arrays.
        blend: Time over which the command moves from one level to the next [s]; the shorter of
            :data:`STAIRCASE_BLEND` and ``transition`` if ``None``; zero steps the command.
        cyclic: Whether the staircase starts again after its last level.

    Returns:
        The staircase command.
    """
    stair = StaircaseCommand()
    values = np.asarray(staircase_levels(spec) if levels is None else list(levels), dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    stair.levels = wp.array(values, dtype=float, device=device)
    stair.hold = float(spec.staircase.hold if hold is None else hold)
    stair.period = stair.hold + float(transition)
    stair.blend = min(STAIRCASE_BLEND, float(transition)) if blend is None else float(blend)
    if not 0.0 <= stair.blend <= float(transition) + 1.0e-12:
        raise ValueError("the blend time must be between zero and the transition time")
    stair.cyclic = bool(cyclic)
    stair.clock = wp.zeros(1, dtype=float, device=device)
    stair.level_index = wp.zeros(2, dtype=int, device=device)
    return stair


class SpringPreload:
    """Device copy of the spring preload bias, see :func:`create_spring_preload`."""

    def __init__(self):
        self.dof: wp.array = None
        """Indices into ``Control.joint_f``."""
        self.bias: wp.array = None
        """Generalized force written at those indices [N*m]."""

    def apply(self, control: newton.Control):
        """Write the bias into the joint forces (a kernel launch; safe inside a captured graph).

        Args:
            control: Control whose ``joint_f`` receives the preload of the flexion joints.
        """
        wp.launch(
            spring_preload_kernel,
            dim=self.dof.shape[0],
            inputs=[self.dof, self.bias],
            outputs=[control.joint_f],
            device=self.dof.device,
        )


def create_spring_preload(model: newton.Model, skeletons, scale: float = 1.0) -> SpringPreload:
    """Allocate the spring preload of the given digits on the model's device.

    The preload is a constant ``Control.joint_f`` entry per flexion joint. The
    solver does not change ``joint_f``, so one :meth:`SpringPreload.apply`
    after creating the control is enough; call it every step if the example
    clears or overwrites ``joint_f``.

    Args:
        model: The finalized model.
        skeletons: One :class:`DigitSkeleton` or a sequence of them.
        scale: Factor on the preload (1 is the design value).

    Returns:
        The preload helper.
    """
    if isinstance(skeletons, DigitSkeleton):
        skeletons = [skeletons]
    dof = [d for s in skeletons for d in s.dof_flexion]
    bias = [scale * float(b) for s in skeletons for b in s.joint_f_bias]
    preload = SpringPreload()
    preload.dof = wp.array(dof, dtype=int, device=model.device)
    preload.bias = wp.array(bias, dtype=float, device=model.device)
    return preload


# ---------------------------------------------------------------------------
# Rod cable model (stations as flanged collision barrels, routed rod, end anchors)
# ---------------------------------------------------------------------------
# The cable of a digit is a chain of small capsules (a rod) laid on the tangent-line and arc route. Every
# contact station is a barrel with two flanges; the rod is kept in the grooves by contact only. The rod is
# tied to the slider and to the distal link by ball joints, and the digit's joints, the rod joints and the
# two anchors form one articulation with closed loops.

ROD_FIRST_COLLISION_GROUP = 2
"""Collision group of the first rod cable; every further cable of a digit or a hand takes the next group."""

ROD_FPS = 60
"""Frame rate the rod solver settings were tuned for [Hz]."""

ROD_SUBSTEPS = 30
"""Substeps per 60 Hz frame for the rod cable."""

ROD_ITERATIONS = 32
"""Solver iterations per substep for the rod cable.

30 substeps x 32 iterations keeps the joint angles of every hold within 2 deg of the closed form, with no margin in
the iteration count: 30 x 16 misses that (3.46 deg, and one hold does not settle).
"""


@dataclass
class RodCable:
    """Indices of the rod cable of one digit added by :func:`add_rod_cable`."""

    name: str
    bodies: list[int]
    """Rod segment bodies from the slider anchor to the tip anchor."""
    rod_joints: list[int]
    """Rod joints; joint ``k`` connects segment ``k`` (parent) to segment ``k + 1`` (child)."""
    anchor_joints: tuple[int, int]
    """Ball joints at the slider and at the tip."""
    station_shapes: dict[str, tuple[int, ...]]
    """Collision shapes per station: barrel, flange on ``-z``, flange on ``+z`` (barrel only with planar guides)."""
    rod_shapes: list[int]
    """Capsule shapes of the rod segments."""
    segment_length: float
    """Rest length of a segment [m]."""
    stretch_ke: float
    """Stretch stiffness of one rod joint, ``EA / segment_length`` [N/m]."""
    start_travel: float
    """Slider travel of the start pose [m]."""
    collision_group: int
    """Collision group shared by the rod capsules and the station shapes of this digit."""
    weight_gravity: tuple[float, float, float] | None = None
    """Gravity the caller must apply as body forces (see :func:`create_body_weight`); ``None`` if the model has it."""
    cable: str = SINGLE_CABLE_NAME
    """Name of the cable in the digit's spec."""
    plane_z: float = 0.0
    """Offset of the cable plane from the digit mid-plane along digit ``+z`` [m]."""

    @property
    def joints(self) -> list[int]:
        """All joints of the cable in the order they were added."""
        return [*self.rod_joints, *self.anchor_joints]


def _transform_to_frame(xform) -> tuple[np.ndarray, np.ndarray]:
    """Origin and rotation matrix (axes as columns) of a transform."""
    xform = wp.transform(*xform)
    rot = np.array(wp.quat_to_matrix(wp.transform_get_rotation(xform)), dtype=float).reshape(3, 3)
    return np.array(wp.transform_get_translation(xform), dtype=float), rot


def configure_rod_builder(builder: newton.ModelBuilder):
    """Make ``finalize()`` keep the very small inertia of rod segments.

    A rod segment of 1 or 2 mm of a 1 mm cord has principal inertias of the
    order of 1e-13 kg*m^2. By default ``finalize()`` shifts every principal inertia below
    1e-10 kg*m^2 up by about 1e-6 kg*m^2. The rod models keep the authored
    segment inertia; the solver from :func:`create_rod_solver` regularizes
    the angular solve instead.

    Args:
        builder: The model builder; call this before ``finalize()``.
    """
    builder.inertia_absolute_tolerance = 1.0e-20
    builder.inertia_correction = 0.0


def add_rod_stations(
    builder: newton.ModelBuilder,
    spec: HandSpec,
    skeleton: DigitSkeleton,
    collision_group: int = ROD_FIRST_COLLISION_GROUP,
    contact_ke: float | None = None,
    cable=None,
) -> dict[str, tuple[int, int, int]]:
    """Add the collision shapes of every contact station of one cable of a digit.

    A station is a barrel cylinder (radius from the barrel rule, half height
    = groove half width + flange thickness) and two flange cylinders whose
    inner faces are one groove half width from the cable plane (the digit
    mid-plane on a one-cable digit). The shapes have no mass. Station shapes
    do not collide with each other. On a digit with several cables the
    stations of two cables on one joint sit side by side on the joint axis,
    one cable-plane pitch apart.

    Args:
        builder: The model builder.
        spec: The hand spec.
        skeleton: The digit skeleton.
        collision_group: Collision group shared with the rod of this digit.
            Use a group that no other shape uses (the builder default is 1).
        contact_ke: Contact stiffness [N/m]; the rod contact stiffness of the spec if ``None``.
        cable: Cable name or index; ``None`` for the only cable of a one-cable digit.

    Returns:
        Shape indices ``(barrel, flange on -z, flange on +z)`` per station name.
    """
    hardware = spec.station_hardware
    cable_spec = spec.digits[skeleton.name].cable(cable)
    plane_z = cable_spec.plane_z
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=spec.rod.contact_ke if contact_ke is None else contact_ke,
        kd=spec.rod.contact_kd,
        mu=spec.cable.friction_default,
        gap=spec.rod.contact_gap,
        collision_group=collision_group,
    )
    half_flange = 0.5 * hardware.flange_thickness
    flange_z = hardware.groove_half_width + half_flange
    radii = {station.station: station for station in rod_stations(spec, skeleton.name, cable)}
    shapes = {}
    for station in cable_spec.stations:
        if station.radius == 0.0:
            continue
        body = skeleton.bodies[station.body]
        center = wp.transform(wp.vec3(station.pos[0], station.pos[1], plane_z), wp.quat_identity())
        if station.body == "base":
            center = wp.transform_multiply(skeleton.base_xform_digit, center)
        rot = wp.transform_get_rotation(center)
        rod_station = radii[station.name]
        triple = []
        parts = [
            ("barrel", 0.0, rod_station.collision_barrel_radius, hardware.groove_half_width + 2.0 * half_flange),
            ("flange_neg", -flange_z, rod_station.flange_radius, half_flange),
            ("flange_pos", flange_z, rod_station.flange_radius, half_flange),
        ]
        for suffix, z, radius, half_height in parts:
            triple.append(
                builder.add_shape_cylinder(
                    body=body,
                    xform=wp.transform(wp.transform_point(center, wp.vec3(0.0, 0.0, z)), rot),
                    radius=radius,
                    half_height=half_height,
                    cfg=cfg,
                    label=f"{skeleton.name}_{station.name}_{suffix}",
                )
            )
        shapes[station.name] = tuple(triple)
    flat = [shape for triple in shapes.values() for shape in triple]
    for i, a in enumerate(flat):
        for b in flat[i + 1 :]:
            builder.add_shape_collision_filter_pair(a, b)
    return shapes


def add_rod_cable(
    builder: newton.ModelBuilder,
    spec: HandSpec,
    skeleton: DigitSkeleton,
    q=(0.0, 0.0, 0.0),
    collision_group: int = ROD_FIRST_COLLISION_GROUP,
    contact_ke: float | None = None,
    stretch_damping: float = 10.0,
    bend_damping: float = 1.0e-5,
    cable=None,
) -> RodCable:
    """Add the station shapes, the routed rod and its two end anchors of one cable to a digit.

    The rod starts on the route of :func:`rod_start_pose` at joint angles
    ``q``. The skeleton must have been added at the same ``q`` and at the
    slider travel of that start pose (``rod_start_pose(spec, name, q).travel``),
    so that the first rod vertex is the slider anchor. The rod is straight and
    unstretched at rest; stretch and shear stiffness per joint are
    ``EA / segment length``, bend and twist stiffness are the cable rigidity
    over the segment length. The rod mass is the cable's mass per length times
    its length. The anchors are ball joints that keep the collision between
    the end segments and the stations of their parent body.

    The rod joints and anchors are not added to an articulation, see
    :func:`add_rod_finger_articulation`.

    Args:
        builder: The model builder.
        spec: The hand spec.
        skeleton: The digit skeleton.
        q: Joint angles of the start pose [rad].
        collision_group: Collision group of the rod capsules and station
            shapes of this digit; rod-rod pairs are filtered, so the rod only
            touches its own stations.
        contact_ke: Contact stiffness [N/m]; the rod contact stiffness of the spec if ``None``.
        stretch_damping: Stretch and shear damping per rod joint [N*s/m].
        bend_damping: Bend and twist damping per rod joint [N*m*s/rad].
        cable: Cable name or index; ``None`` for the only cable of a one-cable
            digit (:func:`add_rod_cables` adds all cables of a digit, each in
            its own collision group).

    Returns:
        The indices of the cable.
    """
    name = skeleton.name
    digit = spec.digits[name]
    cable_index = digit.cable_index(cable)
    cable_spec = digit.cables[cable_index]
    single = len(digit.cables) == 1
    cable_label = f"{name}_cable" if single else f"{name}_{cable_spec.name}_cable"
    base = skeleton.bodies["base"]
    base_world = wp.transform(*builder.body_q[base]) if base >= 0 else wp.transform_identity()
    origin, rotation = _transform_to_frame(wp.transform_multiply(base_world, skeleton.base_xform_digit))
    pose = rod_start_pose_3d(spec, name, origin, rotation, q, cable=cable_index)
    travel = float(builder.joint_q[skeleton.q_sliders[cable_index]])
    if abs(travel - pose.travel) > 1.0e-7:
        raise ValueError(
            f"{name}: the slider starts at travel {travel:.9f} m but the rod start pose needs {pose.travel:.9f} m"
        )

    station_shapes = add_rod_stations(builder, spec, skeleton, collision_group, contact_ke, cable=cable_index)

    count = len(pose.points) - 1
    seg = pose.segment_length
    cable = spec.cable
    rod = newton.Rod(
        pose.points,
        radius=spec.rod.radius,
        rest_straight=True,
        stretch_rigidity=cable.EA,
        shear_rigidity=cable.EA,
        bend_rigidity=cable.bend_rigidity,
        twist_rigidity=cable.twist_rigidity,
    )
    cfg = newton.ModelBuilder.ShapeConfig(
        density=cable.linear_density / (math.pi * spec.rod.radius**2),
        ke=spec.rod.contact_ke if contact_ke is None else contact_ke,
        kd=spec.rod.contact_kd,
        mu=cable.friction_default,
        gap=spec.rod.contact_gap,
        collision_group=collision_group,
    )
    bodies, rod_joints = builder.add_rod(
        rod=rod,
        cfg=cfg,
        total_mass=cable.linear_density * count * seg,
        stretch_damping=stretch_damping,
        shear_damping=stretch_damping,
        bend_damping=bend_damping,
        twist_damping=bend_damping,
        wrap_in_articulation=False,
        label=cable_label,
    )
    rod_bodies = set(bodies)
    rod_shapes = [shape for shape, body in enumerate(builder.shape_body) if body in rod_bodies]
    for i, a in enumerate(rod_shapes):
        for b in rod_shapes[i + 1 :]:
            builder.add_shape_collision_filter_pair(a, b)

    # end anchors in the cable plane: on the cable's slider and on the body of the last station
    anchor_a = cable_spec.stations[0]
    anchor_t = cable_spec.stations[-1]
    plane_z = cable_spec.plane_z
    joint_a = builder.add_joint_ball(
        parent=skeleton.slider_body(cable_index),
        child=bodies[0],
        parent_xform=wp.transform(wp.vec3(anchor_a.pos[0], anchor_a.pos[1], plane_z), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * seg), wp.quat_identity()),
        friction=0.0,
        collision_filter_parent=False,
        label=f"{cable_label}_anchor_slider",
    )
    joint_t = builder.add_joint_ball(
        parent=skeleton.bodies[anchor_t.body],
        child=bodies[-1],
        parent_xform=wp.transform(wp.vec3(anchor_t.pos[0], anchor_t.pos[1], plane_z), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * seg), wp.quat_identity()),
        friction=0.0,
        collision_filter_parent=False,
        label=f"{cable_label}_anchor_tip",
    )
    return RodCable(
        name=name,
        bodies=list(bodies),
        rod_joints=list(rod_joints),
        anchor_joints=(joint_a, joint_t),
        station_shapes=station_shapes,
        rod_shapes=rod_shapes,
        segment_length=seg,
        stretch_ke=cable.EA / seg,
        start_travel=pose.travel,
        collision_group=collision_group,
        cable=cable_spec.name,
        plane_z=plane_z,
    )


def add_rod_cables(
    builder: newton.ModelBuilder,
    spec: HandSpec,
    skeleton: DigitSkeleton,
    q=(0.0, 0.0, 0.0),
    collision_group: int = ROD_FIRST_COLLISION_GROUP,
    **kwargs,
) -> list[RodCable]:
    """Add one rod per cable of a digit, see :func:`add_rod_cable`.

    Every cable gets its own collision group (``collision_group``,
    ``collision_group + 1``, ...), so a rod touches only its own stations:
    not the neighboring pulley on the same joint and not the other rod.

    Args:
        builder: The model builder.
        spec: The hand spec.
        skeleton: The digit skeleton, added at ``q`` and with every slider at
            the travel of its rod start pose (see :func:`rod_start_travels`).
        q: Joint angles of the start pose [rad].
        collision_group: Collision group of the first cable.
        **kwargs: Further arguments of :func:`add_rod_cable`.

    Returns:
        The rod cables in the order of the spec.
    """
    count = len(spec.digits[skeleton.name].cables)
    return [
        add_rod_cable(builder, spec, skeleton, q, collision_group=collision_group + c, cable=c, **kwargs)
        for c in range(count)
    ]


def rod_start_travels(spec: HandSpec, name: str, q=(0.0, 0.0, 0.0)) -> list[float]:
    """Slider travel of the rod start pose of every cable of a digit.

    Args:
        spec: The hand spec.
        name: Digit name.
        q: Joint angles of the start pose [rad].

    Returns:
        One travel per cable [m], in the order of the spec.
    """
    return [rod_start_pose(spec, name, q, cable=c).travel for c in range(len(spec.digits[name].cables))]


def add_rod_finger_articulation(builder: newton.ModelBuilder, skeleton: DigitSkeleton, cable, label: str | None = None):
    """Wrap the joints of one digit and its rod cables in one articulation with closed loops.

    The joints must be contiguous: add the skeleton, then the cables, and
    nothing else in between.

    Args:
        builder: The model builder.
        skeleton: The digit skeleton (with its own base joint, if any).
        cable: The :class:`RodCable` of the digit, or the list of all its rod cables.
        label: Label of the articulation.
    """
    cables = [cable] if isinstance(cable, RodCable) else list(cable)
    joints = list(skeleton.joints)
    if skeleton.base_joint is not None:
        joints = [skeleton.base_joint, *joints]
    for item in cables:
        joints = [*joints, *item.joints]
    if joints != list(range(joints[0], joints[0] + len(joints))):
        raise ValueError("the joints of the digit and its rod cable are not contiguous")
    builder.add_articulation(joints, label=label or f"{skeleton.name}_rod_finger", allow_closed_loops=True)


def build_rod_finger_cables_model(
    spec: HandSpec | None = None,
    name: str = "index",
    base: str = "kinematic",
    base_xform: wp.transform | None = None,
    q=(0.0, 0.0, 0.0),
    gravity=(0.0, 0.0, -GRAVITY),
    device=None,
    contact_ke: float | None = None,
    visual_materials: dict | None = None,
) -> tuple[newton.Model, DigitSkeleton, list[RodCable]]:
    """Build and finalize one digit with one rod per cable.

    One slider, one set of flanged stations and one rod per cable, each cable
    in its own collision group.

    The model itself has no gravity: the caller applies the weights as body
    forces every substep with :func:`create_body_weight` and the gravity
    stored in :attr:`RodCable.weight_gravity` of every cable. With the solver
    of :func:`create_rod_solver` body forces enter inside the solve, whereas
    model gravity shifts the inertial target, and at 16 iterations the links
    then sag several times too far under gravity in the flexion plane.

    Args:
        spec: The hand spec; the default spec if ``None``.
        name: Digit name.
        base: ``"kinematic"`` or ``"fixed"``, see :func:`add_single_finger`.
        base_xform: World pose of the base body; identity if ``None``.
        q: Joint angles of the start pose [rad].
        gravity: Gravity vector in the world frame [m/s^2]; with the default the flexion axis is vertical.
        device: Device of the model.
        contact_ke: Rod contact stiffness [N/m]; the value of the spec if ``None``.
        visual_materials: If given, the visual meshes of the digit are attached with this look per
            material tag (see :func:`add_digit_visuals` and :func:`visual_materials`), and the station
            collision shapes, which the meshes replace on screen, are hidden.

    Returns:
        The finalized model, the digit skeleton and the rod cables in the order of the spec.
    """
    if spec is None:
        spec = default_hand_spec()
    gravity = tuple(float(v) for v in gravity)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    configure_rod_builder(builder)
    skeleton = add_single_finger(builder, spec, name, base, base_xform, q, rod_start_travels(spec, name, q))
    cables = add_rod_cables(builder, spec, skeleton, q, contact_ke=contact_ke)
    add_rod_finger_articulation(builder, skeleton, cables)
    if visual_materials is not None:
        add_digit_visuals(builder, spec, skeleton, materials=visual_materials)
        for cable in cables:
            for shapes in cable.station_shapes.values():
                for shape in shapes:
                    builder.shape_flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)
    builder.color()
    model = builder.finalize(device=device)
    for cable in cables:
        cable.weight_gravity = gravity
    return model, skeleton, cables


def build_single_rod_finger_model(
    spec: HandSpec | None = None, name: str = "ring", **kwargs
) -> tuple[newton.Model, DigitSkeleton, RodCable]:
    """Build and finalize a one-cable digit with its rod cable.

    :func:`build_rod_finger_cables_model` for a digit with one cable, with the
    cable returned on its own.

    Args:
        spec: The hand spec; the default spec if ``None``.
        name: Name of a digit with one cable.
        **kwargs: Further arguments of :func:`build_rod_finger_cables_model`.

    Returns:
        The finalized model, the digit skeleton and the rod cable.

    Raises:
        ValueError: If the digit has more than one cable.
    """
    if spec is None:
        spec = default_hand_spec()
    if len(spec.digits[name].cables) != 1:
        raise ValueError(f"{name} has {len(spec.digits[name].cables)} cables; use build_rod_finger_cables_model")
    model, skeleton, cables = build_rod_finger_cables_model(spec, name, **kwargs)
    return model, skeleton, cables[0]


def rod_inertia_reference(spec: HandSpec) -> float:
    """Reference inertia of the rod solver's angular regularization for the rod of a spec.

    Args:
        spec: The hand spec.

    Returns:
        ``spec.rod.settings_for_1mm["reference_inertia"]`` [kg*m^2]: 2.5e-7 for
        the default spec. Shorter rod segments need a smaller reference.
    """
    return float(spec.rod.settings_for_1mm["reference_inertia"])


ROD_CONTACT_BUFFER = 512
"""Contacts per body the rod solver keeps for one cable of 1 mm segments."""


def rod_contact_buffer(spec: HandSpec, name: str) -> int:
    """Contacts per body the rod solver must keep for one digit of a spec.

    The body that carries the first-joint stations sees a barrel contact and
    two flange contacts for every wrapped rod segment of every cable, so the
    number grows with the cables of the digit and with shorter segments:
    :data:`ROD_CONTACT_BUFFER` times ``1 mm / segment length`` (at least half
    of it), summed over the cables, each with its own segment length (see
    :func:`rod_segment_target`): 1024 for a two-cable finger of the default
    spec, 512 for a one-cable finger and 256 for the thumb with its 2 mm
    segments. (Measured at rest on the two-cable index finger with 1 mm
    segments: 359 contacts on the base body; a buffer of 256 overflows there
    and then leaves tension peaks of 3.2 N instead of 2.6 N at a rest tension
    of 2 N.)

    Args:
        spec: The hand spec.
        name: Digit name.

    Returns:
        The value for ``rigid_body_contact_buffer_size``.
    """
    scales = [max(0.5, 1.0e-3 / rod_segment_target(spec, name, c)) for c in range(len(spec.digits[name].cables))]
    return int(math.ceil(ROD_CONTACT_BUFFER * sum(scales)))


def rod_seating_levels(spec: HandSpec, name: str) -> list[list[float]]:
    """Tension levels of the seating stroke of one digit with rod cables.

    A rod that is loaded from its start pose can come to rest a few
    micrometers above a barrel and then loses tension along its length as if
    it had friction. One stroke of every cable from the rest tension to the
    seating tension and back removes that state; what a rod model shows
    counts from the end of the stroke on. All cables of a digit are stroked
    together.

    Args:
        spec: The hand spec.
        name: Digit name.

    Returns:
        One row per level with one tension per cable [N], for
        ``create_staircase_command(levels=...)``: the
        ``seating_stroke_tensions`` of ``spec.rod.settings_for_1mm`` (rest,
        16 N, rest for the default spec).
    """
    tensions = spec.rod.settings_for_1mm["seating_stroke_tensions"]
    return [[float(tension)] * len(spec.digits[name].cables) for tension in tensions]


def rod_solver_settings(
    substep_dt: float,
    iterations: int = ROD_ITERATIONS,
    inertia_reference: float | None = None,
    spec: HandSpec | None = None,
) -> dict[str, Any]:
    """Default :class:`~newton.solvers.SolverVBD` settings for the rod cable, see :func:`create_rod_solver`.

    Args:
        substep_dt: Substep length the solver will be stepped with [s].
        iterations: VBD iterations per substep.
        inertia_reference: Reference inertia of the angular regularization [kg*m^2];
            :func:`rod_inertia_reference` of the spec if ``None``.
        spec: The hand spec the model was built from; the default spec if ``None``.

    Returns:
        Keyword arguments for the solver constructor. The contact buffer per
        body is the largest :func:`rod_contact_buffer` over the digits of the
        spec (1024 for the default spec), enough for any single finger; the
        hand needs :func:`rod_hand_contact_buffer`.
    """
    if spec is None:
        spec = default_hand_spec()
    if inertia_reference is None:
        inertia_reference = rod_inertia_reference(spec)
    return {
        "iterations": iterations,
        "friction_epsilon": 0.01,
        "rigid_contact_history": True,
        "rigid_avbd_gamma": 0.999,
        "rigid_body_contact_buffer_size": max(rod_contact_buffer(spec, name) for name in spec.digits),
        "rigid_joint_linear_ke": STRUCTURAL_JOINT_KE,
        "rigid_joint_angular_ke": STRUCTURAL_JOINT_KE,
        "rigid_joint_linear_kd": 0.0,
        "rigid_joint_angular_kd": 0.0,
        "rigid_compliant_alm": True,
        "rigid_limit_rho_floor": True,
        "rigid_external_force_mode": "solve",
        "rigid_articulation_solve": "block_sparse_levels",
        "rigid_articulation_levels_launch": "level",
        "rigid_articulation_contact_active_set": True,
        "rigid_articulation_relaxation": 0.65,
        "rigid_articulation_angular_regularization": inertia_reference / (substep_dt * substep_dt),
    }


def create_rod_solver(
    model: newton.Model,
    substep_dt: float,
    iterations: int = ROD_ITERATIONS,
    inertia_reference: float | None = None,
    spec: HandSpec | None = None,
    **overrides,
):
    """Create the VBD solver with the settings used for the rod cable.

    Step the solver with :data:`ROD_SUBSTEPS` substeps per 60 Hz frame. The
    ``local`` articulation solve does not hold this finger (angle errors of
    tens of degrees at 16 and at 64 iterations), and the contact history must
    stay on (without it the finger closes under the rest tension).

    Defaults: block-sparse level-scheduled articulation solve with the contact
    active set, relaxation 0.65, structural joint stiffness
    :data:`STRUCTURAL_JOINT_KE`, compliant
    augmented Lagrangian, limit support floor, contact history with the
    buffer of :func:`rod_contact_buffer` for the digit with the most cables of
    the spec (1024 contacts per body for the default spec), and an angular
    regularization of :func:`rod_inertia_reference` ``/ substep_dt^2``
    (2.5e-7 kg*m^2 for the default spec) so that rod segments with their
    physical inertia take the angular step of a body with the reference
    inertia. External forces enter inside the solve
    (``rigid_external_force_mode="solve"``): the spring preload is a large
    constant joint force on light links, and with the default mode the
    iterations do not converge on it once the rod cable is part of the
    articulation (the first joint then flexes under the rest tension, at 16
    and at 32 iterations; the skeleton without cable is not affected).

    Args:
        model: The finalized model.
        substep_dt: Substep length the solver will be stepped with [s].
        iterations: VBD iterations per substep.
        inertia_reference: Reference inertia of the angular regularization [kg*m^2];
            :func:`rod_inertia_reference` of the spec if ``None``.
        spec: The hand spec the model was built from; the default spec if ``None``.
        **overrides: Solver keyword arguments that replace the defaults, e.g.
            ``rigid_body_contact_buffer_size`` for a body that carries more rod contacts than the default.

    Returns:
        The solver.
    """
    settings = rod_solver_settings(substep_dt, iterations, inertia_reference, spec)
    settings.update(overrides)
    if settings["rigid_articulation_solve"] == "local":
        for key in ("rigid_articulation_levels_launch", "rigid_articulation_contact_active_set"):
            settings.pop(key, None)
    return newton.solvers.SolverVBD(model, **settings)


def create_rod_collision_pipeline(model: newton.Model) -> newton.CollisionPipeline:
    """Create the collision pipeline for rod cables: contacts are matched to the latest frame for warm starts.

    Args:
        model: The finalized model.

    Returns:
        The collision pipeline.
    """
    return newton.CollisionPipeline(model, contact_matching="latest")


@wp.kernel
def body_weight_kernel(
    body_mass: wp.array[float],
    gravity: wp.vec3,
    body_f: wp.array[wp.spatial_vector],
):
    """Add the weight of every body to its external wrench (a force at the center of mass)."""
    b = wp.tid()
    body_f[b] = body_f[b] + wp.spatial_vector(body_mass[b] * gravity, wp.vec3(0.0, 0.0, 0.0))


class BodyWeight:
    """Gravity as external body forces, see :func:`create_body_weight`."""

    def __init__(self, model: newton.Model, gravity):
        self.model = model
        self.gravity = wp.vec3(*[float(v) for v in gravity])
        """Gravity vector in the world frame [m/s^2]."""

    def apply(self, state: newton.State):
        """Add the weights to the body forces; call after ``state.clear_forces()`` in every substep.

        Args:
            state: State whose ``body_f`` receives the weights.
        """
        wp.launch(
            body_weight_kernel,
            dim=self.model.body_count,
            inputs=[self.model.body_mass, self.gravity],
            outputs=[state.body_f],
            device=self.model.device,
        )


def create_body_weight(model: newton.Model, gravity=(0.0, 0.0, -GRAVITY)) -> BodyWeight:
    """Apply gravity as external body forces on a model that was built without gravity.

    With ``rigid_external_force_mode="solve"`` (the setting of
    :func:`create_rod_solver`) external forces enter inside the solve, which
    converges on light links held by stiff joints; model gravity shifts the
    inertial target instead and needs several times more iterations. Use this
    whenever gravity can load the flexion joints.

    Args:
        model: The finalized model, built with zero gravity.
        gravity: Gravity vector in the world frame [m/s^2].

    Returns:
        The helper; call :meth:`BodyWeight.apply` in every substep (a kernel
        launch; safe inside a captured graph).
    """
    return BodyWeight(model, gravity)


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` by quaternions ``q`` (``x, y, z, w``), row by row."""
    u = q[..., :3]
    w = q[..., 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _quat_rotate_inv_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _quat_rotate_np(np.concatenate([-q[..., :3], q[..., 3:4]], axis=-1), v)


class RodTensionProbe:
    """Tension and confinement read-back of a rod cable from body poses.

    The force of rod joint ``k`` is ``stretch_ke * u`` with ``u`` the offset
    of the child anchor from the parent anchor in the parent frame (shear and
    stretch stiffness are equal). Its magnitude, with the sign of the axial
    component, is the cable tension at that vertex.

    Args:
        model: The finalized model.
        skeleton: The digit skeleton.
        cable: The rod cable.
    """

    def __init__(self, model: newton.Model, skeleton: DigitSkeleton, cable: RodCable):
        joints = np.array(cable.rod_joints)
        anchors = np.array(cable.anchor_joints)
        both = np.concatenate([joints, anchors])
        self._parent = model.joint_parent.numpy()[both]
        self._child = model.joint_child.numpy()[both]
        self._x_p = model.joint_X_p.numpy()[both].astype(np.float64)
        self._x_c = model.joint_X_c.numpy()[both].astype(np.float64)
        self._n = len(joints)
        dof = model.joint_qd_start.numpy()[joints]
        # rod joint DoF order: shear x, shear y, stretch z, bend x, bend y, twist z
        self.stretch_ke = model.joint_target_ke.numpy()[dof + 2].astype(np.float64)
        """Stretch stiffness per rod joint [N/m]."""
        self.shear_ke = model.joint_target_ke.numpy()[dof].astype(np.float64)
        """Shear stiffness per rod joint [N/m]."""
        self._bodies = np.array(cable.bodies)
        self._half = 0.5 * cable.segment_length
        self._base = skeleton.bodies["base"]
        self._plane_z = float(cable.plane_z)
        self._base_xform_digit = np.array(skeleton.base_xform_digit, dtype=np.float64)
        # Grooves: the barrel shape of a station sits at the station center on the body that carries it,
        # with the pulley axis along its local z; the widest shape of the station (a flange, if there is
        # one) bounds the part of the rod that the groove holds.
        shape_body = model.shape_body.numpy()
        shape_xform = model.shape_transform.numpy().astype(np.float64)
        shape_scale = model.shape_scale.numpy().astype(np.float64)
        self._groove_body = np.array([shape_body[shapes[0]] for shapes in cable.station_shapes.values()], dtype=int)
        self._groove_xform = np.array([shape_xform[shapes[0]] for shapes in cable.station_shapes.values()])
        self._groove_radius = np.array(
            [max(shape_scale[shape][0] for shape in shapes) for shapes in cable.station_shapes.values()]
        )

    def _vertices(self, body_q: np.ndarray) -> np.ndarray:
        """World positions of the rod vertices, shape ``(segments + 1, 3)``."""
        seg = body_q[self._bodies]
        axis = _quat_rotate_np(seg[:, 3:], np.tile([0.0, 0.0, 1.0], (len(seg), 1)))
        return np.concatenate([seg[:1, :3] - self._half * axis[:1], seg[:, :3] + self._half * axis])

    def _offsets(self, body_q: np.ndarray) -> np.ndarray:
        body_q = np.asarray(body_q, dtype=np.float64)
        pq, cq = body_q[self._parent], body_q[self._child]
        x_p = pq[:, :3] + _quat_rotate_np(pq[:, 3:], self._x_p[:, :3])
        x_c = cq[:, :3] + _quat_rotate_np(cq[:, 3:], self._x_c[:, :3])
        return x_c - x_p, pq[:, 3:]

    def tension(self, body_q: np.ndarray) -> np.ndarray:
        """Cable tension at every rod joint, actuator end first.

        Args:
            body_q: Body poses, shape ``(body_count, 7)``.

        Returns:
            Tensions [N], shape ``(segments - 1,)``.
        """
        d, pq = self._offsets(body_q)
        u = _quat_rotate_inv_np(pq[: self._n], d[: self._n])
        force = np.stack([self.shear_ke * u[:, 0], self.shear_ke * u[:, 1], self.stretch_ke * u[:, 2]], axis=1)
        return np.sign(u[:, 2]) * np.linalg.norm(force, axis=1)

    def axial_tension(self, body_q: np.ndarray) -> np.ndarray:
        """Axial part of the tension only, ``stretch_ke`` times the stretch along the parent segment.

        Args:
            body_q: Body poses, shape ``(body_count, 7)``.

        Returns:
            Axial tensions [N], shape ``(segments - 1,)``.
        """
        d, pq = self._offsets(body_q)
        return self.stretch_ke * _quat_rotate_inv_np(pq[: self._n], d[: self._n])[:, 2]

    def end_tensions(self, body_q: np.ndarray) -> tuple[float, float]:
        """Tension at the actuator end and at the tip end (first and last rod joint).

        Args:
            body_q: Body poses, shape ``(body_count, 7)``.

        Returns:
            The two tensions [N].
        """
        tension = self.tension(body_q)
        return float(tension[0]), float(tension[-1])

    def anchor_gaps(self, body_q: np.ndarray) -> np.ndarray:
        """Opening of the slider and tip ball anchors.

        Args:
            body_q: Body poses, shape ``(body_count, 7)``.

        Returns:
            The two openings [m], shape ``(2,)``.
        """
        d, _ = self._offsets(body_q)
        return np.linalg.norm(d[self._n :], axis=1)

    def lateral_offset(self, body_q: np.ndarray) -> np.ndarray:
        """Signed distance of every rod vertex from the digit plane on the base body [m].

        The plane is the mid-plane of the digit frame as mounted on the base
        body. Links that give sideways under a load carry their grooves and
        the rod with them, and that motion counts here; whether the rod is
        still in its grooves is what :meth:`groove_offset` measures.

        Args:
            body_q: Body poses, shape ``(body_count, 7)``.

        Returns:
            Offsets along digit ``+z`` [m], shape ``(segments + 1,)``.
        """
        body_q = np.asarray(body_q, dtype=np.float64)
        points = self._vertices(body_q)
        digit = self._base_xform_digit
        origin = digit[:3]
        normal = _quat_rotate_np(digit[None, 3:], np.array([[0.0, 0.0, 1.0]]))[0]
        if self._base >= 0:
            base = body_q[self._base]
            origin = base[:3] + _quat_rotate_np(base[None, 3:], origin[None])[0]
            normal = _quat_rotate_np(base[None, 3:], normal[None])[0]
        return (points - origin) @ normal - self._plane_z

    def groove_offset(self, body_q: np.ndarray) -> np.ndarray:
        """Signed distance of every rod vertex from the mid-plane of the groove that holds it [m].

        A vertex is held by a station while it lies within the station's
        largest radius (the flange rim) of the pulley axis. Its offset is
        measured along that axis in the frame of the body that carries the
        station, so a link that gives sideways together with its rod reads
        zero. The groove leaves the rod a side clearance of groove half width
        minus rod radius.

        Args:
            body_q: Body poses, shape ``(body_count, 7)``.

        Returns:
            Offsets along the pulley axes [m], shape ``(segments + 1,)``; zero
            for the vertices in the free spans between the stations.
        """
        body_q = np.asarray(body_q, dtype=np.float64)
        points = self._vertices(body_q)
        offset = np.zeros(len(points))
        nearest = np.full(len(points), np.inf)
        for body, xform, radius in zip(self._groove_body, self._groove_xform, self._groove_radius, strict=True):
            center = xform[:3]
            axis = _quat_rotate_np(xform[None, 3:], np.array([[0.0, 0.0, 1.0]]))[0]
            if body >= 0:
                pose = body_q[body]
                center = pose[:3] + _quat_rotate_np(pose[None, 3:], center[None])[0]
                axis = _quat_rotate_np(pose[None, 3:], axis[None])[0]
            relative = points - center
            along = relative @ axis
            radial = np.linalg.norm(relative - np.outer(along, axis), axis=1)
            held = (radial <= radius) & (radial < nearest)
            offset[held] = along[held]
            nearest[held] = radial[held]
        return offset


# ---------------------------------------------------------------------------
# Massless tendon cable model (tendon links from `tendon_links`, actuator coupling)
# ---------------------------------------------------------------------------
# The cable of a digit is one routed tendon: an attachment on the slider, a static rolling link per guide and
# joint sheave, and an attachment on the distal link. The tendon has no mass and no collision shapes.

TENDON_FPS = 60
"""Frame rate the tendon solver settings were tuned for [Hz]."""

TENDON_SUBSTEPS = 20
"""Solver steps per frame the tendon solver settings were tuned for."""


def set_digit_limit_gains(builder: newton.ModelBuilder, skeleton: DigitSkeleton, limit_ke: float, limit_kd: float):
    """Overwrite the limit stiffness and damping of the flexion joints of a digit in a builder.

    Args:
        builder: The model builder that holds the skeleton.
        skeleton: The digit skeleton.
        limit_ke: Limit stiffness [N*m/rad].
        limit_kd: Limit damping [N*m*s/rad].
    """
    for dof in skeleton.dof_flexion:
        builder.joint_limit_ke[dof] = limit_ke
        builder.joint_limit_kd[dof] = limit_kd


@dataclass
class DigitTendon:
    """Indices of the tendon of one digit added by :func:`add_digit_tendon`."""

    name: str
    tendon: int
    """Tendon index in the model."""
    link_start: int
    """Index of the first tendon link (the slider attachment)."""
    link_count: int
    seg_start: int
    """Index of the first tendon segment; segment ``seg_start + k`` is the span between stations ``k`` and ``k + 1``."""
    seg_count: int
    stations: list[str]
    """Station name per link, actuator end first."""
    compliance: np.ndarray
    """Compliance per span [m/N], shape ``(seg_count,)``."""
    rest_length: np.ndarray
    """Rest length of the free part of every span at the build pose [m], shape ``(seg_count,)``."""
    pretension: float
    """Cable tension at the actuator at the build pose [N]."""
    mu: float
    """Friction coefficient on the rolling links."""
    cable: str = SINGLE_CABLE_NAME
    """Name of the cable in the digit's spec."""

    @property
    def segments(self) -> slice:
        """Slice of this tendon's spans in the per-segment solver arrays."""
        return slice(self.seg_start, self.seg_start + self.seg_count)


def add_digit_tendon(
    builder: newton.ModelBuilder,
    spec: HandSpec,
    skeleton: DigitSkeleton,
    mu: float | None = None,
    pretension: float | None = None,
    q=(0.0, 0.0, 0.0),
    travel: float = 0.0,
    damping: float | None = None,
    cable=None,
) -> DigitTendon:
    """Add the massless tendon of one cable of a digit to a builder.

    The links follow :func:`tendon_links`: an attachment on the slider, a
    static rolling link per guide (orientation +1) and joint sheave
    (orientation -1) with the link axis along digit ``+z``, and an attachment
    on the distal link. Every span carries its share of the cable and anchor
    compliance. The rest length of a span is its free length at the build pose
    minus ``tension * compliance``, so the cable carries ``pretension`` at
    the build pose and the slider travel of the closed-form tables (measured
    from the state at the rest tension) can be compared directly. With
    friction the span tensions at the build pose follow the loading capstan
    profile from ``pretension`` at the actuator, the state the friction tables
    start from. The actuator span is cut shorter by the opening of the three
    hinges under that pretension, so the travel is zero at the rest tension
    after the hinges have opened (the joint stiffness of the solver must match
    ``spec.cable.hinge_compliance``).

    Call ``builder.color()`` after all tendons are added.

    Args:
        builder: The model builder; the skeleton must already be in it.
        spec: The hand spec.
        skeleton: The digit skeleton from :func:`add_digit_skeleton`.
        mu: Friction coefficient on the rolling links; the spec default if ``None``.
        pretension: Cable tension at the build pose [N]; the rest tension of the spec if ``None``.
        q: Joint angles the skeleton was built at [rad].
        travel: Travel of the cable's slider the skeleton was built at [m].
        damping: Span damping; the spec value if ``None``.
        cable: Cable name or index; ``None`` for the only cable of a one-cable
            digit (:func:`add_digit_tendons` adds all cables of a digit).

    Returns:
        The tendon indices.
    """
    name = skeleton.name
    digit = spec.digits[name]
    cable_index = digit.cable_index(cable)
    if mu is None:
        mu = spec.cable.friction_default
    if pretension is None:
        pretension = spec.cable.tension_rest
    if damping is None:
        damping = spec.tendon.damping
    links = tendon_links(spec, name, cable_index)
    spans = digit_route(digit, q, travel, cable=cable_index).spans
    if mu > 0.0:
        capstan = CapstanStatics(spec, name)
        span_tension = capstan.tensions(pretension, mu, capstan.LOADING, q)
    else:
        span_tension = [pretension] * len(spans)
    # The flexion hinges open under the pretension (hinge compliance times the tension at each hinge), which
    # shortens the cable path. The cable is cut shorter by the same amount (taken from the actuator span), so
    # that the slider travel is zero at the build pose once the hinges have opened. A hinge that carries
    # other cables as well opens under their pretension too.
    arms = moment_arm_matrix(digit)
    hinge_opening = spec.cable.hinge_compliance * sum(
        span_tension[k]
        + pretension * sum(1 for c in range(len(digit.cables)) if c != cable_index and arms[c, j] != 0.0)
        for j, k in joint_sheave_stations(digit, cable_index).items()
    )
    axis = tuple(float(v) for v in spec.tendon.axis_in_digit_frame)
    link_types = {"ATTACHMENT": newton.TendonLinkType.ATTACHMENT, "ROLLING": newton.TendonLinkType.ROLLING}

    tendon = builder.add_tendon()
    link_start = len(builder.tendon_link_body)
    seg_start = len(builder.tendon_seg_compliance)
    rest = []
    for k, link in enumerate(links):
        rolling = link.type == "ROLLING"
        rest_length = -1.0
        if k > 0:
            rest_length = float(spans[k - 1]) - span_tension[k - 1] * link.compliance
            if k == 1:
                rest_length -= hinge_opening
            rest.append(rest_length)
        link_offset, link_axis = link.offset, axis
        if link.body == "base":
            # stations on the base are given in the digit frame; the base body frame may differ (palm)
            base_xform = skeleton.base_xform_digit
            link_offset = tuple(wp.transform_point(base_xform, wp.vec3(*link.offset)))
            link_axis = tuple(wp.transform_vector(base_xform, wp.vec3(*axis)))
        builder.add_tendon_link(
            skeleton.bodies[link.body],
            link_type=int(link_types[link.type]),
            radius=link.radius,
            orientation=link.orientation,
            mu=mu if rolling else 0.0,
            offset=link_offset,
            axis=link_axis,
            compliance=link.compliance,
            damping=damping,
            rest_length=rest_length,
        )
    return DigitTendon(
        name=name,
        tendon=tendon,
        link_start=link_start,
        link_count=len(links),
        seg_start=seg_start,
        seg_count=len(links) - 1,
        stations=[link.station for link in links],
        compliance=np.array([link.compliance for link in links[1:]]),
        rest_length=np.array(rest),
        pretension=float(pretension),
        mu=float(mu),
        cable=digit.cables[cable_index].name,
    )


def add_digit_tendons(
    builder: newton.ModelBuilder,
    spec: HandSpec,
    skeleton: DigitSkeleton,
    mu: float | None = None,
    pretension: float | None = None,
    q=(0.0, 0.0, 0.0),
    travel=0.0,
    damping: float | None = None,
) -> list[DigitTendon]:
    """Add one massless tendon per cable of a digit, see :func:`add_digit_tendon`.

    The tendons of a digit with several cables lie in their own planes
    (``plane_z`` either side of the digit mid-plane), so two rolling links of
    different cables on the same joint sit side by side on the joint axis.

    Args:
        builder: The model builder; the skeleton must already be in it.
        spec: The hand spec.
        skeleton: The digit skeleton from :func:`add_digit_skeleton`.
        mu: Friction coefficient on the rolling links; the spec default if
            ``None``. Friction needs a digit with one cable.
        pretension: Cable tension at the build pose [N], the same for every
            cable; the rest tension of the spec if ``None``.
        q: Joint angles the skeleton was built at [rad].
        travel: Slider travel the skeleton was built at [m]; one value for all sliders or one per cable.
        damping: Span damping; the spec value if ``None``.

    Returns:
        The tendon indices, one entry per cable in the order of the spec.
    """
    count = len(spec.digits[skeleton.name].cables)
    travels = [float(travel)] * count if np.ndim(travel) == 0 else [float(v) for v in travel]
    return [
        add_digit_tendon(builder, spec, skeleton, mu, pretension, q, travels[c], damping, cable=c) for c in range(count)
    ]


def build_tendon_finger_cables_model(
    spec: HandSpec | None = None,
    name: str = "index",
    mu: float | None = None,
    base: str = "fixed",
    base_xform: wp.transform | None = None,
    q=(0.0, 0.0, 0.0),
    travel=0.0,
    pretension: float | None = None,
    gravity=(0.0, 0.0, -GRAVITY),
    device=None,
    limit_gains: tuple[float, float] | None = None,
    visual_materials: dict | None = None,
) -> tuple[newton.Model, DigitSkeleton, list[DigitTendon]]:
    """Build and finalize one digit with one massless tendon per cable.

    The same model as :func:`build_tendon_finger_model` (see there for the
    base, the gravity rule and the solver settings) for a digit with any
    number of cables: one slider and one tendon per cable.

    Args:
        spec: The hand spec; the default spec if ``None``.
        name: Digit name.
        mu: Friction coefficient on the rolling links; the spec default if
            ``None``. Friction needs a digit with one cable.
        base: ``"kinematic"`` or ``"fixed"``, see :func:`add_single_finger`.
        base_xform: World pose of the base body; identity if ``None``.
        q: Initial joint angles [rad].
        travel: Initial slider travel [m]; one value for all sliders or one per cable.
        pretension: Cable tension at the build pose [N]; the rest tension of the spec if ``None``.
        gravity: Gravity vector in the world frame [m/s^2].
        device: Device of the model.
        limit_gains: Limit stiffness [N*m/rad] and damping [N*m*s/rad] of the
            flexion joints; ``None`` keeps the values of the spec.
        visual_materials: If given, the visual meshes of the digit are attached with this look per
            material tag (see :func:`add_digit_visuals` and :func:`visual_materials`).

    Returns:
        The finalized model, the digit skeleton and the tendon indices of
        every cable in the order of the spec.
    """
    if spec is None:
        spec = default_hand_spec()
    builder = newton.ModelBuilder(gravity=gravity)
    skeleton = add_single_finger(builder, spec, name, base, base_xform, q, travel)
    if limit_gains is not None:
        set_digit_limit_gains(builder, skeleton, limit_gains[0], limit_gains[1])
    tendons = add_digit_tendons(builder, spec, skeleton, mu=mu, pretension=pretension, q=q, travel=travel)
    joints = skeleton.joints
    if skeleton.base_joint is not None:
        joints = [skeleton.base_joint, *joints]
    builder.add_articulation(joints, label=f"{name}_finger")
    if visual_materials is not None:
        add_digit_visuals(builder, spec, skeleton, materials=visual_materials)
    builder.color()
    model = builder.finalize(device=device)
    return model, skeleton, tendons


def build_tendon_finger_model(
    spec: HandSpec | None = None,
    name: str = "index",
    mu: float | None = None,
    base: str = "fixed",
    base_xform: wp.transform | None = None,
    q=(0.0, 0.0, 0.0),
    travel: float = 0.0,
    pretension: float | None = None,
    gravity=(0.0, 0.0, -GRAVITY),
    device=None,
    limit_gains: tuple[float, float] | None = None,
    visual_materials: dict | None = None,
) -> tuple[newton.Model, DigitSkeleton, DigitTendon]:
    """Build and finalize one digit with its massless tendon.

    The skeleton joints form one articulation, the tendon is added before the
    bodies are colored. With the default pose the flexion axis is the world
    ``+z`` axis and gravity does not load the joints. The default base is a
    dynamic body on a fixed joint, the configuration the closed-form staircase
    was checked on; a kinematic base also works with the settings of
    :func:`tendon_solver_settings`.

    Model gravity is only safe while it does not load the joints (the default
    pose). Whenever gravity can act in the flexion plane, build the model with
    ``gravity=(0, 0, 0)`` and apply the weights as body forces every substep
    with :func:`create_body_weight`: model gravity shifts the inertial target,
    and the body-by-body solve does not recover from that (the finger sags by
    tens of degrees).

    Args:
        spec: The hand spec; the default spec if ``None``.
        name: Digit name.
        mu: Friction coefficient on the rolling links; the spec default if ``None``.
        base: ``"kinematic"`` or ``"fixed"``, see :func:`add_single_finger`.
        base_xform: World pose of the base body; identity if ``None``.
        q: Initial joint angles [rad].
        travel: Initial slider travel [m].
        pretension: Cable tension at the build pose [N]; the rest tension of the spec if ``None``.
        gravity: Gravity vector in the world frame [m/s^2].
        device: Device of the model.
        limit_gains: Limit stiffness [N*m/rad] and damping [N*m*s/rad] of the
            flexion joints; ``None`` (default) keeps the values of the spec,
            which are the soft stops the tendon digits need (see
            :attr:`JointSpec.limit_ke`).
        visual_materials: If given, the visual meshes of the digit are attached with this look per
            material tag (see :func:`add_digit_visuals` and :func:`visual_materials`).

    Returns:
        The finalized model, the digit skeleton and the tendon indices.
    """
    if spec is None:
        spec = default_hand_spec()
    builder = newton.ModelBuilder(gravity=gravity)
    skeleton = add_single_finger(builder, spec, name, base, base_xform, q, travel)
    if limit_gains is not None:
        set_digit_limit_gains(builder, skeleton, limit_gains[0], limit_gains[1])
    tendon = add_digit_tendon(builder, spec, skeleton, mu=mu, pretension=pretension, q=q, travel=travel)
    joints = skeleton.joints
    if skeleton.base_joint is not None:
        joints = [skeleton.base_joint, *joints]
    builder.add_articulation(joints, label=f"{name}_finger")
    if visual_materials is not None:
        add_digit_visuals(builder, spec, skeleton, materials=visual_materials)
    builder.color()
    model = builder.finalize(device=device)
    return model, skeleton, tendon


def tendon_solver_settings(friction: bool = False) -> dict[str, Any]:
    """Default :class:`~newton.solvers.SolverVBD` settings for the tendon digits.

    Tuned for :data:`TENDON_FPS` frames per second with
    :data:`TENDON_SUBSTEPS` solver steps per frame.

    Args:
        friction: ``True`` if any rolling link has friction; this selects the
            direct material solve, whose failures are reported by
            ``solver.check_tendon_material()``.

    Returns:
        Keyword arguments for the solver constructor.
    """
    settings = {
        "iterations": 16,
        "rigid_articulation_solve": "local",
        "rigid_time_integration": "bdf1",
        "rigid_limit_rho_floor": True,
        # the spring preload is a constant joint force on light links held by stiff limits
        "rigid_external_force_mode": "solve",
        # the same structural joint stiffness as the rod model; each flexion hinge opens by the cable
        # tension over this stiffness, which the closed-form travel includes (spec.cable.hinge_compliance)
        "rigid_joint_linear_ke": STRUCTURAL_JOINT_KE,
        "rigid_joint_angular_ke": STRUCTURAL_JOINT_KE,
        # the solver default; lower values make the loaded upper stops chatter at the tension limit
        "rigid_avbd_gamma": 0.999,
        "tendon_settle_tol": 1.0e-4,
    }
    if friction:
        settings["tendon_material_direct"] = True
    return settings


def make_tendon_solver(model: newton.Model, friction: bool | None = None, **overrides) -> newton.solvers.SolverVBD:
    """Create the solver for a model with tendon digits.

    Args:
        model: The model.
        friction: Whether to use the friction settings of
            :func:`tendon_solver_settings`; if ``None``, they are used when any
            tendon link of the model has friction.
        **overrides: Solver keyword arguments that replace the defaults.

    Returns:
        The solver. Do not call ``solver.reset()`` on it; build a new solver instead.
    """
    if friction is None:
        friction = model.tendon_link_count > 0 and bool(np.any(model.tendon_link_mu.numpy() > 0.0))
    settings = tendon_solver_settings(friction)
    settings.update(overrides)
    return newton.solvers.SolverVBD(model, **settings)


def tendon_span_tensions(solver, tendon: DigitTendon) -> np.ndarray:
    """Material tension of every span of a digit's tendon, actuator side first.

    Args:
        solver: The solver that steps the model.
        tendon: The tendon indices of the digit.

    Returns:
        One tension per free span [N].
    """
    return solver.tendon_seg_material_tension.numpy()[tendon.segments].astype(float)


def tendon_end_tensions(solver, tendon: DigitTendon) -> tuple[float, float]:
    """Tension of the span at the actuator and of the span at the tip anchor.

    Args:
        solver: The solver that steps the model.
        tendon: The tendon indices of the digit.

    Returns:
        The two tensions [N].
    """
    tension = tendon_span_tensions(solver, tendon)
    return float(tension[0]), float(tension[-1])


def tendon_tangent_points(solver, model: newton.Model, state: newton.State, tendon: DigitTendon):
    """World-space end points of every free span of a digit's tendon.

    Args:
        solver: The solver that steps the model.
        model: The model.
        state: The state to read the body poses from.
        tendon: The tendon indices of the digit.

    Returns:
        Two arrays of shape ``(seg_count, 3)`` [m]: where span ``k`` leaves
        station ``k`` and where it arrives on station ``k + 1``. In the digit
        frame they are ``p_out[:-1]`` and ``p_in[1:]`` of :func:`digit_route`.
    """
    left, right = get_tendon_attachment_worlds(solver, model, state)
    return left[tendon.segments].astype(float), right[tendon.segments].astype(float)


def _tendon_cable_length(solver, model: newton.Model, state: newton.State, tendon: DigitTendon) -> float:
    """Geometric length of a digit's tendon at the current pose, free spans plus arcs.

    Args:
        solver: The solver that steps the model.
        model: The model.
        state: The state to read the body poses from.
        tendon: The tendon indices of the digit.

    Returns:
        The length [m].
    """
    return float(get_tendon_total_lengths(solver, model, state)[tendon.tendon])


def _tendon_unstretched_length(solver, tendon: DigitTendon) -> float:
    """Unstretched cable length the solver conserves for a digit's tendon.

    Args:
        solver: The solver that steps the model.
        tendon: The tendon indices of the digit.

    Returns:
        The length [m].
    """
    return float(solver.tendon_total_cable.numpy()[tendon.tendon])


def _tendon_stretch(solver, tendon: DigitTendon) -> float:
    """Stretch the span tensions and compliances predict for a digit's tendon.

    Args:
        solver: The solver that steps the model.
        tendon: The tendon indices of the digit.

    Returns:
        The stretch [m].
    """
    return float(np.dot(tendon_span_tensions(solver, tendon), tendon.compliance))


def _tendon_cable_lines(solver, model: newton.Model, state: newton.State) -> tuple[wp.array, wp.array]:
    """Line segments that draw all tendons of a model, free spans and arcs.

    Args:
        solver: The solver that steps the model.
        model: The model.
        state: The state to read the body poses from.

    Returns:
        Start and end points [m] as ``wp.vec3`` arrays of length ``line_count``.
    """
    return get_tendon_cable_lines(solver, model, state)


# ---------------------------------------------------------------------------
# Drawing helpers (tension colors, cable pieces of both cable models, visual meshes of one digit)
# ---------------------------------------------------------------------------

CABLE_COLORS = {"curl": (26 / 255, 191 / 255, 255 / 255), "knuckle": (255 / 255, 41 / 255, 143 / 255)}
"""Base color (RGB) of a cable by its role: the curl cable cyan-blue (``#1ABFFF``), the knuckle cable
magenta-pink (``#FF298F``).

The hue says which cable it is; the single cable of a one-cable digit (role
``"flexor"``) takes the curl hue. Change the look of both cable models here.
"""

CABLE_COLOR_REST_BRIGHTNESS = 0.45
"""Brightness of a cable at the rest tension relative to its base color; a cable at the top of the tension
scale has 1."""

CABLE_COLOR_REST_TENSION = 2.0
"""Lower end of the tension scale of the cable colors [N]: the rest tension of the cables."""

XRAY_SHELL_MATERIALS = ("frame", "cover", "tip", "housing", "pad", "actuator")
"""Material tags of the visual meshes that hide the cable; drawn translucent in the x-ray view."""


def tension_colors(
    tension, tension_max: float, role: str = "curl", tension_rest: float = CABLE_COLOR_REST_TENSION
) -> np.ndarray:
    """Colors of one cable: the hue of its role, the brightness of its tension.

    The hue identifies the cable (:data:`CABLE_COLORS`: curl cable cyan-blue,
    knuckle cable magenta-pink; the single cable of a one-cable digit takes
    the curl hue). The tension sets the brightness on one fixed scale for
    every digit and both cable models: the factor on the base color rises
    linearly from :data:`CABLE_COLOR_REST_BRIGHTNESS` at ``tension_rest`` to 1
    at ``tension_max``.

    Args:
        tension: Cable tensions [N], any shape; values outside the scale are clamped.
        tension_max: Upper end of the scale [N].
        role: Role of the cable: ``"curl"``, ``"knuckle"`` or ``"flexor"``
            (:attr:`DigitCableSpec.role`).
        tension_rest: Lower end of the scale [N].

    Returns:
        RGB colors, shape ``(*tension.shape, 3)``, ``float32``.
    """
    base = np.asarray(CABLE_COLORS.get(role, CABLE_COLORS["curl"]), dtype=float)
    u = np.clip((np.asarray(tension, dtype=float) - tension_rest) / (tension_max - tension_rest), 0.0, 1.0)
    brightness = CABLE_COLOR_REST_BRIGHTNESS + (1.0 - CABLE_COLOR_REST_BRIGHTNESS) * u
    return (brightness[..., None] * base).astype(np.float32)


def rod_segment_tensions(probe: RodTensionProbe, body_q: np.ndarray) -> np.ndarray:
    """Cable tension per rod segment: the mean of the tensions at its two rod joints.

    Args:
        probe: The tension probe of the rod cable.
        body_q: Body poses, shape ``(body_count, 7)``.

    Returns:
        Tensions [N], shape ``(segments,)``; an end segment has the tension of its one rod joint.
    """
    joint = probe.tension(body_q)
    return 0.5 * (np.concatenate([joint[:1], joint]) + np.concatenate([joint, joint[-1:]]))


def tendon_cable_pieces(
    solver, model: newton.Model, state: newton.State, tendon: DigitTendon, arc_step: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straight pieces that draw the tendon of one digit, with the tension each piece carries.

    The free spans are one piece each; the arc on a guide or sheave is cut
    into chords of at most ``arc_step`` and carries the mean tension of its two
    spans. The pieces follow the cable from the actuator to the tip.

    Args:
        solver: The solver that steps the model.
        model: The model.
        state: The state to draw.
        tendon: The tendon of the digit.
        arc_step: Largest angle of one arc chord [rad].

    Returns:
        Start points [m] and end points [m], shape ``(pieces, 3)``, and the
        tension per piece [N], shape ``(pieces,)``.
    """
    left, right = tendon_tangent_points(solver, model, state, tendon)
    tension = tendon_span_tensions(solver, tendon)
    links = slice(tendon.link_start, tendon.link_start + tendon.link_count)
    link_body = model.tendon_link_body.numpy()[links]
    link_offset = model.tendon_link_offset.numpy()[links].astype(float)
    link_axis = model.tendon_link_axis.numpy()[links].astype(float)
    body_q = state.body_q.numpy().astype(float)
    starts, ends, values = [], [], []
    for k in range(tendon.seg_count):
        starts.append(left[k])
        ends.append(right[k])
        values.append(tension[k])
        if k + 1 >= tendon.seg_count:
            break
        # arc on station k + 1, from the end of span k to the start of span k + 1
        center, normal = link_offset[k + 1], link_axis[k + 1]
        if link_body[k + 1] >= 0:
            pose = body_q[link_body[k + 1]]
            center = pose[:3] + _quat_rotate_np(pose[3:], center)
            normal = _quat_rotate_np(pose[3:], normal)
        r0, r1 = right[k] - center, left[k + 1] - center
        angle = math.atan2(float(np.dot(np.cross(r0, r1), normal)), float(np.dot(r0, r1)))
        count = max(1, math.ceil(abs(angle) / arc_step))
        side = np.cross(normal, r0)
        points = [center + r0 * math.cos(angle * j / count) + side * math.sin(angle * j / count) for j in range(count)]
        points.append(left[k + 1])
        for j in range(count):
            starts.append(points[j])
            ends.append(points[j + 1])
            values.append(0.5 * (tension[k] + tension[k + 1]))
    return np.array(starts), np.array(ends), np.array(values)


def cable_piece_transforms(starts, ends) -> tuple[np.ndarray, np.ndarray]:
    """Poses and half lengths of capsules or cylinders (axis along local ``z``) that span straight pieces.

    Args:
        starts: Start points [m], shape ``(pieces, 3)``.
        ends: End points [m], shape ``(pieces, 3)``.

    Returns:
        Transforms ``(px, py, pz, qx, qy, qz, qw)``, shape ``(pieces, 7)``,
        ``float32``, and half lengths [m], shape ``(pieces,)``.
    """
    starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    delta = ends - starts
    length = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(length, 1.0e-12)[:, None]
    # shortest rotation from +z to the direction: q = (z x d, 1 + z.d), normalized
    quat = np.stack([-direction[:, 1], direction[:, 0], np.zeros(len(delta)), 1.0 + direction[:, 2]], axis=1)
    flipped = quat[:, 3] < 1.0e-9  # direction = -z: half a turn about x
    quat[flipped] = (1.0, 0.0, 0.0, 0.0)
    quat /= np.linalg.norm(quat, axis=1)[:, None]
    xforms = np.concatenate([0.5 * (starts + ends), quat], axis=1).astype(np.float32)
    return xforms, 0.5 * length


def load_hand_visual_parts(spec: HandSpec | None = None) -> list:
    """Visual meshes of the hand for a spec.

    The shipped USD asset holds the meshes of the default spec. It is used
    when ``spec`` is ``None`` or equal to the default spec and ``pxr`` is
    installed. For any other spec, and without ``pxr``, the meshes are
    generated from the spec, so they always fit the hand that is built.

    Args:
        spec: The hand spec; the default spec if ``None``.

    Returns:
        The visual parts (``cable_hand_visuals.VisualPart``).
    """
    default = default_hand_spec().to_dict()
    data = default if spec is None else spec.to_dict()
    if data == default:
        try:
            return cable_hand_visuals.load_visuals_usd()
        except ImportError:
            pass
    return cable_hand_visuals.build_hand_visuals(data)


def visual_materials(xray: bool = False, opacity: float = 0.22) -> dict[str, dict]:
    """Look per material tag of the visual meshes.

    Args:
        xray: If ``True``, the shells that hide the cable (:data:`XRAY_SHELL_MATERIALS`) are translucent.
        opacity: Opacity of the shells in the x-ray view.

    Returns:
        A copy of ``cable_hand_visuals.MATERIALS``, with ``opacity`` set on the shells for the x-ray view.
    """
    materials = {tag: dict(look) for tag, look in cable_hand_visuals.MATERIALS.items()}
    if xray:
        for tag in XRAY_SHELL_MATERIALS:
            materials[tag]["opacity"] = opacity
    return materials


def set_xray_look(model: newton.Model, parts=None, opacity: float = 0.22) -> int:
    """Make the shells of the visual meshes of a finalized model translucent.

    For models whose builder attached the visuals with the default look
    (``visuals=True`` of the hand builders): the shapes of the parts that hide
    the cables (:data:`XRAY_SHELL_MATERIALS`) are found by their labels and
    get the given opacity in ``model.shape_opacity``. Call it before the model
    is given to a viewer.

    Args:
        model: The finalized model.
        parts: Visual parts of the hand; :func:`load_hand_visual_parts` if ``None``.
        opacity: Opacity of the shells.

    Returns:
        Number of shapes changed.
    """
    if parts is None:
        parts = load_hand_visual_parts()
    shells = {f"visual/{part.body}/{part.name}" for part in parts if part.material in XRAY_SHELL_MATERIALS}
    values = model.shape_opacity.numpy()
    changed = [i for i, label in enumerate(model.shape_label) if label in shells]
    values[changed] = opacity
    model.shape_opacity.assign(values)
    return len(changed)


def hide_rod_stations(model: newton.Model, cables) -> int:
    """Stop drawing the station collision shapes of rod cables in a finalized model.

    The visual meshes show the same pulleys; the collision shapes keep
    colliding. Call it before the model is given to a viewer.

    Args:
        model: The finalized model.
        cables: One :class:`RodCable`, a list of them, or a mapping of them per actuator key.

    Returns:
        Number of shapes hidden.
    """
    if isinstance(cables, RodCable):
        cables = {cables.name: cables}
    shapes = [shape for cable in cables.values() for group in cable.station_shapes.values() for shape in group]
    flags = model.shape_flags.numpy()
    flags[shapes] &= ~int(newton.ShapeFlags.VISIBLE)
    model.shape_flags.assign(flags)
    return len(shapes)


def add_digit_visuals(
    builder: newton.ModelBuilder,
    spec: HandSpec,
    skeleton: DigitSkeleton,
    parts=None,
    materials=None,
    bracket: bool = True,
) -> dict[str, int]:
    """Attach the visual meshes of one digit to a single-digit skeleton (see :func:`add_single_finger`).

    The links and the carriage take their own parts. The parts of the digit
    that the hand carries on its palm (actuator, first guide and first sheave;
    for the thumb also the column) are attached to the base body, moved from
    the palm frame into the digit frame at zero opposition. A bracket (a back
    plate with a post under every axle and under the actuator, visual only)
    stands in for the palm.

    Args:
        builder: The model builder that holds the skeleton.
        spec: The hand spec.
        skeleton: The digit skeleton.
        parts: Visual parts of the hand; :func:`load_hand_visual_parts` if ``None``.
        materials: Look per material tag; the default look if ``None``.
        bracket: Whether to add the base bracket.

    Returns:
        Mapping from ``"<body key>/<part name>"`` to the shape index.
    """
    if parts is None:
        parts = load_hand_visual_parts(spec)
    name = skeleton.name
    palm_to_digit = wp.transform_inverse(frame_to_transform(*digit_frame_in_palm(spec, name)))
    sliders = [skeleton.slider_key(c) for c in range(len(skeleton.cable_names))]
    own = {f"{name}/{key}": key for key in ("link1", "link2", "link3", *sliders)}
    selected = []
    for part in parts:
        if part.body in own:
            selected.append(dataclasses.replace(part, body=own[part.body]))
            continue
        if part.body == "palm" and part.name.startswith(f"{name}_"):
            xform = palm_to_digit * wp.transform(wp.vec3(*map(float, part.pos)), wp.quat(*map(float, part.quat)))
        elif part.body == "thumb_column" and name == "thumb":
            xform = wp.transform(wp.vec3(*map(float, part.pos)), wp.quat(*map(float, part.quat)))
        else:
            continue
        xform = skeleton.base_xform_digit * xform
        selected.append(
            dataclasses.replace(
                part,
                body="base",
                pos=np.array(wp.transform_get_translation(xform), dtype=float),
                quat=np.array(wp.transform_get_rotation(xform), dtype=float),
            )
        )
    if bracket:
        # A back plate behind the base parts and the first link, with a post under every pulley axle and two
        # under the actuator, so that the single digit stands on a bracket instead of floating parts.
        points, posts, lowest = [], [], 0.0
        for part in selected:
            if part.body not in ("base", "link1"):
                continue
            quat = np.asarray(part.quat, dtype=float)[None]
            vertices = _quat_rotate_np(quat, part.vertices.astype(float)) + np.asarray(part.pos, dtype=float)
            lowest = min(lowest, float(vertices[:, 2].min()))
            if part.body != "base":
                continue
            points.append(vertices[:, :2])
            if part.material in ("sheave", "guide"):
                posts.append((*part.pos[:2], float(part.pos[2]), 1.2e-3))
            elif part.material == "actuator":
                lo, hi = vertices.min(axis=0), vertices.max(axis=0)
                long_axis = 0 if hi[0] - lo[0] > hi[1] - lo[1] else 1
                for u in (0.25, 0.75):
                    foot = 0.5 * (lo[:2] + hi[:2])
                    foot[long_axis] = lo[long_axis] + u * (hi[long_axis] - lo[long_axis])
                    posts.append((float(foot[0]), float(foot[1]), float(lo[2]), 2.0e-3))
        if points:
            mesh = cable_hand_visuals.build_bracket_mesh(np.vstack(points), lowest - 0.5e-3, posts)
            selected.append(
                cable_hand_visuals.VisualPart(
                    "bracket",
                    "base",
                    np.ascontiguousarray(mesh[0], dtype=np.float32),
                    np.ascontiguousarray(mesh[1], dtype=np.int32),
                    "housing",
                )
            )
    return cable_hand_visuals.add_visuals_to_builder(builder, selected, skeleton.bodies, materials)


# ---------------------------------------------------------------------------
# Hand (palm on a wrist, thumb column, five digits; hand-level control; massless-tendon hand)
# ---------------------------------------------------------------------------
# The whole hand is one articulation with the palm as root. Joint order: the root joint of the palm, then
# per digit in the order of `spec.digits`: [thumb opposition], MCP, PIP, DIP, slider, and directly after
# them whatever the cable model of that digit adds (the rod joints and anchors of the rod model; nothing
# for the massless tendon). `add_hand_palm` and `add_hand_digit` build it digit by digit so that a cable
# can follow its digit; `add_hand_skeleton` loops over the digits.

HAND_WRISTS = ("fixed", "kinematic")
"""Wrist kinds: a dynamic palm on a fixed joint, or a kinematic palm on a free joint that a script moves."""


@dataclass
class HandSkeleton:
    """Indices and controller data of the hand skeleton, see :func:`add_hand_skeleton`.

    A positive opposition angle lifts the thumb out of the palm plane toward
    the palmar side and across the palm.
    """

    wrist: str
    """``"fixed"`` or ``"kinematic"``, see :data:`HAND_WRISTS`."""
    palm: int
    """Body index of the palm."""
    palm_xform: wp.transform
    """World pose of the palm frame at build time."""
    root_joint: int
    """Joint that holds the palm: a fixed joint, or a free joint for the kinematic palm."""
    q_root: int
    """Index of the first root joint coordinate in ``joint_q`` (the kinematic palm has seven)."""
    opposition_limit_gains: tuple[float, float]
    """Limit stiffness [N*m/rad] and damping [N*m*s/rad] of the opposition hinge."""
    actuator_gains: tuple[float, float]
    """Position gain [N/m] and velocity gain [N*s/m] of the slider drives."""
    thumb_column: int = -1
    """Body index of the thumb column, -1 until the thumb is added."""
    joint_opposition: int = -1
    """Joint index of the thumb opposition hinge."""
    q_opposition: int = -1
    """Index of the opposition angle in ``joint_q``."""
    dof_opposition: int = -1
    """Index of the opposition hinge in ``joint_qd``, ``joint_f`` and the drive gain arrays."""
    opposition_lower: float = 0.0
    """Lower opposition limit [rad]."""
    opposition_upper: float = 0.0
    """Upper opposition limit [rad]."""
    opposition_ke: float = 0.0
    """Opposition drive stiffness [N*m/rad]."""
    opposition_kd: float = 0.0
    """Opposition drive damping [N*m*s/rad]."""
    opposition_torque_limit: float = 0.0
    """Largest opposition drive torque [N*m]."""
    opposition_speed_limit: float = 0.0
    """Largest rate of the opposition reference [rad/s]."""
    opposition_parent_xform: wp.transform = field(default_factory=wp.transform_identity)
    """Pose of the opposition hinge frame in the palm frame."""
    opposition_child_xform: wp.transform = field(default_factory=wp.transform_identity)
    """Pose of the opposition hinge frame in the thumb column frame."""
    opposition_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """Hinge axis in the hinge frame."""
    digits: dict[str, DigitSkeleton] = field(default_factory=dict)
    """Skeleton per digit, in the order the digits were added."""
    digit_joints: dict[str, list[int]] = field(default_factory=dict)
    """Skeleton joints per digit in the order they were added: [opposition], MCP, PIP, DIP, slider."""
    bodies: dict[str, int] = field(default_factory=dict)
    """Body index per visual body key: ``"palm"``, ``"thumb_column"``, ``"<digit>/link1"`` ..
    ``"<digit>/link3"``, ``"<digit>/slider"`` (the keys of ``cable_hand_visuals.body_keys``)."""

    @property
    def names(self) -> list[str]:
        """Digit names in the order they were added."""
        return list(self.digits)

    @property
    def skeletons(self) -> list[DigitSkeleton]:
        """Digit skeletons in the order they were added."""
        return list(self.digits.values())

    @property
    def joints(self) -> list[int]:
        """All skeleton joints in the order they were added, root joint first."""
        return [self.root_joint, *(joint for joints in self.digit_joints.values() for joint in joints)]

    @property
    def actuator_cables(self) -> list[tuple[str, str]]:
        """Digit name and cable name of every linear actuator, in the order of the governor arrays."""
        return [(name, cable) for name, skeleton in self.digits.items() for cable in skeleton.cable_names]

    @property
    def actuators(self) -> list[str]:
        """Key of every linear actuator (one per cable), in the order of the governor arrays.

        The key of a digit's first cable is the digit name; a further cable
        has ``"<digit>_<cable name>"`` (``"index"``, ``"index_knuckle"``), like
        the slider body keys. Cables, tensions, travels and routes of the hand
        are keyed or ordered by these keys.
        """
        return [
            actuator_key(name, c, cable)
            for name, skeleton in self.digits.items()
            for c, cable in enumerate(skeleton.cable_names)
        ]

    def actuator_index(self, digit: str, cable: int | str | None = 0) -> int:
        """Index of a cable's actuator in the governor arrays.

        Args:
            digit: Digit name.
            cable: Cable name or index within the digit; the first cable by default.

        Returns:
            The index into :attr:`actuators`.
        """
        start = 0
        for name, skeleton in self.digits.items():
            if name == digit:
                return start + skeleton.cable_index(cable)
            start += len(skeleton.cable_names)
        raise ValueError(f"the hand has no digit {digit!r}")


def actuator_key(digit: str, cable_index: int, cable_name: str) -> str:
    """Key of a cable and its actuator in the hand-level dictionaries.

    Args:
        digit: Digit name.
        cable_index: Index of the cable in the digit.
        cable_name: Name of the cable.

    Returns:
        The digit name for the first cable of a digit, ``"<digit>_<cable name>"`` for a further one.
    """
    return digit if cable_index == 0 else f"{digit}_{cable_name}"


def hand_cable_tensions(spec: HandSpec, commands: dict[str, Any], names=None) -> dict[str, float]:
    """Turn one command per digit into one tension per cable.

    A digit command is one of:

    - a number: every cable of the digit carries that tension [N];
    - a dict by cable name, e.g. ``{"curl": 14.0, "knuckle": 20.0}`` [N];
    - for a two-cable digit, the name of a pose of :func:`two_cable_poses`
      (``"hook"``, ``"knuckle_only"``, ...);
    - for a two-cable digit, a dict with ``"curl"`` [N] and ``"knuckle_angle"``
      [rad] and optionally ``"knuckle_torque"`` [N*m]: the knuckle tension that
      holds that first-joint angle at that curl tension
      (:func:`cable_tensions_for_knuckle`).

    Commands may also be given per actuator key (``"index_knuckle": 20.0``);
    such an entry overrides what the digit command gives for that cable.

    Args:
        spec: The hand spec.
        commands: Command per digit name (or tension per actuator key).
        names: Digits to resolve; the digits named in ``commands`` if ``None``.

    Returns:
        Tension [N] per actuator key (see :attr:`HandSkeleton.actuators`) for
        every cable of the digits that have a command.
    """
    out: dict[str, float] = {}
    keys = {}
    for name, digit in spec.digits.items():
        for c, cable in enumerate(digit.cables):
            keys[actuator_key(name, c, cable.name)] = (name, cable.name)
    for name in commands if names is None else names:
        if name not in spec.digits or name not in commands:
            continue
        digit = spec.digits[name]
        command = commands[name]
        if isinstance(command, str):
            tension = two_cable_poses(spec, name)[command]
        elif isinstance(command, dict) and "knuckle_angle" in command:
            tension = cable_tensions_for_knuckle(
                spec, name, command["curl"], command["knuckle_angle"], command.get("knuckle_torque", 0.0)
            )
        elif isinstance(command, dict):
            tension = command
        else:
            tension = dict.fromkeys(digit.cable_names, float(command))
        for c, cable in enumerate(digit.cables):
            if cable.name in tension:
                out[actuator_key(name, c, cable.name)] = float(tension[cable.name])
    for key, value in commands.items():
        if key in keys and key not in spec.digits:
            out[key] = float(value)
    return out


def _yaw_matrix(angle: float) -> np.ndarray:
    """Rotation about ``+y`` in the convention of the spec: a positive angle turns ``+x`` toward ``-z``."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _palm_mass_properties(spec: HandSpec) -> tuple[np.ndarray, np.ndarray]:
    """Center of mass [m] and inertia about it [kg*m^2] of the palm, in the palm frame.

    The palm mass is spread over the solid boxes of the spec in proportion to
    their volume (overlaps are counted twice).
    """
    y0, y1 = spec.palm.thickness_y
    parts = []
    for box in spec.palm.solid_boxes:
        by0, by1 = (y0, y1) if box.y is None else box.y
        size = np.array([box.x[1] - box.x[0], by1 - by0, box.z[1] - box.z[0]])
        center = np.array([0.5 * (box.x[0] + box.x[1]), 0.5 * (by0 + by1), 0.5 * (box.z[0] + box.z[1])])
        rot = np.eye(3)
        if box.axes is not None:
            # the box is given in its own frame at the pivot
            rot = np.array(box.axes, dtype=float).T
            center = np.array(box.pivot) + rot @ center
        elif box.yaw is not None and box.pivot is not None:
            rot = _yaw_matrix(box.yaw)
            pivot = np.array(box.pivot)
            center = pivot + rot @ (center - pivot)
        parts.append((float(np.prod(size)), center, rot, size))
    volume = sum(part[0] for part in parts)
    com = sum(part[0] * part[1] for part in parts) / volume
    inertia = np.zeros((3, 3))
    for part_volume, center, rot, size in parts:
        mass = spec.palm.mass * part_volume / volume
        sx, sy, sz = size * size
        local = mass / 12.0 * np.diag([sy + sz, sx + sz, sx + sy])
        d = center - com
        inertia += rot @ local @ rot.T + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return com, inertia


def add_hand_palm(
    builder: newton.ModelBuilder,
    spec: HandSpec,
    wrist: str = "fixed",
    palm_xform: wp.transform | None = None,
    actuator_gains: tuple[float, float] | None = None,
) -> HandSkeleton:
    """Add the palm and its root joint to a builder and start a hand skeleton.

    The palm body frame is the palm frame. With ``wrist="fixed"`` the palm is
    a dynamic body on a fixed joint to the world. With ``wrist="kinematic"``
    the palm is a kinematic body on a free joint; a :class:`WristDriver`
    writes its pose at run time.

    Args:
        builder: The model builder.
        spec: The hand spec.
        wrist: ``"fixed"`` or ``"kinematic"``.
        palm_xform: World pose of the palm frame; identity if ``None``.
        actuator_gains: Position gain [N/m] and velocity gain [N*s/m] of the
            slider drives; the values of the spec if ``None``.

    Returns:
        The hand skeleton without digits; add them with :func:`add_hand_digit`.
    """
    if wrist not in HAND_WRISTS:
        raise ValueError(f"wrist must be one of {HAND_WRISTS}, got {wrist!r}")
    if palm_xform is None:
        palm_xform = wp.transform_identity()
    com, inertia = _palm_mass_properties(spec)
    palm = builder.add_link(
        xform=palm_xform,
        com=wp.vec3(*com.tolist()),
        inertia=wp.mat33(*inertia.flatten().tolist()),
        mass=spec.palm.mass,
        is_kinematic=wrist == "kinematic",
        label="palm",
    )
    if wrist == "kinematic":
        root_joint = builder.add_joint_free(child=palm, label="wrist")
    else:
        root_joint = builder.add_joint_fixed(
            parent=-1, child=palm, parent_xform=palm_xform, child_xform=wp.transform_identity(), label="wrist"
        )
    # the opposition hinge takes the stop gains of the flexion joints
    stop = spec.digits["thumb"].joints[0]
    opposition_limit_gains = (stop.limit_ke, stop.limit_kd)
    if actuator_gains is None:
        actuator_gains = (spec.actuator.kp, spec.actuator.kd)
    return HandSkeleton(
        wrist=wrist,
        palm=palm,
        palm_xform=palm_xform,
        root_joint=root_joint,
        q_root=builder.joint_q_start[root_joint],
        opposition_limit_gains=(float(opposition_limit_gains[0]), float(opposition_limit_gains[1])),
        actuator_gains=(float(actuator_gains[0]), float(actuator_gains[1])),
        bodies={"palm": palm},
    )


def _add_thumb_column(builder: newton.ModelBuilder, spec: HandSpec, hand: HandSkeleton, opposition: float):
    """Add the thumb column on its opposition hinge; the column body frame is the thumb digit frame."""
    mount = spec.digits["thumb"].mount
    column = spec.palm.thumb_column
    drive = spec.actuator.opposition
    origin0, rot0 = digit_frame_in_palm(spec, "thumb", 0.0)
    origin, rot = digit_frame_in_palm(spec, "thumb", opposition)
    hinge = np.array(mount.hinge_point)
    # mass: a solid cylinder on the hinge line, centered between the two ends of the column root
    line = rot0.T @ np.array(mount.slider_line_direction)  # toward the thumb, in the column frame
    shift = 0.5 * (column.root_ahead_of_hinge_point - column.root_behind_hinge_point)
    com = rot0.T @ (hinge - origin0) + shift * line
    length = column.root_ahead_of_hinge_point + column.root_behind_hinge_point
    axial = 0.5 * column.mass * column.radius**2
    transverse = column.mass / 12.0 * (3.0 * column.radius**2 + length**2)
    inertia = transverse * np.eye(3) + (axial - transverse) * np.outer(line, line)
    body = builder.add_link(
        xform=wp.transform_multiply(hand.palm_xform, frame_to_transform(origin, rot)),
        com=wp.vec3(*com.tolist()),
        inertia=wp.mat33(*inertia.flatten().tolist()),
        mass=column.mass,
        label="thumb_column",
    )
    # hinge frame: at the hinge point, parallel to the palm frame; at zero opposition it has the same
    # world pose seen from both bodies
    parent_xform = wp.transform(wp.vec3(*hinge.tolist()), wp.quat_identity())
    child_xform = wp.transform_multiply(wp.transform_inverse(frame_to_transform(origin0, rot0)), parent_xform)
    limit_ke, limit_kd = hand.opposition_limit_gains
    joint = builder.add_joint_revolute(
        parent=hand.palm,
        child=body,
        parent_xform=parent_xform,
        child_xform=child_xform,
        axis=wp.vec3(*mount.hinge_axis),
        limit_lower=mount.hinge_lower,
        limit_upper=mount.hinge_upper,
        limit_ke=limit_ke,
        limit_kd=limit_kd,
        target_q=opposition,
        target_ke=drive["ke"],
        target_kd=drive["kd"],
        actuator_mode=newton.JointTargetMode.POSITION,
        friction=0.0,
        label="thumb_opposition",
    )
    builder.joint_q[builder.joint_q_start[joint]] = float(opposition)
    hand.thumb_column = body
    hand.joint_opposition = joint
    hand.q_opposition = builder.joint_q_start[joint]
    hand.dof_opposition = builder.joint_qd_start[joint]
    hand.opposition_lower = mount.hinge_lower
    hand.opposition_upper = mount.hinge_upper
    hand.opposition_ke = float(drive["ke"])
    hand.opposition_kd = float(drive["kd"])
    hand.opposition_torque_limit = float(drive["torque_limit"])
    hand.opposition_speed_limit = float(drive["speed_limit"])
    hand.opposition_parent_xform = parent_xform
    hand.opposition_child_xform = child_xform
    hand.opposition_axis = tuple(float(v) for v in mount.hinge_axis)
    hand.bodies["thumb_column"] = body


def add_hand_digit(
    builder: newton.ModelBuilder,
    spec: HandSpec,
    hand: HandSkeleton,
    name: str,
    opposition: float = 0.0,
    q=(0.0, 0.0, 0.0),
    travel: float = 0.0,
) -> DigitSkeleton:
    """Add one digit to a hand skeleton.

    A finger is carried by the palm at :func:`digit_frame_in_palm`. The thumb
    is carried by the thumb column, which this function adds first on the
    direct-drive opposition hinge; the thumb slider rides on the column, so
    the thumb cable never crosses the opposition hinge. The joints of the
    digit are contiguous: [opposition], MCP, PIP, DIP, slider. Add the cable
    of the digit right after this call if it has joints of its own.

    Args:
        builder: The model builder.
        spec: The hand spec.
        hand: The hand skeleton from :func:`add_hand_palm`.
        name: Digit name.
        opposition: Initial thumb opposition angle [rad]; ignored for fingers.
        q: Initial joint angles [rad].
        travel: Initial slider travel [m].

    Returns:
        The digit skeleton; it is also stored in ``hand.digits``.
    """
    if name in hand.digits:
        raise ValueError(f"digit {name!r} is already part of the hand")
    joints = []
    if isinstance(spec.digits[name].mount, ThumbMountSpec):
        _add_thumb_column(builder, spec, hand, opposition)
        joints.append(hand.joint_opposition)
        skeleton = add_digit_skeleton(builder, spec, name, hand.thumb_column, q=q, travel=travel)
    else:
        xform = frame_to_transform(*digit_frame_in_palm(spec, name))
        skeleton = add_digit_skeleton(builder, spec, name, hand.palm, base_xform_digit=xform, q=q, travel=travel)
    kp, kd = hand.actuator_gains
    for dof in skeleton.dof_sliders:
        builder.joint_target_ke[dof] = kp
        builder.joint_target_kd[dof] = kd
    skeleton.slider_kp, skeleton.slider_kd = kp, kd
    joints += skeleton.joints
    hand.digits[name] = skeleton
    hand.digit_joints[name] = joints
    # body keys: "<digit>/slider" is the carriage of the first cable, "<digit>/slider_<cable>" of a further one
    for key in (*BODY_KEYS[1:], *(skeleton.slider_key(c) for c in range(len(skeleton.cable_names)))):
        hand.bodies[f"{name}/{key}"] = skeleton.bodies[key]
    return skeleton


def add_hand_skeleton(
    builder: newton.ModelBuilder,
    spec: HandSpec,
    wrist: str = "fixed",
    palm_xform: wp.transform | None = None,
    opposition: float = 0.0,
    q: dict | None = None,
    travel: dict | None = None,
    actuator_gains: tuple[float, float] | None = None,
    after_digit=None,
) -> HandSkeleton:
    """Add the rigid skeleton of the hand to a builder, without articulation.

    24 rigid bodies with all five digits of the default spec: the palm, the
    thumb column, three links per digit and one slider per cable (seven).
    The skeleton serves both cable models; a
    cable model that adds joints passes ``after_digit`` and adds the cable of
    each digit there, which keeps the joint order one contiguous block per
    digit. Finish with :func:`add_hand_articulation`.

    Args:
        builder: The model builder.
        spec: The hand spec.
        wrist: ``"fixed"`` or ``"kinematic"``, see :func:`add_hand_palm`.
        palm_xform: World pose of the palm frame; identity if ``None``.
        opposition: Initial thumb opposition angle [rad].
        q: Initial joint angles per digit name [rad]; missing digits start straight.
        travel: Initial slider travel per digit name [m]; missing digits start at zero.
        actuator_gains: See :func:`add_hand_palm`.
        after_digit: Callable ``(builder, hand, skeleton)`` run after each digit was added.

    Returns:
        The hand skeleton.
    """
    hand = add_hand_palm(builder, spec, wrist, palm_xform, actuator_gains)
    for name in spec.digits:
        skeleton = add_hand_digit(
            builder,
            spec,
            hand,
            name,
            opposition=opposition,
            q=(0.0, 0.0, 0.0) if q is None else q.get(name, (0.0, 0.0, 0.0)),
            travel=0.0 if travel is None else travel.get(name, 0.0),
        )
        if after_digit is not None:
            after_digit(builder, hand, skeleton)
    return hand


def add_hand_articulation(builder: newton.ModelBuilder, hand: HandSkeleton, label: str = "hand") -> list[int]:
    """Wrap the whole hand in one articulation with the palm as root.

    Every joint added since the root joint of the palm belongs to the hand:
    the skeleton joints and the joints the cable model added after each
    digit. Closed loops are allowed when there are such cable joints.

    Args:
        builder: The model builder.
        hand: The hand skeleton.
        label: Label of the articulation.

    Returns:
        The joint indices of the articulation.
    """
    joints = list(range(hand.root_joint, builder.joint_count))
    skeleton_joints = hand.joints
    if skeleton_joints != sorted(skeleton_joints) or skeleton_joints[0] != joints[0]:
        raise ValueError("the skeleton joints of the hand are not in ascending order from the root joint")
    builder.add_articulation(joints, label=label, allow_closed_loops=len(joints) > len(skeleton_joints))
    return joints


def add_hand_visuals(
    builder: newton.ModelBuilder, hand: HandSkeleton, parts=None, materials=None, spec: HandSpec | None = None
) -> dict[str, int]:
    """Attach the visual meshes of the hand to the bodies of a hand skeleton.

    Args:
        builder: The model builder that holds the skeleton.
        hand: The hand skeleton.
        parts: Visual parts; :func:`load_hand_visual_parts` of ``spec`` if ``None``.
        materials: Look per material tag; the default look if ``None``.
        spec: The hand spec the skeleton was built from; the default spec if ``None``.

    Returns:
        Mapping from ``"<body key>/<part name>"`` to the shape index.
    """
    if parts is None:
        parts = load_hand_visual_parts(spec)
    return cable_hand_visuals.add_visuals_to_builder(builder, parts, hand.bodies, materials)


def _build_hand_skeleton_model(
    spec: HandSpec | None = None,
    wrist: str = "fixed",
    palm_xform: wp.transform | None = None,
    opposition: float = 0.0,
    q: dict | None = None,
    travel: dict | None = None,
    gravity=(0.0, 0.0, 0.0),
    device=None,
    visuals: bool = False,
) -> tuple[newton.Model, HandSkeleton]:
    """Build and finalize the skeleton-only model of the hand (no cables).

    Args:
        spec: The hand spec; the default spec if ``None``.
        wrist: ``"fixed"`` or ``"kinematic"``, see :func:`add_hand_palm`.
        palm_xform: World pose of the palm frame; identity if ``None``.
        opposition: Initial thumb opposition angle [rad].
        q: Initial joint angles per digit name [rad].
        travel: Initial slider travel per digit name [m].
        gravity: Gravity vector of the model in the world frame [m/s^2].
        device: Device of the model.
        visuals: If ``True``, attach the shipped visual meshes.

    Returns:
        The finalized model and the hand skeleton.
    """
    if spec is None:
        spec = default_hand_spec()
    builder = newton.ModelBuilder(gravity=gravity)
    hand = add_hand_skeleton(builder, spec, wrist, palm_xform, opposition, q, travel)
    add_hand_articulation(builder, hand)
    if visuals:
        add_hand_visuals(builder, hand, spec=spec)
    model = builder.finalize(device=device)
    return model, hand


@wp.kernel
def wrist_driver_kernel(
    body: int,
    body_com: wp.array[wp.vec3],
    target: wp.array[wp.transform],
    dt: float,
    steps_left: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Move the kinematic palm one step along the straight path to the commanded pose."""
    steps = wp.max(steps_left[0], 1)
    alpha = 1.0 / float(steps)
    pose = body_q[body]
    p0 = wp.transform_get_translation(pose)
    r0 = wp.transform_get_rotation(pose)
    goal = target[0]
    p1 = p0 + alpha * (wp.transform_get_translation(goal) - p0)
    turn = wp.mul(wp.transform_get_rotation(goal), wp.quat_inverse(r0))
    if turn[3] < 0.0:
        turn = wp.quat(-turn[0], -turn[1], -turn[2], -turn[3])
    axis = wp.vec3(0.0, 0.0, 0.0)
    angle = float(0.0)
    wp.quat_to_axis_angle(turn, axis, angle)
    r1 = wp.normalize(wp.mul(wp.quat_from_axis_angle(axis, alpha * angle), r0))
    com = body_com[body]
    velocity = (p1 + wp.quat_rotate(r1, com) - p0 - wp.quat_rotate(r0, com)) / dt
    body_q[body] = wp.transform(p1, r1)
    body_qd[body] = wp.spatial_vector(velocity, axis * (alpha * angle / dt))
    steps_left[0] = steps - 1


class WristDriver:
    """Writes the pose and velocity of the kinematic palm from a commanded pose.

    The command is a palm pose and the number of solver steps in which to
    reach it. Every :meth:`step` moves the palm by an equal share of the
    remaining way (straight line and constant rotation rate) and writes the
    matching velocity, so a script that sets the pose of the end of each
    frame gets a smooth motion over the substeps of that frame, also inside a
    captured graph. When the count has run out the palm stays at the target.

    Create it with :func:`create_wrist_driver`.
    """

    def __init__(self):
        self.model: newton.Model = None
        self.body: int = -1
        """Body index of the palm."""
        self.target: wp.array = None
        """Commanded world pose of the palm frame, shape ``(1,)``."""
        self.steps_left: wp.array = None
        """Solver steps left to reach the target, shape ``(1,)``."""

    def set_target(self, pose, steps: int = 1):
        """Command a palm pose to be reached in a number of solver steps.

        Args:
            pose: Position [m] and quaternion ``x, y, z, w`` of the palm frame in the world.
            steps: Number of solver steps over which the pose is reached.
        """
        self.target.assign(np.asarray([[float(v) for v in pose]], dtype=np.float32))
        self.steps_left.fill_(max(int(steps), 1))

    def step(self, state: newton.State, dt: float):
        """Advance the palm pose in ``state``; call before the solver step of every substep.

        Args:
            state: State the solver step starts from.
            dt: Substep length [s].
        """
        wp.launch(
            wrist_driver_kernel,
            dim=1,
            inputs=[self.body, self.model.body_com, self.target, dt],
            outputs=[self.steps_left, state.body_q, state.body_qd],
            device=self.model.device,
        )


def create_wrist_driver(model: newton.Model, hand: HandSkeleton) -> WristDriver:
    """Allocate the wrist driver of a hand with a kinematic palm; the target starts at the build pose.

    Args:
        model: The finalized model.
        hand: The hand skeleton, built with ``wrist="kinematic"``.

    Returns:
        The wrist driver.
    """
    if hand.wrist != "kinematic":
        raise ValueError("the wrist driver needs a hand built with wrist='kinematic'")
    driver = WristDriver()
    driver.model = model
    driver.body = hand.palm
    driver.target = wp.array([hand.palm_xform], dtype=wp.transform, device=model.device)
    driver.steps_left = wp.ones(1, dtype=int, device=model.device)
    return driver


@wp.kernel
def opposition_drive_kernel(
    body_q: wp.array[wp.transform],
    palm: int,
    column: int,
    parent_xform: wp.transform,
    child_xform: wp.transform,
    axis: wp.vec3,
    target: wp.array[float],
    lower: float,
    upper: float,
    ke: float,
    torque_limit: float,
    speed_limit: float,
    dt: float,
    target_index: int,
    reference: wp.array[float],
    angle: wp.array[float],
    joint_target_q: wp.array[float],
):
    """Rate- and torque-limited reference of the thumb opposition drive."""
    # measured opposition angle: twist of the column against the palm about the hinge axis
    q_parent = wp.mul(wp.transform_get_rotation(body_q[palm]), wp.transform_get_rotation(parent_xform))
    q_child = wp.mul(wp.transform_get_rotation(body_q[column]), wp.transform_get_rotation(child_xform))
    rel = wp.mul(wp.quat_inverse(q_parent), q_child)
    phi = 2.0 * wp.atan2(wp.dot(wp.vec3(rel[0], rel[1], rel[2]), axis), rel[3])
    if phi > wp.pi:
        phi = phi - 2.0 * wp.pi
    if phi < -wp.pi:
        phi = phi + 2.0 * wp.pi
    goal = wp.clamp(target[0], lower, upper)
    ref = reference[0] + wp.clamp(goal - reference[0], -speed_limit * dt, speed_limit * dt)  # speed limit
    ref = wp.clamp(ref, phi - torque_limit / ke, phi + torque_limit / ke)  # torque limit
    reference[0] = ref
    angle[0] = phi
    joint_target_q[target_index] = ref


class HandControl:
    """Hand-level control: slider governor, thumb opposition, spring preload, weight and wrist.

    Everything is a kernel launch on device arrays, so :meth:`step` can run
    inside a captured graph; the commands are written between graph launches
    with the ``set_*`` methods. Create it with :func:`create_hand_control`.

    One substep is::

        control.step(state_0, ctrl, dt)  # clear forces, weights, opposition, governor, wrist
        solver.step(state_0, state_1, ctrl, None, dt)
        state_0, state_1 = state_1, state_0
    """

    def __init__(self):
        self.model: newton.Model = None
        self.spec: HandSpec = None
        self.hand: HandSkeleton = None
        self.governor: ActuatorGovernor = None
        """Governor of all sliders, one actuator per cable in the order of ``hand.actuators``."""
        self.preload: SpringPreload = None
        """Spring preload of all flexion joints."""
        self.weight: BodyWeight | None = None
        """Gravity as body forces, or ``None`` without gravity."""
        self.wrist: WristDriver | None = None
        """Wrist driver of a kinematic palm, or ``None`` for a fixed wrist."""
        self.opposition_target: wp.array = None
        """Commanded thumb opposition angle [rad], shape ``(1,)``."""
        self.opposition_reference: wp.array = None
        """Opposition drive target after the speed and torque limits [rad], shape ``(1,)``."""
        self.opposition_angle: wp.array = None
        """Opposition angle measured at the last update [rad], shape ``(1,)``."""
        self._opposition_target_index: int = -1

    def set_tension(self, tension):
        """Command cable tensions; switches all actuators to force mode.

        The tensions are written as given, stepped (see
        :meth:`ActuatorGovernor.set_command`); a :class:`HandScript` blends
        them between keyframes, which is how a two-cable digit should be
        commanded.

        Args:
            tension: One value for all cables, one per actuator in the order of
                ``hand.actuators`` [N], or a dict of commands per digit (see
                :func:`hand_cable_tensions`: a tension, a tension per cable or
                a pose name); cables without a command keep theirs.
        """
        if isinstance(tension, dict):
            command = self.governor.command.numpy()
            keys = self.hand.actuators
            for key, value in hand_cable_tensions(self.spec, tension).items():
                command[keys.index(key)] = value
            tension = command
        self.governor.set_command(tension, GOVERNOR_FORCE)

    def set_travel(self, travel):
        """Command slider travels; switches all actuators to position mode.

        Args:
            travel: One value for all sliders or one per actuator in the order of ``hand.actuators`` [m].
        """
        self.governor.set_command(travel, GOVERNOR_POSITION)

    def set_opposition(self, angle: float):
        """Command the thumb opposition angle.

        Args:
            angle: Opposition angle [rad].
        """
        self.opposition_target.fill_(float(angle))

    def set_wrist_target(self, pose, steps: int = 1):
        """Command the palm pose of a kinematic wrist, see :meth:`WristDriver.set_target`.

        Args:
            pose: Position [m] and quaternion ``x, y, z, w`` of the palm frame in the world.
            steps: Number of solver steps over which the pose is reached.
        """
        if self.wrist is None:
            raise ValueError("the hand has a fixed wrist")
        self.wrist.set_target(pose, steps)

    def apply_preload(self, control: newton.Control):
        """Write the spring preload into the joint forces; once after creating the control is enough.

        Args:
            control: Control whose ``joint_f`` receives the preload of the flexion joints.
        """
        self.preload.apply(control)

    def step(self, state: newton.State, control: newton.Control, dt: float):
        """Clear the body forces of ``state`` and write all hand inputs for one solver step.

        Args:
            state: State the solver step starts from.
            control: Control that receives the drive targets.
            dt: Substep length [s].
        """
        state.clear_forces()
        if self.weight is not None:
            self.weight.apply(state)
        hand = self.hand
        if hand.joint_opposition >= 0:
            wp.launch(
                opposition_drive_kernel,
                dim=1,
                inputs=[
                    state.body_q,
                    hand.palm,
                    hand.thumb_column,
                    hand.opposition_parent_xform,
                    hand.opposition_child_xform,
                    wp.vec3(*hand.opposition_axis),
                    self.opposition_target,
                    hand.opposition_lower,
                    hand.opposition_upper,
                    hand.opposition_ke,
                    hand.opposition_torque_limit,
                    hand.opposition_speed_limit,
                    dt,
                    self._opposition_target_index,
                ],
                outputs=[self.opposition_reference, self.opposition_angle, control.joint_target_q],
                device=self.model.device,
            )
        self.governor.step(state, control, dt)
        # The wrist moves last: the governor and the opposition drive measure the slider travel and the
        # opposition angle from the body poses, which must all belong to the same instant. With the palm
        # already moved, the travel is off by the palm's step and the cable tension by kp times that.
        if self.wrist is not None:
            self.wrist.step(state, dt)


def create_hand_control(
    model: newton.Model,
    spec: HandSpec,
    hand: HandSkeleton,
    gravity=None,
    start_tension: float | None = None,
    speed_filter: float = GOVERNOR_SPEED_FILTER,
    speed_margin: float = GOVERNOR_SPEED_MARGIN,
) -> HandControl:
    """Allocate the hand-level control on the model's device.

    The governor starts in force mode at the rest tension, the opposition
    command at the build angle and the wrist target at the build pose. The
    slider gains are those the skeleton was built with (``actuator_gains`` of
    :func:`add_hand_palm`).

    Args:
        model: The finalized model, built with zero gravity if ``gravity`` is given.
        spec: The hand spec.
        hand: The hand skeleton.
        gravity: Gravity vector in the world frame [m/s^2], applied as body
            forces (see :func:`create_body_weight`); ``None`` for no weight.
        start_tension: Cable tension the model was built with [N] (``pretension`` of
            :func:`build_tendon_hand_model`); the rest tension of the spec if ``None``.
        speed_filter: Filter time of the carriage speed guard of the governor [s].
        speed_margin: Margin of the governor reference rate below the speed limit [m/s].

    Returns:
        The hand control.
    """
    device = model.device
    control = HandControl()
    control.model = model
    control.spec = spec
    control.hand = hand
    skeletons = hand.skeletons
    control.governor = create_actuator_governor(
        model, spec, skeletons, speed_filter=speed_filter, speed_margin=speed_margin
    )
    if start_tension is None:
        start_tension = spec.cable.tension_rest
    # The cables are built carrying `start_tension`, so the drives hold it from the first step: the
    # references start that far ahead of the sliders. (A start transient of the solver remains: the span
    # tensions move by up to about 10 % during the first 0.05 s while the stops and bearings take up load.)
    lead = [start_tension / skeleton.slider_kp for skeleton in skeletons for _ in skeleton.cable_names]
    control.governor.s_ref.assign(control.governor.s_ref.numpy() + np.array(lead, dtype=np.float32))
    control.preload = create_spring_preload(model, skeletons)
    if gravity is not None:
        control.weight = create_body_weight(model, gravity)
    if hand.wrist == "kinematic":
        control.wrist = create_wrist_driver(model, hand)
    start = 0.0
    if hand.joint_opposition >= 0:
        start = float(model.joint_q.numpy()[hand.q_opposition])
        control._opposition_target_index = int(model.joint_target_q_start.numpy()[hand.joint_opposition])
    control.opposition_target = wp.full(1, start, dtype=float, device=device)
    control.opposition_reference = wp.full(1, start, dtype=float, device=device)
    control.opposition_angle = wp.full(1, start, dtype=float, device=device)
    return control


def build_tendon_hand_model(
    spec: HandSpec | None = None,
    mu: float | None = None,
    wrist: str = "fixed",
    palm_xform: wp.transform | None = None,
    opposition: float = 0.0,
    q: dict | None = None,
    pretension: float | None = None,
    gravity=(0.0, 0.0, 0.0),
    device=None,
    limit_gains: tuple[float, float] | None = None,
    visuals: bool = False,
) -> tuple[newton.Model, HandSkeleton, dict[str, DigitTendon]]:
    """Build and finalize the hand with one massless tendon per cable.

    The tendons are returned per actuator key (``hand.actuators``: the digit
    name for a digit's first cable, ``"index_knuckle"`` for a further one).
    The model has no gravity by default: model gravity makes the tendon
    digits collapse, so the weight is applied as body forces (``gravity`` of
    :func:`create_hand_control`). The flexion joints and the opposition hinge
    get the joint stops of the spec, which are the soft stops the tendon digits
    need (:attr:`JointSpec.limit_ke`). The
    tendons are added before the bodies are colored. Use
    :func:`make_tendon_hand_solver` for the solver.

    Args:
        spec: The hand spec; the default spec if ``None``.
        mu: Friction coefficient on the rolling links; the spec default if ``None``.
        wrist: ``"fixed"`` or ``"kinematic"``, see :func:`add_hand_palm`.
        palm_xform: World pose of the palm frame; identity if ``None``.
        opposition: Initial thumb opposition angle [rad].
        q: Initial joint angles per digit name [rad]; the slider travel is the cable take-up of that pose.
        pretension: Cable tension at the build pose [N]; the rest tension of the spec if ``None``.
        gravity: Gravity vector of the model [m/s^2]; keep zero and apply the weight as body forces.
        device: Device of the model.
        limit_gains: Limit stiffness [N*m/rad] and damping [N*m*s/rad] of the
            flexion joints; ``None`` keeps the values of the spec.
        visuals: If ``True``, attach the shipped visual meshes.

    Returns:
        The finalized model, the hand skeleton and the tendon indices per actuator key.
    """
    if spec is None:
        spec = default_hand_spec()
    q = {} if q is None else {name: tuple(float(v) for v in angles) for name, angles in q.items()}
    travel = hand_takeup(spec, q)
    builder = newton.ModelBuilder(gravity=gravity)
    hand = add_hand_skeleton(
        builder,
        spec,
        wrist,
        palm_xform,
        opposition,
        q,
        travel,
    )
    tendons = add_hand_tendons(builder, spec, hand, mu, pretension, q, travel, limit_gains)
    add_hand_articulation(builder, hand)
    if visuals:
        add_hand_visuals(builder, hand, spec=spec)
    builder.color()
    model = builder.finalize(device=device)
    return model, hand, tendons


def hand_takeup(spec: HandSpec, q: dict) -> dict[str, list[float]]:
    """Slider travels that take up the cables of a hand pose.

    Args:
        spec: The hand spec.
        q: Joint angles per digit name [rad].

    Returns:
        Per digit name, the take-up of every cable of the digit [m].
    """
    return {
        name: [cable_takeup(spec.digits[name], angles, c) for c in range(len(spec.digits[name].cables))]
        for name, angles in q.items()
    }


def add_hand_tendons(
    builder: newton.ModelBuilder,
    spec: HandSpec,
    hand: HandSkeleton,
    mu: float | None = None,
    pretension: float | None = None,
    q: dict | None = None,
    travel: dict | None = None,
    limit_gains: tuple[float, float] | None = None,
) -> dict[str, DigitTendon]:
    """Add one massless tendon per cable of every digit of a hand skeleton.

    Args:
        builder: The model builder that holds the skeleton.
        spec: The hand spec.
        hand: The hand skeleton.
        mu: Friction coefficient on the rolling links; the spec default if ``None``.
        pretension: Cable tension at the build pose [N]; the rest tension of the spec if ``None``.
        q: Joint angles the digits were built at, per digit name [rad].
        travel: Slider travels the digits were built at, per digit name [m] (one per cable).
        limit_gains: Limit stiffness [N*m/rad] and damping [N*m*s/rad] of the
            flexion joints; ``None`` keeps the values of the spec.

    Returns:
        The tendon indices per actuator key, in the order of ``hand.actuators``.
    """
    q = {} if q is None else q
    travel = {} if travel is None else travel
    tendons = {}
    for name, skeleton in hand.digits.items():
        if limit_gains is not None:
            set_digit_limit_gains(builder, skeleton, limit_gains[0], limit_gains[1])
        added = add_digit_tendons(
            builder,
            spec,
            skeleton,
            mu=mu,
            pretension=pretension,
            q=q.get(name, (0.0, 0.0, 0.0)),
            travel=travel.get(name, 0.0),
        )
        for c, tendon in enumerate(added):
            tendons[actuator_key(name, c, tendon.cable)] = tendon
    return tendons


def make_tendon_hand_solver(model: newton.Model, friction: bool | None = None, **overrides) -> newton.solvers.SolverVBD:
    """Create the solver for the tendon hand: the settings of :func:`tendon_solver_settings`.

    Run it at :data:`TENDON_FPS` frames per second with
    :data:`TENDON_SUBSTEPS` solver steps per frame.

    Args:
        model: The model from :func:`build_tendon_hand_model`.
        friction: See :func:`make_tendon_solver`.
        **overrides: Solver keyword arguments that replace the defaults.

    Returns:
        The solver. Do not call ``solver.reset()`` on it; build a new solver instead.
    """
    return make_tendon_solver(model, friction, **overrides)


def hand_joint_state(model: newton.Model, state: newton.State, hand: HandSkeleton) -> dict[str, Any]:
    """Read the joint coordinates of the hand from the body poses of a state.

    Runs inverse kinematics into ``state.joint_q`` and ``state.joint_qd``.

    Args:
        model: The model.
        state: The state to read.
        hand: The hand skeleton.

    Returns:
        ``"q"``: flexion angles [rad], shape ``(digit_count, 3)`` in the
        order of ``hand.digits``; ``"travel"``: slider travels [m], shape
        ``(actuator_count,)`` in the order of ``hand.actuators`` (one per
        cable); ``"opposition"``: thumb opposition angle [rad] (zero without
        a thumb).
    """
    newton.eval_ik(model, state, state.joint_q, state.joint_qd)
    joint_q = state.joint_q.numpy().astype(float)
    skeletons = hand.skeletons
    return {
        "q": np.array([joint_q[list(skeleton.q_flexion)] for skeleton in skeletons]),
        "travel": np.array([joint_q[index] for skeleton in skeletons for index in skeleton.q_sliders]),
        "opposition": float(joint_q[hand.q_opposition]) if hand.joint_opposition >= 0 else 0.0,
    }


def hand_span_tensions(solver, tendons: dict[str, DigitTendon]) -> dict[str, np.ndarray]:
    """Material tension of every span of every tendon of the hand, actuator side first.

    Args:
        solver: The solver that steps the model.
        tendons: The tendon indices per actuator key.

    Returns:
        One array of span tensions [N] per actuator key.
    """
    tension = solver.tendon_seg_material_tension.numpy().astype(float)
    return {name: tension[tendon.segments] for name, tendon in tendons.items()}


def _hand_cable_routes(spec: HandSpec, hand: HandSkeleton, body_q) -> dict[str, CableRoute]:
    """Tangent-line and arc route of every cable of the hand through its stations at the current body poses.

    The station centers are taken from the body poses and written in the
    digit frame, so the routes show whether each cable still runs as designed:
    the wrap of every guide and sheave (``route.wraps[1:-1]``) must stay
    above zero and below 180 deg (the design keeps 20 to 160 deg).

    Args:
        spec: The hand spec.
        hand: The hand skeleton.
        body_q: Body poses, shape ``(body_count, 7)``.

    Returns:
        The route per actuator key (``hand.actuators``).
    """
    body_q = np.asarray(body_q, dtype=float)
    routes = {}
    for name, skeleton in hand.digits.items():
        frame = np.array(
            wp.transform_multiply(wp.transform(*body_q[skeleton.bodies["base"]]), skeleton.base_xform_digit)
        )
        for c, cable in enumerate(spec.digits[name].cables):
            centers = []
            for station in cable.stations:
                local = np.array([station.pos[0], station.pos[1], cable.plane_z])
                if station.body == "base":  # stations on the base are given in the digit frame
                    centers.append(local)
                    continue
                pose = body_q[skeleton.bodies[station.body]]
                world = pose[:3] + _quat_rotate_np(pose[3:], local)
                centers.append(_quat_rotate_inv_np(frame[3:], world - frame[:3]))
            centers = np.array(centers)
            radii = [station.radius for station in cable.stations]
            senses = [station.sense for station in cable.stations]
            routes[actuator_key(name, c, cable.name)] = route_through(centers[:, :2], radii, senses)
    return routes


def _require_kinematic_palm(wrist: str):
    """Refuse a rod hand on a palm that is not prescribed."""
    if wrist != "kinematic":
        raise ValueError(
            f"the rod hand needs a kinematic palm (wrist='kinematic'), got wrist={wrist!r}: with rod cables the "
            "body that carries the first stations must be prescribed; hold the wrist target still for a fixed wrist"
        )


def build_rod_hand_model(
    spec: HandSpec | None = None,
    wrist: str = "kinematic",
    palm_xform: wp.transform | None = None,
    opposition: float = 0.0,
    q: dict | None = None,
    device=None,
    contact_ke: float | None = None,
    visuals: bool = False,
) -> tuple[newton.Model, HandSkeleton, dict[str, RodCable]]:
    """Build and finalize the hand with one rod per cable.

    The stations and the rods of a digit are added right after the joints of
    that digit (:func:`add_rod_cables`), so the whole hand is one articulation
    with the palm as root, closed loops, and one contiguous block of joints
    per digit: [opposition], MCP, PIP, DIP, one slider per cable, then per
    cable the rod joints and the two anchors. The default spec gives 24 rigid
    bodies and 931 rod segments (955 bodies): seven cables, 1 mm segments on
    the six finger cables and 2 mm on the thumb's.

    The palm is always kinematic (a prescribed body): the rods start
    reliably only when the body that carries the first stations of the
    fingers is prescribed, as the base of the single-finger models is. With a
    palm held by a fixed joint (a dynamic body inside the articulation) rods
    of 1 mm segments blow up in the first frame in most runs. A scene that
    wants a fixed wrist leaves the wrist target at the build pose; any other
    ``wrist`` raises.

    Collision: every cable has its own collision group for its rod capsules
    and station shapes, and rod-rod and station-station pairs of a cable are
    filtered. A rod therefore touches the stations of its own cable and
    nothing else: no other rod, no other digit, no link, not the palm.

    The model has no gravity: apply the weight as body forces (``gravity`` of
    :func:`create_hand_control`). Use :func:`make_rod_hand_solver` and
    :func:`create_rod_collision_pipeline`. The solver factory matters: the
    palm carries the first stations of four fingers and collects the rod
    contacts of all of them, more than the per-body contact buffer of a
    single finger holds; :func:`make_rod_hand_solver` sets the buffer of
    :func:`rod_hand_contact_buffer` (3072 for the default spec). An overflow
    is only printed to the console, see that function.

    Args:
        spec: The hand spec; the default spec if ``None``.
        wrist: Must be ``"kinematic"`` (see above and :func:`add_hand_palm`).
        palm_xform: World pose of the palm frame; identity if ``None``.
        opposition: Initial thumb opposition angle [rad].
        q: Joint angles of the start pose per digit name [rad]; the slider
            travels are those of the rod start poses (:func:`rod_start_travels`).
        device: Device of the model.
        contact_ke: Rod contact stiffness [N/m]; the value of the spec if ``None``.
        visuals: If ``True``, attach the shipped visual meshes.

    Returns:
        The finalized model, the hand skeleton and the rod cable per actuator
        key (``hand.actuators``).

    Raises:
        ValueError: If ``wrist`` is not ``"kinematic"``.
    """
    if spec is None:
        spec = default_hand_spec()
    _require_kinematic_palm(wrist)
    names = list(spec.digits)
    q = {} if q is None else {name: tuple(float(v) for v in angles) for name, angles in q.items()}
    start_q = {name: q.get(name, (0.0, 0.0, 0.0)) for name in names}
    travel = {name: rod_start_travels(spec, name, start_q[name]) for name in names}
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    configure_rod_builder(builder)
    cables: dict[str, RodCable] = {}

    def add_cable(builder, hand, skeleton):
        # one rod and one collision group per cable; the keys are those of ``hand.actuators``
        added = add_rod_cables(
            builder,
            spec,
            skeleton,
            start_q[skeleton.name],
            collision_group=ROD_FIRST_COLLISION_GROUP + len(cables),
            contact_ke=contact_ke,
        )
        for c, cable in enumerate(added):
            cables[actuator_key(skeleton.name, c, cable.cable)] = cable

    hand = add_hand_skeleton(
        builder,
        spec,
        wrist,
        palm_xform,
        opposition,
        start_q,
        travel,
        after_digit=add_cable,
    )
    add_hand_articulation(builder, hand)
    if visuals:
        add_hand_visuals(builder, hand, spec=spec)
    builder.color()
    model = builder.finalize(device=device)
    return model, hand, cables


def rod_hand_contact_buffer(spec: HandSpec) -> int:
    """Contacts per body the rod solver must keep for the hand of a spec.

    The solver keeps a contact history per body. The palm carries the first
    stations of every finger, so its contacts are those of all fingers
    together: the sum of :func:`rod_contact_buffer` over the fingers; the
    thumb's stations sit on the thumb column, which needs the thumb's value.
    The larger of the two is returned: 3072 for the default spec (four
    fingers, six cables of 1 mm segments, 512 each). When the buffer is too
    small the solver does not raise and does not warn through
    :mod:`warnings`: it prints ``Per-body rigid contact buffer overflowed`` to
    the console and drops the contacts beyond the buffer, so check the
    console output after changing the stations, the contact gap or the number
    of digits on one body.

    Args:
        spec: The hand spec.

    Returns:
        The value for ``rigid_body_contact_buffer_size``.
    """
    thumbs = [name for name in spec.digits if isinstance(spec.digits[name].mount, ThumbMountSpec)]
    on_palm = sum(rod_contact_buffer(spec, name) for name in spec.digits if name not in thumbs)
    on_column = sum(rod_contact_buffer(spec, name) for name in thumbs)
    return max(on_palm, on_column)


def make_rod_hand_solver(
    model: newton.Model,
    substep_dt: float | None = None,
    iterations: int = ROD_ITERATIONS,
    spec: HandSpec | None = None,
    **overrides,
) -> newton.solvers.SolverVBD:
    """Create the solver for the rod hand: the settings of :func:`create_rod_solver`.

    The hand-specific settings are the contact buffer per body
    (:func:`rod_hand_contact_buffer`) and the reference inertia of the spec
    (:func:`rod_inertia_reference`). Create the collision pipeline before
    the solver; a captured graph needs the contact history allocated.

    Args:
        model: The model from :func:`build_rod_hand_model`.
        substep_dt: Substep length the solver will be stepped with [s]; a 60 Hz
            frame over :data:`ROD_SUBSTEPS` if ``None``.
        iterations: VBD iterations per substep.
        spec: The hand spec the model was built from; the default spec if ``None``.
        **overrides: Solver keyword arguments that replace the defaults.

    Returns:
        The solver.
    """
    if spec is None:
        spec = default_hand_spec()
    if substep_dt is None:
        substep_dt = 1.0 / (ROD_FPS * ROD_SUBSTEPS)
    overrides.setdefault("rigid_body_contact_buffer_size", rod_hand_contact_buffer(spec))
    inertia_reference = overrides.pop("inertia_reference", rod_inertia_reference(spec))
    return create_rod_solver(model, substep_dt, iterations, inertia_reference, spec, **overrides)


def create_rod_hand_probes(
    model: newton.Model, hand: HandSkeleton, cables: dict[str, RodCable]
) -> dict[str, RodTensionProbe]:
    """Create the tension and confinement probe of every rod cable of the hand.

    Args:
        model: The finalized model.
        hand: The hand skeleton.
        cables: The rod cable per actuator key (``hand.actuators``; seven for the default spec).

    Returns:
        One probe per actuator key, in the order of ``cables``.
    """
    return {key: RodTensionProbe(model, hand.digits[cable.name], cable) for key, cable in cables.items()}


def hand_rod_state(probes: dict[str, RodTensionProbe], body_q) -> dict[str, np.ndarray]:
    """Read the rod cables of the hand from the body poses.

    Args:
        probes: The probes from :func:`create_rod_hand_probes`.
        body_q: Body poses, shape ``(body_count, 7)``.

    Returns:
        Arrays over the cables (the actuator keys) in the order of ``probes``:
        ``"tension_actuator"`` and ``"tension_tip"`` (tension at the first
        and last rod joint [N]), ``"tension_min"`` and ``"tension_max"``
        (extremes along the rod [N]), ``"groove_offset"`` (largest distance of
        a rod vertex from the mid-plane of the groove that holds it, measured
        on the body that carries the groove [m]; the groove leaves the rod a
        side clearance of groove half width minus rod radius), ``"lateral"``
        (largest distance of a rod vertex from the digit plane on the base
        body [m]; this includes the sideways give of the links, see
        :meth:`RodTensionProbe.lateral_offset`) and ``"anchor_gap"`` (opening
        of the slider and tip anchors [m], shape ``(cable_count, 2)``).
    """
    body_q = np.asarray(body_q)
    tensions = [probe.tension(body_q) for probe in probes.values()]
    return {
        "tension_actuator": np.array([t[0] for t in tensions]),
        "tension_tip": np.array([t[-1] for t in tensions]),
        "tension_min": np.array([t.min() for t in tensions]),
        "tension_max": np.array([t.max() for t in tensions]),
        "groove_offset": np.array([np.abs(probe.groove_offset(body_q)).max() for probe in probes.values()]),
        "lateral": np.array([np.abs(probe.lateral_offset(body_q)).max() for probe in probes.values()]),
        "anchor_gap": np.array([probe.anchor_gaps(body_q) for probe in probes.values()]),
    }


# ---------------------------------------------------------------------------
# Grasping (pads, table and objects, keyframe script, grasp read-back; shared by both cable models)
# ---------------------------------------------------------------------------
# Scene frame: ``z`` up, the table top at ``z = 0``. Only the pads of the hand collide: one capsule per
# phalanx and two on the palm. Pads, objects and the table share one collision group; the rod cables and
# their stations use other groups, so a rod never touches a pad, an object or the table.

GRASP_COLLISION_GROUP = 1
"""Collision group of pads, objects and the table (rod cables start at :data:`ROD_FIRST_COLLISION_GROUP`)."""

GRASP_CONTACT_BUFFER = 128
"""Contact buffer per body for a grasp scene: an enveloped object touches up to 17 pads and the table."""

GRASP_OBJECTS = ("can", "ball", "block")
"""Names of the graspable objects of the spec."""

GRASP_SEAT_BALL_RADIUS = 0.003
"""Radius of the three seat balls that cradle a round object on its pedestal [m]."""

MINIMUM_JERK_PEAK_RATE = 1.875
"""Peak rate of the minimum-jerk blend, in units of distance over duration."""

MINIMUM_JERK_PEAK_ACCELERATION = 10.0 / math.sqrt(3.0)
"""Peak second derivative of the minimum-jerk blend, in units of distance over duration squared."""

HAND_SCRIPT_WRIST_SPEED = 0.5
"""Default limit of the scripted wrist speed [m/s]."""

HAND_SCRIPT_WRIST_ACCELERATION = GRAVITY
"""Default limit of the scripted wrist acceleration [m/s^2]: the 1 g the spring preload is sized for."""

HAND_SCRIPT_WRIST_ANGULAR_SPEED = math.pi
"""Default limit of the scripted wrist turning rate [rad/s]."""


@dataclass
class HandPads:
    """Collision pads of a hand, see :func:`add_hand_pads`.

    The arrays list the pads in the order of :attr:`keys`.
    """

    keys: list[str]
    """Pad keys: ``"<digit>/link1"`` .. ``"<digit>/link3"`` and ``"palm/<palm pad name>"``."""
    shapes: dict[str, int]
    """Shape index per pad key."""
    body: np.ndarray
    """Body index per pad, shape ``(pad_count,)``."""
    p0: np.ndarray
    """First end of the capsule axis in the body frame [m], shape ``(pad_count, 3)``."""
    p1: np.ndarray
    """Second end of the capsule axis in the body frame [m], shape ``(pad_count, 3)``."""
    radius: np.ndarray
    """Capsule radius [m], shape ``(pad_count,)``."""
    self_collision: bool
    """Whether pads of different digits (and digit pads against palm pads) collide."""


def _capsule_placement(p0, p1) -> tuple[wp.transform, float]:
    """Shape pose and half height of a capsule whose axis runs from ``p0`` to ``p1`` [m]."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    rot = wp.quat_identity()
    if length > 0.0:
        rot = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*(axis / length).tolist()))
    return wp.transform(wp.vec3(*(0.5 * (p0 + p1)).tolist()), rot), 0.5 * length


def grasp_shape_config(
    spec: HandSpec, mu: float, visible: bool = True, contact_ke: float | None = None
) -> newton.ModelBuilder.ShapeConfig:
    """Shape settings of pads, objects and the table: contact stiffness, the gap of the spec, no mass.

    Args:
        spec: The hand spec.
        mu: Friction coefficient of the shape. The solver uses the geometric
            mean of the two shapes of a pair.
        visible: Whether the shape is drawn.
        contact_ke: Contact stiffness [N/m]; the pad contact stiffness of the spec if ``None``.

    Returns:
        The shape configuration.
    """
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 0.0
    cfg.ke = spec.pads.ke if contact_ke is None else contact_ke
    cfg.mu = mu
    cfg.gap = spec.pads.gap
    cfg.collision_group = GRASP_COLLISION_GROUP
    cfg.is_visible = visible
    return cfg


def add_hand_pads(
    builder: newton.ModelBuilder,
    spec: HandSpec,
    hand: HandSkeleton,
    self_collision: bool = False,
    visible: bool = True,
    contact_ke: float | None = None,
) -> HandPads:
    """Add the collision pads of the spec to a hand skeleton: one capsule per phalanx and the palm pads.

    The pads are the only collision shapes of the skeleton and carry no mass.
    They collide with everything else in :data:`GRASP_COLLISION_GROUP`
    (objects, the table). Among themselves:

    - ``self_collision=False`` (default): no pad touches another pad. In the
      design no finger pad reaches the palm or a palm pad in the full fist and
      the pads of one digit stay 1 mm apart, so a fist needs no pad-pad
      contact. The price: the thumb can pass through a finger, and a pinch
      without an object between the pads is not held by contact.
    - ``self_collision=True``: pads of different digits collide, and digit
      pads collide with the palm pads; pads of the same digit and the two
      palm pads stay filtered.

    Works for both cable models: call it after the skeleton and the cables
    were added, before the bodies are colored.

    Args:
        builder: The model builder that holds the skeleton.
        spec: The hand spec.
        hand: The hand skeleton.
        self_collision: Whether pads of different digits collide.
        visible: Whether the pad capsules are drawn; switch off when the visual meshes are attached.
        contact_ke: Contact stiffness [N/m]; the pad contact stiffness of the spec if ``None``.

    Returns:
        The pads.
    """
    cfg = grasp_shape_config(spec, spec.pads.mu, visible, contact_ke)
    keys, shapes, bodies, p0s, p1s, radii, groups = [], {}, [], [], [], [], []
    for name in hand.digits:
        for pad in spec.digits[name].pads:
            keys.append(f"{name}/{pad.body}")
            bodies.append(hand.bodies[f"{name}/{pad.body}"])
            p0s.append((pad.p0[0], pad.p0[1], 0.0))
            p1s.append((pad.p1[0], pad.p1[1], 0.0))
            radii.append(pad.radius)
            groups.append(name)
    for pad in spec.palm.pads:
        keys.append(f"palm/{pad.name}")
        bodies.append(hand.palm)
        p0s.append(tuple(pad.p0))
        p1s.append(tuple(pad.p1))
        radii.append(pad.radius)
        groups.append("palm")
    for key, body, p0, p1, radius in zip(keys, bodies, p0s, p1s, radii, strict=True):
        xform, half_height = _capsule_placement(p0, p1)
        shapes[key] = builder.add_shape_capsule(
            body=body,
            xform=xform,
            radius=radius,
            half_height=half_height,
            cfg=cfg,
            label=f"pad/{key}",
        )
    for i, a in enumerate(keys):
        for j in range(i + 1, len(keys)):
            if not self_collision or groups[i] == groups[j]:
                builder.add_shape_collision_filter_pair(shapes[a], shapes[keys[j]])
    return HandPads(
        keys=keys,
        shapes=shapes,
        body=np.array(bodies, dtype=int),
        p0=np.array(p0s, dtype=float),
        p1=np.array(p1s, dtype=float),
        radius=np.array(radii, dtype=float),
        self_collision=self_collision,
    )


@dataclass
class GraspScene:
    """Indices and geometry of a grasp scene, see :func:`add_grasp_scene`."""

    pads: HandPads
    """Pads of the hand."""
    table_shape: int
    """Shape index of the table (a static box with its top at ``z = 0``)."""
    pedestal_shape: int
    """Shape index of the pedestal under the object, -1 without one."""
    seat_shapes: list[int]
    """Shape indices of the three seat balls on the pedestal of a round object; empty otherwise."""
    pedestal_height: float
    """Height of the object's lowest point above the table at the start [m], zero without a pedestal."""
    object_name: str
    """Name of the object in ``spec.objects``."""
    object_kind: str
    """``"cylinder"`` (axis along the object's ``z``), ``"sphere"`` or ``"box"``."""
    object_size: tuple[float, ...]
    """Cylinder: radius and half height; sphere: radius; box: the three half extents [m]."""
    object_mass: float
    """Object mass [kg]."""
    object_body: int
    """Body index of the object, a free body."""
    object_shape: int
    """Shape index of the object."""
    object_xform: wp.transform
    """World pose of the object at build time."""


def grasp_object_properties(spec: HandSpec, name: str) -> tuple[str, tuple[float, ...], float, np.ndarray]:
    """Shape, mass and inertia of a graspable object of the spec.

    The objects are homogeneous solids: the can a cylinder (inertia
    ``m r^2 / 2`` about its axis and ``m (3 r^2 + h^2) / 12`` across), the
    ball a sphere (``2 m r^2 / 5``), the block a box
    (``m (b^2 + c^2) / 12`` per axis). The center of mass is the body origin.

    Args:
        spec: The hand spec.
        name: Object name, one of :data:`GRASP_OBJECTS`.

    Returns:
        Kind (``"cylinder"``, ``"sphere"`` or ``"box"``), size [m] (cylinder:
        radius and half height along the object's ``z``; sphere: radius; box:
        half extents), mass [kg] and the principal inertia [kg*m^2], shape ``(3,)``.
    """
    data = spec.objects[name]
    mass = float(data["mass"])
    if "size" in data:
        size = tuple(0.5 * float(v) for v in data["size"])
        sx, sy, sz = (4.0 * v * v for v in size)
        inertia = mass / 12.0 * np.array([sy + sz, sx + sz, sx + sy])
        return "box", size, mass, inertia
    radius = float(data["radius"])
    if "height" in data:
        height = float(data["height"])
        across = mass / 12.0 * (3.0 * radius * radius + height * height)
        return "cylinder", (radius, 0.5 * height), mass, np.array([across, across, 0.5 * mass * radius * radius])
    return "sphere", (radius,), mass, np.full(3, 0.4 * mass * radius * radius)


def grasp_object_rest_height(spec: HandSpec, name: str) -> float:
    """Height of the object's center above the surface it stands on, in its upright pose.

    Args:
        spec: The hand spec.
        name: One of :data:`GRASP_OBJECTS`.

    Returns:
        The height [m].
    """
    kind, size, _, _ = grasp_object_properties(spec, name)
    return size[0] if kind == "sphere" else (size[1] if kind == "cylinder" else size[2])


def add_grasp_scene(
    builder: newton.ModelBuilder,
    spec: HandSpec,
    pads: HandPads,
    object_name: str,
    object_xform: wp.transform | None = None,
    table_half_size: tuple[float, float, float] = (0.2, 0.2, 0.01),
    pedestal: tuple[float, float] | None = None,
    contact_ke: float | None = None,
) -> GraspScene:
    """Add the table, an optional pedestal and one graspable object to a builder.

    Friction: the solver uses the geometric mean of the two shapes of a pair.
    The object gets the pad coefficient and the table (and pedestal) the
    square of the table coefficient over the pad coefficient, so that
    pad-object pairs have ``spec.pads.mu`` and object-table pairs
    ``spec.pads.table_mu``. All shapes use the contact gap of the spec and
    one contact stiffness.

    Args:
        builder: The model builder; ``z`` is up and the table top is ``z = 0``.
        spec: The hand spec.
        pads: The pads of the hand, from :func:`add_hand_pads`.
        object_name: One of :data:`GRASP_OBJECTS`.
        object_xform: World pose of the object; upright above the origin, standing
            on the table or the pedestal, if ``None``.
        table_half_size: Half extents of the table box [m].
        pedestal: Radius and height [m] of a static cylinder under the object
            (centered below the object's start position), or ``None``. The
            height is that of the object's lowest point. Under the ball the
            post is a little lower and carries three seat balls of radius
            :data:`GRASP_SEAT_BALL_RADIUS` on its rim: a ball on a flat top
            rolls off at the slightest push when it is set down.
        contact_ke: Contact stiffness [N/m]; the pad contact stiffness of the spec if ``None``.

    Returns:
        The scene.
    """
    kind, size, mass, inertia = grasp_object_properties(spec, object_name)
    surface_mu = spec.pads.table_mu**2 / spec.pads.mu
    surface_cfg = grasp_shape_config(spec, surface_mu, contact_ke=contact_ke)
    pedestal_height = 0.0 if pedestal is None else float(pedestal[1])
    if object_xform is None:
        height = pedestal_height + grasp_object_rest_height(spec, object_name)
        object_xform = wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity())
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
    if pedestal is not None:
        position = wp.transform_get_translation(object_xform)
        radius = float(pedestal[0])
        post_height = pedestal_height
        if kind == "sphere":
            # seat: the ball rests on three small balls on the rim of the post, its bottom at the pedestal height
            drop = math.sqrt((size[0] + GRASP_SEAT_BALL_RADIUS) ** 2 - radius**2)
            post_height = pedestal_height + size[0] - drop
            for i in range(3):
                angle = (0.25 + i / 3.0) * 2.0 * math.pi
                center = (position[0] + radius * math.cos(angle), position[1] + radius * math.sin(angle), post_height)
                seat_shapes.append(
                    builder.add_shape_sphere(
                        body=-1,
                        xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                        radius=GRASP_SEAT_BALL_RADIUS,
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
    body = builder.add_body(
        xform=object_xform,
        inertia=wp.mat33(*np.diag(inertia).flatten().tolist()),
        mass=mass,
        label=f"grasp/{object_name}",
    )
    cfg = grasp_shape_config(spec, spec.pads.mu, contact_ke=contact_ke)
    if kind == "cylinder":
        shape = builder.add_shape_cylinder(body=body, radius=size[0], half_height=size[1], cfg=cfg, label=object_name)
    elif kind == "sphere":
        shape = builder.add_shape_sphere(body=body, radius=size[0], cfg=cfg, label=object_name)
    else:
        shape = builder.add_shape_box(body=body, hx=size[0], hy=size[1], hz=size[2], cfg=cfg, label=object_name)
    return GraspScene(
        pads=pads,
        table_shape=table_shape,
        pedestal_shape=pedestal_shape,
        seat_shapes=seat_shapes,
        pedestal_height=pedestal_height,
        object_name=object_name,
        object_kind=kind,
        object_size=size,
        object_mass=mass,
        object_body=body,
        object_shape=shape,
        object_xform=object_xform,
    )


def build_tendon_hand_grasp_model(
    spec: HandSpec | None = None,
    object_name: str = "can",
    object_xform: wp.transform | None = None,
    palm_xform: wp.transform | None = None,
    opposition: float = 0.0,
    mu: float | None = None,
    pretension: float | None = None,
    wrist: str = "kinematic",
    device=None,
    pedestal: tuple[float, float] | None = None,
    contact_ke: float | None = None,
    visuals: bool = False,
) -> tuple[newton.Model, HandSkeleton, dict[str, DigitTendon], GraspScene]:
    """Build and finalize the massless-tendon hand with pads, a table and one object.

    The hand is that of :func:`build_tendon_hand_model` (same skeleton, soft
    stops and tendons) with a kinematic wrist by default. The model has no
    gravity: pass ``gravity=(0, 0, -GRAVITY)`` to :func:`create_hand_control`,
    which then also applies the weight of the object. Use
    :func:`make_tendon_hand_grasp_solver` and
    :func:`create_grasp_collision_pipeline`.

    Args:
        spec: The hand spec; the default spec if ``None``.
        object_name: One of :data:`GRASP_OBJECTS`.
        object_xform: World pose of the object, see :func:`add_grasp_scene`.
        palm_xform: World pose of the palm frame at the start; identity if ``None``.
        opposition: Initial thumb opposition angle [rad].
        mu: Cable friction coefficient on the rolling links; the spec default if ``None``.
        pretension: Cable tension at the build pose [N]; the rest tension of the spec if ``None``.
        wrist: ``"kinematic"`` or ``"fixed"``, see :func:`add_hand_palm`.
        device: Device of the model.
        pedestal: Radius and height [m] of a pedestal under the object, or ``None``.
        contact_ke: Contact stiffness of all shapes [N/m]; the pad contact stiffness of the spec if ``None``.
        visuals: If ``True``, attach the shipped visual meshes and hide the pad capsules.

    Returns:
        The finalized model, the hand skeleton, the tendon indices per actuator key and the scene.
    """
    if spec is None:
        spec = default_hand_spec()
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    hand = add_hand_skeleton(builder, spec, wrist, palm_xform, opposition)  # joint stops of the spec
    tendons = add_hand_tendons(builder, spec, hand, mu, pretension)
    add_hand_articulation(builder, hand)
    pads = add_hand_pads(builder, spec, hand, visible=not visuals, contact_ke=contact_ke)
    scene = add_grasp_scene(builder, spec, pads, object_name, object_xform, pedestal=pedestal, contact_ke=contact_ke)
    if visuals:
        add_hand_visuals(builder, hand, spec=spec)
    builder.color()
    model = builder.finalize(device=device)
    return model, hand, tendons, scene


def grasp_contact_settings() -> dict[str, Any]:
    """Solver settings a grasp scene adds to those of the cable model: contact warm start and buffer size.

    Returns:
        Keyword arguments for the solver constructor.
    """
    return {"rigid_contact_history": True, "rigid_body_contact_buffer_size": GRASP_CONTACT_BUFFER}


def make_tendon_hand_grasp_solver(model: newton.Model, **overrides) -> newton.solvers.SolverVBD:
    """Create the solver for the tendon hand in a grasp scene.

    The settings of :func:`make_tendon_hand_solver` plus
    :func:`grasp_contact_settings`. Create the collision pipeline
    (:func:`create_grasp_collision_pipeline`) before the solver: a captured
    graph needs the contact history allocated.

    Args:
        model: The model from :func:`build_tendon_hand_grasp_model`.
        **overrides: Solver keyword arguments that replace the defaults.

    Returns:
        The solver.
    """
    settings = grasp_contact_settings()
    settings.update(overrides)
    return make_tendon_hand_solver(model, **settings)


def build_rod_hand_grasp_model(
    spec: HandSpec | None = None,
    object_name: str = "can",
    object_xform: wp.transform | None = None,
    palm_xform: wp.transform | None = None,
    opposition: float = 0.0,
    wrist: str = "kinematic",
    device=None,
    pedestal: tuple[float, float] | None = None,
    contact_ke: float | None = None,
    visuals: bool = False,
) -> tuple[newton.Model, HandSkeleton, dict[str, RodCable], GraspScene]:
    """Build and finalize the rod hand with pads, a table and one object.

    The hand is that of :func:`build_rod_hand_model` (same skeleton, stations
    and rods, one collision group per cable) on its kinematic palm; pads and
    scene are those of
    :func:`build_tendon_hand_grasp_model`. The rods and stations keep their
    own collision groups, so the only shape pairs that may collide are a rod
    with a station of its own cable, a pad with the object or the scene, and
    the object with the scene. The object is a free body outside the hand
    articulation.

    The model has no gravity: pass ``gravity=(0, 0, -GRAVITY)`` and
    ``start_tension=0`` to :func:`create_hand_control`. Use
    :func:`make_rod_hand_grasp_solver` and
    :func:`create_grasp_collision_pipeline`, and start a run with a seating
    stroke of the cables (``seating_tension`` of :func:`grasp_plan`).

    Args:
        spec: The hand spec; the default spec if ``None``.
        object_name: One of :data:`GRASP_OBJECTS`.
        object_xform: World pose of the object, see :func:`add_grasp_scene`.
        palm_xform: World pose of the palm frame at the start; identity if ``None``.
        opposition: Initial thumb opposition angle [rad].
        wrist: Must be ``"kinematic"``, see :func:`build_rod_hand_model`.
        device: Device of the model.
        pedestal: Radius and height [m] of a pedestal under the object, or ``None``.
        contact_ke: Contact stiffness of pads, object and scene [N/m];
            the pad contact stiffness of the spec if ``None``. The rod contacts keep the rod value of the spec.
        visuals: If ``True``, attach the shipped visual meshes and hide the pad capsules.

    Returns:
        The finalized model, the hand skeleton, the rod cable per actuator key and the scene.

    Raises:
        ValueError: If ``wrist`` is not ``"kinematic"`` (see :func:`build_rod_hand_model`).
    """
    if spec is None:
        spec = default_hand_spec()
    _require_kinematic_palm(wrist)
    start_q = (0.0, 0.0, 0.0)
    travel = {name: rod_start_travels(spec, name, start_q) for name in spec.digits}
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    configure_rod_builder(builder)
    cables: dict[str, RodCable] = {}

    def add_cable(builder, hand, skeleton):
        added = add_rod_cables(
            builder,
            spec,
            skeleton,
            start_q,
            collision_group=ROD_FIRST_COLLISION_GROUP + len(cables),
        )
        for c, cable in enumerate(added):
            cables[actuator_key(skeleton.name, c, cable.cable)] = cable

    hand = add_hand_skeleton(builder, spec, wrist, palm_xform, opposition, travel=travel, after_digit=add_cable)
    add_hand_articulation(builder, hand)
    pads = add_hand_pads(builder, spec, hand, visible=not visuals, contact_ke=contact_ke)
    scene = add_grasp_scene(builder, spec, pads, object_name, object_xform, pedestal=pedestal, contact_ke=contact_ke)
    if visuals:
        add_hand_visuals(builder, hand, spec=spec)
    builder.color()
    model = builder.finalize(device=device)
    return model, hand, cables, scene


def make_rod_hand_grasp_solver(
    model: newton.Model,
    substep_dt: float | None = None,
    iterations: int = ROD_ITERATIONS,
    spec: HandSpec | None = None,
    **overrides,
) -> newton.solvers.SolverVBD:
    """Create the solver for the rod hand in a grasp scene: the settings of :func:`make_rod_hand_solver`.

    The rod solver already keeps the contact history, and its contact buffer
    per body (:func:`rod_hand_contact_buffer`) covers the pad contacts of the
    object. Create the collision pipeline
    (:func:`create_grasp_collision_pipeline`) before the solver.

    Args:
        model: The model from :func:`build_rod_hand_grasp_model`.
        substep_dt: Substep length the solver will be stepped with [s]; a 60 Hz
            frame over :data:`ROD_SUBSTEPS` if ``None``.
        iterations: VBD iterations per substep.
        spec: The hand spec the model was built from; the default spec if ``None``.
        **overrides: Solver keyword arguments that replace the defaults.

    Returns:
        The solver.
    """
    return make_rod_hand_solver(model, substep_dt, iterations, spec, **overrides)


def create_grasp_collision_pipeline(model: newton.Model) -> newton.CollisionPipeline:
    """Create the collision pipeline of a grasp scene; contacts are matched to the latest frame for warm starts.

    The same pipeline as :func:`create_rod_collision_pipeline`, so one
    pipeline serves the rod hand in a grasp scene.

    Args:
        model: The finalized model of a grasp scene.

    Returns:
        The collision pipeline.
    """
    return create_rod_collision_pipeline(model)


# --- keyframe script -------------------------------------------------------------------------------------


@dataclass
class HandKeyframe:
    """One keyframe of a hand script; a field left at ``None`` keeps the value of the previous keyframe."""

    time: float
    """Time of the keyframe [s]."""
    wrist: Any = None
    """World pose of the palm frame: position [m] and quaternion ``x, y, z, w``."""
    opposition: float | None = None
    """Thumb opposition angle [rad]."""
    tension: Any = None
    """Cable tension command [N]: one value for all cables, or a mapping of commands per digit name or
    actuator key (see :func:`hand_cable_tensions`: a tension for all cables of a digit, a tension per
    cable, the name of a two-cable pose, or a tension for one actuator key such as ``"index_knuckle"``).
    The cables named here are in force mode from this keyframe on."""
    travel: dict[str, float] | None = None
    """Slider travel command per actuator key [m]; these actuators are in position mode from this
    keyframe on."""


def minimum_jerk(u):
    """Minimum-jerk blend ``10 u^3 - 15 u^4 + 6 u^5``; rate and curvature vanish at both ends.

    Args:
        u: Normalized time, a number or an array; clamped to [0, 1].

    Returns:
        The blend factor between 0 and 1, with the shape of ``u``.
    """
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (10.0 + u * (6.0 * u - 15.0))


def resolve_hand_keyframes(
    keyframes,
    names,
    start_wrist=None,
    start_opposition: float = 0.0,
    start_tension: float = 0.0,
    spec: HandSpec | None = None,
) -> dict[str, np.ndarray]:
    """Fill the open fields of a keyframe list and return it as arrays.

    Args:
        keyframes: :class:`HandKeyframe` items with strictly increasing times; at least two.
        names: Actuator keys in the order of the hand (``hand.actuators``), one command column each.
        start_wrist: Palm pose before the first keyframe that sets one; identity if ``None``.
        start_opposition: Opposition angle before the first keyframe that sets one [rad].
        start_tension: Cable tension before the first keyframe that sets one [N].
        spec: The hand spec. With it, a tension mapping holds commands per
            digit (:func:`hand_cable_tensions`); without it, tensions per
            name of ``names``.

    Returns:
        ``"times"`` [s], shape ``(key_count,)``; ``"wrist"`` (position [m] and
        quaternion, consecutive quaternions in the same hemisphere), shape
        ``(key_count, 7)``; ``"opposition"`` [rad], shape ``(key_count,)``;
        ``"command"`` (tension [N] or travel [m]) and ``"mode"``
        (:data:`GOVERNOR_FORCE` or :data:`GOVERNOR_POSITION`), shape
        ``(key_count, actuator_count)``.
    """
    names = list(names)
    keyframes = list(keyframes)
    if len(keyframes) < 2:
        raise ValueError("a hand script needs at least two keyframes")
    times = np.array([key.time for key in keyframes], dtype=float)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("keyframe times must increase strictly")
    wrist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] if start_wrist is None else start_wrist, dtype=float)
    opposition = float(start_opposition)
    command = np.full(len(names), float(start_tension))
    mode = np.full(len(names), GOVERNOR_FORCE, dtype=int)
    rows = {"wrist": [], "opposition": [], "command": [], "mode": []}
    for key in keyframes:
        if key.wrist is not None:
            pose = np.array([float(v) for v in key.wrist])
            pose[3:] /= np.linalg.norm(pose[3:])
            if float(pose[3:] @ wrist[3:]) < 0.0:
                pose[3:] = -pose[3:]
            wrist = pose
        if key.opposition is not None:
            opposition = float(key.opposition)
        if key.tension is not None:
            if isinstance(key.tension, dict):
                tension = key.tension if spec is None else hand_cable_tensions(spec, key.tension)
                for name, value in tension.items():
                    command[names.index(name)] = float(value)
                    mode[names.index(name)] = GOVERNOR_FORCE
            else:
                command = np.broadcast_to(np.asarray(key.tension, dtype=float), command.shape).copy()
                mode[:] = GOVERNOR_FORCE
        if key.travel is not None:
            for name, value in key.travel.items():
                command[names.index(name)] = float(value)
                mode[names.index(name)] = GOVERNOR_POSITION
        rows["wrist"].append(wrist.copy())
        rows["opposition"].append(opposition)
        rows["command"].append(command.copy())
        rows["mode"].append(mode.copy())
    return {"times": times, **{name: np.array(values) for name, values in rows.items()}}


def _slerp_np(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    """Spherical interpolation between two unit quaternions ``x, y, z, w`` of the same hemisphere."""
    cos_half = float(np.clip(q0 @ q1, -1.0, 1.0))
    half = math.acos(cos_half)
    if half < 1.0e-6:
        q = (1.0 - s) * q0 + s * q1
    else:
        q = (math.sin((1.0 - s) * half) * q0 + math.sin(s * half) * q1) / math.sin(half)
    return q / np.linalg.norm(q)


def evaluate_hand_keyframes(keys: dict[str, np.ndarray], t: float) -> dict[str, Any]:
    """Evaluate resolved keyframes at time ``t`` on the host; the reference of :func:`hand_script_kernel`.

    Between two keyframes every value moves on the minimum-jerk blend: the
    palm position on a straight line, the palm orientation at a constant
    axis, the opposition angle and the cable commands linearly in the blend.
    A digit whose mode changes between two keyframes keeps the earlier mode
    and value until the later keyframe and switches there (the governor then
    moves at its speed limit). Before the first and after the last keyframe
    the values are held.

    Args:
        keys: The arrays from :func:`resolve_hand_keyframes`.
        t: Time [s].

    Returns:
        ``"wrist"``, shape ``(7,)``; ``"opposition"`` [rad]; ``"command"``
        and ``"mode"``, shape ``(digit_count,)``.
    """
    times = keys["times"]
    k = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2))
    s = float(minimum_jerk((t - times[k]) / (times[k + 1] - times[k])))
    w0, w1 = keys["wrist"][k], keys["wrist"][k + 1]
    wrist = np.concatenate([(1.0 - s) * w0[:3] + s * w1[:3], _slerp_np(w0[3:], w1[3:], s)])
    c0, c1 = keys["command"][k], keys["command"][k + 1]
    m0, m1 = keys["mode"][k], keys["mode"][k + 1]
    held = c1 if s >= 1.0 else c0
    return {
        "wrist": wrist,
        "opposition": float((1.0 - s) * keys["opposition"][k] + s * keys["opposition"][k + 1]),
        "command": np.where(m0 == m1, (1.0 - s) * c0 + s * c1, held),
        "mode": (m1 if s >= 1.0 else m0).copy(),
    }


def hand_keyframe_wrist_peaks(keys: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Peak wrist speed, acceleration and turning rate of every segment of resolved keyframes.

    On the minimum-jerk blend a move over distance ``d`` in time ``T`` peaks
    at ``1.875 d / T`` and ``5.77 d / T^2``.

    Args:
        keys: The arrays from :func:`resolve_hand_keyframes`.

    Returns:
        ``"speed"`` [m/s], ``"acceleration"`` [m/s^2] and ``"angular_speed"``
        [rad/s] of the palm frame origin, shape ``(key_count - 1,)`` each.
    """
    duration = np.diff(keys["times"])
    wrist = keys["wrist"]
    distance = np.linalg.norm(np.diff(wrist[:, :3], axis=0), axis=1)
    cos_half = np.clip(np.abs(np.sum(wrist[1:, 3:] * wrist[:-1, 3:], axis=1)), 0.0, 1.0)
    angle = 2.0 * np.arccos(cos_half)
    return {
        "speed": MINIMUM_JERK_PEAK_RATE * distance / duration,
        "acceleration": MINIMUM_JERK_PEAK_ACCELERATION * distance / duration**2,
        "angular_speed": MINIMUM_JERK_PEAK_RATE * angle / duration,
    }


@wp.kernel
def hand_script_kernel(
    times: wp.array[float],
    wrist: wp.array[wp.transform],
    opposition: wp.array[float],
    command: wp.array2d[float],
    mode: wp.array2d[int],
    dt: float,
    step_count: wp.array[int],
    wrist_target: wp.array[wp.transform],
    wrist_steps_left: wp.array[int],
    opposition_target: wp.array[float],
    governor_command: wp.array[float],
    governor_mode: wp.array[int],
):
    """Write the commands of the end of the coming solver step and advance the step count.

    The time is the step count times ``dt``; a clock summed up in single precision drifts.
    """
    step = step_count[0] + 1
    t = float(step) * dt
    last = times.shape[0] - 2
    k = int(0)
    for i in range(1, last + 1):
        if t >= times[i]:
            k = i
    u = wp.clamp((t - times[k]) / (times[k + 1] - times[k]), 0.0, 1.0)
    s = u * u * u * (10.0 + u * (6.0 * u - 15.0))
    p0 = wp.transform_get_translation(wrist[k])
    p1 = wp.transform_get_translation(wrist[k + 1])
    rot = wp.quat_slerp(wp.transform_get_rotation(wrist[k]), wp.transform_get_rotation(wrist[k + 1]), s)
    wrist_target[0] = wp.transform(p0 + s * (p1 - p0), wp.normalize(rot))
    wrist_steps_left[0] = 1
    opposition_target[0] = opposition[k] + s * (opposition[k + 1] - opposition[k])
    for d in range(governor_command.shape[0]):
        value = command[k, d]
        value_mode = mode[k, d]
        if value_mode == mode[k + 1, d]:
            value = value + s * (command[k + 1, d] - value)
        elif u >= 1.0:
            value = command[k + 1, d]
            value_mode = mode[k + 1, d]
        governor_command[d] = value
        governor_mode[d] = value_mode
    step_count[0] = step


class HandScript:
    """Timed keyframes of wrist pose, thumb opposition and cable commands, played on the device.

    :meth:`step` is one kernel launch that writes the commands of a
    :class:`HandControl` (wrist target, opposition target, governor command
    and mode), so a whole frame of substeps can run inside a captured graph
    without host code. One substep is::

        script.step(control, dt)  # commands of the end of this substep
        control.step(state_0, ctrl, dt)
        collision_pipeline.collide(state_0, contacts)  # in a scene with contacts
        solver.step(state_0, state_1, ctrl, contacts, dt)
        state_0, state_1 = state_1, state_0

    The interpolation is that of :func:`evaluate_hand_keyframes`. The wrist
    follows the script exactly (kinematic palm); the opposition angle and the
    cable commands are targets that the opposition drive and the actuator
    governor follow within their speed and force limits. Create it with
    :func:`create_hand_script`.
    """

    def __init__(self):
        self.keys: dict[str, np.ndarray] = {}
        """The resolved keyframes, see :func:`resolve_hand_keyframes`."""
        self.times: wp.array = None
        self.wrist: wp.array = None
        self.opposition: wp.array = None
        self.command: wp.array = None
        self.mode: wp.array = None
        self.step_count: wp.array = None
        """Number of substeps played, shape ``(1,)``; the script time is this count times the substep length."""
        self._wrist_target: wp.array = None
        self._wrist_steps_left: wp.array = None

    @property
    def duration(self) -> float:
        """Time of the last keyframe [s]."""
        return float(self.keys["times"][-1])

    def reset(self):
        """Rewind the script to time zero."""
        self.step_count.zero_()

    def evaluate(self, t: float) -> dict[str, Any]:
        """Evaluate the script at a time on the host, see :func:`evaluate_hand_keyframes`.

        Args:
            t: Time from the start of the script [s].

        Returns:
            The commands at that time: ``"wrist"``, ``"opposition"``, ``"command"`` and ``"mode"``.
        """
        return evaluate_hand_keyframes(self.keys, t)

    def step(self, control: HandControl, dt: float):
        """Write the commands of the end of the coming substep into ``control``; call before ``control.step``.

        Args:
            control: The hand control whose commands the script writes.
            dt: Substep length [s].
        """
        wrist = control.wrist
        wp.launch(
            hand_script_kernel,
            dim=1,
            inputs=[self.times, self.wrist, self.opposition, self.command, self.mode, dt],
            outputs=[
                self.step_count,
                self._wrist_target if wrist is None else wrist.target,
                self._wrist_steps_left if wrist is None else wrist.steps_left,
                control.opposition_target,
                control.governor.command,
                control.governor.mode,
            ],
            device=self.times.device,
        )


def create_hand_script(
    model: newton.Model,
    spec: HandSpec,
    hand: HandSkeleton,
    keyframes,
    wrist_speed_limit: float = HAND_SCRIPT_WRIST_SPEED,
    wrist_acceleration_limit: float = HAND_SCRIPT_WRIST_ACCELERATION,
    wrist_angular_speed_limit: float = HAND_SCRIPT_WRIST_ANGULAR_SPEED,
) -> HandScript:
    """Put a keyframe list on the model's device.

    Fields that no keyframe has set yet take the build state of the hand:
    the build pose of the palm, the build opposition angle and the rest
    tension of the spec.

    Args:
        model: The finalized model.
        spec: The hand spec.
        hand: The hand skeleton.
        keyframes: :class:`HandKeyframe` items with strictly increasing times; at least two.
        wrist_speed_limit: Largest allowed peak speed of the palm frame origin [m/s].
        wrist_acceleration_limit: Largest allowed peak acceleration of the palm frame origin [m/s^2].
        wrist_angular_speed_limit: Largest allowed peak turning rate of the palm [rad/s].

    Returns:
        The script.

    Raises:
        ValueError: If a segment exceeds a wrist limit.
    """
    start_opposition = float(model.joint_q.numpy()[hand.q_opposition]) if hand.joint_opposition >= 0 else 0.0
    keys = resolve_hand_keyframes(
        keyframes,
        hand.actuators,
        np.array(hand.palm_xform, dtype=float),
        start_opposition,
        spec.cable.tension_rest,
        spec=spec,
    )
    peaks = hand_keyframe_wrist_peaks(keys)
    limits = {
        "speed": wrist_speed_limit,
        "acceleration": wrist_acceleration_limit,
        "angular_speed": wrist_angular_speed_limit,
    }
    for name, limit in limits.items():
        worst = int(np.argmax(peaks[name]))
        if peaks[name][worst] > limit:
            raise ValueError(
                f"wrist {name} {peaks[name][worst]:.3g} exceeds the limit {limit:.3g} between the keyframes at "
                f"{keys['times'][worst]:.3g} s and {keys['times'][worst + 1]:.3g} s"
            )
    device = model.device
    script = HandScript()
    script.keys = keys
    script.times = wp.array(keys["times"], dtype=float, device=device)
    script.wrist = wp.array(keys["wrist"].astype(np.float32), dtype=wp.transform, device=device)
    script.opposition = wp.array(keys["opposition"], dtype=float, device=device)
    script.command = wp.array(keys["command"], dtype=float, device=device)
    script.mode = wp.array(keys["mode"], dtype=int, device=device)
    script.step_count = wp.zeros(1, dtype=int, device=device)
    script._wrist_target = wp.zeros(1, dtype=wp.transform, device=device)
    script._wrist_steps_left = wp.zeros(1, dtype=int, device=device)
    return script


# --- grasp plans -----------------------------------------------------------------------------------------

GRASP_PHASES = {
    "lead_in": 0.2,
    "approach": 0.8,
    "close": 1.0,
    "oppose": 0.7,
    "thumb": 0.6,
    "settle": 0.5,
    "lift": 1.0,
    "hold": 1.2,
    "tilt": 0.8,
    "tilt_hold": 0.4,
    "untilt": 0.8,
    "lower": 1.0,
    "rest": 0.3,
    "thumb_open": 0.4,
    "unoppose": 0.7,
    "open": 0.6,
    "wait": 0.6,
    "retreat": 0.8,
    "end": 0.3,
}
"""Default duration of every phase of a grasp plan [s], in the order they are played."""

GRASP_SEATING_PHASES = {"seat": 0.75, "unseat": 0.75, "seat_rest": 0.5}
"""Duration of the phases of the cable seating stroke that can precede a grasp plan [s]."""

GRASP_POSES = {
    # Side power grasp: the can stands upright, the palm faces it (palm normal along world +x), the fingers
    # point along world -y and the thumb side is up. The palm z of the can center and the commanded thumb
    # opposition are design numbers and come from the spec (``grasp_center_z`` and ``grasp_opposition`` of
    # the can): the can sits toward the thumb, and the little finger, which would catch the bottom rim,
    # stays open. The pedestal keeps the open little finger (splayed toward the table) above the table.
    "can": {
        "palm_axes": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        "digits": ("index", "middle", "ring", "thumb"),
        "tilt_axis": (0.0, 1.0, 0.0),
        "pedestal": (0.025, 0.040),
    },
    # From above: the ball lies on a pedestal that leaves room for the fingers below it, the palm faces
    # down, the fingers point along world +x and the thumb side is world +y.
    # On the two-cable layout the thumb column sits nearer the wrist, and the numbers follow from that:
    # - the pedestal is 60 mm high: on a 40 mm pedestal the straight thumb meets the table at 77 deg of
    #   opposition and never reaches the ball;
    # - the ball sits 18 mm toward the thumb, under the index, middle and ring finger (at 5 mm the index
    #   finger pushes it 9 mm aside while closing), and the straight thumb then meets it at about 124 deg;
    # - the thumb cable holds 12 N: at the hold tension the thumb curls under the ball and is caught
    #   between ball and pedestal at the set-down;
    # - the closing fingers draw the ball 2.3 mm up into the palm, so at the pick-up height it would be
    #   released 2.3 mm above its seat, and from there the opening fingers sweep it off. The hand
    #   therefore sets the ball down 1 mm lower than it picked it up: the ball is released 1.3 mm above
    #   the seat, without touching it, and drops into it (it stays for 0.5, 1.0 and 1.5 mm; at 2 mm the
    #   hand presses the ball on its seat with 17.5 N);
    # - the pedestal is 12 mm in radius (the widest that leaves the post below the ball).
    "ball": {
        "palm_axes": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        "center_z": 0.018,
        "opposition": 130.0 * _DEG,
        "digits": ("index", "middle", "ring", "little", "thumb"),
        "tilt_axis": (1.0, 0.0, 0.0),
        "pedestal": (0.012, 0.060),
        "thumb_tension": 12.0,
        "set_down_clearance": -0.001,
    },
    # Three-digit pinch from above: the block stands upright on the table, the palm faces down 70 mm above
    # its top. The thumb is turned to the pinch angle of the design before the hand comes down and stays
    # straight on its lower stops, 1 mm from one side face; the index and middle finger close on the
    # opposite face and press the block against the thumb. The thumb cable stays at the rest tension.
    # With two cables per finger the curl cable sets the shape of the outer joints (9 N: the fingertip pad
    # faces the block) and the knuckle cable swings the finger onto the block. The fingers reach the block
    # from a knuckle tension of 26 N; from 32 N the first joint rests on its upper stop and the grip force
    # stays at 0.5 to 0.6 N per finger (block weight 0.39 N) up to the tension limit of 60 N, so the block
    # is neither drawn into the hand nor thrown out.
    "block": {
        "palm_axes": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        "center": (0.0535, 0.100, 0.022),
        "opposition": 118.0 * _DEG,
        "digits": ("index", "middle"),
        "thumb_first": True,
        "curl_tension": 9.0,
        "hold_tension": 40.0,
        "tilt": 15.0 * _DEG,
        "tilt_axis": (1.0, 0.0, 0.0),
        "pedestal": None,
    },
}
"""Grasp pose per object: palm axes in the world frame at the grasp (``x``, ``y``, ``z`` of the palm
frame), palm ``z`` of the object center [m] (the center then rests on the palm pads) or the whole
``"center"`` in the palm frame [m], commanded thumb opposition [rad] (for can and ball beyond the angle
at which the straight thumb meets the object, so the opposition drive presses the thumb on at its torque
limit; ``"center_z"`` and ``"opposition"`` of the can are in the spec instead), the digits that close,
the world axis of the tilt, and pedestal radius and height [m]. Optional:
``"thumb_first"`` (the thumb turns into opposition before the approach and back after the retreat),
``"hold_tension"`` [N] and ``"tilt"`` [rad] where they differ from the defaults of :func:`grasp_plan`,
``"curl_tension"`` [N] (a two-cable finger then closes with its curl cable at this tension and its
knuckle cable at the hold tension), ``"thumb_tension"`` [N] (tension of the thumb cable in the grasp
where it differs from the hold tension) and ``"set_down_clearance"`` [m] (see :func:`grasp_plan`)."""


@dataclass
class GraspPlan:
    """A scripted grasp of one object, see :func:`grasp_plan`."""

    object_name: str
    object_xform: wp.transform
    """World pose of the object at the start."""
    pedestal: tuple[float, float] | None
    """Radius and height of the pedestal under the object [m], or ``None``."""
    palm_start: wp.transform
    """World pose of the palm frame at the start (build the hand there)."""
    palm_grasp: wp.transform
    """World pose of the palm frame at the grasp."""
    opposition: float
    """Thumb opposition angle of the grasp [rad]."""
    hold_tension: float
    """Cable tension of the hold [N]."""
    lift: float
    """Height of the lift [m]."""
    tilt: float
    """Tilt angle in the lifted hold [rad]."""
    keyframes: list[HandKeyframe]
    """The keyframes for :func:`create_hand_script`."""
    phases: dict[str, tuple[float, float]]
    """Start and end time of every phase [s], keyed as :data:`GRASP_PHASES`."""


def object_center_on_palm_pads(spec: HandSpec, radius: float) -> tuple[float, float]:
    """Center of a round object that rests on both palm pads.

    Args:
        spec: The hand spec.
        radius: Radius of the object [m].

    Returns:
        The center as palm ``(x, y)`` [m].
    """
    a, b = spec.palm.pads[0], spec.palm.pads[1]
    half = 0.5 * abs(a.p0[0] - b.p0[0])
    reach = radius + 0.5 * (a.radius + b.radius)
    return 0.5 * (a.p0[0] + b.p0[0]), 0.5 * (a.p0[1] + b.p0[1]) + math.sqrt(reach * reach - half * half)


def grasp_plan(
    spec: HandSpec,
    object_name: str,
    hold_tension: float | None = None,
    opposition: float | None = None,
    lift: float = 0.1,
    tilt: float | None = None,
    approach: float = 0.08,
    standoff: float = 0.001,
    center_z: float | None = None,
    phases: dict[str, float] | None = None,
    seating_tension: float | None = None,
    set_down_clearance: float | None = None,
) -> GraspPlan:
    """Keyframes of a complete grasp: approach, close, oppose the thumb, lift, hold, tilt, lower, open, retreat.

    At the grasp the object rests on both palm pads: the palm comes in along
    its normal and stops ``standoff`` short of touching. The fingers close
    first (cables to the hold tension), then the thumb turns into opposition
    until it meets the object, then the thumb cable is pulled to the hold
    tension. The tilt turns the palm
    about a world axis through the center of the lifted object, so the object
    stays in place while the direction of gravity changes in the hand: for
    the can the palm turns to face partly downward, for the ball the hand
    rolls about the finger direction. The hand then returns and lowers the
    object to where it was picked up. The release is the grasp in reverse:
    the thumb cable goes slack, the thumb turns back, and only then the
    fingers open. (With the fingers opened first, the opposition drive keeps
    pressing the thumb on and pushes the freed object aside.) Then the hand
    retreats.

    The block is pinched instead (see :data:`GRASP_POSES`): the thumb turns
    to the pinch angle first, the hand comes down with the straight thumb
    beside the block, the index and middle finger close and press the block
    against the thumb; after the set-down the fingers open, the hand
    retreats and the thumb turns back.

    Args:
        spec: The hand spec.
        object_name: ``"can"`` (side power grasp, on a pedestal), ``"ball"``
            (from above, on a pedestal) or ``"block"`` (three-digit pinch from above, on the table).
        hold_tension: Cable tension of the hold [N]; the value of
            :data:`GRASP_POSES` if it has one, else the grasp hold tension of the spec.
        opposition: Commanded thumb opposition of the grasp [rad]; the value of :data:`GRASP_POSES` if ``None``.
        lift: Height of the lift [m].
        tilt: Tilt angle in the lifted hold [rad]; the value of
            :data:`GRASP_POSES` if it has one, else 30 deg.
        approach: Distance of the start pose from the grasp pose along the palm normal [m].
        standoff: Gap between the palm pads and the object at the grasp pose [m]; not used for the block.
        center_z: Palm ``z`` of the object center at the grasp [m]; the value
            of :data:`GRASP_POSES` if ``None``.
        phases: Durations that replace those of :data:`GRASP_PHASES` (and of
            :data:`GRASP_SEATING_PHASES`) [s].
        set_down_clearance: Height above the pick-up height at which the hand
            stops when it lowers the object [m]; the value of
            :data:`GRASP_POSES` if it has one, else zero. With zero the palm
            returns to the grasp pose; a negative value sets an object that
            the closing fingers drew up into the palm back on its support.
            The wrist is position-driven: a fingertip that
            has come to lie under the object is then pressed between the
            object and its support with whatever force it takes (seen with
            the ball at 15 and 20 N hold tension, where a fingertip ends up
            below the ball). A clearance lets the object drop the last bit
            instead; 5 mm was not enough in that case.
        seating_tension: If given, the plan starts with a seating stroke for
            rod cables at the start pose: all cables to this tension [N] and
            back to the rest tension, then a rest hold (phases ``"seat"``,
            ``"unseat"``, ``"seat_rest"``). A rod can start carried just above
            a pulley and then acts as if it had friction until it has run
            once. The fingers close in the air during the stroke, so the
            start pose must be far enough from the object (``approach`` of
            0.14 m clears the object at 16 N).

    Returns:
        The plan.
    """
    if object_name not in GRASP_POSES:
        raise ValueError(f"no grasp plan for {object_name!r}; plans exist for {tuple(GRASP_POSES)}")
    pose = dict(GRASP_POSES[object_name])
    # design numbers of the grasp that the spec carries (the can) take the place of the pose entries
    design = spec.objects[object_name]
    if "grasp_center_z" in design:
        pose["center_z"] = design["grasp_center_z"]
    if "grasp_opposition" in design:
        pose["opposition"] = design["grasp_opposition"]
    if hold_tension is None:
        hold_tension = pose.get("hold_tension", spec.cable.tension_hold)
    if tilt is None:
        tilt = pose.get("tilt", 30.0 * _DEG)
    if opposition is None:
        opposition = pose["opposition"]
    if set_down_clearance is None:
        set_down_clearance = pose.get("set_down_clearance", 0.0)
    thumb_tension = pose.get("thumb_tension")
    if thumb_tension is None:
        thumb_tension = hold_tension
    durations = dict(GRASP_PHASES)
    durations.update(phases or {})
    pedestal = pose["pedestal"]
    height = grasp_object_rest_height(spec, object_name) + (0.0 if pedestal is None else pedestal[1])
    center_world = np.array([0.0, 0.0, height])
    rotation = np.array(pose["palm_axes"], dtype=float).T
    if "center" in pose:
        x, y, z = pose["center"]
        center_palm = np.array([x, y, z if center_z is None else center_z])
    else:
        x, y = object_center_on_palm_pads(spec, spec.objects[object_name]["radius"])
        z = pose["center_z"] if center_z is None else center_z
        center_palm = np.array([x, y + standoff, z])
    origin = center_world - rotation @ center_palm
    up = np.array([0.0, 0.0, lift])
    grasp = np.array(frame_to_transform(origin, rotation), dtype=float)
    start = np.array(frame_to_transform(origin - approach * rotation[:, 1], rotation), dtype=float)
    lifted = np.array(frame_to_transform(origin + up, rotation), dtype=float)
    set_down = grasp.copy()
    set_down[2] += set_down_clearance
    turn = np.array(wp.quat_from_axis_angle(wp.vec3(*pose["tilt_axis"]), float(tilt)), dtype=float)
    pivot = center_world + up
    tilted = np.concatenate([pivot + _quat_rotate_np(turn, origin + up - pivot), _quat_multiply_np(turn, grasp[3:])])
    rest = spec.cable.tension_rest
    # Close command per finger: the hold tension on every cable, or, where the pose gives a curl tension, a
    # two-cable finger keeps that on its curl cable (the shape) and takes the hold tension on its knuckle
    # cable (the press).
    fingers: dict[str, Any] = {}
    for name in pose["digits"]:
        if name == "thumb":
            continue
        fingers[name] = hold_tension
        if "curl_tension" in pose and spec.digits[name].cable_names == ["curl", "knuckle"]:
            fingers[name] = {"curl": pose["curl_tension"], "knuckle": hold_tension}
    changes = {
        "lead_in": {},
        "approach": {"wrist": grasp},
        "close": {"tension": fingers},
        "oppose": {"opposition": opposition},
        "thumb": {"tension": {"thumb": thumb_tension}} if "thumb" in pose["digits"] else {},
        "settle": {},
        "lift": {"wrist": lifted},
        "hold": {},
        "tilt": {"wrist": tilted},
        "tilt_hold": {},
        "untilt": {"wrist": lifted},
        "lower": {"wrist": set_down},
        "rest": {},
        "thumb_open": {"tension": {"thumb": rest}},
        "unoppose": {"opposition": 0.0},
        "open": {"tension": rest},
        "wait": {},
        "retreat": {"wrist": start},
        "end": {},
    }
    if pose.get("thumb_first"):
        order = list(changes)
        for name, before in (("oppose", "approach"), ("unoppose", "end")):
            order.remove(name)
            order.insert(order.index(before), name)
        changes = {name: changes[name] for name in order}
    if seating_tension is not None:
        seating = {"seat": {"tension": seating_tension}, "unseat": {"tension": rest}, "seat_rest": {}}
        changes = {**seating, **changes}
        durations = {**GRASP_SEATING_PHASES, **durations}
    keyframes = [HandKeyframe(0.0, wrist=start, opposition=0.0, tension=rest)]
    windows = {}
    t = 0.0
    for name, change in changes.items():
        windows[name] = (t, t + durations[name])
        t += durations[name]
        keyframes.append(HandKeyframe(t, **change))
    return GraspPlan(
        object_name=object_name,
        object_xform=wp.transform(wp.vec3(*center_world.tolist()), wp.quat_identity()),
        pedestal=pedestal,
        palm_start=wp.transform(*start.tolist()),
        palm_grasp=wp.transform(*grasp.tolist()),
        opposition=float(opposition),
        hold_tension=float(hold_tension),
        lift=float(lift),
        tilt=float(tilt),
        keyframes=keyframes,
        phases=windows,
    )


# --- grasp read-back -------------------------------------------------------------------------------------


def _quat_multiply_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Product of two quaternions ``x, y, z, w``."""
    av, bv = a[:3], b[:3]
    return np.concatenate([a[3] * bv + b[3] * av + np.cross(av, bv), [a[3] * b[3] - av @ bv]])


def object_surface_distance(kind: str, size, points) -> np.ndarray:
    """Signed distance of points to the surface of an object, in the object frame [m]; negative inside.

    Args:
        kind: ``"cylinder"`` (axis along ``z``), ``"sphere"`` or ``"box"``.
        size: Cylinder: radius and half height; sphere: radius; box: half extents [m].
        points: Points in the object frame [m], shape ``(..., 3)``.

    Returns:
        The distances, shape ``(...)``.
    """
    points = np.asarray(points, dtype=float)
    if kind == "sphere":
        return np.linalg.norm(points, axis=-1) - size[0]
    if kind == "cylinder":
        d = np.stack([np.linalg.norm(points[..., :2], axis=-1) - size[0], np.abs(points[..., 2]) - size[1]], axis=-1)
    else:
        d = np.abs(points) - np.asarray(size, dtype=float)
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    return outside + np.minimum(d.max(axis=-1), 0.0)


def grasp_state(model: newton.Model, state: newton.State, hand: HandSkeleton, scene: GraspScene) -> dict[str, Any]:
    """Read the object and the pads of a grasp scene from a state.

    The pad distance is geometric: the capsule axis of each pad is sampled at
    17 points, and the smallest signed distance to the object surface minus
    the pad radius is reported. A negative value is the penetration depth.

    Args:
        model: The model.
        state: The state to read.
        hand: The hand skeleton.
        scene: The grasp scene.

    Returns:
        ``"object_pose"``: world pose of the object (position [m], quaternion), shape ``(7,)``;
        ``"object_velocity"`` [m/s] and ``"object_angular_velocity"`` [rad/s]:
        velocity of its center of mass and its turning rate, world frame, shape ``(3,)``;
        ``"object_in_palm"``: pose of the object in the palm frame, shape ``(7,)``
        (the reference for slip, see :func:`grasp_slip`);
        ``"pad_distance"`` [m]: distance between each pad and the object,
        shape ``(pad_count,)`` in the order of ``scene.pads.keys``.
    """
    body_q = state.body_q.numpy().astype(float)
    body_qd = state.body_qd.numpy().astype(float)
    pose = body_q[scene.object_body]
    palm = body_q[hand.palm]
    palm_inverse = np.concatenate([-palm[3:6], palm[6:]])
    in_palm = np.concatenate(
        [_quat_rotate_inv_np(palm[3:], pose[:3] - palm[:3]), _quat_multiply_np(palm_inverse, pose[3:])]
    )
    pads = scene.pads
    samples = np.linspace(0.0, 1.0, 17)
    distance = np.empty(len(pads.keys))
    for i, body in enumerate(pads.body):
        local = pads.p0[i] + samples[:, None] * (pads.p1[i] - pads.p0[i])
        world = body_q[body, :3] + np.array([_quat_rotate_np(body_q[body, 3:], p) for p in local])
        in_object = np.array([_quat_rotate_inv_np(pose[3:], p - pose[:3]) for p in world])
        distance[i] = object_surface_distance(scene.object_kind, scene.object_size, in_object).min() - pads.radius[i]
    return {
        "object_pose": pose,
        "object_velocity": body_qd[scene.object_body, :3],
        "object_angular_velocity": body_qd[scene.object_body, 3:],
        "object_in_palm": in_palm,
        "pad_distance": distance,
    }


def grasp_slip(object_in_palm, reference) -> tuple[float, float]:
    """Motion of the object relative to the palm since a reference instant.

    Args:
        object_in_palm: Pose of the object in the palm frame, from :func:`grasp_state`.
        reference: The same pose at the reference instant.

    Returns:
        Distance moved by the object's center in the palm frame [m] and the
        angle turned by the object in the palm frame [rad].
    """
    a = np.asarray(object_in_palm, dtype=float)
    b = np.asarray(reference, dtype=float)
    cos_half = min(abs(float(a[3:] @ b[3:])), 1.0)
    return float(np.linalg.norm(a[:3] - b[:3])), 2.0 * math.acos(cos_half)


# ---------------------------------------------------------------------------
# Choreography (named hand poses as cable tensions, and a keyframe sequence through them)
# ---------------------------------------------------------------------------
# A pose is data: a cable tension per digit, a thumb opposition angle and optionally a wrist pose. The joint
# angles follow from the closed form. The hand has no link-link collision, so the poses and their order are
# chosen such that no digit meets another digit or the palm: a folded thumb stands above the palm, clear of
# the curled fingers, and it is opposed before it is flexed and straightened before it is laid back.

CHOREOGRAPHY_FOLDED_THUMB = (math.radians(80.0), 20.0)
"""Opposition angle [rad] and cable tension [N] of the folded thumb of the count poses and the fist."""

CHOREOGRAPHY_SEATING = (math.radians(60.0), 16.0)
"""Opposition angle [rad] and cable tension [N] of the seating stroke; the thumb is opposed so that it
clears the index finger while every digit closes to this tension."""

CHOREOGRAPHY_PINCH_GAP = 2.5e-3
"""Gap left between the two pads of a pinch pose [m].

The hand has no pad contact in this sequence, and near a pinch the thumb pad
moves about 19 mm per newton of thumb cable tension, so the gap also has to
cover the pose error of the cable model (about 1 mm with the rod cables).
"""

CHOREOGRAPHY_PINCH = {
    "index": (math.radians(118.0), "pinch_ready"),
    "middle": (math.radians(128.0), "pinch_ready"),
}
"""Pinch poses with a two-cable finger: thumb opposition angle [rad] and the finger's two-cable pose
(first joint held on its upper stop by the knuckle cable, second joint part bent by the curl cable)."""


@dataclass
class HandPose:
    """A named hand pose: cable tensions, thumb opposition and the joint angles that follow from them."""

    name: str
    tension: dict[str, float]
    """Cable tension per actuator key [N] (every cable of the hand, see :attr:`HandSkeleton.actuators`)."""
    opposition: float
    """Thumb opposition angle [rad]."""
    angles: dict[str, tuple[float, float, float]]
    """Expected joint angles MCP, PIP, DIP per digit name [rad], from the closed form."""
    wrist: tuple[float, ...] | None = None
    """World pose of the palm frame (position [m] and quaternion ``x, y, z, w``), or ``None`` for the home pose."""


def hand_pose_angles(spec: HandSpec, tension: dict[str, Any], opposition: float = 0.0, gravity=None) -> dict:
    """Closed-form joint angles of every digit at given cable tensions.

    Args:
        spec: The hand spec.
        tension: Commands per digit name or actuator key (see
            :func:`hand_cable_tensions`); cables without one are at the rest tension.
        opposition: Thumb opposition angle [rad].
        gravity: Gravity vector in the palm frame [m/s^2]; ``None`` for no gravity. Only its part in the
            flexion plane of a digit loads that digit's joints (:func:`gravity_angles`).

    Returns:
        Joint angles MCP, PIP, DIP per digit name [rad].
    """
    angles = {}
    given = hand_cable_tensions(spec, tension)
    for name, digit in spec.digits.items():
        value = [
            given.get(actuator_key(name, c, cable.name), spec.cable.tension_rest)
            for c, cable in enumerate(digit.cables)
        ]
        if gravity is None:
            q = frictionless_angles(digit, value)
        else:
            _origin, rot = digit_frame_in_palm(spec, name, opposition)
            g = rot.T @ np.asarray(gravity, dtype=float)
            q = gravity_angles(digit, value, (g[0] / GRAVITY, g[1] / GRAVITY), GRAVITY)
        angles[name] = tuple(float(v) for v in q)
    return angles


def _segment_segment_distance(p1, q1, p2, q2) -> float:
    """Smallest distance between the segments ``p1 q1`` and ``p2 q2`` [m]."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    c, b = float(d1 @ r), float(d1 @ d2)
    denom = a * e - b * b
    s = min(max((b * f - c * e) / denom, 0.0), 1.0) if denom > 1.0e-18 else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t, s = 0.0, min(max(-c / a, 0.0), 1.0)
    elif t > 1.0:
        t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return float(np.linalg.norm(p1 + s * d1 - p2 - t * d2))


def pinch_pad_gap(spec: HandSpec, finger: str, opposition: float, q_thumb, q_finger) -> float:
    """Gap between the distal pad of the thumb and the distal pad of a finger [m]; negative if they overlap.

    Args:
        spec: The hand spec.
        finger: Name of the finger.
        opposition: Thumb opposition angle [rad].
        q_thumb: Thumb joint angles [rad].
        q_finger: Finger joint angles [rad].

    Returns:
        Distance between the two pad capsule axes minus both pad radii [m].
    """
    axes = []
    for name, q in (("thumb", q_thumb), (finger, q_finger)):
        digit = spec.digits[name]
        origin, rot = digit_frame_in_palm(spec, name, opposition)
        origins, angles = link_frames(digit, q)
        pad = digit.pads[-1]
        points = np.array([frame_point(origins, angles, pad.body, point) for point in (pad.p0, pad.p1)])
        lifted = lift_points(points, origin, rot)
        axes.append((lifted[0], lifted[1], pad.radius))
    (a0, a1, ra), (b0, b1, rb) = axes
    return _segment_segment_distance(a0, a1, b0, b1) - ra - rb


def pinch_thumb_tension(
    spec: HandSpec,
    finger: str,
    opposition: float,
    finger_tension: float,
    gap: float = CHOREOGRAPHY_PINCH_GAP,
    gravity=None,
) -> float:
    """Thumb cable tension at which the thumb pad stands ``gap`` in front of a finger pad.

    The thumb closes toward the finger as its tension rises from the value at
    which its first joint starts to move; the tension is found by bisection on
    the closed-form pose (with gravity if given).

    Args:
        spec: The hand spec.
        finger: Name of the finger.
        opposition: Thumb opposition angle [rad].
        finger_tension: Command of the finger (see :func:`hand_cable_tensions`):
            a cable tension [N], or for a two-cable finger a tension per cable or a pose name.
        gap: Pad gap to leave [m].
        gravity: Gravity vector in the palm frame [m/s^2], or ``None``.

    Returns:
        The thumb cable tension [N].

    Raises:
        ValueError: If the pads do not come that close anywhere on the thumb's closing path.
    """

    def pad_gap(thumb_tension):
        q = hand_pose_angles(spec, {finger: finger_tension, "thumb": thumb_tension}, opposition, gravity)
        return pinch_pad_gap(spec, finger, opposition, q["thumb"], q[finger])

    if pad_gap(spec.cable.tension_rest) <= gap:
        raise ValueError(f"the thumb pad is already within {gap} m of the {finger} pad with a straight thumb")

    # walk up from the rest tension to the first tension at which the pads are closer than the gap
    low = spec.cable.tension_rest
    high = None
    for value in np.arange(low, 16.0, 0.05):
        if pad_gap(float(value)) <= gap:
            high = float(value)
            break
        low = float(value)
    if high is None:
        raise ValueError(f"the thumb pad does not come within {gap} m of the {finger} pad")
    for _ in range(40):
        mid = 0.5 * (low + high)
        if pad_gap(mid) <= gap:
            high = mid
        else:
            low = mid
    return low


def hand_poses(spec: HandSpec, gravity=None) -> dict[str, HandPose]:
    """The pose library of the choreography.

    ``rest``: every cable at the rest tension, thumb laid back (the build
    pose). ``one`` .. ``four``: index, middle, ring and little finger straight up to
    that count, the others closed at 32 N on every cable (every joint on its
    upper stop), the thumb folded (:data:`CHOREOGRAPHY_FOLDED_THUMB`).
    ``five``: all straight, thumb laid back. ``hook_knuckle``: what only a
    two-cable finger can do: the index finger curls its outer joints fully on
    a straight first joint (a hook) while the middle finger bends its first
    joint 60 deg with straight outer joints. ``pinch_index``, ``pinch_middle``:
    the two pinch poses of the design (:data:`CHOREOGRAPHY_PINCH`), the thumb
    tension solved so that the pads stand :data:`CHOREOGRAPHY_PINCH_GAP`
    apart. ``fist``: four fingers closed, thumb folded.

    Args:
        spec: The hand spec; its index and middle finger carry two cables.
        gravity: Gravity vector in the palm frame at the home wrist pose
            [m/s^2]; ``None`` for no gravity. It enters the expected angles
            and the thumb tension of the pinch poses.

    Returns:
        The poses by name, in the order they appear in the choreography. The
        tensions of a pose name every cable of the hand by its actuator key.
    """
    rest = spec.cable.tension_rest
    names = list(spec.digits)
    fingers = [name for name in names if not isinstance(spec.digits[name].mount, ThumbMountSpec)]
    two_cable = [name for name in fingers if len(spec.digits[name].cables) == 2]
    closed = 32.0
    fold_angle, fold_tension = CHOREOGRAPHY_FOLDED_THUMB
    table: list[tuple[str, dict[str, Any], float]] = [("rest", {}, 0.0)]
    for count, label in enumerate(("one", "two", "three", "four"), start=1):
        tension = dict.fromkeys(fingers[count:], closed)
        tension["thumb"] = fold_tension
        table.append((label, tension, fold_angle))
    table.append(("five", {}, 0.0))
    commands = ("hook", "knuckle_only")
    table.append(("hook_knuckle", {name: commands[k % 2] for k, name in enumerate(two_cable)}, 0.0))
    for finger in ("index", "middle"):
        angle, command = CHOREOGRAPHY_PINCH[finger]
        thumb = pinch_thumb_tension(spec, finger, angle, command, CHOREOGRAPHY_PINCH_GAP, gravity)
        table.append((f"pinch_{finger}", {finger: command, "thumb": thumb}, angle))
    table.append(("fist", {**dict.fromkeys(fingers, closed), "thumb": fold_tension}, fold_angle))
    keys = [actuator_key(name, c, cable.name) for name in names for c, cable in enumerate(spec.digits[name].cables)]
    poses = {}
    for label, tension, angle in table:
        given = hand_cable_tensions(spec, tension)
        full = {key: float(given.get(key, rest)) for key in keys}
        poses[label] = HandPose(label, full, float(angle), hand_pose_angles(spec, full, angle, gravity))
    return poses


def hand_seating_keyframes(spec: HandSpec, start: float = 0.0) -> list[HandKeyframe]:
    """Keyframes of the seating stroke: every cable from the rest tension to 16 N and back.

    A rod cable can start carried by a pulley; one stroke lets it slide into
    place before anything is measured or shown. The thumb is opposed first
    (:data:`CHOREOGRAPHY_SEATING`) so that it clears the index finger while
    all digits close, and laid back afterward. The stroke ends with 0.5 s at
    rest.

    Args:
        spec: The hand spec.
        start: Time of the first keyframe [s].

    Returns:
        The keyframes; the last one is the start of whatever follows.
    """
    rest = spec.cable.tension_rest
    angle, tension = CHOREOGRAPHY_SEATING
    steps = [(0.0, 0.0, rest), (0.7, angle, rest), (0.5, angle, tension), (0.2, angle, tension), (0.5, angle, rest)]
    steps += [(0.7, 0.0, rest), (0.5, 0.0, rest)]
    keys = []
    t = start
    for duration, opposition, value in steps:
        t += duration
        keys.append(HandKeyframe(time=t, opposition=opposition, tension=value))
    return keys


def hand_choreography_keyframes(
    spec: HandSpec,
    home_wrist=None,
    gravity=None,
    start: float = 0.0,
) -> tuple[list[HandKeyframe], list[dict[str, Any]], dict[str, HandPose]]:
    """Keyframes of the hand choreography: wave, count one to five, pinch, fist, open.

    16.6 s, and loopable: the sequence
    starts and ends in the ``rest`` pose at the home wrist pose. The timing
    follows what the hand can do: a finger needs 1.2 s between straight and
    closed (33 mm of cable at the governor's reference rate; 1.4 s on the curl
    cable of a two-cable finger, which is ramped at 24 mm/s), the folded thumb
    0.9 s, the opposition drive 3 rad/s. A cable command changes smoothly
    over 0.6 s and the pose is expected at the end of the hold that follows:
    the digit that moves last is at rest for the last 0.1 s of every hold.

    - Wave (1.05 s): the palm turns about its normal by 20 deg to both
      sides about the wrist end of the palm, while the fingers ripple (one
      after the other to 7.5 N and back). The wave is a wrist motion: it has
      no pose and no entry in the schedule.
    - Count (7.6 s): ``one`` .. ``five``. The thumb is opposed before it is
      flexed on the way into ``one`` and straightened before it is laid back
      on the way into ``five``.
    - Two-cable pose (1.25 s): ``hook_knuckle``: the index finger curls its
      outer joints on a straight first joint while the middle finger bends
      its first joint with straight outer joints.
    - Pinch (2.7 s): ``pinch_index``, then ``pinch_middle``; the thumb backs
      off to the rest tension between the two.
    - Fist (2.25 s): the thumb is straightened and brought to the folded
      thumb's opposition angle, the fingers close, then the thumb is flexed.
    - Open (1.75 s): the thumb straightens, everything returns to ``rest``.

    Args:
        spec: The hand spec.
        home_wrist: World pose of the palm frame at rest (position [m] and
            quaternion ``x, y, z, w``); identity if ``None``. With ``None``
            the keyframes carry no wrist pose and also suit a fixed wrist,
            without the wave rotation.
        gravity: Gravity vector in the palm frame at the home pose [m/s^2],
            see :func:`hand_poses`.
        start: Time of the first keyframe [s] (e.g. the end of
            :func:`hand_seating_keyframes`).

    Returns:
        The keyframes; the schedule, one entry per named hold with ``"pose"``,
        ``"start"`` and ``"end"`` [s] (the pose is expected at ``"end"``);
        and the pose library.
    """
    poses = hand_poses(spec, gravity)
    wave_angle = math.radians(20.0)
    rest = spec.cable.tension_rest
    names = list(spec.digits)
    fingers = [name for name in names if name != "thumb"]
    keys: list[HandKeyframe] = []
    schedule: list[dict[str, Any]] = []
    t = float(start)

    def key(duration, tension=None, opposition=None, wrist=None):
        nonlocal t
        t += duration
        keys.append(HandKeyframe(time=t, wrist=wrist, opposition=opposition, tension=tension))

    def hold(name, duration):
        nonlocal t
        schedule.append({"pose": name, "start": t, "end": t + duration})
        key(duration)

    def wrist_turned(angle):
        if home_wrist is None:
            return None
        home = wp.transform(*[float(v) for v in home_wrist])
        turn = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(angle)))
        return tuple(float(v) for v in wp.transform_multiply(home, turn))

    key(0.0, tension=dict(poses["rest"].tension), opposition=0.0, wrist=wrist_turned(0.0))
    # A cable command changes over `ramp` seconds (smooth); the governor then pays the cable in or out at its
    # reference rate, so a pose that needs the full travel is reached about 1.3 s after the command starts.
    # The hold of the schedule begins when the command has arrived and ends when the pose is expected.
    ramp = 0.6
    # The holds are as long as the digit that moves last needs to be at rest for the last 0.1 s of its hold
    # (the curl cables of the two-cable fingers move at 24 mm/s). The loop is 16.6 s.
    # wave: three swings, a finger ripple on top
    ripple = 7.5
    swings = [(0.3, 1.0), (0.45, -1.0), (0.3, 0.0)]
    for k, (duration, side) in enumerate(swings):
        tension = dict.fromkeys(fingers, rest)
        tension[fingers[k % len(fingers)]] = ripple
        key(duration, tension=tension, wrist=wrist_turned(side * wave_angle))
    # count: into "one" with the thumb opposed first, then flexed
    fold_angle = CHOREOGRAPHY_FOLDED_THUMB[0]
    one = poses["one"]
    # The thumb cable is commanded together with the opposition: the thumb only starts to flex above 6 N,
    # and by then the opposition drive has lifted it past the index finger.
    key(ramp, tension=dict(one.tension), opposition=fold_angle)
    hold("one", 1.0)
    for label in ("two", "three", "four"):
        key(ramp, tension=dict(poses[label].tension))
        # The opening finger comes to rest on its lower stops before the hold ends; the middle finger needs
        # the longest hold (its curl cable pays out at 24 mm/s, the one-cable fingers' cables at 28 mm/s).
        hold(label, {"two": 1.05, "three": 0.95, "four": 0.95}[label])
    key(0.4, tension={"thumb": rest})  # straighten the folded thumb where it stands
    key(0.7, opposition=0.0)
    hold("five", 0.15)
    # the pose only two-cable fingers can take: one curls its outer joints on a straight knuckle, the
    # other bends its knuckle with straight outer joints (19 mm of curl cable: 0.7 s after the command)
    key(0.5, tension=dict(poses["hook_knuckle"].tension))
    hold("hook_knuckle", 0.75)
    # pinch: thumb across first, closing onto the finger pad at the end
    pinches = (("pinch_index", 1.05, 0.0, 0.3), ("pinch_middle", 0.65, 0.25, 0.45))
    for label, move, back_off, wait in pinches:
        pose = poses[label]
        if back_off > 0.0:
            key(back_off, tension={"thumb": rest})  # back off before the other finger comes in
        before = next(k.opposition for k in reversed(keys) if k.opposition is not None)
        across = before + 0.6 * (pose.opposition - before)
        key(0.6 * move, tension={**pose.tension, "thumb": rest}, opposition=across)
        key(0.4 * move, tension={"thumb": pose.tension["thumb"]}, opposition=pose.opposition)
        hold(label, wait)
    # fist: thumb straight and to the folded angle while the fingers close, then flexed
    fist = poses["fist"]
    # The thumb leaves first: at the pinch angle of the middle finger the closing middle finger would
    # otherwise sweep through the place where the thumb tip still stands.
    key(0.25, tension={"thumb": rest}, opposition=fold_angle)
    key(0.55, tension={**fist.tension, "thumb": rest})
    key(0.3, tension={"thumb": fist.tension["thumb"]})
    hold("fist", 1.15)
    # open: thumb straight first, then laid back while the fingers open
    key(0.4, tension={**{n: 0.5 * (rest + fist.tension[n]) for n in fingers}, "thumb": rest})
    key(0.7, tension=dict(poses["rest"].tension), opposition=0.0)
    hold("rest", 0.65)
    return keys, schedule, poses


def hand_choreography(
    model: newton.Model,
    spec: HandSpec,
    hand: HandSkeleton,
    gravity=None,
    seating: bool = True,
) -> tuple[HandScript, list[dict[str, Any]], dict[str, HandPose]]:
    """Create the choreography script for a hand model.

    With a kinematic wrist the wave turns the palm about the pose the hand was
    built at; with a fixed wrist the fingers ripple only.

    Args:
        model: The finalized model of the hand (rod or tendon cables).
        spec: The hand spec.
        hand: The hand skeleton.
        gravity: Gravity vector in the **world** frame [m/s^2] that the hand
            control applies as body forces, or ``None``. It is turned into the
            palm frame of the build pose for the pose library.
        seating: If ``True``, the script starts with the seating stroke of
            :func:`hand_seating_keyframes` and the choreography follows it.

    Returns:
        The script (see :class:`HandScript`), the schedule of the named holds
        (script times [s]; its first entry is the rest hold that ends the
        seating stroke, if any) and the pose library.
    """
    home = np.array(hand.palm_xform, dtype=float)
    gravity_palm = None
    if gravity is not None:
        gravity_palm = _quat_rotate_inv_np(home[3:], np.asarray(gravity, dtype=float))
    keys: list[HandKeyframe] = []
    schedule: list[dict[str, Any]] = []
    start = 0.0
    if seating:
        keys = hand_seating_keyframes(spec)
        start = keys[-1].time
        schedule.append({"pose": "rest", "start": start - 0.5, "end": start, "seating": True})
        keys = keys[:-1]  # the choreography's own first keyframe stands at this time
    more, holds, poses = hand_choreography_keyframes(
        spec,
        home_wrist=home if hand.wrist == "kinematic" else None,
        gravity=gravity_palm,
        start=start,
    )
    script = create_hand_script(model, spec, hand, [*keys, *more])
    return script, [*schedule, *holds], poses
