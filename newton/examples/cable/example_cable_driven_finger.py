# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable-driven finger: two cables on two linear actuators close three joints against return springs.

A single finger on a fixed base. Every joint has a return spring with preload
that opens the finger. The index and the middle finger are closed by two
cables, each pulled by its own linear actuator:

- the **knuckle cable** runs over a sheave on the first joint and ends on the
  proximal link: it bends the first joint only;
- the **curl cable** runs over a pulley at every joint to the fingertip. On
  the first joint it passes a small idler, so it bends the two outer joints
  long before it moves the first one.

With two tensions the finger takes poses one cable cannot: a bent first joint
with straight outer joints, a hook (outer joints curled on a straight first
joint), and anything between. The default loop goes through such poses. The
ring finger, the little finger and the thumb (`--digit`) have one cable, the
flexor, which closes the first, then the second, then the third joint as its
tension rises; their loop is a tension staircase.

The cables are drawn in the color of their role: the curl cable (and the
single flexor) cyan-blue, the knuckle cable magenta-pink. The brightness shows
the tension on a fixed scale: dim at the rest tension of 2 N, the full color
at the 60 N limit.
Expand "Example Options" to compare the commanded and measured tensions, the
joint angles and the travel of every actuator with the hand calculation
(moment arms times tensions against the springs at every joint; cable
take-up plus elastic stretch for the travel).

Two cable models share the finger, the actuator governor (force limit, speed
limit, rest tension) and the loop:

- `--cable-model tendon` (default): massless routed tendons. They run on the
  CPU and on CUDA and follow the hand calculation closely. Cable friction on
  the guides and sheaves (`--cable-friction`) is available for the one-cable
  digits.
- `--cable-model rod`: every cable is a chain of 1 mm rod segments that is
  kept on flanged pulleys by contact only. It shows cable mass, bending and
  contact, costs more per frame, is meant for CUDA, and has no validated
  cable friction (`--cable-friction` is rejected). The loop starts with a
  seating stroke (every cable to 16 N and back). `--rigid-articulation-solve`
  selects how the solver treats the closed chain of rod segments and finger
  joints: `block_sparse_levels` (default) or the slower sequential
  `block_sparse_joints`.

`--actuation force` commands the cable tensions; `--actuation position`
commands the actuator travels that the hand calculation gives for the same
tensions (without cable friction only). `--tension-pairs` replaces the loop of
a two-cable finger by your own list of `curl,knuckle` tensions;
`--tension-levels` replaces the staircase of a one-cable digit. The commands
move from hold to hold smoothly, over 0.6 s, not in steps: a stepped knuckle
command lets the curl cable of a two-cable finger dip toward slack. `--xray`
draws the finger shells translucent so the cables and the pulleys show.

Success criteria:
- Two-cable finger: the first joint follows the knuckle cable and the outer joints follow the curl
  cable; in the hook the first joint stays straight while the outer joints curl. One-cable digit:
  the joints close in the order first, second, third and open in the reverse order. No jitter, and
  the cables stay on their pulleys.
- At the end of every hold the joint angles and the travel of every actuator agree with the hand
  calculation shown in the panel: within 0.3 deg and 0.1 mm for the tendons without friction; for the
  rods, after their seating stroke, within 1.3 to 2.0 deg (by digit) and 0.4 mm, with holds of 1 s.
- The two cables of a finger differ in color, and a cable brightens with its tension. At the end of
  every frame the measured tensions stay between zero and the limit, with a margin of 3 N for the
  tendons and of 5 N anywhere along a rod, and after the start (after the seating stroke for rods) no
  cable falls below 0.5 N.

Runs:
- `python -m newton.examples cable_driven_finger`
  - The index finger with two massless tendons goes through its poses: knuckle bent with straight
    outer joints, hook, pinch pose, fist. Measured angles and travels sit on the hand calculation
    at every hold.
- `python -m newton.examples cable_driven_finger --xray`
  - The same with translucent shells: the two cable paths, side by side on the first joint, show.
- `python -m newton.examples cable_driven_finger --tension-pairs 2,2 2,20 2,28 18,28 18,2`
  - Your own loop: first the knuckle cable alone, then the curl cable on top, then the hook.
- `python -m newton.examples cable_driven_finger --actuation position`
  - The actuator travels are commanded instead of the tensions; the finger reaches the same poses.
- `python -m newton.examples cable_driven_finger --digit ring`
  - A one-cable finger: one flexor closes the three joints in a fixed order.
- `python -m newton.examples cable_driven_finger --digit ring --cable-friction 0.05`
  - With friction on the pulleys the tension at the fingertip is lower than at the actuator while
    closing and higher while opening; the finger lags the frictionless calculation in both directions.
- `python -m newton.examples cable_driven_finger --digit thumb`
  - The thumb, which has a different cable route to its actuator, closes in the same order.
- `python -m newton.examples cable_driven_finger --cable-model rod --device cuda:0 --xray`
  - Two rods in neighboring grooves on the first joint, visible inside the translucent shells.
"""

import itertools
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cable import cable_hand

# Agreement with the hand calculation at the end of a hold, per cable model: joint angle [deg], actuator
# travel [mm] and tension at both ends of every cable relative to the reference [-]. Measured over the
# full reference tables: the frictionless tendons stay within 0.07 deg and 0.02 mm on every digit (the
# joint stops give by 0.06 to 0.07 deg at the tension limit). With cable friction (one-cable digits)
# every pose inside the friction band is an equilibrium, so a finger that arrives with a little overshoot
# rests off the capstan table: the ring finger is within 0.72 deg, 0.12 mm and 1.5 % of the tip tension at
# a friction of 0.02 and of 0.05 (blended commands, 0.5 s holds); the tolerance leaves room for the other
# digits and for user-given levels.
HOLD_TOLERANCE = {
    "tendon": {"angle": 0.3, "travel": 0.1, "tension": 0.02},
    "tendon_friction": {"angle": 2.5, "travel": 0.4, "tension": 0.1},
    "rod": {"travel": 0.4, "tension": 0.03},  # the angle tolerance is per digit, see below
}
# Joint angle tolerance of the rod cables per digit [deg], measured with 1 mm segments (after the seating
# stroke, holds of at least 1 s, 30 substeps of 32 iterations). The thumb's rod has 2 mm segments and a
# tolerance of 2 deg.
ROD_ANGLE_TOLERANCE = {"index": 1.3, "middle": 1.8, "ring": 1.6, "little": 1.3, "thumb": 2.0}
# Shortest hold of the rod cables [s]: a rod finger needs about a second to settle on a level.
ROD_HOLD = 1.0
# Largest cable friction coefficient: the tendon was compared with the hand calculation at 0.02 and 0.05.
FRICTION_MAX = 0.05
# Commanding the travel leaves the tension to the mechanism: wherever the finger rests on its stops the
# cable is as stiff as its own stretch (about 8 N per 0.1 mm), so the tension is checked more loosely.
POSITION_TENSION_TOLERANCE = 0.25
# Margin on the tension limit for the tension seen at frame ends [N], per cable model.
TENSION_MARGIN = {"tendon": 3.0, "rod": 5.0}
# Lowest tension any cable may show at a frame end after the start (after the seating stroke for rods) [N]:
# no cable may come close to slack. The commands are blended from level to level for this reason.
TENSION_FLOOR = 0.5
# After the seating stroke of the rod cables (rest, 16 N, rest) the tension at both ends of every cable must
# be within this fraction of the rest tension; a larger offset means a rod did not seat on its pulleys.
ROD_REST_TOLERANCE = 0.03


def two_cable_loop(spec, digit: str) -> list[dict[str, float]]:
    """Default loop of a two-cable finger: tensions by cable name, one entry per hold."""
    rest = spec.cable.tension_rest
    poses = cable_hand.two_cable_poses(spec, digit)
    straight = {"curl": rest, "knuckle": rest}
    return [
        straight,
        poses["knuckle_only"],  # first joint at 60 deg, outer joints straight
        {"curl": rest, "knuckle": 32.0},  # first joint on its upper stop
        straight,
        {"curl": 9.0, "knuckle": rest},  # outer joints part curled, first joint straight
        poses["hook"],
        poses["pinch_ready"],
        poses["full_fist"],
    ]


class Example:
    FPS = 60

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.cable_model = args.cable_model
        self.digit = args.digit
        self.friction = float(args.cable_friction)
        self.position_mode = args.actuation == "position"
        rod = self.cable_model == "rod"
        if not math.isfinite(self.friction) or not 0.0 <= self.friction <= FRICTION_MAX:
            raise ValueError(f"cable-friction must be in [0, {FRICTION_MAX:g}]")
        if rod and self.friction > 0.0:
            raise ValueError("cable friction is not validated for the rod cable; use --cable-model tendon")
        if rod and self.position_mode:
            raise ValueError(
                "--actuation position is not validated with the rod cable (a rod's rest travel is known only "
                "after its seating stroke); use --actuation force or --cable-model tendon"
            )
        if self.position_mode and self.friction > 0.0:
            raise ValueError(
                "cable friction is not validated with --actuation position: with the travel commanded, friction "
                "leaves the cable tension anywhere inside its band; use --actuation force"
            )
        if not rod and args.rigid_articulation_solve is not None:
            raise ValueError("rigid-articulation-solve applies to the rod cable; the tendon needs the local solve")

        spec = cable_hand.default_hand_spec()
        digit = spec.digits[self.digit]
        self.cable_names = digit.cable_names
        self.cable_roles = [cable.role for cable in digit.cables]
        self.rest_tension = spec.cable.tension_rest
        self.tension_limit = spec.cable.tension_max
        cables = len(self.cable_names)
        if cables > 1 and self.friction > 0.0:
            raise ValueError(
                f"cable friction is available for the one-cable digits (ring, little, thumb); {self.digit} has "
                f"{cables} cables"
            )

        # Reference: the hand calculation at every hold of the loop. Every row carries one tension, one
        # travel and one tip tension per cable.
        self.table = self._reference_table(spec, args)
        seating = 0
        if rod:
            # Seating stroke: all cables together from the rest tension to 16 N and back, then the loop. The
            # first two holds (rest, 16 N) are not compared; the third, at rest after the stroke, is the
            # travel reference and must be within 3 % of the rest tension.
            levels = cable_hand.rod_seating_levels(spec, self.digit)  # rest, 16 N, rest
            stroke = [cable_hand.frictionless_row(spec, self.digit, level) for level in levels]
            self.table = [*(dict(row, tip_tension=list(row["T"])) for row in stroke), *self.table]
            seating = 2
        self.seating_levels = seating
        self.rod_rest_error = None

        # Model. Gravity acts along the flexion axis (world -z), so it does not load the joints, and it is
        # applied as body forces: both cable models take external forces inside the solve.
        self.frame_dt = 1.0 / self.FPS
        gravity = (0.0, 0.0, -cable_hand.GRAVITY)
        look = cable_hand.visual_materials(xray=args.xray)
        device = wp.get_device()
        if not rod:
            self.substeps = cable_hand.TENDON_SUBSTEPS
            self.model, self.skeleton, self.cables = cable_hand.build_tendon_finger_cables_model(
                spec, self.digit, mu=self.friction, gravity=(0.0, 0.0, 0.0), device=device, visual_materials=look
            )
            self.solver = cable_hand.make_tendon_solver(self.model)
            self.pipeline = None
            self.contacts = None
            self.probes = None
            travel_start = [0.0] * cables
        else:
            self.substeps = cable_hand.ROD_SUBSTEPS
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            cable_hand.configure_rod_builder(builder)
            travel_start = cable_hand.rod_start_travels(spec, self.digit)
            self.skeleton = cable_hand.add_single_finger(builder, spec, self.digit, travel=travel_start)
            self.cables = cable_hand.add_rod_cables(builder, spec, self.skeleton)
            cable_hand.add_rod_finger_articulation(builder, self.skeleton, self.cables)
            cable_hand.add_digit_visuals(builder, spec, self.skeleton, materials=look)
            for rod_cable in self.cables:  # the visual meshes show the pulleys; hide their collision shapes
                for shapes in rod_cable.station_shapes.values():
                    for shape in shapes:
                        builder.shape_flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)
            builder.color()
            self.model = builder.finalize(device=device)
            # the pipeline comes first: the solver sizes its contact history from it
            self.pipeline = cable_hand.create_rod_collision_pipeline(self.model)
            self.contacts = self.pipeline.contacts()
            solve = args.rigid_articulation_solve or "block_sparse_levels"
            self.solver = cable_hand.create_rod_solver(
                self.model,
                self.frame_dt / self.substeps,
                inertia_reference=cable_hand.rod_inertia_reference(spec),
                rigid_body_contact_buffer_size=cable_hand.rod_contact_buffer(spec, self.digit),
                rigid_articulation_solve=solve,
            )
            self.probes = [cable_hand.RodTensionProbe(self.model, self.skeleton, r) for r in self.cables]
            self.rod_shapes = [np.array(r.rod_shapes) for r in self.cables]
        self.sim_dt = self.frame_dt / self.substeps
        self.weight = cable_hand.create_body_weight(self.model, gravity)

        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        cable_hand.create_spring_preload(self.model, self.skeleton).apply(self.control)

        self.q_index = list(self.skeleton.q_flexion)
        self.slider_index = list(self.skeleton.q_sliders)
        # Travel is measured from the state at the rest tension: zero for the tendons; for the rods the
        # travel at the end of the seating stroke.
        self.travel_zero = np.array(travel_start, dtype=float)
        self.travel_zero_measured = not rod

        # Command: one row of the table per period. The command is blended from row to row over 0.6 s (the
        # minimum-jerk curve of the hand's keyframe scripts), not stepped: a stepped knuckle command lets the
        # curl cable of a two-cable finger dip toward slack. The move to a row then gets the time the slowest
        # actuator needs for its largest travel step at its reference rate (the curl cable of a two-cable
        # finger is ramped more slowly than the others), half the blend time and a little to settle.
        actuator = spec.actuator
        hold = max(spec.staircase.hold, ROD_HOLD) if rod else spec.staircase.hold
        travel = np.array([row["travel"] for row in self.table])
        steps = np.abs(np.diff(np.vstack([travel, travel[:1]]), axis=0)).max(axis=0)
        rates = np.array([cable_hand.actuator_reference_rate(spec, self.digit, c) for c in range(cables)])
        blend = cable_hand.STAIRCASE_BLEND
        move = max(blend, float((steps / rates).max()) + 0.5 * blend) + 0.15
        self.frames_per_level = math.ceil((move + hold) * self.FPS - 1.0e-9)
        period = self.frames_per_level * self.frame_dt
        self.cycle_frames = self.frames_per_level * len(self.table)
        if self.position_mode:
            # the drive needs the lead `tension / kp` over the slider to carry the tension
            lead = np.array([row["T"] for row in self.table]) / actuator.kp
            commands = np.array(travel_start)[None, :] + travel + lead
            mode = cable_hand.GOVERNOR_POSITION
        else:
            commands = np.array([row["T"] for row in self.table])
            mode = cable_hand.GOVERNOR_FORCE
        self.staircase = cable_hand.create_staircase_command(
            spec, levels=commands, hold=hold, transition=period - hold, device=device, blend=blend, cyclic=True
        )
        self.governor = cable_hand.create_actuator_governor(self.model, spec, self.skeleton, mode=mode)

        # Read-back and bookkeeping.
        self.joint_q = wp.clone(self.model.joint_q)
        self.joint_qd = wp.clone(self.model.joint_qd)
        self.frame = 0
        self.sim_time = 0.0
        self.level = 0
        self.row = self.table[0]
        self.q = np.zeros(3)
        self.travel = np.zeros(cables)
        self.tension_actuator = np.full(cables, self.rest_tension)
        self.tension_tip = np.full(cables, self.rest_tension)
        self.tension_min = math.inf
        self.tension_max = -math.inf
        # lowest tension of every cable after the start (after the seating stroke for rods), at frame ends
        self.tension_floor = np.full(cables, math.inf)
        self.holds = []
        self.finite = True
        self.cable_radius = 0.5 * spec.cable.diameter

        self.viewer.set_model(self.model)
        # Look down the flexion axis, framed on the base and the sweep of the finger from straight to closed;
        # the thumb's actuator lies across the digit and a second actuator sits further back: a wider view.
        if self.digit == "thumb":
            camera_pos = wp.vec3(0.03, -0.078, 0.26)
        elif cables > 1:
            camera_pos = wp.vec3(-0.02, 0.0, 0.26)
        else:
            camera_pos = wp.vec3(0.0025, 0.0, 0.187)
        self.viewer.set_camera(pos=camera_pos, pitch=-80.0, yaw=90.0)
        if not rod:
            # the tendons have no shapes: they are drawn as cylinders (unit prototype, scaled per piece)
            piece = newton.Mesh.create_cylinder(1.0, 1.0, up_axis=newton.Axis.Z, segments=12, compute_inertia=False)
            self.viewer.log_mesh(
                "/cable_piece",
                wp.array(piece.vertices, dtype=wp.vec3),
                wp.array(piece.indices, dtype=wp.int32),
                hidden=True,
            )
        self._update_readback()

        self.graph = None
        if self.model.device.is_cuda:
            # Capture after set_model() so mouse-picking buffers are included.
            # The even substep count returns both state buffers to their starting roles.
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph

    def _reference_table(self, spec, args) -> list[dict]:
        """Rows of the hand calculation for the loop; ``T``, ``travel`` and ``tip_tension`` are per cable."""
        digit = spec.digits[self.digit]
        lo, hi = self.rest_tension, self.tension_limit
        if len(digit.cables) == 1:
            if args.tension_pairs is not None:
                raise ValueError(f"tension-pairs needs a two-cable finger; {self.digit} has one cable")
            if args.tension_levels is not None:
                levels = [float(v) for v in args.tension_levels]
                if any(b <= a for a, b in itertools.pairwise(levels)):
                    raise ValueError("tension-levels must increase")
                if not lo <= levels[0] or not levels[-1] <= hi:
                    raise ValueError(f"tension-levels must be in [{lo:g}, {hi:g}] N")
                if levels[0] > lo:
                    levels.insert(0, lo)
                if len(levels) < 2:
                    raise ValueError("tension-levels needs a level above the rest tension")
                spec.staircase.tensions = tuple(levels)
            rows = cable_hand.staircase_table(spec, self.digit, self.friction)
            return [dict(row, T=[row["T"]], travel=[row["travel"]], tip_tension=[row["tip_tension"]]) for row in rows]
        if args.tension_levels is not None:
            raise ValueError(f"{self.digit} has two cables: use --tension-pairs CURL,KNUCKLE ...")
        if args.tension_pairs is None:
            loop = two_cable_loop(spec, self.digit)
        else:
            loop = []
            for token in args.tension_pairs:
                try:
                    curl, knuckle = (float(v) for v in token.split(","))
                except ValueError as error:
                    raise ValueError(f"tension-pairs takes CURL,KNUCKLE in newtons, got {token!r}") from error
                if not (lo <= curl <= hi and lo <= knuckle <= hi):
                    raise ValueError(f"tension-pairs must be in [{lo:g}, {hi:g}] N")
                loop.append({"curl": curl, "knuckle": knuckle})
            if len(loop) < 2:
                raise ValueError("tension-pairs needs at least two pairs")
        rows = []
        for tension in loop:
            row = cable_hand.frictionless_row(spec, self.digit, tension)
            rows.append(dict(row, tip_tension=list(row["T"])))
        return rows

    def simulate(self):
        """Advance one frame with operations that can be captured and replayed on CUDA."""
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.weight.apply(self.state_0)
            self.viewer.apply_forces(self.state_0)
            self.staircase.step(self.governor.command, self.sim_dt)
            self.governor.step(self.state_0, self.control, self.sim_dt)
            if self.pipeline is not None:
                self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.level = (self.frame % self.cycle_frames) // self.frames_per_level
        self.row = self._reference_row(self.level, self.frame // self.cycle_frames)
        self.frame += 1
        self.sim_time += self.frame_dt
        self._update_readback()
        if self.frame % self.frames_per_level == 0:
            self._record_hold()

    def _reference_row(self, level: int, loop: int) -> dict:
        """Return the row of the hand calculation that a hold of the loop is compared with.

        The first rest hold of the first loop is reached from the build state
        and is compared with the loading row. In every later loop the cable
        arrives at the rest tension from above, so with cable friction it
        still carries the unloading profile: that rest hold is compared with
        the last row of the table, the rest row of the unloading branch.
        """
        if level == 0 and loop > 0 and self.friction > 0.0:
            return self.table[-1]
        return self.table[level]

    def _update_readback(self):
        """Read joint angles, actuator travels and cable tensions of the current state."""
        newton.eval_ik(self.model, self.state_0, self.joint_q, self.joint_qd)
        joint_q = self.joint_q.numpy()
        self.q = joint_q[self.q_index].astype(float)
        self.travel = joint_q[self.slider_index].astype(float) - self.travel_zero
        if self.probes is None:
            tensions = [cable_hand.tendon_span_tensions(self.solver, tendon) for tendon in self.cables]
        else:
            body_q = self.state_0.body_q.numpy()
            tensions = [probe.tension(body_q) for probe in self.probes]
            # the rod segments are model shapes: write their colors where the viewer reads them
            colors = self.model.shape_color.numpy()
            for probe, shapes, role in zip(self.probes, self.rod_shapes, self.cable_roles, strict=True):
                segment_tension = cable_hand.rod_segment_tensions(probe, body_q)
                colors[shapes] = cable_hand.tension_colors(segment_tension, self.tension_limit, role)
            self.model.shape_color.assign(colors)
        self.tension_actuator = np.array([float(t[0]) for t in tensions])
        self.tension_tip = np.array([float(t[-1]) for t in tensions])
        every = np.concatenate(tensions)
        self.finite = self.finite and bool(np.isfinite(joint_q).all() and np.isfinite(every).all())
        if self.frame > 0:
            self.tension_min = min(self.tension_min, float(every.min()))
            self.tension_max = max(self.tension_max, float(every.max()))
        # the tendons take the first 0.2 s to reach the rest tension from the build state; the rods are
        # counted once their seating stroke is done
        started = self.travel_zero_measured if self.probes is not None else self.frame > 0.2 * self.FPS
        if started:
            self.tension_floor = np.minimum(self.tension_floor, [float(t.min()) for t in tensions])

    def _record_hold(self):
        """Compare the state at the end of a hold with the hand calculation."""
        loop = (self.frame - 1) // self.cycle_frames
        if self.level < self.seating_levels:
            if not self.travel_zero_measured and self.level == 0 and loop == 0:
                return  # the rest hold before the seating stroke
            if self.level == self.seating_levels - 1:
                return  # the seating hold itself is not compared
        if not self.travel_zero_measured:
            # first compared hold of the rods, at the rest tension after the seating stroke: the travel reference
            self.travel_zero = self.travel_zero + self.travel
            self.travel = np.zeros_like(self.travel)
            self.travel_zero_measured = True
            # how far the cables are from the rest tension after the stroke, at both ends
            ends = np.concatenate([self.tension_actuator, self.tension_tip])
            self.rod_rest_error = float(np.abs(ends / self.rest_tension - 1.0).max())
        row = self.row
        tension = np.array(row["T"])
        tip = np.array(row["tip_tension"])
        self.holds.append(
            {
                "level": self.level,
                "tension": [float(v) for v in tension],
                "angle_error": float(np.abs(np.degrees(self.q - np.array(row["q"]))).max()),
                "travel_error": 1.0e3 * float(np.abs(self.travel - np.array(row["travel"])).max()),
                "actuator_error": float(np.abs(self.tension_actuator / tension - 1.0).max()),
                "tip_error": float(np.abs(self.tension_tip / tip - 1.0).max()),
            }
        )
        del self.holds[: -len(self.table)]  # keep one loop

    def gui(self, ui):
        """Show the commands, the measured cable tensions and the finger pose against the hand calculation."""
        row = self.row
        ui.text(f"Cable model: {self.cable_model}, digit: {self.digit}")
        for c, name in enumerate(self.cable_names):
            if self.position_mode:
                command = f"travel {1.0e3 * row['travel'][c]:.2f} mm (for {row['T'][c]:.0f} N)"
            else:
                command = f"tension {row['T'][c]:.1f} N"
            ui.text(f"{name} cable: commanded {command}")
            ui.text(f"  at actuator {self.tension_actuator[c]:6.2f} N, at its end {self.tension_tip[c]:6.2f} N")
        ui.separator()
        ui.text("Measured / hand calculation at the hold")
        for name, measured, reference in zip(("Joint 1", "Joint 2", "Joint 3"), self.q, row["q"], strict=True):
            ui.text(f"{name}: {math.degrees(measured):6.2f} / {math.degrees(reference):6.2f} deg")
        for c, name in enumerate(self.cable_names):
            ui.text(f"Travel, {name} actuator: {1.0e3 * self.travel[c]:6.2f} / {1.0e3 * row['travel'][c]:6.2f} mm")
        if self.holds:
            last = self.holds[-1]
            at = " / ".join(f"{v:.0f}" for v in last["tension"])
            ui.text_wrapped(
                f"Last hold ({at} N): angle error {last['angle_error']:.2f} deg, "
                f"travel error {last['travel_error']:.2f} mm"
            )
        if len(self.cable_names) > 1:
            ui.text_wrapped(
                "Cable color: curl cable cyan-blue, knuckle cable magenta-pink; brighter with tension "
                f"(full color at {self.tension_limit:g} N)."
            )
        else:
            ui.text_wrapped(f"Cable color: brighter with tension (full color at {self.tension_limit:g} N).")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.probes is None:
            xforms, scales, colors = [], [], []
            for tendon, role in zip(self.cables, self.cable_roles, strict=True):
                starts, ends, tension = cable_hand.tendon_cable_pieces(self.solver, self.model, self.state_0, tendon)
                piece_xforms, half_length = cable_hand.cable_piece_transforms(starts, ends)
                radius = np.full(len(piece_xforms), self.cable_radius)
                xforms.append(piece_xforms)
                scales.append(np.stack([radius, radius, half_length], axis=1).astype(np.float32))
                colors.append(cable_hand.tension_colors(tension, self.tension_limit, role))
            self.viewer.log_instances(
                "/cable",
                "/cable_piece",
                wp.array(np.concatenate(xforms), dtype=wp.transform),
                wp.array(np.concatenate(scales), dtype=wp.vec3),
                wp.array(np.concatenate(colors), dtype=wp.vec3),
                None,
            )
        self.viewer.end_frame()

    def test_final(self):
        """Check the completed holds against the hand calculation when run with --test."""
        if not self.finite or not np.isfinite(self.state_0.body_q.numpy()).all():
            raise ValueError("Non-finite finger state")
        if not self.holds:
            raise ValueError(f"No completed hold: run at least {self.frames_per_level} frames")
        key = "tendon_friction" if self.friction > 0.0 else self.cable_model
        tolerance = dict(HOLD_TOLERANCE[key])
        if self.cable_model == "rod":
            tolerance["angle"] = ROD_ANGLE_TOLERANCE[self.digit]
            if self.rod_rest_error is None:
                raise ValueError(
                    f"The rod seating stroke was not completed: run at least {3 * self.frames_per_level} frames"
                )
            if self.rod_rest_error > ROD_REST_TOLERANCE:
                raise ValueError(
                    f"After the seating stroke the rod cables are {100.0 * self.rod_rest_error:.1f} % off the rest "
                    f"tension (allowed {100.0 * ROD_REST_TOLERANCE:.0f} %): a rod did not seat on its pulleys"
                )
        if self.position_mode:
            tolerance["tension"] = max(tolerance["tension"], POSITION_TENSION_TOLERANCE)
        for hold in self.holds:
            at = " / ".join(f"{v:.0f}" for v in hold["tension"])
            label = f"hold {hold['level']} at {at} N"
            if hold["angle_error"] > tolerance["angle"]:
                raise ValueError(f"{label}: joint angle is {hold['angle_error']:.3f} deg off the hand calculation")
            if hold["travel_error"] > tolerance["travel"]:
                raise ValueError(f"{label}: actuator travel is {hold['travel_error']:.3f} mm off the hand calculation")
            for end in ("actuator", "tip"):
                if hold[f"{end}_error"] > tolerance["tension"]:
                    raise ValueError(f"{label}: tension at the {end} is {100.0 * hold[f'{end}_error']:.1f} % off")
        margin = TENSION_MARGIN[self.cable_model]
        if self.tension_min < -margin or self.tension_max > self.tension_limit + margin:
            raise ValueError(f"Cable tension left its range: {self.tension_min:.2f} to {self.tension_max:.2f} N")
        if float(self.tension_floor.min()) < TENSION_FLOOR:
            lowest = ", ".join(f"{name} {t:.2f}" for name, t in zip(self.cable_names, self.tension_floor, strict=True))
            raise ValueError(f"A cable came close to slack (lowest tension per cable [N]: {lowest})")
        last = self.holds[-1]
        ends_at_rest = self.frame % self.frames_per_level == 0 and max(last["tension"]) == self.rest_tension
        if ends_at_rest and np.abs(self.tension_actuator - self.rest_tension).max() > 0.1 * self.rest_tension:
            raise ValueError(f"Cable tension at rest is {self.tension_actuator.max():.3f} N")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--cable-model",
            choices=["tendon", "rod"],
            default="tendon",
            help="Cable model: massless routed tendons, or chains of rod segments held on the pulleys by contact.",
        )
        parser.add_argument(
            "--cable-friction",
            type=float,
            default=0.0,
            help="Cable friction coefficient on the guides and sheaves, 0 to 0.05, the validated range "
            "(one-cable digits, tendon with --actuation force only).",
        )
        parser.add_argument(
            "--rigid-articulation-solve",
            choices=["block_sparse_levels", "block_sparse_joints"],
            default=None,
            help="Articulation solve of the rod cables and the finger joints (rod only): block_sparse_levels "
            "(default) or the slower sequential block_sparse_joints.",
        )
        parser.add_argument(
            "--actuation",
            choices=["force", "position"],
            default="force",
            help="Command the cable tensions, or the actuator travels that give the same tensions.",
        )
        parser.add_argument(
            "--digit",
            choices=["index", "middle", "ring", "little", "thumb"],
            default="index",
            help="Digit of the hand to build: index and middle have two cables, the others one.",
        )
        parser.add_argument(
            "--tension-pairs",
            nargs="+",
            default=None,
            metavar="CURL,KNUCKLE",
            help="Loop of a two-cable finger: the tensions of the curl and the knuckle cable at every hold [N].",
        )
        parser.add_argument(
            "--tension-levels",
            type=float,
            nargs="+",
            default=None,
            metavar="N",
            help="Increasing cable tension levels of a one-cable digit [N]; the loop visits them up and back down.",
        )
        parser.add_argument(
            "--xray",
            action="store_true",
            default=False,
            help="Draw the finger shells translucent so the cables and the pulleys show.",
        )
        # one loop of the default finger (eight holds of 133 frames)
        parser.set_defaults(num_frames=1064)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
