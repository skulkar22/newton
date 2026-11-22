# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable-driven hand: five underactuated digits pose, pinch and grasp.

A right hand of human size with four fingers and a thumb. Every digit has
three hinge joints with a return spring and preload at every joint. Seven
cables close the digits, each pulled by its own linear actuator in the palm
through a governor with a force limit, a speed limit and a rest tension:

- the index and the middle finger have two cables. The **knuckle cable**
  bends the first joint; the **curl cable** runs to the fingertip and bends
  the two outer joints. Together they give poses one cable cannot: a hook, a
  bent knuckle with straight outer joints, and a pinch in which the knuckle
  cable closes the finger onto the object and holds the first joint, while
  the curl cable and the joint stops set the fingertip force.
- the ring finger, the little finger and the thumb have one flexor cable: a
  rising tension closes their joints in the order first, second, third.

The thumb also swings across the palm on a direct-drive opposition hinge. The
palm sits on a wrist that follows a script. The cables are drawn in the color
of their role: curl cables and single flexors cyan-blue, knuckle cables
magenta-pink. The brightness shows the tension on a fixed scale: dim at the
rest tension of 2 N, the full color at the 60 N limit. Expand "Example
Options" for the commanded and measured tension of every cable and the joint
angles of every digit.

Two cable models (`--cable-model`):

- `tendon` (default): one massless routed tendon per cable. It runs on the
  CPU and on CUDA and follows the hand calculation closely.
- `rod`: every cable is a chain of short rod segments (1 mm; 2 mm on the
  thumb) held on flanged pulleys by contact only, on a prescribed palm. It is
  meant for CUDA. `--rigid-articulation-solve` selects how the solver treats
  the closed chains of rod segments and hand joints: `block_sparse_levels`
  (default) or the slower sequential `block_sparse_joints`.

The cables of this hand run without friction: cable friction is validated for
digits with one cable only (see the finger example, `--digit ring
--cable-friction 0.05`).

Two scenes (`--scene`):

- `choreography` (default): after one stroke of all cables, which seats rod
  cables on their pulleys, the hand waves, counts from one to five, shows a
  hook on the index finger beside a bent knuckle on the middle finger, pinches
  with the index and the middle finger, makes a fist and opens, in a loop of
  about 20 s. Every pose is a set of cable tensions and a thumb opposition angle;
  the joint angles follow from the springs.
- `grasp`: the hand takes an object (`--object can` in a side power grasp
  and `--object ball` from above, both from a pedestal; `--object block` in a
  three-digit pinch from the table), lifts it by 100 mm, tilts it, sets it
  down and lets go. Only the pads of the hand touch the object. `--object`
  applies to this scene only.

`--xray` draws the shells translucent so the cables and the pulleys show.

Success criteria:
- Choreography: the hand runs through wave, one to five, hook and knuckle, two pinches and a fist
  without jitter and without a digit passing through another digit or the palm; at the end of every
  pose hold the joint angles match the hand calculation (tendon within 0.5 deg; rod within 1.3 to
  2.0 deg by digit).
- Grasp: the object is lifted by 100 mm with the hand, stays in place in the hand during the lifted
  hold and the tilt (slip below 2 mm), is set down within a few millimeters of where it stood, and
  rests there after the hand has let go.
- The two cables of the index and of the middle finger differ in color, and every cable brightens
  with its commanded tension; at the end of every frame the measured tensions stay between zero and
  the 60 N limit, with a margin of 3 N for the moves, and after the start no cable falls below 0.5 N.
  With the rod model the two ends of every cable agree within 3 % in tension after the seating stroke
  and the cables stay in the grooves of their pulleys: no rod vertex in a groove is further than the
  side clearance from the groove's mid-plane.

Runs:
- `python -m newton.examples cable_driven_hand`
  - The tendon hand waves, counts one to five, hooks the index finger while the middle finger bends
    at the knuckle only, pinches twice, makes a fist and opens again.
- `python -m newton.examples cable_driven_hand --xray`
  - The same with translucent shells: seven cables run from the actuators in the palm into the digits,
    two of them side by side in the index and in the middle finger.
- `python -m newton.examples cable_driven_hand --scene grasp`
  - The hand closes around the can, the thumb opposes last; the can is lifted, tilted and set down.
- `python -m newton.examples cable_driven_hand --scene grasp --object ball`
  - The hand comes down on the ball, closes three fingers under it and turns the thumb onto it, lifts
    and rolls it, and sets it back on its seat.
- `python -m newton.examples cable_driven_hand --scene grasp --object block`
  - The thumb turns across the palm first; the curl cables shape the index and middle finger, the
    knuckle cables swing them onto the block and press it against the thumb, and the block is lifted
    from the table, tilted and set down.
- `python -m newton.examples cable_driven_hand --cable-model rod --device cuda:0`
  - After the seating stroke the rod hand runs the same choreography; the rods stay in their grooves.
- `python -m newton.examples cable_driven_hand --cable-model rod --scene grasp --device cuda:0 --xray`
  - The rod hand grasps the can; the rod segments are visible inside the translucent shells.
- `python -m newton.examples cable_driven_hand --cable-model rod --scene grasp --object ball --device cuda:0`
  - The rod hand takes the ball from above, lifts and rolls it, and sets it back on its seat. With rod
    cables the grasp scene takes the can and the ball; the pinch of the block is refused with a message.
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cable import cable_hand

# Largest difference between a joint angle and the hand calculation at the end of a pose hold of the
# choreography [deg], and the same for the thumb opposition angle. Every pose is expected at the end of
# the hold that follows its move. Measured over the whole sequence on the tendon hand: joints within
# 0.06 deg, opposition within 0.3 deg (the drive is still settling at the end of the short hold of "five").
# The rod tolerances are those of the rod fingers by digit (1 mm segments; the thumb's rod has 2 mm
# segments and the 2 deg of that segment length); the rod hand's pose sequence was measured within 1.2 deg.
POSE_TOLERANCE = {"tendon": 0.5}
ROD_POSE_TOLERANCE = {"index": 1.3, "middle": 1.8, "ring": 1.6, "little": 1.3, "thumb": 2.0}
OPPOSITION_TOLERANCE = 0.5
# After the seating stroke of the rod cables the tension at the far end of every cable must be within this
# fraction of the tension at its actuator (both averaged over the last 0.25 s of the rest hold); a larger
# difference means a rod did not seat on its pulleys.
ROD_REST_TOLERANCE = 0.03
# Lowest tension any cable may show at a frame end after the start (after the seating stroke for rods) [N].
TENSION_FLOOR = 0.5
# Grasp: the lift must match the scripted height within this distance [m] (the closing fingers draw the
# ball about 2 mm up into the palm), the object may move in the hand by this much during the lifted hold
# [m], and it must come to rest within this distance of where it stood [m].
LIFT_TOLERANCE = 0.005
SLIP_TOLERANCE = 0.002
REST_TOLERANCE = 0.01
# Margin on the tension limit for the tension seen at frame ends during the moves [N].
TENSION_MARGIN = 3.0
# Color of the grasped object.
OBJECT_COLOR = (0.58, 0.47, 0.82)


@wp.kernel
def count_clamped_pivots(clamped: wp.array[wp.int32], total: wp.array[int]):
    """Add the solver's clamped-pivot flag of the last step to a running total."""
    total[0] = total[0] + int(clamped[0])


class Example:
    FPS = 60

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.cable_model = args.cable_model
        self.scene_name = args.scene
        rod = self.cable_model == "rod"
        if self.scene_name != "grasp" and args.object is not None:
            raise ValueError("object applies to the grasp scene; add --scene grasp")
        object_name = args.object or cable_hand.GRASP_OBJECTS[0]
        if not rod and args.rigid_articulation_solve is not None:
            raise ValueError("rigid-articulation-solve applies to the rod cable; the tendon needs the local solve")
        if rod and self.scene_name == "grasp" and object_name == "block":
            # In a run of this example the block was lifted and set down, but while the hand withdrew a knuckle
            # cable fell to 0.47 N, below the 0.5 N this example requires of every cable.
            raise ValueError(
                "the pinch of the block is not validated with rod cables (a knuckle cable comes close to slack "
                "while the hand withdraws); use --object can or --object ball, or --cable-model tendon"
            )

        spec = cable_hand.default_hand_spec()
        self.tension_limit = spec.cable.tension_max
        self.frame_dt = 1.0 / self.FPS
        self.substeps = cable_hand.ROD_SUBSTEPS if rod else cable_hand.TENDON_SUBSTEPS
        self.sim_dt = self.frame_dt / self.substeps
        gravity = (0.0, 0.0, -cable_hand.GRAVITY)
        device = wp.get_device()
        solve = args.rigid_articulation_solve or "block_sparse_levels"
        seating_tension = cable_hand.CHOREOGRAPHY_SEATING[1]

        # Model, collision pipeline and solver. A pipeline is created before its solver: the solver sizes
        # its contact history from it. Gravity is applied as body forces by the hand control.
        self.pipeline = None
        self.plan = None
        self.grasp = None
        if self.scene_name == "grasp":
            if rod:
                # the fingers close in the air during the seating stroke, so the hand starts further away
                self.plan = cable_hand.grasp_plan(spec, object_name, approach=0.14, seating_tension=seating_tension)
                self.model, self.hand, self.cables, self.grasp = cable_hand.build_rod_hand_grasp_model(
                    spec,
                    object_name,
                    object_xform=self.plan.object_xform,
                    palm_xform=self.plan.palm_start,
                    pedestal=self.plan.pedestal,
                    device=device,
                    visuals=True,
                )
            else:
                self.plan = cable_hand.grasp_plan(spec, object_name)
                self.model, self.hand, self.cables, self.grasp = cable_hand.build_tendon_hand_grasp_model(
                    spec,
                    object_name,
                    object_xform=self.plan.object_xform,
                    palm_xform=self.plan.palm_start,
                    pedestal=self.plan.pedestal,
                    device=device,
                    visuals=True,
                )
            self.pipeline = cable_hand.create_grasp_collision_pipeline(self.model)
            if rod:
                self.solver = cable_hand.make_rod_hand_grasp_solver(
                    self.model, self.sim_dt, spec=spec, rigid_articulation_solve=solve
                )
            else:
                self.solver = cable_hand.make_tendon_hand_grasp_solver(self.model)
        else:
            # Home pose: fingers up, the palm toward the camera (world -y), the thumb side along world +x.
            home = cable_hand.frame_to_transform(
                (0.0, 0.0, 0.03), np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
            )
            if rod:
                self.model, self.hand, self.cables = cable_hand.build_rod_hand_model(
                    spec, wrist="kinematic", palm_xform=home, device=device, visuals=True
                )
                self.pipeline = cable_hand.create_rod_collision_pipeline(self.model)
                self.solver = cable_hand.make_rod_hand_solver(
                    self.model, self.sim_dt, spec=spec, rigid_articulation_solve=solve
                )
            else:
                self.model, self.hand, self.cables = cable_hand.build_tendon_hand_model(
                    spec, wrist="kinematic", palm_xform=home, device=device, visuals=True
                )
                self.solver = cable_hand.make_tendon_hand_solver(self.model)
        self.contacts = None if self.pipeline is None else self.pipeline.contacts()
        if args.xray:
            cable_hand.set_xray_look(self.model)
        if self.grasp is not None:
            # one object color for both cable models, off the tension scale of the cables
            colors = self.model.shape_color.numpy()
            colors[self.grasp.object_shape] = OBJECT_COLOR
            self.model.shape_color.assign(colors)

        # Hand control (governor of the seven actuators, thumb opposition drive, spring preload, weights,
        # wrist) and the script that writes its commands.
        self.hand_control = cable_hand.create_hand_control(
            self.model, spec, self.hand, gravity=gravity, start_tension=0.0 if rod else None
        )
        self.schedule = []
        self.poses = {}
        if self.plan is not None:
            self.script = cable_hand.create_hand_script(self.model, spec, self.hand, self.plan.keyframes)
        else:
            # The sequence starts with one stroke of all cables, which seats rod cables on their pulleys. It
            # is given the gravity that the hand control applies: the tensions of the pinch poses depend on it.
            self.script, self.schedule, self.poses = cable_hand.hand_choreography(
                self.model, spec, self.hand, gravity=gravity, seating=True
            )
        self.loop_frames = round(self.script.duration * self.FPS)

        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.hand_control.apply_preload(self.control)

        # Digits (joint angles) and actuators (one per cable: commands, tensions and colors). The cables,
        # the probes and the tensions of the hand are keyed by actuator: "index" is the index finger's curl
        # cable, "index_knuckle" its knuckle cable.
        self.names = list(self.hand.names)
        self.actuators = list(self.hand.actuators)
        self.cable_roles = {
            key: spec.digits[digit].cable(cable).role
            for key, (digit, cable) in zip(self.actuators, self.hand.actuator_cables, strict=True)
        }
        # Rod cables: tension probes, segment shapes for the colors, groove clearance and the pivot counter.
        self.probes = None
        self.pivot_total = None
        # Largest sideways offset of a rod vertex in a groove, measured on the body that carries the groove
        # (the check; the offset from the digit plane on the palm would also count the sideways give of the
        # links under the grasp load; kept for reference).
        self.groove_offset_max = 0.0
        self.groove_clearance = spec.station_hardware.groove_half_width - spec.rod.radius
        if rod:
            cable_hand.hide_rod_stations(self.model, self.cables)
            self.probes = cable_hand.create_rod_hand_probes(self.model, self.hand, self.cables)
            self.rod_shapes = {name: np.array(cable.rod_shapes) for name, cable in self.cables.items()}
            if self.solver.rigid_articulation_pivot_clamped is not None:
                self.pivot_total = wp.zeros(1, dtype=int, device=device)

        # Read-back and bookkeeping.
        count = len(self.names)
        self.frame = 0
        self.sim_time = 0.0
        self.script_time = 0.0
        self.q = np.zeros((count, 3))
        self.q_lower = np.array([[joint.lower for joint in spec.digits[name].joints] for name in self.names])
        self.q_upper = np.array([[joint.upper for joint in spec.digits[name].joints] for name in self.names])
        self.opposition = 0.0
        self.command = np.full(len(self.actuators), spec.cable.tension_rest)
        self.opposition_command = 0.0
        self.tension_actuator = np.full(len(self.actuators), spec.cable.tension_rest)
        self.tension_tip = np.full(len(self.actuators), spec.cable.tension_rest)
        self.tension_min = math.inf
        self.tension_max = -math.inf
        # Lowest tension of every cable after the start, at frame ends. The tendons take the first 0.2 s to
        # reach the rest tension from the build state; the rods are counted once their seating stroke is done.
        self.tension_floor = np.full(len(self.actuators), math.inf)
        self.floor_start = 0.2
        self.rod_rest_error = None
        self.seating_tail = []
        if rod:
            if self.plan is not None:
                self.floor_start = self.plan.phases["seat_rest"][1]
            else:
                self.floor_start = next(hold["end"] for hold in self.schedule if hold.get("seating"))
        self.finite = True
        self.pose_checks = []
        self.object_start = None
        self.object_rise = 0.0
        self.object_speed = 0.0
        self.slip_reference = None
        self.slip = None
        self.lift_at_hold = None
        self.hold_complete = False
        self.set_down_error = None
        self.rest_speed = None
        self.cable_radius = 0.5 * spec.cable.diameter

        self.viewer.set_model(self.model)
        if self.scene_name == "grasp":
            self.viewer.set_camera(pos=wp.vec3(0.30, -0.32, 0.30), pitch=-24.0, yaw=133.0)
        else:
            self.viewer.set_camera(pos=wp.vec3(0.0, -0.37, 0.135), pitch=-2.0, yaw=90.0)
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

    def simulate(self):
        """Advance one frame with operations that can be captured and replayed on CUDA."""
        for _ in range(self.substeps):
            self.script.step(self.hand_control, self.sim_dt)
            self.hand_control.step(self.state_0, self.control, self.sim_dt)  # clears the forces first
            self.viewer.apply_forces(self.state_0)
            if self.pipeline is not None:
                self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            if self.pivot_total is not None:
                wp.launch(
                    count_clamped_pivots,
                    dim=1,
                    inputs=[self.solver.rigid_articulation_pivot_clamped],
                    outputs=[self.pivot_total],
                    device=self.model.device,
                )

    def step(self):
        if self.frame > 0 and self.frame % self.loop_frames == 0:
            self.script.reset()  # play the sequence again
            self.slip_reference = None
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.frame += 1
        self.sim_time += self.frame_dt
        self.script_time = ((self.frame - 1) % self.loop_frames + 1) * self.frame_dt
        self._update_readback()
        if self.plan is not None:
            self._track_grasp()
        else:
            self._check_poses()

    def _update_readback(self):
        """Read joint angles, cable tensions and commands of the current state."""
        joints = cable_hand.hand_joint_state(self.model, self.state_0, self.hand)
        self.q = joints["q"]
        self.opposition = joints["opposition"]
        command = self.script.evaluate(self.script_time)
        self.command = np.asarray(command["command"], dtype=float)
        self.opposition_command = float(command["opposition"])
        if self.probes is None:
            tension = cable_hand.hand_span_tensions(self.solver, self.cables)
            low = min(float(t.min()) for t in tension.values())
            high = max(float(t.max()) for t in tension.values())
            lowest = np.array([float(tension[key].min()) for key in self.actuators])
            self.tension_actuator = np.array([tension[key][0] for key in self.actuators])
            self.tension_tip = np.array([tension[key][-1] for key in self.actuators])
        else:
            body_q = self.state_0.body_q.numpy()
            rods = cable_hand.hand_rod_state(self.probes, body_q)
            low, high = float(rods["tension_min"].min()), float(rods["tension_max"].max())
            lowest = np.asarray(rods["tension_min"], dtype=float)
            self.tension_actuator, self.tension_tip = rods["tension_actuator"], rods["tension_tip"]
            self.groove_offset_max = max(self.groove_offset_max, float(rods["groove_offset"].max()))
            # the rod segments are model shapes: write their colors where the viewer reads them
            colors = self.model.shape_color.numpy()
            for key, probe in self.probes.items():
                segment_tension = cable_hand.rod_segment_tensions(probe, body_q)
                colors[self.rod_shapes[key]] = cable_hand.tension_colors(
                    segment_tension, self.tension_limit, self.cable_roles[key]
                )
            self.model.shape_color.assign(colors)
        self.finite = self.finite and bool(np.isfinite(self.q).all() and math.isfinite(low) and math.isfinite(high))
        if self.frame > 0:
            self.tension_min = min(self.tension_min, low)
            self.tension_max = max(self.tension_max, high)
        elapsed = self.frame * self.frame_dt
        if self.probes is not None and self.rod_rest_error is None:
            # Rest hold after the seating stroke: a rod that lies on its pulleys carries the same tension at
            # both ends. The rule is on the tension at the far end over the tension at the actuator, both
            # averaged over the last 0.25 s of the hold.
            self.seating_tail.append((self.tension_actuator.copy(), self.tension_tip.copy()))
            del self.seating_tail[: -round(0.25 * self.FPS)]
            if elapsed >= self.floor_start - 1.0e-9:
                actuator = np.mean([a for a, _ in self.seating_tail], axis=0)
                tip = np.mean([t for _, t in self.seating_tail], axis=0)
                self.rod_rest_error = float(np.abs(tip / actuator - 1.0).max())
        if elapsed > self.floor_start:
            self.tension_floor = np.minimum(self.tension_floor, lowest)

    def _check_poses(self):
        """At the end of a pose hold, compare the joint angles with the pose's closed form."""
        for hold in self.schedule:
            if math.floor(hold["end"] * self.FPS + 1.0e-6) != round(self.script_time * self.FPS):
                continue
            pose = self.poses[hold["pose"]]
            expected = np.array([pose.angles[name] for name in self.names])
            error = np.abs(np.degrees(self.q - expected)).max(axis=1)
            self.pose_checks.append(
                {
                    "pose": hold["pose"],
                    "time": self.script_time,
                    "angle_error": {name: float(e) for name, e in zip(self.names, error, strict=True)},
                    "opposition_error": abs(math.degrees(self.opposition - pose.opposition)),
                }
            )
        del self.pose_checks[: -len(self.schedule)]  # keep one loop

    def _track_grasp(self):
        """Follow the object: rise, slip in the lifted hold, and where it comes to rest."""
        state = cable_hand.grasp_state(self.model, self.state_0, self.hand, self.grasp)
        position = state["object_pose"][:3]
        if self.object_start is None:
            self.object_start = np.array(self.plan.object_xform, dtype=float)[:3]
        self.object_rise = float(position[2] - self.object_start[2])
        self.object_speed = float(np.linalg.norm(state["object_velocity"]))
        self.finite = self.finite and bool(np.isfinite(state["object_pose"]).all())
        hold_start, hold_end = self.plan.phases["hold"]
        if hold_start <= self.script_time <= hold_end + 1.0e-6:
            if self.slip_reference is None:
                self.slip_reference = state["object_in_palm"]
            self.slip = cable_hand.grasp_slip(state["object_in_palm"], self.slip_reference)
            self.lift_at_hold = self.object_rise
            self.hold_complete = self.hold_complete or self.script_time >= hold_end - 1.0e-6
        if self.script_time >= self.script.duration - 1.0e-6:
            # end of a loop: where the object rests after the release, and how fast it still moves
            self.set_down_error = float(np.linalg.norm(position - self.object_start))
            self.rest_speed = self.object_speed

    def gui(self, ui):
        """Show commanded and measured cable tensions, joint angles, opposition and the grasp state."""
        ui.text(f"Cable model: {self.cable_model}, scene: {self.scene_name}, t = {self.script_time:5.2f} s")
        ui.text("Cable: commanded / at actuator / at its end [N]")
        for i, key in enumerate(self.actuators):
            ui.text(
                f"{key:14s} {self.command[i]:5.1f} / {self.tension_actuator[i]:5.1f} / {self.tension_tip[i]:5.1f}"
                f"   ({self.cable_roles[key]})"
            )
        ui.text("Digit: joints [deg]")
        for i, name in enumerate(self.names):
            angles = " ".join(f"{math.degrees(v):5.1f}" for v in self.q[i])
            ui.text(f"{name:6s} {angles}")
        ui.text(
            f"Thumb opposition: {math.degrees(self.opposition):5.1f} deg"
            f" (commanded {math.degrees(self.opposition_command):5.1f})"
        )
        ui.separator()
        if self.plan is not None:
            ui.text(f"Object above its start: {1.0e3 * self.object_rise:6.1f} mm")
            if self.slip is not None:
                ui.text(f"Slip in the lifted hold: {1.0e3 * self.slip[0]:.2f} mm, {math.degrees(self.slip[1]):.2f} deg")
        elif self.pose_checks:
            last = self.pose_checks[-1]
            ui.text_wrapped(
                f"Last pose '{last['pose']}': largest joint angle error {max(last['angle_error'].values()):.2f} deg"
            )
        ui.text_wrapped(
            "Cable color: curl cables and single flexors cyan-blue, knuckle cables magenta-pink; brighter with "
            "tension "
            f"(full color at {self.tension_limit:g} N)."
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.probes is None:
            pieces, colors = [], []
            for key, tendon in self.cables.items():
                piece = cable_hand.tendon_cable_pieces(self.solver, self.model, self.state_0, tendon)
                pieces.append(piece[:2])
                # hue by the role of the cable, brightness by the tension of the piece
                colors.append(cable_hand.tension_colors(piece[2], self.tension_limit, self.cable_roles[key]))
            starts, ends = (np.concatenate(part) for part in zip(*pieces, strict=True))
            xforms, half_length = cable_hand.cable_piece_transforms(starts, ends)
            radius = np.full(len(xforms), self.cable_radius)
            scales = np.stack([radius, radius, half_length], axis=1).astype(np.float32)
            self.viewer.log_instances(
                "/cable",
                "/cable_piece",
                wp.array(xforms, dtype=wp.transform),
                wp.array(scales, dtype=wp.vec3),
                wp.array(np.concatenate(colors), dtype=wp.vec3),
                None,
            )
        self.viewer.end_frame()

    def test_final(self):
        """Check the poses or the grasp, the cable tensions and the rod cables when run with --test."""
        if not self.finite or not np.isfinite(self.state_0.body_q.numpy()).all():
            raise ValueError("Non-finite hand state")
        if self.tension_min < -TENSION_MARGIN or self.tension_max > self.tension_limit + TENSION_MARGIN:
            raise ValueError(f"Cable tension left its range: {self.tension_min:.2f} to {self.tension_max:.2f} N")
        if self.probes is not None:
            if self.groove_offset_max > self.groove_clearance:
                raise ValueError(
                    f"A rod cable left its groove: {1.0e3 * self.groove_offset_max:.3f} mm off the groove mid-plane"
                )
            if self.pivot_total is not None and int(self.pivot_total.numpy()[0]) > 0:
                raise ValueError(f"The articulation solve clamped {int(self.pivot_total.numpy()[0])} pivots")
            if self.rod_rest_error is None:
                frames = math.ceil(self.floor_start * self.FPS)
                raise ValueError(f"The rod seating stroke was not completed: run at least {frames} frames")
            if self.rod_rest_error > ROD_REST_TOLERANCE:
                raise ValueError(
                    f"After the seating stroke the two ends of a rod cable differ by {100.0 * self.rod_rest_error:.1f} % "
                    f"in tension (allowed {100.0 * ROD_REST_TOLERANCE:.0f} %): a rod did not seat on its pulleys"
                )
        if float(self.tension_floor.min()) < TENSION_FLOOR:
            lowest = ", ".join(f"{key} {t:.2f}" for key, t in zip(self.actuators, self.tension_floor, strict=True))
            raise ValueError(f"A cable came close to slack (lowest tension per cable [N]: {lowest})")
        if self.plan is None:
            checks = self.pose_checks
            if not checks:
                first = min(hold["end"] for hold in self.schedule)
                raise ValueError(f"No completed pose hold: run at least {math.ceil(first * self.FPS)} frames")
            tolerances = ROD_POSE_TOLERANCE if self.probes is not None else None
            for check in checks:
                # the digit that is furthest over its own tolerance
                excess = {
                    name: error - (tolerances[name] if tolerances else POSE_TOLERANCE["tendon"])
                    for name, error in check["angle_error"].items()
                }
                name = max(excess, key=excess.get)
                error = check["angle_error"][name]
                if excess[name] > 0.0:
                    raise ValueError(f"Pose '{check['pose']}': {name} is {error:.2f} deg off the hand calculation")
                if check["opposition_error"] > OPPOSITION_TOLERANCE:
                    raise ValueError(f"Pose '{check['pose']}': opposition is {check['opposition_error']:.2f} deg off")
            return
        # Grasp rules, the same for both cable models.
        if not self.hold_complete:
            frames = round(self.plan.phases["hold"][1] * self.FPS)
            raise ValueError(f"The lifted hold was not completed: run at least {frames} frames")
        if abs(self.lift_at_hold - self.plan.lift) > LIFT_TOLERANCE:
            raise ValueError(f"The object was lifted by {1.0e3 * self.lift_at_hold:.1f} mm")
        if self.slip[0] > SLIP_TOLERANCE:
            raise ValueError(f"The object slipped {1.0e3 * self.slip[0]:.2f} mm in the lifted hold")
        if self.set_down_error is not None:
            if self.set_down_error > REST_TOLERANCE or self.rest_speed > 0.01:
                raise ValueError(
                    f"The object ends {1.0e3 * self.set_down_error:.1f} mm from its start at {self.rest_speed:.3f} m/s"
                )

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
            "--scene",
            choices=["choreography", "grasp"],
            default="choreography",
            help="Scripted poses in a loop, or a grasp of an object from a pedestal.",
        )
        parser.add_argument(
            "--object",
            choices=list(cable_hand.GRASP_OBJECTS),
            default=None,
            help="Object of the grasp scene (grasp scene only; the default is the can): power grasp of a can, "
            "enveloping grasp of a ball, pinch of a block.",
        )
        parser.add_argument(
            "--rigid-articulation-solve",
            choices=["block_sparse_levels", "block_sparse_joints"],
            default=None,
            help="Articulation solve of the rod cables and the hand joints (rod only): block_sparse_levels "
            "(default) or the slower sequential block_sparse_joints.",
        )
        parser.add_argument(
            "--xray",
            action="store_true",
            default=False,
            help="Draw the hand shells translucent so the cables and the pulleys show.",
        )
        # one loop of the choreography (seating stroke 3.1 s, sequence 16.6 s), so a plain --test run checks
        # every pose up to the final rest; a grasp loop is shorter (12.7 s with tendons, 14.7 s with rods)
        # and is covered as well
        parser.set_defaults(num_frames=1182)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
