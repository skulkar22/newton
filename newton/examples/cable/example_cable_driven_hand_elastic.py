# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable-driven hand lifts headphones with a flexible headband.

The hand of the `cable_driven_hand` example (five digits, seven cables on
seven linear actuators, massless routed tendons) picks up a pair of
headphones. The headphones are one reduced elastic body: a floating frame
plus six mode shapes of the headband, with the two ear cups riding rigidly on
the band ends. The rigid links of the hand, its cables and the reduced
elastic body are solved together by one solver.

The headphones sit with each ear cup in a cradle. The hand comes down on the
crown of the headband, the four fingers curl under the crown cushion and press
the band against the palm, and the hand lifts the headphones by 100 mm. The
band bends visibly under the grip: the ear cups come about 18 mm closer
(15 % of their distance). In the lifted hold the hand moves up and down once,
so the cups swing and settle, and rolls by 30 deg and back. Then it sets the
headphones back into the cradles and lets go.

The fingers are moved by cable tension only and the wrist follows a script.
The headphones are a free body: no joint or attachment ties them to the
hand, they are held by the contact and the friction of the pads. Expand
"Example Options" for the cup distance, the modal coordinates and the cable
tensions.

`--band-stiffness` scales the Young's modulus of the headband (20 MPa, a soft
band that bends visibly): with 250 the band is practically rigid and the cup
distance stays within a fraction of a millimeter; below 0.75 the band has not
settled at the end of the hold. `--xray` draws the hand shells translucent so
the cables show. The headphones and their stand are drawn as a finished
product on the simulated shapes: the drawn band is swept through the deformed
collision mesh of the band, the cups ride on the simulated cylinders, and the
pads of the stand are the simulated cradle faces.
The mode shapes are linear and come from a beam model of the band: the
example shows a stable flexible object in the hand, not accurate deflections.

Success criteria:
- The headphones are lifted by 100 mm with the hand and stay in place in the hand during the lifted
  hold (slip below 2 mm).
- The headband visibly flexes: the distance between the ear cups changes by more than 2 mm while the
  hand closes and lifts, and it has settled (within 0.5 mm) at the end of the lifted hold. With
  `--band-stiffness 250` it changes by less than 0.5 mm.
- After the release the headphones rest in their cradles with their body frame within 16 mm of where
  it started (12.5 mm measured), and the distance between the ear cups is within 9.5 mm of its rest
  value (7.6 mm measured): the band is still bent when the cups come down, so the cups land a little
  inward and slide when the hand lets go; the friction of the cradles holds the rest.
- At the end of every frame the cable tensions stay between 0.5 N (no cable goes slack) and the 60 N
  limit, with a margin of 3 N on the limit for the moves.

Runs:
- `python -m newton.examples cable_driven_hand_elastic`
  - The hand hooks the headband and lifts the headphones; the ear cups move against each other, swing
    after the shake and come to rest; the headphones are set back into the cradles.
- `python -m newton.examples cable_driven_hand_elastic --band-stiffness 250`
  - The same grasp with a practically rigid band: the cups do not move against each other.
- `python -m newton.examples cable_driven_hand_elastic --xray`
  - The same with translucent shells: the cables of the four fingers brighten as they pull.
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cable import cable_hand, cable_hand_objects

# The lift must match the scripted height within this distance [m], the headphones may move in the hand by
# this much during the lifted hold [m], and they must come to rest within this distance of their start [m].
LIFT_TOLERANCE = 0.005
SLIP_TOLERANCE = 0.002
# Where the headphones end. The soft band is still bent when the cups come down into their cradles (the ear
# cups are about 18 mm closer than at rest in the hold), and after the set-down the palm pads still press
# on the band; when the hand leaves, the band opens and the cups slide in their cradles. REST_TOLERANCE
# limits the distance of the body frame, a mean over the whole body, from where it started: 12.5 mm
# (measured); the limit is that with a margin of a quarter. (In the same run one ear cup ends 12.5 mm over
# the edge of its cradle: a different quantity of the same size, which no limit checks.)
REST_TOLERANCE = 0.016
# Change of the cup distance that counts as visible flex of the default band [m], the largest change left
# over the last 0.3 s of the lifted hold [m], and the largest change allowed for a band of 250 times the
# stiffness [m].
FLEX_MINIMUM = 0.002
FLEX_SETTLED = 0.0005
FLEX_RIGID = 0.0005
# After the release the friction of the cradles keeps the band off its rest shape, for the same reason as
# above. FLEX_REST limits how far the distance between the ear cups stays from its rest value: 7.6 mm
# short (measured); the limit is that with a margin of a quarter [m].
FLEX_REST = 0.0095
# Margin on the tension limit for the tension seen at frame ends during the moves [N].
TENSION_MARGIN = 3.0
# No cable may go slack: lowest tension allowed at a frame end [N].
TENSION_FLOOR = 0.5
# The collision meshes the drawn headphones are built on.
MESH_NAMES = ("band", "cup_left", "cup_right")


class Example:
    FPS = 60

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.band_stiffness = float(args.band_stiffness)
        if not math.isfinite(self.band_stiffness) or not 0.75 <= self.band_stiffness <= 2500.0:
            raise ValueError("band-stiffness must be in [0.75, 2500]")
        spec = cable_hand.default_hand_spec()
        self.tension_limit = spec.cable.tension_max
        self.frame_dt = 1.0 / self.FPS
        self.substeps = cable_hand.TENDON_SUBSTEPS
        self.sim_dt = self.frame_dt / self.substeps
        device = wp.get_device()

        # The headphones: modal basis and meshes, then the grasp plan and the model. Gravity is applied as
        # body forces by the hand control; the free-free modes of the headphones carry no net weight.
        self.reduced = cable_hand_objects.headphones_reduced_model(stiffness_scale=self.band_stiffness)
        self.plan = cable_hand_objects.headphones_grasp_plan(spec, self.reduced)
        self.model, self.hand, self.cables, self.scene = cable_hand_objects.build_hand_headphones_model(
            spec,
            self.reduced,
            object_xform=self.plan.object_xform,
            palm_xform=self.plan.palm_start,
            device=device,
            visuals=True,
        )
        if args.xray:
            cable_hand.set_xray_look(self.model)
        # What is drawn: a finished pair of headphones and a stand, built on the simulated shapes (the band
        # collision mesh with its crown cushion, the two cup cylinders, the cradle faces). The plain shapes
        # themselves are hidden: the table and cradle boxes here, the meshes of the elastic body in the viewer.
        flags = self.model.shape_flags.numpy()
        for shape in (self.scene.table_shape, *self.scene.cradle_shapes):
            flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags.assign(flags)
        self.look = cable_hand_objects.HeadphonesLook(*(self.reduced.meshes[name][0] for name in MESH_NAMES))
        self.display_triangles = {}
        # A pipeline is created before its solver: the solver sizes its contact history from it.
        self.pipeline = cable_hand.create_grasp_collision_pipeline(self.model)
        self.solver = cable_hand_objects.make_hand_headphones_solver(self.model)
        self.contacts = self.pipeline.contacts()

        # Hand control (governor of the seven actuators, spring preload, weights, wrist) and the script
        # that writes its commands.
        self.hand_control = cable_hand.create_hand_control(
            self.model, spec, self.hand, gravity=(0.0, 0.0, -cable_hand.GRAVITY)
        )
        self.script = cable_hand.create_hand_script(self.model, spec, self.hand, self.plan.keyframes)
        self.loop_frames = round(self.script.duration * self.FPS)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.hand_control.apply_preload(self.control)

        # Actuators (one per cable): "index" is the index finger's curl cable, "index_knuckle" its knuckle cable.
        self.actuators = list(self.hand.actuators)
        self.cable_roles = {
            key: spec.digits[digit].cable(cable).role
            for key, (digit, cable) in zip(self.actuators, self.hand.actuator_cables, strict=True)
        }

        # Read-back and bookkeeping.
        self.frame = 0
        self.sim_time = 0.0
        self.script_time = 0.0
        self.command = np.full(len(self.actuators), spec.cable.tension_rest)
        self.tension_actuator = np.full(len(self.actuators), spec.cable.tension_rest)
        self.tension_min = math.inf
        self.tension_max = -math.inf
        self.finite = True
        self.object_start = np.array(self.plan.object_xform, dtype=float)[:3]
        self.object_rise = 0.0
        self.object_speed = 0.0
        self.mode_q = np.zeros(self.scene.mode_count)
        self.mode_rate = 0.0
        self.cup_spread = 0.0
        self.flex_max = 0.0
        self.flex_hold = []
        self.slip_reference = None
        self.slip = None
        self.lift_at_hold = None
        self.hold_complete = False
        self.set_down_error = None
        self.rest_speed = None
        self.rest_spread = None
        self.cable_radius = 0.5 * spec.cable.diameter

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_bodies = False  # the plain collision meshes; the display parts are drawn instead
        self.viewer.set_camera(pos=wp.vec3(0.36, -0.30, 0.30), pitch=-22.0, yaw=140.0)
        # the stand: a base plate and two saddles whose pads are the simulated cradle faces (static)
        start = cable_hand_objects.headphones_state(self.model, self.state_0, self.hand, self.scene)
        self.stand = cable_hand_objects.headphones_stand_parts(start["cup_centers"], self.look.cup_radius)
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
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

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
        self._track_grasp()

    def _update_readback(self):
        """Read cable tensions and commands of the current state."""
        self.command = np.asarray(self.script.evaluate(self.script_time)["command"], dtype=float)
        tension = cable_hand.hand_span_tensions(self.solver, self.cables)
        low = min(float(t.min()) for t in tension.values())
        high = max(float(t.max()) for t in tension.values())
        self.tension_actuator = np.array([tension[key][0] for key in self.actuators])
        self.finite = self.finite and math.isfinite(low) and math.isfinite(high)
        if self.frame > 0:
            self.tension_min = min(self.tension_min, low)
            self.tension_max = max(self.tension_max, high)

    def _track_grasp(self):
        """Follow the headphones: rise, flex of the band, slip in the lifted hold, and where they come to rest."""
        state = cable_hand_objects.headphones_state(self.model, self.state_0, self.hand, self.scene)
        position = state["object_pose"][:3]
        self.object_rise = float(position[2] - self.object_start[2])
        self.object_speed = float(np.linalg.norm(state["object_velocity"]))
        self.mode_q = state["mode_q"]
        self.mode_rate = float(np.abs(state["mode_qd"]).max())
        self.cup_spread = float(state["cup_spread"])
        self.finite = self.finite and bool(np.isfinite(state["object_pose"]).all() and np.isfinite(self.mode_q).all())
        phases = self.plan.phases
        hold_start, hold_end = phases["hold"]
        # flex while the hand closes, lifts, shakes and tilts; the set-down presses the cups into the cradles
        if phases["close"][0] <= self.script_time <= phases["lower"][0]:
            self.flex_max = max(self.flex_max, abs(self.cup_spread))
        if hold_start <= self.script_time <= hold_end + 1.0e-6:
            # slip is judged at the crown point, under the palm pads: the body frame is a mean over the whole
            # body and shifts against the crown when the soft band flexes, without any slip at the grip
            crown = cable_hand_objects.headphones_crown_in_palm(self.model, self.state_0, self.hand, self.scene)
            if self.slip_reference is None:
                self.slip_reference = crown
                self.flex_hold = []
            self.slip = float(np.linalg.norm(crown - self.slip_reference))
            self.lift_at_hold = self.object_rise
            if self.script_time >= hold_end - 0.3:
                self.flex_hold.append(self.cup_spread)
            self.hold_complete = self.hold_complete or self.script_time >= hold_end - 1.0e-6
        if self.script_time >= self.script.duration - 1.0e-6:
            # end of a loop: where the headphones rest after the release, and how fast they still move
            self.set_down_error = float(np.linalg.norm(position - self.object_start))
            self.rest_speed = max(self.object_speed, self.mode_rate)
            self.rest_spread = self.cup_spread

    def gui(self, ui):
        """Show the state of the headband, the grasp and the cable tensions."""
        ui.text(f"Band stiffness x{self.band_stiffness:g}, t = {self.script_time:5.2f} s")
        ui.text(f"Cup distance against rest: {1.0e3 * self.cup_spread:6.2f} mm")
        ui.text("Modal coordinates [mm]: " + " ".join(f"{1.0e3 * v:6.2f}" for v in self.mode_q))
        ui.text(f"Headphones above their start: {1.0e3 * self.object_rise:6.1f} mm")
        if self.slip is not None:
            ui.text(f"Slip of the crown in the palm during the lifted hold: {1.0e3 * self.slip:.2f} mm")
        ui.separator()
        ui.text("Cable: commanded / at actuator [N]")
        for i, key in enumerate(self.actuators):
            ui.text(f"{key:14s} {self.command[i]:5.1f} / {self.tension_actuator[i]:5.1f}   ({self.cable_roles[key]})")
        ui.text_wrapped(
            "Cable color: curl cables and single flexors cyan-blue, knuckle cables magenta-pink; brighter with "
            f"tension (full color at {self.tension_limit:g} N)."
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # the headphones: the display parts on the deformed collision meshes (the triangles never change)
        meshes = cable_hand_objects.headphones_mesh_vertices(self.model, self.state_0, self.scene, MESH_NAMES)
        for part in [*self.look.parts(*(meshes[name] for name in MESH_NAMES)), *self.stand]:
            name = part["name"]
            if name not in self.display_triangles:
                self.display_triangles[name] = wp.array(part["triangles"].ravel(), dtype=wp.int32)
            self.viewer.log_mesh(
                f"/headphones/{name}",
                wp.array(part["vertices"], dtype=wp.vec3),
                self.display_triangles[name],
                normals=wp.array(part["normals"], dtype=wp.vec3),
                **cable_hand_objects.HEADPHONES_MATERIALS[part["material"]],
            )
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
        """Check the lift, the flex of the band, the slip and the rest state when run with --test."""
        if not self.finite or not np.isfinite(self.state_0.body_q.numpy()).all():
            raise ValueError("Non-finite state")
        if self.tension_min < TENSION_FLOOR or self.tension_max > self.tension_limit + TENSION_MARGIN:
            raise ValueError(f"Cable tension left its range: {self.tension_min:.2f} to {self.tension_max:.2f} N")
        if not self.hold_complete:
            frames = round(self.plan.phases["hold"][1] * self.FPS)
            raise ValueError(f"The lifted hold was not completed: run at least {frames} frames")
        if abs(self.lift_at_hold - self.plan.lift) > LIFT_TOLERANCE:
            raise ValueError(f"The headphones were lifted by {1.0e3 * self.lift_at_hold:.1f} mm")
        if self.slip > SLIP_TOLERANCE:
            raise ValueError(f"The headphones slipped {1.0e3 * self.slip:.2f} mm in the lifted hold")
        if self.band_stiffness >= 250.0:
            if self.flex_max > FLEX_RIGID:
                raise ValueError(f"The stiff band flexed: cup distance changed by {1.0e3 * self.flex_max:.2f} mm")
        elif self.band_stiffness <= 1.0 and self.flex_max < FLEX_MINIMUM:
            raise ValueError(f"The band did not flex: cup distance changed by {1.0e3 * self.flex_max:.2f} mm")
        if not self.flex_hold:
            raise ValueError("No cup distance was recorded over the last 0.3 s of the lifted hold")
        if float(np.ptp(self.flex_hold)) > FLEX_SETTLED:
            raise ValueError(f"The band still swings at the end of the hold: {1.0e3 * np.ptp(self.flex_hold):.2f} mm")
        if self.set_down_error is not None:
            if self.set_down_error > REST_TOLERANCE or self.rest_speed > 0.01:
                raise ValueError(
                    f"The headphones end {1.0e3 * self.set_down_error:.1f} mm from their start at "
                    f"{self.rest_speed:.3f} m/s"
                )
            if abs(self.rest_spread) > FLEX_REST:
                raise ValueError(f"The band is {1.0e3 * self.rest_spread:.2f} mm off its rest shape after the release")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--band-stiffness",
            type=float,
            default=1.0,
            help="Factor on the Young's modulus of the headband, 0.75 to 2500; 250 makes it practically rigid.",
        )
        parser.add_argument(
            "--xray",
            action="store_true",
            default=False,
            help="Draw the hand shells translucent so the cables and the pulleys show.",
        )
        # one loop of the grasp (12.75 s): a plain --test run covers the release and the rest in the cradles
        parser.set_defaults(num_frames=765)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
