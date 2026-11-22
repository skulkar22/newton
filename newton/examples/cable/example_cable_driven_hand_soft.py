# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable-driven hand picks up and squeezes a soft foam ball.

The hand of the `cable_driven_hand` example (five digits, seven cables on
seven linear actuators, massless routed tendons) picks up a foam ball of
56 mm, the kind a hand squeezes. The ball is a finite-element soft body: a
tetrahedral mesh of particles with a soft foam core under a denser skin. The
rigid links of the hand, its cables and the soft body are solved together by
one solver, and the contact between the pads and the particles acts on both
sides: the fingers stop on the ball, and the ball gives way under them.

The ball rests in a seat on a post. The hand comes down on it, the fingers
close under it slowly and draw it about 7 mm up into the palm, the thumb
turns onto it, and the hand rises by 100 mm: the ball ends about 107 mm above
its seat. In the lifted hold the cable tensions of the fingers are raised
from 25 N to 45 N (the thumb from 12 N to 30 N), held and lowered again: the
whole hand closes on the ball, which is squeezed between the fingers, the
thumb and the palm, bulges out between them and springs back. Then the hand
lowers it into the seat, lets go and retreats. One loop takes 17.8 s.

The pads and the seat touch the ball with a contact as soft as the foam's
own side of it (2 kN/m): with the stiff pads of the rigid grasp scenes the
fingers chatter on the ball. A limit of this scene: the pick needs the slow
close (2.5 s). When the fingers close in 2.2 s or less they push the light
ball off its seat before they are under it.

The fingers are moved by cable tension only, the thumb opposition is a motor
and the wrist follows a script. The ball is a set of free particles: no joint
or attachment ties it to the hand, it is held by the contact and the friction
of the pads.

The ball is drawn with a fine sphere that is carried by the tetrahedra and
lies one particle contact radius outside the particles: the surface on which
the ball touches the pads and the post, so no gap shows where they touch. The
coarse simulation mesh itself is not drawn. Expand "Example Options" for the
width of the ball between the thumb and the fingertips, its volume and the
cable tensions. `--xray` draws the hand shells translucent so the cables show.

Success criteria:
- The ball is lifted with the hand: it ends about 107 mm above where it started for a scripted lift
  of 100 mm, because the closing hand draws it about 7 mm into the palm (measured 107.3 mm on the CPU
  and on CUDA; the test asks for 107 mm within 3 mm). It stays in the hand through the lifted hold
  and the squeeze.
- The squeeze is visible: the gap between the thumb pad and the fingertip pads of the index and the
  middle finger closes from 32.8 mm to 23.2 mm (29 %; the test asks for 6 mm), the deformation of the
  ball (root mean square over its particles, against its rest shape) grows from 1.8 mm to 3.2 mm (the
  test asks for 1 mm more), and the ball springs back when the tension is lowered. The CPU and CUDA
  runs agree within 0.05 mm.
- No tetrahedron inverts and no particle shoots out.
- The ball is lowered into its seat and released there: it rests within a few millimeters of where it
  started and has its rest shape again.
- At the end of every frame the cable tensions stay between 0.5 N (no cable goes slack) and the 60 N
  limit, with a margin of 3 N on the limit for the moves.

Runs:
- `python -m newton.examples cable_driven_hand_soft`
  - The hand takes the ball from its seat, lifts it, squeezes it so that it flattens between thumb
    and fingers and bulges, lets it spring back and sets it back into the seat.
- `python -m newton.examples cable_driven_hand_soft --xray`
  - The same with translucent shells: the cables brighten with the squeeze.
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cable import cable_hand, cable_hand_objects

# The closing hand draws the ball this far up into the palm before the lift [m]: the ball ends that much
# higher above its start than the scripted lift (measured 7.3 mm on the CPU and on CUDA). The rise of
# the ball must match the scripted lift plus this draw within the tolerance [m].
LIFT_DRAW = 0.007
LIFT_TOLERANCE = 0.003
# The ball may move in the hand by this much during the lifted hold [m], and it must come to rest within
# this distance of where it started [m].
SLIP_TOLERANCE = 0.002
REST_TOLERANCE = 0.01
# A squeeze must raise the deformation of the ball (root mean square over the particles, against the
# rest shape) by at least this much over the hold [m] (measured 1.45 mm on the CPU and on CUDA) and close
# the gap between the thumb pad and the fingertip pads by at least this much [m] (measured 9.6 mm of
# 32.8 mm on both); at most this fraction of the added deformation may be left when the tension is back at
# the hold value.
SQUEEZE_DEFORMATION = 0.001
SQUEEZE_CLOSING = 0.006
RECOVERY_FRACTION = 0.5
# Largest deformation left at the end, as the root mean square over the particles [m].
SHAPE_TOLERANCE = 0.0005
# Margin on the tension limit for the tension seen at frame ends during the moves [N].
TENSION_MARGIN = 3.0
# No cable may go slack: lowest tension allowed at a frame end [N].
TENSION_FLOOR = 0.5
# Color of the ball: one muted color, off the tension scale of the cables.
OBJECT_COLOR = (0.62, 0.42, 0.36)


class Example:
    FPS = 60

    def __init__(self, viewer, args):
        self.viewer = viewer
        # the default hand, with the ball of its spec at the size of the foam ball
        spec = cable_hand_objects.soft_hand_spec()
        self.tension_limit = spec.cable.tension_max
        self.frame_dt = 1.0 / self.FPS
        self.substeps = cable_hand.TENDON_SUBSTEPS
        self.sim_dt = self.frame_dt / self.substeps
        gravity = (0.0, 0.0, -cable_hand.GRAVITY)
        device = wp.get_device()

        # Grasp plan and model. The model has no gravity: the hand control applies the weight of the bodies
        # and the particle weight helper that of the particles.
        self.plan = cable_hand_objects.soft_object_grasp_plan(spec)
        self.model, self.hand, self.cables, self.scene = cable_hand_objects.build_hand_soft_object_model(
            spec,
            object_xform=self.plan.object_xform,
            palm_xform=self.plan.palm_start,
            pedestal=self.plan.pedestal,
            device=device,
            visuals=True,
        )
        if args.xray:
            cable_hand.set_xray_look(self.model)
        # The surface to draw: a fine sphere carried by the tetrahedra, one particle contact radius outside
        # the particles (where the ball touches the pads and the post).
        self.display = cable_hand_objects.create_soft_display_surface(self.scene)
        self.display_indices = wp.array(self.display.triangles.flatten(), dtype=wp.int32)
        # A pipeline is created before its solver: the solver sizes its contact history from it.
        self.pipeline = cable_hand_objects.create_soft_object_collision_pipeline(self.model)
        self.solver = cable_hand_objects.make_hand_soft_object_solver(self.model)
        self.contacts = self.pipeline.contacts()

        # Hand control (governor of the seven actuators, thumb opposition drive, spring preload, weights,
        # wrist), the weight of the particles, and the script that writes the commands.
        self.hand_control = cable_hand.create_hand_control(self.model, spec, self.hand, gravity=gravity)
        self.particle_weight = cable_hand_objects.create_particle_weight(self.model, gravity)
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
        self.pulses = sum(name.startswith("squeeze_hold") for name in self.plan.phases)

        # Read-back and bookkeeping.
        self.frame = 0
        self.sim_time = 0.0
        self.script_time = 0.0
        self.command = np.full(len(self.actuators), spec.cable.tension_rest)
        self.tension_actuator = np.full(len(self.actuators), spec.cable.tension_rest)
        self.tension_min = math.inf
        self.tension_max = -math.inf
        self.finite = True
        self.object_start = None
        self.object_rise = 0.0
        self.object_speed = 0.0
        self.width = math.nan
        self.column = math.nan
        self.grip_particles = None
        self.volume = 1.0
        self.volume_rest = None
        self.volume_ratio_min = 1.0
        self.deformation = 0.0
        self.slip_reference = None
        self.slip = None
        self.lift_at_hold = None
        self.width_hold = None
        self.width_squeezed = {}
        self.deformation_hold = None
        self.deformation_squeezed = {}
        self.deformation_recovered = None
        self.rise_recovered = None
        self.loop_records = None  # the hold and squeeze readings of the last complete loop
        self.hold_complete = False
        self.set_down_error = None
        self.rest_speed = None
        self.rest_deformation = None
        self.cable_radius = 0.5 * spec.cable.diameter

        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False  # the coarse simulation surface; the display surface is drawn instead
        self.viewer.set_camera(pos=wp.vec3(0.30, -0.32, 0.30), pitch=-24.0, yaw=133.0)
        # the tendons have no shapes: they are drawn as cylinders (unit prototype, scaled per piece)
        piece = newton.Mesh.create_cylinder(1.0, 1.0, up_axis=newton.Axis.Z, segments=12, compute_inertia=False)
        self.viewer.log_mesh(
            "/cable_piece",
            wp.array(piece.vertices, dtype=wp.vec3),
            wp.array(piece.indices, dtype=wp.int32),
            hidden=True,
        )
        self._update_readback()
        self._track_grasp()

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
            self.particle_weight.apply(self.state_0)
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.frame > 0 and self.frame % self.loop_frames == 0:
            self.script.reset()  # play the sequence again
            # the readings of the finished loop are kept: a run that ends before the next squeeze is judged
            # by them
            self.loop_records = self._squeeze_records()
            self.slip_reference = None
            self.grip_particles = None
            self.width_squeezed = {}
            self.deformation_squeezed = {}
            self.deformation_recovered = None
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.frame += 1
        self.sim_time += self.frame_dt
        self.script_time = ((self.frame - 1) % self.loop_frames + 1) * self.frame_dt
        self._update_readback()
        self._track_grasp()

    def _squeeze_records(self):
        """Return the hold and squeeze readings of the current loop."""
        return {
            "run_time": self.script_time,
            "width_hold": self.width_hold,
            "deformation_hold": self.deformation_hold,
            "width_squeezed": self.width_squeezed,
            "deformation_squeezed": self.deformation_squeezed,
            "deformation_recovered": self.deformation_recovered,
            "rise_recovered": self.rise_recovered,
        }

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
        """Follow the ball: rise, slip, width between the pads, volume, and where it comes to rest."""
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        palm_q = body_q[self.hand.palm]
        state = cable_hand_objects.soft_object_state(self.scene, particle_q)
        self.finite = self.finite and bool(np.isfinite(particle_q).all())
        position = state["center"]
        if self.object_start is None:
            self.object_start = position.copy()
            self.volume_rest = state["volume"]
            return
        self.object_rise = float(position[2] - self.object_start[2])
        self.volume = state["volume"] / self.volume_rest
        self.volume_ratio_min = min(self.volume_ratio_min, state["volume_ratio_min"])
        self.deformation = float(np.sqrt(np.mean(state["deformation"] ** 2)))
        self.object_speed = float(np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max())
        phases = self.plan.phases
        time = self.script_time
        hold_start, hold_end = phases["hold"]
        if hold_start <= time <= hold_end + 1.0e-6:
            in_palm = cable_hand_objects.soft_object_in_palm(self.scene, particle_q, palm_q)
            if self.slip_reference is None:
                self.slip_reference = in_palm
                # the material between the thumb pad and the fingertip pads of index and middle finger
                self.grip_particles = cable_hand_objects.soft_grip_particles(self.scene, particle_q, body_q)
            # how far the ball moved in the palm since the start of the hold [m]
            self.slip = float(cable_hand.grasp_slip(in_palm, self.slip_reference)[0])
            self.lift_at_hold = self.object_rise
        # From the hold to the release. Width: the gap between the surfaces of those pads, which hold the
        # ball between them. Column: the length of that material along the line between the pads.
        self.width = self.column = math.nan
        if self.grip_particles is not None and len(self.grip_particles) > 1 and time <= phases["rest"][1]:
            self.width = cable_hand_objects.soft_grip_line(self.scene, body_q)[2]
            self.column = cable_hand_objects.soft_grip_width(self.scene, particle_q, body_q, self.grip_particles)
        if hold_start <= time <= hold_end + 1.0e-6:
            self.width_hold = self.width
            self.deformation_hold = self.deformation
            self.hold_complete = self.hold_complete or time >= hold_end - 1.0e-6
        # each squeeze is read at the first frame at or after the end of its hold
        for pulse in range(1, self.pulses + 1):
            if pulse not in self.width_squeezed and time >= phases[f"squeeze_hold{pulse}"][1] - 1.0e-6:
                self.width_squeezed[pulse] = self.width
                self.deformation_squeezed[pulse] = self.deformation
        if self.deformation_recovered is None and time >= phases["recover"][1] - 1.0e-6:
            self.deformation_recovered = self.deformation
            self.rise_recovered = self.object_rise
        if time >= self.script.duration - 1.0e-6:
            # end of a loop: where the ball rests after the release, how fast it still moves, its shape
            self.set_down_error = float(np.linalg.norm(position - self.object_start))
            self.rest_speed = self.object_speed
            self.rest_deformation = self.deformation

    def gui(self, ui):
        """Show the state of the ball, the grasp and the cable tensions."""
        ui.text(f"t = {self.script_time:5.2f} s")
        ui.text(f"Ball above its start: {1.0e3 * self.object_rise:6.1f} mm")
        if math.isfinite(self.width):
            hold = "" if self.width_hold is None else f" (in the hold {1.0e3 * self.width_hold:5.1f} mm)"
            ui.text(f"Width between thumb pad and fingertip pads: {1.0e3 * self.width:5.1f} mm{hold}")
            ui.text(f"Column of material between them: {1.0e3 * self.column:5.1f} mm")
        ui.text(f"Volume: {100.0 * self.volume:5.1f} %, most compressed tetrahedron: {self.volume_ratio_min:.2f}")
        if self.slip is not None:
            ui.text(f"Slip in the lifted hold: {1.0e3 * self.slip:.2f} mm")
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
        # the ball: the display surface for the current particle positions
        vertices, normals = cable_hand_objects.soft_display_surface_vertices(
            self.display, self.state_0.particle_q.numpy()
        )
        self.viewer.log_mesh(
            "/ball",
            wp.array(vertices, dtype=wp.vec3),
            self.display_indices,
            normals=wp.array(normals, dtype=wp.vec3),
            color=OBJECT_COLOR,
            roughness=0.9,
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
        """Check the lift, the squeezes, the mesh and the rest state when run with --test."""
        if not self.finite or not np.isfinite(self.state_0.body_q.numpy()).all():
            raise ValueError("Non-finite state")
        if self.volume_ratio_min <= 0.0:
            raise ValueError("A tetrahedron of the ball inverted")
        if self.tension_min < TENSION_FLOOR or self.tension_max > self.tension_limit + TENSION_MARGIN:
            raise ValueError(f"Cable tension left its range: {self.tension_min:.2f} to {self.tension_max:.2f} N")
        if not self.hold_complete:
            frames = round(self.plan.phases["hold"][1] * self.FPS)
            raise ValueError(f"The lifted hold was not completed: run at least {frames} frames")
        lift = self.plan.lift + LIFT_DRAW
        if abs(self.lift_at_hold - lift) > LIFT_TOLERANCE:
            raise ValueError(f"The ball was lifted by {1.0e3 * self.lift_at_hold:.1f} mm, not {1.0e3 * lift:.1f} mm")
        if self.slip > SLIP_TOLERANCE:
            raise ValueError(f"The ball slipped {1.0e3 * self.slip:.2f} mm in the lifted hold")
        if not math.isfinite(self.width_hold):
            raise ValueError("The ball is not between the thumb pad and the fingertip pads in the hold")
        # a run that went past the end of a squeeze hold must have read that squeeze; a run that ends in a
        # later loop before its last squeeze is judged by the readings of the last complete loop
        records = self._squeeze_records()
        if self.loop_records is not None and len(self.deformation_squeezed) < self.pulses:
            records = self.loop_records
        deformation_hold, width_hold = records["deformation_hold"], records["width_hold"]
        for pulse in range(1, self.pulses + 1):
            if records["run_time"] < self.plan.phases[f"squeeze_hold{pulse}"][1] - 1.0e-6:
                continue
            if pulse not in records["deformation_squeezed"]:
                raise ValueError(f"Squeeze {pulse} was not recorded")
            deformation, width = records["deformation_squeezed"][pulse], records["width_squeezed"][pulse]
            if not deformation - deformation_hold >= SQUEEZE_DEFORMATION:
                raise ValueError(
                    f"Squeeze {pulse} changed the deformation from {1.0e3 * deformation_hold:.2f} to "
                    f"{1.0e3 * deformation:.2f} mm"
                )
            if not width_hold - width >= SQUEEZE_CLOSING:
                raise ValueError(
                    f"Squeeze {pulse} changed the width from {1.0e3 * width_hold:.1f} to {1.0e3 * width:.1f} mm"
                )
        if records["deformation_recovered"] is not None:
            squeeze = max(records["deformation_squeezed"].values()) - deformation_hold
            if not records["deformation_recovered"] - deformation_hold <= RECOVERY_FRACTION * squeeze:
                raise ValueError(
                    "After the squeezes the ball keeps a deformation of "
                    f"{1.0e3 * records['deformation_recovered']:.2f} mm"
                )
            if abs(records["rise_recovered"] - lift) > 2.0 * LIFT_TOLERANCE:
                raise ValueError(f"After the squeezes the ball is {1.0e3 * records['rise_recovered']:.1f} mm up")
        if self.set_down_error is not None:
            if self.set_down_error > REST_TOLERANCE or self.rest_speed > 0.01:
                raise ValueError(
                    f"The ball ends {1.0e3 * self.set_down_error:.1f} mm from its start at {self.rest_speed:.3f} m/s"
                )
            if self.rest_deformation > SHAPE_TOLERANCE:
                raise ValueError(f"The ball keeps a deformation of {1.0e3 * self.rest_deformation:.2f} mm")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--xray",
            action="store_true",
            default=False,
            help="Draw the hand shells translucent so the cables and the pulleys show.",
        )
        # one loop of the grasp (17.8 s): a plain --test run covers the squeeze, the set-down and the rest in
        # the seat
        parser.set_defaults(num_frames=1068)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
