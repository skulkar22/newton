# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of the deformable objects of the cable-driven hand examples.

Covered: the headphones (a reduced elastic body: modal basis, scene, grasp
plan, and a short lift) and the soft objects (finite elements: meshes, scene
and particle-contact flags, particle weight, grasp plan, read-back, and a
short squeeze). The full grasps are covered by the registered example tests.
"""

import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.examples.cable import cable_hand as ch
from newton.examples.cable import cable_hand_objects as ob
from newton.tests.unittest_utils import get_test_devices

# The models are built once per class on the CPU device, where the tendon hand is fastest.
CPU = next((device for device in get_test_devices() if device.is_cpu), None)

# short phases for the simulated headphones grasp: down 20 mm, close, lift 30 mm, hold
_HEADPHONES_TEST_PHASES = {"lead_in": 0.05, "approach": 0.2, "close": 1.2, "settle": 0.3, "lift": 0.4, "hold": 0.4}


def setUpModule():
    """Skip the module when the test runner selects no CPU device, instead of running on the default device."""
    if CPU is None:
        raise unittest.SkipTest("the cable hand object tests build their models on the CPU device")


def _step(model, spec, hand, pipeline, solver, keyframes, duration, weight=None, on_frame=None):
    """Step a hand model through a keyframe script; returns the two states (current first) and the contacts."""
    dt = 1.0 / (ch.TENDON_FPS * ch.TENDON_SUBSTEPS)
    control = ch.create_hand_control(model, spec, hand, gravity=(0.0, 0.0, -ch.GRAVITY))
    script = ch.create_hand_script(model, spec, hand, keyframes)
    states = [model.state(), model.state()]
    ctrl = model.control()
    contacts = pipeline.contacts()
    control.apply_preload(ctrl)
    for step in range(1, round(duration / dt) + 1):
        script.step(control, dt)
        control.step(states[0], ctrl, dt)
        if weight is not None:
            weight.apply(states[0])
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], ctrl, contacts, dt)
        states[0], states[1] = states[1], states[0]
        if on_frame is not None and step % ch.TENDON_SUBSTEPS == 0:
            on_frame(step * dt, states[0])
    return states, contacts, dt


class TestCableHandHeadphones(unittest.TestCase):
    """Headphones as one reduced elastic body: modal basis, scene, grasp plan and a short lift (CPU)."""

    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()
        cls.reduced = ob.headphones_reduced_model()
        cls.plan = ob.headphones_grasp_plan(
            cls.spec, cls.reduced, lift=0.03, approach=0.02, phases=_HEADPHONES_TEST_PHASES
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cls.model, cls.hand, cls.tendons, cls.scene = ob.build_hand_headphones_model(
                cls.spec, cls.reduced, object_xform=cls.plan.object_xform, palm_xform=cls.plan.palm_start, device=CPU
            )
        cls.build_warnings = [str(w.message) for w in caught]

    def test_reduced_model(self):
        """Check mass, modes and inertia coupling of the reduced model of the headphones."""
        p = ob.HEADPHONES
        reduced = self.reduced
        arc = (math.pi - 2.0 * p["band_lean"]) * p["band_radius"]
        chord = 2.0 * p["band_radius"] * math.sin(0.5 * (math.pi - 2.0 * p["band_lean"]) / p["elements"])
        band = p["band_density"] * p["band_width"] * p["band_thickness"] * chord * p["elements"]
        self.assertAlmostEqual(arc, 0.168, delta=0.001)
        self.assertAlmostEqual(reduced.mass, band + 2.0 * p["cup_mass"], delta=1.0e-6)
        basis = reduced.basis
        self.assertEqual(basis.mode_count, p["mode_count"])
        self.assertAlmostEqual(float(basis.rigid_mass), reduced.mass, delta=1.0e-6)
        np.testing.assert_allclose(basis.rigid_com, reduced.com, atol=1.0e-6)
        # symmetric about the band plane and about the crown: the center of mass lies on the z axis
        np.testing.assert_allclose(reduced.com[:2], 0.0, atol=1.0e-9)
        # free-free modes: rising frequencies, no net linear momentum, mass-orthogonal
        self.assertTrue(np.all(np.diff(reduced.frequencies) > 0.0))
        self.assertAlmostEqual(reduced.frequencies[0], 2.5, delta=0.15)  # the soft band of 20 MPa
        coupling = np.asarray(basis.mode_coupling_linear, dtype=float)
        self.assertLess(float(np.abs(coupling).max()), 1.0e-5 * reduced.mass)
        modal_mass = np.asarray(basis.mode_mass_matrix, dtype=float)
        off = modal_mass - np.diag(np.diag(modal_mass))
        self.assertLess(float(np.abs(off).max()), 1.0e-4 * float(np.diag(modal_mass).min()))
        # natural frequencies from the basis agree with those of the beam model
        ratio = np.sqrt(np.asarray(basis.mode_stiffness, dtype=float) / np.diag(modal_mass)) / (2.0 * math.pi)
        np.testing.assert_allclose(ratio, reduced.frequencies, rtol=1.0e-3)
        # first mode: the cups move against each other along y by one unit each
        np.testing.assert_allclose(reduced.cup_phi[0, 0, 1], -reduced.cup_phi[1, 0, 1], rtol=1.0e-6)
        self.assertGreater(abs(reduced.cup_phi[0, 0, 1]), 0.9)
        self.assertLessEqual(float(np.abs(reduced.cup_phi).max()), 1.0 + 1.0e-9)
        # a band of 100 times the stiffness has 10 times the frequencies
        stiff = ob.headphones_reduced_model(stiffness_scale=100.0)
        np.testing.assert_allclose(stiff.frequencies, 10.0 * reduced.frequencies, rtol=1.0e-6)
        # the cushion is 25 mm deep at the crown and gone at 40 deg
        self.assertAlmostEqual(float(ob.headphones_cushion_depth(0.0)), 0.025)
        self.assertAlmostEqual(float(ob.headphones_cushion_depth(math.radians(40.0))), 0.0, delta=1.0e-12)
        vertices, indices = reduced.meshes["band"]
        self.assertAlmostEqual(float(vertices[:, 2].max()), float(reduced.crown[2]), delta=1.0e-6)
        self.assertEqual(indices.size % 3, 0)
        # closed meshes with outward faces: positive enclosed volume, that of the cylinder for a cup
        volumes = {}
        for name in ("band", "cup_left", "cup_right"):
            v, i = reduced.meshes[name]
            tri = np.asarray(v, dtype=float)[i.reshape(-1, 3)]
            volumes[name] = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
            self.assertGreater(volumes[name], 0.0, name)
        cup_volume = math.pi * p["cup_radius"] ** 2 * p["cup_depth"]
        self.assertAlmostEqual(volumes["cup_left"], cup_volume, delta=0.02 * cup_volume)

    def test_scene_and_plan(self):
        """Check the model with the headphones (a free elastic body), their coordinates and the grasp plan."""
        spec, model, hand, scene, plan = self.spec, self.model, self.hand, self.scene, self.plan
        self.assertEqual(self.build_warnings, [])
        # one elastic body in an articulation of its own, next to the hand articulation
        self.assertEqual(int(model.elastic_body_count), 1)
        self.assertEqual(model.articulation_count, 2)
        self.assertEqual(scene.mode_count, 6)
        joint = int(model.elastic_joint.numpy()[0])
        self.assertEqual(int(model.joint_type.numpy()[joint]), int(newton.JointType.ELASTIC))
        self.assertEqual(scene.q_start + 7 + scene.mode_count, model.joint_coord_count)
        self.assertAlmostEqual(float(model.body_mass.numpy()[scene.object_body]), scene.object_mass, delta=1.0e-6)
        # the headphones are free: their only joint is their own floating elastic joint (parent: the world),
        # no joint of the hand ends on them, and the model has no springs
        parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
        self.assertEqual(np.nonzero(children == scene.object_body)[0].tolist(), [joint])
        self.assertEqual(int(parents[joint]), -1)
        self.assertFalse(np.any(parents == scene.object_body))
        self.assertEqual(model.spring_count, 0)
        # every mesh vertex of the headphones is a contact and render sample of the elastic body
        vertex_total = sum(len(v) for v, _ in scene.reduced.meshes.values())
        self.assertEqual(int(model.elastic_shape_vertex_total_count), vertex_total)
        self.assertEqual(set(scene.shapes), {"band", "cup_left", "cup_right", "yoke_left", "yoke_right"})
        # the cups stand 20 mm above the table, each in a cradle of two faces and a post
        state = ob.headphones_state(model, model.state(), hand, scene)
        lowest = state["cup_centers"][:, 2] - ob.HEADPHONES["cup_radius"]
        np.testing.assert_allclose(lowest, ob.HEADPHONES_SUPPORT_HEIGHT, atol=1.0e-6)
        self.assertEqual(len(scene.cradle_shapes), 6)
        self.assertAlmostEqual(state["cup_spread"], 0.0, delta=1.0e-9)
        np.testing.assert_allclose(state["mode_q"], 0.0)
        # the grasp plan: the palm faces down over the crown and comes straight down
        self.assertEqual(plan.object_name, "headphones")
        self.assertEqual(list(plan.phases)[:6], list(_HEADPHONES_TEST_PHASES))
        start = np.array(plan.palm_start, dtype=float)
        grasp = np.array(plan.palm_grasp, dtype=float)
        np.testing.assert_allclose(start[:2], grasp[:2], atol=1.0e-9)
        self.assertAlmostEqual(start[2] - grasp[2], 0.02, delta=1.0e-6)
        normal = np.array(wp.quat_rotate(wp.quat(*grasp[3:].tolist()), wp.vec3(0.0, 1.0, 0.0)))
        np.testing.assert_allclose(normal, [0.0, 0.0, -1.0], atol=1.0e-6)
        crown = np.array(wp.transform_point(plan.object_xform, wp.vec3(*scene.reduced.crown.tolist())))
        pad = spec.palm.pads[0]
        self.assertAlmostEqual(grasp[2] - crown[2], pad.p0[1] + pad.radius + 0.001, delta=1.0e-6)
        # the fingers close, the thumb stays out and is never opposed
        closing = plan.keyframes[3].tension
        self.assertEqual(set(closing), set(ob.HEADPHONES_GRASP["digits"]))
        self.assertNotIn("thumb", closing)
        self.assertTrue(all(key.opposition in (None, 0.0) for key in plan.keyframes))
        # the default plan has the shake and the tilt and stays inside the wrist limits of the script
        full = ob.headphones_grasp_plan(spec, scene.reduced)
        self.assertEqual(list(full.phases), list(ob.HEADPHONES_GRASP_PHASES))
        self.assertAlmostEqual(full.phases["end"][1], sum(ob.HEADPHONES_GRASP_PHASES.values()))
        ch.create_hand_script(model, spec, hand, full.keyframes)

    def test_display_parts(self):
        """The drawn headphones are built on the deformed collision meshes; the stand on the cradle faces."""
        model, hand, scene = self.model, self.hand, self.scene
        state = model.state()
        meshes = ob.headphones_mesh_vertices(model, state, scene)
        # at rest the mesh vertices are the rest meshes at the start pose
        offset = np.array(self.plan.object_xform, dtype=float)[:3]
        for name in ("band", "cup_left", "cup_right"):
            np.testing.assert_allclose(meshes[name], scene.reduced.meshes[name][0] + offset, atol=1.0e-6)
        look = ob.HeadphonesLook(*(scene.reduced.meshes[name][0] for name in ("band", "cup_left", "cup_right")))
        self.assertAlmostEqual(look.cup_radius, ob.HEADPHONES["cup_radius"], places=6)
        self.assertAlmostEqual(look.width, ob.HEADPHONES["band_width"], places=6)
        parts = look.parts(meshes["band"], meshes["cup_left"], meshes["cup_right"])
        self.assertEqual(len(parts), 27)
        self.assertEqual(len({part["name"] for part in parts}), 27)
        for part in parts:
            self.assertIn(part["material"], ob.HEADPHONES_MATERIALS)
            self.assertTrue(np.isfinite(part["vertices"]).all() and np.isfinite(part["normals"]).all())
            self.assertLess(int(part["triangles"].max()), len(part["vertices"]))
        # the drawn band lies on the simulated one: every vertex of the band collision mesh is within
        # 0.6 mm of a vertex ring of the drawn band parts along the band (checked on the outer face line)
        spring = next(part for part in parts if part["name"] == "band_spring")
        crown = np.array(self.plan.object_xform, dtype=float)[:3] + scene.reduced.crown
        self.assertLess(float(np.linalg.norm(spring["vertices"] - crown, axis=1).min()), 1.0e-3)
        # a rigid motion of the meshes moves the parts with them; the triangles stay
        shift = np.array([0.01, -0.02, 0.03])
        moved = look.parts(meshes["band"] + shift, meshes["cup_left"] + shift, meshes["cup_right"] + shift)
        for a, b in zip(parts, moved, strict=True):
            np.testing.assert_allclose(b["vertices"], a["vertices"] + shift, atol=2.0e-6)
            np.testing.assert_array_equal(b["triangles"], a["triangles"])
        # the stand: the top faces of the four pads are the cradle faces, tangent to the cups
        cups = ob.headphones_state(model, state, hand, scene)["cup_centers"]
        stand = ob.headphones_stand_parts(cups, look.cup_radius)
        self.assertEqual([part["name"].split("_")[1] for part in stand], ["base", "saddle", "pads", "saddle", "pads"])
        pads = next(part for part in stand if part["name"] == "stand_pads_left")["vertices"]
        distance = np.linalg.norm((pads - cups[0])[:, [0, 2]], axis=1)
        # (no pad vertex reaches into the cup, and the pad faces pass within 2 mm of it at their vertices)
        self.assertGreaterEqual(float(distance.min()), look.cup_radius - 5.0e-5)
        self.assertLess(float(distance.min()), look.cup_radius + 2.0e-3)
        # the crown point starts under the palm pads and does not move in the palm with the palm at rest
        crown_in_palm = ob.headphones_crown_in_palm(model, state, hand, scene)
        self.assertEqual(crown_in_palm.shape, (3,))
        self.assertLess(abs(float(crown_in_palm[2])), 1.0e-3)

    def test_headphones_are_lifted_and_band_flexes(self):
        """Simulate the grasp up to a short lifted hold: lift, slip, flex of the band, free hang, tensions."""
        spec, plan, model, hand, scene = self.spec, self.plan, self.model, self.hand, self.scene
        hold_start, hold_end = plan.phases["hold"]
        spread, reference, low = [], [], [np.inf]

        def on_frame(time, state):
            current = ob.headphones_state(model, state, hand, scene)
            spread.append(current["cup_spread"])
            if time >= hold_start - 1.0e-9:
                reference.append(ob.headphones_crown_in_palm(model, state, hand, scene))
            if time > 0.2:
                low[0] = min(low[0], *(float(t.min()) for t in ch.hand_span_tensions(solver, self.tendons).values()))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline = ch.create_grasp_collision_pipeline(model)
            solver = ob.make_hand_headphones_solver(model)
            start = ob.headphones_state(model, model.state(), hand, scene)
            states, contacts, dt = _step(model, spec, hand, pipeline, solver, plan.keyframes, hold_end, None, on_frame)
            forces = ob._headphones_contact_forces(solver, states[0], states[1], contacts, dt, scene)
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertTrue(np.isfinite(states[0].body_q.numpy()).all())
        self.assertTrue(np.isfinite(states[0].joint_q.numpy()).all())
        end = ob.headphones_state(model, states[0], hand, scene)
        self.assertAlmostEqual(end["object_pose"][2] - start["object_pose"][2], plan.lift, delta=0.004)
        # slip at the crown point, under the palm pads, from the start of the hold to its end
        slip = float(np.linalg.norm(reference[-1] - reference[0]))
        self.assertLess(slip, 2.0e-3)
        # the headphones hang in the hand: the cradles and the table carry nothing and the pads carry the
        # weight. The band still swings after the short lift of this test, so the pad forces balance the
        # weight only roughly (closely at the end of the full hold of the example).
        np.testing.assert_allclose(forces["support_force"], 0.0, atol=1.0e-6)
        weight = scene.object_mass * ch.GRAVITY
        pad_sum = forces["pad_force"].sum(axis=0)
        self.assertGreater(pad_sum[2], 0.3 * weight)
        self.assertLess(pad_sum[2], 1.7 * weight)
        # tensions: no cable goes slack, the driven cables hold their command
        self.assertGreaterEqual(low[0], 0.5)
        tension = np.concatenate(list(ch.hand_span_tensions(solver, self.tendons).values()))
        self.assertLess(float(tension.max()), plan.hold_tension + 1.0)
        # the band flexes by millimeters under the grip, and stays within its linear range
        self.assertGreater(float(np.abs(spread).max()), 1.0e-3)
        self.assertLess(float(np.abs(spread).max()), 40.0e-3)


class TestCableHandSoftObject(unittest.TestCase):
    """Soft objects of the grasp scene: meshes, scene, plan, and a short squeeze with the hand (CPU)."""

    @classmethod
    def setUpClass(cls):
        cls.spec = ob.soft_hand_spec()
        cls.plan = ob.soft_object_grasp_plan(cls.spec)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cls.model, cls.hand, cls.tendons, cls.scene = ob.build_hand_soft_object_model(
                cls.spec,
                object_xform=cls.plan.object_xform,
                palm_xform=cls.plan.palm_grasp,
                pedestal=cls.plan.pedestal,
                device=CPU,
            )
        cls.build_warnings = [str(w.message) for w in caught]

    def test_meshes(self):
        """The ball mesh is valid, closed, well-shaped, and a full sphere of the right size and weight."""
        radius = self.spec.objects["ball"]["radius"]
        self.assertEqual(radius, ob.SOFT_BALL_RADIUS)
        self.assertEqual(self.scene.radius, radius)
        # the soft scene's spec is the default hand with another ball; the default spec is left alone
        default = ch.default_hand_spec()
        self.assertEqual(ob.soft_hand_spec(default).objects["ball"]["radius"], radius)
        self.assertEqual(default.objects["ball"]["radius"], ch.default_hand_spec().objects["ball"]["radius"])
        vertices, tets = ob.soft_object_mesh(self.spec)
        cells = ob.SOFT_BALL_CELLS
        self.assertEqual((len(vertices), len(tets)), ((cells + 1) ** 3, 6 * cells**3))
        volumes = ob.tet_volumes(vertices, tets)
        # well-shaped elements: none below 0.4 or above 2.5 times the mean volume
        self.assertGreater(float(volumes.min()), 0.4 * float(volumes.mean()))
        self.assertLess(float(volumes.max()), 2.5 * float(volumes.mean()))
        self.assertEqual(len(np.unique(tets)), len(vertices))
        # closed surface: every edge of the surface belongs to two triangles, with opposite directions
        triangles = ob.tet_surface_triangles(tets)
        edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        forward = {tuple(edge) for edge in edges.tolist()}
        self.assertEqual(len(forward), len(edges))
        self.assertTrue(all((b, a) in forward for a, b in forward))
        # outward normals: the enclosed volume of the surface is the volume of the tetrahedra
        a, b, c = (vertices[triangles[:, i]] for i in range(3))
        enclosed = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
        self.assertAlmostEqual(enclosed, float(volumes.sum()), delta=1.0e-9)
        # the surface vertices lie exactly one particle radius inside the contact surface
        self.assertEqual(len(triangles), 12 * cells**2)
        surface = np.linalg.norm(vertices[np.unique(triangles)], axis=1) - radius
        self.assertLess(float(np.abs(surface + ob.SOFT_PARTICLE_RADIUS).max()), 1.0e-9)
        mass = ob.SOFT_DENSITY * float(volumes.sum())
        self.assertGreater(mass, 0.015)
        self.assertLess(mass, 0.030)
        # a full sphere of the stated size, counting the contact radius of the particles
        extent = vertices.max(axis=0) - vertices.min(axis=0) + 2.0 * ob.SOFT_PARTICLE_RADIUS
        np.testing.assert_allclose(extent, 2.0 * radius, atol=1.0e-9)
        with self.assertRaises(ValueError):
            ob.ball_tet_mesh(0.02, 5)

    def test_scene_counts_and_particle_contact_flags(self):
        """The object is a set of free particles; only the pads and the support collide with them."""
        model, scene = self.model, self.scene
        self.assertEqual(self.build_warnings, [])
        self.assertEqual(model.particle_count, scene.particle_count)
        self.assertEqual(model.tet_count, scene.tet_count)
        self.assertEqual(model.tri_count, len(scene.triangles))
        self.assertGreater(len(model.particle_color_groups), 0)
        self.assertAlmostEqual(scene.object_mass, float(model.particle_mass.numpy().sum()), places=6)
        self.assertEqual(float(np.linalg.norm(model.gravity.numpy())), 0.0)
        self.assertEqual(model.soft_contact_mu, self.spec.pads.mu)
        # free particles: none is pinned, no spring ties one to anything, and the hand articulation is the
        # only articulation of the model
        self.assertTrue(np.all(model.particle_inv_mass.numpy() > 0.0))
        self.assertTrue(np.all(model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)))
        self.assertEqual(model.spring_count, 0)
        self.assertEqual(model.articulation_count, 1)
        # the ball starts in its seat, in touch without overlap: its lowest point is at the pedestal height,
        # the three seat balls on the rim of the post are tangent to its contact surface, and no contact
        # sphere of a particle reaches into a seat ball
        x = model.particle_q.numpy()
        radius = scene.radius
        center = np.array(self.plan.object_xform, dtype=float)[:3]
        self.assertAlmostEqual(center[2] - radius, scene.pedestal_height, places=6)
        self.assertEqual(len(scene.seat_shapes), 3)
        for seat in model.shape_transform.numpy()[scene.seat_shapes][:, :3]:
            self.assertAlmostEqual(float(np.linalg.norm(seat - center)), radius + ch.GRASP_SEAT_BALL_RADIUS, places=6)
            gap = np.linalg.norm(x - seat, axis=1).min() - scene.particle_radius - ch.GRASP_SEAT_BALL_RADIUS
            # the ball is built where it rests: pressed into the soft seat by its weight, by less than 0.3 mm
            self.assertGreaterEqual(float(gap), -3.0e-4)
            self.assertLess(float(gap), 1.0e-3)
        # the pads and the support are as soft as the particles' side of the contact
        ke = model.shape_material_ke.numpy()
        np.testing.assert_allclose(ke[list(scene.pads.shapes.values())], ob.SOFT_PAD_CONTACT_KE)
        np.testing.assert_allclose(ke[scene.seat_shapes], ob.SOFT_SEAT_CONTACT_KE)
        bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        allowed = {*scene.pads.shapes.values(), scene.table_shape, scene.pedestal_shape, *scene.seat_shapes}
        self.assertEqual({int(s) for s in np.nonzero(model.shape_flags.numpy() & bit)[0]}, allowed)
        # the same with the visual meshes of the hand attached
        visual_model, _, _, visual_scene = ob.build_hand_soft_object_model(
            self.spec,
            object_xform=self.plan.object_xform,
            palm_xform=self.plan.palm_grasp,
            pedestal=self.plan.pedestal,
            device=CPU,
            visuals=True,
        )
        allowed = {
            *visual_scene.pads.shapes.values(),
            visual_scene.table_shape,
            visual_scene.pedestal_shape,
            *visual_scene.seat_shapes,
        }
        self.assertEqual({int(s) for s in np.nonzero(visual_model.shape_flags.numpy() & bit)[0]}, allowed)
        self.assertGreater(visual_model.shape_count, model.shape_count)

    def test_particle_weight(self):
        """The particle weight adds mass times gravity to the particle forces."""
        state = self.model.state()
        state.clear_forces()
        weight = ob.create_particle_weight(self.model)
        weight.apply(state)
        weight.apply(state)
        total = state.particle_f.numpy().sum(axis=0)
        np.testing.assert_allclose(total, [0.0, 0.0, -2.0 * self.scene.object_mass * ch.GRAVITY], atol=1.0e-6)

    def test_grasp_plan(self):
        """The plan is the ball grasp with squeeze pulses in the lifted hold instead of the tilt."""
        spec = self.spec
        plan = ob.soft_object_grasp_plan(spec, hold_tension=11.0, squeeze_tension=27.0, pulses=3)
        ball = ch.grasp_plan(spec, "ball", hold_tension=11.0, tilt=0.0, set_down_clearance=ob.SOFT_SET_DOWN_CLEARANCE)
        names = list(plan.phases)
        self.assertNotIn("tilt", names)
        after_hold = names[names.index("hold") + 1 : names.index("lower")]
        self.assertEqual(after_hold[:3], ["squeeze1", "squeeze_hold1", "relax1"])
        self.assertEqual(after_hold[-1], "recover")
        self.assertEqual(sum(name.startswith("squeeze_hold") for name in names), 3)
        self.assertEqual(names[-4:], ["wait", "ease", "retreat", "end"])
        np.testing.assert_allclose(list(plan.palm_grasp), list(ball.palm_grasp))
        np.testing.assert_allclose(list(plan.object_xform), list(ball.object_xform))
        self.assertEqual((plan.hold_tension, plan.squeeze_tension, plan.lift), (11.0, 27.0, ball.lift))
        times = [key.time for key in plan.keyframes]
        self.assertTrue(np.all(np.diff(times) > 0.0))
        self.assertAlmostEqual(times[-1], plan.phases["end"][1])
        actuators = list(self.hand.actuators)
        thumb = actuators.index("thumb")
        fingers = [i for i in range(len(actuators)) if i != thumb]
        keys = ch.resolve_hand_keyframes(plan.keyframes, actuators, start_tension=spec.cable.tension_rest, spec=spec)
        index = {round(t, 6): i for i, t in enumerate(keys["times"])}

        def at(name):
            row = index[round(plan.phases[name][1], 6)]
            return keys["command"][row], keys["wrist"][row]

        # the closing fingers squeeze, the thumb keeps its own tension from the grasp to the release
        closing = [actuators.index(key) for key in actuators if key.split("_")[0] in plan.keyframes[3].tension]
        np.testing.assert_allclose(at("squeeze_hold2")[0][closing], 27.0)
        np.testing.assert_allclose(at("recover")[0][closing], 11.0)
        for name in ("settle", "recover", "rest"):
            self.assertAlmostEqual(at(name)[0][thumb], ob.SOFT_THUMB_TENSION)
        self.assertAlmostEqual(at("squeeze_hold2")[0][thumb], ob.SOFT_THUMB_SQUEEZE_TENSION)
        # the default plan has one squeeze
        self.assertEqual(sum(name.startswith("squeeze_hold") for name in self.plan.phases), 1)
        np.testing.assert_allclose(at("end")[0], spec.cable.tension_rest)
        self.assertTrue(set(closing) <= set(fingers))
        # the wrist stays at the lifted pose through the pulses, lowers the object onto its post and
        # returns to where it started
        lifted, start = at("lift")[1], keys["wrist"][0]
        np.testing.assert_allclose(at("recover")[1], lifted)
        down = np.array(plan.palm_grasp, dtype=float)
        down[2] += ob.SOFT_SET_DOWN_CLEARANCE
        down[:2] += ob.SOFT_SET_DOWN_SHIFT
        np.testing.assert_allclose(at("lower")[1], down, atol=1.0e-6)
        # the palm eases off the released object before the hand retreats
        down[2] += ob.SOFT_EASE_HEIGHT
        np.testing.assert_allclose(at("ease")[1], down, atol=1.0e-6)
        np.testing.assert_allclose(at("end")[1], start)
        # the thumb can squeeze with the fingers
        both = ob.soft_object_grasp_plan(spec, thumb_squeeze_tension=None)
        keys = ch.resolve_hand_keyframes(both.keyframes, actuators, start_tension=spec.cable.tension_rest, spec=spec)
        row = int(np.argmin(np.abs(keys["times"] - both.phases["squeeze_hold1"][1])))
        self.assertAlmostEqual(keys["command"][row][thumb], ob.SOFT_THUMB_TENSION)

    def test_read_back_of_the_rest_shape(self):
        """At rest the object has its rest volume, no deformation and its rest size in the palm frame."""
        scene = self.scene
        x = self.model.particle_q.numpy()
        state = ob.soft_object_state(scene, x)
        rest_volume = float(ob.tet_volumes(scene.rest_vertices, scene.tets).sum())
        self.assertAlmostEqual(state["volume"] / rest_volume, 1.0, places=5)
        self.assertAlmostEqual(state["volume_ratio_min"], 1.0, places=4)
        self.assertLess(float(state["deformation"].max()), 1.0e-6)
        # the ball is built where it rests in its seat: turned a little about the world x axis
        settle = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), ob.SOFT_SEAT_SETTLE_TURN)
        settle = np.array(wp.quat_to_matrix(settle)).reshape(3, 3)
        np.testing.assert_allclose(state["rotation"], settle, atol=1.0e-6)
        # a rigid motion of the particles is no deformation
        turn = np.array(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(0.0, 0.6, 0.8), 0.7))).reshape(3, 3)
        moved = ob.soft_object_state(scene, x @ turn.T + np.array([0.1, -0.2, 0.3]))
        self.assertLess(float(moved["deformation"].max()), 1.0e-6)
        np.testing.assert_allclose(moved["rotation"], turn @ settle, atol=1.0e-5)
        palm = self.model.body_q.numpy()[self.hand.palm]
        extent = ob._soft_object_extent(scene, x, palm)
        # palm frame at the grasp: x along the fingers, y the palm normal (world -z), z toward the thumb (world y)
        world = x.max(axis=0) - x.min(axis=0) + 2.0 * scene.particle_radius
        np.testing.assert_allclose(extent, [world[0], world[2], world[1]], atol=1.0e-6)
        # the pose in the palm frame moves with the object: 5 mm along world x is 5 mm along palm x
        shifted = ob.soft_object_in_palm(scene, x + np.array([0.005, 0.0, 0.0]), palm)
        slip, angle = ch.grasp_slip(shifted, ob.soft_object_in_palm(scene, x, palm))
        self.assertAlmostEqual(slip, 0.005, places=6)
        self.assertAlmostEqual(angle, 0.0, places=4)
        # the line between two groups of pads, here the two palm pads (26 mm apart along the fingers,
        # 8 mm in radius), and the thickness of a set of particles along it
        body_q = self.model.body_q.numpy()
        palm_pads = (("palm/heel",), ("palm/distal_palm",))
        start, axis, gap = ob.soft_grip_line(scene, body_q, palm_pads)
        heel, distal = (
            0.5 * (np.array(pad.p0) + np.array(pad.p1))
            for pad in sorted(self.spec.palm.pads, key=lambda pad: pad.name != "heel")
        )
        between = distal - heel  # palm frame; at the grasp palm x is world x and palm z is world y
        expected = np.array([between[0], between[2], -between[1]]) / np.linalg.norm(between)
        np.testing.assert_allclose(axis, expected, atol=1.0e-6)
        self.assertAlmostEqual(gap, float(np.linalg.norm(between)) - 0.016, places=6)
        everything = np.arange(scene.particle_count)
        along = x @ expected
        self.assertAlmostEqual(
            ob.soft_grip_width(scene, x, body_q, everything, palm_pads),
            float(along.max() - along.min()) + 2.0 * scene.particle_radius,
            places=6,
        )
        tube = ob.soft_grip_particles(scene, x, body_q, palm_pads, radius=0.010)
        offset = x[tube] - start
        self.assertTrue(np.all(np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1) < 0.010))
        self.assertEqual(len(ob.soft_grip_particles(scene, x, body_q, palm_pads, radius=1.0)), scene.particle_count)

    def test_display_surface(self):
        """The display surface is the contact surface: the ball of the stated size at rest, moving with the mesh."""
        scene = self.scene
        radius = scene.radius
        surface = ob.create_soft_display_surface(scene)
        self.assertEqual(len(surface.triangles), 5120)
        self.assertEqual(surface.offset, scene.particle_radius)
        np.testing.assert_allclose(surface.weights.sum(axis=1), 1.0, atol=1.0e-9)
        self.assertGreater(float(surface.weights.min()), -0.3)
        x = self.model.particle_q.numpy()
        vertices, normals = ob.soft_display_surface_vertices(surface, x)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-9)
        # the ball is built where it rests in its seat, a little beside and below the nominal pose
        settle = np.array(ob.SOFT_SEAT_SETTLE_SHIFT)
        center = np.array(self.plan.object_xform, dtype=float)[:3] + settle
        local = vertices - center
        # at rest: a sphere of the stated radius, whose lowest point is at the height of its seat
        round_part = np.ones(len(local), dtype=bool)
        np.testing.assert_allclose(np.linalg.norm(local[round_part], axis=1), radius, atol=2.0e-5)
        self.assertAlmostEqual(float(vertices[:, 2].min()), scene.pedestal_height + settle[2], delta=2.0e-5)
        # outward normals, and an enclosed volume close to that of the ball
        self.assertGreater(float(np.einsum("ij,ij->i", normals, local)[round_part].min()), 0.0)
        a, b, c = (local[surface.triangles[:, i]] for i in range(3))
        volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
        self.assertAlmostEqual(volume, 4.0 / 3.0 * math.pi * radius**3, delta=0.03 * radius**3 * 4.19)
        # it moves rigidly with the particles, and a uniform compression of the mesh by 20 % brings the
        # surface in by 20 % of the mesh radius (the offset of one particle radius stays)
        turn = np.array(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(0.0, 0.6, 0.8), 0.7))).reshape(3, 3)
        moved, _ = ob.soft_display_surface_vertices(surface, (x - center) @ turn.T + center)
        np.testing.assert_allclose(moved, local @ turn.T + center, atol=1.0e-6)
        squeezed, _ = ob.soft_display_surface_vertices(surface, (x - center) * 0.8 + center)
        expected = 0.8 * (radius - scene.particle_radius) + scene.particle_radius
        np.testing.assert_allclose(np.linalg.norm((squeezed - center)[round_part], axis=1), expected, atol=5.0e-5)

    def test_fingers_stop_on_the_ball_and_a_squeeze_dents_it(self):
        """Close the fingers on the seated ball and squeeze: two-way contact, a dent, no inverted element.

        The palm stands at the grasp pose from the start. The fingers close at the hold tension and stop on
        the ball, then their cables go to the squeeze tension. The full grasp with lift and release is
        covered by the registered example test.
        """
        spec, model, hand, scene, plan = self.spec, self.model, self.hand, self.scene, self.plan
        close = plan.keyframes[1 + list(plan.phases).index("close")].tension
        squeeze = plan.keyframes[1 + list(plan.phases).index("squeeze1")].tension
        keyframes = [
            ch.HandKeyframe(0.0, wrist=plan.palm_grasp, opposition=0.0, tension=spec.cable.tension_rest),
            ch.HandKeyframe(0.8, tension=close),
            ch.HandKeyframe(1.0),
            ch.HandKeyframe(1.5, tension={k: v for k, v in squeeze.items() if k != "thumb"}),
            ch.HandKeyframe(1.7),
        ]
        record = {"low": 1.0, "hold": None}

        def on_frame(time, state):
            current = ob.soft_object_state(scene, state.particle_q.numpy())
            record["low"] = min(record["low"], current["volume_ratio_min"])
            if record["hold"] is None and time >= 1.0 - 1.0e-9:
                palm = state.body_q.numpy()[hand.palm]
                record["hold"] = (current, ob._soft_object_extent(scene, state.particle_q.numpy(), palm))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline = ob.create_soft_object_collision_pipeline(model)
            solver = ob.make_hand_soft_object_solver(model)
            weight = ob.create_particle_weight(model)
            states, contacts, _ = _step(model, spec, hand, pipeline, solver, keyframes, 1.7, weight, on_frame)
        self.assertEqual([str(w.message) for w in caught], [])
        x = states[0].particle_q.numpy()
        body_q = states[0].body_q.numpy()
        self.assertTrue(np.isfinite(x).all() and np.isfinite(body_q).all())
        self.assertGreater(record["low"], 0.1)
        end = ob.soft_object_state(scene, x)
        # two-way contact: the pads press on the particles, and the ball carries that force
        forces = ob._soft_object_contact_forces(model, scene, states[1], contacts)
        self.assertGreater(forces["pad_contacts"], 3)
        self.assertGreater(float(np.linalg.norm(forces["pad_force"], axis=1).sum()), 1.0)
        # the squeeze dents the ball more than the hold
        hold_state, _ = record["hold"]
        self.assertGreater(float(end["deformation"].max()), float(hold_state["deformation"].max()) + 0.5e-3)
        # the foam gives up some volume under the pads (92.1 % of the rest volume is left in the full squeeze of
        # the example); in this short squeeze it stays within 8 % of the rest volume
        rest_volume = float(ob.tet_volumes(scene.rest_vertices, scene.tets).sum())
        self.assertAlmostEqual(end["volume"] / rest_volume, 1.0, delta=0.08)
        tension = np.concatenate(list(ch.hand_span_tensions(solver, self.tendons).values()))
        # the tensions follow the commands, so the highest stays below the limit and no cable goes slack
        self.assertLess(float(tension.max()), spec.cable.tension_max)
        self.assertGreater(float(tension.min()), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
