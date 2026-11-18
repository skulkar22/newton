# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate appending rigid and elastic components to declared articulations."""

import unittest

import numpy as np
import warp as wp

import newton


class TestArticulationExtension(unittest.TestCase):
    def make_chain(self):
        """Declare a root articulation and leave two following joints unowned."""
        builder = newton.ModelBuilder()
        bodies = [builder.add_link(mass=1.0) for _ in range(3)]
        joints = [builder.add_joint_fixed(-1, bodies[0])]
        builder.add_articulation(joints, label="robot")
        joints.extend(builder.add_joint_fixed(bodies[i - 1], bodies[i]) for i in (1, 2))
        return builder, bodies, joints

    def assert_rejected(self, builder, articulation, joints, message):
        """Check that rejected extensions leave articulation metadata intact."""
        ends, owners = list(builder.articulation_end), list(builder.joint_articulation)
        with self.assertRaisesRegex(ValueError, message):
            builder.extend_articulation(articulation, joints)
        self.assertEqual(list(builder.articulation_end), ends)
        self.assertEqual(list(builder.joint_articulation), owners)

    def test_append(self):
        """Extend in successive calls without changing articulation identity or labels."""
        builder, _bodies, joints = self.make_chain()
        builder.extend_articulation(0, [joints[1]])
        builder.extend_articulation(0, [joints[2]])
        self.assertEqual(builder.articulation_count, 1)
        self.assertEqual(builder.articulation_start, [0])
        self.assertEqual(builder.articulation_end, [3])
        self.assertEqual(builder.articulation_label, ["robot"])
        self.assertEqual(builder.joint_articulation, [0, 0, 0])

    def test_replicate_ranges(self):
        """Preserve extended rigid ranges through batched replication."""
        builder, _bodies, joints = self.make_chain()
        builder.extend_articulation(0, joints[1:])
        copies = newton.ModelBuilder()
        copies.replicate(builder, 3)
        model = copies.finalize(device="cpu")
        np.testing.assert_array_equal(model.articulation_start.numpy(), [0, 3, 6, 9])
        np.testing.assert_array_equal(model.articulation_end.numpy(), [3, 6, 9])
        np.testing.assert_array_equal(model.joint_articulation.numpy(), np.repeat([0, 1, 2], 3))

    def test_invalid_ranges(self):
        """Reject invalid articulation IDs, joint indices, gaps, duplicates and reordering."""
        for articulation, indices, message in (
            (-1, [1], "Articulation index"),
            (1, [1], "Articulation index"),
            (0, [], "no joints"),
            (0, [-1], "Joint index"),
            (0, [3], "Joint index"),
            (0, [2], "contiguous"),
            (0, [1, 1], "contiguous"),
            (0, [2, 1], "contiguous"),
            (0, [0, 1], "already belongs"),
        ):
            with self.subTest(articulation=articulation, joints=indices):
                builder, _bodies, _joints = self.make_chain()
                self.assert_rejected(builder, articulation, indices, message)

    def test_owned_joints(self):
        """Never absorb a neighboring articulation's joint span."""
        builder, _bodies, joints = self.make_chain()
        builder.add_articulation(joints[1:], label="other")
        self.assert_rejected(builder, 0, joints[1:], "already belongs")

    def test_world_mismatch(self):
        """Require appended joints to belong to the articulation's world."""
        builder = newton.ModelBuilder()
        builder.begin_world()
        body = builder.add_link(mass=1.0)
        builder.add_articulation([builder.add_joint_fixed(-1, body)])
        builder.end_world()
        builder.begin_world()
        body = builder.add_link(mass=1.0)
        pending = builder.add_joint_fixed(-1, body)
        self.assert_rejected(builder, 0, [pending], "world")
        builder.end_world()

    def test_kinematic_child(self):
        """Preserve the restriction that only world-rooted bodies may be kinematic."""
        builder, bodies, joints = self.make_chain()
        builder.body_flags[bodies[1]] = newton.BodyFlags.KINEMATIC
        self.assert_rejected(builder, 0, joints[1:], "kinematic")

    def test_cross_articulation_parent(self):
        """Reject appended joints whose parent belongs exclusively to another articulation."""
        builder = newton.ModelBuilder()
        first, second, tool = [builder.add_link(mass=1.0) for _ in range(3)]
        builder.add_articulation([builder.add_joint_fixed(-1, first)], label="first")
        builder.add_articulation([builder.add_joint_fixed(-1, second)], label="second")
        pending = builder.add_joint_fixed(first, tool)
        self.assert_rejected(builder, 1, [pending], "Articulations cannot be connected")

    def test_loop_opt_in(self):
        """Require explicit loop support for repeated children and cyclic physical joints."""
        builder, bodies, joints = self.make_chain()
        builder.extend_articulation(0, joints[1:])
        closing = builder.add_joint_fixed(bodies[-1], bodies[0])
        self.assert_rejected(builder, 0, [closing], "allow_closed_loops")
        builder.extend_articulation(0, [closing], allow_closed_loops=True)
        self.assertEqual(builder.articulation_end, [closing + 1])
        self.assertEqual(builder.joint_articulation[closing], 0)

        builder, bodies, _joints = self.make_chain()
        builder.joint_parent[1] = bodies[2]
        self.assert_rejected(builder, 0, [1, 2], "allow_closed_loops")

    def test_imported_elastic_component(self):
        """Attach a modal component after URDF import and preserve its state through replication."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_urdf(
            '<robot name="robot"><link name="base"><inertial><mass value="1"/>'
            '<inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>'
            "</inertial></link></robot>",
            floating=False,
        )
        self.assertEqual(builder.articulation_count, 1)
        start = builder.joint_count
        tool = builder.add_link_elastic(
            mass=0.1,
            inertia=wp.mat33(np.eye(3) * 0.01),
            mode_count=1,
            mode_mass=[1.0],
            mode_stiffness=[10.0],
            mode_q=[0.03],
            mode_shape_fn=lambda position: np.array([[position[0], 0.0, 0.0]]),
        )
        builder.add_joint_fixed(0, tool)
        # ELASTIC owner joints store state but do not constrain the body twice.
        builder.extend_articulation(0, list(range(start, builder.joint_count)))
        model = builder.finalize(device="cpu")
        np.testing.assert_array_equal(model.joint_articulation.numpy(), [0, 0, 0])
        state = model.state()
        self.assertAlmostEqual(float(state.joint_q.numpy()[model.joint_q_start.numpy()[start] + 7]), 0.03)

        copies = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _ in range(3):
            copies.add_world(builder)
        copies.color()
        replicated = copies.finalize(device="cpu")
        np.testing.assert_array_equal(replicated.articulation_start.numpy(), [0, 3, 6, 9])
        np.testing.assert_array_equal(replicated.articulation_end.numpy(), [3, 6, 9])
        np.testing.assert_array_equal(replicated.joint_articulation.numpy(), np.repeat([0, 1, 2], 3))
        for backend in ("schur", "coupled"):
            solver = newton.solvers.SolverVBD(
                replicated,
                iterations=2,
                rigid_articulation_solve="block_sparse_joints",
                rigid_articulation_elastic_solve=backend,
            )
            initial, output = replicated.state(), replicated.state()
            solver.step(initial, output, replicated.control(), None, 0.001)
            self.assertTrue(np.isfinite(output.body_q.numpy()).all())
            modal = output.joint_q.numpy()[replicated.joint_q_start.numpy()[[1, 4, 7]] + 7]
            np.testing.assert_allclose(modal, modal[0], atol=1.0e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
