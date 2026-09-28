# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check supported overlapping routes and latched invalid accepted wraps."""

import unittest

import warp as wp

import newton
from newton.tests.test_tendon_capstan import build_interpenetrating_neighbor_route, build_oriented_dynamic_route
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_same_winding_overlap_activation(test, device):
    """Equal-radius external tangent remains x=-radius despite circle overlap."""
    for solver_type in (newton.solvers.SolverVBD, newton.solvers.SolverXPBD):
        model, upper, link = build_interpenetrating_neighbor_route(device)
        model.body_color_groups = [wp.array([i], dtype=int, device=device) for i in range(model.body_count)]
        solver = solver_type(model, iterations=1)
        poses = model.body_q.numpy()
        poses[upper, 2] = -0.25
        candidate = int(model.tendon_link_body.numpy()[link])
        poses[candidate, 2] = -0.375
        # The external tangent is x=-0.3. Candidate radius is 0.1.
        for initial in (False, True):
            for x, expected in ((-0.25, True), (0.0, False)):
                active = solver.tendon_link_active.numpy()
                active[link] = initial
                solver.tendon_link_active.assign(active)
                poses[candidate, 0] = x
                model.body_q.assign(poses)
                solver._update_tendon_link_active(model, model.body_q)
                test.assertEqual(bool(solver.tendon_link_active.numpy()[link]), expected)


def test_initial_invalid_route_check(test, device):
    """An invalid initial route cannot become a successful direct solve."""
    for solver_type in (newton.solvers.SolverVBD, newton.solvers.SolverXPBD):
        model, _, link = build_oriented_dynamic_route(1, device, dynamic=False)
        model.body_color_groups = [wp.array([i], dtype=int, device=device) for i in range(model.body_count)]
        model.tendon_seg_compliance.fill_(0.01)
        solver = solver_type(model, iterations=1, tendon_material_direct=True)
        with test.assertRaisesRegex(RuntimeError, f"Unsupported accepted tendon route: tendon 0, ROLLING link {link}"):
            solver.check_tendon_material()


def test_accepted_invalid_route_check(test, device):
    """Intermediate wraps do not latch; accepted wraps persist across graph replay."""
    for capture in (False, True) if device.is_cuda else (False,):
        model, body, link = build_oriented_dynamic_route(1, device, dynamic=False)
        model.body_color_groups = [wp.array([i], dtype=int, device=device) for i in range(model.body_count)]
        model.tendon_seg_compliance.fill_(0.01)
        poses = model.body_q.numpy()
        poses[body, 0] = 0.0
        model.body_q.assign(poses)
        solver = newton.solvers.SolverVBD(
            model, iterations=1, tendon_material_direct=True, tendon_alm=True, tendon_alm_per_segment=True
        )
        state = model.state()
        solver.body_q_prev.assign(state.body_q)
        solver._snapshot_tendon_step_state()
        solver._update_tendon_routing(state, 1.0 / 1200.0, False)
        poses[body, 0] = 0.15
        state.body_q.assign(poses)
        solver._update_tendon_routing(state, 1.0 / 1200.0, False)
        solver.check_tendon_material()
        if capture:
            with wp.ScopedCapture(device=device) as captured:
                solver._finalize_tendon_routing(state, 1.0 / 1200.0)
            wp.capture_launch(captured.graph)
        else:
            solver._finalize_tendon_routing(state, 1.0 / 1200.0)
        with test.assertRaisesRegex(RuntimeError, f"Unsupported accepted tendon route: tendon 0, ROLLING link {link}"):
            solver.check_tendon_material()
        poses[body, 0] = 0.0
        state.body_q.assign(poses)
        solver._finalize_tendon_routing(state, 1.0 / 1200.0)
        with test.assertRaisesRegex(RuntimeError, "Unsupported accepted tendon route"):
            solver.check_tendon_material()


class TestTendonRouteAcceptance(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_same_winding_overlap_activation,
        test_initial_invalid_route_check,
        test_accepted_invalid_route_check,
    ):
        # The invalid accepted-route case deliberately emits the solver diagnostic;
        # its RuntimeError and latched link identity are asserted in the test.
        add_function_test(
            TestTendonRouteAcceptance,
            _test.__name__,
            _test,
            devices=[_device],
            check_output=_test is not test_accepted_invalid_route_check,
        )

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
