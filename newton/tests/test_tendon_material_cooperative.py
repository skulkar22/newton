# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check full-warp ownership, mixed precision, and packed failure isolation."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.tendon_material_cooperative import (
    solve_tendon_material_nonlinear_component as project_cooperative,
)
from newton._src.solvers.tendon_material_nonlinear import (
    TendonMaterialNonlinearState,
    allocate_tendon_material_nonlinear_state,
    solve_tendon_material_nonlinear_component,
)
from newton.tests.test_tendon_material_nonlinear import _assert_certificate, _fixture, _launch, _project_exact
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel(enable_backward=False)
def _batch_cooperative(state: TendonMaterialNonlinearState, counts: wp.array[int]):
    row = wp.tid() // 32
    project_cooperative(state, counts[row], 1, row, 40, 40, 500.0, 10.0, 0.01, 0.003, 1.0e-6, 64, 1.0e-5, wp.tid() % 32)


@wp.kernel(enable_backward=False)
def _batch_scalar(state: TendonMaterialNonlinearState, counts: wp.array[int]):
    row = wp.tid()
    solve_tendon_material_nonlinear_component(
        state, counts[row], 1, row, 40, 40, 500.0, 10.0, 0.01, 0.003, 1.0e-6, 64, 1.0e-5
    )


def test_packed_warps_cold_warm_capture(test, device):
    """Isolate 1-32-span components, padding, and invalid rows across full warps."""
    if not device.is_cuda:
        test.skipTest("Cooperative projection requires CUDA")
    rng = np.random.default_rng(1414)
    counts = [1, 2, 3, 4, 5, 8, 16, 31, 32] * 3 + [0, 33, 2, 2, 2]
    rows = len(counts)
    shape = (rows, 40)
    values = {
        "reference": np.zeros(shape, np.float32),
        "length": np.ones(shape, np.float32),
        "damping": np.zeros(shape, np.float32),
        "cap": np.ones(shape, np.float32),
    }
    for row, n in enumerate(counts[:27]):
        lengths = rng.uniform(0.003, 0.1, n)
        caps = rng.uniform(1.02, 1.1, n - 1)
        signs = np.ones(n - 1) if row < 9 else -np.ones(n - 1) if row < 18 else rng.choice([-1, 1], n - 1)
        forces = np.full(n, 10.0)
        for i in range(n - 1):
            forces[i + 1] = forces[i] / caps[i] ** signs[i]
        damping = forces * rng.uniform(-0.05, 0.05, n)
        lower, upper = np.zeros(n), np.ones(n)
        # Independent scalar inversion constructs known capstan-face solutions.
        for _ in range(64):
            strain = (lower + upper) * 0.5
            force = 500.0 * (1.0 + 4.5 * (1.0 + np.tanh((strain - 0.01) / 0.003))) * strain
            lower = np.where(force < forces - damping, strain, lower)
            upper = np.where(force < forces - damping, upper, strain)
        strain = (lower + upper) * 0.5
        extension = lengths * strain / (1.0 + strain)
        flow = signs * rng.uniform(0.01, 0.03, n - 1) * min(extension)
        reference = extension + np.diff(np.r_[0.0, flow, 0.0])
        for name, value in (("length", lengths), ("cap", caps), ("damping", damping), ("reference", reference)):
            values[name][row, : len(value)] = value
    values["reference"][-3, 0] = np.nan
    values["cap"][-2, 0] = 0.5
    values["reference"][-1, 0] = 2.0
    states = [allocate_tendon_material_nonlinear_state(rows * 40, rows * 40, rows, device=device) for _ in range(2)]
    for state in states:
        for name, value in values.items():
            getattr(state, name).assign(value.ravel())
        state.output.fill_(-123.0)
    count_array = wp.array(counts, dtype=int, device=device)

    def launch():
        wp.launch(_batch_cooperative, dim=rows * 32, block_dim=256, inputs=[states[0], count_array], device=device)
        wp.launch(_batch_scalar, dim=rows, inputs=[states[1], count_array], device=device)

    def verify():
        cooperative, scalar = states
        status = cooperative.status.numpy()
        np.testing.assert_array_equal(status, scalar.status.numpy())
        test.assertTrue(np.all(status[:27] > 0))
        test.assertTrue(np.all(status[27:] < 0))
        out = cooperative.output.numpy().reshape(shape)
        exact = cooperative.corrected.numpy().reshape(shape)
        reference = scalar.corrected.numpy().reshape(shape)
        for row, n in enumerate(counts[:27]):
            np.testing.assert_allclose(exact[row, :n], reference[row, :n], rtol=1.0e-10, atol=1.0e-15)
            test.assertTrue(np.all(out[row, n:] == -123.0))
            e = exact[row, :n]
            l = values["length"][row, :n].astype(float)
            b = values["damping"][row, :n].astype(float)
            ref = values["reference"][row, :n].astype(float)
            cap = values["cap"][row, : n - 1].astype(float)
            strain = np.maximum(e, 0.0) / np.maximum(l - e, 1.0e-8)
            ea = 500.0 * (1.0 + 4.5 * (1.0 + np.tanh((strain - float(np.float32(0.01))) / float(np.float32(0.003)))))
            force = np.maximum(ea * strain + b, 0.0)
            tol = 1.0e-5 * max(force.max(), 1.0e-20)
            test.assertLessEqual(abs((e - ref).sum()), 1.0e-5 * (np.abs(e).sum() + np.abs(ref).sum()))
            test.assertTrue(np.all(l - out[row, :n] >= float(np.float32(1.0e-6))))
            plus, minus = cap * force[1:] - force[:-1], cap * force[:-1] - force[1:]
            test.assertTrue(np.all(plus >= -tol))
            test.assertTrue(np.all(minus >= -tol))
            flow = np.cumsum(ref - e)[:-1]
            flow_tol = tol * np.min((l - e) / ea)
            test.assertTrue(np.all(np.abs(plus[flow > flow_tol]) <= tol))
            test.assertTrue(np.all(np.abs(minus[flow < -flow_tol]) <= tol))
        test.assertTrue(np.all(out[27:] == -123.0))
        for name, value in values.items():
            np.testing.assert_array_equal(getattr(cooperative, name).numpy().reshape(shape), value)

    launch()
    verify()
    values["reference"][:27] *= np.float32(0.93)
    for state in states:
        state.reference.assign(values["reference"].ravel())
    launch()
    verify()
    with wp.ScopedCapture(device=device) as capture:
        launch()
    for _ in range(3):
        wp.capture_launch(capture.graph)
    verify()


def test_cancelling_force_and_slack_reference(test, device):
    """Converge a tiny positive-force/slack mixture that rejects FP32 log-force state."""
    reference = [3.24100796e-7, 8.40337748e-7, 1.43555314e-6, -0.0054484377615, 1.88501517e-8]
    lengths = [0.015019939281, 0.016829738393, 0.029593167827, 0.027204332873, 0.001038029790]
    damping = [0.004218131304, -0.017151853070, -0.017603445798, 0.006672004238, -0.001149063697]
    caps = [1.020232439, 1.208043933, 1.025783062, 1.153415084]
    state, inputs, count = _fixture(device, reference, lengths, caps, damping, ratio=10.0)
    inputs[2], inputs[4], inputs[5], inputs[6], inputs[8] = 500.0, 0.01, 0.003, 1.0e-6, 1.0e-5
    for _ in range(2):
        _launch(_project_exact, dim=count, inputs=inputs, device=device)
        _assert_certificate(
            test,
            state,
            5,
            ea_low=500.0,
            ratio=10.0,
            transition_strain=0.01,
            transition_width=0.003,
            tolerance=1.0e-5,
            allow_slack=True,
            exact=True,
        )


class TestTendonMaterialCooperative(unittest.TestCase):
    """Exercise the native mixed-precision projection."""


for _test in (test_packed_warps_cold_warm_capture, test_cancelling_force_and_slack_reference):
    add_function_test(TestTendonMaterialCooperative, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
