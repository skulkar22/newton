# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare cooperative ALM projection with its unchanged scalar equations."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.tendon_alm_material import (
    TendonALMMaterialState,
    project_alm_material,
    project_alm_material_cooperative,
)
from newton._src.solvers.tendon_material_state import TendonMaterialState
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel(enable_backward=False)
def _scalar(state: TendonMaterialState, counts: wp.array[int], stride: int):
    component = wp.tid()
    project_alm_material(
        state.alm,
        state.ids,
        state.initial,
        state.compliance,
        state.cap,
        state.upper,
        state.output,
        state.status,
        state.faces,
        state.valid,
        state.pieces,
        component * stride,
        counts[component],
        state.ea_low,
        state.ea_ratio,
        state.transition_strain,
        state.transition_width,
    )


@wp.kernel(enable_backward=False)
def _cooperative(state: TendonMaterialState, counts: wp.array[int], stride: int):
    component = wp.tid() // 32
    lane = wp.tid() % 32
    project_alm_material_cooperative(
        state.alm,
        state.ids,
        state.initial,
        state.compliance,
        state.cap,
        state.upper,
        state.output,
        state.status,
        state.faces,
        state.valid,
        state.pieces,
        component * stride,
        counts[component],
        state.ea_low,
        state.ea_ratio,
        state.transition_strain,
        state.transition_width,
        lane,
    )


def _workspace(device, counts, stride, *, nonlinear):
    rng = np.random.default_rng(1709)
    size = len(counts) * stride
    state = TendonMaterialState()
    state.alm = TendonALMMaterialState()
    state.alm.dt = 1.0 / 1200.0
    state.ea_low = 500.0 if nonlinear else 0.0
    state.ea_ratio = 10.0
    state.transition_strain = 0.01
    state.transition_width = 0.003
    reference = np.zeros(size, np.float32)
    length = rng.uniform(0.01, 0.1, size).astype(np.float32)
    multiplier = rng.uniform(0.0, 30.0, size).astype(np.float32)
    damping = rng.choice([0.0, 5.0, 50.0], size).astype(np.float32)
    damping_force = rng.uniform(-10.0, 10.0, size).astype(np.float32)
    penalty = rng.choice([0.0, 1.0e4, 1.0e6], size).astype(np.float32)
    ids = np.arange(size, dtype=np.int32)
    caps = rng.uniform(1.0, 2.5, size).astype(np.float32)
    for component, count in enumerate(counts):
        row = component * stride
        span = slice(row, row + count)
        # Reverse physical segment IDs: packed component rows are not segment IDs.
        ids[span] = ids[span][::-1]
        if component % 4 == 0:
            reference[span] = -0.001
            multiplier[span] = 0.0
            damping_force[span] = 0.0
        else:
            reference[span] = rng.uniform(-0.0001, 0.0003, count)
        if component % 7 == 0 and count > 1:
            # Nonphysical cap: failure must agree without leaking across warps.
            caps[row] = 0.5
    for name in ("initial", "compliance", "upper", "output"):
        setattr(state, name, wp.zeros(size, dtype=float, device=device))
    state.ids = wp.array(ids, dtype=int, device=device)
    state.cap = wp.array(caps, dtype=float, device=device)
    for name in ("status", "faces", "valid", "pieces"):
        setattr(state, name, wp.zeros(size, dtype=int, device=device))
    for name in ("extension", "offset", "force"):
        setattr(state.alm, name, wp.zeros(size, dtype=float, device=device))
    state.alm.iterations = wp.zeros(size, dtype=int, device=device)
    for name, data in (
        ("reference", reference),
        ("length", length),
        ("multiplier", multiplier),
        ("damping", damping),
        ("damping_force", damping_force),
        ("penalty", penalty),
    ):
        setattr(state.alm, name, wp.array(data, dtype=float, device=device))
    state.alm.physical_compliance = wp.full(size, 5.0e-6, dtype=float, device=device)
    return state


def test_cooperative_alm_matches_scalar(test, device, capture=False):
    """Match scalar ALM results exactly for mixed components, laws, and failures."""
    if not device.is_cuda:
        test.skipTest("The cooperative ALM projection requires CUDA")
    counts = np.array([1, 2, 3, 5, 9, 15, 17, 31, 32] * 3, np.int32)
    stride = 37
    for nonlinear in (False, True):
        reference = _workspace(device, counts, stride, nonlinear=nonlinear)
        result = _workspace(device, counts, stride, nonlinear=nonlinear)
        device_counts = wp.array(counts, dtype=int, device=device)
        # Replay twice to cover cold and cached active faces, including mixed failures.
        for _ in range(2):
            wp.launch(_scalar, len(counts), inputs=[reference, device_counts, stride], device=device)
            if capture:
                with wp.ScopedCapture(device=device) as graph:
                    wp.launch(
                        _cooperative,
                        len(counts) * 32,
                        inputs=[result, device_counts, stride],
                        device=device,
                        block_dim=32,
                    )
                wp.capture_launch(graph.graph)
            else:
                wp.launch(
                    _cooperative, len(counts) * 32, inputs=[result, device_counts, stride], device=device, block_dim=32
                )
            for name in ("status", "faces", "valid", "pieces", "output"):
                np.testing.assert_array_equal(
                    getattr(result, name).numpy(),
                    getattr(reference, name).numpy(),
                    err_msg=f"{name}, nonlinear={nonlinear}",
                )
            for name in ("force", "penalty", "iterations"):
                np.testing.assert_array_equal(
                    getattr(result.alm, name).numpy(),
                    getattr(reference.alm, name).numpy(),
                    err_msg=f"ALM {name}, nonlinear={nonlinear}",
                )
        test.assertGreater(np.count_nonzero(result.status.numpy()[::stride] > 0), len(counts) // 2)


def test_cooperative_alm_graph(test, device):
    """Preserve the same projection and failure results during graph replay."""
    test_cooperative_alm_matches_scalar(test, device, capture=True)


class TestTendonALMCooperative(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(
        TestTendonALMCooperative, "test_matches_scalar", test_cooperative_alm_matches_scalar, devices=[_device]
    )
    add_function_test(TestTendonALMCooperative, "test_graph", test_cooperative_alm_graph, devices=[_device])


if __name__ == "__main__":
    unittest.main()
