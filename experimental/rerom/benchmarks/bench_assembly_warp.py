# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark WarpTetAssembler (CPU, CUDA; gather and atomic scatter) against the numpy assembler.

Usage (from the Newton checkout, with its environment)::

    python experimental/rerom/benchmarks/bench_assembly_warp.py --out results.json

The numpy reference and ``make_beam_mesh`` come from ``adaptive_rom.fe_nonlinear`` in
``ADAPTIVE_ROM_ROOT`` (default: the exp-re-rom workspace four levels up).

Timed operations (median and min wall time per call over the repeats, after warm-up; every
Warp call ends with a host synchronisation because the energy is returned as a float):

* ``numpy``: ``NonlinearAssembler.evaluate`` (force + tangent, SciPy CSR).
* ``kernel``: ``evaluate`` with device-resident float64 positions, force left on the device and the
  internal BSR tangent (no copies) -- the assembly cost proper.
* ``host_io``: ``evaluate`` from numpy positions to a numpy force and a BSR copy (H2D + D2H of
  vectors, one device BSR copy).
* ``force_only``: device-resident ``evaluate(need_tangent=False)``.
* ``to_csc``: device-to-host CSC conversion of the tangent (values copy + cached permutation).
* ``rotations``: ``element_rotations`` of all elements (device-resident input and output, followed
  by a device synchronisation).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import warp as wp

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PKG_ROOT)
from rerom.assembly_warp import WarpTetAssembler  # noqa: E402

wp.config.quiet = True


def _timeit(fn, repeats, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {"median_ms": 1e3 * statistics.median(times), "min_ms": 1e3 * min(times), "repeats": repeats}


def _state(mesh, seed=0):
    rng = np.random.default_rng(seed)
    X = mesh.nodes
    x = X.copy()
    x[:, 2] += 0.2 * X[:, 0] ** 2
    x += 1e-4 * np.ptp(X, axis=0).max() * rng.standard_normal(x.shape)
    return x.reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="JSON output path")
    parser.add_argument("--material", default="stvk")
    parser.add_argument("--cpu-repeats", type=int, default=10)
    parser.add_argument("--gpu-repeats", type=int, default=100)
    args = parser.parse_args()

    root = os.environ.get("ADAPTIVE_ROM_ROOT", os.path.abspath(os.path.join(_PKG_ROOT, "..", "..", "..")))
    sys.path.append(root)
    fe = importlib.import_module("adaptive_rom.fe_nonlinear")
    meshes = {
        # M1 beam (adaptive_rom/experiments/cases.py "beam"): 2,025 nodes, 7,680 tets
        "beam_M1": fe.make_beam_mesh(1.0, 0.025, 0.025, 80, 4, 4, density=1100.0, young=3.0e7, poisson=0.35),
        # thin panel-like plate: 7,503 nodes (22,509 DOF), 28,800 tets
        "panel_23k": fe.make_beam_mesh(0.6, 0.4, 0.004, 60, 40, 2, density=2700.0, young=7.0e10, poisson=0.33),
    }
    devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
    out = {
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "warp": wp.__version__,
            "cpu": platform.processor() or platform.machine(),
            "cuda_devices": [wp.get_device(d).name for d in devices if d != "cpu"],
            "material": args.material,
        },
        "cases": {},
    }
    for name, mesh in meshes.items():
        x = _state(mesh)
        case = {"n_nodes": mesh.num_nodes, "n_dof": 3 * mesh.num_nodes, "n_tets": mesh.num_tets}
        ref = fe.NonlinearAssembler(mesh, args.material)
        case["numpy"] = _timeit(lambda ref=ref, x=x: ref.evaluate(x), args.cpu_repeats)
        _, f0, K0 = ref.evaluate(x)
        print(f"{name}: {case['n_dof']} DOF, {case['n_tets']} tets; numpy evaluate {case['numpy']['median_ms']:.1f} ms")
        for dev in devices:
            reps = args.cpu_repeats if dev == "cpu" else args.gpu_repeats
            for scatter in ("gather", "atomic"):
                t0 = time.perf_counter()
                asm = WarpTetAssembler.from_mesh(mesh, args.material, dev, scatter=scatter)
                setup_ms = 1e3 * (time.perf_counter() - t0)
                x_dev = wp.array(x.reshape(-1, 3), dtype=wp.vec3d, device=dev)
                key = f"{dev}/{scatter}"
                rec = {"setup_ms": setup_ms, "nnz_blocks": asm.nnz_blocks}
                rec["kernel"] = _timeit(
                    lambda a=asm, xd=x_dev: a.evaluate(xd, device_output=True, copy_tangent=False), reps
                )
                rec["host_io"] = _timeit(lambda a=asm, x=x: a.evaluate(x), reps)
                rec["force_only"] = _timeit(
                    lambda a=asm, xd=x_dev: a.evaluate(xd, need_tangent=False, device_output=True), reps
                )
                _, f, K = asm.evaluate(x_dev, copy_tangent=False)
                asm.to_scipy_csc(K)  # build the cached permutation outside the timing
                rec["to_csc"] = _timeit(lambda a=asm, K=K: a.to_scipy_csc(K), reps)
                rec["rotations"] = _timeit(
                    lambda a=asm, xd=x_dev, d=dev: (
                        a.element_rotations(xd, device_output=True),
                        wp.synchronize_device(d),
                    ),
                    reps,
                )
                Kc = asm.to_scipy_csc(K)
                rec["rel_err_K_vs_numpy"] = float(abs(Kc - K0).max() / abs(K0).max())
                rec["rel_err_f_vs_numpy"] = float(np.abs(f - f0).max() / np.abs(f0).max())
                case[key] = rec
                print(
                    f"  {key:14s} kernel {rec['kernel']['median_ms']:8.3f} ms  host_io {rec['host_io']['median_ms']:8.3f}"
                    f"  force {rec['force_only']['median_ms']:7.3f}  to_csc {rec['to_csc']['median_ms']:7.3f}"
                    f"  rot {rec['rotations']['median_ms']:7.3f}  setup {setup_ms:7.1f}"
                    f"  errK {rec['rel_err_K_vs_numpy']:.1e}"
                )
        out["cases"][name] = case
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
