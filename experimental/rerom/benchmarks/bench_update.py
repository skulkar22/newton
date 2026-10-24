# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark one re-ROM update (and the per-step controller cost) before/after the device-side update.

Usage (from the Newton checkout, with its environment)::

    python experimental/rerom/benchmarks/bench_update.py --device cpu --out bench_update_cpu.json
    python experimental/rerom/benchmarks/bench_update.py --device cuda:0 --out bench_update_cuda.json

Model: a cantilever clamped at its root face (``ModalGeneratorFEM`` with the root nodes fixed, 12 modes,
held by a fixed joint at the root centre, gravity), deformed to the linear static gravity deflection scaled
to a 5 % L tip deflection before every update. Meshes: the M1 beam ``make_beam_mesh(1, 0.025, 0.025, 80,
4, 4)`` (6,075 DOF) and the panel-like plate ``make_beam_mesh(0.6, 0.4, 0.004, 60, 40, 2)`` (22,509 DOF).

Configurations:

* ``before``: legacy host path (``UpdatePath.HOST``), separate ARPACK/Ritz factorizations, numpy assembler.
* ``warp_asm``: the same with the Warp tangent assembler on the model device (WP-D.1 alone).
* ``after``: device-side update (``UpdatePath.DEVICE``), shared factorization, Warp assembler.

Per update: median wall time of ``relinearize`` (followed by a device synchronization) and of its split
(``info["timings"]``), and the host round trips (:func:`rerom.update_kernels.count_host_transfers`).
Per step (no update): ``apply_prestress`` + ``poll`` (velocity history and rotation trigger, 2 deg).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import warnings

import numpy as np
import warp as wp

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PKG_ROOT)
import rerom  # noqa: E402
from rerom.assembly_warp import WarpTetAssembler  # noqa: E402
from rerom.update_kernels import count_host_transfers  # noqa: E402

import newton  # noqa: E402

wp.config.quiet = True

MESHES = {
    "beam_M1": (1.0, 0.025, 0.025, 80, 4, 4),
    "panel_22k": (0.6, 0.4, 0.004, 60, 40, 2),
}
CONFIGS = {
    "before": {"path": rerom.UpdatePath.HOST, "shared": False, "warp": False},
    "warp_asm": {"path": rerom.UpdatePath.HOST, "shared": False, "warp": True},
    "after": {"path": rerom.UpdatePath.DEVICE, "shared": True, "warp": True},
}
SPLIT = ("assembly", "factorization", "eigensolve", "projection", "integrals", "rewrite", "total")


def build(mesh_key, device, mode_count=12):
    mesh = rerom.make_beam_mesh(*MESHES[mesh_key], density=1000.0, young=1.0e9, poisson=0.3)
    asm = rerom.NumpyTetAssembler(mesh, "stvk")
    root = mesh.node_sets["root"]
    gen = newton.ModalGeneratorFEM(
        mesh.nodes,
        asm.mass_matrix,
        asm.linear_stiffness(),
        mode_count=mode_count,
        damping_ratio=0.02,
        fixed_node_indices=root,
    )
    basis = gen.build()
    anchor = mesh.nodes[root].mean(axis=0)
    return mesh, asm, gen, basis, anchor


def make_model(basis, anchor, device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    body = builder.add_body_elastic(xform=wp.transform_identity(), modal_basis=basis, lock_inertia=True)
    builder.add_joint_fixed(
        parent=-1,
        child=body,
        parent_xform=wp.transform(wp.vec3(*anchor), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(*anchor), wp.quat_identity()),
    )
    builder.color()
    return builder.finalize(device=device), body


def static_q(gen, asm, mesh, m):
    """Linear static modal gravity deflection scaled to a 5 % L tip deflection."""
    Phi = gen.modal_matrix[:, :m]
    f = asm.mass_matrix @ np.tile([0.0, 0.0, -9.81], mesh.nodes.shape[0])
    k = np.einsum("im,im->m", Phi, asm.linear_stiffness() @ Phi)
    q = (Phi.T @ f) / k
    tip = float(np.max(np.abs((Phi @ q).reshape(-1, 3)[:, 2])))
    return q * (0.05 * float(np.ptp(mesh.nodes[:, 0])) / tip)


def bench_config(mesh_key, device, cfg_key, eigensolver, repeats, warmup, prebuilt):
    mesh, asm, gen, basis, anchor = prebuilt
    cfg = CONFIGS[cfg_key]
    model, body = make_model(basis, anchor, device)
    solver = newton.solvers.SolverVBD(model, iterations=4)
    a = WarpTetAssembler.from_mesh(mesh, device=device) if cfg["warp"] else asm
    ctrl = rerom.ReROMController(
        model,
        solver,
        body,
        a,
        rerom.FreeBodyAdapter(gen.modal_matrix, mesh.nodes, fixed_nodes=mesh.node_sets["root"]),
        mode_count=12,
        basis=rerom.BasisOptions(n_ritz=2, shared_factorization=cfg["shared"]),
        trigger=rerom.RotationTrigger(theta_tol=np.radians(2.0)),
        eigensolver=eigensolver,
        policy=rerom.UpdatePolicy(path=cfg["path"]),
    )
    state, control = model.state(), model.control()
    q = static_q(gen, asm, mesh, 12)
    jq0 = state.joint_q.numpy().copy()
    jq0[ctrl.q_start + 7 : ctrl.q_start + 7 + 12] = q

    def deform():
        state.joint_q.assign(jq0)
        state.joint_qd.zero_()

    rows, totals, transfers = [], [], None
    for k in range(warmup + repeats):
        deform()
        wp.synchronize_device(device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with count_host_transfers() as n:
                t0 = time.perf_counter()
                info = ctrl.relinearize(state, control)
                wp.synchronize_device(device)
                t1 = time.perf_counter()
        if k >= warmup:
            rows.append(info["timings"])
            totals.append(t1 - t0)
            transfers = dict(n)
    split = {key: 1e3 * statistics.median([r.get(key, 0.0) for r in rows]) for key in SPLIT}
    split["wall_synced"] = 1e3 * statistics.median(totals)

    # per step: prestress + poll (no update: the trigger threshold is not reached at q = 0)
    deform()
    ctrl.relinearize(state, control)
    ctrl.steps_since_update = 1
    for _ in range(3):
        ctrl.apply_prestress(control)
        ctrl.poll(state, 1.0e-3)
    wp.synchronize_device(device)
    times = []
    with count_host_transfers() as n_step:
        for _ in range(50):
            t0 = time.perf_counter()
            ctrl.apply_prestress(control)
            ctrl.poll(state, 1.0e-3)
            wp.synchronize_device(device)
            times.append(time.perf_counter() - t0)
    step_cost = {
        "median_ms": 1e3 * statistics.median(times),
        "transfers_per_step": {k: v / 50 for k, v in n_step.items()},
    }
    t0 = time.perf_counter()
    for _ in range(10):
        solver.step(state, model.state(), control, None, 1.0e-3)
    wp.synchronize_device(device)
    solver_step_ms = 1e3 * (time.perf_counter() - t0) / 10
    return {
        "mesh": mesh_key,
        "dof": int(3 * mesh.nodes.shape[0]),
        "device": device,
        "config": cfg_key,
        "eigensolver": eigensolver,
        "update_ms": split,
        "host_transfers_per_update": transfers,
        "per_step": step_cost,
        "solver_step_ms_for_reference": solver_step_ms,
        "repeats": repeats,
        "eigenvalues_last": info["eigenvalues"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--meshes", nargs="+", default=list(MESHES))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--eigensolvers", nargs="+", default=["arpack", "subspace"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    wp.init()
    results = []
    for mesh_key in a.meshes:
        prebuilt = build(mesh_key, a.device)
        for eigensolver in a.eigensolvers:
            for cfg_key in a.configs:
                r = bench_config(mesh_key, a.device, cfg_key, eigensolver, a.repeats, a.warmup, prebuilt)
                results.append(r)
                u = r["update_ms"]
                print(
                    f"{mesh_key:10s} {a.device:7s} {eigensolver:8s} {cfg_key:8s} "
                    + " ".join(f"{k}={u[k]:.2f}" for k in (*SPLIT, "wall_synced"))
                    + f" | transfers {r['host_transfers_per_update']} | step {r['per_step']['median_ms']:.3f} ms",
                    flush=True,
                )
    meta = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "warp": wp.__version__,
        "cpu": platform.processor(),
        "gpu": wp.get_device(a.device).name if a.device != "cpu" else None,
    }
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
