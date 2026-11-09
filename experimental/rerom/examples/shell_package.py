# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: TID253
"""Run a shell package (triangles, offset weld ties, RBE2 clamp pads) through the re-ROM
CalculiX panel path: CalculiX linear / NLGEOM statics of the thin-solid model, and Newton runs with the fixed
Craig-Bampton basis or with re-ROM, for the package's 'full' and 'reduced' clamp configurations.

The thin-solid model is one layer of C3D8I bricks built from the package mesh with every triangle split into
three quads (``--refine`` subdivides further); the package ties (C.mtx) map to the thin-solid layout exactly
for rigid motion (see ``CcxModel.general_tie_rows``). Results are mid-surface displacements at the package
nodes, keyed by node id.

Examples (from the Newton checkout; CalculiX via $REROM_CCX or ccx on PATH):

    python experimental/rerom/examples/shell_package.py --package PKG --out OUT fe --nlgeom --linear
    python experimental/rerom/examples/shell_package.py --package PKG --out OUT newton --method fixed
    python experimental/rerom/examples/shell_package.py --package PKG --out OUT newton --method rerom
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rerom.ccx import ShellMesh  # noqa: E402
from rerom.panel import PanelCcxModel, PanelReROM, interface_list  # noqa: E402


def build(pkg, work, refine=0, nthreads=16):
    mesh = ShellMesh.from_shell_package(pkg, triangles="split", refine=refine)
    model = PanelCcxModel(mesh, work, nthreads=nthreads)
    return mesh, model


def configs(mesh):
    ids = [c["clamp_id"] for c in mesh.clamps]
    out = {}
    for name in ("full", "reduced"):
        out[name] = [k for k, c in enumerate(mesh.clamps) if c["active"].get(name, 0) == 1]
    return ids, out


def mid_at_source(model, u):
    um, _ = model.mid_and_director(np.asarray(u))
    return um[: model.mesh.orig_count]


def summary(model, u, label):
    um = mid_at_source(model, u)
    k = int(np.argmin(um[:, 2]))
    return {
        "label": label,
        "min_uz_m": float(um[k, 2]),
        "min_uz_node_id": int(model.mesh.node_ids[k]),
        "max_disp_m": float(np.linalg.norm(um, axis=1).max()),
    }


def save_field(path, model, u):
    um = mid_at_source(model, u)
    np.savez_compressed(path, node_id=model.mesh.node_ids[: model.mesh.orig_count], u_mid=um, u_solid=np.asarray(u))


def run_fe(a, mesh, model, out):
    ids, cfg = configs(mesh)
    g = np.array([0.0, 0.0, -9.81]) * a.gravity_factor
    res = {}
    for name in a.configs:
        active = [ids[k] for k in cfg[name]]
        released = [ids[k] for k in range(len(ids)) if k not in cfg[name]]
        fixed = model.clamp_dofs(active)
        for kind in (["linear"] if a.linear else []) + (["nlgeom"] if a.nlgeom else []):
            for n_incr in a.n_incr if kind == "nlgeom" else [1]:
                tag = f"{name}_{kind}" + (f"_inc{n_incr}" if kind == "nlgeom" else "")
                t0 = time.perf_counter()
                r = model.static(
                    fixed, g, nlgeom=(kind == "nlgeom"), rigid_clamps=released, welds=True, n_incr=n_incr, label=tag
                )
                s = summary(model, r["u"], tag)
                s.update(wall_s=time.perf_counter() - t0, increments=r["increments"], active=active, released=released)
                res[tag] = s
                save_field(out / f"fe_{tag}.npz", model, r["u"])
                print(json.dumps(s), flush=True)
    if a.mg_load:
        t0 = time.perf_counter()
        _K, M = model.rest_matrices()
        f = model.gravity_load(M, g)
        t_m = time.perf_counter() - t0
        for name in a.configs:
            active = [ids[k] for k in cfg[name]]
            released = [ids[k] for k in range(len(ids)) if k not in cfg[name]]
            tag = f"{name}_linear_mg"
            t0 = time.perf_counter()
            r = model.static(model.clamp_dofs(active), [0.0, 0.0, 0.0], nlgeom=False, rigid_clamps=released,
                             welds=True, label=tag, nodal_forces=f)
            s = summary(model, r["u"], tag)
            s.update(wall_s=time.perf_counter() - t0, wall_matrices_s=t_m, active=active, released=released)
            res[tag] = s
            save_field(out / f"fe_{tag}.npz", model, r["u"])
            print(json.dumps(s), flush=True)
    (out / "fe_summary.json").write_text(json.dumps(res, indent=1))


def run_newton(a, mesh, model, out):
    ids, cfg = configs(mesh)
    t0 = time.perf_counter()
    K, M = model.rest_matrices()
    C = model.weld_rows()
    deps = getattr(model, "weld_dependents", None)
    itf = interface_list(model)
    t_mat = time.perf_counter() - t0
    g = np.array([0.0, 0.0, -9.81]) * a.gravity_factor
    shift = mesh.points[: mesh.orig_count].mean(axis=0)
    log_lines = []

    def log(s):
        log_lines.append(s)
        print(s, flush=True)

    run = PanelReROM(
        model, M, K, C, itf,
        n_modes=a.modes, n_ritz=a.ritz, gravity=g, shift=shift, zeta=a.zeta, iterations=a.iterations,
        recovery="rotated", tol=a.tol, check_every=a.check_every, min_interval=a.min_interval,
        device=a.device, ke=a.ke, ka=a.ka, rerom=(a.method == "rerom"), log=log, weld_dependents=deps,
    )
    hist, snaps = [], []
    step = 0
    phase_res = {}
    for name in ("full", "reduced"):
        run.set_active(cfg[name], step=step, t=step * a.dt)
        for _ in range(a.steps):
            run.scene.step(a.dt)
            step += 1
            run.maybe_update(step, step * a.dt)
            if step % a.history_every == 0:
                um = mid_at_source(model, run.total_displacement())
                hist.append((step, step * a.dt, float(um[:, 2].min()), name))
            if a.snapshot_every and step % a.snapshot_every == 0:
                snaps.append((step, mid_at_source(model, run.total_displacement()).astype(np.float32)))
        u = run.total_displacement()
        u_eval = run.eval_state(u) if a.method == "rerom" else u
        s = summary(model, u_eval, f"{a.method}_{name}")
        last = [h[2] for h in hist if h[3] == name][-max(1, 100 // a.history_every) :]
        s["drift_last_100_steps_um"] = float((max(last) - min(last)) * 1e6) if last else None
        phase_res[name] = s
        save_field(out / f"newton_{a.method}_{name}.npz", model, u_eval)
        print(json.dumps(s), flush=True)
    res = {
        "method": a.method, "modes": a.modes, "ritz": a.ritz, "dt": a.dt, "steps_per_phase": a.steps,
        "iterations": a.iterations, "zeta": a.zeta, "ke": a.ke, "ka": a.ka, "tol": a.tol,
        "check_every": a.check_every, "min_interval": a.min_interval, "device": a.device,
        "gravity_factor": a.gravity_factor, "refine": a.refine, "phases": phase_res,
        "updates": run.events, "checks": run.checks, "wall_matrices_s": t_mat, "init_wall_s": run.init_wall,
        "cb_info": run.cb.info if a.method == "fixed" else None, "dofs": int(model.n_dof),
        "interfaces": [it["name"] for it in itf], "configs": {k: [ids[i] for i in v] for k, v in cfg.items()},
    }
    (out / f"newton_{a.method}.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(out / f"newton_{a.method}_history.npz", hist=np.array([h[:3] for h in hist]),
                        phase=np.array([h[3] for h in hist]))
    if snaps:
        np.savez_compressed(out / f"newton_{a.method}_frames.npz", step=np.array([s[0] for s in snaps]),
                            u_mid=np.stack([s[1] for s in snaps]))
    (out / f"newton_{a.method}.log").write_text("\n".join(log_lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--gravity-factor", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true", help="build the model, print the preflight facts, stop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    fe = sub.add_parser("fe")
    fe.add_argument("--configs", nargs="+", default=["full", "reduced"])
    fe.add_argument("--linear", action="store_true")
    fe.add_argument("--nlgeom", action="store_true")
    fe.add_argument("--n-incr", type=int, nargs="+", default=[10])
    fe.add_argument("--mg-load", action="store_true",
                    help="also a linear static with gravity as M g on the nodes (the load Newton's ROM sees)")
    nw = sub.add_parser("newton")
    nw.add_argument("--method", choices=["fixed", "rerom"], required=True)
    nw.add_argument("--modes", type=int, default=64)
    nw.add_argument("--ritz", type=int, default=2)
    nw.add_argument("--dt", type=float, default=1e-3)
    nw.add_argument("--steps", type=int, default=6000, help="steps per clamp phase")
    nw.add_argument("--iterations", type=int, default=10)
    nw.add_argument("--zeta", type=float, default=1.0)
    nw.add_argument("--ke", type=float, default=1e10)
    nw.add_argument("--ka", type=float, default=1e9)
    nw.add_argument("--tol", type=float, default=0.01, help="re-ROM displacement-correction trigger")
    nw.add_argument("--check-every", type=int, default=10)
    nw.add_argument("--min-interval", type=int, default=10)
    nw.add_argument("--device", default="cpu")
    nw.add_argument("--history-every", type=int, default=10)
    nw.add_argument("--snapshot-every", type=int, default=25)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mesh, model = build(a.package, out / "ccx_work", refine=a.refine, nthreads=a.threads)
    info = model_info(mesh, model)
    print(json.dumps(info), flush=True)
    if a.dry_run:
        (out / f"dry_run_{a.cmd}.json").write_text(json.dumps(info, indent=1))
        return
    (run_fe if a.cmd == "fe" else run_newton)(a, mesh, model, out)


def model_info(mesh, model):
    """Model facts for the preflight: size, analytic mass (planar quads x thickness x density), clamps, ties."""
    X, Q = mesh.points, mesh.quads
    area = 0.5 * np.linalg.norm(np.cross(X[Q[:, 2]] - X[Q[:, 0]], X[Q[:, 3]] - X[Q[:, 1]]), axis=1)
    mass = {}
    for p, prop in mesh.parts.items():
        sel = mesh.elem_part == p
        mass[p] = float((area[sel] * prop["thickness"] * prop["rho"]).sum())
    ids, cfg = configs(mesh)
    C, deps = (model.weld_rows(), model.weld_dependents) if mesh.shell_C is not None else (model.weld_rows(), None)
    return {
        "shell_nodes": int(mesh.n_nodes), "source_nodes": int(mesh.orig_count), "quads": int(len(Q)),
        "dof": int(model.n_dof), "mass_kg": float(sum(mass.values())), "mass_per_panel_kg": mass,
        "materials": {p: {"E": v["E"], "nu": v["nu"], "rho": v["rho"], "t": v["thickness"]} for p, v in mesh.parts.items()},
        "clamps": {c["clamp_id"]: {"pad_shell_nodes": int(len(c["nodes"])), "source_pad_nodes": int(len(c["source_nodes"])),
                                    "centre": [float(v) for v in c.get("centre", [])]} for c in mesh.clamps},
        "configs": {k: {"active": [ids[i] for i in v], "released": [ids[i] for i in range(len(ids)) if i not in v]}
                    for k, v in cfg.items()},
        "source_tie_rows": None if mesh.shell_C is None else int(mesh.shell_C.shape[0]),
        "solid_tie_rows": int(C.shape[0]), "dependents_unique": None if deps is None else bool(len(set(deps.tolist())) == len(deps)),
    }


if __name__ == "__main__":
    main()
