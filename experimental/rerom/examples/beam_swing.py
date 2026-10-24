# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# re-ROM Example: Beam Swing
#
# A soft cantilever, clamped at its root face and released from rest in
# the horizontal position, swings down under gravity (large rotation of
# the tip). The same reduced elastic body (12 clamped modes) is simulated
# with SolverVBD twice:
#
#   * fixed-basis FFRF: the linear modal basis of the rest shape is kept;
#   * re-ROM: a rerom.ReROMController re-linearizes the body about its
#     current shape whenever the element-rotation trigger fires, using the
#     default UpdatePolicy (device-side update path), the warm-started
#     restricted subspace eigensolver and two static Ritz vectors.
#
# Both are compared with a short implicit nonlinear FE reference
# (rerom.FullFEReference, Newmark average acceleration, StVK) on the same
# mesh. The example prints the tip error of each reduced model, the
# number of updates, the mean update cadence and the per-update cost.
#
# This example is experimental and lives outside the newton package; see
# experimental/rerom/README.md.
#
# Command: python experimental/rerom/examples/beam_swing.py
#          python experimental/rerom/examples/beam_swing.py --device cuda:0
#
###########################################################################

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import warp as wp

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import rerom  # noqa: E402
from rerom.assembly_warp import WarpTetAssembler  # noqa: E402

import newton  # noqa: E402

wp.config.quiet = True

# Geometry and density of the recorded M1 beam (L/h = 40) on a coarse mesh; the coarse linear tets lock in
# bending, so the default Young's modulus is lower than M1's 3e7 Pa to keep a large swing.
LENGTH, WIDTH, HEIGHT = 1.0, 0.025, 0.025
DENSITY, POISSON = 1100.0, 0.35
GRAVITY = -9.81


def build_case(nx, ny, nz, mode_count, young):
    """Mesh, host assembler, clamped modal basis and the root anchor point."""
    mesh = rerom.make_beam_mesh(LENGTH, WIDTH, HEIGHT, nx, ny, nz, density=DENSITY, young=young, poisson=POISSON)
    asm = rerom.NumpyTetAssembler(mesh, "stvk")
    root = mesh.node_sets["root"]
    gen = newton.ModalGeneratorFEM(
        mesh.nodes,
        asm.mass_matrix,
        asm.linear_stiffness(),
        mode_count=mode_count,
        damping_ratio=0.0,
        fixed_node_indices=root,
    )
    basis = gen.build()
    anchor = mesh.nodes[root].mean(axis=0)
    return mesh, asm, gen, basis, anchor


def build_model(basis, anchor, device):
    """One reduced elastic body held at its root centre by a fixed joint."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, GRAVITY))
    body = builder.add_body_elastic(xform=wp.transform_identity(), modal_basis=basis, lock_inertia=True)
    builder.add_joint_fixed(
        parent=-1,
        child=body,
        parent_xform=wp.transform(wp.vec3(*anchor), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(*anchor), wp.quat_identity()),
    )
    builder.color()
    return builder.finalize(device=device), body


def simulate(args, case, *, adaptive):
    """Run one reduced model; returns the world tip trajectory (steps + 1, 3) and the controller."""
    mesh, _asm, gen, basis, anchor = case
    model, body = build_model(basis, anchor, args.device)
    solver = newton.solvers.SolverVBD(model, iterations=args.iterations)
    trigger_options = rerom.TriggerOptions(max_updates=None if adaptive else 0)
    # The fixed-basis run uses the same controller with an update budget of zero: it never re-linearizes,
    # its pre-stress stays zero, and it only serves to read the world node positions.
    ctrl = rerom.ReROMController(
        model,
        solver,
        body,
        WarpTetAssembler.from_mesh(mesh, "stvk", device=args.device),
        rerom.FreeBodyAdapter(gen.modal_matrix, mesh.nodes, fixed_nodes=mesh.node_sets["root"]),
        mode_count=args.modes,
        basis=rerom.BasisOptions(n_ritz=2),
        trigger=rerom.RotationTrigger(theta_tol=np.radians(args.theta_deg), options=trigger_options),
        eigensolver=args.eigensolver,
        policy=rerom.UpdatePolicy(),  # default: UpdatePath.DEVICE
    )
    tip_nodes = mesh.node_sets["tip"]
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    tips = [ctrl.world_field(state_0)[tip_nodes].mean(axis=0)]
    start = time.perf_counter()
    for _ in range(args.steps):
        ctrl.apply_prestress(control)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, args.dt)
        state_0, state_1 = state_1, state_0
        ctrl.after_step(state_0, control, args.dt)
        tips.append(ctrl.world_field(state_0)[tip_nodes].mean(axis=0))
    wp.synchronize_device(args.device)
    return np.asarray(tips), ctrl, time.perf_counter() - start


def reference(args, case):
    """Implicit nonlinear FE tip trajectory on the same mesh (host, SciPy)."""
    mesh, asm, _gen, _basis, _anchor = case
    fe = rerom.FullFEReference(asm, gravity=(0.0, 0.0, GRAVITY), clamped_nodes=mesh.node_sets["root"])
    start = time.perf_counter()
    res = fe.run(dt=args.dt, steps=args.steps, tip_nodes=mesh.node_sets["tip"])
    return res["tip_trajectory"], time.perf_counter() - start


def tip_angle_deg(tip, anchor):
    d = tip - anchor
    return float(np.degrees(np.arctan2(-d[2], d[0])))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed-basis FFRF vs re-ROM beam swing.")
    parser.add_argument("--device", default="cpu", help="Newton Model and tangent assembler device.")
    parser.add_argument("--steps", type=int, default=300, help="Time steps (default 0.6 s, the first swing).")
    parser.add_argument("--dt", type=float, default=2.0e-3, help="Time step [s].")
    parser.add_argument("--nx", type=int, default=20, help="Hex cells along the beam.")
    parser.add_argument("--ny", type=int, default=2, help="Hex cells across the width.")
    parser.add_argument("--nz", type=int, default=2, help="Hex cells across the height.")
    parser.add_argument("--young", type=float, default=1.0e7, help="Young's modulus [Pa].")
    parser.add_argument("--modes", type=int, default=12, help="Reduced mode count.")
    parser.add_argument("--theta-deg", type=float, default=0.5, help="Rotation trigger tolerance [deg].")
    parser.add_argument("--eigensolver", default="subspace", help="'subspace', 'arpack' or 'dense'.")
    parser.add_argument("--iterations", type=int, default=10, help="SolverVBD iterations per step.")
    parser.add_argument("--no-reference", action="store_true", help="Skip the implicit FE reference.")
    args = parser.parse_args(argv)

    wp.init()
    case = build_case(args.nx, args.ny, args.nz, args.modes, args.young)
    mesh, anchor = case[0], case[4]
    print(
        f"beam: {mesh.nodes.shape[0]} nodes, {mesh.tets.shape[0]} tets, {args.modes} modes; "
        f"device {args.device}; dt {args.dt:g} s x {args.steps} steps; eigensolver {args.eigensolver}; "
        f"trigger {args.theta_deg:g} deg"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Sturm-check and interpolation notices of the experimental package
        tip_fixed, _, wall_fixed = simulate(args, case, adaptive=False)
        tip_rerom, ctrl, wall_rerom = simulate(args, case, adaptive=True)

    updates = ctrl.update_count
    update_ms = [1.0e3 * s["timings"]["total"] for s in ctrl.stats]
    print(f"fixed-basis FFRF: wall {wall_fixed:.2f} s (first run: includes Warp module loading)")
    print(
        f"re-ROM:           wall {wall_rerom:.2f} s, {updates} updates"
        + (
            f" (every {args.steps / updates:.1f} steps), per update {np.mean(update_ms):.1f} ms mean, "
            f"{np.median(update_ms):.1f} ms median"
            if updates
            else ""
        )
    )
    diff = np.linalg.norm(tip_rerom - tip_fixed, axis=1) / LENGTH
    print(f"tip difference fixed vs re-ROM: max {diff.max():.4f} L, final {diff[-1]:.4f} L")

    tip_ref = None
    if not args.no_reference:
        tip_ref, wall_ref = reference(args, case)
        print(f"implicit FE reference: wall {wall_ref:.2f} s")
        for name, tip in (("fixed-basis FFRF", tip_fixed), ("re-ROM", tip_rerom)):
            err = np.linalg.norm(tip - tip_ref, axis=1) / LENGTH
            print(f"tip error {name:<16}: RMS {np.sqrt(np.mean(err**2)):.4f} L, max {err.max():.4f} L")

    print("tip drop angle [deg] (from the horizontal, about the root centre):")
    header = f"{'step':>6} {'t [s]':>7} {'fixed':>8} {'re-ROM':>8}" + ("" if tip_ref is None else f" {'FE ref':>8}")
    print(header)
    for k in range(0, args.steps + 1, max(args.steps // 6, 1)):
        row = f"{k:>6} {k * args.dt:>7.3f} {tip_angle_deg(tip_fixed[k], anchor):>8.2f} "
        row += f"{tip_angle_deg(tip_rerom[k], anchor):>8.2f}"
        if tip_ref is not None:
            row += f" {tip_angle_deg(tip_ref[k], anchor):>8.2f}"
        print(row)


if __name__ == "__main__":
    main()
