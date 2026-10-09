<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Experimental per-segment tendon ALM

Review branch based on Shubhankar's `vbd-tendon-alm-stretch` at
`f72ebdb7efd50ebfd7a58d8f1d2a1b6bce82ba26`. This is an opt-in VBD experiment,
not a production-readiness claim or a change to the upstream tendon PR.
The joint-limit treatment and CUDA optimization are separate commits.

## Enable

Add these options to an existing VBD solver configuration:

```python
solver = newton.solvers.SolverVBD(
    model,
    iterations=32,
    tendon_material_direct=True,
    tendon_alm=True,
    tendon_alm_per_segment=True,
    # Separate experiment, enabled for the reported Toy4 comparison:
    rigid_joint_limit_alm=True,
    # Keep the model's other joint, material, damping, and sigmoid settings.
)
```

Both new options default to false. Per-segment ALM requires direct material
solving and tendon ALM. The existing soft-tendon guard remains; the benchmark
uses `tendon_alm_min_stiffness_ratio=0` to select ALM for all six tendons.
Reconstruct the solver when changing these experimental modes. These are
runtime solver controls, not new authored model or USD properties.

Call `solver.check_tendon_material()` after stepping or replaying a CUDA
graph, outside capture. Accepted-pose failures remain explicit; there is no
fallback to material sweeps.

## What needs review

### Independent forces and a consistent material solve

A shared tendon multiplier cannot retain arbitrary unequal tensions across
a sticking frictional roller. In the frozen two-span regression, valid
material loads near 10 N and 20 N produced endpoint forces near 14.8 N and
15.2 N with the shared algorithm. The independent rows retain the original
loads and the corresponding roller torque.

For a linear undamped span, the force is:

```text
T = max((extension + multiplier / penalty) / (compliance + 1 / penalty), 0)
```

The material projection uses that same force, including the shifted
extension and effective compliance. It removes the shift on output so
physical rest length, rather than the shifted numerical quantity, is
conserved. Body forces, roller reactions, dual updates, and diagnostics use
the same ALM law. The roller-reaction correction also applies when retaining
the shared-ALM option; that option is not a bit-identical baseline of Shubhankar's
original source. Use his unmodified commit for baseline comparisons.

### Nonlinear, damped, and short spans

The nonlinear iteration differentiates the complete ALM force, including
the extension-dependent multiplier weight. Each candidate must satisfy the
actual force law, friction inequalities, slip direction, material inventory,
and rest-length bounds before publication.

Positive damping at engagement is limited to the taut extension formed
during the timestep, not preceding slack travel. Negative damping can
release a stored multiplier. Certified slack rows release their multiplier
exactly; precision checks avoid indefinitely decaying unrepresentable loads.

Rows whose length cannot respond to body motion use the physical material
law without inertial regularization. If a movable short span reaches an
artificial force ceiling caused by its numerical ALM penalty, increase that
penalty and retry. Physical compliance and the minimum rest length do not
change. Review the convergence and penalty-adjustment policy in particular.

### Accepted routing and joint limits

Per-segment ALM performs an accepted-pose cleanup for dynamic rollers that
cross zero wrap during a step. It rebases on the accepted material before
merging, resets obsolete duals, and projects only changed tendons. Fixed
rollers are not detached; activation still uses step-start hysteresis.

The independent `rigid_joint_limit_alm` option treats revolute stops as
compliant ALM rows. Authored stiffness remains physical; damping acts on
penetration rate. Tests compare two-body impacts against an independent
backward-Euler compliant-stop reference, including momentum and energy
checks. This does not implement hard limits or change all joint types.

### CUDA optimization

One warp cooperates on each tendon. Segment coefficients, nonlinear force
evaluations, and trial updates run across lanes; the coupled finite-friction
projection and acceptance checks remain on the leader. CPU retains the
scalar path. Tolerances, force laws, and iteration caps are unchanged.

## Validation and reproduction

From the branch, with the development dependencies installed:

```console
uv run --extra dev --with warp-lang==1.17.0 python -m unittest newton.tests.test_tendon_segment_alm newton.tests.test_tendon_segment_alm_extended newton.tests.test_tendon_alm_cooperative newton.tests.test_vbd_compliant_limits newton.tests.test_tendon_material_vbd newton.tests.test_tendon_material_integration newton.tests.test_tendon_material_nonlinear_integration -v
```

Regressions cover sticking/sliding and reversal, unequal compliance,
roller torque, sigmoid derivatives, damping engagement and cancellation,
slack release, immobile/short spans, dynamic routing, CPU/CUDA and graph
replay. The shared-force, nonlinear-derivative, slack-release, accepted-route,
and joint-impact regressions were observed failing before their corrections.
Cooperative/scalar checks cover 1--32 segments and compare accepted results.

The optimized full Toy4 run matches the corrected scalar implementation's
366 recorded joint samples and all 53 saved final arrays exactly. That checks
preservation by the optimization, not absolute physical accuracy.

## Full Toy4 timing

Original updated Toy4, all three copies, six tendons / 90 segment slots,
6.1 simulated seconds, 60 Hz frames, 20 timesteps per frame, 32 VBD iterations.
Sigmoid-low compliance, friction 0.25, tendon damping 5, slider damping 1,
ALM penalty scale 5, minimum stiffness ratio 0. No geometry or slider-travel
overrides. Both sources have identical hashes for all 130 initial model arrays.
Toy script SHA256:
`2348f89636ac7b9cdb6999bab3fd24d412589d7e454faccf01d2a68376d496d0`.
The private input script and video are intentionally not checked in.

| Paired comparison | Before | After | Runtime reduction |
| --- | ---: | ---: | ---: |
| Corrected scalar to cooperative | 507.573 s | 378.947 s | 25.34% |
| Shubhankar's original to corrected cooperative | 366.597 s | 287.152 s | 21.67% |

These are separate paired runs on an RTX 3080 Laptop GPU with Warp 1.17.0.
Do not mix times across rows: laptop clocks/temperature vary. In each pair,
versions alternate exclusive GPU execution per frame. Timings include warmed
CUDA graph replay and completion synchronization, excluding compilation,
capture, CPU readbacks, diagnostics, and rendering. Each is one complete
paired trajectory, not a repeated-run statistical estimate.
The second row compares different corrected physics, not identical trajectories.
A sampled single-step material profile was 168.697 ms scalar versus 83.743 ms
cooperative; this is not a whole-run kernel aggregate.

## Remaining limitations

- This does not establish correctness or performance for every application setup.
- Existing direct-solver capacity and capstan-ratio bounds remain; see
  [direct material documentation](direct_tendon_material.md).
- Unsupported fixed-roller wraps remain unsupported. The inherited activation
  guard for touching same-winding neighbors is not generally corrected here.
- Some translated-copy trajectory differences and small accumulated material
  inventory drift remain. No claim of exact long-run length conservation is made.
- The opening material hold, empirical numerical penalties, and optional
  joint-limit formulation still need review. Defaults are not recommendations
  for arbitrary stiffnesses, masses, or timestep sizes.
