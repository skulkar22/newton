# BDF2 time-stepping prototype for sparse rigid VBD — notes for agents

Scope: this file covers `newton/_src/solvers/vbd/`. The repository root `AGENTS.md` still applies.

**Status: experimental prototype on branch `vbd-bdf2-prototype`. It is opt-in, and the default path is unchanged.**
Do not make BDF2 the default, remove the BDF1 path, or merge this branch into an integration branch without approval from the branch owner (Shubhankar Kulkarni).

## What problem it solves

VBD takes backward-Euler (BDF1) steps. BDF1 removes energy from every oscillation at a rate of about ω²·dt per second.
- A passive pendulum therefore loses a large share of its swing energy (in the study, about half in 30 s at dt 0.5 ms) and drifts out of phase.
- Shrinking dt only helps linearly.

BDF2 uses the last two steps instead of one. It is second-order accurate and barely damps slow motions, while still strongly damping very stiff modes (joints, contacts).

## How to turn it on

```python
solver = newton.solvers.SolverVBD(
    model,
    rigid_articulation_solve="block_sparse_joints",  # required
    rigid_compliant_alm=True,
    rigid_time_integration="bdf2",  # default "bdf1"
)
```

The env var `NEWTON_VBD_TIME_INTEGRATION=bdf1|bdf2` overrides the argument. It exists so harnesses can switch without code edits, and is a prototype convenience only.

Prototype switches for studies (defaults shown; all CUDA-graph safe because they are read once at construction):

- `NEWTON_VBD_BDF2_RATES=1`: BDF2 velocity in every in-solve rate (`0` reproduces the first prototype's 1.5x over-damping).
- `NEWTON_VBD_BDF2_GYRO=1`: gyroscopic term, explicit at the extrapolated rate (`0` off, `2` implicit Newton step; tested less accurate).
- `NEWTON_VBD_WARM_START`: displacement warm start, `1` by default with BDF2 and `0` with BDF1.
- `NEWTON_VBD_BDF2_STARTUP=constant_velocity|gravity|free_fall`: synthetic history for fresh worlds (`rigid_bdf2_startup`, see the table below).
- `NEWTON_VBD_EXT_FORCE=target|solve`: where `body_f` and mapped `joint_f` enter (`rigid_external_force_mode`; both integrators; default `target` for BDF1, `solve` for BDF2). `target` shifts the inertial target by h²M⁻¹f; `solve` keeps the target kinematic + gravity and adds the wrench as a constant force in the solve (`add_external_wrench_to_solve`, launched right after the per-iteration zeroing of `body_forces`), so the velocity read-back `v_pred + d/ĥ` and the BDF2 increment stay exact.
- `NEWTON_VBD_EXT_FORCE_GUESS=0|1` (`rigid_external_force_guess`, default off): with `EXT_FORCE=solve` and articulations, one unrelaxed block-sparse solve `H d0 = f_ext` (or `f_n − f_{n−1}` on warm-started bodies) before the iterations (`ext_force_guess_rhs`, `override_articulation_rhs_scalar`, the `rhs_override` input of the serial kernel). It removes the lag of `solve` mode at relaxation < 1 without the over-response of `target` mode.
- `NEWTON_VBD_WARM_GAMMA=1.0`: fraction of the previous displacement used by the warm start (0 = start at the inertial target).
- `NEWTON_VBD_RHO_ARMATURE=0|1` (`rigid_axis_support_armature`; default on for BDF2, off for BDF1): add `armature/h²` to the support (ALM rho) of revolute drive/limit rows (`add_armature_to_axis_support`). A limit between a very light link and a heavy one (G1 waist_roll → torso) otherwise gets its rho from the light link alone and does not hold: torso falling onto a 0.52 rad limit through a 0.047 kg link overshoots 2.4–3.0 rad at 4 ms (0.54 rad even at 0.8 ms, BDF1 too); with the armature term 0.007–0.010 rad.
- `NEWTON_VBD_LIMIT_RHO_FLOOR=0|1` (`rigid_limit_rho_floor`; default on for BDF2, off for BDF1): floor the support (ALM rho) of every limit row (revolute, prismatic, D6) at `9·(K + D/h)`, the floor drive and structural rows already have (`floor_limit_axis_support`). Covers models without armature: light-link chain at 4 ms / 7 it, torso falling onto +0.52 rad, overshoot 2.4–3.0 rad without it, ≤ 0.006 rad with it for armature 0, 1e-4 and 1e-2, both integrators, gravity-only and held by a PD.
- `NEWTON_VBD_BDF2_KINEMATIC_RATE=1`: BDF2 rate reference of kinematic bodies `x_n − (ĥ/dt)(x_n − x_{n−1})` (`bdf2_kinematic_rate_reference`), so a user-animated kinematic body has velocity +V in the solve and in `body_qd` (the first prototype gave −V/2 and dragged a box resting on a moving platform backwards).
- `NEWTON_VBD_BDF2_THETA_POSE_LIN/ANG=1e-6`: F6 tolerances. With theta `event`/`contact`, a dynamic body whose pose differs from the solver's last final pose (an external write without `reset()`) makes its world take BDF1 steps (`bdf2_theta_detect_pose_writes`).
- `NEWTON_VBD_BDF2_EXACT_ROT=1`: compose the BDF2 rotational prediction (`exp(ĥω_pred)·exp(Δ_n/3)`), step increment (`exp(φ)·exp(pred)`) and sparse velocity read-back (`exp(φ)·exp(ĥω_pred)`) from small rotation vectors (`rigid_bdf2_exact_rotation`); `0` adds the vectors (first prototype, first-order for large rotations about different axes).
- `NEWTON_VBD_BDF2_EXACT_ROTOR=composed|linear|absolute` (`rigid_bdf2_rotor_residual`): BDF2 rotor (armature) residual. `composed` (default) is the twist of the composed parent/child rotation increments (`_bdf2_exact_hinge_increment`, small terms); `linear` is the first prototype's `J·(pred+φ) − delta`, whose O(pred²) error times armature/ĥ² blows up light links under explicit torque (the G1 waist failure); `absolute` differences absolute angles as BDF1 does (stable, but 3.5× worse float32 pendulum floor).

Recommended for whole-body humanoids (from the G1 study):
- Use the block-sparse articulation solve; BDF2 requires it.
- Set `rigid_articulation_relaxation=1.0`.
- Use the warm start, which is on by default with BDF2.
- Feed time-varying PD targets sampled at the **end** of the step (t + dt). An implicit step balances forces there, and start-of-step targets add a one-step lag to both integrators.

**Guards:** BDF2 raises `ValueError` unless `block_sparse_joints` is used, the model has no joint mimics (the displacement path must be active) and there are no particles.

## How it maps to the code

With ĥ = (2/3)·dt, BDF2 is the same VBD minimisation as BDF1, but uses ĥ instead of dt and a different inertial target.

- **Inertial target:** `x̃ = x_n + Δ_n/3 + ĥ·v_pred`, where `v_pred = 4/3·v_n − 1/3·v_{n−1} + ĥ·a_ext`.
- **Velocity after the solve:** `v = v_pred + (x − x̃)/ĥ`.
- **Rotations:** handled the same way in the tangent space.

| Piece | Where |
|---|---|
| Option, guards, buffers | `solver_vbd.py`: `rigid_time_integration` arg, `_setup_time_integration()` |
| Per-step switch | `step()`: `_bdf2_step = enabled` (constant, no Python branch per step); `rigid_dt = 2/3·dt`. That `rigid_dt` is passed to init, iterations, contact export and finalize. Particles still get `dt`. |
| Start-up and reset | Worlds flagged in `_rigid_pose_rebaseline_mask` (construction, `reset(world_mask)`) start from a synthetic history inside `bdf2_predict_rigid_bodies`, chosen by `rigid_bdf2_startup` / `NEWTON_VBD_BDF2_STARTUP`: `gravity` (default: constant-gravity body history, constant-velocity rotor history; keeps the DevTech pendulum accuracy), `constant_velocity` (`v_{n−1} = v_n`, `x_n − x_{n−1} = dt·v_n`; leaves a permanent −½·dt·a lag, 2–3× worse pendulum error), `free_fall` (first prototype, ablation only: adds `body_f`/`joint_f` accelerations, which kicks a held light link by `dt·τ/(4·I_link)`). Every step uses the same h, so a CUDA graph captured on the first step is exact and resets are per world. |
| BDF2 inertial target | `rigid_vbd_kernels.py`: `bdf2_predict_rigid_bodies`. Writes `body_inertia_q`, `v_pred`/`ω_pred` (with the gyroscopic torque at `2ω_n − ω_{n−1}`), the prediction offsets `pred_lin`/`pred_ang`, the **rate reference pose** `x_n + (x_n − x_{n−1})/3`, and rebaselines `body_q_prev` for fresh worlds. |
| In-solve rates | `solver_vbd.py`: `_rate_reference_q` (rate reference on BDF2 steps, `body_q_prev` otherwise) is passed wherever a rate or slip is formed: joint assembly, `update_duals_joint`, body-body contact accumulation and duals, contact-force export, `solve_rigid_body`. Pose uses (coordinate unwrap, friction anchor, C0) keep `body_q_prev`. |
| Warm start | `rigid_sparse_articulation_kernels.py`: `warm_start_body_displacement`; `solver_vbd.py`: `_init_body_displacement`. Buffers are allocated at construction. |
| Velocity + exact increments | `rigid_vbd_kernels.py`: `update_body_velocity`. On a BDF2 step `ω = ω_pred + φ/ĥ`. It stores `inc = pred_offset + solver_displacement` (BDF1 step: `inc = v·dt`). |
| Rotor (armature) target | `rigid_sparse_articulation_kernels.py`: `predict_articulation_armature_bdf2`. Writes the target and `delta = target − q_n` from small terms. |
| Rotor residual offset | `rigid_sparse_articulation_kernels.py`: `bdf2_armature_offset`. Computes `q(x̃) − target = J·pred_offset − delta`. |
| Rotor residual in the solve | `_assemble_articulation_joint_scalar`: when `armature_incremental` is set, the residual is `J·body_disp + offset`. |

## BDF2 stability (branch `vbd-bdf2-stability`, Phase 1 of the 2026-10-12 plan)

Why the prototype blew up on G1 with explicit torques (`Control.joint_f`), in order of impact:

1. **Rotor residual linearisation (fixed by `EXACT_ROTOR=composed`).** The BDF2 armature residual was `J·(pred + φ) − delta`, first order in the bodies' rotation increments. An explicit torque rotates a light link's inertial target by `ĥ²τ/I_link` (0.1–0.7 rad on the G1 waist/ankle links, about axes off the hinge), the solve rotates it back (`φ ≈ −pred`), and the O(|pred|²) error becomes a phantom hinge angle multiplied by `armature/ĥ²` (armature 500–2500× the link inertia). It does not shrink with iterations. BDF1 evaluates `q(x) − target` exactly and never saw it.
2. **Force-shifted iteration start (fixed by `EXT_FORCE=solve`).** Putting `body_f`/`joint_f` into `x̃` starts the iteration `ĥ²M⁻¹f` away from the constrained solution. Even BDF1 at relaxation 0.65 keeps a steady-state error from it (light-link chain: roll 0.285 rad instead of the converged 0.508). In `solve` mode the wrench is a constant force in the solve (`add_external_wrench_to_solve`); the minimizer is unchanged.
3. **Start-up history (F1).** The free-fall synthetic history kicked held links by `dt·τ/(4·I_link)` because it included `joint_f`/`body_f`. The default is now a gravity-only history (in `solve` mode `free_fall` and `gravity` coincide); with the theta selection fresh worlds take real BDF1 start-up steps instead. A constant-velocity history removes the kick too but leaves a −½·dt·a velocity lag (2–3× worse DevTech pendulum).
4. **Non-smooth events (D5).** BDF2 treats an impact as a trend (rebound ~0.2 of the drop height with a critically damped contact). The theta selection takes BDF1 steps around events.

### Theta selection (F2, Stage A: per world)

- `rigid_bdf2_theta_policy` / `NEWTON_VBD_BDF2_THETA`: `off`, `fresh`, `event` (default), `event_nofresh` (events, but fresh worlds use the synthetic `rigid_bdf2_startup` history instead of BDF1 start-up steps), `contact`, `bdf1`, `bdf2` (the last two force the selection for verification). `SolverVBD.bdf2_theta_stats()` returns the BDF2 fraction of world-steps and per-detector event counts.
- Every step, before the solve, `_select_bdf2_theta` runs device-only detectors and writes `world_bdf2[world]` (1 = BDF2 step, 0 = BDF1 step):
  `bdf2_theta_detect_dofs` (`|joint_f_n − joint_f_{n−1}| > abs + rel·|f_{n−1}|`, joint limit engaging/releasing from `state.joint_q`), `bdf2_theta_detect_bodies` (velocity second difference `v_n − 2v_{n−1} + v_{n−2}` above `lin/ang + rel·|v|`, `body_f` jump), `bdf2_theta_count_contacts` (contacts predicted active this step, `separation + dt·approach rate < 0`; an event when the per-world count changes), the rebaseline mask (construction, `reset()`). `bdf2_theta_select_worlds` holds BDF1 for `NEWTON_VBD_BDF2_THETA_HOLD` (2) steps after an event; `bdf2_theta_expand_bodies` writes the per-body `RigidStepSize` (`h`, `inv_h2`, `bdf2`).
- Thresholds: `NEWTON_VBD_BDF2_THETA_FORCE_ABS` (1.0) / `_FORCE_REL` (0.5), `_VEL_LIN` (0.05 m/s) / `_VEL_ANG` (0.5 rad/s) / `_VEL_REL` (0.2). Phase 2 tunes them.
- Every rigid kernel that used the scalar step takes `RigidStepSize` and calls `rigid_step_h` / `rigid_step_inv_h2` / `rigid_step_bdf2` for its body (joints: the child; contacts: shape 0's body, else shape 1's). All bodies of a world share one value, so each joint and contact row has one step. With `per_body == 0` (theta off, BDF1) the helpers return the scalar, so those paths are unchanged.
- A BDF1 step of a BDF2 solver runs the BDF1 code verbatim: `integrate_rigid_body` prediction (in `bdf2_predict_rigid_bodies`), `body_q_prev` as the rate reference, the BDF1 rotor target and absolute rotor residual, no warm start, BDF1 read-back. It also records the history the next BDF2 step needs (`hist_qd`, the step increment, the rotor rate), so returning to BDF2 needs no synthetic history.
- Kernels that unit tests launch directly keep their signatures; the theta variants are separate kernels sharing an `_impl` function (`step_body_body_contact_C0_lambda_theta`, `update_duals_joint_theta`, `update_duals_body_body_contacts_theta`).
- Proofs (CPU fixtures, `stability-runs/identity.sh`): BDF1 default == 8afa14fc; theta `off` == the previous BDF2; theta forced `bdf2` == the previous BDF2; theta forced `bdf1` == BDF1 (8afa14fc) — bit for bit, including contacts.
- Start-up choice, measured on the RC (default `event` = BDF1 start-up steps vs `event_nofresh` = synthetic gravity history):

  | Test | `event` | `event_nofresh` |
  |---|---|---|
  | DevTech pendulum, RMS over 8 s, CPU (0.125/0.25/0.5/1/2 ms) | 4.11/1.53/2.05/3.17/6.95 e-4 | 4.35/1.59/2.29/3.35/7.99 e-4 |
  | D1 hinge, solve mode, joint_qd after 1 / 50 steps (exact 0.0799 / 3.9944) | 0.0798 / 3.9943 | 0.0532 / 3.9544 |
  | D1 hinge, reset mid-run, after 1 step (exact 8.0687) | 8.0686 | 8.0020 |
  | G1 air_cycle / coordinated, 4 ms / 8 it, worst joint error vs converged reference | 0.0352° / 0.0377° | 0.0315° / 0.0375° |
  | Partner input, internal 4 ms / explicit 0.8 ms (waist_pitch overshoot, q RMS) | 0.003, 0.0037 / 0.004, 0.0034 | identical |

  Worst relative regression: `event` +12 % (air_cycle), `event_nofresh` −33 % first-step velocity on the D1 hinge (a
  permanent −½·dt·a lag of the synthetic history). Default: `event`. With theta `event`, the first step of a fresh world
  (and the pose-write detector on it) is always BDF1; `event_nofresh` suppresses all events on fresh worlds.
- Doctor Legs (contact-rich, 8 it, suite agent): `event` path_rms 0.112 m vs `off`/`fresh` 0.090 vs BDF1 0.123; closure/penetration 351 µm / 5.6 mm vs 430 / 7.5 — `event` runs mostly as BDF1 there (policy trade-off).
- Stage B (per-body θ inside a world) is not implemented. It would blend only the inertial term and keep the world step for interaction rates; do it only if Phase 2 shows the per-world fallback costs accuracy in contact-rich worlds.

### GPU bit-identity of the legacy paths (keep it)

CPU fixtures can be bit-identical while the GPU is not: NVRTC contracts `a*b + c` into FMA only when the product has a single use. Two rules keep the switched-off paths bitwise equal to the pinned prototype on the GPU:
- Small rotations are composed with `rotation_vector_compose` (third-order BCH series below 0.03 rad, quaternion product above, selected with `wp.where`), and the rotor twist is taken from the rotated small vector. Conjugating a near-identity quaternion by a full joint-frame quaternion (the first `composed` rotor) loses most digits of a 1e-5 rad increment and made the pendulum worse at 0.125–0.25 ms.
- Never add a branch that reuses a sub-expression of an existing sum. The first version of `EXACT_ROT` recomputed `h * w_pred` inside an `if`; the shared product stopped the contraction of `inc/3 + h*w_pred` and changed BDF2 by 1 ulp from the first step. The composition now runs in its own kernel (`bdf2_compose_inertial_rotation`), and the prediction kernel only stores `step_d_ang`/`step_rot_n` for it.
- Do not wrap an existing kernel body in a `@wp.func` that both the old and a new kernel call. That changed BDF1 G1 results on the GPU (`update_duals_joint`). Kernels that need a theta variant keep the original kernel verbatim; the `*_theta` kernel calls an `_impl` copy. Keep the two in sync.
Check with `stability-runs/gpu_legacy_all.sh` (chain and fixed-base G1, BDF1 and the pinned BDF2 switches, cuda:1) against the pinned build.

### Switch summary and defaults

| Switch | Env | BDF2 default | Pinned prototype |
|---|---|---|---|
| `rigid_bdf2_startup` | `NEWTON_VBD_BDF2_STARTUP` | `gravity` (not used when the theta policy bootstraps fresh worlds) | `free_fall` |
| `rigid_external_force_mode` | `NEWTON_VBD_EXT_FORCE` | `solve` (BDF1 default stays `target`) | `target` |
| `rigid_bdf2_exact_rotation` | `NEWTON_VBD_BDF2_EXACT_ROT` | `1` | `0` |
| `rigid_bdf2_rotor_residual` | `NEWTON_VBD_BDF2_EXACT_ROTOR` | `composed` | `linear` |
| `rigid_bdf2_theta_policy` | `NEWTON_VBD_BDF2_THETA` | `event` | `off` |
| `rigid_axis_support_armature` | `NEWTON_VBD_RHO_ARMATURE` | on (BDF1: off) | off |
| `rigid_limit_rho_floor` | `NEWTON_VBD_LIMIT_RHO_FLOOR` | on (BDF1: off) | off |
| `rigid_bdf2_kinematic_rate` | `NEWTON_VBD_BDF2_KINEMATIC_RATE` | on | off |
| `rigid_external_force_guess` | `NEWTON_VBD_EXT_FORCE_GUESS` | off | off |

Setting the pinned column reproduces 8afa14fc BDF2 bit for bit. All switches are read at construction; per-step state is on the device, so a CUDA graph captured at construction replays exactly.

### Open follow-ups (not in this branch)

- The same under-weighted rho exists in joint dry-friction rows (bounded by the friction limit), in the tendon ALM row on `dev/rigid-vbd-integration` (`tendon_alm=True`: rho from the lightest attachment, one-sided, retained multiplier, so slack-to-taut behaves like a limit hit), and in contact rows on free light bodies (10 kg box on a free 0.05 kg plate at 4 ms sinks 5.2 mm vs 0.5 mm static; articulated links are protected by the structural term). Contact logic was out of scope here.
- Friction stick measured from the anchor pose, `eval_joint_forces` with the solver's rate reference and step, contact-row support from the subtree mass, theta = 0 for unconverged bodies, per-body theta (Stage B).

## The one rule that matters: never difference rounded absolute poses

Absolute positions and quaternions are float32. Their rounding (~1e-7 relative) is the same size as one step's motion at small dt.
- **What breaks:** any quantity formed as `x_n − x_{n−1}`, `q(x) − q_n`, or `(x − x_n)/dt` from absolute poses injects noise every step.
- **Why BDF2 is exposed:** BDF1's damping hid this noise. BDF2 does not, and the first prototype's error *grew* as dt shrank because of it.
- **What to do instead:** carry increments as sums of small terms: the prediction offset plus the solver displacement (`_body_disp`, `_body_disp_ang`). v1 and v2 of the COM-drift fix rely on the same idea.
- **Open issue in BDF1:** the BDF1 rotor residual still uses `q(x) − target` from absolute angles. That should be fixed too.

## Known limitations (prototype)

- Fixed time step: the BDF2 coefficients assume a constant dt; a step-size change needs the variable-step formula.
- Joint mimics and particles are not supported with BDF2 (the constructor refuses them).
- The rotor increment `J·Δθ` is exact to second order for hinges (child and parent increments differ only along the axis).
- In-solve rates still difference absolute poses against the rate reference (same float32 precision as BDF1 has today).
- A float32 rotation floor limits fast spins at very small steps for both integrators.

## How to check a change

- **Unit tests:** run them with the default BDF1. They must match the parent commit.
  `uv run --extra dev -m newton.tests -p test_solver_vbd.py`
- **Accuracy:** passive double pendulum against an exact ODE solution, 30 s, several dt. Error must fall when dt shrinks; if it rises, look for rounded-pose differencing.
- **Conservation:** free two-link body without gravity. COM drift and momentum should stay at µm / 1e-6 levels.
- **Robots:** G1 smoke cases and a contact case.
- **CUDA graphs:** capture after one eager step.

The full study, its scripts and data live in the shared package `vbd-bdf2-prototype` (report.html, AGENTS.md, scripts/, data/).
