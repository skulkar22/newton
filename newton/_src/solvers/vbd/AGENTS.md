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
| Start-up and reset | Worlds flagged in `_rigid_pose_rebaseline_mask` (construction, `reset(world_mask)`) start from a synthetic history inside `bdf2_predict_rigid_bodies`, chosen by `rigid_bdf2_startup` / `NEWTON_VBD_BDF2_STARTUP`: `constant_velocity` (default: `v_{n−1} = v_n`, `x_n − x_{n−1} = dt·v_n`, the same rule as the rotor history in `predict_articulation_armature_bdf2`), `gravity` (adds g to the linear history only), `free_fall` (first prototype, ablation only: adds `body_f`/`joint_f` accelerations, which kicks a held light link by `dt·τ/(4·I_link)`). Every step uses the same h, so a CUDA graph captured on the first step is exact and resets are per world. |
| BDF2 inertial target | `rigid_vbd_kernels.py`: `bdf2_predict_rigid_bodies`. Writes `body_inertia_q`, `v_pred`/`ω_pred` (with the gyroscopic torque at `2ω_n − ω_{n−1}`), the prediction offsets `pred_lin`/`pred_ang`, the **rate reference pose** `x_n + (x_n − x_{n−1})/3`, and rebaselines `body_q_prev` for fresh worlds. |
| In-solve rates | `solver_vbd.py`: `_rate_reference_q` (rate reference on BDF2 steps, `body_q_prev` otherwise) is passed wherever a rate or slip is formed: joint assembly, `update_duals_joint`, body-body contact accumulation and duals, contact-force export, `solve_rigid_body`. Pose uses (coordinate unwrap, friction anchor, C0) keep `body_q_prev`. |
| Warm start | `rigid_sparse_articulation_kernels.py`: `warm_start_body_displacement`; `solver_vbd.py`: `_init_body_displacement`. Buffers are allocated at construction. |
| Velocity + exact increments | `rigid_vbd_kernels.py`: `update_body_velocity`. On a BDF2 step `ω = ω_pred + φ/ĥ`. It stores `inc = pred_offset + solver_displacement` (BDF1 step: `inc = v·dt`). |
| Rotor (armature) target | `rigid_sparse_articulation_kernels.py`: `predict_articulation_armature_bdf2`. Writes the target and `delta = target − q_n` from small terms. |
| Rotor residual offset | `rigid_sparse_articulation_kernels.py`: `bdf2_armature_offset`. Computes `q(x̃) − target = J·pred_offset − delta`. |
| Rotor residual in the solve | `_assemble_articulation_joint_scalar`: when `armature_incremental` is set, the residual is `J·body_disp + offset`. |

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
