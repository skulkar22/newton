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

**Guards:** BDF2 raises `ValueError` unless `block_sparse_joints` is used, the model has no joint mimics (the displacement path must be active) and there are no particles.

## How it maps to the code

With ĥ = (2/3)·dt, BDF2 is the same VBD minimisation as BDF1, but uses ĥ instead of dt and a different inertial target.

- **Inertial target:** `x̃ = x_n + Δ_n/3 + ĥ·v_pred`, where `v_pred = 4/3·v_n − 1/3·v_{n−1} + ĥ·a_ext`.
- **Velocity after the solve:** `v = v_pred + (x − x̃)/ĥ`.
- **Rotations:** handled the same way in the tangent space.

| Piece | Where |
|---|---|
| Option, guards, buffers | `solver_vbd.py`: `rigid_time_integration` arg, `_setup_time_integration()` |
| Per-step switch | `step()`: `_bdf2_step = enabled and history_ready`; `rigid_dt = 2/3·dt` on BDF2 steps. That `rigid_dt` is passed to init, iterations, contact export and finalize. Particles still get `dt`. |
| History reset | `reset()` sets `_bdf2_history_ready = False`. The first step after construction or reset is BDF1. |
| Save v_{n−1} on the BDF1 step | `rigid_vbd_kernels.py`: `bdf2_save_history` |
| BDF2 inertial target | `rigid_vbd_kernels.py`: `bdf2_predict_rigid_bodies`. Writes `body_inertia_q`, `v_pred`/`ω_pred`, and the prediction offsets `pred_lin`/`pred_ang`. |
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

- Gyroscopic terms are left out of `ω_pred`.
- Joint damping, friction and drive damping inside the iterations still use `(x − x_n)/ĥ`. That is first-order consistent, not exact BDF2.
- The rotor increment `J·Δθ` is exact for planar hinge chains and first-order accurate for general 3-D rotation.
- The history flag is global, so a partial-world `reset()` restarts every world with one BDF1 step.
- The BDF1/BDF2 choice per step is a Python flag. Capture CUDA graphs only after one eager step.
- Behaviour changes where BDF1 damping was helping: actuated and contact-rich cases (Doctor Legs, G1 tracking) need retuning. See the report.

## How to check a change

- **Unit tests:** run them with the default BDF1. They must match the parent commit.
  `uv run --extra dev -m newton.tests -p test_solver_vbd.py`
- **Accuracy:** passive double pendulum against an exact ODE solution, 30 s, several dt. Error must fall when dt shrinks; if it rises, look for rounded-pose differencing.
- **Conservation:** free two-link body without gravity. COM drift and momentum should stay at µm / 1e-6 levels.
- **Robots:** G1 smoke cases and a contact case.
- **CUDA graphs:** capture after one eager step.

The full study, its scripts and data live in the shared package `vbd-bdf2-prototype` (report.html, AGENTS.md, scripts/, data/).
