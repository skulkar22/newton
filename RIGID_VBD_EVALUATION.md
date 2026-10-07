# Experimental rigid VBD evaluation

This branch is a personal-fork integration candidate for exploratory controller evaluation. Upstream Newton main is unchanged. It combines main `31f525bd017b8d0d14794b4973e628a62606625a` (observed 7 October 2026) with #4179 at `dbb88c2d822d4593d56799418f8b4f08d1335560`, which already includes #4150 at `d089cd680a0118568879b5ab9fdf31e863b42a0b`. #4179 remains an upstream draft titled DO NOT MERGE; this fork is not upstream approval. #4264 is not merged.

## Start

```bash
git clone --branch dev/rigid-vbd-integration https://github.com/skulkar22/newton.git
cd newton
uv sync --locked --extra dev --extra torch-cu13
uv run --no-sync -m newton.tests -k test_vbd_sparse_articulation --no-cache-clear
```

The tested workstation uses Python 3.12.14, Warp 1.18.0, NumPy 2.5.3, Torch 2.14.0+cu130, MuJoCo 3.14.0, and an RTX PRO 6000 Blackwell with driver 595.84. Dependency and CUDA selection on another workstation must match its hardware. Record the checked-out SHA and runtime versions with results.

For external torque control, use `rigid_articulation_solve="block_sparse_joints"`. Zero **both** `joint_target_ke` and `joint_target_kd` before constructing the solver, then write application-bounded efforts to `Control.joint_f`. Setting `joint_target_mode` alone does not disable VBD drives. Map joint names to their coordinate and DOF offsets; `joint_f` uses DOF ordering, including any free-base DOFs. Keep authored passive damping, friction and armature separate from controller gains. VBD does not implement native effort/velocity limiting here; clip commands explicitly. Obtain floating-base feedback from body transforms and velocities rather than assuming free-base joint-coordinate arrays are updated.

Our supported G1 torque fixture uses a 1 ms physics/controller step, 12 iterations, relaxation 0.65, compliant ALM, structural joint stiffness 1e6 and damping 500. External body PD is 300/8; finger PD is 30/1. The supported-contact case explicitly uses contact stiffness 100,000 N/m and damping 100 N·s/m. These are fixture settings, not tuned SONIC settings or a WBC recommendation. Doctor Legs uses a distinct legacy-ALM calibration and 4 ms step; do not mix the two configurations implicitly.

## Validation and limitations

The refreshed main baseline passed **1,586 tests, with 109 skips**. The integrated targeted suites finished at **1,754 passed, 1 inherited failure, 109 skipped**. These are selected suites, not full CI. The remaining sparse CUDA fixed-frame force assertion measures 0.03346539 against 0.03 and reproduces identically on untouched #4179 under Warp 1.18; the tolerance is unchanged.

Doctor Legs completed its 400-step / 8-second walking-policy case on CPU and CUDA. Ten G1 implicit-drive smoke/soak/refinement rollouts passed their existing checks. Eight bounded-external-torque G1 rollouts (64 simulated seconds / 80,000 steps), two analytical torque probes, CPU/GPU comparisons and duplicate-drive controls were rerun. Supported motion is stable, but **contact accuracy is not cleared**:

- Fixed-pelvis coordinated motion agrees within 0.02549° under 1 ms → 0.5 ms refinement. Guided foot loading differs by **3.0498° maximum / 0.30381° RMS**, exceeding the unchanged 0.573° / 0.201° limits. Controller sampling changes with physics timestep too; root cause is not isolated.
- Guided feet carry 354.49 N normally and 425.86 N with an added 20% downward load. Penetration is 1.266 mm at 1 ms with the explicit contact material above. The initial 2,500 N/m material still fails penetration at about 24.4 mm; no threshold was relaxed.
- Free-base zero-gravity COM drift persists: CUDA measures 1.50 mm at 4 ms and **30.28 mm at 1 ms** over 10 s. The 1 ms result reproduces exactly on untouched #4179 with Warp 1.18. The supported fixtures do not qualify momentum conservation or unsupported balance.

Physics runs and animations use merge checkpoint `45bcab127f2e9784528dbc99eb41dc2284830da9`. The subsequent publication commit changes only seven mimic-test calls to the current keyword-only `coeffs=` API and this evaluation note; solver/model code is identical. The corrected feature suite passed 166 tests with the one inherited assertion above. The first run's 62 API errors are preserved in local evidence and are resolved by that test-call migration.

- Sparse friction remains regularized Coulomb (`-joint_friction*tanh(qd/0.01)`), so there is no projected static-hold guarantee and near-zero creep can occur. This is joint friction, not the foot/contact sliding-friction coefficient. #4179's projected friction, including mimic endpoints, is specific to **local compliant ALM**.
- #4264 (`0dee06d5683aa12821332c8073e3a869dc0e56aa`) is comparison-only. It is not a drop-in sparse-backend implementation and rejects positive friction on either endpoint of supported mimic pairs, even disabled pairs. It was not merged into this branch.
- Sparse armature support is revolute-only; contact Hessians remain body-diagonal. The refresh preserves main's body-particle attachment reactions in both sparse and local-fallback body groups, with a focused CPU/GPU regression. This does not establish a monolithic global rigid/particle/contact solve.
- Supported G1 motion does not establish unsupported balance or actual SONIC behavior. Controller integration, reset/batching, self-collision and policy-level graph capture remain unqualified. The latest independent-agent review is pending.

The G1 fixture uses the public model at GR00T-WholeBodyControl commit `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`, `gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml` (SHA-256 `58c82f77753db54f8a6ca0a8e020e142b015d8587db6cfc442f9a75f2bc444c6`). The external robot harnesses, raw logs and animations are held separately in the local validation workspace; they are not installed by cloning this repository. This branch contains the solver changes and in-repository tests, not a SONIC controller adapter.
