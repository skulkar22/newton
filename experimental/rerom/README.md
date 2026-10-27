# re-ROM: adaptive re-linearization of reduced elastic bodies (experimental)

> **Status: experimental.** This folder lives outside the `newton` package: it is not installed with Newton,
> is not part of Newton's public API and has no API stability guarantee. Names, options and behavior can change
> or disappear without notice.

A research package built on Newton's reduced elastic (FFRF) bodies and `SolverVBD`, kept next to the code it
exercises. The eigensolve and the basis construction run on the host (numpy/SciPy, or a pluggable eigensolver
backend); the rest of an update, the per-step trigger and the pre-stress run as Warp kernels on the Model's device.

**In five lines.** A reduced elastic body deforms with a fixed linear modal basis, which goes wrong under large
rotation and stress. re-ROM watches the body with a cheap trigger; when it fires, it re-linearizes about the
current shape, builds a new basis from tangent eigenmodes plus static Ritz vectors, and rewrites the body's
reduced arrays and state in place. Between updates Newton steps the body exactly as a fixed-basis body. The
result tracks large deformation far better than one fixed basis, at the cost of frequent updates.

## What re-ROM is

1. A Newton reduced elastic body moves its nodes as `x = p + R (r_ref + Phi q)` with a *linear* modal basis.
2. Large deformation makes that basis wrong: linear modes stretch under rotation, and the tangent stiffness
   changes with stress.
3. re-ROM keeps the full nodal field outside the Model and, when a trigger fires (element rotation, energy
   mismatch, IMEX residual, validity), re-linearizes about the current shape `r_new = r_ref + Phi q`.
4. It evaluates the nonlinear internal force and tangent `K_T(r_new)`, computes tangent eigenmodes on the
   complement of the rigid motions, and adds static Ritz vectors (`K_T^-1 f_int`, `K_T^-1 f_ext`, Krylov).
5. A Rayleigh-Ritz rotation makes the basis M-orthonormal and `K_T`-diagonal.
6. Every ROM array of the body (modal mass/stiffness/damping, inertia integrals, rigid properties, endpoint,
   render and contact samples, joint frames) is rewritten in place.
7. The state is reset with `q = 0` and an M-orthogonal, momentum-consistent velocity split.
8. The pre-stress `g0 = Phi^T f_int(r_new)` enters through the modal rows of `control.joint_f`.
9. Craig-Bampton bodies with interface frames use the same machinery with constraint modes from `K_T`.
10. An optional IMEX hyper-reduced remainder adds the lagged nonlinear modal force between updates.
11. Every update is a transaction: the candidate is built without writing anything, validated, then committed
    or rejected with a reason code (see "Update safety and diagnostics").

## Layout

| module | contents |
|---|---|
| `rerom/controller.py` | `ReROMController` (one body) with the `BodyAdapter` strategy: `FreeBodyAdapter`, `InterfaceBodyAdapter`; `MultiFrameController` (thin aggregate for CB chains) |
| `rerom/options.py` | keyword-only `BasisOptions`, `TriggerOptions`, `UpdatePolicy`, `ValidationOptions`, `NonlinearForceOptions`; enums `RestRotationMode`, `IndefiniteGuard`, `TransferMode`, `RejectReason` |
| `rerom/triggers.py` | `RotationTrigger`, `EnergyMismatchTrigger`, `ResidualTrigger`, `ValidityTrigger`, `CompositeTrigger` on a shared gating base `Trigger` |
| `rerom/numerics.py` | the shared numerics (rigid vectors, M-orthonormal blocks, Ritz/Krylov enrichment, Rayleigh-Ritz, modal derivatives, state projection, quaternions) |
| `rerom/eigen.py` | the eigensolver seam: `resolve_backend`, `RestrictedArpackBackend`, `RestrictedSubspaceBackend`, `ShiftedRestrictedSolver` |
| `rerom/eigen_warpack.py` | optional Warpack + cuDSS backends (`WarpackKrylovBackend`, `WarpackLOBPCGBackend`) and the GPU restricted-solver factory (`WarpackContext`) |
| `rerom/model_io.py` | every `Model`/`State`/`Control` array read and write of the host path (batched), joint rest-rotation propagation |
| `rerom/update_kernels.py` | device-side update (`DeviceUpdater`): recovery, packed gather/scatter, inertia integrals, sample/endpoint rewrites, state projection, per-step trigger and pre-stress kernels |
| `rerom/assembly.py` | `TangentAssembler` protocol, `NumpyTetAssembler`, meshes, `Material`, `default_assembler` |
| `rerom/assembly_warp.py` | `WarpTetAssembler` (CPU/CUDA tangent assembly, see `BENCHMARKS.md`) |
| `rerom/craig_bampton.py` | CB basis with interface frames, segments, chains, linear statics |
| `rerom/hyper_reduction.py` | element-subset modal force (ECSW, stratified sampling) for the IMEX remainder |
| `rerom/fe_reference.py` | `FullFEReference` (implicit Newmark FE), `BaseMotion`, `PrescribedMotionReference` |
| `rerom/strain.py` | strain measures and strain-comparison metrics |
| `rerom/ccx.py`, `rerom/panel.py` | CalculiX-coupled shell/thin-solid panels (need a `ccx` executable) |
| `examples/beam_swing.py` | runnable comparison: fixed-basis FFRF vs re-ROM vs an implicit FE reference |
| `benchmarks/` | `bench_update.py` (per-update cost), `bench_assembly_warp.py` (tangent assembly); see `BENCHMARKS.md` |
| `docs/ARCHITECTURE.md` | proposal for integrating re-ROM into Newton as a flag on the reduced elastic body |
| `tests/` | unittest suite (CPU; CUDA and CalculiX cases when available) |

## How to run

From the Newton checkout (the package directory is put on `sys.path`):

```python
import sys

sys.path.insert(0, "experimental/rerom")
import newton, rerom

asm = rerom.default_assembler(mesh)  # Warp on CUDA when available, else numpy
ctrl = rerom.ReROMController(
    model,
    solver,
    body,
    asm,
    rerom.FreeBodyAdapter(gen.modal_matrix, mesh.nodes, fixed_nodes=root_nodes),
    basis=rerom.BasisOptions(n_ritz=2),
    trigger=rerom.RotationTrigger(theta_tol=np.radians(0.2)),
    eigensolver="arpack",  # or "subspace", "dense", or an eigensolver backend (see below)
)
for _ in range(steps):
    ctrl.apply_prestress(control, state_0)
    state_0.clear_forces()
    solver.step(state_0, state_1, control, None, dt)
    state_0, state_1 = state_1, state_0
    ctrl.after_step(state_0, control, dt)
```

All commands run from the Newton checkout. The package needs SciPy, which Newton's `dev` extra provides.

Tests (CPU; the CUDA cases run when a device is present, the CalculiX cases when `ccx` is found or
`$REROM_CCX` points to it):

```bash
uv run --extra dev python -m unittest discover -s experimental/rerom/tests -t experimental/rerom/tests
```

Example (a soft cantilever released from the horizontal; under 10 s with a warm Warp kernel cache, longer on
the first run while kernels compile; `--help` lists the mesh, trigger, eigensolver and step options):

```bash
uv run --extra dev python experimental/rerom/examples/beam_swing.py              # Model on cpu
uv run --extra dev python experimental/rerom/examples/beam_swing.py --device cuda:0
```

It prints the tip error of the fixed-basis and the re-ROM run against `FullFEReference` (implicit Newmark FE
on the same mesh), the number of updates, the mean update cadence and the per-update time. With the defaults
(189 nodes, 12 modes, 0.5 deg rotation trigger, subspace eigensolver, 300 steps of 2 ms) the recorded runs
gave a tip RMS error of 0.24 L for the fixed basis and 0.0075 L for re-ROM, with 109 updates (one every
2.8 steps) at about 4 ms each, on both `cpu` and `cuda:0`.

Benchmarks (see `BENCHMARKS.md` for the tables and their environment):

```bash
uv run --extra dev python experimental/rerom/benchmarks/bench_update.py --device cpu --out bench_update_cpu.json
uv run --extra dev python experimental/rerom/benchmarks/bench_assembly_warp.py --help
```

## Device-side update and CUDA graphs

`UpdatePolicy(path=UpdatePath.DEVICE)` (default) keeps everything except the eigensolve on the Model's
device: one packed device-to-host copy at the start of an update (modal state, `r_new = r_ref + Phi q`,
`f_int`, the BSR values of `K_T` from a `WarpTetAssembler` on the same device), the host builds the basis,
one packed host-to-device copy brings it back, and Warp kernels rewrite the inertia integrals, modal mass
matrix, render/shape/endpoint samples, rigid properties and the projected state in place. The solver is
refreshed with `SolverVBD.refresh_elastic_mass_solve_data()` (in place), so a CUDA graph captured around
the step stays valid across updates:

```python
with wp.ScopedCapture() as capture:
    ctrl.apply_prestress(control)  # one kernel: modal rows of control.joint_f
    state_0.clear_forces()
    solver.step(state_0, state_1, control, None, dt)
    ...
for _ in range(steps):
    wp.capture_launch(capture.graph)
    fire, reason, metrics = ctrl.poll(state_0, dt)  # device trigger: one 3-value read
    if fire:
        ctrl.relinearize(state_0, control)  # outside the graph; no recapture needed
```

`UpdatePolicy(path=UpdatePath.HOST)` is the legacy numpy path; with
`BasisOptions(shared_factorization=False)` it reproduces the recorded runs bit for bit. The device path
agrees with it to roundoff (see `BENCHMARKS.md` for the acceptance runs and timings).
`BasisOptions.shared_factorization` (default `True`) factors `K_T - sigma M` once per update for the
eigensolve, the Ritz vectors and the modal derivatives.

## The eigensolver seam: plugging in a GPU backend (Warpack + cuDSS, FSAI-CG + LOBPCG)

The controller's `eigensolver` argument takes a name (`"arpack"`, `"subspace"`, `"dense"`), a
`newton.ModalEigensolver`, or any object that satisfies Newton's eigensolver backend protocol
(`newton.ModalEigensolverBackend`; the result type is `newton.ModalEigenResult`). Subclassing is not
required; the protocol is duck-typed:

* attributes `name` (str), `supports_inertia_count` (bool) and `device` (`"cpu"` for host backends);
* `solve(stiffness, mass, count, *, deflation=None, sigma=None, tol=0.0, factorization=None, zero_threshold=0.0)`
  returning the eigensolver result (`values`, `vectors`, optional `inertia_count` and `inertia_shift`, `backend`,
  `wall_s`).

```python
import newton


class WarpackBackend:  # duck-typed; newton._src.sim.modal_sparse.WarpackBackend is a placeholder documenting the contract
    name = "warpack"
    supports_inertia_count = False  # True only if solve() returns a Sturm count (e.g. cuDSS LDL^T inertia)

    def __init__(self, device="cuda:0"):
        self.device = device

    def solve(
        self, stiffness, mass, count, *, deflation=None, sigma=None, tol=0.0, factorization=None, zero_threshold=0.0
    ):
        # stiffness, mass: host SciPy CSC (newton._src.sim.modal_sparse.to_backend_matrix converts or passes them through);
        # deflation: M-orthonormal rigid motions (n, 6) of a free body, or None for a clamped body.
        values, vectors = ...  # move to the device, solve, return float64 host arrays
        return newton.ModalEigenResult(values, vectors, inertia_count=None, backend=self.name)


ctrl = rerom.ReROMController(..., eigensolver=WarpackBackend("cuda:0"))
```

What the backend receives and must return:

* Input: the free-DOF pencil `(K_T, M)` at the new reference (the tangent stiffness is pre-stressed, so it can be
  indefinite and does not have the rigid motions in its null space), `count`, the rigid-motion `deflation` for a
  free body, and a shift `sigma` at or below the lowest wanted eigenvalue. The host `factorization` is `None` for
  a non-`"cpu"` device.
* Output: at least `count` eigenpairs outside `deflation`, eigenvalues ascending, eigenvectors M-orthonormal and
  M-orthogonal to `deflation`, as float64 host arrays; raise `ValueError` on non-convergence.
* Sturm check: with `supports_inertia_count = True` the backend returns `inertia_count` and `inertia_shift`, and
  the result is checked for missed eigenvalues. With `False` (or `inertia_count=None`) the check is skipped with
  a warning, so missed or repeated eigenvalues are not detected.

The controller validates the result (`modal_sparse.validate_eigen_result`), adds its static Ritz vectors and
re-diagonalizes (Rayleigh-Ritz). A backend with a `previous_vectors` attribute (like
`newton.ModalSubspaceIterationBackend`) is warm-started with the previous basis at every update. For Craig-Bampton
bodies an explicitly passed backend also solves the fixed-interface eigenproblem.

The named built-ins `"arpack"` and `"subspace"` resolve to the *restricted* backends of `rerom/eigen.py`
(`RestrictedArpackBackend`, `RestrictedSubspaceBackend`), which use the exact inverse on the rigid-motion
complement; Newton's generic backends project the rigid motions out after a plain shift-invert solve, which is not
the inverse of the restricted pencil for a pre-stressed `K_T`. The restricted backends reproduce the recorded M1/WP3a runs bit for bit.

### Warpack backends (optional, experimental): `rerom/eigen_warpack.py`

[Warpack](https://github.com/alecjacobson/warpack) is Alec Jacobson's device-resident sparse eigensolver library for
Warp (MIT licence). `rerom/eigen_warpack.py` runs it on the update's pencil, with cuDSS 0.8 for the sparse
factorization through Warpack's own `CuDSSInverse` and its `warp-cudss` binding (`pip install '<warpack checkout>[cudss]'`).
All of it is optional and imported lazily (a clear `ImportError` otherwise):

```python
from rerom import eigen_warpack as ew

ctx = ew.WarpackContext(device="cuda:0", assembler=warp_assembler, free_dofs=free_dofs)  # one per body
# factorization="auto" (default): "warpack" (CuDSSInverse) when warp-cudss is importable, else "nvmath" (adapter)
ctrl = rerom.ReROMController(
    ...,
    basis=rerom.BasisOptions(n_ritz=2, restricted_solver_factory=ctx),
    eigensolver=ew.WarpackKrylovBackend(ctx),  # or ew.WarpackLOBPCGBackend(ctx) (warm-started)
)
```

* `WarpackKrylovBackend`: `warpack.GeneralizedKrylovSchur`, shift-invert, M inner product, cold start; Sturm count
  from the cuDSS `L D L^T` inertia (Haynsworth-corrected for the rigid restriction); the shift is lowered when the
  tangent has eigenvalues below it, so negative eigenvalues are returned.
* `WarpackLOBPCGBackend`: `warpack.BlockLOBPCG` warm-started from the previous basis, the shift-invert inverse as
  preconditioner.
* `BasisOptions.restricted_solver_factory=ctx`: the update's static Ritz and modal-derivative solves use the same
  GPU factorization (one numeric factorization per update; the symbolic analysis and all Warpack workspaces are
  kept across updates). With the device update path and a `WarpTetAssembler` the tangent goes from the assembler's
  device BSR values to the solver without a host copy.
* The pencil is posed with the projected stiffness `P^T K_T P` (`P = I - Y (M Y)^T`) and the exact restricted
  inverse, because a pre-stressed `K_T` does not keep the rigid motions invariant (Warpack's own rigid-mode
  `constraints` assume they are).

Factorization (`WarpackContext(factorization=...)`):

* `"warpack"` (`NativeCudssFactor`, the default when available): Warpack's `CuDSSInverse` on its pinned `warp-cudss`
  binding (`e1d1be41`, cuDSS 0.8.0.10, `nvidia-cudss-cu12`). One setup per pattern, then per update the binding's
  numeric refactorization; the Sturm count reads `CuDSSInverse.inertia()`. `CuDSSInverse` fixes the right-hand-side
  width at setup (1 here, what Krylov-Schur applies), so the rigid restriction, the LOBPCG preconditioner and the Ritz
  and modal-derivative solves run row by row on the same factorization (no second factorization). No residual
  correction by default (`refinement_steps=0`): Warpack's default single step for indefinite factors broke the rigid
  restriction's cancellation at the near-singular free-body shift (Krylov-Schur stalled at 9/12 pairs on the free
  test beam).
* `"nvmath"` (`CudssFactor`): this package's adapter of Warpack's inverse protocol on cuDSS 0.8.0 through
  nvmath-python (`nvidia-cudss-cu13`), with one block solve for any width. It was the only path before the binding
  was installable, stays as the alternative and is the fallback of `"auto"`. Both binaries are `libcudss.so.0`: in
  one process the first one loaded serves both.

Both pass the same accuracy gates, and the Krylov-Schur update costs the same with either. Warpack's own test suite
with the binding: 120 passed, 3 failed, 2 skipped on the 11-file subset that previously gave 86 passed, 37 failed
(`No module named 'warp_cudss'`). The 3 failures are CuPy reference tests (`libcublas.so.13` missing in that
environment). Full suite: 3,179 passed, 93 failed, 287 skipped. None of the failures involves cuDSS: they are missing
upstream Warp FSAI/block-Jacobi, Ada-only specializations on this sm_120 GPU, and the CuPy libraries. When cuDSS reports
perturbed pivots in the Sturm factorization the count is not certified: the backend returns no inertia count and the
Sturm check is skipped with a warning. Warpack is by Alec Jacobson; the `warp-cudss` binding comes from his
repository `alecjacobson/warp-cuDSS`.

Without these options nothing changes (the defaults are bit-identical). Measured accuracy and cost: `BENCHMARKS.md`
section 3.

## Options summary

All option groups are keyword-only dataclasses in `rerom/options.py`; the docstrings there are authoritative.

| group | main fields | default behavior |
|---|---|---|
| `BasisOptions` | `n_ritz`, `modal_derivatives`, `ritz_krylov`, `rebasis`, `shared_factorization`, `subspace_iterations`, `eig_margin`, `damping_ratio`, `velocity_enrichment` | 2 static Ritz vectors (`K_T^-1 f_int`, `K_T^-1 f_ext`, then Krylov), the rest tangent eigenmodes; one factorization of `K_T - sigma M` per update; no velocity enrichment |
| `TriggerOptions` (on every trigger) | `min_interval`, `max_updates`, `max_interval`, `check_every`, `n_sample_elements`, `element_ids`, `audit_suppressed` | check every step, no budget, 512 sampled elements; suppressed alarms recorded |
| triggers | `RotationTrigger(theta_tol, disp_tol)`, `EnergyMismatchTrigger`, `ResidualTrigger`, `ValidityTrigger`, `CompositeTrigger(triggers=[...])` | `RotationTrigger` at 0.5 deg element rotation since the last reference (4 deg before; see below) |
| `UpdatePolicy` | `path` (`UpdatePath.DEVICE` / `HOST`), `freeze_in_contact`, `freeze_after_release`, `freeze_from_step`, `guard_indefinite` (`IndefiniteGuard.KEEP` / `CLAMP` / `REJECT`), `rest_rotation` (`RestRotationMode`), `notify`, `diagnostics`, `validation` | device path, no freezes, negative tangent eigenvalues written as they are, joint rest rotations kept by counter-rotation, default validation |
| `ValidationOptions` (`UpdatePolicy.validation`) | `det_f_floor`, `orthonormality_tol`, `rank_tol`, `eigen_residual_tol`, `negative_stiffness_tol`, `interface_cond_max`, `transfer_energy_budget`, `backoff_steps`, `max_consecutive_rejections` | reject non-finite candidates, `det F <= 0`, non-positive mass, rank-deficient or non-M-orthonormal bases, ill-conditioned interface transfers; no eigen-residual or energy budget; 10-step back-off, stop after 5 rejections in a row |
| `NonlinearForceOptions` | `element_ids`, `weights`, `interval`, `fold_interval`, `manifold` | off unless passed: IMEX hyper-reduced nonlinear remainder between updates (free bodies only) |

**Trigger tolerance.** The rotation angle is a proxy, and the tolerance it needs depends on geometry and
slenderness (spurious stretch energy `~ (L/h)^2 theta^2`), mode content and time step. On the coarse beam of
`examples/beam_swing.py` (189 nodes, 12 modes, E = 10 MPa, 0.6 s against backward-Euler FE) an external review
measured 16.95 % L tip RMS error with the former 4 deg default, 0.22 % L at 0.5 deg and 0.03 % L at 0.2 deg; more
VBD iterations, a smaller step or ARPACK did not repair 4 deg. The default is therefore 0.5 deg; pass
`theta_tol=math.radians(4.0)` for the former behavior. A rotation criterion cannot see stretch without
rotation; combine triggers where that matters:

```python
trigger = rerom.CompositeTrigger(
    triggers=[
        rerom.RotationTrigger(theta_tol=math.radians(0.5)),
        rerom.EnergyMismatchTrigger(tol=0.05),
        rerom.ValidityTrigger(det_f_floor=0.0),  # runs before, and is never suppressed by, the gating
    ],
    options=rerom.TriggerOptions(min_interval=1),
)
```

## Update safety and diagnostics

**Raw and operational spectra.** `controller.lam_raw` is the restricted tangent spectrum of the last update
(negative values kept, `info["eigenvalues"]`); `controller.lam` is the operational spectrum the solver steps
(`info["eigenvalues_operational"]`). Every consumer uses the operational one: the written modal stiffness and
damping, `rom_strain_energy`, the IMEX linear term and remainder (`f_tot'(0) = Lambda_op`), `ResidualTrigger`.
With `IndefiniteGuard.KEEP` they are equal; `CLAMP` makes the clamped spectrum the model (a regularization, logged
in `controller.clamp_log`); `REJECT` refuses an update whose restricted tangent has a materially negative
eigenvalue. Tangent folds (`NonlinearForceOptions.fold_interval`) apply the same guard: `eig T` is the raw spectrum;
with `CLAMP` the guarded spectrum is written and the IMEX split is shifted in the clamped directions so that the
force and the model energy at the fold point are unchanged and the remainder does not restore the negative
stiffness (`clamp_log` entries with `"source": "fold"`); with `REJECT` the fold is skipped and recorded in
`controller.folds` (`"skipped": "negative_stiffness"`).

**Transactional updates.** `relinearize` builds the whole candidate (reference, basis, spectra, pre-stress,
inertia, endpoint/sample maps, transferred state, adapter state such as the interface frame map and endpoint
offset) without touching the Model, the State, the controller or the adapter, checks it against
`ValidationOptions`, then commits it or returns `{"accepted": False, "reason": ..., "stage": ...}` with a
`RejectReason` code. A rejection changes nothing (tested bit for bit on both update paths, CPU and CUDA), is
logged in `controller.rejections`, and makes the triggers back off for `backoff_steps` steps; after
`max_consecutive_rejections` in a row updates stop until `controller.request_update()`. Accuracy alarms that a
gate (min interval, budget, freeze, back-off) suppresses are counted in `controller.accuracy_unverified`
(count, last step, reason, gate); validity alarms in `controller.validity_alarms`.

**Transfer diagnostics.** Each update's `info` reports the `transfer_mode` (`TransferMode.FREE`,
`PRESCRIBED_FRAME`, `INTERFACE_CONSTRAINED`), `KE_before`, `KE_after` and the signed `delta_KE` [J],
`velocity_projection_residual_energy` (`1/2 e^T M e`), the absolute and relative momentum changes (`dP`, `dL`,
`dP_rel`, `dL_rel`), the maximum nodal position change, the interface velocity mismatch, the minimum `det F` of the
new reference and the raw / operational minimum eigenvalues. Only the free mode guarantees
`delta_KE = -velocity_projection_residual_energy`; the interface-constrained correction is affine and can raise
the kinetic energy. `KE_lost_projection` is kept as an alias of the residual and is an energy loss only in the
free mode. The constrained correction minimizes the kinetic energy in the reduced metric `Phi^T M Phi`.
`W_rom_quadratic` (alias `W_rom`) is the quadratic model energy at the update point and `W_rom_model` the
corrected (IMEX) model energy.

**Velocity enrichment.** `BasisOptions(velocity_enrichment=True)` (off by default) adds the M-orthogonal
residual of the old elastic velocity to the candidate space, evicting its highest-frequency Rayleigh-Ritz
direction, so the transfer keeps the current velocity within the fixed mode count.

**Not implemented.** Rolling the solver state back and retrying a step with a smaller time step after a
rejection (it needs a Newton-side checkpoint of the state, contacts and solver caches): a rejection keeps the
previous basis, which does not make the already advanced state accurate. Static load directions for contact
or other applied loads in the basis (only `f_int`, gravity / `external_force_fn`, velocity and acceleration
candidates are used). A matched ROM/FE contact validation case.

## Known limitations

* **Update cadence is limited by linear reconstruction.** Between updates the body moves in a linear basis, so a
  rotating slender segment picks up a spurious stretch of order `(L/h)^2 theta^2`. Slender parts need an update
  every one to a few steps for a few percent tip accuracy; more Ritz vectors, modal derivatives or fresh
  eigenmodes do not change this. Panels and compliant parts in quasi-static motion amortize much better.
* **Indefinite rebuilt tangent.** Under strong compression or contact pressing (for example a cushion press) the
  rebuilt `K_T` can have negative eigenvalues. `IndefiniteGuard.CLAMP` raises them to a positive floor (the
  clamped spectrum becomes the operational model) and `IndefiniteGuard.REJECT` refuses the update and keeps the
  previous basis; neither resolves a physical instability, and there is no rollback/retry. Gate triggers with
  `UpdatePolicy(freeze_in_contact=True)` or settling checks, which also avoids update limit cycles in contact.
* **The trigger is an accuracy proxy.** No rotation angle is safe for every geometry (see "Trigger tolerance");
  the energy and validity checks are sampled; a reduced-force or sampled residual estimator that sees load
  directions outside the basis is not implemented.
* **The host eigensolve dominates an update.** With the device path the sparse factorization and eigensolve on
  the host are most of the update time (see `BENCHMARKS.md`); a GPU backend plugs into the seam above but none
  ships here.
* **Basis construction is still host-side.** The Ritz vectors, the Rayleigh-Ritz projection and
  orthonormalization run on the host after one packed device-to-host copy, and
  `SolverVBD.refresh_elastic_mass_solve_data()` still computes its small Schur-form inverse on the host.
* **Joint frames.** `RestRotationMode.COUNTER_ROTATE` is exact only when all endpoints of a body rotate equally;
  articulations with updates at both ends need per-joint rest frames (`RestRotationMode.REST_FRAMES`, which
  requires `SolverVBD.set_joint_rest_frames`).
* The CalculiX panel modules need an external `ccx` executable.

## Further reading

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): a proposal for integrating re-ROM into Newton as a flag on
  the reduced elastic body, the solver step sequence, and the design constraints the experiments established.
* [`BENCHMARKS.md`](BENCHMARKS.md): per-update cost before and after the device-side update, tangent assembly,
  CUDA graph validity and acceptance against the recorded runs.

## Provenance

The package was migrated from a frozen experiment package used in the re-ROM study and validated against its
recorded runs: the M1 beam swing and the WP3a best configuration are reproduced bit for bit by the host path
with `BasisOptions(shared_factorization=False)`, and to roundoff (at most 2.0e-6 L tip difference) by the default
device path (`BENCHMARKS.md`, "Acceptance"). The recorded runs, the migration mapping and the full results are
in the re-ROM experiment report, which is kept outside this repository.
