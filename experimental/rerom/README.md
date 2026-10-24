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
   mismatch, IMEX residual), re-linearizes about the current shape `r_new = r_ref + Phi q`.
4. It evaluates the nonlinear internal force and tangent `K_T(r_new)`, computes tangent eigenmodes on the
   complement of the rigid motions, and adds static Ritz vectors (`K_T^-1 f_int`, `K_T^-1 f_ext`, Krylov).
5. A Rayleigh-Ritz rotation makes the basis M-orthonormal and `K_T`-diagonal.
6. Every ROM array of the body (modal mass/stiffness/damping, inertia integrals, rigid properties, endpoint,
   render and contact samples, joint frames) is rewritten in place.
7. The state is reset with `q = 0` and an M-orthogonal, momentum-consistent velocity split.
8. The pre-stress `g0 = Phi^T f_int(r_new)` enters through the modal rows of `control.joint_f`.
9. Craig-Bampton bodies with interface frames use the same machinery with constraint modes from `K_T`.
10. An optional IMEX hyper-reduced remainder adds the lagged nonlinear modal force between updates.

## Layout

| module | contents |
|---|---|
| `rerom/controller.py` | `ReROMController` (one body) with the `BodyAdapter` strategy: `FreeBodyAdapter`, `InterfaceBodyAdapter`; `MultiFrameController` (thin aggregate for CB chains) |
| `rerom/options.py` | keyword-only `BasisOptions`, `TriggerOptions`, `UpdatePolicy`, `NonlinearForceOptions`; enums `RestRotationMode`, `IndefiniteGuard` |
| `rerom/triggers.py` | `RotationTrigger`, `EnergyMismatchTrigger`, `ResidualTrigger` on a shared gating base `Trigger` |
| `rerom/numerics.py` | the shared numerics (rigid vectors, M-orthonormal blocks, Ritz/Krylov enrichment, Rayleigh-Ritz, modal derivatives, state projection, quaternions) |
| `rerom/eigen.py` | the eigensolver seam: `resolve_backend`, `RestrictedArpackBackend`, `RestrictedSubspaceBackend`, `ShiftedRestrictedSolver` |
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
(`newton.EigenSolverBackend` on this branch; a pending rename calls it `newton.ModalEigensolverBackend`, and the
result type `newton.EigenResult` becomes `newton.ModalEigenResult`). Subclassing is not required; the protocol
is duck-typed:

* attributes `name` (str), `supports_inertia_count` (bool) and `device` (`"cpu"` for host backends);
* `solve(stiffness, mass, count, *, deflation=None, sigma=None, tol=0.0, factorization=None, zero_threshold=0.0)`
  returning the eigensolver result (`values`, `vectors`, optional `inertia_count` and `inertia_shift`, `backend`,
  `wall_s`).

```python
import newton


class WarpackBackend:  # duck-typed; newton.WarpackBackend is a placeholder that documents the contract
    name = "warpack"
    supports_inertia_count = False  # True only if solve() returns a Sturm count (e.g. cuDSS LDL^T inertia)

    def __init__(self, device="cuda:0"):
        self.device = device

    def solve(
        self, stiffness, mass, count, *, deflation=None, sigma=None, tol=0.0, factorization=None, zero_threshold=0.0
    ):
        # stiffness, mass: host SciPy CSC (newton.to_backend_matrix converts or passes them through);
        # deflation: M-orthonormal rigid motions (n, 6) of a free body, or None for a clamped body.
        values, vectors = ...  # move to the device, solve, return float64 host arrays
        return newton.EigenResult(values, vectors, inertia_count=None, backend=self.name)


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
`newton.SubspaceIterationBackend`) is warm-started with the previous basis at every update. For Craig-Bampton
bodies an explicitly passed backend also solves the fixed-interface eigenproblem.

The named built-ins `"arpack"` and `"subspace"` resolve to the *restricted* backends of `rerom/eigen.py`
(`RestrictedArpackBackend`, `RestrictedSubspaceBackend`), which use the exact inverse on the rigid-motion
complement; Newton's generic backends project the rigid motions out after a plain shift-invert solve, which is not
the inverse of the restricted pencil for a pre-stressed `K_T`. The restricted backends reproduce the recorded M1/WP3a runs bit for bit.

## Options summary

All option groups are keyword-only dataclasses in `rerom/options.py`; the docstrings there are authoritative.

| group | main fields | default behavior |
|---|---|---|
| `BasisOptions` | `n_ritz`, `modal_derivatives`, `ritz_krylov`, `rebasis`, `shared_factorization`, `subspace_iterations`, `eig_margin`, `damping_ratio` | 2 static Ritz vectors (`K_T^-1 f_int`, `K_T^-1 f_ext`, then Krylov), the rest tangent eigenmodes; one factorization of `K_T - sigma M` per update |
| `TriggerOptions` (on every trigger) | `min_interval`, `max_updates`, `max_interval`, `check_every`, `n_sample_elements`, `element_ids` | check every step, no budget, 512 sampled elements |
| triggers | `RotationTrigger(theta_tol, disp_tol)`, `EnergyMismatchTrigger`, `ResidualTrigger` | `RotationTrigger` at 4 deg element rotation since the last reference |
| `UpdatePolicy` | `path` (`UpdatePath.DEVICE` / `HOST`), `freeze_in_contact`, `freeze_after_release`, `freeze_from_step`, `guard_indefinite` (`IndefiniteGuard.KEEP` / `CLAMP`), `rest_rotation` (`RestRotationMode`), `notify`, `diagnostics` | device path, no freezes, negative tangent eigenvalues written as they are, joint rest rotations kept by counter-rotation |
| `NonlinearForceOptions` | `element_ids`, `weights`, `interval`, `fold_interval`, `manifold` | off unless passed: IMEX hyper-reduced nonlinear remainder between updates (free bodies only) |

## Known limitations

* **Update cadence is limited by linear reconstruction.** Between updates the body moves in a linear basis, so a
  rotating slender segment picks up a spurious stretch of order `(L/h)^2 theta^2`. Slender parts need an update
  every one to a few steps for a few percent tip accuracy; more Ritz vectors, modal derivatives or fresh
  eigenmodes do not change this. Panels and compliant parts in quasi-static motion amortize much better.
* **Indefinite rebuilt tangent.** Under strong compression or contact pressing (for example a cushion press) the
  rebuilt `K_T` can have negative eigenvalues. `IndefiniteGuard.CLAMP` raises them to a positive floor and logs
  it; refusing the update and keeping the previous basis is not implemented. Gate triggers with
  `UpdatePolicy(freeze_in_contact=True)` or settling checks, which also avoids update limit cycles in contact.
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
