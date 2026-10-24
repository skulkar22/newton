# re-ROM: adaptive re-linearization of reduced elastic bodies (experimental)

**Status: experimental.** This folder is *not* part of the `newton` package, has no API stability
guarantee and is not installed with Newton. It is a research package built on Newton's reduced elastic
(FFRF) bodies and `SolverVBD`, kept next to the code it exercises: the eigensolve and the basis
construction run on the host (numpy/SciPy, or a pluggable GPU backend); the rest of an update, the
per-step trigger and the pre-stress run as Warp kernels on the Model's device.

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
    eigensolver="arpack",  # or "subspace", "dense", a newton.EigenSolverBackend
)
for _ in range(steps):
    ctrl.apply_prestress(control, state_0)
    state_0.clear_forces()
    solver.step(state_0, state_1, control, None, dt)
    state_0, state_1 = state_1, state_0
    ctrl.after_step(state_0, control, dt)
```

Tests (CPU; the CUDA cases run when a device is present, the CalculiX cases when `ccx` is found or
`$REROM_CCX` points to it):

```bash
uv run python -m unittest discover -s experimental/rerom/tests -t experimental/rerom/tests
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

## Plugging in a GPU eigensolver (Warpack + cuDSS, FSAI-CG + LOBPCG)

The controller accepts any object that satisfies `newton.EigenSolverBackend` (`name`,
`supports_inertia_count`, `device`, `solve(stiffness, mass, count, *, deflation, sigma, tol, factorization,
zero_threshold) -> newton.EigenResult`):

```python
class WarpackBackend(newton.WarpackBackend):  # or any duck-typed class
    def solve(
        self, stiffness, mass, count, *, deflation=None, sigma=None, tol=0.0, factorization=None, zero_threshold=0.0
    ):
        ...  # stiffness/mass: host SciPy CSC (newton.to_backend_matrix passes them through for a
        # non-"cpu" device); deflation: M-orthonormal rigid motions (n, 6) or None (clamped body)
        return newton.EigenResult(values, vectors, inertia_count=None, backend=self.name)


ctrl = rerom.ReROMController(..., eigensolver=WarpackBackend(device="cuda:0"))
```

The backend receives the free-DOF pencil `(K_T, M)` at the new reference and must return at least `count`
ascending, M-orthonormal eigenpairs orthogonal to `deflation`; the controller validates the result
(`modal_sparse.validate_eigen_result`; without an inertia count the Sturm check is skipped with a warning),
then adds its Ritz vectors and re-diagonalizes. A backend with a `previous_vectors` attribute (like
`newton.SubspaceIterationBackend`) is warm-started with the previous basis at every update. For
Craig-Bampton bodies an explicitly passed backend also solves the fixed-interface eigenproblem.
The named built-ins `"arpack"` and `"subspace"` resolve to the *restricted* backends of `rerom/eigen.py`,
which use the exact inverse on the rigid-motion complement (a pre-stressed `K_T` does not have the rigid
motions in its null space); they reproduce the recorded M1/WP3a runs bit for bit.

## Provenance

Migrated from the frozen experiment package `exp-re-rom/adaptive_rom/` (mapping in
`exp-re-rom/notes/p2_wpC_migration.md`); the M1 beam swing and the WP3a best configuration are reproduced
bit for bit by the new package (`exp-re-rom/results/p2_wpC/acceptance.json`).
