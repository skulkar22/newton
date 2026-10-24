# Integrated architecture idea: re-ROM as a flag on the reduced elastic link (for discussion with Miles)

Status: proposal only (2026-10-24). The integrated design below is not implemented. It was written from the
phase-1/2 re-ROM experiments (recorded outside this repository, see section 7). The code in this folder is an
external controller (`rerom.ReROMController`) around `SolverVBD`; this note describes how the same mechanics
would look if owned by Newton.

Since the note was written, several rows of the "Today" column below moved to the device in this package:
`rerom/assembly_warp.py` (`WarpTetAssembler`, Warp tangent assembly into a BSR matrix),
`rerom/update_kernels.py` (`DeviceUpdater`: in-place rewrite kernels, device trigger and pre-stress) and the in-place
`SolverVBD.refresh_elastic_mass_solve_data()`; see [`../README.md`](../README.md) and
[`../BENCHMARKS.md`](../BENCHMARKS.md). The eigensolve and basis construction still run on the host.

## 1. What re-ROM needs, independent of where it lives

| Need | Today (external controller) | Integrated |
|---|---|---|
| Full mesh, material, full nodal basis Φ (N,m,3), mass, mutable reference shape r_ref | Python object outside Newton | Model arrays per adaptive body (`elastic_full_*`), or a `ModalBasis.full_field` record carried through the builder |
| Tangent stiffness K_T(x) and internal force f_int(x) at the deformed shape | numpy StVK assembler (23–34 ms at 6k DOF) | Warp per-tet kernel → `warp.sparse.BsrMatrix`, pattern built once |
| Eigensolve + static Ritz solves | SciPy ARPACK / subspace iteration on host (27–112 ms) | `EigenSolverBackend` seam in `modal_sparse`: SciPy, subspace, Warpack + cuDSS, FSAI-CG + LOBPCG |
| New modes → Newton arrays (five coupling integrals, modal mass matrix, endpoint/render/shape φ, rigid props) | host numpy + `.assign()` (≈66 host round trips per update) | device kernels writing in place into the fixed-size `elastic_*` arrays |
| State transfer at a switch (rigid part → frame twist, rest → new modes), pre-stress g₀ | host; pre-stress via `control.joint_f` modal rows | device kernel; dedicated `elastic_mode_prestress` array so user modal forces keep working |
| Solver cache refresh | `notify_model_changed` reallocates mass-solve data → CUDA graph recapture | in-place refresh (`refresh_elastic_mass_solve_data`), captured step graph stays valid |
| Joint handling when endpoint frames move | counter-rotating `model.body_q` (exact for one endpoint only) | per-joint rest frames owned by the solver (`SolverVBD.set_joint_rest_frames`, patch exists) |
| Trigger | host read of `joint_q` + sampled element rotations | device kernel; gated by contact state and settling; min interval; update budget |

## 2. User-facing shape

```python
basis = newton.ModalGeneratorFEM(..., mode_count=12, full_field=True).build()  # keeps mesh, Φ, M
b = builder.add_body_elastic(xform, modal_basis=basis, adaptive=True)  # opt-in per body
solver = newton.solvers.SolverVBD(
    model,
    elastic_adaptive=newton.ElasticAdaptiveOptions(
        trigger=RotationTrigger(deg=1.0, min_interval=10, freeze_in_contact=True),
        basis=BasisOptions(n_ritz=2, modal_derivatives=0),
        eigensolver=ElasticEigensolver.SUBSPACE,  # or .WARPACK, or a backend instance
        guard_indefinite=True,
    ),
)
solver.step(state_in, state_out, control, contacts, dt)  # nothing else
```

Bodies without `adaptive=True` add no arrays and take the existing fixed-basis path unchanged.

## 3. Step sequence inside SolverVBD

1. Normal VBD step (captured graph) with the body's current Φ, Λ, g₀.
2. Trigger kernel (captured or outside, cheap): element rotation since r_ref on a sampled element set;
   contact/settling gate; min interval. Writes a per-body flag.
3. If any flag set (host reads one int; outside capture):
   a. assemble f_int, K_T on device (per-tet kernel, BSR);
   b. eigensolver backend: lowest m−n_ritz tangent modes with rigid deflation (free body) or interface frames
      held (CB body); Ritz: K_T⁻¹ f_int, K_T⁻¹ f_ext (shared factorization);
   c. small Rayleigh–Ritz re-diagonalization (m×m) so M_red = I, K_red = diag; indefinite guard: if any
      eigenvalue < 0, keep the previous basis for this step and log;
   d. device kernels: r_ref ← r_ref + Φq; g₀ ← Φ_newᵀ f_int; inertia integrals; endpoint/render/shape φ,ψ;
      rigid mass/COM/inertia; momentum-consistent state transfer; q ← 0;
   e. in-place refresh of the elastic mass-solve data; rest frames untouched (solver-owned).
4. Continue stepping the captured graph.

## 4. Why integrated rather than external

- One flag for the user; no per-step hooks; no second object to keep in sync with the solver.
- The solver knows when an update is safe (between steps, outside capture) and owns the caches it must refresh.
- Joint rest frames and the pre-stress array are solver/model concepts; handling them from outside required
  workarounds (`body_q` counter-rotation, `joint_f` hijack).
- One eigensolver seam serves the modal generators and the updater, so Warpack plugs in once.
- Craig–Bampton bodies with interface frames (chains, the headphone band) use the same path: constraint
  modes from K_T with interfaces held, fixed-interface modes from the K_T eigenproblem.

## 5. Costs, risks, open questions for Miles

- Model growth for adaptive bodies (mesh + Φ: ~0.3 MB per 2k-node body at 12 modes; fine).
- SciPy stays an optional host dependency until a device eigensolver exists; the seam must degrade gracefully.
- SolverVBD complexity: the update stage is a few hundred lines plus kernels; keep it in a separate module
  (`ReducedElasticUpdater`) that the solver calls.
- Ownership: does the update belong to the solver (knows timing/caches) or to `sim` (owns the Model)? Proposal:
  sim owns data + numerics, solver owns scheduling and cache refresh.
- Pre-stress array vs `control.joint_f`: a model array is cleaner and leaves user modal forces intact.

## 6. Design constraints established by the experiments (do not re-learn)

- Linear reconstruction between updates caps the interval at a few steps for slender parts (spurious stretch
  ~θ²/2, force ratio ~ (L/h)² θ²); force corrections, more Ritz/Krylov vectors, modal derivatives as linear
  vectors, and basis libraries do not change this; a quadratic-manifold reconstruction in the kernels gives
  3–6×. Panels and compliant parts in quasi-static motion amortize (7–19 updates per 3,000 steps).
- Static Ritz vectors are essential (eigen-only or re-referencing without them locks).
- Triggers must be gated by contact and settling (worn-headphones limit cycle) and updates must be refused
  when the rebuilt tangent is indefinite (cushion press).
- A reduced body jointed to moving bodies at both ends converges slowly in SolverVBD (10³–10⁴ iterations);
  per-joint rest frames are required for updates in articulations.

## 7. Where the evidence is

- In this repository: [`../README.md`](../README.md) (what the package does, options, eigensolver seam, known
  limitations), [`../BENCHMARKS.md`](../BENCHMARKS.md) (update cost before/after the device-side update, tangent
  assembly, CUDA graph validity, acceptance against the recorded runs), the tests under [`../tests/`](../tests/)
  and the runnable comparison [`../examples/beam_swing.py`](../examples/beam_swing.py).
- The per-joint rest frames referenced above are used through `RestRotationMode.REST_FRAMES`
  (`rerom/options.py`), which needs `SolverVBD.set_joint_rest_frames`.
- The experiment records behind sections 5 and 6 (findings record, audited report, formulation notes, the
  block-sparse fix note and the rest-frame patch) live in the separate re-ROM experiment report, not in this
  repository.
