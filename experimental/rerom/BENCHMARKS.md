# re-ROM benchmarks

1. [Update cost before/after the device-side update](#1-update-cost) (WP-D.2-5: device kernels, shared
   factorization, in-place solver refresh, device trigger and pre-stress).
2. [Tangent assembly](#2-tangent-assembly-warptetassembler) (WP-D.1: `WarpTetAssembler`).

Environment: Intel Core Ultra 9 285K (24 cores), NVIDIA RTX PRO 6000 Blackwell Workstation Edition
(driver 595.99.02), Python 3.12.14, numpy 2.5.3, SciPy 1.18.1, Warp 1.18.0. Sub-millisecond GPU numbers
and differences below ~10 % between runs are within run-to-run noise (single machine, no pinning).

## 1. Update cost

### Command

```bash
python experimental/rerom/benchmarks/bench_update.py --device cpu --out bench_update_cpu.json
python experimental/rerom/benchmarks/bench_update.py --device cuda:0 --out bench_update_cuda.json
```

Model: cantilever clamped at its root face (12 modes, fixed root joint, gravity), set to the linear static
gravity deflection scaled to a 5 % L tip deflection before each update, `SolverVBD(iterations=4)`. Median of
5 updates after warm-up. `--device` is the Newton Model device (and the `WarpTetAssembler` device); the
eigensolve and the basis construction run on the host in every row.

Paths:

* **before**: legacy host path (`UpdatePath.HOST`, `BasisOptions(shared_factorization=False)`), numpy
  tangent assembler.
* **warp asm**: the same with `WarpTetAssembler` on the Model device (WP-D.1 alone).
* **after**: device-side update (`UpdatePath.DEVICE`), shared factorization, `WarpTetAssembler`.

Columns are `info["timings"]` (ms): *assembly* (tangent + internal force, incl. recovery and the gather),
*factorization* (all sparse factorizations of the update), *eigensolve*, *projection* (Ritz vectors, modal
derivatives, orthonormalization and state projection), *integrals* (inertia integrals and modal mass),
*rewrite* (Model writes and solver refresh), *total*; *wall* is the wall time of `relinearize` followed by a
device synchronization. *Round trips* count `wp.array.numpy()` reads (d2h), `assign`s from host data (h2d)
and host-initialized `wp.array` constructions (alloc) per update (`rerom.update_kernels.count_host_transfers`).
*Per step* is `apply_prestress` + `poll` (velocity history and rotation trigger at 2 deg) with no update
firing, and its transfers.

### CPU (Model and assembler on `cpu`)

| case | eigensolver | path | assembly | factor. | eigensolve | projection | integrals | rewrite | **total** | wall | round trips d2h / h2d / alloc | per step [ms] | per-step d2h / h2d |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| beam 6,075 DOF | ARPACK | before | 26.4 | 42.2 | 37.5 | 11.0 | 1.9 | 1.2 | **120.1** | 121.3 | 52 / 24 / 4 | 0.856 | 3 / 1 |
| beam 6,075 DOF | ARPACK | warp asm | 8.2 | 45.1 | 40.9 | 11.5 | 2.0 | 1.3 | **108.9** | 110.0 | 57 / 25 / 4 | 0.148 | 4 / 3 |
| beam 6,075 DOF | ARPACK | after | 5.1 | 25.5 | 42.0 | 8.8 | 3.6 | 0.5 | **85.6** | 86.7 | 17 / 5 / 0 | 0.155 | 1 / 0 |
| beam 6,075 DOF | subspace | before | 25.1 | 17.6 | 11.0 | 8.4 | 1.7 | 1.2 | **65.0** | 66.0 | 52 / 24 / 4 | 0.802 | 3 / 1 |
| beam 6,075 DOF | subspace | warp asm | 7.7 | 18.6 | 11.7 | 8.4 | 1.8 | 1.2 | **49.6** | 50.6 | 57 / 25 / 4 | 0.137 | 4 / 3 |
| beam 6,075 DOF | subspace | after | 5.0 | 23.3 | 10.9 | 8.3 | 3.4 | 0.5 | **51.7** | 52.8 | 17 / 5 / 0 | 0.144 | 1 / 0 |
| panel 22,509 DOF | ARPACK | before | 112.2 | 835.3 | 1559.0 | 66.8 | 7.3 | 8.6 | **2573.5** | 2580.0 | 52 / 24 / 4 | 1.070 | 3 / 1 |
| panel 22,509 DOF | ARPACK | warp asm | 36.2 | 843.3 | 1591.8 | 67.4 | 8.5 | 8.1 | **2566.4** | 2573.1 | 57 / 25 / 4 | 0.239 | 4 / 3 |
| panel 22,509 DOF | ARPACK | after | 20.5 | 438.2 | 1534.2 | 104.1 | 14.1 | 0.7 | **2134.3** | 2145.3 | 17 / 5 / 0 | 0.299 | 1 / 0 |
| panel 22,509 DOF | subspace | before | 112.9 | 390.1 | 82.8 | 66.2 | 6.9 | 9.5 | **668.3** | 675.0 | 52 / 24 / 4 | 1.074 | 3 / 1 |
| panel 22,509 DOF | subspace | warp asm | 38.7 | 399.1 | 86.9 | 65.1 | 7.2 | 8.6 | **612.1** | 630.9 | 57 / 25 / 4 | 0.246 | 4 / 3 |
| panel 22,509 DOF | subspace | after | 20.6 | 429.7 | 87.6 | 75.9 | 12.9 | 0.6 | **624.7** | 634.8 | 17 / 5 / 0 | 0.315 | 1 / 0 |

### cuda:0 (Model and assembler on `cuda:0`)

| case | eigensolver | path | assembly | factor. | eigensolve | projection | integrals | rewrite | **total** | wall | round trips d2h / h2d / alloc | per step [ms] | per-step d2h / h2d |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| beam 6,075 DOF | ARPACK | before | 27.1 | 43.5 | 38.2 | 10.9 | 1.9 | 2.2 | **123.9** | 125.2 | 52 / 24 / 4 | 0.976 | 3 / 1 |
| beam 6,075 DOF | ARPACK | warp asm | 3.9 | 35.3 | 42.4 | 10.5 | 2.0 | 2.2 | **96.1** | 96.8 | 57 / 25 / 4 | 0.210 | 4 / 3 |
| beam 6,075 DOF | ARPACK | after | 2.0 | 18.4 | 41.0 | 8.3 | 1.4 | 0.8 | **72.2** | 72.2 | 17 / 5 / 0 | 0.099 | 1 / 0 |
| beam 6,075 DOF | subspace | before | 24.3 | 16.3 | 11.2 | 8.8 | 1.7 | 2.1 | **64.5** | 65.1 | 52 / 24 / 4 | 0.920 | 3 / 1 |
| beam 6,075 DOF | subspace | warp asm | 3.9 | 16.4 | 11.2 | 8.7 | 1.8 | 2.2 | **44.1** | 44.7 | 57 / 25 / 4 | 0.205 | 4 / 3 |
| beam 6,075 DOF | subspace | after | 1.9 | 17.5 | 10.9 | 8.3 | 1.3 | 0.8 | **40.8** | 40.9 | 17 / 5 / 0 | 0.095 | 1 / 0 |
| panel 22,509 DOF | ARPACK | before | 113.0 | 809.4 | 2369.2 | 69.8 | 7.3 | 10.2 | **3377.0** | 3382.1 | 52 / 24 / 4 | 1.177 | 3 / 1 |
| panel 22,509 DOF | ARPACK | warp asm | 15.8 | 810.1 | 1543.7 | 72.1 | 7.0 | 11.3 | **2465.2** | 2470.9 | 57 / 25 / 4 | 0.286 | 4 / 3 |
| panel 22,509 DOF | ARPACK | after | 6.0 | 408.6 | 1409.1 | 134.9 | 5.0 | 1.2 | **1940.6** | 1942.4 | 17 / 5 / 0 | 0.115 | 1 / 0 |
| panel 22,509 DOF | subspace | before | 100.8 | 371.4 | 82.5 | 57.7 | 7.0 | 3.6 | **629.2** | 632.9 | 52 / 24 / 4 | 1.075 | 3 / 1 |
| panel 22,509 DOF | subspace | warp asm | 14.2 | 364.5 | 84.2 | 66.4 | 6.6 | 11.0 | **546.6** | 552.0 | 57 / 25 / 4 | 0.290 | 4 / 3 |
| panel 22,509 DOF | subspace | after | 5.9 | 385.0 | 82.4 | 63.6 | 3.4 | 0.9 | **542.0** | 543.2 | 17 / 5 / 0 | 0.111 | 1 / 0 |

### Observations

* The update stays dominated by the host sparse factorization and eigensolve. The device path cuts
  assembly by 80-95 %, the Model rewrite by 55-95 %, device-to-host copies per update from 52 to 17 and
  host-to-device copies from 24 to 5, and removes all host-initialized allocations (so `wp.array` pointers,
  and captured graphs, survive an update).
* Shared factorization: ARPACK updates do one factorization instead of two, factorization time
  42-44 -> 18-26 ms (beam) and 809-843 -> 408-438 ms (panel). The subspace path already shared its LU; its
  factorization differences (16-23 ms beam, 365-430 ms panel) are run-to-run noise of the same code.
* Net per-update total, before -> after: beam ARPACK 120 -> 86 ms (CPU) / 124 -> 72 ms (cuda:0); beam
  subspace 65 -> 52 / 64 -> 41 ms; panel ARPACK 2574 -> 2134 / 3377 -> 1941 ms; panel subspace 668 -> 625
  / 629 -> 542 ms. The plan's recorded 61 / 176 ms baselines came from a
  different harness and are not comparable; the "before" rows here are re-measured with the same harness as
  the "after" rows. The
  panel ARPACK "before" eigensolve on cuda:0 (2369 ms vs 1559 ms on CPU) is host-side variance: the
  eigensolve runs on the host in both rows.
* Not improved: *projection* on the panel with the device path (66 -> 104 ms CPU, 70 -> 135 ms cuda:0, ARPACK
  rows; subspace rows within noise). Cause not isolated; profile it before scaling to larger meshes.
* Per step without an update: 0.80-1.18 ms (host joint_q read, numpy trigger, numpy pre-stress) ->
  0.10-0.12 ms on cuda:0 and 0.14-0.32 ms on CPU, with one 3-value device-to-host read (the trigger flag)
  and no host-to-device copy.

### Host round trips

Measured by `count_host_transfers()` around one warm `relinearize` and around 50 `apply_prestress` +
`poll` calls; see the table. Of the device path's 17 reads and 5 writes per update, the controller itself
makes 2 reads (the packed gather of state, `r_new`, `f_int` and the `K_T` block values; one diagnostics read
of momentum / kinetic energy, dropped with `UpdatePolicy(diagnostics=False)`) and 1 write (the packed basis
upload): `test_device_update.TestHostRoundTrips` asserts exactly these counts with the solver refresh
disabled. The remaining 15 reads / 4 writes come from `SolverVBD.refresh_elastic_mass_solve_data()`, which
still computes the Schur-form inverse reduced mass on the host (12 `numpy()` reads of the Model inertia
arrays in `_compute_elastic_mass_solve_data`, plus the effective-mass refresh) and `assign`s the 4
mobility arrays in place (inferred from the difference between the benchmark and the test; not separately
instrumented). Moving that small dense computation to a kernel is the remaining host round trip.

### CUDA graph capture

* `newton/tests/test_vbd_elastic_mass_refresh.py::test_captured_step_graph_stays_valid` (cuda:0): a step
  graph captured before an inertia change replays like an uncaptured run after
  `SolverVBD.refresh_elastic_mass_solve_data()`. **Pass.**
* `experimental/rerom/tests/test_device_update.py::TestGraphCapture` (cuda:0): pre-stress + two steps
  captured once, a device-path update between replays; the replayed graph matches an uncaptured run and sees
  the new basis (a control without the update differs). **Pass.**

### Acceptance (recorded M1 beam swing and WP3a best configuration)

Tip trajectory error vs the recorded runs (`L` = 1 m), Model on `cpu`, numpy assembler as recorded:

| run | M1 (600 steps, ARPACK) | WP3a (1200 steps, subspace + IMEX + ECSW) |
| --- | --- | --- |
| host path, `shared_factorization=False` | 0 (bit for bit), 469 updates at the recorded steps | 0 (bit for bit), 166 updates at the recorded steps |
| device path (default) | 8.9e-7 L, same update steps | 2.0e-6 L, same update steps |
| host path, 1-ulp perturbation of the initial state (noise floor) | 1.0e-6 L | 1.2e-6 L |

The device path differs from the host path only in summation order, so it is not bit-for-bit; its WP3a error
exceeds a strict 1e-6 L but is within 2x of the 1-ulp perturbation floor of that run, with identical update
steps. On `cuda:0` (device path, Warp assembler) the trajectories match the host path on `cuda:0` to
8.7e-7 L (M1) and 2.4e-6 L (WP3a), with identical update steps; both differ from the CPU recording by
up to 3.9e-4 L for WP3a because the solver itself runs on another device (the host path on `cuda:0` alone
gives the same 3.9e-4 L).

## 2. Tangent assembly (`WarpTetAssembler`)


Assembly of the total-Lagrangian tet4 energy, internal force and full tangent with
`rerom/assembly_warp.py` (`WarpTetAssembler`) against the numpy reference
`adaptive_rom.fe_nonlinear.NonlinearAssembler`. Kernel commit `09e4a7d7`.

### Command

```bash
python experimental/rerom/benchmarks/bench_assembly_warp.py --material stvk --out bench_stvk.json
python experimental/rerom/benchmarks/bench_assembly_warp.py --material neo_hookean --out bench_nh.json
```

`ADAPTIVE_ROM_ROOT` must point to the directory containing `adaptive_rom/`; by default the
script looks four levels above `experimental/rerom/benchmarks/`.

Environment: Intel Core Ultra 9 285K (24 cores; Warp CPU kernels and the numpy einsum
assembly both run on one thread), NVIDIA RTX PRO 6000 Blackwell Workstation Edition (driver
595.99.02), Python 3.12.14, numpy 2.5.3, SciPy 1.18.1, Warp 1.18.0. Medians over 10 (CPU) or 100
(CUDA) calls after 3 warm-up calls.

### Meshes

| case | mesh | nodes | DOF | tets | 3x3 blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| `beam_M1` | `make_beam_mesh(1.0, 0.025, 0.025, 80, 4, 4)` (M1 beam) | 2,025 | 6,075 | 7,680 | 24,057 |
| `panel_23k` | `make_beam_mesh(0.6, 0.4, 0.004, 60, 40, 2)` (thin plate) | 7,503 | 22,509 | 28,800 | 90,507 |

### Results, StVK (median ms per call)

| case | path | assemble (device x, no copies) | from/to numpy | force only | to CSC (host) | rotations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| beam_M1 | numpy reference | 22.4 | - | - | - | - |
| beam_M1 | Warp CPU, gather | 2.86 | 4.77 | 0.48 | 0.21 | 1.50 |
| beam_M1 | Warp CPU, atomic (auto) | 2.44 | 4.22 | 0.45 | 0.20 | 1.47 |
| beam_M1 | Warp cuda:0, gather (auto) | 0.164 | 0.407 | 0.067 | 0.43 | 0.076 |
| beam_M1 | Warp cuda:0, atomic | 0.141 | 0.381 | 0.060 | 0.44 | 0.077 |
| panel_23k | numpy reference | 88.1 | - | - | - | - |
| panel_23k | Warp CPU, gather | 13.5 | 21.2 | 1.75 | 1.05 | 5.40 |
| panel_23k | Warp CPU, atomic (auto) | 9.16 | 15.7 | 1.60 | 0.97 | 5.36 |
| panel_23k | Warp cuda:0, gather (auto) | 0.211 | 0.472 | 0.092 | 1.35 | 0.080 |
| panel_23k | Warp cuda:0, atomic | 0.174 | 0.434 | 0.088 | 1.38 | 0.081 |

Neo-Hookean gives the same picture (`../../../results/p2_wpD1_bench_assembly_neo_hookean.json` in the
exp-re-rom workspace; numpy 19.4 / 74.9 ms; CUDA gather 0.153 / 0.200 ms; CUDA to
CSC 0.43 / 1.39 ms; CPU atomic 2.22 / 8.37 ms).
Maximum relative difference of the tangent from the numpy reference: StVK 3.7e-16 / 5.4e-16
(CPU), 1.0e-14 / 1.2e-14 (CUDA); neo-Hookean 4.3e-16 / 6.0e-16 (CPU), 7.1e-15 / 9.3e-15 (CUDA).

Column definitions:

* *assemble*: `evaluate(x_dev, device_output=True, copy_tangent=False)` with float64 positions
  already on the device: one element kernel (+ gather kernels), energy reduced on the host (this
  is the only synchronisation). Tangent stays in the internal `BsrMatrix`.
* *from/to numpy*: `evaluate(x_numpy)`: host-to-device positions, numpy force, and a device copy of
  the BSR tangent (the default `copy_tangent=True`).
* *to CSC*: `to_scipy_csc(K)`: one device-to-host copy of the block values plus a cached
  permutation into the scalar CSC pattern (built once, outside the timing).
* *rotations*: `element_rotations` of all tets (polar Newton iteration), device output, followed by
  a device synchronisation.

### Observations

* GPU assembly of force and tangent is ~0.15-0.21 ms for both meshes, 130x (beam) to 420x
  (panel) faster than the numpy assembler. At these sizes it is bound by launch and
  synchronisation overhead; a first, preliminary run (log `p2_wpD1_bench_pre_commit_draft.log`)
  measured 0.34-0.40 ms for the same calls, so treat sub-millisecond GPU numbers as +-2x.
* For a host eigensolver the device-to-host CSC conversion (0.43 / 1.37 ms) costs more than the
  GPU assembly itself; assembly plus conversion is still ~38x (beam) and ~55x (panel) faster than numpy.
* Warp on the CPU (one thread) is 6.5-10x faster than the numpy einsum assembly.
* Gather versus atomic scatter: atomics are 15 % faster on CUDA (0.141 vs 0.164 ms beam,
  0.174 vs 0.211 ms panel) and 15-30 % faster on the CPU. The gather path is bitwise
  deterministic on every device; on CUDA the float64 atomic path is not (summation order). Warp
  runs CPU launches serially, so atomics are deterministic there. Default `scatter="auto"`
  therefore uses atomics on the CPU and gather on CUDA, where 0.02-0.04 ms buys reproducible
  tangents (and therefore reproducible eigenvectors and re-ROM trajectories).
* Setup (pattern, gather lists, mass, device upload): 9-17 ms (beam) and 33-49 ms (panel). The
  first CUDA construction in a process also loads the kernel module (~80 ms from cache). The first
  ever CUDA compile of the module (float64 polar/SVD and unrolled 12x12 blocks) took ~60 s and is
  cached in the Warp kernel cache afterwards.
* The numpy reference times (22.4 ms beam) are at the low end of the 23-34 ms recorded earlier
  for the M1 beam.
