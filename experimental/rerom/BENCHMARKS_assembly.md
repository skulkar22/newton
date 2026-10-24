# Tangent assembly benchmarks (WarpTetAssembler)

Assembly of the total-Lagrangian tet4 energy, internal force and full tangent with
`rerom/assembly_warp.py` (`WarpTetAssembler`) against the numpy reference
`adaptive_rom.fe_nonlinear.NonlinearAssembler`. Kernel commit `09e4a7d7`.

## Command

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

## Meshes

| case | mesh | nodes | DOF | tets | 3x3 blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| `beam_M1` | `make_beam_mesh(1.0, 0.025, 0.025, 80, 4, 4)` (M1 beam) | 2,025 | 6,075 | 7,680 | 24,057 |
| `panel_23k` | `make_beam_mesh(0.6, 0.4, 0.004, 60, 40, 2)` (thin plate) | 7,503 | 22,509 | 28,800 | 90,507 |

## Results, StVK (median ms per call)

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

## Observations

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
