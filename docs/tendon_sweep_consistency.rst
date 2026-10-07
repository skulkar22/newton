.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Tendon material-sweep consistency: review notes
=============================================

Scope
-----

This experimental correction is based on ``10715f13``. It uses circular
rollers and the existing sweep solver; it does not require the non-circular
profile prototype or the direct material solver. No public solver options or
defaults are added. The solver correction and the independent example setup
corrections are separate commits.

What changes
------------

* Retain each junction's net slip within a material solve. Before revisiting
  that junction, undo its previous trial transfer, retain neighboring
  transfers, and project again. This permits retraction of an excessive
  earlier transfer instead of stopping at an incorrect point inside the
  friction cone.
* Form the complete rolling no-slip trial before friction projection. Remove
  the beta-scaled differential rolling transport formerly applied after the
  sweeps. Friction limits slip and transmitted torque; it does not scale the
  prescribed rolling displacement.
* In XPBD, project the accumulated roller spin impulse and apply its change,
  including retraction on unloading. Independently limiting each iteration's
  impulse increment would not limit their accumulated result.
* In VBD, transmit the full sticking moment and limit out-of-cone trial
  moments using current adjacent tensions. Reuse the already evaluated span
  tension, but evaluate its neighbor at the current body trial, not from a
  stale diagnostic cache. The local positive-semidefinite Hessian freezes the
  limiter; its derivative is not included.

The capstan factor ``(rho - 1) / (rho + 1)`` remains in the physical bound on
the tension difference. It is the unconditional scaling of rolling motion
and reaction that is removed, not that bound.

This is not an early-out fix. On a frozen four-span diagnostic, disabling
early-out changed the old solve from 2 to 256 sweeps with bitwise identical,
incorrect output. The underlying iteration was stuck. The corrected
iteration needs more sweeps to satisfy the net-slip condition.

Validation
----------

The new ``test_tendon_material_sweeps`` module covers CPU and CUDA:

* analytical finite-friction net-slip solutions, both traversal directions,
  with adaptive early-out enabled and disabled;
* linear and sigmoid compliance, damping, slack and sticking;
* full rolling transport, absence of unloaded grip, and the rest-length floor;
* accumulated XPBD reaction, reversal and unloading;
* VBD sticking and sliding reactions, including a slack adjacent span.

Four independent analytical material tests were run against the unfixed
kernel with only its signature adapted; their assertions fail there.
The corrected focused material/reaction suite passes all 16 CPU/CUDA cases.

The one-circle and two-circles-on-one-body finite-friction reproducers run
for eight simulated seconds and include two dynamic-roller activations. In
XPBD, the backward rotation at the second activation falls from about
15.26 degrees to 0.00015 degrees for one circle, and from 9.34 degrees to
0.00124 degrees for two circles. The maximum trajectory-angle differences
from the direct reference are about 0.0019 and 0.0143 degrees, respectively.
The corresponding VBD reproducers also remove the activation glitch.

Existing tests were corrected where they demanded grip without tension,
strictly increasing rotation above the sticking threshold, or material-floor
saturation in an unloaded fixture. Loaded transport now has explicit
pretension and an analytical displacement check. The fixed rolling-chain
fixture uses supported signed wraps and a series-compliance tension
reference with float32-aware tolerances.

Reproduce the focused suite from this checkout::

    uv run --extra dev -m unittest newton.tests.test_tendon_material_sweeps

The broader validation covered these modules on CPU and available CUDA::

    uv run --extra dev -m unittest newton.tests.test_tendon_capstan newton.tests.test_tendon_material_sweeps newton.tests.test_tendon_vbd newton.tests.test_tendon_equilibrium newton.tests.test_cable newton.tests.test_fixed_tendon newton.tests.test_spatial_tendon

That broader run is not all green; see the known limitations below. Subsequent
focused reruns cover the VBD tension-reuse optimization and corrected fixtures.

Performance
-----------

Matched measurements on 2026-10-07 use the same circular pair scene, finite
friction coefficient 0.2, compliance 1e-4 m/N, damping 1 N s/m, 10 substeps,
32 body iterations and material settle tolerance 1e-5. Each run covers eight
simulated seconds (480 frames). Only warmed CUDA graph replay and completion
are timed; compilation, rendering and diagnostic readbacks are excluded.

For each solver, run order was unfixed, corrected, corrected, unfixed. On an
RTX 3080 Laptop GPU with Warp 1.15.0.dev20260626, the results were:

======  ====================  =====================  ================
Solver  Unfixed runs [s]      Corrected runs [s]     Mean-time change
======  ====================  =====================  ================
XPBD    14.585, 14.377        14.145, 13.711         -3.8%
VBD     21.768, 21.775        23.636, 23.719         +8.8%
======  ====================  =====================  ================

During these linear runs, periodic GPU-clock samples were 1800--1815 MHz
with no thermal-throttling flag. The nonlinear full-scene follow-up did hit
thermal throttling and is deliberately excluded from this table. These are
two runs per variant on a small reproducer, not a general speed guarantee.
The trajectories differ because the original response was incorrect; this
is a before/after cost comparison, not an equal-accuracy benchmark.

The material kernel itself can be substantially slower even when the whole
scene overhead is small. Earlier isolated frozen-state tests at default
settle tolerance showed about 1.8x cost for four linear spans, 3.9x for four
sigmoid spans, and up to 17x for sixteen sigmoid spans. In those cases the
old solver stopped after two sweeps with an incorrect net-slip result; the
corrected solver took 7--32 sweeps. These are material-only stress cases,
not whole-simulation slowdowns, and are not general performance guarantees.

Known limitations and separate example corrections
--------------------------------------------------

The correction is ready for review, not a claim that all tendon scenarios or
legacy examples are validated. In particular:

* The VBD cable-machine, rolling-pulley, dynamic-capstan and kinematic-capstan
  long-running examples still fail. All four also fail on the unfixed
  baseline. Their length-drift or payload-crossing failures are not hidden
  by new expected-failure annotations or relaxed thresholds.
* The full VBD XY-table trajectory is not validated by its half-second
  reference-prefix test. That prefix retains per-step checks and now uses
  only prefix-appropriate final checks; its reference tolerances are tighter.
* Total material drift is not eliminated. The circular pair repro has about
  28--31 micrometres of drift with corrected sweeps versus 18--19 micrometres
  with the old implementation. Finite output alone is not proof of accuracy.
* Unsupported wraps, stiff-body/joint convergence and the existing VBD
  slack/damping gate are not redesigned by this change.

The separate example correction commit constrains the equal-weight
equilibrium fixture to vertical motion (angled spans otherwise pull the
weights inward), and changes compound-pulley compliance from 1e-6 to 1e-5 m/N
to reduce force sensitivity to position rounding at that scene scale. The
original final assertions pass for both XPBD and VBD. This is a change to
those fixtures, not a solver improvement at the former stiffness.

Experimental-branch integration
------------------------------

This local port applies ``cf643a3c20d12454c463fd4bfa098d4773f9f878``
to experimental base ``1a775c105181b187b09acf3a53b45e2cd0cd91ce``.
It preserves the direct material specialization, cooperative CUDA backend,
explicit route orientation, and accepted-route activation handling. The
cooperative entry point accepts the sweep scratch argument for launch
compatibility; its direct projection does not use it.

The corrected sweep and accumulated XPBD reaction are enabled in their
respective paths. VBD's non-direct path uses the current-tension reaction
from the selected correction. Its existing direct/ALM reaction stays intact,
including the per-segment ALM full endpoint-wrench guard from ``f11f0e91``.
Independent clipping of that mode's rim moment would violate internal
force/moment balance.

The generic damping and material/body-coupling changes on the later profile
branch are not included. Elliptical and convex pulley features are not
included. This correction alone does not qualify the complete experimental
branch on current Newton main or certify arbitrary damping/ALM combinations.

The inherited geometry-only roller helper is retained separately for ALM
support-stiffness estimation. Its three callers retain their original
formula; this estimate is not the physical current-tension reaction.
