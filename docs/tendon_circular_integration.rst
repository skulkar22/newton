Circular tendon mechanics integration
=====================================

This experimental adaptation extracts shape-independent mechanics from
``6bcd973e7327d882f07a46f32186dad134ceb327``, checked against
``a1f16439b6ecfe0926e34703595f5d557fc35304``. It follows the sweep correction
``cf643a3c20d12454c463fd4bfa098d4773f9f878`` on experimental base ``1a775c10``.
No elliptical or convex routing implementation is included.

XPBD now solves coupled sliding spans together, including repeated-body mass
cross terms, and retracts accumulated impulses consistently. Non-ALM VBD
projects material before each body color and condenses frictionless spans into
one curvature contribution. Their damping uses the full contact-point motion.
The builder colors all bodies within each free-material block.

The established per-segment ALM formulation retains its engagement damping,
multiplier update, accepted-route handling, and full endpoint wrench. It does
not use the new condensed VBD curvature. Passing the non-ALM Kelvin-Voigt tests
does not establish equivalence of the ALM force law; that combination requires
separate qualification. Full Toy4 accuracy and performance are recorded with
the final source and runtime, outside the publishable code tree.

Current-main adaptation
-----------------------

This forward port preserves current-main joint ALM, observables, restitution,
angular winding, mimic support, and the tendon-free sparse articulation solver.
Tendons currently require local VBD with VBD-owned rigid integration. Combining
tendons with ``block_sparse_joints`` or an external rigid solver is rejected.
The separate legacy ``rigid_joint_limit_alm=True`` experiment has not been
ported over current-main joint ALM; the compatibility argument accepts False.
This restriction does not disable tendon ALM.

Tendon links require an existing body. Use an explicit zero-mass body for a
fixed world anchor. Builder copies and replication remap tendon body and link
indices, and preserve body-local attachment geometry. Reconstruct the solver
for a fresh tendon episode: ``SolverVBD.reset`` currently rejects tendon models
because it does not restore their material, routing, and multiplier history.

Current main requires vector gravity and one articulation for joint-connected
bodies. Older model scripts must migrate those authoring details while
preserving their physical parameters. Upstream default rigid compliant ALM is
retained; legacy penalty ramp settings no longer control that default path.
