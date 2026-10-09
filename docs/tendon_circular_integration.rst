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
