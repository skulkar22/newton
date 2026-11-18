Add experimental `ModelBuilder.extend_articulation()` to append contiguous,
unowned joints to an imported or previously declared articulation. Validate
ownership, worlds, tree structure, kinematic roots and cross-articulation
connections before changing the range. Elastic owner joints remain state-only
entries and loop-closing physical joints require explicit opt-in.
