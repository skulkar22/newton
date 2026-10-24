Add pluggable eigensolver backends for the finite-element modal generators:

  - `ModalGeneratorFEM` and `ModalGeneratorCraigBamptonFEM` accept a `ModalEigensolver` member or any `EigenSolverBackend` as `eigensolver`, in addition to the existing names.
  - `DenseBackend`, `ScipyArpackBackend`, and `SubspaceIterationBackend` (warm-startable shift-invert subspace iteration, also selected by `"subspace"`) implement the protocol and return an `EigenResult`.
  - `WarpackBackend` is a placeholder that documents the GPU backend contract; `resolve_eigensolver` and `to_backend_matrix` help select backends and convert Warp BSR matrices.
