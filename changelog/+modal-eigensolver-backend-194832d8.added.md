Add pluggable eigensolver backends for the finite-element modal generators:

  - `ModalGeneratorFEM` and `ModalGeneratorCraigBamptonFEM` accept a `ModalEigensolver` member or any `ModalEigensolverBackend` as `eigensolver`, in addition to the existing names.
  - `ModalDenseBackend`, `ModalScipyArpackBackend`, and `ModalSubspaceIterationBackend` (warm-startable shift-invert subspace iteration, also selected by `"subspace"`) implement the protocol and return a `ModalEigenResult`.
