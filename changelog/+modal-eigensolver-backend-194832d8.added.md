Add pluggable eigensolver backends for the finite-element modal generators:

  - `ModalGeneratorFEM` and `ModalGeneratorCraigBamptonFEM` accept a `ModalEigensolver` member or any `ModalEigensolverBackend` as `eigensolver`, in addition to the existing names.
  - `ModalDenseBackend`, `ModalScipyArpackBackend`, and `ModalSubspaceIterationBackend` (shift-invert subspace iteration, also selected by `"subspace"`, that can warm-start from its previous solve with `warm_start=True`) implement the protocol and return a `ModalEigenResult`.
