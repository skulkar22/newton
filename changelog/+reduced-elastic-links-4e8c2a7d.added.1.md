Add modal basis generators for reduced elastic links:

  - `ModalGeneratorBeam` for Euler-Bernoulli bending, torsion, and axial modes, including cantilever eigenfunctions of every order, with volume-integrated inertia.
  - `ModalGeneratorFEM` for eigenmodes of nodal finite-element matrices, with consistent-mass inertia integrals and automatic rigid-body mode removal for free-free models.
  - `ModalGeneratorCraigBampton` for Craig-Bampton reduced matrices, with every interface registered as an attachment frame.
  - `ModalGeneratorPOD` for proper orthogonal decomposition of displacement snapshots, and `ModalGeneratorSampled` for externally computed modes.

The FEM and Craig-Bampton generators diagonalize the mass and stiffness matrices exactly. A damping matrix is projected onto the modes, and its off-diagonal part is discarded with a warning or rejected with `damping_projection="error"`.
