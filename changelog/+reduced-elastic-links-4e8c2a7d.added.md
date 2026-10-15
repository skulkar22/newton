Add experimental reduced elastic links to `SolverVBD`, using a floating frame plus linear modal coordinates:

  - `ModelBuilder.add_link_elastic()` and `ModelBuilder.add_body_elastic()` create elastic links from a shared `ModalBasis`; joints, joint coordinates, and contacts act on the deformed attachments, and `ViewerGL` renders the deformed shapes.
  - The frame and modal equations derive from one kinetic energy with deformation-dependent inertia, so a free elastic body conserves its momentum and its motion does not depend on a uniform velocity boost or on the choice of frame origin.
  - Mass, center of mass, and inertia default to the rigid properties of the modal basis, with a warning when explicit values disagree with it.
  - `ModalBasis` stores the mode-pair inertia integrals and full modal mass matrix, computed in bounded memory, and named attachment frames for joint endpoints.
  - The coupled frame-and-mode blocks are assembled in parallel over modes and columns and solved with a shared-memory or dense factorization selected by `elastic_block_solver`, falling back to the dense solve for bases too wide for shared memory.
  - Elastic frames track their displacement from the inertial target, so frames far from the origin keep their ballistic center of mass.

See the Reduced Elastic Links concept page for the formulation, units, and limitations.
