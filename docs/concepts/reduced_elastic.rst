.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _Reduced Elastic Links:

Reduced Elastic Links
=====================

.. experimental::

   Reduced elastic links, :class:`ModalBasis`, the ``ModalGenerator*`` classes,
   :meth:`ModelBuilder.add_link_elastic`, :meth:`ModelBuilder.add_body_elastic`,
   and :attr:`JointType.ELASTIC` may change without a deprecation period.
   :class:`~newton.solvers.SolverVBD` is currently the only solver that
   simulates them.

A reduced elastic link is a body whose deformation is a linear combination of a
small number of mode shapes moving with a floating reference frame
(floating-frame-of-reference formulation). It suits slender or stiff
structural parts, such as linkage arms, beams, and grippers, whose deformation
is small compared to their rigid motion but still affects the dynamics.

Create one from a :class:`ModalBasis`, usually produced by one of the
generators described below:

.. code-block:: python

   basis = newton.ModalGeneratorBeam(
       length=1.0,
       half_width_y=0.02,
       half_width_z=0.01,
       mode_specs=[{"type": "bending_z", "boundary": "cantilever-tip"}],
       density=2700.0,
       young_modulus=7.0e10,
   ).build()

   builder = newton.ModelBuilder()
   beam = builder.add_body_elastic(xform=wp.transform_identity(), modal_basis=basis)
   builder.color()
   model = builder.finalize()
   solver = newton.solvers.SolverVBD(model)

:meth:`~ModelBuilder.add_body_elastic` creates a free elastic body in its own
articulation; :meth:`~ModelBuilder.add_link_elastic` creates a link that is
connected by ordinary joints, such as
:meth:`~ModelBuilder.add_joint_revolute`, using their parent and child
transforms on the elastic link as attachment points.

Kinematics and state
--------------------

Each elastic link owns a :attr:`JointType.ELASTIC` joint that stores its state
and adds no constraint. The body frame is the floating reference frame with
pose ``(p, R)``; the body's center of mass :attr:`Model.body_com` is the *rest*
center of mass :math:`\mathbf{c}` in that frame. A body-local material point
:math:`\mathbf{r}_s` moves to

.. math::

   \mathbf{x}_s = \mathbf{p} + \mathbf{R}\,(\mathbf{r}_s + \boldsymbol{\Phi}_s \mathbf{q}),

where :math:`\boldsymbol{\Phi}_s` (``sample_phi``) holds the translational
values of the ``n`` modes at the point and :math:`\mathbf{q}` the modal
coordinates. The angular mode values :math:`\boldsymbol{\Psi}_s`
(``sample_psi``) give the small material rotation :math:`\boldsymbol{\Psi}_s
\mathbf{q}` used by joint orientations.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Array
     - Layout
     - Meaning
   * - ``joint_q``
     - ``[p (3), quat xyzw (4), q (n)]``
     - Frame origin position and orientation in world, then the modal
       coordinates.
   * - ``joint_qd``
     - ``[v_c (3), omega (3), qdot (n)]``
     - World velocity of the rest center of mass ``p + R c``, world angular
       velocity, then the modal velocities, matching the
       :ref:`twist convention <Twist conventions>` of free joints.
   * - ``joint_f``
     - ``[f (3), tau (3), Q (n)]``
     - World force at the rest center of mass and world torque on the frame,
       then the generalized modal forces.

:attr:`State.body_q` and :attr:`State.body_qd` hold the frame pose and twist.
The *deformed* center of mass is :math:`\mathbf{p} + \mathbf{R}(\mathbf{c} +
\bar{\mathbf{S}}\mathbf{q}/m)`. On a free body it follows the ballistic path,
to first order in the time step; the frame origin and the rest center of mass
generally do not.

Inertia
-------

With :math:`\mathbf{y}_s = \mathbf{r}_s - \mathbf{c} + \boldsymbol{\Phi}_s
\mathbf{q}` the deformed offset from the rest center of mass, the kinetic
energy is

.. math::

   T = \tfrac{1}{2} \sum_s m_s \left| \mathbf{v}_c
       + \boldsymbol{\omega} \times \mathbf{R}\mathbf{y}_s
       + \mathbf{R}\boldsymbol{\Phi}_s \dot{\mathbf{q}} \right|^2
     = \tfrac{1}{2}\, \boldsymbol{\nu}^T \mathbf{M}(\mathbf{q})\, \boldsymbol{\nu},
   \qquad
   \boldsymbol{\nu} = (\mathbf{R}^T\mathbf{v}_c,\ \mathbf{R}^T\boldsymbol{\omega},\ \dot{\mathbf{q}}).

The reduced mass matrix in frame coordinates is

.. math::

   \mathbf{M}(\mathbf{q}) =
   \begin{bmatrix}
     m\,\mathbf{1} & -[\bar{\mathbf{S}}\mathbf{q}]_\times & \bar{\mathbf{S}} \\
     [\bar{\mathbf{S}}\mathbf{q}]_\times & \mathbf{I}(\mathbf{q}) & -\bar{\mathbf{G}}(\mathbf{q}) \\
     \bar{\mathbf{S}}^T & -\bar{\mathbf{G}}(\mathbf{q})^T & \mathbf{M}_f
   \end{bmatrix}

and is assembled from the first and second moments of the mode shapes, which
:class:`ModalBasis` stores per mode ``i`` and mode pair ``(i, j)``:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Symbol
     - :class:`ModalBasis` attribute
     - Definition (lumped form)
   * - :math:`\bar{\mathbf{S}}_i`
     - ``mode_coupling_linear``
     - :math:`\sum_s m_s \boldsymbol{\phi}_i`
   * - :math:`\mathbf{G}_i`
     - ``mode_coupling_angular``
     - :math:`\sum_s m_s \boldsymbol{\phi}_i \times \mathbf{r}_s`
   * - :math:`\mathbf{B}_i`
     - ``mode_coupling_centrifugal``
     - :math:`\sum_s m_s \boldsymbol{\phi}_i \mathbf{r}_s^T`
   * - :math:`\mathbf{C}_{ij}`
     - ``mode_coupling_coriolis``
     - :math:`\sum_s \boldsymbol{\phi}_j \times m_s \boldsymbol{\phi}_i`
   * - :math:`\mathbf{P}_{ij}`
     - ``mode_coupling_quadratic``
     - :math:`\sum_s \boldsymbol{\phi}_i\, m_s \boldsymbol{\phi}_j^T`
   * - :math:`(\mathbf{M}_f)_{ij}`
     - ``mode_mass_matrix`` (diagonal ``mode_mass``)
     - :math:`\operatorname{tr} \mathbf{P}_{ij}`

The blocks of :math:`\mathbf{M}(\mathbf{q})` follow from these integrals,
referred to the rest center of mass: :math:`\bar{\mathbf{G}}_i(\mathbf{q}) =
\mathbf{G}_i + \mathbf{c} \times \bar{\mathbf{S}}_i - \sum_j \mathbf{C}_{ij}
q_j` and

.. math::

   \mathbf{I}(\mathbf{q}) = \mathbf{I}_0
     + \sum_i q_i \left( 2 \operatorname{tr}\mathbf{B}^c_i\,\mathbf{1} - \mathbf{B}^c_i - \mathbf{B}^{cT}_i \right)
     + \operatorname{tr}\mathbf{D}\,\mathbf{1} - \tfrac{1}{2}(\mathbf{D} + \mathbf{D}^T),
   \qquad
   \mathbf{B}^c_i = \mathbf{B}_i - \bar{\mathbf{S}}_i \mathbf{c}^T,\quad
   \mathbf{D} = \sum_{ij} q_i q_j \mathbf{P}_{ij}.

The frame mass :math:`m`, rest center of mass :math:`\mathbf{c}`, and rest
inertia :math:`\mathbf{I}_0` must describe the same mass distribution as the
mode integrals. :meth:`~ModelBuilder.add_link_elastic` therefore defaults them
to the basis's ``rigid_mass``, ``rigid_com``, and ``rigid_inertia`` and warns
when explicit values differ by more than ``inertia_tolerance``.

Frame and modal rows are derived from the same kinetic energy. The frame
rows are backward differences of the linear and angular momentum carried by
the deformation, and the modal rows contain the matching gravity, frame
acceleration, centrifugal, and Coriolis loads. As a result:

- a free body conserves its linear and angular momentum in the discrete
  balance, up to solver convergence and ``float32`` roundoff;
- the motion is invariant, up to ``float32`` roundoff, to a uniform velocity
  boost and to the choice of the frame origin relative to the material (only
  :math:`\mathbf{c}`, ``p``, and the integrals change).

Units and normalization
-----------------------

The units below assume *displacement-valued* modal coordinates, in which
:math:`\boldsymbol{\phi}` is dimensionless, as produced by
:class:`ModalGeneratorBeam` and :class:`ModalGeneratorPOD`.
:class:`ModalGeneratorFEM` and :class:`ModalGeneratorCraigBampton` return
*mass-normalized* modes instead.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Quantity
     - Displacement-valued
     - Mass-normalized
   * - ``q`` / ``qdot`` (``mode_q``, ``mode_qd``)
     - m, m/s
     - kg\ :sup:`1/2`\ ·m, kg\ :sup:`1/2`\ ·m/s
   * - ``sample_phi`` / ``sample_psi``
     - 1, rad/m
     - kg\ :sup:`-1/2`, rad/(kg\ :sup:`1/2`\ ·m)
   * - ``mode_mass``, ``mode_mass_matrix``, ``P``, ``C``
     - kg
     - 1 (``mode_mass_matrix`` is the identity)
   * - ``mode_stiffness``
     - N/m
     - s\ :sup:`-2`
   * - ``mode_damping``
     - N·s/m
     - s\ :sup:`-1`
   * - ``S`` (``mode_coupling_linear``)
     - kg
     - kg\ :sup:`1/2`
   * - ``G``, ``B`` (``mode_coupling_angular``, ``_centrifugal``)
     - kg·m
     - kg\ :sup:`1/2`\ ·m
   * - Modal entries of ``joint_f``
     - N
     - N·kg\ :sup:`-1/2`

Any other scaling works as well. The dynamics are invariant under
:math:`\boldsymbol{\phi} \to \boldsymbol{\phi}/a`, :math:`\mathbf{q} \to a
\mathbf{q}` when the mode mass, stiffness, damping, and pair integrals are
divided by :math:`a^2` and the single-mode integrals by :math:`a`.
:meth:`ModalBasis.mass_orthonormalize` converts a basis with a non-diagonal
modal mass matrix into an equivalent mass-orthonormal one when its stiffness
and damping stay diagonal; the solver also accepts the full matrix directly.

Modal generators
----------------

All generators return a :class:`ModalBasis`. They differ in which inertia
integrals they can compute exactly:

.. list-table::
   :header-rows: 1
   :widths: 18 30 22 30

   * - Generator
     - Inertia integrals and rigid properties
     - Angular values ``psi``
     - Attachment frames
   * - :class:`ModalGeneratorSampled`
     - All lumped from ``sample_mass``; none without it.
     - Supplied, or estimated with ``derive_psi``.
     - :meth:`ModalBasis.add_frame`
   * - :class:`ModalGeneratorPOD`
     - All lumped from a uniform ``total_mass``; none without it.
       Stiffness is a heuristic.
     - Estimated with ``derive_psi``.
     - :meth:`ModalBasis.add_frame`
   * - :class:`ModalGeneratorFEM`
     - All exact from the consistent nodal mass matrix.
     - Estimated with ``derive_psi``.
     - :meth:`ModalBasis.add_frame`
   * - :class:`ModalGeneratorCraigBampton`,
       :class:`ModalGeneratorCraigBamptonFEM`
     - ``Mf``, ``S``, ``G``, and rigid properties exact from the reduced
       matrices; ``B``, ``C``, and ``P`` only with ``sample_mass``, lumped
       over the recovery samples.
     - Estimated at recovery samples, exact at interfaces.
     - Every interface.
   * - :class:`ModalGeneratorBeam`
     - Diagonal analytic centerline ``Mf``; ``S``, ``G``, ``B``, ``C``, and
       ``P`` and prism rigid properties by volume quadrature, with rotary
       inertia neglected consistently.
     - Analytic.
     - :meth:`ModalBasis.add_frame`

When an integral is unavailable it is zero, and the corresponding coupling is
absent. In particular, a Craig-Bampton basis without ``sample_mass`` keeps its
rotational inertia frozen at :math:`\mathbf{I}_0` and has no Coriolis
coupling.

The FEM and Craig-Bampton generators project a damping matrix onto the
retained modes and keep only its diagonal. Pass
``damping_projection="error"`` to reject damping that the undamped modes do
not diagonalize; ``damping_off_diagonal_ratio`` records the discarded
coupling. The Craig-Bampton generator also rejects damping that acts on rigid
motion. The FEM generator does not: the rigid-body part of, for example,
mass-proportional damping of a free-free model is dropped, so the floating
frame is undamped.

:class:`ModalGeneratorFEM` accepts dense arrays or SciPy sparse matrices.
``eigensolver="dense"`` solves the full spectrum with LAPACK, densifying
sparse input, and suits models up to a few thousand DOFs.
``eigensolver="arpack"`` keeps the matrices sparse and computes only the
lowest modes with ARPACK shift-invert Lanczos on a sparse LU factorization of
:math:`\mathbf{K} - \sigma\mathbf{M}`; it requires ``mode_count``. The
rigid-body modes of a free-free model are deflated analytically, and a Sturm
sequence count from a second factorization verifies that no mode below the
retained ones was missed; when no gap in the spectrum allows that check, a
warning says so. The residual of every returned eigenpair is checked as
well. The default ``"auto"`` uses ARPACK when any input matrix is sparse, or
when the free DOF count exceeds ``dense_dof_limit`` and ``mode_count`` is
set. Both paths orient each mode so that its largest-magnitude entry is
positive; within a repeated eigenvalue the basis can differ between them. The
default shift :math:`\sigma = -10^{-6}\,\|\mathbf{K}\|_1 /
\|\mathbf{M}\|_1` keeps the shifted matrix definite for free-free models,
which require a negative shift because their rigid-body eigenvalues are zero;
for constrained meshes whose stiffest and softest modes differ by many
decades, an ``arpack_shift`` just below the lowest wanted eigenvalue
converges faster. A shift at an eigenvalue raises. On this path the
zero-energy threshold is relative to a Gershgorin-type bound
:math:`\max_i \sum_j |K_{ij}| / M_{ii}` on the largest eigenvalue, which is
close for three-DOF meshes but can exceed it about 40 times for
consistent-mass shells with rotational DOFs, as in the six-DOF interior of
:class:`ModalGeneratorCraigBamptonFEM`, whose fixed-interface eigenvalues at
or below ``eigenvalue_tolerance`` times that bound are reported as
mechanisms. The
factorization dominates time and memory: its fill grows much faster for
volume meshes than for shell meshes of the same size. For large models, also
pass ``sample_node_indices`` (sample points that coincide with nodes are
otherwise detected automatically, but all others are interpolated from every
node).

:class:`ModalGeneratorCraigBamptonFEM` performs the fixed-interface
Craig-Bampton reduction itself, from full nodal mass, stiffness, and optional
damping matrices with three (``[ux, uy, uz]``) or six
(``[ux, uy, uz, rx, ry, rz]``) DOFs per node, and passes the reduced
matrices and the translational recovery of the sample nodes to
:class:`ModalGeneratorCraigBampton`. Each interface is a six-DOF frame
defined by a list of nodes: ``"node"`` uses one six-DOF node directly,
``"rbe2"`` ties the nodes rigidly to the frame, and ``"rbe3"`` defines the
frame as the least-squares rigid fit of the node translations without
stiffening them. The static constraint modes and the lowest fixed-interface
modes reuse one sparse factorization of the interior stiffness, with every
frame fixed; a mechanism that remains, such as a part that is not tied to the
rest, raises. The generator exposes the intermediate matrices and a report of
residuals and timings, and :meth:`~ModalGeneratorCraigBamptonFEM.save_npz`
writes the reduced model as arrays named like the generator arguments.

The eigensolver of both finite-element generators is pluggable. Besides the
names above, ``eigensolver`` accepts a :class:`ModalEigensolver` member or any
object that satisfies the :class:`ModalEigensolverBackend` protocol: a ``name``, a
``device``, a ``supports_inertia_count`` flag, and a ``solve`` method that
returns the lowest eigenpairs of :math:`\mathbf{K}\boldsymbol{\phi} =
\lambda\mathbf{M}\boldsymbol{\phi}` as a :class:`ModalEigenResult`, in ascending
order, :math:`\mathbf{M}`-orthonormal, and orthogonal to the deflated
rigid-body motions. The generators pass the free (or interior) DOF matrices
in SciPy CSC form with the shift and tolerance from ``arpack_shift`` and
``arpack_tol``, and check the result: a backend that computes only part of
the spectrum either returns a Sturm sequence count, which must match the
eigenvalues it returned below the count's shift, or sets
``supports_inertia_count`` to ``False``, in which case the check is skipped
with a warning. Host (``"cpu"``) backends share the validated sparse
factorization of :math:`\mathbf{K} - \sigma\mathbf{M}`; a device backend
receives none. :class:`ModalDenseBackend` and :class:`ModalScipyArpackBackend`
wrap the built-in paths; :class:`ModalSubspaceIterationBackend`
(``"subspace"``) runs shift-invert block subspace iteration with
Rayleigh-Ritz on that factorization. Constructed with ``warm_start=True``, it
reuses the modes of its previous solve as the start block, as when an
adaptive reduced model updates its basis; such a backend is stateful and
should not be shared between generators. The ``"subspace"`` name always
selects a cold-started backend. A GPU backend, such as a sparse direct
(cuDSS) or preconditioned LOBPCG solver, can be supplied as any object that
satisfies the protocol; this build contains none.

Assemblies of separately meshed parts are joined with linear multi-point
constraints :math:`\mathbf{C}\mathbf{u} = 0`, which
:class:`ConstraintElimination` removes with a coordinate transformation
:math:`\mathbf{u} = \mathbf{T}\mathbf{q}`; the Craig-Bampton generator
applies it first when given ``constraints`` or a ready ``transformation``.
Node-to-node ties keep :math:`\mathbf{T}` Boolean, so the tied matrices are
as sparse as the mesh; other rows, such as rigid links or averaging elements,
are eliminated by pivoting on one dependent DOF per row. Constraints between
coincident nodes and rigid links preserve rigid-body motion, which
:meth:`~ConstraintElimination.rigid_motion_residual` checks; constraints
between nodes at different positions preserve it only when they include the
lever arm. :meth:`ModalGeneratorCraigBamptonFEM.reduce` rejects constraints
that ground the body or resist rigid motion, because the reduced model would
lack its six rigid-body modes. :func:`newton.utils.load_sparse_matrix` reads matrices exported as
Matrix Market or SciPy ``.npz`` files. All generators require symmetric
matrices in full storage and reject a relative asymmetry above
:math:`10^{-8}`: a matrix exported as one triangle must be expanded first,
which :func:`~newton.utils.load_sparse_matrix` does for Matrix Market files
declared ``symmetric`` and checks with ``symmetric=True``.

Joints, attachments, and contacts
---------------------------------

Joints attached to an elastic link act on the deformed attachment frame: its
origin moves by :math:`\boldsymbol{\Phi}\mathbf{q}` and its orientation
rotates by :math:`\boldsymbol{\Psi}\mathbf{q}`. Drives, limits, friction, and
the reported ``joint_q`` of such joints are measured between the deformed
frames. The attachment's mode values are chosen in this order:

1. An attachment frame of the basis within 1e-7 m (for example a
   Craig-Bampton interface).
2. A material sample at the same point.
3. Otherwise a new massless sample whose values are inverse-distance
   interpolated from the existing samples.

Interpolated values are only approximate. Place an exact sample, or add an
attachment frame with :meth:`ModalBasis.add_frame`, at every joint location.

:class:`~newton.solvers.SolverVBD` measures a joint's angular error against the
rest relative rotation of its parent and child frames, evaluated with
:attr:`Model.joint_X_p` and :attr:`Model.joint_X_c`, so editing a joint frame
moves the joint target with it. Code that updates an elastic basis in place
and moves an endpoint frame to the deformed reference, by rewriting the joint
frames and resetting :math:`\mathbf{q}`, would therefore change the target of
every joint at that endpoint. When the endpoints of a body rotate differently,
as in a closed loop through the body, rotating the body's rest pose cannot
compensate for all of them. In that case, pass the original frames to
:meth:`~newton.solvers.SolverVBD.set_joint_rest_frames` once, after creating
the solver and before capturing a CUDA graph: the joints then act on the
rewritten frames but keep their original targets.

Contacts are generated between the deformed vertices of the shapes on an
elastic link and rigid shapes; shape pairs in which both shapes belong to
elastic links produce no contacts. :class:`~newton.viewer.ViewerGL` renders the
deformed shapes.

Solver
------

:class:`~newton.solvers.SolverVBD` solves the frame and the modes of each
elastic link together as one dense ``(6 + n) x (6 + n)`` block per body and
color. Assembly is parallel over modes: each iteration forms the mode-pair sums
(:math:`\bar{\mathbf{G}}(\mathbf{q})`, :math:`\sum_j \mathbf{C}_{ij}
\dot{q}_j`, :math:`\sum_j \mathbf{P}_{ij} q_j`) with one thread per body and
mode, evaluates each joint at a deformed elastic attachment once, and fills
each block with one thread per column.

Elastic frames accumulate their update as a displacement from the inertial
target, like bodies in the block-sparse articulation solve, so a frame far
from the origin keeps its ballistic center of mass as accurately as a rigid
body. Models with joint mimics, or with ``rigid_soft_enable_dat=True``, update
absolute poses instead.

Relevant options:

- ``elastic_block_solver``: ``"tiled"`` keeps each block in shared memory,
  which needs about ``4 * ((6 + n)**2 + 6 + n)`` bytes. The requirement is
  measured from the compiled kernel when the solver is constructed; a device
  with 101,376 bytes of shared memory per block fits 151 modes, and the CPU
  uses Warp's 256 KiB tile arena. ``"dense"`` factorizes in global memory with
  32 x 32 shared panels and supports any width that fits device memory.
  ``"auto"`` (default) uses the tiled solve for every block width that fits
  and the dense solve, with one warning, for the others.
- Bodies of one color are grouped by block width (``6`` plus the next power of
  two, at least 8 modes), so small bases are not padded to the largest one.
- ``elastic_body_relaxation`` under-relaxes the block update, and
  ``elastic_solver_metrics`` enables
  :meth:`~newton.solvers.SolverVBD.elastic_mode_solve_metrics`.

With ``rigid_articulation_solve="block_sparse_joints"``, elastic links inside
an articulation are solved in the articulation's direct factorization: their
modes are eliminated exactly onto the frame blocks of the link and its joint
neighbors. These links use ``rigid_articulation_relaxation`` and report no
block metrics. See :ref:`Loop closure inside an articulation` for the joint
features this mode does not support.

Each iteration factors every link's ``n x n`` modal block once and solves all
of its right-hand sides (the modal residual and six columns per neighbor
frame) in the same pass, before the articulation's frame system is factored.
The modal blocks of all links in articulations are padded to the largest mode
count among them. ``elastic_block_solver`` selects the factorization as for
the local solve: the tiled path keeps one modal block in shared memory (about
``4 * (n**2 + 7 * n)`` bytes, so a device with 101,376 bytes per block fits
about 150 modes), and the dense path factors it in global memory. The cost per
step is then close to the local solve's. Measured on an RTX PRO 6000 Blackwell
GPU (Warp 1.18, 8 iterations, CUDA graph replay) for a rigid base carrying
elastic beam links, with the local solve in parentheses:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Modes per link
     - 1 link [ms/step]
     - 8 links [ms/step]
   * - 32
     - 1.6 (1.3)
     - 2.2 (2.2)
   * - 105
     - 3.5 (2.9)
     - 4.6 (4.9)
   * - 151
     - 5.2 (4.0)
     - 6.9 (7.0)

On the CPU the sparse step costs 0.8 to 1.4 times the local step for the same
cases (for example 51 ms against 37 ms for eight 151-mode links).

A link whose dense frame-and-mode block is not positive definite, typically
because of an inconsistent modal mass or stiffness, cannot be eliminated. Its
frame is then solved with its own frame rows only, and its modes take a
diagonal (Jacobi) step after the frame solve, as the local block solve's
fallback does; this converges slowly or not at all, and the solver warns once.

Approximations and limits
-------------------------

- **Linear elasticity.** The modes are fixed shapes with constant stiffness;
  there is no geometric (stress) stiffening, so a fast-spinning or highly
  preloaded link is softer than in a nonlinear model.
- **Quadratic velocity terms.** The centrifugal and Coriolis loads are
  re-evaluated at every solver iteration but are not part of the block
  Hessian. Fast rotation needs more iterations.
- **Time integration.** The update is first order in time. Modal amplitude
  and phase errors grow with :math:`\Delta t\,\omega_k`, so resolve the highest
  retained modal frequency :math:`\omega_k` or accept damping of the high
  modes.
- **Massless modes.** A mode with zero ``mode_mass`` responds quasi-statically.
  Under compliant ALM, contacts and joint drives or limits whose only mobility
  comes from massless modes have no support and are dropped.
- **Joint terms on elastic attachments.** For a joint attached to an elastic
  link, joint friction and passive joint damping are not assembled into the
  elastic block, so they act on the other (rigid) body only, and mimic
  reactions reach the elastic frame but not its modes. The block-sparse
  articulation solve rejects joint friction, passive joint damping, and joint
  armature on joints attached to elastic links, and rejects joint mimics
  anywhere in a model whose articulations contain elastic links.
- **Elastic–elastic joints** in the block-sparse articulation solve replace the
  mode–mode coupling by a conservative per-link term, which keeps the
  converged solution but needs about four times as many iterations as the
  local solve; see :ref:`Loop closure inside an articulation`.
- **Precision.** States are ``float32``. With the displacement-form frame
  update, an elastic link far from the origin is as accurate as a rigid body
  there; with the absolute-pose update (joint mimics or
  ``rigid_soft_enable_dat=True``), per-step frame motion below the ``float32``
  spacing of the position is lost.
- **Legacy shape callables.** ``mode_shape_fn`` provides joint attachment
  values only, with zero angular values and no inertia coupling.

Warnings
--------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Warning
     - Meaning and remedy
   * - Modes are not mass-orthogonal (:class:`ModalBasis`)
     - An explicit diagonal ``mode_mass`` discards a significant off-diagonal
       modal mass. Omit ``mode_mass`` or pass ``mode_mass_matrix``; optionally
       call :meth:`ModalBasis.mass_orthonormalize`.
   * - Mass properties differ from the modal basis (:class:`ModelBuilder`)
     - The explicit ``mass``, ``com``, or ``inertia`` is inconsistent with the
       basis. Omit them to use the basis's rigid properties.
   * - Indefinite reduced mass matrix (:class:`~newton.solvers.SolverVBD`)
     - :math:`\mathbf{M}(\mathbf{0})` is not positive definite, typically
       because of inconsistent frame inertia. The solver falls back to explicit
       frame/mode coupling, which can diverge under stiff or compliant joints.
   * - Distributing Craig-Bampton interface
     - The interface motion differs from the coincident recovery sample, as
       for an RBE3-style element. Joints use the interface frame; contacts and
       rendering use the material sample. Expected for distributing
       interfaces.
   * - Damping not diagonal in the modal basis
     - Off-diagonal damping was discarded. Use modal or Rayleigh damping, or
       ``damping_projection="error"``.
   * - Negative modal diagonal entries clamped
     - The projected stiffness or damping has negative entries beyond roundoff;
       check the input matrices.
   * - ``derive_psi`` neighborhoods are rank-deficient
     - Samples are collinear or coplanar; unresolved rotation directions are
       set toward zero. Provide angular values or a 3D sample cloud.
   * - Block widths use the dense block solve (:class:`~newton.solvers.SolverVBD`)
     - A modal basis exceeds the tiled solve's shared memory; the slower dense
       solve is used. Pass ``elastic_block_solver="dense"`` to silence it.
   * - Could not Cholesky-factor the modal block (:class:`~newton.solvers.SolverVBD`)
     - In the block-sparse articulation solve, an elastic link's frame-and-mode
       block is not positive definite, so its modes take a diagonal step
       instead of being eliminated. Check the modal mass, stiffness, and frame
       inertia. Reported by the next ``step()`` or ``notify_model_changed()``
       call outside CUDA graph capture.
