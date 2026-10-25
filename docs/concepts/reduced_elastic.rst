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
   * - :class:`ModalGeneratorCraigBampton`
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

With ``rigid_articulation_solve="block_sparse_joints"``, elastic links in an
articulation with at least one other body are solved in the articulation's
direct factorization (see :ref:`Loop closure inside an articulation` for the
joint features this mode does not support) and use
``rigid_articulation_relaxation``. With the default
``rigid_articulation_elastic_solve="coupled"`` each such link is one block row
of size ``6 + n``: its body-axes block (inertia, inertial coupling, modal mass,
stiffness, and damping, contacts) is rotated to world axes about the center of
mass, every joint adds its exact projection onto both bodies' frame and modal
coordinates, and the whole articulation is factored at once with each row at
its own size. Directly jointed elastic links are therefore coupled exactly. An
articulation whose only body is an elastic link (for example from
:meth:`~newton.ModelBuilder.add_body_elastic`) is left to the local block solve
above and uses ``elastic_body_relaxation``.

If the factorization of an articulation fails (a pivot that is not positive
and finite, or a non-finite update), that articulation takes a diagonal
(Jacobi) step for the iteration, its entry of
:meth:`~newton.solvers.SolverVBD.rigid_articulation_elastic_solve_status` is
set, and the solver warns (see the table below). Its block metrics are NaN.
The diagonal step is not damped, so a system that stays indefinite can diverge
over many steps. In a test that removes the mode-mode coupling of two hinged
elastic beams from the assembled system, every factorization failed and the
state was no longer finite after 32 steps. Fix the model, or raise
``rigid_articulation_diagonal_regularization`` when the frame blocks are
nearly singular.

:meth:`~newton.solvers.SolverVBD.elastic_mode_solve_metrics` is filled with
NaN at the start of each step, and a NaN entry means the body was not
evaluated. Locally solved links report the norms of their own block. Links in a
coupled articulation report the norms of their rows of the whole articulation
system, so ``H delta`` includes the joint coupling to the other bodies'
updates and the frame regularization; frame rows are in world axes, which
leaves the norms unchanged. Links solved with
``rigid_articulation_elastic_solve="schur"``, links of an articulation whose
factorization failed, and every link before the first step or when no iteration ran stay NaN.

Accuracy of the articulation backends, as the maximum absolute difference of
the body poses and joint coordinates (positions in m, quaternion components,
and modal coordinates) over the trajectory from a local solve with 384
iterations, measured with the scripts accompanying the change on the CPU with
Warp 1.18.0. First, two directly hinged elastic beams (three modes
each), 120 steps of 1/480 s, in which the largest change of any entry is 0.65:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Iterations
     - ``"coupled"``
     - ``"schur"``
     - local
   * - 12
     - 8.7e-6
     - 1.7e-2
     -
   * - 24
     - 7.0e-6
     - 1.0e-2
     - 4.7e-4
   * - 48
     - 6.8e-6
     - 4.3e-3
     -
   * - 96
     - 8.5e-6
     - 8.5e-4
     - 4.8e-6
   * - 192
     - 8.6e-6
     - 3.9e-5
     -
   * - 384
     - 7.4e-6
     - 6.0e-6
     -

The coupled solve reaches the ``float32`` floor of this measure at 12
iterations; the Schur backend needs about 384. Second, seven articulations at
24 iterations, 60 steps (1/120 s for the loop, with relaxation 0.65; 1/480 s
otherwise):

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Articulation
     - ``"coupled"``
     - ``"schur"``
     - local
   * - Rigid base with an elastic leaf link
     - 3.4e-7
     - 5.2e-7
     - 6.3e-7
   * - Two directly hinged elastic links
     - 2.5e-6
     - 1.0e-2
     - 4.2e-4
   * - Elastic, rigid, elastic chain (1 and 3 modes)
     - 1.6e-6
     - 1.6e-6
     - 1.8e-3
   * - Elastic coupler and rocker loop
     - 2.6e-5
     - 2.6e-5
     - 1.2e-1
   * - Kinematic elastic parent with a rigid child
     - 4.6e-6
     - 4.6e-6
     - 3.9e-6
   * - Rigid base with two elastic leaves (2 and 5 modes)
     - 1.1e-6
     - 1.1e-6
     - 6.4e-7
   * - Three directly hinged elastic links (2, 4, 3 modes)
     - 2.8e-5
     - 6.9e-3
     - 1.4e-3

The same measures on a CUDA device are within about a factor of 2 of these values.

Cost of the backends, measured with the benchmark scripts accompanying the
change on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU with Warp
1.18.0; these are measurements of one device and these models, not
guarantees. The models are revolute chains of beams: a
chain of six rigid links with one or two elastic links, chains of elastic links
only, a four-bar linkage whose coupler is elastic, and 64 copies of a chain in
one model. Each backend ran 60 steps of 1/480 s at increasing iteration counts
(4, 8, 12, 16, 24, 48, 96, 192); the iterations to target are the fewest whose
largest difference from a local solve with 384 iterations, over every step and
every body pose and joint coordinate (modal coordinates included), is at most
1e-4, and the time to target is the time per step at that count, from a warmed
CUDA graph replay in one process:

.. list-table::
   :header-rows: 1
   :widths: 40 20 25 15

   * - Model
     - Iterations to target (local / ``"schur"`` / ``"coupled"``)
     - Time to target [ms/step] (local / ``"schur"`` / ``"coupled"``)
     - ``"coupled"`` / best other
   * - Rigid chain, one elastic link, 8 modes
     - 12 / 8 / 8
     - 1.45 / 1.38 / 1.03
     - 0.75
   * - Rigid chain, one elastic link, 32 modes
     - 12 / 12 / 12
     - 2.44 / 3.02 / 2.46
     - 1.01
   * - Rigid chain, one elastic link, 105 modes
     - 12 / 8 / 8
     - 5.27 / 4.31 / 3.66
     - 0.85
   * - Rigid chain, two elastic links, 8 modes
     - 12 / 8 / 8
     - 2.08 / 1.43 / 1.18
     - 0.82
   * - Rigid chain, two elastic links, 32 modes
     - 12 / 8 / 8
     - 3.56 / 2.12 / 1.99
     - 0.94
   * - Rigid chain, two elastic links, 105 modes
     - 12 / 8 / 8
     - 7.83 / 4.84 / 4.98
     - 1.03
   * - Two elastic links, 32 modes each
     - 24 / 12 / 8
     - 5.53 / 2.88 / 1.68
     - 0.58
   * - Two elastic links, 151 modes each
     - 24 / 12 / 8
     - 18.46 / 10.37 / 6.79
     - 0.65
   * - Four elastic links, 32 modes each
     - 24 / 16 / 8
     - 6.53 / 4.26 / 2.35
     - 0.55
   * - Four elastic links, 151 modes each
     - 24 / 16 / 8
     - 23.74 / 16.05 / 10.80
     - 0.67
   * - Four-bar with an elastic coupler, 32 modes
     - 8 / 8 / 8
     - 1.61 / 1.98 / 1.61
     - 1.00
   * - 64 copies of the one-link chain, 32 modes
     - 12 / 12 / 12
     - 3.87 / 4.34 / 3.94
     - 1.02

At a fixed 16 iterations, timed as the median of 5 fresh processes, a coupled
iteration costs 1.08, 1.17, 1.20, and 1.33 times a local one for chains of
three elastic links with 105 and 151 modes and of four elastic links with 105
and 151 modes. Its time to target is nevertheless lower because it needs 8
iterations where the local solve needs 24 and the Schur backend 16. The
coupled CUDA kernel runs 256 threads per articulation with high register use,
so it suits few large articulations; with many small ones occupancy drops. In a
kernel micro-benchmark, factoring and solving 1,024 four-link chains with 8
modes per link took 0.32 ms.

An articulation whose only body is an elastic link uses the local block solve
under ``"coupled"``, so its cost is that of the local solve (16 iterations,
median of 5 fresh processes):

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Model
     - ``"coupled"`` [ms/step]
     - ``"schur"`` [ms/step]
     - local [ms/step]
   * - Standalone elastic link, 8 modes
     - 0.716
     - 1.053
     - 0.716
   * - Standalone elastic link, 32 modes
     - 1.558
     - 1.793
     - 1.559
   * - Standalone elastic link, 105 modes
     - 3.982
     - 4.562
     - 3.989
   * - Standalone elastic link, 151 modes
     - 5.717
     - 6.757
     - 5.721

Memory of the coupled factorization: the stored values (the lower block
triangle with full diagonal blocks, after fill) against the dense system of
``6 + m`` rows per body, and the solver arrays of the coupled and Schur
backends. The CUDA kernel uses 59,280 bytes of shared memory per thread block
in every case, as reported by the kernel at construction:

.. list-table::
   :header-rows: 1
   :widths: 34 16 16 12 22

   * - Model
     - Stored values
     - Dense values
     - Fill ratio
     - Solver arrays [KiB] (``"coupled"`` / ``"schur"``)
   * - Four elastic links, 151 modes each
     - 172,543
     - 394,384
     - 0.44
     - 686.7 / 437.1
   * - Rigid chain, two elastic links, 105 modes
     - 27,702
     - 66,564
     - 0.42
     - 117.8 / 122.8

The ``"schur"`` backend, kept for comparison, eliminates each link's modes
onto the frame blocks of the link and its joint neighbors. Each iteration
factors every link's ``n x n`` modal block once and solves all of its
right-hand sides (the modal residual and six columns per neighbor frame) in
the same pass, before the articulation's 6 x 6 frame system is factored; the
modal blocks are padded to the largest mode count among the articulated links.
``elastic_block_solver`` selects that factorization as for the local solve: the
tiled path keeps one modal block in shared memory (about
``4 * (n**2 + 7 * n)`` bytes), and the dense path factors it in global memory. A
link whose frame-and-mode block is not positive definite cannot be
eliminated: its frame is then solved with its own frame rows only, its modes
take a diagonal (Jacobi) step after the frame solve, and the solver warns
once. Under ``"schur"`` a joint between two elastic links replaces their
mode-mode coupling by a conservative per-link term, which keeps the converged
solution but slows convergence, as the tables above show.

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
   * - Coupled articulation solve could not factor or solve (:class:`~newton.solvers.SolverVBD`)
     - In the block-sparse articulation solve with
       ``rigid_articulation_elastic_solve="coupled"``, an articulation's system
       had a pivot that is not positive and finite, or a non-finite update, so
       it took a diagonal step, which is not damped and can diverge if the
       failure persists. Check the modal mass, stiffness, and damping and
       :meth:`~newton.solvers.SolverVBD.rigid_articulation_elastic_solve_status`;
       a larger ``rigid_articulation_diagonal_regularization`` can make the frame
       blocks definite. Raised once per solver, at the first failure, by the
       next ``step()`` or ``notify_model_changed()`` call outside CUDA graph
       capture;
       :meth:`~newton.solvers.SolverVBD.rigid_articulation_elastic_solve_failure_count`
       counts every failure.
   * - Could not Cholesky-factor the modal block (:class:`~newton.solvers.SolverVBD`)
     - In the block-sparse articulation solve with
       ``rigid_articulation_elastic_solve="schur"``, an elastic link's
       frame-and-mode block is not positive definite, so its modes take a
       diagonal step instead of being eliminated. Check the modal mass,
       stiffness, and frame inertia. Reported once by the next ``step()`` or
       ``notify_model_changed()`` call outside CUDA graph capture.
