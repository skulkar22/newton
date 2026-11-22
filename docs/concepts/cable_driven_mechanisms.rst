.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Cable-Driven Mechanisms
=======================

This page explains how to simulate a cable-driven finger with
:class:`~newton.solvers.SolverVBD`: how the mechanism works, the two ways to
model its cable, the settings that matter, and how closely each model follows
a hand calculation. All helper functions named here live in
:github:`newton/examples/cable/cable_hand.py`. They are example code, not
part of the public API. The runnable example is
:github:`newton/examples/cable/example_cable_driven_finger.py`
(``python -m newton.examples cable_driven_finger``), and
:doc:`/tutorials/02_cable_driven_finger` walks through the same steps in a
notebook.

All quantities are in SI units unless a unit is written out.

A cable-driven finger
---------------------

A digit has three hinge joints. Cables close it and springs open it. Nothing
pushes: a cable can only pull.

- **Return springs with preload.** A torsion spring at every joint opens the
  digit. Each spring is preloaded, so its joint does not move until the cable
  torque exceeds a threshold.
- **Limits.** A cable rests at 2 N so that it never goes slack. The tension
  limit is 60 N.

The default hand has two kinds of digit.

**Two cables (index and middle finger).** Each cable has its own linear
actuator.

- The **knuckle cable** runs over a sheave of 8 mm radius on the first joint
  and ends on the first link. It bends the first joint only: from 13.75 N to
  29.5 N with the curl cable at rest.
- The **curl cable** runs over a pulley at every joint to an anchor in the
  fingertip. On the second and third joint its radii are 6.5 and 5 mm, and the
  springs there are light: the second joint moves between 4 N and 12 N, the
  third between 8 N and 18 N. On the first joint it passes an idler of 5 mm
  radius, so it moves the first joint only from 20.8 N on, when the outer
  joints are already closed.
- The two tensions give two degrees of freedom of pose: a bent first joint
  with straight outer joints, a hook (outer joints curled on a straight first
  joint), a pinch pose, a fist. ``two_cable_poses`` returns the tensions of
  these poses.
- **Cross-term.** The curl cable adds its tension times 5 mm to the torque on
  the first joint. To hold a first-joint angle while the curl tension changes,
  the knuckle tension is lowered by 5/8 of the change of the curl tension
  (``cable_tensions_for_knuckle``).

**One cable (ring finger, little finger, thumb).** One flexor runs from its
actuator over a sheave at every joint (8, 6.5 and 5 mm) to the fingertip.

- **Closing order.** The spring thresholds are graded. The first joint moves
  between 6 N and 14 N, the second between 10 N and 22 N, the third between
  16 N and 30 N. A rising tension closes the joints in the order first,
  second, third, and a falling tension opens them in the reverse order.
- A free digit is closed at 30 N; the rest of the range is grip force.

Every digit is underactuated: three joints on one or two actuators. The
springs decide the pose in free motion, and contact with an object decides it
during a grasp.

Two cable models
----------------

The same skeleton, springs, actuator and command work with either cable model.

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * -
     - Rod cable
     - Massless routed tendon
   * - What it is
     - Per cable, a chain of short capsule bodies (1 mm long for a 1 mm
       cable) joined by stretch and bend joints. Flanged pulleys are collision
       shapes, and contact alone keeps a cable in its grooves.
     - Per cable, one tendon with an attachment on the actuator, a rolling
       link per pulley and an attachment at its far end. It has no mass and
       no collision shapes.
   * - What it resolves
     - Cable mass, bending, contact with the pulleys, and a cable that leaves
       a groove.
     - The cable path, stretch, and capstan friction on the pulleys.
   * - Builder
     - ``build_rod_finger_cables_model``, ``create_rod_solver``,
       ``create_rod_collision_pipeline``
     - ``build_tendon_finger_cables_model``, ``make_tendon_solver``
   * - Size of a two-cable finger
     - 288 bodies, most of them rod segments
     - 6 bodies
   * - Step size at 60 Hz
     - 30 substeps, 32 iterations
     - 20 substeps, 16 iterations
   * - Devices
     - Meant for CUDA
     - CPU and CUDA
   * - Cable friction
     - Not validated
     - Validated at 0.02 and 0.05 on the one-cable digits
   * - Agreement with the hand calculation
     - See :ref:`cable-driven-agreement`
     - 0.07 deg

Use the tendon when the cable is a transmission: you need the joint angles,
the actuator travel and the tension, and you want them to match a hand
calculation. Use the rod cable when the cable itself is part of the question:
its contact with the pulleys, its mass, or what happens when it leaves a
groove.

Neither model is cheap per step: a finger of either kind runs well below real
time. A second cable does not make the tendon finger slower. Halving the
tendon finger's iterations to 8 is not a safe saving: measured on a one-cable
digit, one hold then missed a 0.2 deg angle target (0.203 deg), and with cable
friction the finger settled more slowly (tip tension 5 to 7 % off at the end
of a hold). This was not measured on the two-cable fingers. The examples keep
16 iterations.

Routing a flexor
----------------

A cable touches a list of **stations**. Each station is a circle fixed to
one body: an anchor on the actuator carriage, guides, one sheave per joint the
cable passes, and an anchor at its far end. A cable that runs to the fingertip
has six contact stations (five on the thumb); the knuckle cable has two. The
two cables of a finger run in parallel planes 3 mm apart, side by side on the
first joint.

- The sheave of a joint sits on the joint axis and is fixed to the body on the
  base side of that joint. Its radius is the moment arm of the cable about the
  joint (8.0, 6.5 and 5.0 mm for a single flexor; 5.0, 6.5 and 5.0 mm for the
  curl cable; 8.0 mm for the knuckle cable). ``moment_arm_matrix`` returns
  the arms of every cable of a digit about every joint.
- A guide sits next to a sheave, on the other side of the cable. It bends the
  cable onto the sheave so that the sheave has wrap when the finger is
  straight.
- ``digit_route`` returns the path of one cable through its stations at any
  pose: tangent points, free spans, wrap angles and total length.
  ``cable_takeup`` returns the cable taken up at a pose,
  :math:`\sum_j r_{cj} q_j`.

**A flexor unwraps as its joint flexes.** The cable runs on the palmar side of
the joint axis. When the joint flexes by :math:`q_j`, the two bodies turn
toward each other on that side and the cable needs less arc on the sheave: the
wrap is its straight-finger value minus :math:`q_j`. On the index finger the
three pulleys of the curl cable go from 120, 128 and 113 deg when straight to
30, 28 and 43 deg when fully flexed. A guide's wrap never changes.

**Wrap margins.** Keep the wrap on every joint sheave between 20 deg and
160 deg over the whole joint range. Below 20 deg the rod cable has few
segments on the sheave, and at zero the tendon loses the sheave. The tendon
rejects a wrap above 180 deg. In the default hand the smallest wrap is
26 deg and the largest is 132 deg.

The total wrap sets the sensitivity to friction: with a friction coefficient
:math:`\mu` the tension that reaches the fingertip is the actuator tension
times :math:`e^{-\mu \theta_\text{total}}`.

Springs and preload in VBD
--------------------------

A return spring with preload gives the torque
:math:`\tau_j = \tau_{0,j} + k_j q_j` toward extension, where :math:`q_j`
is measured from the lower joint limit. The preload :math:`\tau_{0,j}` acts
even at :math:`q_j = 0`.

A joint drive cannot express this by itself. VBD clamps drive targets to the
joint limits, so a target below the lower limit, which would produce the
preload, is not available. The spring is therefore built from two parts:

- The stiffness is the joint drive: ``target_ke`` is :math:`k_j`, the target
  is the lower limit, and ``target_kd`` is the joint damping.
- The preload is a constant generalized force :math:`-\tau_{0,j}` in
  ``Control.joint_f``. ``create_spring_preload`` writes it.

``add_digit_skeleton`` sets this up for the three joints of a digit.

The actuator governor
---------------------

VBD has no joint effort or speed limits. The linear actuator is a prismatic
joint with a position drive, and a small rule, the governor, moves the drive
target every substep. With ``s`` the measured travel of the carriage:

.. code-block:: text

   lo   = s + force_min / kp        # target that gives the rest tension
   hi   = s + force_limit / kp      # target that gives the force limit
   goal = clamp(goal_in, lo, hi)    # goal_in: commanded travel, or s + F_cmd / kp
   s_ref += clamp(goal - s_ref, -speed_limit * dt, +speed_limit * dt)
   s_ref  = min(s_ref, hi)          # the force limit wins over the speed limit
   drive target = s_ref

This gives three properties:

- **Force limit.** The drive force ``kp * (s_ref - s)`` never exceeds the
  limit (60 N).
- **Speed limit.** The target is rate-limited in both directions (30 mm/s), so
  opening is paced like closing. The limit is on the rate of the target: the
  carriage itself can exceed it for single substeps (35.8 mm/s was the largest
  peak in the hand's pose sequence, on the curl carriage of the index finger).
- **Slack take-up.** The drive force never falls below the rest tension
  (2 N), so the actuator takes up slack by itself.

The command is a cable tension (``GOVERNOR_FORCE``) or a carriage travel
(``GOVERNOR_POSITION``). The helper also holds the carriage itself, not only
the target, inside the speed limit when its load drops: the target then runs
2 mm/s below the limit. ``create_actuator_governor`` allocates one governor
for all actuators of the digits it is given, one command per cable;
call ``ActuatorGovernor.step`` once per substep before the solver step. All of
it is kernel launches, so it runs inside a captured CUDA graph.

**Two cables on one finger need both carriages to keep up.** A cable that
passes a joint on a pulley must be taken up when another cable moves that
joint. When the curl cable alone is stepped to a high tension and bends the
first joint, the knuckle carriage has to take up 8 mm per radian while the
curl carriage pays in 5 mm per radian. With both references at the same rate
the knuckle cable goes slack for a moment. The governor therefore has a
reference rate per actuator: the curl cable of a two-cable finger is ramped at
24 mm/s (``reference_rate_limit`` of the cable in the spec,
``actuator_reference_rate``), the other cables at the actuator's rate. In a
curl step to 60 N the lowest knuckle tension then stays at about 1 N instead
of dropping to zero, and curl moves take about 20 % longer.

Closed-form statics
-------------------

These formulas are the hand calculation that both cable models are compared
with.

Without friction
^^^^^^^^^^^^^^^^

Every span of cable :math:`c` carries its actuator tension :math:`T_c`. With
:math:`r_{cj}` the moment arm of cable :math:`c` about joint :math:`j` (zero
where the cable does not pass the joint), the torque balance of joint
:math:`j` gives

.. math::

   q_j(T) = \operatorname{clip}\!\left(\frac{\sum_c r_{cj}\, T_c - \tau_{0,j}}{k_j},\; 0,\; q_{\max,j}\right)

(``frictionless_angles``; for a digit with one cable the sum has one term).
The travel of the actuator of cable :math:`c`, measured from the state with
every cable at the rest tension :math:`T_0`, is the cable take-up plus
everything that stretches:

.. math::

   s_c(T) = \sum_j r_{cj}\, q_j(T) + (T_c - T_0)\left(\frac{L_c}{EA} + C_\text{anchor}\right)
            + C_\text{hinge} \sum_{j:\, r_{cj} \neq 0}\; \sum_{d:\, r_{dj} \neq 0} (T_d - T_0)

(``frictionless_row``).

- The take-up couples the actuators of a finger: the curl carriage travels
  when only the knuckle cable bends the first joint.
- :math:`L_c / EA` is the cable stretch. The default cable has
  :math:`EA = 2000` N; a cable of 150 mm gives 75 µm/N.
- :math:`C_\text{anchor}` is the compliance of the two cable anchors,
  20 µm/N in total.
- :math:`C_\text{hinge}` is the compliance of one flexion hinge. A hinge
  carries the tensions of all cables that have a pulley on it, and VBD holds
  a joint with a finite structural stiffness, so each hinge opens by that
  load over the stiffness. With a joint stiffness of 1e5 N/m this is 10 µm/N
  per hinge. The last term adds the opening of the hinges a cable passes.

At 60 N the stretch terms add up to about 7 mm on a cable that runs to the
fingertip, next to about 30 mm of take-up. Leaving out the hinge term alone
makes the travel 1.7 mm short.

With capstan friction
^^^^^^^^^^^^^^^^^^^^^

The friction tables are worked out for the digits with one cable. Walking
from the actuator toward the fingertip, each station with wrap
:math:`\theta_k` multiplies the tension:

.. math::

   T_k = T_{k-1}\, e^{-\mu \theta_k} \quad \text{(loading)}, \qquad
   T_k = T_{k-1}\, e^{+\mu \theta_k} \quad \text{(unloading)}.

- The wrap on the sheave of joint :math:`j` is its straight-finger wrap minus
  :math:`q_j`, so the three joint equations are solved one after the other.
- The torque on joint :math:`j` is :math:`r_j` times the tension of the span
  that *leaves* its sheave, because the sheave is carried by the body on the
  base side of the joint.
- When the actuator starts to pay out, the slip reverses from the actuator end
  first. A joint stays where it was at the peak until the reversed front has
  passed its sheave. This is the hysteresis.

``friction_tables`` returns the loading and unloading tables, and
``staircase_table`` the reference for a whole tension staircase. For the ring
finger:

.. list-table::
   :header-rows: 1

   * - :math:`\mu`
     - Joints start to move at [N], loading
     - Fully flexed at [N], loading
     - Leave the upper stops below [N], unloading
     - Tip tension over actuator tension, straight / flexed
   * - 0
     - 6.0 / 10.0 / 16.0
     - 14.0 / 22.0 / 30.0
     - same as loading
     - 1 / 1
   * - 0.02
     - 6.5 / 11.4 / 19.0
     - 14.6 / 23.9 / 34.4
     - 13.4 / 20.2 / 26.2
     - 0.798 / 0.873
   * - 0.05
     - 7.2 / 13.7 / 24.3
     - 15.5 / 27.2 / 42.1
     - 12.7 / 17.8 / 21.4
     - 0.568 / 0.713

With gravity
^^^^^^^^^^^^

``gravity_angles`` adds the weight of the links to the torque balance and the
weight of the actuator carriage to the cable tension. With gravity in the
flexion plane the first joint of the ring finger sits up to 11 deg away from
the gravity-free calculation, and that of the thumb up to 12 deg (closed
form, over all tensions and gravity directions). The example therefore keeps
the flexion axis vertical, where gravity does not load the joints.

Settings that matter
--------------------

The settings below are the ones both examples use. Three of the rod settings
were measured on the two-cable fingers with 1 mm segments, and the results
are stated with them: the iteration count, the reference inertia and the
contact buffer. The other observations (what goes wrong with the other
choice, and by how much) were measured on one-cable digits with 2 mm rod
segments; the two-cable fingers run with the same settings and meet the
agreement numbers given further down.

Shared by both models
^^^^^^^^^^^^^^^^^^^^^

- **External forces inside the solve.** Use
  ``rigid_external_force_mode="solve"``. The spring preload is a large
  constant joint force on light links. With the default mode the iterations
  do not converge on it once a cable is attached: the first joint flexes under
  the rest tension.
- **Gravity as body forces.** Build the model with zero gravity and add the
  weights to ``State.body_f`` every substep (``create_body_weight``). Model
  gravity shifts the inertial target instead, and the light links then sag
  several times too far when gravity acts in the flexion plane.
- **Joint stiffness and the hinge compliance it implies.** Both models use
  ``rigid_joint_linear_ke=1e5``. This is where the 10 µm/N per hinge in the
  travel formula comes from; measured values are 10.0 to 10.2 µm/N. A
  different joint stiffness needs a different hinge compliance in the hand
  calculation.
- **Soft joint stops.** The flexion stops use 300 N·m/rad and
  0.01 N·m·s/rad. A stop gives by 0.07 deg at the tension limit. The
  tendon model needs soft stops, see below; the rod model works with them too.

Rod cable
^^^^^^^^^

- **Block-sparse solve.** Use
  ``rigid_articulation_solve="block_sparse_levels"``. The ``local`` solve
  does not hold the rod finger: the joint angles are tens of degrees off, at
  16 and at 64 iterations.
- **Contact history.** Keep ``rigid_contact_history=True`` and create the
  collision pipeline with ``contact_matching="latest"``. Without the history
  the finger closes under the rest tension. Create the pipeline before the
  solver.
- **Contact buffer per body.** A link that carries flanged pulleys collects
  many rod contacts, more with shorter segments and with a second cable: a
  two-cable finger with 1 mm segments sees about 360 contacts on one body and
  overflows a buffer of 256. ``rod_contact_buffer`` returns the size for a
  digit (512 per cable with 1 mm segments); pass it as
  ``rigid_body_contact_buffer_size``. An overflow is only reported on the
  console.
- **Step size.** 30 substeps with 32 iterations per 60 Hz frame. There is no
  margin in the iteration count: with 16 iterations the two-cable finger
  misses the 2 deg angle rule (3.46 deg, and one hold does not settle).
- **Segment length and inertia.** The rods use 1 mm segments. The solver's
  angular regularization is set from a reference inertia
  (``rod_inertia_reference``, 2.5e-7 kg·m² for 1 mm segments), so that the
  short, light segments take the angular step of a larger body.
- **Pulley radius.** A chain of straight segments on a barrel takes up a
  little more cable per radian than an ideal cable. ``rod_barrel_radii`` makes
  the collision barrel slightly smaller so that the take-up equals the nominal
  radius.
- **Flanges.** The flanges leave a side clearance beside the cable; the two
  cables of a finger run in neighboring grooves on the first joint.
  ``RodTensionProbe.groove_offset`` returns the sideways offset of a rod in
  its groove, measured on the link that carries the groove, and
  ``RodTensionProbe.lateral_offset`` the offset from the digit plane on the
  palm, which adds the sideways give of the links under load.
- **Seating stroke.** Before anything is compared, run every cable once to
  16 N and back. A rod can start carried just above a pulley, and it then
  loses tension there as if it had friction until it has run once.

``create_rod_solver`` holds these settings.

Massless tendon
^^^^^^^^^^^^^^^

- **Local solve.** Tendons require ``rigid_articulation_solve="local"``.
- **Why the local solve needs soft stops.** The local solve updates one body
  at a time. A loaded joint stop is a stiff row between two light links, and a
  row like that can only be satisfied by many small alternating moves of the
  two links. In this finger most stops carry load most of the time: the
  preload holds the outer joints on their lower stops while the first joint
  moves. With stiff stops (1e4 N·m/rad) the moving joint drags links that
  are locked to their neighbors, the finger settles slowly, and it chatters
  at the tension limit. With the soft stops above it settles in about 0.2 s.
- **Limit floor.** Keep ``rigid_limit_rho_floor=True``. Without it the stops
  give way under load.
- **Step size.** 20 substeps with 16 iterations per 60 Hz frame, first-order
  time integration (``rigid_time_integration="bdf1"``).
- **Actuator damping.** The settle time depends on the velocity gain of the
  actuator, not on the joint stiffness. 20 N·s/m settles in about 0.2 s;
  200 N·s/m needs 1 to 2.4 s.
- **Friction.** With friction on the pulleys, ``make_tendon_solver`` selects
  the direct material solve. Drive the finger through the governor.
- **Coloring.** Call ``ModelBuilder.color()`` after all tendons are added.

``tendon_solver_settings`` holds these settings.

.. _cable-driven-agreement:

Agreement with the hand calculation
-----------------------------------

The test is a tension staircase without friction, with 0.5 s per move and a
hold at every level. A one-cable digit goes through 2, 8, 12, 16, 20, 26, 32
and 60 N and back down. A two-cable finger goes through each cable alone
(the other at rest), a grid of combined tensions, and the named poses. The
numbers are the largest difference from the closed form at the end of a hold.
The rod columns hold the larger of the values of the single finger and of the
hand; see the rod entry below for how they were run.

.. list-table::
   :header-rows: 1

   * - Digit
     - Cables
     - Tendon, joint angle [deg]
     - Tendon, travel [mm]
     - Rod, joint angle [deg]
     - Rod, travel [mm]
   * - Index
     - 2
     - 0.06
     - 0.012
     - 0.75
     - 0.13
   * - Middle
     - 2
     - 0.06
     - 0.011
     - 1.24
     - 0.22
   * - Ring
     - 1
     - 0.07
     - 0.018
     - 1.08
     - 0.13
   * - Little
     - 1
     - 0.07
     - 0.018
     - 0.80
     - 0.14
   * - Thumb
     - 1
     - 0.07
     - 0.018
     - 1.68
     - 0.23

- **Tendon.** The 0.06 to 0.07 deg is the deflection of a joint stop at 60 N.
  At the lower levels a two-cable finger is within 0.02 deg. The tension at both
  ends of every cable is within 0.04 % of the command.
- **Tendon with friction** (one-cable digits). With blended commands and
  0.5 s holds, the ring finger is within 0.68 deg, 0.11 mm and 1.5 % of the
  tip tension of the capstan tables at :math:`\mu = 0.02`, and within
  0.72 deg, 0.12 mm and 1.3 % at :math:`\mu = 0.05`, over the whole staircase
  up and down. Every pose inside the friction band is an
  equilibrium, so a finger that arrives with overshoot rests further from the
  table.
- **Rod.** A rod digit follows the closed form less closely than a tendon
  digit. After the seating stroke and with holds of 1 s, the largest joint
  angle difference of the single fingers is 0.75 deg (index), 1.24 deg
  (middle), 1.08 deg (ring), 0.79 deg (little) and 0.90 deg (thumb, with
  1 mm segments), with the travel within 0.22 mm. In the hand the same digits
  are within 0.63, 1.16, 1.08, 0.80 and 1.68 deg (the thumb with its 2 mm
  segments) and 0.23 mm. The examples test a rod digit against 1.3 deg
  (index, little), 1.6 deg (ring), 1.8 deg (middle) and 2.0 deg (thumb), and
  against 0.4 mm of travel. The joint angle error is the polygon effect of a
  chain of straight segments on small pulleys. A frictionless round pulley keeps tension times
  lever arm of the free span the same on both of its sides. For a chain, that
  lever arm swings with the position of the segment joints on the pulley, by
  :math:`(l / R)^2 / 8` for segments of length :math:`l` on a pulley of
  radius :math:`R`: 2 % for 2 mm segments on a 5 mm pulley and 0.5 % for
  1 mm segments. The tension therefore changes by up to that much from one
  side of a pulley to the other, and a joint in mid-range moves with it. This
  is a property of the segment model, not a solver error, and only shorter
  segments reduce it. The rods use 1 mm segments, the thumb's rod 2 mm.

Limits
------

- **Rod friction is not validated.** With friction of 0.02 and 0.05 the rod
  finger does not deliver the tip tension the capstan formula predicts. Use
  the tendon model for friction.
- **A rod needs time and a seating stroke.** A rod finger needs holds of about
  1 s or longer to settle on a level. On the GPU the state inside the seating
  stroke differs from run to run: in one of four runs of a two-cable finger
  the first joint stood 25 deg short inside the stroke, and the return of the
  stroke cleared it. Compare only after the stroke.
- **The tendon costs time per step.** A tendon finger has a handful of
  bodies, but its cost is set by the substeps and iterations of the solve,
  not by the size of the model: a frame of it runs well below real time.
- **Cable friction on two-cable fingers is not validated.** The capstan
  tables cover the digits with one cable.
- **One body at a time.** The local solve that tendons need converges slowly
  on stiff rows between light bodies. Stiff joint stops, stiff structural
  joints and heavy actuator damping all slow the finger down.
- **Do not reset a tendon solver.** Build a new solver instead of calling
  ``SolverVBD.reset()`` on a model with tendons.
- **Gravity in the flexion plane** changes the joint angles of a one-cable
  digit by up to 11 to 12 deg (closed form). Compare with ``gravity_angles``,
  not with the gravity-free tables.
- **Blend the tension commands of a two-cable digit.** The keyframe scripts
  and both examples move a command from one value to the next with a
  minimum-jerk curve over about 0.6 s (``StaircaseCommand``, ``HandScript``);
  ``HandControl.set_tension`` and ``ActuatorGovernor.set_command`` write a
  command as given, stepped. A stepped knuckle command that drives the first
  joint onto its stop while the curl cable is at the rest tension lets the
  curl cable dip to about 0.4 N for a moment (not slack; a rod stays in its
  grooves). In the opposite case, a curl command that bends the first joint,
  the slower curl reference rate (24 mm/s) is what keeps the knuckle cable
  from going slack.
- **Mass and inertia are those of simple solids.** Each body's mass comes from
  the design's mass budget, and its inertia is that of a simple solid carrying
  that mass: a link is a uniform box of its envelope with the center of mass
  at mid length, a carriage a cube, the thumb column a cylinder, the palm a
  set of boxes. The material tags of the visual parts set only the look, and
  rod segments take their mass from the cord's linear density. Compared with
  the part meshes (glass-filled nylon frames, brass and steel pulleys), the
  moment of inertia of each link about its flexion axis is within 11 %. The
  real centers of mass of the first two links sit 5 to 13 mm further out,
  because the pulleys are at the distal ends: the gravity torque at a joint is
  up to 17 % larger and the inertia reflected at a joint up to 19 % larger
  than in the model. Against the return spring preloads the gravity torque
  changes by at most 2.6 % of the preload. The thumb column is outside this
  comparison: its mass budget covers parts that have no mesh.
- **The hinges open under cable load.** The flexion hinges are compliant
  constraints (structural joint stiffness 1e5 N/m). With all cables of the
  tendon hand at the same tension, a hinge passed by one cable opens by
  0.40 mm at 40 N and by 0.60 mm at the 60 N limit, and the first joint of a
  two-cable finger, which two cables pass, by 0.80 mm and 1.20 mm: the sum of
  the tensions over the stiffness. The hinge axes stay parallel (below
  0.001 deg). The closed-form travel includes this opening. A stiffer joint
  setting was not studied.
- **The palm shell covers the ulnar ends of the palm pads.** The drawn palm
  pads are the upper surfaces of their collision capsules, but an object that
  touches the ulnar fifth of the heel pad or the last twentieth of the distal
  palm pad would be drawn 2 to 3.5 mm into the palm shell, which stands above
  the capsule there; the shipped grasps load the other parts of the pads.

Full hand
---------

The same helper module builds a right hand of human size with either cable
model. The example is
:github:`newton/examples/cable/example_cable_driven_hand.py`
(``python -m newton.examples cable_driven_hand``).

The hand
^^^^^^^^

- **Digits.** Four fingers and a thumb, each with its own link lengths, three
  joints and three return springs. The index and the middle finger have two
  cables, the ring finger, the little finger and the thumb one: seven cables
  on seven linear actuators.
- **Palm.** The palm carries the first guides and sheaves of every finger and
  all seven actuators. The fingers are splayed: the index finger by 5 deg
  toward the thumb, the ring and the little finger by 5 and 10 deg away from
  it.
- **Thumb opposition.** The thumb stands on a column that turns on one hinge
  across the palm, from 0 to 130 deg. This hinge has no cable. It is driven
  directly: a position drive of 20 N·m/rad with a torque limit of 0.8 N·m
  and a speed limit of 3 rad/s. The hinge line is the thumb's own cable line,
  so the thumb cable never crosses a joint and opposition does not change the
  thumb's flexion: with held tensions the joint angles of all digits change
  by at most 0.2 deg during a sweep from 0 to 130 deg and back (the swinging
  thumb's own joints) and return to within 0.01 deg.
- **Wrist.** The palm is a kinematic body that follows a scripted pose, or a
  body fixed to the world.
- **Size.** The whole hand is one articulation with the palm as root: 24 rigid
  bodies, 0.45 kg, 47 tendon links on seven tendons.
- **Keys.** Cables, actuators, tensions and travels of the hand are keyed by
  actuator: the digit name for a digit's first cable (``"index"`` is the
  index finger's curl cable) and ``"<digit>_<cable>"`` for a further one
  (``"index_knuckle"``). ``HandSkeleton.actuators`` lists them in the order of
  the governor arrays.

``build_tendon_hand_model`` builds the hand with massless tendons and
``make_tendon_hand_solver`` creates its solver with the settings of the
single finger. ``build_rod_hand_model`` and ``make_rod_hand_solver`` are the
counterparts for rod cables. The rod hand is built on a prescribed
(kinematic) palm: on a dynamic palm held by a fixed joint the fingers' 1 mm
rods do not start reliably, on a prescribed palm they start every time. A
scene that wants a still hand holds the palm still. The thumb's rod uses 2 mm
segments (``rod_segment_length`` of its cable in the spec), because 1 mm
segments on the thumb's light column do not start.

Hand-level control and step order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``create_hand_control`` returns one object for everything that acts on the
hand: the governor of the seven actuators, the opposition drive, the spring
preload, the weights as body forces, and the wrist. A ``HandScript`` holds
timed keyframes of wrist pose, opposition angle and cable commands, and writes
them into the control. A cable command is given per digit and resolved by
``hand_cable_tensions``: a tension for every cable of the digit, a tension
per cable (``{"curl": 9.0, "knuckle": 20.0}``), the name of a two-cable pose
(``"hook"``), a curl tension with the first-joint angle to hold, or a tension
for one actuator key. One substep is:

.. code-block:: python

   script.step(hand_control, dt)  # commands of the end of this substep
   hand_control.step(state_0, control, dt)  # forces, opposition, governor, then wrist
   pipeline.collide(state_0, contacts)  # only in a scene with contacts
   solver.step(state_0, state_1, control, contacts, dt)
   state_0, state_1 = state_1, state_0

The order inside ``HandControl.step`` matters. The governor and the opposition
drive measure the carriage travel and the opposition angle from the body
poses, and all poses must belong to the same instant. The wrist therefore
moves last. With the palm moved first, each travel is off by the palm's step
and each cable tension by the position gain times that.

Every call is a kernel launch, so a whole frame runs inside a captured CUDA
graph.

A rod hand starts with a **seating stroke**: every cable is pulled to 16 N and
released once before anything is shown or measured. A rod can start carried
just above a pulley, and it then loses tension there as if it had friction
until it has run once. The choreography starts with this stroke for both cable
models, with the thumb opposed so that it clears the closing fingers.

Two scenes
^^^^^^^^^^

- **Choreography.** ``hand_choreography`` returns a script of about 20 s:
  the seating stroke, then a loop of 16.6 s: wave, count from one to five, a
  pose only two-cable fingers can take (the index finger hooks while the
  middle finger bends at its first joint only), pinch with the index and with
  the middle finger, fist, open. A pose is data: a tension per cable and an
  opposition angle. The joint angles follow from the closed form, so the
  function also returns the expected angles of every pose and the time at
  which each pose is reached, the end of the hold that follows its move.
  Give it the gravity that the hand control applies: the cable tensions of the
  pinch poses depend on it. The hand has no contact between its links, so the
  poses and their order are chosen such that no digit meets another digit or
  the palm.
- **Grasp.** ``grasp_plan`` returns the keyframes of a complete grasp of a can
  (side power grasp), a ball (from above) or a block (three-digit pinch from
  above). For the can and the ball: approach, close the fingers,
  oppose the thumb, pull the thumb cable, lift by 100 mm, hold, tilt by
  30 deg, lower, release in reverse order, retreat. Only pads collide: one
  capsule per phalanx and two on the palm, against the object, the table and
  the pedestal. With the tendon hand:

  - **Can** (index, middle and ring finger and the thumb at 40 N): lifted by
    100.9 mm, moves 0.03 mm in the hand during the lifted hold and 0.06 mm
    during the tilt, and comes to rest 2.6 mm from where it was set down. The
    plan closes three fingers and the thumb: with all four fingers the can
    grasp fails.
  - **Ball** (fingers at 40 N, thumb at 12 N): lifted by 102.3 mm (the
    closing fingers draw it 2 mm up into the palm), moves 0.03 mm in the
    lifted hold and 0.06 mm during the tilt, and rolls back into its seat.
    The hand sets it down 1 mm lower than it picked it up; the seat carries
    nothing while the hand lowers and rests. While the fingers open, one pad
    still presses on the ball with about 4 to 5 N for about 0.75 s, and the
    seat carries 4.6 to 5.4 N during that time (the ball weighs 0.78 N); once
    the hand has left the seat carries the weight. Released from the pick-up
    height, the opening fingers sweep the ball off. The grasp works with the
    ball placed 15 to 20 mm toward the thumb in the palm only.
  - **Block** (index and middle finger against the straight thumb): the curl
    cables hold the shape of the fingers at 9 N and the knuckle cables swing
    them onto the block. The pinch lifts and releases the block for knuckle
    tensions from 26 N to the 60 N limit. From about 32 N the first joints
    rest on their stops and the grip force stays near 0.5 N per finger (three
    times the block's weight on the thumb): the knuckle tension closes the
    pinch, it does not scale the grip force. At 28 to 34 N the block slides
    2.4 to 3.4 mm in the hand at the start of the lift and stays held.

  No cable falls below 1.6 N in these runs.

  With rod cables the example offers the grasps of the can and the ball. In
  the retreat after the release the knuckle cables of the index and the middle
  finger dip to between 0.5 and 1.2 N, so the block pinch is offered with
  massless tendons only.

Two further examples let the same hand pick up objects that are not rigid.
Both run with massless tendons only: the rod grasp builder builds its own
scene with one of the three rigid objects.

- **Reduced elastic object.** ``cable_driven_hand_elastic`` lets the hand pick
  up a pair of headphones. The headphones are one reduced elastic body (see
  :doc:`reduced_elastic`): a floating frame plus six mode shapes of the
  headband, with the two ear cups riding rigidly on the band ends. The mode
  shapes come from a small beam model of the band arc and are handed to
  :class:`~newton.ModalGeneratorSampled`; they are linear and meant to give a
  stable, visibly flexible object, not accurate deflections. The four fingers
  hook under the crown cushion at 20 N of cable tension and press the band
  against the palm pads. The hand lifts the headphones by 98.2 mm against a
  scripted 100 mm (measured at the body frame, which shifts when the band
  bends). The headphones are a free body in an articulation of
  their own: only contact and friction of the pads hold them, and the supports
  carry nothing while they are lifted. At the end of the hold the pads carry
  1.124 N against a weight of 1.119 N. The crown of the band slips in the palm
  by 0.05 mm over the hold and by at most 0.17 mm through shake and tilt. The
  lowest cable tension of the run is 1.99 N.

  The band is soft and bends visibly. Its nominal modulus is 20 MPa, and
  ``--band-stiffness`` is a factor on it, allowed from 0.75 to 2500 (250 is
  practically rigid). In the hold the ear cups are 18.0 mm closer than at rest
  (15 % of their 117.5 mm), 23.9 mm at most; they swing when the hand shakes
  them and settle again. A band 250 times stiffer flexes by 0.08 mm.

  Limits of this example: the grip is one-sided, so one cup swings in more
  than the other. After the set-down the band is still bent in the hand and
  the cups come down inward of their cradles: one cup ends 12.5 mm over the
  edge of its cradle and the band stays 7.6 mm closed. A slower release does
  not change that. The displacement accuracy of the reduced elastic band was
  not checked and is not claimed. The example uses the massless tendons only.

  The example draws a finished pair of headphones and a stand on the simulated
  shapes (``HeadphonesLook``, ``headphones_stand_parts``): the drawn band is
  swept through the deformed collision mesh, the cups ride on the simulated
  cylinders, and the pads of the stand are the cradle faces. Helpers:
  ``cable_hand_objects.headphones_*``.
- **Finite-element object.** ``cable_driven_hand_soft`` lets the hand pick up
  and squeeze a soft foam ball of 56 mm, a tetrahedral mesh of 729 particles
  and 3072 tetrahedra with a stable Neo-Hookean material (a foam core of
  30 kPa under a skin four times stiffer, 19.3 g). Rigid links, cables and the
  soft body are solved by the one :class:`~newton.solvers.SolverVBD`, and the
  contact between the pads and the particles acts on both sides. The ball is a
  set of free particles: no joint, spring or constraint holds it, and its seat
  carries nothing while it is lifted. The hand comes down on the ball in its
  seat, closes around it in 2.5 s with 25 N on the finger cables and lifts it
  by 107.3 mm against a scripted 100 mm (the closing hand draws the ball 7 mm
  up into the palm). In the lifted hold the finger cables rise to 45 N and the
  thumb cable from 12 N to 30 N, are held for a second and lowered again,
  once: the gap between the thumb pad and the fingertip pads closes from
  32.8 mm to 23.2 mm (29 %), the ball is 5.7 mm flatter along the palm normal
  and 5.5 mm wider across than at rest (10 % of its 56 mm), and its volume is
  92.1 % in the squeeze. Then the hand lowers the ball into its seat and lets
  go; the ball returns to 100.00 % of its volume and comes to rest 2.4 mm
  from where it started, and no element inverts. The lowest cable tension of
  the run is 1.82 N, the highest 45.0 N: the tensions follow the commands.
  One loop takes 17.8 s. A CPU run and a CUDA run of it agree within 0.01 mm
  in the lift and in the closing of the pad gap.

  The pads and the seat touch the foam with a contact of 2e3 N/m instead of
  the 1e5 N/m of the rigid grasp scenes (see below); with the stiff contact
  the fingers chatter on the ball by several degrees from frame to frame.
  A limit of this scene: the pick needs the slow close. When the fingers
  close in 2.2 s or less they push the light ball off its seat before they
  are under it. The example uses the massless tendons only.

  The model has no gravity of its own (the hand control applies the weights of
  the bodies), so ``create_particle_weight`` applies the weight of the
  particles; only the pads and the support collide with the particles. Contact
  acts at the particle radius (3 mm), so the ball is drawn with
  ``create_soft_display_surface``: a fine sphere carried by the tetrahedra and
  offset outward by the particle radius, the surface on which the ball touches
  the pads and its seat. Helpers: ``cable_hand_objects.soft_*``.

Grasp contact stiffness
^^^^^^^^^^^^^^^^^^^^^^^

The pads, the object and the table use a contact stiffness of 1e5 N/m
(``pads.ke`` of the hand spec), a tenth of the value the design started with.
The reason is the local solve of the tendon hand, which solves one body at a
time: a contact of stiffness :math:`k` against a light object converges
poorly in 16 iterations when :math:`k\,\Delta t^2` is well above the object's
mass. At 1e6 N/m and a substep of 0.83 ms this is 0.69 kg against 0.15 kg for
the can, and on a hand with one cable per finger the can was thrown out of
the closing fingers in two of eight runs. At 1e5 N/m (0.07 kg) the same runs
held, and the rigid grasp scenes of this hand use that value. Take the
comparison as a guide, not as a limit: the block (0.040 kg) is lighter than
0.07 kg and is held. A pad force of 15 N then indents the contact by 0.15 mm.

The foam ball scene uses 2e3 N/m on the pads and on the seat. A soft body is
a set of particles, and the contact has to be matched to the mass of one
particle (26 mg), not to the whole ball (0.019 kg). The solver takes the mean
of the particle side (2e3 N/m) and the shape side of a contact: with pads of
1e5 N/m that mean is 51e3 N/m and :math:`k\,\Delta t^2` is 35 g, 1300 times
the mass of a particle, and the fingers chatter on the ball. With both sides
at 2e3 N/m it is 1.4 g, 50 times the mass of a particle; the foam around the
particle carries the rest, and the grasp is still.

Agreement and cost of the hands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tendon hand was given all five digits through their staircases together
(44 holds, the two-cable fingers through their two-input lists).

- **The digits behave in the hand as they do alone.** Over all holds the hand
  is within 0.07 deg, 0.02 mm and 0.04 % of the tension of the closed form,
  the single-finger numbers. The same holds with the thumb opposed by 60 deg.
- **Idle digits are not disturbed.** With one digit driven through its
  staircase, the joints of the other four move by at most 0.0002 deg and
  their cables stay within 0.005 N of the rest tension.
- **Moving wrist.** A wrist that moves by 100 mm and 45 deg while the tensions
  are held does not move digits that rest on their stops. Digits held between
  stops swing by up to 0.25 deg while the wrist accelerates (1 deg with
  gravity) and return to within 0.02 deg.
- **Pose sequence.** At the end of every hold of the choreography, averaged
  over its last 0.1 s, the joint angles are within 0.06 deg of the closed form
  of the pose, with and without gravity, and no cable falls below 1.3 N.
- **Cost.** The hand runs below real time, at 20 substeps and 16 iterations
  per frame. The cost is set by the substeps and iterations of the solve more
  than by the number of bodies: the whole hand costs little more than a
  single finger.
