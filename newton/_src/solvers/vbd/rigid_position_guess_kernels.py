# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional rescaling of the rigid-body starting position guess in :class:`~newton.solvers.SolverVBD`.

The kernel lives in its own Warp module so that the default solver path (scale ``1.0``,
where it is never launched) compiles exactly the same modules as before the option existed.
"""

import warp as wp


@wp.kernel(enable_backward=False, module="unique")
def scale_rigid_position_guess_acceleration(
    dt: float,
    position_guess_acceleration_scale: float,
    gravity: wp.array[wp.vec3],
    body_world: wp.array[wp.int32],
    body_f: wp.array[wp.spatial_vector],
    body_inv_mass: wp.array[float],
    body_q: wp.array[wp.transform],
):
    """Rescale the external-acceleration part of the forward-step position guess.

    Runs right after ``forward_step_rigid_bodies``, which wrote the same predicted pose to the
    starting guess ``body_q`` and to the inertial target. Only the translation of the guess of
    dynamic bodies changes; the inertial target, predicted velocity, and rotation are untouched.

    Args:
        dt: Time step [s] used by the forward step.
        position_guess_acceleration_scale: Fraction of the external acceleration kept in the guess.
        gravity: Gravity vector array (world frame) [m/s²].
        body_world: World index for each body.
        body_f: Wrench integrated by the forward step (world frame) [N, N·m].
        body_inv_mass: Inverse masses used by the forward step (0 for kinematic bodies) [1/kg].
        body_q: Starting position guess (input/output).
    """
    tid = wp.tid()
    inv_m = body_inv_mass[tid]
    if inv_m == 0.0:
        return
    # Same gravity lookup and linear acceleration as the forward step's integrate_rigid_body.
    a_ext = gravity[body_world[tid]] + inv_m * wp.spatial_top(body_f[tid])
    shift = (position_guess_acceleration_scale - 1.0) * dt * dt * a_ext
    q = body_q[tid]
    body_q[tid] = wp.transform(wp.transform_get_translation(q) + shift, wp.transform_get_rotation(q))
