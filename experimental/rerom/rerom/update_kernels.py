# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-side re-ROM update: Warp kernels and the per-body :class:`DeviceUpdater`.

The host part of an update (eigensolve, Ritz vectors, Rayleigh-Ritz) needs ``K_T``, ``f_int`` and the new
reference ``r_new``; everything else stays on the Model's device:

1. **Gather** (one device-to-host copy): :func:`_recover_kernel` forms ``r_new = r_ref + Phi q`` and the
   body-frame nodal velocity ``u = R^T v_c + (R^T omega) x (r_new - c) + Phi qd`` from ``state.joint_q`` /
   ``joint_qd``; the tangent assembler (when it is a :class:`~rerom.assembly_warp.WarpTetAssembler` on the
   same device) evaluates at ``r_new`` without synchronizing; ``M u`` and its momenta are reduced on the
   device; the modal state, momenta, energy, ``r_new``, ``f_int`` and the BSR values of ``K_T`` are packed
   into one staging buffer and copied to the host once.
2. **Scatter** (one host-to-device copy): the host packs the new nodal basis, rigid properties, modal
   mass/stiffness/damping, pre-stress, endpoint samples (and, for interface bodies, the projected state)
   into one inbox buffer.
3. **Rewrite kernels** (no host copies) write in place into the fixed-size Model/State/Control arrays:
   ``M Phi`` (BSR), the five inertia coupling integrals and the modal mass matrix
   (:func:`newton._src.sim.modal._reduced_inertia_from_pairs` semantics, float64 accumulation, deterministic
   fixed-order chunk reductions), render-point / shape-vertex rows (``local += phi_old q``, ``phi = W phi``)
   and endpoint frames/rows, rigid mass/COM/inertia, the momentum-consistent state projection (rigid part
   -> frame twist about the new COM, remainder -> M-orthogonal projection on the new modes), ``q = 0`` and
   the pre-stress rows of ``control.joint_f``.

An optional third copy (``diagnostics=True``) reads the momentum / kinetic-energy bookkeeping of the update
(a few doubles). Per step, :meth:`DeviceUpdater.write_prestress` (one kernel), :meth:`DeviceUpdater.poll`
(velocity history) and :meth:`DeviceUpdater.rotation_check` (element rotations of a sampled element set,
one 3-value flag read) avoid the ``joint_q`` / ``joint_qd`` host reads of the host path.

On the CPU device Warp arrays alias host memory, so "copies" are memcpys; the pair integral then uses a
BLAS GEMM on the aliased buffers by default (``cpu_blas``), which is faster than a serial Warp loop.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import warp as wp

__all__ = ["DeviceUpdater", "GatherResult", "InboxLayout", "count_host_transfers", "reference_reduced_inertia"]

CHUNKS = 64
"""Fixed chunk count of the deterministic two-pass reductions (independent of the device)."""


# ----------------------------------------------------------------------------------------------
# helpers (float64)
# ----------------------------------------------------------------------------------------------


@wp.func
def _quat_matrix(x: wp.float64, y: wp.float64, z: wp.float64, w: wp.float64) -> wp.mat33d:
    """Rotation matrix of an xyzw quaternion, normalized on the fly (``numerics.quat_to_matrix``)."""
    n = x * x + y * y + z * z + w * w
    s = wp.float64(0.0)
    if n > wp.float64(0.0):
        s = wp.float64(2.0) / n
    one = wp.float64(1.0)
    return wp.mat33d(
        one - s * (y * y + z * z),
        s * (x * y - z * w),
        s * (x * z + y * w),
        s * (x * y + z * w),
        one - s * (x * x + z * z),
        s * (y * z - x * w),
        s * (x * z - y * w),
        s * (y * z + x * w),
        one - s * (x * x + y * y),
    )


@wp.func
def _frame_rotation(joint_q: wp.array[wp.float32], q_start: int) -> wp.mat33d:
    return _quat_matrix(
        wp.float64(joint_q[q_start + 3]),
        wp.float64(joint_q[q_start + 4]),
        wp.float64(joint_q[q_start + 5]),
        wp.float64(joint_q[q_start + 6]),
    )


@wp.func
def _vec3_at(a: wp.array[wp.float32], start: int) -> wp.vec3d:
    return wp.vec3d(wp.float64(a[start]), wp.float64(a[start + 1]), wp.float64(a[start + 2]))


@wp.func
def _flat_vec3(a: wp.array[wp.float64], start: int) -> wp.vec3d:
    return wp.vec3d(a[start], a[start + 1], a[start + 2])


@wp.func
def _qmul(a: wp.vec4d, b: wp.vec4d) -> wp.vec4d:
    """Hamilton product of xyzw quaternions stored as vec4d (``numerics.quat_mul``)."""
    ax = a[0]
    ay = a[1]
    az = a[2]
    aw = a[3]
    bx = b[0]
    by = b[1]
    bz = b[2]
    bw = b[3]
    return wp.vec4d(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


@wp.func
def _quat_from_rotvec(theta: wp.vec3d) -> wp.vec4d:
    angle = wp.length(theta)
    if angle < wp.float64(1.0e-12):
        return wp.vec4d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    axis = theta / angle
    s = wp.sin(wp.float64(0.5) * angle)
    return wp.vec4d(axis[0] * s, axis[1] * s, axis[2] * s, wp.cos(wp.float64(0.5) * angle))


@wp.func
def _chunk_range(c: int, n: int, chunks: int) -> wp.vec2i:
    per = (n + chunks - 1) // chunks
    a = wp.min(c * per, n)
    b = wp.min(a + per, n)
    return wp.vec2i(a, b)


# ----------------------------------------------------------------------------------------------
# kernels: recovery, sparse products, reductions
# ----------------------------------------------------------------------------------------------


@wp.kernel
def _recover_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    q_start: int,
    qd_start: int,
    m: int,
    r_ref: wp.array[wp.vec3d],
    phi: wp.array[wp.vec3d],
    com: wp.array[wp.vec3d],
    with_velocity: int,
    x_out: wp.array[wp.vec3d],
    u_out: wp.array[wp.vec3d],
    vel_el_out: wp.array[wp.vec3d],
):
    """``x = r_ref + Phi q``; optionally ``u = R^T v_c + (R^T w) x (x - c) + Phi qd`` and ``Phi qd``."""
    n = wp.tid()
    x = r_ref[n]
    v = wp.vec3d()
    for i in range(m):
        p = phi[n * m + i]
        x = x + p * wp.float64(joint_q[q_start + 7 + i])
        if with_velocity != 0:
            v = v + p * wp.float64(joint_qd[qd_start + 6 + i])
    x_out[n] = x
    if with_velocity != 0:
        R = _frame_rotation(joint_q, q_start)
        RT = wp.transpose(R)
        vl = RT @ _vec3_at(joint_qd, qd_start)
        wl = RT @ _vec3_at(joint_qd, qd_start + 3)
        u_out[n] = (vl + wp.cross(wl, x - com[0])) + v
        vel_el_out[n] = v


@wp.kernel
def _bsr_mv_kernel(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.mat33d],
    x: wp.array[wp.vec3d],
    k: int,
    y: wp.array[wp.vec3d],
):
    """``y[n, i] = sum_b M[n, col_b] x[col_b, i]`` for ``k`` interleaved vector fields (``[node * k + i]``)."""
    t = wp.tid()
    n = t // k
    i = t - n * k
    acc = wp.vec3d()
    for b in range(offsets[n], offsets[n + 1]):
        acc = acc + values[b] @ x[columns[b] * k + i]
    y[t] = acc


@wp.kernel
def _moments_partial_kernel(
    x: wp.array[wp.vec3d],
    u: wp.array[wp.vec3d],
    mu: wp.array[wp.vec3d],
    n_nodes: int,
    chunks: int,
    partial: wp.array[wp.float64],
):
    """Per chunk: ``sum M u``, ``sum x x M u``, ``sum u . M u`` (7 values)."""
    c = wp.tid()
    r = _chunk_range(c, n_nodes, chunks)
    P = wp.vec3d()
    L = wp.vec3d()
    k = wp.float64(0.0)
    for n in range(r[0], r[1]):
        f = mu[n]
        P = P + f
        L = L + wp.cross(x[n], f)
        k = k + wp.dot(u[n], f)
    for j in range(3):
        partial[c * 7 + j] = P[j]
        partial[c * 7 + 3 + j] = L[j]
    partial[c * 7 + 6] = k


@wp.kernel
def _dot_partial_kernel(
    a: wp.array[wp.vec3d], b: wp.array[wp.vec3d], n_nodes: int, chunks: int, partial: wp.array[wp.float64]
):
    c = wp.tid()
    r = _chunk_range(c, n_nodes, chunks)
    k = wp.float64(0.0)
    for n in range(r[0], r[1]):
        k = k + wp.dot(a[n], b[n])
    partial[c] = k


@wp.kernel
def _scalar_partial_kernel(a: wp.array[wp.float64], n: int, chunks: int, partial: wp.array[wp.float64]):
    c = wp.tid()
    r = _chunk_range(c, n, chunks)
    k = wp.float64(0.0)
    for j in range(r[0], r[1]):
        k = k + a[j]
    partial[c] = k


@wp.kernel
def _sum_partials_kernel(
    partial: wp.array[wp.float64], width: int, chunks: int, out: wp.array[wp.float64], offset: int
):
    """``out[offset + w] = sum_c partial[c * width + w]`` in a fixed order (deterministic)."""
    w = wp.tid()
    acc = wp.float64(0.0)
    for c in range(chunks):
        acc = acc + partial[c * width + w]
    out[offset + w] = acc


@wp.kernel
def _pairs_partial_kernel(
    phi: wp.array[wp.vec3d],
    mphi: wp.array[wp.vec3d],
    m: int,
    n_nodes: int,
    chunks: int,
    partial: wp.array[wp.float64],
):
    """``pairs[(3i+a), (3j+b)] = sum_n phi[n, i][a] (M phi)[n, j][b]`` per chunk."""
    t = wp.tid()
    w = 9 * m * m
    c = t // w
    rem = t - c * w
    row = rem // (3 * m)
    col = rem - row * (3 * m)
    i = row // 3
    a = row - 3 * i
    j = col // 3
    b = col - 3 * j
    r = _chunk_range(c, n_nodes, chunks)
    acc = wp.float64(0.0)
    for n in range(r[0], r[1]):
        acc = acc + phi[n * m + i][a] * mphi[n * m + j][b]
    partial[t] = acc


@wp.kernel
def _mode_integrals_partial_kernel(
    x: wp.array[wp.vec3d],
    mphi: wp.array[wp.vec3d],
    m: int,
    n_nodes: int,
    chunks: int,
    partial: wp.array[wp.float64],
):
    """Per chunk and mode: ``sum M phi`` (3), ``sum (M phi) x x`` (3), ``sum outer(M phi, x)`` (9)."""
    t = wp.tid()
    c = t // m
    i = t - c * m
    r = _chunk_range(c, n_nodes, chunks)
    lin = wp.vec3d()
    ang = wp.vec3d()
    cen = wp.mat33d()
    for n in range(r[0], r[1]):
        f = mphi[n * m + i]
        p = x[n]
        lin = lin + f
        ang = ang + wp.cross(f, p)
        cen = cen + wp.outer(f, p)
    base = (c * m + i) * 15
    for k in range(3):
        partial[base + k] = lin[k]
        partial[base + 3 + k] = ang[k]
        for l in range(3):
            partial[base + 6 + 3 * k + l] = cen[k, l]


@wp.kernel
def _modal_dot_partial_kernel(
    mphi: wp.array[wp.vec3d],
    w_el: wp.array[wp.vec3d],
    m: int,
    n_nodes: int,
    chunks: int,
    partial: wp.array[wp.float64],
):
    """Per chunk and mode: ``sum_n (M phi_i)[n] . w[n]``."""
    t = wp.tid()
    c = t // m
    i = t - c * m
    r = _chunk_range(c, n_nodes, chunks)
    acc = wp.float64(0.0)
    for n in range(r[0], r[1]):
        acc = acc + wp.dot(mphi[n * m + i], w_el[n])
    partial[t] = acc


@wp.kernel
def _node_norm_kernel(x: wp.array[wp.vec3d], r_ref: wp.array[wp.vec3d], out: wp.array[wp.float64]):
    n = wp.tid()
    out[n] = wp.length(x[n] - r_ref[n])


@wp.kernel
def _max_partial_kernel(a: wp.array[wp.float64], n: int, chunks: int, partial: wp.array[wp.float64]):
    c = wp.tid()
    r = _chunk_range(c, n, chunks)
    k = wp.float64(0.0)
    for j in range(r[0], r[1]):
        k = wp.max(k, a[j])
    partial[c] = k


@wp.kernel
def _trigger_flag_kernel(
    angle_partial: wp.array[wp.float64],
    angle_chunks: int,
    has_theta: int,
    theta_tol: wp.float64,
    disp_partial: wp.array[wp.float64],
    disp_chunks: int,
    has_disp: int,
    disp_tol: wp.float64,
    inv_length: wp.float64,
    flag: wp.array[wp.float64],
):
    """``flag = [code, theta_max, disp_rel]``; code 1 displacement, 2 rotation, 0 none (host-path order)."""
    code = wp.float64(0.0)
    theta = wp.float64(0.0)
    disp = wp.float64(0.0)
    if has_disp != 0:
        for c in range(disp_chunks):
            disp = wp.max(disp, disp_partial[c])
        disp = disp * inv_length
        if disp > disp_tol:
            code = wp.float64(1.0)
    if has_theta != 0 and code == wp.float64(0.0):
        for c in range(angle_chunks):
            theta = wp.max(theta, angle_partial[c])
        if theta > theta_tol:
            code = wp.float64(2.0)
    flag[0] = code
    flag[1] = theta
    flag[2] = disp


# ----------------------------------------------------------------------------------------------
# kernels: packing
# ----------------------------------------------------------------------------------------------


@wp.kernel
def _pack_header_kernel(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    q_start: int,
    qd_start: int,
    m: int,
    qdd: wp.array[wp.float64],
    gravity: wp.array[wp.vec3f],
    world: int,
    out: wp.array[wp.float64],
    qdd_offset: int,
    gravity_offset: int,
):
    """``[q (m), qd (m), p (3), quat (4), v_c (3), omega (3)]`` at 0, the velocity history at ``qdd_offset``
    and the body's world gravity at ``gravity_offset``."""
    gv = gravity[world]
    for k in range(3):
        out[gravity_offset + k] = wp.float64(gv[k])
    for i in range(m):
        out[i] = wp.float64(joint_q[q_start + 7 + i])
        out[m + i] = wp.float64(joint_qd[qd_start + 6 + i])
        out[qdd_offset + i] = qdd[i]
    for k in range(7):
        out[2 * m + k] = wp.float64(joint_q[q_start + k])
    for k in range(6):
        out[2 * m + 7 + k] = wp.float64(joint_qd[qd_start + k])


@wp.kernel
def _pack_vec3_kernel(src: wp.array[wp.vec3d], out: wp.array[wp.float64], offset: int):
    n = wp.tid()
    v = src[n]
    out[offset + 3 * n] = v[0]
    out[offset + 3 * n + 1] = v[1]
    out[offset + 3 * n + 2] = v[2]


@wp.kernel
def _pack_mat33_kernel(src: wp.array[wp.mat33d], out: wp.array[wp.float64], offset: int):
    b = wp.tid()
    v = src[b]
    for i in range(3):
        for j in range(3):
            out[offset + 9 * b + 3 * i + j] = v[i, j]


@wp.kernel
def _pack_flag_kernel(src: wp.array[wp.int32], out: wp.array[wp.float64], offset: int):
    out[offset] = wp.float64(src[0])


@wp.kernel
def _unpack_vec3_kernel(src: wp.array[wp.float64], offset: int, out: wp.array[wp.vec3d]):
    t = wp.tid()
    out[t] = _flat_vec3(src, offset + 3 * t)


# ----------------------------------------------------------------------------------------------
# kernels: rewrite
# ----------------------------------------------------------------------------------------------


@wp.func
def _solve_dense(a: wp.array[wp.float64], b: wp.array[wp.float64], n: int):
    """In-place Gaussian elimination with partial pivoting: ``a`` (n*n, overwritten), ``b`` -> solution."""
    for k in range(n):
        piv = k
        big = wp.abs(a[k * n + k])
        for r in range(k + 1, n):
            v = wp.abs(a[r * n + k])
            if v > big:
                big = v
                piv = r
        if piv != k:
            for c in range(n):
                t = a[k * n + c]
                a[k * n + c] = a[piv * n + c]
                a[piv * n + c] = t
            t = b[k]
            b[k] = b[piv]
            b[piv] = t
        d = a[k * n + k]
        for r in range(k + 1, n):
            f = a[r * n + k] / d
            if f != wp.float64(0.0):
                for c in range(k, n):
                    a[r * n + c] = a[r * n + c] - f * a[k * n + c]
                b[r] = b[r] - f * b[k]
    for kk in range(n):
        k = n - 1 - kk
        s = b[k]
        for c in range(k + 1, n):
            s = s - a[k * n + c] * b[c]
        b[k] = s / a[k * n + k]


@wp.kernel
def _rigid_split_kernel(
    staging: wp.array[wp.float64],
    moments_offset: int,
    inbox: wp.array[wp.float64],
    rigid_offset: int,
    mode: int,
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    q_start: int,
    qd_start: int,
    com_old: wp.array[wp.vec3d],
    scratch: wp.array[wp.float64],
    coeff: wp.array[wp.float64],
):
    """Frame twist of the new reference (body frame, about the new COM): ``coeff = [t, w]``.

    mode 0 (free dynamic frame): M-orthogonal rigid split ``G coeff = Y^T M u`` with ``Y`` the rigid motions
    of ``r_new`` about ``c_new`` (``Y^T M u = [sum M u, sum x x M u - c_new x sum M u]``).
    mode 1/2 (constrained basis or kinematic frame): keep the frame twist, ``t = R^T v_c + w_l x (c_new - c)``.
    """
    c_new = _flat_vec3(inbox, rigid_offset)
    R = _frame_rotation(joint_q, q_start)
    RT = wp.transpose(R)
    if mode == 0:
        P = _flat_vec3(staging, moments_offset)
        L = _flat_vec3(staging, moments_offset + 3)
        rot = L - wp.cross(c_new, P)
        for r in range(6):
            for c in range(6):
                scratch[r * 6 + c] = inbox[rigid_offset + 3 + r * 6 + c]
        for k in range(3):
            coeff[k] = P[k]
            coeff[3 + k] = rot[k]
        _solve_dense(scratch, coeff, 6)
    else:
        wl = RT @ _vec3_at(joint_qd, qd_start + 3)
        t = RT @ _vec3_at(joint_qd, qd_start) + wp.cross(wl, c_new - com_old[0])
        for k in range(3):
            coeff[k] = t[k]
            coeff[3 + k] = wl[k]


@wp.kernel
def _elastic_velocity_kernel(
    mode: int,
    u: wp.array[wp.vec3d],
    vel_el: wp.array[wp.vec3d],
    x: wp.array[wp.vec3d],
    coeff: wp.array[wp.float64],
    inbox: wp.array[wp.float64],
    rigid_offset: int,
    w_el: wp.array[wp.vec3d],
):
    """Old elastic nodal velocity: ``u - (t + w x (x - c_new))`` (mode 0) or ``Phi_old qd`` (modes 1, 2)."""
    n = wp.tid()
    if mode == 0:
        t = wp.vec3d(coeff[0], coeff[1], coeff[2])
        w = wp.vec3d(coeff[3], coeff[4], coeff[5])
        w_el[n] = u[n] - (t + wp.cross(w, x[n] - _flat_vec3(inbox, rigid_offset)))
    else:
        w_el[n] = vel_el[n]


@wp.kernel
def _modal_state_kernel(
    mode: int,
    m: int,
    pairs: wp.array[wp.float64],
    proj: wp.array[wp.float64],
    inbox: wp.array[wp.float64],
    qd_given_offset: int,
    scratch: wp.array[wp.float64],
    qd_new: wp.array[wp.float64],
    work: wp.array[wp.float64],
    diag: wp.array[wp.float64],
    diag_offset: int,
):
    """``Mf qd = Phi_new^T M w_el`` (``Mf = trace pairs``) or the given ``qd``; kinetic-energy bookkeeping.

    ``proj = [Phi_new^T M w_el (m), w_el^T M w_el]``; ``diag[diag_offset:] = [KE_el, KE_lost]``.
    """
    w3 = 3 * m
    for i in range(m):
        for j in range(m):
            mf = wp.float64(0.0)
            for a in range(3):
                mf = mf + pairs[(3 * i + a) * w3 + 3 * j + a]
            scratch[i * m + j] = mf
            work[m + i * m + j] = mf
        if mode == 2:
            qd_new[i] = inbox[qd_given_offset + i]
        else:
            qd_new[i] = proj[i]
    if mode != 2:
        _solve_dense(scratch, qd_new, m)
    # KE lost = 1/2 |w_el - Phi_new qd|_M^2 = 1/2 (w M w - 2 qd . a + qd^T Mf qd)
    s = wp.float64(0.0)
    for i in range(m):
        acc = wp.float64(0.0)
        for j in range(m):
            acc = acc + work[m + i * m + j] * qd_new[j]
        s = s + qd_new[i] * (acc - wp.float64(2.0) * proj[i])
    wmw = proj[m]
    diag[diag_offset] = wp.float64(0.5) * wmw
    diag[diag_offset + 1] = wp.float64(0.5) * (wmw + s)


@wp.kernel
def _write_modal_kernel(
    m: int,
    max_modes: int,
    e: int,
    mode_start: int,
    pairs: wp.array[wp.float64],
    integrals: wp.array[wp.float64],
    inbox: wp.array[wp.float64],
    modal_offset: int,
    mode_mass: wp.array[wp.float32],
    mode_stiffness: wp.array[wp.float32],
    mode_damping: wp.array[wp.float32],
    coupling_linear: wp.array[wp.vec3f],
    coupling_angular: wp.array[wp.vec3f],
    coupling_centrifugal: wp.array[wp.mat33f],
    coupling_coriolis: wp.array[wp.vec3f],
    coupling_quadratic: wp.array[wp.mat33f],
    mode_mass_matrix: wp.array[wp.float32],
):
    """Modal diagonals, the five coupling integrals and ``Mf = trace P`` of one body (pair block ``e``)."""
    t = wp.tid()
    i = t // m
    j = t - i * m
    w3 = 3 * m
    P = wp.mat33d()
    Pji = wp.mat33d()
    for a in range(3):
        for b in range(3):
            P[a, b] = pairs[(3 * i + a) * w3 + 3 * j + b]
            Pji[a, b] = pairs[(3 * j + a) * w3 + 3 * i + b]
    idx = e * max_modes * max_modes + i * max_modes + j
    coupling_quadratic[idx] = wp.mat33f(P)
    mode_mass_matrix[idx] = wp.float32(P[0, 0] + P[1, 1] + P[2, 2])
    # C[i, j] = sum phi_j x (m phi_i): antisymmetric differences of P[j, i]
    coupling_coriolis[idx] = wp.vec3f(
        wp.float32(Pji[1, 2] - Pji[2, 1]), wp.float32(Pji[2, 0] - Pji[0, 2]), wp.float32(Pji[0, 1] - Pji[1, 0])
    )
    if j == 0:
        g = mode_start + i
        base = i * 15
        mode_mass[g] = wp.float32(inbox[modal_offset + i])
        mode_stiffness[g] = wp.float32(inbox[modal_offset + m + i])
        mode_damping[g] = wp.float32(inbox[modal_offset + 2 * m + i])
        coupling_linear[g] = wp.vec3f(
            wp.float32(integrals[base]), wp.float32(integrals[base + 1]), wp.float32(integrals[base + 2])
        )
        coupling_angular[g] = wp.vec3f(
            wp.float32(integrals[base + 3]), wp.float32(integrals[base + 4]), wp.float32(integrals[base + 5])
        )
        cen = wp.mat33f()
        for a in range(3):
            for b in range(3):
                cen[a, b] = wp.float32(integrals[base + 6 + 3 * a + b])
        coupling_centrifugal[g] = cen


@wp.kernel
def _sample_shift_kernel(
    rows: wp.array[wp.int32],
    staging: wp.array[wp.float64],
    m: int,
    max_modes: int,
    phi_rows: wp.array[wp.vec3f],
    local: wp.array[wp.vec3f],
):
    """Render point / shape vertex rest position: ``local += phi_old q`` (old rows, before the rewrite)."""
    s = wp.tid()
    row = rows[s]
    shift = wp.vec3d()
    for i in range(m):
        shift = shift + wp.vec3d(phi_rows[row * max_modes + i]) * staging[i]
    local[row] = wp.vec3f(wp.vec3d(local[row]) + shift)


@wp.kernel
def _sample_rows_kernel(
    rows: wp.array[wp.int32],
    ptr: wp.array[wp.int32],
    nodes: wp.array[wp.int32],
    weights: wp.array[wp.float64],
    phi: wp.array[wp.vec3d],
    m: int,
    max_modes: int,
    phi_rows: wp.array[wp.vec3f],
):
    """New per-mode sample rows ``phi_row[s, i] = sum_k W[s, k] phi_new[node_k, i]`` (one thread per row and mode)."""
    t = wp.tid()
    s = t // m
    i = t - s * m
    acc = wp.vec3d()
    for k in range(ptr[s], ptr[s + 1]):
        acc = acc + weights[k] * phi[nodes[k] * m + i]
    phi_rows[rows[s] * max_modes + i] = wp.vec3f(acc)


@wp.kernel
def _endpoint_kernel(
    ep_joint: wp.array[wp.int32],
    ep_side: wp.array[wp.int32],
    ep_index: wp.array[wp.int32],
    staging: wp.array[wp.float64],
    inbox: wp.array[wp.float64],
    ep_offset: int,
    n_ep: int,
    m: int,
    max_modes: int,
    endpoint_phi: wp.array[wp.vec3f],
    endpoint_psi: wp.array[wp.vec3f],
    joint_X_p: wp.array[wp.transformf],
    joint_X_c: wp.array[wp.transformf],
    theta_out: wp.array[wp.vec3d],
):
    """Move an endpoint frame to its deformed pose at ``q`` (linear endpoint update) and install new rows."""
    k = wp.tid()
    j = ep_joint[k]
    ep = ep_index[k]
    X = joint_X_p[j]
    if ep_side[k] == 1:
        X = joint_X_c[j]
    pos = wp.vec3d(wp.transform_get_translation(X))
    theta = wp.vec3d()
    for i in range(m):
        qi = staging[i]
        pos = pos + wp.vec3d(endpoint_phi[ep * max_modes + i]) * qi
        theta = theta + wp.vec3d(endpoint_psi[ep * max_modes + i]) * qi
    r = wp.transform_get_rotation(X)
    rot = wp.vec4d(wp.float64(r[0]), wp.float64(r[1]), wp.float64(r[2]), wp.float64(r[3]))
    if wp.length(theta) > wp.float64(1.0e-12):
        rot = _qmul(_quat_from_rotvec(theta), rot)
        rot = rot / wp.length(rot)
    theta_out[k] = theta
    Xn = wp.transformf(
        wp.vec3f(pos), wp.quatf(wp.float32(rot[0]), wp.float32(rot[1]), wp.float32(rot[2]), wp.float32(rot[3]))
    )
    if ep_side[k] == 1:
        joint_X_c[j] = Xn
    else:
        joint_X_p[j] = Xn
    for i in range(m):
        base = ep_offset + (k * m + i) * 3
        endpoint_phi[ep * max_modes + i] = wp.vec3f(_flat_vec3(inbox, base))
        endpoint_psi[ep * max_modes + i] = wp.vec3f(_flat_vec3(inbox, base + 3 * n_ep * m))


@wp.kernel
def _counter_rotate_kernel(
    theta: wp.array[wp.vec3d],
    n_ep: int,
    body: int,
    apply: int,
    body_q: wp.array[wp.transformf],
    diag: wp.array[wp.float64],
    diag_offset: int,
):
    """Counter-rotate the body's rest rotation by the mean endpoint rotation; ``diag`` gets the spread."""
    big = wp.float64(0.0)
    mean = wp.vec3d()
    for k in range(n_ep):
        big = wp.max(big, wp.length(theta[k]))
        mean = mean + theta[k]
    spread = wp.float64(0.0)
    if n_ep > 0 and big > wp.float64(1.0e-12):
        mean = mean / wp.float64(n_ep)
        for k in range(n_ep):
            spread = wp.max(spread, wp.length(theta[k] - mean))
        if apply != 0:
            Q = _quat_from_rotvec(mean)
            Qinv = wp.vec4d(-Q[0], -Q[1], -Q[2], Q[3])
            X = body_q[body]
            r = wp.transform_get_rotation(X)
            rot = _qmul(wp.vec4d(wp.float64(r[0]), wp.float64(r[1]), wp.float64(r[2]), wp.float64(r[3])), Qinv)
            rot = rot / wp.length(rot)
            body_q[body] = wp.transformf(
                wp.transform_get_translation(X),
                wp.quatf(wp.float32(rot[0]), wp.float32(rot[1]), wp.float32(rot[2]), wp.float32(rot[3])),
            )
    diag[diag_offset] = spread


@wp.kernel
def _rigid_write_kernel(
    inbox: wp.array[wp.float64],
    rigid_offset: int,
    body: int,
    frame_dynamic: int,
    body_mass: wp.array[wp.float32],
    body_com: wp.array[wp.vec3f],
    body_inertia: wp.array[wp.mat33f],
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33f],
    com_dev: wp.array[wp.vec3d],
):
    """Rigid mass, COM, inertia (inverses for a dynamic frame) from the inbox ``[c(3), G(36), m, I(9), 1/m, I^-1(9)]``."""
    c = _flat_vec3(inbox, rigid_offset)
    o = rigid_offset + 39
    mass = inbox[o]
    I = wp.mat33d()
    Ii = wp.mat33d()
    for a in range(3):
        for b in range(3):
            I[a, b] = inbox[o + 1 + 3 * a + b]
            Ii[a, b] = inbox[o + 11 + 3 * a + b]
    body_mass[body] = wp.float32(mass)
    body_com[body] = wp.vec3f(c)
    body_inertia[body] = wp.mat33f(I)
    if frame_dynamic != 0:
        body_inv_mass[body] = wp.float32(inbox[o + 10])
        body_inv_inertia[body] = wp.mat33f(Ii)
    com_dev[0] = c


@wp.kernel
def _write_state_kernel(
    mode: int,
    m: int,
    q_start: int,
    qd_start: int,
    body: int,
    coeff: wp.array[wp.float64],
    qd_new: wp.array[wp.float64],
    inbox: wp.array[wp.float64],
    given_offset: int,
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    has_body_qd: int,
    body_qd: wp.array[wp.spatial_vectorf],
    qd_hist: wp.array[wp.float64],
):
    """``q = 0``, frame twist (world, ``v_c = R t``, ``omega = R w``) and modal velocities."""
    R = _frame_rotation(joint_q, q_start)
    v = wp.vec3d()
    w = wp.vec3d()
    if mode == 2:
        v = _flat_vec3(inbox, given_offset)
        w = _flat_vec3(inbox, given_offset + 3)
    else:
        v = R @ wp.vec3d(coeff[0], coeff[1], coeff[2])
        w = R @ wp.vec3d(coeff[3], coeff[4], coeff[5])
    for i in range(m):
        joint_q[q_start + 7 + i] = wp.float32(0.0)
        joint_qd[qd_start + 6 + i] = wp.float32(qd_new[i])
        qd_hist[i] = qd_new[i]
    for k in range(3):
        joint_qd[qd_start + k] = wp.float32(v[k])
        joint_qd[qd_start + 3 + k] = wp.float32(w[k])
    if has_body_qd != 0:
        body_qd[body] = wp.spatial_vectorf(
            wp.float32(v[0]),
            wp.float32(v[1]),
            wp.float32(v[2]),
            wp.float32(w[0]),
            wp.float32(w[1]),
            wp.float32(w[2]),
        )


@wp.kernel
def _copy_f64_kernel(src: wp.array[wp.float64], offset: int, dst: wp.array[wp.float64]):
    t = wp.tid()
    dst[t] = src[offset + t]


@wp.kernel
def _prestress_kernel(f: wp.array[wp.float64], start: int, joint_f: wp.array[wp.float32]):
    """Modal rows of ``control.joint_f``: ``joint_f[qd_start + 6 + i] = modal_force - g0 (- f_nl)``."""
    i = wp.tid()
    joint_f[start + i] = wp.float32(f[i])


@wp.kernel
def _velocity_history_kernel(
    joint_qd: wp.array[wp.float32],
    qd_start: int,
    inv_dt: wp.float64,
    hist: wp.array[wp.float64],
    qdd: wp.array[wp.float64],
):
    """``qdd = (qd - hist) / dt``, ``hist = qd`` (the host path's modal acceleration estimate)."""
    i = wp.tid()
    qd = wp.float64(joint_qd[qd_start + 6 + i])
    qdd[i] = (qd - hist[i]) * inv_dt
    hist[i] = qd


@wp.kernel
def _vec3_from_flat_kernel(src: wp.array[wp.float64], out: wp.array[wp.vec3d]):
    t = wp.tid()
    out[t] = _flat_vec3(src, 3 * t)


# ----------------------------------------------------------------------------------------------
# host round-trip counting (benchmarks, tests)
# ----------------------------------------------------------------------------------------------


@contextlib.contextmanager
def count_host_transfers():
    """Count host <-> Warp array data movements inside the block (any device, any caller).

    ``d2h``: ``wp.array.numpy()`` calls; ``h2d``: ``wp.array.assign`` from host data; ``h2d_alloc``: Warp arrays
    created from host data. On the CPU device these are views or memcpys, but they are the call sites that
    become device<->host copies (and, for reads, synchronizations) on a GPU. Yields a dict updated in place.
    """
    from warp._src import types as wtypes  # noqa: PLC0415

    counts = {"d2h": 0, "h2d": 0, "h2d_alloc": 0}
    cls = wtypes.array
    orig_numpy, orig_assign, orig_init = cls.numpy, cls.assign, cls._init_from_data
    depth = [0]  # inside numpy()/assign(): their internal temporaries are not separate transfers

    def numpy(self, *args, **kwargs):
        if depth[0] == 0:
            counts["d2h"] += 1
        depth[0] += 1
        try:
            return orig_numpy(self, *args, **kwargs)
        finally:
            depth[0] -= 1

    def assign(self, src):
        if depth[0] == 0 and not isinstance(src, cls):
            counts["h2d"] += 1
        depth[0] += 1
        try:
            return orig_assign(self, src)
        finally:
            depth[0] -= 1

    def init(self, data, *args, **kwargs):
        if depth[0] == 0 and not isinstance(data, cls):
            counts["h2d_alloc"] += 1
        return orig_init(self, data, *args, **kwargs)

    cls.numpy, cls.assign, cls._init_from_data = numpy, assign, init
    try:
        yield counts
    finally:
        cls.numpy, cls.assign, cls._init_from_data = orig_numpy, orig_assign, orig_init


# ----------------------------------------------------------------------------------------------
# reference (numpy) of the integral kernels, for tests
# ----------------------------------------------------------------------------------------------


def reference_reduced_inertia(points: np.ndarray, phi: np.ndarray, mass_phi: np.ndarray) -> dict:
    """float64 integrals exactly as ``_reduced_inertia_from_pairs`` computes them before its float32 cast."""
    points = np.asarray(points, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    mass_phi = np.asarray(mass_phi, dtype=np.float64)
    s, m = phi.shape[:2]
    w = 3 * m
    pairs = phi.transpose(1, 2, 0).reshape((w, s)) @ mass_phi.reshape((s, w))
    quadratic = pairs.reshape((m, 3, m, 3)).transpose(0, 2, 1, 3)
    p_ji = np.swapaxes(quadratic, 0, 1)
    coriolis = np.stack(
        [p_ji[..., 1, 2] - p_ji[..., 2, 1], p_ji[..., 2, 0] - p_ji[..., 0, 2], p_ji[..., 0, 1] - p_ji[..., 1, 0]],
        axis=-1,
    )
    return {
        "pairs": pairs,
        "quadratic": quadratic,
        "mass_matrix": np.einsum("ijcc->ij", quadratic),
        "coriolis": coriolis,
        "linear": mass_phi.sum(axis=0),
        "angular": np.cross(mass_phi, points[:, None, :]).sum(axis=0),
        "centrifugal": np.einsum("smc,sd->mcd", mass_phi, points),
    }


# ----------------------------------------------------------------------------------------------
# DeviceUpdater
# ----------------------------------------------------------------------------------------------


@dataclass
class InboxLayout:
    """Offsets of the host-to-device inbox (float64)."""

    phi: int
    rigid: int  # c_new(3), G(36), mass(1), inertia(9), inv_mass(1), inv_inertia(9) = 59
    modal: int  # mode_mass, stiffness, damping, prestress, qd_given (5 m)
    given: int  # v_c(3), omega(3) of mode 2
    ep: int  # ep_phi (E m 3), ep_psi (E m 3)
    size: int


@dataclass
class GatherResult:
    """Host view of the update-start gather."""

    q: np.ndarray
    qd: np.ndarray
    p: np.ndarray
    quat: np.ndarray
    v_c: np.ndarray
    omega: np.ndarray
    qdd: np.ndarray
    gravity: np.ndarray  # world gravity of the body's world (3,)
    moments: np.ndarray  # [sum M u (3), sum x x M u (3), u . M u] (body frame, about the body origin)
    x_new: np.ndarray  # (N, 3)
    W: float | None
    f_int: np.ndarray | None
    K_values: np.ndarray | None
    inverted: bool


class DeviceUpdater:
    """Device buffers and kernel launches of one body's device-side update (see the module docstring).

    Args:
        ctrl: The owning :class:`rerom.ReROMController` (layout, mass matrix, sample weights, basis).
        cpu_blas: On the CPU device, form the pair integral with a BLAS GEMM on the aliased buffers
            instead of the serial Warp kernel (same float64 definition; summation order differs).
    """

    def __init__(self, ctrl, *, cpu_blas: bool = True):
        import scipy.sparse as sp

        self.ctrl = ctrl
        model = ctrl.model
        self.device = model.device
        dev = self.device
        self.cpu_blas = bool(cpu_blas) and dev.is_cpu
        lay = ctrl.layout
        self.lay = lay
        N, m = ctrl.N, ctrl.m
        self.N, self.m = N, m
        self.n_ep = len(lay.endpoints)

        # mass matrix as 3x3 BSR (one upload)
        Mb = sp.csr_matrix(ctrl.M).tobsr(blocksize=(3, 3))
        Mb.sort_indices()
        self._m_offsets = wp.array(Mb.indptr.astype(np.int32), dtype=wp.int32, device=dev)
        self._m_columns = wp.array(Mb.indices.astype(np.int32), dtype=wp.int32, device=dev)
        self._m_values = wp.array(np.ascontiguousarray(Mb.data, dtype=np.float64), dtype=wp.mat33d, device=dev)

        # persistent nodal buffers
        self.r_ref = wp.zeros(N, dtype=wp.vec3d, device=dev)
        self.phi = wp.zeros(N * m, dtype=wp.vec3d, device=dev)
        self.mphi = wp.zeros(N * m, dtype=wp.vec3d, device=dev)
        self.com = wp.zeros(1, dtype=wp.vec3d, device=dev)
        self.x = wp.zeros(N, dtype=wp.vec3d, device=dev)
        self.u = wp.zeros(N, dtype=wp.vec3d, device=dev)
        self.vel_el = wp.zeros(N, dtype=wp.vec3d, device=dev)
        self.mu = wp.zeros(N, dtype=wp.vec3d, device=dev)
        self.w_el = wp.zeros(N, dtype=wp.vec3d, device=dev)
        self.qd_hist = wp.zeros(m, dtype=wp.float64, device=dev)
        self.qdd = wp.zeros(m, dtype=wp.float64, device=dev)
        self.qd_new = wp.zeros(m, dtype=wp.float64, device=dev)
        self.prestress = wp.zeros(m, dtype=wp.float64, device=dev)
        self.prestress_host: np.ndarray | None = None
        self.coeff = wp.zeros(6, dtype=wp.float64, device=dev)
        self.scratch = wp.zeros(max(36, m * m), dtype=wp.float64, device=dev)
        self.work = wp.zeros(m + m * m, dtype=wp.float64, device=dev)
        self.pairs = wp.zeros(9 * m * m, dtype=wp.float64, device=dev)
        self.integrals = wp.zeros(15 * m, dtype=wp.float64, device=dev)
        self.proj = wp.zeros(m + 1, dtype=wp.float64, device=dev)
        width = max(9 * m * m, 15 * m, m, 7)
        self.partial = wp.zeros(CHUNKS * width, dtype=wp.float64, device=dev)
        self.theta = wp.zeros(max(self.n_ep, 1), dtype=wp.vec3d, device=dev)
        # diagnostics: [P_l after (3), L_l after (3), uMu after, KE_el, KE_lost, spread]
        self.diag = wp.zeros(10, dtype=wp.float64, device=dev)

        # staging (device -> host)
        asm = getattr(ctrl.assembler, "inner", ctrl.assembler)  # unwrap rerom.assembly.HostTangentAssembler
        self.device_assembly = bool(
            hasattr(asm, "evaluate_device") and getattr(asm, "device", None) is not None and asm.device == dev
        )
        self.device_assembler = asm if self.device_assembly else None
        self.nnzb = int(asm.nnz_blocks) if self.device_assembly else 0
        self.n_tets = int(asm.n_tets) if self.device_assembly else 0
        self.off_moments = 2 * m + 13
        self.off_energy = self.off_moments + 7
        self.off_inverted = self.off_energy + 1
        self.off_qdd = self.off_inverted + 1
        self.off_gravity = self.off_qdd + m
        self.off_x = self.off_gravity + 3
        self.off_f = self.off_x + 3 * N
        self.off_K = self.off_f + (3 * N if self.device_assembly else 0)
        self.staging_size = self.off_K + 9 * self.nnzb
        self.staging = wp.zeros(self.staging_size, dtype=wp.float64, device=dev)

        # inbox (host -> device)
        E = self.n_ep
        off_phi = 0
        off_rigid = off_phi + 3 * N * m
        off_modal = off_rigid + 59
        off_given = off_modal + 5 * m
        off_ep = off_given + 6
        self.inbox_layout = InboxLayout(off_phi, off_rigid, off_modal, off_given, off_ep, off_ep + 6 * E * m)
        self.inbox = wp.zeros(self.inbox_layout.size, dtype=wp.float64, device=dev)
        self._inbox_host = np.zeros(self.inbox_layout.size, dtype=np.float64)

        # endpoints
        self._ep_joint = wp.array(np.array([ep.joint for ep in lay.endpoints] or [0], np.int32), device=dev)
        self._ep_side = wp.array(np.array([ep.side for ep in lay.endpoints] or [0], np.int32), device=dev)
        self._ep_index = wp.array(np.array([ep.endpoint for ep in lay.endpoints] or [0], np.int32), device=dev)

        # sample rows: render points, shape vertices (CSR of the sample->node weights)
        a, b = lay.render_range
        self._render = self._sample_csr(np.arange(a, b), lay.render_samples)
        rows, samples = [], []
        offsets = np.cumsum([0] + [r[1] - r[0] for r in lay.shape_ranges])[:-1]
        for (sa, sb), k0 in zip(lay.shape_ranges, offsets, strict=True):
            rows.append(np.arange(sa, sb))
            samples.append(lay.shape_samples[k0 : k0 + (sb - sa)])
        self._shape = (
            self._sample_csr(np.concatenate(rows), np.concatenate(samples)) if rows else self._sample_csr([], [])
        )

        # trigger
        self._trigger_ids = None
        self._trigger_angles = None
        self._node_norm = wp.zeros(N, dtype=wp.float64, device=dev)
        self._trig_partial = wp.zeros(CHUNKS, dtype=wp.float64, device=dev)
        self._disp_partial = wp.zeros(CHUNKS, dtype=wp.float64, device=dev)
        self.flag = wp.zeros(3, dtype=wp.float64, device=dev)
        self.x_trig = wp.zeros(N, dtype=wp.vec3d, device=dev)

        if self.cpu_blas:  # aliasing host views of the CPU-device buffers (no copies)
            self._cpu_views = (
                self.phi.numpy().reshape(N, 3 * m),
                self.mphi.numpy().reshape(N, 3 * m),
                self.pairs.numpy().reshape(3 * m, 3 * m),
            )

        self.upload_basis(ctrl.Phi, ctrl.r_ref, ctrl.com)

    # ------------------------------------------------------------------------------------------
    def _sample_csr(self, rows, samples):
        dev = self.device
        rows = np.asarray(rows, dtype=np.int64)
        samples = np.asarray(samples, dtype=np.int64)
        if rows.size == 0:
            return None
        W = self.ctrl.sample_weights[samples].tocsr()
        W.sort_indices()
        return (
            int(rows.size),
            wp.array(rows.astype(np.int32), dtype=wp.int32, device=dev),
            wp.array(W.indptr.astype(np.int32), dtype=wp.int32, device=dev),
            wp.array(W.indices.astype(np.int32), dtype=wp.int32, device=dev),
            wp.array(W.data.astype(np.float64), dtype=wp.float64, device=dev),
        )

    def upload_basis(self, Phi: np.ndarray, r_ref: np.ndarray, com: np.ndarray) -> None:
        """Full upload of the basis, reference and COM (construction, folds, external changes)."""
        nodal = np.ascontiguousarray(np.transpose(np.asarray(Phi, np.float64).reshape(self.N, 3, -1), (0, 2, 1)))
        self.phi.assign(nodal.reshape(-1, 3))
        self.r_ref.assign(np.asarray(r_ref, np.float64).reshape(-1, 3))
        self.com.assign(np.asarray(com, np.float64).reshape(1, 3))

    def set_prestress(self, f: np.ndarray, *, only_if_changed: bool = False) -> None:
        """Upload the modal pre-stress rows (``modal_force - g0 - f_nl``); ``only_if_changed`` compares with the
        last host value first (no copy when equal)."""
        f = np.asarray(f, dtype=np.float64).reshape(self.m)
        if only_if_changed and self.prestress_host is not None and np.array_equal(f, self.prestress_host):
            return
        self.prestress.assign(f)
        self.prestress_host = f.copy()

    # ------------------------------------------------------------------------------------------
    def _launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)

    def _reduce(self, kernel_inputs_partial, width, out, offset, kernel, dim_per_chunk=1):
        self._launch(kernel, CHUNKS * dim_per_chunk, kernel_inputs_partial)
        self._launch(_sum_partials_kernel, width, [self.partial, width, CHUNKS, out, offset])

    def _bsr_mv(self, x, k, y):
        self._launch(_bsr_mv_kernel, self.N * k, [self._m_offsets, self._m_columns, self._m_values, x, k, y])

    def _recover(self, state, with_velocity: bool, x_out=None):
        lay = self.lay
        self._launch(
            _recover_kernel,
            self.N,
            [
                state.joint_q,
                state.joint_qd,
                lay.q_start,
                lay.qd_start,
                self.m,
                self.r_ref,
                self.phi,
                self.com,
                int(with_velocity),
                self.x if x_out is None else x_out,
                self.u,
                self.vel_el,
            ],
        )

    def _moments(self, out, offset):
        """Reduce ``[sum M u, sum x x M u, u . M u]`` of the current ``x``, ``u`` into ``out[offset:]``."""
        self._bsr_mv(self.u, 1, self.mu)
        self._reduce([self.x, self.u, self.mu, self.N, CHUNKS, self.partial], 7, out, offset, _moments_partial_kernel)

    # ------------------------------------------------------------------------------------------
    # stage 1: gather
    # ------------------------------------------------------------------------------------------
    def gather(self, state, *, need_assembly: bool) -> GatherResult:
        """Recover ``r_new`` and ``u``, assemble on the device (when possible) and copy one staging buffer."""
        lay, m, N = self.lay, self.m, self.N
        st = self.staging
        self._recover(state, True)
        self._launch(
            _pack_header_kernel,
            1,
            [
                state.joint_q,
                state.joint_qd,
                lay.q_start,
                lay.qd_start,
                m,
                self.qdd,
                self.ctrl.model.gravity,
                max(lay.world, 0),
                st,
                self.off_qdd,
                self.off_gravity,
            ],
        )
        self._moments(st, self.off_moments)
        self._launch(_pack_vec3_kernel, N, [self.x, st, self.off_x])
        assembled = need_assembly and self.device_assembly
        if assembled:
            We, f, Kv, inverted = self.device_assembler.evaluate_device(self.x, need_tangent=True)
            self._reduce([We, self.n_tets, CHUNKS, self.partial], 1, st, self.off_energy, _scalar_partial_kernel)
            self._launch(_pack_flag_kernel, 1, [inverted, st, self.off_inverted])
            self._launch(_pack_vec3_kernel, N, [f, st, self.off_f])
            self._launch(_pack_mat33_kernel, self.nnzb, [Kv, st, self.off_K])
        host = st.numpy()
        if self.device.is_cpu:
            host = host.copy()
        return GatherResult(
            q=host[:m].copy(),
            qd=host[m : 2 * m].copy(),
            p=host[2 * m : 2 * m + 3].copy(),
            quat=host[2 * m + 3 : 2 * m + 7].copy(),
            v_c=host[2 * m + 7 : 2 * m + 10].copy(),
            omega=host[2 * m + 10 : 2 * m + 13].copy(),
            qdd=host[self.off_qdd : self.off_qdd + m].copy(),
            gravity=host[self.off_gravity : self.off_gravity + 3].copy(),
            moments=host[self.off_moments : self.off_moments + 7].copy(),
            x_new=host[self.off_x : self.off_x + 3 * N].reshape(N, 3),
            W=float(host[self.off_energy]) if assembled else None,
            f_int=host[self.off_f : self.off_f + 3 * N] if assembled else None,
            K_values=host[self.off_K : self.off_K + 9 * self.nnzb] if assembled else None,
            inverted=bool(host[self.off_inverted] != 0.0) if assembled else False,
        )

    # ------------------------------------------------------------------------------------------
    # stage 2/3: scatter + rewrite
    # ------------------------------------------------------------------------------------------
    def pack_inbox(
        self,
        *,
        Phi: np.ndarray,
        com_new: np.ndarray,
        G: np.ndarray | None,
        mass: float,
        inertia: np.ndarray,
        inv_mass: float,
        inv_inertia: np.ndarray,
        mode_mass: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
        prestress: np.ndarray,
        qd_given: np.ndarray | None = None,
        v_c_given: np.ndarray | None = None,
        omega_given: np.ndarray | None = None,
        ep_phi: np.ndarray | None = None,
        ep_psi: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fill the host inbox (no copy yet)."""
        L, m, N, E = self.inbox_layout, self.m, self.N, self.n_ep
        h = self._inbox_host
        h[L.phi : L.phi + 3 * N * m] = np.transpose(np.asarray(Phi).reshape(N, 3, m), (0, 2, 1)).reshape(-1)
        r = L.rigid
        h[r : r + 3] = com_new
        h[r + 3 : r + 39] = 0.0 if G is None else np.asarray(G, np.float64).reshape(-1)
        h[r + 39] = mass
        h[r + 40 : r + 49] = np.asarray(inertia, np.float64).reshape(-1)
        h[r + 49] = inv_mass
        h[r + 50 : r + 59] = np.asarray(inv_inertia, np.float64).reshape(-1)
        o = L.modal
        h[o : o + m] = mode_mass
        h[o + m : o + 2 * m] = stiffness
        h[o + 2 * m : o + 3 * m] = damping
        h[o + 3 * m : o + 4 * m] = prestress
        h[o + 4 * m : o + 5 * m] = 0.0 if qd_given is None else qd_given
        h[L.given : L.given + 3] = 0.0 if v_c_given is None else v_c_given
        h[L.given + 3 : L.given + 6] = 0.0 if omega_given is None else omega_given
        if E:
            h[L.ep : L.ep + 3 * E * m] = np.asarray(ep_phi, np.float64).reshape(-1)
            h[L.ep + 3 * E * m : L.ep + 6 * E * m] = np.asarray(ep_psi, np.float64).reshape(-1)
        return h

    def _integrals(self) -> None:
        """``M Phi`` and the pair / per-mode integrals of the current ``phi`` at ``x`` (device buffers)."""
        m, N = self.m, self.N
        self._bsr_mv(self.phi, m, self.mphi)
        if self.cpu_blas:
            phi, mphi, pairs = self._cpu_views
            np.matmul(phi.T, mphi, out=pairs)
        else:
            self._reduce(
                [self.phi, self.mphi, m, N, CHUNKS, self.partial],
                9 * m * m,
                self.pairs,
                0,
                _pairs_partial_kernel,
                dim_per_chunk=9 * m * m,
            )
        self._reduce(
            [self.x, self.mphi, m, N, CHUNKS, self.partial],
            15 * m,
            self.integrals,
            0,
            _mode_integrals_partial_kernel,
            dim_per_chunk=m,
        )

    def rewrite(self, state, control, *, mode: int, counter_rotate: bool, diagnostics: bool) -> np.ndarray | None:
        """One host-to-device copy of the inbox, then the rewrite kernels (in place). Returns the diagnostics."""
        model = self.ctrl.model
        lay, m, N, L = self.lay, self.m, self.N, self.inbox_layout
        inbox, st = self.inbox, self.staging
        inbox.assign(self._inbox_host)

        # new basis, M Phi, integrals
        self._launch(_unpack_vec3_kernel, N * m, [inbox, L.phi, self.phi])
        self._integrals()

        # state projection
        self._launch(
            _rigid_split_kernel,
            1,
            [
                st,
                self.off_moments,
                inbox,
                L.rigid,
                mode,
                state.joint_q,
                state.joint_qd,
                lay.q_start,
                lay.qd_start,
                self.com,
                self.scratch,
                self.coeff,
            ],
        )
        self._launch(
            _elastic_velocity_kernel, N, [mode, self.u, self.vel_el, self.x, self.coeff, inbox, L.rigid, self.w_el]
        )
        self._reduce([self.mphi, self.w_el, m, N, CHUNKS, self.partial], m, self.proj, 0, _modal_dot_partial_kernel, m)
        self._bsr_mv(self.w_el, 1, self.mu)
        self._reduce([self.w_el, self.mu, N, CHUNKS, self.partial], 1, self.proj, m, _dot_partial_kernel)
        self._launch(
            _modal_state_kernel,
            1,
            [
                mode,
                m,
                self.pairs,
                self.proj,
                inbox,
                L.modal + 4 * m,
                self.scratch,
                self.qd_new,
                self.work,
                self.diag,
                7,
            ],
        )

        # model arrays
        self._launch(
            _write_modal_kernel,
            m * m,
            [
                m,
                lay.Mx,
                lay.e,
                lay.mode_start,
                self.pairs,
                self.integrals,
                inbox,
                L.modal,
                model.elastic_mode_mass,
                model.elastic_mode_stiffness,
                model.elastic_mode_damping,
                model.elastic_mode_coupling_linear,
                model.elastic_mode_coupling_angular,
                model.elastic_mode_coupling_centrifugal,
                model.elastic_mode_coupling_coriolis,
                model.elastic_mode_coupling_quadratic,
                model.elastic_mode_mass_matrix,
            ],
        )
        if self.n_ep:
            self._launch(
                _endpoint_kernel,
                self.n_ep,
                [
                    self._ep_joint,
                    self._ep_side,
                    self._ep_index,
                    st,
                    inbox,
                    L.ep,
                    self.n_ep,
                    m,
                    lay.Mx,
                    model.elastic_endpoint_phi,
                    model.elastic_endpoint_psi,
                    model.joint_X_p,
                    model.joint_X_c,
                    self.theta,
                ],
            )
            self._launch(
                _counter_rotate_kernel,
                1,
                [self.theta, self.n_ep, lay.body, int(counter_rotate), model.body_q, self.diag, 9],
            )
        for csr, local, rows in (
            (self._render, model.elastic_render_point_local, model.elastic_render_point_phi),
            (self._shape, model.elastic_shape_vertex_local, model.elastic_shape_vertex_phi),
        ):
            if csr is not None:
                n_rows, rws, ptr, nodes, wts = csr
                self._launch(_sample_shift_kernel, n_rows, [rws, st, m, lay.Mx, rows, local])
                self._launch(_sample_rows_kernel, n_rows * m, [rws, ptr, nodes, wts, self.phi, m, lay.Mx, rows])
        self._launch(
            _rigid_write_kernel,
            1,
            [
                inbox,
                L.rigid,
                lay.body,
                int(lay.frame_dynamic),
                model.body_mass,
                model.body_com,
                model.body_inertia,
                model.body_inv_mass,
                model.body_inv_inertia,
                self.com,
            ],
        )

        # state, reference, pre-stress
        self._launch(
            _write_state_kernel,
            1,
            [
                mode,
                m,
                lay.q_start,
                lay.qd_start,
                lay.body,
                self.coeff,
                self.qd_new,
                inbox,
                L.given,
                state.joint_q,
                state.joint_qd,
                int(state.body_qd is not None),
                state.body_qd
                if state.body_qd is not None
                else wp.empty(0, dtype=wp.spatial_vectorf, device=self.device),
                self.qd_hist,
            ],
        )
        wp.copy(self.r_ref, self.x)
        self._launch(_copy_f64_kernel, m, [inbox, L.modal + 3 * m, self.prestress])
        self.prestress_host = self._inbox_host[L.modal + 3 * m : L.modal + 4 * m].copy()
        if control is not None:
            self.write_prestress(control)
        if not diagnostics:
            return None
        # after-update momenta of the new state (q = 0, x = r_new)
        self._recover(state, True)
        self._moments(self.diag, 0)
        return self.diag.numpy().copy()

    # ------------------------------------------------------------------------------------------
    # per step
    # ------------------------------------------------------------------------------------------
    def write_prestress(self, control) -> None:
        """Pre-stress kernel into the modal rows of ``control.joint_f`` (graph-capturable)."""
        self._launch(_prestress_kernel, self.m, [self.prestress, self.lay.qd_start + 6, control.joint_f])

    def poll(self, state, dt: float | None) -> None:
        """Velocity history for the modal acceleration estimate (graph-capturable)."""
        if dt is None or dt <= 0.0:
            return
        self._launch(
            _velocity_history_kernel,
            self.m,
            [state.joint_qd, self.lay.qd_start, 1.0 / float(dt), self.qd_hist, self.qdd],
        )

    def rotation_check(
        self,
        state,
        element_ids: np.ndarray | None,
        theta_tol: float | None,
        disp_tol: float | None,
        length: float,
    ) -> tuple[float, float, float]:
        """Device rotation / displacement criterion; one 3-value device-to-host read. Returns ``(code, theta, disp)``."""
        dev = self.device
        asm = self.device_assembler
        has_theta = theta_tol is not None and asm is not None
        self._recover(state, False, x_out=self.x_trig)
        n_ang = 0
        if has_theta:
            if self._trigger_ids is None:
                ids = np.arange(asm.n_tets) if element_ids is None else np.asarray(element_ids).reshape(-1)
                self._trigger_ids = wp.array(ids.astype(np.int32), dtype=wp.int32, device=dev)
                self._trigger_angles = wp.zeros(int(ids.size), dtype=wp.float64, device=dev)
            n_ang = int(self._trigger_ids.shape[0])
            asm.element_rotations(
                self.x_trig, self._trigger_ids, reference_flat=self.r_ref, device_output=True, out=self._trigger_angles
            )
            self._launch(_max_partial_kernel, CHUNKS, [self._trigger_angles, n_ang, CHUNKS, self._trig_partial])
        if disp_tol is not None:
            self._launch(_node_norm_kernel, self.N, [self.x_trig, self.r_ref, self._node_norm])
            self._launch(_max_partial_kernel, CHUNKS, [self._node_norm, self.N, CHUNKS, self._disp_partial])
        self._launch(
            _trigger_flag_kernel,
            1,
            [
                self._trig_partial,
                CHUNKS,
                int(has_theta and n_ang > 0),
                wp.float64(theta_tol if theta_tol is not None else 0.0),
                self._disp_partial,
                CHUNKS,
                int(disp_tol is not None),
                wp.float64(disp_tol if disp_tol is not None else 0.0),
                wp.float64(1.0 / length),
                self.flag,
            ],
        )
        f = self.flag.numpy()
        return float(f[0]), float(f[1]), float(f[2])

    def full_field_device(self, state) -> wp.array:
        """``r_ref + Phi q`` on the device (internal buffer)."""
        self._recover(state, False, x_out=self.x_trig)
        return self.x_trig
