# ruff: noqa: TID253, B007
"""Full nonlinear FE references for validating re-ROM runs (host, numpy/scipy).

Contents
--------
PointLoad                   dead load on a node set
best_fit_rotation           Kabsch rotation between two point clouds (tip-section rotation)
FullFEReference             implicit Newmark (average acceleration, beta=1/4, gamma=1/2) dynamics with full
                            Newton iterations and SciPy sparse LU solves; gravity, point loads, clamped node sets,
                            optional Rayleigh damping; static load-stepping solver
BaseMotion                  analytic rigid motion of a base frame: x_c(t) = d(t) + R(t) X_c about a pivot, with
                            exact velocities and accelerations (single fixed rotation axis, C2 envelope)
PrescribedMotionReference   :class:`FullFEReference` with time-varying Dirichlet data on a clamped node set (rigid
                            base motion), clamp reaction forces, their work, energy and momentum balances

Moving-base kinematics (world frame, rest/mesh coordinates X = base-frame coordinates):
    theta(t) = theta_amp * env(t) * sin(w t + phi_r),           R(t) = exp(theta(t) [a]x)   (fixed unit axis a)
    d(t)     = env(t) * (A_y sin(w t + phi_y) e_y + A_z sin(w t + phi_z) e_z)
    x(X, t)  = d(t) + c + R(t) (X - c)                           (c = pivot, the clamp centre)
    v        = d' + omega x (x - d - c),   omega = theta' a
    a        = d'' + alpha x r + omega x (omega x r),  alpha = theta'' a,  r = R (X - c)
env(t) is a quintic smoothstep envelope (C2: zero velocity and acceleration at the ramp ends), so a body at rest
at t = 0 is consistent with the base data.

Conventions as in :mod:`rerom.assembly`: ``x_flat[3 n + i]``, ``f_int = dW/dx``, ``M a + f_int = f_ext``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from .assembly import NumpyTetAssembler

__all__ = [
    "BaseMotion",
    "FullFEReference",
    "PointLoad",
    "PrescribedMotionReference",
    "best_fit_rotation",
    "envelope",
]

# ----------------------------------------------------------------------------------------------
# Full nonlinear reference solver
# ----------------------------------------------------------------------------------------------


@dataclass
class PointLoad:
    """Dead load on a node set.

    force: (3,) array, or callable t -> (3,) array, or callable t -> (len(node_ids),3) array.
    mode:  "total"    the (3,) force is split equally among node_ids
           "per_node" the (3,) force is applied to every node (or per-node array as returned)
    """

    node_ids: np.ndarray
    force: object
    mode: str = "total"

    def evaluate(self, t: float, n_dof: int) -> np.ndarray:
        ids = np.asarray(self.node_ids, dtype=np.int64)
        fv = self.force(t) if callable(self.force) else self.force
        fv = np.asarray(fv, dtype=np.float64)
        out = np.zeros((n_dof // 3, 3))
        if fv.ndim == 1:
            per = fv / ids.size if self.mode == "total" else fv
            np.add.at(out, ids, np.broadcast_to(per, (ids.size, 3)))
        else:
            np.add.at(out, ids, fv)
        return out.reshape(-1)


def best_fit_rotation(X_ref: np.ndarray, x_cur: np.ndarray) -> tuple[np.ndarray, float]:
    """Kabsch: rotation R minimising sum |(x - xc) - R (X - Xc)|^2; returns (R, angle [rad])."""
    A = np.asarray(X_ref, float) - np.mean(X_ref, axis=0)
    B = np.asarray(x_cur, float) - np.mean(x_cur, axis=0)
    U, _, Vt = np.linalg.svd(B.T @ A)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    ang = float(np.arccos(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)))
    return R, ang


class FullFEReference:
    """Implicit nonlinear FE dynamics: Newmark average acceleration + full Newton.

    Time integration (beta = 1/4, gamma = 1/2 by default; trapezoidal rule):
        a_{n+1} = (x_{n+1} - x_n - dt v_n)/(beta dt^2) - (1/(2 beta) - 1) a_n
        v_{n+1} = v_n + dt ((1-gamma) a_n + gamma a_{n+1})
        r(x_{n+1}) = M a_{n+1} + C v_{n+1} + f_int(x_{n+1}) - f_ext(t_{n+1}) = 0
        J = M/(beta dt^2) + C gamma/(beta dt) + K_T(x_{n+1})
    Newton on the free DOFs with scipy.sparse.linalg.splu each iteration (full Newton).
    Newton predictor: constant velocity x^0 = x_n + dt v_n (default).  The textbook
    constant-acceleration predictor x_n + dt v_n + dt^2/2 a_n amplifies unresolved
    high-frequency content of a_n by ~(omega dt)^2/2 and made Newton diverge on bent beams.
    Converged when ||r_free|| <= rtol * max(||f_ext||, ||f_int||, ||M a||, f_floor) or the
    Newton increment ||dx|| <= xtol * char_length.  Clamped nodes are fixed at their x0
    positions (homogeneous velocity Dirichlet).  C = alpha M + beta_K K_0 (Rayleigh with the
    constant rest stiffness).  Gravity enters as f_g = M g (consistent mass).

    Average acceleration is unconditionally stable and non-dissipative for linear systems and
    conserves linear momentum exactly (f_int is translation-invariant); for nonlinear elasticity
    energy is conserved only approximately (O(dt^2) oscillation, no secular drift in practice).
    """

    def __init__(
        self,
        assembler: NumpyTetAssembler,
        *,
        gravity=(0.0, 0.0, -9.81),
        clamped_nodes: Sequence[int] | None = None,
        rayleigh_alpha: float = 0.0,
        rayleigh_beta: float = 0.0,
        newmark_beta: float = 0.25,
        newmark_gamma: float = 0.5,
        rtol: float = 1e-9,
        xtol: float = 1e-11,
        max_newton: int = 30,
        predictor: str = "constant_velocity",
        verbose: bool = False,
    ):
        if predictor not in ("constant_velocity", "constant_acceleration", "constant_position"):
            raise ValueError(predictor)
        self.predictor = predictor
        self.asm = assembler
        self.M = assembler.mass_matrix.tocsr()
        n = assembler.n_dof
        self.n_dof = n
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.f_grav = self.M @ np.tile(self.gravity, n // 3)
        self.clamped = (
            np.zeros(0, np.int64) if clamped_nodes is None else np.unique(np.asarray(clamped_nodes, np.int64))
        )
        fixed = np.zeros(n, dtype=bool)
        fixed[(3 * self.clamped[:, None] + np.arange(3)).ravel()] = True
        self.free = np.nonzero(~fixed)[0]
        self.fixed_mask = fixed
        self.alpha, self.beta_k = float(rayleigh_alpha), float(rayleigh_beta)
        self.C = None
        if self.alpha != 0.0 or self.beta_k != 0.0:
            self.C = (self.alpha * self.M + self.beta_k * assembler.linear_stiffness()).tocsr()
        self.nb, self.ng = float(newmark_beta), float(newmark_gamma)
        self.rtol, self.xtol, self.max_newton = rtol, xtol, max_newton
        self.verbose = verbose
        ext = assembler.mesh.nodes.max(axis=0) - assembler.mesh.nodes.min(axis=0)
        self.char_length = float(np.linalg.norm(ext))
        self.last_result = None

    # -- helpers ---------------------------------------------------------------------------
    def external_force(self, t: float, loads=None, gravity_scale: float = 1.0) -> np.ndarray:
        f = gravity_scale * self.f_grav.copy()
        for ld in loads or ():
            if isinstance(ld, PointLoad):
                f += ld.evaluate(t, self.n_dof)
            else:  # (node_ids, force[, mode])
                f += PointLoad(*ld).evaluate(t, self.n_dof)
        return f

    def gravity_potential(self, x_flat: np.ndarray) -> float:
        return float(-np.dot(self.f_grav, x_flat - self.asm.X_flat))

    def _ff(self, A):
        return A[self.free][:, self.free]

    # -- dynamics --------------------------------------------------------------------------
    def run(
        self,
        x0: np.ndarray | None = None,
        v0: np.ndarray | None = None,
        dt: float = 1e-3,
        steps: int = 100,
        loads=None,
        record_every: int = 1,
        record_velocity: bool = False,
        tip_nodes: Sequence[int] | None = None,
        callback: Callable | None = None,
    ) -> dict:
        """Integrate `steps` steps of size dt from (x0, v0) (flat (3N,) or (N,3); defaults rest/zero).

        Returns dict with: times (T,), x_hist (T,N,3), v_hist (T,N,3) if record_velocity,
        kinetic, strain, gravity_potential, external_work (point loads, trapezoidal),
        total_energy = kinetic + strain + gravity_potential - external_work (all (T,)),
        energy (dict of the same), linear_momentum (T,3), angular_momentum (T,3) about origin,
        newton_iters (steps,), step_wall (steps,) seconds, tip_trajectory (T,3) when tip_nodes.
        callback(step, t, x, v) is called after each step if given.
        """
        asm, M, C = self.asm, self.M, self.C
        n = self.n_dof
        x = asm.X_flat.copy() if x0 is None else np.asarray(x0, float).reshape(-1).copy()
        v = np.zeros(n) if v0 is None else np.asarray(v0, float).reshape(-1).copy()
        v[self.fixed_mask] = 0.0
        free = self.free
        b, g = self.nb, self.ng
        c0, c1 = 1.0 / (b * dt * dt), g / (b * dt)
        Mff = self._ff(M).tocsc()
        Mc = M * c0 if C is None else (M * c0 + C * c1)

        t = 0.0
        fext = self.external_force(t, loads)
        fpoint = fext - self.f_grav
        W, fint, _ = asm.evaluate(x, need_tangent=False)
        rhs = fext - fint - (C @ v if C is not None else 0.0)
        a = np.zeros(n)
        a[free] = spla.spsolve(Mff, rhs[free])

        rec = {"times": [], "x": [], "v": [], "ke": [], "w": [], "vg": [], "wext": [], "p": [], "L": []}
        wext = 0.0

        def record(t, x, v, W):
            xv = x.reshape(-1, 3)
            mv = (M @ v).reshape(-1, 3)
            rec["times"].append(t)
            rec["x"].append(xv.copy())
            if record_velocity:
                rec["v"].append(v.reshape(-1, 3).copy())
            rec["ke"].append(0.5 * float(v @ (M @ v)))
            rec["w"].append(W)
            rec["vg"].append(self.gravity_potential(x))
            rec["wext"].append(wext)
            rec["p"].append(mv.sum(axis=0))
            rec["L"].append(np.cross(xv, mv).sum(axis=0))

        record(t, x, v, W)
        iters = np.zeros(steps, dtype=np.int64)
        walls = np.zeros(steps)
        for step in range(steps):
            t0 = time.perf_counter()
            t_new = (step + 1) * dt
            fext_new = self.external_force(t_new, loads)
            fpoint_new = fext_new - self.f_grav
            xn, vn, an = x, v, a
            if self.predictor == "constant_acceleration":
                x = xn + dt * vn + 0.5 * dt * dt * an
            elif self.predictor == "constant_velocity":
                x = xn + dt * vn
            else:  # "constant_position"
                x = xn.copy()
            x[self.fixed_mask] = xn[self.fixed_mask]
            converged = False
            x_prev, dx = None, None
            for it in range(1, self.max_newton + 1):
                for _ in range(30):  # neo-Hookean inversion guard: halve the last Newton step
                    try:
                        W, fint, K = asm.evaluate(x, need_tangent=True)
                        break
                    except FloatingPointError:
                        if x_prev is None:
                            raise
                        dx = 0.5 * dx
                        x = x_prev.copy()
                        x[free] += dx
                a = c0 * (x - xn - dt * vn) - (0.5 / b - 1.0) * an
                v = vn + dt * ((1.0 - g) * an + g * a)
                Ma = M @ a
                r = Ma + fint - fext_new
                if C is not None:
                    r += C @ v
                rf = r[free]
                scale = max(np.linalg.norm(fext_new[free]), np.linalg.norm(fint[free]), np.linalg.norm(Ma[free]), 1e-12)
                rn = np.linalg.norm(rf)
                if self.verbose > 1:
                    print(f"  newton {it}: |r|={rn:.3e} scale={scale:.3e}")
                if rn <= self.rtol * scale:
                    converged = True
                    break
                Jff = self._ff((Mc + K).tocsr()).tocsc()
                dx = spla.splu(Jff).solve(-rf)
                x_prev = x
                x = x.copy()
                x[free] += dx
                if np.linalg.norm(dx) <= self.xtol * self.char_length:
                    a = c0 * (x - xn - dt * vn) - (0.5 / b - 1.0) * an
                    v = vn + dt * ((1.0 - g) * an + g * a)
                    W, fint, _ = asm.evaluate(x, need_tangent=False)
                    converged = True
                    break
            if not converged:
                raise RuntimeError(f"Newton did not converge at step {step} (|r|={rn:.3e}, scale={scale:.3e})")
            iters[step] = it
            wext += 0.5 * float(np.dot(fpoint + fpoint_new, x - xn))
            fpoint = fpoint_new
            walls[step] = time.perf_counter() - t0
            if self.verbose:
                print(f"step {step + 1}/{steps} t={t_new:.4f} newton={it} wall={walls[step]:.3f}s")
            if callback is not None:
                callback(step, t_new, x, v)
            if (step + 1) % record_every == 0 or step + 1 == steps:
                record(t_new, x, v, W)

        res = {
            "times": np.asarray(rec["times"]),
            "x_hist": np.asarray(rec["x"]),
            "kinetic": np.asarray(rec["ke"]),
            "strain": np.asarray(rec["w"]),
            "gravity_potential": np.asarray(rec["vg"]),
            "external_work": np.asarray(rec["wext"]),
            "linear_momentum": np.asarray(rec["p"]),
            "angular_momentum": np.asarray(rec["L"]),
            "newton_iters": iters,
            "step_wall": walls,
            "dt": dt,
            "final_x": x.copy(),
            "final_v": v.copy(),
            "final_a": a.copy(),
        }
        if record_velocity:
            res["v_hist"] = np.asarray(rec["v"])
        res["total_energy"] = res["kinetic"] + res["strain"] + res["gravity_potential"] - res["external_work"]
        res["energy"] = {k: res[k] for k in ("kinetic", "strain", "gravity_potential", "external_work", "total_energy")}
        if tip_nodes is not None:
            res["tip_trajectory"] = res["x_hist"][:, np.asarray(tip_nodes)].mean(axis=1)
        self.last_result = res
        return res

    def tip_trajectory(self, node_ids: Sequence[int], result: dict | None = None) -> np.ndarray:
        """(T,3) mean position of node_ids over the recorded history (default: last run)."""
        res = self.last_result if result is None else result
        return res["x_hist"][:, np.asarray(node_ids)].mean(axis=1)

    # -- statics ---------------------------------------------------------------------------
    def solve_static(
        self,
        loads=None,
        x0: np.ndarray | None = None,
        n_increments: int = 10,
        gravity_scale: float = 1.0,
        t: float = 0.0,
        max_newton: int = 50,
    ) -> dict:
        """Static equilibrium f_int(x) = s f_ext by load stepping s = k/n_increments with Newton
        (backtracking on the residual norm).  Requires clamped nodes.  Returns dict with x
        (3N,), history of x per increment, newton iterations per increment."""
        if self.clamped.size == 0:
            raise ValueError("static solve needs clamped nodes")
        asm, free = self.asm, self.free
        x = asm.X_flat.copy() if x0 is None else np.asarray(x0, float).reshape(-1).copy()
        f_full = self.external_force(t, loads, gravity_scale)
        hist, its = [x.copy()], []
        for k in range(1, n_increments + 1):
            fext = f_full * (k / n_increments)
            scale = max(np.linalg.norm(fext[free]), 1e-12)
            for it in range(1, max_newton + 1):
                W, fint, K = asm.evaluate(x)
                r = (fint - fext)[free]
                rn = np.linalg.norm(r)
                if rn <= self.rtol * scale:
                    break
                dx = spla.splu(self._ff(K).tocsc()).solve(-r)
                # backtracking (Armijo) on the total potential Pi = W - fext.x along dx; the full
                # Newton step is accepted whenever it lowers Pi (a residual increase is allowed).
                pi0 = W - float(fext @ x)
                slope = float(r @ dx)
                step = 1.0
                for _ in range(20):
                    xt = x.copy()
                    xt[free] += step * dx
                    try:
                        Wt = asm.evaluate(xt, need_tangent=False)[0]
                        if slope >= 0.0 or Wt - float(fext @ xt) <= pi0 + 1e-4 * step * slope:
                            break
                    except FloatingPointError:
                        pass
                    step *= 0.5
                x = xt
                if np.linalg.norm(step * dx) <= self.xtol * self.char_length:
                    break
            else:
                raise RuntimeError(f"static Newton did not converge at increment {k} (|r|={rn:.3e})")
            its.append(it)
            hist.append(x.copy())
        return {"x": x, "x_increments": np.asarray(hist), "newton_iters": np.asarray(its)}


# ----------------------------------------------------------------------------------------------
# Prescribed rigid base motion
# ----------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------
# envelope with derivatives
# ----------------------------------------------------------------------------------------------
def _smooth(x):
    """Quintic smoothstep s(x) and its first two derivatives w.r.t. x (clamped to [0, 1])."""
    if x <= 0.0:
        return 0.0, 0.0, 0.0
    if x >= 1.0:
        return 1.0, 0.0, 0.0
    s = x * x * x * (x * (6 * x - 15) + 10)
    ds = 30 * x * x * (x - 1) ** 2
    dds = 60 * x * (2 * x * x - 3 * x + 1)
    return s, ds, dds


def envelope(t: float, t_on: float, t_full: float, t_off: float, t_zero: float):
    """env, env', env'' : 0 before t_on, C2 ramp to 1 at t_full, 1 until t_off, C2 ramp to 0 at t_zero."""
    a, da, dda = _smooth((t - t_on) / (t_full - t_on))
    k1 = 1.0 / (t_full - t_on)
    da, dda = da * k1, dda * k1 * k1
    b, db, ddb = _smooth((t - t_off) / (t_zero - t_off))
    k2 = 1.0 / (t_zero - t_off)
    db, ddb = db * k2, ddb * k2 * k2
    c, dc, ddc = 1.0 - b, -db, -ddb
    return a * c, da * c + a * dc, dda * c + 2 * da * dc + a * ddc


def _sin_env(t, amp, w, phase, env):
    e, de, dde = env
    s, c = math.sin(w * t + phase), math.cos(w * t + phase)
    f = amp * e * s
    df = amp * (de * s + e * w * c)
    ddf = amp * (dde * s + 2 * de * w * c - e * w * w * s)
    return f, df, ddf


def quat_xyzw_axis_angle(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = math.sin(0.5 * ang)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * ang)])


def _rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


# ----------------------------------------------------------------------------------------------
# base motion
# ----------------------------------------------------------------------------------------------
@dataclass
class BaseMotion:
    """Analytic rigid base motion (see module docstring). Angles in radians, lengths in metres."""

    freq_hz: float = 2.0
    rot_amp: float = math.radians(30.0)
    axis: tuple = (0.0, 0.5, 1.0)
    rot_phase: float = 0.0
    trans_amp_y: float = 0.25
    trans_amp_z: float = 0.10
    trans_phase_y: float = 0.0
    trans_phase_z: float = 0.5 * math.pi
    env_times: tuple = (0.2, 0.7, 2.5, 3.0)  # t_on, t_full, t_off, t_zero
    pivot: tuple = (0.0, 0.0, 0.0)
    scale: float = 1.0  # global scale of rotation and translation (0 -> base at rest)

    def __post_init__(self):
        a = np.asarray(self.axis, float)
        self.a = a / np.linalg.norm(a)
        self.c = np.asarray(self.pivot, float)
        self.w = 2.0 * math.pi * self.freq_hz

    def describe(self) -> dict:
        return {
            "freq_hz": self.freq_hz,
            "rot_amp_deg": math.degrees(self.rot_amp),
            "axis": self.a.tolist(),
            "rot_phase": self.rot_phase,
            "trans_amp_y_m": self.trans_amp_y,
            "trans_amp_z_m": self.trans_amp_z,
            "trans_phase_y": self.trans_phase_y,
            "trans_phase_z": self.trans_phase_z,
            "env_times": list(self.env_times),
            "pivot": self.c.tolist(),
            "scale": self.scale,
            "formula": "x = d(t) + c + R(theta(t) a) (X - c); theta = th_amp env sin(w t + phi_r); "
            "d = env (A_y sin(w t + phi_y) e_y + A_z sin(w t + phi_z) e_z); env = C2 quintic ramps",
        }

    def _scalars(self, t):
        env = envelope(t, *self.env_times)
        th = _sin_env(t, self.scale * self.rot_amp, self.w, self.rot_phase, env)
        dy = _sin_env(t, self.scale * self.trans_amp_y, self.w, self.trans_phase_y, env)
        dz = _sin_env(t, self.scale * self.trans_amp_z, self.w, self.trans_phase_z, env)
        return th, dy, dz

    def state(self, t: float) -> dict:
        """Base frame state at t: R, d, d', d'', theta, omega, alpha (world), pose [p, quat xyzw] of the
        base frame whose origin is the mesh origin (p = d + c - R c)."""
        th, dy, dz = self._scalars(t)
        R = _rot(self.a, th[0])
        d = np.array([0.0, dy[0], dz[0]])
        dd = np.array([0.0, dy[1], dz[1]])
        ddd = np.array([0.0, dy[2], dz[2]])
        p = d + self.c - R @ self.c
        return {
            "t": t,
            "theta": th[0],
            "R": R,
            "d": d,
            "d_dot": dd,
            "d_ddot": ddd,
            "omega": th[1] * self.a,
            "alpha": th[2] * self.a,
            "p": p,
            "quat": quat_xyzw_axis_angle(self.a, th[0]) if self.a.any() else np.array([0, 0, 0, 1.0]),
        }

    def pose(self, t: float) -> np.ndarray:
        s = self.state(t)
        return np.concatenate([s["p"], s["quat"]])

    def poses(self, dt: float, steps: int) -> np.ndarray:
        P = np.stack([self.pose(n * dt) for n in range(steps + 1)])
        P[:, 3:] /= np.linalg.norm(P[:, 3:], axis=1, keepdims=True)
        return P

    def origin_twist(self, t: float):
        """World linear velocity of the base-frame ORIGIN (mesh origin) and world angular velocity."""
        s = self.state(t)
        r0 = s["p"] - (s["d"] + self.c)  # = -R c
        return s["d_dot"] + np.cross(s["omega"], r0), s["omega"]

    def nodal(self, X: np.ndarray, t: float):
        """World positions, velocities and accelerations (n,3) of base-fixed points X (n,3)."""
        s = self.state(t)
        r = (np.asarray(X, float) - self.c) @ s["R"].T
        x = s["d"] + self.c + r
        w, al = s["omega"], s["alpha"]
        v = s["d_dot"] + np.cross(w, r)
        a = s["d_ddot"] + np.cross(al, r) + np.cross(w, np.cross(w, r))
        return x, v, a


# ----------------------------------------------------------------------------------------------
# full nonlinear FE reference with prescribed clamp motion
# ----------------------------------------------------------------------------------------------
class PrescribedMotionReference(FullFEReference):
    """Implicit nonlinear FE (Newmark average acceleration + full Newton) with a moving clamped node set.

    Dirichlet data on the clamped DOFs at every t_{n+1}: positions, velocities and accelerations of the rigid
    base motion (``motion.nodal``) - the exact data, not Newmark-differentiated positions. Free DOFs follow the
    standard Newmark update. The residual is solved on the free rows only; the clamped rows give the clamp
    reaction ``r_c = (M a + f_int - f_ext)_c`` (force the base applies to the body). Balances recorded:

    * energy:   E = KE + W + V_g - W_R,  W_R = sum_n 1/2 (r_n + r_{n+1}) . (x_{n+1} - x_n)_c  (trapezoidal)
    * momentum: P_{n+1} - P_n - dt/2 (F_n + F_{n+1}),  F = sum of gravity + reactions (exact for the free rows;
      O(dt^3) per step from the exact clamped velocities).
    """

    def __init__(
        self,
        assembler: NumpyTetAssembler,
        motion: BaseMotion,
        *,
        clamped_nodes: Sequence[int],
        scheme: str = "newmark",
        **kw,
    ):
        """``scheme``: "newmark" (average acceleration, non-dissipative; default) or "backward_euler"
        (x_{n+1} = x_n + dt v_{n+1}, v_{n+1} = v_n + dt a_{n+1}: the implicit-Euler time discretization that
        Newton's SolverVBD uses, so that the comparison with VBD-based solvers can separate time-integration
        dissipation from model-reduction error)."""
        if scheme not in ("newmark", "backward_euler"):
            raise ValueError(scheme)
        super().__init__(assembler, clamped_nodes=clamped_nodes, **kw)
        self.scheme = scheme
        self.motion = motion
        self.X = assembler.mesh.nodes.astype(np.float64)
        self.Xc = self.X[self.clamped]
        self.fixed_idx = np.nonzero(self.fixed_mask)[0]

    def clamp_data(self, t: float):
        x, v, a = self.motion.nodal(self.Xc, t)
        return x.reshape(-1), v.reshape(-1), a.reshape(-1)

    def rigid_initial_state(self, t: float = 0.0):
        """Whole body moving rigidly with the base at t (positions, velocities, accelerations, flat)."""
        x, v, a = self.motion.nodal(self.X, t)
        return x.reshape(-1), v.reshape(-1), a.reshape(-1)

    def run(
        self,
        x0=None,
        v0=None,
        dt=1e-3,
        steps=100,
        loads=None,
        record_every=1,
        record_velocity=False,
        tip_nodes=None,
        callback: Callable | None = None,
        a0=None,
        t0: float = 0.0,
    ):
        """Integrate from t0 (defaults: body rigidly attached to the base at t0, base velocity everywhere).

        Returns the FullFEReference.run dictionary plus: reaction_force (T,3) resultant clamp force,
        reaction_moment (T,3) about the world origin, reaction_work (T,), energy_balance (T,) =
        kinetic + strain + gravity_potential - reaction_work - external_work, momentum_balance_residual (steps,3),
        newton_converged (steps,) bool.
        """
        asm, M, C = self.asm, self.M, self.C
        if C is not None:
            raise NotImplementedError("damping is not supported with prescribed motion")
        if loads:
            raise NotImplementedError("point loads are not supported with prescribed motion")
        n = self.n_dof
        fx, free = self.fixed_idx, self.free
        xr, vr, _ar = self.rigid_initial_state(t0)
        x = xr.copy() if x0 is None else np.asarray(x0, float).reshape(-1).copy()
        v = vr.copy() if v0 is None else np.asarray(v0, float).reshape(-1).copy()
        xc, vc, ac = self.clamp_data(t0)
        x[fx], v[fx] = xc, vc
        b, g = self.nb, self.ng
        be = self.scheme == "backward_euler"
        c0 = 1.0 / (dt * dt) if be else 1.0 / (b * dt * dt)
        ka = 0.0 if be else (0.5 / b - 1.0)  # a_{n+1} = c0 (x - x_n - dt v_n) - ka a_n
        if be:
            g = 1.0  # v_{n+1} = v_n + dt a_{n+1} = (x_{n+1} - x_n)/dt
        Mff = self._ff(M).tocsc()
        fext = self.f_grav
        W, fint, _ = asm.evaluate(x, need_tangent=False)
        a = np.zeros(n)
        a[fx] = ac
        if a0 is None:
            rhs = fext - fint - M[:, fx] @ ac
            a[free] = spla.spsolve(Mff, rhs[free])
        else:
            a[free] = np.asarray(a0, float).reshape(-1)[free]
        rvec = M @ a + fint - fext  # reaction (nonzero on fixed rows only, up to the initial solve)

        rec = {
            "times": [],
            "x": [],
            "v": [],
            "ke": [],
            "w": [],
            "vg": [],
            "wr": [],
            "p": [],
            "L": [],
            "Fr": [],
            "Mr": [],
        }
        wr = 0.0

        def record(t, x, v, W, r):
            xv = x.reshape(-1, 3)
            mv = (M @ v).reshape(-1, 3)
            rc = np.zeros(n)
            rc[fx] = r[fx]
            rc3 = rc.reshape(-1, 3)
            rec["times"].append(t)
            rec["x"].append(xv.copy())
            if record_velocity:
                rec["v"].append(v.reshape(-1, 3).copy())
            rec["ke"].append(0.5 * float(v @ (M @ v)))
            rec["w"].append(W)
            rec["vg"].append(self.gravity_potential(x))
            rec["wr"].append(wr)
            rec["p"].append(mv.sum(axis=0))
            rec["L"].append(np.cross(xv, mv).sum(axis=0))
            rec["Fr"].append(rc3.sum(axis=0))
            rec["Mr"].append(np.cross(xv, rc3).sum(axis=0))

        def total_force(r):
            return fext.reshape(-1, 3).sum(axis=0) + r[fx].reshape(-1, 3).sum(axis=0)

        t = t0
        record(t, x, v, W, rvec)
        iters = np.zeros(steps, dtype=np.int64)
        walls = np.zeros(steps)
        conv_flags = np.zeros(steps, dtype=bool)
        mom_res = np.zeros((steps, 3))
        F_prev = total_force(rvec)
        for step in range(steps):
            ts = time.perf_counter()
            t_new = t0 + (step + 1) * dt
            xn, vn, an, rn_prev = x, v, a, rvec
            xc, vc, ac = self.clamp_data(t_new)
            if self.predictor == "constant_acceleration":
                x = xn + dt * vn + 0.5 * dt * dt * an
            elif self.predictor == "constant_velocity":
                x = xn + dt * vn
            else:
                x = xn.copy()
            x[fx] = xc
            converged = False
            x_prev, dx = None, None
            rn = np.inf
            scale = 1.0
            for it in range(1, self.max_newton + 1):
                for _ in range(30):
                    try:
                        W, fint, K = asm.evaluate(x, need_tangent=True)
                        break
                    except FloatingPointError:
                        if x_prev is None:
                            raise
                        dx = 0.5 * dx
                        x = x_prev.copy()
                        x[free] += dx
                a = c0 * (x - xn - dt * vn) - ka * an
                a[fx] = ac
                Ma = M @ a
                r = Ma + fint - fext
                rf = r[free]
                scale = max(np.linalg.norm(fext[free]), np.linalg.norm(fint[free]), np.linalg.norm(Ma[free]), 1e-12)
                rn = np.linalg.norm(rf)
                if rn <= self.rtol * scale:
                    converged = True
                    break
                Jff = (self._ff(K) + c0 * self._ff(M)).tocsc()
                dx = spla.splu(Jff).solve(-rf)
                x_prev = x
                x = x.copy()
                x[free] += dx
                if np.linalg.norm(dx) <= self.xtol * self.char_length:
                    W, fint, _ = asm.evaluate(x, need_tangent=False)
                    a = c0 * (x - xn - dt * vn) - ka * an
                    a[fx] = ac
                    r = M @ a + fint - fext
                    converged = True
                    break
            if not converged:
                raise RuntimeError(f"Newton did not converge at step {step} (|r|={rn:.3e}, scale={scale:.3e})")
            v = vn + dt * ((1.0 - g) * an + g * a)
            v[fx] = vc
            rvec = r
            conv_flags[step] = converged
            iters[step] = it
            wr += 0.5 * float(np.dot(rn_prev[fx] + rvec[fx], x[fx] - xn[fx]))
            F_new = total_force(rvec)
            P_old = (M @ vn).reshape(-1, 3).sum(axis=0)
            P_new = (M @ v).reshape(-1, 3).sum(axis=0)
            mom_res[step] = P_new - P_old - (dt * F_new if be else 0.5 * dt * (F_prev + F_new))
            F_prev = F_new
            walls[step] = time.perf_counter() - ts
            if self.verbose:
                print(f"step {step + 1}/{steps} t={t_new:.4f} newton={it} wall={walls[step]:.3f}s")
            if callback is not None:
                callback(step, t_new, x, v)
            if (step + 1) % record_every == 0 or step + 1 == steps:
                record(t_new, x, v, W, rvec)
                step + 1

        res = {
            "times": np.asarray(rec["times"]),
            "x_hist": np.asarray(rec["x"]),
            "kinetic": np.asarray(rec["ke"]),
            "strain": np.asarray(rec["w"]),
            "gravity_potential": np.asarray(rec["vg"]),
            "reaction_work": np.asarray(rec["wr"]),
            "external_work": np.zeros(len(rec["times"])),
            "linear_momentum": np.asarray(rec["p"]),
            "angular_momentum": np.asarray(rec["L"]),
            "reaction_force": np.asarray(rec["Fr"]),
            "reaction_moment": np.asarray(rec["Mr"]),
            "newton_iters": iters,
            "newton_converged": conv_flags,
            "step_wall": walls,
            "dt": dt,
            "final_x": x.copy(),
            "final_v": v.copy(),
            "final_a": a.copy(),
            "momentum_balance_residual": mom_res,
        }
        if record_velocity:
            res["v_hist"] = np.asarray(rec["v"])
        res["total_energy"] = res["kinetic"] + res["strain"] + res["gravity_potential"]
        res["energy_balance"] = res["total_energy"] - res["reaction_work"]
        res["energy"] = {
            k: res[k]
            for k in ("kinetic", "strain", "gravity_potential", "reaction_work", "total_energy", "energy_balance")
        }
        if tip_nodes is not None:
            res["tip_trajectory"] = res["x_hist"][:, np.asarray(tip_nodes)].mean(axis=1)
        self.last_result = res
        return res
