"""All ``newton.Model`` / ``State`` / ``Control`` array reads and writes of the re-ROM controller.

Nothing else in the package touches Model arrays. Reads and writes are batched:

- :class:`BodyLayout` reads the static bookkeeping of one reduced elastic body once (indices, endpoint,
  render-point and shape-vertex ranges).
- :class:`ModelArrays` is the per-update cache: each Model array is gathered at most once (first use) and
  scattered at most once (:meth:`ModelArrays.flush`) per update, whatever the number of edits.
- :func:`read_joint_state` / :func:`write_body_state` gather and scatter ``joint_q``, ``joint_qd`` (and
  ``body_qd``) once; :func:`write_modal_joint_f` writes the modal rows of ``control.joint_f``.
- :class:`JointRestPropagator` re-propagates rest rotations of ``model.body_q`` over the joint graph.

Conventions (Newton reduced elastic bodies): ``joint_q = [p(3), quat xyzw(4), q(m)]``,
``joint_qd = [v_c(3) world velocity of the rest COM, omega(3) world, qd(m)]``; per-mode sample rows are
``item * elastic_max_mode_count + mode``; pair blocks ``e Mx^2 + i Mx + j``.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import newton

from .numerics import quat_angle, quat_conj, quat_from_rotvec, quat_mul

__all__ = [
    "BodyLayout",
    "EndpointRef",
    "JointRestPropagator",
    "ModelArrays",
    "read_joint_state",
    "write_body_state",
    "write_modal_joint_f",
]


@dataclass
class EndpointRef:
    """A joint endpoint on the elastic body."""

    joint: int
    side: int  # 0 parent (joint_X_p), 1 child (joint_X_c)
    endpoint: int
    sample: int


@dataclass
class BodyLayout:
    """Static Newton bookkeeping of one reduced elastic body (read once)."""

    body: int
    e: int
    joint: int
    m: int
    q_start: int
    qd_start: int
    mode_start: int
    Mx: int
    world: int
    basis_index: int
    frame_dynamic: bool
    endpoints: list[EndpointRef] = field(default_factory=list)
    render_range: tuple[int, int] = (0, 0)
    render_samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    shape_ranges: list[tuple[int, int]] = field(default_factory=list)
    shape_samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @classmethod
    def from_model(cls, model, body: int, m: int) -> BodyLayout:
        """Read and validate the layout of ``body`` (owner joint ELASTIC with ``7 + m`` / ``6 + m`` coordinates)."""
        body = int(body)
        e = int(model.body_elastic_index.numpy()[body])
        if e < 0:
            raise ValueError(f"body {body} is not a reduced elastic body")
        joint = int(model.elastic_joint.numpy()[e])
        jt = int(model.joint_type.numpy()[joint])
        if jt != int(newton.JointType.ELASTIC):
            raise ValueError(f"owner joint {joint} has type {jt}, expected ELASTIC")
        q_starts = model.joint_q_start.numpy()
        qd_starts = model.joint_qd_start.numpy()
        n_mode_model = int(model.elastic_mode_count.numpy()[e])
        if n_mode_model != m:
            raise ValueError(f"model mode count {n_mode_model} != controller mode count {m}")
        if int(q_starts[joint + 1] - q_starts[joint]) != 7 + m:
            raise ValueError("elastic joint coordinate layout is not [7 frame + m modes]")
        if int(qd_starts[joint + 1] - qd_starts[joint]) != 6 + m:
            raise ValueError("elastic joint velocity layout is not [6 frame + m modes]")
        basis_index = int(model.elastic_basis.numpy()[e])
        if basis_index < 0:
            raise ValueError("the elastic body has no ModalBasis (mode_shape_fn bodies are unsupported)")
        lay = cls(
            body=body,
            e=e,
            joint=joint,
            m=int(m),
            q_start=int(q_starts[joint]),
            qd_start=int(qd_starts[joint]),
            mode_start=int(model.elastic_mode_start.numpy()[e]),
            Mx=int(model.elastic_max_mode_count),
            world=int(model.body_world.numpy()[body]) if model.body_world is not None else 0,
            basis_index=basis_index,
            frame_dynamic=float(model.body_inv_mass.numpy()[body]) > 0.0,
        )
        jp = model.joint_parent.numpy()
        jc = model.joint_child.numpy()
        pe = model.joint_parent_elastic_endpoint.numpy()
        ce = model.joint_child_elastic_endpoint.numpy()
        ep_sample = model.elastic_endpoint_sample.numpy() if model.elastic_endpoint_count > 0 else np.zeros(0, int)
        for j in range(model.joint_count):
            if j == joint:
                continue
            if jp[j] == body and pe[j] >= 0:
                lay.endpoints.append(EndpointRef(j, 0, int(pe[j]), int(ep_sample[pe[j]])))
            if jc[j] == body and ce[j] >= 0:
                lay.endpoints.append(EndpointRef(j, 1, int(ce[j]), int(ep_sample[ce[j]])))
        rp_start = model.elastic_render_point_start.numpy()
        rp_count = model.elastic_render_point_count.numpy()
        rp_sample = model.elastic_render_point_sample.numpy()
        lay.render_range = (int(rp_start[e]), int(rp_start[e] + rp_count[e]))
        lay.render_samples = rp_sample[lay.render_range[0] : lay.render_range[1]].astype(np.int64)
        if model.elastic_shape_count > 0:
            sb = model.elastic_shape_body.numpy()
            vs = model.elastic_shape_vertex_start.numpy()
            vc = model.elastic_shape_vertex_count.numpy()
            vsample = model.elastic_shape_vertex_sample.numpy()
            parts = []
            for i in range(model.elastic_shape_count):
                if int(sb[i]) == body:
                    lay.shape_ranges.append((int(vs[i]), int(vs[i] + vc[i])))
                    parts.append(vsample[vs[i] : vs[i] + vc[i]])
            if parts:
                lay.shape_samples = np.concatenate(parts).astype(np.int64)
        return lay

    @property
    def mode_slice(self) -> slice:
        """Slice of the body's modes in the per-mode Model arrays."""
        return slice(self.mode_start, self.mode_start + self.m)

    def pair_index(self) -> np.ndarray:
        """Flat indices of the body's ``m x m`` pair block."""
        base = self.e * self.Mx * self.Mx
        return (base + np.arange(self.m)[:, None] * self.Mx + np.arange(self.m)[None, :]).reshape(-1)

    def sample_rows(self, items) -> np.ndarray:
        """Row indices ``(P, m)`` of per-mode sample arrays for ``items``."""
        return np.asarray(items, dtype=np.int64)[:, None] * self.Mx + np.arange(self.m)[None, :]


class ModelArrays:
    """Per-update gather/scatter cache of Model arrays (one ``.numpy()`` and one ``.assign`` per array)."""

    def __init__(self, model):
        self.model = model
        self._data: dict[str, np.ndarray] = {}
        self._dirty: set[str] = set()

    def get(self, name: str) -> np.ndarray:
        """Host copy of ``model.<name>`` (gathered on first use)."""
        a = self._data.get(name)
        if a is None:
            a = getattr(self.model, name).numpy()
            self._data[name] = a
        return a

    def edit(self, name: str) -> np.ndarray:
        """Like :meth:`get`, and mark the array for :meth:`flush`."""
        a = self.get(name)
        self._dirty.add(name)
        return a

    def put(self, name: str, idx, values) -> None:
        """``model.<name>[idx] = values`` (deferred to :meth:`flush`)."""
        self.edit(name)[idx] = values

    def flush(self) -> None:
        """Scatter every edited array once and drop the cache."""
        for name in self._dirty:
            getattr(self.model, name).assign(self._data[name])
        self._data.clear()
        self._dirty.clear()

    def discard(self) -> None:
        """Drop the cache without writing."""
        self._data.clear()
        self._dirty.clear()


# ----------------------------------------------------------------------------------------------
# state / control
# ----------------------------------------------------------------------------------------------


def read_joint_state(state) -> tuple[np.ndarray, np.ndarray]:
    """One gather of ``(joint_q, joint_qd)`` (host copies)."""
    return state.joint_q.numpy(), state.joint_qd.numpy()


def write_body_state(state, lay: BodyLayout, *, q=None, v_c=None, omega=None, qd=None, jq=None, jqd=None) -> None:
    """Write the body's modal coordinates, frame twist and modal velocities (one scatter per array).

    ``jq``/``jqd``: host copies to edit (gathered here when ``None``); ``body_qd`` mirrors the frame twist.
    """
    if jq is None or jqd is None:
        jq, jqd = read_joint_state(state)
    m = lay.m
    if q is not None:
        jq[lay.q_start + 7 : lay.q_start + 7 + m] = q
    if v_c is not None:
        jqd[lay.qd_start : lay.qd_start + 3] = v_c
    if omega is not None:
        jqd[lay.qd_start + 3 : lay.qd_start + 6] = omega
    if qd is not None:
        jqd[lay.qd_start + 6 : lay.qd_start + 6 + m] = qd
    state.joint_q.assign(jq)
    state.joint_qd.assign(jqd)
    if state.body_qd is not None and (v_c is not None or omega is not None):
        bqd = state.body_qd.numpy()
        if v_c is not None:
            bqd[lay.body, :3] = v_c
        if omega is not None:
            bqd[lay.body, 3:] = omega
        state.body_qd.assign(bqd)


def write_modal_joint_f(control, rows: list[tuple[BodyLayout, np.ndarray]]) -> None:
    """Write ``control.joint_f[modal rows] = f`` for several bodies with one gather and one scatter."""
    jf = control.joint_f.numpy()
    for lay, f in rows:
        s = lay.qd_start + 6
        jf[s : s + lay.m] = np.asarray(f).astype(np.float32)
    control.joint_f.assign(jf)


# ----------------------------------------------------------------------------------------------
# endpoint frames and rest rotations
# ----------------------------------------------------------------------------------------------


def move_endpoint_frames(arrays: ModelArrays, lay: BodyLayout, q: np.ndarray, ep_phi, ep_psi) -> list[np.ndarray]:
    """Move the body's joint endpoint frames to their deformed pose at ``q`` and install new samples.

    ``pos += phi_old^T q``, ``rot = quat(psi_old^T q) rot`` (Newton's linear endpoint update), then the
    endpoint ``phi``/``psi`` rows are replaced by ``ep_phi[i]``/``ep_psi[i]`` (m, 3). Returns the endpoint
    rotation vectors ``psi_old^T q``.
    """
    m, Mx = lay.m, lay.Mx
    ep_phi_arr = arrays.edit("elastic_endpoint_phi")
    ep_psi_arr = arrays.edit("elastic_endpoint_psi")
    Xp = arrays.edit("joint_X_p")
    Xc = arrays.edit("joint_X_c")
    rotations = []
    for i, ep in enumerate(lay.endpoints):
        rows = ep.endpoint * Mx + np.arange(m)
        phi_old = ep_phi_arr[rows].astype(np.float64)
        psi_old = ep_psi_arr[rows].astype(np.float64)
        X = Xp if ep.side == 0 else Xc
        pos = X[ep.joint, :3].astype(np.float64) + phi_old.T @ q
        theta = psi_old.T @ q
        rot = X[ep.joint, 3:7].astype(np.float64)
        if np.linalg.norm(theta) > 1.0e-12:
            rot = quat_mul(quat_from_rotvec(theta), rot)
            rot /= np.linalg.norm(rot)
        rotations.append(theta)
        X[ep.joint, :3] = pos
        X[ep.joint, 3:7] = rot
        ep_phi_arr[rows] = ep_phi[i]
        ep_psi_arr[rows] = ep_psi[i]
    return rotations


def counter_rotate_body_rest(arrays: ModelArrays, lay: BodyLayout, rotations: list[np.ndarray]) -> float:
    """Counter-rotate the body's rest rotation in ``model.body_q`` by the mean endpoint rotation.

    ``rot(body_q') = rot(body_q) Q^-1``, ``Q = quat(mean theta)``. Exact only when every endpoint of the body
    carries the same rotation; returns the spread (max deviation from the mean) [rad] and warns above 1e-6.
    """
    thetas = np.array(rotations)
    mismatch = 0.0
    if thetas.size and np.max(np.linalg.norm(thetas, axis=1)) > 1.0e-12:
        theta_mean = thetas.mean(axis=0)
        mismatch = float(np.max(np.linalg.norm(thetas - theta_mean, axis=1)))
        if mismatch > 1.0e-6:
            warnings.warn(
                f"elastic endpoints of body {lay.body} rotated differently (spread {mismatch:.2e} rad); joint rest "
                "rotations are only approximately preserved by RestRotationMode.COUNTER_ROTATE",
                stacklevel=3,
            )
        Qm = quat_from_rotvec(theta_mean)
        Qinv = np.array([-Qm[0], -Qm[1], -Qm[2], Qm[3]])
        bq = arrays.edit("body_q")
        r = quat_mul(bq[lay.body, 3:7].astype(np.float64), Qinv)
        bq[lay.body, 3:7] = r / np.linalg.norm(r)
    return mismatch


def shift_sample_rest_positions(arrays: ModelArrays, lay: BodyLayout, q: np.ndarray, rp_phi, sh_phi) -> None:
    """Render points and shape vertices: ``local += phi_old q`` and install the new per-mode rows."""
    m, Mx = lay.m, lay.Mx
    a, b = lay.render_range
    if b > a:
        loc = arrays.edit("elastic_render_point_local")
        phi_arr = arrays.edit("elastic_render_point_phi")
        rows = np.arange(a, b)[:, None] * Mx + np.arange(m)[None, :]
        old = phi_arr[rows].astype(np.float64)
        loc[a:b] = loc[a:b] + np.einsum("pmc,m->pc", old, q)
        phi_arr[rows] = rp_phi
    if lay.shape_ranges and sh_phi is not None:
        loc = arrays.edit("elastic_shape_vertex_local")
        phi_arr = arrays.edit("elastic_shape_vertex_phi")
        offset = 0
        for a, b in lay.shape_ranges:
            rows = np.arange(a, b)[:, None] * Mx + np.arange(m)[None, :]
            old = phi_arr[rows].astype(np.float64)
            loc[a:b] = loc[a:b] + np.einsum("pmc,m->pc", old, q)
            phi_arr[rows] = sh_phi[offset : offset + (b - a)]
            offset += b - a


def write_modal_arrays(arrays: ModelArrays, lay: BodyLayout, mode_mass, lam, damping, mode_mass_matrix, integrals):
    """Modal mass/stiffness/damping, the five inertia integrals and the modal mass matrix of the body."""
    sl = lay.mode_slice
    arrays.put("elastic_mode_mass", sl, mode_mass)
    arrays.put("elastic_mode_stiffness", sl, lam * mode_mass)
    arrays.put("elastic_mode_damping", sl, damping)
    arrays.put("elastic_mode_coupling_linear", sl, integrals.mode_coupling_linear)
    arrays.put("elastic_mode_coupling_angular", sl, integrals.mode_coupling_angular)
    arrays.put("elastic_mode_coupling_centrifugal", sl, integrals.mode_coupling_centrifugal)
    pair_idx = lay.pair_index()
    arrays.put("elastic_mode_coupling_coriolis", pair_idx, integrals.mode_coupling_coriolis.reshape(-1, 3))
    arrays.put("elastic_mode_coupling_quadratic", pair_idx, integrals.mode_coupling_quadratic.reshape(-1, 3, 3))
    arrays.put("elastic_mode_mass_matrix", pair_idx, mode_mass_matrix.astype(np.float32).reshape(-1))


def write_rigid_properties(arrays: ModelArrays, lay: BodyLayout, mass: float, com: np.ndarray, inertia: np.ndarray):
    """Body mass, COM and inertia (and their inverses for a dynamic frame)."""
    arrays.put("body_mass", lay.body, mass)
    arrays.put("body_com", lay.body, com)
    arrays.put("body_inertia", lay.body, inertia)
    if lay.frame_dynamic:
        arrays.put("body_inv_mass", lay.body, 1.0 / mass if mass > 0 else 0.0)
        arrays.put("body_inv_inertia", lay.body, np.linalg.inv(inertia))


def rotate_sample_rows(arrays: ModelArrays, lay: BodyLayout, Q: np.ndarray) -> None:
    """Rotate the per-mode sample rows (endpoint phi/psi, render, shape vertices): ``new_j = sum_i old_i Q_ij``."""

    def rot(name, items):
        a = arrays.edit(name)
        rows = lay.sample_rows(items)
        old = a[rows].astype(np.float64)  # (P, m, 3)
        a[rows] = np.einsum("pic,ij->pjc", old, Q)

    if lay.endpoints:
        eps = [ep.endpoint for ep in lay.endpoints]
        rot("elastic_endpoint_phi", eps)
        rot("elastic_endpoint_psi", eps)
    a, b = lay.render_range
    if b > a:
        rot("elastic_render_point_phi", np.arange(a, b))
    for a, b in lay.shape_ranges:
        rot("elastic_shape_vertex_phi", np.arange(a, b))


class JointRestPropagator:
    """Keep every rest-target joint's original rest relative rotation by re-propagating ``model.body_q``.

    Rest-target joints are all joints except FREE and ELASTIC (owner) joints. Rest rotations are
    propagated over the undirected graph of those joints from ``root`` (``-1``: the world), whose rotation
    is kept: for joint ``j`` reached from its parent, ``R_c = R_p rot(X_p) T_j rot(X_c)^-1``; from its
    child, the parent is solved instead; a world-parent joint reached from its child rotates its world
    frame ``X_p``. Reproduces every joint's rest target for any tree topology; raises on a loop.
    """

    def __init__(self, model, root: int | None = None):
        self.model = model
        self.root = -1 if root is None else int(root)
        jt = model.joint_type.numpy()
        skip = {int(newton.JointType.FREE), int(newton.JointType.ELASTIC)}
        self.joints = [j for j in range(model.joint_count) if int(jt[j]) not in skip]
        self.reset_targets()

    @staticmethod
    def _rel(j, bq, Xp, Xc, jp, jc):
        p, c = int(jp[j]), int(jc[j])
        Xpj = Xp[j, 3:7].astype(np.float64)
        qp = Xpj if p < 0 else quat_mul(bq[p, 3:7].astype(np.float64), Xpj)
        qc = quat_mul(bq[c, 3:7].astype(np.float64), Xc[j, 3:7].astype(np.float64))
        return quat_mul(quat_conj(qp), qc)

    def _arrays(self):
        m = self.model
        return m.body_q.numpy(), m.joint_X_p.numpy(), m.joint_X_c.numpy(), m.joint_parent.numpy(), m.joint_child.numpy()

    def reset_targets(self) -> None:
        """Record the current rest relative rotations as targets and the propagation order."""
        bq, Xp, Xc, jp, jc = self._arrays()
        self.target = {j: self._rel(j, bq, Xp, Xc, jp, jc) for j in self.joints}
        adj: dict[int, list[int]] = {}
        for j in self.joints:
            adj.setdefault(int(jp[j]), []).append(j)
            adj.setdefault(int(jc[j]), []).append(j)
        self.order: list[tuple[int, int, int]] = []
        seen = {self.root}
        used: set[int] = set()
        dq = deque([self.root])
        while dq:
            b = dq.popleft()
            for j in adj.get(b, []):
                if j in used:
                    continue
                used.add(j)
                other = int(jc[j]) if int(jp[j]) == b else int(jp[j])
                if other in seen:
                    raise ValueError("kinematic loop in the rest-target joint graph: rest rotations need a tree")
                if other >= 0:
                    seen.add(other)
                    dq.append(other)
                self.order.append((j, b, other))

    def fix(self) -> None:
        """Re-propagate the rest rotations (one gather and one scatter of ``body_q``, ``joint_X_p`` if needed)."""
        m = self.model
        bq, Xp, Xc, jp, _jc = self._arrays()
        Xp_changed = False
        for j, known, other in self.order:
            T = self.target[j]
            Xpj = Xp[j, 3:7].astype(np.float64)
            Xcj = Xc[j, 3:7].astype(np.float64)
            if other < 0:  # world-parent joint reached from its child: rotate its world frame X_p
                qc = quat_mul(bq[known, 3:7].astype(np.float64), Xcj)
                q = quat_mul(qc, quat_conj(T))
                Xp[j, 3:7] = q / np.linalg.norm(q)
                Xp_changed = True
                continue
            if known == int(jp[j]):  # solve for the child: q_c = q_p T Xc^-1
                qp = Xpj if known < 0 else quat_mul(bq[known, 3:7].astype(np.float64), Xpj)
                q = quat_mul(quat_mul(qp, T), quat_conj(Xcj))
            else:  # solve for the parent: q_p_body = q_c T^-1 Xp^-1
                qc = quat_mul(bq[known, 3:7].astype(np.float64), Xcj)
                q = quat_mul(quat_mul(qc, quat_conj(T)), quat_conj(Xpj))
            bq[other, 3:7] = q / np.linalg.norm(q)
        m.body_q.assign(bq)
        if Xp_changed:
            m.joint_X_p.assign(Xp)

    def error(self) -> float:
        """Max angle [rad] between the current and the target rest relative rotations."""
        bq, Xp, Xc, jp, jc = self._arrays()
        e = 0.0
        for j in self.joints:
            d = quat_mul(quat_conj(self.target[j]), self._rel(j, bq, Xp, Xc, jp, jc))
            e = max(e, quat_angle(d))
        return e


# ----------------------------------------------------------------------------------------------
# initial reads
# ----------------------------------------------------------------------------------------------


def read_modal_diagonals(model, lay: BodyLayout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Modal mass, stiffness and damping of the body (float64)."""
    sl = lay.mode_slice
    mm = model.elastic_mode_mass.numpy()[sl].astype(np.float64)
    kk = model.elastic_mode_stiffness.numpy()[sl].astype(np.float64)
    cc = model.elastic_mode_damping.numpy()[sl].astype(np.float64)
    return mm, kk, cc


def read_body_inertial(model, body: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Model mass, inertia and COM of ``body`` (float64)."""
    return (
        float(model.body_mass.numpy()[body]),
        model.body_inertia.numpy()[body].astype(np.float64),
        model.body_com.numpy()[body].astype(np.float64),
    )


def sample_weight_replay(model, lay: BodyLayout, basis, r0: np.ndarray, base_sample_count: int | None):
    """Sparse ``W (S, N)`` with ``sample_phi = W @ nodal_phi`` for every sample of the body's ``ModalBasis``.

    Base samples: the rest nodes (identity) or generator-style coincident/IDW over the nodes. Appended
    samples: replay of the builder's ``ModalBasis.add_samples`` batches (each joint endpoint, then each
    elastic body's render points, then each elastic shape, over every body sharing the basis), each new
    sample IDW-interpolated from the samples that existed before its batch. Returns ``(W, mapped_count)``.
    """
    import scipy.sparse as sp

    from .numerics import idw_rows  # noqa: PLC0415

    N = int(r0.shape[0])
    pts = basis.sample_points
    S = int(pts.shape[0])
    eps = float(basis.interpolation_epsilon)
    nodes32 = r0.astype(np.float32)
    n_base = base_sample_count
    if n_base is None:
        if S >= N and np.array_equal(pts[:N], nodes32):
            n_base = N
        else:
            raise ValueError("pass base_sample_count: the basis samples do not start with the rest nodes")
    if n_base == N and np.array_equal(pts[:N], nodes32):
        W_base = sp.identity(N, format="csr")
    else:
        from scipy.spatial import cKDTree

        Wb = np.zeros((n_base, N))
        dist, idx = cKDTree(nodes32.astype(np.float64)).query(pts[:n_base].astype(np.float64), k=2)
        for i in range(n_base):
            if dist[i, 0] == 0.0 and dist[i, 1] > 0.0:
                Wb[i, idx[i, 0]] = 1.0
            else:
                Wb[i] = idw_rows(nodes32, pts[i : i + 1], eps)[0]
        W_base = sp.csr_matrix(Wb)
    blocks = [W_base]

    m = model
    eb = m.elastic_basis.numpy()
    bodies_e = np.nonzero(eb == lay.basis_index)[0]
    body_ids = {int(b) for b in m.elastic_body.numpy()[bodies_e]}
    batches: list[np.ndarray] = []
    if m.elastic_endpoint_count > 0:
        jp = m.joint_parent.numpy()
        jc = m.joint_child.numpy()
        pe = m.joint_parent_elastic_endpoint.numpy()
        ce = m.joint_child_elastic_endpoint.numpy()
        ep_sample = m.elastic_endpoint_sample.numpy()
        for j in range(m.joint_count):
            if pe[j] >= 0 and int(jp[j]) in body_ids:
                batches.append(np.array([ep_sample[pe[j]]]))
            if ce[j] >= 0 and int(jc[j]) in body_ids:
                batches.append(np.array([ep_sample[ce[j]]]))
    rp_start = m.elastic_render_point_start.numpy()
    rp_count = m.elastic_render_point_count.numpy()
    rp_sample = m.elastic_render_point_sample.numpy()
    for e in sorted(int(x) for x in bodies_e):
        batches.append(rp_sample[rp_start[e] : rp_start[e] + rp_count[e]])
    if m.elastic_shape_count > 0:
        sb = m.elastic_shape_body.numpy()
        vs = m.elastic_shape_vertex_start.numpy()
        vc = m.elastic_shape_vertex_count.numpy()
        vsample = m.elastic_shape_vertex_sample.numpy()
        for i in range(m.elastic_shape_count):
            if int(sb[i]) in body_ids:
                batches.append(vsample[vs[i] : vs[i] + vc[i]])
    count = n_base
    for raw in batches:
        batch = np.asarray(raw, dtype=np.int64)
        new = np.unique(batch[batch >= count])
        if new.size == 0:
            continue
        if not np.array_equal(new, np.arange(count, count + new.size)):
            warnings.warn("unexpected sample creation order while replaying the builder batches", stacklevel=3)
        rows = idw_rows(pts[:count], pts[new], eps)
        prior = sp.vstack(blocks, format="csr")[:count]
        blocks.append(sp.csr_matrix(np.asarray(prior.T @ rows.T).T))
        count = int(new[-1]) + 1
    W = sp.vstack(blocks, format="csr")
    if W.shape[0] < S:  # samples created by other paths (unreferenced here): zero rows
        W = sp.vstack([W, sp.csr_matrix((S - W.shape[0], N))], format="csr")
    return W, count
