# ruff: noqa: TID253
"""Craig-Bampton (CB) bodies with interface frames for re-ROM: basis, segments, chains, linear statics.

A CB body's floating frame is its ROOT face (an RBE2 patch that *is* the body frame: ``phi = psi = 0``
there in every basis vector); an optional TIP face is an RBE2 patch tied to a 6-DOF interface frame at
its centroid, ``u_j = t + theta x (x_j - p)``. Basis about the current reference (root held)::

    [Psi (6 constraint modes: unit tip-frame motions, interior from -K_II^-1 K_Ib A),
     interface-held vectors (Ritz K_II^-1 f_I, given extra vectors, fixed-interface eigenmodes)]

then Rayleigh-Ritz rotated so ``Phi^T M Phi = I`` and ``Phi^T K Phi`` is diagonal. The tip-frame motion of
every basis vector is exact (the frame map ``F``, 6 x m), so the interface endpoint ``phi``/``psi`` are
exact. Neighbouring bodies welded by FIXED joints at coincident frames give a displacement-conforming
interface (RBE2 on both sides). See ``exp-re-rom/notes/v2_wp3b_multiframe.md`` for the RBE2 vs RBE3 choice.

The re-ROM update of such a body is :class:`rerom.InterfaceBodyAdapter`; :class:`rerom.MultiFrameController`
drives a chain.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import newton
from newton._src.sim.modal import _reduced_inertia_from_pairs

from .assembly import NumpyTetAssembler, TetMesh
from .numerics import MOrthonormalBlock, node_dofs, rayleigh_ritz, rbe2_rows, rigid_properties, skew, to_nodal

__all__ = [
    "CBChain",
    "CBSegment",
    "build_cb_chain",
    "cb_basis",
    "full_fe_linear_static",
    "install_interface_endpoint_maps",
    "linear_chain_static",
    "make_segments",
    "modal_basis_for",
    "newton_cb_chain_static",
    "segment_from_mesh",
]


@dataclass
class CBSegment:
    """One CB body cut from a tet mesh: rest geometry in the body frame, assembler, interface node sets, basis."""

    index: int
    global_ids: np.ndarray  # (n,) global node ids
    local_nodes: np.ndarray  # (n,3) rest nodes in the body frame (origin = root-face centroid)
    origin: np.ndarray  # (3,) world position of the body origin at rest
    tets_local: np.ndarray
    global_tets: np.ndarray  # (E_s,) indices into the global tet array
    asm: NumpyTetAssembler  # nonlinear assembler on the segment's own elements (rest = local_nodes)
    M: sp.csr_matrix
    root_ids: np.ndarray
    tip_ids: np.ndarray  # empty for a body without a tip interface
    tip_point: np.ndarray | None  # tip-frame position in the body frame (rest)
    x_range: tuple
    Phi: np.ndarray | None = None  # (3n, m)
    lam: np.ndarray | None = None
    frame_map: np.ndarray | None = None  # (6, m) tip-frame [t; theta] per modal coordinate (None: no tip)
    basis_info: dict = field(default_factory=dict)
    body: int = -1
    q_start: int = -1
    qd_start: int = -1

    @property
    def has_tip(self) -> bool:
        return self.tip_ids.size > 0

    @property
    def n_nodes(self) -> int:
        return int(self.local_nodes.shape[0])


def cb_basis(
    x_ref: np.ndarray,
    K,
    M,
    root_ids: np.ndarray,
    tip_ids: np.ndarray,
    tip_pos: np.ndarray | None,
    mode_count: int,
    *,
    ritz_loads=(),
    extra_vectors=(),
    eig_margin: int = 2,
    drop_tol: float = 1.0e-6,
    eigensolver=None,
):
    """Craig-Bampton basis about ``x_ref`` (N,3) with stiffness ``K`` (linear or tangent).

    Coordinates: root face held (body frame); tip face (if any) = RBE2 patch of a 6-DOF frame at
    ``tip_pos``; interior I = everything else. Basis before rotation: ``[Psi (6), interface-held]``
    where the interface-held group is ``ritz_loads`` solved with ``K_II`` (``K_II^-1 f_I``),
    ``extra_vectors`` (taken as given; their rigid tip-face part is removed through the constraint modes)
    and the lowest fixed-interface eigenmodes filling up to ``mode_count``. The fixed-interface eigenmodes
    use ARPACK shift-invert at ``sigma = 0`` on the same ``K_II`` factorization (``eigensolver=None``) or
    the given ``newton.ModalEigensolverBackend``.

    Returns ``(Phi (3N,m), lam (m,), F (6,m) or None, info)`` with ``Phi^T M Phi = I``,
    ``Phi^T K Phi = diag(lam)``, ``F`` the tip-frame motion per modal coordinate.
    """
    t0 = time.perf_counter()
    x_ref = np.asarray(x_ref, dtype=np.float64).reshape(-1, 3)
    n = x_ref.shape[0]
    ndof = 3 * n
    K = sp.csr_matrix(K)
    M = sp.csr_matrix(M)
    fixed = np.zeros(ndof, dtype=bool)
    fixed[node_dofs(root_ids)] = True
    has_tip = tip_ids is not None and len(tip_ids) > 0
    b = node_dofs(tip_ids) if has_tip else np.zeros(0, np.int64)
    fixed[b] = True
    I = np.nonzero(~fixed)[0]
    K_II = sp.csc_matrix(K[I][:, I])
    M_II = sp.csc_matrix(M[I][:, I])
    info = {"interior_dofs": int(I.size)}
    tf = time.perf_counter()
    lu = spla.splu(K_II)
    info["t_factor"] = time.perf_counter() - tf

    n_psi = 6 if has_tip else 0
    cols = []
    if has_tip:
        A = rbe2_rows(x_ref[tip_ids], np.asarray(tip_pos, dtype=np.float64))
        K_Ib = K[I][:, b]
        rhs = -(K_Ib @ A)
        Psi_I = lu.solve(np.asarray(rhs))
        Psi = np.zeros((ndof, 6))
        Psi[b] = A
        Psi[I] = Psi_I
        cols.append(Psi)
    n_held = mode_count - n_psi
    if n_held < 0:
        raise ValueError("mode_count must be >= 6 for a segment with a tip interface")

    # interface-held group (zero on root and tip faces); M-orthonormal Gram-Schmidt
    held = MOrthonormalBlock(M_II, n=I.size, drop_tol=drop_tol)

    tr = time.perf_counter()
    for name, load in ritz_loads:
        if held.count >= n_held:
            break
        f = np.asarray(load, dtype=np.float64)
        if not np.any(f[I]):
            held.dropped.append(name)
            continue
        held.accept(name, lu.solve(f[I]))
    for name, vec in extra_vectors:
        if held.count >= n_held:
            break
        v = np.asarray(vec, dtype=np.float64)
        if has_tip:  # remove the rigid tip-face part (least-squares frame fit) through the constraint modes
            fr = np.linalg.lstsq(A, v[b], rcond=None)[0]
            v = v - Psi @ fr
        held.accept(name, v[I])
    info["t_ritz"] = time.perf_counter() - tr
    n_eig = n_held - held.count
    te = time.perf_counter()
    if n_eig > 0:
        k = min(n_eig + eig_margin + len(held.accepted), I.size - 1)
        if eigensolver is None:
            op = spla.LinearOperator(K_II.shape, matvec=lu.solve, dtype=np.float64)
            rng = np.random.default_rng(12345)
            v0 = lu.solve(rng.standard_normal(I.size))
            vals, vecs = spla.eigsh(K_II, k=k, M=M_II, sigma=0.0, which="LM", OPinv=op, v0=v0, tol=0.0)
        else:
            res = eigensolver.solve(K_II, M_II, k)
            vals, vecs = np.asarray(res.values), np.asarray(res.vectors)
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        info["fixed_interface_eigs"] = vals.tolist()
        info["negative_fixed_interface_eigs"] = int(np.count_nonzero(vals < 0))
        j = 0
        target = held.count + n_eig
        while held.count < target and j < vecs.shape[1]:
            held.accept(f"eig{j}", vecs[:, j])
            j += 1
        if held.count < target:
            raise RuntimeError("could not complete the interface-held basis")
    info["t_eig"] = time.perf_counter() - te
    if held.dropped:
        info["dropped"] = list(held.dropped)
    H = np.zeros((ndof, held.count))
    H[I] = held.columns
    cols.append(H)
    B = np.column_stack(cols)
    info["held_names"] = list(held.accepted)

    lam, Phi, Q = rayleigh_ritz(B, K, M)
    F = None
    if has_tip:
        FB = np.zeros((6, B.shape[1]))
        FB[:, :6] = np.eye(6)
        F = FB @ Q
        # exactness check of the frame map: tip face displacement = A F
        info["frame_map_residual"] = float(
            np.max(np.abs(Phi[b] - rbe2_rows(x_ref[tip_ids], np.asarray(tip_pos)) @ F))
            / max(np.max(np.abs(Phi[b])), 1e-30)
        )
    info["t_total"] = time.perf_counter() - t0
    info["t_rayleigh"] = info["t_total"] - info["t_factor"] - info["t_ritz"] - info["t_eig"]
    return Phi, lam, F, info


# ----------------------------------------------------------------------------------------------
# segments
# ----------------------------------------------------------------------------------------------


def _split(nodes, tets, n_segments, cuts_x):
    x = nodes[:, 0]
    xs = np.unique(np.round(x, 9))
    ncell = len(xs) - 1
    if cuts_x is None:
        cuts = [round(k * ncell / n_segments) for k in range(n_segments + 1)]
    else:
        cuts = [0, *(int(np.argmin(np.abs(xs - c))) for c in cuts_x), ncell]
        if len(cuts) - 1 != n_segments:
            raise ValueError("len(cuts_x) must be n_segments - 1")
    tmin = x[tets].min(axis=1)
    tmax = x[tets].max(axis=1)
    out = []
    for k in range(n_segments):
        xa, xb = xs[cuts[k]], xs[cuts[k + 1]]
        tmask = (tmin >= xa - 1e-9) & (tmax <= xb + 1e-9)
        out.append((float(xa), float(xb), np.nonzero(tmask)[0]))
    if sum(o[2].size for o in out) != tets.shape[0]:
        raise ValueError("every tet must belong to exactly one segment")
    return out


def make_segments(nodes, tets, rho, E, nu, n_segments, cuts_x=None) -> list[CBSegment]:
    """Cut a tet mesh at x stations into ``n_segments`` CB segments (StVK assemblers, root/tip face sets)."""
    nodes = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    segs = []
    for k, (xa, xb, tet_ids) in enumerate(_split(nodes, tets, n_segments, cuts_x)):
        st = tets[tet_ids]
        gids = np.unique(st)
        remap = -np.ones(nodes.shape[0], dtype=np.int64)
        remap[gids] = np.arange(gids.size)
        tl = remap[st]
        X = nodes[gids]
        root_ids = np.nonzero(np.abs(X[:, 0] - xa) < 1e-9)[0]
        tip_ids = np.nonzero(np.abs(X[:, 0] - xb) < 1e-9)[0] if k < n_segments - 1 else np.zeros(0, np.int64)
        origin = X[root_ids].mean(axis=0)
        local = X - origin
        tip_point = (X[tip_ids].mean(axis=0) - origin) if tip_ids.size else None
        mesh = TetMesh(local, tl, rho, E, nu)
        asm = NumpyTetAssembler(mesh, "stvk")
        segs.append(
            CBSegment(
                index=k,
                global_ids=gids,
                local_nodes=local,
                origin=origin,
                tets_local=tl,
                global_tets=tet_ids,
                asm=asm,
                M=sp.csr_matrix(asm.mass_matrix),
                root_ids=root_ids,
                tip_ids=tip_ids,
                tip_point=tip_point,
                x_range=(xa, xb),
            )
        )
    return segs


def segment_from_mesh(mesh: TetMesh, root_ids, tip_ids, *, mode_count=16, ritz_loads=(), material="stvk") -> CBSegment:
    """One CB body from a whole mesh: body origin = root-face centroid, tip frame = tip-face centroid.

    ``ritz_loads`` are ``(name, f)`` in the body frame (e.g. the gravity load) added to the initial basis.
    """
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    root = np.asarray(root_ids, dtype=np.int64)
    tip = np.asarray(tip_ids, dtype=np.int64)
    origin = nodes[root].mean(axis=0)
    local = nodes - origin
    m_local = TetMesh(local, mesh.tets, mesh.density, mesh.young, mesh.poisson)
    asm = NumpyTetAssembler(m_local, material)
    seg = CBSegment(
        index=0,
        global_ids=np.arange(nodes.shape[0]),
        local_nodes=local,
        origin=origin,
        tets_local=mesh.tets,
        global_tets=np.arange(mesh.tets.shape[0]),
        asm=asm,
        M=sp.csr_matrix(asm.mass_matrix),
        root_ids=root,
        tip_ids=tip,
        tip_point=local[tip].mean(axis=0) if tip.size else None,
        x_range=(0.0, 1.0),
    )
    seg.Phi, seg.lam, seg.frame_map, seg.basis_info = cb_basis(
        local, asm.linear_stiffness(), seg.M, root, tip, seg.tip_point, mode_count, ritz_loads=ritz_loads
    )
    return seg


def modal_basis_for(seg: CBSegment, label: str) -> newton.ModalBasis:
    """``newton.ModalBasis`` of a CB segment (samples = rest nodes, inertia integrals, rigid properties)."""
    n = seg.n_nodes
    m = seg.Phi.shape[1]
    nodal_phi = to_nodal(seg.Phi, n)  # (n, m, 3)
    mass_phi = to_nodal(seg.M @ seg.Phi, n)
    integ = _reduced_inertia_from_pairs(seg.local_nodes, nodal_phi, mass_phi)
    mass, com, inertia = rigid_properties(seg.M, seg.local_nodes)
    Mm = seg.Phi.T @ (seg.M @ seg.Phi)
    return newton.ModalBasis(
        sample_points=seg.local_nodes.astype(np.float32),
        sample_phi=nodal_phi.astype(np.float32),
        mode_mass=np.diag(Mm).copy(),
        mode_stiffness=seg.lam * np.diag(Mm),
        mode_damping=np.zeros(m),
        mode_coupling_linear=integ.mode_coupling_linear,
        mode_coupling_angular=integ.mode_coupling_angular,
        mode_coupling_centrifugal=integ.mode_coupling_centrifugal,
        mode_coupling_coriolis=integ.mode_coupling_coriolis,
        mode_coupling_quadratic=integ.mode_coupling_quadratic,
        mode_mass_matrix=0.5 * (Mm + Mm.T),
        rigid_mass=mass,
        rigid_com=com,
        rigid_inertia=inertia,
        label=label,
    )


def install_interface_endpoint_maps(model, segments, offset=None) -> None:
    """Overwrite elastic endpoint phi/psi of CB bodies with the exact frame maps.

    An endpoint at ``tip_point + d`` moves rigidly with the RBE2 tip frame (``phi = F_t + F_r x d``,
    ``psi = F_r``); one at ``d`` is rigid with the body frame (``phi = psi = 0``); ``d = offset`` (default 0).
    """
    d = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)
    Mx = int(model.elastic_max_mode_count)
    ep_phi = model.elastic_endpoint_phi.numpy()
    ep_psi = model.elastic_endpoint_psi.numpy()
    pe = model.joint_parent_elastic_endpoint.numpy()
    ce = model.joint_child_elastic_endpoint.numpy()
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    by_body = {s.body: s for s in segments}
    Xp = model.joint_X_p.numpy()
    Xc = model.joint_X_c.numpy()
    for j in range(model.joint_count):
        for ep, b, X in ((pe[j], jp[j], Xp), (ce[j], jc[j], Xc)):
            if ep < 0 or b not in by_body:
                continue
            s = by_body[int(b)]
            m = s.Phi.shape[1]
            rows = ep * Mx + np.arange(m)
            pos = X[j, :3].astype(np.float64)
            if s.has_tip and np.linalg.norm(pos - (s.tip_point + d)) < 1e-6:
                F = s.frame_map
                ep_phi[rows] = (F[:3] + np.cross(F[3:].T, d).T).T
                ep_psi[rows] = F[3:].T
            elif np.linalg.norm(pos - d) < 1e-6:  # rigid with the body frame
                ep_phi[rows] = 0.0
                ep_psi[rows] = 0.0
            else:
                raise ValueError(f"joint {j} endpoint on body {b} is neither at the root nor at the tip frame")
    model.elastic_endpoint_phi.assign(ep_phi)
    model.elastic_endpoint_psi.assign(ep_psi)


@dataclass
class CBChain:
    """A chain of CB bodies welded at coincident interface frames (root of segment 0 clamped)."""

    model: object
    segments: list
    joints: list  # [clamp, interface_0_1, ...]
    n_nodes: int
    n_tets: int
    gravity: float


def _transform(pos, quat=None):
    import warp as wp  # noqa: PLC0415

    p = wp.vec3(*np.asarray(pos, dtype=np.float32))
    q = wp.quat_identity() if quat is None else wp.quat(*np.asarray(quat, dtype=np.float32))
    return wp.transform(p, q)


def build_cb_chain(
    nodes,
    tets,
    rho,
    E,
    nu,
    *,
    n_segments: int,
    cuts_x=None,
    mode_count: int = 12,
    n_ritz_initial: int = 0,
    gravity: float = 9.81,
    device=None,
    articulation: bool = False,
) -> CBChain:
    """N CB bodies from one tet mesh, welded by FIXED joints at coincident interface frames, root clamped.

    ``n_ritz_initial > 0`` adds the interface-held gravity Ritz vector to the initial basis (the ``f_int``
    candidate is zero at rest), as the re-ROM controller would at a null update. ``articulation``: bodies
    are articulation links and the joints one articulation (for
    ``SolverVBD(rigid_articulation_solve="block_sparse_joints")``).
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    segs = make_segments(nodes, tets, rho, E, nu, n_segments, cuts_x)
    gvec = np.array([0.0, 0.0, -float(gravity)])
    for s in segs:
        K0 = s.asm.linear_stiffness()
        loads = [("f_ext", s.M @ np.tile(gvec, s.n_nodes))] if n_ritz_initial > 0 else []
        s.Phi, s.lam, s.frame_map, s.basis_info = cb_basis(
            s.local_nodes, K0, s.M, s.root_ids, s.tip_ids, s.tip_point, mode_count, ritz_loads=loads
        )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -float(gravity)))
    add = builder.add_link_elastic if articulation else builder.add_body_elastic
    for s in segs:
        s.body = add(
            xform=_transform(s.origin),
            modal_basis=modal_basis_for(s, f"cb_segment_{s.index}"),
            label=f"cb_segment_{s.index}",
        )
    joints = [
        builder.add_joint_fixed(
            parent=-1,
            child=segs[0].body,
            parent_xform=_transform(segs[0].origin),
            child_xform=_transform(np.zeros(3)),
            label="clamp_root",
        )
    ]
    for a, b in itertools.pairwise(segs):
        gap = np.linalg.norm(a.origin + a.tip_point - b.origin)
        if gap > 1e-9:
            raise ValueError(f"interface frames of segments {a.index}/{b.index} do not coincide ({gap:.2e} m)")
        joints.append(
            builder.add_joint_fixed(
                parent=a.body,
                child=b.body,
                parent_xform=_transform(a.tip_point),
                child_xform=_transform(np.zeros(3)),
                label=f"interface_{a.index}_{b.index}",
            )
        )
    if articulation:
        builder.add_articulation(joints, label="cb_chain")
    builder.color()
    model = builder.finalize(device=device)
    eb = model.elastic_body.numpy()
    ej = model.elastic_joint.numpy()
    qs = model.joint_q_start.numpy()
    qds = model.joint_qd_start.numpy()
    for s in segs:
        k = int(np.nonzero(eb == s.body)[0][0])
        s.q_start = int(qs[ej[k]])
        s.qd_start = int(qds[ej[k]])
    chain = CBChain(
        model=model,
        segments=segs,
        joints=joints,
        n_nodes=int(nodes.shape[0]),
        n_tets=int(np.asarray(tets).shape[0]),
        gravity=float(gravity),
    )
    install_interface_endpoint_maps(model, segs)
    return chain


# ----------------------------------------------------------------------------------------------
# linear statics (validation)
# ----------------------------------------------------------------------------------------------


def full_fe_linear_static(nodes, tets, rho, E, nu, g_vec, clamp_ids):
    """Full FE linear static displacement (3N,) under gravity with the clamp nodes fixed."""
    mesh = TetMesh(np.asarray(nodes, float), np.asarray(tets, np.int64), rho, E, nu)
    asm = NumpyTetAssembler(mesh, "stvk")
    K = asm.linear_stiffness()
    f = asm.mass_matrix @ np.tile(np.asarray(g_vec, float), mesh.num_nodes)
    free = np.ones(3 * mesh.num_nodes, bool)
    free[node_dofs(clamp_ids)] = False
    u = np.zeros(3 * mesh.num_nodes)
    u[free] = spla.spsolve(sp.csc_matrix(K[free][:, free]), f[free])
    return u


def linear_chain_static(segs: list, n_nodes: int, g_vec):
    """Linear static solution of a welded CB chain (exact kinematic welds, root clamped).

    Segment k's body frame moves rigidly with segment k-1's tip frame (``frame_map``); node
    displacement = t_k + theta_k x X + Phi_k q_k. With M-orthonormal, K-diagonal bases the reduced
    stiffness is diag(lam) and ``Q = lam^-1 T^T f``. Returns (u_global averaged (3N,), Q).
    """
    mtot = sum(s.Phi.shape[1] for s in segs)
    offs = np.cumsum([0] + [s.Phi.shape[1] for s in segs])
    t_frame = np.zeros((3, mtot))
    r_frame = np.zeros((3, mtot))
    blocks = []
    for k, s in enumerate(segs):
        cols = slice(offs[k], offs[k + 1])
        Uk = t_frame[None] - np.einsum("nij,jm->nim", np.stack([skew(x) for x in s.local_nodes]), r_frame)
        Uk[:, :, cols] += s.Phi.reshape(-1, 3, s.Phi.shape[1])
        blocks.append(Uk.reshape(-1, mtot))
        if s.frame_map is None:
            break
        t_new = t_frame - skew(s.tip_point) @ r_frame
        t_new[:, cols] += s.frame_map[:3]
        r_new = r_frame.copy()
        r_new[:, cols] += s.frame_map[3:]
        t_frame, r_frame = t_new, r_new
    lam = np.concatenate([s.lam for s in segs])
    f = np.concatenate([s.M @ np.tile(np.asarray(g_vec, float), s.n_nodes) for s in segs])
    T = np.vstack(blocks)
    Q = (T.T @ f) / lam
    u = np.zeros(3 * n_nodes)
    cnt = np.zeros(n_nodes)
    row = 0
    for s in segs:
        n_rows = 3 * s.n_nodes
        u[node_dofs(s.global_ids)] += T[row : row + n_rows] @ Q
        cnt[s.global_ids] += 1
        row += n_rows
    u /= np.repeat(cnt, 3)
    return u, Q


def newton_cb_chain_static(segs: list, n_nodes: int, g_vec, *, coupling="rbe2", fixed_mode_count=6):
    """Linear static of a chain built with Newton's ``ModalGeneratorCraigBamptonFEM`` (free-free CB).

    Every segment gets interfaces at its root and tip faces (``coupling`` for the cuts; the clamped
    root face of segment 0 always RBE2). Interface frames are shared between neighbours (exact weld),
    frame 0 is fixed. The global reduced system is solved densely; returns
    (u_global averaged (3N,), info).
    """
    from newton._src.sim.modal_craig_bampton_fem import ModalGeneratorCraigBamptonFEM  # noqa: PLC0415

    nseg = len(segs)
    nframes = nseg + 1  # frame k = root of segment k, frame nseg = tip of the last segment
    sizes = []
    gens = []
    for k, s in enumerate(segs):
        ifs = [
            {
                "node_indices": s.root_ids,
                "coupling": "rbe2" if k == 0 else coupling,
                "position": np.zeros(3),
                "name": "root",
            }
        ]
        tip_ids = (
            s.tip_ids if s.has_tip else np.nonzero(np.abs(s.local_nodes[:, 0] - s.local_nodes[:, 0].max()) < 1e-9)[0]
        )
        tip_pos = s.tip_point if s.has_tip else s.local_nodes[tip_ids].mean(axis=0)
        ifs.append(
            {"node_indices": tip_ids, "coupling": coupling if s.has_tip else "rbe3", "position": tip_pos, "name": "tip"}
        )
        gen = ModalGeneratorCraigBamptonFEM(
            s.local_nodes,
            s.M,
            s.asm.linear_stiffness(),
            ifs,
            fixed_mode_count=fixed_mode_count,
            dofs_per_node=3,
            sample_mass=None,
        )
        gen.reduce()
        gens.append(gen)
        sizes.append(gen.reduced_stiffness_matrix.shape[0] - 12)
    nz = 6 * nframes + int(sum(sizes))
    Kg = np.zeros((nz, nz))
    fg = np.zeros(nz)
    maps = []
    off = 6 * nframes
    for k, (s, gen) in enumerate(zip(segs, gens, strict=True)):
        idx = np.concatenate([np.arange(6 * k, 6 * k + 12), off + np.arange(sizes[k])])
        off += sizes[k]
        Kr = gen.reduced_stiffness_matrix
        T = gen.full_recovery_matrix()
        f = T.T @ (s.M @ np.tile(np.asarray(g_vec, float), s.n_nodes))
        Kg[np.ix_(idx, idx)] += Kr
        fg[idx] += f
        maps.append((idx, T))
    free = np.arange(6, nz)
    z = np.zeros(nz)
    z[free] = np.linalg.solve(Kg[np.ix_(free, free)], fg[free])
    u = np.zeros(3 * n_nodes)
    cnt = np.zeros(n_nodes)
    for s, (idx, T) in zip(segs, maps, strict=True):
        u[node_dofs(s.global_ids)] += T @ z[idx]
        cnt[s.global_ids] += 1
    u /= np.repeat(cnt, 3)
    gap = 0.0
    for a in range(nseg - 1):
        sa, sb = segs[a], segs[a + 1]
        _, ia, ib = np.intersect1d(sa.global_ids, sb.global_ids, return_indices=True)
        ua = (maps[a][1] @ z[maps[a][0]]).reshape(-1, 3)[ia]
        ub = (maps[a + 1][1] @ z[maps[a + 1][0]]).reshape(-1, 3)[ib]
        gap = max(gap, float(np.max(np.linalg.norm(ua - ub, axis=1))))
    return u, {"interface_node_gap": gap, "reduced_dofs": nz - 6}
