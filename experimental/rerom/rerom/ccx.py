# ruff: noqa: TID253, PLW1510  (moved as-is from the experiment package)
"""WP0 (v2): file coupling to CalculiX ccx 2.21 for geometrically nonlinear shell panels.

Why thin solids and not S4 shells
---------------------------------
ccx has no native shell element: every S4 shell is expanded internally into one C3D8I brick
(incompatible-mode 8-node solid) whose nodes are two generated nodes per shell node at
x -/+ (t/2) n, plus 3 x 3 incompatible-mode ("bubble") DOFs per element that ccx carries as extra
nodes.  Exported matrices (``SOLVER=MATRIXSTORAGE``) are in those generated DOFs, the ``.dof``
file labels BOTH generated nodes with the parent shell node number, and the generated nodes
cannot be addressed by ``*BOUNDARY``; prescribing shell rotations turns nodes into rigid knots.
A full deformed state therefore cannot be prescribed on an S4 model.

``CcxModel`` builds the same C3D8I brick layer explicitly (``element="C3D8I"``, default): two
user nodes per shell node, ccx node id ``2 i + 1`` (bottom, x - t/2 n) and ``2 i + 2`` (top,
x + t/2 n) for shell node row ``i``.  The stiffness/mass matrices are identical to ccx's own S4
expansion when the same nodal normals are used (``element="S4"`` with ``*NORMAL`` is kept for
that check).  The bubble DOFs are condensed statically, so every exported quantity lives on

    DOF layout:  q[6 i + 0:3] = bottom-node displacement (x, y, z) of shell node i
                 q[6 i + 3:6] = top-node displacement    (x, y, z) of shell node i

i.e. 6 N DOFs for N shell nodes (same count as a 6-DOF shell model; mid-surface translation
u = (bot + top) / 2, director change (top - bot) / t).

Sign convention (as in fe_nonlinear): f_int = dW/dq (the force the body exerts against
deformation), M a + f_int(q) = f_ext, K_T = d f_int / dq (symmetric).  With every node
prescribed and no load, ccx's reaction force RF equals f_int.

Material: ccx linear elastic + NLGEOM = St. Venant-Kirchhoff (total Lagrangian), so f_int is a
function of the state only.

Welds (coincident node pairs a-b) are written exactly for finite rotations: node b's bottom and
top nodes follow node a's mid point and director,
    u_b^bot = (1/2 + tb/(2 ta)) u_a^bot + (1/2 - tb/(2 ta)) u_a^top
    u_b^top = (1/2 - tb/(2 ta)) u_a^bot + (1/2 + tb/(2 ta)) u_a^top,
which is linear and exact when both nodes share the rest normal (welded pairs get the averaged
normal).  The same rows are available as C (C q = 0) for constraint elimination in Python.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# CalculiX executable: $REROM_CCX, else ``ccx`` on PATH, else the exp-re-rom user-space build (tools/ccx/ccx.sh)
CCX_DEFAULT = Path(
    os.environ.get("REROM_CCX")
    or shutil.which("ccx")
    or Path(__file__).resolve().parents[4] / "tools" / "ccx" / "ccx.sh"
)

__all__ = ["CcxError", "CcxModel", "ShellMesh", "make_plate_mesh", "rotvec_to_matrix"]


class CcxError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------
# mesh
# ----------------------------------------------------------------------------------------------
@dataclass
class ShellMesh:
    """Quad shell mid-surface mesh.  Rows (0-based) index nodes; ``node_ids`` keep external IDs."""

    points: np.ndarray  # (N, 3)
    quads: np.ndarray  # (E, 4) node rows, right-hand normal = shell normal
    elem_part: np.ndarray  # (E,) part name per element
    parts: dict  # part -> dict(thickness, E, nu, rho)
    welds: list = field(default_factory=list)  # [(row_a, row_b)] coincident welded pairs
    node_ids: np.ndarray | None = None
    elem_ids: np.ndarray | None = None
    clamps: list = field(default_factory=list)  # [dict(clamp_id, nodes(rows), active{...}, ...)]
    # General linear tie constraints of the source package in its 6-DOF shell layout (u, theta per node,
    # node-major, columns indexed by this mesh's node rows): C_shell q_shell = 0. Used when the package
    # ties are not coincident-node pairs (e.g. offset welds to a triangle); see CcxModel.weld_rows().
    shell_C: object = None
    # For meshes refined from a source mesh: parents[i] = tuple of the source/parent rows node i interpolates
    # (a single row for an original node). orig_count = number of source nodes (rows 0..orig_count-1).
    parents: list | None = None
    orig_count: int | None = None
    # Source triangles (rows), used for the in-plane rotation (curl) of a triangle in the tie constraints.
    source_tris: np.ndarray | None = None

    def __post_init__(self):
        self.points = np.asarray(self.points, float)
        self.quads = np.asarray(self.quads, np.int64)
        self.elem_part = np.asarray(self.elem_part)
        if self.node_ids is None:
            self.node_ids = np.arange(1, len(self.points) + 1)
        if self.elem_ids is None:
            self.elem_ids = np.arange(1, len(self.quads) + 1)

    @property
    def n_nodes(self):
        return len(self.points)

    def element_normals(self):
        X = self.points
        q = self.quads
        n = np.cross(X[q[:, 2]] - X[q[:, 0]], X[q[:, 3]] - X[q[:, 1]])
        area = 0.5 * np.linalg.norm(n, axis=1)
        return n / (2 * area)[:, None], area

    def node_thickness(self):
        t = np.full(self.n_nodes, np.nan)
        for p, prop in self.parts.items():
            rows = np.unique(self.quads[self.elem_part == p])
            if np.any(np.isfinite(t[rows]) & (t[rows] != prop["thickness"])):
                raise ValueError("a node is shared by parts of different thickness; not supported")
            t[rows] = prop["thickness"]
        return t

    def nodal_normals(self, share_weld_normals=True):
        """Area-weighted average of adjacent element normals (each node only sees its own part's
        elements because parts do not share nodes).  Welded pairs get their common average so
        the weld rows are exact for finite rotation."""
        n_e, area = self.element_normals()
        nn = np.zeros_like(self.points)
        for k in range(4):
            np.add.at(nn, self.quads[:, k], n_e * area[:, None])
        if share_weld_normals:
            for a, b in self.welds:
                s = nn[a] + nn[b]
                nn[a] = s
                nn[b] = s
        return nn / np.linalg.norm(nn, axis=1)[:, None]

    @classmethod
    def from_shell_package(cls, pkg, triangles="error", refine=0):
        """Read a shell package (nodes/elements/panels.csv, C.mtx ties, clamps).

        ``triangles``: what to do with 3-node elements. "error" (default) refuses them; "split" divides every
        triangle into three quads (vertex, edge midpoints, centroid), keeping every source node at its row and
        appending the new nodes. One-layer thin-solid wedges lock in bending (a clamped plate under self-weight
        sags 72 % too little at a 5 mm element size), so triangles are not written as wedges.
        ``refine``: subdivide every quad into four (edge midpoints, centroid) this many times after the split.
        Ties in C.mtx that are coincident-node pairs become ``welds`` (exact for finite rotation, as before);
        any other tie rows are kept as ``shell_C`` in the 6-DOF shell layout and mapped to the thin-solid
        layout by :meth:`CcxModel.weld_rows`. Clamp patches grow with the refinement: a new node belongs to a
        patch when all nodes it interpolates do. Clamp centres are read from clamps.csv when present."""
        import scipy.io

        pkg = Path(pkg)
        rd = lambda f: list(csv.DictReader(open(pkg / f)))  # noqa: E731
        nodes = rd("nodes.csv")
        node_ids = np.array([int(r["node_id"]) for r in nodes])
        X = np.array([[float(r[k]) for k in ("x_m", "y_m", "z_m")] for r in nodes])
        row = {v: i for i, v in enumerate(node_ids.tolist())}
        els = rd("elements.csv")
        is_tri = np.array([(e.get("n4") or "").strip() == "" for e in els])
        if is_tri.any() and triangles != "split":
            raise ValueError("the package has triangles; pass triangles='split' (one-layer wedges lock in bending)")
        parts = {}
        for p in rd("panels.csv"):
            parts[p["panel_id"].strip()] = {
                "thickness": float(p["thickness_m"]),
                "E": float(p["youngs_modulus_pa"]),
                "nu": float(p["poisson_ratio"]),
                "rho": float(p["density_kg_m3"]),
            }
        n0 = len(X)
        pts = list(X)
        parents = [(i,) for i in range(n0)]
        edge_mid = {}

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in edge_mid:
                edge_mid[key] = len(pts)
                pts.append(0.5 * (pts[a] + pts[b]))
                parents.append(key)
            return edge_mid[key]

        quads, quad_part, src_tris, quad_src = [], [], [], []
        for k, e in enumerate(els):
            part = e["panel_id"].strip()
            if is_tri[k]:
                a, b, c = (row[int(e[f])] for f in ("n1", "n2", "n3"))
                src_tris.append((a, b, c))
                mab, mbc, mca = mid(a, b), mid(b, c), mid(c, a)
                g = len(pts)
                pts.append((X[a] + X[b] + X[c]) / 3.0)
                parents.append((a, b, c))
                quads += [(a, mab, g, mca), (b, mbc, g, mab), (c, mca, g, mbc)]
                quad_part += [part] * 3
                quad_src += [int(e["element_id"])] * 3
            else:
                quads.append(tuple(row[int(e[f])] for f in ("n1", "n2", "n3", "n4")))
                quad_part.append(part)
                quad_src.append(int(e["element_id"]))
        for _ in range(int(refine)):
            edge_mid = {}
            new_q, new_part, new_src = [], [], []
            for q, part, src in zip(quads, quad_part, quad_src, strict=True):
                m = [mid(q[i], q[(i + 1) % 4]) for i in range(4)]
                c = len(pts)
                pts.append(np.mean([pts[v] for v in q], axis=0))
                parents.append(tuple(q))
                new_q += [(q[0], m[0], c, m[3]), (m[0], q[1], m[1], c), (c, m[1], q[2], m[2]), (m[3], c, m[2], q[3])]
                new_part += [part] * 4
                new_src += [src] * 4
            quads, quad_part, quad_src = new_q, new_part, new_src
        P = np.array(pts)
        n_all = len(P)
        welds, shell_C = [], None
        if (pkg / "C.mtx").exists():
            C = sp.csr_matrix(scipy.io.mmread(pkg / "C.mtx"))
            coincident = True
            for r in range(C.shape[0]):
                cols = C.indices[C.indptr[r] : C.indptr[r + 1]]
                vals = C.data[C.indptr[r] : C.indptr[r + 1]]
                if len(cols) != 2 or cols[0] % 6 != cols[1] % 6 or abs(vals.sum()) > 1e-12 * abs(vals).max():
                    coincident = False
                    break
            if coincident:
                seen = set()
                for r in range(C.shape[0]):
                    cols = C.indices[C.indptr[r] : C.indptr[r + 1]]
                    vals = C.data[C.indptr[r] : C.indptr[r + 1]]
                    pair = (int(cols[np.argmax(vals)] // 6), int(cols[np.argmin(vals)] // 6))
                    if pair not in seen:
                        seen.add(pair)
                        welds.append(pair)
            else:  # general ties: re-index the columns to this mesh (source rows keep their index)
                Cc = C.tocoo()
                shell_C = sp.csr_matrix((Cc.data, (Cc.row, Cc.col)), shape=(C.shape[0], 6 * n_all))
        elif (pkg / "weld_pairs.csv").exists():
            welds = [(row[int(w["node_a"])], row[int(w["node_b"])]) for w in rd("weld_pairs.csv")]
        for a, b in welds:
            if np.linalg.norm(X[a] - X[b]) > 1e-9:
                raise ValueError("weld pairs must be coincident")

        # patch membership through the refinement
        def grow(rows):
            inside = np.zeros(n_all, bool)
            inside[list(rows)] = True
            for i in range(n0, n_all):
                inside[i] = all(inside[p] for p in parents[i])
            return np.nonzero(inside)[0]

        clamps = []
        if (pkg / "clamps.csv").exists():
            patch = {}
            for r in rd("clamp_nodes.csv"):
                patch.setdefault(r["clamp_id"].strip(), []).append(row[int(r["node_id"])])
            for c in rd("clamps.csv"):
                cid = c["clamp_id"].strip()
                rec = {
                    "clamp_id": cid,
                    "panel_id": c["panel_id"].strip(),
                    "nodes": grow(patch[cid]),
                    "source_nodes": np.array(sorted(patch[cid])),
                    "active": {k[7:]: int(v) for k, v in c.items() if k.startswith("active_")},
                }
                if all(k in c for k in ("centre_x_m", "centre_y_m", "centre_z_m")):
                    rec["centre"] = np.array([float(c[k]) for k in ("centre_x_m", "centre_y_m", "centre_z_m")])
                clamps.append(rec)
        new_ids = np.concatenate([node_ids, -np.arange(1, n_all - n0 + 1)])  # generated nodes: negative ids
        return cls(
            points=P,
            quads=np.array(quads, np.int64),
            elem_part=np.array(quad_part),
            parts=parts,
            welds=welds,
            node_ids=new_ids,
            elem_ids=np.array(quad_src),
            clamps=clamps,
            shell_C=shell_C,
            parents=parents,
            orig_count=n0,
            source_tris=np.array(src_tris, np.int64) if src_tris else None,
        )


def make_plate_mesh(nx=6, ny=3, lx=0.3, ly=0.1, t=0.002, E=2.1e11, nu=0.3, rho=7850.0, camber=0.0):
    """Small structured (optionally cambered) plate for tests."""
    xs, ys = np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1)
    P = np.array([[x, y, camber * (1 - (2 * x / lx - 1) ** 2)] for y in ys for x in xs])
    idx = lambda i, j: j * (nx + 1) + i  # noqa: E731
    Q = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for j in range(ny) for i in range(nx)])
    return ShellMesh(
        points=P,
        quads=Q,
        elem_part=np.array(["P"] * len(Q)),
        parts={"P": {"thickness": t, "E": E, "nu": nu, "rho": rho}},
    )


def rotvec_to_matrix(w):
    """Rodrigues: (..., 3) rotation vectors -> (..., 3, 3)."""
    w = np.asarray(w, float)
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    K = np.zeros((*w.shape[:-1], 3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(th > 1e-12, np.sin(th) / th, 1.0)
        b = np.where(th > 1e-12, (1 - np.cos(th)) / th**2, 0.5)
    return np.eye(3) + a * K + b * (K @ K)


# ----------------------------------------------------------------------------------------------
# .dat / matrix parsing
# ----------------------------------------------------------------------------------------------
_FLOAT = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"


def parse_dat_blocks(path):
    """Return {keyword: [ (time, ndarray rows) ... ]} for node/element print blocks in a .dat."""
    out = {}
    lines = Path(path).read_text().splitlines()
    i = 0
    hdr = re.compile(r"^\s*([a-z][a-z .]*?)\s*\((.*?)\)\s*for set\s+(\S+)\s+and time\s+(" + _FLOAT + ")")
    while i < len(lines):
        m = hdr.match(lines[i])
        if m:
            key = m.group(1).strip()
            t = float(m.group(4).replace("D", "E"))
            i += 1
            rows = []
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and lines[i].strip() and not hdr.match(lines[i]):
                try:
                    rows.append([float(v.replace("D", "E")) for v in lines[i].split()])
                except ValueError:
                    break
                i += 1
            out.setdefault(key, []).append((t, np.array(rows)))
            continue
        i += 1
    return out


def parse_dat_eigen(path):
    """Frequencies (Hz) of all *FREQUENCY steps in a .dat (EIGENVALUE OUTPUT table)."""
    lines = Path(path).read_text().splitlines()
    freqs = []
    for i, ln in enumerate(lines):
        if "E I G E N V A L U E   O U T P U T" in ln:
            j = i + 1
            while j < len(lines) and "MODE NO" not in lines[j]:
                j += 1
            j += 1
            while j < len(lines) and not (lines[j].split() and lines[j].split()[0].isdigit()):
                j += 1  # skip the unit header lines and blanks
            while j < len(lines) and lines[j].strip():
                v = lines[j].split()
                if not v[0].isdigit():
                    break
                freqs.append(float(v[3].replace("D", "E")))
                j += 1
    return np.array(freqs)


def read_coo_upper(path, n):
    """ccx MATRIXSTORAGE file: 'row col value' (1-based, one triangle incl. diagonal) -> symmetric CSR."""
    with open(path) as fh:
        a = np.array(fh.read().split(), dtype=float).reshape(-1, 3)  # ~3x faster than np.fromfile
    i = a[:, 0].astype(np.int64) - 1
    j = a[:, 1].astype(np.int64) - 1
    v = a[:, 2]
    off = i != j
    A = sp.coo_matrix((np.r_[v, v[off]], (np.r_[i, j[off]], np.r_[j, i[off]])), shape=(n, n))
    return A.tocsr()


def block_diag_inverse(A):
    """Inverse of a symmetric sparse matrix that is block diagonal up to a permutation
    (the element-internal incompatible-mode blocks), via its connected components."""
    from scipy.sparse.csgraph import connected_components

    A = sp.csr_matrix(A)
    nc, lab = connected_components(A, directed=False)
    order = np.argsort(lab, kind="stable")
    sizes = np.bincount(lab)
    if np.all(sizes == sizes[0]):
        m = sizes[0]
        idx = order.reshape(nc, m)  # global indices of each block
        pos = np.empty(A.shape[0], np.int64)
        pos[idx] = np.arange(m)[None, :]
        C = A.tocoo()
        blocks = np.zeros((nc, m, m))
        blocks[lab[C.row], pos[C.row], pos[C.col]] = C.data
        inv = np.linalg.inv(blocks)
        rows = np.repeat(idx, m, axis=1).reshape(-1)
        cols = np.tile(idx, (1, m)).reshape(-1)
        return sp.csr_matrix((inv.reshape(-1), (rows, cols)), shape=A.shape)
    return sp.csr_matrix(spla.inv(A.tocsc()))


def read_dof_labels(path):
    """ccx .dof: one line per equation, 'node.direction'."""
    lab = []
    for ln in Path(path).read_text().split():
        node, d = ln.split(".")
        lab.append((int(node), int(d)))
    return np.array(lab, dtype=np.int64)


# ----------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------
class CcxModel:
    """File-coupled CalculiX model of a shell mesh as one layer of C3D8I bricks (see module doc)."""

    def __init__(
        self, mesh: ShellMesh, workdir, element="C3D8I", ccx=None, nthreads=1, share_weld_normals=True, keep_files=False
    ):
        if element not in ("C3D8I", "S4"):
            raise ValueError(element)
        self.mesh = mesh
        self.element = element
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.ccx = str(ccx or CCX_DEFAULT)
        self.nthreads = int(nthreads)
        self.keep_files = keep_files
        self.normals = mesh.nodal_normals(share_weld_normals)
        self.thickness = mesh.node_thickness()
        N = mesh.n_nodes
        self.n_dof = 6 * N
        h = 0.5 * self.thickness[:, None] * self.normals
        self.solid_points = np.empty((2 * N, 3))
        self.solid_points[0::2] = mesh.points - h
        self.solid_points[1::2] = mesh.points + h
        self.calls = []  # (label, wall seconds)
        self._counter = 0
        self._Kbb_cache = None

    # ---------------- geometry / DOF helpers ----------------
    @property
    def rest_q(self):
        """Rest coordinates in the 6N layout."""
        return self.solid_points.reshape(-1)

    def mid_and_director(self, q_disp):
        """6N displacement -> (mid-surface displacement (N,3), director change (N,3))."""
        s = q_disp.reshape(-1, 2, 3)
        return 0.5 * (s[:, 0] + s[:, 1]), (s[:, 1] - s[:, 0]) / self.thickness[:, None]

    def from_midsurface(self, u_mid, rotvec=None, R=None):
        """Thin-solid displacement of a mid-surface field u_mid (N,3) with nodal finite rotations
        (rotation vectors or matrices) applied to the rest director (inextensible fibre)."""
        if R is None:
            R = rotvec_to_matrix(np.zeros_like(u_mid) if rotvec is None else rotvec)
        d = 0.5 * self.thickness[:, None] * self.normals
        dn = np.einsum("nij,nj->ni", R, d) - d
        out = np.empty((self.mesh.n_nodes, 2, 3))
        out[:, 0] = u_mid - dn
        out[:, 1] = u_mid + dn
        return out.reshape(-1)

    def weld_rows(self):
        """C (6 per weld, C q = 0): node b bottom/top expressed through node a (exact, see doc).

        With general package ties (``mesh.shell_C``) the rows come from :meth:`general_tie_rows` instead and
        :attr:`weld_dependents` names the dependent 6N DOF of every row."""
        if self.mesh.shell_C is not None:
            return self.general_tie_rows()[0]
        rows, cols, vals = [], [], []
        r = 0
        for a, b in self.mesh.welds:
            ta, tb = self.thickness[a], self.thickness[b]
            cp, cm = 0.5 + tb / (2 * ta), 0.5 - tb / (2 * ta)
            for layer, (ca_bot, ca_top) in ((0, (cp, cm)), (1, (cm, cp))):
                for d in range(3):
                    rows += [r, r, r]
                    cols += [6 * b + 3 * layer + d, 6 * a + d, 6 * a + 3 + d]
                    vals += [1.0, -ca_bot, -ca_top]
                    r += 1
        return sp.csr_matrix((vals, (rows, cols)), shape=(r, self.n_dof))

    # ---------------- general (offset) ties ----------------
    def _tri_curl(self, tri):
        """Coefficients g_i (3 nodes x 3) with theta_n = sum_i g_i . u_mid_i: in-plane rotation (half the curl of
        the linear in-plane displacement) of a source triangle about its face normal. Exact for rigid motion."""
        X = self.mesh.points[list(tri)]
        n = np.cross(X[1] - X[0], X[2] - X[0])
        n /= np.linalg.norm(n)
        e1 = (X[1] - X[0]) / np.linalg.norm(X[1] - X[0])
        e2 = np.cross(n, e1)
        xy = np.stack([(X - X[0]) @ e1, (X - X[0]) @ e2], axis=1)
        a2 = (xy[1, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1]) - (xy[2, 0] - xy[0, 0]) * (xy[1, 1] - xy[0, 1])
        g = np.zeros((3, 3))
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            dNdx = (xy[j, 1] - xy[k, 1]) / a2
            dNdy = (xy[k, 0] - xy[j, 0]) / a2
            g[i] = 0.5 * (dNdx * e2 - dNdy * e1)
        return g, n

    def general_tie_rows(self, cond_max=1e8):
        """Map the package's 6-DOF shell ties (``mesh.shell_C``) to the thin-solid 6N layout.

        Shell DOFs of node k in terms of its bottom/top solid displacements (n_k nodal normal, t_k thickness):
            u_k      = (bot_k + top_k) / 2
            theta_k  = theta_perp + n_k (w - theta_perp . n_T) / (n_k . n_T),   theta_perp = n_k x (top_k - bot_k) / t_k,
        w being the in-plane rotation (half curl) of the tie's source triangle about its face normal n_T (thin
        solids have no drilling DOF); the form is exact for rigid rotation. Per tie (one dependent node): its three translation rows are mapped as they are; its three rotation
        rows are projected onto the dependent node's tangent plane (two rows), because the dependent node has no
        drilling DOF and its thickness stretch stays free. Each tie gives five rows, normalised so that five
        chosen dependent DOFs form an identity block (the dependent node's own solid DOFs, or the master nodes'
        when the dependent lies in a clamp pad). Both mapped forms are exact for rigid translation and, to first
        order, rigid rotation. Returns (C (r x 6N) csr, dependents (r,)) and stores ``weld_dependents``."""
        import scipy.linalg

        if getattr(self, "_general_ties", None) is not None:
            return self._general_ties
        C = sp.csr_matrix(self.mesh.shell_C)
        N = self.mesh.n_nodes
        nrm, th = self.normals, self.thickness
        pad_nodes = set()
        for c in self.mesh.clamps:
            pad_nodes.update(int(v) for v in c["nodes"])
        tri_of = {}
        if self.mesh.source_tris is not None:
            for t_raw in self.mesh.source_tris:
                t = tuple(int(v) for v in t_raw)
                tri_of.setdefault(tuple(sorted(t)), t)
                for a, b in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2])):
                    tri_of.setdefault(tuple(sorted((a, b))), t)
        # group rows by dependent shell node (the +1 entry)
        groups = {}
        for r in range(C.shape[0]):
            cols = C.indices[C.indptr[r] : C.indptr[r + 1]]
            vals = C.data[C.indptr[r] : C.indptr[r + 1]]
            one = np.nonzero(vals == 1.0)[0]
            if len(one) != 1:
                raise ValueError(f"tie row {r} has no unique +1 dependent entry")
            j = int(cols[one[0]])
            groups.setdefault(j // 6, {})[j % 6] = (cols, vals)
        out_rows, deps, used, pad_groups = [], [], set(), []
        skew = lambda v: np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])  # noqa: E731
        for dep, comps in groups.items():
            if sorted(comps) != [0, 1, 2, 3, 4, 5]:
                raise ValueError(f"tie of node row {dep} does not have the 6 rows ux..rz")
            masters = sorted({int(c) // 6 for c in comps[0][0]} - {dep})
            tri = tri_of.get(tuple(masters))
            if tri is None:
                raise ValueError(f"no source triangle contains the masters {masters} of tie node row {dep}")
            g, nT = self._tri_curl(tri)

            def add(acc, col, v):
                if v != 0.0:
                    acc[col] = acc.get(col, 0.0) + v

            def expand(cols, vals, dep=dep, nT=nT, tri=tri, g=g):
                acc = {}
                for col, v in zip(cols, vals, strict=True):
                    k, a = int(col) // 6, int(col) % 6
                    if a < 3:
                        add(acc, 6 * k + a, 0.5 * v)
                        add(acc, 6 * k + 3 + a, 0.5 * v)
                        continue
                    a -= 3
                    S = skew(nrm[k])  # theta_perp = S(n) (top - bot) / t
                    for b in range(3):
                        add(acc, 6 * k + 3 + b, v * S[a, b] / th[k])
                        add(acc, 6 * k + b, -v * S[a, b] / th[k])
                    if k != dep:
                        # rotation about the node normal from the triangle's in-plane rotation (about its face
                        # normal nT): theta.n_k = (curl - theta_perp.nT) / (n_k.nT), exact for rigid rotation
                        cn = nrm[k][a] / float(nrm[k] @ nT)
                        for i, ti in enumerate(tri):
                            for b in range(3):
                                c = v * cn * g[i, b]
                                add(acc, 6 * ti + b, 0.5 * c)
                                add(acc, 6 * ti + 3 + b, 0.5 * c)
                        SnT = nT @ S  # theta_perp . nT = sum_b (nT^T S)_b (top_b - bot_b) / t
                        for b in range(3):
                            add(acc, 6 * k + 3 + b, -v * cn * SnT[b] / th[k])
                            add(acc, 6 * k + b, v * cn * SnT[b] / th[k])
                return acc

            trans = [expand(*comps[a]) for a in range(3)]
            rot = [expand(*comps[3 + a]) for a in range(3)]
            n = nrm[dep]
            e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            proj = []
            for e in (e1, e2):
                acc = {}
                for a in range(3):
                    for col, v in rot[a].items():
                        add(acc, col, e[a] * v)
                proj.append(acc)
            rows5 = trans + proj
            if dep in pad_nodes:
                pad_groups.append((dep, masters, rows5))
                continue
            allcols = sorted(set().union(*[r.keys() for r in rows5]))
            pos = {c: i for i, c in enumerate(allcols)}
            A = np.zeros((5, len(allcols)))
            for i, r in enumerate(rows5):
                for col, v in r.items():
                    A[i, pos[col]] = v
            cand = [c for c in (6 * dep + j for j in range(6)) if c in pos and c not in used]
            _, _, piv = scipy.linalg.qr(A[:, [pos[c] for c in cand]], pivoting=True)
            sel = [cand[i] for i in piv[:5]]
            B = A[:, [pos[c] for c in sel]]
            if np.linalg.cond(B) > cond_max:
                raise ValueError(f"tie of node row {dep}: dependent block is ill-conditioned ({np.linalg.cond(B):.3g})")
            A = np.linalg.solve(B, A)
            A[np.abs(A) < 1e-14 * np.abs(A).max()] = 0.0
            for i, c in enumerate(sel):
                A[:, pos[c]] = 0.0
                A[i, pos[c]] = 1.0
                used.add(c)
                deps.append(c)
                out_rows.append({allcols[j]: A[i, j] for j in np.nonzero(A[i])[0]})
        # Ties whose dependent node lies in a clamp pad (rigid): their dependents come from the master panel. Masters
        # are shared between neighbouring ties, so all such rows are made explicit TOGETHER (one pivoted QR), which
        # also leaves no chained dependents. Numerically redundant rows (relative pivot < 1e-10) are dropped and counted.
        self.tie_redundant_rows = 0
        if pad_groups:
            rows_p = [r for _d, _m, rs in pad_groups for r in rs]
            allcols = sorted(set().union(*[r.keys() for r in rows_p]))
            pos = {c: i for i, c in enumerate(allcols)}
            A = np.zeros((len(rows_p), len(allcols)))
            for i, r in enumerate(rows_p):
                for col, v in r.items():
                    A[i, pos[col]] = v
            cand = [c for c in allcols if (c // 6) not in pad_nodes and c not in used]
            Ac = A[:, [pos[c] for c in cand]]
            # row rank of the pad-tie rows on the candidate DOFs
            _, Rr, _ = scipy.linalg.qr(Ac.T, pivoting=True)
            d = np.abs(np.diag(Rr))
            rank = int(np.sum(d > 1e-10 * d.max()))
            dfull = np.abs(np.diag(scipy.linalg.qr(A.T, pivoting=True, mode="r")[0]))
            rank_full = int(np.sum(dfull > 1e-10 * dfull.max()))
            self.tie_pad_rank = {"rows": len(rows_p), "rank_non_pad_dofs": rank, "rank_all_dofs": rank_full}
            if rank < rank_full:
                raise ValueError(f"pad ties need pad DOFs as dependents (rank {rank} < {rank_full}); not supported")
            if rank < len(rows_p):  # truly redundant rows: keep a maximal independent subset
                _, _, prow = scipy.linalg.qr(Ac.T, pivoting=True)
                keep_rows = np.sort(prow[:rank])
                self.tie_redundant_rows = len(rows_p) - rank
                A, Ac = A[keep_rows], Ac[keep_rows]
            _, _, piv = scipy.linalg.qr(Ac, pivoting=True)
            sel = [cand[i] for i in piv[: A.shape[0]]]
            B = A[:, [pos[c] for c in sel]]
            if np.linalg.cond(B) > cond_max:
                raise ValueError(f"pad ties: dependent block is ill-conditioned ({np.linalg.cond(B):.3g})")
            A = np.linalg.solve(B, A)
            A[np.abs(A) < 1e-14 * np.abs(A).max()] = 0.0
            for i, c in enumerate(sel):
                A[:, pos[c]] = 0.0
                A[i, pos[c]] = 1.0
                used.add(c)
                deps.append(c)
                out_rows.append({allcols[j]: A[i, j] for j in np.nonzero(A[i])[0]})
            self.tie_pad_rows = int(A.shape[0])
        rr, cc, vv = [], [], []
        for i, r in enumerate(out_rows):
            for col, v in r.items():
                rr.append(i)
                cc.append(col)
                vv.append(v)
        Cs = sp.csr_matrix((vv, (rr, cc)), shape=(len(out_rows), 6 * N))
        self.weld_dependents = np.array(deps, np.int64)
        self._general_ties = (Cs, self.weld_dependents)
        return self._general_ties

    def clamp_dofs(self, clamp_ids):
        rows = np.concatenate([c["nodes"] for c in self.mesh.clamps if c["clamp_id"] in set(clamp_ids)])
        return (6 * np.unique(rows)[:, None] + np.arange(6)).reshape(-1)

    def gravity_load(self, M, g_vec):
        """Consistent gravity load f = M * (g repeated on every solid node)."""
        a = np.tile(np.asarray(g_vec, float), 2 * self.mesh.n_nodes)
        return M @ a

    # ---------------- input writing ----------------
    def _mesh_cards(self, welds):
        m = self.mesh
        L = []
        if self.element == "C3D8I":
            L.append("*NODE, NSET=NALL")
            L += [f"{k + 1}, {x:.13e}, {y:.13e}, {z:.13e}" for k, (x, y, z) in enumerate(self.solid_points)]
            for p in m.parts:
                L.append(f"*ELEMENT, TYPE=C3D8I, ELSET=E_{p}")
                for e in np.nonzero(m.elem_part == p)[0]:
                    q = m.quads[e]
                    bot, top = [2 * i + 1 for i in q], [2 * i + 2 for i in q]
                    L.append(f"{e + 1}, " + ", ".join(str(v) for v in bot + top))
        else:  # S4 with explicit normals so ccx's expansion equals ours
            L.append("*NODE, NSET=NALL")
            L += [f"{k + 1}, {x:.13e}, {y:.13e}, {z:.13e}" for k, (x, y, z) in enumerate(m.points)]
            for p in m.parts:
                L.append(f"*ELEMENT, TYPE=S4, ELSET=E_{p}")
                for e in np.nonzero(m.elem_part == p)[0]:
                    L.append(f"{e + 1}, " + ", ".join(str(i + 1) for i in m.quads[e]))
            L.append("*NORMAL")
            for e, q in enumerate(m.quads):
                for i in q:
                    nx, ny, nz = self.normals[i]
                    L.append(f"{e + 1}, {i + 1}, {nx:.13e}, {ny:.13e}, {nz:.13e}")
        L.append("*ELSET, ELSET=EALL")
        L += [f"E_{p}" for p in m.parts]
        for p, prop in m.parts.items():
            L += [
                f"*MATERIAL, NAME=M_{p}",
                "*ELASTIC",
                f"{prop['E']:.13e}, {prop['nu']:.13e}",
                "*DENSITY",
                f"{prop['rho']:.13e}",
            ]
            if self.element == "C3D8I":
                L.append(f"*SOLID SECTION, ELSET=E_{p}, MATERIAL=M_{p}")
            else:
                L += [f"*SHELL SECTION, ELSET=E_{p}, MATERIAL=M_{p}", f"{prop['thickness']:.13e}"]
        if welds and (m.welds or m.shell_C is not None):
            if self.element != "C3D8I":
                raise NotImplementedError("welds are only written for the thin-solid model")
            C = self.weld_rows().tocsr()
            dep = self.weld_dependents if m.shell_C is not None else None
            for r in range(C.shape[0]):
                cols = C.indices[C.indptr[r] : C.indptr[r + 1]]
                vals = C.data[C.indptr[r] : C.indptr[r + 1]]
                if dep is not None:
                    order = sorted(range(len(cols)), key=lambda k: 0 if cols[k] == dep[r] else 1)
                else:
                    order = sorted(
                        range(len(cols)), key=lambda k: 0 if (cols[k] // 6 == self.mesh.welds[r // 6][1]) else 1
                    )  # dependent first
                terms = []
                for k in order:
                    node = 2 * (cols[k] // 6) + (cols[k] % 6) // 3 + 1
                    terms.append(f"{node}, {cols[k] % 3 + 1}, {vals[k]:.13e}")
                # ccx reads at most 12 entries (4 terms) per line
                L += ["*EQUATION", str(len(terms))] + [", ".join(terms[i : i + 4]) for i in range(0, len(terms), 4)]
        return L

    def _run(self, label, lines, env_extra=None):
        self._counter += 1
        job = f"{label}_{self._counter:03d}"
        (self.workdir / f"{job}.inp").write_text("\n".join(lines) + "\n")
        env = dict(os.environ)
        env.update(
            OMP_NUM_THREADS=str(self.nthreads),
            CCX_NPROC_STIFFNESS=str(self.nthreads),
            CCX_NPROC_EQUATION_SOLVER=str(self.nthreads),
            CCX_NPROC_RESULTS=str(self.nthreads),
        )
        if env_extra:
            env.update(env_extra)
        t0 = time.perf_counter()
        p = subprocess.run([self.ccx, "-i", job], cwd=self.workdir, env=env, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        (self.workdir / f"{job}.log").write_text(p.stdout + p.stderr + f"\n# exit={p.returncode} wall={dt:.4f}s\n")
        self.calls.append((job, dt))
        if p.returncode != 0 or "*ERROR" in p.stdout or "Job finished" not in p.stdout:
            raise CcxError(f"ccx job {job} failed (exit {p.returncode}); see {self.workdir / (job + '.log')}")
        return job, dt

    def _cleanup(self, job, keep=(".inp", ".log", ".dat", ".sta")):
        if self.keep_files:
            return
        for f in self.workdir.glob(job + ".*"):
            if f.suffix not in keep:
                f.unlink()

    # ---------------- matrix export ----------------
    def _read_storage(self, job, need_mass=True):
        lab = read_dof_labels(self.workdir / f"{job}.dof")
        n = len(lab)
        K = read_coo_upper(self.workdir / f"{job}.sti", n)
        Mf = self.workdir / f"{job}.mas"
        M = read_coo_upper(Mf, n) if need_mass and Mf.exists() and Mf.stat().st_size > 0 else None
        N = self.mesh.n_nodes
        if self.element == "C3D8I":
            real = lab[:, 0] <= 2 * N
            target = 3 * (lab[real, 0] - 1) + lab[real, 1] - 1
        else:  # S4 expansion: each shell node label appears twice, (bottom, top) in that order
            real = lab[:, 0] <= N
            target = np.empty(real.sum(), np.int64)
            seen = {}
            for k, (node, d) in enumerate(lab[real]):
                c = seen.get((node, d), 0)
                seen[(node, d)] = c + 1
                target[k] = 6 * (node - 1) + 3 * c + d - 1
        return K, M, real, target, lab

    def _condense(self, K, M, real, target, mass="guyan"):
        """Static (bubble) condensation onto the real nodes, then permutation to the 6N layout."""
        r = np.nonzero(real)[0]
        b = np.nonzero(~real)[0]
        K = K.tocsr()
        Krr, Krb, Kbb = K[r][:, r], K[r][:, b], K[b][:, b]
        Xs = (block_diag_inverse(Kbb) @ Krb.T.tocsr()).tocsr()  # Kbb^-1 Kbr, sparse (element blocks)
        Kc = (Krr - Krb @ Xs).tocsr()
        P = sp.csr_matrix((np.ones(len(r)), (target, np.arange(len(r)))), shape=(self.n_dof, len(r)))
        Kc = (P @ Kc @ P.T).tocsr()
        Mc = None
        if M is not None:
            M = M.tocsr()
            Mrr = M[r][:, r]
            if mass == "guyan":
                Mrb, Mbb = M[r][:, b], M[b][:, b]
                Mrr = Mrr - Mrb @ Xs - (Mrb @ Xs).T + Xs.T @ (Mbb @ Xs)
            Mc = (P @ sp.csr_matrix(Mrr) @ P.T).tocsr()
        if len(r) != self.n_dof:
            raise CcxError(f"exported {len(r)} real DOFs, expected {self.n_dof} (constrained DOFs dropped?)")
        Kc = 0.5 * (Kc + Kc.T)
        return Kc, Mc

    def rest_matrices(self, mass="guyan"):
        """Free-free K, M (6N x 6N, untied; use weld_rows() for C).  Bubble DOFs condensed."""
        L = ["** rest matrices", *self._mesh_cards(welds=False)]
        L += ["*STEP", "*FREQUENCY, SOLVER=MATRIXSTORAGE", "*END STEP"]
        job, _dt = self._run("rest", L)
        K, M, real, target, _lab = self._read_storage(job)
        out = self._condense(K, M, real, target, mass)
        self._cleanup(job)
        return out

    def raw_rest_matrices(self):
        """Uncondensed export (K, M, labels) for inspection."""
        L = ["** raw rest matrices", *self._mesh_cards(welds=False)]
        L += ["*STEP", "*FREQUENCY, SOLVER=MATRIXSTORAGE", "*END STEP"]
        job, _ = self._run("raw", L)
        K, M, real, target, lab = self._read_storage(job)
        return K, M, lab, real, target

    def frequencies(self, n=12, welds=True, f_low_hz=0.5):
        """ccx's own ARPACK eigen-solution (bubble DOFs kept, consistent mass)."""
        L = ["** frequencies", *self._mesh_cards(welds=welds)]
        # ARPACK shift-invert finds the n eigenvalues nearest the shift; only those above f_low are
        # printed, so ask for 6 more to cover the rigid-body modes of a free-free model.
        L += ["*STEP", "*FREQUENCY", f"{n + 6}, {f_low_hz}", "*END STEP"]
        job, _dt = self._run("freq", L)
        return parse_dat_eigen(self.workdir / f"{job}.dat")

    # ---------------- nonlinear state evaluation ----------------
    def evaluate(self, q_disp, tangent=True, strains=True, nlgeom=True):
        """Prescribe the 6N displacement field on every node (one NLGEOM increment, bubbles solved
        by ccx) and return dict(f_int, K_T, strain, wall).  f_int = reaction force RF.
        K_T comes from a PERTURBATION *FREQUENCY step with SOLVER=MATRIXSTORAGE about the
        prescribed state, all prescriptions released, bubbles condensed."""
        if self.element != "C3D8I":
            raise CcxError("a full state can only be prescribed on the thin-solid model")
        q = np.asarray(q_disp, float).reshape(-1, 3)
        L = ["** prescribed-state evaluation", *self._mesh_cards(welds=False)]
        L += ["*STEP" + (", NLGEOM" if nlgeom else ""), "*STATIC", "1., 1.", "*BOUNDARY"]
        L += [f"{k + 1}, {d + 1}, {d + 1}, {q[k, d]:.13e}" for k in range(len(q)) for d in range(3)]
        L += ["*NODE PRINT, NSET=NALL, TOTALS=NO", "RF"]
        if strains:
            L += ["*EL PRINT, ELSET=EALL", "E"]
        L += ["*END STEP"]
        if tangent:
            L += ["*STEP, PERTURBATION", "*FREQUENCY, SOLVER=MATRIXSTORAGE", "*BOUNDARY, OP=NEW", "*END STEP"]
        job, dt = self._run("eval", L)
        blocks = parse_dat_blocks(self.workdir / f"{job}.dat")
        rf = blocks["forces"][-1][1]
        f = np.zeros((2 * self.mesh.n_nodes, 3))
        f[rf[:, 0].astype(int) - 1] = rf[:, 1:4]
        out = {"f_int": f.reshape(-1), "wall": dt, "job": job}
        if strains and "strains" in blocks:
            e = blocks["strains"][-1][1]  # elem, ip, exx, eyy, ezz, exy, exz, eyz
            ne = len(self.mesh.quads)
            nip = int(e[:, 1].max())
            E = np.zeros((ne, nip, 6))
            E[e[:, 0].astype(int) - 1, e[:, 1].astype(int) - 1] = e[:, 2:8]
            out["strain"] = E
        if tangent:
            K, _, real, target, _lab = self._read_storage(job, need_mass=False)
            out["K_T"], _ = self._condense(K, None, real, target)
        self._cleanup(job)
        return out

    def static_nlgeom(self, fixed_dofs, g_vec=(0.0, 0.0, -9.81), welds=True, n_incr=10, extra_cards=()):
        """Nonlinear static solution under gravity with the listed 6N DOFs held at zero.
        Returns dict(u (6N), wall)."""
        fixed = np.unique(np.asarray(fixed_dofs, np.int64))
        L = ["** nonlinear static", *self._mesh_cards(welds=welds)]
        g = np.asarray(g_vec, float)
        gn = np.linalg.norm(g)
        L += ["*BOUNDARY"]
        L += [f"{2 * (k // 6) + (k % 6) // 3 + 1}, {k % 3 + 1}, {k % 3 + 1}" for k in fixed]
        L += [
            "*STEP, NLGEOM, INC=1000",
            "*STATIC",
            f"{1.0 / n_incr}, 1., 1e-6, {1.0 / n_incr}",
            "*DLOAD",
            f"EALL, GRAV, {gn:.13e}, {g[0] / gn:.13e}, {g[1] / gn:.13e}, {g[2] / gn:.13e}",
        ]
        L += list(extra_cards)
        L += ["*NODE PRINT, NSET=NALL, TOTALS=NO", "U", "*END STEP"]
        job, dt = self._run("static", L)
        blocks = parse_dat_blocks(self.workdir / f"{job}.dat")
        U = blocks["displacements"][-1][1]
        u = np.zeros((2 * self.mesh.n_nodes, 3))
        u[U[:, 0].astype(int) - 1] = U[:, 1:4]
        self._cleanup(job)
        return {"u": u.reshape(-1), "wall": dt, "job": job, "time": blocks["displacements"][-1][0]}
