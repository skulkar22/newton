# ruff: noqa: RUF059
"""Strain-level accuracy comparison: full FE vs FFRF ROM vs re-ROM (v2 WP2).

Shape and frame convention (shared by tets, CalculiX shells/solids, and every comparison)
------------------------------------------------------------------------------------------
A strain history is a :class:`StrainSeries`:

- ``times``     (T,)          simulated time [s]
- ``tensor``    (T, P, 3, 3)  symmetric strain tensors with TENSORIAL shear components
                              (eps_xy, not gamma_xy = 2 eps_xy), dimensionless.
                              P = number of "points": tet elements for our own fields; for CalculiX
                              results P may be elements, integration points or nodes (frd nodal
                              extrapolation), but a reference and a variant must use the same P set
                              in the same order.
- ``measure``   str           "green_lagrange" (default for all comparisons), "small", "biot", "ccx_E"...
- ``centroids`` (P, 3)        REST (undeformed) location of each point, used for probe selection.
- ``weights``   (P,)          optional quadrature weights (element rest volumes) for weighted RMS.

Frame: components are in the case's body/clamp frame axes (x along the span, z up). The
Green-Lagrange strain E = (F^T F - I)/2 is a material tensor: it is invariant under rigid motion
of the deformed field, so it does not depend on whether the snapshot is stored in the world or
the body frame. The small-strain tensor eps = sym(F) - I is NOT rotation invariant and is
meaningless at large rotation (an 80 deg rigid rotation produces |eps| ~ 0.8); it is provided for
small-deflection checks only. The Biot strain U - I (polar decomposition F = R U) is objective and
equals eps to first order.

CalculiX: ``*EL FILE, E`` (frd, NLGEOM: Lagrange strain) gives six components per point. Use
:func:`tensor_from_voigt` with the component order and shear convention of the file; the frd
convention (XX YY ZZ XY YZ ZX, tensorial shear) is the default, but WP4 must confirm both against a
ccx uniform-shear patch test before use (the .dat ``*EL PRINT`` order may differ from the frd order;
not checked here).

Derived scalars (per time and point): von Mises equivalent strain
eps_eq = sqrt(2/3 dev(e):dev(e)), maximum principal strain, Frobenius norm.

Comparison metrics (:func:`compare_strain`) normalise every error by the reference's maximum of the
same quantity over all times and points, so that "0.05" means "5 % of the reference peak strain".

This module holds the library part only; the plotting, GIF and summary-table helpers stay in the
frozen experiment package (``exp-re-rom/adaptive_rom/strain.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

I3 = np.eye(3)

# ------------------------------------------------------------------------------------------
# Kinematics and strain measures
# ------------------------------------------------------------------------------------------


class TetKinematics:
    """Linear-tet deformation gradients relative to a rest mesh (vectorised over time)."""

    def __init__(self, rest_nodes, tets):
        self.rest = np.asarray(rest_nodes, dtype=np.float64)
        self.tets = np.asarray(tets, dtype=np.int64)
        p = self.rest[self.tets]
        Dm = np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
        self.Dm_inv = np.linalg.inv(Dm)  # (E,3,3)
        self.volume = np.abs(np.linalg.det(Dm)) / 6.0
        self.centroids = p.mean(axis=1)

    @property
    def n_elements(self) -> int:
        return self.tets.shape[0]

    def deformation_gradient(self, X):
        """X (N,3) or (T,N,3) current positions -> F (E,3,3) or (T,E,3,3)."""
        X = np.asarray(X, dtype=np.float64)
        q = X[..., self.tets, :]  # (...,E,4,3)
        Ds = np.stack((q[..., 1, :] - q[..., 0, :], q[..., 2, :] - q[..., 0, :], q[..., 3, :] - q[..., 0, :]), axis=-1)
        return Ds @ self.Dm_inv


def green_lagrange(F):
    """E = (F^T F - I)/2, shape (...,3,3)."""
    return 0.5 * (np.swapaxes(F, -1, -2) @ F - I3)


def small_strain(F):
    """eps = (F + F^T)/2 - I (symmetric displacement gradient; not objective)."""
    return 0.5 * (F + np.swapaxes(F, -1, -2)) - I3


def biot_strain(F):
    """U - I with F = R U (polar decomposition by SVD; objective, = eps to first order)."""
    U_, s, Vt = np.linalg.svd(F)
    return np.einsum("...ki,...k,...kj->...ij", Vt, s, Vt) - I3


def von_mises_strain(e):
    """Equivalent strain sqrt(2/3 dev(e):dev(e)) (= |e_xx| for incompressible uniaxial strain)."""
    tr = np.trace(e, axis1=-2, axis2=-1)
    dev = e - (tr / 3.0)[..., None, None] * I3
    return np.sqrt(2.0 / 3.0 * np.einsum("...ij,...ij->...", dev, dev))


def principal_strains(e):
    """Eigenvalues ascending, shape (...,3)."""
    return np.linalg.eigvalsh(0.5 * (e + np.swapaxes(e, -1, -2)))


def max_principal_strain(e):
    return principal_strains(e)[..., -1]


def frobenius(e):
    return np.sqrt(np.einsum("...ij,...ij->...", e, e))


SCALARS = {
    "von_mises": von_mises_strain,
    "max_principal": max_principal_strain,
    "frobenius": frobenius,
}
_COMP = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "yz": (1, 2), "zx": (2, 0), "xz": (0, 2)}


def scalar_of(e, name):
    """Scalar strain measure by name: von_mises | max_principal | frobenius | xx | yy | ... (component)."""
    if name in SCALARS:
        return SCALARS[name](e)
    i, j = _COMP[name]
    return e[..., i, j]


def tensor_from_voigt(v, order=("xx", "yy", "zz", "xy", "yz", "zx"), engineering_shear=False):
    """(...,6) six-component strain -> (...,3,3) symmetric tensor (tensorial shear)."""
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros((*v.shape[:-1], 3, 3))
    for k, name in enumerate(order):
        i, j = _COMP[name]
        val = v[..., k] * (0.5 if (engineering_shear and i != j) else 1.0)
        out[..., i, j] = val
        out[..., j, i] = val
    return out


@dataclass
class StrainSeries:
    times: np.ndarray  # (T,)
    tensor: np.ndarray  # (T,P,3,3)
    measure: str = "green_lagrange"
    centroids: np.ndarray | None = None  # (P,3) rest locations
    weights: np.ndarray | None = None  # (P,)
    label: str = ""
    meta: dict = field(default_factory=dict)

    def scalar(self, name="von_mises"):
        return scalar_of(self.tensor, name)

    def subset(self, mask):
        """Restrict to points where mask (P,) is True."""
        return StrainSeries(
            self.times,
            self.tensor[:, mask],
            self.measure,
            None if self.centroids is None else self.centroids[mask],
            None if self.weights is None else self.weights[mask],
            self.label,
            dict(self.meta),
        )


_MEASURES = {"green_lagrange": green_lagrange, "small": small_strain, "biot": biot_strain}


def strain_series(kin: TetKinematics, X_series, times, measure="green_lagrange", label="", chunk=16):
    """Snapshot series (T,N,3) -> StrainSeries with tensor (T,E,3,3). Vectorised, chunked over T."""
    X_series = np.asarray(X_series)
    fn = _MEASURES[measure]
    out = np.empty((X_series.shape[0], kin.n_elements, 3, 3))
    for a in range(0, X_series.shape[0], chunk):
        out[a : a + chunk] = fn(kin.deformation_gradient(X_series[a : a + chunk]))
    return StrainSeries(np.asarray(times, dtype=np.float64), out, measure, kin.centroids, kin.volume, label)


# ------------------------------------------------------------------------------------------
# Comparison
# ------------------------------------------------------------------------------------------


def _align(ref: StrainSeries, var: StrainSeries, time_tol=1e-9, interpolate=False):
    if ref.tensor.shape[1:] != var.tensor.shape[1:]:
        raise ValueError(f"point sets differ: {ref.tensor.shape[1:]} vs {var.tensor.shape[1:]}")
    if var.times.shape == ref.times.shape and np.max(np.abs(var.times - ref.times)) <= time_tol:
        return var.tensor
    if not interpolate:
        raise ValueError("time grids differ; pass interpolate=True to interpolate the variant linearly")
    T = var.tensor.reshape(var.tensor.shape[0], -1)
    out = np.stack([np.interp(ref.times, var.times, T[:, k], left=np.nan, right=np.nan) for k in range(T.shape[1])], 1)
    return out.reshape(ref.tensor.shape)


def _wrms(a, w, axis=-1):
    """Weighted RMS over the point axis (NaN-safe)."""
    if w is None:
        return np.sqrt(np.nanmean(a * a, axis=axis))
    w = np.asarray(w, dtype=np.float64)
    return np.sqrt(np.nansum(a * a * w, axis=axis) / np.sum(w))


@dataclass
class StrainComparison:
    times: np.ndarray
    scalar_name: str
    s_ref: np.ndarray  # (T,P)
    s_var: np.ndarray  # (T,P)
    err: np.ndarray  # (T,P) signed scalar error / scalar norm
    terr: np.ndarray  # (T,P) ||E_var - E_ref||_F / tensor norm
    scalar_norm: float
    tensor_norm: float
    curves: dict
    metrics: dict

    def error_map(self, ti, kind="scalar_abs"):
        return np.abs(self.err[ti]) if kind == "scalar_abs" else self.terr[ti]


def compare_strain(
    ref: StrainSeries,
    var: StrainSeries,
    *,
    scalar="von_mises",
    use_weights=True,
    time_tol=1e-9,
    interpolate=False,
    select_times=None,
    quantiles=(0.5, 0.95, 0.99),
    active_frac=0.1,
):
    """Element-wise strain error of ``var`` against ``ref`` (same point set, same or interpolated times).

    Normalisation: scalar errors by max_t,p |s_ref|; tensor errors by max_t,p ||E_ref||_F.
    RMS = sqrt(mean over times of the (volume-)weighted mean over points of err^2).
    ``select_times``: {name: time index} for which per-time scalars are also reported.
    ``rms_err_active``: the same RMS restricted to the "active" points whose reference peak scalar is at least
    ``active_frac`` x the global reference max (the strained region; still normalised by the global max).
    """
    Ev = _align(ref, var, time_tol, interpolate)
    Er = ref.tensor
    w = ref.weights if use_weights else None
    s_ref, s_var = scalar_of(Er, scalar), scalar_of(Ev, scalar)
    sn = float(np.nanmax(np.abs(s_ref)))
    tn = float(np.nanmax(frobenius(Er)))
    sn = sn if sn > 0 else 1.0
    tn = tn if tn > 0 else 1.0
    err = (s_var - s_ref) / sn
    terr = frobenius(Ev - Er) / tn
    ae = np.abs(err)
    rms_t, max_t = _wrms(err, w), np.nanmax(ae, axis=1)
    trms_t, tmax_t = _wrms(terr, w), np.nanmax(terr, axis=1)
    peak_ref_t, peak_var_t = np.nanmax(s_ref, axis=1), np.nanmax(s_var, axis=1)
    curves = {
        "rms": rms_t,
        "max": max_t,
        "tensor_rms": trms_t,
        "tensor_max": tmax_t,
        "peak_ref": peak_ref_t,
        "peak_var": peak_var_t,
    }
    i_pk = int(np.nanargmax(peak_ref_t))
    # "active" points: reference peak (over time) of the scalar >= active_frac x global reference max
    act = np.nanmax(np.abs(s_ref), axis=0) >= active_frac * sn
    wa = None if w is None else np.asarray(w)[act]
    m = {
        "scalar": scalar,
        "measure": ref.measure,
        "scalar_norm_ref_max": sn,
        "tensor_norm_ref_max": tn,
        "rms_err": float(np.sqrt(np.nanmean(rms_t**2))),
        "max_err": float(np.nanmax(ae)),
        "tensor_rms_err": float(np.sqrt(np.nanmean(trms_t**2))),
        "tensor_max_err": float(np.nanmax(terr)),
        "max_err_time": float(ref.times[int(np.nanargmax(max_t))]),
        "active_frac": active_frac,
        "active_points": int(act.sum()),
        "rms_err_active": float(np.sqrt(np.nanmean(_wrms(err[:, act], wa) ** 2))) if act.any() else None,
        "peak_strain_ref": float(np.nanmax(s_ref)),
        "peak_strain_var": float(np.nanmax(s_var)),
        "peak_strain_ratio": float(np.nanmax(s_var) / np.nanmax(s_ref)) if np.nanmax(s_ref) > 0 else None,
        "peak_strain_ratio_at_ref_peak_time": float(peak_var_t[i_pk] / peak_ref_t[i_pk])
        if peak_ref_t[i_pk] > 0
        else None,
        "ref_peak_time": float(ref.times[i_pk]),
        "points": int(Er.shape[1]),
        "times": int(Er.shape[0]),
    }
    nz = ae[1:] if ae.shape[0] > 1 else ae
    for q in quantiles:
        m[f"abs_err_q{int(round(100 * q))}"] = float(np.nanquantile(nz, q))
    for name, ti in (select_times or {}).items():
        m[f"rms_err_at_{name}"] = float(rms_t[ti])
        m[f"max_err_at_{name}"] = float(max_t[ti])
        m[f"tensor_rms_err_at_{name}"] = float(trms_t[ti])
        m[f"time_{name}"] = float(ref.times[ti])
    return StrainComparison(ref.times, scalar, s_ref, s_var, err, terr, sn, tn, curves, m)


def probe_points(centroids, points: dict):
    """{name: rest point (3,)} -> {name: index of nearest centroid}."""
    c = np.asarray(centroids)
    return {k: int(np.argmin(np.linalg.norm(c - np.asarray(p)[None], axis=1))) for k, p in points.items()}


def probe_metrics(ref: StrainSeries, var: StrainSeries, probes: dict, component="xx", interpolate=False):
    """Strain-time histories at probe points; abs max error normalised by the probe's reference peak."""
    Ev = _align(ref, var, interpolate=interpolate)
    out = {}
    for name, p in probes.items():
        r, v = scalar_of(ref.tensor[:, p], component), scalar_of(Ev[:, p], component)
        scale = float(np.nanmax(np.abs(r))) or 1.0
        out[name] = {
            "index": int(p),
            "ref_absmax": scale,
            "var_absmax": float(np.nanmax(np.abs(v))),
            "abs_max_err": float(np.nanmax(np.abs(v - r))),
            "abs_max_err_over_probe_peak": float(np.nanmax(np.abs(v - r)) / scale),
            "rms_err_over_probe_peak": float(np.sqrt(np.nanmean((v - r) ** 2)) / scale),
        }
    return out
