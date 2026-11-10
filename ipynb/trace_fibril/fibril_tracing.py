"""
fibril_tracing.py
=================

Improved pipeline for tracing dark chromospheric fibrils in broadband Hβ
(DKIST/VBI) after removing the photospheric contribution estimated from a
co-temporal G-band image.  It is a re-implementation of the processing chain
prototyped in ``bg_removal_test.ipynb`` with

* the notebook's photospheric removal (min–max normalisation, polynomial fit of
  Hβ versus G-band, subtraction), optionally **tile-wise** (``fit_tile``) and with
  a small blur of the G-band (``gband_blur``) so that bright points are matched;
* a much faster, vectorised version of every tracing stage (float32,
  threaded multi-scale Hessian, ``bincount``-based arc filtering, KD-tree
  endpoint linking, no masked arrays);
* quality improvements that can be switched on individually:

  - ``centerline="nms"``: ridge centre-lines from non-maximum suppression
    along the Hessian normal instead of threshold → opening → skeletonize;
  - **dark-ridge gate** and **bright-point exclusion zones** (``bp_exclude``)
    against the Hessian "shoulder" rings around bright points;
  - **two-sided darkness test** (``two_sided``): a fibril must be darker than
    both of its sides, which rejects edges and bright-point rims;
  - optional **co-oriented G-band lane filter** (``use_bg_filter``) that removes
    traces running *along* intergranular lanes;
  - fragments are linked *before* the length cut; tangents come from the last
    ``tangent_window`` pixels; bridges are cubic Hermite curves that must stay
    smoother than ``link_rmin`` and keep ridge response along their length;
  - **junction resolution** (``resolve_junctions``): skeleton components are
    split at junctions and arcs are re-joined pairwise by tangent continuity, so
    crossing or touching fibrils become separate polylines;
  - **kink splitting** (``kink_split``): a trace is split where its heading turns
    by more than ``kink_angle_deg`` within ``2*kink_window`` pixels;
  - **orientation-field trimming** (``field_trim``): a local dominant fibril
    direction is estimated (response-weighted, ``field_scale`` px) and traces are
    cut where they run more than ``field_max_misalign_deg`` off it for at least
    ``field_min_run`` px in a coherent neighbourhood; the same field vetoes
    bridges that cross it (``link_field_check``);
  - **per-pixel two-sided trimming** (``ts_trim``): runs of at least
    ``ts_trim_min_run`` px that are not darker than both sides are cut out.

* a ``bin`` factor: the inputs are block-averaged and every pixel-unit
  parameter is rescaled automatically.

Typical use
-----------
>>> from fibril_tracing import run_pipeline, load_destretched
>>> Hb, Gb = load_destretched()
>>> res = run_pipeline(Hb, Gb, bin=1, centerline="nms")
>>> res["polylines"]        # list of (N, 2) float arrays, one per fibril trace
>>> res["traces"]           # the same traces drawn as a 1-px bool image
>>> res["trace_table"]      # one row per trace (length, width, contrast, kinks…)
>>> res["timings"]          # per-stage wall-clock seconds

``v1_params()`` reproduces the first improved version (no tile fit, no
bright-point handling, straight-line bridges, no junction resolution) and
``notebook_like_params()`` the notebook's own algorithm.
"""
from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage import filters, measure
from skimage.draw import line as draw_line
from skimage.morphology import remove_small_objects, skeletonize, disk
from skan import Skeleton, summarize

HERE = os.path.dirname(os.path.abspath(__file__))
N_THREADS = max(1, min(6, (os.cpu_count() or 2) // 2))

__all__ = [
    "Params", "v1_params", "v2_params", "notebook_like_params", "orientation_field", "misalignment_along",
    "split_at_runs", "run_pipeline", "load_destretched", "bin_image",
    "remove_photosphere", "bright_point_zone", "hessian_ridge", "ridge_nms", "make_mask",
    "filter_skeleton_by_background", "filter_paths_cooriented", "prune_short_branches",
    "straight_branches", "link_endpoints", "resolve_junctions", "split_kinks", "polylines_to_image",
    "scale_polylines", "paths_from_skeleton", "trace_table", "trace_table_from_polylines",
    "trace_metrics", "kink_stats", "contrast_along", "coverage", "upsample_skeleton", "Timer",
]


# ----------------------------------------------------------------------------
# timing helper
# ----------------------------------------------------------------------------
class Timer:
    """Collects wall-clock durations of named stages."""

    def __init__(self, verbose=True):
        self.times = {}
        self.verbose = verbose

    @contextmanager
    def __call__(self, name):
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        self.times[name] = self.times.get(name, 0.0) + dt
        if self.verbose:
            print(f"  [{dt:7.2f} s] {name}", flush=True)

    def total(self):
        return sum(v for k, v in self.times.items() if not k.startswith(" "))


# ----------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------
@dataclass
class Params:
    """All tunable parameters, in *full-resolution* pixel units.

    ``Params.scaled(bin)`` returns a copy with every length divided by ``bin``
    and every area divided by ``bin**2`` (percentiles and ratios untouched).
    """
    # -- photospheric removal
    quiet_box: tuple = (0, 600, 0, 600)           # r0, r1, c0, c1 (global fit only)
    bp_box: tuple = (1500, 2000, 2500, 3000)      # (global fit only)
    bp_pct: tuple = (97.0, 98.0)
    bp_min_size: int = 30
    bp_open_radius: int = 2
    poly_deg: int = 2
    gband_blur: float = 2.0        # px; Gaussian blur of the G-band before the fit (0 = none)
    fit_tile: int = 512            # px; tile-wise fit with 50 % overlap (0 = notebook's single global fit)
    fit_subsample: int = 2
    # -- ridge response of the merged image
    sigmas: tuple = (2, 3, 4, 5, 6, 7)
    black_ridges: bool = True
    low_pct: float = 70.0
    high_pct: float = 90.0
    dark_gate: bool = True         # zero the response where the image is brighter than its local background
    dark_gate_delta: float = 0.5   # tolerance in units of the robust scatter of (image - background)
    bg_scale: float = 15.0         # half-width (px) of the local-background median filter
    # -- bright-point exclusion zones
    bp_exclude: bool = True
    bpx_pct: tuple = (98.0, 99.3)  # hysteresis percentiles on the normalised G-band intensity
    bpx_min_area: int = 20
    bpx_max_area: int = 1500
    bpx_dilate: int = 8            # px added around each compact bright feature
    bpx_use_merged: bool = False   # also exclude bright blobs of the merged image (used when no G-band)
    bpx_merged_k: float = 4.0      # blob threshold in robust sigmas above the local background
    # -- two-sided darkness test (per branch)
    two_sided: bool = True
    ts_k: float = 1.5              # sample at ±ts_k·sigma_best along the ridge normal
    ts_min_contrast: float = 1.0   # median of min(side contrasts) must exceed this × pixel noise
    ts_max_asym: float = 0.8       # median of |c+ − c−| / max(c+, c−) must stay below this
    # -- mask mode (notebook behaviour)
    min_size: int = 60
    open_radius: int = 2
    # -- nms mode
    nms_min_len: int = 10
    nms_tol: float = 0.98
    # -- G-band background filter
    use_bg_filter: bool = False
    bg_mode: str = "cooriented"    # "cooriented" (response and orientation along the arc) or "overlap" (notebook rule)
    bg_sigmas: tuple = (1, 2, 3, 4)
    bg_low_pct: float = 70.0
    bg_high_pct: float = 90.0
    bg_min_size: int = 60
    bg_open_radius: int = 2
    bg_dilation: int = 5
    bg_overlap_frac: float = 0.3
    bg_overlap_len: int = 5
    bg_angle_deg: float = 25.0     # cooriented mode: max angle between trace tangent and lane direction
    # -- skeleton analysis
    prune_len: float = 35.0
    min_branch_len: float = 10.0
    tortuosity_max: float = 1.2
    rmin: float = 30.0
    curvature_smooth: float = 10.0
    curvature_pct: float = 75.0
    curvature_min_len: float = 40.0
    keep_only_straight: bool = True
    link_before_length_cut: bool = True
    # -- endpoint linking
    link_gap: float = 50.0
    link_angle_deg: float = 30.0
    link_alpha: float = 0.0        # notebook used 1.0 (tolerance doubled for short gaps)
    link_passes: int = 3
    tangent_window: int = 10
    bridge_check: bool = True
    bridge_frac: float = 0.6
    link_rmin: float = 25.0        # minimum radius of curvature of the Hermite bridge (0 = straight lines, no test)
    link_pos_tol: float = 1.5      # px; lateral offset always allowed (pixel quantisation of the centre-line)
    link_offset_frac: float = 0.1  # lateral offset allowed in addition, as a fraction of the gap
    # -- junction resolution and kinks
    resolve_junctions: bool = True
    junction_max_turn_deg: float = 40.0
    junction_cluster_len: float = 2.5
    kink_split: bool = True
    kink_window: int = 15          # half-window (px) of the turning-angle measurement (round 2: 10)
    kink_angle_deg: float = 45.0   # (round 2: 60)
    # -- local fibril-orientation field (round 3)
    field_scale: float = 60.0      # px; Gaussian scale of the response-weighted orientation average
    field_min_coherence: float = 0.3
    field_max_misalign_deg: float = 35.0
    field_trim: bool = True        # split traces at runs misaligned with the local field
    field_min_run: float = 15.0    # px; shorter misaligned runs are tolerated
    link_field_check: bool = True  # reject bridges whose direction disagrees with a coherent field
    # -- per-pixel two-sided trimming of the final traces (round 3)
    ts_trim: bool = True
    ts_trim_min_run: float = 10.0  # px
    ts_trim_asym: float = 0.9      # (0.8 removes 6 % more length for +0.006 in the two-sided fraction)
    ts_trim_min_contrast: float = 1.0   # × pixel noise
    field_bin: int = 4             # the field is computed on a binned response (16× cheaper, < 0.1° difference)
    # -- final clean-up
    final_prune_len: float = 20.0
    final_min_len: float = 30.0
    grouped_dilation: int = 2
    grouped_min_size: int = 500

    _LENGTHS = ("sigmas", "bg_sigmas", "prune_len", "min_branch_len", "rmin", "bg_scale",
                "curvature_smooth", "curvature_min_len", "link_gap", "tangent_window", "final_prune_len",
                "final_min_len", "nms_min_len", "bg_dilation", "bg_overlap_len",
                "open_radius", "bp_open_radius", "bg_open_radius", "grouped_dilation",
                "gband_blur", "fit_tile", "bpx_dilate", "link_rmin", "kink_window", "junction_cluster_len",
                "link_pos_tol", "field_scale", "field_min_run", "ts_trim_min_run")
    _AREAS = ("min_size", "bp_min_size", "bg_min_size", "grouped_min_size", "bpx_min_area", "bpx_max_area")
    _BOXES = ("quiet_box", "bp_box")
    _INTS = ("open_radius", "bp_open_radius", "bg_open_radius", "bg_dilation", "grouped_dilation",
             "tangent_window", "bg_overlap_len", "nms_min_len", "fit_tile", "bpx_dilate", "kink_window")

    def scaled(self, bin: int) -> "Params":
        d = asdict(self)
        if bin == 1:
            return Params(**d)
        for k in self._LENGTHS:
            v = d[k]
            if isinstance(v, (tuple, list)):
                d[k] = tuple(sorted(set(max(1.0, s / bin) for s in v)))      # sigma >= 1 px
            elif k in self._INTS:
                d[k] = int(round(v / bin))
            else:
                d[k] = v / bin
        d["tangent_window"] = max(4, d["tangent_window"])
        d["kink_window"] = max(4, d["kink_window"])
        for k in self._AREAS:
            d[k] = max(1, int(round(d[k] / bin ** 2)))
        for k in self._BOXES:
            d[k] = tuple(int(x // bin) for x in d[k])
        return Params(**d)


def v2_params(**overrides) -> Params:
    """Round-2 defaults (no orientation-field trimming / bridging check, no
    per-pixel two-sided trimming, kink threshold 60° over 20 px)."""
    d = dict(field_trim=False, link_field_check=False, ts_trim=False, kink_window=10, kink_angle_deg=60.0)
    d.update(overrides)
    return Params(**d)


def v1_params(**overrides) -> Params:
    """First improved version (round 1): no tile fit, no bright-point
    handling, straight-line bridges with the notebook's gap-scaled angle
    tolerance, component-based traces."""
    d = dict(gband_blur=0.0, fit_tile=0, bp_exclude=False, two_sided=False, link_alpha=1.0,
             link_rmin=0.0, resolve_junctions=False, kink_split=False, bg_mode="overlap")
    d.update(overrides)
    return v2_params(**d)


def notebook_like_params(**overrides) -> Params:
    """Parameters that reproduce the notebook's *algorithm* inside the fast code."""
    d = dict(dark_gate=False, min_branch_len=20.0, rmin=1e9, curvature_min_len=1e9,
             link_before_length_cut=False, bridge_check=False, tangent_window=1,
             link_passes=2, final_min_len=20.0, final_prune_len=20.0, nms_tol=1.0)
    d.update(overrides)
    return v1_params(**d)


# ----------------------------------------------------------------------------
# I/O and binning
# ----------------------------------------------------------------------------
def load_destretched(cache=os.path.join(HERE, "cache", "destretched.npz"), dtype=np.float32):
    """Load the destretched Hβ / G-band pair produced by ``baseline_pipeline.py``."""
    d = np.load(cache)
    return d["Hbeta"].astype(dtype), d["Gband"].astype(dtype)


def bin_image(img, bin: int):
    """Block-average ``img`` by an integer factor (edges cropped)."""
    if bin == 1:
        return img
    r = (img.shape[0] // bin) * bin
    c = (img.shape[1] // bin) * bin
    return img[:r, :c].reshape(r // bin, bin, c // bin, bin).mean(axis=(1, 3), dtype=np.float32)


def _minmax(img):
    img = np.asarray(img, dtype=np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def _percentiles(a, q, step=4):
    sub = a[::step, ::step] if a.ndim == 2 else a[::step]
    return np.percentile(sub, q)


def _noise(img, step=3):
    """Robust pixel noise from horizontal neighbour differences."""
    d = np.diff(np.asarray(img, np.float32)[::step, ::step], axis=1)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


# ----------------------------------------------------------------------------
# photospheric removal
# ----------------------------------------------------------------------------
def remove_photosphere(Hb, Gb, P: Params):
    """Return merged = norm(Hβ) − P_deg(norm(G-band)).

    ``P.fit_tile == 0``: single global fit on the quiet-Sun box plus the G-band
    bright points inside ``bp_box`` (the notebook's recipe).  Otherwise the
    polynomial is fitted in overlapping tiles of ``fit_tile`` px (step
    ``fit_tile/2``) on all their pixels and the coefficient maps are
    interpolated bilinearly, so plage and quiet Sun get their own response.
    ``P.gband_blur`` blurs the G-band first (its bright points are sharper than
    their Hβ counterparts).
    """
    Gs = ndi.gaussian_filter(np.asarray(Gb, np.float32), P.gband_blur) if P.gband_blur > 0 else Gb
    Hn, Gn = _minmax(Hb), _minmax(Gs)
    bp = make_mask(Gn, P.bp_pct[0], P.bp_pct[1], P.bp_min_size, P.bp_open_radius)
    r0, r1, c0, c1 = P.bp_box
    q0, q1, q2, q3 = P.quiet_box
    bpc = bp[r0:r1, c0:c1]
    x = np.concatenate([Gn[q0:q1, q2:q3].ravel(), Gn[r0:r1, c0:c1][bpc]]).astype(np.float64)
    y = np.concatenate([Hn[q0:q1, q2:q3].ravel(), Hn[r0:r1, c0:c1][bpc]]).astype(np.float64)
    coef = np.polyfit(x, y, P.poly_deg) if x.size > P.poly_deg + 1 else np.zeros(P.poly_deg + 1)
    T = int(P.fit_tile or 0)
    if T <= 0 or T > min(Hn.shape):
        model = np.polyval(coef, Gn).astype(np.float32)
        return (Hn - model).astype(np.float32), coef
    step = max(1, T // 2)
    s = max(1, int(P.fit_subsample))
    rs = np.arange(0, Hn.shape[0] - T + 1, step)
    cs = np.arange(0, Hn.shape[1] - T + 1, step)
    coefs = np.zeros((len(rs), len(cs), P.poly_deg + 1))
    for i, rr in enumerate(rs):
        for j, cc in enumerate(cs):
            gx = Gn[rr:rr + T, cc:cc + T][::s, ::s].ravel().astype(np.float64)
            hy = Hn[rr:rr + T, cc:cc + T][::s, ::s].ravel().astype(np.float64)
            coefs[i, j] = np.polyfit(gx, hy, P.poly_deg)
    yy, xx = np.mgrid[:Hn.shape[0], :Hn.shape[1]]
    fi = np.clip((yy - T / 2) / step, 0, len(rs) - 1).astype(np.float32)
    fj = np.clip((xx - T / 2) / step, 0, len(cs) - 1).astype(np.float32)
    model = np.zeros(Hn.shape, np.float32)
    for k in range(P.poly_deg + 1):
        ck = ndi.map_coordinates(coefs[:, :, k], [fi, fj], order=1, mode="nearest").astype(np.float32)
        model += ck * Gn ** (P.poly_deg - k)
    return (Hn - model).astype(np.float32), coef


# ----------------------------------------------------------------------------
# multi-scale Hessian ridge response (identical to skimage.filters.sato in 2-D)
# ----------------------------------------------------------------------------
def _hessian(img, sigma, truncate=8.0):
    """Hessian from Gaussian derivatives, exactly as
    ``skimage.feature.hessian_matrix(use_gaussian_derivatives=True)`` (two
    successive first-derivative passes with sigma/sqrt(2)), except that the
    kernel truncation is always ``truncate`` sigmas (skimage uses 100 sigmas
    whenever any sigma <= 1, which is 10-20x slower for the same output)."""
    s = sigma / np.sqrt(2.0)
    g = lambda a, order: ndi.gaussian_filter(a, sigma=(s, s), order=order, mode="reflect", truncate=truncate)
    gr = g(img, (1, 0))
    gc = g(img, (0, 1))
    return g(gr, (1, 0)), g(gr, (0, 1)), g(gc, (0, 1))


def _ridge_one_scale(img, sigma, want_dir):
    Hrr, Hrc, Hcc = _hessian(img, sigma)
    half = 0.5 * (Hrr + Hcc)
    rad = np.sqrt(np.square(0.5 * (Hrr - Hcc)) + np.square(Hrc))
    l1 = half + rad
    resp = (sigma ** 2) * np.maximum(l1, 0)
    if not want_dir:
        return resp, None, None
    nr = np.where(np.abs(Hrc) > 1e-12, Hrc, l1 - Hcc)
    nc = np.where(np.abs(Hrc) > 1e-12, l1 - Hrr, Hrc)
    norm = np.hypot(nr, nc)
    norm[norm == 0] = 1.0
    return resp, (nr / norm).astype(np.float32), (nc / norm).astype(np.float32)


def hessian_ridge(img, sigmas, black_ridges=True, want_dir=True, n_threads=N_THREADS):
    """Multi-scale Hessian ridge response: ``resp`` (max over scales of
    sigma²·max(λ₁,0), identical to ``skimage.filters.sato``), ``sigma_best``
    and the unit ridge normal ``(nr, nc)`` at that scale."""
    img = np.asarray(img, dtype=np.float32)
    if not black_ridges:
        img = -img
    sigmas = list(sigmas)
    resp = np.zeros(img.shape, np.float32)
    sbest = np.zeros(img.shape, np.float32)
    nr = np.zeros(img.shape, np.float32) if want_dir else None
    nc = np.zeros(img.shape, np.float32) if want_dir else None

    def work(s):
        return s, _ridge_one_scale(img, s, want_dir)

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for s, (r, dr, dc) in ex.map(work, sigmas):
            upd = r > resp
            resp[upd] = r[upd]
            sbest[upd] = s
            if want_dir:
                nr[upd] = dr[upd]
                nc[upd] = dc[upd]
    return resp, sbest, nr, nc


# ----------------------------------------------------------------------------
# local background, gates and exclusion zones
# ----------------------------------------------------------------------------
def local_background(img, scale=15, bin=4):
    """Median-filtered background (half-width ``scale`` px) computed on a
    ``bin``-binned copy and interpolated back."""
    small = bin_image(np.asarray(img, np.float32), bin)
    size = max(3, int(round((2 * scale + 1) / bin)) | 1)
    bg_small = ndi.median_filter(small, size=size, mode="nearest")
    bg = ndi.zoom(bg_small, bin, order=1, mode="nearest")
    out = np.empty(img.shape, np.float32)
    r, c = min(bg.shape[0], img.shape[0]), min(bg.shape[1], img.shape[1])
    out[:r, :c] = bg[:r, :c]
    out[r:, :] = out[r - 1:r, :]
    out[:, c:] = out[:, c - 1:c]
    return out


_K8 = np.ones((3, 3), np.uint8)


def _compact_blobs(mask, min_area, max_area):
    lbl, n = ndi.label(mask, structure=_K8)
    if n == 0:
        return mask & False
    area = np.bincount(lbl.ravel(), minlength=n + 1)
    keep = (area >= min_area) & (area <= max_area)
    keep[0] = False
    return keep[lbl]


def bright_point_zone(Gb, merged, P: Params):
    """Exclusion zone around compact bright features: hysteresis on the
    normalised G-band intensity (``bpx_pct``), area in [``bpx_min_area``,
    ``bpx_max_area``], optionally bright blobs of the merged image, dilated by
    ``bpx_dilate`` px.  Fibril traces end at these zones instead of wrapping
    around the Hessian shoulder of the bright point."""
    zone = np.zeros(merged.shape, bool)
    if Gb is not None and np.ptp(Gb) > 0:
        Gn = _minmax(Gb)
        lo, hi = _percentiles(Gn, P.bpx_pct)
        zone |= _compact_blobs(filters.apply_hysteresis_threshold(Gn, lo, hi), P.bpx_min_area, P.bpx_max_area)
    if P.bpx_use_merged:
        dev = merged - local_background(merged, P.bg_scale)
        mad = 1.4826 * np.median(np.abs(dev[::4, ::4] - np.median(dev[::4, ::4])))
        zone |= _compact_blobs(dev > P.bpx_merged_k * mad, P.bpx_min_area, P.bpx_max_area)
    if P.bpx_dilate > 0 and zone.any():
        zone = ndi.binary_dilation(zone, structure=disk(int(P.bpx_dilate)))
    return zone


# ----------------------------------------------------------------------------
# centre-line extraction
# ----------------------------------------------------------------------------
def make_mask(enhanced, low_pct=70, high_pct=92, min_size=50, open_radius=2):
    """Notebook's hysteresis mask (plain ndarrays, sub-sampled percentiles)."""
    enhanced = np.asarray(enhanced)
    lo, hi = _percentiles(enhanced, (low_pct, high_pct))
    mask = filters.apply_hysteresis_threshold(enhanced, lo, hi)
    mask = remove_small_objects(mask, min_size=min_size)
    if open_radius > 0:
        mask = ndi.binary_opening(mask, structure=disk(open_radius))
    return mask


def ridge_nms(resp, nr, nc, low, high, min_len=10, tol=0.98):
    """Thin ridge centre-lines: pixels whose response is (within a factor
    ``tol``) a local maximum along the ridge normal (±1 px, bilinear), then
    hysteresis (``low``/``high``) on the thin map, then 1-px thinning."""
    cand = resp >= low
    rr, cc = np.nonzero(cand)
    v0 = resp[rr, cc]
    dr, dc = nr[rr, cc], nc[rr, cc]
    vp = ndi.map_coordinates(resp, [rr + dr, cc + dc], order=1, mode="nearest")
    vm = ndi.map_coordinates(resp, [rr - dr, cc - dc], order=1, mode="nearest")
    keep = (v0 >= tol * vp) & (v0 >= tol * vm)
    thin = np.zeros(resp.shape, bool)
    thin[rr[keep], cc[keep]] = True
    thin_resp = np.where(thin, resp, 0.0).astype(np.float32)
    thin = filters.apply_hysteresis_threshold(thin_resp, low, high)
    thin = skeletonize(thin)
    thin = remove_small_objects(thin, min_size=max(2, int(min_len)), connectivity=2)
    return thin


# ----------------------------------------------------------------------------
# skeleton utilities
# ----------------------------------------------------------------------------
def _neighbour_count(skel):
    return ndi.convolve(skel.astype(np.uint8), _K8, mode="constant", cval=0) - skel.astype(np.uint8)


def endpoints(skel):
    return skel & (_neighbour_count(skel) == 1)


def junctions(skel):
    return skel & (_neighbour_count(skel) >= 3)


def _skan(skel):
    sk = Skeleton(skel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = summarize(sk, separator="_")
    return sk, tab


def paths_from_skeleton(skel):
    """Ordered coordinate arrays of all skan paths of a bool skeleton."""
    if not skel.any():
        return []
    sk, _ = _skan(skel)
    return [sk.path_coordinates(i).astype(float) for i in range(sk.n_paths)]


def _smooth_path(coords, sigma):
    if sigma <= 0 or coords.shape[0] < 5:
        return coords
    return np.column_stack([ndi.gaussian_filter1d(coords[:, i].astype(float), sigma, mode="nearest")
                            for i in range(2)])


def _curvature(coords):
    if coords.shape[0] < 5:
        return np.zeros(0)
    d1 = np.gradient(coords, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5 + 1e-12
    k = num / den
    return k[2:-2] if k.size > 4 else k


def _unit_tangents(coords, smooth=3.0):
    sm = _smooth_path(coords, smooth)
    t = np.gradient(sm, axis=0)
    n = np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)
    return t / n[:, None]


# ----------------------------------------------------------------------------
# G-band background filters
# ----------------------------------------------------------------------------
def filter_skeleton_by_background(skel, bg_skel, dilation_radius=5, min_overlap_fraction=0.3,
                                  min_overlap_length=5, mask=None):
    """Vectorised version of the notebook's ``filter_mask_by_background``
    (arcs overlapping the dilated background skeleton are removed)."""
    junc = junctions(skel)
    arcs = measure.label(skel & ~junc, connectivity=2)
    n = int(arcs.max())
    bg_dil = ndi.binary_dilation(bg_skel, structure=disk(dilation_radius))
    npx = np.bincount(arcs.ravel(), minlength=n + 1)
    nov = np.bincount(arcs[bg_dil].ravel(), minlength=n + 1)
    frac = nov / np.maximum(npx, 1)
    remove = (frac > min_overlap_fraction) & (nov > min_overlap_length)
    remove[0] = False
    keep_lut = ~remove
    keep_lut[0] = False
    skel_keep = keep_lut[arcs]
    skel_keep |= junc & ndi.binary_dilation(skel_keep, structure=disk(dilation_radius))
    info = dict(n_arcs=n, n_removed=int(remove.sum()), n_kept=int(n - remove.sum()))
    if mask is None:
        return skel_keep, None, info
    _, (ir, ic) = distance_transform_edt(~skel, return_indices=True)
    mask_keep = mask & skel_keep[ir, ic]
    return skel_keep, mask_keep, info


def filter_paths_cooriented(skel, g_resp, g_nr, g_nc, g_low, P: Params):
    """Remove skeleton paths that run *along* G-band dark lanes: more than
    ``bg_overlap_frac`` of the path's pixels have G-band ridge response ≥
    ``g_low`` *and* a tangent within ``bg_angle_deg`` of the lane direction."""
    if not skel.any():
        return skel.copy(), dict(n_arcs=0, n_removed=0, n_kept=0)
    sk, _ = _skan(skel)
    out = skel.copy()
    cos_lim = np.cos(np.deg2rad(P.bg_angle_deg))
    n_removed = 0
    for i in range(sk.n_paths):
        c = sk.path_coordinates(i).astype(float)
        if len(c) < 5:
            continue
        t = _unit_tangents(c)
        rr, cc = np.round(c).astype(int).T
        gdir = np.column_stack([-g_nc[rr, cc], g_nr[rr, cc]])          # lane direction ⟂ its normal
        along = (g_resp[rr, cc] >= g_low) & (np.abs((t * gdir).sum(1)) >= cos_lim)
        if along.mean() > P.bg_overlap_frac and along.sum() > P.bg_overlap_len:
            inner = c[1:-1] if len(c) > 2 else c
            r2, c2 = np.round(inner).astype(int).T
            out[r2, c2] = False
            n_removed += 1
    out = remove_small_objects(out, min_size=2, connectivity=2)
    return out, dict(n_arcs=int(sk.n_paths), n_removed=n_removed, n_kept=int(sk.n_paths - n_removed))


# ----------------------------------------------------------------------------
# pruning and branch tests
# ----------------------------------------------------------------------------
def prune_short_branches(skel, min_len=35.0, drop_isolated_below=None):
    """Remove spurs / junction–junction segments / cycles shorter than
    ``min_len`` (branch_type ≥ 1)."""
    if not skel.any():
        return skel.copy()
    sk, tab = _skan(skel)
    out = skel.copy()
    short = tab.index[(tab["branch_type"].values >= 1) & (tab["branch_distance"].values < min_len)]
    for idx in short:
        coords = sk.path_coordinates(idx)
        if coords.shape[0] > 2:
            coords = coords[1:-1]
        rr, cc = np.round(coords).astype(int).T
        out[rr, cc] = False
    if drop_isolated_below is not None:
        iso = tab.index[(tab["branch_type"].values == 0) & (tab["branch_distance"].values < drop_isolated_below)]
        for idx in iso:
            rr, cc = np.round(sk.path_coordinates(idx)).astype(int).T
            out[rr, cc] = False
    out = remove_small_objects(out, min_size=2, connectivity=2)
    return out


def two_sided_contrast(coords, merged, nr, nc, sigma_best, k=1.5):
    """Side contrasts (I(+d) − I_c, I(−d) − I_c) at ±k·sigma along the normal."""
    rr, cc = np.round(coords).astype(int).T
    rr = np.clip(rr, 0, merged.shape[0] - 1)
    cc = np.clip(cc, 0, merged.shape[1] - 1)
    d = k * np.maximum(sigma_best[rr, cc], 1.0)
    Ip = ndi.map_coordinates(merged, [rr + d * nr[rr, cc], cc + d * nc[rr, cc]], order=1, mode="nearest")
    Im = ndi.map_coordinates(merged, [rr - d * nr[rr, cc], cc - d * nc[rr, cc]], order=1, mode="nearest")
    Ic = merged[rr, cc]
    return Ip - Ic, Im - Ic


def straight_branches(skel, P: Params, sigma_best=None, resp=None, merged=None, nr=None, nc=None, noise=None):
    """Measure every branch of ``skel`` (length, chord, tortuosity, curvature,
    width, two-sided contrast) and return (skeleton of kept branches, table)."""
    sk, tab = _skan(skel)
    length = tab["branch_distance"].values
    chord = tab["euclidean_distance"].values
    tort = length / np.maximum(chord, 1e-6)
    n = len(tab)
    kappa_q = np.zeros(n)
    width = np.full(n, np.nan)
    mean_resp = np.full(n, np.nan)
    ts_min = np.full(n, np.nan)
    ts_asym = np.full(n, np.nan)
    kappa_limit = 1.0 / P.rmin
    do_ts = P.two_sided and merged is not None and nr is not None and sigma_best is not None
    if do_ts and noise is None:
        noise = _noise(merged)
    for i in range(n):
        if length[i] < P.min_branch_len:
            continue
        coords = sk.path_coordinates(i)
        if length[i] >= P.curvature_min_len:
            k = _curvature(_smooth_path(coords, P.curvature_smooth))
            kappa_q[i] = np.percentile(k, P.curvature_pct) if k.size else 0.0
        rr, cc = np.round(coords).astype(int).T
        if sigma_best is not None:
            width[i] = 2.355 * np.mean(sigma_best[rr, cc])
        if resp is not None:
            mean_resp[i] = np.mean(resp[rr, cc])
        if do_ts:
            cp, cm = two_sided_contrast(coords, merged, nr, nc, sigma_best, P.ts_k)
            ts_min[i] = np.median(np.minimum(cp, cm))
            ts_asym[i] = np.median(np.abs(cp - cm) / np.maximum(np.maximum(cp, cm), 1e-9))
    long_enough = length >= P.min_branch_len
    straight = (tort <= P.tortuosity_max) & (kappa_q <= kappa_limit)
    keep = long_enough & (straight if P.keep_only_straight else True)
    two_sided_ok = np.ones(n, bool)
    if do_ts:
        two_sided_ok = (ts_min > P.ts_min_contrast * noise) & (ts_asym <= P.ts_max_asym)
        keep &= two_sided_ok
    tab = tab.assign(tortuosity=tort, curvature_q=kappa_q, width_px=width, mean_resp=mean_resp,
                     ts_min=ts_min, ts_asym=ts_asym, is_straight=straight, two_sided_ok=two_sided_ok, keep=keep)
    out = np.zeros(skel.shape, bool)
    for i in np.nonzero(keep)[0]:
        rr, cc = np.round(sk.path_coordinates(i)).astype(int).T
        out[rr, cc] = True
    return out, tab


# ----------------------------------------------------------------------------
# endpoint linking
# ----------------------------------------------------------------------------
_OFFS = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])


def _tangent_at_endpoints(skel, pts, window):
    """Unit tangent pointing *outwards* at every endpoint, from the pixel
    ``window`` steps back along the skeleton (stops at junctions)."""
    H, W = skel.shape
    dirs = np.zeros((len(pts), 2))
    ok = np.zeros(len(pts), bool)
    for k, (r, c) in enumerate(pts):
        prev = (r, c)
        cur = (r, c)
        chain = [(r, c)]
        for _ in range(window):
            nb = []
            for dr, dc in _OFFS:
                rr, cc = cur[0] + dr, cur[1] + dc
                if 0 <= rr < H and 0 <= cc < W and skel[rr, cc] and (rr, cc) != prev and (rr, cc) != cur:
                    nb.append((rr, cc))
            if len(nb) > 1 and prev != cur:
                nb = [p for p in nb if max(abs(p[0] - prev[0]), abs(p[1] - prev[1])) > 1] or nb[:1]
            if len(nb) != 1:
                break
            prev, cur = cur, nb[0]
            chain.append(cur)
        if len(chain) < 3:
            if len(chain) == 2:
                v = np.array([r - chain[1][0], c - chain[1][1]], float)
                dirs[k] = v / np.hypot(*v)
                ok[k] = True
            continue
        # principal direction of the chain (robust to a 1-px hook at the very end),
        # oriented outwards (from the chain's centre towards the endpoint)
        ch = np.array(chain, float)
        ctr = ch.mean(0)
        _, _, vt = np.linalg.svd(ch - ctr, full_matrices=False)
        v = vt[0]
        if np.dot(v, ch[0] - ctr) < 0:
            v = -v
        dirs[k] = v
        ok[k] = True
    return dirs, ok


class _UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[rb] = ra
        return True


def hermite_bridge(p0, d0, p1, d1):
    """Cubic Hermite curve from ``p0`` (leaving along ``d0``) to ``p1``
    (arriving along ``-d1``; ``d1`` is the outward tangent at ``p1``).
    Returns the sampled points and the maximum curvature."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    L = max(np.hypot(*(p1 - p0)), 1e-6)
    m0 = L * np.asarray(d0, float)
    m1 = -L * np.asarray(d1, float)
    n = max(8, int(2 * L))
    t = np.linspace(0.0, 1.0, n)[:, None]
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    pts = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
    d1_ = (6 * t ** 2 - 6 * t) * p0 + (3 * t ** 2 - 4 * t + 1) * m0 + (-6 * t ** 2 + 6 * t) * p1 + (3 * t ** 2 - 2 * t) * m1
    d2_ = (12 * t - 6) * p0 + (6 * t - 4) * m0 + (-12 * t + 6) * p1 + (6 * t - 2) * m1
    kappa = np.abs(d1_[:, 0] * d2_[:, 1] - d1_[:, 1] * d2_[:, 0]) / (np.hypot(d1_[:, 0], d1_[:, 1]) ** 3 + 1e-9)
    return pts, float(kappa.max())


def _rasterize(pts, shape):
    """Pixel chain through the rounded sample points (consecutive points joined by lines)."""
    p = np.round(pts).astype(int)
    p[:, 0] = np.clip(p[:, 0], 0, shape[0] - 1)
    p[:, 1] = np.clip(p[:, 1], 0, shape[1] - 1)
    rr, cc = [p[0, 0]], [p[0, 1]]
    for (r0, c0), (r1, c1) in zip(p[:-1], p[1:]):
        if r0 == r1 and c0 == c1:
            continue
        lr, lc = draw_line(r0, c0, r1, c1)
        rr.extend(lr[1:]); cc.extend(lc[1:])
    return np.array(rr), np.array(cc)


def link_endpoints(skel, P: Params, resp=None, low=None, n_passes=None, field=None):
    """Bridge gaps between trace endpoints.

    Candidate pairs come from a KD-tree (distance ≤ ``link_gap``).  A pair is
    accepted when both outward tangents point towards each other within the
    angle tolerance, the two tangents are collinear with each other, the two
    endpoints belong to different traces (union–find), the bridge (a cubic
    Hermite curve honouring both tangents when ``link_rmin > 0``, a straight
    line otherwise) bends with a radius of curvature ≥ ``link_rmin``, does not
    run into another trace and keeps ridge response ≥ ``low`` on at least
    ``bridge_frac`` of its pixels.  Pairs are accepted greedily by score and the
    procedure is repeated up to ``link_passes`` times.
    """
    out = skel.copy()
    tol0 = np.deg2rad(P.link_angle_deg)
    all_links = []
    n_passes = P.link_passes if n_passes is None else n_passes
    use_hermite = P.link_rmin > 0
    for _ in range(n_passes):
        lbl, nlab = ndi.label(out, structure=_K8)
        pts = np.column_stack(np.nonzero(endpoints(out)))
        if len(pts) < 2:
            break
        dirs, ok = _tangent_at_endpoints(out, pts, P.tangent_window)
        if ok.sum() < 2:
            break
        tree = cKDTree(pts[ok])
        idx_ok = np.nonzero(ok)[0]
        pairs = tree.query_pairs(P.link_gap, output_type="ndarray")
        if len(pairs) == 0:
            break
        i = idx_ok[pairs[:, 0]]
        j = idx_ok[pairs[:, 1]]
        same = lbl[pts[i, 0], pts[i, 1]] == lbl[pts[j, 0], pts[j, 1]]
        i, j = i[~same], j[~same]
        v = pts[j] - pts[i]
        dist = np.hypot(v[:, 0], v[:, 1])
        vhat = v / np.maximum(dist, 1e-9)[:, None]
        ang0 = np.arccos(np.clip((dirs[i] * vhat).sum(1), -1, 1))
        ang1 = np.arccos(np.clip((dirs[j] * -vhat).sum(1), -1, 1))
        coll = np.arccos(np.clip((dirs[i] * -dirs[j]).sum(1), -1, 1))      # collinearity of the two tangents
        # angle tolerance: notebook's optional gap scaling plus the direction uncertainty of a
        # pixel-quantised gap vector (atan(pos_tol / gap): 37° at 2 px, 3° at 30 px)
        eff = tol0 * (1 + P.link_alpha * (1 - dist / P.link_gap)) + np.arctan2(P.link_pos_tol, dist)
        # lateral offset of each endpoint from the other's tangent line
        off_i = np.abs(v[:, 0] * dirs[i][:, 1] - v[:, 1] * dirs[i][:, 0])
        off_j = np.abs(v[:, 0] * dirs[j][:, 1] - v[:, 1] * dirs[j][:, 0])
        offset = np.maximum(off_i, off_j)
        good = (ang0 <= eff) & (ang1 <= eff) & (coll <= 2 * tol0 + np.arctan2(P.link_pos_tol, dist))
        if field is not None:
            # the bridge must not cross a coherent fibril direction
            theta, coh = field
            mr = np.clip(np.round(0.5 * (pts[i, 0] + pts[j, 0])).astype(int), 0, theta.shape[0] - 1)
            mc = np.clip(np.round(0.5 * (pts[i, 1] + pts[j, 1])).astype(int), 0, theta.shape[1] - 1)
            phi = np.arctan2(v[:, 0], v[:, 1])
            mis = np.abs(((phi - theta[mr, mc]) + np.pi / 2) % np.pi - np.pi / 2)
            good &= ~((coh[mr, mc] > P.field_min_coherence) & (mis > np.deg2rad(P.field_max_misalign_deg)))
        i, j, dist, ang0, ang1, eff, offset = i[good], j[good], dist[good], ang0[good], ang1[good], eff[good], offset[good]
        offset_ok = offset <= P.link_pos_tol + P.link_offset_frac * dist
        score = dist / P.link_gap + 0.5 * (ang0 + ang1) / np.maximum(eff, 1e-6)
        order = np.argsort(score)
        uf = _UF(nlab + 1)
        used = np.zeros(len(pts), bool)
        links = []
        for k in order:
            a, b = i[k], j[k]
            if used[a] or used[b]:
                continue
            la, lb = lbl[pts[a, 0], pts[a, 1]], lbl[pts[b, 0], pts[b, 1]]
            if uf.find(la) == uf.find(lb):
                continue
            if use_hermite:
                curve, kmax = hermite_bridge(pts[a], dirs[a], pts[b], dirs[b])
                # a bridge is accepted when its lateral offset is within the quantisation
                # allowance, or when the smoothest curve honouring both tangents is gentle enough
                if not offset_ok[k] and kmax > 1.0 / P.link_rmin:
                    continue
                rr, cc = _rasterize(curve, out.shape)
            else:
                rr, cc = draw_line(int(pts[a, 0]), int(pts[a, 1]), int(pts[b, 0]), int(pts[b, 1]))
            ri, ci = rr[1:-1], cc[1:-1]
            if ri.size > 2 and out[ri[1:-1], ci[1:-1]].any():
                continue
            if resp is not None and low is not None and ri.size >= 3:
                if np.mean(resp[ri, ci] >= low) < P.bridge_frac:
                    continue
            used[a] = used[b] = True
            uf.union(la, lb)
            links.append(((int(pts[a, 0]), int(pts[a, 1])), (int(pts[b, 0]), int(pts[b, 1])), rr, cc))
        if not links:
            break
        for _, _, rr, cc in links:
            out[rr, cc] = True
        all_links.extend([(a, b) for a, b, _, _ in links])
        out = skeletonize(out)
    return out, all_links


# ----------------------------------------------------------------------------
# junction resolution, kink splitting, polylines
# ----------------------------------------------------------------------------
def _dedupe(c):
    if len(c) < 2:
        return c
    keep = np.ones(len(c), bool)
    keep[1:] = np.any(np.abs(np.diff(c, axis=0)) > 1e-9, axis=1)
    return c[keep]


def resolve_junctions(skel, P: Params):
    """Split ``skel`` at its junctions and re-join the arcs pairwise by
    tangent continuity (turning angle ≤ ``junction_max_turn_deg``), so that
    crossing or touching fibrils become separate polylines.  Tiny paths
    between adjacent junction pixels (≤ ``junction_cluster_len``) are treated
    as one junction.  Returns a list of ordered (N, 2) float coordinate arrays."""
    if not skel.any():
        return []
    sk, tab = _skan(skel)
    n = sk.n_paths
    coords = [sk.path_coordinates(i).astype(float) for i in range(n)]
    src = tab["node_id_src"].values.astype(int)
    dst = tab["node_id_dst"].values.astype(int)
    L = tab["branch_distance"].values
    deg = sk.degrees
    uf = _UF(int(max(src.max(), dst.max())) + 1)
    is_conn = np.zeros(n, bool)
    for i in range(n):
        if L[i] <= P.junction_cluster_len and deg[src[i]] >= 3 and deg[dst[i]] >= 3:
            uf.union(src[i], dst[i])
            is_conn[i] = True
    k = max(3, int(P.tangent_window))
    ends = {}
    for i in range(n):
        if is_conn[i] or len(coords[i]) < 2:
            continue
        c = coords[i]
        kk = min(k, len(c) - 1)
        for e, node, a in ((0, src[i], c[0] - c[kk]), (1, dst[i], c[-1] - c[-1 - kk])):
            nrm = np.hypot(*a)
            if nrm == 0 or deg[node] < 3:
                continue
            ends.setdefault(uf.find(node), []).append((i, e, a / nrm))
    partner = {}
    max_turn = np.deg2rad(P.junction_max_turn_deg)
    for node, lst in ends.items():
        cand = []
        for x in range(len(lst)):
            for y in range(x + 1, len(lst)):
                i, e, a = lst[x]
                j, f, b = lst[y]
                if i == j:
                    continue
                turn = np.arccos(np.clip(np.dot(a, -b), -1, 1))
                if turn <= max_turn:
                    cand.append((turn, x, y))
        cand.sort()
        used = set()
        for turn, x, y in cand:
            if x in used or y in used:
                continue
            used.add(x); used.add(y)
            i, e, _ = lst[x]
            j, f, _ = lst[y]
            partner[(i, e)] = (j, f)
            partner[(j, f)] = (i, e)
    visited = np.zeros(n, bool)
    polylines = []
    for i0 in range(n):
        if visited[i0] or is_conn[i0] or len(coords[i0]) < 2:
            continue
        cur, end = i0, 0
        seen = set()
        while (cur, end) in partner and (cur, end) not in seen:
            seen.add((cur, end))
            j, f = partner[(cur, end)]
            cur, end = j, 1 - f
        pts = []
        while not visited[cur]:
            visited[cur] = True
            c = coords[cur] if end == 0 else coords[cur][::-1]
            pts.append(c)
            exit_key = (cur, 1 - end)
            if exit_key in partner:
                j, f = partner[exit_key]
                cur, end = j, f
            else:
                break
        if pts:
            polylines.append(_dedupe(np.concatenate(pts)))
    return polylines


def split_kinks(polylines, P: Params):
    """Split polylines where the heading turns by more than ``kink_angle_deg``
    between the tangents ``kink_window`` px before and after a point."""
    w = max(2, int(P.kink_window))
    thr = np.deg2rad(P.kink_angle_deg)
    out = []
    n_kinks = 0
    for c in polylines:
        if len(c) < 2 * w + 3:
            out.append(c)
            continue
        t = _unit_tangents(c)
        ang = np.unwrap(np.arctan2(t[:, 0], t[:, 1]))
        turn = np.abs(ang[2 * w:] - ang[:-2 * w])
        peaks = []
        for s in range(len(turn)):
            if turn[s] > thr and turn[s] >= turn[max(0, s - 2 * w):s + 2 * w + 1].max():
                if not peaks or s - peaks[-1] > 2 * w:
                    peaks.append(s)
        prev = 0
        for s in peaks:
            idx = s + w
            if idx - prev > 1:
                out.append(c[prev:idx])
            prev = idx + 1
            n_kinks += 1
        if len(c) - prev > 1:
            out.append(c[prev:])
    return out, n_kinks


def polyline_length(c):
    return float(np.hypot(*np.diff(c, axis=0).T).sum()) if len(c) > 1 else 0.0


def polylines_to_image(polylines, shape):
    out = np.zeros(shape, bool)
    for c in polylines:
        if len(c) == 0:
            continue
        rr, cc = _rasterize(c, shape)
        out[rr, cc] = True
    return out


def scale_polylines(polylines, bin):
    if bin == 1:
        return polylines
    return [c * bin + (bin - 1) / 2.0 for c in polylines]


# ----------------------------------------------------------------------------
# local fibril-orientation field and run-based trimming (round 3)
# ----------------------------------------------------------------------------
def orientation_field(resp, nr, nc, scale=60.0, bin=4):
    """Dominant local ridge orientation and its coherence.

    The ridge direction (perpendicular to the Hessian normal) is averaged in the
    double-angle representation, weighted by the ridge response, over a Gaussian
    of ``scale`` px.  Returns ``theta`` (angle of the dominant direction measured
    from the column axis, with the sine along rows) and ``coherence`` in [0, 1]
    (1 = all response in one direction, 0 = isotropic)."""
    dr, dc = -nc, nr
    a = resp * (dc * dc - dr * dr)
    b = resp * (2.0 * dr * dc)
    bin = max(1, int(bin)) if scale / max(1, int(bin)) >= 4 else 1
    if bin > 1:                       # the field is smooth: average on a binned grid, then interpolate back
        a, b, r = bin_image(a, bin), bin_image(b, bin), bin_image(resp, bin)
        sc = scale / bin
    else:
        r, sc = resp, scale
    c2 = ndi.gaussian_filter(a, sc)
    s2 = ndi.gaussian_filter(b, sc)
    w = ndi.gaussian_filter(r, sc) + 1e-12
    theta = 0.5 * np.arctan2(s2, c2)
    coh = np.clip(np.hypot(c2, s2) / w, 0, 1)
    if bin > 1:
        def up(x):
            z = ndi.zoom(x, bin, order=1, mode="nearest")
            out = np.empty(resp.shape, np.float32)
            rr, cc = min(z.shape[0], resp.shape[0]), min(z.shape[1], resp.shape[1])
            out[:rr, :cc] = z[:rr, :cc]
            out[rr:, :] = out[rr - 1:rr, :]
            out[:, cc:] = out[:, cc - 1:cc]
            return out
        # interpolate the double-angle components, not the angle (which wraps)
        c2u, s2u = up(c2), up(s2)
        theta = 0.5 * np.arctan2(s2u, c2u)
        coh = up(coh)
    return theta.astype(np.float32), coh.astype(np.float32)


def misalignment_along(coords, theta, coh, smooth=3.0):
    """Per-point angle (deg) between the trace tangent and the local dominant
    orientation, and the local coherence."""
    t = _unit_tangents(coords, smooth)
    ang = np.arctan2(t[:, 0], t[:, 1])
    rr, cc = np.round(coords).astype(int).T
    rr = np.clip(rr, 0, theta.shape[0] - 1)
    cc = np.clip(cc, 0, theta.shape[1] - 1)
    mis = np.abs(((ang - theta[rr, cc]) + np.pi / 2) % np.pi - np.pi / 2)
    return np.degrees(mis), coh[rr, cc]


def _failing_runs(mask, min_run):
    """(start, stop) index pairs of runs of True at least ``min_run`` long."""
    if not mask.any():
        return []
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.nonzero(d == 1)[0]
    stops = np.nonzero(d == -1)[0]
    return [(a, b) for a, b in zip(starts, stops) if b - a >= min_run]


def split_at_runs(polylines, fail_fn, min_run, min_len):
    """Cut every polyline where ``fail_fn(coords)`` is True for at least
    ``min_run`` consecutive points; the failing points are removed and the
    remaining pieces shorter than ``min_len`` px are dropped.  Returns the new
    polylines, the number of removed runs and the number of removed points."""
    out, n_runs, n_pts = [], 0, 0
    for c in polylines:
        if len(c) < 3:
            out.append(c)
            continue
        runs = _failing_runs(np.asarray(fail_fn(c), bool), min_run)
        if not runs:
            out.append(c)
            continue
        prev = 0
        for a, b in runs:
            if a - prev >= 2:
                out.append(c[prev:a])
            prev = b
            n_runs += 1
            n_pts += b - a
        if len(c) - prev >= 2:
            out.append(c[prev:])
    out = [c for c in out if polyline_length(c) >= min_len]
    return out, n_runs, n_pts


# ----------------------------------------------------------------------------
# per-trace tables and metrics
# ----------------------------------------------------------------------------
def _pca_extent(pts):
    ctr = pts.mean(0)
    if len(pts) > 2:
        cov = np.cov((pts - ctr).T)
        evals, evecs = np.linalg.eigh(cov)
        ax = evecs[:, -1]
        proj = (pts - ctr) @ ax
        return float(proj.max() - proj.min()), float(np.degrees(np.arctan2(ax[1], ax[0])) % 180), ctr
    return float(len(pts)), np.nan, ctr


def trace_table_from_polylines(polylines, merged=None, resp=None, sigma_best=None, bg_scale=15,
                               kink_window=10):
    """One row per polyline trace."""
    rows = []
    bg = local_background(merged, bg_scale) if (merged is not None and bg_scale) else None
    shape = None
    for arr in (merged, resp, sigma_best):
        if arr is not None:
            shape = arr.shape
            break
    for k, c in enumerate(polylines, start=1):
        length = polyline_length(c)
        extent, angle, ctr = _pca_extent(c)
        row = dict(trace_id=k, n_pts=int(len(c)), length=length, extent=extent,
                   tortuosity=float(length / max(extent, 1e-6)), angle_deg=angle,
                   r_mean=float(ctr[0]), c_mean=float(ctr[1]),
                   r_start=float(c[0, 0]), c_start=float(c[0, 1]), r_end=float(c[-1, 0]), c_end=float(c[-1, 1]))
        if len(c) >= 2 * kink_window + 3:
            t = _unit_tangents(c)
            ang = np.unwrap(np.arctan2(t[:, 0], t[:, 1]))
            row["max_turn_deg"] = float(np.degrees(np.abs(ang[2 * kink_window:] - ang[:-2 * kink_window]).max()))
        else:
            row["max_turn_deg"] = np.nan
        if shape is not None:
            rr, cc = np.round(c).astype(int).T
            rr = np.clip(rr, 0, shape[0] - 1)
            cc = np.clip(cc, 0, shape[1] - 1)
            if merged is not None:
                row["mean_intensity"] = float(merged[rr, cc].mean())
                if bg is not None:
                    row["contrast"] = float((bg[rr, cc] - merged[rr, cc]).mean())
            if resp is not None:
                row["mean_resp"] = float(resp[rr, cc].mean())
            if sigma_best is not None:
                row["width_px"] = float(2.355 * sigma_best[rr, cc].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def trace_table(skel, merged=None, resp=None, sigma_best=None, bg_scale=15):
    """One row per connected component of a bool skeleton (component-based
    traces, as in round 1)."""
    lbl, n = ndi.label(skel, structure=_K8)
    if n == 0:
        return pd.DataFrame(), lbl
    sk, tab = _skan(skel)
    comp_of_path = np.array([lbl[tuple(np.round(sk.path_coordinates(i)[0]).astype(int))] for i in range(sk.n_paths)])
    length = np.bincount(comp_of_path, weights=sk.path_lengths(), minlength=n + 1)[1:]
    npix = np.bincount(lbl.ravel(), minlength=n + 1)[1:]
    objs = ndi.find_objects(lbl)
    rows = []
    bg = local_background(merged, bg_scale) if (merged is not None and bg_scale) else None
    for k, sl in enumerate(objs, start=1):
        rr, cc = np.nonzero(lbl[sl] == k)
        rr = rr + sl[0].start
        cc = cc + sl[1].start
        pts = np.column_stack([rr, cc]).astype(float)
        extent, angle, ctr = _pca_extent(pts)
        row = dict(trace_id=k, n_pix=int(npix[k - 1]), length=float(length[k - 1]),
                   extent=float(extent), tortuosity=float(length[k - 1] / max(extent, 1e-6)),
                   angle_deg=float(angle), r_mean=float(ctr[0]), c_mean=float(ctr[1]))
        if merged is not None:
            row["mean_intensity"] = float(merged[rr, cc].mean())
            if bg is not None:
                row["contrast"] = float((bg[rr, cc] - merged[rr, cc]).mean())
        if resp is not None:
            row["mean_resp"] = float(resp[rr, cc].mean())
        if sigma_best is not None:
            row["width_px"] = float(2.355 * sigma_best[rr, cc].mean())
        rows.append(row)
    return pd.DataFrame(rows), lbl


def kink_stats(polylines, window=10, angle_deg=45.0):
    """Kinks (heading change > ``angle_deg`` within ``2*window`` px) per
    1000 px of path, and percentiles of the per-path maximum turn."""
    thr = np.deg2rad(angle_deg)
    n_k, total, maxturn = 0, 0.0, []
    for c in polylines:
        c = np.asarray(c, float)
        total += polyline_length(c)
        if len(c) < 2 * window + 3:
            continue
        t = _unit_tangents(c)
        ang = np.unwrap(np.arctan2(t[:, 0], t[:, 1]))
        turn = np.abs(ang[2 * window:] - ang[:-2 * window])
        peaks = (turn > thr) & (turn == ndi.maximum_filter1d(turn, 2 * window + 1))
        n_k += int(peaks.sum())
        maxturn.append(np.degrees(turn.max()))
    pct = np.percentile(maxturn, [50, 75, 90, 95]) if maxturn else [np.nan] * 4
    return dict(kinks_per_kpx=1000.0 * n_k / max(total, 1e-9), max_turn_p50=float(pct[0]),
                max_turn_p75=float(pct[1]), max_turn_p90=float(pct[2]), max_turn_p95=float(pct[3]))


def contrast_along(skel, merged, bg_scale=15, null_shifts=((0, 12), (12, 0), (0, -12), (-12, 0))):
    """Mean (local background − intensity) at trace pixels, and the same for
    the trace shifted sideways (null)."""
    bg = local_background(merged, bg_scale)
    diff = bg - merged
    sig = float(diff[skel].mean())
    nulls = []
    for dr, dc in null_shifts:
        sh = np.roll(np.roll(skel, dr, 0), dc, 1)
        nulls.append(float(diff[sh].mean()))
    return dict(contrast=sig, null=float(np.mean(nulls)), ratio=sig / max(np.mean(nulls), 1e-9),
                frac_positive=float((diff[skel] > 0).mean()))


def two_sided_fraction(skel, merged, nr, nc, sigma_best, k=1.5):
    """Fraction of trace pixels darker than both sides (pixel-level, independent
    of the branch-level test)."""
    rr, cc = np.nonzero(skel)
    if rr.size == 0:
        return np.nan
    cp, cm = two_sided_contrast(np.column_stack([rr, cc]), merged, nr, nc, sigma_best, k)
    return float(((cp > 0) & (cm > 0)).mean())


def orientation_agreement(skel, nr, nc, smooth=3.0):
    sk, _ = _skan(skel)
    angs = []
    for i in range(sk.n_paths):
        coords = sk.path_coordinates(i)
        if coords.shape[0] < 7:
            continue
        t = _unit_tangents(coords, smooth)
        rr, cc = np.round(coords).astype(int).T
        n = np.column_stack([nr[rr, cc], nc[rr, cc]])
        ang = np.degrees(np.arcsin(np.clip(np.abs((t * n).sum(1)), 0, 1)))
        angs.append(ang[3:-3])
    if not angs:
        return np.nan
    return float(np.median(np.concatenate(angs)))


def trace_metrics(skel, merged=None, nr=None, nc=None, min_long=100.0, name=""):
    """Summary numbers for a bool trace skeleton (component-based)."""
    tt, _ = trace_table(skel)
    if tt.empty:
        return dict(name=name, n_traces=0)
    L = tt["length"].values
    m = dict(name=name, n_traces=int(len(tt)), total_length=float(L.sum()),
             mean_length=float(L.mean()), median_length=float(np.median(L)),
             p90_length=float(np.percentile(L, 90)), max_length=float(L.max()),
             frac_length_long=float(L[L >= min_long].sum() / L.sum()),
             n_long=int((L >= min_long).sum()), n_pix=int(skel.sum()))
    if merged is not None:
        m.update({f"c_{k}": v for k, v in contrast_along(skel, merged).items()})
    if nr is not None:
        m["tangent_vs_ridge_deg"] = orientation_agreement(skel, nr, nc)
    return m


def coverage(skel_a, skel_b, tol=3.0):
    """Fraction of pixels of ``skel_a`` within ``tol`` px of ``skel_b`` and vice versa."""
    da = distance_transform_edt(~skel_b)
    db = distance_transform_edt(~skel_a)
    return dict(a_in_b=float((da[skel_a] <= tol).mean()), b_in_a=float((db[skel_b] <= tol).mean()))


def upsample_skeleton(skel, bin, shape):
    """Map a bool skeleton found on a ``bin``-binned image back to full resolution."""
    if bin == 1:
        return skel
    return polylines_to_image(scale_polylines(paths_from_skeleton(skel), bin), shape)


# ----------------------------------------------------------------------------
# full pipeline
# ----------------------------------------------------------------------------
def run_pipeline(Hb, Gb, bin=1, centerline="nms", P: Params | None = None, verbose=True,
                 merged=None, keep_arrays=True):
    """Run photospheric removal + fibril tracing.

    Parameters
    ----------
    Hb, Gb : 2-D arrays
        Destretched (co-aligned) Hβ and G-band images at full resolution.
        ``Gb`` may be ``None`` when ``merged`` is given (no G-band products then).
    bin : int
        Block-averaging factor applied to both images before processing.
    centerline : {"nms", "mask"}
    P : Params (full-resolution units; scaled internally by ``bin``)
    merged : optional pre-computed merged image at the requested binning.
    """
    P0 = Params() if P is None else P
    P = P0.scaled(bin)
    T = Timer(verbose=verbose)
    res = dict(bin=bin, centerline=centerline, params=P)

    with T("bin"):
        Hb_b = bin_image(np.asarray(Hb, np.float32), bin) if Hb is not None else None
        Gb_b = bin_image(np.asarray(Gb, np.float32), bin) if Gb is not None else None
    if merged is None:
        with T("photospheric removal"):
            merged, coef = remove_photosphere(Hb_b, Gb_b, P)
            res["poly_coef"] = coef
    merged = np.asarray(merged, np.float32)
    noise = _noise(merged)
    res["noise"] = noise

    with T("hessian ridge (merged)"):
        resp, sbest, nr, nc = hessian_ridge(merged, P.sigmas, P.black_ridges, want_dir=True)
    if P.dark_gate:
        with T("dark-ridge gate"):
            bg = local_background(merged, P.bg_scale)
            dev = merged - bg
            mad = 1.4826 * np.median(np.abs(dev[::4, ::4] - np.median(dev[::4, ::4])))
            bright = dev > P.dark_gate_delta * mad
            resp[bright] = 0.0
            res["dark_gate_frac"] = float(bright.mean())
    bp_zone = None
    if P.bp_exclude:
        with T("bright-point exclusion"):
            bp_zone = bright_point_zone(Gb_b, merged, P)
            resp[bp_zone] = 0.0
            res["bp_zone_frac"] = float(bp_zone.mean())
    low, high = _percentiles(resp, (P.low_pct, P.high_pct))
    res.update(low=float(low), high=float(high))

    if centerline == "mask":
        with T("mask + skeletonize"):
            mask = make_mask(resp, P.low_pct, P.high_pct, P.min_size, P.open_radius)
            skel = skeletonize(mask)
    elif centerline == "nms":
        with T("ridge NMS"):
            mask = None
            skel = ridge_nms(resp, nr, nc, low, high, P.nms_min_len, P.nms_tol)
    else:
        raise ValueError(centerline)

    if P.use_bg_filter and Gb_b is not None and np.ptp(Gb_b) > 0:
        with T("G-band background filter"):
            g_resp, _, g_nr, g_nc = hessian_ridge(Gb_b, P.bg_sigmas, True, want_dir=(P.bg_mode == "cooriented"))
            if P.bg_mode == "cooriented":
                g_low = float(_percentiles(g_resp, P.bg_low_pct))
                skel, info = filter_paths_cooriented(skel, g_resp, g_nr, g_nc, g_low, P)
            else:
                g_mask = make_mask(g_resp, P.bg_low_pct, P.bg_high_pct, P.bg_min_size, P.bg_open_radius)
                skel, mask_f, info = filter_skeleton_by_background(skel, skeletonize(g_mask), P.bg_dilation,
                                                                   P.bg_overlap_frac, P.bg_overlap_len, mask)
                if mask_f is not None:
                    mask = mask_f
            res["bg_filter_info"] = info

    with T("prune short branches"):
        skel_clean = prune_short_branches(skel, P.prune_len)
    with T("branch measurements / straightness / two-sided"):
        straight, btab = straight_branches(skel_clean, P, sbest, resp, merged, nr, nc, noise)
    if not P.link_before_length_cut:
        with T("length cut (before linking)"):
            if straight.any():
                tt, lbl = trace_table(straight)
                drop = tt["trace_id"].values[tt["length"].values < P.final_min_len]
                straight &= ~np.isin(lbl, drop)
    field = None
    if P.field_trim or P.link_field_check:
        with T("orientation field"):
            field = orientation_field(resp, nr, nc, P.field_scale, bin=P.field_bin)
    with T("endpoint linking"):
        linked, links = link_endpoints(straight, P, resp if P.bridge_check else None, low,
                                       field=field if P.link_field_check else None)
    with T("final prune"):
        final = prune_short_branches(linked, P.final_prune_len)
        final = remove_small_objects(final, min_size=max(2, int(P.final_min_len)), connectivity=2)

    n_kinks = 0
    with T("junction resolution / kinks / table"):
        if P.resolve_junctions:
            polylines = resolve_junctions(final, P)
        else:
            polylines = paths_from_skeleton(final)
        if P.kink_split:
            polylines, n_kinks = split_kinks(polylines, P)
        trim_info = {}
        if P.ts_trim:
            def _ts_fail(c):
                cp, cm = two_sided_contrast(c, merged, nr, nc, sbest, P.ts_k)
                asym = np.abs(cp - cm) / np.maximum(np.maximum(cp, cm), 1e-9)
                return (np.minimum(cp, cm) < P.ts_trim_min_contrast * noise) | (asym > P.ts_trim_asym)
            polylines, n_r, n_p = split_at_runs(polylines, _ts_fail, P.ts_trim_min_run, P.final_min_len)
            trim_info.update(ts_runs=n_r, ts_points=n_p)
        if P.field_trim and field is not None:
            def _field_fail(c):
                mis, co = misalignment_along(c, field[0], field[1])
                return (mis > P.field_max_misalign_deg) & (co > P.field_min_coherence)
            polylines, n_r, n_p = split_at_runs(polylines, _field_fail, P.field_min_run, P.final_min_len)
            trim_info.update(field_runs=n_r, field_points=n_p)
        res["trim_info"] = trim_info
        polylines = [c for c in polylines if polyline_length(c) >= P.final_min_len]
        if P.resolve_junctions or P.kink_split or P.ts_trim or P.field_trim:
            final = polylines_to_image(polylines, final.shape)
            ttab = trace_table_from_polylines(polylines, merged, resp, sbest, P.bg_scale, int(P.kink_window))
        else:
            ttab, lbl = trace_table(final, merged, resp, sbest, bg_scale=P.bg_scale)
            if not ttab.empty:
                drop = ttab["trace_id"].values[ttab["length"].values < P.final_min_len]
                if len(drop):
                    final &= ~np.isin(lbl, drop)
                    ttab = ttab[ttab["length"].values >= P.final_min_len].reset_index(drop=True)
        grouped_mask = None
        if mask is not None:
            grouped_mask = ndi.binary_dilation(final, structure=disk(P.grouped_dilation)) & mask
            grouped_mask = remove_small_objects(grouped_mask, min_size=P.grouped_min_size)

    res.update(timings=T.times, total_time=T.total(), n_links=len(links), n_kinks_split=n_kinks,
               branch_table=btab, trace_table=ttab, polylines=polylines)
    if keep_arrays:
        res.update(field_theta=field[0] if field is not None else None, field_coh=field[1] if field is not None else None)
        res.update(merged=merged, resp=resp, sigma_best=sbest, nr=nr, nc=nc, mask=mask, bp_zone=bp_zone,
                   skeleton_raw=skel, skeleton=skel_clean, straight_skeleton=straight,
                   linked_skeleton=linked, traces=final, grouped_mask=grouped_mask, Hb=Hb_b, Gb=Gb_b)
    else:
        res.update(traces=final, merged=merged)
    if verbose:
        print(f"  total {T.total():.1f} s; traces: {len(ttab)}; links: {len(links)}; kinks split: {n_kinks}; trims: {res.get('trim_info')}")
    return res
