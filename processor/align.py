"""
Alignment on one bin grid: spike counts (pynapple TsGroup.count) or ROI
traces averaged per bin, behaviour resampled per bin, derived variables
(speed, run direction, linearised position, wheel velocity, keypoint speeds),
and the analysis epoch as a bin mask (pynapple IntervalSet).
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pynapple as nap
from scipy import sparse
from scipy.ndimage import gaussian_filter1d

from processor.nwbio import Series, safe_name

log = logging.getLogger("neuro-align")

CM_PER = {"cm": 1.0, "centimeters": 1.0, "centimeter": 1.0, "m": 100.0, "meters": 100.0, "meter": 100.0,
          "mm": 0.1, "millimeters": 0.1, "millimeter": 0.1}
RUN_SPEED_CM_S = 3.0
MIN_RUN_S = 0.5


@dataclass
class Grid:
    t0: float
    bin_size: float
    n: int

    @property
    def edges(self) -> np.ndarray:
        return self.t0 + self.bin_size * np.arange(self.n + 1)

    @property
    def centres(self) -> np.ndarray:
        return self.t0 + self.bin_size * (np.arange(self.n) + 0.5)

    @property
    def interval(self) -> nap.IntervalSet:
        return nap.IntervalSet(start=self.t0, end=self.t0 + self.bin_size * self.n)


@dataclass
class Var:
    name: str
    kind: str
    unit: str
    values: np.ndarray        # per bin
    source: str
    circular: bool = False
    note: str = ""


@dataclass
class Behaviour:
    vars: list[Var] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def get(self, name):
        return next((v for v in self.vars if v.name == name), None)

    def add(self, var: Var):
        if self.get(var.name) is None:
            self.vars.append(var)


def make_grid(t0: float, t1: float, bin_size: float) -> Grid:
    n = int(np.floor((t1 - t0) / bin_size + 1e-9))
    return Grid(float(t0), float(bin_size), max(n, 0))


# --------------------------------------------------------------------------------------------- neural

def bin_spikes(spikes: list[np.ndarray], grid: Grid, chunk: int = 64) -> np.ndarray:
    """Spike counts per bin (bins x units, float32), counted with pynapple in chunks of units so the
    int64 intermediate stays small."""
    X = np.zeros((grid.n, len(spikes)), dtype=np.float32)
    e = grid.edges
    for a in range(0, len(spikes), chunk):
        part = spikes[a:a + chunk]
        tsg = nap.TsGroup({i: nap.Ts(t=s) for i, s in enumerate(part)}, time_support=grid.interval)
        counts = tsg.count(bin_size=grid.bin_size, ep=grid.interval)
        c = np.asarray(counts.values)
        if c.shape[0] != grid.n or not np.allclose(np.asarray(counts.t), grid.centres, atol=grid.bin_size * 1e-3):
            # Floating-point edge effects: count exactly on our edges instead.
            c = np.stack([np.diff(np.searchsorted(s, e, side="left")) for s in part], axis=1)
        X[:, a:a + len(part)] = c
        del counts, c, tsg
    return X


def bin_matrix(t: np.ndarray, grid: Grid):
    idx = np.floor((t - grid.t0) / grid.bin_size).astype(np.int64)
    ok = (idx >= 0) & (idx < grid.n) & np.isfinite(t)
    return idx, ok


def resample(t: np.ndarray, v: np.ndarray, grid: Grid, circular: bool = False, max_gap: float | None = None) -> np.ndarray:
    """Values per bin: the mean of the samples in a bin when sampled faster than the bins, else linear
    interpolation at bin centres; NaN across gaps longer than max_gap (default max(0.5 s, 3 samples, 2 bins))."""
    v = np.asarray(v, dtype=float)
    squeeze = v.ndim == 1
    if squeeze:
        v = v[:, None]
    if circular:
        out = resample(t, np.c_[np.cos(v[:, 0]), np.sin(v[:, 0])], grid, False, max_gap)
        ang = np.mod(np.arctan2(out[:, 1], out[:, 0]), 2 * np.pi)
        return ang
    t = np.asarray(t, dtype=float)
    order = np.argsort(t, kind="stable")
    if np.any(np.diff(order) < 0):
        t, v = t[order], v[order]
    valid = np.all(np.isfinite(v), axis=1) & np.isfinite(t)
    out = np.full((grid.n, v.shape[1]), np.nan)
    if valid.sum() < 2 or grid.n == 0:
        return out[:, 0] if squeeze else out
    tv, vv = t[valid], v[valid]
    dt = float(np.median(np.diff(tv)))
    gap = max_gap if max_gap is not None else max(0.5, 3 * dt, 2 * grid.bin_size)
    c = grid.centres
    if dt < grid.bin_size:
        idx, ok = bin_matrix(tv, grid)
        cnt = np.bincount(idx[ok], minlength=grid.n).astype(float)
        for j in range(v.shape[1]):
            s = np.bincount(idx[ok], weights=vv[ok, j], minlength=grid.n)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[:, j] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        have = cnt > 0
        if (~have).any() and have.sum() >= 2:
            hc = c[have]
            for j in range(v.shape[1]):
                out[~have, j] = np.interp(c[~have], hc, out[have, j])
            # distance from each empty bin to the nearest sample
            pos = np.searchsorted(tv, c)
            prev = tv[np.clip(pos - 1, 0, len(tv) - 1)]
            nxt = tv[np.clip(pos, 0, len(tv) - 1)]
            far = (np.minimum(np.abs(c - prev), np.abs(nxt - c)) > gap / 2) | (nxt - prev > gap)
            out[~have & far] = np.nan
    else:
        for j in range(v.shape[1]):
            out[:, j] = np.interp(c, tv, vv[:, j])
        pos = np.searchsorted(tv, c)
        prev = tv[np.clip(pos - 1, 0, len(tv) - 1)]
        nxt = tv[np.clip(pos, 0, len(tv) - 1)]
        far = (nxt - prev > gap) & (pos > 0) & (pos < len(tv))
        out[far] = np.nan
    out[(c < tv[0] - grid.bin_size) | (c > tv[-1] + grid.bin_size)] = np.nan
    return out[:, 0] if squeeze else out


def resample_hold(t: np.ndarray, v: np.ndarray, grid: Grid) -> np.ndarray:
    """Zero-order hold at bin centres: the last sample at or before each centre. For event-driven samplers
    (a rotary encoder writes a sample only when the wheel turns), where the value holds between samples."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    out = np.full(grid.n, np.nan)
    if len(t) == 0:
        return out
    o = np.argsort(t, kind="stable")
    t, v = t[o], v[o]
    i = np.searchsorted(t, grid.centres, side="right") - 1
    good = i >= 0
    out[good] = v[i[good]]
    return out


def bin_traces(planes, grid: Grid) -> np.ndarray:
    """ROI traces averaged per bin (bins x ROIs, float32); every plane onto the same grid."""
    n_cols = sum(sl.stop - sl.start for _, _, sl in planes)
    X = np.full((grid.n, n_cols), np.nan, dtype=np.float32)
    for t, data, sl in planes:
        if sl.stop == sl.start:
            continue
        idx, ok = bin_matrix(t, grid)
        rows = idx[ok]
        S = sparse.csr_matrix((np.ones(len(rows)), (rows, np.flatnonzero(ok))), shape=(grid.n, len(t)))
        cnt = np.asarray(S.sum(axis=1)).ravel()
        d = np.nan_to_num(np.asarray(data, dtype=np.float64))
        sums = S @ d
        with np.errstate(invalid="ignore", divide="ignore"):
            means = sums / cnt[:, None]
        empty = cnt == 0
        if empty.any() and (~empty).sum() >= 2:
            # bins between two frames (bin smaller than a frame period): interpolate, not NaN
            c = grid.centres
            for j in range(means.shape[1]):
                means[empty, j] = np.interp(c[empty], c[~empty], means[~empty, j])
            means[(c < t[0]) | (c > t[-1])] = np.nan
        X[:, sl] = means.astype(np.float32)
    return X


# --------------------------------------------------------------------------------------------- behaviour

def nan_smooth(x: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaN (normalised convolution) and keeps NaN where there was no data."""
    if sigma_bins <= 0:
        return x.copy()
    m = np.isfinite(x)
    num = gaussian_filter1d(np.where(m, x, 0.0), sigma_bins, mode="nearest")
    den = gaussian_filter1d(m.astype(float), sigma_bins, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~m] = np.nan
    return out


def derivative(x: np.ndarray, bin_size: float, sigma_bins: float, max_step: float | None = None) -> np.ndarray:
    """d/dt of a per-bin signal after smoothing; steps larger than max_step per bin (teleports, wraps) -> NaN."""
    xs = nan_smooth(x, sigma_bins)
    d = np.gradient(xs) / bin_size
    if max_step is not None:
        raw = np.abs(np.r_[np.diff(x), 0.0])
        jump = raw > max_step
        jump = jump | np.r_[False, jump[:-1]]
        bad = gaussian_filter1d(jump.astype(float), max(sigma_bins, 1.0), mode="nearest") > 0.05
        d[bad] = np.nan
    return d


def clean_sentinels(v: np.ndarray) -> tuple[np.ndarray, int]:
    """1D position: a constant value at the minimum or maximum used for 'not on the track' (e.g. -500 during
    a virtual-reality inter-trial interval), far from the rest of the data, becomes NaN."""
    x = v.copy()
    fin = np.isfinite(x)
    if fin.sum() < 100:
        return x, 0
    n_bad = 0
    for _ in range(2):
        f = x[np.isfinite(x)]
        lo, hi = f.min(), f.max()
        for val in (lo, hi):
            at = np.isclose(x, val)
            if at.sum() < 0.005 * fin.sum():
                continue
            rest = f[~np.isclose(f, val)]
            if len(rest) < 10:
                continue
            rr = np.percentile(rest, 99.5) - np.percentile(rest, 0.5)
            dist = (np.percentile(rest, 0.5) - val) if val == lo else (val - np.percentile(rest, 99.5))
            if rr > 0 and dist > 0.2 * rr:
                n_bad += int(at.sum())
                x[at] = np.nan
    return x, n_bad


def linear_track(xy: np.ndarray, force: bool | None) -> tuple[np.ndarray | None, dict]:
    """Linearise 2D positions on a linear track: projection on the first principal axis, starting at 0."""
    ok = np.all(np.isfinite(xy), axis=1)
    info = {"detected": False}
    if ok.sum() < 50 or force is False:
        return None, info
    p = xy[ok]
    mu = p.mean(axis=0)
    cov = np.cov((p - mu).T)
    w, vec = np.linalg.eigh(cov)
    s1, s2 = np.sqrt(max(w[-1], 0)), np.sqrt(max(w[0], 0))
    ratio = float(s2 / s1) if s1 > 0 else 1.0
    axis = vec[:, -1]
    if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
        axis = -axis
    info.update(axis_ratio=round(ratio, 3), axis=[float(a) for a in axis])
    is_linear = ratio < 0.2
    if force is None and not is_linear:
        return None, info
    proj = (xy - mu) @ axis
    proj = proj - np.nanpercentile(proj, 0.5)
    info["detected"] = bool(is_linear)
    info["forced"] = bool(force)
    return proj, info


def to_cm_factor(unit: str) -> float | None:
    u = (unit or "").strip().lower().replace("/s", "").replace(" per second", "")
    return CM_PER.get(u)


def build_behaviour(series: list[Series], grid: Grid, params: dict) -> Behaviour:
    """Per-bin behaviour variables from the raw series of every behaviour file (first found wins)."""
    B = Behaviour()
    bs = grid.bin_size
    sig = max(params["SMOOTH_S"] / bs, 0.0)
    wanted = params["BEHAVIOR_VARS"]

    pos = [s for s in series if s.kind == "position"]
    multi = [s for s in pos if s.v.ndim > 1 and s.v.shape[1] >= 2]
    one = [s for s in pos if s.v.ndim == 1 or s.v.shape[1] == 1]
    kps = [s for s in series if s.kind == "keypoint"]
    pos_unit = ""
    if multi:
        s = multi[0]
        xy = resample(s.t, s.v[:, :2], grid)
        pos_unit = s.unit
        B.add(Var("x", "position_x", s.unit, xy[:, 0], s.source))
        B.add(Var("y", "position_y", s.unit, xy[:, 1], s.source))
    elif kps and len({s.container for s in kps}) == 1 and not any(s.kind.startswith("wheel") for s in series):
        # No Position, one camera, no wheel (a freely moving animal filmed from one view): the centroid of the
        # pose keypoints stands in for the position.
        cols = []
        centre = [s for s in kps if s.name.lower().replace("_", "").replace("-", "") in
                  ("bodycenter", "bodycentre", "centroid", "center", "centre", "midbody", "bodymid", "spine2", "mid")]
        for s in (centre[:1] or kps):
            v = s.v[:, :2].copy()
            if s.confidence is not None:
                v[s.confidence < params["POSE_MIN_CONFIDENCE"]] = np.nan
            cols.append(resample(s.t, v, grid))
        stack = np.stack(cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xy = np.nanmean(stack, axis=0)
        pos_unit = kps[0].unit
        src = f"pose keypoint '{centre[0].name}'" if centre else f"centroid of {len(kps)} pose keypoints"
        B.add(Var("x", "position_x", pos_unit, xy[:, 0], src))
        B.add(Var("y", "position_y", pos_unit, xy[:, 1], src))
        multi = [None]
    if one:
        s = one[0]
        v = s.v.ravel() if s.v.ndim == 1 else s.v[:, 0]
        v, n_bad = clean_sentinels(v)
        if n_bad:
            B.warnings.append(f"{s.name}: {n_bad} samples at an off-track constant value set to missing")
        lin = resample(s.t, v, grid)
        B.add(Var("position", "position", s.unit, lin, s.source, note="1D position from the file"))
        pos_unit = pos_unit or s.unit
        B.info["linear_track"] = {"source": "1D position in the file"}
    elif multi and params["LINEARIZE"] is not False:
        xy = np.c_[B.get("x").values, B.get("y").values]
        lin, info = linear_track(xy, params["LINEARIZE"])
        B.info["linear_track"] = info
        if lin is not None:
            B.add(Var("position", "position", pos_unit, lin, B.get("x").source,
                      note="linearised: projection on the track's long axis"))

    # Speed: a recorded running speed first, else from position.
    sp = [s for s in series if s.kind == "speed" and (s.v.ndim == 1 or s.v.shape[1] == 1)]
    if sp:
        s = sp[0]
        B.add(Var("speed", "speed", s.unit, resample(s.t, s.v.ravel(), grid), s.source))
    elif B.get("x") is not None:
        x, y = B.get("x").values, B.get("y").values
        rng = np.nanpercentile(np.r_[x, y], 99) - np.nanpercentile(np.r_[x, y], 1) if np.isfinite(x).any() else None
        step = 0.2 * rng if rng else None
        vx, vy = derivative(x, bs, sig, step), derivative(y, bs, sig, step)
        B.add(Var("speed", "speed", f"{pos_unit}/s" if pos_unit else "1/s", np.hypot(vx, vy), "derived from x, y"))
    elif B.get("position") is not None:
        p = B.get("position").values
        rng = np.nanpercentile(p, 99) - np.nanpercentile(p, 1) if np.isfinite(p).any() else None
        v = derivative(p, bs, sig, 0.2 * rng if rng else None)
        B.add(Var("speed", "speed", f"{pos_unit}/s" if pos_unit else "1/s", np.abs(v), "derived from position"))

    # Run direction on a linear track (+1 increasing position, -1 decreasing, 0 still).
    if B.get("position") is not None and B.get("speed") is not None:
        p = B.get("position").values
        rng = np.nanpercentile(p, 99) - np.nanpercentile(p, 1) if np.isfinite(p).any() else None
        v = derivative(p, bs, sig, 0.2 * rng if rng else None)
        thr = run_threshold(B.get("speed"), params)
        d = np.where(np.isfinite(v) & (np.abs(v) >= thr), np.sign(v), 0.0)
        d[~np.isfinite(p)] = np.nan
        B.add(Var("run_direction", "direction", "sign", d, "sign of d(position)/dt while running",
                  note=f"+1 / -1 when |velocity| >= {thr:g}, else 0"))

    ang = [s for s in series if s.kind == "angle"]
    if ang:
        s = ang[0]
        a = s.v.ravel() if s.v.ndim == 1 else s.v[:, 0]
        deg = "deg" in (s.unit or "").lower() or np.nanmax(np.abs(a)) > 2 * np.pi + 0.5
        a = np.deg2rad(a) if deg else a
        B.add(Var("head_direction", "angle", "rad", resample(s.t, np.mod(a, 2 * np.pi), grid, circular=True),
                  s.source, circular=True, note="converted from degrees" if deg else ""))

    wpos = [s for s in series if s.kind == "wheel_position"]
    wvel = [s for s in series if s.kind == "wheel_velocity"]
    if wpos:
        s = wpos[0]
        w = resample_hold(s.t, s.v.ravel() if s.v.ndim == 1 else s.v[:, 0], grid)
        B.add(Var("wheel_position", "wheel_position", s.unit, w, s.source))
        if not wvel:
            B.add(Var("wheel_velocity", "wheel_velocity", f"{s.unit}/s" if s.unit else "1/s",
                      derivative(w, bs, max(params["SMOOTH_S"] / 2 / bs, 0.5)), "derived from wheel position"))
    if wvel:
        s = wvel[0]
        B.add(Var("wheel_velocity", "wheel_velocity", s.unit, resample(s.t, s.v.ravel(), grid), s.source))

    pup = [s for s in series if s.kind == "pupil" and (s.v.ndim == 1 or s.v.shape[1] == 1)]
    if pup:
        pup.sort(key=lambda s: (0 if "diam" in s.name.lower() else 1 if "area" in s.name.lower()
                                else 2 if "radius" in s.name.lower() else 3, "smooth" not in s.name.lower()))
        s = pup[0]
        B.add(Var("pupil", "pupil", s.unit, resample(s.t, s.v.ravel(), grid), s.source, note=s.name))

    licks = [s for s in series if s.kind == "licks"]
    if licks:
        s = licks[0]
        idx, ok = bin_matrix(s.t, grid)
        cnt = np.bincount(idx[ok], minlength=grid.n).astype(float)
        B.add(Var("licks", "licks", "count per bin", cnt, s.source))

    # Keypoint speeds (ndx-pose): confidence-masked, gaps interpolated, smoothed.
    # Pupil keypoints (pupil size is its own variable) are not used; a keypoint tracked by several
    # cameras is taken from the camera with the highest frame rate.
    kps_use = []
    for s in sorted((s for s in kps if "pupil" not in s.name.lower()),
                    key=lambda s: -(1 / np.median(np.diff(s.t)) if len(s.t) > 1 else 0)):
        if s.name not in [k.name for k in kps_use]:
            kps_use.append(s)
    names = [s.name for s in kps_use]
    for s in kps_use:
        nm = s.name if names.count(s.name) == 1 else f"{s.container}_{s.name}"
        v = s.v[:, :2].copy()
        if s.confidence is not None:
            v[s.confidence < params["POSE_MIN_CONFIDENCE"]] = np.nan
        xy = resample(s.t, v, grid)
        if not np.isfinite(xy).any():
            continue
        span = np.nanpercentile(xy, 99, axis=0) - np.nanpercentile(xy, 1, axis=0)
        if np.all(span < 10):
            B.info.setdefault("static_keypoints", []).append(nm)   # a landmark (lick spout), not a body part
            continue
        if np.isfinite(xy[:, 0]).mean() < 0.5:
            B.info.setdefault("rarely_tracked_keypoints", []).append(nm)   # below confidence most of the time
            continue
        rng = float(np.max(span))
        step = 0.5 * rng if rng else None
        spd = np.hypot(derivative(xy[:, 0], bs, sig, step), derivative(xy[:, 1], bs, sig, step))
        B.add(Var(f"{safe_name(nm)}_speed", "keypoint_speed", f"{s.unit}/s" if s.unit else "1/s", spd, s.source))
    for s in [s for s in series if s.kind == "keypoint_speed"]:
        B.add(Var(safe_name(s.name), "keypoint_speed", s.unit, resample(s.t, s.v.ravel() if s.v.ndim == 1 else s.v[:, 0], grid), s.source))

    # Other series: only when asked for by name.
    for s in series:
        nm = safe_name(s.name)
        if s.kind != "keypoint" and not s.events and nm in wanted["extra"] and B.get(nm) is None:
            if s.v.ndim == 1 or s.v.shape[1] == 1:
                B.add(Var(nm, "other", s.unit, resample(s.t, s.v.ravel(), grid), s.source))
            else:
                for j in range(min(s.v.shape[1], 3)):
                    B.add(Var(f"{nm}_{j}", "other", s.unit, resample(s.t, s.v[:, j], grid), s.source))

    # Selection: auto keeps everything recognised; a list keeps those named (plus 'keypoint_speeds' = all).
    if not wanted["auto"]:
        keep = set(wanted["names"]) | set(wanted["extra"])
        missing = [n for n in wanted["names"] if n != "keypoint_speeds" and B.get(n) is None]
        if missing:
            B.warnings.append(f"BEHAVIOR_VARS not found in this session: {', '.join(missing)}")
        B.vars = [v for v in B.vars if v.name in keep or (v.kind == "keypoint_speed" and "keypoint_speeds" in keep)]
    # Drop variables with no data in the grid.
    empty = [v.name for v in B.vars if not np.isfinite(v.values).any()]
    if empty:
        B.warnings.append(f"no data inside the aligned window for: {', '.join(empty)}")
        B.vars = [v for v in B.vars if v.name not in empty]
    return B


def run_threshold(speed: Var, params: dict) -> float:
    if params["RUN_SPEED"] is not None:
        return float(params["RUN_SPEED"])
    f = to_cm_factor(speed.unit)
    if f:
        return RUN_SPEED_CM_S / f
    v = speed.values[np.isfinite(speed.values)]
    return float(0.1 * np.percentile(v, 95)) if len(v) else 0.0


def mask_to_intervals(mask: np.ndarray, grid: Grid) -> nap.IntervalSet:
    m = np.r_[False, mask.astype(bool), False]
    d = np.diff(m.astype(int))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    if len(starts) == 0:
        return nap.IntervalSet(start=[], end=[])
    return nap.IntervalSet(start=grid.t0 + starts * grid.bin_size, end=grid.t0 + ends * grid.bin_size)


def intervals_to_mask(iset: nap.IntervalSet, grid: Grid) -> np.ndarray:
    if len(iset) == 0:
        return np.zeros(grid.n, dtype=bool)
    c = grid.centres
    st, en = np.asarray(iset.start), np.asarray(iset.end)
    i = np.searchsorted(st, c, side="right") - 1
    ok = i >= 0
    out = np.zeros(grid.n, dtype=bool)
    out[ok] = c[ok] < en[i[ok]]
    return out


def run_epoch(speed: Var, grid: Grid, params: dict) -> tuple[np.ndarray, float]:
    thr = run_threshold(speed, params)
    m = np.isfinite(speed.values) & (speed.values >= thr)
    iset = mask_to_intervals(m, grid)
    if len(iset):
        iset = iset.merge_close_intervals(MIN_RUN_S).drop_short_intervals(MIN_RUN_S)
    return intervals_to_mask(iset, grid), thr


def table_epoch(df, grid: Grid) -> np.ndarray:
    st = np.asarray(df["start_time"], dtype=float)
    en = np.asarray(df["stop_time"], dtype=float)
    ok = np.isfinite(st) & np.isfinite(en) & (en > st)
    if not ok.any():
        return np.zeros(grid.n, dtype=bool)
    iset = nap.IntervalSet(start=st[ok], end=en[ok])   # pynapple sorts and merges overlaps
    return intervals_to_mask(iset, grid)
