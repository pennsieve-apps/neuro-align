"""
Reading NWB with pynwb (+ ndx-pose): the neural side (Units or ophys ROI
traces), the behaviour side (Position, CompassDirection, BehavioralTimeSeries,
PupilTracking, BehavioralEvents, IBL wheel, ndx-pose PoseEstimation) and every
TimeIntervals table (trials and others) as events.

Everything comes back as plain numpy arrays in session-clock seconds; the
binning (processor.align) builds the pynapple objects.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("neuro-align")

DAY = 86400.0
SPEED_NAMES = {"speed", "running_speed", "run_speed", "runningspeed", "velocity", "running_velocity",
               "linear_velocity", "locomotion", "body_speed", "centroid_speed", "treadmill_speed", "ball_speed"}
ANGLE_NAMES = {"heading", "head_direction", "headdirection", "hd", "head_angle", "orientation", "azimuth",
               "head_orientation"}
POSITION_NAMES = {"position", "pos", "linear_position", "linearized_position", "linearised_position", "linpos",
                  "track_position", "position_on_track", "lin_pos", "centroid", "xy", "location"}
SKIP_ACQ_TYPES = {"ElectricalSeries", "ImageSeries", "TwoPhotonSeries", "OnePhotonSeries", "OpticalSeries",
                  "PatchClampSeries", "CurrentClampSeries", "VoltageClampSeries"}
SKIP_MODULES = {"ophys", "ecephys", "icephys"}


@dataclass
class Neural:
    kind: str                      # "units" | "roi"
    source: str                    # e.g. "units", "ophys/Deconvolved/plane0"
    signal: str                    # "spike counts" | "deconvolved" | "dff" | "fluorescence"
    ids: list[str]
    meta: list[dict]               # region, label, depth, x, y, file
    spikes: list[np.ndarray] | None = None                 # units
    planes: list[tuple[np.ndarray, np.ndarray, slice]] | None = None   # roi: (t, data T x n, columns)
    dropped: dict = field(default_factory=dict)
    dropped_regions: dict = field(default_factory=dict)   # region -> neurons dropped
    filters: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    frame_period: float | None = None

    def t_range(self) -> tuple[float, float]:
        if self.kind == "units":
            firsts = [s[0] for s in self.spikes if len(s)]
            lasts = [s[-1] for s in self.spikes if len(s)]
            return (float(min(firsts)), float(max(lasts))) if firsts else (np.nan, np.nan)
        return (float(min(p[0][0] for p in self.planes)), float(max(p[0][-1] for p in self.planes)))


@dataclass
class Series:
    name: str          # NWB object name
    kind: str          # position | angle | speed | pupil | licks | wheel_position | wheel_velocity | keypoint | other
    t: np.ndarray
    v: np.ndarray      # (T,) or (T, D); for licks: event times in t, v = ones
    unit: str
    source: str        # file::path in the NWB
    container: str = ""
    confidence: np.ndarray | None = None
    events: bool = False


def _s(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    return "" if x is None else str(x)


def series_times(ts, n: int, warnings: list[str], label: str) -> np.ndarray:
    """Timestamps of a TimeSeries; starting_time + n / rate when it has none (with a rate sanity check)."""
    if ts.timestamps is not None:
        t = ts.get_timestamps() if hasattr(ts, "get_timestamps") else ts.timestamps
        return np.asarray(t[:] if not isinstance(t, np.ndarray) else t, dtype=float)
    rate = float(ts.rate)
    t0 = float(ts.starting_time or 0.0)
    if 0 < rate < 1 and n / rate > 2 * DAY and n * rate < 2 * DAY:
        warnings.append(f"{label}: rate attribute {rate:g} Hz would make {n} samples last {n / rate / DAY:.0f} days; "
                        f"it is the sampling period, used {1 / rate:g} Hz")
        rate = 1.0 / rate
    return t0 + np.arange(n, dtype=float) / rate


def series_data(ts) -> np.ndarray:
    d = np.asarray(ts.data[:])
    if d.dtype.kind in "fiu":
        conv = float(getattr(ts, "conversion", 1.0) or 1.0)
        off = float(getattr(ts, "offset", 0.0) or 0.0)
        d = d.astype(float)
        if conv != 1.0 or off != 0.0:
            d = d * conv + off
    return d


# --------------------------------------------------------------------------------------------- neural

LABEL_COLUMNS = ("label", "quality", "ibl_quality_score", "KSLabel", "kslabel", "ks_label", "bombcell_label", "phy_annotations")
REGION_COLUMNS = ("brain_area", "brain_region", "location", "region", "acronym", "allen_ccf_acronym", "area")
DEPTH_COLUMNS = ("depth", "depth_um", "cluster_depths", "peak_depth", "depths", "distance_from_probe_tip_um", "y")


def _column(table, name):
    try:
        if name not in table.colnames:
            return None
        col = table[name]
    except (KeyError, AttributeError):
        return None
    if type(col).__name__ == "VectorIndex":
        return None   # ragged
    try:
        d = col.data[:]
    except Exception:
        return None
    d = np.asarray(d)
    if d.ndim > 1:
        d = d[:, 0] if d.shape[1] >= 1 else d.ravel()
    return d


def _ragged(col):
    """(offsets, flat values) of a VectorIndex column."""
    ends = np.asarray(col.data[:], dtype=np.int64)
    return ends, col.target


def unit_good_mask(units, accepted: set[str] | None, warnings: list[str]):
    """Which units pass the label filter (accepted labels; None = no filter), and each unit's label."""
    n = len(units.id)
    labels = np.array([""] * n, dtype=object)
    good = np.ones(n, dtype=bool)
    used = None
    for name in LABEL_COLUMNS:
        d = _column(units, name)
        if d is None:
            continue
        used = name
        if d.dtype.kind in "OSU":
            labels = np.array([_s(x).strip() for x in d], dtype=object)
            if accepted is not None:
                good = np.array([x.lower() in accepted for x in labels])
        else:
            d = d.astype(float)
            if name == "phy_annotations":
                # Steinmetz et al. 2019: 1 = MUA, 2 = good, 3 = unsorted; >= 2 are the units to analyse.
                labels = np.where(d >= 2, "good", np.where(d == 1, "mua", "noise")).astype(object)
            elif name in ("label", "ibl_quality_score") and np.nanmax(d) <= 1.0 + 1e-9:
                # IBL: label = fraction of the single-unit metrics passed; 1 = good.
                labels = np.where(np.isclose(d, 1.0), "good", "mua").astype(object)
            else:
                labels = np.array([f"{x:g}" for x in d], dtype=object)
            if accepted is not None:
                good = np.array([str(x).lower() in accepted for x in labels])
        break
    if accepted is not None and used is not None and not good.any():
        warnings.append(f"no unit has a {used} in {sorted(accepted)}; kept all units (set UNIT_LABELS=all to silence)")
        good[:] = True
        used = f"{used} (ignored: nothing passed)"
    return good, labels, used


def unit_regions(nwb, units) -> np.ndarray:
    n = len(units.id)
    for name in REGION_COLUMNS:
        d = _column(units, name)
        if d is not None and d.dtype.kind in "OSU":
            vals = np.array([_s(x) for x in d], dtype=object)
            if any(vals):
                return vals
    regions = np.array([""] * n, dtype=object)
    try:
        if "electrodes" in units.colnames and nwb.electrodes is not None and "location" in nwb.electrodes.colnames:
            loc = np.array([_s(x) for x in nwb.electrodes["location"].data[:]], dtype=object)
            col = units["electrodes"]
            if type(col).__name__ == "VectorIndex":
                ends, target = _ragged(col)
                rows = np.asarray(target.data[:], dtype=np.int64)
                starts = np.r_[0, ends[:-1]]
                for i in range(n):
                    r = rows[starts[i]:ends[i]]
                    vals = [loc[j] for j in r if 0 <= j < len(loc) and loc[j]]
                    if vals:
                        u, c = np.unique(vals, return_counts=True)
                        regions[i] = u[np.argmax(c)]
            else:
                rows = np.asarray(col.data[:], dtype=np.int64)
                regions = np.array([loc[j] if 0 <= j < len(loc) else "" for j in rows], dtype=object)
    except Exception as exc:   # region is optional metadata
        log.debug("regions from electrodes failed: %s", exc)
    if not any(regions):
        try:
            if "electrode_group" in units.colnames:
                regions = np.array([_s(getattr(g, "location", "")) for g in units["electrode_group"].data[:]], dtype=object)
        except Exception:
            pass
    return regions


def read_units(nwb, file_label: str, accepted: set[str] | None) -> Neural:
    units = nwb.units
    warnings: list[str] = []
    ids = np.asarray(units.id.data[:])
    st = units["spike_times"]
    ends, target = _ragged(st)
    flat = np.asarray(target.data[:], dtype=float)
    starts = np.r_[0, ends[:-1]]
    good, labels, used = unit_good_mask(units, accepted, warnings)
    regions = unit_regions(nwb, units)
    depth = None
    for name in DEPTH_COLUMNS:
        d = _column(units, name)
        if d is not None and d.dtype.kind in "fiu":
            depth = d.astype(float)
            break
    ctype = None
    for name in ("cell_type", "celltype", "putative_cell_type", "cell_class"):
        d = _column(units, name)
        if d is not None and d.dtype.kind in "OSU":
            ctype = np.array([_s(x) for x in d], dtype=object)
            break
    spikes, meta, keep_ids = [], [], []
    dropped_regions: dict = {}
    n_label = 0
    for i in range(len(ids)):
        if not good[i]:
            n_label += 1
            r = _s(regions[i]) or "unknown"
            dropped_regions[r] = dropped_regions.get(r, 0) + 1
            continue
        s = np.sort(flat[starts[i]:ends[i]])
        spikes.append(s)
        keep_ids.append(str(ids[i]))
        meta.append({"region": _s(regions[i]) or "unknown", "label": _s(labels[i]),
                     "depth": float(depth[i]) if depth is not None else np.nan, "x": np.nan, "y": np.nan,
                     "cell_type": _s(ctype[i]) if ctype is not None else "", "file": file_label})
    nu = Neural(kind="units", source="units", signal="spike counts", ids=keep_ids, meta=meta, spikes=spikes,
                warnings=warnings)
    nu.dropped["label"] = n_label
    nu.dropped_regions = dropped_regions
    nu.filters["label_column"] = used
    nu.filters["accepted_labels"] = sorted(accepted) if accepted is not None else "all"
    nu.filters["n_units_in_file"] = int(len(ids))
    return nu


ROI_KINDS = ("deconvolved", "dff", "fluorescence")


def _roi_rank(name: str, typ: str) -> str | None:
    n = name.lower()
    if any(k in n for k in ("neuropil", "background", "fneu")):
        return None
    if any(k in n for k in ("deconv", "spks", "spike", "inferred", "cascade")):
        return "deconvolved"
    if typ == "DfOverF" or any(k in n for k in ("dff", "df_over_f", "dfoverf", "deltaf")):
        return "dff"
    if typ == "Fluorescence":
        return "fluorescence"
    return None


def _series_rank(name: str) -> str | None:
    n = name.lower()
    if any(k in n for k in ("residual", "neuropil", "background", "raw", "fneu")):
        return "skip"
    if n in ("s", "spks") or any(k in n for k in ("deconv", "spike", "event", "inferred", "cascade", "oasis")):
        return "deconvolved"
    if n in ("c", "denoised", "dff", "df_f") or "dff" in n or "denois" in n:
        return "dff"
    return None


def _same_rois(series) -> bool:
    try:
        # imaging planes each have their own PlaneSegmentation; series over one table are alternative signals
        first = series[0].rois
        return all(r.rois.table is first.table for r in series[1:])
    except Exception:
        return False


def _roi_centroids(ps, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(rows)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    try:
        if "pixel_mask" in ps.colnames:
            ends, target = _ragged(ps["pixel_mask"])
            pm = target.data[:]
            names = pm.dtype.names or ()
            px = np.asarray(pm["x"] if "x" in names else pm[:, 0], dtype=float)
            py = np.asarray(pm["y"] if "y" in names else pm[:, 1], dtype=float)
            w = np.asarray(pm["weight"] if "weight" in names else np.ones(len(px)), dtype=float)
            starts = np.r_[0, ends[:-1]]
            for k, r in enumerate(rows):
                sl = slice(starts[r], ends[r])
                ww = w[sl]
                if ww.sum() > 0:
                    x[k] = float((px[sl] * ww).sum() / ww.sum())
                    y[k] = float((py[sl] * ww).sum() / ww.sum())
        elif "image_mask" in ps.colnames:
            im = ps["image_mask"].data
            if np.prod(im.shape) * 4 < 400e6:
                arr = np.asarray(im[:], dtype=float)
                yy, xx = np.mgrid[: arr.shape[1], : arr.shape[2]]
                for k, r in enumerate(rows):
                    m = arr[r]
                    if m.sum() > 0:
                        x[k] = float((xx * m).sum() / m.sum())
                        y[k] = float((yy * m).sum() / m.sum())
    except Exception as exc:
        log.debug("ROI centroids failed: %s", exc)
    return x, y


def _roi_accepted(ps, rows: np.ndarray) -> tuple[np.ndarray, str | None]:
    for name in ("accepted", "iscell", "is_cell", "accept"):
        if name in ps.colnames:
            try:
                d = np.asarray(ps[name].data[:], dtype=float)
            except Exception:
                continue
            if d.ndim > 1:
                d = d[:, 0]
            return d[rows] > 0.5, name
    return np.ones(len(rows), dtype=bool), None


def find_roi_containers(nwb):
    out = []
    for mname, mod in nwb.processing.items():
        for name, di in mod.data_interfaces.items():
            typ = type(di).__name__
            if typ not in ("Fluorescence", "DfOverF"):
                continue
            kind = _roi_rank(name, typ)
            if kind is not None:
                out.append((kind, mname, name, di))
    return out


def read_rois(nwb, file_label: str, trace: str) -> Neural:
    conts = find_roi_containers(nwb)
    if not conts:
        raise ValueError("no Fluorescence / DfOverF ROI traces")
    # Several series over the same ROIs in one container are alternative signals (CaImAn / Minian 'C',
    # 'events', 'residual'), not imaging planes: each is a candidate of its own.
    cands = []
    for ckind, mname, cname, cont in conts:
        series = list(cont.roi_response_series.items())
        if len(series) > 1 and _same_rois([r for _, r in series]):
            for sname, rrs in series:
                sk = _series_rank(sname)
                if sk == "skip":
                    continue
                cands.append((sk or ckind, mname, f"{cname}/{sname}", cont, [(sname, rrs)]))
        else:
            cands.append((ckind, mname, cname, cont, series))
    order = list(ROI_KINDS) if trace == "auto" else [trace]
    chosen = None
    for kind in order:
        chosen = next((c for c in cands if c[0] == kind), None)
        if chosen:
            break
    if chosen is None:
        raise ValueError(f"no {trace} ROI traces (found: {', '.join(f'{c[1]}/{c[2]}' for c in conts)})")
    kind, mname, cname, cont, series = chosen
    warnings: list[str] = []
    planes, ids, meta = [], [], []
    multi = len(series) > 1
    n_total = n_rej = 0
    dropped_regions: dict = {}
    acc_col = None
    periods = []
    col0 = 0
    for sname, rrs in series:
        data = rrs.data
        t = series_times(rrs, data.shape[0], warnings, f"{file_label}::{mname}/{cname}/{sname}")
        arr = np.asarray(data[:], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] != len(t) and arr.shape[1] == len(t):
            warnings.append(f"{cname}/{sname}: data stored as ROIs x time; transposed")
            arr = arr.T
        rows = np.asarray(rrs.rois.data[:], dtype=np.int64)
        ps = rrs.rois.table
        roi_ids = np.asarray(ps.id.data[:])[rows]
        acc, acc_col = _roi_accepted(ps, rows)
        n_total += len(rows)
        n_rej += int((~acc).sum())
        x, y = _roi_centroids(ps, rows)
        region = ""
        try:
            region = _s(ps.imaging_plane.location)
        except Exception:
            pass
        keep = np.flatnonzero(acc)
        if (~acc).any():
            dropped_regions[region or "unknown"] = dropped_regions.get(region or "unknown", 0) + int((~acc).sum())
        arr = arr[:, keep]
        planes.append((t, arr, slice(col0, col0 + len(keep))))
        col0 += len(keep)
        periods.append(float(np.median(np.diff(t))) if len(t) > 1 else np.nan)
        for k in keep:
            ids.append(f"{sname}_{roi_ids[k]}" if multi else str(roi_ids[k]))
            meta.append({"region": region or "unknown", "label": "accepted" if acc_col else "",
                         "depth": np.nan, "x": float(x[k]), "y": float(y[k]), "cell_type": "", "file": file_label})
    nr = Neural(kind="roi", source=f"{mname}/{cname}", signal=kind, ids=ids, meta=meta, planes=planes,
                warnings=warnings, frame_period=float(np.nanmax(periods)) if periods else None)
    nr.dropped["not_accepted"] = n_rej
    nr.dropped_regions = dropped_regions
    nr.filters["accepted_column"] = acc_col
    nr.filters["n_rois_in_file"] = n_total
    nr.filters["trace"] = f"{mname}/{cname} ({kind})"
    nr.filters["planes"] = [s for s, _ in series]
    return nr


# --------------------------------------------------------------------------------------------- behaviour

def classify(name: str, container_type: str) -> str:
    n = name.lower()
    if container_type == "Position":
        return "position"
    if container_type == "CompassDirection":
        return "angle"
    if container_type == "PupilTracking" or "pupil" in n:
        return "pupil"
    if "lick" in n:
        return "licks"
    if "acc" in n:
        return "other"
    if "wheel" in n:
        return "wheel_velocity" if any(k in n for k in ("vel", "speed")) else "wheel_position"
    if n in SPEED_NAMES:
        return "speed"
    if n in ANGLE_NAMES or "head_dir" in n or "heading" in n:
        return "angle"
    if n in POSITION_NAMES:
        return "position"
    if n.endswith("_speed"):
        return "keypoint_speed"
    return "other"


def _iter_series(cont):
    for attr in ("spatial_series", "time_series"):
        d = getattr(cont, attr, None)
        if d:
            yield from d.items()


def read_behavior(nwb, file_label: str, warnings: list[str]) -> tuple[list[Series], list]:
    """All behaviour time series of one file, and its TimeIntervals-like tables as DataFrames."""
    import pandas as pd

    out: list[Series] = []
    tables = []

    def add_ts(ts, name, ctype, where):
        try:
            n = ts.data.shape[0]
            if n == 0:
                return
            kind = classify(name, ctype)
            label = f"{file_label}::{where}"
            if kind == "licks" or ctype == "BehavioralEvents":
                d = np.asarray(ts.data[:])
                t = series_times(ts, n, warnings, label)
                if kind != "licks":
                    tables.append((name, pd.DataFrame({"start_time": t, "stop_time": t})))
                    return
                if d.dtype.kind == "b":
                    ev = t[d.astype(bool)]
                else:
                    d = d.astype(float)
                    if d.ndim > 1 or np.any(np.abs(d - np.round(d)) > 1e-6) or n > 2e6:
                        return   # an analog lick sensor, not lick events
                    if np.all(np.isin(d, (0, 1))) and d.sum() == n:
                        ev = t
                    else:
                        reps = np.clip(np.round(d), 0, 1000).astype(int)
                        ev = np.repeat(t, reps)
                out.append(Series(name, "licks", ev, np.ones(len(ev)), "licks", label, ctype, events=True))
                return
            d = series_data(ts)
            if d.dtype.kind not in "fiub":
                return
            t = series_times(ts, n, warnings, label)
            unit = _s(getattr(ts, "unit", ""))
            out.append(Series(name, kind, t, d.astype(float), unit, label, ctype))
        except Exception as exc:
            warnings.append(f"{file_label}::{where}: unreadable ({type(exc).__name__}: {exc}), skipped")

    def add_pose(pe, cname, where):
        for kp, pes in pe.pose_estimation_series.items():
            try:
                n = pes.data.shape[0]
                t = series_times(pes, n, warnings, f"{file_label}::{where}/{kp}")
                d = series_data(pes)
                conf = np.asarray(pes.confidence[:], dtype=float) if pes.confidence is not None else None
                kpn = kp[len("PoseEstimationSeries"):] if kp.startswith("PoseEstimationSeries") and len(kp) > 20 else kp
                out.append(Series(kpn, "keypoint", t, d, _s(pes.unit), f"{file_label}::{where}/{kp}", cname,
                                  confidence=conf))
            except Exception as exc:
                warnings.append(f"{file_label}::{where}/{kp}: unreadable ({exc}), skipped")

    def add_events(ev, name, where):
        """ndx-events Events (timestamps only): licks become the licks variable, others point events."""
        try:
            t = np.asarray(ev.timestamps[:], dtype=float)
            if "lick" in name.lower():
                out.append(Series(name, "licks", t, np.ones(len(t)), "licks", f"{file_label}::{where}", "Events",
                                  events=True))
            else:
                tables.append((name, pd.DataFrame({"start_time": t, "stop_time": t})))
        except Exception as exc:
            warnings.append(f"{file_label}::{where}: unreadable ({exc}), skipped")

    def add_intervals(ti, name):
        try:
            tables.append((name, intervals_frame(ti)))
        except Exception as exc:
            warnings.append(f"{file_label}: table {name} unreadable ({exc}), skipped")

    mods = sorted(nwb.processing.items(), key=lambda kv: (kv[0] != "behavior", kv[0]))
    for mname, mod in mods:
        if mname.lower() in SKIP_MODULES:
            continue
        for name, di in mod.data_interfaces.items():
            ctype = type(di).__name__
            where = f"processing/{mname}/{name}"
            if ctype == "PoseEstimation":
                add_pose(di, name, where)
            elif ctype == "BehavioralEpochs":
                for sname, isr in di.interval_series.items():
                    try:
                        d = np.asarray(isr.data[:])
                        if isr.timestamps is not None and len(isr.timestamps) == 2 * len(d):
                            t = np.asarray(isr.timestamps[:], dtype=float)   # (start, stop) pairs
                            st, en = t[0::2], t[1::2]
                        else:
                            t = series_times(isr, len(d), warnings, where)
                            st, en = t[d > 0], t[d < 0]
                        m = min(len(st), len(en))
                        tables.append((sname, pd.DataFrame({"start_time": st[:m], "stop_time": en[:m]})))
                    except Exception as exc:
                        warnings.append(f"{file_label}::{where}/{sname}: unreadable ({exc}), skipped")
            elif ctype == "TimeIntervals":
                add_intervals(di, name)
            elif ctype in ("Position", "CompassDirection", "BehavioralTimeSeries", "PupilTracking",
                           "BehavioralEvents", "EyeTracking"):
                for sname, ts in _iter_series(di):
                    add_ts(ts, sname, ctype, f"{where}/{sname}")
            elif ctype in ("TimeSeries", "SpatialSeries"):
                add_ts(di, name, ctype, where)
            elif hasattr(di, "pose_estimation_series"):
                add_pose(di, name, where)
            elif getattr(di, "timestamps", None) is not None and getattr(di, "data", None) is None:
                add_events(di, name, where)
    for name, obj in nwb.acquisition.items():
        ctype = type(obj).__name__
        if ctype in SKIP_ACQ_TYPES:
            continue
        where = f"acquisition/{name}"
        if ctype in ("Position", "CompassDirection", "BehavioralTimeSeries", "PupilTracking", "BehavioralEvents"):
            for sname, ts in _iter_series(obj):
                add_ts(ts, sname, ctype, f"{where}/{sname}")
        elif ctype in ("TimeSeries", "SpatialSeries") and classify(name, ctype) != "other":
            add_ts(obj, name, ctype, where)
        elif ctype == "PoseEstimation":
            add_pose(obj, name, where)
    if nwb.intervals:
        for name, ti in nwb.intervals.items():
            add_intervals(ti, name)
    # NWB 2.9+ EventsTable (pynwb >= 4): timestamp [+ duration] rows
    for name, et in (getattr(nwb, "events", None) or {}).items():
        try:
            df = intervals_frame_cols(et)
            ts = np.asarray(df.pop("timestamp"), dtype=float)
            if "lick" in name.lower():
                out.append(Series(name, "licks", ts, np.ones(len(ts)), "licks", f"{file_label}::events/{name}",
                                  "EventsTable", events=True))
                continue
            dur = np.asarray(df.pop("duration"), dtype=float) if "duration" in df else np.zeros(len(ts))
            df.insert(0, "start_time", ts)
            df.insert(1, "stop_time", ts + np.nan_to_num(dur))
            tables.append((name, df))
        except Exception as exc:
            warnings.append(f"{file_label}: events table {name} unreadable ({exc}), skipped")
    return out, tables


def intervals_frame(ti):
    """A TimeIntervals table as a DataFrame of scalar columns (ragged string columns joined with ';')."""
    return intervals_frame_cols(ti)


def intervals_frame_cols(ti):
    import pandas as pd

    cols = {}
    for name in ti.colnames:
        col = ti[name]
        try:
            if type(col).__name__ == "VectorIndex":
                ends = np.asarray(col.data[:], dtype=np.int64)
                vals = col.target.data[:]
                if len(vals) and isinstance(vals[0], (str, bytes)):
                    starts = np.r_[0, ends[:-1]]
                    cols[name] = [";".join(_s(v) for v in vals[a:b]) for a, b in zip(starts, ends)]
                continue
            if type(col).__name__ == "DynamicTableRegion":
                continue
            d = col.data[:]
            d = np.asarray(d)
            if d.ndim != 1:
                continue
            if d.dtype.kind in "OSU":
                d = np.array([_s(x) for x in d], dtype=object)
            cols[name] = d
        except Exception:
            continue
    df = pd.DataFrame(cols)
    return df


def reference_time(nwb):
    t = getattr(nwb, "timestamps_reference_time", None) or getattr(nwb, "session_start_time", None)
    return t


def identity(nwb) -> dict:
    out = {"identifier": _s(nwb.identifier), "session_description": _s(nwb.session_description),
           "session_start_time": nwb.session_start_time.isoformat() if nwb.session_start_time else None,
           "timestamps_reference_time": nwb.timestamps_reference_time.isoformat() if nwb.timestamps_reference_time else None}
    if nwb.subject is not None:
        out["subject"] = {k: _s(getattr(nwb.subject, k, None)) for k in ("subject_id", "species", "sex", "age", "genotype")
                          if getattr(nwb.subject, k, None) is not None}
    return out


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower() or "var"
