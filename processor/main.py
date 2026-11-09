"""
neuro-align: neural activity and behaviour of a session on one time grid.

Reads NWB files from the input ports `neural` (Units with spike_times, or
processing/ophys ROI traces) and `behavior` (Position, CompassDirection,
BehavioralTimeSeries, PupilTracking, BehavioralEvents, IBL wheel, ndx-pose
PoseEstimation, TimeIntervals; may be the same file as `neural`), plus an
optional sessions.csv, and writes per session (conventions.md, "Decoding"):

  neuro-align/<session_id>/aligned.npz     t, X (bins x neurons), Y (bins x behaviour), neuron_ids,
                                           neuron_meta, behavior_names/units/kinds, epoch_mask, bin_size
  neuro-align/<session_id>/spikes.npz      spike times per unit (units only)
  neuro-align/<session_id>/events.parquet  every TimeIntervals row (trials, ...) and point events
  neuro-align/<session_id>/manifest.json   bin size, variables (units, sources), filters, clock checks
  neuro-align/<session_id>/<session_id>_{raster,regions,timeline}.png
  neuro-align/{summary.json, provenance.json, sessions.csv}
"""
from __future__ import annotations

import csv
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from processor.common import app_version, peak_rss_mb, rel, set_threads, tool_versions, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("neuro-align")
log.setLevel(logging.INFO)
for noisy in ("matplotlib", "PIL", "fontTools", "h5py", "hdmf", "pynwb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "neuro-align"
DEFAULTS = {
    "NEURAL_SOURCE": "auto",
    "ROI_TRACE": "auto",
    "UNIT_LABELS": "good",
    "MIN_RATE_HZ": "0.1",
    "BRAIN_AREAS": "",
    "BEHAVIOR_VARS": "auto",
    "LINEARIZE": "auto",
    "BIN_SIZE_S": "auto",
    "EPOCH": "all",
    "RUN_SPEED": "auto",
    "SMOOTH_S": "0.1",
    "POSE_MIN_CONFIDENCE": "0.5",
    "MAX_DURATION_S": "0",
    "THREADS": "0",
}
PARAMETERS = tuple(DEFAULTS)
CANONICAL_VARS = {"position", "x", "y", "speed", "run_direction", "head_direction", "wheel_position",
                  "wheel_velocity", "pupil", "licks", "keypoint_speeds"}
UNIT_BIN_S = 0.025
CITATIONS = ["10.7554/eLife.85786", "10.7554/eLife.78362"]


class ParameterError(ValueError):
    pass


def read_parameters(env=os.environ) -> dict:
    raw = {k: (env.get(k).strip() if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p: dict = {}
    p["NEURAL_SOURCE"] = raw["NEURAL_SOURCE"].lower()
    if p["NEURAL_SOURCE"] not in ("auto", "units", "roi"):
        raise ParameterError("NEURAL_SOURCE must be auto, units or roi")
    p["ROI_TRACE"] = raw["ROI_TRACE"].lower()
    if p["ROI_TRACE"] not in ("auto", "deconvolved", "dff", "fluorescence"):
        raise ParameterError("ROI_TRACE must be auto, deconvolved, dff or fluorescence")
    labels = [x.strip().lower() for x in raw["UNIT_LABELS"].split(",") if x.strip()]
    p["UNIT_LABELS"] = None if (not labels or "all" in labels) else set(labels)
    p["BRAIN_AREAS"] = [x.strip() for x in raw["BRAIN_AREAS"].split(",") if x.strip()]
    toks = [x.strip() for x in raw["BEHAVIOR_VARS"].split(",") if x.strip()]
    from processor.nwbio import safe_name

    auto = not toks or "auto" in [t.lower() for t in toks]
    names = [t.lower() for t in toks if t.lower() in CANONICAL_VARS]
    extra = [safe_name(t) for t in toks if t.lower() not in CANONICAL_VARS and t.lower() != "auto"]
    p["BEHAVIOR_VARS"] = {"auto": auto, "names": names, "extra": extra}
    lin = raw["LINEARIZE"].lower()
    if lin not in ("auto", "true", "false"):
        raise ParameterError("LINEARIZE must be auto, true or false")
    p["LINEARIZE"] = None if lin == "auto" else lin == "true"
    b = raw["BIN_SIZE_S"].lower()
    if b in ("auto", "frame"):
        p["BIN_SIZE_S"] = b
    else:
        try:
            p["BIN_SIZE_S"] = float(b)
        except ValueError as exc:
            raise ParameterError("BIN_SIZE_S must be auto, frame or a number of seconds") from exc
        if not 0.001 <= p["BIN_SIZE_S"] <= 10:
            raise ParameterError("BIN_SIZE_S must be between 0.001 and 10 s")
    p["EPOCH"] = raw["EPOCH"]
    rs = raw["RUN_SPEED"].lower()
    try:
        p["RUN_SPEED"] = None if rs == "auto" else float(rs)
        p["MIN_RATE_HZ"] = float(raw["MIN_RATE_HZ"])
        p["SMOOTH_S"] = float(raw["SMOOTH_S"])
        p["POSE_MIN_CONFIDENCE"] = float(raw["POSE_MIN_CONFIDENCE"])
        p["MAX_DURATION_S"] = float(raw["MAX_DURATION_S"])
        p["THREADS"] = int(raw["THREADS"])
    except ValueError as exc:
        raise ParameterError(f"bad numeric parameter: {exc}") from exc
    if p["MIN_RATE_HZ"] < 0 or p["SMOOTH_S"] < 0 or p["MAX_DURATION_S"] < 0 or p["THREADS"] < 0 \
            or (p["RUN_SPEED"] is not None and p["RUN_SPEED"] < 0):
        raise ParameterError("MIN_RATE_HZ, SMOOTH_S, MAX_DURATION_S, RUN_SPEED and THREADS must be >= 0")
    if not 0 <= p["POSE_MIN_CONFIDENCE"] <= 1:
        raise ParameterError("POSE_MIN_CONFIDENCE must be in [0, 1]")
    return p


def params_for_record(p: dict) -> dict:
    out = dict(p)
    out["UNIT_LABELS"] = "all" if p["UNIT_LABELS"] is None else ",".join(sorted(p["UNIT_LABELS"]))
    out["BRAIN_AREAS"] = ",".join(p["BRAIN_AREAS"])
    bv = p["BEHAVIOR_VARS"]
    out["BEHAVIOR_VARS"] = ",".join((["auto"] if bv["auto"] else []) + bv["names"] + bv["extra"])
    out["LINEARIZE"] = "auto" if p["LINEARIZE"] is None else str(p["LINEARIZE"]).lower()
    out["RUN_SPEED"] = "auto" if p["RUN_SPEED"] is None else p["RUN_SPEED"]
    return out


def region_ok(region: str, areas: list[str]) -> bool:
    if not areas:
        return True
    r = (region or "").lower()
    return any(r.startswith(a.lower()) for a in areas)


def overlap(a, b) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def series_span(series) -> tuple[float, float] | None:
    import numpy as np

    lo, hi = math.inf, -math.inf
    for s in series:
        if s.events or len(s.t) == 0:
            continue
        v = s.v if s.v.ndim == 1 else s.v[:, 0]
        ok = np.isfinite(v) & np.isfinite(s.t)
        if ok.any():
            lo, hi = min(lo, float(s.t[ok][0])), max(hi, float(s.t[ok][-1]))
    return (lo, hi) if lo < hi else None


def process_session(sess, params: dict, input_dir: Path, top: Path) -> dict:
    import numpy as np
    import pandas as pd
    import pynwb

    import ndx_pose  # noqa: F401  (registers the ndx-pose types before reading)
    from processor import align as A
    from processor import figures as F
    from processor import nwbio as N

    t_start = time.monotonic()
    sid = sess.session_id
    warnings = list(sess.warnings)
    status = {"session_id": sid, "neural_files": [f.rel for f in sess.neural],
              "behavior_files": [f.rel for f in sess.behavior]}
    if not sess.neural:
        raise ValueError("no neural data (Units or ophys traces) for this session")
    if not sess.behavior:
        warnings.append("no behaviour for this session: aligned.npz has neural data only (Y has no columns)")

    ios, nwbs = {}, {}

    def open_nwb(f):
        if f.path not in nwbs:
            io = pynwb.NWBHDF5IO(str(f.path), "r")
            ios[f.path] = io
            nwbs[f.path] = io.read()
        return nwbs[f.path]

    try:
        # ---------------------------------------------------------------- neural
        src = params["NEURAL_SOURCE"]
        if src == "auto":
            src = "units" if any(f.has_units for f in sess.neural) else "roi"
        if src == "units":
            files = [f for f in sess.neural if f.has_units]
            if not files:
                raise ValueError("NEURAL_SOURCE=units but no file of this session has a Units table with spikes")
            parts = [N.read_units(open_nwb(f), f.rel, params["UNIT_LABELS"]) for f in files]
            neural = parts[0]
            if len(parts) > 1:
                for k, p in enumerate(parts):
                    p.ids = [f"f{k}_{i}" for i in p.ids]
                neural = N.Neural(kind="units", source="units", signal="spike counts",
                                  ids=sum((p.ids for p in parts), []), meta=sum((p.meta for p in parts), []),
                                  spikes=sum((p.spikes for p in parts), []),
                                  warnings=sum((p.warnings for p in parts), []))
                for p in parts:
                    for k, v in p.dropped.items():
                        neural.dropped[k] = neural.dropped.get(k, 0) + v
                    for k, v in p.dropped_regions.items():
                        neural.dropped_regions[k] = neural.dropped_regions.get(k, 0) + v
                neural.filters = {"files": [p.filters for p in parts]}
            nfile = files[0]
        else:
            files = [f for f in sess.neural if f.has_ophys]
            if not files:
                raise ValueError("NEURAL_SOURCE=roi but no file of this session has ophys ROI traces")
            nfile = files[0]
            if len(files) > 1:
                warnings.append(f"{len(files)} ophys files for this session; used {nfile.rel}")
            neural = N.read_rois(open_nwb(nfile), nfile.rel, params["ROI_TRACE"])
        warnings += neural.warnings
        nwb_n = open_nwb(nfile)
        ref_n = N.reference_time(nwb_n)
        ident = N.identity(nwb_n)

        # Region filter.
        if params["BRAIN_AREAS"]:
            keep = [i for i, m in enumerate(neural.meta) if region_ok(m["region"], params["BRAIN_AREAS"])]
            dropped = [i for i in range(len(neural.ids)) if i not in set(keep)]
            for i in dropped:
                r = neural.meta[i]["region"]
                neural.dropped_regions[r] = neural.dropped_regions.get(r, 0) + 1
            neural.dropped["brain_area"] = len(dropped)
            neural.ids = [neural.ids[i] for i in keep]
            neural.meta = [neural.meta[i] for i in keep]
            if neural.kind == "units":
                neural.spikes = [neural.spikes[i] for i in keep]
            else:
                new, col = [], 0
                ks = set(keep)
                for t, data, sl in neural.planes:
                    cols = [j - sl.start for j in range(sl.start, sl.stop) if j in ks]
                    new.append((t, data[:, cols], slice(col, col + len(cols))))
                    col += len(cols)
                neural.planes = new
        if neural.kind == "units":
            nz = [i for i, s in enumerate(neural.spikes) if len(s)]
            if len(nz) < len(neural.ids):
                neural.dropped["no_spikes"] = len(neural.ids) - len(nz)
                neural.ids = [neural.ids[i] for i in nz]
                neural.meta = [neural.meta[i] for i in nz]
                neural.spikes = [neural.spikes[i] for i in nz]
        if not neural.ids:
            raise ValueError(f"no neuron left after the filters (dropped: {neural.dropped})")
        n_range = neural.t_range()

        # ---------------------------------------------------------------- behaviour
        series, tables, clock = [], [], {"neural_range_s": list(n_range), "files": []}
        for f in sess.behavior:
            nwb_b = open_nwb(f)
            s_f, t_f = N.read_behavior(nwb_b, f.rel, warnings)
            entry = {"file": f.rel, "shift_s": 0.0}
            ref_b = N.reference_time(nwb_b)
            span = series_span(s_f)
            delta = (ref_b - ref_n).total_seconds() if (ref_b is not None and ref_n is not None and f.path != nfile.path) else 0.0
            if abs(delta) > 1e-3 and span is not None:
                ov0 = overlap(n_range, span)
                ov1 = overlap(n_range, (span[0] + delta, span[1] + delta))
                if ov1 > ov0:
                    for s in s_f:
                        s.t = s.t + delta
                    for _, df in t_f:
                        for c in ("start_time", "stop_time"):
                            if c in df:
                                df[c] = df[c] + delta
                    span = (span[0] + delta, span[1] + delta)
                    entry["shift_s"] = delta
                    warnings.append(f"{f.rel}: its timestamps_reference_time is {delta:+.3f} s from the neural file's; "
                                    f"shifted its times by that much onto the neural session clock")
                else:
                    warnings.append(f"{f.rel}: its timestamps_reference_time differs by {delta:+.3f} s from the neural "
                                    f"file's, but its times already overlap the neural data; not shifted")
            if span is not None:
                entry["range_s"] = list(span)
                ov = overlap(n_range, span)
                frac = ov / (span[1] - span[0]) if span[1] > span[0] else 0.0
                entry["overlap_fraction"] = round(frac, 4)
                if frac < 0.5:
                    warnings.append(f"CLOCK: behaviour in {f.rel} spans {span[0]:.1f}-{span[1]:.1f} s but the neural data "
                                    f"{n_range[0]:.1f}-{n_range[1]:.1f} s; only {100 * frac:.0f}% overlaps. Are both on "
                                    f"the same session clock?")
                elif frac < 0.9:
                    warnings.append(f"{100 * (1 - frac):.0f}% of the behaviour in {f.rel} lies outside the neural "
                                    f"recording ({n_range[0]:.1f}-{n_range[1]:.1f} s)")
            clock["files"].append(entry)
            series += s_f
            seen = {n for n, _ in tables}
            tables += [(n, df) for n, df in t_f if n not in seen]

        b_span = series_span(series)
        if b_span is not None:
            win = (max(n_range[0], b_span[0]), min(n_range[1], b_span[1]))
            if win[1] <= win[0]:
                raise ValueError(f"neural data ({n_range[0]:.1f}-{n_range[1]:.1f} s) and behaviour "
                                 f"({b_span[0]:.1f}-{b_span[1]:.1f} s) do not overlap: check that both are on the "
                                 f"same session clock")
        else:
            win = n_range
            if sess.behavior:
                warnings.append("no continuous behaviour variable found; the window is the neural recording")
        if params["MAX_DURATION_S"] > 0 and win[1] - win[0] > params["MAX_DURATION_S"]:
            warnings.append(f"window cut to the first {params['MAX_DURATION_S']:g} s (MAX_DURATION_S)")
            win = (win[0], win[0] + params["MAX_DURATION_S"])
        if b_span is not None:
            for side, gap in (("before the first", n_range[0] - b_span[0]), ("after the last", b_span[1] - n_range[1])):
                if gap > max(5.0, 0.01 * (b_span[1] - b_span[0])):
                    warnings.append(f"behaviour continues {gap:.1f} s {side} neural sample; the aligned window is "
                                    f"restricted to {win[0]:.1f}-{win[1]:.1f} s, where both have data")
        clock["behavior_range_s"] = list(b_span) if b_span else None
        clock["window_s"] = list(win)

        # ---------------------------------------------------------------- grid
        bsz = params["BIN_SIZE_S"]
        if bsz == "auto":
            bsz = UNIT_BIN_S if neural.kind == "units" else neural.frame_period
        elif bsz == "frame":
            if neural.kind != "roi":
                raise ValueError("BIN_SIZE_S=frame needs ROI (imaging) data")
            bsz = neural.frame_period
        bsz = round(float(bsz), 9)
        t0 = win[0]
        if neural.kind == "roi" and params["BIN_SIZE_S"] in ("auto", "frame"):
            t_first = neural.planes[0][0]
            k = np.searchsorted(t_first, win[0])
            t0 = max(win[0], float(t_first[min(k, len(t_first) - 1)]) - bsz / 2)
        grid = A.make_grid(t0, win[1], float(bsz))
        if grid.n < 10:
            raise ValueError(f"aligned window too short ({win[1] - win[0]:.2f} s for {bsz} s bins)")
        dur = grid.n * grid.bin_size
        if neural.kind == "units":
            neural.spikes = [s[(s >= grid.edges[0]) & (s < grid.edges[-1])] for s in neural.spikes]
            rates = np.array([len(s) / dur for s in neural.spikes])
            keep = np.flatnonzero(rates >= params["MIN_RATE_HZ"]) if params["MIN_RATE_HZ"] > 0 else np.arange(len(rates))
            neural.dropped["min_rate"] = int(len(neural.ids) - len(keep))
            for i in sorted(set(range(len(neural.ids))) - set(keep.tolist())):
                r = neural.meta[i]["region"]
                neural.dropped_regions[r] = neural.dropped_regions.get(r, 0) + 1
            if len(keep) == 0:
                raise ValueError(f"no unit passes MIN_RATE_HZ={params['MIN_RATE_HZ']} in the aligned window")
            neural.spikes = [neural.spikes[i] for i in keep]
            rates = rates[keep]
            X = A.bin_spikes(neural.spikes, grid)
        else:
            X = A.bin_traces(neural.planes, grid)
            neural.planes = None
            sd = np.nanstd(X, axis=0)
            keep = np.flatnonzero(np.isfinite(sd) & (sd > 0))
            neural.dropped["flat_or_empty"] = int(len(neural.ids) - len(keep))
            if len(keep) == 0:
                raise ValueError("every ROI trace is empty or flat in the aligned window")
            if len(keep) < X.shape[1]:
                X = X[:, keep]
            rates = np.nanmean(X, axis=0)
        neural.ids = [neural.ids[i] for i in keep]
        neural.meta = [neural.meta[i] for i in keep]

        B = A.build_behaviour(series, grid, params)
        warnings += B.warnings
        names = [v.name for v in B.vars]
        Y = np.stack([v.values for v in B.vars], axis=1).astype(np.float32) if B.vars else np.zeros((grid.n, 0), np.float32)

        # ---------------------------------------------------------------- events + epoch
        if B.info.get("laps"):
            tables.append(("laps_detected", pd.DataFrame(B.info["laps"], columns=["start_time", "stop_time", "direction"])))
        frames = []
        for name, df in tables:
            if df is None or not len(df) or "start_time" not in df:
                continue
            d = df.copy()
            d.insert(0, "table", name)
            frames.append(d)
        events = pd.concat(frames, ignore_index=True, sort=False) if frames else \
            pd.DataFrame({"table": pd.Series(dtype=str), "start_time": pd.Series(dtype=float),
                          "stop_time": pd.Series(dtype=float)})
        for c in events.columns:
            if events[c].dtype == object or str(events[c].dtype).startswith("string"):
                events[c] = events[c].map(lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))
        ep = params["EPOCH"]
        epoch_info = {"name": ep}
        if ep.lower() == "all":
            epoch = np.ones(grid.n, dtype=bool)
        elif ep.lower() == "run":
            sp = B.get("speed")
            if sp is None:
                raise ValueError("EPOCH=run needs a speed (or a position to derive it from); none found")
            rdv = B.get("run_direction")
            epoch, thr = A.run_epoch(sp, grid, params, rdv)
            lapwise = rdv is not None and "detected laps" in rdv.note
            epoch_info.update(speed_threshold=thr, speed_unit=sp.unit,
                              definition="running along the track inside detected laps" if lapwise else
                              f"speed >= threshold for >= {A.MIN_RUN_S} s")
        else:
            names_t = {n.lower(): n for n, _ in tables}
            key = ep.lower()
            if key not in names_t:
                raise ValueError(f"EPOCH={ep!r}: no such TimeIntervals table (have: {', '.join(sorted(names_t)) or 'none'})")
            df = dict(tables)[names_t[key]]
            epoch = A.table_epoch(df, grid)
            epoch_info.update(table=names_t[key], n_intervals=int(len(df)))
        epoch_info.update(n_bins=int(epoch.sum()), seconds=round(float(epoch.sum() * grid.bin_size), 3),
                          fraction=round(float(epoch.mean()), 4))
        if epoch.sum() == 0:
            raise ValueError(f"EPOCH={ep!r} selects no time inside the aligned window")

        # ---------------------------------------------------------------- write
        out = top / sid
        out.mkdir(parents=True, exist_ok=True)
        meta_dt = np.dtype([("id", "U64"), ("region", "U64"), ("label", "U32"), ("cell_type", "U32"), ("depth", "f8"),
                            ("x", "f8"), ("y", "f8"), ("rate_hz", "f8")])
        meta = np.array([(i, m["region"][:64], m["label"][:32], m.get("cell_type", "")[:32], m["depth"], m["x"], m["y"], float(r))
                         for i, m, r in zip(neural.ids, neural.meta, rates)], dtype=meta_dt)
        np.savez_compressed(
            out / "aligned.npz", t=grid.centres, X=X.astype(np.float32), Y=Y, neuron_ids=np.array(neural.ids, dtype="U64"),
            neuron_meta=meta, behavior_names=np.array(names, dtype="U64"),
            behavior_units=np.array([v.unit for v in B.vars], dtype="U64"),
            behavior_kinds=np.array([v.kind for v in B.vars], dtype="U32"), epoch_mask=epoch,
            bin_size=np.float64(grid.bin_size), session_id=np.array(sid), neural_kind=np.array(neural.kind))
        if neural.kind == "units":
            lens = np.array([len(s) for s in neural.spikes], dtype=np.int64)
            np.savez_compressed(out / "spikes.npz", times=np.concatenate(neural.spikes) if neural.spikes else np.zeros(0),
                                offsets=np.r_[0, np.cumsum(lens)].astype(np.int64),
                                neuron_ids=np.array(neural.ids, dtype="U64"))
        events.to_parquet(out / "events.parquet", index=False)

        variables = []
        for v in B.vars:
            vals = v.values[np.isfinite(v.values)]
            variables.append({"name": v.name, "kind": v.kind, "unit": v.unit, "source": v.source,
                              "circular": v.circular, "note": v.note,
                              "range": [float(vals.min()), float(vals.max())] if len(vals) else None,
                              "fraction_valid": round(float(np.isfinite(v.values).mean()), 4)})
        regions_kept: dict = {}
        for m in neural.meta:
            regions_kept[m["region"]] = regions_kept.get(m["region"], 0) + 1
        manifest = {
            "session_id": sid, "app": APP_NAME, "version": app_version(),
            "bin_size_s": grid.bin_size, "n_bins": grid.n, "t_start": grid.edges[0], "t_stop": grid.edges[-1],
            "time": "seconds on the session clock of the neural file; t = bin centres",
            "neural": {"kind": neural.kind, "source": neural.source, "signal": neural.signal,
                       "X": "spike counts per bin" if neural.kind == "units" else f"mean {neural.signal} per bin",
                       "files": status["neural_files"], "n_neurons": len(neural.ids), "regions": regions_kept,
                       "filters": neural.filters | {"min_rate_hz": params["MIN_RATE_HZ"] if neural.kind == "units" else None,
                                                    "brain_areas": params["BRAIN_AREAS"] or None},
                       "dropped": neural.dropped, "frame_period_s": neural.frame_period},
            "behavior": {"files": status["behavior_files"], "variables": variables,
                         "linear_track": B.info.get("linear_track"),
                         "laps": {"n": len(B.info.get("laps", [])),
                                  "seconds": round(sum(b - a for a, b, _ in B.info.get("laps", [])), 2),
                                  "table": "laps_detected" if B.info.get("laps") else None},
                         "smooth_s": params["SMOOTH_S"], "pose_min_confidence": params["POSE_MIN_CONFIDENCE"]},
            "epoch": epoch_info,
            "events": {"tables": sorted(events["table"].unique().tolist()) if len(events) else [],
                       "n_rows": int(len(events))},
            "clock": clock,
            "source": ident,
            "sheet": sess.sheet,
            "files": {"aligned": "aligned.npz", "spikes": "spikes.npz" if neural.kind == "units" else None,
                      "events": "events.parquet"},
            "warnings": warnings,
        }
        write_json(out / "manifest.json", manifest)

        figures = []
        try:
            figures.append(F.raster(out / f"{sid}_raster.png", sid, neural.kind, grid.centres, X,
                                    neural.spikes if neural.kind == "units" else None, neural.meta, Y, names,
                                    [v.unit for v in B.vars], epoch, grid.bin_size))
            figures.append(F.counts_by_region(out / f"{sid}_regions.png", sid, neural.kind, regions_kept,
                                              neural.dropped_regions))
            spans = [("neural", *n_range)] + [(Path(e["file"]).name[:40], *e["range_s"]) for e in clock["files"] if "range_s" in e]
            figures.append(F.timeline(out / f"{sid}_timeline.png", sid, spans, (grid.edges[0], grid.edges[-1])))
        except Exception as exc:   # a figure is a convenience
            log.exception("%s: figure failed: %s", sid, exc)
            warnings.append(f"figure failed: {exc}")

        status.update(
            status="ok", neural_kind=neural.kind, neural_source=neural.source, n_neurons=len(neural.ids),
            dropped=neural.dropped, regions=regions_kept, n_bins=grid.n, bin_size_s=grid.bin_size,
            duration_s=round(dur, 3), window_s=[round(grid.edges[0], 4), round(grid.edges[-1], 4)],
            behavior_variables=names, epoch=epoch_info, n_events=int(len(events)),
            event_tables=manifest["events"]["tables"], warnings=warnings,
            figures=[(rel(p, top), c) for p, c in figures], seconds=round(time.monotonic() - t_start, 2),
            subject_id=sess.sheet.get("subject_id") or ident.get("subject", {}).get("subject_id"),
            condition=sess.sheet.get("condition"), group=sess.sheet.get("group"))
        return status
    finally:
        for io in ios.values():
            try:
                io.close()
            except Exception:
                pass


def main() -> int:
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 1
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        params = read_parameters()
    except ParameterError as exc:
        log.error("%s", exc)
        return 2
    threads = set_threads(params["THREADS"])

    from processor.inputs import discover

    found = discover(input_dir)
    for name, reason in found.skipped.items():
        log.warning("%s: skipped, %s", name, reason)
    if not found.sessions:
        log.error("no NWB file with units, ophys traces or behaviour under %s", input_dir)
        return 1
    log.info("%d NWB files, %d sessions; NEURAL_SOURCE=%s, BIN_SIZE_S=%s, EPOCH=%s, %d threads",
             len(found.files), len(found.sessions), params["NEURAL_SOURCE"], params["BIN_SIZE_S"], params["EPOCH"],
             threads)
    top = output_dir / APP_NAME
    top.mkdir(parents=True, exist_ok=True)
    statuses = []
    all_warnings = []
    for sess in found.sessions:
        t = time.monotonic()
        if not sess.neural:
            msg = f"{sess.session_id}: behaviour only, no neural data; skipped ({', '.join(f.rel for f in sess.behavior)})"
            log.warning(msg)
            all_warnings.append(msg)
            statuses.append({"session_id": sess.session_id, "status": "skipped", "error": "no neural data",
                             "behavior_files": [f.rel for f in sess.behavior], "neural_files": []})
            continue
        try:
            st = process_session(sess, params, input_dir, top)
            for w in st["warnings"]:
                log.warning("%s: %s", sess.session_id, w)
            all_warnings += [f"{sess.session_id}: {w}" for w in st["warnings"]]
            log.info("%s: %s %d neurons (%s), %d bins of %.4g s (%.0f s), behaviour [%s], epoch %s %.0f%%, %d events, %.1fs",
                     sess.session_id, st["neural_kind"], st["n_neurons"],
                     ", ".join(f"{k} {v}" for k, v in sorted(st["regions"].items(), key=lambda kv: -kv[1])[:6]),
                     st["n_bins"], st["bin_size_s"], st["duration_s"], ", ".join(st["behavior_variables"]),
                     st["epoch"]["name"], 100 * st["epoch"]["fraction"], st["n_events"], time.monotonic() - t)
        except Exception as exc:   # one bad session is skipped, not fatal
            log.exception("%s: failed: %s", sess.session_id, exc)
            st = {"session_id": sess.session_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                  "neural_files": [f.rel for f in sess.neural], "behavior_files": [f.rel for f in sess.behavior],
                  "seconds": round(time.monotonic() - t, 2)}
            all_warnings.append(f"{sess.session_id}: failed: {exc}")
        st["sheet"] = sess.sheet
        statuses.append(st)
    ok = [s for s in statuses if s["status"] == "ok"]

    # sessions.csv (first app of the decoding chain)
    extra = sorted({k for s in statuses for k in (s.get("sheet") or {}) if k not in ("session_id", "path", "subject_id", "condition", "group")})
    cols = ["session_id", "path", "behavior_paths", "subject_id", "condition", "group", "status", "neural_kind",
            "n_neurons", "n_bins", "bin_size_s", "n_behavior_vars", "epoch_fraction"] + extra
    with open(top / "sessions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for s in statuses:
            sh = s.get("sheet") or {}
            w.writerow([s["session_id"], ";".join(s.get("neural_files", [])), ";".join(s.get("behavior_files", [])),
                        s.get("subject_id") or sh.get("subject_id", ""), sh.get("condition", ""), sh.get("group", ""),
                        s["status"], s.get("neural_kind", ""), s.get("n_neurons", ""), s.get("n_bins", ""),
                        s.get("bin_size_s", ""), len(s.get("behavior_variables", [])) if s["status"] == "ok" else "",
                        (s.get("epoch") or {}).get("fraction", "")] + [sh.get(k, "") for k in extra])

    figures = []
    for s in ok:
        figures += [{"path": p, "caption": c} for p, c in s.get("figures", [])]
    summary = {
        "app": APP_NAME, "version": app_version(),
        "sessions": {s["session_id"]: ({k: s.get(k) for k in (
            "status", "neural_kind", "neural_source", "n_neurons", "regions", "dropped", "n_bins", "bin_size_s",
            "duration_s", "window_s", "behavior_variables", "epoch", "n_events", "event_tables", "subject_id",
            "condition", "group")} if s["status"] == "ok" else {"status": s["status"], "error": s.get("error")})
            for s in statuses},
        "n_sessions": len(ok), "n_failed": len(statuses) - len(ok),
        "figures": figures,
        "tables": ["sessions.csv"] + [f"{s['session_id']}/events.parquet" for s in ok],
        "warnings": all_warnings,
    }
    write_json(top / "summary.json", summary)
    write_json(top / "provenance.json", {
        "app": APP_NAME, "version": app_version(), "variant": None,
        "tools": tool_versions(("pynwb", "hdmf", "ndx-pose", "pynapple", "numpy", "scipy", "pandas", "pyarrow",
                                "h5py", "matplotlib")),
        "parameters": params_for_record(params),
        "inputs": sorted({f.rel for f in found.files}),
        "started": started.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.monotonic() - t0, 2), "peak_rss_mb": peak_rss_mb(), "gpu": None,
        "sessions": {s["session_id"]: {"status": s["status"], "seconds": s.get("seconds")} | (
            {"error": s["error"]} if s.get("error") else {}) for s in statuses},
        "citations": CITATIONS,
    })
    if not ok:
        log.error("every session failed or had no neural data")
        return 1
    log.info("done: %d sessions ok, %d failed/skipped, %.1fs, peak RSS %s MB -> %s", len(ok), len(statuses) - len(ok),
             time.monotonic() - t0, peak_rss_mb(), top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
