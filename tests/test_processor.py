"""neuro-align on the synthetic fixture (both layouts), against the ground truth, without Docker."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.make_fixture import make


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fixture")
    make(d, "both")
    return d


def run(input_dir: Path, output_dir: Path, **params) -> int:
    from processor import main as app

    env = {"INPUT_DIR": str(input_dir), "OUTPUT_DIR": str(output_dir)} | {k: str(v) for k, v in params.items()}
    old = {k: os.environ.get(k) for k in list(env) + list(app.PARAMETERS)}
    for k in app.PARAMETERS:
        os.environ.pop(k, None)
    os.environ.update(env)
    try:
        return app.main()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="session")
def linked_out(fixture_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("out_linked")
    assert run(fixture_dir / "linked", out) == 0
    return out / "neuro-align"


def load(top: Path, sid: str):
    d = np.load(top / sid / "aligned.npz")
    m = json.loads((top / sid / "manifest.json").read_text())
    return d, m


def test_units_session(linked_out, fixture_dir):
    gt = json.loads((fixture_dir / "ground_truth.json").read_text())["track"]
    d, m = load(linked_out, "sub-synth01_ses-track")
    ids = list(d["neuron_ids"])
    # 'mua' units and units under MIN_RATE_HZ are dropped, every other unit kept
    assert set(ids) == set(gt["place"]) | set(gt["speed"]) | set(gt["untuned"])
    assert m["neural"]["dropped"]["label"] == 5 and m["neural"]["dropped"]["min_rate"] == 3
    assert d["X"].shape == (len(d["t"]), len(ids)) and d["X"].dtype == np.float32
    assert np.allclose(np.diff(d["t"]), 0.025)
    names = list(d["behavior_names"])
    for v in ("x", "y", "position", "speed", "run_direction", "licks", "nose_speed", "tailbase_speed"):
        assert v in names, v
    assert m["behavior"]["linear_track"]["detected"]
    # every end-to-end run of the simulated track is found as a lap
    assert m["behavior"]["laps"]["n"] == gt["n_laps"]
    # spike counts match the raw spike times in the window
    sp = np.load(linked_out / "sub-synth01_ses-track" / "spikes.npz")
    assert list(sp["neuron_ids"]) == ids
    lens = np.diff(sp["offsets"])
    assert np.array_equal(lens, d["X"].sum(axis=0).astype(int))
    # place cells fire most where ground truth puts their field (on the linearised position)
    pos = d["Y"][:, names.index("position")]
    ok = np.isfinite(pos)
    errs = []
    for uid, centre in gt["place"].items():
        j = ids.index(uid)
        w = d["X"][ok, j]
        errs.append(abs(np.sum(w * pos[ok]) / w.sum() - centre))
    assert np.median(errs) < 10
    # speed cells correlate with speed
    spd = d["Y"][:, names.index("speed")]
    ok = np.isfinite(spd)
    r = [np.corrcoef(d["X"][ok, ids.index(u)], spd[ok])[0, 1] for u in gt["speed"]]
    assert min(r) > 0.15
    meta = d["neuron_meta"]
    assert set(meta["region"]) == {"CA1", "MEC"}
    ev = pd.read_parquet(linked_out / "sub-synth01_ses-track" / "events.parquet")
    assert {"laps", "trials", "lick_times"} <= set(ev["table"]) or {"laps", "trials"} <= set(ev["table"])
    assert (ev.loc[ev.table == "trials", "start_time"] < ev.loc[ev.table == "trials", "stop_time"]).all()
    for f in ("raster", "regions", "timeline"):
        assert (linked_out / "sub-synth01_ses-track" / f"sub-synth01_ses-track_{f}.png").exists()


def test_roi_session(linked_out, fixture_dir):
    gt = json.loads((fixture_dir / "ground_truth.json").read_text())["vr"]
    d, m = load(linked_out, "sub-synth02_ses-vr")
    assert m["neural"]["kind"] == "roi" and "Deconvolved" in m["neural"]["source"]
    assert set(d["neuron_ids"]) == set(gt["place"]) | set(gt["untuned"])
    assert abs(float(d["bin_size"]) - 1 / 15.5) < 1e-6
    names = list(d["behavior_names"])
    pos = d["Y"][:, names.index("position")]
    assert np.nanmin(pos) > -10   # the -500 'off track' value became missing
    assert "spikes" not in os.listdir(linked_out / "sub-synth02_ses-vr")


def test_summary_and_provenance(linked_out):
    s = json.loads((linked_out / "summary.json").read_text())
    p = json.loads((linked_out / "provenance.json").read_text())
    assert set(s) >= {"app", "version", "sessions", "figures", "tables", "warnings"}
    assert all(f["caption"] and (linked_out / f["path"]).exists() for f in s["figures"])
    assert set(p) == {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished",
                      "wall_seconds", "peak_rss_mb", "gpu", "sessions", "citations"}
    rows = list(csv.DictReader(open(linked_out / "sessions.csv")))
    assert {r["session_id"] for r in rows} == {"sub-synth01_ses-track", "sub-synth02_ses-vr"}


def test_flat_layout_with_sheet(fixture_dir, tmp_path):
    assert run(fixture_dir / "flat", tmp_path) == 0
    rows = {r["session_id"]: r for r in csv.DictReader(open(tmp_path / "neuro-align" / "sessions.csv"))}
    assert set(rows) == {"track01", "vr01"}
    assert rows["track01"]["condition"] == "control" and rows["track01"]["status"] == "ok"
    d, _ = load(tmp_path / "neuro-align", "track01")
    assert "position" in list(d["behavior_names"])


def test_epochs_and_filters(fixture_dir, tmp_path):
    assert run(fixture_dir / "linked", tmp_path / "a", EPOCH="run", BRAIN_AREAS="MEC") == 0
    d, m = load(tmp_path / "a" / "neuro-align", "sub-synth01_ses-track")
    assert set(d["neuron_meta"]["region"]) == {"MEC"} and len(d["neuron_ids"]) == 10
    assert 0.3 < d["epoch_mask"].mean() < 0.95
    spd = d["Y"][:, list(d["behavior_names"]).index("speed")]
    assert np.nanmedian(spd[d["epoch_mask"]]) > 10
    assert m["epoch"]["definition"] == "running along the track inside detected laps"
    assert run(fixture_dir / "linked", tmp_path / "b", EPOCH="trials", NEURAL_SOURCE="units") == 0
    d, m = load(tmp_path / "b" / "neuro-align", "sub-synth01_ses-track")
    assert m["epoch"]["table"] == "trials" and 0.3 < d["epoch_mask"].mean() < 1
    # an unknown table fails the session (and the run, since only the units session has neural units)
    assert run(fixture_dir / "linked", tmp_path / "c", EPOCH="nosuchtable", NEURAL_SOURCE="units") == 1
    assert run(fixture_dir / "linked", tmp_path / "d", BIN_SIZE_S="0.1", BEHAVIOR_VARS="position,speed") == 0
    d, _ = load(tmp_path / "d" / "neuro-align", "sub-synth01_ses-track")
    assert list(d["behavior_names"]) == ["position", "speed"] and float(d["bin_size"]) == 0.1


def test_bad_parameters(fixture_dir, tmp_path):
    assert run(fixture_dir / "linked", tmp_path, NEURAL_SOURCE="lfp") == 2
    assert run(fixture_dir / "linked", tmp_path, BIN_SIZE_S="abc") == 2
    assert run(fixture_dir / "linked", tmp_path, LINEARIZE="maybe") == 2


def test_clock_mismatch(tmp_path, fixture_dir):
    """Behaviour on another clock (10000 s later, same reference time) fails the session with a clear message."""
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.behavior import Position, SpatialSeries

    start = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    (tmp_path / "in" / "neural").mkdir(parents=True)
    (tmp_path / "in" / "behavior").mkdir(parents=True)
    os.symlink(fixture_dir / "linked" / "neural" / "sub-synth01_ses-track_units.nwb",
               tmp_path / "in" / "neural" / "sub-synth01_ses-track_units.nwb")
    b = NWBFile("late", "late", start, timestamps_reference_time=start)
    mod = b.create_processing_module("behavior", "x")
    t = 10000 + np.arange(0, 100, 1 / 30)
    mod.add(Position(spatial_series=SpatialSeries(name="p", data=np.c_[t % 200, t * 0], timestamps=t, unit="cm",
                                                  reference_frame="x")))
    with NWBHDF5IO(str(tmp_path / "in" / "behavior" / "sub-synth01_ses-track_pose.nwb"), "w") as io:
        io.write(b)
    assert run(tmp_path / "in", tmp_path / "out") == 1
    prov = json.loads((tmp_path / "out" / "neuro-align" / "provenance.json").read_text())
    assert "do not overlap" in prov["sessions"]["sub-synth01_ses-track"]["error"]


def test_helpers():
    from processor.align import Grid, clean_sentinels, resample, resample_hold
    from processor.inputs import bids_session

    assert bids_session("sub-NR-0031_ses-caa5dddc-9290-4e27-9f5e-575ba3598614_desc-processed.nwb") == "sub-NR-0031_ses-caa5dddc"
    assert bids_session("sub-Buddy_ses-Buddy-06272013_behavior+ecephys.nwb") == "sub-Buddy_ses-Buddy-06272013"
    g = Grid(0.0, 0.1, 100)
    # event-driven encoder: value holds between samples
    h = resample_hold(np.array([0.0, 5.0]), np.array([1.0, 2.0]), g)
    assert np.all(h[g.centres < 5] == 1) and np.all(h[g.centres > 5] == 2)
    # slow samples are interpolated; a long gap becomes missing
    t = np.r_[np.arange(0, 3, 0.2), np.arange(8, 10, 0.2)]
    r = resample(t, t * 2, g)
    assert np.allclose(r[(g.centres > 0.5) & (g.centres < 2.5)], 2 * g.centres[(g.centres > 0.5) & (g.centres < 2.5)])
    assert np.isnan(r[(g.centres > 4) & (g.centres < 7)]).all()
    # circular mean across the wrap
    a = resample(np.linspace(0, 10, 1000), np.where(np.arange(1000) % 2, 0.1, 2 * np.pi - 0.1), g, circular=True)
    assert np.all((a < 0.05) | (a > 2 * np.pi - 0.05))
    x = np.r_[np.linspace(0, 400, 900), np.full(100, -500.0)]
    c, n = clean_sentinels(x)
    assert n == 100 and np.isnan(c[-100:]).all()


def test_rate_is_period(tmp_path):
    """A TimeSeries whose 'rate' holds the sampling period (seen on DANDI) is read with 1/rate."""
    from pynwb import TimeSeries

    from processor.nwbio import series_times

    ts = TimeSeries(name="w", data=np.zeros(1_000_000), unit="mm", rate=0.0004, starting_time=10.0)
    w = []
    t = series_times(ts, 1_000_000, w, "w")
    assert w and abs(t[-1] - (10 + 999_999 / 2500)) < 1e-6
