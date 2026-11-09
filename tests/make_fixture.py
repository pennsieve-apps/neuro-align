"""
Synthetic NWB sessions with known ground truth, for the tests and for `review.generate`.

  sub-synth01_ses-track  units (neural NWB) + behaviour (a second NWB: Position, laps/trials,
                         licks, ndx-pose keypoints). 300 s on a 200 cm linear track:
                         40 place cells (CA1), 10 speed cells (MEC), 10 untuned (CA1),
                         5 units labelled 'mua' and 3 good units under 0.1 Hz (both dropped).
  sub-synth02_ses-vr     one behavior+ophys NWB: 36 ROIs (30 accepted: 20 place-tuned, 10 untuned;
                         6 rejected), Deconvolved + DfOverF at 15.5 Hz, a 1D virtual-track position
                         with a -500 'off track' value between laps, running speed.

Layouts: linked/neural/..., linked/behavior/... (as linked ports deliver them) and flat/ (all files
side by side, with a sessions.csv carrying a condition column). ground_truth.json sits next to them.

    python -m tests.make_fixture OUT [--layout linked|flat|both]
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

T_TRACK = 300.0
TRACK_CM = 200.0
POS_HZ = 30.0
START = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


def track_trajectory(duration: float, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Back-and-forth runs on a linear track with pauses at the ends; returns t, position, speed, laps."""
    dt = 1.0 / 1000.0
    t = np.arange(0, duration, dt)
    pos = np.zeros_like(t)
    laps = []
    x, i, direction = 0.0, 0, 1
    while i < len(t):
        pause = int(rng.uniform(1.5, 3.0) / dt)
        pos[i:i + pause] = x
        i += pause
        if i >= len(t):
            break
        v = rng.uniform(18, 45)
        n = int(TRACK_CM / v / dt)
        start = i
        prof = np.sin(np.linspace(0, np.pi, n)) ** 0.5
        steps = prof / prof.sum() * TRACK_CM * direction
        seg = x + np.cumsum(steps)
        m = min(n, len(t) - i)
        pos[i:i + m] = seg[:m]
        i += m
        if m == n:
            laps.append((t[start], t[min(i, len(t) - 1)], direction))
        x = seg[m - 1] if m else x
        direction = -direction
    speed = np.abs(np.gradient(pos, dt))
    return t, pos, speed, laps


def poisson_spikes(t: np.ndarray, rate: np.ndarray, rng) -> np.ndarray:
    dt = t[1] - t[0]
    k = rng.random(len(t)) < rate * dt
    return t[k] + rng.uniform(0, dt, k.sum())


def write_units_session(neural_path: Path, behavior_path: Path, rng) -> dict:
    from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton, Skeletons
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.behavior import BehavioralEvents, Position, SpatialSeries
    from pynwb.file import Subject

    t, pos, speed, laps = track_trajectory(T_TRACK, rng)
    gt = {"session_id": "sub-synth01_ses-track", "place": {}, "speed": [], "untuned": [], "mua": [], "low_rate": []}
    nwb = NWBFile("synthetic units on a linear track", "synth01-units", START, timestamps_reference_time=START,
                  subject=Subject(subject_id="synth01", species="Mus musculus"))
    nwb.add_unit_column("label", "curation label: good / mua")
    nwb.add_unit_column("brain_area", "brain region")
    nwb.add_unit_column("depth", "depth along the probe (um)")
    uid = 0
    centres = np.linspace(10, 190, 40)
    for c in centres:
        rate = 0.2 + 15.0 * np.exp(-0.5 * ((pos - c) / 10.0) ** 2) * (speed > 2)
        nwb.add_unit(spike_times=poisson_spikes(t, rate, rng), label="good", brain_area="CA1", depth=1000 + uid * 5, id=uid)
        gt["place"][str(uid)] = float(c)
        uid += 1
    for _ in range(10):
        rate = 0.5 + 0.3 * speed
        nwb.add_unit(spike_times=poisson_spikes(t, rate, rng), label="good", brain_area="MEC", depth=2000 + uid, id=uid)
        gt["speed"].append(str(uid))
        uid += 1
    for _ in range(10):
        nwb.add_unit(spike_times=poisson_spikes(t, np.full(len(t), 3.0), rng), label="good", brain_area="CA1",
                     depth=1100 + uid, id=uid)
        gt["untuned"].append(str(uid))
        uid += 1
    for _ in range(5):
        nwb.add_unit(spike_times=poisson_spikes(t, np.full(len(t), 8.0), rng), label="mua", brain_area="CA1",
                     depth=1200 + uid, id=uid)
        gt["mua"].append(str(uid))
        uid += 1
    for _ in range(3):
        nwb.add_unit(spike_times=np.sort(rng.uniform(0, T_TRACK, 5)), label="good", brain_area="CA1", depth=1300 + uid,
                     id=uid)
        gt["low_rate"].append(str(uid))
        uid += 1
    with NWBHDF5IO(str(neural_path), "w") as io:
        io.write(nwb)

    # Behaviour file: what pose-estimate / pose-kinematics or a lab's tracking would give.
    b = NWBFile("synthetic behaviour on a linear track", "synth01-behavior", START, timestamps_reference_time=START,
                subject=Subject(subject_id="synth01", species="Mus musculus"))
    step = int(1000 / POS_HZ)
    tp = t[::step]
    x = pos[::step]
    y = 5.0 + rng.normal(0, 0.8, len(tp))
    mod = b.create_processing_module("behavior", "tracking")
    mod.add(Position(spatial_series=SpatialSeries(name="position", data=np.c_[x, y], timestamps=tp, unit="cm",
                                                  reference_frame="track start", description="animal position")))
    end_times = np.array([l[1] for l in laps])
    licks = np.sort(np.concatenate([e + rng.uniform(0.05, 1.0, 4) for e in end_times]))
    mod.add(BehavioralEvents(time_series=TimeSeries(name="lick_times", data=np.ones(len(licks), dtype=bool),
                                                    timestamps=licks, unit="n.a.", description="licks")))
    for name in ("laps", "trials"):
        tbl = b.create_time_intervals(name, f"{name}: one run along the track") if name != "trials" else None
        if name == "trials":
            b.add_trial_column("direction", "+1 = increasing position")
            for s, e, d in laps:
                b.add_trial(start_time=float(s), stop_time=float(e), direction=int(d))
        else:
            tbl.add_column("direction", "+1 = increasing position")
            for s, e, d in laps:
                tbl.add_row(start_time=float(s), stop_time=float(e), direction=int(d))
    # ndx-pose keypoints in pixels (4 px / cm), nose ahead of the tail base in the running direction
    vel = np.gradient(x, tp)
    head = np.sign(vel + 1e-9)
    nose = np.c_[x * 4 + head * 20, y * 4] + rng.normal(0, 1, (len(tp), 2))
    tail = np.c_[x * 4 - head * 20, y * 4] + rng.normal(0, 1, (len(tp), 2))
    conf = np.clip(rng.normal(0.95, 0.05, len(tp)), 0, 1)
    conf[rng.random(len(tp)) < 0.02] = 0.1
    skel = Skeleton(name="mouse", nodes=["nose", "tailbase"], edges=np.array([[0, 1]], dtype="uint8"))
    series = [PoseEstimationSeries(name=n, description=n, data=d, unit="pixels", reference_frame="top-left",
                                   timestamps=tp, confidence=conf, confidence_definition="likelihood")
              for n, d in (("nose", nose), ("tailbase", tail))]
    pe = PoseEstimation(name="PoseEstimation", pose_estimation_series=series, description="synthetic",
                        scorer="synthetic", source_software="tests.make_fixture", skeleton=skel)
    mod.add(Skeletons(skeletons=[skel]))
    mod.add(pe)
    with NWBHDF5IO(str(behavior_path), "w") as io:
        io.write(b)
    gt["n_laps"] = len(laps)
    return gt


def write_ophys_session(path: Path, rng) -> dict:
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.behavior import BehavioralTimeSeries
    from pynwb.file import Subject
    from pynwb.ophys import DfOverF, Fluorescence, ImageSegmentation, OpticalChannel, RoiResponseSeries

    fps, dur = 15.5, 240.0
    n = int(fps * dur)
    tf = np.arange(n) / fps
    t, pos, speed, laps = vr_trajectory(dur, rng)
    p_f = np.interp(tf, t, pos)
    s_f = np.interp(tf, t, speed)
    on_track = np.interp(tf, t, (pos >= 0).astype(float)) > 0.5
    nwb = NWBFile("synthetic 2p on a virtual linear track", "synth02", START, timestamps_reference_time=START,
                  subject=Subject(subject_id="synth02", species="Mus musculus"))
    dev = nwb.create_device("microscope")
    plane = nwb.create_imaging_plane(name="ImagingPlane", optical_channel=OpticalChannel("green", "GCaMP", 510.0),
                                     description="synthetic", device=dev, excitation_lambda=920.0, imaging_rate=fps,
                                     indicator="GCaMP6f", location="hippocampus, CA1")
    ophys = nwb.create_processing_module("ophys", "optical physiology")
    seg = ImageSegmentation()
    ps = seg.create_plane_segmentation("ROIs", plane, "PlaneSegmentation")
    ps.add_column("iscell", "suite2p iscell (is_cell, probability)")
    n_roi = 36
    gt = {"session_id": "sub-synth02_ses-vr", "place": {}, "untuned": [], "rejected": []}
    centres = np.linspace(20, 380, 20)
    rates = []
    for k in range(n_roi):
        px, py = rng.integers(5, 120, 2)
        mask = [(int(px + dx), int(py + dy), 1.0) for dx in range(3) for dy in range(3)]
        acc = k < 30
        ps.add_roi(pixel_mask=mask, iscell=[1.0 if acc else 0.0, 0.9 if acc else 0.2], id=k)
        if k < 20:
            r = 0.05 + 2.0 * np.exp(-0.5 * ((p_f - centres[k]) / 20.0) ** 2) * on_track
            gt["place"][str(k)] = float(centres[k])
        elif k < 30:
            r = np.full(n, 0.3)
            gt["untuned"].append(str(k))
        else:
            r = np.full(n, 0.3)
            gt["rejected"].append(str(k))
        rates.append(r)
    ophys.add(seg)
    R = np.stack(rates, axis=1)
    spikes = rng.poisson(R / fps * 3).astype(float)
    kernel = np.exp(-np.arange(30) / (0.7 * fps))
    dff = np.stack([np.convolve(spikes[:, j], kernel)[:n] for j in range(n_roi)], axis=1) * 0.3
    dff += rng.normal(0, 0.05, dff.shape)
    region = ps.create_roi_table_region("all ROIs", region=list(range(n_roi)))
    ophys.add(Fluorescence(name="Deconvolved", roi_response_series=RoiResponseSeries(
        name="plane0", data=spikes.astype(np.float32), rois=region, unit="a.u.", rate=fps, starting_time=0.0)))
    ophys.add(DfOverF(roi_response_series=RoiResponseSeries(
        name="plane0", data=dff.astype(np.float32), rois=region, unit="dF/F", rate=fps, starting_time=0.0)))
    beh = nwb.create_processing_module("behavior", "virtual reality")
    p_out = np.where(on_track, p_f, -500.0)
    beh.add(BehavioralTimeSeries(time_series=[
        TimeSeries(name="position", data=p_out, timestamps=tf, unit="cm", description="position on the virtual track"),
        TimeSeries(name="speed", data=s_f, timestamps=tf, unit="cm/s", description="running speed")]))
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    gt["n_laps"] = len(laps)
    return gt


def vr_trajectory(duration: float, rng):
    """One-way runs along a 400 cm virtual track, teleport back, 2 s inter-trial interval (position -1 = off)."""
    dt = 1 / 1000
    t = np.arange(0, duration, dt)
    pos = np.full(len(t), -1.0)
    i, laps = 0, []
    while i < len(t):
        i += int(2.0 / dt)
        v = rng.uniform(30, 60)
        n = int(400 / v / dt)
        m = min(n, len(t) - i)
        if m <= 0:
            break
        pos[i:i + m] = np.linspace(0, 400, n)[:m]
        laps.append((t[i], t[min(i + m, len(t) - 1)]))
        i += m
    speed = np.where(pos >= 0, np.abs(np.gradient(np.where(pos >= 0, pos, 0), dt)), 0.0)
    speed = np.clip(speed, 0, 100)
    return t, pos, speed, laps


def make(out: Path, layout: str = "both", seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_src"
    work.mkdir(exist_ok=True)
    gt = {"track": write_units_session(work / "sub-synth01_ses-track_units.nwb",
                                       work / "sub-synth01_ses-track_pose.nwb", rng),
          "vr": write_ophys_session(work / "sub-synth02_ses-vr_behavior+ophys.nwb", rng)}
    if layout in ("linked", "both"):
        d = out / "linked"
        (d / "neural").mkdir(parents=True, exist_ok=True)
        (d / "behavior").mkdir(parents=True, exist_ok=True)
        shutil.copy(work / "sub-synth01_ses-track_units.nwb", d / "neural")
        shutil.copy(work / "sub-synth01_ses-track_pose.nwb", d / "behavior")
        shutil.copy(work / "sub-synth02_ses-vr_behavior+ophys.nwb", d / "neural")
    if layout in ("flat", "both"):
        d = out / "flat"
        d.mkdir(parents=True, exist_ok=True)
        for f in work.iterdir():
            shutil.copy(f, d)
        (d / "sessions.csv").write_text(
            "session_id,path,subject_id,condition\n"
            "track01,sub-synth01_ses-track_units.nwb,synth01,control\n"
            "track01,sub-synth01_ses-track_pose.nwb,synth01,control\n"
            "vr01,sub-synth02_ses-vr_behavior+ophys.nwb,synth02,control\n")
    shutil.rmtree(work)
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=1))
    return gt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layout", choices=("linked", "flat", "both"), default="both")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    make(a.out, a.layout, a.seed)
    print(f"wrote {a.out}")
