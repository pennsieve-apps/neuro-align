"""
Finding the input: NWB files anywhere under INPUT_DIR (following symlinks), a
sessions.csv sheet, session ids per conventions.md, what each file holds, and
which files make up the neural and behaviour side of each session.

Linked runs put the files in INPUT_DIR/neural/ and INPUT_DIR/behavior/
(placement: subfolder); unlinked runs may have them side by side. The folder
is a hint only: a file is used for what it contains.
"""
from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("neuro-align")

SESSION_RE = re.compile(r"sub-([A-Za-z0-9-]+)_ses-([A-Za-z0-9-]+)")
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
NEURAL_PORT = "neural"
BEHAVIOR_PORTS = ("behavior", "behaviour")
SHEET_NAME = "sessions.csv"


def sanitise(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", name).strip("-") or "session"


def bids_session(text: str) -> str | None:
    m = SESSION_RE.search(text)
    if not m:
        return None
    sub, ses = m.group(1), m.group(2)
    if UUID_RE.match(ses):
        ses = ses[:8]
    return f"sub-{sub}_ses-{ses}"


@dataclass
class NwbFile:
    path: Path
    rel: str
    port: str | None            # "neural" / "behavior" folder hint, or None
    has_units: bool = False
    has_ophys: bool = False
    has_behavior: bool = False
    session_id: str = ""
    sheet: dict = field(default_factory=dict)

    @property
    def has_neural(self) -> bool:
        return self.has_units or self.has_ophys


@dataclass
class Session:
    session_id: str
    neural: list[NwbFile]
    behavior: list[NwbFile]
    sheet: dict
    warnings: list[str] = field(default_factory=list)


@dataclass
class Found:
    files: list[NwbFile]
    sessions: list[Session]
    sheet_paths: list[Path]
    skipped: dict


def walk_files(root: Path):
    """Every file under root, following symlinked folders, without loops."""
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in sorted(filenames):
            if not f.startswith("."):
                yield Path(dirpath) / f


def read_sheets(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        try:
            with open(p, newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    row = {(k or "").strip(): (v or "").strip() for k, v in row.items() if k}
                    if row.get("session_id") and row.get("path"):
                        rows.append(row)
        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            log.warning("%s: unreadable sessions sheet (%s), ignored", p, exc)
    return rows


def match_sheet(path: Path, root: Path, rows: list[dict]) -> dict | None:
    names = {path.name, path.stem}
    try:
        names.update(p.name for p in path.relative_to(root).parents if p.name)
    except ValueError:
        pass
    relp = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    for row in rows:
        want = row["path"].rstrip("/")
        if want in names or Path(want).name in names or Path(want).stem in names or relp.endswith(want):
            return row
    return None


def session_id_for(path: Path, root: Path, rows: list[dict]) -> tuple[str, dict]:
    row = match_sheet(path, root, rows)
    if row:
        return sanitise(row["session_id"]), row
    sid = bids_session(path.name)
    if sid is None:
        try:
            for parent in path.relative_to(root).parents:
                if parent.name and (sid := bids_session(parent.name)):
                    break
        except ValueError:
            pass
    stem = path.name[:-4] if path.name.endswith(".nwb") else path.stem
    return sid or sanitise(stem), {}


def probe_content(path: Path) -> tuple[bool, bool, bool]:
    """(has units with spikes, has ophys traces, has behaviour) from the HDF5 layout, without pynwb."""
    import h5py

    with h5py.File(path, "r") as f:
        has_units = "units/spike_times" in f and f["units/spike_times"].shape[0] > 0
        has_ophys = False
        has_behavior = False
        for mod_name, mod in (f.get("processing") or {}).items():
            for name, obj in mod.items():
                nt = obj.attrs.get("neurodata_type", b"")
                nt = nt.decode() if isinstance(nt, bytes) else str(nt)
                if nt in ("Fluorescence", "DfOverF"):
                    has_ophys = True
                if nt in ("Position", "CompassDirection", "BehavioralTimeSeries", "PupilTracking",
                          "BehavioralEvents", "BehavioralEpochs", "PoseEstimation", "SpatialSeries",
                          "EyeTracking"):
                    has_behavior = True
                if nt == "TimeSeries" and any(k in name.lower() for k in ("wheel", "lick", "pupil", "speed", "position")):
                    has_behavior = True
        for name, obj in (f.get("acquisition") or {}).items():
            nt = obj.attrs.get("neurodata_type", b"")
            nt = nt.decode() if isinstance(nt, bytes) else str(nt)
            if nt in ("Position", "CompassDirection", "BehavioralTimeSeries", "PupilTracking", "SpatialSeries",
                      "PoseEstimation") or (nt == "TimeSeries" and any(
                    k in name.lower() for k in ("wheel", "lick", "pupil", "speed", "position"))):
                has_behavior = True
        ev = f.get("events")
        if ev is not None and len(ev):
            has_behavior = True
        intervals = f.get("intervals")
        if intervals is not None:
            for name, obj in intervals.items():
                if "start_time" in obj and obj["start_time"].shape[0] > 0:
                    has_behavior = True
    return has_units, has_ophys, has_behavior


def port_hint(path: Path, root: Path) -> str | None:
    try:
        first = path.relative_to(root).parts[0].lower()
    except (ValueError, IndexError):
        return None
    if first == NEURAL_PORT:
        return "neural"
    if first in BEHAVIOR_PORTS:
        return "behavior"
    return None


def discover(root: Path) -> Found:
    all_files = list(walk_files(root))
    sheet_paths = sorted((p for p in all_files if p.name == SHEET_NAME), key=lambda p: (len(p.parts), str(p)))
    rows = read_sheets(sheet_paths)
    files, skipped = [], {}
    for p in all_files:
        if not p.name.lower().endswith(".nwb"):
            continue
        relp = str(p.relative_to(root))
        try:
            u, o, b = probe_content(p)
        except Exception as exc:   # not HDF5 / unreadable: skipped, not fatal
            skipped[relp] = f"not readable as NWB/HDF5 ({type(exc).__name__}: {exc})"
            continue
        sid, row = session_id_for(p, root, rows)
        nf = NwbFile(path=p, rel=relp, port=port_hint(p, root), has_units=u, has_ophys=o, has_behavior=b,
                     session_id=sid, sheet=row or {})
        if not (u or o or b):
            skipped[relp] = "no units, ophys traces or behaviour found"
            continue
        files.append(nf)
    return Found(files=files, sessions=pair_sessions(files), sheet_paths=sheet_paths, skipped=skipped)


def pair_sessions(files: list[NwbFile]) -> list[Session]:
    by_sid: dict[str, list[NwbFile]] = {}
    for f in files:
        by_sid.setdefault(f.session_id, []).append(f)
    sessions = []
    for sid, group in by_sid.items():
        neural = [f for f in group if f.has_neural and f.port == "neural"] or \
                 [f for f in group if f.has_neural and f.port != "behavior"] or \
                 [f for f in group if f.has_neural]
        # Behaviour: files given as behaviour first, then the other files' own behaviour (the neural
        # file of a DANDI processed session has both); a variable found earlier wins.
        behavior = [f for f in group if f.has_behavior and f.port == "behavior"]
        behavior += [f for f in group if f.has_behavior and f not in behavior and f.port != "behavior"]
        # The same file linked into both ports (or twice) is read once.
        reals = {os.path.realpath(f.path) for f in neural}
        uniq, seen = [], set()
        for f in behavior:
            r = os.path.realpath(f.path)
            if r in seen:
                continue
            seen.add(r)
            if r in reals:
                f = next(n for n in neural if os.path.realpath(n.path) == r)
            uniq.append(f)
        behavior = uniq
        sheet = next((f.sheet for f in group if f.sheet), {})
        sessions.append(Session(sid, neural, behavior, sheet))
    # One neural-only and one behaviour-only session with different names: almost certainly a pair.
    n_only = [s for s in sessions if s.neural and not s.behavior]
    b_only = [s for s in sessions if s.behavior and not s.neural]
    if len(n_only) == 1 and len(b_only) == 1:
        n, b = n_only[0], b_only[0]
        n.behavior = b.behavior
        n.warnings.append(f"neural session {n.session_id!r} and behaviour session {b.session_id!r} have different "
                          f"names; paired them because each was the only one of its kind (add a sessions.csv to "
                          f"pair explicitly)")
        sessions.remove(b)
    return sorted(sessions, key=lambda s: s.session_id)
