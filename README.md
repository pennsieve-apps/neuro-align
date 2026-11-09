# neuro-align

neuro-align is built on **[pynapple](https://github.com/pynapple-org/pynapple)**, developed by
Guillaume Viejo, Adrien Peyrache and colleagues (Peyrache lab, Montreal Neurological Institute,
McGill University, and the Flatiron Institute; [Viejo et al. 2023, eLife](https://doi.org/10.7554/eLife.85786)),
and on the **[Neurodata Without Borders](https://www.nwb.org)** reader
[pynwb](https://github.com/NeurodataWithoutBorders/pynwb) (Rübel, Tritt, Ly, Dichter and colleagues,
[Rübel et al. 2022, eLife](https://doi.org/10.7554/eLife.78362)) with the
[ndx-pose](https://github.com/rly/ndx-pose) extension (Ryan Ly, Ben Dichter, Alexander Mathis and
contributors). Pennsieve packages these tools as a workflow step; it did not develop them. The demo
data are DANDI datasets by Grosmark & Buzsáki (000044), the International Brain Laboratory (000409)
and Plusnin, Pospelov et al. (002047); see [Citations](#citations).

The first step of the **Neural Population Dynamics & Decoding** workflow:

- **Neural Population Dynamics & Decoding:** `neuro-align → {neuro-encoding, neuro-embedding, neuro-decoding} → neuro-report`

**For a neuroscientist.** To ask what neurons encode, or to decode behaviour from them, neural
activity and behaviour have to sit on the same time axis, in the same time bins, over the same part
of the session. That bookkeeping is where analyses quietly go wrong: a pose file whose clock starts
at zero while the spikes start at 1000 s, a wheel sampled only when it turns, a `rate` attribute
that is really a sampling period, a virtual-reality position of -500 between laps, noise units
mixed in with good ones. neuro-align reads a session's NWB files (spike-sorted units or calcium
imaging ROIs, plus behaviour: position, speed, head direction, pose keypoints, wheel, pupil, licks,
trials), cleans them, bins them on the session clock, marks the part of the session to analyse
(everything, trials, or running) and tells you, with figures, what it kept and what looks wrong.

**For a computational analyst.** Per session: `aligned.npz` with `t` (bin centres, s, session
clock), `X` (bins × neurons, float32: spike counts per bin from pynapple `TsGroup.count`, or the
mean ROI activity per bin), `Y` (bins × behaviour variables, float32, NaN where unknown),
`neuron_ids`, `neuron_meta` (structured: id, region, label, cell_type, depth, x, y, rate_hz),
`behavior_names`/`behavior_units`/`behavior_kinds`, `epoch_mask` (bool per bin), `bin_size`;
`spikes.npz` (all spike times of the kept units inside the window: `times`, `offsets`,
`neuron_ids`); `events.parquet` (every TimeIntervals row: trials, laps, wheel movements, sleep
states, ... and point events); `manifest.json` (what came from where, units, filters, counts
dropped, clock checks). All bins of the aligned window are written; `epoch_mask` says which ones
to analyse, so downstream apps can also use the rest.

## Ports

| Port | Direction | Content |
|---|---|---|
| `neural` | in (required) | NWB with a `Units` table (`spike_times`) or `processing/ophys` ROI traces (`Deconvolved`, `DfOverF`, `Fluorescence` + `PlaneSegmentation`): ephys-curate's units NWB, ophys-traces' ophys NWB, or a DANDI file |
| `behavior` | in (optional) | NWB with `processing/behavior` (Position, CompassDirection, BehavioralTimeSeries, PupilTracking, BehavioralEvents, ndx-pose PoseEstimation, IBL wheel) and/or TimeIntervals / EventsTables: pose-kinematics' or pose-estimate's NWB or a DANDI file. A neural file that also holds behaviour (DANDI processed files) is used as is |
| `sessions` | in (optional) | `sessions.csv` (`session_id, path, subject_id, condition, group, ...`) |
| `aligned` | out | `neuro-align/<session_id>/{aligned.npz, spikes.npz, events.parquet, manifest.json, *_raster.png, *_regions.png, *_timeline.png}`, `neuro-align/{sessions.csv, summary.json, provenance.json}` |

Files are found by content, recursively, following symlinks; the `neural/` and `behavior/` folders
are only hints. Sessions are named per conventions.md (sessions.csv, else `sub-<x>_ses-<y>`, else
the file stem) and neural and behaviour files are joined on that name. When there is exactly one
neural-only and one behaviour-only session, they are paired with a warning.

## Method

**Neural data.** `NEURAL_SOURCE=auto` takes units when a file has spike times, else ROIs.

- *Units*: kept when their curation label is in `UNIT_LABELS` (default `good`), read from the
  first of `label`, `quality`, `ibl_quality_score` (IBL: 1 = passes every single-unit metric),
  `KSLabel`, `bombcell_label`, `phy_annotations` (Steinmetz et al.: ≥ 2 = good); when the column
  exists but no unit passes, all are kept with a warning. Region from `brain_area` / `location` /
  ... columns, else the most common `location` of the unit's electrodes, else the electrode group;
  `BRAIN_AREAS` keeps regions starting with the given names. Units under `MIN_RATE_HZ` in the
  aligned window are dropped. Several units files of one session (one per probe) are merged.
- *ROIs*: the first available of a deconvolved signal (names with deconv/spks/spike/event), dF/F,
  fluorescence (`ROI_TRACE` overrides); neuropil, background and residual series are never used.
  Several series over one PlaneSegmentation are alternative signals (CaImAn/Minian `C`, `events`,
  `residual`); series over different PlaneSegmentations are planes and are put on one grid. Only
  ROIs with `accepted` (or `iscell` > 0.5) are kept; flat or empty traces are dropped.

**Behaviour.** From every behaviour file of the session (the first source of a variable wins):
`x`, `y` (Position; if there is none, a body-centre keypoint, or the keypoint centroid, of a
single-camera pose without a wheel); `position` (a 1D position in the file, or 2D position
linearised on its long axis when the path is a linear track, `LINEARIZE`; when the file's 1D
position covers less than half the time the 2D position does, as in Grosmark's lap-only
`linearized_position`, the linearised 2D position is used and the file's own is kept under its
name); `speed` (a recorded running speed, else from position); `run_direction`: on a linear track
laps are detected (from the last moment in one end zone, the outer 10% of the track, to the first
moment in the other; 0.5-60 s; written to events.parquet as `laps_detected`) and `run_direction`
is +1/−1 inside a lap while moving along the track at ≥ `RUN_SPEED`, 0 elsewhere (end platforms,
pauses, turns, grooming); without laps it is the sign of the velocity while faster than
`RUN_SPEED`; `head_direction` (CompassDirection or heading series; degrees converted to radians, binned
as a circular mean); `wheel_position` (zero-order hold, since encoders write a sample only when the
wheel turns) and `wheel_velocity`; `pupil` (diameter, else area); `licks` (count per bin);
`<keypoint>_speed` for ndx-pose keypoints (confidence below `POSE_MIN_CONFIDENCE` removed, pupil
keypoints, static landmarks and keypoints tracked less than half the time skipped, the fastest
camera used when several see the same keypoint). Fast signals are averaged per bin, slow ones
interpolated at bin centres; gaps longer than max(0.5 s, 3 samples, 2 bins) stay NaN. Speeds and
velocities are derived after Gaussian smoothing (`SMOOTH_S`), with teleports and wraps removed.
A constant off-track value at the edge of a 1D position (e.g. −500 between virtual laps) is
removed.

**Time.** All times stay on the neural file's session clock. The aligned window is where neural
data and behaviour overlap; bins are `BIN_SIZE_S` wide (25 ms for units, one imaging frame for
ROIs, aligned on the frames). Checks: a behaviour file whose `timestamps_reference_time` differs
from the neural file's is shifted by the difference when that makes it overlap (warning either
way); a `rate` attribute that is really the sampling period (some DANDI files) is inverted; a
warning when less than 90% of a behaviour file lies inside the neural recording (a CLOCK warning
below 50%) or when behaviour continues past the last spike (IBL: units stop at 2001.7 s, the window
stops there); the session fails when they do not overlap at all.

**Epoch.** `EPOCH=all`, `trials`, `run` (on a linear track: running along the track inside
detected laps; elsewhere smoothed speed above `RUN_SPEED`, 3 cm/s by default, gaps < 0.5 s closed
and runs < 0.5 s dropped, with pynapple IntervalSets) or the name of any TimeIntervals table (e.g.
`laps_detected`, `WheelMovementIntervals`). A speed threshold alone is not enough on a track:
head-tracking jitter on Grosmark's end platforms exceeds 3 cm/s 75% of the time off the laps.

## Parameters

| Name | Default | What it does |
|---|---|---|
| `NEURAL_SOURCE` | `auto` | `units`, `roi` or `auto` |
| `ROI_TRACE` | `auto` | `deconvolved`, `dff`, `fluorescence` or `auto` (in that order) |
| `UNIT_LABELS` | `good` | curation labels kept (comma-separated); `all` keeps every unit |
| `MIN_RATE_HZ` | `0.1` | drop units firing less in the aligned window |
| `BRAIN_AREAS` | (empty) | keep neurons whose region starts with one of these (comma-separated) |
| `BEHAVIOR_VARS` | `auto` | `auto`, or names to keep (`position,speed,keypoint_speeds,...`); other NWB series names are added (`auto,face_motion_energy`) |
| `LINEARIZE` | `auto` | linearise 2D position on a linear track: `auto`, `true`, `false` |
| `BIN_SIZE_S` | `auto` | bin width (s), `frame`, or `auto` (0.025 s units / frame for ROIs) |
| `EPOCH` | `all` | `all`, `trials`, `run`, or a TimeIntervals table name |
| `RUN_SPEED` | `auto` | running threshold in the speed's unit (`auto` = 3 cm/s) |
| `SMOOTH_S` | `0.1` | smoothing (s) before speeds/velocities are derived |
| `POSE_MIN_CONFIDENCE` | `0.5` | keypoints below this confidence are missing |
| `MAX_DURATION_S` | `0` | cap the aligned window (s); 0 = none |
| `THREADS` | `0` | BLAS/numba threads; 0 = the container's CPU quota |

## Outputs

`neuro-align/<session_id>/aligned.npz`, `spikes.npz`, `events.parquet` (columns `table,
start_time, stop_time` + the table's own columns), `manifest.json` (`bin_size_s`, `n_bins`,
`t_start`, `t_stop`, `neural` {kind, source, signal, files, n_neurons, regions, filters, dropped},
`behavior` {variables [{name, kind, unit, source, circular, range, fraction_valid}],
linear_track}, `epoch`, `events`, `clock` {neural_range_s, behavior files with range, overlap and
shift, window_s}, `source` identity, `sheet`, `warnings`); figures:

- `<session_id>_raster.png`: spike raster coloured by region (or ROI heatmap) over the most active
  60 s, with the behaviour traces below on the same clock.
- `<session_id>_regions.png`: neurons kept and dropped per brain region.
- `<session_id>_timeline.png`: the time ranges of the neural data and of each behaviour file, and
  the aligned window.

`neuro-align/sessions.csv` (`session_id, path, behavior_paths, subject_id, condition, group, status,
neural_kind, n_neurons, n_bins, bin_size_s, n_behavior_vars, epoch_fraction` + sheet columns),
`summary.json` (headline numbers per session, figures with captions, warnings), `provenance.json`.
Behaviour variable kinds (`behavior_kinds`): `position`, `position_x`, `position_y`, `speed`,
`direction`, `angle`, `wheel_position`, `wheel_velocity`, `pupil`, `licks`, `keypoint_speed`,
`other`.

## Demo results

| Session | Neural | Behaviour | Window |
|---|---|---|---|
| Grosmark & Buzsáki, `sub-Buddy_ses-Buddy-06272013` (DANDI 000044), `EPOCH=run` | 58 CA1 units (38 pyramidal, 20 interneurons; 68 in the file, 10 below 0.1 Hz on the maze) | x, y (m), track position linearised from x, y (the file's `linearized_position` covers only 7% of the maze time, inside laps), speed, run direction | 2328 s (the maze epoch: position exists only there), 93,117 bins of 25 ms; 57 laps detected (328 s; the authors' trials table has 66 laps, 218 s, 211 s of which fall in detected laps); running inside laps 266 s |
| IBL `sub-NR-0031_ses-caa5dddc` (DANDI 000409) | 186 units with `ibl_quality_score` = 1 of 1359 (Anterior pretectal nucleus 70, Midbrain reticular nucleus 35, Field CA1 31, ...) | wheel position/velocity (rad), pupil diameter, licks, paw and nose speeds (Lightning Pose), 404 trials + wheel movements | 0.1–2001.7 s (warning: behaviour continues 84 s past the last spike) |
| CA1 1p calcium, WT and 5xFAD, 6 sessions (DANDI 002047) | 42–264 accepted ROIs (`events`, DRIADA calcium events) | x, y (cm), speed, head direction, 13 keypoint speeds | 719 s each at the 29.8 Hz frame rate |
| Steinmetz et al. 2019 `sub-Richards` (DANDI 000017) | 490 units (`phy_annotations` ≥ 2; ORB, MOs, OLF, root) | wheel (mm; its `rate` attribute was a period: corrected), pupil area, licks, 260 trials | 2955 s |
| Sosa et al. 2025 `sub-m11_ses-04` (DANDI 001361), `EPOCH=run` | 168 accepted ROIs (Deconvolved) | 1D virtual track (−500 off-track values removed), speed, licks | 1352 s at 15.5 Hz; 80 laps detected; running inside laps 702 s (52%) |

Synthetic fixture (`make test`): the 'mua' and near-silent units are dropped, the 2D track is
detected and linearised, all 33 simulated laps are found, spike-weighted mean positions of place cells are within 10 cm of their
planted centres (median), speed cells correlate with speed, rejected ROIs and the off-track
value are removed, and behaviour on another clock fails the session with a clear message.

## Runtime

`cpu: 2048` (2 vCPU), `memory: 4096` MiB, `timeoutSeconds: 900`, `standard`. Measured on 2026-11-09
with the image built from the commit that added this table (the HOME fix after `4d80f17`), in
Docker (linux/amd64 under emulation on an M-series Mac, `--cpus 2 --memory 4g`, read-only root,
`/tmp` tmpfs, no HOME set), while neuro-encoding was running natively on the same machine, so the
wall times are pessimistic. Peak memory is the container's cgroup `memory.peak` (it includes the
page cache of the NWB file being read). Grosmark and Sosa with `EPOCH=run`, the others with the
defaults:

| Data | Wall time | Peak memory |
|---|---|---|
| Synthetic fixture (2 sessions) | 49 s | 848 MB |
| IBL session (186 units, 2002 s, 7 behaviour variables) | 128 s | 1346 MB |
| Grosmark (5.4 GB NWB, 58 units, 2328 s, 57 laps) | 65 s | 662 MB |
| 002047, 6 calcium sessions in one run | 145 s | 1287 MB |
| Steinmetz `sub-Richards` (490 units, 2955 s) | 123 s | 1226 MB |
| Sosa `sub-m11_ses-04` (168 ROIs) | 54 s | 752 MB |

Within the Lambda limits (≤ 10240 MiB, ≤ 6 vCPU, ≤ 900 s), so the platform may run it as a Lambda
step. It is Lambda-safe: it writes only to `OUTPUT_DIR` and `$TMPDIR` (matplotlib and numba caches
are set under `$TMPDIR` at startup; the font cache is baked into `/opt/neuro-align/mpl`), and the
fixture runs with `docker run --read-only --user 65534:65534 --tmpfs /tmp` and `HOME`, `TMPDIR`,
`INPUT_DIR`, `OUTPUT_DIR` under `/tmp`. Memory grows with bins × neurons (`X` is float32): 2 hours of 600 units at 25 ms is 1.7 GB, so
use `MAX_DURATION_S` or a larger `BIN_SIZE_S` for very long sessions. Image: 1.34 GB.

## Limitations

- It aligns; it does not repair. If a pose file and a units file are on genuinely different clocks
  with the same `timestamps_reference_time`, nothing can tell, except the warnings and the
  timeline figure. Check them.
- Linearisation is a projection on the long axis, right for straight tracks only; for W-mazes,
  circular or T-mazes use the authors' linearised position (a 1D Position series is preferred when
  present) or `LINEARIZE=false`.
- Variable recognition is by NWB type and series name (`speed`, `heading`, `wheel`, `pupil`,
  `lick`, ...). Unrecognised series are left out unless named in `BEHAVIOR_VARS`.
- Unit quality labels are read as found; units without labels are all kept (with `MIN_RATE_HZ`).
- ROI activity is averaged per bin as it is in the file (deconvolved events, dF/F or raw
  fluorescence are not comparable across ROIs of different files).
- `spikes.npz` holds spike times inside the aligned window only; for ROI data there is none.

## Running it locally

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python      # pytest on the synthetic fixture (no Docker)
make run PYTHON=.venv/bin/python       # build the image and run it over data/fixture/linked/
```

## Licences

Our code is Apache-2.0 (`LICENSE`). pynwb, hdmf, ndx-pose, NumPy, SciPy, pandas and h5py are
BSD-3-Clause, pynapple MIT, pyarrow Apache-2.0, Matplotlib its PSF-based licence. Nothing in the
image is GPL. Details in `THIRD_PARTY.md`. The demo datasets are CC-BY-4.0.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author
lists in `CITATION.cff`.

- **pynapple**: Viejo G, Levenstein D, Skromne Carrasco S, Mehrotra D, Mahallati S, Vite GR,
  Denny H, Sjulson L, Battaglia FP, Peyrache A. Pynapple, a toolbox for data analysis in
  neuroscience. *eLife* 2023;12:e85786. doi:10.7554/eLife.85786, PMID 37843985.
- **NWB / pynwb**: Rübel O, Tritt A, Ly R, Dichter BK, Ghosh S, Niu L, Baker P, Soltesz I, Ng L,
  Svoboda K, Frank L, Bouchard KE. The Neurodata Without Borders ecosystem for neurophysiological
  data science. *eLife* 2022;11:e78362. doi:10.7554/eLife.78362, PMID 36193886.
- **ndx-pose** (software, no paper): Ly R, Dichter B, Mathis A, Maree L, Brozdowski C, Mayorquin H,
  Pereira T, Berrigan E, Adkisson P, Trapani A. https://github.com/rly/ndx-pose.
- **Demo data, linear track**: Grosmark AD, Buzsáki G. Diversity in neural firing dynamics supports
  both rigid and learned hippocampal sequences. *Science* 2016;351(6280):1440-1443.
  doi:10.1126/science.aad1935, PMID 27013730. Dataset: Grosmark AD, Buzsáki G, DANDI 000044,
  doi:10.48324/dandi.000044/0.261106.0236.
- **Demo data, IBL**: International Brain Laboratory et al. A brain-wide map of neural activity
  during complex behaviour. *Nature* 2025;645:177-191. doi:10.1038/s41586-025-09235-0,
  PMID 40903598. Dataset: International Brain Laboratory, IBL - Brain Wide Map, DANDI 000409,
  doi:10.48324/dandi.000409/0.260309.1324.
- **Demo data, CA1 calcium in 5xFAD and WT mice**: Plusnin V, Pospelov N, DANDI 002047,
  doi:10.48324/dandi.002047/0.261106.0903.
- **Test data**: Steinmetz NA, Zatka-Haas P, Carandini M, Harris KD. Distributed coding of choice,
  action and engagement across the mouse brain. *Nature* 2019;576(7786):266-273.
  doi:10.1038/s41586-019-1787-x, PMID 31776518; dataset DANDI 000017,
  doi:10.48324/dandi.000017/0.240329.1926. Sosa M, Plitt MH, Giocomo LM. A flexible hippocampal
  population code for experience relative to reward. *Nat Neurosci* 2025;28(7):1497-1509.
  doi:10.1038/s41593-025-01985-4, PMID 40500314; dataset DANDI 001361 (draft, no DOI yet),
  https://dandiarchive.org/dandiset/001361/draft.

## Acknowledgements

We thank the pynapple team (Peyrache lab and Flatiron Institute), the Neurodata Without Borders
team and the NWB community, the ndx-pose authors, and the DANDI Archive. The demo data were shared
by Grosmark and Buzsáki, the International Brain Laboratory, Plusnin, Pospelov and colleagues,
Steinmetz, Carandini, Harris and colleagues, and Sosa, Plitt and Giocomo, under CC-BY-4.0.
