# sample-input

neuro-align reads NWB files: `neural` (units or calcium-imaging ROIs) and, optionally, `behavior`
(any NWB with behaviour; the same file is fine). Linked into a workflow they arrive in
`$INPUT_DIR/neural/` and `$INPUT_DIR/behavior/`; unlinked, side by side anywhere under `$INPUT_DIR`.

## Synthetic fixture (no download)

```sh
make fixture PYTHON=.venv/bin/python   # data/fixture/{linked,flat}/ + data/fixture/ground_truth.json
make run PYTHON=.venv/bin/python       # Docker over data/fixture/linked/ -> data/output/
```

- `sub-synth01_ses-track`: a units NWB (40 place cells, 10 speed cells, 10 untuned, 5 'mua' and 3
  near-silent units that must be dropped) plus a separate behaviour NWB (2D position on a 200 cm
  linear track, laps and trials tables, licks, ndx-pose keypoints). Expected: 60 units, position
  linearised, place-cell spikes centred on their ground-truth fields.
- `sub-synth02_ses-vr`: one behavior+ophys NWB (36 ROIs, 6 rejected by `iscell`; Deconvolved and
  DfOverF at 15.5 Hz; 1D virtual-track position with a -500 'off track' value). Expected: 30 ROIs,
  15.5 Hz bins, the off-track value removed.

## Real data (DANDI, CC-BY-4.0)

The demo data of the workflows (`scripts/stage-demo-data.py` in the workflow repository):

- **Grosmark & Buzsáki 2016**, DANDI 000044: `sub-Buddy/sub-Buddy_ses-Buddy-06272013_behavior+ecephys.nwb`
  (5.4 GB; CA1 units, linear-track position). Give it as `neural` alone.
- **IBL Brain Wide Map**, DANDI 000409: `sub-NR-0031/sub-NR-0031_ses-caa5dddc-..._desc-processed_behavior+ecephys.nwb`
  (units, trials, wheel, pupil, licks, Lightning Pose). Give it as `neural` alone.
- **Hippocampal CA1 1p calcium + pose, WT and 5xFAD**, DANDI 002047: six `..._behavior+image+ophys.nwb`
  files with a `sessions.csv` (`condition` = genotype).

Small files to try first: DANDI 000017 (Steinmetz et al. 2019)
`sub-Richards/sub-Richards_ses-20171031T120000.nwb` (204 MB: units in 8 regions, wheel, pupil,
licks, trials) and DANDI 001361 (Sosa, Plitt & Giocomo 2025) `sub-m11/sub-m11_ses-04_behavior+ophys.nwb`
(90 MB: 2p CA1 on a virtual track; use `EPOCH=run`).

```sh
dandi download "https://dandiarchive.org/dandiset/000017/draft" --path sub-Richards/sub-Richards_ses-20171031T120000.nwb
docker run --rm --platform linux/amd64 --network none --memory 4g \
  -e INPUT_DIR=/data/input -e OUTPUT_DIR=/data/output \
  -v "$PWD/sub-Richards_ses-20171031T120000.nwb:/data/input/neural/sub-Richards_ses-20171031T120000.nwb:ro" \
  -v "$PWD/out:/data/output" neuro-align
```
