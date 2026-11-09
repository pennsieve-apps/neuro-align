# Third-party software

This app is Apache-2.0 (see `LICENSE`). The image it builds contains the following third-party
components, installed from PyPI at the pinned versions in `requirements.txt`; their licences apply
to the image. Nothing here is GPL. No model weights or data are bundled.

| Component | What | Authors / copyright | Licence | Source |
|---|---|---|---|---|
| pynwb 4.2.0, hdmf 6.2.0 | Reading NWB files (Units, ophys, behaviour, TimeIntervals). Called as libraries. | The NWB team / Neurodata Without Borders (Lawrence Berkeley National Laboratory and contributors) | BSD-3-Clause | https://github.com/NeurodataWithoutBorders/pynwb, https://github.com/hdmf-dev/hdmf |
| ndx-pose 0.4.0 | The NWB extension for pose estimation (PoseEstimation, PoseEstimationSeries, Skeletons), imported so pose files are read with their types. | Ryan Ly, Ben Dichter, Alexander Mathis, Liezl Maree, Chris Brozdowski, Heberto Mayorquin, Talmo Pereira, Elizabeth Berrigan, Paul Adkisson, Alessandra Trapani | BSD-3-Clause | https://github.com/rly/ndx-pose |
| pynapple 0.11.4 | Time series objects, spike counting in bins (`TsGroup.count`), interval sets (run epochs, trials). | Guillaume Viejo, Adrien Peyrache and contributors (Peyrache lab, McGill University; Flatiron Institute) | MIT | https://github.com/pynapple-org/pynapple |
| NumPy 2.5.4, SciPy 1.18.1 | Arrays, sparse binning, Gaussian smoothing. | NumPy and SciPy developers | BSD-3-Clause | https://numpy.org, https://scipy.org |
| pandas 3.1.1, pyarrow 26.0.1 | events.parquet. | pandas developers; Apache Software Foundation | BSD-3-Clause; Apache-2.0 | https://pandas.pydata.org, https://arrow.apache.org |
| h5py 3.16.0 | HDF5 access (content probing, pynwb backend). | Andrew Collette and contributors | BSD-3-Clause | https://www.h5py.org |
| Matplotlib 3.11.2, Pillow 13.0.0 | QC figures. | Matplotlib development team; Jeffrey A. Clark and contributors | Matplotlib licence (PSF-based); MIT-CMU | https://matplotlib.org, https://python-pillow.org |
| numba 0.68.1, llvmlite 0.50.0, xarray 2026.9.0, scikit-learn 1.9.1, joblib, threadpoolctl, rich, tabulate, narwhals | Dependencies of pynapple (versions in `requirements.txt`). | Their developers | BSD-2/3-Clause, Apache-2.0 (xarray), MIT (rich, tabulate, narwhals) | PyPI |
| jsonschema, referencing, rpds-py, ruamel.yaml, attrs, contourpy, cycler, fonttools, kiwisolver, packaging, pyparsing, python-dateutil, six, typing_extensions, tzdata, platformdirs, Pygments, markdown-it-py, mdurl | Dependencies of hdmf, pandas and Matplotlib. | Their developers | MIT / BSD / Apache-2.0 / PSF-2.0 | PyPI |
| python:3.12.11-slim-bookworm | Base image (Debian 12 + CPython 3.12). | Python Software Foundation; Debian | PSF (Python); Debian packages under their own free licences | https://hub.docker.com/_/python |
