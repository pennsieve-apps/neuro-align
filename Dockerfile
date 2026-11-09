FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /processor

# Pinned Python dependencies (all wheels): pynwb + ndx-pose to read NWB, pynapple to bin, matplotlib for figures.
COPY requirements.txt /processor/requirements.txt
RUN pip install --only-binary=:all: -r /processor/requirements.txt

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba

# A passwd entry for uid 1000 (some libraries call getpass); build matplotlib's font cache now and check that
# the stack imports, so a run needs no writes outside /tmp and no network.
RUN useradd --uid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin processor \
    && python -c "import matplotlib.pyplot, pynwb, ndx_pose, pynapple, pyarrow" \
    && mkdir -p /tmp/numba /tmp/.cache && chmod -R a+rwX /tmp/matplotlib /tmp/numba /tmp/.cache

USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
