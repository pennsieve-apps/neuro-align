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
    USER=app \
    MPLBACKEND=Agg

# A passwd entry for uid 1000 (ECS); on Lambda the step runs as another least-privileged user, with HOME,
# TMPDIR, INPUT_DIR and OUTPUT_DIR under /tmp and everything else read-only. Writable caches are set under
# $TMPDIR at startup (processor/common.py runtime_env); the matplotlib font cache is built here into
# /opt/neuro-align/mpl and copied from there. Also checks that the stack imports.
RUN useradd --uid 1000 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin app \
    && mkdir -p /opt/neuro-align/mpl \
    && MPLCONFIGDIR=/opt/neuro-align/mpl NUMBA_CACHE_DIR=/opt/neuro-align/numba HOME=/opt/neuro-align python -c "import matplotlib.pyplot, pynwb, ndx_pose, pynapple, pyarrow" \
    && rm -rf /opt/neuro-align/numba /opt/neuro-align/.cache && chmod -R a+rX /opt/neuro-align

USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
