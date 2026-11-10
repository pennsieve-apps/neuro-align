"""Small helpers shared by the processor modules: threads, JSON, versions, memory."""
from __future__ import annotations

import json
import math
import os
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def cgroup_cpus() -> int:
    """CPUs from the cgroup quota (v2 cpu.max, v1 cfs), else the affinity mask; never the host's count alone."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, math.floor(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        per = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0 and per > 0:
            return max(1, q // per)
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def runtime_env(app: str) -> str:
    """Writable caches under $TMPDIR (the only writable places on a Lambda step are INPUT_DIR, OUTPUT_DIR,
    HOME and TMPDIR); the matplotlib font cache baked into /opt/<app>/mpl is copied there, not rebuilt."""
    import shutil
    import tempfile

    tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
    # A HOME that cannot be written (the image's user has none; ECS may not set one) breaks libraries that
    # keep a cache there (pynwb/hdmf via platformdirs): use one under TMPDIR instead.
    home = os.environ.get("HOME", "")
    if not home or not os.path.isdir(home) or not os.access(home, os.W_OK):
        home = os.path.join(tmp, "home")
        os.makedirs(home, exist_ok=True)
        os.environ["HOME"] = home
    if not os.access(os.environ.get("XDG_CACHE_HOME") or home, os.W_OK):
        os.environ["XDG_CACHE_HOME"] = os.path.join(tmp, "cache")
    mpl = os.environ.setdefault("MPLCONFIGDIR", os.path.join(tmp, "mpl"))
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tmp, "numba"))
    baked = Path("/opt") / app / "mpl"
    try:
        Path(mpl).mkdir(parents=True, exist_ok=True)
        if baked.is_dir() and not any(Path(mpl).glob("fontlist*.json")):
            for f in baked.glob("fontlist*.json"):
                shutil.copy(f, mpl)
    except OSError:
        pass
    return tmp


def set_threads(threads: int) -> int:
    """Set the BLAS/OpenMP/numba thread counts (before numpy is imported where possible)."""
    n = threads if threads > 0 else cgroup_cpus()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = str(n)
    return n


def clean_json(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return clean_json(obj.tolist())
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return clean_json(obj.item())
        except (ValueError, AttributeError):
            return str(obj)
    return obj


def write_json(path: Path, data) -> None:
    with open(path, "w") as fh:
        json.dump(clean_json(data), fh, indent=2)
        fh.write("\n")


def peak_rss_mb() -> float | None:
    """Peak memory: the container's cgroup peak when readable, else this process's max RSS."""
    for p in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return round(int(Path(p).read_text().strip()) / 2**20, 1)
        except (OSError, ValueError):
            pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (2**20 if sys.platform == "darwin" else 1024), 1)   # bytes on macOS, KiB on Linux
    except (ImportError, OSError):
        return None


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions(packages) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for pkg in packages:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def rel(path, root: Path):
    if path is None:
        return None
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)
