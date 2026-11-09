"""QC figures: raster/heatmap + behaviour traces over a 60 s window, neuron counts by region, time coverage."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 150
# Categorical slots in fixed order (identity), neutral inks for text, one-hue ramp for magnitude.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = "Blues"
WINDOW_S = 60.0
TRACE_ORDER = ("position", "x", "y", "speed", "head_direction", "wheel_velocity", "wheel_position", "pupil", "licks")

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9, "axes.titleweight": "bold",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def region_colors(regions: list[str]) -> dict:
    """Regions by neuron count, each a fixed slot; past eight they fold to grey ('other')."""
    u, c = np.unique(regions, return_counts=True)
    order = [r for _, r in sorted(zip(-c, u))]
    return {r: (SERIES[i] if i < len(SERIES) else OTHER) for i, r in enumerate(order)}


def pick_window(t: np.ndarray, epoch: np.ndarray, Y: np.ndarray, names: list[str]) -> tuple[float, float]:
    """The 60 s window with the most behaviour going on (variance of the first trace) inside the epoch."""
    t0, t1 = t[0], t[-1]
    if t1 - t0 <= WINDOW_S:
        return float(t0), float(t1)
    bs = t[1] - t[0]
    w = int(round(WINDOW_S / bs))
    j = next((names.index(n) for n in TRACE_ORDER if n in names), None)
    score = epoch.astype(float)
    if j is not None:
        y = Y[:, j]
        ok = np.isfinite(y)
        yz = np.where(ok, y - np.nanmean(y), 0.0)
        score = score * ok + 0.5 * (yz ** 2) / (np.nanvar(y) + 1e-12) * ok
    cs = np.r_[0, np.cumsum(score)]
    tot = cs[w:] - cs[:-w]
    step = max(1, int(round(10.0 / bs)))
    k = int(np.argmax(tot[::step]) * step)
    return float(t0 + k * bs), float(t0 + k * bs + WINDOW_S)


def raster(path: Path, sid: str, kind: str, t: np.ndarray, X: np.ndarray, spikes, meta, Y: np.ndarray,
           names: list[str], units: list[str], epoch: np.ndarray, bin_size: float) -> tuple[Path, str]:
    a, b = pick_window(t, epoch, Y, names)
    sel = (t >= a) & (t < b)
    traces = [n for n in TRACE_ORDER if n in names]
    traces += [n for n in names if n not in traces and n != "run_direction"]
    traces = traces[:5]
    regions = [m["region"] for m in meta]
    depth = np.array([m["depth"] for m in meta], dtype=float)
    order = sorted(range(len(meta)), key=lambda i: (regions[i], -np.nan_to_num(depth[i], nan=0.0), i))
    cmap = region_colors(regions)
    nrow = 1 + len(traces)
    fig = plt.figure(figsize=(9, 4.2 + 0.75 * len(traces)))
    gs = fig.add_gridspec(nrow, 1, height_ratios=[4] + [1] * len(traces), hspace=0.18)
    ax = fig.add_subplot(gs[0])
    if kind == "units" and spikes is not None:
        for row, i in enumerate(order):
            s = spikes[i]
            s = s[(s >= a) & (s < b)]
            ax.scatter(s, np.full(len(s), row), s=1.2, marker="|", linewidths=0.5, color=cmap[regions[i]],
                       rasterized=True)
        ax.set_ylim(-1, len(order))
        ax.set_ylabel(f"units ({len(order)})")
    else:
        Z = X[sel][:, order].astype(float)
        mu, sd = np.nanmean(Z, axis=0), np.nanstd(Z, axis=0)
        Z = (Z - mu) / np.where(sd > 0, sd, 1)
        im = ax.imshow(Z.T, aspect="auto", origin="lower", cmap=SEQ, vmin=0, vmax=3,
                       extent=[t[sel][0], t[sel][-1] + bin_size, -0.5, len(order) - 0.5], interpolation="nearest")
        ax.set_ylabel(f"ROIs ({len(order)})")
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.set_label("activity (z)")
    # epoch shading
    m = epoch[sel]
    tt = t[sel]
    if m.any() and not m.all():
        ax.fill_between(tt, -1, len(order), where=m, color=GRID, alpha=0.6, step="mid", zorder=0, linewidth=0)
    ax.set_xlim(a, b)
    ax.tick_params(labelbottom=False)
    ax.set_title(f"{sid}: {kind} and behaviour, {a:.0f}-{b:.0f} s (session clock)", loc="left")
    handles = [plt.Line2D([], [], color=c, marker="s", linestyle="", markersize=6, label=f"{r}")
               for r, c in list(cmap.items())[:8]]
    if handles and kind == "units":
        ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1),
                  title="region", title_fontsize=7)
    for k, n in enumerate(traces):
        axk = fig.add_subplot(gs[k + 1], sharex=ax)
        j = names.index(n)
        y = Y[sel, j]
        if n == "licks":
            ev = t[sel][np.nan_to_num(y) > 0] + bin_size / 2
            axk.vlines(ev, 0, 1, color=INK2, linewidth=0.6)
            axk.set_yticks([])
        else:
            axk.plot(t[sel], y, color=INK2, linewidth=1.0)
            f = y[np.isfinite(y)]
            if len(f) > 10:
                lo, hi = np.percentile(f, [0.5, 99.5])
                if hi > lo:
                    pad = 0.1 * (hi - lo)
                    axk.set_ylim(lo - pad, hi + pad)
        axk.set_ylabel(f"{n}\n({units[j]})" if units[j] and n != "licks" else n, rotation=0, ha="right", va="center", fontsize=7)
        axk.grid(axis="y", color=GRID, linewidth=0.5)
        if k < len(traces) - 1:
            axk.tick_params(labelbottom=False)
    fig.axes[-1].set_xlabel("time (s)")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    what = "spike raster (one row per unit, coloured by brain region, sorted by region and depth)" if kind == "units" \
        else "ROI activity (z-scored per ROI)"
    cap = (f"{sid}: {what} over the {b - a:.0f} s with the most behaviour, above the behaviour variables on the same "
           f"session clock. Activity should change with behaviour (e.g. bursts as the animal runs through a place "
           f"field); flat rows or behaviour that looks shifted in time point to a clock or unit-selection problem."
           + (" Grey shading marks the analysis epoch." if m.any() and not m.all() else ""))
    return path, cap


def counts_by_region(path: Path, sid: str, kind: str, kept: dict, dropped: dict) -> tuple[Path, str]:
    regions = sorted(set(kept) | set(dropped), key=lambda r: -(kept.get(r, 0) + dropped.get(r, 0)))[:25]
    fig, ax = plt.subplots(figsize=(max(4, 0.45 * len(regions) + 2), 3.2))
    x = np.arange(len(regions))
    k = [kept.get(r, 0) for r in regions]
    d = [dropped.get(r, 0) for r in regions]
    ax.bar(x, k, width=0.7, color=SERIES[0], label="kept")
    ax.bar(x, d, width=0.7, bottom=np.array(k) + 0.0, color=GRID, edgecolor="white", linewidth=1, label="dropped")
    for xi, ki in zip(x, k):
        ax.text(xi, ki, str(ki), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, regions, rotation=45 if len(regions) > 6 else 0, ha="right" if len(regions) > 6 else "center")
    noun = "units" if kind == "units" else "ROIs"
    ax.set_ylabel(noun)
    ax.set_title(f"{sid}: {noun} by brain region", loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    filt = "label (good), firing rate and region filters" if kind == "units" else "the accepted-ROI filter"
    cap = (f"{sid}: {noun} per brain region kept for analysis (blue, count on top) and dropped by {filt} (grey). "
           f"Check that the regions you care about have enough {noun}; 'unknown' means the file has no location.")
    return path, cap


def timeline(path: Path, sid: str, spans: list[tuple[str, float, float]], window: tuple[float, float]) -> tuple[Path, str]:
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(spans) + 1.2))
    for i, (name, a, b) in enumerate(spans):
        ax.plot([a, b], [i, i], color=SERIES[0] if i == 0 else INK2, linewidth=5, solid_capstyle="butt")
    ax.axvspan(window[0], window[1], color=GRID, alpha=0.7, zorder=0, linewidth=0)
    ax.set_yticks(range(len(spans)), [s[0] for s in spans], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("session clock (s)")
    ax.set_title(f"{sid}: time ranges of neural data and behaviour", loc="left")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    cap = (f"{sid}: when each signal has data, on the session clock; the shaded band is the aligned window "
           f"(where neural data and behaviour overlap). Bars that do not overlap mean the files are on "
           f"different clocks.")
    return path, cap
