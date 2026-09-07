"""Standalone matplotlib figures for the retrieval ablation.

    uv run --with matplotlib python -m eval.plots
    uv run --with matplotlib python -m eval.plots --results eval/results/ablation-....json

Writes PNG + SVG to assets/eval/. No titles, no annotations, no analysis: a
figure shows the data and the prose around it does the arguing. Anything a
caption would say does not belong inside the axes.

Palette is the dataviz reference instance, used in its documented fixed slot
order. That order is validated for *adjacent* pairs in both modes, which is
the pairlist lines and bars use; only the first three slots are validated
all-pairs, so the scatter uses a single series with direct labels rather than
five colours. No JS runtime was available to re-run the validator here, so
nothing is re-picked by eye - the published order is used as published.
"""
import argparse
import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "eval"
RESULTS = Path(__file__).parent / "results"

# Fixed slot order from the reference palette. Never cycled, never re-ordered.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

MODES = ["semantic", "keyword", "bm25", "hybrid", "hybrid_bm25"]
MODE_COLOR = dict(zip(MODES, SERIES))


def style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": SECONDARY,
        "axes.labelsize": 9,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        # Only the left and bottom rules survive; a box around data is ink
        # that is not data.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "figure.dpi": 200,
    })


FORMATS = ("png", "svg")


def save(fig, name: str):
    """One directory per format, so a README can glob `assets/eval/png/*` and
    a print or edit workflow can take the vector copies without filtering."""
    for ext in FORMATS:
        out = ASSETS / ext
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.15)
    print(f"  {name}." + "/.".join(FORMATS))


def by_mode(rows, metric):
    """{mode: ([k...], [value...], [ci...])} from ablation A."""
    out = {}
    for mode in MODES:
        pts = sorted(
            (r for r in rows if r["config"]["mode"] == mode),
            key=lambda r: r["config"]["top_k"],
        )
        if pts:
            out[mode] = (
                [p["config"]["top_k"] for p in pts],
                [p["summary"][metric] for p in pts],
                [p["summary"][f"{metric}_ci"] for p in pts],
            )
    return out


def fig_recall_precision(data):
    """Recall and precision against depth, one panel each, shared x.

    Two panels rather than two y-axes: recall rises with k and precision falls,
    and putting them on one pair of axes would make the crossing point look
    like a fact about retrieval instead of a fact about the scales chosen.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    # No confidence bands here. Five of them at 10% alpha stack into a wash
    # that hides the curves they belong to; the intervals are shown properly
    # in the ranking-quality figure, one depth at a time.
    for ax, metric, label in zip(axes, ("recall", "precision"), ("recall@k", "precision@k")):
        for mode, (ks, ys, _) in by_mode(data["a"], metric).items():
            ax.plot(ks, ys, color=MODE_COLOR[mode], label=mode, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 5, 10, 20, 50])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), columnspacing=1.6)
    fig.tight_layout()
    save(fig, "retrieval-depth")


def fig_ranking_quality(data, k=10):
    """Ranking quality at one depth, grouped by metric."""
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["ndcg", "map", "mrr", "r_precision"]
    labels = ["nDCG@10", "MAP@10", "MRR", "R-precision"]
    rows = {r["config"]["mode"]: r for r in data["a"] if r["config"]["top_k"] == k}

    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(metrics))
    # Capped rather than filling the slot: five bars across a unit slot leave
    # the remainder as air, and the pitch is wider than the bar so neighbours
    # are separated by surface rather than by a stroke.
    width, pitch = 0.115, 0.145
    for i, mode in enumerate(MODES):
        if mode not in rows:
            continue
        s = rows[mode]["summary"]
        off = (i - (len(MODES) - 1) / 2) * pitch
        ax.bar(x + off, [s[m] for m in metrics], width,
               yerr=[s[f"{m}_ci"] for m in metrics],
               color=MODE_COLOR[mode], label=mode, zorder=3,
               error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 0})
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    save(fig, "ranking-quality")


def fig_fusion_weight(data):
    """nDCG against the keyword side's RRF weight, faceted by depth."""
    import matplotlib.pyplot as plt

    ks = sorted({r["config"]["top_k"] for r in data["w"]})
    fig, axes = plt.subplots(1, len(ks), figsize=(9, 3.0), sharey=True)
    for ax, k in zip(axes, ks):
        for i, mode in enumerate(("hybrid", "hybrid_bm25")):
            pts = sorted(
                (r for r in data["w"]
                 if r["config"]["mode"] == mode and r["config"]["top_k"] == k
                 and r["config"]["rrf_k"] == 60),
                key=lambda r: r["config"]["keyword_weight"],
            )
            ax.plot([p["config"]["keyword_weight"] for p in pts],
                    [p["summary"]["ndcg"] for p in pts],
                    color=MODE_COLOR[mode], label=mode, marker="o",
                    markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.axvline(0.1, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xlabel("keyword weight")
        ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelrotation=0)
        ax.set_title(f"k={k}", fontsize=8, color=MUTED, pad=6)
    axes[0].set_ylabel("nDCG")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fusion-weight")


def fig_latency_quality(data):
    """Quality against latency. One series, points labelled - the palette's
    first three slots are the only ones validated all-pairs, and a scatter
    needs all-pairs separation."""
    import matplotlib.pyplot as plt

    quality = {r["config"]["mode"]: r["summary"]["ndcg"]
               for r in data["a"] if r["config"]["top_k"] == 10}
    latency = {t["config"]["mode"]: t["latency_ms"]["total"]
               for t in data["timings"]
               if t["config"]["top_k"] == 10 and not t["config"]["rerank"]}

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    span = max(latency.values())
    for mode in MODES:
        if mode not in quality or mode not in latency:
            continue
        ax.scatter(latency[mode], quality[mode], s=70, color=SERIES[0],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        # A label that would run past the right edge flips to the left of its
        # dot. Measuring first rather than clipping: a cropped label is worse
        # than no label.
        right = latency[mode] > 0.72 * span
        ax.annotate(mode, (latency[mode], quality[mode]),
                    textcoords="offset points",
                    xytext=(-9 if right else 9, -3),
                    ha="right" if right else "left",
                    fontsize=8, color=SECONDARY)
    ax.set_xlabel("mean latency per query (ms)")
    ax.set_ylabel("nDCG@10")
    ax.set_xlim(0, span * 1.08)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "latency-quality")


def fig_latency_breakdown(data):
    """Where the time goes, per mode, at one depth."""
    import matplotlib.pyplot as plt
    import numpy as np

    stages = ["embed", "pinecone", "hydrate", "keyword_sql", "bm25", "fuse"]
    rows = {t["config"]["mode"]: t["latency_ms"] for t in data["timings"]
            if t["config"]["top_k"] == 10 and not t["config"]["rerank"]}
    modes = [m for m in MODES if m in rows]

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    left = np.zeros(len(modes))
    # Sequential ramp: these are parts of one magnitude, not five identities.
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
    for stage, color in zip(stages, ramp):
        vals = np.array([rows[m].get(stage, 0.0) for m in modes])
        if vals.sum() == 0:
            continue
        ax.barh(modes, vals, left=left, height=0.42, color=color,
                label=stage, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        left += vals
    ax.set_xlabel("mean latency per query (ms)")
    ax.invert_yaxis()
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    save(fig, "latency-breakdown")


def fig_rerank(data):
    """Reranked against plain, at matched output size."""
    import matplotlib.pyplot as plt
    import numpy as np

    plain = {r["config"]["top_k"]: r["summary"]
             for r in data["a"] if r["config"]["mode"] == "hybrid"}
    ks = sorted({r["config"]["top_k"] for r in data["b"]})
    best, err = [], []
    for k in ks:
        at_k = [r for r in data["b"] if r["config"]["top_k"] == k]
        top = max(at_k, key=lambda r: r["summary"]["ndcg"])
        best.append(top["summary"]["ndcg"])
        err.append(top["summary"]["ndcg_ci"])

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(len(ks))
    ax.errorbar(x - 0.08, [plain[k]["ndcg"] for k in ks],
                yerr=[plain[k]["ndcg_ci"] for k in ks], fmt="o", markersize=7,
                color=SERIES[0], label="no rerank", elinewidth=1,
                capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.errorbar(x + 0.08, best, yerr=err, fmt="o", markersize=7,
                color=SERIES[1], label="reranked", elinewidth=1,
                capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for i, k in enumerate(ks):
        ax.plot([i - 0.08, i + 0.08], [plain[k]["ndcg"], best[i]],
                color=BASELINE, linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("nDCG")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save(fig, "rerank-matched")


def main():
    parser = argparse.ArgumentParser(description="Retrieval ablation figures")
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    path = args.results or max(RESULTS.glob("ablation-*.json"), key=lambda p: p.stat().st_mtime)
    data = json.loads(path.read_text())
    print(f"{path.name} -> {ASSETS}")

    style()
    fig_recall_precision(data)
    fig_ranking_quality(data)
    fig_fusion_weight(data)
    fig_rerank(data)
    fig_latency_quality(data)
    fig_latency_breakdown(data)


if __name__ == "__main__":
    main()
