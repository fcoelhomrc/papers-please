"""Standalone matplotlib figures for the retrieval ablation.

    uv run python -m eval.plots
    uv run python -m eval.plots --results eval/results/ablation-....json

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

# Figures are written per embedding model for the same reason results are
# stored that way: a number measured on one embedder is not comparable to one
# measured on another, and a shared output directory would let the second
# sweep silently overwrite the first's evidence.
EMBED_MODEL = "bge-small"


def results() -> Path:
    from eval.results_store import RESULTS_ROOT

    return RESULTS_ROOT / EMBED_MODEL

# Fixed slot order from the reference palette. Never cycled, never re-ordered:
# a sixth series takes slot 6, it does not wrap back to slot 1. Wrapping put
# `hyde` and `none` in the same blue on the query-arms figure, which is the
# one thing a categorical palette must never do.
# Both modes are selected, not flipped: the dark column is the same eight hues
# stepped for the dark surface, per the reference palette.
THEMES = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"],
        "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "baseline": "#c3c2b7", "surface": "#fcfcfb",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
                   "#d55181", "#008300", "#9085e9", "#e66767"],
        "ramp": ["#104281", "#184f95", "#256abf", "#2a78d6", "#5598e7", "#9ec5f4"],
        "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "baseline": "#383835", "surface": "#1a1a19",
    },
}

# Set by style() for the mode currently rendering.
SERIES: list[str] = []
RAMP: list[str] = []
INK = SECONDARY = MUTED = GRID = BASELINE = SURFACE = ""
THEME = "light"

# Display names. Keys are what the results JSON and config.yaml hold; these
# are what a reader sees. Spelled out, because "hybrid" and "hybrid_bm25" do
# not say what is being fused, and that is the whole distinction between them.
LABEL = {
    # retrieval modes
    "semantic": "Dense",
    "keyword": "TS-Rank",
    "bm25": "BM25",
    "hybrid": "Hybrid (Dense + TS-Rank)",
    "hybrid_bm25": "Hybrid (Dense + BM25)",
    # metrics
    "ndcg": "nDCG", "map": "MAP", "mrr": "MRR", "r_precision": "R-Precision",
    "recall": "Recall", "precision": "Precision", "hit_rate": "Hit Rate",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_recall": "Context Recall",
    "llm_context_precision_with_reference": "Context Precision",
    # query arms
    "none": "Original Query",
    "decompose": "Decomposition",
    "decompose+orig": "Decomposition + Original",
    "multi_query": "Multi-Query",
    "multi_query+orig": "Multi-Query + Original",
    "hyde": "HyDE",
    # latency stages
    "embed": "Embed", "pinecone": "Pinecone", "hydrate": "SQL Hydrate",
    "keyword_sql": "TS-Rank SQL", "fuse": "Fuse", "rerank": "Rerank",
    # axes
    "top_k": "Top-K", "keyword_weight": "Keyword Weight",
    "latency_ms": "Latency (ms)", "metric": "Metric",
}


def label(key) -> str:
    """Display name for a key, or a Title Cased fallback.

    The fallback exists so a new mode or metric shows up readable rather than
    as raw snake_case the moment it is added and before anyone names it.
    """
    return LABEL.get(key, str(key).replace("_", " ").title())


MODES = ["semantic", "keyword", "bm25", "hybrid", "hybrid_bm25"]


def mode_color() -> dict[str, str]:
    """Resolved per render: SERIES holds the current theme's steps."""
    return dict(zip(MODES, SERIES))

# Every label in every figure is the literal key from the ablation output or
# from config.yaml - `ndcg`, `top_k`, `keyword_weight`, `rerank_candidates`,
# `hybrid_bm25`. Prettified names ("nDCG@10", "no rerank") were invented here
# and matched nothing a reader could grep for in the results JSON or the
# config, which makes a figure impossible to trace back to the run.
#
# One geometry for all of them, so they read as one set: same panel height,
# same bar width and pitch, legend always above the axes.
PANEL_H = 3.3
W1, W2 = 7.4, 9.4        # single panel, two panels
BAR_W, BAR_PITCH = 0.115, 0.145
LEGEND = {"loc": "upper center", "bbox_to_anchor": (0.5, 1.14), "columnspacing": 1.4}


def style(mode: str):
    global SERIES, RAMP, INK, SECONDARY, MUTED, GRID, BASELINE, SURFACE, THEME
    import matplotlib as mpl

    t = THEMES[mode]
    THEME = mode
    SERIES, RAMP = t["series"], t["ramp"]
    INK, SECONDARY, MUTED = t["ink"], t["secondary"], t["muted"]
    GRID, BASELINE, SURFACE = t["grid"], t["baseline"], t["surface"]

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
        "legend.labelcolor": SECONDARY,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "figure.dpi": 200,
    })


def facet(ax, text: str):
    """Which slice of the data this panel holds - `top_k=10`, `mode=bm25`.

    Bare key=value inside the axes. It is the panel's identity, not a caption:
    without it a faceted figure is unreadable, and with anything more than the
    key and its value it becomes the title this figure is not allowed to have.
    """
    ax.text(0.02, 0.97, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=MUTED)


FORMATS = ("png", "svg")


def save(fig, name: str):
    """One directory per format, so a README can glob `assets/eval/png/*` and
    a print or edit workflow can take the vector copies without filtering."""
    for ext in FORMATS:
        out = ASSETS / ext / THEME / EMBED_MODEL
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.15)
    import matplotlib.pyplot as plt

    plt.close(fig)


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

    fig, axes = plt.subplots(1, 2, figsize=(W2, PANEL_H), sharex=True)
    # No confidence bands here. Five of them at 10% alpha stack into a wash
    # that hides the curves they belong to; the intervals are shown properly
    # in the ranking-quality figure, one depth at a time.
    for ax, metric in zip(axes, ("recall", "precision")):
        for mode, (ks, ys, _) in by_mode(data["a"], metric).items():
            ax.plot(ks, ys, color=mode_color()[mode], label=label(mode), zorder=3)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 5, 10, 20, 50])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel(label("top_k"))
        ax.set_ylabel(label(metric))
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), **LEGEND)
    fig.tight_layout()
    save(fig, "retrieval-depth")


def fig_ranking_quality(data, k=10):
    """Ranking quality at one depth, grouped by metric."""
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["ndcg", "map", "mrr", "r_precision"]
    rows = {r["config"]["mode"]: r for r in data["a"] if r["config"]["top_k"] == k}

    fig, ax = plt.subplots(figsize=(W1, PANEL_H))
    x = np.arange(len(metrics))
    # Capped rather than filling the slot: five bars across a unit slot leave
    # the remainder as air, and the pitch is wider than the bar so neighbours
    # are separated by surface rather than by a stroke.
    width, pitch = BAR_W, BAR_PITCH
    for i, mode in enumerate(MODES):
        if mode not in rows:
            continue
        s = rows[mode]["summary"]
        off = (i - (len(MODES) - 1) / 2) * pitch
        ax.bar(x + off, [s[m] for m in metrics], width,
               yerr=[s[f"{m}_ci"] for m in metrics],
               color=mode_color()[mode], label=label(mode), zorder=3,
               error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 0})
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in metrics])
    ax.set_xlabel(label("metric"))
    facet(ax, f"Top-K = {k}")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=len(MODES), **LEGEND)
    fig.tight_layout()
    save(fig, "ranking-quality")


def fig_fusion_weight(data):
    """nDCG against the keyword side's RRF weight, faceted by depth."""
    import matplotlib.pyplot as plt

    ks = sorted({r["config"]["top_k"] for r in data["w"]})
    fig, axes = plt.subplots(1, len(ks), figsize=(W2, PANEL_H), sharey=True)
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
                    color=mode_color()[mode], label=label(mode), marker="o",
                    markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.axvline(0.1, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        # Facet identity rides in the axis label rather than a panel title.
        ax.set_xlabel(label("keyword_weight"))
        facet(ax, f"Top-K = {k}")
        ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
        # Margin to the left of 0.1 so the facet marker does not sit on the
        # line marking the shipped value.
        ax.set_xlim(0.02, 1.08)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(label("ndcg"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), **LEGEND)
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

    fig, ax = plt.subplots(figsize=(W1, PANEL_H))
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
        ax.annotate(label(mode), (latency[mode], quality[mode]),
                    textcoords="offset points",
                    xytext=(-9 if right else 9, -3),
                    ha="right" if right else "left",
                    fontsize=8, color=SECONDARY)
    ax.set_xlabel(label("latency_ms"))
    ax.set_ylabel(label("ndcg"))
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

    fig, ax = plt.subplots(figsize=(W1, PANEL_H))
    left = np.zeros(len(modes))
    # Sequential ramp: these are parts of one magnitude, not five identities.
    ramp = RAMP
    for stage, color in zip(stages, ramp):
        vals = np.array([rows[m].get(stage, 0.0) for m in modes])
        if vals.sum() == 0:
            continue
        ax.barh([label(m) for m in modes], vals, left=left, height=0.42,
                color=color, label=label(stage), edgecolor=SURFACE,
                linewidth=1.5, zorder=3)
        left += vals
    ax.set_xlabel(label("latency_ms"))
    ax.invert_yaxis()
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=len(stages), **LEGEND)
    fig.tight_layout()
    save(fig, "latency-breakdown")


def fig_rerank(data):
    """Reranked against plain at matched output size, as bars from zero.

    Bars rather than the dot-and-interval plot this was: on a zoomed y-axis
    the intervals looked enormous next to every other figure, when they are
    the ordinary +/-0.05 to +/-0.10 that 100 questions buys. From a zero
    baseline they read at their real size, and the figure matches the rest.

    The intervals are genuinely wide at Top-K = 1, where nDCG is 0 or 1 per
    question and nothing averages out. That is the measurement, not a
    plotting artefact.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plain = {r["config"]["top_k"]: r["summary"]
             for r in data.get("a", []) if r["config"]["mode"] == "hybrid"}
    ks = sorted({r["config"]["top_k"] for r in data["b"]} & set(plain))

    best = []
    for k in ks:
        at_k = [r for r in data["b"] if r["config"]["top_k"] == k]
        best.append(max(at_k, key=lambda r: r["summary"]["ndcg"])["summary"])

    fig, ax = plt.subplots(figsize=(W1, PANEL_H))
    x = np.arange(len(ks))
    for i, (series, name) in enumerate((
        ([plain[k] for k in ks], "Rerank: Off"),
        (best, "Rerank: On"),
    )):
        off = (i - 0.5) * BAR_PITCH
        ax.bar(x + off, [s["ndcg"] for s in series], BAR_W,
               yerr=[s["ndcg_ci"] for s in series], color=SERIES[i],
               label=name, zorder=3,
               error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 0})
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel(label("top_k"))
    ax.set_ylabel(label("ndcg"))
    ax.set_ylim(0, 1)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, **LEGEND)
    fig.tight_layout()
    save(fig, "rerank-matched")


def fig_query_arms(data):
    """Every arm against `none`, faceted by mode.

    Faceted rather than filtered to one mode, because `hyde` only runs on
    semantic - it embeds a passage, so it means nothing to a lexical
    retriever. Showing a single mode dropped it from the figure entirely
    without saying so, which is the worst way for an arm to be absent.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    modes = sorted({r["config"]["mode"] for r in data["c"]})
    ks = sorted({r["config"]["top_k"] for r in data["c"]})
    # Ordered by how much each transform rewrites the question, which is the
    # order the scores come out in. `none` first as the baseline.
    order = ["none", "decompose+orig", "decompose",
             "multi_query+orig", "multi_query", "hyde"]
    arms = [a for a in order if any(r["config"]["arm"] == a for r in data["c"])]
    color = dict(zip(arms, SERIES))

    fig, axes = plt.subplots(1, len(modes), figsize=(W2, PANEL_H), sharey=True)
    for ax, mode in zip(np.atleast_1d(axes), modes):
        rows = [r for r in data["c"] if r["config"]["mode"] == mode]
        present = [a for a in arms if any(r["config"]["arm"] == a for r in rows)]
        x = np.arange(len(ks))
        for i, arm in enumerate(present):
            vals, errs = [], []
            for k in ks:
                m = [r for r in rows
                     if r["config"]["arm"] == arm and r["config"]["top_k"] == k]
                vals.append(m[0]["summary"]["ndcg"] if m else np.nan)
                errs.append(m[0]["summary"]["ndcg_ci"] if m else 0)
            off = (i - (len(present) - 1) / 2) * BAR_PITCH
            ax.bar(x + off, vals, BAR_W, yerr=errs, color=color[arm],
                   label=label(arm), zorder=3,
                   error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 0})
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel(label("top_k"))
        facet(ax, f"Retriever = {label(mode)}")
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].set_ylabel(label("ndcg"))

    # One legend for both panels, listing every arm - including the one that
    # only appears in a single facet.
    handles = [plt.Rectangle((0, 0), 1, 1, color=color[a]) for a in arms]
    fig.legend(handles, [label(a) for a in arms], ncol=3, **LEGEND)
    fig.tight_layout()
    save(fig, "query-arms")


def load_results(explicit: Path | None = None) -> dict:
    """The newest full run, with any newer single-ablation re-run merged over it.

    Re-running one ablation writes a file carrying only that key, so globbing
    for the newest file alone hands the figures a result with no `a` in it.
    Merging keeps a targeted re-run from invalidating every other figure.
    """
    if explicit:
        return json.loads(explicit.read_text())

    # A target can have judged runs and no ablation at all: BM25 is swept as a
    # retrieval *mode*, so it has no encoder to ablate and no ablation-all-*
    # file, but it does have every query arm judged. Returning empty here lets
    # main() skip the ablation figures by its usual route and still draw the
    # judged ones, which is the documented behaviour for a partial result.
    ablations = sorted(results().glob("ablation-all-*.json"),
                       key=lambda p: p.stat().st_mtime)
    if not ablations:
        print("  no ablation on disk - judged figures only")
        return {}

    full = ablations[-1]
    data = json.loads(full.read_text())
    print(f"base: {full.name}")
    for part in results().glob("ablation-*.json"):
        if part == full or part.stat().st_mtime <= full.stat().st_mtime:
            continue
        blob = json.loads(part.read_text())
        for key in ("a", "b", "c", "w", "pool", "timings"):
            if key in blob:
                data[key] = blob[key]
                print(f"  merged {key!r} from {part.name}")
    return data


def judged_results() -> dict | None:
    """The baseline run the single-run figures describe.

    The `none` arm, not the newest file. These two figures carry no arm label,
    and when only `none` existed "newest" and "the baseline" were the same
    file. They stopped being the same the moment the arms were swept: the last
    run to land is whichever arm the sweep happened to end on, so
    judged-metrics and judge-vs-labels would have started reporting HyDE while
    still reading as the headline number.

    Falls back to newest so a results directory holding only arm runs still
    renders something rather than nothing.
    """
    by_arm = judged_by_arm()
    if "none" in by_arm:
        return by_arm["none"]
    newest = sorted(results().glob("judged-*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(newest[-1].read_text()) if newest else None


def judged_by_arm() -> dict[str, dict]:
    """The newest judged run per arm, keyed by arm.

    Newest wins because an arm gets re-run when something about the harness is
    fixed, and an older run of the same arm measured a different pipeline.
    Only full runs count - a --sample smoke has a different n and would sit in
    the same figure as though it were comparable.
    """
    out: dict[str, tuple[float, dict]] = {}
    for path in results().glob("judged-*.json"):
        blob = json.loads(path.read_text())
        arm = blob.get("retrieval_config", {}).get("arm")
        if not arm or blob.get("n_questions", 0) < 100:
            continue
        stamp = path.stat().st_mtime
        if arm not in out or stamp > out[arm][0]:
            out[arm] = (stamp, blob)
    return {arm: blob for arm, (_, blob) in out.items()}


ARM_ORDER = ["none", "decompose+orig", "decompose",
             "multi_query+orig", "multi_query", "hyde"]


def retriever_facet(runs: dict[str, dict]) -> str:
    """The facet marker naming the retriever these arms were measured on.

    Read off the runs rather than asserted. Hardcoding "Dense" was correct
    while only encoders were judged, and became a false label the moment BM25
    was swept through the same arms - the lexical figure announced itself as
    the dense one. Mixed modes return "" rather than picking one: an
    unlabelled facet is recoverable, a confidently wrong one is not.
    """
    modes = {r.get("retrieval_config", {}).get("mode") for r in runs.values()}
    if len(modes) != 1:
        return ""
    mode = modes.pop()
    return f"Retriever = {label(mode)}" if mode else ""


def fig_judged_arms(runs: dict[str, dict]):
    """The judged metrics per query arm.

    The free branch measured every arm on chunk-id retrieval and found all of
    them behind the untransformed question. This asks the different question:
    whether that gap reaches the answer a reader would actually see.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    arms = [a for a in ARM_ORDER if a in runs]
    fig, ax = plt.subplots(figsize=(W2, PANEL_H))
    x = np.arange(len(JUDGED_METRICS))
    for i, arm in enumerate(arms):
        rows = runs[arm]["per_question"]
        means, errs = [], []
        for m in JUDGED_METRICS:
            mean, half = _mean_ci([r.get(m) for r in rows])
            means.append(mean)
            errs.append(half)
        ax.bar(x + (i - (len(arms) - 1) / 2) * BAR_PITCH, means, BAR_W,
               yerr=errs, color=SERIES[i], label=label(arm), zorder=3,
               error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 0})
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in JUDGED_METRICS])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    facet(ax, retriever_facet(runs))
    ax.legend(ncol=3, **LEGEND)
    fig.tight_layout()
    save(fig, "judged-query-arms")


def _mean_ci(values: list[float]) -> tuple[float, float]:
    import math

    vals = [v for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return (vals[0] if vals else 0.0), 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, 1.96 * math.sqrt(var) / math.sqrt(len(vals))


JUDGED_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
]


def fig_judged(judged):
    """The four LLM-judged metrics, with intervals from the per-question rows.

    Intervals are computed here rather than read off the run, because the
    judged output stores means; the spread across questions is what says
    whether two of these are distinguishable.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    rows = judged["per_question"]
    fig, ax = plt.subplots(figsize=(W1, PANEL_H))
    x = np.arange(len(JUDGED_METRICS))
    means, errs = [], []
    for m in JUDGED_METRICS:
        mean, half = _mean_ci([r.get(m) for r in rows])
        means.append(mean)
        errs.append(half)
    ax.bar(x, means, BAR_W * 2.6, yerr=errs, color=SERIES[0], zorder=3,
           error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 0})
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in JUDGED_METRICS])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "judged-metrics")


def fig_judge_vs_labels(judged):
    """The judge's view of retrieval against the chunk-id labels'.

    The reason the judged branch exists. Chunk-id labels score a correct
    retrieval of a chunk the question was not seeded from as a miss, so they
    are a lower bound; this is how much lower.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    free = judged["retrieval_metrics"]
    rows = judged["per_question"]
    pairs = [
        ("Recall", free["recall"], free["recall_ci"], "context_recall"),
        ("Precision", free["map"], free["map_ci"],
         "llm_context_precision_with_reference"),
    ]

    fig, ax = plt.subplots(figsize=(W1, PANEL_H))
    x = np.arange(len(pairs))
    for i, (source, name) in enumerate((("free", "Chunk-ID Labels"),
                                        ("judge", "LLM Judge"))):
        vals, errs = [], []
        for _, fv, fc, jkey in pairs:
            if source == "free":
                vals.append(fv)
                errs.append(fc)
            else:
                mean, half = _mean_ci([r.get(jkey) for r in rows])
                vals.append(mean)
                errs.append(half)
        ax.bar(x + (i - 0.5) * BAR_PITCH, vals, BAR_W, yerr=errs,
               color=SERIES[i], label=name, zorder=3,
               error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 0})
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pairs])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, **LEGEND)
    fig.tight_layout()
    save(fig, "judge-vs-labels")


def main():
    parser = argparse.ArgumentParser(description="Retrieval ablation figures")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--embed-model", default=None,
                        help="which embedder's results to plot (default: configured)")
    args = parser.parse_args()

    global EMBED_MODEL
    if args.embed_model:
        EMBED_MODEL = args.embed_model
    else:
        from config import load

        EMBED_MODEL = load().embedder.model
    print(f"embedder: {EMBED_MODEL}")

    data = load_results(args.results)

    # Each figure is skipped rather than fatal when its ablation is absent, so
    # a partial result still renders whatever it does contain.
    figures = [
        ("a", fig_recall_precision),
        ("a", fig_ranking_quality),
        ("w", fig_fusion_weight),
        ("b", fig_rerank),
        ("timings", fig_latency_quality),
        ("timings", fig_latency_breakdown),
        ("c", fig_query_arms),
    ]
    judged = judged_results()
    by_arm = judged_by_arm()
    for mode in THEMES:
        style(mode)
        for key, fn in figures:
            if key not in data:
                continue
            fn(data)
        if judged:
            fig_judged(judged)
            fig_judge_vs_labels(judged)
        if len(by_arm) > 1:
            fig_judged_arms(by_arm)
        print(f"  {mode}: {sum(1 for k, _ in figures if k in data)} figures")
    missing = sorted({k for k, _ in figures if k not in data})
    if missing:
        print(f"  (skipped, absent from results: {', '.join(missing)})")
    print(f"-> {ASSETS}/<ext>/<mode>/")


if __name__ == "__main__":
    main()
