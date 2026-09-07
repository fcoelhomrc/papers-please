"""The eval figures committed under assets/eval/ and linked from the README.

    uv run --with matplotlib python -m eval.figures
    uv run --with matplotlib python -m eval.figures --only query-arms judged-metrics

Reads only cached results - the JSON under eval/results/ and the human labels
under eval/labelling/. Nothing here issues a search, an embedding or an LLM
call, so regenerating every figure is seconds and costs nothing.

One module, one identity
-----------------------
This replaces two generators that had drifted into two visual languages: an
SVG-by-hand `figures.py` (indigo cards, its own type scale) reading sweeps
that no longer exist, and a matplotlib `plots.py` (dataviz palette, no titles,
light only) reading the ablations that do. Figures from the two sat beside
each other in the README looking like they came from different projects.

Coherence here is structural rather than a matter of taste, and it is worth
naming what actually does the work:

* **Every figure is the same width with the same margins.** `tight_layout`
  and `bbox_inches="tight"` are what broke this before - they crop to the
  content, so each figure ended up with its own margins and the set refused
  to stack. `frame()` reserves fixed bands instead and nothing crops.
* **Every figure carries the same furniture**: title, one-line subtitle
  stating the finding, and a footer naming the run the numbers came from.
* **One palette**, the dataviz reference instance in its documented slot
  order, with a selected dark step per slot rather than an automatic flip.
* **One vocabulary.** `LABEL` maps every config key to the prose the plan
  uses, so a legend reads "hybrid (dense + BM25)" and never `hybrid_bm25`.

Everything the figures describe is the current evaluation - the 100-question
curated Ragas set over the 5x20 arXiv corpus, per `docs/rag-evaluation.md`.
The four figures that described the old 12-document, hand-labelled eval are
gone rather than regenerated: their inputs (`sweep-*.json`,
`thresholds-*.json`) no longer exist, and the retriever they measured does
not either.

Form notes, where the choice was not obvious
--------------------------------------------
Dot plots carry most of the comparisons rather than grouped bars. Five
retrieval modes across four metrics is twenty bars and twenty error bars;
as rows the mode is named by its own axis tick, which makes identity
independent of colour and sidesteps the light-mode contrast relief rule
entirely. Bars are kept only where the quantity is a magnitude read against
zero (the latency breakdown).
"""
import argparse
import json

from pathlib import Path

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "eval"
RESULTS = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Palette. The dataviz reference instance, both modes selected (the dark
# column is the same eight hues stepped for the dark surface, not a flip).
# Slot order is the CVD-safety mechanism - never cycled, never re-picked.
# ---------------------------------------------------------------------------
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
        # Ordinal ramp: the step nearest the surface must still clear 2:1, so
        # this starts at sequential step 250 rather than at 100.
        "ramp": ["#86b6ef", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "recede": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
        # Mirrored for the dark surface: no darker than step 600 at the end
        # nearest the surface, so the ramp runs light-ward instead.
        "ramp": ["#184f95", "#256abf", "#3987e5", "#5598e7", "#86b6ef", "#b7d3f6"],
        "recede": "#383835",
    },
}

# ---------------------------------------------------------------------------
# Vocabulary. Config keys are identifiers; figures are read by people. Every
# label a reader sees is looked up here, in the wording docs/rag-evaluation.md
# uses, so no legend ever shows a raw key.
# ---------------------------------------------------------------------------
MODES = ["semantic", "keyword", "bm25", "hybrid", "hybrid_bm25"]

LABEL = {
    # retrieval modes, per search.py's five
    "semantic": "dense",
    "keyword": "FTS · ts_rank",
    "bm25": "FTS · BM25",
    "hybrid": "hybrid (dense + ts_rank)",
    "hybrid_bm25": "hybrid (dense + BM25)",
    # query-transformation arms
    "none": "question as asked",
    "hyde": "HyDE",
    "multi_query": "multi-query",
    "multi_query+orig": "multi-query + original",
    "decompose": "decomposition",
    "decompose+orig": "decomposition + original",
    # latency stages
    "embed": "embed query",
    "pinecone": "Pinecone query",
    "hydrate": "hydrate from Postgres",
    "keyword_sql": "Postgres FTS",
    "bm25_rank": "BM25 ranking",
    "fuse": "RRF fuse",
    # ragas synthesizers
    "single_hop_specifc_query_synthesizer": "single-hop specific",
    "multi_hop_specific_query_synthesizer": "multi-hop specific",
    "multi_hop_abstract_query_synthesizer": "multi-hop abstract",
    # judged metrics
    "faithfulness": "faithfulness",
    "answer_relevancy": "response relevancy",
    "llm_context_precision_with_reference": "LLM context precision",
    "context_recall": "LLM context recall",
    # judge case types
    "supported": "supported",
    "paraphrase": "paraphrase",
    "true_but_absent": "true but absent",
    "corrupted_number": "corrupted number",
    "corrupted_entity": "corrupted entity",
    "overgeneralised": "overgeneralised",
    "unsupported_cause": "unsupported cause",
    "negation": "negation",
    "unrelated": "unrelated",
}

METRIC_LABEL = {
    "ndcg": "nDCG",
    "map": "MAP",
    "mrr": "MRR",
    "r_precision": "R-precision",
    "recall": "recall",
    "precision": "precision",
    "hit_rate": "hit rate",
}


def label(key):
    return LABEL.get(key, str(key).replace("_", " "))


# ---------------------------------------------------------------------------
# Layout. Fixed bands, identical on every figure - this is what makes the set
# read as one system, and it is why nothing here uses tight_layout.
# ---------------------------------------------------------------------------
FIG_W = 8.6          # inches; every figure, without exception
M_LEFT = 0.86      # default axes gutter; dot plots widen it for row labels
M_TEXT = 0.86      # where title, subtitle and footer always start
M_RIGHT = 0.30
BAND_TOP = 1.00      # title + subtitle + legend
BAND_BOT = 0.76      # x-axis label + provenance footer

FORMATS = ("png", "svg")


def rc(theme):
    import matplotlib as mpl

    t = THEMES[theme]
    mpl.rcParams.update({
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "text.color": t["ink"],
        "axes.labelcolor": t["secondary"],
        "axes.labelsize": 9,
        "axes.edgecolor": t["baseline"],
        "axes.linewidth": 0.8,
        # Only the left and bottom rules survive: a box around data is ink
        # that is not data.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.color": t["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",     # never dashed; a dashed grid reads as a threshold
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "figure.dpi": 200,
    })


class Frame:
    """One figure's fixed geometry, plus the furniture every figure carries."""

    # A subtitle longer than this runs off the fixed figure width and is
    # silently clipped at the right edge, which is how three of these shipped
    # the first time. Asserting turns that into a failure at generation.
    SUBTITLE_MAX = 104

    def __init__(self, theme, title, subtitle, plot_h, ncols=1,
                 m_left=M_LEFT, wspace=0.20, sharey=False, sharex=False,
                 width_ratios=None):
        import matplotlib.pyplot as plt

        assert len(subtitle) <= self.SUBTITLE_MAX, (
            f"subtitle is {len(subtitle)} chars, over {self.SUBTITLE_MAX}: {subtitle!r}"
        )
        self.t = THEMES[theme]
        self.h = plot_h + BAND_TOP + BAND_BOT
        self.fig, axes = plt.subplots(
            1, ncols, figsize=(FIG_W, self.h), sharey=sharey, sharex=sharex,
            gridspec_kw={"width_ratios": width_ratios} if width_ratios else None,
        )
        self.axes = [axes] if ncols == 1 else list(axes)
        self.fig.subplots_adjust(
            left=m_left / FIG_W, right=1 - M_RIGHT / FIG_W,
            top=1 - BAND_TOP / self.h, bottom=BAND_BOT / self.h, wspace=wspace,
        )
        self.m_left = m_left
        # Title, subtitle and footer hang off a constant text margin, NOT off
        # the axes margin. A dot plot needs a wide gutter for its row labels,
        # and letting the furniture follow that gutter both misaligned the
        # titles across the set and pushed the footer off the right edge.
        x = M_TEXT / FIG_W
        # DejaVu Sans has no semibold, so 600 falls back to 700 with a warning.
        self.fig.text(x, 1 - 0.30 / self.h, title, fontsize=12.5, fontweight="bold",
                      color=self.t["ink"], va="center")
        self.fig.text(x, 1 - 0.53 / self.h, subtitle, fontsize=9,
                      color=self.t["secondary"], va="center")

    @property
    def ax(self):
        return self.axes[0]

    def legend(self, handles=None, labels=None, ncol=None, ax=None):
        """Figure-level, always in the same band under the subtitle."""
        src = ax or self.axes[0]
        if handles is None:
            handles, labels = src.get_legend_handles_labels()
        if not handles:
            return
        self.fig.legend(handles, labels, ncol=ncol or len(handles),
                        loc="upper center", columnspacing=1.5, handletextpad=0.6,
                        bbox_to_anchor=(0.5, 1 - 0.62 / self.h),
                        labelcolor=self.t["secondary"])

    def footer(self, text):
        self.fig.text(M_TEXT / FIG_W, 0.20 / self.h, text, fontsize=7.5,
                      color=self.t["muted"], va="center")

    def grid(self, axis="y"):
        for ax in self.axes:
            ax.grid(axis=axis, zorder=0)
            ax.set_axisbelow(True)
        return self

    def rows(self, names, ax=None, texts=None):
        """Categorical y-axis for a dot plot, top-to-bottom in the given order.

        The tick marks and the left spine come off every panel, not only the
        labelled one - on a shared-y row of panels the unlabelled ones
        otherwise sprout a stub beside each row that reads as a stray mark.
        """
        a = ax or self.axes[0]
        a.set_yticks(range(len(names)))
        a.set_yticklabels(texts or [label(n) for n in names],
                          color=self.t["secondary"], fontsize=8.5)
        a.set_ylim(len(names) - 0.5, -0.5)
        for other in self.axes:
            other.tick_params(axis="y", length=0)
            other.spines["left"].set_visible(False)
        return a


def save(fig, name, theme):
    for ext in FORMATS:
        out = ASSETS / ext
        out.mkdir(parents=True, exist_ok=True)
        # No bbox_inches="tight": cropping to content is exactly what gives
        # every figure different margins and breaks the set.
        fig.savefig(out / f"{name}-{theme}.{ext}")
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------------------
# Loading. Everything is read from cache; nothing is recomputed.
# ---------------------------------------------------------------------------
def _stamp(path):
    """Sort key from the ISO stamp in the filename, not from mtime.

    Copying results between checkouts rewrites mtimes and silently reorders
    the runs; the name carries the truth.
    """
    return path.stem.split("-")[-1]


def latest(pattern):
    paths = sorted(RESULTS.glob(pattern), key=_stamp)
    return paths[-1] if paths else None


def load_ablation():
    """The newest full run, with any newer single-ablation re-run merged over it.

    Re-running one ablation writes a file carrying only that key, so taking
    the newest file alone hands the figures a result with no `a` in it.
    """
    full = latest("ablation-all-*.json")
    if not full:
        return None
    data = json.loads(full.read_text())
    data["_source"] = full.stem
    for part in sorted(RESULTS.glob("ablation-*.json"), key=_stamp):
        if part == full or _stamp(part) <= _stamp(full):
            continue
        blob = json.loads(part.read_text())
        for key in ("a", "b", "c", "w", "pool", "timings"):
            if key in blob:
                data[key] = blob[key]
                data[f"_source_{key}"] = part.stem
    return data


def load_json(pattern):
    path = latest(pattern)
    if not path:
        return None
    data = json.loads(path.read_text())
    data["_source"] = path.stem
    return data


def load_judge():
    """Cached judge verdicts scored against the committed human labels."""
    path = RESULTS / "judge-verdicts.json"
    if not path.is_file():
        return None
    from eval.judge_cases import load_labels, scoring_labels
    from eval.judge_kappa import by_case_type, score

    verdicts = json.loads(path.read_text())
    human = load_labels()
    if not human:
        return None
    return {
        "model": verdicts["model"],
        "primary": score(scoring_labels(human), verdicts["verdicts"]),
        "sensitivity": score(scoring_labels(human, drop_contested=True), verdicts["verdicts"]),
        "by_case": by_case_type(human, verdicts["verdicts"]),
    }


def provenance(data, extra=""):
    bits = [f"{data.get('n_questions', '?')} curated questions"]
    if data.get("embed_model"):
        bits.append(f"{data['embed_model']}, {data.get('chunk_max_tokens')}-token chunks")
    if extra:
        bits.append(extra)
    bits.append(data.get("_source", ""))
    return "  ·  ".join(b for b in bits if b)


# ---------------------------------------------------------------------------
# Free branch - retrieval, no LLM anywhere
# ---------------------------------------------------------------------------
def by_mode(rows, metric):
    out = {}
    for mode in MODES:
        pts = sorted((r for r in rows if r["config"]["mode"] == mode),
                     key=lambda r: r["config"]["top_k"])
        if pts:
            out[mode] = ([p["config"]["top_k"] for p in pts],
                         [p["summary"][metric] for p in pts],
                         [p["summary"][f"{metric}_ci"] for p in pts])
    return out




def fig_retrieval_depth(theme, data, free):
    """Recall and precision against depth, with precision's arithmetic ceiling.

    Two panels rather than two y-axes: recall rises with k and precision falls,
    and one pair of axes would make the crossing point look like a fact about
    retrieval instead of a fact about the scales chosen.

    The ceiling is the point. Most questions in this set have one or two
    relevant chunks, so precision@10 cannot exceed ~0.16 however perfect the
    retriever is - the falling curve is arithmetic, not a defect, and drawing
    it without the ceiling has misled every reader of the old figure.
    """
    import matplotlib.pyplot as plt

    f = Frame(theme, "Retrieval quality against how deep you read",
              "Recall climbs with k; precision falls because it must - the dashed "
              "line is the most it could score.",
              plot_h=2.55, ncols=2, sharex=True, sharey=True, wspace=0.20)
    t = f.t

    for ax, metric in zip(f.axes, ("recall", "precision")):
        for i, (mode, (ks, ys, _)) in enumerate(by_mode(data["a"], metric).items()):
            ax.plot(ks, ys, color=t["series"][i], zorder=3, label=label(mode))
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 5, 10, 20, 50])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("k (chunks returned)")
        ax.set_ylabel(f"{metric}@k")
        # Both panels on 0-1. Letting each autoscale made precision's collapse
        # look steeper than recall's climb when it is the shallower of the two,
        # and it cut the ceiling off at k=1 where the ceiling is 1.0 by
        # definition.
        ax.set_ylim(0, 1.0)

    # Precision's ceiling, from the relevant-chunk counts the free run already
    # records: mean(min(n_relevant, k) / k). Most questions have one or two
    # relevant chunks, so precision@10 cannot exceed ~0.16 however perfect the
    # retriever - the falling curve is arithmetic, not a defect.
    if free:
        n_rel = [q["n_relevant"] for q in free["per_question"] if q.get("n_relevant")]
        if n_rel:
            ks = [1, 3, 5, 10, 20, 50]
            ceiling = [sum(min(n, k) for n in n_rel) / (len(n_rel) * k) for k in ks]
            f.axes[1].plot(ks, ceiling, color=t["muted"], linewidth=1.1,
                           linestyle=(0, (4, 3)), zorder=2)
            f.axes[1].annotate("the most precision@k could be", (ks[1], ceiling[1]),
                               textcoords="offset points", xytext=(8, 6),
                               ha="left", fontsize=7.5, color=t["muted"])

    f.grid()
    f.legend(ncol=5, ax=f.axes[0])
    f.footer(provenance(data, "retrieval ablation A"))
    return f.fig


def fig_ranking_quality(theme, data, k=10):
    """Rank-aware quality at the shipped depth, as rows rather than bars.

    Five modes across four metrics is twenty bars with twenty whiskers. As
    rows the mode is named by its axis tick, so identity never rests on
    colour, and the confidence intervals stay readable at their real width.
    """
    f = Frame(theme, f"Rank-aware quality at k={k}",
              "Dots are means, whiskers 95% intervals. Overlapping intervals are "
              "a tie: at n=100, so is 5pp.",
              plot_h=2.5, ncols=4, m_left=1.62, wspace=0.16, sharey=True)
    t = f.t
    metrics = ["ndcg", "map", "mrr", "r_precision"]
    rows = {r["config"]["mode"]: r["summary"] for r in data["a"]
            if r["config"]["top_k"] == k}
    modes = [m for m in MODES if m in rows]

    for ax, metric in zip(f.axes, metrics):
        for i, mode in enumerate(modes):
            s = rows[mode]
            ax.errorbar(s[metric], i, xerr=s[f"{metric}_ci"], fmt="o", markersize=7,
                        color=t["series"][i], ecolor=t["muted"], elinewidth=1,
                        capsize=0, markeredgecolor=t["surface"], markeredgewidth=1.5,
                        zorder=3)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel(METRIC_LABEL[metric] + (f"@{k}" if metric in ("ndcg", "map") else ""))
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
    f.rows(modes)
    f.footer(provenance(data, "retrieval ablation A"))
    return f.fig


def fig_question_difficulty(theme, free):
    """Where the difficulty actually lives: the synthesizer, not the topic.

    This is the figure the corpus design predicts and nothing plotted before.
    Ragas' three synthesizers ask structurally different questions, and the
    spread between them dwarfs the spread between the five topics.

    One axes rather than two side by side: the two groupings share the x scale
    and the whole point is comparing the spread within each, which a pair of
    panels with different row counts actively hinders - the rows stop lining
    up and the long synthesizer names get squeezed into whatever margin is
    left over.
    """
    synths = sorted(free["by_synthesizer"],
                    key=lambda s: -free["by_synthesizer"][s]["recall"])
    topics = sorted(free["by_topic"], key=lambda s: -free["by_topic"][s]["recall"])
    f = Frame(theme, "What makes a question hard",
              "Multi-hop questions cost about half the recall of single-hop ones. "
              "The topic barely matters.",
              plot_h=3.05, m_left=2.15)
    t = f.t
    metrics = [("recall", 0), ("ndcg", 1), ("map", 2)]

    # One blank row between the groups, so the separator has somewhere to sit.
    GAP = len(synths)
    entries = ([(r, free["by_synthesizer"], k) for r, k in enumerate(synths)]
               + [(GAP + 1 + r, free["by_topic"], k) for r, k in enumerate(topics)])

    ax = f.ax
    for row, blob, key in entries:
        for metric, slot in metrics:
            ax.plot(blob[key][metric], row, "o", markersize=6.5,
                    color=t["series"][slot], markeredgecolor=t["surface"],
                    markeredgewidth=1.4, zorder=3,
                    label=METRIC_LABEL[metric] if row == 0 else None)
    ax.axhline(GAP, color=t["grid"], linewidth=1, zorder=1)
    for row, caption in ((0, "by Ragas synthesizer"), (GAP + 1, "by corpus topic")):
        ax.annotate(caption, (0.02, row - 0.46), fontsize=7.5, style="italic",
                    color=t["muted"], va="center")

    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("score")
    labels, names = [], []
    for row, blob, key in entries:
        names.append(key)
        labels.append(f"{label(key)}  (n={blob[key]['n']})")
    # The blank separator row needs a placeholder in both lists.
    names.insert(GAP, "")
    labels.insert(GAP, "")
    f.rows(names, texts=labels)
    f.grid(axis="x")
    f.legend(ncol=3)
    f.footer(provenance(free, f"free branch · {label(free['config']['mode'])} "
                              f"k={free['config']['top_k']}"))
    return f.fig


def fig_query_arms(theme, data):
    """Every query transformation against leaving the question alone.

    The finding is a null one, so the baseline is drawn as the reference and
    everything else is measured against it: emphasis on the one series that
    matters, the rest receding. HyDE only ever ran on the dense arm - the
    empty row says so rather than quietly dropping it, which is what the
    previous version of this figure did to it.
    """
    ARMS = ["none", "decompose+orig", "decompose", "multi_query+orig",
            "hyde", "multi_query"]
    k = 10
    f = Frame(theme, f"Query transformations, at k={k}",
              "Every arm costs nDCG against the untransformed question, in both "
              "retrievers and at every depth measured.",
              plot_h=2.05, ncols=2, m_left=1.72, wspace=0.14, sharey=True)
    t = f.t

    for ax, mode in zip(f.axes, ("semantic", "bm25")):
        rows = {r["config"]["arm"]: r["summary"] for r in data["c"]
                if r["config"]["mode"] == mode and r["config"]["top_k"] == k}
        base = rows.get("none", {}).get("ndcg")
        if base is not None:
            ax.axvline(base, color=t["muted"], linewidth=0.9,
                       linestyle=(0, (4, 3)), zorder=1)
        for row, arm in enumerate(ARMS):
            if arm not in rows:
                ax.text(0.03, row, "not run", fontsize=7.5, style="italic",
                        color=t["muted"], va="center")
                continue
            s = rows[arm]
            # Emphasis: the baseline carries the colour, the arms recede.
            colour = t["series"][0] if arm == "none" else t["recede"]
            ax.errorbar(s["ndcg"], row, xerr=s["ndcg_ci"], fmt="o", markersize=7,
                        color=colour, ecolor=t["muted"], elinewidth=1, capsize=0,
                        markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3)
            if arm != "none" and base:
                # A right-aligned value column rather than a label above the
                # dot: above, each delta sat closer to the row above it than
                # to its own, and the top one collided with the baseline rule.
                ax.annotate(f"{s['ndcg'] - base:+.3f}", (1.1, row), ha="right",
                            va="center", fontsize=7.5, color=t["muted"])
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel(f"nDCG@{k}   ·   {label(mode)}")
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
    f.rows(ARMS)
    src = data.get("_source_c", data.get("_source", ""))
    f.footer(f"{data.get('n_questions', '?')} curated questions  ·  "
             f"dashed line: the question as asked  ·  {src}")
    return f.fig


def fig_fusion_weight(theme, data):
    """nDCG against the keyword side's RRF weight, faceted by depth."""
    ks = sorted({r["config"]["top_k"] for r in data["w"]})
    f = Frame(theme, "How much the keyword side should count",
              "The shipped default was 0.1, fitted on the old corpus. On this one "
              "the curve is monotonic the other way.",
              plot_h=2.35, ncols=len(ks), wspace=0.16, sharey=True)
    t = f.t

    for ax, k in zip(f.axes, ks):
        for i, mode in enumerate(("hybrid", "hybrid_bm25")):
            pts = sorted((r for r in data["w"]
                          if r["config"]["mode"] == mode
                          and r["config"]["top_k"] == k
                          and r["config"]["rrf_k"] == 60),
                         key=lambda r: r["config"]["keyword_weight"])
            if not pts:
                continue
            ax.plot([p["config"]["keyword_weight"] for p in pts],
                    [p["summary"]["ndcg"] for p in pts],
                    color=t["series"][i], marker="o", markersize=4.5,
                    markeredgecolor=t["surface"], markeredgewidth=1.4,
                    label=label(mode), zorder=3)
        ax.axvline(0.1, color=t["muted"], linewidth=0.9,
                   linestyle=(0, (4, 3)), zorder=1)
        # k goes in the axis label, not a panel title: a title sits in the
        # band the legend already occupies and the two collide.
        ax.set_xlabel(f"keyword weight  ·  k={k}")
        ax.set_xticks([0.1, 0.5, 1.0])
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
    f.axes[0].set_ylabel("nDCG")
    f.legend(ncol=2, ax=f.axes[0])
    f.footer(provenance(data, "ablation W · RRF k=60 · dashed line: the old default"))
    return f.fig


def fig_rerank(theme, data):
    """The cross-encoder against plain retrieval, at matched output size.

    Matched output size is the whole point: reranking a 40-candidate pool down
    to 10 has to be compared against plainly returning 10, not against
    returning 40, or the comparison measures depth instead of ranking.
    """
    f = Frame(theme, "What the cross-encoder buys",
              "Nothing, here: the best rerank at each depth loses to plain "
              "retrieval, well inside the intervals.",
              plot_h=2.5, m_left=0.92)
    t = f.t
    plain = {r["config"]["top_k"]: r["summary"] for r in data["a"]
             if r["config"]["mode"] == "hybrid"}
    ks = sorted({r["config"]["top_k"] for r in data["b"]})
    best, err = [], []
    for k in ks:
        top = max((r for r in data["b"] if r["config"]["top_k"] == k),
                  key=lambda r: r["summary"]["ndcg"])
        best.append(top["summary"]["ndcg"])
        err.append(top["summary"]["ndcg_ci"])

    ax = f.ax
    x = list(range(len(ks)))
    for i, k in enumerate(ks):
        # Between the two dots, not down the middle - drawn at a single x it
        # reads as a third error bar rather than as the pairing.
        ax.plot([i - 0.09, i + 0.09], [plain[k]["ndcg"], best[i]],
                color=t["baseline"], linewidth=0.9, zorder=2)
    ax.errorbar([i - 0.09 for i in x], [plain[k]["ndcg"] for k in ks],
                yerr=[plain[k]["ndcg_ci"] for k in ks], fmt="o", markersize=7.5,
                color=t["series"][0], label="no rerank", ecolor=t["muted"],
                elinewidth=1, capsize=0, markeredgecolor=t["surface"],
                markeredgewidth=1.5, zorder=3)
    ax.errorbar([i + 0.09 for i in x], best, yerr=err, fmt="o", markersize=7.5,
                color=t["series"][1], label="reranked", ecolor=t["muted"],
                elinewidth=1, capsize=0, markeredgecolor=t["surface"],
                markeredgewidth=1.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_xlim(-0.5, len(ks) - 0.5)
    ax.set_ylabel("nDCG")
    ax.set_xlabel("chunks returned")
    f.grid()
    f.legend(ncol=2)
    f.footer(provenance(data, "ablation B · hybrid (dense + ts_rank)"))
    return f.fig


def fig_latency_quality(theme, data, k=10):
    """Quality against latency.

    One series with direct labels rather than five colours: a scatter needs
    all-pairs colour separation and only the palette's first three slots
    validate all-pairs, so identity goes on the label instead.
    """
    f = Frame(theme, "What each retriever costs to run",
              "Fusion pays for both sides and lands slower than either. Mean "
              "wall-clock per query, cold cache.",
              plot_h=2.55, m_left=0.92)
    t = f.t
    quality = {r["config"]["mode"]: r["summary"]["ndcg"] for r in data["a"]
               if r["config"]["top_k"] == k}
    latency = {x["config"]["mode"]: x["latency_ms"]["total"] for x in data["timings"]
               if x["config"]["top_k"] == k and not x["config"]["rerank"]}

    ax = f.ax
    span = max(latency.values())
    for mode in MODES:
        if mode not in quality or mode not in latency:
            continue
        ax.scatter(latency[mode], quality[mode], s=80, color=t["series"][0],
                   edgecolor=t["surface"], linewidth=1.5, zorder=3)
        right = latency[mode] > 0.62 * span
        ax.annotate(label(mode), (latency[mode], quality[mode]),
                    textcoords="offset points", xytext=(-10 if right else 10, -3),
                    ha="right" if right else "left", fontsize=8, color=t["secondary"])
    ax.set_xlabel("mean latency per query (ms)")
    ax.set_ylabel(f"nDCG@{k}")
    ax.set_xlim(0, span * 1.12)
    f.grid()
    f.footer(provenance(data, f"timings · k={k}, no rerank"))
    return f.fig


def fig_latency_breakdown(theme, data, k=10):
    """Where the time goes. A sequential ramp, because these are parts of one
    magnitude rather than five identities."""
    f = Frame(theme, "Where the time goes",
              "Postgres full-text search dominates every mode that uses it; "
              "ranking itself is free by comparison.",
              plot_h=2.2, m_left=1.72)
    t = f.t
    stages = ["embed", "pinecone", "hydrate", "keyword_sql", "bm25", "fuse"]
    rows = {x["config"]["mode"]: x["latency_ms"] for x in data["timings"]
            if x["config"]["top_k"] == k and not x["config"]["rerank"]}
    modes = [m for m in MODES if m in rows]

    ax = f.ax
    left = [0.0] * len(modes)
    for stage, colour in zip(stages, t["ramp"]):
        vals = [rows[m].get(stage, 0.0) for m in modes]
        if sum(vals) == 0:
            continue
        ax.barh(range(len(modes)), vals, left=left, height=0.36, color=colour,
                label=label("bm25_rank" if stage == "bm25" else stage),
                # A 2px surface gap between segments, not a border around them.
                edgecolor=t["surface"], linewidth=1.5, zorder=3)
        left = [a + b for a, b in zip(left, vals)]
    for i, total in enumerate(left):
        ax.annotate(f"{total:.0f} ms", (total, i), textcoords="offset points",
                    xytext=(7, 0), va="center", fontsize=8, color=t["secondary"])
    ax.set_xlim(0, max(left) * 1.16)
    ax.set_xlabel("mean latency per query (ms)")
    f.rows(modes)
    f.grid(axis="x")
    f.legend(ncol=6)
    f.footer(provenance(data, f"timings · k={k}, no rerank"))
    return f.fig


def fig_exact_vs_ragas(theme, free):
    """The exact chunk-id metrics against Ragas' own non-LLM ones.

    They rest on different ground truth. Ours is set arithmetic on integer
    chunk ids; Ragas' is Levenshtein at a 0.5 threshold over the same seed
    strings. Reading them side by side is the check that the exact scoring is
    not quietly measuring something else.
    """
    f = Frame(theme, "Exact chunk-id scoring against Ragas' string matching",
              "Both treat the seed chunks as the whole relevant set, so both are "
              "lower bounds.",
              plot_h=1.15, m_left=1.62)
    t = f.t
    o = free["overall"]
    pairs = [
        ("recall", [("exact chunk ids", o["recall"], o["recall_ci"]),
                    ("Ragas non-LLM", o["ragas_recall"], o["ragas_recall_ci"])]),
        ("precision", [("exact chunk ids", o["map"], o["map_ci"]),
                       ("Ragas non-LLM", o["ragas_precision"], o["ragas_precision_ci"])]),
    ]
    ax = f.ax
    for row, (_, points) in enumerate(pairs):
        for i, (name, value, ci) in enumerate(points):
            ax.errorbar(value, row, xerr=ci, fmt="o", markersize=7.5,
                        color=t["series"][i], ecolor=t["muted"], elinewidth=1,
                        capsize=0, markeredgecolor=t["surface"], markeredgewidth=1.5,
                        zorder=3, label=name if row == 0 else None)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("score")
    f.rows(["recall", "precision"], texts=["recall", "precision / MAP"])
    f.grid(axis="x")
    f.legend(ncol=2)
    f.footer(provenance(free, f"free branch · {label(free['config']['mode'])} "
                              f"k={free['config']['top_k']}"))
    return f.fig


# ---------------------------------------------------------------------------
# Paid branch - generation and the judge
# ---------------------------------------------------------------------------
def fig_judged_metrics(theme, judged):
    """The four judged metrics, with and without the abstentions.

    Response relevancy scores a correct abstention 0 by construction - it
    generates questions from the answer and compares embeddings, and "the
    library does not cover this" produces nothing to compare. Reporting the
    two conditions side by side is the plan's mitigation, not a hedge.
    """
    metrics = ["faithfulness", "answer_relevancy",
               "llm_context_precision_with_reference", "context_recall"]
    n_abs = judged["n_abstentions"]
    f = Frame(theme, "The judged metrics",
              f"{judged['n_questions']} questions, of which {n_abs} were correct "
              "abstentions - and a correct abstention scores 0 on relevancy.",
              plot_h=1.95, m_left=1.72)
    t = f.t
    ax = f.ax
    for row, metric in enumerate(metrics):
        for i, (key, name) in enumerate((("means", "all questions"),
                                         ("means_excluding_abstentions",
                                          f"excluding the {n_abs} abstentions"))):
            value = judged[key][metric]
            ax.plot(value, row + (i - 0.5) * 0.26, "o", markersize=7.5,
                    color=t["series"][i], markeredgecolor=t["surface"],
                    markeredgewidth=1.5, zorder=3, label=name if row == 0 else None)
            ax.annotate(f"{value:.3f}", (value, row + (i - 0.5) * 0.26),
                        textcoords="offset points", xytext=(10, 0), va="center",
                        fontsize=7.5, color=t["muted"])
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("score")
    f.rows(metrics)
    f.grid(axis="x")
    f.legend(ncol=2)
    spend = judged["judge_spend"]
    f.footer(f"answerer {judged['answerer_model']}  ·  judge {judged['judge_model']}  ·  "
             f"${spend['usd']:.3f} per run  ·  {judged['_source']}")
    return f.fig


def fig_judge_agreement(theme, judge):
    """Whether the judge is worth believing, against human labels.

    Every judged number rests on this and nothing plotted it. Accuracy alone
    would flatter a judge that answered "supported" to everything, which is
    why kappa is the headline; the error direction is the other half - all
    seven mistakes are unsupported statements waved through, so the judge's
    bias is toward missing hallucinations rather than inventing them.
    """
    p, s = judge["primary"], judge["sensitivity"]
    f = Frame(theme, "Is the judge worth believing?",
              f"κ {p['kappa']:.2f} against {p['n']} human labels ({p['accuracy']:.0%} "
              f"agreement); κ {s['kappa']:.2f} dropping the contested 4.",
              plot_h=2.45, ncols=2, m_left=1.42, wspace=0.62,
              width_ratios=[1.25, 1])
    t = f.t

    # Left: where the disagreements are, by the kind of statement.
    order = sorted(judge["by_case"], key=lambda c: (judge["by_case"][c]["correct"]
                                                    / judge["by_case"][c]["n"], c))
    ax = f.axes[0]
    for row, case in enumerate(order):
        d = judge["by_case"][case]
        share = d["correct"] / d["n"]
        ax.barh(row, share, height=0.5, color=t["series"][0], zorder=3)
        ax.annotate(f"{d['correct']}/{d['n']}", (share, row),
                    textcoords="offset points", xytext=(7, 0), va="center",
                    fontsize=7.5, color=t["muted"])
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("agreement, by kind of statement")
    f.rows(order, ax=ax)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    # Right: the error direction, which is the part that matters downstream.
    ax = f.axes[1]
    # The two error rows carry the second categorical slot; the two agreement
    # rows recede. Emphatically NOT the status palette: a green bar on "waved
    # through" - which is the hallucination the judge would miss - inverts the
    # only reading that matters.
    cells = [("fp", "waved through", True), ("fn", "wrongly rejected", True),
             ("tn", "correctly rejected", False), ("tp", "correctly accepted", False)]
    for row, (key, name, is_error) in enumerate(cells):
        colour = t["series"][1] if is_error else t["recede"]
        ax.barh(row, p[key], height=0.5, color=colour, zorder=3)
        ax.annotate(str(p[key]), (p[key], row), textcoords="offset points",
                    xytext=(7, 0), va="center", fontsize=8, color=t["secondary"])
    ax.set_xlim(0, max(p[c[0]] for c in cells) * 1.24)
    ax.set_xlabel(f"statements  ·  all {p['fp'] + p['fn']} errors in one direction")
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([c[1] for c in cells], color=t["secondary"], fontsize=8.5)
    ax.set_ylim(len(cells) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    f.footer(f"judge {judge['model']}  ·  labels from eval/labelling/labels.json  ·  "
             "verdicts cached in eval/results/judge-verdicts.json")
    return f.fig


# ---------------------------------------------------------------------------
FIGURES = {
    "retrieval-depth": ("ablation+free", fig_retrieval_depth),
    "ranking-quality": ("ablation", fig_ranking_quality),
    "question-difficulty": ("free", fig_question_difficulty),
    "query-arms": ("ablation", fig_query_arms),
    "fusion-weight": ("ablation", fig_fusion_weight),
    "rerank-matched": ("ablation", fig_rerank),
    "latency-quality": ("ablation", fig_latency_quality),
    "latency-breakdown": ("ablation", fig_latency_breakdown),
    "exact-vs-ragas": ("free", fig_exact_vs_ragas),
    "judged-metrics": ("judged", fig_judged_metrics),
    "judge-agreement": ("judge", fig_judge_agreement),
}

# Which ablation key each figure needs, so a partial re-run degrades to a
# skip with a reason instead of a traceback.
NEEDS = {
    "retrieval-depth": ("a",), "ranking-quality": ("a",), "query-arms": ("c",),
    "fusion-weight": ("w",), "rerank-matched": ("a", "b"),
    "latency-quality": ("a", "timings"), "latency-breakdown": ("timings",),
}


def main():
    parser = argparse.ArgumentParser(description="Regenerate the committed eval figures")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help=f"one or more of: {', '.join(FIGURES)}")
    parser.add_argument("--themes", nargs="+", default=list(THEMES),
                        choices=list(THEMES))
    args = parser.parse_args()

    wanted = args.only or list(FIGURES)
    unknown = [n for n in wanted if n not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s): {', '.join(unknown)}")

    sources = {
        "ablation": load_ablation(),
        "free": load_json("free-*.json"),
        "judged": load_json("judged-*.json"),
        "judge": load_judge(),
    }
    for name, blob in sources.items():
        print(f"{name:>9}: {blob['_source'] if blob and '_source' in blob else ('ok' if blob else 'MISSING')}")
    print(f"-> {ASSETS}")

    written, skipped = 0, []
    for name in wanted:
        kind, fn = FIGURES[name]
        needed = [k for k in kind.split("+")]
        if any(sources[k] is None for k in needed):
            skipped.append(f"{name}: no {', '.join(k for k in needed if sources[k] is None)} results")
            continue
        missing = [k for k in NEEDS.get(name, ()) if k not in (sources["ablation"] or {})]
        if missing:
            skipped.append(f"{name}: ablation has no {', '.join(missing)}")
            continue
        for theme in args.themes:
            rc(theme)
            save(fn(theme, *(sources[k] for k in needed)), name, theme)
        written += 1
        print(f"  {name}")

    for note in skipped:
        print(f"  (skipped) {note}")
    print(f"{written} figures x {len(args.themes)} themes x {len(FORMATS)} formats")


if __name__ == "__main__":
    main()
