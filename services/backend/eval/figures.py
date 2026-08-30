"""Generates the eval figures committed under assets/eval/ and linked from the README.

    uv run python -m eval.figures

Regenerating is one command with no browser, no plotting library and no
network: the figures are SVG written directly, so adding a new eval config
means re-running the sweep and re-running this. That was the requirement -
these have to stay current as configurations accumulate, and anything needing
a headless browser to rasterize would rot.

Two files per figure, `-light` and `-dark`, wired into the README with
<picture> + prefers-color-scheme so the plots follow GitHub's theme instead
of glowing white in dark mode.

Design follows the app's own UI rather than inventing a second identity:
zinc/stone neutrals, indigo accent, 14px card radius, the same type scale.
The categorical palette is anchored on that indigo and was checked with the
dataviz validator against both surfaces - all checks pass in both, with
tritan separation in the 6-8 floor band, which is legal because every series
carries a direct label.
"""
import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "eval"
RESULTS = Path(__file__).parent / "results"

THEMES = {
    "light": {
        "card": "#ffffff", "page": "#fafaf9", "border": "#e7e5e4",
        "ink": "#1c1917", "muted": "#78716c", "faint": "#a8a29e",
        "grid": "#eeecea", "chip": "#f5f5f4",
        "series": {"semantic": "#4f46e5", "keyword": "#ea580c", "hybrid": "#0d9488"},
    },
    "dark": {
        "card": "#18181b", "page": "#0c0c0d", "border": "#27272a",
        "ink": "#fafafa", "muted": "#a1a1aa", "faint": "#71717a",
        "grid": "#232326", "chip": "#202023",
        "series": {"semantic": "#6366f1", "keyword": "#ea580c", "hybrid": "#0d9488"},
    },
}

FONT = ("ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',"
        "Helvetica,Arial,sans-serif")


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Plot:
    """One plot area inside a card: axes, gridlines, series, direct labels."""

    def __init__(self, t, x, y, w, h, xlabel, ylabel, xticks, xpos, ymax=1.0):
        self.t, self.x, self.y, self.w, self.h = t, x, y, w, h
        self.xlabel, self.ylabel = xlabel, ylabel
        self.xticks, self.xpos, self.ymax = xticks, xpos, ymax
        self.pad_r = 92  # room for the direct labels
        self.out = []

    def sx(self, v):
        return self.x + (self.w - self.pad_r) * self.xpos(v)

    def sy(self, v):
        return self.y + self.h - (self.h * (v / self.ymax))

    def axes(self):
        t = self.t
        for frac in (0, 0.25, 0.5, 0.75, 1.0):
            val = frac * self.ymax
            yy = self.sy(val)
            self.out.append(
                f'<line x1="{self.x}" y1="{yy:.1f}" x2="{self.x + self.w - self.pad_r}" '
                f'y2="{yy:.1f}" stroke="{t["grid"]}" stroke-width="1"/>'
            )
            self.out.append(
                f'<text x="{self.x - 9}" y="{yy + 3.5:.1f}" text-anchor="end" '
                f'font-size="10.5" fill="{t["faint"]}">{val:.2f}</text>'
            )
        for tick in self.xticks:
            self.out.append(
                f'<text x="{self.sx(tick):.1f}" y="{self.y + self.h + 17:.1f}" '
                f'text-anchor="middle" font-size="10.5" fill="{t["faint"]}">{esc(tick)}</text>'
            )
        self.out.append(
            f'<text x="{self.x + (self.w - self.pad_r) / 2:.0f}" y="{self.y + self.h + 36:.0f}" '
            f'text-anchor="middle" font-size="11" fill="{t["muted"]}">{esc(self.xlabel)}</text>'
        )
        cy = self.y + self.h / 2
        self.out.append(
            f'<text transform="rotate(-90 {self.x - 36:.0f} {cy:.0f})" x="{self.x - 36:.0f}" '
            f'y="{cy:.0f}" text-anchor="middle" font-size="11" fill="{t["muted"]}">'
            f'{esc(self.ylabel)}</text>'
        )
        return self

    def series(self, points_by_name, colors=None, label_fmt="{name}", label_x=None):
        """points_by_name: {name: [(xval, yval), ...]} already ordered."""
        t = self.t
        colors = colors or t["series"]
        labels = []
        for name, pts in points_by_name.items():
            if not pts:
                continue
            c = colors[name] if isinstance(colors, dict) else colors
            d = " ".join(
                f"{'M' if i == 0 else 'L'}{self.sx(px):.1f},{self.sy(py):.1f}"
                for i, (px, py) in enumerate(pts)
            )
            self.out.append(
                f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" '
                f'stroke-linejoin="round" stroke-linecap="round"/>'
            )
            for px, py in pts:
                self.out.append(
                    f'<circle cx="{self.sx(px):.1f}" cy="{self.sy(py):.1f}" r="4" '
                    f'fill="{c}" stroke="{t["card"]}" stroke-width="2"/>'
                )
            lx, ly = pts[-1]
            labels.append({
                "x": label_x if label_x is not None else self.sx(lx) + 11,
                "y": self.sy(ly), "c": c,
                "text": label_fmt.format(name=name, value=ly),
            })

        # Curves converge, so their end labels land on the same pixel. Push
        # them apart keeping order - three labels stacked is worse than none,
        # and the labels are the secondary encoding the palette relies on.
        labels.sort(key=lambda s: s["y"])
        for i in range(1, len(labels)):
            if labels[i]["y"] - labels[i - 1]["y"] < 14:
                labels[i]["y"] = labels[i - 1]["y"] + 14
        over = labels[-1]["y"] - (self.y + self.h) if labels else 0
        if over > 0:
            for s in labels:
                s["y"] -= over
        for s in labels:
            self.out.append(
                f'<text x="{s["x"]:.1f}" y="{s["y"] + 4:.1f}" font-size="11" '
                f'font-weight="600" fill="{s["c"]}">{esc(s["text"])}</text>'
            )
        return self

    def note(self, x, y, text, color=None):
        self.out.append(
            f'<text x="{x:.0f}" y="{y:.0f}" font-size="10.5" '
            f'fill="{color or self.t["faint"]}">{esc(text)}</text>'
        )
        return self

    def svg(self):
        return "".join(self.out)


def card(t, w, h, title, subtitle, body, legend=None):
    """The app's card: 14px radius, 1px border, generous padding."""
    leg = ""
    if legend:
        # Under the subtitle, not at the foot of the card: at the bottom it
        # sat on top of the x-axis title.
        lx = 24
        for name, color in legend:
            leg += (
                f'<rect x="{lx}" y="{69}" width="10" height="10" rx="3" fill="{color}"/>'
                f'<text x="{lx + 15}" y="{78}" font-size="11.5" fill="{t["muted"]}">'
                f'{esc(name)}</text>'
            )
            lx += 24 + 7.0 * len(name)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" \
viewBox="0 0 {w} {h}" font-family="{FONT}">
<rect width="{w}" height="{h}" rx="14" fill="{t['card']}" stroke="{t['border']}"/>
<text x="24" y="34" font-size="15" font-weight="600" fill="{t['ink']}" \
letter-spacing="-0.01em">{esc(title)}</text>
<text x="24" y="53" font-size="12" fill="{t['muted']}">{esc(subtitle)}</text>
{body}{leg}
</svg>"""


def log_pos(values):
    import math

    lo, hi = math.log(min(values)), math.log(max(values))
    return lambda v: (math.log(v) - lo) / (hi - lo) if hi > lo else 0.5


def lin_pos(values):
    lo, hi = min(values), max(values)
    return lambda v: (v - lo) / (hi - lo) if hi > lo else 0.5


def _latest_json(pattern):
    paths = sorted(RESULTS.glob(pattern))
    return json.loads(paths[-1].read_text()) if paths else None


def _by_mode(results, pred, xkey, ykey):
    out = {}
    for r in results:
        if pred(r):
            out.setdefault(r["mode"], []).append((r[xkey], r[ykey]))
    for m in out:
        out[m].sort()
    return out


def fig_recall_vs_k(t, sweep):
    res = sweep["results"]
    ks = sorted({r["k"] for r in res if not r["rerank"]})
    W, H = 900, 372
    body = ""
    for i, (metric, label) in enumerate((("recall", "Recall@k"), ("ndcg", "nDCG@k"))):
        p = Plot(t, 80 + i * 445, 112, 400, 186, "k (results returned)", label, ks, log_pos(ks))
        p.axes().series(_by_mode(res, lambda r: not r["rerank"], "k", metric))
        body += p.svg()
    return card(
        t, W, H,
        "Retrieval quality against how many results you ask for",
        "Recall keeps climbing with k; nDCG flattens once the extra results stop being relevant.",
        body,
        legend=[(n, t["series"][n]) for n in ("semantic", "keyword", "hybrid")],
    )


def fig_precision_recall(t, sweep):
    res = sweep["results"]
    W, H = 900, 376
    body = ""
    no_rr, rr = {}, {}
    pool = max((r["top_k"] for r in res if r["rerank"]), default=None)
    for r in res:
        if not r["rerank"]:
            no_rr.setdefault(r["mode"], []).append((r["recall"], r["precision"], r["k"]))
        elif r["top_k"] == pool:
            rr.setdefault(r["mode"], []).append((r["recall"], r["precision"], r["k"]))
    for d in (no_rr, rr):
        for m in d:
            d[m].sort(key=lambda x: x[2])

    for i, (data, sub) in enumerate(((no_rr, "no reranking"), (rr, f"reranked from {pool} candidates"))):
        p = Plot(t, 80 + i * 445, 124, 400, 178, "recall", "precision",
                 [0.0, 0.25, 0.5, 0.75, 1.0], lambda v: v)
        p.axes()
        # Pinned to a right-hand column: anchored to each curve's last point
        # they landed on top of the k= annotations, which cluster in exactly
        # the same corner as the curves converge.
        p.series({m: [(r, pr) for r, pr, _ in pts] for m, pts in data.items()},
                 label_x=p.x + p.w - p.pad_r + 14)
        ref = "semantic" if "semantic" in data else next(iter(data), None)
        if ref:
            for r, pr, k in data[ref]:
                # below the point once the curve dives into the crowded corner
                dy = 14 if pr < 0.22 else -8
                p.note(p.sx(r) - 8, p.sy(pr) + dy, f"k={k}")
        p.note(80 + i * 445, 100, sub, t["muted"])
        body += p.svg()

    return card(
        t, W, H,
        "Precision–recall operating curves",
        "Asking for more can only raise recall while diluting precision — a retriever is a curve, not a point.",
        body,
        legend=[(n, t["series"][n]) for n in ("semantic", "keyword", "hybrid")],
    )


def fig_keyword_fix(t, before, after):
    res_b = [r for r in before["results"] if not r["rerank"] and r["mode"] == "keyword"]
    res_a = [r for r in after["results"] if not r["rerank"] and r["mode"] == "keyword"]
    ks = sorted({r["k"] for r in res_a})
    W, H = 445, 372
    p = Plot(t, 80, 112, 330, 186, "k (results returned)", "Recall@k", ks, log_pos(ks))
    p.pad_r = 74
    p.axes()
    p.series(
        {"after": sorted((r["k"], r["recall"]) for r in res_a),
         "before": sorted((r["k"], r["recall"]) for r in res_b)},
        colors={"after": t["series"]["hybrid"], "before": t["faint"]},
    )
    return card(
        t, W, H,
        "Keyword search: AND vs OR matching",
        "Flat with k is a matching failure, not a ranking one.",
        p.svg(),
        legend=[("before", t["faint"]), ("after", t["series"]["hybrid"])],
    )


def fig_abstention(t, thresholds):
    res = sorted(thresholds["results"], key=lambda r: r["min_rerank_score"])
    floors = [r["min_rerank_score"] for r in res]
    W, H = 445, 372
    p = Plot(t, 80, 112, 330, 186, "minimum cross-encoder score", "share",
             [f for f in floors if f % 4 == 0], lin_pos(floors))
    p.pad_r = 84
    p.axes()
    p.series(
        {"recall": [(r["min_rerank_score"], r["recall"]) for r in res],
         "abstains": [(r["min_rerank_score"], r.get("abstention_precision", 0)) for r in res]},
        colors={"recall": t["series"]["semantic"], "abstains": t["series"]["keyword"]},
    )
    x = p.sx(-8.0)
    p.out.insert(0, (
        f'<line x1="{x:.1f}" y1="{p.y}" x2="{x:.1f}" y2="{p.y + p.h}" '
        f'stroke="{t["faint"]}" stroke-width="1" stroke-dasharray="3 3"/>'
    ))
    p.note(x - 66, p.y + 12, "default −8.0", t["muted"])
    return card(
        t, W, H,
        "What a score floor buys, and costs",
        "Left of the line the floor is free; right of it recall pays for abstention.",
        p.svg(),
        legend=[("recall", t["series"]["semantic"]), ("abstains correctly", t["series"]["keyword"])],
    )


def main():
    sweeps = sorted(RESULTS.glob("sweep-*.json"))
    if not sweeps:
        raise SystemExit("no sweep results - run `python -m eval.sweep` first")
    after = json.loads(sweeps[-1].read_text())
    before = json.loads(sweeps[0].read_text())

    thresholds = None
    for path in sorted(RESULTS.glob("thresholds-*.json"), reverse=True):
        data = json.loads(path.read_text())
        if data["results"] and "min_rerank_score" in data["results"][0]:
            thresholds = data
            break

    ASSETS.mkdir(parents=True, exist_ok=True)
    written = []
    for theme, t in THEMES.items():
        figs = {
            "recall-vs-k": fig_recall_vs_k(t, after),
            "precision-recall": fig_precision_recall(t, after),
            "keyword-fix": fig_keyword_fix(t, before, after),
        }
        if thresholds:
            figs["abstention"] = fig_abstention(t, thresholds)
        for name, svg in figs.items():
            p = ASSETS / f"{name}-{theme}.svg"
            p.write_text(svg)
            written.append(p)

    for p in sorted(written):
        print(f"  assets/eval/{p.name}  ({p.stat().st_size // 1024} kB)")
    print(f"{len(written)} figures written")


if __name__ == "__main__":
    main()
