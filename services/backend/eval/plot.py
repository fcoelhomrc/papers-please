"""Renders a retrieval sweep (eval/sweep.py output) as a self-contained HTML
report: inline SVG line charts plus the full numbers as a table.

No plotting library - the charts are a few hundred lines of generated SVG,
which keeps the report a single file with no CDN, no build step, and no new
runtime dependency for something that draws six line charts.

Line chart, not bars: k is an ordered numeric axis and the question is the
shape of the curve (where does recall stop improving), which is what a line
answers and a bar chart obscures. Three series = three retrieval modes, so
color is doing identity work - a categorical palette, assigned in fixed
order, plus direct labels at each line's end. The table is not decoration:
the light-mode aqua sits below 3:1 on the surface, and a table view is the
documented relief for that.
"""
import json
from pathlib import Path

REPORTS_DIR = Path(__file__).parent / "reports"

# Categorical slots 1-3, validated for both surfaces (CVD + normal-vision
# separation, all-pairs) via the dataviz validator. Do not re-pick by eye.
SERIES = {
    "semantic": {"light": "#2a78d6", "dark": "#3987e5"},
    "keyword": {"light": "#eb6834", "dark": "#d95926"},
    "hybrid": {"light": "#1baf7a", "dark": "#199e70"},
}

METRICS = [
    ("recall", "Recall@k", "Share of relevant papers found"),
    ("ndcg", "nDCG@k", "Rank-aware, credits every relevant paper"),
    ("hit_rate", "Hit rate@k", "At least one relevant paper in the results"),
    ("mrr", "MRR", "Reciprocal rank of the first relevant paper"),
    ("precision", "Precision@k", "Share of the k slots that are relevant"),
]

W, H = 520, 300
# r is wide enough for the longest end label ("semantic 0.99" at 11px) to sit
# inside the viewBox - overflow:visible would let it escape the SVG but the
# grid cell still clips it.
PAD = {"l": 52, "r": 100, "t": 16, "b": 40}


def _scale(ks, values):
    x0, x1 = PAD["l"], W - PAD["r"]
    y0, y1 = H - PAD["b"], PAD["t"]
    kmin, kmax = min(ks), max(ks)

    def sx(k):
        # log-ish spacing: k values are 1,3,5,10,20,50 - linear spacing would
        # squash everything below 10 into the left margin
        import math
        lo, hi = math.log(kmin), math.log(kmax)
        return x0 + (x1 - x0) * ((math.log(k) - lo) / (hi - lo) if hi > lo else 0.5)

    def sy(v):
        return y0 + (y1 - y0) * v  # values are 0..1

    return sx, sy


def _chart(metric_key, title, subtitle, rows, ks, xlabel="k (results returned)") -> str:
    sx, sy = _scale(ks, None)
    parts = [
        f'<figure class="chart"><figcaption><h3>{title}</h3><p>{subtitle}</p></figcaption>',
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title} by k for each retrieval mode">',
    ]

    # gridlines + y axis, hairline and recessive
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = sy(frac)
        parts.append(
            f'<line class="grid" x1="{PAD["l"]}" y1="{y:.1f}" x2="{W - PAD["r"]}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{PAD["l"] - 8}" y="{y + 4:.1f}" text-anchor="end">{frac:.2f}</text>'
        )
    for k in ks:
        parts.append(
            f'<text class="tick" x="{sx(k):.1f}" y="{H - PAD["b"] + 18}" text-anchor="middle">{k}</text>'
        )
    parts.append(
        f'<text class="axis-title" x="{(PAD["l"] + W - PAD["r"]) / 2:.0f}" y="{H - 4}" '
        f'text-anchor="middle">{xlabel}</text>'
    )

    label_slots = []
    for mode in SERIES:
        pts = [(k, rows[(mode, k)][metric_key]) for k in ks if (mode, k) in rows]
        if not pts:
            continue
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(k):.1f},{sy(v):.1f}" for i, (k, v) in enumerate(pts)
        )
        parts.append(f'<path class="line" style="--c:var(--s-{mode})" d="{d}"/>')
        for k, v in pts:
            parts.append(
                f'<circle class="dot" style="--c:var(--s-{mode})" cx="{sx(k):.1f}" '
                f'cy="{sy(v):.1f}" r="4"><title>{mode} · k={k} · {metric_key}={v:.3f}</title></circle>'
            )
        lk, lv = pts[-1]
        label_slots.append({"mode": mode, "x": sx(lk) + 10, "y": sy(lv), "v": lv})

    # Direct labels are the documented relief for the light-mode contrast
    # warning, so they have to stay legible - but these series converge (hit
    # rate all reaches ~1.0, precision all collapses toward 0), and three
    # labels stacked on the same pixel is worse than none. Push them apart to
    # a minimum spacing, keeping their vertical order, so each stays next to
    # its own line without overlapping a neighbour.
    label_slots.sort(key=lambda s: s["y"])
    min_gap = 13.0
    for i in range(1, len(label_slots)):
        if label_slots[i]["y"] - label_slots[i - 1]["y"] < min_gap:
            label_slots[i]["y"] = label_slots[i - 1]["y"] + min_gap
    # keep the dodged stack inside the plot box
    overflow = label_slots[-1]["y"] - (H - PAD["b"]) if label_slots else 0
    if overflow > 0:
        for s in label_slots:
            s["y"] -= overflow

    for s in label_slots:
        parts.append(
            f'<text class="dlabel" style="--c:var(--s-{s["mode"]})" x="{s["x"]:.1f}" '
            f'y="{s["y"] + 4:.1f}">{s["mode"]} {s["v"]:.2f}</text>'
        )

    parts.append("</svg></figure>")
    return "".join(parts)


def _table(results) -> str:
    head = (
        "<tr><th>Mode</th><th>top_k</th><th>Rerank</th><th>k</th>"
        + "".join(f"<th>{label}</th>" for _, label, _ in METRICS)
        + "<th>Abstention</th></tr>"
    )
    body = []
    for r in results:
        rr = f"→{r['rerank_top_k']}" if r["rerank"] else "off"
        cells = "".join(f"<td>{r.get(k, float('nan')):.3f}</td>" for k, _, _ in METRICS)
        ab = r.get("abstention_precision")
        body.append(
            f"<tr><td>{r['mode']}</td><td>{r['top_k']}</td><td>{rr}</td><td>{r['k']}</td>"
            f"{cells}<td>{'—' if ab is None else f'{ab:.3f}'}</td></tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


def render(sweep: dict) -> str:
    results = sweep["results"]
    no_rerank = {(r["mode"], r["k"]): r for r in results if not r["rerank"]}
    ks = sorted({k for _, k in no_rerank})

    charts = "".join(_chart(key, label, sub, no_rerank, ks) for key, label, sub in METRICS)

    # Second view: hold the returned count fixed at 5 (a realistic generation
    # budget) and vary how many candidates the cross-encoder got to choose
    # from. Answers "does feeding the reranker more candidates help?", which
    # the first view cannot show because it holds rerank off.
    RERANK_CUT = 5
    reranked = {
        (r["mode"], r["top_k"]): r
        for r in results
        if r["rerank"] and r["rerank_top_k"] == RERANK_CUT
    }
    rr_ks = sorted({k for _, k in reranked})
    rerank_charts = ""
    if rr_ks:
        rerank_charts = "".join(
            _chart(key, label, sub, reranked, rr_ks,
                   xlabel=f"candidates reranked down to {RERANK_CUT}")
            for key, label, sub in METRICS[:3]
        )

    best = max(results, key=lambda r: r.get("recall", 0))
    best_line = (
        f"{best['mode']}, top_k={best['top_k']}, "
        f"rerank {'→' + str(best['rerank_top_k']) if best['rerank'] else 'off'}"
    )

    return f"""<style>
.viz-root {{
  color-scheme: light;
  --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --text-muted: #78766f; --grid: #e6e5e1;
  --s-semantic: {SERIES['semantic']['light']};
  --s-keyword: {SERIES['keyword']['light']};
  --s-hybrid: {SERIES['hybrid']['light']};
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{
    color-scheme: dark;
    --surface-1: #1a1a19; --text-primary: #fff; --text-secondary: #c3c2b7;
    --text-muted: #96948a; --grid: #302f2c;
    --s-semantic: {SERIES['semantic']['dark']};
    --s-keyword: {SERIES['keyword']['dark']};
    --s-hybrid: {SERIES['hybrid']['dark']};
  }}
}}
:root[data-theme="dark"] .viz-root {{
  color-scheme: dark;
  --surface-1: #1a1a19; --text-primary: #fff; --text-secondary: #c3c2b7;
  --text-muted: #96948a; --grid: #302f2c;
  --s-semantic: {SERIES['semantic']['dark']};
  --s-keyword: {SERIES['keyword']['dark']};
  --s-hybrid: {SERIES['hybrid']['dark']};
}}
.viz-root {{ background: var(--surface-1); color: var(--text-primary);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Helvetica, sans-serif;
  padding: 32px 24px; max-width: 1180px; margin: 0 auto; }}
h1 {{ font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }}
h2 {{ font-size: 17px; margin: 36px 0 10px; }}
h3 {{ font-size: 14px; margin: 0 0 2px; }}
.sub {{ color: var(--text-secondary); margin: 0 0 18px; }}
.meta {{ color: var(--text-muted); font-size: 13px; margin: 0 0 8px; }}
.legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 14px 0 22px;
  font-size: 13px; color: var(--text-secondary); }}
.legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
.legend i {{ width: 11px; height: 11px; border-radius: 3px; background: var(--c); }}
.grid-charts {{ display: grid; gap: 22px;
  grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); }}
.chart {{ margin: 0; }}
figcaption p {{ margin: 0 0 4px; font-size: 12.5px; color: var(--text-muted); }}
svg {{ width: 100%; height: auto; overflow: visible; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.tick {{ fill: var(--text-muted); font-size: 10.5px; }}
.axis-title {{ fill: var(--text-secondary); font-size: 11px; }}
.line {{ fill: none; stroke: var(--c); stroke-width: 2;
  stroke-linejoin: round; stroke-linecap: round; }}
.dot {{ fill: var(--c); stroke: var(--surface-1); stroke-width: 2; }}
.dlabel {{ fill: var(--c); font-size: 11px; font-weight: 600; }}
.tablewrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-size: 13px; width: 100%; margin-top: 8px; }}
th, td {{ text-align: right; padding: 5px 9px; border-bottom: 1px solid var(--grid);
  font-variant-numeric: tabular-nums; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--text-secondary); font-weight: 600; }}
code {{ background: color-mix(in srgb, var(--grid) 60%, transparent);
  padding: 1px 5px; border-radius: 4px; font-size: 0.92em; }}
</style>

<div class="viz-root">
<h1>Retrieval sweep — recall vs. k</h1>
<p class="sub">Rank metrics for every retrieval mode across <code>top_k</code>, computed
against the labelled eval set. No LLM judge involved.</p>
<p class="meta">Run {sweep['run_at'][:19]}Z · {sweep['n_questions']} questions ·
embed <code>{sweep['embed_model']}</code> · reranker <code>{sweep['reranker_model']}</code> ·
RRF k={sweep['rrf_k']} · hybrid pool {sweep['hybrid_candidates']} ·
{len(results)} configurations</p>

<div class="legend">
  <span style="--c:var(--s-semantic)"><i></i>semantic — dense vectors</span>
  <span style="--c:var(--s-keyword)"><i></i>keyword — Postgres full-text</span>
  <span style="--c:var(--s-hybrid)"><i></i>hybrid — RRF fusion of both</span>
</div>

<h2>Without reranking</h2>
<p class="meta">How many results you ask for, against what you get back.</p>
<div class="grid-charts">{charts}</div>

<h2>With reranking — candidates in, {RERANK_CUT} out</h2>
<p class="meta">Final result count held at {RERANK_CUT}; the x axis is how many
candidates the cross-encoder ranked. More candidates is not automatically better.</p>
<div class="grid-charts">{rerank_charts}</div>

<h2>Every configuration</h2>
<p class="meta">Best recall: {best_line} ({best.get('recall', 0):.3f}).
Abstention = share of the {results[0].get('n_abstention', 0)} no-answer questions
where retrieval correctly returned nothing relevant.</p>
<div class="tablewrap">{_table(results)}</div>
</div>"""


def write_report(sweep_path: Path) -> Path:
    sweep = json.loads(Path(sweep_path).read_text())
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"{Path(sweep_path).stem}.html"
    out.write_text(render(sweep))
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = max((Path(__file__).parent / "results").glob("sweep-*.json"))
    print(write_report(path))
