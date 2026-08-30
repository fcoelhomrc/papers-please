"""One page holding every eval we've run, plus the retrieval operating curves.

    uv run python -m eval.summary          # -> eval/reports/summary.html

The table comes from eval/ledger.jsonl (committed, so a run still appears
after its raw JSON is cleaned up). The curves need per-config detail, so they
come from the newest sweep JSON if it's still on disk and are simply omitted
if it isn't - a missing chart is better than a stale one.

The precision-recall curve is the point of this page. Retrieval theory says
that as k grows recall rises and precision falls, so a retriever traces a
curve through PR space rather than sitting at a point; comparing retrievers
means comparing curves, and a curve bowing toward the top-right dominates.
Reading our measured curves against that expectation is what catches a
retriever that is merely returning more rather than returning better.
"""
import json
from pathlib import Path

from eval.ledger import JUDGED_METRICS, RETRIEVAL_METRICS, load
from eval.plot import PAD, SERIES, W, H, REPORTS_DIR, _table as _sweep_table

RESULTS_DIR = Path(__file__).parent / "results"

ALL_METRICS = JUDGED_METRICS + RETRIEVAL_METRICS

# Ten metric columns plus config is wider than any screen. Abbreviated so the
# one-table view the page is for stays readable without horizontal scrolling
# on a laptop; the full names are in the tooltips.
SHORT = {
    "faithfulness": ("faith", "faithfulness (judged)"),
    "answer_relevancy": ("ans rel", "answer_relevancy (judged)"),
    "context_precision": ("ctx prec", "context_precision (judged)"),
    "context_recall": ("ctx rec", "context_recall (judged)"),
    "recall": ("recall", "recall@k"),
    "precision": ("prec", "precision@k"),
    "hit_rate": ("hit", "hit_rate@k"),
    "mrr": ("mrr", "mean reciprocal rank"),
    "ndcg": ("ndcg", "nDCG@k"),
    "abstention_precision": ("abst", "abstention_precision"),
}


def _pr_scale():
    """Both axes are 0..1 shares, so the same linear mapping serves each."""
    x0, x1 = PAD["l"], W - PAD["r"]
    y0, y1 = H - PAD["b"], PAD["t"]
    return (lambda v: x0 + (x1 - x0) * v), (lambda v: y0 + (y1 - y0) * v)


def _pr_chart(title, subtitle, series_points, label_series=None) -> str:
    """series_points: {mode: [(recall, precision, k), ...]} ordered by k."""
    sx, sy = _pr_scale()
    parts = [
        f'<figure class="chart"><figcaption><h3>{title}</h3><p>{subtitle}</p></figcaption>',
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">',
    ]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y, x = sy(frac), sx(frac)
        parts.append(f'<line class="grid" x1="{PAD["l"]}" y1="{y:.1f}" x2="{W-PAD["r"]}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{PAD["l"]-8}" y="{y+4:.1f}" text-anchor="end">{frac:.2f}</text>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{H-PAD["b"]+18}" text-anchor="middle">{frac:.2f}</text>')
    parts.append(
        f'<text class="axis-title" x="{(PAD["l"]+W-PAD["r"])/2:.0f}" y="{H-4}" '
        f'text-anchor="middle">recall</text>'
    )
    parts.append(
        f'<text class="axis-title" transform="rotate(-90 14 {H/2:.0f})" x="14" '
        f'y="{H/2:.0f}" text-anchor="middle">precision</text>'
    )

    end_labels = []
    for mode, pts in series_points.items():
        if not pts:
            continue
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(r):.1f},{sy(p):.1f}" for i, (r, p, _) in enumerate(pts)
        )
        parts.append(f'<path class="line" style="--c:var(--s-{mode})" d="{d}"/>')
        for r, p, k in pts:
            parts.append(
                f'<circle class="dot" style="--c:var(--s-{mode})" cx="{sx(r):.1f}" '
                f'cy="{sy(p):.1f}" r="4"><title>{mode} · k={k} · recall={r:.3f} '
                f'precision={p:.3f}</title></circle>'
            )
        # k is the curve's parameter, so it has to be readable somewhere - but
        # 18 labels across three series is noise. Label one reference series
        # and let the shared k values carry over; the rest is in tooltips and
        # the table.
        if mode == label_series:
            for r, p, k in pts:
                parts.append(
                    f'<text class="klabel" x="{sx(r)+7:.1f}" y="{sy(p)-6:.1f}">k={k}</text>'
                )
        lr, lp, _ = pts[-1]
        end_labels.append({"mode": mode, "x": sx(lr) + 9, "y": sy(lp)})

    # The curves all converge toward precision 0 at high k, so their end
    # labels land on the same pixel. Push them apart, keeping order.
    end_labels.sort(key=lambda s: s["y"])
    for i in range(1, len(end_labels)):
        if end_labels[i]["y"] - end_labels[i - 1]["y"] < 13.0:
            end_labels[i]["y"] = end_labels[i - 1]["y"] + 13.0
    overflow = end_labels[-1]["y"] - (H - PAD["b"]) if end_labels else 0
    if overflow > 0:
        for s_ in end_labels:
            s_["y"] -= overflow
    for s_ in end_labels:
        parts.append(
            f'<text class="dlabel" style="--c:var(--s-{s_["mode"]})" x="{s_["x"]:.1f}" '
            f'y="{s_["y"]+4:.1f}">{s_["mode"]}</text>'
        )
    parts.append("</svg></figure>")
    return "".join(parts)


def _ledger_table(rows) -> str:
    cols = [m for m in ALL_METRICS if any(m in r["metrics"] for r in rows)]
    head = (
        "<tr><th>Run</th><th>Kind</th><th>Best config</th><th>Model</th><th>N</th>"
        + "".join(
            f'<th title="{SHORT.get(c, (c, c))[1]}">{SHORT.get(c, (c, c))[0]}</th>'
            for c in cols
        )
        + "</tr>"
    )
    body = []
    for r in sorted(rows, key=lambda r: r["run_at"]):
        cells = "".join(
            f"<td>{r['metrics'][c]:.3f}</td>" if c in r["metrics"] else '<td class="na">—</td>'
            for c in cols
        )
        n = r.get("n_configs")
        body.append(
            f"<tr><td><code>{r['id']}</code></td>"
            f"<td><span class='kind kind-{r['kind']}'>{r['kind'].replace('_',' ')}</span></td>"
            f"<td>{r['variant']}{f' <span class=dim>({n} configs)</span>' if n else ''}</td>"
            f"<td><code>{r.get('model') or '—'}</code></td>"
            f"<td>{r.get('n_questions','—')}</td>{cells}</tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


def headline(rows, sweep) -> str:
    """A dashboard is scanned, not read top-to-bottom, so the state that would
    change what you do next goes above the table rather than inside it."""
    judged = [r for r in rows if r["kind"] == "judged"]
    sweeps = [r for r in rows if r["kind"] == "retrieval_sweep"]

    stats = [("Runs recorded", str(len(rows)), "judged, retrieval and threshold")]

    if sweeps:
        best = max(sweeps, key=lambda r: r["metrics"].get("ndcg", 0))
        stats.append(
            (
                "Best retrieval",
                f"{best['metrics'].get('ndcg', 0):.3f}",
                f"nDCG · {best['variant']}",
            )
        )
    if judged:
        b = max(judged, key=lambda r: r["metrics"].get("faithfulness", 0))
        stats.append(
            (
                "Best faithfulness",
                f"{b['metrics'].get('faithfulness', 0):.3f}",
                f"judged · {b['variant']} pipeline",
            )
        )
    if sweep:
        ab = max(
            (r.get("abstention_precision", 0) for r in sweep["results"]), default=0
        )
        stats.append(
            (
                "Abstention",
                f"{ab:.3f}",
                "share of no-answer questions correctly returning nothing",
            )
        )

    cards = "".join(
        f'<div class="stat"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-note">{note}</div></div>'
        for label, value, note in stats
    )
    return f'<div class="stats">{cards}</div>'


def latest_sweep() -> dict | None:
    paths = sorted(RESULTS_DIR.glob("sweep-*.json"))
    if not paths:
        return None
    data = json.loads(paths[-1].read_text())
    data["_source"] = paths[-1].name
    return data


def render(rows, sweep) -> str:
    charts = ""
    if sweep:
        res = sweep["results"]
        no_rr = {}
        for r in res:
            if not r["rerank"]:
                no_rr.setdefault(r["mode"], []).append((r["recall"], r["precision"], r["k"]))
        for m in no_rr:
            no_rr[m].sort(key=lambda t: t[2])

        rr = {}
        pool = max((r["top_k"] for r in res if r["rerank"]), default=None)
        for r in res:
            if r["rerank"] and r["top_k"] == pool:
                rr.setdefault(r["mode"], []).append((r["recall"], r["precision"], r["k"]))
        for m in rr:
            rr[m].sort(key=lambda t: t[2])

        charts = (
            _pr_chart(
                "Precision–recall, no reranking",
                "Each point is one k. Theory: recall up, precision down as k grows.",
                no_rr, label_series="semantic",
            )
            + _pr_chart(
                f"Precision–recall, reranked from {pool} candidates",
                "k is where the reranked list is cut. The three curves sit on top "
                "of each other: reranking a wide pool erases the retrieval mode's "
                "contribution entirely.",
                rr, label_series="semantic",
            )
        )

    table_note = (
        f"Curves from <code>{sweep['_source']}</code>."
        if sweep
        else "No sweep JSON on disk, so the curves are omitted; the table is from the ledger."
    )

    return f"""<style>
.viz-root {{ color-scheme: light;
  --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --text-muted:#78716c; --grid:#e7e5e4; --accent:#4338ca; --card:#ffffff;
  --s-semantic:{SERIES['semantic']['light']}; --s-keyword:{SERIES['keyword']['light']};
  --s-hybrid:{SERIES['hybrid']['light']}; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) .viz-root {{
  color-scheme: dark; --surface-1:#1a1a19; --text-primary:#fff; --text-secondary:#c3c2b7;
  --text-muted:#a1a1aa; --grid:#27272a; --accent:#818cf8; --card:#18181b;
  --s-semantic:{SERIES['semantic']['dark']}; --s-keyword:{SERIES['keyword']['dark']};
  --s-hybrid:{SERIES['hybrid']['dark']}; }} }}
:root[data-theme="dark"] .viz-root {{
  color-scheme: dark; --surface-1:#1a1a19; --text-primary:#fff; --text-secondary:#c3c2b7;
  --text-muted:#96948a; --grid:#302f2c;
  --s-semantic:{SERIES['semantic']['dark']}; --s-keyword:{SERIES['keyword']['dark']};
  --s-hybrid:{SERIES['hybrid']['dark']}; }}
.viz-root {{ background:var(--surface-1); color:var(--text-primary);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Helvetica,sans-serif;
  padding:32px 24px; max-width:1240px; margin:0 auto; }}
h1 {{ font-size:24px; margin:0 0 4px; letter-spacing:-0.01em; }}
h2 {{ font-size:17px; margin:34px 0 8px; }}
h3 {{ font-size:14px; margin:0 0 2px; }}
.sub {{ color:var(--text-secondary); margin:0 0 6px; }}
.meta {{ color:var(--text-muted); font-size:13px; margin:0 0 10px; }}
figcaption p {{ margin:0 0 4px; font-size:12.5px; color:var(--text-muted); }}
.grid-charts {{ display:grid; gap:22px; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); }}
.chart {{ margin:0; }}
svg {{ width:100%; height:auto; overflow:visible; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.tick {{ fill:var(--text-muted); font-size:10.5px; }}
.axis-title {{ fill:var(--text-secondary); font-size:11px; }}
.line {{ fill:none; stroke:var(--c); stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }}
.dot {{ fill:var(--c); stroke:var(--surface-1); stroke-width:2; }}
.dlabel {{ fill:var(--c); font-size:11px; font-weight:600; }}
.klabel {{ fill:var(--text-muted); font-size:9.5px; }}
.tablewrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; font-size:13px; width:100%; }}
th,td {{ text-align:right; padding:6px 9px; border-bottom:1px solid var(--grid);
  font-variant-numeric:tabular-nums; white-space:nowrap; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),
th:nth-child(3),td:nth-child(3) {{ text-align:left; }}
th {{ color:var(--text-secondary); font-weight:600; }}
td.na {{ color:var(--text-muted); }}
.dim {{ color:var(--text-muted); }}
code {{ background:color-mix(in srgb,var(--grid) 60%,transparent); padding:1px 5px;
  border-radius:4px; font-size:0.92em; }}
.kind {{ display:inline-block; padding:1px 7px; border-radius:99px; font-size:11px; }}
.kind-judged {{ background:color-mix(in srgb,var(--s-keyword) 18%,transparent); color:var(--s-keyword); }}
.kind-retrieval_sweep {{ background:color-mix(in srgb,var(--s-hybrid) 18%,transparent); color:var(--s-hybrid); }}
.kind-threshold_sweep {{ background:color-mix(in srgb,var(--s-semantic) 18%,transparent); color:var(--s-semantic); }}
.legend {{ display:flex; gap:18px; flex-wrap:wrap; margin:10px 0 20px; font-size:13px;
  color:var(--text-secondary); }}
.legend span {{ display:inline-flex; align-items:center; gap:7px; }}
.legend i {{ width:11px; height:11px; border-radius:3px; background:var(--c); }}
.stats {{ display:grid; gap:12px; margin:20px 0 8px;
  grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); }}
.stat {{ background:var(--card); border:1px solid var(--grid); border-radius:10px;
  padding:14px 16px; display:flex; flex-direction:column; gap:2px; }}
.stat-label {{ font-size:11px; text-transform:uppercase; letter-spacing:.05em;
  color:var(--text-muted); }}
.stat-value {{ font-size:28px; font-weight:600; font-variant-numeric:tabular-nums;
  letter-spacing:-0.02em; }}
.stat-note {{ font-size:12px; color:var(--text-muted); }}
:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important;
  transition:none !important; }} }}
</style>

<title>papers-please · evaluation results</title>
<div class="viz-root">
<h1>Evaluation results</h1>
<p class="sub">Every eval run recorded so far, and the retrieval operating curves.</p>
<p class="meta">{len(rows)} runs from <code>eval/ledger.jsonl</code>. {table_note}
Blank cells mean the metric doesn't apply to that run kind - judged runs have no
recall labels, retrieval runs have no LLM judge.</p>

{headline(rows, sweep)}

<h2>All runs</h2>
<div class="tablewrap">{_ledger_table(rows)}</div>

<h2>Precision–recall operating curves</h2>
<p class="meta">The trade-off retrieval theory predicts: asking for more results can only
find more of what's relevant (recall rises) while diluting what comes back (precision
falls). A retriever is therefore a curve, not a point, and one curve dominates another
by sitting above and to the right of it.</p>
<div class="legend">
  <span style="--c:var(--s-semantic)"><i></i>semantic</span>
  <span style="--c:var(--s-keyword)"><i></i>keyword</span>
  <span style="--c:var(--s-hybrid)"><i></i>hybrid</span>
</div>
<div class="grid-charts">{charts}</div>

<h2>Newest sweep, every configuration</h2>
<div class="tablewrap">{_sweep_table(sweep["results"]) if sweep else ""}</div>
</div>"""


def main():
    rows = load()
    if not rows:
        raise SystemExit("ledger is empty - run `python -m eval.ingest` first")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "summary.html"
    out.write_text(render(rows, latest_sweep()))
    print(out)
    return out


if __name__ == "__main__":
    main()
