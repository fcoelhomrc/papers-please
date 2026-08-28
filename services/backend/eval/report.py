"""Renders eval/run.py's output into a committed markdown report.

Written to eval/reports/ (NOT gitignored, unlike eval/results/*.json -
these are meant to be kept and reviewed, not regenerated scratch output).

Deliberately kept inside services/backend/ rather than repo-root docs/:
the backend container only bind-mounts services/backend as /app, so it has
zero filesystem visibility outside that subtree - a path reaching above
/app would silently write somewhere on the container's own ephemeral
filesystem instead of the host repo (this happened once - the report
looked like it wrote successfully, but landed in a location that doesn't
exist outside the container and would be lost on teardown).
"""
from datetime import datetime, timezone
from pathlib import Path

REPORTS_DIR = Path(__file__).parent / "reports"


def _fmt(score) -> str:
    try:
        return f"{float(score):.3f}"
    except (TypeError, ValueError):
        return "—"


def _truncate(text: str, n: int = 90) -> str:
    text = str(text).replace("\n", " ").replace("|", "\\|").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def write_markdown_report(output: dict, dataset_rows: list[dict], model_name: str, judge_model_name: str) -> Path:
    variant = output["variant"]
    means = output["means"]
    per_question = output["per_question"]
    metric_names = list(means.keys())

    grounded = [r for r in dataset_rows if r.get("category") == "grounded"]
    edge = [r for r in dataset_rows if r.get("category") == "edge_case"]

    lines = []
    lines.append(f"# Eval report — `{variant}` pipeline")
    lines.append("")
    lines.append(f"- **Run at (UTC)**: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- **Pipeline model**: `{model_name}`")
    lines.append(f"- **Judge model**: `{judge_model_name}` (via Ragas' `LangchainLLMWrapper` — your Anthropic API key, no separate judge key)")
    lines.append(f"- **Dataset**: `eval/dataset.jsonl` — {len(dataset_rows)} questions ({len(grounded)} grounded, {len(edge)} edge case)")
    lines.append("- **Trigger**: manual only (`uv run python -m eval.run --variant ...`) — not wired into compose, CI, or any automated pipeline.")
    lines.append("")

    lines.append("## Summary (mean across all questions)")
    lines.append("")
    lines.append("| Metric | Mean score |")
    lines.append("|---|---|")
    for name in metric_names:
        lines.append(f"| {name} | {_fmt(means[name])} |")
    lines.append("")

    lines.append("## Summary by category")
    lines.append("")
    lines.append("| Category | N | " + " | ".join(metric_names) + " |")
    lines.append("|---|---|" + "---|" * len(metric_names))
    for cat_label, cat_rows in (("grounded", grounded), ("edge_case", edge)):
        idx_by_question = {r["user_input"]: r for r in per_question}
        matched = [idx_by_question[r["question"]] for r in cat_rows if r["question"] in idx_by_question]
        if not matched:
            continue
        cell = []
        for name in metric_names:
            vals = [m[name] for m in matched if name in m and m[name] == m[name]]  # drop NaN
            cell.append(_fmt(sum(vals) / len(vals)) if vals else "—")
        lines.append(f"| {cat_label} | {len(matched)} | " + " | ".join(cell) + " |")
    lines.append("")

    lines.append("## Per-question results")
    lines.append("")
    lines.append("| # | Category | Subtype/Domain | Question | " + " | ".join(metric_names) + " |")
    lines.append("|---|---|---|---|" + "---|" * len(metric_names))
    by_question = {r["user_input"]: r for r in per_question}
    for i, row in enumerate(dataset_rows, 1):
        scored = by_question.get(row["question"], {})
        tag = row.get("subtype") or row.get("domain") or ""
        cells = [_fmt(scored.get(name)) for name in metric_names]
        lines.append(
            f"| {i} | {row['category']} | {tag} | {_truncate(row['question'])} | " + " | ".join(cells) + " |"
        )
    lines.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = REPORTS_DIR / f"{variant}-{timestamp}.md"
    out_path.write_text("\n".join(lines))
    return out_path
