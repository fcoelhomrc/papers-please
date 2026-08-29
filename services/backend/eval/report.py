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


# Which versioned prompt each variant actually sends. Listing both on every
# report would credit a score to a prompt that never ran - a `fixed` run
# never loads the orchestrator prompt, and vice versa.
PROMPT_BY_VARIANT = {"fixed": "fixed_rag", "agentic": "orchestrator"}


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
    prompt_name = PROMPT_BY_VARIANT.get(variant)
    prompt_version = (output.get("prompt_versions") or {}).get(prompt_name)
    if prompt_version:
        lines.append(
            f"- **Prompt**: `prompts/{prompt_name}/{prompt_version}.md` — edits create a new "
            "version file, so this score stays attributable to an exact prompt."
        )
    retrieval = output.get("retrieval") or {}
    if retrieval:
        lines.append(
            f"- **Retrieval**: embed model `{retrieval.get('embed_model')}`, "
            f"query prefix `{retrieval.get('query_prompt', '').strip()}` "
            "(prescribed by the model card, not a tunable prompt)"
        )
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

    rm = output.get("retrieval_metrics") or {}
    if rm:
        lines.append("## Retrieval (no judge involved)")
        lines.append("")
        lines.append(
            f"Scored against `relevant_source_ids` labels over the "
            f"{rm.get('n_retrieval', 0)} questions that have a relevant paper; the "
            f"{rm.get('n_abstention', 0)} abstention questions are scored separately "
            "(recall is undefined when nothing is relevant)."
        )
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|---|---|")
        for name in ("recall", "precision", "hit_rate", "mrr", "ndcg"):
            if name in rm:
                lines.append(f"| {name} | {_fmt(rm[name])} |")
        if "abstention_precision" in rm:
            lines.append(f"| abstention_precision | {_fmt(rm['abstention_precision'])} |")
            lines.append(f"| mean_false_positives | {_fmt(rm['mean_false_positives'])} |")
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
