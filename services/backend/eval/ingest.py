"""Backfills eval/ledger.jsonl from whatever eval artifacts exist on disk.

Two artifact shapes, because they were built at different times:
  - eval/reports/*.md   - judged Ragas runs. Their JSON is gitignored scratch
                          and the early ones are already gone, so the
                          committed markdown is the only surviving record;
                          its summary table gets parsed back out.
  - eval/results/*.json - retrieval sweeps and threshold sweeps.

Idempotent: re-running adds only what's missing.
"""
import json
import re
from pathlib import Path

from eval.ledger import append

REPORTS_DIR = Path(__file__).parent / "reports"
RESULTS_DIR = Path(__file__).parent / "results"


def _bullet(text: str, label: str) -> str | None:
    m = re.search(rf"^- \*\*{re.escape(label)}\*\*: (.+)$", text, re.M)
    if not m:
        return None
    return m.group(1).strip().strip("`").split("`")[0].strip()


def parse_judged_report(path: Path) -> dict | None:
    """Pull the summary table and metadata back out of a committed report."""
    text = path.read_text()
    variant = re.search(r"# Eval report — `(\w+)` pipeline", text)
    if not variant:
        return None

    means = {}
    section = text.split("## Summary (mean across all questions)")[-1].split("##")[0]
    for name, score in re.findall(r"^\| (\w+) \| ([\d.]+) \|$", section, re.M):
        means[name] = float(score)

    prompt = re.search(r"`prompts/(\w+)/(v\d+)\.md`", text)
    return {
        "id": path.stem,
        "kind": "judged",
        "run_at": _bullet(text, "Run at (UTC)") or "",
        "variant": variant.group(1),
        "model": _bullet(text, "Pipeline model") or "",
        "judge_model": _bullet(text, "Judge model") or "",
        "prompt": f"{prompt.group(1)}/{prompt.group(2)}" if prompt else None,
        "n_questions": 50,
        "metrics": means,
        "source": f"eval/reports/{path.name}",
    }


def _best(results: list[dict], metric: str) -> dict:
    return max(results, key=lambda r: r.get(metric, 0))


def parse_sweep(path: Path) -> dict | None:
    """A sweep is many configs; the ledger records the run plus its best
    configuration by nDCG, which is the rank-aware metric that credits every
    relevant document rather than only the first."""
    data = json.loads(path.read_text())
    if data.get("kind") != "retrieval_sweep":
        return None

    best = _best(data["results"], "ndcg")
    return {
        "id": path.stem,
        "kind": "retrieval_sweep",
        "run_at": data["run_at"],
        "variant": f"{best['mode']} top_k={best['top_k']}"
        + (f" rerank->{best['rerank_top_k']}" if best["rerank"] else " rerank=off"),
        "model": data.get("embed_model", ""),
        "reranker": data.get("reranker_model", ""),
        "n_questions": data.get("n_questions", 0),
        "n_configs": len(data["results"]),
        "metrics": {
            m: best[m]
            for m in ("recall", "precision", "hit_rate", "mrr", "ndcg", "abstention_precision")
            if m in best
        },
        "source": f"eval/results/{path.name}",
    }


def parse_thresholds(path: Path) -> dict | None:
    data = json.loads(path.read_text())
    if data.get("kind") != "threshold_sweep":
        return None
    best = _best(data["results"], "ndcg")
    return {
        "id": path.stem,
        "kind": "threshold_sweep",
        "run_at": data["run_at"],
        "variant": f"{best['mode']} vec>={best.get('min_vector_score', 0):.2f}",
        "model": data.get("embed_model", ""),
        "n_questions": data.get("n_questions", 0),
        "n_configs": len(data["results"]),
        "metrics": {
            m: best[m]
            for m in ("recall", "precision", "hit_rate", "mrr", "ndcg", "abstention_precision")
            if m in best
        },
        "source": f"eval/results/{path.name}",
    }


def ingest_all() -> int:
    added = 0
    for path in sorted(REPORTS_DIR.glob("*.md")):
        rec = parse_judged_report(path)
        if rec and append(rec):
            added += 1
            print(f"  + judged      {rec['id']}")
    for path in sorted(RESULTS_DIR.glob("sweep-*.json")):
        rec = parse_sweep(path)
        if rec and append(rec):
            added += 1
            print(f"  + sweep       {rec['id']}")
    for path in sorted(RESULTS_DIR.glob("thresholds-*.json")):
        rec = parse_thresholds(path)
        if rec and append(rec):
            added += 1
            print(f"  + thresholds  {rec['id']}")
    return added


if __name__ == "__main__":
    n = ingest_all()
    print(f"{n} new run(s) recorded in eval/ledger.jsonl")
