"""Unit tests for the eval ledger, its ingestion, and the summary render."""
import json

import pytest

from eval.ingest import parse_judged_report, parse_sweep, parse_thresholds
from eval.ledger import append, load
from eval.summary import render


class TestLedger:
    def test_append_and_load(self, tmp_path):
        p = tmp_path / "l.jsonl"
        assert append({"id": "a", "metrics": {}}, p) is True
        assert [r["id"] for r in load(p)] == ["a"]

    def test_append_is_idempotent_on_id(self, tmp_path):
        """Re-ingesting existing artifacts must not duplicate rows."""
        p = tmp_path / "l.jsonl"
        append({"id": "a", "metrics": {}}, p)
        assert append({"id": "a", "metrics": {"recall": 1.0}}, p) is False
        assert len(load(p)) == 1

    def test_record_without_id_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="'id'"):
            append({"metrics": {}}, tmp_path / "l.jsonl")

    def test_missing_ledger_loads_empty(self, tmp_path):
        assert load(tmp_path / "nope.jsonl") == []


JUDGED_MD = """# Eval report — `agentic` pipeline

- **Run at (UTC)**: 2026-08-28T16:36:11.785161+00:00
- **Pipeline model**: `claude-haiku-4-5`
- **Judge model**: `claude-haiku-4-5` (via Ragas)
- **Prompt**: `prompts/orchestrator/v2.md` — edits create a new version file.
- **Dataset**: `eval/dataset.jsonl` — 50 questions

## Summary (mean across all questions)

| Metric | Mean score |
|---|---|
| faithfulness | 0.564 |
| answer_relevancy | 0.740 |

## Summary by category
"""


class TestParseJudged:
    def test_extracts_means_and_metadata(self, tmp_path):
        p = tmp_path / "agentic-20260828T163611Z.md"
        p.write_text(JUDGED_MD)

        rec = parse_judged_report(p)

        assert rec["kind"] == "judged"
        assert rec["variant"] == "agentic"
        assert rec["model"] == "claude-haiku-4-5"
        assert rec["prompt"] == "orchestrator/v2"
        assert rec["metrics"] == {"faithfulness": 0.564, "answer_relevancy": 0.740}

    def test_id_is_the_filename_stem(self, tmp_path):
        """Stable across re-ingestion - that's what makes append idempotent."""
        p = tmp_path / "fixed-20260828T152436Z.md"
        p.write_text(JUDGED_MD)
        assert parse_judged_report(p)["id"] == "fixed-20260828T152436Z"

    def test_non_report_markdown_ignored(self, tmp_path):
        p = tmp_path / "notes.md"
        p.write_text("# Some other document\n")
        assert parse_judged_report(p) is None


def sweep_json(tmp_path, name="sweep-20260829T230921Z.json"):
    p = tmp_path / name
    p.write_text(json.dumps({
        "kind": "retrieval_sweep", "run_at": "2026-08-29T23:09:21+00:00",
        "embed_model": "bge-small", "reranker_model": "ms-marco", "n_questions": 50,
        "results": [
            {"mode": "semantic", "top_k": 5, "rerank": False, "rerank_top_k": None,
             "k": 5, "recall": 0.81, "precision": 0.18, "hit_rate": 0.83,
             "mrr": 0.79, "ndcg": 0.787, "abstention_precision": 0.0},
            {"mode": "hybrid", "top_k": 20, "rerank": True, "rerank_top_k": 5,
             "k": 5, "recall": 0.85, "precision": 0.19, "hit_rate": 0.86,
             "mrr": 0.83, "ndcg": 0.830, "abstention_precision": 0.5},
        ],
    }))
    return p


class TestParseSweep:
    def test_records_the_best_config_by_ndcg(self, tmp_path):
        """A sweep is many configs; the ledger row has to name which one its
        numbers describe, or the table is unreadable."""
        rec = parse_sweep(sweep_json(tmp_path))

        assert rec["kind"] == "retrieval_sweep"
        assert rec["metrics"]["ndcg"] == 0.830
        assert "hybrid" in rec["variant"] and "rerank->5" in rec["variant"]
        assert rec["n_configs"] == 2

    def test_threshold_sweep_not_parsed_as_retrieval_sweep(self, tmp_path):
        p = tmp_path / "thresholds-x.json"
        p.write_text(json.dumps({"kind": "threshold_sweep", "results": []}))
        assert parse_sweep(p) is None

    def test_threshold_sweep_parsed(self, tmp_path):
        p = tmp_path / "thresholds-20260829T222404Z.json"
        p.write_text(json.dumps({
            "kind": "threshold_sweep", "run_at": "2026-08-29T22:24:04+00:00",
            "embed_model": "bge-small", "n_questions": 50,
            "results": [{"mode": "semantic", "min_vector_score": 0.65, "recall": 0.72,
                         "ndcg": 0.72, "abstention_precision": 0.5}],
        }))
        rec = parse_thresholds(p)
        assert rec["kind"] == "threshold_sweep"
        assert "0.65" in rec["variant"]


class TestRender:
    def rows(self):
        return [
            {"id": "fixed-1", "kind": "judged", "run_at": "2026-08-28T15:00:00+00:00",
             "variant": "fixed", "model": "claude-haiku-4-5", "n_questions": 50,
             "metrics": {"faithfulness": 0.829, "context_recall": 0.760}},
            {"id": "sweep-1", "kind": "retrieval_sweep", "run_at": "2026-08-29T23:00:00+00:00",
             "variant": "hybrid top_k=20", "model": "bge-small", "n_questions": 50,
             "n_configs": 72, "metrics": {"recall": 0.85, "ndcg": 0.83}},
        ]

    def test_one_table_holds_both_run_kinds(self):
        html = render(self.rows(), None)
        assert "fixed-1" in html and "sweep-1" in html

    def test_inapplicable_metrics_are_blank_not_zero(self):
        """A retrieval run didn't score 0.0 on faithfulness - it has no judge.
        Rendering a zero would read as a catastrophic result."""
        html = render(self.rows(), None)
        assert 'class="na">—' in html

    def test_survives_a_missing_sweep_json(self):
        """The JSON is gitignored scratch, so the page has to degrade to the
        ledger table rather than fail."""
        html = render(self.rows(), None)
        assert "curves are omitted" in html

    def test_pr_curve_drawn_when_sweep_present(self, tmp_path):
        sweep = json.loads(sweep_json(tmp_path).read_text())
        sweep["_source"] = "sweep-x.json"
        html = render(self.rows(), sweep)
        assert "Precision–recall" in html
        assert "--s-semantic" in html

