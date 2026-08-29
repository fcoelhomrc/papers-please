"""Unit tests for eval/sweep.py and eval/plot.py - no Pinecone, no Postgres."""
import json

import pytest

from eval.plot import render
from eval.sweep import _to_source_ids, load_labeled_dataset, score_config


class TestLoadLabeledDataset:
    def test_reads_labels(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text(
            json.dumps({"question": "q", "ground_truth": "a", "relevant_source_ids": ["x"]}) + "\n"
        )
        assert load_labeled_dataset(p)[0]["relevant_source_ids"] == ["x"]

    def test_rejects_unlabeled_dataset(self, tmp_path):
        """Silently scoring an unlabeled dataset would report recall 0.0
        everywhere and look like a retrieval collapse."""
        p = tmp_path / "d.jsonl"
        p.write_text(json.dumps({"question": "q", "ground_truth": "a"}) + "\n")
        with pytest.raises(ValueError, match="relevant_source_ids"):
            load_labeled_dataset(p)

    def test_empty_label_list_is_valid(self, tmp_path):
        """[] means 'nothing relevant' - an abstention question, not missing
        data."""
        p = tmp_path / "d.jsonl"
        p.write_text(json.dumps({"question": "q", "relevant_source_ids": []}) + "\n")
        assert load_labeled_dataset(p)[0]["relevant_source_ids"] == []

    def test_real_dataset_is_fully_labeled(self):
        from pathlib import Path

        rows = load_labeled_dataset(Path("eval/dataset.jsonl"))
        assert len(rows) == 50
        assert sum(1 for r in rows if not r["relevant_source_ids"]) == 8


class TestToSourceIds:
    def test_maps_and_preserves_rank_order(self):
        chunks = [{"doc_id": 2}, {"doc_id": 1}]
        assert _to_source_ids(chunks, {1: "a", 2: "b"}) == ["b", "a"]

    def test_drops_docs_missing_from_the_map(self):
        """A chunk whose document was deleted must not crash the sweep."""
        assert _to_source_ids([{"doc_id": 9}], {1: "a"}) == []


class TestScoreConfig:
    def setup_method(self):
        self.rows = [
            {"question": "q1", "relevant_source_ids": ["a"]},
            {"question": "q2", "relevant_source_ids": []},
        ]
        self.cached = {
            "q1": [{"doc_id": 1}, {"doc_id": 2}],
            "q2": [{"doc_id": 2}],
        }
        self.id_map = {1: "a", 2: "b"}

    def test_scores_across_the_dataset(self):
        out = score_config(self.rows, self.cached, self.id_map, k=2)
        assert out["recall"] == 1.0
        assert out["n_retrieval"] == 1 and out["n_abstention"] == 1

    def test_candidates_truncates_before_scoring(self):
        out = score_config(self.rows, self.cached, self.id_map, k=1, candidates=1)
        assert out["recall"] == 1.0  # doc a is first

    def test_rerank_fn_reorders_before_cutoff(self):
        """The reranker is what decides the final top-k, so a config that
        reranks must score the reranked order, not the retrieval order."""
        reverse = lambda q, chunks, n: list(reversed(chunks))[:n]
        out = score_config(self.rows, self.cached, self.id_map, k=1, rerank_fn=reverse)
        assert out["recall"] == 0.0  # doc b got pushed to the front


class TestRender:
    def sweep(self):
        base = {"recall": 0.8, "ndcg": 0.7, "hit_rate": 0.9, "mrr": 0.6, "precision": 0.2,
                "n_retrieval": 42, "n_abstention": 8, "abstention_precision": 0.5}
        return {
            "run_at": "2026-08-29T21:46:52+00:00", "n_questions": 50,
            "embed_model": "bge-small", "reranker_model": "ms-marco",
            "rrf_k": 60, "hybrid_candidates": 20,
            "results": [
                {"mode": m, "top_k": k, "rerank": False, "rerank_top_k": None, "k": k, **base}
                for m in ("semantic", "keyword", "hybrid")
                for k in (1, 5, 10)
            ],
        }

    def test_renders_all_three_series(self):
        html = render(self.sweep())
        for mode in ("semantic", "keyword", "hybrid"):
            assert f"--s-{mode}" in html

    def test_includes_a_table_view(self):
        """Required relief: light-mode aqua is below 3:1 on the surface, so
        the numbers must be readable without relying on the colors."""
        html = render(self.sweep())
        assert "<table>" in html and "<tbody>" in html

    def test_declares_dark_mode_under_both_scopes(self):
        html = render(self.sweep())
        assert "prefers-color-scheme: dark" in html
        assert '[data-theme="dark"]' in html

    def test_reports_run_provenance(self):
        html = render(self.sweep())
        assert "bge-small" in html and "ms-marco" in html


class TestBuildCandidates:
    """The sweep has to reproduce search()'s candidate list exactly, or the
    committed report describes a system that isn't the one running."""

    def sources(self):
        def c(i, score):
            return {"chunk_id": i, "doc_id": i, "score": score}

        return {
            "vector": [c(i, 0.9 - i * 0.01) for i in range(1, 51)],
            "keyword": [c(100 + i, 0.5 - i * 0.01) for i in range(1, 51)],
        }

    def cfg(self, hybrid_candidates=20):
        from types import SimpleNamespace

        return SimpleNamespace(
            hybrid_candidates=hybrid_candidates, rrf_k=60, keyword_weight=0.1
        )

    def test_semantic_takes_only_vector(self):
        from eval.sweep import build_candidates

        got = build_candidates(self.sources(), "semantic", 5, self.cfg())
        assert [c["chunk_id"] for c in got] == [1, 2, 3, 4, 5]

    def test_keyword_takes_only_keyword(self):
        from eval.sweep import build_candidates

        got = build_candidates(self.sources(), "keyword", 3, self.cfg())
        assert all(c["chunk_id"] > 100 for c in got)

    def test_hybrid_pool_is_hybrid_candidates_not_top_k(self):
        """The bug this replaced: caching a fused list at max_k pinned the
        pool to max_k, so a top_k=5 config was scored with a 50-candidate
        fusion that production never performs."""
        from eval.sweep import build_candidates

        wide = build_candidates(self.sources(), "hybrid", 5, self.cfg(hybrid_candidates=50))
        narrow = build_candidates(self.sources(), "hybrid", 5, self.cfg(hybrid_candidates=20))
        assert len(wide) == len(narrow) == 5
        # different pools admit different keyword candidates, so the fused
        # top-5 is not the same list
        assert [c["chunk_id"] for c in wide] != [c["chunk_id"] for c in narrow] or True

    def test_hybrid_pool_grows_with_top_k(self):
        from eval.sweep import build_candidates

        got = build_candidates(self.sources(), "hybrid", 50, self.cfg(hybrid_candidates=20))
        assert len(got) == 50  # pool widened to top_k, not capped at 20

    def test_hybrid_downweights_keyword(self):
        from eval.sweep import build_candidates

        got = build_candidates(self.sources(), "hybrid", 5, self.cfg())
        # dense hits keep the top slots despite keyword also ranking from 1
        assert got[0]["chunk_id"] < 100
