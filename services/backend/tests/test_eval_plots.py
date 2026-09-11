"""Guards on which run the single-run judged figures describe.

`judged-metrics` and `judge-vs-labels` carry no arm label in the plot, so the
run they read has to be chosen, not inherited from directory mtimes. It used
to be "newest file", which was the baseline only for as long as `none` was the
only arm on disk.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import plots  # noqa: E402


def _run(tmp: Path, arm: str, marker: float, *, n=100):
    """A judged result file thin enough to identify by one number."""
    blob = {
        "kind": "judged",
        "n_questions": n,
        "retrieval_config": {"arm": arm},
        "retrieval_metrics": {"recall": marker, "recall_ci": 0.0,
                              "map": marker, "map_ci": 0.0},
        "per_question": [{"faithfulness": marker} for _ in range(n)],
    }
    p = tmp / f"judged-{arm}-2026.json"
    p.write_text(json.dumps(blob))
    return p


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "results", lambda: tmp_path)
    return tmp_path


def test_prefers_the_none_arm_over_a_newer_arm(results_dir):
    """The sweep ends on hyde; the baseline figure must still show none."""
    import os

    none = _run(results_dir, "none", 0.11)
    hyde = _run(results_dir, "hyde", 0.99)
    # hyde lands after none, as it did in the real sweep.
    os.utime(none, (1, 1))
    os.utime(hyde, (2, 2))

    assert plots.judged_results()["retrieval_metrics"]["recall"] == 0.11


def test_falls_back_to_newest_when_there_is_no_baseline(results_dir):
    import os

    a = _run(results_dir, "multi_query", 0.22)
    b = _run(results_dir, "hyde", 0.33)
    os.utime(a, (1, 1))
    os.utime(b, (2, 2))

    assert plots.judged_results()["retrieval_metrics"]["recall"] == 0.33


def test_no_results_is_none_not_a_crash(results_dir):
    assert plots.judged_results() is None


def test_sample_runs_are_not_treated_as_baselines(results_dir):
    """A --sample smoke has a different n and is not comparable."""
    _run(results_dir, "none", 0.44, n=5)
    _run(results_dir, "multi_query", 0.55)

    by_arm = plots.judged_by_arm()
    assert "none" not in by_arm
    assert set(by_arm) == {"multi_query"}


def test_judged_by_arm_keeps_the_newest_run_of_each_arm(results_dir):
    import os

    old = results_dir / "judged-none-old.json"
    old.write_text(json.dumps({
        "n_questions": 100, "retrieval_config": {"arm": "none"},
        "retrieval_metrics": {"recall": 0.1}, "per_question": [],
    }))
    new = results_dir / "judged-none-new.json"
    new.write_text(json.dumps({
        "n_questions": 100, "retrieval_config": {"arm": "none"},
        "retrieval_metrics": {"recall": 0.9}, "per_question": [],
    }))
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    assert plots.judged_by_arm()["none"]["retrieval_metrics"]["recall"] == 0.9


class TestAblationOptional:
    """A target can be judged without being ablated.

    BM25 is swept as a retrieval mode, so it has no encoder to ablate and no
    `ablation-all-*.json` - but every query arm is judged against it. That used
    to raise `max() arg is an empty sequence` and produce no figures at all.
    """

    def test_missing_ablation_returns_empty_not_a_crash(self, results_dir):
        assert plots.load_results(None) == {}

    def test_an_ablation_is_still_read_when_present(self, results_dir):
        (results_dir / "ablation-all-2026.json").write_text(
            json.dumps({"a": [{"k": 1}], "timings": []})
        )
        data = plots.load_results(None)
        assert data["a"] == [{"k": 1}]

    def test_a_targeted_rerun_merges_over_the_base(self, results_dir):
        import os

        base = results_dir / "ablation-all-2026.json"
        base.write_text(json.dumps({"a": ["old"], "c": ["old"]}))
        part = results_dir / "ablation-c-2027.json"
        part.write_text(json.dumps({"c": ["new"]}))
        os.utime(base, (1, 1))
        os.utime(part, (2, 2))

        data = plots.load_results(None)
        assert data["c"] == ["new"]
        assert data["a"] == ["old"]


class TestRetrieverFacet:
    """The query-arms facet must name the retriever it actually measured."""

    def _runs(self, *modes):
        return {f"arm{i}": {"retrieval_config": {"mode": m}, "per_question": []}
                for i, m in enumerate(modes)}

    def test_bm25_runs_are_not_labelled_dense(self):
        assert plots.retriever_facet(self._runs("bm25")) == "Retriever = BM25"

    def test_semantic_runs_are_labelled_dense(self):
        assert plots.retriever_facet(self._runs("semantic")) == "Retriever = Dense"

    def test_every_arm_sharing_one_mode_still_reads_that_mode(self):
        assert plots.retriever_facet(self._runs(*["bm25"] * 5)) == "Retriever = BM25"

    def test_mixed_modes_claim_nothing(self):
        """Better an unlabelled facet than a confidently wrong one."""
        assert plots.retriever_facet(self._runs("bm25", "semantic")) == ""

    def test_a_missing_mode_claims_nothing(self):
        assert plots.retriever_facet({"none": {"per_question": []}}) == ""
